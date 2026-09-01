#!/usr/bin/env python3
"""Render a bounded private review gallery from a deterministic CSV manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import yaml

from seamless_curation.review_renderer import (
    RenderSettings,
    link_source_media,
    load_joint_provider,
    normalize_review_config,
    read_manifest,
    render_record,
    write_gallery,
)


def _atomic_private_json(path: Path, payload: Any) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    os.chmod(temporary, 0o600)
    os.replace(temporary, path)
    os.chmod(path, 0o600)


def _item_result_path(results_dir: Path, review_item_id: str) -> Path:
    digest = hashlib.sha256(review_item_id.encode("utf-8")).hexdigest()[:20]
    return results_dir / f"{digest}.json"


def _load_item_results(records: list[dict[str, str]], results_dir: Path) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for record in records:
        path = _item_result_path(results_dir, record["review_item_id"])
        if path.exists():
            results.append(json.loads(path.read_text(encoding="utf-8")))
        else:
            results.append({**record, "status": "pending", "media_file": None})
    return results


def main() -> None:
    os.umask(0o077)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/review_gallery_dryrun.yaml"))
    parser.add_argument("--manifest", type=Path, help="override config manifest")
    parser.add_argument("--index", type=int, help="render exactly one zero-based manifest row")
    parser.add_argument("--limit", type=int, help="render at most this many rows")
    parser.add_argument("--html-only", action="store_true", help="reuse result manifest and rebuild HTML")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--strict", action="store_true", help="stop on the first rendering error")
    parser.add_argument(
        "--gallery-name",
        default="index",
        help="basename for the HTML and result manifest, so a Pass-1 subset gallery "
        "can reuse the already-rendered clips without clobbering the full index",
    )
    parser.add_argument(
        "--rubric",
        type=Path,
        help="approved Pass-2 rubric YAML; omit for the Pass-1 free-text gallery",
    )
    parser.add_argument("--title", help="override the gallery heading")
    args = parser.parse_args()
    if "/" in args.gallery_name or args.gallery_name.startswith("."):
        raise ValueError("--gallery-name must be a plain basename")

    config = normalize_review_config(yaml.safe_load(args.config.read_text(encoding="utf-8")))
    source_root = Path(config["source_root"]).resolve()
    output_root = Path(config["private_output_root"])
    output_root.mkdir(parents=True, exist_ok=True, mode=0o700)
    os.chmod(output_root, 0o700)
    results_dir = output_root / "item_results"
    results_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
    os.chmod(results_dir, 0o700)
    manifest_path = args.manifest or Path(config["manifest"])
    all_records = read_manifest(manifest_path)
    records = all_records
    if args.index is not None:
        if not 0 <= args.index < len(records):
            raise IndexError(f"--index {args.index} outside manifest with {len(records)} rows")
        records = [records[args.index]]
    elif args.limit is not None:
        records = records[: args.limit]

    result_path = output_root / f"{args.gallery_name}_render_results.json"
    if args.html_only:
        if args.index is not None or args.limit is not None:
            raise ValueError("--html-only aggregates the full manifest; do not combine it with --index/--limit")
        results = _load_item_results(all_records, results_dir)
    else:
        settings = RenderSettings(**dict(config.get("render") or {}))
        provider = load_joint_provider(config.get("joint_provider"))
        results: list[dict[str, Any]] = []
        for record in records:
            try:
                results.append(
                    render_record(
                        record,
                        source_root=source_root,
                        output_root=output_root,
                        settings=settings,
                        provider=provider,
                        overwrite=args.overwrite,
                    )
                )
            except Exception as exc:
                failure = {**record, "status": "error", "media_file": None, "error": f"{type(exc).__name__}: {exc}"}
                results.append(failure)
            _atomic_private_json(
                _item_result_path(results_dir, record["review_item_id"]),
                results[-1],
            )
            if args.strict and results[-1]["status"] == "error":
                raise RuntimeError(str(results[-1]["error"]))

    # Array tasks only write their unique clip, sidecar, and item result. A
    # single post-array --html-only call performs the shared aggregation.
    if args.index is not None:
        print(json.dumps({"output_root": str(output_root), "counts": {results[0]["status"]: 1}}, sort_keys=True))
        return

    _atomic_private_json(result_path, results)

    # Link the full source recordings under the gallery root before the HTML is
    # written, so the page only ever offers links that resolve.
    linked = 0
    if bool(config.get("link_source_media", True)):
        linked = link_source_media(output_root, results, source_root)

    rubric = yaml.safe_load(args.rubric.read_text(encoding="utf-8")) if args.rubric else None
    gallery_path = output_root / f"{args.gallery_name}.html"
    title = args.title or str(config.get("gallery_title", "Seamless Interaction review"))
    write_gallery(gallery_path, results, title, rubric)
    counts: dict[str, int] = {}
    for result in results:
        status = str(result.get("status", "unknown"))
        counts[status] = counts.get(status, 0) + 1
    counts["source_media_linked"] = linked
    print(json.dumps({"output_root": str(output_root), "gallery": str(gallery_path), "counts": counts}, sort_keys=True))


if __name__ == "__main__":
    main()

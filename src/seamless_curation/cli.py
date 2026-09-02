"""``seamless-curation <stage>`` — the whole pipeline behind one entry point.

Stages, in order. Each reads only the outputs of the ones before it, so any of
them can be re-run alone.

===============  =========================================================
``population``   eligible participant files, from the M-1 inventory
``scan``         window-resolution gesture measurement (Slurm array)
``select``       apply gates, choose candidate review clips
``render``       review card + review clip per candidate (Slurm array)
``review``       serve the review app and persist verdicts
``queue``        list the review queue for a reviewer that is not the app
``manifest``     fold verdicts into the accepted manifests
``verify``       read sampled manifest rows back out of the source tree
``stats``        census and funnel tables for the report
===============  =========================================================
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Sequence

import pandas as pd

from .config import DEFAULT_CONFIG, RunConfig, load_config


def _provenance() -> dict[str, Any]:
    """Git identity of the code that produced an artefact, best effort."""

    def run(*args: str) -> str:
        try:
            return subprocess.run(
                ["git", *args], capture_output=True, text=True, timeout=15, check=False
            ).stdout.strip()
        except (OSError, subprocess.SubprocessError):
            return ""

    return {
        "git_sha": run("rev-parse", "HEAD"),
        "git_dirty": bool(run("status", "--porcelain")),
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str), encoding="utf-8")
    os.replace(temporary, path)


# ---------------------------------------------------------------- population
def cmd_population(config: RunConfig, args: argparse.Namespace) -> int:
    from .corpus import build_population, eligibility_counts

    inventory = pd.read_parquet(config.inventory)
    population = build_population(inventory)
    population = population.loc[population["vendor"].isin(config.vendors)].reset_index(drop=True)
    config.output_root.mkdir(parents=True, exist_ok=True)
    population.to_parquet(config.population_path, index=False)

    counts = eligibility_counts(population)
    hours = (
        population.loc[population["eligible"], "observed_duration_s"].sum() / 3600.0
    )
    summary = {
        **counts.as_dict(),
        "eligible_participant_hours": round(float(hours), 1),
        "by_vendor": population.groupby("vendor")["eligible"].agg(["size", "sum"]).to_dict("index"),
        **_provenance(),
    }
    _write_json(config.output_root / "population_summary.json", summary)
    print(json.dumps(summary, indent=2, default=str))
    return 0


# ---------------------------------------------------------------------- scan
def cmd_scan(config: RunConfig, args: argparse.Namespace) -> int:
    from .corpus import shard_of
    from .scan import membership_hash, scan_shard, shard_is_current, write_shard

    population = pd.read_parquet(config.population_path)
    population = population.loc[population["eligible"]].reset_index(drop=True)
    if args.limit:
        population = population.head(args.limit)
    tasks = args.tasks or config.scan_tasks
    population["shard"] = shard_of(population["file_id"], tasks)
    shard = population.loc[population["shard"] == args.task_index].reset_index(drop=True)

    fingerprint = config.scan.fingerprint()
    membership = membership_hash(shard["file_id"])
    if not args.overwrite and shard_is_current(
        config.shard_dir, args.task_index, fingerprint, membership
    ):
        print(f"task {args.task_index}: up to date ({len(shard)} files)")
        return 0

    started = time.perf_counter()
    files, windows = scan_shard(
        shard,
        config.source_root,
        config.scan,
        log=(lambda message: print(f"task {args.task_index}: {message}", flush=True))
        if args.verbose
        else None,
    )
    elapsed = time.perf_counter() - started
    write_shard(
        config.shard_dir,
        args.task_index,
        files,
        windows,
        {
            "status": "complete",
            "fingerprint": fingerprint,
            "task_index": args.task_index,
            "task_count": tasks,
            "membership": membership,
            "files": int(len(files)),
            "windows": int(len(windows)),
            "wall_s": round(elapsed, 1),
            "seconds_per_file": round(elapsed / max(1, len(files)), 3),
            **_provenance(),
        },
    )
    print(
        f"task {args.task_index}: {len(files)} files, {len(windows)} windows, "
        f"{elapsed:.0f}s ({elapsed / max(1, len(files)):.2f}s/file)"
    )
    return 0


def cmd_gather(config: RunConfig, args: argparse.Namespace) -> int:
    from .scan import read_shards

    tasks = args.tasks or config.scan_tasks
    files = read_shards(config.shard_dir, tasks, "files")
    windows = read_shards(config.shard_dir, tasks, "windows")
    files.to_parquet(config.files_path, index=False)
    windows.to_parquet(config.windows_path, index=False)
    status = files["scan_status"].value_counts().to_dict()
    summary = {
        "files": int(len(files)),
        "windows": int(len(windows)),
        "scan_status": {str(k): int(v) for k, v in status.items()},
        "scanned_hours": round(float(files.get("duration_s", pd.Series(dtype=float)).sum() / 3600), 1),
        **_provenance(),
    }
    _write_json(config.output_root / "scan_summary.json", summary)
    print(json.dumps(summary, indent=2))
    return 0


# -------------------------------------------------------------------- select
def cmd_select(config: RunConfig, args: argparse.Namespace) -> int:
    from .gates import apply_gates, build_review_items, gate_funnel, select_clips

    windows = pd.read_parquet(config.windows_path)
    if windows.empty:
        raise SystemExit(
            f"{config.windows_path} has no rows: the scan measured no windows at all. "
            f"Check scan.window_seconds ({config.scan.window_seconds:.0f} s) against the "
            "recordings' length, then re-run scan and gather."
        )
    gated = apply_gates(windows, config.gates)
    funnel = gate_funnel(gated)
    # The funnel is written before anything can go wrong downstream: when a gate
    # rejects everything it is the one artefact that says which clause did it.
    config.output_root.mkdir(parents=True, exist_ok=True)
    funnel.to_csv(config.output_root / "gate_funnel.csv", index=False)
    clips = select_clips(gated, max_per_file=config.max_clips_per_file, seed=config.select_seed)
    items = build_review_items(
        clips,
        max_files_per_participant=config.max_files_per_participant,
        seed=config.select_seed,
    )
    # Only clips belonging to a surviving review item are candidates.
    if items.empty:
        clips = clips.iloc[0:0].copy()
        for column in ("review_item_id", "review_rank"):
            clips[column] = pd.Series(dtype="object" if column.endswith("id") else "int64")
    else:
        clips = clips.loc[clips["file_id"].isin(set(items["file_id"]))].copy()
        clips = clips.merge(
            items[["file_id", "review_item_id", "review_rank"]], on="file_id", how="left"
        )

    clips.to_parquet(config.candidates_path, index=False)
    items.to_csv(config.review_manifest_path, index=False)
    if items.empty:
        print(funnel.to_string(index=False))
        raise SystemExit(
            "no window passed the gates; gate_funnel.csv names the clause that rejected them"
        )

    summary = {
        "windows": int(len(gated)),
        "qualifying_windows": int(gated["qualifies"].sum()),
        "candidate_clips": int(len(clips)),
        "candidate_hours": round(float(clips["window_seconds"].sum() / 3600.0), 1),
        "review_items": int(len(items)),
        "candidate_participants": int(items["participant_key"].nunique()) if len(items) else 0,
        "seconds_per_review_item": round(
            float(clips["window_seconds"].sum() / max(1, len(items))), 1
        ),
        "by_vendor": items.groupby("vendor")["clip_seconds"].agg(["size", "sum"]).to_dict("index")
        if len(items)
        else {},
        "gates": config.gates.as_dict(),
        **_provenance(),
    }
    _write_json(config.output_root / "select_summary.json", summary)
    print(funnel.to_string(index=False))
    print(json.dumps(summary, indent=2, default=str))
    return 0


# -------------------------------------------------------------------- render
def cmd_render(config: RunConfig, args: argparse.Namespace) -> int:
    from .render import render_task

    manifest = pd.read_csv(config.review_manifest_path)
    tasks = args.tasks or config.render_tasks
    if args.limit:
        manifest = manifest.head(args.limit)
    rows = manifest.iloc[args.task_index :: tasks]
    done = render_task(
        rows,
        config,
        overwrite=args.overwrite,
        card_only=args.card_only,
        log=lambda message: print(f"task {args.task_index}: {message}", flush=True),
    )
    print(f"task {args.task_index}: {done} of {len(rows)} clips ready")
    return 0


# -------------------------------------------------------------------- review
def cmd_review(config: RunConfig, args: argparse.Namespace) -> int:
    from .review_app import serve

    serve(config, host=args.host, port=args.port, open_browser=False)
    return 0


def cmd_queue(config: RunConfig, args: argparse.Namespace) -> int:
    """Print the review work queue, for a reviewer that is not the web app.

    The web app is the interactive front end; this is the same queue as a list,
    so a batch pass — a second person working offline, or a model-assisted
    pre-screen — can be pointed at exactly the items that still need a verdict
    and hand its results back through ``import-verdicts``.
    """

    from .review_app import build_queue
    from .review_store import VerdictStore

    queue = build_queue(config)
    judged = set(VerdictStore(config.verdict_log).resolve().get("review_item_id", []))
    if args.unreviewed:
        queue = [item for item in queue if item["review_item_id"] not in judged]
    if args.limit:
        queue = queue[: args.limit]
    for item in queue:
        item["card_path"] = str(config.media_root / item["card"])
        item["clip_path"] = str(config.media_root / item["clip"]) if item.get("clip") else None
    if args.format == "json":
        print(json.dumps(queue, indent=2, default=str))
    else:
        for item in queue:
            print(f"{item['review_item_id']}\t{item['file_id']}\t{item['card_path']}")
    return 0


def cmd_import_verdicts(config: RunConfig, args: argparse.Namespace) -> int:
    from .review_store import VerdictStore

    store = VerdictStore(config.verdict_log)
    added = store.import_file(Path(args.path))
    print(f"imported {added} verdicts into {config.verdict_log}")
    return 0


# ------------------------------------------------------------------ manifest
def cmd_manifest(config: RunConfig, args: argparse.Namespace) -> int:
    from .manifest import build_manifests

    summary = build_manifests(config)
    _write_json(config.output_root / "manifest_summary.json", {**summary, **_provenance()})
    print(json.dumps(summary, indent=2, default=str))
    return 0


def cmd_verify(config: RunConfig, args: argparse.Namespace) -> int:
    from .manifest import verify_manifest

    summary = verify_manifest(config, sample=args.sample)
    print(json.dumps(summary, indent=2))
    return 1 if summary["failures"] else 0


# --------------------------------------------------------------------- stats
def cmd_stats(config: RunConfig, args: argparse.Namespace) -> int:
    from .report import write_report

    path = write_report(config, Path(args.out) if args.out else None)
    print(f"wrote {path}")
    return 0


STAGES = {
    "population": cmd_population,
    "scan": cmd_scan,
    "gather": cmd_gather,
    "select": cmd_select,
    "render": cmd_render,
    "review": cmd_review,
    "queue": cmd_queue,
    "import-verdicts": cmd_import_verdicts,
    "manifest": cmd_manifest,
    "verify": cmd_verify,
    "stats": cmd_stats,
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="seamless-curation", description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    sub = parser.add_subparsers(dest="stage", required=True)

    sub.add_parser("population", help="build the eligible population table")

    scan = sub.add_parser("scan", help="measure one shard of the population")
    scan.add_argument("--task-index", type=int, default=0)
    scan.add_argument("--tasks", type=int)
    scan.add_argument("--limit", type=int, help="only the first N eligible files (smoke test)")
    scan.add_argument("--overwrite", action="store_true")
    scan.add_argument("--verbose", action="store_true")

    gather = sub.add_parser("gather", help="concatenate scan shards, refusing gaps")
    gather.add_argument("--tasks", type=int)

    sub.add_parser("select", help="apply gates and choose candidate clips")

    render = sub.add_parser("render", help="render review artefacts for one shard")
    render.add_argument("--task-index", type=int, default=0)
    render.add_argument("--tasks", type=int)
    render.add_argument("--limit", type=int)
    render.add_argument("--overwrite", action="store_true")
    render.add_argument("--card-only", action="store_true", help="skip the video, cards only")

    review = sub.add_parser("review", help="serve the review app on localhost")
    review.add_argument("--host", default="127.0.0.1")
    review.add_argument("--port", type=int, default=8765)

    queue = sub.add_parser("queue", help="list the review queue for an offline reviewer")
    queue.add_argument("--unreviewed", action="store_true", help="only items with no verdict yet")
    queue.add_argument("--limit", type=int)
    queue.add_argument("--format", choices=("tsv", "json"), default="tsv")

    imp = sub.add_parser("import-verdicts", help="append verdicts from a JSON/JSONL export")
    imp.add_argument("path")

    sub.add_parser("manifest", help="build the accepted manifests from verdicts")

    verify = sub.add_parser("verify", help="read sampled manifest rows from the source tree")
    verify.add_argument("--sample", type=int, default=24)

    stats = sub.add_parser("stats", help="write the run report")
    stats.add_argument("--out")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = load_config(args.config)
    return STAGES[args.stage](config, args)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

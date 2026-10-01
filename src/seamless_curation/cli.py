"""``python -m seamless_curation --config <yaml> <stage>`` — the annotation pipeline.

Stages, in order. Each reads only the outputs of the ones before it, so any of
them can be re-run alone.

================  ==========================================================
``catalog``       every filelist recording + Meta's metadata tables (seconds)
``scan``          continuous measurements per interaction (Slurm array)
``scan-video``    optional pixel pass, one keyframe per clip (Slurm array)
``annotate``      labels, scores, links, aggregates; validate; publish
``validate``      re-validate the published tables; ``--read-back N`` re-reads
                  N sampled clips from the release
``report``        annotation_report.md: coverage, distributions, validation
``schema-docs``   regenerate docs/annotation_schema.md from the registry
================  ==========================================================

Nothing here selects, excludes or ranks data: every recording in Meta's
filelist ends up with a row, and every clip of every measured recording too.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Sequence

import pandas as pd

from .config import DEFAULT_CONFIG, RunConfig, load_config


def _provenance() -> dict[str, Any]:
    """Git identity of the code that produced an artefact, best effort."""

    def run(*args: str) -> str:
        try:
            return subprocess.run(["git", *args], capture_output=True, text=True, timeout=15,
                                  check=False).stdout.strip()
        except (OSError, subprocess.SubprocessError):
            return ""

    return {"git_sha": run("rev-parse", "HEAD"), "git_dirty": bool(run("status", "--porcelain")),
            "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}


def _code_hash(config_path: Path | None = None) -> dict[str, str]:
    """Content hashes of every pipeline source file (and the config), so a published
    run names the exact code that produced it even when the tree is uncommitted."""

    from hashlib import sha256

    here = Path(__file__).parent
    digest = sha256()
    files = sorted(here.glob("*.py"))
    extra = [Path(config_path)] if config_path else []
    extra += sorted(Path("configs/validation").glob("*.csv"))
    for path in files + extra:
        if path.exists():
            digest.update(str(path.name).encode())
            digest.update(path.read_bytes())
    return {"code_sha256": digest.hexdigest(), "code_files": ",".join(p.name for p in files)}


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str), encoding="utf-8")
    os.replace(temporary, path)


def _write_parquet(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    frame.to_parquet(temporary, index=False)
    os.replace(temporary, path)


def _ensure_root(config: RunConfig) -> None:
    config.output_root.mkdir(parents=True, exist_ok=True)
    try:
        os.chmod(config.output_root, 0o700)
    except OSError:
        pass


def _free_space_gb(path: Path) -> float:
    stat = os.statvfs(path)
    return stat.f_bavail * stat.f_frsize / 1e9


def _settings(config: RunConfig):
    from .scan import ScanSettings

    return ScanSettings(clip_seconds=config.clip_seconds, model_root=str(config.model_root),
                        gesture=config.gesture)


def _catalog_tables(config: RunConfig) -> dict[str, pd.DataFrame]:
    from .annotate import CATALOG_TABLES

    missing = [t for t in CATALOG_TABLES if not (config.catalog_dir / f"{t}.parquet").exists()]
    if missing:
        raise FileNotFoundError(f"run `catalog` first: {missing} missing in {config.catalog_dir}")
    return {t: pd.read_parquet(config.catalog_dir / f"{t}.parquet") for t in CATALOG_TABLES}


def _assign_tasks(recordings: pd.DataFrame, tasks: int) -> pd.Series:
    from .corpus import shard_of

    return shard_of(recordings["interaction_key"], tasks)


# ------------------------------------------------------------------ catalog
def cmd_catalog(config: RunConfig, args: argparse.Namespace) -> int:
    from .catalog import build_catalog

    _ensure_root(config)
    inventory = pd.read_parquet(config.inventory)
    catalog = build_catalog(inventory, config.metadata_root)
    counts = {}
    for name in ("recordings", "participants", "sessions", "prompts", "interactions"):
        frame = getattr(catalog, name)
        _write_parquet(frame, config.catalog_dir / f"{name}.parquet")
        counts[name] = int(len(frame))
    _write_json(config.catalog_dir / "catalog_summary.json", {"rows": counts, **_provenance()})
    print(json.dumps(counts, indent=2))
    return 0


# ------------------------------------------------------------------ scan
def cmd_scan(config: RunConfig, args: argparse.Namespace) -> int:
    from .scan import membership_hash, scan_members, shard_is_current, write_shard

    recordings = pd.read_parquet(config.catalog_dir / "recordings.parquet")
    args.tasks = args.tasks or config.scan_tasks
    limited = bool(args.limit or args.interactions)
    if args.limit:
        keep = recordings["interaction_key"].drop_duplicates().sort_values().head(args.limit)
        recordings = recordings.loc[recordings["interaction_key"].isin(keep)]
    if args.interactions:
        keep = set(json.loads(Path(args.interactions).read_text(encoding="utf-8")))
        recordings = recordings.loc[recordings["interaction_key"].isin(keep)]
    task = _assign_tasks(recordings, args.tasks)
    rows = recordings.loc[task == args.task_index]
    settings = _settings(config)
    fingerprint = settings.fingerprint()
    membership = membership_hash(rows)
    shard_dir = config.scan_dir if not limited else config.scan_dir.with_name("scan_shards_limited")
    if not args.force and shard_is_current(shard_dir, args.task_index, fingerprint, membership):
        print(f"task {args.task_index}: current, skipped")
        return 0
    if _free_space_gb(config.output_root) < 5.0:
        raise RuntimeError(f"less than 5 GB free under {config.output_root}")
    started = time.time()
    frames = scan_members(rows, config.source_root, settings,
                          log=lambda message: print(f"[task {args.task_index}] {message}", flush=True))
    marker = {
        "status": "complete", "task_index": args.task_index, "tasks": args.tasks,
        "fingerprint": fingerprint, "membership": membership,
        "recordings": int(len(rows)), "interactions": int(rows["interaction_key"].nunique()),
        "clips": int(len(frames["clips"])), "seconds": round(time.time() - started, 1),
        **_provenance(),
    }
    write_shard(shard_dir, args.task_index, frames, marker)
    print(json.dumps({k: marker[k] for k in ("recordings", "interactions", "clips", "seconds")}))
    return 0


def cmd_scan_video(config: RunConfig, args: argparse.Namespace) -> int:
    from .scan import marker_path, shard_path
    from .scan_video import marker as video_marker, measure_shard

    shard = marker_path(config.scan_dir, args.task_index)
    if not shard.exists():
        raise FileNotFoundError(f"scan shard {args.task_index} is not complete")
    out = config.video_dir / f"task_{args.task_index:04d}.video.parquet"
    out_marker = config.video_dir / f"task_{args.task_index:04d}.complete.json"
    expected = video_marker(shard, {"short_side": config.video_short_side,
                                    "video_source": _video_source_hash()})
    if not args.force and out.exists() and out_marker.exists():
        if json.loads(out_marker.read_text(encoding="utf-8")).get("expected") == expected:
            print(f"task {args.task_index}: current, skipped")
            return 0
    catalog = pd.read_parquet(config.catalog_dir / "recordings.parquet")
    scanned = pd.read_parquet(shard_path(config.scan_dir, args.task_index, "recordings"))
    clips = pd.read_parquet(shard_path(config.scan_dir, args.task_index, "clips"),
                            columns=["clip_id", "file_id", "clip_index", "start_s", "end_s"])
    records = catalog.merge(scanned[["file_id", "measurement_status", "quarter_turns"]], on="file_id")
    records = records.loc[records["measurement_status"] == "measured"]
    started = time.time()
    frame = measure_shard(records, clips, config.source_root, short_side=config.video_short_side,
                          log=lambda message: print(f"[video {args.task_index}] {message}", flush=True))
    _write_parquet(frame, out)
    _write_json(out_marker, {"status": "complete", "expected": expected, "clips": int(len(frame)),
                             "seconds": round(time.time() - started, 1), **_provenance()})
    return 0


def _video_source_hash() -> str:
    from hashlib import sha256

    here = Path(__file__).parent
    digest = sha256()
    for name in ("video_quality.py", "scan_video.py", "media_repair.py"):
        digest.update((here / name).read_bytes())
    return digest.hexdigest()[:16]


def _read_video(config: RunConfig, tasks: int) -> pd.DataFrame | None:
    paths = [config.video_dir / f"task_{i:04d}.video.parquet" for i in range(tasks)]
    present = [p for p in paths if p.exists()]
    if not present:
        return None
    if len(present) != len(paths):
        raise RuntimeError(f"scan-video is partial: {len(present)}/{len(paths)} shards; finish it or remove {config.video_dir}")
    return pd.concat([pd.read_parquet(p) for p in present], ignore_index=True)


# ------------------------------------------------------------------ annotate
def cmd_annotate(config: RunConfig, args: argparse.Namespace) -> int:
    from hashlib import sha256

    from . import schema
    from .annotate import ScanTables, build_tables, write_tables
    from .scan import SHARD_KINDS, membership_hash, read_shards
    from .validate import validate_tables

    catalog = _catalog_tables(config)
    recordings = catalog["recordings"]
    shard_dir = config.scan_dir
    tasks = config.scan_tasks
    subset = args.limited or args.partial
    full_assignment = _assign_tasks(recordings, tasks)
    if subset:
        # --limited: the smoke-test shards; --partial: whichever real shards are done
        # (to inspect a running scan). Neither publishes to annotations/.
        shard_dir = config.scan_dir.with_name("scan_shards_limited") if args.limited else config.scan_dir
        present = sorted(int(p.name[5:9]) for p in shard_dir.glob("task_*.complete.json"))
        tasks = (args.tasks or config.scan_tasks) if args.limited else config.scan_tasks
        keep = pd.concat([pd.read_parquet(shard_dir / f"task_{i:04d}.recordings.parquet", columns=["file_id"])
                          for i in present])["file_id"]
        recordings = recordings.loc[recordings["file_id"].isin(keep)]
        keep_inter = set(recordings["interaction_key"])
        for name in ("interactions",):
            catalog[name] = catalog[name].loc[catalog[name]["interaction_key"].isin(keep_inter)]
        catalog["recordings"] = recordings
        catalog["sessions"] = catalog["sessions"].loc[catalog["sessions"]["session_key"].isin(set(recordings["session_key"]))]
        # participants stay complete: suffix siblings and partners may fall outside the sample
    settings = _settings(config)
    fingerprint = settings.fingerprint()
    assignment = _assign_tasks(recordings, tasks)
    memberships = {int(i): membership_hash(group) for i, group in recordings.groupby(assignment)}
    if args.partial:
        full = _catalog_tables(config)["recordings"]
        memberships = {int(i): membership_hash(group) for i, group in full.groupby(full_assignment)}
        stale = [i for i in present
                 if json.loads((shard_dir / f"task_{i:04d}.complete.json").read_text()).get("fingerprint") != fingerprint
                 or json.loads((shard_dir / f"task_{i:04d}.complete.json").read_text()).get("membership") != memberships.get(i)]
        if stale:
            raise SystemExit(f"{len(stale)} completed shards are stale: {stale[:5]}")
    if subset:
        frames = {kind: pd.concat([pd.read_parquet(shard_dir / f"task_{i:04d}.{kind}.parquet") for i in present],
                                  ignore_index=True) for kind in SHARD_KINDS}
    else:
        frames = {kind: read_shards(shard_dir, tasks, kind, fingerprint=fingerprint, memberships=memberships)
                  for kind in SHARD_KINDS}
    scan = ScanTables(**{("moi" if k == "moi" else k): v for k, v in frames.items()})
    video = None if (subset or args.skip_video) else _read_video(config, tasks)
    started = time.time()
    tables, reference = build_tables(catalog, scan, rules=config.posture_rules(),
                                     clip_seconds=config.clip_seconds, video=video)
    report = validate_tables(tables, clip_seconds=config.clip_seconds,
                             catalog_file_ids=pd.Index(recordings["file_id"]))
    provenance = {**_provenance(), **_code_hash(Path(args.config)), "config_hash": config.config_hash(),
                  "scan_fingerprint": fingerprint,
                  "clip_seconds": config.clip_seconds, "run_id": config.run_id,
                  "posture_rules": asdict(config.posture_rules()),
                  "video_pass": video is not None, "annotate_seconds": round(time.time() - started, 1)}
    run_hash = sha256(json.dumps([provenance["config_hash"], fingerprint, provenance["code_sha256"],
                                  provenance["created_utc"]]).encode()).hexdigest()[:16]
    summary = {name: int(len(frame)) for name, frame in tables.items()}
    if not report.ok and not args.allow_invalid:
        _write_json(config.output_root / "validation_failed.json", report.as_dict())
        print(json.dumps(report.errors[:20], indent=2))
        raise SystemExit(f"validation failed ({len(report.errors)} errors); tables NOT published")
    target = (config.output_root / "annotations_limited" if args.limited
              else config.output_root / "annotations_partial" if args.partial else config.annotations_dir)
    write_tables(tables, target, run_hash=run_hash, provenance=provenance, side_files={
        "field_catalog.csv": schema.catalog_frame(),
        "expressivity_reference.json": reference,
        "validation_report.json": report.as_dict(),
        "annotation_summary.json": {"rows": summary, "run_hash": run_hash},
    })
    print(json.dumps({"rows": summary, "validation_ok": report.ok, "errors": len(report.errors)}, indent=2))
    return 0


# ------------------------------------------------------------------ validate / report / docs
def cmd_validate(config: RunConfig, args: argparse.Namespace) -> int:
    from .dataset import load_tables, read_back
    from .validate import validate_tables

    root = config.annotations_dir
    tables = load_tables(root)
    catalog = pd.read_parquet(config.catalog_dir / "recordings.parquet", columns=["file_id"])
    report = validate_tables(tables, clip_seconds=config.clip_seconds,
                             catalog_file_ids=pd.Index(catalog["file_id"]))
    result = report.as_dict()
    if args.read_back:
        result["read_back"] = read_back(tables, config.source_root, sample=args.read_back)
    _write_json(config.output_root / "validation_rerun.json", result)
    print(json.dumps({"ok": report.ok, "errors": report.errors[:10],
                      "read_back": result.get("read_back", {}).get("summary")}, indent=2, default=str))
    return 0 if report.ok and result.get("read_back", {}).get("summary", {}).get("failed", 0) == 0 else 1


def cmd_report(config: RunConfig, args: argparse.Namespace) -> int:
    from .report import write_report

    path = write_report(config, Path(args.root) if args.root else None)
    print(f"wrote {path}")
    return 0


def cmd_schema_docs(config: RunConfig, args: argparse.Namespace) -> int:
    from .schema_docs import render

    out = Path(args.out)
    out.write_text(render(), encoding="utf-8")
    print(f"wrote {out}")
    return 0


STAGES = {
    "catalog": cmd_catalog,
    "scan": cmd_scan,
    "scan-video": cmd_scan_video,
    "annotate": cmd_annotate,
    "validate": cmd_validate,
    "report": cmd_report,
    "schema-docs": cmd_schema_docs,
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="seamless-curation", description=__doc__.splitlines()[0])
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    sub = parser.add_subparsers(dest="stage", required=True)
    sub.add_parser("catalog", help="catalog every recording and Meta's metadata")
    scan = sub.add_parser("scan", help="measure one shard of interactions")
    scan.add_argument("--task-index", type=int, required=True)
    scan.add_argument("--tasks", type=int, default=0, help="shard count (default: config scan.tasks)")
    scan.add_argument("--limit", type=int, default=0,
                      help="smoke test: only the first N interactions, written to scan_shards_limited/")
    scan.add_argument("--interactions", default="",
                      help="smoke test: a JSON list of interaction keys, written to scan_shards_limited/")
    scan.add_argument("--force", action="store_true")
    video = sub.add_parser("scan-video", help="pixel pass over one scan shard")
    video.add_argument("--task-index", type=int, required=True)
    video.add_argument("--force", action="store_true")
    annotate = sub.add_parser("annotate", help="label, score, link, validate and publish")
    annotate.add_argument("--limited", action="store_true", help="annotate the --limit smoke-test shards")
    annotate.add_argument("--partial", action="store_true",
                          help="annotate only the completed shards of a running scan (to annotations_partial/)")
    annotate.add_argument("--tasks", type=int, default=0)
    annotate.add_argument("--skip-video", action="store_true",
                          help="publish without the pixel-pass columns (visual_status='not_run')")
    annotate.add_argument("--allow-invalid", action="store_true",
                          help="publish even if validation fails (never for a real run)")
    validate = sub.add_parser("validate", help="re-validate the published tables")
    validate.add_argument("--read-back", type=int, default=0, help="also re-read N sampled clips from the release")
    report = sub.add_parser("report", help="write annotation_report.md")
    report.add_argument("--root", default="", help="tables directory (default: the run's annotations/)")
    docs = sub.add_parser("schema-docs", help="regenerate the field reference")
    docs.add_argument("--out", default="docs/annotation_schema.md")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = load_config(args.config)
    return STAGES[args.stage](config, args)


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())

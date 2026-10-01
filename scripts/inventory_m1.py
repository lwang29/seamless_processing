#!/usr/bin/env python3
"""Restartable, header-only corpus inventory for Session 2 M-1.

The scanner derives expected paths from ``filelist.csv``, stats each sibling,
and invokes ffprobe on the MP4 container only. It never opens JSON, NPZ, or WAV
files. A separate catalog command walks directory entries and records path/stat
metadata without opening any participant payload.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import stat
import subprocess
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Any, Iterable, Sequence

import pandas as pd
import yaml


FILE_ID_RE = re.compile(
    r"^V(?P<vendor_id>\d{2})_S(?P<session_id>\d{4})_"
    r"I(?P<interaction_id>\d{8})_P(?P<participant_id>[A-Za-z0-9]+)$"
)
EXPECTED_FILELIST_COLUMNS = {
    "file_id",
    "label",
    "split",
    "batch_idx",
    "archive_idx",
    "has_imitator_movement",
    "has_annotation_1p",
    "has_annotation_3p",
}
MODALITY_EXTENSIONS = (".json", ".mp4", ".npz", ".wav")
SAFE_CONFIG_PATH = re.compile(r"^[A-Za-z0-9_./-]+$")


@dataclass(frozen=True)
class InventoryConfig:
    config_path: Path
    worktree: Path
    source_root: Path
    metadata_root: Path
    output_root: Path
    header_shards: Path
    completion_markers: Path
    catalog_path: Path
    benchmark_path: Path
    summary_dir: Path
    task_manifest: Path
    task_count: int
    ffprobe_bin: str
    probe_timeout_s: int
    workers_per_task: int
    benchmark_seed: int
    benchmark_samples: int
    benchmark_workers: tuple[int, ...]
    require_source_not_effectively_writable: bool
    require_clean_git_for_jobs: bool
    config_hash: str


@dataclass(frozen=True)
class Provenance:
    git_sha: str
    git_dirty: bool


#: Where provenance is read from. The census was first run when the platform
#: mounted ``.git`` read-only and history lived in ``.git-session``; that orphaned
#: directory is now stale (hundreds of paths read as dirty against it), so the
#: repository's own ``.git`` is preferred and ``.git-session`` is the fallback.
GIT_DIR_NAMES = (".git", ".git-session")


def _find_worktree(start: Path) -> Path:
    for candidate in (start, *start.parents):
        if any((candidate / name).is_dir() for name in GIT_DIR_NAMES):
            return candidate.resolve()
    raise ValueError(f"could not find a git directory ({', '.join(GIT_DIR_NAMES)}) above {start}")


def _git_dir(worktree: Path) -> Path:
    for name in GIT_DIR_NAMES:
        if (worktree / name).is_dir():
            return worktree / name
    raise ValueError(f"no git directory in {worktree}")


def _resolve_repo_path(worktree: Path, value: Any, field: str) -> Path:
    raw = str(value)
    if not SAFE_CONFIG_PATH.fullmatch(raw):
        raise ValueError(f"unsafe characters in {field}: {raw!r}")
    path = (worktree / raw).resolve()
    return path


def _require_under(path: Path, parent: Path, field: str) -> None:
    if path != parent and parent not in path.parents:
        raise ValueError(f"{field} must resolve beneath {parent}, got {path}")


def load_config(path: str | Path) -> InventoryConfig:
    config_path = Path(path).expanduser().resolve()
    raw_bytes = config_path.read_bytes()
    raw = yaml.safe_load(raw_bytes) or {}
    worktree = _find_worktree(config_path.parent)
    source_root = _resolve_repo_path(worktree, raw["source_root"], "source_root")
    metadata_root = _resolve_repo_path(worktree, raw["metadata_root"], "metadata_root")
    output_root = _resolve_repo_path(worktree, raw["outputs"]["root"], "outputs.root")
    _require_under(output_root, worktree, "outputs.root")
    if output_root == source_root or source_root in output_root.parents:
        raise ValueError("output root must not be the source root or beneath it")

    def output_path(key: str) -> Path:
        value = _resolve_repo_path(output_root, raw["outputs"][key], f"outputs.{key}")
        _require_under(value, output_root, f"outputs.{key}")
        return value

    task_manifest = _resolve_repo_path(worktree, raw["task_manifest"], "task_manifest")
    _require_under(task_manifest, worktree, "task_manifest")
    task_count = int(raw["task_count"])
    if not 1 <= task_count <= 1001:
        raise ValueError("task_count must be in [1, 1001] (Slurm MaxArraySize=1001)")
    probe = raw["probe"]
    workers = int(probe["workers_per_task"])
    if not 1 <= workers <= 16:
        raise ValueError("probe.workers_per_task must be in [1, 16]")
    timeout = int(probe["timeout_s"])
    if timeout <= 0:
        raise ValueError("probe.timeout_s must be positive")
    benchmark = raw["benchmark"]
    benchmark_workers = tuple(int(value) for value in benchmark["worker_counts"])
    if benchmark_workers != (1, 4, 8, 16):
        raise ValueError("benchmark.worker_counts must be exactly [1, 4, 8, 16]")
    safety = raw["safety"]
    configured_extensions = tuple(safety["payload_extensions"])
    if configured_extensions != MODALITY_EXTENSIONS:
        raise ValueError(f"payload_extensions must be {MODALITY_EXTENSIONS}")
    return InventoryConfig(
        config_path=config_path,
        worktree=worktree,
        source_root=source_root,
        metadata_root=metadata_root,
        output_root=output_root,
        header_shards=output_path("header_shards"),
        completion_markers=output_path("completion_markers"),
        catalog_path=output_path("catalog"),
        benchmark_path=output_path("benchmark"),
        summary_dir=output_path("summary"),
        task_manifest=task_manifest,
        task_count=task_count,
        ffprobe_bin=str(probe["ffprobe_bin"]),
        probe_timeout_s=timeout,
        workers_per_task=workers,
        benchmark_seed=int(benchmark["seed"]),
        benchmark_samples=int(benchmark["samples_per_worker_setting"]),
        benchmark_workers=benchmark_workers,
        require_source_not_effectively_writable=bool(
            safety["require_source_not_effectively_writable"]
        ),
        require_clean_git_for_jobs=bool(safety["require_clean_git_for_jobs"]),
        config_hash=hashlib.sha256(raw_bytes).hexdigest(),
    )


def git_provenance(config: InventoryConfig) -> Provenance:
    base = [
        "git",
        f"--git-dir={_git_dir(config.worktree)}",
        f"--work-tree={config.worktree}",
    ]
    sha = subprocess.run(
        [*base, "rev-parse", "HEAD"], check=True, capture_output=True, text=True
    ).stdout.strip()
    status = subprocess.run(
        [*base, "status", "--porcelain", "--untracked-files=normal"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    return Provenance(sha, bool(status.strip()))


def source_mount(config: InventoryConfig) -> dict[str, str]:
    completed = subprocess.run(
        ["findmnt", "-T", str(config.source_root), "-n", "-o", "TARGET,SOURCE,FSTYPE,OPTIONS"],
        check=True,
        capture_output=True,
        text=True,
    )
    parts = completed.stdout.strip().split(maxsplit=3)
    if len(parts) != 4:
        raise RuntimeError(f"unexpected findmnt output: {completed.stdout!r}")
    target, source, fstype, options = parts
    return {"target": target, "source": source, "fstype": fstype, "options": options}


def source_effectively_writable(config: InventoryConfig) -> bool:
    """Check this job's effective credentials without attempting a write."""
    return os.access(config.source_root, os.W_OK, effective_ids=True)


def preflight(config: InventoryConfig, *, require_clean: bool) -> tuple[Provenance, dict[str, str]]:
    if not config.source_root.is_dir():
        raise FileNotFoundError(config.source_root)
    if not config.metadata_root.is_dir():
        raise FileNotFoundError(config.metadata_root)
    mount = source_mount(config)
    writable = source_effectively_writable(config)
    mount["effective_writable"] = str(writable).lower()
    if config.require_source_not_effectively_writable and writable:
        raise RuntimeError(
            "source root is effectively writable by this job; refusing all inventory work"
        )
    provenance = git_provenance(config)
    if require_clean and config.require_clean_git_for_jobs and provenance.git_dirty:
        raise RuntimeError("production inventory jobs require a clean git worktree")
    return provenance, mount


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.parent / f".{path.name}.{uuid.uuid4().hex}.tmp"
    try:
        with temporary.open("x", encoding="utf-8") as handle:
            json.dump(payload, handle, sort_keys=True, indent=2, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.parent / f".{path.name}.{uuid.uuid4().hex}.tmp"
    try:
        frame.to_csv(temporary, index=False)
        with temporary.open("rb") as handle:
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_parquet(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.parent / f".{path.name}.{uuid.uuid4().hex}.tmp"
    try:
        frame.to_parquet(temporary, index=False, compression="zstd")
        with temporary.open("rb") as handle:
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def load_filelist(config: InventoryConfig) -> pd.DataFrame:
    path = config.metadata_root / "filelist.csv"
    frame = pd.read_csv(
        path,
        dtype={
            "file_id": "string",
            "label": "string",
            "split": "string",
            "batch_idx": "int64",
            "archive_idx": "int64",
            "has_imitator_movement": "int8",
            "has_annotation_1p": "int8",
            "has_annotation_3p": "int8",
        },
        keep_default_na=False,
    )
    missing = EXPECTED_FILELIST_COLUMNS - set(frame.columns)
    if missing:
        raise ValueError(f"filelist.csv missing columns: {sorted(missing)}")
    if frame["file_id"].duplicated().any():
        raise ValueError("filelist.csv contains duplicate file_id values")
    parsed = frame["file_id"].str.extract(FILE_ID_RE)
    if parsed.isna().any(axis=None):
        bad = frame.loc[parsed.isna().any(axis=1), "file_id"].head().tolist()
        raise ValueError(f"unparseable file_id values: {bad}")
    for column in parsed:
        frame[column] = parsed[column].astype("string")
    frame["vendor"] = "V" + frame["vendor_id"]
    frame["source_relbase"] = frame.apply(
        lambda row: (
            Path(str(row["label"]))
            / str(row["split"])
            / f"{int(row['batch_idx']):04d}"
            / f"{int(row['archive_idx']):04d}"
            / str(row["file_id"])
        ).as_posix(),
        axis=1,
    )
    return frame


def make_task_plan(row_count: int, task_count: int) -> pd.DataFrame:
    if row_count <= 0:
        raise ValueError("row_count must be positive")
    if not 1 <= task_count <= min(1001, row_count):
        raise ValueError("task_count must not exceed row_count or MaxArraySize")
    base, remainder = divmod(row_count, task_count)
    rows: list[dict[str, int]] = []
    start = 0
    for task_index in range(task_count):
        count = base + (1 if task_index < remainder else 0)
        stop = start + count
        rows.append(
            {
                "task_index": task_index,
                "start_row": start,
                "stop_row_exclusive": stop,
                "row_count": count,
            }
        )
        start = stop
    assert start == row_count
    return pd.DataFrame(rows)


def command_plan(config: InventoryConfig) -> None:
    preflight(config, require_clean=False)
    frame = load_filelist(config)
    plan = make_task_plan(len(frame), config.task_count)
    _atomic_csv(config.task_manifest, plan)
    print(
        json.dumps(
            {
                "filelist_rows": len(frame),
                "tasks": len(plan),
                "min_rows_per_task": int(plan.row_count.min()),
                "max_rows_per_task": int(plan.row_count.max()),
                "task_manifest": str(config.task_manifest),
            },
            sort_keys=True,
        )
    )


def _fraction(raw: Any) -> float | None:
    if raw in (None, "", "0/0", "N/A"):
        return None
    try:
        value = float(Fraction(str(raw)))
    except (ValueError, ZeroDivisionError):
        return None
    return value if math.isfinite(value) else None


def _number(raw: Any, constructor: type[int] | type[float]) -> int | float | None:
    if raw in (None, "", "N/A"):
        return None
    try:
        value = constructor(raw)
    except (TypeError, ValueError):
        return None
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _ffprobe_mp4(config: InventoryConfig, path: Path) -> dict[str, Any]:
    command = [
        config.ffprobe_bin,
        "-v",
        "error",
        "-show_entries",
        (
            "stream=index,codec_type,codec_name,width,height,r_frame_rate,avg_frame_rate,"
            "time_base,start_time,duration,nb_frames,sample_rate,channels:"
            "format=format_name,start_time,duration"
        ),
        "-of",
        "json",
        str(path),
    ]
    completed = subprocess.run(
        command,
        check=False,
        capture_output=True,
        text=True,
        timeout=config.probe_timeout_s,
        env={**os.environ, "LC_ALL": "C"},
    )
    if completed.returncode != 0:
        return {
            "probe_status": "ffprobe_error",
            "probe_returncode": completed.returncode,
            "probe_error": completed.stderr.strip()[:1000],
        }
    try:
        payload = json.loads(completed.stdout)
    except json.JSONDecodeError as error:
        return {
            "probe_status": "json_decode_error",
            "probe_returncode": completed.returncode,
            "probe_error": str(error),
        }
    streams = payload.get("streams", [])
    video = next((item for item in streams if item.get("codec_type") == "video"), {})
    audio = next((item for item in streams if item.get("codec_type") == "audio"), {})
    fmt = payload.get("format", {})
    return {
        "probe_status": "ok" if video else "no_video_stream",
        "probe_returncode": completed.returncode,
        "probe_error": "",
        "format_name": fmt.get("format_name"),
        "format_start_s": _number(fmt.get("start_time"), float),
        "format_duration_s": _number(fmt.get("duration"), float),
        "stream_count": len(streams),
        "video_stream_present": bool(video),
        "video_codec": video.get("codec_name"),
        "video_width": _number(video.get("width"), int),
        "video_height": _number(video.get("height"), int),
        "video_r_frame_rate": video.get("r_frame_rate"),
        "video_r_fps": _fraction(video.get("r_frame_rate")),
        "video_avg_frame_rate": video.get("avg_frame_rate"),
        "video_avg_fps": _fraction(video.get("avg_frame_rate")),
        "video_time_base": video.get("time_base"),
        "video_start_s": _number(video.get("start_time"), float),
        "video_duration_s": _number(video.get("duration"), float),
        "video_nb_frames": _number(video.get("nb_frames"), int),
        "embedded_audio_present": bool(audio),
        "embedded_audio_codec": audio.get("codec_name"),
        "embedded_audio_sample_rate": _number(audio.get("sample_rate"), int),
        "embedded_audio_channels": _number(audio.get("channels"), int),
        "embedded_audio_start_s": _number(audio.get("start_time"), float),
        "embedded_audio_duration_s": _number(audio.get("duration"), float),
    }


def _stat(path: Path) -> tuple[bool, int | None]:
    try:
        result = path.stat()
    except FileNotFoundError:
        return False, None
    if not stat.S_ISREG(result.st_mode):
        return False, None
    return True, result.st_size


def probe_metadata_row(
    config: InventoryConfig,
    row: dict[str, Any],
    provenance: Provenance,
) -> dict[str, Any]:
    base = config.source_root / str(row["source_relbase"])
    result = dict(row)
    result["source_relbase"] = str(row["source_relbase"])
    all_present = True
    for extension in MODALITY_EXTENSIONS:
        present, size = _stat(base.with_suffix(extension))
        key = extension.removeprefix(".")
        result[f"{key}_present"] = present
        result[f"{key}_size_bytes"] = size
        all_present = all_present and present
    result["all_modalities_present"] = all_present
    mp4_path = base.with_suffix(".mp4")
    if result["mp4_present"]:
        started = time.monotonic()
        try:
            result.update(_ffprobe_mp4(config, mp4_path))
        except subprocess.TimeoutExpired as error:
            result.update(
                {
                    "probe_status": "timeout",
                    "probe_returncode": None,
                    "probe_error": str(error)[:1000],
                }
            )
        result["probe_wall_s"] = time.monotonic() - started
    else:
        result.update(
            {
                "probe_status": "missing_mp4",
                "probe_returncode": None,
                "probe_error": "",
                "probe_wall_s": None,
            }
        )
    result["config_hash"] = config.config_hash
    result["git_sha"] = provenance.git_sha
    result["git_dirty"] = provenance.git_dirty
    return result


def _task_paths(config: InventoryConfig, task_index: int) -> tuple[Path, Path]:
    name = f"task_{task_index:04d}"
    return config.header_shards / f"{name}.parquet", config.completion_markers / f"{name}.json"


def _marker_matches(
    marker_path: Path,
    output_path: Path,
    config: InventoryConfig,
    provenance: Provenance,
    start: int,
    stop: int,
) -> bool:
    if not marker_path.is_file() or not output_path.is_file():
        return False
    try:
        marker = json.loads(marker_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return (
        marker.get("status") == "complete"
        and marker.get("config_hash") == config.config_hash
        and marker.get("git_sha") == provenance.git_sha
        and marker.get("git_dirty") is False
        and provenance.git_dirty is False
        and marker.get("start_row") == start
        and marker.get("stop_row_exclusive") == stop
    )


def command_headers(config: InventoryConfig, task_index: int, workers: int | None) -> None:
    provenance, mount = preflight(config, require_clean=True)
    plan = pd.read_csv(config.task_manifest)
    selected = plan.loc[plan.task_index.eq(task_index)]
    if len(selected) != 1:
        raise ValueError(f"task_index {task_index} not found exactly once")
    task = selected.iloc[0]
    start = int(task.start_row)
    stop = int(task.stop_row_exclusive)
    output_path, marker_path = _task_paths(config, task_index)
    if _marker_matches(marker_path, output_path, config, provenance, start, stop):
        print(json.dumps({"task_index": task_index, "status": "skipped_complete"}))
        return
    filelist = load_filelist(config).iloc[start:stop].copy()
    records = filelist.to_dict(orient="records")
    worker_count = config.workers_per_task if workers is None else workers
    if not 1 <= worker_count <= 16:
        raise ValueError("workers must be in [1, 16]")
    started = time.monotonic()
    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        observations = list(
            executor.map(lambda row: probe_metadata_row(config, row, provenance), records)
        )
    elapsed = time.monotonic() - started
    result = pd.DataFrame(observations)
    _atomic_parquet(output_path, result)
    status_counts = result.probe_status.value_counts(dropna=False).to_dict()
    _atomic_json(
        marker_path,
        {
            "status": "complete",
            "task_index": task_index,
            "start_row": start,
            "stop_row_exclusive": stop,
            "row_count": len(result),
            "workers": worker_count,
            "wall_s": elapsed,
            "files_per_s": len(result) / elapsed,
            "probe_status_counts": {str(key): int(value) for key, value in status_counts.items()},
            "config_hash": config.config_hash,
            "git_sha": provenance.git_sha,
            "git_dirty": provenance.git_dirty,
            "source_mount": mount,
        },
    )
    print(
        json.dumps(
            {
                "task_index": task_index,
                "status": "complete",
                "rows": len(result),
                "wall_s": elapsed,
                "files_per_s": len(result) / elapsed,
                "output": str(output_path),
            },
            sort_keys=True,
        )
    )


def _walk_catalog(source_root: Path) -> Iterable[dict[str, Any]]:
    root_text = os.fspath(source_root)
    for directory, directory_names, file_names in os.walk(root_text, followlinks=False):
        directory_names.sort()
        file_names.sort()
        for name in file_names:
            path = Path(directory) / name
            try:
                stat_result = path.stat()
            except FileNotFoundError:
                yield {
                    "relative_path": path.relative_to(source_root).as_posix(),
                    "extension": path.suffix.lower(),
                    "file_id": path.stem,
                    "size_bytes": None,
                    "stat_status": "vanished",
                }
                continue
            yield {
                "relative_path": path.relative_to(source_root).as_posix(),
                "extension": path.suffix.lower(),
                "file_id": path.stem,
                "size_bytes": stat_result.st_size,
                "stat_status": "ok",
            }


def command_catalog(config: InventoryConfig) -> None:
    provenance, mount = preflight(config, require_clean=True)
    marker_path = config.catalog_path.with_suffix(config.catalog_path.suffix + ".complete.json")
    if marker_path.is_file() and config.catalog_path.is_file():
        try:
            marker = json.loads(marker_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            marker = {}
        if (
            marker.get("status") == "complete"
            and marker.get("config_hash") == config.config_hash
            and marker.get("git_sha") == provenance.git_sha
            and not marker.get("git_dirty", True)
            and not provenance.git_dirty
        ):
            print(json.dumps({"status": "skipped_complete", "output": str(config.catalog_path)}))
            return
    started = time.monotonic()
    frame = pd.DataFrame(_walk_catalog(config.source_root))
    elapsed = time.monotonic() - started
    frame["config_hash"] = config.config_hash
    frame["git_sha"] = provenance.git_sha
    frame["git_dirty"] = provenance.git_dirty
    _atomic_parquet(config.catalog_path, frame)
    _atomic_json(
        marker_path,
        {
            "status": "complete",
            "file_count": len(frame),
            "total_bytes": int(frame.size_bytes.fillna(0).sum()),
            "wall_s": elapsed,
            "paths_per_s": len(frame) / elapsed,
            "config_hash": config.config_hash,
            "git_sha": provenance.git_sha,
            "git_dirty": provenance.git_dirty,
            "source_mount": mount,
        },
    )
    print(json.dumps({"status": "complete", "files": len(frame), "wall_s": elapsed}))


def _benchmark_order(frame: pd.DataFrame, seed: int) -> pd.DataFrame:
    order = frame["file_id"].map(
        lambda value: hashlib.sha256(f"{seed}:{value}".encode("utf-8")).digest()
    )
    return frame.assign(_order=order).sort_values("_order").drop(columns="_order")


def command_benchmark(
    config: InventoryConfig,
    worker_counts: Sequence[int] | None,
    samples_per_setting: int | None,
) -> None:
    provenance, mount = preflight(config, require_clean=True)
    counts = tuple(worker_counts or config.benchmark_workers)
    if counts != (1, 4, 8, 16):
        raise ValueError("worker counts must be exactly 1 4 8 16")
    sample_count = samples_per_setting or config.benchmark_samples
    if sample_count <= 0:
        raise ValueError("samples_per_setting must be positive")
    frame = _benchmark_order(load_filelist(config), config.benchmark_seed)
    required = sample_count * len(counts)
    if len(frame) < required:
        raise ValueError("not enough disjoint rows for benchmark")
    results: list[dict[str, Any]] = []
    for setting_index, workers in enumerate(counts):
        sample = frame.iloc[setting_index * sample_count : (setting_index + 1) * sample_count]
        records = sample.to_dict(orient="records")
        started = time.monotonic()
        with ThreadPoolExecutor(max_workers=workers) as executor:
            observations = list(
                executor.map(lambda row: probe_metadata_row(config, row, provenance), records)
            )
        elapsed = time.monotonic() - started
        observed = pd.DataFrame(observations)
        logical_bytes = int(observed.mp4_size_bytes.fillna(0).sum())
        status = observed.probe_status.value_counts(dropna=False).to_dict()
        result = {
            "workers": workers,
            "sample_count": sample_count,
            "sample_start": setting_index * sample_count,
            "sample_stop_exclusive": (setting_index + 1) * sample_count,
            "wall_s": elapsed,
            "files_per_s": len(observed) / elapsed,
            "logical_mp4_bytes": logical_bytes,
            "logical_mp4_mib_per_s": logical_bytes / elapsed / (1024**2),
            "probe_status_counts": {str(key): int(value) for key, value in status.items()},
        }
        results.append(result)
        print(json.dumps(result, sort_keys=True))
    _atomic_json(
        config.benchmark_path,
        {
            "measurement": "ffprobe header throughput; no payload byte scan",
            "cache_condition": "cold-ish: deterministic disjoint files per worker setting; OS/NFS caches uncontrolled",
            "logical_byte_warning": (
                "logical_mp4_mib_per_s divides represented file sizes by wall time; "
                "it is not physical bytes read by ffprobe"
            ),
            "seed": config.benchmark_seed,
            "results": results,
            "config_hash": config.config_hash,
            "git_sha": provenance.git_sha,
            "git_dirty": provenance.git_dirty,
            "source_mount": mount,
        },
    )


def _read_header_shards(config: InventoryConfig) -> pd.DataFrame:
    plan = pd.read_csv(config.task_manifest)
    paths = [_task_paths(config, int(index))[0] for index in plan.task_index]
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise RuntimeError(f"missing {len(missing)} header shards; first: {missing[:3]}")
    frames = [pd.read_parquet(path) for path in paths]
    result = pd.concat(frames, ignore_index=True)
    if len(result) != int(plan.row_count.sum()):
        raise RuntimeError("header shard row count does not equal task manifest")
    if result.file_id.duplicated().any():
        raise RuntimeError("duplicate file_id across header shards")
    return result


def _duration_column(frame: pd.DataFrame) -> pd.Series:
    # MP4 format duration is the container-level observed duration. Stream-level
    # duration is a fallback only when the format field is absent.
    return frame["format_duration_s"].fillna(frame["video_duration_s"])


def _join_metadata(config: InventoryConfig, inventory: pd.DataFrame) -> pd.DataFrame:
    interactions = pd.read_csv(
        config.metadata_root / "interactions.csv", dtype="string", keep_default_na=False
    ).rename(columns={"Unnamed: 0": "interaction_row"})
    if interactions.prompt_hash.duplicated().any():
        raise ValueError("interactions.prompt_hash is not unique")
    joined = inventory.merge(
        interactions[["prompt_hash", "interaction_type", "ipc_a", "ipc_b"]],
        left_on="interaction_id",
        right_on="prompt_hash",
        how="left",
        validate="many_to_one",
        indicator="interaction_join",
    )
    participants = pd.read_csv(
        config.metadata_root / "participants.csv", dtype="string", keep_default_na=False
    )[["vendor_id", "participant_id"]].drop_duplicates()
    participants["participant_metadata_present"] = True
    joined = joined.merge(
        participants,
        on=["vendor_id", "participant_id"],
        how="left",
        validate="many_to_one",
    )
    joined["participant_metadata_present"] = joined.participant_metadata_present.fillna(False)
    relationships = pd.read_csv(
        config.metadata_root / "relationships.csv", dtype="string", keep_default_na=False
    )
    relationships["vendor_id"] = relationships.vendor_id.str.removeprefix("V")
    relationships["session_id"] = relationships.session_id.astype(int).astype(str).str.zfill(4)
    relationships = relationships.drop_duplicates(["vendor_id", "session_id"])
    relationships["relationship_metadata_present"] = True
    joined = joined.merge(
        relationships,
        on=["vendor_id", "session_id"],
        how="left",
        validate="many_to_one",
    )
    joined["relationship_metadata_present"] = joined.relationship_metadata_present.fillna(False)
    joined["observed_duration_s"] = _duration_column(joined)
    return joined


def _dyad_duration_summary(frame: pd.DataFrame, groups: list[str]) -> pd.DataFrame:
    interaction_keys = list(
        dict.fromkeys([*groups, "vendor_id", "session_id", "interaction_id"])
    )
    per_interaction = (
        frame.groupby(interaction_keys, dropna=False)
        .agg(
            member_files=("file_id", "size"),
            members_with_duration=("observed_duration_s", "count"),
            interaction_duration_s=("observed_duration_s", "mean"),
            member_duration_min_s=("observed_duration_s", "min"),
            member_duration_max_s=("observed_duration_s", "max"),
        )
        .reset_index()
    )
    per_interaction["member_duration_spread_s"] = (
        per_interaction.member_duration_max_s - per_interaction.member_duration_min_s
    )
    summary = (
        per_interaction.groupby(groups, dropna=False)
        .agg(
            interaction_count=("interaction_id", "size"),
            interactions_with_duration=("interaction_duration_s", "count"),
            dyad_hours=("interaction_duration_s", lambda value: value.sum(min_count=1) / 3600),
            one_member_interactions=("member_files", lambda value: int((value == 1).sum())),
            two_member_interactions=("member_files", lambda value: int((value == 2).sum())),
            other_member_interactions=("member_files", lambda value: int((~value.isin([1, 2])).sum())),
            duration_spread_p95_s=("member_duration_spread_s", lambda value: value.quantile(0.95)),
            duration_spread_max_s=("member_duration_spread_s", "max"),
        )
        .reset_index()
    )
    return summary


def _write_summary_tables(config: InventoryConfig, joined: pd.DataFrame, catalog: pd.DataFrame) -> None:
    out = config.summary_dir
    out.mkdir(parents=True, exist_ok=True)
    _atomic_parquet(out / "inventory_joined.parquet", joined)

    groups = ["vendor_id", "label", "split"]
    participant_duration = (
        joined.groupby(groups, dropna=False)
        .agg(
            participant_files=("file_id", "size"),
            complete_bundles=("all_modalities_present", "sum"),
            videos_with_duration=("observed_duration_s", "count"),
            participant_stream_hours=("observed_duration_s", lambda value: value.sum(min_count=1) / 3600),
            distinct_participants=("participant_id", "nunique"),
        )
        .reset_index()
    )
    dyad = _dyad_duration_summary(joined, groups)
    duration_summary = participant_duration.merge(dyad, on=groups, how="left", validate="one_to_one")
    _atomic_csv(out / "duration_by_vendor_label_split.csv", duration_summary)

    rate_raster = (
        joined.groupby(
            ["vendor_id", "video_r_frame_rate", "video_avg_frame_rate", "video_width", "video_height"],
            dropna=False,
        )
        .agg(
            participant_files=("file_id", "size"),
            participant_stream_hours=("observed_duration_s", lambda value: value.sum(min_count=1) / 3600),
        )
        .reset_index()
        .sort_values(["vendor_id", "participant_files"], ascending=[True, False])
    )
    _atomic_csv(out / "rate_raster_by_vendor.csv", rate_raster)

    participants = (
        joined.groupby(["vendor_id", "participant_id"], dropna=False)
        .agg(
            file_count=("file_id", "size"),
            files_with_duration=("observed_duration_s", "count"),
            duration_hours=("observed_duration_s", lambda value: value.sum(min_count=1) / 3600),
            labels=("label", lambda value: "+".join(sorted(set(value.dropna())))),
            splits=("split", lambda value: "+".join(sorted(set(value.dropna())))),
            participant_metadata_present=("participant_metadata_present", "all"),
        )
        .reset_index()
    )
    _atomic_csv(out / "participant_summary.csv", participants)

    activity_groups = ["interaction_type", "vendor_id", "label", "split"]
    activity_files = (
        joined.groupby(activity_groups, dropna=False)
        .agg(
            participant_files=("file_id", "size"),
            participant_stream_hours=("observed_duration_s", lambda value: value.sum(min_count=1) / 3600),
            distinct_participants=("participant_id", "nunique"),
        )
        .reset_index()
    )
    activity_dyads = _dyad_duration_summary(joined, activity_groups)
    activity = activity_files.merge(
        activity_dyads, on=activity_groups, how="left", validate="one_to_one"
    )
    _atomic_csv(out / "activity_summary.csv", activity)

    # Exact size signature supplied in the brief, cross-checked against ffprobe's
    # independent no-video-stream status. This is inventory characterization,
    # not a filtering threshold.
    placeholders = joined.loc[
        joined.mp4_size_bytes.eq(261) & joined.wav_size_bytes.eq(58)
    ].copy()
    placeholders["no_video_stream"] = placeholders.probe_status.eq("no_video_stream")
    placeholder_columns = [
        "file_id",
        "vendor_id",
        "label",
        "split",
        "source_relbase",
        "json_size_bytes",
        "mp4_size_bytes",
        "npz_size_bytes",
        "wav_size_bytes",
        "probe_status",
        "no_video_stream",
    ]
    _atomic_csv(out / "empty_placeholder_size_signature.csv", placeholders[placeholder_columns])

    expected_paths: dict[str, str] = {}
    for row in joined[["file_id", "source_relbase"]].itertuples(index=False):
        for extension in MODALITY_EXTENSIONS:
            expected_paths[f"{row.source_relbase}{extension}"] = row.file_id
    catalog = catalog.copy()
    catalog["expected_file_id"] = catalog.relative_path.map(expected_paths)
    catalog["is_expected_participant_payload"] = catalog.expected_file_id.notna()
    payload = catalog.loc[catalog.extension.isin(MODALITY_EXTENSIONS)]
    extra = payload.loc[~payload.is_expected_participant_payload]
    _atomic_csv(
        out / "coverage_extra_local_payloads.csv",
        extra[["relative_path", "extension", "file_id", "size_bytes", "stat_status"]],
    )
    missing_rows: list[dict[str, str]] = []
    actual_paths = set(catalog.relative_path)
    for relative_path, file_id in expected_paths.items():
        if relative_path not in actual_paths:
            missing_rows.append(
                {
                    "file_id": file_id,
                    "relative_path": relative_path,
                    "extension": Path(relative_path).suffix,
                }
            )
    missing = pd.DataFrame(missing_rows, columns=["file_id", "relative_path", "extension"])
    _atomic_csv(out / "coverage_missing_expected_payloads.csv", missing)
    bundle_presence = joined[[
        "file_id", "vendor_id", "label", "split", "batch_idx", "archive_idx",
        "source_relbase", "json_present", "mp4_present", "npz_present", "wav_present",
        "all_modalities_present",
    ]]
    _atomic_csv(
        out / "coverage_incomplete_expected_bundles.csv",
        bundle_presence.loc[~bundle_presence.all_modalities_present],
    )

    physical_bytes = (
        catalog.groupby("extension", dropna=False)
        .agg(files=("relative_path", "size"), bytes=("size_bytes", "sum"))
        .reset_index()
    )
    physical_bytes["scope"] = "all_physical_files_in_source_including_archives"
    expected_bytes = []
    for extension in MODALITY_EXTENSIONS:
        name = extension.removeprefix(".")
        expected_bytes.append(
            {
                "extension": extension,
                "files": int(joined[f"{name}_present"].sum()),
                "bytes": int(joined[f"{name}_size_bytes"].fillna(0).sum()),
                "scope": "metadata_expected_extracted_payloads_present_locally",
            }
        )
    bytes_table = pd.concat([physical_bytes, pd.DataFrame(expected_bytes)], ignore_index=True)
    _atomic_csv(out / "bytes_by_extension_and_scope.csv", bytes_table)

    join_summary = {
        "filelist_rows": len(joined),
        "complete_expected_bundles": int(joined.all_modalities_present.sum()),
        "incomplete_expected_bundles": int((~joined.all_modalities_present).sum()),
        "missing_expected_payload_paths": len(missing),
        "extra_local_payload_paths": len(extra),
        "interaction_join_matched": int(joined.interaction_join.eq("both").sum()),
        "interaction_join_missing": int(joined.interaction_join.ne("both").sum()),
        "participant_metadata_rows_matched": int(joined.participant_metadata_present.sum()),
        "participant_metadata_rows_missing": int((~joined.participant_metadata_present).sum()),
        "relationship_metadata_rows_matched": int(joined.relationship_metadata_present.sum()),
        "relationship_metadata_rows_missing": int((~joined.relationship_metadata_present).sum()),
        "physical_file_count": len(catalog),
        "physical_total_bytes": int(catalog.size_bytes.fillna(0).sum()),
        "exact_size_placeholder_candidates": len(placeholders),
        "placeholder_candidates_no_video_stream": int(placeholders.no_video_stream.sum()),
    }
    _atomic_json(out / "coverage_and_join_summary.json", join_summary)


def command_summarize(config: InventoryConfig) -> None:
    provenance, mount = preflight(config, require_clean=True)
    inventory = _read_header_shards(config)
    if not config.catalog_path.is_file():
        raise FileNotFoundError(config.catalog_path)
    catalog = pd.read_parquet(config.catalog_path)
    joined = _join_metadata(config, inventory)
    _write_summary_tables(config, joined, catalog)
    _atomic_json(
        config.summary_dir / "_COMPLETE.json",
        {
            "status": "complete",
            "inventory_rows": len(joined),
            "catalog_rows": len(catalog),
            "config_hash": config.config_hash,
            "git_sha": provenance.git_sha,
            "git_dirty": provenance.git_dirty,
            "source_mount": mount,
        },
    )
    print(json.dumps({"status": "complete", "rows": len(joined)}))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/inventory_m1.yaml"))
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("plan")
    headers = subparsers.add_parser("headers")
    headers.add_argument("--task-index", type=int, required=True)
    headers.add_argument("--workers", type=int)
    subparsers.add_parser("catalog")
    benchmark = subparsers.add_parser("benchmark")
    benchmark.add_argument("--worker-counts", type=int, nargs=4)
    benchmark.add_argument("--samples-per-setting", type=int)
    subparsers.add_parser("summarize")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = load_config(args.config)
    if args.command == "plan":
        command_plan(config)
    elif args.command == "headers":
        command_headers(config, args.task_index, args.workers)
    elif args.command == "catalog":
        command_catalog(config)
    elif args.command == "benchmark":
        command_benchmark(config, args.worker_counts, args.samples_per_setting)
    elif args.command == "summarize":
        command_summarize(config)
    else:  # pragma: no cover
        raise AssertionError(args.command)


if __name__ == "__main__":
    main()

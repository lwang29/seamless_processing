"""Restartable per-file/per-window extraction plumbing."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import re
import subprocess
import uuid
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import soundfile as sf
import yaml

from . import __version__
from .features import (
    GUARD_BAND_DEFAULT_FRAMES,
    Measurement,
    as_binary_mask,
    duration_mismatch_s,
    guard_invalid_runs,
    hand_availability,
    root_translation_accel,
    valid_frac,
    valid_frac_all,
    valid_frac_conjunction,
    wrist_speed_p90,
)


REQUIRED_MANIFEST_COLUMNS = {
    "file_id",
    "label",
    "split",
    "vendor",
    "source_relbase",
}
SAFE_IDENTIFIER = re.compile(r"^[A-Za-z0-9_.-]+$")


ROW_SCHEMA = pa.schema(
    [
        pa.field("file_id", pa.string(), nullable=False),
        pa.field("label", pa.string(), nullable=False),
        pa.field("split", pa.string(), nullable=False),
        pa.field("vendor", pa.string(), nullable=False),
        pa.field("source_relbase", pa.string(), nullable=False),
        pa.field("window_id", pa.string(), nullable=False),
        pa.field("window_index", pa.int32(), nullable=False),
        pa.field("window_start_frame", pa.int64(), nullable=False),
        pa.field("window_end_frame_exclusive", pa.int64(), nullable=False),
        pa.field("window_start_s", pa.float64(), nullable=False),
        pa.field("window_end_s", pa.float64(), nullable=False),
        pa.field("n_frames", pa.int64(), nullable=False),
        # Session 2: the three released masks are reported separately because
        # their conjunction was NaN on 52% of Session-1 rows whenever the
        # optional movement mask was absent. `valid_frac_all` is retained
        # unchanged for continuity with Session-1 outputs.
        pa.field("valid_frac_all", pa.float64()),
        pa.field("valid_frac_smplh", pa.float64()),
        pa.field("valid_frac_box", pa.float64()),
        pa.field("valid_frac_movement", pa.float64()),
        pa.field("valid_frac_smplh_and_box", pa.float64()),
        pa.field("valid_frac_smplh_guarded", pa.float64()),
        pa.field("valid_frac_box_guarded", pa.float64()),
        pa.field("valid_frac_movement_guarded", pa.float64()),
        pa.field("valid_frac_smplh_and_box_guarded", pa.float64()),
        pa.field("guard_band_frames", pa.int32(), nullable=False),
        pa.field("hand_avail_frac_left", pa.float64()),
        pa.field("hand_avail_frac_right", pa.float64()),
        pa.field("hand_avail_status", pa.string(), nullable=False),
        pa.field("hand_avail_basis", pa.string(), nullable=False),
        pa.field("accel_mm_per_frame2", pa.float64()),
        pa.field("wrist_speed_p90", pa.float64()),
        pa.field("duration_mismatch_s", pa.float64()),
        pa.field("smplh_mask_available", pa.bool_(), nullable=False),
        pa.field("movement_mask_available", pa.bool_(), nullable=False),
        pa.field("box_mask_available", pa.bool_(), nullable=False),
        pa.field("mask_status", pa.string(), nullable=False),
        pa.field("valid_status", pa.string(), nullable=False),
        pa.field("valid_split_status", pa.string(), nullable=False),
        pa.field("accel_status", pa.string(), nullable=False),
        pa.field("accel_basis", pa.string(), nullable=False),
        pa.field("translation_scale_to_nominal_mm", pa.float64(), nullable=False),
        pa.field("physical_units_verified", pa.bool_(), nullable=False),
        pa.field("wrist_status", pa.string(), nullable=False),
        pa.field("wrist_basis", pa.string(), nullable=False),
        pa.field("duration_status", pa.string(), nullable=False),
        pa.field("duration_fps_source", pa.string(), nullable=False),
        pa.field("duration_fps_status", pa.string(), nullable=False),
        pa.field("duration_basis", pa.string(), nullable=False),
        pa.field("frame_count_status", pa.string(), nullable=False),
        pa.field("git_sha", pa.string(), nullable=False),
        pa.field("git_dirty", pa.bool_(), nullable=False),
        pa.field("config_hash", pa.string(), nullable=False),
        pa.field("extractor_version", pa.string(), nullable=False),
    ]
)


@dataclass(frozen=True)
class HarnessConfig:
    config_path: Path
    worktree: Path
    source_root: Path
    manifest: Path
    output_root: Path
    per_file_dir: str
    completion_dir: str
    error_dir: str
    nominal_fps: float
    window_frames: int
    hop_frames: int
    translation_scale_to_nominal_mm: float
    translation_basis: str
    physical_units_verified: bool
    wrist_basis: str
    guard_band_frames: int
    ffprobe_bin: str
    config_hash: str


@dataclass(frozen=True)
class Provenance:
    git_sha: str
    git_dirty: bool


@dataclass(frozen=True)
class ProcessResult:
    file_id: str
    status: str
    rows: int
    output_path: Path | None
    marker_path: Path | None
    message: str = ""


def _find_worktree(start: Path) -> Path:
    for candidate in (start, *start.parents):
        if (candidate / ".git-session").is_dir():
            return candidate
    raise ValueError(f"could not find .git-session above {start}")


def _exact_frame_count(seconds: Any, fps: float, field: str) -> int:
    value = float(seconds) * fps
    rounded = round(value)
    if not np.isclose(value, rounded, rtol=0.0, atol=1e-9) or rounded <= 0:
        raise ValueError(f"{field} * fps must be a positive integer, got {value}")
    return int(rounded)


def _relative_output_subdir(value: Any, field: str) -> str:
    path = Path(str(value))
    if path.is_absolute() or not path.parts or any(part in {".", ".."} for part in path.parts):
        raise ValueError(f"{field} must be a nonempty relative path without '.' or '..'")
    return path.as_posix()


def load_config(path: str | Path) -> HarnessConfig:
    config_path = Path(path).expanduser().resolve()
    raw_bytes = config_path.read_bytes()
    raw = yaml.safe_load(raw_bytes) or {}
    worktree = _find_worktree(config_path.parent)

    source_root = (worktree / str(raw["source_root"])).resolve()
    manifest = (worktree / str(raw["manifest"])).resolve()
    output_cfg = raw["outputs"]
    configured_output_root = Path(
        _relative_output_subdir(output_cfg["root"], "outputs.root")
    )
    output_root = (worktree / configured_output_root).resolve()
    if output_root != worktree and worktree not in output_root.parents:
        raise ValueError("outputs.root resolved outside the worktree")
    if output_root == source_root or source_root in output_root.parents:
        raise ValueError("output root must not be inside the read-only source root")

    per_file_dir = _relative_output_subdir(
        output_cfg.get("per_file_dir", "files"), "outputs.per_file_dir"
    )
    completion_dir = _relative_output_subdir(
        output_cfg.get("completion_dir", "complete"), "outputs.completion_dir"
    )
    error_dir = _relative_output_subdir(
        output_cfg.get("error_dir", "errors"), "outputs.error_dir"
    )

    window = raw["window"]
    fps = float(window["fps"])
    if not np.isfinite(fps) or fps <= 0:
        raise ValueError("window.fps must be positive")
    measurements = raw["measurements"]
    physical_units_verified = measurements.get("physical_units_verified")
    if physical_units_verified is not False:
        raise ValueError("Session 1 requires measurements.physical_units_verified: false")

    translation_scale = float(measurements["translation_scale_to_nominal_mm"])
    if not np.isfinite(translation_scale) or translation_scale <= 0:
        raise ValueError("measurements.translation_scale_to_nominal_mm must be positive")

    guard_band_frames = int(
        measurements.get("invalid_guard_band_frames", GUARD_BAND_DEFAULT_FRAMES)
    )
    if guard_band_frames < 0:
        raise ValueError("measurements.invalid_guard_band_frames must be nonnegative")

    return HarnessConfig(
        config_path=config_path,
        worktree=worktree,
        source_root=source_root,
        manifest=manifest,
        output_root=output_root,
        per_file_dir=per_file_dir,
        completion_dir=completion_dir,
        error_dir=error_dir,
        nominal_fps=fps,
        window_frames=_exact_frame_count(window["duration_s"], fps, "window.duration_s"),
        hop_frames=_exact_frame_count(window["hop_s"], fps, "window.hop_s"),
        translation_scale_to_nominal_mm=translation_scale,
        translation_basis=str(measurements["translation_basis"]),
        physical_units_verified=False,
        wrist_basis=str(measurements["wrist_basis"]),
        guard_band_frames=guard_band_frames,
        ffprobe_bin=str(measurements.get("ffprobe_bin", "ffprobe")),
        config_hash=hashlib.sha256(raw_bytes).hexdigest(),
    )


def git_provenance(config: HarnessConfig) -> Provenance:
    common = [
        "git",
        f"--git-dir={config.worktree / '.git-session'}",
        f"--work-tree={config.worktree}",
    ]
    sha = subprocess.run(
        [*common, "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    status = subprocess.run(
        [*common, "status", "--porcelain", "--untracked-files=normal"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    return Provenance(git_sha=sha, git_dirty=bool(status.strip()))


def read_manifest(config: HarnessConfig) -> list[dict[str, str]]:
    with config.manifest.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        missing = REQUIRED_MANIFEST_COLUMNS - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"manifest missing columns: {sorted(missing)}")
        rows = [dict(row) for row in reader]
    for row in rows:
        for field in ("file_id", "label", "split", "vendor"):
            if not SAFE_IDENTIFIER.fullmatch(row[field]):
                raise ValueError(f"unsafe {field} in manifest: {row[field]!r}")
        relbase = Path(row["source_relbase"])
        if relbase.is_absolute() or ".." in relbase.parts:
            raise ValueError(f"unsafe source_relbase: {row['source_relbase']!r}")
    return rows


def _partitioned_path(config: HarnessConfig, directory: str, row: dict[str, str], suffix: str) -> Path:
    path = (
        config.output_root
        / directory
        / row["label"]
        / row["split"]
        / row["vendor"]
        / f"{row['file_id']}{suffix}"
    ).resolve()
    if path != config.output_root and config.output_root not in path.parents:
        raise ValueError("resolved output path escaped outputs.root")
    return path


def _atomic_write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.parent / f".{path.name}.{uuid.uuid4().hex}.tmp"
    try:
        with temporary.open("x", encoding="utf-8") as handle:
            json.dump(value, handle, sort_keys=True, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_write_parquet(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.parent / f".{path.name}.{uuid.uuid4().hex}.tmp"
    try:
        table = pa.Table.from_pylist(rows, schema=ROW_SCHEMA)
        pq.write_table(table, temporary, compression="zstd")
        with temporary.open("rb") as handle:
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _completion_matches(path: Path, output: Path, config: HarnessConfig, provenance: Provenance) -> bool:
    if not path.is_file() or not output.is_file():
        return False
    try:
        marker = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return (
        marker.get("status") == "complete"
        and marker.get("config_hash") == config.config_hash
        and marker.get("git_sha") == provenance.git_sha
        and marker.get("git_dirty") is False
        and provenance.git_dirty is False
    )


def _load_npz_arrays(path: Path) -> tuple[dict[str, np.ndarray | None], set[str]]:
    wanted = {
        "translation": "smplh:translation",
        "keypoints": "boxes_and_keypoints:keypoints",
        "smplh_mask": "smplh:is_valid",
        "movement_mask": "movement:is_valid",
        "box_mask": "boxes_and_keypoints:is_valid_box",
    }
    arrays: dict[str, np.ndarray | None] = {name: None for name in wanted}
    with np.load(path, allow_pickle=False) as archive:
        keys = set(archive.files)
        for name, key in wanted.items():
            if key in keys:
                arrays[name] = np.asarray(archive[key])
    return arrays, keys


def _frame_count(arrays: dict[str, np.ndarray | None]) -> tuple[int, str]:
    precedence = ("translation", "keypoints", "smplh_mask", "movement_mask", "box_mask")
    lengths: dict[str, int] = {}
    for name in precedence:
        array = arrays[name]
        if array is not None and np.asarray(array).ndim >= 1:
            lengths[name] = int(len(np.asarray(array)))
    if not lengths:
        return 0, "missing_frame_arrays"
    basis = next(name for name in precedence if name in lengths)
    n_frames = lengths[basis]
    if len(set(lengths.values())) == 1:
        status = "empty" if n_frames == 0 else "ok"
        return n_frames, f"{status}:basis={basis}"
    detail = ",".join(f"{name}={length}" for name, length in lengths.items())
    return n_frames, f"length_mismatch:basis={basis};{detail}"


def _wav_duration(path: Path) -> tuple[float, str]:
    if not path.is_file():
        return float("nan"), "missing_wav"
    try:
        info = sf.info(path)
    except Exception as exc:  # libsndfile exposes several exception types
        return float("nan"), f"wav_probe_error:{type(exc).__name__}"
    if info.frames < 0 or info.samplerate <= 0:
        return float("nan"), "invalid_wav_metadata"
    return float(info.frames / info.samplerate), "ok"


def _duration_video_fps(
    path: Path,
    config_fps: float,
    ffprobe_bin: str,
) -> tuple[float, str, str]:
    """Return duration FPS, its provenance, and status.

    A file with no video stream is unavailable by definition.  A present video
    stream whose average-frame-rate metadata cannot be parsed uses the nominal
    config FPS only as an explicitly labelled fallback.
    """

    if not path.is_file():
        return float("nan"), "unavailable", "unavailable_missing_mp4"
    command = [
        ffprobe_bin,
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=avg_frame_rate",
        "-of",
        "json",
        str(path),
    ]
    try:
        completed = subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
            timeout=30,
        )
        payload = json.loads(completed.stdout)
    except (OSError, subprocess.SubprocessError, json.JSONDecodeError) as exc:
        return (
            config_fps,
            f"config.window.fps={config_fps:g}",
            f"fallback_config_ffprobe_error:{type(exc).__name__}",
        )
    streams = payload.get("streams", [])
    if not streams:
        return float("nan"), "unavailable", "unavailable_no_video_stream"
    raw_rate = streams[0].get("avg_frame_rate")
    try:
        rate = Fraction(str(raw_rate))
        fps = float(rate)
    except (ValueError, ZeroDivisionError, TypeError, OverflowError):
        fps = float("nan")
    if not np.isfinite(fps) or fps <= 0:
        return (
            config_fps,
            f"config.window.fps={config_fps:g}",
            f"fallback_config_unusable_avg_frame_rate:{raw_rate}",
        )
    return fps, f"video.avg_frame_rate={raw_rate}", "ok_video_avg_frame_rate"


def _mask_for_window(
    mask: np.ndarray | None,
    n_frames: int,
    start: int,
    end: int,
) -> np.ndarray | None:
    if mask is None or len(mask) != n_frames:
        return None
    return mask[start:end]


def _array_for_window(
    array: np.ndarray | None,
    n_frames: int,
    start: int,
    end: int,
) -> np.ndarray | None:
    if array is None or np.asarray(array).ndim < 1 or len(array) != n_frames:
        return None
    return np.asarray(array)[start:end]


def _merge_status(measurement: Measurement, context: str) -> str:
    return measurement.status if context == "ok" else f"{measurement.status};{context}"


def _window_rows(
    manifest_row: dict[str, str],
    arrays: dict[str, np.ndarray | None],
    config: HarnessConfig,
    provenance: Provenance,
    wav_duration_s: float,
    wav_status: str,
    duration_fps: float,
    duration_fps_source: str,
    duration_fps_status: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    masks: dict[str, np.ndarray | None] = {}
    mask_normalization: dict[str, str] = {}
    for name in ("smplh_mask", "movement_mask", "box_mask"):
        masks[name], mask_normalization[name] = as_binary_mask(arrays[name])

    # Guarding happens on the whole-file mask, before any window slicing: a
    # window guarded in isolation cannot see a failure starting one frame past
    # its own end.
    guarded_masks = {
        name: guard_invalid_runs(mask, config.guard_band_frames)
        for name, mask in masks.items()
    }

    n_frames, frame_count_status = _frame_count(arrays)
    lengths_match = {
        name: mask is not None and len(mask) == n_frames for name, mask in masks.items()
    }
    mask_status = ";".join(
        f"{name.removesuffix('_mask')}={mask_normalization[name]}"
        + ("" if masks[name] is None or lengths_match[name] else ":length_mismatch")
        for name in ("smplh_mask", "movement_mask", "box_mask")
    )
    mismatch = duration_mismatch_s(n_frames, duration_fps, wav_duration_s)
    duration_status = f"wav={wav_status};fps={duration_fps_status}"
    if wav_status != "ok" or not np.isfinite(duration_fps):
        mismatch = float("nan")

    common = {
        "file_id": manifest_row["file_id"],
        "label": manifest_row["label"],
        "split": manifest_row["split"],
        "vendor": manifest_row["vendor"],
        "source_relbase": manifest_row["source_relbase"],
        "n_frames": n_frames,
        "duration_mismatch_s": mismatch,
        "smplh_mask_available": arrays["smplh_mask"] is not None,
        "movement_mask_available": arrays["movement_mask"] is not None,
        "box_mask_available": arrays["box_mask"] is not None,
        "mask_status": mask_status,
        "accel_basis": (
            "provisional_second_difference_of_root_translation;"
            f"basis={config.translation_basis};nominal_mm_per_frame2;physical_units_UNVERIFIED"
        ),
        "translation_scale_to_nominal_mm": config.translation_scale_to_nominal_mm,
        "physical_units_verified": config.physical_units_verified,
        "wrist_basis": config.wrist_basis,
        "guard_band_frames": config.guard_band_frames,
        "hand_avail_basis": (
            "fraction of frames whose 21-point COCO-WholeBody hand XY block is "
            "finite and not exact all-zero blank; confidence not thresholded; "
            "left=91:112 right=112:133"
        ),
        "duration_status": duration_status,
        "duration_fps_source": duration_fps_source,
        "duration_fps_status": duration_fps_status,
        "duration_basis": (
            "abs(annotation_frames/duration_fps-wav_header_duration);"
            "duration_fps_is_separate_from_fixed_config_window_grid"
        ),
        "frame_count_status": frame_count_status,
        "git_sha": provenance.git_sha,
        "git_dirty": provenance.git_dirty,
        "config_hash": config.config_hash,
        "extractor_version": __version__,
    }

    rows: list[dict[str, Any]] = []
    if n_frames < config.window_frames:
        return rows, {
            "n_frames": n_frames,
            "frame_count_status": frame_count_status,
            "mask_status": mask_status,
            "duration_status": duration_status,
            "duration_fps_source": duration_fps_source,
            "duration_fps_status": duration_fps_status,
        }

    for window_index, start in enumerate(
        range(0, n_frames - config.window_frames + 1, config.hop_frames)
    ):
        end = start + config.window_frames
        smplh = _mask_for_window(masks["smplh_mask"], n_frames, start, end)
        movement = _mask_for_window(masks["movement_mask"], n_frames, start, end)
        box = _mask_for_window(masks["box_mask"], n_frames, start, end)
        valid = valid_frac_all(smplh, movement, box)
        has_length_mismatch = any(
            mask is not None and len(mask) != n_frames for mask in masks.values()
        )
        mask_context = (
            "one_or_more_mask_length_mismatch" if has_length_mismatch else "ok"
        )

        translation = _array_for_window(arrays["translation"], n_frames, start, end)
        accel = root_translation_accel(
            translation,
            smplh,
            config.translation_scale_to_nominal_mm,
        )
        keypoints = _array_for_window(arrays["keypoints"], n_frames, start, end)
        wrist = wrist_speed_p90(keypoints, box)

        window_masks = {"smplh": smplh, "box": box, "movement": movement}
        window_guarded = {
            short: _mask_for_window(guarded_masks[f"{short}_mask"], n_frames, start, end)
            for short in ("smplh", "box", "movement")
        }
        split = {
            short: valid_frac(window_masks[short], name=short)
            for short in ("smplh", "box", "movement")
        }
        split_guarded = {
            short: valid_frac(window_guarded[short], name=short)
            for short in ("smplh", "box", "movement")
        }
        # smplh and box are the two masks the body-and-hands model depends on.
        # movement stays out of the conjunction: it is optional metadata and
        # must never reduce a body measurement.
        core = valid_frac_conjunction(window_masks, required=("smplh", "box"))
        core_guarded = valid_frac_conjunction(window_guarded, required=("smplh", "box"))
        left_hand = hand_availability(keypoints, "left")
        right_hand = hand_availability(keypoints, "right")

        row = dict(common)
        row.update(
            {
                "window_id": f"{manifest_row['file_id']}:{start:08d}",
                "window_index": window_index,
                "window_start_frame": start,
                "window_end_frame_exclusive": end,
                "window_start_s": start / config.nominal_fps,
                "window_end_s": end / config.nominal_fps,
                "valid_frac_all": valid.value,
                "valid_frac_smplh": split["smplh"].value,
                "valid_frac_box": split["box"].value,
                "valid_frac_movement": split["movement"].value,
                "valid_frac_smplh_and_box": core.value,
                "valid_frac_smplh_guarded": split_guarded["smplh"].value,
                "valid_frac_box_guarded": split_guarded["box"].value,
                "valid_frac_movement_guarded": split_guarded["movement"].value,
                "valid_frac_smplh_and_box_guarded": core_guarded.value,
                "hand_avail_frac_left": left_hand.value,
                "hand_avail_frac_right": right_hand.value,
                "accel_mm_per_frame2": accel.value,
                "wrist_speed_p90": wrist.value,
                "valid_status": _merge_status(valid, mask_context),
                "valid_split_status": ";".join(
                    [
                        *(f"{short}={split[short].status}" for short in ("smplh", "box", "movement")),
                        f"smplh_and_box={core.status}",
                    ]
                ),
                "hand_avail_status": f"left={left_hand.status};right={right_hand.status}",
                "accel_status": accel.status,
                "wrist_status": wrist.status,
            }
        )
        rows.append(row)
    return rows, {
        "n_frames": n_frames,
        "frame_count_status": frame_count_status,
        "mask_status": mask_status,
        "duration_status": duration_status,
        "duration_fps_source": duration_fps_source,
        "duration_fps_status": duration_fps_status,
    }


def process_file(
    manifest_row: dict[str, str],
    config: HarnessConfig,
    provenance: Provenance,
    force: bool = False,
) -> ProcessResult:
    output_path = _partitioned_path(config, config.per_file_dir, manifest_row, ".parquet")
    marker_path = _partitioned_path(config, config.completion_dir, manifest_row, ".json")
    error_path = _partitioned_path(config, config.error_dir, manifest_row, ".json")
    if not force and _completion_matches(marker_path, output_path, config, provenance):
        marker = json.loads(marker_path.read_text(encoding="utf-8"))
        return ProcessResult(
            file_id=manifest_row["file_id"],
            status="skipped",
            rows=int(marker["rows"]),
            output_path=output_path,
            marker_path=marker_path,
        )

    source_base = (config.source_root / manifest_row["source_relbase"]).resolve()
    if config.source_root != source_base and config.source_root not in source_base.parents:
        raise ValueError("resolved source path escaped source root")
    npz_path = Path(f"{source_base}.npz")
    wav_path = Path(f"{source_base}.wav")
    try:
        if not npz_path.is_file():
            raise FileNotFoundError(npz_path)
        arrays, npz_keys = _load_npz_arrays(npz_path)
        wav_duration_s, duration_status = _wav_duration(wav_path)
        mp4_path = Path(f"{source_base}.mp4")
        duration_fps, duration_fps_source, duration_fps_status = _duration_video_fps(
            mp4_path,
            config.nominal_fps,
            config.ffprobe_bin,
        )
        rows, summary = _window_rows(
            manifest_row,
            arrays,
            config,
            provenance,
            wav_duration_s,
            duration_status,
            duration_fps,
            duration_fps_source,
            duration_fps_status,
        )
        _atomic_write_parquet(output_path, rows)
        marker = {
            "status": "complete",
            "file_id": manifest_row["file_id"],
            "rows": len(rows),
            "output": str(output_path.relative_to(config.worktree)),
            "source_npz": str(npz_path.relative_to(config.source_root)),
            "npz_keys_present": sorted(npz_keys),
            "config_hash": config.config_hash,
            "git_sha": provenance.git_sha,
            "git_dirty": provenance.git_dirty,
            "extractor_version": __version__,
            **summary,
        }
        _atomic_write_json(marker_path, marker)
        error_path.unlink(missing_ok=True)
        return ProcessResult(
            file_id=manifest_row["file_id"],
            status="complete",
            rows=len(rows),
            output_path=output_path,
            marker_path=marker_path,
        )
    except Exception as exc:
        _atomic_write_json(
            error_path,
            {
                "status": "error_not_complete",
                "file_id": manifest_row["file_id"],
                "error_type": type(exc).__name__,
                "message": str(exc),
                "config_hash": config.config_hash,
                "git_sha": provenance.git_sha,
            },
        )
        return ProcessResult(
            file_id=manifest_row["file_id"],
            status="error",
            rows=0,
            output_path=None,
            marker_path=None,
            message=f"{type(exc).__name__}: {exc}",
        )


def select_manifest_rows(
    rows: list[dict[str, str]],
    limit: int | None = None,
    index: int | None = None,
) -> list[dict[str, str]]:
    if limit is not None:
        if limit < 0:
            raise ValueError("--limit must be nonnegative")
        rows = rows[:limit]
    if index is not None:
        if index < 0 or index >= len(rows):
            raise IndexError(f"--index {index} outside selected manifest range 0..{len(rows)-1}")
        rows = [rows[index]]
    return rows


def run(
    config: HarnessConfig,
    manifest_rows: Iterable[dict[str, str]],
    force: bool = False,
) -> list[ProcessResult]:
    provenance = git_provenance(config)
    return [process_file(row, config, provenance, force=force) for row in manifest_rows]

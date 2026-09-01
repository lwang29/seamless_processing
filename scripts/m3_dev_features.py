#!/usr/bin/env python3
"""M-3: characterize prototype signal distributions on the dev sample.

The unit of measurement is a fixed-length span, not a whole file, so that a
signal's spread within a recording is visible rather than averaged away. Spans
are chosen on a deterministic stride from the start of each file and are not
steered towards valid frames: a characterization that quietly skipped failures
would describe a corpus we do not have.

No threshold is applied and nothing is rejected on quality grounds. The one
parameter that unavoidably needs a number, the rest-pose displacement, is
reported as a parameter with its own percentile distribution alongside.
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from fractions import Fraction
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from scripts.smplh_fk import create_neutral_smplh, forward_smplh_joints  # noqa: E402
from seamless_curation.features import (  # noqa: E402
    as_binary_mask,
    guard_invalid_runs,
    valid_frac,
    valid_frac_conjunction,
)
from seamless_curation.m3_features import (  # noqa: E402
    angular_velocity_signals,
    hand_quality,
    metric_3d_signals,
    normalized_2d_kinematics,
    rest_exit_frac,
    speaking_fraction,
)
from seamless_curation.review_renderer import load_vad  # noqa: E402


SMPLH_KEYS = (
    "smplh:global_orient",
    "smplh:body_pose",
    "smplh:left_hand_pose",
    "smplh:right_hand_pose",
    "smplh:translation",
)


def _worktree() -> Path:
    here = Path(__file__).resolve().parent
    for candidate in (here, *here.parents):
        if (candidate / ".git-session").is_dir():
            return candidate
    raise RuntimeError("could not locate .git-session")


def _git_provenance(worktree: Path) -> dict[str, Any]:
    base = ["git", f"--git-dir={worktree / '.git-session'}", f"--work-tree={worktree}"]
    sha = subprocess.run([*base, "rev-parse", "HEAD"], check=True, capture_output=True, text=True)
    status = subprocess.run(
        [*base, "status", "--porcelain", "--untracked-files=normal"],
        check=True,
        capture_output=True,
        text=True,
    )
    return {"git_sha": sha.stdout.strip(), "git_dirty": bool(status.stdout.strip())}


def _measured_fps(mp4: Path, ffprobe: str) -> tuple[float, str]:
    """Always the file's own measured average rate; never an assumed 30."""

    if not mp4.is_file():
        return float("nan"), "missing_mp4"
    command = [
        ffprobe, "-v", "error", "-select_streams", "v:0",
        "-show_entries", "stream=avg_frame_rate,width,height", "-of", "json", str(mp4),
    ]
    try:
        payload = json.loads(
            subprocess.run(command, check=True, capture_output=True, text=True, timeout=60).stdout
        )
    except (OSError, subprocess.SubprocessError, json.JSONDecodeError) as exc:
        return float("nan"), f"ffprobe_error:{type(exc).__name__}"
    streams = payload.get("streams") or []
    if not streams:
        return float("nan"), "no_video_stream"
    try:
        fps = float(Fraction(str(streams[0]["avg_frame_rate"])))
    except (KeyError, ValueError, ZeroDivisionError):
        return float("nan"), "unusable_avg_frame_rate"
    if not np.isfinite(fps) or fps <= 0:
        return float("nan"), "unusable_avg_frame_rate"
    return fps, "ok"


def _raster(mp4: Path, ffprobe: str) -> tuple[int, int, str]:
    command = [
        ffprobe, "-v", "error", "-select_streams", "v:0",
        "-show_entries", "stream=width,height", "-of", "json", str(mp4),
    ]
    try:
        payload = json.loads(
            subprocess.run(command, check=True, capture_output=True, text=True, timeout=60).stdout
        )
        stream = (payload.get("streams") or [{}])[0]
        return int(stream["width"]), int(stream["height"]), "ok"
    except (OSError, subprocess.SubprocessError, json.JSONDecodeError, KeyError, ValueError):
        return 0, 0, "raster_unavailable"


def _span_starts(n_frames: int, span_frames: int, spans: int) -> list[int]:
    if n_frames < span_frames:
        return []
    if spans <= 1:
        return [0]
    last = n_frames - span_frames
    return sorted({int(round(index * last / (spans - 1))) for index in range(spans)})


def _fk_root_relative_mm(
    model: Any,
    payload: dict[str, np.ndarray],
    batch_size: int,
    device: str,
) -> np.ndarray:
    """FK with translation zeroed and the pelvis subtracted, in millimetres."""

    frames = len(payload["smplh:global_orient"])
    chunks: list[np.ndarray] = []
    for offset in range(0, frames, batch_size):
        stop = min(frames, offset + batch_size)
        real = stop - offset
        chunk: dict[str, np.ndarray] = {}
        for key, array in payload.items():
            values = array[offset:stop]
            if real < batch_size:
                values = np.concatenate(
                    (values, np.repeat(values[-1:], batch_size - real, axis=0)), axis=0
                )
            chunk[key] = values
        _, root_relative = forward_smplh_joints(
            model, chunk, device=device, include_translation=False
        )
        chunks.append(root_relative[:real])
    return np.concatenate(chunks, axis=0).astype(np.float64) * 1000.0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/m3_dev_features.yaml"))
    parser.add_argument("--limit", type=int, help="process at most this many manifest rows")
    args = parser.parse_args()

    worktree = _worktree()
    config = yaml.safe_load((worktree / args.config).read_text(encoding="utf-8"))
    source_root = (worktree / str(config["source_root"]).lstrip("./")).resolve()
    output_dir = worktree / str(config["outputs"]["root"]).lstrip("./")
    output_dir.mkdir(parents=True, exist_ok=True)
    span_frames = int(config["span"]["frames"])
    spans_per_file = int(config["span"]["per_file"])
    guard = int(config["measurement"]["invalid_guard_band_frames"])
    rest_mm = float(config["measurement"]["rest_exit_displacement_mm"])
    ffprobe = str(config["measurement"].get("ffprobe_bin", "ffprobe"))
    device = str(config["smplh"].get("device", "cpu"))
    batch_size = int(config["smplh"].get("batch_size", 256))

    with (worktree / str(config["manifest"]).lstrip("./")).open(newline="", encoding="utf-8") as handle:
        manifest = list(csv.DictReader(handle))
    if args.limit is not None:
        manifest = manifest[: args.limit]

    model = create_neutral_smplh(
        worktree / str(config["smplh"]["model_root"]).lstrip("./"),
        flat_hand_mean=bool(config["smplh"]["flat_hand_mean"]),
        batch_size=batch_size,
        device=device,
    )

    rows: list[dict[str, Any]] = []
    file_notes: list[dict[str, Any]] = []
    for row in manifest:
        base = source_root / row["source_relbase"]
        note: dict[str, Any] = {
            "file_id": row["file_id"],
            "vendor": row["vendor"],
            "label": row["label"],
            "split": row["split"],
        }
        fps, fps_status = _measured_fps(base.with_suffix(".mp4"), ffprobe)
        width, height, raster_status = _raster(base.with_suffix(".mp4"), ffprobe)
        note.update({"fps": fps, "fps_status": fps_status, "raster_status": raster_status,
                     "width": width, "height": height})
        if fps_status != "ok" or raster_status != "ok":
            note["status"] = f"skipped:{fps_status}/{raster_status}"
            file_notes.append(note)
            continue
        try:
            with np.load(base.with_suffix(".npz"), allow_pickle=False) as archive:
                names = set(archive.files)
                missing = [key for key in (*SMPLH_KEYS, "boxes_and_keypoints:keypoints") if key not in names]
                if missing:
                    note["status"] = "skipped:missing_arrays:" + ",".join(missing)
                    file_notes.append(note)
                    continue
                keypoints = np.asarray(archive["boxes_and_keypoints:keypoints"])
                boxes = (
                    np.asarray(archive["boxes_and_keypoints:box"])
                    if "boxes_and_keypoints:box" in names
                    else None
                )
                smplh_mask, _ = as_binary_mask(archive["smplh:is_valid"])
                box_mask, _ = as_binary_mask(archive["boxes_and_keypoints:is_valid_box"])
                movement_mask, _ = (
                    as_binary_mask(archive["movement:is_valid"])
                    if "movement:is_valid" in names
                    else (None, "missing")
                )
                pose = {key: np.asarray(archive[key], dtype=np.float32) for key in SMPLH_KEYS}
        except (OSError, ValueError, KeyError) as exc:
            note["status"] = f"skipped:npz_error:{type(exc).__name__}"
            file_notes.append(note)
            continue

        n_frames = min(
            len(keypoints), *(len(array) for array in pose.values()),
            *(len(m) for m in (smplh_mask, box_mask) if m is not None),
        )
        vad_intervals = load_vad(base.with_suffix(".json"))
        file_vad = speaking_fraction(vad_intervals, n_frames=n_frames, fps=fps)
        guarded = {
            name: guard_invalid_runs(mask, guard)
            for name, mask in (("smplh", smplh_mask), ("box", box_mask), ("movement", movement_mask))
        }
        starts = _span_starts(n_frames, span_frames, spans_per_file)
        note.update({"status": "ok", "n_frames": n_frames, "spans": len(starts),
                     "movement_mask_present": movement_mask is not None})
        file_notes.append(note)

        for span_index, start in enumerate(starts):
            end = start + span_frames
            span_keypoints = keypoints[start:end]
            span_pose = {key: array[start:end] for key, array in pose.items()}
            span_masks = {
                "smplh": smplh_mask[start:end] if smplh_mask is not None else None,
                "box": box_mask[start:end] if box_mask is not None else None,
                "movement": movement_mask[start:end] if movement_mask is not None else None,
            }
            span_guarded = {
                name: (mask[start:end] if mask is not None else None)
                for name, mask in guarded.items()
            }
            record: dict[str, Any] = {
                "file_id": row["file_id"],
                "vendor": row["vendor"],
                "label": row["label"],
                "split": row["split"],
                "span_index": span_index,
                "span_start_frame": start,
                "span_frames": span_frames,
                "fps": fps,
                "width": width,
                "height": height,
                "guard_band_frames": guard,
                **{f"valid_frac_{name}": valid_frac(mask, name=name).value
                   for name, mask in span_masks.items()},
                **{f"valid_frac_{name}_guarded": valid_frac(mask, name=name).value
                   for name, mask in span_guarded.items()},
                "valid_frac_smplh_and_box": valid_frac_conjunction(
                    span_masks, required=("smplh", "box")
                ).value,
                "valid_frac_smplh_and_box_guarded": valid_frac_conjunction(
                    span_guarded, required=("smplh", "box")
                ).value,
                # Span-local, so this column is not a file-level number wearing a
                # per-span label. The file-level value is kept beside it.
                **speaking_fraction(
                    vad_intervals, n_frames=span_frames, fps=fps,
                    start_frame=start, media_frames=n_frames,
                ),
                "speaking_frac_file": file_vad.get("speaking_frac"),
                "vad_file_status": file_vad.get("vad_status"),
            }
            record.update(
                normalized_2d_kinematics(
                    span_keypoints, width=width, height=height, fps=fps,
                    boxes=boxes[start:end] if boxes is not None else None,
                )
            )
            for hand in ("left", "right"):
                record.update(
                    hand_quality(
                        span_keypoints,
                        span_pose[f"smplh:{hand}_hand_pose"],
                        hand=hand, width=width, height=height,
                    )
                )
            record.update(
                angular_velocity_signals(span_pose, fps=fps, smplh_valid=span_masks["smplh"])
            )
            try:
                root_relative_mm = _fk_root_relative_mm(model, span_pose, batch_size, device)
            except Exception as exc:  # FK failure must not lose the 2D signals
                record["metric3d_status"] = f"fk_error:{type(exc).__name__}:{exc}"
            else:
                record.update(
                    metric_3d_signals(
                        root_relative_mm, fps=fps, frame_valid=span_masks["smplh"]
                    )
                )
                record.update(
                    rest_exit_frac(
                        root_relative_mm,
                        displacement_mm=rest_mm,
                        frame_valid=span_masks["smplh"],
                    )
                )
            rows.append(record)

    frame = pd.DataFrame(rows)
    frame.to_parquet(output_dir / "span_features.parquet", index=False)
    pd.DataFrame(file_notes).to_csv(output_dir / "file_status.csv", index=False)
    summary = {
        "spans": len(frame),
        "files_attempted": len(manifest),
        "files_with_spans": int(frame.file_id.nunique()) if len(frame) else 0,
        "span_frames": span_frames,
        "spans_per_file": spans_per_file,
        "guard_band_frames": guard,
        "rest_exit_displacement_mm": rest_mm,
        "smplh": {
            "gender": "neutral", "betas": "zeros(16)", "use_pca": False,
            "flat_hand_mean": bool(config["smplh"]["flat_hand_mean"]),
            "device": device, "batch_size": batch_size,
        },
        "fk_translation": "zeroed; pelvis subtracted; metres x 1000 = millimetres",
        **_git_provenance(worktree),
    }
    (output_dir / "run_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

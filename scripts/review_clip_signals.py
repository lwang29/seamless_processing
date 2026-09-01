#!/usr/bin/env python3
"""Compute the M-3 signal set over the exact interval of each review clip.

This is the substrate for P0.3's central question: do any computed signals
separate the clips humans call bad? Correlating ratings against signals measured
over a different interval than the reviewer watched would be meaningless, so the
window here is exactly ``[start_frame, start_frame + round(10 s * avg_fps))`` —
the same interval the renderer encoded.
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

from scripts.m3_dev_features import (  # noqa: E402
    SMPLH_KEYS,
    _fk_root_relative_mm,
    _git_provenance,
    _raster,
    _worktree,
)
from scripts.smplh_fk import create_neutral_smplh  # noqa: E402
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


CLIP_DURATION_S = 10.0


def _clip_fps(mp4: Path, ffprobe: str) -> float:
    command = [
        ffprobe, "-v", "error", "-select_streams", "v:0",
        "-show_entries", "stream=avg_frame_rate", "-of", "json", str(mp4),
    ]
    payload = json.loads(
        subprocess.run(command, check=True, capture_output=True, text=True, timeout=60).stdout
    )
    streams = payload.get("streams") or []
    if not streams:
        raise ValueError("no video stream")
    return float(Fraction(str(streams[0]["avg_frame_rate"])))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/session2_review.yaml"))
    parser.add_argument("--manifest", type=Path, help="override the config manifest")
    parser.add_argument(
        "--out", type=Path, default=Path("outputs/session2/review_clip_signals")
    )
    parser.add_argument("--limit", type=int)
    parser.add_argument("--ffprobe", default="/usr/bin/ffprobe")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=256)
    args = parser.parse_args()

    worktree = _worktree()
    config = yaml.safe_load((worktree / args.config).read_text(encoding="utf-8"))
    source_root = (worktree / str(config["source_root"]).lstrip("./")).resolve()
    manifest_path = worktree / str(
        args.manifest or str(config["outputs"]["manifest"]).lstrip("./")
    )
    guard = int(config["measurement"]["invalid_guard_band_frames"])
    out_dir = worktree / args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    with manifest_path.open(newline="", encoding="utf-8") as handle:
        records = list(csv.DictReader(handle))
    if args.limit is not None:
        records = records[: args.limit]

    model = create_neutral_smplh(
        worktree / str(config["model_root"]).lstrip("./"),
        flat_hand_mean=bool(config["smplh"]["flat_hand_mean"]),
        batch_size=args.batch_size,
        device=args.device,
    )

    rows: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    # Media is opened once per distinct clip; duplicate review items reuse it,
    # so agreement duplicates cost nothing here either.
    computed: dict[str, dict[str, Any]] = {}
    for record in records:
        clip_id = record["clip_id"]
        identity = {
            "review_item_id": record["review_item_id"],
            "clip_id": clip_id,
            "file_id": record["file_id"],
            "sample_group": record["sample_group"],
            "vendor": record.get("vendor", ""),
            "label": record.get("label", ""),
            "split": record.get("split", ""),
            "activity_type": record.get("activity_type", ""),
            "selection_reason": record.get("selection_reason", ""),
        }
        if clip_id in computed:
            rows.append({**identity, **computed[clip_id]})
            continue
        if record.get("render_policy", "render") == "metadata_only":
            skipped.append({**identity, "skip_reason": "metadata_only_unrenderable_bundle"})
            continue
        base = source_root / record["source_relbase"]
        try:
            fps = _clip_fps(base.with_suffix(".mp4"), args.ffprobe)
            width, height, raster_status = _raster(base.with_suffix(".mp4"), args.ffprobe)
            if raster_status != "ok" or not np.isfinite(fps) or fps <= 0:
                raise ValueError(f"raster={raster_status} fps={fps}")
            start = int(record["start_frame"])
            frames = int(round(CLIP_DURATION_S * fps))
            end = start + frames
            with np.load(base.with_suffix(".npz"), allow_pickle=False) as archive:
                names = set(archive.files)
                keypoints = np.asarray(archive["boxes_and_keypoints:keypoints"][start:end])
                boxes = (
                    np.asarray(archive["boxes_and_keypoints:box"][start:end])
                    if "boxes_and_keypoints:box" in names
                    else None
                )
                smplh_mask_full, _ = as_binary_mask(archive["smplh:is_valid"])
                box_mask_full, _ = as_binary_mask(archive["boxes_and_keypoints:is_valid_box"])
                movement_full, _ = (
                    as_binary_mask(archive["movement:is_valid"])
                    if "movement:is_valid" in names
                    else (None, "missing")
                )
                pose = {
                    key: np.asarray(archive[key][start:end], dtype=np.float32)
                    for key in SMPLH_KEYS
                }
            if len(keypoints) != frames or any(len(a) != frames for a in pose.values()):
                raise ValueError("clip interval not fully covered by annotations")
            masks = {
                "smplh": smplh_mask_full[start:end] if smplh_mask_full is not None else None,
                "box": box_mask_full[start:end] if box_mask_full is not None else None,
                "movement": movement_full[start:end] if movement_full is not None else None,
            }
            # Guard on the whole-file mask, then slice, so a failure starting one
            # frame past the clip still shortens the clip's guarded fraction.
            guarded = {
                name: (
                    guard_invalid_runs(full, guard)[start:end] if full is not None else None
                )
                for name, full in (
                    ("smplh", smplh_mask_full), ("box", box_mask_full), ("movement", movement_full)
                )
            }
            signals: dict[str, Any] = {
                "clip_start_frame": start,
                "clip_frames": frames,
                "fps": fps,
                "width": width,
                "height": height,
                "guard_band_frames": guard,
                **{f"valid_frac_{n}": valid_frac(m, name=n).value for n, m in masks.items()},
                **{f"valid_frac_{n}_guarded": valid_frac(m, name=n).value for n, m in guarded.items()},
                "valid_frac_smplh_and_box": valid_frac_conjunction(
                    masks, required=("smplh", "box")
                ).value,
                "valid_frac_smplh_and_box_guarded": valid_frac_conjunction(
                    guarded, required=("smplh", "box")
                ).value,
                # Span-local over exactly the rendered interval.
                **speaking_fraction(
                    load_vad(base.with_suffix(".json")),
                    n_frames=frames, fps=fps, start_frame=start,
                ),
                **normalized_2d_kinematics(
                    keypoints, width=width, height=height, fps=fps, boxes=boxes
                ),
            }
            for hand in ("left", "right"):
                signals.update(
                    hand_quality(
                        keypoints, pose[f"smplh:{hand}_hand_pose"],
                        hand=hand, width=width, height=height,
                    )
                )
            signals.update(angular_velocity_signals(pose, fps=fps, smplh_valid=masks["smplh"]))
            root_relative_mm = _fk_root_relative_mm(model, pose, args.batch_size, args.device)
            signals.update(
                metric_3d_signals(root_relative_mm, fps=fps, frame_valid=masks["smplh"])
            )
            signals.update(
                rest_exit_frac(
                    root_relative_mm, displacement_mm=100.0, frame_valid=masks["smplh"]
                )
            )
        except Exception as exc:
            skipped.append({**identity, "skip_reason": f"{type(exc).__name__}: {exc}"})
            continue
        computed[clip_id] = signals
        rows.append({**identity, **signals})

    frame = pd.DataFrame(rows)
    frame.to_parquet(out_dir / "clip_signals.parquet", index=False)
    pd.DataFrame(skipped).to_csv(out_dir / "skipped_clips.csv", index=False)
    summary = {
        "manifest": str(manifest_path.relative_to(worktree)),
        "review_items_with_signals": len(frame),
        "distinct_clips_measured": len(computed),
        "skipped": len(skipped),
        "skip_reasons": (
            pd.DataFrame(skipped).skip_reason.value_counts().to_dict() if skipped else {}
        ),
        "clip_duration_s": CLIP_DURATION_S,
        "guard_band_frames": guard,
        **_git_provenance(worktree),
    }
    (out_dir / "run_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Run the three failure-mode detectors over a random sample of files.

One NPZ read per file supplies everything: the released 2D keypoints and boxes,
the validity masks, and the SMPL-H pose arrays that forward kinematics turns
into both the joint angles FM1 needs and the projected points and mesh vertices
FM2 tests. Nothing here reads VAD.

FK runs on **every frame**, on CPU, because it is fast enough — 3,705 frames/s
measured — and because FM2 is stated as "every tracked SMPL-H point on every
frame". Subsampling would have made the strictest of the three detectors the only
approximate one.

Measurements only. Thresholds are applied afterwards by
``scripts/v00_fm_analyze.py`` so that re-tuning never requires re-reading media.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from fractions import Fraction
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from scripts.smplh_fk import (  # noqa: E402
    create_neutral_smplh,
    forward_smplh_joints,
    project_hmr2_full_frame,
)
from seamless_curation.features import as_binary_mask  # noqa: E402
from seamless_curation.media_repair import (  # noqa: E402
    AnnotationTimebase,
    RasterRepair,
    body_roll_deg,
    probe_sample_aspect,
    quarter_turns_from_roll,
)
from scripts.probe_audio_quality import (  # noqa: E402
    audio_quality,
    vad_intervals,
    voice_isolation_db,
)
from seamless_curation.v00_detectors import (  # noqa: E402
    in_frame_violations,
    leg_visibility,
    posture_angles,
    signed_inset_px,
    smplh_invalid_runs,
    smplh_reprojection_error,
    sitting_measures_2d,
    static_hand_measure,
    tracker_discontinuity,
    windowed_in_frame_pass_frac,
)


SMPLH_KEYS = (
    "smplh:global_orient", "smplh:body_pose", "smplh:left_hand_pose",
    "smplh:right_hand_pose", "smplh:translation",
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
        check=True, capture_output=True, text=True,
    )
    return {"git_sha": sha.stdout.strip(), "git_dirty": bool(status.stdout.strip())}


def build_sample(
    inventory: Path,
    *,
    per_label: int | None = None,
    seed: int,
    vendors: Sequence[str] = ("00",),
    per_vendor: int | None = None,
    include_rasters: Sequence[str] = (),
) -> pd.DataFrame:
    """Random draw over the eligible pool, charades excluded.

    Two stratifications, because the vendors do not have the same structure. The
    V00 work draws equally per **label**, which is what makes naturalistic and
    improvised flag rates comparable. Reaching outside V00 that stratification
    stops making sense — V02 is naturalistic only — so ``per_vendor`` draws
    equally per **vendor** instead and lets the label mix fall out as it is.
    """

    frame = pd.read_parquet(inventory)
    eligible = frame.loc[
        frame.vendor_id.isin(list(vendors))
        & frame.all_modalities_present.astype(bool)
        & frame.probe_status.eq("ok")
        & frame.video_stream_present.astype(bool)
        & frame.observed_duration_s.notna()
        & frame.observed_duration_s.ge(30.0)
        & frame.video_nb_frames.notna()
        & frame.interaction_type.ne("charades")
    ].copy()
    # Partner lookup covers the whole eligible set, not just the sample, so a
    # sampled file can still find its partner when the partner was not sampled.
    partners: dict[str, str] = {}
    for _, group in eligible.groupby(["vendor_id", "session_id", "interaction_id"], sort=False):
        if len(group) != 2:
            continue
        rows = list(group.itertuples(index=False))
        partners[rows[0].file_id] = rows[1].source_relbase
        partners[rows[1].file_id] = rows[0].source_relbase

    parts = []
    if per_vendor is not None:
        for vendor in sorted(set(vendors)):
            subset = eligible.loc[eligible.vendor_id.eq(vendor)]
            parts.append(subset.sample(n=min(per_vendor, len(subset)), random_state=seed))
    else:
        for label in ("naturalistic", "improvised"):
            subset = eligible.loc[eligible.label.eq(label)]
            parts.append(subset.sample(n=min(per_label or 0, len(subset)), random_state=seed))
    if include_rasters:
        # Whole rasters rather than a share of them. Some are tiny -- 20 files of
        # 2180x3840, 28 of 1080x960 -- so a random draw over the corpus misses
        # them entirely, and they are exactly the ones a reviewer needs to see.
        raster = (eligible.video_width.astype("Int64").astype(str) + "x"
                  + eligible.video_height.astype("Int64").astype(str))
        parts.append(eligible.loc[raster.isin(list(include_rasters))])
    sample = (
        pd.concat(parts, ignore_index=True)
        .drop_duplicates(subset="file_id")
        .sort_values("file_id")
        .reset_index(drop=True)
    )
    sample["partner_source_relbase"] = sample.file_id.map(partners).fillna("")
    sample["sample_index"] = np.arange(len(sample))
    return sample


def _fk_all_frames(
    model: Any,
    pose: dict[str, np.ndarray],
    batch_size: int,
    *,
    width: int,
    height: int,
    vertex_stride: int = 1,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Camera-frame joints, root-relative joints, and the per-frame worst vertex.

    The mesh is reduced to one number per frame — the smallest signed distance
    from any vertex to a raster edge — inside the batch loop. Keeping the whole
    projected mesh would be 6,890 x 2 float64 per frame, about 1 GB for a
    ten-minute recording, for a quantity FM2 immediately collapses to a minimum.
    """

    frames = len(pose["smplh:global_orient"])
    camera_parts: list[np.ndarray] = []
    root_parts: list[np.ndarray] = []
    inset_parts: list[np.ndarray] = []
    for offset in range(0, frames, batch_size):
        stop = min(frames, offset + batch_size)
        real = stop - offset
        chunk: dict[str, np.ndarray] = {}
        for key, array in pose.items():
            values = array[offset:stop]
            if real < batch_size:
                values = np.concatenate(
                    (values, np.repeat(values[-1:], batch_size - real, axis=0)), axis=0
                )
            chunk[key] = values
        camera, root, vertices = forward_smplh_joints(
            model, chunk, device="cpu", include_translation=True, return_vertices=True
        )
        camera_parts.append(camera[:real])
        root_parts.append(root[:real])
        projected = project_hmr2_full_frame(
            vertices[:real, ::vertex_stride], width=width, height=height
        )
        inset_parts.append(signed_inset_px(projected, width, height).min(axis=1))
    return (
        np.concatenate(camera_parts, axis=0),
        np.concatenate(root_parts, axis=0).astype(np.float64) * 1000.0,
        np.concatenate(inset_parts, axis=0),
    )


def _file_size(path: Path) -> int:
    try:
        return int(path.stat().st_size)
    except OSError:
        return -1


def process_file(
    row: Any, source_root: Path, model: Any, settings: dict[str, Any]
) -> dict[str, Any]:
    record: dict[str, Any] = {
        "file_id": row.file_id,
        "vendor_id": row.vendor_id,
        "source_relbase": row.source_relbase,
        "partner_source_relbase": row.partner_source_relbase,
        "label": row.label,
        "split": row.split,
        "interaction_type": row.interaction_type,
        "session_id": row.session_id,
        "interaction_id": row.interaction_id,
        "participant_id": row.participant_id,
        "observed_duration_s": float(row.observed_duration_s),
        "width": int(row.video_width),
        "height": int(row.video_height),
    }
    base = source_root / row.source_relbase

    # How the stored raster differs from the one worth looking at, and whether
    # the released arrays line up with the video at all. Round 6: V01 stores
    # three anamorphic rasters, V03 stores two quarter-turned ones, and a handful
    # of files carry an annotation grid the container does not match. None of
    # these are visible to the three detectors, so they are measured here.
    sample_aspect = probe_sample_aspect(base.with_suffix(".mp4"))
    container_frames = (
        int(row.video_nb_frames) if pd.notna(getattr(row, "video_nb_frames", None)) else -1
    )
    try:
        nominal_fps = float(Fraction(str(row.video_r_frame_rate)))
    except (ValueError, ZeroDivisionError, TypeError):
        nominal_fps = float("nan")
    wav_bytes = _file_size(base.with_suffix(".wav"))
    # Audio, new in Round 8. One WAV read per file; the isolation term also
    # reads the two VAD sidecars, which are small JSON.
    record.update(audio_quality(base.with_suffix(".wav")))
    partner_base = (
        source_root / row.partner_source_relbase if row.partner_source_relbase else None
    )
    own_vad = vad_intervals(base.with_suffix(".json"))
    # Seconds of released speech. Separates "no voice at all" from "speaks
    # rarely", which no whole-file level can: see Thresholds.audio_dead_by_vendor.
    record["audio_vad_seconds"] = float(sum(end - start for start, end in own_vad))
    record["audio_voice_isolation_db"] = (
        voice_isolation_db(
            base.with_suffix(".wav"),
            own_vad,
            vad_intervals(partner_base.with_suffix(".json")),
        )
        if partner_base is not None else float("nan")
    )
    record.update(
        {
            "geom_sample_aspect": sample_aspect,
            "container_frames": container_frames,
            "nominal_fps": nominal_fps,
            # 183 eligible files ship a 58-byte WAV: a RIFF header and no audio.
            "audio_bytes": wav_bytes,
            "audio_empty": bool(wav_bytes <= 1024),
        }
    )
    try:
        fps = float(Fraction(str(row.video_avg_frame_rate)))
    except (ValueError, ZeroDivisionError, TypeError):
        fps = float("nan")
    record["fps"] = fps
    if not np.isfinite(fps) or fps <= 0:
        record["status"] = "unusable_fps"
        return record

    try:
        with np.load(base.with_suffix(".npz"), allow_pickle=False) as archive:
            names = set(archive.files)
            missing = [k for k in (*SMPLH_KEYS, "boxes_and_keypoints:keypoints") if k not in names]
            if missing:
                record["status"] = "missing_arrays:" + ",".join(missing)
                return record
            keypoints = np.asarray(archive["boxes_and_keypoints:keypoints"])
            boxes = (
                np.asarray(archive["boxes_and_keypoints:box"])
                if "boxes_and_keypoints:box" in names else None
            )
            smplh_mask, _ = (
                as_binary_mask(archive["smplh:is_valid"]) if "smplh:is_valid" in names else (None, "")
            )
            box_mask, _ = (
                as_binary_mask(archive["boxes_and_keypoints:is_valid_box"])
                if "boxes_and_keypoints:is_valid_box" in names else (None, "")
            )
            pose = {key: np.asarray(archive[key], dtype=np.float32) for key in SMPLH_KEYS}
    except (OSError, ValueError, KeyError) as exc:
        record["status"] = f"npz_error:{type(exc).__name__}"
        return record

    frames = min(len(keypoints), *(len(a) for a in pose.values()))
    if frames == 0:
        # Present, probeable, real video duration -- and zero-length annotation
        # arrays. The inventory's all_modalities_present only checks that the
        # sibling files exist, not that they hold anything. Seen in V01.
        record["status"] = "empty_annotations"
        return record
    cap = int(settings["max_frames"])
    if frames > cap:
        # Only the longest tail of files is affected; the cap is recorded so a
        # truncated measurement is never mistaken for a whole-file one.
        record["frames_truncated_from"] = int(frames)
        frames = cap
    record["frames_analyzed"] = int(frames)
    annotation_frames = int(len(keypoints))
    timebase = AnnotationTimebase(
        nominal_fps=nominal_fps if np.isfinite(nominal_fps) and nominal_fps > 0 else fps,
        annotation_frames=annotation_frames,
        container_frames=container_frames if container_frames > 0 else annotation_frames,
    )
    roll = body_roll_deg(keypoints)
    repair = RasterRepair(
        width=int(record["width"]), height=int(record["height"]),
        sample_aspect=sample_aspect, quarter_turns=quarter_turns_from_roll(roll),
    )
    record.update(
        {
            "geom_body_roll_deg": roll,
            "geom_quarter_turns": int(repair.quarter_turns),
            "geom_anamorphic": bool(repair.is_anamorphic),
            "geom_rotated": bool(repair.is_rotated),
            "geom_display_raster": "{}x{}".format(*repair.display_size),
            "annotation_frames": annotation_frames,
            "timebase_index_drift_s": float(timebase.index_drift_s),
            "timebase_consistent": bool(timebase.is_consistent),
        }
    )
    keypoints = keypoints[:frames]
    pose = {key: array[:frames] for key, array in pose.items()}
    smplh_slice = smplh_mask[:frames] if smplh_mask is not None else None
    box_slice = box_mask[:frames] if box_mask is not None else None
    record["valid_frac_smplh"] = (
        float(np.asarray(smplh_slice, bool).mean()) if smplh_slice is not None else None
    )
    record["valid_frac_box"] = (
        float(np.asarray(box_slice, bool).mean()) if box_slice is not None else None
    )

    width, height = record["width"], record["height"]
    try:
        camera_joints, root_relative_mm, vertex_inset = _fk_all_frames(
            model, pose, int(settings["batch_size"]),
            width=width, height=height,
            vertex_stride=int(settings.get("vertex_stride", 1)),
        )
    except Exception as exc:
        record["status"] = f"fk_error:{type(exc).__name__}"
        return record

    projected = project_hmr2_full_frame(camera_joints, width=width, height=height)

    # FM1 primary is now the FK posture geometry; the 2D ratios are recorded for
    # the distributions but no longer decide anything.
    record.update(sitting_measures_2d(keypoints))
    record.update(posture_angles(root_relative_mm, frame_valid=smplh_slice))
    record.update(
        in_frame_violations(
            projected, width=width, height=height, smplh_valid=smplh_slice,
            vertex_worst_inset_px=vertex_inset, keypoints=keypoints,
            margin_px=float(settings["in_frame_margin_px"]),
            joint_inset_px=float(settings.get("smplh_joint_inset_px", 0.0)),
            min_violation_run_frames=int(settings.get("min_violation_run_frames", 1)),
        )
    )
    record.update(smplh_invalid_runs(smplh_slice, frames))
    record.update(leg_visibility(keypoints, width=width, height=height))
    # The direct form of "did the SMPL parameters stay where they should be",
    # which is what FM2 is now actually gating on via the released validity flag.
    record.update(
        smplh_reprojection_error(projected, keypoints, smplh_valid=smplh_slice)
    )
    record["fm2_windowed_pass_frac"] = windowed_in_frame_pass_frac(
        projected, width=width, height=height, smplh_valid=smplh_slice,
        window_frames=int(round(float(settings["window_s"]) * fps)),
        margin_px=float(settings["in_frame_margin_px"]),
    )
    record.update(
        static_hand_measure(
            keypoints,
            radius_shoulder_widths=float(settings["static_radius_shoulder_widths"]),
            extra_radii=tuple(float(r) for r in settings["static_extra_radii"]),
        )
    )
    if boxes is not None:
        record.update(tracker_discontinuity(boxes[:frames], box_slice))

    record["status"] = "ok"
    return record


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/v00_fm_detect.yaml"))
    parser.add_argument("--task-index", type=int, default=0)
    parser.add_argument("--task-count", type=int)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()

    worktree = _worktree()
    config = yaml.safe_load((worktree / args.config).read_text(encoding="utf-8"))
    source_root = (worktree / str(config["source_root"]).lstrip("./")).resolve()
    inventory = worktree / str(config["inventory"]).lstrip("./")
    shard_dir = worktree / str(config["outputs"]["root"]).lstrip("./") / str(config["outputs"]["shards"])
    shard_dir.mkdir(parents=True, exist_ok=True)
    task_count = args.task_count or int(config["task_count"])
    settings = dict(config["measurement"])

    sampling = dict(config["sampling"])
    sample = build_sample(
        inventory,
        per_label=sampling.get("per_label"),
        per_vendor=sampling.get("per_vendor"),
        vendors=tuple(str(v) for v in sampling.get("vendors", ("00",))),
        include_rasters=tuple(str(r) for r in sampling.get("include_rasters", ())),
        seed=int(sampling["seed"]),
    )
    shard = sample.loc[sample.sample_index % task_count == args.task_index].copy()
    if args.limit is not None:
        shard = shard.iloc[: args.limit]

    model = create_neutral_smplh(
        worktree / str(config["smplh"]["model_root"]).lstrip("./"),
        flat_hand_mean=bool(config["smplh"]["flat_hand_mean"]),
        batch_size=int(settings["batch_size"]),
        device="cpu",
    )
    started = time.perf_counter()
    rows = [process_file(row, source_root, model, settings) for row in shard.itertuples(index=False)]
    elapsed = time.perf_counter() - started

    frame = pd.DataFrame(rows)
    path = shard_dir / f"task_{args.task_index:04d}.parquet"
    temporary = path.with_name(f".{path.name}.tmp")
    frame.to_parquet(temporary, index=False)
    temporary.replace(path)
    marker = {
        "status": "complete", "task_index": args.task_index, "task_count": task_count,
        "rows": len(frame), "wall_s": elapsed,
        "files_per_s": len(frame) / elapsed if elapsed > 0 else None,
        **_git_provenance(worktree),
    }
    (shard_dir / f"task_{args.task_index:04d}.complete.json").write_text(
        json.dumps(marker, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(marker, sort_keys=True))


if __name__ == "__main__":
    main()

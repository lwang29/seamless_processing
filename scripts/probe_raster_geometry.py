#!/usr/bin/env python3
"""Which odd rasters are anamorphic, which are rotated, and is the SMPL-H fit stuck to it?

The reviewer noticed that a set of V01 clips look horizontally stretched, and
that nearly all of them trip FM1 without anyone sitting down. The container says
why: every V01 raster declares a 9:16 display aspect, but only 1080x1920 stores
square pixels. The rest are stored wide and meant to be squeezed on playback.

The question that decides what we can do about it is not whether the *video* can
be un-stretched -- it plainly can -- but whether the *released SMPL-H* can. The
HMR camera is isotropic, so a single 3D body cannot project to an anisotropically
stretched 2D person unless the pose itself absorbed the stretch. Two shape-only
Procrustes fits settle it:

    A: projected SMPL-H  vs  released keypoints as stored
    B: projected SMPL-H  vs  released keypoints with the pixel aspect applied

Similarity alignment (translation + one isotropic scale, no rotation) is used so
that neither camera centring nor overall scale can flatter either side; only the
*shape* is compared. A << B means the released pose is bound to the stretched
image and un-stretching the video does not rescue the annotation.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from fractions import Fraction
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from scripts.smplh_fk import (  # noqa: E402
    COCO_BODY23_TO_SMPLH,
    create_neutral_smplh,
    forward_smplh_joints,
    project_hmr2_full_frame,
)
from seamless_curation.m3_features import usable_keypoint_mask  # noqa: E402

SMPLH_KEYS = (
    "smplh:global_orient", "smplh:body_pose", "smplh:left_hand_pose",
    "smplh:right_hand_pose", "smplh:translation",
)
# Body-17 only. M-4 validated the correspondence for these and no others, and
# the face and hand points would dominate a shape fit without adding evidence.
BODY17 = np.arange(5, 17)
SHOULDERS, HIPS = (5, 6), (11, 12)
FRAME_STRIDE = 15


def probe_sample_aspect(video_path: Path) -> tuple[float, str, str]:
    """Pixel aspect ratio the container asks the player to apply."""
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
         "stream=sample_aspect_ratio,display_aspect_ratio", "-of", "csv=p=0:nk=1",
         str(video_path)],
        check=False, capture_output=True, text=True,
    )
    # `csv=nk=1` puts both entries on one comma-separated line, and each entry is
    # itself colon-separated, so the split order matters.
    fields = [f.strip() for f in result.stdout.strip().replace("\n", ",").split(",") if f.strip()]
    if not fields:
        return 1.0, "", ""
    sar_text = fields[0].replace(":", "/")
    dar_text = fields[1] if len(fields) > 1 else ""
    try:
        sar = float(Fraction(sar_text))
    except (ValueError, ZeroDivisionError):
        sar = 1.0
    return (sar if sar > 0 else 1.0), fields[0], dar_text


def body_roll_deg(points: np.ndarray, usable: np.ndarray) -> float:
    ok = usable[:, SHOULDERS[0]] & usable[:, SHOULDERS[1]] & usable[:, HIPS[0]] & usable[:, HIPS[1]]
    if not ok.any():
        return float("nan")
    shoulder = 0.5 * (points[:, SHOULDERS[0], :2] + points[:, SHOULDERS[1], :2])
    hip = 0.5 * (points[:, HIPS[0], :2] + points[:, HIPS[1], :2])
    axis = (hip - shoulder)[ok]
    return float(np.median(np.degrees(np.arctan2(axis[:, 0], axis[:, 1]))))


def anatomy_ratio(points: np.ndarray, usable: np.ndarray, x_scale: float = 1.0) -> float:
    """Shoulder width over torso length, in image pixels.

    A posture-robust read on horizontal stretch: both distances live on the same
    rigid torso, so their ratio is fixed by anatomy up to how the person is
    turned, and an anisotropic raster inflates the numerator only.
    """
    ok = usable[:, SHOULDERS[0]] & usable[:, SHOULDERS[1]] & usable[:, HIPS[0]] & usable[:, HIPS[1]]
    if not ok.any():
        return float("nan")
    scaled = points[..., :2].copy()
    scaled[..., 0] *= x_scale
    width = np.linalg.norm(scaled[:, SHOULDERS[0]] - scaled[:, SHOULDERS[1]], axis=-1)
    torso = np.linalg.norm(
        0.5 * (scaled[:, SHOULDERS[0]] + scaled[:, SHOULDERS[1]])
        - 0.5 * (scaled[:, HIPS[0]] + scaled[:, HIPS[1]]), axis=-1
    )
    with np.errstate(invalid="ignore", divide="ignore"):
        ratio = width[ok] / np.where(torso[ok] > 1e-6, torso[ok], np.nan)
    return float(np.nanmedian(ratio))


def similarity_residual(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Per-frame shape mismatch after the best translation + isotropic scale.

    Rotation is deliberately *not* fitted: an image-plane rotation is exactly the
    quarter turn some V03 files carry, and absorbing it would hide the thing
    another part of this probe is measuring.
    """
    residuals = np.full(len(source), np.nan)
    for index, (a, b) in enumerate(zip(source, target)):
        finite = np.isfinite(a).all(axis=-1) & np.isfinite(b).all(axis=-1)
        if finite.sum() < 6:
            continue
        p, q = a[finite], b[finite]
        p = p - p.mean(axis=0)
        q = q - q.mean(axis=0)
        norm = float((p * p).sum())
        if norm < 1e-9:
            continue
        scale = float((p * q).sum()) / norm
        scaled_target_norm = np.sqrt((q * q).sum() / len(q))
        if scaled_target_norm < 1e-9:
            continue
        residuals[index] = float(
            np.sqrt(((scale * p - q) ** 2).sum(axis=-1).mean())
        ) / scaled_target_norm
    return residuals


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", type=int, default=0)
    parser.add_argument("--tasks", type=int, default=1)
    parser.add_argument("--per-raster", type=int, default=60)
    parser.add_argument("--seed", type=int, default=20260924)
    parser.add_argument("--out", type=Path, default=ROOT / "outputs/session3/raster_geometry")
    args = parser.parse_args()

    inventory = pd.read_parquet(ROOT / "outputs/02_inventory/summary/inventory_joined.parquet")
    inventory["raster"] = (inventory.video_width.astype("Int64").astype(str) + "x"
                           + inventory.video_height.astype("Int64").astype(str))
    eligible = inventory.loc[
        inventory.all_modalities_present.astype(bool)
        & inventory.probe_status.eq("ok")
        & inventory.video_stream_present.astype(bool)
        & inventory.observed_duration_s.ge(30.0)
        & inventory.interaction_type.ne("charades")
    ]
    pool = (
        eligible.groupby(["vendor_id", "raster"], group_keys=False)
        .apply(lambda g: g.sample(n=min(len(g), args.per_raster), random_state=args.seed), include_groups=True)
        .sort_values("file_id")
        .iloc[args.task::args.tasks]
    )

    model = create_neutral_smplh(ROOT / "model_files", flat_hand_mean=True, batch_size=256)
    rows = []
    for row in pool.itertuples(index=False):
        base = ROOT / "seamless_interaction" / row.source_relbase
        record = {"file_id": row.file_id, "vendor_id": row.vendor_id, "raster": row.raster,
                  "width": int(row.video_width), "height": int(row.video_height)}
        sar, sar_text, dar_text = probe_sample_aspect(base.with_suffix(".mp4"))
        record.update({"sar": sar, "sar_text": sar_text, "dar_text": dar_text})
        try:
            with np.load(base.with_suffix(".npz"), allow_pickle=False) as archive:
                keypoints = np.asarray(archive["boxes_and_keypoints:keypoints"])[::FRAME_STRIDE]
                payload = {k: np.asarray(archive[k])[::FRAME_STRIDE] for k in SMPLH_KEYS}
                valid = np.asarray(archive["smplh:is_valid"])[::FRAME_STRIDE].astype(bool)
        except (OSError, ValueError, KeyError) as error:
            rows.append({**record, "status": f"read_error:{type(error).__name__}"})
            continue
        if len(keypoints) == 0 or keypoints.shape[1] < 17:
            rows.append({**record, "status": "empty_annotations"})
            continue

        usable = usable_keypoint_mask(keypoints)
        record["roll_deg"] = body_roll_deg(keypoints, usable)
        record["ratio_stored"] = anatomy_ratio(keypoints, usable)
        record["ratio_sar_applied"] = anatomy_ratio(keypoints, usable, x_scale=sar)
        record["frames"] = int(len(keypoints))
        record["valid_frac"] = float(valid.mean())

        try:
            joints_camera, _ = forward_smplh_joints(model, payload)
            projected = project_hmr2_full_frame(
                joints_camera, int(row.video_width), int(row.video_height)
            )[:, COCO_BODY23_TO_SMPLH[BODY17]]
        except (RuntimeError, ValueError) as error:
            rows.append({**record, "status": f"fk_error:{type(error).__name__}"})
            continue

        target = keypoints[:, BODY17, :2].astype(np.float64).copy()
        target[~usable[:, BODY17]] = np.nan
        corrected = target.copy()
        corrected[..., 0] *= sar
        keep = valid if valid.any() else np.ones(len(target), dtype=bool)
        record["shape_residual_stored"] = float(
            np.nanmedian(similarity_residual(projected[keep], target[keep]))
        )
        record["shape_residual_sar_applied"] = float(
            np.nanmedian(similarity_residual(projected[keep], corrected[keep]))
        )
        rows.append({**record, "status": "ok"})

    args.out.mkdir(parents=True, exist_ok=True)
    destination = args.out / f"part_{args.task:03d}.parquet"
    pd.DataFrame(rows).to_parquet(destination, index=False)
    print(f"task {args.task}: {len(rows)} files -> {destination}")


if __name__ == "__main__":
    main()

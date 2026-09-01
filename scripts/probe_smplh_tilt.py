#!/usr/bin/env python3
"""Why do visibly-upright participants read as leaning with bent knees in 3D?

Two symptoms, tested separately because they can have different causes:

  TILT  — the whole body leans forward in Panel C's side view. Panel C plots
          CAMERA-frame coordinates, so a camera that is not level renders an
          upright person as leaning. That would be a visualization choice, not a
          pose error. Discriminator: a camera effect is constant within a
          recording session; a real lean is not.

  KNEES — the knee angle is a pure joint angle, invariant to every global
          rotation, so no plotting choice can create or remove it. If it reads
          bent, the released parameters say bent. The question is whether the
          image constrains it at all.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("/sailhome/lw29/seamless_processing")
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from scripts.smplh_fk import (  # noqa: E402
    CameraHypothesis, create_neutral_smplh, forward_smplh_joints, project_hmr2_full_frame,
)
from seamless_curation.features import as_binary_mask  # noqa: E402
from seamless_curation.m3_features import usable_keypoint_mask  # noqa: E402

PELVIS, NECK = 0, 12
HIP = {"L": 1, "R": 2}
KNEE = {"L": 4, "R": 5}
ANKLE = {"L": 7, "R": 8}
FOOT = {"L": 10, "R": 11}
# COCO body index -> SMPL-H joint, for the twelve landmarks M-4 validated.
UPPER = [(5, 16), (6, 17), (7, 18), (8, 19), (9, 20), (10, 21)]   # shoulders/elbows/wrists
LOWER = [(11, 1), (12, 2), (13, 4), (14, 5), (15, 7), (16, 8)]    # hips/knees/ankles

KEYS = ("smplh:global_orient", "smplh:body_pose", "smplh:left_hand_pose",
        "smplh:right_hand_pose", "smplh:translation")


def angle(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    na, nb = np.linalg.norm(a, axis=-1), np.linalg.norm(b, axis=-1)
    with np.errstate(invalid="ignore", divide="ignore"):
        c = np.sum(a * b, axis=-1) / (na * nb)
    return np.degrees(np.arccos(np.clip(c, -1, 1)))


def fk(model, pose, stride, width, height, zero_knees=False):
    n = len(pose["smplh:global_orient"])
    idx = np.arange(0, n, stride)
    sub = {k: np.ascontiguousarray(v[idx]) for k, v in pose.items()}
    if zero_knees:
        body = sub["smplh:body_pose"].reshape(len(idx), -1).copy()
        # body_pose holds joints 1..21; joint j lives at [(j-1)*3 : j*3].
        for joint in (4, 5):
            body[:, (joint - 1) * 3: joint * 3] = 0.0
        sub["smplh:body_pose"] = body
    out, batch = [], 256
    m = len(idx)
    for off in range(0, m, batch):
        stop = min(m, off + batch)
        real = stop - off
        chunk = {k: v[off:stop] for k, v in sub.items()}
        if real < batch:
            chunk = {k: np.concatenate((v, np.repeat(v[-1:], batch - real, 0)), 0)
                     for k, v in chunk.items()}
        cam, _ = forward_smplh_joints(model, chunk, device="cpu", include_translation=True)
        out.append(cam[:real])
    cam = np.concatenate(out, 0).astype(np.float64)
    return cam, project_hmr2_full_frame(cam, width=width, height=height), idx


def reproj(projected, keypoints, pairs, usable, scale):
    errs = []
    for coco, smplh in pairs:
        d = np.linalg.norm(projected[:, smplh, :] - keypoints[:, coco, :2], axis=-1)
        errs.append(np.where(usable[:, coco], d, np.nan))
    with np.errstate(invalid="ignore"):
        return float(np.nanmedian(np.nanmedian(np.stack(errs, 1), axis=1))) / scale


def main() -> None:
    task = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    tasks = int(sys.argv[2]) if len(sys.argv) > 2 else 1
    limit = int(sys.argv[3]) if len(sys.argv) > 3 else 130
    stride = 10

    scan = pd.read_parquet(ROOT / "outputs/session2/v00_fm_briefing/scan_with_flags.parquet")
    # Only files the pipeline keeps and that FM1 says are standing: the case the
    # reviewer is asking about.
    pool = scan[scan.passes_all & scan.fm1_sitting.ne(True)].sort_values("file_id")
    pool = pool.iloc[task::tasks].head(limit)
    model = create_neutral_smplh(ROOT / "model_files", flat_hand_mean=True,
                                 batch_size=256, device="cpu")
    cam_hyp = CameraHypothesis()
    rows = []
    for row in pool.itertuples(index=False):
        base = ROOT / "seamless_interaction" / row.source_relbase
        try:
            with np.load(base.with_suffix(".npz"), allow_pickle=False) as a:
                if any(k not in a.files for k in KEYS):
                    continue
                pose = {k: np.asarray(a[k], dtype=np.float32) for k in KEYS}
                kps = np.asarray(a["boxes_and_keypoints:keypoints"])
                valid, _ = as_binary_mask(a["smplh:is_valid"])
        except (OSError, ValueError, KeyError):
            continue
        W, H = int(row.width), int(row.height)
        cam, proj, idx = fk(model, pose, stride, W, H)
        kps_s = kps[idx]
        usable = usable_keypoint_mask(kps_s)
        ok = np.asarray(valid[idx], bool) if valid is not None else np.ones(len(idx), bool)
        if ok.sum() < 10:
            continue
        cam, proj, kps_s, usable = cam[ok], proj[ok], kps_s[ok], usable[ok]

        up = cam[:, NECK] - cam[:, PELVIS]           # body axis, camera frame
        up /= np.linalg.norm(up, axis=-1, keepdims=True)
        # Pitch of the body axis out of the image plane. 0 = the person's up-axis
        # lies in the image plane, which is what a level camera would give for a
        # vertical person. Positive = the head is further from the camera.
        pitch = np.degrees(np.arcsin(np.clip(up[:, 2], -1, 1)))
        # In-image lean, left/right: this one the video would show directly.
        roll = np.degrees(np.arctan2(up[:, 0], -up[:, 1]))

        knee = np.nanmean([angle(cam[:, HIP[s]] - cam[:, KNEE[s]],
                                 cam[:, ANKLE[s]] - cam[:, KNEE[s]]) for s in "LR"], 0)
        # Foot relative to the body axis. A flat foot under an upright person is
        # roughly perpendicular to the leg.
        foot = np.nanmean([angle(cam[:, FOOT[s]] - cam[:, ANKLE[s]], up) for s in "LR"], 0)

        sw = np.linalg.norm(kps_s[:, 5, :2] - kps_s[:, 6, :2], axis=1)
        scale = float(np.median(sw[sw > 1e-6])) if (sw > 1e-6).any() else np.nan

        record = {
            "file_id": row.file_id, "session_id": row.session_id,
            "participant_id": row.participant_id, "label": row.label,
            "frames": int(len(cam)),
            "pitch_deg": float(np.median(pitch)),
            "pitch_sd_deg": float(np.std(pitch)),
            "roll_deg": float(np.median(roll)),
            "knee_deg": float(np.nanmedian(knee)),
            "foot_vs_body_deg": float(np.nanmedian(foot)),
            "reproj_upper_sw": reproj(proj, kps_s, UPPER, usable, scale),
            "reproj_lower_sw": reproj(proj, kps_s, LOWER, usable, scale),
        }
        # The decisive test: force both knees straight and reproject. If the image
        # never constrained the bend, the error will barely move.
        _, proj0, _ = fk(model, pose, stride, W, H, zero_knees=True)
        proj0 = proj0[ok]
        record["reproj_lower_sw_straight"] = reproj(proj0, kps_s, LOWER, usable, scale)
        record["reproj_upper_sw_straight"] = reproj(proj0, kps_s, UPPER, usable, scale)
        record["focal_px"] = cam_hyp.focal_pixels(W, H)
        record["fov_deg"] = 2 * np.degrees(np.arctan(H / 2 / cam_hyp.focal_pixels(W, H)))
        record["median_depth_m"] = float(np.median(cam[:, PELVIS, 2]))
        rows.append(record)

    out = ROOT / f"outputs/session2/tilt_probe/part_{task:02d}.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(out, index=False)
    print(f"task {task}: {len(rows)} files -> {out}")


if __name__ == "__main__":
    main()

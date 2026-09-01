#!/usr/bin/env python3
"""Side-by-side: the video frame, Panel C as it renders now, and two corrections.

Panel C plots camera-frame coordinates, so a camera that is not level draws an
upright person leaning. This renders four columns for the same frame so the
question "rendering or pose?" can be answered by eye:

  1. the video frame itself
  2. side view in camera frame, exactly what Panel C shows today
  3. side view after removing only the per-file median body tilt — the correction
     a settings change could make
  4. side view after removing the tilt AND straightening both knees, to show how
     much of the remaining bend is articulation rather than orientation
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

ROOT = Path("/sailhome/lw29/seamless_processing")
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from scripts.smplh_fk import create_neutral_smplh, forward_smplh_joints  # noqa: E402
from seamless_curation.features import as_binary_mask  # noqa: E402

PELVIS, NECK = 0, 12
EDGES = [(0, 1), (0, 2), (1, 4), (2, 5), (4, 7), (5, 8), (7, 10), (8, 11),
         (0, 3), (3, 6), (6, 9), (9, 12), (12, 15), (9, 13), (9, 14),
         (13, 16), (14, 17), (16, 18), (17, 19), (18, 20), (19, 21)]
KEYS = ("smplh:global_orient", "smplh:body_pose", "smplh:left_hand_pose",
        "smplh:right_hand_pose", "smplh:translation")
SIZE = 460


def draw(joints, title, colour=(70, 220, 255)):
    canvas = np.full((SIZE, SIZE, 3), 16, np.uint8)
    pts = joints[:22, [2, 1]]                      # side view: depth, vertical
    span = np.abs(joints[:22] - joints[PELVIS]).max() * 2.4 or 1.0
    scale = SIZE * 0.40 / span
    centre = np.array([SIZE / 2, SIZE / 2 + 30])
    xy = (pts - joints[PELVIS][[2, 1]]) * scale + centre
    cv2.line(canvas, (0, int(centre[1])), (SIZE, int(centre[1])), (48, 48, 48), 1)
    for a, b in EDGES:
        p, q = xy[a].astype(int), xy[b].astype(int)
        cv2.line(canvas, tuple(p), tuple(q), colour, 2, cv2.LINE_AA)
    for p in xy.astype(int):
        cv2.circle(canvas, tuple(p), 3, colour, -1, cv2.LINE_AA)
    cv2.putText(canvas, title, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.52,
                (235, 235, 235), 1, cv2.LINE_AA)
    cv2.putText(canvas, "<- camera", (10, SIZE - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.42,
                (140, 150, 165), 1, cv2.LINE_AA)
    return canvas


def rotate_upright(joints):
    """Rotate so the pelvis-to-neck axis is vertical in the drawing plane."""

    up = joints[NECK] - joints[PELVIS]
    theta = np.arctan2(up[2], -up[1])              # tilt within the z/y plane
    c, s = np.cos(theta), np.sin(theta)
    rotation = np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
    # -y is "up" in this frame, so rotate about x to bring the axis onto it.
    turned = joints @ rotation.T
    return turned


def fk_frame(model, pose, index, zero_knees=False):
    chunk = {}
    for key, array in pose.items():
        row = np.asarray(array[index: index + 1], dtype=np.float32).reshape(1, -1)
        if zero_knees and key == "smplh:body_pose":
            row = row.copy()
            for joint in (4, 5):
                row[:, (joint - 1) * 3: joint * 3] = 0.0
        chunk[key] = np.repeat(row, 256, axis=0)
    camera, _ = forward_smplh_joints(model, chunk, device="cpu", include_translation=True)
    return camera[0].astype(np.float64)


def main() -> None:
    out_dir = ROOT / "outputs/session2/side_views"
    out_dir.mkdir(parents=True, exist_ok=True)
    scan = pd.read_parquet(ROOT / "outputs/session2/v00_fm_briefing/scan_with_flags.parquet")
    pool = scan[scan.passes_all & scan.fm1_sitting.ne(True)].sort_values("file_id")
    files = list(pool.file_id)[:: max(1, len(pool) // 4)][:4]
    model = create_neutral_smplh(ROOT / "model_files", flat_hand_mean=True,
                                 batch_size=256, device="cpu")

    panels = []
    for file_id in files:
        row = pool[pool.file_id.eq(file_id)].iloc[0]
        base = ROOT / "seamless_interaction" / row.source_relbase
        with np.load(base.with_suffix(".npz"), allow_pickle=False) as a:
            pose = {k: np.asarray(a[k], dtype=np.float32) for k in KEYS}
            valid, _ = as_binary_mask(a["smplh:is_valid"])
        index = int(np.flatnonzero(np.asarray(valid, bool))[len(valid) // 3])

        raw = fk_frame(model, pose, index)
        straight = fk_frame(model, pose, index, zero_knees=True)
        seconds = index / float(row.fps)
        grab = subprocess.run(
            ["ffmpeg", "-hide_banner", "-loglevel", "error", "-ss", f"{seconds:.3f}",
             "-i", str(base.with_suffix(".mp4")), "-frames:v", "1",
             "-vf", f"scale=-2:{SIZE}", "-f", "rawvideo", "-pix_fmt", "bgr24", "-"],
            capture_output=True, timeout=90,
        )
        width = len(grab.stdout) // (SIZE * 3)
        frame = (np.frombuffer(grab.stdout[: SIZE * width * 3], np.uint8)
                 .reshape(SIZE, width, 3).copy() if width else np.zeros((SIZE, SIZE, 3), np.uint8))
        cv2.putText(frame, "video", (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.52,
                    (235, 235, 235), 2, cv2.LINE_AA)

        strip = np.concatenate([
            frame,
            draw(raw, "Panel C today (camera frame)", (70, 220, 255)),
            draw(rotate_upright(raw), "tilt removed", (120, 235, 140)),
            draw(rotate_upright(straight), "tilt removed + knees straight", (150, 170, 255)),
        ], axis=1)
        label = np.full((30, strip.shape[1], 3), 10, np.uint8)
        cv2.putText(label, f"{file_id}   t={seconds:.1f}s", (8, 21),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 210, 225), 1, cv2.LINE_AA)
        panels.append(np.concatenate([label, strip], axis=0))

    widest = max(p.shape[1] for p in panels)
    padded = [np.pad(p, ((0, 0), (0, widest - p.shape[1]), (0, 0))) for p in panels]
    path = out_dir / "side_view_comparison.png"
    cv2.imwrite(str(path), np.concatenate(padded, axis=0))
    print(path)


if __name__ == "__main__":
    main()

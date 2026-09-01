#!/usr/bin/env python3
"""How often is the recorded video rotated 90 degrees from upright?

Spotted in a V03 landscape file that **passes** all three checks: the participant
lies sideways in the frame. The released keypoints and the SMPL-H fit both track
the rotated person correctly, in raster coordinates, so nothing in the pipeline
notices — every detector is either rotation-invariant or works in the raster
frame.

Measures the in-image angle of the shoulder-midpoint-to-hip-midpoint axis from
straight down. Near 0 is upright; near +/-90 is a quarter turn.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("/sailhome/lw29/seamless_processing")
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from seamless_curation.m3_features import usable_keypoint_mask  # noqa: E402

SHOULDERS, HIPS = (5, 6), (11, 12)


def body_roll_deg(keypoints: np.ndarray) -> float:
    points = np.asarray(keypoints, dtype=np.float64)
    if points.ndim != 3 or points.shape[1] < 17 or len(points) == 0:
        return float("nan")
    usable = usable_keypoint_mask(points)
    ok = (usable[:, SHOULDERS[0]] & usable[:, SHOULDERS[1]]
          & usable[:, HIPS[0]] & usable[:, HIPS[1]])
    if not ok.any():
        return float("nan")
    shoulder = 0.5 * (points[:, SHOULDERS[0], :2] + points[:, SHOULDERS[1], :2])
    hip = 0.5 * (points[:, HIPS[0], :2] + points[:, HIPS[1], :2])
    axis = (hip - shoulder)[ok]          # points "down the body" in image space
    # 0 = the body runs top-to-bottom of the frame; +/-90 = it runs sideways.
    return float(np.median(np.degrees(np.arctan2(axis[:, 0], axis[:, 1]))))


def main() -> None:
    task = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    tasks = int(sys.argv[2]) if len(sys.argv) > 2 else 1

    inventory = pd.read_parquet(ROOT / "outputs/02_inventory/summary/inventory_joined.parquet")
    eligible = inventory.loc[
        inventory.vendor_id.isin(["01", "02", "03"])
        & inventory.all_modalities_present.astype(bool)
        & inventory.probe_status.eq("ok")
        & inventory.video_stream_present.astype(bool)
        & inventory.observed_duration_s.ge(30.0)
        & inventory.interaction_type.ne("charades")
    ].copy()
    eligible["shape"] = np.where(
        eligible.video_width < eligible.video_height, "portrait",
        np.where(eligible.video_width > eligible.video_height, "landscape", "square"),
    )
    # Every landscape and square file, since those are the suspicious ones and
    # they are rare, plus a portrait control of comparable size.
    odd = eligible.loc[eligible["shape"].ne("portrait")]
    control = eligible.loc[eligible["shape"].eq("portrait")].sample(n=600, random_state=20260923)
    pool = pd.concat([odd, control]).sort_values("file_id").iloc[task::tasks]

    rows = []
    for row in pool.itertuples(index=False):
        base = ROOT / "seamless_interaction" / row.source_relbase
        try:
            with np.load(base.with_suffix(".npz"), allow_pickle=False) as archive:
                if "boxes_and_keypoints:keypoints" not in archive.files:
                    continue
                keypoints = np.asarray(archive["boxes_and_keypoints:keypoints"])[::30]
        except (OSError, ValueError, KeyError):
            continue
        rows.append(
            {
                "file_id": row.file_id, "vendor_id": row.vendor_id,
                "shape": row.shape,
                "raster": f"{int(row.video_width)}x{int(row.video_height)}",
                "body_roll_deg": body_roll_deg(keypoints),
            }
        )

    out = ROOT / f"outputs/session2/vendor_roll/part_{task:02d}.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(out, index=False)
    print(f"task {task}: {len(rows)} files -> {out}")


if __name__ == "__main__":
    main()

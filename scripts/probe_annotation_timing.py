#!/usr/bin/env python3
"""Do the released annotations line up, frame for frame, with the video?

Chasing `V01_S1607_I00000135_P2569`, where the reviewer saw the keypoint overlay
and the SMPL-H projection drift away from the person. That file declares a
nominal 48000/1001 fps and carries 3,261 annotated frames, but the container
holds only 2,807 — an average of 41.3 fps. Both cover the same 68.005 s, so the
annotation grid is uniform and the video's is not, and anything that pairs the
two by integer index walks off by 454 frames (9.5 s) by the end.

Reads only the zip central directory and the .npy header of each array, so it
never decompresses a keypoint block. That makes it cheap enough to run over the
entire eligible pool rather than a sample.
"""

from __future__ import annotations

import argparse
import sys
import zipfile
from fractions import Fraction
from pathlib import Path

import numpy as np
import pandas as pd
from numpy.lib import format as npy_format

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

# Every per-frame array in the bundle. They should all agree with each other and
# with the video; when they do not, which ones disagree is the diagnosis.
FRAME_KEYS = (
    "boxes_and_keypoints:keypoints",
    "boxes_and_keypoints:is_valid_box",
    "smplh:body_pose",
    "smplh:is_valid",
)


def array_lengths(npz_path: Path) -> dict[str, int]:
    """First-axis length of each per-frame array, from headers alone."""
    lengths: dict[str, int] = {}
    with zipfile.ZipFile(npz_path) as archive:
        available = {name.removesuffix(".npy"): name for name in archive.namelist()}
        for key in FRAME_KEYS:
            member = available.get(key)
            if member is None:
                continue
            with archive.open(member) as stream:
                version = npy_format.read_magic(stream)
                shape, _, _ = npy_format._read_array_header(stream, version)
            lengths[key] = int(shape[0]) if shape else 0
    return lengths


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", type=int, default=0)
    parser.add_argument("--tasks", type=int, default=1)
    parser.add_argument("--out", type=Path,
                        default=ROOT / "outputs/session3/annotation_timing")
    args = parser.parse_args()

    inventory = pd.read_parquet(ROOT / "outputs/02_inventory/summary/inventory_joined.parquet")
    eligible = inventory.loc[
        inventory.all_modalities_present.astype(bool)
        & inventory.probe_status.eq("ok")
        & inventory.video_stream_present.astype(bool)
        & inventory.observed_duration_s.ge(30.0)
        & inventory.interaction_type.ne("charades")
    ].sort_values("file_id").iloc[args.task::args.tasks]

    rows = []
    for row in eligible.itertuples(index=False):
        npz_path = ROOT / "seamless_interaction" / f"{row.source_relbase}.npz"
        try:
            lengths = array_lengths(npz_path)
        except (OSError, zipfile.BadZipFile, ValueError) as error:
            rows.append({"file_id": row.file_id, "vendor_id": row.vendor_id,
                         "status": f"read_error:{type(error).__name__}"})
            continue
        keypoint_frames = lengths.get("boxes_and_keypoints:keypoints", -1)
        nominal_fps = float(Fraction(str(row.video_r_frame_rate)))
        duration = float(row.observed_duration_s)
        container_frames = float(row.video_nb_frames) if pd.notna(row.video_nb_frames) else np.nan
        rows.append(
            {
                "file_id": row.file_id,
                "vendor_id": row.vendor_id,
                "status": "ok",
                "raster": f"{int(row.video_width)}x{int(row.video_height)}",
                "nominal_fps": nominal_fps,
                "container_frames": container_frames,
                "keypoint_frames": keypoint_frames,
                "smplh_frames": lengths.get("smplh:body_pose", -1),
                "box_frames": lengths.get("boxes_and_keypoints:is_valid_box", -1),
                "duration_s": duration,
                # What a constant-rate annotator would have produced.
                "expected_frames_at_nominal": duration * nominal_fps,
                # How far apart the two grids end up by the last frame, if you
                # pair them by integer index. This is what the reviewer saw.
                "index_drift_s": (keypoint_frames - container_frames) / nominal_fps
                                 if np.isfinite(container_frames) and nominal_fps > 0 else np.nan,
            }
        )

    args.out.mkdir(parents=True, exist_ok=True)
    destination = args.out / f"part_{args.task:03d}.parquet"
    pd.DataFrame(rows).to_parquet(destination, index=False)
    print(f"task {args.task}: {len(rows)} files -> {destination}")


if __name__ == "__main__":
    main()

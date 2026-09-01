#!/usr/bin/env python3
"""Where in the body does the pose distortion live?

A global rotation — which is what a tilted camera produces — is absorbed
entirely by ``global_orient`` and changes no body-relative quantity, so it
cannot affect a model trained on root-relative or torso-relative joints. What
*can* affect such a model is articulation: a joint bent away from rest.

This reads the released ``body_pose`` directly and reports, per joint, how far
from rest it is held. No forward kinematics and no camera are involved, so
nothing here can be a projection or rendering artefact.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("/sailhome/lw29/seamless_processing")
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from seamless_curation.features import as_binary_mask  # noqa: E402

# SMPL-H body_pose holds joints 1..21 in kinematic order.
BODY_JOINTS = {
    1: "L_hip", 2: "R_hip", 3: "spine1", 4: "L_knee", 5: "R_knee", 6: "spine2",
    7: "L_ankle", 8: "R_ankle", 9: "spine3", 10: "L_foot", 11: "R_foot",
    12: "neck", 13: "L_collar", 14: "R_collar", 15: "head", 16: "L_shoulder",
    17: "R_shoulder", 18: "L_elbow", 19: "R_elbow", 20: "L_wrist", 21: "R_wrist",
}
REGION = {
    "legs": (1, 2, 4, 5, 7, 8, 10, 11),
    "torso": (3, 6, 9),
    "head_neck": (12, 15),
    "arms": (13, 14, 16, 17, 18, 19, 20, 21),
}


def main() -> None:
    task = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    tasks = int(sys.argv[2]) if len(sys.argv) > 2 else 1
    limit = int(sys.argv[3]) if len(sys.argv) > 3 else 60

    scan = pd.read_parquet(ROOT / "outputs/session2/v00_fm_briefing/scan_with_flags.parquet")
    pool = scan[scan.passes_all & scan.fm1_sitting.ne(True)].sort_values("file_id")
    pool = pool.iloc[task::tasks].head(limit)

    rows = []
    for row in pool.itertuples(index=False):
        base = ROOT / "seamless_interaction" / row.source_relbase
        try:
            with np.load(base.with_suffix(".npz"), allow_pickle=False) as a:
                body = np.asarray(a["smplh:body_pose"], dtype=np.float64)
                valid, _ = as_binary_mask(a["smplh:is_valid"])
        except (OSError, ValueError, KeyError):
            continue
        n = len(body)
        body = body.reshape(n, -1, 3)
        if body.shape[1] < 21:
            continue
        ok = np.asarray(valid[:n], bool) if valid is not None else np.ones(n, bool)
        if ok.sum() < 30:
            continue
        body = body[ok][::5]
        # Axis-angle magnitude = how far this joint is rotated from rest.
        magnitude = np.degrees(np.linalg.norm(body, axis=-1))

        record = {
            "file_id": row.file_id, "session_id": row.session_id,
            "participant_id": row.participant_id, "label": row.label,
            "frames": int(len(body)),
        }
        for index, name in BODY_JOINTS.items():
            series = magnitude[:, index - 1]
            record[f"{name}_deg_p50"] = float(np.median(series))
            # How much this joint *moves* over the recording, as opposed to how
            # far it is held from rest. A constant offset is learnable; a
            # fluctuating one is noise.
            record[f"{name}_deg_sd"] = float(np.std(series))
        for region, members in REGION.items():
            record[f"{region}_deg_p50"] = float(
                np.median(magnitude[:, [m - 1 for m in members]])
            )
        rows.append(record)

    out = ROOT / f"outputs/session2/joint_bend/part_{task:02d}.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(out, index=False)
    print(f"task {task}: {len(rows)} files -> {out}")


if __name__ == "__main__":
    main()

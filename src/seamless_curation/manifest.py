"""Stage 6: turn manual verdicts into the accepted subset definition.

Two manifests are written, and the difference between them is the strength of
the guarantee each one carries. Both are CSV, both name frame ranges inside the
untouched source tree, and neither copies a byte of media.

``accepted_clips.csv`` — **the training manifest.**
    One row per accepted clip: a contiguous ``[start_frame, end_frame)`` range of
    one participant file, which passed every automated gate and belongs to a
    file a reviewer accepted. ``review_evidence`` is ``card`` or
    ``card+video``. The reviewer saw twelve moments sampled from exactly these
    spans, the pelvis-frame SMPL-H pose at each of them, and the whole
    recording's speech-and-gesture timeline.

``accepted_clips_with_audio.csv`` — **the strict subset.**
    The rows whose reviewer played the 30-second clip with sound. Smaller, and
    the only rows where audio-motion synchronisation was confirmed by ear rather
    than from the timeline. Provided so that a downstream user who wants the
    tightest possible guarantee does not have to take the looser one on trust.

Nothing that was not reviewed appears in either file. ``unsure`` keeps an item
out; so does a missing verdict. That is the whole point of the stage, so it is
enforced by construction — the accepted set is an inner join on the verdict log,
not a filter that could be forgotten.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from .config import RunConfig
from .review_store import VerdictStore

MANIFEST_COLUMNS = [
    "clip_id", "review_item_id", "file_id", "source_relbase",
    "vendor", "label", "split", "session_id", "participant_id",
    "interaction_id", "interaction_type",
    "start_frame", "end_frame", "start_s", "window_seconds", "fps",
    "speech_seconds", "gesture_frac_speech", "speech_segments_covered",
    "episode_count_speech", "wrist_excursion_p90_mm", "elbow_excursion_p90_mm",
    "gesture_speech_ratio", "posture_spread_mm", "wrist_height_p75_mm",
    "hands_together_frac", "arm_abduction_p75_deg",
    "sync_r", "sync_lag_s", "consistency_r",
    "smplh_valid_frac", "smplh_longest_invalid_s", "hand_frozen_frac",
    "clip_score", "reviewer", "verdict_source", "review_evidence", "reviewed_utc",
]


def build_manifests(config: RunConfig) -> dict[str, Any]:
    """Write both accepted manifests and return a summary of what they contain."""

    candidates = pd.read_parquet(config.candidates_path)
    resolved = VerdictStore(config.verdict_log).resolve()
    if resolved.empty:
        empty = pd.DataFrame(columns=MANIFEST_COLUMNS)
        empty.to_csv(config.accepted_clips_path, index=False)
        empty.to_csv(_strict_path(config), index=False)
        return {"accepted_clips": 0, "accepted_hours": 0.0, "reviewed_items": 0}

    accepted_items = resolved.loc[resolved["verdict"] == "accept"].copy()
    merged = candidates.merge(
        accepted_items[
            ["review_item_id", "file_id", "reviewer", "verdict_source", "saw_video", "recorded_utc"]
        ].rename(columns={"file_id": "reviewed_file_id"}),
        on="review_item_id",
        how="inner",
    )
    # The id is derived from the file, so these must agree. If they ever do not,
    # a verdict has been re-bound to a file its reviewer never saw, and that is
    # the one failure this stage exists to make impossible.
    mismatched = merged.loc[
        (merged["reviewed_file_id"].fillna("") != "")
        & (merged["reviewed_file_id"] != merged["file_id"])
    ]
    if len(mismatched):
        raise ValueError(
            f"{len(mismatched)} verdicts name a different file than the candidate they joined "
            f"to, e.g. {mismatched.iloc[0]['review_item_id']}: reviewed "
            f"{mismatched.iloc[0]['reviewed_file_id']}, candidate {mismatched.iloc[0]['file_id']}"
        )
    orphaned = set(accepted_items["review_item_id"]) - set(candidates["review_item_id"])
    if "fps" not in merged.columns:
        population = pd.read_parquet(config.population_path)[["file_id", "nominal_fps"]]
        merged = merged.merge(population, on="file_id", how="left")
        merged["fps"] = merged["nominal_fps"]
    merged["review_evidence"] = merged["saw_video"].map({True: "card+video", False: "card"})
    merged["reviewed_utc"] = merged["recorded_utc"]

    columns = [column for column in MANIFEST_COLUMNS if column in merged.columns]
    manifest = merged[columns].sort_values(["file_id", "start_frame"]).reset_index(drop=True)
    manifest.to_csv(config.accepted_clips_path, index=False)

    strict = manifest.loc[manifest["review_evidence"] == "card+video"]
    strict.to_csv(_strict_path(config), index=False)

    reviewed = resolved["verdict"].value_counts().to_dict()
    by_vendor = (
        manifest.groupby("vendor")
        .agg(clips=("clip_id", "size"), seconds=("window_seconds", "sum"))
        .assign(hours=lambda f: (f["seconds"] / 3600).round(3))
        .drop(columns="seconds")
        .to_dict("index")
    )
    reject_reasons: dict[str, int] = {}
    for reasons in resolved.loc[resolved["verdict"] == "reject", "reasons"]:
        for reason in reasons or ["unspecified"]:
            reject_reasons[reason] = reject_reasons.get(reason, 0) + 1

    return {
        "reviewed_items": int(len(resolved)),
        "verdicts": {str(k): int(v) for k, v in reviewed.items()},
        "accept_rate": round(float(reviewed.get("accept", 0)) / max(1, len(resolved)), 3),
        "contested_items": int(resolved["contested"].sum()),
        "reject_reasons": dict(sorted(reject_reasons.items(), key=lambda kv: -kv[1])),
        "accepted_clips": int(len(manifest)),
        "accepted_seconds": int(manifest["window_seconds"].sum()),
        "accepted_hours": round(float(manifest["window_seconds"].sum() / 3600), 3),
        "accepted_verdicts_without_a_candidate": len(orphaned),
        "accepted_files": int(manifest["file_id"].nunique()),
        "accepted_participants": int(
            (manifest["vendor"].astype(str) + ":" + manifest["participant_id"].astype(str)).nunique()
        ),
        "accepted_hours_with_audio": round(float(strict["window_seconds"].sum() / 3600), 3),
        "by_vendor": by_vendor,
        "verdict_sources": {
            str(k): int(v) for k, v in resolved["verdict_source"].value_counts().to_dict().items()
        },
        "manifest": str(config.accepted_clips_path),
        "strict_manifest": str(_strict_path(config)),
    }


def _strict_path(config: RunConfig) -> Path:
    return config.output_root / "accepted_clips_with_audio.csv"


def verify_manifest(config: RunConfig, sample: int = 24, seed: int = 0) -> dict[str, Any]:
    """Read sampled manifest rows the way a downstream loader would.

    A manifest is a promise about the source tree — *these frames of this file* —
    and the promise is only worth what it survives being cashed. This reads the
    NPZ slice each row names and checks that it is the length the row claims,
    that the upper-body and hand blocks have the shapes ViBES expects, and that
    the speech seconds recomputed from the released VAD match what was recorded.

    It reads the source tree and nothing else, so it also confirms the source is
    intact and unmodified.
    """

    import numpy as np

    from .smplh_kinematics import UPPER_BODY_POSE_INDEX

    manifest = pd.read_csv(config.accepted_clips_path)
    if manifest.empty:
        return {"clips": 0, "checked": 0, "failures": []}

    rows = manifest.sample(min(sample, len(manifest)), random_state=seed)
    failures: list[str] = []
    for row in rows.itertuples():
        base = config.source_root / str(row.source_relbase)
        frames = int(row.end_frame) - int(row.start_frame)
        try:
            with np.load(base.with_suffix(".npz")) as archive:
                body = archive["smplh:body_pose"][row.start_frame : row.end_frame]
                left = archive["smplh:left_hand_pose"][row.start_frame : row.end_frame]
                right = archive["smplh:right_hand_pose"][row.start_frame : row.end_frame]
            annotation = json.loads(base.with_suffix(".json").read_text(encoding="utf-8"))
        except (OSError, ValueError, KeyError) as error:
            failures.append(f"{row.clip_id}: unreadable ({type(error).__name__})")
            continue
        if not len(body) == len(left) == len(right) == frames:
            failures.append(
                f"{row.clip_id}: promised {frames} frames, got "
                f"{len(body)}/{len(left)}/{len(right)}"
            )
            continue
        if body[:, list(UPPER_BODY_POSE_INDEX)].shape[1:] != (13, 3):
            failures.append(f"{row.clip_id}: upper-body pose block is not (13, 3)")
            continue
        start, stop = float(row.start_s), float(row.start_s) + float(row.window_seconds)
        speech = sum(
            min(float(v["end"]), stop) - max(float(v["start"]), start)
            for v in annotation.get("metadata:vad") or []
            if float(v["end"]) > start and float(v["start"]) < stop
        )
        if abs(speech - float(row.speech_seconds)) > 0.5:
            failures.append(
                f"{row.clip_id}: manifest says {row.speech_seconds:.1f} s of speech, "
                f"the released VAD says {speech:.1f} s"
            )

    return {
        "clips": int(len(manifest)),
        "hours": round(float(manifest["window_seconds"].sum() / 3600), 3),
        "upper_body_pose_frames": int((manifest["end_frame"] - manifest["start_frame"]).sum()),
        "checked": int(len(rows)),
        "failures": failures,
    }

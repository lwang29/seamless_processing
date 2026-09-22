"""Stage 6: write the accepted subset definition.

The production decision is **automatic**. ``accepted_clips.csv`` is the output
of tier-1 gates plus tier-2 qualification (:mod:`seamless_curation.qualify`) and
contains no human or model verdict in its causal path. Manual review was removed
from the production pipeline deliberately: at corpus scale it is the binding
constraint, and every criterion a reviewer was applying has been converted into
a measured clause with a stated threshold.

Three files are written, and the difference between them is the strength of the
guarantee each carries. All are CSV, all name frame ranges inside the untouched
source tree, and none copies a byte of media.

``accepted_clips.csv`` — **the training manifest, fully automated.**
    One row per qualifying clip: a contiguous ``[start_frame, end_frame)`` range
    of one participant file that passed every tier-1 gate and every tier-2
    disqualifier, and scored at or above the quality threshold. Carries its
    dimension scores and ``exclusion_flags`` so any row can be traced to the clauses that
    admitted it.

``reviewed_clips.csv`` — **the reviewed subset.**
    Rows whose file a reviewer looked at and accepted. A *development* artefact:
    it is the labelled set the automated stage is validated against.

    **Most of its reviewers are models, not people.** At the time of writing it
    is 17% ``human``, the rest ``model:*`` plus three ``policy:*`` rows from the
    one-handed rubric decision. Do not read it as human sign-off; read
    ``verdict_source`` per row, and use ``accepted_clips_with_audio.csv`` if
    what you want is verdicts a person took with the clip playing. Producing any
    of this is optional; the pipeline runs to completion without it.

``accepted_clips_with_audio.csv`` — **the strict subset.**
    The reviewed rows whose reviewer played the clip with sound, the only rows
    where audio-motion synchronisation was confirmed by ear. Also development.

The relationship between the first two is a *measurement*, not an assumption:
``build_manifests`` computes the automated decision's agreement with every
verdict it can join to and returns it in the summary, so a regression in the
automated stage shows up as a number rather than as a surprise downstream.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from .config import RunConfig
from .qualify import qualification_funnel, qualify
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

#: Tier-2 columns carried into the automated manifest so a decision is auditable
#: from the manifest alone, without re-running anything.
QUALITY_COLUMNS = [
    "gesture_quality", "dim_posture", "dim_persistence", "dim_vigour", "dim_integrity",
    "articulation_ratio", "step_cosine_p50", "arm_speed_speech_p50_mm_s",
    "wrist_range_mm", "episode_median_s", "exclusion_flags",
]

#: Columns of the automated production manifest.
AUTOMATED_COLUMNS = [
    column for column in MANIFEST_COLUMNS
    if column not in ("reviewer", "verdict_source", "review_evidence", "reviewed_utc")
] + QUALITY_COLUMNS + ["decision_source"]


def _ensure_fps(frame: pd.DataFrame, config: RunConfig) -> pd.DataFrame:
    if "fps" in frame.columns and frame["fps"].notna().all():
        return frame
    population = pd.read_parquet(config.population_path)[["file_id", "nominal_fps"]]
    frame = frame.merge(population, on="file_id", how="left")
    frame["fps"] = frame["nominal_fps"] if "fps" not in frame else frame["fps"].fillna(frame["nominal_fps"])
    return frame.drop(columns=["nominal_fps"])


def build_manifests(config: RunConfig) -> dict[str, Any]:
    """Write the automated manifest, the reviewed subset, and agreement between them."""

    candidates = pd.read_parquet(config.candidates_path)
    scored = qualify(candidates, config.qualifiers)
    scored.to_parquet(config.qualified_path)
    funnel = qualification_funnel(scored)
    funnel.to_csv(config.qualification_funnel_path, index=False)

    accepted = _ensure_fps(scored.loc[scored["qualified"]].copy(), config)
    accepted["decision_source"] = "automated"
    columns = [column for column in AUTOMATED_COLUMNS if column in accepted.columns]
    manifest = accepted[columns].sort_values(["file_id", "start_frame"]).reset_index(drop=True)
    manifest.to_csv(config.accepted_clips_path, index=False)

    summary: dict[str, Any] = {
        "decision": "automated",
        "candidates_in": int(len(scored)),
        "accepted_clips": int(len(manifest)),
        "accepted_seconds": int(manifest["window_seconds"].sum()) if len(manifest) else 0,
        "accepted_hours": round(float(manifest["window_seconds"].sum() / 3600), 3) if len(manifest) else 0.0,
        "accepted_files": int(manifest["file_id"].nunique()) if len(manifest) else 0,
        "accepted_participants": int(
            (manifest["vendor"].astype(str) + ":" + manifest["participant_id"].astype(str)).nunique()
        ) if len(manifest) else 0,
        "qualified_rate": round(float(len(manifest)) / max(1, len(scored)), 4),
        "by_vendor": (
            manifest.groupby("vendor")
            .agg(clips=("clip_id", "size"), seconds=("window_seconds", "sum"))
            .assign(hours=lambda f: (f["seconds"] / 3600).round(3))
            .drop(columns="seconds")
            .to_dict("index")
        ) if len(manifest) else {},
        "funnel": {row.clause: int(row.clips_failed_here) for row in funnel.itertuples()},
        "qualifiers": config.qualifiers.as_dict(),
        "manifest": str(config.accepted_clips_path),
    }
    # The production manifest is already written above. Review is a development
    # input, so a malformed or half-written verdict log must not be able to
    # abort this stage -- otherwise review is back in the critical path by the
    # back door, which is the one thing this design is for.
    try:
        summary.update(_review_artefacts(config, scored))
    except ValueError:
        raise
    except Exception as error:  # noqa: BLE001 - deliberately broad; see above
        summary["review"] = {
            "reviewed_items": 0,
            "error": f"{type(error).__name__}: {error}",
            "note": "verdict log could not be read; production output is unaffected",
        }
    return summary


def _review_artefacts(config: RunConfig, scored: pd.DataFrame) -> dict[str, Any]:
    """The reviewed subset and the automated decision's agreement with it.

    Entirely optional: with no verdict log the production manifest above is
    already written and this contributes an empty block.
    """

    resolved = VerdictStore(config.verdict_log).resolve()
    if resolved.empty:
        for path in (config.reviewed_clips_path, _strict_path(config)):
            pd.DataFrame(columns=MANIFEST_COLUMNS).to_csv(path, index=False)
        return {"review": {"reviewed_items": 0, "note": "no verdicts; production output is unaffected"}}

    # Integrity first, over EVERY verdict that joins -- not only the accepts.
    # A reject bound to the wrong file never reaches a manifest, so an
    # accepts-only check would miss it, but it still corrupts the agreement
    # numbers in the summary, which is where a mis-binding does its damage.
    file_of_candidate = scored.drop_duplicates("review_item_id").set_index("review_item_id")["file_id"]
    claimed = resolved.set_index("review_item_id")["file_id"].reindex(file_of_candidate.index).dropna()
    claimed = claimed[claimed.astype(str) != ""]
    disagree = claimed[claimed.astype(str) != file_of_candidate.reindex(claimed.index).astype(str)]
    if len(disagree):
        first = disagree.index[0]
        raise ValueError(
            f"{len(disagree)} verdicts name a different file than the candidate they join to, "
            f"e.g. {first}: verdict says {disagree.iloc[0]!r}, candidate says "
            f"{file_of_candidate[first]!r}"
        )

    accepted_items = resolved.loc[resolved["verdict"] == "accept"]
    merged = scored.merge(
        accepted_items[
            ["review_item_id", "file_id", "reviewer", "verdict_source", "saw_video", "recorded_utc"]
        ].rename(columns={"file_id": "reviewed_file_id"}),
        on="review_item_id",
        how="inner",
    )
    merged = _ensure_fps(merged, config)
    merged["review_evidence"] = merged["saw_video"].map({True: "card+video", False: "card"})
    merged["reviewed_utc"] = merged["recorded_utc"]
    columns = [column for column in MANIFEST_COLUMNS if column in merged.columns]
    reviewed = merged[columns].sort_values(["file_id", "start_frame"]).reset_index(drop=True)
    reviewed.to_csv(config.reviewed_clips_path, index=False)
    strict = reviewed.loc[reviewed["review_evidence"] == "card+video"]
    strict.to_csv(_strict_path(config), index=False)

    agreement = _agreement(scored, resolved)
    reject_reasons: dict[str, int] = {}
    for reasons in resolved.loc[resolved["verdict"] == "reject", "reasons"]:
        for reason in reasons or ["unspecified"]:
            reject_reasons[reason] = reject_reasons.get(reason, 0) + 1

    return {
        "review": {
            "reviewed_items": int(len(resolved)),
            "verdicts": {str(k): int(v) for k, v in resolved["verdict"].value_counts().items()},
            "reject_reasons": dict(sorted(reject_reasons.items(), key=lambda kv: -kv[1])),
            "reviewed_clips": int(len(reviewed)),
            "reviewed_hours": round(float(reviewed["window_seconds"].sum() / 3600), 3) if len(reviewed) else 0.0,
            "reviewed_hours_with_audio": round(float(strict["window_seconds"].sum() / 3600), 3) if len(strict) else 0.0,
            "reviewed_clips_manifest": str(config.reviewed_clips_path),
            "strict_manifest": str(_strict_path(config)),
        },
        "agreement_with_review": agreement,
    }


def _agreement(scored: pd.DataFrame, resolved: pd.DataFrame) -> dict[str, Any]:
    """Confusion of the automated file-level decision against every verdict.

    The automated stage decides per clip; a reviewer decided per file. The
    comparable quantity is therefore "does any clip of this file qualify",
    which is what the file-level manifest membership means.
    """

    per_file = scored.groupby("review_item_id")["qualified"].max()
    out: dict[str, Any] = {}
    source_column = resolved["verdict_source"].astype(str)
    for source, subset in (
        # Strictly first-hand human verdicts. Verdicts with a `policy:` source
        # were converted by a rubric decision rather than observed, so folding
        # them in would let a rule we wrote grade its own homework.
        ("human", resolved.loc[source_column == "human"]),
        ("model", resolved.loc[source_column.str.startswith("model")]),
        ("all", resolved),
    ):
        judged = subset.loc[subset["verdict"].isin(["accept", "reject"])]
        joined = judged.join(per_file, on="review_item_id", how="inner")
        if joined.empty:
            continue
        gold = (joined["verdict"] == "accept").to_numpy()
        predicted = joined["qualified"].fillna(False).to_numpy(dtype=bool)
        tp = int((gold & predicted).sum())
        fp = int((~gold & predicted).sum())
        fn = int((gold & ~predicted).sum())
        tn = int((~gold & ~predicted).sum())
        out[source] = {
            "n": int(len(joined)),
            "true_accept": tp, "false_accept": fp, "false_reject": fn, "true_reject": tn,
            "precision": round(tp / max(1, tp + fp), 3),
            "recall": round(tp / max(1, tp + fn), 3),
            "specificity": round(tn / max(1, tn + fp), 3),
            "accuracy": round((tp + tn) / max(1, len(joined)), 3),
        }
    return out


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

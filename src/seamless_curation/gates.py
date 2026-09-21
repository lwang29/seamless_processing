"""Stage 3: turn window measurements into candidate review clips.

Two groups of gate, kept apart because they answer different questions and
because only one of them is allowed to be strict.

**Tracking gates** ask whether the released SMPL-H upper body is something ViBES
could train on. They are deliberately *loose* about things the PI ruled out of
scope. In particular the v0 pipeline required ``smplh:is_valid`` on every frame
of the file, and that single clause rejected 62.6% of V00 — more than every
other check combined. The PI's position is that a hand briefly leaving the image
is fine as long as the parameters stay stable, so validity is gated on the
window, with an allowance for total invalid time and a separate, tighter limit
on any *continuous* invalid stretch. What actually degrades an invalid frame is
measured directly instead: on invalid frames roughly 40% of hand-pose vectors
are bit-identical to the previous frame while body pose is unaffected, so
``hand_frozen_frac`` is the gate that matters for hands.

**Gesture gates** ask the PI's question, in two halves. The *activity* clauses
ask whether the arms move while the person is speaking; the *variety* clauses
ask whether they move to somewhere different. Both halves are necessary, and the
second was added only after the first manual pass rejected 22 of 36 items for
static hands that every activity clause had passed — hands clasped at the waist
shuffle fast enough to satisfy any speed rule while never leaving one place.

=============================== =============================================
clause                           what it stops
=============================== =============================================
``min_speech_seconds``           judging gesture in a window with no speech
``min_gesture_frac_speech``      hands still while the person talks
``min_speech_segments_covered``  one brief adjustment scored as gesturing
``min_episodes_speech``          a single continuous sweep, however large
``min_wrist_excursion_mm``       micro-motion and tracking wobble
``min_elbow_excursion_mm``       wrist-only fidgets with no arm behind them
``min_gesture_speech_ratio``     constant undirected fidgeting
``min_posture_spread_mm``        the same posture over and over
``min_wrist_height_p75_mm``      hands at the sides or parked at the waist
``max_hands_together_frac``      clasped hands
``min_arm_abduction_p75_deg``    elbows pinned to the ribs
=============================== =============================================

**Every gesture and posture measure is the maximum over the two hands** — the
more mobile wrist for ``posture_spread_mm``, the higher wrist for
``wrist_height_p75_mm``, the more abducted arm for ``arm_abduction_p75_deg``,
the larger excursion for both excursion clauses, and ``np.maximum`` over the two
arm-speed tracks for everything derived from speed. That is deliberate and it is
the reason one-handed gesturing passes: a participant who gestures with one arm
while the other rests in their lap is co-speech motion ViBES should learn, and
averaging the two hands would halve every one of these measures and delete them.
``max_hands_together_frac`` is the only two-hand clause and it is an upper bound,
so a parked hand held away from the gesturing one does not trip it either.
``test_one_handed_gesture_qualifies`` in ``tests/test_gates_and_selection.py``
locks this in. See ``docs/review_rubric.md`` for the matching reviewer rule.

``min_gesture_speech_ratio`` is deliberately mild. Some genuinely expressive
people move while listening too, and a hard co-speech contrast would delete
them; its job is only to reject constant undirected fidgeting.

**No gate is a jitter-magnitude gate.** 106 of 135 candidate quality signals
measured on the dev corpus correlate with gesture activity at |rho| up to 0.903,
including every acceleration and high-pass-residual variant and the one designed
specifically to suppress smooth motion. Gating on any of them is arithmetically
a gate on how much the participant gestured. The two noise guards used here are
outside that family: ``consistency_r`` compares two *independent* measurements
of the same arm, and ``step_cosine_p50`` looks at the *direction* of successive
displacements, which separates jump-out-and-back detector spikes (cosine -0.9 or
below) from real motion (positive) regardless of magnitude.

Nothing in this module accepts data. It produces *candidates* for review.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from .corpus import stable_key, stable_unit_interval


@dataclass(frozen=True)
class Gates:
    """Every cut point, with the reason it sits where it does."""

    # ---- tracking quality -------------------------------------------------
    #: Share of the window on which the released SMPL-H fit is flagged valid.
    #: 0.90 rather than 1.00: at 1.00 the check rejected 62.6% of V00, and the
    #: PI ruled that a hand briefly leaving frame is not a reason to reject.
    min_smplh_valid_frac: float = 0.90
    #: A continuous invalid stretch is different in kind from scattered invalid
    #: frames: it is long enough to be a tracking break rather than an occluded
    #: instant, and it is what a reviewer sees as the mesh going wrong.
    max_smplh_invalid_run_s: float = 1.0
    #: Frames whose 90-parameter hand pose is bit-identical to the previous
    #: frame. This is the measured damage behind an invalid frame, and hand
    #: pose is exactly what ViBES needs.
    max_hand_frozen_frac: float = 0.05
    #: Released 2D keypoints are zero-filled wholesale on box-invalid frames,
    #: and confidence is bimodal (~0.9 or exactly 0), so this is in practice
    #: "the detector found the person in nine frames out of ten".
    min_kp_conf_p10: float = 0.30
    #: A real arm does not exceed 4 m/s; a sustained excess is a tracking break.
    max_implausible_frac: float = 0.002
    #: Agreement between the SMPL-H arm and the released 2D arm. Read the
    #: caveat: this is only meaningful once the gesture gates have established
    #: that there *is* motion to agree about, because two near-static channels
    #: correlate at chance. It is applied after them for that reason.
    min_consistency_r: float = 0.45
    #: Median cosine between successive 2D displacement steps. Detector noise
    #: is antiparallel; motion is not.
    min_step_cosine_p50: float = -0.30

    # ---- co-speech gesture ------------------------------------------------
    #: Seconds of the participant's own speech inside the window. Below this
    #: there is not enough co-speech material to judge, let alone to train on.
    min_speech_seconds: float = 8.0
    #: Share of speaking frames on which the arms are active.
    min_gesture_frac_speech: float = 0.35
    #: Share of the window's utterances (VAD segments >= 0.8 s) that contain a
    #: gesture episode. This is the clause that rejects a single adjustment.
    min_speech_segments_covered: float = 0.40
    #: Distinct gesture episodes overlapping speech.
    min_episodes_speech: int = 3
    #: Wrist excursion from its own within-window median, p90, millimetres.
    #: Betas are all zero, so this is comparable across participants without
    #: normalisation. 80 mm is roughly a third of an upper-arm length.
    min_wrist_excursion_mm: float = 80.0
    #: The arm has to be involved, not just the hand: a wrist flick with a
    #: pinned elbow is not a co-speech gesture.
    min_elbow_excursion_mm: float = 35.0
    #: Mild: rejects constant undirected fidgeting without deleting people who
    #: also move while listening.
    min_gesture_speech_ratio: float = 1.05

    # ---- posture variety --------------------------------------------------
    # Added after the first manual pass. The activity clauses above ask whether
    # the wrists are *moving*; 22 of 36 reviewed items were rejected for
    # ``static_hands`` anyway, and the notes all said the same thing: "the same
    # posture in ten of the twelve moments, only the fingers change". Hands
    # clasped at the waist satisfy every speed and travel clause while never
    # leaving one place. These four clauses name that failure directly.
    #
    #: Mean distance between wrist positions sampled 2.5 s apart during speech —
    #: the same comparison the card's twelve thumbnails invite a reviewer to
    #: make. Shuffling inside one posture scores near zero however fast it is.
    min_posture_spread_mm: float = 150.0
    #: Height of the higher wrist above the shoulder midpoint, p75 over speech
    #: frames. Hands at the sides sit near -550 mm and clasped at the waist near
    #: -350 mm; gesture space starts around the lower chest.
    min_wrist_height_p75_mm: float = -300.0
    #: Share of speech frames with the wrists within 180 mm of each other.
    max_hands_together_frac: float = 0.55
    #: Upper arm away from the torso axis, p75 over speech frames. Elbows pinned
    #: to the ribs cannot make a co-speech gesture, and this says so without
    #: reference to how fast anything moved.
    min_arm_abduction_p75_deg: float = 17.0

    def as_dict(self) -> dict[str, float]:
        return {key: float(value) for key, value in asdict(self).items()}


#: Order matters only for reporting: the first failing clause is recorded as the
#: reason, so the histogram of reasons reads as a funnel.
TRACKING_CLAUSES: tuple[tuple[str, str, str, str], ...] = (
    ("smplh_valid_frac", ">=", "min_smplh_valid_frac", "smplh_invalid"),
    ("smplh_longest_invalid_s", "<=", "max_smplh_invalid_run_s", "smplh_invalid_run"),
    ("hand_frozen_frac", "<=", "max_hand_frozen_frac", "hand_pose_frozen"),
    ("kp_conf_p10", ">=", "min_kp_conf_p10", "keypoints_missing"),
    ("implausible_frac", "<=", "max_implausible_frac", "implausible_speed"),
)

GESTURE_CLAUSES: tuple[tuple[str, str, str, str], ...] = (
    ("speech_seconds", ">=", "min_speech_seconds", "too_little_speech"),
    ("gesture_frac_speech", ">=", "min_gesture_frac_speech", "static_while_speaking"),
    ("speech_segments_covered", ">=", "min_speech_segments_covered", "gesture_not_sustained"),
    ("episode_count_speech", ">=", "min_episodes_speech", "too_few_episodes"),
    ("wrist_excursion_p90_mm", ">=", "min_wrist_excursion_mm", "motion_too_small"),
    ("elbow_excursion_p90_mm", ">=", "min_elbow_excursion_mm", "wrist_only_motion"),
    ("gesture_speech_ratio", ">=", "min_gesture_speech_ratio", "motion_not_speech_linked"),
    ("posture_spread_mm", ">=", "min_posture_spread_mm", "one_posture_only"),
    ("wrist_height_p75_mm", ">=", "min_wrist_height_p75_mm", "hands_below_gesture_space"),
    ("hands_together_frac", "<=", "max_hands_together_frac", "hands_clasped"),
    ("arm_abduction_p75_deg", ">=", "min_arm_abduction_p75_deg", "elbows_pinned"),
)

#: Applied last, because both clauses are only interpretable once there is
#: motion: two static channels agree at chance, and a hand that never moves
#: produces no large steps to take a direction from.
NOISE_CLAUSES: tuple[tuple[str, str, str, str], ...] = (
    ("consistency_r", ">=", "min_consistency_r", "channels_disagree"),
    ("step_cosine_p50", ">=", "min_step_cosine_p50", "motion_is_detector_noise"),
)

ALL_CLAUSES = TRACKING_CLAUSES + GESTURE_CLAUSES + NOISE_CLAUSES


def apply_gates(windows: pd.DataFrame, gates: Gates) -> pd.DataFrame:
    """Add ``qualifies`` and ``fail_reason`` to a window table.

    A window with a non-finite measurement fails on that clause rather than
    passing by accident; ``gesture_status != 'ok'`` fails outright.
    """

    frame = windows.copy()
    reason = pd.Series(pd.NA, index=frame.index, dtype=object)
    limits = gates.as_dict()

    # Cast through object and fill: on a nullable string column `pd.NA != "ok"`
    # is `pd.NA`, which is falsy in a mask, so a window whose status never got
    # written would slip past this clause and be judged on its NaN measures.
    status = frame.get("gesture_status", pd.Series("ok", index=frame.index))
    bad_status = status.astype(object).where(status.notna(), "missing") != "ok"
    reason[bad_status] = "unmeasurable"

    for column, operator, limit_name, label in ALL_CLAUSES:
        if column not in frame.columns:
            raise KeyError(f"window table is missing the measured column {column!r}")
        values = pd.to_numeric(frame[column], errors="coerce")
        # Non-finite is not a measurement. NaN already fails below, and an
        # infinity would satisfy every lower bound, so both are excluded here
        # rather than relying on the comparison to do it.
        values = values.where(np.isfinite(values))
        limit = limits[limit_name]
        ok = values >= limit if operator == ">=" else values <= limit
        failed = (~ok.fillna(False)) & reason.isna()
        reason[failed] = label

    frame["fail_reason"] = reason
    frame["qualifies"] = reason.isna()
    return frame


def gate_funnel(gated: pd.DataFrame) -> pd.DataFrame:
    """Counts by first-failing clause, in clause order, plus the pass row."""

    order = ["unmeasurable"] + [label for *_, label in ALL_CLAUSES]
    counts = gated["fail_reason"].value_counts(dropna=True)
    rows = [
        {"stage": label, "windows": int(counts.get(label, 0))}
        for label in order
        if counts.get(label, 0) or label in {"unmeasurable"}
    ]
    rows.append({"stage": "qualifies", "windows": int(gated["qualifies"].sum())})
    total = int(len(gated))
    for row in rows:
        row["share"] = row["windows"] / total if total else 0.0
    return pd.DataFrame(rows)


def clip_score(windows: pd.DataFrame) -> pd.Series:
    """Rank qualifying windows for review, best first.

    The score is a *review ordering*, never an acceptance criterion. It favours
    windows that are unambiguous to judge — plenty of speech, gesture spread
    across the utterances, real amplitude — so that a reviewer's time goes to
    clips where a verdict is quick and to clips that are worth the disk.
    """

    def unit(column: str, scale: float) -> pd.Series:
        return (pd.to_numeric(windows[column], errors="coerce") / scale).clip(0.0, 1.0)

    return (
        0.25 * unit("gesture_frac_speech", 1.0)
        + 0.20 * unit("speech_segments_covered", 1.0)
        + 0.20 * unit("posture_spread_mm", 320.0)
        + 0.15 * unit("wrist_excursion_p90_mm", 300.0)
        + 0.10 * unit("speech_seconds", 20.0)
        + 0.10 * unit("episode_count_speech", 8.0)
    ).fillna(0.0)


def select_clips(
    gated: pd.DataFrame,
    *,
    max_per_file: int = 8,
    seed: str = "vibes",
) -> pd.DataFrame:
    """Choose non-overlapping candidate clips within each file.

    Overlap matters: windows are struck every ``hop_seconds`` and neighbours
    share most of their frames, so taking the top-N by score would put three
    views of the same eight seconds in the training set. Selection is greedy by
    score with an overlap veto. Ties break on a seeded hash of the clip id, so
    the choice does not depend on row order, numpy version or platform.
    """

    qualifying = gated.loc[gated["qualifies"]].copy()
    if qualifying.empty:
        return qualifying.assign(clip_id=pd.Series(dtype=object))
    qualifying["clip_score"] = clip_score(qualifying)
    qualifying["tie_break"] = [
        stable_unit_interval(str(f), str(s), salt=seed)
        for f, s in zip(qualifying["file_id"], qualifying["start_frame"])
    ]
    # A fresh index: selection collects labels and looks them up with .loc, and
    # a duplicated label from a caller's index would silently multiply rows.
    qualifying = qualifying.reset_index(drop=True).sort_values(
        ["file_id", "clip_score", "tie_break"], ascending=[True, False, True]
    )

    chosen: list[Any] = []
    for _, group in qualifying.groupby("file_id", sort=False):
        taken: list[tuple[int, int]] = []
        for index, row in group.iterrows():
            start, stop = int(row["start_frame"]), int(row["end_frame"])
            if any(start < b and a < stop for a, b in taken):
                continue
            taken.append((start, stop))
            chosen.append(index)
            if len(taken) >= max_per_file:
                break
    selected = qualifying.loc[chosen].copy()
    selected["clip_id"] = [
        f"{row.file_id}_f{int(row.start_frame):06d}" for row in selected.itertuples()
    ]
    return selected.sort_values(["file_id", "start_frame"]).reset_index(drop=True)


#: Columns rolled up from a file's clips onto its review item.
_ITEM_MEANS = (
    "gesture_frac_speech", "speech_segments_covered", "wrist_excursion_p90_mm",
    "elbow_excursion_p90_mm", "gesture_speech_ratio", "sync_r", "consistency_r",
    "smplh_valid_frac", "hand_frozen_frac", "hand_artic_p75_rad_s",
    "arm_speed_speech_p50_mm_s", "torso_travel_mm_s_p50", "step_cosine_p50",
    "posture_spread_mm", "wrist_height_p75_mm", "hands_together_frac",
    "arm_abduction_p75_deg",
)


def build_review_items(
    clips: pd.DataFrame,
    *,
    max_files_per_participant: int = 12,
    seed: str = "vibes",
) -> pd.DataFrame:
    """One review item per file, ordered so that reviewing a prefix is balanced.

    **The review unit is the file, not the clip.** A file's card samples the
    frames that would actually be accepted, and its verdict applies to all of
    that file's clips at once — which is what makes the manual pass affordable:
    a reviewed V00 file yields around a hundred seconds of accepted data instead
    of thirty.

    The order is a stratified round robin over participants rather than a
    ranking by score. Reviewing the first N items therefore gives a set spread
    across people, vendors and conditions, not N clips of the three most
    animated participants in the corpus; and because participants enter in a
    seeded, score-independent order, the accepted subset does not inherit the
    score's biases. Within a participant, their best file comes first.

    ``max_files_per_participant`` bounds how much of one person can enter the
    training set. The corpus is extremely unbalanced — one V00 participant
    appears in 221 files — and an unbalanced training set is a worse training
    set even when every clip in it is good.
    """

    if clips.empty:
        return clips.assign(review_rank=pd.Series(dtype="int64"))

    frame = clips.copy()
    frame["participant_key"] = (
        frame["vendor"].astype(str) + ":" + frame["participant_id"].astype(str)
    )
    grouped = frame.groupby(
        ["participant_key", "file_id", "vendor", "label", "split",
         "session_id", "participant_id", "interaction_id", "interaction_type",
         "source_relbase"],
        as_index=False,
    )
    items = grouped.agg(
        clips=("clip_id", "size"),
        clip_seconds=("window_seconds", "sum"),
        first_start_frame=("start_frame", "min"),
        last_end_frame=("end_frame", "max"),
        item_score=("clip_score", "mean"),
        **{name: (name, "mean") for name in _ITEM_MEANS if name in frame.columns},
    )

    items["tie_break"] = [
        stable_unit_interval(str(f), salt=seed) for f in items["file_id"]
    ]
    items = items.sort_values(["participant_key", "item_score", "tie_break"],
                              ascending=[True, False, True])
    items["file_rank_in_participant"] = items.groupby("participant_key").cumcount()
    if max_files_per_participant > 0:
        items = items.loc[items["file_rank_in_participant"] < max_files_per_participant]

    # Stratified round robin: participants in a seeded order, each contributing
    # their best remaining file to each pass.
    items["participant_order"] = [
        stable_unit_interval(str(p), salt=f"{seed}:participant") for p in items["participant_key"]
    ]
    items = items.sort_values(
        ["file_rank_in_participant", "participant_order", "tie_break"]
    ).reset_index(drop=True)
    items["review_rank"] = np.arange(len(items), dtype="int64")
    # The id is derived from the file, never from its position in the queue.
    # A positional id silently re-binds every stored verdict to a different
    # participant the moment the candidate set changes — one file dropping out
    # of a re-run would put files nobody looked at into the accepted manifest
    # under the name of the reviewer who accepted their former neighbour.
    items["review_item_id"] = [
        "r" + stable_key(str(vendor), str(file_id))
        for vendor, file_id in zip(items["vendor"], items["file_id"])
    ]
    if items["review_item_id"].duplicated().any():
        raise ValueError("review_item_id collision; widen stable_key")
    return items

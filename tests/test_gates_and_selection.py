"""Gates decide nothing on their own; selection must not smuggle anything past them."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from seamless_curation.gates import (
    ALL_CLAUSES,
    Gates,
    apply_gates,
    build_review_items,
    clip_score,
    gate_funnel,
    select_clips,
)


def good_window(**overrides) -> dict:
    row = {
        "gesture_status": "ok",
        "smplh_valid_frac": 1.0,
        "smplh_longest_invalid_s": 0.0,
        "hand_frozen_frac": 0.0,
        "kp_conf_p10": 0.88,
        "implausible_frac": 0.0,
        "consistency_r": 0.9,
        "step_cosine_p50": 0.3,
        "speech_seconds": 14.0,
        "gesture_frac_speech": 0.7,
        "speech_segments_covered": 0.9,
        "episode_count_speech": 5,
        "wrist_excursion_p90_mm": 260.0,
        "elbow_excursion_p90_mm": 90.0,
        "gesture_speech_ratio": 2.0,
        "posture_spread_mm": 240.0,
        "wrist_height_p75_mm": -160.0,
        "hands_together_frac": 0.05,
        "arm_abduction_p75_deg": 30.0,
        "window_seconds": 30.0,
        "file_id": "V00_S1_I1_P1",
        "vendor": "V00",
        "label": "improvised",
        "split": "train",
        "session_id": "S1",
        "participant_id": "P1",
        "interaction_id": "I1",
        "interaction_type": "ipc_conversation",
        "source_relbase": "improvised/train/0/0/V00_S1_I1_P1",
        "start_frame": 0,
        "end_frame": 900,
        "start_s": 0.0,
    }
    row.update(overrides)
    return row


def test_a_clean_window_passes_every_clause() -> None:
    gated = apply_gates(pd.DataFrame([good_window()]), Gates())
    assert bool(gated["qualifies"].iloc[0])
    assert pd.isna(gated["fail_reason"].iloc[0])


@pytest.mark.parametrize(
    "override,expected",
    [
        ({"smplh_valid_frac": 0.5}, "smplh_invalid"),
        ({"smplh_longest_invalid_s": 3.0}, "smplh_invalid_run"),
        ({"hand_frozen_frac": 0.4}, "hand_pose_frozen"),
        ({"kp_conf_p10": 0.0}, "keypoints_missing"),
        ({"implausible_frac": 0.2}, "implausible_speed"),
        ({"speech_seconds": 2.0}, "too_little_speech"),
        ({"gesture_frac_speech": 0.05}, "static_while_speaking"),
        ({"speech_segments_covered": 0.1}, "gesture_not_sustained"),
        ({"episode_count_speech": 1}, "too_few_episodes"),
        ({"wrist_excursion_p90_mm": 20.0}, "motion_too_small"),
        ({"elbow_excursion_p90_mm": 5.0}, "wrist_only_motion"),
        ({"gesture_speech_ratio": 0.4}, "motion_not_speech_linked"),
        ({"posture_spread_mm": 40.0}, "one_posture_only"),
        ({"wrist_height_p75_mm": -520.0}, "hands_below_gesture_space"),
        ({"hands_together_frac": 0.95}, "hands_clasped"),
        ({"arm_abduction_p75_deg": 6.0}, "elbows_pinned"),
        ({"consistency_r": 0.0}, "channels_disagree"),
        ({"step_cosine_p50": -0.95}, "motion_is_detector_noise"),
    ],
)
def test_each_clause_has_a_named_reason(override, expected) -> None:
    gated = apply_gates(pd.DataFrame([good_window(**override)]), Gates())
    assert not bool(gated["qualifies"].iloc[0])
    assert gated["fail_reason"].iloc[0] == expected


def test_a_missing_measurement_fails_rather_than_passing() -> None:
    """NaN must never be read as "the clause did not object"."""

    gated = apply_gates(pd.DataFrame([good_window(gesture_frac_speech=float("nan"))]), Gates())
    assert not bool(gated["qualifies"].iloc[0])
    assert gated["fail_reason"].iloc[0] == "static_while_speaking"


def test_an_unmeasurable_window_is_named_as_such() -> None:
    gated = apply_gates(
        pd.DataFrame([good_window(gesture_status="window_too_short:5")]), Gates()
    )
    assert gated["fail_reason"].iloc[0] == "unmeasurable"


def test_a_missing_column_is_an_error_not_a_silent_pass() -> None:
    frame = pd.DataFrame([good_window()]).drop(columns=["wrist_excursion_p90_mm"])
    with pytest.raises(KeyError, match="wrist_excursion_p90_mm"):
        apply_gates(frame, Gates())


def test_the_funnel_accounts_for_every_window() -> None:
    rows = [good_window(), good_window(speech_seconds=1.0), good_window(hand_frozen_frac=0.9)]
    funnel = gate_funnel(apply_gates(pd.DataFrame(rows), Gates()))
    assert funnel["windows"].sum() == len(rows)
    assert funnel.loc[funnel["stage"] == "qualifies", "windows"].iloc[0] == 1


def test_selection_never_returns_overlapping_clips() -> None:
    """Neighbouring windows share 20 of their 30 seconds; both must not be taken."""

    rows = [
        good_window(start_frame=start, end_frame=start + 900, gesture_frac_speech=score)
        for start, score in ((0, 0.9), (300, 0.85), (600, 0.8), (900, 0.75), (1200, 0.7))
    ]
    clips = select_clips(apply_gates(pd.DataFrame(rows), Gates()), max_per_file=8)
    spans = sorted(zip(clips["start_frame"], clips["end_frame"]))
    for (a_start, a_stop), (b_start, _) in zip(spans, spans[1:]):
        assert b_start >= a_stop, f"{spans} overlap"
    assert len(clips) == 2  # 0-900 and 900-1800 out of the five offered


def test_selection_is_capped_per_file() -> None:
    rows = [
        good_window(start_frame=index * 900, end_frame=(index + 1) * 900)
        for index in range(10)
    ]
    clips = select_clips(apply_gates(pd.DataFrame(rows), Gates()), max_per_file=3)
    assert len(clips) == 3


def test_selection_is_deterministic_and_order_independent() -> None:
    rows = [
        good_window(start_frame=index * 900, end_frame=(index + 1) * 900)
        for index in range(6)
    ]
    gated = apply_gates(pd.DataFrame(rows), Gates())
    first = select_clips(gated, max_per_file=3, seed="x")
    second = select_clips(gated.iloc[::-1].reset_index(drop=True), max_per_file=3, seed="x")
    assert sorted(first["clip_id"]) == sorted(second["clip_id"])


def test_review_items_are_files_and_are_capped_per_participant() -> None:
    rows = []
    for file_index in range(20):
        rows.append(
            good_window(
                file_id=f"V00_S1_I{file_index}_P1",
                interaction_id=f"I{file_index}",
                source_relbase=f"improvised/train/0/0/V00_S1_I{file_index}_P1",
            )
        )
    clips = select_clips(apply_gates(pd.DataFrame(rows), Gates()), max_per_file=8)
    items = build_review_items(clips, max_files_per_participant=5)

    assert len(items) == 5
    assert items["participant_key"].nunique() == 1
    assert list(items["review_rank"]) == [0, 1, 2, 3, 4]


def test_review_order_alternates_between_participants() -> None:
    """Reviewing a prefix of the queue must give a spread of people, not one person.

    Selection ranks by score inside a participant, but the queue interleaves
    participants, so the first N items are N different people wherever possible.
    """

    rows = []
    for participant in ("P1", "P2", "P3"):
        for file_index in range(4):
            rows.append(
                good_window(
                    file_id=f"V00_S1_I{participant}{file_index}_P{participant}",
                    participant_id=participant,
                    interaction_id=f"I{participant}{file_index}",
                    source_relbase=f"x/{participant}/{file_index}",
                )
            )
    clips = select_clips(apply_gates(pd.DataFrame(rows), Gates()), max_per_file=8)
    items = build_review_items(clips, max_files_per_participant=4).sort_values("review_rank")

    assert list(items["participant_key"].head(3).sort_values()) == ["V00:P1", "V00:P2", "V00:P3"]


def test_clip_score_is_an_ordering_not_a_gate() -> None:
    """Score must be monotone in the things a reviewer would call better."""

    weak = pd.DataFrame([good_window(gesture_frac_speech=0.4, wrist_excursion_p90_mm=100.0)])
    strong = pd.DataFrame([good_window(gesture_frac_speech=0.95, wrist_excursion_p90_mm=320.0)])
    assert clip_score(strong).iloc[0] > clip_score(weak).iloc[0]


def test_every_clause_names_a_column_the_scan_produces() -> None:
    """A gate on a column nobody measures would silently never fire."""

    from seamless_curation.scan import ScanSettings  # noqa: F401 - import guard only

    produced = set(good_window())
    for column, _, _, _ in ALL_CLAUSES:
        assert column in produced, f"{column} is gated but not measured"


def test_one_handed_gesture_qualifies() -> None:
    """The gates must pass a one-handed gesturer (rubric decision, 2026-09-21).

    Stated at the gate level with the measures a one-handed participant actually
    produces: the gesturing arm sets every max-over-hands measure, and the
    parked arm shows up only as a wider hand separation, which is an upper-bound
    clause and so is not tripped.
    """

    row = good_window(hands_together_frac=0.0, wrist_height_p75_mm=-180.0)
    gated = apply_gates(pd.DataFrame([row]), Gates())
    assert bool(gated["qualifies"].iloc[0]), gated["fail_reason"].iloc[0]


def test_the_funnel_reports_every_clause_including_the_silent_ones() -> None:
    """A clause that never fires must still appear, as a zero.

    "0" and "absent" are different facts. The first says nothing in this corpus
    looked like that; the second says nothing at all, and is what a mis-wired
    clause also looks like. On the live corpus ``hand_pose_frozen`` fires zero
    times out of 2.42 M windows, and that is worth being able to see.
    """

    gated = apply_gates(pd.DataFrame([good_window(), good_window(kp_conf_p10=0.01)]), Gates())
    funnel = gate_funnel(gated)

    reported = set(funnel["stage"])
    for *_, flag in ALL_CLAUSES:
        assert flag in reported, f"{flag} vanished from the funnel"
    assert {"unmeasurable", "qualifies"} <= reported
    assert len(funnel) == len(ALL_CLAUSES) + 2
    assert int(funnel.loc[funnel.stage == "keypoints_missing", "windows"].iloc[0]) == 1
    assert int(funnel.loc[funnel.stage == "hand_pose_frozen", "windows"].iloc[0]) == 0

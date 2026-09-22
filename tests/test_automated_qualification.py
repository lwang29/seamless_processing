"""The production decision, stated as assertions about named failure modes.

Manual review is no longer in the production path, so these tests *are* the
acceptance criteria. Each one names a failure mode from the PI's brief, builds a
bundle whose motion is exactly that failure, runs it through the real
measurement and both gate tiers, and asserts the outcome.

The positive cases matter as much as the negative ones. A filter that rejects
everything passes every exclusion test ever written; ``test_subtle_but_genuine``
is the one that stops this from silently becoming that filter.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from seamless_curation.gates import Gates, apply_gates
from seamless_curation.gesture import build_tracks, window_measures
from seamless_curation.qualify import (
    DIMENSIONS,
    DISQUALIFIERS,
    RAMPS,
    WEIGHTS,
    REQUIRED_COLUMNS,
    Qualifiers,
    add_quality,
    articulation_ratio,
    qualification_funnel,
    qualify,
)

from tests.conftest import FPS, make_bundle

METADATA = dict(
    file_id="V00_S1_I1_P1", vendor="V00", label="improvised", split="train",
    session_id="S1", participant_id="0001", interaction_id="I1",
    interaction_type="ipc_conversation", source_relbase="improvised/train/0/0/V00_S1_I1_P1",
    start_frame=0, end_frame=900, start_s=0.0, window_seconds=30.0,
)


def decide(model_root, **bundle_kwargs) -> dict:
    """Measure a synthetic bundle and run it through both gate tiers."""

    bundle = make_bundle(**bundle_kwargs)
    tracks = build_tracks(bundle.payload, bundle.vad, fps=FPS, model_root=str(model_root))
    row = window_measures(tracks, 0, 900)
    row.update(METADATA, clip_id="c", review_item_id="r")
    gated = apply_gates(pd.DataFrame([row]), Gates())
    scored = qualify(gated, Qualifiers())
    record = scored.iloc[0].to_dict()
    record["accepted"] = bool(record["qualifies"]) and bool(record["qualified"])
    return record


# --------------------------------------------------------------- the positives
def test_clear_co_speech_gesture_is_accepted(model_root) -> None:
    """Large two-handed gesturing locked to the speech. The thing we want."""

    result = decide(model_root, shoulder_swing_deg=55.0, episodic=True)
    assert result["accepted"], (result["fail_reason"], result["exclusion_flags"])
    assert result["gesture_quality"] > 0.9


def test_one_handed_gesture_is_accepted(model_root) -> None:
    """Rubric decision 2026-09-21; every measure is a max over the two hands."""

    result = decide(model_root, shoulder_swing_deg=55.0, one_handed=True, episodic=True)
    assert result["accepted"], (result["fail_reason"], result["exclusion_flags"])


def test_subtle_but_genuine_gesture_is_not_discarded(model_root) -> None:
    """The false-negative safeguard, and the reason the score compensates.

    A restrained speaker: the swing is less than half the clear case, so the
    posture dimension scores only moderately. They still pass, because their
    gesturing is sustained through every utterance, directionally coherent and
    speech-locked, and those dimensions carry the score.

    If this test starts failing, the pipeline has begun deleting quiet
    gesturers, which is the failure the PI's brief warns about in the opposite
    direction from static hands.
    """

    result = decide(model_root, shoulder_swing_deg=22.0, episodic=True)

    assert result["accepted"], (result["fail_reason"], result["exclusion_flags"])
    assert result["dim_posture"] < 0.75, "fixture is supposed to be modest in amplitude"
    assert result["dim_persistence"] > 0.9 and result["dim_integrity"] > 0.9
    assert result["gesture_quality"] > Qualifiers().min_gesture_quality


# --------------------------------------------------------------- the negatives
def test_static_hands_while_speaking_is_rejected(model_root) -> None:
    """The PI's central complaint: technically clean, hands do not move."""

    result = decide(model_root, shoulder_swing_deg=0.0, episodic=True)

    assert not result["accepted"]
    assert result["smplh_valid_frac"] == 1.0, "the point is that tracking is clean"
    assert "static_while_speaking" in result["exclusion_flags"]


def test_tracking_jitter_is_not_mistaken_for_gesture(model_root) -> None:
    """16 mm of per-frame noise is 480 mm/s -- eight times the speed floor.

    Magnitude alone would call this gesturing for most of the window. It is
    rejected because the motion never *travels* and because successive steps are
    antiparallel, which is a direction test, not a magnitude test.
    """

    result = decide(model_root, shoulder_swing_deg=0.0, jitter_mm=16.0, episodic=True)

    assert not result["accepted"]
    assert result["step_cosine_p50"] < 0.0, "detector noise reverses direction each step"
    assert result["gesture_frac_speech"] < 0.2


def test_global_body_movement_is_not_gesture(model_root) -> None:
    """A participant turning to face their partner, arms rigid on the torso.

    This asserts the *mechanism*, not just the outcome, because the outcome
    alone is uninformative: a bundle with no arm motion is rejected whether or
    not the torso frame works, so a test that only checks "rejected" would pass
    on a pipeline that had no global-motion handling at all. An earlier version
    of this test did exactly that — it drove ``smplh:translation``, which
    nothing in the pipeline reads (forward kinematics puts the pelvis at the
    origin), so it demonstrated nothing.

    So: drive a 50 deg whole-body yaw, show the joints genuinely move in world
    coordinates, and show the torso-frame measures see none of it.
    """

    import numpy as np

    from seamless_curation.smplh_kinematics import L_WRIST, forward_kinematics

    bundle = make_bundle(shoulder_swing_deg=0.0, global_yaw_deg=50.0, episodic=True)

    # The body really does move: reconstruct world-frame joints and measure.
    pose = np.concatenate(
        [
            bundle.payload["smplh:global_orient"][:, None, :],
            bundle.payload["smplh:body_pose"],
            bundle.payload["smplh:left_hand_pose"],
            bundle.payload["smplh:right_hand_pose"],
        ],
        axis=1,
    )
    world, _, _ = forward_kinematics(pose, str(model_root))
    travelled_mm = float(
        np.linalg.norm(world[:, L_WRIST].max(axis=0) - world[:, L_WRIST].min(axis=0)) * 1000
    )
    assert travelled_mm > 100.0, "fixture must actually move the body in the world"

    # And the pipeline sees none of it.
    result = decide(model_root, shoulder_swing_deg=0.0, global_yaw_deg=50.0, episodic=True)

    assert result["wrist_range_mm"] < 1.0, "torso frame must remove whole-body rotation"
    assert result["gesture_frac_speech"] < 0.05
    assert not result["accepted"]
    assert "static_while_speaking" in result["exclusion_flags"]


def test_pure_translation_is_invisible_by_construction(model_root) -> None:
    """``smplh:translation`` is never read, so it cannot be mistaken for gesture.

    Worth stating as a test rather than a comment: it is the reason the
    pipeline needs no threshold for camera or body translation at all.
    """

    still = decide(model_root, shoulder_swing_deg=0.0, episodic=True)
    drifting = decide(model_root, shoulder_swing_deg=0.0, global_drift_mm=400.0, episodic=True)

    assert drifting["wrist_range_mm"] == pytest.approx(still["wrist_range_mm"])
    assert drifting["gesture_frac_speech"] == pytest.approx(still["gesture_frac_speech"])

    gesturing = decide(model_root, shoulder_swing_deg=55.0, episodic=True)
    drifting_and_gesturing = decide(
        model_root, shoulder_swing_deg=55.0, global_drift_mm=400.0, episodic=True
    )
    assert drifting_and_gesturing["accepted"] and gesturing["accepted"]
    assert drifting_and_gesturing["wrist_range_mm"] == pytest.approx(
        gesturing["wrist_range_mm"]
    ), "translation must not change the measurement in either direction"


def test_a_single_brief_adjustment_is_not_gesturing(model_root) -> None:
    """One 0.8 s movement and then stillness, however large that movement is."""

    result = decide(model_root, shoulder_swing_deg=55.0, single_adjustment=True)

    assert not result["accepted"]
    # The movement is genuinely large in extent: the bounding-box range is set
    # by a single outlier frame and reads high.
    assert result["wrist_range_mm"] > 400.0
    # But it is 0.8 s out of 30, so p90 excursion from the resting median is
    # ~0: the robust measure already refuses to be impressed by it.
    assert result["wrist_excursion_p90_mm"] < 50.0

    # This is the case that justifies having disqualifiers at all. The
    # composite score rates this clip ABOVE the accept threshold -- a single
    # emphatic movement looks good on posture and vigour, and the score is a
    # weighted mean, so it cannot see that everything happened at once. What
    # rejects the clip is the episode and coverage clauses: one event cannot be
    # three episodes and cannot cover many utterances, however large it is.
    #
    # If the disqualifiers were ever folded into the score "for simplicity",
    # this clip would be accepted. That is what this assertion is guarding.
    assert result["gesture_quality"] > Qualifiers().min_gesture_quality, (
        "fixture no longer exercises the score-would-accept-it case"
    )
    for clause in ("static_while_speaking", "too_few_episodes", "gesture_not_sustained"):
        assert clause in result["exclusion_flags"], result["exclusion_flags"]


def test_motion_unrelated_to_speech_is_rejected(model_root) -> None:
    """Real, large, sustained motion -- but it happens while the person is silent."""

    result = decide(model_root, shoulder_swing_deg=55.0, gesture_in_silence=True)

    assert not result["accepted"]
    assert result["wrist_range_mm"] > 500.0, "the motion is genuinely large"
    assert result["gesture_frac_speech"] < 0.3, "but not while speaking"


# ------------------------------------------------------- properties of the rule
def test_peak_excursion_is_excluded_from_the_score() -> None:
    """It separates the labelled set backwards; scoring it rewards the beanie case."""

    scored = {column for column, _, _ in RAMPS.values()}
    assert "wrist_excursion_p90_mm" not in scored
    assert "elbow_excursion_p90_mm" not in scored


def test_every_ramp_and_dimension_is_wired_up() -> None:
    referenced = {part for parts in DIMENSIONS.values() for part in parts}
    assert referenced == set(RAMPS), "a ramp exists that no dimension uses, or vice versa"
    assert set(WEIGHTS) == set(DIMENSIONS)
    assert sum(WEIGHTS.values()) == pytest.approx(1.0)


def test_ramps_are_clipped_and_monotone() -> None:
    frame = pd.DataFrame({column: [low - 1e6, low, (low + high) / 2, high, high + 1e6]
                          for column, low, high in RAMPS.values()})
    for column in REQUIRED_COLUMNS:
        if column not in frame.columns:
            frame[column] = 20.0
    scored = add_quality(frame)
    for name, (_, _, _) in ((n, v) for n, v in RAMPS.items()):
        values = scored[f"q_{name}"].to_numpy()
        assert values.min() >= 0.0 and values.max() <= 1.0
        assert np.all(np.diff(values) >= -1e-9), f"{name} is not monotone"


def test_a_missing_measurement_fails_its_clause_rather_than_passing() -> None:
    """NaN and +inf must not satisfy a bound they were never measured against."""

    good = {column: 1e6 for column, _, _ in RAMPS.values()}
    good.update({column: 1e6 for column, _, _, _ in DISQUALIFIERS})
    good.update(window_seconds=30.0, file_id="f", clip_id="c", torso_travel_mm_s_p50=1.0,
                arm_speed_p50_mm_s=1e6)
    rows = [dict(good), dict(good, step_cosine_p50=np.nan), dict(good, consistency_r=np.inf)]
    rows[2]["consistency_r"] = np.nan
    scored = qualify(pd.DataFrame(rows), Qualifiers())

    assert bool(scored["qualified"].iloc[0])
    assert not bool(scored["qualified"].iloc[1])
    assert "motion_is_detector_noise" in scored["exclusion_flags"].iloc[1]
    assert not bool(scored["qualified"].iloc[2])


def test_flags_list_every_failure_and_fail_stage_names_the_first() -> None:
    row = {column: -1e6 for column, _, _, _ in DISQUALIFIERS}
    row.update({column: 0.0 for column, _, _ in RAMPS.values()})
    row.update(window_seconds=30.0, file_id="f", clip_id="c",
               torso_travel_mm_s_p50=1e6, arm_speed_p50_mm_s=0.0)
    scored = qualify(pd.DataFrame([row]), Qualifiers())

    flags = scored["exclusion_flags"].iloc[0].split(";")
    assert len(flags) > 3, "a row this bad fails many clauses; all should be reported"
    assert scored["fail_stage"].iloc[0] == flags[0]
    assert scored["fail_stage"].iloc[0] == DISQUALIFIERS[0][3] or flags[0] in {
        flag for _, _, _, flag in DISQUALIFIERS
    }


def test_articulation_ratio_survives_a_motionless_torso() -> None:
    """Dividing by a still torso must not produce inf and must not fail the clip."""

    frame = pd.DataFrame({"torso_travel_mm_s_p50": [0.0, 1e-9, 20.0],
                          "arm_speed_p50_mm_s": [200.0, 200.0, 200.0]})
    ratio = articulation_ratio(frame)

    assert np.isfinite(ratio).all()
    assert ratio.iloc[0] == pytest.approx(200.0)
    assert ratio.iloc[2] == pytest.approx(10.0)


def test_funnel_accounts_for_every_clip() -> None:
    rows = []
    for quality in (0.0, 0.3, 0.9):
        row = {column: 1e6 for column, _, _, _ in DISQUALIFIERS}
        row.update({column: low + quality * (high - low) for column, low, high in RAMPS.values()})
        row.update(window_seconds=30.0, file_id="f", clip_id=f"c{quality}",
                   torso_travel_mm_s_p50=1.0, arm_speed_p50_mm_s=1e6)
        rows.append(row)
    scored = qualify(pd.DataFrame(rows), Qualifiers())
    funnel = qualification_funnel(scored)

    assert int(funnel["clips_failed_here"].sum()) == len(scored)


def test_the_flags_column_is_not_shadowed_by_a_pandas_attribute() -> None:
    """A column called `flags` is unreachable as `df.flags`.

    `DataFrame.flags` is a pandas built-in returning a Flags object, so
    `manifest.flags` silently hands back the wrong thing instead of raising —
    which is worse than an error, because the caller gets an object and only
    finds out much later. The column is `exclusion_flags` for that reason.
    """

    row = {column: 1e6 for column, _, _, _ in DISQUALIFIERS}
    row.update({column: 1e6 for column, _, _ in RAMPS.values()})
    row.update(window_seconds=30.0, file_id="f", clip_id="c",
               torso_travel_mm_s_p50=1.0, arm_speed_p50_mm_s=1e6)
    scored = qualify(pd.DataFrame([row]), Qualifiers())

    assert "exclusion_flags" in scored.columns
    assert "flags" not in scored.columns
    assert isinstance(scored.exclusion_flags, pd.Series)


def test_the_dimension_weights_are_actually_applied() -> None:
    """`gesture_quality` must be the weighted mean, not any other combination.

    The suite previously asserted only that WEIGHTS summed to 1.0, which says
    nothing about whether they are used. A mutation replacing the weighted mean
    with `max(axis=1)` passed every test in this file and shipped in a commit:
    it inflated the median score from 0.55 to 0.85 and admitted 2,800 extra
    clips before it was caught by hand.

    So: build rows whose dimensions differ, and check the result against the
    weights arithmetically. Max, min, and the unweighted mean all fail this.
    """

    rows = []
    for posture in (0.0, 1.0):
        row = {column: low for column, low, _ in RAMPS.values()}
        for name, (column, low, high) in RAMPS.items():
            row[column] = high if (name in DIMENSIONS["posture"] and posture) else low
        row.update({c: 1e6 for c, _, _, _ in DISQUALIFIERS})
        row.update(window_seconds=30.0, file_id="f", clip_id=f"c{posture}",
                   torso_travel_mm_s_p50=1.0, arm_speed_p50_mm_s=1e6)
        rows.append(row)
    scored = add_quality(pd.DataFrame(rows))

    for _, row in scored.iterrows():
        expected = sum(row[f"dim_{d}"] * w for d, w in WEIGHTS.items()) / sum(WEIGHTS.values())
        assert row["gesture_quality"] == pytest.approx(expected), "not the weighted mean"

    dims = scored[[f"dim_{d}" for d in DIMENSIONS]]
    assert not np.allclose(scored["gesture_quality"], dims.max(axis=1)), "this is max, not a mean"
    assert not np.allclose(scored["gesture_quality"], dims.mean(axis=1)), "weights are being ignored"

    # Raising the posture dimension must raise the score by exactly its weight.
    low, high = scored["gesture_quality"].iloc[0], scored["gesture_quality"].iloc[1]
    delta = (scored["dim_posture"].iloc[1] - scored["dim_posture"].iloc[0]) * WEIGHTS["posture"]
    assert high - low == pytest.approx(delta)

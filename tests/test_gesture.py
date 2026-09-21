"""The measure has to say the right thing about motion whose truth we set.

Each test here is one of the four traps the PI named, stated as an assertion.
"""

from __future__ import annotations

import numpy as np
import pytest

from seamless_curation.gesture import (
    GestureParams,
    build_tracks,
    speech_mask,
    sliding_windows,
    window_measures,
)

from tests.conftest import FPS, make_bundle


def measure(bundle, model_root, start=0, stop=900, **kwargs):
    tracks = build_tracks(
        bundle.payload, bundle.vad, fps=FPS, model_root=str(model_root), **kwargs
    )
    return tracks, window_measures(tracks, start, stop)


def test_a_swinging_arm_is_gesture_and_a_still_one_is_not(model_root) -> None:
    _, moving = measure(make_bundle(shoulder_swing_deg=55.0), model_root)
    _, still = measure(make_bundle(shoulder_swing_deg=0.0), model_root)

    assert moving["gesture_frac_speech"] > 0.8
    assert moving["wrist_excursion_p90_mm"] > 150
    assert still["gesture_frac_speech"] < 0.05
    assert still["wrist_excursion_p90_mm"] < 20


def test_realistic_jitter_in_place_is_not_gesture(model_root) -> None:
    """A wrist that shakes but never travels must not register as active.

    This is the ``min_travel_mm`` clause. Frame-to-frame speed alone would call
    a vibrating wrist active for most of the recording: 8 mm of per-frame noise
    is 240 mm/s, four times the speed floor, and fifteen times the 12-16 mm/s
    resting floor measured on the most static real files in the corpus.
    """

    _, jittery = measure(make_bundle(shoulder_swing_deg=0.0, jitter_mm=16.0), model_root)

    assert jittery["arm_speed_p50_mm_s"] > 60, "the speed clause alone would pass this"
    assert jittery["gesture_frac_speech"] < 0.15
    assert jittery["wrist_excursion_p90_mm"] < 80


def test_large_noise_is_caught_by_the_two_channel_guards(model_root) -> None:
    """Noise big enough to travel is caught by direction, not by magnitude.

    The travel clause protects against vibration inside a 35 mm ball. Noise
    larger than that does move, so it has to be rejected on other grounds: it
    does not appear in the independent 2D channel (``consistency_r``) and its
    successive steps are antiparallel (``step_cosine_p50``). Gating on the
    *magnitude* of the noise is not an option — every such signal measured on
    the dev corpus correlates with genuine gesture activity up to rho 0.903.
    """

    from seamless_curation.gates import Gates, apply_gates
    import pandas as pd

    _, noisy = measure(make_bundle(shoulder_swing_deg=0.0, jitter_mm=60.0, seed=11), model_root)
    row = {**noisy, "gesture_status": "ok"}
    gated = apply_gates(pd.DataFrame([row]), Gates())

    assert noisy["gesture_frac_speech"] > 0.9, "magnitude alone is fooled by this"
    assert noisy["consistency_r"] < 0.2, "the 2D channel does not see it"
    assert noisy["step_cosine_p50"] < -0.5, "successive steps reverse: detector noise"
    assert not bool(gated["qualifies"].iloc[0])


def test_global_body_motion_is_removed_by_the_torso_frame(model_root) -> None:
    """Translating the whole body must change no gesture measure.

    A participant who sways or steps around the platform moves the torso frame
    with them; the wrists inside it do not move.
    """

    still = make_bundle(shoulder_swing_deg=0.0)
    drifting = make_bundle(shoulder_swing_deg=0.0, global_drift_mm=400.0)
    _, a = measure(still, model_root)
    _, b = measure(drifting, model_root)

    assert a["gesture_frac_speech"] == pytest.approx(b["gesture_frac_speech"], abs=1e-9)
    assert a["wrist_excursion_p90_mm"] == pytest.approx(b["wrist_excursion_p90_mm"], abs=1e-6)


def test_one_brief_adjustment_does_not_cover_the_utterances(model_root) -> None:
    """A single movement scores one episode in one utterance, not sustained gesture."""

    bundle = make_bundle(shoulder_swing_deg=70.0, single_adjustment=True)
    _, once = measure(bundle, model_root)

    assert once["episode_count_speech"] <= 1
    assert once["speech_segments_covered"] <= 0.4

    # The point of the coverage clause: an amplitude-only rule sees a big
    # movement here. Restricted to the window it happens in, the excursion is
    # large; spread over thirty seconds it is not, and the gates reject it.
    from seamless_curation.gates import Gates, apply_gates
    import pandas as pd

    tracks, _ = measure(bundle, model_root)
    burst = window_measures(tracks, 0, 60)
    assert burst["wrist_excursion_p90_mm"] > 40
    gated = apply_gates(pd.DataFrame([{**once, "gesture_status": "ok"}]), Gates())
    assert not bool(gated["qualifies"].iloc[0])


def test_gesture_is_scored_separately_inside_and_outside_speech(model_root) -> None:
    """The speech split is what makes the measure *co-speech* rather than motion."""

    frames = 900
    speech_only = np.zeros(frames, dtype=bool)
    speech_only[: frames // 2] = True
    bundle = make_bundle(shoulder_swing_deg=55.0, speech=((0.0, 15.0),))
    tracks = build_tracks(bundle.payload, bundle.vad, fps=FPS, model_root=str(model_root))
    # Silence the second half of the motion by hand so the contrast is exact.
    tracks.active[frames // 2 :] = False
    row = window_measures(tracks, 0, frames)

    assert row["gesture_frac_speech"] > 0.8
    assert row["gesture_frac_silence"] < 0.05
    assert row["gesture_speech_ratio"] > 5


def test_frozen_hand_pose_is_detected(model_root) -> None:
    """The measured damage behind an ``smplh:is_valid == False`` frame."""

    bundle = make_bundle(shoulder_swing_deg=55.0, hand_freeze_from=450)
    _, row = measure(bundle, model_root)

    assert row["hand_frozen_frac"] == pytest.approx(449 / 900, abs=0.01)


def test_invalid_runs_are_measured_not_just_counted(model_root) -> None:
    valid = np.ones(900, dtype=bool)
    valid[100:145] = False          # 1.5 s continuous
    valid[500:503] = False          # 0.1 s
    bundle = make_bundle(shoulder_swing_deg=55.0, smplh_valid=valid)
    _, row = measure(bundle, model_root)

    assert row["smplh_valid_frac"] == pytest.approx(1 - 48 / 900, abs=1e-6)
    assert row["smplh_longest_invalid_s"] == pytest.approx(1.5, abs=1e-6)


def test_speech_mask_matches_the_released_intervals() -> None:
    mask = speech_mask([{"start": 1.0, "end": 2.0}, {"start": 3.0, "end": 3.5}], 150, 30.0)

    assert mask.sum() == 45
    assert mask[30] and mask[59] and not mask[60]
    assert mask[90] and not mask[105]


def test_sliding_windows_tile_without_gaps_and_drop_the_tail() -> None:
    assert sliding_windows(100, 30, 10) == [(0, 30), (10, 40), (20, 50), (30, 60), (40, 70), (50, 80), (60, 90), (70, 100)]
    assert sliding_windows(20, 30, 10) == []


def test_window_measures_refuses_a_window_too_short_to_mean_anything(model_root) -> None:
    tracks = build_tracks(
        make_bundle().payload, make_bundle().vad, fps=FPS, model_root=str(model_root)
    )
    assert window_measures(tracks, 0, 10)["gesture_status"].startswith("window_too_short")


def test_the_two_channels_agree_on_real_motion_and_not_on_noise(model_root) -> None:
    """``consistency_r`` is the independent-measurement guard.

    The synthetic 2D wrists follow the same phase as the SMPL-H arms when there
    is motion, and follow only their own noise when there is not.
    """

    _, moving = measure(make_bundle(shoulder_swing_deg=55.0), model_root)
    _, noise = measure(make_bundle(shoulder_swing_deg=0.0, jitter_mm=40.0, seed=3), model_root)

    assert moving["consistency_r"] > 0.5
    assert not (noise["consistency_r"] > 0.5)


def test_parameters_are_data_not_constants(model_root) -> None:
    """A stricter speed floor has to change the answer, or it is not a parameter."""

    bundle = make_bundle(shoulder_swing_deg=25.0)
    _, loose = measure(bundle, model_root, params=GestureParams(min_speed_mm_s=30.0))
    _, tight = measure(bundle, model_root, params=GestureParams(min_speed_mm_s=400.0))

    assert loose["gesture_frac_speech"] > tight["gesture_frac_speech"]


def test_one_handed_gesture_scores_like_two_handed(model_root) -> None:
    """One arm gesturing must measure the same as two, because of max-over-hands.

    Decided 2026-09-21 (``docs/review_rubric.md``): a participant who gestures
    with one arm while the other rests in their lap is co-speech motion ViBES
    should learn. Every activity and posture measure is therefore the maximum
    over the two hands, never a mean.

    This test is the guard on that. If someone changes any of these measures to
    average the two sides, the parked arm halves the number and this fails —
    which is exactly the silent data loss the assertions exist to prevent.
    """

    _, both = measure(make_bundle(shoulder_swing_deg=55.0), model_root)
    _, single = measure(make_bundle(shoulder_swing_deg=55.0, one_handed=True), model_root)

    for name in (
        "wrist_excursion_p90_mm",
        "elbow_excursion_p90_mm",
        "posture_spread_mm",
        "arm_abduction_p75_deg",
        "gesture_frac_speech",
    ):
        assert single[name] == pytest.approx(both[name], rel=0.05), (
            f"{name}: one-handed {single[name]:.1f} vs two-handed {both[name]:.1f}. "
            "A mean over hands would roughly halve this."
        )

    # The parked hand must not read as clasped: the clause is an upper bound and
    # the resting arm is held away from the gesturing one.
    assert single["hands_together_frac"] < 0.55

    # And the still arm must not be mistaken for a tracking freeze.
    assert single["hand_frozen_frac"] < 0.05

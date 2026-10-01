"""The measure has to say the right thing about motion whose truth we set.

Each of the first tests is one of the four traps the PI named, stated as an
assertion on :func:`clip_measures` (the annotate-everything interface, registry
column names). The gates are gone from these tests: the pipeline no longer
filters, so what is asserted is the measurement, not a pass/fail verdict.

The last block pins the adaptation: the prior-reused columns must reproduce the
previous pipeline's ``window_measures`` (git tag ``v1-cospeech-filter``) exactly
on a 30-fps file, as golden values captured from it, and every adapted or fresh
column must follow its registry NA rule.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from seamless_curation import schema
from seamless_curation.clips import frames_for
from seamless_curation.gesture import (
    CLIP_FLAGS,
    CLIP_FLOAT_MEASURES,
    CLIP_INT_MEASURES,
    GestureParams,
    build_tracks,
    clip_measures,
    recording_measures,
    speech_mask,
)

from tests.conftest import FPS, FRAMES, make_bundle


def tracks_for(bundle, model_root, **kwargs):
    mask = speech_mask(bundle.vad, len(bundle.payload["smplh:body_pose"]), FPS)
    return build_tracks(bundle.payload, mask, fps=FPS, model_root=str(model_root), **kwargs)


def measure(bundle, model_root, start=0, stop=FRAMES, **kwargs):
    tracks = tracks_for(bundle, model_root, **kwargs)
    return tracks, clip_measures(tracks, start, stop)


def same(a, b) -> bool:
    """Exact equality, NaN equal to NaN."""

    if isinstance(a, float) and isinstance(b, float) and math.isnan(a) and math.isnan(b):
        return True
    return a == b


# ============================================================ the four traps
def test_a_swinging_arm_is_active_motion_and_a_still_one_is_not(model_root) -> None:
    _, moving = measure(make_bundle(shoulder_swing_deg=55.0), model_root)
    _, still = measure(make_bundle(shoulder_swing_deg=0.0), model_root)

    assert moving["arm_active_frac_speech"] > 0.8
    assert moving["wrist_excursion_p90_mm"] > 150
    assert still["arm_active_frac_speech"] < 0.05
    assert still["wrist_excursion_p90_mm"] < 20


def test_realistic_jitter_in_place_is_not_active_motion(model_root) -> None:
    """A wrist that shakes but never travels must not register as active.

    This is the ``min_travel_mm`` clause. Frame-to-frame speed alone would call
    a vibrating wrist active for most of the recording: 8 mm of per-frame noise
    is 240 mm/s, four times the speed floor, and fifteen times the 12-16 mm/s
    resting floor measured on the most static real files in the corpus.
    """

    _, jittery = measure(make_bundle(shoulder_swing_deg=0.0, jitter_mm=16.0), model_root)

    assert jittery["arm_speed_p50_mm_s"] > 60, "the speed clause alone would pass this"
    assert jittery["arm_active_frac_speech"] < 0.15
    assert jittery["wrist_excursion_p90_mm"] < 80


def test_large_noise_is_exposed_by_the_two_channel_measures(model_root) -> None:
    """Noise big enough to travel is exposed by direction, not by magnitude.

    The travel clause protects against vibration inside a 35 mm ball. Noise
    larger than that does move, so the activity measures are fooled; what tells
    it apart is that it does not appear in the independent 2D channel
    (``consistency_r``) and that its successive steps are antiparallel
    (``step_cosine_p50``). The magnitude of the noise cannot be the tell: every
    such signal measured on the dev corpus correlates with genuine gesture
    activity up to rho 0.903.
    """

    _, noisy = measure(make_bundle(shoulder_swing_deg=0.0, jitter_mm=60.0, seed=11), model_root)
    _, real = measure(make_bundle(shoulder_swing_deg=55.0), model_root)

    assert noisy["arm_active_frac_speech"] > 0.9, "magnitude alone is fooled by this"
    assert noisy["consistency_r"] < 0.2, "the 2D channel does not see it"
    assert noisy["step_cosine_p50"] < -0.5, "successive steps reverse: detector noise"
    assert real["consistency_r"] > 0.5 and real["step_cosine_p50"] > noisy["step_cosine_p50"]


def test_global_body_motion_is_removed_by_the_torso_frame(model_root) -> None:
    """Translating the whole body must change no motion measure."""

    _, a = measure(make_bundle(shoulder_swing_deg=0.0), model_root)
    _, b = measure(make_bundle(shoulder_swing_deg=0.0, global_drift_mm=400.0), model_root)

    assert a["arm_active_frac_speech"] == pytest.approx(b["arm_active_frac_speech"], abs=1e-9)
    assert a["wrist_excursion_p90_mm"] == pytest.approx(b["wrist_excursion_p90_mm"], abs=1e-6)


def test_one_brief_adjustment_does_not_cover_the_utterances(model_root) -> None:
    """A single movement scores one episode in one utterance, not sustained motion."""

    bundle = make_bundle(shoulder_swing_deg=70.0, single_adjustment=True)
    tracks, once = measure(bundle, model_root)

    assert once["arm_episode_count"] <= 1
    assert once["speech_segments_with_motion_frac"] <= 0.4
    assert once["arm_active_frac"] < 0.05

    # Restricted to the two seconds it happens in, the movement is large; over
    # thirty seconds it is one brief event, which is what the clip says.
    burst = clip_measures(tracks, 0, 60)
    assert burst["wrist_excursion_p90_mm"] > 40
    assert burst["arm_active_frac"] > 5 * once["arm_active_frac"]


def test_motion_is_scored_separately_inside_and_outside_speech(model_root) -> None:
    """The speech split is what makes the measure *co-speech* rather than motion."""

    bundle = make_bundle(shoulder_swing_deg=55.0, speech=((0.0, 15.0),))
    tracks = tracks_for(bundle, model_root)
    tracks.active[FRAMES // 2 :] = False  # silence the second half so the contrast is exact
    row = clip_measures(tracks, 0, FRAMES)

    assert row["arm_active_frac_speech"] > 0.8
    assert row["arm_active_frac_silence"] < 0.05


def test_frozen_hand_pose_is_detected_and_skipped_by_articulation(model_root) -> None:
    """Frozen frames count toward ``hand_frozen_frac`` and never toward articulation."""

    bundle = make_bundle(shoulder_swing_deg=55.0, hand_freeze_from=450)
    tracks, row = measure(bundle, model_root)

    assert row["hand_frozen_frac"] == pytest.approx(449 / 900, abs=1e-9)
    # Frames 451.. are frozen; frame 450 (the last real step) is still a measurement.
    assert int(tracks.hand_speed_ok.sum()) == 451
    assert not tracks.hand_speed_ok[451:].any()
    expected = float(np.percentile(tracks.hand_speed[:451], 75))
    assert row["hand_artic_p75_rad_s"] == pytest.approx(expected, rel=1e-12)
    # The frozen half reads 0 rad/s; including it would have halved the p75.
    assert row["hand_artic_p75_rad_s"] > float(np.percentile(tracks.hand_speed, 75))

    # Fewer than 2 s of real hand frames: not measured.
    tracks_short, short = measure(make_bundle(shoulder_swing_deg=55.0, hand_freeze_from=40), model_root)
    assert int(tracks_short.hand_speed_ok.sum()) < frames_for(2.0, FPS)
    assert math.isnan(short["hand_artic_p75_rad_s"])


def test_the_frame_after_a_frozen_run_is_not_a_hand_measurement(model_root) -> None:
    bundle = make_bundle(shoulder_swing_deg=55.0)
    for key in ("smplh:left_hand_pose", "smplh:right_hand_pose"):
        bundle.payload[key][200:260] = bundle.payload[key][199]
    tracks = tracks_for(bundle, model_root)

    assert tracks.hand_frozen[200:260].all() and not tracks.hand_frozen[260]
    # 260 jumps from the frozen pose to the live one: the whole run's change in one step.
    assert not tracks.hand_speed_ok[200:261].any()
    assert tracks.hand_speed_ok[199] and tracks.hand_speed_ok[261]


def test_invalid_runs_are_measured_not_just_counted(model_root) -> None:
    valid = np.ones(900, dtype=bool)
    valid[100:145] = False          # 1.5 s continuous
    valid[500:503] = False          # 0.1 s
    _, row = measure(make_bundle(shoulder_swing_deg=55.0, smplh_valid=valid), model_root)

    assert row["smplh_valid_frac"] == pytest.approx(1 - 48 / 900, abs=1e-6)
    assert row["smplh_longest_invalid_s"] == pytest.approx(1.5, abs=1e-6)


def test_the_two_channels_agree_on_real_motion_and_not_on_noise(model_root) -> None:
    _, moving = measure(make_bundle(shoulder_swing_deg=55.0), model_root)
    _, noise = measure(make_bundle(shoulder_swing_deg=0.0, jitter_mm=40.0, seed=3), model_root)

    assert moving["consistency_r"] > 0.5
    assert not (noise["consistency_r"] > 0.5)


def test_parameters_are_data_not_constants(model_root) -> None:
    """A stricter speed floor has to change the answer, or it is not a parameter."""

    bundle = make_bundle(shoulder_swing_deg=25.0)
    _, loose = measure(bundle, model_root, params=GestureParams(min_speed_mm_s=30.0))
    _, tight = measure(bundle, model_root, params=GestureParams(min_speed_mm_s=400.0))

    assert loose["arm_active_frac_speech"] > tight["arm_active_frac_speech"]


def test_one_handed_motion_scores_like_two_handed(model_root) -> None:
    """One arm moving must measure the same as two, because of max-over-hands.

    Decided 2026-09-21 (``docs/review_rubric.md``): a participant who gestures
    with one arm while the other rests in their lap is co-speech motion ViBES
    should learn, so every activity and posture measure is the maximum over the
    two hands, never a mean. If one of them is changed to average the sides, the
    parked arm halves it and this fails.
    """

    _, both = measure(make_bundle(shoulder_swing_deg=55.0), model_root)
    _, single = measure(make_bundle(shoulder_swing_deg=55.0, one_handed=True), model_root)

    for name in (
        "wrist_excursion_p90_mm",
        "elbow_excursion_p90_mm",
        "wrist_pose_spread_mm",
        "arm_abduction_p75_deg",
        "arm_active_frac_speech",
    ):
        assert single[name] == pytest.approx(both[name], rel=0.05), (
            f"{name}: one-handed {single[name]:.1f} vs two-handed {both[name]:.1f}. "
            "A mean over hands would roughly halve this."
        )
    assert single["hands_together_frac"] < 0.55   # the parked hand is not "clasped"
    assert single["hand_frozen_frac"] < 0.05      # and the still arm is not a freeze


# ============================================================ adaptation
#: The ``prior:reused`` clip columns, in the order of the golden tuples below,
#: with the previous pipeline's ``window_measures`` name for each.
PRIOR_REUSED = {
    # new registry name -> old window_measures name
    "arm_speed_p50_mm_s": "arm_speed_p50_mm_s",
    "arm_speed_speech_p50_mm_s": "arm_speed_speech_p50_mm_s",
    "wrist_range_mm": "wrist_range_mm",
    "wrist_excursion_p90_mm": "wrist_excursion_p90_mm",
    "elbow_range_mm": "elbow_range_mm",
    "elbow_excursion_p90_mm": "elbow_excursion_p90_mm",
    "kp_conf_p10": "kp_conf_p10",
    "consistency_r": "consistency_r",
    "step_cosine_p50": "step_cosine_p50",
    "smplh_valid_frac": "smplh_valid_frac",
    "smplh_longest_invalid_s": "smplh_longest_invalid_s",
    "subject_present_frac": "box_valid_frac",
    "hand_frozen_frac": "hand_frozen_frac",
    "arm_active_frac": "gesture_frac",
    "arm_active_frac_speech": "gesture_frac_speech",
    "arm_active_frac_silence": "gesture_frac_silence",
    "arm_episode_count": "episode_count",
    "spine_motion_mm_s_p50": "torso_travel_mm_s_p50",
    "speech_motion_sync_r": "sync_r",
    "speech_motion_sync_lag_s": "sync_lag_s",
    "speech_segment_count": "speech_segment_count",
    "speech_segments_with_motion_frac": "speech_segments_covered",
}

#: Bundles the golden values were measured on; every one also has
#: ``smplh:is_valid`` False on frames 300-339.
GOLDEN_BUNDLES = {
    "steady_swing": dict(shoulder_swing_deg=55.0),
    "episodic_jitter": dict(shoulder_swing_deg=55.0, episodic=True, jitter_mm=10.0, seed=4),
    "one_handed_frozen": dict(shoulder_swing_deg=30.0, one_handed=True, hand_freeze_from=400),
}

NAN = float("nan")
#: ``(bundle, start, stop) -> values in PRIOR_REUSED order``, captured 2026-09-24
#: from the previous pipeline itself: ``build_tracks(payload, bundle.vad)`` then
#: ``window_measures`` at git tag ``v1-cospeech-filter``, 30 fps, values as the
#: shortest float64 ``repr``. The last working tree that still had
#: ``window_measures`` gave the same numbers bit for bit, and so did
#: ``clip_measures`` on the precomputed-mask path.
PRIOR_REUSED_GOLDEN: dict[tuple[str, int, int], tuple[float | int, ...]] = {
    ("steady_swing", 0, 900): (
        1062.653568841399, 1062.653568841399, 858.189852958187, 420.78230336138097,
        442.34010795968953, 215.82062166485, 0.5999999841054281, 0.9977104430389302,
        0.999999999999401, 0.9555555555555556, 1.3333333333333333, 1.0,
        0.0, 1.0, 1.0, 1.0,
        1, 0.0, 0.04046383264943142, 1.5,
        3, 1.0,
    ),
    ("steady_swing", 0, 60): (
        1062.653568841399, 1062.6535688180036, 858.189852958187, 418.75925595158884,
        442.34010795968953, 215.82062166485, 0.5999999841054281, 0.9982354100435538,
        0.999999999999401, 1.0, 0.0, 1.0,
        0.0, 1.0, 1.0, 1.0,
        1, 0.0, NAN, NAN,
        1, 1.0,
    ),
    ("steady_swing", 123, 456): (
        1117.6908774495355, 1007.6162602332624, 857.7715536097609, 459.45259398534836,
        442.34010795968953, 233.33162669593165, 0.5999999841054281, 0.9976746278815083,
        0.999999999999401, 0.8798798798798799, 1.3333333333333333, 1.0,
        0.0, 1.0, 1.0, 1.0,
        1, 0.0, 0.0372885212329534, 1.5,
        2, 1.0,
    ),
    ("episodic_jitter", 0, 900): (
        380.64872443086176, 1076.9886938693587, 868.8980552157195, 448.45997611026087,
        452.36414361944304, 230.53200511304922, 0.5999999841054281, 0.9967843778412995,
        0.9980734101552915, 0.9555555555555556, 1.3333333333333333, 1.0,
        0.0, 0.6177777777777778, 1.0, 0.044444444444444446,
        3, 0.0, 0.9970067703451941, 0.0,
        3, 1.0,
    ),
    ("episodic_jitter", 0, 60): (
        130.00021130921874, 944.4715550173767, 478.4917718951183, 439.64497469290586,
        255.01747538818015, 223.67597956957056, 0.5999999841054281, 0.9972808328683606,
        0.9970503887302065, 1.0, 0.0, 1.0,
        0.0, 0.5666666666666667, 1.0, 0.13333333333333333,
        1, 0.0, NAN, NAN,
        1, 1.0,
    ),
    ("episodic_jitter", 123, 456): (
        642.0554123946031, 1077.2446634629555, 863.8554677908542, 442.3291323106668,
        451.2264061446862, 226.1956365250125, 0.5999999841054281, 0.9970411828730581,
        0.9980174861056177, 0.8798798798798799, 1.3333333333333333, 1.0,
        0.0, 0.7237237237237237, 1.0, 0.041666666666666664,
        2, 0.0, 0.9159776884012466, 0.0,
        2, 1.0,
    ),
    ("one_handed_frozen", 0, 900): (
        581.9311795437561, 581.9311795437561, 510.6333509113481, 250.3919789958371,
        262.9272450263888, 129.3428098008024, 0.5999999841054281, 0.9975705823496817,
        0.9999999999972556, 0.9555555555555556, 1.3333333333333333, 1.0,
        0.5544444444444444, 1.0, 1.0, 1.0,
        1, 0.0, 0.04077953080964214, 1.5,
        3, 1.0,
    ),
    ("one_handed_frozen", 0, 60): (
        581.9311795437561, 581.9311795464489, 510.6333509113481, 248.60074949431905,
        262.9272450263888, 129.3428098008024, 0.5999999841054281, 0.9981202354002617,
        0.9999999999972556, 1.0, 0.0, 1.0,
        0.0, 1.0, 1.0, 1.0,
        1, 0.0, NAN, NAN,
        1, 1.0,
    ),
    ("one_handed_frozen", 123, 456): (
        612.3224138194703, 551.5399452680421, 510.55809498824055, 270.36499876495105,
        262.9272450263888, 138.28965046816202, 0.5999999841054281, 0.9975355137963796,
        0.9999999999972556, 0.8798798798798799, 1.3333333333333333, 1.0,
        0.16516516516516516, 1.0, 1.0, 1.0,
        1, 0.0, 0.03728852122669302, 1.5,
        2, 1.0,
    ),
}


def test_the_golden_table_covers_exactly_the_prior_reused_columns() -> None:
    measured = CLIP_FLOAT_MEASURES + CLIP_INT_MEASURES
    reused = {name for name in measured if schema.field("clips", name).provenance == "prior:reused"}
    assert set(PRIOR_REUSED) == reused
    assert all(len(values) == len(PRIOR_REUSED) for values in PRIOR_REUSED_GOLDEN.values())
    assert {case for case, _, _ in PRIOR_REUSED_GOLDEN} == set(GOLDEN_BUNDLES)


@pytest.mark.parametrize("case", sorted(GOLDEN_BUNDLES))
def test_prior_reused_columns_reproduce_the_previous_pipeline(model_root, case) -> None:
    """The registry says ``prior:reused``: same definition, same number.

    At 30 fps every ``frames_for`` conversion equals the old ``round()``, so the
    per-frame tracks are identical to the previous pipeline's and so is every
    reused aggregate. The tolerance (rel 1e-12, integers and NaN exact) only
    absorbs floating-point differences across machines.
    """

    valid = np.ones(FRAMES, dtype=bool)
    valid[300:340] = False
    tracks = tracks_for(make_bundle(smplh_valid=valid, **GOLDEN_BUNDLES[case]), model_root)
    rows = {(start, stop): values for (name, start, stop), values in PRIOR_REUSED_GOLDEN.items() if name == case}
    assert rows
    for (start, stop), values in rows.items():
        row = clip_measures(tracks, start, stop)
        for column, expected in zip(PRIOR_REUSED, values):
            actual = row[column]
            where = (case, start, stop, column, actual, expected)
            if isinstance(expected, int):
                assert type(actual) is int and actual == expected, where
            elif math.isnan(expected):
                assert isinstance(actual, float) and math.isnan(actual), where
            else:
                assert isinstance(actual, float), where
                assert math.isclose(actual, expected, rel_tol=1e-12, abs_tol=0.0), where


def test_one_frame_rule_matches_round_at_30fps_and_fixes_2997() -> None:
    """Why reuse is exact at 30 fps, and what the ceiling changes at 29.97."""

    params = GestureParams()
    for seconds in (params.travel_window_s, params.merge_gap_s, params.min_episode_s,
                    params.min_speech_segment_s, params.min_overlap_s, params.posture_sample_s,
                    params.sync_bin_s):
        assert frames_for(seconds, 30.0) == int(round(seconds * 30.0))
    assert int(round(params.merge_gap_s * 29.97)) == 7
    assert frames_for(params.merge_gap_s, 29.97) == frames_for(params.merge_gap_s, 30.0) == 8


def test_build_tracks_takes_the_precomputed_speech_mask(model_root) -> None:
    bundle = make_bundle(shoulder_swing_deg=55.0)
    mask = np.zeros(FRAMES, dtype=bool)
    mask[100:400] = True
    tracks = build_tracks(bundle.payload, mask, fps=FPS, model_root=str(model_root))

    assert np.array_equal(tracks.speech, mask)
    assert tracks.joints is not None and tracks.joints.shape == (FRAMES, 52, 3)
    assert tracks.head_speed.shape == (FRAMES,) and tracks.hand_speed_ok.shape == (FRAMES,)
    with pytest.raises(ValueError):
        build_tracks(bundle.payload, mask[:-1], fps=FPS, model_root=str(model_root))
    with pytest.raises(TypeError):  # the VAD list the previous pipeline passed
        build_tracks(bundle.payload, bundle.vad, fps=FPS, model_root=str(model_root))


def test_head_speed_is_relative_to_the_upper_spine(model_root) -> None:
    """Nodding reads as head motion; bending the upper spine (head carried along) does not."""

    time = np.arange(FRAMES) / FPS
    amplitude = np.deg2rad(20.0)
    theta = amplitude * np.sin(2 * np.pi * time / 2.0)

    nod = make_bundle()
    nod.payload["smplh:body_pose"][:, 14, 0] = theta          # head, joint 15
    _, nodding = measure(nod, model_root)
    # |d theta/dt| = A (2 pi / P) |cos|; the median of |cos| over whole periods is cos(pi/4).
    expected = np.degrees(amplitude * 2 * np.pi / 2.0 * np.cos(np.pi / 4))
    assert nodding["head_speed_p50_deg_s"] == pytest.approx(expected, rel=0.03)
    assert nodding["head_speed_p75_deg_s"] > nodding["head_speed_p50_deg_s"]

    lean = make_bundle()
    lean.payload["smplh:body_pose"][:, 8, 0] = theta          # spine3, joint 9
    _, leaning = measure(lean, model_root)
    assert leaning["head_speed_p75_deg_s"] < 0.01


def test_burstiness_separates_strokes_from_steady_motion(model_root) -> None:
    _, steady = measure(make_bundle(shoulder_swing_deg=55.0), model_root)
    _, strokes = measure(make_bundle(shoulder_swing_deg=55.0, episodic=True), model_root)
    _, still = measure(make_bundle(shoulder_swing_deg=0.0), model_root)

    assert steady["arm_speed_cv"] < 0.1
    assert strokes["arm_speed_cv"] > 0.4
    assert math.isnan(still["arm_speed_cv"]), "mean speed < 1 mm/s: undefined, not 0"


def test_missing_is_nan_not_zero(model_root) -> None:
    _, still = measure(make_bundle(shoulder_swing_deg=0.0), model_root)
    assert still["arm_episode_count"] == 0
    assert math.isnan(still["arm_episode_median_s"]), "no episode: NaN (was 0.0)"

    _, mute = measure(make_bundle(shoulder_swing_deg=55.0, speech=()), model_root)
    for name in ("arm_active_frac_speech", "arm_speed_speech_p50_mm_s", "wrist_height_speech_p75_mm",
                 "speech_segments_with_motion_frac", "speech_motion_sync_r", "speech_motion_sync_lag_s"):
        assert math.isnan(mute[name]), name
    assert mute["speech_segment_count"] == 0
    assert np.isfinite(mute["wrist_height_p75_mm"]) and np.isfinite(mute["arm_active_frac_silence"])


def test_arm_space_measures_read_all_frames(model_root) -> None:
    """The old speech-frame reading survives only as the ``_speech`` column."""

    tracks = tracks_for(make_bundle(shoulder_swing_deg=55.0), model_root)
    # Hands raised by 300 mm only while speaking: the frame set now matters.
    tracks.wrists[tracks.speech, :, 1] += 300.0
    new = clip_measures(tracks, 0, FRAMES)

    higher = np.maximum(tracks.wrists[:, 0, 1], tracks.wrists[:, 1, 1])
    assert new["wrist_height_p75_mm"] == pytest.approx(np.percentile(higher, 75))
    assert new["wrist_height_speech_p75_mm"] == pytest.approx(np.percentile(higher[tracks.speech], 75))
    assert new["wrist_height_speech_p75_mm"] > new["wrist_height_p75_mm"]
    together = np.linalg.norm(tracks.wrists[:, 0] - tracks.wrists[:, 1], axis=1) < 180.0
    assert new["hands_together_frac"] == pytest.approx(together.mean())
    # Without speech the old column silently fell back to all frames; the
    # all-frames column is unaffected and the speech-only one is NaN.
    mute = tracks_for(make_bundle(shoulder_swing_deg=55.0, speech=()), model_root)
    mute_row = clip_measures(mute, 0, FRAMES)
    mute_higher = np.maximum(mute.wrists[:, 0, 1], mute.wrists[:, 1, 1])
    assert mute_row["wrist_height_p75_mm"] == pytest.approx(np.percentile(mute_higher, 75))
    assert math.isnan(mute_row["wrist_height_speech_p75_mm"])


def test_motion_cut_flags_mark_episodes_that_straddle_the_clip_edges(model_root) -> None:
    moving = tracks_for(make_bundle(shoulder_swing_deg=55.0), model_root)
    assert moving.active.all()
    middle = clip_measures(moving, 300, 600)
    assert middle["motion_cut_at_start"] and middle["motion_cut_at_end"]
    first = clip_measures(moving, 0, 300)
    assert not first["motion_cut_at_start"] and first["motion_cut_at_end"]
    last = clip_measures(moving, 600, FRAMES)
    assert last["motion_cut_at_start"] and not last["motion_cut_at_end"]

    still = tracks_for(make_bundle(shoulder_swing_deg=0.0), model_root)
    row = clip_measures(still, 300, 600)
    assert row["motion_cut_at_start"] is False and row["motion_cut_at_end"] is False


def test_a_clip_shorter_than_two_seconds_is_na_except_the_flags(model_root) -> None:
    tracks = tracks_for(make_bundle(shoulder_swing_deg=55.0), model_root)
    row = clip_measures(tracks, 850, 900)

    assert all(math.isnan(row[name]) for name in CLIP_FLOAT_MEASURES)
    assert all(row[name] is None for name in CLIP_INT_MEASURES)
    assert row["motion_cut_at_start"] is True and row["motion_cut_at_end"] is False


def test_clip_and_recording_keys_are_registry_columns(model_root) -> None:
    tracks = tracks_for(make_bundle(shoulder_swing_deg=55.0, episodic=True), model_root)
    row = clip_measures(tracks, 0, FRAMES)
    clip_columns = set(schema.columns("clips"))

    assert set(row) == set(CLIP_FLOAT_MEASURES) | set(CLIP_INT_MEASURES) | set(CLIP_FLAGS)
    assert set(row) <= clip_columns
    for name, value in row.items():
        dtype = schema.field("clips", name).dtype
        if dtype == "bool":
            assert isinstance(value, bool), name
        elif dtype.startswith("int"):
            assert isinstance(value, int) and not isinstance(value, bool), name
        else:
            assert isinstance(value, float), name

    rec = recording_measures(tracks)
    assert set(rec) <= set(schema.columns("recordings"))
    assert rec["recording_subject_present_frac"] == 1.0
    assert 0.0 <= rec["recording_hand_frozen_frac"] < 0.01


def test_recording_measures_cover_every_frame(model_root) -> None:
    valid = np.ones(FRAMES, dtype=bool)
    valid[:90] = False
    tracks = tracks_for(make_bundle(shoulder_swing_deg=55.0, smplh_valid=valid, hand_freeze_from=600), model_root)
    rec = recording_measures(tracks)

    assert rec["recording_smplh_valid_frac"] == pytest.approx(810 / 900)
    assert rec["recording_hand_frozen_frac"] == pytest.approx(299 / 900)


# ============================================================ VAD conversion
def test_speech_mask_matches_the_released_intervals() -> None:
    mask = speech_mask([{"start": 1.0, "end": 2.0}, {"start": 3.0, "end": 3.5}], 150, 30.0)

    assert mask.sum() == 45
    assert mask[30] and mask[59] and not mask[60]
    assert mask[90] and not mask[105]


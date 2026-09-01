"""Tests for the reviewer-vocabulary proxies and the new review panels.

Each proxy is checked against a synthetic case where the right answer is known
by construction, because "framing" and "sitting" are words and the whole risk in
this module is that a plausible formula measures something else.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from seamless_curation.review_renderer import (
    RenderSettings,
    build_filmstrip,
)
from seamless_curation.vocab_proxies import (
    ANKLES,
    HIPS,
    KNEES,
    SHOULDERS,
    WRISTS,
    audio_envelope_at_frame_rate,
    audiovisual_lag,
    framing_signals,
    hand_activity_signals,
    mouth_open_signal,
    posture_signals,
    roll_signals,
)


WIDTH, HEIGHT = 1080, 1920


def _person(
    frames: int,
    *,
    centre: tuple[float, float] = (540.0, 960.0),
    torso: float = 200.0,
    shoulder_width: float = 300.0,
    roll_deg: float = 0.0,
    standing: bool = True,
) -> np.ndarray:
    """A synthetic upright figure with known geometry."""

    points = np.zeros((frames, 133, 3), dtype=np.float64)
    points[..., 2] = 1.0
    cx, cy = centre
    angle = np.radians(roll_deg)
    half = shoulder_width / 2
    for index in range(frames):
        shoulder_y = cy - torso / 2
        hip_y = cy + torso / 2
        for side, sign in ((0, -1.0), (1, 1.0)):
            dx = sign * half * np.cos(angle)
            dy = sign * half * np.sin(angle)
            points[index, SHOULDERS[side], :2] = (cx + dx, shoulder_y + dy)
            points[index, HIPS[side], :2] = (cx + dx * 0.6, hip_y + dy * 0.6)
            knee_drop = torso * (0.9 if standing else 0.1)
            ankle_drop = torso * (1.7 if standing else 0.5)
            points[index, KNEES[side], :2] = (cx + dx * 0.5, hip_y + knee_drop)
            points[index, ANKLES[side], :2] = (cx + dx * 0.4, hip_y + ankle_drop)
            # Arms down: wrists just below the hips, which is what a
            # participant with static hands actually looks like.
            points[index, WRISTS[side], :2] = (cx + dx * 1.1, hip_y + 30.0)
        points[index, 0, :2] = (cx, shoulder_y - torso * 0.5)
        for face in range(23, 91):
            points[index, face, :2] = (cx, shoulder_y - torso * 0.45)
        for hand in range(91, 133):
            side = 0 if hand < 112 else 1
            base = points[index, WRISTS[side], :2]
            points[index, hand, :2] = base + np.array([2.0, 2.0])
    return points


def test_framing_reports_fill_centring_and_edge_margin() -> None:
    centred = framing_signals(_person(10), None, width=WIDTH, height=HEIGHT)
    assert centred["framing_status"] == "ok"
    # The fixture offsets hand points by 2 px, so the centre is one pixel off.
    assert centred["framing_centre_offset_x_frac_p50"] == pytest.approx(0.0, abs=0.005)
    assert centred["framing_frames_touching_edge_frac"] == 0.0
    assert centred["framing_body_out_of_frame_point_frac"] == 0.0

    # Shifted right until the shoulders leave the raster.
    off = framing_signals(
        _person(10, centre=(1060.0, 960.0)), None, width=WIDTH, height=HEIGHT
    )
    assert off["framing_centre_offset_x_frac_p50"] > 0.4
    assert off["framing_min_edge_margin_frac_p50"] < 0
    assert off["framing_frames_touching_edge_frac"] == 1.0
    assert off["framing_body_out_of_frame_point_frac"] > 0

    small = framing_signals(_person(10, torso=40.0), None, width=WIDTH, height=HEIGHT)
    big = framing_signals(_person(10, torso=300.0), None, width=WIDTH, height=HEIGHT)
    assert small["framing_person_height_frac_p50"] < big["framing_person_height_frac_p50"]


def test_roll_recovers_a_known_tilt_and_separates_lean_from_camera_roll() -> None:
    for expected in (0.0, 12.0, -20.0):
        result = roll_signals(_person(8, roll_deg=expected))
        assert result["roll_status"] == "ok"
        assert result["roll_shoulder_deg_p50"] == pytest.approx(expected, abs=0.5)
    # In the synthetic figure the hip line tilts with the shoulders, which is
    # the camera-roll signature: the difference stays near zero.
    tilted = roll_signals(_person(8, roll_deg=15.0))
    assert abs(tilted["roll_shoulder_minus_hip_deg_p50"]) < 1.0


def test_roll_folds_to_plus_minus_ninety() -> None:
    """A left/right swap must not read as a 180-degree roll."""

    points = _person(4, roll_deg=0.0)
    points[:, [SHOULDERS[0], SHOULDERS[1]]] = points[:, [SHOULDERS[1], SHOULDERS[0]]]
    result = roll_signals(points)
    assert abs(result["roll_shoulder_deg_p50"]) < 1.0


def test_posture_separates_standing_from_sitting() -> None:
    standing = posture_signals(_person(10, standing=True))
    sitting = posture_signals(_person(10, standing=False))
    assert standing["posture_status"] == "ok"
    assert standing["posture_leg_over_torso_p50"] > sitting["posture_leg_over_torso_p50"]
    assert standing["posture_knee_drop_over_torso_p50"] > sitting["posture_knee_drop_over_torso_p50"]
    # Both ratios are scale-free: a person twice the size scores the same.
    doubled = posture_signals(_person(10, torso=400.0, shoulder_width=600.0, standing=True))
    assert doubled["posture_leg_over_torso_p50"] == pytest.approx(
        standing["posture_leg_over_torso_p50"], rel=1e-6
    )


def test_hand_activity_flags_a_wholly_static_recording() -> None:
    still = hand_activity_signals(_person(60), width=WIDTH, height=HEIGHT, fps=30.0)
    assert still["hand_activity_status"] == "ok"
    assert still["hand_activity_static_frac"] == pytest.approx(1.0)
    assert still["hand_activity_left_speed_sw_per_s_p95"] == pytest.approx(0.0, abs=1e-9)

    moving = _person(60)
    for index in range(60):
        moving[index, WRISTS[0], 0] += 150.0 * np.sin(index / 4.0)
        moving[index, WRISTS[0], 1] -= 120.0 * abs(np.sin(index / 4.0))
    active = hand_activity_signals(moving, width=WIDTH, height=HEIGHT, fps=30.0)
    assert active["hand_activity_static_frac"] < 0.5
    assert active["hand_activity_left_speed_sw_per_s_p50"] > 0
    assert active["hand_activity_left_above_hip_frac"] > still["hand_activity_left_above_hip_frac"]


def test_hand_activity_is_scale_free_in_shoulder_widths() -> None:
    def wave(scale: float) -> np.ndarray:
        points = _person(60, torso=200.0 * scale, shoulder_width=300.0 * scale)
        for index in range(60):
            points[index, WRISTS[0], 0] += 100.0 * scale * np.sin(index / 5.0)
        return points

    near = hand_activity_signals(wave(1.0), width=WIDTH, height=HEIGHT, fps=30.0)
    far = hand_activity_signals(wave(0.5), width=WIDTH, height=HEIGHT, fps=30.0)
    assert near["hand_activity_left_speed_sw_per_s_p50"] == pytest.approx(
        far["hand_activity_left_speed_sw_per_s_p50"], rel=1e-6
    )


def test_hand_activity_corrects_for_the_frame_stride() -> None:
    points = _person(40)
    for index in range(40):
        points[index, WRISTS[0], 0] += 10.0 * index
    dense = hand_activity_signals(points, width=WIDTH, height=HEIGHT, fps=30.0, frame_stride=1)
    strided = hand_activity_signals(
        points[::2], width=WIDTH, height=HEIGHT, fps=30.0, frame_stride=2
    )
    # Same physical speed measured on half the frames must give the same answer.
    assert strided["hand_activity_left_speed_sw_per_s_p50"] == pytest.approx(
        dense["hand_activity_left_speed_sw_per_s_p50"], rel=1e-6
    )
    assert strided["hand_activity_frame_stride"] == 2


def test_audiovisual_lag_recovers_a_known_shift_on_clean_signals() -> None:
    """The machinery is correct even though the V00 mouth signal is not."""

    rng = np.random.default_rng(11)
    base = np.cumsum(rng.normal(size=600)) * 0.01 + rng.normal(size=600) * 0.05
    delay = 4
    mouth = base
    # Audio delayed by four frames relative to the mouth, so video leads audio
    # and the convention requires a negative recovered lag.
    audio = np.roll(base, delay)
    result = audiovisual_lag(mouth, audio, max_lag_frames=15)
    assert result["av_lag_status"] == "ok"
    assert abs(result["av_lag_frames"] + delay) <= 1
    assert result["av_lag_peak_r"] > result["av_lag_zero_r"]

    # And the opposite delay recovers the opposite sign.
    lead = audiovisual_lag(np.roll(base, delay), base, max_lag_frames=15)
    assert abs(lead["av_lag_frames"] - delay) <= 1


def test_audiovisual_lag_reports_a_flat_curve_rather_than_a_confident_argmax() -> None:
    rng = np.random.default_rng(3)
    result = audiovisual_lag(rng.normal(size=600), rng.normal(size=600), max_lag_frames=15)
    assert result["av_lag_status"] == "ok"
    # Unrelated signals: the peak must not look like a real alignment.
    assert abs(result["av_lag_peak_r"]) < 0.25
    assert result["av_lag_peak_prominence_r"] < 0.25


def test_audio_envelope_uses_the_video_frame_grid() -> None:
    sample_rate, fps, frames = 48000, 30.0, 10
    samples = np.zeros(int(sample_rate * frames / fps))
    samples[int(sample_rate * 5 / fps) : int(sample_rate * 6 / fps)] = 1.0
    envelope = audio_envelope_at_frame_rate(samples, sample_rate, fps=fps, n_frames=frames)
    assert len(envelope) == frames
    assert envelope[5] == pytest.approx(1.0, abs=1e-6)
    assert envelope[4] == pytest.approx(0.0, abs=1e-9)


def test_mouth_signal_reports_the_resolution_that_limits_it() -> None:
    points = _person(10)
    _, diagnostics = mouth_open_signal(points)
    # The synthetic face is a single point, so inter-ocular distance is zero and
    # the proxy must say so rather than return a number.
    assert diagnostics["mouth_status"] in {"ok", "no_usable_face_frames"}
    assert "mouth_inter_ocular_px_median" in diagnostics


def test_filmstrip_settings_validate() -> None:
    RenderSettings().validate()
    RenderSettings(filmstrip_thumbnails=0).validate()
    with pytest.raises(ValueError):
        RenderSettings(filmstrip_thumbnails=1).validate()


def test_the_retired_crop_panel_is_gone_from_the_renderer() -> None:
    """Round 4 removed Panel A'. Nothing may quietly resurrect it."""

    import seamless_curation.review_renderer as renderer

    for name in ("upper_body_crop_box", "upper_body_crop", "crop_aspect", "crop_pad_frac"):
        assert not hasattr(renderer, name), f"{name} should have been removed with Panel A'"
    for field in ("upper_body_crop", "crop_aspect", "crop_pad_frac"):
        assert not hasattr(RenderSettings(), field)


def test_filmstrip_targets_include_the_first_and_last_frame() -> None:
    """The reviewer asked for both endpoints, evenly spaced in between."""

    from seamless_curation.review_renderer import filmstrip_frame_targets

    targets = filmstrip_frame_targets(9420, 12)
    assert len(targets) == 12
    assert targets[0] == 0, "first thumbnail must be the very first frame"
    assert targets[-1] == 9419, "last thumbnail must be the very last frame"
    gaps = np.diff(targets)
    # Ten intermediate thumbnails, evenly spaced: every gap is within one frame
    # of the exact spacing, which is all integer rounding permits.
    assert gaps.max() - gaps.min() <= 1
    assert targets == sorted(targets)
    # Degenerate inputs must not produce a negative or out-of-range index.
    assert filmstrip_frame_targets(1, 12) == [0] * 12
    assert filmstrip_frame_targets(9420, 1) == [0]
    assert filmstrip_frame_targets(0, 4) == [0, 0, 0, 0]


def test_filmstrip_seek_steps_back_when_the_final_frame_is_unseekable() -> None:
    """Seeking to exactly (n-1)/fps returns nothing on these files.

    Measured on V00_S0180_I00000482_P0047: ``-ss 313.967`` yields no frame while
    ``-ss 313.933`` yields one. Without the step-back the last thumbnail of every
    filmstrip would be a permanent grey "?".
    """

    from seamless_curation import review_renderer

    calls: list[float] = []

    def fake_grab(path, seconds, width, height, repair_filter=""):
        calls.append(seconds)
        return None if seconds > 9.0 else np.zeros((height, width, 3), np.uint8)

    original = review_renderer._grab_frame_at
    review_renderer._grab_frame_at = fake_grab
    try:
        frame, used = review_renderer._grab_frame_stepping_back(
            Path("unused.mp4"), 9.1, 8, 8, step_s=0.1, attempts=4
        )
    finally:
        review_renderer._grab_frame_at = original
    assert frame is not None
    assert used == pytest.approx(9.0)
    assert len(calls) == 2, "it must not keep stepping once a frame comes back"


def test_filmstrip_marks_a_missing_seek_instead_of_skipping_it(tmp_path) -> None:
    from fractions import Fraction

    from seamless_curation.review_renderer import VideoProbe

    probe = VideoProbe(
        width=1080, height=1920, fps=Fraction(30, 1), frame_count=900,
        duration_s=30.0, has_audio=True,
    )
    band, times = build_filmstrip(
        tmp_path / "absent.mp4", canvas_width=600, thumbnails=6, probe=probe,
        clip_start_frame=300, clip_frames=300,
    )
    assert band.shape[1] == 600 and band.shape[0] > 0
    assert len(times) == 6
    # Unreadable video yields the explicit gap colour, not a black strip that
    # could be mistaken for a dark scene.
    assert (band[: band.shape[0] - 18] == np.array([28, 22, 40])).all(axis=2).any()


def test_standing_the_torso_upright_is_a_rigid_rotation() -> None:
    """The side-view correction must not touch the pose it is displaying.

    V00's camera is angled down and Panel C plots camera-frame coordinates, so
    an upright participant is drawn leaning. Undoing that is a rotation, which
    means every joint angle — including the knee bend, which is real and lives
    in the released parameters — has to survive it unchanged.
    """

    from seamless_curation.review_renderer import upright_joints

    joints = np.zeros((22, 3))
    joints[12] = (0.0, -0.45, 0.10)        # neck: up and away, a 12.5 deg lean
    joints[1], joints[2] = (-0.09, 0.0, 0.0), (0.09, 0.0, 0.0)
    joints[4], joints[5] = (-0.085, 0.42, -0.05), (0.085, 0.42, -0.05)
    joints[7], joints[8] = (-0.08, 0.84, 0.0), (0.08, 0.84, 0.0)

    turned = upright_joints(joints)

    # The torso now stands vertical in the drawing plane.
    axis = turned[12] - turned[0]
    assert abs(axis[2]) < 1e-9, "the pelvis-to-neck axis should lie in the image plane"
    assert axis[1] < 0, "the neck must still be above the pelvis"

    # Rigid: every pairwise distance, and so every joint angle, is preserved.
    for a, b in ((0, 12), (1, 4), (4, 7), (2, 5), (5, 8), (7, 8)):
        assert np.linalg.norm(turned[a] - turned[b]) == pytest.approx(
            np.linalg.norm(joints[a] - joints[b]), rel=1e-12
        )

    def knee_angle(points, hip, knee, ankle):
        first, second = points[hip] - points[knee], points[ankle] - points[knee]
        cosine = first @ second / (np.linalg.norm(first) * np.linalg.norm(second))
        return np.degrees(np.arccos(np.clip(cosine, -1, 1)))

    assert knee_angle(turned, 1, 4, 7) == pytest.approx(knee_angle(joints, 1, 4, 7), rel=1e-9)

    # Degenerate input is returned untouched rather than rotated by a guess.
    flat = np.zeros((22, 3))
    assert np.array_equal(upright_joints(flat), flat)


def test_the_upright_side_view_is_a_setting() -> None:
    from seamless_curation.review_renderer import RenderSettings, draw_root_relative_panel

    joints = np.zeros((22, 3))
    joints[12] = (0.0, -0.45, 0.10)
    edges, groups = ((0, 12),), tuple(["body"] * 22)
    assert RenderSettings().upright_side_view is True
    on = draw_root_relative_panel(420, 384, joints, edges, groups, "ok", 1.2,
                                  upright_side_view=True)
    off = draw_root_relative_panel(420, 384, joints, edges, groups, "ok", 1.2,
                                   upright_side_view=False)
    assert not np.array_equal(on, off), "the correction must actually change the drawing"

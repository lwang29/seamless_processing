"""Tests for the M-3 prototype signals.

The two that matter most are the geodesic-versus-naive rotation comparison and
the NaN-aware high-pass filter: both are places where a plausible-looking
implementation produces numbers that are wrong by orders of magnitude.
"""

from __future__ import annotations

import numpy as np
import pytest

from seamless_curation.m3_features import (
    JOINT_GROUPS,
    geodesic_angular_velocity,
    hand_quality,
    metric_3d_signals,
    normalized_2d_kinematics,
    speaking_fraction,
    usable_keypoint_mask,
)


def _keypoints(frames: int, *, value: float = 100.0) -> np.ndarray:
    points = np.zeros((frames, 133, 3), dtype=np.float64)
    points[..., 0] = value
    points[..., 1] = value
    points[..., 2] = 1.0
    return points


def test_geodesic_rate_ignores_axis_angle_wrapping() -> None:
    """A tiny rotation written as a near-2-pi vector must read as tiny."""

    angle = 0.05
    axis = np.array([0.0, 0.0, 1.0])
    # Frame 0 at +angle/2, frame 1 at the same rotation expressed the long way
    # round: magnitude 2*pi - angle/2 about the negated axis.
    first = axis * (angle / 2)
    second = -axis * (2 * np.pi - angle / 2)
    stacked = np.stack([first, second])[:, None, :]

    naive = np.linalg.norm(stacked[1] - stacked[0], axis=-1)[0]
    geodesic = geodesic_angular_velocity(stacked, fps=1.0)[0, 0]

    assert naive > 6.0, "the naive delta should show the wrap artifact"
    assert geodesic == pytest.approx(0.0, abs=1e-6)
    assert naive / max(geodesic, 1e-9) > 1e5


def test_geodesic_rate_recovers_a_known_rotation_rate() -> None:
    fps = 30.0
    step = 0.2
    axis = np.array([0.0, 1.0, 0.0])
    frames = np.stack([axis * (step * index) for index in range(5)])[:, None, :]
    rates = geodesic_angular_velocity(frames, fps=fps)
    assert np.allclose(rates[:, 0], step * fps, atol=1e-6)


def test_geodesic_rate_respects_the_validity_mask() -> None:
    frames = np.zeros((4, 1, 3))
    frames[:, 0, 0] = [0.0, 0.1, 0.2, 0.3]
    valid = np.array([True, True, False, True])
    rates = geodesic_angular_velocity(frames, fps=1.0, valid=valid)
    assert np.isfinite(rates[0, 0])
    assert np.isnan(rates[1, 0]) and np.isnan(rates[2, 0])


def test_high_pass_jitter_is_not_inflated_by_zero_filled_neighbours() -> None:
    """A zero-filled frame must not manufacture jitter in its neighbours.

    Box-invalid frames zero-fill every keypoint. Treating that origin as a real
    position would put the largest fake jitter exactly next to every tracking
    failure.
    """

    frames = 40
    points = _keypoints(frames, value=500.0)
    # One zero-filled frame in the middle, as the release encodes box-invalid.
    points[20] = 0.0
    result = normalized_2d_kinematics(points, width=1080, height=1920, fps=30.0)
    assert result["kin2d_status"] == "ok"
    # The trajectory is otherwise perfectly still, so all genuine jitter is zero.
    assert result["jitter2d_hp_body17_rasterfrac_p95"] == pytest.approx(0.0, abs=1e-12)
    assert result["usable_frac_body17"] == pytest.approx(39 / 40)


def test_zero_filled_frames_do_not_register_as_motion() -> None:
    frames = 30
    points = _keypoints(frames, value=500.0)
    points[10:15] = 0.0
    result = normalized_2d_kinematics(points, width=1080, height=1920, fps=30.0)
    # Speed is only measured across pairs usable in both frames, so jumping to
    # and from the origin contributes nothing.
    assert result["speed2d_body17_rasterfrac_per_s_p95"] == pytest.approx(0.0, abs=1e-12)


def test_usable_mask_rejects_zero_fill_and_nonpositive_confidence() -> None:
    points = _keypoints(3)
    points[0, 5, :2] = 0.0
    points[1, 6, 2] = 0.0
    points[2, 7, 0] = np.nan
    mask = usable_keypoint_mask(points)
    assert not mask[0, 5] and not mask[1, 6] and not mask[2, 7]
    assert mask[0, 4] and mask[1, 5] and mask[2, 6]


def test_normalization_is_isotropic_across_rasters() -> None:
    """The same physical motion in two rasters must give the same speed."""

    def moving(frames: int, scale: float) -> np.ndarray:
        points = np.zeros((frames, 133, 3))
        points[..., 2] = 1.0
        for index in range(frames):
            points[index, :, 0] = (100 + 10 * index) * scale
            points[index, :, 1] = (200 + 10 * index) * scale
        return points

    small = normalized_2d_kinematics(moving(20, 1.0), width=1080, height=1920, fps=30.0)
    large = normalized_2d_kinematics(moving(20, 2.0), width=2160, height=3840, fps=30.0)
    assert small["speed2d_body17_rasterfrac_per_s_p50"] == pytest.approx(
        large["speed2d_body17_rasterfrac_per_s_p50"], rel=1e-9
    )


def test_hand_quality_detects_a_frozen_pose_and_a_flat_hand() -> None:
    frames = 30
    points = _keypoints(frames)
    frozen = np.tile(np.full((1, 15, 3), 0.3), (frames, 1, 1))
    result = hand_quality(points, frozen, hand="left", width=1080, height=1920)
    assert result["hand_left_pose_status"] == "ok"
    assert result["hand_left_pose_frozen_step_frac"] == pytest.approx(1.0)
    assert result["hand_left_pose_temporal_var_mean"] == pytest.approx(0.0)

    flat = np.zeros((frames, 15, 3))
    flat_result = hand_quality(points, flat, hand="left", width=1080, height=1920)
    # With flat_hand_mean=True an exactly zero pose is a flat hand.
    assert flat_result["hand_left_pose_exact_zero_frac"] == pytest.approx(1.0)
    assert flat_result["hand_left_pose_norm_median_rad"] == pytest.approx(0.0)


def test_hand_quality_counts_out_of_frame_points() -> None:
    points = _keypoints(10, value=50.0)
    points[:, JOINT_GROUPS["left_hand"], 0] = -20.0
    result = hand_quality(points, None, hand="left", width=1080, height=1920)
    assert result["hand_left_out_of_frame_point_frac"] == pytest.approx(1.0)
    assert result["hand_left_pose_status"] == "missing_hand_pose"


def test_metric_3d_units_are_millimetres_per_frame_squared() -> None:
    frames, fps = 20, 30.0
    joints = np.zeros((frames, 73, 3))
    # Constant 2 mm/frame^2 along x, so the second difference is exactly 2.
    for index in range(frames):
        joints[index, :, 0] = index * index
    result = metric_3d_signals(joints, fps=fps)
    assert result["metric3d_status"] == "ok"
    assert result["accel3d_body_mm_per_frame2_p50"] == pytest.approx(2.0, rel=1e-9)
    # Wrist speed is a first difference times fps.
    expected = (frames - 1) ** 2 - (frames - 2) ** 2
    assert result["wrist3d_left_speed_mm_per_s_p99"] <= expected * fps


def test_speaking_fraction_merges_overlapping_intervals() -> None:
    result = speaking_fraction([(0.0, 2.0), (1.0, 3.0)], n_frames=300, fps=30.0)
    assert result["vad_status"] == "ok"
    assert result["vad_interval_count"] == 1
    assert result["speaking_frac"] == pytest.approx(3.0 / 10.0)
    assert not result["vad_span_overruns_media"]


def test_speaking_fraction_clips_to_media_and_flags_overrun() -> None:
    result = speaking_fraction([(0.0, 20.0)], n_frames=300, fps=30.0)
    assert result["speaking_frac"] == pytest.approx(1.0)
    assert result["vad_span_overruns_media"] is True
    assert speaking_fraction([], n_frames=300, fps=30.0)["speaking_frac"] == 0.0


def test_speaking_fraction_is_window_local() -> None:
    """A file-level value repeated per span would be a different measurement."""

    from seamless_curation.m3_features import speaking_fraction

    fps, span = 30.0, 150  # 5 s spans
    # Speech only in the first 5 seconds of a 20-second file.
    intervals = [(0.0, 5.0)]
    first = speaking_fraction(intervals, n_frames=span, fps=fps, start_frame=0, media_frames=600)
    later = speaking_fraction(intervals, n_frames=span, fps=fps, start_frame=150, media_frames=600)
    whole = speaking_fraction(intervals, n_frames=600, fps=fps, media_frames=600)

    assert first["speaking_frac"] == pytest.approx(1.0)
    assert later["speaking_frac"] == pytest.approx(0.0)
    assert whole["speaking_frac"] == pytest.approx(0.25)
    # The overrun check stays a property of the file, not of the window.
    assert not first["vad_span_overruns_media"] and not later["vad_span_overruns_media"]
    assert first["vad_segments_in_window"] == 1 and later["vad_segments_in_window"] == 0


def test_speaking_fraction_counts_partial_overlap_once() -> None:
    from seamless_curation.m3_features import speaking_fraction

    # A 4-second turn straddling the window boundary at t = 5 s.
    result = speaking_fraction(
        [(3.0, 7.0)], n_frames=150, fps=30.0, start_frame=0, media_frames=600
    )
    assert result["speaking_frac"] == pytest.approx(2.0 / 5.0)
    assert result["vad_total_speech_s"] == pytest.approx(2.0)
    # The segment statistic reports the whole turn, not the clipped part.
    assert result["vad_longest_segment_s"] == pytest.approx(4.0)

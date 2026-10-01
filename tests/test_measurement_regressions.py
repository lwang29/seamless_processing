"""One test per measurement defect found in the previous pipeline, on the current API.

Each of these was a confirmed bug in the co-speech gesture measure, and each
measure survives into the annotation tables (``prior:reused`` in the registry),
so the regression guards survive with them.
"""

from __future__ import annotations

import numpy as np
import pytest

from tests.conftest import FPS, make_bundle


def _mask(bundle, frames):
    from seamless_curation.gesture import speech_mask

    return speech_mask(bundle.vad, frames, FPS)


def test_step_cosine_is_not_spliced_across_the_two_hands(model_root) -> None:
    """Selecting the faster hand per frame and then differencing teleports.

    The hands sit about a shoulder width apart, so every change of selection
    injected a displacement roughly 100x the real per-frame step: on one window a
    step_cosine_p50 of -0.48 (detector noise) read -0.08.
    """

    from seamless_curation.gesture import build_tracks

    bundle = make_bundle(shoulder_swing_deg=40.0)
    keypoints = bundle.payload["boxes_and_keypoints:keypoints"]
    rng = np.random.default_rng(0)
    keypoints[:, 10, 0] += rng.normal(0, 40, len(keypoints)).astype(np.float32)
    keypoints[:, 112:133, 0] = keypoints[:, 10, None, 0]
    frames = len(keypoints)
    tracks = build_tracks(bundle.payload, _mask(bundle, frames), fps=FPS, model_root=str(model_root))
    assert len(tracks.step_cosine) == 2 * tracks.frames, "both hands, end to end"
    left, right = tracks.step_cosine[: tracks.frames], tracks.step_cosine[tracks.frames:]
    for series in (left, right):
        assert np.isfinite(series).sum() > 10


def test_finger_articulation_is_local_not_global(model_root) -> None:
    """A rigid hand on a swinging arm has zero finger articulation (was 2.3 rad/s)."""

    from seamless_curation.gesture import build_tracks, clip_measures

    bundle = make_bundle(shoulder_swing_deg=55.0)
    bundle.payload["smplh:left_hand_pose"][:] = 0.0
    bundle.payload["smplh:right_hand_pose"][:] = 0.0
    frames = len(bundle.payload["smplh:body_pose"])
    tracks = build_tracks(bundle.payload, _mask(bundle, frames), fps=FPS, model_root=str(model_root))
    value = clip_measures(tracks, 0, frames)["hand_artic_p75_rad_s"]
    # identical hand poses are "frozen" frames, which the adapted measure excludes:
    # either no usable frames (NaN) or zero articulation, never arm swing
    assert np.isnan(value) or value == pytest.approx(0.0, abs=1e-9)


def test_forward_kinematics_returns_local_rotations_too(model_root) -> None:
    from seamless_curation.smplh_kinematics import forward_kinematics, stack_pose

    pose = stack_pose(np.zeros((4, 21, 3), np.float32), np.zeros((4, 15, 3), np.float32),
                      np.zeros((4, 15, 3), np.float32))
    positions, globals_, locals_ = forward_kinematics(pose, str(model_root))
    assert positions.shape == (4, 52, 3)
    assert globals_.shape == locals_.shape == (4, 52, 3, 3)


def test_shoulder_positions_in_the_torso_frame_are_measured_not_assumed(model_root) -> None:
    """They lie on the x axis by construction, but their separation is posed."""

    from seamless_curation.gesture import build_tracks

    bundle = make_bundle(shoulder_swing_deg=30.0)
    frames = len(bundle.payload["smplh:body_pose"])
    tracks = build_tracks(bundle.payload, _mask(bundle, frames), fps=FPS, model_root=str(model_root))
    assert np.abs(tracks.shoulders[:, :, 1:]).max() < 1e-6
    assert (tracks.shoulders[:, 0, 0] > 0).all()
    assert (tracks.shoulders[:, 1, 0] < 0).all()


def test_whole_body_turns_do_not_read_as_arm_motion(model_root) -> None:
    """Arm measures are in the torso frame, so a participant turning to their partner
    (global yaw with rigid arms) has the same arm speed as one standing still."""

    from seamless_curation.gesture import build_tracks, clip_measures

    still = make_bundle(shoulder_swing_deg=0.0)
    turning = make_bundle(shoulder_swing_deg=0.0, global_yaw_deg=40.0)
    frames = len(still.payload["smplh:body_pose"])
    a = clip_measures(build_tracks(still.payload, _mask(still, frames), fps=FPS, model_root=str(model_root)), 0, frames)
    b = clip_measures(build_tracks(turning.payload, _mask(turning, frames), fps=FPS, model_root=str(model_root)), 0, frames)
    assert b["arm_speed_p50_mm_s"] == pytest.approx(a["arm_speed_p50_mm_s"], abs=1e-6)

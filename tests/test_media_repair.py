"""Tests for the raster and timebase repairs.

The point transforms have to agree *exactly* with what OpenCV does to the pixels,
because a review panel draws the two on top of each other: a keypoint that lands
one pixel off a rotated frame is a bug the reviewer would report as a tracking
failure. Each rotation test therefore checks the transform against the image
operation rather than against a hand-derived formula.
"""

from __future__ import annotations

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")

from seamless_curation.media_repair import (
    AnnotationTimebase,
    RasterRepair,
    body_roll_deg,
    ffmpeg_repair_filter,
    quarter_turns_from_roll,
    repair_box,
    repair_camera_joints,
    repair_frame,
    repair_points,
)


def test_identity_repair_leaves_everything_alone():
    repair = RasterRepair(1080, 1920)
    assert repair.is_identity
    assert repair.display_size == (1080, 1920)
    points = np.array([[[10.0, 20.0, 0.9]]])
    assert np.array_equal(repair_points(points, repair), points)


def test_rejects_impossible_geometry():
    with pytest.raises(ValueError):
        RasterRepair(0, 100)
    with pytest.raises(ValueError):
        RasterRepair(100, 100, sample_aspect=0.0)
    with pytest.raises(ValueError):
        RasterRepair(100, 100, quarter_turns=4)


def test_v01_square_raster_squeezes_to_the_declared_portrait():
    repair = RasterRepair(2160, 2160, sample_aspect=9 / 16)
    assert repair.is_anamorphic
    assert repair.display_size == (1215, 2160)
    # 1215/2160 is 9:16, which is what the container's display aspect declares.
    assert repair.display_size[0] / repair.display_size[1] == pytest.approx(9 / 16)


def test_six_point_seven_percent_correction_is_above_the_square_floor():
    # 1012x1920 carries 270:253. Small, but real, and it lands on exactly 1080.
    repair = RasterRepair(1012, 1920, sample_aspect=270 / 253)
    assert repair.is_anamorphic
    assert repair.display_size == (1080, 1920)


def test_a_hundredth_of_a_percent_counts_as_square():
    assert not RasterRepair(1080, 1920, sample_aspect=1.005).is_anamorphic


@pytest.mark.parametrize("turns", [0, 1, 2, 3])
def test_point_rotation_matches_what_opencv_does_to_the_pixels(turns):
    """A lit pixel and its transformed coordinate must stay on the same spot."""
    height, width = 7, 11
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    x, y = 8, 2
    frame[y, x] = 255
    repair = RasterRepair(width, height, quarter_turns=turns)
    rotated = repair_frame(frame, repair)
    moved = repair_points(np.array([float(x), float(y)]), repair)
    assert rotated.shape[1::-1] == repair.display_size
    found = np.argwhere(rotated[..., 0] == 255)[0]
    assert (int(round(moved[0])), int(round(moved[1]))) == (int(found[1]), int(found[0]))


def test_one_turn_stands_up_a_body_lying_toward_the_left_edge():
    # roll -90 is "ground on the left": feet toward x=0, head toward x=W.
    assert quarter_turns_from_roll(-90.0) == 1
    assert quarter_turns_from_roll(90.0) == 3
    assert quarter_turns_from_roll(-88.8) == 1
    assert quarter_turns_from_roll(4.5) == 0
    assert quarter_turns_from_roll(float("nan")) == 0

    width, height = 400, 200
    shoulder = np.array([300.0, 100.0])
    hip = np.array([200.0, 100.0])          # hip to the left of the shoulder
    keypoints = np.zeros((1, 17, 3), dtype=np.float64)
    keypoints[0, 5] = keypoints[0, 6] = (*shoulder, 1.0)
    keypoints[0, 11] = keypoints[0, 12] = (*hip, 1.0)
    roll = body_roll_deg(keypoints)
    assert roll == pytest.approx(-90.0)

    repair = RasterRepair(width, height, quarter_turns=quarter_turns_from_roll(roll))
    moved = repair_points(keypoints[..., :2], repair)
    # After the turn the hip must sit below the shoulder in image coordinates.
    assert moved[0, 11, 1] > moved[0, 5, 1]
    assert body_roll_deg(np.concatenate([moved, keypoints[..., 2:]], axis=-1)) == pytest.approx(0.0)


def test_squeeze_and_turn_compose_in_the_documented_order():
    repair = RasterRepair(2160, 2160, sample_aspect=9 / 16, quarter_turns=1)
    assert repair.display_size == (2160, 1215)
    frame = np.zeros((2160, 2160, 3), dtype=np.uint8)
    frame[100, 1600] = 255
    out = repair_frame(frame, repair)
    assert out.shape[1::-1] == repair.display_size
    moved = repair_points(np.array([1600.0, 100.0]), repair)
    found = np.argwhere(out[..., 0] > 0)
    assert len(found) >= 1
    assert abs(moved[0] - found[:, 1].mean()) <= 1.5
    assert abs(moved[1] - found[:, 0].mean()) <= 1.5


def test_box_corners_are_reordered_after_a_turn():
    repair = RasterRepair(100, 50, quarter_turns=1)
    box = repair_box(np.array([10.0, 5.0, 30.0, 20.0]), repair)
    assert box[0] <= box[2] and box[1] <= box[3]
    # Width and height swap under a quarter turn.
    assert box[2] - box[0] == pytest.approx(15.0)
    assert box[3] - box[1] == pytest.approx(20.0)


def test_three_dimensional_rotation_is_rigid_and_four_turns_is_the_identity():
    rng = np.random.default_rng(11)
    joints = rng.normal(size=(4, 52, 3))
    repair = RasterRepair(100, 100, quarter_turns=1)
    turned = repair_camera_joints(joints, repair)
    # A rigid rotation: every pairwise distance and the depth axis survive.
    original = np.linalg.norm(joints[:, :, None] - joints[:, None, :], axis=-1)
    rotated = np.linalg.norm(turned[:, :, None] - turned[:, None, :], axis=-1)
    assert np.allclose(original, rotated)
    assert np.allclose(joints[..., 2], turned[..., 2])
    four = joints
    for _ in range(4):
        four = repair_camera_joints(four, repair)
    assert np.allclose(four, joints)


def test_three_dimensional_rotation_agrees_with_the_two_dimensional_one():
    """The 3D turn must reproject onto the 2D turn, or Panels B and C disagree."""
    width = height = 240
    focal = 500.0
    joints = np.array([[[0.10, -0.20, 4.0], [-0.15, 0.30, 4.0], [0.05, 0.05, 4.0]]])

    def project(points, w, h):
        return np.stack(
            (focal * points[..., 0] / points[..., 2] + w / 2.0,
             focal * points[..., 1] / points[..., 2] + h / 2.0),
            axis=-1,
        )

    repair = RasterRepair(width, height, quarter_turns=1)
    direct = repair_points(project(joints, width, height), repair)
    turned_width, turned_height = repair.display_size
    via_3d = project(repair_camera_joints(joints, repair), turned_width, turned_height)
    # One pixel of slack: repair_points uses the (W-1) pixel-centre convention
    # and the projection uses W/2 for the principal point.
    assert np.allclose(direct, via_3d, atol=1.0)


def test_ffmpeg_filter_matches_the_array_transform():
    assert ffmpeg_repair_filter(RasterRepair(1080, 1920)) == ""
    assert ffmpeg_repair_filter(
        RasterRepair(2160, 2160, sample_aspect=9 / 16)
    ) == "scale=1215:2160,setsar=1"
    assert ffmpeg_repair_filter(
        RasterRepair(3840, 2160, quarter_turns=1)
    ) == "transpose=2,setsar=1"


def test_drifting_timebase_is_detected_and_maps_by_time():
    # V01_S1607_I00000135_P2569 as released.
    timebase = AnnotationTimebase(
        nominal_fps=48000 / 1001, annotation_frames=3261, container_frames=2807
    )
    assert not timebase.is_consistent
    assert timebase.index_drift_s == pytest.approx(9.47, abs=0.02)
    assert timebase.index_at(0.0) == 0
    assert timebase.index_at(68.0) == 3261 - 1          # clamped at the last frame
    assert timebase.index_at(34.0) == pytest.approx(1630, abs=1)


def test_consistent_timebase_tolerates_a_single_frame():
    assert AnnotationTimebase(30.0, 900, 900).is_consistent
    assert AnnotationTimebase(30.0, 900, 899).is_consistent
    assert not AnnotationTimebase(30.0, 900, 897).is_consistent


def test_pillarbox_crop_is_measured_then_applied():
    from seamless_curation.media_repair import measure_pillarbox, pillarbox_for

    # A 540-wide bright block centred in a 1080-wide frame, as V03's 1080x960 is.
    frames = np.zeros((3, 40, 1080), dtype=np.uint8)
    frames[:, :, 270:810] = 200
    assert measure_pillarbox(frames) == (270, 270)
    # One dark frame must not turn a lit column into padding.
    frames[1] = 0
    assert measure_pillarbox(frames) == (270, 270)
    assert pillarbox_for(1080, 960) == (270, 270)
    assert pillarbox_for(1080, 1920) == (0, 0)


def test_cropping_moves_points_and_pixels_together():
    repair = RasterRepair(1080, 960, pillarbox=(270, 270))
    assert repair.is_cropped and not repair.is_identity
    assert repair.display_size == (540, 960)
    frame = np.zeros((960, 1080, 3), dtype=np.uint8)
    frame[500, 600] = 255
    out = repair_frame(frame, repair)
    moved = repair_points(np.array([600.0, 500.0]), repair)
    assert out.shape[1::-1] == (540, 960)
    found = np.argwhere(out[..., 0] == 255)[0]
    assert (int(moved[0]), int(moved[1])) == (int(found[1]), int(found[0]))
    assert ffmpeg_repair_filter(repair) == "crop=540:960:270:0,setsar=1"


def test_crop_then_squeeze_then_turn_stays_in_that_order():
    repair = RasterRepair(2180, 3840, pillarbox=(0, 20), sample_aspect=0.5, quarter_turns=1)
    assert repair.cropped_size == (2160, 3840)
    assert repair.squeezed_size == (1080, 3840)
    assert repair.display_size == (3840, 1080)
    assert ffmpeg_repair_filter(repair) == "crop=2160:3840:0:0,scale=1080:3840,transpose=2,setsar=1"


def test_a_crop_that_would_leave_nothing_is_refused():
    with pytest.raises(ValueError):
        RasterRepair(100, 100, pillarbox=(60, 40))


def test_the_renderer_actually_applies_the_crop():
    """Regression: v7 shipped the crop and never called it, so the bars stayed.

    The unit tests all passed because they exercised `RasterRepair` directly.
    Nothing checked that the renderer *built* one with a pillarbox in it.
    """
    from seamless_curation.review_renderer import RenderSettings, VideoProbe, _raster_repair
    from fractions import Fraction

    keypoints = np.zeros((4, 17, 3))
    keypoints[..., 2] = 1.0
    keypoints[:, 5] = keypoints[:, 6] = (540.0, 200.0, 1.0)
    keypoints[:, 11] = keypoints[:, 12] = (540.0, 500.0, 1.0)

    padded = VideoProbe(1080, 960, Fraction(30), 900, 30.0, True, Fraction(30), 1.0)
    repair = _raster_repair(padded, keypoints, RenderSettings())
    assert repair.pillarbox == (270, 270)
    assert repair.display_size == (540, 960)

    plain = VideoProbe(1080, 1920, Fraction(30), 900, 30.0, True, Fraction(30), 1.0)
    assert _raster_repair(plain, keypoints, RenderSettings()).is_identity


def test_a_container_a_frame_short_holds_rather_than_failing():
    """One file has 11,600 annotations and 11,599 stored frames.

    That is inside the timebase tolerance, so the renderer takes the plain
    sequential path -- and a clip reaching the very end used to abort the whole
    render. A frame or two is rounding; more than that is still a fault.
    """
    from seamless_curation.review_renderer import _frames_at_annotation_times

    class Capture:
        def __init__(self, frames):
            self.frames, self.index = frames, 0
        def set(self, *_):
            return True
        def read(self):
            if self.index >= len(self.frames):
                return False, None
            frame = self.frames[self.index]
            self.index += 1
            return True, frame

    timebase = AnnotationTimebase(nominal_fps=30.0, annotation_frames=10, container_frames=10)
    assert timebase.is_consistent
    frames = [np.full((2, 2, 3), i, np.uint8) for i in range(8)]
    out = list(_frames_at_annotation_times(Capture(list(frames)), 0, 10, timebase))
    assert len(out) == 10
    assert np.array_equal(out[-1], frames[-1]) and np.array_equal(out[-2], frames[-1])

    # Three frames short is a real problem and still raises.
    with pytest.raises(RuntimeError):
        list(_frames_at_annotation_times(Capture(list(frames)), 0, 11, timebase))

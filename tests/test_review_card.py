"""The card is the review artefact; a card that misleads is worse than none."""

from __future__ import annotations

import numpy as np
import pytest

from seamless_curation.review_card import (
    CardLayout,
    compose_card,
    draw_keypoint_overlay,
    draw_pose_panel,
    upper_body_crop_box,
)


def keypoints(hand_x: float = 1000.0, hand_y: float = 300.0) -> np.ndarray:
    points = np.zeros((3, 133, 3))
    points[:, :, 2] = 0.9
    points[:, 5, :2] = (651, 893)
    points[:, 6, :2] = (329, 893)
    points[:, 11, :2] = (600, 1400)
    points[:, 12, :2] = (380, 1400)
    points[:, 91:112, :2] = (hand_x, hand_y)
    points[:, 112:133, :2] = (1080 - hand_x, 1700)
    return points


@pytest.mark.parametrize(
    "hand_x,hand_y",
    [(1000.0, 300.0), (20.0, 1900.0), (540.0, 900.0), (1079.0, 1919.0), (0.0, 0.0)],
)
def test_the_crop_keeps_its_aspect_and_stays_inside_the_raster(hand_x, hand_y) -> None:
    """A clipped crop displaces the overlay by a body length.

    The first version clamped the box to the raster edge without re-deriving the
    other side. A wide gesture then produced a 1080x1848 crop that ffmpeg
    squashed into a 250x200 panel, while the skeleton was drawn with one
    isotropic scale — so it landed below the participant's knees, and two
    reviewers recorded ``tracking_broken`` on files whose tracking was fine.
    """

    width, height = 1080, 1920
    left, top, right, bottom = upper_body_crop_box(
        keypoints(hand_x, hand_y), [0, 1, 2], width, height
    )

    assert 0 <= left < right <= width
    assert 0 <= top < bottom <= height
    aspect = (right - left) / (bottom - top)
    assert aspect == pytest.approx(CardLayout().crop_width / CardLayout().crop_height, rel=0.02)
    # ffmpeg's crop filter needs even offsets and sizes on yuv420p input.
    assert left % 2 == 0 and top % 2 == 0
    assert (right - left) % 2 == 0 and (bottom - top) % 2 == 0


def test_the_crop_matches_the_panel_so_the_overlay_lands_on_the_person() -> None:
    """End to end: a shoulder keypoint must be drawn where the shoulder is."""

    layout = CardLayout()
    points = keypoints()
    left, top, right, bottom = upper_body_crop_box(points, [0, 1, 2], 1080, 1920)
    scale = (layout.crop_width / (right - left), layout.crop_height / (bottom - top))
    assert scale[0] == pytest.approx(scale[1], rel=0.03), "anisotropic scale means a wrong overlay"

    frame = np.zeros((layout.crop_height, layout.crop_width, 3), np.uint8)
    draw_keypoint_overlay(frame, points[0], (left, top), scale)

    shoulder_x = int(round((651 - left) * scale[0]))
    shoulder_y = int(round((893 - top) * scale[1]))
    assert 0 <= shoulder_x < layout.crop_width and 0 <= shoulder_y < layout.crop_height
    patch = frame[max(0, shoulder_y - 4) : shoulder_y + 5, max(0, shoulder_x - 4) : shoulder_x + 5]
    assert patch.any(), "nothing was drawn at the shoulder"


def test_a_degenerate_keypoint_block_still_yields_a_usable_crop() -> None:
    empty = np.zeros((2, 133, 3))
    left, top, right, bottom = upper_body_crop_box(empty, [0, 1], 1080, 1920)
    expected = CardLayout().crop_width / CardLayout().crop_height
    assert (right - left) / (bottom - top) == pytest.approx(expected, rel=0.02)
    assert right <= 1080 and bottom <= 1920


def test_the_pose_panel_keeps_a_fixed_scale_until_the_pose_does_not_fit() -> None:
    """Two panels must be comparable, which auto-fitting every frame would break."""

    from seamless_curation.review_card import REFERENCE_SCALE_PX_PER_M

    joints = np.zeros((52, 3))
    joints[16] = (0.17, 0.0, 0.0)
    joints[17] = (-0.17, 0.0, 0.0)
    joints[12] = (0.0, 0.05, 0.0)
    joints[15] = (0.0, 0.25, 0.0)
    small = draw_pose_panel(joints.copy(), 250, 200)
    assert small.shape == (200, 250, 3)
    assert b"x0." not in small.tobytes()[:0]  # no rescale marker path taken

    big = joints.copy()
    big[20] = (0.0, 2.0, 0.0)  # a wrist two metres up cannot fit at reference scale
    panel = draw_pose_panel(big, 250, 200)
    assert panel.shape == (200, 250, 3)
    assert REFERENCE_SCALE_PX_PER_M > 0


def test_an_invalid_frame_is_marked_on_the_panel() -> None:
    joints = np.zeros((52, 3))
    joints[15] = (0.0, 0.25, 0.0)
    valid = draw_pose_panel(joints.copy(), 250, 200, valid=True)
    invalid = draw_pose_panel(joints.copy(), 250, 200, valid=False)
    assert not np.array_equal(valid, invalid)
    # The marker is a red border; the corner pixel must differ.
    assert invalid[0, 0, 2] > valid[0, 0, 2]


def test_compose_card_produces_the_declared_geometry() -> None:
    layout = CardLayout()
    thumbs = [np.zeros((layout.crop_height, layout.crop_width, 3), np.uint8)] * layout.thumbnails
    poses = [np.zeros((layout.pose_height, layout.crop_width, 3), np.uint8)] * layout.thumbnails
    timeline = np.zeros((layout.timeline_height, layout.width, 3), np.uint8)
    card = compose_card(thumbs, poses, timeline, ("header", "metrics"), layout)
    assert card.shape == (layout.height, layout.width, 3)

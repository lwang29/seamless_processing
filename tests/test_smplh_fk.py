from __future__ import annotations

import numpy as np

from scripts.smplh_fk import (
    COCO_BODY23_TO_SMPLH,
    COCO_LEFT_HAND_TO_SMPLH,
    COCO_RIGHT_HAND_TO_SMPLH,
    CameraHypothesis,
    SMPLH_JOINT_NAMES,
    project_hmr2_full_frame,
)


def test_joint_maps_cover_expected_groups() -> None:
    assert len(SMPLH_JOINT_NAMES) == 73
    assert COCO_BODY23_TO_SMPLH.shape == (23,)
    assert COCO_LEFT_HAND_TO_SMPLH.shape == (21,)
    assert COCO_RIGHT_HAND_TO_SMPLH.shape == (21,)
    assert COCO_LEFT_HAND_TO_SMPLH[0] == 20
    assert COCO_RIGHT_HAND_TO_SMPLH[0] == 21
    assert COCO_LEFT_HAND_TO_SMPLH[-1] == 67
    assert COCO_RIGHT_HAND_TO_SMPLH[-1] == 72
    for mapping in (
        COCO_BODY23_TO_SMPLH,
        COCO_LEFT_HAND_TO_SMPLH,
        COCO_RIGHT_HAND_TO_SMPLH,
    ):
        assert np.all((mapping >= 0) & (mapping < len(SMPLH_JOINT_NAMES)))


def test_camera_constants_match_stated_hypothesis() -> None:
    camera = CameraHypothesis()
    assert camera.weak_perspective_depth_numerator == 39.0625
    assert camera.focal_pixels(1080, 1920) == 37500.0
    assert camera.focal_pixels(2160, 2160) == 42187.5


def test_projection_uses_raster_centre_and_perspective_depth() -> None:
    camera = CameraHypothesis()
    joints = np.asarray(
        [[[0.0, 0.0, 10.0], [1.0, -2.0, 10.0], [1.0, -2.0, 20.0]]]
    )
    projected = project_hmr2_full_frame(joints, width=1080, height=1920, camera=camera)
    np.testing.assert_allclose(projected[0, 0], [540.0, 960.0])
    np.testing.assert_allclose(projected[0, 1], [4290.0, -6540.0])
    np.testing.assert_allclose(projected[0, 2], [2415.0, -2790.0])


def test_projection_rejects_bad_shapes_and_rasters() -> None:
    with np.testing.assert_raises(ValueError):
        project_hmr2_full_frame(np.zeros((2, 4)), 1080, 1920)
    with np.testing.assert_raises(ValueError):
        project_hmr2_full_frame(np.zeros((2, 3)), 0, 1920)

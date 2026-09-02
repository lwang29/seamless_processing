"""The NumPy forward kinematics has to be the same function ``smplx`` computes.

That equivalence is the licence for the whole scan: it is what makes a
full-corpus pass I/O bound instead of a 300-CPU-hour torch job. If it drifts,
every millimetre threshold in ``gates.py`` silently means something else.
"""

from __future__ import annotations

import numpy as np
import pytest

from seamless_curation.smplh_kinematics import (
    L_SHOULDER,
    L_WRIST,
    NECK,
    PELVIS,
    R_SHOULDER,
    R_WRIST,
    axis_angle_to_matrix,
    forward_kinematics,
    geodesic_speed,
    rest_joints,
    stack_pose,
    to_torso_frame,
    torso_basis,
)


@pytest.fixture
def pose() -> np.ndarray:
    rng = np.random.default_rng(11)
    frames = 24
    return stack_pose(
        rng.normal(0, 0.35, (frames, 21, 3)).astype(np.float32),
        rng.normal(0, 0.25, (frames, 15, 3)).astype(np.float32),
        rng.normal(0, 0.25, (frames, 15, 3)).astype(np.float32),
        global_orient=rng.normal(0, 0.6, (frames, 3)).astype(np.float32),
    )


def test_it_agrees_with_smplx_to_under_a_micrometre(pose, model_root) -> None:
    smplx = pytest.importorskip("smplx", reason="smplx not installed")
    torch = pytest.importorskip("torch", reason="torch not installed")

    frames = len(pose)
    model = smplx.create(
        str(model_root), model_type="smplh", gender="neutral", ext="npz",
        use_pca=False, flat_hand_mean=True, num_betas=16, batch_size=frames,
    ).eval()

    def tensor(block: np.ndarray) -> "torch.Tensor":
        return torch.from_numpy(np.ascontiguousarray(block.reshape(frames, -1).astype(np.float32)))

    with torch.inference_mode():
        result = model(
            betas=torch.zeros((frames, 16)),
            global_orient=tensor(pose[:, 0]),
            body_pose=tensor(pose[:, 1:22]),
            left_hand_pose=tensor(pose[:, 22:37]),
            right_hand_pose=tensor(pose[:, 37:52]),
            transl=torch.zeros((frames, 3)),
        )
    reference = result.joints.numpy()[:, :52]
    reference = reference - reference[:, :1]

    positions, _globals, _locals = forward_kinematics(pose, str(model_root))
    assert np.abs(reference - positions).max() * 1000 < 0.001  # millimetres


def test_the_torso_frame_is_orthonormal_and_right_handed(pose, model_root) -> None:
    joints, _globals, _locals = forward_kinematics(pose, str(model_root))
    _origin, basis = torso_basis(joints)

    identity = np.einsum("fij,fkj->fik", basis, basis)
    # The 1e-9 tolerance is the normalisation epsilon in torso_basis, not noise.
    assert np.abs(identity - np.eye(3)).max() < 1e-7
    assert np.linalg.det(basis).min() > 0.999


def test_a_whole_body_rotation_leaves_torso_coordinates_unchanged(model_root) -> None:
    """The point of the frame: turning does not register as arm motion."""

    rng = np.random.default_rng(3)
    body = rng.normal(0, 0.3, (8, 21, 3)).astype(np.float32)
    hands = rng.normal(0, 0.2, (8, 15, 3)).astype(np.float32)

    still = stack_pose(body, hands, hands, global_orient=np.zeros((8, 3), np.float32))
    turned = stack_pose(
        body, hands, hands,
        global_orient=np.tile(np.array([0.0, 1.1, 0.0], np.float32), (8, 1)),
    )
    a, _, _ = forward_kinematics(still, str(model_root))
    b, _, _ = forward_kinematics(turned, str(model_root))

    left = to_torso_frame(a, (L_WRIST, R_WRIST))
    right = to_torso_frame(b, (L_WRIST, R_WRIST))
    assert np.abs(left - right).max() < 1e-6


def test_torso_coordinates_are_millimetres(pose, model_root) -> None:
    joints, _globals, _locals = forward_kinematics(pose, str(model_root))
    shoulders = to_torso_frame(joints, (L_SHOULDER, R_SHOULDER))
    separation = np.linalg.norm(shoulders[:, 0] - shoulders[:, 1], axis=1)
    assert 250 < separation.mean() < 450  # a human shoulder width, in mm


def test_the_rest_skeleton_is_a_constant_of_the_release(model_root) -> None:
    """Betas are all zero, so millimetres compare across participants."""

    joints, parents = rest_joints(str(model_root))
    assert joints.shape == (52, 3) and parents[0] == -1
    assert np.linalg.norm(joints[L_SHOULDER] - joints[R_SHOULDER]) * 1000 == pytest.approx(347.6, abs=1)
    assert np.linalg.norm(joints[NECK] - joints[PELVIS]) * 1000 == pytest.approx(512.7, abs=1)


def test_axis_angle_round_trips_and_handles_a_zero_rotation() -> None:
    rng = np.random.default_rng(5)
    vectors = rng.normal(0, 1.0, (7, 3))
    vectors[0] = 0.0
    matrices = axis_angle_to_matrix(vectors)
    assert np.abs(matrices[0] - np.eye(3)).max() < 1e-12
    identity = np.einsum("fij,fkj->fik", matrices, matrices)
    assert np.abs(identity - np.eye(3)).max() < 1e-9


def test_geodesic_speed_is_composed_not_subtracted() -> None:
    """Naive axis-angle deltas exceed 100 rad/s on 43.5% of dev spans."""

    frames, fps = 40, 30.0
    angle = np.linspace(0, 2 * np.pi, frames)          # wraps through +/-pi
    vectors = np.stack([np.zeros(frames), np.zeros(frames), angle], axis=1)[:, None, :]
    speed = geodesic_speed(axis_angle_to_matrix(vectors), fps)

    expected = (2 * np.pi / (frames - 1)) * fps
    assert np.abs(speed - expected).max() < 1e-6
    naive = np.abs(np.diff(angle, prepend=angle[0])) * fps
    assert naive.max() < 1e3  # this synthetic case is smooth; the real one is not
    assert speed.shape == (frames, 1)

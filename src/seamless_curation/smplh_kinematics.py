"""SMPL-H forward kinematics without PyTorch, and the upper-body frames we score in.

The release stores axis-angle pose only: 1 global orient, 21 body joints, and
15+15 hand joints. Every project convention fixes the shape (neutral model, all
sixteen betas zero), so the rest-pose joint locations are a constant and forward
kinematics is 52 chained rigid transforms — no blend skinning, no vertices, no
GPU. That matters because the scan touches every file in the corpus:
:func:`forward_kinematics` costs ~0.12 s for a 6,900-frame file in pure NumPy
against ~2 s through ``smplx`` on CPU, and it agrees with ``smplx`` to under a
micrometre (see ``tests/test_smplh_kinematics.py``).

Two frames of reference are used, and the difference between them is the whole
point of the gesture measure:

``pelvis frame``
    ``global_orient`` is dropped, so the participant's whole-body turn and the
    camera's view of them are removed. Torso lean and arm motion both survive.

``torso frame``
    Origin at the shoulder midpoint, axes built from the shoulder line and the
    pelvis-to-neck axis. A wrist coordinate here is *arm articulation* only:
    walking, swaying and leaning move the origin and the axes with the body, so
    they contribute nothing. This is the frame the PI's "global body movement
    must not count as gesturing" requirement asks for.

Because all betas are zero the skeleton is metrically identical in every file,
so millimetre thresholds are directly comparable across participants without
any per-subject normalisation.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np

# SMPL-H kinematic tree indices. Names come from scripts-era SMPLH_JOINT_NAMES
# and are re-stated here so this module stands alone.
PELVIS = 0
SPINE1, SPINE2, SPINE3 = 3, 6, 9
NECK, HEAD = 12, 15
L_COLLAR, R_COLLAR = 13, 14
L_SHOULDER, R_SHOULDER = 16, 17
L_ELBOW, R_ELBOW = 18, 19
L_WRIST, R_WRIST = 20, 21
# Articulated finger joints: 22..36 left, 37..51 right.
L_HAND_JOINTS = tuple(range(22, 37))
R_HAND_JOINTS = tuple(range(37, 52))

#: Joints ViBES trains on: torso, neck, head, collars, shoulders, arms, wrists,
#: and both articulated hands. The legs are deliberately absent — lower-body
#: quality is out of scope by instruction.
UPPER_BODY_JOINTS: tuple[int, ...] = (
    PELVIS, SPINE1, SPINE2, SPINE3, NECK, HEAD, L_COLLAR, R_COLLAR,
    L_SHOULDER, R_SHOULDER, L_ELBOW, R_ELBOW, L_WRIST, R_WRIST,
) + L_HAND_JOINTS + R_HAND_JOINTS

#: Indices into the released ``smplh:body_pose`` array (which excludes the root),
#: i.e. ``body_pose[:, UPPER_BODY_POSE_INDEX]`` are the upper-body rotations.
UPPER_BODY_POSE_INDEX: tuple[int, ...] = tuple(
    j - 1 for j in (SPINE1, SPINE2, SPINE3, NECK, HEAD, L_COLLAR, R_COLLAR,
                    L_SHOULDER, R_SHOULDER, L_ELBOW, R_ELBOW, L_WRIST, R_WRIST)
)

MODEL_FILENAME = "SMPLH_NEUTRAL.npz"


@lru_cache(maxsize=4)
def rest_joints(model_root: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """Rest-pose joint locations (52, 3) in metres and the parent index array.

    ``model_root`` is the directory holding ``SMPLH_NEUTRAL.npz`` — either
    ``model_files`` or ``model_files/smplh``. The asset is research-licensed: it
    is read in place and never copied.
    """

    root = Path(model_root)
    candidates = [root / MODEL_FILENAME, root / "smplh" / MODEL_FILENAME]
    for path in candidates:
        if path.exists():
            break
    else:
        raise FileNotFoundError(f"{MODEL_FILENAME} not found under {root}")
    with np.load(path, allow_pickle=True) as model:
        joints = np.asarray(model["J_regressor"] @ model["v_template"], dtype=np.float64)
        parents = np.asarray(model["kintree_table"][0], dtype=np.int64).copy()
    if joints.shape != (52, 3):
        raise ValueError(f"expected 52 rest joints, got {joints.shape}")
    parents[0] = -1
    return joints, parents


def axis_angle_to_matrix(vectors: np.ndarray) -> np.ndarray:
    """Rodrigues formula on a ``(..., 3)`` stack, returning ``(..., 3, 3)``.

    Axis-angle is composed as rotations, never subtracted: the release does not
    wrap its vectors canonically and naive differences produce ~2-pi spikes.
    """

    aa = np.asarray(vectors, dtype=np.float64)
    theta = np.linalg.norm(aa, axis=-1, keepdims=True)
    axis = np.where(theta > 1e-8, aa / np.maximum(theta, 1e-12), 0.0)
    skew = np.zeros(aa.shape[:-1] + (3, 3), dtype=np.float64)
    skew[..., 0, 1] = -axis[..., 2]
    skew[..., 0, 2] = axis[..., 1]
    skew[..., 1, 0] = axis[..., 2]
    skew[..., 1, 2] = -axis[..., 0]
    skew[..., 2, 0] = -axis[..., 1]
    skew[..., 2, 1] = axis[..., 0]
    eye = np.broadcast_to(np.eye(3), skew.shape).copy()
    sin = np.sin(theta)[..., None]
    cos = np.cos(theta)[..., None]
    return eye + sin * skew + (1.0 - cos) * (skew @ skew)


def stack_pose(
    body_pose: np.ndarray,
    left_hand_pose: np.ndarray,
    right_hand_pose: np.ndarray,
    global_orient: np.ndarray | None = None,
) -> np.ndarray:
    """Assemble the released arrays into one ``(frames, 52, 3)`` axis-angle stack.

    ``global_orient=None`` zeroes the root rotation, which is what puts the
    result in the pelvis frame.
    """

    frames = len(body_pose)
    root = (
        np.zeros((frames, 1, 3), dtype=np.float64)
        if global_orient is None
        else np.asarray(global_orient, dtype=np.float64).reshape(frames, 1, 3)
    )
    return np.concatenate(
        [
            root,
            np.asarray(body_pose, dtype=np.float64).reshape(frames, 21, 3),
            np.asarray(left_hand_pose, dtype=np.float64).reshape(frames, 15, 3),
            np.asarray(right_hand_pose, dtype=np.float64).reshape(frames, 15, 3),
        ],
        axis=1,
    )


def forward_kinematics(
    pose: np.ndarray, model_root: str | Path
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Positions (frames, 52, 3) in metres, and global and local rotations.

    The pelvis sits at the origin; add ``smplh:translation`` if camera-frame
    positions are wanted, which nothing in the gesture measure does.

    Both rotation stacks are returned because they answer different questions
    and confusing them is easy. A *global* rotation carries the whole chain
    above the joint, so the global rotation of a finger contains the shoulder
    and the elbow: a rigid, motionless hand on a swinging arm has a large global
    finger rotation rate and a zero local one. Finger articulation must be read
    from the local stack.
    """

    rest, parents = rest_joints(model_root)
    local = axis_angle_to_matrix(pose)
    frames = len(pose)
    positions = np.zeros((frames, 52, 3), dtype=np.float64)
    globals_ = np.empty((frames, 52, 3, 3), dtype=np.float64)
    globals_[:, 0] = local[:, 0]
    for joint in range(1, 52):
        parent = parents[joint]
        globals_[:, joint] = globals_[:, parent] @ local[:, joint]
        offset = rest[joint] - rest[parent]
        positions[:, joint] = positions[:, parent] + globals_[:, parent] @ offset
    return positions, globals_, local


def torso_basis(joints: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Shoulder-midpoint origin and a right-handed torso rotation, per frame.

    Returns ``(origin, basis)`` where ``basis @ (p - origin)`` maps a pelvis-frame
    point into torso coordinates: ``+x`` toward the participant's left shoulder,
    ``+y`` up the spine, ``+z`` completing the frame.
    """

    left = joints[:, L_SHOULDER]
    right = joints[:, R_SHOULDER]
    origin = 0.5 * (left + right)
    x = left - right
    x = x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-9)
    up = joints[:, NECK] - joints[:, PELVIS]
    up = up - (up * x).sum(1, keepdims=True) * x
    up = up / (np.linalg.norm(up, axis=1, keepdims=True) + 1e-9)
    z = np.cross(x, up)
    basis = np.stack([x, up, z], axis=1)
    return origin, basis


def to_torso_frame(joints: np.ndarray, indices: tuple[int, ...]) -> np.ndarray:
    """Selected joints expressed in the torso frame, in **millimetres**."""

    origin, basis = torso_basis(joints)
    relative = joints[:, list(indices)] - origin[:, None, :]
    return np.einsum("fij,fkj->fki", basis, relative) * 1000.0


def geodesic_speed(rotations: np.ndarray, fps: float) -> np.ndarray:
    """Per-frame angular speed in rad/s from a ``(frames, J, 3, 3)`` stack.

    Composition, not subtraction: ``angle(R_t^T R_{t+1})``. Result is
    ``(frames, J)`` with the first row repeated so the length is preserved.
    """

    if len(rotations) < 2:
        return np.zeros(rotations.shape[:2], dtype=np.float64)
    relative = np.einsum("fjab,fjac->fjbc", rotations[:-1], rotations[1:])
    trace = np.trace(relative, axis1=-2, axis2=-1)
    angle = np.arccos(np.clip((trace - 1.0) * 0.5, -1.0, 1.0))
    return np.concatenate([angle[:1], angle], axis=0) * fps

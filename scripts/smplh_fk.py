"""SMPL-H FK and HMR-style full-frame projection primitives.

The body-model import is deliberately lazy: pure camera/joint-map unit tests do
not require PyTorch or the research-licensed model.  The model asset is read in
place and is never copied by this module.

Observed payload facts and project choices are kept separate:

* Observed: the release stores 21 body and 15+15 hand axis-angle vectors.
* Choice: neutral SMPL-H, 16 all-zero betas, and no hand PCA.
* Hypothesis under test: translation is HMR 2.0 crop-camera translation after
  conversion to the full image, projected with
  ``f_px = 5000 / 256 * max(width, height)`` and raster-centre principal point.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np


SMPLH_JOINT_NAMES: tuple[str, ...] = (
    "pelvis", "left_hip", "right_hip", "spine1", "left_knee",
    "right_knee", "spine2", "left_ankle", "right_ankle", "spine3",
    "left_foot", "right_foot", "neck", "left_collar", "right_collar",
    "head", "left_shoulder", "right_shoulder", "left_elbow",
    "right_elbow", "left_wrist", "right_wrist", "left_index1",
    "left_index2", "left_index3", "left_middle1", "left_middle2",
    "left_middle3", "left_pinky1", "left_pinky2", "left_pinky3",
    "left_ring1", "left_ring2", "left_ring3", "left_thumb1",
    "left_thumb2", "left_thumb3", "right_index1", "right_index2",
    "right_index3", "right_middle1", "right_middle2", "right_middle3",
    "right_pinky1", "right_pinky2", "right_pinky3", "right_ring1",
    "right_ring2", "right_ring3", "right_thumb1", "right_thumb2",
    "right_thumb3", "nose", "right_eye", "left_eye", "right_ear",
    "left_ear", "left_big_toe", "left_small_toe", "left_heel",
    "right_big_toe", "right_small_toe", "right_heel", "left_thumb",
    "left_index", "left_middle", "left_ring", "left_pinky", "right_thumb",
    "right_index", "right_middle", "right_ring", "right_pinky",
)

# COCO-WholeBody indices 0:23 (17 body + 6 feet) in released order.
COCO_BODY23_TO_SMPLH = np.asarray(
    [52, 54, 53, 56, 55, 16, 17, 18, 19, 20, 21, 1, 2, 4, 5, 7, 8,
     57, 58, 59, 60, 61, 62],
    dtype=np.int64,
)

# Each COCO hand is wrist, then four thumb joints/tip, index, middle, ring,
# pinky.  SMPL-H's kinematic order differs, hence these explicit maps.
COCO_LEFT_HAND_TO_SMPLH = np.asarray(
    [20, 34, 35, 36, 63, 22, 23, 24, 64, 25, 26, 27, 65,
     31, 32, 33, 66, 28, 29, 30, 67],
    dtype=np.int64,
)
COCO_RIGHT_HAND_TO_SMPLH = np.asarray(
    [21, 49, 50, 51, 68, 37, 38, 39, 69, 40, 41, 42, 70,
     46, 47, 48, 71, 43, 44, 45, 72],
    dtype=np.int64,
)


@dataclass(frozen=True)
class CameraHypothesis:
    """Explicit parameters of the HMR full-frame camera hypothesis."""

    base_focal_length: float = 5000.0
    model_input_size: float = 256.0

    @property
    def weak_perspective_depth_numerator(self) -> float:
        return 2.0 * self.base_focal_length / self.model_input_size

    def focal_pixels(self, width: int, height: int) -> float:
        return self.base_focal_length / self.model_input_size * max(width, height)


def project_hmr2_full_frame(
    joints_camera_m: np.ndarray,
    width: int,
    height: int,
    camera: CameraHypothesis = CameraHypothesis(),
) -> np.ndarray:
    """Project camera-frame joints under the camera hypothesis.

    Parameters
    ----------
    joints_camera_m:
        Array ``(..., J, 3)``.  The unit cancels in perspective division, but
        SMPL-H returns metres and the supplied translation is tested as having
        the same coordinate scale.
    width, height:
        Native released-video raster.  No normalization or resizing occurs.
    """

    joints = np.asarray(joints_camera_m, dtype=np.float64)
    if joints.shape[-1] != 3:
        raise ValueError(f"expected (..., J, 3), got {joints.shape}")
    if width <= 0 or height <= 0:
        raise ValueError("width and height must be positive")
    depth = joints[..., 2]
    focal = camera.focal_pixels(width, height)
    with np.errstate(divide="ignore", invalid="ignore"):
        x = focal * joints[..., 0] / depth + width / 2.0
        y = focal * joints[..., 1] / depth + height / 2.0
    return np.stack((x, y), axis=-1)


def create_neutral_smplh(
    model_root: str | Path,
    *,
    flat_hand_mean: bool,
    batch_size: int,
    device: str = "cpu",
) -> Any:
    """Create the exact project-convention model requested for M-4."""

    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    try:
        import smplx
    except ImportError as exc:  # pragma: no cover - depends on cluster env
        raise RuntimeError(
            "smplx is unavailable; use the staged ViBES environment"
        ) from exc

    model = smplx.create(
        str(model_root),
        model_type="smplh",
        gender="neutral",
        ext="npz",
        use_pca=False,
        flat_hand_mean=flat_hand_mean,
        num_betas=16,
        batch_size=batch_size,
    )
    return model.to(device).eval()


def forward_smplh_joints(
    model: Any,
    payload: Mapping[str, np.ndarray],
    *,
    device: str = "cpu",
    include_translation: bool = True,
    return_vertices: bool = False,
) -> tuple[np.ndarray, np.ndarray] | tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run FK and return camera-frame and pelvis-relative joints in metres.

    With ``return_vertices`` the camera-frame mesh (6890 x 3, metres) comes back
    as a third element. The forward pass itself costs nothing extra: ``smplx``
    runs ``lbs`` and produces the vertices regardless, and ``return_verts=False``
    only discards them. FM2 tests the surface rather than joint centres; see
    :func:`seamless_curation.v00_detectors.in_frame_violations` for what that is
    actually worth, which is less than the raw joint-to-surface gap suggests.
    """

    try:
        import torch
    except ImportError as exc:  # pragma: no cover - depends on cluster env
        raise RuntimeError("PyTorch is unavailable") from exc

    required = (
        "smplh:global_orient",
        "smplh:body_pose",
        "smplh:left_hand_pose",
        "smplh:right_hand_pose",
        "smplh:translation",
    )
    missing = [key for key in required if key not in payload]
    if missing:
        raise KeyError(f"missing SMPL-H arrays: {missing}")

    batch = len(payload["smplh:global_orient"])
    if batch < 1:
        raise ValueError("empty FK batch")

    def tensor(key: str) -> Any:
        array = np.asarray(payload[key], dtype=np.float32).reshape(batch, -1)
        return torch.from_numpy(np.ascontiguousarray(array)).to(device)

    translation = tensor("smplh:translation")
    if not include_translation:
        translation = torch.zeros_like(translation)

    with torch.inference_mode():
        result = model(
            betas=torch.zeros((batch, 16), dtype=torch.float32, device=device),
            global_orient=tensor("smplh:global_orient"),
            body_pose=tensor("smplh:body_pose"),
            left_hand_pose=tensor("smplh:left_hand_pose"),
            right_hand_pose=tensor("smplh:right_hand_pose"),
            transl=translation,
            return_verts=return_vertices,
        )
    joints = result.joints.detach().cpu().numpy().astype(np.float32, copy=False)
    if joints.shape[1:] != (73, 3):
        raise RuntimeError(f"expected 73 SMPL-H joints, got {joints.shape}")
    root_relative = joints - joints[:, :1]
    if not return_vertices:
        return joints, root_relative
    vertices = result.vertices
    if vertices is None:
        raise RuntimeError("SMPL-H forward pass returned no vertices")
    return joints, root_relative, vertices.detach().cpu().numpy().astype(np.float32, copy=False)

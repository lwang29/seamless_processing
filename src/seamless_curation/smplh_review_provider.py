"""Adapter from acceptance-tested M-4 primitives to review joint sequences."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import numpy as np

from .smplh_fk import (
    create_neutral_smplh,
    forward_smplh_joints,
    project_hmr2_full_frame,
)

from .review_renderer import JointSequence


# SMPL-H kinematic body (including its ankle-attached foot joints), full
# articulated hands, and fingertips. The auxiliary face landmarks 52:57 and
# toe/heel landmarks 57:63 are not drawn. M-4 found structured errors in the
# latter; silently displaying them as equally validated would be misleading.
DISPLAY_RAW_INDICES: tuple[int, ...] = tuple(range(52)) + tuple(range(63, 73))

RAW_EDGES: tuple[tuple[int, int], ...] = (
    # Pelvis, legs, spine, head, and arms.
    (0, 1), (0, 2), (0, 3), (1, 4), (2, 5), (3, 6),
    (4, 7), (5, 8), (6, 9), (7, 10), (8, 11), (9, 12),
    (12, 13), (12, 14), (12, 15), (13, 16), (14, 17),
    (16, 18), (17, 19), (18, 20), (19, 21),
    # Left: index, middle, pinky, ring, thumb plus accepted fingertips.
    (20, 22), (22, 23), (23, 24), (24, 64),
    (20, 25), (25, 26), (26, 27), (27, 65),
    (20, 28), (28, 29), (29, 30), (30, 67),
    (20, 31), (31, 32), (32, 33), (33, 66),
    (20, 34), (34, 35), (35, 36), (36, 63),
    # Right hand.
    (21, 37), (37, 38), (38, 39), (39, 69),
    (21, 40), (40, 41), (41, 42), (42, 70),
    (21, 43), (43, 44), (44, 45), (45, 72),
    (21, 46), (46, 47), (47, 48), (48, 71),
    (21, 49), (49, 50), (50, 51), (51, 68),
)

_RAW_TO_DISPLAY = {raw: displayed for displayed, raw in enumerate(DISPLAY_RAW_INDICES)}
DISPLAY_EDGES: tuple[tuple[int, int], ...] = tuple(
    (_RAW_TO_DISPLAY[first], _RAW_TO_DISPLAY[second])
    for first, second in RAW_EDGES
)
DISPLAY_GROUPS: tuple[str, ...] = tuple(
    "body" if raw <= 21 else "left_hand" if raw in set(range(22, 37)) | set(range(63, 68)) else "right_hand"
    for raw in DISPLAY_RAW_INDICES
)


class ValidatedSmplhReviewProvider:
    """Neutral, beta-zero, full-axis-angle provider accepted by M-4."""

    name = "smplh_m4_v1_neutral_flattrue_hmr2_fullframe"

    def __init__(
        self,
        model_root: str | Path,
        *,
        flat_hand_mean: bool = True,
        batch_size: int = 64,
        device: str = "cpu",
    ) -> None:
        if flat_hand_mean is not True:
            raise ValueError("only flat_hand_mean=True passed the M-4 acceptance comparison")
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        self.model_root = Path(model_root)
        self.batch_size = int(batch_size)
        self.device = device
        self._model: Any | None = None

    def _get_model(self) -> Any:
        if self._model is None:
            self._model = create_neutral_smplh(
                self.model_root,
                flat_hand_mean=True,
                batch_size=self.batch_size,
                device=self.device,
            )
        return self._model

    def load(
        self,
        source_base: Path,
        start_frame: int,
        end_frame: int,
        raster_size_wh: tuple[int, int],
    ) -> JointSequence:
        if not 0 <= start_frame < end_frame:
            raise ValueError(f"invalid FK interval [{start_frame}, {end_frame})")
        required = (
            "smplh:global_orient", "smplh:body_pose", "smplh:left_hand_pose",
            "smplh:right_hand_pose", "smplh:translation",
        )
        with np.load(source_base.with_suffix(".npz"), allow_pickle=False) as archive:
            missing = [key for key in required if key not in archive.files]
            if missing:
                raise KeyError(f"missing SMPL-H arrays: {missing}")
            payload = {
                key: np.asarray(archive[key][start_frame:end_frame], dtype=np.float32)
                for key in required
            }
        frames = end_frame - start_frame
        if any(len(array) != frames for array in payload.values()):
            raise ValueError("FK interval extends beyond one or more SMPL-H arrays")

        camera_chunks: list[np.ndarray] = []
        root_chunks: list[np.ndarray] = []
        model = self._get_model()
        for offset in range(0, frames, self.batch_size):
            stop = min(frames, offset + self.batch_size)
            real_count = stop - offset
            chunk: dict[str, np.ndarray] = {}
            for key, array in payload.items():
                values = array[offset:stop]
                if real_count < self.batch_size:
                    padding = np.repeat(values[-1:], self.batch_size - real_count, axis=0)
                    values = np.concatenate((values, padding), axis=0)
                chunk[key] = values
            camera, root = forward_smplh_joints(
                model, chunk, device=self.device, include_translation=True,
            )
            camera_chunks.append(camera[:real_count])
            root_chunks.append(root[:real_count])

        camera_all = np.concatenate(camera_chunks, axis=0)
        root_all = np.concatenate(root_chunks, axis=0)
        width, height = raster_size_wh
        projected_all = project_hmr2_full_frame(camera_all, width=width, height=height)
        indices = np.asarray(DISPLAY_RAW_INDICES, dtype=np.int64)
        projected = projected_all[:, indices].astype(np.float32, copy=False)
        root_relative = root_all[:, indices].astype(np.float32, copy=False)
        return JointSequence(
            projected_xy_px=projected,
            root_relative_xyz_m=root_relative,
            edges=DISPLAY_EDGES,
            groups=DISPLAY_GROUPS,
            status=(
                "M-4 accepted: neutral, betas=0, use_pca=False, "
                "flat_hand_mean=True; auxiliary face/toe/heel points not drawn"
            ),
        )


def create_provider(settings: Mapping[str, Any]) -> ValidatedSmplhReviewProvider:
    """Trusted ``module:function`` factory used by review-gallery YAML."""

    allowed = {"model_root", "flat_hand_mean", "batch_size", "device"}
    unknown = set(settings) - allowed
    if unknown:
        raise ValueError(f"unknown SMPL-H review-provider settings: {sorted(unknown)}")
    if "model_root" not in settings:
        raise ValueError("model_root is required")
    return ValidatedSmplhReviewProvider(**dict(settings))


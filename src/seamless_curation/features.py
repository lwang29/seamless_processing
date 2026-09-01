"""Small, threshold-free measurements used by the Session 1 harness.

The acceleration measurement is intentionally limited to the released SMPL-H
root translation.  It is not joint acceleration from forward kinematics.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


LEFT_HAND_SLICE = slice(91, 112)
RIGHT_HAND_SLICE = slice(112, 133)
BODY_WRIST_INDICES = (9, 10)

# Session 1 found a one-frame optimistic-valid edge at the terminal black cut in
# `invalid_02.mp4`: the released mask calls a frame valid that visibly is not.
# The guard band widens each invalid run symmetrically so a window abutting a
# failure is not scored as clean. Three frames is the documented default, not a
# measured optimum.
GUARD_BAND_DEFAULT_FRAMES = 3


@dataclass(frozen=True)
class Measurement:
    value: float
    status: str


def as_binary_mask(value: Any) -> tuple[np.ndarray | None, str]:
    """Normalize a released flag without inventing a validity threshold.

    Float flags are required to be exactly 0 or 1.  Non-finite and non-binary
    entries are represented as invalid, and the status records that condition.
    """

    if value is None:
        return None, "missing"
    array = np.asarray(value)
    if array.ndim == 0:
        return None, "invalid_shape"
    if array.ndim > 1:
        flattened = array.reshape(array.shape[0], -1)
        if flattened.shape[1] != 1:
            return None, "invalid_shape"
        array = flattened[:, 0]
    if array.dtype.kind not in "bifu":
        return None, f"invalid_dtype:{array.dtype}"

    finite = np.isfinite(array)
    binary = (array == 0) | (array == 1)
    mask = finite & (array == 1)
    if not finite.all():
        return mask.astype(bool, copy=False), "non_finite_entries_treated_invalid"
    if not binary.all():
        return mask.astype(bool, copy=False), "non_binary_entries_treated_invalid"
    return mask.astype(bool, copy=False), "ok"


def valid_frac_all(
    smplh_mask: np.ndarray | None,
    movement_mask: np.ndarray | None,
    box_mask: np.ndarray | None,
) -> Measurement:
    """Fraction for the AND of all three masks, or NaN if any is absent."""

    names = ("smplh", "movement", "box")
    masks = (smplh_mask, movement_mask, box_mask)
    missing = [name for name, mask in zip(names, masks) if mask is None]
    if missing:
        return Measurement(float("nan"), "missing_masks:" + ",".join(missing))
    lengths = [len(np.asarray(mask)) for mask in masks]
    if len(set(lengths)) != 1:
        return Measurement(float("nan"), "mask_length_mismatch")
    if lengths[0] == 0:
        return Measurement(float("nan"), "empty_window")
    combined = np.asarray(smplh_mask, dtype=bool)
    combined = combined & np.asarray(movement_mask, dtype=bool)
    combined = combined & np.asarray(box_mask, dtype=bool)
    return Measurement(float(combined.mean()), "ok")


def valid_frac(mask: np.ndarray | None, *, name: str) -> Measurement:
    """Fraction of valid frames for exactly one released mask.

    Splitting the masks is what stops one absent modality from erasing the
    other two.  A missing mask stays NaN with an explicit status; it is never
    silently treated as valid.
    """

    if mask is None:
        return Measurement(float("nan"), f"missing_mask:{name}")
    array = np.asarray(mask, dtype=bool).reshape(-1)
    if array.size == 0:
        return Measurement(float("nan"), f"empty_window:{name}")
    return Measurement(float(array.mean()), "ok")


def valid_frac_conjunction(
    masks: dict[str, np.ndarray | None],
    *,
    required: tuple[str, ...],
) -> Measurement:
    """Fraction where every mask named in ``required`` is valid.

    Only the masks the caller declares necessary participate.  Optional
    modalities must be passed through :func:`valid_frac` instead, so that an
    unavailable movement mask cannot reduce a body-and-hands measurement.
    """

    selected = [masks.get(name) for name in required]
    missing = [name for name, mask in zip(required, selected) if mask is None]
    if missing:
        return Measurement(float("nan"), "missing_masks:" + ",".join(sorted(missing)))
    arrays = [np.asarray(mask, dtype=bool).reshape(-1) for mask in selected]
    lengths = {array.size for array in arrays}
    if len(lengths) != 1:
        return Measurement(float("nan"), "mask_length_mismatch")
    if arrays[0].size == 0:
        return Measurement(float("nan"), "empty_window")
    combined = arrays[0].copy()
    for array in arrays[1:]:
        combined &= array
    return Measurement(float(combined.mean()), "ok")


def guard_invalid_runs(mask: np.ndarray | None, guard_frames: int) -> np.ndarray | None:
    """Symmetrically widen every invalid run by ``guard_frames`` frames.

    Apply this to a whole-file mask before slicing windows.  Guarding a slice
    in isolation cannot see a failure that begins one frame outside it.
    """

    if guard_frames < 0:
        raise ValueError("guard_frames must be nonnegative")
    if mask is None:
        return None
    valid = np.asarray(mask, dtype=bool).reshape(-1)
    if guard_frames == 0 or valid.size == 0:
        return valid.copy()
    invalid = ~valid
    widened = invalid.copy()
    for shift in range(1, guard_frames + 1):
        widened[shift:] |= invalid[:-shift]
        widened[:-shift] |= invalid[shift:]
    return ~widened


def root_translation_accel(
    translation: np.ndarray | None,
    smplh_mask: np.ndarray | None,
    scale_to_nominal_mm: float,
) -> Measurement:
    """Mean root-translation second difference over valid triplets.

    The result is only nominally millimetres/frame^2.  The caller must retain
    the explicit basis and physical-units-unverified provenance columns.
    """

    if translation is None:
        return Measurement(float("nan"), "missing_translation")
    if smplh_mask is None:
        return Measurement(float("nan"), "missing_smplh_mask")
    points = np.asarray(translation)
    mask = np.asarray(smplh_mask, dtype=bool).reshape(-1)
    if points.ndim != 2 or points.shape[1] != 3:
        return Measurement(float("nan"), "invalid_translation_shape")
    if len(points) != len(mask):
        return Measurement(float("nan"), "translation_mask_length_mismatch")
    if len(points) < 3:
        return Measurement(float("nan"), "insufficient_frames")
    if not np.isfinite(scale_to_nominal_mm) or scale_to_nominal_mm <= 0:
        return Measurement(float("nan"), "invalid_translation_scale")

    valid_triplets = mask[:-2] & mask[1:-1] & mask[2:]
    valid_triplets &= np.isfinite(points[:-2]).all(axis=1)
    valid_triplets &= np.isfinite(points[1:-1]).all(axis=1)
    valid_triplets &= np.isfinite(points[2:]).all(axis=1)
    if not valid_triplets.any():
        return Measurement(float("nan"), "no_valid_triplets")
    scaled = points.astype(np.float64, copy=False) * scale_to_nominal_mm
    second_difference = scaled[2:] - 2.0 * scaled[1:-1] + scaled[:-2]
    magnitudes = np.linalg.norm(second_difference[valid_triplets], axis=1)
    return Measurement(float(magnitudes.mean()), "ok")


def wrist_speed_p90(
    keypoints: np.ndarray | None,
    box_mask: np.ndarray | None,
) -> Measurement:
    """90th percentile body-wrist displacement in released pixels/frame.

    COCO-WholeBody body indices 9 and 10 are used.  Confidence is deliberately
    not thresholded: only finite XY coordinates and adjacent box-valid frames
    are eligible.
    """

    if keypoints is None:
        return Measurement(float("nan"), "missing_keypoints")
    if box_mask is None:
        return Measurement(float("nan"), "missing_box_mask")
    points = np.asarray(keypoints)
    mask = np.asarray(box_mask, dtype=bool).reshape(-1)
    if points.ndim != 3 or points.shape[1] <= max(BODY_WRIST_INDICES) or points.shape[2] < 2:
        return Measurement(float("nan"), "invalid_keypoint_shape")
    if len(points) != len(mask):
        return Measurement(float("nan"), "keypoint_mask_length_mismatch")
    if len(points) < 2:
        return Measurement(float("nan"), "insufficient_frames")

    adjacent_valid = mask[:-1] & mask[1:]
    all_speeds: list[np.ndarray] = []
    for index in BODY_WRIST_INDICES:
        xy = points[:, index, :2].astype(np.float64, copy=False)
        finite_pairs = np.isfinite(xy[:-1]).all(axis=1) & np.isfinite(xy[1:]).all(axis=1)
        eligible = adjacent_valid & finite_pairs
        if eligible.any():
            speed = np.linalg.norm(xy[1:] - xy[:-1], axis=1)
            all_speeds.append(speed[eligible])
    if not all_speeds:
        return Measurement(float("nan"), "no_valid_adjacent_wrist_frames")
    return Measurement(float(np.percentile(np.concatenate(all_speeds), 90)), "ok")


def duration_mismatch_s(n_frames: int, frame_rate: float, wav_duration_s: float) -> float:
    """Absolute annotation-frame versus WAV-duration mismatch."""

    if n_frames < 0 or frame_rate <= 0 or not np.isfinite(wav_duration_s):
        return float("nan")
    return float(abs(n_frames / frame_rate - wav_duration_s))


def zero_velocity_run_max(positions: np.ndarray) -> int:
    """Maximum run of exact zero-velocity steps (no empirical epsilon)."""

    points = np.asarray(positions)
    if points.ndim < 2 or len(points) < 2:
        return 0
    flattened = points.reshape(len(points), -1)
    pair_finite = np.isfinite(flattened[:-1]).all(axis=1)
    pair_finite &= np.isfinite(flattened[1:]).all(axis=1)
    zero_steps = pair_finite & np.all(flattened[1:] == flattened[:-1], axis=1)
    best = current = 0
    for is_zero in zero_steps:
        current = current + 1 if is_zero else 0
        best = max(best, current)
    return int(best)


def hand_availability(keypoints: np.ndarray | None, hand: str) -> Measurement:
    """Harness-grade per-hand availability with an explicit status.

    A frame counts as available when the hand's whole 21-point XY block is
    finite and not an exact all-zero blank.  Session 1 showed that blanking a
    hand changes no existing signal, so hands need their own column rather than
    being inferred from body validity.  Confidence is not thresholded here.
    """

    if hand not in ("left", "right"):
        raise ValueError("hand must be 'left' or 'right'")
    if keypoints is None:
        return Measurement(float("nan"), "missing_keypoints")
    points = np.asarray(keypoints)
    if points.ndim != 3 or points.shape[1] < 133 or points.shape[2] < 2:
        return Measurement(float("nan"), f"invalid_keypoint_shape:{points.shape}")
    if len(points) == 0:
        return Measurement(float("nan"), "empty_window")
    selected = points[:, LEFT_HAND_SLICE if hand == "left" else RIGHT_HAND_SLICE, :2]
    finite = np.isfinite(selected).all(axis=(1, 2))
    nonblank = np.any(selected != 0, axis=(1, 2))
    return Measurement(float((finite & nonblank).mean()), "ok")


def hand_availability_frac(keypoints: np.ndarray, hand: str) -> float:
    """Fraction of frames whose complete hand XY block is finite and nonblank.

    A blank block means every released XY value is exactly zero.  Confidence is
    not thresholded.  This helper is test scaffolding, not a harness feature.
    """

    points = np.asarray(keypoints)
    if points.ndim != 3 or points.shape[1] < 133 or points.shape[2] < 2 or len(points) == 0:
        return float("nan")
    if hand == "left":
        selected = points[:, LEFT_HAND_SLICE, :2]
    elif hand == "right":
        selected = points[:, RIGHT_HAND_SLICE, :2]
    else:
        raise ValueError("hand must be 'left' or 'right'")
    finite = np.isfinite(selected).all(axis=(1, 2))
    nonblank = np.any(selected != 0, axis=(1, 2))
    return float((finite & nonblank).mean())

"""Facial expression annotations from Meta's released Imitator features (V00 only).

Meta ships per-frame face features under ``movement:*`` in the NPZ of every V00
recording and of no other vendor (``filelist.has_imitator_movement`` is 1 for
all 42,932 V00 files and 0 for all 86,438 V01-V03 files). This module reads the
four that carry an interpretable, documented-enough quantity -- the validity
flag, the facial action unit intensities and the continuous arousal and valence
-- and summarises them per clip and per recording. Nothing is imputed for
V01-V03: their columns are NA with ``face_features_status = 'not_provided'``,
because a face model has never been run on their video.

What the arrays are (25 V00 NPZs probed read-only: 17 random, 8 end to end)
---------------------------------------------------------------------------
* ``movement:is_valid`` is float32 ``(T, 1)`` with values exactly 0/1 and the same
  ``T`` as the SMPL-H arrays in every file. Invalid frames come in whole
  60-frame (2 s) processing blocks: every invalid run starts at a frame index
  ``= 0 mod 60``. It is a processing-chunk flag, not a face-visibility flag (face
  keypoint confidence is unchanged on those frames). **Whole files can be
  invalid**: 3 of the 17 random V00 files (V00_S0502_I00000130_P0643,
  V00_S1629_I00001084_P0935A, V00_S1916_I00000992_P0946A) have no valid frame
  at all, which :func:`recording_face` reports as ``'all_invalid'``; the other
  22 files are 0.75-1.00 valid (one 0.50).
* ``movement:FAUValue`` is ``(T, 24)`` in ``[0, 2]``. **AUs 16, 17 and 18 are
  dead** (:data:`DEAD_AUS`): over the 14 files with valid frames their maxima
  are 0.0027-0.027, 0.0003-0.005 and < 1e-13 (AU 18 is constant within each
  file), against 2.0 for the live AUs, so they are dropped and every FAU
  statistic is over the 21 live AUs. The AU ordering is not documented.
* ``movement:emotion_arousal`` / ``movement:emotion_valence`` are ``(T, 1)``;
  arousal spans about -0.4 to 1.3, valence -0.6 to 0.6 (orientation, 12 files).
  Class names of ``emotion_scores`` are undocumented, so the logits are not used.
* On invalid frames every one of these arrays is zero-filled. In one of the 14
  random files with valid frames (V00_S0977_I00000225_P1048) the FAU zero-fill
  is shifted two frames earlier than ``is_valid`` at 12 block edges: 24 invalid
  frames carry FAU values and 24 valid frames carry an all-zero FAU row, while
  arousal/valence follow ``is_valid`` exactly. FAU statistics therefore use the frames that are valid
  *and* whose 21 live AUs are not all exactly zero (a zero row is a fill, never
  a measurement: it occurs on no valid frame of the 10 other files checked);
  emotion statistics use the valid frames.

Clip columns (``[start, stop)`` on the pose grid)
------------------------------------------------
``face_valid_frac`` is the share of the clip's frames with ``is_valid``. The
five expression measures need at least 2 s of usable frames
(``frames_for(2.0, fps)``: 60 at 30 fps), since a single 60-frame block is the
unit of invalidity and a statistic over less than one block is noise; below
that, and for every non-V00 clip, they are NaN.

* ``fau_intensity_mean``: mean over usable frames of the mean live-AU intensity.
* ``fau_variability``: mean over the 21 live AUs of each AU's within-clip
  standard deviation (population SD, ``ddof=0``) -- how much the face moves,
  independent of its resting intensity.
* ``emotion_arousal_mean`` / ``emotion_valence_mean`` / ``emotion_arousal_sd``
  (``ddof=0``) over valid frames.

Frames are pooled across invalid gaps inside a clip (the SDs are over the
valid frames, not per contiguous run). Over the 60 clips of the 8 end-to-end
V00 files: face_valid_frac mean 0.91, fau_intensity_mean median 0.17,
fau_variability median 0.161 (p5-p95 0.099-0.208), arousal mean 0.19, valence
mean -0.12, arousal SD 0.17; no clip fell below the 2 s minimum.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np

from .clips import frames_for

#: FAUValue columns that never carry signal (see the module docstring).
DEAD_AUS: tuple[int, ...] = (16, 17, 18)
#: Released FAUValue width.
N_AUS = 24
LIVE_AUS: tuple[int, ...] = tuple(i for i in range(N_AUS) if i not in DEAD_AUS)
#: Expression measures need this much usable face time in the clip.
MIN_VALID_S = 2.0

KEY_VALID = "movement:is_valid"
KEY_FAU = "movement:FAUValue"
KEY_AROUSAL = "movement:emotion_arousal"
KEY_VALENCE = "movement:emotion_valence"
REQUIRED_KEYS: tuple[str, ...] = (KEY_VALID, KEY_FAU, KEY_AROUSAL, KEY_VALENCE)

CLIP_MEASURES: tuple[str, ...] = (
    "fau_intensity_mean", "fau_variability", "emotion_arousal_mean",
    "emotion_valence_mean", "emotion_arousal_sd",
)
FACE_STATUSES: tuple[str, ...] = ("available", "not_provided", "all_invalid", "not_measured")


@dataclass
class FaceTrack:
    """One recording's face features on its pose grid (dead AUs removed)."""

    valid: np.ndarray  # (T,) bool: movement:is_valid (and finite arousal/valence)
    fau: np.ndarray  # (T, 21) float32: live AUs of movement:FAUValue
    arousal: np.ndarray  # (T,) float32
    valence: np.ndarray  # (T,) float32

    @property
    def n_frames(self) -> int:
        return int(self.valid.shape[0])

    @property
    def fau_usable(self) -> np.ndarray:
        """Valid frames whose live AUs are not an all-zero fill row."""

        return _fau_usable(self.valid, self.fau)


def _fau_usable(valid: np.ndarray, fau: np.ndarray) -> np.ndarray:
    return valid & (fau != 0).any(axis=1) & np.isfinite(fau).all(axis=1)


def _rows(array: Any, n_frames: int) -> np.ndarray:
    """First ``n_frames`` rows of a released array as ``(n_frames, width)``; zero rows pad a short one."""

    values = np.asarray(array, dtype=np.float32)
    if values.ndim == 0:
        values = values.reshape(1, 1)
    width = int(np.prod(values.shape[1:], dtype=np.int64)) if values.ndim > 1 else 1
    values = values.reshape(values.shape[0], width)
    out = np.zeros((n_frames, values.shape[1]), dtype=np.float32)
    n = min(n_frames, values.shape[0])
    out[:n] = values[:n]
    return out


def face_track(payload: Mapping[str, Any], n_frames: int) -> FaceTrack | None:
    """The recording's face features, or ``None`` when Meta released none (non-V00).

    ``payload`` is the NPZ (or any mapping of its keys). Arrays are cut or
    padded (as invalid) to ``n_frames``, the pose grid, so clip ranges index
    them directly; in the release they always have exactly ``n_frames`` rows.
    """

    if any(key not in payload for key in REQUIRED_KEYS):
        return None
    n = max(0, int(n_frames))
    flag = _rows(payload[KEY_VALID], n)[:, 0]
    fau_all = _rows(payload[KEY_FAU], n)
    live = [i for i in LIVE_AUS if i < fau_all.shape[1]] + list(range(N_AUS, fau_all.shape[1]))
    fau = np.ascontiguousarray(fau_all[:, live], dtype=np.float32)
    arousal = _rows(payload[KEY_AROUSAL], n)[:, 0]
    valence = _rows(payload[KEY_VALENCE], n)[:, 0]
    valid = (flag > 0.5) & np.isfinite(flag) & np.isfinite(arousal) & np.isfinite(valence)
    return FaceTrack(valid=valid, fau=fau, arousal=arousal, valence=valence)


def _nan_clip() -> dict[str, float]:
    return {name: float("nan") for name in CLIP_MEASURES}


def clip_face(track: FaceTrack | None, start: int, stop: int, fps: float) -> dict[str, Any]:
    """Face columns of the clip ``[start, stop)`` (pose frames)."""

    out: dict[str, Any] = {"face_valid_frac": float("nan"), **_nan_clip()}
    if track is None:
        return out
    first = max(0, int(start))
    last = min(track.n_frames, int(stop))
    if last <= first:
        return out
    valid = track.valid[first:last]
    out["face_valid_frac"] = float(valid.mean())
    need = frames_for(MIN_VALID_S, fps) if fps and math.isfinite(float(fps)) and fps > 0 else math.inf
    if int(valid.sum()) >= need:
        arousal = track.arousal[first:last][valid].astype(np.float64)
        valence = track.valence[first:last][valid].astype(np.float64)
        out["emotion_arousal_mean"] = float(arousal.mean())
        out["emotion_valence_mean"] = float(valence.mean())
        out["emotion_arousal_sd"] = float(arousal.std())
    fau = track.fau[first:last]
    usable = _fau_usable(valid, fau)
    if int(usable.sum()) >= need:
        values = fau[usable].astype(np.float64)
        out["fau_intensity_mean"] = float(values.mean())
        out["fau_variability"] = float(values.std(axis=0).mean())
    return out


def recording_face(track: FaceTrack | None) -> dict[str, Any]:
    """``face_features_status`` and ``recording_face_valid_frac`` of one recording."""

    if track is None:
        return {"face_features_status": "not_provided", "recording_face_valid_frac": float("nan")}
    if track.n_frames == 0:  # a zero-frame NPZ is unmeasured; the annotator says so too
        return {"face_features_status": "not_measured", "recording_face_valid_frac": float("nan")}
    frac = float(track.valid.mean())
    return {"face_features_status": "available" if track.valid.any() else "all_invalid",
            "recording_face_valid_frac": frac}

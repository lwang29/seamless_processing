"""Computable proxies for the reviewer's Pass-1 failure-mode vocabulary.

The five terms come from the user's own review of V00 clips — framing, roll,
sitting, static hands, and desync lag — so unlike Session 2's earlier signals
these are not invented categories. Each function below is a *proxy*: it measures
something correlated with the term, and the docstring says exactly what it
measures so the gap between proxy and concept stays visible.

Two design rules run through all of them:

* **File-level questions get file-level measurements.** "Hands static for the
  entire conversation" cannot be answered from a ten-second window, so the
  static-hand proxy takes a strided sample across the whole recording.
* **Normalize by the person, not the raster, wherever the concept is about the
  person.** Roll and sitting are body geometry; shoulder width and torso length
  are the right denominators. Framing is genuinely about the raster, so that one
  normalizes by the frame.

One of the five proxies was tested and **failed**: the audio/video-lag proxy
cannot work at V00's framing, because the released face block resolves the mouth
across only a handful of pixels. :func:`audiovisual_lag` records the measured
outcome. Desync candidates for review are therefore selected from
container-level mismatch instead, and the gap is reported rather than papered
over with a scattered argmax.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from .m3_features import JOINT_GROUPS, usable_keypoint_mask


# COCO-WholeBody body-17 indices.
NOSE = 0
SHOULDERS = (5, 6)          # left, right
ELBOWS = (7, 8)
WRISTS = (9, 10)
HIPS = (11, 12)
KNEES = (13, 14)
ANKLES = (15, 16)
UPPER_BODY = (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12)

# The released 68-point face block starts at index 23. The standard 0-indexed
# iBUG layout was confirmed empirically from mean landmark geometry on a V00
# file: 0-16 jaw contour, 17-26 brows, 27-35 nose, 36-47 eyes, 48-59 outer mouth,
# 60-67 inner mouth. Inner-lip centres 62 and 66 sit only 0.011 of face height
# apart, so `mouth_open_signal` uses the outer-lip lines instead.
FACE_OFFSET = 23
INNER_LIP_UPPER = FACE_OFFSET + 62
INNER_LIP_LOWER = FACE_OFFSET + 66
OUTER_LIP_UPPER = FACE_OFFSET + 51
OUTER_LIP_LOWER = FACE_OFFSET + 57
LEFT_EYE_OUTER = FACE_OFFSET + 36
RIGHT_EYE_OUTER = FACE_OFFSET + 45


def _midpoint(points: np.ndarray, pair: tuple[int, int]) -> np.ndarray:
    return 0.5 * (points[:, pair[0], :2] + points[:, pair[1], :2])


def _pair_usable(usable: np.ndarray, pair: tuple[int, int]) -> np.ndarray:
    return usable[:, pair[0]] & usable[:, pair[1]]


def _stats(prefix: str, values: np.ndarray, percentiles: tuple[int, ...] = (5, 50, 95)) -> dict[str, float]:
    values = values[np.isfinite(values)]
    if values.size == 0:
        return {f"{prefix}_p{p}": float("nan") for p in percentiles} | {
            f"{prefix}_mean": float("nan"),
            f"{prefix}_n": 0,
        }
    result = {f"{prefix}_p{p}": float(np.percentile(values, p)) for p in percentiles}
    result[f"{prefix}_mean"] = float(values.mean())
    result[f"{prefix}_n"] = int(values.size)
    return result


def framing_signals(
    keypoints: np.ndarray,
    boxes: np.ndarray | None,
    *,
    width: int,
    height: int,
) -> dict[str, Any]:
    """How the person sits inside the raster.

    Measures four separable things the word "framing" bundles together: how much
    of the frame the person fills, how far off-centre they are, how close they
    come to an edge, and whether released keypoints land outside the raster at
    all. Reported separately because a person who is too small and a person who
    is half out of frame are different problems with different fixes.
    """

    points = np.asarray(keypoints, dtype=np.float64)
    if points.ndim != 3 or points.shape[1] < 133 or width <= 0 or height <= 0:
        return {"framing_status": f"invalid_input:{points.shape}"}
    if len(points) == 0:
        return {"framing_status": "empty"}
    usable = usable_keypoint_mask(points)
    result: dict[str, Any] = {"framing_status": "ok"}

    body = points[:, JOINT_GROUPS["body17"], :2]
    body_usable = usable[:, JOINT_GROUPS["body17"]]
    inside = (
        (body[..., 0] >= 0) & (body[..., 0] < width)
        & (body[..., 1] >= 0) & (body[..., 1] < height)
    )
    considered = body_usable
    result["framing_body_out_of_frame_point_frac"] = (
        float((considered & ~inside).sum() / considered.sum()) if considered.any() else float("nan")
    )
    result["framing_frames_with_any_body_out_of_frame"] = float(
        (considered & ~inside).any(axis=1).mean()
    )

    # Person extent from usable keypoints, which is robust when the released box
    # is zero-filled on invalid frames.
    xs = np.where(usable, points[..., 0], np.nan)
    ys = np.where(usable, points[..., 1], np.nan)
    with np.errstate(invalid="ignore"):
        x_min, x_max = np.nanmin(xs, axis=1), np.nanmax(xs, axis=1)
        y_min, y_max = np.nanmin(ys, axis=1), np.nanmax(ys, axis=1)
    person_h = y_max - y_min
    person_w = x_max - x_min
    result.update(_stats("framing_person_height_frac", person_h / height))
    result.update(_stats("framing_person_width_frac", person_w / width))

    centre_x = 0.5 * (x_min + x_max)
    centre_y = 0.5 * (y_min + y_max)
    result.update(_stats("framing_centre_offset_x_frac", (centre_x - width / 2) / width))
    result.update(_stats("framing_centre_offset_y_frac", (centre_y - height / 2) / height))

    # Smallest margin to any raster edge, as a fraction of the shorter side.
    short_side = float(min(width, height))
    margins = np.stack(
        [x_min, width - x_max, y_min, height - y_max], axis=1
    ) / short_side
    # A frame with no usable keypoint is all-NaN; nanmin would warn and return
    # NaN anyway, so skip those rows explicitly.
    measurable = np.isfinite(margins).any(axis=1)
    min_margin = np.full(len(margins), np.nan)
    if measurable.any():
        with np.errstate(invalid="ignore"):
            min_margin[measurable] = np.nanmin(margins[measurable], axis=1)
    result.update(_stats("framing_min_edge_margin_frac", min_margin))
    result["framing_frames_touching_edge_frac"] = float(
        np.nanmean((min_margin <= 0).astype(np.float64))
    )

    if boxes is not None:
        box = np.asarray(boxes, dtype=np.float64)
        if box.ndim == 2 and box.shape[1] >= 4 and len(box) == len(points):
            heights = np.abs(box[:, 3] - box[:, 1])
            valid = np.isfinite(heights) & (heights > 0)
            result.update(_stats("framing_box_height_frac", heights[valid] / height))
    return result


def roll_signals(keypoints: np.ndarray) -> dict[str, Any]:
    """In-plane tilt of the shoulder and hip lines, in degrees.

    Zero is a horizontal shoulder line. This conflates camera roll with a person
    leaning, which is deliberate: the reviewer's term "roll" covers the visible
    result either way, and the two are not separable from 2D keypoints alone.
    The shoulder-versus-hip difference is reported because a genuine camera roll
    tilts both lines equally while a lean does not.
    """

    points = np.asarray(keypoints, dtype=np.float64)
    if points.ndim != 3 or points.shape[1] < 17:
        return {"roll_status": f"invalid_input:{points.shape}"}
    if len(points) == 0:
        return {"roll_status": "empty"}
    usable = usable_keypoint_mask(points)
    result: dict[str, Any] = {"roll_status": "ok"}

    angles: dict[str, np.ndarray] = {}
    for name, pair in (("shoulder", SHOULDERS), ("hip", HIPS)):
        ok = _pair_usable(usable, pair)
        left = points[:, pair[0], :2]
        right = points[:, pair[1], :2]
        delta = right - left
        angle = np.degrees(np.arctan2(delta[:, 1], delta[:, 0]))
        # Fold to (-90, 90] so a left/right label swap cannot read as 180 deg.
        angle = (angle + 90.0) % 180.0 - 90.0
        angle = np.where(ok, angle, np.nan)
        angles[name] = angle
        result.update(_stats(f"roll_{name}_deg", angle))
        result.update(_stats(f"roll_{name}_abs_deg", np.abs(angle)))
    difference = angles["shoulder"] - angles["hip"]
    result.update(_stats("roll_shoulder_minus_hip_deg", difference))
    finite = angles["shoulder"][np.isfinite(angles["shoulder"])]
    result["roll_shoulder_deg_iqr"] = (
        float(np.percentile(finite, 75) - np.percentile(finite, 25)) if finite.size else float("nan")
    )
    return result


def posture_signals(keypoints: np.ndarray) -> dict[str, Any]:
    """Standing-versus-sitting proxies from 2D body geometry only.

    Two independent ratios, both scale-free because they divide by torso length:

    * ``leg_over_torso`` — hip-to-ankle distance over shoulder-to-hip distance.
      Standing is roughly 2; sitting foreshortens the thigh and shortens this.
    * ``knee_drop_over_torso`` — how far below the hips the knees sit. Standing
      puts the knees well below the hips; sitting brings them level or in front.

    A third column reports how often the ankles are usable at all, because a
    seated participant is often cropped above the feet and the absence is itself
    evidence.
    """

    points = np.asarray(keypoints, dtype=np.float64)
    if points.ndim != 3 or points.shape[1] < 17:
        return {"posture_status": f"invalid_input:{points.shape}"}
    if len(points) == 0:
        return {"posture_status": "empty"}
    usable = usable_keypoint_mask(points)
    result: dict[str, Any] = {"posture_status": "ok"}

    shoulder = _midpoint(points, SHOULDERS)
    hip = _midpoint(points, HIPS)
    knee = _midpoint(points, KNEES)
    ankle = _midpoint(points, ANKLES)
    shoulder_ok = _pair_usable(usable, SHOULDERS)
    hip_ok = _pair_usable(usable, HIPS)
    knee_ok = _pair_usable(usable, KNEES)
    ankle_ok = _pair_usable(usable, ANKLES)

    torso = np.linalg.norm(hip - shoulder, axis=1)
    torso_ok = shoulder_ok & hip_ok & (torso > 1e-6)
    result["posture_ankles_usable_frac"] = float(ankle_ok.mean())
    result["posture_knees_usable_frac"] = float(knee_ok.mean())
    result["posture_torso_measurable_frac"] = float(torso_ok.mean())

    leg = np.linalg.norm(ankle - hip, axis=1)
    ratio = np.where(torso_ok & ankle_ok, leg / np.where(torso > 0, torso, np.nan), np.nan)
    result.update(_stats("posture_leg_over_torso", ratio))

    # Positive means knees below hips in image coordinates (y grows downwards).
    drop = np.where(
        torso_ok & knee_ok, (knee[:, 1] - hip[:, 1]) / np.where(torso > 0, torso, np.nan), np.nan
    )
    result.update(_stats("posture_knee_drop_over_torso", drop))
    result.update(_stats("posture_torso_px", np.where(torso_ok, torso, np.nan)))
    return result


def hand_activity_signals(
    keypoints: np.ndarray,
    *,
    width: int,
    height: int,
    fps: float,
    frame_stride: int = 1,
) -> dict[str, Any]:
    """Whole-recording hand motion, normalized by the person's own scale.

    Wrist travel is divided by shoulder width, so a participant standing further
    from the camera is not scored as less animated. ``frame_stride`` records that
    a strided sample was used, and speeds are corrected for it, because the
    reviewer's "static for the entire conversation" is a file-level claim and a
    strided sample is how it stays affordable.

    ``static_frac`` is the fraction of sampled frames in which both wrists stay
    within a tenth of a shoulder width of their own median position. The tenth is
    a stated parameter, and the percentile columns beside it are the
    parameter-free form of the same measurement.
    """

    points = np.asarray(keypoints, dtype=np.float64)
    if points.ndim != 3 or points.shape[1] < 133:
        return {"hand_activity_status": f"invalid_input:{points.shape}"}
    if len(points) < 2 or not np.isfinite(fps) or fps <= 0:
        return {"hand_activity_status": "insufficient_frames"}
    usable = usable_keypoint_mask(points)
    shoulder_ok = _pair_usable(usable, SHOULDERS)
    shoulder_width = np.linalg.norm(
        points[:, SHOULDERS[0], :2] - points[:, SHOULDERS[1], :2], axis=1
    )
    scale = np.median(shoulder_width[shoulder_ok & (shoulder_width > 1e-6)]) if shoulder_ok.any() else float("nan")
    result: dict[str, Any] = {
        "hand_activity_status": "ok",
        "hand_activity_frame_stride": int(frame_stride),
        "hand_activity_frames_sampled": int(len(points)),
        "hand_activity_shoulder_width_px": float(scale) if np.isfinite(scale) else float("nan"),
    }
    if not np.isfinite(scale) or scale <= 0:
        result["hand_activity_status"] = "no_shoulder_scale"
        return result

    effective_fps = fps / max(1, frame_stride)
    static_flags: list[np.ndarray] = []
    for side, index in (("left", WRISTS[0]), ("right", WRISTS[1])):
        wrist = points[:, index, :2]
        ok = usable[:, index]
        step_ok = ok[:-1] & ok[1:]
        speed = np.linalg.norm(wrist[1:] - wrist[:-1], axis=1) / scale * effective_fps
        result.update(_stats(f"hand_activity_{side}_speed_sw_per_s", np.where(step_ok, speed, np.nan)))

        present = wrist[ok]
        if present.size:
            reference = np.median(present, axis=0)
            distance = np.linalg.norm(wrist - reference, axis=1) / scale
            distance = np.where(ok, distance, np.nan)
            result.update(_stats(f"hand_activity_{side}_displacement_sw", distance))
            static_flags.append(np.nan_to_num(distance, nan=0.0) <= 0.10)
        else:
            result.update(_stats(f"hand_activity_{side}_displacement_sw", np.array([])))

        # Wrist height relative to the hips, in torso units: a hand that never
        # rises above the hips is the visible form of "not gesturing".
        hip = _midpoint(points, HIPS)
        shoulder = _midpoint(points, SHOULDERS)
        torso = np.linalg.norm(hip - shoulder, axis=1)
        torso_ok = _pair_usable(usable, HIPS) & _pair_usable(usable, SHOULDERS) & (torso > 1e-6)
        raised = np.where(
            torso_ok & ok, (hip[:, 1] - wrist[:, 1]) / np.where(torso > 0, torso, np.nan), np.nan
        )
        result.update(_stats(f"hand_activity_{side}_above_hip_torso_units", raised))
        result[f"hand_activity_{side}_above_hip_frac"] = float(
            np.nanmean((raised > 0).astype(np.float64))
        )

    if static_flags:
        both_static = np.logical_and.reduce(static_flags)
        result["hand_activity_static_frac"] = float(both_static.mean())
        result["hand_activity_static_threshold_shoulder_widths"] = 0.10
    else:
        result["hand_activity_static_frac"] = float("nan")
    return result


def mouth_open_signal(keypoints: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    """Per-frame normalized mouth opening from the released face block.

    Outer-lip lines (landmarks 50–52 against 56–58 in the 68-point layout) rather
    than the inner pair 62/66: the layout was confirmed empirically from the
    landmark geometry, and the inner-lip centres sit only 0.011 of face height
    apart on average, so their distance is dominated by landmark noise. The
    outer-lip separation includes constant lip thickness, which shifts the signal
    but not its variation.

    Normalized by inter-ocular distance, which is also returned in pixels because
    it is the measurement that decides whether this signal can work at all —
    see :func:`audiovisual_lag` for the measured outcome on V00.
    """

    points = np.asarray(keypoints, dtype=np.float64)
    if points.ndim != 3 or points.shape[1] < 133:
        return np.zeros(0), {"mouth_status": f"invalid_input:{points.shape}"}
    usable = usable_keypoint_mask(points)
    upper_indices = [FACE_OFFSET + index for index in (50, 51, 52)]
    lower_indices = [FACE_OFFSET + index for index in (56, 57, 58)]
    needed = [*upper_indices, *lower_indices, LEFT_EYE_OUTER, RIGHT_EYE_OUTER]
    ok = np.logical_and.reduce([usable[:, index] for index in needed])
    inter_ocular = np.linalg.norm(
        points[:, LEFT_EYE_OUTER, :2] - points[:, RIGHT_EYE_OUTER, :2], axis=1
    )
    upper = points[:, upper_indices, :2].mean(axis=1)
    lower = points[:, lower_indices, :2].mean(axis=1)
    separation = np.linalg.norm(upper - lower, axis=1)
    scale = np.where(ok & (inter_ocular > 1e-6), inter_ocular, np.nan)
    normalized = separation / scale
    finite = np.isfinite(normalized)
    diagnostics = {
        "mouth_status": "ok" if finite.any() else "no_usable_face_frames",
        "mouth_usable_frame_frac": float(finite.mean()),
        "mouth_open_median": float(np.nanmedian(normalized)) if finite.any() else float("nan"),
        # The resolution limit. Below roughly 150 px the mouth spans a handful of
        # pixels and lip motion is not recoverable.
        "mouth_inter_ocular_px_median": (
            float(np.nanmedian(np.where(ok, inter_ocular, np.nan))) if ok.any() else float("nan")
        ),
    }
    return normalized, diagnostics


def _activity_envelope(series: np.ndarray, window: int) -> np.ndarray:
    """Smoothed absolute first difference: "is this thing moving right now"."""

    values = np.asarray(series, dtype=np.float64)
    if values.size < 2:
        return np.zeros(0)
    difference = np.abs(np.diff(values, prepend=values[0]))
    kernel = np.ones(window) / window
    present = np.isfinite(difference).astype(np.float64)
    totals = np.convolve(np.nan_to_num(difference, nan=0.0), kernel, mode="same")
    counts = np.convolve(present, kernel, mode="same") + 1e-12
    return totals / counts


def audiovisual_lag(
    mouth_open: np.ndarray,
    audio_envelope: np.ndarray,
    *,
    max_lag_frames: int = 15,
    activity_frames: int = 9,
) -> dict[str, Any]:
    """Cross-correlate mouth *activity* against speech *activity* over lags.

    Activity envelopes — smoothed absolute first difference — rather than the raw
    signals, because "does the mouth move while there is speech" is far more
    robust than "does mouth height track instantaneous amplitude".

    Sign convention, verified by a round-trip test: ``av_lag_frames`` is positive
    when the mouth moves *after* the matching audio, i.e. **audio leads video**,
    and negative when the audio arrives late.

    **Measured outcome on V00: this proxy does not work.** On eight dev files the
    peak correlation had median 0.151 and range −0.145 to 0.294, and the argmax
    lag scattered across −4, −3, −2, 0, 3, 9, 9, 12 frames with no concentration
    at zero. The cause is resolution: V00's inter-ocular distance is 79–126 px,
    so the mouth spans a handful of pixels and the released whole-body face block
    cannot resolve lip motion. ``av_lag_peak_r`` must therefore be read before
    ``av_lag_frames``, and on V00 it is never high enough to make the lag
    meaningful. Retained so the negative result stays reproducible and so it can
    be re-tested on a vendor framed closer.
    """

    mouth = np.asarray(mouth_open, dtype=np.float64)
    audio = np.asarray(audio_envelope, dtype=np.float64)
    length = min(len(mouth), len(audio))
    if length < 4 * max_lag_frames or max_lag_frames < 1:
        return {"av_lag_status": f"insufficient_frames:{length}"}
    mouth, audio = mouth[:length], audio[:length]
    usable = np.isfinite(mouth) & np.isfinite(audio)
    if usable.sum() < 4 * max_lag_frames:
        return {"av_lag_status": f"insufficient_usable_frames:{int(usable.sum())}"}

    mouth_hp = _activity_envelope(np.where(usable, mouth, np.nan), activity_frames)
    audio_hp = _activity_envelope(np.where(usable, audio, np.nan), activity_frames)

    correlations: list[float] = []
    lags = list(range(-max_lag_frames, max_lag_frames + 1))
    for lag in lags:
        if lag >= 0:
            a, b = mouth_hp[lag:], audio_hp[: length - lag]
            weight = usable[lag:] & usable[: length - lag]
        else:
            a, b = mouth_hp[: length + lag], audio_hp[-lag:]
            weight = usable[: length + lag] & usable[-lag:]
        if weight.sum() < 3:
            correlations.append(float("nan"))
            continue
        first, second = a[weight], b[weight]
        if np.std(first) <= 0 or np.std(second) <= 0:
            correlations.append(float("nan"))
            continue
        correlations.append(float(np.corrcoef(first, second)[0, 1]))
    array = np.asarray(correlations, dtype=np.float64)
    if not np.isfinite(array).any():
        return {"av_lag_status": "no_finite_correlation"}
    best = int(np.nanargmax(array))
    zero = lags.index(0)
    finite = array[np.isfinite(array)]
    return {
        "av_lag_status": "ok",
        "av_lag_frames": int(lags[best]),
        "av_lag_peak_r": float(array[best]),
        "av_lag_zero_r": float(array[zero]),
        "av_lag_peak_minus_zero_r": float(array[best] - array[zero]),
        # Peak prominence: how far the best lag stands above the typical lag. A
        # value near zero means the curve is flat and the argmax is noise.
        "av_lag_peak_prominence_r": float(array[best] - np.nanmedian(finite)),
        "av_lag_max_lag_frames": int(max_lag_frames),
    }


def audio_envelope_at_frame_rate(
    samples: np.ndarray,
    sample_rate: int,
    *,
    fps: float,
    n_frames: int,
) -> np.ndarray:
    """Per-video-frame RMS energy of a mono audio signal.

    Frame boundaries come from the video rate, so the envelope shares the
    annotation grid and no resampling of either stream is implied.
    """

    mono = np.asarray(samples, dtype=np.float64)
    if mono.ndim > 1:
        mono = mono.mean(axis=1)
    if sample_rate <= 0 or not np.isfinite(fps) or fps <= 0 or n_frames <= 0:
        return np.zeros(0)
    edges = np.round(np.arange(n_frames + 1) * sample_rate / fps).astype(np.int64)
    edges = np.clip(edges, 0, len(mono))
    envelope = np.full(n_frames, np.nan)
    for index in range(n_frames):
        chunk = mono[edges[index] : edges[index + 1]]
        if chunk.size:
            envelope[index] = math.sqrt(float(np.mean(chunk * chunk)))
    return envelope

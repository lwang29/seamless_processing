"""M-3 prototype signals: distributions only, no thresholds and no gates.

Three conventions in here are choices, not facts read from the payload, and
each is named in the returned column so a later reader cannot mistake one for
the other:

* **Raster normalization.** Released 2D keypoints are divided by
  ``max(width, height)`` before anything else, so a 3840x2160 file and a
  1080x1920 file produce comparable numbers. A single isotropic divisor is used
  rather than dividing x by width and y by height, because per-axis division
  changes the aspect ratio and would make a diagonal motion's speed depend on
  its direction. Box-height-normalized variants are reported alongside, since
  raster normalization still leaves a person who is twice as far from the
  camera moving half as fast.
* **Jitter.** Two definitions are computed. ``accel`` is the plain
  second-difference magnitude and is expected to rise with genuine motion.
  ``jitter_hp`` is the residual after subtracting a centred 5-frame moving
  average, which suppresses smooth trajectories and keeps frame-to-frame noise.
  Reporting both is what makes the motion-versus-noise question answerable
  instead of assumed.
* **Angular velocity.** Always rotation composition, never axis-angle
  subtraction. Session 1 measured raw axis-angle deltas of 6.3 rad that were
  really 0.05 rad.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np


# Released COCO-WholeBody layout. Face landmarks are recorded for completeness
# but excluded from the body/hand aggregate signals: the model is body + hands.
JOINT_GROUPS: dict[str, slice] = {
    "body17": slice(0, 17),
    "feet6": slice(17, 23),
    "face68": slice(23, 91),
    "left_hand": slice(91, 112),
    "right_hand": slice(112, 133),
}
BODY_WRIST_INDICES = {"left": 9, "right": 10}
BODY_HIP_INDICES = {"left": 11, "right": 12}

# SMPL-H FK output indices (73-joint layout used by scripts.smplh_fk).
FK_PELVIS = 0
FK_WRISTS = {"left": 20, "right": 21}
FK_BODY_JOINTS = tuple(range(22))
FK_LEFT_HAND = tuple(range(22, 37)) + tuple(range(63, 68))
FK_RIGHT_HAND = tuple(range(37, 52)) + tuple(range(68, 73))

MOVING_AVERAGE_FRAMES = 5
PERCENTILES = (5, 25, 50, 75, 90, 95, 99)


@dataclass(frozen=True)
class Signal:
    """One measured number with the basis that makes it interpretable."""

    value: float
    status: str
    basis: str


def _nan_signal(status: str, basis: str) -> Signal:
    return Signal(float("nan"), status, basis)


def _percentile_dict(prefix: str, values: np.ndarray) -> dict[str, float]:
    if values.size == 0:
        return {f"{prefix}_p{p}": float("nan") for p in PERCENTILES} | {
            f"{prefix}_mean": float("nan")
        }
    result = {
        f"{prefix}_p{p}": float(np.percentile(values, p)) for p in PERCENTILES
    }
    result[f"{prefix}_mean"] = float(values.mean())
    return result


def _centred_moving_average(series: np.ndarray, window: int) -> np.ndarray:
    """Centred, NaN-aware moving average with edge frames held, along axis 0.

    Only finite neighbours contribute, and the divisor is the count of finite
    neighbours rather than the window width. Substituting zero for an unusable
    neighbour would drag the local mean towards the image origin and inflate the
    high-pass residual at every usable frame adjacent to a zero-filled one —
    which is exactly the neighbourhood of every box-invalid run, so the
    artificial jitter would land precisely where tracking already failed.
    """

    if window <= 1:
        return series.astype(np.float64, copy=True)
    half = window // 2
    padded = np.concatenate(
        (
            np.repeat(series[:1], half, axis=0),
            series,
            np.repeat(series[-1:], half, axis=0),
        ),
        axis=0,
    ).astype(np.float64, copy=False)
    flat = padded.reshape(len(padded), -1)
    finite = np.isfinite(flat)
    values = np.where(finite, flat, 0.0)
    kernel = np.ones(window)
    totals = np.stack(
        [np.convolve(values[:, column], kernel, mode="valid") for column in range(flat.shape[1])],
        axis=1,
    )
    counts = np.stack(
        [
            np.convolve(finite[:, column].astype(np.float64), kernel, mode="valid")
            for column in range(flat.shape[1])
        ],
        axis=1,
    )
    with np.errstate(invalid="ignore", divide="ignore"):
        smoothed = np.where(counts > 0, totals / counts, np.nan)
    return smoothed.reshape(series.shape)


def usable_keypoint_mask(keypoints: np.ndarray) -> np.ndarray:
    """Per-point usability: finite XY, positive confidence, not exact zero.

    Confidence is not compared against a tuned cutoff. ``> 0`` only removes the
    release's own zero-fill, which Session 1 observed on box-invalid frames.
    """

    points = np.asarray(keypoints, dtype=np.float64)
    xy = points[..., :2]
    confidence = points[..., 2]
    finite = np.isfinite(xy).all(axis=-1) & np.isfinite(confidence)
    nonzero = np.any(xy != 0.0, axis=-1)
    return finite & nonzero & (confidence > 0.0)


def normalized_2d_kinematics(
    keypoints: np.ndarray,
    *,
    width: int,
    height: int,
    fps: float,
    boxes: np.ndarray | None = None,
) -> dict[str, float]:
    """Per-joint-group normalized speed, acceleration, and high-pass jitter.

    Speeds are reported per second using the file's measured ``fps``. Only
    frame pairs whose keypoint is usable in both frames contribute, so zero-fill
    does not register as a large jump into or out of the origin.
    """

    points = np.asarray(keypoints, dtype=np.float64)
    if points.ndim != 3 or points.shape[1] < 133 or points.shape[2] < 3:
        return {"kin2d_status": f"invalid_keypoint_shape:{points.shape}"}
    if width <= 0 or height <= 0:
        return {"kin2d_status": "invalid_raster"}
    if not np.isfinite(fps) or fps <= 0:
        return {"kin2d_status": "invalid_fps"}
    if len(points) < 3:
        return {"kin2d_status": "insufficient_frames"}

    raster_scale = float(max(width, height))
    usable = usable_keypoint_mask(points)
    xy = points[..., :2] / raster_scale
    # Unusable points must not contribute a spurious displacement, and NaN is
    # the only value that cannot be silently averaged in.
    xy = np.where(usable[..., None], xy, np.nan)
    # NaN-aware, so an unusable neighbour is skipped rather than counted as the
    # image origin.
    smoothed = _centred_moving_average(xy, MOVING_AVERAGE_FRAMES)
    residual = xy - smoothed

    result: dict[str, Any] = {"kin2d_status": "ok", "kin2d_raster_scale_px": raster_scale}
    box_height = None
    if boxes is not None:
        box = np.asarray(boxes, dtype=np.float64)
        if box.ndim == 2 and box.shape[1] >= 4 and len(box) == len(points):
            heights = np.abs(box[:, 3] - box[:, 1])
            heights = heights[np.isfinite(heights) & (heights > 0)]
            if heights.size:
                box_height = float(np.median(heights))
    result["kin2d_median_box_height_px"] = box_height if box_height else float("nan")

    for group, index in JOINT_GROUPS.items():
        selected = xy[:, index, :]
        pair_ok = np.isfinite(selected[:-1]).all(axis=-1) & np.isfinite(selected[1:]).all(axis=-1)
        speed = np.linalg.norm(selected[1:] - selected[:-1], axis=-1) * fps
        speed_values = speed[pair_ok]
        triple_ok = pair_ok[:-1] & pair_ok[1:]
        accel = np.linalg.norm(
            selected[2:] - 2.0 * selected[1:-1] + selected[:-2], axis=-1
        ) * fps * fps
        accel_values = accel[triple_ok]
        jitter = np.linalg.norm(residual[:, index, :], axis=-1)
        jitter_values = jitter[np.isfinite(jitter) & usable[:, index]]

        result.update(_percentile_dict(f"speed2d_{group}_rasterfrac_per_s", speed_values))
        result.update(_percentile_dict(f"accel2d_{group}_rasterfrac_per_s2", accel_values))
        result.update(_percentile_dict(f"jitter2d_hp_{group}_rasterfrac", jitter_values))
        result[f"usable_frac_{group}"] = float(usable[:, index].mean())
        if box_height:
            scale = raster_scale / box_height
            result[f"speed2d_{group}_boxheightfrac_per_s_p50"] = (
                float(np.percentile(speed_values, 50)) * scale if speed_values.size else float("nan")
            )
            result[f"jitter2d_hp_{group}_boxheightfrac_p50"] = (
                float(np.percentile(jitter_values, 50)) * scale if jitter_values.size else float("nan")
            )
        else:
            result[f"speed2d_{group}_boxheightfrac_per_s_p50"] = float("nan")
            result[f"jitter2d_hp_{group}_boxheightfrac_p50"] = float("nan")
    return result


def _run_lengths(flags: np.ndarray) -> tuple[int, int, float]:
    """Count, maximum, and median length of ``True`` runs."""

    if flags.size == 0 or not flags.any():
        return 0, 0, float("nan")
    padded = np.concatenate(([False], flags, [False]))
    edges = np.flatnonzero(padded[1:] != padded[:-1])
    lengths = edges[1::2] - edges[0::2]
    return int(len(lengths)), int(lengths.max()), float(np.median(lengths))


def hand_quality(
    keypoints: np.ndarray,
    hand_pose: np.ndarray | None,
    *,
    hand: str,
    width: int,
    height: int,
) -> dict[str, Any]:
    """Availability, framing, run structure, and finger-parameter variation.

    ``hand_pose`` is the released ``(N, 15, 3)`` full axis-angle array. With the
    project's ``flat_hand_mean=True`` convention an all-zero pose is a flat
    hand, so an exactly-zero pose vector is a threshold-free near-default
    detector; its temporal variance separates a genuinely still hand from a
    frozen one.
    """

    if hand not in ("left", "right"):
        raise ValueError("hand must be 'left' or 'right'")
    prefix = f"hand_{hand}"
    points = np.asarray(keypoints, dtype=np.float64)
    if points.ndim != 3 or points.shape[1] < 133 or points.shape[2] < 3 or len(points) == 0:
        return {f"{prefix}_status": f"invalid_keypoint_shape:{points.shape}"}

    index = JOINT_GROUPS["left_hand" if hand == "left" else "right_hand"]
    usable = usable_keypoint_mask(points)[:, index]
    xy = points[:, index, :2]
    result: dict[str, Any] = {f"{prefix}_status": "ok"}
    result[f"{prefix}_usable_keypoint_frac"] = float(usable.mean())
    frame_complete = usable.all(axis=1)
    frame_any = usable.any(axis=1)
    result[f"{prefix}_complete_frame_frac"] = float(frame_complete.mean())
    result[f"{prefix}_any_point_frame_frac"] = float(frame_any.mean())

    inside = (xy[..., 0] >= 0) & (xy[..., 0] < width) & (xy[..., 1] >= 0) & (xy[..., 1] < height)
    # Out-of-frame is only meaningful where the release actually placed a point.
    considered = usable & np.isfinite(xy).all(axis=-1)
    result[f"{prefix}_out_of_frame_point_frac"] = (
        float((considered & ~inside).sum() / considered.sum()) if considered.any() else float("nan")
    )
    result[f"{prefix}_out_of_frame_frame_frac"] = float(
        (considered.any(axis=1) & ~(considered & inside).any(axis=1)).mean()
    )

    runs, longest, median_run = _run_lengths(~frame_complete)
    result[f"{prefix}_unavailable_runs"] = runs
    result[f"{prefix}_unavailable_run_max"] = longest
    result[f"{prefix}_unavailable_run_median"] = median_run

    if hand_pose is None:
        result[f"{prefix}_pose_status"] = "missing_hand_pose"
        return result
    pose = np.asarray(hand_pose, dtype=np.float64)
    if pose.ndim == 3:
        pose = pose.reshape(len(pose), -1)
    if pose.ndim != 2 or pose.shape[1] != 45 or len(pose) != len(points):
        result[f"{prefix}_pose_status"] = f"invalid_hand_pose_shape:{np.shape(hand_pose)}"
        return result
    finite_rows = np.isfinite(pose).all(axis=1)
    if not finite_rows.any():
        result[f"{prefix}_pose_status"] = "no_finite_hand_pose_frames"
        return result
    finite_pose = pose[finite_rows]
    norms = np.linalg.norm(finite_pose, axis=1)
    result[f"{prefix}_pose_status"] = "ok"
    result[f"{prefix}_pose_norm_median_rad"] = float(np.median(norms))
    result[f"{prefix}_pose_norm_p05_rad"] = float(np.percentile(norms, 5))
    result[f"{prefix}_pose_norm_p95_rad"] = float(np.percentile(norms, 95))
    result[f"{prefix}_pose_exact_zero_frac"] = float((norms == 0.0).mean())
    # Mean over the 45 finger parameters of their variance across time: a value
    # near zero means the hand configuration never changes.
    result[f"{prefix}_pose_temporal_var_mean"] = float(finite_pose.var(axis=0).mean())
    if len(finite_pose) >= 2:
        repeated = np.all(finite_pose[1:] == finite_pose[:-1], axis=1)
        _, frozen_max, _ = _run_lengths(repeated)
        result[f"{prefix}_pose_frozen_step_frac"] = float(repeated.mean())
        result[f"{prefix}_pose_frozen_run_max"] = int(frozen_max)
    else:
        result[f"{prefix}_pose_frozen_step_frac"] = float("nan")
        result[f"{prefix}_pose_frozen_run_max"] = 0
    return result


def _axis_angle_to_matrix(vectors: np.ndarray) -> np.ndarray:
    """Rodrigues formula for ``(..., 3)`` axis-angle to ``(..., 3, 3)``."""

    vector = np.asarray(vectors, dtype=np.float64)
    angle = np.linalg.norm(vector, axis=-1, keepdims=True)
    safe = np.where(angle > 0, angle, 1.0)
    axis = vector / safe
    x, y, z = axis[..., 0], axis[..., 1], axis[..., 2]
    zero = np.zeros_like(x)
    skew = np.stack(
        [
            np.stack([zero, -z, y], axis=-1),
            np.stack([z, zero, -x], axis=-1),
            np.stack([-y, x, zero], axis=-1),
        ],
        axis=-2,
    )
    sin = np.sin(angle)[..., None]
    cos = np.cos(angle)[..., None]
    identity = np.broadcast_to(np.eye(3), skew.shape).copy()
    return identity + sin * skew + (1.0 - cos) * (skew @ skew)


def geodesic_angular_velocity(
    axis_angle: np.ndarray,
    *,
    fps: float,
    valid: np.ndarray | None = None,
) -> np.ndarray:
    """Per-frame-pair, per-joint geodesic rotation rate in rad/s.

    The relative rotation ``R_t^T R_{t+1}`` is formed first and its angle read
    from the trace. Subtracting axis-angle vectors instead produces spurious
    near-2-pi spikes because the released vectors are not canonically wrapped.
    """

    vector = np.asarray(axis_angle, dtype=np.float64)
    if vector.ndim == 2:
        vector = vector.reshape(len(vector), -1, 3)
    if vector.ndim != 3 or vector.shape[-1] != 3:
        raise ValueError(f"expected (N, J, 3) axis-angle, got {np.shape(axis_angle)}")
    if len(vector) < 2 or not np.isfinite(fps) or fps <= 0:
        return np.zeros((0, vector.shape[1]))
    finite = np.isfinite(vector).all(axis=-1)
    matrices = _axis_angle_to_matrix(np.nan_to_num(vector, nan=0.0))
    relative = np.einsum("njab,njac->njbc", matrices[:-1], matrices[1:])
    trace = np.einsum("njii->nj", relative)
    angle = np.arccos(np.clip((trace - 1.0) / 2.0, -1.0, 1.0))
    eligible = finite[:-1] & finite[1:]
    if valid is not None:
        mask = np.asarray(valid, dtype=bool).reshape(-1)
        if len(mask) == len(vector):
            eligible &= (mask[:-1] & mask[1:])[:, None]
    return np.where(eligible, angle * fps, np.nan)


def angular_velocity_signals(
    payload: Mapping[str, np.ndarray],
    *,
    fps: float,
    smplh_valid: np.ndarray | None,
) -> dict[str, float]:
    """Geodesic rotation rates for the root, body, and each hand."""

    groups = {
        "root": "smplh:global_orient",
        "body": "smplh:body_pose",
        "left_hand": "smplh:left_hand_pose",
        "right_hand": "smplh:right_hand_pose",
    }
    result: dict[str, float] = {}
    for name, key in groups.items():
        array = payload.get(key)
        if array is None:
            result[f"angvel_{name}_status"] = f"missing:{key}"
            continue
        try:
            rates = geodesic_angular_velocity(array, fps=fps, valid=smplh_valid)
        except ValueError as exc:
            result[f"angvel_{name}_status"] = f"error:{exc}"
            continue
        values = rates[np.isfinite(rates)]
        result[f"angvel_{name}_status"] = "ok" if values.size else "no_eligible_pairs"
        result.update(_percentile_dict(f"angvel_{name}_rad_per_s", values))
        # The naive difference is recorded once so the size of the error the
        # composition avoids stays visible in the output rather than only in
        # Session 1's prose.
        if name == "body":
            raw = np.asarray(array, dtype=np.float64)
            if raw.ndim == 2:
                raw = raw.reshape(len(raw), -1, 3)
            if len(raw) >= 2:
                naive = np.linalg.norm(raw[1:] - raw[:-1], axis=-1) * fps
                naive = naive[np.isfinite(naive)]
                result["angvel_body_naive_axisangle_delta_rad_per_s_p99"] = (
                    float(np.percentile(naive, 99)) if naive.size else float("nan")
                )
                result["angvel_body_naive_axisangle_delta_rad_per_s_max"] = (
                    float(naive.max()) if naive.size else float("nan")
                )
    return result


def metric_3d_signals(
    root_relative_mm: np.ndarray,
    *,
    fps: float,
    frame_valid: np.ndarray | None = None,
) -> dict[str, float]:
    """Root-relative metric acceleration and wrist speed from FK output.

    ``root_relative_mm`` is ``(N, 73, 3)`` in millimetres with the pelvis
    subtracted and the HMR camera translation removed, so these are body-frame
    motion numbers and not tracker motion.
    """

    joints = np.asarray(root_relative_mm, dtype=np.float64)
    if joints.ndim != 3 or joints.shape[1] < 73 or joints.shape[2] != 3:
        return {"metric3d_status": f"invalid_fk_shape:{joints.shape}"}
    if len(joints) < 3 or not np.isfinite(fps) or fps <= 0:
        return {"metric3d_status": "insufficient_frames_or_fps"}
    eligible = np.isfinite(joints).all(axis=(1, 2))
    if frame_valid is not None:
        mask = np.asarray(frame_valid, dtype=bool).reshape(-1)
        if len(mask) == len(joints):
            eligible &= mask
    result: dict[str, Any] = {"metric3d_status": "ok"}

    groups = {
        "body": np.asarray(FK_BODY_JOINTS),
        "left_hand": np.asarray(FK_LEFT_HAND),
        "right_hand": np.asarray(FK_RIGHT_HAND),
    }
    triple = eligible[:-2] & eligible[1:-1] & eligible[2:]
    pair = eligible[:-1] & eligible[1:]
    for name, index in groups.items():
        selected = joints[:, index, :]
        # Millimetres per frame squared, as requested: the frame is the
        # release's own annotation step, so no resampling is implied.
        accel = np.linalg.norm(
            selected[2:] - 2.0 * selected[1:-1] + selected[:-2], axis=-1
        )
        values = accel[triple].reshape(-1)
        values = values[np.isfinite(values)]
        result.update(_percentile_dict(f"accel3d_{name}_mm_per_frame2", values))
    for side, joint in FK_WRISTS.items():
        wrist = joints[:, joint, :]
        speed = np.linalg.norm(wrist[1:] - wrist[:-1], axis=-1) * fps
        values = speed[pair]
        values = values[np.isfinite(values)]
        result.update(_percentile_dict(f"wrist3d_{side}_speed_mm_per_s", values))
        # Parameter-free gesture-space proxy: is the wrist above the hip in the
        # root-relative frame? SMPL-H y grows downwards in the released camera
        # convention, hence the comparison direction.
        hip = joints[:, 1 if side == "left" else 2, :]
        above = (wrist[:, 1] < hip[:, 1]) & eligible
        result[f"wrist3d_{side}_above_hip_frac"] = (
            float(above[eligible].mean()) if eligible.any() else float("nan")
        )
    both = np.concatenate(
        [
            np.linalg.norm(joints[1:, FK_WRISTS["left"], :] - joints[:-1, FK_WRISTS["left"], :], axis=-1),
            np.linalg.norm(joints[1:, FK_WRISTS["right"], :] - joints[:-1, FK_WRISTS["right"], :], axis=-1),
        ]
    )
    tiled = np.concatenate([pair, pair])
    values = both[tiled] * fps
    values = values[np.isfinite(values)]
    result.update(_percentile_dict("wrist3d_either_speed_mm_per_s", values))
    result["metric3d_eligible_frames"] = int(eligible.sum())
    return result


def rest_exit_frac(
    root_relative_mm: np.ndarray,
    *,
    displacement_mm: float,
    frame_valid: np.ndarray | None = None,
) -> dict[str, float]:
    """Fraction of frames whose wrists leave the per-file median wrist position.

    ``displacement_mm`` is a characterization parameter, exposed so the number
    can be re-derived at any other value; the accompanying percentiles are the
    threshold-free version of the same measurement. It is not a filter.
    """

    joints = np.asarray(root_relative_mm, dtype=np.float64)
    if joints.ndim != 3 or joints.shape[1] < 73:
        return {"rest_exit_status": "invalid_fk_shape"}
    eligible = np.isfinite(joints).all(axis=(1, 2))
    if frame_valid is not None:
        mask = np.asarray(frame_valid, dtype=bool).reshape(-1)
        if len(mask) == len(joints):
            eligible &= mask
    if not eligible.any():
        return {"rest_exit_status": "no_eligible_frames"}
    result: dict[str, Any] = {
        "rest_exit_status": "ok",
        "rest_exit_displacement_mm_parameter": displacement_mm,
    }
    for side, joint in FK_WRISTS.items():
        wrist = joints[eligible][:, joint, :]
        reference = np.median(wrist, axis=0)
        distance = np.linalg.norm(wrist - reference, axis=-1)
        result[f"rest_exit_{side}_frac"] = float((distance > displacement_mm).mean())
        result.update(_percentile_dict(f"wrist3d_{side}_displacement_mm", distance))
    return result


def merge_vad_intervals(vad_intervals: list[tuple[float, float]]) -> list[tuple[float, float]]:
    """Merge overlapping VAD intervals so double annotation cannot exceed one."""

    ordered = sorted((max(0.0, start), end) for start, end in vad_intervals if end > start)
    merged: list[list[float]] = []
    for start, end in ordered:
        if merged and start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return [(start, end) for start, end in merged]


def speaking_fraction(
    vad_intervals: list[tuple[float, float]],
    *,
    n_frames: int,
    fps: float,
    start_frame: int = 0,
    media_frames: int | None = None,
) -> dict[str, float]:
    """Speaking fraction and turn structure over one measurement window.

    ``start_frame`` makes the measurement window-local. A file-level speaking
    fraction repeated across every span of that file would be a per-file number
    masquerading as a per-span one: it would make the reported distribution
    file-weighted while claiming a span count, and it would attenuate any
    correlation against genuinely span-local signals.

    ``media_frames`` is the whole file's length and is used only for the
    annotation-overrun check, which is a property of the file rather than of the
    window.
    """

    if n_frames <= 0 or not np.isfinite(fps) or fps <= 0 or start_frame < 0:
        return {"vad_status": "invalid_timing"}
    window_start = start_frame / fps
    window_end = (start_frame + n_frames) / fps
    window_duration = window_end - window_start
    media_duration = (
        (media_frames / fps) if media_frames and media_frames > 0 else window_end
    )
    merged = merge_vad_intervals(vad_intervals)
    if not merged:
        return {
            "vad_status": "no_intervals",
            "speaking_frac": 0.0,
            "vad_interval_count": 0,
            "vad_segments_in_window": 0,
            "vad_total_speech_s": 0.0,
            "vad_median_segment_s": float("nan"),
            "vad_longest_segment_s": float("nan"),
            "vad_annotated_span_s": 0.0,
            "vad_span_overruns_media": False,
        }
    overlaps = np.asarray(
        [
            max(0.0, min(end, window_end) - max(start, window_start))
            for start, end in merged
        ],
        dtype=np.float64,
    )
    in_window = overlaps > 0
    # Segment-length statistics describe the segments that intersect this
    # window, using their full lengths: a turn is a property of the speech, not
    # of where the window happens to cut it.
    lengths = np.asarray([end - start for start, end in merged], dtype=np.float64)
    windowed_lengths = lengths[in_window]
    return {
        "vad_status": "ok",
        "speaking_frac": float(overlaps.sum() / window_duration),
        "vad_interval_count": int(len(merged)),
        "vad_segments_in_window": int(in_window.sum()),
        "vad_total_speech_s": float(overlaps.sum()),
        "vad_median_segment_s": (
            float(np.median(windowed_lengths)) if windowed_lengths.size else float("nan")
        ),
        "vad_longest_segment_s": (
            float(windowed_lengths.max()) if windowed_lengths.size else float("nan")
        ),
        "vad_annotated_span_s": float(merged[-1][1]),
        "vad_span_overruns_media": bool(merged[-1][1] > media_duration + 1.0 / fps),
    }

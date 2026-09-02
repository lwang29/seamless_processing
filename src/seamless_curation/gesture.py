"""Co-speech upper-body gesture measures.

The question this module answers is narrow and was set by the PI: *while the
participant is speaking, do their hands and arms move naturally and visibly,
rather than staying in essentially the same position?* Four traps have to be
avoided, and each one is a named guard below.

**Global body movement is not gesturing.** Everything is measured in the torso
frame (:func:`seamless_curation.smplh_kinematics.to_torso_frame`), whose origin
and axes ride with the shoulders. A participant who sways, turns, steps or
leans moves the frame, not the wrists inside it. ``torso_travel_mm`` records how
much whole-body motion was removed so the removal is auditable rather than
assumed.

**Tracking jitter is not gesturing.** Two independent guards. First, a frame
only counts as active if the wrist has *travelled* — a displacement over a
0.5 s window, not an instantaneous speed — so a wrist that vibrates in place
never accumulates. Second, the released 2D keypoints and the released SMPL-H
fit are separate measurements of the same arm, and
``consistency_r`` correlates their speed envelopes: real motion appears in both,
a fit that is wobbling on its own appears in one.

**One brief adjustment is not gesturing.** Activity is resolved into *episodes*
(runs of at least :attr:`GestureParams.min_episode_s`, joined across gaps of at
most :attr:`GestureParams.merge_gap_s`), and the headline coverage measure is
``speech_segments_covered`` — the share of the window's own-speech segments that
contain at least one episode. A single hand adjustment scores one episode in one
segment however large it is.

**Motion while silent is not co-speech gesture.** Every activity measure is
computed separately over own-speech frames and over the rest, and
``gesture_speech_ratio`` and ``sync_r`` compare them. A participant who fidgets
constantly scores a ratio near 1; a participant who gestures while they talk
scores well above it.

Amplitudes are in millimetres and are directly comparable across participants:
all sixteen SMPL-H betas are zero in this release, so every file has the
identical skeleton.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from .smplh_kinematics import (
    L_ELBOW,
    L_HAND_JOINTS,
    L_SHOULDER,
    L_WRIST,
    R_ELBOW,
    R_HAND_JOINTS,
    R_SHOULDER,
    R_WRIST,
    forward_kinematics,
    geodesic_speed,
    stack_pose,
    to_torso_frame,
)

# Released COCO-WholeBody indices used for the independent 2D channel.
COCO_SHOULDERS = (5, 6)
COCO_ELBOWS = (7, 8)
COCO_WRISTS = (9, 10)
COCO_UPPER_BODY = (0, 5, 6, 7, 8, 9, 10, 11, 12)
COCO_LEFT_HAND = tuple(range(91, 112))
COCO_RIGHT_HAND = tuple(range(112, 133))


@dataclass(frozen=True)
class GestureParams:
    """Every constant the measure depends on, in one auditable place.

    None of these is a *gate*; they parameterise the measurement. Gates live in
    :mod:`seamless_curation.gates` so a threshold can move without rescanning.
    """

    #: Zero-phase Hann smoothing applied before differentiating, in frames.
    #: The released SMPL-H trajectories carry essentially no energy above 3 Hz
    #: (measured: <0.05% of wrist power above 5 Hz), so this is a light touch
    #: that removes single-frame outliers without attenuating any gesture.
    smooth_frames: int = 5
    #: Half-width of the central difference used for speed, in frames.
    speed_halfwidth: int = 2
    #: Window over which "has the wrist actually travelled" is asked, seconds.
    travel_window_s: float = 0.5
    #: A frame is active when the faster wrist exceeds this speed *and* has
    #: travelled at least ``min_travel_mm`` across ``travel_window_s``.
    #: 60 mm/s sits about four times the resting floor measured on the most
    #: static files in the corpus (12-16 mm/s).
    min_speed_mm_s: float = 60.0
    min_travel_mm: float = 35.0
    #: Episode morphology.
    merge_gap_s: float = 0.25
    min_episode_s: float = 0.30
    #: A VAD segment shorter than this is a backchannel, not an utterance, and
    #: is not counted when asking which speech segments carry a gesture.
    min_speech_segment_s: float = 0.80
    #: An episode must overlap a speech segment by this much to cover it.
    min_overlap_s: float = 0.20
    #: Physiologically impossible wrist speed; a real arm does not exceed this
    #: for more than a frame or two, so a sustained excess is a tracking break.
    implausible_speed_mm_s: float = 4000.0
    #: Interval between the postures compared by ``posture_spread_mm``, seconds.
    #: 2.5 s is the spacing of the review card's twelve thumbnails, so the
    #: measure and the reviewer are looking at the same comparison.
    posture_sample_s: float = 2.5
    #: Wrists closer than this are clasped or held together, which is the
    #: characteristic rest posture behind most static-hands rejections.
    hands_together_mm: float = 180.0
    #: Bins used for the speech/gesture cross-correlation, seconds.
    sync_bin_s: float = 0.5
    #: Lags searched by the cross-correlation, seconds either side.
    sync_max_lag_s: float = 1.5


def _hann_smooth(series: np.ndarray, frames: int) -> np.ndarray:
    """Zero-phase Hann smoothing along axis 0, edge-padded."""

    if frames <= 1 or len(series) <= frames:
        return np.asarray(series, dtype=np.float64)
    kernel = np.hanning(frames + 2)[1:-1]
    kernel = kernel / kernel.sum()
    pad = frames // 2
    array = np.asarray(series, dtype=np.float64)
    flat = array.reshape(len(array), -1)
    padded = np.pad(flat, ((pad, pad), (0, 0)), mode="edge")
    out = np.empty_like(flat)
    for column in range(flat.shape[1]):
        out[:, column] = np.convolve(padded[:, column], kernel, mode="valid")[: len(flat)]
    return out.reshape(array.shape)


def _central_speed(track: np.ndarray, fps: float, halfwidth: int) -> np.ndarray:
    """Speed of a ``(frames, 3)`` track by central difference, same length out."""

    n = len(track)
    speed = np.zeros(n, dtype=np.float64)
    if n <= 2 * halfwidth:
        return speed
    delta = np.linalg.norm(track[2 * halfwidth :] - track[: -2 * halfwidth], axis=1)
    speed[halfwidth : n - halfwidth] = delta * fps / (2.0 * halfwidth)
    speed[:halfwidth] = speed[halfwidth]
    speed[n - halfwidth :] = speed[n - halfwidth - 1]
    return speed


def _rolling_travel(track: np.ndarray, frames: int) -> np.ndarray:
    """Diagonal of the axis-aligned bounding box over a centred window, mm.

    This is the jitter guard: a wrist vibrating inside a 10 mm ball has a
    non-trivial instantaneous speed but a near-zero bounding box, so it never
    accumulates travel however fast it shakes.

    The bounding-box diagonal, not the maximum displacement from the window's
    centre — it is within a factor of sqrt(3) of it, is monotone in the same
    thing, and is O(n) with a sliding-window view where the exact quantity is
    not. The threshold is calibrated against this measure, so the two are
    consistent; what matters is that it is zero for motion that stays put.
    """

    n = len(track)
    if frames < 2 or n < 2:
        return np.zeros(n, dtype=np.float64)
    half = max(1, frames // 2)
    padded = np.pad(track, ((half, half), (0, 0)), mode="edge")
    # Cumulative max/min per axis over the sliding window gives the bounding box
    # diagonal, which upper-bounds displacement and is O(n) with stride tricks.
    windows = np.lib.stride_tricks.sliding_window_view(padded, 2 * half + 1, axis=0)
    extent = windows.max(axis=-1) - windows.min(axis=-1)
    return np.linalg.norm(extent, axis=1)[:n]


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """Half-open ``[start, stop)`` intervals of True."""

    if mask.size == 0:
        return []
    padded = np.concatenate(([False], mask.astype(bool), [False]))
    edges = np.flatnonzero(padded[1:] != padded[:-1])
    return list(zip(edges[0::2].tolist(), edges[1::2].tolist()))


def _close_and_open(mask: np.ndarray, close_frames: int, open_frames: int) -> np.ndarray:
    """Binary closing then opening: bridge short gaps, then drop short runs."""

    out = mask.astype(bool).copy()
    if close_frames > 0:
        for start, stop in _runs(~out):
            if stop - start <= close_frames and start > 0 and stop < len(out):
                out[start:stop] = True
    if open_frames > 0:
        for start, stop in _runs(out):
            if stop - start < open_frames:
                out[start:stop] = False
    return out


def speech_mask(vad: Sequence[Mapping[str, float]], frames: int, fps: float) -> np.ndarray:
    """Per-frame own-speech mask from the released VAD intervals."""

    mask = np.zeros(frames, dtype=bool)
    for segment in vad or ():
        start = int(round(float(segment["start"]) * fps))
        stop = int(round(float(segment["end"]) * fps))
        if stop > start:
            mask[max(0, start) : min(frames, stop)] = True
    return mask


def usable_keypoints(keypoints: np.ndarray) -> np.ndarray:
    """``(frames, points)`` mask of released 2D points that carry information.

    Invalid frames are exact zero-fill in this release, and the confidence
    channel is zero there, so both conditions are required.
    """

    points = np.asarray(keypoints, dtype=np.float64)
    finite = np.isfinite(points).all(axis=-1)
    nonzero = (points[..., :2] != 0).any(axis=-1)
    confident = points[..., 2] > 0
    return finite & nonzero & confident


@dataclass
class GestureTracks:
    """Per-frame series the window aggregation consumes.

    Everything here is length ``frames`` and aligned to the released annotation
    grid, so a window is a plain slice.
    """

    fps: float
    frames: int
    params: GestureParams
    #: Faster-wrist speed in the torso frame, mm/s.
    arm_speed: np.ndarray
    #: Faster-wrist travel over ``travel_window_s``, mm.
    arm_travel: np.ndarray
    #: Cleaned activity mask.
    active: np.ndarray
    #: Wrist and elbow positions in the torso frame, mm, shape (frames, 2, 3).
    wrists: np.ndarray
    elbows: np.ndarray
    #: Shoulder positions in the torso frame, mm, shape (frames, 2, 3). The
    #: frame is built on them, so these are exactly ``(+/-d/2, 0, 0)`` — but
    #: ``d`` is the *posed* shoulder separation, which the collar joints move by
    #: several percent, so it is carried rather than assumed constant.
    shoulders: np.ndarray
    #: Finger articulation speed, rad/s, faster hand.
    hand_speed: np.ndarray
    #: Shoulder-midpoint travel in the pelvis frame, mm/s — the motion the torso
    #: frame removed.
    torso_speed: np.ndarray
    #: Own-speech mask from the released VAD.
    speech: np.ndarray
    #: Released per-frame validity flags.
    smplh_valid: np.ndarray
    box_valid: np.ndarray
    #: Hand pose frozen: this frame's 90-vector is bit-identical to the last.
    hand_frozen: np.ndarray
    #: Mean confidence of the upper-body 2D keypoints.
    kp_conf: np.ndarray
    #: Faster-hand speed from the released 2D keypoints, shoulder widths/s.
    arm_speed_2d: np.ndarray
    #: Cosine between successive 2D displacement steps, pooled over both hands
    #: and restricted to steps large enough to have a direction; NaN elsewhere.
    #: Length is ``2 * frames`` — the left hand's series followed by the right
    #: hand's — because it is only ever read as a distribution, and a window
    #: slice of it must therefore take both halves.
    step_cosine: np.ndarray
    #: Pelvis-frame joint positions, metres, ``(frames, 52, 3)``. Only populated
    #: when ``build_tracks(keep_joints=True)``; the scan does not need them and
    #: they are 8 MB for a four-minute file, but the review card draws them.
    joints: np.ndarray | None = None


def build_tracks(
    payload: Mapping[str, np.ndarray],
    vad: Sequence[Mapping[str, float]],
    *,
    fps: float,
    model_root: str,
    params: GestureParams | None = None,
    keep_joints: bool = False,
) -> GestureTracks:
    """Compute every per-frame series for one participant file."""

    params = params or GestureParams()
    body = np.asarray(payload["smplh:body_pose"])
    left = np.asarray(payload["smplh:left_hand_pose"])
    right = np.asarray(payload["smplh:right_hand_pose"])
    frames = len(body)
    # A ragged bundle would silently desynchronise pose from keypoints and every
    # measure downstream would be comparing different instants.
    lengths = {
        key: len(np.asarray(payload[key]))
        for key in (
            "smplh:left_hand_pose", "smplh:right_hand_pose", "smplh:is_valid",
            "boxes_and_keypoints:keypoints", "boxes_and_keypoints:is_valid_box",
        )
        if key in payload
    }
    ragged = {key: value for key, value in lengths.items() if value != frames}
    if ragged:
        raise ValueError(f"released arrays disagree with body_pose ({frames} frames): {ragged}")

    pose = stack_pose(body, left, right, global_orient=None)
    joints, _globals, locals_ = forward_kinematics(pose, model_root)

    torso = to_torso_frame(
        joints, (L_WRIST, R_WRIST, L_ELBOW, R_ELBOW, L_SHOULDER, R_SHOULDER)
    )
    wrists = torso[:, :2]
    elbows = torso[:, 2:4]
    shoulders = torso[:, 4:]
    smooth = _hann_smooth(wrists.reshape(frames, -1), params.smooth_frames).reshape(frames, 2, 3)

    speeds = np.stack(
        [_central_speed(smooth[:, s], fps, params.speed_halfwidth) for s in range(2)], axis=1
    )
    travel_frames = max(2, int(round(params.travel_window_s * fps)))
    travels = np.stack([_rolling_travel(smooth[:, s], travel_frames) for s in range(2)], axis=1)
    arm_speed = speeds.max(axis=1)
    arm_travel = travels.max(axis=1)

    raw_active = (arm_speed > params.min_speed_mm_s) & (arm_travel > params.min_travel_mm)
    active = _close_and_open(
        raw_active,
        close_frames=int(round(params.merge_gap_s * fps)),
        open_frames=int(round(params.min_episode_s * fps)),
    )

    # Local rotations, not global: a global finger rotation carries the whole
    # shoulder-elbow-wrist chain, so a rigid hand on a swinging arm would report
    # large "finger articulation". Measured on a synthetic swing with the hand
    # pose identically zero, the global reading is 2.3 rad/s and the local one
    # is exactly 0.
    finger = geodesic_speed(locals_[:, list(L_HAND_JOINTS + R_HAND_JOINTS)], fps)
    hand_speed = np.maximum(
        finger[:, : len(L_HAND_JOINTS)].mean(axis=1), finger[:, len(L_HAND_JOINTS) :].mean(axis=1)
    )

    shoulder_mid = 0.5 * (joints[:, L_SHOULDER] + joints[:, R_SHOULDER]) * 1000.0
    torso_speed = _central_speed(
        _hann_smooth(shoulder_mid, params.smooth_frames), fps, params.speed_halfwidth
    )

    smplh_valid = np.asarray(payload.get("smplh:is_valid", np.ones(frames, bool))).reshape(-1).astype(bool)
    box_valid = np.asarray(
        payload.get("boxes_and_keypoints:is_valid_box", np.ones(frames, bool))
    ).reshape(-1).astype(bool)

    hands = np.concatenate([left.reshape(frames, -1), right.reshape(frames, -1)], axis=1)
    hand_frozen = np.zeros(frames, dtype=bool)
    if frames > 1:
        hand_frozen[1:] = (hands[1:] == hands[:-1]).all(axis=1)

    keypoints = np.asarray(payload["boxes_and_keypoints:keypoints"], dtype=np.float64)
    kp_conf, arm_speed_2d, step_cosine = _two_dimensional_channel(keypoints, fps, params)

    return GestureTracks(
        fps=float(fps),
        frames=frames,
        params=params,
        arm_speed=arm_speed,
        arm_travel=arm_travel,
        active=active,
        wrists=smooth,
        elbows=elbows,
        shoulders=shoulders,
        hand_speed=hand_speed,
        torso_speed=torso_speed,
        speech=speech_mask(vad, frames, fps),
        smplh_valid=smplh_valid,
        box_valid=box_valid,
        hand_frozen=hand_frozen,
        kp_conf=kp_conf,
        arm_speed_2d=arm_speed_2d,
        step_cosine=step_cosine,
        joints=joints if keep_joints else None,
    )


def _two_dimensional_channel(
    keypoints: np.ndarray, fps: float, params: GestureParams
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Upper-body 2D confidence, shoulder-normalised hand speed, step coherence.

    Deliberately independent of the SMPL-H fit: these are the released
    whole-body 2D detections, so agreement between the two channels is evidence
    that the motion is the participant's and not the fit's.

    Hand *centroid* rather than the single wrist keypoint: the mean of a hand's
    21 released points carries about 40% less near-Nyquist noise than keypoint 9
    or 10 alone (measured on twelve files: 0.605x the noise standard deviation,
    lower in eleven of them).

    ``step_cosine`` is the jitter discriminator. Detector noise in this release
    is dominated by single-frame jump-out-and-back spikes: among consecutive
    large steps, the median cosine between successive displacement vectors is
    -0.91 to -0.999 on still hands and *positive* (+0.32 to +0.86) on moving
    ones. Direction, not magnitude, is what separates the two — which is why it
    escapes the confound that sinks every jitter-magnitude signal.
    """

    frames = len(keypoints)
    nan = np.full(frames, np.nan)
    if keypoints.ndim != 3 or keypoints.shape[1] < 133:
        return np.zeros(frames), np.zeros(frames), nan
    usable = usable_keypoints(keypoints)
    conf = np.where(usable[:, list(COCO_UPPER_BODY)], keypoints[:, list(COCO_UPPER_BODY), 2], 0.0)
    kp_conf = conf.mean(axis=1)

    left_sh = keypoints[:, COCO_SHOULDERS[0], :2]
    right_sh = keypoints[:, COCO_SHOULDERS[1], :2]
    width = np.linalg.norm(left_sh - right_sh, axis=1)
    ok = usable[:, COCO_SHOULDERS[0]] & usable[:, COCO_SHOULDERS[1]] & (width > 1e-6)
    if not ok.any():
        return kp_conf, np.zeros(frames), nan
    scale = float(np.median(width[ok]))
    midpoint = 0.5 * (left_sh + right_sh)

    speeds, raw_tracks = [], []
    for hand, wrist in ((COCO_LEFT_HAND, COCO_WRISTS[0]), (COCO_RIGHT_HAND, COCO_WRISTS[1])):
        points = keypoints[:, list(hand), :2]
        good_points = usable[:, list(hand)]
        count = good_points.sum(axis=1)
        centroid = np.where(
            count[:, None] > 0,
            (points * good_points[..., None]).sum(axis=1) / np.maximum(count, 1)[:, None],
            keypoints[:, wrist, :2],
        )
        good = ((count > 0) | usable[:, wrist]) & ok
        relative = np.where(good[:, None], (centroid - midpoint) / scale, np.nan)
        filled = _forward_fill(relative)
        raw_tracks.append(filled)
        track = np.concatenate([filled, np.zeros((frames, 1))], axis=1)
        speeds.append(
            _central_speed(_hann_smooth(track, params.smooth_frames), fps, params.speed_halfwidth)
        )

    # Step coherence is measured on each hand's *own* track and the two sets of
    # cosines are pooled. Selecting the faster hand per frame and differencing
    # the result would splice the two tracks: the hands sit about a shoulder
    # width apart, so every change of selection injects a displacement ~100x the
    # real per-frame step. On one measured window that turned a step_cosine_p50
    # of -0.48 (detector noise, correctly rejected) into -0.08 (accepted).
    coherence = np.concatenate([_step_coherence(track) for track in raw_tracks])
    return kp_conf, np.maximum(speeds[0], speeds[1]), coherence


def _step_coherence(track: np.ndarray, min_step: float = 0.01) -> np.ndarray:
    """Cosine between successive displacement steps, NaN where both are tiny.

    ``min_step`` is in shoulder widths; 0.01 sits above the measured 2D noise
    floor of 0.011-0.026 SW only marginally, which is deliberate — the point is
    to catch the noise steps and ask what direction they went.
    """

    frames = len(track)
    out = np.full(frames, np.nan)
    if frames < 3:
        return out
    step = np.diff(track, axis=0)
    norm = np.linalg.norm(step, axis=1)
    big = norm[:-1] > min_step
    big &= norm[1:] > min_step
    dot = (step[:-1] * step[1:]).sum(axis=1)
    denominator = norm[:-1] * norm[1:]
    with np.errstate(invalid="ignore", divide="ignore"):
        cosine = np.where(big & (denominator > 1e-12), dot / denominator, np.nan)
    out[1:-1] = cosine
    return out


def _forward_fill(values: np.ndarray) -> np.ndarray:
    """Fill NaN rows with the last finite row, then the first finite row."""

    out = np.array(values, dtype=np.float64, copy=True)
    good = np.isfinite(out).all(axis=1)
    if not good.any():
        return np.zeros_like(out)
    index = np.where(good, np.arange(len(out)), 0)
    np.maximum.accumulate(index, out=index)
    out = out[index]
    first = int(np.argmax(good))
    out[:first] = out[first]
    return out


def _percentile(values: np.ndarray, q: float) -> float:
    finite = values[np.isfinite(values)]
    return float(np.percentile(finite, q)) if finite.size else float("nan")


def _pearson(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 4 or np.std(a) <= 1e-12 or np.std(b) <= 1e-12:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def _episode_intervals(active: np.ndarray) -> list[tuple[int, int]]:
    return _runs(active)


def window_measures(tracks: GestureTracks, start: int, stop: int) -> dict[str, Any]:
    """Aggregate one ``[start, stop)`` slice into the measures the gates read."""

    params = tracks.params
    fps = tracks.fps
    n = stop - start
    if n < int(round(2.0 * fps)):
        return {"gesture_status": f"window_too_short:{n}"}

    speech = tracks.speech[start:stop]
    active = tracks.active[start:stop]
    speed = tracks.arm_speed[start:stop]
    wrists = tracks.wrists[start:stop]
    elbows = tracks.elbows[start:stop]
    shoulders = tracks.shoulders[start:stop]

    out: dict[str, Any] = {
        "gesture_status": "ok",
        "window_frames": int(n),
        "window_seconds": float(n / fps),
        "speech_seconds": float(speech.sum() / fps),
        "speech_frac": float(speech.mean()),
    }

    # --- validity and tracking quality -----------------------------------
    valid = tracks.smplh_valid[start:stop]
    invalid_runs = _runs(~valid)
    out["smplh_valid_frac"] = float(valid.mean())
    out["smplh_longest_invalid_s"] = float(
        max((b - a for a, b in invalid_runs), default=0) / fps
    )
    out["box_valid_frac"] = float(tracks.box_valid[start:stop].mean())
    out["hand_frozen_frac"] = float(tracks.hand_frozen[start:stop].mean())
    out["kp_conf_p10"] = _percentile(tracks.kp_conf[start:stop], 10)
    out["implausible_frac"] = float((speed > params.implausible_speed_mm_s).mean())
    out["consistency_r"] = _pearson(speed, tracks.arm_speed_2d[start:stop])
    # step_cosine holds both hands end to end, so a window is two slices.
    half = len(tracks.step_cosine) // 2
    cosine = np.concatenate(
        [tracks.step_cosine[start:stop], tracks.step_cosine[half + start : half + stop]]
    )
    finite = cosine[np.isfinite(cosine)]
    out["step_cosine_p50"] = float(np.median(finite)) if finite.size >= 8 else float("nan")
    out["step_cosine_n"] = int(finite.size)

    # --- amplitude --------------------------------------------------------
    for name, track in (("wrist", wrists), ("elbow", elbows)):
        centre = np.median(track, axis=0)
        excursion = np.linalg.norm(track - centre, axis=2).max(axis=1)
        out[f"{name}_excursion_p90_mm"] = _percentile(excursion, 90)
        out[f"{name}_range_mm"] = float(
            np.linalg.norm(track.max(axis=0) - track.min(axis=0), axis=1).max()
        )
    out["hand_artic_p75_rad_s"] = _percentile(tracks.hand_speed[start:stop], 75)
    out.update(_posture_measures(wrists, elbows, shoulders, speech, fps, params))
    out["torso_travel_mm_s_p50"] = _percentile(tracks.torso_speed[start:stop], 50)
    out["arm_speed_p50_mm_s"] = _percentile(speed, 50)
    out["arm_speed_speech_p50_mm_s"] = _percentile(speed[speech], 50) if speech.any() else float("nan")

    # --- activity, split by speech ----------------------------------------
    out["gesture_frac"] = float(active.mean())
    out["gesture_frac_speech"] = float(active[speech].mean()) if speech.any() else float("nan")
    out["gesture_frac_silence"] = float(active[~speech].mean()) if (~speech).any() else float("nan")
    speech_rate = out["gesture_frac_speech"]
    silence_rate = out["gesture_frac_silence"]
    if np.isfinite(speech_rate) and np.isfinite(silence_rate):
        out["gesture_speech_ratio"] = float((speech_rate + 0.02) / (silence_rate + 0.02))
    else:
        out["gesture_speech_ratio"] = float("nan")

    # --- episode structure -------------------------------------------------
    episodes = _episode_intervals(active)
    out["episode_count"] = len(episodes)
    out["episode_median_s"] = (
        float(np.median([b - a for a, b in episodes]) / fps) if episodes else 0.0
    )
    overlap_frames = max(1, int(round(params.min_overlap_s * fps)))
    speaking_episodes = [
        (a, b) for a, b in episodes if speech[a:b].sum() >= overlap_frames
    ]
    out["episode_count_speech"] = len(speaking_episodes)
    out["gesture_seconds_speech"] = float((active & speech).sum() / fps)

    segments = [
        (a, b)
        for a, b in _runs(speech)
        if (b - a) >= int(round(params.min_speech_segment_s * fps))
    ]
    out["speech_segment_count"] = len(segments)
    if segments:
        covered = sum(
            1 for a, b in segments if (active[a:b].sum() >= overlap_frames)
        )
        out["speech_segments_covered"] = covered / len(segments)
    else:
        out["speech_segments_covered"] = float("nan")

    out.update(_sync_measures(tracks, start, stop))
    return out


def _posture_measures(
    wrists: np.ndarray,
    elbows: np.ndarray,
    shoulders: np.ndarray,
    speech: np.ndarray,
    fps: float,
    params: GestureParams,
) -> dict[str, float]:
    """Does the participant visit *different* arm postures while speaking?

    The activity measures answer "are the wrists moving"; a participant with
    their hands clasped at the waist, shuffling their fingers, answers yes. What
    the manual review kept rejecting was different: the same posture over and
    over. These four measures name that directly.

    ``posture_spread_mm``
        Mean distance between wrist positions sampled ``posture_sample_s`` apart
        during speech — literally the comparison a reviewer makes across the
        card's twelve thumbnails. Shuffling inside one posture scores near zero
        however fast the shuffling is.
    ``wrist_height_p75_mm``
        Height of the higher wrist above the shoulder midpoint, p75 over speech
        frames. Gesture space is chest height and above; the rest postures are
        hands at the sides (about -550 mm) and clasped at the waist (-350 mm).
    ``hands_together_frac``
        Share of speech frames with the two wrists within
        ``hands_together_mm`` — the clasped-hands rest posture.
    ``arm_abduction_p75_deg``
        Angle of the upper arm away from the torso axis, p75 over speech frames:
        elbows pinned to the ribs cannot make a co-speech gesture, and this says
        so without reference to how fast anything moved.
    """

    if not speech.any():
        speech = np.ones(len(wrists), dtype=bool)
    step = max(1, int(round(params.posture_sample_s * fps)))
    samples = wrists[speech][::step]
    if len(samples) >= 3:
        # Mean pairwise distance per wrist, then the more mobile of the two.
        spreads = []
        for side in range(2):
            points = samples[:, side]
            differences = points[:, None, :] - points[None, :, :]
            distances = np.linalg.norm(differences, axis=-1)
            upper = distances[np.triu_indices(len(points), k=1)]
            spreads.append(float(upper.mean()))
        spread = max(spreads)
    else:
        spread = float("nan")

    higher = np.maximum(wrists[:, 0, 1], wrists[:, 1, 1])
    separation = np.linalg.norm(wrists[:, 0] - wrists[:, 1], axis=1)
    # Upper arm against the torso's vertical axis: 0 deg is straight down.
    upper_arm = elbows - shoulders
    axis = np.array([0.0, -1.0, 0.0])
    with np.errstate(invalid="ignore", divide="ignore"):
        cosine = (upper_arm @ axis) / (np.linalg.norm(upper_arm, axis=-1) + 1e-9)
    abduction = np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))

    return {
        "posture_spread_mm": spread,
        "wrist_height_p75_mm": _percentile(higher[speech], 75),
        "hands_together_frac": float((separation[speech] < params.hands_together_mm).mean()),
        "arm_abduction_p75_deg": _percentile(abduction[speech].max(axis=1), 75),
    }


def _sync_measures(tracks: GestureTracks, start: int, stop: int) -> dict[str, float]:
    """Cross-correlate the gesture-activity envelope with the speech envelope.

    Both envelopes are binned at ``sync_bin_s``. The peak correlation says how
    strongly gesture and speech co-occur; the lag at the peak says whether the
    two streams are aligned, which is the machine-checkable half of the PI's
    "motion looks reasonably synchronised with speech".

    A lag is only meaningful when the peak is: read ``sync_r`` before
    ``sync_lag_s``, exactly as the retired mouth-motion proxy taught.
    """

    params = tracks.params
    fps = tracks.fps
    bin_frames = max(1, int(round(params.sync_bin_s * fps)))
    usable = ((stop - start) // bin_frames) * bin_frames
    if usable < 8 * bin_frames:
        return {"sync_r": float("nan"), "sync_lag_s": float("nan"), "sync_r_zero": float("nan")}
    speech = tracks.speech[start : start + usable].reshape(-1, bin_frames).mean(axis=1)
    speed = tracks.arm_speed[start : start + usable].reshape(-1, bin_frames).mean(axis=1)
    if np.std(speech) <= 1e-9 or np.std(speed) <= 1e-9:
        return {"sync_r": float("nan"), "sync_lag_s": float("nan"), "sync_r_zero": float("nan")}
    max_lag = int(round(params.sync_max_lag_s / params.sync_bin_s))
    best_r, best_lag = -2.0, 0
    zero_r = float("nan")
    for lag in range(-max_lag, max_lag + 1):
        if lag >= 0:
            a, b = speed[lag:], speech[: len(speech) - lag]
        else:
            a, b = speed[: len(speed) + lag], speech[-lag:]
        r = _pearson(a, b)
        if lag == 0:
            zero_r = r
        if np.isfinite(r) and r > best_r:
            best_r, best_lag = r, lag
    return {
        "sync_r": float(best_r) if best_r > -2.0 else float("nan"),
        "sync_lag_s": float(best_lag * params.sync_bin_s),
        "sync_r_zero": float(zero_r),
    }


def sliding_windows(frames: int, window_frames: int, hop_frames: int) -> list[tuple[int, int]]:
    """Half-open window bounds covering ``frames``; the tail is dropped."""

    if frames < window_frames or window_frames <= 0 or hop_frames <= 0:
        return []
    starts = range(0, frames - window_frames + 1, hop_frames)
    return [(s, s + window_frames) for s in starts]

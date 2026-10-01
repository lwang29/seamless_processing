"""How the participant sits in the picture, and whether the tracker kept them.

Nothing here filters. Every measure describes one clip (or one recording) so a
user can select framing, visibility and tracking quality for their own purpose;
the categorical labels built on them (``visible_extent``, ``facing_direction``)
are applied in :mod:`seamless_curation.annotate`.

**Visible means confident *and* inside the raster.** The released 2D keypoints
are extrapolated beyond the picture: up to 6.3% of all keypoint coordinates lie
outside the raster, and in V03 ``P1437`` up to 72% of frames have an ankle at
confidence >= 0.5 that is outside it. So a joint is visible on a frame when
its confidence is >= 0.5, it lies in ``0 <= x < width, 0 <= y < height`` of
the *stored* raster, and the tracking box is valid (invalid-box frames are
exact zero-fill, which would otherwise sit at the in-raster pixel (0, 0)).
Inside-the-raster is invariant to the quarter turn that stands a rotated V03
picture upright, so stored coordinates need no repair for this test. A pair
(shoulders, hips, knees, ankles, wrists) is visible when both of its points are.

**Framing scale.** ``box_height_frac`` is the tracking-box height over the
picture height *in the upright display orientation*: on a quarter-turned file
(3840x2160 stored, person lying sideways) the display height is the stored
width, so the box's x-extent over the stored width is used. The box is first
clipped to the raster. The anamorphic squeeze of V01 acts on x only and does not
change a height, so it is ignored. ``box_edge_contact_frac`` is the share of
valid-box frames whose clipped box comes within 1% of any raster edge (all four
edges, so the turn does not matter).

**Tracker continuity.** ``box_center_jump_max`` is the largest displacement of
the box centre between successive valid-box frames, in box diagonals (mean
diagonal of the two frames): scale-free, rotation-invariant, and large when the
tracker re-acquires or switches subject. A jump is counted in a clip only when
both of its frames are in the clip. On the 52-file orientation sample the
largest jump was <= 0.20 box heights (0 above 0.3), and with this module the
largest per-file jump on 11 files (all vendors and rasters) was 0.030-0.154
diagonals, so the 0.5-diagonal ``flag_tracker_jump`` threshold sits well above
normal tracking. The same run shows why visibility needs the raster test: the
room-camera V03_S0965_I00000162_P3338, whose knees are really out of frame,
reads ``knees_visible_frac`` 0.11 and ``ankles_visible_frac`` 0.00 (clip
medians), while its partner P3339 reads 1.00 for both.

**Facing angle.** The body's forward axis is ``R_root @ e_z`` (SMPL-H rest
frame: +x left, +y up, +z forward), ``R_root`` from ``smplh:global_orient`` in
the HMR camera frame (x right, y *down*, z away from the camera). The
direction toward the camera is ``-z``; the camera's field of view is ~2.9 deg
(f = 5000/256 * max(W, H) px), so the true line of sight differs from ``-z`` by
under 1.5 deg and the translation (a sentinel on invalid frames) is not needed.
The angle is unsigned, so it is invariant to the camera roll of the rotated
files. It is the *pelvis* orientation, so spine flexion does not enter it.
Severe-anamorphic files (2160x2160, 1920x1080) are NaN: the fit had to distort
the 3D pose to match a stretched person, so its orientation is not a
measurement of the body.

Sign check (2026-09-24, 42 random files: 6 per vendor x raster).
``global_orient`` is a ~180 deg rotation about x (``[3.06, -0.003, -0.045]`` on
V00_S1245_I00001163_P0955A), which maps +z to ~-z. Independently, on every
upright square-pixel file the subject's left shoulder (COCO 5) is to the image
right of the right shoulder (COCO 6) on 100% of valid frames -- they face the
camera -- and the angle reads small: group medians 4.1 deg (V00), 3.5 (V01
1080x1920), 4.4 (V02), 6.2 (V03 2160x3840), 5.2 (V03 3840x2160, quarter-turned);
file medians 2.4-13.8 deg. A flipped sign would read ~175. The contract
expected ~10-30 deg "including the ~13 deg camera tilt"; the pelvis forward
axis is pitched only ~4-7 deg from the optical axis (V00_S1230 P0407A: +7.0,
V03_S1136 P3527: -3.8), so the typical value is lower than that guess. The
1920x1080 anamorphic files read 10.7-29.1 deg (NaN here, as severe).

**Reprojection error.** The SMPL-H fit is projected back into the picture and
compared with the independent 2D keypoints -- a tracking-quality signal that
does not depend on how much the person moves. Camera-frame joints are
``R_root @ joints_pelvis + J0_rest + translation`` (``joints_pelvis`` from the
numpy FK with the root rotation dropped; ``J0_rest`` the neutral model's rest
pelvis, a constant because all betas are zero -- without it the projection is
off by ~0.5 torso lengths), projected with
:func:`seamless_curation.smplh_fk.project_hmr2_full_frame` on the stored
raster. Joints: shoulders, elbows, wrists and hips (COCO 5..12 <->
``COCO_BODY23_TO_SMPLH[5:13]``); the feet are excluded because their landmarks
produce structured outliers of thousands of pixels. Per frame the error is the
mean, over those joints that are visible, of the pixel distance divided by the
2D shoulder width in pixels (so the shoulders must both be visible); every 3rd
frame (on the recording's frame grid) is measured, skipping invalid-box frames
and frames with ``|translation| > 1e3`` (the ~1e12 sentinel).

Measured (same 42 files, recording medians; group median, file range):
V00 0.079 (0.063-0.153), V01 1080x1920 0.076 (0.061-0.149), V02 0.097
(0.066-0.119), V03 3840x2160 0.112 (0.103-0.136), V01 2160x2160 0.119
(0.112-0.152), V01 1920x1080 0.133 (0.125-0.143), V03 2160x3840 0.190
(0.092-0.214). The v0 per-vendor medians were 0.079 / 0.071 / 0.111 / 0.127.
So the anamorphic rasters sit above the square-pixel portrait rasters of the
same vendor, as the media_repair residuals predict -- but V03 2160x3840 is
higher still, and that is not a joint-level fit error: in its high files every
joint, shoulders included, is off by ~0.2 shoulder widths, because the whole
projected body sits ~0.19 SW lower and 8-12% larger than the keypoints
(V03_S1136_I00000206_P3527: offset +94 px, scale 1.118; V03_S0948 P1574, a
normal file of the same raster: offset -1.5 px, scale 0.964). That is a camera
mismatch for part of V03's portrait recordings; read the error as "fit and
keypoints disagree", not as "the pose is wrong".
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np

from .gesture import short_clip_frames, usable_keypoints
from .media_repair import body_roll_deg, quarter_turns_from_roll
from .smplh_fk import COCO_BODY23_TO_SMPLH, project_hmr2_full_frame
from .smplh_kinematics import axis_angle_to_matrix

#: Confidence at which a released keypoint counts as detected. Confidence is not
#: a probability (it exceeds 1); 0.5 is the threshold every earlier visibility
#: measure in this repository used.
VISIBLE_CONF = 0.5
#: A box edge within this fraction of the raster dimension touches the edge.
EDGE_MARGIN_FRAC = 0.01
#: Box-centre jump, in box diagonals, above which the tracker is flagged.
TRACKER_JUMP_DIAGONALS = 0.5
#: Frame stride of the reprojection measurement.
REPROJECTION_STRIDE = 3
#: ``|translation|`` above this is the invalid-frame sentinel (~1e12), not a fit.
TRANSLATION_SENTINEL_M = 1e3
#: The neutral SMPL-H rest pelvis (``smplh_kinematics.rest_joints(...)[0][0]``),
#: metres. A constant of the model because every beta is zero in the release;
#: ``tests/test_framing.py`` checks it against the model file when available.
REST_PELVIS_M = np.array([-0.0017950595290383117, -0.2233334459925592, 0.02821912553071819])

COCO_PAIRS: dict[str, tuple[int, int]] = {
    "shoulders": (5, 6),
    "hips": (11, 12),
    "knees": (13, 14),
    "ankles": (15, 16),
    "wrists": (9, 10),
}
COCO_HANDS = tuple(range(91, 133))
#: COCO body keypoints compared by the reprojection error, and their SMPL-H joints.
REPROJECTION_COCO = tuple(range(5, 13))
REPROJECTION_SMPLH = tuple(int(j) for j in COCO_BODY23_TO_SMPLH[5:13])

CLIP_COLUMNS: tuple[str, ...] = (
    "shoulders_visible_frac", "hips_visible_frac", "knees_visible_frac", "ankles_visible_frac",
    "wrists_in_frame_frac", "hands_kp_conf_p50", "box_height_frac", "box_edge_contact_frac",
    "facing_angle_deg_p50", "facing_angle_range_deg", "reprojection_error_p50_sw",
    "box_center_jump_max", "flag_tracker_jump",
)


def detect_quarter_turns(keypoints: np.ndarray) -> int:
    """Counter-clockwise quarter turns that stand the stored picture upright.

    :func:`seamless_curation.media_repair.body_roll_deg` (median in-image angle
    of the shoulder-to-hip axis) over frames where all four points are usable
    (:func:`seamless_curation.gesture.usable_keypoints`: finite, not the
    zero-fill, confidence > 0), then
    :func:`~seamless_curation.media_repair.quarter_turns_from_roll`. Detected
    per file rather than per raster: 9 of the 1,336 3840x2160 files are already
    upright.
    """

    points = np.asarray(keypoints)
    if points.ndim != 3 or len(points) == 0:
        return 0
    return int(quarter_turns_from_roll(body_roll_deg(points, usable_keypoints(points))))


def visible_mask(
    keypoints: np.ndarray, width: int, height: int, box_valid: np.ndarray
) -> np.ndarray:
    """``(frames, 133)``: confidence >= 0.5, inside the stored raster, valid box.

    With an unknown raster (``width`` or ``height`` <= 0) nothing can be shown
    to be inside it, and every point is reported not visible.
    """

    points = np.asarray(keypoints)
    frames = len(points)
    if points.ndim != 3 or frames == 0 or width <= 0 or height <= 0:
        return np.zeros(points.shape[:2] if points.ndim == 3 else (frames, 0), dtype=bool)
    x, y, conf = points[..., 0], points[..., 1], points[..., 2]
    with np.errstate(invalid="ignore"):
        inside = (x >= 0) & (x < width) & (y >= 0) & (y < height)
        confident = conf >= VISIBLE_CONF
    valid = np.asarray(box_valid).reshape(-1).astype(bool)
    return confident & inside & np.isfinite(points).all(axis=-1) & valid[:, None]


@dataclass
class FramingTrack:
    """Per-frame framing series; a clip is a plain slice of each.

    Every array is length ``frames`` on the released pose grid. Float series are
    NaN on frames where the quantity is not measured (invalid box, unknown
    raster, not a reprojection sample, severe anamorphic for facing).
    """

    fps: float
    frames: int
    width: int
    height: int
    quarter_turns: int
    smplh_anamorphic: str
    #: Both a width and a height were supplied (the raster is known).
    raster_known: bool
    box_valid: np.ndarray
    #: Both points of the pair visible (:func:`visible_mask`).
    shoulders_visible: np.ndarray
    hips_visible: np.ndarray
    knees_visible: np.ndarray
    ankles_visible: np.ndarray
    wrists_in_frame: np.ndarray
    #: Mean confidence of the 42 hand keypoints (COCO 91..132); 0 on zero-fill.
    hands_kp_conf: np.ndarray
    #: Clipped box height / upright display height; NaN on invalid-box frames.
    box_height_frac: np.ndarray
    #: Clipped box within 1% of a raster edge (read on valid-box frames only).
    box_edge_contact: np.ndarray
    #: Box-centre jump from the previous valid-box frame, box diagonals; NaN
    #: where there is no previous valid-box frame.
    box_jump: np.ndarray
    #: Index of that previous valid-box frame, -1 where none.
    box_jump_from: np.ndarray
    #: Angle between the body's forward axis and the direction to the camera.
    facing_angle_deg: np.ndarray
    #: Reprojection error in 2D shoulder widths on measured sample frames.
    reprojection_error_sw: np.ndarray


def _pair_visible(visible: np.ndarray, pair: tuple[int, int]) -> np.ndarray:
    if visible.shape[1] <= max(pair):
        return np.zeros(len(visible), dtype=bool)
    return visible[:, pair[0]] & visible[:, pair[1]]


def framing_track(
    payload: Mapping[str, np.ndarray],
    joints_pelvis: np.ndarray | None,
    *,
    fps: float,
    width: int,
    height: int,
    quarter_turns: int,
    smplh_anamorphic: str,
    rest_pelvis: np.ndarray | None = None,
) -> FramingTrack:
    """Compute every per-frame framing series for one recording.

    ``joints_pelvis`` is ``gesture.GestureTracks.joints`` (pelvis frame, metres,
    root rotation dropped); ``None`` makes the reprojection error NaN.
    ``width``/``height`` are the stored raster (0 when unknown). ``rest_pelvis``
    overrides :data:`REST_PELVIS_M` (e.g. from
    ``smplh_kinematics.rest_joints(model_root)[0][0]``).
    """

    keypoints = np.asarray(payload["boxes_and_keypoints:keypoints"])
    frames = int(len(keypoints))
    width = int(width) if width and width > 0 else 0
    height = int(height) if height and height > 0 else 0
    raster_known = width > 0 and height > 0
    box_valid = np.asarray(
        payload.get("boxes_and_keypoints:is_valid_box", np.ones(frames, bool))
    ).reshape(-1).astype(bool)
    if len(box_valid) != frames:
        raise ValueError(f"is_valid_box has {len(box_valid)} frames, keypoints have {frames}")

    visible = visible_mask(keypoints, width, height, box_valid)
    pairs = {name: _pair_visible(visible, pair) for name, pair in COCO_PAIRS.items()}

    if keypoints.ndim == 3 and keypoints.shape[1] >= max(COCO_HANDS) + 1:
        hand_conf = np.asarray(keypoints[:, list(COCO_HANDS), 2], dtype=np.float64)
        hand_conf = np.where(np.isfinite(hand_conf), hand_conf, 0.0)
        hands_kp_conf = np.where(box_valid, hand_conf.mean(axis=1), 0.0)
    else:
        hands_kp_conf = np.zeros(frames)

    box_height, edge, jump, jump_from = _box_series(payload, box_valid, frames, width, height, quarter_turns)
    anamorphic = str(smplh_anamorphic or "none")
    orient = _root_rotations(payload, frames)
    facing = _facing_angle(orient, box_valid, severe=(anamorphic == "severe"))
    reprojection = _reprojection_error(
        payload, joints_pelvis, orient, keypoints, visible, box_valid, width, height,
        REST_PELVIS_M if rest_pelvis is None else np.asarray(rest_pelvis, dtype=np.float64),
    )
    return FramingTrack(
        fps=float(fps), frames=frames, width=width, height=height,
        quarter_turns=int(quarter_turns), smplh_anamorphic=anamorphic, raster_known=raster_known,
        box_valid=box_valid,
        shoulders_visible=pairs["shoulders"], hips_visible=pairs["hips"],
        knees_visible=pairs["knees"], ankles_visible=pairs["ankles"],
        wrists_in_frame=pairs["wrists"],
        hands_kp_conf=hands_kp_conf,
        box_height_frac=box_height, box_edge_contact=edge, box_jump=jump, box_jump_from=jump_from,
        facing_angle_deg=facing, reprojection_error_sw=reprojection,
    )


def _box_series(
    payload: Mapping[str, np.ndarray], box_valid: np.ndarray, frames: int,
    width: int, height: int, quarter_turns: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    nan = np.full(frames, np.nan)
    jump_from = np.full(frames, -1, dtype=np.int64)
    if "boxes_and_keypoints:box" not in payload:
        return nan, np.zeros(frames, bool), nan.copy(), jump_from
    box = np.asarray(payload["boxes_and_keypoints:box"], dtype=np.float64).reshape(-1, 4)
    if len(box) != frames:
        raise ValueError(f"box has {len(box)} frames, keypoints have {frames}")
    x1, y1, x2, y2 = box.T
    usable = box_valid & np.isfinite(box).all(axis=1) & (x2 > x1) & (y2 > y1)

    height_frac = nan.copy()
    edge = np.zeros(frames, dtype=bool)
    if width > 0 and height > 0:
        cx1, cx2 = np.clip(x1, 0, width), np.clip(x2, 0, width)
        cy1, cy2 = np.clip(y1, 0, height), np.clip(y2, 0, height)
        if int(quarter_turns) % 2:
            extent = (cx2 - cx1) / width      # display height is the stored width
        else:
            extent = (cy2 - cy1) / height
        height_frac = np.where(usable, extent, np.nan)
        with np.errstate(invalid="ignore"):
            touches = (
                (cx1 <= EDGE_MARGIN_FRAC * width) | (cy1 <= EDGE_MARGIN_FRAC * height)
                | (cx2 >= (1 - EDGE_MARGIN_FRAC) * width) | (cy2 >= (1 - EDGE_MARGIN_FRAC) * height)
            )
        edge = usable & touches

    jump = nan.copy()
    index = np.flatnonzero(usable)
    if len(index) >= 2:
        centre = np.stack([0.5 * (x1 + x2), 0.5 * (y1 + y2)], axis=1)
        diagonal = np.hypot(x2 - x1, y2 - y1)
        now, before = index[1:], index[:-1]
        scale = 0.5 * (diagonal[now] + diagonal[before])
        jump[now] = np.linalg.norm(centre[now] - centre[before], axis=1) / scale
        jump_from[now] = before
    return height_frac, edge, jump, jump_from


def _root_rotations(payload: Mapping[str, np.ndarray], frames: int) -> np.ndarray | None:
    if "smplh:global_orient" not in payload:
        return None
    orient = np.asarray(payload["smplh:global_orient"], dtype=np.float64).reshape(-1, 3)
    if len(orient) != frames:
        raise ValueError(f"global_orient has {len(orient)} frames, keypoints have {frames}")
    finite = np.isfinite(orient).all(axis=1)
    rotations = axis_angle_to_matrix(np.where(finite[:, None], orient, 0.0))
    rotations[~finite] = np.nan
    return rotations


def _facing_angle(rotations: np.ndarray | None, box_valid: np.ndarray, *, severe: bool) -> np.ndarray:
    frames = len(box_valid)
    if rotations is None or severe:
        return np.full(frames, np.nan)
    forward_z = rotations[:, 2, 2]  # z component of R_root @ e_z
    # Toward the camera is -z, so cos(angle) = forward . (0, 0, -1) = -forward_z.
    angle = np.degrees(np.arccos(np.clip(-forward_z, -1.0, 1.0)))
    return np.where(box_valid & np.isfinite(forward_z), angle, np.nan)


def _reprojection_error(
    payload: Mapping[str, np.ndarray],
    joints_pelvis: np.ndarray | None,
    rotations: np.ndarray | None,
    keypoints: np.ndarray,
    visible: np.ndarray,
    box_valid: np.ndarray,
    width: int,
    height: int,
    rest_pelvis: np.ndarray,
) -> np.ndarray:
    frames = len(box_valid)
    out = np.full(frames, np.nan)
    if (
        joints_pelvis is None or rotations is None or width <= 0 or height <= 0
        or "smplh:translation" not in payload or visible.shape[1] <= max(REPROJECTION_COCO)
    ):
        return out
    joints = np.asarray(joints_pelvis, dtype=np.float64)
    translation = np.asarray(payload["smplh:translation"], dtype=np.float64).reshape(-1, 3)
    if len(joints) != frames or len(translation) != frames:
        raise ValueError("joints / translation length disagrees with the keypoints")
    sample = np.zeros(frames, dtype=bool)
    sample[::REPROJECTION_STRIDE] = True
    with np.errstate(invalid="ignore"):
        plausible = np.isfinite(translation).all(axis=1) & (
            np.linalg.norm(translation, axis=1) <= TRANSLATION_SENTINEL_M
        )
    coco = list(REPROJECTION_COCO)
    seen = visible[:, coco]
    shoulders_seen = seen[:, 0] & seen[:, 1]  # COCO 5 and 6 lead REPROJECTION_COCO
    take = np.flatnonzero(
        sample & box_valid & plausible & shoulders_seen & np.isfinite(rotations).all(axis=(1, 2))
    )
    if len(take) == 0:
        return out
    body = joints[take][:, list(REPROJECTION_SMPLH)]
    camera = np.einsum("nij,nkj->nki", rotations[take], body) + rest_pelvis + translation[take][:, None]
    projected = project_hmr2_full_frame(camera, width, height)
    observed = np.asarray(keypoints[take][:, coco, :2], dtype=np.float64)
    shoulder_width = np.linalg.norm(observed[:, 0] - observed[:, 1], axis=1)
    distance = np.linalg.norm(projected - observed, axis=-1)
    mask = seen[take] & (camera[..., 2] > 0) & np.isfinite(distance)
    with np.errstate(invalid="ignore", divide="ignore"):
        per_frame = (np.where(mask, distance, 0.0).sum(axis=1) / mask.sum(axis=1)) / shoulder_width
    good = (mask.sum(axis=1) > 0) & (shoulder_width > 1e-6) & np.isfinite(per_frame)
    out[take[good]] = per_frame[good]
    return out


def _frac(mask: np.ndarray) -> float:
    return float(mask.mean()) if mask.size else float("nan")


def _nan_percentile(values: np.ndarray, q: float) -> float:
    finite = values[np.isfinite(values)]
    return float(np.percentile(finite, q)) if finite.size else float("nan")


def clip_framing(track: FramingTrack, start: int, stop: int) -> dict[str, Any]:
    """The registry ``clips`` framing columns for ``[start, stop)``.

    NA follows the registry: visibility, hand confidence and facing need a clip
    of at least ``round(2 * fps)`` frames; box measures need a valid-box frame
    (two for the jump); reprojection needs one measured frame. Visibility with an
    unknown raster is NaN. ``flag_tracker_jump`` is never NA.
    """

    start = max(0, int(start))
    stop = min(track.frames, int(stop))
    n = max(0, stop - start)
    nan = float("nan")
    s = slice(start, stop)
    short = n < short_clip_frames(track.fps)
    out: dict[str, Any] = {}

    for name, series in (
        ("shoulders_visible_frac", track.shoulders_visible), ("hips_visible_frac", track.hips_visible),
        ("knees_visible_frac", track.knees_visible), ("ankles_visible_frac", track.ankles_visible),
        ("wrists_in_frame_frac", track.wrists_in_frame),
    ):
        out[name] = nan if (short or not track.raster_known) else _frac(series[s])
    out["hands_kp_conf_p50"] = nan if short else _nan_percentile(track.hands_kp_conf[s], 50)

    valid = track.box_valid[s]
    out["box_height_frac"] = _nan_percentile(track.box_height_frac[s], 50)
    out["box_edge_contact_frac"] = (
        float(track.box_edge_contact[s][valid].mean())
        if (valid.any() and track.raster_known) else nan
    )
    facing = track.facing_angle_deg[s]
    if short or not np.isfinite(facing).any():
        out["facing_angle_deg_p50"] = nan
        out["facing_angle_range_deg"] = nan
    else:
        out["facing_angle_deg_p50"] = _nan_percentile(facing, 50)
        out["facing_angle_range_deg"] = _nan_percentile(facing, 90) - _nan_percentile(facing, 10)
    out["reprojection_error_p50_sw"] = _nan_percentile(track.reprojection_error_sw[s], 50)

    inside = (track.box_jump_from[s] >= start) & np.isfinite(track.box_jump[s])
    jump = float(track.box_jump[s][inside].max()) if inside.any() else nan
    out["box_center_jump_max"] = jump
    out["flag_tracker_jump"] = bool(np.isfinite(jump) and jump > TRACKER_JUMP_DIAGONALS)
    return out


def recording_framing(track: FramingTrack) -> dict[str, float]:
    """Whole-recording framing: ``recording_wrists_in_frame_frac`` and
    ``recording_reprojection_error_p50_sw``, over all of the recording's frames."""

    return {
        "recording_wrists_in_frame_frac": (
            _frac(track.wrists_in_frame) if (track.raster_known and track.frames > 0) else float("nan")
        ),
        "recording_reprojection_error_p50_sw": _nan_percentile(track.reprojection_error_sw, 50),
    }

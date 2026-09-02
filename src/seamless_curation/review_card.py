"""The review card: one PNG that answers all four review questions at a glance.

The v0 review artefact was a 30-second video per clip, and it made the manual
pass impossible to scale for three measured reasons. Its window was placed by a
seeded uniform hash, so 17.9% of the clips a reviewer was asked to judge for
co-speech gesture contained *no speech at all*. Its hands were 19 px wide,
because a 1080x1920 portrait frame was squeezed into a 216x384 panel and the
upper-body crop that existed to fix that was retired. And it cost 39 s and 2 MB
to produce and 30 s of wall-clock to watch, so triaging ten thousand clips meant
83 hours of playback.

A card is a different trade. It is ~7 s and ~0.5 MB to produce, it is read in a
few seconds, and it puts the four questions side by side:

1. **Is the SMPL-H tracking valid?** Band rows 1 and 3 draw the reprojected
   SMPL-H skeleton on the actual video frame, cropped to the upper body, so a
   skeleton that has come off the person is obvious.
2. **Do the arms and hands really move?** Twelve moments spread across the spans
   that would be accepted, at ~250 px per crop, so a hand is ~90 px and fingers
   are readable.
3. **Is the motion natural rather than noise?** Rows 2 and 4 show the same
   moments as the pelvis-frame SMPL-H skeleton from a three-quarter view — the
   pose ViBES would train on, with no video to flatter it.
4. **Is it synchronised with speech?** The timeline plots own speech, partner
   speech, the arm-speed trace and the detected gesture episodes on one axis,
   over the whole recording, with the accepted spans marked.

The card is a *triage and verification* artefact, not a replacement for
listening: the review app plays the rendered clip with audio beside it, and the
reviewer is expected to use it whenever the card leaves any doubt.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import cv2
import numpy as np

from .gesture import COCO_LEFT_HAND, COCO_RIGHT_HAND, COCO_UPPER_BODY, GestureTracks
from .smplh_kinematics import (
    HEAD,
    L_ELBOW,
    L_HAND_JOINTS,
    L_SHOULDER,
    L_WRIST,
    NECK,
    PELVIS,
    R_ELBOW,
    R_HAND_JOINTS,
    R_SHOULDER,
    R_WRIST,
    SPINE3,
)

# ---------------------------------------------------------------- appearance
BACKGROUND = (18, 18, 18)
PANEL = (30, 30, 30)
TEXT = (235, 235, 235)
MUTED = (150, 150, 150)
SPEECH = (150, 90, 40)       # BGR: own speech, blue
PARTNER = (60, 90, 130)      # BGR: partner speech, brown
GESTURE = (90, 190, 90)      # BGR: gesture episode, green
ACCEPTED = (110, 220, 110)
TRACE = (235, 235, 235)
SKELETON = (0, 225, 255)     # BGR yellow, matches the v0 galleries
LEFT_HAND_COLOR = (255, 140, 40)
RIGHT_HAND_COLOR = (40, 120, 255)

#: Upper-body edges over SMPL-H joint indices.
BODY_EDGES: tuple[tuple[int, int], ...] = (
    (PELVIS, SPINE3), (SPINE3, NECK), (NECK, HEAD),
    (NECK, L_SHOULDER), (NECK, R_SHOULDER),
    (L_SHOULDER, L_ELBOW), (L_ELBOW, L_WRIST),
    (R_SHOULDER, R_ELBOW), (R_ELBOW, R_WRIST),
)
#: Five fingers per hand, each a three-joint chain rooted at the wrist.
_FINGER_CHAINS = tuple(
    tuple(range(base, base + 3)) for base in range(0, 15, 3)
)


def _hand_edges(wrist: int, joints: Sequence[int]) -> tuple[tuple[int, int], ...]:
    edges: list[tuple[int, int]] = []
    for chain in _FINGER_CHAINS:
        previous = wrist
        for offset in chain:
            edges.append((previous, joints[offset]))
            previous = joints[offset]
    return tuple(edges)


LEFT_HAND_EDGES = _hand_edges(L_WRIST, L_HAND_JOINTS)
RIGHT_HAND_EDGES = _hand_edges(R_WRIST, R_HAND_JOINTS)

#: 2D keypoints the crop is built from, in two groups. Legs and the 68 face
#: points are excluded: legs are out of scope, and the face block would swamp
#: every percentile with 68 points that all sit in one place.
CROP_BODY_POINTS: tuple[int, ...] = COCO_UPPER_BODY
CROP_HAND_POINTS: tuple[int, ...] = COCO_LEFT_HAND + COCO_RIGHT_HAND
CROP_POINTS: tuple[int, ...] = CROP_BODY_POINTS + CROP_HAND_POINTS


@dataclass(frozen=True)
class CardLayout:
    columns: int = 6
    bands: int = 2
    crop_width: int = 250
    crop_height: int = 250
    pose_height: int = 200
    header_height: int = 64
    timeline_height: int = 150
    margin: int = 6

    @property
    def thumbnails(self) -> int:
        return self.columns * self.bands

    @property
    def width(self) -> int:
        return self.columns * self.crop_width

    @property
    def height(self) -> int:
        band = self.crop_height + self.pose_height
        return self.header_height + self.bands * band + self.timeline_height


def _text(
    canvas: np.ndarray,
    message: str,
    origin: tuple[int, int],
    *,
    scale: float = 0.42,
    color: tuple[int, int, int] = TEXT,
    thickness: int = 1,
) -> None:
    cv2.putText(canvas, message, origin, cv2.FONT_HERSHEY_DUPLEX, scale, color, thickness, cv2.LINE_AA)


def upper_body_crop_box(
    keypoints: np.ndarray,
    frame_indices: Sequence[int],
    width: int,
    height: int,
    *,
    pad: float = 0.10,
    aspect: float = 1.0,
) -> tuple[int, int, int, int]:
    """A single crop rectangle covering the upper body across the shown frames.

    One box for the whole card rather than a per-frame box: a crop that chases
    the hands makes every thumbnail a different scale, and the reviewer then
    cannot tell a big gesture from a zoom. Percentile bounds rather than min/max
    so one stray keypoint does not blow the crop out to the whole raster.

    The returned box always has the requested ``aspect`` and always lies inside
    the raster. Getting that wrong is not cosmetic: an earlier version clamped
    the box to the raster edge without re-deriving the other side, so a wide
    gesture produced a 1080x1848 crop that ffmpeg then squashed into a 250x200
    panel — and the skeleton, drawn with a single isotropic scale, landed a body
    length below the participant. Two reviewers read that as broken SMPL-H
    tracking on files whose tracking was fine.
    """

    # The body and the hands are percentiled separately and then unioned. Pooled,
    # the 42 hand points outvote the 9 body points four to one, so the head — one
    # point in twelve frames, and the topmost — falls below the 5th percentile and
    # is cropped away. The head is worth keeping: it is the fastest way to see
    # that the drawn skeleton is on the participant at all.
    def box(group: tuple[int, ...]) -> tuple[float, float, float, float] | None:
        block = keypoints[list(frame_indices)][:, list(group), :]
        good = (block[..., 2] > 0) & ((block[..., :2] != 0).any(axis=-1))
        if good.sum() < 4:
            return None
        xs, ys = block[..., 0][good], block[..., 1][good]
        return (*np.percentile(xs, [5, 95]), *np.percentile(ys, [5, 95]))

    body, hands = box(CROP_BODY_POINTS), box(CROP_HAND_POINTS)
    corners = [b for b in (body, hands) if b is not None]
    if not corners:
        return _fit_aspect(0, 0, width, height, width, height, aspect)
    x0 = min(c[0] for c in corners)
    x1 = max(c[1] for c in corners)
    y0 = min(c[2] for c in corners)
    y1 = max(c[3] for c in corners)
    box_w = max(64.0, x1 - x0) * (1.0 + 2 * pad)
    box_h = max(64.0, y1 - y0) * (1.0 + 2 * pad)
    if box_w / box_h < aspect:
        box_w = box_h * aspect
    else:
        box_h = box_w / aspect
    centre_x, centre_y = 0.5 * (x0 + x1), 0.5 * (y0 + y1)
    return _fit_aspect(
        centre_x - box_w / 2, centre_y - box_h / 2, box_w, box_h, width, height, aspect
    )


def _fit_aspect(
    left: float, top: float, box_w: float, box_h: float,
    width: int, height: int, aspect: float,
) -> tuple[int, int, int, int]:
    """Shrink the box to fit the raster at the given aspect, then slide it inside."""

    box_w = min(box_w, float(width))
    box_h = min(box_h, float(height))
    # Re-derive whichever side the shrink broke, then shrink the other if that
    # in turn overflows. Two passes are enough because each only ever reduces.
    if box_w / box_h > aspect:
        box_w = box_h * aspect
    else:
        box_h = box_w / aspect
    if box_w > width:
        box_w, box_h = float(width), width / aspect
    if box_h > height:
        box_h, box_w = float(height), height * aspect

    left = min(max(0.0, left), width - box_w)
    top = min(max(0.0, top), height - box_h)
    # ffmpeg's crop filter wants even offsets and sizes on yuv420p input.
    x = int(left) & ~1
    y = int(top) & ~1
    w = max(32, int(box_w) & ~1)
    h = max(32, int(box_h) & ~1)
    w = min(w, width - x)
    h = min(h, height - y)
    return x, y, x + w, y + h


#: Joints the pose panel draws, and therefore the joints it frames on.
DRAWN_JOINTS: tuple[int, ...] = (
    PELVIS, SPINE3, NECK, HEAD, L_SHOULDER, R_SHOULDER,
    L_ELBOW, R_ELBOW, L_WRIST, R_WRIST,
) + tuple(L_HAND_JOINTS) + tuple(R_HAND_JOINTS)

#: Pixels per metre when the pose fits without shrinking. Chosen so that a
#: person with their hands at their sides fills the panel: head-to-hands is
#: about 0.95 m on the zero-beta skeleton every file shares.
REFERENCE_SCALE_PX_PER_M = 195.0


def draw_pose_panel(
    joints: np.ndarray,
    width: int,
    height: int,
    *,
    yaw_degrees: float = 28.0,
    valid: bool = True,
) -> np.ndarray:
    """Orthographic three-quarter view of the pelvis-frame upper body.

    A three-quarter yaw rather than a front view: a frontal orthographic
    projection hides motion toward and away from the camera, and a large part of
    what a co-speech gesture does is exactly that.

    The scale is *fixed* at :data:`REFERENCE_SCALE_PX_PER_M` unless the pose does
    not fit, in which case it shrinks just enough. Keeping it fixed is what makes
    two panels comparable — a panel that always auto-fitted would draw a small
    gesture and a large one at the same apparent size, which is the one thing the
    reviewer is being asked to tell apart. The shrink factor is printed when it
    fires, so a rescaled panel is never mistaken for a normal one.
    """

    canvas = np.full((height, width, 3), PANEL, dtype=np.uint8)
    if joints is None or not np.isfinite(joints).all():
        _text(canvas, "no pose", (8, height // 2), color=MUTED)
        return canvas

    yaw = math.radians(yaw_degrees)
    rotation = np.array(
        [[math.cos(yaw), 0.0, math.sin(yaw)], [0.0, 1.0, 0.0], [-math.sin(yaw), 0.0, math.cos(yaw)]]
    )
    points = joints @ rotation.T
    drawn = points[list(DRAWN_JOINTS)]
    centre = np.array([drawn[:, 0].mean(), 0.5 * (drawn[:, 1].max() + drawn[:, 1].min())])
    extent_x = max(1e-3, drawn[:, 0].max() - drawn[:, 0].min())
    extent_y = max(1e-3, drawn[:, 1].max() - drawn[:, 1].min())
    scale = min(
        REFERENCE_SCALE_PX_PER_M,
        (width - 12) / extent_x,
        (height - 22) / extent_y,
    )
    xs = (points[:, 0] - centre[0]) * scale + width / 2.0
    # SMPL-H y grows upward; image y grows down.
    ys = -(points[:, 1] - centre[1]) * scale + height / 2.0

    def pixel(index: int) -> tuple[int, int]:
        return int(round(xs[index])), int(round(ys[index]))

    shoulder_y = int(round(0.5 * (ys[L_SHOULDER] + ys[R_SHOULDER])))
    cv2.line(canvas, (0, shoulder_y), (width, shoulder_y), (52, 52, 52), 1)
    for edges, color, thickness in (
        (BODY_EDGES, SKELETON, 2),
        (LEFT_HAND_EDGES, LEFT_HAND_COLOR, 1),
        (RIGHT_HAND_EDGES, RIGHT_HAND_COLOR, 1),
    ):
        for first, second in edges:
            cv2.line(canvas, pixel(first), pixel(second), color, thickness, cv2.LINE_AA)
    cv2.circle(canvas, pixel(HEAD), 9, SKELETON, 1, cv2.LINE_AA)
    for index in (L_WRIST, R_WRIST):
        cv2.circle(canvas, pixel(index), 3, (255, 255, 255), -1, cv2.LINE_AA)
    if scale < REFERENCE_SCALE_PX_PER_M - 1e-6:
        _text(canvas, f"x{scale / REFERENCE_SCALE_PX_PER_M:.2f}", (width - 40, height - 5),
              scale=0.34, color=MUTED)
    if not valid:
        cv2.rectangle(canvas, (0, 0), (width - 1, height - 1), (40, 40, 210), 2)
        _text(canvas, "smplh invalid", (6, 14), scale=0.36, color=(90, 90, 235))
    return canvas


def draw_keypoint_overlay(
    frame: np.ndarray,
    keypoints: np.ndarray,
    offset: tuple[int, int],
    scale: tuple[float, float],
) -> None:
    """Draw the released 2D arms and hands onto an already-cropped frame.

    Arms and hands only. The v0 overlay drew the full body-17 skeleton including
    a torso quadrilateral and a triangle over the face, and at this crop scale it
    covered the very thing the reviewer is looking at. What has to be checkable
    here is whether the tracked arm is on the participant's arm.
    """

    def point(index: int) -> tuple[int, int] | None:
        x, y, confidence = keypoints[index]
        if confidence <= 0 or (x == 0 and y == 0):
            return None
        return int(round((x - offset[0]) * scale[0])), int(round((y - offset[1]) * scale[1]))

    for first, second in ((5, 7), (7, 9), (6, 8), (8, 10), (5, 6)):
        a, b = point(first), point(second)
        if a and b:
            cv2.line(frame, a, b, SKELETON, 1, cv2.LINE_AA)
    for index in (5, 6, 7, 8):
        spot = point(index)
        if spot:
            cv2.circle(frame, spot, 2, SKELETON, -1, cv2.LINE_AA)
    for hand, wrist, color in (
        (COCO_LEFT_HAND, 9, LEFT_HAND_COLOR),
        (COCO_RIGHT_HAND, 10, RIGHT_HAND_COLOR),
    ):
        for index in hand:
            spot = point(index)
            if spot:
                cv2.circle(frame, spot, 1, color, -1, cv2.LINE_AA)
        spot = point(wrist)
        if spot:
            cv2.circle(frame, spot, 3, color, 1, cv2.LINE_AA)


def draw_timeline(
    tracks: GestureTracks,
    spans: Sequence[tuple[int, int]],
    marks: Sequence[int],
    partner_speech: np.ndarray | None,
    width: int,
    height: int,
) -> np.ndarray:
    """Whole-recording speech, gesture and arm-speed on one time axis."""

    canvas = np.full((height, width, 3), PANEL, dtype=np.uint8)
    frames = tracks.frames
    if frames < 2:
        return canvas

    def column(index: int) -> int:
        return int(round(index / (frames - 1) * (width - 1)))

    speech_top, speech_h = 16, 14
    partner_top, partner_h = 32, 9
    trace_top, trace_h = 46, height - 46 - 34
    span_top, span_h = height - 30, 11

    from .gesture import _runs  # local import keeps the run-length helper in one place

    for start, stop in _runs(tracks.speech):
        cv2.rectangle(canvas, (column(start), speech_top), (column(stop - 1), speech_top + speech_h), SPEECH, -1)
    if partner_speech is not None and len(partner_speech) == frames:
        for start, stop in _runs(partner_speech):
            cv2.rectangle(
                canvas, (column(start), partner_top), (column(stop - 1), partner_top + partner_h), PARTNER, -1
            )
    for start, stop in _runs(tracks.active):
        cv2.rectangle(
            canvas, (column(start), trace_top), (column(stop - 1), trace_top + trace_h), (34, 62, 34), -1
        )

    ceiling = max(120.0, float(np.percentile(tracks.arm_speed, 98)))
    columns = np.linspace(0, frames - 1, width).astype(int)
    values = tracks.arm_speed[columns] / ceiling
    ys = (trace_top + trace_h - np.clip(values, 0, 1) * trace_h).astype(int)
    polyline = np.stack([np.arange(width), ys], axis=1).astype(np.int32)
    cv2.polylines(canvas, [polyline], False, TRACE, 1, cv2.LINE_AA)

    for start, stop in spans:
        cv2.rectangle(canvas, (column(start), span_top), (column(stop - 1), span_top + span_h), ACCEPTED, -1)
    for index in marks:
        cv2.line(canvas, (column(index), span_top - 4), (column(index), span_top), (200, 200, 200), 1)

    _text(canvas, "whole recording:", (4, 11), scale=0.36, color=TEXT)
    _text(canvas, "own speech", (120, 11), scale=0.34, color=(220, 200, 170))
    _text(canvas, "partner speech", (215, 11), scale=0.34, color=(180, 190, 210))
    _text(canvas, f"arm speed (0-{ceiling:.0f} mm/s)", (335, 11), scale=0.34, color=TEXT)
    _text(canvas, "gesture episode", (500, 11), scale=0.34, color=(140, 220, 140))
    _text(canvas, "accepted spans", (625, 11), scale=0.34, color=(120, 220, 120))
    _text(canvas, "| thumbnails marked with ticks", (740, 11), scale=0.34, color=MUTED)
    seconds_total = int(frames / tracks.fps)
    for second in range(0, seconds_total + 1, 30):
        x = column(int(second * tracks.fps))
        tall = second % 60 == 0
        cv2.line(canvas, (x, height - 16), (x, height - 16 + (7 if tall else 4)), MUTED, 1)
        if tall:
            _text(canvas, f"{second // 60}:00", (max(0, x - 12), height - 2), scale=0.32, color=MUTED)
    return canvas


def compose_card(
    thumbnails: Sequence[np.ndarray],
    poses: Sequence[np.ndarray],
    timeline: np.ndarray,
    header_lines: tuple[str, str],
    layout: CardLayout,
) -> np.ndarray:
    """Assemble header, alternating crop/pose bands, and the timeline."""

    canvas = np.full((layout.height, layout.width, 3), BACKGROUND, dtype=np.uint8)
    _text(canvas, header_lines[0], (8, 22), scale=0.52)
    _text(canvas, header_lines[1], (8, 46), scale=0.42, color=MUTED)

    y = layout.header_height
    for band in range(layout.bands):
        for column in range(layout.columns):
            index = band * layout.columns + column
            if index >= len(thumbnails):
                continue
            x = column * layout.crop_width
            canvas[y : y + layout.crop_height, x : x + layout.crop_width] = thumbnails[index]
            canvas[
                y + layout.crop_height : y + layout.crop_height + layout.pose_height,
                x : x + layout.crop_width,
            ] = poses[index]
        y += layout.crop_height + layout.pose_height
    canvas[y : y + layout.timeline_height] = timeline
    return canvas

"""The 30-second review clip: video, reprojected SMPL-H, 3D pose, and audio.

This is the second of the two review artefacts. The first is the card
(:mod:`seamless_curation.review_card`), which is what a reviewer reads for most
items; this is what they play when the card leaves a doubt, and it is the only
artefact that carries **sound**, so it is what an audio-motion synchronisation
judgement rests on.

The layout is inherited from the v0 galleries and kept because it was validated
there: a whole-file filmstrip band, Panel A (the video frame with the released
2D keypoints and the S/B/M validity dots), Panel B (the same frame with the
SMPL-H forward-kinematic skeleton reprojected onto it), Panel C (two orthographic
views of the root-relative joints), and a waveform strip shaded with the
participant's own VAD and their partner's. What is *not* inherited is where the
clip sits in the recording: v0 placed it by a seeded uniform hash and 17.9% of
its clips contained no speech at all, whereas the window here comes from the
candidate selection and is speech-anchored by construction.

The renderer deliberately knows nothing about SMPL-H camera conventions. A
validated :class:`ReviewJointProvider` must supply projected and root-relative
joints; the unavailable provider makes Panels B and C visibly unavailable rather
than drawing a plausible but unverified skeleton.
"""

from __future__ import annotations

import hashlib
import importlib
import json
import math
import os
import re
import subprocess
from dataclasses import asdict, dataclass, replace
from fractions import Fraction
from pathlib import Path
from typing import Any, Iterator, Mapping, Protocol, Sequence

import cv2
import numpy as np
import soundfile as sf

from .media_repair import (
    AnnotationTimebase,
    RasterRepair,
    body_roll_deg,
    ffmpeg_repair_filter,
    pillarbox_for,
    quarter_turns_from_roll,
    repair_box,
    repair_camera_joints,
    repair_frame,
    repair_points,
)


BODY_EDGES: tuple[tuple[int, int], ...] = (
    (0, 1), (0, 2), (1, 3), (2, 4), (5, 6), (5, 7), (7, 9),
    (6, 8), (8, 10), (5, 11), (6, 12), (11, 12), (11, 13),
    (13, 15), (12, 14), (14, 16),
)
HAND_EDGES: tuple[tuple[int, int], ...] = (
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
)

GROUP_COLORS = {
    "body": (0, 230, 255),       # BGR yellow
    "left_hand": (255, 130, 30), # BGR blue
    "right_hand": (35, 110, 255),# BGR orange
    "unknown": (220, 220, 220),
}
MASK_COLORS = {
    True: (40, 205, 40),
    False: (35, 35, 230),
    None: (125, 125, 125),
}

# Overlays are drawn on the native-resolution frame and the panel is then
# downscaled to fit a 480-px-tall canvas, which for a 2160x3840 V03 file is a
# 10x reduction. A thickness chosen in native pixels therefore lands well below
# one output pixel and INTER_AREA averages the skeleton away. Thickness is
# instead chosen so it survives the known downscale: the whole point of Panels A
# and B is that a human can see whether the skeleton fits the person.
TARGET_OUTPUT_LINE_PX = 2.0

# How many frames a container may fall short of its annotation grid before the
# renderer treats it as a fault rather than as rounding. Matches
# AnnotationTimebase.is_consistent, which tolerates one frame either way.
MAX_TAIL_HOLD_FRAMES = 2


def overlay_thickness(width: int, height: int, display_scale: float = 1.0) -> int:
    """Native-pixel line thickness that renders as ~2 px after downscaling."""

    if not math.isfinite(display_scale) or display_scale <= 0:
        display_scale = 1.0
    native = max(2.0, min(width, height) / 650.0)
    return int(max(2, math.ceil(max(native, TARGET_OUTPUT_LINE_PX / display_scale))))


def panel_display_scale(source_w: int, source_h: int, width: int, height: int) -> float:
    """The factor ``_fit_panel`` will apply, so callers can pre-compensate."""

    if source_w <= 0 or source_h <= 0:
        return 1.0
    return min(width / source_w, height / source_h)


@dataclass(frozen=True)
class VideoProbe:
    width: int
    height: int
    fps: Fraction
    frame_count: int
    duration_s: float
    has_audio: bool
    # The rate the released per-frame arrays are sampled on. Usually identical
    # to `fps`, but on a variable-rate file the average and the nominal diverge
    # and only the nominal one indexes the annotations correctly.
    nominal_fps: Fraction = Fraction(0)
    # Pixel aspect the container asks a player to apply. V01 stores three
    # anamorphic rasters; every other vendor stores square pixels.
    sample_aspect: float = 1.0


@dataclass(frozen=True)
class JointSequence:
    """Validated FK/projection output for one contiguous frame interval.

    ``projected_xy_px`` is in the native released-video raster.  The 3D array
    is pelvis-subtracted and in metres.  Both arrays use the same joint order,
    described by ``groups`` and connected by ``edges``.
    """

    projected_xy_px: np.ndarray | None
    root_relative_xyz_m: np.ndarray | None
    edges: tuple[tuple[int, int], ...]
    groups: tuple[str, ...]
    status: str

    def validate(self, expected_frames: int) -> None:
        arrays = (self.projected_xy_px, self.root_relative_xyz_m)
        for array, final_dim, name in (
            (arrays[0], 2, "projected_xy_px"),
            (arrays[1], 3, "root_relative_xyz_m"),
        ):
            if array is None:
                continue
            if array.ndim != 3 or array.shape[0] != expected_frames or array.shape[2] != final_dim:
                raise ValueError(
                    f"{name} must have shape ({expected_frames}, J, {final_dim}); "
                    f"got {array.shape}"
                )
            if len(self.groups) != array.shape[1]:
                raise ValueError(f"groups has {len(self.groups)} entries for {array.shape[1]} joints")
        if arrays[0] is not None and arrays[1] is not None and arrays[0].shape[1] != arrays[1].shape[1]:
            raise ValueError("projected and root-relative joint counts differ")
        joint_count = len(self.groups)
        if any(a < 0 or b < 0 or a >= joint_count or b >= joint_count for a, b in self.edges):
            raise ValueError("joint edge index outside provider joint order")


class ReviewJointProvider(Protocol):
    """Acceptance-tested FK/projection adapter consumed by the renderer."""

    name: str

    def load(
        self,
        source_base: Path,
        start_frame: int,
        end_frame: int,
        raster_size_wh: tuple[int, int],
    ) -> JointSequence:
        ...


class UnavailableJointProvider:
    name = "unavailable"

    def __init__(self, reason: str = "M-4 reprojection acceptance test has not passed") -> None:
        self.reason = reason

    def load(
        self,
        source_base: Path,
        start_frame: int,
        end_frame: int,
        raster_size_wh: tuple[int, int],
    ) -> JointSequence:
        del source_base, start_frame, end_frame, raster_size_wh
        return JointSequence(None, None, (), (), f"unavailable: {self.reason}")


def load_joint_provider(spec: Mapping[str, Any] | None) -> ReviewJointProvider:
    """Load a trusted local provider factory, or return explicit unavailable.

    A factory is written as ``package.module:function`` and receives the
    ``settings`` mapping.  This hook is intentionally isolated so the renderer
    cannot accidentally implement its own camera interpretation.
    """

    spec = dict(spec or {})
    factory_name = spec.get("factory")
    if not factory_name:
        return UnavailableJointProvider(str(spec.get("unavailable_reason", "M-4 not accepted")))
    if not isinstance(factory_name, str) or ":" not in factory_name:
        raise ValueError("joint_provider.factory must be 'module:function'")
    module_name, attribute = factory_name.split(":", 1)
    factory = getattr(importlib.import_module(module_name), attribute)
    provider = factory(dict(spec.get("settings") or {}))
    if not hasattr(provider, "load") or not hasattr(provider, "name"):
        raise TypeError("joint provider factory returned an incompatible object")
    return provider


@dataclass(frozen=True)
class RenderSettings:
    # Round 4: 30 s, up from 10. A ten-second excerpt was too short to judge
    # whether a gesture pattern holds, and the reviewer asked for more context.
    duration_s: float = 30.0
    output_height: int = 480
    audio_strip_height: int = 96
    max_video_panel_width: int = 680
    skeleton_panel_width: int = 420
    skeleton_half_extent_m: float = 1.2
    crf: int = 23
    preset: str = "fast"
    # Whole-file filmstrip above the clip, for the file-level questions even a
    # thirty-second window cannot answer. Ten rather than twelve: at twelve the
    # thumbnails were too small to read.
    filmstrip_thumbnails: int = 10
    # Stand the torso axis vertical in Panel C's side view. V00's camera is
    # angled down, and Panel C plots camera-frame coordinates, so without this
    # an upright participant is drawn leaning by a median 13 degrees. See
    # upright_joints; it is a rigid rotation and changes no joint angle.
    upright_side_view: bool = True
    # Apply the container's pixel aspect and any quarter turn before drawing.
    # Off only for a deliberate before-and-after; see media_repair for why the
    # squeeze fixes the picture and the 2D points but not the released SMPL-H.
    repair_raster: bool = True
    # Seconds between keyframes in the rendered clip. libx264's default GOP put
    # three keyframes in a thirty-second clip, which made the scrubber unusable.
    keyframe_seconds: float = 1.0

    @property
    def visual_height(self) -> int:
        return self.output_height - self.audio_strip_height

    def validate(self) -> None:
        # The clip length used to be pinned to exactly 10 s so it could not
        # drift silently. It is now a real setting, but still bounded: a clip
        # long enough to lose the reviewer's attention defeats a review gallery,
        # and the length is part of the render fingerprint either way.
        if not math.isfinite(self.duration_s) or not 1.0 <= self.duration_s <= 120.0:
            raise ValueError("duration_s must be a finite number of seconds in [1, 120]")
        if self.output_height < 240 or self.output_height % 2:
            raise ValueError("output_height must be an even integer >= 240")
        if self.audio_strip_height < 64 or self.audio_strip_height >= self.output_height:
            raise ValueError("audio_strip_height must be >=64 and smaller than output_height")
        if self.visual_height % 2:
            raise ValueError("visual panel height must be even")
        if self.skeleton_half_extent_m <= 0:
            raise ValueError("skeleton_half_extent_m must be positive")
        if self.filmstrip_thumbnails and self.filmstrip_thumbnails < 2:
            raise ValueError("filmstrip_thumbnails must be 0 (disabled) or at least 2")
        if not math.isfinite(self.keyframe_seconds) or not 0.1 <= self.keyframe_seconds <= 10.0:
            raise ValueError("keyframe_seconds must be in [0.1, 10]")


def probe_video(path: Path) -> VideoProbe:
    command = [
        "ffprobe", "-v", "error", "-select_streams", "v:0",
        "-show_entries",
        "stream=width,height,avg_frame_rate,r_frame_rate,sample_aspect_ratio,nb_frames,duration"
        ":format=duration",
        "-of", "json", str(path),
    ]
    result = subprocess.run(command, check=True, capture_output=True, text=True)
    payload = json.loads(result.stdout)
    streams = payload.get("streams") or []
    if not streams:
        raise ValueError("no video stream")
    stream = streams[0]
    fps_text = str(stream.get("avg_frame_rate", "0/0"))
    try:
        fps = Fraction(fps_text)
    except (ValueError, ZeroDivisionError) as exc:
        raise ValueError(f"unusable avg_frame_rate {fps_text!r}") from exc
    if fps <= 0:
        raise ValueError(f"unusable avg_frame_rate {fps_text!r}")
    try:
        nominal_fps = Fraction(str(stream.get("r_frame_rate", "0/0")))
    except (ValueError, ZeroDivisionError):
        nominal_fps = Fraction(0)
    if nominal_fps <= 0:
        nominal_fps = fps
    try:
        sample_aspect = float(Fraction(str(stream.get("sample_aspect_ratio", "1:1")).replace(":", "/")))
    except (ValueError, ZeroDivisionError):
        sample_aspect = 1.0
    if sample_aspect <= 0:
        sample_aspect = 1.0
    duration_s = float(stream.get("duration") or payload.get("format", {}).get("duration") or 0.0)
    frame_text = stream.get("nb_frames")
    frame_count = int(frame_text) if frame_text not in (None, "N/A") else int(round(duration_s * float(fps)))
    audio_probe = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "a:0", "-show_entries", "stream=index", "-of", "csv=p=0", str(path)],
        check=True,
        capture_output=True,
        text=True,
    )
    return VideoProbe(
        width=int(stream["width"]),
        height=int(stream["height"]),
        fps=fps,
        frame_count=frame_count,
        duration_s=duration_s,
        has_audio=bool(audio_probe.stdout.strip()),
        nominal_fps=nominal_fps,
        sample_aspect=sample_aspect,
    )


def _safe_stem(value: str) -> str:
    stem = re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._")
    if not stem:
        raise ValueError(f"identifier has no safe filename characters: {value!r}")
    return stem[:180]


def _point(keypoints: np.ndarray, index: int) -> tuple[int, int] | None:
    x, y, confidence = keypoints[index]
    if not np.isfinite((x, y, confidence)).all() or confidence <= 0:
        return None
    return int(round(float(x))), int(round(float(y)))


def _draw_edges_2d(
    frame: np.ndarray,
    keypoints: np.ndarray,
    edges: Sequence[tuple[int, int]],
    offset: int,
    color: tuple[int, int, int],
    thickness: int,
) -> None:
    height, width = frame.shape[:2]
    for first, second in edges:
        p1, p2 = _point(keypoints, offset + first), _point(keypoints, offset + second)
        if p1 is None or p2 is None:
            continue
        if 0 <= p1[0] < width and 0 <= p1[1] < height and 0 <= p2[0] < width and 0 <= p2[1] < height:
            cv2.line(frame, p1, p2, color, thickness, cv2.LINE_AA)


def draw_released_overlay(
    frame: np.ndarray,
    keypoints: np.ndarray,
    box: np.ndarray,
    smplh_valid: bool | None,
    box_valid: bool | None,
    movement_valid: bool | None,
    *,
    draw_indicators: bool = True,
    display_scale: float = 1.0,
) -> np.ndarray:
    output = frame.copy()
    height, width = output.shape[:2]
    thickness = overlay_thickness(width, height, display_scale)
    radius = max(2, thickness + 1)
    if np.asarray(box).shape == (4,) and np.isfinite(box).all():
        x1, y1, x2, y2 = (int(round(float(value))) for value in box)
        cv2.rectangle(output, (x1, y1), (x2, y2), MASK_COLORS[box_valid], thickness)
    _draw_edges_2d(output, keypoints, BODY_EDGES, 0, GROUP_COLORS["body"], thickness)
    _draw_edges_2d(output, keypoints, HAND_EDGES, 91, GROUP_COLORS["left_hand"], thickness)
    _draw_edges_2d(output, keypoints, HAND_EDGES, 112, GROUP_COLORS["right_hand"], thickness)
    for start, stop, group in ((0, 17, "body"), (91, 112, "left_hand"), (112, 133, "right_hand")):
        for index in range(start, stop):
            point = _point(keypoints, index)
            if point is not None and 0 <= point[0] < width and 0 <= point[1] < height:
                cv2.circle(output, point, radius, GROUP_COLORS[group], -1, cv2.LINE_AA)
    if draw_indicators:
        draw_mask_dots(output, smplh_valid, box_valid, movement_valid)
    return output


def draw_mask_dots(
    frame: np.ndarray,
    smplh_valid: bool | None,
    box_valid: bool | None,
    movement_valid: bool | None,
) -> None:
    height, width = frame.shape[:2]
    scale = max(0.65, min(width, height) / 900)
    radius = max(9, int(round(13 * scale)))
    spacing = radius * 3
    x0, y = width - spacing * 3 - radius, radius * 2 + 30
    overlay = frame.copy()
    cv2.rectangle(overlay, (x0 - radius, y - radius * 2), (width - 4, y + radius * 2), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)
    for offset, (label, value) in enumerate((("S", smplh_valid), ("B", box_valid), ("M", movement_valid))):
        center = (x0 + offset * spacing, y)
        cv2.circle(frame, center, radius, MASK_COLORS[value], -1, cv2.LINE_AA)
        cv2.putText(frame, label, (center[0] - radius // 2, center[1] + radius // 2), cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), max(1, radius // 5), cv2.LINE_AA)


def _joint_color(groups: Sequence[str], first: int, second: int) -> tuple[int, int, int]:
    for index in (first, second):
        group = groups[index] if index < len(groups) else "unknown"
        if group in ("left_hand", "right_hand"):
            return GROUP_COLORS[group]
    return GROUP_COLORS.get(groups[first] if first < len(groups) else "unknown", GROUP_COLORS["unknown"])


def draw_projected_joints(
    frame: np.ndarray,
    xy_px: np.ndarray | None,
    edges: Sequence[tuple[int, int]],
    groups: Sequence[str],
    status: str,
    display_scale: float = 1.0,
) -> np.ndarray:
    output = frame.copy()
    height, width = output.shape[:2]
    thickness = overlay_thickness(width, height, display_scale)
    if xy_px is None:
        output = (output.astype(np.float32) * 0.30).astype(np.uint8)
        _centered_message(output, ("SMPL-H projection unavailable", status))
        return output
    for first, second in edges:
        p1, p2 = xy_px[first], xy_px[second]
        if not np.isfinite(p1).all() or not np.isfinite(p2).all():
            continue
        points = tuple(int(round(float(v))) for v in (*p1, *p2))
        if 0 <= points[0] < width and 0 <= points[1] < height and 0 <= points[2] < width and 0 <= points[3] < height:
            cv2.line(output, points[:2], points[2:], _joint_color(groups, first, second), thickness, cv2.LINE_AA)
    for index, point in enumerate(xy_px):
        if np.isfinite(point).all() and 0 <= point[0] < width and 0 <= point[1] < height:
            cv2.circle(output, tuple(np.rint(point).astype(int)), thickness + 1, GROUP_COLORS.get(groups[index], GROUP_COLORS["unknown"]), -1, cv2.LINE_AA)
    return output


def _centered_message(canvas: np.ndarray, lines: Sequence[str]) -> None:
    height, width = canvas.shape[:2]
    scale = max(0.42, min(0.75, width / 700))
    wrapped: list[str] = []
    max_width = max(40, width - 20)
    for line in lines:
        words = line.split()
        current = ""
        for word in words:
            candidate = f"{current} {word}".strip()
            candidate_width = cv2.getTextSize(candidate, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)[0][0]
            if current and candidate_width > max_width:
                wrapped.append(current)
                current = word
            else:
                current = candidate
        if current:
            wrapped.append(current)
    for line_index, shown in enumerate(wrapped):
        size, _ = cv2.getTextSize(shown, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)
        x = max(8, (width - size[0]) // 2)
        y = height // 2 + (line_index - (len(wrapped) - 1) / 2) * 28
        cv2.putText(canvas, shown, (x, int(y)), cv2.FONT_HERSHEY_SIMPLEX, scale, (230, 230, 230), 1, cv2.LINE_AA)


def _draw_orthographic_view(
    canvas: np.ndarray,
    joints_m: np.ndarray,
    edges: Sequence[tuple[int, int]],
    groups: Sequence[str],
    rect: tuple[int, int, int, int],
    horizontal_axis: int,
    title: str,
    half_extent_m: float,
) -> None:
    x0, y0, x1, y1 = rect
    cv2.rectangle(canvas, (x0, y0), (x1 - 1, y1 - 1), (65, 65, 65), 1)
    cv2.putText(canvas, title, (x0 + 8, y0 + 48), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (220, 220, 220), 1, cv2.LINE_AA)
    center_x, center_y = (x0 + x1) / 2, (y0 + y1) / 2 + 12
    scale = 0.44 * min(x1 - x0, y1 - y0) / half_extent_m

    def project(point: np.ndarray) -> tuple[int, int] | None:
        if not np.isfinite(point).all():
            return None
        # M-4's accepted camera-frame output has +y downward after the
        # observed near-pi global x rotation. Preserve anatomical orientation
        # on screen: head above the pelvis and feet below it.
        return int(round(center_x + point[horizontal_axis] * scale)), int(round(center_y + point[1] * scale))

    for first, second in edges:
        p1, p2 = project(joints_m[first]), project(joints_m[second])
        if p1 is not None and p2 is not None:
            cv2.line(canvas, p1, p2, _joint_color(groups, first, second), 2, cv2.LINE_AA)
    for index, point in enumerate(joints_m):
        projected = project(point)
        if projected is not None:
            cv2.circle(canvas, projected, 2, GROUP_COLORS.get(groups[index], GROUP_COLORS["unknown"]), -1, cv2.LINE_AA)


SMPLH_PELVIS_INDEX, SMPLH_NECK_INDEX = 0, 12


def upright_joints(joints_m: np.ndarray) -> np.ndarray:
    """Rotate about the camera x-axis so the torso axis stands vertical.

    Panel C plots **camera-frame** coordinates, so a camera that is not level
    draws an upright participant leaning. V00 was shot from above eye level,
    angled down: measured over 440 standing participants that pass all three
    checks, the median pelvis-to-neck axis sits 13.1 degrees out of the image
    plane, head toward the camera. That reads as a forward lean and it is the
    camera, not the person — within a recording session the angle varies by
    1.88 degrees against 5.10 degrees between sessions, which is the signature
    of a fixed rig rather than of posture.

    Undoing it costs one rotation and makes the side view answer the question it
    looks like it is answering. It changes nothing about the pose: a rigid
    rotation leaves every joint angle identical, so the residual knee bend stays
    visible — correctly, because that one is real and lives in the released
    parameters.
    """

    joints = np.asarray(joints_m, dtype=np.float64)
    if joints.ndim != 2 or joints.shape[0] <= SMPLH_NECK_INDEX:
        return joints
    axis = joints[SMPLH_NECK_INDEX] - joints[SMPLH_PELVIS_INDEX]
    if not np.isfinite(axis).all() or np.linalg.norm(axis) < 1e-9:
        return joints
    # Screen up is -y, so bring the axis onto -y by rotating in the z/y plane.
    angle = np.arctan2(axis[2], -axis[1])
    cos, sin = np.cos(angle), np.sin(angle)
    rotation = np.array([[1.0, 0.0, 0.0], [0.0, cos, -sin], [0.0, sin, cos]])
    return joints @ rotation.T


def draw_root_relative_panel(
    width: int,
    height: int,
    joints_m: np.ndarray | None,
    edges: Sequence[tuple[int, int]],
    groups: Sequence[str],
    status: str,
    half_extent_m: float,
    upright_side_view: bool = True,
) -> np.ndarray:
    canvas = np.full((height, width, 3), 18, dtype=np.uint8)
    if joints_m is None:
        _centered_message(canvas, ("Root-relative 3D unavailable", status))
        return canvas
    shown = upright_joints(joints_m) if upright_side_view else joints_m
    label = "side: z/y upright" if upright_side_view else "side: z/y"
    midpoint = width // 2
    _draw_orthographic_view(canvas, shown, edges, groups, (0, 0, midpoint, height), 0, "front: x/y", half_extent_m)
    _draw_orthographic_view(canvas, shown, edges, groups, (midpoint, 0, width, height), 2, label, half_extent_m)
    return canvas


def _fit_panel(frame: np.ndarray, width: int, height: int) -> np.ndarray:
    source_h, source_w = frame.shape[:2]
    scale = min(width / source_w, height / source_h)
    resized_w = max(2, int(round(source_w * scale)))
    resized_h = max(2, int(round(source_h * scale)))
    resized = cv2.resize(frame, (resized_w, resized_h), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    x, y = (width - resized_w) // 2, (height - resized_h) // 2
    canvas[y:y + resized_h, x:x + resized_w] = resized
    return canvas


def _grab_frame_at(
    video_path: Path, seconds: float, width: int, height: int, repair_filter: str = ""
) -> np.ndarray | None:
    """One scaled BGR frame near ``seconds``, via ffmpeg keyframe seek."""

    chain = f"{repair_filter},scale={width}:{height}" if repair_filter else f"scale={width}:{height}"
    command = [
        "ffmpeg", "-hide_banner", "-loglevel", "error",
        "-ss", f"{max(0.0, seconds):.3f}", "-i", str(video_path),
        "-frames:v", "1", "-vf", chain,
        "-f", "rawvideo", "-pix_fmt", "bgr24", "-",
    ]
    try:
        completed = subprocess.run(command, capture_output=True, timeout=60)
    except (OSError, subprocess.SubprocessError):
        return None
    expected = width * height * 3
    if completed.returncode != 0 or len(completed.stdout) < expected:
        return None
    return np.frombuffer(completed.stdout[:expected], dtype=np.uint8).reshape(height, width, 3)


def filmstrip_frame_targets(total_frames: int, count: int) -> list[int]:
    """Thumbnail frame indices: both endpoints included, the rest evenly spaced.

    The reviewer asked for the first thumbnail to be the very first frame of the
    recording and the last to be the very last, with the intermediate ones evenly
    spaced between them. The previous scheme sampled bin *centres*
    (``(i + 0.5) / count``), which deliberately excluded both ends — so a framing
    problem present only at the start or only at the end of a recording was
    invisible in the band that exists to show file-level behaviour.
    """

    last = max(0, int(total_frames) - 1)
    if count <= 1:
        return [0]
    return [int(round(index / (count - 1) * last)) for index in range(count)]


def _grab_frame_stepping_back(
    video_path: Path,
    seconds: float,
    width: int,
    height: int,
    *,
    step_s: float,
    attempts: int = 4,
    repair_filter: str = "",
) -> tuple[np.ndarray | None, float]:
    """``_grab_frame_at``, retreating a frame at a time if the seek finds nothing.

    Seeking to exactly the last frame's nominal timestamp fails on these files.
    Measured on V00_S0180_I00000482_P0047 (``nb_frames`` 9420, 30 fps, duration
    314.000 s): ``-ss 313.967`` — that is ``(n-1)/fps`` — returns no frame at all,
    while ``-ss 313.933`` returns one. ``nb_frames`` therefore over-counts the
    seekable frames by one here, and a naive "seek to the last frame" would draw a
    permanent grey ``?`` at the end of every filmstrip.

    Stepping back rather than clamping to a fixed offset keeps the thumbnail as
    close to the true end as the container actually allows, and the time that
    worked is returned so the printed timestamp stays honest.
    """

    for attempt in range(max(1, attempts)):
        candidate = seconds - attempt * step_s
        if candidate < 0:
            break
        frame = _grab_frame_at(video_path, candidate, width, height, repair_filter)
        if frame is not None:
            return frame, candidate
    return None, seconds


def build_filmstrip(
    video_path: Path,
    *,
    canvas_width: int,
    thumbnails: int,
    probe: VideoProbe,
    clip_start_frame: int,
    clip_frames: int,
    repair: RasterRepair | None = None,
) -> tuple[np.ndarray, list[float]]:
    """A strip of thumbnails spanning the whole recording, with a clip marker.

    Most of the reviewer's vocabulary is file-level: whether hands are static
    "for the entire conversation", or whether framing stays consistent, cannot be
    judged from one excerpt however long. Thumbnails are **full frames**, so a
    framing change across the recording stays visible.

    The first thumbnail is the first frame and the last is the last frame, with
    the rest evenly spaced between them; a red marker shows where the excerpt
    sits, so the clip is read in context rather than as an unplaceable fragment.
    """

    count = max(2, thumbnails)
    repair = repair or RasterRepair(probe.width, probe.height)
    display_width, display_height = repair.display_size
    thumb_w = max(24, canvas_width // count)
    thumb_h = max(16, int(round(thumb_w * display_height / display_width)))
    thumb_h += thumb_h % 2
    axis_h = 18
    band = np.zeros((thumb_h + axis_h, canvas_width, 3), dtype=np.uint8)

    total = max(1, probe.frame_count)
    fps = float(probe.fps)
    targets = filmstrip_frame_targets(total, count)
    times: list[float] = [target / fps for target in targets]
    # ffmpeg input-side `-ss` seeks to the nearest keyframe without decoding the
    # intervening frames. OpenCV's CAP_PROP_POS_FRAMES decodes forward from the
    # previous keyframe instead, which turned twelve thumbnails into minutes per
    # clip on cold NFS and dominated the whole render.
    for index, seconds_in in enumerate(times):
        x0 = index * thumb_w
        x1 = min(canvas_width, x0 + thumb_w)
        if x1 <= x0:
            break
        frame, seconds_in = _grab_frame_stepping_back(
            video_path, seconds_in, x1 - x0, thumb_h, step_s=1.0 / fps,
            repair_filter=ffmpeg_repair_filter(repair),
        )
        times[index] = seconds_in
        if frame is None:
            # A seek that fails is shown as an explicit gap, not skipped: a
            # missing thumbnail is itself information about the file.
            band[:thumb_h, x0:x1] = (28, 22, 40)
            cv2.putText(band, "?", (x0 + thumb_w // 2 - 4, thumb_h // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (140, 140, 200), 1, cv2.LINE_AA)
            continue
        band[:thumb_h, x0:x1] = frame
        cv2.rectangle(band, (x0, 0), (x1 - 1, thumb_h - 1), (70, 70, 70), 1)
        minutes, seconds = divmod(int(seconds_in), 60)
        cv2.putText(band, f"{minutes:d}:{seconds:02d}", (x0 + 3, thumb_h - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.32, (240, 240, 240), 1, cv2.LINE_AA)

    axis_top = thumb_h
    cv2.rectangle(band, (0, axis_top), (canvas_width, axis_top + axis_h), (18, 20, 26), -1)
    start_x = int(round(clip_start_frame / total * (canvas_width - 1)))
    end_x = int(round(min(total, clip_start_frame + clip_frames) / total * (canvas_width - 1)))
    cv2.line(band, (0, axis_top + axis_h // 2), (canvas_width, axis_top + axis_h // 2),
             (70, 70, 70), 1, cv2.LINE_AA)
    cv2.rectangle(band, (start_x, axis_top + 3), (max(start_x + 2, end_x), axis_top + axis_h - 4),
                  (60, 60, 235), -1)
    total_minutes, total_seconds = divmod(int(total / float(probe.fps)), 60)
    cv2.putText(band, f"whole file {total_minutes:d}:{total_seconds:02d} - red marks this clip",
                (6, axis_top + axis_h - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.36,
                (170, 178, 190), 1, cv2.LINE_AA)
    return band, times


def _panel_label(frame: np.ndarray, label: str) -> None:
    """Label sized to its own text and clipped to the panel it belongs to.

    A fixed-width badge overflowed into the neighbouring panel once a fourth
    panel made each one narrower, so the two labels ran together.
    """

    scale = 0.5
    (text_w, _), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)
    badge_w = min(frame.shape[1], text_w + 14)
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (badge_w, 26), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
    cv2.putText(frame, label, (6, 19), cv2.FONT_HERSHEY_SIMPLEX, scale, (245, 245, 245), 1, cv2.LINE_AA)


def load_vad(path: Path | None) -> list[tuple[float, float]]:
    if path is None or not path.exists():
        return []
    payload = json.loads(path.read_text(encoding="utf-8"))
    intervals: list[tuple[float, float]] = []
    for record in payload.get("metadata:vad", []):
        try:
            start, end = float(record["start"]), float(record["end"])
        except (KeyError, TypeError, ValueError):
            continue
        if math.isfinite(start) and math.isfinite(end) and end > start:
            intervals.append((start, end))
    return intervals


def waveform_envelope(path: Path, start_s: float, duration_s: float, width: int) -> tuple[np.ndarray, np.ndarray, int]:
    with sf.SoundFile(path) as audio:
        sample_rate = int(audio.samplerate)
        first = max(0, int(round(start_s * sample_rate)))
        audio.seek(min(first, len(audio)))
        samples = audio.read(int(math.ceil(duration_s * sample_rate)), dtype="float32", always_2d=True)
    mono = samples.mean(axis=1) if len(samples) else np.zeros(0, dtype=np.float32)
    if len(mono) == 0:
        return np.zeros(width, np.float32), np.zeros(width, np.float32), sample_rate
    boundaries = np.linspace(0, len(mono), width + 1, dtype=np.int64)
    lower = np.zeros(width, np.float32)
    upper = np.zeros(width, np.float32)
    for index in range(width):
        chunk = mono[boundaries[index]:boundaries[index + 1]]
        if len(chunk):
            lower[index], upper[index] = float(np.min(chunk)), float(np.max(chunk))
    peak = max(1e-6, float(np.max(np.abs(mono))))
    return np.clip(lower / peak, -1, 1), np.clip(upper / peak, -1, 1), sample_rate


def _shade_vad(
    image: np.ndarray,
    intervals: Sequence[tuple[float, float]],
    clip_start_s: float,
    duration_s: float,
    y0: int,
    y1: int,
    color: tuple[int, int, int],
) -> None:
    width = image.shape[1]
    overlay = image.copy()
    for start, end in intervals:
        clipped_start = max(start, clip_start_s)
        clipped_end = min(end, clip_start_s + duration_s)
        if clipped_end <= clipped_start:
            continue
        x0 = int(math.floor((clipped_start - clip_start_s) / duration_s * width))
        x1 = int(math.ceil((clipped_end - clip_start_s) / duration_s * width))
        cv2.rectangle(overlay, (max(0, x0), y0), (min(width - 1, x1), y1), color, -1)
    cv2.addWeighted(overlay, 0.32, image, 0.68, 0, image)


def audio_strip_background(
    width: int,
    height: int,
    lower: np.ndarray,
    upper: np.ndarray,
    own_vad: Sequence[tuple[float, float]],
    partner_vad: Sequence[tuple[float, float]],
    clip_start_s: float,
    duration_s: float,
    sample_rate: int,
) -> np.ndarray:
    image = np.full((height, width, 3), 24, dtype=np.uint8)
    midpoint = height // 2
    _shade_vad(image, own_vad, clip_start_s, duration_s, 0, midpoint - 1, (150, 75, 15))
    _shade_vad(image, partner_vad, clip_start_s, duration_s, midpoint, height - 1, (20, 95, 170))
    amplitude = max(5, int(height * 0.32))
    center = height // 2
    for x in range(min(width, len(lower), len(upper))):
        y0 = int(round(center - upper[x] * amplitude))
        y1 = int(round(center - lower[x] * amplitude))
        cv2.line(image, (x, y0), (x, y1), (225, 225, 225), 1)
    cv2.line(image, (0, center), (width - 1, center), (115, 115, 115), 1)
    cv2.putText(image, "own VAD", (8, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.43, (180, 205, 255), 1, cv2.LINE_AA)
    cv2.putText(image, "partner VAD", (8, height - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.43, (255, 195, 150), 1, cv2.LINE_AA)
    right_label = f"released WAV {sample_rate} Hz | {clip_start_s:.2f}-{clip_start_s + duration_s:.2f}s"
    label_size, _ = cv2.getTextSize(right_label, cv2.FONT_HERSHEY_SIMPLEX, 0.40, 1)
    cv2.putText(image, right_label, (max(8, width - label_size[0] - 8), 17), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (200, 200, 200), 1, cv2.LINE_AA)
    return image


def add_playhead(background: np.ndarray, progress: float) -> np.ndarray:
    frame = background.copy()
    x = int(round(np.clip(progress, 0, 1) * (frame.shape[1] - 1)))
    cv2.line(frame, (x, 0), (x, frame.shape[0] - 1), (40, 40, 245), 2, cv2.LINE_AA)
    return frame


def _mask_at(mask: np.ndarray | None, index: int) -> bool | None:
    if mask is None or index >= len(mask):
        return None
    return bool(mask[index])


def _raster_repair(
    probe: VideoProbe, keypoints: np.ndarray, settings: RenderSettings
) -> RasterRepair:
    """What to do to this file's frames before anyone looks at them.

    All three corrections are read off the released bundle rather than
    configured: the pixel aspect from the container, the quarter turn from the
    released keypoints, and the black columns from the stored raster. A file that
    needs none gets an identity repair and takes the unchanged path.
    """
    if not settings.repair_raster:
        return RasterRepair(probe.width, probe.height)
    roll = body_roll_deg(keypoints) if len(keypoints) else float("nan")
    return RasterRepair(
        width=probe.width,
        height=probe.height,
        sample_aspect=probe.sample_aspect,
        quarter_turns=quarter_turns_from_roll(roll),
        pillarbox=pillarbox_for(probe.width, probe.height),
    )


def _frames_at_annotation_times(
    capture: Any, start_index: int, count: int, timebase: AnnotationTimebase
) -> Iterator[np.ndarray]:
    """Yield the video frame that was on screen at each annotation's timestamp.

    When the two grids agree -- every file but a handful -- this is the plain
    sequential read the renderer has always done, so nothing changes for them.
    When they disagree the decoder is walked by presentation time instead, and a
    frame is held across a gap rather than letting the grids slide apart.
    """
    annotation_fps = timebase.nominal_fps
    if annotation_fps <= 0:
        raise ValueError("nominal_fps must be positive")
    if timebase.is_consistent:
        capture.set(cv2.CAP_PROP_POS_FRAMES, start_index)
        held: np.ndarray | None = None
        for offset in range(count):
            ok, frame = capture.read()
            if ok:
                held = frame
                yield frame
                continue
            # The container can hold a frame or two fewer than its annotation
            # grid -- inside the timebase tolerance, so this is the fast path --
            # and a clip that reaches the very end then runs out. Holding the
            # last frame for that is honest; failing the render is not. Anything
            # larger than the tolerance is a real problem and still raises.
            shortfall = count - offset
            if held is not None and shortfall <= MAX_TAIL_HOLD_FRAMES:
                for _ in range(shortfall):
                    yield held
                return
            raise RuntimeError(f"video decode stopped at source frame {start_index + offset}")
        return

    start_time = start_index / annotation_fps
    capture.set(cv2.CAP_PROP_POS_MSEC, max(0.0, start_time - 1.0) * 1000.0)
    held: np.ndarray | None = None
    pending: np.ndarray | None = None
    pending_time = -math.inf
    for offset in range(count):
        target = (start_index + offset) / annotation_fps
        while pending_time <= target:
            if pending is not None:
                held = pending
            # POS_MSEC reports the timestamp of the frame about to be decoded.
            position = capture.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
            ok, frame = capture.read()
            if not ok:
                pending, pending_time = None, math.inf
                break
            pending, pending_time = frame, position
        if held is None:
            held = pending
        if held is None:
            raise RuntimeError(f"video decode stopped before annotation {start_index + offset}")
        yield held


def _load_annotations(base: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray | None]:
    with np.load(base.with_suffix(".npz"), allow_pickle=False) as archive:
        keypoints = np.asarray(archive["boxes_and_keypoints:keypoints"])
        boxes = np.asarray(archive["boxes_and_keypoints:box"])
        box_valid = np.asarray(archive["boxes_and_keypoints:is_valid_box"]).reshape(-1).astype(bool)
        smplh_valid = np.asarray(archive["smplh:is_valid"]).reshape(-1).astype(bool)
        movement_valid = (
            np.asarray(archive["movement:is_valid"]).reshape(-1).astype(bool)
            if "movement:is_valid" in archive.files
            else None
        )
    return keypoints, boxes, smplh_valid, box_valid, movement_valid


def _render_fingerprint(record: Mapping[str, str], settings: RenderSettings, provider_name: str) -> str:
    relevant = {
        "source_relbase": record["source_relbase"],
        "start_frame": record["start_frame"],
        "settings": asdict(settings),
        "provider": provider_name,
        # v6: Panel C stands the torso axis vertical in the side view.
        # v7: pixel-aspect and quarter-turn repair, annotations indexed by
        # timestamp instead of frame number, one keyframe per second.
        # v8: the pillarbox crop, which v7 implemented but never applied.
        "renderer_version": 8,
    }
    return hashlib.sha256(json.dumps(relevant, sort_keys=True).encode()).hexdigest()


def _write_private_json(path: Path, payload: Any) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    os.chmod(temporary, 0o600)
    os.replace(temporary, path)
    os.chmod(path, 0o600)


def render_record(
    record: Mapping[str, str],
    source_root: Path,
    output_root: Path,
    settings: RenderSettings,
    provider: ReviewJointProvider,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Render one manifest record; all derivatives remain in ``output_root``."""

    settings.validate()
    if record.get("render_policy", "render") == "metadata_only":
        return {**record, "status": "metadata_only", "media_file": None, "joint_status": "not_run"}
    base = source_root / record["source_relbase"]
    stem = _safe_stem(record["clip_id"])
    final_path = output_root / f"{stem}.mp4"
    sidecar_path = output_root / f"{stem}.render.json"
    fingerprint = _render_fingerprint(record, settings, provider.name)
    if not overwrite and final_path.exists() and sidecar_path.exists():
        sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
        if sidecar.get("render_fingerprint") == fingerprint:
            return {**record, **sidecar, "status": "reused", "media_file": final_path.name}

    probe = probe_video(base.with_suffix(".mp4"))
    keypoints, boxes, smplh_valid, box_valid, movement_valid = _load_annotations(base)

    # `start_frame` indexes the released annotations, which are uniform at the
    # nominal rate. On a variable-rate file the container holds fewer frames than
    # that, so decoding by the same integer index slides the overlay off the
    # person -- 9.5 s by the end of V01_S1607_I00000135_P2569. Everything below
    # works in the annotation timebase and pulls the video frame that was on
    # screen at each annotation's timestamp.
    annotation_fps = float(probe.nominal_fps or probe.fps)
    timebase = AnnotationTimebase(
        nominal_fps=annotation_fps,
        annotation_frames=int(len(keypoints)),
        container_frames=int(probe.frame_count),
    )
    start_frame = int(record["start_frame"])
    render_frames = int(round(settings.duration_s * annotation_fps))
    end_frame = start_frame + render_frames
    if start_frame < 0 or end_frame > min(len(keypoints), len(boxes), len(smplh_valid), len(box_valid)):
        raise ValueError(
            f"full {settings.duration_s:g}-second interval is not covered by released annotations"
        )
    clip_start_s = start_frame / annotation_fps
    # The annotation grid can run a frame or two past the last stored frame; the
    # decoder holds the final frame across that, which is honest. A file whose
    # arrays run *seconds* past its video is a different animal -- 12 eligible
    # files carry arrays several times longer than the recording -- and freezing
    # on one frame for that long would read as a stuck video, so it is refused.
    overhang_s = (clip_start_s + settings.duration_s) - probe.duration_s
    if overhang_s > 1.0:
        raise ValueError(
            f"annotation interval [{clip_start_s:.2f}, {clip_start_s + settings.duration_s:.2f}) s "
            f"runs {overhang_s:.2f} s past the {probe.duration_s:.2f} s of video"
        )

    repair = _raster_repair(probe, keypoints, settings)
    repaired_width, repaired_height = repair.display_size
    if not repair.is_identity:
        keypoints = np.concatenate(
            (repair_points(keypoints[..., :2], repair), keypoints[..., 2:]), axis=-1
        )
        boxes = repair_box(boxes, repair)

    joints = provider.load(base, start_frame, end_frame, (probe.width, probe.height))
    joints.validate(render_frames)
    if not repair.is_identity:
        joints = replace(
            joints,
            projected_xy_px=None if joints.projected_xy_px is None
            else repair_points(joints.projected_xy_px, repair),
            root_relative_xyz_m=None if joints.root_relative_xyz_m is None
            else repair_camera_joints(joints.root_relative_xyz_m, repair),
        )
    visual_height = settings.visual_height
    native_scaled_width = int(round(repaired_width / repaired_height * visual_height))
    video_panel_width = max(120, min(settings.max_video_panel_width, native_scaled_width))
    video_panel_width += video_panel_width % 2
    skeleton_width = settings.skeleton_panel_width + settings.skeleton_panel_width % 2

    canvas_width = video_panel_width * 2 + skeleton_width
    canvas_width += canvas_width % 2

    display_scale = panel_display_scale(
        repaired_width, repaired_height, video_panel_width, visual_height
    )
    filmstrip: np.ndarray | None = None
    if settings.filmstrip_thumbnails:
        filmstrip, _ = build_filmstrip(
            base.with_suffix(".mp4"), canvas_width=canvas_width,
            thumbnails=settings.filmstrip_thumbnails, probe=probe,
            clip_start_frame=start_frame, clip_frames=render_frames,
            repair=repair,
        )
    wav_path = base.with_suffix(".wav")
    if not wav_path.exists():
        raise FileNotFoundError(f"released WAV is required for review audio: {wav_path}")
    # 183 eligible files ship a 58-byte WAV -- a RIFF header with no samples --
    # and none of them are V00, which is why this never came up. Muxing one with
    # `-shortest` silently truncated a 30 s clip to 20 s of video.
    audio_status = "released" if wav_path.stat().st_size > 1024 else "missing: released WAV is empty"
    lower, upper, sample_rate = waveform_envelope(wav_path, clip_start_s, settings.duration_s, canvas_width)
    partner_relbase = record.get("partner_source_relbase", "").strip()
    partner_json = (source_root / partner_relbase).with_suffix(".json") if partner_relbase else None
    own_vad = load_vad(base.with_suffix(".json"))
    partner_vad = load_vad(partner_json)
    strip_background = audio_strip_background(
        canvas_width, settings.audio_strip_height, lower, upper, own_vad, partner_vad,
        clip_start_s, settings.duration_s, sample_rate,
    )

    silent_path = output_root / f".{stem}.silent.tmp.mp4"
    muxed_path = output_root / f".{stem}.muxed.tmp.mp4"
    for path in (silent_path, muxed_path):
        if path.exists():
            path.unlink()
    output_height = settings.output_height + (0 if filmstrip is None else filmstrip.shape[0])
    output_height += output_height % 2
    if filmstrip is not None and output_height > settings.output_height + filmstrip.shape[0]:
        # Keep the band and the panels exactly stacked when the parity fix adds a
        # row, by padding the band rather than stretching a panel.
        filmstrip = np.concatenate(
            (filmstrip, np.zeros((1, canvas_width, 3), dtype=np.uint8)), axis=0
        )
    keyframe_interval = max(1, int(round(annotation_fps * settings.keyframe_seconds)))
    encoder_command = [
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
        "-f", "rawvideo", "-pixel_format", "bgr24",
        "-video_size", f"{canvas_width}x{output_height}",
        "-framerate", f"{Fraction(annotation_fps).limit_denominator(100000)}",
        "-i", "-", "-an", "-c:v", "libx264", "-preset", settings.preset,
        # A keyframe every second. libx264's default GOP of 250 put three of them
        # in a thirty-second clip, so dragging the scrubber could only ever land
        # on 0, 10 or 20 seconds. See docs on why the served range support
        # matters too -- both are needed for the bar to be draggable.
        "-g", str(keyframe_interval), "-keyint_min", str(keyframe_interval),
        "-sc_threshold", "0",
        "-crf", str(settings.crf), "-pix_fmt", "yuv420p", str(silent_path),
    ]
    capture = cv2.VideoCapture(str(base.with_suffix(".mp4")))
    if not capture.isOpened():
        raise RuntimeError(f"OpenCV could not open {base.with_suffix('.mp4')}")
    frames = _frames_at_annotation_times(capture, start_frame, render_frames, timebase)
    encoder = subprocess.Popen(encoder_command, stdin=subprocess.PIPE)
    written = 0
    try:
        assert encoder.stdin is not None
        for relative_index, frame in enumerate(frames):
            frame = repair_frame(frame, repair) if not repair.is_identity else frame
            source_index = start_frame + relative_index
            current_smplh_valid = _mask_at(smplh_valid, source_index)
            current_box_valid = _mask_at(box_valid, source_index)
            current_movement_valid = _mask_at(movement_valid, source_index)
            panel_a = draw_released_overlay(
                frame, keypoints[source_index], boxes[source_index],
                current_smplh_valid, current_box_valid, current_movement_valid,
                draw_indicators=False,
                display_scale=display_scale,
            )
            projected = None if joints.projected_xy_px is None else joints.projected_xy_px[relative_index]
            panel_b = draw_projected_joints(
                frame, projected, joints.edges, joints.groups, joints.status,
                display_scale=display_scale,
            )
            root_relative = None if joints.root_relative_xyz_m is None else joints.root_relative_xyz_m[relative_index]
            panel_c = draw_root_relative_panel(
                skeleton_width, visual_height, root_relative, joints.edges, joints.groups,
                joints.status, settings.skeleton_half_extent_m,
                upright_side_view=settings.upright_side_view,
            )
            panel_a = _fit_panel(panel_a, video_panel_width, visual_height)
            panel_b = _fit_panel(panel_b, video_panel_width, visual_height)
            if projected is None:
                _centered_message(panel_b, ("Projection unavailable", joints.status))
            _panel_label(panel_a, "A | 2D + masks")
            _panel_label(panel_b, "B | SMPL-H")
            _panel_label(panel_c, "C | 3D")
            draw_mask_dots(
                panel_a, current_smplh_valid, current_box_valid,
                current_movement_valid,
            )
            visual_row = np.concatenate((panel_a, panel_b, panel_c), axis=1)
            progress = relative_index / max(1, render_frames - 1)
            stacked = [visual_row, add_playhead(strip_background, progress)]
            if filmstrip is not None:
                stacked.insert(0, filmstrip)
            output_frame = np.concatenate(stacked, axis=0)
            encoder.stdin.write(output_frame.tobytes())
            written += 1
    finally:
        capture.release()
        if encoder.stdin is not None:
            encoder.stdin.close()
        encoder_status = encoder.wait()
    if encoder_status != 0 or written != render_frames:
        raise RuntimeError(f"video encoder failed: status={encoder_status}, frames={written}/{render_frames}")
    os.chmod(silent_path, 0o600)

    actual_duration_s = render_frames / annotation_fps
    if audio_status == "released":
        audio_input = ["-ss", f"{clip_start_s:.9f}", "-t", f"{actual_duration_s:.9f}", "-i", str(wav_path)]
    else:
        audio_input = ["-f", "lavfi", "-t", f"{actual_duration_s:.9f}",
                       "-i", f"anullsrc=r={sample_rate}:cl=mono"]
    mux_command = [
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
        "-i", str(silent_path), *audio_input, "-map", "0:v:0", "-map", "1:a:0",
        "-c:v", "copy", "-c:a", "aac", "-b:a", "128k", "-ac", "1",
        # `apad` plus an output `-t` instead of `-shortest`: a WAV that runs out
        # early must not decide how much video the reviewer gets to see.
        "-af", "apad", "-t", f"{actual_duration_s:.9f}",
        "-movflags", "+faststart", str(muxed_path),
    ]
    try:
        subprocess.run(mux_command, check=True)
        os.chmod(muxed_path, 0o600)
        os.replace(muxed_path, final_path)
        os.chmod(final_path, 0o600)
    finally:
        if silent_path.exists():
            silent_path.unlink()
        if muxed_path.exists():
            muxed_path.unlink()

    result = {
        "render_fingerprint": fingerprint,
        "status": "rendered",
        "media_file": final_path.name,
        "start_frame": start_frame,
        "end_frame": end_frame,
        "start_s": clip_start_s,
        "duration_s": actual_duration_s,
        "avg_frame_rate": f"{probe.fps.numerator}/{probe.fps.denominator}",
        "annotation_fps": annotation_fps,
        "native_raster": f"{probe.width}x{probe.height}",
        "output_raster": f"{canvas_width}x{output_height}",
        "filmstrip_thumbnails": settings.filmstrip_thumbnails,
        "rendered_frames": written,
        "audio_source": str(wav_path.relative_to(source_root)),
        "audio_status": audio_status,
        "audio_sample_rate": sample_rate,
        "raster_repair": (
            "none" if repair.is_identity
            else " + ".join(
                part for part in (
                    f"cropped {sum(repair.pillarbox)} black columns" if repair.is_cropped else "",
                    f"pixel aspect {repair.sample_aspect:.4f}" if repair.is_anamorphic else "",
                    f"{repair.quarter_turns * 90} deg turn" if repair.is_rotated else "",
                ) if part
            )
        ),
        "repaired_raster": f"{repaired_width}x{repaired_height}",
        "annotation_timebase": "consistent" if timebase.is_consistent
                               else f"drifts {timebase.index_drift_s:+.2f} s by the last frame",
        "partner_vad_status": "available" if partner_json and partner_json.exists() else "unavailable",
        "joint_provider": provider.name,
        "joint_status": joints.status,
    }
    _write_private_json(sidecar_path, result)
    return {**record, **result}

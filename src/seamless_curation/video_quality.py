"""What one decoded frame per clip says about the picture: blur, exposure, clutter, camera motion.

Everything else in the annotation reads the released arrays; this is the only
pass that looks at pixels, and it runs as its own stage (``scan-video``) because
it has to open every MP4. Until it has run, every column below is NA and
``visual_status`` is ``'not_run'`` (:func:`empty_measures` builds that row).

Which frame
-----------
One frame per clip, the **keyframe at or before the clip midpoint**. Seeking to a
keyframe costs one GOP lookup and one intra decode, never a run of predicted
frames. Measured on one file per vendor x raster cell (12 cells, read-only):
every stream is H.264 with a fixed 250-frame GOP -- 8.33 s at 30 fps, 8.34 s at
29.97, 5.5 s on the ~45 fps V01 2160x2160 files -- so for any clip longer than
two GOPs the keyframe before the midpoint is inside the clip. Only tail clips
shorter than that can land on a keyframe before ``start_s``; for those we decode
forward to the first frame at or after ``start_s``, capped at
:data:`FORWARD_DECODE_LIMIT` = 300 frames (one GOP is 250), and report
``'no_frame'`` if the clip holds no frame (the window lies past the end of the
stored video, which happens on files whose annotation outlasts the container;
see :class:`media_repair.AnnotationTimebase`). A forward-decoded frame is a P/B
frame, not an I frame; on the same 12 files the Laplacian variance of the 12
frames after a keyframe stays within 5% of the keyframe's own, so the frame type
does not bias sharpness.

Times are on the annotation clock: ``frame_time_s`` is the frame's presentation
time minus the stream's start time, so the first stored frame is 0 s, the time
of released array index 0. Two sampled cells (V01 1080x1920 at 29.98 fps, V03
2180x3840) start their stream at 0.033 s, one frame; the rest start at 0.

Pixels
------
``frame.to_ndarray(format='gray')`` returns full-range 8-bit luma: on a 10-bit
limited-range V00 frame the stored Y spans 78-850 and the gray image 4-229,
matching ``(Y - 64) / 876 * 255`` to 0.33 levels, and 8-bit limited-range frames
are expanded the same way. So the thresholds below are on a 0-255 display scale
regardless of bit depth. The gray frame is then made upright exactly as the
review renderer does (:func:`media_repair.repair_frame`: pillarbox crop,
pixel-aspect squeeze, quarter turns) and resized so its short side is
``short_side`` (540 px; INTER_AREA when shrinking, INTER_LINEAR for the 640x480
V03 raster, the only one that is upscaled). Every measure is taken on that
image, so a 4K and a 1080p file are compared at the same scale. The subject box
is the released stored-raster box at the frame's time (``box_at``), mapped
through :func:`media_repair.repair_box` and the same scale, and clipped to the
frame.

Measures
--------
``frame_sharpness``
    Variance of the 3x3 Laplacian over the pixels inside the subject box
    (computed on the whole frame, then restricted, so the box border is not a
    filter edge). High = sharp detail on the person, low = defocus or motion
    blur. Content-dependent (a patterned shirt scores higher than a plain one),
    so it ranks frames of a similar kind rather than being an absolute focus
    measure, and it differs by vendor and raster (per-file medians on one file
    per cell: 136 V01 1080x1920, 252 V00, 291 V02, 792 V03 2160x3840, 820 V03
    1080x1920), so compare it within a raster. NA when there is no box at
    the frame time or the mapped box covers fewer than :data:`MIN_BOX_PIXELS`
    pixels of the frame.
``frame_luma_mean``
    Mean luma of the whole upright frame (padding excluded): exposure.
``frame_luma_clipped_frac``
    Share of pixels at <= 5 or >= 250: crushed shadows or blown highlights.
``background_edge_density``
    Share of ``Canny(100, 200)`` edge pixels outside the subject box (grown by
    :data:`BOX_MARGIN_PX` so the person's own outline is not counted): how
    cluttered the background is. NA when there is no box at the frame time (we
    cannot tell subject from background). A valid box lying wholly outside the
    frame leaves the whole frame as background.
``camera_shift_px``
    Magnitude of ``cv2.phaseCorrelate`` between this clip's frame and the
    previous measured clip's frame of the same file, in px at 540 px short side,
    after filling the union of both frames' subject boxes (grown by the same
    margin) with each frame's mean, so the person's own movement is not read as
    the camera's, and applying a Hann window (without it the wrap-around at the
    image border dominates: on a synthetic 8x3 px shift the unwindowed estimate
    was 219 px off, windowed 0.3 px). Real frames translated by 3, 8.1 and
    22.4 px come back as 2.8-3.1, 7.5-8.1 and 22.3-22.4 px on 10 vendor x raster
    cells, plain V00/V02 backdrops included. Consecutive clips of the 12 probe
    files, all fixed cameras, measure 0.01-0.67 px. NA for the first measured
    clip of a file, when the two
    frames differ in size, or when either frame's background is featureless
    (standard deviation below :data:`MIN_BACKGROUND_STD` levels, where phase
    correlation returns an arbitrary peak).

Failure handling: the function never raises. A file that cannot be opened or
has no video stream, or any error while seeking/decoding, marks that window and
every later one ``'decode_error'`` (the demuxer state after a failed read is not
trustworthy); windows before it keep their measurements.

Throughput, one file per vendor x raster, whole files, pinned to one core (the
same to within 10% with PyAV's default 16 slice threads): 0.010 s/clip (V03
640x480), 0.017 (1080x960), 0.025-0.056 (1080x1920 of every vendor, 1920x1080),
0.074 (3840x2160), 0.10 (2180x3840), 0.12 (2160x3840), 0.14 (V01 2160x2160,
10-bit). Weighted by the corpus's clip counts per cell that is about 21.5
CPU-hours for the 1.03M clips, dominated by the 357k V03 2160x3840 clips.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np

from .media_repair import RasterRepair, repair_box, repair_frame

#: Registry columns this module fills, in table order.
VISUAL_COLUMNS: tuple[str, ...] = (
    "visual_status",
    "frame_time_s",
    "frame_sharpness",
    "frame_luma_mean",
    "frame_luma_clipped_frac",
    "background_edge_density",
    "camera_shift_px",
)
VISUAL_STATUSES: tuple[str, ...] = ("measured", "not_run", "no_frame", "decode_error")

DEFAULT_SHORT_SIDE = 540
#: Frames decoded past the seek point looking for one inside the clip. Every
#: probed stream has a 250-frame GOP, so this always reaches the next keyframe.
FORWARD_DECODE_LIMIT = 300
#: Luma at or beyond these (0-255 full range) counts as clipped.
LUMA_CLIPPED_LOW = 5
LUMA_CLIPPED_HIGH = 250
CANNY_LOW = 100
CANNY_HIGH = 200
#: Fewer box pixels than this (8x8 at 540 px short side) is no sharpness sample.
MIN_BOX_PIXELS = 64
#: A background flatter than this (gray-level SD) gives phase correlation no peak.
MIN_BACKGROUND_STD = 1.0
#: Border added around the subject box (px at 540 px short side) before anything
#: is counted as background. Canny puts a silhouette edge within one pixel of the
#: step, so wherever the person touches a tight detector box (head top, feet,
#: hands) the outline would otherwise leak into the background count: on a
#: textured patch in a flat frame the leak alone was 0.0009, the same size as the
#: 0.001-0.002 densities of the plain V00/V02 backdrops.
BOX_MARGIN_PX = 4
#: Slack when comparing a frame's time with a clip edge; far below one frame.
TIME_TOLERANCE_S = 1e-3


@dataclass(frozen=True)
class ClipWindow:
    """One clip's span on the annotation clock (``start_s`` inclusive, ``end_s`` exclusive)."""

    clip_index: int
    start_s: float
    end_s: float

    @property
    def mid_s(self) -> float:
        return 0.5 * (self.start_s + self.end_s)


@dataclass
class UprightFrame:
    """A sampled frame after repair and scaling, with its subject box.

    ``box`` is integer ``(x1, y1, x2, y2)`` in this image, clipped to it (possibly
    empty when the released box lies off-frame); ``None`` when there was no box.
    """

    time_s: float
    image: np.ndarray
    box: tuple[int, int, int, int] | None


def empty_measures(status: str = "not_run") -> dict:
    """The row for a clip the pixel pass did not measure."""
    if status not in VISUAL_STATUSES or status == "measured":
        raise ValueError(f"not an unmeasured visual status: {status!r}")
    row: dict = {name: float("nan") for name in VISUAL_COLUMNS}
    row["visual_status"] = status
    return row


def upright_gray(
    gray: np.ndarray, repair: RasterRepair, short_side: int = DEFAULT_SHORT_SIDE
) -> tuple[np.ndarray, float, float]:
    """Repair a stored-raster gray frame and scale it to ``short_side``.

    Returns the image and the x/y factors from display-raster (repaired, unscaled)
    coordinates to it.
    """
    import cv2

    if gray.ndim != 2:
        raise ValueError("expected a single-channel frame")
    if gray.shape != (repair.height, repair.width):
        raise ValueError(
            f"decoded frame is {gray.shape[1]}x{gray.shape[0]}, "
            f"repair describes {repair.width}x{repair.height}"
        )
    image = repair_frame(gray, repair)
    height, width = image.shape
    scale = short_side / min(height, width)
    out_w = max(1, int(round(width * scale)))
    out_h = max(1, int(round(height * scale)))
    if (out_w, out_h) != (width, height):
        interpolation = cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR
        image = cv2.resize(image, (out_w, out_h), interpolation=interpolation)
    return np.ascontiguousarray(image), out_w / width, out_h / height


def map_box(
    box_xyxy: np.ndarray | None,
    repair: RasterRepair,
    scale_xy: tuple[float, float],
    shape: tuple[int, int],
) -> tuple[int, int, int, int] | None:
    """Stored-raster xyxy box -> integer box in the upright scaled image, clipped.

    ``None`` in (or a non-finite box) is ``None`` out. A box lying off the frame
    comes back empty (``x2 <= x1`` or ``y2 <= y1``), not ``None``.
    """
    if box_xyxy is None:
        return None
    box = np.asarray(box_xyxy, dtype=np.float64).reshape(-1)
    if box.shape != (4,) or not np.isfinite(box).all():
        return None
    moved = repair_box(box, repair)
    height, width = shape
    sx, sy = scale_xy
    x1 = int(np.clip(math.floor(moved[0] * sx), 0, width))
    y1 = int(np.clip(math.floor(moved[1] * sy), 0, height))
    x2 = int(np.clip(math.ceil(moved[2] * sx), 0, width))
    y2 = int(np.clip(math.ceil(moved[3] * sy), 0, height))
    return x1, y1, max(x1, x2), max(y1, y2)


def _box_pixels(box: tuple[int, int, int, int] | None) -> int:
    if box is None:
        return 0
    return max(0, box[2] - box[0]) * max(0, box[3] - box[1])


def _grown(
    box: tuple[int, int, int, int], shape: tuple[int, int], margin: int = BOX_MARGIN_PX
) -> tuple[int, int, int, int]:
    """The box plus a margin, clipped to the frame; an empty box stays empty."""
    if not _box_pixels(box):
        return box
    height, width = shape
    x1, y1, x2, y2 = box
    return max(0, x1 - margin), max(0, y1 - margin), min(width, x2 + margin), min(height, y2 + margin)


def frame_measures(image: np.ndarray, box: tuple[int, int, int, int] | None) -> dict:
    """Sharpness, exposure and background clutter of one upright 8-bit gray frame."""
    import cv2

    image = np.ascontiguousarray(image, dtype=np.uint8)
    total = image.size
    out = {
        "frame_luma_mean": float(image.mean()),
        "frame_luma_clipped_frac": float(
            np.count_nonzero((image <= LUMA_CLIPPED_LOW) | (image >= LUMA_CLIPPED_HIGH)) / total
        ),
        "frame_sharpness": float("nan"),
        "background_edge_density": float("nan"),
    }
    if box is None:
        return out
    x1, y1, x2, y2 = box
    inside = _box_pixels(box)
    if inside >= MIN_BOX_PIXELS:
        laplacian = cv2.Laplacian(image, cv2.CV_64F)
        out["frame_sharpness"] = float(laplacian[y1:y2, x1:x2].var())
    gx1, gy1, gx2, gy2 = grown = _grown(box, image.shape)
    excluded = _box_pixels(grown)
    outside = total - excluded
    if outside > 0:
        edges = cv2.Canny(image, CANNY_LOW, CANNY_HIGH) > 0
        edge_inside = int(np.count_nonzero(edges[gy1:gy2, gx1:gx2])) if excluded else 0
        out["background_edge_density"] = float(
            (int(np.count_nonzero(edges)) - edge_inside) / outside
        )
    return out


def background_shift(previous: UprightFrame, current: UprightFrame) -> float:
    """Phase-correlation shift (px) of the background between two frames of one file."""
    import cv2

    if previous.image.shape != current.image.shape:
        return float("nan")
    height, width = current.image.shape
    masked = np.zeros((height, width), dtype=bool)
    for box in (previous.box, current.box):
        if _box_pixels(box):
            x1, y1, x2, y2 = _grown(box, (height, width))
            masked[y1:y2, x1:x2] = True
    background = ~masked
    if not background.any():
        return float("nan")
    planes = []
    for frame in (previous, current):
        plane = frame.image.astype(np.float32)
        if float(plane[background].std()) < MIN_BACKGROUND_STD:
            return float("nan")
        plane[masked] = float(plane.mean())
        planes.append(plane)
    window = cv2.createHanningWindow((width, height), cv2.CV_32F)
    (dx, dy), _response = cv2.phaseCorrelate(planes[0], planes[1], window)
    shift = float(math.hypot(dx, dy))
    return shift if math.isfinite(shift) else float("nan")


def _seek_frame(container, stream, window: ClipWindow, stream_start_s: float):
    """The (annotation time, frame) sampled for a window, or None when it holds none."""
    time_base = float(stream.time_base)
    target = window.mid_s + stream_start_s
    offset = int(math.floor(max(target, stream_start_s) / time_base))
    container.seek(offset, stream=stream, backward=True, any_frame=False)
    decoded = 0
    for frame in container.decode(stream):
        decoded += 1
        if frame.time is not None:
            time_s = float(frame.time) - stream_start_s
            if time_s >= window.end_s - TIME_TOLERANCE_S:
                return None
            if time_s >= window.start_s - TIME_TOLERANCE_S:
                return time_s, frame
        if decoded > FORWARD_DECODE_LIMIT:
            return None
    return None


def measure_video(
    mp4_path: Path,
    windows: list[ClipWindow],
    *,
    repair: RasterRepair,
    box_at: Callable[[float], np.ndarray | None],
    short_side: int = DEFAULT_SHORT_SIDE,
) -> list[dict]:
    """Measure one frame per clip window of one MP4.

    Returns one dict per window, in the order given, holding ``clip_index`` and
    the :data:`VISUAL_COLUMNS`. ``box_at(t)`` gives the released stored-raster
    xyxy subject box at annotation time ``t`` seconds, or ``None``. The camera
    shift compares consecutive windows in time order, whatever order they were
    passed in. Never raises.
    """
    rows: list[dict | None] = [None] * len(windows)
    order = sorted(range(len(windows)), key=lambda i: (windows[i].start_s, windows[i].clip_index))

    def fail_from(position: int) -> None:
        for i in order[position:]:
            rows[i] = empty_measures("decode_error")

    container = None
    try:
        import av

        container = av.open(str(mp4_path))
        if not container.streams.video:
            raise ValueError("no video stream")
        stream = container.streams.video[0]
        stream_start_s = (
            float(stream.start_time * stream.time_base) if stream.start_time is not None else 0.0
        )
    except Exception:
        fail_from(0)
        _close_quietly(container)
        return [_with_index(row, window) for row, window in zip(rows, windows)]

    previous: UprightFrame | None = None
    try:
        for position, i in enumerate(order):
            window = windows[i]
            try:
                sampled = _seek_frame(container, stream, window, stream_start_s)
                if sampled is None:
                    rows[i] = empty_measures("no_frame")
                    continue
                time_s, frame = sampled
                image, sx, sy = upright_gray(frame.to_ndarray(format="gray"), repair, short_side)
                box = map_box(box_at(time_s), repair, (sx, sy), image.shape)
                current = UprightFrame(time_s=time_s, image=image, box=box)
                row = {"visual_status": "measured", "frame_time_s": time_s}
                row.update(frame_measures(image, box))
                row["camera_shift_px"] = (
                    background_shift(previous, current) if previous is not None else float("nan")
                )
                rows[i] = {name: row[name] for name in VISUAL_COLUMNS}
                previous = current
            except Exception:
                fail_from(position)
                break
    finally:
        _close_quietly(container)
    return [_with_index(row, window) for row, window in zip(rows, windows)]


def _close_quietly(container) -> None:
    if container is None:
        return
    try:
        container.close()
    except Exception:
        pass


def _with_index(row: dict | None, window: ClipWindow) -> dict:
    out = {"clip_index": int(window.clip_index)}
    out.update(row if row is not None else empty_measures("decode_error"))
    return out

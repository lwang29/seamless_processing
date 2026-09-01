"""What has to be fixed about a released bundle before it can be looked at.

Three defects turned up in the V01/V02/V03 review, all of them properties of how
the media was stored rather than of what was recorded:

**Anamorphic rasters.** Every V01 file declares a 9:16 display aspect, but only
1080x1920 stores square pixels. 2160x2160 carries a 9:16 pixel aspect and
1920x1080 carries 81:256, so both are stored horizontally stretched and are meant
to be squeezed on playback. Nothing in the pipeline applied the pixel aspect, so
the reviewer saw wide people and FM1 read them as seated.

**Quarter-turned video.** 1,336 V03 files are stored 3840x2160 and 60 are stored
640x480, with the participant lying sideways in the frame. No rotation tag: the
turn is baked into the pixels.

**A drifting annotation timebase.** The released arrays are sampled on a uniform
grid at the container's *nominal* rate, but some files hold fewer stored frames
than that rate implies over their duration. Pairing the two by integer index then
walks off progressively -- 9.5 s by the end of `V01_S1607_I00000135_P2569`.

The first two are image-plane transforms and this module implements both, for
frames, for released 2D points, and for camera-frame 3D joints. They are exact:
the HMR camera is a centred pinhole with square pixels, so rotating the image a
quarter turn about the principal point is precisely a rotation of the camera
about its optical axis.

**What the repairs cannot fix is the released SMPL-H on an anamorphic file.** An
anisotropic scale is not a rigid transform and the fitted camera is isotropic, so
the only way the optimiser could match a stretched 2D person was to distort the
3D pose. Measured, on 60 files per raster: aligning the projected body to the
released keypoints by translation and one isotropic scale leaves a median shape
residual of 0.113 against the raster as stored and 0.218 once the pixel aspect is
applied, against 0.048-0.074 for square-pixel rasters. The pose is bound to the
stretched image. Un-stretching the video and the keypoints is correct and worth
doing; it does not recover the annotation.
"""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path

import numpy as np

# A pixel aspect this close to square is treated as square. 1012x1920 carries
# 270:253, a 6.7% correction; that is above this floor and so is honoured.
SQUARE_PIXEL_TOLERANCE = 0.01
# How far from a quarter turn the measured body roll may sit and still count.
# The rotated V03 files land within a degree or two of exactly +/-90.
QUARTER_TURN_TOLERANCE_DEG = 30.0

COCO_SHOULDERS = (5, 6)
COCO_HIPS = (11, 12)


@dataclass(frozen=True)
class RasterRepair:
    """How the stored raster differs from the one worth looking at.

    Applied in a fixed order -- crop, then pixel-aspect squeeze, then rotation --
    and every point transform follows the same order. ``quarter_turns`` counts
    90-degree **counter-clockwise** rotations.

    ``pillarbox`` is ``(left, right)`` columns of padding to discard. It is
    cosmetic: 1080x960 is a 540x960 frame with 270 black columns each side and
    2180x3840 is 2160x3840 with 20 on the right, and both fit *as well as any
    ordinary raster* (shape residual 0.057 and 0.061 against a 0.048-0.074 normal
    band), so the padding costs the annotation nothing. Cropping makes the review
    panel bigger and the aspect consistent; it does not change a measurement.
    """

    width: int
    height: int
    sample_aspect: float = 1.0
    quarter_turns: int = 0
    pillarbox: tuple[int, int] = (0, 0)

    def __post_init__(self) -> None:
        if self.width <= 0 or self.height <= 0:
            raise ValueError("raster dimensions must be positive")
        if not np.isfinite(self.sample_aspect) or self.sample_aspect <= 0:
            raise ValueError("sample_aspect must be a positive, finite ratio")
        if self.quarter_turns not in (0, 1, 2, 3):
            raise ValueError("quarter_turns must be 0, 1, 2 or 3")
        left, right = self.pillarbox
        if left < 0 or right < 0 or left + right >= self.width:
            raise ValueError("pillarbox must leave at least one column of content")

    @property
    def is_anamorphic(self) -> bool:
        return abs(self.sample_aspect - 1.0) > SQUARE_PIXEL_TOLERANCE

    @property
    def is_rotated(self) -> bool:
        return self.quarter_turns != 0

    @property
    def is_cropped(self) -> bool:
        return any(self.pillarbox)

    @property
    def is_identity(self) -> bool:
        return not self.is_anamorphic and not self.is_rotated and not self.is_cropped

    @property
    def cropped_size(self) -> tuple[int, int]:
        """Size after the pillarbox crop, before anything else."""
        return self.width - sum(self.pillarbox), self.height

    @property
    def squeezed_size(self) -> tuple[int, int]:
        """Size after the crop and the pixel aspect, before any rotation."""
        width, height = self.cropped_size
        if not self.is_anamorphic:
            return width, height
        return max(1, int(round(width * self.sample_aspect))), height

    @property
    def display_size(self) -> tuple[int, int]:
        width, height = self.squeezed_size
        return (height, width) if self.quarter_turns % 2 else (width, height)


def probe_sample_aspect(video_path: Path) -> float:
    """The pixel aspect the container asks a player to apply.

    An absent or unparseable tag means square pixels, which is what every vendor
    other than V01 stores.
    """
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=sample_aspect_ratio", "-of", "csv=p=0:nk=1",
         str(video_path)],
        check=False, capture_output=True, text=True,
    )
    text = result.stdout.strip().replace(":", "/")
    try:
        ratio = float(Fraction(text))
    except (ValueError, ZeroDivisionError):
        return 1.0
    return ratio if ratio > 0 else 1.0


def body_roll_deg(keypoints: np.ndarray, usable: np.ndarray | None = None) -> float:
    """In-image angle of the shoulder-to-hip axis, 0 when the body runs downward.

    Uses the released 2D keypoints rather than anything fitted, so it costs one
    array read and cannot inherit a pose-estimation error.
    """
    points = np.asarray(keypoints, dtype=np.float64)
    if points.ndim != 3 or len(points) == 0 or points.shape[1] <= max(COCO_HIPS):
        return float("nan")
    if usable is None:
        usable = np.ones(points.shape[:2], dtype=bool)
    ok = (usable[:, COCO_SHOULDERS[0]] & usable[:, COCO_SHOULDERS[1]]
          & usable[:, COCO_HIPS[0]] & usable[:, COCO_HIPS[1]])
    if not ok.any():
        return float("nan")
    shoulder = 0.5 * (points[:, COCO_SHOULDERS[0], :2] + points[:, COCO_SHOULDERS[1], :2])
    hip = 0.5 * (points[:, COCO_HIPS[0], :2] + points[:, COCO_HIPS[1], :2])
    axis = (hip - shoulder)[ok]
    return float(np.median(np.degrees(np.arctan2(axis[:, 0], axis[:, 1]))))


def quarter_turns_from_roll(roll_deg: float) -> int:
    """Counter-clockwise quarter turns that stand the measured body upright.

    A body running left across the frame (feet toward the left edge, roll near
    -90) needs the right edge brought to the top, which is one turn.
    """
    if not np.isfinite(roll_deg):
        return 0
    for turns, centre in ((1, -90.0), (3, 90.0), (2, 180.0), (2, -180.0)):
        if abs(roll_deg - centre) <= QUARTER_TURN_TOLERANCE_DEG:
            return turns
    return 0


def repair_points(points_xy: np.ndarray, repair: RasterRepair) -> np.ndarray:
    """Map released image points into the repaired raster.

    Shape is preserved and only the last axis is touched, so this works equally
    on ``(F, J, 2)`` keypoints, ``(F, J, 3)`` keypoints-with-score, and a single
    ``(2,)`` point.
    """
    output = np.array(points_xy, dtype=np.float64, copy=True)
    if output.shape[-1] < 2:
        raise ValueError("points must carry at least x and y on the last axis")
    if repair.is_identity:
        return output
    output[..., 0] -= repair.pillarbox[0]
    output[..., 0] *= repair.sample_aspect
    width, height = repair.squeezed_size
    for _ in range(repair.quarter_turns):
        x = output[..., 0].copy()
        y = output[..., 1].copy()
        output[..., 0] = y
        output[..., 1] = (width - 1) - x
        width, height = height, width
    return output


def repair_box(box_xyxy: np.ndarray, repair: RasterRepair) -> np.ndarray:
    """Map an ``(..., 4)`` xyxy box, re-sorting the corners after rotation."""
    box = np.asarray(box_xyxy, dtype=np.float64)
    if box.shape[-1] != 4:
        raise ValueError("expected an (..., 4) xyxy box")
    corners = np.stack(
        (box[..., 0:2], box[..., 2:4]), axis=-2
    )  # (..., 2, 2)
    moved = repair_points(corners, repair)
    lower = moved.min(axis=-2)
    upper = moved.max(axis=-2)
    return np.concatenate((lower, upper), axis=-1)


def repair_camera_joints(joints_xyz: np.ndarray, repair: RasterRepair) -> np.ndarray:
    """Rotate camera-frame 3D joints to match the repaired image.

    Only the rotation applies. The pixel-aspect squeeze has no 3D counterpart --
    that is exactly why an anamorphic file's released pose cannot be repaired --
    so it is deliberately not approximated here.
    """
    joints = np.array(joints_xyz, dtype=np.float64, copy=True)
    if joints.shape[-1] != 3:
        raise ValueError("expected (..., 3) camera-frame joints")
    for _ in range(repair.quarter_turns % 4):
        x = joints[..., 0].copy()
        joints[..., 0] = joints[..., 1]
        joints[..., 1] = -x
    return joints


def repair_frame(frame: np.ndarray, repair: RasterRepair) -> np.ndarray:
    """Squeeze then rotate one decoded BGR frame."""
    import cv2

    image = frame
    if repair.is_cropped:
        left, right = repair.pillarbox
        image = image[:, left:image.shape[1] - right]
    if repair.is_anamorphic:
        width, height = repair.squeezed_size
        # INTER_AREA when shrinking, which is every V01 case except 1012x1920;
        # it is the only interpolation that does not alias a 1.8x downscale.
        shrinking = width < image.shape[1]
        image = cv2.resize(image, (width, height),
                           interpolation=cv2.INTER_AREA if shrinking else cv2.INTER_LINEAR)
    for _ in range(repair.quarter_turns % 4):
        image = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return image


def ffmpeg_repair_filter(repair: RasterRepair) -> str:
    """The same transform as an ffmpeg filter chain, or an empty string."""
    stages: list[str] = []
    if repair.is_cropped:
        width, height = repair.cropped_size
        stages.append(f"crop={width}:{height}:{repair.pillarbox[0]}:0")
    if repair.is_anamorphic:
        width, height = repair.squeezed_size
        stages.append(f"scale={width}:{height}")
    stages.extend(["transpose=2"] * (repair.quarter_turns % 4))
    if stages:
        stages.append("setsar=1")
    return ",".join(stages)


@dataclass(frozen=True)
class AnnotationTimebase:
    """How released per-frame arrays line up with the stored video frames.

    The arrays are uniform at ``nominal_fps``; the container may not be. Where
    they disagree, ``index_drift_s`` is how far apart a pairing by integer index
    ends up by the last frame -- the visible symptom is the keypoint overlay
    sliding off the person as a clip plays.
    """

    nominal_fps: float
    annotation_frames: int
    container_frames: int

    @property
    def index_drift_s(self) -> float:
        if self.nominal_fps <= 0:
            return float("nan")
        return (self.annotation_frames - self.container_frames) / self.nominal_fps

    @property
    def is_consistent(self) -> bool:
        """Within one frame. Anything larger drifts visibly over 30 seconds."""
        return abs(self.annotation_frames - self.container_frames) <= 1

    def index_at(self, seconds: float) -> int:
        """Annotation index for a decoded frame shown at ``seconds``."""
        if not np.isfinite(seconds) or self.nominal_fps <= 0:
            return 0
        return int(min(max(round(seconds * self.nominal_fps), 0),
                       max(self.annotation_frames - 1, 0)))


# Rasters measured to be a portrait frame padded with black columns. Keyed by
# stored size because the padding is a property of the vendor's export, not of
# the recording: every sampled file of each raster showed the same bars to within
# two pixels. Measured with scripts/probe_raster_geometry.py --pillarbox.
KNOWN_PILLARBOX: dict[tuple[int, int], tuple[int, int]] = {
    (1080, 960): (270, 270),     # 540x960 content, 9:16, in a 9:8 raster
    (2180, 3840): (0, 20),       # 2160x3840 content with a bar on the right only
}


def pillarbox_for(width: int, height: int) -> tuple[int, int]:
    """Black columns to discard for a known padded raster, else none."""
    return KNOWN_PILLARBOX.get((int(width), int(height)), (0, 0))


def measure_pillarbox(
    frames: np.ndarray, *, brightness_floor: int = 8
) -> tuple[int, int]:
    """Columns of black at each edge, over the brightest value each column reaches.

    Taking the maximum over several frames rather than one guards against a
    genuinely dark frame -- a participant in dark clothing against a dark wall --
    being read as padding.
    """
    stack = np.asarray(frames)
    if stack.ndim == 4:
        stack = stack.max(axis=-1)
    if stack.ndim != 3 or stack.size == 0:
        return 0, 0
    profile = stack.reshape(-1, stack.shape[-1]).max(axis=0)
    lit = profile > brightness_floor
    if not lit.any():
        return 0, 0
    left = int(np.argmax(lit))
    right = int(np.argmax(lit[::-1]))
    return left, right

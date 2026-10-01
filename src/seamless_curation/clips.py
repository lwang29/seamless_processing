"""The clip grid: how a recording is cut into fixed-length clips.

A *recording* is one participant's files for one interaction (one file id). A
*clip* is a fixed-length, non-overlapping segment of a recording. The grid is
defined in **time**, not frames: clip ``k`` covers ``[k*L, (k+1)*L)`` seconds,
mapped to frames with the recording's own rate. Two consequences, both
deliberate:

* Partners line up by index. The two recordings of an interaction share the
  interaction's active-time origin, so clip ``k`` of one participant and clip
  ``k`` of the other cover the same wall-clock interval even when their frame
  rates differ (1,030+ pairs mix 30 and 29.97 fps, where frame-index alignment
  drifts). The old pipeline's frame-based hop drifted 0.01 s per hop at 29.97.
* Nothing is dropped. The final clip is kept however short it is and is marked
  ``is_partial``; every frame of the recording belongs to exactly one clip. The
  old sliding windows discarded the tail (157 h over the corpus).

Clips are defined on the released pose grid (the NPZ frame count), because that
is the grid every per-frame annotation lives on. A recording with no pose array
has no clips; its recording row says why.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

#: Default clip length. 30 s is the unit the previous pipeline measured and
#: reviewed in; it is long enough for an activity or posture pattern to show and
#: short enough to be one training example. It is a config value, not a claim.
DEFAULT_CLIP_SECONDS = 30.0


@dataclass(frozen=True)
class ClipBounds:
    clip_index: int
    start_frame: int
    end_frame: int  # half-open
    fps: float
    nominal_seconds: float

    @property
    def n_frames(self) -> int:
        return self.end_frame - self.start_frame

    @property
    def start_s(self) -> float:
        return self.start_frame / self.fps

    @property
    def end_s(self) -> float:
        return self.end_frame / self.fps

    @property
    def seconds(self) -> float:
        return self.n_frames / self.fps

    @property
    def is_partial(self) -> bool:
        # A full clip can be one frame short of L*fps at 29.97 fps; "partial"
        # means the recording ended inside the clip, not a rounding frame.
        return self.seconds < self.nominal_seconds - 1.5 / self.fps


def clip_grid(n_frames: int, fps: float, clip_seconds: float = DEFAULT_CLIP_SECONDS) -> list[ClipBounds]:
    """Tile ``[0, n_frames)`` into time-aligned clips; the tail is kept."""

    if n_frames <= 0:
        return []
    if not (fps > 0 and clip_seconds > 0):
        raise ValueError("fps and clip_seconds must be positive")
    bounds: list[ClipBounds] = []
    index = 0
    start = 0
    while start < n_frames:
        stop = min(int(round((index + 1) * clip_seconds * fps)), n_frames)
        if stop <= start:  # only reachable for absurd fps/length combinations
            stop = min(start + 1, n_frames)
        bounds.append(ClipBounds(index, start, stop, float(fps), float(clip_seconds)))
        index += 1
        start = stop
    return bounds


def clip_count(n_frames: int, fps: float, clip_seconds: float = DEFAULT_CLIP_SECONDS) -> int:
    return len(clip_grid(n_frames, fps, clip_seconds))


def frames_for(seconds: float, fps: float) -> int:
    """Frames spanning ``seconds`` at ``fps``, by one rule everywhere: ceiling.

    ``round()`` made a 0.25 s gap 8 frames at 30 fps but 7 at 29.97 fps
    (``round(7.4925)``), so the same parameter meant different things per vendor.
    The ceiling (with a tolerance for float error) gives 8 at both.
    """

    return max(0, int(math.ceil(float(seconds) * float(fps) - 1e-6)))

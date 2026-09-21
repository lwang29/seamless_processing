"""Read the accepted subset.

The manifest is a promise about the source tree — *these frames of this file* —
and nothing in this pipeline ever copies or modifies the release. This module is
the other half of that promise: it turns a manifest row into the arrays a
trainer wants, so that a downstream project never has to know how the release is
laid out on disk.

Everything a clip needs shares one path stem, ``<source_root>/<source_relbase>``:

===========  ===========================================================
``.npz``     SMPL-H pose, 2D keypoints, and the movement feature blocks
``.wav``     that participant's own audio, 48 kHz mono float32
``.mp4``     that participant's own video
``.json``    transcript, VAD segments, and other released annotation
===========  ===========================================================

The release stores one *participant* per stem, not one interaction, so the audio
and video here are already the single speaker the pose belongs to. There is no
channel to separate and no risk of training on the partner's voice.

Usage::

    from seamless_curation.dataset import load_manifest, iter_clips

    manifest = load_manifest("outputs/vibes_upper_body_v1/export/clips_verified.csv")
    for clip in iter_clips(manifest, "seamless_interaction"):
        clip.upper_body_pose   # (frames, 13, 3) axis-angle, the ViBES body input
        clip.left_hand_pose    # (frames, 15, 3)
        clip.audio             # (samples,) float32 at clip.sample_rate
        clip.speech            # ((start_s, end_s), ...) relative to the clip

Or from a shell, as a self-test::

    python -m seamless_curation.dataset outputs/.../clips_verified.csv
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

import numpy as np
import pandas as pd

from .smplh_kinematics import UPPER_BODY_POSE_INDEX

#: The four files that share a stem. Keys are what :func:`clip_paths` returns.
SOURCE_SUFFIXES: Mapping[str, str] = {
    "pose": ".npz",
    "audio": ".wav",
    "video": ".mp4",
    "annotation": ".json",
}

#: ``smplh:body_pose`` rows that make up the upper body: spine 1-3, neck, head,
#: both collars, shoulders, elbows and wrists. ``UPPER_BODY_POSE_INDEX`` is
#: already expressed in body-pose rows (the release drops the pelvis into
#: ``global_orient``, so row r is tree joint r+1), so it is used as-is. This is
#: the same indexing ``manifest.verify_manifest`` checks the shape of.
UPPER_BODY_ROWS: tuple[int, ...] = tuple(UPPER_BODY_POSE_INDEX)


@dataclass(frozen=True)
class Clip:
    """One accepted clip, loaded. Pose arrays are axis-angle, radians."""

    clip_id: str
    file_id: str
    source_relbase: str
    vendor: str
    split: str
    participant_key: str
    fps: float
    start_frame: int
    end_frame: int

    body_pose: np.ndarray            # (frames, 21, 3)
    left_hand_pose: np.ndarray       # (frames, 15, 3)
    right_hand_pose: np.ndarray      # (frames, 15, 3)
    global_orient: np.ndarray        # (frames, 3)
    translation: np.ndarray          # (frames, 3)
    smplh_valid: np.ndarray          # (frames,) bool

    speech: tuple[tuple[float, float], ...]
    audio: np.ndarray | None = None  # (samples,) float32
    sample_rate: int | None = None

    @property
    def frames(self) -> int:
        return int(self.end_frame - self.start_frame)

    @property
    def seconds(self) -> float:
        return self.frames / self.fps

    @property
    def upper_body_pose(self) -> np.ndarray:
        """(frames, 13, 3) — torso, neck, head, shoulders, elbows, wrists.

        This is the body block ViBES trains on; the hands come separately.
        """

        return self.body_pose[:, list(UPPER_BODY_ROWS)]

    def __repr__(self) -> str:  # pragma: no cover - convenience
        audio = "none" if self.audio is None else f"{len(self.audio)}@{self.sample_rate}"
        return (
            f"Clip({self.clip_id} {self.frames}f/{self.seconds:.1f}s "
            f"@{self.fps:g}fps speech={len(self.speech)} audio={audio})"
        )


#: Identifier columns that must stay strings. ``participant_id`` is the reason:
#: most are digits but some carry a suffix ("P0040A"), so pandas infers ``int64``
#: on one chunk and ``object`` on another and warns about mixed types — and a
#: participant key silently cast to a number no longer joins to the manifest.
ID_COLUMNS: tuple[str, ...] = (
    "file_id", "clip_id", "review_item_id", "session_id", "participant_id",
    "interaction_id", "source_relbase", "vendor", "label", "split",
)


def load_manifest(path: str | Path) -> pd.DataFrame:
    """Read a manifest CSV, keeping frame indices integral and ids textual."""

    frame = pd.read_csv(path, dtype={column: str for column in ID_COLUMNS})
    for column in ("start_frame", "end_frame"):
        if column in frame.columns:
            frame[column] = frame[column].astype(int)
    return frame


def clip_paths(row: Mapping[str, Any], source_root: str | Path) -> dict[str, Path]:
    """The four source files for a manifest row. Nothing is read."""

    base = Path(source_root) / str(row["source_relbase"])
    return {kind: base.with_suffix(suffix) for kind, suffix in SOURCE_SUFFIXES.items()}


def _speech_segments(
    annotation_path: Path, start_s: float, stop_s: float
) -> tuple[tuple[float, float], ...]:
    """Released VAD for this participant, clipped to the window and rebased to it."""

    try:
        annotation = json.loads(annotation_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return ()
    segments = []
    for entry in annotation.get("metadata:vad") or []:
        begin, end = float(entry["start"]), float(entry["end"])
        if end <= start_s or begin >= stop_s:
            continue
        segments.append((max(begin, start_s) - start_s, min(end, stop_s) - start_s))
    return tuple(segments)


def load_clip(
    row: Mapping[str, Any],
    source_root: str | Path,
    *,
    with_audio: bool = True,
    with_speech: bool = True,
) -> Clip:
    """Load the frames one manifest row names.

    Only the named slice is read: the NPZ is opened lazily and the WAV is seeked
    to, so loading a 30-second clip out of a four-minute recording costs a
    30-second read, not a four-minute one.
    """

    if hasattr(row, "_asdict"):          # a namedtuple from df.itertuples()
        row = row._asdict()
    paths = clip_paths(row, source_root)
    start, stop = int(row["start_frame"]), int(row["end_frame"])
    fps = float(row["fps"])

    with np.load(paths["pose"]) as archive:
        body = np.asarray(archive["smplh:body_pose"][start:stop], dtype=np.float32)
        left = np.asarray(archive["smplh:left_hand_pose"][start:stop], dtype=np.float32)
        right = np.asarray(archive["smplh:right_hand_pose"][start:stop], dtype=np.float32)
        orient = np.asarray(archive["smplh:global_orient"][start:stop], dtype=np.float32)
        translation = np.asarray(archive["smplh:translation"][start:stop], dtype=np.float32)
        valid = np.asarray(archive["smplh:is_valid"][start:stop], dtype=bool)

    if len(body) != stop - start:
        raise ValueError(
            f"{row.get('clip_id', row['file_id'])}: manifest promises {stop - start} frames, "
            f"the NPZ yielded {len(body)}. The source tree does not match the manifest."
        )

    start_s = start / fps
    stop_s = stop / fps

    audio: np.ndarray | None = None
    sample_rate: int | None = None
    if with_audio:
        import soundfile  # optional: only needed when audio is actually wanted

        info = soundfile.info(str(paths["audio"]))
        sample_rate = int(info.samplerate)
        begin = int(round(start_s * sample_rate))
        count = int(round((stop_s - start_s) * sample_rate))
        # A clip at the very end of a recording can ask for a few samples past
        # the last one; clamp rather than fail, and pad so the length is exact.
        begin = max(0, min(begin, info.frames))
        audio, _ = soundfile.read(
            str(paths["audio"]), start=begin, frames=min(count, info.frames - begin),
            dtype="float32", always_2d=False,
        )
        audio = np.atleast_1d(np.asarray(audio, dtype=np.float32))
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        if len(audio) < count:
            audio = np.pad(audio, (0, count - len(audio)))

    speech = _speech_segments(paths["annotation"], start_s, stop_s) if with_speech else ()

    return Clip(
        clip_id=str(row.get("clip_id", "")),
        file_id=str(row["file_id"]),
        source_relbase=str(row["source_relbase"]),
        vendor=str(row.get("vendor", "")),
        split=str(row.get("split", "")),
        participant_key=f"{row.get('vendor', '')}:{row.get('participant_id', '')}",
        fps=fps,
        start_frame=start,
        end_frame=stop,
        body_pose=body,
        left_hand_pose=left,
        right_hand_pose=right,
        global_orient=orient,
        translation=translation,
        smplh_valid=valid,
        speech=speech,
        audio=audio,
        sample_rate=sample_rate,
    )


def iter_clips(
    manifest: pd.DataFrame | str | Path,
    source_root: str | Path,
    *,
    with_audio: bool = True,
    with_speech: bool = True,
    skip_errors: bool = False,
) -> Iterator[Clip]:
    """Load every row of a manifest in order.

    ``skip_errors`` turns an unreadable clip into a warning instead of a raise,
    which is what a long training run wants and what a verification pass does
    not.
    """

    frame = load_manifest(manifest) if not isinstance(manifest, pd.DataFrame) else manifest
    for row in frame.to_dict("records"):
        try:
            yield load_clip(row, source_root, with_audio=with_audio, with_speech=with_speech)
        except Exception as error:  # noqa: BLE001 - re-raised unless asked not to
            if not skip_errors:
                raise
            import warnings

            warnings.warn(f"{row.get('clip_id', row.get('file_id'))}: {error}", stacklevel=2)


def _main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover - CLI demo
    """``python -m seamless_curation.dataset <manifest.csv> [source_root]``."""

    import argparse

    parser = argparse.ArgumentParser(description="Load the first clips of a manifest.")
    parser.add_argument("manifest")
    parser.add_argument("source_root", nargs="?", default="seamless_interaction")
    parser.add_argument("--limit", type=int, default=3)
    parser.add_argument("--no-audio", action="store_true")
    args = parser.parse_args(argv)

    frame = load_manifest(args.manifest)
    print(f"{len(frame)} clips, {frame['window_seconds'].sum() / 3600:.2f} hours")
    for index, clip in enumerate(
        iter_clips(frame.head(args.limit), args.source_root, with_audio=not args.no_audio)
    ):
        print(f"\n[{index}] {clip}")
        print(f"     file          {clip.file_id}  ({clip.vendor}, split={clip.split})")
        print(f"     upper_body    {clip.upper_body_pose.shape}  {clip.upper_body_pose.dtype}")
        print(f"     hands         {clip.left_hand_pose.shape} + {clip.right_hand_pose.shape}")
        print(f"     smplh_valid   {clip.smplh_valid.mean():.1%} of frames")
        if clip.audio is not None:
            print(f"     audio         {clip.audio.shape} @ {clip.sample_rate} Hz, "
                  f"peak {np.abs(clip.audio).max():.3f}")
        talk = sum(end - begin for begin, end in clip.speech)
        print(f"     speech        {len(clip.speech)} segments, {talk:.1f}s of {clip.seconds:.1f}s")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(_main())

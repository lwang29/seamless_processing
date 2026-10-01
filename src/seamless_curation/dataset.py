"""Read the annotation tables, join their levels safely, and load clip data.

Two things a downstream user needs, kept in one module:

**The annotation tables** (``<root>/annotations/*.parquet``, one per level; see
``docs/annotation_schema.md``). :func:`load_tables` reads them and checks they
come from one run (every file carries the same run hash). :func:`load_clips`
returns the clip table with any parent level joined in under a **level prefix**
— ``session__relationship``, ``interaction__posture_pair``,
``participant__bfi_extraversion`` — and optionally the time-aligned partner's
clip (``partner__*``) and the partner participant (``partner_participant__*``).
The prefix is the point: a conversation-level value on a clip row is visibly a
property of the conversation, repeated on each of its clips, not an independent
per-clip observation. Aggregate conversation-level values from the
interactions table, not by averaging clip rows.

**The released arrays for a clip** (:func:`load_clip`): the frames a clip row
names, read from the release in place. Nothing is ever copied or modified.

===========  ===========================================================
``.npz``     SMPL-H pose, 2D keypoints, and (V00 only) the movement features
``.wav``     the participant's own microphone, 48 kHz mono
``.mp4``     the participant's own camera
``.json``    transcript, VAD segments, and Meta's MOI annotations
===========  ===========================================================

The release stores one participant per stem, so the audio and video are the
participant's own channel — but not clean of the partner: close-worn
microphones separate own from partner voice by ~16-23 dB (medians), the V03 room
camera by ~9 dB (``recordings.voice_isolation_db``, ``room_camera_rig``).

Usage::

    from seamless_curation.dataset import load_clips, iter_clip_data

    clips = load_clips("/simurgh/group/lw29/seamless_annotations/annotations_v1/annotations",
                       levels=("recording", "session"), partner=True)
    subset = clips[(clips.posture == "standing") & (clips.expressivity_level == "high")
                   & (clips.speaking_role == "speaking")]
    for clip in iter_clip_data(subset, "seamless_interaction"):
        clip.body_pose, clip.left_hand_pose, clip.audio, clip.speech
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

import numpy as np
import pandas as pd

from . import schema

#: The four files that share a stem. Keys are what :func:`clip_paths` returns.
SOURCE_SUFFIXES: Mapping[str, str] = {
    "pose": ".npz",
    "audio": ".wav",
    "video": ".mp4",
    "annotation": ".json",
}

#: level name -> (table, join key on clips, prefix)
PARENT_LEVELS: Mapping[str, tuple[str, str]] = {
    "recording": ("recordings", "file_id"),
    "interaction": ("interactions", "interaction_key"),
    "window": ("interaction_windows", "window_id"),
    "session": ("sessions", "session_key"),
    "participant": ("participants", "participant_key"),
}


# ============================================================================= tables
def _resolve(root: str | Path) -> Path:
    """Pin the published version: ``annotations`` is a symlink swapped atomically on
    each publish, so resolving it once means every table below comes from one run."""

    return Path(root).resolve()


def _read(root: Path, table: str, columns: Sequence[str] | None) -> tuple[pd.DataFrame, str]:
    import pyarrow.parquet as pq

    if table not in schema.TABLES:
        raise KeyError(f"unknown table {table!r}; tables are {schema.TABLES}")
    arrow = pq.read_table(root / f"{table}.parquet", columns=list(columns) if columns else None)
    digest = (arrow.schema.metadata or {}).get(b"seamless_curation.run_hash", b"").decode()
    return arrow.to_pandas(), digest


def load_table(root: str | Path, table: str, columns: Sequence[str] | None = None) -> pd.DataFrame:
    """One annotation table (``root`` is the ``annotations`` directory)."""

    return _read(_resolve(root), table, columns)[0]


def run_hash(root: str | Path, table: str) -> str:
    import pyarrow.parquet as pq

    metadata = pq.read_schema(_resolve(root) / f"{table}.parquet").metadata or {}
    return metadata.get(b"seamless_curation.run_hash", b"").decode()


def _one_run(hashes: Mapping[str, str]) -> None:
    if len(set(hashes.values())) > 1:
        raise RuntimeError(f"tables come from different annotate runs: {dict(hashes)}")


def load_tables(root: str | Path, tables: Sequence[str] = schema.TABLES) -> dict[str, pd.DataFrame]:
    """Several tables, refusing to mix tables written by different runs.

    The run hash is checked on the tables actually read (not on a separate peek).
    """

    pinned = _resolve(root)
    frames, hashes = {}, {}
    for table in tables:
        frames[table], hashes[table] = _read(pinned, table, None)
    _one_run(hashes)
    return frames


def load_clips(root: str | Path, *, levels: Sequence[str] = (), partner: bool = False,
               columns: Sequence[str] | None = None) -> pd.DataFrame:
    """The clip table, with parent levels (and optionally the partner) joined under prefixes.

    ``levels`` is any of ``recording``, ``interaction``, ``window``, ``session``,
    ``participant``. Parent columns arrive as ``<level>__<column>``; the
    partition columns (vendor/label/split) and keys are not repeated. With
    ``partner=True`` the time-aligned partner clip's columns arrive as
    ``partner__<column>`` (NA where ``partner_link_status != 'linked'``) and
    the partner's participant row as ``partner_participant__<column>``.
    """

    pinned = _resolve(root)
    if columns is not None:
        needed = {"clip_id"} | ({"file_id", "partner_clip_id"} if partner else set())
        needed |= {PARENT_LEVELS[level][1] for level in levels if level in PARENT_LEVELS}
        columns = list(dict.fromkeys([*columns, *sorted(needed - set(columns))]))
    hashes: dict[str, str] = {}
    clips, hashes["clips"] = _read(pinned, "clips", columns)
    base_columns = set(clips.columns)
    skip = set(schema.PARTITIONS)
    for level in levels:
        if level not in PARENT_LEVELS:
            raise KeyError(f"unknown level {level!r}; choose from {sorted(PARENT_LEVELS)}")
        table, key = PARENT_LEVELS[level]
        parent, hashes[table] = _read(pinned, table, None)
        keep = [c for c in parent.columns if c != key and c not in skip]
        parent = parent[[key, *keep]].rename(columns={c: f"{level}__{c}" for c in keep})
        clips = clips.merge(parent, on=key, how="left", validate="many_to_one")
    if partner:
        full = clips[[c for c in clips.columns if c in base_columns]]
        partner_cols = [c for c in full.columns if c != "clip_id"]
        mate = full[["clip_id", *partner_cols]].rename(
            columns={"clip_id": "partner_clip_id", **{c: f"partner__{c}" for c in partner_cols}})
        clips = clips.merge(mate, on="partner_clip_id", how="left", validate="many_to_one")
        recordings, hashes["recordings"] = _read(pinned, "recordings", ["file_id", "partner_participant_key"])
        participants, hashes["participants"] = _read(pinned, "participants", None)
        people = participants.rename(columns={c: f"partner_participant__{c}" for c in participants.columns
                                              if c != "participant_key"})
        clips = (clips.merge(recordings, on="file_id", how="left")
                 .merge(people.rename(columns={"participant_key": "partner_participant_key"}),
                        on="partner_participant_key", how="left"))
    _one_run(hashes)
    return clips


# ============================================================================= clip data
@dataclass
class Clip:
    """The released arrays for one clip row."""

    clip_id: str
    file_id: str
    source_relbase: str
    participant_key: str
    fps: float
    start_frame: int
    end_frame: int
    body_pose: np.ndarray        # (frames, 21, 3) axis-angle
    left_hand_pose: np.ndarray   # (frames, 15, 3)
    right_hand_pose: np.ndarray  # (frames, 15, 3)
    global_orient: np.ndarray    # (frames, 3)
    translation: np.ndarray      # (frames, 3); ~1e12 sentinel on invalid-box frames
    smplh_valid: np.ndarray      # (frames,) bool
    keypoints: np.ndarray | None  # (frames, 133, 3) when requested
    speech: tuple[tuple[float, float], ...] | None  # own-speech segments relative to the clip (the
    # tables' mask: VAD, or transcript words where VAD is empty/stops early); None where the
    # clip's speech is unannotated (speech_annotation_status != 'annotated')
    audio: np.ndarray | None     # None when the WAV holds no samples (recordings.audio_status 'empty')
    sample_rate: int | None
    audio_samples: int = 0       # real samples in ``audio``; the rest (clip past the WAV's end) is zero padding

    @property
    def frames(self) -> int:
        return self.end_frame - self.start_frame

    @property
    def seconds(self) -> float:
        return self.frames / self.fps


def clip_paths(row: Mapping[str, Any], source_root: str | Path) -> dict[str, Path]:
    """The four source files for a row with ``source_relbase``. Nothing is read."""

    base = Path(source_root) / str(row["source_relbase"])
    return {kind: base.with_suffix(suffix) for kind, suffix in SOURCE_SUFFIXES.items()}


def _speech_segments(annotation_path: Path, n_frames: int, fps: float, start_s: float,
                     stop_s: float) -> tuple[tuple[float, float], ...]:
    """The own-speech mask the tables use (``speech.speech_track``), clipped and rebased."""

    from .speech import speech_track

    try:
        annotation = json.loads(annotation_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return ()
    segments = []
    for begin, end in speech_track(annotation, n_frames, fps).segments:
        if end <= start_s or begin >= stop_s:
            continue
        segments.append((max(begin, start_s) - start_s, min(end, stop_s) - start_s))
    return tuple(segments)


def load_clip(row: Mapping[str, Any], source_root: str | Path, *, with_audio: bool = True,
              with_speech: bool = True, with_keypoints: bool = False) -> Clip:
    """Load the frames a clip row names (``row`` needs source_relbase and fps:
    use ``load_clips(levels=("recording",))`` or pass a recording-joined row).

    Audio is a true seek; pose is not (an NPZ member decompresses whole).
    """

    if hasattr(row, "_asdict"):
        row = row._asdict()
    row = dict(row)
    for key in ("source_relbase", "fps"):
        if key not in row and f"recording__{key}" in row:
            row[key] = row[f"recording__{key}"]
    paths = clip_paths(row, source_root)
    start, stop = int(row["start_frame"]), int(row["end_frame"])
    fps = float(row["fps"])
    with np.load(paths["pose"]) as archive:
        full_body = archive["smplh:body_pose"]
        n_total = int(len(full_body))
        body = np.asarray(full_body[start:stop], dtype=np.float32)
        left = np.asarray(archive["smplh:left_hand_pose"][start:stop], dtype=np.float32)
        right = np.asarray(archive["smplh:right_hand_pose"][start:stop], dtype=np.float32)
        orient = np.asarray(archive["smplh:global_orient"][start:stop], dtype=np.float32)
        translation = np.asarray(archive["smplh:translation"][start:stop], dtype=np.float32)
        valid = np.asarray(archive["smplh:is_valid"][start:stop], dtype=bool).reshape(-1)
        keypoints = (np.asarray(archive["boxes_and_keypoints:keypoints"][start:stop], dtype=np.float32)
                     if with_keypoints else None)
    if len(body) != stop - start:
        raise ValueError(f"{row.get('clip_id', row['file_id'])}: row promises {stop - start} frames, "
                         f"the NPZ yielded {len(body)}. The source tree does not match the tables.")
    start_s, stop_s = start / fps, stop / fps
    audio: np.ndarray | None = None
    sample_rate: int | None = None
    real = 0
    if with_audio:
        import soundfile

        info = soundfile.info(str(paths["audio"]))
        if info.frames > 0:
            sample_rate = int(info.samplerate)
            begin = max(0, min(int(round(start_s * sample_rate)), info.frames))
            count = int(round((stop_s - start_s) * sample_rate))
            audio, _ = soundfile.read(str(paths["audio"]), start=begin, frames=min(count, info.frames - begin),
                                      dtype="float32", always_2d=False)
            audio = np.atleast_1d(np.asarray(audio, dtype=np.float32))
            if audio.ndim > 1:
                audio = audio.mean(axis=1)
            real = int(len(audio))
            if len(audio) < count:
                audio = np.pad(audio, (0, count - len(audio)))
    unannotated = ("speech_annotation_status" in row and row["speech_annotation_status"] != "annotated") or (
        "own_speech_s" in row and pd.isna(row["own_speech_s"]))
    if not with_speech:
        speech = ()
    elif unannotated:
        speech = None
    else:
        speech = _speech_segments(paths["annotation"], n_total, fps, start_s, stop_s)
    return Clip(clip_id=str(row.get("clip_id", "")), file_id=str(row["file_id"]),
                source_relbase=str(row["source_relbase"]), participant_key=str(row.get("participant_key", "")),
                fps=fps, start_frame=start, end_frame=stop, body_pose=body, left_hand_pose=left,
                right_hand_pose=right, global_orient=orient, translation=translation, smplh_valid=valid,
                keypoints=keypoints, speech=speech, audio=audio, sample_rate=sample_rate, audio_samples=real)


def iter_clip_data(rows: pd.DataFrame, source_root: str | Path, *, skip_errors: bool = False,
                   **kwargs: Any) -> Iterator[Clip]:
    """Load every row in order; ``skip_errors`` warns instead of raising."""

    for row in rows.to_dict("records"):
        try:
            yield load_clip(row, source_root, **kwargs)
        except Exception as error:  # noqa: BLE001 - re-raised unless asked not to
            if not skip_errors:
                raise
            import warnings

            warnings.warn(f"{row.get('clip_id', row.get('file_id'))}: {error}", stacklevel=2)


def read_back(tables: Mapping[str, pd.DataFrame], source_root: str | Path, *, sample: int = 60,
              seed: int = 0) -> dict[str, Any]:
    """Re-read sampled clips from the release and check them against their rows.

    Checks the frame count the row promises, and that the own-speech seconds
    recomputed from the release match ``own_speech_s`` (within 0.5 s) for every
    clip whose speech is annotated, whatever the speech source.
    """

    clips = tables["clips"]
    rec = tables["recordings"][["file_id", "source_relbase", "fps", "speech_source"]]
    rows = clips.sample(n=min(sample, len(clips)), random_state=seed).merge(rec, on="file_id", how="left")
    results = []
    for row in rows.to_dict("records"):
        outcome = {"clip_id": row["clip_id"], "ok": True, "problems": []}
        try:
            clip = load_clip(row, source_root, with_audio=False)
            if clip.frames != int(row["n_frames_clip"]):
                outcome["problems"].append("frame count")
            if clip.speech is not None:
                seconds = sum(b - a for a, b in clip.speech)
                if abs(seconds - float(row["own_speech_s"])) > 0.5:
                    outcome["problems"].append(f"speech {seconds:.2f} vs {row['own_speech_s']:.2f}")
        except Exception as error:  # noqa: BLE001
            outcome["problems"].append(f"{type(error).__name__}: {error}")
        outcome["ok"] = not outcome["problems"]
        results.append(outcome)
    failed = [r for r in results if not r["ok"]]
    return {"summary": {"checked": len(results), "failed": len(failed)}, "failures": failed[:20]}

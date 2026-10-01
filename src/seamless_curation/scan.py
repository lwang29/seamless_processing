"""Measure every recording once: continuous values only, one interaction at a time.

The scan is the only stage that reads the release in bulk (NPZ pose and
keypoints, JSON VAD/transcript/MOI annotations, WAV audio), so it does one
thing: turn each recording into **continuous measurements** on the clip grid.
No threshold, label, flag or normalisation is applied here — those live in
:mod:`seamless_curation.annotate`, so re-tuning a posture band or an
expressivity reference costs one annotate run and reads no media.

**The unit of work is the interaction, not the file.** Both members of a
conversation are measured in the same task, one after the other, which is what
lets the pair measures — speech overlap and turn-taking per window, how loudly
the partner bleeds into each microphone, transcript echo, annotations copied
between the two members — be computed once, from both recordings, instead of
twice from one side each. Partners sit in different release archives (98.6% of
pairs), so the shard assignment is by ``interaction_key``.

**Every recording gets a row.** A file that cannot be measured (no NPZ/JSON, an
unreadable archive, the 439 zero-frame NPZs behind missing videos) gets a
recording row with its ``measurement_status`` and no clips. Nothing is skipped
silently; ``annotate`` refuses to proceed if any catalog recording is missing
from the gathered shards.

Outputs per shard (``task_NNNN.<kind>.parquet`` plus a marker):

``recordings``  one row per recording (catalog order), recording-level measures
``clips``       one row per clip of the time-aligned grid (tail kept)
``bins``        one row per 1-s bin: the continuous posture measures
``moi``         one row per released MOI annotation entry
``windows``     one row per dyad window (clip index shared by the members)
``pairs``       one row per interaction: pair speech measures, MOI duplication

Shards are restartable: a marker records the code fingerprint (every module
whose code changes a number, the parameters, the SMPL-H model file) and the
membership (the files *and* the per-file inputs — fps, raster, anamorphic class,
path — the measurement depends on); a shard is recomputed unless both match.
"""

from __future__ import annotations

import gc
import io
import json
import math
import os
import time
import traceback
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from functools import lru_cache
from hashlib import sha256
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from . import audio, face, framing, moi, posture, speech
from .clips import clip_grid
from .gesture import GestureParams, build_tracks, clip_measures, recording_measures
from .ids import clip_id, window_id

#: Arrays every measured recording must have. ``global_orient``/``translation``
#: feed facing direction and reprojection; ``box`` feeds framing.
REQUIRED_NPZ_KEYS: tuple[str, ...] = (
    "smplh:body_pose",
    "smplh:left_hand_pose",
    "smplh:right_hand_pose",
    "smplh:global_orient",
    "smplh:translation",
    "smplh:is_valid",
    "boxes_and_keypoints:keypoints",
    "boxes_and_keypoints:is_valid_box",
    "boxes_and_keypoints:box",
)
#: Present only in V00 (Meta's Imitator features). Absent keys are not an error.
OPTIONAL_NPZ_KEYS: tuple[str, ...] = face.REQUIRED_KEYS

#: Modules whose source determines a measured number (hashed into the fingerprint).
MEASUREMENT_MODULES: tuple[str, ...] = (
    "scan.py", "gesture.py", "smplh_kinematics.py", "smplh_fk.py", "media_repair.py",
    "framing.py", "posture.py", "face.py", "audio.py", "speech.py", "moi.py", "clips.py", "ids.py",
)
SHARD_KINDS: tuple[str, ...] = ("recordings", "clips", "bins", "moi", "windows", "pairs")
#: Catalog columns a measurement depends on; hashed into the shard membership.
MEMBERSHIP_COLUMNS: tuple[str, ...] = (
    "file_id", "interaction_key", "source_relbase", "fps", "video_width", "video_height",
    "smplh_anamorphic", "npz_present", "json_present", "wav_present",
)
PAIR_TICK_HZ = 10.0
AUDIO_TICK_HZ = 20.0


@dataclass(frozen=True)
class ScanSettings:
    """Everything that changes the numbers, hashed into the shard marker."""

    clip_seconds: float = 30.0
    model_root: str = "model_files"
    gesture: GestureParams = field(default_factory=GestureParams)

    def fingerprint(self) -> str:
        payload = {
            "clip_seconds": self.clip_seconds,
            "gesture": asdict(self.gesture),
            "required_npz_keys": list(REQUIRED_NPZ_KEYS),
            "optional_npz_keys": list(OPTIONAL_NPZ_KEYS),
            "measurement_source": _measurement_source_hash(),
            "model": _model_hash(self.model_root),
        }
        return sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:16]


@lru_cache(maxsize=1)
def _measurement_source_hash() -> str:
    here = Path(__file__).parent
    digest = sha256()
    for name in MEASUREMENT_MODULES:
        digest.update(name.encode())
        digest.update((here / name).read_bytes())
    return digest.hexdigest()[:16]


@lru_cache(maxsize=4)
def _model_hash(model_root: str) -> str:
    from .smplh_kinematics import MODEL_FILENAME

    root = Path(model_root)
    for path in (root / MODEL_FILENAME, root / "smplh" / MODEL_FILENAME):
        if path.exists():
            return sha256(path.read_bytes()).hexdigest()[:16]
    return "missing"


def membership_hash(catalog_rows: pd.DataFrame) -> str:
    """Identifies *which* recordings a shard covered and the inputs it measured them with."""

    frame = catalog_rows.reindex(columns=list(MEMBERSHIP_COLUMNS)).sort_values("file_id")
    digest = sha256()
    for row in frame.itertuples(index=False):
        digest.update("\x1f".join("" if pd.isna(v) else str(v) for v in row).encode())
        digest.update(b"\x1e")
    return digest.hexdigest()[:16]


# ----------------------------------------------------------------------------- reading
def read_npz(source: Path | io.BytesIO) -> dict[str, np.ndarray]:
    with np.load(source) as archive:
        names = set(archive.files)
        missing = [key for key in REQUIRED_NPZ_KEYS if key not in names]
        if missing:
            raise KeyError(f"missing arrays {missing}")
        payload = {key: archive[key] for key in REQUIRED_NPZ_KEYS}
        for key in OPTIONAL_NPZ_KEYS:
            if key in names:
                payload[key] = archive[key]
    return payload


@dataclass
class RawFiles:
    """One recording's NPZ, JSON and WAV bytes, fetched ahead of measurement.

    The scan is I/O-bound on the network filesystem (~90% of wall time waiting
    on reads, 7.5 TB in total), and one sequential reader per task leaves most of
    the server's bandwidth idle. Fetching the next interactions' files on a few
    threads while the current one is measured keeps several reads in flight.
    ``None`` means the file does not exist; an ``OSError`` is kept as text and
    becomes the recording's ``unreadable`` status.
    """

    npz: bytes | None = None
    json: bytes | None = None
    wav: bytes | None = None
    error: str = ""


def fetch_raw(record: Mapping[str, Any], source_root: Path) -> RawFiles:
    base = source_root / str(record["source_relbase"])
    raw = RawFiles()
    for suffix, present_key in ((".npz", "npz_present"), (".json", "json_present"), (".wav", "wav_present")):
        if not bool(record.get(present_key)):
            continue
        try:
            setattr(raw, suffix[1:], base.with_suffix(suffix).read_bytes())
        except FileNotFoundError:
            pass
        except OSError as error:
            raw.error = f"{type(error).__name__}: {error}"[:300]
    return raw


@dataclass
class Member:
    """One recording's measurements, plus what the pair step needs from it."""

    file_id: str
    recording: dict[str, Any]
    clips: list[dict[str, Any]] = field(default_factory=list)
    bins: pd.DataFrame | None = None
    events: list[dict[str, Any]] = field(default_factory=list)
    parties: set[str] = field(default_factory=set)
    speech_track: Any = None
    audio_track: Any = None
    clip_ends_s: list[float] = field(default_factory=list)

    @property
    def measured(self) -> bool:
        return self.recording.get("measurement_status") == "measured"


def _nan_if_none(value: Any) -> Any:
    return float("nan") if value is None else value


def measure_recording(record: Mapping[str, Any], source_root: Path, settings: ScanSettings,
                      raw: RawFiles | None = None) -> Member:
    """Measure one recording. A failure is data (``measurement_status``), never an exception.

    ``raw`` carries prefetched file bytes; without it the files are read here.
    """

    started = time.perf_counter()
    file_id = str(record["file_id"])
    base = source_root / str(record["source_relbase"])
    row: dict[str, Any] = {"file_id": file_id}
    member = Member(file_id=file_id, recording=row)

    def finish(status: str, error: str = "") -> Member:
        row["measurement_status"] = status
        row["scan_error"] = error
        row["scan_seconds"] = float(time.perf_counter() - started)
        row.setdefault("recording_n_clips", 0)
        return member

    if not (bool(record.get("npz_present")) and bool(record.get("json_present"))):
        return finish("missing_files")
    if raw is None:
        raw = fetch_raw(record, source_root)
    if raw.error:
        return finish("unreadable", raw.error)
    if raw.npz is None or raw.json is None:
        return finish("missing_files")
    try:
        payload = read_npz(io.BytesIO(raw.npz))
        annotation = json.loads(raw.json.decode("utf-8"))
    except Exception as error:  # BadZipFile, zlib.error, bad JSON, missing arrays, ...
        return finish("unreadable", f"{type(error).__name__}: {error}"[:300])
    raw.npz = raw.json = None  # release the compressed bytes early
    n_frames = int(len(payload["smplh:body_pose"]))
    row["n_frames"] = n_frames
    if n_frames == 0:
        return finish("no_frames")
    fps = float(record.get("fps") or float("nan"))
    if not (fps == fps and fps > 0):
        return finish("unreadable", "no container frame rate for a non-empty pose array")
    width = int(record["video_width"]) if pd.notna(record.get("video_width")) else 0
    height = int(record["video_height"]) if pd.notna(record.get("video_height")) else 0
    anamorphic = str(record.get("smplh_anamorphic") or "none")

    try:
        track = speech.speech_track(annotation, n_frames, fps)
        tracks = build_tracks(payload, track.mask, fps=fps, model_root=settings.model_root,
                              params=settings.gesture)
        turns = framing.detect_quarter_turns(payload["boxes_and_keypoints:keypoints"])
        frame_track = framing.framing_track(
            payload, tracks.joints, fps=fps, width=width, height=height,
            quarter_turns=turns, smplh_anamorphic=anamorphic,
        )
        visible = framing.visible_mask(
            payload["boxes_and_keypoints:keypoints"], width, height,
            np.asarray(payload["boxes_and_keypoints:is_valid_box"]).reshape(-1).astype(bool),
        )
        pframes = posture.posture_frames(tracks.joints, visible, smplh_anamorphic=anamorphic, fps=fps)
        bins = posture.posture_bins(pframes, fps)
        face_track = face.face_track(payload, n_frames)
        audio_track = (audio.read_audio(None, data=raw.wav) if raw.wav is not None
                       else audio.read_audio(base.with_suffix(".wav")))  # 'missing' when absent
        raw.wav = None
        own20 = speech.time_mask(track, AUDIO_TICK_HZ)
        duration_s = n_frames / fps
        events = moi.parse_events(annotation, file_id, duration_s)
        parties = moi.annotated_parties(annotation)
        grid = clip_grid(n_frames, fps, settings.clip_seconds)
    except Exception as error:  # a bug or a pathological file: record it, keep the shard alive
        return finish("unreadable", f"{type(error).__name__}: {error} | "
                      + traceback.format_exc(limit=3).replace("\n", " ")[-300:])

    try:
        clips: list[dict[str, Any]] = []
        for bounds in grid:
            start, stop = bounds.start_frame, bounds.end_frame
            centre = 0.5 * (bounds.start_s + bounds.end_s)
            clip = {
                "clip_id": clip_id(file_id, bounds.clip_index, settings.clip_seconds),
                "file_id": file_id,
                "clip_index": bounds.clip_index,
                "start_frame": start,
                "end_frame": stop,
                "n_frames_clip": bounds.n_frames,
                "start_s": bounds.start_s,
                "end_s": bounds.end_s,
                "clip_seconds": bounds.seconds,
                "is_partial": bool(bounds.is_partial),
                "is_last": bounds.clip_index == len(grid) - 1,
                "relative_position": float(min(1.0, max(0.0, centre / duration_s))),
            }
            clip.update(clip_measures(tracks, start, stop))
            clip.update(framing.clip_framing(frame_track, start, stop))
            clip.update(posture.clip_posture_measures(pframes, start, stop, fps=fps))
            clip.update(face.clip_face(face_track, start, stop, fps))
            clip.update(audio.clip_audio(audio_track, own20, bounds.start_s, bounds.end_s))
            clip.update(speech.clip_speech(track, start, stop, fps))
            # MOI times are whole seconds on the nominal grid [k*L, (k+1)*L); frame-rounded
            # bounds would put an event ending at 30 s into clip 1 at 29.97 fps.
            nominal_start = bounds.clip_index * settings.clip_seconds
            nominal_end = min((bounds.clip_index + 1) * settings.clip_seconds, duration_s)
            clip.update(moi.clip_moi(events, parties, nominal_start, nominal_end))
            clips.append(clip)

        # Indices from the grid actually used (frame boundaries), not nominal k*L:
        # at 29.97 fps clip 4 ends at 150.0165 s. The end index is the clip holding
        # the event's last instant, so an event ending exactly on a boundary does
        # not name the next clip.
        starts = np.asarray([b.clip_index * settings.clip_seconds for b in grid], dtype=np.float64)
        last_index = len(grid) - 1
        for event in events:
            start_s, end_s = float(event["moi_start_s"]), float(event["moi_end_s"])
            if not (start_s == start_s and end_s == end_s):
                event["moi_clip_index_start"] = event["moi_clip_index_end"] = None
                continue
            lo, hi = min(start_s, end_s), max(start_s, end_s)
            first = int(np.searchsorted(starts, lo, side="right")) - 1
            last = int(np.searchsorted(starts, hi, side="left")) - 1 if hi > lo else first
            event["moi_clip_index_start"] = int(min(max(first, 0), last_index))
            event["moi_clip_index_end"] = int(min(max(last, first, 0), last_index))

        # Speech is annotated only as far as the audio it was derived from: a WAV
        # shorter than the pose grid (226 files by > 1 s) leaves the rest unknown.
        annotated_until = float(duration_s)
        if audio_track.status == "ok" and audio_track.duration_s == audio_track.duration_s:
            last_speech = max([end for _, end in track.segments] + [end for _, end, _ in track.words] + [0.0])
            annotated_until = min(float(duration_s), max(float(audio_track.duration_s), last_speech))
        row.update({
            "speech_annotated_until_s": annotated_until,
            "recording_duration_s": float(duration_s),
            "recording_n_clips": len(clips),
            "quarter_turns": int(turns),
        })
        row.update(recording_measures(tracks))
        row.update(framing.recording_framing(frame_track))
        row.update(posture.recording_posture_measures(pframes))
        row.update(face.recording_face(face_track))
        row.update(speech.recording_speech(track))
        row.update(audio.recording_audio(audio_track, own20, None))
        row.update(moi.recording_moi(events, parties, duration_s))
    except Exception as error:  # a per-clip bug or pathological file: a status, not a dead shard
        return finish("unreadable", f"{type(error).__name__}: {error} | "
                      + traceback.format_exc(limit=3).replace("\n", " ")[-300:])

    bins = bins.copy()
    bins.insert(0, "file_id", file_id)
    member.clips = clips
    member.bins = bins
    member.events = events
    member.parties = parties
    member.speech_track = track
    member.audio_track = audio_track
    member.clip_ends_s = [c["end_s"] for c in clips]
    del tracks, frame_track, pframes, payload
    return finish("measured")


# ----------------------------------------------------------------------------- pairs
def pair_measures(interaction_key: str, members: Sequence[Member], clip_seconds: float) -> tuple[list[dict], dict]:
    """Window and interaction rows from both members (singletons get member-2 NA)."""

    ordered = sorted(members, key=lambda m: m.file_id)  # member 1/2 order = sorted participant id
    first = ordered[0]
    second = ordered[1] if len(ordered) > 1 else None
    both = second is not None and first.measured and second.measured
    n_windows = max((len(m.clips) for m in ordered), default=0)
    ticks = {}
    if both:
        ticks = {m.file_id: speech.time_mask(m.speech_track, PAIR_TICK_HZ) for m in ordered}
    windows: list[dict[str, Any]] = []
    for k in range(n_windows):
        clip_a = first.clips[k] if k < len(first.clips) else None
        clip_b = second.clips[k] if (second is not None and k < len(second.clips)) else None
        present = [c for c in (clip_a, clip_b) if c is not None]
        start_s = k * clip_seconds
        end_s = min(c["end_s"] for c in present)
        window = {
            "window_id": window_id(interaction_key, k, clip_seconds),
            "interaction_key": interaction_key,
            "window_index": k,
            "member_clip_id_1": clip_a["clip_id"] if clip_a else None,
            "member_clip_id_2": clip_b["clip_id"] if clip_b else None,
            "window_start_s": float(start_s),
            "window_end_s": float(end_s),
        }
        if both and clip_a is not None and clip_b is not None:
            window.update(speech.pair_window(ticks[first.file_id], ticks[second.file_id],
                                             start_s, end_s, hz=PAIR_TICK_HZ))
        windows.append(window)

    pair: dict[str, Any] = {"interaction_key": interaction_key}
    if both:
        end_s = min(first.recording["speech_annotated_until_s"], second.recording["speech_annotated_until_s"])
        pair.update(speech.pair_interval(
            ticks[first.file_id], ticks[second.file_id], end_s,
            first.recording.get("recording_own_speech_s", 0.0),
            second.recording.get("recording_own_speech_s", 0.0), hz=PAIR_TICK_HZ,
        ))
        for own, other in ((first, second), (second, first)):
            own20 = speech.time_mask(own.speech_track, AUDIO_TICK_HZ)
            other20 = speech.time_mask(other.speech_track, AUDIO_TICK_HZ)
            own.recording.update(audio.recording_audio(own.audio_track, own20, other20))
            # An unannotated partner transcript is unknown, not "no echo".
            own.recording["recording_transcript_echo_frac"] = (
                speech.transcript_echo_frac(own.speech_track.words, other.speech_track.words)
                if other.speech_track.words else float("nan"))
        moi.mark_duplicates(first.events, second.events)
        for party in ("3P", "1P"):
            pair[f"moi_duplication_{party.lower()}"] = moi.duplication_status(
                first.events, second.events, party, first.parties, second.parties)
    return windows, pair


# ----------------------------------------------------------------------------- shards
def scan_members(catalog_rows: pd.DataFrame, source_root: Path, settings: ScanSettings,
                 *, log: Any = None, progress_every: int = 25, prefetch_depth: int = 2,
                 prefetch_workers: int = 4) -> dict[str, pd.DataFrame]:
    """Scan every interaction of a shard. Returns one DataFrame per shard kind.

    The files of the next ``prefetch_depth`` interactions are read on
    ``prefetch_workers`` threads while the current one is measured.
    """

    out: dict[str, list] = {kind: [] for kind in SHARD_KINDS}
    groups = catalog_rows.groupby("interaction_key", sort=True)
    ordered = [(key, group.to_dict("records")) for key, group in groups]
    pool = ThreadPoolExecutor(max_workers=prefetch_workers) if prefetch_workers > 0 else None
    pending: dict[int, list[Future]] = {}

    def schedule(index: int) -> None:
        if pool is not None and index < len(ordered) and index not in pending:
            pending[index] = [pool.submit(fetch_raw, record, source_root) for record in ordered[index][1]]

    for ahead in range(prefetch_depth + 1):
        schedule(ahead)
    for count, (interaction_key, records) in enumerate(ordered, start=1):
        index = count - 1
        schedule(index + prefetch_depth)
        futures = pending.pop(index, None)
        raws = [f.result() for f in futures] if futures else [None] * len(records)
        members = [measure_recording(record, source_root, settings, raw)
                   for record, raw in zip(records, raws)]
        del raws
        windows, pair = pair_measures(str(interaction_key), members, settings.clip_seconds)
        for member in members:
            out["recordings"].append(member.recording)
            out["clips"].extend(member.clips)
            if member.bins is not None and len(member.bins):
                out["bins"].append(member.bins)
            out["moi"].extend(member.events)
        out["windows"].extend(windows)
        out["pairs"].append(pair)
        del members
        if count % 25 == 0:
            gc.collect()
        if log is not None and count % progress_every == 0:
            log(f"{count}/{len(ordered)} interactions")
    if pool is not None:
        pool.shutdown(wait=True)
    frames = {kind: pd.DataFrame(rows) for kind, rows in out.items() if kind != "bins"}
    frames["bins"] = (pd.concat(out["bins"], ignore_index=True) if out["bins"]
                      else pd.DataFrame(columns=["file_id", "bin_index"]))
    return frames


def shard_path(shard_dir: Path, task_index: int, kind: str) -> Path:
    return shard_dir / f"task_{task_index:04d}.{kind}.parquet"


def marker_path(shard_dir: Path, task_index: int) -> Path:
    return shard_dir / f"task_{task_index:04d}.complete.json"


def shard_is_current(shard_dir: Path, task_index: int, fingerprint: str, membership: str,
                     kinds: Sequence[str] = SHARD_KINDS) -> bool:
    marker = marker_path(shard_dir, task_index)
    if not marker.exists() or not all(shard_path(shard_dir, task_index, k).exists() for k in kinds):
        return False
    try:
        recorded = json.loads(marker.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    return (recorded.get("status") == "complete" and recorded.get("fingerprint") == fingerprint
            and recorded.get("membership") == membership)


def write_shard(shard_dir: Path, task_index: int, frames: Mapping[str, pd.DataFrame],
                marker: Mapping[str, Any]) -> None:
    """Atomic same-directory writes of every table, then the marker."""

    shard_dir.mkdir(parents=True, exist_ok=True)
    for kind, frame in frames.items():
        path = shard_path(shard_dir, task_index, kind)
        temporary = path.with_suffix(".tmp")
        _stable_dtypes(frame).to_parquet(temporary, index=False)
        os.replace(temporary, path)
    temporary = marker_path(shard_dir, task_index).with_suffix(".tmp")
    temporary.write_text(json.dumps(dict(marker), indent=2, sort_keys=True), encoding="utf-8")
    os.replace(temporary, marker_path(shard_dir, task_index))


def _stable_dtypes(frame: pd.DataFrame) -> pd.DataFrame:
    """Object columns of numbers/None become float, so shards concatenate cleanly.

    A column that is all ``None`` in one shard would otherwise be written as Arrow
    ``null`` and concatenate to ``object`` with the float columns of other shards.
    Final tables are cast to the registry schema in ``annotate``; this only keeps
    the intermediate shards well-typed.
    """

    out = frame.copy()
    for column in out.columns:
        series = out[column]
        if series.dtype != object:
            continue
        values = series.dropna()
        if values.empty:
            out[column] = series.astype("float64")
        elif values.map(lambda v: isinstance(v, (bool, np.bool_))).all():
            out[column] = series.astype("boolean")
        elif values.map(lambda v: isinstance(v, (int, float, np.integer, np.floating))
                        and not isinstance(v, (bool, np.bool_))).all():
            out[column] = pd.to_numeric(series, errors="coerce").astype("float64")
        else:
            out[column] = series.astype("string")
    return out


def read_shards(shard_dir: Path, expected_tasks: int, kind: str, *, fingerprint: str | None = None,
                memberships: Mapping[int, str] | None = None) -> pd.DataFrame:
    """Concatenate every shard of ``kind``; refuse on any gap, stale fingerprint or membership."""

    problems: list[str] = []
    frames = []
    for index in range(expected_tasks):
        path = shard_path(shard_dir, index, kind)
        marker = marker_path(shard_dir, index)
        if not path.exists() or not marker.exists():
            problems.append(f"task {index}: missing")
            continue
        if fingerprint is not None or memberships is not None:
            recorded = json.loads(marker.read_text(encoding="utf-8"))
            if fingerprint is not None and recorded.get("fingerprint") != fingerprint:
                problems.append(f"task {index}: fingerprint {recorded.get('fingerprint')} != {fingerprint}")
            if memberships is not None and recorded.get("membership") != memberships.get(index):
                problems.append(f"task {index}: membership changed since the scan")
        frames.append(pd.read_parquet(path))
    if problems:
        raise RuntimeError(f"{len(problems)} of {expected_tasks} {kind} shards unusable: {problems[:5]}")
    frames = [f for f in frames if len(f)]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

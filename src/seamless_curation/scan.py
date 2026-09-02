"""Stage 2: measure every eligible file once, at window resolution.

One pass over a participant file yields two kinds of row:

``window rows``
    One per sliding window (default 30 s, 10 s hop). These are what the gates
    read, because gesture and speech are both bursty: over 5-second spans the
    dev corpus has 44.7% with no speech at all and more than half with both
    wrists inside 100 mm of their own median, against whole-file figures that
    look nothing like either. A whole-file verdict averages a minute of
    gesturing together with five minutes of stillness and calls the file fine.

``file rows``
    One per file, holding the population join keys plus roll-ups of the window
    rows. Useful for census tables and for participant-level quotas; never used
    to accept data on its own.

The scan is measurement only. No threshold is applied here, so re-tuning a gate
costs one ``select`` run and reads no media — the same separation the v0
pipeline had between ``v00_fm_scan`` and ``v00_fm_analyze``, kept because it was
the right call.

Shards are restartable. A shard writes ``task_NNNN.parquet`` plus a marker
recording the code fingerprint and the scan settings; a rerun with an identical
fingerprint skips the shard, and a changed one recomputes it. The v0 scan wrote
markers that nothing ever read and that omitted the config hash, so a shard
could not be attributed to the settings that produced it.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import asdict, dataclass
from functools import lru_cache
from hashlib import sha256
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence

import numpy as np
import pandas as pd

from .gesture import GestureParams, build_tracks, sliding_windows, window_measures

#: Only these arrays are pulled out of the NPZ. The movement:* block is 60% of
#: the bytes and none of it is used: ``movement:is_valid`` is all-zero on files
#: whose SMPL-H and keypoints are perfect, and the hypernet/expression features
#: are not upper-body pose.
NPZ_KEYS: tuple[str, ...] = (
    "smplh:body_pose",
    "smplh:left_hand_pose",
    "smplh:right_hand_pose",
    "smplh:is_valid",
    "boxes_and_keypoints:keypoints",
    "boxes_and_keypoints:is_valid_box",
)


@dataclass(frozen=True)
class ScanSettings:
    """Everything that changes the numbers, hashed into the shard marker."""

    window_seconds: float = 30.0
    hop_seconds: float = 10.0
    model_root: str = "model_files"
    gesture: GestureParams = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.gesture is None:
            object.__setattr__(self, "gesture", GestureParams())
        if not 5.0 <= self.window_seconds <= 120.0:
            raise ValueError("window_seconds must be in [5, 120]")
        if not 1.0 <= self.hop_seconds <= self.window_seconds:
            raise ValueError("hop_seconds must be in [1, window_seconds]")

    def fingerprint(self) -> str:
        """Identifies the settings *and the code* that produced a shard.

        The measurement source is hashed rather than a hand-maintained version
        number. A version number has to be remembered, and forgetting it is
        silent and expensive: shards computed before and after an edit would
        carry the same marker and be concatenated into one table. Hashing
        ``gesture.py`` and ``smplh_kinematics.py`` makes a stale shard impossible
        by construction, at the cost of recomputing after a comment change —
        which is the right way round.
        """

        payload = {
            "window_seconds": self.window_seconds,
            "hop_seconds": self.hop_seconds,
            "gesture": asdict(self.gesture),
            "npz_keys": list(NPZ_KEYS),
            "measurement_source": _measurement_source_hash(),
        }
        return sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:16]


@lru_cache(maxsize=1)
def _measurement_source_hash() -> str:
    """SHA-256 over the modules whose code determines a measured number."""

    here = Path(__file__).parent
    digest = sha256()
    for name in ("gesture.py", "smplh_kinematics.py"):
        digest.update((here / name).read_bytes())
    return digest.hexdigest()[:16]


def read_bundle(base: Path) -> tuple[dict[str, np.ndarray], list[dict[str, float]]]:
    """Load only what the measure needs from one participant bundle."""

    with np.load(base.with_suffix(".npz")) as archive:
        payload = {key: archive[key] for key in NPZ_KEYS if key in archive.files}
    missing = [key for key in NPZ_KEYS if key not in payload]
    if missing:
        raise KeyError(f"{base.name}: missing arrays {missing}")
    annotation = json.loads(base.with_suffix(".json").read_text(encoding="utf-8"))
    return payload, list(annotation.get("metadata:vad") or [])


def scan_file(
    base: Path,
    *,
    fps: float,
    settings: ScanSettings,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Measure one file. Returns ``(file_row, window_rows)``.

    A failure is data, not an exception: the file row carries ``scan_status``
    and the window list is empty, so a shard never dies on one bad bundle and
    the census can account for every file.
    """

    started = time.perf_counter()
    try:
        payload, vad = read_bundle(base)
    except (OSError, ValueError, KeyError, EOFError) as error:
        return {"scan_status": f"read_error:{type(error).__name__}", "scan_seconds": 0.0}, []

    frames = int(len(payload["smplh:body_pose"]))
    window_frames = int(round(settings.window_seconds * fps))
    hop_frames = int(round(settings.hop_seconds * fps))
    if frames < window_frames:
        return (
            {"scan_status": f"too_few_frames:{frames}", "frames": frames, "scan_seconds": 0.0},
            [],
        )

    tracks = build_tracks(
        payload, vad, fps=fps, model_root=settings.model_root, params=settings.gesture
    )
    windows = sliding_windows(frames, window_frames, hop_frames)
    rows: list[dict[str, Any]] = []
    for index, (start, stop) in enumerate(windows):
        measures = window_measures(tracks, start, stop)
        measures.update(
            {
                "window_index": index,
                "start_frame": int(start),
                "end_frame": int(stop),
                "start_s": float(start / fps),
            }
        )
        rows.append(measures)

    file_row: dict[str, Any] = {
        "scan_status": "ok",
        "frames": frames,
        "fps": float(fps),
        "duration_s": float(frames / fps),
        "window_count": len(rows),
        "speech_frac_file": float(tracks.speech.mean()),
        "speech_seconds_file": float(tracks.speech.sum() / fps),
        "smplh_valid_frac_file": float(tracks.smplh_valid.mean()),
        "box_valid_frac_file": float(tracks.box_valid.mean()),
        "hand_frozen_frac_file": float(tracks.hand_frozen.mean()),
        "gesture_frac_file": float(tracks.active.mean()),
        "scan_seconds": float(time.perf_counter() - started),
    }
    return file_row, rows


def scan_shard(
    population: pd.DataFrame,
    source_root: Path,
    settings: ScanSettings,
    *,
    progress_every: int = 100,
    log: Any = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Scan every row of a shard. Returns ``(file_frame, window_frame)``."""

    file_rows: list[dict[str, Any]] = []
    window_rows: list[dict[str, Any]] = []
    for count, record in enumerate(population.to_dict("records"), start=1):
        base = source_root / str(record["source_relbase"])
        fps = float(record["nominal_fps"])
        file_row, windows = scan_file(base, fps=fps, settings=settings)
        identity = {
            "file_id": record["file_id"],
            "vendor": record["vendor"],
            "label": record["label"],
            "split": record["split"],
            "session_id": record["session_id"],
            "participant_id": record["participant_id"],
            "interaction_id": record["interaction_id"],
            "interaction_type": record["interaction_type"],
            "source_relbase": record["source_relbase"],
        }
        file_rows.append({**identity, **file_row})
        for window in windows:
            window_rows.append({**identity, **window})
        if log is not None and count % progress_every == 0:
            log(f"{count}/{len(population)} files, {len(window_rows)} windows")
    files = pd.DataFrame(file_rows)
    windows_frame = pd.DataFrame(window_rows)
    return files, windows_frame


def marker_path(shard_dir: Path, task_index: int) -> Path:
    return shard_dir / f"task_{task_index:04d}.complete.json"


def shard_paths(shard_dir: Path, task_index: int) -> tuple[Path, Path, Path]:
    return (
        shard_dir / f"task_{task_index:04d}.files.parquet",
        shard_dir / f"task_{task_index:04d}.windows.parquet",
        marker_path(shard_dir, task_index),
    )


def membership_hash(file_ids: Iterable[str]) -> str:
    """Identifies *which* files a shard covered, not just how it measured them."""

    digest = sha256()
    for file_id in sorted(str(f) for f in file_ids):
        digest.update(file_id.encode("utf-8"))
        digest.update(b"\x1f")
    return digest.hexdigest()[:16]


def shard_is_current(
    shard_dir: Path, task_index: int, fingerprint: str, membership: str | None = None
) -> bool:
    """True when this shard was produced by these settings *over these files*.

    Membership matters as much as the settings. Without it, a ``--limit 2``
    smoke test writes a complete-looking shard and the real un-limited scan is
    then a no-op; so is re-running at a different ``--tasks``, which repartitions
    every file. In both cases ``gather`` succeeds and reports a fraction of the
    corpus as the whole of it.
    """

    files, windows, marker = shard_paths(shard_dir, task_index)
    if not (files.exists() and windows.exists() and marker.exists()):
        return False
    try:
        recorded = json.loads(marker.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    if recorded.get("status") != "complete" or recorded.get("fingerprint") != fingerprint:
        return False
    return membership is None or recorded.get("membership") == membership


def write_shard(
    shard_dir: Path,
    task_index: int,
    files: pd.DataFrame,
    windows: pd.DataFrame,
    marker: Mapping[str, Any],
) -> None:
    """Atomic same-directory write of both tables and then the marker."""

    shard_dir.mkdir(parents=True, exist_ok=True)
    files_path, windows_path, marker_path_ = shard_paths(shard_dir, task_index)
    for frame, path in ((files, files_path), (windows, windows_path)):
        temporary = path.with_suffix(".tmp")
        frame.to_parquet(temporary, index=False)
        os.replace(temporary, path)
    temporary = marker_path_.with_suffix(".tmp")
    temporary.write_text(json.dumps(dict(marker), indent=2, sort_keys=True), encoding="utf-8")
    os.replace(temporary, marker_path_)


def read_shards(shard_dir: Path, expected_tasks: int, kind: str) -> pd.DataFrame:
    """Concatenate every shard, refusing to proceed if any is missing.

    The v0 analysis globbed whatever shards happened to exist, so a round in
    which five of sixty-four tasks died silently reported its flag rates over
    59/64 of the sample. Completeness is checked here instead.
    """

    if kind not in {"files", "windows"}:
        raise ValueError("kind must be 'files' or 'windows'")
    missing = [
        index
        for index in range(expected_tasks)
        if not (shard_dir / f"task_{index:04d}.{kind}.parquet").exists()
    ]
    if missing:
        raise FileNotFoundError(
            f"{len(missing)} of {expected_tasks} {kind} shards missing: {missing[:10]}"
        )
    frames = [
        pd.read_parquet(shard_dir / f"task_{index:04d}.{kind}.parquet")
        for index in range(expected_tasks)
    ]
    return pd.concat(frames, ignore_index=True)

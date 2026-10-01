"""Audio level annotations of every recording and clip, from the released WAVs.

Every WAV is reduced once to a 50 ms RMS level envelope (:func:`read_audio`,
``level_db = 20*log10(rms + 1e-10)`` in dBFS, full scale 1.0, tick ``i``
covering ``[i/20, (i+1)/20)`` s). The recording and clip columns are statistics
of that envelope, alone or restricted to the 20 Hz own/partner speech masks of
:func:`seamless_curation.speech.time_mask` (aligned by tick index and truncated
to the shorter, since the WAV and the pose grid can differ in length). Nothing
is filtered: a dead or empty track gets its status and NA measures.

What the files are (68 WAVs probed read-only: 15 random pairs across the five
close-mic/room rigs, 24 random single files, 6 more pairs end to end through
this module, and the V03_S0763_I00000515 pair)
------------------------------------------------------------------------------
* 48 kHz mono, stored as 32-bit FLOAT whose values lie exactly on the 16-bit
  grid (every non-zero ``|x| >= 2**-15``; peaks below 1.0). They are read as
  float32 in 60 s blocks, so a 218M-sample file never sits in memory whole and
  no float->int16 conversion can wrap an over-range sample (the contract asked
  for a whole-file int16 read to save memory; streaming saves more and is
  exact). Reading takes 0.5-6 s per uncached file. On all 14 end-to-end files the
  envelope has exactly as many ticks as ``speech.time_mask(track, 20)``.
* 578 WAVs are 58-byte headers with no samples ('empty'; none in V00).
* **Exact zeros are common in healthy files.** Quiet stretches of this
  denoised, 16-bit-quantised audio round to 0: 0.2-29 % of samples on the 66
  normal files (32 % on V00_S0644_I00000129_P0799), against 34 % / 38 % on the
  V03_S0763_I00000515 dropout pair. ``audio_exact_zero_frac`` is published as
  defined, but **it does not separate dropouts from healthy files**: 31 of the
  66 normal files, and 21 of the 24 V00 files, exceed 5 %. What does separate
  them in this sample is the share of 50 ms ticks at the 16-bit floor (level
  < -88 dBFS, RMS ~1.3 LSB): at most 25 % of ticks on the 24 random files,
  84 % / 96 % on the dropout pair. None of the 30 pair-probe files has an
  all-zero tick (-200 dBFS), and on the 26 single files no own-speech tick sits
  below -88 dBFS (not even on the dropout pair, whose VAD marks speech only
  where there is signal), so the -200 dBFS of digital silence does not reach
  the speech statistics in practice.

Recording columns
-----------------
* ``audio_envelope_dynamics_db`` = p95 - p50 of the whole envelope: the v0 FM4
  measure (dead tracks 0.20-1.35 dB, usable 1.70-69.04 on 48 labels; here
  10.2-70.8 on normal files, and 1.75 on the dropout file P1651, just above the
  1.5 dB cut). Unchanged except for the epsilon (1e-10, v0 1e-12), which only
  moves all-zero ticks.
* ``recording_own_speech_level_db``: median envelope level over the own-speech
  ticks (>= 1 s of them): the typical speaking level, vendor-dependent by
  mic gain (probe medians V00 -51, V02 -47, V03 room -39, V01 -29, V03 portrait
  -25 dBFS).
* ``recording_partner_only_level_db`` and ``voice_isolation_db`` are **equivalent
  levels** (Leq: ``10*log10`` of the mean power over the ticks), not medians --
  a deviation from the contract, for a real-data reason. The released audio is
  bleed-suppressed, so the *median* partner-only tick is the file's noise floor
  (on 26 probed files it sits a median 5.5 dB below the file's overall median
  level, never more than 4.2 dB above it, down to -90 dBFS at the 16-bit
  floor), and median(own-only) - median(partner-only) measured the speech level
  above the floor: 22-65 dB, with room-camera files (29-47 dB) inside the
  close-mic range. The mean power over partner-only ticks is dominated by the
  ticks where the partner's voice actually comes through, i.e. by what bleeds
  in. ``voice_isolation_db = Leq(own & ~partner) - Leq(partner & ~own)``, each
  over >= 5 s of ticks. Over 18 V00 files it is a median 20.7 dB (1.9-35.9),
  V01 16.8 (14.6-20.7, n=6), V02 23.8 (14.0-30.8, n=6), V03 portrait 19.5
  (6.5-25.4, n=4) and the V03 room camera 9.5 (4.6-15.9, n=6).
  v0 pooled the power over the two *full* VADs, overlap included, and its
  documented 7.3 dB (V00) / 2.2 dB (room camera) are reproduced that way (here
  7.5 / 1.3). Inference: those small numbers are mostly overlap, not bleed --
  with the partner's VAD covering a few percent of the participant's own
  speech, the participant's voice dominates the pooled "partner" power (with
  7 % overlap and negligible bleed the pooled ratio is about
  ``-10*log10(0.07)`` = 11.5 dB whatever the microphone), and a room camera's
  VAD also fires on the other person. Excluding overlap ticks, as
  the contract asks, removes that. As v0 found, it is a measurement, not a
  verdict; room-camera files remain the lowest.

Clip columns (``[start_s, end_s)`` in seconds on the recording's clock)
------------------------------------------------------------------------
Ticks ``round(start_s*20) .. round(end_s*20)`` of the envelope (the rounding of
the speech masks). ``audio_rms_db_p50`` is their median level, NaN when the clip
holds less than 1 s of audio (the WAV can end before the video; 226 files are
more than 1 s short) -- a stricter NA condition than the registry's "audio not
ok", for the same reason the speech statistics have minimum durations.
``own_speech_level_db`` is the median over the clip's own-speech ticks (>= 1 s
of them) and ``vocal_level_range_db`` their p90 - p10 (>= 3 s): how much the
voice's energy moves. Over the 100 clips of the 14 end-to-end files it is a
median 27.8 dB (p5-p95 23.2-37.8); recording-wide it reads 19-33 dB on close
mics and 46-52 dB on the room camera, whose own-speech ticks probably include
the far-field partner.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from .clips import frames_for

#: Level-envelope hop: 50 ms (20 Hz), the grid of speech.time_mask(track, 20).
HOP_S = 0.05
LEVEL_HZ = 1.0 / HOP_S
#: 20*log10(rms + LEVEL_EPS): an all-zero frame reads -200 dBFS.
LEVEL_EPS = 1e-10
SILENT_FRAME_DB = 20.0 * math.log10(LEVEL_EPS)
#: WAVs of at most this many bytes carry no audio (the release's 578 header-only 58-byte files).
EMPTY_WAV_MAX_BYTES = 1024
#: A 50-ms tick below this level is at the 16-bit quantisation floor: digital
#: silence rather than a quiet room. Healthy denoised tracks spend <= 25% of their
#: ticks there (24 random files); the known dropout pair V03_S0763_I00000515
#: spends 84% / 96%. Exact-zero *samples*, by contrast, are common in healthy
#: tracks (0.2-32%: quiet stretches of 16-bit audio round to 0).
FLOOR_TICK_DB = -88.0
#: Ticks per streamed read block (60 s): bounds memory on the 218M-sample files.
_BLOCK_TICKS = 1200


@dataclass
class AudioTrack:
    """One WAV reduced to its 50 ms level envelope."""

    status: str  # 'ok' | 'empty' | 'unreadable' | 'missing'
    sample_rate: int | None
    duration_s: float  # NaN unless ok
    level_db: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.float32))
    zero_frac: float = float("nan")  # share of (mono) samples exactly 0; NaN unless ok


def _not_ok(status: str, sample_rate: int | None = None) -> AudioTrack:
    return AudioTrack(status=status, sample_rate=sample_rate, duration_s=float("nan"),
                      level_db=np.zeros(0, dtype=np.float32), zero_frac=float("nan"))


def _tick_bounds(n_ticks: int, sample_rate: int) -> np.ndarray:
    """Sample index where each 50 ms tick starts: ``round(i * sr * HOP_S)``.

    Exact integer hops at the release's 48 kHz (2,400 samples); for a rate that
    is not a multiple of 20 the hop length alternates so tick ``i`` still starts
    at ``i / 20`` s and never drifts from the 20 Hz speech masks.
    """

    return np.rint(np.arange(n_ticks + 1, dtype=np.float64) * sample_rate * HOP_S).astype(np.int64)


def read_audio(path: Path | None, data: bytes | None = None) -> AudioTrack:
    """Read one WAV into its 50 ms RMS level envelope (dBFS, full scale 1.0).

    ``data`` holds the file's bytes when the scan has already fetched them (it
    prefetches the next interaction's files to hide network-filesystem latency);
    ``path`` is then ignored. ``data=None`` with a missing path is 'missing'.
    """

    import io

    import soundfile as sf  # imported here: only audio annotation needs it

    if data is None:
        path = Path(path)
        try:
            size = path.stat().st_size
        except FileNotFoundError:
            return _not_ok("missing")
        except OSError:
            return _not_ok("unreadable")
        source: Any = str(path)
    else:
        size = len(data)
        source = io.BytesIO(data)
    if size <= EMPTY_WAV_MAX_BYTES:
        return _not_ok("empty")
    try:
        with sf.SoundFile(source) as handle:
            sample_rate = int(handle.samplerate)
            declared = int(handle.frames)
            if declared <= 0 or sample_rate <= 0:
                return _not_ok("empty", sample_rate if sample_rate > 0 else None)
            n_ticks = frames_for(declared / sample_rate, LEVEL_HZ)
            bounds = _tick_bounds(n_ticks, sample_rate)
            bounds[-1] = min(int(bounds[-1]), declared)
            levels: list[np.ndarray] = []
            zeros = 0
            total = 0
            for first in range(0, n_ticks, _BLOCK_TICKS):
                last = min(first + _BLOCK_TICKS, n_ticks)
                want = int(bounds[last] - bounds[first])
                block = handle.read(frames=want, dtype="float32", always_2d=True)
                got = int(block.shape[0])
                if got == 0:
                    break
                mono = block[:, 0] if block.shape[1] == 1 else block.mean(axis=1, dtype=np.float64)
                zeros += int(np.count_nonzero(mono == 0))
                total += got
                starts = bounds[first:last] - bounds[first]
                starts = starts[starts < got]
                power = np.square(mono, dtype=np.float64)
                sums = np.add.reduceat(power, starts)
                counts = np.diff(np.append(starts, got))
                levels.append(20.0 * np.log10(np.sqrt(sums / counts) + LEVEL_EPS))
                if got < want:  # the header over-declared the length
                    break
    except (RuntimeError, OSError, ValueError, EOFError):  # LibsndfileError is a RuntimeError
        return _not_ok("unreadable")
    if total == 0:
        return _not_ok("empty", sample_rate)
    level_db = np.concatenate(levels).astype(np.float32)
    return AudioTrack(status="ok", sample_rate=sample_rate, duration_s=total / sample_rate,
                      level_db=level_db, zero_frac=zeros / total)


# ----------------------------------------------------------------------------- statistics
#: Minimum speech time behind each level statistic.
MIN_OWN_SPEECH_S = 1.0
MIN_RANGE_SPEECH_S = 3.0
MIN_ISOLATION_S = 5.0
#: A clip level needs at least this much audio inside the clip.
MIN_CLIP_AUDIO_S = 1.0

_NAN = float("nan")


def _ticks(seconds: float) -> int:
    return frames_for(seconds, LEVEL_HZ)


def _aligned(level: np.ndarray, *masks: np.ndarray | None) -> tuple[np.ndarray, ...]:
    """Envelope and masks cut to their common length (alignment is by tick index)."""

    arrays = [np.asarray(level, dtype=np.float64)]
    arrays += [np.asarray(m, dtype=bool).reshape(-1) for m in masks if m is not None]
    n = min(a.shape[0] for a in arrays)
    return tuple(a[:n] for a in arrays)


def _median(values: np.ndarray, min_count: int) -> float:
    return float(np.median(values)) if values.size >= max(1, min_count) else _NAN


def equivalent_level_db(level_db: np.ndarray) -> float:
    """Leq: ``10*log10`` of the mean power of 50 ms levels (NaN when empty)."""

    values = np.asarray(level_db, dtype=np.float64)
    if values.size == 0:
        return _NAN
    return float(10.0 * np.log10(np.mean(np.power(10.0, values / 10.0))))


def _leq(values: np.ndarray, min_count: int) -> float:
    return equivalent_level_db(values) if values.size >= max(1, min_count) else _NAN


def envelope_dynamics_db(level_db: np.ndarray) -> float:
    """p95 - p50 of the level envelope (the FM4 dead-track measure)."""

    values = np.asarray(level_db, dtype=np.float64)
    if values.size == 0:
        return _NAN
    p50, p95 = np.percentile(values, [50.0, 95.0])
    return float(p95 - p50)


def recording_audio(track: AudioTrack | None, own_20hz: np.ndarray | None,
                    partner_20hz: np.ndarray | None) -> dict[str, Any]:
    """Recording-level audio columns.

    ``own_20hz`` / ``partner_20hz`` are ``speech.time_mask(track, 20.0)`` of the
    participant and of the partner (``None`` without a measured partner).
    ``voice_isolation_db`` is NaN unless both the own-only and the partner-only
    ticks reach 5 s.
    """

    if track is None:
        track = _not_ok("missing")
    out: dict[str, Any] = {
        "audio_status": track.status,
        "audio_duration_s": _NAN,
        "audio_envelope_dynamics_db": _NAN,
        "audio_exact_zero_frac": _NAN,
        "audio_floor_tick_frac": _NAN,
        "recording_own_speech_level_db": _NAN,
        "recording_partner_only_level_db": _NAN,
        "voice_isolation_db": _NAN,
    }
    if track.status != "ok":
        return out
    out["audio_duration_s"] = float(track.duration_s)
    out["audio_envelope_dynamics_db"] = envelope_dynamics_db(track.level_db)
    out["audio_exact_zero_frac"] = float(track.zero_frac)
    levels = np.asarray(track.level_db, dtype=np.float64)
    out["audio_floor_tick_frac"] = float((levels < FLOOR_TICK_DB).mean()) if levels.size else _NAN
    if own_20hz is None:
        return out
    # The own-speech level is a property of this track alone: never cut it to the
    # partner's (possibly shorter) mask.
    level_own, own_only_mask = _aligned(track.level_db, own_20hz)
    out["recording_own_speech_level_db"] = _median(level_own[own_only_mask], _ticks(MIN_OWN_SPEECH_S))
    if partner_20hz is None:  # no measured partner: nothing can be called partner-only
        return out
    level, own, partner = _aligned(track.level_db, own_20hz, partner_20hz)
    need = _ticks(MIN_ISOLATION_S)
    partner_only = _leq(level[partner & ~own], need)
    own_only = _leq(level[own & ~partner], need)
    out["recording_partner_only_level_db"] = partner_only
    if math.isfinite(partner_only) and math.isfinite(own_only):
        out["voice_isolation_db"] = own_only - partner_only
    return out


def clip_audio(track: AudioTrack | None, own_20hz: np.ndarray | None,
               start_s: float, end_s: float) -> dict[str, Any]:
    """Clip-level audio columns for the clip ``[start_s, end_s)`` (seconds)."""

    out: dict[str, Any] = {"audio_rms_db_p50": _NAN, "own_speech_level_db": _NAN,
                           "vocal_level_range_db": _NAN}
    if track is None or track.status != "ok":
        return out
    first = max(0, int(round(float(start_s) * LEVEL_HZ)))
    last = int(round(float(end_s) * LEVEL_HZ))
    level = np.asarray(track.level_db, dtype=np.float64)[first:max(first, last)]
    out["audio_rms_db_p50"] = _median(level, _ticks(MIN_CLIP_AUDIO_S))
    if own_20hz is None:
        return out
    own = np.asarray(own_20hz, dtype=bool).reshape(-1)[first:max(first, last)]
    level, own = _aligned(level, own)
    speech = level[own]
    out["own_speech_level_db"] = _median(speech, _ticks(MIN_OWN_SPEECH_S))
    if speech.size >= _ticks(MIN_RANGE_SPEECH_S):
        p10, p90 = np.percentile(speech, [10.0, 90.0])
        out["vocal_level_range_db"] = float(p90 - p10)
    return out

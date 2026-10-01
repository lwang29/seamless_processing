"""Own speech, dyadic turn-taking and transcript echo, from Meta's VAD and transcripts.

Each recording's JSON carries two speech annotations of the recorded participant:
``metadata:vad``, a list of ``{start, end}`` intervals in seconds, and
``metadata:transcript``, a list of segments whose ``words`` carry
``{word, start, end, score}``. This module turns them into one own-speech mask
per recording, the clip- and recording-level speech columns, and the pair
measures that need both members' masks (overlap, mutual silence, turn switches)
or both transcripts (echo). It reads no audio: loudness is ``audio.py``'s job,
and it takes the 20 Hz masks built here.

Where the mask comes from (``speech_source``)
---------------------------------------------
* **'vad'** whenever the VAD has at least one interval with ``end > start``
  (finite numbers). VAD is millisecond-precise and never overruns the media;
  2,160 sampled JSONs (400 random, 100 per vendor, plus every zero-speech file
  of the v0 scan) held no zero-length or out-of-file interval. Intervals map to
  pose frames with ``round(t * fps)``, exactly as the previous pipeline did
  (:func:`seamless_curation.gesture.speech_mask`, reused).
* **'transcript_only'** when the VAD is empty but the transcript has timed
  words. An empty VAD is a fact about the annotator, not the recording: 1,766 of
  118,570 files scanned in v0 had no VAD speech, and 673 of the 1,760 of them
  re-read here have timed words (median 150 words; V03 536, V02 90, V01 33,
  V00 14). A VAD-only mask would call those people silent and their partners
  'speaking' over nobody.
* **'none'** when both are empty. Silence and a missing annotation are then
  indistinguishable, so the pair measures are NA (the scan's ``speech_status``)
  and ``speaking_role`` is 'unknown'; the own-speech columns read 0, which the
  source column qualifies.

**Word spans are bridged across gaps of up to 0.5 s** (:data:`WORD_GAP_BRIDGE_S`)
before they become the transcript-only mask. Raw word spans leave out the pauses
between words that VAD includes: over the 394 random files that have both, the
union of raw word spans is a median 0.76 of the VAD time (p10-p90 0.66-1.11,
IoU with VAD 0.61). Bridging gaps of <= 0.5 s gives 0.98 (0.87-1.31) and IoU
0.77 (0.25 s: 0.91; 1.0 s: 1.10), so the fallback mask measures the same thing
as the VAD mask it stands in for. ``SpeechTrack.segments`` holds the bridged
spans.

Words
-----
Words with a null ``start`` or ``end`` are dropped (570 of 246,752 words, in 264
of the 2,160 files). A word belongs to the clip containing its midpoint,
``[start_s, end_s)`` of the clip. ``recording_word_count`` counts the words whose
midpoint lies on the recording's pose grid ``[0, n_frames / fps)``, so the clip
counts of a recording sum to it; one word falls past the grid in 8 of 2,160
files (the audio runs a fraction of a second past the last pose frame).

Clip columns
------------
``own_speech_s`` / ``own_speech_frac`` are the mask's seconds and share over the
clip's frames. ``speech_rate_wps`` = ``word_count / own_speech_s``, NaN below
1 s of own speech (a ratio over a fraction of a second is noise).
``speech_cut_at_start`` is True when the mask is on both at the clip's first
frame and at the frame before it (an utterance began before the clip);
``speech_cut_at_end`` likewise across the clip's end. At the recording's first
and last frame nothing is observed beyond the boundary, so the flags are False
there (a recording that starts mid-utterance is not flagged).

Pair measures (10 Hz ticks)
---------------------------
Both members' masks are rebuilt on a common time grid (:func:`time_mask`;
tick ``i`` covers ``[i/hz, (i+1)/hz)``, on when its centre is inside an
interval), because pose grids differ between members (1,147 pairs mix frame
rates). Over a window, ``overlap`` = ticks where both speak / ticks where either
speaks (NaN if nobody speaks: 0/0 is not "no overlap"); ``mutual_silence`` =
ticks where neither speaks / all ticks; the *floor holder* of a tick is its sole
speaker, and a turn switch is a change between successive holders, ignoring
ticks with no speaker or with both (A, both, A is no switch; A, silence, B is
one). The interaction versions use the whole shared interval, the balance
``min/max`` of the members' own-speech seconds (NaN when both are 0) and the
switch rate per minute. High overlap is either real overlapping talk or speaker
bleed; the echo measure below separates them.

Transcript echo (speaker-bleed indicator)
-----------------------------------------
``transcript_echo_frac`` is the share of own words (lowercased, punctuation
stripped, at least 5 letters) that also occur in the partner's transcript with
midpoints within 0.3 s. If the partner's microphone picked the participant up,
the partner's ASR transcribes the same words at the same time. The orientation
probe (3+ letters, word starts) found 0-4% in 48 of 52 files and 13-21% in a
quarter-turned room-camera pair (V03_S0965); a 100-pair probe found 75% in
V02_S3198_I00000767. Five letters drop the short function words ('yeah',
'okay', 'the') that both people say at the same moment for real; NaN below 20
eligible own words, where one or two chance matches would dominate. With this
definition (5+ letters, midpoints) four ordinary pairs (V00, V01, V03 VAD and
V03 transcript-only) read 0.00 for both members, V03_S0965_I00000162 0.15 and
0.11 (overlap 0.52, against 0.00-0.14 in the ordinary pairs) and
V02_S3198_I00000767 0.34 and 0.49; two pairs with speech_source 'none' on both
sides read NaN. A member whose partner has no timed words reads 0.0 when it
has 20 eligible words (nothing was echoed); the pair's ``speech_status`` says
whether the partner was annotated at all.

Verification on those eight pairs (read-only): clip ``own_speech_s`` and
``word_count`` sum exactly to the recording values in every file, and the
10 Hz tick masks reproduce the pose-grid own-speech seconds within 1.7%.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from functools import cached_property
from typing import Any, Mapping, Sequence

import numpy as np

from .clips import frames_for
from .gesture import speech_mask

SPEECH_SOURCES: tuple[str, ...] = ("vad", "vad+transcript_tail", "transcript_only", "none")
#: VAD that stops early: when at least ``TAIL_MIN_WORDS`` timed words start more
#: than ``TAIL_GAP_S`` after the last VAD interval, the VAD is taken to have been
#: truncated and those words' bridged spans are added (``vad+transcript_tail``).
#: Seen in ~1.8% of V03 recordings (e.g. V03_S1338_I00000487_P2947: VAD ends at
#: 73.6 s, 260 words follow to 192.9 s at the file's own-speech level).
TAIL_MIN_WORDS = 20
TAIL_GAP_S = 5.0

#: Gaps between consecutive transcript words up to this long are bridged when
#: the words stand in for an empty VAD (see the module docstring for the evidence).
WORD_GAP_BRIDGE_S = 0.5
#: speech_rate_wps needs at least this much own speech in the clip.
MIN_RATE_SPEECH_S = 1.0
#: Tick rate of the pair (dyad) measures.
PAIR_HZ = 10.0

_NON_WORD = re.compile(r"[\W_]+", flags=re.UNICODE)


@dataclass
class SpeechTrack:
    """One recording's own speech on its pose grid, with the timed words."""

    mask: np.ndarray  # (T,) bool own speech on the pose grid
    source: str  # 'vad' | 'transcript_only' | 'none'
    segments: list[tuple[float, float]]  # seconds: VAD intervals, or bridged word spans
    words: list[tuple[float, float, str]]  # (start, end, word), null timings dropped
    duration_s: float  # n_frames / fps

    @property
    def n_frames(self) -> int:
        return int(self.mask.shape[0])

    @cached_property
    def word_midpoints(self) -> np.ndarray:
        """Sorted word midpoints in seconds (for counting words in a span)."""

        if not self.words:
            return np.zeros(0, dtype=np.float64)
        mids = np.array([0.5 * (start + end) for start, end, _ in self.words], dtype=np.float64)
        return np.sort(mids)


# ----------------------------------------------------------------------------- parsing
def _finite(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def vad_segments(annotation: Mapping[str, Any]) -> list[tuple[float, float]]:
    """The released VAD intervals with ``end > start``, sorted by start."""

    out: list[tuple[float, float]] = []
    for segment in annotation.get("metadata:vad") or ():
        if not isinstance(segment, Mapping):
            continue
        start, end = _finite(segment.get("start")), _finite(segment.get("end"))
        if start is not None and end is not None and end > start:
            out.append((start, end))
    out.sort()
    return out


def timed_words(annotation: Mapping[str, Any]) -> list[tuple[float, float, str]]:
    """Every transcript word with numeric start and end, in time order."""

    words: list[tuple[float, float, str]] = []
    for segment in annotation.get("metadata:transcript") or ():
        if not isinstance(segment, Mapping):
            continue
        for word in segment.get("words") or ():
            if not isinstance(word, Mapping):
                continue
            start, end = _finite(word.get("start")), _finite(word.get("end"))
            if start is None or end is None:
                continue
            text = word.get("word")
            words.append((start, end, "" if text is None else str(text)))
    words.sort(key=lambda item: (item[0], item[1]))  # stable: ties keep release order
    return words


def merge_spans(spans: Sequence[tuple[float, float]], gap_s: float) -> list[tuple[float, float]]:
    """Union of ``[start, end]`` spans, joining neighbours separated by <= ``gap_s``."""

    ordered = sorted((float(a), float(b)) for a, b in spans if b > a)
    merged: list[tuple[float, float]] = []
    for start, end in ordered:
        if merged and start - merged[-1][1] <= gap_s:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged


def _mask(segments: Sequence[tuple[float, float]], n: int, rate: float) -> np.ndarray:
    return speech_mask([{"start": a, "end": b} for a, b in segments], n, rate)


def speech_track(
    annotation: Mapping[str, Any], n_frames: int, fps: float, *, word_gap_s: float = WORD_GAP_BRIDGE_S
) -> SpeechTrack:
    """Own speech of one recording: VAD, else bridged transcript words, else none."""

    n_frames = int(n_frames)
    duration_s = n_frames / float(fps) if fps and fps > 0 else float("nan")
    vad = vad_segments(annotation)
    words = timed_words(annotation)
    if vad:
        source, segments = "vad", vad
        last_vad = max(end for _, end in vad)
        tail = [(start, end) for start, end, _ in words if start > last_vad + TAIL_GAP_S]
        if len(tail) >= TAIL_MIN_WORDS:
            source = "vad+transcript_tail"
            segments = sorted(vad + merge_spans(tail, word_gap_s))
    elif words:
        source = "transcript_only"
        segments = merge_spans([(start, end) for start, end, _ in words], word_gap_s)
    else:
        source, segments = "none", []
    mask = _mask(segments, n_frames, float(fps)) if segments else np.zeros(n_frames, dtype=bool)
    return SpeechTrack(mask=mask, source=source, segments=segments, words=words, duration_s=duration_s)


def time_mask(track: SpeechTrack, hz: float) -> np.ndarray:
    """Own speech on a ``hz`` time grid: ``ceil(duration * hz)`` ticks from t = 0.

    Tick ``i`` covers ``[i/hz, (i+1)/hz)``; intervals map with ``round(t * hz)``,
    the rule the pose-grid mask uses, so a tick is on when its centre is inside
    an interval. Masks of the two members of a pair line up by index.
    """

    n = frames_for(track.duration_s, hz) if math.isfinite(track.duration_s) else 0
    if not track.segments or n == 0:
        return np.zeros(n, dtype=bool)
    return _mask(track.segments, n, float(hz))


# ----------------------------------------------------------------------------- clip / recording
def _count_between(sorted_values: np.ndarray, low: float, high: float) -> int:
    """Values in ``[low, high)``."""

    if sorted_values.size == 0:
        return 0
    return int(np.searchsorted(sorted_values, high, side="left")
               - np.searchsorted(sorted_values, low, side="left"))


def clip_speech(track: SpeechTrack, start: int, stop: int, fps: float) -> dict[str, Any]:
    """Speech columns of the clip covering pose frames ``[start, stop)``."""

    n = track.n_frames
    if not (0 <= start < stop <= n):
        raise ValueError(f"clip frames [{start}, {stop}) outside the recording's {n} frames")
    segment = track.mask[start:stop]
    own_s = float(np.count_nonzero(segment)) / float(fps)
    words = _count_between(track.word_midpoints, start / float(fps), stop / float(fps))
    return {
        "own_speech_s": own_s,
        "own_speech_frac": float(np.count_nonzero(segment)) / float(stop - start),
        "word_count": int(words),
        "speech_rate_wps": float(words / own_s) if own_s >= MIN_RATE_SPEECH_S else float("nan"),
        "speech_cut_at_start": bool(start > 0 and track.mask[start - 1] and track.mask[start]),
        "speech_cut_at_end": bool(stop < n and track.mask[stop - 1] and track.mask[stop]),
    }


def recording_speech(track: SpeechTrack) -> dict[str, Any]:
    """Recording-level speech columns (the pose grid is the recording)."""

    n = track.n_frames
    on = int(np.count_nonzero(track.mask))
    own_s = on * track.duration_s / n if n else 0.0
    words = _count_between(track.word_midpoints, 0.0, track.duration_s) if n else 0
    return {
        "speech_source": track.source,
        "recording_own_speech_s": float(own_s),
        "recording_own_speech_frac": float(on / n) if n else float("nan"),
        "recording_word_count": int(words),
    }


# ----------------------------------------------------------------------------- pairs
@dataclass(frozen=True)
class _PairCounts:
    ticks: int
    both: int
    either: int
    neither: int
    switches: int


def _pair_counts(ticks_a: np.ndarray, ticks_b: np.ndarray, first: int, last: int) -> _PairCounts:
    stop = min(int(last), len(ticks_a), len(ticks_b))
    begin = max(0, int(first))
    if stop <= begin:
        return _PairCounts(0, 0, 0, 0, 0)
    a = np.asarray(ticks_a[begin:stop], dtype=bool)
    b = np.asarray(ticks_b[begin:stop], dtype=bool)
    both = int(np.count_nonzero(a & b))
    either = int(np.count_nonzero(a | b))
    holder = np.where(a & ~b, 1, np.where(b & ~a, 2, 0))
    holders = holder[holder > 0]
    switches = int(np.count_nonzero(holders[1:] != holders[:-1])) if holders.size > 1 else 0
    return _PairCounts(stop - begin, both, either, stop - begin - either, switches)


def pair_window(
    ticks_a: np.ndarray, ticks_b: np.ndarray, start_s: float, end_s: float, hz: float = PAIR_HZ
) -> dict[str, Any]:
    """Dyad speech measures over ``[start_s, end_s)`` from both members' tick masks."""

    counts = _pair_counts(ticks_a, ticks_b, round(start_s * hz), round(end_s * hz))
    if counts.ticks == 0:
        return {"window_speech_overlap_frac": float("nan"), "window_mutual_silence_frac": float("nan"),
                "window_turn_switches": None}
    return {
        "window_speech_overlap_frac": counts.both / counts.either if counts.either else float("nan"),
        "window_mutual_silence_frac": counts.neither / counts.ticks,
        "window_turn_switches": int(counts.switches),
    }


def pair_interval(
    ticks_a: np.ndarray,
    ticks_b: np.ndarray,
    end_s: float,
    own_s_a: float,
    own_s_b: float,
    hz: float = PAIR_HZ,
) -> dict[str, Any]:
    """Interaction speech measures over the shared interval ``[0, end_s)``."""

    counts = _pair_counts(ticks_a, ticks_b, 0, round(end_s * hz))
    own = [_finite(own_s_a), _finite(own_s_b)]
    if None in own or max(own) <= 0:
        balance = float("nan")
    else:
        balance = min(own) / max(own)
    if counts.ticks == 0:
        return {"interaction_speech_overlap_frac": float("nan"),
                "interaction_mutual_silence_frac": float("nan"),
                "interaction_speech_balance": balance,
                "interaction_turn_switch_rate_per_min": float("nan")}
    minutes = counts.ticks / float(hz) / 60.0
    return {
        "interaction_speech_overlap_frac": counts.both / counts.either if counts.either else float("nan"),
        "interaction_mutual_silence_frac": counts.neither / counts.ticks,
        "interaction_speech_balance": balance,
        "interaction_turn_switch_rate_per_min": counts.switches / minutes,
    }


def normalise_word(word: str) -> str:
    """Lowercase with punctuation and whitespace removed ('Today,' -> 'today')."""

    return _NON_WORD.sub("", str(word).lower())


def transcript_echo_frac(
    own_words: Sequence[tuple[float, float, str]],
    partner_words: Sequence[tuple[float, float, str]],
    *,
    tol_s: float = 0.3,
    min_letters: int = 5,
    min_words: int = 20,
) -> float:
    """Share of eligible own words also in the partner's transcript within ``tol_s``."""

    eligible: list[tuple[float, str]] = []
    for start, end, text in own_words:
        token = normalise_word(text)
        if sum(ch.isalpha() for ch in token) >= min_letters:
            eligible.append((0.5 * (start + end), token))
    if len(eligible) < min_words:
        return float("nan")
    index: dict[str, list[float]] = {}
    for start, end, text in partner_words:
        index.setdefault(normalise_word(text), []).append(0.5 * (start + end))
    lookup = {token: np.sort(np.asarray(mids)) for token, mids in index.items()}
    hits = 0
    for mid, token in eligible:
        mids = lookup.get(token)
        if mids is None:
            continue
        j = int(np.searchsorted(mids, mid))
        near = [abs(mids[k] - mid) for k in (j - 1, j) if 0 <= k < mids.size]
        if near and min(near) <= tol_s + 1e-9:
            hits += 1
    return hits / len(eligible)

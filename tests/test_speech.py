"""Own speech, pair turn-taking and transcript echo on annotations whose answer we set.

Every fixture is a hand-built JSON fragment (``metadata:vad`` and
``metadata:transcript`` in the released shapes), so each test states what the
measure must say about a known conversation rather than what it said about a file.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from seamless_curation import speech
from seamless_curation.speech import (
    SpeechTrack,
    clip_speech,
    pair_interval,
    pair_window,
    recording_speech,
    speech_track,
    time_mask,
    transcript_echo_frac,
)

FPS = 30.0


def word(text: str, start: float | None, end: float | None) -> dict:
    return {"word": text, "start": start, "end": end, "score": 0.9}


def annotation(vad=(), words=()) -> dict:
    return {
        "metadata:vad": [{"start": a, "end": b} for a, b in vad],
        "metadata:transcript": [{"start": 0.0, "end": 1.0, "transcript": "x", "words": list(words)}],
    }


def track_from_ticks(on: np.ndarray, hz: float = 10.0) -> SpeechTrack:
    """A track whose segments reproduce ``on`` exactly on a ``hz`` grid."""

    segments, i = [], 0
    while i < len(on):
        if on[i]:
            j = i
            while j < len(on) and on[j]:
                j += 1
            segments.append((i / hz, j / hz))
            i = j
        else:
            i += 1
    return SpeechTrack(mask=np.zeros(0, bool), source="vad", segments=segments, words=[],
                       duration_s=len(on) / hz)


# ----------------------------------------------------------------------------- source
def test_vad_is_the_source_when_present_and_maps_with_round() -> None:
    ann = annotation(vad=[(1.0, 2.0), (3.02, 3.05)], words=[word("hello", 5.0, 5.5)])
    track = speech_track(ann, 300, FPS)
    assert track.source == "vad"
    assert track.segments == [(1.0, 2.0), (3.02, 3.05)]
    # round(3.02*30)=91, round(3.05*30)=92 (banker's rounding of 91.5): one frame
    expected = np.zeros(300, bool)
    expected[30:60] = True
    expected[91:92] = True
    assert np.array_equal(track.mask, expected)
    # words are kept even when the VAD is the mask
    assert track.words == [(5.0, 5.5, "hello")]


def test_empty_vad_falls_back_to_bridged_word_spans() -> None:
    words = [word("one", 1.0, 1.3), word("two", 1.6, 2.0), word("three", 4.0, 4.4)]
    ann = annotation(vad=[], words=words)
    track = speech_track(ann, 300, FPS)
    assert track.source == "transcript_only"
    # 0.3 s gap bridged (<= 0.5 s), 2.0 s gap is not
    assert track.segments == [(1.0, 2.0), (4.0, 4.4)]
    assert track.mask[30:60].all() and not track.mask[60:120].any() and track.mask[120:132].all()
    assert recording_speech(track)["speech_source"] == "transcript_only"


def test_zero_length_vad_does_not_count_as_speech() -> None:
    ann = annotation(vad=[(2.0, 2.0)], words=[word("hello", 1.0, 1.5)])
    assert speech_track(ann, 300, FPS).source == "transcript_only"


def test_no_vad_and_no_timed_words_is_none() -> None:
    ann = annotation(vad=[], words=[word("ghost", None, None), word("half", 1.0, None)])
    track = speech_track(ann, 300, FPS)
    assert track.source == "none"
    assert track.words == [] and track.segments == []
    assert not track.mask.any()
    rec = recording_speech(track)
    assert rec == {"speech_source": "none", "recording_own_speech_s": 0.0,
                   "recording_own_speech_frac": 0.0, "recording_word_count": 0}


def test_missing_keys_are_none_not_an_error() -> None:
    track = speech_track({}, 90, FPS)
    assert track.source == "none" and track.mask.shape == (90,)


def test_null_timed_words_are_dropped_and_words_are_time_sorted() -> None:
    ann = {"metadata:vad": [{"start": 0.0, "end": 9.0}],
           "metadata:transcript": [
               {"words": [word("later", 5.0, 5.2), word("null", None, 3.0)]},
               {"words": [word("early", 1.0, 1.2)]},
           ]}
    track = speech_track(ann, 300, FPS)
    assert [w for _, _, w in track.words] == ["early", "later"]


# ----------------------------------------------------------------------------- clips
def test_word_belongs_to_the_clip_holding_its_midpoint() -> None:
    # clip boundary at 1.0 s (frame 30): midpoint 0.99 -> clip 0, 1.0 -> clip 1
    words = [word("before", 0.90, 1.08), word("atedge", 0.95, 1.05), word("after", 1.5, 1.7)]
    track = speech_track(annotation(vad=[(0.0, 2.0)], words=words), 60, FPS)
    first, second = clip_speech(track, 0, 30, FPS), clip_speech(track, 30, 60, FPS)
    assert first["word_count"] == 1  # 'before' (mid 0.99)
    assert second["word_count"] == 2  # 'atedge' (mid exactly 1.0) and 'after'
    assert first["word_count"] + second["word_count"] == recording_speech(track)["recording_word_count"]


def test_words_past_the_pose_grid_are_not_counted() -> None:
    track = speech_track(annotation(vad=[(0.0, 1.0)], words=[word("a", 0.2, 0.4), word("b", 2.9, 3.2)]), 90, FPS)
    assert recording_speech(track)["recording_word_count"] == 1


def test_clip_speech_seconds_fraction_and_rate() -> None:
    words = [word(f"w{i}", 2.0 + 0.4 * i, 2.2 + 0.4 * i) for i in range(10)]  # mids 2.1 .. 5.7
    track = speech_track(annotation(vad=[(2.0, 6.0)], words=words), 300, FPS)
    out = clip_speech(track, 0, 300, FPS)
    assert out["own_speech_s"] == pytest.approx(4.0)
    assert out["own_speech_frac"] == pytest.approx(0.4)
    assert out["word_count"] == 10
    assert out["speech_rate_wps"] == pytest.approx(2.5)


def test_speech_rate_is_nan_below_one_second_of_speech() -> None:
    track = speech_track(annotation(vad=[(1.0, 1.9)], words=[word("quick", 1.2, 1.5)]), 300, FPS)
    out = clip_speech(track, 0, 300, FPS)
    assert out["word_count"] == 1
    assert math.isnan(out["speech_rate_wps"])
    silent = speech_track(annotation(), 300, FPS)
    assert math.isnan(clip_speech(silent, 0, 300, FPS)["speech_rate_wps"])
    assert clip_speech(silent, 0, 300, FPS)["own_speech_s"] == 0.0


def test_speech_cut_flags_only_where_an_utterance_crosses_the_boundary() -> None:
    track = speech_track(annotation(vad=[(0.5, 1.5), (2.9, 3.0)]), 120, FPS)  # frames 15-45, 87-90
    first, second = clip_speech(track, 0, 30, FPS), clip_speech(track, 30, 60, FPS)
    assert first["speech_cut_at_end"] and second["speech_cut_at_start"]
    assert not first["speech_cut_at_start"] and not second["speech_cut_at_end"]
    # speech that ends exactly at a boundary is not cut
    third = clip_speech(track, 60, 90, FPS)
    assert not third["speech_cut_at_end"]
    # at the recording's own edges nothing is observed beyond
    edge = speech_track(annotation(vad=[(0.0, 4.0)]), 120, FPS)
    assert not clip_speech(edge, 0, 30, FPS)["speech_cut_at_start"]
    assert not clip_speech(edge, 90, 120, FPS)["speech_cut_at_end"]


def test_clip_sums_equal_recording_values() -> None:
    rng = np.random.default_rng(0)
    starts = np.sort(rng.uniform(0, 95, 20))
    vad = [(float(s), float(s + rng.uniform(0.2, 2.0))) for s in starts]
    words = [word("w", float(s), float(s) + 0.2) for s in rng.uniform(0, 99.5, 50)]
    n = 2997  # 100 s at 29.97
    track = speech_track(annotation(vad=vad, words=words), n, 29.97)
    from seamless_curation.clips import clip_grid

    clips = [clip_speech(track, b.start_frame, b.end_frame, 29.97) for b in clip_grid(n, 29.97, 30.0)]
    rec = recording_speech(track)
    assert sum(c["own_speech_s"] for c in clips) == pytest.approx(rec["recording_own_speech_s"])
    assert sum(c["word_count"] for c in clips) == rec["recording_word_count"]


def test_clip_outside_the_recording_raises() -> None:
    track = speech_track(annotation(), 30, FPS)
    with pytest.raises(ValueError):
        clip_speech(track, 0, 31, FPS)


# ----------------------------------------------------------------------------- time grid
def test_time_mask_length_and_alignment_across_frame_rates() -> None:
    ann = annotation(vad=[(1.0, 2.0)])
    a = speech_track(ann, 300, 30.0)
    b = speech_track(ann, 2997, 29.97)  # 100 s
    ta, tb = time_mask(a, 10.0), time_mask(b, 10.0)
    assert ta.shape == (100,) and tb.shape == (1000,)
    assert ta[10:20].all() and not ta[:10].any() and not ta[20:].any()
    assert np.array_equal(ta, tb[:100])
    assert time_mask(speech_track(annotation(), 300, FPS), 20.0).shape == (200,)
    # ceil: 10.01 s -> 101 ticks at 10 Hz
    assert time_mask(speech_track(ann, 1001, 100.0), 10.0).shape == (101,)


# ----------------------------------------------------------------------------- pairs
def test_overlap_silence_and_switches_on_constructed_ticks() -> None:
    # A: 0-1 s, B: 1-2 s, both 2-2.5 s, silence 2.5-3 s, A 3-4 s  (10 Hz ticks)
    a = np.zeros(40, bool)
    b = np.zeros(40, bool)
    a[0:10] = True
    b[10:20] = True
    a[20:25] = b[20:25] = True
    a[30:40] = True
    out = pair_window(a, b, 0.0, 4.0)
    assert out["window_speech_overlap_frac"] == pytest.approx(5 / 35)
    assert out["window_mutual_silence_frac"] == pytest.approx(5 / 40)
    # holders: A, B, (both ignored), (silence ignored), A -> A->B, B->A = 2
    assert out["window_turn_switches"] == 2


def test_turn_switch_ignores_both_and_silence_between_holders() -> None:
    a = np.array([1, 1, 1, 0, 0, 1, 1], bool)
    b = np.array([0, 0, 1, 0, 0, 0, 0], bool)
    # holders: A A (both) - - A A -> no switch (A, both, A)
    assert pair_window(a, b, 0.0, 0.7)["window_turn_switches"] == 0
    a2 = np.array([1, 0, 0, 0], bool)
    b2 = np.array([0, 0, 0, 1], bool)
    # A, silence, silence, B -> one switch
    assert pair_window(a2, b2, 0.0, 0.4)["window_turn_switches"] == 1


def test_window_where_nobody_speaks_has_nan_overlap_not_zero() -> None:
    silent = np.zeros(300, bool)
    out = pair_window(silent, silent, 0.0, 30.0)
    assert math.isnan(out["window_speech_overlap_frac"])
    assert out["window_mutual_silence_frac"] == 1.0
    assert out["window_turn_switches"] == 0


def test_window_selects_its_ticks_and_empty_window_is_na() -> None:
    a = np.zeros(600, bool)
    b = np.zeros(600, bool)
    a[300:310] = True  # only in the second window
    first, second = pair_window(a, b, 0.0, 30.0), pair_window(a, b, 30.0, 60.0)
    assert first["window_mutual_silence_frac"] == 1.0
    assert second["window_mutual_silence_frac"] == pytest.approx(290 / 300)
    empty = pair_window(a, b, 60.0, 90.0)  # past both masks
    assert math.isnan(empty["window_mutual_silence_frac"]) and empty["window_turn_switches"] is None


def test_pair_interval_balance_rate_and_nan_semantics() -> None:
    a = np.zeros(1200, bool)  # 2 minutes
    b = np.zeros(1200, bool)
    for k in range(0, 1200, 100):  # alternate every 5 s: A 0-5, B 5-10, ...
        a[k:k + 50] = True
        b[k + 50:k + 100] = True
    out = pair_interval(a, b, 120.0, 60.0, 30.0)
    assert out["interaction_speech_overlap_frac"] == 0.0
    assert out["interaction_mutual_silence_frac"] == 0.0
    assert out["interaction_speech_balance"] == pytest.approx(0.5)
    assert out["interaction_turn_switch_rate_per_min"] == pytest.approx(23 / 2.0)
    none = pair_interval(np.zeros(1200, bool), np.zeros(1200, bool), 120.0, 0.0, 0.0)
    assert math.isnan(none["interaction_speech_balance"])
    assert math.isnan(none["interaction_speech_overlap_frac"])
    assert none["interaction_mutual_silence_frac"] == 1.0
    one_sided = pair_interval(a, np.zeros(1200, bool), 120.0, 60.0, 0.0)
    assert one_sided["interaction_speech_balance"] == 0.0


def test_pair_interval_uses_only_the_shared_interval() -> None:
    a = np.ones(1000, bool)
    b = np.zeros(1200, bool)  # longer member; the extra 20 s are not shared
    b[1000:] = True
    out = pair_interval(a, b, 100.0, 100.0, 20.0)
    assert out["interaction_speech_overlap_frac"] == 0.0
    assert out["interaction_mutual_silence_frac"] == 0.0


def test_pair_measures_from_tracks_on_different_frame_rates() -> None:
    ann_a = annotation(vad=[(0.0, 10.0)])
    ann_b = annotation(vad=[(10.0, 20.0)])
    ta = time_mask(speech_track(ann_a, 900, 30.0), 10.0)
    tb = time_mask(speech_track(ann_b, 899, 29.97), 10.0)
    out = pair_window(ta, tb, 0.0, 30.0)
    assert out["window_speech_overlap_frac"] == 0.0
    assert out["window_turn_switches"] == 1
    assert out["window_mutual_silence_frac"] == pytest.approx(1 / 3, abs=0.01)


def test_track_from_ticks_helper_round_trips() -> None:
    on = np.array([0, 1, 1, 0, 1, 0, 0, 1], bool)
    assert np.array_equal(time_mask(track_from_ticks(on), 10.0), on)


# ----------------------------------------------------------------------------- echo
def _words(tokens, t0=0.0, step=0.5):
    return [(t0 + i * step, t0 + i * step + 0.3, tok) for i, tok in enumerate(tokens)]


LONG = [f"token{chr(97 + i % 26)}{chr(97 + i // 26)}" for i in range(40)]  # 40 distinct 7-letter words


def test_echo_counts_same_word_within_tolerance() -> None:
    own = _words(LONG)
    partner = _words(LONG[:10], t0=0.2) + _words(LONG[10:20], t0=5.0 + 1.0)  # 0.2 s and 1.0 s offsets
    # first 10 within 0.3 s; the next 10 are 1.0 s late -> not echoes
    assert transcript_echo_frac(own, partner) == pytest.approx(10 / 40)


def test_echo_normalises_case_and_punctuation_and_requires_five_letters() -> None:
    own = [(i * 1.0, i * 1.0 + 0.2, "Today," if i % 2 == 0 else "yeah") for i in range(44)]
    partner = [(i * 1.0 + 0.1, i * 1.0 + 0.3, "today" if i % 2 == 0 else "yeah") for i in range(44)]
    # 22 eligible own words ('today'), all echoed; 'yeah' (4 letters) is never eligible
    assert transcript_echo_frac(own, partner) == 1.0


def test_echo_is_nan_below_twenty_eligible_words() -> None:
    own = _words(LONG[:19]) + _words(["yeah", "okay", "the"] * 10, t0=100.0)
    assert math.isnan(transcript_echo_frac(own, _words(LONG[:19])))
    assert not math.isnan(transcript_echo_frac(_words(LONG[:20]), []))
    assert transcript_echo_frac(_words(LONG[:20]), []) == 0.0


def test_echo_different_words_at_the_same_time_do_not_match() -> None:
    own = _words(LONG[:20])
    partner = _words(LONG[20:40])
    assert transcript_echo_frac(own, partner) == 0.0


def test_normalise_word() -> None:
    assert speech.normalise_word("Don't!") == "dont"
    assert speech.normalise_word("  HELLO  ") == "hello"


def test_a_vad_that_stops_early_is_extended_by_the_transcript_tail() -> None:
    from seamless_curation.speech import speech_track

    words = [{"word": "hello", "start": 60.0 + 0.5 * i, "end": 60.3 + 0.5 * i, "score": 0.9} for i in range(30)]
    annotation = {"metadata:vad": [{"start": 1.0, "end": 10.0}],
                  "metadata:transcript": [{"start": 1.0, "end": 80.0, "transcript": "", "words": words}]}
    track = speech_track(annotation, 90 * 30, 30.0)
    assert track.source == "vad+transcript_tail"
    assert track.mask[int(62 * 30)] and track.mask[int(5 * 30)] and not track.mask[int(30 * 30)]
    few = dict(annotation, **{"metadata:transcript": [{"start": 60, "end": 70, "transcript": "", "words": words[:10]}]})
    assert speech_track(few, 90 * 30, 30.0).source == "vad"

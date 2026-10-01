"""Audio level annotations on synthetic WAVs whose levels we set.

Every WAV is written with ``soundfile`` into the test's temporary directory, in
the release's format (32-bit FLOAT) but at low sample rates so the files stay
small. Tones whose period divides the 50 ms hop give every tick an exact RMS
(``A / sqrt(2)``), so the tests can state the level each measure must report.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from seamless_curation import audio, schema, speech
from seamless_curation.audio import (
    HOP_S,
    SILENT_FRAME_DB,
    AudioTrack,
    clip_audio,
    envelope_dynamics_db,
    equivalent_level_db,
    read_audio,
    recording_audio,
)

SR = 8000  # hop = 400 samples; a 100 Hz tone has exactly 5 periods per tick


def amplitude_for(rms_db: float) -> float:
    """Sine amplitude whose RMS is ``rms_db`` dBFS."""

    return math.sqrt(2.0) * 10.0 ** (rms_db / 20.0)


def tone(seconds: float, rms_db: float, sr: int = SR, freq: float = 100.0) -> np.ndarray:
    n = int(round(seconds * sr))
    return amplitude_for(rms_db) * np.sin(2.0 * np.pi * freq * np.arange(n) / sr)


def per_tick_tone(levels_db: np.ndarray, sr: int = SR) -> np.ndarray:
    """One exact tone level per 50 ms tick."""

    hop = int(round(sr * HOP_S))
    return np.concatenate([tone(hop / sr, level, sr) for level in levels_db])


def write_wav(path: Path, samples: np.ndarray, sr: int = SR) -> Path:
    sf.write(str(path), np.asarray(samples, dtype=np.float32), sr, subtype="FLOAT")
    return path


def mask_20hz(n: int, *intervals: tuple[float, float]) -> np.ndarray:
    out = np.zeros(n, dtype=bool)
    for start, end in intervals:
        out[int(round(start * 20)) : int(round(end * 20))] = True
    return out


def reference_levels(samples: np.ndarray, sr: int) -> np.ndarray:
    """Whole-array reference for the streamed envelope (tail tick kept)."""

    hop = int(round(sr * HOP_S))
    n_ticks = math.ceil(len(samples) / hop - 1e-9)
    out = []
    for i in range(n_ticks):
        chunk = np.asarray(samples[i * hop : (i + 1) * hop], dtype=np.float64)
        out.append(20 * np.log10(np.sqrt(np.mean(chunk**2)) + 1e-10))
    return np.array(out)


# ----------------------------------------------------------------------------- reading
def test_tone_level_is_exact_and_ticks_cover_the_file(tmp_path):
    track = read_audio(write_wav(tmp_path / "a.wav", tone(3.0, -20.0)))
    assert track.status == "ok" and track.sample_rate == SR
    assert track.duration_s == pytest.approx(3.0)
    assert track.level_db.shape == (60,) and track.level_db.dtype == np.float32
    assert np.allclose(track.level_db, -20.0, atol=1e-3)


def test_partial_tail_tick_is_kept_on_the_mask_grid(tmp_path):
    # 3.01 s -> ceil(60.2) = 61 ticks, like speech.time_mask's ceil(duration * 20)
    track = read_audio(write_wav(tmp_path / "a.wav", tone(3.01, -30.0)))
    assert len(track.level_db) == 61


def test_streamed_blocks_equal_the_whole_file_reference(tmp_path):
    # 150.02 s at 1 kHz: three read blocks (1200 + 1200 + 601 ticks), a 20-sample tail tick
    rng = np.random.default_rng(3)
    sr = 1000
    gain = np.repeat(10 ** rng.uniform(-4, -0.5, size=151), sr)[: 150_020]
    samples = (rng.standard_normal(150_020) * gain).astype(np.float32)
    track = read_audio(write_wav(tmp_path / "long.wav", samples, sr))
    expected = reference_levels(samples, sr)
    assert len(track.level_db) == len(expected) == 3001
    assert np.allclose(track.level_db, expected, atol=1e-3)


def test_non_integer_hop_does_not_drift(tmp_path):
    # 22,050 Hz: a 50 ms hop is 1102.5 samples; ticks still start at i / 20 s
    sr = 22050
    track = read_audio(write_wav(tmp_path / "a.wav", 0.1 * np.ones(sr * 10), sr))
    assert len(track.level_db) == 200
    assert np.allclose(track.level_db, 20 * np.log10(0.1), atol=1e-4)
    bounds = audio._tick_bounds(200, sr)
    assert bounds[100] == sr * 5 and bounds[-1] == sr * 10


def test_stereo_is_mixed_to_mono(tmp_path):
    left = tone(2.0, -20.0)
    track = read_audio(write_wav(tmp_path / "s.wav", np.stack([left, np.zeros_like(left)], axis=1)))
    assert np.allclose(track.level_db, -20.0 - 20 * math.log10(2.0), atol=1e-3)
    cancel = read_audio(write_wav(tmp_path / "c.wav", np.stack([left, -left], axis=1)))
    assert cancel.zero_frac == 1.0 and np.all(cancel.level_db == np.float32(SILENT_FRAME_DB))


def test_missing_empty_and_unreadable_files(tmp_path):
    missing = read_audio(tmp_path / "nope.wav")
    empty = read_audio(write_wav(tmp_path / "empty.wav", np.zeros(0)))
    assert (tmp_path / "empty.wav").stat().st_size <= audio.EMPTY_WAV_MAX_BYTES  # a header, like the 58-byte WAVs
    junk_path = tmp_path / "junk.wav"
    junk_path.write_bytes(np.random.default_rng(0).integers(0, 256, 4096, dtype=np.uint8).tobytes())
    junk = read_audio(junk_path)
    assert (missing.status, empty.status, junk.status) == ("missing", "empty", "unreadable")
    for track in (missing, empty, junk):
        assert math.isnan(track.duration_s) and math.isnan(track.zero_frac) and track.level_db.size == 0


# ----------------------------------------------------------------------------- recording measures
def test_dead_constant_signal_has_no_envelope_dynamics(tmp_path):
    rng = np.random.default_rng(1)
    hum = read_audio(write_wav(tmp_path / "hum.wav", tone(60.0, -23.0)))
    static = read_audio(write_wav(tmp_path / "static.wav", 0.05 * rng.standard_normal(60 * SR)))
    levels = np.where(rng.random(1200) < 0.3, -20.0, -65.0)  # a voice: bursts over a floor
    voice = read_audio(write_wav(tmp_path / "voice.wav", per_tick_tone(levels)))
    own = np.zeros(1200, dtype=bool)
    dead = [recording_audio(t, own, None)["audio_envelope_dynamics_db"] for t in (hum, static)]
    live = recording_audio(voice, own, None)["audio_envelope_dynamics_db"]
    assert max(dead) < 1.5  # v0 FM4 cut: dead tracks read 0.20-1.35 dB
    assert live == pytest.approx(45.0, abs=1e-3)


def test_dropout_zeros_are_counted_and_read_as_digital_silence(tmp_path):
    samples = np.concatenate([tone(20.0, -25.0), np.zeros(20 * SR)])
    track = read_audio(write_wav(tmp_path / "drop.wav", samples))
    rec = recording_audio(track, np.zeros(800, dtype=bool), None)
    assert rec["audio_exact_zero_frac"] == pytest.approx(0.5, abs=1e-3)
    assert np.all(track.level_db[400:] == np.float32(SILENT_FRAME_DB))
    assert rec["audio_duration_s"] == pytest.approx(40.0)


def conversation(tmp_path) -> AudioTrack:
    """0-20 s own voice (-20 dBFS), 20-40 s partner bleed (-40), 40-60 s floor (-80)."""

    samples = np.concatenate([tone(20.0, -20.0), tone(20.0, -40.0), tone(20.0, -80.0)])
    return read_audio(write_wav(tmp_path / "conv.wav", samples))


def test_level_gap_between_own_only_and_partner_only(tmp_path):
    track = conversation(tmp_path)
    own, partner = mask_20hz(1200, (0.0, 20.0)), mask_20hz(1200, (20.0, 40.0))
    rec = recording_audio(track, own, partner)
    assert rec["audio_status"] == "ok"
    assert rec["recording_own_speech_level_db"] == pytest.approx(-20.0, abs=1e-3)
    assert rec["recording_partner_only_level_db"] == pytest.approx(-40.0, abs=1e-3)
    assert rec["voice_isolation_db"] == pytest.approx(20.0, abs=1e-3)


def test_overlap_ticks_count_for_neither_side_of_the_isolation(tmp_path):
    track = conversation(tmp_path)
    own = mask_20hz(1200, (0.0, 20.0))
    partner = mask_20hz(1200, (10.0, 40.0))  # 10-20 s both speak: excluded from both levels
    rec = recording_audio(track, own, partner)
    assert rec["recording_partner_only_level_db"] == pytest.approx(-40.0, abs=1e-3)
    assert rec["voice_isolation_db"] == pytest.approx(20.0, abs=1e-3)
    assert rec["recording_own_speech_level_db"] == pytest.approx(-20.0, abs=1e-3)  # all own ticks


def test_partner_level_is_the_power_mean_not_the_floor(tmp_path):
    # Bleed-suppressed audio: during the partner's turns only a quarter of the ticks carry
    # bleed (-40 dBFS); the rest sit at the floor (-80). The median tick would report the
    # floor; the equivalent level reports what bleeds in.
    partner_ticks = np.where(np.arange(400) % 4 == 0, -40.0, -80.0)
    levels = np.concatenate([np.full(400, -20.0), partner_ticks, np.full(400, -80.0)])
    track = read_audio(write_wav(tmp_path / "gated.wav", per_tick_tone(levels)))
    own, partner = mask_20hz(1200, (0.0, 20.0)), mask_20hz(1200, (20.0, 40.0))
    rec = recording_audio(track, own, partner)
    expected = 10 * math.log10(0.25 * 1e-4 + 0.75 * 1e-8)
    assert rec["recording_partner_only_level_db"] == pytest.approx(expected, abs=1e-3)
    assert rec["voice_isolation_db"] == pytest.approx(-20.0 - expected, abs=1e-3)
    assert np.median(track.level_db[400:800]) == pytest.approx(-80.0, abs=1e-3)


def test_isolation_needs_five_seconds_each_side_and_a_partner(tmp_path):
    track = conversation(tmp_path)
    own = mask_20hz(1200, (0.0, 20.0))
    short = recording_audio(track, own, mask_20hz(1200, (20.0, 24.9)))
    assert math.isnan(short["recording_partner_only_level_db"]) and math.isnan(short["voice_isolation_db"])
    enough = recording_audio(track, own, mask_20hz(1200, (20.0, 25.0)))
    assert enough["voice_isolation_db"] == pytest.approx(20.0, abs=1e-3)
    own_short = recording_audio(track, mask_20hz(1200, (0.0, 4.0)), mask_20hz(1200, (20.0, 40.0)))
    assert own_short["recording_own_speech_level_db"] == pytest.approx(-20.0, abs=1e-3)  # >= 1 s
    assert math.isnan(own_short["voice_isolation_db"])  # own-only < 5 s
    alone = recording_audio(track, own, None)
    assert math.isnan(alone["recording_partner_only_level_db"]) and math.isnan(alone["voice_isolation_db"])
    assert alone["recording_own_speech_level_db"] == pytest.approx(-20.0, abs=1e-3)
    silent = recording_audio(track, mask_20hz(1200, (0.0, 0.95)), None)
    assert math.isnan(silent["recording_own_speech_level_db"])  # < 1 s own speech


def test_masks_align_by_index_and_truncate_to_the_shorter(tmp_path):
    track = conversation(tmp_path)  # 1200 ticks
    long_own = mask_20hz(1500, (0.0, 20.0), (61.0, 75.0))  # the pose grid runs past the WAV
    rec = recording_audio(track, long_own, mask_20hz(1100, (20.0, 40.0)))
    assert rec["recording_own_speech_level_db"] == pytest.approx(-20.0, abs=1e-3)
    assert rec["voice_isolation_db"] == pytest.approx(20.0, abs=1e-3)


def test_speech_time_mask_lines_up_with_the_envelope(tmp_path):
    track = conversation(tmp_path)
    annotation = {"metadata:vad": [{"start": 0.0, "end": 20.0}], "metadata:transcript": []}
    partner_annotation = {"metadata:vad": [{"start": 20.0, "end": 40.0}], "metadata:transcript": []}
    own = speech.time_mask(speech.speech_track(annotation, 1800, 30.0), 20.0)
    partner = speech.time_mask(speech.speech_track(partner_annotation, 1800, 30.0), 20.0)
    rec = recording_audio(track, own, partner)
    assert rec["voice_isolation_db"] == pytest.approx(20.0, abs=1e-3)


def test_not_ok_tracks_give_status_and_na_measures(tmp_path):
    own = np.ones(100, dtype=bool)
    for track in (read_audio(tmp_path / "nope.wav"), read_audio(write_wav(tmp_path / "e.wav", np.zeros(0))), None):
        rec = recording_audio(track, own, own)
        assert rec["audio_status"] in ("missing", "empty")
        assert all(math.isnan(v) for k, v in rec.items() if k != "audio_status")
        clip = clip_audio(track, own, 0.0, 30.0)
        assert all(math.isnan(v) for v in clip.values())


# ----------------------------------------------------------------------------- clip measures
def test_clip_levels_and_vocal_range(tmp_path):
    # 0-10 s own speech alternating -20/-30 dBFS per tick, then 10-30 s at -60
    levels = np.concatenate([np.tile([-20.0, -30.0], 100), np.full(400, -60.0)])
    track = read_audio(write_wav(tmp_path / "v.wav", per_tick_tone(levels)))
    own = mask_20hz(600, (0.0, 10.0))
    clip = clip_audio(track, own, 0.0, 30.0)
    assert clip["audio_rms_db_p50"] == pytest.approx(-60.0, abs=1e-3)
    assert clip["own_speech_level_db"] == pytest.approx(-25.0, abs=1e-3)
    assert clip["vocal_level_range_db"] == pytest.approx(10.0, abs=1e-3)
    later = clip_audio(track, own, 10.0, 30.0)
    assert later["audio_rms_db_p50"] == pytest.approx(-60.0, abs=1e-3)
    assert math.isnan(later["own_speech_level_db"]) and math.isnan(later["vocal_level_range_db"])


def test_clip_speech_minimums(tmp_path):
    track = read_audio(write_wav(tmp_path / "v.wav", tone(30.0, -30.0)))
    for seconds, has_level, has_range in ((0.9, False, False), (1.0, True, False), (2.9, True, False), (3.0, True, True)):
        clip = clip_audio(track, mask_20hz(600, (5.0, 5.0 + seconds)), 0.0, 30.0)
        assert math.isfinite(clip["own_speech_level_db"]) == has_level, seconds
        assert math.isfinite(clip["vocal_level_range_db"]) == has_range, seconds


def test_clip_past_the_end_of_a_short_wav(tmp_path):
    track = read_audio(write_wav(tmp_path / "short.wav", tone(60.0, -30.0)))
    own = np.ones(2000, dtype=bool)
    beyond = clip_audio(track, own, 60.0, 90.0)
    assert all(math.isnan(v) for v in beyond.values())
    sliver = clip_audio(track, own, 59.5, 89.5)  # 0.5 s of audio in the clip
    assert math.isnan(sliver["audio_rms_db_p50"])
    most = clip_audio(track, own, 58.0, 88.0)  # 2 s of audio
    assert most["audio_rms_db_p50"] == pytest.approx(-30.0, abs=1e-3)
    assert most["own_speech_level_db"] == pytest.approx(-30.0, abs=1e-3)


def test_clip_ticks_use_the_mask_rounding(tmp_path):
    levels = np.concatenate([np.full(20, -20.0), np.full(20, -40.0)])  # 1 s / 1 s
    track = read_audio(write_wav(tmp_path / "r.wav", per_tick_tone(levels)))
    # [0.51, 1.49): ticks round(10.2)=10 .. round(29.8)=30, ten of each level -> median -30.
    # Flooring both ends (10 .. 29) would give -20, ceiling both (11 .. 30) -40.
    clip = clip_audio(track, np.ones(40, dtype=bool), 0.51, 1.49)
    assert clip["audio_rms_db_p50"] == pytest.approx(-30.0, abs=1e-3)


# ----------------------------------------------------------------------------- helpers and registry
def test_statistic_helpers():
    assert equivalent_level_db(np.array([-20.0, -20.0])) == pytest.approx(-20.0)
    assert equivalent_level_db(np.array([-20.0, -200.0])) == pytest.approx(-20.0 - 10 * math.log10(2), abs=1e-6)
    assert math.isnan(equivalent_level_db(np.array([])))
    assert envelope_dynamics_db(np.linspace(0.0, 100.0, 101)) == pytest.approx(45.0)
    assert math.isnan(envelope_dynamics_db(np.array([])))


def test_output_keys_are_registered_at_their_level(tmp_path):
    track = conversation(tmp_path)
    own = mask_20hz(1200, (0.0, 20.0))
    for key in recording_audio(track, own, own):
        assert schema.field("recordings", key).level == "recording"
    for key in clip_audio(track, own, 0.0, 30.0):
        field = schema.field("clips", key)
        assert field.dtype == "float32" and field.nullable


def test_the_own_speech_level_is_not_cut_to_a_shorter_partner(tmp_path) -> None:
    """Regression: the pair step aligned the envelope to the partner's mask, so a
    recording whose partner was shorter had its own-speech level measured over the
    partner's duration only (NaN when that held < 1 s of own speech)."""

    import numpy as np
    import soundfile

    from seamless_curation import audio

    rate = 16_000
    t = np.arange(60 * rate) / rate
    signal = (0.1 * np.sin(2 * np.pi * 200 * t) * (t >= 40)).astype(np.float32) + 1e-4
    path = tmp_path / "own.wav"
    soundfile.write(str(path), signal, rate, subtype="FLOAT")
    track = audio.read_audio(path)
    own = np.zeros(1200, bool)
    own[800:1200] = True                       # speaks 40-60 s
    partner = np.zeros(400, bool)              # partner's recording is 20 s long
    partner[:200] = True
    alone = audio.recording_audio(track, own, None)
    paired = audio.recording_audio(track, own, partner)
    assert np.isfinite(alone["recording_own_speech_level_db"])
    assert paired["recording_own_speech_level_db"] == alone["recording_own_speech_level_db"]

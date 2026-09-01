#!/usr/bin/env python3
"""Measure what makes the reviewer-flagged audio sound broken.

Twenty-one clips were reported as glitchy and unusable, and two more as silent.
Three hypotheses were tested and two survived.

**Dead audio.** Two of the flagged clips, and both of the "silent" ones, carry a
speech level of -63 to -88 dB against a corpus median near -23, with a *flat*
long-term spectrum -- a noise floor, not a voice. That is a clean detection.

**A shared or far-field microphone.** The rest sound wrong because they are not
close-mic recordings. Comparing a file's own energy during its own speech against
its energy during the *partner's* speech, using the released VAD, a lavalier
gives 9 to 25 dB of separation and these give 1 to 9. Both speakers arrive at
equal level, with room reverb and automatic gain riding over the top, which is
what the reviewer heard as glitchy. It also explains why the flagged files
cluster in V03's room-camera rasters.

**Fragmentation did not survive.** The flagged waveforms look chopped, but so
does every dyadic recording -- each participant is silent about half the time --
and the burst statistics of flagged and unflagged files are identical (median
burst 0.17 s either way). Recorded here because it is the obvious hypothesis and
it is wrong.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

FRAME_S = 0.010
# A frame is "dead" at 45 dB below the file's own speech level. Real speech
# between syllables rarely falls that far; a gate or a lost packet does.
DEAD_DB = -45.0
SPEECH_PERCENTILE = 95.0


def audio_quality(path: Path) -> dict[str, float | str]:
    try:
        samples, rate = sf.read(str(path), dtype="float32", always_2d=False)
    except (RuntimeError, OSError) as error:
        return {"audio_status": f"read_error:{type(error).__name__}"}
    signal = np.asarray(samples, dtype=np.float64).reshape(-1)
    if signal.size == 0:
        return {"audio_status": "empty"}

    hop = max(1, int(FRAME_S * rate))
    frames = signal[: (len(signal) // hop) * hop].reshape(-1, hop)
    if len(frames) < 10:
        return {"audio_status": "too_short"}
    energy = np.sqrt((frames ** 2).mean(axis=1))
    speech_level = float(np.percentile(energy, SPEECH_PERCENTILE))

    record: dict[str, float | str] = {
        "audio_status": "ok",
        "audio_seconds": float(len(signal) / rate),
        "audio_sample_rate": float(rate),
        "audio_speech_level_db": float(20 * np.log10(speech_level + 1e-12)),
        "audio_peak": float(np.abs(signal).max()),
        "audio_clip_frac": float((np.abs(signal) >= 0.999).mean()),
    }

    # A recording with no speech in it at all: the 95th-percentile frame is
    # already at the noise floor, and the spectrum is flat rather than shaped
    # like a voice. Both of the reviewer's "silent" files land here.
    spectrum = np.abs(np.fft.rfft(frames[: min(len(frames), 4000)] * np.hanning(hop), axis=-1)) ** 2
    mean_spectrum = spectrum.mean(axis=0) + 1e-20
    flatness = float(np.exp(np.log(mean_spectrum).mean()) / mean_spectrum.mean())
    record["audio_spectral_flatness"] = flatness

    # Round 14: the measure that actually works. Static and buzz have no
    # dynamics -- the level is the same whether anyone is speaking or not --
    # while any track carrying a voice swings. Over 44 hand-labelled files the
    # two classes do not overlap: static 0.20-1.35 dB, usable 1.70-69.04 dB.
    # It needs no VAD, no partner and no absolute level, which is why it works
    # where three previous attempts did not.
    envelope = 20 * np.log10(energy + 1e-12)
    record["audio_envelope_dynamics_db"] = float(
        np.percentile(envelope, 95) - np.percentile(envelope, 50)
    )

    if speech_level <= 1e-6:
        record.update({"audio_dead_frac": 1.0, "audio_gate_rate_per_s": 0.0,
                       "audio_median_burst_s": 0.0, "audio_live_frac": 0.0})
        return record

    live = energy > speech_level * (10 ** (DEAD_DB / 20))
    record["audio_live_frac"] = float(live.mean())
    record["audio_dead_frac"] = float(1.0 - live.mean())

    # Transitions into speech per second of speech. A gated or lossy recording
    # re-enters speech many times a second; continuous talking does not.
    onsets = int(np.count_nonzero(live[1:] & ~live[:-1]))
    live_seconds = float(live.sum()) * FRAME_S
    record["audio_gate_rate_per_s"] = onsets / live_seconds if live_seconds > 0 else 0.0

    changes = np.flatnonzero(np.diff(live.astype(np.int8)))
    bounds = np.concatenate(([0], changes + 1, [len(live)]))
    runs = np.diff(bounds)
    live_runs = runs[live[bounds[:-1]]]
    record["audio_median_burst_s"] = float(np.median(live_runs) * FRAME_S) if len(live_runs) else 0.0
    return record


def vad_intervals(json_path: Path) -> list[tuple[float, float]]:
    """Released speech segments, whatever key the bundle files them under."""
    try:
        payload = json.loads(json_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return []

    def walk(node):
        if isinstance(node, dict):
            for key, value in node.items():
                if "vad" in key.lower() and isinstance(value, list):
                    yield value
                else:
                    yield from walk(value)
        elif isinstance(node, list):
            for value in node:
                yield from walk(value)

    for candidate in walk(payload):
        segments = []
        for entry in candidate:
            if isinstance(entry, (list, tuple)) and len(entry) >= 2:
                segments.append((float(entry[0]), float(entry[1])))
            elif isinstance(entry, dict) and "start" in entry and "end" in entry:
                segments.append((float(entry["start"]), float(entry["end"])))
        if segments:
            return segments
    return []


def voice_isolation_db(
    wav_path: Path, own_vad: list[tuple[float, float]], partner_vad: list[tuple[float, float]]
) -> float:
    """How much louder this track is during its own speech than during the partner's.

    A close-worn microphone hears its wearer far better than the other person; a
    room camera hears both equally. This is the measurement that separates the
    reviewer's glitchy clips from good ones, and it needs no assumption about
    what a "good" absolute level is.
    """
    if not own_vad or not partner_vad:
        return float("nan")
    try:
        samples, rate = sf.read(str(wav_path), dtype="float32", always_2d=False)
    except (RuntimeError, OSError):
        return float("nan")
    signal = np.asarray(samples, dtype=np.float64).reshape(-1)
    if signal.size == 0:
        return float("nan")

    def level(intervals: list[tuple[float, float]]) -> float:
        mask = np.zeros(len(signal), dtype=bool)
        for start, end in intervals:
            mask[max(0, int(start * rate)):min(len(signal), int(end * rate))] = True
        if not mask.any():
            return float("nan")
        return float(np.sqrt((signal[mask] ** 2).mean()))

    own, partner = level(own_vad), level(partner_vad)
    if not np.isfinite(own) or not np.isfinite(partner):
        return float("nan")
    return float(20 * np.log10((own + 1e-12) / (partner + 1e-12)))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", type=int, default=0)
    parser.add_argument("--tasks", type=int, default=1)
    parser.add_argument("--per-vendor", type=int, default=250)
    parser.add_argument("--seed", type=int, default=20261101)
    parser.add_argument("--file-list", type=Path,
                        help="newline-separated file_ids to include whatever the draw says")
    parser.add_argument("--out", type=Path, default=ROOT / "outputs/session3/audio_quality")
    args = parser.parse_args()

    inventory = pd.read_parquet(ROOT / "outputs/02_inventory/summary/inventory_joined.parquet")
    inventory["raster"] = (inventory.video_width.astype("Int64").astype(str) + "x"
                           + inventory.video_height.astype("Int64").astype(str))
    eligible = inventory.loc[
        inventory.all_modalities_present.astype(bool)
        & inventory.probe_status.eq("ok")
        & inventory.video_stream_present.astype(bool)
        & inventory.observed_duration_s.ge(30.0)
        & inventory.interaction_type.ne("charades")
    ]
    parts = [
        eligible.groupby("vendor_id", group_keys=False).apply(
            lambda g: g.sample(n=min(len(g), args.per_vendor), random_state=args.seed)
        ),
        # Every file of the odd rasters: they are where the reports came from and
        # they are too rare for a random draw to reach.
        eligible.loc[eligible.raster.isin(
            ["3840x2160", "640x480", "1080x960", "2180x3840", "1012x1920", "1920x1080"]
        )],
    ]
    if args.file_list and args.file_list.exists():
        wanted = [line.strip() for line in args.file_list.read_text().splitlines() if line.strip()]
        parts.append(eligible.loc[eligible.file_id.isin(wanted)])
    pool = (pd.concat(parts).drop_duplicates(subset="file_id")
            .sort_values("file_id").iloc[args.task::args.tasks])

    partners: dict[str, str] = {}
    for _, group in eligible.groupby(["vendor_id", "session_id", "interaction_id"], sort=False):
        if len(group) == 2:
            first, second = list(group.source_relbase)
            ids = list(group.file_id)
            partners[ids[0]] = second
            partners[ids[1]] = first

    rows = []
    for row in pool.itertuples(index=False):
        base = ROOT / "seamless_interaction" / row.source_relbase
        partner_relbase = partners.get(row.file_id)
        isolation = float("nan")
        if partner_relbase:
            isolation = voice_isolation_db(
                base.with_suffix(".wav"),
                vad_intervals(base.with_suffix(".json")),
                vad_intervals((ROOT / "seamless_interaction" / partner_relbase).with_suffix(".json")),
            )
        rows.append({
            "file_id": row.file_id, "vendor_id": row.vendor_id, "raster": row.raster,
            "video_seconds": float(row.observed_duration_s),
            "wav_bytes": int(row.wav_size_bytes) if pd.notna(row.wav_size_bytes) else -1,
            "audio_voice_isolation_db": isolation,
            **audio_quality(base.with_suffix(".wav")),
        })

    args.out.mkdir(parents=True, exist_ok=True)
    destination = args.out / f"part_{args.task:03d}.parquet"
    pd.DataFrame(rows).to_parquet(destination, index=False)
    print(f"task {args.task}: {len(rows)} files -> {destination}")


if __name__ == "__main__":
    main()

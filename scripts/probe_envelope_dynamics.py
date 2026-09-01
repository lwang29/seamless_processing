#!/usr/bin/env python3
"""Envelope dynamic range per file, over an existing scan's population.

Static and buzz sit at one level whether anyone is speaking or not; a track
carrying a voice swings. Over 48 hand-labelled files the two classes do not
overlap -- unusable 0.20-1.35 dB, usable 1.70-69.04 -- which is what FM4 now
gates on. This measures it for scans that already exist so the corpus rate can be
read without a full rescan.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]


def envelope_dynamics_db(wav_path: Path) -> tuple[float, str]:
    try:
        samples, rate = sf.read(str(wav_path), dtype="float32", always_2d=False)
    except (RuntimeError, OSError):
        return float("nan"), "read_error"
    signal = np.asarray(samples, dtype=np.float64).reshape(-1)
    if signal.size < rate:
        return float("nan"), "empty"
    hop = max(1, int(0.05 * rate))
    frames = signal[: (len(signal) // hop) * hop].reshape(-1, hop)
    envelope = 20 * np.log10(np.sqrt((frames ** 2).mean(axis=1)) + 1e-12)
    return float(np.percentile(envelope, 95) - np.percentile(envelope, 50)), "ok"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scan", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--task", type=int, default=0)
    parser.add_argument("--tasks", type=int, default=1)
    args = parser.parse_args()
    scan = pd.read_parquet(args.scan)
    rows = []
    for row in scan.iloc[args.task::args.tasks].itertuples(index=False):
        value, status = envelope_dynamics_db(
            ROOT / "seamless_interaction" / f"{row.source_relbase}.wav"
        )
        rows.append({"file_id": row.file_id, "audio_envelope_dynamics_db": value,
                     "dynamics_status": status})
    args.out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(args.out.with_suffix(f".{args.task:03d}.parquet"), index=False)
    print(f"task {args.task}: {len(rows)}")


if __name__ == "__main__":
    main()

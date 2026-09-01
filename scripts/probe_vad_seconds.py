#!/usr/bin/env python3
"""Seconds of released speech per file, over an existing scan's population.

The Round-12 answers showed FM4's absolute level floor cannot tell "speaks
rarely" from "no voice at all" -- all six of its V00 firings were the former.
The released VAD can: every confirmed-dead file has zero seconds of it, five of
the six V00 false positives have 1.6 to 79.1, and an empty VAD occurs in 0 of 400
random V00 files.

This measures it over scans that already exist, so the candidate rule can be
sampled and put to a reviewer without re-reading any audio. It reads only the
JSON sidecar.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from scripts.probe_audio_quality import vad_intervals  # noqa: E402


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
        intervals = vad_intervals(ROOT / "seamless_interaction" / f"{row.source_relbase}.json")
        rows.append({"file_id": row.file_id,
                     "audio_vad_seconds": float(sum(b - a for a, b in intervals))})
    args.out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(args.out.with_suffix(f".{args.task:03d}.parquet"), index=False)
    print(f"task {args.task}: {len(rows)} files")


if __name__ == "__main__":
    main()

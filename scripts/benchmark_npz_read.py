#!/usr/bin/env python3
"""Measure NPZ+JSON read throughput, which is what bounds Stage 1.

M-1's benchmark measures ``ffprobe`` header probes and therefore says nothing
about payload throughput. Stage 1 reads whole NPZ members (DEFLATE-compressed,
so they cannot be memory-mapped) plus the small JSON, and that is the cost this
measures.

Worker sets are disjoint and chosen deterministically, so a warm cache from one
setting cannot inflate the next. Caches are not flushed — that needs privileges
we do not have — so this is "cold-ish" in the same sense as M-1's benchmark.
"""

from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


# Exactly the arrays a Stage-1 signal needs: validity masks, released 2D
# keypoints and boxes, and SMPL-H pose. No MP4 and no WAV.
STAGE1_KEYS = (
    "boxes_and_keypoints:keypoints",
    "boxes_and_keypoints:box",
    "boxes_and_keypoints:is_valid_box",
    "smplh:is_valid",
    "smplh:global_orient",
    "smplh:body_pose",
    "smplh:left_hand_pose",
    "smplh:right_hand_pose",
    "smplh:translation",
)


def _read_one(source_root: Path, relbase: str) -> tuple[int, int, str]:
    """Return (npz bytes read, json bytes read, status)."""

    base = source_root / relbase
    npz_path = base.with_suffix(".npz")
    json_path = base.with_suffix(".json")
    try:
        total = 0
        with np.load(npz_path, allow_pickle=False) as archive:
            for key in STAGE1_KEYS:
                if key in archive.files:
                    total += int(np.asarray(archive[key]).nbytes)
            if "movement:is_valid" in archive.files:
                total += int(np.asarray(archive["movement:is_valid"]).nbytes)
        payload = json_path.read_bytes()
        json.loads(payload)
        return npz_path.stat().st_size, len(payload), "ok"
    except Exception as exc:
        return 0, 0, f"{type(exc).__name__}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=Path("seamless_interaction"))
    parser.add_argument(
        "--inventory",
        type=Path,
        default=Path("outputs/02_inventory/summary/inventory_joined.parquet"),
    )
    parser.add_argument("--out", type=Path, default=Path("outputs/02_inventory/benchmark/npz_read_scaling.json"))
    parser.add_argument("--samples", type=int, default=96)
    parser.add_argument("--workers", type=int, nargs="+", default=[1, 4, 8, 16])
    parser.add_argument("--seed", type=int, default=20260805)
    args = parser.parse_args()

    frame = pd.read_parquet(args.inventory)
    population = frame.loc[
        frame.all_modalities_present.astype(bool) & frame.probe_status.eq("ok")
    ].copy()
    ordered = population.sample(frac=1.0, random_state=args.seed).reset_index(drop=True)

    results: list[dict[str, Any]] = []
    offset = 0
    for workers in args.workers:
        subset = ordered.iloc[offset : offset + args.samples]
        offset += args.samples
        relbases = list(subset.source_relbase)
        start = time.perf_counter()
        with ThreadPoolExecutor(max_workers=workers) as pool:
            outcomes = list(pool.map(lambda rel: _read_one(args.source_root, rel), relbases))
        wall = time.perf_counter() - start
        npz_bytes = sum(item[0] for item in outcomes)
        json_bytes = sum(item[1] for item in outcomes)
        statuses: dict[str, int] = {}
        for item in outcomes:
            statuses[item[2]] = statuses.get(item[2], 0) + 1
        results.append(
            {
                "workers": workers,
                "files": len(relbases),
                "wall_s": wall,
                "files_per_s": len(relbases) / wall,
                "npz_bytes_on_disk": npz_bytes,
                "json_bytes": json_bytes,
                "mib_per_s": (npz_bytes + json_bytes) / wall / 2**20,
                "status_counts": statuses,
                "sample_start": offset - args.samples,
            }
        )
        print(json.dumps(results[-1], sort_keys=True))

    payload = {
        "measurement": "whole-NPZ Stage-1 array read plus JSON parse; no MP4, no WAV",
        "cache_condition": "cold-ish: disjoint deterministic file sets per worker setting",
        "seed": args.seed,
        "stage1_keys": list(STAGE1_KEYS),
        "results": results,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


if __name__ == "__main__":
    main()

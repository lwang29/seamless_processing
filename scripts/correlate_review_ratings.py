#!/usr/bin/env python3
"""Correlate exported human review ratings against every computed clip signal.

Run this once the reviewers have exported Pass-2 ratings. It answers three
questions and writes each one out separately:

1. For every rating item, which signals separate high from low ratings?
2. Which rating items does **no** signal separate? That list is the negative
   result the session is meant to be able to state plainly: those failure modes
   need a purpose-built detector because nothing we currently compute sees them.
3. How consistent were the reviewers on the blinded duplicate items? A strong
   correlation against a rating item nobody can reproduce is not a finding.

The script applies no threshold to the data and selects nothing.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from seamless_curation.review_ratings import (  # noqa: E402
    SEPARATION_FLOOR,
    correlate_ratings_with_signals,
    duplicate_agreement,
    load_rating_exports,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ratings", type=Path, nargs="+", required=True,
                        help="one or more exported rating JSON files")
    parser.add_argument("--signals", type=Path,
                        default=Path("outputs/session2/review_clip_signals/clip_signals.parquet"))
    parser.add_argument("--rubric", type=Path, required=True,
                        help="the approved rubric the ratings were collected with")
    parser.add_argument("--duplicate-map", type=Path,
                        default=Path("outputs/session2/review_sampling/pass2_duplicate_map.json"))
    parser.add_argument("--out", type=Path,
                        default=Path("outputs/session2/review_rating_analysis"))
    parser.add_argument("--separation-floor", type=float, default=SEPARATION_FLOOR)
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    rubric = yaml.safe_load(args.rubric.read_text(encoding="utf-8"))
    rating_items = [str(item["id"]) for item in rubric.get("items") or []]
    overall = str((rubric.get("overall") or {}).get("id", "would_train"))

    ratings = load_rating_exports(args.ratings)
    signals = pd.read_parquet(args.signals)
    result = correlate_ratings_with_signals(
        ratings, signals, rating_items=rating_items,
        separation_floor=args.separation_floor, overall_column=overall,
    )
    result["correlations"].to_csv(args.out / "rating_signal_correlations.csv", index=False)
    result["would_train_separation"].to_csv(args.out / "overall_signal_separation.csv", index=False)

    agreement: dict[str, Any] = {}
    if args.duplicate_map.is_file():
        duplicate_map = json.loads(args.duplicate_map.read_text(encoding="utf-8"))
        # Averaging first would hide disagreement, so agreement is measured on
        # the raw per-item rows before any deduplication.
        computed = duplicate_agreement(
            ratings, duplicate_map, rating_items=rating_items, overall_column=overall
        )
        computed["scales"].to_csv(args.out / "duplicate_agreement_scales.csv", index=False)
        agreement = {
            "scales": computed["scales"].to_dict(orient="records"),
            overall: computed["would_train"],
        }

    summary = {
        "rating_files": [str(path) for path in args.ratings],
        "reviewers": sorted(ratings.reviewer_id.dropna().unique().tolist()),
        "rating_items": rating_items,
        "overall_item": overall,
        "clips_analyzed": result["clips_analyzed"],
        "signals_considered": result["signals_considered"],
        "separation_floor": result["separation_floor"],
        "best_signal_per_item": result["best_signal_per_item"],
        "items_without_a_separating_signal": result["items_without_a_separating_signal"],
        "rating_items_absent_from_export": result["rating_items_absent_from_export"],
        "duplicate_agreement": agreement,
    }
    (args.out / "analysis_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True, default=str), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True, default=str))


if __name__ == "__main__":
    main()

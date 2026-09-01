#!/usr/bin/env python3
"""Compare V00 naturalistic against V00 improvised on the reviewer's vocabulary.

This is the quantitative half of the naturalistic-versus-improvised decision. The
human review is the other half, and neither replaces the other: these proxies say
how often a measurable correlate of each vocabulary term occurs in each label,
while only a human can say whether the clips it flags are actually unusable.

Reported per label: the distribution of every proxy, a rank-based effect size for
the difference, and the tail counts that matter for curation. Effect sizes use
the rank-biserial form derived from Mann-Whitney U, which needs no distributional
assumption and is directly readable as "probability that a random naturalistic
file scores higher than a random improvised one".
"""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


# Proxy columns to compare, grouped by the reviewer's own vocabulary term, with
# the direction that means "worse" so the report can speak plainly.
COMPARISONS: list[tuple[str, str, str]] = [
    ("framing", "framing_person_height_frac_p50", "lower_is_worse"),
    ("framing", "framing_min_edge_margin_frac_p50", "lower_is_worse"),
    ("framing", "framing_frames_touching_edge_frac", "higher_is_worse"),
    ("framing", "framing_body_out_of_frame_point_frac", "higher_is_worse"),
    ("framing", "framing_centre_offset_x_frac_p50", "magnitude_is_worse"),
    ("roll", "roll_shoulder_abs_deg_p50", "higher_is_worse"),
    ("roll", "roll_shoulder_deg_iqr", "higher_is_worse"),
    ("sitting", "posture_leg_over_torso_p50", "lower_is_worse"),
    ("sitting", "posture_knee_drop_over_torso_p50", "lower_is_worse"),
    ("sitting", "posture_ankles_usable_frac", "lower_is_worse"),
    ("static_hands", "hand_activity_static_frac", "higher_is_worse"),
    ("static_hands", "hand_activity_left_speed_sw_per_s_p50", "lower_is_worse"),
    ("static_hands", "hand_activity_right_speed_sw_per_s_p50", "lower_is_worse"),
    ("static_hands", "hand_activity_left_above_hip_frac", "lower_is_worse"),
    ("static_hands", "hand_activity_right_above_hip_frac", "lower_is_worse"),
    ("speech", "file_speaking_frac", "lower_is_worse"),
    ("tracking", "valid_frac_smplh", "lower_is_worse"),
    ("tracking", "valid_frac_smplh_and_box", "lower_is_worse"),
    ("tracking", "valid_frac_movement", "lower_is_worse"),
    ("desync_proxy_failed", "best_av_lag_peak_r", "not_interpretable"),
]
PERCENTILES = [1, 5, 25, 50, 75, 95, 99]


def _rank_biserial(first: np.ndarray, second: np.ndarray) -> tuple[float, float]:
    """(P(first > second) with ties at half, rank-biserial correlation)."""

    if first.size == 0 or second.size == 0:
        return float("nan"), float("nan")
    combined = np.concatenate([first, second])
    ranks = pd.Series(combined).rank().to_numpy()
    n1, n2 = float(first.size), float(second.size)
    u = ranks[: first.size].sum() - n1 * (n1 + 1) / 2
    probability = u / (n1 * n2)
    return float(probability), float(2 * probability - 1)


def describe(values: pd.Series) -> dict[str, Any]:
    numeric = pd.to_numeric(values, errors="coerce").dropna()
    row: dict[str, Any] = {"n": int(numeric.size)}
    if numeric.empty:
        return row | {f"p{p}": float("nan") for p in PERCENTILES}
    for percentile in PERCENTILES:
        row[f"p{percentile}"] = float(numeric.quantile(percentile / 100))
    row["mean"] = float(numeric.mean())
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--shards", type=Path, default=Path("outputs/session2/v00_pool_signals/shards")
    )
    parser.add_argument("--out", type=Path, default=Path("outputs/session2/v00_pool_signals"))
    args = parser.parse_args()

    paths = sorted(glob.glob(str(args.shards / "*.parquet")))
    if not paths:
        raise SystemExit(f"no shards under {args.shards}")
    pool = pd.concat([pd.read_parquet(path) for path in paths], ignore_index=True)
    args.out.mkdir(parents=True, exist_ok=True)
    pool.to_parquet(args.out / "pool_signals.parquet", index=False)

    ok = pool.loc[pool.status.eq("ok")].copy()
    natural = ok.loc[ok.label.eq("naturalistic")]
    improvised = ok.loc[ok.label.eq("improvised")]

    rows: list[dict[str, Any]] = []
    for term, column, direction in COMPARISONS:
        if column not in ok.columns:
            continue
        first = pd.to_numeric(natural[column], errors="coerce").dropna().to_numpy()
        second = pd.to_numeric(improvised[column], errors="coerce").dropna().to_numpy()
        probability, effect = _rank_biserial(first, second)
        record: dict[str, Any] = {
            "vocabulary_term": term,
            "column": column,
            "worse_direction": direction,
            "n_naturalistic": int(first.size),
            "n_improvised": int(second.size),
            "p_naturalistic_higher": probability,
            "rank_biserial": effect,
        }
        for name, values in (("naturalistic", natural[column]), ("improvised", improvised[column])):
            summary = describe(values)
            for key in ("p5", "p50", "p95"):
                record[f"{name}_{key}"] = summary.get(key)
        rows.append(record)
    comparison = pd.DataFrame(rows)
    comparison.to_csv(args.out / "label_comparison.csv", index=False)

    per_label_distributions: list[pd.DataFrame] = []
    for label, subset in (("naturalistic", natural), ("improvised", improvised), ("all", ok)):
        records = []
        for _, column, _ in COMPARISONS:
            if column not in subset.columns:
                continue
            records.append({"label": label, "column": column, **describe(subset[column])})
        per_label_distributions.append(pd.DataFrame(records))
    pd.concat(per_label_distributions, ignore_index=True).to_csv(
        args.out / "distributions_by_label.csv", index=False
    )

    # Tail counts. Cut points are stated in the output and are reporting
    # descriptions, not selection thresholds: they answer "how often does this
    # look bad" for two labels on the same ruler.
    tails = {
        "person_fills_under_40pct_of_frame_height": ("framing_person_height_frac_p50", "<", 0.40),
        "any_frame_touching_raster_edge_over_5pct": ("framing_frames_touching_edge_frac", ">", 0.05),
        "shoulder_tilt_over_10_deg": ("roll_shoulder_abs_deg_p50", ">", 10.0),
        "leg_to_torso_under_1_2_sitting_candidate": ("posture_leg_over_torso_p50", "<", 1.2),
        "ankles_tracked_under_50pct": ("posture_ankles_usable_frac", "<", 0.50),
        "hands_static_over_50pct_of_file": ("hand_activity_static_frac", ">", 0.50),
        "hands_static_over_80pct_of_file": ("hand_activity_static_frac", ">", 0.80),
        "left_hand_above_hips_under_5pct": ("hand_activity_left_above_hip_frac", "<", 0.05),
        "speaking_under_10pct_of_file": ("file_speaking_frac", "<", 0.10),
        "smplh_validity_under_90pct": ("valid_frac_smplh", "<", 0.90),
    }
    tail_rows: list[dict[str, Any]] = []
    for name, (column, operator, cut) in tails.items():
        if column not in ok.columns:
            continue
        record: dict[str, Any] = {"tail": name, "column": column, "cut": f"{operator} {cut}"}
        for label, subset in (("naturalistic", natural), ("improvised", improvised), ("all", ok)):
            series = pd.to_numeric(subset[column], errors="coerce")
            usable = series.notna()
            flagged = (series < cut) if operator == "<" else (series > cut)
            record[f"{label}_n"] = int(usable.sum())
            record[f"{label}_flagged"] = int((flagged & usable).sum())
            record[f"{label}_frac"] = (
                float((flagged & usable).sum() / usable.sum()) if usable.any() else float("nan")
            )
        tail_rows.append(record)
    tail_table = pd.DataFrame(tail_rows)
    tail_table.to_csv(args.out / "tail_counts_by_label.csv", index=False)

    summary = {
        "pool_rows": int(len(pool)),
        "pool_rows_ok": int(len(ok)),
        "status_counts": pool.status.value_counts().to_dict(),
        "by_label": ok.label.value_counts().to_dict(),
        "by_activity": ok.interaction_type.value_counts().to_dict(),
        "distinct_participants": int(ok.participant_id.nunique()),
        "av_lag_status_counts": (
            pool.av_lag_status.value_counts().to_dict() if "av_lag_status" in pool.columns else {}
        ),
        "av_lag_proxy_verdict": (
            "FAILED on V00: peak correlation is too low for the argmax lag to be "
            "meaningful; see reports/02_review_findings.md"
        ),
        "biggest_label_differences": comparison.assign(
            magnitude=comparison.rank_biserial.abs()
        ).nlargest(8, "magnitude")[
            ["vocabulary_term", "column", "rank_biserial", "naturalistic_p50", "improvised_p50"]
        ].to_dict("records"),
        "note": (
            "Tail cut points are reporting descriptions, not selection thresholds. "
            "The pool is a seeded random sample of eligible V00 files, so an extreme "
            "here is the most extreme in the sample, not the corpus maximum."
        ),
    }
    (args.out / "analysis_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True, default=str), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True, default=str)[:3000])


if __name__ == "__main__":
    main()

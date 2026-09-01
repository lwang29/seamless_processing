#!/usr/bin/env python3
"""Summarize M-3 distributions and correlate quality signals with activity.

The question this answers is not "is the data good" but "does each candidate
quality signal measure quality, or does it measure motion?". A jitter statistic
that rises with wrist speed is describing gesture, not noise, and would reject
the most animated speakers if used as a quality gate. Every such signal is
flagged, and a residualized variant is computed so the effect size of the fix is
visible rather than asserted.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


# Gesture activity references. Two are used, not one: a 2D reference is
# available on every file, while the metric 3D reference is the one that is
# unit-meaningful, and a signal that tracks both is unambiguous.
ACTIVITY_REFERENCES = {
    "activity_2d_wrist_speed": "speed2d_body17_rasterfrac_per_s_p90",
    "activity_3d_wrist_speed_mm_per_s": "wrist3d_either_speed_mm_per_s_p90",
    "activity_hand_angular_rate": "angvel_left_hand_rad_per_s_p90",
}
# Signals whose intended reading is "this recording is degraded".
QUALITY_SIGNAL_PREFIXES = (
    "valid_frac_", "jitter2d_hp_", "accel2d_", "accel3d_",
    "hand_left_usable", "hand_right_usable", "hand_left_out_of_frame",
    "hand_right_out_of_frame", "hand_left_complete_frame", "hand_right_complete_frame",
    "hand_left_unavailable_run", "hand_right_unavailable_run",
    "hand_left_pose_temporal_var", "hand_right_pose_temporal_var",
    "hand_left_pose_frozen", "hand_right_pose_frozen",
    "hand_left_pose_exact_zero", "hand_right_pose_exact_zero",
)
PERCENTILES = [1, 5, 25, 50, 75, 95, 99]
# Reporting cut for "this quality signal is substantially explained by motion".
ACTIVITY_CONFOUND_FLOOR = 0.50


def _numeric_columns(frame: pd.DataFrame) -> list[str]:
    columns: list[str] = []
    for name in frame.columns:
        if name.endswith(("_status", "_basis")) or name in {
            "file_id", "vendor", "label", "split", "span_index", "span_start_frame",
        }:
            continue
        series = pd.to_numeric(frame[name], errors="coerce")
        if series.notna().sum() >= 3 and series.nunique(dropna=True) > 1:
            columns.append(name)
    return columns


def _spearman(left: pd.Series, right: pd.Series) -> tuple[float, int]:
    usable = left.notna() & right.notna()
    if usable.sum() < 3:
        return float("nan"), int(usable.sum())
    a = left[usable].rank().to_numpy()
    b = right[usable].rank().to_numpy()
    if np.std(a) == 0 or np.std(b) == 0:
        return float("nan"), int(usable.sum())
    return float(np.corrcoef(a, b)[0, 1]), int(usable.sum())


def describe(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for name in columns:
        series = pd.to_numeric(frame[name], errors="coerce")
        row: dict[str, Any] = {
            "signal": name,
            "n": int(series.notna().sum()),
            "n_missing": int(series.isna().sum()),
            "mean": float(series.mean()),
            "std": float(series.std()),
        }
        for percentile in PERCENTILES:
            row[f"p{percentile}"] = float(series.quantile(percentile / 100))
        rows.append(row)
    return pd.DataFrame(rows)


def residualize(target: pd.Series, activity: pd.Series) -> pd.Series:
    """Remove the rank-linear part of ``activity`` from ``target``.

    Ranks rather than raw values, so a monotone but non-linear relationship is
    removed too and the result stays comparable to the Spearman figures used
    everywhere else in this report.
    """

    usable = target.notna() & activity.notna()
    result = pd.Series(np.nan, index=target.index, dtype=float)
    if usable.sum() < 3:
        return result
    y = target[usable].rank().to_numpy()
    x = activity[usable].rank().to_numpy()
    if np.std(x) == 0:
        return result
    slope = np.cov(y, x, bias=True)[0, 1] / np.var(x)
    result.loc[usable] = y - slope * (x - x.mean()) - y.mean()
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--features",
        type=Path,
        default=Path("outputs/session2/m3_dev_features/span_features.parquet"),
    )
    parser.add_argument("--out", type=Path, default=Path("outputs/session2/m3_dev_features"))
    args = parser.parse_args()

    frame = pd.read_parquet(args.features)
    args.out.mkdir(parents=True, exist_ok=True)
    columns = _numeric_columns(frame)

    describe(frame, columns).to_csv(args.out / "distributions_all_spans.csv", index=False)
    per_vendor: list[pd.DataFrame] = []
    for (vendor, label), group in frame.groupby(["vendor", "label"]):
        table = describe(group, columns)
        table.insert(0, "vendor", vendor)
        table.insert(1, "label", label)
        per_vendor.append(table)
    pd.concat(per_vendor, ignore_index=True).to_csv(
        args.out / "distributions_by_vendor_label.csv", index=False
    )

    available_references = {
        name: column for name, column in ACTIVITY_REFERENCES.items() if column in frame.columns
    }
    quality = [
        name for name in columns
        if name.startswith(QUALITY_SIGNAL_PREFIXES) and name not in available_references.values()
    ]

    rows: list[dict[str, Any]] = []
    for name in quality:
        target = pd.to_numeric(frame[name], errors="coerce")
        record: dict[str, Any] = {"quality_signal": name}
        worst = 0.0
        for reference_name, column in available_references.items():
            activity = pd.to_numeric(frame[column], errors="coerce")
            rho, n = _spearman(target, activity)
            record[f"rho_{reference_name}"] = rho
            record[f"n_{reference_name}"] = n
            if np.isfinite(rho):
                worst = max(worst, abs(rho))
        record["max_abs_rho_vs_activity"] = worst
        record["activity_confounded"] = bool(worst >= ACTIVITY_CONFOUND_FLOOR)
        primary = available_references.get("activity_3d_wrist_speed_mm_per_s")
        if primary is not None and record["activity_confounded"]:
            residual = residualize(target, pd.to_numeric(frame[primary], errors="coerce"))
            rho_after, _ = _spearman(residual, pd.to_numeric(frame[primary], errors="coerce"))
            record["rho_after_residualizing_on_3d_wrist_speed"] = rho_after
        rows.append(record)
    confound = pd.DataFrame(rows).sort_values(
        "max_abs_rho_vs_activity", ascending=False, kind="mergesort"
    )
    confound.to_csv(args.out / "quality_vs_activity_correlations.csv", index=False)

    # Cross-correlation among the whole numeric signal set, so the report can
    # state which signals are redundant rather than guessing.
    numeric = frame[columns].apply(pd.to_numeric, errors="coerce")
    ranks = numeric.rank()
    correlation = ranks.corr(method="pearson", min_periods=3)
    correlation.to_csv(args.out / "signal_rank_correlation_matrix.csv")

    summary = {
        "spans": int(len(frame)),
        "files": int(frame.file_id.nunique()),
        "numeric_signals": len(columns),
        "activity_references": available_references,
        "activity_confound_floor": ACTIVITY_CONFOUND_FLOOR,
        "quality_signals_examined": len(quality),
        "activity_confounded_signals": sorted(
            confound.loc[confound.activity_confounded, "quality_signal"].tolist()
        ),
        "top_confounded": confound.head(12)[
            ["quality_signal", "max_abs_rho_vs_activity"]
        ].to_dict(orient="records"),
    }
    (args.out / "analysis_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True)[:4000])


if __name__ == "__main__":
    main()

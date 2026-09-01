"""Join human review ratings to computed signals and say what nothing detects.

The most valuable output of this analysis is the negative one. If reviewers
consistently flag a failure mode and no computed signal separates the clips they
flagged, that is a measured statement that the released masks are insufficient
and a purpose-built detector is required. The reporting below is therefore
symmetric: for every rating item it names the best-separating signals *and*
records the item in ``items_without_a_separating_signal`` when nothing clears
the stated separation floor.

Nothing here selects data or sets a gate. ``separation_floor`` is a reporting
cut for "did any signal track this at all", not a quality threshold.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd


# Reporting cut only: below this absolute Spearman rho we are unwilling to claim
# that a signal tracks a rating item at all.
SEPARATION_FLOOR = 0.30
# Columns that identify or annotate rather than measure.
NON_SIGNAL_COLUMNS = {
    "review_item_id", "clip_id", "file_id", "source_relbase",
    "partner_source_relbase", "sample_group", "vendor", "label", "split",
    "activity_type", "selection_reason", "render_policy", "reviewer_id",
    "free_text", "complete", "signals_json",
}


def load_rating_export(path: str | Path) -> pd.DataFrame:
    """Flatten one browser rating export into a tidy frame.

    Scale values arrive as strings from radio inputs and the overall question
    arrives as ``yes``/``no``; both are converted here so downstream code never
    has to guess. Unanswered items become NaN rather than a filled-in default.
    """

    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    schema = str(payload.get("schema", ""))
    if schema not in ("structured_ratings_v1", "exploratory_free_text_v1"):
        raise ValueError(f"unrecognized rating export schema: {schema!r}")
    rows: list[dict[str, Any]] = []
    for entry in payload.get("entries", []):
        row: dict[str, Any] = {
            "review_item_id": str(entry.get("review_item_id", "")),
            "reviewer_id": str(payload.get("reviewer_id", "")),
            "free_text": str(entry.get("free_text", "")),
            "export_schema": schema,
            "rubric_hash": str(payload.get("rubric_hash", "")),
        }
        for item_id, value in dict(entry.get("ratings") or {}).items():
            text = str(value).strip().lower()
            if text in ("yes", "true"):
                row[item_id] = True
            elif text in ("no", "false"):
                row[item_id] = False
            else:
                try:
                    row[item_id] = float(text)
                except ValueError:
                    row[item_id] = np.nan
        rows.append(row)
    if not rows:
        raise ValueError("rating export contains no entries")
    return pd.DataFrame(rows)


def load_rating_exports(paths: Iterable[str | Path]) -> pd.DataFrame:
    """Concatenate several reviewers' exports, keeping reviewer identity."""

    frames = [load_rating_export(path) for path in paths]
    if not frames:
        raise ValueError("no rating exports supplied")
    return pd.concat(frames, ignore_index=True)


def _spearman(left: np.ndarray, right: np.ndarray) -> float:
    """Spearman rho via Pearson on ranks, with ties averaged."""

    if len(left) < 3:
        return float("nan")
    left_ranks = pd.Series(left).rank().to_numpy()
    right_ranks = pd.Series(right).rank().to_numpy()
    if np.std(left_ranks) == 0 or np.std(right_ranks) == 0:
        return float("nan")
    return float(np.corrcoef(left_ranks, right_ranks)[0, 1])


def _auc(positive: np.ndarray, negative: np.ndarray) -> float:
    """Rank-based AUC: probability a positive outranks a negative."""

    if positive.size == 0 or negative.size == 0:
        return float("nan")
    combined = np.concatenate([positive, negative])
    ranks = pd.Series(combined).rank().to_numpy()
    positive_rank_sum = ranks[: positive.size].sum()
    n_pos, n_neg = float(positive.size), float(negative.size)
    return float((positive_rank_sum - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def signal_columns(signals: pd.DataFrame) -> list[str]:
    """Numeric measurement columns, excluding identifiers and status strings."""

    columns: list[str] = []
    for name in signals.columns:
        if name in NON_SIGNAL_COLUMNS or name.endswith("_status") or name.endswith("_basis"):
            continue
        series = pd.to_numeric(signals[name], errors="coerce")
        if series.notna().sum() >= 3 and series.nunique(dropna=True) > 1:
            columns.append(name)
    return columns


def correlate_ratings_with_signals(
    ratings: pd.DataFrame,
    signals: pd.DataFrame,
    *,
    rating_items: Sequence[str],
    separation_floor: float = SEPARATION_FLOOR,
    overall_column: str = "would_train",
) -> dict[str, Any]:
    """Rank-correlate every signal against every rating item.

    Duplicated review items are averaged per clip first, so a clip rated twice
    does not count twice. The returned ``items_without_a_separating_signal`` list
    is the negative result: rating items that no available signal tracks.
    """

    merged = ratings.merge(signals, on="review_item_id", how="inner", suffixes=("", "_signal"))
    if merged.empty:
        raise ValueError("no review_item_id overlap between ratings and signals")
    available = [item for item in rating_items if item in merged.columns]
    missing = [item for item in rating_items if item not in merged.columns]
    columns = signal_columns(signals)

    records: list[dict[str, Any]] = []
    for item in available:
        target = pd.to_numeric(merged[item], errors="coerce")
        for name in columns:
            values = pd.to_numeric(merged[name], errors="coerce")
            usable = target.notna() & values.notna()
            rho = _spearman(target[usable].to_numpy(), values[usable].to_numpy()) if usable.sum() >= 3 else float("nan")
            records.append(
                {
                    "rating_item": item,
                    "signal": name,
                    "n": int(usable.sum()),
                    "spearman": rho,
                    "abs_spearman": abs(rho) if np.isfinite(rho) else float("nan"),
                }
            )
    correlations = pd.DataFrame(records)
    if not correlations.empty:
        correlations = correlations.sort_values(
            ["rating_item", "abs_spearman"], ascending=[True, False], kind="mergesort"
        ).reset_index(drop=True)

    undetected: list[str] = []
    best_per_item: dict[str, Any] = {}
    for item in available:
        subset = correlations[correlations.rating_item.eq(item)]
        best = subset.abs_spearman.max() if not subset.empty else float("nan")
        best_per_item[item] = {
            "best_abs_spearman": float(best) if np.isfinite(best) else None,
            "best_signals": subset.nlargest(5, "abs_spearman")[
                ["signal", "spearman", "n"]
            ].to_dict(orient="records"),
        }
        if not np.isfinite(best) or best < separation_floor:
            undetected.append(item)

    binary_records: list[dict[str, Any]] = []
    if overall_column in merged.columns:
        flag = merged[overall_column]
        positive_mask = flag.eq(True)
        negative_mask = flag.eq(False)
        for name in columns:
            values = pd.to_numeric(merged[name], errors="coerce")
            positive = values[positive_mask & values.notna()].to_numpy()
            negative = values[negative_mask & values.notna()].to_numpy()
            auc = _auc(positive, negative)
            binary_records.append(
                {
                    "signal": name,
                    "n_would_train": int(positive.size),
                    "n_would_not_train": int(negative.size),
                    "median_would_train": float(np.median(positive)) if positive.size else float("nan"),
                    "median_would_not_train": float(np.median(negative)) if negative.size else float("nan"),
                    "auc": auc,
                    "auc_distance_from_chance": abs(auc - 0.5) if np.isfinite(auc) else float("nan"),
                }
            )
    binary = pd.DataFrame(binary_records)
    if not binary.empty:
        binary = binary.sort_values(
            "auc_distance_from_chance", ascending=False, kind="mergesort"
        ).reset_index(drop=True)

    return {
        "correlations": correlations,
        "would_train_separation": binary,
        "items_without_a_separating_signal": undetected,
        "best_signal_per_item": best_per_item,
        "rating_items_absent_from_export": missing,
        "separation_floor": separation_floor,
        "clips_analyzed": int(merged.review_item_id.nunique()),
        "signals_considered": len(columns),
    }


def duplicate_agreement(
    ratings: pd.DataFrame,
    duplicate_map: Mapping[str, str],
    *,
    rating_items: Sequence[str],
    overall_column: str = "would_train",
) -> dict[str, Any]:
    """Repeat-item agreement, per scale and for the overall question.

    Agreement is measured only on pairs where both members were actually rated,
    and the count of usable pairs is reported next to every figure so a high
    agreement computed from two pairs cannot be read as a strong result.
    """

    indexed = ratings.set_index("review_item_id")
    scale_rows: list[dict[str, Any]] = []
    for item in rating_items:
        if item not in ratings.columns:
            continue
        differences: list[float] = []
        for duplicate_id, original_id in duplicate_map.items():
            if duplicate_id not in indexed.index or original_id not in indexed.index:
                continue
            first = pd.to_numeric(pd.Series([indexed.at[original_id, item]]), errors="coerce").iloc[0]
            second = pd.to_numeric(pd.Series([indexed.at[duplicate_id, item]]), errors="coerce").iloc[0]
            if np.isfinite(first) and np.isfinite(second):
                differences.append(abs(float(first) - float(second)))
        array = np.asarray(differences, dtype=float)
        scale_rows.append(
            {
                "rating_item": item,
                "pairs": int(array.size),
                "exact_agreement": float((array == 0).mean()) if array.size else float("nan"),
                "within_one_agreement": float((array <= 1).mean()) if array.size else float("nan"),
                "mean_abs_difference": float(array.mean()) if array.size else float("nan"),
            }
        )

    matches: list[bool] = []
    if overall_column in ratings.columns:
        for duplicate_id, original_id in duplicate_map.items():
            if duplicate_id not in indexed.index or original_id not in indexed.index:
                continue
            first = indexed.at[original_id, overall_column]
            second = indexed.at[duplicate_id, overall_column]
            if isinstance(first, (bool, np.bool_)) and isinstance(second, (bool, np.bool_)):
                matches.append(bool(first) == bool(second))
    return {
        "scales": pd.DataFrame(scale_rows),
        "would_train": {
            "pairs": len(matches),
            "agreement": float(np.mean(matches)) if matches else float("nan"),
        },
    }

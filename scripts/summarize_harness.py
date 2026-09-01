#!/usr/bin/env python3
"""Verify and summarize restartable Session 1 harness outputs."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


FEATURE_COLUMNS = (
    "valid_frac_all",
    "accel_mm_per_frame2",
    "wrist_speed_p90",
    "duration_mismatch_s",
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("outputs/harness"))
    parser.add_argument("--expected-files", type=int)
    args = parser.parse_args()

    root = args.root.resolve()
    parquet_paths = sorted((root / "files").rglob("*.parquet"))
    marker_paths = sorted((root / "complete").rglob("*.json"))
    error_paths = sorted((root / "errors").rglob("*.json"))
    temporary_paths = sorted(root.rglob("*.tmp")) + sorted(root.rglob(".*.tmp"))

    markers = [json.loads(path.read_text(encoding="utf-8")) for path in marker_paths]
    if args.expected_files is not None:
        assert len(parquet_paths) == args.expected_files
        assert len(marker_paths) == args.expected_files
    assert not error_paths
    assert not temporary_paths
    assert all(marker["status"] == "complete" for marker in markers)

    tables = [pq.read_table(path) for path in parquet_paths]
    table = pa.concat_tables(tables) if tables else pa.table({})
    rows = table.num_rows
    marker_rows = sum(int(marker["rows"]) for marker in markers)
    assert rows == marker_rows

    summary: dict[str, object] = {
        "root": str(root),
        "parquet_files": len(parquet_paths),
        "completion_markers": len(marker_paths),
        "error_records": len(error_paths),
        "temporary_files": len(temporary_paths),
        "rows": rows,
        "zero_row_files": sorted(marker["file_id"] for marker in markers if marker["rows"] == 0),
        "marker_statuses": dict(sorted(Counter(marker["status"] for marker in markers).items())),
        "marker_git_sha": sorted({marker["git_sha"] for marker in markers}),
        "marker_config_hash": sorted({marker["config_hash"] for marker in markers}),
        "marker_git_dirty": sorted({marker["git_dirty"] for marker in markers}),
    }
    if rows:
        summary.update(
            {
                "rows_by_label": dict(
                    sorted(Counter(table["label"].to_pylist()).items())
                ),
                "rows_by_vendor": dict(
                    sorted(Counter(table["vendor"].to_pylist()).items())
                ),
                "feature_arrow_null_rows": {
                    name: table[name].null_count for name in FEATURE_COLUMNS
                },
                "feature_finite_rows": {
                    name: int(np.isfinite(np.asarray(table[name].to_pylist(), dtype=float)).sum())
                    for name in FEATURE_COLUMNS
                },
                "feature_nan_rows": {
                    name: int(np.isnan(np.asarray(table[name].to_pylist(), dtype=float)).sum())
                    for name in FEATURE_COLUMNS
                },
                "valid_statuses": dict(
                    sorted(Counter(table["valid_status"].to_pylist()).items())
                ),
                "accel_statuses": dict(
                    sorted(Counter(table["accel_status"].to_pylist()).items())
                ),
                "wrist_statuses": dict(
                    sorted(Counter(table["wrist_status"].to_pylist()).items())
                ),
                "duration_fps_statuses": dict(
                    sorted(Counter(table["duration_fps_status"].to_pylist()).items())
                ),
                "frame_count_statuses": dict(
                    sorted(Counter(table["frame_count_status"].to_pylist()).items())
                ),
                "window_lengths": sorted(
                    {
                        int(end) - int(start)
                        for start, end in zip(
                            table["window_start_frame"].to_pylist(),
                            table["window_end_frame_exclusive"].to_pylist(),
                        )
                    }
                ),
                "git_sha": sorted(set(table["git_sha"].to_pylist())),
                "config_hash": sorted(set(table["config_hash"].to_pylist())),
                "git_dirty": sorted(set(table["git_dirty"].to_pylist())),
                "physical_units_verified": sorted(
                    set(table["physical_units_verified"].to_pylist())
                ),
            }
        )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

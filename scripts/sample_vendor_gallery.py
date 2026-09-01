#!/usr/bin/env python3
"""Draw a vendor-stratified exploratory sample: what does the pipeline do elsewhere?

Six pools — pass and flagged for each of V01, V02, V03 — each a plain random
draw. Unlike the V00 galleries there is no per-failure-mode quota and no
preference for singly-flagged files, because the question here is not "does each
detector fire on what it claims to". It is "what is in these vendors that we have
never looked at", and any curation of the draw works against that.

The pass/fail split is what a **V00-tuned** pipeline says. Treat it as a
convenient way to see both ends of the distribution, not as ground truth: FM1's
cut was fitted against V00 hand labels, and these vendors have different cameras,
rasters and frame rates.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

MANIFEST_COLUMNS = [
    "review_item_id", "clip_id", "file_id", "source_relbase",
    "partner_source_relbase", "sample_group", "vendor_stratum", "label_stratum",
    "start_frame", "vendor", "label", "split", "activity_type",
    "selection_reason", "signals_json", "render_policy",
]
SIGNAL_KEYS = (
    "fm1_knee_between_torso_p50", "fm2_smplh_valid_frac", "fm3_static_frac",
    "reproj_shoulder_widths_p50", "observed_duration_s",
    "fm2_longest_invalid_run_frames", "valid_frac_box",
)


def _worktree() -> Path:
    here = Path(__file__).resolve().parent
    for candidate in (here, *here.parents):
        if (candidate / ".git-session").is_dir():
            return candidate
    raise RuntimeError("could not locate .git-session")


def _git(worktree: Path) -> dict[str, Any]:
    base = ["git", f"--git-dir={worktree / '.git-session'}", f"--work-tree={worktree}"]
    sha = subprocess.run([*base, "rev-parse", "HEAD"], check=True, capture_output=True, text=True)
    status = subprocess.run(
        [*base, "status", "--porcelain", "--untracked-files=normal"],
        check=True, capture_output=True, text=True,
    )
    return {"git_sha": sha.stdout.strip(), "git_dirty": bool(status.stdout.strip())}


def _unit(*parts: Any) -> float:
    digest = hashlib.sha256("|".join(str(p) for p in parts).encode()).digest()
    return int.from_bytes(digest[:8], "big") / 2.0**64


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/vendors_fm_detect.yaml"))
    parser.add_argument("--gallery-config", type=Path,
                        default=Path("configs/vendors_review.yaml"))
    args = parser.parse_args()

    worktree = _worktree()
    gallery = yaml.safe_load((worktree / args.gallery_config).read_text(encoding="utf-8"))
    detect = yaml.safe_load((worktree / args.config).read_text(encoding="utf-8"))
    scan = pd.read_parquet(
        worktree / str(detect["outputs"]["root"]).lstrip("./") / "scan_with_flags.parquet"
    )
    manifest_path = worktree / str(gallery["outputs"]["manifest"]).lstrip("./")
    report_dir = worktree / str(gallery["outputs"]["report_dir"]).lstrip("./")
    report_dir.mkdir(parents=True, exist_ok=True)
    sampling = gallery["sampling"]
    seed = int(sampling["seed"])
    per_cell = int(sampling["per_vendor_per_verdict"])
    clip_duration_s = float(gallery["clip"]["duration_s"])

    eligible = scan.loc[
        [
            int(r.frames_analyzed) >= int(round(clip_duration_s * float(r.fps)))
            for r in scan.itertuples(index=False)
        ]
    ].copy()

    chosen: list[dict[str, Any]] = []
    used: set[str] = set()

    def take(pool: pd.DataFrame, count: int, group: str, reason: str) -> int:
        available = pool.loc[~pool.file_id.isin(used)].copy()
        if available.empty or count <= 0:
            return 0
        available["draw"] = [_unit(seed, group, fid) for fid in available.file_id]
        taken = available.nsmallest(min(count, len(available)), "draw")
        for row in taken.itertuples(index=False):
            used.add(row.file_id)
            chosen.append({"row": row, "sample_group": group, "selection_reason": reason})
        return len(taken)

    for vendor in sorted(eligible.vendor_id.unique()):
        pool = eligible.loc[eligible.vendor_id.eq(vendor)]
        take(
            pool.loc[pool.passes_all], per_cell, f"v{vendor}_pass",
            f"random draw from V{vendor} files the current pipeline keeps",
        )
        take(
            pool.loc[~pool.passes_all], per_cell, f"v{vendor}_flagged",
            f"random draw from V{vendor} files the current pipeline rejects",
        )

    records: list[dict[str, Any]] = []
    for item in chosen:
        row = item["row"]
        fps = float(row.fps)
        render_frames = int(round(clip_duration_s * fps))
        span = max(0, int(row.frames_analyzed) - render_frames)
        # A plain random position. These clips are for spotting failure modes we
        # have not named yet, so anchoring on a known one would bias what gets
        # seen toward what we already know about.
        start = max(0, min(int(_unit(seed, "start", row.file_id) * (span + 1)), span))
        signals = {
            key: (float(getattr(row, key)) if pd.notna(getattr(row, key, None)) else None)
            for key in SIGNAL_KEYS if hasattr(row, key)
        }
        signals.update(
            {
                "vendor": f"V{row.vendor_id}",
                "raster": f"{int(row.width)}x{int(row.height)}",
                "fps": round(fps, 3),
                "flags_fired": str(row.flags_fired) or "none",
                "verdict": "PASSES the current pipeline" if bool(row.passes_all)
                           else f"REJECTED by {str(row.flags_fired) or 'an undetermined check'}",
            }
        )
        records.append(
            {
                "review_item_id": "", "clip_id": "",
                "file_id": row.file_id, "source_relbase": row.source_relbase,
                "partner_source_relbase": row.partner_source_relbase,
                "sample_group": item["sample_group"],
                "vendor_stratum": f"V{row.vendor_id}", "label_stratum": row.label,
                "start_frame": start, "vendor": f"V{row.vendor_id}", "label": row.label,
                "split": row.split, "activity_type": row.interaction_type,
                "selection_reason": item["selection_reason"],
                "signals_json": json.dumps(signals, sort_keys=True),
                "render_policy": "render",
            }
        )

    for index, record in enumerate(records):
        record["clip_id"] = (
            f"{record['sample_group']}_{index:03d}_{record['file_id']}"
            f"_f{int(record['start_frame']):07d}"
        )
        record["review_item_id"] = f"vend_{index:03d}"

    manifest = pd.DataFrame(records, columns=MANIFEST_COLUMNS)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(manifest_path, index=False)
    for group in sorted(manifest.sample_group.unique()):
        manifest.loc[manifest.sample_group.eq(group)].to_csv(
            manifest_path.with_name(f"vendors_review_{group}.csv"), index=False
        )
    for vendor in sorted(manifest.vendor_stratum.unique()):
        manifest.loc[manifest.vendor_stratum.eq(vendor)].to_csv(
            manifest_path.with_name(f"vendors_review_{vendor.lower()}.csv"), index=False
        )

    summary = {
        "scanned_files": int(len(scan)),
        "eligible_for_a_30s_clip": int(len(eligible)),
        "clips": len(records),
        "by_group": manifest.sample_group.value_counts().to_dict(),
        "by_vendor_and_label": [
            {"vendor": v, "label": l, "clips": int(c)}
            for (v, l), c in manifest.groupby(["vendor_stratum", "label_stratum"]).size().items()
        ],
        "rasters": manifest.signals_json.str.extract(r'"raster": "([^"]+)"')[0]
                   .value_counts().to_dict(),
        "distinct_participants": int(
            scan.set_index("file_id").loc[manifest.file_id].participant_id.nunique()
        ),
        "distinct_sessions": int(
            scan.set_index("file_id").loc[manifest.file_id].session_id.nunique()
        ),
        "seed": seed,
        **_git(worktree),
    }
    (report_dir / "vendor_sample.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True, default=str), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True, default=str))


if __name__ == "__main__":
    main()

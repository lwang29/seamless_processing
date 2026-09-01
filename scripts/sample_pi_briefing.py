#!/usr/bin/env python3
"""Draw the corpus-wide briefing sample: four vendors, seven strata each.

Two passing strata split by recording condition, and one per check that can
reject. Every draw is plain random within its stratum -- this page exists to show
what the pipeline does, so a curated selection would misrepresent it.

Three strata are empty by construction rather than by chance, and the page says
so rather than silently omitting them:

* **V02 improvised** -- V02 recorded no improvised material at all.
* **V00 and V02 FM0** -- neither has any of the four excluded rasters, and
  neither has a file whose annotation grid misses its video.
* **V01 and V02 FM1**, **V00 FM4** -- switched off for those vendors, each for a
  reason a review established.

Clip position is random except for FM2, where the clip is centred on the longest
continuous stretch the SMPL-H validity flag calls untrusted, so the frames the
check objected to are the frames on screen.
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
VENDORS = ("00", "01", "02", "03")
# (stratum key, how a file qualifies)
FLAG_CELLS = (
    "fm0_unusable_source", "fm1_sitting", "fm2_smplh_invalid",
    "fm3_static_hands", "fm4_audio_dead",
)
SIGNAL_KEYS = (
    "fm1_knee_between_torso_p50", "fm1_hip_flexion_deg_p50",
    "fm2_smplh_valid_frac", "fm2_longest_invalid_run_frames",
    "fm3_static_frac", "audio_envelope_dynamics_db", "observed_duration_s",
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
    status = subprocess.run([*base, "status", "--porcelain", "--untracked-files=normal"],
                            check=True, capture_output=True, text=True)
    return {"git_sha": sha.stdout.strip(), "git_dirty": bool(status.stdout.strip())}


def _unit(*parts: Any) -> float:
    digest = hashlib.sha256("|".join(str(p) for p in parts).encode()).digest()
    return int.from_bytes(digest[:8], "big") / 2.0**64


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/pi_briefing_detect.yaml"))
    parser.add_argument("--gallery-config", type=Path,
                        default=Path("configs/pi_briefing_review.yaml"))
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
    clip_duration_s = float(gallery["clip"]["duration_s"])
    per_cell = int(sampling["per_cell"])

    scan = scan.copy()
    scan["clip_frames"] = [
        int(round(clip_duration_s * float(r.fps))) for r in scan.itertuples(index=False)
    ]
    eligible = scan.loc[scan.status.eq("ok") & (scan.frames_analyzed >= scan.clip_frames)].copy()

    chosen: list[dict[str, Any]] = []
    used: set[str] = set()
    counts: dict[str, int] = {}

    def take(pool: pd.DataFrame, count: int, group: str, reason: str,
             centre_on_invalid: bool = False) -> int:
        available = pool.loc[~pool.file_id.isin(used)].copy()
        if available.empty or count <= 0:
            counts[group] = 0
            return 0
        available["draw"] = [_unit(seed, group, fid) for fid in available.file_id]
        taken = available.nsmallest(min(count, len(available)), "draw")
        for row in taken.itertuples(index=False):
            used.add(row.file_id)
            chosen.append({"row": row, "sample_group": group, "selection_reason": reason,
                           "centre_on_invalid": centre_on_invalid})
        counts[group] = len(taken)
        return len(taken)

    for vendor in VENDORS:
        vend = eligible.loc[eligible.vendor_id.eq(vendor)]
        passing = vend.loc[vend.passes_all.fillna(False).astype(bool)]
        for label in ("improvised", "naturalistic"):
            take(
                passing.loc[passing.label.eq(label)], per_cell,
                f"v{vendor}_pass_{label}",
                f"random draw from V{vendor} {label} files the pipeline keeps",
            )
        for flag in FLAG_CELLS:
            take(
                vend.loc[vend[flag].fillna(False).astype(bool)], per_cell,
                f"v{vendor}_{flag}",
                f"random draw from V{vendor} files that tripped {flag}",
                centre_on_invalid=(flag == "fm2_smplh_invalid"),
            )

    records: list[dict[str, Any]] = []
    for item in chosen:
        row = item["row"]
        fps = float(row.fps)
        render_frames = int(round(clip_duration_s * fps))
        # Bound by whichever is smallest: the annotations, the duration, and the
        # frames the container actually holds. One file has 11,600 annotations
        # against 11,599 stored frames, which is inside the timebase tolerance and
        # so takes the plain sequential decode -- and then runs out one frame early.
        video_frames = int(float(row.observed_duration_s) * fps)
        container = int(getattr(row, "container_frames", 0) or 0)
        limits = [int(row.frames_analyzed), video_frames]
        if container > 0:
            limits.append(container)
        span = max(0, min(limits) - render_frames)

        def _int(name: str, missing: int) -> int:
            value = getattr(row, name, None)
            return missing if value is None or pd.isna(value) else int(value)

        longest = _int("fm2_longest_invalid_run_frames", 0)
        run_start = _int("fm2_longest_invalid_run_start_frame", -1)
        suffix = ""
        if item["centre_on_invalid"] and longest > 0 and run_start >= 0:
            start = run_start - max(0, (render_frames - longest) // 2)
            suffix = "; positioned on the longest untrusted stretch"
        else:
            start = int(_unit(seed, "start", row.file_id) * (span + 1))
        start = max(0, min(start, span))

        signals = {
            key: (float(getattr(row, key)) if pd.notna(getattr(row, key, None)) else None)
            for key in SIGNAL_KEYS if hasattr(row, key)
        }
        # Seconds read better than frames for a reader who has never seen the scan.
        if signals.get("fm2_longest_invalid_run_frames") is not None:
            signals["fm2_longest_invalid_run_s"] = signals.pop(
                "fm2_longest_invalid_run_frames") / fps
        fired = [f for f in FLAG_CELLS if bool(getattr(row, f, False))]
        signals.update(
            {
                "audio_status": str(getattr(row, "audio_status", "")) or "unknown",
                "vendor": f"V{row.vendor_id}",
                "condition": str(row.label),
                "verdict": "kept" if bool(row.passes_all) else
                           "rejected by " + " and ".join(f.split("_")[0].upper() for f in fired),
                "fm1_scope": str(getattr(row, "fm1_sitting_scope", "session")),
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
                "selection_reason": item["selection_reason"] + suffix,
                "signals_json": json.dumps(signals, sort_keys=True),
                "render_policy": "render",
            }
        )

    for index, record in enumerate(records):
        record["clip_id"] = (f"{record['sample_group']}_{index:03d}_{record['file_id']}"
                             f"_f{int(record['start_frame']):07d}")
        record["review_item_id"] = f"pi_{index:03d}"

    manifest = pd.DataFrame(records, columns=MANIFEST_COLUMNS)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(manifest_path, index=False)

    population = {
        f"V{v}": {
            "scanned": int(scan.vendor_id.eq(v).sum()),
            "improvised": int((scan.vendor_id.eq(v) & scan.label.eq("improvised")).sum()),
            "naturalistic": int((scan.vendor_id.eq(v) & scan.label.eq("naturalistic")).sum()),
            **{f: int((scan.vendor_id.eq(v) & scan[f].fillna(False).astype(bool)).sum())
               for f in FLAG_CELLS},
        }
        for v in VENDORS
    }
    summary = {
        "scanned_files": int(len(scan)),
        "eligible_for_a_clip": int(len(eligible)),
        "clips": len(records),
        "draw_counts": counts,
        "empty_strata": sorted(k for k, n in counts.items() if n == 0),
        "population": population,
        "distinct_participants": int(
            scan.set_index("file_id").loc[manifest.file_id].participant_id.nunique()
        ),
        "distinct_sessions": int(
            scan.set_index("file_id").loc[manifest.file_id].session_id.nunique()
        ),
        "seed": seed,
        **_git(worktree),
    }
    (report_dir / "pi_briefing_sample.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True, default=str), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True, default=str))


if __name__ == "__main__":
    main()

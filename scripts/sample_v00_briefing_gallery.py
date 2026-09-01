#!/usr/bin/env python3
"""Draw the five-category briefing sample from a fresh V00 scan.

Five pools, each a plain random draw within its category so the gallery shows an
honest cross-section rather than a curated best-of:

  1. improvised clips that pass all three checks
  2. naturalistic clips that pass all three checks
  3. clips flagged FM1 (seated)
  4. clips flagged FM2 (untrusted SMPL-H)
  5. clips flagged FM3 (static hands)

The flagged pools are drawn from files where **only** that detector fired, when
there are enough of them, so a viewer attributing what they see to the named
check is not being misled by a second problem in the same clip.

FM2 clips are positioned to contain the longest continuous SMPL-H-invalid
stretch, using the frame index the scanner recorded. That matters more here than
in the working gallery: FM2's whole claim is about those frames, and a clip that
happened to miss them would show a viewer nothing.
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
    "partner_source_relbase", "sample_group", "label_stratum", "start_frame",
    "vendor", "label", "split", "activity_type", "selection_reason",
    "signals_json", "render_policy",
]
SIGNAL_KEYS = (
    "fm1_knee_between_torso_p50", "unit_flagged_frac",
    "fm2_smplh_valid_frac", "fm3_static_frac", "observed_duration_s",
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


def fm2_start_frame(row: Any, render_frames: int, span: int) -> tuple[int, bool]:
    """Place the clip so it contains the longest SMPL-H-invalid stretch.

    Centred on the stretch when it fits, so the viewer sees the fit fail and
    recover. When the stretch is longer than the clip, start a little before it
    and let the clip run into it — there is nothing to centre on.
    """

    start = getattr(row, "fm2_longest_invalid_run_start_frame", -1)
    length = getattr(row, "fm2_longest_invalid_run_frames", 0)
    try:
        start, length = int(start), int(length)
    except (TypeError, ValueError):
        return 0, False
    if start < 0 or length <= 0:
        return 0, False
    if length >= render_frames:
        lead = int(round(2.0 * float(row.fps)))
        return max(0, min(start - lead, span)), True
    centred = start - (render_frames - length) // 2
    return max(0, min(centred, span)), True


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/v00_fm_briefing_detect.yaml"))
    parser.add_argument("--gallery-config", type=Path,
                        default=Path("configs/v00_fm_briefing_review.yaml"))
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
    per_category = int(sampling["per_category"])
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

    # Passing clips first: they are the product, and drawing them before the
    # flagged pools keeps them a clean random sample of what survives.
    for label in ("improvised", "naturalistic"):
        take(
            eligible.loc[eligible.passes_all & eligible.label.eq(label)],
            per_category, f"pass_{label}",
            f"random draw from {label} files that passed all three checks",
        )
    for flag, group in (
        ("fm1_sitting", "flagged_fm1"),
        ("fm2_smplh_invalid", "flagged_fm2"),
        ("fm3_static_hands", "flagged_fm3"),
    ):
        pool = eligible.loc[eligible[flag].eq(True)]
        only = pool.loc[pool.flag_count.eq(1)]
        got = take(only, per_category, group, f"random draw from files where only {flag} fired")
        take(pool, per_category - got, group, f"random draw from files where {flag} fired")

    records: list[dict[str, Any]] = []
    for item in chosen:
        row = item["row"]
        fps = float(row.fps)
        render_frames = int(round(clip_duration_s * fps))
        span = max(0, int(row.frames_analyzed) - render_frames)
        anchored = False
        if item["sample_group"] == "flagged_fm2":
            start, anchored = fm2_start_frame(row, render_frames, span)
        else:
            start = int(_unit(seed, "start", row.file_id) * (span + 1))
        start = max(0, min(start, span))
        signals = {
            key: (float(getattr(row, key)) if pd.notna(getattr(row, key, None)) else None)
            for key in SIGNAL_KEYS if hasattr(row, key)
        }
        run_frames = float(getattr(row, "fm2_longest_invalid_run_frames", 0) or 0)
        signals["fm2_longest_invalid_run_s"] = run_frames / fps if run_frames else 0.0
        note = ""
        if item["sample_group"] == "flagged_fm2":
            note = (
                f"Clip positioned on the longest untrusted stretch "
                f"({run_frames / fps:.1f} s starting at "
                f"{int(getattr(row, 'fm2_longest_invalid_run_start_frame', 0)) / fps:.0f} s "
                f"into the recording)."
                if anchored else
                "No SMPL-H-invalid run was recorded for this file; clip position is random."
            )
        records.append(
            {
                "review_item_id": "", "clip_id": "",
                "file_id": row.file_id, "source_relbase": row.source_relbase,
                "partner_source_relbase": row.partner_source_relbase,
                "sample_group": item["sample_group"], "label_stratum": row.label,
                "start_frame": start, "vendor": "V00", "label": row.label,
                "split": row.split, "activity_type": row.interaction_type,
                "selection_reason": item["selection_reason"],
                "signals_json": json.dumps({**signals, "clip_note": note}, sort_keys=True),
                "render_policy": "render",
            }
        )

    for index, record in enumerate(records):
        record["clip_id"] = (
            f"{record['sample_group']}_{index:03d}_{record['file_id']}"
            f"_f{int(record['start_frame']):07d}"
        )
        record["review_item_id"] = f"brief_{index:03d}"

    manifest = pd.DataFrame(records, columns=MANIFEST_COLUMNS)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(manifest_path, index=False)

    anchored_count = int(
        manifest.signals_json.str.contains("Clip positioned on the longest untrusted").sum()
    )
    summary = {
        "scanned_files": int(len(scan)),
        "eligible_for_a_30s_clip": int(len(eligible)),
        "clips": len(records),
        "by_group": manifest.sample_group.value_counts().to_dict(),
        "fm2_clips_anchored_on_the_invalid_run": anchored_count,
        "fm2_clips": int(manifest.sample_group.eq("flagged_fm2").sum()),
        "distinct_participants": int(
            scan.set_index("file_id").loc[manifest.file_id].participant_id.nunique()
        ),
        "seed": seed,
        **_git(worktree),
    }
    (report_dir / "briefing_sample.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True, default=str), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True, default=str))


if __name__ == "__main__":
    main()

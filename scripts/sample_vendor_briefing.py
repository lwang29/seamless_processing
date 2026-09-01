#!/usr/bin/env python3
"""Draw the Round-7 vendor gallery: adjudication asks, then a fresh stratified sample.

Two halves with different jobs.

**The asks.** Round-6 review left four questions that no measurement can settle,
because each of them is "does this look right to a person". They get their own
sections, deliberately loaded rather than random: the V03 posture ladder spans
the measure FM1 cuts on so the labels land where the cut has to go, and the
raster sections show one example of every odd raster in the corpus.

**The sample.** A plain random draw, stratified by vendor and by what the
pipeline said -- pass, or each flag on its own, or several at once. Random
position within the file, except for FM2, where the clip is centred on the
longest continuous stretch the SMPL-H validity flag calls untrusted, so the
frames the check objected to are the frames on screen.
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
    "fm1_knee_between_torso_p50", "fm1_shin_verticality_p50", "fm1_ankles_visible_frac",
    "fm2_smplh_valid_frac", "fm3_static_frac", "reproj_shoulder_widths_p50",
    "observed_duration_s", "unit_flagged_frac", "geom_body_roll_deg",
    "timebase_index_drift_s",
)
# The posture ladder: FM1 cuts knee_between_torso at 0.43 and the seven V03
# files the reviewer confirmed seated all landed between 0.435 and 0.586, just
# above it. Labels are worth most where the two classes actually overlap, so the
# draw is spread evenly across a band that brackets the cut on both sides.
POSTURE_BANDS = ((0.20, 0.32), (0.32, 0.40), (0.40, 0.46), (0.46, 0.52),
                 (0.52, 0.58), (0.58, 0.66), (0.66, 0.80))


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


def _flag_cell(row: Any) -> str:
    """Which stratum a file belongs to: passed, one named flag, or several."""
    if bool(row.passes_all):
        return "pass"
    fired = [f for f in str(row.flags_fired or "").split("+") if f]
    if len(fired) == 1:
        return fired[0]
    return "multi" if fired else "undetermined"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/vendors_r7_detect.yaml"))
    parser.add_argument("--gallery-config", type=Path,
                        default=Path("configs/vendors_r7_review.yaml"))
    parser.add_argument("--rare-config", type=Path,
                        default=Path("configs/vendors_r7_rare_detect.yaml"),
                        help="a companion scan covering every file of the rare rasters, "
                             "which a 1,500-per-vendor draw would miss")
    args = parser.parse_args()

    worktree = _worktree()
    gallery = yaml.safe_load((worktree / args.gallery_config).read_text(encoding="utf-8"))
    detect = yaml.safe_load((worktree / args.config).read_text(encoding="utf-8"))
    scan = pd.read_parquet(
        worktree / str(detect["outputs"]["root"]).lstrip("./") / "scan_with_flags.parquet"
    )
    scan["from_random_draw"] = True
    rare_path = (worktree / str(
        yaml.safe_load((worktree / args.rare_config).read_text(encoding="utf-8"))
        ["outputs"]["root"]).lstrip("./") / "scan_with_flags.parquet")
    if rare_path.exists():
        rare = pd.read_parquet(rare_path)
        rare["from_random_draw"] = False
        # The rare rasters hold 20 to 102 files each, so the random draw catches
        # one or none of some of them. They are kept out of the population
        # statistics -- `from_random_draw` -- and used only for the ask sections.
        scan = pd.concat(
            [scan, rare.loc[~rare.file_id.isin(scan.file_id)]], ignore_index=True
        )
    manifest_path = worktree / str(gallery["outputs"]["manifest"]).lstrip("./")
    report_dir = worktree / str(gallery["outputs"]["report_dir"]).lstrip("./")
    report_dir.mkdir(parents=True, exist_ok=True)
    sampling = gallery["sampling"]
    seed = int(sampling["seed"])
    clip_duration_s = float(gallery["clip"]["duration_s"])

    scan = scan.copy()
    scan["raster"] = scan.width.astype(int).astype(str) + "x" + scan.height.astype(int).astype(str)
    scan["clip_frames"] = [
        int(round(clip_duration_s * float(r.fps))) for r in scan.itertuples(index=False)
    ]
    eligible = scan.loc[
        scan.status.eq("ok") & (scan.frames_analyzed >= scan.clip_frames)
    ].copy()
    eligible["cell"] = [_flag_cell(r) for r in eligible.itertuples(index=False)]

    chosen: list[dict[str, Any]] = []
    used: set[str] = set()

    def take(pool: pd.DataFrame, count: int, group: str, reason: str,
             centre_on_invalid: bool = False) -> int:
        available = pool.loc[~pool.file_id.isin(used)].copy()
        if available.empty or count <= 0:
            return 0
        available["draw"] = [_unit(seed, group, fid) for fid in available.file_id]
        taken = available.nsmallest(min(count, len(available)), "draw")
        for row in taken.itertuples(index=False):
            used.add(row.file_id)
            chosen.append({"row": row, "sample_group": group, "selection_reason": reason,
                           "centre_on_invalid": centre_on_invalid})
        return len(taken)

    counts: dict[str, int] = {}

    # ---- Ask 1: the V03 posture ladder -------------------------------------
    per_band = int(sampling["posture_per_band"])
    ladder = eligible.loc[
        eligible.vendor_id.eq("03") & eligible.fm1_knee_between_torso_p50.notna()
        & eligible.fm1_shin_verticality_p50.gt(0)   # shins the right way up, so the
    ]                                               # label is about posture, not framing
    for low, high in POSTURE_BANDS:
        band = ladder.loc[ladder.fm1_knee_between_torso_p50.between(low, high, inclusive="left")]
        counts[f"ask_posture_{low:.2f}"] = take(
            band, per_band, "ask_posture_v03",
            f"V03, knee_between_torso in [{low:.2f}, {high:.2f}) — posture calibration",
        )

    # ---- Ask 2 and 3: one section per odd raster ---------------------------
    per_raster = int(sampling["per_odd_raster"])
    for raster in sampling["odd_rasters"]:
        pool = eligible.loc[eligible.raster.eq(raster)]
        counts[f"ask_raster_{raster}"] = take(
            pool, per_raster, f"ask_raster_{raster}",
            f"every eligible file of this raster shares one geometry; this is a random draw of {raster}",
        )

    # ---- Ask 4: anamorphic V01, repaired ------------------------------------
    counts["ask_anamorphic"] = take(
        eligible.loc[eligible.geom_anamorphic & eligible.raster.eq("2160x2160")],
        int(sampling["per_ask"]), "ask_anamorphic_v01",
        "V01 2160x2160: the picture and the 2D points are repairable, the released pose is not",
    )

    # ---- Ask 5: files whose annotation grid does not match the video --------
    # Half a second, not one frame: 23% of files are off by a frame or two, which
    # nobody could see, and 0.18% are off by seconds, which is what was reported.
    counts["ask_timebase"] = take(
        eligible.loc[eligible.timebase_index_drift_s.abs() > 0.5],
        int(sampling["per_ask"]), "ask_timebase",
        "the released annotation grid and the container disagree; rendered with the fix applied",
    )

    # ---- The stratified sample ----------------------------------------------
    per_cell = int(sampling["per_vendor_per_cell"])
    for vendor in sorted(eligible.vendor_id.unique()):
        for cell in ("pass", "fm1_sitting", "fm2_smplh_invalid", "fm3_static_hands", "multi"):
            pool = eligible.loc[
                eligible.from_random_draw & eligible.vendor_id.eq(vendor)
                & eligible.cell.eq(cell)
            ]
            group = f"v{vendor}_{cell}"
            counts[group] = take(
                pool, per_cell, group,
                f"random draw from V{vendor} files the pipeline sorted into '{cell}'",
                centre_on_invalid=cell in ("fm2_smplh_invalid", "multi"),
            )

    records: list[dict[str, Any]] = []
    for item in chosen:
        row = item["row"]
        fps = float(row.fps)
        render_frames = int(round(clip_duration_s * fps))
        # Bound by the video as well as by the annotations. On a file whose
        # arrays outrun its recording, the annotation length alone would place
        # the clip past the end of the video.
        video_frames = int(float(row.observed_duration_s) * fps)
        span = max(0, min(int(row.frames_analyzed), video_frames) - render_frames)
        # Not `or -1`: an invalid stretch that begins at frame 0 is a real one,
        # and `0 or -1` is -1, which silently sent those clips to a random spot.
        def _int(name: str, missing: int) -> int:
            value = getattr(row, name, None)
            return missing if value is None or pd.isna(value) else int(value)

        longest = _int("fm2_longest_invalid_run_frames", 0)
        run_start = _int("fm2_longest_invalid_run_start_frame", -1)
        if item["centre_on_invalid"] and longest > 0 and run_start >= 0:
            # Put the untrusted stretch in the middle of the clip when it fits,
            # and at the start of it when it does not.
            start = run_start - max(0, (render_frames - longest) // 2)
            reason_suffix = "; positioned on the longest untrusted stretch"
        else:
            start = int(_unit(seed, "start", row.file_id) * (span + 1))
            reason_suffix = ""
        start = max(0, min(start, span))
        signals = {
            key: (float(getattr(row, key)) if pd.notna(getattr(row, key, None)) else None)
            for key in SIGNAL_KEYS if hasattr(row, key)
        }
        repair_parts = []
        if bool(row.geom_anamorphic):
            repair_parts.append(f"pixel aspect {float(row.geom_sample_aspect):.4f}")
        if bool(row.geom_rotated):
            repair_parts.append(f"{int(row.geom_quarter_turns) * 90}° turn")
        signals.update(
            {
                "vendor": f"V{row.vendor_id}",
                "stored_raster": f"{int(row.width)}x{int(row.height)}",
                "shown_raster": str(row.geom_display_raster),
                "repair_applied": " + ".join(repair_parts) if repair_parts else "none",
                "fps": round(fps, 3),
                "flags_fired": str(row.flags_fired) or "none",
                "audio": "empty WAV" if bool(row.audio_empty) else "present",
                "timebase": "consistent" if bool(row.timebase_consistent)
                            else f"drifts {float(row.timebase_index_drift_s):+.1f} s",
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
                "selection_reason": item["selection_reason"] + reason_suffix,
                "signals_json": json.dumps(signals, sort_keys=True),
                "render_policy": "render",
            }
        )

    for index, record in enumerate(records):
        record["clip_id"] = (f"{record['sample_group']}_{index:03d}_{record['file_id']}"
                             f"_f{int(record['start_frame']):07d}")
        record["review_item_id"] = f"r7_{index:03d}"

    manifest = pd.DataFrame(records, columns=MANIFEST_COLUMNS)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(manifest_path, index=False)

    summary = {
        "scanned_files": int(len(scan)),
        "eligible_for_a_clip": int(len(eligible)),
        "clips": len(records),
        "by_group": manifest.sample_group.value_counts().to_dict(),
        "cell_population": eligible.loc[eligible.from_random_draw]
                           .groupby(["vendor_id", "cell"]).size()
                           .rename("files").reset_index().to_dict("records"),
        "rare_raster_files_added": int((~scan.from_random_draw).sum()),
        "draw_counts": counts,
        "rasters": manifest.signals_json.str.extract(r'"stored_raster": "([^"]+)"')[0]
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
    (report_dir / "vendor_r7_sample.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True, default=str), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True, default=str))


if __name__ == "__main__":
    main()

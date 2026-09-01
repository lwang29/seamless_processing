#!/usr/bin/env python3
"""Draw the Round-12 V00 gallery: adjudication asks, then a fresh stratified sample.

Round 7's asks were answered, and the answers moved four thresholds. This round
checks the moved thresholds and asks the two questions the new measurements
raised.

**The asks.** FM1's V03 cut went from 0.43 to 0.54 on 58 hand labels, so the
posture ladder is redrawn tight around the new boundary rather than spread over
the whole range -- that is where a wrong cut now costs something. The audio
ladder is new: a far-field or shared microphone is what the reviewer heard as
glitchy, it is measurable as poor separation between a speaker's own voice and
their partner's, and it separates the room-camera rasters cleanly *as
populations* while overlapping badly per file. Labels decide whether it can
become a detector or stays a signal.

**The sample.** A plain random draw, stratified by vendor and by what the
pipeline said -- pass, or each flag on its own, or several at once. Random
position within the file, except for FM2, where the clip is centred on the
longest continuous stretch the SMPL-H validity flag calls untrusted.
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
    "vad_s",
    "fm1_hip_flexion_deg_p50", "fm1_hip_flexion_deg_p25", "fm1_hip_flexion_deg_p75",
    "fm1_knee_between_torso_p50", "fm1_shin_verticality_p50", "fm1_ankles_visible_frac",
    "fm2_smplh_valid_frac", "fm3_static_frac", "reproj_shoulder_widths_p50",
    "observed_duration_s", "unit_flagged_frac", "geom_body_roll_deg",
    "timebase_index_drift_s", "audio_voice_isolation_db", "audio_speech_level_db",
    "audio_spectral_flatness",
)
# Round 12 turns back to V00, and the gap there is not the cut -- it is what
# happens above it. All 66 V00 labels come from the flagged pool or the
# adjudicate pool, so they sit near 0.43 by construction, and FM1's
# false-negative rate over the *comfortably passing* population has never been
# measured. On V03 that is exactly where the problem lived: seated participants
# reading 0.435 to 0.586, well clear of the cut. These bands sweep the pass side.
PASS_BANDS = ((0.43, 0.50), (0.50, 0.57), (0.57, 0.64), (0.64, 0.80))
# And a narrower ladder just below the cut, where a false positive would live.
REJECT_BANDS = ((0.30, 0.37), (0.37, 0.43))
POSTURE_MEASURE = "fm1_knee_between_torso_p50"
CELLS = ("pass", "fm0_unusable_source", "fm1_sitting", "fm2_smplh_invalid",
         "fm3_static_hands", "fm4_audio_dead", "multi")


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
    parser.add_argument("--config", type=Path, default=Path("configs/v00_r12_detect.yaml"))
    parser.add_argument("--gallery-config", type=Path,
                        default=Path("configs/audio_rule_r13_review.yaml"))
    parser.add_argument("--rare-config", type=Path, default=None,
                        help="unused for V00: all 41,205 eligible files are 1080x1920")
    args = parser.parse_args()

    worktree = _worktree()
    gallery = yaml.safe_load((worktree / args.gallery_config).read_text(encoding="utf-8"))
    detect = yaml.safe_load((worktree / args.config).read_text(encoding="utf-8"))
    # Both populations: this round's question spans all four vendors, and the
    # joined frame already carries the VAD measurement for each.
    joined = worktree / "outputs/session3/vad_seconds/joined.parquet"
    if joined.exists():
        scan = pd.read_parquet(joined)
    else:
        scan = pd.concat([
            pd.read_parquet(worktree / p / "scan_with_flags.parquet")
            for p in ("outputs/session3/v00_fm_scan_r12",
                      "outputs/session3/vendors_fm_scan_r11")
        ], ignore_index=True)
    scan = scan.drop_duplicates(subset="file_id").reset_index(drop=True)
    scan["from_random_draw"] = True
    rare_path = (worktree / str(
        yaml.safe_load((worktree / args.rare_config).read_text(encoding="utf-8"))
        ["outputs"]["root"]).lstrip("./") / "scan_with_flags.parquet") if args.rare_config else None
    if rare_path is not None and rare_path.exists():
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
    # Seconds of released speech, measured over these same scans by
    # scripts/probe_vad_seconds.py so no audio has to be re-read.
    if "vad_s" not in scan.columns:
        scan["vad_s"] = float("nan")
    scan["vad_empty"] = scan.vad_s.fillna(1.0) < 0.5
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

    def take(pool: pd.DataFrame, count: int, group: str, reason: str = "",
             centre_on_invalid: bool = False) -> int:
        reason = reason or group
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

    # ---- The audio rule: where the level floor and the released VAD disagree
    # FM4's absolute level floor cannot tell "speaks rarely" from "no voice at
    # all" -- all six of its V00 firings were the former, which is why it is now
    # off for V00. Seconds of released speech can: every confirmed-dead file has
    # zero. Before swapping one rule for the other, the files they disagree about
    # have to be looked at, in both directions.
    per_side = int(sampling["disagree_per_side"])
    vad_empty = eligible.vad_empty.fillna(False).astype(bool)
    fm4 = eligible.fm4_audio_dead.fillna(False).astype(bool)
    # V00's FM4 is retired, so use the level rule directly there rather than the
    # flag, or every V00 file would look like a disagreement.
    level_rule = (eligible.audio_speech_level_db < -55.0) & (eligible.audio_spectral_flatness > 0.05)
    for vendor in sorted(eligible.vendor_id.unique()):
        v = eligible.vendor_id.eq(vendor)
        counts[f"ask_vad_only_{vendor}"] = take(
            eligible.loc[v & vad_empty & ~level_rule], per_side, "ask_vad_only",
            f"V{vendor}: no released speech at all, but the level floor is silent",
        )
        counts[f"ask_level_only_{vendor}"] = take(
            eligible.loc[v & ~vad_empty & level_rule], per_side, "ask_level_only",
            f"V{vendor}: the level floor fires, but the file has released speech in it",
        )
    # Both rules agree: the control, so the asks are not all edge cases.
    counts["ask_both_agree"] = take(
        eligible.loc[vad_empty & level_rule], int(sampling["per_ask"]), "ask_both_agree",
        "both rules call this dead",
    )

    # ---- The stratified sample ----------------------------------------------
    # V00 has both recording conditions, so the sample is split by label as well
    # as by outcome -- that is the comparison this vendor's earlier rounds used.
    per_cell = int(sampling["per_label_per_cell"])
    for label in sorted(eligible.vendor_id.unique()):
        for cell in CELLS:
            pool = eligible.loc[
                eligible.from_random_draw & eligible.vendor_id.eq(label) & eligible.cell.eq(cell)
            ]
            group = f"v{label}_{cell}"
            counts[group] = take(
                pool, per_cell, group,
                f"random draw from V{label} files the pipeline sorted into '{cell}'",
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
                "audio_status": str(getattr(row, "audio_status", "")) or "unknown",
                "vendor": f"V{row.vendor_id}",
                "stored_raster": f"{int(row.width)}x{int(row.height)}",
                "shown_raster": str(row.geom_display_raster),
                "repair_applied": " + ".join(repair_parts) if repair_parts else "none",
                "fps": round(fps, 3),
                "flags_fired": str(row.flags_fired) or "none",
                "audio": "empty WAV" if bool(row.audio_empty) else "present",
                "timebase": "consistent" if bool(row.timebase_consistent)
                            else f"drifts {float(row.timebase_index_drift_s):+.1f} s",
                "fm0_reasons": str(getattr(row, "fm0_reasons", "") or "none"),
                "fm1_scope": (
                    f"this file (session is {float(row.unit_flagged_frac):.0%} seated"
                    f" over {int(row.unit_files)} files)"
                    if str(getattr(row, "fm1_sitting_scope", "session")) == "file"
                    else f"session majority over {int(row.unit_files)} files"
                ),
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
        record["review_item_id"] = f"r13_{index:03d}"

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
    (report_dir / "audio_rule_r13_sample.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True, default=str), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True, default=str))


if __name__ == "__main__":
    main()

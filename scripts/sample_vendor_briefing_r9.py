#!/usr/bin/env python3
"""Draw the Round-9 vendor gallery: adjudication asks, then a fresh stratified sample.

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
    "fm1_knee_between_torso_p50", "fm1_shin_verticality_p50", "fm1_ankles_visible_frac",
    "fm2_smplh_valid_frac", "fm3_static_frac", "reproj_shoulder_widths_p50",
    "observed_duration_s", "unit_flagged_frac", "geom_body_roll_deg",
    "timebase_index_drift_s", "audio_voice_isolation_db", "audio_speech_level_db",
    "audio_spectral_flatness",
)
# Round 9: FM1 on V03 now reads **hip flexion**, so the ladder moves with it.
# On 90 labels, standing files bottom out at 136 degrees and seated ones top out
# at 152, and the cut sits at 146. These bands bracket that gap.
POSTURE_BANDS = ((120.0, 136.0), (136.0, 146.0), (146.0, 152.0), (152.0, 162.0))
POSTURE_MEASURE = "fm1_hip_flexion_deg_p50"
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
    parser.add_argument("--config", type=Path, default=Path("configs/vendors_r9_detect.yaml"))
    parser.add_argument("--gallery-config", type=Path,
                        default=Path("configs/vendors_r9_review.yaml"))
    parser.add_argument("--rare-config", type=Path,
                        default=Path("configs/vendors_r9_rare_detect.yaml"),
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

    # ---- Ask 1: is FM1's new V03 cut of 0.54 in the right place? -----------
    per_band = int(sampling["posture_per_band"])
    ladder = eligible.loc[
        eligible.vendor_id.eq("03") & eligible[POSTURE_MEASURE].notna()
        & eligible.fm1_shin_verticality_p50.gt(0)   # shins the right way up, so the
    ]                                               # label is about posture, not framing
    for low, high in POSTURE_BANDS:
        band = ladder.loc[ladder[POSTURE_MEASURE].between(low, high, inclusive="left")]
        counts[f"ask_posture_{low:.0f}"] = take(
            band, per_band, "ask_posture_v03",
            f"V03, hip flexion {low:.0f}-{high:.0f} degrees — around the new 146 cut",
        )

    # ---- Ask 2: did excluding the room-camera rasters close the audio problem?
    # Voice isolation was refuted as a per-file detector by the Round-8 answers,
    # so this asks the question that is left: is there any unusable audio among
    # the files that now *survive*? Drawn from survivors only, spread across
    # vendors, with no reference to the isolation measure at all.
    survivors = eligible.loc[eligible.passes_all.fillna(False).astype(bool)]
    per_vendor_audio = int(sampling["audio_per_vendor"])
    for vendor in sorted(survivors.vendor_id.unique()):
        counts[f"ask_audio_{vendor}"] = take(
            survivors.loc[survivors.vendor_id.eq(vendor)], per_vendor_audio, "ask_audio",
            f"V{vendor}, survives the whole pipeline — is the audio usable?",
        )

    # ---- Ask 3: the two padded rasters, cropped for real this time ----------
    for raster in sampling["cropped_rasters"]:
        counts[f"ask_crop_{raster}"] = take(
            eligible.loc[eligible.raster.eq(raster)], int(sampling["per_odd_raster"]),
            "ask_cropped",
            f"{raster}, with the black columns actually removed this round",
        )

    # ---- The stratified sample ----------------------------------------------
    per_cell = int(sampling["per_vendor_per_cell"])
    for vendor in sorted(eligible.vendor_id.unique()):
        for cell in CELLS:
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
        record["review_item_id"] = f"r9_{index:03d}"

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
    (report_dir / "vendor_r9_sample.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True, default=str), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True, default=str))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Build a review gallery stratified by cohort, pass/flag, failure mode, and label.

The gallery has one job: let the reviewer confirm that each detector fires on
what it claims to fire on. So every failure mode gets a guaranteed quota even
when it is rare, and the pass group is a plain random draw so the "good" side
stays an honest base rate rather than a curated best-of.

Round 4 splits the gallery into two cohorts the reviewer asked to keep separate:

* **recheck** — the same files as the Round-3 gallery, so the detector changes
  can be judged on clips already inspected once. Each card carries the verdict
  it had in Round 3 alongside the verdict it has now.
* **fresh** — files never shown before, drawn from the same scan but disjoint at
  the file level and preferring participants the recheck cohort does not use.

A third group, **fm1_adjudicate**, is not a detector demonstration at all: it is
the list of participants the FM1 rule cannot confidently classify, one clip each,
which the reviewer offered to settle by eye.

Clips are 30-second excerpts of files whose verdict was decided over their whole
length. For FM2 the excerpt is anchored on the frame the scanner recorded as the
start of the worst violation, otherwise the reviewer would be shown thirty
seconds that look fine while the detector fired elsewhere; the filmstrip carries
the rest of the recording.
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



FLAGS = ["fm1_sitting", "fm2_smplh_invalid", "fm3_static_hands"]
MANIFEST_COLUMNS = [
    "review_item_id", "clip_id", "file_id", "source_relbase",
    "partner_source_relbase", "sample_group", "cohort", "label_stratum",
    "start_frame", "vendor", "label", "split", "activity_type",
    "selection_reason", "signals_json", "render_policy",
]
SIGNAL_KEYS = (
    "fm1_knee_between_torso_p50", "fm1_hip_flexion_deg_p50",
    "fm1_shin_verticality_p50", "fm1_leg_over_torso_p50",
    "unit_flagged_frac", "unit_files", "unit_hip_flexion_min", "unit_knee_between_min",
    "fm2_smplh_valid_frac", "reproj_shoulder_widths_p50", "reproj_px_p50",
    "fm2_in_frame_and_valid_frac", "fm2_violation_run_max_frames",
    "fm2_violation_runs", "fm2_windowed_pass_frac",
    "fm2_vertex_worst_inset_min_px", "fm2_surface_correction_px_p50",
    "fm2_pass_frac_round3_joints_only",
    "fm3_static_frac", "observed_duration_s",
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


def _fm2_anchor(row: Any, lead_in_s: float) -> int | None:
    """Clip start that puts the worst FM2 violation just after the lead-in.

    The scanner records ``fm2_longest_violation_start_frame`` as it goes, so this
    needs no forward kinematics and — more to the point — there is no second
    implementation of the rule here that could disagree with the detector about
    which frames are bad.
    """

    start = getattr(row, "fm2_longest_violation_start_frame", None)
    if start is None or not np.isfinite(float(start)) or int(start) < 0:
        return None
    return max(0, int(start) - int(round(lead_in_s * float(row.fps))))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/v00_fm_detect.yaml"))
    parser.add_argument("--scan", type=Path)
    parser.add_argument("--gallery-config", type=Path, default=Path("configs/v00_fm_review.yaml"))
    args = parser.parse_args()

    worktree = _worktree()
    gallery = yaml.safe_load((worktree / args.gallery_config).read_text(encoding="utf-8"))
    detect = yaml.safe_load((worktree / args.config).read_text(encoding="utf-8"))
    scan_path = args.scan or (
        worktree / str(detect["outputs"]["root"]).lstrip("./") / "scan_with_flags.parquet"
    )
    scan = pd.read_parquet(scan_path)
    manifest_path = worktree / str(gallery["outputs"]["manifest"]).lstrip("./")
    report_dir = worktree / str(gallery["outputs"]["report_dir"]).lstrip("./")
    report_dir.mkdir(parents=True, exist_ok=True)
    sampling = gallery["sampling"]
    seed = int(sampling["seed"])
    clip_duration_s = float(gallery["clip"]["duration_s"])
    lead_in_s = float(sampling.get("fm2_anchor_lead_in_s", 5.0))

    # The Round-3 gallery's files define the recheck cohort. Reading the previous
    # manifest rather than re-deriving it means the cohort is exactly the set the
    # reviewer already looked at, whatever each clip was originally there to
    # show.
    previous_path = worktree / str(sampling["previous_manifest"]).lstrip("./")
    previous = pd.read_csv(previous_path)
    previous_groups = dict(zip(previous.file_id, previous.sample_group))
    recheck_ids = set(previous.file_id)

    # A 30-second clip needs 30 seconds of analysed frames to sit in.
    render_frames_needed = {
        row.file_id: int(round(clip_duration_s * float(row.fps)))
        for row in scan.itertuples(index=False)
    }
    scan = scan.loc[
        [
            int(row.frames_analyzed) >= render_frames_needed[row.file_id]
            for row in scan.itertuples(index=False)
        ]
    ].copy()

    chosen: list[dict[str, Any]] = []
    used: set[str] = set()

    def verdict_group(row: Any) -> str:
        if bool(row.passes_all):
            return "pass"
        if int(row.flag_count) >= 2:
            return "multi"
        fired = str(row.flags_fired) or "undetermined"
        return fired

    def take(subset: pd.DataFrame, count: int, group: str, cohort: str, reason: str) -> int:
        available = subset.loc[~subset.file_id.isin(used)].copy()
        if available.empty or count <= 0:
            return 0
        available["draw"] = [_unit(seed, group, reason, fid) for fid in available.file_id]
        taken = available.nsmallest(min(count, len(available)), "draw")
        for row in taken.itertuples(index=False):
            used.add(row.file_id)
            chosen.append(
                {"row": row, "sample_group": group, "cohort": cohort, "selection_reason": reason}
            )
        return len(taken)

    # --- cohort 1: every file from the Round-3 gallery, re-judged ---------------
    recheck = scan.loc[scan.file_id.isin(recheck_ids)]
    for row in recheck.itertuples(index=False):
        used.add(row.file_id)
        previous_group = previous_groups.get(row.file_id, "unknown")
        chosen.append(
            {
                "row": row,
                "sample_group": f"recheck_{verdict_group(row)}",
                "cohort": "recheck",
                "selection_reason": (
                    f"shown in Round 3 as {previous_group}; re-judged under the Round-4 detectors"
                ),
                "round3_group": previous_group,
            }
        )
    missing = sorted(recheck_ids - set(recheck.file_id))

    # --- cohort 2: fresh files, stratified the same way ------------------------
    # Participants already represented in the recheck cohort teach us less: the
    # reviewer established that a participant's posture is constant, so a second
    # file from the same person mostly re-tests the same judgement.
    seen_participants = set(recheck.participant_id)
    fresh_pool = scan.loc[~scan.file_id.isin(used)]

    def fresh_take(pool: pd.DataFrame, count: int, group: str, reason: str) -> None:
        unseen = pool.loc[~pool.participant_id.isin(seen_participants)]
        got = take(unseen, count, group, "fresh", f"{reason}; participant not in the recheck cohort")
        take(pool, count - got, group, "fresh", reason)

    for flag in FLAGS:
        quota = int(sampling["flagged_per_mode_per_label"])
        for label in ("naturalistic", "improvised"):
            pool = fresh_pool.loc[fresh_pool[flag].eq(True) & fresh_pool.label.eq(label)]
            single = pool.loc[pool.flag_count.eq(1)]
            fresh_take(single, quota, f"fresh_flagged_{flag}", f"{flag}: only this mode fired")
            still = quota - sum(
                1 for c in chosen
                if c["sample_group"] == f"fresh_flagged_{flag}" and c["row"].label == label
            )
            fresh_take(pool, still, f"fresh_flagged_{flag}", f"{flag}: with other modes")
    for label in ("naturalistic", "improvised"):
        fresh_take(
            fresh_pool.loc[fresh_pool.flag_count.ge(2) & fresh_pool.label.eq(label)],
            int(sampling["multi_flag_per_label"]), "fresh_flagged_multi",
            "two or more detectors fired",
        )
    for label in ("naturalistic", "improvised"):
        fresh_take(
            fresh_pool.loc[fresh_pool.passes_all & fresh_pool.label.eq(label)],
            int(sampling["pass_per_label"]), f"fresh_pass_{label}",
            "random draw from files passing all three detectors",
        )

    # --- cohort 3: the FM1 questions for the reviewer --------------------------
    # The reviewer offered to settle participants the rule cannot classify. This
    # group is that worklist, not a detector demonstration, and it is ordered by
    # how much the answer would change. One clip per participant: their posture
    # is constant, so one look settles all of their files.
    def adjudicate(pool: pd.DataFrame, question: str, limit: int | None = None) -> None:
        units = sorted(pool.sitting_unit.unique(), key=str)
        for index, unit in enumerate(units):
            if limit is not None and index >= limit:
                break
            group = pool.loc[pool.sitting_unit.eq(unit)]
            # Show the file that reads most strongly as seated. If that one is a
            # standing person, so is the rest of the unit.
            ordered = group.sort_values("fm1_hip_flexion_deg_p50")
            take(
                ordered.head(1), 1, "fm1_adjudicate", "adjudicate",
                f"{unit} (participant @ session): {question} "
                f"({int(ordered.unit_files_flagged.iloc[0])} of "
                f"{int(ordered.unit_files.iloc[0])} scanned files in this session read as "
                f"seated; hip flexion {ordered.fm1_hip_flexion_deg_p50.iloc[0]:.1f} deg "
                f"against a cut of 118, knee-between "
                f"{ordered.fm1_knee_between_torso_p50.iloc[0]:.3f} against a cut of 0.45) "
                f"— is this person sitting or standing?",
            )

    # (a) Highest stakes: units the rule now calls seated that are ALSO
    #     short-legged, the exact population that produced every Round-3 false
    #     positive. If these are standing, the mechanism is not fixed.
    short_legged = pd.to_numeric(scan.fm1_leg_over_torso_p50, errors="coerce") < 0.85
    adjudicate(
        scan.loc[short_legged & scan.fm1_sitting_file.eq(True)],
        "flagged as SEATED and short-legged, which is exactly what broke Round 3.",
    )
    # (b) Near the cut. 56.3% of units hold a single file and so can never
    #     disagree with themselves, which means the majority rule reports perfect
    #     confidence on more than half the corpus. These are the units where a
    #     threshold move would actually change the answer.
    adjudicate(
        scan.loc[scan.unit_sitting_near_cut.eq(True) & ~scan.file_id.isin(used)],
        "just above the cut — a small threshold move would call this SEATED.",
        limit=int(sampling.get("fm1_near_cut_clips", 18)),
    )
    # (d) Negative controls. Every confirmed-standing label so far comes from the
    #     1.4% of the corpus with short legs, so the rule's false-positive rate on
    #     ordinary standing participants is unmeasured. Without controls the
    #     reviewer's answers cannot be scored: if they say "standing" to all of
    #     (a)-(c), that could mean the rule over-fires or that they are being
    #     asked leading questions.
    controls = scan.loc[
        scan.fm1_sitting.eq(False)
        & ~short_legged
        & ~scan.file_id.isin(used)
    ]
    take(
        controls, int(sampling.get("fm1_control_clips", 10)), "fm1_adjudicate", "adjudicate",
        "CONTROL: the rule says standing and nothing about this file is unusual. "
        "Included unlabelled so the answers to the other adjudication clips can be "
        "scored — is this person sitting or standing?",
    )

    records: list[dict[str, Any]] = []
    for item in chosen:
        row = item["row"]
        fps = float(row.fps)
        render_frames = int(round(clip_duration_s * fps))
        span = max(0, int(row.frames_analyzed) - render_frames)
        anchor: int | None = None
        if bool(row.fm2_smplh_invalid):
            anchor = _fm2_anchor(row, lead_in_s)
        start = anchor if anchor is not None else int(_unit(seed, "start", row.file_id) * (span + 1))
        start = max(0, min(start, span))
        signals = {
            key: (float(getattr(row, key)) if pd.notna(getattr(row, key, None)) else None)
            for key in SIGNAL_KEYS
            if hasattr(row, key)
        }
        signals.update(
            {
                "flags_fired": str(row.flags_fired),
                "flag_count": int(row.flag_count),
                "fm1_sitting_file": bool(row.fm1_sitting_file),
                "fm1_sitting_participant": bool(row.fm1_sitting),
                "fm1_reasons": str(row.fm1_sitting_reasons),
                "fm1_unit": str(row.sitting_unit),
                "fm1_unit_basis": str(row.unit_sitting_basis),
                "clip_anchored_on_fm2_violation": anchor is not None,
            }
        )
        if "round3_group" in item:
            signals["round3_gallery_group"] = item["round3_group"]
        records.append(
            {
                "review_item_id": "", "clip_id": "",
                "file_id": row.file_id, "source_relbase": row.source_relbase,
                "partner_source_relbase": row.partner_source_relbase,
                "sample_group": item["sample_group"], "cohort": item["cohort"],
                "label_stratum": row.label, "start_frame": start,
                "vendor": "V00", "label": row.label, "split": row.split,
                "activity_type": row.interaction_type,
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
        record["review_item_id"] = f"r4_{index:03d}"

    manifest = pd.DataFrame(records, columns=MANIFEST_COLUMNS)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(manifest_path, index=False)
    for group in sorted(manifest.sample_group.unique()):
        manifest.loc[manifest.sample_group.eq(group)].to_csv(
            manifest_path.with_name(f"v00_fm_review_{group}.csv"), index=False
        )
    for cohort in sorted(manifest.cohort.unique()):
        manifest.loc[manifest.cohort.eq(cohort)].to_csv(
            manifest_path.with_name(f"v00_fm_review_cohort_{cohort}.csv"), index=False
        )

    summary = {
        "scanned_files": int(len(scan)),
        "clips": len(records),
        "clip_duration_s": clip_duration_s,
        "by_cohort": manifest.cohort.value_counts().to_dict(),
        "by_group": manifest.sample_group.value_counts().to_dict(),
        "by_group_and_label": [
            {"sample_group": group, "label": label, "clips": int(count)}
            for (group, label), count in manifest.groupby(["sample_group", "label"]).size().items()
        ],
        "recheck_requested": len(recheck_ids),
        "recheck_rendered": int(manifest.cohort.eq("recheck").sum()),
        "recheck_missing_file_ids": missing,
        "fresh_files_disjoint_from_recheck": bool(
            not (set(manifest.loc[manifest.cohort.eq("fresh"), "file_id"]) & recheck_ids)
        ),
        "fresh_clips_from_unseen_participants": int(
            manifest.loc[manifest.cohort.eq("fresh"), "selection_reason"]
            .str.contains("not in the recheck cohort").sum()
        ),
        "anchored_on_fm2_violation": int(
            manifest.signals_json.str.contains('"clip_anchored_on_fm2_violation": true').sum()
        ),
        "distinct_participants": int(
            scan.set_index("file_id").loc[manifest.file_id].participant_id.nunique()
        ),
        "seed": seed,
        **_git(worktree),
    }
    (report_dir / "gallery_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True, default=str), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True, default=str))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Apply thresholds to the V00 failure-mode scan and report what fires.

Nothing here re-reads media. The scan produced continuous measurements; this
applies the cut points from the config, and re-tuning any of them is a config
edit plus one re-run of this script.

Reports what is needed to judge the three detectors: per-label flag rates,
detector overlap, surviving hours, the full distribution of every underlying
measurement, and — new in Round 4 — the participant-level FM1 verdict and the
list of participants the rule cannot confidently classify.
"""

from __future__ import annotations

import argparse
import glob
import json
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from seamless_curation.v00_detectors import (  # noqa: E402
    Thresholds,
    apply_thresholds,
    session_sitting_verdicts,
)


# Order matters only for reading: FM0 first because it decides whether the other
# three are worth consulting at all.
FLAGS = ["fm0_unusable_source", "fm1_sitting", "fm2_smplh_invalid",
         "fm3_static_hands", "fm4_audio_dead"]
# Every continuous quantity a threshold reads, plus the context needed to
# re-tune it.
DISTRIBUTIONS = [
    # FM1 primary: FK posture, thresholded.
    "fm1_knee_between_torso_p50", "fm1_hip_flexion_deg_p50", "fm1_knee_flexion_deg_p50",
    # FM1 secondary: measured and reported, never thresholded. The 2D leg ratios
    # were the Round-3 criteria and scored 0/15 on the reviewer's inspection;
    # they stay in the record so the retirement can be re-checked.
    "fm1_shin_verticality_p50", "fm1_knee_spread_over_torso_p50",
    "fm1_leg_over_torso_p50", "fm1_thigh_over_shin_p50",
    "fm1_thigh_verticality_p50", "fm1_hip_ankle_vertical_over_torso_p50",
    # FM2.
    "fm2_in_frame_frac", "fm2_smplh_valid_frac", "fm2_in_frame_and_valid_frac",
    "fm2_violation_run_max_frames", "fm2_violation_runs",
    "fm2_out_of_frame_point_frac", "fm2_windowed_pass_frac",
    "fm2_term_vertices_frame_frac", "fm2_term_joints_frame_frac",
    "fm2_term_smplh_invalid_frame_frac", "fm2_term_coco_body17_frame_frac",
    "fm2_term_coco_left_hand_frame_frac", "fm2_term_coco_right_hand_frame_frac",
    "fm2_vertex_worst_inset_px_p5", "fm2_vertex_worst_inset_min_px",
    "fm2_surface_correction_px_p50", "fm2_surface_correction_px_p95",
    "fm2_pass_frac_round3_joints_only", "fm2_pass_frac_vertices_only",
    "fm2_pass_frac_coco_only", "fm2_pass_frac_no_coco",
    "fm2_hands_out_of_frame_frame_frac", "fm2_feet_out_of_frame_frame_frac",
    "fm2_head_out_of_frame_frame_frac",
    # Direct SMPL-H fit quality: how far the projected body lands from the
    # released 2D keypoints. This is what FM2's validity flag is a proxy for.
    "reproj_px_p50", "reproj_px_p95", "reproj_shoulder_widths_p50",
    "reproj_shoulder_widths_p95", "reproj_shoulder_widths_on_valid_p50",
    "reproj_shoulder_widths_on_invalid_p50", "reproj_shoulder_width_px",
    # FM3 — unchanged; the reviewer confirmed the calibration.
    "fm3_static_frac", "fm3_worse_wrist_offset_shoulder_widths_p50",
    "fm3_static_frac_at_r0p05", "fm3_static_frac_at_r0p15",
    "fm3_static_frac_at_r0p2", "fm3_static_frac_at_r0p3",
    "tracker_centre_jump_max", "tracker_size_change_max",
    "observed_duration_s",
]
PERCENTILES = [1, 5, 10, 25, 50, 75, 90, 95, 99]

# Every file the reviewer has hand-labelled by eye, kept here so each re-tune
# reports its own confusion matrix instead of trusting a past one. 29 from the
# Round-3 gallery, 37 more from the Round-4 FM1 adjudication group.
LABELLED_SITTING = """
V00_S1685_I00000415_P1160 V00_S0555_I00000134_P0698 V00_S0360_I00000371_P0485
V00_S0360_I00000373_P0485 V00_S0936_I00000477_P1030 V00_S0446_I00000125_P0489
V00_S0555_I00000126_P0698 V00_S0700_I00000579_P0844A V00_S2059_I00001184_P1316A
V00_S1921_I00000264_P1209 V00_S0200_I00000307_P0003A V00_S0112_I00000495_P0162
V00_S0112_I00000483_P0163 V00_S0200_I00000309_P0271
V00_S0216_I00000515_P0293 V00_S0284_I00000495_P0385 V00_S0521_I00000461_P0566
V00_S0521_I00000135_P0663 V00_S0043_I00000540_P0065
""".split()
LABELLED_STANDING = """
V00_S0342_I00000477_P0456 V00_S0728_I00000498_P0876 V00_S0581_I00000135_P0732
V00_S0353_I00000482_P0474 V00_S0675_I00000377_P0831 V00_S1781_I00000712_P1099A
V00_S1621_I00001114_P1099A V00_S1564_I00001111_P1099A V00_S1628_I00001175_P1147A
V00_S1880_I00001019_P1147A V00_S1724_I00001133_P1147A V00_S1626_I00000995_P1147A
V00_S1765_I00001191_P1147A V00_S1532_I00001037_P1099A V00_S1761_I00000724_P1147A
V00_S0113_I00000487_P0008 V00_S0098_I00000372_P0143 V00_S0290_I00000495_P0394
V00_S0571_I00000313_P0719 V00_S0939_I00000514_P1031 V00_S0287_I00000128_P0011
V00_S1163_I00001062_P0012 V00_S0917_I00000132_P0012A V00_S0058_I00000537_P0027A
V00_S0172_I00000375_P0027A V00_S0127_I00000140_P0029A V00_S1848_I00000316_P0047
V00_S0185_I00000386_P0048 V00_S0036_I00000130_P0058 V00_S0044_I00000313_P0059A
V00_S0273_I00000578_P0059A V00_S0143_I00000488_P0071 V00_S1322_I00000007_P0071
V00_S1395_I00000607_P0071 V00_S0094_I00000130_P0073 V00_S0048_I00000377_P0074
V00_S0049_I00000486_P0076 V00_S1904_I00001142_P0946A V00_S1729_I00001019_P0933A
V00_S0190_I00000125_P0262 V00_S1132_I00000335_P1093 V00_S0110_I00000489_P0159
V00_S1478_I00000114_P1146 V00_S0991_I00001059_P0323A V00_S0313_I00000135_P0176
V00_S0697_I00000543_P0851 V00_S0778_I00000478_P0178
""".split()


def describe(series: pd.Series) -> dict[str, Any]:
    numeric = pd.to_numeric(series, errors="coerce").dropna()
    row: dict[str, Any] = {"n": int(numeric.size), "n_missing": int(series.isna().sum())}
    if numeric.empty:
        return row | {f"p{p}": float("nan") for p in PERCENTILES}
    row.update({f"p{p}": float(numeric.quantile(p / 100)) for p in PERCENTILES})
    row["min"] = float(numeric.min())
    row["max"] = float(numeric.max())
    row["mean"] = float(numeric.mean())
    return row


def sweep(frame: pd.DataFrame, column: str, cuts: list[float], *, above: bool) -> list[dict[str, Any]]:
    """Flag rate as a threshold moves, so re-tuning has a curve to read."""

    if column not in frame.columns:
        return []
    series = pd.to_numeric(frame[column], errors="coerce")
    usable = series.notna()
    rows: list[dict[str, Any]] = []
    for cut in cuts:
        fired = (series > cut) if above else (series < cut)
        row: dict[str, Any] = {"column": column, "cut": cut, "direction": "above" if above else "below"}
        for label in ("naturalistic", "improvised", "all"):
            mask = usable if label == "all" else (usable & frame.label.eq(label))
            row[f"{label}_frac"] = float((fired & mask).sum() / mask.sum()) if mask.any() else float("nan")
        rows.append(row)
    return rows


def confusion(frame: pd.DataFrame, column: str) -> dict[str, Any]:
    """How a verdict column scores against the reviewer's 29 hand labels."""

    truth = {f: True for f in LABELLED_SITTING} | {f: False for f in LABELLED_STANDING}
    subset = frame.loc[frame.file_id.isin(truth)].copy()
    subset["truth_sitting"] = subset.file_id.map(truth)
    predicted = subset[column].eq(True)
    actual = subset.truth_sitting.eq(True)
    return {
        "labelled_files_found": int(len(subset)),
        "labelled_files_expected": len(truth),
        "true_positive": int((predicted & actual).sum()),
        "false_negative": int((~predicted & actual).sum()),
        "false_positive": int((predicted & ~actual).sum()),
        "true_negative": int((~predicted & ~actual).sum()),
        "missed_sitting_files": sorted(subset.loc[~predicted & actual, "file_id"]),
        "wrongly_flagged_standing_files": sorted(subset.loc[predicted & ~actual, "file_id"]),
    }


def _rollup(flags: dict[str, Any], names: list[str]) -> dict[str, Any]:
    fired = [name for name in names if flags.get(name) is True]
    unknown = [name for name in names if flags.get(name) is None]
    return {
        "flagged": bool(fired),
        "flag_count": len(fired),
        "flags_fired": "+".join(sorted(fired)),
        "flags_undetermined": "+".join(sorted(unknown)),
        # "Passes" requires every detector to have actually run and said no.
        "passes_all": bool(not fired and not unknown),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/v00_fm_detect.yaml"))
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()

    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    thresholds = Thresholds(**config["thresholds"])
    out = args.out or Path(str(config["outputs"]["root"]))
    shard_dir = out / str(config["outputs"]["shards"])
    paths = sorted(glob.glob(str(shard_dir / "*.parquet")))
    if not paths:
        raise SystemExit(f"no shards under {shard_dir}")
    scan = pd.concat([pd.read_parquet(path) for path in paths], ignore_index=True)
    out.mkdir(parents=True, exist_ok=True)

    ok = scan.loc[scan.status.eq("ok")].copy()
    records = ok.to_dict("records")
    flags = pd.DataFrame([apply_thresholds(row, thresholds) for row in records])
    result = pd.concat([ok.reset_index(drop=True), flags], axis=1)

    # FM1's scope is per-vendor, because the assumption it rests on is. The
    # reviewer established that a V00 participant's posture is constant across
    # their recordings and measurement put the scope of that at the session: only
    # 1.1% of V00 multi-file sessions are mixed. V03 is 16.2% mixed, with a median
    # seated share of exactly half in those sessions, so a majority vote there is
    # close to a coin toss -- and on 108 labelled V03 files it loses both more
    # sitters and more standing files than the per-file reading. V03 therefore
    # uses its own file's verdict. Both columns are always written.
    verdicts = session_sitting_verdicts(records, thresholds)
    result = result.rename(columns={"fm1_sitting": "fm1_sitting_file"})
    verdict_frame = pd.DataFrame.from_dict(verdicts, orient="index").reset_index(drop=True)
    result["sitting_unit"] = (
        result.participant_id.astype(str) + "@" + result.session_id.astype(str)
    )
    result = result.merge(verdict_frame, on="sitting_unit", how="left")
    scope = result.vendor_id.astype(str).map(
        lambda v: thresholds.sitting_unit_scope_by_vendor.get(v, "session")
    )
    result["fm1_sitting_scope"] = scope
    result["fm1_sitting"] = np.where(
        scope.eq("file"), result["fm1_sitting_file"], result["unit_sitting"]
    )
    # np.where flattens None to NaN; put the three-valued logic back, because a
    # file whose posture could not be measured must not read as a clean pass.
    result["fm1_sitting"] = [
        None if pd.isna(v) else bool(v) for v in result["fm1_sitting"]
    ]
    rederived = pd.DataFrame([_rollup(row, FLAGS) for row in result[FLAGS].to_dict("records")])
    for column in ("flagged", "flag_count", "flags_fired", "flags_undetermined", "passes_all"):
        result[column] = rederived[column]
    result.to_parquet(out / "scan_with_flags.parquet", index=False)

    # Per-label flag rates.
    rate_rows: list[dict[str, Any]] = []
    for flag in [*FLAGS, "fm1_sitting_file", "unit_sitting_near_cut"]:
        row: dict[str, Any] = {"detector": flag}
        for label in ("naturalistic", "improvised", "all"):
            subset = result if label == "all" else result.loc[result.label.eq(label)]
            determined = subset[flag].notna()
            fired = subset[flag].eq(True)
            row[f"{label}_n"] = int(determined.sum())
            row[f"{label}_flagged"] = int(fired.sum())
            row[f"{label}_rate"] = float(fired.sum() / determined.sum()) if determined.any() else float("nan")
            row[f"{label}_undetermined"] = int((~determined).sum())
        rate_rows.append(row)
    pd.DataFrame(rate_rows).to_csv(out / "flag_rates_by_label.csv", index=False)

    # Overlap: how often detectors co-fire, and on which combinations.
    overlap_counts = result.flags_fired.value_counts().rename_axis("flags_fired").reset_index(name="clips")
    overlap_counts.to_csv(out / "flag_combinations.csv", index=False)
    pair_rows: list[dict[str, Any]] = []
    for first, second in combinations(FLAGS, 2):
        both = result[first].eq(True) & result[second].eq(True)
        either = result[first].eq(True) | result[second].eq(True)
        pair_rows.append(
            {
                "detector_a": first, "detector_b": second,
                "both": int(both.sum()), "either": int(either.sum()),
                "jaccard": float(both.sum() / either.sum()) if either.any() else float("nan"),
            }
        )
    pd.DataFrame(pair_rows).to_csv(out / "flag_pairwise_overlap.csv", index=False)

    # Distributions, overall and per label.
    dist_rows: list[dict[str, Any]] = []
    for column in DISTRIBUTIONS:
        if column not in result.columns:
            continue
        for label in ("naturalistic", "improvised", "all"):
            subset = result if label == "all" else result.loc[result.label.eq(label)]
            dist_rows.append({"column": column, "label": label, **describe(subset[column])})
    pd.DataFrame(dist_rows).to_csv(out / "measurement_distributions.csv", index=False)

    # Threshold sweeps for every cut the reviewer said they want to re-tune.
    sweeps: list[dict[str, Any]] = []
    sweeps += sweep(result, "fm1_knee_between_torso_p50",
                    [0.35, 0.38, 0.40, 0.41, 0.42, 0.43, 0.44, 0.45, 0.48, 0.50], above=False)
    sweeps += sweep(result, "fm1_hip_flexion_deg_p50", [105, 110, 114, 118, 122, 125], above=False)
    sweeps += sweep(result, "fm2_smplh_valid_frac", [0.90, 0.95, 0.99, 0.999, 1.0], above=False)
    sweeps += sweep(result, "reproj_shoulder_widths_p50", [0.10, 0.15, 0.20, 0.25, 0.30, 0.40], above=True)
    sweeps += sweep(result, "fm2_in_frame_and_valid_frac", [0.90, 0.95, 0.99, 0.999, 1.0], above=False)
    sweeps += sweep(result, "fm2_pass_frac_round3_joints_only", [0.99, 0.999, 1.0], above=False)
    sweeps += sweep(result, "fm2_pass_frac_vertices_only", [0.99, 0.999, 1.0], above=False)
    sweeps += sweep(result, "fm3_static_frac", [0.50, 0.60, 0.70, 0.75, 0.80, 0.90, 0.95], above=True)
    pd.DataFrame(sweeps).to_csv(out / "threshold_sweeps.csv", index=False)

    # The participants the FM1 rule cannot confidently classify. The reviewer
    # offered to adjudicate these by eye, so they are written out as a worklist.
    uncertain = (
        verdict_frame.loc[verdict_frame.unit_sitting_near_cut.eq(True)]
        .sort_values("unit_hip_flexion_min")
    )
    uncertain.to_csv(out / "fm1_units_near_the_cut.csv", index=False)
    verdict_frame.to_csv(out / "fm1_unit_verdicts.csv", index=False)

    # Surviving volume. Participant-hours, then dyad-hours as half of that,
    # which holds because both members of an interaction share a duration.
    hours = result.observed_duration_s.sum() / 3600
    pass_hours = result.loc[result.passes_all].observed_duration_s.sum() / 3600
    by_label = {
        label: {
            "files": int(subset.shape[0]),
            "pass_files": int(subset.passes_all.sum()),
            "pass_rate": float(subset.passes_all.mean()),
            "participant_hours": float(subset.observed_duration_s.sum() / 3600),
            "pass_participant_hours": float(
                subset.loc[subset.passes_all].observed_duration_s.sum() / 3600
            ),
            "hours_weighted_pass_rate": float(
                subset.loc[subset.passes_all].observed_duration_s.sum()
                / subset.observed_duration_s.sum()
            ),
        }
        for label, subset in result.groupby("label")
    }

    # What each FM2 term is worth, and what the Round-3 rule would have said on
    # exactly these files, so the change is a paired comparison and not two
    # numbers from two different runs.
    fm2_variants = {
        name: float((pd.to_numeric(result[column], errors="coerce") < 1.0).mean())
        for name, column in (
            ("round4_all_terms", "fm2_in_frame_and_valid_frac"),
            ("round3_joints_only", "fm2_pass_frac_round3_joints_only"),
            ("vertices_only", "fm2_pass_frac_vertices_only"),
            ("coco_only", "fm2_pass_frac_coco_only"),
            ("no_coco", "fm2_pass_frac_no_coco"),
        )
        if column in result.columns
    }
    round3_flagged = pd.to_numeric(
        result.get("fm2_pass_frac_round3_joints_only"), errors="coerce"
    ) < 1.0
    round4_flagged = pd.to_numeric(result.fm2_in_frame_and_valid_frac, errors="coerce") < 1.0

    summary: dict[str, Any] = {
        "thresholds": config["thresholds"],
        "scan_rows": int(len(scan)),
        "scan_rows_ok": int(len(result)),
        "scan_status_counts": scan.status.value_counts().to_dict(),
        "distinct_participants": int(result.participant_id.nunique()),
        "overall_pass_rate": float(result.passes_all.mean()),
        "overall_flag_rate": float(result.flagged.mean()),
        "clips_with_multiple_flags": int((result.flag_count >= 2).sum()),
        "flag_count_distribution": result.flag_count.value_counts().sort_index().to_dict(),
        "top_flag_combinations": overlap_counts.head(12).to_dict("records"),
        "undetermined_by_detector": {flag: int(result[flag].isna().sum()) for flag in FLAGS},
        "sample_hours": {
            "participant_hours_scanned": float(hours),
            "participant_hours_passing": float(pass_hours),
            "hours_weighted_pass_rate": float(
                result.loc[result.passes_all].observed_duration_s.sum()
                / result.observed_duration_s.sum()
            ),
        },
        "by_label": by_label,
        "fm1": {
            "file_level_flag_rate": float(result.fm1_sitting_file.eq(True).mean()),
            "unit_level_flag_rate": float(result.fm1_sitting.eq(True).mean()),
            "units_seated": int(verdict_frame.unit_sitting.eq(True).sum()),
            "units_total": int(len(verdict_frame)),
            "units_near_the_cut": int(len(uncertain)),
            "units_single_file": int((verdict_frame.unit_files == 1).sum()),
            "files_in_units_near_the_cut": int(result.unit_sitting_near_cut.eq(True).sum()),
            "verdict_basis_counts": verdict_frame.unit_sitting_basis.value_counts().to_dict(),
            "reason_counts": result.fm1_sitting_reasons.value_counts().to_dict(),
            "confusion_file_level_vs_29_labels": confusion(
                result.assign(_file_verdict=result.fm1_sitting_file), "_file_verdict"
            ),
            "confusion_participant_level_vs_29_labels": confusion(result, "fm1_sitting"),
        },
        "fm2": {
            "flag_rate_by_variant": fm2_variants,
            "round4_is_superset_of_round3": bool((round3_flagged & ~round4_flagged).sum() == 0),
            "files_round3_flagged_that_round4_passes": int((round3_flagged & ~round4_flagged).sum()),
            "files_round4_newly_flags": int((~round3_flagged & round4_flagged).sum()),
            "windowed_pass_frac_median": float(
                pd.to_numeric(result.fm2_windowed_pass_frac, errors="coerce").median()
            ),
            "surface_correction_px_median_of_file_medians": float(
                pd.to_numeric(result.get("fm2_surface_correction_px_p50"), errors="coerce").median()
            ),
        },
        "files_truncated": int(result.get("frames_truncated_from", pd.Series(dtype=float)).notna().sum()),
    }
    (out / "analysis_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True, default=str), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True, default=str)[:6000])


if __name__ == "__main__":
    main()

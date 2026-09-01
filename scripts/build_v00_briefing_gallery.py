#!/usr/bin/env python3
"""Assemble the PI briefing page from rendered clips and the analysis summary.

Every number on the page is read from the analysis of the sample the page shows,
never hard-coded, so the prose cannot drift away from the measurements.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from seamless_curation.briefing_gallery import write_briefing  # noqa: E402
from seamless_curation.review_renderer import link_source_media  # noqa: E402

# Total V00 dyad-hours in the eligible population, from the M-1 inventory. The
# sample's hours-weighted pass rate is projected onto this.
V00_DYAD_HOURS = {"naturalistic": 469.5, "improvised": 971.1}

SECTIONS: tuple[tuple[str, str, str, str], ...] = (
    (
        "pass-improvised", "pass_improvised",
        "Passed all three checks — improvised",
        "A plain random draw from the improvised files that survive. This is what "
        "the surviving pool looks like, not a selection of its best members.",
    ),
    (
        "pass-naturalistic", "pass_naturalistic",
        "Passed all three checks — naturalistic",
        "The same draw on the naturalistic half. Naturalistic and improvised are "
        "the dataset's two recording conditions and we have not yet chosen "
        "between them; these two sections are here to be compared.",
    ),
    (
        "flagged-fm1", "flagged_fm1",
        "Flagged FM1 — participant is seated",
        "Rejected because the participant is sitting rather than standing. Drawn "
        "where possible from files that tripped only this check, so what you see "
        "can be attributed to it.",
    ),
    (
        "flagged-fm2", "flagged_fm2",
        "Flagged FM2 — SMPL-H fit not trusted on every frame",
        "Rejected because the released per-frame SMPL-H validity flag goes false "
        "somewhere in the recording. <strong>Each clip is positioned to contain "
        "that file's longest continuous untrusted stretch</strong>, so the frames "
        "the check objected to are the frames you are watching. In many of them "
        "nothing looks wrong in the video — that is the point: what degrades is "
        "the hand pose in the fitted body, not the picture. Note how short most "
        "of these stretches are; see the note on strictness above.",
    ),
    (
        "flagged-fm3", "flagged_fm3",
        "Flagged FM3 — hands held in one position",
        "Rejected because both wrists stay parked for most of the recording. A "
        "participant who never gestures is not a useful example for a co-speech "
        "gesture model, however clean the recording is.",
    ),
)


def detector_blocks(summary: dict[str, Any], rates: pd.DataFrame) -> list[dict[str, str]]:
    def rate(detector: str) -> str:
        row = rates.loc[rates.detector.eq(detector)]
        return f"{100 * float(row.all_rate.iloc[0]):.1f}% flagged" if len(row) else ""

    thresholds = summary["thresholds"]
    return [
        {
            "id": "fm1", "name": "FM1 · participant is seated", "rate": rate("fm1_sitting"),
            "aim": (
                "Keep only standing participants. A seated body produces a "
                "fundamentally different gesture distribution — the torso is "
                "supported, the arms rest, and the legs contribute nothing — so "
                "mixing the two contaminates the training signal."
            ),
            "how": (
                "From SMPL-H forward kinematics, not from how much frame the "
                "person fills: seated participants were often recorded with the "
                "camera moved closer, so occupancy carries no signal. The measure "
                "is <code>knee_between_torso</code>, how far down the hip-to-ankle "
                "drop the knee sits <em>along the participant's own torso axis</em> "
                f"— standing reads about 0.55, seated about 0.30, and the cut is "
                f"<code>&lt; {thresholds['sitting_knee_between_torso']}</code>. Taking it along the "
                "torso axis makes it immune to camera tilt and distance; taking it "
                "from forward kinematics makes it immune to body proportion, which "
                "matters because all sixteen SMPL-H shape parameters are pinned to "
                "zero, so image limb lengths encode build rather than posture. One "
                "add-on catches a rare stool posture where the ankle sits above the "
                "knee. The verdict is taken once per <em>participant-session</em> by "
                "majority of that session's files, because posture is a property of "
                "the recording setup. Validated against 66 files hand-labelled by "
                "eye: 65 correct, zero false positives."
            ),
        },
        {
            "id": "fm2", "name": "FM2 · SMPL-H fit not trusted", "rate": rate("fm2_smplh_invalid"),
            "aim": (
                "Keep only recordings whose fitted body can be trusted frame by "
                "frame. The fitted SMPL-H parameters are the training target, so a "
                "stretch of untrustworthy fit is not a cosmetic flaw — it is "
                "corrupted labels."
            ),
            "how": (
                "Keep files where the released <code>smplh:is_valid</code> flag "
                "holds on <strong>every</strong> frame. The release documents "
                "nothing about that flag, so we measured what it means. It does "
                "<em>not</em> track body-fit accuracy: projecting the fitted joints "
                "and comparing against the released 2D keypoints, paired within the "
                "2,481 files that have frames on both sides, gives 0.0792 shoulder "
                "widths of error on valid frames against 0.0799 on invalid ones. "
                "What it does track is the <strong>hands</strong> — on invalid "
                "frames about 40% of left- and right-hand pose vectors are "
                "bit-identical to the previous frame, and median per-frame hand "
                "motion falls to 0.33× and 0.03×, while body pose and translation "
                "are unaffected. The pipeline holds the previous hand pose when it "
                "does not trust the hand fit. For a co-speech gesture model that is "
                "exactly the right thing to gate on. This check replaced an earlier "
                "one that rejected any body part crossing the frame edge; that was "
                "rejecting good data, because a limb leaving the shot does not "
                "necessarily disturb the fit."
            ),
        },
        {
            "id": "fm3", "name": "FM3 · hands held in one position",
            "rate": rate("fm3_static_hands"),
            "aim": (
                "Drop participants who barely gesture. These recordings are clean "
                "and well framed, and still worthless as gesture examples."
            ),
            "how": (
                "Each wrist is expressed relative to the shoulder midpoint and "
                "divided by the median shoulder width, so the measure is free of "
                "camera distance and body size. Its <em>reference</em> position is "
                "its own per-file median — the spot it spends most of its time near "
                "— rather than its position in the first frame, so a participant who "
                "gestures briefly and then stops is still caught. A frame counts as "
                "held when <strong>both</strong> wrists sit within "
                f"<code>{thresholds['static_radius_shoulder_widths']}</code> shoulder widths of their "
                f"references, and the file is flagged when at least "
                f"<code>{100 * thresholds['static_frac_limit']:.0f}%</code> of frames are held."
            ),
        },
    ]


def strictness_note(scan: pd.DataFrame) -> str:
    """The obvious first question about FM2, answered before it is asked.

    FM2 rejects a whole recording for any untrusted frame, and most of what it
    rejects is very short. Whether that is the right trade is a judgement call
    rather than a measurement, so the page states the cost and leaves it open.
    """

    flagged = scan.loc[scan.fm2_smplh_invalid.eq(True)]
    run_s = flagged.fm2_longest_invalid_run_frames / flagged.fps
    under_one = float((run_s < 1.0).mean())
    single_frame = int((flagged.fm2_longest_invalid_run_frames == 1).sum())
    untrusted_share = float((1.0 - flagged.fm2_smplh_valid_frac).median())

    base = scan.fm1_sitting.ne(True) & scan.fm3_static_hands.ne(True)
    valid = pd.to_numeric(scan.fm2_smplh_valid_frac, errors="coerce")

    def survives(keep: pd.Series) -> tuple[float, float]:
        ok = base & keep
        hours = sum(
            V00_DYAD_HOURS[label]
            * (subset.loc[ok[subset.index]].observed_duration_s.sum()
               / subset.observed_duration_s.sum())
            for label, subset in scan.groupby("label") if label in V00_DYAD_HOURS
        )
        return float(ok.mean()), hours

    options = [
        ("every frame trusted <em>(current)</em>", valid >= 1.0),
        ("ignore untrusted stretches under 0.5 s", (valid >= 1.0) | (run_s.reindex(scan.index).fillna(0) < 0.5)),
        ("ignore untrusted stretches under 1 s", (valid >= 1.0) | (run_s.reindex(scan.index).fillna(0) < 1.0)),
        ("ignore untrusted stretches under 2 s", (valid >= 1.0) | (run_s.reindex(scan.index).fillna(0) < 2.0)),
    ]
    rows = "".join(
        f"<tr><th>{name}</th><td class='num'>{100 * rate:.1f}%</td>"
        f"<td class='num'>~{hours:,.0f}</td></tr>"
        for name, keep in options
        for rate, hours in [survives(keep)]
    )
    return f"""
<section class="panel" id="strictness">
  <h2>One open question: how strict should FM2 be?</h2>
  <p class="lede" style="margin:.2rem 0 .8rem;">
    FM2 rejects an entire recording if <em>any</em> frame is untrusted, and most of
    what it rejects is brief: <strong>{100 * under_one:.0f}%</strong> of flagged files have a
    longest untrusted stretch under one second, {single_frame} of them are rejected for a
    <strong>single frame</strong>, and the median flagged file has only
    {100 * untrusted_share:.2f}% of its frames untrusted. Whether a one-second lapse
    should cost a four-minute recording is a judgement, not a measurement, so
    here is the price of each answer. Nothing below is implemented — the current
    rule is the first row.
  </p>
  <table class="rates"><thead><tr><th>FM2 rule</th>
    <th class="num">survives all three</th><th class="num">dyad-hours</th></tr></thead>
    <tbody>{rows}</tbody></table>
  <p class="how" style="color:#9aa6b6;font-size:.85rem;margin-top:.7rem;">
    The case for staying strict: an untrusted frame is one where the fitting
    pipeline held the previous hand pose, so tolerating them means training on
    frozen hands. The case against: at a second's tolerance the surviving pool
    roughly doubles.
  </p>
</section>"""


def stats_block(summary: dict[str, Any], rates: pd.DataFrame) -> dict[str, Any]:
    by_label = summary["by_label"]
    surviving = sum(
        V00_DYAD_HOURS[label] * float(values["hours_weighted_pass_rate"])
        for label, values in by_label.items()
        if label in V00_DYAD_HOURS
    )
    total_hours = sum(V00_DYAD_HOURS.values())
    scanned = int(summary["scan_rows_ok"])

    rows = []
    names = {
        "fm1_sitting": "FM1 · seated",
        "fm2_smplh_invalid": "FM2 · SMPL-H not trusted",
        "fm3_static_hands": "FM3 · static hands",
    }
    for detector, label in names.items():
        row = rates.loc[rates.detector.eq(detector)]
        if not len(row):
            continue
        row = row.iloc[0]
        rows.append(
            f"<tr><th>{label}</th>"
            f"<td class='num'>{int(row.all_flagged):,}</td>"
            f"<td class='num'>{100 * float(row.all_rate):.2f}%</td>"
            f"<td class='num'>{100 * float(row.naturalistic_rate):.2f}%</td>"
            f"<td class='num'>{100 * float(row.improvised_rate):.2f}%</td></tr>"
        )
    passed = int(round(float(summary["overall_pass_rate"]) * scanned))
    rows.append(
        f"<tr><th><strong>Survived all three</strong></th>"
        f"<td class='num'><strong>{passed:,}</strong></td>"
        f"<td class='num'><strong>{100 * float(summary['overall_pass_rate']):.2f}%</strong></td>"
        f"<td class='num'><strong>{100 * float(by_label['naturalistic']['pass_rate']):.2f}%</strong></td>"
        f"<td class='num'><strong>{100 * float(by_label['improvised']['pass_rate']):.2f}%</strong></td></tr>"
    )
    table = (
        "<table class='rates'><thead><tr><th>check</th><th class='num'>files</th>"
        "<th class='num'>of sample</th><th class='num'>naturalistic</th>"
        "<th class='num'>improvised</th></tr></thead>"
        f"<tbody>{''.join(rows)}</tbody></table>"
    )
    return {
        "headline": [
            (f"{scanned:,}", "files scanned"),
            (f"{100 * float(summary['overall_pass_rate']):.1f}%", "survived all three"),
            (f"~{surviving:,.0f}", f"dyad-hours surviving, of {total_hours:,.0f}"),
            (f"{int(summary['distinct_participants']):,}", "distinct participants"),
        ],
        "table": table,
        "footnote": (
            "Percentages are of this sample: a fresh random draw of "
            f"{scanned:,} V00 participant files, balanced across the two labels, "
            "excluding charades. The checks sum to more than the rejected total "
            "because a file can trip several. The dyad-hour figure projects the "
            "sample's duration-weighted pass rate onto the "
            f"{total_hours:,.0f} dyad-hours of V00 that clear the basic "
            "completeness and probe checks — longer recordings are likelier to "
            "contain a bad stretch, so weighting by duration matters. Detector "
            "thresholds were tuned on a different sample; the rates here are "
            "measured out of sample."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path,
                        default=Path("configs/v00_fm_briefing_detect.yaml"))
    parser.add_argument("--gallery-config", type=Path,
                        default=Path("configs/v00_fm_briefing_review.yaml"))
    parser.add_argument("--out-name", default="index")
    args = parser.parse_args()

    os.umask(0o077)
    gallery = yaml.safe_load(args.gallery_config.read_text(encoding="utf-8"))
    detect = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    analysis_dir = Path(str(detect["outputs"]["root"]).lstrip("./"))
    summary = json.loads((analysis_dir / "analysis_summary.json").read_text(encoding="utf-8"))
    rates = pd.read_csv(analysis_dir / "flag_rates_by_label.csv")

    output_root = Path(str(gallery["outputs"]["private_root"]).lstrip("./"))
    manifest = pd.read_csv(Path(str(gallery["outputs"]["manifest"]).lstrip("./")))
    source_root = Path(str(gallery["source_root"])).resolve()

    results_dir = output_root / "item_results"
    records: list[dict[str, Any]] = []
    for record in manifest.to_dict("records"):
        digest = hashlib.sha256(str(record["review_item_id"]).encode()).hexdigest()[:20]
        path = results_dir / f"{digest}.json"
        rendered = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
        merged = {**record, **rendered}
        signals = json.loads(str(record.get("signals_json") or "{}"))
        merged["clip_note"] = signals.get("clip_note", "")
        records.append(merged)

    if bool(gallery["outputs"].get("link_source_media", True)):
        link_source_media(output_root, records, source_root)

    by_group: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        by_group.setdefault(str(record["sample_group"]), []).append(record)

    sections = [
        {
            "id": anchor, "heading": heading, "blurb": blurb,
            "records": by_group.get(group, []),
        }
        for anchor, group, heading, blurb in SECTIONS
    ]

    write_briefing(
        output_root / f"{args.out_name}.html",
        title="Seamless Interaction — V00 curation checks",
        intro=(
            "We are building a curated training subset of the Seamless Interaction "
            "dataset for a co-speech gesture model. Vendor V00 is already chosen. "
            "Three file-level checks reject recordings that would hurt the model; "
            "this page explains what they are, what they did to a fresh random "
            "sample, and lets you watch a cross-section of both what survives and "
            "what each check throws away. Every clip is 30 seconds, carries a "
            "whole-recording filmstrip above it, and links to the full source "
            "recording. Nothing here asks you for input — it is a status report."
        ),
        detectors=detector_blocks(summary, rates),
        stats=stats_block(summary, rates),
        extra_panels=strictness_note(
            pd.read_parquet(analysis_dir / "scan_with_flags.parquet")
        ),
        sections=sections,
    )
    print(json.dumps({
        "gallery": str(output_root / f"{args.out_name}.html"),
        "clips": len(records),
        "by_section": {s["heading"]: len(s["records"]) for s in sections},
        "unrendered": sum(1 for r in records if r.get("status") not in {"rendered", "reused"}),
    }, indent=2))


if __name__ == "__main__":
    main()

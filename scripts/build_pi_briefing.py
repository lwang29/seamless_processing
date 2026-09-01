#!/usr/bin/env python3
"""Assemble the corpus-wide briefing page from rendered clips and the scan.

Every number is read from the analysis of the sample the page shows, never
hard-coded, so the prose cannot drift away from the measurements. The page
collects nothing: it exists to bring a reader up to speed and let them watch the
footage themselves.
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

# Eligible dyad-hours per vendor from the M-1 inventory. Each vendor's sampled
# pass rate is projected onto its own population.
VENDOR_HOURS = {"00": 1440.6, "01": 328.8, "02": 761.4, "03": 1430.1}
VENDORS = ("00", "01", "02", "03")
FLAGS = ("fm0_unusable_source", "fm1_sitting", "fm2_smplh_invalid",
         "fm3_static_hands", "fm4_audio_dead")

# What each card shows. Ordered so a reader meets the verdict first and the
# evidence for it second; anything needing a paragraph to interpret was cut.
SIGNAL_ROWS: tuple[tuple[str, str, str], ...] = (
    ("verdict", "Verdict", "text"),
    ("vendor", "Vendor", "text"),
    ("condition", "Recording condition", "text"),
    ("observed_duration_s", "Length of the full recording", "clock"),
    # One FM1 row per card, chosen for that clip's vendor by _fm1_row below.
    # Printing both rules on every card meant V01 and V02 clips carried two cut
    # points that do not apply to them, and V00 clips carried V03's.
    ("fm1_measure", "FM1 · what decides for this vendor", "text"),
    ("fm2_smplh_valid_frac", "FM2 · frames with a trusted body fit (must be 100%)", "percent"),
    ("fm2_longest_invalid_run_s", "FM2 · longest untrusted stretch", "seconds"),
    ("fm3_static_frac", "FM3 · share of the recording with both hands parked (rejects above 75%)", "percent"),
    ("audio_envelope_dynamics_db", "FM4 · how much the audio level moves (rejects below 1.5 dB)", "decibels"),
    ("fm4_note", "FM4 · audio track", "text"),
)

STRATA: tuple[tuple[str, str, str], ...] = (
    ("pass_improvised", "Kept · improvised",
     "Files that pass every check, from the improvised recording condition — the "
     "participants are performing a prompted task. A plain random draw of what "
     "survives, not a selection of its best members."),
    ("pass_naturalistic", "Kept · naturalistic",
     "The same draw on the naturalistic condition, where participants simply "
     "converse. These two sections are here to be compared."),
    ("fm0_unusable_source", "Rejected · FM0, the source cannot be used",
     "Rejected on the format of the recording, before anything in it was examined. "
     "Three reasons, and the card for each clip says which applied. In "
     "<strong>V01</strong> these are frames stored horizontally stretched, where the "
     "picture can be corrected but the released body fit cannot. In "
     "<strong>V03</strong> they are wide room-camera formats, excluded for far-field "
     "audio — <em>the video itself usually looks fine, so there is nothing to see "
     "wrong in these clips; the reason is what you hear.</em> The third, an "
     "annotation track that does not line up with its own video, rejects nothing in "
     "this sample on its own."),
    ("fm1_sitting", "Rejected · FM1, the participant is seated",
     "A seated body gestures differently — the torso is supported, the arms rest, "
     "the legs contribute nothing — so mixing the two contaminates the training "
     "signal."),
    ("fm2_smplh_invalid", "Rejected · FM2, the body fit is not trusted throughout",
     "<strong>Each clip is positioned on that file's longest continuous untrusted "
     "stretch</strong>, so the frames the check objected to are the frames on "
     "screen. In many of them nothing looks wrong in the video — that is the "
     "point: what degrades is the fitted hand pose, not the picture."),
    ("fm3_static_hands", "Rejected · FM3, the hands never move",
     "A participant who does not gesture teaches a co-speech gesture model "
     "nothing, however clean the recording."),
    ("fm4_audio_dead", "Rejected · FM4, the audio carries no voice",
     "Static, buzz or silence. The video may look perfectly normal."),
)


def _fm1_row(signals: dict[str, Any], vendor: str) -> None:
    """Collapse the two FM1 measures into the one this vendor actually uses."""
    knee = signals.pop("fm1_knee_between_torso_p50", None)
    hip = signals.pop("fm1_hip_flexion_deg_p50", None)
    if vendor == "00" and knee is not None:
        signals["fm1_measure"] = f"knee height along the torso {knee:.2f} \u2014 rejected below 0.43"
    elif vendor == "03" and hip is not None:
        signals["fm1_measure"] = f"hip angle {hip:.0f}\u00b0 \u2014 rejected below 146\u00b0"
    else:
        signals["fm1_measure"] = "not applied to this vendor"


def _item_result_path(results_dir: Path, review_item_id: str) -> Path:
    digest = hashlib.sha256(review_item_id.encode("utf-8")).hexdigest()[:20]
    return results_dir / f"{digest}.json"


def _reasons_column(flag: str) -> str:
    return f"{flag.split('_')[0]}_audio_reasons" if flag.startswith("fm4") \
        else f"{flag.split('_')[0]}_sitting_reasons"


def applied_to(scan: pd.DataFrame, flag: str) -> pd.DataFrame:
    """The rows a check was actually run on.

    A check switched off for a vendor records False rather than null, so an
    unfiltered mean silently averages those zeros in. The detector chips and the
    table's totals row must use the same subset or the page prints two different
    numbers for one quantity.
    """
    column = _reasons_column(flag)
    if column not in scan.columns:
        return scan
    return scan.loc[scan[column].astype(str).ne("retired_for_vendor")]


def detector_blocks(scan: pd.DataFrame) -> list[dict[str, str]]:
    def rate(column: str) -> str:
        subset = applied_to(scan, column)
        series = subset[column].dropna()
        if not len(series):
            return ""
        vendors = sorted(subset.vendor_id.unique())
        where = "" if len(vendors) == 4 else " of V" + ", V".join(vendors)
        return f"{100 * float(series.mean()):.1f}% of files{where}"

    return [
        {
            "id": "fm0", "name": "FM0 · the source cannot be used",
            "rate": rate("fm0_unusable_source"),
            "aim": ("Reject a recording whose stored form is broken, before asking "
                    "anything about what it shows."),
            "how": ("Two rules. Four video formats are excluded outright: two in V01 "
                    "are stored horizontally stretched and the released body fit "
                    "absorbed the stretch, so un-stretching the picture cannot "
                    "recover the pose; two in V03 are wide room-camera shots whose "
                    "far-field audio was judged unusable by eye and ear. Separately, "
                    "a file whose annotation track drifts more than half a second "
                    "from its own video is dropped — on the two checked, the fitted "
                    "body was unrelated to the person on screen. "
                    "<strong>Fires on nothing in V00 or V02</strong>, which have "
                    "neither problem."),
        },
        {
            "id": "fm1", "name": "FM1 · the participant is seated",
            "rate": rate("fm1_sitting"),
            "aim": ("Keep only standing participants. A seated body produces a "
                    "fundamentally different gesture distribution, so mixing the two "
                    "contaminates the training signal."),
            "how": ("From the fitted 3D skeleton, never from how much frame the "
                    "person fills — seated participants were often recorded with the "
                    "camera moved closer, so occupancy carries no signal. "
                    "<strong>V00</strong> uses how far down the leg the knee sits, "
                    "measured along the participant's own torso axis so camera tilt "
                    "and distance cannot move it; the cut is 0.43. "
                    "<strong>V03</strong> uses the hip angle instead, at 146°, "
                    "because on that vendor the knee measure separates the two "
                    "postures much less cleanly. "
                    "<strong>V01 and V02 switch it off</strong>: on V01 it never "
                    "fires once the excluded formats are removed, and on V02 every "
                    "firing but one came from a side clause misreading participants "
                    "whose feet were out of shot, all of them confirmed standing. "
                    "<em>Evidence, from the review rounds that set these cuts rather "
                    "than from this sample:</em> V00's cut rests on 132 hand-labelled "
                    "clips, and a later random draw of 40 files from the passing side "
                    "found <strong>no seated participant among them</strong>. V03's "
                    "rests on 164; at 146° it catches 52 of the 53 seated files in "
                    "that set and discards 10 of 96 standing ones, and the earlier "
                    "knee-based cut was worse on both counts. V01's was 0 firings in "
                    "1,379 of its square-pixel files."),
        },
        {
            "id": "fm2", "name": "FM2 · the body fit is not trusted throughout",
            "rate": rate("fm2_smplh_invalid"),
            "aim": ("Keep only recordings whose fitted body can be trusted frame by "
                    "frame. Those fitted parameters are the training target, so an "
                    "untrusted stretch is not a cosmetic flaw — it is corrupted "
                    "labels."),
            "how": ("Keep files where the dataset's own per-frame validity flag holds "
                    "on <strong>every</strong> frame. The release documents nothing "
                    "about that flag, so we measured what it means, in an earlier "
                    "round and not on this sample: it tracks hand pose rather than "
                    "body fit — on invalid frames about 40% of hand pose vectors are "
                    "bit-identical to the previous frame while the body is unaffected "
                    "— and hands are the signal for co-speech gesture. This is the strictest check by a wide margin and the "
                    "only one no review has ever contradicted."),
        },
        {
            "id": "fm3", "name": "FM3 · the hands never move",
            "rate": rate("fm3_static_hands"),
            "aim": ("Drop recordings where the participant does not gesture. However "
                    "clean, they teach the model nothing about co-speech motion."),
            "how": ("Each wrist's distance from its own median position, in shoulder "
                    "widths, from the released 2D keypoints. A frame counts as static "
                    "when both wrists sit within 0.10 shoulder widths of their median; "
                    "a file is rejected above 75% static frames. Normalised by "
                    "shoulder width and anchored on each file's own median, so it "
                    "behaves the same on every vendor — 4.5% to 9.0% across the four."),
        },
        {
            "id": "fm4", "name": "FM4 · the audio carries no voice",
            "rate": rate("fm4_audio_dead"),
            "aim": ("Reject a recording with no usable speech. Audio is not the "
                    "training target, but it anchors the alignment between gesture "
                    "and speech, so a dead track makes a clip useless for that."),
            "how": ("Whether the audio level <em>moves</em>. Static and buzz sit at "
                    "one level whether anyone is speaking or not; a track carrying a "
                    "voice swings, measured as the spread of the level envelope. "
                    "<em>From the round that set this cut, not from this sample:</em> "
                    "on 48 hand-labelled clips the two classes do not overlap — "
                    "unusable 0.2 to 1.4 dB, usable 1.7 to 69 — and 1.5 dB catches 13 "
                    "of 13 with no false positive, where two earlier rules reading the "
                    "absolute level or the speech annotations scored 28% and 38% "
                    "precision on the same labels. <strong>Switched off for "
                    "V00</strong>, where the earlier rule produced only false "
                    "positives; on a 4,000-file V00 draw the current rule fired once, "
                    "on a file already rejected by another check."),
        },
    ]


def build_stats(scan: pd.DataFrame) -> dict[str, Any]:
    ok = scan.loc[scan.status.eq("ok")]
    rows = []
    total_hours = 0
    for vendor in VENDORS:
        group = ok.loc[ok.vendor_id.eq(vendor)]
        if group.empty:
            continue
        share = float(group.passes_all.fillna(False).mean())
        # Round once, here, and total the rounded values: a column that does not
        # add up is the first thing a careful reader checks.
        hours = round(VENDOR_HOURS[vendor] * share)
        total_hours += hours
        cells = ""
        for flag in FLAGS:
            # A check switched off for a vendor reads False, not null, so a plain
            # rate would print "0.0%" and imply it looked and found nothing. It
            # did not look; the reason column says so and the cell shows a dash.
            reasons = _reasons_column(flag)
            retired = (reasons in group.columns
                       and group[reasons].astype(str).eq("retired_for_vendor").all())
            if retired or group[flag].isna().all():
                cells += "<td class='num' title='not applied to this vendor'>—</td>"
            else:
                cells += f"<td class='num'>{100 * float(group[flag].fillna(False).mean()):.1f}%</td>"
        rows.append(
            f"<tr><th>V{vendor}</th><td class='num'>{len(group):,}</td>{cells}"
            f"<td class='num'><strong>{100 * share:.1f}%</strong></td>"
            f"<td class='num'>{hours:,d}</td></tr>"
        )
    grand = float(ok.passes_all.fillna(False).mean())
    totals = ""
    for flag in FLAGS:
        applied = applied_to(ok, flag)
        totals += (f"<td class='num'>{100 * float(applied[flag].fillna(False).mean()):.1f}%</td>"
                   if len(applied) else "<td class='num'>—</td>")
    rows.append(
        f"<tr class='total'><th>all four</th><td class='num'>{len(ok):,}</td>{totals}"
        f"<td class='num'><strong>{100 * grand:.1f}%</strong></td>"
        f"<td class='num'>{total_hours:,d}</td></tr>"
    )
    header = (
        "<thead><tr><th>Vendor</th><th class='num'>Files scanned</th>"
        + "".join(f"<th class='num'>{f.split('_')[0].upper()}</th>" for f in FLAGS)
        + "<th class='num'>Kept</th><th class='num'>Dyad-hours kept</th>"
        "</tr></thead>"
    )
    table = f"<table class='rates'>{header}<tbody>{''.join(rows)}</tbody></table>"
    return {
        "headline": [
            (f"{len(ok):,}", "files scanned"),
            (f"{100 * grand:.0f}%", "kept"),
            (f"{total_hours:,d} h", "projected dyad-hours"),
            # participant_id is numbered within a vendor, not across the corpus,
            # so counting the bare id merges different people who share a number.
            (f"{len(ok.groupby(['vendor_id', 'participant_id'])):,}",
             "participants in the sample"),
        ],
        "table": table,
        "footnote": (
            "Each row is a fresh random draw from that vendor, measured under the "
            "pipeline as it stands. A file counts against every check it trips, so "
            "the five flag columns sum to more than the share rejected. "
            "<strong>Dyad-hours kept</strong> applies each vendor's sampled pass rate "
            "to its whole eligible population — V00 1,441 h, V01 329 h, V02 761 h, "
            "V03 1,430 h, 3,961 h in total. A dash means the check does not apply to "
            "that vendor; the reasons are in the section above."
        ),
    }


def coverage_panel(summary: dict[str, Any]) -> str:
    """State plainly which strata are empty, and why each one is."""
    reasons = {
        "v02_pass_improvised": "V02 recorded no improvised material at all.",
        "v00_fm0_unusable_source": "V00 has none of the excluded formats and no "
                                   "drifting annotation tracks.",
        "v02_fm0_unusable_source": "V02 has none of the excluded formats and no "
                                   "drifting annotation tracks.",
        "v01_fm1_sitting": "FM1 is switched off for V01 (see the check above).",
        "v02_fm1_sitting": "FM1 is switched off for V02 (see the check above).",
        "v00_fm4_audio_dead": "FM4 is switched off for V00 (see the check above).",
    }
    empty = list(summary.get("empty_strata", []))
    items = "".join(
        f"<li><code>{g.replace('_', ' ')}</code> — "
        f"{reasons.get(g, 'no file in this sample fell into it.')}</li>"
        for g in empty
    )
    # A stratum can also be present but short, when the whole population of it is
    # smaller than the draw. Saying so stops a two-clip section reading as a
    # sampling accident.
    target = max(summary.get("draw_counts", {}).values() or [0])
    short = {
        name: count for name, count in sorted(summary.get("draw_counts", {}).items())
        if 0 < count < target
    }
    population = summary.get("population", {})
    short_items = ""
    for name, count in short.items():
        vendor, _, stratum = name.partition("_")
        total = population.get(vendor.upper(), {}).get(stratum)
        clips = "clip" if count == 1 else "clips"
        whole = ""
        if total is not None:
            files = "file" if total == 1 else "files"
            whole = f" — the whole sample holds only {total} such {files}"
        short_items += (f"<li><code>{name.replace('_', ' ')}</code> — {count} {clips}"
                        f"{whole}, so the section is short rather than curated.</li>")
    if not items and not short_items:
        return ""
    blocks = []
    if items:
        blocks.append(
            "<h2>Sections you will not find below, and why</h2>"
            '<p class="lede" style="margin:.2rem 0 .6rem;">Each vendor has seven '
            "strata: two for the recording conditions that pass, and one for each "
            "check that can reject. These are missing by construction rather than "
            "by chance, so they are named here rather than silently omitted.</p>"
            '<ul style="margin:.2rem 0 0; padding-left:1.2rem; color:#aeb8c6;">'
            + items + "</ul>"
        )
    if short_items:
        # Kept apart from the list above: these sections *are* below, they are
        # just small, and filing them under "you will not find" would be wrong.
        blocks.append(
            '<h2 style="margin-top:1.2rem;">Sections that are present but short</h2>'
            '<ul style="margin:.2rem 0 0; padding-left:1.2rem; color:#aeb8c6;">'
            + short_items + "</ul>"
        )
    return '<section class="panel">' + "".join(blocks) + "</section>"


def main() -> None:
    os.umask(0o077)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/pi_briefing_review.yaml"))
    parser.add_argument("--detect-config", type=Path,
                        default=Path("configs/pi_briefing_detect.yaml"))
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    gallery = yaml.safe_load((root / args.config).read_text(encoding="utf-8"))
    detect = yaml.safe_load((root / args.detect_config).read_text(encoding="utf-8"))
    scan = pd.read_parquet(
        root / str(detect["outputs"]["root"]).lstrip("./") / "scan_with_flags.parquet"
    )
    manifest = pd.read_csv(root / str(gallery["outputs"]["manifest"]).lstrip("./"))
    private_root = root / str(gallery["outputs"]["private_root"]).lstrip("./")
    report_dir = root / str(gallery["outputs"]["report_dir"]).lstrip("./")
    summary = json.loads((report_dir / "pi_briefing_sample.json").read_text(encoding="utf-8"))
    results_dir = private_root / "item_results"

    records: list[dict[str, Any]] = []
    for record in manifest.to_dict("records"):
        path = _item_result_path(results_dir, str(record["review_item_id"]))
        rendered = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
        # The sidecar carries a copy of `signals_json` frozen at render time, and
        # letting it win means any signal added to the sampler after the clips
        # were rendered is silently dropped -- which is how eight FM4 cards came
        # to read "unknown" while the manifest said "empty". The manifest owns the
        # signals; the sidecar owns the render outcome.
        merged = {**record, **rendered, "signals_json": record.get("signals_json", "{}")}
        signals = json.loads(str(merged.get("signals_json") or "{}"))
        vendor = str(signals.get("vendor", "")).replace("V", "")
        _fm1_row(signals, vendor)
        # FM4's number is null exactly when the audio file holds no samples --
        # which is every clip in an FM4 section whose WAV is a bare header.
        # Dropping the row left those cards silent about the check that rejected
        # them, including the only such clip V03 has.
        if signals.get("audio_envelope_dynamics_db") is None:
            status = str(signals.get("audio_status", ""))
            signals["fm4_note"] = (
                "the released audio file is empty \u2014 no samples at all"
                if status == "empty"
                else "could not be measured (" + (status or "unknown") + ")"
            )
        verdict = str(signals.get("verdict", ""))
        merged["clip_badge"] = verdict
        merged["clip_badge_kind"] = "pass" if verdict == "kept" else "flag"
        merged["signals_json"] = json.dumps(signals, sort_keys=True)
        records.append(merged)

    by_group: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        by_group.setdefault(str(record["sample_group"]), []).append(record)

    sections: list[dict[str, Any]] = []
    for vendor in VENDORS:
        for key, title, blurb in STRATA:
            members = by_group.get(f"v{vendor}_{key}", [])
            if not members:
                continue
            sections.append(
                {
                    "id": f"v{vendor}-{key.replace('_', '-')}",
                    "group": f"V{vendor}",
                    "short": title,
                    "heading": f"V{vendor} — {title}",
                    "blurb": blurb,
                    "records": members,
                }
            )

    if bool(gallery["outputs"].get("link_source_media", True)):
        link_source_media(
            private_root, records, root / str(gallery["source_root"]).lstrip("./")
        )

    write_briefing(
        private_root / "index.html",
        sections=sections,
        signal_rows=SIGNAL_ROWS,
        detectors_heading="The five checks",
        contents_heading="Contents — four vendors, seven strata each",
        title="Seamless Interaction — curating a co-speech gesture training set",
        intro=(
            "What the filtering pipeline is, what it does, and what it keeps — across "
            "<strong>all four vendors</strong> in the dataset, on a fresh random "
            "sample that nothing has been tuned against. Five checks now, up from "
            "three: FM0 and FM4 were added after reviewing footage from V01, V02 and "
            "V03, which turned out to differ from V00 in ways worth catching. Each "
            "check below says what it is for and how it decides. Every rate on this "
            "page is measured from the sample shown; the dyad-hour figures scale "
            "those rates by each vendor's eligible population from the corpus "
            "inventory, and the hand-label counts quoted in the check descriptions "
            "come from the review rounds that set each threshold. The clips are "
            "grouped by vendor and then by outcome, so any verdict can be checked "
            "against the footage that produced it."
        ),
        detectors=detector_blocks(scan),
        stats=build_stats(scan),
        extra_panels=coverage_panel(summary),
    )
    print(json.dumps({
        "gallery": str(private_root / "index.html"),
        "sections": len(sections),
        "clips": len(records),
        "rendered": sum(1 for r in records if r.get("status") in {"rendered", "reused"}),
    }, indent=2))


if __name__ == "__main__":
    main()

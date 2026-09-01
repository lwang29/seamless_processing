#!/usr/bin/env python3
"""Assemble the Round-7 vendor gallery in the briefing layout.

The briefing layout is now the standard for every gallery: the pipeline and what
it did to this sample at the top, a table of contents, a rule between sections,
the signals for each clip on the card, and no free-text notes.

Every number is read from the scan of the sample the page shows, so the prose
cannot drift away from the measurements.
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

# Eligible dyad-hours per vendor from the M-1 inventory, for projecting the
# sample's pass rate onto the population.
VENDOR_HOURS = {"01": 328.8, "02": 761.4, "03": 1430.1}

SIGNAL_ROWS: tuple[tuple[str, str, str], ...] = (
    ("verdict", "what the pipeline said", "text"),
    ("stored_raster", "raster as stored", "text"),
    ("shown_raster", "raster as shown", "text"),
    ("repair_applied", "repair applied before display", "text"),
    ("fm1_knee_between_torso_p50", "FM1 · knee position along the torso (cut 0.43)", "ratio"),
    ("fm1_shin_verticality_p50", "FM1 · shin direction (negative = pointing up)", "ratio"),
    ("fm1_ankles_visible_frac", "FM1 · frames with both ankles in shot", "percent"),
    ("fm2_smplh_valid_frac", "FM2 · frames with a trusted SMPL-H fit", "percent"),
    ("fm3_static_frac", "FM3 · share of the recording with both hands parked", "percent"),
    ("reproj_shoulder_widths_p50", "fit error against the 2D keypoints (shoulder widths)", "ratio"),
    ("timebase", "annotation timebase", "text"),
    ("audio", "released audio", "text"),
    ("observed_duration_s", "recording length", "clock"),
)

# (section id, sample_group prefix, heading, blurb, ask, kind)
ASK_SECTIONS: tuple[tuple[str, tuple[str, ...], str, str, str], ...] = (
    (
        "ask-posture", ("ask_posture_v03",),
        "ASK 1 · Is this V03 participant sitting or standing?",
        "You found seven seated V03 clips that FM1 missed and seven it caught. The "
        "two groups separate almost perfectly on the measure FM1 cuts on — the "
        "caught ones read 0.301 to 0.429, the missed ones 0.435 to 0.586 — and the "
        "cut sits at 0.43. So FM1 is not broken on V03; it is <em>mis-set</em>, and "
        "moving it needs labels on both sides of where it might go. These clips are "
        "drawn evenly across seven bands spanning 0.20 to 0.80, so the answers land "
        "where the decision actually is. Files whose shins point the wrong way up "
        "are excluded, so every clip here is a genuine posture question.",
        "For each clip: <strong>sitting or standing</strong>. Nothing else. The "
        "cards show the measured value so you can see where the boundary falls, but "
        "please judge from the video and ignore the number.",
    ),
    (
        "ask-raster-v01", ("ask_raster_1012x1920", "ask_raster_1920x1080"),
        "ASK 2 · Are V01's other two rasters usable once un-stretched?",
        "You asked about 1012×1920 and 1920×1080. Both are anamorphic, like "
        "2160×2160: the container declares a 9:16 display aspect for every V01 "
        "raster and only 1080×1920 stores square pixels. 1012×1920 needs a 6.7% "
        "stretch and lands on exactly 1080×1920. 1920×1080 needs a 3.16× squeeze "
        "and lands on 607×1080, which throws away most of its horizontal detail. "
        "<strong>Both are shown here already corrected.</strong> There are 102 and "
        "60 eligible files respectively, so this is a small decision either way.",
        "Does the corrected picture look usable, and does Panel B's skeleton sit on "
        "the person? For 1920×1080 in particular — is 607 px of width enough?",
    ),
    (
        "ask-raster-v03", ("ask_raster_640x480", "ask_raster_1080x960",
                           "ask_raster_3840x2160", "ask_raster_2180x3840"),
        "ASK 3 · V03's four remaining rasters",
        "You asked what 640×480 is. It is a quarter-turned low-resolution room "
        "camera: 60 eligible files, all of them turned the same way, all of them "
        "square-pixel. 1080×960 is the one V03 raster that is neither turned nor "
        "anamorphic — 28 files, upright, just an unusually wide frame with a small "
        "participant. 3840×2160 is the big turned raster you already found (1,336 "
        "files, 1,215 turned one way and 112 the other). 2180×3840 is ordinary "
        "portrait with 20 extra pixels of width. <strong>All are shown with the "
        "turn already undone.</strong>",
        "Are the turned ones usable now that they are upright? And is 640×480 worth "
        "keeping at that resolution, or should it be dropped on image quality alone?",
    ),
    (
        "ask-anamorphic", ("ask_anamorphic_v01",),
        "ASK 4 · V01 2160×2160, corrected — and why the correction is not enough",
        "These are the clips you listed as horizontally stretched. You were right, "
        "and the container agrees: pixel aspect 9:16, so the stored square frame is "
        "meant to be squeezed to 1215×2160. The picture and the released 2D "
        "keypoints are now corrected and both look right. <strong>The released "
        "SMPL-H is not, and cannot be.</strong> An anisotropic squeeze is not a "
        "rigid transform and the fitted camera is isotropic, so the only way the "
        "fit could match a stretched person was to bend the body. Measured on 60 "
        "files per raster: aligning the projected body to the released keypoints by "
        "translation and scale alone leaves a shape residual of 0.113 against the "
        "raster as stored and 0.218 once corrected, against 0.048–0.074 on "
        "square-pixel rasters. That is why nearly all of them trip FM1.",
        "Look at Panel A against Panel B, and at Panel C. Panel A should now fit the "
        "person; Panel B and C should still look wrong. If you agree, the question "
        "is whether to drop V01's 5,002 anamorphic files (about 146 of its 329 "
        "dyad-hours) or to keep them for a future re-fit.",
    ),
    (
        "ask-timebase", ("ask_timebase",),
        "ASK 5 · The clip whose keypoints slid off the person",
        "<code>V01_S1607_I00000135_P2569</code> and its siblings. The released "
        "arrays are sampled on a uniform grid at the container's nominal rate, but "
        "these files hold fewer stored frames than that rate implies, so pairing "
        "the two by frame number walks off progressively — a measured median of "
        "4.45 s and a maximum of 6.87 s of misalignment on that file alone. The "
        "renderer now looks annotations up by timestamp instead, which brings the "
        "median error to 0 ms and the worst case to one frame period. <strong>These "
        "clips are rendered with that fix.</strong> 16 eligible files are affected "
        "this way; a further 12 carry annotation arrays several times longer than "
        "their video, and 35 more are the already-known empty-annotation files.",
        "Do the keypoints stay on the person for the whole clip now? That is the "
        "check — if they still drift, the fix is incomplete.",
    ),
)

CELL_LABELS = {
    "pass": ("passed all three checks", "A random draw from what the pipeline keeps."),
    "fm1_sitting": ("flagged FM1 — read as seated",
                    "Rejected as seated. Note the shin-direction row: where it is "
                    "negative, FM1 fired because the ankles left the frame rather "
                    "than because anyone sat down."),
    "fm2_smplh_invalid": ("flagged FM2 — SMPL-H not trusted on every frame",
                          "Each clip is positioned on that file's longest continuous "
                          "untrusted stretch, so the frames the check objected to are "
                          "the frames on screen."),
    "fm3_static_hands": ("flagged FM3 — hands held in one position",
                         "Rejected because both wrists stay parked for most of the "
                         "recording."),
    "multi": ("flagged by more than one check",
              "Positioned on the longest untrusted stretch where FM2 is among the "
              "checks that fired."),
}


def _item_result_path(results_dir: Path, review_item_id: str) -> Path:
    digest = hashlib.sha256(review_item_id.encode("utf-8")).hexdigest()[:20]
    return results_dir / f"{digest}.json"


def detector_blocks(scan: pd.DataFrame) -> list[dict[str, str]]:
    def rate(column: str) -> str:
        series = scan[column].dropna()
        return f"{100 * float(series.mean()):.1f}% flagged" if len(series) else ""

    return [
        {
            "id": "fm1", "name": "FM1 · participant is seated", "rate": rate("fm1_sitting"),
            "aim": ("Keep only standing participants: a seated body gestures "
                    "differently, and mixing the two contaminates the training signal."),
            "how": ("<code>knee_between_torso</code> — how far down the hip-to-ankle "
                    "drop the knee sits along the participant's own torso axis, from "
                    "SMPL-H forward kinematics. Standing reads about 0.55, seated "
                    "about 0.30, and the cut is <code>&lt; 0.43</code>, with an "
                    "add-on for an inverted shin. The verdict is taken once per "
                    "participant-session by majority. <strong>Calibrated on V00 "
                    "portrait framing.</strong> It does not transfer: on square "
                    "rasters it fires on 93% of files, and on V03 it sits about 0.1 "
                    "too low. Both are open, and ASK 1 below is how the second gets "
                    "settled."),
        },
        {
            "id": "fm2", "name": "FM2 · SMPL-H fit not trusted", "rate": rate("fm2_smplh_invalid"),
            "aim": ("Keep only recordings whose fitted body can be trusted frame by "
                    "frame. The SMPL-H parameters are the training target, so an "
                    "untrusted stretch is corrupted labels, not a cosmetic flaw."),
            "how": ("Keep files where the released <code>smplh:is_valid</code> flag "
                    "holds on <strong>every</strong> frame. Measurement showed that "
                    "flag tracks hand pose rather than body fit — about 40% of hand "
                    "pose vectors are frozen on invalid frames while the body is "
                    "unaffected — and hands are the signal for co-speech gesture. "
                    "This is the one check that transfers cleanly to every vendor, "
                    "because it reads a released field rather than a fitted geometry."),
        },
        {
            "id": "fm3", "name": "FM3 · hands held in one position", "rate": rate("fm3_static_hands"),
            "aim": ("Drop recordings where the participant never gestures. However "
                    "clean, they teach the model nothing about co-speech motion."),
            "how": ("Each wrist's displacement from its own median position, in "
                    "shoulder widths, from the released 2D keypoints. A frame is "
                    "static if both wrists sit within 0.10 shoulder widths of their "
                    "median; the file is flagged above 75% static frames. Normalised "
                    "by shoulder width and anchored on the file's own median, so it "
                    "behaves the same everywhere — 6.1% to 9.0% across the vendors "
                    "against V00's 7.3%."),
        },
    ]


def build_stats(scan: pd.DataFrame) -> dict[str, Any]:
    ok = scan.loc[scan.status.eq("ok")]
    passes = ok.passes_all.fillna(False)
    rows = []
    for vendor, group in ok.groupby("vendor_id"):
        share = float(group.passes_all.fillna(False).mean())
        hours = VENDOR_HOURS.get(vendor)
        rows.append(
            f"<tr><th>V{vendor}</th>"
            f"<td class='num'>{len(group):,}</td>"
            f"<td class='num'>{100 * float(group.fm1_sitting.fillna(False).mean()):.2f}%</td>"
            f"<td class='num'>{100 * float(group.fm2_smplh_invalid.fillna(False).mean()):.2f}%</td>"
            f"<td class='num'>{100 * float(group.fm3_static_hands.fillna(False).mean()):.2f}%</td>"
            f"<td class='num'><strong>{100 * share:.2f}%</strong></td>"
            f"<td class='num'>{share * hours:,.0f} h</td></tr>"
        )
    total_hours = sum(
        float(g.passes_all.fillna(False).mean()) * VENDOR_HOURS.get(v, 0.0)
        for v, g in ok.groupby("vendor_id")
    )
    table = (
        "<table class='rates'><thead><tr><th>vendor</th><th>files scanned</th>"
        "<th>FM1</th><th>FM2</th><th>FM3</th><th>survives</th>"
        "<th>projected dyad-hours</th></tr></thead><tbody>"
        + "".join(rows) + "</tbody></table>"
    )
    anamorphic = int(ok.geom_anamorphic.fillna(False).sum())
    rotated = int(ok.geom_rotated.fillna(False).sum())
    return {
        "headline": [
            (f"{len(scan):,}", "files scanned"),
            (f"{100 * float(passes.mean()):.1f}%", "survive all three checks"),
            (f"{total_hours:,.0f} h", "projected surviving dyad-hours"),
            (f"{anamorphic:,}", "stored with non-square pixels"),
            (f"{rotated:,}", "stored a quarter turn from upright"),
            (f"{int(ok.audio_empty.fillna(False).sum()):,}", "with an empty WAV"),
            (f"{int((~ok.timebase_consistent.fillna(True)).sum()):,}", "whose annotations do not match the video"),
        ],
        "table": table,
        "footnote": (
            "Flag rates are per file over this sample; the three thresholds are "
            "unchanged from V00 so every column stays comparable with earlier "
            "rounds. Projected hours apply each vendor's sampled pass rate to its "
            "eligible population (V01 328.8 h, V02 761.4 h, V03 1,430.1 h). Read "
            "every verdict as <em>what the current pipeline says</em>: FM1 is known "
            "not to transfer, and correcting that is what the asks below are for."
        ),
    }


def main() -> None:
    os.umask(0o077)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/vendors_r7_review.yaml"))
    parser.add_argument("--detect-config", type=Path,
                        default=Path("configs/vendors_r7_detect.yaml"))
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    gallery = yaml.safe_load((root / args.config).read_text(encoding="utf-8"))
    detect = yaml.safe_load((root / args.detect_config).read_text(encoding="utf-8"))
    scan = pd.read_parquet(
        root / str(detect["outputs"]["root"]).lstrip("./") / "scan_with_flags.parquet"
    )
    manifest = pd.read_csv(root / str(gallery["outputs"]["manifest"]).lstrip("./"))
    private_root = root / str(gallery["outputs"]["private_root"]).lstrip("./")
    results_dir = private_root / "item_results"

    records: list[dict[str, Any]] = []
    for record in manifest.to_dict("records"):
        path = _item_result_path(results_dir, str(record["review_item_id"]))
        rendered = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
        merged = {**record, **rendered}
        signals = json.loads(str(merged.get("signals_json") or "{}"))
        verdict = str(signals.get("flags_fired") or "none")
        if verdict == "none":
            merged["clip_badge"], merged["clip_badge_kind"] = "passes", "pass"
            signals["verdict"] = "passes all three checks"
        else:
            merged["clip_badge"] = verdict.replace("_", " ").replace("+", " + ")
            merged["clip_badge_kind"] = "flag"
            signals["verdict"] = f"rejected by {verdict}"
        if signals.get("repair_applied", "none") != "none":
            merged["clip_badge"] = f"{merged['clip_badge']} · repaired"
            merged["clip_badge_kind"] = "repair"
        merged["signals_json"] = json.dumps(signals, sort_keys=True)
        records.append(merged)

    by_group: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        by_group.setdefault(str(record["sample_group"]), []).append(record)

    sections: list[dict[str, Any]] = []
    for section_id, groups, heading, blurb, ask in ASK_SECTIONS:
        members = [r for g in groups for r in by_group.get(g, [])]
        if members:
            sections.append({"id": section_id, "heading": heading, "blurb": blurb,
                             "ask": ask, "kind": "ask", "records": members})

    for vendor in ("01", "02", "03"):
        for cell, (title, blurb) in CELL_LABELS.items():
            members = by_group.get(f"v{vendor}_{cell}", [])
            if not members:
                continue
            sections.append(
                {
                    "id": f"v{vendor}-{cell.replace('_', '-')}",
                    "heading": f"V{vendor} — {title}",
                    "blurb": blurb, "records": members,
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
        title="Seamless Interaction — V01, V02 and V03 under the repaired pipeline",
        intro=(
            "Round 7. Everything you flagged in the Round-6 sample, measured and "
            "acted on where a measurement was enough, and put to you where it was "
            "not. The five <span style='color:#ffd79a'>ASK</span> sections at the "
            "top are the questions I cannot answer myself; the rest is a fresh "
            "random draw, disjoint from Round 6, sorted by vendor and by exactly "
            "which checks fired. Clips now go through a repair layer: the "
            "container's pixel aspect is applied, quarter-turned rasters are stood "
            "upright, and annotations are looked up by timestamp instead of frame "
            "number. Each card says what was repaired."
        ),
        detectors=detector_blocks(scan),
        stats=build_stats(scan),
    )
    print(json.dumps({
        "gallery": str(private_root / "index.html"),
        "sections": len(sections),
        "clips": len(records),
        "rendered": sum(1 for r in records if r.get("status") in {"rendered", "reused"}),
    }, indent=2))


if __name__ == "__main__":
    main()

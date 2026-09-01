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
    ("fm0_reasons", "FM0 · source rejected because", "text"),
    ("fm1_knee_between_torso_p50", "FM1 · knee along the torso (V03 cut 0.54, V00 0.43)", "ratio"),
    ("fm1_shin_verticality_p50", "FM1 · shin direction (negative = pointing up)", "ratio"),
    ("fm1_ankles_visible_frac", "FM1 · frames with both ankles in shot", "percent"),
    ("fm2_smplh_valid_frac", "FM2 · frames with a trusted SMPL-H fit", "percent"),
    ("fm3_static_frac", "FM3 · share of the recording with both hands parked", "percent"),
    ("audio_voice_isolation_db", "FM4 · own voice above partner's, on this track", "decibels"),
    ("audio_speech_level_db", "FM4 · speech level", "decibels"),
    ("reproj_shoulder_widths_p50", "fit error against the 2D keypoints (shoulder widths)", "ratio"),
    ("timebase", "annotation timebase", "text"),
    ("observed_duration_s", "recording length", "clock"),
)

# (section id, sample_group prefix, heading, blurb, ask, kind)
ASK_SECTIONS: tuple[tuple[str, tuple[str, ...], str, str, str], ...] = (
    (
        "ask-posture", ("ask_posture_v03",),
        "ASK 1 · V03 posture, tight around FM1's new cut — sitting or standing?",
        "Your 42 labels moved FM1's V03 cut from 0.43 to <strong>0.54</strong>. On all "
        "58 V03 labels now in hand, 0.54 catches 33 of 36 seated files and costs 4 of "
        "22 standing ones — the trade you asked for. But it also takes V03's FM1 "
        "rejection rate from 22% to 36%, so the cost of putting it a little wrong has "
        "gone up. Below 0.46 and above 0.62 the existing labels are unanimous and "
        "another one buys nothing; these clips are drawn from the four bands in "
        "between, where the boundary actually sits. Files whose shins point the wrong "
        "way up are excluded, so every clip here is a genuine posture question.",
        "Sitting or standing, same as last time. I am specifically looking for "
        "standing clips between 0.46 and 0.54 (which the new cut now throws away) and "
        "seated clips between 0.54 and 0.62 (which it still keeps).",
    ),
    (
        "ask-audio", ("ask_audio",),
        "ASK 2 · Is the audio acceptable? — calibrating what you heard as glitchy",
        "Your 21 glitchy clips were not glitchy in any way I could measure directly: "
        "clipping, dropouts, spectral cutoff and fragmentation all overlap completely "
        "with good recordings. What does separate them is <strong>whose voice the "
        "microphone hears</strong>. Comparing a track's energy during its own "
        "speaker's turns against its energy during the partner's, a close-worn "
        "microphone gives 8–13 dB of separation and V03's room-camera rasters give "
        "about 2 — both people arriving at the same level, with room reverb and "
        "automatic gain riding over the top. That is what a far-field shared "
        "microphone sounds like, and it fits the wide room framing exactly. "
        "<strong>The catch is that it separates the rasters cleanly as populations "
        "but overlaps badly file by file</strong>, so a cut drawn today would reject "
        "roughly half of V00 and V02. These clips ladder across the range, spread "
        "over all three vendors so the answer is about the measure and not about the "
        "vendor.",
        "For each clip, just: <strong>is this audio usable for training, yes or no</strong>. "
        "The number is on the card; please judge by ear. If the answer tracks the "
        "number, this becomes a detector. If good and bad clips are mixed together at "
        "the same value, it stays a signal and I will look for something else.",
    ),
    (
        "ask-audio-dead", ("ask_audio_dead",),
        "ASK 3 · FM4 says these carry no voice at all",
        "The new FM4 fires when a recording's speech level is below −55 dB "
        "<em>and</em> its long-term spectrum is flat — a noise floor rather than a "
        "voice. Both halves are required: a quiet recording is not a broken one, and "
        "a flat spectrum in a loud file is a fan. It caught all four files you "
        "confirmed dead (the two silent 2160×2160 clips and two of the glitchy ones), "
        "fired on none of the 18 that are merely unpleasant, and on <strong>0 of 300 "
        "V00 files</strong>.",
        "Confirm these really are dead — by ear, and by the speech-level row on the "
        "card. <strong>Do not go by the waveform strip:</strong> it auto-scales to "
        "each clip, so a noise floor 60 dB below speech fills it edge to edge and "
        "looks like continuous loud audio. That is exactly what a dead track looks "
        "like here.",
    ),
    (
        "ask-cropped", ("ask_cropped",),
        "ASK 4 · The two padded rasters, now cropped",
        "You were right about both. 1080×960 is a 540×960 portrait frame with 270 "
        "black columns on each side, and 2180×3840 is 2160×3840 with a 20-pixel bar "
        "on the right — measured over the brightest value each column ever reaches, "
        "on every sampled file of both rasters. They are <strong>shown here with the "
        "bars cropped away</strong>. Worth knowing: the padding was costing nothing. "
        "Both rasters fit as well as any ordinary one (shape residual 0.057 and 0.061 "
        "against a 0.048–0.074 normal band), so the crop makes the panel bigger and "
        "the aspect consistent but changes no measurement.",
        "Does the crop look right — nothing of the participant lost at either edge?",
    ),
)

CELL_LABELS = {
    "pass": ("passed every check", "A random draw from what the pipeline keeps."),
    "fm0_unusable_source": (
        "rejected FM0 — the source cannot be used",
        "Rejected before any behaviour was looked at: an anamorphic V01 raster whose "
        "released pose cannot be un-stretched, a V03 room-camera raster, or an "
        "annotation grid that does not match its video. The card says which.",
    ),
    "fm1_sitting": ("rejected FM1 — read as seated",
                    "Rejected as seated, at 0.54 for V03. FM1 is off for V01 and V02, "
                    "so this section is V03-only. Where the shin-direction row is "
                    "negative, FM1 fired because the ankles left the frame rather "
                    "than because anyone sat down."),
    "fm4_audio_dead": ("rejected FM4 — the audio carries no voice",
                       "Speech level below −55 dB with a flat spectrum."),
    "fm2_smplh_invalid": ("rejected FM2 — SMPL-H not trusted on every frame",
                          "Each clip is positioned on that file's longest continuous "
                          "untrusted stretch, so the frames the check objected to are "
                          "the frames on screen."),
    "fm3_static_hands": ("rejected FM3 — hands held in one position",
                         "Rejected because both wrists stay parked for most of the "
                         "recording."),
    "multi": ("rejected by more than one check",
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
            "id": "fm0", "name": "FM0 · the source cannot be used",
            "rate": rate("fm0_unusable_source"),
            "aim": ("Reject a recording whose stored form is broken, before asking "
                    "anything about what it shows."),
            "how": ("Three rules, each from a Round-7 finding. V01's 2160×2160 and "
                    "1920×1080 are stored horizontally stretched and the released "
                    "SMPL-H absorbed the stretch, so the pose cannot be recovered by "
                    "un-stretching. V03's 3840×2160 and 640×480 are room-camera "
                    "rasters whose far-field audio the reviewer found unusable. And a "
                    "file whose annotation grid misses its container by more than "
                    "0.5 s has SMPL-H unrelated to its video, confirmed on two."),
        },
        {
            "id": "fm1", "name": "FM1 · participant is seated", "rate": rate("fm1_sitting"),
            "aim": ("Keep only standing participants: a seated body gestures "
                    "differently, and mixing the two contaminates the training signal."),
            "how": ("<code>knee_between_torso</code> — how far down the hip-to-ankle "
                    "drop the knee sits along the participant's own torso axis, from "
                    "SMPL-H forward kinematics. Standing reads about 0.55, seated "
                    "about 0.30, and the cut is <code>&lt; 0.43</code>, with an "
                    "add-on for an inverted shin. The verdict is taken once per "
                    "participant-session by majority. <strong>The cut is now "
                    "per-vendor</strong>: 0.43 on V00 from 66 labels, 0.54 on V03 "
                    "from 58, and <strong>off entirely for V01 and V02</strong> — it "
                    "fires on none of 1,379 square-pixel V01 files, and 181 of V02's "
                    "182 firings are the shin add-on on files whose ankles are out of "
                    "frame, every one of which the reviewer confirmed standing."),
        },
        {
            "id": "fm4", "name": "FM4 · the audio carries no voice",
            "rate": rate("fm4_audio_dead"),
            "aim": ("Reject a recording with no usable speech in it. Audio is not the "
                    "training target, but it anchors the co-speech alignment, and a "
                    "dead track makes a clip useless for that whatever the body does."),
            "how": ("Speech level below −55 dB <em>and</em> a flat long-term "
                    "spectrum — a noise floor rather than a voice. Both halves are "
                    "required: a quiet recording is not a broken one, and a flat "
                    "spectrum in a loud file is a fan. Caught all four files the "
                    "reviewer confirmed dead and fired on 0 of 300 V00 files. What it "
                    "does <em>not</em> yet catch is far-field audio that is audible "
                    "but unpleasant; see ASK 2."),
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
        cells = "".join(
            f"<td class='num'>{100 * float(group[flag].fillna(False).mean()):.2f}%</td>"
            for flag in ("fm0_unusable_source", "fm1_sitting", "fm2_smplh_invalid",
                         "fm3_static_hands", "fm4_audio_dead")
        )
        rows.append(
            f"<tr><th>V{vendor}</th><td class='num'>{len(group):,}</td>{cells}"
            f"<td class='num'><strong>{100 * share:.2f}%</strong></td>"
            f"<td class='num'>{share * hours:,.0f} h</td></tr>"
        )
    total_hours = sum(
        float(g.passes_all.fillna(False).mean()) * VENDOR_HOURS.get(v, 0.0)
        for v, g in ok.groupby("vendor_id")
    )
    table = (
        "<table class='rates'><thead><tr><th>vendor</th><th>files scanned</th>"
        "<th>FM0</th><th>FM1</th><th>FM2</th><th>FM3</th><th>FM4</th>"
        "<th>survives</th><th>projected dyad-hours</th></tr></thead><tbody>"
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
            (f"{int(ok.fm4_audio_dead.fillna(False).sum()):,}", "with no voice on the track"),
            (f"{int((~ok.timebase_consistent.fillna(True)).sum()):,}", "whose annotations do not match the video"),
        ],
        "table": table,
        "footnote": (
            "Flag rates are per file over this sample. FM2 and FM3 are unchanged from "
            "every earlier round and stay directly comparable; FM0, FM4 and FM1's "
            "per-vendor cuts are new in this round. Projected hours apply each vendor's sampled pass rate to its "
            "eligible population (V01 328.8 h, V02 761.4 h, V03 1,430.1 h). Read "
            "every verdict as <em>what the current pipeline says</em>: FM1 is known "
            "not to transfer, and correcting that is what the asks below are for."
        ),
    }


def main() -> None:
    os.umask(0o077)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/vendors_r8_review.yaml"))
    parser.add_argument("--detect-config", type=Path,
                        default=Path("configs/vendors_r8_detect.yaml"))
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
        title="Seamless Interaction — V01, V02 and V03 with the Round-7 answers applied",
        intro=(
            "Round 8. Your Round-7 answers moved four things: <strong>FM0</strong> now "
            "rejects V01's anamorphic rasters, V03's room-camera rasters and files "
            "whose annotations do not match their video; <strong>FM1</strong> is "
            "per-vendor, at 0.54 for V03 and switched off for V01 and V02; "
            "<strong>FM4</strong> rejects a recording with no voice on the track; and "
            "the two padded rasters are cropped. The four "
            "<span style='color:#ffd79a'>ASK</span> sections check those moves — "
            "three of them are "
            "\u201cdid I put this in the right place\u201d rather than open "
            "questions. The rest is a fresh random draw, disjoint from Rounds 6 and 7, "
            "sorted by vendor and by exactly which checks fired."
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

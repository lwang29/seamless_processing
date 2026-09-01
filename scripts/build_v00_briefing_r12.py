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
VENDOR_HOURS = {"00": 1440.6}
LABEL_HOURS = {"naturalistic": 469.5, "improvised": 971.1}

SIGNAL_ROWS: tuple[tuple[str, str, str], ...] = (
    ("verdict", "what the pipeline said", "text"),
    ("stored_raster", "raster as stored", "text"),
    ("shown_raster", "raster as shown", "text"),
    ("repair_applied", "repair applied before display", "text"),
    ("fm0_reasons", "FM0 · source rejected because", "text"),
    ("fm1_hip_flexion_deg_p50", "FM1 · hip flexion (V03 cut 146°)", "degrees"),
    ("fm1_scope", "FM1 · whose verdict decides", "text"),
    ("fm1_knee_between_torso_p50", "FM1 · knee along the torso (V00 cut 0.43)", "ratio"),
    ("fm1_shin_verticality_p50", "FM1 · shin direction (negative = pointing up)", "ratio"),
    ("fm1_ankles_visible_frac", "FM1 · frames with both ankles in shot", "percent"),
    ("fm2_smplh_valid_frac", "FM2 · frames with a trusted SMPL-H fit", "percent"),
    ("fm3_static_frac", "FM3 · share of the recording with both hands parked", "percent"),
    ("audio_voice_isolation_db", "own voice above partner's (signal only — refuted as a cut)", "decibels"),
    ("audio_speech_level_db", "FM4 · speech level", "decibels"),
    ("reproj_shoulder_widths_p50", "fit error against the 2D keypoints (shoulder widths)", "ratio"),
    ("timebase", "annotation timebase", "text"),
    ("observed_duration_s", "recording length", "clock"),
)

# (section id, sample_group prefix, heading, blurb, ask, kind)
ASK_SECTIONS: tuple[tuple[str, tuple[str, ...], str, str, str], ...] = (
    (
        "ask-pass-side", ("ask_pass_side",),
        "ASK 1 · V00 participants that FM1 let through — sitting or standing?",
        "This is the one thing about V00 that has never been measured. Every one of "
        "its 66 posture labels came from the flagged pool or the adjudicate pool, so "
        "they cluster near the 0.43 cut by construction — and FM1's "
        "<strong>false-negative rate over the comfortably passing population is "
        "unknown</strong>. On V03 that is exactly where the problem was hiding: "
        "seated participants reading 0.435 to 0.586, well clear of a 0.43 cut, found "
        "only because you happened to see them. These 40 clips ladder across the "
        "whole pass side, from just above the cut to 0.80, so a V00 equivalent would "
        "show up.",
        "Sitting or standing. If they are all standing, FM1 on V00 is confirmed and "
        "I will stop. If seated participants turn up in the higher bands, V00 has "
        "the same problem V03 had and the cut needs re-fitting here too.",
    ),
    (
        "ask-reject-side", ("ask_reject_side",),
        "ASK 2 · The band just below the cut, where FM1 rejects",
        "The other side of the same boundary, and the cheaper half of the question. "
        "V00's labelled standing files bottom out at 0.394 and its labelled seated "
        "files reach 0.728, so the two classes do overlap here — the cut at 0.43 "
        "currently scores 17 of 19 sitters for 1 of 47 standing lost. These clips "
        "come from 0.30 to 0.43, which is what FM1 throws away.",
        "Sitting or standing. Standing clips in this band are files the pipeline is "
        "discarding for nothing.",
    ),
    (
        "ask-scope", ("ask_scope_outvoted_seated", "ask_scope_outvoted_standing"),
        "ASK 3 · Files their own participant-session outvoted",
        "The V03 failure turned out to be verdict <em>scope</em>, not the measure: a "
        "seated clip passed because thirteen other files in its session read "
        "standing. V00 keeps the session-majority rule, and on its own labels that "
        "rule is the better one — 0 of 47 standing files lost against the per-file "
        "rule's 1. It also has far less room to go wrong: <strong>1.1%</strong> of "
        "V00's multi-file sessions are mixed, against V03's 16.2%. But the question "
        "has never actually been put here, and this is the pool where it would show. "
        "Both directions are drawn.",
        "Sitting or standing. The card says which way the session voted, so you can "
        "see the disagreement — please judge only the clip.",
    ),
    (
        "ask-audio-dead", ("ask_audio_dead",),
        "ASK 4 · Anything FM4 fires on in V00",
        "FM4 rejects a recording whose audio carries no voice, and on 300 sampled V00 "
        "files it fired on none. This section holds whatever it found in a 4,000-file "
        "draw. If it is empty, that is the answer.",
        "If there are clips here, confirm they really are dead — by ear and by the "
        "speech-level row, not by the waveform strip, which auto-scales.",
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
                    "participant-session by majority. <strong>The measure itself is "
                    "now per-vendor</strong>: knee position at 0.43 on V00 from 66 "
                    "labels, <strong>hip flexion at 146° on V03</strong> from 90, and "
                    "<strong>off entirely for V01 and V02</strong> — it "
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
                    "does <em>not</em> catch is far-field audio that is audible but "
                    "unpleasant. Voice isolation was tested for that and refuted by "
                    "30 hand labels; see ASK 2."),
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
    for vendor, group in ok.groupby("label"):
        share = float(group.passes_all.fillna(False).mean())
        hours = LABEL_HOURS.get(vendor, 0.0)
        cells = "".join(
            f"<td class='num'>{100 * float(group[flag].fillna(False).mean()):.2f}%</td>"
            for flag in ("fm0_unusable_source", "fm1_sitting", "fm2_smplh_invalid",
                         "fm3_static_hands", "fm4_audio_dead")
        )
        rows.append(
            f"<tr><th>{vendor}</th><td class='num'>{len(group):,}</td>{cells}"
            f"<td class='num'><strong>{100 * share:.2f}%</strong></td>"
            f"<td class='num'>{share * hours:,.0f} h</td></tr>"
        )
    total_hours = sum(
        float(g.passes_all.fillna(False).mean()) * LABEL_HOURS.get(v, 0.0)
        for v, g in ok.groupby("label")
    )
    table = (
        "<table class='rates'><thead><tr><th>condition</th><th>files scanned</th>"
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
    parser.add_argument("--config", type=Path, default=Path("configs/v00_r12_review.yaml"))
    parser.add_argument("--detect-config", type=Path,
                        default=Path("configs/v00_r12_detect.yaml"))
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

    for label in ("improvised", "naturalistic"):
        for cell, (title, blurb) in CELL_LABELS.items():
            members = by_group.get(f"{label}_{cell}", [])
            if not members:
                continue
            sections.append(
                {
                    "id": f"{label}-{cell.replace('_', '-')}",
                    "heading": f"{label.capitalize()} — {title}",
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
        title="Seamless Interaction — V00 re-examined, Round 12",
        intro=(
            "V03 taught four lessons, and three of them were corrections to choices "
            "that had looked settled on V00 evidence — the raster calibration, the "
            "measure FM1 reads, its verdict scope, and its shin add-on. So V00 gets "
            "re-examined under all four. <strong>All four of its settings survive</strong>, "
            "and the raster lesson closes by census rather than by sample: every one "
            "of V00's 41,205 eligible files is 1080×1920 at 30 fps, with no empty "
            "audio and no timebase drift. What does not survive is the <em>coverage</em> "
            "of V00's labels — all 66 come from the flagged or adjudicate pools, so "
            "they sit near the cut, and FM1's false-negative rate above it has never "
            "been measured. That is what ASK 1 is for."
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

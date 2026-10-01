"""Carry the previous iteration's review labels forward, with the spans they judged.

The co-speech-gesture filtering run (``outputs/vibes_upper_body_v1``) collected
the project's only human labels: 1,245 accept/reject/unsure verdict records on
686 files (100 of them human, taken with video and audio; the rest model
reviewers). They answer "would we train a co-speech gesture model on this
file?", so they are *not* an annotation of the new tables — but they are the
only human judgement of motion quality on disk, and the annotation report uses
them as a weak external check of expressivity.

A verdict record carries only ``review_item_id``/``file_id`` and the
fingerprint of the review card; which 30-second spans the reviewer was shown
lives in the card's JSON sidecar (``artifacts/<run>/clips/<item>.json``: the
spans and thumbnail frames) and, for the 400 items that also got a video, in
``<item>.clip.render.json``. Both, like the verdict log itself, sit outside Git.
This script joins them once, before the old run directory is archived, so no
label loses what it was about.

It has been run, and its inputs are gone: the run directory was archived (all
but ``review_verdicts.jsonl``) and the card sidecars were deleted with the review
media on 2026-09-30. It cannot be re-run. Its two outputs are the record, and
the spans parquet is the only remaining copy of which spans were judged.

Outputs (in ``<out>``):

``gesture_review_verdicts.parquet``
    Every verdict record, unchanged, plus ``is_latest`` (last write wins per
    item, on parsed timestamps — the rule the old store used).
``gesture_review_spans.parquet``
    One row per span shown per item: ``span_source`` says whether it came from
    the card sidecar whose fingerprint matches the item's latest verdict
    (``card_sidecar``), a sidecar with a different fingerprint
    (``card_sidecar_refingerprinted``), or the last candidate table
    (``candidates_table``), plus the 30-s clip indices of the new grid it overlaps.

Usage::

    python scripts/export_legacy_review_labels.py \\
        --run-root outputs/vibes_upper_body_v1 \\
        --media-root artifacts/vibes_upper_body_v1/clips \\
        --out /simurgh/group/lw29/seamless_annotations/annotations_v1/legacy/gesture_review
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

import pandas as pd


def resolve_latest(records: pd.DataFrame) -> pd.Series:
    stamps = pd.to_datetime(records["recorded_utc"], format="mixed", utc=True, errors="coerce")
    order = records.assign(_stamp=stamps, _row=range(len(records))).sort_values(
        ["_stamp", "_row"], kind="stable", na_position="first"
    )
    latest_rows = order.groupby("review_item_id", sort=False).tail(1)["_row"]
    flag = pd.Series(False, index=records.index)
    flag.iloc[latest_rows.to_numpy()] = True
    return flag


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--media-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--clip-seconds", type=float, default=30.0)
    args = parser.parse_args(argv)

    log = args.run_root / "review_verdicts.jsonl"
    rows = [json.loads(line) for line in log.read_text(encoding="utf-8").splitlines() if line.strip()]
    verdicts = pd.DataFrame(rows)
    verdicts["reasons"] = verdicts["reasons"].map(lambda r: "|".join(r) if isinstance(r, list) else str(r or ""))
    verdicts["is_latest"] = resolve_latest(verdicts)

    latest = verdicts.loc[verdicts["is_latest"]].set_index("review_item_id")
    population = pd.read_parquet(args.run_root / "population.parquet", columns=["file_id", "nominal_fps"])
    fps = population.set_index("file_id")["nominal_fps"]
    candidates = pd.read_parquet(
        args.run_root / "candidates.parquet",
        columns=["review_item_id", "file_id", "clip_id", "start_frame", "end_frame"],
    )
    candidates = candidates.dropna(subset=["review_item_id"])

    spans: list[dict] = []
    for item, row in latest.iterrows():
        file_id = str(row["file_id"])
        card_path = args.media_root / f"{item}.json"
        clip_path = args.media_root / f"{item}.clip.render.json"
        video_start = None
        if clip_path.exists():
            video_start = json.loads(clip_path.read_text(encoding="utf-8")).get("start_frame")
        source, shown = "candidates_table", []
        if card_path.exists():
            card = json.loads(card_path.read_text(encoding="utf-8"))
            if card.get("file_id") == file_id and card.get("spans"):
                shown = [tuple(int(v) for v in span) for span in card["spans"]]
                source = ("card_sidecar" if card.get("card_fingerprint") == row.get("card_fingerprint")
                          else "card_sidecar_refingerprinted")
        if not shown:
            subset = candidates.loc[candidates["review_item_id"] == item]
            shown = [(int(a), int(b)) for a, b in zip(subset["start_frame"], subset["end_frame"])]
        rate = float(fps.get(file_id, float("nan")))
        for start, stop in shown:
            first = math.floor(start / rate / args.clip_seconds) if rate == rate else None
            last = math.floor((stop - 1) / rate / args.clip_seconds) if rate == rate else None
            spans.append({
                "review_item_id": item, "file_id": file_id, "span_source": source,
                "start_frame": start, "end_frame": stop, "fps": rate,
                "shown_as_video": video_start is not None and int(video_start) == start,
                "new_clip_index_first": first, "new_clip_index_last": last,
            })

    args.out.mkdir(parents=True, exist_ok=True)
    os.chmod(args.out, 0o700)
    for frame, name in ((verdicts, "gesture_review_verdicts.parquet"), (pd.DataFrame(spans), "gesture_review_spans.parquet")):
        path = args.out / name
        temporary = path.with_suffix(".tmp")
        frame.to_parquet(temporary, index=False)
        os.replace(temporary, path)
        os.chmod(path, 0o600)
    span_frame = pd.DataFrame(spans)
    print(json.dumps({
        "verdict_records": int(len(verdicts)),
        "items": int(verdicts["review_item_id"].nunique()),
        "human_records": int((verdicts["verdict_source"] == "human").sum()),
        "spans": int(len(span_frame)),
        "span_source": span_frame["span_source"].value_counts().to_dict() if len(span_frame) else {},
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

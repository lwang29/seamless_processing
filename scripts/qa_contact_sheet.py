"""Contact sheets of sampled clips with their annotations drawn on, for visual QA.

The posture label has file-level evidence only (hand labels of whole recordings);
bin- and clip-level accuracy was never labelled. This script draws a seeded,
stratified sample of clips (e.g. ``posture`` x ``vendor``), decodes one frame per
clip (the keyframe at or before the clip midpoint, made upright), overlays the
clip's labels, and tiles them into JPEG sheets that a person can label by eye.

It is a development tool: the sheets contain participant video frames, so they
are written under the run's private output root with mode 0600, never into the
repository.

Usage::

    python scripts/qa_contact_sheet.py --config configs/annotations_v1.yaml \\
        --by posture vendor --per-cell 12 --out qa/posture
"""

from __future__ import annotations

import argparse
import math
import os
from pathlib import Path

import numpy as np
import pandas as pd

from seamless_curation.config import load_config
from seamless_curation.corpus import stable_unit_interval
from seamless_curation.dataset import load_clips
from seamless_curation.scan_video import raster_repair
from seamless_curation.video_quality import ClipWindow, _seek_frame, upright_gray

TILE_H = 400
LABEL_COLUMNS = ("posture", "posture_confidence", "hip_flexion_deg_p50", "knee_between_p50",
                 "visible_extent", "expressivity_level")


def sample(clips: pd.DataFrame, by: list[str], per_cell: int, seed: str) -> pd.DataFrame:
    """``per_cell`` clips per stratum, chosen by a hash of the clip id (row-order free)."""

    keyed = clips.assign(_u=[stable_unit_interval(c, salt=seed) for c in clips["clip_id"]])
    return (keyed.sort_values("_u").groupby(by, dropna=False, observed=True).head(per_cell)
            .sort_values(by + ["_u"]).drop(columns="_u"))


def tile(row: pd.Series, source_root: Path) -> np.ndarray:
    import av
    import cv2

    base = source_root / str(row["recording__source_relbase"])
    record = {"video_width": row["recording__video_width"], "video_height": row["recording__video_height"],
              "raster_class": row["recording__raster_class"], "quarter_turns": row["recording__quarter_turns"]}
    repair = raster_repair(record)
    image = np.full((TILE_H, int(TILE_H * 0.6)), 40, np.uint8)
    try:
        with av.open(str(base.with_suffix(".mp4"))) as container:
            stream = container.streams.video[0]
            start = float(stream.start_time * stream.time_base) if stream.start_time is not None else 0.0
            sampled = _seek_frame(container, stream, ClipWindow(int(row["clip_index"]), float(row["start_s"]),
                                                                float(row["end_s"])), start)
            if sampled is not None and repair is not None:
                gray, _, _ = upright_gray(sampled[1].to_ndarray(format="gray"), repair, 540)
                scale = TILE_H / gray.shape[0]
                image = cv2.resize(gray, (max(1, int(gray.shape[1] * scale)), TILE_H), interpolation=cv2.INTER_AREA)
    except Exception:  # noqa: BLE001 - a QA sheet shows the failure as a blank tile
        pass
    tile_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    lines = [str(row["clip_id"])[-24:]] + [
        f"{c}={row[c]:.2f}" if isinstance(row[c], float) else f"{c}={row[c]}" for c in LABEL_COLUMNS]
    for i, text in enumerate(lines):
        cv2.putText(tile_rgb, text, (4, 16 + 16 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 255, 255), 1, cv2.LINE_AA)
    return tile_rgb


def sheet(tiles: list[np.ndarray], columns: int = 6) -> np.ndarray:
    width = max(t.shape[1] for t in tiles)
    padded = [np.pad(t, ((0, 0), (0, width - t.shape[1]), (0, 0))) for t in tiles]
    rows = math.ceil(len(padded) / columns)
    blank = np.zeros_like(padded[0])
    padded += [blank] * (rows * columns - len(padded))
    return np.vstack([np.hstack(padded[r * columns:(r + 1) * columns]) for r in range(rows)])


def main(argv: list[str] | None = None) -> int:
    import cv2

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", default="configs/annotations_v1.yaml")
    parser.add_argument("--by", nargs="+", default=["posture", "vendor"])
    parser.add_argument("--per-cell", type=int, default=12)
    parser.add_argument("--seed", default="qa-v1")
    parser.add_argument("--out", default="qa/posture", help="relative to the run's output root")
    parser.add_argument("--where", default="", help="optional pandas query on the clip table")
    parser.add_argument("--annotations", default="", help="tables directory (default: the run's annotations/)")
    args = parser.parse_args(argv)

    config = load_config(args.config)
    clips = load_clips(args.annotations or config.annotations_dir, levels=("recording",))
    clips = clips.loc[~clips["is_partial"]]
    if args.where:
        clips = clips.query(args.where)
    chosen = sample(clips, args.by, args.per_cell, args.seed)
    out = config.output_root / args.out
    out.mkdir(parents=True, exist_ok=True)
    os.chmod(out, 0o700)
    chosen[["clip_id", *args.by, *LABEL_COLUMNS]].to_csv(out / "sample.csv", index=False)
    for key, group in chosen.groupby(args.by, dropna=False, observed=True):
        name = "_".join(str(k) for k in (key if isinstance(key, tuple) else (key,)))
        tiles = [tile(row, config.source_root) for _, row in group.iterrows()]
        path = out / f"{name}.jpg"
        cv2.imwrite(str(path), sheet(tiles), [cv2.IMWRITE_JPEG_QUALITY, 80])
        os.chmod(path, 0o600)
        print(f"{path} ({len(tiles)} clips)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

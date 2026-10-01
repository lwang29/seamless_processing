"""The pixel pass: one decoded keyframe per clip, for visual-quality measures.

Everything else in the pipeline reads released arrays; this is the only stage
that decodes video, so it is separate and optional. It runs after ``scan``
(it needs each recording's clip grid and detected quarter turns) over the same
shards, one task per scan shard, and writes ``task_NNNN.video.parquet`` next to
a marker carrying the scan shard's fingerprint and membership plus its own.

Rasters are repaired before measuring — pillarbox crop, anamorphic squeeze,
quarter turn — so sharpness, clutter and camera shift are computed on the
picture as it is meant to be seen, at the same 540-px short side everywhere.
The stored pixel aspect of each anamorphic raster class was verified constant by
ffprobe on four files per vendor x raster cell (``sample_aspect`` below).
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

from .media_repair import RasterRepair, pillarbox_for

#: Stored sample aspect ratio (display width / stored width per pixel) by raster class.
SAMPLE_ASPECT: dict[str, float] = {
    "anamorphic_2160x2160": 9 / 16,
    "anamorphic_1920x1080": 81 / 256,
    "anamorphic_1012x1920": 270 / 253,
}


def raster_repair(record: Mapping[str, Any]) -> RasterRepair | None:
    width, height = record.get("video_width"), record.get("video_height")
    if pd.isna(width) or pd.isna(height):
        return None
    width, height = int(width), int(height)
    turns = record.get("quarter_turns")
    return RasterRepair(
        width=width, height=height,
        sample_aspect=SAMPLE_ASPECT.get(str(record.get("raster_class")), 1.0),
        quarter_turns=int(turns) if pd.notna(turns) else 0,
        pillarbox=pillarbox_for(width, height),
    )


def box_lookup(npz_path: Path):
    """``time_s -> stored-raster xyxy box`` (``None`` on invalid-box frames)."""

    with np.load(npz_path) as archive:
        boxes = np.asarray(archive["boxes_and_keypoints:box"], dtype=np.float64)
        valid = np.asarray(archive["boxes_and_keypoints:is_valid_box"]).reshape(-1).astype(bool)
    return boxes, valid


def measure_shard(recordings: pd.DataFrame, clips: pd.DataFrame, source_root: Path, *,
                  short_side: int = 540, log: Any = None) -> pd.DataFrame:
    """Visual measures for every clip of the given measured recordings."""

    from .video_quality import ClipWindow, measure_video

    rows: list[dict[str, Any]] = []
    by_file = {file_id: group.sort_values("clip_index") for file_id, group in clips.groupby("file_id")}
    for count, record in enumerate(recordings.to_dict("records"), start=1):
        group = by_file.get(record["file_id"])
        if group is None or group.empty:
            continue
        base = source_root / str(record["source_relbase"])
        fps = float(record["fps"])
        repair = raster_repair(record)
        started = time.perf_counter()
        results: list[dict[str, Any]]
        if repair is None or not bool(record.get("mp4_present")):
            results = [{"visual_status": "no_frame"} for _ in range(len(group))]
        else:
            try:
                boxes, valid = box_lookup(base.with_suffix(".npz"))

                def box_at(time_s: float, boxes=boxes, valid=valid, fps=fps):
                    index = int(round(time_s * fps))
                    if 0 <= index < len(boxes) and valid[index]:
                        return boxes[index]
                    return None

                windows = [ClipWindow(int(k), float(a), float(b))
                           for k, a, b in zip(group["clip_index"], group["start_s"], group["end_s"])]
                results = measure_video(base.with_suffix(".mp4"), windows, repair=repair,
                                        box_at=box_at, short_side=short_side)
            except Exception as error:  # never let one file kill the shard
                results = [{"visual_status": "decode_error", "visual_error": f"{type(error).__name__}: {error}"[:200]}
                           for _ in range(len(group))]
        seconds = time.perf_counter() - started
        for clip_row, measured in zip(group.to_dict("records"), results):
            rows.append({"clip_id": clip_row["clip_id"], "file_id": record["file_id"],
                         "video_seconds": seconds / max(1, len(group)), **measured})
        if log is not None and count % 50 == 0:
            log(f"{count}/{len(recordings)} recordings")
    return pd.DataFrame(rows)


def marker(shard_marker: Path, extra: Mapping[str, Any]) -> dict[str, Any]:
    recorded = json.loads(shard_marker.read_text(encoding="utf-8"))
    return {"scan_fingerprint": recorded.get("fingerprint"), "scan_membership": recorded.get("membership"),
            **dict(extra)}

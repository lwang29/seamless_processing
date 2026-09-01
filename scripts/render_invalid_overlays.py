#!/usr/bin/env python3
"""Render private 2D-keypoint overlays around selected invalid-frame runs."""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd
import yaml


BODY_EDGES = (
    (0, 1), (0, 2), (1, 3), (2, 4), (5, 6), (5, 7), (7, 9),
    (6, 8), (8, 10), (5, 11), (6, 12), (11, 12), (11, 13),
    (13, 15), (12, 14), (14, 16),
)
HAND_LOCAL_EDGES = (
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
)


def _point(keypoints: np.ndarray, index: int) -> tuple[int, int] | None:
    x, y, confidence = keypoints[index]
    if not np.isfinite([x, y, confidence]).all() or confidence <= 0:
        return None
    return int(round(float(x))), int(round(float(y)))


def _draw_edges(
    frame: np.ndarray,
    keypoints: np.ndarray,
    edges: tuple[tuple[int, int], ...],
    offset: int,
    color: tuple[int, int, int],
    thickness: int,
) -> None:
    height, width = frame.shape[:2]
    for first, second in edges:
        p1 = _point(keypoints, offset + first)
        p2 = _point(keypoints, offset + second)
        if p1 is None or p2 is None:
            continue
        if not (
            0 <= p1[0] < width and 0 <= p1[1] < height
            and 0 <= p2[0] < width and 0 <= p2[1] < height
        ):
            continue
        cv2.line(frame, p1, p2, color, thickness, cv2.LINE_AA)


def _draw_frame(
    frame: np.ndarray,
    frame_index: int,
    keypoints: np.ndarray,
    box: np.ndarray,
    smplh_valid: bool,
    movement_valid: bool | None,
    box_valid: bool,
) -> np.ndarray:
    height, width = frame.shape[:2]
    thickness = max(2, int(round(min(width, height) / 540)))
    radius = max(2, thickness + 1)
    any_invalid = not smplh_valid or not box_valid or movement_valid is False
    state_color = (0, 0, 255) if any_invalid else (0, 200, 0)

    if np.isfinite(box).all():
        x1, y1, x2, y2 = (int(round(float(value))) for value in box)
        cv2.rectangle(frame, (x1, y1), (x2, y2), state_color, thickness)

    _draw_edges(frame, keypoints, BODY_EDGES, 0, (0, 255, 255), thickness)
    _draw_edges(frame, keypoints, HAND_LOCAL_EDGES, 91, (255, 100, 0), thickness)
    _draw_edges(frame, keypoints, HAND_LOCAL_EDGES, 112, (0, 120, 255), thickness)
    for start, stop, color in ((0, 23, (0, 255, 255)), (91, 112, (255, 100, 0)), (112, 133, (0, 120, 255))):
        for index in range(start, stop):
            point = _point(keypoints, index)
            if point is not None and 0 <= point[0] < width and 0 <= point[1] < height:
                cv2.circle(frame, point, radius, color, -1, cv2.LINE_AA)

    movement_text = "missing" if movement_valid is None else str(int(movement_valid))
    lines = (
        f"annotation frame: {frame_index}",
        f"smplh:is_valid={int(smplh_valid)}",
        f"movement:is_valid={movement_text}",
        f"box:is_valid={int(box_valid)}",
    )
    font_scale = max(0.65, min(width, height) / 1100)
    line_height = int(round(34 * font_scale))
    origin_x = max(12, thickness * 4)
    origin_y = max(36, line_height)
    overlay = frame.copy()
    cv2.rectangle(
        overlay,
        (origin_x - 8, origin_y - line_height),
        (origin_x + int(390 * font_scale), origin_y + line_height * (len(lines) - 1) + 8),
        (0, 0, 0),
        -1,
    )
    cv2.addWeighted(overlay, 0.60, frame, 0.40, 0, frame)
    for line_index, line in enumerate(lines):
        cv2.putText(
            frame,
            line,
            (origin_x, origin_y + line_index * line_height),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (255, 255, 255),
            thickness,
            cv2.LINE_AA,
        )
    if any_invalid:
        border = max(6, thickness * 3)
        cv2.rectangle(frame, (0, 0), (width - 1, height - 1), state_color, border)
    return frame


def _resize_dimensions(width: int, height: int, max_height: int) -> tuple[int, int]:
    if height <= max_height:
        return width - width % 2, height - height % 2
    scale = max_height / height
    output_width = int(round(width * scale))
    return output_width - output_width % 2, max_height - max_height % 2


def _contact_grid(frames: list[np.ndarray], columns: int = 3) -> np.ndarray:
    """Tile equal-sized annotated frames without exposing separate frame files."""
    if not frames:
        raise ValueError("contact grid requires at least one frame")
    rows: list[np.ndarray] = []
    height, width = frames[0].shape[:2]
    blank = np.zeros((height, width, 3), dtype=frames[0].dtype)
    for offset in range(0, len(frames), columns):
        row = frames[offset : offset + columns]
        row += [blank] * (columns - len(row))
        rows.append(np.concatenate(row, axis=1))
    return np.concatenate(rows, axis=0)


def _render_one(
    record: dict[str, str],
    source_root: Path,
    output_root: Path,
    fps: float,
    max_height: int,
) -> dict[str, Any]:
    base = source_root / record["source_relbase"]
    start = int(record["start_frame"])
    end = int(record["end_frame"])
    with np.load(base.with_suffix(".npz"), allow_pickle=False) as archive:
        keypoints = archive["boxes_and_keypoints:keypoints"]
        boxes = archive["boxes_and_keypoints:box"]
        box_valid = archive["boxes_and_keypoints:is_valid_box"].astype(bool)
        smplh_valid = archive["smplh:is_valid"].astype(bool)
        movement_valid = (
            archive["movement:is_valid"].reshape(-1).astype(bool)
            if "movement:is_valid" in archive.files
            else None
        )
    capture = cv2.VideoCapture(str(base.with_suffix(".mp4")))
    if not capture.isOpened():
        raise RuntimeError(f"OpenCV could not open {base.with_suffix('.mp4')}")
    video_frame_count = int(round(capture.get(cv2.CAP_PROP_FRAME_COUNT)))
    end = min(end, len(keypoints), video_frame_count)
    if not 0 <= start < end:
        capture.release()
        raise ValueError(f"invalid frame interval [{start}, {end}) for {base.name}")
    capture.set(cv2.CAP_PROP_POS_FRAMES, start)
    input_width = int(round(capture.get(cv2.CAP_PROP_FRAME_WIDTH)))
    input_height = int(round(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    output_width, output_height = _resize_dimensions(input_width, input_height, max_height)

    output_path = output_root / f"{record['clip_id']}.mp4"
    temporary = output_root / f".{record['clip_id']}.tmp.mp4"
    command = [
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
        "-f", "rawvideo", "-pix_fmt", "bgr24",
        "-s", f"{output_width}x{output_height}", "-r", str(fps),
        "-i", "-", "-an", "-c:v", "libx264", "-preset", "fast",
        "-crf", "22", "-pix_fmt", "yuv420p", "-movflags", "+faststart",
        str(temporary),
    ]
    encoder = subprocess.Popen(command, stdin=subprocess.PIPE)
    contact_frames: list[np.ndarray] = []
    target_start = int(record["target_run_start"])
    target_end = int(record["target_run_end"])
    contact_targets = {
        start,
        max(start, target_start - 1),
        min(end - 1, target_start),
        min(end - 1, target_start + 5),
        min(end - 1, target_end - 1),
        end - 1,
    }
    written = 0
    try:
        for frame_index in range(start, end):
            ok, frame = capture.read()
            if not ok:
                raise RuntimeError(f"video decode stopped at frame {frame_index}")
            annotated = _draw_frame(
                frame,
                frame_index,
                keypoints[frame_index],
                boxes[frame_index],
                bool(smplh_valid[frame_index]),
                None if movement_valid is None else bool(movement_valid[frame_index]),
                bool(box_valid[frame_index]),
            )
            if (input_width, input_height) != (output_width, output_height):
                annotated = cv2.resize(
                    annotated, (output_width, output_height), interpolation=cv2.INTER_AREA
                )
            if frame_index in contact_targets:
                contact_frames.append(annotated.copy())
            assert encoder.stdin is not None
            encoder.stdin.write(annotated.tobytes())
            written += 1
    finally:
        capture.release()
        if encoder.stdin is not None:
            encoder.stdin.close()
        return_code = encoder.wait()
    if return_code != 0:
        raise RuntimeError(f"ffmpeg exited with status {return_code}")
    temporary.replace(output_path)
    output_path.chmod(0o600)

    contact_path = output_root / f"{record['clip_id']}.contact.jpg"
    contact_temporary = output_root / f".{record['clip_id']}.tmp.jpg"
    if contact_frames:
        contact = _contact_grid(contact_frames)
        if not cv2.imwrite(str(contact_temporary), contact):
            raise RuntimeError(f"could not write {contact_temporary}")
        contact_temporary.replace(contact_path)
        contact_path.chmod(0o600)

    return {
        **record,
        "rendered_frames": written,
        "input_width": input_width,
        "input_height": input_height,
        "video_frame_count": video_frame_count,
        "annotation_frame_count": len(keypoints),
        "output_width": output_width,
        "output_height": output_height,
        "output_path": str(output_path),
        "contact_path": str(contact_path),
    }


def main() -> None:
    os.umask(0o077)
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/recon.yaml"))
    parser.add_argument("--candidates", type=Path, default=Path("configs/invalid_clip_candidates.csv"))
    parser.add_argument("--limit", type=int)
    parser.add_argument("--max-height", type=int, default=960)
    args = parser.parse_args()

    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    source_root = Path(config["source_root"]).resolve()
    output_root = Path(config["outputs"]["private_media_root"])
    output_root.mkdir(parents=True, exist_ok=True, mode=0o700)
    output_root.chmod(0o700)
    records = list(csv.DictReader(args.candidates.open(newline="", encoding="utf-8")))
    if args.limit is not None:
        records = records[: args.limit]
    results = [
        _render_one(
            record,
            source_root,
            output_root,
            float(config["window"]["fps"]),
            args.max_height,
        )
        for record in records
    ]
    manifest_name = f"render_manifest.{args.candidates.stem}.json"
    manifest_path = output_root / manifest_name
    temporary = output_root / f".{manifest_name}.tmp"
    temporary.write_text(json.dumps(results, indent=2), encoding="utf-8")
    temporary.replace(manifest_path)
    manifest_path.chmod(0o600)
    print(f"rendered {len(results)} private clips under {output_root}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Probe bounded dev media/annotation headers without decoding full streams."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from concurrent.futures import ThreadPoolExecutor
from fractions import Fraction
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml


def _ffprobe(path: Path) -> dict[str, Any]:
    command = [
        "ffprobe",
        "-v",
        "error",
        "-show_streams",
        "-show_format",
        "-of",
        "json",
        str(path),
    ]
    completed = subprocess.run(command, check=True, capture_output=True, text=True)
    return json.loads(completed.stdout)


def _stream(payload: dict[str, Any], codec_type: str) -> dict[str, Any]:
    return next(
        (stream for stream in payload.get("streams", []) if stream.get("codec_type") == codec_type),
        {},
    )


def _fraction(value: str | None) -> float:
    if not value or value == "0/0":
        return float("nan")
    return float(Fraction(value))


def _float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _probe_one(row: dict[str, str], source_root: Path, nominal_fps: float) -> dict[str, Any]:
    base = source_root / row["source_relbase"]
    video_payload = _ffprobe(base.with_suffix(".mp4"))
    wav_payload = _ffprobe(base.with_suffix(".wav"))
    video_stream = _stream(video_payload, "video")
    embedded_audio = _stream(video_payload, "audio")
    wav_stream = _stream(wav_payload, "audio")
    video_format = video_payload.get("format", {})
    wav_format = wav_payload.get("format", {})

    with np.load(base.with_suffix(".npz"), allow_pickle=False) as archive:
        keys = set(archive.files)

        def key_length(key: str) -> int | None:
            return len(archive[key]) if key in keys else None

        n_frames_smplh = key_length("smplh:is_valid")
        n_frames_movement = key_length("movement:is_valid")
        n_frames_box = key_length("boxes_and_keypoints:is_valid_box")

    avg_fps = _fraction(video_stream.get("avg_frame_rate"))
    r_fps = _fraction(video_stream.get("r_frame_rate"))
    wav_duration = _float(wav_format.get("duration", wav_stream.get("duration")))
    video_duration = _float(video_format.get("duration", video_stream.get("duration")))
    return {
        **row,
        "has_smplh_valid": n_frames_smplh is not None,
        "has_movement_valid": n_frames_movement is not None,
        "has_box_valid": n_frames_box is not None,
        "n_frames_smplh": n_frames_smplh,
        "n_frames_movement": n_frames_movement,
        "n_frames_box": n_frames_box,
        "video_width": _int(video_stream.get("width")),
        "video_height": _int(video_stream.get("height")),
        "video_pix_fmt": video_stream.get("pix_fmt"),
        "video_codec": video_stream.get("codec_name"),
        "video_r_fps": r_fps,
        "video_avg_fps": avg_fps,
        "video_time_base": video_stream.get("time_base"),
        "video_start_s": _float(video_stream.get("start_time")),
        "video_duration_s": video_duration,
        "video_nb_frames": _int(video_stream.get("nb_frames")),
        "video_timecode": video_stream.get("tags", {}).get("timecode"),
        "embedded_audio_sample_rate": _int(embedded_audio.get("sample_rate")),
        "embedded_audio_channels": _int(embedded_audio.get("channels")),
        "embedded_audio_start_s": _float(embedded_audio.get("start_time")),
        "embedded_audio_duration_s": _float(embedded_audio.get("duration")),
        "wav_codec": wav_stream.get("codec_name"),
        "wav_sample_fmt": wav_stream.get("sample_fmt"),
        "wav_sample_rate": _int(wav_stream.get("sample_rate")),
        "wav_channels": _int(wav_stream.get("channels")),
        "wav_start_s": _float(wav_stream.get("start_time")),
        "wav_duration_s": wav_duration,
        "duration_mismatch_s": (
            abs(n_frames_smplh / nominal_fps - wav_duration)
            if n_frames_smplh is not None
            else float("nan")
        ),
        "video_wav_duration_delta_s": video_duration - wav_duration,
        "video_frame_count_delta": (
            _int(video_stream.get("nb_frames")) - n_frames_smplh
            if _int(video_stream.get("nb_frames")) is not None and n_frames_smplh is not None
            else None
        ),
        "size_json_bytes": base.with_suffix(".json").stat().st_size,
        "size_mp4_bytes": base.with_suffix(".mp4").stat().st_size,
        "size_npz_bytes": base.with_suffix(".npz").stat().st_size,
        "size_wav_bytes": base.with_suffix(".wav").stat().st_size,
    }


def _load_union(config: dict[str, Any]) -> list[dict[str, str]]:
    sources = {
        "main": Path(config["sample"]["manifest"]),
        "dyad": Path(config["dyad_audit"]["manifest"]),
        "movement": Path(config["movement_audit"]["manifest"]),
    }
    rows: dict[str, dict[str, str]] = {}
    memberships: dict[str, set[str]] = {}
    for membership, path in sources.items():
        frame = pd.read_csv(path, dtype=str)
        for record in frame.to_dict(orient="records"):
            file_id = record["file_id"]
            rows.setdefault(file_id, record)
            memberships.setdefault(file_id, set()).add(membership)
    for file_id, record in rows.items():
        record["audit_membership"] = "+".join(sorted(memberships[file_id]))
    return sorted(rows.values(), key=lambda row: (row["label"], row["file_id"]))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/recon.yaml"))
    parser.add_argument("--output", type=Path, default=Path("outputs/recon/media_probe.parquet"))
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    if not 1 <= args.workers <= 4:
        raise ValueError("Session 1 media probing is capped at four workers")

    config_bytes = args.config.read_bytes()
    config = yaml.safe_load(config_bytes)
    rows = _load_union(config)
    source_root = Path(config["source_root"]).resolve()
    nominal_fps = float(config["window"]["fps"])
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        observations = list(
            executor.map(
                lambda row: _probe_one(row, source_root, nominal_fps),
                rows,
            )
        )
    frame = pd.DataFrame(observations).sort_values(["label", "file_id"])
    frame["config_hash"] = hashlib.sha256(config_bytes).hexdigest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    frame.to_parquet(temporary, index=False)
    temporary.replace(args.output)
    print(f"wrote {len(frame)} rows to {args.output}")


if __name__ == "__main__":
    main()

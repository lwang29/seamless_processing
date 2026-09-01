#!/usr/bin/env python3
"""File-level vocabulary proxies over a bounded V00 candidate pool.

Purpose: give the targeted review sample real extremes for the reviewer's own
vocabulary — framing, roll, sitting, static hands, desync lag — instead of only
the Session-1 signal extremes.

Scope is a bounded pool rather than all 41k V00 files, and the summary records
the pool fraction, so an extreme found here is honestly "the most extreme in a
14% sample" and not "the corpus maximum". Frames are read on a stride because
every one of these five questions is about the recording as a whole.

Audio is read in three short windows rather than whole files: the
audio/video-lag proxy is a global property, and reading 60 seconds instead of
250 makes the pool four times cheaper.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from fractions import Fraction
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import soundfile as sf
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from seamless_curation.features import as_binary_mask, guard_invalid_runs, valid_frac  # noqa: E402
from seamless_curation.review_renderer import load_vad  # noqa: E402
from seamless_curation.m3_features import speaking_fraction  # noqa: E402
from seamless_curation.vocab_proxies import (  # noqa: E402
    audio_envelope_at_frame_rate,
    audiovisual_lag,
    framing_signals,
    hand_activity_signals,
    mouth_open_signal,
    posture_signals,
    roll_signals,
)


def _worktree() -> Path:
    here = Path(__file__).resolve().parent
    for candidate in (here, *here.parents):
        if (candidate / ".git-session").is_dir():
            return candidate
    raise RuntimeError("could not locate .git-session")


def _git_provenance(worktree: Path) -> dict[str, Any]:
    base = ["git", f"--git-dir={worktree / '.git-session'}", f"--work-tree={worktree}"]
    sha = subprocess.run([*base, "rev-parse", "HEAD"], check=True, capture_output=True, text=True)
    status = subprocess.run(
        [*base, "status", "--porcelain", "--untracked-files=normal"],
        check=True, capture_output=True, text=True,
    )
    return {"git_sha": sha.stdout.strip(), "git_dirty": bool(status.stdout.strip())}


def build_pool(inventory: Path, *, pool_size: int, seed: int) -> pd.DataFrame:
    """V00 rows eligible for review, minus charades, as a seeded random pool."""

    frame = pd.read_parquet(inventory)
    eligible = frame.loc[
        frame.vendor_id.eq("00")
        & frame.all_modalities_present.astype(bool)
        & frame.probe_status.eq("ok")
        & frame.video_stream_present.astype(bool)
        & frame.observed_duration_s.notna()
        & frame.observed_duration_s.ge(30.0)
        & frame.video_nb_frames.notna()
        & frame.interaction_type.ne("charades")
    ].copy()
    eligible = eligible.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    pool = eligible.iloc[:pool_size].copy()
    pool["pool_index"] = np.arange(len(pool))
    return pool


def _audio_windows(
    wav_path: Path,
    *,
    fps: float,
    n_frames: int,
    windows: int,
    window_s: float,
) -> list[tuple[int, np.ndarray, int]]:
    """Read a few short spans as (start_frame, mono samples, sample rate)."""

    if not wav_path.is_file():
        return []
    try:
        with sf.SoundFile(wav_path) as audio:
            sample_rate = int(audio.samplerate)
            total = len(audio)
            if sample_rate <= 0 or total <= 0:
                return []
            span_samples = int(round(window_s * sample_rate))
            duration_frames = int(min(n_frames, total / sample_rate * fps))
            span_frames = int(round(window_s * fps))
            if duration_frames < span_frames or span_frames < 1:
                return []
            starts = np.linspace(
                0, max(0, duration_frames - span_frames), max(1, windows)
            ).astype(int)
            result: list[tuple[int, np.ndarray, int]] = []
            for start_frame in sorted(set(int(value) for value in starts)):
                offset = int(round(start_frame / fps * sample_rate))
                if offset >= total:
                    continue
                audio.seek(offset)
                samples = audio.read(span_samples, dtype="float32", always_2d=True)
                if len(samples) < span_samples // 2:
                    continue
                result.append((start_frame, samples.mean(axis=1), sample_rate))
            return result
    except Exception:
        return []


def process_file(
    row: Any,
    source_root: Path,
    *,
    stride: int,
    guard: int,
    audio_windows: int,
    audio_window_s: float,
) -> dict[str, Any]:
    base = source_root / row.source_relbase
    record: dict[str, Any] = {
        "file_id": row.file_id,
        "source_relbase": row.source_relbase,
        "label": row.label,
        "split": row.split,
        "interaction_type": row.interaction_type,
        "session_id": row.session_id,
        "interaction_id": row.interaction_id,
        "participant_id": row.participant_id,
        "observed_duration_s": float(row.observed_duration_s),
        "video_nb_frames": int(row.video_nb_frames),
        "width": int(row.video_width),
        "height": int(row.video_height),
    }
    try:
        fps = float(Fraction(str(row.video_avg_frame_rate)))
    except (ValueError, ZeroDivisionError, TypeError):
        fps = float("nan")
    record["fps"] = fps
    if not np.isfinite(fps) or fps <= 0:
        record["status"] = "unusable_fps"
        return record

    try:
        with np.load(base.with_suffix(".npz"), allow_pickle=False) as archive:
            names = set(archive.files)
            if "boxes_and_keypoints:keypoints" not in names:
                record["status"] = "missing_keypoints"
                return record
            keypoints = np.asarray(archive["boxes_and_keypoints:keypoints"][::stride])
            boxes = (
                np.asarray(archive["boxes_and_keypoints:box"][::stride])
                if "boxes_and_keypoints:box" in names
                else None
            )
            smplh_full, _ = as_binary_mask(archive["smplh:is_valid"]) if "smplh:is_valid" in names else (None, "")
            box_full, _ = (
                as_binary_mask(archive["boxes_and_keypoints:is_valid_box"])
                if "boxes_and_keypoints:is_valid_box" in names else (None, "")
            )
            movement_full, _ = (
                as_binary_mask(archive["movement:is_valid"])
                if "movement:is_valid" in names else (None, "")
            )
            annotation_frames = int(archive["boxes_and_keypoints:keypoints"].shape[0])
    except (OSError, ValueError, KeyError) as exc:
        record["status"] = f"npz_error:{type(exc).__name__}"
        return record

    record["annotation_frames"] = annotation_frames
    record["frames_sampled"] = int(len(keypoints))
    for name, mask in (("smplh", smplh_full), ("box", box_full), ("movement", movement_full)):
        record[f"valid_frac_{name}"] = valid_frac(mask, name=name).value
        record[f"valid_frac_{name}_guarded"] = valid_frac(
            guard_invalid_runs(mask, guard), name=name
        ).value
    if smplh_full is not None and box_full is not None and len(smplh_full) == len(box_full):
        record["valid_frac_smplh_and_box"] = float(
            (np.asarray(smplh_full, bool) & np.asarray(box_full, bool)).mean()
        )

    width, height = record["width"], record["height"]
    record.update(framing_signals(keypoints, boxes, width=width, height=height))
    record.update(roll_signals(keypoints))
    record.update(posture_signals(keypoints))
    record.update(
        hand_activity_signals(
            keypoints, width=width, height=height, fps=fps, frame_stride=stride
        )
    )

    intervals = load_vad(base.with_suffix(".json"))
    record.update(
        {
            f"file_{key}": value
            for key, value in speaking_fraction(
                intervals, n_frames=annotation_frames, fps=fps
            ).items()
        }
    )

    # Audio/video lag on the un-strided frames inside each audio window, so the
    # mouth signal keeps its full temporal resolution where it matters.
    lag_results: list[dict[str, Any]] = []
    windows = _audio_windows(
        base.with_suffix(".wav"), fps=fps, n_frames=annotation_frames,
        windows=audio_windows, window_s=audio_window_s,
    )
    if windows:
        try:
            with np.load(base.with_suffix(".npz"), allow_pickle=False) as archive:
                for start_frame, samples, sample_rate in windows:
                    count = int(round(len(samples) / sample_rate * fps))
                    stop = min(annotation_frames, start_frame + count)
                    if stop - start_frame < 60:
                        continue
                    window_keypoints = np.asarray(
                        archive["boxes_and_keypoints:keypoints"][start_frame:stop]
                    )
                    mouth, diagnostics = mouth_open_signal(window_keypoints)
                    envelope = audio_envelope_at_frame_rate(
                        samples, sample_rate, fps=fps, n_frames=stop - start_frame
                    )
                    outcome = audiovisual_lag(mouth, envelope)
                    outcome.update(diagnostics)
                    lag_results.append(outcome)
        except (OSError, ValueError, KeyError) as exc:
            record["av_lag_status"] = f"npz_reread_error:{type(exc).__name__}"

    usable = [item for item in lag_results if item.get("av_lag_status") == "ok"]
    if usable:
        best = max(usable, key=lambda item: item["av_lag_peak_r"])
        record["av_lag_windows_ok"] = len(usable)
        record["av_lag_windows_attempted"] = len(lag_results)
        # Median lag across windows is the file-level answer; a single window's
        # argmax is too noisy to stand alone.
        record["av_lag_frames_median"] = float(
            np.median([item["av_lag_frames"] for item in usable])
        )
        record["av_lag_frames_spread"] = float(
            np.max([item["av_lag_frames"] for item in usable])
            - np.min([item["av_lag_frames"] for item in usable])
        )
        for key in (
            "av_lag_frames", "av_lag_peak_r", "av_lag_zero_r",
            "av_lag_peak_minus_zero_r", "av_lag_peak_prominence_r",
            "mouth_usable_frame_frac", "mouth_inner_outer_agreement_r",
            "mouth_open_median",
        ):
            record[f"best_{key}"] = best.get(key)
        record["av_lag_status"] = "ok"
    elif lag_results:
        record["av_lag_status"] = lag_results[0].get("av_lag_status", "unknown")
        record["av_lag_windows_ok"] = 0
        record["av_lag_windows_attempted"] = len(lag_results)
    else:
        record.setdefault("av_lag_status", "no_audio_windows")

    record["status"] = "ok"
    return record


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/v00_pool_signals.yaml"))
    parser.add_argument("--task-index", type=int, default=0)
    parser.add_argument("--task-count", type=int)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()

    worktree = _worktree()
    config = yaml.safe_load((worktree / args.config).read_text(encoding="utf-8"))
    source_root = (worktree / str(config["source_root"]).lstrip("./")).resolve()
    inventory = worktree / str(config["inventory"]).lstrip("./")
    out_root = worktree / str(config["outputs"]["root"]).lstrip("./")
    shard_dir = out_root / str(config["outputs"]["shards"])
    shard_dir.mkdir(parents=True, exist_ok=True)
    stride = int(config["sampling"]["frame_stride"])
    guard = int(config["sampling"]["invalid_guard_band_frames"])
    audio_windows = int(config["sampling"]["audio_windows"])
    audio_window_s = float(config["sampling"]["audio_window_s"])
    task_count = args.task_count or int(config["task_count"])

    pool = build_pool(
        inventory,
        pool_size=int(config["sampling"]["pool_size"]),
        seed=int(config["sampling"]["pool_seed"]),
    )
    assignment = pool.pool_index % task_count
    shard = pool.loc[assignment.eq(args.task_index)].copy()
    if args.limit is not None:
        shard = shard.iloc[: args.limit]

    started = time.perf_counter()
    rows = [
        process_file(
            row, source_root, stride=stride, guard=guard,
            audio_windows=audio_windows, audio_window_s=audio_window_s,
        )
        for row in shard.itertuples(index=False)
    ]
    elapsed = time.perf_counter() - started

    frame = pd.DataFrame(rows)
    shard_path = shard_dir / f"task_{args.task_index:04d}.parquet"
    temporary = shard_path.with_name(f".{shard_path.name}.tmp")
    frame.to_parquet(temporary, index=False)
    temporary.replace(shard_path)
    marker = {
        "status": "complete",
        "task_index": args.task_index,
        "task_count": task_count,
        "rows": len(frame),
        "wall_s": elapsed,
        "files_per_s": len(frame) / elapsed if elapsed > 0 else None,
        "pool_size": int(config["sampling"]["pool_size"]),
        "frame_stride": stride,
        **_git_provenance(worktree),
    }
    (shard_dir / f"task_{args.task_index:04d}.complete.json").write_text(
        json.dumps(marker, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(marker, sort_keys=True))


if __name__ == "__main__":
    main()

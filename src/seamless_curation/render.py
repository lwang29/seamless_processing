"""Stage 4: produce the two review artefacts for each review item.

``<review_item_id>.card.png``
    The card. Always produced; it is what the review app shows first and what a
    bulk pass reads.

``<review_item_id>.clip.mp4``
    A 30-second excerpt of the item's best accepted span, with the participant's
    own released audio muxed, for the audio-motion synchronisation check. It is
    rendered by the existing, hardened
    :func:`seamless_curation.review_renderer.render_record`, so it inherits the
    annotation-timebase handling, the raster repairs, the one-keyframe-per-second
    encoding that makes the scrubber usable, and the fingerprinted reuse.

Both are written under ``artifacts/`` with mode 0600 in a 0700 directory:
participant media never leaves the cluster and is never copied into ``outputs/``.

Costs, measured on this cluster: a card is ~6-8 s (twelve ffmpeg keyframe seeks
dominate) and ~0.4 MB; a clip is ~25-40 s and ~2 MB. ``--card-only`` therefore
triages five times as many items per node-hour, which is the right default when
the candidate pool is far larger than the review budget.
"""

from __future__ import annotations

import json
import os
import subprocess
from dataclasses import asdict
from hashlib import sha256
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

import cv2
import numpy as np
import pandas as pd

from .config import RunConfig
from .gesture import build_tracks, speech_mask
from .media_repair import RasterRepair, body_roll_deg, ffmpeg_repair_filter, quarter_turns_from_roll, pillarbox_for, probe_sample_aspect, repair_points
from .review_card import (
    CardLayout,
    compose_card,
    draw_keypoint_overlay,
    draw_pose_panel,
    draw_timeline,
    upper_body_crop_box,
)
from .scan import NPZ_KEYS, read_bundle

CARD_VERSION = 4


def _digest(payload: Mapping[str, Any]) -> str:
    return sha256(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()[:16]


def _grab_crop(
    video_path: Path,
    seconds: float,
    crop: tuple[int, int, int, int],
    size: tuple[int, int],
    repair_filter: str,
) -> np.ndarray | None:
    """One BGR frame near ``seconds``, repaired, cropped, and scaled to ``size``.

    Cropping inside ffmpeg rather than after decode keeps a 4K V03 frame from
    crossing the pipe. ``-ss`` before ``-i`` is a keyframe seek, which is what
    makes twelve grabs cost seven seconds instead of a full decode.
    """

    left, top, right, bottom = crop
    chain = [] if not repair_filter else [repair_filter]
    chain.append(f"crop={right - left}:{bottom - top}:{left}:{top}")
    chain.append(f"scale={size[0]}:{size[1]}")
    command = [
        "ffmpeg", "-hide_banner", "-loglevel", "error",
        "-ss", f"{max(0.0, seconds):.3f}", "-i", str(video_path),
        "-frames:v", "1", "-vf", ",".join(chain),
        "-f", "rawvideo", "-pix_fmt", "bgr24", "-",
    ]
    try:
        completed = subprocess.run(command, capture_output=True, timeout=120)
    except (OSError, subprocess.SubprocessError):
        return None
    expected = size[0] * size[1] * 3
    if completed.returncode != 0 or len(completed.stdout) < expected:
        return None
    return np.frombuffer(completed.stdout[:expected], dtype=np.uint8).reshape(size[1], size[0], 3).copy()


def _partner_speech(source_root: Path, relbase: str | None, frames: int, fps: float) -> np.ndarray | None:
    if not relbase or (isinstance(relbase, float) and np.isnan(relbase)):
        return None
    path = (source_root / str(relbase)).with_suffix(".json")
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return speech_mask(payload.get("metadata:vad") or [], frames, fps)


def _thumbnail_frames(spans: Sequence[tuple[int, int]], count: int) -> list[int]:
    """Frame indices spread evenly over the *union* of the accepted spans.

    Sampling the accepted spans rather than the whole recording is deliberate:
    those are the frames the manifest would contain, so those are the frames the
    reviewer has to have seen. Where the participant is during the parts that
    were *not* selected is visible on the timeline but does not consume a
    thumbnail.
    """

    lengths = [stop - start for start, stop in spans]
    total = sum(lengths)
    if total <= 0 or count <= 0:
        return []
    offsets = [(index + 0.5) / count * total for index in range(count)]
    frames: list[int] = []
    for offset in offsets:
        running = 0
        for (start, stop), length in zip(spans, lengths):
            if offset < running + length:
                frames.append(int(start + (offset - running)))
                break
            running += length
        else:
            frames.append(int(spans[-1][1] - 1))
    return frames


def render_card(
    item: Mapping[str, Any],
    clips: pd.DataFrame,
    config: RunConfig,
    *,
    layout: CardLayout | None = None,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Render one review item's card. Idempotent by fingerprint."""

    layout = layout or CardLayout()
    output_dir = config.media_root
    output_dir.mkdir(parents=True, exist_ok=True)
    os.chmod(config.private_root, 0o700)
    stem = str(item["review_item_id"])
    card_path = output_dir / f"{stem}.card.png"
    sidecar_path = output_dir / f"{stem}.json"

    spans = [(int(row.start_frame), int(row.end_frame)) for row in clips.itertuples()]
    fingerprint = _digest(
        {
            "file": item["source_relbase"],
            "spans": spans,
            "layout": asdict(layout),
            # The gesture parameters draw the timeline's episode shading and the
            # arm-speed trace. Leaving them out let a re-tune reuse a card whose
            # timeline no longer matched the candidates it was selected by.
            "gesture": asdict(config.scan.gesture),
            "version": CARD_VERSION,
        }
    )
    if not overwrite and card_path.exists() and sidecar_path.exists():
        try:
            existing = json.loads(sidecar_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            existing = {}
        if existing.get("card_fingerprint") == fingerprint:
            return {**existing, "status": "reused"}

    base = config.source_root / str(item["source_relbase"])
    payload, vad = read_bundle(base)
    fps = float(item["nominal_fps"]) if item.get("nominal_fps") else 30.0
    tracks = build_tracks(
        payload, vad, fps=fps, model_root=str(config.model_root),
        params=config.scan.gesture, keep_joints=True,
    )

    keypoints = np.asarray(payload["boxes_and_keypoints:keypoints"], dtype=np.float64)
    video_path = base.with_suffix(".mp4")
    raster_w, raster_h = int(item.get("raster_width") or 0), int(item.get("raster_height") or 0)
    if raster_w <= 0 or raster_h <= 0:
        raster_w, raster_h = _probe_raster(video_path)
    repair = RasterRepair(
        width=raster_w,
        height=raster_h,
        sample_aspect=probe_sample_aspect(video_path),
        quarter_turns=quarter_turns_from_roll(body_roll_deg(keypoints) if len(keypoints) else float("nan")),
        pillarbox=pillarbox_for(raster_w, raster_h),
    )
    display_w, display_h = repair.display_size
    repaired = repair_points(keypoints[..., :2], repair)
    repaired_kp = np.concatenate([repaired, keypoints[..., 2:3]], axis=2)

    marks = _thumbnail_frames(spans, layout.thumbnails)
    crop = upper_body_crop_box(repaired_kp, marks, display_w, display_h)
    scale = (
        layout.crop_width / max(1, crop[2] - crop[0]),
        layout.crop_height / max(1, crop[3] - crop[1]),
    )
    repair_filter = ffmpeg_repair_filter(repair)

    thumbnails: list[np.ndarray] = []
    poses: list[np.ndarray] = []
    for index in marks:
        seconds = index / fps
        frame = _grab_crop(
            video_path, seconds, crop, (layout.crop_width, layout.crop_height), repair_filter
        )
        if frame is None:
            frame = np.full((layout.crop_height, layout.crop_width, 3), (24, 24, 24), np.uint8)
        else:
            draw_keypoint_overlay(frame, repaired_kp[index], (crop[0], crop[1]), scale)
        _stamp(frame, f"{seconds:6.1f}s")
        thumbnails.append(frame)
        poses.append(
            draw_pose_panel(
                tracks.joints[index],
                layout.crop_width,
                layout.pose_height,
                valid=bool(tracks.smplh_valid[index]),
            )
        )

    partner = _partner_speech(
        config.source_root, item.get("partner_source_relbase"), tracks.frames, fps
    )
    timeline = draw_timeline(tracks, spans, marks, partner, layout.width, layout.timeline_height)

    def show(key: str, fmt: str) -> str:
        """Format a measure, or say it is missing rather than printing a zero."""

        value = item.get(key)
        try:
            number = float(value)
        except (TypeError, ValueError):
            return "n/a"
        return "n/a" if number != number else format(number, fmt)

    accepted_s = sum(stop - start for start, stop in spans) / fps
    header = (
        f"{item['review_item_id']}  {item['file_id']}  "
        f"{item['vendor']} {item['label']} {item['interaction_type']}  "
        f"{tracks.frames / fps / 60:.1f} min  |  {len(spans)} clips, {accepted_s:.0f} s accepted",
        "gesture-in-speech {}  utterances covered {}  posture spread {} mm  "
        "wrist height {} mm  hands together {}  abduction {} deg  "
        "speech/silence {}x  sync r {}  smplh valid {}".format(
            show("gesture_frac_speech", ".0%"),
            show("speech_segments_covered", ".0%"),
            show("posture_spread_mm", ".0f"),
            show("wrist_height_p75_mm", "+.0f"),
            show("hands_together_frac", ".0%"),
            show("arm_abduction_p75_deg", ".0f"),
            show("gesture_speech_ratio", ".1f"),
            show("sync_r", ".2f"),
            show("smplh_valid_frac", ".0%"),
        ),
    )
    card = compose_card(thumbnails, poses, timeline, header, layout)

    temporary = card_path.with_suffix(".tmp.png")
    cv2.imwrite(str(temporary), card)
    os.chmod(temporary, 0o600)
    os.replace(temporary, card_path)

    record = {
        "review_item_id": stem,
        "file_id": item["file_id"],
        "card": card_path.name,
        "card_fingerprint": fingerprint,
        "card_version": CARD_VERSION,
        "spans": spans,
        "thumbnail_frames": marks,
        "accepted_seconds": round(accepted_s, 1),
        "duration_s": round(tracks.frames / fps, 1),
        "status": "rendered",
    }
    _write_private_json(sidecar_path, record)
    return record


def _probe_raster(path: Path) -> tuple[int, int]:
    command = [
        "ffprobe", "-v", "error", "-select_streams", "v:0",
        "-show_entries", "stream=width,height", "-of", "csv=p=0:s=x", str(path),
    ]
    try:
        out = subprocess.run(command, capture_output=True, text=True, timeout=60).stdout.strip()
        width, height = out.split("x")[:2]
        return int(width), int(height)
    except (OSError, subprocess.SubprocessError, ValueError):
        return 0, 0


def _stamp(frame: np.ndarray, label: str) -> None:
    height = frame.shape[0]
    cv2.rectangle(frame, (0, height - 16), (66, height), (0, 0, 0), -1)
    cv2.putText(frame, label, (2, height - 4), cv2.FONT_HERSHEY_DUPLEX, 0.36, (220, 220, 220), 1, cv2.LINE_AA)


def _write_private_json(path: Path, payload: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str), encoding="utf-8")
    os.chmod(temporary, 0o600)
    os.replace(temporary, path)
    os.chmod(path, 0o600)


def render_clip(
    item: Mapping[str, Any],
    clips: pd.DataFrame,
    config: RunConfig,
    *,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Render the item's representative 30-second clip, with audio.

    The *median*-scoring accepted span, not the best one. Showing a reviewer the
    single most animated thirty seconds of a recording would systematically
    flatter every file and defeat the point of asking them.
    """

    from .review_renderer import RenderSettings, load_joint_provider, render_record

    ordered = clips.sort_values("clip_score").reset_index(drop=True)
    # Lower median: `len // 2` on an even-length list returns the *upper* median,
    # so a two-clip file would always be shown its better thirty seconds.
    pick = ordered.iloc[(len(ordered) - 1) // 2]
    record = {
        "review_item_id": str(item["review_item_id"]),
        "clip_id": f"{item['review_item_id']}.clip",
        "file_id": str(item["file_id"]),
        "source_relbase": str(item["source_relbase"]),
        "start_frame": str(int(pick["start_frame"])),
        "render_policy": "render",
        "signals_json": "{}",
    }
    provider = _joint_provider(str(config.model_root))
    settings = RenderSettings(duration_s=config.clip_seconds)
    result = render_record(
        record, config.source_root, config.media_root, settings, provider, overwrite=overwrite
    )
    return {
        "review_item_id": record["review_item_id"],
        "clip": result.get("media_file"),
        "clip_start_frame": int(pick["start_frame"]),
        "clip_status": result.get("status"),
    }


_PROVIDER_CACHE: dict[str, Any] = {}


def _joint_provider(model_root: str) -> Any:
    from .review_renderer import load_joint_provider

    if model_root not in _PROVIDER_CACHE:
        _PROVIDER_CACHE[model_root] = load_joint_provider(
            {
                "factory": "seamless_curation.smplh_review_provider:create_provider",
                "settings": {"model_root": model_root, "flat_hand_mean": True},
            }
        )
    return _PROVIDER_CACHE[model_root]


def render_task(
    items: pd.DataFrame,
    config: RunConfig,
    *,
    overwrite: bool = False,
    card_only: bool = False,
    log: Callable[[str], None] | None = None,
) -> int:
    """Render every review item in ``items``; failures are recorded, not raised."""

    candidates = pd.read_parquet(config.candidates_path)
    by_file = {key: group for key, group in candidates.groupby("file_id")}
    population = pd.read_parquet(config.population_path).set_index("file_id")
    partners = _partner_index(population)

    done = 0
    for position, item in enumerate(items.to_dict("records"), start=1):
        clips = by_file.get(item["file_id"])
        if clips is None or clips.empty:
            continue
        enriched = dict(item)
        row = population.loc[item["file_id"]] if item["file_id"] in population.index else None
        if row is not None:
            enriched["nominal_fps"] = float(row["nominal_fps"])
            raster = str(row["raster"]).split("x")
            if len(raster) == 2 and raster[0].isdigit():
                enriched["raster_width"], enriched["raster_height"] = int(raster[0]), int(raster[1])
        enriched["partner_source_relbase"] = partners.get(item["file_id"])
        sidecar = config.media_root / f"{item['review_item_id']}.json"
        record: dict[str, Any] = {}
        try:
            record = render_card(enriched, clips, config, overwrite=overwrite)
            if not card_only:
                record.update(render_clip(enriched, clips, config, overwrite=overwrite))
                _write_private_json(sidecar, record)
            done += 1
        except Exception as error:  # noqa: BLE001 - one bad bundle must not kill a shard
            # Merge, never replace. The card may already have been written, and
            # overwriting its sidecar would destroy the fingerprint that makes
            # the render idempotent — so every later run would re-render it.
            _write_private_json(
                sidecar,
                {
                    "review_item_id": item["review_item_id"],
                    "file_id": item["file_id"],
                    **record,
                    "status": f"error:{type(error).__name__}",
                    "detail": str(error)[:400],
                },
            )
        if log and position % 20 == 0:
            log(f"{position}/{len(items)} items")
    return done


def _partner_index(population: pd.DataFrame) -> dict[str, str]:
    """Map each file to the other participant's file in the same interaction.

    Keyed on ``(vendor, session, interaction)``. ``interaction_id`` alone repeats
    across sessions — grouping without the session produced groups of four, six,
    and in one case 1,511 rows, every one of which the "exactly two" test then
    discarded, so 1,306 of 1,309 cards silently drew an empty partner-speech
    track under a legend that still said "partner speech".
    """

    columns = ["file_id", "vendor", "session_id", "interaction_id", "source_relbase"]
    frame = population.reset_index()[columns]
    partners: dict[str, str] = {}
    for _, group in frame.groupby(["vendor", "session_id", "interaction_id"], sort=False):
        if len(group) != 2:
            continue
        first, second = group.itertuples()
        partners[first.file_id] = second.source_relbase
        partners[second.file_id] = first.source_relbase
    return partners

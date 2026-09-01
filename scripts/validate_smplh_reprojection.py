#!/usr/bin/env python3
"""Bounded M-2/M-4 camera and SMPL-H reprojection validation.

This program reads only the explicitly listed participant NPZ payloads and
MP4 headers.  It never writes under the source dataset or model root.  Outputs
are aggregate/numeric tables under an ignored, mode-0700 directory.
"""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import yaml

from smplh_fk import (
    COCO_BODY23_TO_SMPLH,
    COCO_LEFT_HAND_TO_SMPLH,
    COCO_RIGHT_HAND_TO_SMPLH,
    CameraHypothesis,
    create_neutral_smplh,
    forward_smplh_joints,
    project_hmr2_full_frame,
)


REQUIRED_NPZ_KEYS = (
    "smplh:global_orient",
    "smplh:body_pose",
    "smplh:left_hand_pose",
    "smplh:right_hand_pose",
    "smplh:translation",
    "smplh:is_valid",
    "boxes_and_keypoints:box",
    "boxes_and_keypoints:keypoints",
    "boxes_and_keypoints:is_valid_box",
)

GROUPS: tuple[tuple[str, np.ndarray, slice], ...] = (
    ("body23", COCO_BODY23_TO_SMPLH, slice(0, 23)),
    ("body17", COCO_BODY23_TO_SMPLH[:17], slice(0, 17)),
    ("left_hand", COCO_LEFT_HAND_TO_SMPLH, slice(91, 112)),
    ("right_hand", COCO_RIGHT_HAND_TO_SMPLH, slice(112, 133)),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--max-frames-per-file", type=int)
    parser.add_argument("--output-dir", type=Path)
    return parser.parse_args()


def load_manifest(path: Path, limit: int | None) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    required = {"file_id", "label", "split", "vendor", "source_relbase"}
    if not rows or not required.issubset(rows[0]):
        raise ValueError(f"manifest must contain {sorted(required)}")
    if limit is not None:
        if limit < 1:
            raise ValueError("--limit must be positive")
        rows = rows[:limit]
    return rows


def resolve_under(root: Path, relative_base: str, suffix: str) -> Path:
    root_resolved = root.resolve()
    candidate = (root / f"{relative_base}{suffix}").resolve()
    if candidate != root_resolved and root_resolved not in candidate.parents:
        raise ValueError(f"manifest path escapes source root: {relative_base}")
    return candidate


def probe_raster(video_path: Path) -> tuple[int, int, str]:
    command = [
        "ffprobe", "-v", "error", "-select_streams", "v:0",
        "-show_entries", "stream=width,height,avg_frame_rate",
        "-of", "json", str(video_path),
    ]
    result = subprocess.run(command, check=True, capture_output=True, text=True)
    streams = json.loads(result.stdout).get("streams", [])
    if len(streams) != 1:
        raise ValueError(f"expected one video stream in {video_path}, got {len(streams)}")
    stream = streams[0]
    return int(stream["width"]), int(stream["height"]), str(stream["avg_frame_rate"])


def stable_file_seed(seed: int, file_id: str) -> int:
    digest = hashlib.sha256(f"{seed}:{file_id}".encode()).digest()
    return int.from_bytes(digest[:8], byteorder="big", signed=False)


def finite_rows(array: np.ndarray) -> np.ndarray:
    values = np.asarray(array)
    return np.isfinite(values.reshape(len(values), -1)).all(axis=1)


def safe_corr(left: np.ndarray, right: np.ndarray) -> float | None:
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    valid = np.isfinite(left) & np.isfinite(right)
    left, right = left[valid], right[valid]
    if len(left) < 3 or np.ptp(left) == 0 or np.ptp(right) == 0:
        return None
    return float(np.corrcoef(left, right)[0, 1])


def quantiles(values: np.ndarray) -> dict[str, float | int | None]:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return {"n": 0, "q00": None, "q05": None, "q50": None,
                "q95": None, "q100": None}
    result: dict[str, float | int | None] = {"n": int(len(values))}
    for name, probability in (
        ("q00", 0.0), ("q05", 0.05), ("q50", 0.5),
        ("q95", 0.95), ("q100", 1.0),
    ):
        result[name] = float(np.quantile(values, probability))
    return result


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.",
        suffix=".tmp", delete=False,
    ) as handle:
        temporary = Path(handle.name)
        json.dump(json_safe(dict(payload)), handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.chmod(temporary, 0o600)
    os.replace(temporary, path)


def atomic_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    with tempfile.NamedTemporaryFile(
        mode="w", newline="", encoding="utf-8", dir=path.parent,
        prefix=f".{path.name}.", suffix=".tmp", delete=False,
    ) as handle:
        temporary = Path(handle.name)
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(json_safe(list(rows)))
    os.chmod(temporary, 0o600)
    os.replace(temporary, path)


def git_sha(worktree: Path) -> str:
    command = [
        "git", f"--git-dir={worktree / '.git-session'}", f"--work-tree={worktree}",
        "rev-parse", "HEAD",
    ]
    try:
        return subprocess.run(
            command, check=True, capture_output=True, text=True
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unavailable"


def flatten_quantiles(prefix: str, values: np.ndarray) -> dict[str, Any]:
    return {f"{prefix}_{key}": value for key, value in quantiles(values).items()}


def summarize_camera_file(
    row: Mapping[str, str],
    arrays: Mapping[str, np.ndarray],
    width: int,
    height: int,
    camera: CameraHypothesis,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    translation = arrays["smplh:translation"]
    boxes = arrays["boxes_and_keypoints:box"]
    global_orient = arrays["smplh:global_orient"]
    valid = (
        arrays["smplh:is_valid"].astype(bool)
        & arrays["boxes_and_keypoints:is_valid_box"].astype(bool)
        & finite_rows(translation)
        & finite_rows(boxes)
        & (translation[:, 2] > 0)
    )
    frame_indices = np.flatnonzero(valid)
    selected_t = translation[valid].astype(np.float64)
    selected_box = boxes[valid].astype(np.float64)
    box_height_norm = (selected_box[:, 3] - selected_box[:, 1]) / height
    box_width_norm = (selected_box[:, 2] - selected_box[:, 0]) / width
    box_center_x_norm = (selected_box[:, 0] + selected_box[:, 2]) / (2 * width)
    box_center_y_norm = (selected_box[:, 1] + selected_box[:, 3]) / (2 * height)
    tz_times_box_height_norm = selected_t[:, 2] * box_height_norm
    implied_scale = camera.weak_perspective_depth_numerator / selected_t[:, 2]

    product_median = float(np.median(tz_times_box_height_norm))
    product_mad = float(np.median(np.abs(tz_times_box_height_norm - product_median)))
    product_robust_cv = (
        product_mad / abs(product_median) if product_median != 0 else None
    )

    orient_valid = arrays["smplh:is_valid"].astype(bool) & finite_rows(global_orient)
    orient = global_orient[orient_valid].astype(np.float64)
    orient0_distance_to_pi = np.abs(np.abs(orient[:, 0]) - np.pi)

    result: dict[str, Any] = {
        "file_id": row["file_id"], "vendor": row["vendor"],
        "label": row["label"], "split": row["split"],
        "width": width, "height": height,
        "n_total_frames": int(len(translation)),
        "n_camera_valid_frames": int(valid.sum()),
        **flatten_quantiles("tz", selected_t[:, 2]),
        **flatten_quantiles("box_height_over_frame_height", box_height_norm),
        **flatten_quantiles("tz_times_box_height_over_frame_height", tz_times_box_height_norm),
        **flatten_quantiles("implied_s", implied_scale),
        "tz_times_box_height_robust_cv": product_robust_cv,
        "corr_implied_s_vs_box_height_norm": safe_corr(implied_scale, box_height_norm),
        "corr_tx_vs_box_center_x_norm": safe_corr(selected_t[:, 0], box_center_x_norm),
        "corr_ty_vs_box_center_y_norm": safe_corr(selected_t[:, 1], box_center_y_norm),
        "global_orient_0_median": float(np.median(orient[:, 0])),
        "global_orient_abs0_distance_to_pi_median": float(np.median(orient0_distance_to_pi)),
        "global_orient_other_components_abs_median": float(np.median(np.abs(orient[:, 1:]))),
    }
    pooled = {
        "tz": selected_t[:, 2], "box_height_norm": box_height_norm,
        "box_width_norm": box_width_norm,
        "box_center_x_norm": box_center_x_norm,
        "box_center_y_norm": box_center_y_norm,
        "translation_x": selected_t[:, 0], "translation_y": selected_t[:, 1],
        "tz_box_height": tz_times_box_height_norm, "implied_s": implied_scale,
        "global_orient": orient,
        "frame_indices": frame_indices,
    }
    return result, pooled


def valid_fk_frame_mask(arrays: Mapping[str, np.ndarray]) -> np.ndarray:
    valid = (
        arrays["smplh:is_valid"].astype(bool)
        & arrays["boxes_and_keypoints:is_valid_box"].astype(bool)
    )
    for key in REQUIRED_NPZ_KEYS:
        valid &= finite_rows(arrays[key])
    valid &= arrays["smplh:translation"][:, 2] > 0
    valid &= np.abs(arrays["smplh:translation"]).max(axis=1) < 1000
    boxes = arrays["boxes_and_keypoints:box"]
    valid &= boxes[:, 2] > boxes[:, 0]
    valid &= boxes[:, 3] > boxes[:, 1]
    return valid


def sample_frames(
    arrays: Mapping[str, np.ndarray], file_id: str, seed: int, maximum: int
) -> np.ndarray:
    candidates = np.flatnonzero(valid_fk_frame_mask(arrays))
    if len(candidates) == 0:
        raise ValueError(f"no valid FK/reprojection frames in {file_id}")
    if len(candidates) <= maximum:
        return candidates
    generator = np.random.default_rng(stable_file_seed(seed, file_id))
    return np.sort(generator.choice(candidates, size=maximum, replace=False))


def summarize_errors(error_rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    flat_values = sorted({bool(row["flat_hand_mean"]) for row in error_rows})
    for flat in flat_values:
        flat_rows = [row for row in error_rows if bool(row["flat_hand_mean"]) == flat]
        setting: dict[str, Any] = {}
        for group in ("body23", "body17", "left_hand", "right_hand", "all_hands"):
            if group == "all_hands":
                rows = [row for row in flat_rows if row["group"] in ("left_hand", "right_hand")]
            else:
                rows = [row for row in flat_rows if row["group"] == group]
            setting[group] = {
                "error_px": quantiles(np.asarray([row["error_px"] for row in rows])),
                "error_box_height_fraction": quantiles(
                    np.asarray([row["error_box_height_fraction"] for row in rows])
                ),
            }
        result[str(flat).lower()] = setting
    false_key, true_key = "false", "true"
    false_median = result[false_key]["all_hands"]["error_px"]["q50"]
    true_median = result[true_key]["all_hands"]["error_px"]["q50"]
    false_p95 = result[false_key]["all_hands"]["error_px"]["q95"]
    true_p95 = result[true_key]["all_hands"]["error_px"]["q95"]
    selected = True if (true_median, true_p95) < (false_median, false_p95) else False
    result["selected_flat_hand_mean"] = selected
    result["selection_rule"] = "lower all-hands median pixel error, then p95 as tie-break"
    return result


def alternative_body_camera_errors(
    joints_camera: np.ndarray,
    released: np.ndarray,
    boxes: np.ndarray,
    widths: np.ndarray,
    heights: np.ndarray,
    camera: CameraHypothesis,
) -> dict[str, Any]:
    model_body = joints_camera[:, COCO_BODY23_TO_SMPLH]
    observed_body = released[:, :23, :2]
    candidates: dict[str, np.ndarray] = {}

    proposed = np.empty_like(observed_body, dtype=np.float64)
    unscaled = np.empty_like(observed_body, dtype=np.float64)
    box_crop = np.empty_like(observed_body, dtype=np.float64)
    for index, (width, height) in enumerate(zip(widths, heights)):
        proposed[index] = project_hmr2_full_frame(
            model_body[index], int(width), int(height), camera
        )
        depth = model_body[index, :, 2]
        unscaled[index, :, 0] = (
            camera.base_focal_length * model_body[index, :, 0] / depth + width / 2
        )
        unscaled[index, :, 1] = (
            camera.base_focal_length * model_body[index, :, 1] / depth + height / 2
        )
        box = boxes[index]
        center = np.asarray([(box[0] + box[2]) / 2, (box[1] + box[3]) / 2])
        box_size = max(box[2] - box[0], box[3] - box[1])
        crop_xy = np.stack(
            (
                camera.base_focal_length * model_body[index, :, 0] / depth,
                camera.base_focal_length * model_body[index, :, 1] / depth,
            ), axis=-1,
        )
        box_crop[index] = crop_xy * box_size / camera.model_input_size + center

    candidates["proposed_scaled_focal_raster_center"] = proposed
    candidates["unscaled_focal_raster_center"] = unscaled
    candidates["box_height_crop_transform"] = box_crop
    box_heights = boxes[:, 3] - boxes[:, 1]
    result: dict[str, Any] = {}
    for name, predicted in candidates.items():
        error = np.linalg.norm(predicted - observed_body, axis=-1)
        result[name] = {
            "error_px": quantiles(error),
            "error_box_height_fraction": quantiles(error / box_heights[:, None]),
        }
    return result


def main() -> int:
    args = parse_args()
    config_path = args.config.resolve()
    config_bytes = config_path.read_bytes()
    config = yaml.safe_load(config_bytes)
    worktree = Path(__file__).resolve().parents[1]
    source_root = (worktree / config["source_root"]).resolve()
    model_root = (worktree / config["model_root"]).resolve()
    manifest_path = (worktree / config["manifest"]).resolve()
    output_dir = (
        args.output_dir.resolve()
        if args.output_dir is not None
        else (worktree / config["output_dir"]).resolve()
    )
    if source_root == output_dir or source_root in output_dir.parents:
        raise ValueError("output directory must not be inside source dataset")
    if model_root == output_dir or model_root in output_dir.parents:
        raise ValueError("output directory must not be inside model root")
    output_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
    os.chmod(output_dir, 0o700)

    model_config = config["model"]
    expected = {
        "model_type": "smplh", "gender": "neutral", "ext": "npz",
        "use_pca": False, "num_betas": 16,
    }
    for key, value in expected.items():
        if model_config.get(key) != value:
            raise ValueError(f"M-4 requires model.{key}={value!r}")
    if model_config.get("flat_hand_mean_candidates") != [False, True]:
        raise ValueError("both flat_hand_mean candidates [false, true] are required")

    camera_config = config["camera_hypothesis"]
    camera = CameraHypothesis(
        base_focal_length=float(camera_config["base_focal_length"]),
        model_input_size=float(camera_config["model_input_size"]),
    )
    maximum = (
        args.max_frames_per_file
        if args.max_frames_per_file is not None
        else int(config["max_frames_per_file"])
    )
    if maximum < 1:
        raise ValueError("max frames per file must be positive")
    batch_size = int(config["batch_size"])
    device = str(config["device"])
    seed = int(config["sample_seed"])
    manifest = load_manifest(manifest_path, args.limit)

    camera_rows: list[dict[str, Any]] = []
    pooled_camera: dict[str, list[np.ndarray]] = {}
    frame_metadata: list[dict[str, Any]] = []
    selected_arrays: dict[str, list[np.ndarray]] = {
        key: [] for key in REQUIRED_NPZ_KEYS
    }

    for row in manifest:
        npz_path = resolve_under(source_root, row["source_relbase"], ".npz")
        video_path = resolve_under(source_root, row["source_relbase"], ".mp4")
        width, height, avg_frame_rate = probe_raster(video_path)
        with np.load(npz_path, allow_pickle=False) as archive:
            missing = [key for key in REQUIRED_NPZ_KEYS if key not in archive.files]
            if missing:
                raise KeyError(f"{row['file_id']} missing {missing}")
            arrays = {key: archive[key] for key in REQUIRED_NPZ_KEYS}
        lengths = {len(array) for array in arrays.values()}
        if len(lengths) != 1:
            raise ValueError(f"array length mismatch in {row['file_id']}: {lengths}")

        camera_row, pooled = summarize_camera_file(
            row, arrays, width, height, camera
        )
        camera_row["avg_frame_rate"] = avg_frame_rate
        camera_rows.append(camera_row)
        for key, values in pooled.items():
            pooled_camera.setdefault(key, []).append(values)

        indices = sample_frames(arrays, row["file_id"], seed, maximum)
        for key in REQUIRED_NPZ_KEYS:
            selected_arrays[key].append(np.asarray(arrays[key][indices]).copy())
        for frame_index in indices:
            frame_metadata.append({
                "file_id": row["file_id"], "vendor": row["vendor"],
                "label": row["label"], "split": row["split"],
                "frame_index": int(frame_index), "width": width,
                "height": height, "avg_frame_rate": avg_frame_rate,
            })

    concatenated = {
        key: np.concatenate(chunks, axis=0) for key, chunks in selected_arrays.items()
    }
    widths = np.asarray([row["width"] for row in frame_metadata], dtype=np.int64)
    heights = np.asarray([row["height"] for row in frame_metadata], dtype=np.int64)
    boxes = concatenated["boxes_and_keypoints:box"].astype(np.float64)
    keypoints = concatenated["boxes_and_keypoints:keypoints"].astype(np.float64)
    box_heights = boxes[:, 3] - boxes[:, 1]

    error_rows: list[dict[str, Any]] = []
    fk_frame_rows: list[dict[str, Any]] = []
    alternative_camera: dict[str, Any] | None = None
    flat_outputs: dict[bool, np.ndarray] = {}
    for flat_hand_mean in model_config["flat_hand_mean_candidates"]:
        model = create_neutral_smplh(
            model_root, flat_hand_mean=bool(flat_hand_mean),
            batch_size=batch_size, device=device,
        )
        camera_chunks: list[np.ndarray] = []
        root_chunks: list[np.ndarray] = []
        for start in range(0, len(frame_metadata), batch_size):
            stop = min(start + batch_size, len(frame_metadata))
            payload = {key: value[start:stop] for key, value in concatenated.items()}
            joints_camera, joints_root = forward_smplh_joints(
                model, payload, device=device, include_translation=True
            )
            camera_chunks.append(joints_camera)
            root_chunks.append(joints_root)
        joints_camera = np.concatenate(camera_chunks, axis=0)
        joints_root = np.concatenate(root_chunks, axis=0)
        flat_outputs[bool(flat_hand_mean)] = joints_camera

        projected = np.empty((len(frame_metadata), 73, 2), dtype=np.float64)
        for frame, (width, height) in enumerate(zip(widths, heights)):
            projected[frame] = project_hmr2_full_frame(
                joints_camera[frame], int(width), int(height), camera
            )

        for frame, metadata in enumerate(frame_metadata):
            base = {
                **metadata,
                "flat_hand_mean": bool(flat_hand_mean),
                "pelvis_root_relative_abs_max_m": float(np.abs(joints_root[frame, 0]).max()),
                "camera_depth_min_m": float(joints_camera[frame, :, 2].min()),
                "camera_depth_max_m": float(joints_camera[frame, :, 2].max()),
            }
            fk_frame_rows.append(base)
            for group, model_indices, released_slice in GROUPS:
                observed = keypoints[frame, released_slice, :2]
                predicted = projected[frame, model_indices]
                errors = np.linalg.norm(predicted - observed, axis=-1)
                released_start = released_slice.start or 0
                for offset, (model_index, error) in enumerate(zip(model_indices, errors)):
                    error_rows.append({
                        **metadata, "flat_hand_mean": bool(flat_hand_mean),
                        "group": group, "released_keypoint_index": released_start + offset,
                        "smplh_joint_index": int(model_index),
                        "error_px": float(error),
                        "error_box_height_fraction": float(error / box_heights[frame]),
                    })
        if not bool(flat_hand_mean):
            alternative_camera = alternative_body_camera_errors(
                joints_camera, keypoints, boxes, widths, heights, camera
            )
        del model
        gc.collect()

    errors_summary = summarize_errors(error_rows)
    selected_flat = bool(errors_summary["selected_flat_hand_mean"])
    if alternative_camera is None:
        raise AssertionError("flat_hand_mean=False candidate did not run")

    pooled = {key: np.concatenate(chunks) for key, chunks in pooled_camera.items()}
    file_mean_s = np.asarray([
        row["implied_s_q50"] for row in camera_rows
    ], dtype=np.float64)
    file_mean_box = np.asarray([
        row["box_height_over_frame_height_q50"] for row in camera_rows
    ], dtype=np.float64)
    orient = pooled["global_orient"]
    camera_summary = {
        "n_files": len(camera_rows),
        "n_valid_frames": int(sum(row["n_camera_valid_frames"] for row in camera_rows)),
        "tz": quantiles(pooled["tz"]),
        "box_height_over_frame_height": quantiles(pooled["box_height_norm"]),
        "tz_times_box_height_over_frame_height": quantiles(pooled["tz_box_height"]),
        "implied_s_39_0625_over_tz": quantiles(pooled["implied_s"]),
        "correlation_implied_s_vs_box_height_norm_pooled": safe_corr(
            pooled["implied_s"], pooled["box_height_norm"]
        ),
        "correlation_implied_s_vs_box_height_norm_file_medians": safe_corr(
            file_mean_s, file_mean_box
        ),
        "correlation_tx_vs_box_center_x_norm_pooled": safe_corr(
            pooled["translation_x"], pooled["box_center_x_norm"]
        ),
        "correlation_ty_vs_box_center_y_norm_pooled": safe_corr(
            pooled["translation_y"], pooled["box_center_y_norm"]
        ),
        "global_orient_component_0": quantiles(orient[:, 0]),
        "global_orient_abs_component_0_distance_to_pi": quantiles(
            np.abs(np.abs(orient[:, 0]) - np.pi)
        ),
        "global_orient_components_1_2_abs": quantiles(np.abs(orient[:, 1:])),
        "per_file_product_robust_cv": quantiles(np.asarray([
            row["tz_times_box_height_robust_cv"] for row in camera_rows
        ], dtype=np.float64)),
    }

    model_asset = model_root / "smplh" / "SMPLH_NEUTRAL.npz"
    model_stat = model_asset.stat()
    config_hash = hashlib.sha256(config_bytes).hexdigest()
    summary = {
        "scope": {
            "manifest": str(manifest_path), "file_ids": [row["file_id"] for row in manifest],
            "sample_seed": seed, "max_frames_per_file": maximum,
            "n_sampled_frames": len(frame_metadata),
        },
        "provenance": {
            "git_sha_at_run": git_sha(worktree), "config_sha256": config_hash,
            "python": sys.version, "numpy": np.__version__,
            "slurm_job_id": os.environ.get("SLURM_JOB_ID", "unavailable"),
            "slurm_step_id": os.environ.get("SLURM_STEP_ID", "unavailable"),
        },
        "model": {
            "create_settings": {
                "model_path": str(model_root), "model_type": "smplh",
                "gender": "neutral", "ext": "npz", "use_pca": False,
                "num_betas": 16, "batch_size": batch_size,
            },
            "beta": "all-zero project convention (not observed in payload)",
            "flat_hand_mean_candidates": [False, True],
            "selected_flat_hand_mean": selected_flat,
            "asset_resolved_path": str(model_asset.resolve()),
            "asset_size_bytes": model_stat.st_size,
            "asset_mtime_ns": model_stat.st_mtime_ns,
            "asset_copied_or_modified": False,
        },
        "camera_hypothesis": {
            "base_focal_length": camera.base_focal_length,
            "model_input_size": camera.model_input_size,
            "weak_perspective_depth_numerator": camera.weak_perspective_depth_numerator,
            "focal_pixels_formula": "5000 / 256 * max(native_width, native_height)",
            "principal_point": "native raster centre",
            "payload_intrinsics_present": False,
        },
        "evaluation_policy": {
            "keypoint_score_threshold": None,
            "body23_released_indices": "0:23",
            "body17_released_indices": "0:17",
            "left_hand_released_indices": "91:112",
            "right_hand_released_indices": "112:133",
            "frame_requirements": config["evaluation"]["frame_requirements"],
        },
        "m2_camera_characterization": camera_summary,
        "m4_reprojection": errors_summary,
        "camera_candidate_comparison_body23": alternative_camera,
        "root_relative_fk": {
            "pelvis_abs_max_m": float(max(
                row["pelvis_root_relative_abs_max_m"] for row in fk_frame_rows
            )),
            "units": "metres from SMPL-H library; multiply by 1000 for millimetres",
            "translation_zeroing_equivalence": "subtracting output pelvis removes supplied translation",
        },
        "interpretation_guardrail": (
            "This run validates one explicit camera/model convention on a bounded dev sample; "
            "it does not establish physical camera intrinsics stored by the dataset."
        ),
    }

    atomic_csv(output_dir / "camera_per_file.csv", camera_rows)
    atomic_csv(output_dir / "sampled_frames.csv", frame_metadata)
    atomic_csv(output_dir / "reprojection_errors.csv", error_rows)
    atomic_csv(output_dir / "fk_frame_summary.csv", fk_frame_rows)
    atomic_json(output_dir / "summary.json", summary)
    print(json.dumps(json_safe(summary), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

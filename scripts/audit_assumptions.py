#!/usr/bin/env python3
"""Reproduce the bounded Session-1 D4 assumption audit.

The source dataset is opened read-only. The script follows paths exclusively
from the three bounded manifests, refuses a union larger than 200 files, and
writes one atomic JSON result below ``outputs/recon``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from collections import Counter
from datetime import datetime, timezone
from fractions import Fraction
from pathlib import Path, PurePosixPath
from typing import Any, Iterable

# Keep this bounded audit single-threaded even when numerical libraries were
# configured with aggressive defaults by the surrounding cluster environment.
for _name in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ[_name] = "1"

import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation


REPO_ROOT = Path(__file__).resolve().parents[1]
ALLOWED_OUTPUT_ROOT = (REPO_ROOT / "outputs" / "recon").resolve()
POSE_KEYS = (
    "smplh:translation",
    "smplh:global_orient",
    "smplh:body_pose",
    "smplh:left_hand_pose",
    "smplh:right_hand_pose",
)
ROTATION_KEYS = POSE_KEYS[1:]
ALLOWED_LABELS = {"improvised", "naturalistic"}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def _checked_output(path: Path) -> Path:
    candidate = path if path.is_absolute() else REPO_ROOT / path
    candidate = candidate.resolve(strict=False)
    if not _is_relative_to(candidate, ALLOWED_OUTPUT_ROOT):
        raise ValueError(
            f"output must be below {ALLOWED_OUTPUT_ROOT}, got {candidate}"
        )
    if candidate == ALLOWED_OUTPUT_ROOT:
        raise ValueError("output must be a file, not the output directory")
    return candidate


def _checked_source_npz(source_root: Path, relative_base: str) -> Path:
    relative = PurePosixPath(relative_base)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"unsafe source_relbase: {relative_base!r}")
    if len(relative.parts) < 5 or relative.parts[1] != "dev":
        raise ValueError(f"path is outside a bounded dev split: {relative_base!r}")
    candidate = (source_root / Path(*relative.parts)).with_suffix(".npz")
    candidate = candidate.resolve(strict=True)
    if not _is_relative_to(candidate, source_root):
        raise ValueError(f"source path escaped source root: {candidate}")
    return candidate


def _read_manifest(path: Path, membership: str) -> list[dict[str, str]]:
    frame = pd.read_csv(path, dtype=str)
    required = {"file_id", "label", "split", "vendor", "source_relbase"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{path} lacks required columns: {sorted(missing)}")
    records = frame.to_dict(orient="records")
    for record in records:
        if record["split"] != "dev" or record["label"] not in ALLOWED_LABELS:
            raise ValueError(f"non-dev or unexpected label in {path}: {record}")
        record["audit_membership"] = membership
    return records


def _load_populations(args: argparse.Namespace) -> tuple[
    list[dict[str, str]], list[dict[str, str]], pd.DataFrame
]:
    manifest_paths = (
        (args.main_manifest, "main"),
        (args.dyad_manifest, "dyad"),
        (args.movement_manifest, "movement"),
    )
    populations = {
        name: _read_manifest(path, name) for path, name in manifest_paths
    }
    main = populations["main"]
    if len(main) != 120:
        raise ValueError(f"D4 definitions require the 120-file main manifest, got {len(main)}")

    union: dict[str, dict[str, str]] = {}
    memberships: dict[str, set[str]] = {}
    for _, name in manifest_paths:
        for record in populations[name]:
            existing = union.get(record["file_id"])
            if existing is not None and existing["source_relbase"] != record["source_relbase"]:
                raise ValueError(f"conflicting paths for {record['file_id']}")
            union.setdefault(record["file_id"], record.copy())
            memberships.setdefault(record["file_id"], set()).add(name)
    if len(union) > args.max_files:
        raise ValueError(
            f"bounded union has {len(union)} files, exceeding --max-files={args.max_files}"
        )
    for file_id, record in union.items():
        record["audit_membership"] = "+".join(sorted(memberships[file_id]))

    union_rows = sorted(
        union.values(), key=lambda row: (row["label"], row["file_id"])
    )
    media = pd.read_parquet(args.media_probe)
    if media["file_id"].duplicated().any():
        raise ValueError("media probe has duplicate file_id rows")
    expected = {record["file_id"] for record in union_rows}
    observed = set(media["file_id"])
    if expected != observed:
        raise ValueError(
            "media-probe population differs from bounded union: "
            f"missing={sorted(expected-observed)}, extra={sorted(observed-expected)}"
        )
    return main, union_rows, media


def _quantiles(values: np.ndarray, points: Iterable[float]) -> dict[str, float]:
    points = tuple(points)
    measured = np.quantile(values, points)
    return {f"p{point * 100:g}": float(value) for point, value in zip(points, measured)}


def _distribution(
    values: np.ndarray, points: Iterable[float] = (0, 0.01, 0.5, 0.99, 1)
) -> dict[str, Any]:
    values = np.asarray(values)
    return {
        "count": int(values.size),
        "finite_fraction": float(np.isfinite(values).mean()),
        **_quantiles(values, points),
    }


def _pearson(matrix: np.ndarray, left: int, right: int) -> float:
    return float(np.corrcoef(matrix[:, left], matrix[:, right])[0, 1])


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, (np.integer, np.floating, np.bool_)):
        return _json_safe(value.item())
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _archive_inventory(
    union_rows: list[dict[str, str]], source_root: Path
) -> dict[str, Any]:
    keysets: Counter[tuple[str, ...]] = Counter()
    suspicious = {"beta_or_shape": [], "joint": [], "vertex": []}
    for record in union_rows:
        path = _checked_source_npz(source_root, record["source_relbase"])
        with np.load(path, allow_pickle=False) as archive:
            keys = tuple(sorted(archive.files))
        keysets[keys] += 1
        for key in keys:
            lowered = key.lower()
            if "beta" in lowered or "shape" in lowered:
                suspicious["beta_or_shape"].append([record["file_id"], key])
            if "joint" in lowered:
                suspicious["joint"].append([record["file_id"], key])
            if "vert" in lowered:
                suspicious["vertex"].append([record["file_id"], key])
    return {
        "files": len(union_rows),
        "unique_keysets": len(keysets),
        "keysets": [
            {"count": count, "keys": list(keys)}
            for keys, count in sorted(keysets.items(), key=lambda item: -item[1])
        ],
        "matched_keys": suspicious,
    }


def _payload_audit(
    main_rows: list[dict[str, str]], source_root: Path, media: pd.DataFrame
) -> dict[str, Any]:
    media_by_file = media.set_index("file_id")
    ordered = sorted(main_rows, key=lambda row: (row["label"], row["vendor"], row["file_id"]))

    pose_values: dict[str, list[np.ndarray]] = {key: [] for key in POSE_KEYS}
    pose_deltas: dict[str, list[np.ndarray]] = {key: [] for key in POSE_KEYS}
    rotation_norms: dict[str, list[np.ndarray]] = {key: [] for key in ROTATION_KEYS}
    wrap_raw: dict[str, list[np.ndarray]] = {key: [] for key in ROTATION_KEYS}
    wrap_geodesic: dict[str, list[np.ndarray]] = {key: [] for key in ROTATION_KEYS}
    translation_axes: list[np.ndarray] = []
    translation_file_ranges: list[np.ndarray] = []
    camera_frames: list[np.ndarray] = []
    camera_file_means: list[np.ndarray] = []
    camera_within_correlations: list[np.ndarray] = []
    layout_counts: Counter[tuple[str, tuple[int, ...], str]] = Counter()

    total_frames = 0
    valid_frames = 0
    consecutive_valid_pairs = 0
    empty_files: list[str] = []

    keypoint_total = 0
    off_frame = 0
    positive_channel_total = 0
    positive_channel_off_frame = 0
    confidence_total = 0
    confidence_finite = 0
    confidence_negative = 0
    confidence_zero = 0
    confidence_gt_one = 0
    confidence_min = np.inf
    confidence_max = -np.inf
    confidence_samples: list[np.ndarray] = []
    keypoint_per_file: list[dict[str, Any]] = []

    valid_box_frames = 0
    ordered_xyxy_frames = 0
    xyxy_in_frame = 0
    xywh_extent_in_frame = 0

    for record in ordered:
        file_id = record["file_id"]
        path = _checked_source_npz(source_root, record["source_relbase"])
        probe = media_by_file.loc[file_id]
        with np.load(path, allow_pickle=False) as archive:
            required = set(POSE_KEYS) | {
                "smplh:is_valid",
                "boxes_and_keypoints:box",
                "boxes_and_keypoints:is_valid_box",
                "boxes_and_keypoints:keypoints",
            }
            missing = required - set(archive.files)
            if missing:
                raise ValueError(f"{file_id} lacks required keys: {sorted(missing)}")

            valid = archive["smplh:is_valid"].astype(bool)
            box_valid = archive["boxes_and_keypoints:is_valid_box"].astype(bool)
            keypoints = archive["boxes_and_keypoints:keypoints"]
            boxes = archive["boxes_and_keypoints:box"]
            arrays = {key: archive[key] for key in POSE_KEYS}

            lengths = {len(valid), len(box_valid), len(keypoints), len(boxes)}
            lengths.update(len(array) for array in arrays.values())
            if len(lengths) != 1:
                raise ValueError(f"first-dimension mismatch in {file_id}: {sorted(lengths)}")

            frame_count = len(valid)
            total_frames += frame_count
            valid_frames += int(valid.sum())
            consecutive = valid[:-1] & valid[1:]
            consecutive_valid_pairs += int(consecutive.sum())

            for key, array in arrays.items():
                layout_counts[(key, tuple(array.shape[1:]), str(array.dtype))] += 1
                if valid.any():
                    valid_array = array[valid].astype(np.float64, copy=False)
                    pose_values[key].append(valid_array.reshape(-1).copy())
                    if key in ROTATION_KEYS:
                        rotation_norms[key].append(
                            np.linalg.norm(valid_array.reshape(-1, 3), axis=-1)
                        )
                if consecutive.any():
                    before = array[:-1][consecutive].reshape(-1, 3)
                    after = array[1:][consecutive].reshape(-1, 3)
                    delta_norm = np.linalg.norm(after - before, axis=-1)
                    pose_deltas[key].append(delta_norm)
                    if key in ROTATION_KEYS:
                        indices = np.flatnonzero(delta_norm > 3.0)
                        if len(indices):
                            geodesic = (
                                Rotation.from_rotvec(before[indices]).inv()
                                * Rotation.from_rotvec(after[indices])
                            ).magnitude()
                            wrap_raw[key].append(delta_norm[indices])
                            wrap_geodesic[key].append(geodesic)

            if valid.any():
                translation = arrays["smplh:translation"][valid]
                translation_axes.append(translation)
                translation_file_ranges.append(
                    translation.max(axis=0) - translation.min(axis=0)
                )

            if frame_count == 0:
                empty_files.append(file_id)
                keypoint_per_file.append(
                    {
                        "file_id": file_id,
                        "label": record["label"],
                        "vendor": record["vendor"],
                        "resolution": None,
                        "off_frame_fraction": None,
                    }
                )
                continue

            if not np.isfinite(probe["video_width"]) or not np.isfinite(probe["video_height"]):
                raise ValueError(f"nonempty {file_id} has no probed video dimensions")
            width = float(probe["video_width"])
            height = float(probe["video_height"])

            both_valid = valid & box_valid
            camera_indices = np.flatnonzero(both_valid)[::10]
            if len(camera_indices) >= 3:
                sampled_boxes = boxes[camera_indices]
                sampled_translation = arrays["smplh:translation"][camera_indices]
                normalized_box_height = (
                    sampled_boxes[:, 3] - sampled_boxes[:, 1]
                ) / height
                camera = np.column_stack(
                    [
                        sampled_translation[:, 0],
                        sampled_translation[:, 1],
                        sampled_translation[:, 2],
                        (sampled_boxes[:, 0] + sampled_boxes[:, 2]) / (2 * width),
                        (sampled_boxes[:, 1] + sampled_boxes[:, 3]) / (2 * height),
                        (sampled_boxes[:, 2] - sampled_boxes[:, 0]) / width,
                        normalized_box_height,
                        1.0 / normalized_box_height,
                    ]
                )
                camera_frames.append(camera)
                camera_file_means.append(camera.mean(axis=0))
                correlation = np.corrcoef(camera, rowvar=False)
                camera_within_correlations.append(
                    np.asarray(
                        [
                            correlation[0, 3],
                            correlation[1, 4],
                            correlation[2, 6],
                            correlation[2, 7],
                        ]
                    )
                )

            xy = keypoints[..., :2]
            confidence = keypoints[..., 2]
            off = (
                (xy[..., 0] < 0)
                | (xy[..., 0] >= width)
                | (xy[..., 1] < 0)
                | (xy[..., 1] >= height)
            )
            positive = confidence > 0
            keypoint_total += int(off.size)
            off_frame += int(off.sum())
            positive_channel_total += int(positive.sum())
            positive_channel_off_frame += int((off & positive).sum())

            confidence_total += int(confidence.size)
            confidence_finite += int(np.isfinite(confidence).sum())
            confidence_negative += int((confidence < 0).sum())
            confidence_zero += int((confidence == 0).sum())
            confidence_gt_one += int((confidence > 1).sum())
            confidence_min = min(confidence_min, float(np.nanmin(confidence)))
            confidence_max = max(confidence_max, float(np.nanmax(confidence)))
            confidence_samples.append(confidence.reshape(-1)[::101].copy())

            normalized = xy.astype(np.float64, copy=True)
            normalized[..., 0] /= width
            normalized[..., 1] /= height
            keypoint_per_file.append(
                {
                    "file_id": file_id,
                    "label": record["label"],
                    "vendor": record["vendor"],
                    "resolution": f"{int(width)}x{int(height)}",
                    "off_frame_fraction": float(off.mean()),
                    "positive_channel_off_frame_fraction": float(off[positive].mean()),
                    "xmin_over_width": float(normalized[..., 0].min()),
                    "xmax_over_width": float(normalized[..., 0].max()),
                    "ymin_over_height": float(normalized[..., 1].min()),
                    "ymax_over_height": float(normalized[..., 1].max()),
                }
            )

            valid_boxes = boxes[box_valid]
            valid_box_frames += len(valid_boxes)
            ordered_xyxy_frames += int(
                (
                    (valid_boxes[:, 2] > valid_boxes[:, 0])
                    & (valid_boxes[:, 3] > valid_boxes[:, 1])
                ).sum()
            )
            xyxy_in_frame += int(
                (
                    (valid_boxes[:, 0] >= 0)
                    & (valid_boxes[:, 1] >= 0)
                    & (valid_boxes[:, 2] <= width)
                    & (valid_boxes[:, 3] <= height)
                ).sum()
            )
            xywh_extent_in_frame += int(
                (
                    (valid_boxes[:, 0] >= 0)
                    & (valid_boxes[:, 1] >= 0)
                    & (valid_boxes[:, 0] + valid_boxes[:, 2] <= width)
                    & (valid_boxes[:, 1] + valid_boxes[:, 3] <= height)
                ).sum()
            )

    pose: dict[str, Any] = {
        "total_frames": total_frames,
        "smplh_valid_frames": valid_frames,
        "consecutive_valid_pairs": consecutive_valid_pairs,
        "empty_files": empty_files,
        "layouts": [
            {"key": key, "shape_suffix": list(shape), "dtype": dtype, "files": count}
            for (key, shape, dtype), count in sorted(layout_counts.items())
        ],
        "arrays": {},
    }
    for key in POSE_KEYS:
        values = np.concatenate(pose_values[key])
        deltas = np.concatenate(pose_deltas[key])
        entry: dict[str, Any] = {
            "component_distribution": _distribution(
                values, (0, 0.001, 0.01, 0.5, 0.99, 0.999, 1)
            ),
            "adjacent_vector_delta_norm": _distribution(
                deltas, (0, 0.5, 0.95, 0.99, 0.999, 1)
            ),
        }
        if key in ROTATION_KEYS:
            norms = np.concatenate(rotation_norms[key])
            raw = np.concatenate(wrap_raw[key]) if wrap_raw[key] else np.asarray([])
            geodesic = (
                np.concatenate(wrap_geodesic[key])
                if wrap_geodesic[key]
                else np.asarray([])
            )
            entry["rotation_vector_norm"] = {
                **_distribution(norms, (0, 0.5, 0.95, 0.99, 0.999, 1)),
                "fraction_gt_pi": float((norms > np.pi).mean()),
                "fraction_gt_2pi": float((norms > 2 * np.pi).mean()),
            }
            entry["raw_delta_gt_3_wrap_check"] = {
                "count": int(raw.size),
                "raw_distribution": _distribution(raw) if raw.size else None,
                "geodesic_distribution": _distribution(
                    geodesic, (0, 0.5, 0.99, 1)
                )
                if geodesic.size
                else None,
                "geodesic_lt_0_1_fraction": float((geodesic < 0.1).mean())
                if geodesic.size
                else None,
            }
        pose["arrays"][key] = entry

    translations = np.concatenate(translation_axes)
    ranges = np.stack(translation_file_ranges)
    pose["translation_axes"] = {
        str(axis): _distribution(translations[:, axis], (0, 0.001, 0.01, 0.5, 0.99, 0.999, 1))
        for axis in range(3)
    }
    pose["translation_within_file_ranges"] = {
        str(axis): _distribution(ranges[:, axis], (0, 0.5, 0.9, 1))
        for axis in range(3)
    }

    pooled = np.concatenate(camera_frames)
    file_means = np.stack(camera_file_means)
    within = np.stack(camera_within_correlations)
    camera_names = (
        "translation_0_vs_box_center_x_over_width",
        "translation_1_vs_box_center_y_over_height",
        "translation_2_vs_box_height_over_height",
        "translation_2_vs_inverse_box_height_over_height",
    )
    pairs = ((0, 3), (1, 4), (2, 6), (2, 7))
    camera_correlations = {
        name: {
            "pooled_frames": _pearson(pooled, left, right),
            "across_file_means": _pearson(file_means, left, right),
            "within_file_distribution": _distribution(
                within[:, index], (0, 0.1, 0.5, 0.9, 1)
            ),
        }
        for index, (name, (left, right)) in enumerate(zip(camera_names, pairs))
    }
    regressions: dict[str, Any] = {}
    for name, y, x in (
        (
            "translation_0_over_2_vs_box_center_x_over_width",
            pooled[:, 0] / pooled[:, 2],
            pooled[:, 3],
        ),
        (
            "translation_1_over_2_vs_box_center_y_over_height",
            pooled[:, 1] / pooled[:, 2],
            pooled[:, 4],
        ),
    ):
        slope, intercept = np.polyfit(x, y, 1)
        prediction = slope * x + intercept
        r_squared = 1.0 - float(((y - prediction) ** 2).sum()) / float(
            ((y - y.mean()) ** 2).sum()
        )
        regressions[name] = {
            "slope": float(slope),
            "intercept": float(intercept),
            "r_squared": r_squared,
            "zero_crossing": float(-intercept / slope),
        }

    keypoint_frame = pd.DataFrame(keypoint_per_file)
    nonempty_off = keypoint_frame["off_frame_fraction"].dropna().to_numpy()
    confidence_sample = np.concatenate(confidence_samples)
    keypoints_result = {
        "files": len(keypoint_per_file),
        "nonempty_files": int(keypoint_frame["off_frame_fraction"].notna().sum()),
        "total_frames": total_frames,
        "total_keypoints": keypoint_total,
        "aggregate_off_frame_fraction": off_frame / keypoint_total,
        "positive_channel_off_frame_fraction": (
            positive_channel_off_frame / positive_channel_total
        ),
        "per_file_off_frame_distribution": _distribution(
            nonempty_off, (0, 0.5, 0.75, 0.9, 0.95, 0.99, 1)
        ),
        "per_file_normalized_extrema": {
            name: _distribution(keypoint_frame[name].dropna().to_numpy(), (0, 0.01, 0.5, 0.99, 1))
            for name in (
                "xmin_over_width",
                "xmax_over_width",
                "ymin_over_height",
                "ymax_over_height",
            )
        },
        "third_channel": {
            "count": confidence_total,
            "finite_fraction": confidence_finite / confidence_total,
            "minimum": confidence_min,
            "maximum": confidence_max,
            "negative_fraction": confidence_negative / confidence_total,
            "zero_fraction": confidence_zero / confidence_total,
            "greater_than_one_fraction": confidence_gt_one / confidence_total,
            "deterministic_stride_101_distribution": _distribution(
                confidence_sample, (0, 0.001, 0.01, 0.5, 0.99, 0.999, 1)
            ),
        },
        "boxes": {
            "valid_frames": valid_box_frames,
            "ordered_xyxy_fraction": ordered_xyxy_frames / valid_box_frames,
            "xyxy_endpoints_in_native_frame_fraction": xyxy_in_frame / valid_box_frames,
            "xywh_extents_in_native_frame_fraction": xywh_extent_in_frame
            / valid_box_frames,
        },
        "per_file": keypoint_per_file,
    }
    return {
        "pose": pose,
        "camera_coupling": {
            "frame_stride": 10,
            "pooled_frames": len(pooled),
            "files": len(file_means),
            "correlations": camera_correlations,
            "projection_like_regressions": regressions,
        },
        "keypoints_and_boxes": keypoints_result,
    }


def _rate_fraction(value: Any) -> str | None:
    if pd.isna(value):
        return None
    return str(Fraction(float(value)).limit_denominator(100_000))


def _records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    return [_json_safe(record) for record in frame.to_dict(orient="records")]


def _media_audit(media: pd.DataFrame) -> dict[str, Any]:
    frame = media.copy()
    frame["resolution"] = frame.apply(
        lambda row: f"{int(row.video_width)}x{int(row.video_height)}"
        if pd.notna(row.video_width)
        else "unprobeable",
        axis=1,
    )
    frame["nominal_30_abs_mismatch_s"] = (
        frame["n_frames_smplh"] / 30.0 - frame["wav_duration_s"]
    ).abs()
    frame["actual_fps_abs_mismatch_s"] = (
        frame["video_nb_frames"] / frame["video_avg_fps"]
        - frame["wav_duration_s"]
    ).abs()
    frame["video_minus_npz_frames"] = (
        frame["video_nb_frames"] - frame["n_frames_smplh"]
    )

    rates = (
        frame.groupby(
            ["label", "vendor", "resolution", "video_r_fps"], dropna=False
        )
        .size()
        .rename("files")
        .reset_index()
    )
    rates["r_frame_rate_fraction"] = rates["video_r_fps"].map(_rate_fraction)

    average_rates = (
        frame.groupby("video_avg_fps", dropna=False)
        .size()
        .rename("files")
        .reset_index()
    )

    duration = (
        frame.groupby(["label", "vendor", "resolution"], dropna=False)
        .agg(
            files=("file_id", "size"),
            files_with_duration=("nominal_30_abs_mismatch_s", "count"),
            nominal_30_median_s=("nominal_30_abs_mismatch_s", "median"),
            nominal_30_max_s=("nominal_30_abs_mismatch_s", "max"),
            actual_fps_median_s=("actual_fps_abs_mismatch_s", "median"),
            actual_fps_max_s=("actual_fps_abs_mismatch_s", "max"),
        )
        .reset_index()
    )
    nonempty_nominal = frame["nominal_30_abs_mismatch_s"].dropna().to_numpy()
    nonempty_actual = frame["actual_fps_abs_mismatch_s"].dropna().to_numpy()

    delta_counts = frame["video_minus_npz_frames"].value_counts(dropna=False).sort_index()
    placeholders = frame[frame["video_avg_fps"].isna()][
        [
            "file_id",
            "label",
            "vendor",
            "n_frames_smplh",
            "n_frames_box",
            "size_json_bytes",
            "size_mp4_bytes",
            "size_npz_bytes",
            "size_wav_bytes",
        ]
    ]
    return {
        "files": len(frame),
        "r_frame_rate_by_label_vendor_resolution": _records(rates),
        "average_frame_rate_counts": _records(average_rates),
        "duration_by_label_vendor_resolution": _records(duration),
        "duration_overall": {
            "nominal_30_abs_mismatch_s": _distribution(
                nonempty_nominal, (0, 0.5, 0.9, 0.95, 0.99, 1)
            ),
            "actual_fps_abs_mismatch_s": _distribution(
                nonempty_actual, (0, 0.5, 0.9, 0.95, 0.99, 1)
            ),
        },
        "video_minus_npz_frame_count_counts": {
            "unprobeable" if pd.isna(delta) else f"{int(delta):+d}": int(count)
            for delta, count in delta_counts.items()
        },
        "unprobeable_placeholders": _records(placeholders),
    }


def _dyad_audit(dyad_manifest: Path, media: pd.DataFrame) -> dict[str, Any]:
    dyad = pd.read_csv(dyad_manifest, dtype={"member_index": int})
    media_columns_to_drop = [
        column for column in ("interaction_key", "member_index") if column in media.columns
    ]
    joined = dyad[["file_id", "interaction_key", "member_index"]].merge(
        media.drop(columns=media_columns_to_drop),
        on="file_id",
        validate="one_to_one",
    )
    joined = joined.sort_values(["interaction_key", "member_index"])
    fields = (
        "n_frames_smplh",
        "n_frames_movement",
        "n_frames_box",
        "video_nb_frames",
        "video_start_s",
        "embedded_audio_start_s",
        "video_duration_s",
        "wav_duration_s",
    )
    pairs: list[dict[str, Any]] = []
    equality: dict[str, dict[str, int]] = {
        field: {"available_pairs": 0, "equal_pairs": 0} for field in fields
    }
    for interaction_key, group in joined.groupby("interaction_key", sort=True):
        if len(group) != 2:
            raise ValueError(f"{interaction_key} has {len(group)} dyad rows")
        group = group.sort_values("member_index")
        left, right = group.iloc[0], group.iloc[1]
        record: dict[str, Any] = {
            "interaction_key": interaction_key,
            "label": left["label"],
            "vendor": left["vendor"],
            "members": [left["file_id"], right["file_id"]],
            "measurements": {},
        }
        for field in fields:
            a, b = left[field], right[field]
            available = pd.notna(a) and pd.notna(b)
            if available:
                equality[field]["available_pairs"] += 1
                equality[field]["equal_pairs"] += int(a == b)
            record["measurements"][field] = {
                "member_0": a,
                "member_1": b,
                "delta_member_1_minus_0": b - a if available else None,
            }
        pairs.append(record)
    joined["video_minus_npz_frames"] = (
        joined["video_nb_frames"] - joined["n_frames_smplh"]
    )
    nonzero_individual = joined[joined["video_minus_npz_frames"] != 0][
        [
            "interaction_key",
            "member_index",
            "file_id",
            "vendor",
            "n_frames_smplh",
            "video_nb_frames",
            "video_minus_npz_frames",
            "video_start_s",
            "video_avg_fps",
            "video_duration_s",
            "wav_duration_s",
        ]
    ]
    return {
        "dyads": len(pairs),
        "participant_files": len(joined),
        "pair_equality": equality,
        "pairs": pairs,
        "individual_video_npz_frame_count_exceptions": _records(nonzero_individual),
        "timecode_counts": _records(
            joined.groupby(["vendor", "video_timecode"], dropna=False)
            .size()
            .rename("files")
            .reset_index()
        ),
    }


def _write_atomic(payload: dict[str, Any], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("x", encoding="utf-8") as handle:
            json.dump(_json_safe(payload), handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(output)
    finally:
        temporary.unlink(missing_ok=True)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-root", type=Path, default=REPO_ROOT / "seamless_interaction"
    )
    parser.add_argument(
        "--main-manifest", type=Path, default=REPO_ROOT / "configs/dev_sample.csv"
    )
    parser.add_argument(
        "--dyad-manifest", type=Path, default=REPO_ROOT / "configs/dyad_audit.csv"
    )
    parser.add_argument(
        "--movement-manifest",
        type=Path,
        default=REPO_ROOT / "configs/movement_audit.csv",
    )
    parser.add_argument(
        "--media-probe",
        type=Path,
        default=REPO_ROOT / "outputs/recon/media_probe.parquet",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / "outputs/recon/assumption_audit.json",
    )
    parser.add_argument("--max-files", type=int, default=200)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    source_root = args.source_root.resolve(strict=True)
    output = _checked_output(args.output)
    if _is_relative_to(output, source_root) or _is_relative_to(source_root, output.parent):
        raise ValueError("source and output paths must be disjoint")
    if args.max_files > 200 or args.max_files < 1:
        raise ValueError("--max-files must be between 1 and the Session-1 cap of 200")

    main_rows, union_rows, media = _load_populations(args)
    for record in union_rows:
        _checked_source_npz(source_root, record["source_relbase"])

    result = {
        "provenance": {
            "generated_utc": datetime.now(timezone.utc).isoformat(),
            "script": str(Path(__file__).resolve()),
            "script_sha256": _sha256(Path(__file__).resolve()),
            "source_root": str(source_root),
            "source_access": "read_only_by_code; no source write operation exists",
            "main_manifest": str(args.main_manifest.resolve()),
            "main_manifest_sha256": _sha256(args.main_manifest),
            "dyad_manifest": str(args.dyad_manifest.resolve()),
            "dyad_manifest_sha256": _sha256(args.dyad_manifest),
            "movement_manifest": str(args.movement_manifest.resolve()),
            "movement_manifest_sha256": _sha256(args.movement_manifest),
            "media_probe": str(args.media_probe.resolve()),
            "media_probe_sha256": _sha256(args.media_probe),
            "definitions": {
                "pose_population": "configs/dev_sample.csv; smplh:is_valid frames",
                "pose_delta_population": "adjacent pairs with both smplh:is_valid",
                "camera_sample": "every tenth index among frames where SMPL-H and box are valid",
                "keypoint_population": "all keypoints in nonempty main-manifest files",
                "confidence_quantiles": "deterministic [::101] sample per file",
                "media_population": "deduplicated union of main, dyad, and movement manifests",
                "dyad_population": "all 14 interactions in configs/dyad_audit.csv",
            },
        },
        "scope": {
            "main_files": len(main_rows),
            "union_files": len(union_rows),
            "dyad_rows": int(len(pd.read_csv(args.dyad_manifest))),
            "labels": sorted(ALLOWED_LABELS),
            "split": "dev",
            "max_files_guard": args.max_files,
        },
        "archive_inventory": _archive_inventory(union_rows, source_root),
    }
    result.update(_payload_audit(main_rows, source_root, media))
    result["media"] = _media_audit(media)
    result["dyads"] = _dyad_audit(args.dyad_manifest, media)
    _write_atomic(result, output)
    print(f"wrote bounded D4 audit for {len(union_rows)} files to {output}")


if __name__ == "__main__":
    main()

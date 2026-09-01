#!/usr/bin/env python3
"""Read-only numeric audit of released validity masks on bounded dev manifests.

By default, this analyzes the file-id-deduplicated union of the reconnaissance,
dyad, and movement-audit manifests (184 participant-files in Session 1). All
outputs are aggregate/tabular metadata under ``--out-dir``. Source NPZ files are
opened read-only and processed one at a time.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


MASK_KEYS = {
    "smplh": "smplh:is_valid",
    "movement": "movement:is_valid",
    "box": "boxes_and_keypoints:is_valid_box",
}
PAIR_NAMES = (("smplh", "movement"), ("smplh", "box"), ("movement", "box"))
DEFAULT_MANIFESTS = (
    Path("configs/dev_sample.csv"),
    Path("configs/dyad_audit.csv"),
    Path("configs/movement_audit.csv"),
)


def scalar(value):
    if isinstance(value, np.generic):
        return value.item()
    return value


def quantiles(values, ps=(0, .25, .5, .75, .9, .95, .99, 1)):
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return {f"q{int(p * 100):02d}": None for p in ps}
    return {
        f"q{int(p * 100):02d}": float(np.quantile(arr, p, method="linear"))
        for p in ps
    }


def runs_of_false(valid):
    invalid = ~np.asarray(valid, dtype=bool)
    padded = np.r_[False, invalid, False].astype(np.int8)
    delta = np.diff(padded)
    starts = np.flatnonzero(delta == 1)
    ends = np.flatnonzero(delta == -1)
    return list(zip(starts.tolist(), ends.tolist()))


@dataclass
class Sample:
    """Deterministic bounded sample; exact until cap, then regular subsampling."""

    cap: int = 100_000
    chunks: list = field(default_factory=list)
    n_seen: int = 0

    def add(self, values):
        arr = np.asarray(values).reshape(-1)
        arr = arr[np.isfinite(arr)]
        self.n_seen += int(arr.size)
        if arr.size == 0:
            return
        # Limit each file/batch first, deterministically across its extent.
        batch_cap = 2000
        if arr.size > batch_cap:
            idx = np.linspace(0, arr.size - 1, batch_cap, dtype=np.int64)
            arr = arr[idx]
        self.chunks.append(np.asarray(arr, dtype=np.float64))
        # Periodically compact to a regular sample. This is deterministic, not random.
        size = sum(x.size for x in self.chunks)
        if size > self.cap * 2:
            merged = np.concatenate(self.chunks)
            idx = np.linspace(0, merged.size - 1, self.cap, dtype=np.int64)
            self.chunks = [merged[idx]]

    def array(self):
        if not self.chunks:
            return np.array([], dtype=np.float64)
        merged = np.concatenate(self.chunks)
        if merged.size > self.cap:
            idx = np.linspace(0, merged.size - 1, self.cap, dtype=np.int64)
            merged = merged[idx]
        return merged


@dataclass
class StatusStats:
    frames: int = 0
    elements: int = 0
    nan_elements: int = 0
    posinf_elements: int = 0
    neginf_elements: int = 0
    allfinite_frames: int = 0
    any_nonfinite_frames: int = 0
    allzero_frames: int = 0
    prev_eligible_frames: int = 0
    equal_prev_frames: int = 0
    last_valid_eligible_frames: int = 0
    equal_last_valid_frames: int = 0
    within_valid_range_eligible_frames: int = 0
    within_valid_range_frames: int = 0
    outside_valid_range_elements: int = 0
    outside_valid_range_denominator: int = 0
    values: Sample = field(default_factory=Sample)
    frame_rms: Sample = field(default_factory=Sample)
    frame_maxabs: Sample = field(default_factory=Sample)
    diff_rms: Sample = field(default_factory=Sample)
    diff_maxabs: Sample = field(default_factory=Sample)
    minimum: float | None = None
    maximum: float | None = None

    def update_range(self, finite_values):
        if finite_values.size == 0:
            return
        lo = float(np.min(finite_values))
        hi = float(np.max(finite_values))
        self.minimum = lo if self.minimum is None else min(self.minimum, lo)
        self.maximum = hi if self.maximum is None else max(self.maximum, hi)


@dataclass
class ArrayStats:
    key: str
    mask: str
    schemas: Counter = field(default_factory=Counter)
    files: int = 0
    invalid: StatusStats = field(default_factory=StatusStats)
    valid: StatusStats = field(default_factory=StatusStats)


def update_array_stats(stats, array, valid_mask):
    a = np.asarray(array)
    t = len(valid_mask)
    if a.ndim == 0 or a.shape[0] != t:
        raise ValueError("not frame aligned")
    feature_dim = int(np.prod(a.shape[1:], dtype=np.int64)) if a.ndim > 1 else 1
    flat = a.reshape(t, feature_dim)
    stats.files += 1
    stats.schemas[(str(a.dtype), tuple(a.shape[1:]))] += 1

    # Compute exact per-frame summaries once. Use float64 accumulation for RMS.
    finite = np.isfinite(flat)
    allfinite = np.all(finite, axis=1)
    zeros = np.all(flat == 0, axis=1)
    safe = np.where(finite, flat, 0)
    denom = np.maximum(np.sum(finite, axis=1), 1)
    frame_rms = np.sqrt(np.sum(np.square(safe), axis=1, dtype=np.float64) / denom)
    frame_maxabs = np.max(np.abs(safe), axis=1)

    if t > 1:
        prev_equal = np.all(flat[1:] == flat[:-1], axis=1)
        # NaNs intentionally never compare equal, which is appropriate for exact hold.
        d = flat[1:] - flat[:-1]
        dfinite = np.isfinite(d)
        dsafe = np.where(dfinite, d, 0)
        ddenom = np.maximum(np.sum(dfinite, axis=1), 1)
        diff_rms = np.sqrt(np.sum(np.square(dsafe), axis=1, dtype=np.float64) / ddenom)
        diff_maxabs = np.max(np.abs(dsafe), axis=1)
    else:
        prev_equal = np.array([], dtype=bool)
        diff_rms = np.array([], dtype=np.float64)
        diff_maxabs = np.array([], dtype=np.float64)

    valid_values = flat[np.asarray(valid_mask, dtype=bool)]
    valid_value_finite = np.isfinite(valid_values)
    valid_finite_values = valid_values[valid_value_finite]
    feature_has_valid_range = (
        np.any(valid_value_finite, axis=0)
        if valid_values.shape[0]
        else np.zeros(flat.shape[1], dtype=bool)
    )
    has_complete_valid_range = bool(np.all(feature_has_valid_range))
    if valid_values.shape[0]:
        valid_lo = np.min(
            np.where(valid_value_finite, valid_values, np.inf), axis=0
        )
        valid_hi = np.max(
            np.where(valid_value_finite, valid_values, -np.inf), axis=0
        )
    else:
        valid_lo = np.full(flat.shape[1], np.inf)
        valid_hi = np.full(flat.shape[1], -np.inf)

    # Index of most recent valid frame, to distinguish a true last-valid hold from
    # merely matching the immediately preceding invalid frame.
    indices = np.arange(t)
    last_valid = np.maximum.accumulate(np.where(valid_mask, indices, -1))

    for status_name, select in (("invalid", ~valid_mask), ("valid", valid_mask)):
        out = getattr(stats, status_name)
        idx = np.flatnonzero(select)
        out.frames += int(idx.size)
        out.elements += int(idx.size * flat.shape[1])
        if idx.size == 0:
            continue
        block = flat[idx]
        block_finite = finite[idx]
        out.nan_elements += int(np.isnan(block).sum())
        out.posinf_elements += int(np.isposinf(block).sum())
        out.neginf_elements += int(np.isneginf(block).sum())
        out.allfinite_frames += int(allfinite[idx].sum())
        out.any_nonfinite_frames += int((~allfinite[idx]).sum())
        out.allzero_frames += int(zeros[idx].sum())
        vals = block[block_finite]
        out.update_range(vals)
        out.values.add(vals)
        out.frame_rms.add(frame_rms[idx])
        out.frame_maxabs.add(frame_maxabs[idx])

        eligible_prev = idx[idx > 0]
        out.prev_eligible_frames += int(eligible_prev.size)
        if eligible_prev.size:
            out.equal_prev_frames += int(prev_equal[eligible_prev - 1].sum())
            out.diff_rms.add(diff_rms[eligible_prev - 1])
            out.diff_maxabs.add(diff_maxabs[eligible_prev - 1])

        if status_name == "invalid":
            eligible_last = idx[last_valid[idx] >= 0]
            out.last_valid_eligible_frames += int(eligible_last.size)
            if eligible_last.size:
                equality = np.all(
                    flat[eligible_last] == flat[last_valid[eligible_last]], axis=1
                )
                out.equal_last_valid_frames += int(equality.sum())

            if has_complete_valid_range:
                finite_rows = idx[allfinite[idx]]
                out.within_valid_range_eligible_frames += int(finite_rows.size)
                if finite_rows.size:
                    within = np.all(
                        (flat[finite_rows] >= valid_lo) & (flat[finite_rows] <= valid_hi),
                        axis=1,
                    )
                    out.within_valid_range_frames += int(within.sum())
                ranged_elements = block_finite & feature_has_valid_range[None, :]
                out.outside_valid_range_denominator += int(ranged_elements.sum())
                out.outside_valid_range_elements += int(
                    (
                        ranged_elements
                        & ((block < valid_lo[None, :]) | (block > valid_hi[None, :]))
                    ).sum()
                )


def ratio(num, den):
    return None if den == 0 else float(num / den)


def status_to_row(out):
    result = {
        "frames": out.frames,
        "elements": out.elements,
        "nan_element_frac": ratio(out.nan_elements, out.elements),
        "posinf_element_frac": ratio(out.posinf_elements, out.elements),
        "neginf_element_frac": ratio(out.neginf_elements, out.elements),
        "any_nonfinite_frame_frac": ratio(out.any_nonfinite_frames, out.frames),
        "allzero_frame_frac": ratio(out.allzero_frames, out.frames),
        "equal_prev_frame_frac": ratio(out.equal_prev_frames, out.prev_eligible_frames),
        "equal_prev_denominator": out.prev_eligible_frames,
        "equal_last_valid_frame_frac": ratio(
            out.equal_last_valid_frames, out.last_valid_eligible_frames
        ),
        "equal_last_valid_denominator": out.last_valid_eligible_frames,
        "within_per_file_valid_range_frame_frac": ratio(
            out.within_valid_range_frames, out.within_valid_range_eligible_frames
        ),
        "within_per_file_valid_range_denominator": out.within_valid_range_eligible_frames,
        "outside_per_file_valid_range_element_frac": ratio(
            out.outside_valid_range_elements, out.outside_valid_range_denominator
        ),
        "outside_per_file_valid_range_denominator": out.outside_valid_range_denominator,
        "min": out.minimum,
        "max": out.maximum,
    }
    for prefix, sample in (
        ("value", out.values),
        ("frame_rms", out.frame_rms),
        ("frame_maxabs", out.frame_maxabs),
        ("diff_rms", out.diff_rms),
        ("diff_maxabs", out.diff_maxabs),
    ):
        qs = quantiles(sample.array(), ps=(.01, .5, .9, .99))
        result.update({f"{prefix}_{k}": v for k, v in qs.items()})
        result[f"{prefix}_quantile_sample_n"] = int(sample.array().size)
        result[f"{prefix}_population_n"] = int(sample.n_seen)
    return result


def write_csv(path, rows):
    rows = list(rows)
    if not rows:
        return
    fieldnames = []
    seen = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        action="append",
        default=None,
        help=(
            "CSV manifest; repeat to analyze a file_id-deduplicated union "
            "(default: configs/dev_sample.csv, configs/dyad_audit.csv, and "
            "configs/movement_audit.csv)"
        ),
    )
    parser.add_argument(
        "--source-root", type=Path, default=Path("seamless_interaction")
    )
    parser.add_argument(
        "--out-dir", type=Path, default=Path("outputs/recon/validity_audit")
    )
    args = parser.parse_args()
    manifest_paths = tuple(args.manifest or DEFAULT_MANIFESTS)

    source_root_resolved = args.source_root.resolve()
    out_dir_resolved = args.out_dir.resolve()
    try:
        out_dir_resolved.relative_to(source_root_resolved)
    except ValueError:
        pass
    else:
        parser.error("--out-dir must not be inside the read-only source root")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    manifest_by_file_id = {}
    for manifest_path in manifest_paths:
        with manifest_path.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                prior = manifest_by_file_id.get(row["file_id"])
                if prior is not None and prior["source_relbase"] != row["source_relbase"]:
                    raise ValueError(
                        f"conflicting paths for {row['file_id']}: "
                        f"{prior['source_relbase']} vs {row['source_relbase']}"
                    )
                manifest_by_file_id.setdefault(row["file_id"], row)
    manifest = list(manifest_by_file_id.values())

    per_file = []
    run_rows = []
    mask_schema = defaultdict(Counter)
    mask_value_counts = defaultdict(Counter)
    mask_missing = Counter()
    pair_counts = defaultdict(lambda: np.zeros(4, dtype=np.int64))
    pair_file_fracs = defaultdict(lambda: defaultdict(list))
    pair_excluded = Counter()
    pair_missing = Counter()
    pair_empty = Counter()
    array_stats = {}
    nonaligned = Counter()
    keysets = Counter()
    started = time.time()

    for file_index, row in enumerate(manifest, 1):
        path = args.source_root / f"{row['source_relbase']}.npz"
        with np.load(path, allow_pickle=False) as archive:
            keysets[tuple(archive.files)] += 1
            masks = {}
            lengths = {}
            for mask_name, key in MASK_KEYS.items():
                if key not in archive:
                    mask_missing[(mask_name, row["label"])] += 1
                    mask_missing[(mask_name, "all")] += 1
                    per_file.append(
                        {
                            "file_id": row["file_id"],
                            "label": row["label"],
                            "vendor": row["vendor"],
                            "mask": mask_name,
                            "present": False,
                            "frames": "",
                            "invalid_frames": "",
                            "invalid_frac": "",
                            "n_invalid_runs": "",
                            "max_invalid_run": "",
                        }
                    )
                    continue
                raw = archive[key]
                mask_schema[mask_name][(str(raw.dtype), tuple(raw.shape))] += 1
                vals, counts = np.unique(raw, return_counts=True)
                for val, count in zip(vals, counts):
                    mask_value_counts[mask_name][repr(scalar(val))] += int(count)
                if raw.ndim == 2 and raw.shape[1] == 1:
                    flat_mask = raw[:, 0]
                elif raw.ndim == 1:
                    flat_mask = raw
                else:
                    raise ValueError(f"unexpected mask shape {key}: {raw.shape}")
                if not np.all(np.isin(flat_mask, [0, 1, False, True])):
                    raise ValueError(f"non-binary mask values in {path}: {key}")
                masks[mask_name] = flat_mask.astype(bool)
                lengths[mask_name] = len(flat_mask)

                invalid_count = int((~masks[mask_name]).sum())
                mask_runs = runs_of_false(masks[mask_name])
                per_file.append(
                    {
                        "file_id": row["file_id"],
                        "label": row["label"],
                        "vendor": row["vendor"],
                        "mask": mask_name,
                        "present": True,
                        "frames": len(flat_mask),
                        "invalid_frames": invalid_count,
                        "invalid_frac": (
                            None if len(flat_mask) == 0 else invalid_count / len(flat_mask)
                        ),
                        "n_invalid_runs": len(mask_runs),
                        "max_invalid_run": max((e - s for s, e in mask_runs), default=0),
                    }
                )
                for s, e in mask_runs:
                    run_rows.append(
                        {
                            "file_id": row["file_id"],
                            "label": row["label"],
                            "vendor": row["vendor"],
                            "source_relbase": row["source_relbase"],
                            "mask": mask_name,
                            "start_frame": s,
                            "end_frame_exclusive": e,
                            "length_frames": e - s,
                            "total_frames": len(flat_mask),
                            "start_s_nominal_30fps": s / 30,
                            "end_s_nominal_30fps": e / 30,
                            "touches_start": s == 0,
                            "touches_end": e == len(flat_mask),
                        }
                    )

            # Pairwise counts use invalid as the positive class. Require exact
            # frame alignment rather than silently truncating.
            for a_name, b_name in PAIR_NAMES:
                key_base = f"{a_name}__{b_name}"
                if a_name not in masks or b_name not in masks:
                    pair_missing[(key_base, row["label"])] += 1
                    pair_missing[(key_base, "all")] += 1
                    continue
                if lengths[a_name] != lengths[b_name]:
                    pair_excluded[(key_base, row["label"])] += 1
                    pair_excluded[(key_base, "all")] += 1
                    continue
                if lengths[a_name] == 0:
                    pair_empty[(key_base, row["label"])] += 1
                    pair_empty[(key_base, "all")] += 1
                    continue
                ai = ~masks[a_name]
                bi = ~masks[b_name]
                # Order: both valid (TN), A-invalid/B-valid (FN by naming),
                # A-valid/B-invalid (FP), both invalid (TP).
                counts = np.array(
                    [
                        np.sum(~ai & ~bi),
                        np.sum(ai & ~bi),
                        np.sum(~ai & bi),
                        np.sum(ai & bi),
                    ],
                    dtype=np.int64,
                )
                for label_group in (row["label"], "all"):
                    pair_counts[(key_base, label_group)] += counts
                    pair_file_fracs[(key_base, label_group)][a_name].append(ai.mean())
                    pair_file_fracs[(key_base, label_group)][b_name].append(bi.mean())

            # Arrays are assigned to the validity stream with matching semantics.
            for key in archive.files:
                if key in MASK_KEYS.values():
                    continue
                if key.startswith("smplh:"):
                    mask_name = "smplh"
                elif key.startswith("movement:"):
                    mask_name = "movement"
                elif key.startswith("boxes_and_keypoints:"):
                    mask_name = "box"
                else:
                    continue
                arr = archive[key]
                if key == "movement:hypernet_features":
                    nonaligned[(key, str(arr.dtype), tuple(arr.shape), "15_frame_rate")] += 1
                    continue
                if mask_name not in masks:
                    nonaligned[(key, str(arr.dtype), tuple(arr.shape), "mask_absent")] += 1
                    continue
                if arr.ndim == 0 or arr.shape[0] != len(masks[mask_name]):
                    nonaligned[(key, str(arr.dtype), tuple(arr.shape), "length_or_scalar")] += 1
                    continue
                stat_key = (mask_name, key)
                if stat_key not in array_stats:
                    array_stats[stat_key] = ArrayStats(key=key, mask=mask_name)
                update_array_stats(array_stats[stat_key], arr, masks[mask_name])

        elapsed = time.time() - started
        print(
            f"[{file_index:03d}/{len(manifest)}] {row['file_id']} "
            f"elapsed={elapsed:.1f}s",
            flush=True,
        )

    write_csv(args.out_dir / "per_file_masks.csv", per_file)
    write_csv(args.out_dir / "invalid_runs.csv", run_rows)

    # Exact mask and run summaries by label and overall.
    mask_summaries = []
    run_summaries = []
    for mask_name in MASK_KEYS:
        for label in ("improvised", "naturalistic", "all"):
            pf_present = [
                r for r in per_file
                if r["mask"] == mask_name and r["present"]
                and (label == "all" or r["label"] == label)
            ]
            pf = [r for r in pf_present if int(r["frames"]) > 0]
            rr = [
                r for r in run_rows
                if r["mask"] == mask_name and (label == "all" or r["label"] == label)
            ]
            frames = sum(int(r["frames"]) for r in pf)
            invalid = sum(int(r["invalid_frames"]) for r in pf)
            fracs = [float(r["invalid_frac"]) for r in pf]
            maxruns = [int(r["max_invalid_run"]) for r in pf]
            lengths_run = [int(r["length_frames"]) for r in rr]
            mask_summary = {
                "mask": mask_name,
                "label": label,
                "manifest_files": sum(
                    1 for r in manifest if label == "all" or r["label"] == label
                ),
                "files_present": len(pf_present),
                "files_with_frames": len(pf),
                "files_empty": len(pf_present) - len(pf),
                "files_missing": mask_missing[(mask_name, label)],
                "files_any_invalid": sum(x > 0 for x in fracs),
                "files_any_invalid_frac": ratio(sum(x > 0 for x in fracs), len(pf)),
                "frames": frames,
                "invalid_frames": invalid,
                "pooled_invalid_frac": ratio(invalid, frames),
                "mean_file_invalid_frac": float(np.mean(fracs)) if fracs else None,
                **{f"file_invalid_frac_{k}": v for k, v in quantiles(fracs).items()},
                **{f"file_max_run_{k}": v for k, v in quantiles(maxruns).items()},
            }
            mask_summaries.append(mask_summary)

            categories = {
                "len1": lambda n: n == 1,
                "len2_5": lambda n: 2 <= n <= 5,
                "len6_29": lambda n: 6 <= n <= 29,
                "len30_59": lambda n: 30 <= n <= 59,
                "len60_299": lambda n: 60 <= n <= 299,
                "len300plus": lambda n: n >= 300,
            }
            run_summary = {
                "mask": mask_name,
                "label": label,
                "runs": len(rr),
                "invalid_frames": sum(lengths_run),
                "isolated_run_frac": ratio(sum(n == 1 for n in lengths_run), len(rr)),
                "isolated_invalid_frame_frac": ratio(
                    sum(n for n in lengths_run if n == 1), sum(lengths_run)
                ),
                "burst_ge2_run_frac": ratio(sum(n >= 2 for n in lengths_run), len(rr)),
                "burst_ge2_invalid_frame_frac": ratio(
                    sum(n for n in lengths_run if n >= 2), sum(lengths_run)
                ),
                "edge_run_frac": ratio(
                    sum(bool(r["touches_start"] or r["touches_end"]) for r in rr), len(rr)
                ),
                **{f"run_length_{k}": v for k, v in quantiles(lengths_run).items()},
            }
            for category, predicate in categories.items():
                selected = [n for n in lengths_run if predicate(n)]
                run_summary[f"{category}_runs"] = len(selected)
                run_summary[f"{category}_run_frac"] = ratio(len(selected), len(lengths_run))
                run_summary[f"{category}_frames"] = sum(selected)
                run_summary[f"{category}_invalid_frame_frac"] = ratio(
                    sum(selected), sum(lengths_run)
                )
            run_summaries.append(run_summary)
    write_csv(args.out_dir / "mask_summaries.csv", mask_summaries)
    write_csv(args.out_dir / "run_summaries.csv", run_summaries)

    pair_rows = []
    for (pair, label), counts in sorted(pair_counts.items()):
        tn, a_only, b_only, both = map(int, counts)
        n = tn + a_only + b_only + both
        denom_phi = math.sqrt(
            (both + a_only) * (both + b_only) * (tn + a_only) * (tn + b_only)
        )
        names = pair.split("__")
        a_fracs = pair_file_fracs[(pair, label)][names[0]]
        b_fracs = pair_file_fracs[(pair, label)][names[1]]
        if len(a_fracs) > 1 and np.std(a_fracs) > 0 and np.std(b_fracs) > 0:
            file_frac_pearson = float(np.corrcoef(a_fracs, b_fracs)[0, 1])
        else:
            file_frac_pearson = None
        pair_rows.append(
            {
                "pair": pair,
                "label": label,
                "aligned_files": len(a_fracs),
                "excluded_missing_mask_files": pair_missing[(pair, label)],
                "excluded_length_mismatch_files": pair_excluded[(pair, label)],
                "excluded_empty_files": pair_empty[(pair, label)],
                "frame_denominator": n,
                "both_valid": tn,
                f"{names[0]}_invalid_only": a_only,
                f"{names[1]}_invalid_only": b_only,
                "both_invalid": both,
                "agreement_frac": ratio(tn + both, n),
                "invalid_jaccard": ratio(both, both + a_only + b_only),
                "phi_invalid": None if denom_phi == 0 else (both * tn - a_only * b_only) / denom_phi,
                f"p_{names[1]}_invalid_given_{names[0]}_invalid": ratio(both, both + a_only),
                f"p_{names[0]}_invalid_given_{names[1]}_invalid": ratio(both, both + b_only),
                "pearson_across_file_invalid_fracs": file_frac_pearson,
            }
        )
    write_csv(args.out_dir / "pairwise.csv", pair_rows)

    array_rows = []
    for (_, key), stats in sorted(array_stats.items()):
        schemas = "; ".join(
            f"{count}x dtype={dtype} trailing_shape={shape}"
            for (dtype, shape), count in stats.schemas.items()
        )
        for status_name in ("invalid", "valid"):
            array_rows.append(
                {
                    "mask": stats.mask,
                    "array": key,
                    "status": status_name,
                    "files": stats.files,
                    "schemas": schemas,
                    **status_to_row(getattr(stats, status_name)),
                }
            )
    write_csv(args.out_dir / "array_stats.csv", array_rows)

    # Candidate clips: prioritize long internal runs, then single-edge runs, while
    # keeping both labels and all available masks represented. A short window is
    # centered on one transition rather than spanning an arbitrarily long run.
    candidates = []
    used_files = set()

    def candidate_from_run(run):
        choice = dict(run)
        if run["touches_start"] and not run["touches_end"]:
            anchor_s = float(run["end_s_nominal_30fps"])
        else:
            anchor_s = float(run["start_s_nominal_30fps"])
        choice["suggested_clip_start_s_nominal"] = max(0, anchor_s - 3)
        choice["suggested_clip_end_s_nominal"] = min(
            float(run["total_frames"]) / 30, anchor_s + 5
        )
        return choice

    for label in ("improvised", "naturalistic"):
        for mask_name in ("smplh", "movement", "box"):
            choices = [
                r for r in run_rows
                if r["label"] == label and r["mask"] == mask_name
                and r["file_id"] not in used_files
            ]
            choices.sort(
                key=lambda r: (
                    not (r["touches_start"] or r["touches_end"]),
                    not (r["touches_start"] and r["touches_end"]),
                    r["length_frames"],
                ),
                reverse=True,
            )
            if choices:
                choice = candidate_from_run(choices[0])
                candidates.append(choice)
                used_files.add(choice["file_id"])
    # Fill to 10 with the longest internal runs not already represented.
    extras = [
        r for r in run_rows
        if r["file_id"] not in used_files and not r["touches_start"] and not r["touches_end"]
    ]
    extras.sort(key=lambda r: r["length_frames"], reverse=True)
    for r in extras:
        if len(candidates) >= 10:
            break
        if r["file_id"] in used_files:
            continue
        choice = candidate_from_run(r)
        candidates.append(choice)
        used_files.add(choice["file_id"])
    write_csv(args.out_dir / "candidate_clips.csv", candidates)

    metadata = {
        "manifests": [str(path.resolve()) for path in manifest_paths],
        "source_root": str(args.source_root.resolve()),
        "files": len(manifest),
        "elapsed_s": time.time() - started,
        "mask_schema_counts": {
            name: [
                {"dtype": dtype, "shape": list(shape), "files": count}
                for (dtype, shape), count in counts.items()
            ]
            for name, counts in mask_schema.items()
        },
        "mask_value_counts": {
            name: dict(counts) for name, counts in mask_value_counts.items()
        },
        "mask_missing_counts": {
            f"{name}:{label}": count
            for (name, label), count in mask_missing.items()
        },
        "keyset_counts": [
            {"files": count, "keys": list(keys)} for keys, count in keysets.items()
        ],
        "non_frame_aligned_arrays": [
            {
                "array": key,
                "dtype": dtype,
                "shape": list(shape),
                "reason": reason,
                "files": count,
            }
            for (key, dtype, shape, reason), count in nonaligned.items()
        ],
        "quantile_method": "numpy linear; exact for mask/run tables; deterministic regular samples capped at 100000 for array-value/frame summaries",
        "definitions": {
            "invalid": "logical NOT of binary released validity mask; movement float32 (T,1) is squeezed then cast to bool",
            "equal_prev": "all array elements at current frame exactly equal previous frame; denominator excludes frame 0",
            "equal_last_valid": "all elements exactly equal most recent preceding valid frame; invalid leading frames excluded",
            "within_per_file_valid_range": "all finite feature elements in an invalid frame fall within that feature dimension's finite valid min/max in the same array and file",
            "diff_rms": "RMS elementwise current-minus-previous-frame difference, assigned by current frame validity",
        },
    }
    (args.out_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps({"out_dir": str(args.out_dir), "elapsed_s": metadata["elapsed_s"]}))


if __name__ == "__main__":
    main()

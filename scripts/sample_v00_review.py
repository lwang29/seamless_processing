#!/usr/bin/env python3
"""Build the V00-only, label-stratified review manifest for the second review.

Design, following the user's brief:

* **V00 only, charades excluded.** The vendor decision is made; charades is
  excluded because the model is speech-conditioned.
* **Uniform group, uniform within each label.** Two separate equal-size draws —
  one within V00 naturalistic, one within V00 improvised — so each arm is an
  unbiased base-rate sample *of its own label* and the two arms have equal
  precision. Equal allocation is a departure from the corpus label split (60%
  improvised by file count), which is the point: the question being answered is
  "naturalistic versus improvised", and that needs equal power per arm, not
  proportional. The design weights needed to recover a corpus-level rate are
  written into the summary.
* **Targeted group, kept separate.** Extremes from the reviewer's own vocabulary
  proxies plus the Session-1 signal extremes. This group over-represents
  extremes by construction and can never be read as a base rate.
* **grounded_gesture guaranteed representation** in both labels, since it is the
  highest-value pool.
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import subprocess
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import yaml


CLIP_DURATION_S = 10.0
MANIFEST_COLUMNS = [
    "review_item_id", "clip_id", "file_id", "source_relbase",
    "partner_source_relbase", "sample_group", "label_stratum", "start_frame",
    "vendor", "label", "split", "activity_type", "selection_reason",
    "signals_json", "render_policy",
]


@dataclass(frozen=True)
class Candidate:
    file_id: str
    source_relbase: str
    sample_group: str
    label_stratum: str
    selection_reason: str
    signals: dict[str, Any]
    start_frame_override: int | None = None


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


def _unit(*parts: Any) -> float:
    digest = hashlib.sha256("|".join(str(part) for part in parts).encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") / 2.0**64


def load_population(inventory: Path) -> pd.DataFrame:
    frame = pd.read_parquet(inventory)
    frame["vendor"] = "V" + frame.vendor_id.astype(str)
    frame["eligible"] = (
        frame.vendor_id.eq("00")
        & frame.all_modalities_present.astype(bool)
        & frame.probe_status.eq("ok")
        & frame.video_stream_present.astype(bool)
        & frame.observed_duration_s.notna()
        & frame.observed_duration_s.ge(30.0)
        & frame.video_nb_frames.notna()
        & frame.interaction_type.ne("charades")
    )
    return frame


def _partners(frame: pd.DataFrame) -> dict[str, str]:
    partners: dict[str, str] = {}
    for _, group in frame.groupby(["vendor_id", "session_id", "interaction_id"], sort=False):
        if len(group) != 2:
            continue
        rows = list(group.itertuples(index=False))
        partners[rows[0].file_id] = rows[1].source_relbase
        partners[rows[1].file_id] = rows[0].source_relbase
    return partners


def _member_frame_count(archive: Any, name: str) -> int | None:
    """Frame count from the member's ``.npy`` header, without decompressing it.

    ``archive[name].shape[0]`` inflates the entire DEFLATE member — tens of
    megabytes for the keypoint array — purely to learn its length. Reading the
    header is a few hundred bytes. Sampling several hundred candidates made the
    difference between minutes and tens of minutes.
    """

    try:
        with archive.zip.open(f"{name}.npy") as handle:
            version = np.lib.format.read_magic(handle)
            shape, _, _ = np.lib.format._read_array_header(handle, version)
    except (KeyError, OSError, ValueError):
        return None
    return int(shape[0]) if shape else 0


def _annotation_frames(source_root: Path, relbase: str) -> tuple[int, dict[str, Any]]:
    path = (source_root / relbase).with_suffix(".npz")
    required = (
        "boxes_and_keypoints:keypoints", "boxes_and_keypoints:box",
        "boxes_and_keypoints:is_valid_box", "smplh:is_valid",
        "smplh:global_orient", "smplh:body_pose", "smplh:left_hand_pose",
        "smplh:right_hand_pose", "smplh:translation",
    )
    with np.load(path, allow_pickle=False) as archive:
        names = set(archive.files)
        missing = sorted(set(required) - names)
        if missing:
            return 0, {"npz_status": "missing_arrays:" + ",".join(missing)}
        lengths: dict[str, int] = {}
        for name in required:
            count = _member_frame_count(archive, name)
            if count is None:
                return 0, {"npz_status": f"unreadable_header:{name}"}
            lengths[name] = count
        # Only the two small mask arrays are actually decompressed.
        box_valid = np.asarray(archive["boxes_and_keypoints:is_valid_box"]).reshape(-1).astype(bool)
        smplh_valid = np.asarray(archive["smplh:is_valid"]).reshape(-1).astype(bool)
    covered = min(lengths.values())
    return covered, {
        "npz_status": "ok" if len(set(lengths.values())) == 1 else "array_length_mismatch",
        "npz_frames_min": covered,
        "valid_frac_smplh": round(float(smplh_valid.mean()), 6) if smplh_valid.size else None,
        "valid_frac_box": round(float(box_valid.mean()), 6) if box_valid.size else None,
    }


def _resolve(
    candidates: Iterable[Candidate],
    frame: pd.DataFrame,
    partners: dict[str, str],
    source_root: Path,
    seed: int,
    limit: int | None,
    workers: int = 16,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Resolve candidates in deterministic order, reading NPZ headers in parallel.

    Each candidate costs about two seconds of NFS latency, so serial resolution
    of several hundred takes a quarter of an hour. Candidates are resolved in
    ordered batches slightly larger than the remaining quota and the results are
    consumed in the original order, so the outcome is identical to the serial
    version regardless of how the threads interleave.
    """

    by_id = frame.set_index("file_id")
    ordered = list(candidates)
    accepted: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    position = 0
    while position < len(ordered):
        if limit is not None and len(accepted) >= limit:
            break
        needed = len(ordered) - position if limit is None else limit - len(accepted)
        batch = ordered[position : position + max(1, int(needed * 1.25) + 4)]
        position += len(batch)
        with ThreadPoolExecutor(max_workers=workers) as pool:
            resolved = list(
                pool.map(
                    lambda item: _resolve_one(item, by_id, partners, source_root, seed),
                    batch,
                )
            )
        for outcome in resolved:
            if limit is not None and len(accepted) >= limit:
                break
            if "reject_reason" in outcome:
                rejected.append(outcome)
            else:
                accepted.append(outcome)
    return accepted, rejected


def _resolve_one(
    candidate: Candidate,
    by_id: pd.DataFrame,
    partners: dict[str, str],
    source_root: Path,
    seed: int,
) -> dict[str, Any]:
    row = by_id.loc[candidate.file_id]
    signals = dict(candidate.signals)
    record = {
        "review_item_id": "", "clip_id": "",
        "file_id": candidate.file_id,
        "source_relbase": candidate.source_relbase,
        "partner_source_relbase": partners.get(candidate.file_id, ""),
        "sample_group": candidate.sample_group,
        "label_stratum": candidate.label_stratum,
        "vendor": str(row.vendor), "label": str(row.label), "split": str(row.split),
        "activity_type": str(row.interaction_type),
        "selection_reason": candidate.selection_reason,
        "render_policy": "render",
    }
    fps = float(row.video_avg_fps) if pd.notna(row.video_avg_fps) else float("nan")
    if not np.isfinite(fps) or fps <= 0:
        return {**record, "reject_reason": "unusable_avg_frame_rate"}
    render_frames = int(round(CLIP_DURATION_S * fps))
    try:
        covered, npz_summary = _annotation_frames(source_root, candidate.source_relbase)
    except (OSError, ValueError, KeyError) as exc:
        return {**record, "reject_reason": f"npz_read_error:{type(exc).__name__}"}
    signals.update(npz_summary)
    usable = min(int(row.video_nb_frames), covered)
    if usable < render_frames:
        return {**record, "reject_reason": f"usable_frames_{usable}_below_{render_frames}"}
    span = usable - render_frames
    start = (
        candidate.start_frame_override
        if candidate.start_frame_override is not None
        else int(_unit(seed, "start", candidate.file_id) * (span + 1))
    )
    start = max(0, min(start, span))
    signals.update(
        {"render_frames": render_frames, "avg_fps": round(fps, 6),
         "video_nb_frames": int(row.video_nb_frames)}
    )
    record["start_frame"] = start
    record["signals_json"] = json.dumps(signals, sort_keys=True)
    return record


def uniform_candidates(
    frame: pd.DataFrame, label: str, seed: int, guaranteed_activity: str, guaranteed: int
) -> list[Candidate]:
    """Uniform within one label, with a guaranteed floor for one activity.

    The floor is applied by drawing the guaranteed activity's quota first from
    its own uniform draw, then filling the remainder from the whole label. Both
    parts stay uniform within their own frame, and the summary records the floor
    so the realized activity mix is never mistaken for the label's natural one.
    """

    population = frame.loc[frame.eligible & frame.label.eq(label)].copy()
    population["draw"] = [_unit(seed, "uniform", label, file_id) for file_id in population.file_id]
    population = population.sort_values("draw", kind="mergesort")

    floor = population.loc[population.interaction_type.eq(guaranteed_activity)].head(guaranteed)
    remainder = population.loc[~population.file_id.isin(set(floor.file_id))]
    ordered = pd.concat([floor, remainder], ignore_index=True)
    reasons = {
        **{file_id: f"guaranteed_floor:{guaranteed_activity}" for file_id in floor.file_id},
    }
    return [
        Candidate(
            file_id=row.file_id, source_relbase=row.source_relbase,
            sample_group=f"uniform_{label}", label_stratum=label,
            selection_reason=reasons.get(row.file_id, f"uniform random within V00 {label}"),
            signals={"uniform_draw": round(float(row.draw), 9), "label_stratum": label},
        )
        for row in ordered.itertuples(index=False)
    ]


def _pool_extremes(worktree: Path, frame: pd.DataFrame) -> list[tuple[str, str, dict[str, Any]]]:
    """Extremes for the reviewer's own vocabulary, from the V00 pool signals.

    Every selector names the pool it came from, because the pool is a sample:
    these are the most extreme values in roughly a fifth of V00, not corpus
    maxima. Both tails are taken where both are interesting — a person filling
    5% of the frame and one cropped by the edge are different framing failures.
    """

    shards = sorted(glob.glob(str(worktree / "outputs/session2/v00_pool_signals/shards/*.parquet")))
    if not shards:
        return []
    pool = pd.concat([pd.read_parquet(path) for path in shards], ignore_index=True)
    pool = pool.loc[pool.status.eq("ok")]
    eligible_ids = set(frame.loc[frame.eligible, "file_id"])
    pool = pool.loc[pool.file_id.isin(eligible_ids)]
    if pool.empty:
        return []

    selectors: list[tuple[str, str, bool, str]] = [
        # (column, reason, ascending, vocabulary term)
        ("framing_person_height_frac_p50", "smallest_person_in_frame", True, "framing"),
        ("framing_person_height_frac_p50", "largest_person_in_frame", False, "framing"),
        ("framing_min_edge_margin_frac_p50", "closest_to_raster_edge", True, "framing"),
        ("framing_frames_touching_edge_frac", "most_frames_touching_edge", False, "framing"),
        ("framing_body_out_of_frame_point_frac", "most_body_points_outside_raster", False, "framing"),
        ("framing_centre_offset_x_frac_p50", "most_off_centre_horizontally", False, "framing"),
        ("roll_shoulder_abs_deg_p50", "largest_shoulder_line_tilt", False, "roll"),
        ("roll_shoulder_deg_iqr", "most_variable_shoulder_tilt", False, "roll"),
        ("roll_shoulder_minus_hip_deg_p50", "largest_shoulder_versus_hip_tilt", False, "roll"),
        ("posture_leg_over_torso_p50", "lowest_leg_to_torso_ratio_sitting_candidate", True, "sitting"),
        ("posture_knee_drop_over_torso_p50", "knees_least_below_hips_sitting_candidate", True, "sitting"),
        ("posture_ankles_usable_frac", "ankles_least_often_tracked", True, "sitting"),
        ("hand_activity_static_frac", "hands_most_static_across_whole_file", False, "static_hands"),
        ("hand_activity_left_speed_sw_per_s_p50", "slowest_left_wrist_whole_file", True, "static_hands"),
        ("hand_activity_left_above_hip_frac", "left_hand_least_often_above_hips", True, "static_hands"),
        ("hand_activity_right_above_hip_frac", "right_hand_least_often_above_hips", True, "static_hands"),
        ("hand_activity_left_speed_sw_per_s_p95", "fastest_left_wrist_whole_file", False, "static_hands"),
        ("file_speaking_frac", "least_speech_in_whole_file", True, "static_hands"),
    ]
    picks: list[tuple[str, str, dict[str, Any]]] = []
    used: set[str] = set()
    for column, reason, ascending, term in selectors:
        if column not in pool.columns:
            continue
        series = pd.to_numeric(pool[column], errors="coerce")
        subset = pool.loc[series.notna()].copy()
        if subset.empty:
            continue
        subset["value"] = series[series.notna()]
        subset = subset.sort_values(["value", "file_id"], ascending=[ascending, True], kind="mergesort")
        for row in subset.itertuples(index=False):
            if row.file_id in used:
                continue
            used.add(row.file_id)
            picks.append(
                (
                    str(row.file_id),
                    f"vocab_{term}:{reason}",
                    {
                        "vocabulary_term": term, "selector_column": column,
                        "selector_value": float(row.value),
                        "pool": "v00_pool_signals",
                        "targeted_role": "vocabulary_extreme",
                        "label_of_pick": str(row.label),
                    },
                )
            )
            break

    # Desync candidates. The mouth/audio lag proxy failed validation, so these
    # come from container-level mismatch and from the weak proxy's *worst
    # correlation among talkative, well-tracked files*, where a low correlation
    # is least explainable by silence or a turned head. Both are labelled as
    # container-level or weak so nobody reads them as a desync measurement.
    if "best_av_lag_peak_r" in pool.columns and "file_speaking_frac" in pool.columns:
        talkative = pool.loc[
            pd.to_numeric(pool.file_speaking_frac, errors="coerce").ge(0.35)
            & pd.to_numeric(pool.best_mouth_usable_frame_frac, errors="coerce").ge(0.95)
            & pd.to_numeric(pool.best_av_lag_peak_r, errors="coerce").notna()
            & ~pool.file_id.isin(used)
        ].copy()
        if not talkative.empty:
            talkative["value"] = pd.to_numeric(talkative.best_av_lag_peak_r, errors="coerce")
            for ascending, reason in (
                (True, "weakest_mouth_audio_agreement_among_talkative_files"),
                (False, "strongest_mouth_audio_agreement_among_talkative_files"),
            ):
                ordered = talkative.sort_values(
                    ["value", "file_id"], ascending=[ascending, True], kind="mergesort"
                )
                row = ordered.iloc[0]
                if row.file_id in used:
                    continue
                used.add(str(row.file_id))
                picks.append(
                    (
                        str(row.file_id),
                        f"vocab_desync_weak_proxy:{reason}",
                        {
                            "vocabulary_term": "desync_lag",
                            "proxy_status": "FAILED_VALIDATION_on_V00_resolution",
                            "best_av_lag_peak_r": float(row.value),
                            "best_av_lag_frames": float(row.best_av_lag_frames)
                            if pd.notna(row.best_av_lag_frames) else None,
                            "file_speaking_frac": float(row.file_speaking_frac),
                            "targeted_role": "vocabulary_extreme",
                            "label_of_pick": str(row.label),
                        },
                    )
                )
    return picks


def _container_desync_extremes(frame: pd.DataFrame) -> list[tuple[str, str, dict[str, Any]]]:
    """Container-level timing mismatch: the only desync evidence we actually have."""

    eligible = frame.loc[frame.eligible].copy()
    picks: list[tuple[str, str, dict[str, Any]]] = []
    keys = ["vendor_id", "session_id", "interaction_id"]
    grouped = eligible.groupby(keys).observed_duration_s
    eligible["member_spread_s"] = grouped.transform("max") - grouped.transform("min")
    eligible["members"] = grouped.transform("size")
    pairs = eligible.loc[eligible.members.eq(2)]
    if not pairs.empty:
        row = pairs.loc[pairs.member_spread_s.idxmax()]
        picks.append(
            (
                str(row.file_id),
                "vocab_desync_container:largest_within_interaction_duration_spread",
                {
                    "vocabulary_term": "desync_lag",
                    "evidence_level": "container_level_only_not_within_file_drift",
                    "member_spread_s": round(float(row.member_spread_s), 3),
                    "targeted_role": "vocabulary_extreme",
                },
            )
        )
    # Video duration against embedded-audio duration inside the same container.
    if "embedded_audio_duration_s" in eligible.columns:
        delta = (
            pd.to_numeric(eligible.observed_duration_s, errors="coerce")
            - pd.to_numeric(eligible.embedded_audio_duration_s, errors="coerce")
        ).abs()
        candidate = eligible.loc[delta.notna()].copy()
        if not candidate.empty:
            candidate["value"] = delta[delta.notna()]
            row = candidate.loc[candidate.value.idxmax()]
            picks.append(
                (
                    str(row.file_id),
                    "vocab_desync_container:largest_video_versus_embedded_audio_duration_gap",
                    {
                        "vocabulary_term": "desync_lag",
                        "evidence_level": "container_level_only_not_within_file_drift",
                        "abs_duration_gap_s": round(float(row.value), 4),
                        "targeted_role": "vocabulary_extreme",
                    },
                )
            )
    return picks


def _session1_extremes(worktree: Path) -> list[tuple[str, str, dict[str, Any], int | None]]:
    """Session-1 dev signal extremes, restricted to V00 by the caller."""

    picks: list[tuple[str, str, dict[str, Any], int | None]] = []
    parquets = sorted(glob.glob(str(worktree / "outputs/harness/files/*/*/V00/*.parquet")))
    if parquets:
        windows = pd.concat([pd.read_parquet(path) for path in parquets], ignore_index=True)
        per_file = windows.groupby("file_id").agg(
            wrist_max=("wrist_speed_p90", "max"),
            wrist_min=("wrist_speed_p90", "min"),
            mismatch_max=("duration_mismatch_s", "max"),
            valid_min=("valid_frac_smplh_and_box", "min")
            if "valid_frac_smplh_and_box" in windows.columns else ("valid_frac_all", "min"),
        )
        for column, ascending, reason in (
            ("wrist_max", False, "session1_v00_highest_wrist_speed_p90"),
            ("wrist_min", True, "session1_v00_lowest_wrist_speed_p90"),
            ("mismatch_max", False, "session1_v00_largest_duration_mismatch"),
            ("valid_min", True, "session1_v00_lowest_window_validity"),
        ):
            series = per_file[column].dropna()
            if series.empty:
                continue
            file_id = series.idxmin() if ascending else series.idxmax()
            picks.append(
                (str(file_id), reason,
                 {"session1_metric": column, "session1_value": float(series.loc[file_id]),
                  "targeted_role": "session1_extreme"}, None)
            )
    runs_path = worktree / "outputs/recon/validity_audit/invalid_runs.csv"
    if runs_path.is_file():
        runs = pd.read_csv(runs_path)
        runs = runs.loc[runs.vendor.eq("V00")]
        for mask_name in ("smplh", "box", "movement"):
            subset = runs.loc[runs["mask"] == mask_name]
            if subset.empty:
                continue
            longest = subset.loc[subset.length_frames.idxmax()]
            picks.append(
                (
                    str(longest.file_id), f"session1_v00_longest_{mask_name}_invalid_run",
                    {"mask": mask_name, "run_length_frames": int(longest.length_frames),
                     "targeted_role": "session1_extreme"},
                    max(0, int(longest.start_frame) - 90),
                )
            )
    return picks


def targeted_candidates(
    frame: pd.DataFrame, worktree: Path, seed: int, exclude: set[str], total: int
) -> list[Candidate]:
    known = set(frame.loc[frame.eligible, "file_id"])
    by_id = frame.set_index("file_id")
    used = set(exclude)
    candidates: list[Candidate] = []

    extremes: list[tuple[str, str, dict[str, Any], int | None]] = [
        *[(file_id, reason, signals, None) for file_id, reason, signals in _pool_extremes(worktree, frame)],
        *[(file_id, reason, signals, None) for file_id, reason, signals in _container_desync_extremes(frame)],
        *_session1_extremes(worktree),
    ]
    for file_id, reason, signals, start in extremes:
        if file_id in used or file_id not in known:
            continue
        row = by_id.loc[file_id]
        used.add(file_id)
        candidates.append(
            Candidate(
                file_id=file_id, source_relbase=str(row.source_relbase),
                sample_group="targeted", label_stratum=str(row.label),
                selection_reason=f"extreme:{reason}", signals=signals,
                start_frame_override=start,
            )
        )

    remaining = total - len(candidates)
    population = frame.loc[frame.eligible & ~frame.file_id.isin(used)].copy()
    population["stratum"] = population.label + "|" + population.interaction_type
    population["draw"] = [_unit(seed, "targeted", file_id) for file_id in population.file_id]
    population = population.sort_values(["stratum", "draw"], kind="mergesort")
    per_stratum = {
        stratum: list(group.itertuples(index=False))
        for stratum, group in population.groupby("stratum", sort=True)
    }
    order = sorted(per_stratum)
    fill: list[Any] = []
    depth = 0
    while len(fill) < remaining and any(len(per_stratum[key]) > depth for key in order):
        for stratum in order:
            if len(fill) >= remaining:
                break
            if len(per_stratum[stratum]) > depth:
                fill.append(per_stratum[stratum][depth])
        depth += 1
    for row in fill:
        candidates.append(
            Candidate(
                file_id=row.file_id, source_relbase=row.source_relbase,
                sample_group="targeted", label_stratum=str(row.label),
                selection_reason=f"stratified_fill:{row.stratum}",
                signals={"targeted_role": "stratified_fill", "stratum": row.stratum},
            )
        )
    return candidates


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/v00_review.yaml"))
    parser.add_argument(
        "--inventory", type=Path,
        default=Path("outputs/02_inventory/summary/inventory_joined.parquet"),
    )
    args = parser.parse_args()

    worktree = _worktree()
    config = yaml.safe_load((worktree / args.config).read_text(encoding="utf-8"))
    sampling = config["sampling"]
    source_root = (worktree / str(config["source_root"]).lstrip("./")).resolve()
    outputs = config["outputs"]
    manifest_path = worktree / str(outputs["manifest"]).lstrip("./")
    report_dir = worktree / str(outputs["report_dir"]).lstrip("./")
    report_dir.mkdir(parents=True, exist_ok=True)

    frame = load_population(worktree / args.inventory)
    partners = _partners(frame)
    per_label = int(sampling["uniform_per_label"])
    targeted_total = int(sampling["targeted_count"])
    guaranteed_activity = str(sampling["guaranteed_activity"])
    guaranteed = int(sampling["guaranteed_activity_per_label"])

    accepted: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    for label in ("naturalistic", "improvised"):
        seed = int(sampling["uniform_seeds"][label])
        rows, drops = _resolve(
            uniform_candidates(frame, label, seed, guaranteed_activity, guaranteed),
            frame, partners, source_root, seed, limit=per_label,
        )
        accepted.extend(rows)
        rejected.extend(drops)

    targeted_seed = int(sampling["targeted_seed"])
    targeted_rows, targeted_drops = _resolve(
        targeted_candidates(
            frame, worktree, targeted_seed,
            exclude={record["file_id"] for record in accepted}, total=targeted_total,
        ),
        frame, partners, source_root, targeted_seed, limit=targeted_total,
    )
    records = accepted + targeted_rows
    rejected.extend(targeted_drops)

    for index, record in enumerate(records):
        record["clip_id"] = (
            f"{record['sample_group']}_{index:03d}_{record['file_id']}"
            f"_f{int(record['start_frame']):07d}"
        )
        record["review_item_id"] = f"v00_{index:03d}"

    manifest = pd.DataFrame(records, columns=MANIFEST_COLUMNS)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(manifest_path, index=False)
    for label in ("naturalistic", "improvised"):
        subset = manifest.loc[manifest.sample_group.eq(f"uniform_{label}")]
        subset.to_csv(manifest_path.with_name(f"v00_review_uniform_{label}.csv"), index=False)
    manifest.loc[manifest.sample_group.eq("targeted")].to_csv(
        manifest_path.with_name("v00_review_targeted.csv"), index=False
    )

    duplicate_fraction = float(sampling["duplicate_fraction"])
    duplicate_seed = int(sampling["duplicate_seed"])
    duplicate_count = int(round(duplicate_fraction * len(records)))
    duplicates = sorted(
        records, key=lambda record: _unit(duplicate_seed, "duplicate", record["review_item_id"])
    )[:duplicate_count]
    with_duplicates = [dict(record) for record in records]
    for index, record in enumerate(duplicates):
        copy = dict(record)
        copy["review_item_id"] = f"v00_dup_{index:03d}"
        copy["selection_reason"] = f"agreement_duplicate_of:{record['review_item_id']}"
        with_duplicates.append(copy)
    pd.DataFrame(with_duplicates, columns=MANIFEST_COLUMNS).to_csv(
        manifest_path.with_name("v00_review_with_duplicates.csv"), index=False
    )
    (report_dir / "duplicate_map.json").write_text(
        json.dumps(
            {f"v00_dup_{index:03d}": record["review_item_id"]
             for index, record in enumerate(duplicates)},
            indent=2, sort_keys=True,
        ), encoding="utf-8",
    )

    eligible = frame.loc[frame.eligible]
    label_counts = eligible.label.value_counts().to_dict()
    label_hours = (
        eligible.groupby("label").observed_duration_s.sum() / 3600
    ).round(3).to_dict()
    total_files = int(len(eligible))

    def counts(subset: pd.DataFrame, keys: list[str]) -> list[dict[str, Any]]:
        if subset.empty:
            return []
        return subset.groupby(keys, dropna=False).size().reset_index(name="clips").to_dict("records")

    summary = {
        "vendor": "V00 only",
        "charades": "excluded (speech-conditioned model)",
        "eligibility": (
            "V00, all four modalities present, ffprobe ok, video stream, duration >= 30 s, "
            "parseable frame count, interaction_type != charades"
        ),
        "eligible_files": total_files,
        "eligible_files_by_label": label_counts,
        "eligible_participant_hours_by_label": label_hours,
        "design": {
            "uniform_allocation": "equal per label, NOT proportional to the corpus",
            "corpus_label_share_by_files": {
                label: round(count / total_files, 4) for label, count in label_counts.items()
            },
            "reweighting_note": (
                "A corpus-level rate is the label-share-weighted combination of the two "
                "uniform arms; each arm on its own is unbiased for its own label."
            ),
            "guaranteed_activity": guaranteed_activity,
            "guaranteed_activity_per_label": guaranteed,
        },
        "seeds": {
            "uniform": dict(sampling["uniform_seeds"]),
            "targeted": targeted_seed,
            "duplicates": duplicate_seed,
        },
        "counts": {
            "uniform_naturalistic": int((manifest.sample_group == "uniform_naturalistic").sum()),
            "uniform_improvised": int((manifest.sample_group == "uniform_improvised").sum()),
            "targeted": int((manifest.sample_group == "targeted").sum()),
            "targeted_vocabulary_extremes": int(
                manifest.selection_reason.str.startswith("extreme:vocab_").sum()
            ),
            "targeted_session1_extremes": int(
                manifest.selection_reason.str.startswith("extreme:session1_").sum()
            ),
            "targeted_stratified_fill": int(
                manifest.selection_reason.str.startswith("stratified_fill:").sum()
            ),
            "total_clips": len(records),
            "items_with_duplicates": len(with_duplicates),
            "duplicate_items": duplicate_count,
            "rejected_candidates": len(rejected),
        },
        "realized_by_group_and_activity": counts(manifest, ["sample_group", "activity_type"]),
        "realized_by_group_and_split": counts(manifest, ["sample_group", "split"]),
        "targeted_extreme_reasons": sorted(
            manifest.loc[manifest.selection_reason.str.startswith("extreme:"), "selection_reason"]
        ),
        "distinct_participants_sampled": int(
            frame.set_index("file_id").loc[manifest.file_id].participant_id.nunique()
        ),
        "rejected_reason_counts": (
            pd.DataFrame(rejected).reject_reason.value_counts().to_dict() if rejected else {}
        ),
        **_git_provenance(worktree),
    }
    if rejected:
        pd.DataFrame(rejected).to_csv(report_dir / "rejected_candidates.csv", index=False)
    (report_dir / "sampling_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True, default=str), encoding="utf-8"
    )
    print(json.dumps(summary["counts"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

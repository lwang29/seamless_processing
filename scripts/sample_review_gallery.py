#!/usr/bin/env python3
"""Build the Session-2 Phase-0 review manifest: 200 uniform + 100 targeted.

The two groups are drawn and recorded separately and never mixed, because the
uniform group is the only thing that establishes a base rate. The targeted
group deliberately over-represents extremes and therefore cannot be read as a
frequency estimate.

Population and start-frame choice are the only decisions made here. This script
performs no filtering, applies no threshold, and rejects nothing on quality
grounds: a candidate is dropped only when a full 10-second clip provably cannot
be rendered from it (short media, absent annotation coverage, or no video
stream), and every such drop is recorded.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import yaml


CLIP_DURATION_S = 10.0
# Extremes are one clip each unless a selector names several, so the targeted
# group keeps room for the stratified fill that makes it readable per cell.
TARGETED_TOTAL = 100
UNIFORM_TOTAL = 200

MANIFEST_COLUMNS = [
    "review_item_id",
    "clip_id",
    "file_id",
    "source_relbase",
    "partner_source_relbase",
    "sample_group",
    "start_frame",
    "vendor",
    "label",
    "split",
    "activity_type",
    "selection_reason",
    "signals_json",
    "render_policy",
]


@dataclass(frozen=True)
class Candidate:
    file_id: str
    source_relbase: str
    sample_group: str
    selection_reason: str
    signals: dict[str, Any]
    render_policy: str = "render"
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
        check=True,
        capture_output=True,
        text=True,
    )
    return {"git_sha": sha.stdout.strip(), "git_dirty": bool(status.stdout.strip())}


def _stable_unit(*parts: Any) -> float:
    """Deterministic value in [0, 1) from a seed and identifiers.

    A hash of the identifiers is used instead of a shuffled index so that the
    draw for one file does not change when unrelated rows enter or leave the
    population.
    """

    digest = hashlib.sha256("|".join(str(part) for part in parts).encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") / 2.0**64


def load_population(inventory: Path) -> pd.DataFrame:
    frame = pd.read_parquet(inventory)
    frame["vendor"] = "V" + frame.vendor_id.astype(str)
    frame["renderable_headers"] = (
        frame.all_modalities_present.astype(bool)
        & frame.probe_status.eq("ok")
        & frame.video_stream_present.astype(bool)
        & frame.observed_duration_s.notna()
        & frame.observed_duration_s.ge(CLIP_DURATION_S)
        & frame.video_nb_frames.notna()
    )
    return frame


def _interaction_partners(frame: pd.DataFrame) -> dict[str, str]:
    """Map each file_id to its dyad partner's ``source_relbase``.

    The interaction key is (vendor, session, interaction); a member count other
    than two leaves the partner blank rather than guessing.
    """

    partners: dict[str, str] = {}
    keys = ["vendor_id", "session_id", "interaction_id"]
    for _, group in frame.groupby(keys, dropna=False, sort=False):
        if len(group) != 2:
            continue
        rows = list(group.itertuples(index=False))
        partners[rows[0].file_id] = rows[1].source_relbase
        partners[rows[1].file_id] = rows[0].source_relbase
    return partners


def _annotation_frames(source_root: Path, relbase: str) -> tuple[int, dict[str, Any]]:
    """Frame count covered by the arrays the renderer needs, plus mask summary.

    Only the small mask arrays and the array shapes are read. Keypoint and box
    payloads are not materialized.
    """

    path = (source_root / relbase).with_suffix(".npz")
    with np.load(path, allow_pickle=False) as archive:
        names = set(archive.files)
        required = {
            "boxes_and_keypoints:keypoints",
            "boxes_and_keypoints:box",
            "boxes_and_keypoints:is_valid_box",
            "smplh:is_valid",
            "smplh:global_orient",
            "smplh:body_pose",
            "smplh:left_hand_pose",
            "smplh:right_hand_pose",
            "smplh:translation",
        }
        missing = sorted(required - names)
        if missing:
            return 0, {"npz_status": "missing_arrays:" + ",".join(missing)}
        box_valid = np.asarray(archive["boxes_and_keypoints:is_valid_box"]).reshape(-1).astype(bool)
        smplh_valid = np.asarray(archive["smplh:is_valid"]).reshape(-1).astype(bool)
        movement = (
            np.asarray(archive["movement:is_valid"]).reshape(-1)
            if "movement:is_valid" in names
            else None
        )
        lengths = {name: int(archive[name].shape[0]) for name in sorted(required)}
    covered = min(lengths.values())
    summary: dict[str, Any] = {
        "npz_status": "ok" if len(set(lengths.values())) == 1 else "array_length_mismatch",
        "npz_frames_min": covered,
        "valid_frac_smplh": round(float(smplh_valid.mean()), 6) if smplh_valid.size else None,
        "valid_frac_box": round(float(box_valid.mean()), 6) if box_valid.size else None,
        "valid_frac_movement": (
            round(float((movement == 1).mean()), 6) if movement is not None and movement.size else None
        ),
        "movement_mask_present": movement is not None,
    }
    return covered, summary


def _choose_start_frame(
    seed: int,
    file_id: str,
    render_frames: int,
    usable_frames: int,
) -> int:
    """Deterministic start frame for a full clip inside the usable interval.

    Clips are not steered towards valid regions: a base-rate sample that
    silently avoided invalid frames would understate how often reviewers meet
    them.
    """

    span = usable_frames - render_frames
    if span <= 0:
        return 0
    return int(_stable_unit(seed, "start", file_id) * (span + 1))


def _resolve(
    candidates: Iterable[Candidate],
    frame: pd.DataFrame,
    partners: dict[str, str],
    source_root: Path,
    seed: int,
    limit: int | None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Attach start frames and NPZ coverage; report every rejection."""

    by_id = frame.set_index("file_id")
    accepted: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    for candidate in candidates:
        if limit is not None and len(accepted) >= limit:
            break
        row = by_id.loc[candidate.file_id]
        signals = dict(candidate.signals)
        record = {
            "review_item_id": "",
            "clip_id": "",
            "file_id": candidate.file_id,
            "source_relbase": candidate.source_relbase,
            "partner_source_relbase": partners.get(candidate.file_id, ""),
            "sample_group": candidate.sample_group,
            "vendor": str(row.vendor),
            "label": str(row.label),
            "split": str(row.split),
            "activity_type": str(row.interaction_type),
            "selection_reason": candidate.selection_reason,
            "render_policy": candidate.render_policy,
        }
        if candidate.render_policy == "metadata_only":
            record["start_frame"] = 0
            signals["render_policy_reason"] = candidate.selection_reason
            record["signals_json"] = json.dumps(signals, sort_keys=True)
            accepted.append(record)
            continue

        fps = float(row.video_avg_fps) if pd.notna(row.video_avg_fps) else float("nan")
        if not np.isfinite(fps) or fps <= 0:
            rejected.append({**record, "reject_reason": "unusable_avg_frame_rate"})
            continue
        render_frames = int(round(CLIP_DURATION_S * fps))
        try:
            covered, npz_summary = _annotation_frames(source_root, candidate.source_relbase)
        except (OSError, ValueError, KeyError) as exc:
            rejected.append({**record, "reject_reason": f"npz_read_error:{type(exc).__name__}"})
            continue
        signals.update(npz_summary)
        video_frames = int(row.video_nb_frames)
        usable = min(video_frames, covered)
        if usable < render_frames:
            rejected.append(
                {
                    **record,
                    "reject_reason": (
                        f"usable_frames_{usable}_below_render_frames_{render_frames}"
                    ),
                }
            )
            continue
        start = (
            candidate.start_frame_override
            if candidate.start_frame_override is not None
            else _choose_start_frame(seed, candidate.file_id, render_frames, usable)
        )
        start = max(0, min(start, usable - render_frames))
        signals["render_frames"] = render_frames
        signals["avg_fps"] = round(fps, 6)
        signals["video_nb_frames"] = video_frames
        record["start_frame"] = start
        record["signals_json"] = json.dumps(signals, sort_keys=True)
        accepted.append(record)
    return accepted, rejected


def uniform_candidates(frame: pd.DataFrame, seed: int) -> list[Candidate]:
    """Uniform-random over renderable rows, ordered by a stable hash draw."""

    population = frame.loc[frame.renderable_headers].copy()
    population["draw"] = [
        _stable_unit(seed, "uniform", file_id) for file_id in population.file_id
    ]
    population = population.sort_values("draw", kind="mergesort")
    return [
        Candidate(
            file_id=row.file_id,
            source_relbase=row.source_relbase,
            sample_group="uniform_200",
            selection_reason="uniform random over renderable corpus rows",
            signals={"uniform_draw": round(float(row.draw), 9)},
        )
        for row in population.itertuples(index=False)
    ]


def _dev_signal_extremes(worktree: Path) -> list[tuple[str, str, dict[str, Any], int | None]]:
    """Session-1 dev extremes as (file_id, reason, signals, start_frame)."""

    import glob

    picks: list[tuple[str, str, dict[str, Any], int | None]] = []
    parquets = sorted(glob.glob(str(worktree / "outputs/harness/files/*/*/*/*.parquet")))
    if parquets:
        windows = pd.concat([pd.read_parquet(path) for path in parquets], ignore_index=True)
        per_file = windows.groupby("file_id").agg(
            valid_frac_all_min=("valid_frac_all", "min"),
            valid_frac_all_mean=("valid_frac_all", "mean"),
            wrist_speed_p90_max=("wrist_speed_p90", "max"),
            accel_max=("accel_mm_per_frame2", "max"),
            duration_mismatch_max=("duration_mismatch_s", "max"),
        )
        selectors = [
            ("wrist_speed_p90_max", False, "session1_dev_highest_wrist_speed_p90"),
            ("wrist_speed_p90_max", True, "session1_dev_lowest_wrist_speed_p90"),
            ("accel_max", False, "session1_dev_highest_provisional_translation_accel"),
            ("duration_mismatch_max", False, "session1_dev_largest_duration_mismatch"),
            ("valid_frac_all_mean", True, "session1_dev_lowest_mean_valid_frac_all"),
        ]
        for column, ascending, reason in selectors:
            series = per_file[column].dropna()
            if series.empty:
                continue
            file_id = series.idxmin() if ascending else series.idxmax()
            # Anchor the clip on the window that realizes the extreme so the
            # reviewer actually sees the frames that produced the number.
            subset = windows.loc[windows.file_id == file_id]
            metric = column.rsplit("_", 1)[0]
            base_column = {
                "wrist_speed_p90": "wrist_speed_p90",
                "accel": "accel_mm_per_frame2",
                "duration_mismatch": "duration_mismatch_s",
                "valid_frac_all": "valid_frac_all",
            }[metric]
            ordered = subset.dropna(subset=[base_column]).sort_values(
                base_column, ascending=ascending, kind="mergesort"
            )
            start = int(ordered.window_start_frame.iloc[0]) if len(ordered) else None
            picks.append(
                (
                    str(file_id),
                    reason,
                    {"session1_metric": column, "session1_value": float(series.loc[file_id])},
                    start,
                )
            )

    masks_path = worktree / "outputs/recon/validity_audit/per_file_masks.csv"
    runs_path = worktree / "outputs/recon/validity_audit/invalid_runs.csv"
    if masks_path.is_file():
        masks = pd.read_csv(masks_path)
        present = masks.loc[masks.present.astype(bool) & masks.frames.gt(0)]
        for mask_name in ("smplh", "box", "movement"):
            subset = present.loc[present["mask"] == mask_name]
            if subset.empty:
                continue
            worst = subset.loc[subset.invalid_frac.idxmax()]
            picks.append(
                (
                    str(worst.file_id),
                    f"session1_dev_highest_{mask_name}_invalid_frac",
                    {
                        "mask": mask_name,
                        "invalid_frac": float(worst.invalid_frac),
                        "max_invalid_run": float(worst.max_invalid_run),
                    },
                    None,
                )
            )
            fully = subset.loc[subset.invalid_frac.ge(1.0)]
            if not fully.empty:
                picks.append(
                    (
                        str(fully.iloc[0].file_id),
                        f"session1_dev_100pct_{mask_name}_invalid",
                        {"mask": mask_name, "invalid_frac": 1.0},
                        None,
                    )
                )
    if runs_path.is_file():
        runs = pd.read_csv(runs_path)
        for mask_name in ("smplh", "box", "movement"):
            subset = runs.loc[runs["mask"] == mask_name]
            if subset.empty:
                continue
            longest = subset.loc[subset.length_frames.idxmax()]
            # Start three seconds before the run so the transition into it is
            # visible rather than the clip opening mid-failure.
            start = max(0, int(longest.start_frame) - 90)
            picks.append(
                (
                    str(longest.file_id),
                    f"session1_dev_longest_{mask_name}_invalid_run",
                    {
                        "mask": mask_name,
                        "run_length_frames": int(longest.length_frames),
                        "run_start_frame": int(longest.start_frame),
                    },
                    start,
                )
            )
    return picks


def _corpus_extremes(frame: pd.DataFrame) -> list[tuple[str, str, dict[str, Any], int | None]]:
    """Corpus-wide header extremes that Session 1's dev-only scope could not see."""

    picks: list[tuple[str, str, dict[str, Any], int | None]] = []

    placeholder_groups = {
        "placeholder_261b_mp4_58b_wav": frame.mp4_size_bytes.eq(261) & frame.wav_size_bytes.eq(58),
        "placeholder_708b_mp4_58b_wav_v00": frame.mp4_size_bytes.eq(708) & frame.wav_size_bytes.eq(58),
        "placeholder_261b_mp4_with_full_wav": frame.mp4_size_bytes.eq(261) & frame.wav_size_bytes.gt(1000),
        "no_video_stream_multi_mb_mp4": frame.probe_status.eq("no_video_stream")
        & frame.mp4_size_bytes.gt(1_000_000),
        "mp4_container_unreadable_moov_missing": frame.probe_status.eq("ffprobe_error"),
    }
    for reason, selector in placeholder_groups.items():
        subset = frame.loc[selector]
        if subset.empty:
            continue
        row = subset.sort_values("file_id", kind="mergesort").iloc[0]
        picks.append(
            (
                str(row.file_id),
                reason,
                {
                    "mp4_size_bytes": int(row.mp4_size_bytes) if pd.notna(row.mp4_size_bytes) else None,
                    "wav_size_bytes": int(row.wav_size_bytes) if pd.notna(row.wav_size_bytes) else None,
                    "probe_status": str(row.probe_status),
                    "unrenderable": True,
                },
                None,
            )
        )

    silent = frame.loc[frame.wav_size_bytes.eq(58) & frame.probe_status.eq("ok") & frame.renderable_headers]
    if not silent.empty:
        row = silent.sort_values("file_id", kind="mergesort").iloc[0]
        picks.append(
            (
                str(row.file_id),
                "empty_58_byte_wav_with_readable_video",
                {"wav_size_bytes": 58, "probe_status": "ok"},
                None,
            )
        )

    renderable = frame.loc[frame.renderable_headers]
    spread = renderable.copy()
    spread["interaction_key"] = (
        spread.vendor_id + "|" + spread.session_id + "|" + spread.interaction_id
    )
    grouped = spread.groupby("interaction_key").observed_duration_s
    spread["member_spread_s"] = grouped.transform("max") - grouped.transform("min")
    spread = spread.loc[grouped.transform("size").eq(2)]
    if not spread.empty:
        row = spread.loc[spread.member_spread_s.idxmax()]
        picks.append(
            (
                str(row.file_id),
                "largest_within_interaction_member_duration_spread",
                {
                    "member_spread_s": round(float(row.member_spread_s), 3),
                    "observed_duration_s": round(float(row.observed_duration_s), 3),
                },
                None,
            )
        )

    rate = renderable.loc[(renderable.video_avg_fps - renderable.video_r_fps).abs().gt(0.01)]
    if not rate.empty:
        deltas = (rate.video_avg_fps - rate.video_r_fps).abs()
        row = rate.loc[deltas.idxmax()]
        picks.append(
            (
                str(row.file_id),
                "largest_avg_vs_container_frame_rate_disagreement",
                {
                    "video_r_frame_rate": str(row.video_r_frame_rate),
                    "video_avg_fps": round(float(row.video_avg_fps), 6),
                    "abs_delta_fps": round(float(deltas.loc[row.name]), 6),
                },
                None,
            )
        )

    for reason, ascending in (
        ("longest_single_participant_video", False),
        ("shortest_renderable_participant_video", True),
    ):
        ordered = renderable.sort_values(
            ["observed_duration_s", "file_id"], ascending=[ascending, True], kind="mergesort"
        )
        row = ordered.iloc[0]
        picks.append(
            (
                str(row.file_id),
                reason,
                {"observed_duration_s": round(float(row.observed_duration_s), 3)},
                None,
            )
        )

    raster_counts = renderable.groupby(["video_width", "video_height"]).file_id.count()
    for (width, height) in raster_counts.nsmallest(3).index:
        subset = renderable.loc[
            renderable.video_width.eq(width) & renderable.video_height.eq(height)
        ].sort_values("file_id", kind="mergesort")
        row = subset.iloc[0]
        picks.append(
            (
                str(row.file_id),
                f"rarest_raster_{int(width)}x{int(height)}",
                {
                    "video_width": int(width),
                    "video_height": int(height),
                    "corpus_files_with_this_raster": int(raster_counts.loc[(width, height)]),
                },
                None,
            )
        )
    return picks


def targeted_candidates(
    frame: pd.DataFrame, worktree: Path, seed: int, exclude: set[str]
) -> list[Candidate]:
    """Extremes first, then a stratified fill over vendor x label x activity."""

    extremes = _dev_signal_extremes(worktree) + _corpus_extremes(frame)
    known = set(frame.file_id)
    by_id = frame.set_index("file_id")
    candidates: list[Candidate] = []
    used: set[str] = set(exclude)
    for file_id, reason, signals, start in extremes:
        if file_id in used or file_id not in known:
            continue
        row = by_id.loc[file_id]
        used.add(file_id)
        candidates.append(
            Candidate(
                file_id=file_id,
                source_relbase=str(row.source_relbase),
                sample_group="targeted_100",
                selection_reason=f"extreme:{reason}",
                signals={**signals, "targeted_role": "extreme"},
                render_policy=(
                    "render" if bool(row.renderable_headers) else "metadata_only"
                ),
                start_frame_override=start,
            )
        )

    remaining = TARGETED_TOTAL - len(candidates)
    population = frame.loc[frame.renderable_headers & ~frame.file_id.isin(used)].copy()
    population["stratum"] = (
        population.vendor + "|" + population.label + "|" + population.interaction_type
    )
    population["draw"] = [
        _stable_unit(seed, "targeted", file_id) for file_id in population.file_id
    ]
    population = population.sort_values(["stratum", "draw"], kind="mergesort")
    # Round-robin one file per stratum per pass, so every occupied cell is
    # represented before any cell receives a second clip.
    per_stratum = {
        stratum: list(group.itertuples(index=False))
        for stratum, group in population.groupby("stratum", sort=True)
    }
    order = sorted(per_stratum)
    fill: list[Any] = []
    depth = 0
    while len(fill) < remaining and any(len(per_stratum[s]) > depth for s in order):
        for stratum in order:
            if len(fill) >= remaining:
                break
            if len(per_stratum[stratum]) > depth:
                fill.append(per_stratum[stratum][depth])
        depth += 1
    for row in fill:
        candidates.append(
            Candidate(
                file_id=row.file_id,
                source_relbase=row.source_relbase,
                sample_group="targeted_100",
                selection_reason=f"stratified_fill:{row.stratum}",
                signals={"targeted_role": "stratified_fill", "stratum": row.stratum},
            )
        )
    return candidates


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/session2_review.yaml"))
    parser.add_argument(
        "--inventory",
        type=Path,
        default=Path("outputs/02_inventory/summary/inventory_joined.parquet"),
    )
    parser.add_argument("--report", type=Path, default=Path("outputs/session2/review_sampling"))
    args = parser.parse_args()

    worktree = _worktree()
    config = yaml.safe_load((worktree / args.config).read_text(encoding="utf-8"))
    sampling = config["sampling"]
    source_root = (worktree / str(config["source_root"])).resolve()
    manifest_path = worktree / str(config["outputs"]["manifest"]).lstrip("./")
    pass1_path = worktree / str(config["outputs"]["pass1_manifest"]).lstrip("./")
    report_dir = worktree / args.report
    report_dir.mkdir(parents=True, exist_ok=True)

    uniform_seed = int(sampling["uniform"]["seed"])
    targeted_seed = int(sampling["targeted"]["seed"])
    pass1_seed = int(sampling["pass1"]["seed"])
    pass1_count = int(sampling["pass1"]["count"])
    duplicate_fraction = float(sampling["pass2"]["duplicate_fraction"])
    duplicate_seed = int(sampling["pass2"]["duplicate_seed"])

    frame = load_population(worktree / args.inventory)
    partners = _interaction_partners(frame)

    uniform_accepted, uniform_rejected = _resolve(
        uniform_candidates(frame, uniform_seed),
        frame,
        partners,
        source_root,
        uniform_seed,
        limit=UNIFORM_TOTAL,
    )
    uniform_ids = {record["file_id"] for record in uniform_accepted}
    targeted_accepted, targeted_rejected = _resolve(
        targeted_candidates(frame, worktree, targeted_seed, exclude=uniform_ids),
        frame,
        partners,
        source_root,
        targeted_seed,
        limit=TARGETED_TOTAL,
    )

    records = uniform_accepted + targeted_accepted
    for index, record in enumerate(records):
        record["clip_id"] = (
            f"{record['sample_group']}_{index:03d}_{record['file_id']}_f{int(record['start_frame']):07d}"
        )
        record["review_item_id"] = f"item_{index:03d}"

    manifest = pd.DataFrame(records, columns=MANIFEST_COLUMNS)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(manifest_path, index=False)

    # Pass 1 is a seeded subset of the same clips, mixing both groups in the
    # realized proportion so the exploratory pass is not accidentally an
    # extremes-only review.
    pass1_draw = sorted(
        records, key=lambda record: _stable_unit(pass1_seed, "pass1", record["review_item_id"])
    )[:pass1_count]
    pass1 = pd.DataFrame(pass1_draw, columns=MANIFEST_COLUMNS)
    pass1.to_csv(pass1_path, index=False)

    duplicate_count = int(round(duplicate_fraction * len(records)))
    duplicates = sorted(
        records, key=lambda record: _stable_unit(duplicate_seed, "duplicate", record["review_item_id"])
    )[:duplicate_count]
    pass2_records = [dict(record) for record in records]
    for index, record in enumerate(duplicates):
        copy = dict(record)
        copy["review_item_id"] = f"dup_{index:03d}"
        copy["selection_reason"] = f"agreement_duplicate_of:{record['review_item_id']}"
        pass2_records.append(copy)
    pass2_path = manifest_path.with_name("session2_review_pass2.csv")
    pd.DataFrame(pass2_records, columns=MANIFEST_COLUMNS).to_csv(pass2_path, index=False)
    # The reviewer-facing duplicate map is written outside the gallery so
    # rater-agreement scoring is possible without revealing pairs during review.
    (report_dir / "pass2_duplicate_map.json").write_text(
        json.dumps(
            {f"dup_{index:03d}": record["review_item_id"] for index, record in enumerate(duplicates)},
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    rejected = pd.DataFrame(uniform_rejected + targeted_rejected)
    if not rejected.empty:
        rejected.to_csv(report_dir / "rejected_candidates.csv", index=False)

    def realized(subset: pd.DataFrame, keys: list[str]) -> list[dict[str, Any]]:
        if subset.empty:
            return []
        counts = subset.groupby(keys, dropna=False).size().reset_index(name="clips")
        return counts.to_dict(orient="records")

    summary = {
        "clip_duration_s": CLIP_DURATION_S,
        "population_definition": (
            "filelist row with all four modalities present locally, ffprobe status ok, "
            "a video stream, observed duration >= 10 s, and a parseable frame count"
        ),
        "population_rows": int(frame.renderable_headers.sum()),
        "population_excluded_rows": int((~frame.renderable_headers).sum()),
        "seeds": {
            "uniform": uniform_seed,
            "targeted": targeted_seed,
            "pass1": pass1_seed,
            "pass2_duplicates": duplicate_seed,
        },
        "counts": {
            "uniform": len(uniform_accepted),
            "targeted": len(targeted_accepted),
            "targeted_extremes": int(
                sum(1 for record in targeted_accepted if record["selection_reason"].startswith("extreme:"))
            ),
            "targeted_stratified_fill": int(
                sum(
                    1
                    for record in targeted_accepted
                    if record["selection_reason"].startswith("stratified_fill:")
                )
            ),
            "metadata_only": int(
                sum(1 for record in records if record["render_policy"] == "metadata_only")
            ),
            "total_clips": len(records),
            "pass1_items": len(pass1_draw),
            "pass2_items": len(pass2_records),
            "pass2_duplicate_items": duplicate_count,
            "rejected_candidates": len(rejected),
        },
        "realized_uniform_by_vendor_label_split": realized(
            manifest.loc[manifest.sample_group.eq("uniform_200")], ["vendor", "label", "split"]
        ),
        "realized_uniform_by_activity": realized(
            manifest.loc[manifest.sample_group.eq("uniform_200")], ["activity_type"]
        ),
        "realized_targeted_by_vendor_label_activity": realized(
            manifest.loc[manifest.sample_group.eq("targeted_100")],
            ["vendor", "label", "activity_type"],
        ),
        "targeted_extreme_reasons": sorted(
            record["selection_reason"]
            for record in targeted_accepted
            if record["selection_reason"].startswith("extreme:")
        ),
        "rejected_reason_counts": (
            rejected.reject_reason.value_counts().to_dict() if not rejected.empty else {}
        ),
        "manifest": str(manifest_path.relative_to(worktree)),
        "pass1_manifest": str(pass1_path.relative_to(worktree)),
        "pass2_manifest": str(pass2_path.relative_to(worktree)),
        **_git_provenance(worktree),
    }
    (report_dir / "sampling_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(summary["counts"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

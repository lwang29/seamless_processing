"""Stage 7: package the subset for a downstream project.

The manifest stages produce the *decision* — which frames of which files are
accepted. This stage produces the **handover**: the same rows with every path a
loader needs already resolved, split into two tiers with an honest statement of
what each one is worth, plus a dataset card that can be read on its own.

Two tiers, because they answer different questions:

``clips_verified.csv``
    Every clip is in a file a reviewer looked at individually and accepted.
    This is the subset the PI asked for. It is small because review is the
    bottleneck, not because the data is.

``clips_candidate.csv``
    Passed all eighteen automated gates and has *not yet* been reviewed. Files
    that were reviewed and rejected are excluded, so this is strictly better
    than raw gate output. Its precision is estimated from the reviewed sample
    and reported in the card and in ``summary.json`` — it is an estimate with a
    confidence interval, not a promise about any individual clip.

Nothing here copies, moves or modifies the release. Both files are lists of
frame ranges; :mod:`seamless_curation.dataset` turns a row into arrays.
"""

from __future__ import annotations

import json
from math import sqrt
from pathlib import Path
from typing import Any

import pandas as pd

from .config import RunConfig
from .dataset import SOURCE_SUFFIXES, load_manifest
from .review_store import VerdictStore

#: Columns every exported row carries, in this order. Measures follow.
IDENTITY_COLUMNS: tuple[str, ...] = (
    "clip_id", "review_item_id", "file_id", "source_relbase", "vendor", "label", "split",
    "session_id", "participant_id", "interaction_id", "interaction_type",
    "start_frame", "end_frame", "n_frames", "start_s", "window_seconds", "fps",
)

PATH_COLUMNS: tuple[str, ...] = ("pose_path", "audio_path", "video_path", "annotation_path")

MEASURE_COLUMNS: tuple[str, ...] = (
    "speech_seconds", "gesture_frac_speech", "speech_segments_covered", "episode_count_speech",
    "wrist_excursion_p90_mm", "elbow_excursion_p90_mm", "gesture_speech_ratio",
    "posture_spread_mm", "wrist_height_p75_mm", "hands_together_frac", "arm_abduction_p75_deg",
    "sync_r", "sync_lag_s", "consistency_r", "smplh_valid_frac", "smplh_longest_invalid_s",
    "hand_frozen_frac", "clip_score",
)

PROVENANCE_COLUMNS: tuple[str, ...] = ("tier", "reviewed", "reviewer", "verdict_source", "review_evidence")


def wilson_interval(successes: int, total: int, z: float = 1.96) -> tuple[float, float]:
    """95% Wilson score interval. Correct at small n, unlike the normal one."""

    if total == 0:
        return (0.0, 1.0)
    phat = successes / total
    denominator = 1 + z * z / total
    centre = (phat + z * z / (2 * total)) / denominator
    margin = z * sqrt(phat * (1 - phat) / total + z * z / (4 * total * total)) / denominator
    return (max(0.0, centre - margin), min(1.0, centre + margin))


def _add_paths(frame: pd.DataFrame, source_root: Path) -> pd.DataFrame:
    """Resolve the four source files per row, relative to ``source_root``."""

    base = frame["source_relbase"].astype(str)
    for column, (_, suffix) in zip(PATH_COLUMNS, SOURCE_SUFFIXES.items()):
        frame[column] = base + suffix
    return frame


def _ensure_fps(frame: pd.DataFrame, config: RunConfig) -> pd.DataFrame:
    if "fps" in frame.columns and frame["fps"].notna().all():
        return frame
    population = pd.read_parquet(config.population_path)[["file_id", "nominal_fps"]]
    frame = frame.merge(population, on="file_id", how="left")
    frame["fps"] = frame.get("fps").fillna(frame["nominal_fps"]) if "fps" in frame else frame["nominal_fps"]
    return frame.drop(columns=["nominal_fps"])


def _shape(frame: pd.DataFrame, tier: str, reviewed: bool) -> pd.DataFrame:
    frame = frame.copy()
    frame["tier"] = tier
    frame["reviewed"] = reviewed
    frame["n_frames"] = frame["end_frame"].astype(int) - frame["start_frame"].astype(int)
    for column in PROVENANCE_COLUMNS:
        if column not in frame.columns:
            frame[column] = ""
    order = [
        column
        for column in IDENTITY_COLUMNS + PATH_COLUMNS + PROVENANCE_COLUMNS + MEASURE_COLUMNS
        if column in frame.columns
    ]
    return frame[order].sort_values(["file_id", "start_frame"]).reset_index(drop=True)


def _precision_estimate(resolved: pd.DataFrame) -> dict[str, Any]:
    """How often a gate-passing item survives human review.

    Every reviewed item passed the gates, so this is exactly the precision of
    the automated filter. The reviewed items are the queue prefix, and the queue
    is a stratified round-robin over participants, so the sample is
    representative of the pool by construction rather than by assumption.
    """

    human = resolved.loc[~resolved["verdict_source"].astype(str).str.startswith("model")]
    total = int(len(human))
    accepted = int((human["verdict"] == "accept").sum())
    low, high = wilson_interval(accepted, total)
    return {
        "reviewed_by_human": total,
        "accepted": accepted,
        "estimate": round(accepted / total, 3) if total else None,
        "ci95": [round(low, 3), round(high, 3)],
        "note": "share of gate-passing items a human accepted; Wilson 95% interval",
    }


def build_export(config: RunConfig) -> dict[str, Any]:
    """Write both tiers, the dataset card and a summary. Returns the summary."""

    out = config.output_root / "export"
    out.mkdir(parents=True, exist_ok=True)

    candidates = pd.read_parquet(config.candidates_path)
    resolved = VerdictStore(config.verdict_log).resolve()

    verified = (
        load_manifest(config.accepted_clips_path)
        if config.accepted_clips_path.exists()
        else pd.DataFrame()
    )
    if not verified.empty:
        verified = _ensure_fps(verified, config)
        verified = _add_paths(_shape(verified, "verified", True), config.source_root)
    verified.to_csv(out / "clips_verified.csv", index=False)

    judged = set(resolved["review_item_id"]) if not resolved.empty else set()
    pool = candidates.loc[~candidates["review_item_id"].isin(judged)].copy()
    pool = _ensure_fps(pool, config)
    pool = _add_paths(_shape(pool, "candidate", False), config.source_root)
    pool.to_csv(out / "clips_candidate.csv", index=False)

    precision = _precision_estimate(resolved) if not resolved.empty else {}
    hours = lambda f: round(float(f["window_seconds"].sum() / 3600), 2) if len(f) else 0.0
    summary: dict[str, Any] = {
        "run_id": config.run_id,
        "source_root": str(config.source_root),
        "verified": {
            "clips": int(len(verified)), "hours": hours(verified),
            "files": int(verified["file_id"].nunique()) if len(verified) else 0,
            "participants": int(
                (verified["vendor"].astype(str) + ":" + verified["participant_id"].astype(str)).nunique()
            ) if len(verified) else 0,
            "path": str(out / "clips_verified.csv"),
        },
        "candidate": {
            "clips": int(len(pool)), "hours": hours(pool),
            "files": int(pool["file_id"].nunique()) if len(pool) else 0,
            "participants": int(
                (pool["vendor"].astype(str) + ":" + pool["participant_id"].astype(str)).nunique()
            ) if len(pool) else 0,
            "path": str(out / "clips_candidate.csv"),
            "estimated_precision": precision,
        },
        "reviewed_items": int(len(resolved)),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (out / "DATASET.md").write_text(_dataset_card(config, summary, verified, pool), encoding="utf-8")
    return summary


def _vendor_table(frame: pd.DataFrame) -> str:
    if not len(frame):
        return "_(empty)_\n"
    grouped = (
        frame.groupby("vendor")
        .agg(clips=("clip_id", "size"), hours=("window_seconds", lambda s: round(s.sum() / 3600, 2)))
        .reset_index()
    )
    lines = ["| vendor | clips | hours |", "|---|---:|---:|"]
    lines += [f"| {r.vendor} | {r.clips} | {r.hours} |" for r in grouped.itertuples()]
    return "\n".join(lines) + "\n"


def _dataset_card(
    config: RunConfig, summary: dict[str, Any], verified: pd.DataFrame, pool: pd.DataFrame
) -> str:
    precision = summary["candidate"].get("estimated_precision") or {}
    estimate = precision.get("estimate")
    ci = precision.get("ci95") or [None, None]
    pct = f"{estimate:.0%}" if estimate is not None else "not yet estimated"
    band = f"{ci[0]:.0%}–{ci[1]:.0%}" if ci[0] is not None else "n/a"
    v, c = summary["verified"], summary["candidate"]
    return f"""# Seamless Interaction — co-speech upper-body subset (`{config.run_id}`)

Two CSV files. Each row is **a frame range of one participant recording**, not a
copy of it: the release at `{config.source_root}` is never modified, and every
row points back into it. Load a row with
`seamless_curation.dataset.load_clip`, which reads only the frames the row names.

| file | clips | hours | files | participants | what it is |
|---|---:|---:|---:|---:|---|
| `clips_verified.csv` | {v['clips']} | {v['hours']} | {v['files']} | {v['participants']} | a reviewer watched and accepted each one |
| `clips_candidate.csv` | {c['clips']} | {c['hours']} | {c['files']} | {c['participants']} | passed all automated gates, not yet reviewed |

## Which one to use

**Start with `clips_verified.csv`.** Every clip in it belongs to a file a human
inspected individually against `docs/review_rubric.md` — valid upper-body SMPL-H,
arms and hands genuinely moving during speech, motion natural rather than
tracking noise, gesture synchronised with speech.

**Use `clips_candidate.csv` when you need volume more than you need certainty.**
It has passed the same eighteen automated gates and excludes every file a
reviewer rejected, but nobody has looked at these individually. Measured against
{precision.get('reviewed_by_human', 0)} human-reviewed items drawn from the same
pool, **about {pct} of them would survive review** (Wilson 95% interval {band}).
The residual failure is overwhelmingly one mode: a participant whose hands move
enough to pass the thresholds but who is not really gesturing — hands clasped at
the waist, or repeatedly adjusting a hat. That is noise in the training signal,
not corrupt data: the SMPL-H parameters are still valid and still synchronised.

For preliminary training runs where more data wins, the candidate tier is the
right choice. For anything reported as a curated subset, use the verified tier.

## Columns

| group | columns |
|---|---|
| identity | `clip_id`, `review_item_id`, `file_id`, `vendor`, `label`, `split`, `session_id`, `participant_id`, `interaction_id`, `interaction_type` |
| frames | `start_frame`, `end_frame`, `n_frames`, `start_s`, `window_seconds`, `fps` |
| paths | `pose_path`, `audio_path`, `video_path`, `annotation_path` — all **relative to `source_root`** |
| provenance | `tier`, `reviewed`, `reviewer`, `verdict_source`, `review_evidence` |
| measures | the gate measurements for that window, so you can re-filter without re-running anything |

`split` is the **release's** train/dev/test split and is preserved unchanged.
`participant_id` is unique only within a vendor; use `vendor + ":" + participant_id`
as the participant key, which is what the participant counts above do.

## Vendors

Verified tier:

{_vendor_table(verified)}
Candidate tier:

{_vendor_table(pool)}
## What was measured

All motion is expressed in a **torso frame** (origin at the shoulder midpoint,
axes from the shoulder line and the pelvis-to-neck direction), so swaying,
turning and stepping cannot be mistaken for gesture. Gesture is measured **only
inside the participant's own VAD segments**. A frame counts as gesturing only if
the wrist has both speed (≥60 mm/s) and travel (≥35 mm within 0.5 s), which is
what stops jitter in place from qualifying.

Every activity and posture measure is the **maximum over the two hands**, so
one-handed gesturing is accepted (rubric decision 2026-09-21).

Full rationale for each of the eighteen gates, including why no
jitter-*magnitude* gate is used, is in `src/seamless_curation/gates.py` and
`reports/17_cospeech_gesture.md`.

## Caveats worth knowing before you train

1. **The verified tier is limited by review effort, not data.** The candidate
   pool holds {c['hours']} more hours at roughly the quality stated above.
2. **Windows from one file overlap in time only if they do not overlap at all** —
   selection vetoes overlap, so clips from the same file are disjoint frame
   ranges. Concatenating them is safe; they are not contiguous.
3. **Per-participant caps are applied** (≤8 clips per file, ≤12 files per
   participant), so no single talkative participant dominates.
4. **`charades` interactions and four raster formats are excluded** upstream, at
   the population stage, for reasons in `src/seamless_curation/corpus.py`.
5. **Audio is that participant's own channel**, 48 kHz mono float32, already
   separated in the release. There is no partner voice to remove.
"""

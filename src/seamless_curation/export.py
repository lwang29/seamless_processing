"""Stage 7: package the subset for a downstream project.

The decision stages produce *which frames are accepted*. This stage produces the
**handover**: the same rows with every source path resolved, plus a dataset card
that can be read on its own by someone who will never run this pipeline.

Two files, and the difference between them is how the decision was made:

``clips_accepted.csv``
    The production subset. Every row passed tier-1 gates and tier-2
    qualification. Produced entirely by measurement, with no human or model
    judgement anywhere in its causal path, which is what makes it reproducible
    from the source tree and a config file alone.

``clips_reviewed.csv``
    The subset a reviewer also looked at and accepted. A *development* artefact:
    it is the labelled set the automated decision is validated against, and it
    carries a stronger per-clip guarantee for anyone who wants one. It is small,
    and it will stay small, because review does not scale.

The card reports the automated decision's measured agreement with the reviewed
set rather than asserting a quality level, so a downstream reader can see what
the automation is worth instead of taking it on trust.

Nothing here copies, moves or modifies the release.
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
    "episode_median_s", "wrist_excursion_p90_mm", "elbow_excursion_p90_mm", "wrist_range_mm",
    "gesture_speech_ratio", "posture_spread_mm", "wrist_height_p75_mm", "hands_together_frac",
    "arm_abduction_p75_deg", "arm_speed_speech_p50_mm_s", "articulation_ratio",
    "step_cosine_p50", "sync_r", "sync_lag_s", "consistency_r", "smplh_valid_frac",
    "smplh_longest_invalid_s", "hand_frozen_frac", "clip_score",
)

#: Tier-2 scores, carried so a row can be re-thresholded without re-running.
SCORE_COLUMNS: tuple[str, ...] = (
    "gesture_quality", "dim_posture", "dim_persistence", "dim_vigour", "dim_integrity",
    "exclusion_flags",
)

PROVENANCE_COLUMNS: tuple[str, ...] = (
    "tier", "decision_source", "reviewed", "reviewer", "verdict_source", "review_evidence",
)


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
        for column in IDENTITY_COLUMNS + PATH_COLUMNS + PROVENANCE_COLUMNS
        + MEASURE_COLUMNS + SCORE_COLUMNS
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

    def load(path: Path) -> pd.DataFrame:
        return load_manifest(path) if path.exists() else pd.DataFrame()

    accepted = load(config.accepted_clips_path)
    if len(accepted):
        accepted = _add_paths(_shape(accepted, "accepted", False), config.source_root)
    accepted.to_csv(out / "clips_accepted.csv", index=False)

    # The reviewed subset is written from MANIFEST_COLUMNS, which predates
    # tier 2 and so lacks the score columns. Join them back on so both files
    # have one schema -- DATASET.md documents one, and a downstream filter
    # written against the accepted tier must not break on the reviewed one.
    reviewed = load(config.reviewed_clips_path)
    if len(reviewed) and config.qualified_path.exists():
        extra = [c for c in MEASURE_COLUMNS + SCORE_COLUMNS if c not in reviewed.columns]
        if extra:
            scores = pd.read_parquet(config.qualified_path, columns=["clip_id", *extra])
            reviewed = reviewed.merge(scores, on="clip_id", how="left")
    if len(reviewed):
        reviewed = _add_paths(_shape(reviewed, "reviewed", True), config.source_root)
    reviewed.to_csv(out / "clips_reviewed.csv", index=False)

    hours = lambda f: round(float(f["window_seconds"].sum() / 3600), 2) if len(f) else 0.0
    people = lambda f: int(
        (f["vendor"].astype(str) + ":" + f["participant_id"].astype(str)).nunique()
    ) if len(f) else 0

    agreement = {}
    summary_path = config.output_root / "manifest_summary.json"
    if summary_path.exists():
        try:
            agreement = json.loads(summary_path.read_text(encoding="utf-8")).get(
                "agreement_with_review", {}
            )
        except (OSError, ValueError):
            agreement = {}

    summary: dict[str, Any] = {
        "run_id": config.run_id,
        "source_root": str(config.source_root),
        "decision": "automated",
        "accepted": {
            "clips": int(len(accepted)), "hours": hours(accepted),
            "files": int(accepted["file_id"].nunique()) if len(accepted) else 0,
            "participants": people(accepted),
            "path": str(out / "clips_accepted.csv"),
        },
        "reviewed": {
            "clips": int(len(reviewed)), "hours": hours(reviewed),
            "files": int(reviewed["file_id"].nunique()) if len(reviewed) else 0,
            "participants": people(reviewed),
            "path": str(out / "clips_reviewed.csv"),
        },
        "agreement_with_review": agreement,
        "qualifiers": config.qualifiers.as_dict(),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (out / "DATASET.md").write_text(
        _dataset_card(config, summary, accepted, reviewed), encoding="utf-8"
    )
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


def _source_table(frame: pd.DataFrame) -> str:
    """Who actually reviewed the rows of the reviewed subset."""

    if not len(frame) or "verdict_source" not in frame.columns:
        return "_(no reviewed rows)_\n"
    counts = frame["verdict_source"].value_counts()
    lines = ["| verdict source | clips | share |", "|---|---:|---:|"]
    for source, n in counts.items():
        lines.append(f"| `{source}` | {n} | {n / len(frame):.0%} |")
    return "\n".join(lines) + "\n"


def _dataset_card(
    config: RunConfig, summary: dict[str, Any], accepted: pd.DataFrame, reviewed: pd.DataFrame
) -> str:
    a, r = summary["accepted"], summary["reviewed"]
    human = (summary.get("agreement_with_review") or {}).get("human") or {}
    if human:
        agreement = (
            f"Measured against {human['n']} independently reviewed files, the automated "
            f"decision has **precision {human['precision']:.3f}** (of the files it accepts, "
            f"this share were also accepted by a human reviewer) and **recall "
            f"{human['recall']:.3f}** (of the files a human accepted, this share it also "
            f"accepts). It catches {human['true_reject']} of "
            f"{human['true_reject'] + human['false_accept']} files the reviewer rejected."
        )
    else:
        agreement = "No reviewed labels are present in this run, so no agreement is reported."

    q = config.qualifiers
    return f"""# Seamless Interaction - co-speech upper-body subset (`{config.run_id}`)

Two CSV files. Each row is **a frame range of one participant recording**, not a
copy of it: the release at `{config.source_root}` is never modified, and every
row points back into it. Load a row with
`seamless_curation.dataset.load_clip`, which reads only the frames the row names.

| file | clips | hours | files | participants | how it was decided |
|---|---:|---:|---:|---:|---|
| `clips_accepted.csv` | {a['clips']} | {a['hours']} | {a['files']} | {a['participants']} | fully automated |
| `clips_reviewed.csv` | {r['clips']} | {r['hours']} | {r['files']} | {r['participants']} | automated, and a reviewer also accepted it |

## Which one to use

**Use `clips_accepted.csv`.** It is the production subset and it is what the
pipeline is for. Every row passed eighteen tier-1 eligibility gates and then
nine tier-2 disqualifiers plus a composite gesture-quality threshold. No human
or model judgement is anywhere in its causal path, so it is reproducible from
the source tree and a config file.

`clips_reviewed.csv` is a development artefact: the labelled set the automated
decision was calibrated and validated against.

**Most of its reviewers were models, not people** — see the `verdict_source`
column, and `review_evidence == "card+video"` for the rows where a person
watched the clip with sound. Do not read the file as human sign-off.

{_source_table(reviewed)}

{agreement}

## How a clip earned its place

Motion is measured in a **torso frame** (origin at the shoulder midpoint, axes
from the shoulder line and the pelvis-to-neck direction), so swaying, turning
and stepping cannot be counted as gesture. Gesture is measured **only inside the
participant's own voice-activity segments**. A frame counts as gesturing only if
the wrist has both speed (>=60 mm/s) and travel (>=35 mm within 0.5 s).

Tier 2 then applies nine disqualifiers, each naming a pathology rather than a
degree, and a composite score:

| disqualifier | threshold | what it excludes |
|---|---:|---|
| `speech_seconds` | >= {q.min_speech_seconds:g} s | too little speech to judge |
| `gesture_frac_speech` | >= {q.min_gesture_frac_speech:g} | hands static while speaking |
| `wrist_height_p75_mm` | >= {q.min_wrist_height_p75_mm:g} mm | hands parked in the lap or at the sides |
| `episode_count_speech` | >= {q.min_episodes_speech:g} | a single movement |
| `episode_median_s` | >= {q.min_episode_median_s:g} s | a string of twitches |
| `speech_segments_covered` | >= {q.min_speech_segments_covered:g} | gesture not sustained across utterances |
| `articulation_ratio` | >= {q.min_articulation_ratio:g} | motion that is mostly whole-body |
| `step_cosine_p50` | >= {q.min_step_cosine_p50:g} | tracker noise (direction, not magnitude) |
| `consistency_r` | >= {q.min_consistency_r:g} | the two measurement channels disagree |

The composite score combines four dimensions - posture, persistence, vigour and
integrity - and must reach **{q.min_gesture_quality:g}**. It is a weighted mean, not a
conjunction, so strength on several dimensions compensates for modesty on one.
That is deliberate: it is what keeps a restrained but genuine gesturer in the
set. Peak wrist excursion is **excluded** from the score, because it separated
the labelled set backwards - it rewards one big isolated adjustment.

Every row carries its own `gesture_quality`, the four `dim_*` scores and a
`exclusion_flags` string, so any decision can be traced and any threshold re-applied
without re-running the pipeline:

```python
stricter = manifest[manifest.gesture_quality > 0.6]
```

## Columns

| group | columns |
|---|---|
| identity | `clip_id`, `review_item_id`, `file_id`, `vendor`, `label`, `split`, `session_id`, `participant_id`, `interaction_id`, `interaction_type` |
| frames | `start_frame`, `end_frame`, `n_frames`, `start_s`, `window_seconds`, `fps` |
| paths | `pose_path`, `audio_path`, `video_path`, `annotation_path` - all **relative to `source_root`** |
| provenance | `tier`, `decision_source`, `reviewed` |
| measures | every gate measurement for that window |
| scores | `gesture_quality`, `dim_posture`, `dim_persistence`, `dim_vigour`, `dim_integrity`, `exclusion_flags` |

`split` is the **release's** train/dev/test split, preserved unchanged.
`participant_id` is unique only within a vendor; use `vendor + ":" + participant_id`.

## Vendors

Accepted tier:

{_vendor_table(accepted)}
Reviewed tier:

{_vendor_table(reviewed)}
## Caveats worth knowing before you train

1. **Clips from one file are disjoint frame ranges, not contiguous.** Selection
   vetoes overlap. Concatenating them is safe; they are not continuous.
2. **Per-participant caps are applied** (<=8 clips per file, <=12 files per
   participant), so no single talkative participant dominates.
3. **Split on participant, not on clip**, if you are holding data out: one
   participant appears in several files.
4. **`charades` interactions and four raster formats are excluded** upstream.
5. **Audio is that participant's own channel**, 48 kHz mono float32, already
   separated in the release. There is no partner voice to remove.
6. **The thresholds are calibrated, not derived.** They come from agreement with
   a labelled sample; they are documented in `docs/pipeline.md` and are meant to
   be moved with evidence, not treated as physical constants.
"""

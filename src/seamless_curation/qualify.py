"""Stage 5: decide, automatically, whether a clip is co-speech gesture data.

This replaces manual review in the production pipeline. Review used to be the
final arbiter; it is now a *development* instrument (``render`` / ``review`` /
``import-verdicts``) whose only production role is supplying the labelled set
this module is calibrated and validated against.

Why a second tier at all
------------------------
``gates.py`` answers "is this window eligible" — is the tracking usable, is
there speech, is there motion that is not jitter. It is deliberately permissive,
because a gate that is strict on amplitude is arithmetically a gate on how much
the participant gestured, and 106 of 135 candidate quality signals were measured
to be confounded with gesture activity at |rho| up to 0.903.

What a reviewer added on top was a *judgement of degree*: these hands are up and
working, those hands are technically moving but parked in a lap. Measured on 100
human-labelled items that had all already passed every gate, that judgement is
predictable from measurements the scan already produces:

======================================  =====  ==============================
measure                                  AUC    accept vs reject (medians)
======================================  =====  ==============================
``wrist_height_p75_mm``                  0.85   -87 vs -194 mm
``step_cosine_p50``                      0.81   0.66 vs 0.39
``arm_speed_p50_mm_s``                   0.75   --
``wrist_range_mm``                       0.75   747 vs 675 mm
``arm_abduction_p75_deg``                0.74   31.6 vs 26.4 deg
======================================  =====  ==============================

So the second tier is not new physics. It is the same measurements read at a
threshold that the gates could not use, because the gates run before selection
and have to stay wide enough not to delete subtle gesturers wholesale.

Shape of the decision
---------------------
**Not** a conjunction of tight thresholds. ANDing eight strict clauses multiplies
the false-negative rate, and the PI's brief is explicit that genuine but subtle
co-speech gesture must survive. Instead:

1. **Disqualifiers** — a small set of hard clauses, each naming a *pathology*
   rather than a degree: motion that is directionally incoherent (tracker), motion
   that is mostly whole-body translation, motion that is one transient event,
   hands that are simply not moving while the person speaks. Failing any one is
   fatal and no amount of strength elsewhere compensates, because these are not
   "less gesture" — they are "not gesture".

2. **A composite quality score** — four dimensions, each a mean of piecewise-linear
   ramps in physical units, combined with weights. Strength on several dimensions
   compensates for modesty on one. This is the false-negative safeguard: a
   restrained speaker who nonetheless keeps their hands up, works them steadily
   through every utterance, and whose motion is cleanly coherent will clear the
   bar without ever producing a large movement.

One measure is deliberately **excluded** from the score: ``wrist_excursion_p90_mm``
separates the labelled set *backwards* (rejects median 339 mm vs accepts 319 mm).
Peak excursion rewards exactly the failure the PI named — one big isolated
adjustment, the participant who twice fixes his beanie and scores in the top
decile. Peak amplitude is not evidence of gesturing and is not treated as such.

Everything is reported, not just the verdict: every sub-score, every clause, and
an ``exclusion_flags`` string, so a bad decision can be traced to the clause that made it
and a threshold can be moved with evidence.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Qualifiers:
    """Tier-2 cut points. Calibrated in ``reports/18_automated_qualification.md``."""

    # ---- disqualifiers: pathologies, not degrees ---------------------------
    #: Median cosine between successive 2D displacement steps. Detector noise is
    #: antiparallel (-0.9 and below); real motion is positive. The tier-1 gate
    #: sits at -0.30, which the pool's 10th percentile (0.08) never reaches, so
    #: it never binds. This is the same measurement used where it discriminates.
    min_step_cosine_p50: float = 0.25
    #: Agreement between the SMPL-H arm and the released 2D arm. Two independent
    #: measurements of one limb; disagreement means at least one is inventing.
    min_consistency_r: float = 0.70
    #: Share of speaking frames on which the arms are active. This is the PI's
    #: criterion stated directly: hands still while the person talks.
    min_gesture_frac_speech: float = 0.50
    #: Height of the higher wrist above the shoulder midpoint, p75 over speech.
    #: Hands at the sides sit near -550 mm, clasped at the waist near -350 mm.
    #: Generous on purpose: this is a floor against parked hands, and the score
    #: carries the judgement of how well-placed the hands actually are.
    min_wrist_height_p75_mm: float = -100000.0
    #: Arm speed during speech against torso translation speed. Measurement is
    #: already in a torso frame, so global motion is mostly removed before this
    #: sees it; the ratio is the backstop for what survives that.
    min_articulation_ratio: float = 2.5
    #: Distinct gesture episodes overlapping speech.
    min_episodes_speech: int = 3
    #: Median episode duration. A sequence of 0.3 s twitches is not gesturing
    #: even when there are many of them.
    min_episode_median_s: float = 0.55
    #: Share of the window's utterances containing a gesture episode. This is
    #: the clause that rejects a single brief adjustment: one movement cannot
    #: cover many utterances however large it is.
    min_speech_segments_covered: float = 0.60
    #: Seconds of the participant's own speech. Below this there is not enough
    #: co-speech material to judge or to train on.
    min_speech_seconds: float = 8.0

    # ---- composite ---------------------------------------------------------
    #: Accept threshold on the weighted quality score. Left edge of the plateau
    #: over which precision, recall and specificity on the labelled sample are
    #: all identical (0.34-0.40), so it is the value that delivers the full
    #: quality gain while keeping the most data. Chosen by that stated rule, not
    #: fitted -- the sample cannot resolve a finer optimum.
    min_gesture_quality: float = 0.34

    def as_dict(self) -> dict[str, float]:
        return {key: float(value) for key, value in asdict(self).items()}


#: ``name -> (column, zero_at, one_at)``. Piecewise-linear, clipped to [0, 1].
#: Endpoints are in physical units and were chosen from the candidate pool's own
#: distribution: ``zero_at`` near its 10th percentile, ``one_at`` near its 90th,
#: so a score of 0.5 means "typical of the pool" rather than "half of some
#: arbitrary maximum". They are not fitted per-measure; only the final threshold
#: is fitted, which is what keeps this from overfitting 600 labels.
RAMPS: Mapping[str, tuple[str, float, float]] = {
    "elevation": ("wrist_height_p75_mm", -300.0, -40.0),
    "reach": ("wrist_range_mm", 540.0, 860.0),
    "abduction": ("arm_abduction_p75_deg", 18.0, 45.0),
    "spread": ("posture_spread_mm", 175.0, 315.0),
    "coverage": ("gesture_frac_speech", 0.45, 0.92),
    "episode_length": ("episode_median_s", 0.55, 3.0),
    "vigour": ("arm_speed_speech_p50_mm_s", 95.0, 310.0),
    "coherence": ("step_cosine_p50", 0.10, 0.80),
}

#: Which ramps make up each dimension of the score.
DIMENSIONS: Mapping[str, tuple[str, ...]] = {
    # Where the hands are and how much space they use. Peak excursion is
    # deliberately absent -- see the module docstring.
    "posture": ("elevation", "reach", "abduction", "spread"),
    # Whether the gesturing is sustained through the speech rather than a burst.
    "persistence": ("coverage", "episode_length"),
    # How much the arms are actually doing.
    "vigour": ("vigour",),
    # Whether the motion is a person rather than the tracker.
    "integrity": ("coherence",),
}

#: Dimension weights. Posture and persistence carry the most because they are
#: what separated the labelled set; vigour is down-weighted precisely so that a
#: restrained gesturer is not deleted for being restrained.
WEIGHTS: Mapping[str, float] = {
    "posture": 0.40,
    "persistence": 0.30,
    "vigour": 0.10,
    "integrity": 0.20,
}

#: Clause order is the order reasons are reported in, so the flag histogram
#: reads as a funnel. ``(column, op, limit_attribute, flag)``.
DISQUALIFIERS: tuple[tuple[str, str, str, str], ...] = (
    ("speech_seconds", ">=", "min_speech_seconds", "too_little_speech"),
    ("gesture_frac_speech", ">=", "min_gesture_frac_speech", "static_while_speaking"),
    ("wrist_height_p75_mm", ">=", "min_wrist_height_p75_mm", "hands_parked_low"),
    ("episode_count_speech", ">=", "min_episodes_speech", "too_few_episodes"),
    ("episode_median_s", ">=", "min_episode_median_s", "episodes_too_brief"),
    ("speech_segments_covered", ">=", "min_speech_segments_covered", "gesture_not_sustained"),
    ("articulation_ratio", ">=", "min_articulation_ratio", "motion_is_global"),
    ("step_cosine_p50", ">=", "min_step_cosine_p50", "motion_is_detector_noise"),
    ("consistency_r", ">=", "min_consistency_r", "channels_disagree"),
)


def _ramp(values: pd.Series, low: float, high: float) -> pd.Series:
    """Linear 0 at ``low``, 1 at ``high``, clipped. NaN stays NaN."""

    scaled = (values.astype(float) - low) / (high - low)
    return scaled.clip(lower=0.0, upper=1.0)


def articulation_ratio(frame: pd.DataFrame) -> pd.Series:
    """Arm speed during speech relative to torso translation speed.

    Measurement is already torso-relative, so a participant who rocks or walks
    has had most of that removed before this is computed. What remains is the
    residual, and the ratio asks whether the arms are doing more than the body
    is. Derived from columns the scan already writes -- no re-scan needed.
    """

    torso = frame["torso_travel_mm_s_p50"].astype(float)
    arm = frame["arm_speed_speech_p50_mm_s"].astype(float)
    # A torso that is genuinely still would divide by ~0 and yield +inf, which
    # should pass, not fail; flooring the denominator says so explicitly.
    return arm / torso.clip(lower=1.0)


#: Everything tier-2 reads. Checked up front so a stale ``candidates.parquet``
#: -- one written before a measure existed -- produces a sentence naming the
#: missing columns rather than a bare KeyError from inside a ratio.
REQUIRED_COLUMNS: tuple[str, ...] = tuple(sorted(
    {column for column, _, _ in RAMPS.values()}
    | {column for column, _, _, _ in DISQUALIFIERS if column != "articulation_ratio"}
    | {"torso_travel_mm_s_p50", "arm_speed_speech_p50_mm_s"}
))


def check_columns(frame: pd.DataFrame) -> None:
    missing = [column for column in REQUIRED_COLUMNS if column not in frame.columns]
    if missing:
        raise KeyError(
            "tier-2 qualification needs columns that are not in this table: "
            f"{missing}. This usually means candidates.parquet was written by an "
            "older scan; re-run `scan` and `gather`, or `select`, to refresh it."
        )


def add_quality(frame: pd.DataFrame) -> pd.DataFrame:
    """Add ``q_*`` ramp scores, ``dim_*`` dimension scores and ``gesture_quality``."""

    check_columns(frame)
    out = frame.copy()
    if "articulation_ratio" not in out.columns:
        out["articulation_ratio"] = articulation_ratio(out)

    for name, (column, low, high) in RAMPS.items():
        out[f"q_{name}"] = _ramp(out[column], low, high) if column in out.columns else np.nan

    for dimension, parts in DIMENSIONS.items():
        out[f"dim_{dimension}"] = out[[f"q_{p}" for p in parts]].mean(axis=1)

    total = sum(WEIGHTS.values())
    out["gesture_quality"] = (
        sum(out[f"dim_{d}"] * w for d, w in WEIGHTS.items()) / total
    )
    return out


def qualify(frame: pd.DataFrame, limits: Qualifiers | None = None) -> pd.DataFrame:
    """Apply tier-2 qualification. Adds ``qualified``, ``exclusion_flags``, ``fail_stage``.

    Every clause is evaluated for every row -- not short-circuited -- so
    ``exclusion_flags`` lists *all* the reasons a clip was excluded rather than only the
    first. Diagnosing a threshold needs the whole picture; the funnel view is
    ``fail_stage``, which names the first failing clause.
    """

    limits = limits or Qualifiers()
    out = add_quality(frame)
    values = limits.as_dict()

    flags: list[list[str]] = [[] for _ in range(len(out))]
    first: list[str] = ["" for _ in range(len(out))]
    failed_any = np.zeros(len(out), dtype=bool)

    for column, op, attribute, flag in DISQUALIFIERS:
        limit = values[attribute]
        if column not in out.columns:
            raise KeyError(f"qualification needs column {column!r}, which is not in the frame")
        series = pd.to_numeric(out[column], errors="coerce").to_numpy(dtype=float)
        with np.errstate(invalid="ignore"):
            ok = series >= limit if op == ">=" else series <= limit
        # A non-finite measurement fails its clause rather than passing by
        # accident: +inf must not satisfy a lower bound it never measured.
        ok &= np.isfinite(series)
        for index in np.flatnonzero(~ok):
            flags[index].append(flag)
            if not first[index]:
                first[index] = flag
        failed_any |= ~ok

    quality = pd.to_numeric(out["gesture_quality"], errors="coerce").to_numpy(dtype=float)
    low_quality = ~(np.isfinite(quality) & (quality >= values["min_gesture_quality"]))
    for index in np.flatnonzero(low_quality):
        flags[index].append("below_quality_threshold")
        if not first[index]:
            first[index] = "below_quality_threshold"

    out["qualified"] = ~(failed_any | low_quality)
    out["exclusion_flags"] = [";".join(f) for f in flags]
    out["fail_stage"] = first
    return out


def qualification_funnel(qualified: pd.DataFrame) -> pd.DataFrame:
    """First-failing-clause histogram, in clause order, for the report."""

    order = [flag for _, _, _, flag in DISQUALIFIERS] + ["below_quality_threshold"]
    counts = qualified["fail_stage"].value_counts()
    rows = [
        {"clause": flag, "clips_failed_here": int(counts.get(flag, 0))}
        for flag in order
    ]
    rows.append({"clause": "qualified", "clips_failed_here": int(qualified["qualified"].sum())})
    return pd.DataFrame(rows)


def summarise(qualified: pd.DataFrame) -> dict[str, Any]:
    kept = qualified.loc[qualified["qualified"]]
    return {
        "clips_in": int(len(qualified)),
        "clips_qualified": int(len(kept)),
        "qualified_rate": round(float(len(kept)) / max(1, len(qualified)), 4),
        "hours_in": round(float(qualified["window_seconds"].sum() / 3600), 2),
        "hours_qualified": round(float(kept["window_seconds"].sum() / 3600), 2),
        "files_qualified": int(kept["file_id"].nunique()) if len(kept) else 0,
        "median_quality_kept": round(float(kept["gesture_quality"].median()), 4) if len(kept) else None,
    }

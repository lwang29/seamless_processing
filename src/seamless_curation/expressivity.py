"""Clip expressivity: rig-normalised percentile scores over the body-motion measures.

The previous pipeline asked a yes/no question of each window (is this co-speech
gesture good enough to train on?). This module answers a graded one instead:
*how expressive is this clip's body motion, compared with clips recorded on the
same rig?* Nothing is dropped; every clip gets a score or a status saying why it
has none. The raw measures themselves (mm/s, mm, deg/s, rad/s) come from
:mod:`seamless_curation.gesture` and stay in the table; this module only ranks
them.

Channels
--------
Four body channels enter the overall score, with equal weight:

========================  ==========================================================
``expr_energy``           percentile of ``arm_speed_p50_mm_s``
``expr_amplitude``        mean percentile of ``wrist_range_mm`` and ``wrist_excursion_p90_mm``
``expr_head``             percentile of ``head_speed_p75_deg_s``
``expr_hands``            percentile of ``hand_artic_p75_rad_s``
========================  ==========================================================

The design review measured why the composite has exactly these four and no
more. An "activity" channel (``arm_active_frac``, episode rate) is the same arm
speed through a threshold (Spearman 0.90 with energy on 300 sample clips, 0.93
over 846k old windows), and the episode rate is not even monotone in activity
(mean episode count peaks at the 8th ``gesture_frac`` decile, 8.04, and falls to
6.18 at the top one). A variability channel defined as the SD of 1-s binned
speed is amplitude again (Spearman 0.88 with wrist range). Averaging those in
would have counted arm speed twice and amplitude twice. The two amplitude
measures are averaged into *one* channel for the same reason.

Three further channels are published but kept out of the score:

* ``expr_variability`` — percentile of ``arm_speed_cv``, the scale-free
  burstiness of 1-s mean arm speed (Spearman -0.42 with energy, i.e. a
  genuinely different axis). It describes the *temporal shape* of motion, not
  its amount, so it has no business in an "amount of expressivity" mean.
* ``expr_face`` — percentile of ``fau_variability``. Meta released the
  Imitator face features for V00 only, so the channel is ranked against the V00
  reference and is NA for every other group; putting it in the score would make
  the score mean different things per vendor.
* ``expr_vocal`` — percentile of ``vocal_level_range_db`` within the group. It
  exists only for clips with >= 3 s of own speech, so it would make the score
  depend on how much someone talks.

Reference groups and percentiles
--------------------------------
Every channel is a mid-rank percentile against a stored quantile table
(``n_quantiles`` = 1001 quantiles per measure) built from the *reference clips*
of the clip's ``expressivity_reference_group``: clips with status 'measured'
that are not partial. Partial (tail) clips are scored against the reference
but never enter it, because ``wrist_range_mm`` grows with clip length.

Ranking within a group (vendor, with V03 split into the portrait and the
room-camera rig) is the primary normalisation because rigs and fits differ in
motion magnitude at matched behaviour. On 846,300 non-overlapping old windows,
V02 holds 43.0% of the pooled top arm-speed tertile against 29.1% for V03; at
matched speech fraction V02 arm speed is 29-43% above V00 (43.3 vs 30.3 mm/s at
5-30% speech) and hand articulation 25-37% higher, while V02 also has the worst
fit error (reprojection 0.111 vs 0.079 shoulder widths for V00). A pooled score
would largely say "this is V02". ``expressivity_score_pooled`` keeps the
corpus-wide rank for users who accept that: its channels are ranked against the
pooled reference (all groups' reference clips), so two clips with identical raw
values get the same pooled score whatever their vendor, but can differ in the
within-group score.

The mid-rank rule, ``(#quantiles < v + #quantiles <= v) / 2 / n_quantiles``, is
the one the design review asked to be fixed in advance: tied values (motion
measures that are exactly zero) take the middle of their tied block, not its
bottom or top. The v0 windows had ``gesture_frac == 0`` in 7.37% of windows,
where the three usual conventions give 0, 0.037 or 0.074.

The overall ``expressivity_score`` is the mean of the four channel percentiles
*re-ranked* against the same mean over the group's reference clips, so the score
itself is uniform over reference clips: 0.7 means more expressive than 70% of
full-length clips of the same rig. (A plain mean of percentiles concentrates
near 0.5 and would make the 1/3-2/3 level cuts mean something else.) Levels are
'low' (< 1/3), 'medium', 'high' (>= 2/3), 'unknown' when the score is NA.

Status and confidence
---------------------
``expressivity_status`` is decided in this order (first match wins; the order
of the registry's list):

1. ``too_short`` — clip under 2 s (the motion measures are NA there);
2. ``pose_distorted`` — severe anamorphic V01 (2160x2160, 1920x1080). The
   released SMPL-H absorbed a 9:16 or 81:256 stretch that cannot be undone, and
   the posture annotation already treats these poses as unobservable;
3. ``no_subject`` — no frame with a valid tracking box (``subject_present_frac``
   is 0, or NA on a clip long enough to have it);
4. ``incomplete`` — a composite input is NA (typically ``hand_artic_p75_rad_s``
   when fewer than 2 s of the clip have a non-frozen hand pose);
5. ``measured``.

Every ``expr_*``, score, level and confidence value is NA/'unknown' unless the
status is 'measured'; an extra channel is additionally NA when its own input is.
One further NA, not a status: a 'measured' clip whose group has no reference
clip gets NA channels and score (the pooled score still exists). That happens
only when the reference was built from another table (a sample, or the old
windows, which contain no V03 room-camera file); a full-corpus build has
reference clips in every group.

``expressivity_confidence`` = ``min(1, clip_seconds / L) * wrists_in_frame_frac``:
evidence coverage, not P(correct). ``subject_present_frac`` was the design's
observability term but is uninformative in this release (``is_valid_box`` is 1 on
every frame of 52/52 sample files; ``box_valid_frac < 1`` in 0.107% of old
windows); ``wrists_in_frame_frac`` requires a valid box *and* both wrists
confident and inside the raster, which is what actually varies per clip.

Known limitation, documented rather than corrected: the SMPL-H fitter marks fast
motion invalid (old windows: arm speed 43.9 mm/s when fully valid, 73-84 mm/s at
0.5-0.99 valid), and ``hand_artic_p75_rad_s`` is computed over non-frozen hand
frames only, so clips with many invalid frames have their most active frames
removed from the hands channel. ``smplh_valid_frac`` qualifies the raw measures
in the registry for exactly this reason.

Real-data check (old v0 windows, read-only)
-------------------------------------------
``outputs/vibes_upper_body_v1/windows.parquet`` holds 2,423,304 overlapping
30-s windows of 118,570 files at a 10-s hop; ``window_index % 3 == 0`` tiles
each file without overlap (846,300 windows) and stands in for the 30-s clip
grid. Head speed is absent there and was replaced by a constant (so
``expr_head`` is 0.5 everywhere) and ``box_valid_frac`` stood in for
``wrists_in_frame_frac``. The old corpus had already dropped the room-camera
and severe-anamorphic files, so 'V03_room' has no reference there. Measured by
``scratchpad/impl_tmp/7/validate_windows.py``:

* timing: ``build_reference`` 1.6 s + ``score`` 1.6 s on the 846,300 tiling
  windows, 4.2 s + 4.3 s on all 2.42M (a synthetic 1.04M-clip table: 2.3 s +
  1.9 s); the reference JSON is 0.52 MB;
* the within-group score is uniform over the reference: every group's
  low/medium/high shares are 33.30-33.40%, and its mean is 0.500;
* the pooled level puts 45.2% of V02 windows in 'high' against 29.7% of V00,
  28.4% of V01 and 31.3% of V03 portrait (pooled mean score 0.575 for V02,
  0.479-0.484 for the others): the rig confound the within-group score removes;
* 28 windows of 5 files had ``box_valid_frac`` 0 (median arm speed 0.13 mm/s,
  a frozen fit): 'no_subject', where a score would have ranked them as the
  least expressive clips of the corpus;
* Spearman among the channels: energy-hands 0.744, energy-amplitude 0.577,
  amplitude-hands 0.548 (related, not redundant); the score against the old
  ``gesture_frac`` 0.896.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from .schema import EXPRESSIVITY_GROUPS

#: The four body channels of the composite and the raw measures behind each.
COMPOSITE: dict[str, tuple[str, ...]] = {
    "expr_energy": ("arm_speed_p50_mm_s",),
    "expr_amplitude": ("wrist_range_mm", "wrist_excursion_p90_mm"),
    "expr_head": ("head_speed_p75_deg_s",),
    "expr_hands": ("hand_artic_p75_rad_s",),
}
#: Published channels that do not enter ``expressivity_score``.
EXTRA: dict[str, tuple[str, ...]] = {
    "expr_variability": ("arm_speed_cv",),
    "expr_face": ("fau_variability",),
    "expr_vocal": ("vocal_level_range_db",),
}
COMPOSITE_INPUTS: tuple[str, ...] = tuple(m for ms in COMPOSITE.values() for m in ms)
EXTRA_INPUTS: tuple[str, ...] = tuple(m for ms in EXTRA.values() for m in ms)
RAW_MEASURES: tuple[str, ...] = COMPOSITE_INPUTS + EXTRA_INPUTS

NOT_APPLICABLE = "not_applicable"
#: Groups that have a reference population (every group but 'not_applicable').
REFERENCE_GROUPS: tuple[str, ...] = tuple(g for g in EXPRESSIVITY_GROUPS if g != NOT_APPLICABLE)
POOLED = "pooled"
#: Key of the composite-mean quantile table inside each group's quantiles.
COMPOSITE_KEY = "composite_mean"
#: expr_face is ranked against this group only (movement:* exists only in V00).
FACE_GROUP = "V00"
#: V03 rasters recorded by the room-camera rig (catalog.room_camera_rig).
V03_ROOM_RASTERS = frozenset({"rotated_3840x2160", "rotated_640x480"})

#: Clips shorter than this carry NA motion measures (clips.py / gesture.py rule).
MIN_CLIP_SECONDS = 2.0
#: float32 ``clip_seconds`` tolerance: a 60-frame clip at 30 fps is exactly 2 s.
_SECONDS_TOL = 1e-4
LEVEL_CUTS = (1.0 / 3.0, 2.0 / 3.0)
LEVELS = ("low", "medium", "high", "unknown")
STATUSES = ("measured", "too_short", "pose_distorted", "no_subject", "incomplete")
REFERENCE_KIND = "seamless_curation.expressivity_reference"
REFERENCE_VERSION = 1

#: Columns ``score`` returns, in registry order.
SCORE_COLUMNS: tuple[str, ...] = (
    *COMPOSITE, *EXTRA,
    "expressivity_score", "expressivity_level",
    "expressivity_score_pooled", "expressivity_level_pooled",
    "expressivity_confidence", "expressivity_status",
)
#: Clip columns ``score`` and ``build_reference`` read (EXTRA_INPUTS may be absent).
REQUIRED_COLUMNS: tuple[str, ...] = (
    "vendor", "expressivity_reference_group", "clip_seconds", "is_partial",
    "subject_present_frac", "wrists_in_frame_frac", "smplh_anamorphic", *COMPOSITE_INPUTS,
)


# -----------------------------------------------------------------------------
# groups and status
# -----------------------------------------------------------------------------
VENDORS: tuple[str, ...] = ("V00", "V01", "V02", "V03")


def _is_missing(x: Any) -> bool:
    try:
        return bool(pd.isna(x))
    except (TypeError, ValueError):  # array-likes are not a missing scalar
        return False


def _str_array(values: Any) -> np.ndarray:
    """Object array of strings with every missing value (None, NaN, pd.NA) as ''.

    ``pd.NA`` inside an object array makes ``==`` raise, and the string columns
    of the tables are pandas 'string' dtype, so every label comparison goes
    through this.
    """

    a = np.asarray(values.to_numpy(dtype=object) if isinstance(values, pd.Series) else values, dtype=object)
    a = a.reshape(-1) if a.ndim != 1 else a
    return np.where(pd.isna(a), "", a).astype(object)


def reference_group(vendor: str, raster_class: str, smplh_anamorphic: str, measured: bool) -> str:
    """The reference population a recording's clips are ranked against.

    'not_applicable' when the recording was not measured or its pose is severely
    anamorphic; V03 splits into the room-camera rig (3840x2160 and 640x480
    rasters, far-field audio and a different camera geometry) and the portrait
    rig (everything else, pillarboxed portraits included); V00, V01 (portrait
    and the mild 1012x1920 squeeze) and V02 are their vendor. A missing
    ``measured`` counts as not measured; an unknown vendor raises.
    """

    if not isinstance(vendor, str) or vendor not in VENDORS:
        raise ValueError(f"unknown vendor: {vendor!r}")
    severe = isinstance(smplh_anamorphic, str) and smplh_anamorphic == "severe"
    if _is_missing(measured) or not bool(measured) or severe:
        return NOT_APPLICABLE
    if vendor == "V03":
        room = isinstance(raster_class, str) and raster_class in V03_ROOM_RASTERS
        return "V03_room" if room else "V03_portrait"
    return str(vendor)


def reference_groups(
    vendor: Sequence[str] | np.ndarray | pd.Series,
    raster_class: Sequence[str] | np.ndarray | pd.Series,
    smplh_anamorphic: Sequence[str] | np.ndarray | pd.Series,
    measured: Sequence[bool] | np.ndarray | pd.Series,
) -> np.ndarray:
    """Vectorised :func:`reference_group` (object array of group names)."""

    vendor_a = _str_array(vendor)
    raster_a = _str_array(raster_class)
    anam_a = _str_array(smplh_anamorphic)
    measured_o = np.asarray(measured.to_numpy(dtype=object) if isinstance(measured, pd.Series) else measured,
                            dtype=object).reshape(-1)
    measured_a = np.where(pd.isna(measured_o), False, measured_o).astype(bool)
    unknown = ~np.isin(vendor_a, VENDORS)
    if unknown.any():
        raise ValueError(f"unknown vendor(s): {sorted(set(map(str, vendor_a[unknown])))}")
    room = np.isin(raster_a, list(V03_ROOM_RASTERS))
    groups = np.where(vendor_a == "V03", np.where(room, "V03_room", "V03_portrait"), vendor_a)
    groups = np.where(measured_a & (anam_a != "severe"), groups, NOT_APPLICABLE)
    return groups.astype(object)


def _status_array(
    clip_seconds: Any, subject_present_frac: Any, smplh_anamorphic: Any, composite_finite: Any
) -> np.ndarray:
    seconds = np.asarray(clip_seconds, dtype=np.float64)
    subject = np.asarray(subject_present_frac, dtype=np.float64)
    anamorphic = _str_array(smplh_anamorphic)
    finite = np.asarray(composite_finite, dtype=bool)
    too_short = ~(seconds >= MIN_CLIP_SECONDS - _SECONDS_TOL)  # NaN seconds count as short
    distorted = anamorphic == "severe"
    no_subject = ~(subject > 0.0)  # 0 or NA
    status = np.select(
        [too_short, distorted, no_subject, ~finite],
        ["too_short", "pose_distorted", "no_subject", "incomplete"],
        default="measured",
    )
    return status.astype(object)


def expressivity_status(
    clip_seconds: float, subject_present_frac: float, smplh_anamorphic: str,
    composite_inputs_all_finite: bool,
) -> str:
    """Why a clip has (or lacks) an expressivity score; see the module docstring."""

    return str(_status_array([_nan_if_missing(clip_seconds)], [_nan_if_missing(subject_present_frac)],
                             [smplh_anamorphic], [bool(composite_inputs_all_finite)])[0])


def _nan_if_missing(x: Any) -> float:
    return float("nan") if _is_missing(x) else float(x)


# -----------------------------------------------------------------------------
# percentiles
# -----------------------------------------------------------------------------
def quantile_levels(n_quantiles: int) -> np.ndarray:
    return np.linspace(0.0, 1.0, int(n_quantiles))


def percentile(values: np.ndarray, quantiles: np.ndarray) -> np.ndarray:
    """Mid-rank percentile of ``values`` against a sorted quantile table.

    ``(searchsorted(q, v, 'left') + searchsorted(q, v, 'right')) / 2 / len(q)``,
    clipped to [0, 1]: a value below every quantile is 0, above every quantile 1,
    and a value tied with a block of quantiles sits in the block's middle. NaN
    in gives NaN out, and an empty table (a group with no reference clip) gives
    NaN everywhere.

    Implemented with one search over the table's distinct values (the left and
    right positions of a value are the cumulative count before its distinct
    value and that plus the value's multiplicity when it is in the table), which
    halves the dominant cost of :func:`score`; identical to the two searches.
    """

    v = np.asarray(values, dtype=np.float64)
    q = np.asarray(quantiles, dtype=np.float64)
    if q.size == 0:
        return np.full(v.shape, np.nan)
    distinct, first = np.unique(q, return_index=True)  # q is sorted: first = cumulative count before
    count = np.diff(np.r_[first, q.size])
    idx = np.searchsorted(distinct, v, side="left")
    inside = np.minimum(idx, distinct.size - 1)
    exact = (idx < distinct.size) & (distinct[inside] == v)
    left = np.where(idx < distinct.size, first[inside], q.size)
    right = left + np.where(exact, count[inside], 0)
    p = np.clip((left + right) / (2.0 * q.size), 0.0, 1.0)
    return np.where(np.isnan(v), np.nan, p)


def tertile_level(scores: Any) -> np.ndarray:
    """'low' (< 1/3), 'medium', 'high' (>= 2/3), 'unknown' for NA."""

    s = np.asarray(scores, dtype=np.float64)
    level = np.select([np.isnan(s), s < LEVEL_CUTS[0], s < LEVEL_CUTS[1]],
                      ["unknown", "low", "medium"], default="high")
    return level.astype(object)


def _quantile_table(values: np.ndarray, n_quantiles: int) -> list[float]:
    v = np.asarray(values, dtype=np.float64)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return []
    q = np.quantile(v, quantile_levels(n_quantiles))
    q = np.maximum.accumulate(q)  # guard against float non-monotonicity in interpolation
    return [float(x) for x in q]


def _composite_channels(measures: Mapping[str, np.ndarray], tables: Mapping[str, Any]) -> dict[str, np.ndarray]:
    """Channel percentiles of the four body channels plus their plain mean."""

    out: dict[str, np.ndarray] = {}
    for channel, inputs in COMPOSITE.items():
        parts = [percentile(measures[m], tables.get(m, [])) for m in inputs]
        out[channel] = parts[0] if len(parts) == 1 else np.mean(parts, axis=0)
    out[COMPOSITE_KEY] = np.mean([out[c] for c in COMPOSITE], axis=0)
    return out


def _measure_arrays(clips: pd.DataFrame, names: Iterable[str]) -> dict[str, np.ndarray]:
    arrays = {}
    for name in names:
        if name in clips.columns:
            arrays[name] = _float_column(clips, name)
        elif name in COMPOSITE_INPUTS:
            raise KeyError(f"expressivity needs column {name!r}")
        else:  # an extra channel's input not produced (e.g. no audio pass): all NA
            arrays[name] = np.full(len(clips), np.nan)
    return arrays


def _check_columns(clips: pd.DataFrame, names: Iterable[str]) -> None:
    missing = [c for c in names if c not in clips.columns]
    if missing:
        raise KeyError(f"expressivity needs columns {missing}")


def _clip_status(clips: pd.DataFrame, measures: Mapping[str, np.ndarray]) -> np.ndarray:
    finite = np.logical_and.reduce([np.isfinite(measures[m]) for m in COMPOSITE_INPUTS])
    return _status_array(
        _float_column(clips, "clip_seconds"), _float_column(clips, "subject_present_frac"),
        clips["smplh_anamorphic"], finite,
    )


def _float_column(clips: pd.DataFrame, name: str) -> np.ndarray:
    return pd.to_numeric(clips[name], errors="coerce").to_numpy(np.float64, na_value=np.nan)


def _bool_array(series: pd.Series) -> np.ndarray:
    if series.isna().any():
        raise ValueError(f"{series.name} must not contain NA")
    return series.to_numpy(dtype=bool)


# -----------------------------------------------------------------------------
# reference
# -----------------------------------------------------------------------------
def build_reference(clips: pd.DataFrame, n_quantiles: int = 1001) -> dict:
    """Quantile tables of every raw measure and of the composite mean, per group.

    Reference clips are those with expressivity status 'measured' (recomputed
    here from the same columns :func:`score` reads) and ``is_partial`` False.
    For each group of :data:`REFERENCE_GROUPS` and for 'pooled' (the union of
    all groups' reference clips) the result holds, per raw measure, the
    ``n_quantiles`` quantiles at ``linspace(0, 1, n_quantiles)`` of its finite
    values over the reference clips (numpy 'linear' interpolation), and the
    quantiles of the composite mean, where the composite of each reference clip
    is computed through the just-built tables exactly as :func:`score` computes
    it. A measure with no finite reference value has an empty list. The result
    is plain JSON (``json.dumps(ref, allow_nan=False)`` works) and depends only
    on the set of reference rows, not on their order.
    """

    if int(n_quantiles) < 2:
        raise ValueError("n_quantiles must be >= 2")
    n_quantiles = int(n_quantiles)
    _check_columns(clips, ("expressivity_reference_group", "clip_seconds", "is_partial",
                           "subject_present_frac", "smplh_anamorphic"))
    measures = _measure_arrays(clips, RAW_MEASURES)
    status = _clip_status(clips, measures)
    groups = _str_array(clips["expressivity_reference_group"])
    reference_rows = (status == "measured") & ~_bool_array(clips["is_partial"])
    reference_rows &= np.isin(groups, REFERENCE_GROUPS)

    out_groups: dict[str, dict] = {}
    for group in (*REFERENCE_GROUPS, POOLED):
        rows = reference_rows if group == POOLED else reference_rows & (groups == group)
        sub = {m: measures[m][rows] for m in RAW_MEASURES}
        tables: dict[str, list[float]] = {m: _quantile_table(sub[m], n_quantiles) for m in RAW_MEASURES}
        composite = _composite_channels(sub, tables)[COMPOSITE_KEY]
        tables[COMPOSITE_KEY] = _quantile_table(composite, n_quantiles)
        out_groups[group] = {
            "n_clips": int(rows.sum()),
            "n_finite": {m: int(np.isfinite(sub[m]).sum()) for m in RAW_MEASURES},
            "quantiles": tables,
        }
    return {
        "kind": REFERENCE_KIND,
        "version": REFERENCE_VERSION,
        "n_quantiles": n_quantiles,
        "quantile_levels": "linspace(0, 1, n_quantiles), numpy 'linear' interpolation",
        "percentile_rule": "mid-rank: (searchsorted left + searchsorted right) / 2 / n_quantiles",
        "reference_clips": "expressivity_status == 'measured' and not is_partial",
        "composite": {k: list(v) for k, v in COMPOSITE.items()},
        "extra": {k: list(v) for k, v in EXTRA.items()},
        "face_group": FACE_GROUP,
        "groups": out_groups,
    }


def _tables(reference: Mapping, group: str) -> dict[str, np.ndarray]:
    try:
        raw = reference["groups"][group]["quantiles"]
    except KeyError as exc:
        raise ValueError(f"expressivity reference has no group {group!r}") from exc
    return {k: np.asarray(v, dtype=np.float64) for k, v in raw.items()}


# -----------------------------------------------------------------------------
# scoring
# -----------------------------------------------------------------------------
def score(clips: pd.DataFrame, reference: Mapping, *, clip_seconds_nominal: float) -> pd.DataFrame:
    """Every expressivity column for every clip (same index as ``clips``).

    ``clips`` needs :data:`REQUIRED_COLUMNS`; the inputs of the extra channels
    (``arm_speed_cv``, ``fau_variability``, ``vocal_level_range_db``) are used
    when present and treated as NA when absent. ``clip_seconds_nominal`` is the
    grid length L of the confidence term. Scores are float32, levels and status
    are strings; see the module docstring for every definition.
    """

    if not (clip_seconds_nominal and clip_seconds_nominal > 0):
        raise ValueError("clip_seconds_nominal must be positive")
    if reference.get("kind") != REFERENCE_KIND or reference.get("version") != REFERENCE_VERSION:
        raise ValueError("not an expressivity reference of this version")
    _check_columns(clips, REQUIRED_COLUMNS)
    n = len(clips)
    measures = _measure_arrays(clips, RAW_MEASURES)
    status = _clip_status(clips, measures)
    groups = _str_array(clips["expressivity_reference_group"])
    vendor = _str_array(clips["vendor"])
    measured = status == "measured"

    # A measured clip must belong to a rankable group of its own vendor: a
    # mismatch means the recording columns were joined wrongly, not a data fact
    # (unmeasured recordings have no clips; severe-anamorphic ones are
    # 'pose_distorted', never 'measured').
    bad = measured & ~np.isin(groups, REFERENCE_GROUPS)
    if bad.any():
        raise ValueError(f"{int(bad.sum())} measured clips have no reference group "
                         f"(e.g. {sorted(set(map(str, groups[bad])))[:3]})")
    for group in REFERENCE_GROUPS:
        bad = (groups == group) & (vendor != group[:3])
        if bad.any():
            raise ValueError(f"{int(bad.sum())} clips of vendor(s) {sorted(set(map(str, vendor[bad])))} "
                             f"have reference group {group!r}")

    out = {name: np.full(n, np.nan) for name in (*COMPOSITE, *EXTRA, "expressivity_score",
                                                 "expressivity_score_pooled")}
    for group in REFERENCE_GROUPS:
        rows = np.flatnonzero(measured & (groups == group))
        if rows.size == 0:
            continue
        tables = _tables(reference, group)
        sub = {m: measures[m][rows] for m in RAW_MEASURES}
        channels = _composite_channels(sub, tables)
        for channel in COMPOSITE:
            out[channel][rows] = channels[channel]
        out["expressivity_score"][rows] = percentile(channels[COMPOSITE_KEY], tables.get(COMPOSITE_KEY, []))
        out["expr_variability"][rows] = percentile(sub["arm_speed_cv"], tables.get("arm_speed_cv", []))
        out["expr_vocal"][rows] = percentile(sub["vocal_level_range_db"], tables.get("vocal_level_range_db", []))
        if group == FACE_GROUP:
            out["expr_face"][rows] = percentile(sub["fau_variability"], tables.get("fau_variability", []))

    rows = np.flatnonzero(measured)
    if rows.size:
        pooled = _tables(reference, POOLED)
        channels = _composite_channels({m: measures[m][rows] for m in COMPOSITE_INPUTS}, pooled)
        out["expressivity_score_pooled"][rows] = percentile(channels[COMPOSITE_KEY], pooled.get(COMPOSITE_KEY, []))

    seconds = _float_column(clips, "clip_seconds")
    wrists = _float_column(clips, "wrists_in_frame_frac")
    confidence = np.minimum(1.0, seconds / float(clip_seconds_nominal)) * wrists
    confidence = np.where(measured, np.clip(confidence, 0.0, 1.0), np.nan)

    frame: dict[str, Any] = {}
    for name in (*COMPOSITE, *EXTRA):
        frame[name] = out[name].astype(np.float32)
    frame["expressivity_score"] = out["expressivity_score"].astype(np.float32)
    frame["expressivity_level"] = tertile_level(frame["expressivity_score"])
    frame["expressivity_score_pooled"] = out["expressivity_score_pooled"].astype(np.float32)
    frame["expressivity_level_pooled"] = tertile_level(frame["expressivity_score_pooled"])
    frame["expressivity_confidence"] = confidence.astype(np.float32)
    frame["expressivity_status"] = status
    return pd.DataFrame(frame, index=clips.index, columns=list(SCORE_COLUMNS))


# -----------------------------------------------------------------------------
# report and aggregate helpers
# -----------------------------------------------------------------------------
def correlation_matrix(scored: pd.DataFrame, columns: Sequence[str] | None = None) -> pd.DataFrame:
    """Spearman correlations among the ``expr_*`` channels (pairwise complete).

    For the annotation report: it shows how many independent axes the channels
    really span. Pairs with fewer than 3 common finite values are NA.
    """

    cols = list(columns) if columns is not None else [c for c in (*COMPOSITE, *EXTRA) if c in scored.columns]
    data = scored[cols].astype(np.float64)
    return data.corr(method="spearman", min_periods=3)


def _is_measured(scored: pd.DataFrame) -> np.ndarray:
    return _str_array(scored["expressivity_status"]) == "measured"


def recording_expressivity(scored: pd.DataFrame) -> pd.DataFrame:
    """Recording-level aggregates of measured clips, indexed by ``file_id``.

    Needs ``file_id``, ``clip_index``, ``clip_seconds``, ``expressivity_score``,
    ``expressivity_status``. ``recording_expressivity_mean`` is weighted by
    clip_seconds (NA without a measured clip); ``recording_expressivity_clip_sd``
    the unweighted sample SD (ddof=1) of clip scores (NA below 2 clips);
    ``recording_expressivity_trend`` the Spearman correlation of clip score with
    clip_index (NA below 3 clips or when all scores tie). Every file_id of the
    input gets a row. Not part of the module contract: offered to the integrator.
    """

    _check_columns(scored, ("file_id", "clip_index", "clip_seconds", "expressivity_score", "expressivity_status"))
    files = pd.Index(pd.unique(scored["file_id"]), name="file_id")
    scores = scored["expressivity_score"].astype(np.float64)
    keep = _is_measured(scored) & scores.notna().to_numpy()
    d = pd.DataFrame({
        "file_id": scored.loc[keep, "file_id"].to_numpy(),
        "clip_index": scored.loc[keep, "clip_index"].astype(np.float64).to_numpy(),
        "w": scored.loc[keep, "clip_seconds"].astype(np.float64).to_numpy(),
        "s": scores[keep].to_numpy(),
    })
    d["ws"] = d["w"] * d["s"]
    g = d.groupby("file_id", sort=False)
    sums = g[["w", "ws"]].sum()
    count = g.size()
    mean = sums["ws"] / sums["w"].where(sums["w"] > 0)
    sd = g["s"].std(ddof=1)
    d["rs"] = g["s"].rank(method="average")
    d["ri"] = g["clip_index"].rank(method="average")
    d["rs"] -= d.groupby("file_id", sort=False)["rs"].transform("mean")
    d["ri"] -= d.groupby("file_id", sort=False)["ri"].transform("mean")
    d["xy"], d["xx"], d["yy"] = d["rs"] * d["ri"], d["rs"] ** 2, d["ri"] ** 2
    m = d.groupby("file_id", sort=False)[["xy", "xx", "yy"]].sum()
    denom = np.sqrt(m["xx"] * m["yy"])
    trend = (m["xy"] / denom.where(denom > 0)).where(count >= 3)
    out = pd.DataFrame({
        "recording_expressivity_mean": mean,
        "recording_expressivity_clip_sd": sd.where(count >= 2),
        "recording_expressivity_trend": trend.clip(-1.0, 1.0),
    }).reindex(files)
    return out.astype(np.float32)


def interaction_expressivity(scored: pd.DataFrame) -> pd.DataFrame:
    """Interaction-level aggregates, indexed by ``interaction_key``.

    Needs ``interaction_key`` plus the :func:`recording_expressivity` inputs. A
    member counts as measured when it has at least one measured clip.
    ``interaction_expressivity_mean`` is the clip_seconds-weighted mean over both
    members' measured clips; ``interaction_expressivity_gap`` the absolute
    difference of the members' duration-weighted means (with more than two
    measured members, which the release does not contain, the max-min range).
    Both are NA with fewer than two measured members. Not part of the module
    contract: offered to the integrator.
    """

    _check_columns(scored, ("interaction_key", "file_id", "clip_seconds", "expressivity_score",
                            "expressivity_status"))
    keys = pd.Index(pd.unique(scored["interaction_key"]), name="interaction_key")
    scores = scored["expressivity_score"].astype(np.float64)
    keep = _is_measured(scored) & scores.notna().to_numpy()
    d = pd.DataFrame({
        "interaction_key": scored.loc[keep, "interaction_key"].to_numpy(),
        "file_id": scored.loc[keep, "file_id"].to_numpy(),
        "w": scored.loc[keep, "clip_seconds"].astype(np.float64).to_numpy(),
    })
    d["ws"] = d["w"] * scores[keep].to_numpy()
    per_member = d.groupby(["interaction_key", "file_id"], sort=False)[["w", "ws"]].sum()
    per_member["mean"] = per_member["ws"] / per_member["w"].where(per_member["w"] > 0)
    g = per_member.groupby(level="interaction_key", sort=False)
    members = g.size()
    totals = g[["w", "ws"]].sum()
    mean = (totals["ws"] / totals["w"].where(totals["w"] > 0)).where(members >= 2)
    gap = (g["mean"].max() - g["mean"].min()).where(members >= 2)
    out = pd.DataFrame({"interaction_expressivity_mean": mean,
                        "interaction_expressivity_gap": gap}).reindex(keys)
    return out.astype(np.float32)


@dataclass(frozen=True)
class ReferenceSummary:
    """Per-group reference sizes, for logs and the report."""

    n_clips: dict[str, int]

    @classmethod
    def of(cls, reference: Mapping) -> "ReferenceSummary":
        return cls({g: int(v["n_clips"]) for g, v in reference["groups"].items()})


def is_finite_number(x: Any) -> bool:
    try:
        return math.isfinite(float(x))
    except (TypeError, ValueError):
        return False

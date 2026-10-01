"""Sitting versus standing, measured from the fitted legs and gated on what the camera saw.

Posture is an annotation here, never a gate. The retired v0 FM1 seated detector
rejected files on one file-level cut; this module keeps its two measures, adds
an explicit *unclear* band and an *unobserved* state, and splits the work in
two so a threshold can move without a rescan:

* **scan side (continuous only)** — :func:`posture_frames` computes per-frame leg
  geometry, :func:`posture_bins` reduces it to one row per 1-s bin (the compact
  shard the scan writes, ~29M bins over the corpus), and
  :func:`clip_posture_measures` / :func:`recording_posture_measures` report the
  continuous medians. No threshold is applied on this side.
* **annotate side (rules)** — :func:`posture_rule` picks the rule for a
  recording, :func:`classify_bins` applies it to bins, and
  :func:`aggregate_states` / :func:`aggregate_groups` turn bin states into the
  clip- and recording-level label, shares, transitions and confidence. Every
  threshold lives in :class:`PostureRules`.

Every quantity describes the file's own tracked participant, per clip or per
recording; the partner's posture lives on the partner's rows.

Measures (pelvis-frame FK joints, all sixteen betas zero)
---------------------------------------------------------
With ``u = unit(neck - pelvis)`` (SMPL-H 12 and 0):

``hip_flexion``
    Mean over both sides of ``angle(u, knee - hip)`` in degrees: the torso-thigh
    angle. ~180 with straight legs (the SMPL-H rest pose reads 172), ~90 seated.
    It is a pure joint angle, so camera distance and quarter-turned rasters do
    not change it.
``knee_between``
    Mean over sides of ``((knee - pelvis)·(-u)) / ((ankle - pelvis)·(-u))``: how
    far down the hip-to-ankle drop the knee sits, along the participant's own
    torso axis. Standing ~0.5 (rest 0.54), seated 0.15-0.4. Because betas are
    zero the thigh and shin are identical in every file, so the ratio is pose,
    not body proportion — the reason the 2D leg ratios were retired.
``shin_inverted``
    Both ankles above their knees along ``-u``: the feet-on-a-stool-rung
    posture. Measured and reported, never used by a rule (on V03 8/8 shin-only
    firings were standing people with ankles out of frame; on V02 181/182).

Observability gates (each quantity on the landmarks it depends on)
------------------------------------------------------------------
``visible`` is :func:`seamless_curation.framing.visible_mask`: released COCO
confidence >= 0.5 **and** the point inside the stored raster **and** a valid
box. Confidence alone is not visibility (up to 72% of V03 P1437 frames have
ankle confidence >= 0.5 with the ankle outside the raster).

* ``hip_observable``: both hips (COCO 11, 12) and both knees (13, 14) visible.
  Gates ``hip_flexion``. Knees pass on >= 98% of frames in 51/52 sampled files.
* ``ankle_observable``: both knees and both ankles (15, 16) visible. Gates
  ``knee_between`` and ``shin_inverted``. Without it the ankle comes from the
  fit's prior: in the 52-file sample ankles are visible on < 20% of frames in
  12/12 V02 files (11 at exactly 0) while ``knee_between`` still reads
  0.53-0.79 there.

Rules (per 1-s bin; thresholds in :class:`PostureRules`)
--------------------------------------------------------
``v03_hip`` (V03, every raster — angles are rotation-invariant)
    sitting hip < 136°, standing hip > 152°, unclear between. The band is the
    gap in FM1's 164 V03 *file-level* hand labels: no stander below 136.1°, no
    sitter above 151.9° (orient sample: standing minimum 146.8°, seated
    maximum 140.1°). Evidence ``v03_file_labels_164``.
``v00_knee`` (V00)
    sitting knee_between < 0.30, standing >= 0.48, unclear between. FM1's
    Round-4 labels (29 files) put sitters at 0.152-0.406 and standers at
    0.508-0.613; the full 66-file set adds the adjudication pool and overlaps
    the band from both sides (file medians: sitters 0.12-0.41 plus two known
    misses at 0.485 and 0.728, standers 0.39-0.61; see *Validation*). The
    labels were drawn from flagged/adjudication pools (random-draw precision
    was 56% in 0.30-0.43), hence evidence
    ``v00_file_labels_66_selection_biased``. Hip flexion is **not** used on
    V00: a rig offset puts standing V00 hip angles ~25° below V03 (window
    medians 142.7 vs 168.7) and the V03 146° cut flags 62% of standing V00
    windows.
``unvalidated_hip_knee`` (V01 square-pixel incl. mild 1012x1920, V02)
    with knee_between: standing hip >= 135 and kb >= 0.48, sitting hip < 120
    and kb < 0.40; hip-only when ankles are unobserved (standing hip >= 135,
    sitting hip < 120); else unclear. No seated person was seen in these rigs;
    standing minima were 136.6° (V01) and 125.0° (V02), and hip < 120 had zero
    false positives in every vendor. Evidence ``unlabelled``.
``not_measurable_anamorphic`` (smplh_anamorphic 'severe': V01 2160x2160, 1920x1080)
    every bin unobserved. The released SMPL-H absorbed the stretch: standing
    people read hip 88-154° (median 111) and FM1 read 93.6% of these files
    "seated", all artefact. ``not_measured`` (recording unmeasured): unobserved.

Aggregation (a clip's or a recording's bins)
--------------------------------------------
Counts over the ``n`` bins: ``observed = standing + sitting + unclear``,
``decided = standing + sitting``; the four shares are over ``n`` and sum to 1.

1. ``unknown`` if observed < 50% of n or fewer than 3 observed bins.
2. ``unclear`` if decided < 50% of observed (mostly borderline bins).
3. Sustained runs: drop the undecided bins, run-length encode what remains,
   keep runs of >= 5 bins; ``transitions`` counts state changes between
   consecutive kept runs (runs bridge across unclear/unobserved bins).
   ``mixed`` if transitions >= 1 — the same criterion as the transition count,
   so a recording cannot read 'standing' with 2 transitions.
4. else ``standing`` / ``sitting`` if that state holds >= 80% of *decided*
   bins (the critique showed that counting unclear bins in the denominator
   turned 4/45 zero-contrary-evidence standing files 'unclear'), else unclear.

``confidence`` = winning bins / n (standing+sitting for mixed), NaN for
unclear/unknown: evidence support, not P(correct), and without the old
per-vendor validation multiplier (which made it a vendor indicator).
``transitions`` is reported whatever the label (it is a measure).

Validation on real data (2026-09-24, read-only; FK betas = 0, visibility as above)
---------------------------------------------------------------------------------
* The measures reproduce FM1: on the 66 hand-labelled V00 files the
  recording ``knee_between`` median equals the stored
  ``fm1_knee_between_torso_p50`` (``outputs/session*/v00_fm_scan_r5|r12``)
  within 1e-3 on 61/66 files and within 0.006 on all; hip flexion within
  0.1 deg on 57/66 and 1.1 deg on all. The residue is the visibility gate,
  which FM1 did not have; V00 legs are in shot, so it removes little
  (``recording_lower_body_observed_frac`` >= 0.91, 64/66 at >= 0.997).
* Recording labels from the bins with the default rules, against the 66 file
  labels (rows truth, columns label)::

                 sitting  standing  unclear
      sitting          5         2       12      (19)
      standing         0        36       11      (47)

  No stander reads sitting. Of the two sitters that read standing,
  V00_S0043_I00000540_P0065 (median 0.485) matches the documented
  dangling-leg stool sitter; V00_S0216_I00000515_P0293 reads 0.728 with 88%
  standing bins, i.e. extended legs in the fit (not re-checked by eye). The 23
  unclear files are the pools the labels were drawn from: 12 sitters with
  medians 0.30-0.41 and 11 standers 0.39-0.48. The band is doing its job on
  a sample selected for ambiguity, not a random one. 516 clips of these files
  against the *file* label: sitting files 52 sitting / 7 standing / 92
  unclear / 1 unknown; standing files 279 standing / 0 sitting / 82 unclear /
  3 unknown.
* V03_S1821_I00000010_P3720 (seated throughout): ``sitting``, confidence 1.0,
  all 13 clips sitting. V03_S1821_I00000010_P3624 (sits, perches, sits):
  ``mixed`` with 3 transitions (clips 0 and 10 mixed, 1-4 and 11-12 sitting,
  5-9 standing: the perched block).
* Annotate-side cost: :func:`classify_bins` on 28.8M bins with a per-bin rule
  array takes 1.8 s, :func:`aggregate_groups` over 1.03M groups 1.4 s
  (integer groups) to 2.2 s (string clip ids), peak RSS 2.9 GB.

Documented limitations
----------------------
Perching on a high stool with extended legs reads as standing on every leg
signal (V03 P3624's perched block: hip 125-167°, kb 0.51-0.61). 'standing'
means upright with legs extended. The thresholds were fitted on file-level
medians; bin- and clip-level accuracy was never labelled.

Deviations from the module contract (and why)
---------------------------------------------
* ``smplh_anamorphic == 'severe'`` makes every *measure* NaN (so every bin
  and clip is unobserved/NaN), but ``hip_observable``/``ankle_observable``
  keep their visibility meaning. The registry defines
  ``lower_body_observed_frac`` as the share of frames with hips and knees
  visible; reporting 0 for a severe file whose legs are in shot would say the
  legs are out of frame, which is false. Posture on those files is still
  never measured (rule ``not_measurable_anamorphic``).
* ``posture_frames`` and ``clip_posture_measures`` take an optional ``fps``:
  the registry makes ``lower_body_observed_frac`` NA for clips shorter than
  ``round(2*fps)`` frames, which cannot be decided without the rate
  (released rates include 44.9 and 47.95 fps, so 60 frames is not universal).
  If neither supplies it, ``clip_posture_measures`` warns and reports the
  observed share for every non-empty clip rather than raising (the scan's clip
  loop has no per-file error handler, so a raise would lose the shard).
* Only ``lower_body_observed_frac`` is NA for short clips (the registry gives
  the other three clip measures their own 50%-observable NA rule and no
  short-clip rule).
* The mean over sides is a NaN-skipping mean (as the v0 detector's), and a
  side's ``knee_between`` is NaN when its ankle is not below the pelvis along
  ``-u`` (drop <= 1 µm), rather than dividing by ~0.
* :func:`posture_rule` takes ``measured=`` to emit ``not_measured``;
  :func:`classify_bins` can return int8 codes (``as_codes=True``);
  :func:`aggregate_states`/:func:`aggregate_groups` take ``prefix=`` (e.g.
  ``'recording_'``) and accept codes or strings.
* ``smplh:is_valid`` is not consulted (not in the contract). Its median valid
  fraction is >= 0.994 in every vendor, so against the 50%-of-bin gate it
  cannot move a bin.
"""

from __future__ import annotations

import math
import warnings
from dataclasses import asdict, dataclass, fields
from typing import Any, Mapping

import numpy as np
import pandas as pd

from .smplh_kinematics import NECK, PELVIS

# SMPL-H kinematic-tree indices of the legs (left, right).
SMPLH_HIPS = (1, 2)
SMPLH_KNEES = (4, 5)
SMPLH_ANKLES = (7, 8)
# Released COCO-WholeBody body indices (left, right).
COCO_HIPS = (11, 12)
COCO_KNEES = (13, 14)
COCO_ANKLES = (15, 16)

ANAMORPHIC_CLASSES = ("none", "mild", "severe")

#: Bin states, in code order. ``classify_bins(..., as_codes=True)`` returns
#: these indices as int8.
STATES = ("standing", "sitting", "unclear", "unobserved")
STANDING, SITTING, UNCLEAR, UNOBSERVED = 0, 1, 2, 3

RULE_V03 = "v03_hip"
RULE_V00 = "v00_knee"
RULE_UNVALIDATED = "unvalidated_hip_knee"
RULE_ANAMORPHIC = "not_measurable_anamorphic"
RULE_NOT_MEASURED = "not_measured"
RULES = (RULE_V03, RULE_V00, RULE_UNVALIDATED, RULE_ANAMORPHIC, RULE_NOT_MEASURED)

#: Aggregate keys without a level prefix, in output order.
AGGREGATE_KEYS = (
    "posture", "posture_confidence", "posture_standing_frac", "posture_sitting_frac",
    "posture_unclear_frac", "posture_unobserved_frac", "posture_transitions",
)
BIN_COLUMNS = (
    "bin_index", "hip_flexion_deg", "knee_between", "shin_inverted_frac",
    "hip_observed_frac", "ankle_observed_frac",
)

#: A bin/clip/recording value needs this share of its frames observable.
MIN_OBSERVABLE_SHARE = 0.5
#: Ankle drop below which knee_between is undefined, metres.
_MIN_ANKLE_DROP_M = 1e-6


# =============================================================================
# scan side
# =============================================================================
@dataclass
class PostureFrames:
    """Per-frame leg geometry for one recording, on the released pose grid."""

    hip_flexion: np.ndarray  # (T,) deg; NaN where not hip-observable (or severe)
    knee_between: np.ndarray  # (T,); NaN where not ankle-observable (or severe)
    shin_inverted: np.ndarray  # (T,) 1.0/0.0; NaN where not ankle-observable (or severe)
    hip_observable: np.ndarray  # (T,) bool: both hips AND both knees visible
    ankle_observable: np.ndarray  # (T,) bool: both knees AND both ankles visible
    smplh_anamorphic: str = "none"
    fps: float | None = None

    @property
    def n_frames(self) -> int:
        return int(len(self.hip_flexion))


def _unit(vectors: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vectors, axis=-1, keepdims=True)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(norm > 1e-9, vectors / np.where(norm > 1e-9, norm, 1.0), np.nan)


def _nanmean_sides(stack: np.ndarray) -> np.ndarray:
    """Mean over axis 0 skipping NaN, NaN where every side is NaN (no warning)."""

    finite = np.isfinite(stack)
    count = finite.sum(axis=0)
    total = np.where(finite, stack, 0.0).sum(axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(count > 0, total / np.maximum(count, 1), np.nan)


def posture_frames(
    joints_pelvis: np.ndarray,
    visible: np.ndarray,
    *,
    smplh_anamorphic: str,
    fps: float | None = None,
) -> PostureFrames:
    """Per-frame hip flexion, knee position and shin inversion, gated on visibility.

    ``joints_pelvis`` is ``(T, 52, 3)`` (or more joints) pelvis-frame FK
    positions in metres — ``gesture.GestureTracks.joints``. ``visible`` is the
    ``(T, 133)`` COCO-WholeBody visibility mask from ``framing.visible_mask``.
    ``fps`` is only stored, for the short-clip rule of
    :func:`clip_posture_measures`.
    """

    joints = np.asarray(joints_pelvis, dtype=np.float64)
    if joints.ndim != 3 or joints.shape[1] <= max(NECK, *SMPLH_ANKLES) or joints.shape[2] != 3:
        raise ValueError(f"joints_pelvis must be (T, 52, 3), got {joints.shape}")
    frames = len(joints)
    vis = np.asarray(visible)
    if vis.ndim != 2 or vis.shape[0] != frames or vis.shape[1] <= max(COCO_ANKLES):
        raise ValueError(f"visible must be (T, 133) aligned with the joints, got {vis.shape}")
    vis = vis.astype(bool)
    anamorphic = str(smplh_anamorphic)
    if anamorphic not in ANAMORPHIC_CLASSES:
        raise ValueError(f"smplh_anamorphic must be one of {ANAMORPHIC_CLASSES}, got {anamorphic!r}")

    knees_visible = vis[:, COCO_KNEES[0]] & vis[:, COCO_KNEES[1]]
    hip_observable = vis[:, COCO_HIPS[0]] & vis[:, COCO_HIPS[1]] & knees_visible
    ankle_observable = knees_visible & vis[:, COCO_ANKLES[0]] & vis[:, COCO_ANKLES[1]]

    pelvis = joints[:, PELVIS]
    up = _unit(joints[:, NECK] - pelvis)
    down = -up
    hip_sides, between_sides, shin_up = [], [], []
    for hip_j, knee_j, ankle_j in zip(SMPLH_HIPS, SMPLH_KNEES, SMPLH_ANKLES):
        hip, knee, ankle = joints[:, hip_j], joints[:, knee_j], joints[:, ankle_j]
        cosine = np.sum(up * _unit(knee - hip), axis=-1)
        hip_sides.append(np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0))))
        knee_drop = np.sum((knee - pelvis) * down, axis=-1)
        ankle_drop = np.sum((ankle - pelvis) * down, axis=-1)
        # An ankle that is not below the pelvis is not a posture this ratio
        # describes; leave it undefined instead of dividing by ~0.
        with np.errstate(invalid="ignore", divide="ignore"):
            between_sides.append(
                np.where(ankle_drop > _MIN_ANKLE_DROP_M,
                         knee_drop / np.where(ankle_drop > _MIN_ANKLE_DROP_M, ankle_drop, 1.0),
                         np.nan)
            )
        shin_up.append(np.sum((ankle - knee) * down, axis=-1))
    hip_flexion = _nanmean_sides(np.stack(hip_sides))
    knee_between = _nanmean_sides(np.stack(between_sides))
    shin = np.stack(shin_up)
    shin_inverted = np.where(np.isfinite(shin).all(axis=0), ((shin < 0).all(axis=0)).astype(np.float64), np.nan)

    measurable = anamorphic != "severe"
    hip_ok = hip_observable & measurable
    ankle_ok = ankle_observable & measurable
    return PostureFrames(
        hip_flexion=np.where(hip_ok, hip_flexion, np.nan),
        knee_between=np.where(ankle_ok, knee_between, np.nan),
        shin_inverted=np.where(ankle_ok, shin_inverted, np.nan),
        hip_observable=hip_observable,
        ankle_observable=ankle_observable,
        smplh_anamorphic=anamorphic,
        fps=None if fps is None else float(fps),
    )


def bin_edges(n_frames: int, fps: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``(bin_index, start, stop)`` of the 1-s bins ``[round(b*fps), round((b+1)*fps)) ∩ [0, T)``.

    ``np.rint`` rounds half to even on the same float product as Python's
    ``round``, so bin ``30k`` starts exactly where ``clips.clip_grid`` starts
    clip ``k`` of a 30-s grid: bins nest in clips for any integer clip length.
    """

    if not fps > 0:
        raise ValueError("fps must be positive")
    if n_frames <= 0:
        empty = np.zeros(0, dtype=np.int64)
        return empty, empty, empty
    count = int(math.ceil(n_frames / float(fps))) + 2
    index = np.arange(count + 1, dtype=np.int64)
    edges = np.rint(index * float(fps)).astype(np.int64)
    starts = edges[:-1]
    stops = np.minimum(edges[1:], n_frames)
    keep = (starts < n_frames) & (stops > starts)
    return index[:-1][keep], starts[keep], stops[keep]


def bin_clip_index(bin_index: np.ndarray, clip_seconds: float) -> np.ndarray:
    """Clip (on a ``clip_seconds`` grid) holding each 1-s bin, by the bin's start time.

    Exact nesting for integer ``clip_seconds``; for a fractional length a bin
    straddling a clip boundary is assigned to the clip its start falls in.
    """

    if not clip_seconds > 0:
        raise ValueError("clip_seconds must be positive")
    index = np.asarray(bin_index, dtype=np.float64)
    return np.floor(index / float(clip_seconds) + 1e-9).astype(np.int64)


def _binned(values: np.ndarray, starts: np.ndarray, stops: np.ndarray) -> np.ndarray:
    """``(bins, width)`` matrix of ``values`` per bin, NaN-padded past each bin's end."""

    width = int((stops - starts).max())
    index = starts[:, None] + np.arange(width, dtype=np.int64)[None, :]
    inside = index < stops[:, None]
    matrix = np.asarray(values, dtype=np.float64)[np.where(inside, index, 0)]
    matrix[~inside] = np.nan
    return matrix


def _gated_reduce(matrix: np.ndarray, length: np.ndarray, how: str) -> np.ndarray:
    """Row median/mean over finite entries when they cover >= 50% of the row's frames."""

    finite = np.isfinite(matrix)
    count = finite.sum(axis=1)
    ok = (count > 0) & (count >= MIN_OBSERVABLE_SHARE * length)
    out = np.full(len(matrix), np.nan)
    if ok.any():
        rows = matrix[ok]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            out[ok] = np.nanmedian(rows, axis=1) if how == "median" else np.nanmean(rows, axis=1)
    return out


def posture_bins(frames: PostureFrames, fps: float) -> pd.DataFrame:
    """One row per 1-s bin: the compact, threshold-free posture shard.

    Columns (float32 except ``bin_index`` int32): ``hip_flexion_deg`` (median over
    hip-observable frames when they are >= 50% of the bin, else NaN),
    ``knee_between`` and ``shin_inverted_frac`` (same over ankle-observable
    frames), ``hip_observed_frac`` and ``ankle_observed_frac`` (share of the
    bin's frames observable). The last bin may be shorter than 1 s.
    """

    index, starts, stops = bin_edges(frames.n_frames, fps)
    if index.size == 0:
        return pd.DataFrame({
            "bin_index": np.zeros(0, np.int32),
            **{name: np.zeros(0, np.float32) for name in BIN_COLUMNS[1:]},
        })
    length = (stops - starts).astype(np.float64)
    columns: dict[str, np.ndarray] = {"bin_index": index.astype(np.int32)}
    columns["hip_flexion_deg"] = _gated_reduce(_binned(frames.hip_flexion, starts, stops), length, "median")
    columns["knee_between"] = _gated_reduce(_binned(frames.knee_between, starts, stops), length, "median")
    columns["shin_inverted_frac"] = _gated_reduce(_binned(frames.shin_inverted, starts, stops), length, "mean")
    for name, mask in (("hip_observed_frac", frames.hip_observable),
                       ("ankle_observed_frac", frames.ankle_observable)):
        cumulative = np.concatenate(([0], np.cumsum(np.asarray(mask, dtype=np.int64))))
        columns[name] = (cumulative[stops] - cumulative[starts]) / length
    return pd.DataFrame({
        name: (values if name == "bin_index" else np.asarray(values, dtype=np.float32))
        for name, values in columns.items()
    })


def _span_measure(values: np.ndarray, length: int, how: str) -> float:
    finite = values[np.isfinite(values)]
    if length <= 0 or finite.size == 0 or finite.size < MIN_OBSERVABLE_SHARE * length:
        return float("nan")
    return float(np.median(finite) if how == "median" else np.mean(finite))


def clip_posture_measures(
    frames: PostureFrames, start: int, stop: int, *, fps: float | None = None
) -> dict[str, float]:
    """Continuous clip posture measures over the half-open frame range ``[start, stop)``.

    ``hip_flexion_deg_p50``/``knee_between_p50``/``shin_inverted_frac`` are NaN
    unless >= 50% of the clip's frames are hip-/ankle-observable (and never
    measured on severe anamorphic files); ``lower_body_observed_frac`` is NaN
    only for clips shorter than ``round(2*fps)`` frames.
    """

    rate = fps if fps is not None else frames.fps
    known_rate = rate is not None and float(rate) > 0
    if not known_rate:
        # Never raise here: the scan's clip loop sits outside its per-file
        # error handler, so an exception would lose the whole shard.
        warnings.warn(
            "clip_posture_measures has no frame rate (pass fps= here or to "
            "posture_frames); the < 2 s NA rule for lower_body_observed_frac "
            "is not applied",
            RuntimeWarning,
            stacklevel=2,
        )
    total = frames.n_frames
    lo, hi = max(0, int(start)), min(total, int(stop))
    length = max(0, hi - lo)
    short = known_rate and length < int(round(2.0 * float(rate)))
    observed = float(np.mean(frames.hip_observable[lo:hi])) if length > 0 else float("nan")
    return {
        "hip_flexion_deg_p50": _span_measure(frames.hip_flexion[lo:hi], length, "median"),
        "knee_between_p50": _span_measure(frames.knee_between[lo:hi], length, "median"),
        "shin_inverted_frac": _span_measure(frames.shin_inverted[lo:hi], length, "mean"),
        "lower_body_observed_frac": float("nan") if short else observed,
    }


def recording_posture_measures(frames: PostureFrames) -> dict[str, float]:
    """Whole-recording medians; NaN under the same 50%-observable rule (and for T = 0)."""

    total = frames.n_frames
    return {
        "recording_hip_flexion_deg_p50": _span_measure(frames.hip_flexion, total, "median"),
        "recording_knee_between_p50": _span_measure(frames.knee_between, total, "median"),
        "recording_lower_body_observed_frac": (
            float(np.mean(frames.hip_observable)) if total > 0 else float("nan")
        ),
    }


# =============================================================================
# annotate side
# =============================================================================
@dataclass(frozen=True)
class PostureRules:
    """Every posture threshold, in one config-overridable place (see module docstring)."""

    v03_sitting_below_deg: float = 136.0
    v03_standing_above_deg: float = 152.0
    v00_sitting_below: float = 0.30
    v00_standing_at_or_above: float = 0.48
    other_standing_hip_at_or_above: float = 135.0
    other_sitting_hip_below: float = 120.0
    other_standing_knee_at_or_above: float = 0.48
    other_sitting_knee_below: float = 0.40
    #: A sustained run: this many decided bins in a row (after dropping undecided).
    min_run_bins: int = 5
    #: standing/sitting when that state holds this share of decided bins.
    majority: float = 0.8
    #: unknown when observed bins are below this share of all bins ...
    min_observed_share: float = 0.5
    #: ... or fewer than this many.
    min_observed_bins: int = 3
    #: unclear when decided bins are below this share of observed bins.
    min_decided_share: float = 0.5

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_mapping(cls, overrides: Mapping[str, Any] | None) -> "PostureRules":
        """Defaults with ``overrides`` applied; unknown keys are an error, not ignored."""

        overrides = dict(overrides or {})
        known = {f.name for f in fields(cls)}
        unknown = set(overrides) - known
        if unknown:
            raise ValueError(f"unknown posture rule keys {sorted(unknown)}")
        return cls(**overrides)


def _vendor(vendor: str) -> str:
    text = str(vendor)
    return text if text.startswith("V") else "V" + text.zfill(2)


def posture_rule(vendor: str, smplh_anamorphic: str, *, measured: bool = True) -> tuple[str, str]:
    """``(posture_rule, posture_rule_evidence)`` for a recording.

    ``measured=False`` (no pose arrays) gives ``not_measured``. Severe anamorphic
    wins over the vendor; an unrecognised vendor is not measured.
    """

    if not measured:
        return RULE_NOT_MEASURED, "not_applicable"
    if str(smplh_anamorphic) == "severe":
        return RULE_ANAMORPHIC, "not_applicable"
    code = _vendor(vendor)
    if code == "V03":
        return RULE_V03, "v03_file_labels_164"
    if code == "V00":
        return RULE_V00, "v00_file_labels_66_selection_biased"
    if code in ("V01", "V02"):
        return RULE_UNVALIDATED, "unlabelled"
    return RULE_NOT_MEASURED, "not_applicable"


def classify_bins(
    rule: np.ndarray | str,
    hip: np.ndarray,
    knee: np.ndarray,
    rules: PostureRules | None = None,
    *,
    as_codes: bool = False,
) -> np.ndarray:
    """Bin states under ``rule`` (one string, or one per bin), vectorised.

    Returns an object array of :data:`STATES` strings, or int8 codes
    (``STANDING``..``UNOBSERVED``) with ``as_codes=True`` — the codes are what
    a 29M-bin corpus pass should carry.
    """

    rules = rules or PostureRules()
    hip = np.asarray(hip, dtype=np.float64).reshape(-1)
    knee = np.asarray(knee, dtype=np.float64).reshape(-1)
    if hip.shape != knee.shape:
        raise ValueError("hip and knee must have the same length")
    count = hip.size
    if isinstance(rule, str):
        rule_codes = np.zeros(count, dtype=np.int64)
        rule_names = [rule]
    else:
        rule_codes, uniques = pd.factorize(np.asarray(rule, dtype=object).reshape(-1)
                                           if not isinstance(rule, (pd.Series, pd.Categorical))
                                           else rule)
        if len(rule_codes) != count:
            raise ValueError("rule must be a string or one value per bin")
        if (rule_codes < 0).any():
            raise ValueError("rule has missing values")
        rule_names = [str(name) for name in uniques]
    unknown = set(rule_names) - set(RULES)
    if unknown:
        raise ValueError(f"unknown posture rules {sorted(unknown)}")

    out = np.full(count, UNOBSERVED, dtype=np.int8)
    hip_ok = np.isfinite(hip)
    knee_ok = np.isfinite(knee)
    for code, name in enumerate(rule_names):
        where = rule_codes == code
        if name == RULE_V03:
            observed = where & hip_ok
            sitting = observed & (hip < rules.v03_sitting_below_deg)
            standing = observed & (hip > rules.v03_standing_above_deg)
        elif name == RULE_V00:
            observed = where & knee_ok
            sitting = observed & (knee < rules.v00_sitting_below)
            standing = observed & (knee >= rules.v00_standing_at_or_above)
        elif name == RULE_UNVALIDATED:
            observed = where & hip_ok
            hip_standing = hip >= rules.other_standing_hip_at_or_above
            hip_sitting = hip < rules.other_sitting_hip_below
            standing = observed & hip_standing & (~knee_ok | (knee >= rules.other_standing_knee_at_or_above))
            sitting = observed & hip_sitting & (~knee_ok | (knee < rules.other_sitting_knee_below))
        else:  # not_measurable_anamorphic, not_measured
            continue
        out[observed] = UNCLEAR
        out[standing] = STANDING
        out[sitting] = SITTING
    if as_codes:
        return out
    return np.asarray(STATES, dtype=object)[out]


def _state_codes(states: Any) -> np.ndarray:
    """int8 state codes from codes, strings, or a categorical; unknown values raise."""

    if isinstance(states, pd.Series):
        values = states.array
    else:
        values = states
    if isinstance(values, pd.Categorical) or isinstance(getattr(values, "dtype", None), pd.CategoricalDtype):
        categorical = pd.Categorical(values)
        mapping = np.array([STATES.index(str(c)) if str(c) in STATES else -1
                            for c in categorical.categories], dtype=np.int64)
        raw = categorical.codes.astype(np.int64)
        codes = np.where(raw >= 0, mapping[np.maximum(raw, 0)], -1)
    else:
        array = np.asarray(values)
        if array.dtype.kind in "iu":
            codes = array.astype(np.int64)
            if codes.size and (codes.min() < 0 or codes.max() > UNOBSERVED):
                raise ValueError("state codes must be in 0..3")
            return codes.astype(np.int8).reshape(-1)
        codes = pd.Categorical(array.reshape(-1), categories=list(STATES)).codes.astype(np.int64)
    if codes.size and (codes < 0).any():
        raise ValueError(f"states must be one of {STATES}")
    return codes.astype(np.int8).reshape(-1)


def _sustained_transitions(codes: np.ndarray, min_run_bins: int) -> int:
    """Reference (per-group) transition count: drop undecided, RLE, keep long runs."""

    decided = [int(c) for c in codes if c in (STANDING, SITTING)]
    runs: list[list[int]] = []
    for state in decided:
        if runs and runs[-1][0] == state:
            runs[-1][1] += 1
        else:
            runs.append([state, 1])
    kept = [state for state, length in runs if length >= min_run_bins]
    return sum(1 for previous, current in zip(kept, kept[1:]) if previous != current)


def aggregate_states(states: Any, rules: PostureRules | None = None, *, prefix: str = "") -> dict[str, Any]:
    """Label, confidence, shares and transitions of one clip's/recording's bin states.

    Keys are :data:`AGGREGATE_KEYS` with ``prefix`` prepended. With no bins at
    all (an unmeasured recording) the shares are NaN and transitions None.
    """

    rules = rules or PostureRules()
    codes = _state_codes(states)
    n = int(codes.size)
    counts = np.bincount(codes.astype(np.int64), minlength=4)
    standing, sitting, unclear, unobserved = (int(c) for c in counts[:4])
    observed = standing + sitting + unclear
    decided = standing + sitting
    transitions = _sustained_transitions(codes, rules.min_run_bins) if n else None

    confidence = float("nan")
    if n == 0 or observed < rules.min_observed_share * n or observed < rules.min_observed_bins:
        label = "unknown"
    elif decided < rules.min_decided_share * observed:
        label = "unclear"
    elif transitions >= 1:
        label = "mixed"
        confidence = decided / n
    elif standing / decided >= rules.majority:
        label = "standing"
        confidence = standing / n
    elif sitting / decided >= rules.majority:
        label = "sitting"
        confidence = sitting / n
    else:
        label = "unclear"

    def share(count: int) -> float:
        return count / n if n else float("nan")

    values = (label, confidence, share(standing), share(sitting), share(unclear),
              share(unobserved), transitions)
    return {prefix + key: value for key, value in zip(AGGREGATE_KEYS, values)}


def _empty_groups(group_col: str, prefix: str) -> pd.DataFrame:
    frame = pd.DataFrame({group_col: pd.Series([], dtype=object)})
    frame[prefix + "posture"] = pd.Series([], dtype=object)
    for key in AGGREGATE_KEYS[1:6]:
        frame[prefix + key] = pd.Series([], dtype=np.float32)
    frame[prefix + "posture_transitions"] = pd.Series([], dtype=np.int16)
    return frame


def aggregate_groups(
    bins: pd.DataFrame,
    group_col: str,
    state_col: str,
    rules: PostureRules | None = None,
    *,
    order_col: str | None = "bin_index",
    prefix: str = "",
) -> pd.DataFrame:
    """:func:`aggregate_states` for every group at once, with no per-group Python.

    Bins are ordered within a group by ``order_col`` when that column exists
    (else by row order). Returns one row per group, in order of first
    appearance: ``group_col`` then the aggregate keys (posture object strings,
    shares/confidence float32, transitions int16). ~29M bins / ~1M groups is
    a sort, a few bincounts and one pass of run-length encoding.
    """

    rules = rules or PostureRules()
    if len(bins) == 0:
        return _empty_groups(group_col, prefix)
    groups, uniques = pd.factorize(bins[group_col], sort=False)
    if (groups < 0).any():
        raise ValueError(f"{group_col} has missing values")
    groups = groups.astype(np.int64)
    states = _state_codes(bins[state_col]).astype(np.int64)
    n_groups = len(uniques)

    if order_col is not None and order_col in bins.columns:
        position = np.asarray(bins[order_col].to_numpy(), dtype=np.int64)
        in_order = bool(np.all((groups[1:] > groups[:-1])
                               | ((groups[1:] == groups[:-1]) & (position[1:] >= position[:-1]))))
        order = None if in_order else np.lexsort((position, groups))
    else:
        order = None if bool(np.all(groups[1:] >= groups[:-1])) else np.argsort(groups, kind="stable")
    if order is not None:
        groups = groups[order]
        states = states[order]

    counts = np.bincount(groups * 4 + states, minlength=n_groups * 4).reshape(n_groups, 4)
    standing, sitting, unclear, unobserved = (counts[:, k] for k in range(4))
    n = counts.sum(axis=1)

    # Sustained runs over the decided bins only, bridged across undecided ones.
    transitions = np.zeros(n_groups, dtype=np.int64)
    decided_mask = states <= SITTING
    g = groups[decided_mask]
    s = states[decided_mask]
    if g.size:
        new_run = np.ones(g.size, dtype=bool)
        new_run[1:] = (g[1:] != g[:-1]) | (s[1:] != s[:-1])
        run_starts = np.flatnonzero(new_run)
        run_length = np.diff(np.append(run_starts, g.size))
        keep = run_length >= rules.min_run_bins
        kept_group = g[run_starts][keep]
        kept_state = s[run_starts][keep]
        if kept_group.size > 1:
            change = (kept_group[1:] == kept_group[:-1]) & (kept_state[1:] != kept_state[:-1])
            transitions = np.bincount(kept_group[1:][change], minlength=n_groups)

    observed = standing + sitting + unclear
    decided = standing + sitting
    total = n.astype(np.float64)
    unknown = (observed < rules.min_observed_share * total) | (observed < rules.min_observed_bins)
    low_decided = ~unknown & (decided < rules.min_decided_share * observed)
    open_ = ~unknown & ~low_decided
    with np.errstate(invalid="ignore", divide="ignore"):
        standing_share = standing / decided
        sitting_share = sitting / decided
    mixed = open_ & (transitions >= 1)
    is_standing = open_ & ~mixed & (standing_share >= rules.majority)
    is_sitting = open_ & ~mixed & ~is_standing & (sitting_share >= rules.majority)

    labels = np.full(n_groups, "unclear", dtype=object)
    labels[unknown] = "unknown"
    labels[mixed] = "mixed"
    labels[is_standing] = "standing"
    labels[is_sitting] = "sitting"
    confidence = np.full(n_groups, np.nan)
    confidence[is_standing] = standing[is_standing] / total[is_standing]
    confidence[is_sitting] = sitting[is_sitting] / total[is_sitting]
    confidence[mixed] = decided[mixed] / total[mixed]

    out = pd.DataFrame({group_col: np.asarray(uniques)})
    out[prefix + "posture"] = labels
    out[prefix + "posture_confidence"] = confidence.astype(np.float32)
    for key, count in (("posture_standing_frac", standing), ("posture_sitting_frac", sitting),
                       ("posture_unclear_frac", unclear), ("posture_unobserved_frac", unobserved)):
        out[prefix + key] = (count / total).astype(np.float32)
    out[prefix + "posture_transitions"] = transitions.astype(np.int16)
    return out

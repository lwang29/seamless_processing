"""File-level detectors for the three V00 failure modes the reviewer named.

Scope is deliberately narrow. The reviewer retracted the earlier corpus-wide
vocabulary after seeing the V00 gallery: roll, audio/video desync, glitchy
tracking, unnatural hand motion, and noisy body pose are all out of scope, and
no detector here attempts them. What remains:

* **FM1 sitting** — from leg geometry, never from how much frame the person
  fills. Seated participants were often recorded with the camera moved closer,
  so occupancy and box size carry no signal. See :func:`sitting_measures_2d` and
  :func:`posture_angles`.
* **FM2 framing** — no part of the participant's body leaves the raster on any
  frame, and SMPL-H is valid on every frame.
* **FM3 static hands** — the wrists hold one position for most of the recording.

Every threshold is a parameter with no default baked into the measurement: the
functions return continuous quantities, and flagging happens in one place
(:func:`apply_thresholds`) so the cut points can move without recomputation.

All three are **file-level**. FM2's "every frame" therefore means every frame of
the whole recording, which is much stricter than the same rule over a short
window; the windowed variant is computed alongside so the difference in cost is
visible rather than assumed.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np

from .m3_features import usable_keypoint_mask


# SMPL-H kinematic-tree indices (see scripts.smplh_fk.SMPLH_JOINT_NAMES).
SMPLH_PELVIS = 0
SMPLH_NECK = 12
SMPLH_HIP = {"left": 1, "right": 2}
SMPLH_KNEE = {"left": 4, "right": 5}
SMPLH_ANKLE = {"left": 7, "right": 8}
SMPLH_WRIST = {"left": 20, "right": 21}
SMPLH_SHOULDER = {"left": 16, "right": 17}

# Joints tested for the in-frame rule: the SMPL-H kinematic body including its
# ankle-attached foot joints, both articulated hands, and the fingertips. The
# auxiliary face landmarks (52-56) and toe/heel extras (57-62) are excluded
# because M-4 never validated them — a toe landmark with a 3,355 px reprojection
# error would flag framing failures that are really projection failures.
IN_FRAME_JOINTS: tuple[int, ...] = tuple(range(52)) + tuple(range(63, 73))

# Released COCO-WholeBody groups used by the 2D cross-check.
COCO_BODY_FEET = tuple(range(0, 23))
COCO_HANDS = tuple(range(91, 133))
COCO_SHOULDERS = (5, 6)
COCO_WRISTS = {"left": 9, "right": 10}


@dataclass(frozen=True)
class Thresholds:
    """Every cut point in one place, so re-tuning needs no recomputation."""

    # FM1, from SMPL-H forward kinematics. Two rounds of reviewer inspection have
    # now cut this down to a single measure plus one rare-posture add-on.
    #
    # Retired in Round 4: the 2D ratios. leg_over_torso < 0.85 scored 0/15 and
    # knee_spread > 0.90 missed all four seated participants it was not selected
    # on; both encode body proportion rather than posture, because all sixteen
    # SMPL-H betas are zero so FK limbs are identical across files while 2D limbs
    # are not.
    #
    # Retired in Round 5: hip flexion. On 66 hand-labelled files every one of the
    # four it flags that knee_between does not is a **standing** person, and it
    # costs 29 extra files corpus-wide with no validated hit among them. It is
    # still measured and reported; it just no longer decides.
    #
    # knee_between_torso sits on a plateau: any cut in 0.41-0.44 gives the same
    # 18/19 sitting and 46/47 standing on the labels. 0.43 is the midpoint.
    sitting_knee_between_torso: float = 0.43
    # Sign change, not a magnitude: negative means the ankle is *above* the
    # knee, the feet-on-a-stool-rung posture. Fires on one file in 4,000 of V00
    # and the reviewer confirmed that file is seated.
    sitting_shin_verticality: float = 0.0
    # Round 11: on V03 the add-on is pure cost. Across 149 hand labels it flags 8
    # files that hip flexion does not, and **none of the eight is seated** -- they
    # are the ankles-out-of-frame case, where the fit puts the feet somewhere
    # arbitrary and the shin comes out inverted. Only 6 of the 18 standing files
    # lost at the 146 cut had both ankles in shot on every frame. Dropping it for
    # V03 halves the standing loss, from 18 of 96 to 10, and costs no sitter.
    sitting_shin_addon_by_vendor: Mapping[str, bool] = field(
        default_factory=lambda: {"00": True, "01": True, "02": True, "03": False}
    )
    # Unit level. The reviewer observed that a participant either stands in all
    # of their recordings or sits in all of them; measurement says the scope of
    # that is the recording *session*, not the person. See
    # :func:`session_sitting_verdicts`. One clause only: a majority of the unit's
    # files.
    sitting_seated_file_frac: float = 0.50
    # Round 10: that observation is true of V00 and false of V03. Share of
    # multi-file participant-sessions where some files read seated and others do
    # not: **V00 1.1%** (12 of 1,072), **V03 16.2%** (210 of 1,293), and in the
    # mixed V03 units the seated share has a median of exactly 0.50 -- so the
    # majority vote is close to a coin toss. On 108 labelled V03 files it is
    # worse than the per-file verdict on both axes at once (35 of 46 sitters and
    # 13 of 62 standing lost, against 40 and 7), which is how a seated clip
    # reached the reviewer having passed FM1: its own file read seated and the
    # other thirteen files in its session outvoted it.
    sitting_unit_scope_by_vendor: Mapping[str, str] = field(
        default_factory=lambda: {"00": "session", "01": "session",
                                 "02": "session", "03": "file"}
    )
    # A unit that is not seated but whose best file comes this close to the cut
    # is reported as near the boundary, so moving the threshold has a visible
    # worklist behind it rather than only a count.
    sitting_near_cut_knee_between_torso: float = 0.06
    # Round 9: FM1 is per-vendor in *which measure it reads*, not only where the
    # cut sits. Each entry is ``(measure, cut)``, or ``None`` to switch the check
    # off for that vendor.
    #
    # V00 keeps knee position at 0.43, validated on 66 hand labels.
    #
    # V03 uses **hip flexion**, which Round 5 retired on V00 evidence and which
    # turns out to be the right measure here. On 90 V03 hand labels, evenly split:
    #
    #     rule                sitters caught   standing lost
    #     knee_between < 0.54     41 / 45         12 / 45
    #     hip_flexion  < 146      44 / 45          4 / 45
    #
    # Better on both axes, and it rejects 26% of V03 against knee's 33%. It also
    # wins *inside* the band where knee position is ambiguous -- 0.50 to 0.63,
    # where 9 sitters and 38 standing files overlap -- at AUC 0.977, which is the
    # one comparison the label draw cannot have biased, since the bands were
    # drawn on knee position and not on this. Standing files bottom out at 136
    # degrees and seated ones top out at 152, so 146 is inside a real gap rather
    # than on a knife edge; it is chosen toward recall, as asked.
    #
    # V01 and V02 switch FM1 off. On 1,379 square-pixel V01 files it fires on
    # *none*, and V01's anamorphic rasters -- where it fires on 92% -- are
    # excluded by FM0. On V02, 181 of its 182 firings are the shin add-on on
    # files whose ankles are out of frame, every one of which the reviewer
    # confirmed standing.
    sitting_by_vendor: Mapping[str, tuple[str, float] | None] = field(
        default_factory=lambda: {
            "00": ("knee_between", 0.43),
            "01": None,
            "02": None,
            "03": ("hip_flexion", 146.0),
        }
    )
    # FM2 as of Round 5: the released per-frame SMPL-H validity flag must hold on
    # this fraction of frames. 1.0 is "valid on every frame". The geometric
    # in-frame measures below are still computed and reported, but no longer
    # decide anything — the PI's position is that a body part leaving the raster
    # is only a problem when it degrades the fit, and the fit has its own flag.
    smplh_valid_required_frac: float = 1.0
    # FM2 geometry, retained as measurement-only knobs.
    in_frame_margin_px: float = 0.0
    in_frame_required_frac: float = 1.0
    min_violation_run_frames: int = 1
    # FM4, new in Round 8: the released audio carries no voice. Both halves are
    # required, because either alone has false positives -- a genuinely quiet
    # recording is not broken, and a flat spectrum in a loud file is a fan or an
    # air conditioner rather than a dead microphone. Together they caught all
    # four files the reviewer confirmed dead and none of the 18 that are merely
    # unpleasant, and fired on 0 of 300 V00 files.
    # Round 14: FM4 gates on the *dynamics* of the level envelope, not on the
    # level itself. Static and buzz sit at the same level whether anyone is
    # speaking or not; a track carrying a voice does not. On 48 hand-labelled
    # files the two classes are cleanly separated -- unusable 0.20-1.35 dB,
    # usable 1.70-69.04 -- and 1.5 dB catches 13 of 13 with no false positive.
    #
    # This replaces the level floor, which scored 10.5% precision on those
    # labels, and supersedes the empty-VAD idea, which scored 21.7%: the VAD
    # fails on plenty of working recordings, and a participant who barely speaks
    # still has a working microphone, audibly so when their partner talks.
    audio_min_envelope_dynamics_db: float = 1.5
    # Retained as measurements only; neither decides anything now.
    audio_dead_speech_level_db: float = -55.0
    audio_dead_spectral_flatness: float = 0.05
    # Round 13: FM4 is off for V00, on the reviewer's instruction after all six of
    # its firings in a 4,000-file draw turned out to be participants who simply
    # speak rarely. The cause is that the floor is absolute while the vendors'
    # levels are not: V00's median speech level is -41.8 dB, so -55 dB sits only
    # 13.2 dB below typical and catches its 1st percentile, where on V01 the same
    # floor is 36.4 dB below the median and catches only genuinely dead tracks.
    #
    # Making the floor relative to the vendor median does not rescue it -- at 30
    # dB below it still fires on 3 of the 6 -- because "rarely speaks" and "no
    # voice at all" look alike to any whole-file average. What does separate them
    # is `audio_vad_seconds`: all six confirmed-dead files have **zero** released
    # speech, five of the six V00 false positives have 1.6 to 79.1 seconds, and
    # an empty VAD occurs in 0 of 400 random V00 files. That is a better rule and
    # it is measured and reported, but it is not wired in yet: it flags a larger
    # set than the level rule outside V00 (5.25% of V01 against 3.21%) and that
    # difference is unvalidated.
    audio_dead_by_vendor: Mapping[str, bool] = field(
        default_factory=lambda: {"00": False, "01": True, "02": True, "03": True}
    )
    # FM3: a wrist within this many shoulder widths of its own reference counts
    # as "the same position".
    static_radius_shoulder_widths: float = 0.10
    static_frac_limit: float = 0.75

    # FM0, new in Round 8: the source is unusable whatever the recording shows.
    # These are the reviewer's decisions after seeing each raster corrected, not
    # inferences, so each carries the reason it was taken.
    #
    #   2160x2160, 1920x1080 -- V01's badly anamorphic rasters. The picture and
    #     the 2D points correct cleanly but the released SMPL-H does not, because
    #     the isotropic fitted camera forced the stretch into the pose.
    #   640x480, 3840x2160  -- V03's room-camera rasters. Far-field audio the
    #     reviewer found unusable, and 640x480 is 0.3 MP besides. The reviewer's
    #     word on 3840x2160 was "leaning towards filtering them all out"; one of
    #     its files did have acceptable audio, so this is the costlier of the two
    #     calls and the easiest to reverse.
    #
    # Kept, with a crop: 1080x960 is 540x960 padded and 2180x3840 is 2160x3840
    # padded, and both fit as well as any ordinary raster. Kept as-is: 1012x1920,
    # whose 6.7% correction leaves the fit inside the normal band.
    excluded_rasters: tuple[str, ...] = ("2160x2160", "1920x1080", "640x480", "3840x2160")
    # A released annotation grid that misses the container by more than this is
    # not a rendering nuisance: the reviewer checked two such files and found the
    # SMPL-H tracking wholly unrelated to the video in both.
    max_timebase_drift_s: float = 0.5


def _angle_between(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    """Angle in degrees between two stacks of vectors, per row."""

    a = np.asarray(first, dtype=np.float64)
    b = np.asarray(second, dtype=np.float64)
    na = np.linalg.norm(a, axis=-1)
    nb = np.linalg.norm(b, axis=-1)
    with np.errstate(invalid="ignore", divide="ignore"):
        cosine = np.sum(a * b, axis=-1) / (na * nb)
    cosine = np.where((na > 1e-9) & (nb > 1e-9), cosine, np.nan)
    return np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))


def _percentiles(prefix: str, values: np.ndarray, points: Sequence[int] = (5, 25, 50, 75, 95)) -> dict[str, float]:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return {f"{prefix}_p{p}": float("nan") for p in points} | {f"{prefix}_n": 0}
    out = {f"{prefix}_p{p}": float(np.percentile(finite, p)) for p in points}
    out[f"{prefix}_n"] = int(finite.size)
    return out


def posture_angles(
    root_relative_mm: np.ndarray, frame_valid: np.ndarray | None = None
) -> dict[str, Any]:
    """FM1: seated-vs-standing geometry from SMPL-H forward kinematics.

    Hip flexion is the angle between the torso direction (pelvis to neck) and the
    thigh (hip to knee). Knee flexion is the angle at the knee between thigh and
    shin. Both are pure joint angles, so a camera moved closer to a seated
    participant changes nothing — which is exactly why frame occupancy was
    rejected as a signal.

    The decisive measure is ``fm1_knee_between_torso``: how far down the leg the
    knee sits, measured along the participant's own torso axis rather than along
    the image's vertical.

        u = unit(neck - pelvis);  ratio = (knee - pelvis)·(-u) / (ankle - pelvis)·(-u)

    Standing puts the knee near the midpoint of the hip-to-ankle drop and reads
    about 0.55; sitting lifts the knee toward hip height and reads about 0.3.
    Because the ratio is taken along the torso axis it is invariant to camera
    tilt, to distance, and — unlike anything derived from 2D limb lengths — to
    body proportion.

    That last point is what makes it the primary. Every FK quantity here is
    computed with **all sixteen betas pinned to zero**, so the thigh is 376.8 mm
    and the shin 400.6 mm in every file regardless of who is being recorded, and
    the FK output is pure pose. The 2D ratio ``fm1_leg_over_torso`` does encode
    build, which is why it flagged fifteen short-legged *standing* participants
    and no seated ones: it was measuring body proportion, not posture. See
    :func:`sitting_measures_2d` for the 2D measures, all now reported rather than
    thresholded.
    """

    joints = np.asarray(root_relative_mm, dtype=np.float64)
    if joints.ndim != 3 or joints.shape[1] < 22 or joints.shape[2] != 3:
        return {"posture_status": f"invalid_fk_shape:{joints.shape}"}
    eligible = np.isfinite(joints).all(axis=(1, 2))
    if frame_valid is not None:
        mask = np.asarray(frame_valid, dtype=bool).reshape(-1)
        if len(mask) == len(joints):
            eligible &= mask
    if not eligible.any():
        return {"posture_status": "no_eligible_frames"}

    usable = joints[eligible]
    torso = usable[:, SMPLH_NECK, :] - usable[:, SMPLH_PELVIS, :]
    result: dict[str, Any] = {
        "posture_status": "ok",
        "posture_frames_used": int(eligible.sum()),
    }
    # Unit vector along the torso, pointing from the pelvis toward the neck; its
    # negation is "down the body" in the participant's own frame.
    torso_length = np.linalg.norm(torso, axis=-1, keepdims=True)
    with np.errstate(invalid="ignore", divide="ignore"):
        down = -torso / np.where(torso_length > 1e-6, torso_length, np.nan)

    hip_all: list[np.ndarray] = []
    knee_all: list[np.ndarray] = []
    between_all: list[np.ndarray] = []
    for side in ("left", "right"):
        hip = usable[:, SMPLH_HIP[side], :]
        knee = usable[:, SMPLH_KNEE[side], :]
        ankle = usable[:, SMPLH_ANKLE[side], :]
        pelvis = usable[:, SMPLH_PELVIS, :]
        hip_flexion = _angle_between(torso, knee - hip)
        knee_flexion = _angle_between(hip - knee, ankle - knee)
        knee_drop = np.sum((knee - pelvis) * down, axis=-1)
        ankle_drop = np.sum((ankle - pelvis) * down, axis=-1)
        # A non-positive ankle drop means the ankle is not below the pelvis at
        # all, which is not a posture this ratio describes; leave it undefined
        # rather than let a near-zero denominator invent an extreme value.
        with np.errstate(invalid="ignore", divide="ignore"):
            between = knee_drop / np.where(ankle_drop > 1e-6, ankle_drop, np.nan)
        result.update(_percentiles(f"fm1_hip_flexion_{side}_deg", hip_flexion))
        result.update(_percentiles(f"fm1_knee_flexion_{side}_deg", knee_flexion))
        hip_all.append(hip_flexion)
        knee_all.append(knee_flexion)
        between_all.append(between)
    # The decision statistic is the mean of the two sides per frame, then the
    # median over frames: one crossed leg should not read as sitting, and a
    # momentary crouch should not either.
    # A frame where neither side could be measured is legitimately all-NaN, and
    # nanmean warns on it. That is expected, not a defect, and 4,000 scan logs
    # should not carry it.
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", "Mean of empty slice", RuntimeWarning)
        result.update(_percentiles("fm1_hip_flexion_deg", np.nanmean(np.stack(hip_all), axis=0)))
        result.update(_percentiles("fm1_knee_flexion_deg", np.nanmean(np.stack(knee_all), axis=0)))
        result.update(
            _percentiles("fm1_knee_between_torso", np.nanmean(np.stack(between_all), axis=0))
        )
    return result


def sitting_measures_2d(keypoints: np.ndarray) -> dict[str, Any]:
    """FM1 primary: leg geometry from the released 2D keypoints.

    The SMPL-H forward-kinematic leg angles are **not usable** for this. A
    participant confirmed by eye to be standing reads 87 degrees of knee flexion
    and 125 degrees of hip flexion, which is anatomically a deep squat. That is
    consistent with M-4, which validated body-17 and the hands but found
    structured extreme errors on the foot landmarks and never accepted the legs.
    The released COCO body-17 hips, knees, and ankles were validated, so FM1 uses
    those and :func:`posture_angles` is retained only as an uncalibrated
    secondary.

    Four scale-free ratios, all medians over the file. Measured on one confirmed
    standing file against one confirmed seated file:

    ==============================  =========  =======
    measure                          standing   seated
    ==============================  =========  =======
    ``shin_verticality``               +1.000   −0.832
    ``knee_spread_over_torso``          0.353    1.412
    ``leg_over_torso``                  1.028    0.667
    ``thigh_over_shin``                 1.309    8.345
    ==============================  =========  =======

    ``shin_verticality`` is the signed cosine of the shin against straight-down,
    and it changes sign for a participant on a stool with their feet on a rung:
    the ankle sits *above* the knee. It will not, however, catch someone on a
    chair with their feet flat on the floor, whose shin is vertical exactly as a
    standing person's is. That case is what ``knee_spread_over_torso`` and
    ``leg_over_torso`` are for. No single ratio covers every seated posture,
    which is why FM1 is a disjunction and why the gallery has to adjudicate it.
    """

    points = np.asarray(keypoints, dtype=np.float64)
    if points.ndim != 3 or points.shape[1] < 17:
        return {"fm1_2d_status": f"invalid_keypoint_shape:{points.shape}"}
    if len(points) == 0:
        return {"fm1_2d_status": "empty"}
    usable = usable_keypoint_mask(points)

    def midpoint(pair: tuple[int, int]) -> np.ndarray:
        return 0.5 * (points[:, pair[0], :2] + points[:, pair[1], :2])

    def both(pair: tuple[int, int]) -> np.ndarray:
        return usable[:, pair[0]] & usable[:, pair[1]]

    shoulders, hips = (5, 6), (11, 12)
    knees, ankles = (13, 14), (15, 16)
    shoulder, hip = midpoint(shoulders), midpoint(hips)
    knee, ankle = midpoint(knees), midpoint(ankles)
    torso = np.linalg.norm(hip - shoulder, axis=1)
    thigh = np.linalg.norm(knee - hip, axis=1)
    shin = np.linalg.norm(ankle - knee, axis=1)
    eligible = (
        both(shoulders) & both(hips) & both(knees) & both(ankles)
        & (torso > 1e-6) & (thigh > 1e-6) & (shin > 1e-6)
    )
    result: dict[str, Any] = {
        "fm1_2d_status": "ok" if eligible.any() else "no_eligible_frames",
        "fm1_2d_eligible_frac": float(eligible.mean()),
    }
    if not eligible.any():
        return result

    # y grows downwards, so a positive verticality means the lower joint really
    # is below the upper one.
    # Denominators are already known positive on eligible frames; elsewhere they
    # can be zero, so divide under a guard rather than filtering warnings.
    safe_torso = np.where(eligible, torso, np.nan)
    safe_thigh = np.where(eligible, thigh, np.nan)
    safe_shin = np.where(eligible, shin, np.nan)
    with np.errstate(invalid="ignore", divide="ignore"):
        measures = {
            "fm1_shin_verticality": (ankle[:, 1] - knee[:, 1]) / safe_shin,
            "fm1_thigh_verticality": (knee[:, 1] - hip[:, 1]) / safe_thigh,
            "fm1_knee_spread_over_torso": np.abs(
                points[:, knees[0], 0] - points[:, knees[1], 0]
            ) / safe_torso,
            "fm1_leg_over_torso": np.linalg.norm(ankle - hip, axis=1) / safe_torso,
            "fm1_thigh_over_shin": safe_thigh / safe_shin,
            "fm1_hip_ankle_vertical_over_torso": (ankle[:, 1] - hip[:, 1]) / safe_torso,
        }
    for name, values in measures.items():
        result.update(_percentiles(name, np.where(eligible, values, np.nan)))
    return result


COCO_GROUP_INDICES: dict[str, tuple[int, ...]] = {
    "body17": tuple(range(0, 17)),
    "feet6": tuple(range(17, 23)),
    "face68": tuple(range(23, 91)),
    "left_hand": tuple(range(91, 112)),
    "right_hand": tuple(range(112, 133)),
}
# Round 4 default. feet6 and face68 were measured on 180 files and add exactly
# zero files beyond body17 + hands, while sitting a long way inside the raster
# (worst-per-file inset p5 of +140 px for the face, +24 px for the feet), so
# testing them buys nothing and risks flagging on landmark noise.
DEFAULT_COCO_GROUPS: tuple[str, ...] = ("body17", "left_hand", "right_hand")


def signed_inset_px(xy: np.ndarray, width: int, height: int) -> np.ndarray:
    """Distance from each point to the nearest raster edge; negative is outside."""

    points = np.asarray(xy, dtype=np.float64)
    x, y = points[..., 0], points[..., 1]
    return np.minimum.reduce([x, (width - 1) - x, y, (height - 1) - y])


def _drop_short_runs(mask: np.ndarray, minimum: int) -> np.ndarray:
    """Zero out True runs shorter than ``minimum`` frames."""

    array = np.asarray(mask, dtype=bool)
    if minimum <= 1 or not array.any():
        return array
    padded = np.concatenate(([False], array, [False]))
    edges = np.flatnonzero(padded[1:] != padded[:-1])
    kept = array.copy()
    for start, stop in zip(edges[0::2], edges[1::2]):
        if stop - start < minimum:
            kept[start:stop] = False
    return kept


def in_frame_violation_frames(
    *,
    frames: int,
    width: int,
    height: int,
    projected_xy: np.ndarray | None = None,
    vertex_worst_inset_px: np.ndarray | None = None,
    keypoints: np.ndarray | None = None,
    smplh_valid: np.ndarray | None = None,
    joint_indices: Sequence[int] = IN_FRAME_JOINTS,
    coco_groups: Sequence[str] = DEFAULT_COCO_GROUPS,
    margin_px: float = 0.0,
    joint_inset_px: float = 0.0,
    min_violation_run_frames: int = 1,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Per-frame FM2 verdict, and each term that contributed to it.

    Shared by the scanner and the gallery sampler so a clip can be anchored on
    exactly the frames the detector objected to, rather than on a second
    reimplementation that might disagree.
    """

    terms: dict[str, np.ndarray] = {}
    combined = np.zeros(frames, dtype=bool)

    if vertex_worst_inset_px is not None:
        inset = np.asarray(vertex_worst_inset_px, dtype=np.float64).reshape(-1)[:frames]
        bad = np.zeros(frames, dtype=bool)
        bad[: len(inset)] = ~(inset >= -margin_px)
        terms["vertices"] = bad
        combined |= bad

    if projected_xy is not None:
        points = np.asarray(projected_xy, dtype=np.float64)[:frames]
        index = np.asarray([i for i in joint_indices if i < points.shape[1]], dtype=np.int64)
        inset = signed_inset_px(points[:, index, :], width, height)
        finite = np.isfinite(points[:, index, :]).all(axis=-1)
        # A non-finite projection is not evidence of good framing.
        bad_point = ~(inset >= joint_inset_px - margin_px) | ~finite
        bad = np.zeros(frames, dtype=bool)
        bad[: len(points)] = bad_point.any(axis=1)
        terms["joints"] = bad
        terms["_joint_bad_point"] = bad_point
        terms["_joint_index"] = index
        combined |= bad

    if keypoints is not None:
        points = np.asarray(keypoints, dtype=np.float64)[:frames]
        usable = usable_keypoint_mask(points)
        for group in coco_groups:
            members = [i for i in COCO_GROUP_INDICES.get(group, ()) if i < points.shape[1]]
            if not members:
                continue
            selected = points[:, members, :2]
            inset = signed_inset_px(selected, width, height)
            # Only keypoints the detector actually placed can testify; a
            # zero-filled row is absence of evidence, not a body outside frame.
            bad_point = ~(inset >= -margin_px) & usable[:, members]
            bad = np.zeros(frames, dtype=bool)
            bad[: len(points)] = bad_point.any(axis=1)
            terms[f"coco_{group}"] = bad
            combined |= bad

    if smplh_valid is not None:
        valid = np.asarray(smplh_valid, dtype=bool).reshape(-1)
        invalid = np.zeros(frames, dtype=bool)
        invalid[: len(valid)] = ~valid[:frames]
        terms["smplh_invalid"] = invalid
        combined |= invalid

    return _drop_short_runs(combined, min_violation_run_frames), terms


def in_frame_violations(
    projected_xy: np.ndarray,
    *,
    width: int,
    height: int,
    smplh_valid: np.ndarray | None,
    vertex_worst_inset_px: np.ndarray | None = None,
    keypoints: np.ndarray | None = None,
    joint_indices: Sequence[int] = IN_FRAME_JOINTS,
    coco_groups: Sequence[str] = DEFAULT_COCO_GROUPS,
    margin_px: float = 0.0,
    joint_inset_px: float = 0.0,
    min_violation_run_frames: int = 1,
) -> dict[str, Any]:
    """FM2: does any part of the participant's body leave the raster, ever?

    Round 3 tested SMPL-H **joint centres** and missed a case the reviewer found
    by eye: in V00_S0180_I00000482_P0047 at t = 222.57 s the participant's left
    elbow crosses the right edge, and the file scored a perfect
    ``fm2_in_frame_and_valid_frac`` of 1.0. Two causes, both measured:

    1. **Joint centres are not the body surface.** The elbow *joint* sat 7.7 px
       inside while the mesh surface reached 24.5 px outside — a 32.2 px gap.
    2. **The projection is pulled inward on the arms.** Against the released 2D
       keypoints the SMPL-H projection reads further inside by a median of 11.0 px
       at the left elbow and 9.2 px at the left shoulder, and reads further
       *outside* by 17-20 px at the ankles.

    So the rule now tests the **mesh surface**, and cross-checks it against the
    released 2D keypoints, which are an independent observation of where the
    person actually is. A frame is a violation if any of these holds:

    * an SMPL-H mesh vertex is outside the raster;
    * ``smplh:is_valid`` is false;
    * a confidently-detected COCO body-17 or hand keypoint is outside the raster.

    **Do not over-sell the mesh term.** The joint-centre-to-surface gap has a
    median of 38.8 px per frame across the 4,000-file scan, but that is *not*
    what the vertex term is worth, because ``IN_FRAME_JOINTS`` already contains
    indices 63-72 — the ten fingertips, which ``smplx`` produces by selecting
    mesh vertices, not by kinematics. The joint set was therefore already part
    surface. Measured on 4,000 files, going from joints to the full mesh moves
    the flag rate from 66.70% to 68.70%: **80 files, 2.00 points**. It is real
    (and it is what catches the reviewer's file), but it is a 2-point effect, not
    a 38-pixel one. The joint term is kept because it costs nothing and its
    subsumption is worth continuing to measure: joints currently flag zero files
    the vertices do not.

    Full composition over the 4,000-file scan: joints 66.70% → +vertices 68.70%
    → +SMPL-H validity 78.65% → +COCO body-17 **+0 files** → +COCO hands 81.10%.
    The rule is a **strict superset** of Round 3's: zero files the old rule
    flagged are now passed, and 145 are newly flagged.

    Every term is optional so the contribution of each can be reported
    separately, and ``joint_inset_px`` survives as a knob even though the vertex
    rule supersedes it. The released detection box is deliberately *not* a term:
    its coordinates are clamped to the raster, so it can report that something
    was cut but never how much.
    """

    points = np.asarray(projected_xy, dtype=np.float64)
    if points.ndim != 3 or points.shape[2] != 2:
        return {"fm2_status": f"invalid_projection_shape:{points.shape}"}
    if len(points) == 0 or width <= 0 or height <= 0:
        return {"fm2_status": "empty_or_invalid_raster"}

    frames = len(points)
    bad, terms = in_frame_violation_frames(
        frames=frames, width=width, height=height, projected_xy=points,
        vertex_worst_inset_px=vertex_worst_inset_px, keypoints=keypoints,
        smplh_valid=smplh_valid, joint_indices=joint_indices, coco_groups=coco_groups,
        margin_px=margin_px, joint_inset_px=joint_inset_px,
        min_violation_run_frames=min_violation_run_frames,
    )
    valid = (
        np.asarray(smplh_valid, dtype=bool).reshape(-1)[:frames]
        if smplh_valid is not None else np.ones(frames, dtype=bool)
    )
    joint_index = terms.get("_joint_index")
    joint_bad_point = terms.get("_joint_bad_point")
    runs_count, runs_max = _run_stats(bad)
    result: dict[str, Any] = {
        "fm2_status": "ok",
        "fm2_frames": int(frames),
        "fm2_joints_tested": int(len(joint_index)) if joint_index is not None else 0,
        "fm2_in_frame_and_valid_frac": float((~bad).mean()),
        "fm2_smplh_valid_frac": float(valid.mean()) if len(valid) else float("nan"),
        "fm2_violation_runs": runs_count,
        "fm2_violation_run_max_frames": runs_max,
        # Where the worst violation is, so the gallery can anchor a clip on it
        # without re-running forward kinematics per clip — and, more to the
        # point, without a second implementation of the rule that could disagree
        # with this one about which frames are bad.
        "fm2_longest_violation_start_frame": longest_violation_start(bad),
    }
    # Per-term frame fractions, so the report can say which evidence carried
    # each decision and the reviewer can retire a term without a rescan.
    for name, mask in terms.items():
        if not name.startswith("_"):
            result[f"fm2_term_{name}_frame_frac"] = float(np.asarray(mask).mean())
    # "In frame" alone, with the validity term removed, for comparability with
    # the Round-3 column of the same name.
    geometric = np.zeros(frames, dtype=bool)
    for name, mask in terms.items():
        if not name.startswith("_") and name != "smplh_invalid":
            geometric |= np.asarray(mask)
    result["fm2_in_frame_frac"] = float((~geometric).mean())

    # Pass fractions for term subsets, so retiring or restoring a term is a
    # config edit rather than a 4,000-file rescan — and so the Round-3 rule can
    # be compared against this one on exactly the same files.
    subsets = {
        "round3_joints_only": ("joints", "smplh_invalid"),
        "vertices_only": ("vertices", "smplh_invalid"),
        "coco_only": ("coco_body17", "coco_left_hand", "coco_right_hand", "smplh_invalid"),
        "no_coco": ("vertices", "joints", "smplh_invalid"),
    }
    for label, members in subsets.items():
        union = np.zeros(frames, dtype=bool)
        for name in members:
            if name in terms:
                union |= np.asarray(terms[name])
        union = _drop_short_runs(union, min_violation_run_frames)
        result[f"fm2_pass_frac_{label}"] = float((~union).mean())
    if vertex_worst_inset_px is not None:
        inset = np.asarray(vertex_worst_inset_px, dtype=np.float64).reshape(-1)[:frames]
        result.update(_percentiles("fm2_vertex_worst_inset_px", inset))
        result["fm2_vertex_worst_inset_min_px"] = float(np.nanmin(inset)) if inset.size else float("nan")
        if joint_bad_point is not None and joint_index is not None:
            joint_inset = signed_inset_px(points[:, joint_index, :], width, height).min(axis=1)
            # The body-surface correction: how much further out the mesh reaches
            # than the outermost joint centre. Reported so the size of the Round-3
            # blind spot is a number in the record, not an argument.
            result.update(_percentiles("fm2_surface_correction_px", joint_inset - inset))
    if joint_bad_point is not None and joint_index is not None:
        result["fm2_out_of_frame_point_frac"] = float(joint_bad_point.mean())
        regions = {
            "hands": [i for i in joint_index if i >= 22 and i != 52],
            "feet": [i for i in joint_index if i in (7, 8, 10, 11)],
            "head": [i for i in joint_index if i in (12, 15)],
        }
        for name, members in regions.items():
            if not members:
                continue
            position = [int(np.where(joint_index == m)[0][0]) for m in members]
            result[f"fm2_{name}_out_of_frame_frame_frac"] = float(
                joint_bad_point[:, position].any(axis=1).mean()
            )
    return result


def longest_violation_start(flags: np.ndarray) -> int:
    """First frame of the longest True run, or -1 if there is none."""

    array = np.asarray(flags, dtype=bool)
    if array.size == 0 or not array.any():
        return -1
    padded = np.concatenate(([False], array, [False]))
    edges = np.flatnonzero(padded[1:] != padded[:-1])
    starts, stops = edges[0::2], edges[1::2]
    return int(starts[int(np.argmax(stops - starts))])


def smplh_invalid_runs(smplh_valid: np.ndarray | None, frames: int) -> dict[str, Any]:
    """Where and how long the SMPL-H-invalid stretches are.

    FM2 decides on ``smplh:is_valid`` alone, so a gallery clip that is supposed
    to show why a file was flagged has to be positioned on an *invalid* stretch —
    not on the longest run of the combined geometric mask, which is a different
    set of frames and may not overlap it at all.
    """

    if smplh_valid is None:
        return {"fm2_invalid_runs": 0, "fm2_longest_invalid_run_frames": 0,
                "fm2_longest_invalid_run_start_frame": -1}
    invalid = ~np.asarray(smplh_valid, dtype=bool).reshape(-1)[:frames]
    count, longest = _run_stats(invalid)
    return {
        "fm2_invalid_runs": count,
        "fm2_longest_invalid_run_frames": longest,
        "fm2_longest_invalid_run_start_frame": longest_violation_start(invalid),
    }


def _run_stats(flags: np.ndarray) -> tuple[int, int]:
    array = np.asarray(flags, dtype=bool)
    if array.size == 0 or not array.any():
        return 0, 0
    padded = np.concatenate(([False], array, [False]))
    edges = np.flatnonzero(padded[1:] != padded[:-1])
    lengths = edges[1::2] - edges[0::2]
    return int(len(lengths)), int(lengths.max())


def windowed_in_frame_pass_frac(
    projected_xy: np.ndarray,
    *,
    width: int,
    height: int,
    smplh_valid: np.ndarray | None,
    window_frames: int,
    joint_indices: Sequence[int] = IN_FRAME_JOINTS,
    margin_px: float = 0.0,
) -> float:
    """Fraction of non-overlapping windows that would pass FM2 on their own.

    File-level FM2 rejects an entire four-minute recording for one bad frame.
    This says how much of the same file a windowed rule would keep, so the cost
    of the file-level choice is a measured number rather than an intuition.
    """

    points = np.asarray(projected_xy, dtype=np.float64)
    if points.ndim != 3 or len(points) < window_frames or window_frames < 1:
        return float("nan")
    index = np.asarray([i for i in joint_indices if i < points.shape[1]], dtype=np.int64)
    selected = points[:, index, :]
    finite = np.isfinite(selected).all(axis=-1)
    inside = (
        (selected[..., 0] >= -margin_px) & (selected[..., 0] < width + margin_px)
        & (selected[..., 1] >= -margin_px) & (selected[..., 1] < height + margin_px)
    )
    frame_ok = (inside & finite).all(axis=1)
    if smplh_valid is not None and len(smplh_valid) == len(points):
        frame_ok &= np.asarray(smplh_valid, dtype=bool).reshape(-1)
    count = len(points) // window_frames
    if count == 0:
        return float("nan")
    trimmed = frame_ok[: count * window_frames].reshape(count, window_frames)
    return float(trimmed.all(axis=1).mean())


COCO_KNEES = (13, 14)
COCO_ANKLES = (15, 16)


def leg_visibility(
    keypoints: np.ndarray, *, width: int, height: int
) -> dict[str, Any]:
    """How much of the legs the camera actually caught.

    Round-6 review turned up eleven V02 and V03 clips where the participant is
    standing and FM1 fires anyway. All eleven fire on ``shin_inverted`` rather
    than on knee position: with the ankles outside the frame the fit puts them
    somewhere arbitrary, and a shin that points upward reads as seated.

    This measures the same thing from the released 2D keypoints instead of from
    the fit, so it is independent of whatever the fit did with the missing legs,
    and it separates "FM1 saw the legs and judged them" from "FM1 never saw the
    legs". It changes no verdict; it says which of the two happened.
    """
    points = np.asarray(keypoints, dtype=np.float64)
    if points.ndim != 3 or len(points) == 0 or points.shape[1] <= max(COCO_ANKLES):
        return {"fm1_ankles_visible_frac": None, "fm1_knees_visible_frac": None,
                "fm1_leg_visibility_status": "unmeasurable"}
    usable = usable_keypoint_mask(points)

    def visible(indices: tuple[int, int]) -> float:
        inside = (
            (points[..., 0] >= 0) & (points[..., 0] <= width - 1)
            & (points[..., 1] >= 0) & (points[..., 1] <= height - 1)
        )
        both = np.ones(len(points), dtype=bool)
        for index in indices:
            both &= usable[:, index] & inside[:, index]
        return float(both.mean())

    return {
        "fm1_ankles_visible_frac": visible(COCO_ANKLES),
        "fm1_knees_visible_frac": visible(COCO_KNEES),
        "fm1_leg_visibility_status": "ok",
    }


def static_hand_measure(
    keypoints: np.ndarray,
    *,
    radius_shoulder_widths: float,
    extra_radii: Sequence[float] = (),
) -> dict[str, Any]:
    """FM3: how long do the wrists hold one position?

    Operational definition, stated explicitly because "the same position" is not
    self-defining:

    1. Wrist positions are expressed relative to the shoulder midpoint and
       divided by the median shoulder width, so the measure is free of camera
       distance, body size, and any drift of the person around the platform.
    2. Each wrist's **reference position** is its own per-file median in that
       frame of reference — the position it spends most of its time near.
    3. A frame counts as *held* when **both** wrists are within
       ``radius_shoulder_widths`` of their references.
    4. ``fm3_static_frac`` is the fraction of usable frames that are held.

    Using the per-file median rather than the first frame matters: a participant
    who gestures for the first two seconds and is then motionless is static, and
    anchoring on frame zero would miss it.

    ``extra_radii`` evaluates the same fraction at other radii in one pass, so
    the radius can be re-tuned from the report without re-reading any media.
    """

    points = np.asarray(keypoints, dtype=np.float64)
    if points.ndim != 3 or points.shape[1] < 133:
        return {"fm3_status": f"invalid_keypoint_shape:{points.shape}"}
    if len(points) < 2:
        return {"fm3_status": "insufficient_frames"}
    usable = usable_keypoint_mask(points)
    shoulder_ok = usable[:, COCO_SHOULDERS[0]] & usable[:, COCO_SHOULDERS[1]]
    left_sh = points[:, COCO_SHOULDERS[0], :2]
    right_sh = points[:, COCO_SHOULDERS[1], :2]
    widths = np.linalg.norm(left_sh - right_sh, axis=1)
    good = shoulder_ok & (widths > 1e-6)
    if not good.any():
        return {"fm3_status": "no_shoulder_reference"}
    scale = float(np.median(widths[good]))
    midpoint = 0.5 * (left_sh + right_sh)

    result: dict[str, Any] = {
        "fm3_status": "ok",
        "fm3_shoulder_width_px": scale,
        "fm3_radius_shoulder_widths": float(radius_shoulder_widths),
        "fm3_frames": int(len(points)),
    }
    distances: dict[str, np.ndarray] = {}
    for side, index in COCO_WRISTS.items():
        ok = usable[:, index] & good
        relative = (points[:, index, :2] - midpoint) / scale
        relative = np.where(ok[:, None], relative, np.nan)
        if not np.isfinite(relative).all(axis=1).any():
            return {"fm3_status": f"no_usable_{side}_wrist_frames"}
        reference = np.nanmedian(relative, axis=0)
        distance = np.linalg.norm(relative - reference, axis=1)
        distances[side] = distance
        result.update(_percentiles(f"fm3_{side}_offset_shoulder_widths", distance))

    both = np.isfinite(distances["left"]) & np.isfinite(distances["right"])
    result["fm3_usable_frame_frac"] = float(both.mean())
    if not both.any():
        return {**result, "fm3_status": "no_frames_with_both_wrists"}
    worse = np.maximum(distances["left"], distances["right"])[both]
    result["fm3_static_frac"] = float((worse <= radius_shoulder_widths).mean())
    for radius in extra_radii:
        result[f"fm3_static_frac_at_r{radius:g}".replace(".", "p")] = float(
            (worse <= radius).mean()
        )
    result.update(_percentiles("fm3_worse_wrist_offset_shoulder_widths", worse))
    return result


# COCO body-17 indices 5..16 — shoulders, elbows, wrists, hips, knees, ankles —
# and the SMPL-H joints M-4 matched them to. The first five COCO body points
# (nose, eyes, ears) map to auxiliary face joints 52-56 and the six foot points
# to 57-62; M-4 never validated either group, so neither is compared here.
REPROJECTION_COCO = tuple(range(5, 17))
REPROJECTION_SMPLH = (16, 17, 18, 19, 20, 21, 1, 2, 4, 5, 7, 8)


def smplh_reprojection_error(
    projected_xy: np.ndarray,
    keypoints: np.ndarray,
    *,
    smplh_valid: np.ndarray | None = None,
) -> dict[str, Any]:
    """How far the fitted SMPL-H body lands from where the person actually is.

    This is the direct form of the question the PI asked: not "did a limb cross
    the raster edge" but "did the SMPL parameters stay roughly where they should
    be". It compares the projected SMPL-H joints against the released 2D
    keypoints — an independent observation of the same person — over the twelve
    body landmarks M-4 validated.

    The error is reported both in pixels and in **shoulder widths**. The
    normalised form is the one to threshold on: a 25 px error means something
    different on a participant filling the frame than on one standing further
    back, and the reviewer has already established that the camera moves closer
    for seated participants.

    It is also split by ``smplh:is_valid``, because the released flag is
    undocumented and the only way to find out whether it tracks fit quality is
    to measure fit quality on both sides of it.
    """

    points = np.asarray(projected_xy, dtype=np.float64)
    detected = np.asarray(keypoints, dtype=np.float64)
    if points.ndim != 3 or detected.ndim != 3 or detected.shape[1] < 133:
        return {"reproj_status": f"invalid_shapes:{points.shape},{detected.shape}"}
    frames = min(len(points), len(detected))
    if frames == 0:
        return {"reproj_status": "empty"}
    points, detected = points[:frames], detected[:frames]
    if points.shape[1] <= max(REPROJECTION_SMPLH):
        return {"reproj_status": f"too_few_smplh_joints:{points.shape[1]}"}

    coco = np.asarray(REPROJECTION_COCO, dtype=np.int64)
    smplh = np.asarray(REPROJECTION_SMPLH, dtype=np.int64)
    usable = usable_keypoint_mask(detected)[:, coco]
    fitted = points[:, smplh, :]
    observed = detected[:, coco, :2]
    distance = np.linalg.norm(fitted - observed, axis=-1)
    distance = np.where(usable & np.isfinite(distance), distance, np.nan)

    left, right = detected[:, COCO_SHOULDERS[0], :2], detected[:, COCO_SHOULDERS[1], :2]
    widths = np.linalg.norm(left - right, axis=1)
    shoulder_ok = (
        usable_keypoint_mask(detected)[:, COCO_SHOULDERS[0]]
        & usable_keypoint_mask(detected)[:, COCO_SHOULDERS[1]]
        & (widths > 1e-6)
    )
    if not shoulder_ok.any():
        return {"reproj_status": "no_shoulder_reference"}
    scale = float(np.median(widths[shoulder_ok]))

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", "All-NaN slice encountered", RuntimeWarning)
        warnings.filterwarnings("ignore", "Mean of empty slice", RuntimeWarning)
        per_frame_px = np.nanmedian(distance, axis=1)
    result: dict[str, Any] = {
        "reproj_status": "ok",
        "reproj_shoulder_width_px": scale,
        "reproj_frames": int(frames),
        "reproj_usable_frame_frac": float(np.isfinite(per_frame_px).mean()),
    }
    result.update(_percentiles("reproj_px", per_frame_px))
    result.update(_percentiles("reproj_shoulder_widths", per_frame_px / scale))

    # Does the released validity flag actually track fit quality? Reported on
    # both sides so the question has an answer rather than an assumption.
    if smplh_valid is not None:
        valid = np.asarray(smplh_valid, dtype=bool).reshape(-1)[:frames]
        if len(valid) == frames:
            for name, mask in (("on_valid", valid), ("on_invalid", ~valid)):
                subset = per_frame_px[mask]
                subset = subset[np.isfinite(subset)]
                result[f"reproj_shoulder_widths_{name}_p50"] = (
                    float(np.median(subset) / scale) if subset.size else float("nan")
                )
                result[f"reproj_{name}_frames"] = int(subset.size)
    return result


def session_sitting_verdicts(
    rows: Sequence[Mapping[str, Any]], thresholds: Thresholds
) -> dict[str, dict[str, Any]]:
    """One seated/standing verdict per **participant within a recording session**.

    The reviewer observed that a participant is either standing in every one of
    their recordings or seated in every one, and offered that as a way to make
    FM1 robust. The observation is right; measurement puts the scope of it at the
    *session*, not the person. A chair, a camera height and a standing platform
    are set up once per session, so posture is a property of the setup.

    P0003A settles it. They appear in **eight** sessions. In S0200 the reviewer
    confirmed by eye that they are seated in a low chair with crossed legs, and
    hip flexion there reads 88.2 degrees. In the other seven sessions hip flexion
    runs 130.6 to 156.1, and a rendered clip from S0126 shows them plainly
    standing on a platform. Aggregating over the person would call all ten files
    seated and wrongly reject nine of them. Across the whole 4,000-file scan the
    same pattern holds in aggregate: the median within-session spread of hip
    flexion is 2.76 degrees against 7.11 degrees between sessions, and units that
    disagree internally fall from 6.81% of multi-file participants to **2.52%** of
    multi-file participant-sessions.

    Session scoping also removes a rule that could not be justified. An earlier
    draft carried a "deep" clause — one file far inside the seated range speaks
    for the whole participant — purely to rescue P0003A. Leave-one-participant-out
    broke on exactly that participant, which is the signature of a parameter
    fitted to one example. Under session scoping P0003A's seated session contains
    one file, that file flags, and the majority clause carries it with nothing
    fitted. The deep clause changes **no** verdict on the 4,000-file scan and is
    gone.

    So there is one clause: a majority of the unit's files read as seated.

    A unit that is not seated but has either an internal disagreement or a file
    within ``sitting_near_cut_*`` of a threshold is returned as **near the
    boundary**. That flag exists because 56.3% of units hold exactly one file,
    and a single-file unit can never disagree with itself — without a near-cut
    channel the majority rule would report perfect confidence on more than half
    the corpus.

    Caveat that matters for anything built on this: fractions are taken over the
    unit's *scanned* files, not over everything recorded in that session.
    """

    grouped: dict[str, list[Mapping[str, Any]]] = {}
    for row in rows:
        participant = str(row.get("participant_id") or "").strip()
        session = str(row.get("session_id") or "").strip()
        if participant:
            grouped.setdefault(f"{participant}@{session}", []).append(row)

    verdicts: dict[str, dict[str, Any]] = {}
    for unit, member_rows in grouped.items():
        flagged = 0
        determined = 0
        hips: list[float] = []
        betweens: list[float] = []
        for row in member_rows:
            file_flag = apply_thresholds(row, thresholds).get("fm1_sitting")
            if file_flag is None:
                continue
            determined += 1
            flagged += int(bool(file_flag))
            for key, sink in (
                ("fm1_hip_flexion_deg_p50", hips),
                ("fm1_knee_between_torso_p50", betweens),
            ):
                try:
                    value = float(row.get(key))  # type: ignore[arg-type]
                except (TypeError, ValueError):
                    continue
                if np.isfinite(value):
                    sink.append(value)

        base = {
            "sitting_unit": unit,
            "unit_files": determined or len(member_rows),
            "unit_files_flagged": flagged,
        }
        if determined == 0:
            verdicts[unit] = {
                **base, "unit_sitting": None, "unit_sitting_basis": "undetermined",
                "unit_flagged_frac": float("nan"),
                "unit_hip_flexion_min": float("nan"),
                "unit_knee_between_min": float("nan"),
                "unit_sitting_near_cut": False,
            }
            continue

        fraction = flagged / determined
        hip_min = min(hips) if hips else float("nan")
        between_min = min(betweens) if betweens else float("nan")
        seated = fraction >= thresholds.sitting_seated_file_frac
        near_cut = bool(
            not seated
            and (
                flagged > 0
                or (
                    np.isfinite(between_min)
                    and between_min
                    < thresholds.sitting_knee_between_torso
                    + thresholds.sitting_near_cut_knee_between_torso
                )
            )
        )
        verdicts[unit] = {
            **base,
            "unit_sitting": bool(seated),
            "unit_sitting_basis": "majority" if seated else "none",
            "unit_flagged_frac": float(fraction),
            "unit_hip_flexion_min": float(hip_min),
            "unit_knee_between_min": float(between_min),
            "unit_sitting_near_cut": near_cut,
        }
    return verdicts


def tracker_discontinuity(boxes: np.ndarray, box_valid: np.ndarray | None) -> dict[str, Any]:
    """A separate check for the case FM2's in-frame rule cannot see.

    If a second person walks into shot, the tracked participant may stay wholly
    inside the frame, so the in-frame rule stays silent. What does change is the
    box: a tracker that switches subject jumps. This measures the largest
    frame-to-frame centre jump and size change, both normalized by box height.

    It is a proxy for a subject switch, not a person counter. A genuine
    multi-person detector does not exist here, and this cannot see a bystander
    the tracker correctly ignores.
    """

    array = np.asarray(boxes, dtype=np.float64)
    if array.ndim != 2 or array.shape[1] < 4 or len(array) < 2:
        return {"tracker_status": f"invalid_box_shape:{array.shape}"}
    valid = (
        np.asarray(box_valid, dtype=bool).reshape(-1)
        if box_valid is not None and len(box_valid) == len(array)
        else np.ones(len(array), dtype=bool)
    )
    heights = np.abs(array[:, 3] - array[:, 1])
    centre = np.stack([0.5 * (array[:, 0] + array[:, 2]), 0.5 * (array[:, 1] + array[:, 3])], axis=1)
    pair = valid[:-1] & valid[1:] & (heights[:-1] > 1e-6) & (heights[1:] > 1e-6)
    if not pair.any():
        return {"tracker_status": "no_valid_adjacent_boxes"}
    jump = np.linalg.norm(centre[1:] - centre[:-1], axis=1) / heights[:-1]
    size_change = np.abs(heights[1:] - heights[:-1]) / heights[:-1]
    return {
        "tracker_status": "ok",
        "tracker_centre_jump_max": float(np.nanmax(jump[pair])),
        "tracker_centre_jump_p99": float(np.nanpercentile(jump[pair], 99)),
        "tracker_size_change_max": float(np.nanmax(size_change[pair])),
        "tracker_box_height_cv": float(
            np.nanstd(heights[valid]) / np.nanmean(heights[valid])
            if valid.any() and np.nanmean(heights[valid]) > 0 else np.nan
        ),
    }


def apply_thresholds(row: Mapping[str, Any], thresholds: Thresholds) -> dict[str, Any]:
    """Turn measurements into flags. The only place a cut point is applied.

    A measurement that could not be taken produces ``None`` rather than a pass:
    an undetectable failure mode is not an absent one, and silently passing those
    files would quietly bias the surviving set.
    """

    def number(key: str) -> float:
        value = row.get(key)
        try:
            return float(value)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            return float("nan")

    flags: dict[str, Any] = {}

    raster = f"{row.get('width')}x{row.get('height')}"
    drift = number("timebase_index_drift_s")
    reasons_zero = []
    if raster in thresholds.excluded_rasters:
        reasons_zero.append(f"raster_{raster}")
    if np.isfinite(drift) and abs(drift) > thresholds.max_timebase_drift_s:
        reasons_zero.append("timebase_mismatch")
    flags["fm0_unusable_source"] = bool(reasons_zero)
    flags["fm0_reasons"] = "+".join(reasons_zero)

    shin = number("fm1_shin_verticality_p50")
    vendor = str(row.get("vendor_id", "00"))
    policy = thresholds.sitting_by_vendor.get(
        vendor, ("knee_between", thresholds.sitting_knee_between_torso)
    )
    measure_key = {
        "knee_between": "fm1_knee_between_torso_p50",
        "hip_flexion": "fm1_hip_flexion_deg_p50",
    }
    if policy is None:
        # Deliberately off for this vendor, which is not the same as "measured
        # and found clean": the measurements are still recorded, and `retired`
        # says so rather than letting the file read as a validated pass.
        flags["fm1_sitting"] = False
        flags["fm1_sitting_reasons"] = "retired_for_vendor"
    else:
        measure, cut = policy
        value = number(measure_key[measure])
        if row.get("posture_status") != "ok" or not np.isfinite(value):
            flags["fm1_sitting"] = None
        else:
            # One measure plus one rare-posture add-on, both below their cut.
            reasons = []
            if value < cut:
                reasons.append(measure)
            if (thresholds.sitting_shin_addon_by_vendor.get(vendor, True)
                    and np.isfinite(shin) and shin < thresholds.sitting_shin_verticality):
                reasons.append("shin_inverted")
            flags["fm1_sitting"] = bool(reasons)
            flags["fm1_sitting_reasons"] = "+".join(reasons)

    valid_frac = number("fm2_smplh_valid_frac")
    if row.get("fm2_status") != "ok" or not np.isfinite(valid_frac):
        flags["fm2_smplh_invalid"] = None
    else:
        flags["fm2_smplh_invalid"] = bool(valid_frac < thresholds.smplh_valid_required_frac)

    static = number("fm3_static_frac")
    if row.get("fm3_status") != "ok" or not np.isfinite(static):
        flags["fm3_static_hands"] = None
    else:
        flags["fm3_static_hands"] = bool(static >= thresholds.static_frac_limit)

    level = number("audio_speech_level_db")
    flatness = number("audio_spectral_flatness")
    status = str(row.get("audio_status", ""))
    if not thresholds.audio_dead_by_vendor.get(vendor, True):
        if "audio_status" in row:
            flags["fm4_audio_dead"] = False
            flags["fm4_audio_reasons"] = "retired_for_vendor"
    elif "audio_status" not in row:
        # A scan taken before FM4 existed. Not evaluated, rather than undetermined:
        # every earlier round's flag columns have to stay comparable, and an
        # absent *column* is a different thing from an unreadable WAV.
        pass
    elif status in {"empty", "too_short"} or status.startswith("read_error"):
        flags["fm4_audio_dead"] = True
        flags["fm4_audio_reasons"] = status
    else:
        dynamics = number("audio_envelope_dynamics_db")
        if status != "ok" or not np.isfinite(dynamics):
            flags["fm4_audio_dead"] = None
        else:
            dead = dynamics < thresholds.audio_min_envelope_dynamics_db
            flags["fm4_audio_dead"] = bool(dead)
            flags["fm4_audio_reasons"] = "no_dynamics" if dead else ""

    names = [name for name, value in flags.items() if value is True]
    unknown = [name for name, value in flags.items() if value is None]
    flags["flagged"] = bool(names)
    flags["flag_count"] = len(names)
    flags["flags_fired"] = "+".join(sorted(names))
    flags["flags_undetermined"] = "+".join(sorted(unknown))
    # "Passes" requires every detector to have actually run and said no.
    flags["passes_all"] = bool(not names and not unknown)
    return flags

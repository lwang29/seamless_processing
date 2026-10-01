"""Posture: leg geometry from the fitted pose, gated on what the camera saw.

The synthetic poses are real SMPL-H forward kinematics (the neutral model, all
betas zero): the zero pose is a standing person (hip 172 deg, knee_between
0.54) and hips flexed -90 deg about x with knees +90 deg is a seated one (hip
96 deg, knee_between 0.25). Every assertion about a label goes through the
same functions the scan and the annotate step call, so a threshold or a
visibility gate that drifts shows up here.
"""

from __future__ import annotations

import math
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from seamless_curation import posture
from seamless_curation.clips import clip_grid
from seamless_curation.posture import (
    AGGREGATE_KEYS,
    COCO_ANKLES,
    COCO_HIPS,
    COCO_KNEES,
    SITTING,
    STANDING,
    UNOBSERVED,
    PostureRules,
    aggregate_groups,
    aggregate_states,
    bin_clip_index,
    classify_bins,
    clip_posture_measures,
    posture_bins,
    posture_frames,
    posture_rule,
    recording_posture_measures,
)
from seamless_curation.smplh_kinematics import forward_kinematics, stack_pose

SEATED = (-math.pi / 2, math.pi / 2)
STANDING_POSE = (0.0, 0.0)
FEET_UP = (-math.pi / 2, -math.pi / 2)  # thigh forward, shin folded upward


# ----------------------------------------------------------------------------- builders
def leg_joints(model_root: Path, hip: np.ndarray, knee: np.ndarray) -> np.ndarray:
    """Pelvis-frame FK joints with both hips/knees rotated about x per frame."""

    hip = np.asarray(hip, dtype=np.float64).reshape(-1)
    knee = np.asarray(knee, dtype=np.float64).reshape(-1)
    frames = hip.size
    body = np.zeros((frames, 21, 3))
    body[:, 0, 0] = body[:, 1, 0] = hip  # joints 1, 2 (hips) -> body_pose 0, 1
    body[:, 3, 0] = body[:, 4, 0] = knee  # joints 4, 5 (knees) -> body_pose 3, 4
    pose = stack_pose(body, np.zeros((frames, 15, 3)), np.zeros((frames, 15, 3)))
    joints, _, _ = forward_kinematics(pose, model_root)
    return joints


def constant_joints(model_root: Path, frames: int, angles: tuple[float, float]) -> np.ndarray:
    return leg_joints(model_root, np.full(frames, angles[0]), np.full(frames, angles[1]))


def all_visible(frames: int) -> np.ndarray:
    return np.ones((frames, 133), dtype=bool)


def recording_label(frames: posture.PostureFrames, fps: float, vendor: str, anamorphic: str = "none") -> dict:
    bins = posture_bins(frames, fps)
    rule, _ = posture_rule(vendor, anamorphic)
    states = classify_bins(rule, bins["hip_flexion_deg"].to_numpy(), bins["knee_between"].to_numpy())
    return aggregate_states(states)


# ----------------------------------------------------------------------------- per-frame geometry
def test_standing_and_seated_geometry(model_root: Path) -> None:
    stand = posture_frames(constant_joints(model_root, 4, STANDING_POSE), all_visible(4), smplh_anamorphic="none")
    sit = posture_frames(constant_joints(model_root, 4, SEATED), all_visible(4), smplh_anamorphic="none")
    assert np.allclose(stand.hip_flexion, 172.0, atol=1.0)
    assert np.allclose(stand.knee_between, 0.543, atol=0.01)
    assert np.all(stand.shin_inverted == 0.0)
    assert np.allclose(sit.hip_flexion, 95.8, atol=1.0)
    assert np.allclose(sit.knee_between, 0.246, atol=0.01)
    assert np.all(sit.hip_observable) and np.all(sit.ankle_observable)


def test_feet_up_inverts_the_shin_and_leaves_knee_between_undefined(model_root: Path) -> None:
    # Ankles above the pelvis along the torso axis: the ratio has no meaning, so
    # it is NaN rather than a division by ~0; the shin flag fires.
    frames = posture_frames(constant_joints(model_root, 3, FEET_UP), all_visible(3), smplh_anamorphic="none")
    assert np.all(frames.shin_inverted == 1.0)
    assert np.all(np.isnan(frames.knee_between))
    assert np.all(np.isfinite(frames.hip_flexion))


def test_hip_angle_is_rotation_invariant(model_root: Path) -> None:
    joints = constant_joints(model_root, 2, SEATED)
    angle = np.radians(37.0)
    rotation = np.array([[math.cos(angle), -math.sin(angle), 0.0],
                         [math.sin(angle), math.cos(angle), 0.0], [0.0, 0.0, 1.0]])
    turned = joints @ rotation.T
    a = posture_frames(joints, all_visible(2), smplh_anamorphic="none")
    b = posture_frames(turned, all_visible(2), smplh_anamorphic="none")
    assert np.allclose(a.hip_flexion, b.hip_flexion)
    assert np.allclose(a.knee_between, b.knee_between)


def test_observability_gates_follow_the_landmarks(model_root: Path) -> None:
    joints = constant_joints(model_root, 4, STANDING_POSE)
    visible = all_visible(4)
    visible[0, COCO_ANKLES[0]] = False  # ankle out of shot: hips still measurable
    visible[1, COCO_KNEES[1]] = False  # a knee out of shot: nothing measurable
    visible[2, COCO_HIPS[0]] = False  # a hip out of shot: ankle quantities still gated on knees+ankles
    frames = posture_frames(joints, visible, smplh_anamorphic="none")
    assert frames.hip_observable.tolist() == [True, False, False, True]
    assert frames.ankle_observable.tolist() == [False, False, True, True]
    assert np.isfinite(frames.hip_flexion).tolist() == [True, False, False, True]
    assert np.isfinite(frames.knee_between).tolist() == [False, False, True, True]
    assert np.isnan(frames.shin_inverted[:2]).all() and np.isfinite(frames.shin_inverted[2:]).all()


def test_input_validation(model_root: Path) -> None:
    joints = constant_joints(model_root, 3, STANDING_POSE)
    with pytest.raises(ValueError):
        posture_frames(joints, all_visible(4), smplh_anamorphic="none")
    with pytest.raises(ValueError):
        posture_frames(joints, all_visible(3), smplh_anamorphic="squeezed")
    with pytest.raises(ValueError):
        posture_frames(joints[:, :5], all_visible(3), smplh_anamorphic="none")


# ----------------------------------------------------------------------------- labels end to end
@pytest.mark.parametrize("vendor", ["V00", "V01", "V02", "V03"])
def test_standing_and_seated_recordings_under_every_rule(model_root: Path, vendor: str) -> None:
    fps, frames = 30.0, 300
    stand = posture_frames(constant_joints(model_root, frames, STANDING_POSE), all_visible(frames),
                           smplh_anamorphic="none")
    sit = posture_frames(constant_joints(model_root, frames, SEATED), all_visible(frames),
                         smplh_anamorphic="none")
    standing = recording_label(stand, fps, vendor)
    sitting = recording_label(sit, fps, vendor)
    assert standing["posture"] == "standing" and standing["posture_confidence"] == 1.0
    assert sitting["posture"] == "sitting" and sitting["posture_confidence"] == 1.0
    assert standing["posture_transitions"] == 0 and sitting["posture_transitions"] == 0


def test_sit_to_stand_within_a_clip_is_mixed(model_root: Path) -> None:
    fps, frames = 30.0, 900  # one 30-s clip: 12 s seated, 6 s rising, 12 s standing
    progress = np.clip((np.arange(frames) / fps - 12.0) / 6.0, 0.0, 1.0)
    joints = leg_joints(model_root, SEATED[0] * (1 - progress), SEATED[1] * (1 - progress))
    frames_ = posture_frames(joints, all_visible(frames), smplh_anamorphic="none", fps=fps)
    bins = posture_bins(frames_, fps)
    assert len(bins) == 30
    states = classify_bins("v03_hip", bins["hip_flexion_deg"].to_numpy(), bins["knee_between"].to_numpy())
    out = aggregate_states(states)
    assert out["posture"] == "mixed"
    assert out["posture_transitions"] == 1
    decided = out["posture_standing_frac"] + out["posture_sitting_frac"]
    assert out["posture_confidence"] == pytest.approx(decided)
    assert out["posture_unclear_frac"] > 0  # the rise passes through the unclear band
    measures = clip_posture_measures(frames_, 0, frames)
    assert measures["lower_body_observed_frac"] == 1.0
    assert 95.0 < measures["hip_flexion_deg_p50"] < 172.5


def test_legs_out_of_frame_is_unknown(model_root: Path) -> None:
    fps, frames = 30.0, 300
    visible = all_visible(frames)
    visible[:, list(COCO_KNEES + COCO_ANKLES)] = False
    pf = posture_frames(constant_joints(model_root, frames, STANDING_POSE), visible, smplh_anamorphic="none", fps=fps)
    out = recording_label(pf, fps, "V03")
    assert out["posture"] == "unknown"
    assert np.isnan(out["posture_confidence"])
    assert out["posture_unobserved_frac"] == 1.0
    clip = clip_posture_measures(pf, 0, frames)
    assert np.isnan(clip["hip_flexion_deg_p50"]) and np.isnan(clip["knee_between_p50"])
    assert np.isnan(clip["shin_inverted_frac"])
    assert clip["lower_body_observed_frac"] == 0.0  # a measured zero, not NA
    rec = recording_posture_measures(pf)
    assert np.isnan(rec["recording_hip_flexion_deg_p50"])
    assert rec["recording_lower_body_observed_frac"] == 0.0


def test_severe_anamorphic_is_never_measured(model_root: Path) -> None:
    fps, frames = 30.0, 300
    pf = posture_frames(constant_joints(model_root, frames, SEATED), all_visible(frames),
                        smplh_anamorphic="severe", fps=fps)
    assert np.isnan(pf.hip_flexion).all() and np.isnan(pf.knee_between).all()
    assert np.isnan(pf.shin_inverted).all()
    # Visibility keeps its meaning: the legs are in shot, the pose is what is unusable.
    assert pf.hip_observable.all()
    assert posture_rule("V01", "severe") == ("not_measurable_anamorphic", "not_applicable")
    out = recording_label(pf, fps, "V01", "severe")
    assert out["posture"] == "unknown" and out["posture_unobserved_frac"] == 1.0
    # ... and it is unknown even if a caller applied a measuring rule by mistake.
    assert recording_label(pf, fps, "V03")["posture"] == "unknown"
    clip = clip_posture_measures(pf, 0, frames)
    assert np.isnan(clip["hip_flexion_deg_p50"]) and np.isnan(clip["knee_between_p50"])
    rec = recording_posture_measures(pf)
    assert np.isnan(rec["recording_hip_flexion_deg_p50"]) and np.isnan(rec["recording_knee_between_p50"])


def test_v00_ignores_the_hip_angle(model_root: Path) -> None:
    # Hips flexed with the knee straight under them is not seated by knee
    # position: V00's rule reads the knee, V03's reads the hip.
    fps, frames = 30.0, 300
    joints = constant_joints(model_root, frames, (-math.pi / 2, 0.0))  # hip 96, kb 0.63
    pf = posture_frames(joints, all_visible(frames), smplh_anamorphic="none", fps=fps)
    assert recording_label(pf, fps, "V00")["posture"] == "standing"
    assert recording_label(pf, fps, "V03")["posture"] == "sitting"


# ----------------------------------------------------------------------------- rules
def test_vendor_dispatch() -> None:
    assert posture_rule("V03", "none") == ("v03_hip", "v03_file_labels_164")
    assert posture_rule("V00", "none") == ("v00_knee", "v00_file_labels_66_selection_biased")
    assert posture_rule("V01", "mild") == ("unvalidated_hip_knee", "unlabelled")
    assert posture_rule("V01", "none") == ("unvalidated_hip_knee", "unlabelled")
    assert posture_rule("V02", "none") == ("unvalidated_hip_knee", "unlabelled")
    assert posture_rule("03", "none") == ("v03_hip", "v03_file_labels_164")
    assert posture_rule("V01", "severe") == ("not_measurable_anamorphic", "not_applicable")
    assert posture_rule("V03", "none", measured=False) == ("not_measured", "not_applicable")
    assert posture_rule("V09", "none") == ("not_measured", "not_applicable")


def test_classify_bins_bands() -> None:
    nan = float("nan")
    hip = np.array([135.9, 136.0, 152.0, 152.1, nan])
    assert classify_bins("v03_hip", hip, np.full(5, 0.5)).tolist() == [
        "sitting", "unclear", "unclear", "standing", "unobserved"]
    knee = np.array([0.299, 0.30, 0.479, 0.48, nan])
    # V00 ignores the hip entirely, even when it is missing.
    assert classify_bins("v00_knee", np.full(5, nan), knee).tolist() == [
        "sitting", "unclear", "unclear", "standing", "unobserved"]
    hip = np.array([135.0, 135.0, 134.9, 119.9, 119.9, 119.9, 130.0, nan])
    knee = np.array([0.48, 0.47, 0.60, 0.39, 0.40, nan, nan, 0.2])
    assert classify_bins("unvalidated_hip_knee", hip, knee).tolist() == [
        "standing", "unclear", "unclear", "sitting", "unclear", "sitting", "unclear", "unobserved"]
    for rule in ("not_measurable_anamorphic", "not_measured"):
        assert classify_bins(rule, np.array([170.0, 90.0]), np.array([0.5, 0.2])).tolist() == [
            "unobserved", "unobserved"]
    with pytest.raises(ValueError):
        classify_bins("v04_hip", np.array([170.0]), np.array([0.5]))


def test_classify_bins_per_bin_rules_and_codes() -> None:
    rules = np.array(["v03_hip", "v00_knee", "unvalidated_hip_knee", "not_measured"], dtype=object)
    hip = np.array([170.0, 100.0, 100.0, 170.0])
    knee = np.array([0.20, 0.50, 0.30, 0.5])
    assert classify_bins(rules, hip, knee).tolist() == ["standing", "standing", "sitting", "unobserved"]
    codes = classify_bins(rules, hip, knee, as_codes=True)
    assert codes.dtype == np.int8 and codes.tolist() == [STANDING, STANDING, SITTING, UNOBSERVED]
    with pytest.raises(ValueError):
        classify_bins(np.array(["v03_hip", None], dtype=object), hip[:2], knee[:2])


def test_rules_are_overridable() -> None:
    rules = PostureRules.from_mapping({"v03_sitting_below_deg": 150.0})
    assert classify_bins("v03_hip", np.array([145.0]), np.array([0.5]), rules).tolist() == ["sitting"]
    with pytest.raises(ValueError):
        PostureRules.from_mapping({"v03_sitting_below": 150.0})


# ----------------------------------------------------------------------------- aggregation
def codes(text: str) -> np.ndarray:
    return np.array(["TSUO".index(c) for c in text], dtype=np.int8)


@pytest.mark.parametrize("text, label", [
    ("T" * 10, "standing"),
    ("S" * 10, "sitting"),
    ("T" * 8 + "S" * 2, "standing"),          # 2 contrary bins: not a sustained run
    ("T" * 7 + "S" * 3, "unclear"),           # 70% of decided: no majority
    ("T" * 4 + "U" * 6, "unclear"),           # decided < 50% of observed
    ("T" * 5 + "U" * 5, "standing"),          # decided == 50% of observed is enough
    ("T" * 4 + "O" * 6, "unknown"),           # observed < 50% of n
    ("TT" + "O", "unknown"),                  # fewer than 3 observed bins
    ("T" * 6 + "S" * 6, "mixed"),
    ("T" * 5 + "UUO" + "S" * 5, "mixed"),     # runs bridge across undecided bins
    ("T" * 3 + "U" + "T" * 3 + "S" * 5, "mixed"),  # 3+3 standing merge into one run of 6
])
def test_aggregate_label_rules(text: str, label: str) -> None:
    out = aggregate_states(codes(text))
    assert out["posture"] == label
    shares = [out[k] for k in AGGREGATE_KEYS[2:6]]
    assert sum(shares) == pytest.approx(1.0)
    if label in ("unclear", "unknown"):
        assert np.isnan(out["posture_confidence"])
    else:
        assert 0.0 < out["posture_confidence"] <= 1.0


def test_aggregate_confidence_and_transitions() -> None:
    out = aggregate_states(codes("T" * 9 + "U"))
    assert out["posture_confidence"] == pytest.approx(0.9)
    mixed = aggregate_states(codes("T" * 5 + "S" * 5 + "T" * 5 + "OO"))
    assert mixed["posture_transitions"] == 2
    assert mixed["posture_confidence"] == pytest.approx(15 / 17)
    # Transitions are a measure: counted even when the label is not 'mixed'.
    unclear = aggregate_states(codes("T" * 5 + "S" * 5 + "U" * 12))
    assert unclear["posture"] == "unclear" and unclear["posture_transitions"] == 1
    # String states and codes agree; the prefix only renames.
    as_strings = np.asarray(posture.STATES, dtype=object)[codes("T" * 5 + "S" * 5)]
    assert aggregate_states(as_strings) == aggregate_states(codes("T" * 5 + "S" * 5))
    prefixed = aggregate_states(codes("TTT"), prefix="recording_")
    assert set(prefixed) == {"recording_" + key for key in AGGREGATE_KEYS}
    with pytest.raises(ValueError):
        aggregate_states(np.array(["standing", "seated"], dtype=object))


def test_aggregate_without_bins_is_unmeasured() -> None:
    out = aggregate_states(np.zeros(0, dtype=np.int8))
    assert out["posture"] == "unknown" and out["posture_transitions"] is None
    assert all(np.isnan(out[k]) for k in AGGREGATE_KEYS[1:6])


def _assert_group_matches(grouped: pd.DataFrame, bins: pd.DataFrame, group_col: str, state_col: str,
                          rules: PostureRules) -> None:
    by_group = grouped.set_index(group_col)
    assert len(by_group) == bins[group_col].nunique()
    for group, part in bins.groupby(group_col, sort=False):
        expected = aggregate_states(part.sort_values("bin_index")[state_col].to_numpy(), rules)
        row = by_group.loc[group]
        for key in AGGREGATE_KEYS:
            value, want = row[key], expected[key]
            if isinstance(want, float) and math.isnan(want):
                assert math.isnan(value), (group, key)
            elif isinstance(want, float):
                assert value == pytest.approx(want, rel=1e-6), (group, key)
            else:
                assert value == want, (group, key)


def test_grouped_aggregator_equals_per_group_function() -> None:
    rng = np.random.default_rng(7)
    parts = []
    for group in range(400):
        n = int(rng.integers(1, 70))
        style = group % 4
        if style == 0:  # blocky sequences: sustained runs and transitions
            states = np.repeat(rng.integers(0, 4, size=8), rng.integers(1, 12, size=8))[:n]
        elif style == 1:
            states = rng.choice(4, size=n, p=[0.7, 0.1, 0.1, 0.1])
        elif style == 2:
            states = rng.choice(4, size=n, p=[0.1, 0.1, 0.1, 0.7])
        else:
            states = rng.integers(0, 4, size=n)
        states = states[:n] if states.size >= n else np.pad(states, (0, n - states.size), constant_values=3)
        parts.append(pd.DataFrame({"clip_id": f"c{group:04d}", "bin_index": np.arange(n, dtype=np.int32),
                                   "state": states.astype(np.int8)}))
    bins = pd.concat(parts, ignore_index=True)
    rules = PostureRules(min_run_bins=4)
    ordered = aggregate_groups(bins, "clip_id", "state", rules)
    _assert_group_matches(ordered, bins, "clip_id", "state", rules)
    assert ordered["posture_transitions"].sum() > 0 and (ordered["posture"] == "mixed").any()
    assert set(ordered["posture"]) >= {"standing", "unknown", "unclear", "mixed"}

    # Shuffled rows and string states give the same answer.
    shuffled = bins.sample(frac=1.0, random_state=3).reset_index(drop=True)
    shuffled["state"] = np.asarray(posture.STATES, dtype=object)[shuffled["state"].to_numpy()]
    again = aggregate_groups(shuffled, "clip_id", "state", rules).set_index("clip_id").loc[ordered["clip_id"]]
    pd.testing.assert_frame_equal(again.reset_index(), ordered, check_dtype=False)

    assert ordered["posture_confidence"].dtype == np.float32
    assert ordered["posture_transitions"].dtype == np.int16
    shares = ordered[list(AGGREGATE_KEYS[2:6])].sum(axis=1)
    assert np.allclose(shares, 1.0, atol=1e-6)


def test_grouped_aggregator_empty_and_prefix() -> None:
    empty = aggregate_groups(pd.DataFrame({"file_id": [], "bin_index": [], "state": []}), "file_id", "state",
                             prefix="recording_")
    assert list(empty.columns) == ["file_id"] + ["recording_" + key for key in AGGREGATE_KEYS]
    assert len(empty) == 0


# ----------------------------------------------------------------------------- bins and NaN semantics
def test_bins_schema_and_gates(model_root: Path) -> None:
    fps, frames = 30.0, 75  # 2.5 s: two full bins and a half bin
    visible = all_visible(frames)
    visible[0:15, COCO_KNEES[0]] = False  # bin 0: exactly 50% hip-observable -> measured
    visible[30:46, COCO_KNEES[0]] = False  # bin 1: 14/30 observable -> NaN
    pf = posture_frames(constant_joints(model_root, frames, STANDING_POSE), visible, smplh_anamorphic="none")
    bins = posture_bins(pf, fps)
    assert list(bins.columns) == list(posture.BIN_COLUMNS)
    assert bins["bin_index"].dtype == np.int32
    assert all(bins[c].dtype == np.float32 for c in posture.BIN_COLUMNS[1:])
    assert bins["bin_index"].tolist() == [0, 1, 2]
    assert bins["hip_observed_frac"].tolist() == pytest.approx([0.5, 14 / 30, 1.0])
    hip = bins["hip_flexion_deg"].to_numpy()
    assert np.isfinite(hip[0]) and np.isnan(hip[1]) and np.isfinite(hip[2])
    assert np.isnan(bins["knee_between"].to_numpy()[1])
    assert bins["shin_inverted_frac"].to_numpy()[2] == 0.0


def test_bins_nest_in_clips_at_29_97_fps() -> None:
    fps, frames = 30000 / 1001, 5400  # just over 180 s
    pf = posture.PostureFrames(
        hip_flexion=np.full(frames, 170.0), knee_between=np.full(frames, 0.5),
        shin_inverted=np.zeros(frames), hip_observable=np.ones(frames, bool),
        ankle_observable=np.ones(frames, bool),
    )
    bins = posture_bins(pf, fps)
    index, starts, stops = posture.bin_edges(frames, fps)
    assert starts[0] == 0 and stops[-1] == frames and np.all(starts[1:] == stops[:-1])
    clip_of_bin = bin_clip_index(index, 30.0)
    for clip in clip_grid(frames, fps, 30.0):
        inside = clip_of_bin == clip.clip_index
        assert starts[inside].min() == clip.start_frame and stops[inside].max() == clip.end_frame
    assert len(bins) == len(index)


def test_clip_measures_nan_rules(model_root: Path) -> None:
    fps = 47.95  # a released rate where 60 frames is not 2 s
    frames = 400
    visible = all_visible(frames)
    visible[:, COCO_ANKLES[1]] = False  # ankles out of shot everywhere
    visible[200:, COCO_HIPS[0]] = False  # hips visible in the first half only
    pf = posture_frames(constant_joints(model_root, frames, STANDING_POSE), visible,
                        smplh_anamorphic="none", fps=fps)
    full = clip_posture_measures(pf, 0, 400)
    assert full["hip_flexion_deg_p50"] == pytest.approx(172.0, abs=1.0)  # exactly 50%
    assert np.isnan(full["knee_between_p50"]) and np.isnan(full["shin_inverted_frac"])
    assert full["lower_body_observed_frac"] == 0.5
    late = clip_posture_measures(pf, 190, 400)  # 10/210 hip-observable
    assert np.isnan(late["hip_flexion_deg_p50"])
    short = clip_posture_measures(pf, 0, 95)  # < round(2 * 47.95) = 96 frames
    assert np.isnan(short["lower_body_observed_frac"])
    assert np.isfinite(short["hip_flexion_deg_p50"])  # the 50%-observable rule still applies
    assert clip_posture_measures(pf, 0, 96)["lower_body_observed_frac"] == 1.0
    # fps passed to the call wins over the stored one.
    assert np.isnan(clip_posture_measures(pf, 0, 96, fps=60.0)["lower_body_observed_frac"])
    empty = clip_posture_measures(pf, 400, 400)
    assert all(np.isnan(v) for v in empty.values())


def test_clip_measures_without_fps_warn_instead_of_raising(model_root: Path) -> None:
    pf = posture_frames(constant_joints(model_root, 30, STANDING_POSE), all_visible(30), smplh_anamorphic="none")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        out = clip_posture_measures(pf, 0, 30)
    assert any(issubclass(w.category, RuntimeWarning) for w in caught)
    assert out["lower_body_observed_frac"] == 1.0


def test_recording_measures_empty_recording() -> None:
    empty = posture.PostureFrames(*(np.zeros(0) for _ in range(3)), np.zeros(0, bool), np.zeros(0, bool))
    out = recording_posture_measures(empty)
    assert set(out) == {"recording_hip_flexion_deg_p50", "recording_knee_between_p50",
                        "recording_lower_body_observed_frac"}
    assert all(np.isnan(v) for v in out.values())
    assert len(posture_bins(empty, 30.0)) == 0


# ----------------------------------------------------------------------------- real file
#: file id -> (expected recording label, minimum transitions). P3720 sits
#: throughout; P3624 sits, stands (a perched block that reads as standing) and
#: sits again, so it must come out mixed.
REAL_FILES = {
    "V03_S1821_I00000010_P3720": ("sitting", 0),
    "V03_S1821_I00000010_P3624": ("mixed", 1),
}


@pytest.mark.parametrize("file_id, expected", sorted(REAL_FILES.items()))
def test_real_file_smoke(repo_root: Path, model_root: Path, file_id: str, expected: tuple[str, int]) -> None:
    inventory = repo_root / "outputs/02_inventory/summary/inventory_joined.parquet"
    if not inventory.exists():
        pytest.skip("inventory not available")
    table = pd.read_parquet(inventory, columns=["file_id", "vendor", "source_relbase", "video_width",
                                                "video_height", "video_avg_fps"])
    row = table.loc[table["file_id"] == file_id]
    if row.empty:
        pytest.skip(f"{file_id} not in the inventory")
    row = row.iloc[0]
    npz = repo_root / "seamless_interaction" / f"{row['source_relbase']}.npz"
    if not npz.exists():
        pytest.skip(f"{npz} not available")
    with np.load(npz) as bundle:
        pose = stack_pose(bundle["smplh:body_pose"], bundle["smplh:left_hand_pose"],
                          bundle["smplh:right_hand_pose"])
        keypoints = np.asarray(bundle["boxes_and_keypoints:keypoints"])
        box_valid = np.asarray(bundle["boxes_and_keypoints:is_valid_box"]).reshape(-1).astype(bool)
    joints, _, _ = forward_kinematics(pose, model_root)
    width, height = int(row["video_width"]), int(row["video_height"])
    x, y, conf = keypoints[..., 0], keypoints[..., 1], keypoints[..., 2]
    visible = (conf >= 0.5) & (x >= 0) & (x < width) & (y >= 0) & (y < height) & box_valid[:, None]
    fps = float(row["video_avg_fps"])
    pf = posture_frames(joints, visible, smplh_anamorphic="none", fps=fps)
    out = recording_label(pf, fps, str(row["vendor"]))
    label, min_transitions = expected
    assert out["posture"] == label
    assert out["posture_transitions"] >= min_transitions
    if label == "sitting":
        assert out["posture_transitions"] == 0
    measures = recording_posture_measures(pf)
    assert 0.0 <= measures["recording_lower_body_observed_frac"] <= 1.0

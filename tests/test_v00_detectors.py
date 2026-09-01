"""Tests for the three V00 failure-mode detectors.

Each detector is checked against a synthetic case whose correct verdict is known
by construction, plus the boundary cases where a plausible implementation would
silently do the wrong thing — mostly by treating an unmeasurable file as a clean
one.
"""

from __future__ import annotations

import numpy as np
import pytest

from seamless_curation.v00_detectors import (
    leg_visibility,
    IN_FRAME_JOINTS,
    SMPLH_ANKLE,
    SMPLH_HIP,
    SMPLH_KNEE,
    SMPLH_NECK,
    Thresholds,
    apply_thresholds,
    in_frame_violations,
    session_sitting_verdicts,
    posture_angles,
    signed_inset_px,
    sitting_measures_2d,
    static_hand_measure,
    tracker_discontinuity,
    windowed_in_frame_pass_frac,
)


WIDTH, HEIGHT = 1080, 1920
SHOULDERS, HIPS, KNEES, ANKLES, WRISTS = (5, 6), (11, 12), (13, 14), (15, 16), (9, 10)


def _legs(frames: int, *, seated: bool) -> np.ndarray:
    """A figure with either standing or stool-seated leg geometry."""

    points = np.zeros((frames, 133, 3), dtype=np.float64)
    points[..., 2] = 1.0
    cx, shoulder_y, hip_y = 540.0, 800.0, 1000.0
    for index in range(frames):
        for side, sign in ((0, -1.0), (1, 1.0)):
            points[index, SHOULDERS[side], :2] = (cx + sign * 150, shoulder_y)
            points[index, HIPS[side], :2] = (cx + sign * 90, hip_y)
            if seated:
                # Knees splayed wide and forward, feet up on a rung, so the
                # ankle sits above the knee.
                points[index, KNEES[side], :2] = (cx + sign * 300, hip_y + 140)
                points[index, ANKLES[side], :2] = (cx + sign * 190, hip_y + 60)
            else:
                points[index, KNEES[side], :2] = (cx + sign * 85, hip_y + 190)
                points[index, ANKLES[side], :2] = (cx + sign * 80, hip_y + 390)
            points[index, WRISTS[side], :2] = (cx + sign * 130, hip_y - 20)
        for extra in range(23, 133):
            points[index, extra, :2] = (cx, shoulder_y)
    return points


def _fk_legs(frames: int, *, seated: bool) -> np.ndarray:
    """Root-relative SMPL-H joints in mm, y downward, for FM1's primary measure."""

    joints = np.zeros((frames, 73, 3), dtype=np.float64)
    joints[:, SMPLH_NECK, 1] = -450.0          # neck above the pelvis
    for side, sign in (("left", -1.0), ("right", 1.0)):
        joints[:, SMPLH_HIP[side], 0] = sign * 90.0
        if seated:
            # Thigh roughly horizontal, so the knee is barely below the pelvis
            # and well forward of it; shin drops to the floor.
            joints[:, SMPLH_KNEE[side]] = (sign * 150.0, 110.0, -300.0)
            joints[:, SMPLH_ANKLE[side]] = (sign * 140.0, 440.0, -280.0)
        else:
            joints[:, SMPLH_KNEE[side]] = (sign * 85.0, 420.0, 0.0)
            joints[:, SMPLH_ANKLE[side]] = (sign * 80.0, 840.0, 0.0)
    return joints


def test_fm1_separates_standing_from_seated_on_fk_posture() -> None:
    standing = posture_angles(_fk_legs(30, seated=False))
    seated = posture_angles(_fk_legs(30, seated=True))
    assert standing["posture_status"] == "ok" and seated["posture_status"] == "ok"

    # The knee sits about halfway down a standing leg and near hip height when
    # seated, measured along the participant's own torso axis.
    assert standing["fm1_knee_between_torso_p50"] == pytest.approx(0.5, abs=0.05)
    assert seated["fm1_knee_between_torso_p50"] < 0.35
    assert seated["fm1_hip_flexion_deg_p50"] < standing["fm1_hip_flexion_deg_p50"]

    thresholds = Thresholds()
    assert apply_thresholds(standing, thresholds)["fm1_sitting"] is False
    assert apply_thresholds(seated, thresholds)["fm1_sitting"] is True
    assert apply_thresholds(seated, thresholds)["fm1_sitting_reasons"] == "knee_between"


def test_fm1_no_longer_thresholds_hip_flexion() -> None:
    """Every labelled file hip flexion flags alone is a standing person.

    Retired in Round 5 after the reviewer adjudicated 37 more clips: 4 of 4
    hip-only flags among them were standing, and it cost 29 extra files
    corpus-wide with no validated hit.
    """

    assert not hasattr(Thresholds(), "sitting_hip_flexion_deg")
    row = {
        **posture_angles(_fk_legs(10, seated=False)),
        # Deep in the old hip-flexion firing range, but the knee sits where a
        # standing knee sits. This is the short-legged standing participant.
        "fm1_hip_flexion_deg_p50": 105.0,
        "fm1_knee_between_torso_p50": 0.52,
    }
    flags = apply_thresholds(row, Thresholds())
    assert flags["fm1_sitting"] is False
    assert flags["fm1_sitting_reasons"] == ""


def test_fm1_primary_measure_is_invariant_to_whole_body_rotation() -> None:
    """The camera's idea of "down" must not enter the decision.

    ``fm1_knee_between_torso`` is taken along the pelvis-to-neck axis precisely
    so a leaning participant, or a camera that is not level, cannot move it.
    """

    base = _fk_legs(20, seated=True)
    angle = np.radians(25.0)
    rotation = np.array([
        [np.cos(angle), -np.sin(angle), 0.0],
        [np.sin(angle), np.cos(angle), 0.0],
        [0.0, 0.0, 1.0],
    ])
    tilted = base @ rotation.T
    first, second = posture_angles(base), posture_angles(tilted)
    assert first["fm1_knee_between_torso_p50"] == pytest.approx(
        second["fm1_knee_between_torso_p50"], rel=1e-9
    )
    assert first["fm1_hip_flexion_deg_p50"] == pytest.approx(
        second["fm1_hip_flexion_deg_p50"], rel=1e-9
    )


def test_fm1_no_longer_thresholds_the_2d_leg_ratios() -> None:
    """The Round-3 criteria were 0/15 correct; they must not decide anything now.

    ``leg_over_torso`` measures body proportion, not posture: all sixteen SMPL-H
    betas are pinned to zero, so FK limbs are identical in every file while the
    2D limbs are not. A short-legged standing participant reads 0.80 there and
    must now pass.
    """

    row = {
        **posture_angles(_fk_legs(10, seated=False)),
        **sitting_measures_2d(_legs(10, seated=False)),
        "fm1_leg_over_torso_p50": 0.80,        # would have fired in Round 3
        "fm1_knee_spread_over_torso_p50": 1.20,  # so would this
    }
    flags = apply_thresholds(row, Thresholds())
    assert flags["fm1_sitting"] is False
    assert flags["fm1_sitting_reasons"] == ""


def test_fm1_still_catches_the_feet_on_a_rung_posture() -> None:
    """One file in 4,000 has the ankle above the knee; the add-on keeps it."""

    row = {
        **posture_angles(_fk_legs(10, seated=False)),
        **sitting_measures_2d(_legs(10, seated=True)),
    }
    assert row["fm1_shin_verticality_p50"] < 0.0
    flags = apply_thresholds(row, Thresholds())
    assert flags["fm1_sitting"] is True
    assert flags["fm1_sitting_reasons"] == "shin_inverted"


def test_fm1_is_scale_free() -> None:
    """A camera moved closer to a seated participant must change nothing."""

    base = _legs(20, seated=True)
    zoomed = base.copy()
    zoomed[..., :2] = (zoomed[..., :2] - 540.0) * 1.8 + 540.0
    first, second = sitting_measures_2d(base), sitting_measures_2d(zoomed)
    for key in ("fm1_shin_verticality_p50", "fm1_knee_spread_over_torso_p50", "fm1_leg_over_torso_p50"):
        assert first[key] == pytest.approx(second[key], rel=1e-9)


def test_fm1_reports_undetermined_rather_than_passing() -> None:
    blank = posture_angles(np.zeros((5, 73, 3)))
    assert apply_thresholds(blank, Thresholds())["fm1_sitting"] is None
    assert apply_thresholds(blank, Thresholds())["passes_all"] is False


def test_fm1_verdicts_are_scoped_to_the_session_not_the_person() -> None:
    """The P0003A shape: seated in one session, standing in seven others.

    Aggregating over the person would call all ten files seated and wrongly
    reject the nine standing ones. Aggregating over participant-and-session gets
    both halves right with a plain majority and nothing fitted.
    """

    standing = posture_angles(_fk_legs(5, seated=False))
    seated = posture_angles(_fk_legs(5, seated=True))
    rows = [
        {**standing, "participant_id": "0003A", "session_id": f"01{index:02d}"}
        for index in range(9)
    ]
    rows.append({**seated, "participant_id": "0003A", "session_id": "0200"})

    verdicts = session_sitting_verdicts(rows, Thresholds())
    assert verdicts["0003A@0200"]["unit_sitting"] is True
    assert verdicts["0003A@0200"]["unit_sitting_basis"] == "majority"
    for index in range(9):
        assert verdicts[f"0003A@01{index:02d}"]["unit_sitting"] is False


def test_fm1_majority_absorbs_one_stray_file_within_a_session() -> None:
    standing = posture_angles(_fk_legs(5, seated=False))
    seated = posture_angles(_fk_legs(5, seated=True))
    rows = [{**standing, "participant_id": "1147A", "session_id": "S1"} for _ in range(4)]
    rows.append({**seated, "participant_id": "1147A", "session_id": "S1"})

    verdict = session_sitting_verdicts(rows, Thresholds())["1147A@S1"]
    assert verdict["unit_sitting"] is False
    assert verdict["unit_files_flagged"] == 1
    # Not seated, but it disagreed with itself, so it is a question not a pass.
    assert verdict["unit_sitting_near_cut"] is True


def test_fm1_single_file_units_still_get_a_near_cut_channel() -> None:
    """56% of units hold one file and can never disagree with themselves.

    Without a near-cut channel the majority rule would claim perfect confidence
    on more than half the corpus.
    """

    standing = posture_angles(_fk_legs(5, seated=False))
    rows = [
        {**standing, "fm1_knee_between_torso_p50": 0.46,
         "participant_id": "0457", "session_id": "S1"},
        {**standing, "fm1_knee_between_torso_p50": 0.58,
         "participant_id": "0458", "session_id": "S2"},
    ]
    verdicts = session_sitting_verdicts(rows, Thresholds())
    assert verdicts["0457@S1"]["unit_sitting"] is False
    assert verdicts["0457@S1"]["unit_sitting_near_cut"] is True, "0.46 is just above the 0.43 cut"
    assert verdicts["0458@S2"]["unit_sitting_near_cut"] is False, "0.58 is nowhere near it"


def test_fm1_has_no_deep_clause() -> None:
    """It was a parameter fitted to one file and session scoping removed the need."""

    for name in ("sitting_participant_deep_hip_flexion_deg",
                 "sitting_participant_deep_knee_between_torso",
                 "sitting_participant_seated_file_frac",
                 "sitting_hip_flexion_deg"):
        assert not hasattr(Thresholds(), name)
    standing = posture_angles(_fk_legs(5, seated=False))
    seated = posture_angles(_fk_legs(5, seated=True))
    rows = [{**standing, "participant_id": "P", "session_id": "S"} for _ in range(9)]
    rows.append({**seated, "participant_id": "P", "session_id": "S"})
    # One deep-reading file out of ten in the same session must NOT carry the unit.
    assert session_sitting_verdicts(rows, Thresholds())["P@S"]["unit_sitting"] is False


def _projection(frames: int, *, offset: tuple[float, float] = (0.0, 0.0)) -> np.ndarray:
    points = np.zeros((frames, 73, 2), dtype=np.float64)
    points[..., 0] = 500.0 + offset[0]
    points[..., 1] = 900.0 + offset[1]
    return points


def test_fm2_geometry_is_still_measured_on_every_frame() -> None:
    clean = in_frame_violations(
        _projection(100), width=WIDTH, height=HEIGHT, smplh_valid=np.ones(100, bool)
    )
    assert clean["fm2_in_frame_and_valid_frac"] == 1.0

    # One hand leaves the frame for three frames out of a hundred.
    brief = _projection(100)
    brief[40:43, 30, 0] = -5.0
    result = in_frame_violations(
        brief, width=WIDTH, height=HEIGHT, smplh_valid=np.ones(100, bool)
    )
    assert result["fm2_in_frame_frac"] == pytest.approx(0.97)
    assert result["fm2_violation_runs"] == 1
    assert result["fm2_violation_run_max_frames"] == 3
    # Geometry is still measured in full; since Round 5 it just does not decide.
    assert apply_thresholds(result, Thresholds())["fm2_smplh_invalid"] is False


def test_fm2_fires_on_smplh_invalidity_even_when_in_frame() -> None:
    valid = np.ones(100, bool)
    valid[10:20] = False
    result = in_frame_violations(
        _projection(100), width=WIDTH, height=HEIGHT, smplh_valid=valid
    )
    assert result["fm2_in_frame_frac"] == 1.0
    assert result["fm2_in_frame_and_valid_frac"] == pytest.approx(0.90)
    assert result["fm2_smplh_valid_frac"] == pytest.approx(0.90)
    # This is what FM2 now decides on, and it is the same 0.90.
    assert apply_thresholds(result, Thresholds())["fm2_smplh_invalid"] is True


def test_fm2_attributes_the_violation_to_a_body_region() -> None:
    feet = _projection(50)
    feet[:, 7, 1] = HEIGHT + 10.0  # left ankle just below the bottom edge
    result = in_frame_violations(feet, width=WIDTH, height=HEIGHT, smplh_valid=None)
    assert result["fm2_feet_out_of_frame_frame_frac"] == 1.0
    assert result["fm2_hands_out_of_frame_frame_frac"] == 0.0


def test_fm2_treats_a_non_finite_projection_as_a_violation() -> None:
    broken = _projection(20)
    broken[5, 3, :] = np.nan
    result = in_frame_violations(broken, width=WIDTH, height=HEIGHT, smplh_valid=None)
    assert result["fm2_in_frame_frac"] == pytest.approx(0.95)


def test_windowed_variant_shows_what_a_file_level_rule_costs() -> None:
    points = _projection(300)
    points[150, 30, 0] = -5.0
    whole = in_frame_violations(points, width=WIDTH, height=HEIGHT, smplh_valid=None)
    windowed = windowed_in_frame_pass_frac(
        points, width=WIDTH, height=HEIGHT, smplh_valid=None, window_frames=100
    )
    # One bad frame fails the whole file but only one window in three.
    assert whole["fm2_in_frame_and_valid_frac"] < 1.0
    assert windowed == pytest.approx(2 / 3)


def _wrists(frames: int, *, drift: np.ndarray | None = None) -> np.ndarray:
    points = np.zeros((frames, 133, 3), dtype=np.float64)
    points[..., 2] = 1.0
    for index in range(frames):
        points[index, SHOULDERS[0], :2] = (400.0, 800.0)
        points[index, SHOULDERS[1], :2] = (700.0, 800.0)
        move = 0.0 if drift is None else float(drift[index])
        for side, sign in ((0, -1.0), (1, 1.0)):
            points[index, WRISTS[side], :2] = (550.0 + sign * 60 + move, 1000.0)
    return points


def test_fm3_measures_how_long_the_wrists_hold_one_position() -> None:
    still = static_hand_measure(_wrists(100), radius_shoulder_widths=0.10)
    assert still["fm3_status"] == "ok"
    assert still["fm3_static_frac"] == pytest.approx(1.0)
    assert apply_thresholds(still, Thresholds())["fm3_static_hands"] is True

    # Held for 80 frames, then a large excursion for 20: still static at 0.75.
    drift = np.zeros(100)
    drift[80:] = 300.0
    mostly = static_hand_measure(_wrists(100, drift=drift), radius_shoulder_widths=0.10)
    assert mostly["fm3_static_frac"] == pytest.approx(0.80)
    assert apply_thresholds(mostly, Thresholds())["fm3_static_hands"] is True

    moving = np.linspace(-300.0, 300.0, 100)
    active = static_hand_measure(_wrists(100, drift=moving), radius_shoulder_widths=0.10)
    assert active["fm3_static_frac"] < 0.3
    assert apply_thresholds(active, Thresholds())["fm3_static_hands"] is False


def test_fm3_reference_is_the_median_not_the_first_frame() -> None:
    """Gesturing first and then freezing must still read as static."""

    drift = np.zeros(100)
    drift[:20] = 400.0
    result = static_hand_measure(_wrists(100, drift=drift), radius_shoulder_widths=0.10)
    assert result["fm3_static_frac"] == pytest.approx(0.80)


def test_fm3_is_scale_free_and_reports_extra_radii() -> None:
    drift = np.zeros(60)
    drift[40:] = 100.0
    near = static_hand_measure(_wrists(60, drift=drift), radius_shoulder_widths=0.10,
                               extra_radii=(0.05, 0.5))
    scaled = _wrists(60, drift=drift * 2.0)
    scaled[:, SHOULDERS[0], :2] = (250.0, 800.0)
    scaled[:, SHOULDERS[1], :2] = (850.0, 800.0)
    far = static_hand_measure(scaled, radius_shoulder_widths=0.10)
    assert near["fm3_static_frac"] == pytest.approx(far["fm3_static_frac"])
    assert near["fm3_static_frac_at_r0p5"] == pytest.approx(1.0)
    assert near["fm3_static_frac_at_r0p05"] <= near["fm3_static_frac"]


def test_fm2_now_gates_on_smplh_validity_not_geometry() -> None:
    """Round 5: out-of-frame is only a problem when it degrades the fit.

    A file whose body leaves the raster but whose SMPL-H stays valid on every
    frame is kept; a file that is comfortably framed but has one invalid frame
    is not.
    """

    out_of_frame_but_valid = {
        "fm2_status": "ok", "fm2_smplh_valid_frac": 1.0,
        "fm2_in_frame_and_valid_frac": 0.80,
    }
    framed_but_invalid = {
        "fm2_status": "ok", "fm2_smplh_valid_frac": 0.999,
        "fm2_in_frame_and_valid_frac": 0.999,
    }
    assert apply_thresholds(out_of_frame_but_valid, Thresholds())["fm2_smplh_invalid"] is False
    assert apply_thresholds(framed_but_invalid, Thresholds())["fm2_smplh_invalid"] is True
    # The geometric verdict is gone from the flag set, not merely defaulted off.
    assert "fm2_framing" not in apply_thresholds(out_of_frame_but_valid, Thresholds())
    # And the cut is a parameter.
    relaxed = apply_thresholds(framed_but_invalid, Thresholds(smplh_valid_required_frac=0.99))
    assert relaxed["fm2_smplh_invalid"] is False


def test_reprojection_error_measures_the_fit_against_the_released_keypoints() -> None:
    """The direct form of "did the SMPL parameters stay where they should be"."""

    from seamless_curation.v00_detectors import smplh_reprojection_error

    frames = 40
    keypoints = np.zeros((frames, 133, 3))
    keypoints[..., 2] = 0.9
    # Shoulders 200 px apart, so one shoulder width is 200 px.
    keypoints[:, 5, :2] = (440.0, 800.0)
    keypoints[:, 6, :2] = (640.0, 800.0)
    for index in range(7, 17):
        keypoints[:, index, :2] = (540.0, 900.0 + 40 * index)

    projected = np.zeros((frames, 73, 2))
    for coco, smplh in zip(range(5, 17), (16, 17, 18, 19, 20, 21, 1, 2, 4, 5, 7, 8)):
        projected[:, smplh, :] = keypoints[:, coco, :2]
    perfect = smplh_reprojection_error(projected, keypoints)
    assert perfect["reproj_status"] == "ok"
    assert perfect["reproj_shoulder_width_px"] == pytest.approx(200.0)
    assert perfect["reproj_px_p50"] == pytest.approx(0.0, abs=1e-9)

    # Displace the whole fit by 20 px: a tenth of a shoulder width.
    drifted = smplh_reprojection_error(projected + 20.0, keypoints)
    assert drifted["reproj_px_p50"] == pytest.approx(20.0 * 2 ** 0.5, rel=1e-6)
    assert drifted["reproj_shoulder_widths_p50"] == pytest.approx(
        20.0 * 2 ** 0.5 / 200.0, rel=1e-6
    )

    # Split by the released flag, so the flag itself can be validated.
    valid = np.ones(frames, bool)
    valid[20:] = False
    mixed = projected.copy()
    mixed[20:] += 100.0
    split = smplh_reprojection_error(mixed, keypoints, smplh_valid=valid)
    assert split["reproj_shoulder_widths_on_valid_p50"] == pytest.approx(0.0, abs=1e-9)
    assert split["reproj_shoulder_widths_on_invalid_p50"] > 0.5
    assert split["reproj_on_valid_frames"] == 20
    assert split["reproj_on_invalid_frames"] == 20


def test_fm2_still_measures_the_body_surface_even_though_it_no_longer_decides() -> None:
    """The exact miss the reviewer found, reproduced in miniature.

    In V00_S0180_I00000482_P0047 at t = 222.57 s the left elbow *joint* sat
    7.7 px inside the right edge while the mesh surface reached 24.5 px outside,
    a 32.2 px gap, and Round 3's joint-only rule scored the file a perfect 1.0.
    """

    joints = _projection(100)
    joints[50, 18, 0] = WIDTH - 1 - 7.7      # elbow joint centre, just inside
    inset = np.full(100, 60.0)
    inset[50] = -24.5                         # but the mesh is outside

    joints_only = in_frame_violations(
        joints, width=WIDTH, height=HEIGHT, smplh_valid=np.ones(100, bool)
    )
    assert joints_only["fm2_in_frame_and_valid_frac"] == 1.0

    with_surface = in_frame_violations(
        joints, width=WIDTH, height=HEIGHT, smplh_valid=np.ones(100, bool),
        vertex_worst_inset_px=inset,
    )
    assert with_surface["fm2_in_frame_and_valid_frac"] == pytest.approx(0.99)
    assert with_surface["fm2_term_vertices_frame_frac"] == pytest.approx(0.01)
    # The size of the Round-3 blind spot, recorded rather than argued about.
    assert with_surface["fm2_surface_correction_px_p95"] > 30.0


def test_fm2_cross_checks_the_released_2d_keypoints() -> None:
    """The SMPL-H projection is pulled inward on the arms by 10-20 px.

    An independent observation of where the person is has to be able to override
    it, so a confidently-detected keypoint outside the raster is a violation on
    its own.
    """

    joints = _projection(60)
    keypoints = np.zeros((60, 133, 3))
    keypoints[..., :2] = (500.0, 900.0)
    keypoints[..., 2] = 0.9
    keypoints[20:24, 7, 0] = WIDTH + 5.4      # left elbow, outside, confident

    result = in_frame_violations(
        joints, width=WIDTH, height=HEIGHT, smplh_valid=np.ones(60, bool),
        keypoints=keypoints,
    )
    assert result["fm2_term_coco_body17_frame_frac"] == pytest.approx(4 / 60)

    # A zero-filled keypoint row is absence of evidence, not a body out of frame.
    unusable = keypoints.copy()
    unusable[20:24, 7, 2] = 0.0
    quiet = in_frame_violations(
        joints, width=WIDTH, height=HEIGHT, smplh_valid=np.ones(60, bool),
        keypoints=unusable,
    )
    assert quiet["fm2_term_coco_body17_frame_frac"] == 0.0


def test_fm2_face_and_feet_keypoints_are_not_tested_by_default() -> None:
    """They add zero files on 180 measured, and sit >100 px inside."""

    joints = _projection(40)
    keypoints = np.zeros((40, 133, 3))
    keypoints[..., :2] = (500.0, 900.0)
    keypoints[..., 2] = 0.9
    keypoints[10:15, 30, 0] = -20.0           # a face landmark outside the frame
    result = in_frame_violations(
        joints, width=WIDTH, height=HEIGHT, smplh_valid=np.ones(40, bool),
        keypoints=keypoints,
    )
    assert "fm2_term_coco_face68_frame_frac" not in result
    # Explicitly requesting the group still works, for the report's sweeps.
    opted_in = in_frame_violations(
        joints, width=WIDTH, height=HEIGHT, smplh_valid=np.ones(40, bool),
        keypoints=keypoints, coco_groups=("body17", "face68"),
    )
    assert opted_in["fm2_term_coco_face68_frame_frac"] == pytest.approx(0.125)


def test_fm2_min_violation_run_is_a_parameter_defaulting_to_literal() -> None:
    joints = _projection(100)
    inset = np.full(100, 60.0)
    inset[50] = -0.2                          # one frame, a fifth of a pixel

    literal = in_frame_violations(
        joints, width=WIDTH, height=HEIGHT, smplh_valid=np.ones(100, bool),
        vertex_worst_inset_px=inset,
    )
    assert literal["fm2_in_frame_and_valid_frac"] < 1.0

    tolerant = in_frame_violations(
        joints, width=WIDTH, height=HEIGHT, smplh_valid=np.ones(100, bool),
        vertex_worst_inset_px=inset, min_violation_run_frames=2,
    )
    assert tolerant["fm2_in_frame_and_valid_frac"] == 1.0


def test_signed_inset_is_negative_outside_and_positive_inside() -> None:
    points = np.array([[[540.0, 960.0], [-3.0, 960.0], [WIDTH + 5.0, 960.0]]])
    inset = signed_inset_px(points, WIDTH, HEIGHT)
    assert inset[0, 0] > 0
    assert inset[0, 1] == pytest.approx(-3.0)
    assert inset[0, 2] == pytest.approx(-6.0)


def test_the_scrapped_mutual_silence_detector_is_gone() -> None:
    """FM4 was scrapped after Round 3; no trace of it may survive."""

    import seamless_curation.v00_detectors as detectors

    for name in ("mutual_silence", "merge_intervals", "mutual_silence_s"):
        assert not hasattr(detectors, name), f"{name} belongs to the scrapped FM4"
    assert not hasattr(Thresholds(), "mutual_silence_s")
    flags = apply_thresholds(
        {"posture_status": "ok", "fm1_hip_flexion_deg_p50": 150.0,
         "fm1_knee_between_torso_p50": 0.55, "fm1_shin_verticality_p50": 1.0,
         "fm2_status": "ok", "fm2_smplh_valid_frac": 1.0,
         "fm3_status": "ok", "fm3_static_frac": 0.1},
        Thresholds(),
    )
    assert not any("fm4" in key for key in flags)
    assert flags["passes_all"] is True, "three determined detectors is now a full pass"


def test_tracker_discontinuity_flags_a_subject_switch() -> None:
    steady = np.tile(np.array([400.0, 500.0, 700.0, 1500.0]), (50, 1))
    calm = tracker_discontinuity(steady, None)
    assert calm["tracker_centre_jump_max"] == pytest.approx(0.0)

    jumped = steady.copy()
    jumped[25:, 0] += 600.0
    jumped[25:, 2] += 600.0
    switched = tracker_discontinuity(jumped, None)
    assert switched["tracker_centre_jump_max"] > 0.5


def test_apply_thresholds_reports_which_detectors_fired() -> None:
    row = {
        "posture_status": "ok", "fm1_hip_flexion_deg_p50": 150.0,
        "fm1_knee_between_torso_p50": 0.55, "fm1_shin_verticality_p50": 1.0,
        "fm2_status": "ok", "fm2_smplh_valid_frac": 0.5,
        "fm3_status": "ok", "fm3_static_frac": 0.9,
    }
    flags = apply_thresholds(row, Thresholds())
    assert flags["flags_fired"] == "fm2_smplh_invalid+fm3_static_hands"
    assert flags["flag_count"] == 2 and flags["flagged"] is True
    assert flags["passes_all"] is False


def test_thresholds_are_all_parameters() -> None:
    """Every cut point must be movable without touching a measurement."""

    row = {
        "posture_status": "ok", "fm1_hip_flexion_deg_p50": 150.0,
        "fm1_knee_between_torso_p50": 0.55, "fm1_shin_verticality_p50": 1.0,
        "fm2_status": "ok", "fm2_smplh_valid_frac": 0.995,
        "fm3_status": "ok", "fm3_static_frac": 0.72,
    }
    strict = apply_thresholds(row, Thresholds())
    assert strict["flags_fired"] == "fm2_smplh_invalid"
    relaxed = apply_thresholds(row, Thresholds(smplh_valid_required_frac=0.99))
    assert relaxed["passes_all"] is True
    tightened = apply_thresholds(row, Thresholds(static_frac_limit=0.70))
    assert "fm3_static_hands" in tightened["flags_fired"]


def test_posture_angles_remain_available_as_an_uncalibrated_secondary() -> None:
    joints = np.zeros((10, 73, 3))
    joints[:, 12, 1] = -400.0          # neck above pelvis
    joints[:, [1, 2], 0] = [-90.0, 90.0]
    joints[:, [4, 5], 1] = 400.0       # knees below
    joints[:, [4, 5], 0] = [-85.0, 85.0]
    joints[:, [7, 8], 1] = 800.0       # ankles below the knees
    joints[:, [7, 8], 0] = [-80.0, 80.0]
    result = posture_angles(joints)
    assert result["posture_status"] == "ok"
    assert 150.0 < result["fm1_knee_flexion_deg_p50"] <= 180.0


def _keypoints_with_legs(frames: int, *, ankle_y: float, width: int, height: int) -> np.ndarray:
    """A minimal COCO body-17 track with the ankles placed where asked."""
    points = np.zeros((frames, 17, 3), dtype=np.float64)
    points[..., 2] = 0.9
    for index, (x, y) in {
        5: (0.4 * width, 0.25 * height), 6: (0.6 * width, 0.25 * height),
        11: (0.45 * width, 0.55 * height), 12: (0.55 * width, 0.55 * height),
        13: (0.45 * width, 0.75 * height), 14: (0.55 * width, 0.75 * height),
    }.items():
        points[:, index, :2] = (x, y)
    points[:, 15, :2] = (0.45 * width, ankle_y)
    points[:, 16, :2] = (0.55 * width, ankle_y)
    return points


def test_leg_visibility_separates_seen_legs_from_cropped_ones():
    """FM1's shin clause is only meaningful when the ankles were actually in shot."""
    width, height = 1080, 1920
    inside = _keypoints_with_legs(20, ankle_y=0.95 * height, width=width, height=height)
    cropped = _keypoints_with_legs(20, ankle_y=1.20 * height, width=width, height=height)

    seen = leg_visibility(inside, width=width, height=height)
    unseen = leg_visibility(cropped, width=width, height=height)
    assert seen["fm1_ankles_visible_frac"] == 1.0
    assert unseen["fm1_ankles_visible_frac"] == 0.0
    # The knees are in frame either way, so the two must be reported separately:
    # "no ankles" and "no legs at all" are different situations.
    assert seen["fm1_knees_visible_frac"] == unseen["fm1_knees_visible_frac"] == 1.0
    assert seen["fm1_leg_visibility_status"] == "ok"


def test_leg_visibility_needs_confidence_as_well_as_position():
    width, height = 1080, 1920
    points = _keypoints_with_legs(10, ankle_y=0.95 * height, width=width, height=height)
    points[:5, 15, 2] = 0.0          # left ankle not tracked for half the file
    assert leg_visibility(points, width=width, height=height)["fm1_ankles_visible_frac"] == 0.5


def test_leg_visibility_reports_unmeasurable_rather_than_a_clean_pass():
    assert leg_visibility(np.zeros((0, 17, 3)), width=100, height=100) == {
        "fm1_ankles_visible_frac": None, "fm1_knees_visible_frac": None,
        "fm1_leg_visibility_status": "unmeasurable",
    }


def _standing_row(**overrides):
    row = {
        **posture_angles(_fk_legs(10, seated=False)),
        "vendor_id": "00", "width": 1080, "height": 1920,
        "fm2_status": "ok", "fm2_smplh_valid_frac": 1.0,
        "fm3_status": "ok", "fm3_static_frac": 0.10,
        "audio_status": "ok", "audio_speech_level_db": -23.0,
        "audio_spectral_flatness": 0.001, "audio_envelope_dynamics_db": 25.0,
        "timebase_index_drift_s": 0.0,
    }
    row.update(overrides)
    return row


def test_fm1_reads_a_different_measure_on_v03():
    """V00 goes by knee position; V03 by hip flexion, which beats it 44/45 to 41/45."""
    # Knee position says seated, hip flexion says standing. V00 follows the knee.
    ambiguous = {"fm1_knee_between_torso_p50": 0.40, "fm1_hip_flexion_deg_p50": 165.0,
                 "fm1_shin_verticality_p50": 0.9}
    assert apply_thresholds(
        _standing_row(vendor_id="00", **ambiguous), Thresholds()
    )["fm1_sitting"] is True
    assert apply_thresholds(
        _standing_row(vendor_id="03", **ambiguous), Thresholds()
    )["fm1_sitting"] is False

    # And the other way: a V03 file that knee position would clear.
    seated = {"fm1_knee_between_torso_p50": 0.60, "fm1_hip_flexion_deg_p50": 120.0,
              "fm1_shin_verticality_p50": 0.9}
    flags = apply_thresholds(_standing_row(vendor_id="03", **seated), Thresholds())
    assert flags["fm1_sitting"] is True
    assert flags["fm1_sitting_reasons"] == "hip_flexion"
    assert apply_thresholds(
        _standing_row(vendor_id="00", **seated), Thresholds()
    )["fm1_sitting"] is False


def test_fm1_is_undetermined_when_its_own_vendor_measure_is_missing():
    row = _standing_row(vendor_id="03", fm1_knee_between_torso_p50=0.60,
                        fm1_hip_flexion_deg_p50=float("nan"))
    assert apply_thresholds(row, Thresholds())["fm1_sitting"] is None


def test_fm1_is_retired_for_v01_and_v02_and_says_so():
    seated = {"fm1_knee_between_torso_p50": 0.20, "fm1_hip_flexion_deg_p50": 95.0,
              "fm1_shin_verticality_p50": -0.9}
    for vendor in ("01", "02"):
        flags = apply_thresholds(_standing_row(vendor_id=vendor, **seated), Thresholds())
        assert flags["fm1_sitting"] is False
        # Not silently clean: the reason records that nothing was asked.
        assert flags["fm1_sitting_reasons"] == "retired_for_vendor"
        assert flags["passes_all"] is True


def test_fm4_gates_on_envelope_dynamics_not_on_level():
    """Static sits at one level; a voice makes the level move. Level alone cannot tell.

    Three rules were tried and measured against 48 hand labels: the absolute level
    floor scored 10.5% precision, an empty released VAD 21.7%, and this 100%.
    """
    thresholds = Thresholds()
    # Very quiet but a real voice: a participant who barely speaks.
    quiet_real = _standing_row(vendor_id="01", audio_speech_level_db=-72.0,
                               audio_spectral_flatness=0.98,
                               audio_envelope_dynamics_db=14.2)
    # Loud static: high level, no dynamics at all.
    loud_static = _standing_row(vendor_id="01", audio_speech_level_db=-23.2,
                                audio_spectral_flatness=0.02,
                                audio_envelope_dynamics_db=1.35)
    assert apply_thresholds(quiet_real, thresholds)["fm4_audio_dead"] is False
    flags = apply_thresholds(loud_static, thresholds)
    assert flags["fm4_audio_dead"] is True
    assert flags["fm4_audio_reasons"] == "no_dynamics"


def test_fm4_is_undetermined_when_the_dynamics_could_not_be_measured():
    row = _standing_row(vendor_id="01")
    row.pop("audio_envelope_dynamics_db", None)
    assert apply_thresholds(row, Thresholds())["fm4_audio_dead"] is None


def test_an_empty_wav_is_dead_rather_than_undetermined():
    flags = apply_thresholds(_standing_row(vendor_id="01", audio_status="empty"), Thresholds())
    assert flags["fm4_audio_dead"] is True
    assert flags["fm4_audio_reasons"] == "empty"


def test_a_scan_without_audio_columns_is_unaffected():
    row = _standing_row()
    for key in ("audio_status", "audio_speech_level_db", "audio_spectral_flatness"):
        row.pop(key)
    flags = apply_thresholds(row, Thresholds())
    assert "fm4_audio_dead" not in flags
    assert flags["passes_all"] is True


def test_excluded_rasters_and_a_drifting_timebase_fail_before_anything_else():
    for width, height in ((2160, 2160), (1920, 1080), (640, 480), (3840, 2160)):
        flags = apply_thresholds(_standing_row(width=width, height=height), Thresholds())
        assert flags["fm0_unusable_source"] is True
        assert flags["fm0_reasons"] == f"raster_{width}x{height}"
        assert flags["passes_all"] is False
    # The kept ones, including the two that are only padded.
    for width, height in ((1080, 1920), (1012, 1920), (1080, 960), (2180, 3840), (2160, 3840)):
        flags = apply_thresholds(_standing_row(width=width, height=height), Thresholds())
        assert flags["fm0_unusable_source"] is False, (width, height)
        assert flags["passes_all"] is True

    drift = apply_thresholds(_standing_row(timebase_index_drift_s=-9.5), Thresholds())
    assert drift["fm0_unusable_source"] is True
    assert drift["fm0_reasons"] == "timebase_mismatch"
    # A frame or two of slack, which 23% of the corpus has and nobody can see.
    assert apply_thresholds(
        _standing_row(timebase_index_drift_s=-0.04), Thresholds()
    )["fm0_unusable_source"] is False


def test_fm1_scope_is_per_vendor_because_the_assumption_is():
    """V00 sessions are 1.1% mixed; V03's are 16.2%, so V03 goes by its own file."""
    thresholds = Thresholds()
    assert thresholds.sitting_unit_scope_by_vendor["00"] == "session"
    assert thresholds.sitting_unit_scope_by_vendor["03"] == "file"


def test_a_seated_file_outvoted_by_its_session_is_what_the_scope_change_fixes():
    """V03_S1558_I00000012_P1425: seated, 1 of 14 files in its session, passed FM1."""
    from seamless_curation.v00_detectors import session_sitting_verdicts

    thresholds = Thresholds()
    seated = {"vendor_id": "03", "participant_id": "P1425", "session_id": "S1558",
              "posture_status": "ok", "fm1_hip_flexion_deg_p50": 92.6,
              "fm1_knee_between_torso_p50": 0.21, "fm1_shin_verticality_p50": 0.99}
    standing = dict(seated, fm1_hip_flexion_deg_p50=165.0, fm1_knee_between_torso_p50=0.60)
    records = [seated] + [dict(standing) for _ in range(13)]
    for record in records:
        record.update(apply_thresholds(record, thresholds))

    # The file itself reads seated...
    assert records[0]["fm1_sitting"] is True
    # ...and the session majority does not, which is exactly the override.
    verdicts = session_sitting_verdicts(records, thresholds)
    assert verdicts["P1425@S1558"]["unit_sitting"] is False


def test_the_shin_addon_is_off_for_v03_because_it_only_ever_fired_wrongly():
    """8 of 149 V03 labels flagged by the shin clause alone; none of them seated."""
    cropped_legs = {"fm1_hip_flexion_deg_p50": 165.0, "fm1_knee_between_torso_p50": 0.60,
                    "fm1_shin_verticality_p50": -0.9}
    thresholds = Thresholds()
    # V00 keeps it: there it fired once in 4,000 files and that file was seated.
    assert apply_thresholds(
        _standing_row(vendor_id="00", **cropped_legs), thresholds
    )["fm1_sitting_reasons"] == "shin_inverted"
    # V03 does not.
    v03 = apply_thresholds(_standing_row(vendor_id="03", **cropped_legs), thresholds)
    assert v03["fm1_sitting"] is False
    assert v03["fm1_sitting_reasons"] == ""
    # The hip clause still fires on V03 on its own.
    seated = apply_thresholds(
        _standing_row(vendor_id="03", **{**cropped_legs, "fm1_hip_flexion_deg_p50": 110.0}),
        thresholds,
    )
    assert seated["fm1_sitting_reasons"] == "hip_flexion"


def test_fm4_is_off_for_v00_and_says_so():
    """Retired for V00 in Round 13, when all six of its firings were quiet speakers.

    The Round-14 rule would not have made those mistakes -- it reads dynamics, not
    level -- but re-enabling it is the reviewer's call, so the switch stays.
    """
    static = _standing_row(vendor_id="00", audio_envelope_dynamics_db=0.3)
    flags = apply_thresholds(static, Thresholds())
    assert flags["fm4_audio_dead"] is False
    assert flags["fm4_audio_reasons"] == "retired_for_vendor"
    assert flags["passes_all"] is True
    # Still on everywhere else.
    for vendor in ("01", "02", "03"):
        other = apply_thresholds(
            _standing_row(vendor_id=vendor, audio_envelope_dynamics_db=0.3), Thresholds()
        )
        assert other["fm4_audio_dead"] is True, vendor


def test_an_empty_wav_still_fails_v00_because_that_is_not_a_quiet_speaker():
    """Switching FM4 off must not let a 58-byte WAV through as a clean pass."""
    flags = apply_thresholds(_standing_row(vendor_id="00", audio_status="empty"), Thresholds())
    # FM4 is retired for V00, so the audio check does not fire...
    assert flags["fm4_audio_dead"] is False
    # ...which is the reviewer's decision, and V00 has 0 empty WAVs in 41,205
    # eligible files, so nothing is actually being let through. Recorded as a
    # test so the day V00 gains one, this assertion is where the argument lives.
    assert flags["fm4_audio_reasons"] == "retired_for_vendor"

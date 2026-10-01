"""Framing, visibility and tracker continuity, on arrays whose truth we set.

The synthetic payloads here are built directly in the released layout
(``boxes_and_keypoints:*``, ``smplh:global_orient``, ``smplh:translation``); the
reprojection tests build camera-consistent keypoints by projecting known joints
with the same HMR camera, so the expected error is exact. None of them needs the
SMPL-H model. One smoke test at the end runs on real files when the release is
mounted, and skips otherwise.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from seamless_curation import framing, schema
from seamless_curation.clips import clip_grid
from seamless_curation.smplh_fk import project_hmr2_full_frame

FPS = 30.0
W, H = 1080, 1920
UPRIGHT = (np.pi, 0.0, 0.0)  # HMR camera: y down, so an upright body is 180 deg about x

#: Pelvis-frame positions (m) of the eight SMPL-H joints the reprojection uses,
#: in REPROJECTION_SMPLH order: L/R shoulder, L/R elbow, L/R wrist, L/R hip.
BODY8 = np.array([
    [0.18, 0.30, 0.0], [-0.18, 0.30, 0.0],
    [0.24, 0.05, 0.0], [-0.24, 0.05, 0.0],
    [0.26, -0.20, 0.05], [-0.26, -0.20, 0.05],
    [0.09, -0.08, 0.0], [-0.09, -0.08, 0.0],
])


def joints_pelvis(frames: int) -> np.ndarray:
    joints = np.zeros((frames, 52, 3))
    joints[:, list(framing.REPROJECTION_SMPLH)] = BODY8
    return joints


def make_payload(
    frames: int = 300,
    *,
    width: int = W,
    height: int = H,
    yaw_deg: np.ndarray | float = 0.0,
    translation=(0.0, 0.1, 40.0),
    conf: float = 0.9,
    box=(340.0, 400.0, 740.0, 1700.0),
) -> dict[str, np.ndarray]:
    """A person standing in the picture, keypoints = projection of :data:`BODY8`."""

    yaw = np.broadcast_to(np.asarray(yaw_deg, dtype=np.float64), (frames,))
    rotations = Rotation.from_rotvec(np.tile(UPRIGHT, (frames, 1))) * Rotation.from_euler("y", yaw, degrees=True)
    orient = rotations.as_rotvec()
    trans = np.tile(np.asarray(translation, dtype=np.float64), (frames, 1))
    camera = np.einsum("nij,kj->nki", rotations.as_matrix(), BODY8) + framing.REST_PELVIS_M + trans[:, None]
    projected = project_hmr2_full_frame(camera, width, height)

    keypoints = np.zeros((frames, 133, 3), dtype=np.float32)
    keypoints[:, :, 0] = width / 2
    keypoints[:, :, 1] = height / 2
    keypoints[:, :, 2] = conf
    keypoints[:, 5:13, :2] = projected
    keypoints[:, 13, :2] = (width / 2 + 60, height * 0.75)   # knees
    keypoints[:, 14, :2] = (width / 2 - 60, height * 0.75)
    keypoints[:, 15, :2] = (width / 2 + 60, height * 0.95)   # ankles
    keypoints[:, 16, :2] = (width / 2 - 60, height * 0.95)
    return {
        "boxes_and_keypoints:keypoints": keypoints,
        "boxes_and_keypoints:is_valid_box": np.ones(frames, dtype=bool),
        "boxes_and_keypoints:box": np.tile(np.asarray(box, dtype=np.float32), (frames, 1)),
        "smplh:global_orient": orient.astype(np.float32),
        "smplh:translation": trans.astype(np.float32),
    }


def track_for(payload, *, width=W, height=H, quarter_turns=0, anamorphic="none", joints=True):
    frames = len(payload["boxes_and_keypoints:keypoints"])
    return framing.framing_track(
        payload, joints_pelvis(frames) if joints else None, fps=FPS, width=width, height=height,
        quarter_turns=quarter_turns, smplh_anamorphic=anamorphic,
    )


# ============================================================ visibility
def test_confident_keypoints_outside_the_raster_are_not_visible() -> None:
    """Keypoints are extrapolated beyond the picture; confidence alone is not visibility."""

    points = np.zeros((4, 133, 3), dtype=np.float32)
    points[:, :, 2] = 0.9
    points[:, 0, :2] = (500, 900)       # inside
    points[:, 1, :2] = (500, 1950)      # below the raster at conf 0.9
    points[:, 2, :2] = (-3, 900)        # left of it
    points[:, 3, :2] = (W, 900)         # x == width is outside (half-open)
    points[:, 4, :2] = (500, 900)
    points[:, 4, 2] = 0.49              # inside but not confident
    points[:, 5, :2] = (500, 900)
    points[:, 5, 2] = 1.3               # confidence exceeds 1 in this release
    box_valid = np.array([True, True, True, False])

    visible = framing.visible_mask(points, W, H, box_valid)

    assert visible[:3, 0].all() and visible[:3, 5].all()
    assert not visible[:, 1:5].any()
    assert not visible[3].any(), "an invalid box hides everything (zero-fill frames)"
    assert not framing.visible_mask(points, 0, 0, box_valid).any(), "unknown raster: nothing provable"


def test_pair_visibility_fractions_per_clip() -> None:
    payload = make_payload(300)
    kp = payload["boxes_and_keypoints:keypoints"]
    kp[150:, 15, 1] = H + 40          # left ankle leaves the bottom for the second half
    kp[:60, 10, 0] = -20              # right wrist leaves the frame for the first 2 s
    kp[:, 91:133, 2] = 0.7
    row = framing.clip_framing(track_for(payload), 0, 300)

    assert row["ankles_visible_frac"] == pytest.approx(0.5)
    assert row["knees_visible_frac"] == 1.0
    assert row["hips_visible_frac"] == 1.0 and row["shoulders_visible_frac"] == 1.0
    assert row["wrists_in_frame_frac"] == pytest.approx(240 / 300)
    assert row["hands_kp_conf_p50"] == pytest.approx(0.7, abs=1e-6)


def test_unknown_raster_leaves_visibility_na() -> None:
    track = track_for(make_payload(120), width=0, height=0)
    row = framing.clip_framing(track, 0, 120)

    for name in ("shoulders_visible_frac", "ankles_visible_frac", "wrists_in_frame_frac",
                 "box_height_frac", "box_edge_contact_frac", "reprojection_error_p50_sw"):
        assert math.isnan(row[name]), name
    assert math.isnan(framing.recording_framing(track)["recording_wrists_in_frame_frac"])


# ============================================================ box
def test_box_height_is_measured_in_the_upright_orientation() -> None:
    """A quarter-turned 3840x2160 file shows its person along the stored x axis."""

    box = (960.0, 700.0, 2880.0, 1500.0)    # 1920 px along x, 800 px along y
    payload = make_payload(90, width=3840, height=2160, box=box)
    rotated = framing.clip_framing(track_for(payload, width=3840, height=2160, quarter_turns=1), 0, 90)
    as_stored = framing.clip_framing(track_for(payload, width=3840, height=2160, quarter_turns=0), 0, 90)

    assert rotated["box_height_frac"] == pytest.approx(1920 / 3840)
    assert as_stored["box_height_frac"] == pytest.approx(800 / 2160)


def test_box_height_is_clipped_to_the_raster() -> None:
    payload = make_payload(90, box=(300.0, 960.0, 700.0, 2400.0))
    row = framing.clip_framing(track_for(payload), 0, 90)

    assert row["box_height_frac"] == pytest.approx(0.5)
    assert row["box_edge_contact_frac"] == 1.0


def test_edge_contact_is_within_one_percent_of_any_edge() -> None:
    payload = make_payload(100, box=(300.0, 400.0, 700.0, 1700.0))
    box = payload["boxes_and_keypoints:box"]
    box[:25, 3] = H - 5            # bottom, within 1%
    box[25:50, 0] = 0.005 * W      # left edge
    box[50:60, 1] = 0.02 * H       # 2% from the top: not touching
    payload["boxes_and_keypoints:is_valid_box"][90:] = False
    box[90:, 3] = H                # touching, but on invalid frames
    row = framing.clip_framing(track_for(payload), 0, 100)

    assert row["box_edge_contact_frac"] == pytest.approx(50 / 90)


def test_a_box_centre_jump_flags_the_tracker() -> None:
    payload = make_payload(300)
    box = payload["boxes_and_keypoints:box"]
    diagonal = float(np.hypot(400.0, 1300.0))
    box[150:, [0, 2]] += 0.8 * diagonal * 0.6     # centre moves 0.48 diagonals in x ...
    box[150:, [1, 3]] += 0.8 * diagonal * 0.8     # ... and 0.64 in y: 0.8 diagonals in all
    track = track_for(payload)

    row = framing.clip_framing(track, 0, 300)
    assert row["box_center_jump_max"] == pytest.approx(0.8, rel=1e-4)
    assert row["flag_tracker_jump"] is True
    # A clip that starts on the landing frame has no frame before the jump in it.
    after = framing.clip_framing(track, 150, 300)
    assert after["box_center_jump_max"] == pytest.approx(0.0, abs=1e-9)
    assert after["flag_tracker_jump"] is False


def test_a_jump_across_an_invalid_gap_still_counts_and_one_frame_is_na() -> None:
    payload = make_payload(200)
    payload["boxes_and_keypoints:is_valid_box"][100:110] = False
    payload["boxes_and_keypoints:box"][100:, [0, 2]] += 0.6 * float(np.hypot(400.0, 1300.0))
    track = track_for(payload)

    assert framing.clip_framing(track, 0, 200)["box_center_jump_max"] == pytest.approx(0.6, rel=1e-4)
    assert framing.clip_framing(track, 0, 200)["flag_tracker_jump"] is True

    payload["boxes_and_keypoints:is_valid_box"][:] = False
    payload["boxes_and_keypoints:is_valid_box"][5] = True
    lonely = framing.clip_framing(track_for(payload), 0, 200)
    assert math.isnan(lonely["box_center_jump_max"]) and lonely["flag_tracker_jump"] is False


# ============================================================ facing
def test_facing_angle_reads_yaw_away_from_the_camera() -> None:
    frontal = framing.clip_framing(track_for(make_payload(120)), 0, 120)
    assert frontal["facing_angle_deg_p50"] == pytest.approx(0.0, abs=1e-4)

    turned = framing.clip_framing(track_for(make_payload(120, yaw_deg=35.0)), 0, 120)
    assert turned["facing_angle_deg_p50"] == pytest.approx(35.0, abs=1e-3)
    assert turned["facing_angle_range_deg"] == pytest.approx(0.0, abs=1e-3)
    # Unsigned: turning the other way reads the same.
    other = framing.clip_framing(track_for(make_payload(120, yaw_deg=-35.0)), 0, 120)
    assert other["facing_angle_deg_p50"] == pytest.approx(35.0, abs=1e-3)

    sweep = np.linspace(0.0, 80.0, 300)
    turning = framing.clip_framing(track_for(make_payload(300, yaw_deg=sweep)), 0, 300)
    assert turning["facing_angle_deg_p50"] == pytest.approx(40.0, abs=0.2)
    assert turning["facing_angle_range_deg"] == pytest.approx(64.0, abs=0.5)


def test_facing_is_na_on_severe_anamorphic_and_short_clips() -> None:
    severe = framing.clip_framing(track_for(make_payload(120), anamorphic="severe"), 0, 120)
    assert math.isnan(severe["facing_angle_deg_p50"]) and math.isnan(severe["facing_angle_range_deg"])
    mild = framing.clip_framing(track_for(make_payload(120), anamorphic="mild"), 0, 120)
    assert mild["facing_angle_deg_p50"] == pytest.approx(0.0, abs=1e-4)


# ============================================================ reprojection
def test_reprojection_error_is_zero_for_a_consistent_fit_and_scales_in_shoulder_widths() -> None:
    payload = make_payload(90)
    track = track_for(payload)
    measured = np.isfinite(track.reprojection_error_sw)
    assert np.array_equal(np.flatnonzero(measured), np.arange(0, 90, 3)), "every 3rd frame"
    assert np.nanmax(track.reprojection_error_sw) < 1e-6

    kp = payload["boxes_and_keypoints:keypoints"]
    width_px = float(np.linalg.norm(kp[0, 5, :2] - kp[0, 6, :2]))
    kp[:, 7:13, 0] += 0.1 * width_px            # elbows, wrists, hips off by 0.1 SW
    row = framing.clip_framing(track_for(payload), 0, 90)
    assert row["reprojection_error_p50_sw"] == pytest.approx(0.1 * 6 / 8, rel=1e-4)
    assert framing.recording_framing(track_for(payload))["recording_reprojection_error_p50_sw"] == pytest.approx(0.075, rel=1e-4)


def test_reprojection_skips_sentinels_invalid_boxes_and_unseen_joints() -> None:
    payload = make_payload(90)
    payload["smplh:translation"][0:30] = 1e12
    payload["boxes_and_keypoints:is_valid_box"][30:60] = False
    track = track_for(payload)
    assert not np.isfinite(track.reprojection_error_sw[:60]).any()
    assert np.isfinite(track.reprojection_error_sw[60::3]).all()

    # A joint outside the raster is not compared, even if the fit put it far away.
    payload = make_payload(90)
    payload["boxes_and_keypoints:keypoints"][:, 11, :2] = (-500.0, 900.0)
    assert np.nanmax(track_for(payload).reprojection_error_sw) < 1e-6

    # No visible shoulders: no scale, no measurement.
    payload = make_payload(90)
    payload["boxes_and_keypoints:keypoints"][:, 6, 2] = 0.2
    assert math.isnan(framing.clip_framing(track_for(payload), 0, 90)["reprojection_error_p50_sw"])
    assert math.isnan(framing.clip_framing(track_for(make_payload(90), joints=False), 0, 90)["reprojection_error_p50_sw"])


def test_rest_pelvis_constant_is_the_model_rest_pelvis(model_root) -> None:
    from seamless_curation.smplh_kinematics import rest_joints

    rest, _parents = rest_joints(str(model_root))
    assert np.allclose(rest[0], framing.REST_PELVIS_M, atol=1e-12)


# ============================================================ rotation, NA, registry
def _lying(frames: int, feet_left: bool = True) -> np.ndarray:
    points = np.zeros((frames, 133, 3), dtype=np.float32)
    points[:, :, 2] = 0.9
    sign = -1.0 if feet_left else 1.0
    points[:, 5, :2] = (2000.0, 1000.0)
    points[:, 6, :2] = (2000.0, 1200.0)
    points[:, 11, :2] = (2000.0 + sign * 500, 1020.0)
    points[:, 12, :2] = (2000.0 + sign * 500, 1180.0)
    return points


def test_quarter_turns_come_from_the_usable_shoulder_hip_axis() -> None:
    upright = np.zeros((10, 133, 3), dtype=np.float32)
    upright[:, :, 2] = 0.9
    upright[:, 5, :2], upright[:, 6, :2] = (600, 700), (450, 700)
    upright[:, 11, :2], upright[:, 12, :2] = (580, 1100), (470, 1100)
    assert framing.detect_quarter_turns(upright) == 0
    assert framing.detect_quarter_turns(_lying(10, feet_left=True)) == 1
    assert framing.detect_quarter_turns(_lying(10, feet_left=False)) == 3

    # Mostly zero-filled (invalid-box) frames must not vote for "upright".
    mostly_empty = _lying(10)
    mostly_empty[:7] = 0.0
    assert framing.detect_quarter_turns(mostly_empty) == 1
    assert framing.detect_quarter_turns(np.zeros((0, 133, 3))) == 0


def test_a_short_clip_is_na_where_the_registry_says_so() -> None:
    row = framing.clip_framing(track_for(make_payload(300)), 250, 300)

    for name in ("shoulders_visible_frac", "hips_visible_frac", "knees_visible_frac",
                 "ankles_visible_frac", "wrists_in_frame_frac", "hands_kp_conf_p50",
                 "facing_angle_deg_p50", "facing_angle_range_deg"):
        assert math.isnan(row[name]), name
    # Box and reprojection measures are defined by their own inputs, not by length.
    assert np.isfinite(row["box_height_frac"]) and np.isfinite(row["reprojection_error_p50_sw"])
    assert row["flag_tracker_jump"] is False


def test_keys_and_types_follow_the_registry() -> None:
    track = track_for(make_payload(300))
    row = framing.clip_framing(track, 0, 300)
    assert set(row) == set(framing.CLIP_COLUMNS) <= set(schema.columns("clips"))
    for name, value in row.items():
        if schema.field("clips", name).dtype == "bool":
            assert isinstance(value, bool), name
        else:
            assert isinstance(value, float), name
    rec = framing.recording_framing(track)
    assert set(rec) <= set(schema.columns("recordings"))
    assert rec["recording_wrists_in_frame_frac"] == 1.0


# ============================================================ real files
RELEASE = Path(__file__).resolve().parents[1] / "seamless_interaction"
REAL = (
    # relbase, width, height, expected quarter turns, smplh_anamorphic
    ("improvised/train/0028/0030/V00_S1245_I00001163_P0955A", 1080, 1920, 0, "none"),
    ("naturalistic/train/0131/0029/V03_S0965_I00000162_P3339", 3840, 2160, 1, "none"),
    ("improvised/dev/0000/0001/V01_S0346_I00000721_P1694", 2160, 2160, 0, "severe"),
)


@pytest.mark.parametrize("relbase,width,height,turns,anamorphic", REAL)
def test_real_file_smoke(model_root, relbase, width, height, turns, anamorphic) -> None:
    npz = RELEASE / f"{relbase}.npz"
    if not npz.exists():
        pytest.skip("release not mounted")
    from seamless_curation.gesture import build_tracks

    with np.load(npz) as data:
        payload = {key: data[key] for key in data.files if not key.startswith("movement:")}
    frames = len(payload["smplh:body_pose"])
    tracks = build_tracks(payload, np.zeros(frames, dtype=bool), fps=30.0, model_root=str(model_root))
    detected = framing.detect_quarter_turns(payload["boxes_and_keypoints:keypoints"])
    track = framing.framing_track(payload, tracks.joints, fps=30.0, width=width, height=height,
                                  quarter_turns=detected, smplh_anamorphic=anamorphic)
    rows = [framing.clip_framing(track, b.start_frame, b.end_frame) for b in clip_grid(frames, 30.0)]
    rec = framing.recording_framing(track)

    assert detected == turns
    assert 0.03 < rec["recording_reprojection_error_p50_sw"] < 0.25
    assert rec["recording_wrists_in_frame_frac"] > 0.9
    assert not any(row["flag_tracker_jump"] for row in rows)
    full = [row for row in rows[:-1]]
    if anamorphic == "severe":
        assert all(math.isnan(row["facing_angle_deg_p50"]) for row in full)
    else:
        # Participants face the camera: the forward axis is within ~40 deg of it.
        assert all(0.0 <= row["facing_angle_deg_p50"] < 45.0 for row in full)
    assert all(0.3 < row["box_height_frac"] < 1.0 for row in full)

"""Face annotations on synthetic ``movement:*`` payloads in the released shapes.

The builder reproduces what the V00 NPZs hold: ``is_valid`` float32 ``(T, 1)``,
invalidity in whole 60-frame blocks starting at multiples of 60, every array
zero-filled on invalid frames, ``FAUValue`` ``(T, 24)`` with AUs 16-18 dead, and
arousal/valence ``(T, 1)``. Each test states what a measure must say about a
face whose expression we set.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from seamless_curation import schema
from seamless_curation.face import (
    CLIP_MEASURES,
    DEAD_AUS,
    LIVE_AUS,
    FaceTrack,
    clip_face,
    face_track,
    recording_face,
)

FPS = 30.0


def movement_payload(
    n_frames: int = 900,
    invalid_blocks: tuple[int, ...] = (),
    *,
    fau_live: np.ndarray | None = None,
    dead_values: tuple[float, float, float] = (0.01, 0.002, 6e-14),
) -> dict[str, np.ndarray]:
    t = np.arange(n_frames)
    valid = np.ones(n_frames, dtype=bool)
    for block in invalid_blocks:
        valid[block * 60 : (block + 1) * 60] = False
    fau = np.zeros((n_frames, 24), dtype=np.float32)
    if fau_live is None:
        fau_live = np.stack([0.5 + 0.3 * np.sin(2 * np.pi * t / (20 + 3 * j)) for j in range(21)], axis=1)
    fau[:, list(LIVE_AUS)] = fau_live
    fau[:, list(DEAD_AUS)] = dead_values
    arousal = (0.3 + 0.2 * np.sin(2 * np.pi * t / 90)).astype(np.float32)
    valence = (0.1 * np.cos(2 * np.pi * t / 150)).astype(np.float32)
    fau[~valid] = 0
    arousal[~valid] = 0
    valence[~valid] = 0
    return {
        "movement:is_valid": valid.astype(np.float32)[:, None],
        "movement:FAUValue": fau,
        "movement:emotion_arousal": arousal[:, None],
        "movement:emotion_valence": valence[:, None],
        "movement:emotion_scores": np.zeros((n_frames, 8), dtype=np.float32),
        "smplh:body_pose": np.zeros((n_frames, 21, 3), dtype=np.float32),
    }


# ----------------------------------------------------------------------------- the track
def test_no_movement_keys_means_no_track():
    payload = movement_payload()
    non_v00 = {"smplh:body_pose": payload["smplh:body_pose"]}
    assert face_track(non_v00, 900) is None
    partial = {k: v for k, v in payload.items() if k != "movement:emotion_valence"}
    assert face_track(partial, 900) is None


def test_track_shapes_and_dead_aus_removed():
    payload = movement_payload(invalid_blocks=(2,))
    track = face_track(payload, 900)
    assert isinstance(track, FaceTrack)
    assert track.valid.shape == (900,) and track.valid.dtype == bool
    assert track.fau.shape == (900, 21) and track.fau.dtype == np.float32
    assert track.arousal.shape == (900,) and track.valence.shape == (900,)
    assert track.valid.sum() == 840 and not track.valid[120:180].any()
    fau = payload["movement:FAUValue"]
    assert np.array_equal(track.fau[:, 15], fau[:, 15]) and np.array_equal(track.fau[:, 16], fau[:, 19])


def test_npz_round_trip(tmp_path):
    path = tmp_path / "V00_x.npz"
    np.savez(path, **movement_payload(invalid_blocks=(0,)))
    with np.load(path) as archive:
        track = face_track(archive, 900)
    assert track is not None and track.valid.sum() == 840


def test_arrays_are_cut_or_padded_to_the_pose_grid():
    payload = movement_payload(n_frames=600)
    longer = face_track(payload, 660)
    assert longer.valid.shape == (660,) and not longer.valid[600:].any()
    shorter = face_track(payload, 300)
    assert shorter.fau.shape == (300, 21) and shorter.valid.all()


def test_non_finite_emotion_invalidates_the_frame():
    payload = movement_payload()
    payload["movement:emotion_arousal"][10, 0] = np.nan
    track = face_track(payload, 900)
    assert not track.valid[10] and track.valid.sum() == 899


# ----------------------------------------------------------------------------- clip measures
def test_invalid_block_is_excluded_not_averaged_as_zero():
    payload = movement_payload(invalid_blocks=(2,))
    track = face_track(payload, 900)
    clip = clip_face(track, 0, 300, FPS)
    valid = payload["movement:is_valid"][:300, 0] > 0.5
    live = payload["movement:FAUValue"][:300][valid][:, list(LIVE_AUS)].astype(np.float64)
    arousal = payload["movement:emotion_arousal"][:300, 0][valid].astype(np.float64)
    assert clip["face_valid_frac"] == pytest.approx(240 / 300)
    assert clip["fau_intensity_mean"] == pytest.approx(live.mean(), rel=1e-6)
    assert clip["fau_variability"] == pytest.approx(live.std(axis=0).mean(), rel=1e-6)
    assert clip["emotion_arousal_mean"] == pytest.approx(arousal.mean(), rel=1e-6)
    assert clip["emotion_arousal_sd"] == pytest.approx(arousal.std(), rel=1e-6)
    assert clip["emotion_valence_mean"] == pytest.approx(
        payload["movement:emotion_valence"][:300, 0][valid].mean(), abs=1e-7)
    # the zero-fill would have pulled the mean intensity down by a fifth
    assert clip["fau_intensity_mean"] > payload["movement:FAUValue"][:300][:, list(LIVE_AUS)].mean() * 1.2


def test_measures_need_two_seconds_of_valid_frames():
    # blocks 0-3 invalid: frames 240-299 are the only valid ones in [0, 300)
    track = face_track(movement_payload(invalid_blocks=(0, 1, 2, 3)), 900)
    enough = clip_face(track, 0, 300, FPS)
    assert enough["face_valid_frac"] == pytest.approx(0.2)
    assert all(math.isfinite(enough[k]) for k in CLIP_MEASURES)
    short = clip_face(track, 0, 299, FPS)  # 59 valid frames
    assert short["face_valid_frac"] == pytest.approx(59 / 299)
    assert all(math.isnan(short[k]) for k in CLIP_MEASURES)
    # 29.97 fps: 2 s is still 60 frames (ceil(59.94))
    assert all(math.isnan(clip_face(track, 0, 299, 29.97)[k]) for k in CLIP_MEASURES)
    assert all(math.isfinite(clip_face(track, 0, 300, 29.97)[k]) for k in CLIP_MEASURES)


def test_all_invalid_clip_has_zero_valid_frac_and_na_measures():
    track = face_track(movement_payload(invalid_blocks=tuple(range(15))), 900)
    clip = clip_face(track, 0, 900, FPS)
    assert clip["face_valid_frac"] == 0.0
    assert all(math.isnan(clip[k]) for k in CLIP_MEASURES)


def test_dead_aus_do_not_enter_any_statistic():
    quiet = clip_face(face_track(movement_payload(), 900), 0, 900, FPS)
    loud = clip_face(face_track(movement_payload(dead_values=(2.0, 1.5, 1.0)), 900), 0, 900, FPS)
    for key in CLIP_MEASURES:
        assert loud[key] == pytest.approx(quiet[key], rel=1e-9)


def test_fau_variability_is_the_mean_within_clip_sd():
    t = np.arange(900)
    still = movement_payload(fau_live=np.full((900, 21), 0.4))
    assert clip_face(face_track(still, 900), 0, 900, FPS)["fau_variability"] == pytest.approx(0.0, abs=1e-7)
    alternating = np.where((t % 2 == 0)[:, None], 0.4, 0.6) * np.ones((1, 21))
    moving = clip_face(face_track(movement_payload(fau_live=alternating), 900), 0, 900, FPS)
    assert moving["fau_variability"] == pytest.approx(0.1, abs=1e-6)
    assert moving["fau_intensity_mean"] == pytest.approx(0.5, abs=1e-6)


def test_zero_fau_fill_on_a_valid_frame_is_not_a_measurement():
    # One V00 file shifts the FAU zero-fill two frames against is_valid at block edges.
    payload = movement_payload(fau_live=np.full((900, 21), 0.4))
    payload["movement:FAUValue"][100:102] = 0.0  # valid frames, zero FAU row
    track = face_track(payload, 900)
    clip = clip_face(track, 0, 300, FPS)
    assert clip["face_valid_frac"] == 1.0
    assert clip["fau_intensity_mean"] == pytest.approx(0.4, abs=1e-7)
    assert clip["fau_variability"] == pytest.approx(0.0, abs=1e-7)


def test_no_track_gives_na_clip():
    clip = clip_face(None, 0, 900, FPS)
    assert set(clip) == {"face_valid_frac", *CLIP_MEASURES}
    assert all(math.isnan(v) for v in clip.values())


def test_clip_range_is_clamped_to_the_track():
    track = face_track(movement_payload(n_frames=600), 600)
    tail = clip_face(track, 540, 900, FPS)
    assert tail["face_valid_frac"] == 1.0 and math.isfinite(tail["fau_intensity_mean"])
    assert math.isnan(clip_face(track, 600, 900, FPS)["face_valid_frac"])


# ----------------------------------------------------------------------------- recording
def test_recording_status_and_valid_frac():
    missing = recording_face(None)
    assert missing["face_features_status"] == "not_provided" and math.isnan(missing["recording_face_valid_frac"])
    available = recording_face(face_track(movement_payload(invalid_blocks=(1, 5)), 900))
    assert available["face_features_status"] == "available"
    assert available["recording_face_valid_frac"] == pytest.approx(780 / 900)
    dead = recording_face(face_track(movement_payload(invalid_blocks=tuple(range(15))), 900))
    assert dead == {"face_features_status": "all_invalid", "recording_face_valid_frac": 0.0}
    empty = recording_face(face_track(movement_payload(n_frames=0), 0))
    assert empty["face_features_status"] == "not_measured" and math.isnan(empty["recording_face_valid_frac"])


def test_output_keys_are_registered_at_their_level():
    track = face_track(movement_payload(), 900)
    for key in clip_face(track, 0, 900, FPS):
        assert schema.field("clips", key).level == "clip"
    rec = recording_face(track)
    for key in rec:
        assert schema.field("recordings", key).level == "recording"
    assert rec["face_features_status"] in schema.field("recordings", "face_features_status").allowed

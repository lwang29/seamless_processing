"""The handover: a manifest row must load, and the tiers must mean what they say.

These are the two pieces a downstream project touches, so they are the two
pieces most worth pinning. The loader tests assert the *contract* the dataset
card states — shapes, audio length, speech rebased to the clip — rather than
re-deriving it, because a downstream trainer will assume exactly that contract.
"""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from seamless_curation.dataset import (
    ID_COLUMNS,
    UPPER_BODY_ROWS,
    clip_paths,
    iter_clips,
    load_clip,
    load_manifest,
)
from seamless_curation.export import wilson_interval

from tests.conftest import FPS, make_bundle

RELBASE = "improvised/train/0/0/V00_S1_I1_P1"


@pytest.fixture
def source_tree(tmp_path: Path) -> Path:
    """A one-recording release: npz + json + wav at a shared stem."""

    soundfile = pytest.importorskip("soundfile")
    bundle = make_bundle(shoulder_swing_deg=55.0, frames=900)
    directory = tmp_path / "source" / Path(RELBASE).parent
    base = bundle.write(directory, name="V00_S1_I1_P1")
    # 30 s of audio at 48 kHz, matching the 900 frames at 30 fps.
    samples = int(30.0 * 48000)
    tone = (0.1 * np.sin(np.arange(samples) * 2 * np.pi * 220 / 48000)).astype(np.float32)
    soundfile.write(str(base.with_suffix(".wav")), tone, 48000, subtype="FLOAT")
    return tmp_path / "source"


@pytest.fixture
def short_audio_tree(tmp_path: Path) -> Path:
    """Same recording, but the WAV stops 0.4 s before the pose does."""

    soundfile = pytest.importorskip("soundfile")
    bundle = make_bundle(shoulder_swing_deg=55.0, frames=900)
    directory = tmp_path / "source" / Path(RELBASE).parent
    base = bundle.write(directory, name="V00_S1_I1_P1")
    samples = int(29.6 * 48000)
    tone = (0.1 * np.sin(np.arange(samples) * 2 * np.pi * 220 / 48000)).astype(np.float32)
    soundfile.write(str(base.with_suffix(".wav")), tone, 48000, subtype="FLOAT")
    return tmp_path / "source"


def row(**overrides) -> dict:
    record = {
        "clip_id": "V00_S1_I1_P1_f000000",
        "file_id": "V00_S1_I1_P1",
        "source_relbase": RELBASE,
        "vendor": "V00",
        "split": "train",
        "participant_id": "0001",
        "start_frame": 0,
        "end_frame": 450,
        "fps": FPS,
        "window_seconds": 15.0,
    }
    record.update(overrides)
    return record


def test_clip_paths_share_one_stem() -> None:
    paths = clip_paths(row(), "/data")
    assert {p.stem for p in paths.values()} == {"V00_S1_I1_P1"}
    assert paths["pose"].suffix == ".npz" and paths["audio"].suffix == ".wav"
    assert paths["video"].suffix == ".mp4" and paths["annotation"].suffix == ".json"


def test_load_clip_returns_the_frames_the_row_names(source_tree: Path) -> None:
    clip = load_clip(row(), source_tree)

    assert clip.frames == 450
    assert clip.body_pose.shape == (450, 21, 3)
    assert clip.left_hand_pose.shape == (450, 15, 3)
    assert clip.right_hand_pose.shape == (450, 15, 3)
    assert clip.global_orient.shape == (450, 3)
    assert clip.smplh_valid.shape == (450,)
    # The contract the dataset card states for the ViBES body input.
    assert clip.upper_body_pose.shape == (450, 13, 3)


def test_upper_body_rows_are_the_upper_body() -> None:
    """A wrong offset here would train ViBES on legs and nobody would notice."""

    from seamless_curation import smplh_kinematics as k

    # body_pose row r is tree joint r + 1: the release carries the pelvis
    # separately in global_orient.
    joints = {r + 1 for r in UPPER_BODY_ROWS}
    assert joints == {
        k.SPINE1, k.SPINE2, k.SPINE3, k.NECK, k.HEAD, k.L_COLLAR, k.R_COLLAR,
        k.L_SHOULDER, k.R_SHOULDER, k.L_ELBOW, k.R_ELBOW, k.L_WRIST, k.R_WRIST,
    }
    assert len(UPPER_BODY_ROWS) == 13
    # Hips 1-2, knees 4-5, ankles 7-8, feet 10-11 in the SMPL-H tree.
    assert joints.isdisjoint({1, 2, 4, 5, 7, 8, 10, 11})


def test_audio_is_the_matching_slice(source_tree: Path) -> None:
    clip = load_clip(row(start_frame=300, end_frame=750), source_tree)

    assert clip.sample_rate == 48000
    assert len(clip.audio) == int(round(clip.seconds * 48000))
    assert clip.audio.dtype == np.float32
    assert np.abs(clip.audio).max() > 0.05      # it is the tone, not silence


def test_audio_past_the_end_is_padded_not_an_error(short_audio_tree: Path) -> None:
    """Audio can run out before pose does; the clip must still be its full length.

    Real recordings do this: the WAV and the pose track are written by different
    stages and the last fraction of a second does not always survive both. A
    trainer that batches on length would break on a short array.
    """

    clip = load_clip(row(start_frame=0, end_frame=900), short_audio_tree)

    assert clip.frames == 900
    assert len(clip.audio) == int(round(clip.seconds * 48000))   # padded, not short
    assert np.abs(clip.audio[-4000:]).max() == 0.0               # the pad is silence


def test_pose_can_be_loaded_without_audio(source_tree: Path) -> None:
    clip = load_clip(row(), source_tree, with_audio=False)

    assert clip.audio is None and clip.sample_rate is None
    assert clip.body_pose.shape == (450, 21, 3)


def test_speech_is_rebased_to_the_clip(source_tree: Path) -> None:
    """VAD in the release is recording-relative; a clip's consumer wants its own."""

    clip = load_clip(row(start_frame=270, end_frame=720), source_tree)  # 9 s .. 24 s

    assert clip.speech, "the fixture speaks during this window"
    assert all(0.0 <= begin < end <= clip.seconds + 1e-6 for begin, end in clip.speech)
    # Fixture VAD is ((1,6),(9,15),(18,25)); clipped to 9-24 s and rebased that
    # is (0,6) and (9,15).
    flat = [value for segment in clip.speech for value in segment]
    assert flat == pytest.approx([0.0, 6.0, 9.0, 15.0])


def test_a_manifest_that_lies_about_frame_count_raises(source_tree: Path) -> None:
    with pytest.raises(ValueError, match="does not match the manifest"):
        load_clip(row(start_frame=0, end_frame=5000), source_tree, with_audio=False)


def test_iter_clips_can_skip_unreadable_rows(source_tree: Path) -> None:
    frame = pd.DataFrame([row(), row(source_relbase="nope/missing", clip_id="gone")])

    with pytest.warns(UserWarning):
        loaded = list(iter_clips(frame, source_tree, with_audio=False, skip_errors=True))
    assert [c.clip_id for c in loaded] == ["V00_S1_I1_P1_f000000"]

    with pytest.raises(Exception):
        list(iter_clips(frame, source_tree, with_audio=False, skip_errors=False))


def test_load_manifest_keeps_zero_padded_ids_as_text(tmp_path: Path) -> None:
    """``0045`` read as an integer becomes ``45`` and stops joining to anything."""

    path = tmp_path / "m.csv"
    pd.DataFrame([row(participant_id="0045"), row(participant_id="0046")]).to_csv(path, index=False)

    frame = load_manifest(path)
    assert frame["participant_id"].tolist() == ["0045", "0046"]
    assert frame["start_frame"].dtype.kind == "i"
    assert set(ID_COLUMNS) >= {"participant_id", "file_id", "session_id"}


@pytest.mark.parametrize(
    "successes,total,contains",
    [(80, 100, 0.80), (67, 67, 1.0), (0, 10, 0.0)],
)
def test_wilson_interval_brackets_the_point_estimate(successes, total, contains) -> None:
    low, high = wilson_interval(successes, total)
    assert low <= contains + 1e-9 and contains - 1e-9 <= high
    assert 0.0 <= low <= high <= 1.0


def test_wilson_is_not_degenerate_at_a_perfect_score() -> None:
    """67/67 must not claim certainty; the rule of three says about 4.5%."""

    low, high = wilson_interval(67, 67)
    assert high == pytest.approx(1.0)
    assert 0.93 < low < 0.96

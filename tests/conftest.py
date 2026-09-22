"""Synthetic bundles, so the suite runs without the dataset or the body model.

The v0 fixtures read three specific dev recordings off the read-only mount and
asserted that a manifest index still resolved to the same file id. That coupled
every test to a 40 TB NFS export and to a CSV that no longer exists. The
measures under test are geometric, so a bundle can be *constructed* with a known
answer instead — a wrist that traces a known arc while speech is on, or one that
only shakes in place — and the test then states what the measure must say about
it rather than what it happened to say about file 0025.

The one thing that cannot be synthesised is the research-licensed SMPL-H model.
:func:`model_root` skips the tests that need it rather than failing them.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest

FPS = 30.0
FRAMES = 900  # 30 s
JOINTS = 133


@pytest.fixture(scope="session")
def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


@pytest.fixture(scope="session")
def model_root(repo_root: Path) -> Path:
    """The neutral SMPL-H asset, or a skip if it is not staged here."""

    for candidate in (repo_root / "model_files", repo_root / "model_files" / "smplh"):
        if (candidate / "SMPLH_NEUTRAL.npz").exists():
            return repo_root / "model_files"
    pytest.skip("SMPL-H neutral model not available in this environment")


@dataclass
class SyntheticBundle:
    """Released-format arrays with a known ground truth about their motion."""

    payload: dict[str, np.ndarray]
    vad: list[dict[str, float]]
    fps: float = FPS

    def write(self, directory: Path, name: str = "V00_S0001_I00000001_P0001") -> Path:
        directory.mkdir(parents=True, exist_ok=True)
        base = directory / name
        np.savez(base.with_suffix(".npz"), **self.payload)
        base.with_suffix(".json").write_text(
            json.dumps({"id": name, "metadata:transcript": [], "metadata:vad": self.vad}),
            encoding="utf-8",
        )
        return base


def _keypoints(
    frames: int, wrist_xy: np.ndarray, right_wrist_xy: np.ndarray | None = None
) -> np.ndarray:
    """A COCO-WholeBody block with plausible shoulders and the given wrists.

    ``right_wrist_xy`` defaults to the left wrist shifted across the body, which
    is the two-handed case. Passing a fixed track models one-handed gesturing.
    """

    points = np.zeros((frames, JOINTS, 3), dtype=np.float32)
    points[:, :, 2] = 0.9
    points[:, 5, :2] = (620.0, 700.0)   # left shoulder
    points[:, 6, :2] = (460.0, 700.0)   # right shoulder
    points[:, 7, :2] = (660.0, 850.0)
    points[:, 8, :2] = (420.0, 850.0)
    points[:, 9, :2] = wrist_xy
    points[:, 10, :2] = (
        wrist_xy + np.array([-160.0, 0.0]) if right_wrist_xy is None else right_wrist_xy
    )
    for start in (91, 112):
        points[:, start : start + 21, :2] = points[:, 9 if start == 91 else 10, None, :2]
    return points


def _global_orient(frames: int, yaw_deg: float, phase: np.ndarray) -> np.ndarray:
    """Whole-body yaw as an axis-angle about the vertical, swept by ``phase``."""

    orient = np.zeros((frames, 3), dtype=np.float32)
    if yaw_deg:
        orient[:, 1] = np.deg2rad(yaw_deg) * phase
    return orient


def make_bundle(
    *,
    frames: int = FRAMES,
    shoulder_swing_deg: float = 0.0,
    period_s: float = 2.0,
    jitter_mm: float = 0.0,
    single_adjustment: bool = False,
    one_handed: bool = False,
    global_yaw_deg: float = 0.0,
    episodic: bool = False,
    gesture_in_silence: bool = False,
    global_drift_mm: float = 0.0,
    speech: tuple[tuple[float, float], ...] = ((1.0, 6.0), (9.0, 15.0), (18.0, 25.0)),
    smplh_valid: np.ndarray | None = None,
    hand_freeze_from: int | None = None,
    seed: int = 0,
) -> SyntheticBundle:
    """Build a bundle whose motion is exactly what the arguments say.

    ``shoulder_swing_deg`` sweeps both shoulder joints, which moves the wrists in
    the torso frame and is therefore *gesture*. ``global_drift_mm`` translates the
    whole body via the root, which must **not** register as gesture.
    ``global_drift_mm`` writes ``smplh:translation``, which **nothing in the
    pipeline reads** -- forward kinematics places the pelvis at the origin, so
    pure translation is invisible by construction rather than by a threshold.
    ``global_yaw_deg`` is the parameter that actually exercises the torso frame:
    it rotates the whole body through ``smplh:global_orient`` while the arms
    stay rigid relative to the torso, which is what a participant turning to
    face their partner looks like in the data.

    ``jitter_mm`` adds independent per-frame noise. ``single_adjustment`` replaces
    the sweep with one brief movement at the start. ``one_handed`` swings only the
    left shoulder and parks the right arm in the lap, which is the posture the
    rubric decided to accept on 2026-09-21.

    ``episodic`` gates the sweep to the speech segments with a raised-cosine
    envelope, so the bundle produces one gesture episode per utterance separated
    by rest. That is what real co-speech gesture looks like and what the tier-2
    episode clauses are written against; a single uninterrupted 30-second sweep
    is one episode and is correctly rejected by them. ``gesture_in_silence``
    inverts the envelope, giving motion that is real but anti-correlated with
    speech -- the ``not_co_speech`` failure mode.
    """

    rng = np.random.default_rng(seed)
    time = np.arange(frames) / FPS
    phase = np.sin(2 * np.pi * time / period_s)
    if single_adjustment:
        phase = np.zeros(frames)
        burst = slice(0, int(0.8 * FPS))
        phase[burst] = np.sin(np.linspace(0, np.pi, burst.stop))
    elif episodic or gesture_in_silence:
        # One raised-cosine envelope per speech segment: ramp in over 0.4 s,
        # hold, ramp out. Multiplying the carrier by this gives a distinct
        # gesture episode per utterance instead of one continuous sweep.
        envelope = np.zeros(frames)
        ramp_frames = max(1, int(0.4 * FPS))
        for begin, end in speech:
            lo, hi = int(begin * FPS), min(frames, int(end * FPS))
            if hi - lo < 2 * ramp_frames:
                continue
            envelope[lo:hi] = 1.0
            envelope[lo : lo + ramp_frames] = 0.5 * (
                1 - np.cos(np.linspace(0, np.pi, ramp_frames))
            )
            envelope[hi - ramp_frames : hi] = 0.5 * (
                1 + np.cos(np.linspace(0, np.pi, ramp_frames))
            )
        if gesture_in_silence:
            envelope = 1.0 - envelope
        phase = phase * envelope

    body = np.zeros((frames, 21, 3), dtype=np.float32)
    swing = np.deg2rad(shoulder_swing_deg) * phase
    body[:, 15, 2] = swing      # left shoulder, joint 16 in the full tree
    body[:, 16, 2] = 0.0 if one_handed else -swing   # right shoulder
    if jitter_mm:
        # Rotational noise of the size that moves a wrist by ~jitter_mm.
        noise = rng.normal(0.0, jitter_mm / 500.0, size=(frames, 2))
        body[:, 15, 2] += noise[:, 0]
        body[:, 16, 2] += noise[:, 1]

    hands = np.zeros((frames, 15, 3), dtype=np.float32)
    hands[:, :, 0] = 0.05 * phase[:, None]
    left = hands.copy()
    right = np.zeros_like(hands) if one_handed else hands.copy()
    if hand_freeze_from is not None:
        left[hand_freeze_from:] = left[hand_freeze_from]
        right[hand_freeze_from:] = right[hand_freeze_from]
    else:
        # A real fit never produces bit-identical hand parameters on consecutive
        # frames, which is exactly why hand_frozen_frac is a tracking-failure
        # detector. Synthetic poses that rest at exactly zero would trip it
        # spuriously, so add a tremor far below any measurable motion: 1e-4 rad
        # moves a fingertip by about a hundredth of a millimetre.
        left = left + rng.normal(0.0, 1e-4, size=left.shape).astype(np.float32)
        right = right + rng.normal(0.0, 1e-4, size=right.shape).astype(np.float32)

    translation = np.zeros((frames, 3), dtype=np.float32)
    translation[:, 0] = global_drift_mm / 1000.0 * phase

    wrist_xy = np.stack(
        [
            700.0 + 260.0 * shoulder_swing_deg / 60.0 * phase + rng.normal(0, jitter_mm / 8.0, frames),
            760.0 - 120.0 * shoulder_swing_deg / 60.0 * np.abs(phase),
        ],
        axis=1,
    ).astype(np.float32)

    # A parked right wrist, low and near the midline, i.e. resting in the lap.
    right_wrist_xy = (
        np.tile(np.array([[500.0, 980.0]], dtype=np.float32), (frames, 1))
        if one_handed
        else None
    )

    payload = {
        "smplh:body_pose": body,
        "smplh:left_hand_pose": left,
        "smplh:right_hand_pose": right,
        "smplh:global_orient": _global_orient(frames, global_yaw_deg, phase),
        "smplh:translation": translation,
        "smplh:is_valid": (
            np.ones(frames, dtype=bool) if smplh_valid is None else smplh_valid.astype(bool)
        ),
        "boxes_and_keypoints:keypoints": _keypoints(frames, wrist_xy, right_wrist_xy),
        "boxes_and_keypoints:is_valid_box": np.ones(frames, dtype=bool),
    }
    return SyntheticBundle(payload=payload, vad=[{"start": a, "end": b} for a, b in speech])


@pytest.fixture
def gesturing_bundle() -> SyntheticBundle:
    return make_bundle(shoulder_swing_deg=55.0)


@pytest.fixture
def static_bundle() -> SyntheticBundle:
    return make_bundle(shoulder_swing_deg=0.0, jitter_mm=6.0)


@pytest.fixture
def cospeech_bundle() -> SyntheticBundle:
    """The positive case: episodic two-handed gesturing locked to the speech."""

    return make_bundle(shoulder_swing_deg=55.0, episodic=True)


@pytest.fixture
def one_handed_bundle() -> SyntheticBundle:
    """Left arm gesturing, right arm resting in the lap."""

    return make_bundle(shoulder_swing_deg=55.0, one_handed=True)

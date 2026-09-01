from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest
import yaml

from seamless_curation.features import as_binary_mask


@dataclass
class DevSpan:
    file_id: str
    translation: np.ndarray
    keypoints: np.ndarray
    smplh_mask: np.ndarray
    movement_mask: np.ndarray
    box_mask: np.ndarray
    fps: float
    jitter_amplitude: float
    blank_hand_frames: int
    fixture_status: str


@pytest.fixture(scope="session")
def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


@pytest.fixture(scope="session", params=(0, 1, 2), ids=("clean_01", "clean_02", "clean_03"))
def dev_span(repo_root: Path, request: pytest.FixtureRequest) -> DevSpan:
    spec = yaml.safe_load((repo_root / "tests/fixtures/dev_span.yaml").read_text())
    span_spec = spec["spans"][request.param]
    assert span_spec["visually_confirmed_clean"] is True
    harness_config = yaml.safe_load((repo_root / "configs/harness.yaml").read_text())
    with (repo_root / harness_config["manifest"]).open(newline="") as handle:
        manifest = list(csv.DictReader(handle))
    row = manifest[int(span_spec["manifest_index"])]
    assert row["file_id"] == span_spec["file_id"], (
        "visually reviewed fixture manifest index now resolves to a different file"
    )
    base = repo_root / harness_config["source_root"] / row["source_relbase"]
    start = int(span_spec["start_frame"])
    end = start + int(span_spec["n_frames"])
    with np.load(f"{base}.npz", allow_pickle=False) as archive:
        translation = np.asarray(archive["smplh:translation"])[start:end].copy()
        keypoints = np.asarray(archive["boxes_and_keypoints:keypoints"])[start:end].copy()
        smplh_mask, smplh_status = as_binary_mask(archive["smplh:is_valid"][start:end])
        movement_mask, movement_status = as_binary_mask(archive["movement:is_valid"][start:end])
        box_mask, box_status = as_binary_mask(
            archive["boxes_and_keypoints:is_valid_box"][start:end]
        )
    expected = int(span_spec["n_frames"])
    assert len(translation) == len(keypoints) == expected
    assert smplh_mask is not None and movement_mask is not None and box_mask is not None
    assert smplh_status == movement_status == box_status == "ok"
    # Each fixture is both flag-clean and visually reviewed in the private overlay set.
    assert smplh_mask.all() and movement_mask.all() and box_mask.all()
    return DevSpan(
        file_id=row["file_id"],
        translation=translation,
        keypoints=keypoints,
        smplh_mask=smplh_mask,
        movement_mask=movement_mask,
        box_mask=box_mask,
        fps=float(harness_config["window"]["fps"]),
        jitter_amplitude=float(spec["jitter_amplitude_model_units"]),
        blank_hand_frames=int(spec["blank_hand_frames"]),
        fixture_status=str(span_spec["selection_status"]),
    )

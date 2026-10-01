"""Scan bookkeeping: shards, fingerprints, membership, and failures as data."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from seamless_curation import scan
from seamless_curation.corpus import shard_of, stable_unit_interval


def test_shard_assignment_is_stable_and_independent_of_row_order() -> None:
    keys = pd.Series([f"V00_S{i:04d}_I00000001" for i in range(200)])
    first = shard_of(keys, 16)
    shuffled = keys.sample(frac=1.0, random_state=3)
    assert (shard_of(shuffled, 16).sort_index() == first).all()
    assert first.between(0, 15).all()


def test_both_members_of_an_interaction_share_a_shard() -> None:
    rec = pd.DataFrame({"interaction_key": ["V00_S0001_I00000001"] * 2 + ["V00_S0001_I00000002"] * 2})
    shards = shard_of(rec["interaction_key"], 512)
    assert shards.iloc[0] == shards.iloc[1] and shards.iloc[2] == shards.iloc[3]


def test_the_fingerprint_follows_parameters_and_clip_length() -> None:
    from dataclasses import replace

    from seamless_curation.gesture import GestureParams

    base = scan.ScanSettings()
    assert base.fingerprint() == scan.ScanSettings().fingerprint()
    assert base.fingerprint() != scan.ScanSettings(clip_seconds=10.0).fingerprint()
    other = scan.ScanSettings(gesture=replace(GestureParams(), min_speed_mm_s=61.0))
    assert base.fingerprint() != other.fingerprint()


def test_the_fingerprint_covers_every_measurement_module() -> None:
    here = Path(scan.__file__).parent
    for name in scan.MEASUREMENT_MODULES:
        assert (here / name).exists(), name
    for name in ("framing.py", "posture.py", "audio.py", "speech.py", "moi.py", "face.py", "scan.py"):
        assert name in scan.MEASUREMENT_MODULES


def test_membership_follows_the_per_file_inputs() -> None:
    rows = pd.DataFrame({"file_id": ["a", "b"], "interaction_key": ["i", "i"], "source_relbase": ["x/a", "x/b"],
                         "fps": [30.0, 30.0], "video_width": [1080, 1080], "video_height": [1920, 1920],
                         "smplh_anamorphic": ["none", "none"], "npz_present": [True, True],
                         "json_present": [True, True], "wav_present": [True, True]})
    base = scan.membership_hash(rows)
    assert base == scan.membership_hash(rows.iloc[::-1])
    changed = rows.copy()
    changed.loc[0, "fps"] = 29.97
    assert scan.membership_hash(changed) != base


def test_a_shard_is_reused_only_when_fingerprint_and_membership_match(tmp_path: Path) -> None:
    frames = {kind: pd.DataFrame({"file_id": ["a"]}) for kind in scan.SHARD_KINDS}
    scan.write_shard(tmp_path, 3, frames, {"status": "complete", "fingerprint": "f1", "membership": "m1"})
    assert scan.shard_is_current(tmp_path, 3, "f1", "m1")
    assert not scan.shard_is_current(tmp_path, 3, "f2", "m1")
    assert not scan.shard_is_current(tmp_path, 3, "f1", "m2")
    assert not list(tmp_path.glob("*.tmp")), "writes are same-directory temp files then os.replace"


def test_gathering_refuses_a_missing_or_stale_shard(tmp_path: Path) -> None:
    frames = {kind: pd.DataFrame({"file_id": ["a"]}) for kind in scan.SHARD_KINDS}
    scan.write_shard(tmp_path, 0, frames, {"status": "complete", "fingerprint": "f1", "membership": "m0"})
    with pytest.raises(RuntimeError, match="missing"):
        scan.read_shards(tmp_path, 2, "clips")
    scan.write_shard(tmp_path, 1, frames, {"status": "complete", "fingerprint": "old", "membership": "m1"})
    with pytest.raises(RuntimeError, match="fingerprint"):
        scan.read_shards(tmp_path, 2, "clips", fingerprint="f1")
    with pytest.raises(RuntimeError, match="membership"):
        scan.read_shards(tmp_path, 2, "clips", memberships={0: "m0", 1: "changed"})


def test_an_unmeasurable_recording_is_a_row_not_an_exception(tmp_path: Path) -> None:
    record = {"file_id": "V00_S0001_I00000001_P0001", "source_relbase": "nowhere/V00_S0001_I00000001_P0001",
              "npz_present": True, "json_present": True, "fps": 30.0}
    member = scan.measure_recording(record, tmp_path, scan.ScanSettings())
    assert member.recording["measurement_status"] == "missing_files"  # listed but absent on disk
    assert member.clips == [] and member.recording["recording_n_clips"] == 0
    record["npz_present"] = False
    assert scan.measure_recording(record, tmp_path, scan.ScanSettings()).recording["measurement_status"] == "missing_files"


def test_a_corrupt_npz_is_unreadable(tmp_path: Path) -> None:
    base = tmp_path / "V00_S0001_I00000001_P0001"
    base.with_suffix(".npz").write_bytes(b"not a zip archive")
    base.with_suffix(".json").write_text("{}", encoding="utf-8")
    record = {"file_id": base.name, "source_relbase": base.name, "npz_present": True, "json_present": True,
              "fps": 30.0}
    assert scan.measure_recording(record, tmp_path, scan.ScanSettings()).recording["measurement_status"] == "unreadable"


def test_a_zero_frame_npz_is_no_frames(tmp_path: Path) -> None:
    base = tmp_path / "V00_S0001_I00000001_P0001"
    np.savez(base.with_suffix(".npz"), **{key: np.zeros((0, 3), np.float32) for key in scan.REQUIRED_NPZ_KEYS})
    base.with_suffix(".json").write_text(json.dumps({"metadata:vad": []}), encoding="utf-8")
    record = {"file_id": base.name, "source_relbase": base.name, "npz_present": True, "json_present": True,
              "fps": float("nan")}
    assert scan.measure_recording(record, tmp_path, scan.ScanSettings()).recording["measurement_status"] == "no_frames"


def test_stable_unit_interval_is_reproducible_and_in_range() -> None:
    values = [stable_unit_interval("V00", str(i), salt="qa") for i in range(100)]
    assert values == [stable_unit_interval("V00", str(i), salt="qa") for i in range(100)]
    assert all(0.0 <= v < 1.0 for v in values)

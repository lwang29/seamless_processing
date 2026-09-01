from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pandas as pd


SCRIPT = Path(__file__).parents[1] / "scripts" / "inventory_m1.py"
SPEC = importlib.util.spec_from_file_location("inventory_m1", SCRIPT)
assert SPEC and SPEC.loader
inventory_m1 = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = inventory_m1
SPEC.loader.exec_module(inventory_m1)


def test_task_plan_is_contiguous_and_complete() -> None:
    plan = inventory_m1.make_task_plan(129_370, 512)
    assert len(plan) == 512
    assert plan.iloc[0].start_row == 0
    assert plan.iloc[-1].stop_row_exclusive == 129_370
    assert plan.row_count.min() == 252
    assert plan.row_count.max() == 253
    assert (plan.start_row.iloc[1:].to_numpy() == plan.stop_row_exclusive.iloc[:-1].to_numpy()).all()
    assert plan.row_count.sum() == 129_370


def test_file_id_parse_and_expected_path(tmp_path: Path) -> None:
    metadata = tmp_path / "metadata"
    metadata.mkdir()
    pd.DataFrame(
        [
            {
                "file_id": "V03_S1869_I00000302_P5265",
                "label": "naturalistic",
                "split": "train",
                "batch_idx": 12,
                "archive_idx": 7,
                "has_imitator_movement": 0,
                "has_annotation_1p": 0,
                "has_annotation_3p": 0,
            }
        ]
    ).to_csv(metadata / "filelist.csv", index=False)
    config = type("Config", (), {"metadata_root": metadata})()
    frame = inventory_m1.load_filelist(config)
    row = frame.iloc[0]
    assert row.vendor_id == "03"
    assert row.vendor == "V03"
    assert row.session_id == "1869"
    assert row.interaction_id == "00000302"
    assert row.participant_id == "5265"
    assert row.source_relbase == "naturalistic/train/0012/0007/V03_S1869_I00000302_P5265"


def test_probe_stats_all_siblings_but_only_opens_mp4(tmp_path: Path, monkeypatch) -> None:
    source = tmp_path / "source"
    relbase = Path("improvised/dev/0000/0000/V00_S0001_I00000004_P0001")
    base = source / relbase
    base.parent.mkdir(parents=True)
    contents = {
        ".json": b"JSON_PAYLOAD_MUST_NOT_BE_OPENED",
        ".mp4": b"MP4_HEADER_FIXTURE",
        ".npz": b"NPZ_PAYLOAD_MUST_NOT_BE_OPENED",
        ".wav": b"WAV_PAYLOAD_MUST_NOT_BE_OPENED",
    }
    for extension, payload in contents.items():
        base.with_suffix(extension).write_bytes(payload)

    calls: list[Path] = []

    def fake_ffprobe(config, path):
        calls.append(path)
        return {
            "probe_status": "ok",
            "video_stream_present": True,
            "format_duration_s": 10.0,
        }

    monkeypatch.setattr(inventory_m1, "_ffprobe_mp4", fake_ffprobe)
    config = type(
        "Config",
        (),
        {"source_root": source, "config_hash": "config"},
    )()
    provenance = inventory_m1.Provenance("abc123", False)
    row = {
        "file_id": base.name,
        "source_relbase": relbase.as_posix(),
    }
    result = inventory_m1.probe_metadata_row(config, row, provenance)
    assert calls == [base.with_suffix(".mp4")]
    for extension, payload in contents.items():
        key = extension.removeprefix(".")
        assert result[f"{key}_present"] is True
        assert result[f"{key}_size_bytes"] == len(payload)
        assert base.with_suffix(extension).read_bytes() == payload


def test_exact_placeholder_signature_is_not_a_size_threshold() -> None:
    frame = pd.DataFrame(
        {
            "mp4_size_bytes": [260, 261, 262, 261],
            "wav_size_bytes": [58, 58, 58, 59],
        }
    )
    selected = frame.mp4_size_bytes.eq(261) & frame.wav_size_bytes.eq(58)
    assert selected.tolist() == [False, True, False, False]


def test_atomic_json_replaces_without_temporary_residue(tmp_path: Path) -> None:
    output = tmp_path / "marker.json"
    inventory_m1._atomic_json(output, {"status": "complete", "count": 2})
    assert json.loads(output.read_text()) == {"status": "complete", "count": 2}
    assert not list(tmp_path.glob("*.tmp"))
    assert not list(tmp_path.glob(".*.tmp"))


def test_dyad_duration_summary_accepts_vendor_in_group_columns() -> None:
    frame = pd.DataFrame(
        {
            "vendor_id": ["00", "00", "00"],
            "label": ["naturalistic"] * 3,
            "split": ["train"] * 3,
            "session_id": ["0001", "0001", "0002"],
            "interaction_id": ["00000004", "00000004", "00000029"],
            "file_id": ["a", "b", "c"],
            "observed_duration_s": [10.0, 12.0, 20.0],
        }
    )
    summary = inventory_m1._dyad_duration_summary(
        frame, ["vendor_id", "label", "split"]
    )
    assert len(summary) == 1
    row = summary.iloc[0]
    assert row.interaction_count == 2
    assert row.two_member_interactions == 1
    assert row.one_member_interactions == 1
    assert row.dyad_hours == (11.0 + 20.0) / 3600


def test_source_effective_write_guard_uses_access_without_writing(
    tmp_path: Path, monkeypatch
) -> None:
    config = type("Config", (), {"source_root": tmp_path})()
    calls = []

    def fake_access(path, mode, *, effective_ids):
        calls.append((path, mode, effective_ids))
        return False

    monkeypatch.setattr(inventory_m1.os, "access", fake_access)
    assert inventory_m1.source_effectively_writable(config) is False
    assert calls == [(tmp_path, inventory_m1.os.W_OK, True)]

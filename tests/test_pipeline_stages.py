"""Population, scan sharding, verdict store, and the manifest's central promise."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from seamless_curation.corpus import (
    EXCLUDED_RASTERS,
    build_population,
    eligibility_counts,
    shard_of,
    stable_unit_interval,
)
from seamless_curation.review_store import Verdict, VerdictStore
from seamless_curation.scan import (
    ScanSettings,
    read_shards,
    scan_file,
    shard_is_current,
    write_shard,
)

from tests.conftest import make_bundle


# ------------------------------------------------------------------ corpus
def inventory_row(**overrides) -> dict:
    row = {
        "file_id": "V00_S1_I1_P1",
        "vendor": "V00",
        "label": "improvised",
        "split": "train",
        "source_relbase": "improvised/train/0/0/V00_S1_I1_P1",
        "session_id": "S1",
        "participant_id": "P1",
        "interaction_id": "I1",
        "interaction_type": "ipc_conversation",
        "all_modalities_present": True,
        "video_stream_present": True,
        "probe_status": "ok",
        "video_width": 1080,
        "video_height": 1920,
        "video_r_fps": 30.0,
        "video_nb_frames": 6000.0,
        "observed_duration_s": 200.0,
        "npz_size_bytes": 1000,
    }
    row.update(overrides)
    return row


def test_a_normal_file_is_eligible() -> None:
    population = build_population(pd.DataFrame([inventory_row()]))
    assert bool(population["eligible"].iloc[0])
    assert population["raster"].iloc[0] == "1080x1920"


@pytest.mark.parametrize(
    "override,reason",
    [
        ({"all_modalities_present": False}, "incomplete_bundle"),
        ({"video_stream_present": False}, "no_video_stream"),
        ({"probe_status": "no_video_stream"}, "probe_failed"),
        ({"video_width": 2160, "video_height": 2160}, "excluded_raster"),
        ({"video_r_fps": 5.0}, "unusable_frame_rate"),
        ({"video_nb_frames": 7500.0}, "timebase_drift"),
        ({"interaction_type": "charades"}, "no_speech_activity"),
        ({"observed_duration_s": 20.0, "video_nb_frames": 600.0}, "too_short"),
    ],
)
def test_each_ineligibility_is_named(override, reason) -> None:
    population = build_population(pd.DataFrame([inventory_row(**override)]))
    assert not bool(population["eligible"].iloc[0])
    assert population["ineligible_reason"].iloc[0] == reason


def test_seated_posture_and_missing_legs_are_not_grounds_for_exclusion() -> None:
    """The PI's instruction, encoded as a test rather than as a comment.

    Nothing in the population stage looks at posture, leg visibility, or body
    framing. The v0 pipeline's FM1 rejected a quarter of V03 on posture alone.
    """

    import inspect

    from seamless_curation import corpus

    source = inspect.getsource(corpus)
    for forbidden in ("knee", "hip_flexion", "sitting", "shin", "ankle"):
        assert forbidden not in source.lower().replace("#", ""), (
            f"{forbidden!r} is back in the eligibility rules"
        )


def test_excluded_rasters_are_the_ones_whose_smplh_cannot_be_repaired() -> None:
    assert "2160x2160" in EXCLUDED_RASTERS and "1920x1080" in EXCLUDED_RASTERS  # V01 anamorphic
    assert "640x480" in EXCLUDED_RASTERS and "3840x2160" in EXCLUDED_RASTERS    # V03 room cameras
    assert "1080x1920" not in EXCLUDED_RASTERS


def test_eligibility_counts_add_up() -> None:
    rows = [inventory_row(file_id=f"F{i}") for i in range(3)]
    rows.append(inventory_row(file_id="F3", interaction_type="charades"))
    counts = eligibility_counts(build_population(pd.DataFrame(rows)))
    assert counts.total == 4 and counts.eligible == 3
    assert counts.by_reason == {"no_speech_activity": 1}


# ------------------------------------------------------------------- shards
def test_shard_assignment_is_stable_and_independent_of_row_order() -> None:
    ids = pd.Series([f"V00_S{i}_I{i}_P{i}" for i in range(500)])
    first = shard_of(ids, 64)
    second = shard_of(ids.iloc[::-1], 64).iloc[::-1]
    assert list(first) == list(second)
    assert first.between(0, 63).all()
    # Restart safety depends on this: the same file must always land in the same
    # task, or a resubmitted array recomputes a different partition.
    assert list(shard_of(ids.head(10), 64)) == list(first.head(10))


def test_a_shard_is_only_reused_when_the_settings_match(tmp_path: Path) -> None:
    files = pd.DataFrame([{"file_id": "a"}])
    windows = pd.DataFrame([{"file_id": "a", "start_frame": 0}])
    settings = ScanSettings()
    write_shard(tmp_path, 7, files, windows, {"status": "complete", "fingerprint": settings.fingerprint()})

    assert shard_is_current(tmp_path, 7, settings.fingerprint())
    assert not shard_is_current(tmp_path, 7, "a-different-fingerprint")
    assert not shard_is_current(tmp_path, 8, settings.fingerprint())


def test_changing_a_measurement_parameter_changes_the_fingerprint() -> None:
    from dataclasses import replace

    from seamless_curation.gesture import GestureParams

    base = ScanSettings()
    moved = ScanSettings(gesture=replace(GestureParams(), min_speed_mm_s=90.0))
    assert base.fingerprint() != moved.fingerprint()
    assert ScanSettings(window_seconds=20.0).fingerprint() != base.fingerprint()


def test_gathering_refuses_to_silently_skip_a_missing_shard(tmp_path: Path) -> None:
    """The v0 analysis globbed whatever shards existed and reported rates over them."""

    frame = pd.DataFrame([{"file_id": "a"}])
    for index in (0, 1):
        write_shard(tmp_path, index, frame, frame, {"status": "complete", "fingerprint": "x"})
    assert len(read_shards(tmp_path, 2, "files")) == 2
    with pytest.raises(FileNotFoundError, match="missing"):
        read_shards(tmp_path, 3, "files")


def test_scan_file_reports_a_bad_bundle_as_data_not_an_exception(tmp_path: Path) -> None:
    missing = tmp_path / "not_a_file"
    row, windows = scan_file(missing, fps=30.0, settings=ScanSettings())
    assert row["scan_status"].startswith("read_error")
    assert windows == []


def test_scan_file_measures_a_real_bundle(tmp_path: Path, model_root) -> None:
    base = make_bundle(shoulder_swing_deg=55.0, frames=1800).write(tmp_path)
    row, windows = scan_file(base, fps=30.0, settings=ScanSettings(model_root=str(model_root)))
    assert row["scan_status"] == "ok"
    assert row["window_count"] == len(windows) == 4  # 30 s windows, 10 s hop, over 60 s
    assert windows[0]["gesture_frac_speech"] > 0.5


def test_stable_unit_interval_is_reproducible_and_in_range() -> None:
    values = [stable_unit_interval(f"file{i}", salt="s") for i in range(200)]
    assert all(0.0 <= v < 1.0 for v in values)
    assert values == [stable_unit_interval(f"file{i}", salt="s") for i in range(200)]
    assert values != [stable_unit_interval(f"file{i}", salt="t") for i in range(200)]


# ------------------------------------------------------------ verdict store
def test_verdicts_append_and_the_last_one_wins(tmp_path: Path) -> None:
    store = VerdictStore(tmp_path / "v.jsonl")
    store.append(Verdict("r1", "reject", "ann", reasons=("static_hands",), recorded_utc="2026-01-01T00:00:00Z"))
    store.append(Verdict("r1", "accept", "bob", recorded_utc="2026-01-02T00:00:00Z"))
    store.append(Verdict("r2", "unsure", "ann", recorded_utc="2026-01-01T00:00:00Z"))

    resolved = store.resolve().set_index("review_item_id")
    assert resolved.loc["r1", "verdict"] == "accept"
    assert resolved.loc["r1", "reviewer"] == "bob"
    assert bool(resolved.loc["r1", "contested"])
    assert not bool(resolved.loc["r2", "contested"])
    assert store.counts() == {"accept": 1, "reject": 0, "unsure": 1}
    # Nothing is edited in place: the log is also the audit trail.
    assert len(store.records()) == 3


def test_a_torn_final_line_does_not_lose_the_rest(tmp_path: Path) -> None:
    path = tmp_path / "v.jsonl"
    store = VerdictStore(path)
    store.append(Verdict("r1", "accept", "ann"))
    with path.open("a", encoding="utf-8") as handle:
        handle.write('{"review_item_id": "r2", "verdi')
    assert len(store.records()) == 1


def test_an_unknown_verdict_is_refused_at_construction() -> None:
    with pytest.raises(ValueError, match="verdict must be"):
        Verdict("r1", "maybe", "ann")
    with pytest.raises(ValueError, match="reviewer"):
        Verdict("r1", "accept", "")


def test_import_accepts_both_json_and_jsonl(tmp_path: Path) -> None:
    store = VerdictStore(tmp_path / "v.jsonl")
    listing = tmp_path / "a.json"
    listing.write_text(json.dumps([{"review_item_id": "r1", "verdict": "accept", "reviewer": "x"}]))
    lines = tmp_path / "b.jsonl"
    lines.write_text('{"review_item_id": "r2", "verdict": "reject", "reviewer": "x"}\n')
    assert store.import_file(listing) == 1
    assert store.import_file(lines) == 1
    assert store.counts() == {"accept": 1, "reject": 1, "unsure": 0}

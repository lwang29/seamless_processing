from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import soundfile as sf

from seamless_curation.harness import (
    HarnessConfig,
    Provenance,
    _frame_count,
    _relative_output_subdir,
    process_file,
)


def test_output_subdirectory_and_empty_frame_count_guards():
    assert _relative_output_subdir("files/by_vendor", "test") == "files/by_vendor"
    for unsafe in ("/tmp/escape", "../escape", "files/../escape", "."):
        try:
            _relative_output_subdir(unsafe, "test")
        except ValueError:
            pass
        else:
            raise AssertionError(f"unsafe output subdirectory accepted: {unsafe}")

    empty = np.empty((0,), dtype=bool)
    arrays = {
        "translation": np.empty((0, 3), dtype="float32"),
        "keypoints": np.empty((0, 133, 3), dtype="float32"),
        "smplh_mask": empty,
        "movement_mask": None,
        "box_mask": empty,
    }
    assert _frame_count(arrays) == (0, "empty:basis=translation")


def test_atomic_per_file_output_and_restart_marker(tmp_path: Path):
    source_root = tmp_path / "source"
    source_base = source_root / "improvised/dev/0000/0000/test_file"
    source_base.parent.mkdir(parents=True)
    n_frames = 121
    rng = np.random.default_rng(7)
    translation = np.cumsum(rng.normal(size=(n_frames, 3)) * 0.001, axis=0).astype("float32")
    keypoints = rng.normal(size=(n_frames, 133, 3)).astype("float32")
    masks = np.ones(n_frames, dtype=bool)
    np.savez_compressed(
        f"{source_base}.npz",
        **{
            "smplh:translation": translation,
            "smplh:is_valid": masks,
            "boxes_and_keypoints:keypoints": keypoints,
            "boxes_and_keypoints:is_valid_box": masks,
        },
    )
    sf.write(f"{source_base}.wav", np.zeros(2 * 48000, dtype="float32"), 48000, subtype="FLOAT")

    config = HarnessConfig(
        config_path=tmp_path / "config.yaml",
        worktree=tmp_path,
        source_root=source_root.resolve(),
        manifest=tmp_path / "manifest.csv",
        output_root=tmp_path / "output",
        per_file_dir="files",
        completion_dir="complete",
        error_dir="errors",
        nominal_fps=30.0,
        window_frames=120,
        hop_frames=30,
        translation_scale_to_nominal_mm=1000.0,
        translation_basis="smplh_model_space_root_translation",
        physical_units_verified=False,
        wrist_basis="released_coco_wholebody_body_wrists_9_10_native_frame_pixels_per_frame",
        guard_band_frames=3,
        ffprobe_bin="ffprobe",
        config_hash="test-config-hash",
    )
    provenance = Provenance(git_sha="test-git-sha", git_dirty=False)
    row = {
        "file_id": "test_file",
        "label": "improvised",
        "split": "dev",
        "vendor": "V00",
        "source_relbase": "improvised/dev/0000/0000/test_file",
    }

    first = process_file(row, config, provenance)
    assert first.status == "complete" and first.rows == 1
    assert first.output_path is not None and first.output_path.is_file()
    assert first.marker_path is not None and first.marker_path.is_file()
    assert not list(config.output_root.rglob("*.tmp"))
    table = pq.read_table(first.output_path)
    assert table.num_rows == 1
    assert table.column("git_sha")[0].as_py() == "test-git-sha"
    assert table.column("config_hash")[0].as_py() == "test-config-hash"
    assert table.column("physical_units_verified")[0].as_py() is False
    assert math.isnan(table.column("valid_frac_all")[0].as_py())
    assert table.column("valid_status")[0].as_py() == "missing_masks:movement"
    # The absent movement mask no longer erases the two masks that are present.
    assert table.column("valid_frac_smplh")[0].as_py() == 1.0
    assert table.column("valid_frac_box")[0].as_py() == 1.0
    assert table.column("valid_frac_smplh_and_box")[0].as_py() == 1.0
    assert math.isnan(table.column("valid_frac_movement")[0].as_py())
    assert table.column("valid_split_status")[0].as_py() == (
        "smplh=ok;box=ok;movement=missing_mask:movement;smplh_and_box=ok"
    )
    assert table.column("guard_band_frames")[0].as_py() == 3
    assert table.column("hand_avail_frac_left")[0].as_py() == 1.0
    assert table.column("hand_avail_frac_right")[0].as_py() == 1.0
    assert table.column("hand_avail_status")[0].as_py() == "left=ok;right=ok"
    assert math.isnan(table.column("duration_mismatch_s")[0].as_py())
    assert table.column("duration_fps_status")[0].as_py() == "unavailable_missing_mp4"
    marker = json.loads(first.marker_path.read_text())
    assert marker["status"] == "complete"

    second = process_file(row, config, provenance)
    assert second.status == "skipped" and second.rows == first.rows

    # A dirty worktree remains stamped, but cannot reuse a marker whose SHA alone
    # fails to identify the uncommitted implementation.
    dirty = process_file(row, config, Provenance(git_sha="test-git-sha", git_dirty=True))
    assert dirty.status == "complete"
    assert dirty.output_path is not None
    assert pq.read_table(dirty.output_path).column("git_dirty")[0].as_py() is True

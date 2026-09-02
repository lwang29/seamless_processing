"""The one promise the whole pipeline exists to keep.

Nothing reaches ``accepted_clips.csv`` that a reviewer did not accept. It is an
inner join on the verdict log rather than a filter over the candidates, so there
is no code path in which forgetting a condition lets an unreviewed clip through
— but the property is worth asserting anyway, because it is the property.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest
import yaml

from seamless_curation.config import load_config
from seamless_curation.manifest import build_manifests
from seamless_curation.review_store import Verdict, VerdictStore


def candidate(review_item_id: str, clip_index: int = 0, **overrides) -> dict:
    row = {
        "clip_id": f"{review_item_id}_c{clip_index}",
        "review_item_id": review_item_id,
        "file_id": f"F_{review_item_id}",
        "source_relbase": f"improvised/train/0/0/F_{review_item_id}",
        "vendor": "V00",
        "label": "improvised",
        "split": "train",
        "session_id": "S1",
        "participant_id": review_item_id,
        "interaction_id": "I1",
        "interaction_type": "ipc_conversation",
        "start_frame": clip_index * 900,
        "end_frame": (clip_index + 1) * 900,
        "start_s": clip_index * 30.0,
        "window_seconds": 30.0,
        "fps": 30.0,
        "speech_seconds": 14.0,
        "gesture_frac_speech": 0.7,
        "speech_segments_covered": 0.9,
        "episode_count_speech": 5,
        "wrist_excursion_p90_mm": 260.0,
        "elbow_excursion_p90_mm": 90.0,
        "gesture_speech_ratio": 2.0,
        "sync_r": 0.4,
        "sync_lag_s": 0.0,
        "consistency_r": 0.9,
        "smplh_valid_frac": 1.0,
        "smplh_longest_invalid_s": 0.0,
        "hand_frozen_frac": 0.0,
        "clip_score": 0.8,
    }
    row.update(overrides)
    return row


@pytest.fixture
def run(tmp_path: Path):
    output = tmp_path / "out"
    output.mkdir()
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "schema_version": 2,
                "run_id": "test",
                "source_root": str(tmp_path / "src"),
                "model_root": "model_files",
                "inventory": str(tmp_path / "inv.parquet"),
                "outputs": {"root": str(output), "private_root": str(tmp_path / "private")},
            }
        )
    )
    config = load_config(config_path)
    rows = [candidate("r1", 0), candidate("r1", 1), candidate("r2"), candidate("r3"), candidate("r4")]
    pd.DataFrame(rows).to_parquet(config.candidates_path, index=False)
    return config


def test_only_accepted_and_reviewed_clips_reach_the_manifest(run) -> None:
    store = VerdictStore(run.verdict_log)
    store.append(Verdict("r1", "accept", "ann", saw_video=True))
    store.append(Verdict("r2", "reject", "ann", reasons=("static_hands",)))
    store.append(Verdict("r3", "unsure", "ann"))
    # r4 is never reviewed at all.

    summary = build_manifests(run)
    manifest = pd.read_csv(run.accepted_clips_path)

    assert set(manifest["review_item_id"]) == {"r1"}
    assert len(manifest) == 2, "both of r1's clips are accepted by the file-level verdict"
    assert summary["accepted_clips"] == 2
    assert summary["verdicts"] == {"accept": 1, "reject": 1, "unsure": 1}
    assert summary["reject_reasons"] == {"static_hands": 1}


def test_the_strict_manifest_holds_only_verdicts_taken_with_sound(run) -> None:
    store = VerdictStore(run.verdict_log)
    store.append(Verdict("r1", "accept", "ann", saw_video=True))
    store.append(Verdict("r2", "accept", "ann", saw_video=False))

    build_manifests(run)
    everything = pd.read_csv(run.accepted_clips_path)
    with_audio = pd.read_csv(run.output_root / "accepted_clips_with_audio.csv")

    assert set(everything["review_item_id"]) == {"r1", "r2"}
    assert set(with_audio["review_item_id"]) == {"r1"}
    assert set(everything["review_evidence"]) == {"card+video", "card"}


def test_an_overturned_reject_is_honoured(run) -> None:
    """Last write wins, so a second reviewer can rescue a clip."""

    store = VerdictStore(run.verdict_log)
    store.append(Verdict("r1", "reject", "ann", recorded_utc="2026-01-01T00:00:00Z"))
    store.append(Verdict("r1", "accept", "pi", recorded_utc="2026-02-01T00:00:00Z"))

    summary = build_manifests(run)
    manifest = pd.read_csv(run.accepted_clips_path)
    assert set(manifest["review_item_id"]) == {"r1"}
    assert set(manifest["reviewer"]) == {"pi"}
    assert summary["contested_items"] == 1


def test_no_verdicts_means_an_empty_manifest_not_a_crash(run) -> None:
    summary = build_manifests(run)
    assert summary["accepted_clips"] == 0
    assert pd.read_csv(run.accepted_clips_path).empty


def test_the_manifest_records_who_or_what_reviewed_each_clip(run) -> None:
    store = VerdictStore(run.verdict_log)
    store.append(Verdict("r1", "accept", "claude-review-3", verdict_source="model:claude-opus-5"))
    store.append(Verdict("r2", "accept", "ann", verdict_source="human"))

    summary = build_manifests(run)
    manifest = pd.read_csv(run.accepted_clips_path)
    assert set(manifest["verdict_source"]) == {"model:claude-opus-5", "human"}
    assert summary["verdict_sources"] == {"model:claude-opus-5": 1, "human": 1}


def test_the_manifest_names_frame_ranges_and_never_copies_media(run) -> None:
    """The accepted subset is a definition, not a second copy of the dataset."""

    store = VerdictStore(run.verdict_log)
    store.append(Verdict("r1", "accept", "ann"))
    build_manifests(run)
    manifest = pd.read_csv(run.accepted_clips_path)

    assert {"source_relbase", "start_frame", "end_frame", "fps"} <= set(manifest.columns)
    assert (manifest["end_frame"] > manifest["start_frame"]).all()
    assert not any(path.suffix in {".mp4", ".npz", ".wav"} for path in run.output_root.iterdir())


def test_verify_reads_the_frames_the_manifest_promises(tmp_path: Path, model_root) -> None:
    """A manifest is a promise about the source tree; verify cashes it.

    The check exists because a frame range is easy to get subtly wrong — an
    off-by-one, a stale fps, a row whose file has moved — and none of those show
    up until a training run reads a short tensor.
    """

    from seamless_curation.manifest import build_manifests, verify_manifest
    from seamless_curation.review_store import Verdict, VerdictStore
    from tests.conftest import make_bundle

    import yaml

    from seamless_curation.config import load_config

    source = tmp_path / "src" / "improvised" / "train" / "0" / "0"
    name = "V00_S1_I1_P1"
    make_bundle(frames=1800).write(source, name)

    output = tmp_path / "out"
    output.mkdir(parents=True)
    config_path = tmp_path / "c.yaml"
    config_path.write_text(yaml.safe_dump({
        "schema_version": 2, "run_id": "t", "source_root": str(tmp_path / "src"),
        "model_root": str(model_root), "inventory": str(tmp_path / "i.parquet"),
        "outputs": {"root": str(output), "private_root": str(tmp_path / "p")},
    }))
    config = load_config(config_path)

    row = candidate("r1", 0)
    row.update({
        "file_id": name,
        "source_relbase": f"improvised/train/0/0/{name}",
        "start_frame": 300, "end_frame": 1200, "start_s": 10.0, "window_seconds": 30.0,
        # The synthetic VAD is (1,6), (9,15), (18,25); the 10-40 s window
        # therefore overlaps it for 5 + 7 = 12 seconds.
        "speech_seconds": 12.0,
    })
    pd.DataFrame([row]).to_parquet(config.candidates_path, index=False)
    VerdictStore(config.verdict_log).append(Verdict("r1", "accept", "ann", file_id=name))
    build_manifests(config)

    summary = verify_manifest(config, sample=1)
    assert summary["clips"] == 1
    assert summary["upper_body_pose_frames"] == 900
    assert summary["failures"] == [], summary["failures"]


def test_verify_catches_a_frame_range_that_runs_off_the_end(tmp_path: Path, model_root) -> None:
    from seamless_curation.manifest import build_manifests, verify_manifest
    from seamless_curation.review_store import Verdict, VerdictStore
    from tests.conftest import make_bundle

    import yaml

    from seamless_curation.config import load_config

    source = tmp_path / "src" / "improvised" / "train" / "0" / "0"
    name = "V00_S1_I1_P1"
    make_bundle(frames=900).write(source, name)

    output = tmp_path / "out"
    output.mkdir(parents=True)
    config_path = tmp_path / "c.yaml"
    config_path.write_text(yaml.safe_dump({
        "schema_version": 2, "run_id": "t", "source_root": str(tmp_path / "src"),
        "model_root": str(model_root), "inventory": str(tmp_path / "i.parquet"),
        "outputs": {"root": str(output), "private_root": str(tmp_path / "p")},
    }))
    config = load_config(config_path)

    row = candidate("r1", 0)
    row.update({
        "file_id": name,
        "source_relbase": f"improvised/train/0/0/{name}",
        "start_frame": 600, "end_frame": 1500,   # 900 frames promised, 300 available
    })
    pd.DataFrame([row]).to_parquet(config.candidates_path, index=False)
    VerdictStore(config.verdict_log).append(Verdict("r1", "accept", "ann", file_id=name))
    build_manifests(config)

    summary = verify_manifest(config, sample=1)
    assert summary["failures"] and "promised 900 frames" in summary["failures"][0]

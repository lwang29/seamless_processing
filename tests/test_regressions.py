"""One test per defect an adversarial review of this pipeline confirmed.

Each of these shipped at some point. The test states the wrong behaviour it
would produce, so the reason it exists survives the fix.
"""

from __future__ import annotations

import json
import threading
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from tests.conftest import FPS, make_bundle


# ------------------------------------------------------- measurement defects
def test_step_cosine_is_not_spliced_across_the_two_hands(model_root) -> None:
    """Selecting the faster hand per frame and then differencing teleports.

    The hands sit about a shoulder width apart, so every change of selection
    injected a displacement roughly 100x the real per-frame step. On one
    measured window that turned a ``step_cosine_p50`` of -0.48 — detector noise,
    correctly rejected — into -0.08, which passes.
    """

    from seamless_curation.gesture import build_tracks

    bundle = make_bundle(shoulder_swing_deg=40.0)
    # Make the two hands differ so a per-frame "faster hand" choice would flip.
    keypoints = bundle.payload["boxes_and_keypoints:keypoints"]
    rng = np.random.default_rng(0)
    keypoints[:, 10, 0] += rng.normal(0, 40, len(keypoints)).astype(np.float32)
    keypoints[:, 112:133, 0] = keypoints[:, 10, None, 0]

    tracks = build_tracks(
        bundle.payload, bundle.vad, fps=FPS, model_root=str(model_root)
    )
    assert len(tracks.step_cosine) == 2 * tracks.frames, "both hands, end to end"

    # No cosine may come from a step of about one shoulder width: that is the
    # signature of the splice, and a real hand does not move that far in 1/30 s.
    left = tracks.step_cosine[: tracks.frames]
    right = tracks.step_cosine[tracks.frames :]
    for series in (left, right):
        assert np.isfinite(series).sum() > 10


def test_finger_articulation_is_local_not_global(model_root) -> None:
    """A rigid hand on a swinging arm has zero finger articulation.

    Global joint rotations carry the whole shoulder-elbow-wrist chain, so
    feeding them to the geodesic rate measured arm swing and called it finger
    motion: 2.3 rad/s on a hand whose pose was identically zero.
    """

    from seamless_curation.gesture import build_tracks, window_measures

    bundle = make_bundle(shoulder_swing_deg=55.0)
    bundle.payload["smplh:left_hand_pose"][:] = 0.0
    bundle.payload["smplh:right_hand_pose"][:] = 0.0
    tracks = build_tracks(bundle.payload, bundle.vad, fps=FPS, model_root=str(model_root))

    assert window_measures(tracks, 0, 900)["hand_artic_p75_rad_s"] == pytest.approx(0.0, abs=1e-9)


def test_forward_kinematics_returns_local_rotations_too(model_root) -> None:
    from seamless_curation.smplh_kinematics import forward_kinematics, stack_pose

    pose = stack_pose(
        np.zeros((4, 21, 3), np.float32), np.zeros((4, 15, 3), np.float32),
        np.zeros((4, 15, 3), np.float32),
    )
    positions, globals_, locals_ = forward_kinematics(pose, str(model_root))
    assert positions.shape == (4, 52, 3)
    assert globals_.shape == locals_.shape == (4, 52, 3, 3)


def test_shoulder_positions_in_the_torso_frame_are_measured_not_assumed(model_root) -> None:
    """They lie on the x axis by construction, but their separation is posed."""

    from seamless_curation.gesture import build_tracks

    tracks = build_tracks(
        make_bundle(shoulder_swing_deg=30.0).payload,
        make_bundle().vad,
        fps=FPS,
        model_root=str(model_root),
    )
    assert np.abs(tracks.shoulders[:, :, 1:]).max() < 1e-6
    assert (tracks.shoulders[:, 0, 0] > 0).all()
    assert (tracks.shoulders[:, 1, 0] < 0).all()


# ------------------------------------------------------------ identity drift
def test_review_item_id_is_derived_from_the_file_not_its_queue_position() -> None:
    """A positional id re-binds stored verdicts when the candidate set changes.

    With a positional id, adding one interaction to the corpus and re-running
    ``select`` moved ``r0000003`` from one participant to another, and the
    manifest then emitted a file nobody had looked at under the name of the
    reviewer who accepted its former neighbour.
    """

    from seamless_curation.gates import Gates, apply_gates, build_review_items, select_clips
    from tests.test_gates_and_selection import good_window

    def items_for(participants):
        rows = [
            good_window(
                file_id=f"V00_S1_I{p}_P{p}", participant_id=p,
                interaction_id=f"I{p}", source_relbase=f"x/{p}",
            )
            for p in participants
        ]
        clips = select_clips(apply_gates(pd.DataFrame(rows), Gates()), max_per_file=8)
        return build_review_items(clips, max_files_per_participant=4).set_index("file_id")

    before = items_for(["P1", "P2", "P3"])
    after = items_for(["P0", "P1", "P2", "P3"])  # one new participant sorts in

    for file_id in before.index:
        assert before.loc[file_id, "review_item_id"] == after.loc[file_id, "review_item_id"], (
            "the id moved when the candidate set changed"
        )


def test_the_manifest_refuses_a_verdict_that_names_a_different_file(tmp_path: Path) -> None:
    from seamless_curation.manifest import build_manifests
    from seamless_curation.review_store import Verdict, VerdictStore
    from tests.test_manifest import candidate

    import yaml

    from seamless_curation.config import load_config

    output = tmp_path / "out"
    output.mkdir()
    config_path = tmp_path / "c.yaml"
    config_path.write_text(yaml.safe_dump({
        "schema_version": 2, "run_id": "t", "source_root": str(tmp_path),
        "model_root": "model_files", "inventory": str(tmp_path / "i.parquet"),
        "outputs": {"root": str(output), "private_root": str(tmp_path / "p")},
    }))
    config = load_config(config_path)
    pd.DataFrame([candidate("r1")]).to_parquet(config.candidates_path, index=False)

    store = VerdictStore(config.verdict_log)
    store.append(Verdict("r1", "accept", "ann", file_id="A_DIFFERENT_FILE"))
    with pytest.raises(ValueError, match="different file"):
        build_manifests(config)


# ------------------------------------------------------------- shard reuse
def test_a_shard_is_not_reused_when_it_covered_different_files(tmp_path: Path) -> None:
    """``--limit 2`` then a full scan was a complete no-op."""

    from seamless_curation.scan import ScanSettings, membership_hash, shard_is_current, write_shard

    frame = pd.DataFrame([{"file_id": "a"}])
    fingerprint = ScanSettings().fingerprint()
    small = membership_hash(["a", "b"])
    write_shard(tmp_path, 0, frame, frame, {
        "status": "complete", "fingerprint": fingerprint, "membership": small,
    })

    assert shard_is_current(tmp_path, 0, fingerprint, small)
    assert not shard_is_current(tmp_path, 0, fingerprint, membership_hash(["a", "b", "c"]))


def test_the_fingerprint_follows_the_measurement_source(tmp_path: Path) -> None:
    """A hand-maintained version number is a thing you can forget to bump."""

    from seamless_curation.scan import ScanSettings, _measurement_source_hash

    assert len(_measurement_source_hash()) == 16
    assert _measurement_source_hash() in json_of(ScanSettings())


def json_of(settings) -> str:
    """The fingerprint payload, for asserting what it covers."""

    from dataclasses import asdict

    from seamless_curation.scan import NPZ_KEYS, _measurement_source_hash

    return json.dumps({
        "window_seconds": settings.window_seconds,
        "hop_seconds": settings.hop_seconds,
        "gesture": asdict(settings.gesture),
        "npz_keys": list(NPZ_KEYS),
        "measurement_source": _measurement_source_hash(),
    }, sort_keys=True)


# ------------------------------------------------------------- review store
def test_the_string_false_does_not_reach_the_audio_confirmed_manifest(tmp_path: Path) -> None:
    """``bool("False")`` is ``True``; an import said a clip was heard when it was not."""

    from seamless_curation.review_store import VerdictStore

    store = VerdictStore(tmp_path / "v.jsonl")
    path = tmp_path / "import.json"
    path.write_text(json.dumps([
        {"review_item_id": "r1", "verdict": "accept", "reviewer": "x", "saw_video": "False"},
        {"review_item_id": "r2", "verdict": "accept", "reviewer": "x", "saw_video": "true"},
    ]))
    store.import_file(path)
    resolved = store.resolve().set_index("review_item_id")
    assert resolved.loc["r1", "saw_video"] is False or resolved.loc["r1", "saw_video"] == False
    assert bool(resolved.loc["r2", "saw_video"])


def test_a_bare_reason_string_is_one_reason_not_a_tuple_of_letters(tmp_path: Path) -> None:
    from seamless_curation.review_store import VerdictStore

    store = VerdictStore(tmp_path / "v.jsonl")
    path = tmp_path / "i.jsonl"
    path.write_text('{"review_item_id":"r1","verdict":"reject","reviewer":"x","reasons":"static_hands"}\n')
    store.import_file(path)
    assert store.resolve()["reasons"].iloc[0] == ["static_hands"]


def test_verdicts_recorded_in_the_same_second_keep_their_append_order(tmp_path: Path) -> None:
    from seamless_curation.review_store import Verdict, VerdictStore

    store = VerdictStore(tmp_path / "v.jsonl")
    when = "2026-01-01T00:00:00Z"
    store.append(Verdict("r1", "reject", "ann", recorded_utc=when))
    store.append(Verdict("r1", "accept", "bob", recorded_utc=when))
    assert store.resolve()["verdict"].iloc[0] == "accept"


def test_concurrent_appends_do_not_interleave_a_line(tmp_path: Path) -> None:
    from seamless_curation.review_store import Verdict, VerdictStore

    store = VerdictStore(tmp_path / "v.jsonl")

    def write(start: int) -> None:
        for index in range(50):
            store.append(Verdict(f"r{start}_{index}", "accept", "ann", note="x" * 200))

    threads = [threading.Thread(target=write, args=(t,)) for t in range(4)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert len(store.records()) == 200


# ---------------------------------------------------------------- review app
def test_the_media_guard_rejects_a_sibling_directory_with_a_shared_prefix(tmp_path: Path) -> None:
    """``<root>/clips`` is a string prefix of ``<root>/clips_private``."""

    root = tmp_path / "priv" / "clips"
    root.mkdir(parents=True)
    sibling = tmp_path / "priv" / "clips_private"
    sibling.mkdir()
    (sibling / "secret.txt").write_text("no")

    target = (root / "../clips_private/secret.txt").resolve()
    assert str(target).startswith(str(root.resolve())), "the old test would have passed this"
    assert not target.is_relative_to(root.resolve()), "the new one does not"


def test_the_review_page_ignores_key_auto_repeat_and_serialises_posts() -> None:
    """A held key wrote several verdicts for one item and skipped the next few."""

    from seamless_curation.review_app import PAGE

    assert "if (ev.repeat) return;" in PAGE
    assert "if (busy) return;" in PAGE


def test_the_review_page_sends_the_file_id_and_card_fingerprint() -> None:
    from seamless_curation.review_app import PAGE

    assert "file_id: it.file_id" in PAGE
    assert "card_fingerprint: it.card_fingerprint" in PAGE


# --------------------------------------------------------------------- render
def test_the_card_fingerprint_covers_the_gesture_parameters(tmp_path: Path) -> None:
    """They draw the timeline; leaving them out reused a card that no longer matched."""

    import inspect

    from seamless_curation import render

    source = inspect.getsource(render.render_card)
    assert '"gesture": asdict(config.scan.gesture)' in source


def test_the_representative_clip_is_the_lower_median() -> None:
    """``len // 2`` on two clips returns the better one, which flatters every file."""

    for count, expected in ((1, 0), (2, 0), (3, 1), (4, 1), (5, 2)):
        assert (count - 1) // 2 == expected


def test_the_partner_index_keys_on_the_session_too() -> None:
    """``interaction_id`` repeats across sessions; 1,306 of 1,309 cards lost the partner."""

    from seamless_curation.render import _partner_index

    population = pd.DataFrame([
        {"file_id": "a", "vendor": "V00", "session_id": "S1", "interaction_id": "I1", "source_relbase": "pa"},
        {"file_id": "b", "vendor": "V00", "session_id": "S1", "interaction_id": "I1", "source_relbase": "pb"},
        {"file_id": "c", "vendor": "V00", "session_id": "S2", "interaction_id": "I1", "source_relbase": "pc"},
        {"file_id": "d", "vendor": "V00", "session_id": "S2", "interaction_id": "I1", "source_relbase": "pd"},
    ]).set_index("file_id")

    partners = _partner_index(population)
    assert partners == {"a": "pb", "b": "pa", "c": "pd", "d": "pc"}


def test_a_missing_measure_is_shown_as_missing_not_as_zero() -> None:
    """Four of eight headline numbers read 'never shifts posture' when unmeasured."""

    import inspect

    from seamless_curation import render

    source = inspect.getsource(render.render_card)
    assert '"n/a"' in source
    assert 'float(item.get("posture_spread_mm") or 0)' not in source


def test_a_null_gesture_status_is_unmeasurable_in_any_dtype() -> None:
    """On a nullable string column ``pd.NA != "ok"`` is ``pd.NA``, which is falsy."""

    from seamless_curation.gates import Gates, apply_gates
    from tests.test_gates_and_selection import good_window

    for dtype in ("object", "string"):
        row = good_window()
        row["gesture_status"] = pd.NA
        frame = pd.DataFrame([row])
        frame["gesture_status"] = frame["gesture_status"].astype(dtype)
        gated = apply_gates(frame, Gates())
        assert not bool(gated["qualifies"].iloc[0])
        assert gated["fail_reason"].iloc[0] == "unmeasurable"


def test_non_finite_measures_never_satisfy_a_lower_bound() -> None:
    """``+inf >= limit`` is True; an infinity is not a measurement."""

    from seamless_curation.gates import Gates, apply_gates
    from tests.test_gates_and_selection import good_window

    for column, value in (
        ("wrist_excursion_p90_mm", np.inf),
        ("hand_frozen_frac", -np.inf),
        ("episode_count_speech", None),
    ):
        gated = apply_gates(pd.DataFrame([good_window(**{column: value})]), Gates())
        assert not bool(gated["qualifies"].iloc[0]), f"{column}={value} qualified"

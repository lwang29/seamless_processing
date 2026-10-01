"""End to end on a miniature release: pairing, level propagation, missing values.

The release is built by :mod:`tests.builders_pipeline`; see its docstring for the
four interactions and why each exists. These tests assert properties a
downstream user relies on without re-deriving them.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from tests.builders_pipeline import run_pipeline


@pytest.fixture(scope="module")
def run(tmp_path_factory, model_root):
    root = tmp_path_factory.mktemp("mini_release")
    tables, report, cat = run_pipeline(root, model_root)
    return {"tables": tables, "report": report, "catalog": cat, "root": root}


def _rec(run, file_id):
    rec = run["tables"]["recordings"]
    return rec.loc[rec["file_id"] == file_id].iloc[0]


def _inter(run, key):
    inter = run["tables"]["interactions"]
    return inter.loc[inter["interaction_key"] == key].iloc[0]


def test_the_tables_validate(run) -> None:
    assert run["report"].ok, run["report"].errors


def test_every_recording_has_a_row_and_nothing_is_dropped(run) -> None:
    rec = run["tables"]["recordings"]
    assert len(rec) == 7
    missing = _rec(run, "V03_S0001_I00000135_P0100")
    assert missing["measurement_status"] == "missing_files"
    assert missing["recording_n_clips"] == 0
    assert missing["recording_posture"] == "unknown"


def test_the_clip_grid_keeps_the_partial_tail(run) -> None:
    clips = run["tables"]["clips"]
    own = clips.loc[clips["file_id"] == "V00_S0001_I00000129_P0010"].sort_values("clip_index")
    assert list(own["clip_index"]) == [0, 1]
    assert list(own["start_frame"]) == [0, 900] and list(own["end_frame"]) == [900, 1500]
    assert own["is_partial"].tolist() == [False, True]
    assert own["clip_id"].iloc[0] == "V00_S0001_I00000129_P0010_L30_C000"


def test_partner_links_are_symmetric_and_time_aligned(run) -> None:
    clips = run["tables"]["clips"].set_index("clip_id")
    a = "V00_S0001_I00000129_P0010_L30_C001"
    b = "V00_S0001_I00000129_P0020_L30_C001"
    assert clips.loc[a, "partner_clip_id"] == b and clips.loc[b, "partner_clip_id"] == a
    assert clips.loc[a, "partner_link_status"] == "linked"
    assert clips.loc[a, "window_id"] == clips.loc[b, "window_id"] == "V00_S0001_I00000129_L30_W001"


def test_partner_missing_and_unmeasured_partners_are_explicit(run) -> None:
    clips = run["tables"]["clips"]
    solo = clips.loc[clips["file_id"] == "V00_S0002_I00000130_P0030"]
    assert (solo["partner_link_status"] == "partner_missing").all()
    assert solo["partner_clip_id"].isna().all()
    assert (solo["speaking_role"] == "unknown").all()
    half = clips.loc[clips["file_id"] == "V03_S0001_I00000135_P0101"]
    assert (half["partner_link_status"] == "partner_not_measured").all()
    # the measured member sorts second: its clips must still be listed in the dyad windows
    windows = run["tables"]["interaction_windows"]
    mine = windows.loc[windows["interaction_key"] == "V03_S0001_I00000135"]
    assert mine["member_clip_id_1"].isna().all()
    assert set(mine["member_clip_id_2"]) == set(half["clip_id"])
    assert set(mine["window_posture_pair"]) <= {"absent+standing", "absent+unclear", "absent+unknown", "absent+sitting"}
    assert _inter(run, "V00_S0002_I00000130")["pairing_status"] == "partner_missing"
    assert _inter(run, "V00_S0002_I00000130")["posture_pair"] == "absent+standing"


def test_durations_that_disagree_are_not_linked(run) -> None:
    inter = _inter(run, "V01_S0003_I00000139")
    assert inter["pair_duration_agreement"] == "major_mismatch"
    assert bool(inter["pair_fps_differs"])
    assert math.isnan(inter["interaction_speech_overlap_frac"])
    clips = run["tables"]["clips"]
    assert (clips.loc[clips["interaction_key"] == "V01_S0003_I00000139", "partner_link_status"]
            == "not_time_aligned").all()


def test_conversation_level_values_need_both_members(run) -> None:
    both = _inter(run, "V00_S0001_I00000129")
    assert both["speech_status"] == "both_annotated"
    assert 0.0 <= both["interaction_speech_overlap_frac"] <= 1.0
    assert both["interaction_turn_switch_rate_per_min"] > 0
    assert both["interaction_pair_speech_coverage_frac"] == pytest.approx(1.0, abs=1e-3)
    # the conversation's shares are exact aggregates of its dyad windows
    windows = run["tables"]["interaction_windows"]
    mine = windows.loc[windows["interaction_key"] == "V00_S0001_I00000129"]
    dur = mine["window_end_s"] - mine["window_start_s"]
    silence = (mine["window_mutual_silence_frac"] * dur).sum() / dur.sum()
    assert both["interaction_mutual_silence_frac"] == pytest.approx(silence, abs=1e-5)
    assert not math.isnan(both["interaction_expressivity_mean"])
    for key in ("V00_S0002_I00000130", "V03_S0001_I00000135"):
        one = _inter(run, key)
        assert math.isnan(one["interaction_expressivity_mean"]), key
        assert math.isnan(one["interaction_mutual_silence_frac"]), key


def test_speaking_roles_follow_who_talks(run) -> None:
    clips = run["tables"]["clips"].set_index("clip_id")
    # clip 0 is [0, 30): P0010 speaks 1-8 and 20-27 (14 s), P0020 9-18 and 28-30 (11 s)
    assert clips.loc["V00_S0001_I00000129_P0010_L30_C000", "speaking_role"] == "both"
    # clip 1 is [30, 50): P0010 speaks 40-46 (6 s), P0020 30-38 (8 s)
    assert clips.loc["V00_S0001_I00000129_P0020_L30_C001", "speaking_role"] == "both"
    windows = run["tables"]["interaction_windows"].set_index("window_id")
    window = windows.loc["V00_S0001_I00000129_L30_W000"]
    assert window["member_clip_id_1"] == "V00_S0001_I00000129_P0010_L30_C000"
    assert window["member_clip_id_2"] == "V00_S0001_I00000129_P0020_L30_C000"


def test_posture_is_read_per_participant(run) -> None:
    assert _rec(run, "V00_S0001_I00000129_P0010")["recording_posture"] == "standing"
    seated = _rec(run, "V00_S0001_I00000129_P0020")
    assert seated["recording_posture"] == "sitting"
    assert seated["posture_rule"] == "v00_knee"
    assert _inter(run, "V00_S0001_I00000129")["posture_pair"] == "sitting+standing"
    clips = run["tables"]["clips"]
    shares = clips[["posture_standing_frac", "posture_sitting_frac", "posture_unclear_frac",
                    "posture_unobserved_frac"]].sum(axis=1)
    assert np.allclose(shares, 1.0, atol=1e-5)


def test_moi_counts_are_missing_not_zero_when_unannotated(run) -> None:
    clips = run["tables"]["clips"].set_index("clip_id")
    annotated = clips.loc["V00_S0001_I00000129_P0010_L30_C000"]
    assert annotated["moi_3p_count"] == 2  # 10-12 s and 28-30 s
    assert clips.loc["V00_S0001_I00000129_P0010_L30_C001", "moi_3p_count"] == 1  # zero-length at 41 s
    assert pd.isna(clips.loc["V00_S0001_I00000129_P0020_L30_C000", "moi_3p_count"])
    inter = _inter(run, "V00_S0001_I00000129")
    assert inter["moi_3p_coverage"] == "one"
    assert pd.isna(inter["interaction_moi_3p_count"])
    events = run["tables"]["moi_events"]
    zero = events.loc[events["moi_start_s"] == 41]
    assert set(zero["moi_malformed"]) == {"zero_length"}
    boundary = events.loc[(events["moi_start_s"] == 28) & (events["moi_end_s"] == 30)]
    assert set(boundary["moi_clip_index_start"]) == {0} and set(boundary["moi_clip_index_end"]) == {0}


def test_metadata_missing_values_keep_their_reasons(run) -> None:
    par = run["tables"]["participants"].set_index("participant_key")
    assert par.loc["V00_P0020", "metadata_status"] == "undisclosed"
    assert math.isnan(par.loc["V00_P0020", "bfi_extraversion"])
    assert par.loc["V01_P0201", "metadata_status"] == "missing_row"
    ses = run["tables"]["sessions"].set_index("session_key")
    assert ses.loc["V03_S0001", "relationship_status"] == "missing_row"
    assert pd.isna(ses.loc["V03_S0001", "relationship"])
    assert ses.loc["V00_S0001", "relationship"] == "familiar"


def test_conversation_fields_reach_every_clip_under_a_prefix(run, tmp_path) -> None:
    """Both members' clips see the same conversation values, and they are labelled as such."""

    from seamless_curation import annotate
    from seamless_curation.dataset import load_clips

    annotate.write_tables(run["tables"], tmp_path / "annotations", side_files={}, run_hash="test",
                          provenance={})
    clips = load_clips(tmp_path / "annotations", levels=("interaction", "session", "participant"),
                       partner=True)
    pair = clips.loc[clips["interaction_key"] == "V00_S0001_I00000129"]
    assert pair["interaction__posture_pair"].nunique() == 1
    assert pair["session__relationship"].unique().tolist() == ["familiar"]
    assert "posture_pair" not in clips.columns  # never a bare clip column
    first = pair.loc[pair["clip_id"] == "V00_S0001_I00000129_P0010_L30_C000"].iloc[0]
    assert first["partner__participant_key"] == "V00_P0020"
    assert first["partner_participant__metadata_status"] == "undisclosed"
    assert first["participant__metadata_status"] == "numeric"
    # a clip-level value is not propagated to the conversation
    assert "speaking_role" not in run["tables"]["interactions"].columns


def test_speech_past_the_end_of_a_short_wav_is_unknown_not_silence(run) -> None:
    clips = run["tables"]["clips"].set_index("clip_id")
    inside = clips.loc["V00_S0002_I00000130_P0030_L30_C000"]
    beyond = clips.loc["V00_S0002_I00000130_P0030_L30_C001"]  # [30, 35) s, WAV ends at 31 s
    assert inside["own_speech_s"] > 10
    assert pd.isna(beyond["own_speech_s"]) and pd.isna(beyond["arm_active_frac_silence"])
    assert beyond["speaking_role"] == "unknown"
    assert _rec(run, "V00_S0002_I00000130_P0030")["speech_annotated_until_s"] == pytest.approx(31.0, abs=0.05)


def test_a_vad_gap_is_unannotated_not_silence(run) -> None:
    clips = run["tables"]["clips"].set_index("clip_id")
    gap = clips.loc["V01_S0003_I00000139_P0200_L30_C001"]   # VAD 44-46 s only, words 30-40 s
    assert gap["speech_annotation_status"] == "vad_gap"
    assert pd.isna(gap["own_speech_s"]) and gap["speaking_role"] == "unknown"
    assert clips.loc["V01_S0003_I00000139_P0200_L30_C000", "speech_annotation_status"] == "annotated"


def test_the_loader_returns_the_tables_speech_mask(run) -> None:
    from seamless_curation.dataset import load_clip

    clips = run["tables"]["clips"].merge(run["tables"]["recordings"][["file_id", "source_relbase", "fps"]],
                                         on="file_id")
    row = clips.loc[clips["clip_id"] == "V00_S0001_I00000129_P0010_L30_C000"].iloc[0].to_dict()
    clip = load_clip(row, run["root"] / "release", with_audio=False)
    assert abs(sum(b - a for a, b in clip.speech) - row["own_speech_s"]) < 0.1
    gap = clips.loc[clips["clip_id"] == "V01_S0003_I00000139_P0200_L30_C001"].iloc[0].to_dict()
    assert load_clip(gap, run["root"] / "release", with_audio=False).speech is None

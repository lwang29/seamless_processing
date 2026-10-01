"""The metadata catalog, on a miniature release whose answers are known by construction.

The miniature world (13 files, 4 vendors' worth of quirks):

* ``V00_S0001`` (improvised/train): ``P0010`` + ``P0844A`` on prompts 129 and 135.
* ``V00_S0002`` (improvised/dev): ``P0844`` (the bare sibling of ``0844A``) + ``P0020``.
* ``V00_S0003`` (naturalistic/test): ``P0030`` alone - a singleton interaction,
  no relationships row, no participants row.
* ``V00_S0004`` (naturalistic/test): ``P0010`` again (so ``V00_P0010`` spans
  train and test and repeats prompt 129) + ``P0040`` ('Undisclosed' BFI).
* ``V01_S0001`` (naturalistic/train): ``P0010`` (the bare id collides with
  V00's) + ``P0050`` (no V01 row, although V00 has a row for ``0050``); no
  relationships row although ``V00_S0001`` has one; anamorphic rasters.
* ``V03_S0001`` (improvised/train): the room-camera rig and a file with no video.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pytest

from seamless_curation import catalog, schema
from seamless_curation.catalog import (
    build_catalog,
    decode_ipc,
    interaction_type_text_consistency,
    parse_prompt_id,
    participant_components,
    raster_class,
    raster_classes,
    read_metadata_csv,
    task_kind_from_text,
)

GG_TEXT = ("You're going to test your acting skills! You are each given different lists of sentences "
           "below. Read each sentence aloud and act-out a corresponding gesture.")
ROLE_PLAY_TEXT = "Scenario: You are a landlord.\nYour tenant is late with rent.\n\nNegotiate."
STORY_TYPED_DISCUSSION_TEXT = ("What is one word you can use to describe how you and your partner are "
                               "most different? For")
FICTIONAL_TEXT = ("You're going to play a game with your partner. Together you will take turns weaving a "
                  "fictional tale.")

# (file_id, label, split, width, height, fps, has_3p, has_1p, probe_status)
FILES = [
    ("V00_S0001_I00000129_P0010", "improvised", "train", 1080, 1920, 30.0, 1, 0, "ok"),
    ("V00_S0001_I00000129_P0844A", "improvised", "train", 1080, 1920, 30.0, 0, 0, "ok"),
    ("V00_S0001_I00000135_P0010", "improvised", "train", 1080, 1920, 30.0, 0, 0, "ok"),
    ("V00_S0001_I00000135_P0844A", "improvised", "train", 1080, 1920, 30.0, 0, 0, "ok"),
    ("V00_S0002_I00000129_P0020", "improvised", "dev", 1080, 1920, 30.0, 0, 0, "ok"),
    ("V00_S0002_I00000129_P0844", "improvised", "dev", 1080, 1920, 30.0, 0, 0, "ok"),
    ("V00_S0003_I00000130_P0030", "naturalistic", "test", 1080, 1920, 30.0, 0, 1, "ok"),
    ("V00_S0004_I00000129_P0010", "naturalistic", "test", 1080, 1920, 30.0, 0, 0, "ok"),
    ("V00_S0004_I00000129_P0040", "naturalistic", "test", 1080, 1920, 30.0, 0, 0, "ok"),
    ("V01_S0001_I00000139_P0010", "naturalistic", "train", 2160, 2160, 29.97, 0, 0, "ok"),
    ("V01_S0001_I00000139_P0050", "naturalistic", "train", 1012, 1920, 29.97, 0, 0, "ok"),
    ("V03_S0001_I00000130_P0100", "improvised", "train", 3840, 2160, 30.0, 1, 1, "ok"),
    ("V03_S0001_I00000130_P0101", "improvised", "train", None, None, None, 1, 0, "no_video_stream"),
]


def _inventory() -> pd.DataFrame:
    rows = []
    for i, (fid, label, split, w, h, fps, a3, a1, probe) in enumerate(FILES):
        has_video = w is not None
        rows.append({
            "file_id": fid, "label": label, "split": split, "batch_idx": i // 4, "archive_idx": i,
            "has_imitator_movement": int(fid.startswith("V00")), "has_annotation_1p": a1, "has_annotation_3p": a3,
            "source_relbase": f"{label}/{split}/0000/{i:04d}/{fid}",
            "json_present": True, "mp4_present": True, "npz_present": True, "wav_present": True,
            "probe_status": probe,
            "video_width": float(w) if has_video else np.nan, "video_height": float(h) if has_video else np.nan,
            "video_r_fps": fps if has_video else np.nan,
            "video_nb_frames": 3000.0 if has_video else np.nan,
            "video_duration_s": 100.0 if has_video else np.nan,
            "mp4_size_bytes": 25_000_000.0 if has_video else 1_000.0,
            # Census columns the catalog must ignore rather than trust.
            "vendor": "stale", "prompt_hash": "stale",
        })
    return pd.DataFrame(rows)


def _write_metadata(root: Path) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    prompts = pd.DataFrame([
        # grounded gesture: identical A/B text up to whitespace, empty ipc_b -> assignable
        ("00000129", "P0002_v3.4.aac_ANCP_XXXX-148", GG_TEXT, GG_TEXT.replace(" ", "  ", 3), "ANCP", "", "grounded_gesture"),
        ("00000130", "RP2.0_AMCP_ANCP", ROLE_PLAY_TEXT, "Scenario: You are a tenant.", "AMCP", "ANCP", "ipc_conversation"),
        ("00000135", "P0070_v3.2.1.RP_APCM_CGST", STORY_TYPED_DISCUSSION_TEXT, "", "APCM", "CGST", "collaborative_storytelling"),
        ("00000139", "P0100_v3.1.fam_ANCP_XXXX", FICTIONAL_TEXT, "Listen to the tale.", "ANCP", "", "ipc_conversation"),
        # unused prompts
        ("00000200", "P0064_AA_ANCP4134", "Play charades: no spoken words allowed.", "Guess.", "CGST", "", "charades"),
        ("00000201", "P0003_v3.0aac_ANCP_XXXX-7", GG_TEXT, "Other list.", "ANCP", "", "grounded_gesture"),
        ("00000202", "P0005_v3.3.str_AMCM_XXXX", "You will each see the same prompt:", "You will each see the same prompt:",
         "AMCM", "", "ipc_conversation"),
    ], columns=["prompt_hash", "prompt_id_unique", "participant_a_prompt_text", "participant_b_prompt_text",
                "ipc_a", "ipc_b", "interaction_type"])
    prompts.to_csv(root / "interactions.csv", index=True)  # writes Meta's unnamed index column

    bfi = ["extraversion_raw", "agreeableness_raw", "conscientiousness_raw", "neuroticism_raw", "openness_raw"]
    numeric = ["3.5", "4.0", "2.25", "1.5", "4.75"]
    undisclosed = ["Undisclosed"] * 5
    people = [
        (numeric, "00", "0010"), (undisclosed, "00", "0844A"), (numeric, "00", "0844"), (numeric, "00", "0020"),
        (undisclosed, "00", "0040"), (numeric, "00", "0050"), (["1.0", "2.0", "3.0", "4.0", "5.0"], "01", "0010"),
        (numeric, "03", "0100"), (numeric, "03", "0101"),
        (numeric, "00", "0010"),  # an exact duplicate row: dropped, not a conflict
    ]
    pd.DataFrame([dict(zip(bfi, s), vendor_id=v, participant_id=p) for s, v, p in people]).to_csv(
        root / "participants.csv", index=False)

    pd.DataFrame([
        ("V00", "0001", "stranger", "stranger"),
        ("V00", "0002", "familiar", "friends"),
        ("V00", "0004", "familiar", "coworkers"),
        ("V03", "0001", "stranger", "stranger"),
        ("V01", "0002", "familiar", "siblings"),  # a session without files
    ], columns=["vendor_id", "session_id", "relationship", "relationship_detail"]).to_csv(
        root / "relationships.csv", index=False)
    return root


@pytest.fixture(scope="module")
def mini(tmp_path_factory) -> catalog.Catalog:
    root = _write_metadata(tmp_path_factory.mktemp("meta"))
    return build_catalog(_inventory(), root)


def _row(frame: pd.DataFrame, key: str, value: str) -> pd.Series:
    match = frame[frame[key] == value]
    assert len(match) == 1, (key, value, len(match))
    return match.iloc[0]


# -----------------------------------------------------------------------------
# Scalar helpers
# -----------------------------------------------------------------------------


@pytest.mark.parametrize("prompt_id, expected", [
    ("P0002_v3.4.aac_ANCP_XXXX-148", ("P0002", "v3.4", "aac", 148)),
    ("P0003_v3.0aac_ANCP_XXXX-7", ("P0003", "v3.0", "aac", 7)),  # the missing dot
    ("P0070_v3.2.1.RP_APCM_CGST", ("P0070", "v3.2.1", "RP", None)),
    ("RP2.0_AMCP_ANCP", (None, None, "RP", None)),  # x.y is a scenario number, not a version
    ("P0100_v3.1.fam_ANCP_XXXX", ("P0100", "v3.1", "fam", None)),
    ("P0101_v3.9.str_CGST_XXXX", ("P0101", "v3.9", "str", None)),
    ("P0040_v3.0.5.AC_AMCM_XXXX-12", ("P0040", "v3.0.5", "AC", 12)),
    ("P0064_AA_ANCP4134", ("P0064", None, "AA", None)),  # IPC code glued to a number
    ("P0065_DN_AMCP_XXXX-3", ("P0065", None, "DN", 3)),
    ("P0066_ZZ_AMCP_XXXX", ("P0066", None, "other", None)),
    ("garbage", (None, None, "other", None)),
])
def test_parse_prompt_id_families(prompt_id: str, expected: tuple) -> None:
    parsed = parse_prompt_id(prompt_id)
    assert (parsed["template"], parsed["version"], parsed["context"], parsed["item_index"]) == expected


def test_decode_ipc_octants_cognitive_state_and_absent() -> None:
    assert decode_ipc("APCM") == (1, -1, "octant")
    assert decode_ipc("ANCP") == (0, 1, "octant")
    assert decode_ipc("AMCN") == (-1, 0, "octant")
    assert decode_ipc("CGST") == (None, None, "cognitive_state")
    assert decode_ipc("") == (None, None, "absent")
    assert decode_ipc(None) == (None, None, "absent")
    assert decode_ipc(float("nan")) == (None, None, "absent")
    with pytest.raises(ValueError):
        decode_ipc("AXCP")


def test_every_registered_ipc_code_decodes() -> None:
    for code in schema.IPC_CODES:
        agency, communion, kind = decode_ipc(code)
        assert kind == ("cognitive_state" if code == "CGST" else "octant")
        assert (agency is None) == (code == "CGST")


@pytest.mark.parametrize("text, kind", [
    (GG_TEXT, "gesture_sentence"),
    (FICTIONAL_TEXT, "storytelling"),  # opens "play a game" but is fiction: order matters
    ("You're going to play a game with your partner. It is intended to be silly and fun.", "game"),
    ("You are each given different lists of words or phrases below.", "game"),
    (ROLE_PLAY_TEXT, "role_play"),
    ("(Disorganized): As the employee, you forgot the deadline.", "role_play"),
    ("Tell your partner about a time when you felt proud.", "discussion"),
    ("Tell a story about a time you got lost.", "discussion"),  # personal story is not storytelling
    ("Would you rather fly or be invisible", "discussion"),
    (STORY_TYPED_DISCUSSION_TEXT, "discussion"),
    ("You will each see the same prompt:", "unclear"),
    ("", "unclear"),
    (None, "unclear"),
])
def test_task_kind_rules(text, kind: str) -> None:
    assert task_kind_from_text(text) == kind


def test_type_text_consistency() -> None:
    assert interaction_type_text_consistency("ipc_conversation", "discussion") == "consistent"
    assert interaction_type_text_consistency("ipc_conversation", "role_play") == "consistent"
    assert interaction_type_text_consistency("ipc_conversation", "storytelling") == "contradicts_text"
    assert interaction_type_text_consistency("collaborative_storytelling", "storytelling") == "consistent"
    assert interaction_type_text_consistency("collaborative_storytelling", "discussion") == "contradicts_text"
    assert interaction_type_text_consistency("charades", "game") == "consistent"
    assert interaction_type_text_consistency("grounded_gesture", "gesture_sentence") == "consistent"
    assert interaction_type_text_consistency("grounded_gesture", "unclear") == "unclear"
    assert interaction_type_text_consistency(None, "discussion") == "unclear"


def test_raster_class_scalar_and_vectorised_agree() -> None:
    cases = [
        ((1080, 1920), "portrait_1080x1920"), ((2160, 3840), "portrait_2160x3840"),
        ((1012, 1920), "anamorphic_1012x1920"), ((2160, 2160), "anamorphic_2160x2160"),
        ((1920, 1080), "anamorphic_1920x1080"), ((3840, 2160), "rotated_3840x2160"),
        ((640, 480), "rotated_640x480"), ((1080, 960), "pillarbox_1080x960"),
        ((2180, 3840), "pillarbox_2180x3840"), ((720, 1280), "other"), ((1080.5, 1920), "other"),
        ((None, None), "no_video"), ((np.nan, 1920), "no_video"), (("1080", "1920"), "portrait_1080x1920"),
    ]
    for (w, h), expected in cases:
        assert raster_class(w, h) == expected, (w, h)
    widths = [pd.to_numeric(w, errors="coerce") if w is not None else np.nan for (w, _), _ in cases]
    heights = [pd.to_numeric(h, errors="coerce") if h is not None else np.nan for (_, h), _ in cases]
    assert raster_classes(widths, heights).tolist() == [e for _, e in cases]
    assert set(catalog.RASTER_CLASS_BY_SIZE.values()) | {"no_video", "other"} == set(schema.RASTER_CLASSES)


def test_participant_components_are_named_by_their_smallest_key() -> None:
    comps = participant_components(
        [("V00_P0030", "V00_P0020"), ("V00_P0020", "V00_P0010"), ("V01_P0002", "V01_P0001")],
        ["V00_P0010", "V00_P0020", "V00_P0030", "V00_P0040", "V01_P0001", "V01_P0002"],
    )
    assert comps == {
        "V00_P0010": "V00_P0010", "V00_P0020": "V00_P0010", "V00_P0030": "V00_P0010",
        "V00_P0040": "V00_P0040", "V01_P0001": "V01_P0001", "V01_P0002": "V01_P0001",
    }


def test_metadata_csv_key_conflict_raises_and_exact_duplicates_drop(tmp_path: Path) -> None:
    path = tmp_path / "participants.csv"
    header = "extraversion_raw,agreeableness_raw,conscientiousness_raw,neuroticism_raw,openness_raw,vendor_id,participant_id\n"
    path.write_text(header + "1,2,3,4,5,00,0010\n1,2,3,4,5,00,0010\n")
    frame = read_metadata_csv(path, catalog.PARTICIPANTS_CSV_COLUMNS, key=("vendor_id", "participant_id"))
    assert len(frame) == 1 and frame["participant_id"].iloc[0] == "0010"
    path.write_text(header + "1,2,3,4,5,00,0010\n1,2,3,4,4,00,0010\n")
    with pytest.raises(ValueError, match="share a key"):
        read_metadata_csv(path, catalog.PARTICIPANTS_CSV_COLUMNS, key=("vendor_id", "participant_id"))


# -----------------------------------------------------------------------------
# The miniature release
# -----------------------------------------------------------------------------


def test_table_sizes_and_registry_conformance(mini: catalog.Catalog) -> None:
    assert len(mini.recordings) == 13
    assert len(mini.interactions) == 7
    assert len(mini.sessions) == 6
    assert len(mini.participants) == 10
    assert len(mini.prompts) == 7  # includes the three prompts no file uses
    for table, frame in mini.tables().items():
        registered = schema.columns(table)
        assert list(frame.columns) == [c for c in registered if c in set(frame.columns)], table
        # The frame converts to Arrow under the registry's own types and nullability.
        sub = pa.schema([schema.arrow_schema(table).field(c) for c in frame.columns])
        pa.Table.from_pandas(frame, schema=sub, preserve_index=False)
        for column in frame.columns:
            spec = schema.field(table, column)
            if spec.allowed and frame[column].notna().any():
                assert set(frame[column].dropna()) <= set(spec.allowed), (table, column)
            if not spec.nullable:
                assert not frame[column].isna().any(), (table, column)
    assert set(mini.participants.columns) == set(schema.columns("participants")) - {"participant_recorded_hours"}
    assert set(mini.prompts.columns) == set(schema.columns("prompts"))


def test_leading_zeros_and_suffixes_are_preserved(mini: catalog.Catalog) -> None:
    rec = _row(mini.recordings, "file_id", "V00_S0001_I00000129_P0844A")
    assert (rec["session_id"], rec["participant_id"], rec["prompt_hash"]) == ("0001", "0844A", "00000129")
    assert (rec["vendor"], rec["participant_key"], rec["interaction_key"]) == (
        "V00", "V00_P0844A", "V00_S0001_I00000129")
    assert "00000129" in set(mini.prompts["prompt_hash"])
    assert _row(mini.participants, "participant_key", "V00_P0844A")["id_suffix"] == "A"
    assert _row(mini.participants, "participant_key", "V00_P0010")["id_suffix"] == "none"


def test_singleton_interaction(mini: catalog.Catalog) -> None:
    inter = _row(mini.interactions, "interaction_key", "V00_S0003_I00000130")
    assert inter["pairing_status"] == "partner_missing"
    assert inter["interaction_n_recordings"] == 1
    assert inter["member_file_id_1"] == "V00_S0003_I00000130_P0030"
    assert pd.isna(inter["member_file_id_2"])
    # One member carries 1P: a singleton is at most 'one', never 'both'.
    assert inter["moi_1p_coverage"] == "one" and inter["moi_3p_coverage"] == "none"
    rec = _row(mini.recordings, "file_id", "V00_S0003_I00000130_P0030")
    assert pd.isna(rec["partner_file_id"]) and pd.isna(rec["partner_participant_key"])
    ses = _row(mini.sessions, "session_key", "V00_S0003")
    assert ses["session_n_participants"] == 1 and pd.isna(ses["participant_key_2"])
    assert ses["dyad_key"] == "V00_P0030" and ses["dyad_session_count"] == 1


def test_partner_links_and_member_order(mini: catalog.Catalog) -> None:
    a = _row(mini.recordings, "file_id", "V00_S0001_I00000129_P0010")
    b = _row(mini.recordings, "file_id", "V00_S0001_I00000129_P0844A")
    assert a["partner_file_id"] == b["file_id"] and b["partner_file_id"] == a["file_id"]
    assert a["partner_participant_key"] == "V00_P0844A" and b["partner_participant_key"] == "V00_P0010"
    inter = _row(mini.interactions, "interaction_key", "V00_S0001_I00000129")
    assert (inter["member_file_id_1"], inter["member_file_id_2"]) == (a["file_id"], b["file_id"])
    assert inter["pairing_status"] == "paired"
    assert inter["moi_3p_coverage"] == "one"
    assert _row(mini.interactions, "interaction_key", "V03_S0001_I00000130")["moi_3p_coverage"] == "both"


def test_participant_in_two_splits(mini: catalog.Catalog) -> None:
    people = mini.participants.set_index("participant_key")
    assert people.loc["V00_P0010", ["in_train", "in_dev", "in_test"]].tolist() == [True, False, True]
    assert people.loc["V00_P0010", ["in_improvised", "in_naturalistic"]].tolist() == [True, True]
    assert people["flag_split_conflict"][people["flag_split_conflict"]].index.tolist() == ["V00_P0010"]
    inter = mini.interactions.set_index("interaction_key")["n_members_in_other_split"]
    assert inter["V00_S0001_I00000129"] == 1  # P0010 also has test files
    assert inter["V00_S0004_I00000129"] == 1  # P0010 also has train files
    assert inter["V00_S0002_I00000129"] == 0
    assert inter["V01_S0001_I00000139"] == 0  # V01_P0010 is a different person from V00_P0010


def test_suffix_siblings_in_different_splits(mini: catalog.Catalog) -> None:
    people = mini.participants.set_index("participant_key")
    assert people.loc["V00_P0844A", "suffix_sibling_key"] == "V00_P0844"
    assert people.loc["V00_P0844", "suffix_sibling_key"] == "V00_P0844A"
    assert people["suffix_sibling_key"].notna().sum() == 2
    assert sorted(people.index[people["flag_suffix_sibling_split_conflict"]]) == ["V00_P0844", "V00_P0844A"]
    # Siblings are linked, not merged: each keeps its own split flags and component.
    assert not people.loc["V00_P0844A", "flag_split_conflict"]
    assert people.loc["V00_P0844A", "participant_component_id"] != people.loc["V00_P0844", "participant_component_id"]
    sib = mini.interactions.set_index("interaction_key")["n_members_suffix_sibling_other_split"]
    assert sib["V00_S0001_I00000129"] == 1 and sib["V00_S0002_I00000129"] == 1
    assert sib["V00_S0004_I00000129"] == 0


def test_bfi_undisclosed_missing_row_and_vendor_prefix(mini: catalog.Catalog) -> None:
    people = mini.participants.set_index("participant_key")
    assert people.loc["V00_P0010", "metadata_status"] == "numeric"
    assert people.loc["V00_P0010", "bfi_extraversion"] == pytest.approx(3.5)
    assert people.loc["V00_P0010", "bfi_openness"] == pytest.approx(4.75)
    # participants.csv writes vendor '01'; the key is still V01 and the scores are V01's own.
    assert people.loc["V01_P0010", "metadata_status"] == "numeric"
    assert people.loc["V01_P0010", "bfi_extraversion"] == pytest.approx(1.0)
    assert people.loc["V00_P0040", "metadata_status"] == "undisclosed"
    assert people.loc["V00_P0030", "metadata_status"] == "missing_row"
    # A V00 row exists for 0050, but it is never borrowed for V01_P0050.
    assert people.loc["V01_P0050", "metadata_status"] == "missing_row"
    for key in ("V00_P0040", "V00_P0030", "V01_P0050"):
        for domain in catalog.BFI_DOMAINS:
            assert math.isnan(people.loc[key, f"bfi_{domain}"])
    assert "V00_P0050" not in people.index  # metadata rows without files are not participants


def test_relationship_rows_and_session_fields(mini: catalog.Catalog) -> None:
    ses = mini.sessions.set_index("session_key")
    assert ses.loc["V00_S0001", ["relationship", "relationship_detail", "relationship_status"]].tolist() == [
        "stranger", "stranger", "present"]
    assert ses.loc["V00_S0004", "relationship_detail"] == "coworkers"
    # V00_S0001 has a row, but session ids collide across vendors: V01_S0001 does not.
    for key in ("V01_S0001", "V00_S0003"):
        assert ses.loc[key, "relationship_status"] == "missing_row"
        assert pd.isna(ses.loc[key, "relationship"]) and pd.isna(ses.loc[key, "relationship_detail"])
    assert "V01_S0002" not in ses.index
    assert ses.loc["V00_S0001", "session_n_interactions"] == 2
    assert ses.loc["V00_S0001", "session_n_interaction_types"] == 2
    assert (ses.loc["V00_S0001", "participant_key_1"], ses.loc["V00_S0001", "participant_key_2"]) == (
        "V00_P0010", "V00_P0844A")
    assert ses.loc["V00_S0001", "dyad_key"] == "V00_P0010+V00_P0844A"
    assert (ses.loc["V00_S0001", "label"], ses.loc["V00_S0001", "split"]) == ("improvised", "train")


def test_component_ids(mini: catalog.Catalog) -> None:
    comp = mini.participants.set_index("participant_key")["participant_component_id"]
    assert comp["V00_P0844A"] == comp["V00_P0040"] == comp["V00_P0010"] == "V00_P0010"
    assert comp["V00_P0844"] == comp["V00_P0020"] == "V00_P0020"
    assert comp["V00_P0030"] == "V00_P0030"
    assert comp["V01_P0050"] == "V01_P0010"
    assert comp["V03_P0101"] == "V03_P0100"


def test_prompt_repeat_count(mini: catalog.Catalog) -> None:
    rec = mini.recordings.set_index("file_id")["prompt_repeat_count_for_participant"]
    assert rec["V00_S0001_I00000129_P0010"] == 1 and rec["V00_S0004_I00000129_P0010"] == 1
    assert rec["V00_S0001_I00000135_P0010"] == 0  # same participant, a prompt seen once
    assert rec["V00_S0002_I00000129_P0844"] == 0  # same prompt, other people
    assert int(rec.sum()) == 2


def test_prompt_columns(mini: catalog.Catalog) -> None:
    pr = mini.prompts.set_index("prompt_hash")
    gg = pr.loc["00000129"]
    assert (gg["prompt_template"], gg["prompt_version"], gg["prompt_context"], gg["prompt_item_index"]) == (
        "P0002", "v3.4", "aac", 148)
    assert not gg["ab_prompts_differ"] and gg["ipc_member_assignable"]  # whitespace-only difference
    assert pd.isna(gg["ipc_b"]) and gg["ipc_b_kind"] == "absent" and pd.isna(gg["ipc_b_agency"])
    assert (gg["ipc_a_agency"], gg["ipc_a_communion"], gg["ipc_a_kind"]) == (0, 1, "octant")
    rp = pr.loc["00000130"]
    assert rp["task_kind_from_text"] == "role_play" and rp["interaction_type_text_consistency"] == "consistent"
    assert "\n" in rp["participant_a_prompt_text"]  # quoted multi-line field survives
    assert rp["ab_prompts_differ"] and not rp["ipc_member_assignable"]
    assert pd.isna(rp["prompt_template"]) and pd.isna(rp["prompt_version"]) and rp["prompt_context"] == "RP"
    cg = pr.loc["00000135"]
    assert cg["ipc_b"] == "CGST" and cg["ipc_b_kind"] == "cognitive_state" and pd.isna(cg["ipc_b_communion"])
    assert pd.isna(cg["participant_b_prompt_text"])
    assert cg["interaction_type_text_consistency"] == "contradicts_text"
    assert pr.loc["00000139", "interaction_type_text_consistency"] == "contradicts_text"
    assert pr.loc["00000202", "interaction_type_text_consistency"] == "unclear"
    assert pr.loc["00000202", "ipc_member_assignable"]  # same text both roles, no ipc_b
    assert not pr.loc["00000139", "ipc_member_assignable"]  # different texts
    assert pr.loc["00000200", "prompt_context"] == "AA"
    assert pr.loc["00000200", "task_kind_from_text"] == "game"
    assert pr.loc["00000129", "prompt_n_sessions"] == 3 and pr.loc["00000129", "prompt_n_vendors"] == 1
    assert pr.loc["00000130", "prompt_n_sessions"] == 2 and pr.loc["00000130", "prompt_n_vendors"] == 2
    assert pr.loc["00000200", "prompt_n_sessions"] == 0 and pr.loc["00000200", "prompt_n_vendors"] == 0


def test_recording_media_columns(mini: catalog.Catalog) -> None:
    rec = mini.recordings.set_index("file_id")
    v01 = rec.loc["V01_S0001_I00000139_P0010"]
    assert (v01["raster_class"], v01["smplh_anamorphic"], v01["room_camera_rig"]) == (
        "anamorphic_2160x2160", "severe", False)
    assert rec.loc["V01_S0001_I00000139_P0050", "smplh_anamorphic"] == "mild"
    room = rec.loc["V03_S0001_I00000130_P0100"]
    assert room["raster_class"] == "rotated_3840x2160" and room["room_camera_rig"]
    assert room["video_bitrate_mbps"] == pytest.approx(25_000_000 * 8 / 1e6 / 100.0)
    none = rec.loc["V03_S0001_I00000130_P0101"]
    assert none["raster_class"] == "no_video" and not none["room_camera_rig"]
    assert none["smplh_anamorphic"] == "none"
    assert math.isnan(none["fps"]) and math.isnan(none["video_bitrate_mbps"])
    assert pd.isna(none["video_width"]) and pd.isna(none["video_nb_frames"])
    assert rec.loc["V00_S0001_I00000129_P0010", "fps"] == 30.0
    assert rec["has_imitator_movement"].sum() == 9 and rec["has_annotation_3p"].sum() == 3
    assert rec.loc["V00_S0001_I00000129_P0010", "video_width"] == 1080


def test_fraction_frame_rate_text_is_parsed(tmp_path: Path) -> None:
    inventory = _inventory()
    inventory["video_r_fps"] = inventory["video_r_fps"].astype(object)
    inventory.loc[0, "video_r_fps"] = "30000/1001"
    built = build_catalog(inventory, _write_metadata(tmp_path / "meta"))
    fps = built.recordings.set_index("file_id")["fps"]
    assert fps["V00_S0001_I00000129_P0010"] == pytest.approx(30000 / 1001)


def test_more_than_two_members_is_refused(tmp_path: Path) -> None:
    inventory = _inventory()
    extra = inventory.iloc[[0]].copy()
    extra["file_id"] = "V00_S0001_I00000129_P0099"
    with pytest.raises(ValueError, match="more than two"):
        build_catalog(pd.concat([inventory, extra], ignore_index=True), _write_metadata(tmp_path / "meta"))


def test_split_must_be_constant_within_a_session(tmp_path: Path) -> None:
    inventory = _inventory()
    inventory.loc[inventory["file_id"] == "V00_S0001_I00000135_P0010", "split"] = "dev"
    with pytest.raises(ValueError, match="not constant"):
        build_catalog(inventory, _write_metadata(tmp_path / "meta"))


# -----------------------------------------------------------------------------
# The real release (skipped where the census is not staged)
# -----------------------------------------------------------------------------

REPO = Path(__file__).resolve().parents[1]
CENSUS = REPO / "outputs" / "02_inventory" / "summary" / "inventory_joined.parquet"
METADATA = REPO / "datasets" / "seamless_interaction_metadata"


@pytest.mark.skipif(not (CENSUS.exists() and (METADATA / "interactions.csv").exists()),
                    reason="M-1 census or Meta metadata not staged")
def test_real_release_counts() -> None:
    built = build_catalog(pd.read_parquet(CENSUS), METADATA)
    counts = catalog.summary(built)
    assert counts["recordings"] == 129_370
    assert counts["interactions"] == 64_751
    assert counts["partner_missing"] == 132
    assert counts["sessions"] == 5_102
    assert counts["participants"] == 4_307
    assert counts["split_conflicts"] == 26
    assert counts["prompts"] == 1_312
    assert {"00000135", "00000129", "00000139", "00000130"} <= set(counts["prompts_contradicting_text"])
    people = built.participants
    assert int(people["suffix_sibling_key"].notna().sum()) == 102  # 51 stems
    assert people["metadata_status"].value_counts().to_dict() == {
        "numeric": 2_280, "undisclosed": 1_860, "missing_row": 167}
    assert int((built.sessions["relationship_status"] == "missing_row").sum()) == 122
    assert int(built.prompts["prompt_item_index"].notna().sum()) == 322
    assert int((built.prompts["prompt_context"] == "other").sum()) == 0
    assert int(built.recordings["video_bitrate_mbps"].isna().sum()) == 623

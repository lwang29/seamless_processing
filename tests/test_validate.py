"""The validator catches each way the tables can silently break.

Starting from the miniature release's valid tables, inject one fault at a time
and assert the check that owns it fails — and that the untouched tables pass.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from seamless_curation.validate import validate_tables
from tests.builders_pipeline import run_pipeline


@pytest.fixture(scope="module")
def good(tmp_path_factory, model_root):
    tables, report, cat = run_pipeline(tmp_path_factory.mktemp("validate_release"), model_root)
    assert report.ok, report.errors
    return tables, pd.Index(cat.recordings["file_id"])


def _failed(tables, ids, name_part: str) -> list[str]:
    report = validate_tables(tables, clip_seconds=30.0, catalog_file_ids=ids)
    return [c["check"] for c in report.errors if name_part in c["check"]]


def _copy(tables):
    return {name: frame.copy() for name, frame in tables.items()}


def test_duplicate_clip_ids(good) -> None:
    tables, ids = _copy(good[0]), good[1]
    tables["clips"] = pd.concat([tables["clips"], tables["clips"].iloc[:1]], ignore_index=True)
    assert _failed(tables, ids, "clips.clip_id: unique")


def test_a_gap_in_the_tiling(good) -> None:
    tables, ids = _copy(good[0]), good[1]
    clips = tables["clips"]
    row = clips.index[(clips["clip_index"] == 1)][0]
    clips.loc[row, "start_frame"] += 5
    assert _failed(tables, ids, "contiguous")


def test_an_asymmetric_partner_link(good) -> None:
    tables, ids = _copy(good[0]), good[1]
    clips = tables["clips"]
    row = clips.index[clips["partner_clip_id"].notna()][0]
    other = clips.index[clips["partner_clip_id"].notna()][-1]
    clips.loc[row, "partner_clip_id"] = clips.loc[other, "clip_id"]
    assert _failed(tables, ids, "partner clip")


def test_an_enum_value_outside_the_registry(good) -> None:
    tables, ids = _copy(good[0]), good[1]
    tables["clips"].loc[tables["clips"].index[0], "posture"] = "lying"
    assert _failed(tables, ids, "clips.posture: allowed values")


def test_a_zero_that_should_be_missing(good) -> None:
    """An unannotated recording's MOI count must be NA, never 0."""

    tables, ids = _copy(good[0]), good[1]
    clips = tables["clips"]
    row = clips.index[clips["moi_3p_count"].isna()][0]
    clips.loc[row, "moi_3p_count"] = 0
    assert _failed(tables, ids, "moi_3p_count NA iff")


def test_a_partition_that_disagrees_with_its_parent(good) -> None:
    tables, ids = _copy(good[0]), good[1]
    tables["clips"].loc[tables["clips"].index[0], "split"] = "test" if tables["clips"]["split"].iloc[0] != "test" else "dev"
    assert _failed(tables, ids, "clips vs recordings: split")


def test_a_dangling_link(good) -> None:
    tables, ids = _copy(good[0]), good[1]
    tables["clips"].loc[tables["clips"].index[0], "session_key"] = "V09_S9999"
    assert _failed(tables, ids, "clips.session_key -> sessions")


def test_a_missing_catalog_recording(good) -> None:
    tables, ids = _copy(good[0]), good[1]
    tables["recordings"] = tables["recordings"].iloc[1:]
    assert _failed(tables, ids, "every catalog recording present")


def test_an_extra_column(good) -> None:
    tables, ids = _copy(good[0]), good[1]
    tables["clips"]["posture_x"] = 1
    assert _failed(tables, ids, "clips: columns match the registry")


def test_the_untouched_tables_pass(good) -> None:
    tables, ids = good
    report = validate_tables(tables, clip_seconds=30.0, catalog_file_ids=ids)
    assert report.ok and report.as_dict()["n_checks"] > 100

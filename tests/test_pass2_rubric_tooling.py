"""Tests for the Pass-2 rating mechanism and the ratings/signal correlation.

The rubric content itself is not defined here or anywhere else in the
repository: these tests use a placeholder fixture so the mechanism can be proven
without pre-empting the failure-mode vocabulary that Pass 1 exists to produce.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest
import yaml

from seamless_curation.review_renderer import build_gallery_html, normalize_rubric
from seamless_curation.review_ratings import (
    correlate_ratings_with_signals,
    duplicate_agreement,
    load_rating_export,
)


RECORDS = [
    {
        "review_item_id": f"item_{index:03d}",
        "clip_id": f"clip_{index:03d}",
        "file_id": f"V00_S0001_I0000000{index}_P0001",
        "sample_group": "uniform_200",
        "status": "rendered",
        "media_file": f"clip_{index:03d}.mp4",
    }
    for index in range(4)
]


@pytest.fixture()
def rubric(repo_root: Path) -> dict:
    return yaml.safe_load((repo_root / "tests/fixtures/rubric_test_only.yaml").read_text())


def test_pass1_gallery_offers_no_categories() -> None:
    page = build_gallery_html(RECORDS, "Pass 1")
    assert "exploratory_free_text_v1" in page
    # No rating controls and no rating machinery: the only mention of a rubric
    # is the sentence telling the reviewer there deliberately is not one.
    assert "rating" not in page.lower()
    assert '<fieldset class="rubric">' not in page
    assert "rubric-item" not in page
    assert 'type="radio"' not in page
    assert "structured_ratings_v1" not in page


def test_pass2_gallery_renders_every_scale_and_the_overall_question(rubric: dict) -> None:
    page = build_gallery_html(RECORDS, "Pass 2", rubric)
    assert "structured_ratings_v1" in page
    for item in rubric["items"]:
        assert f'data-item-id="{item["id"]}"' in page
        for value in range(item["min"], item["max"] + 1):
            assert f'name="{item["id"]}" value="{value}"' in page
    assert 'data-item-id="would_train"' in page
    assert 'name="would_train" value="yes"' in page
    assert 'name="would_train" value="no"' in page
    # Every card carries the full rubric, and the exported schema records which
    # rubric produced the numbers.
    assert page.count('<fieldset class="rubric">') == len(RECORDS)
    assert "rubric_hash" in page


def test_unapproved_or_underived_rubric_is_refused(rubric: dict) -> None:
    with pytest.raises(ValueError, match="status: approved"):
        normalize_rubric({**rubric, "status": "draft"})
    with pytest.raises(ValueError, match="derived_from"):
        normalize_rubric({**rubric, "derived_from": "NOT_YET_DERIVED"})
    with pytest.raises(ValueError, match="no items"):
        normalize_rubric({**rubric, "items": []})
    with pytest.raises(ValueError, match="snake_case"):
        normalize_rubric({**rubric, "items": [{"id": "Bad Id", "label": "x"}]})
    with pytest.raises(ValueError, match="duplicate"):
        normalize_rubric({**rubric, "items": [{"id": "a"}, {"id": "a"}]})
    with pytest.raises(ValueError, match="min < max"):
        normalize_rubric({**rubric, "items": [{"id": "a", "min": 5, "max": 5}]})


def _export(tmp_path: Path, entries: list[dict], rubric_hash: str = "abc123") -> Path:
    path = tmp_path / "export.json"
    path.write_text(
        json.dumps(
            {
                "schema": "structured_ratings_v1",
                "manifest_hash": "m",
                "reviewer_id": "reviewer_a",
                "rubric_hash": rubric_hash,
                "rubric_derived_from": "pass1",
                "entries": entries,
            }
        ),
        encoding="utf-8",
    )
    return path


def test_load_rating_export_flattens_scales_and_the_binary(tmp_path: Path) -> None:
    path = _export(
        tmp_path,
        [
            {
                "review_item_id": "item_000",
                "free_text": "note",
                "ratings": {"placeholder_dimension_a": "4", "would_train": "yes"},
                "complete": False,
            },
            {
                "review_item_id": "item_001",
                "free_text": "",
                "ratings": {"placeholder_dimension_a": "1", "would_train": "no"},
                "complete": False,
            },
        ],
    )
    frame = load_rating_export(path)
    assert list(frame.review_item_id) == ["item_000", "item_001"]
    assert list(frame.placeholder_dimension_a) == [4.0, 1.0]
    assert list(frame.would_train) == [True, False]
    assert frame.reviewer_id.eq("reviewer_a").all()


def test_correlation_reports_detected_and_undetected_items() -> None:
    # Signal `tracks_a` is built to follow item a exactly; `unrelated` follows
    # nothing. Item b has no signal that moves with it at all, which is the
    # negative result the session is meant to be able to state.
    ratings = pd.DataFrame(
        {
            "review_item_id": [f"item_{i:03d}" for i in range(12)],
            "placeholder_dimension_a": [1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 3, 2],
            "placeholder_dimension_b": [5, 1, 4, 2, 3, 3, 2, 4, 1, 5, 2, 4],
            "would_train": [False] * 6 + [True] * 6,
        }
    )
    signals = pd.DataFrame(
        {
            "review_item_id": ratings.review_item_id,
            "tracks_a": [0.1, 0.11, 0.2, 0.21, 0.3, 0.31, 0.4, 0.41, 0.5, 0.51, 0.3, 0.2],
            "unrelated": [0.5, 0.2, 0.9, 0.1, 0.4, 0.7, 0.3, 0.8, 0.6, 0.05, 0.55, 0.45],
        }
    )
    result = correlate_ratings_with_signals(
        ratings, signals, rating_items=["placeholder_dimension_a", "placeholder_dimension_b"]
    )
    table = result["correlations"]
    best_a = table[table.rating_item.eq("placeholder_dimension_a")].nlargest(1, "abs_spearman")
    assert best_a.signal.iloc[0] == "tracks_a"
    assert best_a.abs_spearman.iloc[0] > 0.95
    best_b = table[table.rating_item.eq("placeholder_dimension_b")].abs_spearman.max()
    assert best_b < 0.6
    assert "placeholder_dimension_b" in result["items_without_a_separating_signal"]
    assert "placeholder_dimension_a" not in result["items_without_a_separating_signal"]
    # The binary question is scored by group separation, not Spearman alone.
    binary = result["would_train_separation"]
    assert set(binary.signal) == {"tracks_a", "unrelated"}
    assert binary.n_would_train.iloc[0] == 6


def test_duplicate_agreement_measures_repeat_consistency() -> None:
    ratings = pd.DataFrame(
        {
            "review_item_id": ["item_000", "dup_000", "item_001", "dup_001"],
            "placeholder_dimension_a": [4, 4, 2, 5],
            "would_train": [True, True, False, True],
        }
    )
    agreement = duplicate_agreement(
        ratings,
        duplicate_map={"dup_000": "item_000", "dup_001": "item_001"},
        rating_items=["placeholder_dimension_a"],
    )
    scales = agreement["scales"]
    assert scales.loc[scales.rating_item.eq("placeholder_dimension_a"), "pairs"].iloc[0] == 2
    assert scales.loc[scales.rating_item.eq("placeholder_dimension_a"), "exact_agreement"].iloc[0] == 0.5
    assert scales.loc[scales.rating_item.eq("placeholder_dimension_a"), "mean_abs_difference"].iloc[0] == 1.5
    assert agreement["would_train"]["pairs"] == 2
    assert agreement["would_train"]["agreement"] == 0.5

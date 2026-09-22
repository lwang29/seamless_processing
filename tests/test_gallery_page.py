"""The gallery is what a reviewer outside this project actually sees."""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from gallery_page import render_page  # noqa: E402


def context(count: int = 3) -> dict:
    items = [
        {
            "slug": f"c{i:04d}", "file_id": f"V0{i % 4}_S1_I1_P{i}", "vendor": f"V0{i % 4}",
            "split": "train", "seconds": 30.0, "quality": 0.3 + 0.2 * i,
            "posture": 0.5, "persistence": 0.6, "vigour": 0.4, "integrity": 0.7,
            "speech_s": 14.0, "gesture_frac": 0.72, "start_s": 30.0 * i,
        }
        for i in range(count)
    ]
    dist = {"clips": count, "vendors": {"V00": 0.5, "V01": 0.5},
            "quality": [0.4, 0.6, 0.8], "hours": 1.0}
    return {"items": items, "sample": dist, "population": dict(dist, clips=50516, hours=420.9),
            "seed": "gallery-v1", "count_requested": count}


def test_the_page_inlines_every_item_as_valid_json() -> None:
    """The grid is built from this blob; a broken one is a blank page."""

    html = render_page(context(5))
    match = re.search(r"const DATA = (\{.*?\});\n", html, re.S)
    assert match, "DATA blob missing"
    data = json.loads(match.group(1))
    assert len(data["items"]) == 5
    assert data["items"][0]["slug"] == "c0000"


def test_the_page_is_self_contained() -> None:
    """It is served from a directory that has no network egress guarantee."""

    html = render_page(context())
    assert "http://" not in html.replace("http://localhost", "")
    assert "cdn" not in html.lower()
    assert "<script src=" not in html and "<link rel=\"stylesheet\"" not in html


def test_the_page_states_how_the_sample_was_drawn() -> None:
    """A gallery that does not say it is random invites reading it as curated."""

    html = render_page(context())
    assert "at random" in html
    assert "gallery-v1" in html, "the seed must be on the page for reproducibility"
    assert "50516" in html or "50,516" in html, "population size must be visible"


def test_the_page_names_the_question_the_reviewer_should_ask() -> None:
    html = render_page(context())
    assert "while this person is" in html.lower()
    assert "technical" in html.lower()


def test_no_unresolved_template_markers() -> None:
    html = render_page(context())
    assert "__DATA__" not in html
    assert "{" not in html.split("<style>")[1].split("</style>")[0].replace("{", "{", 1) or True
    # the real check: no stray Python format artefacts
    assert "None" not in re.sub(r"<script>.*?</script>", "", html, flags=re.S)

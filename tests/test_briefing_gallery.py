"""Tests for the read-only briefing gallery.

The two things that must not regress are that this page collects nothing — it is
shown to someone who is not being asked for judgements, and a stray note box
would invite them — and that FM2 clips actually contain the frames FM2 objected
to, which is the one claim the page makes about its own construction.
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from seamless_curation.briefing_gallery import build_briefing_html  # noqa: E402


def _page(**overrides):
    records = [
        {
            "file_id": "V00_A", "label": "naturalistic", "activity_type": "chat",
            "status": "rendered", "media_file": "clip-a.mp4",
            "source_media_file": "source/clip-a.mp4", "source_media_bytes": 34 * 1024 * 1024,
            "source_media_name": "V00_A.mp4", "clip_note": "positioned on the stretch",
            "signals_json": (
                '{"fm1_knee_between_torso_p50": 0.526, "fm2_smplh_valid_frac": 0.9837,'
                ' "fm2_longest_invalid_run_s": 2.9, "fm3_static_frac": 0.58,'
                ' "observed_duration_s": 204.0, "reproj_px_p50": 26.3}'
            ),
        },
        {
            "file_id": "V00_B", "label": "improvised", "status": "rendered",
            "media_file": "clip-b.mp4",
            "signals_json": '{"fm2_longest_invalid_run_s": 0.0}',
        },
    ]
    kwargs = {
        "sections": [{"id": "sec-one", "heading": "Section one", "blurb": "why", "records": records}],
        "title": "Briefing",
        "intro": "intro text",
        "detectors": [{"id": "fm1", "name": "FM1", "aim": "aim", "how": "how", "rate": "2.2% flagged"}],
        "stats": {"headline": [("4,000", "files scanned")], "table": "<table class='rates'></table>",
                  "footnote": "measured on this sample"},
    }
    kwargs.update(overrides)
    return build_briefing_html(**kwargs)


def test_the_briefing_collects_nothing() -> None:
    """It is a status report. Anything that invites input does not belong."""

    page = _page()
    for fragment in ("<textarea", 'type="radio"', "localStorage", "rubric",
                     "Export", "<form", "<input"):
        assert fragment not in page, f"{fragment} would invite input on a read-only page"


def test_every_section_is_reachable_from_the_contents() -> None:
    sections = [
        {"id": "a", "heading": "First", "blurb": "b", "records": []},
        {"id": "b", "heading": "Second", "blurb": "b", "records": []},
    ]
    page = _page(sections=sections)
    for anchor in ("a", "b"):
        assert f'href="#{anchor}"' in page, "table of contents must link the section"
        assert f'id="{anchor}"' in page, "section must carry the anchor it is linked by"
    # A visible break between categories, and a way back up from each.
    assert page.count('class="divider"') == 2
    assert page.count('href="#contents"') == 2
    assert page.count("2 clips") == 0 and "0 clips" in page


def test_only_the_curated_signal_rows_are_shown() -> None:
    """The reviewer gallery dumps every measurement; this one must not."""

    page = _page()
    assert "knee position along the torso" in page
    assert "frames with a trusted SMPL-H fit" in page
    # Present in the record, deliberately absent from the page.
    assert "reproj_px_p50" not in page
    # No raw column names leak through as labels.
    assert "fm1_knee_between_torso_p50" not in page


def test_a_file_with_no_untrusted_frames_says_none_not_zero_seconds() -> None:
    page = _page()
    assert ">none<" in page, "0.0 s reads like a rounded measurement; it is an absence"
    assert "2.9 s" in page


def test_clips_show_their_first_frame_before_being_played() -> None:
    """A wall of black rectangles tells a reader nothing about a section."""

    page = _page()
    assert 'preload="metadata"' in page
    assert 'preload="none"' not in page


def test_every_clip_can_be_swapped_for_an_in_memory_copy() -> None:
    """Streaming shows the first frame; only the blob makes it seekable.

    Two rounds of fixes to keyframes and to range support did not make the
    progress bar draggable, because an embedded webview replaces the media
    element's resource loader. A blob URL has no loader to refuse the seek.
    """

    page = _page()
    assert page.count('data-clip="') == page.count("<video")
    assert "createObjectURL" in page and "revokeObjectURL" in page


def test_each_clip_title_can_be_copied() -> None:
    page = _page()
    assert 'data-copy="V00_A"' in page
    assert "navigator.clipboard" in page


def test_a_missing_clip_is_stated_rather_than_linked() -> None:
    records = [{"file_id": "V00_C", "status": "error", "error": "decode failed",
                "media_file": None, "signals_json": "{}"}]
    page = _page(sections=[{"id": "s", "heading": "H", "blurb": "b", "records": records}])
    assert 'class="missing"' in page and "decode failed" in page
    assert "<video" not in page


def test_fm2_clips_are_positioned_to_contain_the_untrusted_stretch() -> None:
    """The one claim the page makes about how it was built."""

    from scripts.sample_v00_briefing_gallery import fm2_start_frame

    fps, render_frames = 30.0, 900

    # A short stretch is centred, so the fit is seen to fail and recover.
    row = SimpleNamespace(fps=fps, fm2_longest_invalid_run_start_frame=3000,
                          fm2_longest_invalid_run_frames=60)
    start, anchored = fm2_start_frame(row, render_frames, span=9000)
    assert anchored
    assert start <= 3000 and 3060 <= start + render_frames
    assert start + render_frames // 2 == pytest.approx(3030, abs=1)

    # A stretch longer than the clip cannot be contained; start just before it.
    row = SimpleNamespace(fps=fps, fm2_longest_invalid_run_start_frame=3000,
                          fm2_longest_invalid_run_frames=5000)
    start, anchored = fm2_start_frame(row, render_frames, span=9000)
    assert anchored and 3000 - start == pytest.approx(60, abs=1)

    # Near the end of a recording the clip is clamped, and still overlaps.
    row = SimpleNamespace(fps=fps, fm2_longest_invalid_run_start_frame=8950,
                          fm2_longest_invalid_run_frames=30)
    start, anchored = fm2_start_frame(row, render_frames, span=9000)
    assert anchored and start <= 8950 < start + render_frames

    # No recorded run: say so rather than pretend to anchor.
    row = SimpleNamespace(fps=fps, fm2_longest_invalid_run_start_frame=-1,
                          fm2_longest_invalid_run_frames=0)
    assert fm2_start_frame(row, render_frames, span=9000) == (0, False)


def test_invalid_run_measurement_finds_the_longest_stretch() -> None:
    import numpy as np

    from seamless_curation.v00_detectors import smplh_invalid_runs

    valid = np.ones(100, dtype=bool)
    valid[10:13] = False        # 3 frames
    valid[40:55] = False        # 15 frames, the longest
    valid[90:92] = False        # 2 frames
    result = smplh_invalid_runs(valid, 100)
    assert result["fm2_invalid_runs"] == 3
    assert result["fm2_longest_invalid_run_frames"] == 15
    assert result["fm2_longest_invalid_run_start_frame"] == 40

    clean = smplh_invalid_runs(np.ones(50, dtype=bool), 50)
    assert clean["fm2_longest_invalid_run_start_frame"] == -1
    assert clean["fm2_longest_invalid_run_frames"] == 0
    assert smplh_invalid_runs(None, 50)["fm2_invalid_runs"] == 0


def test_the_readiness_pill_is_gone_but_the_blob_loading_is_not() -> None:
    """Scrubbing is fixed and verified; the running commentary is now noise."""

    page = _page()
    assert "data-ready" not in page and "scrubbing works" not in page
    # The mechanism that fixed it stays.
    assert "createObjectURL" in page and page.count('data-clip="') == page.count("<video")


def test_the_contents_nest_when_sections_declare_a_group() -> None:
    """A 28-section page needs two levels or it cannot be navigated."""

    records = [{"file_id": "V00_A", "status": "rendered", "media_file": "a.mp4",
                "signals_json": "{}"}]
    sections = [
        {"id": "v00-pass", "heading": "V00 — passed", "short": "passed",
         "group": "V00", "blurb": "b", "records": records},
        {"id": "v00-fm1", "heading": "V00 — FM1", "short": "FM1", "group": "V00",
         "blurb": "b", "records": records},
        {"id": "v01-pass", "heading": "V01 — passed", "short": "passed",
         "group": "V01", "blurb": "b", "records": records},
    ]
    page = _page(sections=sections)
    assert page.count('class="toc-group"') == 2          # one per vendor
    assert page.count('href="#v00-pass"') == 2           # group anchor + entry
    assert 'href="#v01-pass"' in page
    # The short label is what the contents show; the full heading stays on the section.
    assert ">passed<" in page and "V00 — passed" in page


def test_numeric_table_headers_are_right_aligned_over_their_numbers() -> None:
    page = _page()
    assert "table.rates th.num, table.rates td.num" in page


def test_the_detectors_heading_is_not_hard_coded_to_three() -> None:
    page = _page()
    assert "The three checks" not in page

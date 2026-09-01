from __future__ import annotations

import csv
import os
from pathlib import Path

import numpy as np
import pytest

from seamless_curation.review_renderer import (
    JointSequence,
    RenderSettings,
    UnavailableJointProvider,
    add_playhead,
    audio_strip_background,
    build_gallery_html,
    load_joint_provider,
    normalize_review_config,
    read_manifest,
    write_gallery,
)
from seamless_curation.smplh_review_provider import (
    DISPLAY_EDGES,
    DISPLAY_GROUPS,
    DISPLAY_RAW_INDICES,
    ValidatedSmplhReviewProvider,
)


def test_unavailable_provider_is_explicit() -> None:
    provider = load_joint_provider(
        {"factory": None, "unavailable_reason": "reprojection validation pending"}
    )
    assert isinstance(provider, UnavailableJointProvider)
    sequence = provider.load(Path("unused"), 10, 20, (1080, 1920))
    sequence.validate(10)
    assert sequence.projected_xy_px is None
    assert sequence.root_relative_xyz_m is None
    assert sequence.status == "unavailable: reprojection validation pending"


def test_joint_sequence_rejects_frame_or_joint_mismatch() -> None:
    sequence = JointSequence(
        projected_xy_px=np.zeros((9, 2, 2), dtype=np.float32),
        root_relative_xyz_m=np.zeros((10, 2, 3), dtype=np.float32),
        edges=((0, 1),),
        groups=("body", "body"),
        status="test",
    )
    with pytest.raises(ValueError, match="projected_xy_px"):
        sequence.validate(10)


def test_review_settings_bound_duration_and_require_an_even_canvas() -> None:
    RenderSettings().validate()
    # Round 4 made clip length a real setting; the default is now 30 s.
    assert RenderSettings().duration_s == 30.0
    RenderSettings(duration_s=10.0).validate()
    for bad in (0.0, -1.0, 121.0, float("nan")):
        with pytest.raises(ValueError, match="duration_s"):
            RenderSettings(duration_s=bad).validate()
    with pytest.raises(ValueError, match="even"):
        RenderSettings(output_height=481).validate()


def test_audio_strip_and_playhead_have_expected_pixels() -> None:
    width, height = 200, 80
    lower = np.full(width, -0.5, dtype=np.float32)
    upper = np.full(width, 0.5, dtype=np.float32)
    background = audio_strip_background(
        width,
        height,
        lower,
        upper,
        own_vad=[(11.0, 12.0)],
        partner_vad=[(17.0, 18.0)],
        clip_start_s=10.0,
        duration_s=10.0,
        sample_rate=48000,
    )
    assert background.shape == (height, width, 3)
    frame = add_playhead(background, 0.5)
    playhead_x = round(0.5 * (width - 1))
    assert frame[height // 2, playhead_x, 2] > frame[height // 2, playhead_x, 0]


def test_gallery_is_free_text_only_and_escapes_metadata() -> None:
    records = [
        {
            "review_item_id": "item-1",
            "clip_id": "clip-1",
            "file_id": "<participant>",
            "sample_group": "uniform_random",
            "vendor": "V00",
            "label": "naturalistic",
            "split": "train",
            "activity_type": "gesture & speech",
            "status": "rendered",
            "media_file": "clip 1.mp4",
            "joint_status": "validated",
            "signals_json": '{"jitter": 0.25}',
        }
    ]
    page = build_gallery_html(records, "Test")
    assert "exploratory_free_text_v1" in page
    assert "Pass 1 intentionally has no predefined rubric" in page
    assert "rating" not in page.lower()
    assert "&lt;participant&gt;" in page
    assert "gesture &amp; speech" in page
    assert "clip%201.mp4" in page
    assert "jitter" in page


def test_manifest_allows_blinded_duplicate_items_but_not_clip_collisions(tmp_path: Path) -> None:
    manifest = tmp_path / "review.csv"
    columns = [
        "review_item_id", "clip_id", "file_id", "source_relbase",
        "sample_group", "start_frame",
    ]
    with manifest.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(
            [
                {"review_item_id": "a", "clip_id": "same", "file_id": "f", "source_relbase": "x/f", "sample_group": "uniform", "start_frame": "30"},
                {"review_item_id": "b", "clip_id": "same", "file_id": "f", "source_relbase": "x/f", "sample_group": "duplicate", "start_frame": "30"},
            ]
        )
    assert len(read_manifest(manifest)) == 2

    text = manifest.read_text(encoding="utf-8").replace("x/f,duplicate,30", "x/other,duplicate,30")
    manifest.write_text(text, encoding="utf-8")
    with pytest.raises(ValueError, match="multiple source intervals"):
        read_manifest(manifest)


def test_gallery_write_is_private(tmp_path: Path) -> None:
    path = tmp_path / "index.html"
    write_gallery(path, [{"review_item_id": "one", "status": "pending"}], "Review")
    assert path.stat().st_mode & 0o777 == 0o600


def test_review_provider_excludes_unaccepted_auxiliary_landmarks() -> None:
    assert tuple(range(57, 63)) == tuple(index for index in range(73) if 57 <= index < 63)
    assert not set(range(52, 63)).intersection(DISPLAY_RAW_INDICES)
    assert set(range(63, 73)).issubset(DISPLAY_RAW_INDICES)
    assert len(DISPLAY_GROUPS) == len(DISPLAY_RAW_INDICES) == 62
    assert max(max(edge) for edge in DISPLAY_EDGES) < len(DISPLAY_RAW_INDICES)


def test_review_provider_refuses_unaccepted_flat_hand_setting() -> None:
    with pytest.raises(ValueError, match="flat_hand_mean=True"):
        ValidatedSmplhReviewProvider("model_files", flat_hand_mean=False)


def test_authoritative_session2_config_normalizes() -> None:
    import yaml

    raw = yaml.safe_load(Path("configs/session2_review.yaml").read_text(encoding="utf-8"))
    config = normalize_review_config(raw)
    assert config["manifest"] == "./configs/session2_review_manifest.csv"
    assert config["private_output_root"] == "./artifacts/private_review_session2"
    assert config["render"]["duration_s"] == 10.0
    assert config["render"]["output_height"] == 480
    assert config["joint_provider"]["settings"]["flat_hand_mean"] is True
    assert config["joint_provider"]["settings"]["model_root"] == "./model_files"


def test_overlay_thickness_survives_the_panel_downscale() -> None:
    """Panels A and B exist to be looked at; sub-pixel lines defeat that."""

    from seamless_curation.review_renderer import overlay_thickness, panel_display_scale

    # The three rasters that dominate the corpus, into a 480-px-tall canvas.
    for width, height in ((1080, 1920), (2160, 3840), (2160, 2160), (3840, 2160)):
        panel_width = max(120, min(680, round(width / height * 384)))
        scale = panel_display_scale(width, height, panel_width, 384)
        thickness = overlay_thickness(width, height, scale)
        assert thickness * scale >= 1.5, (
            f"{width}x{height}: {thickness} native px renders as {thickness * scale:.2f} output px"
        )
        # Without compensation these all fell below one output pixel.
        assert overlay_thickness(width, height, 1.0) * scale < 1.0

    # A panel that is not downscaled must not get absurdly thick lines.
    assert overlay_thickness(640, 480, 0.8) <= 4


def test_every_playable_clip_offers_a_download_link() -> None:
    """A clip worth a closer look has to be openable outside the grid cell."""

    from urllib.parse import quote

    from seamless_curation.review_renderer import build_gallery_html

    records = [
        {
            "review_item_id": "item-1", "clip_id": "clip-1", "file_id": "V00_A",
            "sample_group": "pass", "status": "rendered",
            "media_file": "pass_000_V00_A f0000123.mp4", "duration_s": 30.0,
        },
        {
            "review_item_id": "item-2", "clip_id": "clip-2", "file_id": "V00_B",
            "sample_group": "flagged", "status": "metadata_only", "media_file": None,
        },
    ]
    page = build_gallery_html(records, "Test")

    media = "pass_000_V00_A f0000123.mp4"
    assert f'href="{quote(media)}" download="{media}"' in page
    # One playable clip, one unrenderable: exactly one set of download controls,
    # and the unrenderable card must not link to a file that does not exist.
    # The button carries the real clip length, so it cannot promise ten seconds
    # and hand over thirty.
    assert page.count("30-s panel clip") == 1
    assert "10-s panel clip" not in page
    assert page.count("Open full size") == 1
    assert 'class="missing"' in page
    # No source recording was linked for these records, so no full-video button.
    assert "Download full video" not in page


def test_source_media_links_are_symlinks_and_never_touch_the_source(tmp_path) -> None:
    """The full recording must be linked, not copied, and left unmodified."""

    from seamless_curation.review_renderer import link_source_media

    source_root = tmp_path / "src"
    (source_root / "nat/train").mkdir(parents=True)
    original = source_root / "nat/train/V00_A.mp4"
    original.write_bytes(b"x" * 4096)
    os.chmod(original, 0o444)
    before_mode = original.stat().st_mode
    before_inode = original.stat().st_ino

    output_root = tmp_path / "gallery"
    output_root.mkdir()
    records = [
        {"clip_id": "pass_000_V00_A_f0000123", "source_relbase": "nat/train/V00_A",
         "media_file": "pass_000_V00_A_f0000123.mp4", "status": "rendered"},
        {"clip_id": "pass_001_missing", "source_relbase": "nat/train/absent",
         "media_file": None, "status": "metadata_only"},
    ]
    linked = link_source_media(output_root, records, source_root)

    assert linked == 1
    link = output_root / records[0]["source_media_file"]
    assert link.is_symlink(), "must be a symlink, not a copy"
    assert link.resolve() == original.resolve()
    assert records[0]["source_media_bytes"] == 4096
    assert records[0]["source_media_name"] == "V00_A.mp4"
    # A record whose source is absent gains no link, so the page cannot offer one.
    assert "source_media_file" not in records[1]
    # The read-only source file is untouched: same inode, same mode, same bytes.
    assert original.stat().st_ino == before_inode
    assert original.stat().st_mode == before_mode
    assert original.read_bytes() == b"x" * 4096
    assert oct(os.stat(output_root / "source").st_mode & 0o777) == "0o700"


def test_gallery_offers_the_full_recording_as_the_primary_download() -> None:
    from seamless_curation.review_renderer import build_gallery_html

    records = [
        {
            "review_item_id": "item-1", "clip_id": "clip-1", "file_id": "V00_A",
            "sample_group": "pass", "status": "rendered",
            "media_file": "clip-1.mp4",
            "source_media_file": "source/clip-1.mp4",
            "source_media_bytes": 287 * 1024 * 1024,
            "source_media_name": "V00_A.mp4", "duration_s": 30.0,
        },
        {
            "review_item_id": "item-2", "clip_id": "clip-2", "file_id": "V00_B",
            "sample_group": "pass", "status": "rendered", "media_file": "clip-2.mp4",
        },
    ]
    page = build_gallery_html(records, "Test")

    assert 'href="source/clip-1.mp4" download="V00_A.mp4"' in page
    assert "Download full video" in page
    assert "287 MB" in page
    # The panel clip stays available beside it, and the record with no linked
    # source still gets its clip links but no full-video button.
    assert page.count("30-s panel clip") == 1
    # A record with no recorded duration drops the number rather than inventing
    # one, but still gets the button.
    assert page.count(">panel clip</a>") == 1
    assert page.count("Download full video") == 1

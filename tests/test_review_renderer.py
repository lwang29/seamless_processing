from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from seamless_curation.review_renderer import (
    JointSequence,
    RenderSettings,
    UnavailableJointProvider,
    add_playhead,
    audio_strip_background,
    load_joint_provider,
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


def test_review_provider_excludes_unaccepted_auxiliary_landmarks() -> None:
    assert tuple(range(57, 63)) == tuple(index for index in range(73) if 57 <= index < 63)
    assert not set(range(52, 63)).intersection(DISPLAY_RAW_INDICES)
    assert set(range(63, 73)).issubset(DISPLAY_RAW_INDICES)
    assert len(DISPLAY_GROUPS) == len(DISPLAY_RAW_INDICES) == 62
    assert max(max(edge) for edge in DISPLAY_EDGES) < len(DISPLAY_RAW_INDICES)


def test_review_provider_refuses_unaccepted_flat_hand_setting() -> None:
    with pytest.raises(ValueError, match="flat_hand_mean=True"):
        ValidatedSmplhReviewProvider("model_files", flat_hand_mean=False)


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

"""Tests for the Session-2 small fixes: split validity, guard band, hands.

The blanked-hand case is the point of the whole hand column: Session 1 showed
that erasing a hand moved no existing signal, so the test asserts both that the
new column reacts and that the pre-existing wrist signal still does not.
"""

from __future__ import annotations

import numpy as np
import pytest

from seamless_curation.features import (
    GUARD_BAND_DEFAULT_FRAMES,
    LEFT_HAND_SLICE,
    guard_invalid_runs,
    hand_availability,
    valid_frac,
    valid_frac_all,
    valid_frac_conjunction,
    wrist_speed_p90,
)


def test_split_validity_survives_one_absent_mask(dev_span) -> None:
    masks = {"smplh": dev_span.smplh_mask, "box": dev_span.box_mask, "movement": None}
    # The Session-1 conjunction collapses to NaN, which is exactly the 52%-NaN
    # problem the split was introduced to remove.
    assert np.isnan(valid_frac_all(masks["smplh"], None, masks["box"]).value)

    smplh = valid_frac(masks["smplh"], name="smplh")
    box = valid_frac(masks["box"], name="box")
    movement = valid_frac(masks["movement"], name="movement")
    core = valid_frac_conjunction(masks, required=("smplh", "box"))

    assert smplh.status == box.status == "ok"
    assert smplh.value == 1.0 and box.value == 1.0
    assert core.status == "ok" and core.value == 1.0
    assert np.isnan(movement.value) and movement.status == "missing_mask:movement"


def test_conjunction_reports_missing_required_mask() -> None:
    result = valid_frac_conjunction(
        {"smplh": np.ones(4, dtype=bool), "box": None}, required=("smplh", "box")
    )
    assert np.isnan(result.value)
    assert result.status == "missing_masks:box"


def test_conjunction_detects_length_mismatch() -> None:
    result = valid_frac_conjunction(
        {"smplh": np.ones(4, dtype=bool), "box": np.ones(5, dtype=bool)},
        required=("smplh", "box"),
    )
    assert np.isnan(result.value)
    assert result.status == "mask_length_mismatch"


def test_guard_band_widens_each_invalid_run_symmetrically() -> None:
    mask = np.ones(20, dtype=bool)
    mask[10] = False
    guarded = guard_invalid_runs(mask, 3)
    assert guarded is not None
    assert not guarded[7:14].any()
    assert guarded[:7].all() and guarded[14:].all()
    assert int((~guarded).sum()) == 7


def test_guard_band_clamps_at_array_edges() -> None:
    mask = np.ones(6, dtype=bool)
    mask[0] = False
    guarded = guard_invalid_runs(mask, 3)
    assert guarded is not None
    assert not guarded[:4].any()
    assert guarded[4:].all()


def test_guard_band_zero_is_identity_and_negative_rejected() -> None:
    mask = np.array([True, False, True])
    assert np.array_equal(guard_invalid_runs(mask, 0), mask)
    assert guard_invalid_runs(None, GUARD_BAND_DEFAULT_FRAMES) is None
    with pytest.raises(ValueError):
        guard_invalid_runs(mask, -1)


def test_guard_band_reduces_valid_fraction_on_a_terminal_run() -> None:
    """The `invalid_02.mp4` shape: an invalid run touching the final frame."""

    mask = np.ones(100, dtype=bool)
    mask[90:] = False
    plain = valid_frac(mask, name="box").value
    guarded = valid_frac(guard_invalid_runs(mask, 3), name="box").value
    assert plain == 0.90
    assert guarded == 0.87


def test_hand_availability_detects_a_blanked_hand_that_wrist_speed_misses(dev_span) -> None:
    keypoints = dev_span.keypoints
    baseline_left = hand_availability(keypoints, "left")
    assert baseline_left.status == "ok"

    blanked = keypoints.copy()
    frames = min(dev_span.blank_hand_frames, len(blanked))
    blanked[:frames, LEFT_HAND_SLICE, :2] = 0.0

    blanked_left = hand_availability(blanked, "left")
    blanked_right = hand_availability(blanked, "right")

    assert blanked_left.value < baseline_left.value
    assert blanked_left.value == pytest.approx(
        baseline_left.value - frames / len(blanked), abs=1e-9
    )
    # The untouched hand and the Session-1 wrist signal are both unmoved, which
    # is why hands needed their own column.
    assert blanked_right.value == hand_availability(keypoints, "right").value
    assert (
        wrist_speed_p90(blanked, dev_span.box_mask).value
        == wrist_speed_p90(keypoints, dev_span.box_mask).value
    )


def test_hand_availability_rejects_unknown_hand_and_reports_shapes() -> None:
    with pytest.raises(ValueError):
        hand_availability(np.zeros((2, 133, 3)), "middle")
    assert hand_availability(None, "left").status == "missing_keypoints"
    truncated = hand_availability(np.zeros((2, 17, 3)), "left")
    assert np.isnan(truncated.value)
    assert truncated.status.startswith("invalid_keypoint_shape")

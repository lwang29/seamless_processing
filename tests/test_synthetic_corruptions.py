from __future__ import annotations

import numpy as np

from seamless_curation.features import (
    duration_mismatch_s,
    hand_availability_frac,
    root_translation_accel,
    valid_frac_all,
    wrist_speed_p90,
    zero_velocity_run_max,
)


def _valid_fraction(span, smplh=None, movement=None, box=None):
    return valid_frac_all(
        span.smplh_mask if smplh is None else smplh,
        span.movement_mask if movement is None else movement,
        span.box_mask if box is None else box,
    ).value


def test_missing_movement_mask_is_unknown_not_valid(dev_span):
    measured = valid_frac_all(dev_span.smplh_mask, None, dev_span.box_mask)
    assert np.isnan(measured.value)
    assert measured.status == "missing_masks:movement"


def test_drop_and_duplicate_frames_change_duration_and_duplicate_run(dev_span):
    """Frame count/duplicate detectors respond; known temporal cross-fire is explicit."""

    n_frames = len(dev_span.translation)
    nominal_audio_duration = n_frames / dev_span.fps
    baseline_mismatch = duration_mismatch_s(n_frames, dev_span.fps, nominal_audio_duration)
    baseline_valid = _valid_fraction(dev_span)
    baseline_run = zero_velocity_run_max(dev_span.translation)
    baseline_accel = root_translation_accel(
        dev_span.translation, dev_span.smplh_mask, 1000.0
    ).value

    drop_at = n_frames // 2
    dropped_translation = np.delete(dev_span.translation, drop_at, axis=0)
    dropped_masks = [
        np.delete(mask, drop_at)
        for mask in (dev_span.smplh_mask, dev_span.movement_mask, dev_span.box_mask)
    ]
    dropped_mismatch = duration_mismatch_s(
        len(dropped_translation), dev_span.fps, nominal_audio_duration
    )

    duplicate_copies = 3
    duplicated_translation = np.insert(
        dev_span.translation,
        [drop_at] * duplicate_copies,
        dev_span.translation[drop_at],
        axis=0,
    )
    duplicated_masks = [
        np.insert(mask, [drop_at] * duplicate_copies, mask[drop_at])
        for mask in (dev_span.smplh_mask, dev_span.movement_mask, dev_span.box_mask)
    ]
    duplicated_mismatch = duration_mismatch_s(
        len(duplicated_translation), dev_span.fps, nominal_audio_duration
    )
    dropped_accel = root_translation_accel(
        dropped_translation, dropped_masks[0], 1000.0
    ).value
    duplicated_accel = root_translation_accel(
        duplicated_translation, duplicated_masks[0], 1000.0
    ).value

    assert baseline_mismatch == 0.0
    assert np.isclose(dropped_mismatch, 1.0 / dev_span.fps)
    assert np.isclose(duplicated_mismatch, duplicate_copies / dev_span.fps)
    assert zero_velocity_run_max(duplicated_translation) >= duplicate_copies
    assert zero_velocity_run_max(duplicated_translation) > baseline_run
    assert valid_frac_all(*dropped_masks).value == baseline_valid
    assert valid_frac_all(*duplicated_masks).value == baseline_valid
    # Dropping/duplicating a temporal sample necessarily perturbs local second
    # differences. This is an expected detector overlap, not a false positive.
    assert not np.isclose(dropped_accel, baseline_accel)
    assert not np.isclose(duplicated_accel, baseline_accel)


def test_known_amplitude_jitter_increases_accel_only(dev_span):
    """Alternating root jitter increases acceleration without cross-firing."""

    scale = 1000.0
    baseline_accel = root_translation_accel(
        dev_span.translation, dev_span.smplh_mask, scale
    ).value
    baseline_valid = _valid_fraction(dev_span)
    baseline_wrist = wrist_speed_p90(dev_span.keypoints, dev_span.box_mask).value
    nominal_audio_duration = len(dev_span.translation) / dev_span.fps
    baseline_duration = duration_mismatch_s(
        len(dev_span.translation), dev_span.fps, nominal_audio_duration
    )

    jittered = dev_span.translation.copy()
    alternating = np.where(np.arange(len(jittered)) % 2 == 0, 1.0, -1.0)
    jittered[:, 0] += alternating * dev_span.jitter_amplitude
    jittered_accel = root_translation_accel(jittered, dev_span.smplh_mask, scale).value

    assert jittered_accel > baseline_accel
    assert _valid_fraction(dev_span) == baseline_valid
    assert wrist_speed_p90(dev_span.keypoints, dev_span.box_mask).value == baseline_wrist
    assert duration_mismatch_s(
        len(jittered), dev_span.fps, nominal_audio_duration
    ) == baseline_duration


def test_blank_left_hand_lowers_only_hand_availability(dev_span):
    """An exact-zero hand block changes availability, not the four harness signals."""

    n_blank = dev_span.blank_hand_frames
    start = len(dev_span.keypoints) // 2 - n_blank // 2
    blanked = dev_span.keypoints.copy()
    blanked[start : start + n_blank, 91:112, :2] = 0.0

    baseline_hand = hand_availability_frac(dev_span.keypoints, "left")
    blanked_hand = hand_availability_frac(blanked, "left")
    baseline_valid = _valid_fraction(dev_span)
    baseline_accel = root_translation_accel(
        dev_span.translation, dev_span.smplh_mask, 1000.0
    ).value
    baseline_wrist = wrist_speed_p90(dev_span.keypoints, dev_span.box_mask).value
    blanked_wrist = wrist_speed_p90(blanked, dev_span.box_mask).value
    audio_duration = len(dev_span.translation) / dev_span.fps

    assert baseline_hand == 1.0
    assert np.isclose(blanked_hand, baseline_hand - n_blank / len(blanked))
    assert _valid_fraction(dev_span) == baseline_valid
    assert root_translation_accel(
        dev_span.translation, dev_span.smplh_mask, 1000.0
    ).value == baseline_accel
    assert blanked_wrist == baseline_wrist
    assert duration_mismatch_s(
        len(blanked), dev_span.fps, audio_duration
    ) == 0.0

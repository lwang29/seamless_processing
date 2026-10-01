"""Annotate-time rules that are not covered by the module tests."""

from __future__ import annotations

import numpy as np
import pandas as pd

from seamless_curation import annotate, posture


def test_the_unvalidated_rule_never_decides_sitting() -> None:
    """Visual QA: 24 of 24 sampled V02 'sitting' clips (hip-only and hip+knee) were standing."""

    bins = pd.DataFrame({
        "file_id": ["a"] * 6 + ["b"] * 6,
        "bin_index": list(range(6)) * 2,
        "hip_flexion_deg": [115.0] * 6 + [100.0] * 6,
        "knee_between": [np.nan] * 6 + [0.3] * 6,   # a: ankles out of frame; b: ankles visible
    })
    recordings = pd.DataFrame({"file_id": ["a", "b"], "posture_rule": ["unvalidated_hip_knee"] * 2,
                               "recording_n_clips": [1, 1]})
    out = annotate.aggregate_posture(bins, recordings, posture.PostureRules(), by="file_id").set_index("file_id")
    assert out.loc["a", "posture"] == "unclear"
    assert out.loc["b", "posture"] == "unclear"


def test_v03_hip_only_sitting_is_still_decided() -> None:
    bins = pd.DataFrame({"file_id": ["c"] * 6, "bin_index": range(6), "hip_flexion_deg": [100.0] * 6,
                         "knee_between": [np.nan] * 6})
    recordings = pd.DataFrame({"file_id": ["c"], "posture_rule": ["v03_hip"], "recording_n_clips": [1]})
    out = annotate.aggregate_posture(bins, recordings, posture.PostureRules(), by="file_id")
    assert out["posture"].iloc[0] == "sitting"

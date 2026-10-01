"""Expressivity is a rank, so the tests pin the ranking rules, not magnitudes.

What must hold whatever the corpus looks like: more motion never lowers a
channel or the score; tied values (exact zeros) take the middle of their block;
a clip without a trustworthy measure gets NA and a status saying why, never a
number; each rig is ranked against itself (identical raw values may score
differently on V00 and V02, but never differently in the pooled rank); the
reference is a pure function of the reference rows; and the score is uniform
over the clips that define it, which is what makes the 1/3-2/3 level cuts mean
tertiles.
"""

from __future__ import annotations

import json
import time

import numpy as np
import pandas as pd
import pytest

from seamless_curation import expressivity as ex
from seamless_curation.schema import EXPRESSIVITY_GROUPS

L = 30.0


def make_clips(n: int, *, vendor: str = "V00", group: str | None = None, seed: int = 0,
               scale: float = 1.0) -> pd.DataFrame:
    """Measured, full-length clips with independent positive raw measures."""

    rng = np.random.default_rng(seed)
    frame = pd.DataFrame({
        "file_id": [f"{vendor}_S0001_I{i // 4:08d}_P0001" for i in range(n)],
        "clip_index": np.arange(n) % 4,
        "vendor": vendor,
        "expressivity_reference_group": group or vendor,
        "clip_seconds": np.float32(L),
        "is_partial": False,
        "subject_present_frac": np.float32(1.0),
        "wrists_in_frame_frac": rng.uniform(0.5, 1.0, n).astype(np.float32),
        "smplh_anamorphic": "none",
        "arm_speed_p50_mm_s": (scale * rng.lognormal(3.5, 0.5, n)).astype(np.float32),
        "wrist_range_mm": (scale * rng.lognormal(5.5, 0.4, n)).astype(np.float32),
        "wrist_excursion_p90_mm": (scale * rng.lognormal(4.5, 0.4, n)).astype(np.float32),
        "head_speed_p75_deg_s": (scale * rng.lognormal(2.5, 0.5, n)).astype(np.float32),
        "hand_artic_p75_rad_s": (scale * rng.lognormal(0.0, 0.5, n)).astype(np.float32),
        "arm_speed_cv": rng.uniform(0.1, 1.5, n).astype(np.float32),
        "fau_variability": rng.uniform(0.05, 0.5, n).astype(np.float32),
        "vocal_level_range_db": rng.uniform(5.0, 25.0, n).astype(np.float32),
    })
    return frame


@pytest.fixture(scope="module")
def corpus() -> pd.DataFrame:
    parts = [
        make_clips(3000, vendor="V00", seed=1),
        make_clips(2000, vendor="V01", seed=2),
        make_clips(2500, vendor="V02", seed=3, scale=1.4),  # a rig whose fits move faster
        make_clips(1500, vendor="V03", group="V03_portrait", seed=4),
        make_clips(700, vendor="V03", group="V03_room", seed=5, scale=0.8),
    ]
    return pd.concat(parts, ignore_index=True)


@pytest.fixture(scope="module")
def reference(corpus: pd.DataFrame) -> dict:
    return ex.build_reference(corpus)


def _probe(base: pd.DataFrame, **values) -> pd.DataFrame:
    row = base.iloc[[0]].copy()
    for k, v in values.items():
        row[k] = v
    return row.reset_index(drop=True)


def _rows(base: pd.DataFrame, *overrides: dict) -> pd.DataFrame:
    """One copy of base's first row per override dict (no concat of all-NA rows)."""

    rows = base.iloc[[0] * len(overrides)].reset_index(drop=True)
    for i, values in enumerate(overrides):
        for k, v in values.items():
            rows.loc[i, k] = v
    return rows


# -----------------------------------------------------------------------------
# percentile rule
# -----------------------------------------------------------------------------
def test_percentile_mid_rank_edges_and_nan() -> None:
    q = np.array([1.0, 2.0, 3.0, 4.0])
    p = ex.percentile(np.array([0.0, 1.0, 2.5, 4.0, 9.0, np.nan]), q)
    assert p[0] == 0.0 and p[4] == 1.0
    assert p[1] == pytest.approx(0.125)  # (0 + 1) / 2 / 4
    assert p[2] == pytest.approx(0.5)
    assert p[3] == pytest.approx(0.875)
    assert np.isnan(p[5])
    assert np.isnan(ex.percentile(np.array([1.0, 2.0]), np.array([]))).all()


def test_percentile_equals_the_two_search_definition() -> None:
    rng = np.random.default_rng(5)
    for _ in range(20):
        q = np.sort(np.round(rng.normal(0, 2, rng.integers(1, 60)), 0))  # heavy ties
        v = np.r_[np.round(rng.normal(0, 3, 500), 0), q, np.nan, -np.inf, np.inf]
        expected = np.clip((np.searchsorted(q, v, "left") + np.searchsorted(q, v, "right")) / (2 * q.size), 0, 1)
        expected[np.isnan(v)] = np.nan
        np.testing.assert_array_equal(ex.percentile(v, q), expected)


def test_ties_at_zero_take_the_middle_of_their_block() -> None:
    # 40% of reference clips are motionless (exact zeros): a zero must rank at
    # 0.2, not at 0 (bottom of the block) or 0.4 (top).
    values = np.r_[np.zeros(400), np.linspace(1.0, 100.0, 600)]
    clips = make_clips(1000, seed=9)
    clips["arm_speed_p50_mm_s"] = values.astype(np.float32)
    ref = ex.build_reference(clips)
    q = np.asarray(ref["groups"]["V00"]["quantiles"]["arm_speed_p50_mm_s"])
    assert ex.percentile(np.array([0.0]), q)[0] == pytest.approx(0.2, abs=2e-3)
    scored = ex.score(clips, ref, clip_seconds_nominal=L)
    zero = scored.loc[values == 0, "expr_energy"]
    assert zero.nunique() == 1 and float(zero.iloc[0]) == pytest.approx(0.2, abs=2e-3)


# -----------------------------------------------------------------------------
# monotonicity
# -----------------------------------------------------------------------------
def test_more_arm_speed_never_lowers_energy_or_score(corpus: pd.DataFrame, reference: dict) -> None:
    base = corpus[corpus["vendor"] == "V00"]
    speeds = np.geomspace(1.0, 2000.0, 60).astype(np.float32)
    probe = pd.concat([_probe(base, arm_speed_p50_mm_s=s) for s in speeds], ignore_index=True)
    scored = ex.score(probe, reference, clip_seconds_nominal=L)
    for column in ("expr_energy", "expressivity_score", "expressivity_score_pooled"):
        v = scored[column].to_numpy(np.float64)
        assert np.all(np.diff(v) >= 0), column
        assert v[-1] > v[0], column
    assert scored["expr_energy"].iloc[0] == 0.0 and scored["expr_energy"].iloc[-1] == 1.0
    # The other channels do not move.
    for column in ("expr_amplitude", "expr_head", "expr_hands", "expr_variability"):
        assert scored[column].nunique() == 1, column


def test_every_body_channel_raises_the_score(corpus: pd.DataFrame, reference: dict) -> None:
    base = _probe(corpus[corpus["vendor"] == "V01"])
    low = ex.score(base, reference, clip_seconds_nominal=L)["expressivity_score"].iloc[0]
    for measure in ex.COMPOSITE_INPUTS:
        hi = _probe(base, **{measure: float(base[measure].iloc[0]) * 50.0})
        assert ex.score(hi, reference, clip_seconds_nominal=L)["expressivity_score"].iloc[0] > low, measure
    # Extra channels do not enter the score.
    for measure in ex.EXTRA_INPUTS:
        hi = _probe(base, **{measure: float(base[measure].iloc[0]) * 50.0})
        assert ex.score(hi, reference, clip_seconds_nominal=L)["expressivity_score"].iloc[0] == low, measure


# -----------------------------------------------------------------------------
# statuses and NA
# -----------------------------------------------------------------------------
def test_status_function_order() -> None:
    s = ex.expressivity_status
    assert s(30.0, 1.0, "none", True) == "measured"
    assert s(2.0, 1.0, "mild", True) == "measured"  # exactly 2 s is long enough
    assert s(1.9, 0.0, "severe", False) == "too_short"
    assert s(float("nan"), 1.0, "none", True) == "too_short"
    assert s(30.0, 0.0, "severe", False) == "pose_distorted"
    assert s(30.0, 0.0, "none", False) == "no_subject"
    assert s(30.0, float("nan"), "none", True) == "no_subject"
    assert s(30.0, None, None, True) == "no_subject"
    assert s(30.0, 0.4, "none", False) == "incomplete"


def test_statuses_and_na_propagation(corpus: pd.DataFrame, reference: dict) -> None:
    base = corpus[corpus["vendor"] == "V00"]
    rows = _rows(
        base,
        {},                                                                  # measured
        dict(clip_seconds=1.5, is_partial=True,
             arm_speed_p50_mm_s=np.nan, wrists_in_frame_frac=np.nan),        # too_short
        dict(vendor="V01", expressivity_reference_group="not_applicable",
             smplh_anamorphic="severe"),                                     # pose_distorted
        dict(subject_present_frac=0.0),                                      # no_subject
        dict(hand_artic_p75_rad_s=np.nan),                                   # incomplete
        dict(fau_variability=np.nan, vocal_level_range_db=np.nan,
             arm_speed_cv=np.nan),                                           # measured, extras NA
    )
    scored = ex.score(rows, reference, clip_seconds_nominal=L)
    assert scored["expressivity_status"].tolist() == [
        "measured", "too_short", "pose_distorted", "no_subject", "incomplete", "measured"]
    unmeasured = scored.iloc[1:5]
    for column in (*ex.COMPOSITE, *ex.EXTRA, "expressivity_score", "expressivity_score_pooled",
                   "expressivity_confidence"):
        assert unmeasured[column].isna().all(), column
    assert (unmeasured[["expressivity_level", "expressivity_level_pooled"]] == "unknown").all().all()
    last = scored.iloc[5]
    assert np.isnan(last["expr_face"]) and np.isnan(last["expr_vocal"]) and np.isnan(last["expr_variability"])
    assert np.isfinite(last["expressivity_score"]) and last["expressivity_level"] != "unknown"
    assert list(scored.columns) == list(ex.SCORE_COLUMNS)
    assert all(scored[c].dtype == np.float32 for c in (*ex.COMPOSITE, *ex.EXTRA, "expressivity_score",
                                                       "expressivity_score_pooled",
                                                       "expressivity_confidence"))
    assert set(scored["expressivity_status"]) <= set(ex.STATUSES)


def test_face_channel_is_v00_only(corpus: pd.DataFrame, reference: dict) -> None:
    scored = ex.score(corpus, reference, clip_seconds_nominal=L)
    v00 = corpus["vendor"] == "V00"
    assert scored.loc[v00, "expr_face"].notna().all()
    assert scored.loc[~v00, "expr_face"].isna().all()
    assert scored.loc[~v00, "expr_vocal"].notna().all()


def test_missing_extra_inputs_are_na_not_errors(corpus: pd.DataFrame, reference: dict) -> None:
    clips = corpus.drop(columns=list(ex.EXTRA_INPUTS)).iloc[:50]
    scored = ex.score(clips, reference, clip_seconds_nominal=L)
    assert scored[list(ex.EXTRA)].isna().all().all()
    assert scored["expressivity_score"].notna().all()
    with pytest.raises(KeyError):
        ex.score(corpus.drop(columns=["head_speed_p75_deg_s"]), reference, clip_seconds_nominal=L)


def test_confidence_is_coverage(corpus: pd.DataFrame, reference: dict) -> None:
    base = corpus[corpus["vendor"] == "V02"]
    rows = pd.concat([_probe(base, wrists_in_frame_frac=0.8),
                      _probe(base, wrists_in_frame_frac=0.8, clip_seconds=12.0, is_partial=True)],
                     ignore_index=True)
    conf = ex.score(rows, reference, clip_seconds_nominal=L)["expressivity_confidence"].to_numpy()
    assert conf[0] == pytest.approx(0.8)
    assert conf[1] == pytest.approx(0.8 * 12.0 / 30.0)


def test_string_na_and_pandas_string_dtype(corpus: pd.DataFrame, reference: dict) -> None:
    clips = corpus.iloc[:20].copy()
    for column in ("vendor", "expressivity_reference_group", "smplh_anamorphic"):
        clips[column] = clips[column].astype("string")
    clips.loc[clips.index[:3], "smplh_anamorphic"] = pd.NA
    clips["is_partial"] = clips["is_partial"].astype("boolean")
    scored = ex.score(clips, reference, clip_seconds_nominal=L)
    assert (scored["expressivity_status"] == "measured").all()
    assert ex.build_reference(clips)["groups"]["V00"]["n_clips"] == 20


# -----------------------------------------------------------------------------
# groups
# -----------------------------------------------------------------------------
def test_reference_group_rules() -> None:
    g = ex.reference_group
    assert g("V00", "portrait_1080x1920", "none", True) == "V00"
    assert g("V01", "anamorphic_1012x1920", "mild", True) == "V01"
    assert g("V01", "anamorphic_2160x2160", "severe", True) == "not_applicable"
    assert g("V02", "portrait_1080x1920", "none", False) == "not_applicable"
    assert g("V02", "portrait_1080x1920", "none", None) == "not_applicable"
    assert g("V03", "rotated_3840x2160", "none", True) == "V03_room"
    assert g("V03", "rotated_640x480", "none", True) == "V03_room"
    assert g("V03", "pillarbox_1080x960", "none", True) == "V03_portrait"
    assert g("V03", "portrait_2160x3840", "none", True) == "V03_portrait"
    with pytest.raises(ValueError):
        g("V09", "portrait_1080x1920", "none", True)
    vendors = ["V00", "V01", "V01", "V02", "V03", "V03", "V03"]
    rasters = ["portrait_1080x1920", "anamorphic_1012x1920", "anamorphic_1920x1080",
               "no_video", "rotated_640x480", "pillarbox_2180x3840", None]
    anam = ["none", "mild", "severe", None, "none", "none", pd.NA]
    measured = [True, True, True, pd.NA, True, True, True]
    vec = ex.reference_groups(vendors, rasters, anam, measured)
    assert list(vec) == [g(*a) for a in zip(vendors, rasters, anam, measured)]
    assert set(vec) <= set(EXPRESSIVITY_GROUPS)


def test_groups_are_ranked_separately_but_pooled_is_shared(corpus: pd.DataFrame, reference: dict) -> None:
    base = corpus[corpus["vendor"] == "V00"]
    raw = {m: float(np.median(corpus[m])) for m in ex.RAW_MEASURES}
    v00 = _probe(base, **raw)
    v02 = _probe(base, vendor="V02", expressivity_reference_group="V02", **raw)
    scored = ex.score(pd.concat([v00, v02], ignore_index=True), reference, clip_seconds_nominal=L)
    s = scored["expressivity_score"].to_numpy()
    # V02's fits move 1.4x faster in this corpus: the same raw motion is less
    # expressive for that rig, but corpus-wide it is the same clip.
    assert s[0] > s[1] + 0.1
    assert scored["expr_energy"].iloc[0] > scored["expr_energy"].iloc[1]
    assert scored["expressivity_score_pooled"].iloc[0] == scored["expressivity_score_pooled"].iloc[1]


def test_group_and_vendor_mismatch_raises(corpus: pd.DataFrame, reference: dict) -> None:
    base = corpus[corpus["vendor"] == "V00"]
    with pytest.raises(ValueError):
        ex.score(_probe(base, expressivity_reference_group="V02"), reference, clip_seconds_nominal=L)
    with pytest.raises(ValueError):
        ex.score(_probe(base, expressivity_reference_group="not_applicable"), reference, clip_seconds_nominal=L)
    with pytest.raises(ValueError):
        ex.score(_probe(base), reference, clip_seconds_nominal=0.0)
    with pytest.raises(ValueError):
        ex.score(_probe(base), {"kind": "something else"}, clip_seconds_nominal=L)


# -----------------------------------------------------------------------------
# reference
# -----------------------------------------------------------------------------
def test_reference_is_deterministic_json_and_excludes_non_reference_rows(corpus: pd.DataFrame,
                                                                          reference: dict) -> None:
    shuffled = corpus.sample(frac=1.0, random_state=7).reset_index(drop=True)
    again = ex.build_reference(shuffled)
    assert json.dumps(again, sort_keys=True, allow_nan=False) == json.dumps(reference, sort_keys=True,
                                                                             allow_nan=False)
    groups = reference["groups"]
    assert set(groups) == {*(g for g in EXPRESSIVITY_GROUPS if g != "not_applicable"), "pooled"}
    assert groups["pooled"]["n_clips"] == len(corpus) == sum(
        groups[g]["n_clips"] for g in ex.REFERENCE_GROUPS)
    for g in groups.values():
        for table in g["quantiles"].values():
            assert len(table) in (0, reference["n_quantiles"])
            assert np.all(np.diff(table) >= 0)
        assert ex.COMPOSITE_KEY in g["quantiles"]

    extra = pd.concat([
        corpus,
        _rows(corpus,
              dict(is_partial=True, clip_seconds=10.0, arm_speed_p50_mm_s=1e6),
              dict(hand_artic_p75_rad_s=np.nan),
              dict(vendor="V01", expressivity_reference_group="not_applicable", smplh_anamorphic="severe")),
    ], ignore_index=True)
    ref2 = ex.build_reference(extra)
    assert json.dumps(ref2, sort_keys=True) == json.dumps(reference, sort_keys=True)


def test_empty_group_scores_na_without_crashing() -> None:
    clips = make_clips(300, vendor="V00", seed=3)
    ref = ex.build_reference(clips, n_quantiles=101)
    assert ref["groups"]["V02"]["n_clips"] == 0
    assert ref["groups"]["V02"]["quantiles"]["arm_speed_p50_mm_s"] == []
    probe = _probe(clips, vendor="V02", expressivity_reference_group="V02")
    scored = ex.score(probe, ref, clip_seconds_nominal=L)
    assert scored["expressivity_status"].iloc[0] == "measured"
    assert np.isnan(scored["expressivity_score"].iloc[0])
    assert scored["expressivity_level"].iloc[0] == "unknown"
    assert np.isfinite(scored["expressivity_score_pooled"].iloc[0])


def test_score_is_uniform_over_reference_clips(corpus: pd.DataFrame, reference: dict) -> None:
    scored = ex.score(corpus, reference, clip_seconds_nominal=L)
    for group, idx in corpus.groupby("expressivity_reference_group").groups.items():
        s = np.sort(scored.loc[idx, "expressivity_score"].to_numpy(np.float64))
        expected = (np.arange(s.size) + 0.5) / s.size
        assert np.max(np.abs(s - expected)) < 0.01, group
        shares = scored.loc[idx, "expressivity_level"].value_counts(normalize=True)
        assert all(abs(shares[k] - 1 / 3) < 0.01 for k in ("low", "medium", "high")), group
    pooled = np.sort(scored["expressivity_score_pooled"].to_numpy(np.float64))
    assert np.max(np.abs(pooled - (np.arange(pooled.size) + 0.5) / pooled.size)) < 0.01
    # ...while the pooled rank carries the rig offset the within-group score removes.
    high = scored["expressivity_level_pooled"].eq("high").groupby(corpus["vendor"]).mean()
    assert high["V02"] > 0.5 > high["V00"]


def test_correlation_matrix(corpus: pd.DataFrame, reference: dict) -> None:
    scored = ex.score(corpus, reference, clip_seconds_nominal=L)
    corr = ex.correlation_matrix(scored)
    assert list(corr.columns) == [*ex.COMPOSITE, *ex.EXTRA]
    assert np.allclose(np.diag(corr.to_numpy()), 1.0)
    assert np.allclose(corr.to_numpy(), corr.to_numpy().T, equal_nan=True)


# -----------------------------------------------------------------------------
# aggregates
# -----------------------------------------------------------------------------
def test_recording_and_interaction_aggregates() -> None:
    scored = pd.DataFrame({
        "file_id": ["A", "A", "A", "B", "B", "C"],
        "interaction_key": ["I1", "I1", "I1", "I1", "I1", "I2"],
        "clip_index": [0, 1, 2, 0, 1, 0],
        "clip_seconds": [30.0, 30.0, 15.0, 30.0, 30.0, 30.0],
        "expressivity_score": [0.2, 0.4, 0.9, 0.5, np.nan, 0.7],
        "expressivity_status": ["measured"] * 4 + ["incomplete", "measured"],
    })
    rec = ex.recording_expressivity(scored)
    assert rec.loc["A", "recording_expressivity_mean"] == pytest.approx((6 + 12 + 13.5) / 75)
    assert rec.loc["A", "recording_expressivity_trend"] == pytest.approx(1.0)
    assert np.isnan(rec.loc["B", "recording_expressivity_clip_sd"])
    assert np.isnan(rec.loc["C", "recording_expressivity_trend"])
    inter = ex.interaction_expressivity(scored)
    assert inter.loc["I1", "interaction_expressivity_gap"] == pytest.approx(abs(31.5 / 75 - 0.5), abs=1e-6)
    assert np.isnan(inter.loc["I2", "interaction_expressivity_mean"])


# -----------------------------------------------------------------------------
# scale
# -----------------------------------------------------------------------------
def test_scales_to_a_million_clips() -> None:
    parts = [make_clips(260_000, vendor=v, group=g, seed=i)
             for i, (v, g) in enumerate((("V00", "V00"), ("V01", "V01"), ("V02", "V02"), ("V03", "V03_room")))]
    clips = pd.concat(parts, ignore_index=True)
    t0 = time.perf_counter()
    ref = ex.build_reference(clips)
    scored = ex.score(clips, ref, clip_seconds_nominal=L)
    elapsed = time.perf_counter() - t0
    assert len(scored) == 1_040_000
    assert elapsed < 30.0, elapsed

"""Turn the catalog and the scan's continuous measurements into the published tables.

This is where every judgement lives — posture bands, expressivity references,
speaking roles, partner links, quality labels — and nothing here reads the
release. Re-tuning any of it costs one ``annotate`` run.

The output is eight tables, one per level (see :mod:`seamless_curation.schema`),
cast to the registry's Arrow schema, validated, and written as one directory
that replaces the previous one atomically, so a reader never sees clips from one
run next to recordings from another.

Rules that decide how levels relate, stated once here and enforced by the
validator:

* **A conversation-level value is computed from both members**, once, on the
  interaction (or dyad-window) row. It is never copied onto clip rows, and when
  it needs both members and one is missing or unmeasured it is NA rather than a
  one-sided value dressed up as the conversation's.
* **A clip-level value never propagates upward unchanged.** Recording and
  interaction summaries are recomputed from their own inputs (recording posture
  from all of the recording's 1-s bins, not from clip labels) and carry a level
  prefix.
* **Partners are linked only when they are time-aligned.** ``partner_clip_id``
  is set when both recordings are measured, have clip ``k``, and agree in length
  (``pair_duration_agreement`` is ``agree`` or ``minor_mismatch``).
"""

from __future__ import annotations

import json
import os
import shutil
import time
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

from . import expressivity, posture, schema
from .ids import clip_id as make_clip_id, window_id as make_window_id

CATALOG_TABLES = ("recordings", "participants", "sessions", "prompts", "interactions")
DECIDED = ("standing", "sitting")
#: Clip columns that condition on who is speaking; NA where speech is unannotated.
SPEECH_CONDITIONED = (
    "own_speech_s", "own_speech_frac", "speech_rate_wps", "arm_active_frac_speech",
    "arm_active_frac_silence", "arm_speed_speech_p50_mm_s", "wrist_height_speech_p75_mm",
    "speech_segment_count", "speech_segments_with_motion_frac", "speech_motion_sync_r",
    "speech_motion_sync_lag_s", "own_speech_level_db", "vocal_level_range_db",
)
#: A clip reaching more than this past the speech-annotated span is unannotated.
ANNOTATED_SPAN_TOLERANCE_S = 1.0
#: Meta's VAD also drops out mid-recording and resumes (1,220 recordings; e.g.
#: V02_S3400_I00000120: ~0 VAD for 43 min under 40-200 transcript words/min at the
#: speaker's own level, echo 0). A clip whose timed words outrun its VAD speech —
#: at least this many words, at more than VAD_GAP_WORDS_PER_S words per second of
#: VAD speech (conversational speech runs ~2-4) — is treated as unannotated rather
#: than silent. (Interim: filling such gaps from the words belongs in
#: speech.speech_track at the next rescan.)
VAD_GAP_MIN_WORDS = 10
VAD_GAP_WORDS_PER_S = 8.0


@dataclass
class ScanTables:
    recordings: pd.DataFrame
    clips: pd.DataFrame
    bins: pd.DataFrame
    moi: pd.DataFrame
    windows: pd.DataFrame
    pairs: pd.DataFrame


# ============================================================================= helpers
def _na_float(frame: pd.DataFrame, column: str) -> pd.Series:
    if column in frame.columns:
        return pd.to_numeric(frame[column], errors="coerce").astype("float64")
    return pd.Series(np.nan, index=frame.index, dtype="float64")


def _ensure(frame: pd.DataFrame, column: str, value: Any = np.nan) -> None:
    if column not in frame.columns:
        frame[column] = value


def _level(score: pd.Series) -> pd.Series:
    out = pd.Series("unknown", index=score.index, dtype=object)
    out[score < 1 / 3] = "low"
    out[(score >= 1 / 3) & (score < 2 / 3)] = "medium"
    out[score >= 2 / 3] = "high"
    return out


# ============================================================================= recordings
def build_recordings(catalog: pd.DataFrame, scanned: pd.DataFrame, rules: posture.PostureRules,
                     bins: pd.DataFrame) -> pd.DataFrame:
    missing = sorted(set(catalog["file_id"]) - set(scanned["file_id"]))
    if missing:
        raise RuntimeError(f"{len(missing)} catalog recordings were not scanned, e.g. {missing[:5]}")
    extra = sorted(set(scanned["file_id"]) - set(catalog["file_id"]))
    if extra:
        raise RuntimeError(f"{len(extra)} scanned recordings are not in the catalog, e.g. {extra[:5]}")
    duplicated = scanned["file_id"].duplicated()
    if duplicated.any():
        raise RuntimeError(f"{int(duplicated.sum())} recordings scanned twice")
    rec = catalog.merge(scanned, on="file_id", how="left", validate="one_to_one")
    measured = rec["measurement_status"] == "measured"
    # Defensive: a recording that is not 'measured' carries no measured values, even
    # if a failure happened after some were computed.
    if (~measured).any():
        rec.loc[~measured, "recording_n_clips"] = 0
        for column in scanned.columns:
            if column not in ("file_id", "measurement_status", "scan_error", "scan_seconds", "n_frames",
                              "recording_n_clips") and column in rec.columns:
                rec.loc[~measured, column] = None
    # No VAD and no transcript: silence and a missing annotation cannot be told apart.
    unannotated = rec["speech_source"] == "none"
    # The speaking share is over the span speech is annotated for, not the pose grid.
    span = _na_float(rec, "speech_annotated_until_s")
    rec["recording_own_speech_frac"] = (_na_float(rec, "recording_own_speech_s") / span).where(span > 0).clip(0, 1)
    for column in ("recording_own_speech_s", "recording_own_speech_frac"):
        rec[column] = _na_float(rec, column).where(~unannotated)

    n_frames = _na_float(rec, "n_frames")
    rec["npz_video_frame_diff"] = (n_frames - _na_float(rec, "video_nb_frames")).where(measured)
    rec["timebase_drift_s"] = (n_frames / _na_float(rec, "fps") - _na_float(rec, "video_duration_s")).abs().where(measured)
    status = pd.Series("unknown", index=rec.index, dtype=object)
    status[rec["timebase_drift_s"] <= 0.5] = "consistent"
    status[rec["timebase_drift_s"] > 0.5] = "drift"
    rec["timebase_status"] = status
    rec["recording_n_clips"] = pd.to_numeric(rec["recording_n_clips"], errors="coerce").fillna(0)

    audio_status = rec["audio_status"] if "audio_status" in rec else pd.Series(None, index=rec.index)
    quality = pd.Series(None, index=rec.index, dtype=object)
    ok = audio_status == "ok"
    quality[measured] = "unusable"
    quality[ok] = "ok"
    quality[ok & (_na_float(rec, "recording_own_speech_level_db") < -80.0)] = "silent_during_own_speech"
    quality[ok & (_na_float(rec, "audio_envelope_dynamics_db") < 1.5)] = "dead"
    rec["audio_quality"] = quality

    rule_pairs = [posture.posture_rule(v, a) if m else ("not_measured", "not_applicable")
                  for v, a, m in zip(rec["vendor"], rec["smplh_anamorphic"], measured)]
    rec["posture_rule"] = [r for r, _ in rule_pairs]
    rec["posture_rule_evidence"] = [e for _, e in rule_pairs]
    rec["expressivity_reference_group"] = [
        expressivity.reference_group(v, r, a, bool(m))
        for v, r, a, m in zip(rec["vendor"], rec["raster_class"], rec["smplh_anamorphic"], measured)
    ]

    # face / moi status for unmeasured rows come from Meta's flags, never NA
    face_status = rec.get("face_features_status", pd.Series(None, index=rec.index)).astype(object)
    rec["face_features_status"] = face_status.where(measured, "not_measured")
    parties_1p = rec["has_annotation_1p"].astype(bool)
    parties_3p = rec["has_annotation_3p"].astype(bool)
    flag_status = np.where(parties_1p & parties_3p, "annotated_1p_3p",
                           np.where(parties_3p, "annotated_3p", "not_annotated"))
    moi_status = rec.get("moi_status", pd.Series(None, index=rec.index)).astype(object)
    rec["moi_status"] = moi_status.where(measured & moi_status.notna(), flag_status)

    # recording-level posture: from all of the recording's bins
    agg = aggregate_posture(bins, rec, rules, by="file_id")
    agg = agg.add_prefix("recording_").rename(columns={"recording_file_id": "file_id"})
    rec = rec.merge(agg, on="file_id", how="left")
    unmeasured_posture = rec["recording_posture"].isna()
    rec.loc[unmeasured_posture, "recording_posture"] = "unknown"
    for st in ("standing", "sitting", "unclear"):
        rec.loc[unmeasured_posture & measured, f"recording_posture_{st}_frac"] = 0.0
    rec.loc[unmeasured_posture & measured, "recording_posture_unobserved_frac"] = 1.0
    rec.loc[unmeasured_posture & measured, "recording_posture_transitions"] = 0
    return rec


def aggregate_posture(bins: pd.DataFrame, recordings: pd.DataFrame, rules: posture.PostureRules,
                      *, by: str, clip_seconds: float | None = None) -> pd.DataFrame:
    """Classify 1-s bins with each recording's rule, then aggregate per ``by`` group."""

    if bins is None or bins.empty:
        return pd.DataFrame(columns=[by])
    frame = bins.merge(recordings[["file_id", "posture_rule"]], on="file_id", how="left")
    hip = _na_float(frame, "hip_flexion_deg").to_numpy()
    knee = _na_float(frame, "knee_between").to_numpy()
    rule = frame["posture_rule"].to_numpy(dtype=object)
    state = posture.classify_bins(rule, hip, knee, rules)
    # Refinement from the visual QA (configs/validation/posture_visual_qa_2026-09-24.csv):
    # the unvalidated rule never decides "sitting". All 24 sampled V02 clips it had
    # labelled sitting — 12 decided by the hip angle alone (knee-cropped fits read
    # 112-120 deg) and 12 by hip and knee together — were people standing. Such bins
    # are "unclear"; only V00 and V03, whose thresholds rest on hand labels, can read
    # sitting. (Lives here because rules are applied at annotate time; it belongs in
    # posture.classify_bins when the scan is next re-run.)
    unvalidated_sitting = (rule == "unvalidated_hip_knee") & (np.asarray(state, dtype=object) == "sitting")
    state = np.where(unvalidated_sitting, "unclear", state)
    frame["state"] = state
    if by == "clip_id":
        # group on integers (29M bins at corpus scale), then map back to clip ids
        n_clips = frame["file_id"].map(recordings.set_index("file_id")["recording_n_clips"])
        last = pd.to_numeric(n_clips, errors="coerce").fillna(1).astype(int).to_numpy() - 1
        index = np.floor(frame["bin_index"].to_numpy(dtype=float) / float(clip_seconds)).astype(np.int64)
        index = np.clip(index, 0, np.maximum(last, 0))
        codes, files = pd.factorize(frame["file_id"])
        frame["_group"] = codes.astype(np.int64) * 100_000 + index
        frame = frame.sort_values(["_group", "bin_index"], kind="stable")
        out = posture.aggregate_groups(frame, "_group", "state", rules)
        group = out["_group"].to_numpy(dtype=np.int64)
        out["clip_id"] = [make_clip_id(files[g // 100_000], int(g % 100_000), clip_seconds) for g in group]
        return out.drop(columns=["_group"])
    frame = frame.sort_values([by, "bin_index"], kind="stable")
    return posture.aggregate_groups(frame, by, "state", rules)


# ============================================================================= interactions
def interaction_basics(catalog: pd.DataFrame, rec: pd.DataFrame) -> pd.DataFrame:
    """Pair facts the clip links depend on: who was measured, and do they align in time."""

    inter = catalog.copy()
    frame = rec.assign(_measured=rec["measurement_status"] == "measured",
                       _drift=rec["timebase_status"] == "drift",
                       _none=rec["speech_source"] == "none")
    by = frame.groupby("interaction_key")
    stats = pd.DataFrame({
        "interaction_n_members_measured": by["_measured"].sum(),
        "interaction_duration_s": by["recording_duration_s"].max(),
        "_any_drift": by["_drift"].any(),
        "_speech_none": by["_none"].sum(),
        "_fps_n": by["fps"].agg(lambda s: s.dropna().round(6).nunique()),
    }).reset_index()
    inter = inter.merge(stats, on="interaction_key", how="left", validate="one_to_one")
    both = (inter["interaction_n_members_measured"] == 2) & inter["member_file_id_2"].notna()
    duration = rec.set_index("file_id")["recording_duration_s"]
    d1 = pd.to_numeric(inter["member_file_id_1"].map(duration), errors="coerce")
    d2 = pd.to_numeric(inter["member_file_id_2"].map(duration), errors="coerce")
    inter["pair_duration_diff_s"] = (d1 - d2).abs().where(both)
    inter["pair_fps_differs"] = both & (inter["_fps_n"] > 1)
    agreement = pd.Series("not_applicable", index=inter.index, dtype=object)
    agreement[both] = "agree"
    agreement[both & (inter["pair_duration_diff_s"] > 0.1)] = "minor_mismatch"
    agreement[both & ((inter["pair_duration_diff_s"] > 1.0) | inter["_any_drift"].astype(bool))] = "major_mismatch"
    inter["pair_duration_agreement"] = agreement
    status = pd.Series("not_measured", index=inter.index, dtype=object)
    status[inter["pairing_status"] == "partner_missing"] = "partner_missing"
    status[both & (inter["_speech_none"] == 0)] = "both_annotated"
    status[both & (inter["_speech_none"] == 1)] = "one_unannotated"
    status[both & (inter["_speech_none"] == 2)] = "both_unannotated"
    inter["speech_status"] = status
    return inter.drop(columns=["_any_drift", "_speech_none", "_fps_n"])


PAIR_SPEECH = ("interaction_speech_overlap_frac", "interaction_mutual_silence_frac",
               "interaction_speech_balance", "interaction_turn_switch_rate_per_min")


def pair_speech_from_windows(windows: pd.DataFrame, clips: pd.DataFrame) -> pd.DataFrame:
    """Conversation speech measures over the windows where both members' speech is annotated.

    The windows partition the shared interval, and each carries its mutual-silence
    share and its both/either overlap share, so the conversation's shares follow
    exactly (both = overlap x either, either = duration x (1 - silence)); floor
    changes are summed within windows (a change across a window boundary is not
    counted). Restricting to annotated windows keeps a VAD gap or a short WAV in one
    member from reading as the pair's silence.
    """

    usable = windows.loc[windows["window_mutual_silence_frac"].notna()].copy()
    usable["dur"] = _na_float(usable, "window_end_s") - _na_float(usable, "window_start_s")
    silence = _na_float(usable, "window_mutual_silence_frac")
    usable["neither"] = silence * usable["dur"]
    usable["either"] = usable["dur"] - usable["neither"]
    usable["both"] = _na_float(usable, "window_speech_overlap_frac").fillna(0.0) * usable["either"]
    usable["switches"] = _na_float(usable, "window_turn_switches").fillna(0.0)
    own = clips.set_index("clip_id")["own_speech_s"]
    usable["own_1"] = usable["member_clip_id_1"].map(own)
    usable["own_2"] = usable["member_clip_id_2"].map(own)
    g = usable.groupby("interaction_key")[["dur", "neither", "either", "both", "switches", "own_1", "own_2"]].sum()
    out = pd.DataFrame(index=g.index)
    out["interaction_mutual_silence_frac"] = g["neither"] / g["dur"]
    out["interaction_speech_overlap_frac"] = (g["both"] / g["either"]).where(g["either"] > 0)
    high = g[["own_1", "own_2"]].max(axis=1)
    out["interaction_speech_balance"] = (g[["own_1", "own_2"]].min(axis=1) / high).where(high > 0)
    out["interaction_turn_switch_rate_per_min"] = g["switches"] / g["dur"] * 60.0
    out["_covered_s"] = g["dur"]
    return out.reset_index()


def complete_interactions(inter: pd.DataFrame, rec: pd.DataFrame, pairs: pd.DataFrame,
                          clips: pd.DataFrame, windows: pd.DataFrame) -> pd.DataFrame:
    """Conversation-level summaries, each computed from both members (NA if one is missing)."""

    inter = inter.merge(pairs.drop(columns=[c for c in PAIR_SPEECH if c in pairs.columns]),
                        on="interaction_key", how="left", validate="one_to_one")
    speech = pair_speech_from_windows(windows, clips)
    inter = inter.merge(speech, on="interaction_key", how="left", validate="one_to_one")
    inter["interaction_pair_speech_coverage_frac"] = (
        _na_float(inter, "_covered_s") / _na_float(inter, "interaction_duration_s")).clip(0, 1)
    inter = inter.drop(columns=["_covered_s"])
    for column in PAIR_SPEECH + ("interaction_pair_speech_coverage_frac",):
        _ensure(inter, column)

    counts = clips.groupby("interaction_key").size()
    inter["interaction_n_clips"] = inter["interaction_key"].map(counts).fillna(0).astype(int)

    by_file = rec.set_index("file_id")
    has_partner = inter["member_file_id_2"].notna()
    p1 = inter["member_file_id_1"].map(by_file["recording_posture"]).fillna("unknown")
    p2 = inter["member_file_id_2"].map(by_file["recording_posture"]).fillna("unknown").where(has_partner, "absent")
    inter["posture_pair"] = ["+".join(sorted((a, b))) for a, b in zip(p1, p2)]
    t1 = pd.to_numeric(inter["member_file_id_1"].map(by_file["recording_posture_transitions"]), errors="coerce")
    t2 = pd.to_numeric(inter["member_file_id_2"].map(by_file["recording_posture_transitions"]), errors="coerce")
    transition = pd.Series("unknown", index=inter.index, dtype=object)
    transition[p1.isin(DECIDED) & p2.isin(DECIDED)] = "no"
    transition[(t1 >= 1) | (t2 >= 1)] = "yes"
    inter["any_member_posture_transition"] = transition

    scored = clips.loc[clips["expressivity_status"] == "measured",
                       ["interaction_key", "file_id", "expressivity_score", "clip_seconds"]]
    weighted = scored.assign(w=scored["expressivity_score"] * scored["clip_seconds"])
    per_member = weighted.groupby(["interaction_key", "file_id"]).agg(w=("w", "sum"), s=("clip_seconds", "sum"))
    per_member["mean"] = per_member["w"] / per_member["s"]
    spread = per_member.groupby("interaction_key")["mean"].agg(["size", "min", "max"])
    totals = per_member.groupby("interaction_key")[["w", "s"]].sum()
    two = spread["size"] == 2
    inter["interaction_expressivity_mean"] = inter["interaction_key"].map((totals["w"] / totals["s"]).where(two))
    inter["interaction_expressivity_gap"] = inter["interaction_key"].map((spread["max"] - spread["min"]).where(two))

    both_measured = inter["interaction_n_members_measured"] == 2
    for party in ("3p", "1p"):
        column = f"moi_duplication_{party}"
        _ensure(inter, column, None)
        covered = inter[f"moi_{party}_coverage"] == "both"
        value = inter[column].where(covered & inter[column].notna(), "not_applicable")
        inter[column] = value.where(~covered | both_measured, "not_measured")
    moi_sum = rec.groupby("interaction_key")["recording_moi_3p_count"].sum(min_count=2)
    inter["interaction_moi_3p_count"] = inter["interaction_key"].map(moi_sum).where(
        (inter["moi_3p_coverage"] == "both") & both_measured)
    return inter


# ============================================================================= clips
def build_clips(scanned: pd.DataFrame, rec: pd.DataFrame, inter: pd.DataFrame, bins: pd.DataFrame,
                rules: posture.PostureRules, clip_seconds: float,
                reference: dict | None = None) -> tuple[pd.DataFrame, dict]:
    keys = rec[["file_id", "interaction_key", "session_key", "participant_key", "vendor", "label", "split",
                "partner_file_id", "speech_source", "smplh_anamorphic", "raster_class",
                "expressivity_reference_group", "recording_n_clips", "posture_rule",
                "speech_annotated_until_s"]]
    clips = scanned.merge(keys, on="file_id", how="left", validate="many_to_one")
    # Speech-conditioned values are unknown where speech is unannotated: the
    # recording has no VAD or transcript, or the clip reaches past the audio the
    # annotation was derived from (a WAV shorter than the pose grid).
    words = _na_float(clips, "word_count")
    status = pd.Series("annotated", index=clips.index, dtype=object)
    status[(words >= VAD_GAP_MIN_WORDS) & (words > VAD_GAP_WORDS_PER_S * _na_float(clips, "own_speech_s").fillna(0))] = "vad_gap"
    status[_na_float(clips, "end_s") > _na_float(clips, "speech_annotated_until_s") + ANNOTATED_SPAN_TOLERANCE_S] = "beyond_audio"
    status[clips["speech_source"] == "none"] = "no_speech_annotation"
    clips["speech_annotation_status"] = status
    speech_unknown = status != "annotated"
    for column in SPEECH_CONDITIONED:
        if column in clips.columns:
            clips[column] = clips[column].where(~speech_unknown)
    clips["window_id"] = [make_window_id(i, int(k), clip_seconds) for i, k in zip(clips["interaction_key"], clips["clip_index"])]

    # ---- partner links ---------------------------------------------------
    agreement = clips["interaction_key"].map(inter.set_index("interaction_key")["pair_duration_agreement"])
    partner_status = rec.set_index("file_id")["measurement_status"]
    partner_measured = clips["partner_file_id"].map(partner_status) == "measured"
    candidate = [make_clip_id(p, int(k), clip_seconds) if isinstance(p, str) else None
                 for p, k in zip(clips["partner_file_id"], clips["clip_index"])]
    candidate = pd.Series(candidate, index=clips.index, dtype=object)
    exists = candidate.isin(set(clips["clip_id"]))
    aligned = agreement.isin(["agree", "minor_mismatch"])
    link = pd.Series("linked", index=clips.index, dtype=object)
    link[~exists] = "partner_has_no_clip"
    link[~aligned] = "not_time_aligned"
    link[~partner_measured] = "partner_not_measured"
    link[clips["partner_file_id"].isna()] = "partner_missing"
    clips["partner_link_status"] = link
    clips["partner_clip_id"] = candidate.where(link == "linked")

    # ---- speaking role -------------------------------------------------------
    own = _na_float(clips, "own_speech_frac")
    partner_own = clips["partner_clip_id"].map(clips.set_index("clip_id")["own_speech_frac"])
    partner_own = pd.to_numeric(partner_own, errors="coerce")
    partner_source = clips["partner_file_id"].map(rec.set_index("file_id")["speech_source"])
    known = ((clips["partner_link_status"] == "linked") & (clips["speech_source"] != "none")
             & (partner_source != "none") & own.notna() & partner_own.notna())
    role = pd.Series("both", index=clips.index, dtype=object)
    role[(own >= 0.1) & (own >= 2 * partner_own)] = "speaking"
    role[(partner_own >= 0.1) & (partner_own >= 2 * own)] = "listening"
    role[(own < 0.1) & (partner_own < 0.1)] = "silent"
    role[~known] = "unknown"
    clips["speaking_role"] = role

    # ---- framing labels ------------------------------------------------------
    extent = pd.Series("unknown", index=clips.index, dtype=object)
    extent[_na_float(clips, "shoulders_visible_frac") >= 0.8] = "upper_body"
    extent[_na_float(clips, "hips_visible_frac") >= 0.8] = "to_hips"
    extent[_na_float(clips, "knees_visible_frac") >= 0.8] = "to_knees"
    extent[_na_float(clips, "ankles_visible_frac") >= 0.8] = "full_body"
    clips["visible_extent"] = extent
    facing = _na_float(clips, "facing_angle_deg_p50")
    direction = pd.Series("unknown", index=clips.index, dtype=object)
    direction[facing < 25] = "toward_camera"
    direction[(facing >= 25) & (facing <= 60)] = "angled"
    direction[facing > 60] = "side_on"
    clips["facing_direction"] = direction

    # ---- posture --------------------------------------------------------------
    agg = aggregate_posture(bins, rec, rules, by="clip_id", clip_seconds=clip_seconds)
    clips = clips.merge(agg, on="clip_id", how="left")
    no_bins = clips["posture"].isna()
    clips.loc[no_bins, "posture"] = "unknown"
    for st in ("standing", "sitting", "unclear"):
        clips.loc[no_bins, f"posture_{st}_frac"] = 0.0
    clips.loc[no_bins, "posture_unobserved_frac"] = 1.0
    clips.loc[no_bins, "posture_transitions"] = 0

    # ---- expressivity ---------------------------------------------------------
    if reference is None:
        reference = expressivity.build_reference(clips)
    scored = expressivity.score(clips, reference, clip_seconds_nominal=clip_seconds)
    for column in scored.columns:
        clips[column] = scored[column]

    # ---- redundancy / representativeness ---------------------------------------
    clips = add_distances(clips)
    return clips, reference


DISTANCE_FEATURES = ("expr_energy", "expr_amplitude", "expr_head", "expr_hands",
                     "wrist_height_p75_mm", "hip_flexion_deg_p50", "facing_angle_deg_p50", "own_speech_frac")


def add_distances(clips: pd.DataFrame, features: tuple[str, ...] = DISTANCE_FEATURES,
                  min_dims: int = 4) -> pd.DataFrame:
    out = clips.sort_values(["file_id", "clip_index"], kind="stable").copy()
    reference = out.loc[~out["is_partial"].astype(bool)]
    z = {}
    for name in features:
        values = _na_float(out, name)
        ref = _na_float(reference, name)
        q1, q3 = np.nanpercentile(ref, [25, 75]) if ref.notna().any() else (np.nan, np.nan)
        scale = (q3 - q1) / 1.349 if q3 == q3 and q3 > q1 else np.nan
        z[name] = (values - np.nanmedian(ref)) / scale if scale == scale else values * np.nan
    matrix = pd.DataFrame(z, index=out.index)
    previous = matrix.groupby(out["file_id"]).shift(1)
    diff = (matrix - previous) ** 2
    n = diff.notna().sum(axis=1)
    out["distance_to_previous_clip_z"] = np.sqrt(diff.mean(axis=1)).where(n >= min_dims)
    means = matrix.groupby(out["file_id"]).transform("mean")
    # A feature only informs "distance to the recording mean" where at least two
    # clips of the recording have it; otherwise the lone clip IS the mean (0).
    finite_per_feature = matrix.notna().groupby(out["file_id"]).transform("sum")
    diff_mean = ((matrix - means) ** 2).where(finite_per_feature >= 2)
    n_mean = diff_mean.notna().sum(axis=1)
    featured = (matrix.notna().sum(axis=1) >= min_dims)
    featured_clips = featured.groupby(out["file_id"]).transform("sum")
    out["distance_to_recording_mean_z"] = np.sqrt(diff_mean.mean(axis=1)).where(
        (n_mean >= min_dims) & (featured_clips >= 2))
    return out.loc[clips.index]


# ============================================================================= windows, sessions, ...
def build_windows(scanned: pd.DataFrame, inter: pd.DataFrame, clips: pd.DataFrame) -> pd.DataFrame:
    win = scanned.merge(inter[["interaction_key", "vendor", "label", "split", "speech_status",
                               "pair_duration_agreement"]], on="interaction_key", how="left")
    annotated = clips.set_index("clip_id")["own_speech_frac"].notna()
    # Both members' clip k stay listed whatever the pair's state: the row says which
    # clip of which member has index k. Pair measures need a time-aligned pair.
    aligned = win["pair_duration_agreement"].isin(["agree", "minor_mismatch"])
    usable = ((win["speech_status"] == "both_annotated") & aligned
              & win["member_clip_id_1"].map(annotated).astype("boolean").fillna(False).astype(bool)
              & win["member_clip_id_2"].map(annotated).astype("boolean").fillna(False).astype(bool))
    for column in ("window_speech_overlap_frac", "window_mutual_silence_frac", "window_turn_switches"):
        _ensure(win, column)
        win[column] = _na_float(win, column).where(usable)
    posture_of = clips.set_index("clip_id")["posture"]
    a = win["member_clip_id_1"].map(posture_of).where(win["member_clip_id_1"].notna(), "absent").fillna("absent")
    b = win["member_clip_id_2"].map(posture_of).where(win["member_clip_id_2"].notna(), "absent").fillna("absent")
    win["window_posture_pair"] = ["+".join(sorted((x, y))) for x, y in zip(a, b)]
    return win


def build_sessions(catalog: pd.DataFrame, rec: pd.DataFrame, inter: pd.DataFrame) -> pd.DataFrame:
    ses = catalog.copy()
    duration = inter.groupby("session_key")["interaction_duration_s"].sum(min_count=1)
    ses["session_duration_s"] = ses["session_key"].map(duration)

    def person_verdict(labels: pd.Series) -> str:
        decided = labels[labels.isin(DECIDED)]
        if (labels == "mixed").any() or decided.nunique() > 1:
            return "no"
        return "yes" if len(decided) >= 2 else "unknown"

    verdicts = rec.groupby(["session_key", "participant_key"])["recording_posture"].agg(person_verdict)

    def session_verdict(values: pd.Series) -> str:
        if (values == "no").any():
            return "no"
        return "yes" if (values == "yes").all() else "unknown"

    ses["session_posture_consistent"] = ses["session_key"].map(
        verdicts.groupby(level="session_key").agg(session_verdict)).fillna("unknown")
    return ses


def build_participants(catalog: pd.DataFrame, rec: pd.DataFrame) -> pd.DataFrame:
    par = catalog.copy()
    hours = rec.loc[rec["measurement_status"] == "measured"].groupby("participant_key")["recording_duration_s"].sum() / 3600.0
    par["participant_recorded_hours"] = par["participant_key"].map(hours).fillna(0.0)
    return par


def build_moi(scanned: pd.DataFrame, rec: pd.DataFrame) -> pd.DataFrame:
    if scanned is None or scanned.empty:
        return pd.DataFrame(columns=schema.columns("moi_events"))
    keys = rec[["file_id", "interaction_key", "participant_key", "vendor", "label", "split", "recording_duration_s"]]
    moi = scanned.merge(keys, on="file_id", how="left", validate="many_to_one")
    # An event that lies wholly at or after the recording's end is in no clip.
    duration = _na_float(moi, "recording_duration_s")
    low = np.minimum(_na_float(moi, "moi_start_s"), _na_float(moi, "moi_end_s"))
    outside = ~(low < duration)
    for column in ("moi_clip_index_start", "moi_clip_index_end"):
        moi[column] = moi[column].where(~outside)
    return moi.drop(columns=["recording_duration_s"])


def merge_visual(clips: pd.DataFrame, video: pd.DataFrame | None) -> pd.DataFrame:
    columns = [f.name for f in schema.fields("clips") if f.source == "MP4 keyframe"] + ["frame_time_s"]
    out = clips.drop(columns=[c for c in columns + ["visual_status"] if c in clips.columns])
    if video is None or video.empty:
        out["visual_status"] = "not_run"
        for column in columns:
            out[column] = np.nan
        return out
    keep = ["clip_id", "visual_status"] + [c for c in columns if c in video.columns]
    out = out.merge(video[keep], on="clip_id", how="left")
    out["visual_status"] = out["visual_status"].fillna("not_run")
    return out


# ============================================================================= entry point
def build_tables(catalog: Mapping[str, pd.DataFrame], scan: ScanTables, *, rules: posture.PostureRules,
                 clip_seconds: float, video: pd.DataFrame | None = None,
                 reference: dict | None = None) -> tuple[dict[str, pd.DataFrame], dict]:
    """All eight tables from the catalog and the gathered scan (pure; no IO)."""

    rec = build_recordings(catalog["recordings"], scan.recordings, rules, scan.bins)
    basics = interaction_basics(catalog["interactions"], rec)
    clips, reference = build_clips(scan.clips, rec, basics, scan.bins, rules, clip_seconds, reference)
    clips = merge_visual(clips, video)
    windows = build_windows(scan.windows, basics, clips)
    inter = complete_interactions(basics, rec, scan.pairs, clips, windows)

    # recording-level expressivity summaries, recomputed from the recording's clips
    measured = clips.loc[clips["expressivity_status"] == "measured"]
    grouped = measured.assign(w=measured["expressivity_score"] * measured["clip_seconds"]).groupby("file_id")
    n_scored = rec["file_id"].map(grouped.size()).fillna(0)
    rec["recording_expressivity_mean"] = rec["file_id"].map(grouped["w"].sum() / grouped["clip_seconds"].sum())
    rec["recording_expressivity_clip_sd"] = rec["file_id"].map(grouped["expressivity_score"].std(ddof=1)).where(n_scored >= 2)
    ranks = measured[["file_id", "clip_index", "expressivity_score"]].copy()
    ranks["r_score"] = ranks.groupby("file_id")["expressivity_score"].rank()
    ranks["r_index"] = ranks.groupby("file_id")["clip_index"].rank()
    if len(ranks):
        trend = ranks.groupby("file_id")[["r_score", "r_index"]].corr().xs("r_score", level=1)["r_index"]
    else:
        trend = pd.Series(dtype=float)
    rec["recording_expressivity_trend"] = rec["file_id"].map(trend).where(n_scored >= 3)

    tables = {
        "participants": build_participants(catalog["participants"], rec),
        "sessions": build_sessions(catalog["sessions"], rec, inter),
        "prompts": catalog["prompts"],
        "interactions": inter,
        "interaction_windows": windows,
        "recordings": rec,
        "clips": clips.sort_values("clip_id", kind="stable"),
        "moi_events": build_moi(scan.moi, rec),
    }
    return {name: schema.conform(frame, name) for name, frame in tables.items()}, reference


def write_tables(tables: Mapping[str, pd.DataFrame], directory: Path, *, side_files: Mapping[str, Any],
                 run_hash: str, provenance: Mapping[str, Any]) -> dict[str, Any]:
    """Publish atomically: write a versioned sibling directory, then swap a symlink.

    ``directory`` (e.g. ``annotations``) is a symlink to ``.annotations-<run_hash>``;
    ``os.replace`` of the link is a single atomic step, so a reader always sees one
    complete run, and a crash leaves the previous run published.
    """

    import pyarrow.parquet as pq

    directory = Path(directory)
    parent = directory.parent
    parent.mkdir(parents=True, exist_ok=True)
    version = parent / f".{directory.name}-{run_hash}"
    staging = parent / f".{directory.name}-{run_hash}.staging-{os.getpid()}"
    for stale in (staging, version):
        if stale.exists():
            shutil.rmtree(stale)
    staging.mkdir(mode=0o700)
    manifest: dict[str, Any] = {"run_hash": run_hash, "tables": {}}
    for name, frame in tables.items():
        table = schema.to_arrow(frame, name)
        table = table.replace_schema_metadata({**(table.schema.metadata or {}),
                                               b"seamless_curation.run_hash": run_hash.encode()})
        path = staging / f"{name}.parquet"
        pq.write_table(table, path, compression="zstd")
        manifest["tables"][name] = {"rows": int(len(frame)), "columns": int(frame.shape[1]),
                                    "sha256": sha256(path.read_bytes()).hexdigest()}
    for name, payload in side_files.items():
        path = staging / name
        if isinstance(payload, pd.DataFrame):
            payload.to_csv(path, index=False)
        else:
            path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str), encoding="utf-8")
    (staging / "provenance.json").write_text(json.dumps({**dict(provenance), **manifest},
                                                        indent=2, sort_keys=True, default=str), encoding="utf-8")
    os.chmod(staging, 0o700)
    os.replace(staging, version)
    previous = os.readlink(directory) if directory.is_symlink() else None
    if directory.exists() and not directory.is_symlink():  # a plain directory from an older layout
        os.replace(directory, parent / f".{directory.name}-legacy-{int(time.time())}")
    link = parent / f".{directory.name}.link-{os.getpid()}"
    if link.is_symlink() or link.exists():
        link.unlink()
    os.symlink(version.name, link)
    os.replace(link, directory)
    if previous and previous != version.name and (parent / previous).exists():
        shutil.rmtree(parent / previous)
    return manifest

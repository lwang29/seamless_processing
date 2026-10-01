"""``annotation_report.md``: what the tables contain and how far each label can be trusted.

Four kinds of content, all computed from the published tables (never from the
scan directly), so the report describes exactly what a downstream user loads:

1. **Coverage** — rows per level, measurement status, clips per vendor.
2. **Distributions** — every categorical annotation by vendor, because rigs
   differ and a corpus-wide share hides that.
3. **Validation evidence** — posture against the 66 hand-labelled V00 files;
   expressivity against the dataset's own signals (Meta's 3P moments of
   interest, Imitator facial-action variability) and against the previous
   iteration's human gesture verdicts; the correlation structure of the
   expressivity channels; and, while the previous run's window table still
   exists, an exact cross-check that every measure reused from the old pipeline
   reproduces its old value.
4. **The validation report** of ``annotate`` (structural checks).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from .config import RunConfig
from .dataset import load_tables

LABELS_V00 = Path("configs/validation/v00_posture_labels.csv")

#: reused clip column -> old windows.parquet column (the prior:reused cross-check)
REUSED_COLUMNS = {
    "arm_speed_p50_mm_s": "arm_speed_p50_mm_s", "arm_speed_speech_p50_mm_s": "arm_speed_speech_p50_mm_s",
    "wrist_range_mm": "wrist_range_mm", "wrist_excursion_p90_mm": "wrist_excursion_p90_mm",
    "elbow_range_mm": "elbow_range_mm", "elbow_excursion_p90_mm": "elbow_excursion_p90_mm",
    "kp_conf_p10": "kp_conf_p10", "consistency_r": "consistency_r", "step_cosine_p50": "step_cosine_p50",
    "smplh_valid_frac": "smplh_valid_frac", "smplh_longest_invalid_s": "smplh_longest_invalid_s",
    "subject_present_frac": "box_valid_frac", "hand_frozen_frac": "hand_frozen_frac",
    "arm_active_frac": "gesture_frac", "arm_active_frac_speech": "gesture_frac_speech",
    "arm_active_frac_silence": "gesture_frac_silence", "arm_episode_count": "episode_count",
    "spine_motion_mm_s_p50": "torso_travel_mm_s_p50", "speech_motion_sync_r": "sync_r",
    "speech_motion_sync_lag_s": "sync_lag_s", "speech_segment_count": "speech_segment_count",
    "speech_segments_with_motion_frac": "speech_segments_covered",
}
#: adapted columns compared for information only (definitions changed on purpose)
ADAPTED_COLUMNS = {
    "wrist_height_speech_p75_mm": "wrist_height_p75_mm", "hand_artic_p75_rad_s": "hand_artic_p75_rad_s",
    "arm_episode_median_s": "episode_median_s",
}


def _table(frame: pd.DataFrame, floatfmt: str = ".3f") -> str:
    if frame.empty:
        return "_(empty)_"
    frame = frame.copy()
    header = "| " + " | ".join([str(frame.index.name or "")] + [str(c) for c in frame.columns]) + " |"
    rule = "|" + "---|" * (len(frame.columns) + 1)
    lines = [header, rule]
    # format per column (iterrows would upcast a row of ints and floats to floats)
    integral = {c: pd.api.types.is_integer_dtype(frame[c]) or c in ("n", "rows", "compared", "NA mismatch",
                                                                  "differ", "with MOI")
                for c in frame.columns}
    for index, row in zip(frame.index, frame.itertuples(index=False)):
        cells = []
        for column, value in zip(frame.columns, row):
            if value is None or (isinstance(value, (float, np.floating)) and np.isnan(value)):
                cells.append("")
            elif integral[column] and isinstance(value, (int, float, np.integer, np.floating)):
                cells.append(f"{int(round(float(value))):,}")
            elif isinstance(value, (float, np.floating)):
                cells.append(format(value, floatfmt))
            else:
                cells.append(str(value))
        lines.append("| " + " | ".join([str(index)] + cells) + " |")
    return "\n".join(lines)


def _shares(frame: pd.DataFrame, column: str, by: str = "vendor", *, counts: bool = False) -> pd.DataFrame:
    """Share (or count, for rare categories) of each value by ``by``, plus the row count ``n``."""

    table = pd.crosstab(frame[by], frame[column], normalize=False if counts else "index")
    table.columns = [str(c) for c in table.columns]
    table["n"] = frame.groupby(by).size()
    table.index.name = by
    return table


def auc(positive: Iterable[float], negative: Iterable[float]) -> float:
    """Mann-Whitney AUC (ties count half); NaN if either side is empty."""

    pos = np.asarray([v for v in positive if v == v], dtype=float)
    neg = np.asarray([v for v in negative if v == v], dtype=float)
    if not len(pos) or not len(neg):
        return float("nan")
    ranks = pd.Series(np.concatenate([pos, neg])).rank().to_numpy()
    return float((ranks[: len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def partial_spearman(x: pd.Series, y: pd.Series, z: pd.Series) -> float:
    frame = pd.DataFrame({"x": x, "y": y, "z": z}).dropna().rank()
    if len(frame) < 10:
        return float("nan")
    r = frame.corr()
    rxy, rxz, ryz = r.loc["x", "y"], r.loc["x", "z"], r.loc["y", "z"]
    return float((rxy - rxz * ryz) / np.sqrt(max(1e-12, (1 - rxz ** 2) * (1 - ryz ** 2))))


# ============================================================================= sections
def section_coverage(tables: dict[str, pd.DataFrame]) -> str:
    rec, clips = tables["recordings"], tables["clips"]
    rows = pd.DataFrame({"rows": {t: len(f) for t, f in tables.items()}})
    rows.index.name = "table"
    status = pd.crosstab(rec["vendor"], rec["measurement_status"])
    status.index.name = "vendor"
    per_vendor = clips.groupby("vendor").agg(clips=("clip_id", "size"), hours=("clip_seconds", lambda s: s.sum() / 3600),
                                             partial=("is_partial", "sum"))
    return "\n\n".join([
        "## 1. Coverage", _table(rows), "Recordings by measurement status:", _table(status),
        "Clips (every clip of every measured recording; nothing excluded):", _table(per_vendor, ".1f"),
    ])


def section_distributions(tables: dict[str, pd.DataFrame]) -> str:
    clips, rec, inter = tables["clips"], tables["recordings"], tables["interactions"]
    parts = ["## 2. Distributions by vendor"]
    for column in ("posture", "expressivity_level", "expressivity_level_pooled", "speaking_role",
                   "visible_extent", "facing_direction", "partner_link_status", "visual_status"):
        parts += [f"**clips.{column}**", _table(_shares(clips, column))]
    for column in ("recording_posture", "posture_rule", "audio_quality", "timebase_status", "speech_source",
                   "face_features_status", "moi_status", "raster_class"):
        parts += [f"**recordings.{column}**", _table(_shares(rec.assign(**{column: rec[column].fillna("NA")}), column))]
    transitions = rec.assign(changed=rec["recording_posture_transitions"].fillna(0) >= 1).groupby("vendor")["changed"].mean()
    parts += ["Share of recordings with a sustained standing<->sitting change:",
              _table(transitions.to_frame("share"))]
    for column in ("pair_duration_agreement", "speech_status", "any_member_posture_transition"):
        parts += [f"**interactions.{column}**", _table(_shares(inter, column))]
    for column in ("moi_duplication_3p", "moi_duplication_1p", "moi_3p_coverage"):
        parts += [f"**interactions.{column}** (counts)", _table(_shares(inter, column, counts=True))]
    measured = clips.loc[clips["visual_status"] == "measured"]
    if len(measured):
        visual = measured.merge(rec[["file_id", "raster_class"]], on="file_id").groupby("raster_class")[
            ["frame_sharpness", "frame_luma_mean", "frame_luma_clipped_frac", "background_edge_density",
             "camera_shift_px"]].median()
        visual["n"] = measured.merge(rec[["file_id", "raster_class"]], on="file_id").groupby("raster_class").size()
        visual.index.name = "raster_class"
        parts += ["Pixel pass, medians by raster (sharpness and clutter compare only within a raster; "
                  f"camera shift > 5 px in {float((measured['camera_shift_px'] > 5).mean()):.2%} of clips):",
                  _table(visual, ".3g")]
    top = inter["posture_pair"].value_counts().head(10).to_frame("interactions")
    top.index.name = "posture_pair"
    parts += ["Most common posture pairs:", _table(top)]
    return "\n\n".join(parts)


def section_posture_validation(tables: dict[str, pd.DataFrame]) -> str:
    rec = tables["recordings"]
    parts = ["## 3. Validation evidence", "### 3.1 Posture against the 66 hand-labelled V00 files",
             "Labels: `configs/validation/v00_posture_labels.csv` (file level; drawn from flagged and "
             "adjudication pools of the retired FM1 detector, so they cluster near its cut and overstate "
             "difficulty relative to a random draw). Bin- and clip-level accuracy was never labelled."]
    if not LABELS_V00.exists():
        return "\n\n".join(parts + ["_(label file missing)_"])
    labels = pd.read_csv(LABELS_V00, comment="#")
    joined = labels.merge(rec[["file_id", "recording_posture", "recording_posture_confidence",
                               "recording_knee_between_p50"]], on="file_id", how="left")
    confusion = pd.crosstab(joined["label"], joined["recording_posture"].fillna("absent"))
    confusion.index.name = "hand label"
    decided = joined["recording_posture"].isin(["standing", "sitting"])
    correct = (joined["recording_posture"] == joined["label"]) & decided
    parts += [_table(confusion),
              f"Decided on {int(decided.sum())} of {len(joined)} files; {int(correct.sum())} of those agree "
              f"with the hand label ({correct.sum() / max(1, decided.sum()):.3f}). The rest are 'mixed', "
              "'unclear' or 'unknown' by design (the band between the prior thresholds)."]
    return "\n\n".join(parts)


def section_expressivity_validation(tables: dict[str, pd.DataFrame], legacy_dir: Path | None) -> str:
    clips, rec = tables["clips"], tables["recordings"]
    measured = clips.loc[clips["expressivity_status"] == "measured"]
    parts = ["### 3.2 Expressivity",
             "Expected directions, stated before looking: clips containing an observer-annotated moment of "
             "interest (3P MOI: behaviour deviating from the participant's baseline) should score higher "
             "than other clips of the same annotated recordings (AUC > 0.5); facial-action variability "
             "should correlate positively with body expressivity after controlling for speaking time."]
    channels = ["expr_energy", "expr_amplitude", "expr_head", "expr_hands", "expr_variability", "expr_face", "expr_vocal"]
    corr = measured[channels].corr(method="spearman")
    corr.index.name = "Spearman"
    parts += ["Correlation of the channels over measured clips (energy/amplitude/head/hands form the score):",
              _table(corr, ".2f")]
    annotated = measured.loc[measured["moi_3p_count"].notna()]
    rows = []
    for name, subset in (("all annotated", annotated), ("V00 only (3 partially copied pairs)", annotated[annotated["vendor"] == "V00"]),
                         ("V03 only", annotated[annotated["vendor"] == "V03"])):
        centred = subset["expressivity_score"] - subset.groupby("file_id")["expressivity_score"].transform("mean")
        has = subset["moi_3p_count"] > 0
        rows.append({"subset": name, "clips": len(subset), "with MOI": int(has.sum()),
                     "AUC": auc(subset.loc[has, "expressivity_score"], subset.loc[~has, "expressivity_score"]),
                     "AUC within recording": auc(centred[has], centred[~has])})
    table = pd.DataFrame(rows).set_index("subset")
    parts += ["3P MOI presence per clip (Mann-Whitney AUC of expressivity_score):", _table(table)]
    v00 = measured.loc[measured["vendor"] == "V00"]
    if len(v00):
        parts.append(
            f"V00 clips (n={len(v00):,}): Spearman(expressivity_score, fau_variability) = "
            f"{v00['expressivity_score'].corr(v00['fau_variability'], method='spearman'):.3f}; partial, "
            f"controlling own_speech_frac = {partial_spearman(v00['expressivity_score'], v00['fau_variability'], v00['own_speech_frac']):.3f}.")
    by_posture = measured.groupby(["vendor", "posture"])["expressivity_score"].median().unstack()
    by_posture.index.name = "vendor"
    by_role = measured.groupby(["vendor", "speaking_role"])["expressivity_score"].median().unstack()
    by_role.index.name = "vendor"
    parts += ["Median expressivity_score by posture (a confound check; seated people gesture differently):",
              _table(by_posture), "Median expressivity_score by speaking role:", _table(by_role)]
    if legacy_dir is not None and (legacy_dir / "gesture_review_spans.parquet").exists():
        verdicts = pd.read_parquet(legacy_dir / "gesture_review_verdicts.parquet")
        spans = pd.read_parquet(legacy_dir / "gesture_review_spans.parquet")
        human = verdicts.loc[verdicts["verdict_source"] == "human"].sort_values("recorded_utc").groupby("review_item_id").tail(1)
        human = human.loc[human["verdict"].isin(["accept", "reject"])]
        shown = spans.merge(human[["review_item_id", "verdict"]], on="review_item_id")
        rows = []
        for item, group in shown.groupby("review_item_id"):
            indices = set()
            for a, b in zip(group["new_clip_index_first"], group["new_clip_index_last"]):
                if a == a and b == b:
                    indices.update(range(int(a), int(b) + 1))
            file_clips = measured.loc[(measured["file_id"] == group["file_id"].iloc[0]) & measured["clip_index"].isin(indices)]
            rows.append({"verdict": group["verdict"].iloc[0], "score": file_clips["expressivity_score"].mean()})
        frame = pd.DataFrame(rows)
        if len(frame):
            parts.append(
                f"Previous iteration's human gesture verdicts ({(frame.verdict == 'accept').sum()} accept / "
                f"{(frame.verdict == 'reject').sum()} reject; all had passed the old tier-1 filter): AUC of the mean "
                f"expressivity_score over the judged spans = {auc(frame.loc[frame.verdict == 'accept', 'score'], frame.loc[frame.verdict == 'reject', 'score']):.3f}. "
                "A weak proxy: the verdict was 'good co-speech gesture training data', not expressivity.")
    return "\n\n".join(parts)


def section_reuse_crosscheck(tables: dict[str, pd.DataFrame], legacy_run_root: Path | None) -> str:
    parts = ["### 3.3 Measures reused from the previous pipeline"]
    windows_path = legacy_run_root / "windows.parquet" if legacy_run_root else None
    if windows_path is None or not windows_path.exists():
        return "\n\n".join(parts + ["_(previous window table no longer on disk; see the archived report)_"])
    wanted = sorted(set(REUSED_COLUMNS.values()) | set(ADAPTED_COLUMNS.values()))
    old = pd.read_parquet(windows_path, columns=["file_id", "start_frame", "end_frame", *wanted])
    clips = tables["clips"]
    rec = tables["recordings"][["file_id", "fps", "speech_source"]]
    new = clips.merge(rec, on="file_id").loc[lambda f: (f["fps"] == 30.0) & ~f["is_partial"]]
    # The speech mask is an adapted input (transcript fallback and tail, NA past the
    # annotated span), so speech-conditioned columns are compared only where it is
    # unchanged: VAD-sourced recordings, clips inside the annotated span.
    unchanged_speech = (new["speech_source"] == "vad") & new["own_speech_frac"].notna()
    speech_conditioned = {"arm_speed_speech_p50_mm_s", "arm_active_frac_speech", "arm_active_frac_silence",
                          "speech_motion_sync_r", "speech_motion_sync_lag_s", "speech_segment_count",
                          "speech_segments_with_motion_frac", "wrist_height_speech_p75_mm"}
    joined = new.merge(old, on=["file_id", "start_frame", "end_frame"], suffixes=("", "_old"))
    rows = []
    unchanged = joined["speech_source"].eq("vad") & joined["own_speech_frac"].notna()
    for new_col, old_col in {**REUSED_COLUMNS, **ADAPTED_COLUMNS}.items():
        subset = joined.loc[unchanged] if new_col in speech_conditioned else joined
        a = pd.to_numeric(subset[new_col], errors="coerce").to_numpy(dtype=float)
        old_name = f"{old_col}_old" if f"{old_col}_old" in subset.columns else old_col
        b = pd.to_numeric(subset[old_name], errors="coerce").to_numpy(dtype=float)
        both = np.isfinite(a) & np.isfinite(b)
        tol = 1e-5 * np.maximum(1.0, np.abs(b))
        rows.append({"column": new_col, "old column": old_col, "kind": "reused" if new_col in REUSED_COLUMNS else "adapted",
                     "compared": int(both.sum()), "NA mismatch": int((np.isfinite(a) != np.isfinite(b)).sum()),
                     "differ": int((np.abs(a - b)[both] > tol[both]).sum()),
                     "max |diff|": float(np.nanmax(np.abs(a - b)[both])) if both.any() else float("nan")})
    table = pd.DataFrame(rows).set_index("column")
    parts += [f"Exact comparison on {len(joined):,} full-length 30-fps clips whose frame range equals an old "
              "window (clip k = old window 3k; 29.97-fps files are offset and are not compared); speech-"
              f"conditioned columns on the {int(unchanged.sum()):,} of them whose speech mask is unchanged "
              "(VAD-sourced, inside the annotated span). Values are stored as float32, so |diff| up to ~1e-4 "
              "of the value is rounding. 'reused' columns must not differ beyond that; 'adapted' columns differ "
              "by design (all-frames wrist height, non-frozen hand frames, NA instead of 0 for no episodes).",
              _table(table, ".3g")]
    return "\n\n".join(parts)


def write_report(config: RunConfig, root: Path | None = None) -> Path:
    root = Path(root) if root is not None else config.annotations_dir
    tables = load_tables(root)
    provenance = json.loads((root / "provenance.json").read_text(encoding="utf-8"))
    validation = json.loads((root / "validation_report.json").read_text(encoding="utf-8"))
    sections = [
        f"# Annotation report — {config.run_id}",
        f"Generated from the published tables. Code sha256 {str(provenance.get('code_sha256', ''))[:16]} "
        f"(git HEAD {provenance.get('git_sha', '')[:10]}, dirty={provenance.get('git_dirty')}), "
        f"scan fingerprint {provenance.get('scan_fingerprint')}, "
        f"run hash {provenance.get('run_hash')}, clip length {provenance.get('clip_seconds')} s, "
        f"pixel pass {'included' if provenance.get('video_pass') else 'not run'}.",
        section_coverage(tables),
        section_distributions(tables),
        section_posture_validation(tables),
        section_expressivity_validation(tables, config.legacy_dir / "gesture_review"),
        section_reuse_crosscheck(tables, config.legacy_run_root),
        "## 4. Structural validation",
        f"{validation['n_checks']} checks, {validation['n_errors']} errors, {validation['n_warnings']} warnings "
        "(`validation_report.json`).",
    ]
    path = root / "annotation_report.md"
    path.write_text("\n\n".join(sections) + "\n", encoding="utf-8")
    return path

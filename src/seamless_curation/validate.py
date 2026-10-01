"""Structural validation of the annotation tables.

Every check here is a property a downstream user relies on without re-deriving
it, so a failure is a bug in the pipeline, not a data caveat. ``annotate`` runs
this before it swaps a new set of tables in, and refuses to publish on any
``error``; ``validate`` re-runs it on the published tables.

Checks, in four groups:

**Schema.** Each table has exactly its registered columns, in order, with the
registered Arrow types; non-nullable columns hold no NA; enum columns hold only
allowed values; scores, confidences and fractions lie in [0, 1].

**Keys and joins.** Primary keys are unique and non-null; every link resolves to
an existing row of its target table; partition columns (vendor, label, split)
agree with the parent row, so filtering clips by split is the same as filtering
their sessions by split.

**Coverage and pairing.** Every catalog recording has a row; every measured
recording's clips tile ``[0, n_frames)`` exactly with no gap or overlap and
unmeasured recordings have none; partner links are symmetric (A's partner clip
names A back) and connect clips with the same index in the same interaction.

**Missing-value semantics.** A value that the registry says is NA under a
condition is NA exactly then: MOI counts are NA iff the recording is not
annotated for that party (never 0); face measures are NA for recordings without
Imitator features; posture confidence is NA iff the label is unclear/unknown;
expressivity is NA iff its status is not 'measured'; the four posture shares sum
to 1.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Mapping

import numpy as np
import pandas as pd

from . import schema


@dataclass
class Report:
    checks: list[dict[str, Any]] = field(default_factory=list)

    def add(self, name: str, ok: bool, detail: str = "", *, severity: str = "error", count: int = 0) -> None:
        self.checks.append({"check": name, "ok": bool(ok), "severity": severity,
                            "count": int(count), "detail": detail[:500]})

    @property
    def errors(self) -> list[dict[str, Any]]:
        return [c for c in self.checks if not c["ok"] and c["severity"] == "error"]

    @property
    def ok(self) -> bool:
        return not self.errors

    def as_dict(self) -> dict[str, Any]:
        return {"ok": self.ok, "n_checks": len(self.checks), "n_errors": len(self.errors),
                "n_warnings": sum(1 for c in self.checks if not c["ok"] and c["severity"] == "warning"),
                "checks": self.checks}


def _check(report: Report, name: str, bad: pd.Series | np.ndarray | int, detail: str = "",
           severity: str = "error") -> None:
    if isinstance(bad, (int, np.integer)):
        count = int(bad)
    else:
        # An undecidable comparison (NA) is a failure, never a pass.
        count = int(np.asarray(pd.Series(bad).astype("boolean").fillna(True), dtype=bool).sum())
    report.add(name, count == 0, f"{count} rows. {detail}" if count else detail, severity=severity, count=count)


# ============================================================================= schema
def check_schema(tables: Mapping[str, pd.DataFrame], report: Report) -> None:
    for table in schema.TABLES:
        if table not in tables:
            report.add(f"{table}: present", False, "table missing")
            continue
        frame = tables[table]
        expected = schema.columns(table)
        report.add(f"{table}: columns match the registry", list(frame.columns) == expected,
                   f"extra={sorted(set(frame.columns) - set(expected))[:10]} "
                   f"missing={sorted(set(expected) - set(frame.columns))[:10]}")
        for f in schema.fields(table):
            if f.name not in frame.columns:
                continue
            series = frame[f.name]
            if not f.nullable:
                _check(report, f"{table}.{f.name}: non-null", series.isna())
            if f.allowed:
                values = series.dropna()
                allowed = set(f.allowed)
                bad = ~values.isin(allowed) if not all(isinstance(a, (int, np.integer)) for a in f.allowed) \
                    else ~pd.to_numeric(values).isin(allowed)
                _check(report, f"{table}.{f.name}: allowed values", bad,
                       f"unexpected {sorted(map(str, values[bad].unique()))[:8]}")
            unit_interval = f.value_range == (0.0, 1.0) or (f.role in ("score", "confidence"))
            if unit_interval and f.dtype.startswith("float"):
                values = pd.to_numeric(series, errors="coerce")
                _check(report, f"{table}.{f.name}: within [0, 1]",
                       (values < -1e-6) | (values > 1 + 1e-6))


# ============================================================================= keys
def check_keys(tables: Mapping[str, pd.DataFrame], report: Report) -> None:
    for table, key in schema.PRIMARY_KEYS.items():
        if table in tables and key in tables[table]:
            frame = tables[table]
            _check(report, f"{table}.{key}: unique", frame[key].duplicated())
    for table, frame in tables.items():
        for f in schema.fields(table):
            if f.role != "link" or f.name not in frame.columns:
                continue
            target = schema.LINK_TARGETS[f.name]
            target_key = schema.PRIMARY_KEYS[target]
            if target not in tables:
                continue
            known = pd.Index(tables[target][target_key])
            values = frame[f.name].dropna()
            _check(report, f"{table}.{f.name} -> {target}: resolves", ~values.isin(known))


def _parent_agrees(report: Report, child: pd.DataFrame, parent: pd.DataFrame, key: str, name: str,
                   columns: tuple[str, ...]) -> None:
    present = [c for c in columns if c in child.columns and c in parent.columns]
    merged = child[[key, *present]].merge(parent[[key, *present]].drop_duplicates(key), on=key,
                                          how="left", suffixes=("", "_parent"))
    for column in present:
        bad = merged[column].astype("string") != merged[f"{column}_parent"].astype("string")
        _check(report, f"{name}: {column} agrees with parent", bad.fillna(True))


def check_partitions(tables: Mapping[str, pd.DataFrame], report: Report) -> None:
    all_parts = schema.PARTITIONS
    pairs = (
        ("interactions", "sessions", "session_key", all_parts),
        ("recordings", "interactions", "interaction_key", all_parts),
        ("interaction_windows", "interactions", "interaction_key", all_parts),
        ("clips", "recordings", "file_id", all_parts + ("interaction_key", "session_key", "participant_key")),
        ("moi_events", "recordings", "file_id", all_parts + ("interaction_key", "participant_key")),
        ("recordings", "participants", "participant_key", ("vendor",)),
    )
    for child, parent, key, columns in pairs:
        if child in tables and parent in tables:
            _parent_agrees(report, tables[child], tables[parent], key, f"{child} vs {parent}", columns)


# ============================================================================= coverage & pairing
def check_coverage(tables: Mapping[str, pd.DataFrame], report: Report, *, clip_seconds: float,
                   catalog_file_ids: pd.Index | None = None) -> None:
    rec = tables["recordings"]
    clips = tables["clips"]
    if catalog_file_ids is not None:
        _check(report, "recordings: every catalog recording present",
               ~pd.Index(catalog_file_ids).isin(rec["file_id"]))
    measured = rec["measurement_status"] == "measured"
    n_clips = clips.groupby("file_id").size()
    have = rec["file_id"].map(n_clips).fillna(0).astype(int)
    _check(report, "recordings: recording_n_clips equals clip rows",
           have != rec["recording_n_clips"].astype(int))
    _check(report, "recordings: unmeasured recordings have no clips", (~measured) & (have > 0))
    _check(report, "recordings: measured recordings have clips", measured & (have == 0))

    ordered = clips.sort_values(["file_id", "clip_index"])
    first = ordered.groupby("file_id").head(1)
    _check(report, "clips: grid starts at frame 0", first["start_frame"] != 0)
    _check(report, "clips: clip_index is 0..n-1",
           ordered.groupby("file_id")["clip_index"].transform(lambda s: s.to_numpy() != np.arange(len(s))).astype(bool))
    previous_end = ordered.groupby("file_id")["end_frame"].shift(1)
    gap = previous_end.notna() & (previous_end != ordered["start_frame"])
    _check(report, "clips: tiles are contiguous (no gap/overlap)", gap)
    last = ordered.groupby("file_id").tail(1)
    frames = last["file_id"].map(rec.drop_duplicates("file_id").set_index("file_id")["n_frames"])
    _check(report, "clips: last clip ends at n_frames", last["end_frame"] != frames)
    _check(report, "clips: only the last clip may be partial",
           ordered["is_partial"].astype(bool) & ~ordered["is_last"].astype(bool))
    _check(report, "clips: n_frames_clip = end - start",
           clips["n_frames_clip"] != clips["end_frame"] - clips["start_frame"])
    token = f"_L{float(clip_seconds):g}".replace(".", "p") + "_C"
    prefix_ok = [str(c).startswith(f"{f}{token}") for c, f in zip(clips["clip_id"], clips["file_id"])]
    _check(report, "clips: clip_id encodes file_id and grid", ~np.asarray(prefix_ok, dtype=bool))


def check_pairing(tables: Mapping[str, pd.DataFrame], report: Report) -> None:
    rec = tables["recordings"]
    clips = tables["clips"]
    inter = tables["interactions"]
    rec_unique = rec.drop_duplicates("file_id")
    partner = rec_unique.set_index("file_id")["partner_file_id"]
    has = rec["partner_file_id"].notna()
    back = rec.loc[has, "partner_file_id"].map(partner)
    _check(report, "recordings: partner link symmetric", back != rec.loc[has, "file_id"])
    same = rec.loc[has, "partner_file_id"].map(rec_unique.set_index("file_id")["interaction_key"])
    _check(report, "recordings: partner in the same interaction", same != rec.loc[has, "interaction_key"])
    _check(report, "recordings: partner is another participant",
           rec.loc[has, "partner_participant_key"] == rec.loc[has, "participant_key"])

    linked = clips["partner_clip_id"].notna()
    table = clips.drop_duplicates("clip_id").set_index("clip_id")
    back = clips.loc[linked, "partner_clip_id"].map(table["partner_clip_id"])
    _check(report, "clips: partner clip link symmetric", back != clips.loc[linked, "clip_id"])
    _check(report, "clips: partner clip has the same index",
           clips.loc[linked, "partner_clip_id"].map(table["clip_index"]) != clips.loc[linked, "clip_index"])
    _check(report, "clips: partner clip in the same interaction",
           clips.loc[linked, "partner_clip_id"].map(table["interaction_key"]) != clips.loc[linked, "interaction_key"])
    _check(report, "clips: partner_clip_id set iff linked",
           linked != (clips["partner_link_status"] == "linked"))
    counts = rec.groupby("interaction_key").size()
    _check(report, "interactions: n_recordings matches recordings",
           inter["interaction_key"].map(counts).fillna(0).astype(int) != inter["interaction_n_recordings"].astype(int))
    _check(report, "interactions: partner_missing iff one recording",
           (inter["pairing_status"] == "partner_missing") != (inter["interaction_n_recordings"] == 1))
    windows = tables["interaction_windows"]
    listed = set(windows["member_clip_id_1"].dropna()) | set(windows["member_clip_id_2"].dropna())
    _check(report, "clips: every clip is listed in its dyad window", ~clips["clip_id"].isin(listed))
    for column in ("member_clip_id_1", "member_clip_id_2"):
        linked_w = windows[column].notna()
        _check(report, f"interaction_windows.{column}: clip index equals window index",
               windows.loc[linked_w, column].map(table["clip_index"]) != windows.loc[linked_w, "window_index"])


# ============================================================================= missing-value semantics
def check_semantics(tables: Mapping[str, pd.DataFrame], report: Report) -> None:
    rec = tables["recordings"]
    clips = tables["clips"].merge(
        rec[["file_id", "moi_status", "has_imitator_movement", "smplh_anamorphic"]], on="file_id", how="left")
    three = clips["moi_status"].isin(["annotated_3p", "annotated_1p_3p"])
    one = clips["moi_status"] == "annotated_1p_3p"
    _check(report, "clips: moi_3p_count NA iff not 3P-annotated", clips["moi_3p_count"].isna() == three)
    _check(report, "clips: moi_1p_count NA iff not 1P-annotated", clips["moi_1p_count"].isna() == one)
    rec3 = rec["moi_status"].isin(["annotated_3p", "annotated_1p_3p"]) & (rec["measurement_status"] == "measured")
    _check(report, "recordings: recording_moi_3p_count NA iff not 3P-annotated (measured)",
           (rec["recording_moi_3p_count"].isna() == rec3) & (rec["measurement_status"] == "measured"))
    _check(report, "clips: face measures NA without Imitator features",
           (~clips["has_imitator_movement"].astype(bool)) & clips["fau_variability"].notna())
    decisive = clips["posture"].isin(["standing", "sitting", "mixed"])
    _check(report, "clips: posture_confidence NA iff not decisive", clips["posture_confidence"].isna() == decisive)
    shares = clips[["posture_standing_frac", "posture_sitting_frac", "posture_unclear_frac", "posture_unobserved_frac"]].sum(axis=1)
    _check(report, "clips: posture shares sum to 1", (shares - 1).abs() > 1e-3)
    annotated = clips["speech_annotation_status"] == "annotated"
    _check(report, "clips: own_speech_s NA iff speech is unannotated", clips["own_speech_s"].isna() == annotated)
    _check(report, "clips: speaking_role unknown where own speech is unannotated",
           ~annotated & (clips["speaking_role"] != "unknown"))
    measured = clips["expressivity_status"] == "measured"
    _check(report, "clips: expressivity_score NA iff not measured", clips["expressivity_score"].isna() == measured)
    _check(report, "clips: severe anamorphic has no expressivity or posture",
           (clips["smplh_anamorphic"] == "severe") & (measured | decisive))
    _check(report, "clips: expressivity_level matches score",
           measured & (clips["expressivity_level"] == "unknown"))
    unknown_role = clips["speaking_role"] == "unknown"
    _check(report, "clips: speaking_role unknown when the partner is not linked",
           (clips["partner_link_status"] != "linked") & ~unknown_role)
    inter = tables["interactions"]
    pair_na = ~((inter["speech_status"] == "both_annotated")
                & inter["pair_duration_agreement"].isin(["agree", "minor_mismatch"]))
    _check(report, "interactions: pair speech measures NA unless both annotated and aligned",
           pair_na & inter["interaction_mutual_silence_frac"].notna())
    _check(report, "interactions: expressivity aggregates need both members",
           (inter["interaction_n_members_measured"] < 2) & inter["interaction_expressivity_mean"].notna())
    _check(report, "interactions: moi_3p count NA unless both members annotated and measured",
           ((inter["moi_3p_coverage"] != "both") | (inter["interaction_n_members_measured"] < 2))
           & inter["interaction_moi_3p_count"].notna())
    visual = clips["visual_status"] == "measured"
    _check(report, "clips: visual measures NA iff not measured", clips["frame_luma_mean"].isna() == visual)


def validate_tables(tables: Mapping[str, pd.DataFrame], *, clip_seconds: float,
                    catalog_file_ids: pd.Index | None = None) -> Report:
    report = Report()
    registry = schema.registry_problems()
    report.add("registry is internally consistent", not registry, "; ".join(registry[:5]))
    check_schema(tables, report)
    check_keys(tables, report)
    check_partitions(tables, report)
    if all(t in tables for t in ("recordings", "clips", "interactions", "interaction_windows")):
        check_coverage(tables, report, clip_seconds=clip_seconds, catalog_file_ids=catalog_file_ids)
        check_pairing(tables, report)
        check_semantics(tables, report)
    return report

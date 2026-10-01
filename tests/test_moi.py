"""Meta MOI annotations: event rows, clip/recording counts and pair duplication.

The fixtures are hand-built ``annotations:<KIND>`` lists in the released shape
(``{annotation, start_ts, end_ts}``, integer seconds), including the malformed
entries the release really has: zero length, negative length, past the end.
"""

from __future__ import annotations

import math

import pytest

from seamless_curation import moi
from seamless_curation.ids import moi_id
from seamless_curation.moi import (
    KINDS,
    annotated_parties,
    clip_moi,
    duplication_status,
    mark_duplicates,
    parse_events,
    recording_moi,
)

FILE_A = "V03_S0203_I00000495_P1436"
FILE_B = "V03_S0203_I00000495_P1437"


def entry(text: str, start, end) -> dict:
    return {"annotation": text, "start_ts": start, "end_ts": end}


def moment(kinds, start, end, text="Participant laughs"):
    """One moment filed under several kinds with the same interval (as released)."""

    return {f"annotations:{k}": [entry(f"{text} ({k})", start, end)] for k in kinds}


def merge(*parts) -> dict:
    out: dict = {}
    for part in parts:
        for key, value in part.items():
            out.setdefault(key, []).extend(value)
    return out


THREE_P = ("3P-IS", "3P-R", "3P-V")
ONE_P = ("1P-IS", "1P-R")


# ----------------------------------------------------------------------------- events
def test_parse_events_rows_and_ids() -> None:
    ann = merge(moment(THREE_P + ONE_P, 10, 14), {"annotations:3P-IS": [entry("second", 40, 42)]})
    events = parse_events(ann, FILE_A, 120.0)
    assert len(events) == 6
    first = [e for e in events if e["annotation_kind"] == "3P-IS"]
    assert [e["moi_id"] for e in first] == [moi_id(FILE_A, "3P-IS", 0), moi_id(FILE_A, "3P-IS", 1)]
    assert first[1]["moi_id"] == f"{FILE_A}_M3PIS_001"
    row = first[0]
    assert row["party"] == "3P" and row["file_id"] == FILE_A
    assert (row["moi_start_s"], row["moi_end_s"], row["moi_duration_s"]) == (10.0, 14.0, 4.0)
    assert row["moi_malformed"] == "ok" and row["partner_duplicate_moi_id"] is None
    assert {e["party"] for e in events if e["annotation_kind"].startswith("1P")} == {"1P"}
    assert set(events[0]) >= {"moi_id", "file_id", "annotation_kind", "party", "moi_start_s", "moi_end_s",
                              "moi_duration_s", "moi_text", "moi_malformed", "partner_duplicate_moi_id"}


def test_malformed_entries_are_kept_and_labelled() -> None:
    ann = {"annotations:3P-IS": [
        entry("zero", 20, 20), entry("negative", 179, 171), entry("past end", 118, 125),
        entry("just past", 118, 120), entry("fine", 5, 7),
    ]}
    events = parse_events(ann, FILE_A, 119.6)
    status = {e["moi_text"]: e["moi_malformed"] for e in events}
    assert status == {"zero": "zero_length", "negative": "negative_length", "past end": "beyond_recording",
                      "just past": "ok", "fine": "ok"}  # 120 <= 119.6 + 0.5
    negative = next(e for e in events if e["moi_text"] == "negative")
    assert negative["moi_duration_s"] == -8.0
    assert recording_moi(events, {"3P"}, 119.6)["moi_malformed_count"] == 3


def test_non_numeric_times_are_kept_as_nan_and_not_counted() -> None:
    ann = {"annotations:3P-IS": [entry("untimed", None, "x"), entry("fine", 5, 7)]}
    events = parse_events(ann, FILE_A, 60.0)
    untimed = events[0]
    assert math.isnan(untimed["moi_start_s"]) and math.isnan(untimed["moi_duration_s"])
    assert untimed["moi_malformed"] == "zero_length"
    assert recording_moi(events, {"3P"}, 60.0)["recording_moi_3p_count"] == 1
    assert clip_moi(events, {"3P"}, 0.0, 30.0)["moi_3p_count"] == 1


def test_annotated_parties_from_keys_and_unknown_kinds_ignored() -> None:
    assert annotated_parties({}) == set()
    assert annotated_parties(moment(THREE_P, 1, 2)) == {"3P"}
    assert annotated_parties(moment(THREE_P + ONE_P, 1, 2)) == {"1P", "3P"}
    assert annotated_parties({"annotations:3P-IS": []}) == {"3P"}  # annotated, zero moments
    assert annotated_parties({"annotations:2P-X": [entry("?", 1, 2)]}) == set()
    assert parse_events({"annotations:2P-X": [entry("?", 1, 2)]}, FILE_A, 10.0) == []
    assert KINDS == ("1P-IS", "1P-R", "3P-IS", "3P-R", "3P-V")


# ----------------------------------------------------------------------------- counts
def test_distinct_moments_count_once_across_kinds() -> None:
    ann = merge(moment(THREE_P, 10, 14), moment(THREE_P, 40, 42), moment(ONE_P, 10, 14),
                {"annotations:3P-IS": [entry("repeat of the same interval", 10, 14)]})
    events = parse_events(ann, FILE_A, 90.0)
    rec = recording_moi(events, {"1P", "3P"}, 90.0)
    assert rec["recording_moi_3p_count"] == 2
    assert rec["recording_moi_1p_count"] == 1
    assert rec["recording_moi_3p_rate_per_min"] == pytest.approx(2 / 1.5)
    assert rec["moi_status"] == "annotated_1p_3p"
    assert rec["moi_malformed_count"] == 0


def test_unannotated_is_na_never_zero() -> None:
    rec = recording_moi([], set(), 120.0)
    assert rec["moi_status"] == "not_annotated"
    assert rec["recording_moi_3p_count"] is None and rec["recording_moi_1p_count"] is None
    assert math.isnan(rec["recording_moi_3p_rate_per_min"]) and rec["moi_malformed_count"] is None
    clip = clip_moi([], set(), 0.0, 30.0)
    assert clip["moi_3p_count"] is None and clip["moi_1p_count"] is None
    assert math.isnan(clip["moi_3p_seconds"])
    # 3P annotated, 1P not: the 1P count stays NA while 3P reads a real 0
    events = parse_events(moment(THREE_P, 50, 52), FILE_A, 120.0)
    clip = clip_moi(events, {"3P"}, 0.0, 30.0)
    assert clip["moi_3p_count"] == 0 and clip["moi_3p_seconds"] == 0.0 and clip["moi_1p_count"] is None
    rec = recording_moi(events, {"3P"}, 120.0)
    assert rec["moi_status"] == "annotated_3p" and rec["recording_moi_1p_count"] is None


def test_clip_overlap_rule_and_boundary_moments() -> None:
    ann = merge(
        moment(THREE_P, 25, 35),   # spans the 30-s boundary: both clips
        moment(THREE_P, 20, 30),   # ends exactly at 30: clip 0 only
        moment(THREE_P, 30, 31),   # starts exactly at 30: clip 1 only
        moment(THREE_P, 60, 60),   # zero length at 60: clip 2 (start_s <= t < end_s)
        moment(THREE_P, 45, 45),   # zero length inside clip 1
    )
    events = parse_events(ann, FILE_A, 90.0)
    counts = [clip_moi(events, {"3P"}, k * 30.0, (k + 1) * 30.0)["moi_3p_count"] for k in range(3)]
    assert counts == [2, 3, 1]
    assert recording_moi(events, {"3P"}, 90.0)["recording_moi_3p_count"] == 5


def test_negative_length_moment_is_the_span_between_its_stamps() -> None:
    events = parse_events(moment(THREE_P, 181, 178), FILE_A, 240.0)
    counts = [clip_moi(events, {"3P"}, s, s + 30.0)["moi_3p_count"] for s in (150.0, 180.0, 210.0)]
    assert counts == [1, 1, 0]  # [178, 181] straddles 180 s: never lost
    assert clip_moi(events, {"3P"}, 150.0, 180.0)["moi_3p_seconds"] == pytest.approx(2.0)


def test_moments_beyond_the_recording_count_for_the_recording_not_its_clips() -> None:
    events = parse_events(moment(THREE_P, 173, 179), FILE_A, 157.0)
    assert {e["moi_malformed"] for e in events} == {"beyond_recording"}
    assert recording_moi(events, {"3P"}, 157.0)["recording_moi_3p_count"] == 1
    assert clip_moi(events, {"3P"}, 150.0, 157.0)["moi_3p_count"] == 0


def test_moi_3p_seconds_is_the_union_within_the_clip() -> None:
    ann = merge(moment(THREE_P, 2, 6), moment(THREE_P, 4, 8), moment(THREE_P, 28, 33),
                moment(ONE_P, 10, 20))
    events = parse_events(ann, FILE_A, 60.0)
    clip = clip_moi(events, {"1P", "3P"}, 0.0, 30.0)
    assert clip["moi_3p_seconds"] == pytest.approx(6.0 + 2.0)  # [2,8] + [28,30]
    assert clip["moi_3p_count"] == 3 and clip["moi_1p_count"] == 1


# ----------------------------------------------------------------------------- pairs
def pair(text_a="I was curious to know more about him.", text_b="I was curious to know more about him.",
         start_a=43, start_b=43, kinds=("1P-IS",)):
    a = parse_events({f"annotations:{k}": [entry(text_a, start_a, start_a + 2)] for k in kinds}, FILE_A, 200.0)
    b = parse_events({f"annotations:{k}": [entry(text_b, start_b, start_b + 2)] for k in kinds}, FILE_B, 200.0)
    return a, b


def test_duplicates_with_one_second_offset_are_marked_both_ways() -> None:
    a, b = pair(start_a=200 - 150, start_b=201 - 150)
    mark_duplicates(a, b)
    assert a[0]["partner_duplicate_moi_id"] == b[0]["moi_id"]
    assert b[0]["partner_duplicate_moi_id"] == a[0]["moi_id"]
    assert duplication_status(a, b, "1P", {"1P"}, {"1P"}) == "full"


def test_duplicate_needs_same_kind_normalised_text_and_start_within_two_seconds() -> None:
    a, b = pair(text_b="  i WAS curious to know   more about him ")  # case/space/punctuation only
    mark_duplicates(a, b)
    assert a[0]["partner_duplicate_moi_id"] is not None
    far_a, far_b = pair(start_a=279, start_b=285)  # identical text, 6 s apart
    mark_duplicates(far_a, far_b)
    assert far_a[0]["partner_duplicate_moi_id"] is None
    assert duplication_status(far_a, far_b, "1P", {"1P"}, {"1P"}) == "none"
    other_a, other_b = pair(text_b="I was bored.")
    mark_duplicates(other_a, other_b)
    assert other_b[0]["partner_duplicate_moi_id"] is None
    kind_a = parse_events({"annotations:3P-IS": [entry("same", 10, 12)]}, FILE_A, 60.0)
    kind_b = parse_events({"annotations:3P-V": [entry("same", 10, 12)]}, FILE_B, 60.0)
    mark_duplicates(kind_a, kind_b)
    assert kind_a[0]["partner_duplicate_moi_id"] is None


def test_nearest_partner_event_is_chosen_and_marks_are_reset() -> None:
    a = parse_events({"annotations:3P-IS": [entry("The participant was confused", 50, 52)]}, FILE_A, 200.0)
    b = parse_events({"annotations:3P-IS": [entry("The participant was confused", 48, 49),
                                            entry("The participant was confused", 51, 53)]}, FILE_B, 200.0)
    mark_duplicates(a, b)
    assert a[0]["partner_duplicate_moi_id"] == b[1]["moi_id"]
    assert b[0]["partner_duplicate_moi_id"] == a[0]["moi_id"] and b[1]["partner_duplicate_moi_id"] == a[0]["moi_id"]
    # re-running against a different partner clears stale marks
    mark_duplicates(a, [])
    assert a[0]["partner_duplicate_moi_id"] is None


def test_duplication_status_none_partial_full_not_applicable() -> None:
    both = {"1P", "3P"}
    a = parse_events(merge(moment(THREE_P, 10, 12, "copied"), moment(("3P-IS",), 60, 62, "own")), FILE_A, 120.0)
    b = parse_events(merge(moment(THREE_P, 11, 13, "copied")), FILE_B, 120.0)
    mark_duplicates(a, b)
    assert duplication_status(a, b, "3P", both, both) == "partial"  # A's 'own' moment has no copy
    b_full = parse_events(merge(moment(THREE_P, 11, 13, "copied"), moment(("3P-IS",), 61, 62, "own")), FILE_B, 120.0)
    mark_duplicates(a, b_full)
    assert duplication_status(a, b_full, "3P", both, both) == "full"
    c = parse_events(moment(THREE_P, 90, 92, "different"), FILE_B, 120.0)
    mark_duplicates(a, c)
    assert duplication_status(a, c, "3P", both, both) == "none"
    # party not annotated on both members
    assert duplication_status(a, b, "3P", {"3P"}, set()) == "not_applicable"
    assert duplication_status(a, b, "1P", {"1P", "3P"}, {"3P"}) == "not_applicable"
    # annotated on both but no events of the party: nothing duplicated
    assert duplication_status([], [], "1P", {"1P"}, {"1P"}) == "none"


def test_duplication_counts_only_the_requested_party() -> None:
    a = parse_events(merge(moment(ONE_P, 10, 12, "copied"), moment(("3P-IS",), 40, 42, "a only")), FILE_A, 120.0)
    b = parse_events(merge(moment(ONE_P, 10, 12, "copied"), moment(("3P-IS",), 80, 82, "b only")), FILE_B, 120.0)
    mark_duplicates(a, b)
    both = {"1P", "3P"}
    assert duplication_status(a, b, "1P", both, both) == "full"
    assert duplication_status(a, b, "3P", both, both) == "none"


def test_normalise_text() -> None:
    assert moi.normalise_text('  I was "curious",  REALLY. ') == "i was curious really"
    assert moi.normalise_text(None) == ""

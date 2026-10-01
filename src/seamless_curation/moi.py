"""Meta's moments of interest (MOI): event rows, clip and recording counts, pair duplication.

A subset of recordings carries human annotations of *moments of interest* under
``annotations:<KIND>`` in the JSON, each a list of ``{annotation, start_ts,
end_ts}``. The kinds are the self-report (1P: ``1P-IS`` internal state,
``1P-R`` rationale) and the observer (3P: ``3P-IS`` perceived state, ``3P-R``
rationale, ``3P-V`` visual description). This module keeps every released entry
as one ``moi_events`` row, counts the distinct moments per clip and recording,
and marks entries copied between the two members of a pair. It never drops an
entry: malformed ones are kept and labelled.

What the release holds (read-only scan of every annotated JSON, 1,639 of the
1,663 flagged files readable; the other 24 are absent from the local copy)
-----------------------------------------------------------------------------
* **Key sets.** V00: 1,123 files with ``3P-IS/R/V`` only. V03: 487 files with
  all five kinds and 29 with 3P only. No file has 1P without 3P, no list is
  empty, and the key sets match ``filelist.csv``'s ``has_annotation_*`` flags in
  every file. A party is *annotated* when any of its keys exists
  (:func:`annotated_parties`); an unannotated party's counts are NA, never 0,
  because absence of annotation is not absence of moments of interest.
* **Times** are integers (all 32,116 ``start_ts``/``end_ts``), seconds from the
  recording start, stored as released. Of 16,058 entries 590 have zero length
  (V00 573, V03 17), 36 negative length (12 moments x 3 kinds, all V00, e.g.
  ``V00_S1238_I00001040_P0682A`` 179 -> 171; the worst 379 -> 325) and 5 lie
  beyond the recording (one moment of ``V03_S0242_I00000502_P1477``, 173-179 s
  in a 157 s recording). Per recording: V00 a median 4 distinct 3P moments
  (1-11, 0.65/min), V03 1 (1-17, 0.31/min); 1P median 1 (1-6).
* **Kinds share intervals.** The IS, R and V entries of one moment carry the same
  ``(start, end)`` (all but 5 of 1,639 files have identical interval lists over
  the three 3P kinds), so the entries are not independent moments. A *distinct
  MOI* of a party is a distinct ``(start, end)`` pair among that party's kinds;
  21 same-kind entries repeat an interval and also count once.

Event rows (:func:`parse_events`)
---------------------------------
``moi_id`` is :func:`seamless_curation.ids.moi_id` with the entry's **0-based
position in the released list** (the id maps back to the JSON), ``party`` the
kind's prefix, times and text as released (``moi_duration_s = end - start``,
negative kept). ``moi_malformed`` is, in this order of precedence:
'negative_length', 'zero_length', 'beyond_recording' (ends more than
:data:`BEYOND_TOLERANCE_S` = 0.5 s after the recording's pose grid, which
absorbs the audio running a fraction of a second past the last pose frame),
else 'ok'. Entries whose times are not finite numbers (none in the release)
are kept with NaN times, labelled 'zero_length' (they have no measurable
extent) and excluded from every count. Keys other than the five kinds are
ignored.

Clip and recording counts
-------------------------
A moment overlaps the clip ``[start_s, end_s)`` when ``start < end_s`` and
``end > start_s``; a zero-length moment is a point that belongs to the clip
with ``start_s <= start < end_s``. **A negative-length moment is read as the span
between its two timestamps** (``[end, start]``): taking the contract's
interval rule literally would make ``181 -> 178`` overlap no clip at all when a
clip boundary falls at 180 s, losing a released annotation. ``moi_3p_seconds``
is the union of the distinct 3P spans inside the clip. The recording counts
include moments beyond the recording (they are released annotations of it;
``moi_malformed_count`` says how many entries are suspect), and the 3P rate is
per minute of the pose grid.

``moi_status`` follows ``annotate.py``'s mapping of Meta's flags so measured and
unmeasured rows agree: 'annotated_1p_3p' when both parties are present,
'annotated_3p' for 3P only, else 'not_annotated'. The registry has no value for
1P without 3P; it does not occur in the release and would read 'not_annotated'
with a non-NA 1P count, which the validator's NA check reports.

Pair duplication (:func:`mark_duplicates`, :func:`duplication_status`)
----------------------------------------------------------------------
In V03 the same annotation is sometimes filed under both members of a pair
(``V03_S0203_I00000495``: P1436 and P1437 both self-report "I was curious to
know more about him."). Whole-list identity misses half of them because the
copies can be offset by a second or two (``V03_S1270_I00000059``: 32-36 vs 34-37,
both "Participant feels itchy"), and a text-only rule would match V00's formulaic
phrases ("The participant was frustrated") said at different times. An event is
a duplicate when the partner has an event of **the same kind, the same
normalised text** (lowercase, punctuation stripped, whitespace collapsed; empty
text never matches) **and a start within 2 s**; ``partner_duplicate_moi_id``
points at the nearest such partner event (ties: the earlier-listed one). The
relation is evaluated from each side, so it is not forced to be one-to-one. A
duplicated event's target person is ambiguous.

Over the readable annotated files (572 pairs with both members annotated for
3P, 36 for 1P; 495 annotated interactions have one readable annotated member)
the rule finds: V03 1P 'full' in 7 of 36 pairs (six in session S0203, plus
S1270_I00000059), V03 3P 'full' in 3 and 'partial' in 4 of 51, and V00 3P
'partial' in 3 of 521 (``V00_S1277_I00001117``: all three kinds of one moment
copied verbatim, "eye Tourette" and all; two pairs share one "The participant
was confused" at the same second, which may be a real shared moment). One pair
with identical text is not matched: ``V03_S0203_I00000481`` (285 s vs 279 s,
6 s apart). ``V03_S0203_I00000495`` is 'full' for both parties.
"""

from __future__ import annotations

import math
import re
from typing import Any, Iterable, Mapping, Sequence

from .ids import moi_id

KINDS: tuple[str, ...] = ("1P-IS", "1P-R", "3P-IS", "3P-R", "3P-V")
PARTIES: tuple[str, ...] = ("1P", "3P")
MALFORMED_VALUES: tuple[str, ...] = ("ok", "zero_length", "negative_length", "beyond_recording")
DUPLICATION_VALUES: tuple[str, ...] = ("none", "partial", "full", "not_applicable")
#: An entry ending more than this after the recording is 'beyond_recording'.
BEYOND_TOLERANCE_S = 0.5
#: Start-time tolerance for a copied (duplicate) partner annotation.
DUPLICATE_TOLERANCE_S = 2.0

_PUNCT = re.compile(r"[^\w\s]+", flags=re.UNICODE)
_SPACE = re.compile(r"\s+", flags=re.UNICODE)


def _key(kind: str) -> str:
    return f"annotations:{kind}"


def _time(value: Any) -> float:
    if value is None or isinstance(value, bool):
        return float("nan")
    try:
        number = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return number if math.isfinite(number) else float("nan")


def _finite(*values: Any) -> bool:
    """True when every value is a real, finite number (None, bools and NaN are not)."""

    for value in values:
        if value is None or isinstance(value, bool):
            return False
        try:
            if not math.isfinite(float(value)):
                return False
        except (TypeError, ValueError):
            return False
    return True


def malformed_status(start_s: float, end_s: float, duration_s: float) -> str:
    """'negative_length' > 'zero_length' > 'beyond_recording' > 'ok' (see module docstring)."""

    if not _finite(start_s, end_s):
        return "zero_length"
    if end_s < start_s:
        return "negative_length"
    if end_s == start_s:
        return "zero_length"
    if _finite(duration_s) and end_s > duration_s + BEYOND_TOLERANCE_S:
        return "beyond_recording"
    return "ok"


# ----------------------------------------------------------------------------- events
def annotated_parties(annotation: Mapping[str, Any]) -> set[str]:
    """Parties ('1P', '3P') with at least one of their kinds' keys in the JSON."""

    return {kind[:2] for kind in KINDS if _key(kind) in annotation}


def parse_events(annotation: Mapping[str, Any], file_id: str, duration_s: float) -> list[dict[str, Any]]:
    """One ``moi_events`` row per released entry, in kind order then release order."""

    events: list[dict[str, Any]] = []
    for kind in KINDS:
        entries = annotation.get(_key(kind))
        if not isinstance(entries, Sequence) or isinstance(entries, (str, bytes)):
            continue
        for index, entry in enumerate(entries):
            entry = entry if isinstance(entry, Mapping) else {}
            start, end = _time(entry.get("start_ts")), _time(entry.get("end_ts"))
            text = entry.get("annotation")
            events.append({
                "moi_id": moi_id(file_id, kind, index),
                "file_id": file_id,
                "annotation_kind": kind,
                "party": kind[:2],
                "moi_start_s": start,
                "moi_end_s": end,
                "moi_duration_s": end - start if _finite(start, end) else float("nan"),
                "moi_text": "" if text is None else str(text),
                "moi_malformed": malformed_status(start, end, duration_s),
                "partner_duplicate_moi_id": None,
            })
    return events


def distinct_intervals(events: Iterable[Mapping[str, Any]], party: str) -> list[tuple[float, float]]:
    """Distinct finite ``(start, end)`` pairs of one party's events, sorted."""

    spans = {
        (float(e["moi_start_s"]), float(e["moi_end_s"]))
        for e in events
        if e.get("party") == party and _finite(e.get("moi_start_s"), e.get("moi_end_s"))
    }
    return sorted(spans)


def _overlaps(start: float, end: float, start_s: float, end_s: float) -> bool:
    low, high = min(start, end), max(start, end)  # negative length -> the span between the stamps
    if high == low:
        return start_s <= low < end_s
    return low < end_s and high > start_s


def _union_within(spans: Iterable[tuple[float, float]], start_s: float, end_s: float) -> float:
    clipped = sorted(
        (max(min(a, b), start_s), min(max(a, b), end_s)) for a, b in spans
    )
    total, cur_lo, cur_hi = 0.0, None, None
    for lo, hi in clipped:
        if hi <= lo:
            continue
        if cur_hi is None or lo > cur_hi:
            if cur_hi is not None:
                total += cur_hi - cur_lo
            cur_lo, cur_hi = lo, hi
        else:
            cur_hi = max(cur_hi, hi)
    if cur_hi is not None:
        total += cur_hi - cur_lo
    return float(total)


def clip_moi(events: Sequence[Mapping[str, Any]], parties: set[str], start_s: float, end_s: float) -> dict[str, Any]:
    """MOI columns of the clip ``[start_s, end_s)``; NA for an unannotated party."""

    out: dict[str, Any] = {"moi_3p_count": None, "moi_1p_count": None, "moi_3p_seconds": float("nan")}
    for party in PARTIES:
        if party not in parties:
            continue
        hits = [span for span in distinct_intervals(events, party) if _overlaps(*span, start_s, end_s)]
        out[f"moi_{party.lower()}_count"] = len(hits)
        if party == "3P":
            out["moi_3p_seconds"] = _union_within(hits, float(start_s), float(end_s))
    return out


def moi_status(parties: set[str]) -> str:
    if "3P" in parties:
        return "annotated_1p_3p" if "1P" in parties else "annotated_3p"
    return "not_annotated"


def recording_moi(events: Sequence[Mapping[str, Any]], parties: set[str], duration_s: float) -> dict[str, Any]:
    """Recording-level MOI columns; counts NA for an unannotated party."""

    out: dict[str, Any] = {
        "moi_status": moi_status(parties),
        "recording_moi_3p_count": None,
        "recording_moi_1p_count": None,
        "recording_moi_3p_rate_per_min": float("nan"),
        "moi_malformed_count": None,
    }
    for party in PARTIES:
        if party in parties:
            out[f"recording_moi_{party.lower()}_count"] = len(distinct_intervals(events, party))
    if "3P" in parties and _finite(duration_s) and duration_s > 0:
        out["recording_moi_3p_rate_per_min"] = out["recording_moi_3p_count"] / (float(duration_s) / 60.0)
    if parties:
        out["moi_malformed_count"] = sum(1 for e in events if e.get("moi_malformed") != "ok")
    return out


# ----------------------------------------------------------------------------- pairs
def normalise_text(text: Any) -> str:
    """Lowercase, punctuation stripped, whitespace collapsed ('I was  curious.' -> 'i was curious')."""

    return _SPACE.sub(" ", _PUNCT.sub("", str(text or "").lower())).strip()


def _match_side(own: Sequence[dict[str, Any]], other: Sequence[Mapping[str, Any]], tol_s: float) -> None:
    index: dict[tuple[str, str], list[tuple[float, int, str]]] = {}
    for order, event in enumerate(other):
        text = normalise_text(event.get("moi_text"))
        start = event.get("moi_start_s")
        if text and _finite(start):
            index.setdefault((str(event.get("annotation_kind")), text), []).append(
                (float(start), order, str(event["moi_id"])))
    for event in own:
        event["partner_duplicate_moi_id"] = None
        text = normalise_text(event.get("moi_text"))
        start = event.get("moi_start_s")
        if not text or not _finite(start):
            continue
        candidates = index.get((str(event.get("annotation_kind")), text), ())
        best = min(
            ((abs(float(start) - s), order, mid) for s, order, mid in candidates
             if abs(float(start) - s) <= tol_s + 1e-9),
            default=None,
        )
        if best is not None:
            event["partner_duplicate_moi_id"] = best[2]


def mark_duplicates(
    events_a: Sequence[dict[str, Any]], events_b: Sequence[dict[str, Any]], tol_s: float = DUPLICATE_TOLERANCE_S
) -> None:
    """Set ``partner_duplicate_moi_id`` on both members' events (None where no copy)."""

    _match_side(events_a, events_b, tol_s)
    _match_side(events_b, events_a, tol_s)


def duplication_status(
    events_a: Sequence[Mapping[str, Any]],
    events_b: Sequence[Mapping[str, Any]],
    party: str,
    parties_a: set[str],
    parties_b: set[str],
) -> str:
    """'not_applicable' unless both members are annotated for ``party``; else none/partial/full."""

    if party not in parties_a or party not in parties_b:
        return "not_applicable"
    mine = [e for e in list(events_a) + list(events_b) if e.get("party") == party]
    duplicated = sum(1 for e in mine if e.get("partner_duplicate_moi_id") is not None)
    if duplicated == 0:
        return "none"
    return "full" if duplicated == len(mine) else "partial"

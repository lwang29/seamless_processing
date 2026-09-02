"""Where manual verdicts live: an append-only JSONL log, resolved on read.

The v0 tooling had no verdict store at all. Notes went to browser
``localStorage`` under a namespace derived from a hash of the rendered manifest,
so re-rendering a clip or opening a different sub-page orphaned every note; the
structured-rating path was built and tested but never once used; and the only
durable human labels in the repository are 66 file ids hand-typed into a Python
list. A manifest that is supposed to be gated on manual review has to have
something to join against.

Design, and why:

**Append-only.** A verdict is never edited in place. Changing your mind appends
a new record and the later one wins, so the log is also the audit trail: who
looked at a clip, when, with which card version, and whether anyone overturned
it. A crashed browser or a killed process can lose at most the record being
written.

**Keyed by review item, not by render.** ``review_item_id`` is a hash of
``(vendor, file_id)``, so it is stable across re-renders, across app restarts,
and — the part that matters — across a change to the candidate set. An id
derived from a row's position in the queue would silently re-bind every stored
verdict the moment one file dropped out of a re-run. ``file_id`` is recorded
alongside it and the manifest asserts the two agree.

The card fingerprint is recorded too, so a verdict taken against a card that was
later re-rendered with different spans or different parameters is visible rather
than indistinguishable from a current one.

**Reviewer-attributed.** Every record carries ``reviewer`` and ``verdict_source``
(``human`` or a model identifier). The accepted manifest reports the mix, so
nobody has to guess how a subset was verified.

**Multi-reviewer by construction.** Several reviewers can append to the same log
concurrently — writes are single ``O_APPEND`` lines under a lock — and
:meth:`VerdictStore.resolve` reports disagreement rather than hiding it.
"""

from __future__ import annotations

import json
import os
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence

import pandas as pd

#: The three verdicts a reviewer can give. ``unsure`` is not a failure state: it
#: keeps an item out of the accepted manifest while marking it as seen, which is
#: what stops a hard case from being silently accepted or silently lost.
VERDICTS = ("accept", "reject", "unsure")

#: Reject reasons, chosen to match the questions the review is asking. Free text
#: is also allowed and is kept, but a closed list is what makes the reject side
#: analysable — and the reject histogram is how the automated gates get retuned.
REJECT_REASONS = (
    "static_hands",        # hands barely move while the person speaks
    "not_co_speech",       # movement, but unrelated to their speaking
    "tracking_broken",     # SMPL-H does not follow the person
    "unnatural_motion",    # jitter, popping, implausible pose
    "out_of_sync",         # motion and speech clearly not aligned
    "obscured",            # hands not visible enough to judge or to learn from
    "other",
)


@dataclass(frozen=True)
class Verdict:
    review_item_id: str
    verdict: str
    reviewer: str
    #: The file the reviewer actually looked at. ``review_item_id`` is derived
    #: from it, so this is redundant by construction — which is the point: it
    #: lets the manifest assert that the join it makes is the join the reviewer
    #: intended, rather than trusting the id.
    file_id: str = ""
    verdict_source: str = "human"
    reasons: tuple[str, ...] = ()
    note: str = ""
    card_fingerprint: str = ""
    saw_video: bool = False
    recorded_utc: str = ""

    def __post_init__(self) -> None:
        if self.verdict not in VERDICTS:
            raise ValueError(f"verdict must be one of {VERDICTS}, got {self.verdict!r}")
        if not self.review_item_id:
            raise ValueError("review_item_id is required")
        if not self.reviewer:
            raise ValueError("reviewer is required")
        if not self.recorded_utc:
            object.__setattr__(
                self, "recorded_utc", time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
            )

    def as_json(self) -> str:
        payload = asdict(self)
        payload["reasons"] = list(self.reasons)
        return json.dumps(payload, sort_keys=True)


def _as_bool(value: object) -> bool:
    """Parse a boolean from an import without letting the string "False" be true.

    ``bool("False")`` is ``True``, and an import that got that wrong would put a
    clip into ``accepted_clips_with_audio.csv`` — documented as the rows where
    synchronisation was confirmed *by ear* — on a verdict that says no audio was
    played.
    """

    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "t"}
    return bool(value)


def _as_reasons(value: object) -> tuple[str, ...]:
    """A bare string is one reason, not a tuple of its characters."""

    if value is None:
        return ()
    if isinstance(value, str):
        return (value,) if value.strip() else ()
    return tuple(str(item) for item in value)


class VerdictStore:
    """Append-only JSONL of :class:`Verdict` records."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self._lock = threading.Lock()

    # -------------------------------------------------------------- writing
    def append(self, verdict: Verdict) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        line = verdict.as_json() + "\n"
        with self._lock:
            # O_APPEND makes a single write of a short line atomic between
            # processes on a POSIX filesystem, which is what allows two
            # reviewers and a batch importer to share one log.
            handle = os.open(self.path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
            try:
                os.write(handle, line.encode("utf-8"))
            finally:
                os.close(handle)

    def extend(self, verdicts: Iterable[Verdict]) -> int:
        count = 0
        for verdict in verdicts:
            self.append(verdict)
            count += 1
        return count

    def import_file(self, path: Path) -> int:
        """Append verdicts from a JSON list or a JSONL export."""

        text = Path(path).read_text(encoding="utf-8").strip()
        if not text:
            return 0
        records: list[Mapping[str, Any]]
        if text.lstrip().startswith("["):
            records = json.loads(text)
        else:
            records = [json.loads(line) for line in text.splitlines() if line.strip()]
        return self.extend(
            Verdict(
                review_item_id=str(record["review_item_id"]),
                file_id=str(record.get("file_id") or ""),
                verdict=str(record["verdict"]),
                reviewer=str(record.get("reviewer") or "import"),
                verdict_source=str(record.get("verdict_source") or "human"),
                reasons=_as_reasons(record.get("reasons")),
                note=str(record.get("note") or ""),
                card_fingerprint=str(record.get("card_fingerprint") or ""),
                saw_video=_as_bool(record.get("saw_video")),
                recorded_utc=str(record.get("recorded_utc") or ""),
            )
            for record in records
        )

    # -------------------------------------------------------------- reading
    def records(self) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        out: list[dict[str, Any]] = []
        for line in self.path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except ValueError:
                continue  # a torn final line from a killed writer
        return out

    def resolve(self) -> pd.DataFrame:
        """Current verdict per item: last write wins, with disagreement flagged.

        ``reviewers`` and ``verdicts_seen`` retain the whole history for an item,
        so a table of contested items is one filter away rather than a re-parse
        of the log.
        """

        records = self.records()
        if not records:
            return pd.DataFrame(
                columns=[
                    "review_item_id", "file_id", "verdict", "reviewer", "verdict_source",
                    "reasons", "note", "saw_video", "recorded_utc", "reviewers",
                    "verdicts_seen", "contested", "revisions",
                ]
            )
        frame = pd.DataFrame(records)
        if "file_id" not in frame.columns:
            frame["file_id"] = ""
        # Stable sort on the timestamp alone: two verdicts recorded in the same
        # second keep their append order, which is the order they happened in.
        frame = frame.sort_values("recorded_utc", kind="stable")
        grouped = frame.groupby("review_item_id", sort=False)
        latest = grouped.tail(1).set_index("review_item_id")
        history = grouped.agg(
            reviewers=("reviewer", lambda values: sorted(set(values))),
            verdicts_seen=("verdict", lambda values: sorted(set(values))),
            revisions=("verdict", "size"),
        )
        resolved = latest.join(history).reset_index()
        resolved["contested"] = resolved["verdicts_seen"].map(len) > 1
        return resolved

    def counts(self) -> dict[str, int]:
        resolved = self.resolve()
        if resolved.empty:
            return {verdict: 0 for verdict in VERDICTS}
        counts = resolved["verdict"].value_counts().to_dict()
        return {verdict: int(counts.get(verdict, 0)) for verdict in VERDICTS}

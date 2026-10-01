"""Identifiers for every level of the annotation hierarchy, and nothing else.

The release names one participant's files for one interaction
``V<vendor>_S<session>_I<interaction>_P<participant>``. Four facts about those
parts decide how the keys below are built, and each one has already cost a bug
somewhere:

* **``interaction_id`` is a prompt id, not an instance id.** It equals
  ``interactions.csv:prompt_hash`` and the same value recurs in up to 1,550
  sessions, so one conversation is ``(vendor, session, interaction)``.
* **Session ids collide across vendors** (1,722 of 3,039 bare ids), so a session
  is ``(vendor, session)``.
* **Participant ids collide across vendors** (627 bare ids), so a person is
  ``(vendor, participant)``. 20,533 V00 files carry an ``A``-suffixed id
  (``0844A``); it is part of the id and is never stripped.
* **Every part is a zero-padded string.** Parsing any of them as an integer
  destroys ``00000129``, ``0300`` and ``0844A``.

The key formats reuse the file-id segments so a key can be read at a glance and
every parent key is a prefix of its children's::

    session_key      V00_S0039
    interaction_key  V00_S0039_I00000581
    file_id          V00_S0039_I00000581_P0061        (one recording)
    clip_id          V00_S0039_I00000581_P0061_L30_C003   (clip 3 of the 30-s grid)
    window_id        V00_S0039_I00000581_L30_W003         (the dyad over that interval)
    participant_key  V00_P0061

The grid length is part of the clip and window ids: clip 3 means [90, 120) s only
on a 30-s grid, and a run with another clip length must not reuse the same ids for
different spans.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import pandas as pd

FILE_ID_RE = re.compile(
    r"^V(?P<vendor_id>\d{2})_S(?P<session_id>\d{4})_I(?P<interaction_id>\d{8})"
    r"_P(?P<participant_id>[A-Za-z0-9]+)$"
)

#: Width of the clip index in ``clip_id``. The longest recording is 4,554 s,
#: i.e. 152 thirty-second clips; three digits leave room for 10 s clips.
CLIP_INDEX_WIDTH = 3


@dataclass(frozen=True)
class FileId:
    """The four parts of one recording's file id, as strings."""

    vendor: str  # "V00"
    session_id: str  # "0039"
    interaction_id: str  # "00000581"
    participant_id: str  # "0061" or "0844A"

    @property
    def file_id(self) -> str:
        return f"{self.vendor}_S{self.session_id}_I{self.interaction_id}_P{self.participant_id}"

    @property
    def session_key(self) -> str:
        return session_key(self.vendor, self.session_id)

    @property
    def interaction_key(self) -> str:
        return interaction_key(self.vendor, self.session_id, self.interaction_id)

    @property
    def participant_key(self) -> str:
        return participant_key(self.vendor, self.participant_id)


def parse_file_id(file_id: str) -> FileId:
    match = FILE_ID_RE.match(str(file_id))
    if match is None:
        raise ValueError(f"not a Seamless file id: {file_id!r}")
    parts = match.groupdict()
    return FileId(
        vendor="V" + parts["vendor_id"],
        session_id=parts["session_id"],
        interaction_id=parts["interaction_id"],
        participant_id=parts["participant_id"],
    )


def _vendor(vendor: str) -> str:
    """Accept ``"V00"`` or ``"00"``: relationships.csv and participants.csv disagree."""

    text = str(vendor)
    return text if text.startswith("V") else "V" + text.zfill(2)


def session_key(vendor: str, session_id: str) -> str:
    return f"{_vendor(vendor)}_S{str(session_id).zfill(4)}"


def interaction_key(vendor: str, session_id: str, interaction_id: str) -> str:
    return f"{session_key(vendor, session_id)}_I{str(interaction_id).zfill(8)}"


def participant_key(vendor: str, participant_id: str) -> str:
    return f"{_vendor(vendor)}_P{participant_id}"


def grid_token(clip_seconds: float) -> str:
    """``30.0 -> 'L30'``, ``7.5 -> 'L7p5'``: the clip length as an id-safe token."""

    value = float(clip_seconds)
    if value <= 0:
        raise ValueError("clip_seconds must be positive")
    text = f"{value:g}".replace(".", "p")
    return f"L{text}"


def clip_id(file_id: str, clip_index: int, clip_seconds: float) -> str:
    return f"{file_id}_{grid_token(clip_seconds)}_C{int(clip_index):0{CLIP_INDEX_WIDTH}d}"


def window_id(interaction_key_: str, window_index: int, clip_seconds: float) -> str:
    return f"{interaction_key_}_{grid_token(clip_seconds)}_W{int(window_index):0{CLIP_INDEX_WIDTH}d}"


def moi_id(file_id: str, annotation_kind: str, index: int) -> str:
    """``3P-IS`` entry 2 of a file -> ``<file_id>_M3PIS_002``."""

    return f"{file_id}_M{annotation_kind.replace('-', '')}_{int(index):03d}"


def dyad_key(participant_keys: list[str] | tuple[str, ...]) -> str:
    """Order-free key for a pair (or a lone participant) of participant keys."""

    return "+".join(sorted(str(p) for p in participant_keys))


def add_keys(frame: pd.DataFrame, file_id_column: str = "file_id") -> pd.DataFrame:
    """Vectorised: add vendor/session/interaction/participant parts and keys.

    Raises if any file id does not parse, rather than producing NA keys that
    would later drop out of every join.
    """

    parts = frame[file_id_column].astype("string").str.extract(FILE_ID_RE.pattern)
    bad = parts["vendor_id"].isna()
    if bad.any():
        examples = frame.loc[bad, file_id_column].head(5).tolist()
        raise ValueError(f"{int(bad.sum())} unparseable file ids, e.g. {examples}")
    out = frame.copy()
    vendor = "V" + parts["vendor_id"]
    out["vendor"] = vendor.astype("string")
    out["session_id"] = parts["session_id"].astype("string")
    out["interaction_id"] = parts["interaction_id"].astype("string")
    out["participant_id"] = parts["participant_id"].astype("string")
    out["session_key"] = (vendor + "_S" + parts["session_id"]).astype("string")
    out["interaction_key"] = (out["session_key"] + "_I" + parts["interaction_id"]).astype("string")
    out["participant_key"] = (vendor + "_P" + parts["participant_id"]).astype("string")
    return out

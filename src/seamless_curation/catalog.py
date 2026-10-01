"""The metadata-only parts of every table: what is known before any file is opened.

The census (``outputs/02_inventory/summary/inventory_joined.parquet``, one row
per ``filelist.csv`` file, with the ffprobe/stat results of M-1) and Meta's
three metadata CSVs are enough to build every identity, link, partition and
given-attribute column of the participants, sessions, prompts, interactions
and recordings tables. This module does that and nothing else: no NPZ, JSON,
WAV or MP4 is read, so the result is reproducible in seconds and the scan can
join its measurements onto it by key. Column names, dtypes and NA meanings are
the registry's (:mod:`seamless_curation.schema`); every frame is conformed to
it on the way out, so a registry change that this module does not follow fails
here rather than in the written tables.

Reading the CSVs
----------------
All four files are read ``dtype=str, keep_default_na=False``: every id is a
zero-padded string (``00000129``, ``0300``, ``0844A``) and ``'Undisclosed'``
sits inside the BFI columns, so any numeric inference destroys data.
``interactions.csv`` has quoted multi-line prompt texts (16,267 physical
lines for 1,312 rows), which the C parser handles because the fields are
quoted. The vendor is written ``'00'`` in participants.csv but ``'V00'`` in
relationships.csv; both are normalised through :func:`ids.participant_key` /
:func:`ids.session_key`. Exact duplicate rows are dropped; two rows with the
same key and different values raise (the old inventory's ``drop_duplicates``
would have hidden such a conflict; the current release has none).

Recordings (129,370 rows; one per filelist file)
------------------------------------------------
Keys come from the file id (:mod:`seamless_curation.ids`): ``prompt_hash`` is
the I-segment, which Meta calls ``interaction_id`` but which is a prompt id
reused across up to 1,550 sessions. The census columns (presence, probe
status, raster, frame rate, frame count, stream duration) are carried as
M-1 measured them. ``video_bitrate_mbps = mp4_size_bytes * 8 / 1e6 /
video_duration_s`` (NA without a video stream, i.e. the 623 files whose probe
is not ``ok``).

``raster_class`` is decided by the stored (width, height) alone, because in
this release every raster occurs in exactly one vendor (census crosstab):
1080x1920 V00 42,826 / V01 6,557 / V02 30,146 / V03 30; 2160x3840 V03 42,297;
2160x2160 V01 5,211; 1012x1920 V01 102; 1920x1080 V01 64; 3840x2160 V03
1,405; 640x480 V03 61; 1080x960 V03 28; 2180x3840 V03 20; no video 623. The
V01 anamorphic rasters were stored with non-square pixels and the released
SMPL-H absorbed the stretch (``corpus.EXCLUDED_RASTERS`` dropped 2160x2160
and 1920x1080 for this reason): ``smplh_anamorphic`` is 'mild' for 1012x1920
(a 270:253, 6.7 % squeeze), 'severe' for 2160x2160 and 1920x1080, else
'none'. ``room_camera_rig`` marks V03's 3840x2160 and 640x480 far-field rig
(voice isolation 2.2/3.8 dB against 7-13 dB on close microphones).

``partner_file_id``: interactions have one or two members (64,619 pairs, 132
singletons in 9 single-participant V00 sessions; never more than two, which
is enforced). ``prompt_repeat_count_for_participant`` counts the OTHER
sessions in which the same participant received the same prompt: 10,605
(participant, prompt) pairs over 771 participants recur, and 26,020
recordings have a count >= 1.

Participants (4,307 rows; the vendor-qualified ids that have files)
-------------------------------------------------------------------
``metadata_status``: 'numeric' when all five BFI-2 domain cells parse as
numbers (2,280 participants), 'undisclosed' when the row exists but they do
not (in the release the five are always 'Undisclosed' together: 1,937 of the
4,284 csv rows, 1,860 of them for participants with files), 'missing_row'
when there is no row for this vendor+id (167 participants, 2,700 files;
144 csv rows belong to ids without files). BFI is
never borrowed from another vendor's row, although 39 of the missing ids do
exist under another vendor: bare ids collide across vendors (627 of them).
A row that mixed numbers and 'Undisclosed' would be 'undisclosed' with all
five scores NA, so a score is never published without the status that says
it is complete.

``suffix_sibling_key`` links ``NNNNA`` and ``NNNN`` of the same vendor when
both have files (51 V00 stems; only V00 uses the suffix). The two ids never
share a session, 182 of 183 suffixed metadata rows are 'Undisclosed', and
Meta warns ids are sometimes duplicated or split across people, so whether
siblings are one person is unknown: the link is published, the identity is
not assumed. ``flag_suffix_sibling_split_conflict`` marks both ids of a stem
whose split sets differ (7 stems, 14 ids).

``participant_component_id`` is the connected component of the graph whose
edges join the two participants of each session (union-find), named by its
smallest participant_key. Grouping by it keeps every dyad and every
partner-of-partner on one side of a custom split.

Splits are Meta's, per file, as released. 26 participants have files in two
splits (15 dev|train, 11 test|train; all also span both labels):
``flag_split_conflict``. On interactions, ``n_members_in_other_split`` counts
the members whose participant has any file in a split other than the
interaction's, and ``n_members_suffix_sibling_other_split`` the members whose
suffix sibling does.

Sessions (5,102 rows)
---------------------
``relationship``/``relationship_detail`` are joined on the vendor-qualified
session key (session ids collide across vendors: 1,722 of 3,039 bare ids);
122 sessions have no row (112 V01, the 9 single-participant V00 sessions,
1 V03): ``relationship_status = 'missing_row'`` and both values NA. Label and
split are checked constant within every session and interaction (0
exceptions in the release); a violation raises, because the partition columns
copied onto child rows would otherwise be wrong.

Prompts (1,312 rows, including the 20 prompts no file uses)
-----------------------------------------------------------
:func:`parse_prompt_id` reads the families of ``prompt_id_unique``:
``P####_v3.x.y.RP_<ipc>_<ipc>`` (320), ``RPx.y_<ipc>_<ipc>`` (218; no template
and no version token: ``x.y`` runs 0.0-3.6 and is a scenario number rather
than a prompt-set version, so it stays inside prompt_id_unique),
``P####_v3.x.{fam,str}_<ipc>_XXXX`` (160 + 160), ``P####_v3.x[.]aac_...[-n]``
(250; the dot before ``aac`` is missing in 18), ``P####_v3.x.y.AC_...[-n]``
(64), ``P####_{AA,CC,DN,DP,EO,MN,MP}_...[-n]`` (~140, six of them with the
IPC code glued to a number, ``P0064_AA_ANCP4134``). The ``-n`` item index
occurs on exactly the 322 grounded_gesture prompts.

:func:`decode_ipc` reads ``A{P,N,M}C{P,N,M}`` as agency and communion
+1/0/-1; ``CGST`` (personas "Confused", "Alert", "Zoned out": a cognitive
state, not an octant) and an empty code decode to NA axes.
``ab_prompts_differ`` compares the A and B texts after collapsing whitespace
(1,287 of 1,312 prompts differ, 116,484 of 129,370 files; one prompt differs
only in whitespace). ``ipc_member_assignable`` is true only when both roles
got the same text (and ipc_b is absent or equal to ipc_a): 25 prompts, all
with an empty ipc_b.

:func:`task_kind_from_text` applies transparent keyword rules to the A text,
first match wins, in this order (counts over the 1,312 prompts):

``gesture_sentence``  "test your acting skills" / "list(s) of sentences" /
    "bolded" / "act-out a corresponding gesture": the grounded-gesture game
    (325: all 322 grounded_gesture prompts and 3 ipc_conversation-typed
    fragments of that text).
``storytelling``  "fictional tale|story", "weav... a tale|story", "build the
    story together": collaborative fiction (39). A personal "tell a story
    about a time..." is a discussion, not storytelling ('story' occurs in 62
    ipc_conversation texts).
``game``  "play a game", "list(s) of words or phrases", "silly and fun", "no
    spoken words", "charades" (5). Checked after the two kinds above because
    both of their texts also open "You're going to play a game with your
    partner".
``role_play``  the text opens with "Scenario:" (the RPx.y family) or a
    parenthesised persona tag, "(Disorganized): As the employee..." (the
    P####_v3.x.y.RP family): 538, exactly the 538 RP-family prompts.
``discussion``  an instruction verb of conversation (discuss, tell, talk,
    describe, share, ask, recall, remember, think, reflect, explain,
    brainstorm, suggest, identify, decide, agree, debate, plan), "would you
    rather", "your partner will", "with your partner", conversation/opinion/
    story/experience/advice, or a question mark (401).
``unclear``  nothing matched (4: texts that are only "You will each see the
    same prompt:", truncated in the release; 3,837 files).

``interaction_type_text_consistency`` pairs Meta's type with the kind
(ipc_conversation with discussion or role_play, collaborative_storytelling
with storytelling, charades with game, grounded_gesture with
gesture_sentence); 'unclear' when the kind is. On the release: 1,299 prompts
consistent, 4 unclear, 9 contradicts_text covering 8,007 files - 00000135
(3,091 files; typed storytelling, text "What is one word you can use to
describe how you and your partner are most different? For"), 00000129
(1,350; typed charades, a deep-conversation question), 00000130 (926; typed
ipc_conversation, the "silly and fun" game), 00000131 (868; the charades
word-list fragment), 00000139 (838; typed ipc_conversation, a fictional
tale), 00000137 (748; a gesture-sentence fragment), 00000329 (174; typed
storytelling, a discussion), 00000508 (8) and 00000577 (4). Speech behaviour
where checked follows the type, not the text (Meta's off-by-one prompt
caveat), so the column flags the text, not the type.

Interactions (64,751 rows)
--------------------------
Members are sorted by participant id (the order carries no prompt-role
meaning; Meta publishes no participant-to-role mapping). ``moi_3p_coverage``
/ ``moi_1p_coverage`` count the members whose filelist flag says they carry
that party's MOI annotations: 'both' needs two members, so a singleton
interaction is at most 'one' (none of the 132 singletons is annotated).
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from . import ids, schema

# =============================================================================
# Constants
# =============================================================================

#: Stored (width, height) -> registry raster class. Anything else with a size
#: is 'other'; no size (no video stream) is 'no_video'.
RASTER_CLASS_BY_SIZE: dict[tuple[int, int], str] = {
    (1080, 1920): "portrait_1080x1920",
    (2160, 3840): "portrait_2160x3840",
    (1012, 1920): "anamorphic_1012x1920",
    (2160, 2160): "anamorphic_2160x2160",
    (1920, 1080): "anamorphic_1920x1080",
    (3840, 2160): "rotated_3840x2160",
    (640, 480): "rotated_640x480",
    (1080, 960): "pillarbox_1080x960",
    (2180, 3840): "pillarbox_2180x3840",
}
NO_VIDEO = "no_video"
OTHER_RASTER = "other"

#: Raster class -> how badly the released SMPL-H absorbed the anamorphic stretch.
SMPLH_ANAMORPHIC_BY_CLASS: dict[str, str] = {
    "anamorphic_1012x1920": "mild",
    "anamorphic_2160x2160": "severe",
    "anamorphic_1920x1080": "severe",
}

#: V03's far-field room-camera rig.
ROOM_CAMERA_VENDOR = "V03"
ROOM_CAMERA_CLASSES: tuple[str, ...] = ("rotated_3840x2160", "rotated_640x480")

BFI_DOMAINS: tuple[str, ...] = ("extraversion", "agreeableness", "conscientiousness", "neuroticism", "openness")

#: Meta's interaction_type -> the text-derived task kinds it is consistent with.
CONSISTENT_KINDS: dict[str, frozenset[str]] = {
    "ipc_conversation": frozenset({"discussion", "role_play"}),
    "collaborative_storytelling": frozenset({"storytelling"}),
    "charades": frozenset({"game"}),
    "grounded_gesture": frozenset({"gesture_sentence"}),
}

#: First match wins, in this order (see the module docstring for the evidence).
TASK_KIND_RULES: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("gesture_sentence", re.compile(
        r"test your acting skills|\blists? of sentences\b|\bbolded\b|act[- ]out a corresponding gesture", re.I)),
    ("storytelling", re.compile(
        r"\bfictional (?:tale|story)\b|\bweav\w* an? (?:fictional )?(?:tale|story)\b|\bbuild the story together\b", re.I)),
    ("game", re.compile(
        r"\bplay a game\b|\blists? of words or phrases\b|\bsilly and fun\b|\bno spoken words\b|\bcharades\b", re.I)),
    ("role_play", re.compile(r"^\W*scenario\s*:|^\W*\([^()]{1,40}\)", re.I)),
    ("discussion", re.compile(
        r"\b(?:discuss\w*|tell|talk\w*|describe|share|ask|recall|remember|think|reflect|explain|"
        r"brainstorm|suggest|identify|decide|agree|debate|plan|conversation|opinion|story|stories|"
        r"experience|advice)\b|\bwould you rather\b|\byour partner will\b|\bwith your partner\b|\?", re.I)),
)
UNCLEAR_KIND = "unclear"

_IPC_OCTANT_RE = re.compile(r"^A([PNM])C([PNM])$")
_IPC_AXIS = {"P": 1, "N": 0, "M": -1}
_IPC_COGNITIVE_STATE = "CGST"

_TEMPLATE_RE = re.compile(r"^(P\d{4})(?=_)")
_VERSION_RE = re.compile(r"_v(\d+(?:\.\d+)*)")
_VERSION_CONTEXT_RE = re.compile(r"_v\d+(?:\.\d+)*\.?([A-Za-z]+)_")
_SET_CODE_RE = re.compile(r"^P\d{4}_([A-Za-z]{2,3})_")
_RP_FAMILY_RE = re.compile(r"^RP\d")
_ITEM_INDEX_RE = re.compile(r"-(\d+)$")
_PROMPT_CONTEXTS: frozenset[str] = frozenset(
    a for a in schema.field("prompts", "prompt_context").allowed if a != "other"
)

_PARTICIPANT_ID_RE = re.compile(r"^(?P<stem>\d+)(?P<suffix>[A-Za-z]*)$")

#: Census columns the catalog reads (a missing one raises, naming it).
INVENTORY_COLUMNS: tuple[str, ...] = (
    "file_id", "label", "split", "batch_idx", "archive_idx",
    "has_imitator_movement", "has_annotation_1p", "has_annotation_3p", "source_relbase",
    "json_present", "mp4_present", "npz_present", "wav_present", "probe_status",
    "video_width", "video_height", "video_r_fps", "video_nb_frames", "video_duration_s",
    "mp4_size_bytes",
)
INTERACTIONS_CSV_COLUMNS: tuple[str, ...] = (
    "prompt_hash", "prompt_id_unique", "participant_a_prompt_text", "participant_b_prompt_text",
    "ipc_a", "ipc_b", "interaction_type",
)
PARTICIPANTS_CSV_COLUMNS: tuple[str, ...] = ("vendor_id", "participant_id") + tuple(f"{d}_raw" for d in BFI_DOMAINS)
RELATIONSHIPS_CSV_COLUMNS: tuple[str, ...] = ("vendor_id", "session_id", "relationship", "relationship_detail")

# The catalog-built (metadata-only) columns of each table. The rest of each
# table is measured by the scan and joined later by key.
RECORDING_COLUMNS: tuple[str, ...] = (
    "file_id", "interaction_key", "session_key", "participant_key", "prompt_hash",
    "vendor", "label", "split", "session_id", "participant_id", "batch_idx", "archive_idx",
    "source_relbase", "partner_file_id", "partner_participant_key",
    "has_imitator_movement", "has_annotation_1p", "has_annotation_3p",
    "json_present", "mp4_present", "npz_present", "wav_present", "probe_status",
    "video_width", "video_height", "fps", "video_nb_frames", "video_duration_s", "video_bitrate_mbps",
    "raster_class", "smplh_anamorphic", "room_camera_rig", "prompt_repeat_count_for_participant",
)
PARTICIPANT_COLUMNS: tuple[str, ...] = tuple(
    c for c in schema.columns("participants") if c != "participant_recorded_hours"
)
SESSION_COLUMNS: tuple[str, ...] = (
    "session_key", "vendor", "label", "split", "session_id",
    "relationship", "relationship_detail", "relationship_status",
    "participant_key_1", "participant_key_2", "dyad_key", "dyad_session_count",
    "session_n_participants", "session_n_interactions", "session_n_interaction_types",
)
PROMPT_COLUMNS: tuple[str, ...] = tuple(schema.columns("prompts"))
INTERACTION_COLUMNS: tuple[str, ...] = (
    "interaction_key", "session_key", "prompt_hash", "vendor", "label", "split",
    "interaction_n_recordings", "pairing_status", "member_file_id_1", "member_file_id_2",
    "moi_3p_coverage", "moi_1p_coverage", "n_members_in_other_split",
    "n_members_suffix_sibling_other_split",
)


# =============================================================================
# The catalog
# =============================================================================


@dataclass
class Catalog:
    """Metadata-only parts of the five metadata-bearing tables, registry-conformed."""

    recordings: pd.DataFrame
    participants: pd.DataFrame
    sessions: pd.DataFrame
    prompts: pd.DataFrame
    interactions: pd.DataFrame

    def tables(self) -> dict[str, pd.DataFrame]:
        return {
            "recordings": self.recordings,
            "participants": self.participants,
            "sessions": self.sessions,
            "prompts": self.prompts,
            "interactions": self.interactions,
        }


def build_catalog(inventory: pd.DataFrame, metadata_root: str | Path) -> Catalog:
    """Build every metadata-only column from the M-1 census and Meta's CSVs.

    ``inventory`` is the census (one row per filelist file); ``metadata_root``
    holds ``interactions.csv``, ``participants.csv`` and ``relationships.csv``.
    The recordings, participants, sessions and interactions tables cover
    exactly the census files; the prompts table covers every interactions.csv
    row, used or not.
    """

    root = Path(metadata_root)
    prompts_csv = read_metadata_csv(root / "interactions.csv", INTERACTIONS_CSV_COLUMNS, key=("prompt_hash",))
    participants_csv = read_metadata_csv(
        root / "participants.csv", PARTICIPANTS_CSV_COLUMNS, key=("vendor_id", "participant_id"),
        key_normaliser=lambda f: f["vendor_id"].map(ids._vendor) + "_P" + f["participant_id"],
    )
    relationships_csv = read_metadata_csv(
        root / "relationships.csv", RELATIONSHIPS_CSV_COLUMNS, key=("vendor_id", "session_id"),
        key_normaliser=lambda f: pd.Series(
            [ids.session_key(v, s) for v, s in zip(f["vendor_id"], f["session_id"])], index=f.index, dtype=object
        ),
    )

    recordings = _recordings(inventory)
    prompts = _prompts(prompts_csv, recordings)
    participants = _participants(recordings, participants_csv)
    sessions = _sessions(recordings, relationships_csv, prompts)
    interactions = _interactions(recordings, participants)
    return Catalog(
        recordings=conform(recordings, "recordings", RECORDING_COLUMNS, sort_by="file_id"),
        participants=conform(participants, "participants", PARTICIPANT_COLUMNS, sort_by="participant_key"),
        sessions=conform(sessions, "sessions", SESSION_COLUMNS, sort_by="session_key"),
        prompts=conform(prompts, "prompts", PROMPT_COLUMNS, sort_by="prompt_hash"),
        interactions=conform(interactions, "interactions", INTERACTION_COLUMNS, sort_by="interaction_key"),
    )


# =============================================================================
# Scalar helpers (public)
# =============================================================================


def raster_class(width, height) -> str:
    """Registry raster class of one stored (width, height); 'no_video' if either is missing."""

    if _is_missing(width) or _is_missing(height):
        return NO_VIDEO
    w, h = float(width), float(height)
    if not (w.is_integer() and h.is_integer()):
        return OTHER_RASTER
    return RASTER_CLASS_BY_SIZE.get((int(w), int(h)), OTHER_RASTER)


def raster_classes(width: Sequence | pd.Series, height: Sequence | pd.Series) -> pd.Series:
    """Vectorised :func:`raster_class`."""

    w = pd.to_numeric(pd.Series(width).reset_index(drop=True), errors="coerce")
    h = pd.to_numeric(pd.Series(height).reset_index(drop=True), errors="coerce")
    by_text = {f"{a}x{b}": cls for (a, b), cls in RASTER_CLASS_BY_SIZE.items()}
    integral = (w.round() == w) & (h.round() == h)
    # Non-integral sizes are masked before the integer cast (which would raise).
    text = (w.where(integral).astype("Int64").astype("string") + "x"
            + h.where(integral).astype("Int64").astype("string"))
    out = text.map(by_text).astype(object)
    out = out.where(integral & out.notna(), OTHER_RASTER)
    out[w.isna() | h.isna()] = NO_VIDEO
    return out.astype(object)


def smplh_anamorphic_for(raster_class_: str) -> str:
    """'mild' (1012x1920), 'severe' (2160x2160, 1920x1080) or 'none'."""

    return SMPLH_ANAMORPHIC_BY_CLASS.get(str(raster_class_), "none")


def parse_prompt_id(prompt_id_unique: str) -> dict:
    """Split ``prompt_id_unique`` into template, version, context and item index.

    >>> parse_prompt_id("P0002_v3.4.aac_ANCP_XXXX-148")
    {'template': 'P0002', 'version': 'v3.4', 'context': 'aac', 'item_index': 148}
    >>> parse_prompt_id("RP2.0_AMCP_ANCP")
    {'template': None, 'version': None, 'context': 'RP', 'item_index': None}
    """

    text = str(prompt_id_unique or "").strip()
    template = _TEMPLATE_RE.match(text)
    version = _VERSION_RE.search(text)
    if _RP_FAMILY_RE.match(text):
        context = "RP"
    else:
        after_version = _VERSION_CONTEXT_RE.search(text)
        set_code = _SET_CODE_RE.match(text) if after_version is None else None
        token = after_version.group(1) if after_version else (set_code.group(1) if set_code else None)
        context = token if token in _PROMPT_CONTEXTS else "other"
    item = _ITEM_INDEX_RE.search(text)
    return {
        "template": template.group(1) if template else None,
        "version": ("v" + version.group(1)) if version else None,
        "context": context,
        "item_index": int(item.group(1)) if item else None,
    }


def decode_ipc(code: str | None) -> tuple[int | None, int | None, str]:
    """``'APCM' -> (1, -1, 'octant')``; ``'CGST' -> (None, None, 'cognitive_state')``;
    empty/None -> ``(None, None, 'absent')``. Anything else raises."""

    if code is None or (isinstance(code, float) and math.isnan(code)):
        return None, None, "absent"
    text = str(code).strip()
    if text == "" or text.upper() in ("NA", "<NA>"):
        return None, None, "absent"
    if text == _IPC_COGNITIVE_STATE:
        return None, None, "cognitive_state"
    match = _IPC_OCTANT_RE.match(text)
    if match is None:
        raise ValueError(f"not an interpersonal-circumplex code: {code!r}")
    return _IPC_AXIS[match.group(1)], _IPC_AXIS[match.group(2)], "octant"


def task_kind_from_text(text: str) -> str:
    """Task kind of a prompt text by the keyword rules in :data:`TASK_KIND_RULES`."""

    if text is None or (isinstance(text, float) and math.isnan(text)):
        return UNCLEAR_KIND
    flat = " ".join(str(text).split())
    for kind, pattern in TASK_KIND_RULES:
        if pattern.search(flat):
            return kind
    return UNCLEAR_KIND


def interaction_type_text_consistency(interaction_type: str | None, task_kind: str) -> str:
    """'consistent', 'contradicts_text', or 'unclear' (kind unclear or type unknown)."""

    allowed = CONSISTENT_KINDS.get(str(interaction_type)) if interaction_type is not None else None
    if task_kind == UNCLEAR_KIND or allowed is None:
        return "unclear"
    return "consistent" if task_kind in allowed else "contradicts_text"


def participant_components(pairs: Iterable[tuple[str, str]], participants: Iterable[str]) -> dict[str, str]:
    """Union-find: each participant -> the smallest participant key of its component."""

    parent: dict[str, str] = {str(p): str(p) for p in participants}

    def find(node: str) -> str:
        root = node
        while parent[root] != root:
            root = parent[root]
        while parent[node] != root:
            parent[node], node = root, parent[node]
        return root

    for a, b in pairs:
        a, b = str(a), str(b)
        parent.setdefault(a, a)
        parent.setdefault(b, b)
        ra, rb = find(a), find(b)
        if ra == rb:
            continue
        # The smaller key stays the root, so every root is its component's minimum.
        if rb < ra:
            ra, rb = rb, ra
        parent[rb] = ra
    return {node: find(node) for node in parent}


def read_metadata_csv(
    path: str | Path,
    required: Sequence[str],
    *,
    key: Sequence[str] = (),
    key_normaliser=None,
) -> pd.DataFrame:
    """Read one of Meta's CSVs as strings; drop exact duplicate rows; raise on key conflicts."""

    frame = pd.read_csv(path, dtype=str, keep_default_na=False)
    missing = [c for c in required if c not in frame.columns]
    if missing:
        raise ValueError(f"{Path(path).name} lacks columns {missing}")
    frame = frame.drop(columns=[c for c in frame.columns if str(c).startswith("Unnamed")])
    frame = frame.drop_duplicates().reset_index(drop=True)
    if key:
        normalised = key_normaliser(frame) if key_normaliser is not None else frame[list(key)].agg("\x1f".join, axis=1)
        duplicated = normalised.duplicated(keep=False)
        if duplicated.any():
            examples = sorted(set(normalised[duplicated]))[:5]
            raise ValueError(f"{Path(path).name}: {int(duplicated.sum())} rows share a key with different values, e.g. {examples}")
        frame["_key"] = normalised.to_numpy()
    return frame


# =============================================================================
# Table builders
# =============================================================================


def _recordings(inventory: pd.DataFrame) -> pd.DataFrame:
    missing = [c for c in INVENTORY_COLUMNS if c not in inventory.columns]
    if missing:
        raise ValueError(f"inventory lacks columns {missing}")
    source = inventory.loc[:, list(INVENTORY_COLUMNS)].reset_index(drop=True)
    if source["file_id"].duplicated().any():
        dupes = source.loc[source["file_id"].duplicated(), "file_id"].head(5).tolist()
        raise ValueError(f"inventory has duplicate file ids, e.g. {dupes}")
    frame = ids.add_keys(source)
    frame["prompt_hash"] = frame["interaction_id"]
    for column in ("label", "split", "source_relbase", "probe_status"):
        frame[column] = frame[column].astype("string")
    for column in ("has_imitator_movement", "has_annotation_1p", "has_annotation_3p",
                   "json_present", "mp4_present", "npz_present", "wav_present"):
        frame[column] = _as_flag(frame[column], column)
    for column in ("batch_idx", "archive_idx"):
        frame[column] = pd.to_numeric(frame[column], errors="raise")

    width = pd.to_numeric(frame["video_width"], errors="coerce")
    height = pd.to_numeric(frame["video_height"], errors="coerce")
    frame["video_width"] = width
    frame["video_height"] = height
    frame["raster_class"] = raster_classes(width, height).to_numpy()
    frame["smplh_anamorphic"] = frame["raster_class"].map(smplh_anamorphic_for)
    frame["room_camera_rig"] = (frame["vendor"] == ROOM_CAMERA_VENDOR).to_numpy(bool) & frame[
        "raster_class"].isin(ROOM_CAMERA_CLASSES).to_numpy(bool)

    fps = _parse_rate(frame["video_r_fps"])
    frame["fps"] = fps.where(np.isfinite(fps) & (fps > 0))
    frame["video_nb_frames"] = pd.to_numeric(frame["video_nb_frames"], errors="coerce")
    duration = pd.to_numeric(frame["video_duration_s"], errors="coerce")
    frame["video_duration_s"] = duration
    size = pd.to_numeric(frame["mp4_size_bytes"], errors="coerce")
    frame["video_bitrate_mbps"] = (size * 8.0 / 1e6 / duration).where(duration > 0)

    # Partner links. Interactions are dyads; a third member would make
    # partner_file_id ambiguous, so it is refused rather than guessed.
    members = frame.groupby("interaction_key")["file_id"].transform("size")
    if (members > 2).any():
        bad = frame.loc[members > 2, "interaction_key"].drop_duplicates().head(5).tolist()
        raise ValueError(f"interactions with more than two recordings, e.g. {bad}")
    ordered = frame.sort_values(["interaction_key", "participant_id"], kind="stable")
    by_interaction = ordered.groupby("interaction_key", sort=False)
    first_file = by_interaction["file_id"].transform("first").reindex(frame.index)
    last_file = by_interaction["file_id"].transform("last").reindex(frame.index)
    first_person = by_interaction["participant_key"].transform("first").reindex(frame.index)
    last_person = by_interaction["participant_key"].transform("last").reindex(frame.index)
    is_first = frame["file_id"] == first_file
    paired = members == 2
    frame["partner_file_id"] = last_file.where(is_first, first_file).where(paired, None)
    frame["partner_participant_key"] = last_person.where(is_first, first_person).where(paired, None)
    frame["member_rank"] = by_interaction.cumcount().reindex(frame.index).astype(int)
    frame["interaction_n_recordings"] = members

    sessions_with_prompt = frame.groupby(["participant_key", "prompt_hash"])["session_key"].transform("nunique")
    frame["prompt_repeat_count_for_participant"] = sessions_with_prompt - 1
    return frame


def _prompts(csv: pd.DataFrame, recordings: pd.DataFrame) -> pd.DataFrame:
    frame = csv.drop(columns=["_key"]).copy()
    frame["prompt_hash"] = frame["prompt_hash"].map(_pad_prompt_hash)
    if frame["prompt_hash"].duplicated().any():
        raise ValueError("interactions.csv prompt_hash is not unique after zero padding")
    parsed = pd.DataFrame([parse_prompt_id(v) for v in frame["prompt_id_unique"]], index=frame.index)
    frame["prompt_template"] = parsed["template"]
    frame["prompt_version"] = parsed["version"]
    frame["prompt_context"] = parsed["context"]
    frame["prompt_item_index"] = pd.array(parsed["item_index"].tolist(), dtype="Int64")

    a_text = frame["participant_a_prompt_text"]
    b_text = frame["participant_b_prompt_text"]
    frame["participant_b_prompt_text"] = b_text.where(b_text.str.strip() != "", None)
    same_text = a_text.map(_flatten) == b_text.map(_flatten)
    frame["ab_prompts_differ"] = ~same_text

    for role in ("a", "b"):
        code = frame[f"ipc_{role}"].str.strip()
        frame[f"ipc_{role}"] = code.where(code != "", None)
        decoded = [decode_ipc(c) for c in frame[f"ipc_{role}"]]
        frame[f"ipc_{role}_agency"] = pd.array([d[0] for d in decoded], dtype="Int8")
        frame[f"ipc_{role}_communion"] = pd.array([d[1] for d in decoded], dtype="Int8")
        frame[f"ipc_{role}_kind"] = [d[2] for d in decoded]
    ipc_b_matches = frame["ipc_b"].isna() | (frame["ipc_b"] == frame["ipc_a"])
    frame["ipc_member_assignable"] = same_text & ipc_b_matches.astype(bool)

    frame["task_kind_from_text"] = a_text.map(task_kind_from_text)
    frame["interaction_type_text_consistency"] = [
        interaction_type_text_consistency(t, k)
        for t, k in zip(frame["interaction_type"], frame["task_kind_from_text"])
    ]

    usage = recordings.groupby("prompt_hash").agg(
        prompt_n_sessions=("session_key", "nunique"), prompt_n_vendors=("vendor", "nunique")
    )
    frame = frame.merge(usage, left_on="prompt_hash", right_index=True, how="left")
    for column in ("prompt_n_sessions", "prompt_n_vendors"):
        frame[column] = frame[column].fillna(0).astype(int)
    return frame


def _participants(recordings: pd.DataFrame, csv: pd.DataFrame) -> pd.DataFrame:
    grouped = recordings.groupby("participant_key", sort=True)
    frame = grouped.agg(
        vendor=("vendor", "first"),
        participant_id=("participant_id", "first"),
        participant_n_sessions=("session_key", "nunique"),
        participant_n_interactions=("interaction_key", "nunique"),
        participant_n_recordings=("file_id", "size"),
        participant_n_partners=("partner_participant_key", "nunique"),
    )
    splits = _membership(recordings, "split", schema.SPLITS)
    labels = _membership(recordings, "label", schema.LABELS)
    for split in schema.SPLITS:
        frame[f"in_{split}"] = splits[split].reindex(frame.index).to_numpy(bool)
    for label in schema.LABELS:
        frame[f"in_{label}"] = labels[label].reindex(frame.index).to_numpy(bool)
    split_sets = splits.reindex(frame.index)
    frame["flag_split_conflict"] = split_sets.sum(axis=1).to_numpy() > 1

    parts = frame["participant_id"].astype(str).str.extract(_PARTICIPANT_ID_RE.pattern)
    suffix = parts["suffix"].fillna("")
    frame["id_suffix"] = suffix.where(suffix != "", "none")
    stem_key = (frame["vendor"].astype(str) + "\x1f" + parts["stem"]).where(parts["stem"].notna())
    frame["suffix_sibling_key"] = _suffix_siblings(frame.index.to_series(), stem_key, suffix)
    sibling = frame["suffix_sibling_key"]
    has_sibling = sibling.notna()
    sibling_sets = split_sets.reindex(sibling.where(has_sibling, "").to_numpy()).to_numpy(bool)
    differs = (sibling_sets != split_sets.to_numpy(bool)).any(axis=1)
    frame["flag_suffix_sibling_split_conflict"] = has_sibling.to_numpy() & differs

    bfi = csv.set_index("_key")
    matched = bfi.reindex(frame.index)
    has_row = frame.index.isin(bfi.index)
    scores = pd.DataFrame(
        {d: pd.to_numeric(matched[f"{d}_raw"], errors="coerce") for d in BFI_DOMAINS}, index=frame.index
    )
    numeric = has_row & np.isfinite(scores.to_numpy(float)).all(axis=1)
    frame["metadata_status"] = np.where(numeric, "numeric", np.where(has_row, "undisclosed", "missing_row"))
    for domain in BFI_DOMAINS:
        frame[f"bfi_{domain}"] = scores[domain].where(numeric)

    session_members = recordings[["session_key", "participant_key"]].drop_duplicates().sort_values(
        ["session_key", "participant_key"])
    edges = []
    for _, keys in session_members.groupby("session_key")["participant_key"]:
        keys = keys.tolist()
        edges.extend((keys[0], other) for other in keys[1:])
    components = participant_components(edges, frame.index)
    frame["participant_component_id"] = frame.index.map(components)
    return frame.reset_index()


def _sessions(recordings: pd.DataFrame, relationships: pd.DataFrame, prompts: pd.DataFrame) -> pd.DataFrame:
    _require_constant(recordings, "session_key", ("label", "split"))
    grouped = recordings.groupby("session_key", sort=True)
    frame = grouped.agg(
        vendor=("vendor", "first"), label=("label", "first"), split=("split", "first"),
        session_id=("session_id", "first"),
        session_n_participants=("participant_key", "nunique"),
        session_n_interactions=("interaction_key", "nunique"),
    )
    if (frame["session_n_participants"] > 2).any():
        bad = frame.index[frame["session_n_participants"] > 2][:5].tolist()
        raise ValueError(f"sessions with more than two participants, e.g. {bad}")
    members = recordings[["session_key", "participant_key"]].drop_duplicates().sort_values(
        ["session_key", "participant_key"])
    rank = members.groupby("session_key").cumcount()
    first = members[rank == 0].set_index("session_key")["participant_key"]
    second = members[rank == 1].set_index("session_key")["participant_key"]
    frame["participant_key_1"] = first.reindex(frame.index)
    frame["participant_key_2"] = second.reindex(frame.index).astype(object).where(
        second.reindex(frame.index).notna(), None)
    frame["dyad_key"] = [
        ids.dyad_key([a] if b is None else [a, b])
        for a, b in zip(frame["participant_key_1"], frame["participant_key_2"])
    ]
    frame["dyad_session_count"] = frame.groupby("dyad_key")["dyad_key"].transform("size")

    types = recordings[["session_key", "prompt_hash"]].drop_duplicates().merge(
        prompts[["prompt_hash", "interaction_type"]], on="prompt_hash", how="left")
    frame["session_n_interaction_types"] = types.groupby("session_key")["interaction_type"].nunique().reindex(
        frame.index).fillna(0).astype(int)

    rel = relationships.set_index("_key")
    matched = rel.reindex(frame.index)
    present = frame.index.isin(rel.index)
    frame["relationship_status"] = np.where(present, "present", "missing_row")
    for column in ("relationship", "relationship_detail"):
        values = matched[column].astype(object)
        frame[column] = values.where(values.notna() & (values.astype(str).str.strip() != ""), None)
    return frame.reset_index()


def _interactions(recordings: pd.DataFrame, participants: pd.DataFrame) -> pd.DataFrame:
    _require_constant(recordings, "interaction_key", ("label", "split"))
    people = participants.set_index("participant_key")
    split_flags = people[[f"in_{s}" for s in schema.SPLITS]].to_numpy(bool)
    position = pd.Series(np.arange(len(people)), index=people.index)

    member = recordings[["interaction_key", "participant_key", "split", "member_rank", "file_id",
                         "has_annotation_3p", "has_annotation_1p"]].copy()
    own_split = member["split"].astype(str).map({s: i for i, s in enumerate(schema.SPLITS)})
    if own_split.isna().any():
        raise ValueError(f"unknown split values {sorted(set(member['split'][own_split.isna()]))}")
    own_split = own_split.to_numpy(int)
    rows = position.reindex(member["participant_key"]).to_numpy(int)
    flags = split_flags[rows].copy()
    flags[np.arange(len(flags)), own_split] = False
    member["in_other_split"] = flags.any(axis=1)

    sibling = people["suffix_sibling_key"].reindex(member["participant_key"])
    has_sibling = sibling.notna().to_numpy()
    sibling_rows = position.reindex(sibling.where(sibling.notna(), people.index[0]).to_numpy()).to_numpy(int)
    sibling_flags = split_flags[sibling_rows].copy()
    sibling_flags[np.arange(len(sibling_flags)), own_split] = False
    member["sibling_in_other_split"] = has_sibling & sibling_flags.any(axis=1)

    grouped = recordings.groupby("interaction_key", sort=True)
    frame = grouped.agg(
        session_key=("session_key", "first"), prompt_hash=("prompt_hash", "first"),
        vendor=("vendor", "first"), label=("label", "first"), split=("split", "first"),
        interaction_n_recordings=("file_id", "size"),
    )
    frame["pairing_status"] = np.where(frame["interaction_n_recordings"] == 2, "paired", "partner_missing")
    by_rank = member.set_index(["interaction_key", "member_rank"])["file_id"]
    frame["member_file_id_1"] = by_rank.xs(0, level="member_rank").reindex(frame.index)
    second = by_rank[by_rank.index.get_level_values("member_rank") == 1].droplevel("member_rank")
    frame["member_file_id_2"] = second.reindex(frame.index).astype(object).where(
        second.reindex(frame.index).notna(), None)
    per_interaction = member.groupby("interaction_key")
    for party, flag in (("3p", "has_annotation_3p"), ("1p", "has_annotation_1p")):
        count = per_interaction[flag].sum().reindex(frame.index).astype(int)
        frame[f"moi_{party}_coverage"] = np.where(
            (count == 2) & (frame["interaction_n_recordings"] == 2), "both", np.where(count >= 1, "one", "none"))
    frame["n_members_in_other_split"] = per_interaction["in_other_split"].sum().reindex(frame.index).astype(int)
    frame["n_members_suffix_sibling_other_split"] = (
        per_interaction["sibling_in_other_split"].sum().reindex(frame.index).astype(int))
    return frame.reset_index()


# =============================================================================
# Registry conformance
# =============================================================================

_PANDAS_DTYPES = {
    "string": "string", "bool": "bool", "int8": "Int8", "int16": "Int16", "int32": "Int32",
    "int64": "Int64", "float32": "float32", "float64": "float64",
}


def conform(frame: pd.DataFrame, table: str, columns: Sequence[str], *, sort_by: str | None = None) -> pd.DataFrame:
    """Select ``columns`` in registry order and cast each to its registry dtype.

    Strings become pandas ``string`` (NA as ``<NA>``), nullable integers the
    nullable ``Int*`` types and non-nullable ones numpy integers (as
    :func:`schema.conform` does), floats numpy floats (NA as NaN), bools numpy
    bool. A non-nullable column holding NA raises: that is a bug here, not
    missing data. (:func:`schema.conform` itself needs every registered column
    of the table; the catalog builds only the metadata part of each.)
    """

    registered = schema.columns(table)
    unknown = [c for c in columns if c not in registered]
    if unknown:
        raise KeyError(f"{table}: columns not in the registry: {unknown}")
    absent = [c for c in columns if c not in frame.columns]
    if absent:
        raise KeyError(f"{table}: builder did not produce {absent}")
    wanted = set(columns)
    ordered = [c for c in registered if c in wanted]
    source = frame.sort_values(sort_by, kind="stable") if sort_by else frame
    source = source.reset_index(drop=True)
    out: dict[str, pd.Series] = {}
    for name in ordered:
        spec = schema.field(table, name)
        values = source[name]
        dtype = _PANDAS_DTYPES[spec.dtype]
        if spec.dtype == "bool":
            if values.isna().any():
                raise ValueError(f"{table}.{name}: NA in a bool column")
            cast = values.astype(bool)
        elif spec.dtype == "string":
            cast = values.astype(object).where(values.notna(), None).astype("string")
        elif spec.dtype.startswith("int"):
            numeric = pd.to_numeric(values, errors="raise")
            if not spec.nullable and numeric.isna().any():
                raise ValueError(f"{table}.{name}: {int(numeric.isna().sum())} NA values in a non-nullable column")
            cast = numeric.astype(dtype if spec.nullable else spec.dtype)
        else:
            cast = pd.to_numeric(values, errors="raise").astype(dtype)
        if not spec.nullable and cast.isna().any():
            raise ValueError(f"{table}.{name}: {int(cast.isna().sum())} NA values in a non-nullable column")
        out[name] = cast
    return pd.DataFrame(out)


# =============================================================================
# Internals
# =============================================================================


def _is_missing(value) -> bool:
    if value is None or value is pd.NA:
        return True
    try:
        return bool(np.isnan(float(value)))
    except (TypeError, ValueError):
        return True


def _flatten(text) -> str:
    if text is None or (isinstance(text, float) and math.isnan(text)):
        return ""
    return " ".join(str(text).split())


def _pad_prompt_hash(value: str) -> str:
    text = str(value).strip()
    return text.zfill(8) if text.isdigit() else text


def _parse_rate(values: pd.Series) -> pd.Series:
    """Frame rates as floats; accepts ffprobe's ``'30000/1001'`` text as well as numbers."""

    numeric = pd.to_numeric(values, errors="coerce")
    text = values.astype(str).str.strip()
    fraction = text.str.fullmatch(r"\d+(?:\.\d+)?/\d+(?:\.\d+)?") & numeric.isna()
    if fraction.any():
        parts = text[fraction].str.split("/", expand=True).astype(float)
        numeric[fraction] = (parts[0] / parts[1]).where(parts[1] > 0)
    return numeric.astype(float)


def _as_flag(values: pd.Series, name: str) -> pd.Series:
    """0/1, True/False (numpy, pandas or text) -> bool; anything else raises."""

    mapping = {"1": True, "0": False, "true": True, "false": False, "1.0": True, "0.0": False}
    if values.isna().any():
        raise ValueError(f"inventory.{name}: NA values")
    text = values.astype(str).str.strip().str.lower()
    unknown = sorted(set(text) - set(mapping))
    if unknown:
        raise ValueError(f"inventory.{name}: unexpected values {unknown[:5]}")
    return text.map(mapping).astype(bool)


def _membership(recordings: pd.DataFrame, column: str, values: Sequence[str]) -> pd.DataFrame:
    """participant_key x value -> has at least one recording with that value."""

    table = pd.crosstab(recordings["participant_key"], recordings[column].astype(str)) > 0
    return table.reindex(columns=list(values), fill_value=False)


def _suffix_siblings(keys: pd.Series, stem_key: pd.Series, suffix: pd.Series) -> pd.Series:
    """Link a suffixed id and its bare id when the vendor+stem has exactly these two."""

    frame = pd.DataFrame({"key": keys.to_numpy(), "stem": stem_key.to_numpy(), "suffix": suffix.to_numpy()})
    out = pd.Series([None] * len(frame), index=keys.index, dtype=object)
    usable = frame[frame["stem"].notna()]
    for _, group in usable.groupby("stem"):
        if len(group) != 2:
            continue
        bare = group[group["suffix"] == ""]
        suffixed = group[group["suffix"] != ""]
        if len(bare) != 1 or len(suffixed) != 1:
            continue
        a, b = bare["key"].iloc[0], suffixed["key"].iloc[0]
        out.iloc[bare.index[0]] = b
        out.iloc[suffixed.index[0]] = a
    return out


def _require_constant(recordings: pd.DataFrame, key: str, columns: Iterable[str]) -> None:
    for column in columns:
        distinct = recordings.groupby(key)[column].nunique(dropna=False)
        if (distinct > 1).any():
            bad = distinct.index[distinct > 1][:5].tolist()
            raise ValueError(f"{column} is not constant within {key}, e.g. {bad}")


def summary(catalog: Catalog) -> Mapping[str, object]:
    """Counts the tests and the report quote (all derived from the catalog itself)."""

    prompts = catalog.prompts
    recordings = catalog.recordings
    contradicting = prompts.loc[prompts["interaction_type_text_consistency"] == "contradicts_text", "prompt_hash"]
    return {
        "recordings": int(len(recordings)),
        "interactions": int(len(catalog.interactions)),
        "partner_missing": int((catalog.interactions["pairing_status"] == "partner_missing").sum()),
        "sessions": int(len(catalog.sessions)),
        "participants": int(len(catalog.participants)),
        "split_conflicts": int(catalog.participants["flag_split_conflict"].sum()),
        "prompts": int(len(prompts)),
        "prompts_contradicting_text": sorted(contradicting.astype(str).tolist()),
        "files_contradicting_text": int(recordings["prompt_hash"].isin(set(contradicting)).sum()),
    }

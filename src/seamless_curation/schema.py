"""The annotation schema: every published column, its level, meaning and provenance.

This registry is the single source of truth for the annotation tables. The
validator (:mod:`seamless_curation.validate`) checks every written table against
it, the parquet writer derives each table's Arrow schema from it, and
``docs/annotation_schema.md`` is generated from it — so a column cannot be added,
renamed or re-typed in one place and silently drift in another.

Levels
------
Each table is one level of the hierarchy, and each column lives on exactly one
table. The only names that repeat across tables are keys (``id``, ``link``,
``id_part``, ``group_key``) and the three partition columns (``vendor``,
``label``, ``split``), which are copied onto child rows so a clip can be
filtered by them without a join and which the validator checks agree with the
parent. When a higher level summarises a concept that a clip also has, the
higher-level column carries a level prefix (``recording_posture`` vs the clip's
``posture``), so a plain ``merge``/``JOIN USING`` never produces ``_x``/``_y``
suffixes and a conversation-level value can never be mistaken for a per-clip
observation.

Roles
-----
``id`` primary key; ``link`` foreign key to another table's primary key;
``id_part`` a raw segment of a Meta id; ``group_key`` a grouping key with no
table of its own; ``partition`` vendor/label/split; ``metadata`` a given
attribute; ``measure`` a continuous observation; ``label`` a categorical
annotation; ``score`` a 0-1 normalised annotation; ``confidence`` a 0-1 support
value for a label or score; ``status`` why a value is or is not present;
``flag`` a non-null boolean data-quality indicator (never NA: when a condition
cannot be decided, the paired status column says so).

Provenance (where the value comes from)
---------------------------------------
``meta:as_is`` Meta's released value, unchanged. ``meta:derived`` computed by us
from Meta's released metadata or features. ``prior:reused`` a definition from
this repository's previous (filtering) pipeline, reused unchanged and recomputed
over every clip. ``prior:adapted`` a previous definition or threshold, changed
(the note says how and why). ``fresh`` new in this pipeline.

Evidence (how directly it is observed)
--------------------------------------
``given`` supplied by Meta; ``parsed`` decoded from a Meta id or code;
``measured`` computed from released signals (SMPL-H fits, keypoints, VAD, audio,
video) — note SMPL-H and FAU are themselves model estimates; ``rule_inferred`` a
category or flag obtained by applying a stated rule/threshold to measurements;
``aggregate`` a summary of lower-level rows.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

LEVELS: dict[str, str] = {
    "participants": "participant",
    "sessions": "session",
    "prompts": "prompt",
    "interactions": "interaction",
    "interaction_windows": "interaction_window",
    "recordings": "recording",
    "clips": "clip",
    "moi_events": "moi_event",
}
TABLES: tuple[str, ...] = tuple(LEVELS)
PRIMARY_KEYS: dict[str, str] = {
    "participants": "participant_key",
    "sessions": "session_key",
    "prompts": "prompt_hash",
    "interactions": "interaction_key",
    "interaction_windows": "window_id",
    "recordings": "file_id",
    "clips": "clip_id",
    "moi_events": "moi_id",
}
#: link column -> table whose primary key it references
LINK_TARGETS: dict[str, str] = {
    "participant_key": "participants",
    "partner_participant_key": "participants",
    "participant_key_1": "participants",
    "participant_key_2": "participants",
    "suffix_sibling_key": "participants",
    "session_key": "sessions",
    "prompt_hash": "prompts",
    "interaction_key": "interactions",
    "window_id": "interaction_windows",
    "file_id": "recordings",
    "partner_file_id": "recordings",
    "member_file_id_1": "recordings",
    "member_file_id_2": "recordings",
    "clip_id": "clips",
    "partner_clip_id": "clips",
    "member_clip_id_1": "clips",
    "member_clip_id_2": "clips",
    "partner_duplicate_moi_id": "moi_events",
}
PARTITIONS: tuple[str, ...] = ("vendor", "label", "split")
ROLES = ("id", "link", "id_part", "group_key", "partition", "metadata", "measure",
         "label", "score", "confidence", "status", "flag")
PROVENANCE = ("meta:as_is", "meta:derived", "prior:reused", "prior:adapted", "fresh")
EVIDENCE = ("given", "parsed", "measured", "rule_inferred", "aggregate")
DTYPES = ("string", "bool", "int8", "int16", "int32", "int64", "float32", "float64")

VENDORS = ("V00", "V01", "V02", "V03")
LABELS = ("improvised", "naturalistic")
SPLITS = ("train", "dev", "test")
POSTURES = ("standing", "sitting", "mixed", "unclear", "unknown")
YES_NO_UNKNOWN = ("yes", "no", "unknown")
IPC_CODES = ("ANCP", "AMCP", "ANCM", "APCP", "APCN", "AMCN", "APCM", "AMCM", "CGST")
INTERACTION_TYPES = ("ipc_conversation", "grounded_gesture", "collaborative_storytelling", "charades")
RELATIONSHIP_DETAILS = (
    "stranger", "friends", "coworkers", "family-generic", "familiar-generic",
    "dating/spouse/romantic_partner", "classmates", "siblings", "parent_child",
    "neighbors", "roommates",
)
RASTER_CLASSES = (
    "portrait_1080x1920", "portrait_2160x3840", "anamorphic_1012x1920",
    "anamorphic_2160x2160", "anamorphic_1920x1080", "rotated_3840x2160",
    "rotated_640x480", "pillarbox_1080x960", "pillarbox_2180x3840", "no_video", "other",
)
EXPRESSIVITY_GROUPS = ("V00", "V01", "V02", "V03_portrait", "V03_room", "not_applicable")
POSTURE_RULES = ("v03_hip", "v00_knee", "unvalidated_hip_knee", "not_measurable_anamorphic",
                 "not_measured")
POSTURE_EVIDENCE = ("v03_file_labels_164", "v00_file_labels_66_selection_biased", "unlabelled",
                    "not_applicable")


@dataclass(frozen=True)
class Field:
    table: str
    name: str
    dtype: str
    role: str
    description: str
    unit: str = ""
    allowed: tuple = ()
    nullable: bool = False
    missing: str = ""
    provenance: str = "fresh"
    evidence: str = "measured"
    qualified_by: tuple[str, ...] = ()
    source: str = ""
    note: str = ""
    value_range: tuple[float, float] | None = None

    @property
    def level(self) -> str:
        return LEVELS[self.table]

    def check(self) -> list[str]:
        problems = []
        if self.table not in LEVELS:
            problems.append(f"{self.name}: unknown table {self.table}")
        if self.dtype not in DTYPES:
            problems.append(f"{self.table}.{self.name}: dtype {self.dtype}")
        if self.role not in ROLES:
            problems.append(f"{self.table}.{self.name}: role {self.role}")
        if self.provenance not in PROVENANCE:
            problems.append(f"{self.table}.{self.name}: provenance {self.provenance}")
        if self.evidence not in EVIDENCE:
            problems.append(f"{self.table}.{self.name}: evidence {self.evidence}")
        if self.nullable and not self.missing:
            problems.append(f"{self.table}.{self.name}: nullable without a 'missing' meaning")
        if self.role == "flag" and (self.dtype != "bool" or self.nullable):
            problems.append(f"{self.table}.{self.name}: flags must be non-null bool")
        if self.role in ("score", "confidence") and self.value_range is None:
            problems.append(f"{self.table}.{self.name}: {self.role} needs value_range")
        if not self.description:
            problems.append(f"{self.table}.{self.name}: no description")
        return problems


_REGISTRY: list[Field] = []


def _f(table: str, name: str, dtype: str, role: str, description: str, **kw) -> None:
    _REGISTRY.append(Field(table, name, dtype, role, description, **kw))


def _partitions(table: str, *, with_label_split: bool = True) -> None:
    _f(table, "vendor", "string", "partition",
       "Collection vendor (site), from the file id. Rigs, cameras and microphones differ by vendor.",
       allowed=VENDORS, provenance="meta:derived", evidence="parsed")
    if with_label_split:
        _f(table, "label", "string", "partition",
           "Meta's collection category: 'improvised' (prompted scenarios with at least one "
           "professional actor) or 'naturalistic'. Constant within a session.",
           allowed=LABELS, provenance="meta:as_is", evidence="given", source="filelist.csv")
        _f(table, "split", "string", "partition",
           "Meta's official split, as released (filelist.csv), unchanged. Constant within a "
           "session and interaction. Splits are participant-level by design, but 26 "
           "participants appear in two splits: use participants.flag_split_conflict and "
           "interactions.n_members_in_other_split for leakage-free evaluation.",
           allowed=SPLITS, provenance="meta:as_is", evidence="given", source="filelist.csv")


NA_UNMEASURED = "the recording could not be measured (see recordings.measurement_status)"

# =============================================================================
# participants
# =============================================================================
T = "participants"
_f(T, "participant_key", "string", "id",
   "Vendor-qualified participant key, 'V00_P0061'. Bare participant ids collide across "
   "vendors (627 ids), so the vendor is part of the key. May carry Meta's 'A' suffix.",
   provenance="meta:derived", evidence="parsed")
_partitions(T, with_label_split=False)
_f(T, "participant_id", "string", "id_part",
   "Meta's participant id segment as it appears in file ids, zero-padded, including any "
   "'A' suffix. Not unique across vendors.", provenance="meta:as_is", evidence="given")
_f(T, "id_suffix", "string", "metadata",
   "'A' when the participant id carries Meta's letter suffix (165 V00 ids), else 'none'. "
   "The suffix's meaning is undocumented; 182/183 suffixed metadata rows are 'Undisclosed'.",
   allowed=("none", "A"), provenance="meta:derived", evidence="parsed")
_f(T, "suffix_sibling_key", "string", "link",
   "The key of the same numeric id with/without the 'A' suffix in the same vendor, when "
   "both exist (51 V00 stems). Meta warns ids are sometimes duplicated or split across "
   "people; whether siblings are the same person is unconfirmed (they never share a session).",
   nullable=True, missing="no sibling id exists in this vendor",
   provenance="meta:derived", evidence="parsed")
_f(T, "metadata_status", "string", "status",
   "State of this participant's participants.csv row: 'numeric' (BFI-2 scores present), "
   "'undisclosed' (row present, scores 'Undisclosed'), 'missing_row' (no row for this "
   "vendor+id; BFI is never borrowed from another vendor's row).",
   allowed=("numeric", "undisclosed", "missing_row"), provenance="meta:derived",
   evidence="given", source="participants.csv")
for _domain in ("extraversion", "agreeableness", "conscientiousness", "neuroticism", "openness"):
    _f(T, f"bfi_{_domain}", "float32", "metadata",
       f"BFI-2 {_domain} domain score as released (participants.csv {_domain}_raw): the mean of "
       "the domain's items, 1-5.", unit="BFI-2 domain mean (1-5)", nullable=True,
       missing="metadata_status is 'undisclosed' or 'missing_row'",
       provenance="meta:as_is", evidence="given", source="participants.csv",
       value_range=None)
for _split in SPLITS:
    _f(T, f"in_{_split}", "bool", "metadata",
       f"The participant has at least one recording in Meta's official '{_split}' split.",
       provenance="meta:derived", evidence="aggregate", source="filelist.csv")
_f(T, "flag_split_conflict", "bool", "flag",
   "The participant has recordings in more than one official split (26 participants, all "
   "spanning improvised and naturalistic). Evaluation that must be participant-disjoint "
   "should exclude them (and interactions with n_members_in_other_split > 0).",
   provenance="meta:derived", evidence="aggregate", source="filelist.csv")
_f(T, "flag_suffix_sibling_split_conflict", "bool", "flag",
   "A suffix sibling exists (suffix_sibling_key) and its split set differs from this "
   "participant's: potential hidden leakage if the two ids are one person (7 of 51 stems).",
   provenance="fresh", evidence="aggregate")
for _lab in LABELS:
    _f(T, f"in_{_lab}", "bool", "metadata",
       f"The participant has at least one '{_lab}' recording.",
       provenance="meta:derived", evidence="aggregate")
_f(T, "participant_n_sessions", "int32", "measure", "Sessions the participant appears in.",
   evidence="aggregate", provenance="meta:derived")
_f(T, "participant_n_interactions", "int32", "measure", "Interactions the participant appears in.",
   evidence="aggregate", provenance="meta:derived")
_f(T, "participant_n_recordings", "int32", "measure", "Recordings (files) of the participant.",
   evidence="aggregate", provenance="meta:derived")
_f(T, "participant_n_partners", "int32", "measure",
   "Distinct partners the participant recorded with (1,393 participants have more than one).",
   evidence="aggregate", provenance="meta:derived")
_f(T, "participant_recorded_hours", "float32", "measure",
   "Total duration of the participant's measured recordings.", unit="h", evidence="aggregate")
_f(T, "participant_component_id", "string", "group_key",
   "Connected component of the co-participation graph (participants linked when they share "
   "a session). Grouping by it keeps every dyad, and every partner-of-partner, on one side "
   "of a custom split. Components are very unequal (the largest holds about a quarter of "
   "all files), so it suits grouped cross-validation, not balanced splits. Meta's official "
   "split is always the primary split.", provenance="fresh", evidence="aggregate")

# =============================================================================
# sessions
# =============================================================================
T = "sessions"
_f(T, "session_key", "string", "id",
   "Vendor-qualified session key, 'V00_S0039'. Session ids collide across vendors.",
   provenance="meta:derived", evidence="parsed")
_partitions(T)
_f(T, "session_id", "string", "id_part", "Meta's 4-digit session id segment; not unique across vendors.",
   provenance="meta:as_is", evidence="given")
_f(T, "relationship", "string", "metadata",
   "Dyad familiarity for this session, as released (relationships.csv). Every improvised "
   "session with a row is 'stranger'. Session-level, not dyad-level: 9 of 219 recurring "
   "dyads change value between sessions.",
   allowed=("stranger", "familiar"), nullable=True,
   missing="no relationships.csv row for this session (relationship_status == 'missing_row')",
   provenance="meta:as_is", evidence="given", source="relationships.csv")
_f(T, "relationship_detail", "string", "metadata",
   "Finer relationship category as released (friends, coworkers, family-generic, ...).",
   allowed=RELATIONSHIP_DETAILS, nullable=True,
   missing="no relationships.csv row for this session",
   provenance="meta:as_is", evidence="given", source="relationships.csv")
_f(T, "relationship_status", "string", "status",
   "'present' or 'missing_row' (122 sessions have no relationships.csv row: 112 V01, the 9 "
   "single-participant V00 sessions, 1 V03).", allowed=("present", "missing_row"),
   provenance="meta:derived", evidence="given")
_f(T, "participant_key_1", "string", "link",
   "First participant of the session (sorted key order; the order carries no meaning).",
   provenance="meta:derived", evidence="parsed")
_f(T, "participant_key_2", "string", "link",
   "Second participant of the session (sorted key order).", nullable=True,
   missing="single-participant session (9 V00 sessions)", provenance="meta:derived", evidence="parsed")
_f(T, "dyad_key", "string", "group_key",
   "Order-free key of the session's participant pair, 'V00_P0061+V00_P0062'. 219 dyads "
   "recorded more than one session.", provenance="fresh", evidence="parsed")
_f(T, "dyad_session_count", "int32", "measure", "Sessions this dyad recorded together.",
   provenance="fresh", evidence="aggregate")
_f(T, "session_n_participants", "int8", "measure", "Participants in the session (1 or 2).",
   provenance="meta:derived", evidence="aggregate")
_f(T, "session_n_interactions", "int32", "measure", "Interactions recorded in the session (1-20).",
   provenance="meta:derived", evidence="aggregate")
_f(T, "session_n_interaction_types", "int8", "measure",
   "Distinct Meta interaction_type values among the session's prompts.",
   provenance="meta:derived", evidence="aggregate")
_f(T, "session_duration_s", "float32", "measure",
   "Sum of the session's interaction durations (active time only; Meta's 'meta time' between "
   "interactions is not released).", unit="s", evidence="aggregate", nullable=True,
   missing="no interaction of the session could be measured")
_f(T, "session_posture_consistent", "string", "label",
   "'yes' when every participant's decided recording-level postures (standing/sitting) are "
   "the same across all of their recordings in this session, 'no' when a participant is "
   "standing in one interaction and sitting in another (or any recording is 'mixed'), "
   "'unknown' when a participant has fewer than two decided recordings.",
   allowed=YES_NO_UNKNOWN, evidence="aggregate")

# =============================================================================
# prompts
# =============================================================================
T = "prompts"
_f(T, "prompt_hash", "string", "id",
   "Meta's prompt id (interactions.csv prompt_hash, 8 digits). It equals the I-segment of "
   "file ids, which Meta calls interaction_id, but it identifies a PROMPT reused across up "
   "to 1,550 sessions — never group conversations by it; use interaction_key.",
   provenance="meta:as_is", evidence="given", source="interactions.csv")
_f(T, "prompt_id_unique", "string", "metadata", "Meta's structured prompt identifier, as released.",
   provenance="meta:as_is", evidence="given", source="interactions.csv")
_f(T, "prompt_template", "string", "metadata",
   "Template id parsed from prompt_id_unique ('P0070'); groups prompt variants.",
   nullable=True, missing="prompt_id_unique has no P#### template (the RPx.y family)",
   provenance="meta:derived", evidence="parsed")
_f(T, "prompt_version", "string", "metadata", "Prompt-set version parsed from prompt_id_unique ('v3.4').",
   nullable=True, missing="no version token", provenance="meta:derived", evidence="parsed")
_f(T, "prompt_context", "string", "metadata",
   "Context tag parsed from prompt_id_unique: 'fam'/'str' (written for familiar/stranger "
   "dyads), 'RP' (role-play), 'AC'/'aac' (acting games), other two-letter set codes as "
   "released, 'other' when none parses.",
   allowed=("fam", "str", "RP", "AC", "aac", "AA", "CC", "DN", "DP", "EO", "MN", "MP", "other"),
   provenance="meta:derived", evidence="parsed")
_f(T, "prompt_item_index", "int16", "metadata",
   "The '-n' item suffix of grounded-gesture prompts (0-279).", nullable=True,
   missing="prompt has no item suffix", provenance="meta:derived", evidence="parsed")
_f(T, "participant_a_prompt_text", "string", "metadata",
   "Prompt text given to prompt role A, as released. Meta does not say which participant "
   "was role A.", provenance="meta:as_is", evidence="given", source="interactions.csv")
_f(T, "participant_b_prompt_text", "string", "metadata",
   "Prompt text given to prompt role B, as released.", nullable=True,
   missing="empty in interactions.csv", provenance="meta:as_is", evidence="given",
   source="interactions.csv")
_f(T, "ab_prompts_differ", "bool", "metadata",
   "The A and B prompt texts differ, so the two members had different instructions "
   "(true for ~91% of files).", provenance="meta:derived", evidence="parsed")
_f(T, "ipc_member_assignable", "bool", "metadata",
   "True only when both roles received the identical text, so ipc_a describes both members. "
   "Otherwise Meta provides no participant-to-role mapping and a per-member IPC cannot be "
   "assigned (deliberately not guessed).", provenance="fresh", evidence="parsed")
_f(T, "interaction_type", "string", "metadata",
   "Meta's interaction type for the prompt, as released. For 9 prompts (8,007 files) the "
   "prompt text contradicts it (see interaction_type_text_consistency); speech behaviour where "
   "checked follows the type, not the text, consistent with Meta's off-by-one prompt caveat.",
   allowed=INTERACTION_TYPES, provenance="meta:as_is", evidence="given", source="interactions.csv")
_f(T, "task_kind_from_text", "string", "label",
   "Task kind inferred from the A prompt text by keyword rules (see docs): discussion, "
   "storytelling, game (incl. charades-style acting), gesture_sentence, role_play, unclear. "
   "A topic handle, not ground truth.",
   allowed=("discussion", "storytelling", "game", "gesture_sentence", "role_play", "unclear"),
   evidence="rule_inferred")
_f(T, "interaction_type_text_consistency", "string", "label",
   "Whether interaction_type agrees with task_kind_from_text: 'consistent', "
   "'contradicts_text' (a proxy for Meta's prompt/text misalignment caveat), or 'unclear'.",
   allowed=("consistent", "contradicts_text", "unclear"), evidence="rule_inferred")
for _r in ("a", "b"):
    _f(T, f"ipc_{_r}", "string", "metadata",
       f"Interpersonal-circumplex code of prompt role {_r.upper()}, as released: "
       "A{P,N,M}C{P,N,M} = agency plus/neutral/minus x communion plus/neutral/minus (8 "
       "octants), or CGST (not an octant; personas suggest a cognitive state). An attribute "
       "of the prompt role, not of a participant." + (" ipc_a is never empty." if _r == "a" else ""),
       allowed=IPC_CODES, nullable=(_r == "b"),
       missing="no second IPC code given (single-sided prompt)" if _r == "b" else "",
       provenance="meta:as_is", evidence="given", source="interactions.csv")
    for _axis in ("agency", "communion"):
        _f(T, f"ipc_{_r}_{_axis}", "int8", "metadata",
           f"{_axis.capitalize()} axis of ipc_{_r} decoded to +1/0/-1 (P/N/M).",
           allowed=(-1, 0, 1), nullable=True, missing=f"ipc_{_r} is CGST or absent",
           provenance="meta:derived", evidence="parsed")
    _f(T, f"ipc_{_r}_kind", "string", "metadata",
       f"'octant', 'cognitive_state' (CGST) or 'absent' for ipc_{_r}.",
       allowed=("octant", "cognitive_state", "absent"), provenance="meta:derived", evidence="parsed")
_f(T, "prompt_n_sessions", "int32", "measure",
   "Sessions that used this prompt (median 26, max 1,550). High values mean the same prompt "
   "text recurs across train/dev/test (405 of 407 test prompts also occur in train).",
   provenance="meta:derived", evidence="aggregate")
_f(T, "prompt_n_vendors", "int8", "measure", "Vendors that used this prompt.",
   provenance="meta:derived", evidence="aggregate")

# =============================================================================
# interactions (conversation level)
# =============================================================================
T = "interactions"
_f(T, "interaction_key", "string", "id",
   "Conversation key, 'V00_S0039_I00000581' = (vendor, session, prompt). The unique id of one "
   "interaction instance; the bare I-segment is a prompt id.", provenance="meta:derived", evidence="parsed")
_f(T, "session_key", "string", "link", "Session of this interaction.", provenance="meta:derived", evidence="parsed")
_f(T, "prompt_hash", "string", "link", "Prompt of this interaction (join prompts for type, IPC, text).",
   provenance="meta:derived", evidence="parsed")
_partitions(T)
_f(T, "interaction_n_recordings", "int8", "measure", "Recordings (participants) released for this interaction: 2, or 1 for the 132 partner-missing interactions.",
   provenance="meta:derived", evidence="aggregate")
_f(T, "pairing_status", "string", "status",
   "'paired' (both participants' recordings released) or 'partner_missing' (132 V00 "
   "interactions in 9 single-participant sessions).", allowed=("paired", "partner_missing"),
   provenance="meta:derived", evidence="aggregate")
_f(T, "member_file_id_1", "string", "link",
   "Recording of the first member (members sorted by participant_id; the order carries no "
   "prompt-role meaning).", provenance="meta:derived", evidence="parsed")
_f(T, "member_file_id_2", "string", "link", "Recording of the second member.", nullable=True,
   missing="partner_missing", provenance="meta:derived", evidence="parsed")
_f(T, "interaction_n_members_measured", "int8", "measure",
   "Members whose recording could be measured (0, 1 or 2). Pair aggregates are NA unless both are.",
   evidence="aggregate")
_f(T, "interaction_duration_s", "float32", "measure", "Longer of the members' recording durations.",
   unit="s", nullable=True, missing="no member measured", evidence="aggregate")
_f(T, "pair_duration_diff_s", "float32", "measure", "Absolute difference of the members' durations.",
   unit="s", nullable=True, missing="fewer than two measured members", evidence="aggregate")
_f(T, "pair_fps_differs", "bool", "flag",
   "The members' frame rates differ (1,147 pairs, all V01); clips are aligned by time, so this "
   "matters only for frame-index arithmetic across partners (1,120 interactions). False when "
   "fewer than two members are measured.",
   evidence="aggregate")
_f(T, "pair_duration_agreement", "string", "status",
   "Whether the members' recordings agree in length: 'agree' (<= 0.1 s), 'minor_mismatch' "
   "(<= 1 s), 'major_mismatch' (> 1 s, or either member has timebase drift), "
   "'not_applicable' (fewer than two measured members). A proxy for Meta's ~10% timestamp "
   "misalignment caveat, NOT Meta's list (none is published): 'agree' does not mean "
   "unaffected. partner_clip_id is only set when this is 'agree' or 'minor_mismatch'.",
   allowed=("agree", "minor_mismatch", "major_mismatch", "not_applicable"), evidence="rule_inferred")
_f(T, "speech_status", "string", "status",
   "Speech annotation state of the pair: 'both_annotated' (both members have VAD or "
   "transcript speech), 'one_unannotated', 'both_unannotated' (both have empty VAD and "
   "transcript; silence and missing annotation are indistinguishable), 'partner_missing', "
   "'not_measured'. Pair speech fields are NA unless 'both_annotated' and durations align.",
   allowed=("both_annotated", "one_unannotated", "both_unannotated", "partner_missing", "not_measured"),
   evidence="aggregate")
for _name, _desc, _unit in (
    ("interaction_speech_overlap_frac", "Share of time either member speaks during which both speak. High values can be real overlap or speaker bleed (see recordings.recording_transcript_echo_frac).", ""),
    ("interaction_mutual_silence_frac", "Share of the shared interval in which neither member speaks.", ""),
    ("interaction_speech_balance", "Balance of speaking time, min/max of the members' own-speech seconds (1 = equal, 0 = one-sided).", ""),
    ("interaction_turn_switch_rate_per_min", "Changes of floor holder (the sole speaker at 10 Hz) per minute, counted within dyad windows (a change across a window boundary is not counted).", "1/min"),
):
    _f(T, _name, "float32", "measure", _desc, unit=_unit, nullable=True,
       missing="no dyad window with both members' speech annotated (speech_status != 'both_annotated', pair_duration_agreement 'major_mismatch'/'not_applicable', or every window has an unannotated member clip), or (balance/overlap) neither member speaks",
       evidence="aggregate", source="interaction_windows where both member clips have speech_annotation_status 'annotated'")
_f(T, "interaction_pair_speech_coverage_frac", "float32", "measure",
   "Share of interaction_duration_s covered by the dyad windows the pair speech measures "
   "are computed over (both members' speech annotated): 1 for most conversations; lower "
   "where a member's VAD drops out or its WAV is short.", value_range=(0.0, 1.0), nullable=True,
   missing="as interaction_speech_overlap_frac", evidence="aggregate")
_f(T, "interaction_n_clips", "int32", "measure", "Clips over all members of the interaction.",
   evidence="aggregate")
_f(T, "posture_pair", "string", "label",
   "The members' recording-level posture labels, sorted and joined with '+' "
   "('standing+standing', 'sitting+standing', 'mixed+standing', ...); 'absent' stands in for "
   "a missing partner. A dyad configuration for conditioning; per-window pairs are in "
   "interaction_windows.window_posture_pair.", evidence="aggregate")
_f(T, "any_member_posture_transition", "string", "label",
   "'yes' if either member's recording has a sustained standing<->sitting change, 'no' if "
   "neither does and both are decided, else 'unknown'.", allowed=YES_NO_UNKNOWN, evidence="aggregate")
_f(T, "interaction_expressivity_mean", "float32", "score",
   "Mean clip expressivity_score over both members' measured clips (duration-weighted). A "
   "mean of percentiles concentrates near 0.5; compare conversations by rank, not against "
   "the clip-level 1/3 and 2/3 cuts.", value_range=(0.0, 1.0), nullable=True,
   missing="fewer than two members with measured expressivity", evidence="aggregate")
_f(T, "interaction_expressivity_gap", "float32", "measure",
   "Absolute difference of the two members' mean clip expressivity_score (0-1 scale).",
   nullable=True, missing="fewer than two members with measured expressivity", evidence="aggregate")
for _p in ("3p", "1p"):
    _f(T, f"moi_{_p}_coverage", "string", "status",
       f"How many members have Meta {_p.upper()} MOI annotations: 'both', 'one', 'none' "
       f"(coverage differs within pairs: {'487' if _p == '3p' else '419'} pairs have one member only).",
       allowed=("both", "one", "none"), provenance="meta:derived", evidence="given",
       source="filelist.csv has_annotation_*")
    _f(T, f"moi_duplication_{_p}", "string", "label",
       f"Whether {_p.upper()} MOI events are copied between the members (same kind, same "
       "normalised text, start within 2 s): 'none', 'partial', 'full', 'not_applicable' "
       "(coverage != both) or 'not_measured' (both annotated per Meta's flags but a member's "
       "files could not be read). Mostly V03 (session S0203), plus 3 V00 pairs; a duplicated "
       "event's target person is ambiguous.",
       allowed=("none", "partial", "full", "not_applicable", "not_measured"), evidence="rule_inferred")
_f(T, "interaction_moi_3p_count", "int32", "measure",
   "Sum of the two members' distinct 3P MOI counts (a moment copied to both members, see "
   "moi_duplication_3p, counts once per member).", nullable=True,
   missing="moi_3p_coverage != 'both', or a member's recording could not be measured",
   provenance="meta:derived", evidence="aggregate")
_f(T, "n_members_in_other_split", "int8", "measure",
   "Members whose participant also has recordings in an official split other than this "
   "interaction's (0, 1 or 2). For participant-disjoint evaluation keep split in {dev, test} "
   "AND this == 0: 25% of dev and 13% of test interactions have a member with train files.",
   provenance="fresh", evidence="aggregate")
_f(T, "n_members_suffix_sibling_other_split", "int8", "measure",
   "Members whose A-suffix sibling id has recordings in another split (unconfirmed identity; "
   "a stricter leakage filter).", provenance="fresh", evidence="aggregate")

# =============================================================================
# interaction_windows (the dyad over one clip interval)
# =============================================================================
T = "interaction_windows"
_f(T, "window_id", "string", "id",
   "Dyad window key, '<interaction_key>_L30_W003': the interval [k*L, (k+1)*L) of the "
   "interaction, shared by the two members' clip k.", evidence="parsed")
_f(T, "interaction_key", "string", "link", "Interaction of this window.", evidence="parsed")
_partitions(T)
_f(T, "window_index", "int16", "id_part", "Index k of the window (= clip_index of its member clips).", evidence="parsed")
_f(T, "member_clip_id_1", "string", "link", "Clip k of member 1 (interactions.member_file_id_1); see member_clip_id_2.",
   nullable=True, missing="member 1 has no clip k (shorter recording or unmeasured)", evidence="parsed")
_f(T, "member_clip_id_2", "string", "link",
   "Clip k of member 2. Listed whatever the pair's state; when pair_duration_agreement is "
   "'major_mismatch' the two clips share an index but not necessarily a wall-clock interval "
   "(their partner_clip_id is NA and the window's pair measures are NA).", nullable=True,
   missing="partner missing, or member 2 has no clip k (shorter or unmeasured recording)", evidence="parsed")
_f(T, "window_start_s", "float32", "measure", "Window start (k*L).", unit="s", evidence="measured")
_f(T, "window_end_s", "float32", "measure", "Window end: min of the member clips' ends.", unit="s", evidence="measured")
for _name, _desc in (
    ("window_speech_overlap_frac", "Share of time either member speaks during which both speak."),
    ("window_mutual_silence_frac", "Share of the window in which neither member speaks."),
):
    _f(T, _name, "float32", "measure", _desc, nullable=True,
       missing="both member clips needed, with speech annotated for both and aligned durations; also NA when the window is shorter than one 0.1-s tick (1-frame tail clips), and overlap when neither speaks",
       evidence="measured")
_f(T, "window_turn_switches", "int16", "measure", "Changes of floor holder (sole speaker, 10 Hz) in the window (see window_mutual_silence_frac for NA).",
   nullable=True, missing="as window_mutual_silence_frac", evidence="measured")
_f(T, "window_posture_pair", "string", "label",
   "The two member clips' posture labels, sorted and '+'-joined ('standing+standing', ...); "
   "'absent' for a missing member clip.", evidence="aggregate")

# =============================================================================
# recordings (participant-in-interaction level; Meta's per-file unit)
# =============================================================================
T = "recordings"
_f(T, "file_id", "string", "id",
   "Meta's file id, 'V00_S0039_I00000581_P0061': one participant's files for one interaction "
   "(what the task brief called a clip).", provenance="meta:as_is", evidence="given", source="filelist.csv")
_f(T, "interaction_key", "string", "link", "Conversation this recording belongs to.", provenance="meta:derived", evidence="parsed")
_f(T, "session_key", "string", "link", "Session of this recording.", provenance="meta:derived", evidence="parsed")
_f(T, "participant_key", "string", "link", "The recorded participant (the single tracked subject of every body/face field).",
   provenance="meta:derived", evidence="parsed")
_f(T, "prompt_hash", "string", "link", "Prompt id (the file id's I-segment; Meta's 'interaction_id').",
   provenance="meta:derived", evidence="parsed")
_partitions(T)
_f(T, "session_id", "string", "id_part", "Session id segment (not unique across vendors).", provenance="meta:as_is", evidence="given")
_f(T, "participant_id", "string", "id_part", "Participant id segment (not unique across vendors).", provenance="meta:as_is", evidence="given")
_f(T, "batch_idx", "int16", "metadata", "Release batch index.", provenance="meta:as_is", evidence="given", source="filelist.csv")
_f(T, "archive_idx", "int16", "metadata", "Release archive index. Partners are almost never in the same archive.",
   provenance="meta:as_is", evidence="given", source="filelist.csv")
_f(T, "source_relbase", "string", "metadata",
   "Path of the recording's files relative to the release root, without suffix "
   "('improvised/train/0000/0000/<file_id>'); append .npz/.json/.wav/.mp4.",
   provenance="prior:reused", evidence="parsed", source="M-1 inventory")
_f(T, "partner_file_id", "string", "link", "The other member's recording in this interaction.",
   nullable=True, missing="partner_missing", provenance="meta:derived", evidence="parsed")
_f(T, "partner_participant_key", "string", "link", "The other member.", nullable=True,
   missing="partner_missing", provenance="meta:derived", evidence="parsed")
for _flag, _desc in (
    ("has_imitator_movement", "Meta's flag: the NPZ carries movement:* (Imitator face/emotion) features. Exactly the 42,932 V00 files."),
    ("has_annotation_1p", "Meta's flag: 1P (self-report) MOI annotations exist (491 files, all V03)."),
    ("has_annotation_3p", "Meta's flag: 3P (observer) MOI annotations exist (1,663 files, V00 and V03)."),
):
    _f(T, _flag, "bool", "metadata", _desc, provenance="meta:as_is", evidence="given", source="filelist.csv")
for _mod in ("json", "mp4", "npz", "wav"):
    _f(T, f"{_mod}_present", "bool", "metadata", f"The .{_mod} file exists in the release.",
       provenance="prior:reused", evidence="measured", source="M-1 inventory (stat)")
_f(T, "probe_status", "string", "status", "ffprobe outcome for the MP4 (M-1 census).",
   allowed=("ok", "no_video_stream", "missing_mp4", "ffprobe_error"),
   provenance="prior:reused", evidence="measured", source="M-1 inventory (ffprobe)")
_f(T, "video_width", "int16", "metadata", "Stored video width.", unit="px", nullable=True,
   missing="no video stream", provenance="prior:reused", evidence="measured", source="M-1 inventory")
_f(T, "video_height", "int16", "metadata", "Stored video height.", unit="px", nullable=True,
   missing="no video stream", provenance="prior:reused", evidence="measured", source="M-1 inventory")
_f(T, "fps", "float64", "metadata",
   "Container frame rate (r_frame_rate): 30, 30000/1001, 45000/1501, or odd V01 rates. The "
   "released pose/keypoint arrays follow this grid.", unit="1/s", nullable=True,
   missing="no video stream (such files have zero-frame NPZs)", provenance="prior:reused",
   evidence="measured", source="M-1 inventory")
_f(T, "video_nb_frames", "int32", "metadata", "Frames in the video stream.", nullable=True,
   missing="no video stream", provenance="prior:reused", evidence="measured")
_f(T, "video_duration_s", "float32", "metadata", "Video stream duration.", unit="s", nullable=True,
   missing="no video stream", provenance="prior:reused", evidence="measured")
_f(T, "video_bitrate_mbps", "float32", "measure", "MP4 size / duration: a compression-quality proxy.",
   unit="Mbit/s", nullable=True, missing="no video stream", provenance="fresh", evidence="measured",
   source="M-1 inventory")
_f(T, "raster_class", "string", "label",
   "Stored raster geometry class. anamorphic_*: V01 stored with non-square pixels (2160x2160 "
   "is 9:16, 1920x1080 is 81:256, 1012x1920 is 270:253); the released SMPL-H absorbed the "
   "stretch. rotated_*: V03 stored a quarter turn from upright. pillarbox_*: black side bars.",
   allowed=RASTER_CLASSES, provenance="prior:adapted", evidence="rule_inferred",
   note="Adapted from corpus.EXCLUDED_RASTERS (which dropped these files) into a class label; "
   "SAR per raster verified constant by ffprobe on 4 files per vendor x raster cell.")
_f(T, "smplh_anamorphic", "string", "label",
   "Pose distortion from anamorphic storage: 'none', 'mild' (1012x1920, 6.7%), 'severe' "
   "(2160x2160, 1920x1080: posture and expressivity are not measured).",
   allowed=("none", "mild", "severe"), provenance="prior:adapted", evidence="rule_inferred")
_f(T, "room_camera_rig", "bool", "flag",
   "V03 3840x2160/640x480 room-camera rig: far-field audio, i.e. strong speaker bleed "
   "(voice_isolation_db median 8.8 dB against 16-23 dB on close-worn microphones; transcript echo "
   "0.1-0.35).",
   provenance="prior:adapted", evidence="rule_inferred",
   note="Previously a reason to exclude (corpus.EXCLUDED_RASTERS); now an annotation.")
_f(T, "quarter_turns", "int8", "measure",
   "Counter-clockwise quarter turns that stand the stored picture upright, detected from the "
   "released 2D shoulder-hip axis (media_repair). Detected per file: 9 of the 1,405 3840x2160 "
   "files are already upright.", nullable=True, missing=NA_UNMEASURED,
   provenance="prior:reused", evidence="measured")
_f(T, "measurement_status", "string", "status",
   "'measured', or why not: 'missing_files' (NPZ or JSON absent), 'unreadable', 'no_frames' "
   "(zero-frame NPZ: 476 files, the 439 with no video stream plus 37 V01 2160x2160 files "
   "with video). Unmeasured recordings have no clips.",
   allowed=("measured", "missing_files", "unreadable", "no_frames"), evidence="measured")
_f(T, "n_frames", "int32", "measure",
   "Frames in the released pose arrays (the clip grid); 0 for 'no_frames' recordings.",
   nullable=True, missing="NPZ/JSON absent or unreadable", evidence="measured")
_f(T, "recording_duration_s", "float32", "measure", "n_frames / fps.", unit="s", nullable=True,
   missing=NA_UNMEASURED, evidence="measured")
_f(T, "recording_n_clips", "int32", "measure", "Clips cut from the recording (0 when unmeasured).", evidence="measured")
_f(T, "npz_video_frame_diff", "int32", "measure", "n_frames - video_nb_frames (0 for 86%, within +-1 for 99.8%).",
   nullable=True, missing="unmeasured or no video frame count", evidence="measured")
_f(T, "timebase_drift_s", "float32", "measure",
   "|n_frames/fps - video_duration_s|: how far the released annotation grid runs from the video.",
   unit="s", nullable=True, missing="unmeasured or no video duration", evidence="measured")
_f(T, "timebase_status", "string", "status",
   "'consistent' (drift <= 0.5 s), 'drift' (> 0.5 s: frame-index lookups into the video "
   "misalign; two hand-checked cases had unrelated tracking), 'unknown'.",
   allowed=("consistent", "drift", "unknown"), provenance="prior:adapted", evidence="rule_inferred",
   note="Was the population exclusion 'timebase_drift'; now a status.")
for _name, _desc in (
    ("recording_smplh_valid_frac", "Share of frames with smplh:is_valid. Effectively a hand-pose validity flag (hands freeze on invalid frames; body fit is unaffected)."),
    ("recording_subject_present_frac", "Share of frames with a valid tracking box (boxes_and_keypoints:is_valid_box). On invalid frames keypoints are zero-filled and translation holds a sentinel."),
    ("recording_hand_frozen_frac", "Share of frames whose 90-d hand pose is bit-identical to the previous frame."),
    ("recording_wrists_in_frame_frac", "Share of frames with both wrist keypoints confident (>= 0.5) and inside the raster."),
):
    _f(T, _name, "float32", "measure", _desc, nullable=True, missing=NA_UNMEASURED,
       provenance="prior:reused" if "wrists" not in _name else "fresh", value_range=(0.0, 1.0))
_f(T, "recording_reprojection_error_p50_sw", "float32", "measure",
   "Median distance between projected SMPL-H upper-body joints and the released 2D keypoints, "
   "in 2D shoulder widths (recording medians: portrait 1080x1920 0.094, V03 2160x3840 0.137 — "
   "partly a camera offset in the released fit rather than bad joints — anamorphic V01 "
   "0.11-0.13; compare within raster_class). A tracking-quality signal independent of motion amount.", unit="shoulder widths", nullable=True,
   missing=NA_UNMEASURED + " or no confident keypoints", provenance="prior:adapted",
   note="v0 measured this residual per file (M-4); recomputed with the numpy FK and the HMR2 camera.")
_f(T, "speech_source", "string", "status",
   "Where the own-speech mask comes from: 'vad' (Meta VAD); 'vad+transcript_tail' (the VAD "
   "stops early — >= 20 timed words start > 5 s after its last interval — so the words' spans "
   "are added after it; ~1.8% of V03); 'transcript_only' (VAD empty but the transcript has "
   "timed words: their spans, gaps <= 0.5 s bridged); 'none' (both empty: silence and a "
   "missing annotation are indistinguishable, so speech values are NA).",
   allowed=("vad", "vad+transcript_tail", "transcript_only", "none"), evidence="rule_inferred",
   nullable=True, missing=NA_UNMEASURED)
_f(T, "speech_annotated_until_s", "float32", "measure",
   "End of the span over which own speech is annotated: the WAV's duration (or the last "
   "speech/word end if later), capped at the recording duration. Clips reaching > 1 s past it "
   "(57 measured recordings have a WAV > 1 s shorter than their pose grid) have NA speech values.",
   unit="s", nullable=True, missing=NA_UNMEASURED, evidence="measured")
_f(T, "recording_own_speech_s", "float32", "measure", "Own speech time (speech_source mask).",
   unit="s", nullable=True, missing=NA_UNMEASURED + ", or speech_source is 'none'", provenance="meta:derived")
_f(T, "recording_own_speech_frac", "float32", "measure",
   "Own speech share of the span speech is annotated for (recording_own_speech_s / "
   "speech_annotated_until_s). Mid-recording VAD gaps (clips.speech_annotation_status "
   "'vad_gap') make it an undercount.",
   nullable=True, missing=NA_UNMEASURED + ", or speech_source is 'none'", provenance="meta:derived",
   value_range=(0.0, 1.0))
_f(T, "recording_word_count", "int32", "measure",
   "Timed transcript words whose midpoint lies on the pose grid (clip word_counts sum to it).", nullable=True,
   missing=NA_UNMEASURED, provenance="meta:derived")
_f(T, "recording_transcript_echo_frac", "float32", "measure",
   "Share of own transcript words (5+ letters) that also occur in the partner's transcript "
   "within 0.3 s: a speaker-bleed indicator (typical 0-0.04; cross-talk pairs 0.15-0.49).",
   nullable=True, missing="partner missing/unmeasured, partner has no speech annotation, or fewer than 20 own words",
   evidence="measured", source="both members' transcripts")
_f(T, "audio_status", "string", "status",
   "'ok', 'empty' (58-byte or zero-sample WAV), 'unreadable', 'missing'.",
   allowed=("ok", "empty", "unreadable", "missing"), nullable=True, missing=NA_UNMEASURED)
_f(T, "audio_duration_s", "float32", "measure", "WAV duration.", unit="s", nullable=True,
   missing="audio_status != 'ok'")
_f(T, "audio_envelope_dynamics_db", "float32", "measure",
   "p95 minus median of the 50 ms RMS level over the file. Below 1.5 dB the track carries no "
   "usable signal (v0 FM4: 13/13 dead files caught, 0 false positives on 48 labels).",
   unit="dB", nullable=True, missing="audio_status != 'ok'", provenance="prior:adapted")
_f(T, "audio_exact_zero_frac", "float32", "measure",
   "Share of samples exactly zero. NOT a dropout indicator on its own: quiet stretches of the "
   "denoised 16-bit audio round to 0, so healthy tracks read 0.2-32% (see audio_floor_tick_frac).",
   nullable=True, missing="audio_status != 'ok'", value_range=(0.0, 1.0))
_f(T, "audio_floor_tick_frac", "float32", "measure",
   "Share of 50-ms ticks below -88 dBFS, the 16-bit quantisation floor (digital silence). "
   "Mostly reflects listening time: the released audio is denoised and bleed-suppressed, so a "
   "track sits at the floor while its participant is silent (0-0.7 on healthy files, rising as "
   "own speech falls). Not a dropout detector on its own.",
   nullable=True, missing="audio_status != 'ok'", value_range=(0.0, 1.0))
_f(T, "audio_quality", "string", "label",
   "'unusable' (audio_status != ok), 'dead' (envelope dynamics < 1.5 dB: v0 FM4, 13/13 on 48 "
   "labels), 'silent_during_own_speech' (the median level of the participant's own-speech "
   "ticks is at the digital floor, < -80 dBFS: the VAD says they speak but the track is "
   "silent — a dropout or an audio/annotation misalignment), else 'ok'; first match wins. "
   "Partial dropouts are NOT detected: the floor-tick share tracks listening time in the "
   "denoised audio and does not separate the one known dropout pair from healthy files.",
   allowed=("ok", "dead", "silent_during_own_speech", "unusable"), nullable=True, missing=NA_UNMEASURED,
   provenance="prior:adapted", evidence="rule_inferred")
_f(T, "recording_own_speech_level_db", "float32", "measure",
   "Median 50 ms RMS level over own-speech frames (dBFS). Vendor-dependent by ~20 dB (mic gain).",
   unit="dBFS", nullable=True, missing="audio not ok or < 1 s own speech")
_f(T, "recording_partner_only_level_db", "float32", "measure",
   "Equivalent (energy-mean) level of this track while only the partner speaks: what bleeds "
   "in. Energy mean, not median: the released audio is bleed-suppressed, so the median "
   "partner-only tick is just the noise floor.", unit="dBFS",
   nullable=True, missing="audio not ok, partner missing, or < 5 s partner-only speech")
_f(T, "voice_isolation_db", "float32", "measure",
   "Equivalent level over own-only ticks minus equivalent level over partner-only ticks: how "
   "much louder the participant is on their own track than the bleeding partner (corpus "
   "medians: V00 19.2 dB, V01 19.4, V02 23.1, V03 portrait 16.2, V03 room camera 8.8). Meta's "
   "speaker-bleed caveat, measured; not a per-file verdict (v0 showed its variant does not "
   "separate usable from unusable files). v0's 7-13 dB / 2 dB figures pooled overlap ticks "
   "and are not comparable.", unit="dB", nullable=True,
   missing="audio not ok, partner missing, or < 5 s of own-only or partner-only speech",
   provenance="prior:adapted", source="own WAV + both VADs")
_f(T, "recording_posture", "string", "label",
   "Posture of the participant over the whole recording, from all its 1-s bins (same rules "
   "as clips.posture). 'standing' means upright with legs extended (a person perched on a "
   "high stool with legs extended reads as standing).", allowed=POSTURES, evidence="rule_inferred",
   provenance="prior:adapted")
_f(T, "recording_posture_confidence", "float32", "confidence",
   "Share of the recording's 1-s bins whose measurement supports recording_posture (for "
   "'mixed': bins decided either way). An evidence-support score, not P(correct).",
   value_range=(0.0, 1.0), nullable=True, missing="recording_posture is 'unclear' or 'unknown'",
   evidence="aggregate")
for _st in ("standing", "sitting", "unclear", "unobserved"):
    _f(T, f"recording_posture_{_st}_frac", "float32", "measure",
       f"Share of the recording's 1-s bins classified '{_st}' (the four shares sum to 1).",
       value_range=(0.0, 1.0), nullable=True, missing=NA_UNMEASURED, evidence="aggregate")
_f(T, "recording_posture_transitions", "int16", "measure",
   "Standing<->sitting changes between sustained runs (>= 5 decided 1-s bins) over the recording.",
   nullable=True, missing=NA_UNMEASURED, evidence="aggregate")
_f(T, "posture_rule", "string", "status",
   "Which posture rule applies (by vendor/raster): 'v03_hip' (hip flexion), 'v00_knee' "
   "(knee position along the leg), 'unvalidated_hip_knee' (V01 square-pixel and mild "
   "1012x1920, V02: decides only 'standing' or 'unclear', never 'sitting' — 24 of 24 "
   "visually checked V02 'sitting' clips were standing), 'not_measurable_anamorphic' "
   "(V01 2160x2160, 1920x1080), 'not_measured'.",
   allowed=POSTURE_RULES, provenance="prior:adapted", evidence="rule_inferred")
_f(T, "posture_rule_evidence", "string", "status",
   "What the rule's thresholds rest on: 'v03_file_labels_164' (file-level hand labels, "
   "aggregates only), 'v00_file_labels_66_selection_biased', 'unlabelled', 'not_applicable'. "
   "Bin- and clip-level accuracy was never labelled.", allowed=POSTURE_EVIDENCE,
   provenance="prior:adapted", evidence="given")
_f(T, "recording_hip_flexion_deg_p50", "float32", "measure",
   "Median torso-thigh angle over frames with both hips and knees visible (180 = straight; "
   "seated ~90-125). Rig-dependent: standing V00 reads ~25 deg lower than standing V03.",
   unit="deg", nullable=True, missing="unmeasured, severe anamorphic (2160x2160, 1920x1080), or < 50% hip-observable frames",
   provenance="prior:adapted")
_f(T, "recording_knee_between_p50", "float32", "measure",
   "Median knee position along the hip->ankle drop, measured on the torso axis (standing "
   "~0.5, seated 0.15-0.4), over frames with knees and ankles visible.", nullable=True,
   missing="unmeasured, severe anamorphic, or < 50% ankle-observable frames", provenance="prior:adapted")
_f(T, "recording_lower_body_observed_frac", "float32", "measure",
   "Share of frames with both hips and both knees visible (confident and inside the raster).",
   value_range=(0.0, 1.0), nullable=True, missing=NA_UNMEASURED)
_f(T, "expressivity_reference_group", "string", "status",
   "Reference population the recording's clips are ranked against (vendor, V03 split into "
   "portrait and room-camera rigs); 'not_applicable' for severe-anamorphic or unmeasured.",
   allowed=EXPRESSIVITY_GROUPS, evidence="rule_inferred")
_f(T, "recording_expressivity_mean", "float32", "score",
   "Duration-weighted mean of the recording's measured clip expressivity_score.",
   value_range=(0.0, 1.0), nullable=True, missing="no measured clip", evidence="aggregate")
_f(T, "recording_expressivity_clip_sd", "float32", "measure",
   "SD of clip expressivity_score within the recording (median ~0.19: most recordings span levels).",
   nullable=True, missing="fewer than 2 measured clips", evidence="aggregate")
_f(T, "recording_expressivity_trend", "float32", "measure",
   "Spearman correlation of clip expressivity_score with clip index (+ = rises over the "
   "interaction).", nullable=True, missing="fewer than 3 measured clips", evidence="aggregate")
_f(T, "face_features_status", "string", "status",
   "'available' (movement:* present with valid frames), 'not_provided' (Meta released no "
   "Imitator features: all non-V00 files), 'all_invalid', 'not_measured'.",
   allowed=("available", "not_provided", "all_invalid", "not_measured"), provenance="meta:derived",
   evidence="given")
_f(T, "recording_face_valid_frac", "float32", "measure",
   "Share of frames with movement:is_valid (invalidity comes in 60-frame processing blocks).",
   nullable=True, missing="face_features_status is 'not_provided' or 'not_measured'",
   provenance="meta:derived", value_range=(0.0, 1.0))
_f(T, "moi_status", "string", "status",
   "Meta MOI coverage of this recording: 'annotated_1p_3p', 'annotated_3p', 'not_annotated'. "
   "Absence of annotation is not absence of moments of interest.",
   allowed=("annotated_1p_3p", "annotated_3p", "not_annotated"), provenance="meta:derived", evidence="given")
_f(T, "recording_moi_3p_count", "int32", "measure",
   "Distinct 3P MOIs (distinct start/end intervals across 3P-IS/R/V).", nullable=True,
   missing="no 3P annotation (never 0 for unannotated)", provenance="meta:derived", evidence="aggregate")
_f(T, "recording_moi_1p_count", "int32", "measure", "Distinct 1P MOIs.", nullable=True,
   missing="no 1P annotation", provenance="meta:derived", evidence="aggregate")
_f(T, "recording_moi_3p_rate_per_min", "float32", "measure", "3P MOIs per minute.", unit="1/min",
   nullable=True, missing="no 3P annotation", provenance="meta:derived", evidence="aggregate")
_f(T, "moi_malformed_count", "int32", "measure",
   "MOI entries with zero or negative length or ending beyond the recording.", nullable=True,
   missing="no MOI annotation", provenance="meta:derived")
_f(T, "prompt_repeat_count_for_participant", "int32", "measure",
   "Other sessions in which this participant had the same prompt (10,605 repeats over 771 "
   "participants): repeated content for leakage/diversity control.", provenance="fresh",
   evidence="aggregate")

# =============================================================================
# clips (fixed-length segments of recordings)
# =============================================================================
T = "clips"
_f(T, "clip_id", "string", "id",
   "Clip key, '<file_id>_L30_C003': clip 3 of the 30-s grid, i.e. [90, 120) s. The grid "
   "length is part of the id, so ids from runs with another clip length never collide.",
   evidence="parsed")
_f(T, "file_id", "string", "link", "Recording the clip was cut from.", evidence="parsed")
_f(T, "interaction_key", "string", "link", "Conversation of the clip.", evidence="parsed")
_f(T, "session_key", "string", "link", "Session of the clip.", evidence="parsed")
_f(T, "participant_key", "string", "link", "Participant shown (the single tracked subject).", evidence="parsed")
_f(T, "window_id", "string", "link", "Dyad window this clip belongs to (interaction_windows).", evidence="parsed")
_partitions(T)
_f(T, "clip_index", "int16", "id_part", "Position k of the clip in its recording.", evidence="parsed")
_f(T, "partner_clip_id", "string", "link",
   "The partner's time-aligned clip (same interaction, same clip_index).", nullable=True,
   missing="see partner_link_status", evidence="parsed")
_f(T, "partner_link_status", "string", "status",
   "'linked', 'partner_missing', 'partner_not_measured', 'not_time_aligned' (pair durations "
   "disagree by > 1 s or a member drifts), 'partner_has_no_clip' (partner recording shorter).",
   allowed=("linked", "partner_missing", "partner_not_measured", "not_time_aligned", "partner_has_no_clip"),
   evidence="rule_inferred")
_f(T, "start_frame", "int32", "measure", "First frame (inclusive) on the released pose grid.", evidence="measured")
_f(T, "end_frame", "int32", "measure", "End frame (exclusive).", evidence="measured")
_f(T, "n_frames_clip", "int32", "measure", "end_frame - start_frame.", evidence="measured")
_f(T, "start_s", "float32", "measure", "start_frame / fps.", unit="s", evidence="measured")
_f(T, "end_s", "float32", "measure", "end_frame / fps.", unit="s", evidence="measured")
_f(T, "clip_seconds", "float32", "measure", "Clip duration.", unit="s", evidence="measured")
_f(T, "is_partial", "bool", "flag",
   "The recording ends inside this clip (its last clip, shorter than L). Kept, never dropped; "
   "clips under 2 s carry mostly NA measures.", evidence="measured")
_f(T, "is_last", "bool", "flag", "Last clip of the recording.", evidence="measured")
_f(T, "relative_position", "float32", "measure", "Clip centre / recording duration (0 = start, 1 = end).",
   value_range=(0.0, 1.0), evidence="measured")
for _what, _desc in (("speech", "an own-speech segment"), ("motion", "an active arm-motion episode")):
    for _edge in ("start", "end"):
        _f(T, f"{_what}_cut_at_{_edge}", "bool", "flag",
           f"{_desc.capitalize()} runs across the clip's {_edge} boundary: the action "
           f"{'began before' if _edge == 'start' else 'continues after'} the clip.",
           evidence="measured")
_f(T, "speech_annotation_status", "string", "status",
   "Whether own speech is annotated in this clip: 'annotated'; 'no_speech_annotation' (the "
   "recording has neither VAD nor timed words); 'beyond_audio' (the clip reaches > 1 s past "
   "recordings.speech_annotated_until_s: the WAV is shorter than the pose grid); 'vad_gap' "
   "(>= 10 timed words at > 8 words per second of VAD speech: Meta's VAD dropped out while "
   "the transcript continues — 1,220 recordings). Every speech-conditioned value is NA unless "
   "'annotated', never a false 0.",
   allowed=("annotated", "no_speech_annotation", "beyond_audio", "vad_gap"), evidence="rule_inferred")
_f(T, "own_speech_s", "float32", "measure", "Own speech time in the clip (recordings.speech_source mask).",
   unit="s", provenance="meta:derived", nullable=True, missing="speech unannotated in the clip (speech_annotation_status != 'annotated')")
_f(T, "own_speech_frac", "float32", "measure", "Own speech share of the clip.", value_range=(0.0, 1.0),
   provenance="meta:derived", nullable=True, missing="speech unannotated in the clip (speech_annotation_status != 'annotated')")
_f(T, "speaking_role", "string", "label",
   "Conversational role in the clip, from own (s) and partner (p) speech shares: 'speaking' "
   "(s >= 0.1 and s >= 2p), 'listening' (p >= 0.1 and p >= 2s), 'both', 'silent' (s, p < 0.1), "
   "'unknown' (partner clip not linked, or either clip's speech_annotation_status is not 'annotated').",
   allowed=("speaking", "listening", "both", "silent", "unknown"), evidence="rule_inferred")
_f(T, "word_count", "int32", "measure", "Transcript words whose midpoint falls in the clip.",
   provenance="meta:derived")
_f(T, "speech_rate_wps", "float32", "measure", "word_count / own_speech_s.", unit="words/s",
   nullable=True, missing="own_speech_s < 1 s, or speech unannotated in the clip (speech_annotation_status != 'annotated')", provenance="meta:derived")
# tracking quality -------------------------------------------------------------
for _name, _desc, _prov in (
    ("smplh_valid_frac", "Share of frames with smplh:is_valid (effectively hand-pose validity).", "prior:reused"),
    ("smplh_longest_invalid_s", "Longest run of smplh-invalid frames.", "prior:reused"),
    ("subject_present_frac", "Share of frames with a valid tracking box (was box_valid_frac).", "prior:reused"),
    ("hand_frozen_frac", "Share of frames whose hand pose is bit-identical to the previous frame.", "prior:reused"),
):
    _f(T, _name, "float32", "measure", _desc, unit="s" if _name.endswith("_s") else "",
       provenance=_prov, value_range=None if _name.endswith("_s") else (0.0, 1.0),
       nullable=True, missing="clip shorter than 2 s")
_f(T, "kp_conf_p10", "float32", "measure",
   "10th percentile over frames of mean released 2D keypoint confidence of the nose, "
   "shoulders, elbows, wrists and hips (confidence is not a probability; it exceeds 1).",
   nullable=True, missing="clip shorter than 2 s", provenance="prior:reused")
_f(T, "consistency_r", "float32", "measure",
   "Pearson r between SMPL-H 3D wrist speed and 2D keypoint hand speed: real motion appears in "
   "both channels, a fit wobbling alone in one.", nullable=True,
   missing="clip < 2 s or either speed series constant", provenance="prior:reused")
_f(T, "step_cosine_p50", "float32", "measure",
   "Median cosine between successive 2D hand steps: detector noise reverses (-0.9..-1), "
   "motion continues (+0.3..+0.9). Also partly an 'is there motion' signal.", nullable=True,
   missing="clip < 2 s or fewer than 8 large steps", provenance="prior:reused")
_f(T, "reprojection_error_p50_sw", "float32", "measure",
   "Clip median of the SMPL-H vs 2D keypoint reprojection error (see recordings).",
   unit="shoulder widths", nullable=True, missing="no frame with confident keypoints", provenance="prior:adapted")
_f(T, "box_center_jump_max", "float32", "measure",
   "Largest frame-to-frame jump of the tracking-box centre between valid frames, in box "
   "diagonals: large jumps suggest the tracker re-acquired or switched subject.",
   nullable=True, missing="fewer than 2 valid-box frames")
_f(T, "flag_tracker_jump", "bool", "flag",
   "box_center_jump_max > 0.5 box diagonals (False when it cannot be computed).", evidence="rule_inferred")
# framing / visibility ---------------------------------------------------------
for _part, _idx in (("shoulders", "5,6"), ("hips", "11,12"), ("knees", "13,14"), ("ankles", "15,16")):
    _f(T, f"{_part}_visible_frac", "float32", "measure",
       f"Share of frames with both {_part} (COCO {_idx}) confident (>= 0.5) and inside the "
       "raster. Keypoints are extrapolated beyond the raster, so confidence alone is not visibility.",
       value_range=(0.0, 1.0), nullable=True, missing="clip shorter than 2 s")
_f(T, "wrists_in_frame_frac", "float32", "measure",
   "Share of frames with both wrists confident and inside the raster (hands leaving frame).",
   value_range=(0.0, 1.0), nullable=True, missing="clip shorter than 2 s")
_f(T, "hands_kp_conf_p50", "float32", "measure",
   "Median over frames of mean 2D hand-keypoint confidence (91-132): hand observability.",
   nullable=True, missing="clip shorter than 2 s")
_f(T, "visible_extent", "string", "label",
   "How much of the body is in frame (>= 80% of frames): 'full_body' (ankles), 'to_knees', "
   "'to_hips', 'upper_body' (shoulders), 'unknown'.",
   allowed=("full_body", "to_knees", "to_hips", "upper_body", "unknown"), evidence="rule_inferred")
_f(T, "box_height_frac", "float32", "measure",
   "Median tracking-box height / picture height, in the upright display orientation: framing scale.",
   nullable=True, missing="no valid-box frame")
_f(T, "box_edge_contact_frac", "float32", "measure",
   "Share of valid-box frames whose box touches a raster edge (within 1%): cropping.",
   value_range=(0.0, 1.0), nullable=True, missing="no valid-box frame")
_f(T, "facing_angle_deg_p50", "float32", "measure",
   "Median angle between the body's forward axis (SMPL-H global orientation) and the "
   "direction to the camera: 0 = facing the camera, 90 = side-on. Unsigned, so rotation-"
   "invariant; typical clip medians 2-10 deg.", unit="deg", nullable=True,
   missing="clip < 2 s or severe anamorphic (orientation distorted)")
_f(T, "facing_angle_range_deg", "float32", "measure",
   "p90 - p10 of the facing angle within the clip: turning.", unit="deg", nullable=True,
   missing="as facing_angle_deg_p50")
_f(T, "facing_direction", "string", "label",
   "'toward_camera' (< 25 deg), 'angled' (25-60), 'side_on' (> 60), 'unknown'. The partner's "
   "position relative to the camera is not released, so 'facing the partner' is not inferable.",
   allowed=("toward_camera", "angled", "side_on", "unknown"), evidence="rule_inferred")
# motion -------------------------------------------------------------------------
_MOTION = (
    ("arm_speed_p50_mm_s", "Median faster-wrist speed in the torso frame (whole-body sway and turns removed).", "mm/s", "prior:reused", ""),
    ("arm_speed_speech_p50_mm_s", "Median faster-wrist speed over own-speech frames.", "mm/s", "prior:reused", "no own speech in the clip; speech unannotated in the clip (speech_annotation_status != 'annotated')"),
    ("arm_speed_cv", "Coefficient of variation of 1-s mean wrist speed: bursty (high) vs steady (low) motion; scale-free.", "", "fresh", "mean speed < 1 mm/s"),
    ("wrist_range_mm", "Largest axis-aligned extent of either wrist's path in the torso frame (grows with clip length).", "mm", "prior:reused", ""),
    ("wrist_excursion_p90_mm", "p90 over frames of the larger wrist's distance from its clip-median position.", "mm", "prior:reused", ""),
    ("elbow_range_mm", "As wrist_range_mm, elbows.", "mm", "prior:reused", ""),
    ("elbow_excursion_p90_mm", "As wrist_excursion_p90_mm, elbows.", "mm", "prior:reused", ""),
    ("hand_artic_p75_rad_s", "p75 finger articulation speed (local joint rotations, faster hand) over frames whose hand pose is not frozen.", "rad/s", "prior:adapted", "fewer than 2 s of non-frozen hand frames"),
    ("arm_active_frac", "Share of frames with active arm motion (wrist speed > 60 mm/s AND travel > 35 mm within 0.5 s; gaps <= 0.25 s bridged, runs < 0.3 s dropped). A descriptive definition, not a gate. Was gesture_frac.", "", "prior:reused", ""),
    ("arm_active_frac_speech", "arm_active_frac over own-speech frames.", "", "prior:reused", "no own speech; speech unannotated in the clip (speech_annotation_status != 'annotated')"),
    ("arm_active_frac_silence", "arm_active_frac over non-speech frames.", "", "prior:reused", "clip is all own speech; speech unannotated in the clip (speech_annotation_status != 'annotated')"),
    ("arm_episode_median_s", "Median duration of active-motion episodes (continuous vs discrete motion).", "s", "prior:adapted", "no episode in the clip"),
    ("wrist_height_p75_mm", "p75 over all frames of the higher wrist's height above the shoulder midpoint, along the spine (rest ~ -350 to -550).", "mm", "prior:adapted", ""),
    ("wrist_height_speech_p75_mm", "As wrist_height_p75_mm over own-speech frames only.", "mm", "prior:adapted", "no own speech (the old silent fallback to all frames is removed); speech unannotated in the clip (speech_annotation_status != 'annotated')"),
    ("arm_abduction_p75_deg", "p75 over all frames of the larger upper-arm angle from the torso's down axis.", "deg", "prior:adapted", ""),
    ("hands_together_frac", "Share of frames with the wrists within 180 mm (clasped rest posture).", "", "prior:adapted", ""),
    ("wrist_pose_spread_mm", "The larger, over the two wrists, of the mean pairwise distance between that wrist's positions sampled every 2.5 s over all frames (visits different arm positions vs stays put). Was posture_spread_mm (speech frames).", "mm", "prior:adapted", "fewer than 3 samples"),
    ("spine_motion_mm_s_p50", "Median speed of the shoulder midpoint relative to the pelvis (leaning/swaying of the spine; stepping and turning are not included). Was torso_travel_mm_s_p50.", "mm/s", "prior:reused", ""),
    ("speech_motion_sync_r", "Peak correlation (lags +-1.5 s, 0.5-s bins) between own-speech share and wrist speed.", "", "prior:reused", "speech or speed constant in the clip, or clip < 4 s; speech unannotated in the clip (speech_annotation_status != 'annotated')"),
    ("speech_motion_sync_lag_s", "Lag of the peak (+ = motion follows speech). Read only when sync_r is meaningful.", "s", "prior:reused", "as speech_motion_sync_r; speech unannotated in the clip (speech_annotation_status != 'annotated')"),
    ("head_speed_p50_deg_s", "Median angular speed of the head relative to the upper spine (SMPL-H head vs spine3).", "deg/s", "fresh", ""),
    ("head_speed_p75_deg_s", "p75 of the head's angular speed relative to the upper spine.", "deg/s", "fresh", ""),
)
for _name, _desc, _unit, _prov, _missing in _MOTION:
    _f(T, _name, "float32", "measure", _desc, unit=_unit, provenance=_prov,
       nullable=True, missing=("clip shorter than 2 s" + (f"; {_missing}" if _missing else "")),
       qualified_by=("smplh_valid_frac", "wrists_in_frame_frac"), source="SMPL-H (betas=0 canonical skeleton)")
_f(T, "arm_episode_count", "int16", "measure", "Active-motion episodes in the clip (truncated at clip edges).",
   nullable=True, missing="clip shorter than 2 s", provenance="prior:reused")
_f(T, "speech_segment_count", "int16", "measure", "Own-speech segments of >= 0.8 s in the clip.",
   nullable=True, missing="clip shorter than 2 s, or speech unannotated in the clip (speech_annotation_status != 'annotated')", provenance="prior:reused")
_f(T, "speech_segments_with_motion_frac", "float32", "measure",
   "Share of those speech segments containing >= 0.2 s of active arm motion (co-speech "
   "coverage). Was speech_segments_covered.", nullable=True,
   missing="clip < 2 s, no speech segment, or speech unannotated in the clip (speech_annotation_status != 'annotated')", provenance="prior:reused", value_range=(0.0, 1.0))
# posture ------------------------------------------------------------------------
_f(T, "posture", "string", "label",
   "Posture of the clip's participant: 'standing' (upright, legs extended; includes perching "
   "on a high stool with legs extended), 'sitting', 'mixed' (a sustained change within the "
   "clip; on V03 a sustained forward bend can also produce it), 'unclear' (legs visible but "
   "measurements ambiguous; on V00 mostly standing people), 'unknown' (not measurable: legs "
   "not visible, anamorphic pose). From 1-s bins: see docs for the rules and the visual QA.",
   allowed=POSTURES, provenance="prior:adapted", evidence="rule_inferred",
   qualified_by=("posture_confidence",),
   note="The v0 FM1 seated detector (a rejection gate, retired) adapted into a label with an "
   "unclear band instead of a single forced cut.")
_f(T, "posture_confidence", "float32", "confidence",
   "Share of the clip's 1-s bins whose measurement supports the label (for 'mixed': decided "
   "bins). Evidence support, not P(correct).", value_range=(0.0, 1.0), nullable=True,
   missing="posture is 'unclear' or 'unknown'", evidence="aggregate")
for _st in ("standing", "sitting", "unclear", "unobserved"):
    _f(T, f"posture_{_st}_frac", "float32", "measure",
       f"Share of the clip's 1-s bins classified '{_st}' (the four shares sum to 1).",
       value_range=(0.0, 1.0), evidence="aggregate")
_f(T, "posture_transitions", "int16", "measure",
   "Standing<->sitting changes between sustained runs (>= 5 decided bins) inside the clip.",
   evidence="aggregate")
_f(T, "hip_flexion_deg_p50", "float32", "measure", "Clip median torso-thigh angle (see recordings).",
   unit="deg", nullable=True, missing="< 50% hip-observable frames or severe anamorphic", provenance="prior:adapted")
_f(T, "knee_between_p50", "float32", "measure", "Clip median knee position along the leg (see recordings).",
   nullable=True, missing="< 50% ankle-observable frames or severe anamorphic", provenance="prior:adapted")
_f(T, "shin_inverted_frac", "float32", "measure",
   "Share of ankle-observable frames with both ankles above the knees along the torso axis "
   "(feet up, e.g. on a stool rung).", nullable=True, missing="< 50% ankle-observable frames or severe anamorphic",
   provenance="prior:adapted", value_range=(0.0, 1.0))
_f(T, "lower_body_observed_frac", "float32", "measure",
   "Share of frames with both hips and knees visible (the posture observability gate).",
   value_range=(0.0, 1.0), nullable=True, missing="clip shorter than 2 s")
# face ---------------------------------------------------------------------------
_f(T, "face_valid_frac", "float32", "measure", "Share of frames with movement:is_valid.",
   value_range=(0.0, 1.0), nullable=True, missing="no Imitator features (non-V00)", provenance="meta:derived")
for _name, _desc in (
    ("fau_intensity_mean", "Mean facial action unit intensity (movement:FAUValue, 21 live AUs; 16-18 are dead) over valid frames."),
    ("fau_variability", "Mean over AUs of the within-clip SD of AU intensity: facial expression dynamics."),
    ("emotion_arousal_mean", "Mean movement:emotion_arousal over valid frames."),
    ("emotion_valence_mean", "Mean movement:emotion_valence over valid frames."),
    ("emotion_arousal_sd", "SD of movement:emotion_arousal over valid frames."),
):
    _f(T, _name, "float32", "measure", _desc, nullable=True,
       missing="no Imitator features (non-V00) or < 2 s valid face frames",
       provenance="meta:derived", source="movement:* (V00 only)", qualified_by=("face_valid_frac",))
# audio --------------------------------------------------------------------------
_f(T, "audio_rms_db_p50", "float32", "measure", "Median 50 ms RMS level of the clip.", unit="dBFS",
   nullable=True, missing="audio not ok, or < 1 s of audio in the clip (WAV shorter than the video)")
_f(T, "own_speech_level_db", "float32", "measure", "Median level over own-speech 50 ms frames.",
   unit="dBFS", nullable=True, missing="audio not ok, < 1 s own speech, or speech unannotated in the clip (speech_annotation_status != 'annotated')")
_f(T, "vocal_level_range_db", "float32", "measure",
   "p90 - p10 of the level over own-speech frames: vocal energy variation.", unit="dB",
   nullable=True, missing="audio not ok, < 3 s own speech, or speech unannotated in the clip (speech_annotation_status != 'annotated')")
# MOI ----------------------------------------------------------------------------
_f(T, "moi_3p_count", "int16", "measure", "Distinct 3P MOIs overlapping the clip.", nullable=True,
   missing="recording has no 3P annotation (never 0 for unannotated)", provenance="meta:derived", evidence="aggregate")
_f(T, "moi_1p_count", "int16", "measure", "Distinct 1P MOIs overlapping the clip.", nullable=True,
   missing="recording has no 1P annotation", provenance="meta:derived", evidence="aggregate")
_f(T, "moi_3p_seconds", "float32", "measure", "Union duration of 3P MOIs within the clip.", unit="s",
   nullable=True, missing="recording has no 3P annotation", provenance="meta:derived", evidence="aggregate")
# expressivity -------------------------------------------------------------------
for _name, _desc in (
    ("expr_energy", "Arm energy: percentile of arm_speed_p50_mm_s."),
    ("expr_amplitude", "Arm amplitude: mean percentile of wrist_range_mm and wrist_excursion_p90_mm."),
    ("expr_head", "Head motion: percentile of head_speed_p75_deg_s."),
    ("expr_hands", "Hand articulation: percentile of hand_artic_p75_rad_s."),
    ("expr_variability", "Temporal variability (burstiness): percentile of arm_speed_cv. Not part of expressivity_score."),
    ("expr_face", "Facial expressivity: percentile of fau_variability (V00 only). Not part of expressivity_score."),
    ("expr_vocal", "Vocal energy variation: percentile of vocal_level_range_db (clips with >= 3 s of own speech, any speaking_role). Not part of expressivity_score."),
):
    _f(T, _name, "float32", "score",
       _desc + " Percentiles are mid-rank within the recording's expressivity_reference_group "
       "(full-length measured clips).", value_range=(0.0, 1.0), nullable=True,
       missing="expressivity_status != 'measured' or the input measure is NA", evidence="measured")
_f(T, "expressivity_score", "float32", "score",
   "Overall body expressivity (arms, hands, head): the mean of expr_energy, expr_amplitude, "
   "expr_head and expr_hands, re-ranked within the reference group so it is uniform over "
   "reference clips (0.7 = more expressive than 70% of full-length clips of the same "
   "vendor/rig). Within-rig ranking removes rig/fit offsets (V02 fits move ~30-40% faster at "
   "matched speech); see expressivity_score_pooled for a corpus-wide rank.",
   value_range=(0.0, 1.0), nullable=True, missing="expressivity_status != 'measured'",
   qualified_by=("expressivity_confidence",), evidence="measured")
_f(T, "expressivity_level", "string", "label",
   "'low' (< 1/3), 'medium', 'high' (>= 2/3) of expressivity_score; 'unknown' when NA.",
   allowed=("low", "medium", "high", "unknown"), evidence="rule_inferred")
_f(T, "expressivity_score_pooled", "float32", "score",
   "As expressivity_score but ranked against all reference clips of every vendor: mixes rig "
   "and fit effects with population differences.", value_range=(0.0, 1.0), nullable=True,
   missing="expressivity_status != 'measured'", evidence="measured")
_f(T, "expressivity_level_pooled", "string", "label", "Tertile level of expressivity_score_pooled.",
   allowed=("low", "medium", "high", "unknown"), evidence="rule_inferred")
_f(T, "expressivity_confidence", "float32", "confidence",
   "Evidence coverage for the score: min(1, clip_seconds/L) x share of frames with a valid "
   "box and both wrists in frame. Not P(correct).", value_range=(0.0, 1.0), nullable=True,
   missing="expressivity_status != 'measured'", evidence="aggregate")
_f(T, "expressivity_status", "string", "status",
   "'measured', 'too_short' (< 2 s), 'pose_distorted' (severe anamorphic V01), "
   "'no_subject' (no valid tracking box), 'incomplete' (a composite channel is NA).",
   allowed=("measured", "too_short", "pose_distorted", "no_subject", "incomplete"), evidence="rule_inferred")
# redundancy / representativeness -------------------------------------------------
_f(T, "distance_to_previous_clip_z", "float32", "measure",
   "RMS difference to the previous clip of the same recording over a standardised feature "
   "vector (expressivity channels, wrist height, hip flexion, facing, own speech), in "
   "corpus-SD units: small = redundant with its neighbour.", nullable=True,
   missing="first clip, or either clip lacks the features")
_f(T, "distance_to_recording_mean_z", "float32", "measure",
   "RMS difference to the recording's mean feature vector (same features; a feature counts "
   "only where at least two clips of the recording have it): small = representative of the "
   "participant's behaviour in this interaction.", nullable=True,
   missing="clip lacks the features, or fewer than two clips of the recording have them")
# visual (pixel pass) ---------------------------------------------------------------
_f(T, "visual_status", "string", "status",
   "'measured', 'not_run' (scan-video not run), 'no_frame' (no decodable frame in the clip), "
   "'decode_error'.", allowed=("measured", "not_run", "no_frame", "decode_error"))
_f(T, "frame_time_s", "float32", "measure", "Time of the sampled (key)frame.", unit="s",
   nullable=True, missing="visual_status != 'measured'")
for _name, _desc in (
    ("frame_sharpness", "Variance of the Laplacian inside the subject box on the upright frame downscaled to 540 px short side: blur (low) vs sharp."),
    ("frame_luma_mean", "Mean luma (0-255) of the frame: exposure."),
    ("frame_luma_clipped_frac", "Share of pixels at <= 5 or >= 250 luma: under/over-exposure."),
    ("background_edge_density", "Share of Canny edge pixels outside the subject box: background clutter."),
    ("camera_shift_px", "Phase-correlation shift of the background against the previous clip's frame (px at 540 px short side): camera movement."),
):
    _extra_missing = {"camera_shift_px": " or first measured clip, or featureless background",
                      "frame_sharpness": " or no subject box at the frame",
                      "background_edge_density": " or no subject box at the frame"}.get(_name, "")
    _f(T, _name, "float32", "measure", _desc, nullable=True,
       missing="visual_status != 'measured'" + _extra_missing, source="MP4 keyframe")

# =============================================================================
# moi_events
# =============================================================================
T = "moi_events"
_f(T, "moi_id", "string", "id", "Event key '<file_id>_M3PIS_002' (kind and index within the file).", evidence="parsed")
_f(T, "file_id", "string", "link", "Recording annotated.", evidence="parsed")
_f(T, "interaction_key", "string", "link", "Conversation.", evidence="parsed")
_f(T, "participant_key", "string", "link", "Participant the annotation is about.", evidence="parsed")
_partitions(T)
_f(T, "annotation_kind", "string", "metadata",
   "Meta annotation type: 1P-IS (self-reported internal state), 1P-R (self rationale), 3P-IS "
   "(observer-perceived state), 3P-R (observer rationale), 3P-V (observer visual description).",
   allowed=("1P-IS", "1P-R", "3P-IS", "3P-R", "3P-V"), provenance="meta:as_is", evidence="given")
_f(T, "party", "string", "metadata", "'1P' or '3P'.", allowed=("1P", "3P"), provenance="meta:derived", evidence="given")
_f(T, "moi_start_s", "float32", "measure", "Start, integer seconds from recording start, as released.",
   unit="s", provenance="meta:as_is", evidence="given")
_f(T, "moi_end_s", "float32", "measure", "End, as released.", unit="s", provenance="meta:as_is", evidence="given")
_f(T, "moi_duration_s", "float32", "measure", "moi_end_s - moi_start_s (can be 0 or negative as released).",
   unit="s", provenance="meta:derived", evidence="given")
_f(T, "moi_text", "string", "metadata", "The annotation text, as released.", provenance="meta:as_is", evidence="given")
_f(T, "moi_clip_index_start", "int16", "measure",
   "Clip index containing the event start (on the nominal k*L grid; MOI times are whole seconds).",
   nullable=True, missing="the recording has no clips, or the event lies wholly at or after the recording's end",
   evidence="measured")
_f(T, "moi_clip_index_end", "int16", "measure", "Clip index containing the event's last instant.",
   nullable=True, missing="as moi_clip_index_start", evidence="measured")
_f(T, "moi_malformed", "string", "status",
   "'ok', 'zero_length', 'negative_length', 'beyond_recording' (ends > 0.5 s after the recording).",
   allowed=("ok", "zero_length", "negative_length", "beyond_recording"), provenance="meta:derived",
   evidence="rule_inferred")
_f(T, "partner_duplicate_moi_id", "string", "link",
   "The partner's event with the same kind, same normalised text and start within 2 s: a "
   "copied annotation whose target person is ambiguous.", nullable=True,
   missing="no duplicate in the partner's annotations", evidence="rule_inferred")


# =============================================================================
# registry API
# =============================================================================
def fields(table: str | None = None) -> list[Field]:
    return [f for f in _REGISTRY if table is None or f.table == table]


def columns(table: str) -> list[str]:
    return [f.name for f in fields(table)]


def field(table: str, name: str) -> Field:
    for f in _REGISTRY:
        if f.table == table and f.name == name:
            return f
    raise KeyError(f"{table}.{name} is not registered")


SHARED_ROLES = ("id", "link", "id_part", "group_key", "partition")


def registry_problems() -> list[str]:
    """Static checks of the registry itself (run by the tests)."""

    problems: list[str] = []
    seen: dict[str, list[Field]] = {}
    for f in _REGISTRY:
        problems.extend(f.check())
        seen.setdefault(f.name, []).append(f)
    for name, entries in seen.items():
        tables = [e.table for e in entries]
        if len(set(tables)) != len(tables):
            problems.append(f"{name} registered twice in one table")
        if len(entries) > 1 and not all(e.role in SHARED_ROLES for e in entries):
            problems.append(f"{name} appears in {tables} but is not a key/partition column")
        if len(entries) > 1 and len({e.dtype for e in entries}) > 1:
            problems.append(f"{name} has different dtypes across tables")
    for table, key in PRIMARY_KEYS.items():
        try:
            if field(table, key).role != "id":
                problems.append(f"{table}.{key} must have role id")
        except KeyError:
            problems.append(f"{table} has no primary key column {key}")
    for f in _REGISTRY:
        if f.role == "link" and f.name not in LINK_TARGETS:
            problems.append(f"{f.table}.{f.name}: link without a target table")
    if any(f.name == "interaction_id" for f in _REGISTRY):
        problems.append("interaction_id must not be published: it is a prompt id (use prompt_hash)")
    return problems


def arrow_schema(table: str):
    """The Arrow schema every writer of ``table`` must use (no per-shard inference)."""

    import pyarrow as pa

    mapping = {
        "string": pa.string(), "bool": pa.bool_(), "int8": pa.int8(), "int16": pa.int16(),
        "int32": pa.int32(), "int64": pa.int64(), "float32": pa.float32(), "float64": pa.float64(),
    }
    return pa.schema(
        [pa.field(f.name, mapping[f.dtype], nullable=f.nullable) for f in fields(table)],
        metadata={b"seamless_curation.table": table.encode(), b"level": LEVELS[table].encode()},
    )


def catalog_frame():
    """The registry as a DataFrame (written as field_catalog.csv/.parquet)."""

    import pandas as pd

    rows = []
    for f in _REGISTRY:
        rows.append({
            "table": f.table, "level": f.level, "column": f.name, "dtype": f.dtype, "role": f.role,
            "unit": f.unit, "allowed": "|".join(str(a) for a in f.allowed), "nullable": f.nullable,
            "missing_means": f.missing, "provenance": f.provenance, "evidence": f.evidence,
            "qualified_by": "|".join(f.qualified_by), "source": f.source,
            "description": f.description, "provenance_note": f.note,
        })
    return pd.DataFrame(rows)


def iter_tables() -> Iterable[tuple[str, list[Field]]]:
    for table in TABLES:
        yield table, fields(table)


_PANDAS_NULLABLE_INT = {"int8": "Int8", "int16": "Int16", "int32": "Int32", "int64": "Int64"}


def conform(frame, table: str):
    """Select and order ``table``'s registered columns and cast them to their dtypes.

    Raises ``KeyError`` naming every registered column the frame lacks, and
    ``ValueError`` when a non-nullable column holds NA — the two ways a table can
    silently disagree with its documentation.
    """

    import numpy as np
    import pandas as pd

    missing = [name for name in columns(table) if name not in frame.columns]
    if missing:
        raise KeyError(f"{table}: missing registered columns {missing}")
    out = {}
    for f in fields(table):
        series = frame[f.name]
        if f.dtype == "string":
            values = series.astype("string")
        elif f.dtype == "bool":
            if series.isna().any():
                raise ValueError(f"{table}.{f.name}: NA in a bool column")
            values = series.astype(bool)
        elif f.dtype in _PANDAS_NULLABLE_INT:
            numeric = pd.to_numeric(series, errors="raise")
            if f.nullable:
                values = numeric.round().astype(_PANDAS_NULLABLE_INT[f.dtype])
            else:
                if numeric.isna().any():
                    raise ValueError(f"{table}.{f.name}: NA in a non-nullable integer column")
                values = numeric.round().astype(f.dtype)
        else:
            values = pd.to_numeric(series, errors="raise").astype(f.dtype)
        if not f.nullable and values.isna().any():
            raise ValueError(f"{table}.{f.name}: {int(values.isna().sum())} NA values in a non-nullable column")
        out[f.name] = values.reset_index(drop=True)
    return pd.DataFrame(out)


def to_arrow(frame, table: str):
    """A conformed frame as an Arrow table carrying the registry schema."""

    import pyarrow as pa

    return pa.Table.from_pandas(frame, schema=arrow_schema(table), preserve_index=False)

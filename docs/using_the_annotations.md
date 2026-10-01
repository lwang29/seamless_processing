# Using the annotations

The annotation tables describe **every** recording and every 30-second clip of
Meta's Seamless Interaction release. Nothing has been filtered out: a clip that
is seated, silent, occluded, anamorphic, partial or dull is in the tables with
columns saying so. Building a training or evaluation set is a query you write.

* Tables: `/simurgh/group/lw29/seamless_annotations/annotations_v1/annotations/`
  (one parquet per level; mode 0700 — participant-derived data).
* Every column, its meaning, unit, NA meaning and provenance:
  [`annotation_schema.md`](annotation_schema.md) (generated from the registry).
* How each annotation is computed and how far it can be trusted:
  [`pipeline.md`](pipeline.md) and `annotations/annotation_report.md`.
* The release itself is read in place: `./seamless_interaction/<source_relbase>.{npz,wav,mp4,json}`.

```python
from seamless_curation.dataset import load_clips, load_table, load_tables, iter_clip_data

ROOT = "/simurgh/group/lw29/seamless_annotations/annotations_v1/annotations"
clips = load_clips(ROOT)                                   # clip level only
clips = load_clips(ROOT, levels=("recording", "interaction", "session", "participant"))
```

## Levels and keys

| level | table | key | what a row is |
|---|---|---|---|
| participant | `participants` | `participant_key` `V00_P0061` | one person (vendor-qualified: bare ids collide across vendors) |
| session | `sessions` | `session_key` `V00_S0039` | one dyad's recording block; relationship lives here |
| prompt | `prompts` | `prompt_hash` `00000581` | one prompt: type, IPC codes, texts |
| interaction | `interactions` | `interaction_key` `V00_S0039_I00000581` | one conversation (both members) |
| dyad window | `interaction_windows` | `window_id` `..._L30_W003` | the conversation over one clip interval |
| recording | `recordings` | `file_id` `V00_S0039_I00000581_P0061` | one participant's files for one interaction |
| clip | `clips` | `clip_id` `..._P0061_L30_C003` | seconds [90, 120) of that recording |
| MOI event | `moi_events` | `moi_id` | one Meta 1P/3P annotation entry |

Parent keys are prefixes of child keys, and every child row carries its parents'
keys plus `vendor`/`label`/`split`. **Never group conversations by the file id's
I-segment**: it is a prompt id reused across up to 1,550 sessions. Use
`interaction_key`.

`load_clips(levels=...)` joins parents under a level prefix
(`session__relationship`, `interaction__posture_pair`,
`participant__bfi_extraversion`). The prefix is deliberate: a conversation-level
value on a clip row is a property of the conversation, repeated on each of its
clips — not an independent per-clip observation. To aggregate conversation
properties, work on the `interactions` table, not on clip rows.

## Missing values mean something

A missing value is never a stand-in for zero, and each nullable column's
`missing_means` entry in the schema says exactly when it is NA. The common cases:

* `moi_3p_count` is NA for clips of recordings Meta did not annotate (only 1,663
  of 129,370 files have 3P annotations). NA means "not annotated", not "no moment".
* Face columns (`fau_*`, `emotion_*`, `expr_face`) exist only for V00 — Meta
  released Imitator features for V00 alone (`recordings.face_features_status`).
* `posture` is `unknown` when the legs are not visible or the pose is anamorphic;
  `unclear` when they are visible but the measurement is in the ambiguous band. On
  V01 square-pixel and V02 (no hand labels) only `standing` or `unclear` is decided.
* Own speech and everything conditioned on it (`own_speech_*`, `arm_active_frac_speech`,
  `speaking_role`, ...) is NA / `unknown` where `speech_annotation_status` is not
  `annotated`: no VAD or transcript, a WAV shorter than the video, or Meta's VAD
  dropping out while the transcript continues (`vad_gap`) — never a false silence.
* Pair speech values (`interaction_speech_overlap_frac`, window measures) are NA
  unless both members were measured, annotated, and their recordings agree in
  length; see `speech_status` and `pair_duration_agreement`.
  `interaction_expressivity_mean`/`_gap` need both members measured (lengths may differ).
* Flags (`flag_*`, `is_partial`, ...) are never NA.

## Recipes

### Threshold, filter, sort

```python
high = clips[(clips.expressivity_score > 0.7) & (clips.expressivity_status == "measured")]
standing = clips[clips.posture == "standing"]
confident_standing = clips[(clips.posture == "standing") & (clips.posture_confidence >= 0.9)]
most_active = clips.sort_values("arm_speed_p50_mm_s", ascending=False)
```

`expressivity_score` is ranked within the clip's vendor/rig
(`recordings.expressivity_reference_group`), so 0.7 means "more expressive than
70% of full-length clips of the same rig". Use `expressivity_score_pooled` for a
corpus-wide rank, knowing it mixes rig and fit effects (V02 fits read ~30-40%
faster at matched speech).

### Condition on posture x expressivity x speaking role

```python
subset = clips[(clips.posture == "standing") & (clips.expressivity_level == "high")
               & (clips.speaking_role == "speaking") & ~clips.is_partial]
pd.crosstab([clips.vendor, clips.posture], clips.expressivity_level)   # stratify
```

### Speaker/listener pairs (dyadic models)

```python
pairs = load_clips(ROOT, levels=("interaction", "session"), partner=True)
pairs = pairs[pairs.partner_link_status == "linked"]          # time-aligned partner clip exists
speaker_listener = pairs[(pairs.speaking_role == "speaking") & (pairs.partner__speaking_role == "listening")]
speaker_listener[["clip_id", "partner_clip_id", "session__relationship", "interaction__posture_pair",
                  "partner_participant__bfi_extraversion"]]
```

Dyad-level measures over the same interval (speech overlap, mutual silence,
floor changes) are one row per window in `interaction_windows`, joinable on
`window_id`.

### Conversation-level selection and aggregation

```python
inter = load_table(ROOT, "interactions")
prompts = load_table(ROOT, "prompts")
sessions = load_table(ROOT, "sessions")
conv = inter.merge(prompts, on="prompt_hash").merge(sessions.drop(columns=["vendor", "label", "split"]), on="session_key")
lively = conv[(conv.interaction_expressivity_mean > 0.6) & (conv.interaction_speech_balance > 0.5)]
by_type = conv.groupby("interaction_type")["interaction_expressivity_mean"].mean()
per_conversation = clips.groupby("interaction_key")["expressivity_score"].mean()   # unweighted; the table's
# interaction_expressivity_mean is duration-weighted, so the two differ where partial clips are short
```

`interaction_type` is Meta's label; for 9 prompts (8,007 files) the prompt text
contradicts it (`interaction_type_text_consistency == "contradicts_text"`).

### Leakage-free evaluation that respects Meta's splits

Meta's splits are participant-level by design, but 26 participants appear in two
splits, and a quarter of dev interactions have a member who also has train
files. Keep the official split and drop the leaky rows:

```python
par = load_table(ROOT, "participants")
inter = load_table(ROOT, "interactions")
leaky_people = set(par.loc[par.flag_split_conflict, "participant_key"])
clean_inter = set(inter.loc[inter.n_members_in_other_split == 0, "interaction_key"])
eval_clips = clips[clips.split.isin(["dev", "test"]) & clips.interaction_key.isin(clean_inter)
                   & ~clips.participant_key.isin(leaky_people)]
# stricter: also exclude possible A-suffix aliases
clean_strict = set(inter.loc[(inter.n_members_in_other_split == 0)
                             & (inter.n_members_suffix_sibling_other_split == 0), "interaction_key"])
```

Clips of one recording, interaction, session or participant are correlated. For
any custom cross-validation *within* a split, group by `participant_key` (single
person models), `sessions.dyad_key` or `participants.participant_component_id`
(dyadic models: keeps every partner-of-partner on one side). Do not split by clip.

### Data quality

```python
rec = load_table(ROOT, "recordings")
usable_audio = rec[rec.audio_quality == "ok"]
no_room_bleed = rec[~rec.room_camera_rig]
clean_timebase = rec[rec.timebase_status == "consistent"]
tracked = clips[(clips.smplh_valid_frac > 0.9) & (clips.wrists_in_frame_frac > 0.9) & ~clips.flag_tracker_jump]
sharp = clips[(clips.visual_status == "measured")]   # the pixel pass; compare frame_sharpness within a raster_class
```

Meta's documented caveats and the columns that measure them are listed in
`pipeline.md`, "Meta's caveats". No per-interaction list of affected
interactions is published, so "not flagged" never means "known unaffected".

### Reconstructing a co-speech gesture subset like the previous iteration's

The previous pipeline kept 50,516 clips it judged good co-speech gesture training
data. Its decision is not pre-applied here, but most of its ingredients are
columns, so an **approximate** version is one query:

```python
cospeech = clips[
    # its tier-1 gates that exist as columns
    (clips.smplh_valid_frac >= 0.90) & (clips.smplh_longest_invalid_s <= 1.0) & (clips.hand_frozen_frac <= 0.05)
    & (clips.kp_conf_p10 >= 0.30) & (clips.wrist_excursion_p90_mm >= 80) & (clips.elbow_excursion_p90_mm >= 35)
    & (clips.wrist_pose_spread_mm >= 150) & (clips.hands_together_frac <= 0.55) & (clips.arm_abduction_p75_deg >= 17)
    # its tier-2 disqualifiers that exist as columns
    & (clips.own_speech_s >= 8) & (clips.arm_active_frac_speech >= 0.5)
    & (clips.wrist_height_speech_p75_mm >= -260) & (clips.speech_segments_with_motion_frac >= 0.6)
    & (clips.step_cosine_p50 >= 0.25) & (clips.consistency_r >= 0.70) & (clips.arm_episode_median_s >= 0.55)]
```

Not equivalent: three of its clauses have no column here (episodes during speech,
the arm/torso articulation ratio, its 0.34 `gesture_quality` score), the old
spread and height clauses read speech frames where these read all frames, its
windows sat on a 10-s hop, its caps kept at most 8 clips per file and 12 files per
participant, and it excluded V01's anamorphic and V03's room-camera rasters and all
charades prompts.

## Loading the released arrays for a clip

```python
rows = load_clips(ROOT, levels=("recording",))          # brings recording__source_relbase, recording__fps
for clip in iter_clip_data(rows.head(10), "seamless_interaction", with_keypoints=True):
    clip.body_pose        # (frames, 21, 3) axis-angle
    clip.left_hand_pose   # (frames, 15, 3)
    clip.keypoints        # (frames, 133, 3), confidence can exceed 1
    clip.audio            # the participant's own microphone (partner bleed: recordings.voice_isolation_db)
    clip.speech           # own VAD segments relative to the clip
```

`translation` holds a ~1e12 sentinel on frames without a tracking box.

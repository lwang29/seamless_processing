# The annotation pipeline, step by step

This document describes the pipeline **as implemented**: what each stage reads
and writes, how every annotation is computed, what its thresholds rest on, what
it cannot see, and where each value comes from. The column-by-column reference is
[`annotation_schema.md`](annotation_schema.md), generated from the registry in
`src/seamless_curation/schema.py`; the numbers a run produced (coverage,
distributions, validation evidence) are in `annotations/annotation_report.md`.

**What changed.** Until 2026-09 this repository *filtered* the release down to
30-second co-speech-gesture training clips (tier-1 gates, a tier-2 score, caps
per file and participant). It now **annotates every recording and every clip**,
so a downstream user can threshold, filter, stratify or condition on any
property. No stage makes an inclusion decision: a recording that cannot be
measured still gets a row saying why, and every clip of every measured recording
gets a row. The previous pipeline is preserved at the git tag
`v1-cospeech-filter`; how its measures and labels were carried over is in
[§8](#8-what-was-reused-adapted-replaced-or-removed).

---

## 1. Units and keys

The task this pipeline was redesigned for called one participant's recording of
one interaction a "clip". In this repository that unit is a **recording** (Meta's
file), and a **clip** is a fixed-length segment of a recording — the previous
pipeline already worked on 30-s segments, and segment-level annotation is what
lets posture change within an interaction, clips be matched to a partner's, and
a training window be selected.

| level | key | notes |
|---|---|---|
| participant | `V00_P0061` | bare ids collide across vendors (627 ids); V00 has 165 `A`-suffixed ids |
| session | `V00_S0039` | session ids collide across vendors |
| prompt | `00000581` | Meta's `prompt_hash` = the file id's I-segment ("interaction_id") |
| interaction | `V00_S0039_I00000581` | the conversation; **the I-segment alone is a prompt id, reused by up to 1,550 sessions** |
| recording | `V00_S0039_I00000581_P0061` | Meta's file id |
| clip | `V00_S0039_I00000581_P0061_L30_C003` | clip 3 of the 30-s grid, [90, 120) s; the grid length is part of the id |
| dyad window | `V00_S0039_I00000581_L30_W003` | the conversation over clip 3's interval |

Every key is a string (zero padding and suffixes preserved) and each parent key
is a prefix of its children's. `src/seamless_curation/ids.py` builds them all.

**Whose body.** Every NPZ tracks one subject: the recording's own participant
(one box and one keypoint set per frame; box-centre jumps ≤ 0.2 box heights on a
52-file sample; the partner was never in view in sampled frames, including the
V03 room-camera rasters). So every body, posture, framing, face and motion field
describes the recording's participant, per clip; the partner's values are on the
partner's own clip row (`partner_clip_id`), and `interactions.posture_pair` /
`interaction_windows.window_posture_pair` summarise the dyad.

## 2. Stages

```
seamless_interaction/  (read-only)      datasets/seamless_interaction_metadata/  (Meta CSVs)
        |                                           |
        |          outputs/02_inventory (M-1 census: stat + ffprobe of every filelist row)
        |                                           |
        |                              [catalog]  every recording + participants, sessions,
        |                                           |   prompts, interactions metadata   (seconds)
        +------------------------------> [scan]    continuous measurements per interaction
        |                                           |   Slurm array, 512 tasks            (hours)
        +------------------------------> [scan-video] one keyframe per clip (optional)    (hours)
                                                    |
                                        [annotate]  labels, scores, links, aggregates,
                                                    |   validate, publish atomically      (minutes)
                                        [validate] [report] [schema-docs]
```

| stage | reads | writes (under `outputs.root`) |
|---|---|---|
| `catalog` | census, Meta CSVs | `catalog/{recordings,participants,sessions,prompts,interactions}.parquet` |
| `scan` | catalog, NPZ + JSON + WAV of both members | `scan_shards/task_NNNN.{recordings,clips,bins,moi,windows,pairs}.parquet` + marker |
| `scan-video` | scan shards, MP4 + NPZ box | `video_shards/task_NNNN.video.parquet` + marker |
| `annotate` | catalog, all shards | `annotations/*.parquet` + side files |
| `validate` | published tables | `validation_rerun.json` (`--read-back N` re-reads clips from the release) |
| `report` | published tables, legacy labels | `annotations/annotation_report.md` |
| `schema-docs` | registry | `docs/annotation_schema.md` |

**The scan stores continuous values only.** Every threshold, band, label and
normalisation is applied in `annotate`, which reads no media, so re-tuning a
posture band (the config's `posture:` block) costs one annotate run. The scan's
unit of work is the interaction: both members are measured in one task (partners
sit in different release archives for 98.6% of pairs), so pair measures —
speech overlap per window, partner bleed into each microphone, transcript echo,
annotations copied between members — are computed once from both recordings.
Shards carry a fingerprint (every measurement module's source, the parameters,
the SMPL-H model file) and a membership hash (the files *and* the per-file
inputs: fps, raster, anamorphic class, path); `annotate` refuses a missing,
stale or re-membered shard.

**Publication is atomic.** `annotate` casts every table to the registry's Arrow
schema, runs the validator, and refuses to publish on any error; it writes the
eight tables into a staging directory and swaps it in with one rename. Each
parquet carries the run hash, and `dataset.load_tables` refuses to mix runs.

**Cost** (full corpus): catalog seconds; scan ~4 s of CPU per recording but
I/O-bound — the WAVs are 5.6 TB and the NPZs 1.9 TB — about 3-4 h wall at ~100
concurrent tasks; scan-video ~0.03-0.14 s per clip; annotate a few minutes.

## 3. The clip grid

Clip *k* covers `[k*L, (k+1)*L)` seconds of its recording (`L = clips.seconds`,
30), mapped to frames with the recording's own rate: `start = round(k*L*fps)`.
The final clip is kept however short (`is_partial`), so every frame of every
measured recording belongs to exactly one clip (validated); the previous sliding
windows dropped each file's tail (157 h). Clips live on the released pose grid
(the NPZ frame count), because every per-frame annotation does. Because the grid
is in time, clip *k* of one member and clip *k* of the other cover the same
interval even when their frame rates differ (1,120 measured pairs mix frame rates).
Recordings with no pose array (476 zero-frame NPZs — the 439 behind missing
videos and 37 V01 2160x2160 files — and 189 missing bundles) have no clips.

## 4. How each annotation is computed

### 4.1 Arm, hand and head motion

The arm measures are the previous pipeline's, recomputed on the new grid for
every clip (`prior:reused`): on the 496,789 full-length 30-fps clips whose frames
coincide with an old window, all 22 reused columns reproduce the old values
exactly (float32 rounding only; report §3.3). Two ideas carry them:

* **The torso frame.** Wrist and elbow positions are expressed in a frame whose
  origin is the shoulder midpoint and whose axes follow the shoulder line and the
  spine (forward kinematics in NumPy over the 52-joint SMPL-H tree with the
  neutral model and zero betas, agreeing with `smplx` to under a micrometre).
  Swaying, leaning and turning move the frame, not the wrists in it, so
  whole-body motion is removed by a change of coordinates rather than a
  threshold. Root translation and orientation are not used by any motion measure.
* **Speed and travel.** A frame is *active* when the faster wrist exceeds
  60 mm/s **and** has travelled more than 35 mm within 0.5 s; gaps ≤ 0.25 s are
  bridged and runs < 0.3 s dropped (lengths in frames use one ceiling rule, so
  the definition no longer differs between 30 and 29.97 fps). Vibration in place
  has speed but no travel. `arm_active_frac` and the episode counts are
  descriptive definitions, not gates.

All millimetre values are on one canonical skeleton (betas = 0), so they compare
across participants by construction and do not measure body size.
`hand_artic_p75_rad_s` uses local finger rotations over frames whose hand pose is
not frozen (the release freezes hands when SMPL-H is marked invalid).
`head_speed_*` is the angular speed of the SMPL-H head relative to the upper
spine (agrees with Meta's `movement:alignment_head_rotation` at Spearman
0.86-0.88 on V00). Two noise diagnostics are kept as tracking-quality columns,
not as gates: `consistency_r` (SMPL-H vs 2D-keypoint wrist speed — two
independent channels) and `step_cosine_p50` (detector noise reverses direction,
motion continues). Every jitter-*magnitude* signal the previous work tried
correlated with activity at up to |rho| 0.9, which is why none is published as a
quality score.

### 4.2 Posture (sitting vs standing)

A protocol-driven variable: moderators encourage standing but allow sitting, so
posture varies by participant, by interaction and within an interaction.
**Observed values**: standing, sitting, and changes between them within a
recording (e.g. `V03_S1821_I00000010_P3624` stands, sits, perches, sits). No
lying, kneeling or crouching was seen in any sampled frame, so those are not
categories. The enum is `standing | sitting | mixed | unclear | unknown`.

**Per frame** (pelvis-frame FK joints): `hip_flexion` = the torso-thigh angle
(≈ 180° straight, ≈ 90° seated) and `knee_between` = how far down the hip→ankle
drop the knee sits along the torso axis (≈ 0.5 standing, 0.15-0.4 seated). Each
is only read on frames where its landmarks are *visible* in the released 2D
keypoints (confidence ≥ 0.5 **and** inside the raster — keypoints are
extrapolated beyond the picture, so confidence alone is not visibility): hips and
knees for the angle, knees and ankles for the ratio. Otherwise the fitted legs
are the prior's guess and the value is NA.

**Per 1-s bin** (median of observable frames, ≥ 50% observable), a rule chosen by
vendor because the rigs differ (standing V00 reads ~25° less hip angle than
standing V03):

| rule | vendors | sitting | standing | between | evidence |
|---|---|---|---|---|---|
| `v03_hip` | V03 (all rasters; angles are rotation-invariant) | hip < 136° | hip > 152° | unclear | 164 V03 file labels: no stander below 136.1°, no sitter above 151.9° |
| `v00_knee` | V00 | knee_between < 0.30 | ≥ 0.48 | unclear | 66 V00 file labels (selection-biased); hip angle unused on V00 |
| `unvalidated_hip_knee` | V01 square-pixel, V02 | never (unclear) | hip ≥ 135° and knee ≥ 0.48 (hip only when ankles are out of frame) | unclear | no labels; visual QA: all 24 sampled V02 clips its sitting branch had produced were standing people |
| `not_measurable_anamorphic` | V01 2160x2160, 1920x1080 | — | — | unknown | the released SMPL-H absorbed the stretch (the old detector read 93% "seated") |

The thresholds are the retired v0 seated-posture detector's (FM1), turned from a
single rejection cut into two edges with an **unclear band** between them.

**Aggregation** (per clip over its bins; per recording over all of its bins, not
from clip labels): `unknown` if fewer than half the bins (or < 3) are observed;
`unclear` if fewer than half of the observed bins are decided; `mixed` if a
sustained run (≥ 5 decided bins) of each posture occurs — the same criterion
counts `posture_transitions`; otherwise the majority posture when it has ≥ 80% of
decided bins, else `unclear`. `posture_confidence` is the share of bins that
support the label (NA for unclear/unknown) — evidence support, **not** P(correct).
The four `posture_*_frac` shares sum to 1 and are published so users can apply
their own cut.

**Validation.** On the 66 hand-labelled V00 files
(`configs/validation/v00_posture_labels.csv`), recording-level labels were
decided on 43 and right on 41; the 23 others fell in the unclear band, where the
two labelled groups overlap (0.39-0.41). No stander was labelled sitting. The
known miss is a sitter on a high stool with dangling legs (reads standing).
Bin- and clip-level accuracy was never labelled by the dataset's annotators
(`posture_rule_evidence`), so a **visual QA** of clip labels was run
(`configs/validation/posture_visual_qa_2026-09-24.csv`; one mid-clip keyframe per
clip, 12 random full-length clips per label × vendor): "standing" was right on 48 of
48 clips (12 per vendor), "sitting" on 12/12 V00, 11/12 V03 (one standing or perched)
and 1/1 V01. On V02 the unvalidated rule's "sitting" was wrong every time: 12 clips
decided by the hip angle alone (fits cropped at the knees read 112-120°) and a second
sample of 12 decided by hip and knee together were all standing people, so that rule
no longer decides "sitting" at all (V01 and V02 read only standing or unclear). V00
"unclear" clips were mostly standing (11 of 12): the band is deliberately conservative.
**Limitations**: perching on a high stool with legs extended reads as standing
("standing" = upright with legs extended); a sustained forward bend (≥ 5 s) lowers
the torso-thigh angle and can read as sitting under the V03 rule, so some "mixed"
clips are bends rather than sits.

### 4.3 Expressivity

One overall score plus subdimensions, built on the arm/hand/head measures and
checked against Meta's own signals.

* **Channels** (each a percentile within the clip's reference group):
  `expr_energy` (median wrist speed), `expr_amplitude` (wrist range and p90
  excursion), `expr_head` (head speed p75), `expr_hands` (finger articulation
  p75). Extra subdimensions, not in the score: `expr_variability` (coefficient of
  variation of 1-s wrist speed: bursty vs steady; scale-free), `expr_face`
  (facial-action-unit variability, V00 only), `expr_vocal` (p90-p10 level of own
  speech; clips with ≥ 3 s of own speech, any role). Face and voice stay out of the overall score
  because their coverage is partial — a score whose meaning changes by vendor is
  not "applied consistently".
* **Why these four.** The previous design's six channels double-counted: active
  fraction is a threshold on the same wrist speed (Spearman 0.9), and SD-of-speed
  tracks amplitude (0.88). The four kept are distinct signals (correlations are
  in the report).
* **Score.** `expressivity_score` = the mean of the four channel percentiles,
  re-ranked so it is uniform over the reference clips: 0.7 means more expressive
  than 70% of full-length measured clips of the same **reference group** (V00,
  V01, V02, V03 portrait, V03 room camera). Within-rig ranking is the default
  because rig and fit effects are real: V02 fits read 29-43% faster at matched
  speaking time. `expressivity_score_pooled` ranks against every vendor.
  `expressivity_level` / `_pooled`: low < 1/3 ≤ medium < 2/3 ≤ high.
* **Status and confidence.** `measured`, `too_short` (< 2 s), `pose_distorted`
  (severe anamorphic: the arm pose is wrong too), `no_subject`, `incomplete`.
  `expressivity_confidence` = min(1, clip length / L) × share of frames with both
  wrists in frame — evidence coverage, not P(correct).
* **Reference.** Quantile tables (1,001 points, mid-rank ties) per group are
  written to `expressivity_reference.json`, so scores are reproducible and new
  clips can be placed on the same scale.
* **Validation** (report §3.2, full run). Speaking clips score far higher than
  listening clips (median 0.69-0.71 vs 0.23-0.27 in every vendor). On V00 the score
  correlates with facial-action variability (Spearman 0.38; 0.31 controlling for
  speaking time) — two independent channels agreeing. The previous iteration's 90
  human gesture verdicts separate at AUC 0.58 (a weak proxy: they judged "good
  co-speech training data"). **Not supported**: clips containing an
  observer-annotated moment of interest are not more expressive than other clips of
  the same recordings (AUC 0.49 on V00, 0.55 on V03) — Meta's MOIs mark deviations in
  internal state that are mostly not body motion, so MOI presence is published as
  its own column and is not a subdimension. Seated clips score lower than standing
  ones on V00/V02 (medians 0.30/0.42 vs 0.51/0.52): condition on posture when that
  matters.

### 4.4 Speech, turn-taking and the partner

* **Own speech.** Meta's VAD; when a recording's VAD is empty but its transcript
  has timed words (673 of 1,760 zero-VAD files probed), the mask is built from the
  word spans (gaps ≤ 0.5 s bridged) and `speech_source = transcript_only`; when the
  VAD stops early (≥ 20 timed words start > 5 s after its last interval; ~1.8% of
  V03) the words' spans are added after it (`vad+transcript_tail`); with neither
  VAD nor words, `none` — silence and a missing annotation are then
  indistinguishable, so every speech-conditioned value is NA. Speech is annotated
  only as far as the WAV it was derived from (`speech_annotated_until_s`; 57
  measured recordings have a WAV > 1 s shorter than the pose grid). Meta's VAD also
  drops out mid-recording and resumes (1,220 recordings; e.g. 43 minutes of ~0 VAD
  under 40-200 transcript words/min at the speaker's own level). Every clip carries
  `speech_annotation_status` (`annotated`, `no_speech_annotation`, `beyond_audio`,
  `vad_gap` = ≥ 10 timed words at > 8 words per second of VAD speech); unless it is
  `annotated`, own speech and every speech-conditioned value is NA and
  `speaking_role` is `unknown` — never a false silence. (Filling such gaps from the
  transcript belongs in the scan's speech mask; it is an annotate-time guard until
  the next rescan.)
* **Speaking role** (per clip, own share *s*, partner share *p* from the linked
  partner clip): speaking if s ≥ 0.1 and s ≥ 2p; listening if p ≥ 0.1 and p ≥ 2s;
  silent if both < 0.1; else both; `unknown` when the partner clip is not linked
  or either member's speech source is `none`.
* **Dyad windows and interaction measures** (both members' speech masks at 10 Hz):
  per window, speech overlap (both / either), mutual silence and floor-holder
  changes, NA unless both member clips' speech is annotated and the recordings agree
  in length. The conversation's overlap, mutual silence, turn rate and speaking
  balance are exact aggregates over those windows (`interaction_pair_speech_coverage_frac`
  says how much of the conversation they cover), so a VAD gap or a short WAV in one
  member never reads as the pair's silence.
* **Partner links.** `partner_clip_id` = same interaction, same clip index, set
  only when both recordings are measured and agree in length within 1 s (no
  timebase drift): `partner_link_status` says why otherwise.

### 4.5 Audio

50-ms RMS levels of the participant's WAV (48 kHz float on the 16-bit grid):
the clip median level, own-speech level and its p90-p10 range. Per recording:
envelope dynamics (p95 − median; < 1.5 dB = no usable signal, the v0 FM4 rule, 13/13
on 48 labels), the share of exact-zero samples and of ticks at the digital floor
(both reflect listening time in the denoised audio, not dropouts), and
**voice isolation**: energy-mean level over own-only ticks minus over
partner-only ticks — how loudly the partner bleeds in (medians 16-23 dB on
close mics, 8.8 dB on the V03 room camera). `audio_quality`: `unusable` (missing,
58-byte or unreadable WAV), `dead`, `silent_during_own_speech` (own-speech ticks
at the digital floor: VAD says speaking, the track is silent), else `ok`.
Partial dropouts are not detected (see §7).

### 4.6 Face (V00 only)

Meta released Imitator features only for V00 (`has_imitator_movement` is exactly
the 42,932 V00 files). Per clip over valid frames (`movement:is_valid`, invalid in
60-frame processing blocks; 3 of 17 random V00 files have none): mean facial
action unit intensity and the mean per-AU SD (21 live AUs; AUs 16-18 are dead),
arousal and valence means and arousal SD. NA elsewhere, with
`face_features_status = not_provided`.

### 4.7 Framing, visibility and tracking quality

From the released 2D keypoints and box (visibility = confident **and** inside the
raster): the share of frames with shoulders, hips, knees, ankles and wrists in
view, `visible_extent` (the lowest landmark pair in view ≥ 80% of frames), box
height and edge contact (cropping), hand keypoint confidence. From SMPL-H: facing
angle (body forward axis vs the direction to the camera: 0 = facing it; typical
medians 3-6°; NA for severe anamorphic) and the reprojection error of the fitted
upper-body joints against the 2D keypoints, in shoulder widths (recording medians
0.094 on 1080x1920, 0.137 on V03 2160x3840 — partly a camera offset in the fit
rather than bad joints; compare within a raster).
`box_center_jump_max` / `flag_tracker_jump` catch a tracker that re-acquired or
switched subject. `quarter_turns` is detected per file from the shoulder-hip
axis (9 of the 1,405 3840x2160 files are already upright).

### 4.8 Moments of interest (Meta's 1P/3P annotations)

Every released entry is a row of `moi_events` (text as released). Per clip:
distinct 3P and 1P moments overlapping the clip, and the 3P seconds covered —
**NA when the recording is not annotated for that party, never 0**. Malformed
entries are labelled (593 zero-length entries, 36 negative-length entries = 12
moments, 5 entries = 1 moment ending past the recording). Annotations copied between the two members (same kind, same text,
start within 2 s) are linked (`partner_duplicate_moi_id`) and summarised per
interaction (`moi_duplication_3p`/`_1p`): found in V03 (7 of 36 1P pairs fully
copied, mostly session S0203) and 3 V00 pairs; a copied event's target person is
ambiguous.

### 4.9 Visual quality (the pixel pass)

One keyframe per clip (the keyframe at or before the midpoint; GOPs are 8.33 s),
made upright (crop, squeeze, turn) at 540 px short side: variance of the
Laplacian inside the subject box (sharpness; compare within a raster), mean luma
and clipped share (exposure), Canny edge density outside the box (background
clutter), and phase-correlation shift of the background against the previous
clip's frame (camera movement). `visual_status = not_run` until the pass has run.

### 4.10 Conversation-level and temporal properties

* Interaction rows hold values computed **from both members**: duration
  agreement, fps mismatch, speech status and pair speech measures, posture pair,
  any-member posture transition, mean and gap of the members' expressivity, MOI
  coverage and duplication, and the leakage counts below. A value that needs both
  members is NA when one is missing or unmeasured — never a one-sided value.
* Session rows hold relationship (Meta's, per session: 9 of 219 recurring dyads
  change it), dyad key and recurrence, and whether each participant's posture is
  consistent across the session's interactions. **No within-session order
  exists** in the release (interaction ids are prompt ids; MP4s carry no
  timestamps), so "posture changes across the session" is a consistency flag,
  not a sequence.
* Temporal: clip position in the recording, partial tail, whether an own-speech
  segment or motion episode is cut at either boundary, the recording's
  expressivity trend, and two redundancy measures — RMS difference (in corpus-SD
  units) of a clip's standardised feature vector from the previous clip and from
  the recording mean.

### 4.11 Splits and leakage

`split` is Meta's, as released, constant within every session and interaction.
Meta defines splits at the participant level, but 26 participants appear in two
splits (all span improvised and naturalistic), 25% of dev and 13% of test
interactions have a member who also has train recordings, and 7 of 51 A-suffix
id pairs sit in different splits. Nothing is re-split; instead
`participants.flag_split_conflict`, `flag_suffix_sibling_split_conflict`,
`interactions.n_members_in_other_split` and `n_members_suffix_sibling_other_split`
let a user drop leaky rows, and `participant_component_id` groups participants
connected through shared sessions for grouped cross-validation.

## 5. Meta's caveats, cross-referenced

Meta publishes **no per-interaction list** for any caveat, so no column is
"Meta's flag", and not flagged never means known unaffected.

| Meta caveat | what the tables carry | kind |
|---|---|---|
| ~10% of interactions with timestamp/prompt misalignment | `pair_duration_agreement` (member lengths disagree, or timebase drift); `timebase_status`/`timebase_drift_s`; `interaction_type_text_consistency` (9 prompts, 8,007 files, where the prompt text contradicts the type — consistent with the off-by-one prompt ordering Meta describes); `audio_quality = silent_during_own_speech`; `speech_source = vad+transcript_tail` and `speech_annotated_until_s` (VAD or audio shorter than the recording) | proxies (fresh, rule-inferred) |
| speaker bleed | `voice_isolation_db`, `recording_transcript_echo_frac`, `room_camera_rig`, window/interaction speech overlap | measured + rig class |
| participants leaving frame | `subject_present_frac`, `*_visible_frac`, `wrists_in_frame_frac`, `box_edge_contact_frac`, `visible_extent` | measured |
| duplicate / mismatched participant ids | `id_suffix`, `suffix_sibling_key`, `flag_suffix_sibling_split_conflict`, `metadata_status` | parsed / aggregate |
| MOI timing noise | `moi_malformed`, `partner_duplicate_moi_id`, `moi_duplication_*` | rule-inferred |
| recording-site variation | `vendor`, `raster_class`, `expressivity_reference_group` | given / parsed |

## 6. Missing values, confidence and evidence

* Every nullable column states, in the registry, the exact condition under which
  it is NA; the validator checks the most consequential ones (MOI counts, face
  measures, posture confidence, expressivity, pair measures, visual measures).
* Flags are non-null booleans; a condition that can be undecidable is an enum
  with an explicit value (`unknown`, `not_applicable`, `not_measured`) instead.
* Labels carry a confidence (posture, expressivity) defined as evidence support
  or coverage, stated as such; continuous measures name the columns that qualify
  them (`qualified_by`: e.g. motion → `smplh_valid_frac`, `wrists_in_frame_frac`).
* Each column's `evidence` says whether it is given, parsed, measured,
  rule-inferred or an aggregate, i.e. observed vs inferred.

## 7. Assumptions and deliberate exclusions

**Assumptions**

* The two recordings of an interaction share its active-time origin (durations
  agree within 0.1 s for 63,888 of the 64,116 measured pairs; a VAD-lag probe on 100 pairs
  centred at 0 s). Pairs that disagree by > 1 s are not linked.
* Millimetre measures are on the neutral SMPL-H skeleton (betas = 0).
* The clip length is 30 s (configurable; it is in every clip id).
* Percentile references are the corpus's full-length measured clips of each
  reference group.
* The released 2D keypoints are the visibility ground truth for posture gating.

**Deliberately not included**

* **Per-participant IPC**: Meta gives the IPC codes of prompt roles A and B but
  no mapping from participant to role; 91% of files have role-specific texts.
  Only prompts with identical A/B text are marked `ipc_member_assignable`.
* **Number of visible people / partner in view**: every file tracks one subject
  and sampled frames show only the participant; a person detector over 1.03 M
  frames was not justified by that evidence.
* **Face features for V01-V03**: Meta released none, and the 2D face landmarks
  are too quantised to substitute (mouth opening vs own VAD r ≈ 0.01-0.08).
* **Meta's occlusion-tracking / other movement_v4 features**: not in this
  release's NPZs. Gaze encodings, expression latents and hypernetwork features
  are opaque embeddings and are not summarised.
* **Within-session order and session-level trajectories**: no ordering signal
  exists.
* **Semantic topic** beyond the prompt template and a keyword task kind.
* **Prosody (F0), transcript sentiment, moderator identity**: not computed /
  not released.
* **Partial audio dropouts**: no validated detector (one known case).
* **A "good training data" score**: the previous `gesture_quality` encoded one
  project's inclusion decision and is not republished (see §8).

## 8. What was reused, adapted, replaced or removed

| previous iteration | now | why |
|---|---|---|
| M-1 inventory census (ffprobe + stat of every file) | reused as-is (`prior:reused`) | a neutral census of all 129,370 files |
| FK, torso frame, speed/travel, episodes, arm/hand measures | reused unchanged, recomputed on the new grid for every clip | physically interpretable; 22 columns reproduce the old values exactly on 30-fps files |
| wrist height, abduction, hands-together, pose spread | adapted: all frames (the old silent fallback from speech frames to all frames is gone; a speech-only variant is NA without speech) | one column held two quantities |
| hand articulation | adapted: non-frozen hand frames only | frozen hands read as zero articulation |
| episode median, sync lag | adapted: NA instead of 0 when undefined | 0 is a value |
| eligibility (8 exclusion reasons, first wins) | replaced by status/class columns (`raster_class`, `smplh_anamorphic`, `room_camera_rig`, `timebase_status`, `measurement_status`) | nothing is excluded; all reasons are recorded |
| tier-1 gates, tier-2 disqualifiers, `gesture_quality`, `clip_score`, 8-clips/12-files caps | removed | encoded one inclusion decision; §"Reconstructing" in using_the_annotations.md gives an approximate query and what it cannot reproduce |
| FM1 seated detector (git history only) | adapted into the posture label with an unclear band | was a rejection gate |
| FM4 dead-audio rule | adapted into `audio_quality = dead` + the continuous value | was a rejection rule |
| voice isolation | adapted to an energy-mean own-only vs partner-only difference | the released audio is bleed-suppressed; medians measure the noise floor |
| human gesture verdicts (686 files, 100 human) | carried to `legacy/gesture_review/` with the spans judged; used as weak validation | the only human motion labels |
| review app/cards/renderer, gallery, export, manifests | removed (restorable from the tag) | review of accept/reject decisions |

## 9. Testing

`PYTHONPATH=src:. python -m pytest`. The end-to-end test builds a miniature
release on disk (a normal pair with one seated member and one-sided MOI
annotations, a partner-missing interaction, a pair with an absent member, a pair
with mixed frame rates and mismatched lengths) and runs catalog → scan → annotate
→ validate through the real code, asserting pairing symmetry, partner status,
conversation-level values requiring both members, speaking roles, per-participant
posture, MOI NA-not-0 semantics, metadata missing states, and prefixed
propagation of conversation fields to every clip. The validator's fault-injection
tests corrupt a valid run one way at a time. Module tests cover each annotation
against synthetic inputs with known answers, and a registry test fails if the
field reference is stale.

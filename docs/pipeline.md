# The production pipeline, step by step

This document describes the pipeline **as implemented**. Every threshold quoted
here is the value in `configs/vibes_upper_body.yaml` or the dataclass default it
falls back to, and every count is from the live run. Where a step has a known
limitation or a failure mode it does not catch, that is stated in the step's own
section rather than collected out of sight at the end.

The pipeline is **fully automated**. Manual review is not a step in it. Review
tooling still exists and is documented in [§9](#9-development-only-review-tooling),
but it produces labels used to calibrate and validate the automated decision; the
pipeline runs to completion, and produces its training manifest, with an empty
verdict log. `test_the_production_manifest_needs_no_reviewer_at_all` asserts
exactly that.

---

## What the pipeline is deciding

One question, asked of every 30-second window of every participant recording:

> While this person is speaking, are their hands and arms making natural,
> visible co-speech gestures?

Technical cleanliness is **not** sufficient. A recording with flawless SMPL-H
tracking, clean audio and a participant whose hands rest in their lap for the
whole conversation is a *reject*, and catching that case is the pipeline's main
job. The five exclusions it is built around are:

| # | excluded | caught by |
|---|---|---|
| 1 | hands essentially static while speaking | §5 `static_while_speaking`, §6 `static_while_speaking`, `hands_parked_low` |
| 2 | too little visible upper-body movement to be gesturing | §5 `motion_too_small`, `one_posture_only`, `elbows_pinned`; §6 quality score |
| 3 | apparent motion is tracking noise or SMPL-H jitter | §4 travel requirement; §5 `motion_is_detector_noise`, `channels_disagree`; §6 same two at working thresholds |
| 4 | motion is global body movement, not arm articulation | §4 torso frame; §6 `motion_is_global` |
| 5 | a single brief adjustment or isolated transient | §5 `gesture_not_sustained`, `too_few_episodes`; §6 `episodes_too_brief`, `gesture_not_sustained` |

---

## Order of operations

```
   seamless_interaction/            the release, read-only, never modified
            |
   [1] population    eligibility from the M-1 inventory        -> population.parquet
            |
   [2] scan          per-frame measurement, Slurm array         -> scan_shards/*.parquet
            |
   [3] gather        concatenate shards, refuse gaps            -> windows.parquet
            |
   [4]               (measurement definitions: §4)
            |
   [5] select        tier-1 gates + clip selection              -> candidates.parquet
            |                                                      gate_funnel.csv
   [6] qualify       tier-2 disqualifiers + quality score       -> qualified_clips.parquet
            |                                                      qualification_funnel.csv
   [7] manifest      write the accepted subset                  -> accepted_clips.csv
            |                                                      reviewed_clips.csv (dev)
   [8] export        resolve paths, write the dataset card      -> export/
            |
   [9] verify        read sampled rows back out of the release  -> pass/fail
```

Each stage reads only the outputs of the ones before it, so any stage can be
re-run alone. Re-tuning a tier-2 threshold costs one `qualify` run over a
parquet file — seconds, no media read. Changing a *measurement* changes the scan
fingerprint and every shard recomputes.

---

## 1. Population — eligibility

**What it does.** Selects the participant files worth measuring at all, from the
M-1 inventory of 129,370 files.

**Why.** Measurement costs about a second per file. Excluding what cannot be used
before spending that is the only cheap filter in the pipeline. It also removes
conditions that would otherwise produce *misleading* measurements rather than
merely useless ones.

**Signals.** Container metadata and the released annotation JSON only. No pixels,
no pose.

**Rules.**

| rule | threshold | why |
|---|---|---|
| all four modalities present | `.npz`, `.json`, `.mp4`, `.wav` | a partial bundle cannot be measured or trained on |
| video stream present | — | some bundles carry audio only |
| raster format | exclude `2160x2160`, `1920x1080`, `640x480`, `3840x2160` | these are the formats whose camera geometry the release's fit handles badly; the stretch ends up in the pose itself |
| interaction type | exclude `charades` (reported as `no_speech_activity`) | a game with scripted physical actions, not conversation |
| duration | `>= 40 s` | shorter than one 30-second window plus a hop |
| probe status | `probe_status == "ok"` | the container could not be read reliably |
| frame rate | `20 <= nominal_fps <= 61` | a rate outside this range makes the frame/second conversions wrong |
| timebase drift | annotation grid vs `video_duration_s`, `<= 0.5 s` | a drifting timebase silently misaligns speech against pose |

There is **no VAD-based rule at this stage** — the population is built from the
inventory and the annotation's timebase only, and whether a participant actually
speaks is decided per window by `speech_seconds` in §5. The exclusion label
`no_speech_activity` is the `charades` rule above; it is named for the reason
rather than the mechanism, which is worth knowing when reading the counts.

**Outcome.** 118,570 of 129,370 files eligible. Exclusions, summing to 10,800:
6,741 `excluded_raster`, 2,955 `no_speech_activity` (all of them `charades`),
472 `too_short`, 439 `no_video_stream`, 189 `incomplete_bundle`, 4
`timebase_drift`. `probe_failed` and `unusable_frame_rate` exclude nothing on
this corpus — the first reason recorded per file is the one reported, and
neither is ever first.

**Relation to the goal.** Purely preparatory; it removes nothing on gesture
grounds.

**Limitations.** Raster exclusion is a proxy: it removes formats *known* to be
problematic rather than detecting the distortion directly, so a good recording in
a bad format is lost. The timebase check compares the annotation grid to the
video duration — an earlier version compared the container to its own audio
track, which made three of its four exclusions false positives.

---

## 2-3. Scan and gather — measurement

**What they do.** `scan` measures every eligible file frame by frame and reduces
it to 30-second windows at a 10-second hop; `gather` concatenates the shards.

**Why sharded.** 118,570 files as a 512-task Slurm array. `gather` refuses to
proceed if any shard is missing, so a partial scan cannot silently become a
smaller corpus.

**Signals.** `smplh:body_pose`, `smplh:left_hand_pose`, `smplh:right_hand_pose`,
`smplh:global_orient`, `smplh:is_valid`, `boxes_and_keypoints:keypoints` (the
released COCO-WholeBody-133 2D), and `metadata:vad` from the annotation JSON.

**Restartability.** A shard marker records a fingerprint that hashes the
measurement parameters **and the source of `gesture.py` and
`smplh_kinematics.py`**, plus the set of file ids the shard covered. So changing
a measurement invalidates every shard automatically, and a `--limit` smoke test
cannot make the real scan a no-op.

**Parameters.** Window 30.0 s, hop 10.0 s. 30 s is long enough for a gesture
pattern to show and short enough that a reviewer could watch one; the hop makes
neighbouring windows overlap, so a gesture-dense stretch is found wherever it
starts and selection vetoes the overlap later.

**Pass/fail.** None. This stage makes no decisions — it only measures. A file
that cannot be read at all is recorded as `gesture_status != "ok"` and fails
tier 1's first clause rather than being silently dropped here.

**Outcome.** 2,423,304 windows over 7,550 hours. Zero read errors.

**Limitations.** The hop means a gesture shorter than 10 s can fall between two
window centres and be diluted in both. Windows are also fixed-length, so a
40-second continuous gesture sequence is represented as two overlapping
30-second views rather than as one span; selection then keeps at most one of
them.

---

## 4. What is measured, and the two ideas that make it work

### 4.1 The torso frame — exclusion #4

All joint positions are expressed in a frame whose **origin is the shoulder
midpoint**, whose x-axis runs along the shoulder line, whose y-axis is the
component of pelvis-to-neck perpendicular to it, and whose z is their cross
product. Forward kinematics is pure NumPy over the 52-joint SMPL-H tree,
validated against `smplx` to under a micrometre.

**Why.** A participant who sways, turns, leans or is followed by a moving camera
produces large joint velocities in world coordinates while their arms do nothing.
In the torso frame those motions are, by construction, near zero. This is not a
threshold that can be tuned wrong — it is a change of coordinates, and it is the
primary defence against exclusion #4.

Two separate mechanisms, worth distinguishing:

- **Translation is not read at all.** Forward kinematics places the pelvis at
  the origin and `smplh:translation` is never consumed, so a moving camera or a
  participant walking across the room is invisible rather than thresholded.
- **Rotation and posture are removed by the frame.** A participant turning to
  face their partner changes every world-frame joint position; in the torso
  frame it changes almost nothing.

`test_global_body_movement_is_not_gesture` asserts the second by driving a 50°
whole-body yaw, showing the wrist genuinely travels over 100 mm in world
coordinates, and showing the pipeline measures under 1 mm of wrist range from
it. `test_pure_translation_is_invisible_by_construction` asserts the first.

### 4.2 Speed **and** travel — exclusion #3

A frame counts as *gesturing* only if the wrist has both:

- **speed** `>= 60 mm/s` (smoothed over 5 frames, central difference half-width 2), **and**
- **travel** `>= 35 mm` within a 0.5 s window.

**Why both.** Speed alone is satisfied by vibration. 8 mm of per-frame jitter at
30 fps is 240 mm/s — four times the speed floor, and fifteen times the 12–16 mm/s
resting floor measured on the most static real files in the corpus. Requiring
*displacement over half a second* means motion that does not go anywhere does not
count, however fast it is. `test_realistic_jitter_in_place_is_not_gesture` and
`test_tracking_jitter_is_not_mistaken_for_gesture` assert this.

### 4.3 Episodes — exclusion #5

Gesturing frames are merged into **episodes**: gaps shorter than 0.25 s are
closed, then runs shorter than 0.30 s are dropped. An episode counts as
co-speech if it overlaps *any* speech by at least 0.20 s. The 0.80 s minimum is
separate — it defines what counts as an **utterance** for
`speech_segments_covered`, so a very short vocalisation cannot be one of the
utterances a gesture is required to cover.

**Why.** It converts "how much motion" into "how many separate times, and for
how long each" — which is the distinction between gesturing and adjusting your
glasses once.

### 4.4 Measures produced

Per window, about 35 values. The ones the decisions read:

| measure | meaning |
|---|---|
| `gesture_frac_speech` | share of speaking frames that are gesturing |
| `gesture_seconds_speech` | absolute seconds of gesture during speech |
| `gesture_speech_ratio` | `(speech_rate + 0.02) / (silence_rate + 0.02)`; the smoothing stops a participant who is simply never still while silent from producing an unbounded ratio |
| `episode_count_speech`, `episode_median_s` | how many episodes, and how long each |
| `speech_segments_covered` | share of utterances containing an episode |
| `wrist_excursion_p90_mm`, `elbow_excursion_p90_mm` | p90 distance from the within-window median position |
| `wrist_range_mm` | bounding-box diagonal of wrist travel |
| `posture_spread_mm` | mean pairwise distance between wrist positions sampled 2.5 s apart, **max over the two hands** |
| `wrist_height_p75_mm` | height of the **higher** wrist above the shoulder midpoint, p75 over speech frames |
| `hands_together_frac` | share of speech frames with wrists within 180 mm |
| `arm_abduction_p75_deg` | angle of the **more abducted** upper arm from the torso axis |
| `arm_speed_speech_p50_mm_s`, `torso_travel_mm_s_p50` | median arm speed during speech; median torso translation speed |
| `step_cosine_p50` | median cosine between successive 2D displacement steps |
| `consistency_r` | correlation between SMPL-H arm speed and released-2D arm speed |
| `smplh_valid_frac`, `smplh_longest_invalid_s`, `hand_frozen_frac`, `kp_conf_p10`, `implausible_frac` | tracking integrity |
| `sync_r`, `sync_lag_s` | peak cross-correlation of the gesture-activity and speech envelopes, and its lag |

**Every activity and posture measure is the maximum over the two hands** — the
more mobile wrist, the higher wrist, the more abducted arm. That is why
one-handed gesturing is accepted;
`test_one_handed_gesture_scores_like_two_handed` fails if anyone changes a max
to a mean.

### 4.5 The one thing no measure may be

**No gate is a jitter-*magnitude* gate.** 106 of 135 candidate quality signals
measured on the dev corpus correlate with gesture activity at |rho| up to 0.903
— including every acceleration and high-pass-residual variant, and the one
designed specifically to suppress smooth motion. Gating on any of them is
arithmetically a gate on how much the participant gestured, which would delete
the most expressive people first.

The two noise guards used instead are outside that family:

- `consistency_r` compares **two independent measurements** of the same arm
  (SMPL-H-derived speed vs released 2D keypoint speed). Both can be noisy; they
  cannot be noisy in the same way by accident.
- `step_cosine_p50` looks at the **direction** of successive displacements.
  Detector noise jumps out and back (cosine near −1); real motion continues in a
  direction (positive). It is scale-free.

**Limitation.** `consistency_r` is only meaningful once there *is* motion — two
near-static channels correlate at chance. It is therefore applied after the
gesture clauses, never before.

---

## 5. Tier-1 gates — eligibility of a window

**What it does.** Eighteen clauses; a window must pass all of them to become a
candidate. The **first** failing clause is recorded as `fail_reason`, so the
histogram reads as a funnel.

**Why it is deliberately permissive.** These gates run before selection, over
2.4 M windows, and they cannot distinguish "modest gesturer" from "no gesturer"
without also deleting the modest gesturer. Their job is to remove what is
*unusable*, not to judge degree. Judging degree is §6.

### Tracking integrity

| clause | threshold | what it stops | fired |
|---|---:|---|---:|
| `smplh_valid_frac` | `>= 0.90` | mostly-untracked windows | 224,822 |
| `smplh_longest_invalid_s` | `<= 1.0 s` | a continuous tracking break, as opposed to scattered occluded frames | 73,168 |
| `hand_frozen_frac` | `<= 0.05` | hand-pose vectors bit-identical to the previous frame — the measured damage behind an invalid frame | 0 (36,087 violate) |
| `kp_conf_p10` | `>= 0.30` | the detector did not find the person | 2 |
| `implausible_frac` | `<= 0.002` | sustained wrist speed above 4 m/s | 195 |

Validity is gated on the *window*, not the file. An earlier version required
`smplh:is_valid` on every frame of the file and that single clause rejected
62.6% of V00 — more than every other check combined — for exactly the case the
brief rules out of scope: a hand briefly leaving the image.

**Read that last column carefully.** It counts windows whose *first* failing
clause is this one, which is what makes the table a funnel. It is not the number
of windows that violate the clause. `hand_frozen_frac` is the clearest case:
36,087 windows (1.5%) exceed 0.05, but every one of them already failed an
earlier clause, so the funnel attributes none to it. The clause is doing work;
it is just never the first thing wrong with a window. The same caveat applies to
every row — `channels_disagree` shows 10 because `consistency_r` is applied
last, not because only 10 windows disagree.

### Co-speech gesture

| clause | threshold | what it stops | fired |
|---|---:|---|---:|
| `speech_seconds` | `>= 8.0 s` | judging gesture where there is barely any speech | 801,414 |
| `gesture_frac_speech` | `>= 0.35` | hands still while the person talks | 318,252 |
| `speech_segments_covered` | `>= 0.40` | one brief adjustment scored as gesturing | 7,638 |
| `episode_count_speech` | `>= 3` | a single continuous sweep, however large | 66,332 |
| `wrist_excursion_p90_mm` | `>= 80 mm` | micro-motion and tracking wobble | 18,789 |
| `elbow_excursion_p90_mm` | `>= 35 mm` | wrist flicks with a pinned elbow | 41,077 |
| `gesture_speech_ratio` | `>= 1.05` | constant undirected fidgeting | 110,319 |
| `posture_spread_mm` | `>= 150 mm` | the same posture over and over | 232,593 |
| `wrist_height_p75_mm` | `>= -300 mm` | hands at the sides or parked at the waist | 40,341 |
| `hands_together_frac` | `<= 0.55` | clasped hands | 14,389 |
| `arm_abduction_p75_deg` | `>= 17 deg` | elbows pinned to the ribs | 38,344 |

The four posture-variety clauses (`posture_spread_mm` onward) were added after a
manual pass rejected 22 of 36 items for static hands that every *activity*
clause had passed. Hands clasped at the waist shuffle fast enough to satisfy any
speed rule while never leaving one place. Activity and variety are both
necessary and neither implies the other.

### Noise, applied last

| clause | threshold | fired |
|---|---:|---:|
| `consistency_r` | `>= 0.45` | 10 |
| `step_cosine_p50` | `>= -0.30` | 21,115 |

Both are applied **last**, because both are uninterpretable before the gesture
clauses have established that there is motion to measure: two near-static
channels correlate at chance, and a wrist that never moves produces no steps to
take a direction from. That ordering is why their funnel counts are small — not
because the clauses are inert. Over all 2.42 M windows, 940,302 violate
`step_cosine_p50 >= -0.30`; they simply fail something earlier first.

What *is* true is that these thresholds leave headroom. Among the windows that
survive to become candidates, `step_cosine_p50` has a 10th percentile of +0.06
and a median of +0.60 — the −0.30 floor is far below the surviving
distribution, so a much higher cut is available to a stage that runs after
selection. §6 uses it.

**Limitations.**

- These thresholds are **not calibrated against labels.** They were set from
  the corpus distribution and from the failure modes the v0 pipeline produced.
  The labelled data available was all drawn from files that had *already passed*
  these gates, so it can say nothing about what they wrongly reject. This is the
  single largest unvalidated surface in the pipeline.
- `hand_frozen_frac` fires zero times and is effectively inert here.
- `consistency_r` and `step_cosine_p50` are near-inert at tier-1 values; their
  working thresholds are in §6, and the reason they are loose here is that both
  are uninterpretable until the gesture clauses have established there is motion
  to measure.

**Outcome.** 414,504 of 2,423,304 windows qualify (17.1%).

### Selection

Qualifying windows are reduced to non-overlapping clips: greedy by `clip_score`,
overlapping windows vetoed, at most 8 clips per file and at most 12 files per
participant. The caps exist because one V00 participant appears in 221 files and
an unbalanced training set is a worse training set even when every clip is good.

**Outcome.** 73,883 candidate clips / 615.6 hours over 31,815 files and 3,724
participants.

---

## 6. Tier-2 qualification — the production decision

**What it does.** Decides, for each candidate clip, whether it is co-speech
gesture data. This is the step that replaced manual review.

**Signals.** Only the per-window measures already in `candidates.parquet` —
no media is read and no new measurement is computed, except `articulation_ratio`
which is derived from two existing columns. That is why re-tuning a tier-2
threshold costs seconds.

**Why it exists separately from §5.** §5 is a filter over 2.4 M windows that
must not be strict about degree. What a reviewer added on top was a judgement of
degree — these hands are up and working, those are technically moving but parked
in a lap. Measured on 100 human-labelled files that had **all already passed
every tier-1 gate**, that judgement turns out to be predictable from measurements
the scan already produces:

| measure | AUC (accept vs reject) | accept median | reject median |
|---|---:|---:|---:|
| `wrist_height_p75_mm` | 0.85 | −87 mm | −194 mm |
| `step_cosine_p50` | 0.81 | 0.66 | 0.39 |
| `arm_speed_p50_mm_s` | 0.75 | — | — |
| `wrist_range_mm` | 0.75 | 747 mm | 675 mm |
| `arm_abduction_p75_deg` | 0.74 | 31.6° | 26.4° |

So tier 2 is not new physics. It is the same measurements read at thresholds the
gates could not use.

### 6.1 Structure: disqualifiers, then a score

The decision is **not** a conjunction of tight thresholds. ANDing many strict
clauses multiplies the false-negative rate, and discarding genuine-but-subtle
gesturers is a failure mode in its own right. Instead:

**Disqualifiers** — each names a *pathology*, not a degree. Failing any one is
fatal and nothing compensates, because these are not "less gesture", they are
"not gesture".

| clause | threshold | exclusion | what it means |
|---|---:|:---:|---|
| `speech_seconds` | `>= 8.0 s` | — | not enough speech to judge |
| `gesture_frac_speech` | `>= 0.50` | #1 | hands static while speaking |
| `wrist_height_p75_mm` | `>= -260 mm` | #1 | hands parked in the lap or at the sides |
| `episode_count_speech` | `>= 3` | #5 | a single movement |
| `episode_median_s` | `>= 0.55 s` | #5 | a string of twitches rather than gestures |
| `speech_segments_covered` | `>= 0.60` | #5 | gesture not sustained across utterances |
| `articulation_ratio` | `>= 2.0` | #4 | motion mostly whole-body, not arms |
| `step_cosine_p50` | `>= 0.25` | #3 | tracker noise — direction, not magnitude |
| `consistency_r` | `>= 0.70` | #3 | the two channels disagree |

`articulation_ratio` is `arm_speed_speech_p50_mm_s / max(torso_travel_mm_s_p50, 1)`,
derived from columns the scan already writes. The denominator is floored at
1 mm/s so a genuinely still torso yields a large ratio and passes, rather than
dividing by zero.

**Composite quality score** — four dimensions, each the mean of piecewise-linear
ramps in physical units, combined by weight:

| dimension | weight | ramps (0 at → 1 at) |
|---|---:|---|
| posture | 0.40 | `wrist_height_p75_mm` (−300 → −40 mm); `wrist_range_mm` (540 → 860 mm); `arm_abduction_p75_deg` (18 → 45°); `posture_spread_mm` (175 → 315 mm) |
| persistence | 0.30 | `gesture_frac_speech` (0.45 → 0.92); `episode_median_s` (0.55 → 3.0 s) |
| vigour | 0.10 | `arm_speed_speech_p50_mm_s` (95 → 310 mm/s) |
| integrity | 0.20 | `step_cosine_p50` (0.10 → 0.80) |

A clip must reach **`gesture_quality >= 0.34`**.

Ramp endpoints are near the candidate pool's 10th and 90th percentiles, so 0.5
means "typical of the pool" rather than "half of some arbitrary maximum". They
are **not** fitted per-measure; only the final threshold is calibrated, which is
what keeps 224 labels from being over-fitted by 8 free parameters.

**Where 0.34 comes from.** Not an optimum — the labelled sets cannot resolve
one, and two differently-sampled sets give two different F1 optima (0.32 and
0.00). It is the **left edge of a plateau**: on the representative labelled
sample, precision (0.936), recall (0.948) and specificity (0.615) are identical
for every threshold from 0.34 to 0.40, so 0.34 is the smallest value delivering
the full measurable quality gain, and therefore the one that keeps the most data
(420.9 h against 396.3 h at 0.40) and the highest recall. The rule is stated in
[`../reports/18_automated_qualification.md`](../reports/18_automated_qualification.md)
§4.3, and the full trade-off curve is §4.1 there.

### 6.2 Why this shape protects against false negatives

Because the score is a weighted mean and not a conjunction, a restrained speaker
who keeps their hands up, works them through every utterance and whose motion is
cleanly coherent clears the bar without ever producing a large movement.
`test_subtle_but_genuine_gesture_is_not_discarded` builds exactly that case — a
swing less than half the clear case — and asserts it is accepted, that its
posture dimension is genuinely below 0.75, and that persistence and integrity
carry it.

`vigour` is weighted lowest (0.10) for the same reason: speed is the dimension
most confounded with personality.

### 6.2b Why the disqualifiers cannot be folded into the score

It is tempting to simplify: drop the nine disqualifiers and let the score carry
everything. One constructed case shows why that fails. The
single-brief-adjustment fixture — one 0.8 s movement in 30 seconds — scores
`gesture_quality = 0.419`, **above the 0.34 accept threshold**. A single
emphatic movement looks good on posture and vigour, and a weighted mean has no
way to see that all of it happened at once.

What rejects that clip is `too_few_episodes`, `gesture_not_sustained` and
`static_while_speaking`. The disqualifiers exist precisely because some failures
are structural rather than a matter of degree, and a mean cannot represent
structure. `test_a_single_brief_adjustment_is_not_gesturing` asserts both halves
of this — that the clip is rejected, *and* that the score alone would have
accepted it — so the simplification fails a test rather than quietly readmitting
the failure mode.

### 6.3 One measure deliberately excluded

`wrist_excursion_p90_mm` separates the labelled set **backwards**: rejects
median 339 mm against accepts 319 mm. Peak excursion rewards exactly the failure
the brief names — one big isolated adjustment. The clearest single case in the
corpus is `V02_S5281_I00000280_P5272`: 91% gesture-in-speech, 328 mm posture
spread, top-decile on every activity measure, and ten of twelve sampled moments
show his arms hanging at his sides. The entire score comes from twice adjusting
his beanie.

Peak amplitude is therefore **not** treated as evidence of gesturing.
`test_peak_excursion_is_excluded_from_the_score` asserts it stays out.

### 6.4 What a clip carries out of this step

Every clip — kept or dropped — carries `gesture_quality`, the four `dim_*`
scores, `articulation_ratio`, an `exclusion_flags` string listing **every** clause it
failed, and `fail_stage` naming the first. Flags are not short-circuited:
diagnosing a threshold needs the whole picture, and the funnel view needs the
first. So a downstream reader can re-threshold without re-running anything:

```python
stricter = manifest[manifest.gesture_quality > 0.6]
```

### 6.5 Outcome and measured accuracy

50,516 of 73,883 candidate clips qualify (68.4%) — **420.9 hours** over 24,114
files and 3,501 participants.

**Measured against 90 held-out human labels** — files reviewed by hand with
audio, all of which had already passed tier 1, so this is tier 2's own accuracy:

| | accept every tier-1 candidate | + disqualifiers only | + quality score (**shipped**) |
|---|---:|---:|---:|
| precision | 0.856 | 0.904 | **0.936** |
| recall | 1.000 | 0.974 | **0.948** |
| specificity | 0.000 | 0.385 | **0.615** |
| accuracy | 0.856 | 0.889 | **0.900** |

Both tier-2 layers carry roughly equal weight: the disqualifiers take precision
from 0.856 to 0.904 and specificity from 0 to 0.385, and the score takes them
the rest of the way to 0.936 and 0.615. Neither alone would do.

Bootstrap 95% intervals: precision [0.875, 0.987], recall [0.895, 0.988]. On a
second, deliberately boundary-enriched label set (134 items, sampled evenly
across all 64 strata cells) the same configuration gives precision 0.854 and
recall 0.800 — the score is weaker on hard cases, and the gold figures should
not be read as accuracy on difficult material.

First-failing-clause funnel:

| clause | clips |
|---|---:|
| `too_little_speech` | 0 |
| `static_while_speaking` | 4,958 |
| `hands_parked_low` | 4,516 |
| `too_few_episodes` | 0 |
| `episodes_too_brief` | 256 |
| `gesture_not_sustained` | 437 |
| `motion_is_global` | 684 |
| `motion_is_detector_noise` | 9,254 |
| `channels_disagree` | 146 |
| `below_quality_threshold` | 3,116 |
| **qualified** | **50,516** |

`too_little_speech` and `too_few_episodes` count zero because tier 1 already
enforces them at the same or a stricter value. They are kept as explicit
restatements of the criteria — the clause list is meant to be readable as the
full set of conditions — but they are not doing work.

**Limitations.**

- Thresholds are **calibrated against a labelled sample, not derived from first
  principles.** They are meant to be moved with evidence.
- The labelled set is biased toward what tier 1 passes, because that is all that
  was ever shown to a reviewer. Tier 2's accuracy is therefore only
  characterised *within* the tier-1-passing population — which is the only
  population it is ever applied to, but it means the two tiers cannot be
  re-balanced against each other without new labels.
- `sync_r` is measured and reported but **gates nothing**: no labelled
  comparison exists to calibrate it against. Its distribution is sensible
  (p50 0.33), which is not evidence.
- `posture_spread_mm` uses the **mean** pairwise distance, not the median. The
  beanie case in §6.3 is the failure mode: ten samples in one posture and two in
  another give a high mean even though ten of twelve moments are identical. The
  median would score it near zero and is the obvious next measure; it needs a
  re-scan.

---

## 7. Manifest

**What it does.** Writes `accepted_clips.csv` — one row per qualifying clip, a
contiguous `[start_frame, end_frame)` range of one participant file, carrying
identity, frame range, every measure a gate or qualifier reads, and every
tier-2 score. It is a curated subset of the ~35 measured columns, not all of
them; `qualified_clips.parquet` holds the complete set for every candidate,
kept or dropped.

**Why it is a manifest and not a copy.** The release is read-only and 40 TB. A
frame-range list costs kilobytes, never diverges from the source, and can be
re-filtered without re-exporting.

**Pass/fail.** A clip appears iff `qualified` is true. No other condition, and
in particular no verdict: moving `review_verdicts.jsonl` aside and re-running
produces a byte-identical file.

Also written, when a verdict log exists: `reviewed_clips.csv` and
`accepted_clips_with_audio.csv` (§9), and an **agreement report** comparing the
automated decision against every verdict it can join to. That comparison is a
measurement in the summary, not an assumption, so a regression in tier 2 appears
as a number.

**Outcome.** `accepted_clips.csv`: 50,516 clips / 420.9 hours / 24,114 files /
3,501 participants. `reviewed_clips.csv`: 1,030 clips / 8.6 hours.

**Limitations.** The manifest is only as good as the source tree it points into;
it carries no checksum of the NPZ files, so a corrupted or re-released source
would not be detected here. That is what §8's `verify` is for, and it samples
rather than checking every row.

---

## 8. Export and verify

**What they do.** `export` resolves the four source paths per row (`.npz`,
`.wav`, `.mp4`, `.json`, all relative to `source_root`), writes both tiers as
CSV, and generates `export/DATASET.md` — a card readable by someone who will
never run the pipeline. `verify` samples manifest rows and reads them back out
of the release.

**Why.** A manifest is a promise about a tree it does not own. `export` makes
the promise usable without knowing the release layout;
`seamless_curation.dataset.load_clip` turns a row into pose, hands, audio and
clip-relative VAD, reading only the frames the row names. `verify` is what
stops the promise from silently going stale.

**Signals.** `export` reads the manifests and `manifest_summary.json` (for the
agreement figures printed in the card). `verify` reads the **source tree** and
nothing else.

**Criteria.** `verify` checks, per sampled row: the NPZ slice is exactly the
promised number of frames; the upper-body block is `(n, 13, 3)`; and speech
seconds recomputed from the released VAD match the manifest to within 0.5 s.

**Pass/fail.** Any row failing any check is reported in `failures`; the command
reports the list rather than raising, so one bad row does not hide the rest.

**Outcome.** At the current manifest, `verify --sample 80` checks 80 rows and
reports zero failures across 45,641,520 upper-body pose frames.

**Limitations.** `verify` samples. It establishes that the manifest and the
source agree where checked, not everywhere. It also cannot detect a source file
that was replaced with a *different but equally well-formed* recording — there
is no content hash of the release.

---

## 9. Development-only review tooling

`render`, `review`, `queue` and `import-verdicts` are **not production stages**.
They exist to produce labelled data:

- `render` draws a review card per file — twelve sampled moments with keypoints
  overlaid, the pelvis-frame SMPL-H pose at each, and a timeline of own speech,
  partner speech, arm speed, gesture episodes and accepted spans.
- `review` serves those cards on `127.0.0.1` with the 30-second clip and audio.
- Verdicts append to a JSONL log, keyed by a content-derived `review_item_id`.

Their output is used for three things, all of them development: calibrating
tier-2 thresholds, validating the automated decision, and the
`reviewed_clips.csv` subset. That subset is 17% human by row; the rest is
model review, so it is a labelled set rather than a human-signed one.

**They are never required.** The production manifest is written whether or not
the verdict log exists, and no verdict can add a clip to it or remove one — a
property with its own tests (`test_a_verdict_cannot_change_the_production_manifest`).

The review rubric is [`review_rubric.md`](review_rubric.md); it is the text the
tier-2 clauses were derived from, kept so the translation from qualitative
criterion to measured clause can be audited.

---

## 10. Testing

`tests/test_automated_qualification.py` is the acceptance suite: each of the
five exclusions is built as a synthetic bundle whose motion is exactly that
failure, run through real measurement and both gate tiers, and asserted to be
rejected — **with the right reason**. The positive cases are asserted in the
same file, because a filter that rejects everything satisfies every exclusion
test ever written.

| test | case | expected |
|---|---|---|
| `test_clear_co_speech_gesture_is_accepted` | 55° episodic two-handed | accept |
| `test_one_handed_gesture_is_accepted` | one arm gesturing, one parked | accept |
| `test_subtle_but_genuine_gesture_is_not_discarded` | 22°, sustained, coherent | **accept** |
| `test_static_hands_while_speaking_is_rejected` | clean tracking, no arm motion | reject |
| `test_tracking_jitter_is_not_mistaken_for_gesture` | 16 mm/frame noise | reject |
| `test_global_body_movement_is_not_gesture` | 50° whole-body yaw, rigid arms | reject, and torso-frame range < 1 mm against >100 mm world travel |
| `test_pure_translation_is_invisible_by_construction` | 400 mm translation | measurements bit-identical with and without it |
| `test_a_single_brief_adjustment_is_not_gesturing` | one 0.8 s movement | reject |
| `test_motion_unrelated_to_speech_is_rejected` | large motion, only in silence | reject |

Synthetic fixtures are used so the suite runs without the 40 TB mount; the
measures are geometric, so a bundle can be *constructed* with a known answer.
One fixture detail matters: hand parameters carry a 1e-4 rad tremor, because a
real fit never produces bit-identical consecutive frames and a synthetic pose
resting at exactly zero would trip `hand_frozen_frac` spuriously.

The suite also pins properties of the rule itself: ramps are clipped and
monotone; a non-finite measurement fails its clause rather than passing;
`exclusion_flags` lists every failure while `fail_stage` names the first; the funnel
accounts for every clip and reports every clause including the ones that never
fire.

**Known gap.** The synthetic fixtures establish that each failure mode is caught
and that subtlety survives; they do not establish the *thresholds*. Those are
calibrated on real labelled data and reported in
[`../reports/18_automated_qualification.md`](../reports/18_automated_qualification.md).

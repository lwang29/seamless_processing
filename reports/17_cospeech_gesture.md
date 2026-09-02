# Round 17 — co-speech upper-body gesture, and a manually verified subset

Date: 2026-09-01. Supersedes every detector in
[`v0_measurement_rounds/`](v0_measurement_rounds/README.md); that directory's
`README` says which of its findings are still load-bearing.

This round answers one question the previous fifteen did not:

> whether the person actually makes meaningful hand and arm gestures while
> speaking […] we need co-speech upper-body motion […] please also make sure
> that small tracking jitter, global body movement, or a single brief hand
> adjustment is not mistakenly counted as meaningful gesturing.

Every number below is produced by `seamless-curation stats` from the run's own
artefacts, or is stated here with the command that produced it. Where something
is a choice rather than a measurement it says so.

---

## 1. What the v0 pipeline was actually doing

Three measurements, all from the v0 scans, explain why its output did not
satisfy the review.

**The gesture check was not about speech.** `FM3` expressed each COCO wrist
relative to the shoulder midpoint, took each wrist's own **whole-file** median as
its reference, and flagged a file when **both** wrists sat within 0.10 shoulder
widths of those references for at least 75% of frames. It never read the VAD.
Over 4,000 V00 files its Spearman correlation with the participant's speaking
fraction is **−0.378**, and its flag rate falls monotonically from 24.0% in the
least talkative bin to 2.0% in the most. It was, in effect, a detector for quiet
listeners.

Its two constants have no fit behind them. `0.10` first appears hard-coded in a
Round-2 proxy and `0.75` as a dataclass default; no report in the repository
contains a confusion matrix, precision, recall or AUC for FM3, against 66 stored
labels for FM1 and 48 for FM4.

**It let static-handed clips through, by construction.** Among the 1,134 files
that passed the entire v0 pipeline in the last fresh sample, **27.6%** had both
wrists parked within 0.10 shoulder widths of their own median for at least half
the recording, and the maximum `fm3_static_frac` among keeps was 0.7485 — the
cut let everything below it through untouched. Because the rule requires *both*
wrists parked simultaneously, a participant who rests one arm and moves the
other never trips it: 47.2% of keeps had at least one wrist whose median offset
was under 0.10 shoulder widths.

And because it is whole-file, a recording that gestures for one minute and is
motionless for the next five passes on the average.

**The reviewer was mostly not shown speech.** The v0 review clip was placed by a
seeded uniform hash over the recording. Measured against the released VAD on the
246 windows of the last briefing manifest, **44 (17.9%) contain zero seconds of
the participant's own speech** and 109 (44.3%) contain under 25%; the median own-
speech coverage is 0.336. Roughly half the evidence behind "these clips show
almost no hand movement while the participant is talking" was clips in which the
participant was not talking.

**What rejected the most data was not a gesture check at all.** On the last
V00-only scan the pass rate was 31.18%, and `FM2` — "the released
`smplh:is_valid` flag holds on **every** frame of the file" — rejected **62.58%**
on its own, against 7.12% for static hands and 2.15% for seated posture.

---

## 2. What replaced it

### 2.1 The unit is a window, not a file

Gesture and speech are both bursty. Over 5-second dev spans, 44.7% contain no
speech at all against 6.9% of whole files, and more than half have both wrists
within 100 mm of their own median for the entire span. A whole-file verdict
averages those together.

The scan therefore measures every eligible file on sliding **30-second windows
at a 10-second hop**, and every gate is a statement about a window.

### 2.2 Everything is measured in the torso frame

Origin at the shoulder midpoint; axes from the shoulder line and the
pelvis-to-neck axis; `global_orient` dropped. A participant who sways, turns,
steps or leans moves the frame with them, so **global body motion cannot be
counted as gesture** — asserted in `tests/test_gesture.py`, where translating the
whole body by 400 mm changes no gesture measure by more than 1e-9.

Because all sixteen SMPL-H betas are zero, the skeleton is metrically identical
in every file and millimetre thresholds compare directly across participants
with no normalisation.

### 2.3 Four traps, four named guards

| trap | guard |
|---|---|
| tracking jitter counted as gesture | a frame is active only if the wrist has **travelled** — a bounding-box displacement over 0.5 s — not merely moved fast |
| noise large enough to travel | two *independent* channels: `consistency_r` (SMPL-H arm vs. released 2D arm) and `step_cosine_p50` (successive 2D displacement directions) |
| global body movement | the torso frame, above |
| one brief adjustment | `speech_segments_covered` — the share of the window's utterances (VAD ≥ 0.8 s) containing a gesture episode |
| motion unrelated to speaking | every activity measure computed separately over speech and non-speech frames |

**No gate is a jitter-magnitude gate**, and that is deliberate. 106 of 135
candidate quality signals measured in Session 2 correlate with gesture activity
at |rho| up to 0.903 — including every acceleration and high-pass-residual
variant and the one built specifically to suppress smooth motion. Any of them
used as a quality gate would preferentially delete the most animated speakers.
The two guards used here are outside that family: one compares two measurements
of the same arm, the other looks at direction rather than magnitude.

`step_cosine_p50` is worth naming. Detector noise in this release is dominated
by single-frame jump-out-and-back spikes: among consecutive large steps the
median cosine between successive displacements is **−0.91 to −0.999** on still
hands and **positive (+0.32 to +0.86)** on moving ones.

Both noise guards are applied **after** the gesture clauses, because two
near-static channels correlate at chance and a hand that never moves produces no
large steps to take a direction from. Reading them on a static clip says nothing.

### 2.4 Activity is not variety

The first pass of manual review rejected 22 of 36 items for static hands, and
the notes said the same thing every time: *the same posture in ten of the twelve
moments, only the fingers change*. Hands clasped at the waist satisfy every
speed, travel, episode and excursion clause while never leaving one place, and
none of those measures separated the rejects from the accepts.

Four measures were added that name it directly, and they are the round's main
methodological result:

- **`posture_spread_mm`** — mean distance between wrist positions sampled 2.5 s
  apart during speech. That is the spacing of the review card's twelve
  thumbnails, so the measure and the reviewer make the same comparison.
  Shuffling inside one posture scores near zero however fast the shuffling is.
- **`wrist_height_p75_mm`** — height of the higher wrist above the shoulder
  midpoint. Hands at the sides sit near −550 mm and clasped at the waist near
  −350 mm.
- **`hands_together_frac`** — share of speech frames with the wrists within
  180 mm of each other.
- **`arm_abduction_p75_deg`** — upper arm away from the torso axis. Elbows
  pinned to the ribs cannot make a co-speech gesture, and this says so without
  reference to how fast anything moved.

### 2.5 What no longer rejects anything

- **Seated posture, and legs out of frame.** `FM1` is retired outright. ViBES
  trains the upper body. `tests/test_pipeline_stages.py` asserts that no word
  from that family has reappeared in the eligibility rules.
- **Whole-file SMPL-H validity.** Replaced by a per-window allowance
  (`min_smplh_valid_frac` 0.90) plus a tighter limit on any *continuous* invalid
  stretch (1.0 s), plus `hand_frozen_frac` — the measured damage behind an
  invalid frame, since on invalid frames roughly 40% of hand-pose vectors are
  bit-identical to the previous frame while body pose is unaffected.
- **A hand briefly leaving the image.** Nothing tests in-frame geometry.
- **Dead audio.** Superseded structurally: a file with no voice has an empty
  released VAD, so it cannot reach `min_speech_seconds`.

### 2.6 Vendors

All four, not V00 alone. The v0 restriction rested largely on `FM1`, which
rejected a quarter of V03 on posture, and on the quarter-turned V03 rasters —
whose defect is *global orientation*, which does not enter root-relative
upper-body pose at all. What genuinely cannot be used is still excluded by
raster: V01's anamorphic 2160x2160 and 1920x1080 storage, where the isotropic
fitted camera forced the stretch into the pose itself, and V03's 640x480 and
3840x2160 room cameras, whose far-field audio makes co-speech alignment
meaningless.

---

## 3. Results

The whole corpus, scanned once. 118,570 eligible files, 7,550 participant-hours,
2,423,304 windows, **zero read errors**. The scan took about two and a half
hours of wall time on a 512-task array at one CPU each; the same measurement
through `smplx` would have been a 300-CPU-hour job.

### 3.1 Population

129,370 participant files, 118,570 eligible.

| ineligible | files | why |
|---|---:|---|
| `excluded_raster` | 6,741 | V01 anamorphic, V03 room cameras |
| `no_speech_activity` | 2,955 | `charades` — gestural but speech-free |
| `too_short` | 472 | under 40 s, cannot hold a window |
| `no_video_stream` | 439 | probe found no video |
| `incomplete_bundle` | 189 | a modality missing |
| `timebase_drift` | 4 | annotation grid misses the container |

### 3.2 Gate funnel

First failing clause per window, in clause order.

| stage | windows | share |
|---|---:|---:|
| `smplh_invalid` | 224,822 | 9.3% |
| `smplh_invalid_run` | 73,168 | 3.0% |
| `keypoints_missing` | 2 | 0.0% |
| `implausible_speed` | 195 | 0.0% |
| `too_little_speech` | 801,414 | 33.1% |
| `static_while_speaking` | 318,252 | 13.1% |
| `gesture_not_sustained` | 7,638 | 0.3% |
| `too_few_episodes` | 66,332 | 2.7% |
| `motion_too_small` | 18,789 | 0.8% |
| `wrist_only_motion` | 41,077 | 1.7% |
| `motion_not_speech_linked` | 110,319 | 4.6% |
| `one_posture_only` | 232,593 | 9.6% |
| `hands_below_gesture_space` | 40,341 | 1.7% |
| `hands_clasped` | 14,389 | 0.6% |
| `elbows_pinned` | 38,344 | 1.6% |
| `channels_disagree` | 10 | 0.0% |
| `motion_is_detector_noise` | 21,115 | 0.9% |
| **qualifies** | **414,504** | **17.1%** |

Three things in that table are worth reading twice.

**`too_little_speech` is the largest cut, at a third of all windows.** That is
not a gesture judgement; it is the measurement saying that a third of a
conversational corpus, sliced into uniform thirty-second windows, is one person
listening. The v0 review clip was placed uniformly over the recording and
therefore drew from exactly this population 33% of the time.

**The clauses aimed at the PI's complaint reject 27.1% between them**:
`static_while_speaking` 13.1%, `one_posture_only` 9.6%, `too_few_episodes`
2.7%, `wrist_only_motion` 1.7%. `one_posture_only` — the newest clause, and the
one no activity measure could substitute for — is the second largest of these.

**The two noise guards reject 0.9% and 10 windows respectively**, which looks
like nothing until you condition correctly. They are applied last, and among the
windows that reach them `consistency_r` has a 1st percentile of 0.78 against a
0.45 cut: real motion appears in both the SMPL-H and the released 2D channel
essentially always. `step_cosine_p50` is the interesting one — its median over
*all* windows is −0.04, and its median over windows that have already passed the
gesture clauses is **+0.53**. The measure separates the two populations by half
a unit of cosine, and the 0.9% it rejects are windows whose "motion" is
antiparallel single-frame detector noise.

### 3.3 Candidates

| vendor | condition | review items | candidate hours |
|---|---|---:|---:|
| V00 | improvised | 4,177 | 100 |
| V00 | naturalistic | 6,520 | 117 |
| V01 | improvised | 2,005 | 47 |
| V01 | naturalistic | 994 | 15 |
| V02 | naturalistic | 4,916 | 74 |
| V03 | improvised | 1,078 | 25 |
| V03 | naturalistic | 12,125 | 238 |
| **all** | | **31,815** | **616** |

**73,883 candidate clips, 615.6 hours, over 31,815 files and 3,724
participants** — an average of 69.7 seconds of candidate data per file, and
about 8.5 files per participant after the twelve-file cap.

Two comparisons put that in scale. The v0 pipeline's V00-only keep set was
31.18% of files with no gesture criterion at all; this is 17.1% of *windows*
across four vendors with eleven. And 616 hours is the size of the pool the
review draws from — not the size of the accepted set, which is bounded by how
much of it anyone looks at.

---

## 4. The manual pass

The candidate pool is many times larger than any review budget, so the accepted
subset is bounded by review throughput rather than by data. Two consequences
were designed for rather than worked around.

**The review unit is the file.** Its card samples the frames that would actually
be accepted, and its verdict applies to all of that file's clips at once, so one
review yields around a hundred seconds of accepted data instead of thirty.

**The queue is a stratified round robin over participants**, not a ranking by
score, so reviewing a prefix gives a set spread across people, vendors and
conditions — and the accepted subset does not inherit the score's biases.

Everything about how a verdict was reached is a column, not a claim:
`reviewer`, `verdict_source` (`human` or a model identifier), and
`review_evidence` (`card` or `card+video`). `accepted_clips_with_audio.csv` is
the subset whose reviewer played the clip with sound.

### 4.1 What the manual pass catches that the measures cannot

One item makes the case better than any argument. `V02_S5281_I00000280_P5272`
reaches the review queue with as strong a set of numbers as anything in the
corpus: **gesture-in-speech 91%, utterances covered 100%, posture spread
328 mm** — above the 90th percentile — smplh valid 95%, hands together 9%.

Ten of its twelve moments are both arms hanging straight at his sides. The
entire 328 mm of posture spread comes from the other two, in which he reaches
up and adjusts his beanie.

Every clause fired correctly. `posture_spread_mm` asked whether the wrists visit
different places and they do; nothing in the measurement can ask *why*. The
reviewer wrote: *"the only large-amplitude motion is twice reaching up to adjust
his beanie, so the 328 mm posture spread comes from grooming rather than
gesture"*, and rejected it as `static_hands` and `not_co_speech`.

That is the whole reason the manifest is gated on looking, and it is why the
automated stage is documented as narrowing the pool rather than choosing the
set.

### 4.2 Who did this pass, and what that is worth

**The verdicts in this run were produced by vision model reviewers**, recorded
as `verdict_source: model:claude-opus-5`, each working from
[`docs/review_rubric.md`](../docs/review_rubric.md) — the same text a human reviewer
gets — and each writing a one-sentence note citing what it saw. That is stated
plainly rather than described as "manual review", because the distinction
matters and the manifest carries it in a column.

What it is: a genuine per-item visual inspection of the frames that would enter
the training set, against a written rubric, with an auditable justification for
every verdict. The notes are specific — *"only two postures in twelve moments,
arms hanging at the sides for 1.2–11.2 s then held together at waist level with
elbows pinned and only the fingers rearranging"* — and the pass found two real
defects that no automated measure had (§4.3).

What it is not: hearing the audio. A model reviewer works from the card, so
audio–motion synchronisation is checked against the released VAD drawn on the
timeline — own-speech bars against gesture episodes against the arm-speed trace
— and against the measured `sync_r`, not by ear. Every such verdict is recorded
with `saw_video: false`, and `accepted_clips_with_audio.csv` therefore contains
only what a human has since confirmed with sound through the review app. That
file being small, or empty, is the honest state of the audio check and not an
oversight.

The intended use is that a human works down the same queue in the app,
confirming or overturning. Nothing special is needed for that: the log is
append-only and last-write-wins, so a human verdict simply supersedes the model
one on the items they reach, `contested` marks where the two disagreed, and the
manifest summary reports the mix of sources. The model pass is a floor under the
subset, not a substitute for the PI's eye.

### 4.3 What review and adversarial code review found

**Looking at cards found a rendering bug.** Two reviewers recorded
`tracking_broken` against files whose tracking was fine. The card's upper-body
crop had been clamped to the raster edge without re-deriving the other side, so
a wide gesture produced a 1080x1848 crop that ffmpeg squashed into a 250x200
panel; the skeleton, drawn with a single isotropic scale, then landed a body
length below the participant. Here the automated measures were right and the
*artefact* was wrong — the mirror image of the beanie case in §4.1, and only
looking found either.

**Looking at cards also found the missing measure.** Twenty-two of the first
thirty-six items were rejected for static hands that every activity clause had
passed, and the notes were unanimous: *the same posture in ten of the twelve
moments, only the fingers change*. That is what the four posture-variety clauses
in §2.4 exist for. They did not come from a distribution; they came from reading
reject notes.

**An adversarial code review of the new pipeline confirmed twenty defects**,
five of them serious enough to corrupt the output rather than merely annoy:

| defect | what it did |
|---|---|
| `step_cosine` computed on a per-frame splice of the two hand tracks | the hands sit a shoulder width apart, so each change of which hand was "faster" injected a displacement ~100x the real per-frame step. On one measured window it turned a `step_cosine_p50` of −0.48 (detector noise, correctly rejected) into −0.08 (accepted). |
| `review_item_id` was a positional rank | re-running `select` after any change to the candidate set re-bound every stored verdict to a different participant, putting files nobody had looked at into the accepted manifest under the name of the reviewer who accepted their former neighbour. It is now a hash of `(vendor, file_id)`, the verdict records `file_id`, and the manifest asserts the two agree. |
| shard reuse ignored *which files* the shard covered | a `--limit 2` smoke test made the subsequent full scan a complete no-op, and `gather` then reported two files of six as the whole corpus. |
| finger articulation read *global* joint rotations | a rigid, motionless hand on a swinging arm reported 2.3 rad/s of "finger articulation"; the correct local reading is exactly 0. |
| the review page did not guard key auto-repeat | a held `A` wrote four verdicts against one item and scrolled the next three past the reviewer with none. |

Every one has a test in `tests/test_regressions.py` naming the wrong behaviour
it produced. Two changes came out of the exercise that are worth more than the
individual fixes: the scan fingerprint now hashes the *source* of the
measurement modules rather than a version number someone has to remember to
bump, and the shard marker records its membership — so neither class of silent
staleness can recur.

---

## 5. The accepted subset

Live numbers are in `outputs/vibes_upper_body_v1/manifest_summary.json`, which
`seamless-curation manifest` regenerates from the verdict log. This is the
snapshot at the time of writing, and it will only grow: every figure here is a
function of how many items have been looked at.

| | |
|---|---:|
| review items judged | 552 of 31,815 |
| accept / reject / unsure | 375 / 151 / 26 |
| accept rate | 67.9% |
| **accepted clips** | **848** |
| **accepted hours** | **7.07** |
| accepted files | 375 |
| accepted participants | 375 |
| contested items | 0 |
| verdicts with no candidate | 0 |

One file per participant so far, which is the stratified round robin working as
intended: 552 items in, 552 different people.

**Accept rate by vendor and condition.** It is flat — 0.66 to 0.79 across seven
strata — and that is the useful finding: the gates are not systematically kinder
to one vendor or condition, so the accepted subset inherits the candidate pool's
composition rather than a gate artefact.

| vendor | condition | judged | accept rate |
|---|---|---:|---:|
| V00 | improvised | 66 | 0.70 |
| V00 | naturalistic | 145 | 0.66 |
| V01 | improvised | 41 | 0.78 |
| V01 | naturalistic | 23 | 0.74 |
| V02 | naturalistic | 76 | 0.67 |
| V03 | improvised | 14 | 0.79 |
| V03 | naturalistic | 187 | 0.66 |

**Why items were rejected.** `static_hands` is 80% of the reject reasons given,
and with `not_co_speech` it is 95%. That is the answer to the PI's question:
after eleven automated gesture clauses, what a reviewer still throws out is
overwhelmingly the failure he named, and nothing else comes close.

| reason | items |
|---|---:|
| `static_hands` | 141 |
| `not_co_speech` | 26 |
| `obscured` | 8 |
| `tracking_broken` | 1 |
| `unnatural_motion` | 0 |
| `out_of_sync` | 0 |

`tracking_broken` at 1 in 552 is worth reading against the first review round,
where it was 8 in 36 — every one of those was the card's crop bug (§4.3), not
the data.

**The exchange rate.** 46.1 seconds of accepted data per review item judged, or
67.8 s per item accepted, at 2.26 clips per accepted file. That is the number to
plan with: the candidate pool holds 616 hours, so accepted hours grow
essentially linearly with review effort, and reviewing all 31,815 items would
yield roughly **407 hours**.

`accepted_clips_with_audio.csv` is empty, correctly: no verdict in this pass was
taken with sound. See §4.2.

---

## 6. What is open

- **Coverage of the candidate pool.** 552 of 31,815 review items have been
  judged, so 1.7% of it. The accept rate is flat across vendors and conditions
  (§5), so the remainder projects at about 0.68 and the pool at about 407 hours
  — but that is a projection from 1.7%, and the way to shrink its error bar is
  to review more, not to argue about it.
- **`posture_spread_mm` uses the mean pairwise distance, not the median.** The
  beanie case in §4.1 is the failure mode: ten samples in one posture and two in
  another give a high *mean* pairwise distance even though ten of twelve moments
  are identical. The median pairwise distance would score that near zero and is
  the obvious next measure to try — but it would need a rescan, and the case for
  it should come from the reject histogram of a larger review pass rather than
  from one example.
- **`sync_r`.** Measured, reported, and gating nothing: no labelled comparison
  exists. Its distribution is sensible (p50 0.33) but that is not evidence.
- **Partner audio.** Only the participant's own microphone is muxed, so a
  reviewer cannot hear the turn structure.
- **Window length.** 30 s was chosen so a clip can be watched; whether ViBES
  wants longer contiguous spans with the loader doing the cutting is untested.

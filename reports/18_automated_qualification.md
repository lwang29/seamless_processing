# Round 18 — replacing manual review with an automated decision

Date: 2026-09-21. Supersedes §5–6 of
[`17_cospeech_gesture.md`](17_cospeech_gesture.md); §1–4 of that report (the
measurement design) remain current.

The brief: remove manual review — human or model — from the production
pipeline, and convert the review criteria into automated flags, thresholds and
validation checks. Model review may still be used in development to identify
failure modes and refine criteria, but must not be a required step.

This report is the evidence for the replacement: what the reviewer's judgement
turned out to be predictable from, how the thresholds were set, what the
resulting decision measures against held-out human labels, and — at some
length — what this evidence does **not** support.

---

## 1. Why a second tier, rather than tightening the gates

The tier-1 gates cannot do this job. They run over 2,423,304 windows *before*
selection, and their thresholds are constrained by a measured fact: 106 of 135
candidate quality signals correlate with gesture activity at |rho| up to 0.903.
A gate strict enough to exclude a marginal gesturer is arithmetically a gate on
how much the participant gestured, and it deletes the expressive people first.

So tier 1 answers "is this window usable" and stays wide. 414,504 windows
(17.1%) pass it, becoming 73,883 candidate clips over 615.7 hours.

What a reviewer added on top was a judgement of **degree** — these hands are up
and working, those are technically moving but parked in a lap. The question this
round had to answer was whether that judgement is recoverable from measurement.

---

## 2. It is, and the strongest signal was already being computed

100 files were reviewed by hand, with audio, against
[`../docs/review_rubric.md`](../docs/review_rubric.md). **Every one of them had
already passed every tier-1 gate**, so any separation found among them is
exactly the separation tier 1 could not achieve.

Ranking every measure by how well it separates the reviewer's accept from their
reject (Mann-Whitney AUC; 80 accepts, 13 rejects):

| measure | AUC | p | accept median | reject median |
|---|---:|---:|---:|---:|
| `wrist_height_p75_mm` | 0.850 | 5.6e-05 | −87 mm | −194 mm |
| `step_cosine_p50` | 0.812 | 3.2e-04 | 0.66 | 0.39 |
| `arm_speed_p50_mm_s` | 0.752 | 3.8e-03 | — | — |
| `wrist_range_mm` | 0.751 | 3.9e-03 | 747 mm | 675 mm |
| `arm_abduction_p75_deg` | 0.740 | 5.7e-03 | 31.6° | 26.4° |
| `arm_speed_speech_p50_mm_s` | 0.738 | 6.3e-03 | — | — |
| `gesture_seconds_speech` | 0.727 | 9.1e-03 | — | — |

Two things stand out.

**`wrist_height_p75_mm` is the single best predictor of a human verdict**, and
the tier-1 gate on it sits at −300 mm, *below* the candidate pool's own 10th
percentile (−242 mm). It was gating almost nothing. "Where are the hands
carried" turns out to be most of what a reviewer means by "are they gesturing".

**`step_cosine_p50` is second**, and its tier-1 gate is −0.30 against a pool
10th percentile of **+0.08** — it never bound at all. The noise guard existed,
was correct, and was doing no work. Both of these are used at working
thresholds in tier 2.

### 2.1 One measure separates backwards

`wrist_excursion_p90_mm` has *negative* separation: rejects median 339 mm
against accepts 319 mm. Peak excursion rewards one large isolated movement,
which is precisely the failure mode the brief names. It is therefore **excluded
from the quality score**, and `test_peak_excursion_is_excluded_from_the_score`
keeps it out.

The clearest case in the corpus is `V02_S5281_I00000280_P5272`: 91%
gesture-in-speech, 328 mm posture spread, top-decile on every activity measure.
Ten of twelve sampled moments show his arms hanging at his sides; the entire
score comes from twice adjusting his beanie.

---

## 3. The shape of the decision, and why it is not a conjunction

Nine **disqualifiers**, each naming a pathology rather than a degree, plus a
**composite score** over four dimensions. Full specification in
[`../docs/pipeline.md` §6](../docs/pipeline.md#6-tier-2-qualification--the-production-decision).

The structure is the false-negative safeguard. ANDing nine strict thresholds
multiplies the false-negative rate; a weighted mean lets strength on several
dimensions compensate for modesty on one, so a restrained speaker who keeps
their hands up and works them through every utterance passes without ever
producing a large movement. `vigour` carries the lowest weight (0.10) because
speed is the dimension most confounded with personality rather than with
gesturing.

Ramp endpoints sit near the candidate pool's 10th and 90th percentiles, so 0.5
means "typical of the pool". They are **not** fitted per-measure. Only one
number is chosen from data — the final threshold — which is what keeps eight
free parameters from being fitted to 93 labels.

---

## 4. Setting the threshold: what the data can and cannot do

Two labelled sets, deliberately sampled differently:

- **Gold** — 90 files reviewed by hand with audio (77 accept, 13 reject), drawn
  from the review queue, which is a stratified round-robin over participants.
  Closest thing available to the natural distribution of tier-1-passing files.
  Verdicts with a `policy:` source are excluded: they were converted by a rubric
  decision rather than observed, and folding them in would let a rule we wrote
  grade its own homework.
- **Dev** — 134 files labelled by two independent model reviewers who agreed
  (110 accept, 24 reject), from a 601-item sample drawn **evenly across all 64
  cells** of a 4x4x4 stratification on the three strongest predictors. This is
  deliberately **boundary-enriched**: it over-represents hard cases and is not a
  population estimate. It is a stress test. Inter-reviewer agreement was 93.1%.

The two sets do not overlap.

### 4.1 The trade-off curve

| threshold | gold precision | gold recall | gold specificity | rejects caught | dev precision | dev recall | hours | clips |
|---:|---:|---:|---:|:---:|---:|---:|---:|---:|
| 0.00 | 0.904 | 0.974 | 0.385 | 5/13 | 0.850 | 0.827 | 446.9 | 53,632 |
| 0.24 | 0.904 | 0.974 | 0.385 | 5/13 | 0.850 | 0.827 | 443.1 | 53,171 |
| 0.30 | 0.914 | 0.961 | 0.462 | 6/13 | 0.849 | 0.818 | 433.0 | 51,966 |
| **0.34** | **0.936** | **0.948** | **0.615** | **8/13** | **0.854** | **0.800** | **420.9** | **50,516** |
| 0.40 | 0.936 | 0.948 | 0.615 | 8/13 | 0.860 | 0.782 | 394.7 | 47,365 |
| 0.44 | 0.947 | 0.935 | 0.692 | 9/13 | 0.862 | 0.736 | 370.5 | 44,458 |
| 0.48 | 0.945 | 0.896 | 0.692 | 9/13 | 0.870 | 0.727 | 341.6 | 40,993 |
| 0.54 | 0.956 | 0.844 | 0.769 | 10/13 | 0.861 | 0.618 | 293.5 | 35,223 |
| 0.60 | 0.950 | 0.740 | 0.769 | 10/13 | 0.846 | 0.500 | 239.2 | 28,711 |
| 0.66 | 0.979 | 0.610 | 0.923 | 12/13 | 0.841 | 0.336 | 183.2 | 21,991 |

Monotone and well-behaved on both sets, with no cliff. That is itself a result:
the threshold is a **policy dial**, not a discovered constant.

The two sets agree on direction and disagree on magnitude, exactly as their
sampling designs predict. On the naturally-distributed gold set the score buys a
lot (precision 0.904 to 0.936, specificity 0.385 to 0.615, for 2.6 points of
recall). On the boundary-enriched dev set it buys much less (precision 0.850 to
0.854) and costs more recall (0.827 to 0.800). **On hard cases the score is
weaker.** Anyone reading the gold figures as the accuracy on difficult material
would be over-reading them.

### 4.2 Neither set can pick the point on the curve

- In-sample F1 on gold peaks at **0.32**. Under leave-one-out — re-choosing the
  threshold on the other 89 items for each held-out item — F1 drops by **0.025**,
  and the thresholds LOO selects range from **0.00 to 0.44**. The criterion is
  not stable on this sample.
- On dev, F1 is maximised at **0.00**: with 110 accepts against 24 rejects,
  recall dominates F1 and the score's precision gain cannot pay for it.
- Bootstrap 95% intervals at 0.34 are wide: precision 0.936 [0.875, 0.987],
  recall 0.948 [0.895, 0.988].
- With 13 gold rejections, **specificity moves in steps of 1/13 = 0.077**.

Two differently-sampled label sets give two different F1 optima (0.32 and 0.00),
which is the clearest possible demonstration that F1 optimisation is not
identifying anything real here.

### 4.3 So it is set by a stated rule

> **The threshold is the smallest value at which the precision and specificity
> gains on the representative sample have fully saturated.** That is
> `min_gesture_quality = 0.34`.

On gold, precision (0.936), recall (0.948) and specificity (0.615) are
**identical for every threshold from 0.34 to 0.40** — a plateau. Taking its left
edge delivers the entire measurable quality benefit while keeping the most data
(420.9 h against 396.3 h at 0.40) and the highest recall. It also happens to be
the better choice on the stress set: dev recall is 0.800 at 0.34 against 0.782
at 0.40 and 0.736 at 0.44.

This is a rule, applied to a plateau the data *can* resolve, rather than an
optimum it cannot. Moving it is one line of `configs/vibes_upper_body.yaml` plus
a `qualify` re-run that reads no media.

---

## 5. What the decision measures

At the shipped configuration, against the 90 held-out gold labels:

| | value |
|---|---:|
| precision | **0.936** |
| recall | **0.948** |
| specificity | 0.615 (8 of 13 rejects caught) |
| accuracy | 0.900 |

The baseline matters. Accepting every tier-1 candidate gives precision 0.904,
because tier 1 has already removed 83% of windows. **Tier 2 raises precision
from 0.904 to 0.936 and specificity from 0.385 to 0.615, at a cost of 2.6 points
of recall.** It removes about a third of the residual bad material while keeping
95% of the good.

On the boundary-enriched dev set the same configuration gives precision 0.854
and recall 0.800. Both numbers are true of different populations; the gold
figures describe the corpus the filter actually runs on, the dev figures
describe how it behaves when every case is hard.

Output: **50,516 clips / 420.9 hours** over 24,114 files and 3,501 participants
— 68.4% of candidate clips.

Agreement against the 569 model-labelled items is lower (precision 0.771,
specificity 0.272) and should **not** be read as the automated decision doing
worse. Those labels come from model reviewers measured to be ~13 points too
strict — zero false accepts but only 87% recall against the same human. In many
of those disagreements the automated decision is the one that is right.

---

## 6. Clause ablation: seven of nine disqualifiers are unmeasurable *here*

Removing each disqualifier in turn, threshold held at 0.34:

| clause removed | gold precision | gold recall | gold specificity | hours | delta precision |
|---|---:|---:|---:|---:|---:|
| *(none — full rule)* | 0.936 | 0.948 | 0.615 | 420.9 | — |
| `hands_parked_low` | 0.912 | 0.948 | 0.462 | 446.4 | **−0.023** |
| `motion_is_detector_noise` | 0.926 | 0.974 | 0.538 | 465.4 | **−0.010** |
| `channels_disagree` | 0.936 | 0.948 | 0.615 | 422.1 | 0.000 |
| `episodes_too_brief` | 0.936 | 0.948 | 0.615 | 421.9 | 0.000 |
| `gesture_not_sustained` | 0.936 | 0.948 | 0.615 | 422.5 | 0.000 |
| `motion_is_global` | 0.936 | 0.948 | 0.615 | 423.1 | 0.000 |
| `static_while_speaking` | 0.936 | 0.948 | 0.615 | 424.5 | 0.000 |
| `too_few_episodes` | 0.936 | 0.948 | 0.615 | 420.9 | 0.000 |
| `too_little_speech` | 0.936 | 0.948 | 0.615 | 420.9 | 0.000 |

Only two clauses have a measurable effect on 90 labels — and they are the two
§2 predicted from the AUC ranking: where the hands are carried, and whether the
motion is directionally coherent.

This admits two readings and only one is supported.

**It does not mean the other seven do nothing.** Corpus-wide they exclude real
clips: `motion_is_detector_noise` 9,254, `static_while_speaking` 4,958,
`hands_parked_low` 4,516, `gesture_not_sustained` 437, `motion_is_global` 684,
`episodes_too_brief` 256, `channels_disagree` 146. They simply do not change the
verdict on any of 90 particular files, which is unsurprising when the rarest
fires on 0.1% of the corpus — the expected number of gold items affected is
below one.

**And the constructed cases show at least one of them is load-bearing.** The
single-brief-adjustment fixture scores `gesture_quality = 0.419`, which is
*above* the 0.34 accept threshold. The composite score would accept it: one
emphatic movement looks good on posture and vigour, and a weighted mean cannot
see that everything happened at once. What rejects that clip is
`too_few_episodes`, `gesture_not_sustained` and `static_while_speaking` — three
of the clauses the gold ablation calls inert.
`test_a_single_brief_adjustment_is_not_gesturing` asserts this explicitly,
including that the score alone would have let it through, so that folding the
disqualifiers into the score "for simplicity" fails a test rather than silently
readmitting the failure mode.

**What is true is that seven thresholds are uncalibrated.** They were set from
distributional reasoning and the rubric, not fitted, and neither labelled set
can confirm or refute them. `too_little_speech` and `too_few_episodes` are
strictly redundant with tier 1 at these values and fire zero times on the
corpus; they are kept because the clause list is meant to read as the complete
set of conditions, and a criterion enforced upstream is better restated than
silently omitted.

---

## 7. Validation that does not depend on the labels

Threshold calibration is only half the evidence, and the weaker half. The other
half is constructed: for each failure mode in the brief, a synthetic recording
whose motion **is** that failure, run through real forward kinematics, real
measurement and both gate tiers.

| constructed case | result | asserted by |
|---|---|---|
| clean tracking, arms never move | rejected `static_while_speaking` | `test_static_hands_while_speaking_is_rejected` |
| 16 mm/frame jitter (480 mm/s, 8× the speed floor) | rejected; `step_cosine_p50 < 0` | `test_tracking_jitter_is_not_mistaken_for_gesture` |
| 50° whole-body yaw, arms rigid | rejected; torso-frame range <1 mm against >100 mm world travel | `test_global_body_movement_is_not_gesture` |
| 400 mm pure translation | measurements bit-identical with and without | `test_pure_translation_is_invisible_by_construction` |
| one 0.8 s movement then stillness | rejected | `test_a_single_brief_adjustment_is_not_gesturing` |
| large sustained motion, only while silent | rejected; `gesture_frac_speech < 0.3` | `test_motion_unrelated_to_speech_is_rejected` |
| 55° episodic two-handed gesturing | **accepted**, quality > 0.9 | `test_clear_co_speech_gesture_is_accepted` |
| one arm gesturing, one parked in the lap | **accepted** | `test_one_handed_gesture_is_accepted` |
| 22° — modest amplitude, sustained, coherent | **accepted**, posture dim < 0.75 | `test_subtle_but_genuine_gesture_is_not_discarded` |

The last row is the one that keeps this from degenerating. A filter that rejects
everything satisfies every exclusion test ever written; the subtle case is the
assertion that the false-negative safeguard is real rather than claimed.

**One of these tests was vacuous and an adversarial audit caught it.** The
global-motion case originally drove `smplh:translation` — a field the pipeline
never reads, because forward kinematics places the pelvis at the origin. The
bundle therefore contained no motion the pipeline could see, and the test passed
because the arms were rigid, not because global motion was excluded. It would
have gone on passing if the torso frame had been deleted.

The general lesson is that asserting *the outcome* of a rejection proves
nothing when the trivial case produces the same outcome: every negative fixture
here is rejected, so "rejected" is uninformative unless the test also shows the
input was non-trivial. The replacement drives a 50° whole-body yaw, asserts the
wrist genuinely travels over 100 mm in world coordinates, and asserts the
pipeline measures under 1 mm of wrist range from it — outcome *and* mechanism.

Two fixture details were found the hard way and are worth recording. A single
uninterrupted 30-second sweep is **one episode** and is correctly rejected by the
episode clauses — realistic co-speech gesture had to be modelled as bursts gated
to the speech segments. And synthetic hand poses resting at exactly zero are
*bit-identical* between frames, which trips `hand_frozen_frac`; a 1e-4 rad
tremor (about a hundredth of a millimetre at the fingertip) was added, because a
real fit never produces bit-identical consecutive frames and that is exactly why
the clause is a tracking-failure detector.

---

## 8. What this evidence does not support

- **A precise accuracy figure.** 93 labels, 13 of them rejections. The bootstrap
  interval on precision is [0.875, 0.987] and on recall [0.895, 0.988].
  Quoting 0.936 to three digits is reporting the point estimate, not the
  uncertainty.
- **The threshold being optimal.** It sits on a plateau the data cannot resolve
  (§4.2). It is a policy choice, documented as one.
- **Seven of the nine disqualifier thresholds** (§6). Uncalibrated, retained on
  reasoning.
- **Any claim about speech-gesture synchronisation.** `sync_r` and `sync_lag_s`
  are measured and carried in the manifest but gate nothing, because no labelled
  comparison exists to set a threshold against. Their distribution is sensible
  (p50 0.33), which is not evidence. The pipeline's co-speech guarantee is
  *co-occurrence* — gesture inside the participant's own VAD, spread across
  utterances — not phase alignment.
- **Anything outside the tier-1-passing population.** Every label was taken on a
  file that had already passed tier 1, because that is all a reviewer was ever
  shown. Tier 2 is characterised only where it is applied, which is sound, but
  it means the two tiers cannot be re-balanced against each other without new
  labels drawn from the tier-1 *rejects*.
- **`posture_spread_mm` as a variety measure.** It uses the mean pairwise
  distance. Ten samples in one posture and two in another give a high mean; the
  median would score it near zero. This is the residual of the beanie case and
  the obvious next measure, and it needs a re-scan.

---

## 9. Development review tooling after this change

`render`, `review`, `queue` and `import-verdicts` remain, and are marked
*development* throughout the documentation and the CLI help. They produce the
labelled data this report rests on. They are not production stages, and this is
checked three ways:

- `test_the_production_manifest_needs_no_reviewer_at_all` — empty verdict log,
  full output.
- `test_a_verdict_cannot_change_the_production_manifest` — a reject does not
  remove a qualifying clip, and no verdict column appears in the file.
- **On the real corpus**: moving `review_verdicts.jsonl` aside and re-running
  `manifest` produced a byte-identical `accepted_clips.csv` (md5
  `56dc79bfb05173dc6eba36afa8f7659f`, 50,516 clips / 420.9 h), with the summary
  reporting `reviewed_items: 0` and no agreement block. Restoring the log
  reproduced the same md5 again.

The existing 686 verdicts are retained: 100 human with audio, 552 from model
reviewers. `reviewed_clips.csv` (1,030 clips / 8.6 h) is still written for anyone
who wants per-clip human sign-off, and the agreement table in §5 is regenerated
into `run_report.md` §8 on every `stats` run, so a regression in tier 2 surfaces
as a number rather than as a surprise downstream.

---

## 10. What would most improve this

In order of value per unit of effort:

1. **More labels, drawn near the decision boundary.** The binding constraint on
   every number in this report is 93 gold items. A few hundred would let the
   threshold be selected rather than declared, and would calibrate the seven
   clauses §6 cannot see.
2. **Labels from tier-1 rejects.** Would let the two tiers be balanced against
   each other rather than treated as fixed.
3. **Median pairwise posture spread** (§8), which needs a re-scan.
4. **A labelled synchronisation comparison**, which would let `sync_r` gate
   something instead of being carried along.

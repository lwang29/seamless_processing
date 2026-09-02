# V00 failure-mode detection, Round 4 — FM1 to FM3

Supersedes `reports/03_failure_mode_detection.md`, which is deleted. Round 3's
FM1 and FM2 were both wrong in ways your inspection exposed; FM4 is gone at your
instruction; FM3 is untouched. Same 4,000 V00 files, re-measured, so every
before/after number below is paired rather than two draws compared.

Scan: `sbatch slurm/v00_fm_scan.sbatch` (job 16556066), 64 tasks, 536–655 s each,
mean 613 s. Analysis: `$VIBES scripts/v00_fm_analyze.py`. Gallery:
`$VIBES scripts/sample_v00_fm_gallery.py` then
`sbatch slurm/render_v00_fm_array.sbatch`.

---

## 1. What you asked for, and what happened

| you said | outcome |
|---|---|
| Scrap FM4 entirely | Gone from code, config, tests, docs, and the gallery. `mutual_silence`, `merge_intervals`, and `Thresholds.mutual_silence_s` are deleted; a test asserts they cannot come back. The audio strip's own/partner VAD stays — that was never FM4. |
| Leave FM3 alone | Untouched. Same measure, same 0.75 / 0.10 shoulder-width calibration. |
| FM1 is wrong in both directions | Rebuilt. All 29 of your labels now come out right: 10/10 correctly-flagged sitters kept, **15/15 false positives cleared**, **4/4 false negatives caught**. |
| FM2 misses a body part leaving frame | Rebuilt to test the body **surface** and cross-check the released 2D keypoints. Your example now flags. Flag rate rises 77.5% → **81.1%**, as you anticipated and accepted. |
| Drop Panel A′ | Removed, not defaulted off. |
| Filmstrip endpoints | First thumbnail is frame 0, last is the final frame, ten evenly spaced between. |
| 30-second clips | Done; clip length is now a real setting rather than a hard-coded 10. |
| Explain the S/B/M dots | §7. |

**One thing you said turned out to be right with a different scope**, and it
changed the design — see §2.4. A participant's posture really is constant, but
across a *recording session*, not across everything they ever recorded.

---

## 2. FM1 — sitting. Rebuilt.

### 2.1 Why the old detector failed, mechanically

`smplx` is called with **all sixteen betas pinned to zero**. The consequence is
measurable and exact: the forward-kinematic thigh is **376.79 mm** and the shin
**400.58 mm** in every file, byte-identical, whoever was recorded. So FK output
is *pure pose* and carries no information about body build.

The 2D keypoint ratios are the opposite. `leg_over_torso` is a ratio of image
limb lengths, so it encodes build directly. Your fifteen false positives are all
short-legged standing people — their 2D hip-to-ankle distance in shoulder widths
runs 0.964–1.241 against a corpus median of 1.764, while their torsos are
ordinary. `leg_over_torso < 0.85` was therefore measuring **body proportion, not
posture**, which is why its precision on your inspection was **0 of 15**.

`knee_spread > 0.90` failed the other way. All ten of its correct hits were
correct, but the four sitters you found in the pass group read 0.034, 0.431,
0.482 and 0.640 — one of them *lower* than most standing people. It cannot
generalise. Both 2D criteria are retired: still measured, still in the
distributions, never thresholded.

### 2.2 The new primary: where the knee sits along the torso axis

```
u = unit(neck − pelvis)
knee_between_torso = median over frames of  (knee − pelvis)·(−u) ⁄ (ankle − pelvis)·(−u)
                     averaged over left and right per frame
```

How far down the hip-to-ankle drop the knee sits, measured along the
participant's **own** torso axis rather than the image's vertical. Standing puts
the knee near the midpoint; sitting lifts it toward hip height.

| | your 14 sitting labels | your 15 standing labels |
|---|---|---|
| `knee_between_torso` | 0.152 – 0.406 | 0.508 – 0.613 |
| `hip_flexion_deg` | 88.2 – 113.8 | 119.9 – 141.6 |

Being taken along the torso axis makes it invariant to camera tilt and distance;
being pure FK makes it invariant to build, which is the exact failure it has to
survive. **Neither quantity entered any Round-3 selection criterion**, so their
separation on your labels is not an artefact of how those clips were chosen.

FM1 fires if any of: `knee_between_torso < 0.45`, `hip_flexion < 118°`, or
`shin_verticality < 0` (the ankle *above* the knee — feet on a stool rung; it
fires on exactly 1 file in 4,000 and is kept because that posture is real).

### 2.3 Correcting Round 3's claim about FK legs

Round 3 said SMPL-H forward-kinematic leg angles are unusable, on the strength of
a confirmed standing participant reading 87.4° of knee flexion. **Half of that
was right.** The absolute angles are uncalibrated and that warning still stands
for anything needing a real joint angle. The *ordering* was never checked, and it
is clean. Both facts have the same cause: zeroed betas make the angles wrong in
absolute terms and comparable across files.

### 2.4 Your structural hint, and the correction to its scope

You wrote that a participant is either standing in all their clips or sitting in
all of them. That is the single most useful thing in your review, and it is
correct — but the scope is the **recording session**, not the person. A chair, a
camera height and a standing platform are set up once per session.

P0003A settles it. They appear in **eight** sessions:

| session | files | hip flexion |
|---|---|---|
| S0200 | 1 | **88.2** ← the file you confirmed seated, low chair, legs crossed |
| S0126 | 2 | 130.6, 132.0 |
| S0130 / S0134 / S0192 / S0196 / S0408 / S1216 | 7 total | 137.9 – 156.1 |

The rendered S0126 clip shows them plainly **standing on a platform**. Across the
whole scan the pattern holds in aggregate: median within-session spread of hip
flexion **2.76°** against **7.11°** between sessions, and units that disagree
internally fall from 6.81% of multi-file participants to **2.52%** of multi-file
participant-sessions.

So the FM1 verdict is taken once per **(participant, session)**, by plain
majority of that unit's files. One clause, nothing else.

This also killed a rule I could not defend. An earlier draft carried a "deep"
clause — one file far inside the seated range speaks for the whole participant —
added purely to rescue P0003A. Leave-one-participant-out broke on exactly that
participant, the signature of a parameter fitted to one example. Under session
scoping P0003A's seated session holds one file, it flags, and the majority clause
carries it. The deep clause changes **zero** verdicts on 4,000 files and is gone.

### 2.5 Result on your labels

| | predicted sitting | predicted standing |
|---|---|---|
| **sitting** (14) | **14** | 0 |
| **standing** (15) | 0 | **15** |

Identical at file level and unit level. Nine of your fifteen former false
positives are still flagged — but by **FM2**, not FM1; six now pass everything.

### 2.6 What I do *not* claim, and what I need from you

An adversarial re-analysis found real problems with the confidence I would
otherwise have stated. Reporting them rather than the headline:

1. **The label set is selection-confounded on both sides.** Ten of your fourteen
   sitting labels came from the `knee_spread > 0.90` pool, and inside that pool
   `hip < 118` fires on 33 of 34 files *regardless of posture*. Under a null where
   hip flexion carries no information beyond pool membership, the 25 gallery
   labels carry about **p = 0.013**, not the 2.6 × 10⁻⁴ that "perfect separation
   on 29 files" suggests. Only the four pass-group sitters were unselected on leg
   geometry, and two of those came from the FM3 group, which is itself 1.9×
   enriched for low hip flexion. Genuinely selection-free evidence: **two files**.
2. **File-level recall is 0.706, not 1.0.** Extending your labels to all 96 files
   of the 19 labelled participants via the structural hint, 10 of 34 sitting files
   read as standing at file level. The unit majority absorbs them, which is the
   point of aggregating, but the underlying per-file measure is not perfect.
3. **The short-leg mechanism is reduced, not eliminated.** Of the 55 files with
   `leg_over_torso < 0.85`, **15 still fire** under the new rule — a risk ratio of
   **7.58** against the rest of the corpus. Every confirmed-*standing* label in
   this whole exercise comes from that 1.4% slice, so the false-positive rate on
   ordinary-proportioned standing people is **unmeasured**.
4. **56.3% of units hold one file** and can never disagree with themselves, so the
   majority rule reports perfect confidence on more than half the corpus. A
   `near_cut` channel exists for exactly this reason: 191 units sit within 4° of
   the hip cut or 0.04 of the knee cut without firing.

**The ask.** `fm1_adjudicate.html` in the gallery is 37 clips, one per unit,
built around these gaps:

| clips | what they are |
|---|---|
| 9 | flagged SEATED **and** short-legged — the exact population that broke Round 3. If these are standing, the mechanism is not fixed. |
| 18 | just above a cut — a small threshold move flips them. |
| 10 | **unlabelled controls**: the rule says standing and nothing is unusual. They are there so your answers to the other 27 can be scored rather than just collected. |

One clip settles a whole unit. If the nine short-leg clips come back standing, I
will drop `hip_flexion` and keep `knee_between_torso` alone, which halves
short-leg firing.

---

## 3. FM2 — framing. Rebuilt, stricter.

### 3.1 Your example, diagnosed

`V00_S0180_I00000482_P0047`, elbow leaving frame at the 3:42 mark. At frame 6677
(t = 222.57 s):

| | x, on a 1080-wide raster | verdict |
|---|---|---|
| released COCO left elbow | **1085.4** (conf 0.935) | 6.4 px **outside** |
| SMPL-H projected left elbow | 1072.3 | 6.7 px inside |
| SMPL-H mesh surface | — | **24.5 px outside** |

Two compounding causes. **Joint centres are not the body surface** — the elbow
joint sat inside while the mesh reached 24.5 px out, a 32.2 px gap. And **the
projection is pulled inward on the arms** — against the released 2D keypoints the
SMPL-H projection reads further inside by a median of 11.0 px at the left elbow
and 9.2 px at the left shoulder (and further *out* by 17–20 px at the ankles).

Round 3 tested only projected joint centres, so it scored the file a perfect
`fm2_in_frame_and_valid_frac` of 1.0. It now reads 0.9902, with one violation run
of **92 frames = 3.07 s** starting at the timestamp you named.

### 3.2 The new rule

A frame is a violation if **any** of these holds:

* an SMPL-H **mesh vertex** projects outside the raster;
* `smplh:is_valid` is false;
* a confidently-detected COCO **body-17 or hand** keypoint is outside the raster.

A file is flagged if any frame is a violation (`min_violation_run_frames: 1` —
the literal reading you asked for; 2 is available and drops one-frame noise).

Face-68 and feet-6 keypoints are **not** tested: measured over 180 files they
sit >100 px inside and add exactly zero files. The released detection **box** is
not a term either — its coordinates are clamped to the raster, so it can say that
something was cut but never how much.

### 3.3 What each term is actually worth

Cumulative over all 4,000 files, in composition order:

| term added | new files | cumulative flag rate |
|---|---|---|
| SMPL-H joint centres outside | 2,668 | 66.70% |
| + **mesh vertices** outside | **+80** | 68.70% |
| + `smplh:is_valid` false | +398 | 78.65% |
| + COCO body-17 outside | **+0** | 78.65% |
| + COCO hand keypoints outside | +98 | **81.10%** |

**Do not let me oversell the mesh term.** The per-frame joint-to-surface gap has
a median of 38.8 px, but that is *not* what the vertex term buys, because
`IN_FRAME_JOINTS` already contained indices 63–72 — the ten fingertips, which
`smplx` produces by **selecting mesh vertices**, not by kinematics. The joint set
was already part surface. Going from joints to the full mesh is worth **80 files,
2.00 points**. Real, and it is what catches your file, but a 2-point effect, not
a 38-pixel one. (The joint term is kept because it costs nothing and currently
flags zero files the vertices do not — an invariant worth continuing to measure.)

### 3.4 Paired against Round 3

On the same 4,000 files: Round 3's rule flags 77.475%, Round 4's flags **81.10%**.
It is a **strict superset** — **0** files Round 3 flagged now pass, **145** newly
flagged.

### 3.5 The thing you should know before you look at the gallery

**360 files — 9.00% of the corpus — are flagged by SMPL-H invalidity alone, with
no out-of-frame geometry at all.** 62.05% of files have at least one invalid
SMPL-H frame. And FM2 remains a one-bad-frame-in-four-minutes rule: among the
3,244 flagged files the median `in_frame_and_valid_frac` is **0.9841** and the
median longest violation run is 38 frames, but only 117 of them are tripped by
three frames or fewer. The violations are real; the question is whether a
three-second event should reject a four-minute recording. That is §5.

---

## 4. Flag rates, overlap, and what survives

### 4.1 Per label (4,000 files, 2,000 each)

| detector | naturalistic | improvised | all | undetermined |
|---|---|---|---|---|
| FM1 sitting (unit) | 144 (**7.20%**) | 32 (**1.60%**) | 176 (4.40%) | 0 |
| FM2 framing | 1,552 (**77.60%**) | 1,692 (**84.60%**) | 3,244 (81.10%) | 0 |
| FM3 static hands | 214 (**10.70%**) | 70 (**3.50%**) | 284 (7.10%) | 0 |
| *FM1, file level* | 128 (6.40%) | 29 (1.45%) | 157 (3.93%) | 0 |
| *near an FM1 cut* | 215 (10.75%) | 138 (6.90%) | 353 (8.83%) | — |

Naturalistic is 4.5× more likely to be seated and 3.1× more likely to have static
hands; improvised is worse on framing. Nothing is undetermined — every detector
ran on every file.

### 4.2 Overlap

| combination | files |
|---|---|
| FM2 only | 2,995 |
| *(nothing — passes)* | 566 |
| FM3 only | 156 |
| FM1 + FM2 | 135 |
| FM2 + FM3 | 107 |
| FM1 only | 20 |
| FM1 + FM3 | 14 |
| FM1 + FM2 + FM3 | 7 |

Flag counts: 0 → 566, 1 → 3,171, 2 → 256, 3 → 7. Pairwise Jaccard is tiny —
FM1&FM3 0.048, FM1&FM2 0.043, FM2&FM3 0.033 — so the three detectors are close to
independent and FM2 dominates the rejection.

### 4.3 Surviving volume

**Overall pass rate 14.15%** (566 of 4,000). Naturalistic 14.8%, improvised 13.5%.

Weighting by duration, because long recordings are more likely to contain one bad
frame:

| label | scanned dyad-hours | hours-weighted pass rate | surviving |
|---|---|---|---|
| naturalistic | 469.5 | 13.11% | 61.6 |
| improvised | 971.1 | 12.10% | 117.5 |
| **total** | **1,440.6** | | **179.1** |

**179 dyad-hours survive**, against Round 3's 189 under a looser FM2 and a
detector count that included FM4. A stricter FM2 and a corrected FM1 cost about
10 dyad-hours net.

---

## 5. The FM2 tolerance question is still yours

Unchanged in kind from Round 3, restated with Round-4 numbers. FM2 as specified
is literal: one violating frame rejects the file.

| FM2 rule | files passing FM2 | overall pass rate | surviving dyad-hours |
|---|---|---|---|
| **file level, any frame** (current) | 18.90% | 14.15% | **179** |
| tolerate 0.1% of frames | 25.40% | 20.03% | 275 |
| tolerate 1% of frames | 50.62% | 43.25% | 632 |
| tolerate 5% of frames | 82.12% | 72.52% | 1,084 |
| drop FM2 entirely | 100% | 89.03% | 1,329 |

For scale, Round 3's joint-centre rule left 22.53% passing FM2 against Round 4's
18.90% — the rebuild costs 3.6 points of FM2 pass rate.

The median file would keep **72.7%** of its 30-second windows under a windowed
rule (mean 66.5%). So the choice between "reject the file" and "reject the
window" is worth up to **1,150 dyad-hours** — far larger than anything else on
the table. I have not touched it, because you said to fix FM2 first and see where
we land.

---

## 6. Distributions (all 4,000 files)

Full CSVs in `outputs/session2/v00_fm_scan_r4/`:
`measurement_distributions.csv` (per label too), `threshold_sweeps.csv`,
`flag_rates_by_label.csv`, `flag_combinations.csv`, `flag_pairwise_overlap.csv`,
`fm1_unit_verdicts.csv`, `fm1_units_near_the_cut.csv`.

| measurement | p1 | p5 | p25 | p50 | p75 | p95 | p99 |
|---|---|---|---|---|---|---|---|
| `fm1_knee_between_torso_p50` | 0.352 | 0.468 | 0.521 | 0.547 | 0.572 | 0.607 | 0.633 |
| `fm1_hip_flexion_deg_p50` | 105.6 | 121.6 | 135.7 | 145.3 | 153.3 | 161.9 | 166.0 |
| `fm1_leg_over_torso_p50` *(retired)* | 0.834 | 0.927 | 1.067 | 1.206 | 1.341 | 1.547 | 1.659 |
| `fm1_knee_spread_over_torso_p50` *(retired)* | 0.197 | 0.259 | 0.328 | 0.385 | 0.441 | 0.524 | 0.766 |
| `fm2_in_frame_and_valid_frac` | 0.483 | 0.844 | 0.967 | 0.990 | 0.999 | 1.000 | 1.000 |
| `fm2_violation_run_max_frames` | 0 | 0 | 4 | 26 | 68 | 298 | 1,060 |
| `fm2_windowed_pass_frac` (30 s) | 0 | 0.067 | 0.455 | 0.727 | 1.000 | 1.000 | 1.000 |
| `fm2_surface_correction_px_p50` | 14.3 | 25.8 | 32.9 | 38.8 | 44.3 | 55.6 | 99.1 |
| `fm3_static_frac` | 0 | 0.000 | 0.016 | 0.152 | 0.468 | 0.802 | 0.953 |
| `observed_duration_s` | 70 | 104 | 168 | 232 | 312 | 416 | 492 |

Threshold sweeps, since you said you want to move these:

| cut | flag rate (nat / imp / all) |
|---|---|
| `knee_between < 0.40` | 2.90 / 0.35 / 1.63% |
| **`knee_between < 0.45`** | **5.45 / 1.10 / 3.28%** |
| `knee_between < 0.50` | 15.65 / 9.35 / 12.50% |
| `hip_flexion < 114` | 3.80 / 0.55 / 2.17% |
| **`hip_flexion < 118`** | **5.45 / 1.15 / 3.30%** |
| `hip_flexion < 122` | 8.30 / 2.50 / 5.40% |
| `fm3_static ≥ 0.70` | 13.85 / 4.90 / 9.38% |
| **`fm3_static ≥ 0.75`** | **10.65 / 3.50 / 7.07%** |
| `fm3_static ≥ 0.80` | 8.05 / 2.10 / 5.08% |

FM1 is on a plateau: every file cut in 116–120 gives the same 14/14 and 0/15 on
your labels, so 118 is chosen on flag rate, not on a gap. Both FM1 measures move
naturalistic ~4–5× more than improvised at every cut.

---

## 7. The S/B/M dots in Panel A

The three dots top-right of Panel A are the **dataset's own per-frame validity
flags**, read straight from the NPZ — the dataset's opinion, not ours:

| dot | NPZ key | asserts |
|---|---|---|
| **S** | `smplh:is_valid` | a fitted 3-D body exists for this frame |
| **B** | `boxes_and_keypoints:is_valid_box` | the box and all 133 2-D keypoints were tracked (the box outline turns the same colour) |
| **M** | `movement:is_valid` | the derived movement/expression feature block is present |

**Green** = the dataset marked that stream valid on the frame on screen. **Red** =
invalid. **Grey** = the file carries no such mask at all, which is normal for M.

The dataset ships these flags with **no published definition**, so what each one
checks upstream is `unverified`; everything below is what we measured.

A **red B** is the most serious: on every box-invalid frame audited, the box and
all 133 keypoints were exactly zero and the SMPL-H translation held a ~1e12–1e14
sentinel, with the video showing the participant out of frame or a cut to black —
treat those frames as having no usable person at all. A **red S** should make you
distrust Panels B and C specifically: the pose values stay finite and
plausible-looking and the 2-D tracking often looks fine, but the hand pose tends
to freeze, so the *body fit* is suspect even though the picture is not. A **red
M** is the least alarming: it arrives in exact two-second (60-frame) blocks with
the features zero-filled and no visible change in the video. Pooled invalid rates:
box 0.093%, SMPL-H 4.47%, movement 14.7%.

Sustained red is what matters. A single red frame is common and harmless — but
note that FM2 counts any invalid SMPL-H frame as a violation, which is where 9%
of the corpus is rejected from (§3.5).

---

## 8. The gallery

`artifacts/private_review_v00_fm_r4/` — **433 clips, 30 s each**, three cohorts
kept visibly separate as you asked.

| page | clips | what it is |
|---|---|---|
| `index.html` | 433 | everything |
| `recheck.html` | **202** | the exact files from the Round-3 gallery, re-judged. Each card shows its Round-3 group beside its Round-4 verdict. |
| `fresh.html` | **194** | files never shown before, disjoint from the recheck cohort; 191 of them from participants the recheck cohort does not use. |
| `fm1_adjudicate.html` | 37 | the FM1 questions from §2.6, including 10 unlabelled controls |
| `pass.html`, `flagged_fm1_sitting.html`, `flagged_fm2_framing.html`, `flagged_fm3_static_hands.html`, `flagged_multi.html` | — | cross-cuts by verdict |

175 clips are anchored on the frame the scanner recorded as the start of the
worst FM2 violation, with a 5-second lead-in, so you see the participant leave
frame rather than starting already outside it. The anchor uses the detector's own
per-frame output, so a clip cannot be positioned on frames the detector did not
object to.

**How the 202 recheck clips moved:**

| Round 3 | → FM1 | → FM2 | → FM3 | → multi | → pass |
|---|---|---|---|---|---|
| FM1 sitting (24) | 6 | 9 | 0 | 3 | 6 |
| FM2 framing (24) | 0 | **24** | 0 | 0 | 0 |
| FM3 static hands (24) | 0 | 0 | 22 | 2 | 0 |
| FM4 mutual silence (24) | 0 | 3 | 0 | 0 | **21** |
| multi (16) | 0 | 10 | 4 | 2 | 0 |
| pass (90) | 3 | 20 | 0 | 1 | 66 |

Reading that table: FM2 kept every file it had (the superset property, visible);
the 18 clips that left FM1 are your false positives being released; the 3
pass→FM1 moves are your false negatives being caught; the 20 pass→FM2 moves are
the stricter FM2 finding what the joint-centre rule missed, which is the class
your `V00_S0180` example belongs to. 93 of the 202 now pass.

---

## 9. What these three modes still do not cover

1. **A bystander the tracker correctly ignores.** The NPZ holds one tracked
   participant. Every FM2 term is about that one person's geometry, so a second
   person who never causes a track switch is invisible. Unchanged from Round 3,
   still `unverified`.
2. **Clothing, hair, hats, held objects.** SMPL-H with zeroed betas is a naked,
   average-shape body. A real silhouette extends past the mesh, so "the body left
   frame" is really "the *model* left frame".
3. **Bad framing that never cuts the body**: participant tiny in shot, badly
   off-centre, camera at knee height, tilted horizon. Nothing measures
   composition.
4. **Absolute camera correctness.** Everything rests on the unvalidated HMR
   full-frame hypothesis; M-4 measured a median body-17 reprojection error of
   25.53 px. That is roughly the size of the events FM2 detects, which is
   precisely why the independent released-keypoint term is in the rule.
5. **9% of the corpus is rejected on missing data, not bad framing** (§3.5). That
   is conservative and blind, not informed.
6. **Occlusion.** A body fully inside the raster but behind furniture reads as
   perfectly framed.
7. **FM1's false-positive rate on ordinary-proportioned standing people is
   unmeasured** (§2.6.3). Every standing label so far comes from a 1.4% slice.
8. **Single-file units** — 56.3% of them — get a majority verdict from one vote.
   The `near_cut` channel is the mitigation, not a fix.

---

## 10. What I need from you

1. **The 37 FM1 adjudication clips** (`fm1_adjudicate.html`). Sitting or standing
   for each. The nine short-leg ones decide whether the fix is real; the ten
   controls let me score the rest.
2. **The FM2 tolerance decision** (§5). File-level as specified leaves 179
   dyad-hours; a 1%-of-frames tolerance leaves 632; a 5% tolerance leaves 1,084;
   dropping FM2 leaves 1,329. Worth up to 1,150 dyad-hours, which dwarfs every
   other choice in front of us.
3. **Whether 9% of files rejected purely on `smplh:is_valid`** (§3.5) is what you
   want, or whether FM2 should be geometry-only with invalidity handled
   separately.
4. Still open from Round 2: **naturalistic vs improvised**. Naturalistic is 4.5×
   more likely to be seated and 3.1× more likely to have static hands; improvised
   is 1.09× worse on framing. Naturalistic is the smaller pool (469.5 dyad-hours
   against 971.1) but survives at a slightly higher rate.

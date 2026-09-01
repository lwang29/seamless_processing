# V00 failure-mode detection, Round 5 — FM1 to FM3

Supersedes `reports/04_failure_mode_detection.md`. Two changes, both from your
Round-4 review: FM1 lost its second criterion after you adjudicated 37 more
clips, and **FM2 now gates on SMPL-H validity instead of framing**, per your PI.
FM3 is still untouched. Same 4,000 V00 files, re-measured, so all before/after
numbers are paired.

Scan: `sbatch slurm/v00_fm_scan.sbatch` (job 16567451, 64 tasks). Analysis:
`$VIBES scripts/v00_fm_analyze.py`. Gallery: `sample_v00_fm_gallery.py` then
`sbatch slurm/render_v00_fm_array.sbatch` (job 16567851).

**Headline: pass rate 14.15% → 31.48%, surviving volume 179 → 419 dyad-hours.**

---

## 1. What `smplh:is_valid` actually is

You asked before deciding. Here is what the dataset documents, and then what we
measured — the two are very different in size.

**Documented by the release: nothing.** Only the key name, shape `(N,)`, dtype
bool. There is no published statement of what makes a frame valid.

**Measured, over Sessions 1–5:**

1. It is **not** a data-absence marker. On invalid frames the pose and
   translation arrays are finite, nonzero, and usually varying — not zero-fill,
   not a sentinel, not a last-valid hold. That is what distinguishes it from
   `boxes_and_keypoints:is_valid_box`, which zero-fills the box and all 133
   keypoints and parks a ~1e12–1e14 sentinel in the translation.
2. It does **not** predict how well the fitted body lands on the person. I built
   a direct check for this — project the SMPL-H joints and compare against the
   released 2D keypoints over the twelve body landmarks M-4 validated. Paired
   within each of the 2,481 files that have frames on both sides of the flag:

   | | median fit error |
   |---|---|
   | frames where `smplh:is_valid` is **true** | **0.0792** shoulder widths |
   | frames where it is **false** | **0.0799** shoulder widths |

   A difference of 0.0007 shoulder widths, about 0.2 px. **In 44.9% of files the
   invalid frames fit better than the valid ones.** At file level,
   Spearman(valid fraction, fit error) = **+0.0030**. There is no signal here at
   all.
3. It **is** a hand-pose flag. This is the finding that matters. Measuring
   per-frame pose change on both sides of the flag, within each of 120 files that
   have 5–95% valid frames:

   | array | median change on valid frames | on invalid frames | ratio | frames bit-identical to the previous one, invalid |
   |---|---|---|---|---|
   | `body_pose` | 0.04834 | 0.04308 | 0.82 (p = 0.67, **n.s.**) | 0% |
   | `global_orient` | 0.00339 | 0.00313 | 0.93 | 0% |
   | `translation` | 0.06126 | 0.06219 | 1.02 | 0% |
   | **`left_hand_pose`** | 0.07553 | 0.02284 | **0.33** (p = 6.6e-12) | **41.0%** |
   | **`right_hand_pose`** | 0.07038 | 0.00213 | **0.03** (p = 8.3e-10) | **38.7%** |

   On invalid frames roughly **40% of hand-pose vectors are bit-identical to the
   previous frame** — literally frozen — while on valid frames that never
   happens. The body keeps tracking normally.

**So: `smplh:is_valid` marks frames where the fitting pipeline did not trust the
hand fit and held the previous hand pose.** Every other observation falls out of
that. It explains why the 2D body reprojection sees nothing (the body genuinely
is fine), and it explains Session 1's puzzle of invalid spans over "fully visible
participants with stable 2D keypoints and no obvious transition" — the body was
visible and stable; the hands were frozen.

**I agree with the change, and more strongly than your PI's argument requires.**
For a co-speech gesture model the hands *are* the label. A file with frozen hand
pose on some frames is not a mildly degraded example, it is a corrupted one. The
flag turns out to be pointed at exactly the right thing.

Two honest caveats. The fit check is 2D, so it is blind to **depth** — a
monocular fit can be badly wrong in depth and still project perfectly, and depth
matters for gesture. And it uses the released 2D keypoints as reference, which
are themselves a model output, so a correlated failure of both would look like
agreement. Neither caveat touches the frozen-hand result, which is measured
directly on the released pose arrays.

### What it costs

FM2 is now: **keep files where `smplh:is_valid` holds on every frame.**

| rule | files flagged | naturalistic | improvised |
|---|---|---|---|
| **valid on every frame** (current) | 62.05% | 57.50% | 66.60% |
| ≥ 99.9% valid | 54.00% | 50.80% | 57.20% |
| ≥ 99% valid | 28.48% | 27.30% | 29.65% |
| ≥ 95% valid | 8.33% | 8.50% | 8.15% |

The old geometric measures are still computed and reported — `fm2_in_frame_*`,
the mesh-vertex insets, the surface correction — they just no longer decide
anything. Retiring or restoring them is a config edit, not a rescan.

**Your `V00_S0180_I00000482_P0047` now passes**: the elbow does leave the frame,
and the SMPL-H stays valid on all 9,420 frames with a median fit error of 0.049
shoulder widths (18.3 px). Under Round 4 it was rejected. That is the PI's point
working exactly as intended.

### A direct fit gate, for completeness

Since I built the measure, here is what it would do on its own. Per-file median
fit error is remarkably tight — p1 0.039, p50 0.080, p95 0.141, **max 0.170**
shoulder widths (12.6 / 26.3 / 48.2 / 62.4 px). The p50 of 26.3 px matches M-4's
independent 25.53 px, which is a useful consistency check on the camera.

**There are no catastrophically bad body fits in this corpus by this measure** —
nothing above 0.17 shoulder widths, so a threshold anywhere sensible rejects
either almost nothing or an arbitrary slice of a smooth distribution. A cut at
0.15 rejects 2.10%; at 0.10, 34.35%. I do **not** recommend shipping it as a
gate: there is no natural break to put a threshold at, and it agrees poorly with
the validity flag (at a 0.10 cut: 821 files rejected by both, 1,661 by validity
only, 553 by fit only). It is worth keeping as a recorded measurement.

---

## 2. FM1 — hip flexion retired

Your 37 answers, scored against what the rule predicted:

| group | n | correct | what it tells us |
|---|---|---|---|
| **controls** (unlabelled, rule said standing) | 10 | **10/10** | The rule does not over-fire on ordinary standing participants. This was the gap the Round-4 report flagged as unmeasured; it is now measured and clean. |
| **near-cut** | 18 | 17/18 | One miss: `V00_S0043_I00000540_P0065`. |
| **short-leg** (rule said seated) | 9 | 4/9 | The problem area, as suspected. |

Splitting the short-leg group by *which criterion* fired is decisive:

| fired by | files | your verdict |
|---|---|---|
| `knee_between` (+ hip) | 3 | **3/3 sitting** ✓ |
| `shin_inverted` | 1 | **1/1 sitting** ✓ (P0293 — the one-in-4,000 stool posture) |
| **`hip_flexion` alone** | 2 | **0/2 sitting** ✗ |
| `knee_between` + hip, but knee only marginally | 3 | 0/3 sitting ✗ |

Across all 66 labelled files, **every file that hip flexion flags and
`knee_between` does not is a standing person** — 4 of 4. It was costing 29 extra
files corpus-wide with no validated hit among them. **Hip flexion is retired**;
it is still measured and reported, it just no longer decides.

FM1 is now one measure plus one add-on:

```
knee_between_torso < 0.43   OR   shin_verticality < 0
```
taken per (participant, session) by majority, unchanged from Round 4.

The cut sits on a plateau — any value in 0.41–0.44 gives the same 18/19 sitting
and 46/47 standing — so 0.43 is the midpoint, not a fit.

**On all 66 labels:**

| | predicted sitting | predicted standing |
|---|---|---|
| **sitting** (19) | **18** | 1 |
| **standing** (47) | **0** | **47** |

65/66 at unit level, with **zero false positives**. (At file level it is 64/66:
`V00_S0113_I00000487_P0008` reads 0.394, and the session majority absorbs it.)
Flag rate drops from 4.40% to **2.65%** of files, 55 seated units of 2,451.

### The one remaining miss, and what would fix it

`V00_S0043_I00000540_P0065` — sitting, `knee_between` 0.485, squarely in the
standing band. Your note is the clue: *"appears to be swinging their legs back
and forth at times"*. That is a high stool with the feet off the ground, and on
leg **posture** alone it is close to indistinguishable from standing — the hip is
nearly straight and the knee sits where a standing knee sits.

What is not indistinguishable is that the legs **move**. A standing participant's
ankles are planted; a dangling participant's swing. I have not built this. If you
want the case covered, ankle displacement normalised by shoulder width, measured
over the recording, is the obvious signal and it is a cheap addition to the next
scan. Say the word and I will measure it before proposing a threshold. Note that
not every dangling-feet case is missed: P0162 and P0163 (high chair) read 0.393
and 0.358 and are caught.

---

## 3. Flag rates, overlap, and what survives

| detector | naturalistic | improvised | all | undetermined |
|---|---|---|---|---|
| FM1 sitting (unit) | 91 (**4.55%**) | 15 (**0.75%**) | 106 (2.65%) | 0 |
| FM2 SMPL-H invalid | 1,150 (**57.50%**) | 1,332 (**66.60%**) | 2,482 (62.05%) | 0 |
| FM3 static hands | 214 (**10.70%**) | 70 (**3.50%**) | 284 (7.10%) | 0 |
| *FM1, file level* | 84 (4.20%) | 13 (0.65%) | 97 (2.43%) | 0 |
| *near an FM1 cut* | 227 (11.35%) | 132 (6.60%) | 359 (8.98%) | — |

Naturalistic is **6.1×** more likely to be seated and 3.1× more likely to have
static hands; improvised is 1.16× worse on SMPL-H validity.

| combination | files |
|---|---|
| FM2 only | 2,367 |
| *(nothing — passes)* | 1,259 |
| FM3 only | 210 |
| FM2 + FM3 | 58 |
| FM1 + FM2 | 55 |
| FM1 only | 35 |
| FM1 + FM3 | 14 |
| FM1 + FM2 + FM3 | 2 |

Flag counts: 0 → 1,259, 1 → 2,612, 2 → 127, 3 → 2. Pairwise Jaccard stays tiny
(FM1&FM3 0.043, FM1&FM2 0.023, FM2&FM3 0.022): three near-independent detectors,
with FM2 still doing most of the rejecting.

### Surviving volume

**Pass rate 31.48%** (1,259 of 4,000). Naturalistic 32.25%, improvised 30.70%.

| label | scanned dyad-hours | hours-weighted pass rate | surviving |
|---|---|---|---|
| naturalistic | 469.5 | 29.77% | 139.8 |
| improvised | 971.1 | 28.73% | 279.0 |
| **total** | **1,440.6** | | **418.8** |

**419 dyad-hours, up from 179.** The FM2 change is worth +240 dyad-hours and the
FM1 retune contributes a few more.

If you later want to loosen FM2, the curve is:

| FM2 rule | FM2 pass | overall pass | dyad-hours |
|---|---|---|---|
| **valid on every frame** (current) | 37.95% | **31.48%** | **419** |
| ≥ 99.9% of frames valid | 46.00% | — | — |
| ≥ 99% of frames valid | 71.52% | — | — |
| ≥ 95% of frames valid | 91.67% | — | — |

I would be cautious here. Given §1.3, tolerating invalid frames means tolerating
frozen hands, and hands are the training signal.

---

## 4. Distributions

Full CSVs in `outputs/session2/v00_fm_scan_r5/`.

| measurement | p1 | p5 | p25 | p50 | p75 | p95 | p99 |
|---|---|---|---|---|---|---|---|
| `fm1_knee_between_torso_p50` | 0.352 | 0.468 | 0.521 | 0.547 | 0.572 | 0.607 | 0.633 |
| `fm1_hip_flexion_deg_p50` *(retired)* | 105.6 | 121.6 | 135.7 | 145.3 | 153.3 | 161.9 | 166.0 |
| `fm2_smplh_valid_frac` | — | — | — | — | — | — | — |
| `reproj_shoulder_widths_p50` | 0.039 | — | 0.057 | 0.080 | 0.112 | 0.141 | 0.156 |
| `reproj_px_p50` | 12.6 | — | 18.7 | 26.3 | 37.5 | 48.2 | 53.8 |
| `fm3_static_frac` | 0 | 0.000 | 0.016 | 0.152 | 0.468 | 0.802 | 0.953 |

Sweeps for every live cut:

| cut | flag rate (nat / imp / all) |
|---|---|
| `knee_between < 0.41` | 3.35 / 0.50 / 1.92% |
| **`knee_between < 0.43`** | **4.15 / 0.65 / 2.40%** |
| `knee_between < 0.45` | 5.45 / 1.10 / 3.28% |
| `smplh_valid_frac < 0.99` | 27.30 / 29.65 / 28.48% |
| **`smplh_valid_frac < 1.0`** | **57.50 / 66.60 / 62.05%** |
| **`fm3_static ≥ 0.75`** | **10.65 / 3.50 / 7.07%** |

---

## 5. What these three still do not cover

Carried forward from Round 4, with two changes.

1. **Depth.** The new fit check is 2D and therefore blind to depth error, which
   is the dominant failure mode of monocular fitting and matters for gesture.
   Nothing here measures it. **`unverified`.**
2. **Framing, now deliberately.** FM2 no longer rejects a body leaving the
   raster. A participant can be half out of shot and pass, provided the fit stays
   valid. That is the intended behaviour as of this round; the measures are still
   recorded if you want them back.
3. **A bystander the tracker correctly ignores** — unchanged, still `unverified`.
4. **Composition**: participant tiny in shot, badly off-centre, camera at knee
   height, tilted horizon. Nothing measures it.
5. **A seated participant whose legs dangle and swing** (§2, P0065). One known
   miss; ankle motion is the untested fix.
6. **Occlusion.** A body inside the raster but behind furniture reads as fine.
7. **FM1's remaining short-leg contamination**: 1 of 47 standing labels
   (`P0008`, `knee_between` 0.394) still fires at file level. The session
   majority absorbs it here, but a single-file session would not.
8. **Single-file units** — 1,379 of 2,451 (56.3%) — get a majority verdict from
   one vote. The near-cut channel is the mitigation, not a fix.

---

## 6. Gallery

`artifacts/private_review_v00_fm_r5/` — **425 clips, 30 s, 10 filmstrip
thumbnails** (down from 12; they were too small to read).

| page | clips |
|---|---|
| `index.html` | 425 |
| `recheck.html` | 202 — the same files as Round 3 and 4, re-judged again |
| `fresh.html` | 190 — never shown before |
| `fm1_adjudicate.html` | 33 — remaining FM1 questions, including controls |
| `pass.html`, `flagged_*.html` | cross-cuts by verdict |

No action needed on it unless you want it — the numbers above are the
deliverable this round. 127 of the 202 recheck clips now pass, up from 93.

---

## 7. Open

1. **Ankle motion for the dangling-legs case** (§2). Cheap; say the word.
2. **Depth-error measurement** (§5.1). Harder, and the biggest remaining blind
   spot for a gesture model.
3. **Naturalistic vs improvised**, still undecided. Naturalistic is now 6.1×
   more likely to be seated and 3.1× more likely to have static hands, but
   survives at a marginally higher rate (32.25% vs 30.70%) and is the smaller
   pool (469.5 dyad-hours against 971.1).

# V01, V02 and V03 under a V00-tuned pipeline

An exploratory pass over the three vendors the detectors were **not** built on.
Thresholds are unchanged on purpose: the question is what a V00-tuned pipeline
does elsewhere and where it stops making sense, and moving the cuts would
confound "this vendor is different" with "this cut is different".

**Read every pass/fail label in the gallery as "what the current pipeline says",
not as ground truth.** Two of the three checks turn out not to transfer cleanly,
and the pipeline is blind to the single largest problem I found.

Sample: **3,600 files, 1,200 per vendor**, seed 20260921, eligible pool only
(all modalities present, probe ok, video stream, ≥30 s, not charades). Gallery:
`artifacts/private_review_vendors/`, 330 clips.

---

## 1. What is out there

| vendor | eligible files ≥30 s | dyad-hours | labels | raster | frame rate |
|---|---|---|---|---|---|
| V00 *(reference)* | 41,205 | 1,440.6 | both | uniform 1080×1920 | uniform 30/1 |
| **V01** | 11,596 | 328.8 | both | **1080×1920 (55%), 2160×2160 (43%)**, some 1012×1920, 1920×1080 | 30000/1001, 45000/1501, 30/1 |
| **V02** | 29,502 | 761.4 | **naturalistic only** | uniform 1080×1920 | uniform 30/1 |
| **V03** | 42,881 | 1,430.1 | 91% naturalistic | **2160×3840 (97%)**, 3840×2160, 640×480 | 30000/1001, 30/1, odd fractions |

Together **2,520 dyad-hours** outside V00, against V00's 1,441. Not a rounding
error — worth getting right.

---

## 2. What the pipeline says

| | FM1 seated | FM2 SMPL-H invalid | FM3 static hands | **passes** |
|---|---|---|---|---|
| V00 *(reference)* | 2.23% | 62.95% | 7.32% | **30.35%** |
| V01 | **39.11%** | 68.51% | 8.96% | 17.17% |
| V02 | 7.67% | 76.50% | 6.67% | 18.33% |
| V03 | 16.25% | 80.58% | 6.08% | 12.25% |

Pass rates are roughly half V00's, and FM1 fires seventeen times more often on
V01. That last number is not what it looks like.

---

## 3. FM1 does not survive a change of frame shape

Splitting FM1 by raster shape rather than by vendor:

| shape | files | FM1 rate | `knee_between_torso` median | fit error |
|---|---|---|---|---|
| **portrait** | 3,054 | **8.9%** | 0.601 | 0.104 |
| **landscape** | 44 | **43.2%** | 0.533 | 0.102 |
| **square** | 496 | **93.3%** | **0.289** | 0.097 |

Within V01 the split is total:

| V01 raster | files | FM1 rate | `knee_between_torso` median |
|---|---|---|---|
| 1080×1920 | 683 | **0.0%** | 0.593 |
| 1012×1920 | 10 | 0.0% | 0.576 |
| 2160×2160 | 496 | **93.3%** | 0.289 |
| 1920×1080 | 5 | 100% | −0.303 |

So V01's headline 39% is not 39% of V01 sitting down. It is **one raster**. On
V01's portrait files FM1 fires on precisely zero of 693; on its square files it
fires on 93%.

The measure itself is rotation-invariant and computed from the released pose, so
the raster cannot reach it directly — it reaches it through the *fit*. The
released SMPL-H parameters were produced under a camera convention keyed to the
raster, and on square and landscape frames the fit distorts differently. The
V01 square clip in the gallery shows it plainly: Panel A's 2D keypoints track the
person correctly while Panel B's projected SMPL-H splays the legs outward and
misses the feet. That is the same leg-distortion mechanism as
`reports/06_smplh_pose_geometry.md`, but far worse.

**Consequence.** FM1's cut of 0.43 is calibrated for portrait framing and means
nothing outside it. On non-portrait rasters it is not measuring posture, and its
output should be discarded rather than re-tuned until the underlying fit is
understood. Affected: ~500 of 1,200 sampled V01 files, plus V03's landscape.

---

## 4. FM2 transfers, but everything is worse

FM2 reads the released `smplh:is_valid`, which is vendor-agnostic, so it does
transfer in the sense that it measures the same thing everywhere. What differs is
how much of it there is:

| | files with SMPL-H valid on every frame | median valid fraction | p5 |
|---|---|---|---|
| V00 | 37.05% | 0.99833 | 0.9008 |
| V01 | 31.49% | 0.99496 | 0.7516 |
| V02 | 23.50% | 0.99321 | 0.8260 |
| V03 | **19.42%** | 0.98770 | **0.3610** |

V03's fifth percentile is 0.36 — one file in twenty has more than a third of its
frames untrusted. Given §1 of the Round-5 report, that means frozen hand pose,
which is the training signal.

Independent fit quality agrees. Median reprojection error of the fitted body
against the released 2D keypoints, in shoulder widths:

| V00 | V01 | V02 | V03 |
|---|---|---|---|
| 0.079 | 0.071 | **0.111** | **0.127** |

V02 and V03 fit noticeably worse than V00 despite V02 sharing V00's exact raster
and frame rate. One V03 file reaches 3.51 shoulder widths, a scale of error V00
never produced at all (V00's worst was 0.179) — it is caught, but only by FM2.

FM3 is the one check that behaves the same everywhere (6.1–9.0% against V00's
7.3%), which is what you would expect from a measure normalised by shoulder width
and anchored on each file's own median.

---

## 5. The thing the pipeline cannot see: rotated video

`V03_S0990_I00000107_P3390` **passes all three checks** and the participant is
lying sideways in the frame. The video is stored landscape with portrait content
and the rotation is never applied.

Measuring the in-image angle of the shoulder-to-hip axis over **7,051 files** —
every non-portrait file in the eligible V01/V02/V03 pool, plus a 600-file
portrait control:

| vendor | shape | files | median roll | rotated (\|roll\| 60–120°) |
|---|---|---|---|---|
| V01 | square | 4,967 | −0.00° | **0%** |
| V01 | landscape | 60 | 4.54° | 0% |
| V01 / V02 / V03 | portrait | 600 | 0.00° | 0% |
| **V03** | **landscape** | **1,424** | **−90.00°** | **97%** |

**1,387 V03 files are a quarter turn from upright**, all 3840×2160 or 640×480.

Nothing in the pipeline notices, and that is structural rather than an oversight:
the released keypoints and the SMPL-H fit both track the rotated person correctly
*in raster coordinates*, so FM2 and FM3 are unaffected and FM1 is
rotation-invariant by construction. Every check is either blind to image
orientation or works in the frame where the rotation has already been absorbed.

It is also cheap to detect — the measurement above is one median over the
released keypoints, no forward kinematics — and cheap to fix, since the content
is fine and only needs rotating. I have **not** added a detector; that is a
decision about scope rather than a measurement, and 1,387 files is 3.2% of V03.

Note this is why Panel C is labelled "side: z/y upright" and still draws these
clips horizontally: the upright correction rotates in the depth/vertical plane to
undo camera pitch, and a quarter turn is a roll in the image plane, which it does
not touch.

---

## 6. Two smaller things

**Six V01 files have zero-length annotations.** Real video with a real duration —
144 s to 360 s — passing every inventory check, with `smplh:*` and
`boxes_and_keypoints:keypoints` all shaped `(0, …)`. `all_modalities_present`
only checks that the sibling files exist, not that they hold anything. The
scanner now reports these as `empty_annotations` rather than an opaque
`fk_error`. Six of 1,194 sampled V01 files, so roughly **0.5% of V01**;
Session 1 saw the same shape in V01 as wholly empty bundles.

**`movement:*` is V00-only**, as Session 1 found, so the **M** dot in Panel A is
grey for every clip in this gallery. That is correct behaviour and not a
rendering fault. It also means any future signal built on the movement block
would silently be V00-only.

---

## 7. The gallery

`artifacts/private_review_vendors/` — **330 clips, 30 s each**, free-text notes
on. This is a Pass-1 exploratory pass, so it deliberately offers no categories:
the point is to find failure modes we have not named, and a rubric would steer
you toward the ones we have.

| page | clips |
|---|---|
| `index.html` | 330 |
| `v01.html` / `v02.html` / `v03.html` | 110 each |
| `v01_pass.html`, `v01_flagged.html`, … | 55 each |

Six cells: pass and flagged for each vendor, 55 clips per cell, each a plain
random draw with no per-mode quota and no preference for singly-flagged files.
Clip position is random too — anchoring on a known failure would bias what gets
seen toward what we already know about.

Rasters represented: 1080×1920 (189), 2160×3840 (106), 2160×2160 (31),
3840×2160 (4). 300 distinct participants over 304 sessions.

Each card shows its vendor, raster, frame rate, verdict, and which checks fired.

### Worth looking for

- **V01 square (2160×2160)**: does Panel B match the person? The measurements say
  it often will not.
- **V03 landscape**: the rotated clips. Are they otherwise usable?
- **V02**: it shares V00's exact format but fits worse (0.111 vs 0.079) and
  passes at 18% against 30%. I have no explanation for that and it is the thing I
  would most like your eye on.
- Anything at all in the pass columns, since those are the files the pipeline
  currently says are fine.

---

## 8. What I would conclude so far

1. **FM1 should be gated on portrait framing** or disabled outside V00 until the
   non-portrait fit is understood. It is not measuring posture on square frames.
2. **FM2 and FM3 transfer**, with FM2 rejecting substantially more everywhere
   else — worst on V03 at 80.6%.
3. **Rotated video needs a detector** if V03 is in scope. One median over the
   released keypoints, and it would also be a repair rather than a rejection.
4. **V02 is the interesting case**: same format as V00, much worse fit, no
   improvised material at all. 761 dyad-hours, so it matters.
5. Nothing here changes the V00 conclusions. These vendors are a separate
   decision and the numbers above are the input to it, not the answer.

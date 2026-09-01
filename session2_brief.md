# Session 2 brief (v2) — look at the data first, then measure, then specify

## Resolved paths and inputs

| Item | Path | Notes |
|---|---|---|
| Source dataset | `./seamless_interaction` | read-only, unchanged |
| Metadata | `./datasets/seamless_interaction_metadata` | `interactions.csv`, `participants.csv`, `relationships.csv`, `LICENSE`, `README.md`, `.gitattributes` |
| **Missing metadata** | `filelist.csv` | fetch from `https://raw.githubusercontent.com/facebookresearch/seamless_interaction/main/assets/filelist.csv` into the same folder |
| Curated output root | `./seamless_interaction_clean` | writable; symlink to group storage |
| SMPL-H model | `./model_files/smplh/SMPLH_NEUTRAL.npz` | `SMPLH_MALE.npz` / `SMPLH_FEMALE.npz` also present — **use NEUTRAL**, see M-4 |

All three are symlinks, so they don't consume user quota — but **measure and report actual free space on the `seamless_interaction_clean` target**, since a previous reading of group storage showed only ~865 GiB free.

## What is already known (do not re-derive)

`filelist.csv` has been analyzed. Treat these as given:

- 129,370 participant-files: dev 2,180 / test 2,110 / train 125,080; improvised 35,750 / naturalistic 93,620.
- Vendor composition: **V03 44,252 (34.2%), V00 42,932 (33.2%), V02 30,172 (23.3%), V01 12,014 (9.3%)**.
- **`has_imitator_movement` ≡ `vendor == V00`, exactly** — all 42,932 movement files are V00 and every V00 file has movement. Session 1's V00-only finding is confirmed corpus-wide and is real release versioning. Open decision #3 is closed.
- V00 train: 41,280 files, 1,226 distinct participant IDs, 24,816 improvised / 16,464 naturalistic.
- Human annotations are sparse: `has_annotation_1p` = 491 files, `has_annotation_3p` = 1,663 files. Useful as an evaluation seed, not as training signal.
- Estimated V00 volume ≈ 1,300 dyad-hours (≈517 naturalistic-only) against a ~300-hour target — **verify by measuring actual durations per vendor**, but the headroom is large.
- `filelist.csv` lists 2,180 dev files; local staging has 2,119 extracted. **Check that 61-file gap.**

## Other resolved decisions

- **SMPL-H is now available.** Metric forward kinematics is unblocked. See M-4 for the exact settings and the validation test — do not skip the validation.
- **Movement features are optional metadata, never a gate.** The model is body + hands. Record availability; never let a missing movement mask reduce `valid_frac`.
- **Timing:** always use each file's measured `avg_frame_rate`. Never assume 30. No resampling or repair this session.
- **Target:** a few hundred dyad-hours from 4,065 — roughly 7% retention. Be strict; prefer rejecting borderline data over recovering it. Speaker diversity matters more than volume.

## Hard constraints (unchanged)

Source stays read-only. Nothing heavy on a login node — use Slurm, and remember the 1,001-index array cap. Verify network from *batch* nodes before designing around downloads. **Participant media stays on the cluster**: rendered clips mode `0600`, git-ignored, never uploaded or copied off-cluster. Keep using `git --git-dir=.git-session --work-tree=.`. The SMPL-H asset is research-licensed and non-redistributable — never commit it, never copy it outside the group path.

**No filtering thresholds, no scoring model, no selection decisions this session.**

---

# Phase 0 — Visualization and human review (do this first)

My PI's guidance, which reorders this session: *"The best place to start is to visualize the data first — you'll quickly get a sense of the overall quality. The most common issues are usually missing or unstable tracking, noisy body poses, unnatural hand motions, and occasional synchronization problems. Once you've gone through a few hundred samples, the patterns will become pretty obvious."*

This is not just orientation. It is how we learn which computable signals actually track perceived badness — and it produces the labelled seed set that a learned quality scorer will need later. Build the tooling to serve both purposes.

## P0.1 — Review renderer

Render short clips (**10 s**, ~480 px tall, H.264, **with audio muxed in** so sync problems are audible). Four synchronized panels:

- **Panel A — 2D keypoints on video.** Released COCO-WholeBody keypoints: body skeleton, both hands in distinct colours, bounding box, plus three small mask indicator dots (`smplh:is_valid`, `is_valid_box`, `movement:is_valid`) that change colour per frame.
- **Panel B — projected SMPL-H on video.** Run FK, project the 3D joints into the image, overlay on the same frames. This is the panel that exposes *noisy body poses*: you see the fitted skeleton against the real person.
- **Panel C — root-relative 3D skeleton**, fixed orthographic view (front and side, or a slow orbit). This exposes depth flicker and *unnatural hand configurations* that 2D hides. Plain matplotlib or manual projection is fine.
- **Strip — audio.** Waveform with own-VAD and partner-VAD shaded, and a moving playhead.

**No mesh rasterization.** No pyrender, no EGL, no headless OpenGL. Joints and lines only — it answers the question and avoids a known time sink.

Emit a static HTML index with the clips in a grid, each showing its `file_id`, vendor, label, activity type, and computed signals. I'll view it over SSH port-forward or VS Code Remote; **do not** copy clips off-cluster.

## P0.2 — Gallery sample (~300 clips)

Two clearly separated groups, analyzed separately:

- **200 uniform-random** across the corpus (not just dev — sample train too, read-only). This establishes the **base rate**: what fraction of ordinary data is actually bad. Without this, an extremes-driven gallery will badly mislead us.
- **100 targeted**, stratified across vendor × label × activity type, and deliberately including the extremes of every Session-1 signal (highest/lowest invalid fraction, longest invalid runs, highest jitter, largest duration mismatch, the empty-placeholder bundles, the 100%-invalid files).

Record the sampling seed and write the manifest to `configs/`. Label every clip in the HTML with which group it came from.

## P0.3 — Two-pass review protocol

**Pass 1 — exploratory (~100 clips, free-text).** No rubric. I and my PI watch and write notes. Your job is to build the tool and a simple notes-capture form, not to define the categories. Output: a **vocabulary of failure modes actually present**, in our words, with rough frequencies.

**Pass 2 — structured (~300 clips, including the 100).** *After* Pass 1, draft a rating rubric **derived from the vocabulary we produced**, propose it to me for approval, then capture ratings (1–5 per item plus an overall binary "would you train on this?"). Include ~15% duplicated items to measure rater agreement.

Then do the part that matters most: **correlate the human ratings against every computed signal.** For each of my PI's four named failure modes, report which signals separate good from bad, and — crucially — **which failure modes no current signal detects.** If clips we call "unstable tracking" have unremarkable validity fractions, the masks are insufficient and we need our own detector. That negative result would be the most valuable output of this session.

---

# Phase 1 — Measurements

Run M-1 in parallel with Phase 0 (it's I/O-bound and needs no human).

## M-1. Corpus-wide inventory (metadata + headers only)

Walk the entire dataset collecting **only** paths, file sizes, and `ffprobe` headers. **Do not read NPZ, WAV, or JSON payloads.** Join against the four metadata CSVs.

Report:
- **Actual durations** per vendor × label × split, to replace the estimated hour figures above. This is the number that decides the V00 question.
- `r_frame_rate` and raster distribution by vendor — confirm whether V00 is uniformly exact-30 / 1080×1920 corpus-wide, as dev suggested.
- Distinct participants per vendor, and per-participant file counts and durations (for later diversity quotas).
- Activity-type distribution from `interactions.csv`, with hours for the **pure-visual-communication (charades) activity** — to be excluded from speech-conditioned training — and the **language-grounded gesture game**, which is deliberately gesture-dense and speech-aligned.
- Empty-placeholder bundles corpus-wide (detectable by size alone: ~261-byte MP4, ~58-byte WAV).
- Total bytes and bytes-by-extension, to replace the ~20–25 TiB extrapolation with a measurement.
- Coverage: does every `filelist.csv` row have local files, and vice versa? Explain the 61-file dev gap.
- **Worker-scaling benchmark**: cold-ish read throughput at 1, 4, 8, and 16 workers, so the Stage-1 budget is grounded.

## M-2. Is `smplh:translation` a camera parameter?

Session 1 found tx ≈ box-centre-x (*r*=0.93), ty ≈ box-centre-y (*r*=0.94), tz anti-correlated with box height, tz median 39.16.

**Hypothesis:** it's HMR 2.0's weak-perspective camera translation, `tz = 2f/(s·r)` with f=5000, r=256 → 2·5000/256 = **39.06**.

Tests: (a) is `tz × (box_height / frame_height)` near-constant within a file? (b) does the implied `s = 39.06/tz` track normalized box scale? (c) is `global_orient`'s median first component (3.1152 ≈ π) consistent with a 180° x-axis flip between body and image frames? With SMPL-H now available, (d) the reprojection test in M-4 tests this directly and is the strongest evidence.

If confirmed: recommend dropping root translation from *motion* features, retaining its stability as a *tracking-quality* feature, and renaming the provisional acceleration column to reflect what it measures.

## M-3. Feature prototype on dev

Characterize distributions (**no thresholds**) for:

- normalized 2D keypoint speed and jitter — divide by that file's raster **before anything else**; report per joint group;
- **hand availability and hand-quality signals** per hand: usable-keypoint fraction, out-of-frame fraction, run-length structure, and near-default/flat-hand detection via finger-parameter variance. Session 1's blanked-hand test proved nothing currently catches this, and my PI independently named unnatural hand motion as a top failure mode — treat hands as first-class, not an afterthought;
- **geodesic angular velocity** for body and each hand via rotation composition — **never** axis-angle subtraction (Session 1 found raw deltas of 6.3 rad that were really 0.05 rad);
- gesture activity: wrist-speed percentiles, rest-pose-exit fraction;
- speaking fraction from the embedded VAD arrays;
- once M-4 passes: **metric 3D acceleration (`Accel`) in mm/frame²**, root-relative, and 3D wrist speed in mm/s.

Report the correlation of each quality signal against gesture activity. If jitter correlates strongly with activity, it's measuring motion rather than noise — flag it and propose residualizing.

## M-4. SMPL-H forward kinematics and validation

Load with **exactly** these settings, and treat each as a choice to be documented, not a fact from the data:

```python
import smplx
model = smplx.create("./model_files", model_type="smplh",
                     gender="neutral", ext="npz",
                     use_pca=False, flat_hand_mean=False,
                     num_betas=16, batch_size=B)
```

- **`gender="neutral"`.** MALE/FEMALE are present but must not be used: the dataset has no gender labels, and β=0 with a gendered model gives that gender's mean body — different bone lengths, so millimetres would not be comparable across participants. Neutral gives one consistent skeleton for everyone.
- **`use_pca=False`** — the release stores `(N, 15, 3)` full axis-angle per hand, not PCA coefficients. The library default would silently misread this.
- **`flat_hand_mean`** — **verify empirically**, don't trust my guess of `False`.
- **β = 0** for all participants. Document as a project convention.

**Validation test (the acceptance criterion for this whole workstream):** run FK, project the 3D joints into the image using the camera implied by M-2, and compare against the released 2D keypoints — body joints and, separately, the 42 hand keypoints (COCO-WholeBody indices 91–132). Report median and p95 reprojection error in pixels and as a fraction of box height. **Run it under both `flat_hand_mean` settings and keep whichever gives lower hand error.**

This single test simultaneously validates the joint convention, the hand-pose interpretation, and the camera hypothesis. If reprojection error is large and unstructured, stop and report rather than proceeding — every downstream metric depends on this being right.

Then: compute joints with translation zeroed and **subtract the pelvis** so all motion features are root-relative. SMPL outputs metres; ×1000 gives genuine millimetres, so `accel_mm_per_frame2` can finally become a real measurement.

## Small fixes (already justified)

- **Split `valid_frac_all`** into `valid_frac_smplh`, `valid_frac_box`, `valid_frac_movement` (nullable). The current AND makes 52% of rows `NaN` because of a face-feature mask.
- Add the hand-availability column.
- Add a **guard-band** parameter (default 3 frames) around invalid runs, justified by the one-frame-optimism edge case in `invalid_02.mp4`.

---

# Phase 2 — Frozen pipeline specification

Only after Phase 0 and Phase 1. Write `docs/pipeline_spec_v1.md`, then **stop for my review before implementing.** It must contain:

1. **Signal registry** — every signal with exact formula, units, column name, dtype, source modality, cost tier. Each signal must cite the human-review evidence that it detects something we actually saw.
2. **Gate vs. rank** — which signals are hard gates, which only rank. Justify every gate; each one is a bias we're introducing.
3. **Failure modes with no detector** — from P0.3. State plainly what we currently cannot catch.
4. **Stage graph** with I/O contracts and scope per stage. **Stage 1 must not open MP4 or WAV** — they are ~91% of the bytes and no Stage-1 signal needs them.
5. **Compute and I/O budget** per stage, in wall-clock on this cluster, using M-1's measured throughput.
6. **Slurm execution model** for ~129k files under a 1,001-index array cap: task manifest with contiguous slices, plus restart semantics.
7. **Output schema.** Default: **manifest + compact root-relative motion arrays + 16 kHz mono FLAC. No MP4 copies.** Estimate bytes and check against measured free space at `./seamless_interaction_clean`.
8. **Vendor strategy recommendation** with numbers: can V00 alone deliver ≥300 dyad-hours at ≥500 distinct participants after plausible gating? Give the naturalistic-only variant too, since V00 skews improvised (acted). I'll decide with my PI.
9. **Diversity and bias plan** — per-participant duration cap (~1–2%), participant floor (~500), and the stratified acceptance-rate audit that must pass before any subset ships.
10. **Open questions and risks**, each with what would resolve it.

---

## Reporting

Write to `reports/02_measurements.md`, `reports/02_review_findings.md`, and `docs/pipeline_spec_v1.md`.

Session 1's report was excellent — keep exactly the same discipline: observed numbers with the commands that produced them, `unverified` wherever it applies, and no inference presented as measurement. The habit of separating what the payload shows from what the paper claims is what made it trustworthy.

End with: (1) what the human review and measurements changed about the plan, (2) decisions you need from me, (3) anything now known to be expensive or impossible.
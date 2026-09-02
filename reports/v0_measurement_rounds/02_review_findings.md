# Session 2 — human review findings

Date: 2026-08-05 (America/Los_Angeles)  
Status: **Round 1 complete and reviewed by the user. Round 2 (V00-only,
label-stratified) built and rendered; the naturalistic-versus-improvised review
is with the user.**

This report is now in two parts. **Sections 1–7 describe Round 1**, the
corpus-wide 300-clip gallery, and are kept unchanged as the record of how the
vocabulary was obtained. **Sections 8–11 describe Round 2**, which the user
commissioned after reviewing Round 1: V00 only, stratified by label, with
targeted extremes chosen from the user's own vocabulary and two new renderer
panels.

Round 1's outcome, in the user's words: V00 clips had almost no recording issues
— participants standing, fully in frame, solid backgrounds, correct framing — and
only human issues, specifically a participant not moving their hands for the
entire conversation, or standing in an unnatural position. That corroborates the
measurements rather than merely agreeing with them: V00 is uniform exact-30 at
1080×1920, has the best dev span validity at 91.96% against V03's 76.89%, and has
zero truncated containers and zero empty WAVs.

**The vendor decision is made: V00 only.** What remains open is naturalistic
versus improvised, and Round 2 exists to decide it.

## 0. What is done and what is blocked

| item | round | state |
|---|---|---|
| renderer — four panels, audio muxed | 1 | **done**, with a material defect found and fixed (§2.2) |
| renderer — upper-body crop panel and whole-file filmstrip | 2 | **done** (§8, §11.3) |
| sampling — 200 uniform + 100 targeted corpus-wide | 1 | **done** (§1) |
| sampling — V00 only, 150 + 150 by label + 100 targeted | 2 | **done** (§8.1) |
| rendering | 1 | **done** — 295 clips + 5 metadata-only cards |
| rendering | 2 | **done** — 400 clips, 4 galleries (§11) |
| Pass-1 free-text capture | 1 | **done**; the gallery contains no rating controls at all, by test |
| Pass-1 exploratory review | 1 | **done by the user** — produced the vocabulary this report now uses |
| vocabulary proxies from that review | 2 | **4 of 5 work; desync fails and is reported as failing** (§9) |
| measured naturalistic-vs-improvised comparison | 2 | **done** over 8,000 files (§10) |
| naturalistic-vs-improvised review | 2 | **with the user** |
| Pass-2 rubric | — | **now unblocked**: Pass 1 produced a vocabulary, so a rubric can be drafted for approval whenever the user wants it |
| Pass-2 rating capture | — | **mechanism done and tested** (§5) |
| ratings-versus-signals correlation | — | **harness done and tested** (§6); needs ratings |
| computed clip signals | 1, 2 | **done** for both galleries |

The remaining blocked items are a human-input boundary, not missing
implementation.

## 1. Sampling design and realized composition

Two groups drawn from separate seeds, recorded separately, never mixed. The
uniform group is the only thing that establishes a base rate; the targeted group
deliberately over-represents extremes and **cannot** be read as a frequency
estimate. Every clip is labelled with its group in the HTML.

| parameter | value |
|---|---|
| population | `filelist` row with all four modalities present locally, `ffprobe` status ok, a video stream, duration ≥ 10 s, parseable frame count |
| population size | **128,532** of 129,370 rows (838 excluded) |
| uniform seed / count | **20260805** / 200 |
| targeted seed / count | **20260806** / 100 (22 extremes + 78 stratified fill) |
| Pass-1 subset seed / count | **20260807** / 100 |
| Pass-2 duplicate seed / fraction | **20260808** / 15% → 45 duplicate items |
| clip length | 10.0 s, ~480 px tall, H.264 + AAC muxed |
| start frame | deterministic per file from a seeded hash, **not** steered towards valid frames |
| rejected candidates | **0** |

Manifests: `configs/session2_review_manifest.csv` (300),
`configs/session2_review_pass1.csv` (100),
`configs/session2_review_pass2.csv` (345 items = 300 + 45 duplicates).

A start frame chosen inside valid regions would have produced a base-rate sample
that never shows a reviewer an invalid frame. That would have been the single
easiest way to make this whole exercise misleading, so the draw is uniform over
the whole usable interval.

### 1.1 Realized uniform composition versus the corpus

| vendor | uniform clips | expected at corpus share | corpus share |
|---|---:|---:|---:|
| V00 | 67 | 66.6 | 33.29% |
| V03 | 73 | 68.0 | 33.98% |
| V02 | 39 | 46.9 | 23.44% |
| V01 | 21 | 18.6 | 9.28% |

Composition tracks the corpus; V02 is 1.3 standard deviations low, which is
ordinary sampling noise at n = 200. By label: 55 improvised, 145 naturalistic. By
split: 194 train, 6 test, 0 dev — correct, since dev is 1.6% of the corpus and
the brief asked for corpus-wide sampling rather than dev-only. By activity: 169
`ipc_conversation`, 19 `grounded_gesture`, 8 `charades`, 4
`collaborative_storytelling`.

### 1.2 The 22 targeted extremes

Session-1 dev signal extremes (9): highest and lowest `wrist_speed_p90`; highest
provisional translation acceleration; largest duration mismatch; lowest mean
`valid_frac_all`; highest `smplh` and `box` invalid fraction; longest `smplh` and
`movement` invalid run. Run-anchored clips start 90 frames before the run so the
transition into failure is visible rather than the clip opening mid-failure.

Corpus-wide extremes that Session 1's dev-only scope could not see (13):

| selection reason | why it is in the gallery |
|---|---|
| `placeholder_261b_mp4_58b_wav` | the brief's known empty-bundle signature |
| `placeholder_708b_mp4_58b_wav_v00` | **the V00-only signature the brief's rule misses** |
| `placeholder_261b_mp4_with_full_wav` | placeholder video with real audio and a real transcript |
| `no_video_stream_multi_mb_mp4` | a 1.4 MB MP4 that still has no video stream |
| `mp4_container_unreadable_moov_missing` | a 147 MB truncated container |
| `empty_58_byte_wav_with_readable_video` | silent audio behind good video |
| `largest_within_interaction_member_duration_spread` | the 610 s member spread |
| `largest_avg_vs_container_frame_rate_disagreement` | the 6.82 fps disagreement |
| `longest_single_participant_video`, `shortest_renderable_participant_video` | duration extremes |
| `rarest_raster_640x480`, `_1080x960`, `_2180x3840` | raster tails |

Five of these are unrenderable and appear as `render_policy=metadata_only` cards
that state why, rather than as invented media.

The realized extreme count is 22 rather than the full selector set because
selectors deduplicate: a file already chosen by the uniform draw or by an earlier
extreme is skipped rather than duplicated. In particular the
`highest_movement_invalid_frac`, `longest_box_invalid_run`, and the
`100pct_*_invalid` selectors resolved to files already present. The realized list
is recorded verbatim in `outputs/session2/review_sampling/sampling_summary.json`
under `targeted_extreme_reasons`, so which extremes actually landed is a
measurement rather than an intention.

The 78-clip stratified fill is round-robin across `vendor × label ×
interaction_type`, one clip per occupied cell per pass, so every occupied cell is
represented before any cell gets a second clip.

## 2. Renderer and privacy controls

### 2.1 What each clip shows

Four synchronized regions, 10 s, H.264 with AAC audio muxed so sync problems are
audible:

- **Panel A** — released COCO-WholeBody keypoints on video: body skeleton, both
  hands in distinct colours, bounding box coloured by box validity, and three
  per-frame mask dots (`smplh:is_valid`, `is_valid_box`, `movement:is_valid`).
  The movement dot is grey on every non-V00 clip, which is correct and visible.
- **Panel B** — M-4-accepted SMPL-H FK projected into the same frames.
- **Panel C** — root-relative 3D skeleton, fixed-scale orthographic front and
  side views.
- **Strip** — waveform with own-VAD and partner-VAD shading and a moving
  playhead.

No mesh rasterization, no pyrender, no EGL. Joints and lines only.

Auxiliary face and toe/heel landmarks are deliberately not drawn: M-4 found
structured extreme reprojection errors on the six foot extras (max 3,355 px), and
displaying them as equally validated would misinform the reviewer.

### 2.2 A defect found by looking at the output, and fixed

**Overlays were effectively invisible on high-resolution vendors.** Overlays are
drawn on the native-resolution frame and the panel is then downscaled to fit the
480-px-tall canvas. Native line thickness was `max(2, min(w,h)/650)`, so:

| raster | panel width | downscale | native thickness | **effective output thickness** |
|---|---:|---:|---:|---:|
| 1080×1920 (V00, V02) | 216 px | 0.200× | 2 px | **0.40 px** |
| 2160×3840 (V03, 96.5%) | 216 px | 0.100× | 3 px | **0.30 px** |
| 2160×2160 (V01) | 384 px | 0.178× | 3 px | **0.53 px** |
| 3840×2160 (V03) | 680 px | 0.177× | 3 px | **0.53 px** |

Every case is below one output pixel, and `INTER_AREA` averages a sub-pixel line
into a faint smear. Panels A and B exist so a human can judge whether the
skeleton fits the person, and on V03 — a third of the corpus — the skeleton was
barely discernible.

This was found by extracting a midpoint frame from a rendered V03 clip and
looking at it, which is the same thing the review protocol asks humans to do.
Thickness now pre-compensates the known downscale (target ~2 output px), the
renderer fingerprint is bumped to v2 so existing clips re-render, and a test
asserts `thickness × display_scale ≥ 1.5` for all four dominant rasters while
also asserting the uncompensated value would have been below 1.0.

Had this not been caught, the most likely outcome was reviewers under-reporting
tracking failures on V03 and V01 because they could not see the overlay — a
silent bias in the labelled seed set that the whole quality scorer would inherit.

### 2.3 A remaining renderer limitation, not fixed

Portrait rasters get a narrow video panel: 2160×3840 yields only 216 px of width
for the person. The overlay is now legible, but fine detail — individual finger
joints — remains small. Options, none taken because the brief fixes clip height
at ~480 px: render taller, or add a person-cropped variant. **If reviewers find
V03 hand detail insufficient, a box-cropped Panel A variant is the fix.** Flagged
for the user rather than decided unilaterally.

### 2.4 Privacy controls

| control | state |
|---|---|
| output directory | `artifacts/private_review_session2`, forced mode `0700` |
| media, sidecars, HTML | forced mode `0600` |
| git | `artifacts/` is git-ignored; no clip is committed |
| off-cluster copies | none made; the HTML is intended for SSH port-forward or VS Code Remote |
| audio source | the released participant WAV, clipped at `start_frame / avg_frame_rate`, mono AAC |
| partner VAD | read only from an explicitly supplied paired JSON path; absence is recorded, never inferred by scanning |
| notes export | contains identifiers and text only, never media |

The SMPL-H asset is read in place. It is never copied and never committed.

## 3. Pass 1 — exploratory notes

**Awaiting the user and PI.** The gallery and form are ready.

```bash
# 100-clip exploratory gallery, free text only
artifacts/private_review_session2/pass1_exploratory.html
# all 300 clips
artifacts/private_review_session2/index.html
```

Notes persist in browser local storage; **Export notes JSON** downloads schema
`exploratory_free_text_v1` with one entry per `review_item_id`.

The Pass-1 page is verified by test to contain no rating controls, no rubric
fieldset, no radio inputs, and no `structured_ratings_v1` string. Offering
categories is the thing an exploratory pass exists to avoid, so their absence is
asserted rather than assumed.

## 4. Observed failure-mode vocabulary

**Blocked on Pass 1 by design.** No categories are inferred from the paper, from
the PI's prompt, or from the agent's own inspection of clips.

The one place a rubric-shaped file exists in this repository is
`tests/fixtures/rubric_test_only.yaml`, whose items are named
`placeholder_dimension_a` and `placeholder_dimension_b` precisely so that reading
it cannot seed anyone's vocabulary. It exists to prove the mechanism works.

## 5. Proposed Pass-2 rubric

**Blocked until Pass 1 closes.** The mechanism is complete and tested; only the
content is missing.

The renderer accepts a rubric YAML and refuses one that is not ready:

```yaml
schema_version: 1
status: approved                 # refused unless exactly "approved"
derived_from: <path to the Pass-1 notes export>   # refused if NOT_YET_DERIVED
items:
  - {id: snake_case_id, label: "question text", min: 1, max: 5,
     min_label: "...", max_label: "..."}
overall:
  id: would_train
  label: Would you train on this clip?
```

Validation refuses: a non-approved status, an underived rubric, zero items,
non-snake-case ids, duplicate ids, and an inverted or degenerate scale. Rendering
it produces 1–5 radio scales per item plus the binary overall question on every
card, a per-card completeness indicator, a "jump to next unfinished" control, and
an export under schema `structured_ratings_v1` stamped with the rubric hash and
its `derived_from` provenance.

```bash
python scripts/render_review_gallery.py --config configs/session2_review.yaml \
  --manifest configs/session2_review_pass2.csv --html-only \
  --gallery-name pass2_structured --rubric configs/pass2_rubric.yaml
```

Duplicates are already in the Pass-2 manifest: 45 of 345 items (15%) repeat an
earlier clip under a new `review_item_id`. They share media, so they cost no
extra rendering, and the HTML stores notes by `review_item_id` and never reveals
which items are repeats. The mapping is written outside the gallery at
`outputs/session2/review_sampling/pass2_duplicate_map.json` so agreement can be
scored afterwards without leaking pairs during review.

## 6. Human ratings versus computed signals

**Blocked on ratings. The analysis is implemented, tested, and ready to run.**

### 6.1 The signal substrate exists

Signals are computed over each clip's **exact rendered interval** —
`[start_frame, start_frame + round(10 s × avg_fps))` — so ratings and signals
describe the same frames. Correlating against a differently-chosen window would
be meaningless.

| item | value |
|---|---:|
| distinct clips measured | **295** |
| review items with signals (incl. duplicates) | **339** of 345 |
| skipped | 6, all `metadata_only` unrenderable bundles |
| output | `outputs/session2/review_clip_signals/clip_signals.parquet` |

Every M-3 signal is present per clip: split and guarded validity fractions,
per-hand availability and framing, hand-pose norm/variance/frozen-step, geodesic
angular velocity for root, body, and each hand, raster-normalized 2D speed,
acceleration and high-pass jitter per joint group, metric 3D acceleration and
wrist speed in millimetres, rest-pose exit, and speaking fraction.

### 6.2 What the analysis will report

```bash
python scripts/correlate_review_ratings.py \
  --ratings <exported.json> [...] --rubric configs/pass2_rubric.yaml
```

1. **Per rating item, the best-separating signals** by Spearman rho over every
   numeric signal column.
2. **`items_without_a_separating_signal`** — rating items no signal tracks above
   the reporting floor (default |rho| = 0.30). This is the output the brief
   correctly identifies as the most valuable, and the analysis is built so that
   this list is a first-class result rather than an absence.
3. **Overall-question separation** by rank AUC between "would train" and "would
   not train", with the median signal value in each group and the group sizes, so
   an AUC computed from four clips cannot be read as strong.
4. **Duplicate agreement** per scale (exact, within-one, mean absolute
   difference) and for the binary question, each with its usable pair count.
   Agreement is computed on raw rows before any deduplication; averaging first
   would hide the disagreement being measured.

The tests cover the case that matters: a synthetic signal built to track one
rating item is recovered at |rho| > 0.95, an item that nothing tracks lands in
`items_without_a_separating_signal`, and duplicate agreement returns the correct
fractions on a hand-checked example.

### 6.3 What Phase 1 already predicts the answer will be

M-3 measured these before any human looked at a clip, so they are the priors the
review will test rather than conclusions from it:

| PI-named failure mode | prediction from M-3 |
|---|---|
| noisy body pose | Every jitter and acceleration variant correlates with gesture activity at up to \|rho\| = 0.903. If reviewers' "noisy" ratings correlate with these signals, the correlation is confounded with how much the person moved, and the residualized variant must be checked before believing it. |
| unnatural hand motion | **No signal is expected to separate this.** Hand availability is exactly `valid_frac_box` on all 928 dev spans, `pose_exact_zero_frac` is 0.0 everywhere, and the only hand-specific signal that carries independent information detects a *frozen* pose, not an implausible one. |
| unstable tracking | `valid_frac_smplh` and `invalid_run_max` are the candidates, but Session 1 already found SMPL-H-invalid frames with visually stable 2D keypoints, so partial separation at best. |
| synchronization problems | **No detector exists.** Only container-level duration mismatch is measured, and VAD never overran the media on any dev span. Within-file drift is unmeasured. |

If Pass 2 confirms the second and fourth rows, the negative result stands as
measured: two of the PI's four named failure modes have no detector, and building
one is Session 3 work.

## 7. Reproduction

```bash
# sample (deterministic; rewrites the three manifests identically)
python scripts/sample_review_gallery.py --config configs/session2_review.yaml

# render (job 16520741 then 16521894 after the overlay fix)
sbatch --array=0-299%8 slurm/render_review_gallery_array.sbatch

# aggregate galleries from the already-rendered clips
python scripts/render_review_gallery.py --config configs/session2_review.yaml \
  --html-only --gallery-name index
python scripts/render_review_gallery.py --config configs/session2_review.yaml \
  --manifest configs/session2_review_pass1.csv --html-only \
  --gallery-name pass1_exploratory

# signals over the exact rendered intervals
python scripts/review_clip_signals.py --manifest configs/session2_review_pass2.csv
```

Per-clip render cost was 12.2 s wall and 815 MiB peak RSS on 4 CPUs, and 8–14 s
per array task in practice.

A scheduling observation worth carrying into Stage 1: the first array took **37
minutes** of wall clock (14:36:35 → 15:13:54) for 300 tasks of 8–14 s each at
`%8`, while the partition had 254 idle CPUs. The array was waiting on the
scheduler cycle, not on compute. Raising the throttle mid-flight with
`scontrol update JobId=<id> ArrayTaskThrottle=32` moved tasks in batches of ~32
per cycle instead of ~8. When per-task runtime is comparable to the scheduler
cycle, the throttle is the bottleneck — see `docs/pipeline_spec_v1.md` §6.

---

# Round 2 — V00 only, stratified by label

## 8. What changed and why

The user reviewed Round 1 and decided the vendor. Round 2 answers the one
question that decision left open.

| change | reason given |
|---|---|
| V00 only | decided by the user; corroborated by measurement, not merely consistent with it |
| stratified by label, equal N per arm | the open question is naturalistic *versus* improvised, and a comparison needs equal precision per arm |
| targeted extremes from the user's own vocabulary | Round 1 produced the vocabulary: framing, roll, sitting, static hands, desync lag |
| charades excluded | the model is speech-conditioned |
| `grounded_gesture` floor in each arm | highest-value pool; its natural share is ~6% and would otherwise be nearly invisible |
| Panel A′, an upper-body crop | Round 1's §2.3 flagged ~216 px of panel width; hands are the top concern |
| whole-file filmstrip | most of the vocabulary is file-level and a 10 s window cannot answer it |

### 8.1 Sampling design

| parameter | value |
|---|---|
| eligibility | V00, all four modalities present, `ffprobe` ok, video stream, duration ≥ 30 s, `interaction_type != charades` |
| eligible files | **41,205** — 16,500 naturalistic, 24,705 improvised |
| uniform, naturalistic | 150, seed 20260810, uniform within V00 naturalistic |
| uniform, improvised | 150, seed 20260811, uniform within V00 improvised |
| `grounded_gesture` floor | 25 per arm |
| targeted | 100, seed 20260812 |
| duplicates for agreement | 12%, seed 20260813 |

**Equal allocation is not the corpus split.** V00 is 60.0% improvised / 40.0%
naturalistic by eligible file count. Each arm is an unbiased base-rate sample *of
its own label*; a corpus-level rate is the label-share-weighted combination of
the two, and the weights are written into
`outputs/session2/v00_review_sampling/sampling_summary.json`. Reading the two
arms pooled, without weighting, would overstate naturalistic by half.

The `grounded_gesture` floor is the one deliberate distortion inside an arm. It
is applied by drawing that activity's quota from its own uniform draw first and
filling the remainder uniformly from the whole label, so both parts stay uniform
within their own frame and the realized activity mix is recorded rather than
implied.

## 9. Vocabulary proxies: what was built, and one that failed

Round 1 produced five terms. Four now have working computable proxies; one does
not, and saying so is the honest result.

| term | proxy | status |
|---|---|---|
| framing | person height as a fraction of frame, centre offset, minimum edge margin, fraction of frames touching an edge, body points outside the raster | works |
| roll | shoulder-line and hip-line tilt in degrees, plus their difference (a camera roll tilts both equally; a lean does not) | works |
| sitting | leg-over-torso and knee-drop-over-torso ratios, both scale-free, plus ankle trackability | works but finds nothing — see §10.3 |
| static hands | whole-file wrist travel and displacement normalized by **shoulder width**, plus wrist-above-hip fraction, on a strided sample across the entire recording | works, and is the discriminating signal |
| desync lag | mouth-activity against speech-activity cross-correlation | **fails on V00** |

### 9.1 The desync proxy fails, and why

The 68-point face layout was confirmed empirically from landmark geometry before
anything was built (jaw 0–16, brows 17–26, nose 27–35, eyes 36–47, outer mouth
48–59, inner mouth 60–67). The inner-lip centres 62 and 66 turned out to sit only
0.011 of face height apart, so the outer-lip lines are used instead. The failure
is therefore not an indexing mistake.

It is resolution. **V00's inter-ocular distance is 79–126 px**, so the mouth spans
a handful of pixels and lip motion is below the released whole-body face block's
resolution. Measured on eight dev files:

| measurement | result |
|---|---|
| peak mouth/audio activity correlation | median **0.151**, range −0.145 to 0.294 |
| argmax lag across files | −4, −3, −2, 0, 3, 9, 9, 12 frames — no concentration at zero |
| instantaneous mouth-opening against audio amplitude | \|r\| ≤ 0.22 for every one of six candidate landmark pairs |

Across the full 8,000-file pool the proxy is also **identical between labels**
(rank-biserial 0.0003), which is what a signal that measures nothing looks like.

The machinery is correct — a unit test recovers a synthetic ±4-frame shift with
the right sign — so the code is retained for a vendor framed closer. But **there
is still no working within-file desync detector**, and desync review candidates
are selected from container-level mismatch instead, labelled as such.

## 10. Measured comparison: V00 naturalistic versus V00 improvised

Computed over a seeded random pool of **8,000 eligible V00 files (19.4%)** —
3,158 naturalistic, 4,842 improvised, 1,233 distinct participants — on a 5-frame
stride across each whole recording. All 8,000 processed without error.

This is the quantitative half of the decision. It does not replace the review:
these proxies say how often a measurable correlate occurs, and only a human can
say whether the clips they flag are actually unusable.

### 10.1 The headline: naturalistic has substantially more static hands

| tail | naturalistic | improvised | ratio |
|---|---:|---:|---:|
| hands static > 50% of the file | **27.5%** | **11.9%** | 2.3× |
| hands static > 80% of the file | **7.0%** | **1.7%** | 4.1× |
| left hand above hips < 5% of the file | 2.7% | 1.0% | 2.7× |
| speaking < 10% of the file | 4.2% | 1.5% | 2.9× |

And on the continuous measures, improvised is more animated and more talkative:

| measure | naturalistic p50 | improvised p50 | rank-biserial |
|---|---:|---:|---:|
| right hand above hips, fraction of file | 0.609 | **0.867** | −0.297 |
| left hand above hips, fraction of file | 0.737 | **0.904** | −0.226 |
| hands-static fraction | 0.209 | **0.052** | +0.228 |
| speaking fraction | 0.367 | **0.444** | −0.263 |
| wrist speed, shoulder-widths/s | 0.195 | **0.216** | −0.134 |

Rank-biserial is read as: −0.297 means a random improvised file has its right
hand above the hips more often than a random naturalistic file about 65% of the
time.

**This cuts against the Session-2 §8.3 recommendation.** That recommendation
preferred naturalistic on spontaneity and participant-diversity grounds, before
any gesture-density measurement existed. The user's own top-named failure mode —
"participant not moving hands for the entire conversation" — is 2.3× to 4.1× more
common in naturalistic, and improvised carries more speech per file as well.

The trade is now three-way and explicit:

| | naturalistic | improvised |
|---|---|---|
| dyad-hours available | 469 | 974 |
| distinct participants | **939** | 485 — *below the 500 floor* |
| hands static > 50% of file | 27.5% | **11.9%** |
| speaking fraction, median | 0.367 | **0.444** |
| spontaneity | spontaneous | acted |

Naturalistic wins diversity by 1.9×; improvised wins gesture density and speech
by 2–4× and volume by 2×. Improvised alone fails the participant floor.

### 10.2 Recording quality is effectively identical between labels

Every framing, roll, and tracking proxy is close to indistinguishable, which is
exactly what the user observed by eye and is now measured at n = 8,000:

| measure | naturalistic | improvised |
|---|---:|---:|
| person fills < 40% of frame height | **0.0%** | **0.0%** |
| any frame touching a raster edge > 5% of the file | 6.7% | 5.5% |
| shoulder tilt > 10° | 0.32% | 0.19% |
| SMPL-H validity < 90% | 5.1% | 3.5% |
| body keypoints outside the raster, p95 | 0.13% | 0.11% |

Maximum observed shoulder tilt across the whole pool is 18.4°, and 13.5% of files
have a median tilt of exactly 0°. Roll is real but rare.

### 10.3 There are no sitting participants in V00

The posture proxies find nothing to find, and the negative is worth stating
precisely:

- **Ankles are usable in 100% of frames for 100% of the 8,000 files.** Not one
  file loses the feet, which a seated or table-cropped participant would.
- **No file has the person filling under 40% of frame height.**
- The leg-over-torso ratio has median 1.18 naturalistic / 1.23 improvised with p5
  at 0.91 and 0.94 — a tight unimodal distribution with no low mode.

The 1.2 cut in `tail_counts_by_label.csv` flags 48% of files and is therefore
**not a sitting detector** — it sits at the median. It is reported as
uninformative rather than presented as a prevalence. The honest conclusion is
that V00's protocol is uniformly standing, confirming the user's impression at
n = 8,000, and "sitting" needs no gate for this vendor.

## 11. Round-2 gallery

Manifests: `configs/v00_review_manifest.csv` plus per-arm splits
`v00_review_uniform_naturalistic.csv`, `v00_review_uniform_improvised.csv`,
`v00_review_targeted.csv`, and `v00_review_with_duplicates.csv`.

Galleries under `artifacts/private_review_v00/` (mode `0700`, media `0600`,
git-ignored, cluster-local):

| gallery | contents | use |
|---|---|---|
| `naturalistic.html` | 150 uniform naturalistic clips | judge one arm without interleaving |
| `improvised.html` | 150 uniform improvised clips | judge the other arm |
| `targeted.html` | 100 vocabulary and signal extremes | see what the tails look like |
| `index.html` | all 400 | reference |

Reviewing the two arms separately is deliberate: interleaved clips invite
relative judgement, and the question is how often each arm is unusable on its own
terms.

### 11.1 Realized composition

| group | `ipc_conversation` | `grounded_gesture` | `collaborative_storytelling` | total |
|---|---:|---:|---:|---:|
| uniform naturalistic | 117 | 25 | 8 | 150 |
| uniform improvised | 122 | 25 | 3 | 150 |
| targeted | 50 | 25 | 25 | 100 |

400 clips over **296 distinct participants**, 372 train / 22 dev / 6 test, zero
rejected candidates. The `grounded_gesture` floor landed exactly on 25 per arm
against a natural share of about 6%, so that pool is visible rather than
incidental — but the realized activity mix inside each uniform arm is therefore
*not* the label's natural mix, and the summary records that.

The 100 targeted clips are 28 extremes plus 72 stratified fill. All five
vocabulary terms are represented in the extremes: 6 framing, 3 roll, 3 sitting, 6
static hands, 4 desync (2 container-level, 2 from the failed weak proxy and
labelled as such), plus 6 Session-1 signal extremes.

### 11.2 Two spot checks

The static-hands extreme resolves to `hand_activity_static_frac = 1.0` — a
naturalistic file where both wrists stay within a tenth of a shoulder width of
their own median position for **100%** of the recording. Its filmstrip shows the
participant with hands clasped, unmoving across all twelve thumbnails of the
whole 1:04 interaction. The proxy found precisely the failure mode the reviewer
named, and the filmstrip is what makes it visible in one glance.

The roll extreme resolves to a **18.4°** median shoulder-line tilt, which is the
pool maximum.

### 11.3 Renderer performance note

The filmstrip initially used OpenCV's `CAP_PROP_POS_FRAMES`, which decodes
forward from the previous keyframe. Twelve seeks per clip cost **11 minutes per
clip** at 32 concurrent tasks on cold NFS, against 15 s with input-side
`ffmpeg -ss`. The first array was cancelled at 18/400 and the whole 400-clip
render then completed comfortably. Recorded because the same trap applies to any
future thumbnailing or random-access pass over this corpus.

### 11.4 Signal substrate and reproduction

Every M-3 signal is computed over each clip's exact rendered interval:
**400 distinct clips, 448 review items including the 48 blinded duplicates, zero
skipped**, in `outputs/session2/v00_clip_signals/clip_signals.parquet`. The
movement mask is present on all 400 — V00 carries it on every file, unlike the
other vendors — so `valid_frac_movement` is a real column here rather than a
nullable one. Median `valid_frac_smplh_and_box` over the sampled clips is 1.000.

```bash
sbatch slurm/v00_pool_signals.sbatch                 # 48-task array, 8,000 files
python scripts/analyze_v00_pool.py                   # label comparison
python scripts/sample_v00_review.py --config configs/v00_review.yaml
sbatch slurm/render_v00_review_array.sbatch          # 400 clips
python scripts/render_review_gallery.py --config configs/v00_review.yaml \
  --html-only --gallery-name naturalistic \
  --manifest configs/v00_review_uniform_naturalistic.csv \
  --title 'V00 naturalistic — 150 uniform clips'
python scripts/review_clip_signals.py --config configs/v00_review.yaml \
  --manifest configs/v00_review_with_duplicates.csv \
  --out outputs/session2/v00_clip_signals
```

## 12. What Round 2 needs from you

1. **Review `naturalistic.html` and `improvised.html` separately.** 150 clips
   each, ~25 minutes of video per arm. The filmstrip is above every clip, so
   "hands static throughout" and "framing consistent" are answerable without
   scrubbing.
2. **Then decide naturalistic, improvised, or a mix.** §10 is the measured half:
   naturalistic wins participant diversity 939 to 485 and is the only arm that
   clears the 500 floor alone; improvised wins gesture density and speech by 2–4×
   and volume by 2×. If a mix is acceptable, a per-label quota is the obvious
   shape and nothing in the pipeline prevents it.
3. **Optional: approve a Pass-2 rubric.** Pass 1 produced a vocabulary, so the
   blocker described in §5 is gone. Give the five terms scale definitions and the
   structured gallery and correlation harness run as-is.
4. **Tell me if `targeted.html` shows failure modes the proxies missed.** Those
   are the ones that need new detectors, and that list is more valuable than
   confirmation that the existing proxies work.

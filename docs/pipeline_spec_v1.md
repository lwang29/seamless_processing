# Pipeline specification v1 — DRAFT FOR REVIEW, NOT IMPLEMENTED

Date: 2026-08-05, revised 2026-08-06  
Status: **awaiting user review. No stage below has been implemented.**

**Decided since the first draft: vendor is V00 only.** The user reviewed the
Round-1 gallery and confirmed what §8 recommended — V00 clips show almost no
recording issues, only human ones. §8 is kept as the reasoning that led there;
§8.4 records the decision and what it settles. Naturalistic versus improvised
remains open and now has measured evidence in `reports/02_review_findings.md`
§10 that cuts against the original recommendation.

Every number in this document is a measurement from `reports/02_measurements.md`
or a stated arithmetic consequence of one. Where a value is a choice rather than
a measurement it says so. Where nothing was measured it says **unverified**.

## 0. The one thing that blocks this document from being final

Requirement 1 below says every signal must cite the human-review evidence that
it detects something we actually saw. **That evidence does not exist yet.** The
Phase-0 gallery is rendered and the review tooling is complete, but the Pass-1
exploratory review and Pass-2 structured ratings need the user and PI. Signals
are therefore marked:

- **`evidence: measured`** — a Session-1/2 measurement supports it.
- **`evidence: awaiting-P0.3`** — plausible, but no human confirmation that it
  tracks perceived badness.

**No signal marked `awaiting-P0.3` may become a gate until Pass 2 closes.** The
gate list in §2 is deliberately short for that reason. This is the honest state
of the work, not a placeholder to be filled in later without measurement.

---

## 1. Signal registry

Cost tiers: **T0** = already computed by M-1, free to join. **T1** = NPZ + JSON
only. **T2** = needs FK. **T3** = needs MP4 or WAV payload bytes.

### 1.1 Tier 0 — from the M-1 header inventory (already computed, 129,370 rows)

| column | formula / source | units | dtype | evidence |
|---|---|---|---|---|
| `observed_duration_s` | MP4 `format.duration`, falling back to stream duration | s | float64 | measured |
| `avg_fps` | MP4 `avg_frame_rate` as an exact `Fraction` | fps | float64 | measured — 416 files disagree with the container rate, max 6.82 fps |
| `raster_w`, `raster_h` | video stream width/height | px | int32 | measured — V00 uniformly 1080×1920 |
| `has_video_stream` | `ffprobe` returns a v:0 stream | — | bool | measured — 439 files fail; the brief's size rule misses 154 of them |
| `wav_size_bytes` | `stat` | bytes | int64 | measured — 187 files pair readable video with a 58-byte WAV |
| `mp4_container_readable` | `ffprobe` exit status | — | bool | measured — 7 V03 files are `moov atom not found` at 65–286 MB |
| `all_modalities_present` | all four siblings `stat` | — | bool | measured — 189 incomplete bundles |
| `member_duration_spread_s` | max − min participant duration inside one interaction | s | float64 | measured — up to 610.3 s in V03 |
| `interaction_type` | `interactions.csv` join on `prompt_hash` | — | string | measured — all 129,370 rows join |
| `participant_key` | `(vendor_id, participant_id)` tuple | — | string | measured — 627 IDs collide across vendors |

### 1.2 Tier 1 — NPZ + JSON only (1.7365 TiB, 7.65% of payload)

| column | formula | units | dtype | evidence |
|---|---|---|---|---|
| `valid_frac_smplh` | mean of `smplh:is_valid` | fraction | float64 | measured |
| `valid_frac_box` | mean of `boxes_and_keypoints:is_valid_box` | fraction | float64 | measured |
| `valid_frac_movement` | mean of `movement:is_valid == 1`; **nullable** | fraction | float64 | measured — absent on all non-V00 |
| `valid_frac_smplh_and_box` | mean of the elementwise AND of the two required masks | fraction | float64 | measured — replaces the three-way AND that was NaN on 52% of Session-1 rows |
| `*_guarded` variants | as above after widening every invalid run by `guard_band_frames` (default **3**) on the whole-file mask before slicing | fraction | float64 | measured — justified by the one-frame optimistic-valid edge in `invalid_02.mp4` |
| `invalid_run_max_*` | longest invalid run per mask | frames | int32 | measured — up to 3,740 frames in dev |
| `hand_avail_frac_{left,right}` | fraction of frames whose 21-point hand XY block is finite and not exact-zero | fraction | float64 | measured — **exactly equal to `valid_frac_box` on all 928 dev spans**; keep as a guard, not as coverage |
| `hand_{l,r}_out_of_frame_point_frac` | placed hand points outside the raster ÷ placed points | fraction | float64 | measured — negligible in dev (p95 = 0.0029) |
| `hand_{l,r}_pose_norm_median_rad` | median ‖45-vector‖ of the released hand axis-angle | rad | float64 | measured — p50 = 2.72; `flat_hand_mean=True` makes 0 a flat hand |
| `hand_{l,r}_pose_exact_zero_frac` | fraction of frames with an exactly zero hand pose | fraction | float64 | measured — **0.0 on every dev span**; threshold-free near-default detector |
| `hand_{l,r}_pose_frozen_step_frac` | fraction of adjacent frame pairs with an identical 45-vector | fraction | float64 | measured — ρ = 0.58/0.63 with SMPL-H invalidity, only 0.18 with activity |
| `angvel_{root,body,left_hand,right_hand}_rad_per_s_p{50,90,99}` | geodesic angle of `R_tᵀR_{t+1}` from the trace, × fps | rad/s | float64 | measured — **never axis-angle subtraction**; naive form exceeds 100 rad/s on 43.5% of spans |
| `speed2d_{group}_rasterfrac_per_s_p{50,90}` | ‖Δxy‖ ÷ max(W,H) × fps, per joint group | raster-fraction/s | float64 | measured |
| `jitter2d_hp_{group}_rasterfrac_p50` | residual after a centred 5-frame moving average | raster-fraction | float64 | measured — **activity-confounded at ρ = 0.90; rank only, never gate** |
| `accel2d_{group}_rasterfrac_per_s2_p50` | ‖second difference‖ × fps² | raster-fraction/s² | float64 | measured — activity-confounded at ρ = 0.88 |
| `speaking_frac` | merged `metadata:vad` intervals ÷ **window** duration | fraction | float64 | measured — span-local p50 = 0.109 and **44.7% of 5-s dev spans have no speech**; ρ = 0.40 with activity |
| `vad_interval_count`, `vad_median_segment_s` | merged interval statistics | count, s | int32, float64 | measured |

**Reviewer-vocabulary proxies (Round 2).** Derived from the user's own Pass-1
terms, so unlike the rest of the registry these are named by a human before being
measured. All are Tier 1 except where noted.

| column | vocabulary term | evidence |
|---|---|---|
| `framing_person_height_frac_p50`, `framing_min_edge_margin_frac_p50`, `framing_frames_touching_edge_frac`, `framing_body_out_of_frame_point_frac`, `framing_centre_offset_*` | framing | measured on 8,000 V00 files; no file has the person under 40% of frame height |
| `roll_shoulder_abs_deg_p50`, `roll_shoulder_deg_iqr`, `roll_shoulder_minus_hip_deg_p50` | roll | measured; max observed 18.4°, only 0.24% of files above 10° |
| `posture_leg_over_torso_p50`, `posture_knee_drop_over_torso_p50`, `posture_ankles_usable_frac` | sitting | **superseded — and the Round-2 conclusion was wrong.** These found nothing and I inferred V00 was uniformly standing. Round 3 found real seated participants that these ratios miss, because a stool sitter with feet on a rung still has ankles tracked in every frame. Use `v00_detectors.sitting_measures_2d` instead |
| `hand_activity_static_frac`, `hand_activity_{side}_speed_sw_per_s_*`, `hand_activity_{side}_above_hip_frac` | static hands | measured, whole-file, normalized by shoulder width so camera distance does not bias it. **The discriminating signal between labels** |
| `av_lag_frames`, `av_lag_peak_r` | desync lag | **FAILED validation on V00** — read §3. Retained only so the negative result is reproducible; must never gate or rank |

`hand_activity_*` is the first signal in this registry that was named by a
reviewer before it was built, which is the order the whole Phase-0 design was
meant to produce.

Every `*_status` column travels with its measurement. A missing modality is
recorded, never silently treated as valid — that rule is what stopped the
movement mask from erasing V00-only information in Session 1.

### 1.3 Tier 2 — requires SMPL-H forward kinematics

Fixed convention, frozen by the M-4 acceptance test: neutral model, β = 0 (16
betas), `use_pca=False`, **`flat_hand_mean=True`**, translation zeroed, pelvis
subtracted, metres × 1000.

| column | formula | units | dtype | evidence |
|---|---|---|---|---|
| `accel3d_{body,left_hand,right_hand}_mm_per_frame2_p{50,95}` | ‖second difference‖ of root-relative joints | mm/frame² | float64 | measured — body p50 = 0.519, hands p50 = 1.270; **activity-confounded at ρ = 0.90** |
| `wrist3d_{side}_speed_mm_per_s_p{50,90}` | ‖Δ wrist‖ × fps | mm/s | float64 | measured — p50 = 41.2 mm/s |
| `wrist3d_{side}_displacement_mm_p{50,95}` | distance from that clip's own median wrist position | mm | float64 | measured — the threshold-free form of rest-pose exit |
| `rest_exit_{side}_frac` | fraction of frames beyond `rest_exit_displacement_mm` (default **100**, a characterization parameter) | fraction | float64 | measured — p50 = 0.0; gesture is bursty |
| `camera_translation_stability` | robust CV of `tz × box_height / frame_height` within the clip | fraction | float64 | measured — within-file median CV 1.489%, max 3.895% |

`wrist_above_hip_frac` was computed and is **rejected as a graded signal**: it is
near-binary (603 of 912 dev spans exactly 1.0, 101 exactly 0.0). Recorded, not
used.

**Renaming, as recommended by M-2.** Session 1's `accel_mm_per_frame2` is
renamed `camera_translation_accel_nominal` and reclassified from a motion feature
to a tracking-quality feature. Its dev median was 119.8 against the genuine
root-relative body figure of 0.519 mm/frame² — a factor of 230. It was measuring
the HMR camera, not the participant.

### 1.4 Tier 3 — MP4/WAV payload

Only used to produce outputs, never to compute a Stage-1 signal: `flac_16k_mono`
(transcode) and the private review clips. **No Tier-3 signal exists in this
spec**, which is what makes the Stage-1 no-media rule achievable.

---

## 2. Gate versus rank

Every gate is a bias we introduce. Each one below states the bias it introduces.

### 2.1 Proposed hard gates (all `evidence: measured`)

| gate | rule | files removed (corpus) | bias introduced |
|---|---|---:|---|
| G1 unreadable media | `has_video_stream == false` **or** `mp4_container_readable == false` | 446 | none identifiable — these files carry no usable video at all |
| G2 silent audio | `wav_size_bytes == 58` | 578 total, 187 of them with readable video | removes participants whose audio capture failed; **unverified** whether that correlates with anything else |
| G3 incomplete bundle | `all_modalities_present == false` | 189 | none identifiable |
| G4 pure-visual activity | `interaction_type == "charades"` | 3,189 files / 75.19 dyad-h | removes the most gesture-dense but *speech-free* activity; correct for a speech-conditioned model, and it must be stated as a deliberate distribution change |
| G5 clip length | rendered/extracted window must fit inside both the media and the annotation arrays | 391 files < 30 s | biases against very short recordings |

G1–G3 and G5 are integrity gates. **G4 is the only content gate**, and it is
justified by the model's conditioning, not by quality.

### 2.2 Rank-only signals

`valid_frac_*` (including guarded), `invalid_run_max_*`, `hand_*_pose_frozen_step_frac`,
`hand_*_pose_exact_zero_frac`, `speaking_frac`, `camera_translation_stability`,
and **all** jitter/acceleration variants.

Two reasons the validity fractions are ranked rather than gated for now:

1. They are **not activity-neutral**: `valid_frac_smplh` correlates ρ = −0.35
   with gesture activity on dev. A hard validity gate is a mild bias against
   animated speakers.
2. No human evidence yet says which validity level reviewers find unusable.
   Pass 2 is exactly the experiment that would justify a cut.

### 2.3 Signals that must never gate, with the measurement that forbids it

| signal family | max |ρ| vs gesture activity | why it cannot gate |
|---|---:|---|
| `jitter2d_hp_*` | **0.903** | rejects the most animated speakers |
| `accel2d_*` | 0.880 | same |
| `accel3d_*` | 0.903 | same |
| `hand_*_pose_temporal_var_mean` | 0.842 | a still hand and a broken hand are indistinguishable |

106 of 135 candidate quality signals exceed |ρ| ≥ 0.50. Residualizing on
rank-transformed 3D wrist speed drops all of them to |ρ| ≤ 0.15. **If any of
these is to be used at all, it must be the residualized variant, and only for
ranking.**

---

## 3. Failure modes with no detector

**Superseded for V00 as of 2026-08-07.** After the Round-2 gallery the reviewer
retracted four of the modes below for V00 specifically: glitchy tracking (absent
except when the participant leaves frame), audio/video desync (not observed),
camera roll (negligible), and unnatural hand motion / noisy body pose (the
reviewer will inspect those manually). After the Round-3 gallery they scrapped
extended mutual silence as well. The table is kept as the record of what was
measured, and it still applies to any vendor other than V00. The three modes now
in scope, with working detectors, are in
`reports/04_failure_mode_detection.md`.

One entry below needs a correction rather than a supersession. Round 3 concluded
that SMPL-H forward-kinematic leg angles are unusable, on the strength of a
confirmed standing participant reading 87.4 degrees of knee flexion. The
*absolute* angles are indeed uncalibrated and that warning stands for anything
that needs a real joint angle. The *ordering* is not: on the reviewer's 29 hand
labels, FK posture separates seated from standing with no overlap and no
selection confound, and it is now FM1's primary evidence. The mechanism behind
both facts is the same — all sixteen betas are pinned to zero, so FK output is
pure pose with a fixed 376.8 mm thigh and 400.6 mm shin, which makes the angles
comparable across files and simultaneously wrong in absolute terms.

| failure mode | status | what we have | what is missing |
|---|---|---|---|
| **Unnatural / implausible hand pose** (PI-named) | **no detector for implausibility; a *static*-hand detector now exists and works** | Hand pose is always present; `hand_avail_frac` is *exactly* `valid_frac_box` on 928/928 dev spans, so availability carries no hand-specific information. `pose_exact_zero_frac` is 0.0 everywhere and `pose_frozen_step_frac` only catches a *held* pose. | A learned or kinematic plausibility model — joint-limit violation, inter-finger penetration, or a pose prior likelihood. Nothing in the release flags an articulated-but-wrong hand. Separately, the reviewer's "hands not moving for the entire conversation" **is** now detectable: whole-file wrist travel normalized by shoulder width separates the two labels at 2.3–4.1× (`vocab_proxies.hand_activity_signals`). |
| **Unstable tracking** (PI-named) | **partial, unvalidated** | `valid_frac_smplh`, `invalid_run_max`, `camera_translation_stability`. | Whether these actually fire on the clips reviewers call unstable. Session 1 already found SMPL-H-invalid spans with visually stable 2D keypoints and no obvious transition, so the mask and perceived instability are known to disagree in at least one direction. |
| **Noisy body pose** (PI-named) | **detector is confounded** | Every jitter/accel variant — all measuring motion (ρ up to 0.903). | A noise measure that is orthogonal to motion. Residualization is the proposed fix and it is arithmetically effective (|ρ| ≤ 0.15), but *no evidence yet* that the residual tracks perceived noisiness. |
| **Audio/video desynchronization** (PI-named) | **no detector, and one candidate now ruled out** | `member_duration_spread_s` (up to 610 s) and `duration_mismatch_s` catch gross container-level mismatch only. VAD never overran the media on any dev span. | Any measure of *within-file* drift. The obvious candidate — cross-correlating mouth opening against the audio envelope — was built and **fails on V00**: inter-ocular distance is 79–126 px, peak correlation median 0.151, argmax lag scattered −4 to +12 frames, and the value is identical between labels (rank-biserial 0.0003). The released whole-body face block cannot resolve lip motion at this framing. Needs a dedicated face tracker or an acoustic-visual model. |
| **Blanked / dropped hand** | **not a real failure mode** | Session 1 synthesized it and found nothing caught it. M-3 shows the release never independently drops a hand: box-invalid zero-fills all 133 points, box-valid always carries complete hands. | Nothing — the mode does not occur in dev. Keep the column as a cheap guard in case it occurs outside dev; do not count it as coverage. |
| **Semantic gesture quality** (is the gesture meaningful?) | **out of scope, no detector** | `interaction_type == grounded_gesture` as a proxy for gesture-dense content (377.44 dyad-h corpus-wide, 83.48 V00). | Any content-level measure. Recorded as an open question, not a gap we intend to close this session. |

Four of the PI's four named failure modes are either undetected or detected only
by a confounded signal. **That is the headline result of Phase 1.**

---

## 4. Stage graph with I/O contracts

```
Stage 0  headers        DONE (M-1)     stat + ffprobe header      → files.parquet, header_shards/*
Stage 1  cheap signals  NPZ + JSON     no MP4, no WAV bytes       → stage1/*.parquet
Stage 2  FK signals     NPZ + model    survivors of Stage 1 only  → stage2/*.parquet
Stage 3  selection      parquet only   ranking + quotas           → manifest.parquet
Stage 4  materialize    WAV → FLAC     accepted windows only      → motion/*.npz + audio/*.flac
```

| stage | reads | must not read | writes | scope |
|---|---|---|---|---|
| 0 | directory entries, `stat`, MP4 container header | any payload bytes | `outputs/02_inventory/**` | all 572,700 files / 129,370 rows |
| **1** | `.npz` Stage-1 arrays, `.json` | **`.mp4` and `.wav` payload bytes** | per-shard parquet | rows surviving G1–G5 |
| 2 | `.npz` SMPL-H arrays, SMPL-H model in place | `.mp4`, `.wav` | per-shard parquet | Stage-1 survivors |
| 3 | parquet only | any source file | one manifest | all Stage-2 rows |
| 4 | `.wav` (transcode), `.npz` (motion arrays) | `.mp4` | FLAC + motion arrays | accepted rows |

Stage 1 obtains `avg_fps` and the raster by **joining Stage 0's output**, not by
re-probing. That is what makes "Stage 1 must not open MP4" achievable without
losing the per-file timing the whole project depends on.

The SMPL-H asset is read in place and never copied. Rendered review clips stay
under git-ignored `artifacts/` at mode `0700`/`0600` and are never written to the
clean output root.

---

## 5. Compute and I/O budget

Measured throughput, from this cluster:

| measurement | 1 worker | 4 | 8 | 16 | source |
|---|---:|---:|---:|---:|---|
| `ffprobe` header probes (files/s) | 2.52 | 7.98 | 19.86 | 31.28 | M-1 benchmark, job 16509071 |
| NPZ Stage-1 array read, cold-ish (MiB/s) | 17.8 | 86.0 | 107.1 | **168.4** | `benchmark_npz_read.py`, seed 20260807 |
| NPZ Stage-1 array read, warm (MiB/s) | 253.5 | 994.5 | 1,546.5 | 1,548.7 | same script, re-read seed 20260805 |

The cold/warm gap is a factor of nine. **All budgets below use the cold-ish
figure**; using the warm one would understate Stage 1 by 9×.

### 5.1 Stage budgets

| stage | bytes to read | measured basis | wall-clock estimate |
|---|---:|---|---|
| 0 headers | container headers only | 31.28 files/s at 16 workers | **68.9 min** in one 16-worker task; realized as a 512×2-worker array at 8 concurrent in under 2 h |
| 0 catalog | `readdir`/`lstat` only | 663.6 paths/s measured | **14.5 min** (realized: 00:14:28) |
| 1 cheap signals, corpus-wide | **1.7365 TiB** (NPZ 1.7281 + JSON 0.0085) | 168.4 MiB/s cold-ish at 16 workers | **3.00 node-hours** |
| 1 cheap signals, V00 only | **0.9771 TiB** | same | **1.69 node-hours** |
| 2 FK | no extra I/O beyond Stage 1's arrays | M-3: 928 spans × 150 frames = 139,200 FK frames in 5:01 wall on 1 GPU + 4 CPU, including model load and all 2D work | **462 FK frames/s** measured. 300 dyad-h = 64.8M frames → **38.9 GPU-hours**, or 2.4 h across 16 GPU tasks |
| 3 selection | parquet only, < 10 GiB | pandas on 129k rows took 6 s in M-1 summarize | **minutes** |
| 4 materialize | WAV read + FLAC write | **unverified** — no transcode benchmark run | must be measured before scheduling |

The Stage-2 figure is a measured rate on a real workload, but note it bundles FK
with all the 2D and hand work in the same loop, so it is a *pipeline* rate and is
conservative as a pure-FK number.

**Stage 4 is the one budget line with no measurement.** A FLAC transcode
benchmark is required before it is scheduled; the estimate would otherwise be
invention.

### 5.2 Stage-1 I/O, and why it is the dominant cost

Stage 1 reads whole NPZ members — Session 1 established that DEFLATE-compressed
NPZ members cannot be true-memory-mapped, so there is no partial-read shortcut.
Measurements are in `outputs/02_inventory/benchmark/npz_read_scaling*.json` and
`reports/02_measurements.md` §4.9b.

Cold-ish scaling is strongly sub-linear — 16 workers give 9.5× one worker, and
8 → 16 buys only +57% — so a single node's NFS path saturates near 170 MiB/s for
this access pattern. **Whether N nodes give N× is unverified**; the benchmark used
threads on one node. Plan Stage 1 as ~3 node-hours corpus-wide and measure the
aggregate before assuming a multi-node speed-up.

By file count the cross-check agrees: 129,370 ÷ 11.26 files/s = 3.19 h against
3.00 h from bytes, a 6% spread.

Two consequences that do not depend on the exact rate:

1. Stage 1 reads **7.65%** of the payload bytes. Reading MP4+WAV instead would be
   **13.1× more bytes** for no additional Stage-1 signal.
2. Restricting to V00 halves Stage 1 again, to 0.9771 TiB.

---

## 6. Slurm execution model under the 1,001-index cap

`MaxArraySize = 1001`, verified via `scontrol`. The model below is the one
already proven by M-1's 512-task array.

**Task manifest with contiguous slices.** A CSV assigns each task index a
half-open row range `[start_row, stop_row_exclusive)` over a fixed, sorted row
order. For 129,370 rows at 512 tasks that is 253 rows per task. Contiguity
matters: rows are sorted by `source_relbase`, so one task's reads land in a small
number of shard directories rather than scattering across the whole NFS tree.

```
tasks     = min(1001, ceil(rows / rows_per_task))
concurrency = %8   (the value M-1 actually ran at)
```

For the corpus-wide Stage 1, 512 tasks × 253 rows at 2 workers each is the
already-validated shape. If a single array is not enough, chained arrays over
disjoint slice files are used — never a `--array` above 1001.

**Set the throttle from task duration, not from politeness.** The Session-2
review render made this concrete: 300 tasks of 8–14 s each at `%8` took 37
minutes of wall clock while the partition had 254 idle CPUs, because the array
spent most of its time waiting for the next scheduling cycle rather than running.
Raising the throttle to 32 mid-flight (`scontrol update JobId=<id>
ArrayTaskThrottle=32`) cut the remainder to a few minutes. A rule of thumb that
follows: when per-task runtime is comparable to the scheduler cycle, the throttle
is the bottleneck, so either raise it or make tasks bigger. Stage-1 tasks at 253
files each are ~90 s and are less exposed, but the throttle should still be
chosen deliberately.

**Restart semantics**, carried over unchanged from the Session-1 harness because
it worked:

- Each task writes exactly one shard parquet plus a `_COMPLETE.json` marker,
  both via write-to-temp-then-`os.replace`, so a killed task leaves no partial
  file that a rerun would mistake for output.
- A marker is honoured only when it matches the current `config_hash`, `git_sha`,
  and `git_dirty == false`. A dirty worktree can produce output but can never
  reuse a marker, because the SHA alone does not identify uncommitted code.
- Rerunning the whole array is therefore idempotent and cheap: completed tasks
  return `skipped`.
- Failures write an error JSON and do **not** write a completion marker, so a
  summarize step that requires all markers fails loudly rather than silently
  summarizing a partial corpus. M-1's summarize does exactly this.
- Every job preflight requires `os.access(source, W_OK, effective_ids=True)` to
  be false and every output path to resolve outside the source root. Session 2
  learned that the parent NFS export is mounted `rw` on real compute nodes, so a
  mount-option check is not a safety property.

---

## 7. Output schema

Default, as specified: **manifest + compact root-relative motion arrays + 16 kHz
mono FLAC. No MP4 copies.**

```
seamless_interaction_clean/
  manifest.parquet                     # one row per accepted window
  motion/<vendor>/<file_id>.npz        # compact arrays, float16
  audio/<vendor>/<file_id>.flac        # 16 kHz mono
  provenance.json                      # config hash, git SHA, model convention
```

`motion/*.npz` per accepted window, all root-relative and translation-free:

| array | shape | dtype |
|---|---|---|
| `pose_axis_angle` | (T, 52, 3) — global_orient + 21 body + 15 + 15 hand | float16 |
| `joints_root_relative_mm` | (T, 62, 3) — 22 body + 30 hand + 10 fingertips | float16 |
| `valid_smplh`, `valid_box` | (T,) | bool |
| `valid_movement` | (T,) or absent | bool |

### 7.1 Byte estimate for a 300-dyad-hour subset

600 participant-hours at 30 fps = **64.8M frames**.

| component | size |
|---|---:|
| `pose_axis_angle`, float16 | 18.83 GiB |
| `joints_root_relative_mm`, float16 | 22.45 GiB |
| FLAC 16 kHz mono (~55% of 16-bit PCM) | 35.41 GiB |
| validity masks + manifest | < 1 GiB |
| **total** | **≈ 77 GiB** |

Storing both pose and joints is redundant — joints are recoverable from pose by
FK — but 22 GiB to avoid re-running FK on every training epoch is a good trade.
Dropping `joints_root_relative_mm` brings the total to ≈ 55 GiB.

Optional 2D keypoints would add 48.16 GiB (133 × 3 float16), more than doubling
the motion payload; **not included by default.**

**Against measured free space: 74.417 TiB available at
`./seamless_interaction_clean`. A 77 GiB subset uses 0.10%.** Storage is not a
constraint. For comparison, copying the source MP4s for the same 600
participant-hours would be **0.35 TiB** at V00's measured 608.8 MiB per
participant-hour — still affordable, but 4.6× the entire no-media output for data
the model never reads.

Server-side quota distinct from `df` remains **unverified**.

---

## 8. Vendor strategy recommendation

**Question: can V00 alone deliver ≥300 dyad-hours at ≥500 distinct participants
after plausible gating?**

**Yes, with large headroom on the all-label variant and adequate headroom on the
naturalistic-only variant.**

### 8.1 The V00 ladder (measured at every step)

| step | files | dyad-hours | participants |
|---|---:|---:|---:|
| all V00 `filelist` rows | 42,932 | 1,485.59 | 1,316 |
| readable media + complete bundle (G1–G3) | 42,826 | 1,485.59 | 1,316 |
| exclude `charades` (G4) | 41,255 | 1,443.91 | 1,316 |
| duration ≥ 30 s (G5) | 41,205 | **1,443.85** | **1,312** |
| × dev-measured mean guarded `smplh∧box` validity (0.9760) | | **≈ 1,409** | 1,312 |

**Naturalistic-only variant**, which the PI prefers because V00 skews improvised
(acted):

| variant | files | dyad-hours | participants | after validity |
|---|---:|---:|---:|---:|
| V00 naturalistic only | 16,500 | **469.46** | **939** | ≈ 458 |
| V00 improvised only | 24,705 | 974.39 | **485** | ≈ 951 |

Both targets are cleared by the naturalistic-only variant: **469 ≥ 300 dyad-hours
and 939 ≥ 500 participants.** Two cautions:

- **The naturalistic-only pool tops out near 470 dyad-hours before gating.** A
  500-dyad-hour target is *not* reachable naturalistic-only: 938.9 available
  participant-hours against 1,000 required is 0.94×. If the target ever rises
  above ~460 dyad-hours, improvised data or a second vendor becomes mandatory.
- **The improvised-only variant fails the participant floor** at 485 < 500. V00's
  improvised data is 2.1× the hours of its naturalistic data but spread over
  roughly half as many people. Volume and diversity point in opposite directions
  here, which is exactly the trade the user and PI should decide.

### 8.2 Why V00 is the recommendation beyond the hour count

| property | V00 | others |
|---|---|---|
| frame rate / raster | **uniform exact 30/1 @ 1080×1920, 100% of files** | V03 has 7 combinations, V01 has 12 |
| `avg` vs container rate disagreement | **0 files** | 273 in V01 (max 6.82 fps), 143 in V03 |
| files with readable video + empty WAV | **0** | 64 V01, 64 V02, 59 V03 |
| truncated MP4 containers | **0** | 7 in V03 |
| placeholder-with-real-audio bundles | **0** | 39 in V03 |
| dev extraction gap | **0** | 9 V02, 52 V03 |
| movement features | **all 42,932 files** | none |
| dev span fully-valid (guarded `smplh∧box`) | **91.96%** | V01 89.13%, V02 90.62%, **V03 76.89%** |
| unreadable bundles | 106 / 42,932 = 0.247% | — |

V00's only defect is the 106 empty bundles with a 708-byte MP4 — and those are
detected reliably by `has_video_stream`.

### 8.3 Recommendation

**Recommend V00 naturalistic-only as the primary subset, with V00 improvised as
a labelled, optional supplement.** The naturalistic pool alone meets a 300-hour
target with 1.56× headroom and nearly doubles the participant floor, and V00's
uniform timing removes the mixed-rate problem rather than managing it. Adding
improvised data is the lever if the hour target rises; it should be a separate,
labelled partition so the acted/spontaneous split stays measurable downstream.

**This is a recommendation, not a decision. The user and PI decide.**

### 8.4 Decision, and what Round 2 changed

**V00 only. Decided by the user on 2026-08-06 after reviewing the Round-1
gallery.** The review found almost no recording issues in V00 — participants
standing, fully in frame, solid backgrounds, correct framing — and only human
issues. That is corroboration rather than agreement: it matches the uniform
exact-30 / 1080×1920 timing, the best dev span validity (91.96% against V03's
76.89%), and the zero counts for truncated containers and empty WAVs.

**Naturalistic versus improvised is still open, and §8.3's recommendation now
looks less safe than when it was written.** A Round-2 measurement over 8,000
eligible V00 files (`reports/02_review_findings.md` §10) found that the user's
own top-named failure mode is substantially more common in naturalistic:

| | naturalistic | improvised |
|---|---:|---:|
| hands static > 50% of the file | **27.5%** | 11.9% |
| hands static > 80% of the file | **7.0%** | 1.7% |
| speaking fraction, median | 0.367 | **0.444** |
| right hand above hips, median fraction of file | 0.609 | **0.867** |
| dyad-hours after G1–G5 | 469 | 974 |
| distinct participants | **939** | 485 — below the 500 floor |

§8.3 preferred naturalistic on spontaneity and diversity, before any
gesture-density measurement existed. Diversity still favours naturalistic by
1.9×; gesture density, speech, and volume now favour improvised by 2–4×; and
improvised alone fails the participant floor. Every framing, roll, and tracking
proxy is indistinguishable between the two, and **no participant in the 8,000-file
pool is sitting** — ankles are tracked in 100% of frames in 100% of files — so
posture is not a differentiator either.

A mixed subset with a per-label quota is the option §8.3 did not consider and is
now the one worth costing. It is not proposed here because it is the user's call
and because Round 2's human review is the evidence that should settle it.

---

## 9. Diversity and bias plan

### 9.1 Per-participant duration cap

The cap must be stated relative to the **shipped subset**, not the pool. At the
pool scale a 1% cap binds on only 2 participants and is meaningless; at a
300-dyad-hour target it is load-bearing.

| variant | target | 1% cap | max supply under cap | participants the cap binds |
|---|---|---:|---:|---:|
| V00 all-label | 300 dyad-h | 6.0 h/participant | 1,777.6 participant-h (**2.96× target**) | 91 |
| V00 all-label | 500 dyad-h | 10.0 h | 2,108.9 (2.11×) | 79 |
| V00 naturalistic | 300 dyad-h | 6.0 h | 937.2 (**1.56×**) | 3 |
| V00 naturalistic | 500 dyad-h | 10.0 h | 938.9 (**0.94× — infeasible**) | 0 |

**Recommended cap: 1% of shipped duration per `(vendor, participant_id)`.** It is
affordable in every feasible configuration and it directly bounds the skew M-1
measured: the median participant contributes 0.87 h while p99 is 20.09 h and the
maximum is 35.60 h.

### 9.2 Participant floor

**Floor: 500 distinct `(vendor, participant_id)` pairs.** Keying on the pair is
not pedantry — 627 raw `participant_id` values collide across vendors, so keying
on the bare ID would report 3,630 participants where there are 4,307, a 16%
overcount of nothing.

A 1% cap mathematically forces at least 100 participants. The 500 floor is the
binding constraint and must be checked, not assumed.

Unresolved: 51 V00 numeric stems have both a bare and a letter-suffixed variant
(`0602` / `0602A`). Whether those are one human is **unverified**. If they are,
the V00 participant count is overstated by up to 51 — still far above the floor,
but the audit should report the figure both ways.

### 9.3 Stratified acceptance-rate audit — must pass before any subset ships

Compute the acceptance rate (accepted duration ÷ eligible duration) within every
cell of:

- `vendor × label × interaction_type`
- `participant_key`, reported as the distribution of per-participant acceptance
- `raster × avg_fps` combination
- decile of `speaking_frac`
- **decile of gesture activity (`wrist3d_either_speed_mm_per_s_p90`)** — the
  audit that matters most, because §2.3 shows the confounded signals would skew
  exactly this axis

Failure condition: any cell with ≥ 1% of eligible duration whose acceptance rate
differs from the global rate by more than a factor the user approves. **The
threshold is deliberately left for the user to set**; this session sets no
selection thresholds.

The gesture-activity decile audit is non-negotiable. With jitter correlating at
ρ = 0.903 with activity, a pipeline that accidentally lets a jitter signal
influence selection would silently produce a low-gesture dataset for a
gesture-generation model, and the acceptance-rate audit is the only check that
would catch it.

---

## 10. Open questions and risks

| # | question / risk | what would resolve it |
|---|---|---|
| 1 | **No signal detects an implausible-but-present hand pose.** | Pass-2 ratings on the hand items, then either a kinematic plausibility check (joint limits, finger interpenetration) or a pose-prior likelihood, validated against those ratings. |
| 2 | **No signal detects within-file audio/video desync.** | An explicit sync probe — e.g. correlating VAD against mouth-landmark motion on a bounded sample. Duration agreement is not alignment. |
| 3 | **Residualized jitter is untested against human judgement.** | Pass-2 ratings on the "noisy body pose" item, correlated against both raw and residualized variants. If neither separates, jitter is dropped entirely. |
| 4 | Whether reviewers find the released validity masks meaningful at all. | Pass 2. Session 1 already found SMPL-H-invalid frames that look fine and Session 2 found the mask is mildly activity-correlated. |
| 5 | Whether V00's 51 bare/letter-suffixed ID pairs are the same human. | Vendor documentation, or a face/voice identity check — which needs its own privacy review before being proposed. |
| 6 | **Stage 4 (FLAC transcode) has no measured throughput.** | A bounded transcode benchmark on ~100 files. Not run this session. |
| 7 | Server-side quota at `./seamless_interaction_clean`, distinct from the 74.417 TiB `df` reading. | A quota client, or a bounded large-write probe with the group's agreement. |
| 8 | Whether the 12 bundles with `.json` + `.mp4` but no `.npz` + `.wav` are recoverable from the adjacent tars. | Inspect the corresponding tar members. 13,200 tars totalling 13.496 TiB exist locally. |
| 9 | Whether the 39 V03 placeholder-video-with-real-audio bundles are usable audio-only. | Out of scope for a body-and-hands model; recorded so it is not rediscovered. |
| 10 | The upstream meaning of `smplh:is_valid` and `movement:is_valid`. | Still **unverified** after two sessions. Would need vendor or paper-author input. M-3 added one clue: SMPL-H-invalid spans freeze the hand pose (median 22% of steps) far more than the body pose. |
| 11 | Whether any modality is repairable by interpolation. | Deliberately unanswered; no interpolation or repair was performed in either session. |

---

## Change log against the Session-2 brief's assumptions

| brief assumption | measured outcome |
|---|---|
| V00 ≈ 1,300 dyad-hours, ≈ 517 naturalistic | **1,485.59** and **494.56** (469.46 after G4/G5) |
| ~4,065 dyad-hours total | **4,047.97** (0.4% agreement) |
| Placeholders detectable by size alone (~261-byte MP4, ~58-byte WAV) | Misses **154** files including **all 106 empty V00 bundles** (708-byte MP4). Use `ffprobe` no-video-stream. |
| ~20–25 TiB total | **22.696 TiB** of extracted payload; 36.192 TiB including tars |
| MP4+WAV ≈ 91% of bytes | **92.35%** — confirmed |
| `flat_hand_mean` guessed `False` | Empirically **`True`** (25.9% lower hand median reprojection error, 7/7 files) |
| 61-file dev gap unexplained | **9 V02 + 52 V03 naturalistic dev bundles never extracted**; zero reverse-direction coverage problems |
| Jitter *might* correlate with activity | It does, at **ρ up to 0.903**, for every variant tested. Flagged; residualization proposed and quantified. |

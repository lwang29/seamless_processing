# Curating a Clean Training Subset from the Seamless Interaction Dataset
### A design document for automated + semi-automated quality filtering for co-speech gesture generation

---

## 0. Scope and framing

**Goal.** Select a high-quality subset of Seamless Interaction (SI) for training a co-speech gesture generation model, using methods that a coding agent can implement, that are reproducible, and that are grounded in existing practice for mocap / pose-estimation / multimodal speech-gesture datasets.

**Three decisions that shape everything below.**

**(1) The unit of curation is a window, not an interaction.** SI interactions are 2–10 minutes long. A single 3-second hand-tracking failure should not discard 8 minutes of good data, and a 6-minute interaction that is 90% clean is not "a clean interaction." So: score overlapping windows (recommend 4 s windows, 1 s hop, at 30 Hz → 120 frames), then merge high-scoring windows into contiguous **clean spans** with a minimum duration (≥ 4 s) using two-threshold hysteresis. Every method below is defined at window level and aggregated upward. This is also what makes the numbers work: you are selecting from ~8,000 participant-hours ≈ 29M frames, and you want the good 10–25% of that, wherever it lives.

**(2) Curation must be provenance-aware.** SI's motion annotations are not mocap. Per the technical report, body pose and global orientation come from **HMR 2.0** run on monocular video; hands are separately detected with **ViTPose** and reconstructed with **HaMeR**, then transformed from hand-centric coordinates into each wrist's frame; and **a single canonical body shape (β = 0) is used for every participant**. Audio is 48 kHz lapel-mic, peak-normalized, then passed through an in-house AEC ("Beryl") to suppress speaker bleed; VAD is Silero; word-aligned transcripts are WhisperX.

**Four further provenance facts with direct curation consequences.**

- **`movement_v4` ships a free occlusion signal.** The December 2025 Imitator v4 rollout added two keys absent from v1: **`is_occluded`** and **`pred_vertices`**. `is_occluded` is a per-frame occlusion flag — precisely the signal B2/B3 below otherwise have to infer from keypoints — and `pred_vertices` gives a mesh to check without needing the licensed SMPL-H body files. Use both where available. Two cautions: the rollout is **incremental**, so availability must be checked on the filesystem per file (not from `filelist.csv` alone), and files lacking v4 must be marked `unknown` rather than `not occluded`, or unflagged files win by default.
- **Source footage is UHD 4K portrait (2,160 × 3,840); the released MP4 is 1080p.** Hand crops are therefore ~4× lower resolution than what HaMeR would have seen upstream, and hand-region resolution is the binding constraint on hand quality. Practical consequence: `hand_bbox_px_area` (from the released keypoints) is a legitimate *predictor* of hand-reconstruction reliability and worth computing, and participants further from camera should be expected to have systematically worse hands — check for this before attributing it to vendor quality.
- **Two small strata have known audio pathologies and are identifiable from metadata.** Over 95% of interactions were recorded **common-room** (both participants in one room); **under 5% were separate-room** with life-size monitors, explicitly as an experiment in speaker-bleed reduction. Separately, a subset used **shotgun mics that were replaced due to high speaker bleed**. Both strata are small, atypical, and worth isolating rather than averaging into corpus-wide audio thresholds — the separate-room subset in particular will have unusually *low* bleed and may also have different gaze/mutual-attention behavior, which matters if you condition on the interlocutor.
- **Do not mistake `preferred_vendors_only` for a quality flag.** The official loader defaults to `preferred_vendors_only=True`, which resolves to `["V00", "V01"]` — and the source docstring says these are "preferred vendors with **smaller file sizes**." It is a bandwidth convenience. Inheriting it as a quality prior would silently restrict the corpus to two sites on no quality evidence at all, while also being the exact confound that makes per-vendor acceptance-rate auditing (H1/H2) uninterpretable. Measure vendor quality; do not assume it.

This provenance determines which checks are informative and which are vacuous:

| Generic mocap-QC check | Verdict on SI | Why |
|---|---|---|
| Bone-length variance over time | **Vacuous** | β = 0 for everyone; the rig is rigid by construction. Limb lengths cannot drift. |
| Marker dropout / occlusion gaps | **Reframed** | No markers; instead use the released `is_valid` masks + box validity + off-frame keypoints. |
| Foot skate / floor penetration | **Weak** | Participants stand, but the report notes lower-body motion is poorly captured and legs are ignored in their own models. Low value. |
| Jitter / acceleration outliers | **High value** | Single-frame monocular regressors are the classic source of long-term jitter. |
| Hand–arm kinematic consistency | **Highest value, SI-specific** | Hands and body come from *different models in different coordinate frames*, stitched in post-processing. This seam is the most likely artifact source and no generic playbook will check it. |
| Depth/scale plausibility | **Medium** | The report explicitly names monocular depth ambiguity as a limitation. |

**(3) Some of the worst noise is documented, not discovered.** The dataset card and report disclose: misaligned interaction start/end timestamps and off-by-one prompt ordering affecting **~10% of interactions after correction attempts**; occasional duplicate/split participant IDs; per-vendor variation in speaker bleed and in whether participants stay in frame; and WhisperX word-timestamp instability — **~87% of interactions contain at least one word whose timestamp-derived duration exceeds 3 SD from the mean**. Note also that Meta's own large-scale QA (human 30 s samples + LLM-on-transcript + VLM-on-video) was tuned for **PII/offensive content recall**, not for motion fidelity. The interactions that survived their filter were never screened for gesture quality. That is precisely the gap this pipeline fills.

**Corpus scale for budgeting.** 4,065 dyadic hours; 64,739 interactions; 5,098 sessions; 4,284 participants; ~129k per-participant files (each interaction yields two). Naturalistic 2,745 h / 47,333 interactions; Improvised 1,320 h / 17,406 interactions. Dev and test splits are tiny (tens of hours) and **participant-disjoint** — calibrate thresholds on dev, never on test.

**Design principles.**
1. **Gate, then score, then diversify, then verify.** Cheap deterministic gates first; expensive models only on survivors.
2. **Store signals, not verdicts.** Persist every per-window metric to Parquet. Thresholds will change; recomputation costs GPU-months.
3. **Repair when repair is safe, reject otherwise.** A fixable 2-frame A/V offset should not cost you an hour of gesture data. A 20-frame hand dropout should.
4. **Nothing is trusted until a downstream training ablation shows it helped.** See §6.

---

## 1. Noise taxonomy

What you are filtering out, why it damages a gesture model specifically, and where it is detectable. Priority reflects damage-per-unit-prevalence for **speech → upper-body + hands** generation.

| # | Failure mode | Why it hurts gesture learning | Primary signal | Priority |
|---|---|---|---|---|
| A1 | Missing / unreadable modality, length mismatch across streams | Silent misalignment of audio and motion indices | File inventory, frame counts vs. duration | **Critical** |
| A2 | Duplicated / frozen video frames, fps drift | Zero-velocity plateaus teach the model to freeze | Frame-hash runs, zero-velocity run length | **Critical** |
| B1 | Tracking invalid / dropped frames | Interpolated garbage, or discontinuities at gap edges | `smplh:is_valid`, `boxes_and_keypoints:is_valid_box`, `movement:is_valid` | **Critical** |
| B2 | Hands out of frame or occluded (by torso, interlocutor, props) | The single most important channel for gesture is fabricated | Wrist keypoint in-frame + confidence, HaMeR detection gaps | **Critical** |
| B3 | Truncation (waist-up framing cropping arms), participant leaves frame | Systematically biased arm poses | Box-to-frame margins, keypoint clipping rate | High |
| B4 | Identity/track switch (the interlocutor is tracked instead) | Motion attributed to the wrong speaker | Box IoU continuity, appearance embedding continuity | High |
| C1 | Jitter (short and long-term) from single-frame regression | Model learns high-frequency noise; generated motion buzzes | Accel (2nd difference), high-frequency PSD ratio | **Critical** |
| C2 | Hand↔arm stitching discontinuity, implausible wrist angles | Hands that "pop" or detach; corrupts the beat-gesture channel | Wrist relative-rotation continuity + anatomical limits | **Critical (SI-specific)** |
| C3 | Left/right hand swap | Mirrored gesture semantics | Cross-lateral consistency test | Medium |
| C4 | Depth / global-orientation flicker | Torso rotation noise dominates the loss | Global orient angular velocity, translation jerk | High |
| D1 | A/V offset (per-file, per-vendor) | Destroys the prosody↔beat relationship the model must learn | SyncNet offset/confidence; mouth-aperture × envelope lag | **Critical** |
| D2 | Interaction boundary errors (~10% of interactions) | Clipped utterances, moderator speech at edges, wrong prompt text | VAD truncation at edges, third-speaker diarization, prompt↔transcript similarity | High |
| D3 | Word-timestamp errors (~87% of interactions have ≥1) | Any word-level or semantic-gesture conditioning is misaligned | Word duration z-scores, forced-realignment disagreement | High (if using text) |
| E1 | Residual speaker bleed / AEC artifacts | Model conditions on the *interlocutor's* speech → learns nothing | Cross-channel envelope correlation, energy ratio during other-only VAD | **Critical** |
| E2 | Overlapped speech | Ambiguous conditioning target | pyannote overlap detection | Medium |
| E3 | Wrong-speaker attribution (visible person isn't speaking) | Direct label noise on the core mapping | TalkNet/Light-ASD active-speaker score vs. own-channel VAD | High |
| F1 | Noise, reverb, clipping, mic handling | Degrades audio features; also predicts ASR failure | Brouhaha SNR/C50, clip rate, SQUIM/DNSMOS | Medium |
| G1 | Near-static listening / low gesture activity | Dominates the data; drives the model to mean-pose collapse | Wrist speed percentiles, motion energy | High |
| G2 | Non-communicative motion: prompt reading, phone/prop handling, hands in pockets, adjusting mic, drinking | Learned as if it were co-speech gesture | VLM QA + hand-near-face/body heuristics | Medium |
| G3 | "Meta time" leakage (out-of-character talk with the moderator) | Off-task behavior mislabeled as prompted interaction | Third-speaker diarization, prompt-mismatch | Medium |
| G4 | Charade / pure-visual-communication activity segments | Gesture *without* speech — poison for speech→gesture | `interactions.csv` activity type | **Cheap and critical** |
| H1 | Per-speaker and per-vendor dominance in the curated set | Gesture style is highly idiosyncratic; a few speakers will hijack the model | Post-selection distribution audit | High |
| H2 | Correlated filtering (quality gates preferentially reject certain appearances/clothing/body types) | Fairness + generalization failure baked into the data | Acceptance rate stratified by participant metadata | High |

**Two "cheap metadata wins" worth doing on day one.** `interactions.csv` carries the activity type and prompt text. (a) The **charades-style pure-visual-communication activity** should be excluded from speech-conditioned training outright — it contains gesture deliberately decoupled from speech (it may still be valuable as a separate pretraining or evaluation set). (b) The **language-grounded gesture game**, in which participants read sentences and gesture on emphasized words, is a deliberately gesture-dense, semantically-aligned subset — it is the highest-yield-per-hour material in the corpus for semantic gesture learning, and it is identifiable from metadata alone with zero compute. Curation is not only subtraction.

---

## 2. Method catalogue

Each method is given as: **intuition → signals → implementation → type → cost → manual load → failure modes & mitigations.** Signal names in `snake_case` are the suggested Parquet column names, so the whole catalogue doubles as a schema spec.

---

### M1. Integrity and metadata gating

**Intuition.** A large fraction of "mysteriously bad" training samples are not subtle model failures but plumbing failures: a missing npz, a video 3 frames shorter than its annotation array, an off-by-one in an index. These are free to detect and catastrophic to miss, because they corrupt alignment silently rather than loudly.

**Signals.**
- Presence of each modality per file id: `has_video`, `has_audio`, `has_smplh`, `has_keypoints`, `has_movement`, `has_transcript`, `has_vad` (cross-check against `assets/filelist.csv` flags such as `has_imitator_movement`).
- `n_frames_video`, `n_frames_smplh`, `n_frames_keypoints`, `audio_duration_s`, `container_fps`, `fps_nominal_vs_measured_delta`.
- `duration_mismatch_s = |n_frames_smplh/30 − audio_duration_s|`.
- Decode health: `decode_error`, `pts_monotonic`, `n_duplicate_frames`, `max_duplicate_run`.
- `nan_frac_smplh`, `inf_frac_smplh`, `pose_param_absmax` (catches un-normalized or corrupted arrays).
- Metadata joins: `vendor_id`, `session_id`, `interaction_id`, `participant_id`, `split`, `label` (improvised/naturalistic), `activity_type`, `ipc_a`, `ipc_b`, plus a **participant-integrity flag** from checking whether one `participant_id` co-occurs with inconsistent appearance embeddings across sessions (the report documents duplicate/split IDs).

**Implementation.** `ffprobe` (JSON output) for container metadata; PyAV or `decord` for frame counts and pts monotonicity; `soundfile`/`torchaudio.info` for audio; `numpy.load` with `mmap_mode='r'` so you read array *shapes* without loading gigabytes. Duplicate-frame detection: decode at low resolution, hash frames with `imagehash.phash` or compute mean absolute frame difference and flag runs below a floor. Write one Parquet row per file id.

**Type.** Rule-based.

**Cost.** Very low compute, but **I/O-bound and therefore the real budget item**: the corpus is tens of TB. Combine M1 with M2/M3/M6 in a *single pass* over each file so you touch the bytes once. On 200 CPU workers this is hours, not days.

**Manual inspection.** Effectively none; spot-check ~20 flagged files to confirm the flags mean what you think.

**Failure modes & mitigations.**
- *Off-by-one frame count mismatches are normal* (container vs. annotation). Don't gate at 0 — measure the empirical distribution of `duration_mismatch_s` on dev first, then gate at something like > 0.2 s.
- *Duplicate-frame detection false-fires on genuinely still people.* Require the run to be long (> ~0.5 s) **and** exactly bit-identical, or corroborate with zero-variance in the pose array (M3).
- *`is_valid` semantics are documented only as "validity flags."* Before relying on them, have the agent characterize them empirically on dev: what fraction of frames are invalid, are invalid runs bursty, and does invalidity visually coincide with occlusion? Write that characterization into the repo as a notebook. Do not assume.

---

### M2. Validity-mask and coverage analysis

**Intuition.** SI ships per-frame validity flags for three independent things (`smplh:is_valid`, `movement:is_valid`, `boxes_and_keypoints:is_valid_box`). This is free supervision about where the extraction pipeline itself lacked confidence. Beyond the raw rate, the *structure* of invalidity matters: 5% invalid frames scattered uniformly is repairable by interpolation; 5% concentrated in one 15-second burst means the person walked out of frame and nothing in that region is trustworthy — including the frames adjacent to it, where trackers typically drift before failing outright.

**Signals (per window and per file).**
- `valid_frac_smplh`, `valid_frac_movement`, `valid_frac_box`, `valid_frac_all` (logical AND across streams).
- `max_invalid_run_frames`, `n_invalid_runs`, `invalid_burstiness` (variance-to-mean of run lengths).
- `dist_to_nearest_invalid_frames` — used to add a **guard band**, discarding N frames (suggest 3–5) on either side of every invalid run.
- Keypoint-derived coverage: `wrist_in_frame_frac_l/r`, `hand_kp_conf_mean_l/r`, `elbow_in_frame_frac`, `shoulder_in_frame_frac`, `box_margin_min_px` (distance from box edge to frame edge), `box_area_frac`.
- `n_persons_detected` per frame (should be 1 in a per-participant crop; > 1 indicates the interlocutor intruding and a track-switch risk).

**Implementation.** Pure `numpy` over the npz arrays: run-length encoding via `np.diff` on the boolean mask, sliding-window aggregation with cumulative sums (`np.cumsum` + differencing) so window features are O(T) not O(T·W). Vectorize the whole file at once; this is the cheapest high-value signal in the pipeline.

**Type.** Rule-based / statistical.

**Cost.** Negligible (seconds per file, CPU). Runs in the same pass as M1.

**Manual inspection.** None beyond the initial semantics characterization.

**Use `movement_v4:is_occluded` as a first-class signal where it exists.** The v4 features add an explicit per-frame occlusion flag, which is strictly better than inferring occlusion from keypoint confidence and frame margins. Add `occluded_frac`, `max_occluded_run_frames`, and `v4_available` (three-valued: yes / no / partial) to the window schema. Two disciplines matter: treat missing v4 as **`unknown`, never as `not occluded`**, or files still awaiting rollout will pass a gate they were never evaluated against; and check the correlation between `is_occluded` and your inferred occlusion signals on dev — if they agree closely, you can drop your inferred version and save compute, and if they disagree, you have learned something about what the flag actually means before it costs you anything.

**Failure modes & mitigations.**
- *Validity flags may be optimistic* — a tracker can be confidently wrong. Never use validity as your only pose-quality signal; it gates, M3–M5 score.
- *Guard bands cost data.* Make the band width a config parameter and include it in the ablation.
- *Two hands are not symmetric in importance.* A window with a reliable right hand and a lost left hand is not half-good; for a model predicting both, it is unusable. Gate on `min(valid_frac_hand_l, valid_frac_hand_r)`, not the mean.

---

### M3. Kinematic plausibility and jitter statistics

**Intuition.** Real human motion is band-limited and smooth in a specific way; monocular per-frame pose regressors are not. The standard, literature-sanctioned measure of this failure is the mean per-joint **acceleration error** (`Accel`, the second temporal difference of joint positions), used across the pose-refinement literature (e.g. SmoothNet and follow-ups) precisely because single-frame estimators trade smoothness for per-frame precision. Windows in the tail of the jitter distribution are windows where the tracker was guessing. Independently, joint angles outside anatomical range or angular velocities beyond human capability indicate outright reconstruction failure rather than mere noise.

**Signals.** Compute joint positions by forward-kinematics on the SMPL-H parameters (β = 0, so one canonical rig for everyone — which conveniently makes these metrics directly comparable across participants, in millimetres).

```
accel_mm_per_frame2   = mean_j mean_t || p[t+1,j] - 2*p[t,j] + p[t-1,j] ||
jerk_p95              = 95th pct over (j,t) of ||third difference||
hf_energy_ratio       = PSD power above 6 Hz / total power, per joint-angle channel, averaged
                        (co-speech gesture energy is concentrated below ~5 Hz; content above
                         that is mostly estimator noise — but see failure modes)
ang_vel_max_rad_s     = max over joints of |Δθ| * 30
joint_limit_violation_frac = fraction of frames with any joint outside a per-joint
                        anatomical range table
global_orient_ang_vel_p95, translation_jerk_p95   (depth/orientation flicker, C4)
zero_velocity_run_max = longest run with per-frame displacement < ε (frozen tracking or
                        duplicated frames)
pose_param_std        = per-window std of pose parameters (detects flatlined output)
```

**Published thresholds you can start from.** The CoCoGesture/GES-X pipeline — 40M frames of SMPL-X co-speech gesture pseudo-labeled from talk-show video, i.e. the closest published analogue to this situation — gates on two explicit rules: reject when **wrist rotation exceeds 150° on any axis**, or when **pose changes by more than 25° between adjacent frames at 15 fps** (≈ **12.5°/frame at SI's 30 Hz**). Critically, on a violation they **discard the surrounding ±150 frames**, not just the offending frame, on the reasoning that a detected spike indicates a locally unreliable tracking regime rather than an isolated glitch. Adopt that quarantine convention: it is the difference between removing a symptom and removing the affected region. Treat both numbers as starting points to re-derive from your own dev percentiles, but note they are literature-sanctioned rather than invented.

**Implementation.** `numpy`/`torch` for differences; `smplx` (which supports SMPL-H) or `aitviewer` for forward kinematics — batch FK on GPU for whole files, it is trivially parallel. `scipy.signal.welch` for PSD. Store per-window percentiles rather than means where possible: means are dominated by rare spikes, and you want to distinguish "one bad frame" from "uniformly noisy."

Robustify by converting each metric to a **z-score within (vendor × participant)** strata before thresholding, because baseline jitter varies with lighting, camera distance, and clothing per site. A raw global threshold silently becomes "reject vendor 3."

**Type.** Statistical (with rule-based anatomical gates).

**Cost.** Low. FK + differences for 29M frames is minutes of GPU time or a few CPU-hours; the cost is dominated by reading the npz files, so fold it into the M1/M2 pass.

**Manual inspection.** Render ~50 clips from the top and bottom jitter deciles as skeleton overlays and confirm your eye agrees with the ranking. Do this before trusting the metric — it takes an hour and catches sign errors, unit errors, and coordinate-frame mistakes.

**Failure modes & mitigations.**
- **The central risk: high-frequency energy is signal, not only noise.** Beat gestures and rapid finger movement have real content in the 3–8 Hz band. Aggressive high-frequency gating will preferentially delete exactly the expressive gestures you want. Mitigations: (i) restrict `hf_energy_ratio` to a *very* high threshold used only as a corruption gate, not a quality ranking; (ii) prefer `accel` outliers over spectral ratios for ranking; (iii) always check the correlation between your jitter score and `gesture_activity` (M6) — if it is strongly positive, your "quality" metric is just penalizing motion, and you must residualize it (regress jitter on activity, keep the residual).
- *The report states their own models applied Savitzky-Golay smoothing to the 6D rotations.* If you smooth before scoring, you measure your filter, not the data. Score on raw released parameters; smooth (if at all) downstream, and record it.
- *Anatomical limit tables are approximate* and SMPL-H joint angle conventions are easy to get wrong. Validate the limit table by confirming < 1% violation rate on the cleanest dev windows; if it fires on everything, the convention is wrong, not the humans.

---

### M4. Cross-estimator agreement (independent re-estimation)

**Intuition.** Without ground truth, the best available proxy for annotation error is **disagreement between independent estimators**. If a second pose model — with a different architecture, training set, and failure profile — reconstructs the same window very differently, at least one of them is wrong, and that window is risky. This is the standard trick behind pseudo-ground-truth filtering in 3D human-pose dataset construction, and it is the only method here that can catch *confidently wrong* tracking, which validity flags and smoothness metrics both miss.

**Signals.**
- `pa_mpjpe_vs_ref_mm`: Procrustes-aligned MPJPE between released SMPL-H joints and an independent estimate (removes scale/rotation/translation nuisance, which matters because β = 0 in the release).
- `pa_mpjpe_hands_mm` computed separately for hand joints — this is where you expect the real disagreement.
- `reproj_err_px`: project released SMPL-H joints into the image with an estimated camera and compare to the released 2D keypoints (or freshly detected ones). A cheap, self-contained consistency check requiring no second 3D model.
- `disagreement_p95` over frames, plus `disagreement_run_max` for burst detection.

**Implementation.** Independent estimators: **SMPLer-X** (whole-body, strong hands, several capacity tiers), **OSX**, **Sapiens** (which SI's own rendering pipeline uses for 2D keypoints, so it is a natural choice for the 2D leg), **RTMW/RTMPose** or **ViTPose** for fast 2D, **MediaPipe Holistic** for a very cheap CPU-only sanity channel. For reprojection you need camera intrinsics; estimate a focal length per file from the bounding-box/head-size prior or solve PnP as the SI rendering pipeline does, and — importantly — treat the *temporal consistency* of the reprojection error as more trustworthy than its absolute value.

Run this on a **stratified sample or on gate survivors only**. Full-corpus whole-body re-estimation at 30 fps is on the order of 8,000 GPU-hours; at 5 fps stride on the 25% that survive Tier-0/1 gating it is ~300 GPU-hours, which is a weekend on 8 GPUs. Frame striding is legitimate here because you are estimating a *window-level* agreement statistic, not producing new annotations.

**Type.** ML-based (ensemble disagreement).

**Cost.** **The most expensive stage in the pipeline.** Schedule it late, on survivors, with striding.

**Manual inspection.** Low-to-medium: inspect ~50 high-disagreement windows to learn what disagreement *means* in this corpus (usually occlusion, unusual arm poses, or hand-detection failures) and whether to gate or merely down-rank.

**Failure modes & mitigations.**
- *Disagreement ≠ error.* Both models can be wrong together (correlated failure on occlusion), and the reference model can be worse than the release on some poses. Use disagreement as a **ranking** signal, and reserve hard gating for extreme values corroborated by another signal (e.g. high disagreement **and** low hand-keypoint confidence).
- *Do not "fix" the release with the reference model.* Mixing two motion sources into one training set introduces a domain seam the generative model will happily learn. Pick one source; use the other only as a critic.
- *Coordinate/convention mismatch will dominate your metric* if unhandled (different joint orders, root definitions, up-axes). Mitigation: verify on a handful of clips that PA-MPJPE on visually-clean windows lands in a plausible range (tens of mm, not hundreds) before running at scale. This single check saves days.

---

### M5. Hand-specific and hand↔arm consistency QC *(SI-specific, highest value)*

**Intuition.** In SI, hands and body come from **different models solved in different coordinate frames** and reconciled by a post-processing transform into each wrist's frame. Any error in that reconciliation produces a hand that is individually plausible and globally wrong — flipped, twisted, or popping between frames. Since hand and forearm configuration carries most of the perceptual weight in co-speech gesture, this seam deserves dedicated checks that no off-the-shelf mocap QC will perform. Additionally, HaMeR depends on a hand *detection* step, so hand data has its own dropout process independent of body tracking.

**Signals.**
```
wrist_rel_rot_geodesic_vel     per-frame geodesic distance between consecutive
                               wrist-relative rotations (radians/frame) — pops appear as
                               isolated spikes against a smooth baseline
wrist_rel_rot_limit_violation  wrist flexion/extension and radial/ulnar deviation outside
                               anatomical range in the forearm frame
hand_detect_frac_l/r           fraction of frames with a hand reconstruction at all
hand_gap_run_max_l/r           longest hand dropout run
finger_pose_std_l/r            near-zero std indicates a flat/default hand pasted in
                               (the report notes the body model alone yields flat hands)
lr_swap_score                  correlation of simultaneous large jumps in left and right
                               hand parameters (a swap flips both at once); plus a
                               cross-lateral test that each wrist stays on its expected
                               side of the torso mid-sagittal plane
hand_arm_dir_consistency       angle between the HaMeR-derived wrist orientation and the
                               forearm direction from the body fit; should be within
                               anatomical range and, crucially, temporally smooth
finger_self_penetration_frac   optional mesh-level check (SDF or vertex-distance based)
```

**Implementation.** `numpy` + `scipy.spatial.transform.Rotation` for geodesic rotation distances and frame conversions; `smplx` FK to obtain the forearm axis; `trimesh` if you want penetration checks (skip initially — expensive, low marginal value). Detect the "flat/default hand" case by comparing per-window finger-parameter variance to a corpus-wide floor.

**Type.** Rule-based + statistical.

**Cost.** Low (CPU, seconds per file). Highest value-per-FLOP in the entire pipeline.

**Manual inspection.** Medium and worthwhile: render 30–50 windows flagged by `wrist_rel_rot_geodesic_vel` and by `lr_swap_score`, because both metrics need threshold calibration against what actually looks wrong. Budget half a day.

**Failure modes & mitigations.**
- *Fast real gestures produce large legitimate rotation velocities.* Distinguish spikes (single-frame, non-physical) from sustained fast motion using the ratio of the frame-to-frame jump to the local median velocity, not the absolute value.
- *Anatomical wrist limits depend on forearm pronation*, which monocular estimation gets wrong often. Set limits generously and treat violations as a ranking signal unless extreme.
- *L/R swap detection has low base rate*, so it will be mostly false positives at any sensitive threshold. Use it to build a small review queue rather than an automatic gate.

---

### M6. Gesture activity and motion informativeness

**Intuition.** "Clean" and "useful" are different axes, and conflating them is the classic curation mistake. A perfectly-tracked person standing still while listening is maximally clean and nearly worthless for a speech→gesture model — worse than worthless, since an abundance of static frames is a known driver of regression-to-the-mean collapse. Existing gesture work filters on exactly this basis: recent in-the-wild co-speech gesture work discards samples with minimal gesture activity using the L2 distance between consecutive-frame body keypoints. You want windows that are both well-tracked **and** gesturally alive, and you want to *know* the marginal distribution of activity in your final set rather than discovering it in your samples.

**Signals.**
```
wrist_speed_mean, wrist_speed_p90, wrist_speed_max      (mm/s, both hands)
motion_energy            mean squared velocity over upper-body joints
gesture_space_frac       fraction of frames with wrist outside a rest-pose sphere
                         (i.e. hands actually raised, not hanging or clasped)
n_gesture_units          count of velocity-minima-bounded strokes (segment the wrist
                         speed signal at local minima below a fraction of its median)
elbow_flex_range         range of elbow flexion within window
hand_near_body_frac      proxy for self-adaptors (scratching, adjusting clothing) and
                         for hands-in-pockets
posture_shift_magnitude  root/shoulder translation range (detects weight shifts, walking)
speaking_frac            fraction of frames inside own-channel VAD (from released `vad/`)
listener_flag            speaking_frac below a threshold
```

**Implementation.** All `numpy` on FK joint positions plus the released VAD JSONL. `scipy.signal.find_peaks` for stroke segmentation. Include Laban-style effort/shape descriptors if you want richer style features later; not required for filtering.

**Type.** Statistical.

**Cost.** Negligible; same pass as M2/M3.

**Manual inspection.** None required, but plot the joint distribution of `wrist_speed_p90` against `accel` early — this is the diagnostic that tells you whether your quality metrics are secretly activity metrics.

**Failure modes & mitigations.**
- **Do not simply threshold out low-activity windows.** Listening behavior and gestural rest are real, and a dyadic model needs them; also, a model trained only on high-energy windows will produce a permanently gesticulating avatar. Correct approach: keep `activity` as a **stratification variable**, and sample a *deliberate, documented* mixture (e.g. 60% speaking-with-gesture, 25% speaking-low-gesture, 15% listening) rather than letting a threshold decide implicitly.
- *Activity is confounded with speaker identity and with the improvised/naturalistic split* (professional actors gesture more). Compute activity percentiles within speaker before using them for selection, or you will select for a handful of animated actors.
- *High activity can be non-communicative* (prop handling, fidgeting). Pair with M12/M13 and the VLM check (M15).


---

### M7. Audio quality scoring

**Intuition.** Audio quality matters twice over: directly, because the model conditions on speech features, and indirectly, because audio quality *predicts the reliability of every other audio-derived annotation*. This is empirically established — the Brouhaha work showed Whisper's word accuracy degrades monotonically with predicted SNR, from roughly 83% of words correct in the top SNR decile down to about 38% in the lowest. So an SNR estimate is simultaneously a quality signal and a prior on transcript trustworthiness. SI audio should be good on average (lapel mics, indoor studios) but the report notes a subset was recorded with shotgun mics that were later replaced due to bleed, and that production quality varies by site.

**Signals.**
```
snr_db_mean, snr_db_p10          Brouhaha frame-level speech-to-noise ratio
c50_db                           Brouhaha room-acoustics / reverberation estimate
clip_frac                        fraction of samples at |x| >= 0.999 (peak normalization
                                 in the release makes clipping detection important)
dc_offset, silence_frac
lufs_integrated, lufs_range      pyloudnorm
bandwidth_hz                     spectral rolloff (detects narrowband or resampled audio)
squim_stoi, squim_pesq, squim_sisdr    torchaudio SQUIM objective estimates
dnsmos_ovrl / nisqa_mos          non-intrusive perceptual quality
spectral_flatness_nonspeech      musical-noise / AEC-artifact proxy: unusual flatness in
                                 non-speech regions is a fingerprint of aggressive
                                 suppression, which the release applied
```

**Implementation.** `pyannote/brouhaha` (via `pyannote.audio` + `brouhaha-vad`) for SNR/C50/VAD in one forward pass; `torchaudio.pipelines.SQUIM_OBJECTIVE` and `SQUIM_SUBJECTIVE`; `librosa`/`scipy` for DSP features; `pyloudnorm` for LUFS. Resample to 16 kHz for the neural models, keep 48 kHz for clipping/DSP. All of these batch well on one GPU.

**Type.** ML-based (neural estimators) + rule-based (DSP).

**Cost.** Low-to-moderate: on the order of 100–300× real time on a single modern GPU for the neural estimators with batching, so ~8,000 audio-hours is tens of GPU-hours. Cheap enough to run on the **full corpus** before pose re-estimation — do it early, since it informs D3/E1 and costs little.

**Manual inspection.** Listen to ~30 clips spanning the SNR and MOS deciles. Half an hour, and it calibrates your thresholds better than any amount of reasoning.

**Failure modes & mitigations.**
- *MOS predictors were trained on telephony/denoising distributions* and can rank clean studio speech oddly. Use them for outlier detection (bottom few percent), not fine-grained ranking.
- *The release already applied AEC*, so "clean-sounding" can mean "heavily processed." Aggressive suppression removes bleed but can also gate the target speaker's low-energy speech onsets, which is precisely where gesture-relevant prosody lives. The `spectral_flatness_nonspeech` and `snr_db_p10` features are your handles on this; consider flagging rather than rejecting, and check whether over-suppressed files correlate with one vendor.
- *SNR estimates on overlapping speech are ill-defined.* Compute audio quality on own-VAD, non-overlap regions only.

---

### M8. Speaker attribution: bleed, overlap, and active-speaker verification

**Intuition.** This is, for a dyadic corpus, **the most damaging form of label noise available**, and it is almost invisible in aggregate statistics. If participant A's channel contains audible bleed from B, or if you window a region where A is silent and B is talking, you train the model to produce A's gestures from B's speech. The mapping you are trying to learn is destroyed at the source, and no amount of pose cleaning helps. SI's own documentation lists speaker bleed as a known per-vendor issue, which means this check should be run on 100% of the corpus, not sampled.

**Signals.**
```
bleed_index          mean own-channel energy during "other-speaks-only" VAD intervals,
                     divided by mean own-channel energy during "own-speaks-only"
                     intervals (dB). This is the single most informative number here and
                     costs almost nothing to compute from the released VAD.
xcorr_env_peak       peak normalized cross-correlation of the two participants' log-energy
                     envelopes, and its lag (a nonzero systematic lag also reveals
                     inter-stream timing offsets, cf. D-class faults)
overlap_frac         pyannote overlapped-speech detection
diar_n_speakers      pyannote diarization speaker count on the own channel; >2 suggests
                     moderator presence or meta-time leakage (see D2/G3)
diar_own_speaker_frac fraction of own-channel speech attributed to the dominant cluster
asd_score_mean       TalkNet-ASD or Light-ASD active-speaker probability on the own video
                     crop, averaged over own-VAD frames
asd_vad_agreement    agreement between asd_score > 0.5 and own-channel VAD — the direct
                     test of "is the visible person the one producing this audio"
```

**Implementation.** `pyannote.audio` 3.x pipelines for diarization and overlap; `TalkNet-ASD` or `Light-ASD` (both have runnable inference repos) on face crops derived from released keypoints/boxes; plain `numpy`/`librosa` for envelopes and the bleed index. Compute the bleed index and envelope cross-correlation **first** — they are nearly free and will likely explain a large fraction of your worst-quality windows before you run any neural model.

**Type.** Rule-based (bleed index, envelope correlation) + ML-based (diarization, overlap, ASD).

**Cost.** Bleed/envelope: negligible. Diarization + overlap: moderate (roughly 30–100× real time per GPU), so tens of GPU-hours corpus-wide. ASD: more expensive because it needs face crops and video decode; run it on gate survivors, or on a stratified sample to validate that the cheap bleed index is a sufficient proxy — if `bleed_index` correlates strongly with `asd_vad_agreement` on the sample, you can skip ASD at scale, and you will have earned that shortcut empirically.

**Manual inspection.** Medium. Listen to ~40 clips spanning the `bleed_index` range with both channels available; this is the one signal where human ears vastly outperform metrics and where getting the threshold right has the largest payoff.

**Failure modes & mitigations.**
- *Backchannels break the bleed index.* Genuine short "mm-hm" responses during the partner's turn look identical to bleed at the energy level. Mitigation: exclude intervals shorter than ~300 ms from the "other-only" set, and cross-check with ASD on a sample.
- *pyannote diarization degrades in low SNR* (documented in the Brouhaha analysis). Condition your interpretation of `diar_n_speakers` on `snr_db_mean`; do not gate on diarization in the low-SNR tail.
- *ASD models trained on movie/broadcast data (AVA) transfer imperfectly* to studio portrait video; the TalkNet authors themselves note out-of-domain difficulty and provide an alternative in-the-wild-trained checkpoint. Validate on a hand-labeled set of ~100 windows before trusting it, and prefer the in-the-wild checkpoint.

---

### M9. Audio-visual synchronization: measure, then *correct*

**Intuition.** Gesture generation is fundamentally a timing task; a systematic 100 ms A/V offset shifts every gesture's phase relative to prosody and puts a ceiling on what the model can learn about beat alignment. Multi-stream, multi-site captures accumulate exactly this kind of drift, and Meta's own VLM QA rubric explicitly asked reviewers to check lip sync, cross-participant video sync, and whether audio matches the correct participant — evidence that they considered it a live risk. The important insight is that **sync error is usually a per-file constant and therefore repairable**: measure the offset, shift the annotation index, keep the data. Discarding it is a waste.

**Signals.**
```
syncnet_offset_frames, syncnet_conf, syncnet_min_dist    per file and per 5 s chunk
sync_offset_stability   std of per-chunk offsets (a constant offset is a fixable clock
                        error; a drifting offset means variable frame rate — reject)
mouth_env_lag_frames, mouth_env_r    cheap proxy: lagged Pearson correlation between a
                        mouth-aperture signal (lip-landmark distance from the released
                        facial keypoints, or FAU jaw/lip channels from the movement
                        features) and the speech amplitude envelope, over lags -15..+15
cross_stream_lag_frames envelope cross-correlation lag between the dyad's two audio
                        channels vs. their video motion onsets (catches per-participant
                        stream misalignment, which matters for dyadic conditioning)
```

**Implementation.** `syncnet_python` (Chung & Zisserman) gives an offset, a min-distance and a confidence per face track — note the confidence is defined as the gap between the median and minimum of the distance curve, so low-confidence results usually mean "not enough speech in this chunk," not "badly synced"; weight or discard low-confidence chunks rather than averaging them in. Modern alternatives (sparse-in-space-and-time sync transformers, VocaLiST) are drop-in replacements if SyncNet underperforms on portrait framing.

**The cheap proxy is worth building first**: the mouth-aperture-vs-envelope lagged correlation requires no new model, only the released keypoints/FAU values and the audio, runs at thousands of times real time on CPU, and gives you a per-file offset estimate whose *agreement with SyncNet on a sample* determines whether you need SyncNet at scale at all.

**Add a gesture-based sync estimator, not only a lip-based one.** SyncNet measures *lip*–audio alignment, which is the right signal for detecting a clock offset but the wrong signal for verifying the relationship you actually care about. **GestSync** (Hegde & Zisserman, BMVC 2023) defines the Gesture-Sync task directly — whether a person's gestures are correlated with their speech — with a dual-encoder trained self-supervised on temporal-offset negatives, released code and checkpoints, and **a keypoint-vector variant that consumes 2D keypoints you already have** (no video decode, so it is cheap enough to run broadly). It gives two things SyncNet cannot:

1. **A body-motion sync offset**, which is robust exactly where lip-based sync fails — occluded, turned-away, or low-resolution faces, all common in a wide portrait shot.
2. **Gesture-based active-speaker detection.** Ask whether this participant's *motion* correlates better with their own audio channel or the partner's. A window that correlates better with the partner is mispaired, bleed-contaminated, or has a track/identity error — a single test that cross-checks E1, E3 and B4 at once, and one of the few checks that is genuinely independent of the two trackers everything else derives from.

Two design consequences the GestSync paper makes explicit and that you should not fight: gesture–speech coupling is **looser and slower** than lip–speech, so it needs **longer offset ranges and longer temporal windows** (5–10 s, aggregated) than SyncNet, and it is **weak for speakers who simply do not gesture much**. That second point is the trap: low confidence means "insufficient gesture evidence," *not* "misaligned." Only treat sync as evidence of a defect when the offset is **confidently non-zero**, and always report sync scores against a gesture-activity (M6) control so you can prove you are not just deleting undemonstrative speakers.

If you would rather not depend on an external checkpoint, the same objective is a few hundred lines and a few GPU-hours in-house: InfoNCE between an audio branch (mel or a frozen WavLM layer) and a motion branch on your own 258-d upper-body representation, negatives shifted by ≥ 1 s. Trained on SI, its scores are calibrated to SI, which matters more than raw accuracy for a curation threshold.

**Type.** ML-based (SyncNet) with a statistical fallback.

**Cost.** SyncNet with face detection/cropping is the second-most-expensive stage (video decode dominates). Restrict to survivors, use ~4 chunks × 5 s per file rather than full-length processing, and cache the offsets. The proxy is nearly free.

**Manual inspection.** Low: confirm on ~20 clips that the estimated offset direction convention matches reality (sign errors here are extremely common and will systematically corrupt everything downstream).

**Failure modes & mitigations.**
- *Sign/convention errors.* Write a unit test: take a known-good clip, artificially shift the audio by +5 frames, confirm the estimator reports +5. This is non-negotiable, cheap, and catches the failure that would otherwise silently ruin the dataset.
- *Silent or listening windows give no sync evidence.* Estimate offset at file level from high-confidence speech chunks only, then apply it to all windows in the file.
- *SI-specific interaction with D2:* correcting sync does not fix interaction-boundary errors, and vice versa. Keep them as separate signals.
- *If offsets are drifting rather than constant*, resampling/VFR is the cause; do not attempt to correct — reject, and check whether the affected files cluster by vendor (if they do, you have found a systematic issue worth documenting).

---

### M10. Transcript verification and re-alignment

**Intuition.** Two known problems compound. First, the released transcripts come from WhisperX, which the report acknowledges is tuned for readable output and therefore drops hesitations, false starts, and disfluencies — behaviors whose *visual correlates are exactly what a gesture model should learn* (self-interruptions co-occur with gesture retractions). Second, the wav2vec alignment step is quantitatively unreliable: ~87% of interactions contain at least one word with a duration more than 3 SD from the mean. If your model uses text or word timings for semantic gesture conditioning, this is a first-order data-quality problem, not a detail.

**Signals.**
```
wer_vs_release           WER/CER between the released transcript and an independent
                         re-transcription (disagreement flags either bad audio or bad text)
whisper_avg_logprob      per-segment mean token log-probability
whisper_no_speech_prob   hallucination / empty-speech indicator
whisper_compression_ratio  >~2.4 is Whisper's classic repetition-hallucination signature
lang_id_conf             non-English or code-switched segments (participants were
                         recruited as native English speakers, so low values are anomalies)
word_dur_z_max, word_dur_z_frac   robust z-score of word duration normalized by
                         character/phoneme count; directly reproduces and quantifies the
                         documented timestamp defect
align_score_mean         forced-alignment likelihood from an independent aligner
align_disagreement_ms    median |Δ onset| between released word onsets and re-aligned ones
disfluency_rate          filled-pause/repair rate from a verbatim-style ASR model
oov_rate, prompt_similarity   cosine similarity between the interaction's prompt text
                         (from interactions.csv) and the transcript — the detector for the
                         documented ~10% prompt/timestamp misalignment
prompt_offby1_flag       transcript matches prompt i±1 better than prompt i, within the
                         session's script order → off-by-one error, and often *repairable*
                         by reassigning the metadata rather than discarding the data
```

**Implementation.** `faster-whisper` (large-v3) or WhisperX for re-transcription with word timestamps; **CrisperWhisper** or a similar verbatim-tuned model to recover disfluencies (the report tried this direction and found overall quality lower — but for *detection* of disfluency-dense regions it is fine even if you do not adopt its transcript); `torchaudio.functional.forced_align` or Montreal Forced Aligner or `ctc-segmentation` for independent alignment; `jiwer` for WER; `sentence-transformers` for prompt↔transcript similarity. Re-alignment is the practical fix: keep the released text, replace the timings.

**Type.** ML-based, with statistical post-processing.

**Cost.** Moderate. `faster-whisper` large-v3 with batching runs at roughly 20–50× real time per GPU, so ~8,000 audio-hours is on the order of 200–400 GPU-hours — acceptable, and it doubles as your word-timing fix. Forced alignment alone is much cheaper and may suffice if you trust the released text.

**Manual inspection.** Low: read ~30 transcript diffs to see whether disagreements are ASR errors, bleed, or genuine disfluency.

**Failure modes & mitigations.**
- *Disagreement is symmetric* — it tells you something is wrong, not which side. Break the tie with `snr_db_mean` (M7): low SNR ⇒ suspect both; high SNR with high WER ⇒ suspect the released transcript or bleed.
- *Whisper hallucinates confidently on long silences and on music/noise*, producing fluent nonsense with good logprobs. `no_speech_prob` and `compression_ratio` are the standard guards; also require agreement with the released VAD.
- *Prompt-similarity is noisy* because prompts describe a stance rather than dictating content. Use it as a weak flag: it works well for detecting off-by-one *ordering* errors (a relative comparison within a session) and poorly for detecting absolute mismatch. Prefer the relative test.

---

### M11. Speech–gesture coupling probes

**Intuition.** All the previous methods check whether each modality is internally clean. This one checks the thing you actually care about: **whether this window contains a real, learnable relationship between speech and motion.** A window can have flawless tracking and pristine audio and still be useless — if the audio belongs to the other person, if the sync is wrong, if the person is silently fidgeting, or if the timestamps are off, the coupling is broken even though every per-modality check passes. Coupling probes catch, in one number, an entire class of faults that unimodal checks miss, and they directly target the quantity the model is trained to fit.

**Signals.**
```
beat_align_score    mean over audio beats of exp(-d_min / sigma), where d_min is the time
                    to the nearest motion beat and sigma ~= 0.1 s. Audio beats: onset
                    envelope peaks (librosa) or syllable nuclei; motion beats: local
                    minima of wrist speed / peaks of acceleration. This is the standard
                    "beat alignment / beat consistency" family from gesture generation
                    evaluation, repurposed here as a *data* metric.
lagged_cca_r        canonical correlation between prosody features (f0, energy, MFCC
                    deltas) and motion features (joint velocities), maximized over lags
                    -0.5..+0.5 s; report peak r and the lag at which it occurs
mi_prosody_motion   mutual information between quantized energy and quantized motion
                    energy (nonparametric, catches nonlinear coupling)
coupling_probe_score  learned: train a small contrastive/binary model to distinguish
                    true (audio, motion) window pairs from temporally-shifted or
                    speaker-swapped pairs; use its per-window confidence on held-out data
                    as the coupling score
```

The learned probe deserves emphasis because it has strong precedent: SyncNet itself was bootstrapped by training on noisy pairs, then using the trained model to reject the false positives in its own training set and retraining. The same self-supervised loop applies directly here — a model that cannot tell your window from a shuffled one is telling you the window carries no speech-gesture information, whatever its per-modality scores say. Implement it as a two-tower encoder (small 1D-CNN or transformer per modality) with an InfoNCE loss over in-batch negatives plus hard negatives from ±0.5–2 s shifts and from the partner's audio.

**Implementation.** `librosa` for onsets/beats; `praat-parselmouth` or `torchaudio` for f0/prosody; `sklearn.cross_decomposition.CCA` (or a lagged-CCA loop) and `sklearn.feature_selection.mutual_info_regression`; PyTorch for the probe. Train the probe on the *provisionally clean* pool from Tier 0–1 so it is not learning from garbage, then score everything — including the pool it was trained on, using cross-fitting (k-fold, score each fold with a model trained on the others) so scores are out-of-sample.

**Type.** Statistical (beat/CCA/MI) and ML-based (learned probe).

**Cost.** Beat/CCA/MI: negligible, CPU. Learned probe: small — a few GPU-hours to train, minutes to score, because it consumes precomputed features rather than raw media.

**Manual inspection.** Medium during development: inspect the lowest-coupling windows that passed all other gates. These are the interesting ones, and they will teach you about failure modes you did not anticipate.

**Failure modes & mitigations.**
- **Circularity is the central danger.** If you filter with a coupling probe and then train a gesture model, you have selected the data your model family finds easy, and your evaluation metrics (which reward exactly that coupling) will improve without the model improving. Mitigations: (i) use a *deliberately different* architecture for the probe than the generator; (ii) never let the probe be the dominant term in the composite score; (iii) validate against human ratings (§5) and against a held-out-by-speaker test set that was **not** coupling-filtered.
- *Coupling is legitimately weak for listeners and for semantic (non-beat) gestures.* Iconic gestures need not align to prosodic beats at all. Apply coupling filters only within the speaking-with-gesture stratum, and never to listening windows.
- *Beat definitions are brittle.* Report which definitions you used and check that the score's distribution shifts as expected under an artificial ±200 ms audio shift — if it does not move, the metric is broken.

---

### M12. Unsupervised outlier detection in feature and latent space

**Intuition.** Handcrafted checks catch failures you anticipated. Outlier detection catches the rest. Corrupted motion is, almost by definition, improbable under the distribution of real motion, so density- or reconstruction-based methods will surface novel corruption types (odd tracking artifacts, unmodeled activities, rare rig failures) without you having to enumerate them first. This is the discovery mechanism of the pipeline.

**Signals.**
```
mahalanobis_robust    robust Mahalanobis distance (MinCovDet) in the space of the
                      handcrafted window features from M2/M3/M5/M6
iforest_score, lof_score        Isolation Forest / Local Outlier Factor (sklearn or PyOD)
pca_recon_err         reconstruction error from PCA fit on provisionally-clean windows
ae_recon_err          temporal-convolutional autoencoder reconstruction error on raw
                      motion (a stronger, distribution-aware version of the above)
vq_recon_err, vq_perplexity     if you train a motion VQ-VAE (the standard tokenizer in
                      current gesture models anyway, so this is nearly free): high
                      reconstruction error or degenerate codebook usage marks windows the
                      motion prior cannot represent
knn_density           distance to k-th nearest neighbor in embedding space; separates
                      "rare and corrupt" from "rare and valuable" when combined with the
                      handcrafted signals
```

**Implementation.** `scikit-learn` (`EllipticEnvelope`, `IsolationForest`, `LocalOutlierFactor`, `PCA`), `PyOD` for a broader sweep including deep methods, `faiss` for kNN at 29M-window scale. Fit on the Tier-0/1 survivor pool; score everything; **cross-fit** to keep scores out-of-sample. Train the AE/VQ-VAE on windows that passed the hard gates only.

**Type.** Statistical / ML-based (unsupervised).

**Cost.** Low. Fitting on a few million window feature vectors is CPU-minutes; the AE/VQ-VAE is a few GPU-hours and you likely want it for the generator anyway.

**Manual inspection.** **This is where manual inspection has the highest return.** Do not review random outliers — review the *medoids of outlier clusters* (see M13). Twenty clusters × 3 examples each = 60 clips, and you will discover essentially every corruption family present in the corpus.

**Failure modes & mitigations.**
- **Outlier ≠ bad. This is the failure mode that quietly destroys datasets.** Rare, vigorous, expressive gestures are outliers under any density model, and they are the most valuable data you have. Never gate on outlier score alone. Use it exclusively to *route to review* or as one term among many, and explicitly check whether high-outlier-score windows correlate with high `wrist_speed_p90` — if they do, you are about to delete all the interesting gestures.
- *Fitting on contaminated data* makes corruption look normal. Iterate: fit, remove the confirmed-bad tail, refit (two rounds is enough).
- *Feature scaling dominates Mahalanobis/kNN.* Use rank or quantile transforms, not raw units.

---

### M13. Cluster-then-inspect: making human QC scale

**Intuition.** The binding constraint on curation quality is human attention, and reviewing randomly sampled clips spends it badly: at a 10% corruption rate, 90% of a random review budget confirms what you already knew. Clustering inverts the economics — embed windows, cluster, review a handful of examples per cluster, and label *clusters* rather than clips. One afternoon of review can then propagate decisions across millions of windows, and, as a bonus, the cluster structure is itself a report on what is in the dataset.

**Signals / artifacts.** Window embeddings (from the M12 autoencoder or VQ-VAE encoder, or from a pretrained motion encoder; optionally concatenate DINOv2 image embeddings of a few frames for appearance/scene context). Then: `cluster_id`, `cluster_size`, `dist_to_medoid`, and — after review — `cluster_verdict ∈ {accept, reject, mixed}` with a free-text `cluster_note`.

**Implementation.** `faiss` k-means or `HDBSCAN` (better here, because it leaves genuine noise unclustered rather than forcing it into a cluster) on L2-normalized embeddings; UMAP for a 2D overview plot for the PI; render medoid + 2 random members per cluster as short skeleton-overlay videos into a static HTML contact sheet. Clusters marked `mixed` get escalated to per-window review with the composite score as the ordering.

**Type.** ML-based (unsupervised) + human-in-the-loop.

**Cost.** Low: k-means/HDBSCAN on a few million 128-d vectors is CPU-minutes to an hour; rendering ~150 short clips is minutes.

**Manual inspection.** ~2–4 hours total for 50 clusters. This is the best-value human time in the whole project, by a wide margin.

**Failure modes & mitigations.**
- *Clusters are dominated by identity and scene, not by quality*, if you embed appearance. Use motion-only embeddings for quality clustering; add appearance only when you specifically want to find site-level artifacts.
- *`mixed` clusters are the norm* at coarse granularity. Over-cluster (k ≈ 200–500) rather than under-cluster; small pure clusters are more actionable than large impure ones.
- *Cluster verdicts are coarse supervision* and will be wrong at the margins. Feed them to M14 as *weak* labels (with a lower weight than per-window labels), not as ground truth.

---

### M14. A learned quality model from a small labeled seed

**Intuition.** After M13 you have a few hundred human-labeled windows and ~60 numeric signals per window. That is exactly the setting where a small gradient-boosted model beats hand-tuned thresholds: it learns the interactions between signals (low SNR matters more when bleed is high; jitter matters less when activity is low), produces a single calibrated score, and — critically — makes the whole notion of "quality" auditable via feature importances and partial dependence, which you can show your PI.

**Signals.** Input = the full per-window signal vector from M1–M12 (plus vendor and activity as categorical features). Output = `p_clean` (calibrated probability) and per-signal SHAP attributions for interpretability.

**Implementation.** `LightGBM` or `XGBoost` with monotonic constraints where the direction is known (higher `valid_frac` should never decrease `p_clean` — this both improves data efficiency and prevents embarrassing non-monotonicities); `sklearn`'s `CalibratedClassifierCV` (isotonic) for calibration; **grouped** cross-validation by participant so you measure generalization to unseen speakers, not memorization of them; `shap` for attribution. Then run **active learning**: label the next batch from the region of maximum uncertainty (p ≈ 0.5) plus a random sample for unbiased evaluation, and iterate twice. `cleanlab` can flag inconsistently-labeled review items — useful when two annotators disagree systematically.

**Type.** Supervised ML.

**Cost.** Trivial (minutes on CPU). The cost is annotation, not compute.

**Manual inspection.** 300–600 labeled windows total across 2–3 active-learning rounds; see §5 for the protocol.

**Failure modes & mitigations.**
- *Small-sample overfitting to vendor or speaker.* Group-wise CV is mandatory; also report performance on a held-out *vendor* to check site transfer.
- *Label noise from annotators.* Dual annotation on an overlap subset, report Krippendorff's α, adjudicate disagreements, and treat α < 0.6 as a signal that your rubric is ambiguous rather than that your annotators are careless.
- *The model inherits and amplifies whatever bias is in the seed labels.* If reviewers unconsciously prefer well-lit, light-clothing, frontal subjects, the model will learn that. Mitigation: stratify the seed sample by vendor and by participant metadata, and audit final acceptance rates by stratum (§7).

---

### M15. VLM / LLM semantic QA

**Intuition.** Some faults are semantic and effectively undetectable with signal processing: the participant is reading the prompt off a card, holding a phone, adjusting a lapel mic, has hands in pockets, has broken character to talk to the moderator, or is gesturing at a prop. A VLM can screen for these at scale. There is direct precedent inside this dataset: Meta ran exactly this architecture — VLM on stacked dyad video with a structured rubric, LLM on transcripts — reporting F1 of about 0.91 for the VLM and 0.84 for the text LLM against human review. Their rubric already covers participant visibility (waist-up, hands/shoulders/head fully visible, frame presence), audio comprehension, recording artifacts, A/V sync, cross-participant video sync, and participant engagement. **Reuse and extend that rubric** rather than inventing one: it is validated on this exact corpus, and reusing it makes your curation comparable to theirs.

**Signals.** Structured key-value output per clip, e.g. `vlm_hands_visible`, `vlm_frame_presence`, `vlm_recording_artifact`, `vlm_av_sync_ok`, `vlm_engagement`, plus additions targeted at gesture: `vlm_holding_object`, `vlm_reading_from_paper`, `vlm_hands_occluded_by_body`, `vlm_off_task_or_meta`, `vlm_gesture_present`, each with a three-way answer (yes/no/unsure) and a one-line justification. From the LLM on transcripts: `llm_off_task`, `llm_moderator_present`, `llm_prompt_mismatch`.

**Implementation.** Any capable VLM over a sampled frame grid (e.g. 8–16 frames per 10 s window, downscaled) with a fixed system prompt encoding the rubric and enforced structured (JSON) output; batch aggressively; cache by content hash. **Retain the "unsure" option** — collapsing it to yes/no destroys the calibration that makes this useful, and the SI rubric preserved it for good reason.

**Type.** ML-based (foundation model, zero-shot).

**Cost.** Moderate and the most schedule-sensitive item, since it is usually API-bound. Sample rather than exhaust: run on ~1–3% of windows stratified across the composite score, use it to measure the *residual* fault rate in your accepted set, and run it fully only on the categories where the cheap signals are weakest (prop handling, reading, off-task). Prefer one call per candidate *span* over one per window.

**Manual inspection.** Validate against ~150 human-labeled clips to estimate precision/recall per category before trusting any category. Some categories will be near-useless; drop them rather than averaging them in.

**Failure modes & mitigations.**
- *VLM temporal reasoning on sparse frame grids is weak* — sync and lag questions in particular are near-impossible from 8 frames. Do not ask a VLM anything you can measure with SyncNet or an envelope correlation.
- *Prompt-sensitivity and drift.* Pin model version and prompt text in the config, hash both into the output rows, and re-validate whenever either changes. Without this, your dataset is not reproducible.
- *Cost blowout* from per-window calls. Enforce a hard call budget in code.

---

### M16. Training-dynamics pruning (optional, last, and with a sharp caveat)

**Intuition.** The data-pruning literature ranks examples by training dynamics — EL2N and GraNd (error/gradient norm early in training), forgetting events, influence — and shows that with a good ranking you can beat power-law scaling in dataset size. Applied here, you would train a small gesture model on the Tier-2 survivors and prune by per-window loss or forgetting.

**Signals.** `el2n`, `forgetting_events`, `final_loss`, `loss_variance_across_epochs`, per window.

**Implementation.** Train a small version of the target model for ~5–10 epochs, log per-window loss each epoch, aggregate.

**Type.** ML-based (supervised, model-dependent).

**Cost.** One extra short training run — moderate, and it must happen *after* everything else.

**Manual inspection.** None directly, but do inspect the high-loss tail: it is the fastest way to find corruption your other 15 methods missed.

**Failure modes & mitigations.**
- **The decisive caveat: under label noise, these metrics invert.** Noisy examples are hard examples, so "keep the hardest" — the standard recipe for large datasets — preferentially keeps corruption. Recent work shows forgetting- and EL2N-based pruning can fail outright under label noise, with *reversing* the ranking recovering most of the loss. Since SI's annotations are pseudo-labels from monocular estimation, you are squarely in the noisy-label regime. Therefore: use training dynamics **only to find noise (prune the high-loss tail)**, never to select hard examples for retention, and only after M1–M15 have removed the gross corruption.
- *Model-specific and therefore non-transferable.* Anything pruned this way is pruned relative to one architecture. Record it as a separate manifest column, not baked into the released subset.
- *Hard-but-valuable confusion.* Cross-tabulate the high-loss tail against your handcrafted signals: high loss + clean signals = valuable rare motion (keep); high loss + dirty signals = corruption (drop).

---

## 3. Turning many signals into one decision

**Step 1 — Hard gates (deterministic, non-negotiable, applied at window level).** Any single failure disqualifies. Keep this list short and defensible, since every gate is a bias you are introducing. Log which gate fired for every rejected window; the histogram of gate firings is one of the most informative artifacts the pipeline produces.

**Step 2 — Robust normalization within strata.** For each continuous signal, convert to a percentile or robust z-score computed **within (vendor × participant)**. This prevents a global threshold from silently becoming a vendor filter or a body-type filter, which is the most common way curation pipelines introduce demographic bias.

**Step 3 — Composite score.** Three options, in increasing sophistication:
- **(a) Weighted sum of normalized signals.** Transparent, easy to explain, arbitrary weights. Fine to start.
- **(b) Rank aggregation (Borda count or mean percentile) across signal families.** Avoids inventing weights, robust to outliers and to differing signal scales. **Recommended default.**
- **(c) The learned `p_clean` from M14.** Best performing once you have ≥300 labels, and interpretable via SHAP. Recommended final.

Do not average correlated signals as if independent — group signals into families (integrity, pose fidelity, hand fidelity, audio, attribution, sync, text, coupling), aggregate *within* family first, then across families. Otherwise the family with the most columns dominates.

**Step 4 — Window → span extraction with hysteresis.** Two thresholds: enter a span when score > `τ_high`, continue while score > `τ_low`, require minimum span length ≥ 4 s and drop spans shorter than that. This yields long, contiguous, trainable segments rather than a shredded mask, and the hysteresis prevents single-window dropouts from fragmenting good material. Snap span boundaries away from invalid runs (M2 guard bands) and away from VAD-truncated edges.

**Step 5 — Diversity-aware selection under a budget.** Do **not** take the top-k spans by score. Top-k selects for the intersection of "well-lit, front-facing, low-motion, fluent, single vendor," which is a much narrower distribution than you want and will produce a model that is excellent at one thing.

Instead, impose quality as a *constraint* and optimize coverage within it:
- Threshold on quality (accept spans above a calibrated operating point), then
- **Maximize a submodular facility-location objective** over motion embeddings to pick the budgeted subset (`apricot` library; greedy is near-optimal and scales), or sample a **k-DPP** (`dppy`) for a probabilistic alternative;
- Subject to explicit quotas: **per-participant cap** (suggest no participant exceeding ~1–2% of total duration — gesture style is strongly idiosyncratic and a few animated speakers will otherwise dominate), per-vendor floor and cap, a deliberate speaking/listening/low-gesture mixture (§M6), and coverage across `ipc_a`/`ipc_b` interpersonal-stance categories and activity types;
- With **participant-disjoint** train/val/test splits that respect SI's existing partition — noting the report's point that because participants appear in multiple sessions, participant-level partitioning is effectively clique-level partitioning. Do not re-split; inherit theirs.

---

## 4. Repair versus rejection

Curation is not only subtraction; some of the most common faults are cheap to fix, and fixing them buys back hours of data. But every repair is an intervention on the training distribution and must be recorded in a column, not applied silently.

| Fault | Repair | When *not* to repair |
|---|---|---|
| Short invalid/missing runs (≤ ~5 frames ≈ 165 ms) | `scipy.interpolate.PchipInterpolator` or SLERP on rotations; keep a `repaired_frac` column | Any run longer than ~5 frames: interpolating a 20-frame hand gap invents a gesture |
| Constant A/V offset | Shift the annotation index by the estimated offset; store `applied_offset_frames` | Drifting offset (VFR) — reject |
| Word-timestamp errors | Re-align with an independent forced aligner, keep released text | If audio SNR is low, timestamps are unrecoverable — drop the text conditioning for that span, not the span |
| Interaction boundary clipping | Extend the window using the adjacent interaction's media from the same session (interactions are contiguous slices of one 1-hour recording, so the material exists) | If the neighboring segment is meta-time, trim instead |
| Off-by-one prompt assignment | Reassign metadata after the relative-similarity test (M10); this *recovers* data the ~10% documented error would otherwise waste | If ambiguous, drop the prompt conditioning but keep the motion+audio |
| Mild jitter | Savitzky-Golay, one-Euro, or SmoothNet-style learned refinement | **Be conservative.** Filters trade smoothness for precision, and large kernels over-smooth; a gesture model trained on over-smoothed motion produces over-smoothed motion, which human raters reliably dislike. Prefer rejecting jittery windows over smoothing them for the *training* subset, and if you do smooth, report before/after jitter and PSD so the effect is visible. |

Rule: **never repair and reject with the same signal**. Decide per fault class, write it in the config, and keep an unrepaired copy of the manifest.

---

## 5. End-to-end pipeline

### 5.1 Stage graph

```
S0  INVENTORY          all files      → files.parquet          (M1)
      ├─ integrity, durations, fps, decode health, metadata join
      └─ GATE-0: drop unreadable / modality-incomplete files
                                       ~1-3% expected loss

S1  CHEAP SIGNALS      all windows    → win_motion.parquet     (M2, M3, M5, M6)
      ├─ single pass over npz: validity, jitter, kinematics, hands, activity
      └─ no video decode, no models. This is where most of the value is.

S2  AUDIO + ATTRIBUTION  all files    → win_audio.parquet      (M7, M8)
      ├─ bleed index & envelope xcorr (free) FIRST
      ├─ Brouhaha SNR/C50, SQUIM/DNSMOS, DSP, clipping
      └─ pyannote diarization + overlap

S3  GATE-1 (hard gates)                → candidate window mask
      ├─ integrity, validity, hand availability, bleed, activity-type exclusions
      └─ expect to retain roughly 30-60% of windows; measure, do not assume

S4  SYNC               survivors      → win_sync.parquet       (M9)
      ├─ cheap mouth-aperture proxy on all; SyncNet on survivors
      └─ estimate per-file offset, APPLY correction, re-index annotations

S5  TEXT               survivors      → win_text.parquet       (M10)
      └─ re-transcribe / re-align, prompt-similarity, off-by-one repair

S6  POSE AGREEMENT     survivors (strided / sampled)  → win_agree.parquet   (M4)
      └─ the expensive stage; strided frames, survivors only

S7  LEARNED SIGNALS    survivors      → win_learned.parquet    (M11, M12, M13)
      ├─ AE/VQ-VAE reconstruction error, outlier scores, embeddings
      ├─ coupling probe (cross-fitted)
      └─ cluster + render contact sheets  ──►  HUMAN: cluster review (2-4 h)

S8  SCORING            survivors      → win_score.parquet      (M14)
      ├─ rank aggregation and/or learned p_clean
      └─  ──►  HUMAN: 300-600 window review, 2 rounds active learning

S9  SPAN EXTRACTION                    → spans.parquet
      └─ hysteresis, min-duration, guard bands, boundary snapping

S10 SELECTION                          → subset_v1.parquet
      ├─ quality constraint + submodular coverage + quotas
      └─ distribution audit (per participant / vendor / activity / demographics)

S11 VLM RESIDUAL QA    1-3% sample    → qa_sample.parquet      (M15)
      └─ estimate residual fault rate in the ACCEPTED set; do not gate on it

S12 VALIDATION                         → training ablations    (§6)
      └─ random vs curated vs full, equal budget
```

### 5.2 Compute budget (order-of-magnitude, for ~8,100 participant-hours)

| Stage | Scope | Hardware | Est. cost | Wall-clock |
|---|---|---|---|---|
| S0 inventory | 129k files | CPU | ~10 CPU-h | minutes on 64 workers |
| S1 cheap motion signals | 100% | CPU | ~200–400 CPU-h | ~1–2 h on 200 workers |
| S2 audio + diarization | 100% | 1–8 GPU | ~250–400 GPU-h | ~1.5–2 days on 8 GPUs |
| S4 sync (proxy) | 100% | CPU | ~50 CPU-h | < 1 h |
| S4 sync (SyncNet, 4×5 s/file) | survivors | GPU | ~100–200 GPU-h | ~1 day on 8 GPUs |
| S5 ASR re-transcription | survivors | GPU | ~100–300 GPU-h | ~1–2 days on 8 GPUs |
| **S6 pose re-estimation** | **survivors, 5 fps stride** | **GPU** | **~300–1,000 GPU-h** | **~2–5 days on 8 GPUs** |
| S7 AE/VQ-VAE + embeddings | survivors | GPU | ~10–30 GPU-h | hours |
| S8 scoring / clustering | all | CPU | ~5 CPU-h | minutes |
| S11 VLM QA | 1–3% sample | API | budget-capped | hours |

**The real bottleneck is I/O, not FLOPs.** The corpus is tens of terabytes; a full-corpus re-estimation of pose at native frame rate would be ~8,000 GPU-hours and is what the staged design exists to avoid. Two practical consequences: (i) touch each file's bytes as few times as possible — S1 and the cheap parts of S2/S4 should be one fused pass; (ii) all downstream stages should read *only* the Parquet signal tables (a few GB total), never the media.

### 5.3 Engineering conventions

- **Orchestration:** Slurm array jobs or Ray for the embarrassingly-parallel per-file stages; a thin DAG (Snakemake, Prefect, or plain Makefile) for stage ordering with per-stage completion markers so restarts are cheap.
- **Storage:** one Parquet table per stage, partitioned by `label/split/vendor`, joined on `(file_id, window_start_frame)`. Never mutate; append versioned tables (`win_motion_v1.parquet`).
- **Idempotency:** every worker writes to a temp path and atomically renames; every row carries `code_version` (git SHA), `config_hash`, and `model_versions` so mixed-provenance rows are detectable.
- **Config:** Hydra or a single YAML with every threshold named and defaulted; thresholds must never appear as literals in code.
- **Determinism:** seed everything; sort file lists; pin library versions in a lockfile; record the checkpoint hash of every model used.
- **A synthetic-corruption test suite — build this early.** Take 20 known-clean spans and programmatically inject each fault from §1: drop frames, duplicate frames, add N-frame audio offsets, swap in the partner's audio, mix bleed at known SNRs, swap left/right hand parameters, inject jitter of known amplitude, freeze a hand for 20 frames. Then assert that the corresponding detector fires and that the others do not. This is the single highest-leverage engineering practice available here: it converts "we think the pipeline works" into a test you can re-run after every change, and it will catch sign errors, unit errors, and off-by-one indexing — the three failure modes most likely to silently ruin the final dataset.

### 5.4 Output artifacts

1. `subset_v1.parquet` — the deliverable: `(file_id, start_frame, end_frame, duration_s, p_clean, all signals, applied_offset_frames, repaired_frac, split, participant_id, vendor_id, activity_type)`.
2. `rejections.parquet` — every rejected window with the gate that fired. Reviewers will ask "what did you throw away and why," and this is the answer.
3. `curation_card.md` — a datasheet: retained hours per split, retention rate by vendor / activity / participant-decile, gate-firing histogram, distribution plots before vs. after, human-review agreement statistics, and every threshold used.
4. A one-page HTML report with the UMAP cluster overview and contact sheets, for the PI.

---

## 6. Manual verification protocol

**Sampling design.** Stratify by composite-score decile (over-sampling the boundary deciles around the intended operating point, where threshold decisions are actually made), and cross-stratify by vendor, by `label` (improvised/naturalistic), and by activity type. Target **300–600 windows** total across 2–3 rounds. Include ~15% duplicated items for reliability estimation and ~10% deliberately-corrupted items from the synthetic suite as attention checks.

**Review interface.** A small Gradio or Streamlit app (or Label Studio if you want persistence for free) showing, per item: the video clip with the SMPL-H skeleton/mesh overlaid, the audio waveform with own- and partner-VAD shaded, the transcript with word boundaries, and a side panel of the computed signals. Keyboard-only rating, autosave to JSONL, ~10–20 s per item. Rendering overlays for 600 clips is minutes of compute; do not skip the overlay, because judging tracking quality from raw video is not possible.

**Rubric** (each 1–5, plus an overall binary):
1. **Body tracking fidelity** — does the skeleton follow the torso and arms without popping or drift?
2. **Hand fidelity** — are both hands present, correctly oriented, and free of pops? *(Rate separately; this is the SI-specific failure surface.)*
3. **Speaker correctness** — is the audio clearly this person, with no audible partner bleed?
4. **Sync** — do gesture beats and lip motion match the audio?
5. **Gesture presence** — is there communicative gesture, versus stillness or non-communicative movement?
6. **Overall: would you train on this?** (binary; this is the M14 target label)

**Two cheap amplifiers of a fixed review budget.**

- **Group-level sampling with group-level rejection.** GES-X reviewed clips at a **10:1 ratio** — one sampled clip per group of ten consecutive clips from the same source video — and **discarded the entire group of ten** when the sample looked jittery or abnormal. This exploits the fact that defects in pseudo-labeled data are strongly correlated within a recording: lighting, distance, clothing, mic placement and vendor practice are constant within a session, so a session is close to a single quality draw. In SI the natural group is the **session** (or the interaction), and the same logic applies with roughly an order-of-magnitude better coverage per hour of human time. Use it for the coarse pass; keep per-window review for threshold calibration, where you specifically need within-session variance.
- **Reuse Meta's own QA rubric verbatim for the recording-level items.** Their VLM/human rubric (Appendix: participant visibility including *hands fully visible throughout*; audio presence and clarity; recording artifacts split into audio (buzz/hum, echo, beeping, clipping) and video (frozen frames, blur/pixelation, over/under-exposure); A/V sync; **inter-participant video sync**) is the only published specification of "bad recording" calibrated on this corpus, and they report **F1 0.91 for the VLM and 0.84 for the transcript LLM** against human labels. Adopting their wording makes your accept/reject decisions comparable to theirs and lets you reuse those F1 numbers as a prior on how much to trust an automated judgment per item. Add one item they never had reason to include — **motion quality on the rendered SMPL-H** — since that artifact was never in scope for their QA and is the whole point of yours.

**Analysis.** Krippendorff's α per item on the overlap subset (target ≥ 0.67 for the binary judgment; below 0.6 means fix the rubric, not the annotators). Adjudicate disagreements with a third rater. Then: plot precision and recall of the automatic score against the human binary label, pick the operating point from that curve given the target subset size, and report the expected residual fault rate at that point. That number — "our accepted subset contains an estimated X% faulty windows, 95% CI [a, b]" — is what makes the curation defensible.

---

## 7. Validating that curation actually helped

Filtering is a hypothesis, not a result. The only way to know it worked is a downstream ablation, and it is cheap relative to the pipeline.

**Design:** train the same gesture model, with identical hyperparameters and identical compute budget, on:
- **(A)** a random subset of size N (the honest baseline, and a notoriously strong one in the pruning literature);
- **(B)** the curated subset of size N;
- **(C)** the full data (if affordable), or a random subset of size 4N;
- **(D)** optionally, the *inverse* — the worst-scoring N — as a sanity check that your score has any signal at all. If (D) is not clearly worse than (A), your score is noise and you should find out now rather than after the paper is written.

**Evaluation on a fixed, participant-disjoint, *un-curated* test set.** This is essential: evaluating on a curated test set measures whether train and test were filtered the same way. Metrics: FGD (with a documented, frozen autoencoder checkpoint — FGD is not comparable across autoencoders), beat-alignment/beat-consistency, diversity, jerk/acceleration statistics of generated motion versus real, and a small human preference study (~20 raters, pairwise, on ~30 clips) because objective gesture metrics correlate imperfectly with perceived quality.

**Then iterate on the curation, not just the model.** If (B) ≈ (A), the filtering is not buying anything and you should find out which signal families carry the effect — by ablating gate families one at a time — rather than shipping a complicated pipeline that does nothing.

---

## 8. Pipeline-level failure modes

These are the ones that damage the project rather than a single method, ordered by how often they actually happen.

1. **Correlated filtering / demographic bias.** Monocular pose estimators fail differentially with skin tone, loose or dark clothing, body size, and hair occlusion. A quality filter therefore *silently selects a subpopulation*. **Mitigation:** compute acceptance rate stratified by every available participant attribute and by vendor; if any stratum's acceptance rate is a large multiple below the median, either relax that gate, add per-stratum quotas, or document the limitation explicitly in the curation card. Run this audit at S10 as a blocking check, not as an afterthought.
2. **Diversity collapse.** Aggressive scoring converges on a narrow "easy" distribution: front-facing, moderate-energy, fluent, single-site. **Mitigation:** the submodular/quota selection of §3 Step 5, plus a before/after comparison of marginal distributions in the curation card.
3. **Threshold overfitting.** Thresholds tuned by looking at examples until they "look right" fit the examples you looked at. **Mitigation:** calibrate on dev, freeze, never touch test; hold out one vendor entirely to check transfer.
4. **Circularity.** Filtering with a model from the same family you will train, then evaluating with metrics that reward that family's inductive bias. **Mitigation:** different architecture for the probe than the generator; a human-rated evaluation; an un-curated test set.
5. **Noise/hardness confusion.** Removing everything hard rather than everything wrong — which for pseudo-labeled data means removing the expressive, fast, occluded-but-real gestures. **Mitigation:** cross-tabulate every quality signal against `gesture_activity`, residualize where correlated, and never gate on outlier score alone.
6. **Silent index corruption.** An off-by-one from a sync correction or a frame-rate mismatch that no metric reports. **Mitigation:** the synthetic-corruption test suite (§5.3), plus a final visual QA pass on 50 randomly chosen *accepted* spans rendered with overlays and audio. Watch those 50 clips personally before training on the subset.
7. **Leakage through the clique structure.** Participants appear in multiple sessions with overlapping partners; naive re-splitting leaks identity. **Mitigation:** inherit SI's participant-level partition; never re-split.
8. **Over-smoothing.** Filtering plus smoothing plus a model that regresses to the mean compounds into a lifeless avatar. **Mitigation:** measure the jitter/PSD of the *training data* and of *generated* motion against real motion, and treat "generated motion is smoother than real" as a bug.
9. **Irreproducibility drift.** A VLM version changes, a checkpoint moves, a threshold gets tweaked in a notebook. **Mitigation:** hash configs, model versions, and code SHA into every row; regenerate the curation card from the manifest rather than writing it by hand.

---

## 9. Suggested starting points

**These are starting values to be calibrated on dev, not recommendations.** Every one should end up in a YAML file and be re-fit against the human labels from §6.

| Signal | Suggested initial gate | Note |
|---|---|---|
| `duration_mismatch_s` | reject > 0.2 s | fit to the empirical distribution first |
| `valid_frac_all` (window) | reject < 0.95 | with 3–5 frame guard bands around invalid runs |
| `max_invalid_run_frames` | reject > 5 | shorter runs are interpolable |
| `min(valid_frac_hand_l, valid_frac_hand_r)` | reject < 0.90 | gate on the *worse* hand |
| `wrist_in_frame_frac` | reject < 0.98 | truncation is systematic bias |
| `accel_mm_per_frame2` | reject above ~99th pct within (vendor × participant) | rank signal, not absolute |
| `wrist_rel_rot_geodesic_vel` spikes | reject > 5× local median | calibrate visually first |
| `joint_limit_violation_frac` | reject > 0.02 | validate the limit table first |
| `bleed_index` | reject > −15 dB | **calibrate by listening**; highest-value threshold in the table |
| `overlap_frac` | flag > 0.15 | flag, don't reject, unless training monadic |
| `snr_db_mean` | flag < 15 dB | quality signal *and* a prior on transcript reliability |
| `clip_frac` | reject > 0.001 | |
| `|syncnet_offset_frames|` | correct if ≤ 6 and `syncnet_conf` high; reject if unstable | correction, not rejection |
| `word_dur_z_max` | flag > 3 | reproduces the documented defect |
| `speaking_frac` | stratify, do not gate | see M6 |
| span length | ≥ 4 s, hysteresis τ_high / τ_low | |
| per-participant share | cap at 1–2% of total duration | |

**Rough expectation for the final subset.** From ~8,100 participant-hours, after integrity and validity gating, hand-availability requirements, bleed and sync filtering, and activity stratification, a plausible landing zone is **several hundred to ~1,500 hours of clean spans**, with the tightest, highest-confidence tier (suitable for a first training run or for evaluation) on the order of **100–300 hours**. Treat these as hypotheses to be measured at S3 and S9, not as targets to be hit — and note that even the pessimistic end is larger than every mocap-based co-speech gesture dataset in the literature combined, so you can afford to be strict.

---

## 10. A prioritized plan for a coding agent

Ordered so that value arrives early and the expensive work is informed by the cheap work.

**Week 1 — instrument, don't filter.**
1. Download the **dev** split for both labels (tens of hours; fits locally). Characterize `is_valid` semantics empirically and write it up. *Acceptance: a notebook showing invalid-run length distributions and a visual check that invalidity coincides with occlusion.*
2. Build the file inventory + manifest schema + Parquet writer + Slurm/Ray harness (S0). *Acceptance: `files.parquet` for dev, all columns populated, restartable.*
3. Implement M2/M3/M5/M6 in one fused pass (S1). *Acceptance: `win_motion.parquet` for dev; distribution plots for every signal; the jitter-vs-activity correlation reported.*
4. Build the **synthetic-corruption test suite** and make it pass. *Acceptance: each injected fault detected by its intended signal, with no cross-firing.*
5. Build the overlay renderer and review app. *Acceptance: 50 dev clips reviewable end to end.*

**Week 2 — audio, attribution, sync.**
6. Bleed index + envelope cross-correlation + DSP audio features (free, high value). Listen to 40 clips and set the bleed threshold. *Acceptance: a calibrated `bleed_index` threshold with a written rationale.*
7. Brouhaha + SQUIM + pyannote overlap/diarization (S2).
8. Mouth-aperture sync proxy on all of dev; SyncNet on a sample; measure their agreement. *Acceptance: a decision, with evidence, on whether SyncNet is needed at corpus scale. Sign-convention unit test passing.*
9. Cheap metadata filters: exclude the pure-visual-communication activity; identify the language-grounded gesture game subset. *Acceptance: hour counts per activity type.*

**Week 3 — human labels, learned scoring, first subset.**
10. Embeddings + AE/VQ-VAE + clustering + contact sheets (S7); run the 2–4 hour cluster review.
11. Human review round 1 (200 windows), fit `p_clean` (M14) with grouped CV, then active-learning round 2 (150 windows). *Acceptance: precision/recall curve against human labels; Krippendorff's α reported.*
12. Span extraction + diversity-aware selection + distribution/bias audit + curation card (S9–S10). *Acceptance: `subset_v1.parquet` on dev, with the bias audit passing.*

**Week 4+ — scale out and prove it.**
13. Run S0–S2 on the full corpus; apply Gate-1; then S4–S6 on survivors only.
14. Run the §7 ablation (A/B/C/D). *Acceptance: FGD, beat alignment, and human preference for curated vs. random at equal budget. This is the deliverable that answers your PI's question.*
15. Only then consider M15 residual QA at scale and M16 training-dynamics pruning.

---

## 11. Two things to check before you freeze any of this

**(1) The GENEA Challenge 2026 runs on exactly this dataset — align with it or knowingly diverge.**

The GENEA Challenge 2026, held with the Interactive Social Agents workshop at ECCV 2026, uses the Seamless Interaction dataset for speech-driven 3D gesture generation. That has three practical consequences for a curation effort starting now.

*It gives you the organizers' subset taxonomy for free.* They describe the corpus as four subsets, with sizes that let you sanity-check your own metadata joins:

| Subset | Hours | Participants | Interactions | Curation note |
|---|---|---|---|---|
| Dyadic conversations (free-form + improvised) | ~3,400 | ~4,000 | ~53,000 | the bulk; the default training pool |
| Grounded / "spoken gesture game" (scripted sentence, emphasized keyword, improvised response) | ~379 | ~3,700 | ~6,400 | **highest semantic yield per hour** — keyword-aligned gesture |
| Collaborative storytelling | ~147 | ~3,100 | ~2,800 | turn-taking dense |
| Silent gesture / charades | ~75 | ~1,900 | ~1,600 | **gesture without speech — exclude from audio-conditioned training** |

*It defines the evaluation tasks your subset will implicitly be judged against*: two core tasks (motion realism, speech–gesture alignment) plus two optional ones (semantic appropriateness for a specific spoken word, evaluated on the grounded subset; dyadic appropriateness/alignment to interlocutor behavior). If semantic appropriateness matters to your PI, the grounded subset is not merely one stratum among four — it is the only place in the corpus where word-level gesture grounding is *designed in*, and a curation policy that filters it at the same rate as free conversation is throwing away the scarcest resource you have.

*It publishes segment-construction conventions worth copying* so your windows are comparable to the benchmark rather than idiosyncratic. The organizers' own evaluation segments are **20 s long, chosen to contain at least 2 speaking turns, with a 2 s buffer at each end so that behavior stays within phrase boundaries.** The 2 s buffer in particular is a cheap, principled answer to the interaction-boundary problem (D2): requiring a speech-free margin at both ends of every retained span removes exactly the truncated-utterance windows that the documented ~10% timestamp misalignment produces.

Their timeline had a soft launch in April 2026 and results in July 2026, so baseline code, data conventions, and possibly preprocessing scripts may already be public. Even if you do not enter, adopting their representation and segmentation makes your ablation numbers legible to reviewers.

**(2) Check whether someone has already published a cleaned subset.**

The dataset repo explicitly solicits community contributions to its preprocessing pipelines, and the challenge has given several groups a strong incentive to build and release curated splits over the past few months. Before building a pipeline that will consume weeks of GPU time, spend an hour on: the repo's issues and pull requests, the HuggingFace dataset discussions, the challenge's baseline repository, and recent papers citing the corpus. Two outcomes are both good — you find a curated split and save most of this work, or you find none and you have a defensible novelty claim for the curation methodology itself. Either way, an existing split is a **baseline to compare against**, which strengthens §7's ablation regardless.

---

## 12. References

**Dataset**
- Agrawal, V. et al. *Seamless Interaction: Dyadic Audiovisual Motion Modeling and Large-Scale Dataset.* arXiv:2506.22554 (2025). See §2.4 (technical setup), §3 (representations), §7.2 (QA), and Appendices A.1.4 (transcript limitations) and A.3 (QA rubric).
- Dataset card and tooling: `huggingface.co/datasets/facebook/seamless-interaction`; `github.com/facebookresearch/seamless_interaction` (including the "Known Limitations and Noise in Metadata" section).

**Pose / motion provenance and refinement**
- Goel, S. et al. *Humans in 4D: Reconstructing and Tracking Humans with Transformers* (HMR 2.0), ICCV 2023.
- Pavlakos, G. et al. *Reconstructing Hands in 3D with Transformers* (HaMeR), CVPR 2024.
- Xu, Y. et al. *ViTPose*, NeurIPS 2022. Khirodkar, R. et al. *Sapiens*, ECCV 2024.
- Zeng, A. et al. *SmoothNet: A Plug-and-Play Network for Refining Human Poses in Videos*, ECCV 2022 — source of the acceleration-error (`Accel`) jitter metric and of the over-smoothing caveat.
- Cai, Z. et al. *SMPLer-X: Scaling Up Expressive Human Pose and Shape Estimation*, NeurIPS 2023 — suggested independent estimator for cross-model agreement.
- Loper, M. et al. *SMPL* (2015); Romero, J. et al. *Embodied Hands / MANO* (2017).

**Audio, speaker attribution, synchronization**
- Lavechin, M. et al. *Brouhaha: multi-task training for VAD, speech-to-noise ratio, and C50 room acoustics estimation*, ASRU 2023 — SNR/C50 estimation, and the demonstration that predicted SNR tracks ASR reliability.
- Bredin, H. et al. *pyannote.audio*, ICASSP 2020 — diarization and overlapped-speech detection.
- Tao, R. et al. *Is Someone Speaking? Exploring Long-term Temporal Features for Audio-visual Active Speaker Detection* (TalkNet), ACM MM 2021; Liao, J. et al. *A Light Weight Model for Active Speaker Detection*, CVPR 2023.
- Chung, J. S. & Zisserman, A. *Out of time: automated lip sync in the wild* (SyncNet), ACCV 2016 — offset/confidence definition, and the precedent of using a sync model to clean its own training data.
- Bain, M. et al. *WhisperX: Time-Accurate Speech Transcription of Long-Form Audio*, Interspeech 2023 (the released transcripts' source). Wagner, L. et al. *CrisperWhisper*, 2024 (verbatim/disfluency transcription).
- Silero VAD (2021).

**Gesture datasets, filtering practice, and evaluation**
- Ginosar, S. et al. *Learning Individual Styles of Conversational Gesture* (Speech2Gesture), CVPR 2019 — pseudo-label pose extraction and confidence-based filtering.
- Ahuja, C. et al. PATS dataset. Yoon, Y. et al. *Speech Gesture Generation from the Trimodal Context of Text, Audio, and Speaker Identity*, SIGGRAPH Asia 2020 — source of FGD.
- Liu, H. et al. *BEAT* (ECCV 2022) and *EMAGE / BEAT2* (CVPR 2024) — beat-alignment metrics and the pseudo-label vs. mocap accuracy comparison.
- Hegde, S. et al. *Understanding Co-speech Gestures in-the-wild* (JEGAL), ICCV 2025 — filtering samples with minimal gesture activity via consecutive-frame keypoint distance; WhisperX for word alignment; tri-modal gesture/speech/text embeddings usable as a semantic-relevance signal.
- **Hegde, S. & Zisserman, A. *GestSync: Determining who is speaking without a talking head*, BMVC 2023** — the Gesture-Sync task; self-supervised dual encoder with ≥1 s offset negatives; keypoint-vector and RGB variants; sync-offset prediction and gesture-based active-speaker detection without face input. Code and checkpoints: `github.com/Sindhu-Hegde/gestsync`.
- **Zhu, X. et al. *CoCoGesture / GES-X*, arXiv:2405.16874** — 40M-frame pseudo-labeled SMPL-X gesture corpus; source of the concrete wrist-angle gates (>150° absolute, >25°/frame at 15 fps), the ±150-frame quarantine convention, and the 10:1 group-level manual-inspection protocol with group-level rejection.
- *GES-Inter*, ICLR 2025 — concurrent-speaker curation, speaker separation, identity-consistent audio–pose alignment for dyadic gesture data.
- Kucherenko, T. et al., GENEA Challenge reports (2020, 2022, 2023) — evaluation methodology and the limits of objective gesture metrics; **GENEA Challenge 2026** (ECCV 2026, Interactive Social Agents workshop) runs on Seamless Interaction and defines the subset taxonomy, evaluation tasks, and 20 s / 2 s-buffer segment convention referenced in §11.
- Yamauchi, K. et al., arXiv:2601.12254 (2026) — non-intrusive MOS metrics fail to detect content-level damage from generative speech enhancement; confidence-based filtering does better. Directly relevant because SI audio is already AEC-processed.

**Data selection, pruning, and quality modeling**
- Sorscher, B. et al. *Beyond neural scaling laws: beating power law scaling via data pruning*, NeurIPS 2022.
- Paul, M. et al. *Deep Learning on a Data Diet* (EL2N/GraNd), NeurIPS 2021; Toneva, M. et al. *An Empirical Study of Example Forgetting*, ICLR 2019.
- Recent analyses showing forgetting/EL2N pruning degrades under label noise (and that reversing the ranking recovers performance) — the basis for the M16 caveat.
- Northcutt, C. et al. *Confident Learning* / `cleanlab` — annotation-error detection.
- Sener, O. & Savarese, S. *Active Learning for CNNs: A Core-Set Approach*, ICLR 2018; Schreiber, J. et al. `apricot` (submodular selection); `dppy` (k-DPPs).
- Gebru, T. et al. *Datasheets for Datasets* — the model for the curation card.

**Libraries**
`numpy` · `scipy` · `pandas`/`pyarrow` · `scikit-learn` · `PyOD` · `faiss` · `hdbscan` · `umap-learn` · `lightgbm` · `shap` · `torch` · `smplx` · `PyAV`/`decord` · `ffmpeg`/`ffprobe` · `librosa` · `soundfile` · `torchaudio` (SQUIM, forced alignment) · `pyloudnorm` · `pyannote.audio` · `brouhaha-vad` · `faster-whisper` · `whisperx` · `jiwer` · `sentence-transformers` · `syncnet_python` · `apricot-select` · `cleanlab` · `gradio`/`streamlit` · `ray`/`snakemake` · `hydra-core`
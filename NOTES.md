# Session 1 reconnaissance notes

This file records surprises, dead ends, and open questions encountered while
measuring the dev splits. Observations are not filtering decisions.

## Safety and scope

- `./seamless_interaction` resolves to `/simurgh2/datasets/seamless_interaction`.
- The source is mounted NFSv4 with `ro`; no source-data writes are permitted or
  attempted.
- Current host at session start: `simurgh2.stanford.edu`.
- Scope is limited to improvised/dev and naturalistic/dev, with a planned shared
  sample of 120 participant-files.

## Surprises and contradictions

- The dev data are stored in numeric shard directories alongside tar archives.
  Each observed participant basename has `.json`, `.mp4`, `.npz`, and `.wav`
  siblings. This contradicts the design document's assumed flat modality
  directories and separate transcript/VAD JSONL layout.
- Only improvised dev retains adjacent tar copies (116 archives, 127.369 GiB).
  Counting both trees would nearly double improvised storage.
- Transcript and VAD records are embedded together in each participant JSON.
  No official metadata CSV or JSONL is staged locally.
- NPZ has two profiles: nine core keys everywhere, plus 15 movement keys in V00
  only in the bounded union. `movement:is_valid` is `(N,1) float32`, whereas the
  SMPL-H and box flags are `(N,) bool`.
- The three validity masks encode distinct failure modes and cannot be treated
  as interchangeable votes. Movement invalidity occurs in exact 60-frame-grid
  blocks; its frame-aligned features are almost entirely repeated zero-fill,
  including long blocks with no visible discontinuity in the participant,
  SMPL-H, or released 2D tracking.
- Box-invalid frames use exact zero-fill for the box and all 133 keypoints. In
  both observed files, the same frames carry a huge finite SMPL-H translation
  sentinel instead of zero or NaN. The visual examples correspond to the
  participant leaving frame and a terminal cut to black, including a one-frame
  optimistic-valid edge at the black cut.
- SMPL-H-invalid pose and translation are instead finite, nonzero, and usually
  varying; they are not a simple last-valid-frame hold. Visual examples range
  from possible limb truncation to fully visible participants with stable 2D
  keypoints and no obvious transition, so the mask's precise pipeline meaning
  remains unverified.
- No inspected file contains `movement_v4:is_occluded`, `pred_vertices`, betas,
  3D joints, units, camera intrinsics, or model-version provenance.
- DEFLATE-compressed NPZ members cannot be true-memory-mapped with
  `numpy.load(..., mmap_mode="r")`.
- Two sampled V01 bundles are structurally present but empty: zero-frame core
  arrays, 261-byte zero-stream MP4, 58-byte WAV header, and empty JSON lists.
- Released video is not uniformly 1080p or 30 fps. The bounded union contains
  1080x1920, 2160x2160, 2160x3840, and 3840x2160 video, with rates including
  30, 30000/1001, and other near-30 averages.
- Computing duration mismatch with an assumed 30 fps creates vendor-correlated
  mismatches up to 0.6 s; using probed average fps reduces the 99th percentile
  from about 0.51 s to about 0.04 s.
- The source NFS first-pass read rate was 26 MiB/s for one worker and 114 MiB/s
  aggregate for four; an immediate warm repeat exceeded 2 GiB/s, so cache state
  dominates small benchmarks.
- The repository export reports only about 11 GiB available, likely insufficient
  for a media-bearing curated subset.
- The payload supplies neither a body model nor metric/unit metadata. The
  requested acceleration column can only be a clearly labeled provisional root
  translation statistic in Session 1; metric joint FK is not defensible.
- Rotation vectors are axis-angle but are not canonically wrapped. Raw adjacent
  subtraction produces approximately 2-pi spikes that become small under
  rotation composition/geodesic distance.
- A clean 120-file harness run produced 24,472 full windows. Movement absence
  makes `valid_frac_all` unknown on 12,708 non-V00 rows; it is never silently
  treated as valid.
- Strict detector orthogonality is impossible for a dropped or duplicated
  temporal sample because the fault also changes local acceleration. The test
  suite records that expected coupling explicitly.

## Dead ends

- Normal `git init` cannot create `./.git`: the execution platform mounts that
  path as an empty read-only tmpfs. A functional repository was initialized in
  `./.git-session` and is operated with `git --git-dir=.git-session --work-tree=.`.

## Open questions

- Whether the missing official metadata CSVs can be obtained from an existing
  local checkout or must be downloaded separately.
- Whether the tar copies and extracted shard directories are byte-identical
  across every member; two deterministic spot checks agree in names and sizes.
- What physical units, origin, and axes `smplh:translation` uses. The staged
  files do not encode this, so a millimetre conversion remains unverified.
- Where an appropriately licensed SMPL-H model file is stored, if anywhere.
  The Python package exists but the body-model assets were not found.
- Whether V00-only movement availability is intended release versioning or an
  incomplete staging artifact.
- Which upstream checks produce `smplh:is_valid` and `movement:is_valid`, and
  why many flagged transitions have no visible 2D or video failure.
- The observed zero/sentinel encodings are not motion samples, while finite
  SMPL-H-invalid values are not proven trustworthy. This rules out assuming one
  universal interpolation behavior; whether any modality is repairable, and
  how, remains a later decision. No interpolation or repair was performed in
  this measurement session.
- Where the eventual media-bearing clean dataset can be written: the current
  repository export is too small, and exact filesystem quota is unavailable.
- Whether later windows remain on a nominal 30-Hz annotation grid or are given
  an explicit mixed-rate timing policy. This session made no resampling choice.

# Session 2 notes

## Resolved inputs and environment

- The repository was renamed externally from `seamless_pipeline` to
  `seamless_processing`; the Session-1 `.git-session` history is intact. A
  compatibility symlink at the old path is required because the execution
  sandbox still designates that path as its writable root.
- The clean-output symlink resolves to `/simurgh2/datasets/seamless_interaction_clean`,
  not the nearly-full `/simurgh/group` export. It reported about 74.4 TiB
  available and accepted a zero-byte write probe, which was immediately
  removed.
- The neutral SMPL-H model is present and imports through the existing ViBES
  environment. It is ignored by Git and must never be copied or committed.
- A five-second batch job on `simurgh1` resolved and received HTTP 200 responses
  from PyPI, Hugging Face, and Anaconda endpoints. No package or checkpoint was
  downloaded.
- Source protection is permission-based on real Slurm nodes, not mount-based:
  the parent export is `rw`, while the dataset directory is mode `2755` and is
  not effectively writable by this user. A conservative M-1 preflight exposed
  this discrepancy and aborted before probing files. Future safety checks must
  record the mount but enforce `os.access(..., W_OK, effective_ids=True) ==
  false` plus output-path separation.

## Open Session-2 questions

- The clean export's server-side quota, if distinct from the roughly 75 TiB
  shown by `df`, remains unverified.
- The interaction join key and activity-hour accounting must be demonstrated
  from file IDs plus `interactions.csv`; prompt-table category counts alone are
  not file counts.
- No rubric should be drafted until the user and PI complete the free-text
  exploratory pass. This is a deliberate human-input boundary, not missing
  implementation work.

## Surprises found while completing Session 2

- The brief's empty-placeholder size signature (261-byte MP4 + 58-byte WAV) is
  incomplete. It finds 285 bundles; `ffprobe` finds 439 files with no video
  stream. The 154 it misses include **all 106 empty V00 bundles**, which use a
  708-byte MP4. Three further signature classes exist, one of which pairs a
  261-byte MP4 with a full-size real WAV and transcript. Emptiness detection must
  be `ffprobe`-status based, not size based.
- Two failure classes are invisible to any size check: 7 V03 files are truncated
  MP4 containers (`moov atom not found`) at 65–286 MB, and 187 files pair a
  readable video stream with a 58-byte empty WAV. Zero of either are in V00.
- `participant_id` is unique only within a vendor. 627 IDs appear under more than
  one vendor, so counting bare IDs undercounts participants by 16% (3,630 vs
  4,307 correct pairs). All diversity quotas must key on
  `(vendor_id, participant_id)`.
- Within V00, 51 numeric stems have both a bare and a letter-suffixed variant
  (`0602` and `0602A`). Whether those are the same human is unverified.
- V02 contains no improvised data at all.
- The review renderer had a real defect: overlays were drawn at native resolution
  and the panel then downscaled by 5–10×, so 2–3-pixel skeleton lines rendered at
  0.30–0.53 output pixels and were averaged away. Panels A and B could not answer
  the question they exist for on V03, a third of the corpus. Found by extracting a
  frame and looking at it. Thickness now pre-compensates the downscale.
- The NPZ read benchmark's accidental warm re-read was more informative than the
  intended measurement: 168.4 MiB/s cold-ish versus 1,548.7 MiB/s warm at 16
  workers, a factor of nine. Any Stage-1 budget built on a warm number is wrong
  by that factor.
- Per-hand availability computed from released 2D keypoints is **exactly** equal
  to `valid_frac_box` on all 928 dev spans. The release never drops a hand
  independently, so the new hand-availability column adds no coverage today; it
  is kept as a cheap guard, not as new information.
- Every jitter and acceleration variant tested is confounded with gesture
  activity, up to rho = 0.903, including the high-pass form specifically designed
  to suppress smooth motion and the box-height-normalized form. 106 of 135
  candidate quality signals exceed 0.50. As a gate any of them would
  preferentially reject the most animated speakers.
- Naive axis-angle deltas exceed 100 rad/s on 43.5% of dev spans, with the naive
  maximum up to 813× the geodesic p99. Session 1's warning was an
  understatement.
- Session 1's provisional `accel_mm_per_frame2` had dev median 119.8 where the
  genuine root-relative body figure is 0.519 mm/frame^2 — a factor of 230. It was
  measuring the HMR camera.
- SMPL-H-invalid spans freeze the *hand* pose far more than the body pose: median
  22% of adjacent frame pairs identical (up to 99.3%) against exactly 0% on valid
  spans, with hand-pose temporal variance collapsing by three orders of
  magnitude. This refines Session 1's "not a simple last-valid-frame hold".

## Open questions added while completing Session 2

- Whether multi-node Stage-1 NFS throughput scales beyond the single-node
  ~170 MiB/s ceiling measured here. The benchmark used threads on one node.
- Stage 4's FLAC transcode throughput is unmeasured; its budget line in the
  pipeline spec is explicitly marked as having no measurement.
- Whether the 12 bundles holding `.json` + `.mp4` but no `.npz` + `.wav` are
  recoverable from the 13,200 local tar archives.
- Whether reviewers can judge hand detail in a 216-px-wide portrait panel, or
  whether a box-cropped Panel A variant is needed.

## Round-15 findings (the corpus-wide briefing gallery)

- One page for all four vendors on a fresh 1,500-per-vendor draw, seed 20270420,
  disjoint from every sample any threshold was fitted on: **19.7% kept, 781 of
  3,961 eligible dyad-hours**. V00 30.3%, V01 16.6%, V02 18.1%, V03 10.7%.
- **A dash and a zero mean different things in the rates table.** A check
  switched off for a vendor records `False`, not null, so a naive rate prints
  "0.0%" and implies it looked and found nothing. It did not look. The table now
  reads the reason column, prints a dash for a retired check, and averages the
  totals row only over the vendors a check is applied to.
- **The table misalignment was header alignment, not cell count.** Header and
  body are both nine cells; the numeric headers were left-aligned over
  right-aligned numbers.
- **Two real defects surfaced during the build.** A file with 11,600 annotation
  frames and 11,599 stored frames is *inside* the timebase tolerance, so it takes
  the fast sequential decode -- and a clip reaching the very end ran out one frame
  short and failed the whole render. Fixed twice over: the sampler now bounds the
  start by the container frame count as well as by duration, and the renderer
  holds the last frame for a shortfall of one or two rather than aborting.
- Six strata are empty by construction (V02 has no improvised material; FM0 finds
  nothing in V00 or V02; FM1 is off for V01 and V02; FM4 is off for V00) and two
  more are short because their whole population is. Both are stated on the page.
- `find -perm /022` reports 246 hits in the gallery, all of them **symlinks**. A
  symlink's own mode is always `lrwxrwxrwx` and is never honoured; the targets are
  `-r--r--r--`. Worth remembering before treating that count as a leak.

## Round-15 findings (the corpus-wide briefing gallery)

- One page for all four vendors on a fresh 1,500-per-vendor draw, seed 20270420,
  disjoint from every sample any threshold was fitted on: **18.9% kept, 782 of
  3,961 eligible dyad-hours**. V00 30.3%, V01 16.6%, V02 18.1%, V03 10.7%.
- **The headline stat tiles have had the number and the caption swapped in every
  briefing gallery ever built**, including the one already sent to the PI. The
  renderer unpacked `(number, label)` as `(label, number)`, so the caption
  rendered 1.35rem bold blue and the figure 0.78rem grey. Found only by an
  adversarial review pass; three human-equivalent read-throughs had missed it.
- **A rendered sidecar carries a copy of `signals_json` frozen at render time**,
  and `{**record, **rendered}` let it win the merge. Any signal added to a sampler
  after its clips were rendered therefore vanishes silently. This is a general
  bug in every builder in the repo. The manifest owns the signals; the sidecar
  owns the render outcome.
- **A rate computed over all files and the same rate computed over the vendors a
  check is applied to are different numbers, and both were on the page** -- FM1
  5.6% against 11.3%, FM4 1.0% against 1.3%. Any per-check figure must state its
  denominator when a check is per-vendor.
- **`participant_id` is numbered within a vendor, not across the corpus.**
  Counting the bare id merges different people who share a number: 2,226 distinct
  ids against 2,489 distinct (vendor, participant) pairs, an 11.8% undercount.
- A file with 11,600 annotation frames and 11,599 stored frames is *inside* the
  timebase tolerance, so it takes the fast sequential decode, and a clip reaching
  the very end ran out one frame short and failed the render. Fixed twice: the
  sampler bounds the start by the container count, and the renderer holds the
  last frame for a shortfall of one or two.
- `find -perm /022` reports 246 hits in a gallery, all of them **symlinks**. A
  symlink's own mode is always `lrwxrwxrwx` and is never honoured; the targets are
  `-r--r--r--`. Not a leak.
- **Adversarial review of a finished artifact is worth its cost.** Four checkers
  raised 32 candidates on a page already verified by hand; 17 survived a second
  agent trying to refute each, giving twelve distinct real defects. Three were in
  prose and were the ones most likely to mislead: the FM1 card quoted the
  90-label result while attributing it to the 164-label set (and omitted the
  standing cost, the number that round revised twice); the FM0 section blurb gave
  a body-fit reason for twelve V03 clips whose actual reason is far-field audio
  and whose body fits are fine; and figures borrowed from earlier rounds were
  unattributed, one of them reworded from "1,379 square-pixel files" to "1,379
  usable files", which the page's own table contradicts.
- **Quoting your own earlier reports is where stale numbers get in.** Every
  hard-coded figure in a hand-written explainer is a claim that nothing
  re-derives, so it survives every later revision of the thing it describes.
  Attribute them to their round or recompute them.

## Round-14 findings (FM4 gates on dynamics, not level)

- **Two candidate audio rules died on 48 hand labels.** The Round-8 level floor
  scored 28% precision, and the empty-VAD rule I proposed in Round 13 on the
  strength of six labels scored 38%. The reviewer's own comments say why: static
  can be *loud* (one V01 file buzzes at -23 dB, louder than most working
  recordings), and an empty VAD is a fact about the annotator, not the recording
  -- the VAD fails on working files, and a participant who barely speaks still
  has a live microphone, audibly so when their partner talks.
- **What works is the dynamic range of the level envelope.** 95th percentile
  minus median of the 50 ms frame level: unusable **0.20-1.35 dB**, usable
  **1.70-69.04**, no overlap. The cut at 1.5 dB scores **13 of 13 with zero false
  positives**. It needs no VAD, no partner and no absolute level, which is why it
  survives where three attempts did not, and it is the same judgement the
  reviewer was making by ear -- verifying a track was alive by hearing *someone*.
- **The new rule's firing set is a strict subset of the old rules' union**: of
  8,494 files, the number it flags that neither old rule flagged is **zero**. So
  no gallery was needed this round. It also clears 85 files the old rules were
  wrongly rejecting.
- **Firing rates**: V00 0.03%, V01 3.75%, V02 0.33%, V03 0.33%.
- **FM4 stays off for V00 by instruction**, but the reason has gone: the rule
  makes no mistakes on the 11 V00 labels, fires once in 4,000 V00 files, and that
  one file already fails the pipeline on other grounds -- so re-enabling would
  change zero verdicts.
- One reviewer label was corrected: `V01_S0104_I00000132_P1196` was listed as
  usable but has a 58-byte WAV, and the renderer had substituted generated
  silence, so the clip contained nothing by construction.
- Corpus projects to **794 dyad-hours**. FM2 is now by a wide margin the only
  check rejecting a large share -- 59% to 80% by vendor -- and the only one no
  review has ever contradicted.

## Round-13 findings (V00's FM1 confirmed; FM4 measures the wrong thing)

- **FM1 on V00 is confirmed where its labels had never looked.** A random ladder
  across the whole FM1 pass side, 10 clips in each of four bands from 0.43 to
  0.80, came back **0 seated in 40** -- a 95% upper bound of 7.5% by the rule of
  three. V00 does not have V03's false-negative problem.
- **Do not mix the old V00 labels with the new draws.** The 66 pre-existing ones
  came from the flagged and adjudicate pools; combined with the fresh draws they
  imply a 3.5% seated rate on the pass side, which the 0/40 refutes outright.
  Band-conditional rates from a biased draw are still biased.
- **Below the cut, FM1's precision is 56%** (9 seated of 16 in 0.30-0.43). That
  band is 1.45% of V00, so the standing files it discards are about 0.63% of the
  vendor, 9 dyad-hours of 1,441. Left alone.
- **The session-majority scope is right for V00**, 8 of 11 against the per-file
  rule's 3, and the six it correctly overrides all sit at knee 0.403-0.430 --
  right at the cut, where the per-file measure is noisiest. Exactly the opposite
  of V03, for exactly the reason the rule assumes.
- **FM4's floor is absolute and the vendors are not.** Median speech level: V01
  -18.6 dB, V03 -19.9, V02 -37.1, **V00 -41.8**. So -55 dB sits 36.4 dB below
  V01's median and only **13.2 dB** below V00's, cutting into its 1st percentile.
  All six V00 firings were participants who speak rarely. FM4 is off for V00.
- **A vendor-relative floor does not fix it** -- 30 dB below the median still
  fires on 3 of the 6 -- because "speaks rarely" and "no voice at all" look
  identical to any whole-file average.
- **`audio_vad_seconds` does separate them.** All six confirmed-dead files have
  **zero** seconds of released speech; five of the six V00 false positives have
  1.6 to 79.1. Empty-VAD rate: V00 **0.00%**, V01 5.25%, V02 0.75%, V03 3.25%,
  over 400 files each. Measured and recorded, **not** wired in: the two rules
  disagree about 84 files outside V00 and that difference is unvalidated.
- One V00 clip, `V00_S1669_I00000636_P0106A`, has zero VAD and a -85.7 dB level
  and is indistinguishable from a confirmed-dead file by any measurement
  available, yet was judged not dead. It is in the Round-13 gallery.
- Corpus now projects to **790 dyad-hours**, V00 being 449 of them.

## Round-12 findings (V00 re-examined under the V03 lessons)

- **All four V00 settings survive their re-test.** Knee position beats hip
  flexion *as a replacement*, which Round 5 never tested -- AUC 0.9283 against
  0.9317, but 17 of 19 sitters for 1 of 47 standing lost against 17 for 3.
  Session-majority scope beats per-file, 0 of 47 standing lost against 1, and
  only 1.1% of V00's multi-file sessions are mixed. The shin add-on fires on one
  file in 4,000 and that file is seated, adding a true positive for no false one.
- **The raster lesson closes by census, not by sample.** All **41,205** eligible
  V00 files are 1080x1920 at 30/1, with zero empty WAVs and zero timebase drift;
  over a 4,000-file scan, one distinct pixel aspect and no file needing a quarter
  turn. FM0 fires on nothing in V00.
- **V00 and V03 standing participants differ by 30 degrees of hip flexion**
  (median 129.0 against 159.3). That is why no shared hip cut could have worked
  and why hip flexion has far less headroom above the seated range on V00.
- **What does *not* survive is the coverage of V00's labels.** All 66 came from
  the flagged pool or the Round-4 adjudicate pool, so they cluster at the cut:
  standing p25 0.482, seated p75 0.387. **FM1's false-negative rate over the
  comfortably-passing population has never been measured** -- 3,485 of 4,000
  sampled files pass with a knee position above 0.50 and none has been seen by
  eye. That is exactly where the V03 problem lived. One V00 label already sits at
  0.728 and is seated; it is caught only because the shin add-on happens to fire.
- **FM4 finds 9 dead-audio files in 4,000 V00 files (0.22%)**, the first found in
  this vendor; the earlier 300-file probe finding none is consistent with that
  rate. Voice isolation median 7.65 dB, squarely in the close-mic band.
- V00 projects to **447 dyad-hours** against 341 across V01-V03 combined.
  Naturalistic trips FM3 at 10.55% against improvised's 3.70%, unchanged since
  Round 2.

## Round-11 findings (FM1 settled on 164 V03 labels)

- **The scope change is confirmed, and one direction unanimously.** Of the 28
  files where the two scopes disagree: where the file says standing and the
  session said seated, the file rule is right **14 of 14**. Where the file says
  seated and the session said standing it is 6 of 14 -- a wash, and an
  adversarial draw against the rule being proposed, so breaking even there is
  fine. Overall 20 of 28 against 8. Across all 149 labelled files in a scan:
  per-file 47 of 53 sitters and 19 of 96 standing lost; session majority 36 and
  34.
- **The shin add-on is pure cost on V03.** It flags 8 files that hip flexion does
  not and **none of the eight is seated**; only 6 of the 18 standing files lost
  at 146 degrees had both ankles in shot on every frame. Removing it halves the
  standing loss (18 of 96 -> 10 of 96) and costs no sitter. It stays on for V00,
  where it fires once in 4,000 files and that file was seated.
- **Four V00-validated choices have now failed on V03**: the raster calibration,
  the measure FM1 reads, the verdict scope, and the shin add-on. Treat this as a
  rule, not four findings -- **anything fitted on V00 is fitted on V00**, and V00
  is 1,441 of the corpus's 3,961 dyad-hours, so its own settings have never been
  re-examined under any of these four lessons.
- **The estimated cost of the 146-degree cut rose with every label round**: 8.9%
  of standing files at 90 labels, 12.3% at 123, 18.8% at 149. Each earlier draw
  over-represented clear cases. Half the final rise was the shin add-on rather
  than the cut, which is how it ends at 10.4%.
- **146 degrees kept over 138.** 138 has better accuracy (0.960 against 0.926)
  but leaves 5 of 53 seated participants in, and the difference is 2 dyad-hours
  out of about 148 -- so the recall-biased choice the reviewer asked for costs
  roughly 1% of the vendor.
- **The one seated file 146 still misses reads 151.9 degrees**, inside the
  standing range, which begins at 136.1. No cut separates it.
- **The labels are internally consistent.** Seven files were labelled twice
  across rounds, because the sampler does not exclude already-judged files, and
  all seven agreed. No contradictions in 164.
- V01-V03 now project to **341 dyad-hours** against V00's 419.

## Round-10 findings (FM1's scope, not its measure)

- **A seated V03 clip passed FM1 because its session outvoted it.**
  `V03_S1558_I00000012_P1425` reads 92.6 degrees of hip flexion -- deeply seated
  -- and `fm1_sitting_file` is True. The verdict is taken from a majority of the
  participant-session, and 13 of its 14 files read standing. The measure was
  right; the scope was wrong.
- **The constant-posture assumption is a V00 fact, not a corpus fact.** Share of
  multi-file participant-sessions where some files read seated and others do not:
  **V00 1.1%** (12 of 1,072), **V03 16.2%** (210 of 1,293). In V03's mixed
  sessions the seated share has a median of exactly 0.50, so the majority vote is
  close to a coin toss. This is the third time a V00-validated choice has failed
  to generalise -- after the raster calibration and the choice of measure -- and
  the pattern is now worth stating as a rule: anything fitted on V00 is fitted on
  V00.
- **Per-file beats session-majority on V03, on both axes.** 108 labelled files:
  majority gives 35 of 46 sitters and 13 of 62 standing lost; per-file gives 40
  and 7. Corpus effect is nearly nil in volume (24.40% -> 23.87%) and large in
  which files are caught: 124 were seated-but-outvoted, 154 standing-but-outvoted.
- **Posture also changes within single recordings.** The reviewer flagged one
  clip that starts standing, sits five seconds in, stands again after two
  minutes, and sits at the end. Its hip flexion runs p25 98, median 137, p75 161
  degrees. The median lands on the right side of the cut, so the file is caught,
  but a wide interquartile range is the honest signature and is now on the card.
- **Hip flexion on 123 labels: standing bottoms out at 136.1 degrees, seated tops
  out at 151.9.** The 146 cut catches 57 of 58 sitters for **8 of 65** standing
  lost -- larger than the 4 of 45 the 90-label set gave. 138 would give 54 and 3.
  Still far better than knee position's best of 41 and 12.
- **The audio question is closed.** All 30 surviving clips had usable audio, so
  the raster exclusion handled it and no further detector is needed. Voice
  isolation remains refuted.

## Round-9 findings (a measure change, a refutation, and a shipped bug)

- **FM1 was reading the wrong measure on V03.** With 90 hand labels instead of
  58, knee position's standing quartile is 0.545, so the Round-8 cut of 0.54 sat
  *inside* the standing distribution: 41 of 45 sitters for **12 of 45** standing
  lost, not the 4 of 22 the smaller set implied. **Hip flexion at 146 degrees**
  gives 44 of 45 for 4 of 45 and rejects 26% of V03 against knee's 33% -- better
  on all three axes. Standing bottoms out at 136.1 degrees and seated tops out at
  151.9, so the cut sits in a real gap. LOSO: 43/45 and 3/45, accuracy 0.944
  against 0.822.
- **The label draw was selection-confounded on knee position and the conclusion
  survives it.** Three reasons: hip flexion wins *inside* the knee-ambiguous band
  (AUC 0.977), which is conditional on the selection rather than biased by it;
  selection bias cannot produce a rule that catches more sitters while rejecting
  less of the vendor; and the 136/152 gap is a property of the distributions.
- **Hip flexion was retired in Round 5 on V00 evidence.** That was right for V00
  and wrong to generalise. Same lesson as the raster finding, from another
  direction: a measure validated on one vendor is a measure validated on one
  vendor. `sitting_by_vendor` now carries `(measure, cut)`, not a bare number.
- **Voice isolation is refuted as a per-file detector.** 30 hand labels: accepted
  clips at -1.68, 0.09, 0.34, 0.48, 0.93, 1.36 dB and rejected ones at 0.33,
  2.07, 2.17. An accepted clip at 0.34 against a rejected one at 0.33. Any cut
  catching all three rejects loses nine of 27 accepts. The *mechanism* still
  explains the population difference; the measure predicts no individual verdict.
  What handles the problem is the raster exclusion: 20 of the 21 originally
  reported clips are in rasters FM0 drops.
- **A crop that was implemented, tested and never called.** Round 8 added the
  pillarbox field, the measurement, the ffmpeg filter and four unit tests, then
  constructed the renderer's repair without passing it. Every test passed because
  every test exercised the crop directly; none checked that the *renderer* asked
  for one. There is now a test that does, and end-to-end verification -- a
  rendered sidecar reading `cropped 540 black columns` -- rather than unit tests
  alone.
- **Scrubbing: a green banner plus a green buffer pill rules out the transport.**
  The container was re-checked too (moov first, 30 sync samples, agreeing mvhd
  and mdhd durations, one standard edit list). What is left is the media
  element's resource loader, which an embedded webview replaces. Clips are now
  fetched into memory and played from a blob URL, which has no loader to refuse a
  seek. Only clips near the viewport are fetched and the oldest fourteen are kept.
- V01-V03 now project to **337 dyad-hours**, up from 313: the better FM1 measure
  recovers 23 hours on V03 while catching more sitters.

## Round-8 findings (audio, and the first thresholds to move)

- **"Glitchy audio" is a far-field or shared microphone, not a digital fault.**
  Clipping, dropouts, spectral cutoff and fragmentation all overlap completely
  with good recordings; the fragmentation hypothesis is the seductive one and it
  is wrong, because every dyadic recording is chopped (median live burst 0.17 s
  on both sides). What separates them is *whose voice the microphone hears*:
  comparing a track's energy during its own speaker's turns against the
  partner's, close-worn mics give 7.3-12.8 dB and V03's room-camera rasters give
  2.2 (3840x2160, n=1,318) and 3.8 (640x480, n=55). That is also why the reports
  correlated with odd aspect ratios -- the odd rasters *are* the room-camera rig.
- **But voice isolation does not separate file by file.** Any cut that catches
  the reviewer's labelled clips also takes 31-59% of V00 and V02. Recorded as a
  signal, not shipped as a detector; the gallery's ASK 2 collects the labels
  that would decide it. This is the main open question of the round.
- **FM4 (new): dead audio.** Speech level < -55 dB **and** spectral flatness
  > 0.05. Both halves needed -- a quiet recording is not broken and a flat
  spectrum in a loud file is a fan. Caught all four files the reviewer confirmed
  dead, **none** of the 18 that are merely unpleasant, and 0 of 300 V00 files.
  Fires on 3.80% of V01, 2.13% of V02, 0.87% of V03.
- **FM1 is now per-vendor.** On 58 V03 hand labels the seated distribution sits
  about 0.11 above V00's (median 0.413 against roughly 0.30) while the standing
  distribution is unchanged -- V03's seating, not its camera. 0.43 caught 21 of
  36 sitters; **0.54** catches 33 for 4 of 22 standing lost, which is the
  recall-biased trade the reviewer asked for. It costs real hours: V03 goes from
  13.0% to 9.1% surviving. 0.56 buys one more sitter for another 13% of the
  vendor.
- **FM1 is switched off for V01 and V02**, and the measurements say why rather
  than only that it is safe. V01 fires on **0 of 1,379** square-pixel files --
  its 43.6% rate was entirely the anamorphic rasters FM0 now removes. V02 fires
  on 182 files and **181 of those are the shin add-on**, the ankles-out-of-frame
  mechanism, with exactly one firing on knee position. Recorded as
  `retired_for_vendor`, never as a clean pass.
- **FM0 (new): the source is unusable whatever it shows.** Four rasters and a
  timebase rule, each of them a reviewer decision rather than an inference.
  Rejects 44.6% of V01, 0% of V02, 3.9% of V03. The 3840x2160 exclusion is the
  costliest and least certain (41.8 dyad-hours, one sampled file had fine audio)
  and is one config line to reverse.
- **The padded rasters are exactly as the reviewer read them.** Over the
  brightest value each column ever reaches: 1080x960 is 540x960 with 270 black
  columns each side, 2180x3840 is 2160x3840 with 20 columns on the **right
  only**. Both now cropped. **The padding was costing nothing** -- shape residual
  0.057 and 0.061 against a 0.048-0.074 normal band -- so the crop is cosmetic.
- **Three separate things had to be true for a video scrubber to work**, and
  fixing two of them was not enough. Keyframe density in the clip; HTTP `Range`
  from the server; and, for any transport that offers neither (an editor preview
  pane, a `file://` path), buffering the whole clip so that seeking needs no
  request at all. The third is the only one that is transport-independent, and it
  is the one that was missing. The gallery now also self-reports what the
  transport did with a range request, so this never has to be diagnosed by
  guesswork again.
- V01-V03 together now project to **313 dyad-hours** against V00's 419.

## Round-7 findings (the reviewer's V01/V02/V03 observations, diagnosed)

- **The V01 "stretch" is anamorphic storage, and the container says so.** Every
  V01 raster declares a 9:16 display aspect; only 1080x1920 stores square pixels.
  2160x2160 carries pixel aspect 9:16 (stored 1.78x too wide, 5,002 eligible
  files, 139.0 dyad-hours), 1920x1080 carries 81:256 (3.16x, 60 files), and
  1012x1920 carries 270:253 (6.7%, 102 files). All 31 clips the reviewer listed
  are 2160x2160. Confirmed independently in the keypoints: shoulder width over
  torso length reads 1.103 and 1.786 on the two bad rasters against a 0.589-0.721
  band across every square-pixel raster, and lands back inside it -- 0.624 and
  0.573 -- once the pixel aspect is applied.
- **An anamorphic file's released SMPL-H cannot be repaired.** The fitted camera
  is isotropic, so a single 3D body cannot project to a stretched 2D person; the
  pose absorbed the stretch. Shape-only Procrustes residual, 60 files per raster:
  2160x2160 is 0.113 against the keypoints as stored and **0.218** once
  corrected; square-pixel rasters are 0.048-0.074. Fixing the picture does not
  fix the annotation, and this is the single most consequential finding of the
  round.
- **Quarter-turned video is fully repairable, and changes no flag.** 1,336
  3840x2160 files (1,215 one way, 112 the other, 9 upright) and all 60 640x480
  files, 1,396 eligible in total, 43.8 dyad-hours. A quarter turn about the
  principal point of a centred square-pixel pinhole is exactly a camera rotation
  about the optical axis, so keypoints rotate, `global_orient` absorbs the turn,
  and `body_pose` is untouched. FM1 is rotation-invariant, FM3 is normalised, FM2
  reads a released field -- so nothing moves. `1080x960` (28 files) is *not*
  rotated: upright, square pixels, just an unusually wide frame.
- **FM1 on V03 is mis-set, not broken.** The reviewer's 14 labels separate almost
  perfectly on `knee_between_torso`: caught sitters 0.301-0.429, missed sitters
  0.435-0.586, cut at 0.43. All 14 have shins the right way up. A re-fit needs
  labels on both sides of the candidate cut, which is what the gallery's posture
  ladder collects; guessing a value would bake in an unmeasured number.
- **FM1 fires on standing people through the shin clause, never the knee
  clause.** All eleven legs-cut-off clips the reviewer listed have
  `knee_between_torso` in 0.552-0.677 -- above the cut -- and
  `fm2_feet_out_of_frame_frame_frac` exactly 1.0. Across 3,594 Round-6 files:
  600 fire on `knee_between`, 126 on `shin_inverted`, 8 on both. New measurement
  `fm1_ankles_visible_frac` separates "FM1 judged the legs" from "FM1 never saw
  them"; the verdict is deliberately unchanged.
- **Annotations are on a uniform nominal-rate grid the container may not match.**
  `V01_S1607_I00000135_P2569`: nominal 48000/1001, 3,261 annotation frames,
  2,807 container frames, same 68.005 s. Pairing by frame index misaligns by a
  median 4.45 s and up to 6.87 s; by timestamp, 0 ms median and 41.7 ms worst.
  70 of 125,184 eligible files drift by more than 0.5 s, and reading the array
  lengths splits them: 35 are the known empty-annotation files, 16 are this
  dropped-frame case, 12 carry arrays several times longer than their video, 7
  are smaller mismatches. All V01 except 3 in V02. 98.8% of 160 controls agree
  within one frame.
- **183 eligible files ship a 58-byte WAV** -- a RIFF header with no samples --
  and 226 have a WAV more than a second shorter than the video. None in V00. With
  `-shortest` this silently truncated a 30-second render to 20 seconds; the
  renderer now substitutes generated silence and pins length with `-t`/`apad`.
- **The progress bar had two causes.** libx264's default GOP put three keyframes
  in a 30-second clip, so a seek could only land on 0, 10 or 20 s (now one per
  second). And `http.server` has never implemented HTTP `Range`, so a browser
  reads its `200` as "not seekable" and restores the old position. Fixing either
  alone is not enough; `scripts/serve_gallery.py` covers the second.

## Other vendors (V01, V02, V03)

- **FM1 collapses on non-portrait rasters.** By frame shape, not by vendor:
  portrait 8.9% flagged (knee_between median 0.601), landscape 43.2% (0.533),
  **square 93.3% (0.289)**. Within V01 the split is total -- 0 of 693 portrait
  files flagged against 93% of 496 square ones. The measure is rotation-invariant
  and computed from the released pose, so the raster reaches it through the
  *fit*, not directly: the released SMPL-H was produced under a camera convention
  keyed to the raster. The cut of 0.43 is calibrated for portrait framing and
  means nothing outside it.
- Reading a headline rate per vendor would have hidden this completely. V01's
  39% "seated" is one raster, not a vendor.
- **1,387 V03 files are 90 degrees from upright** -- 97% of its 1,424 landscape
  files, all 3840x2160 or 640x480, measured over 7,051 files. The pipeline is
  structurally blind to it: released keypoints and SMPL-H both track the rotated
  person correctly in raster coordinates, so FM2 and FM3 are unaffected and FM1
  is rotation-invariant. A sampled example passes all three checks.
- **V02 is the puzzle.** Identical format to V00 (uniform 1080x1920, 30/1) but
  median reprojection error 0.111 shoulder widths against V00's 0.079, and an 18%
  pass rate against 30%. No explanation yet. 761 dyad-hours, naturalistic only.
- **Six V01 files have zero-length annotation arrays** with real video and real
  duration. `all_modalities_present` only checks that siblings exist, not that
  they hold anything. Now reported as `empty_annotations`.
- FM3 is the one check that behaves the same everywhere (6.1-9.0% against V00's
  7.3%), which is what a shoulder-width-normalised, self-anchored measure should
  do.

## SMPL-H pose geometry (camera tilt and the knee bend)

- **Panel C plots camera-frame coordinates**, so V00's downward-angled camera
  drew upright participants leaning. Median pelvis-to-neck pitch out of the image
  plane is **-13.1 deg**, head toward the camera. It is the rig, not the person:
  within-session SD 1.88 deg against 5.10 deg between sessions over 109 sessions.
  Fixed by `upright_side_view` (a rigid rotation, so no joint angle moves).
- **The knee bend is real and large**: knees held **59 deg** from straight, hips
  23 deg, foot 131 deg from the body axis where 90 would be flat. Joint angles
  are rotation-invariant, so no display change can produce or remove them.
- **The bend is demanded by the 2D keypoints.** Forcing both knees straight makes
  lower-body reprojection **61% worse** (0.0875 -> 0.1403 shoulder widths), worse
  in 439 of 440 files, p = 8e-74. So it is not free-floating depth noise -- it is
  holding the fitted legs onto the observed keypoints.
- Leading explanation, inferred rather than proven: the assumed camera is
  near-orthographic with **no pitch parameter** (focal 37,500 px, vertical FOV
  **2.93 deg**, fitted depth **39.8 m**), so it cannot represent the leg
  foreshortening a real downward-angled camera produces, and the fit absorbs it
  by bending the legs. Supporting: spine1 correlates with camera tilt at
  Spearman -0.605 and, like the tilt, varies more between sessions (3.97 deg)
  than within (1.47 deg). Against: pitch vs knee bend is -0.385, the wrong sign
  for the simplest single-scalar version of the story.
- **The distortion is confined to the legs**, which are not in the pelvis-to-hand
  chain: torso interior reads spine2 2.2 deg and spine3 2.6 deg from rest. The
  one contaminated link in the gesture chain is spine1 at 20.2 deg, and it is a
  static per-recording offset (within-file 2.3 deg, across-file 4.6 deg).
- A rigid rotation changes nothing body-relative. That is why the camera tilt is
  harmless to a root-canonicalised model and the knee bend would not have been.
- **Still unverified: arm depth.** Reprojection is 2D, so the upper body is
  confirmed to land on the image (0.0707 shoulder widths) but not to be at the
  right depth. For 3D gesture that is the residual risk.

## Round-5 findings

- **`smplh:is_valid` is a hand-pose flag.** The release documents nothing beyond
  the key name and dtype, so this took measuring. On invalid frames the *body*
  fit is indistinguishable from valid frames -- paired within 2,481 files, the
  median 2D fit error against the released keypoints is 0.0792 shoulder widths on
  valid frames and 0.0799 on invalid ones, and in 44.9% of files the invalid
  frames fit *better*. What does change is the hands: **41% of left-hand and 39%
  of right-hand pose vectors on invalid frames are bit-identical to the previous
  frame** (0% on valid frames), and median per-frame hand-pose change drops to
  0.33x and 0.03x. Body pose, global orient and translation are unaffected
  (p = 0.67 for body pose). The pipeline holds the previous hand pose when it does
  not trust the hand fit.
- That resolves the Session-1 puzzle of SMPL-H-invalid spans over "fully visible
  participants with stable 2D keypoints and no obvious transition". The body was
  fine. Only the hands were frozen, and nothing we were looking at showed hands.
- **For a co-speech gesture model the flag is pointed at exactly the right
  thing**, which is a better argument for gating on it than "the SMPL parameters
  are low quality" in general.
- **There are no catastrophically bad body fits in V00.** Per-file median
  reprojection error runs p1 0.039 to p99 0.156 shoulder widths, max 0.170
  (12.6 to 53.8 px, max 62.4). The p50 of 26.3 px matches M-4's independent
  25.53 px. A direct fit-quality gate therefore has no natural threshold, and is
  recorded as a measurement rather than shipped as a detector.
- **Hip flexion is retired from FM1.** Across 66 hand-labelled files, every file
  it flags that `knee_between_torso` does not is a **standing** person -- 4 of 4 --
  and it cost 29 extra files corpus-wide with no validated hit. A criterion can
  separate a label set perfectly and still be pure noise on the corpus; the only
  thing that exposed it was asking the reviewer about the files where the two
  criteria disagree.
- The FM1 negative controls came back **10/10**. Ten unlabelled ordinary standing
  files were mixed into the adjudication group precisely so the answers could be
  scored rather than just collected, and the rule did not over-fire on any.
- **Known miss: a participant on a high stool swinging their legs.**
  `knee_between` reads 0.485, squarely standing, because the hip is nearly
  straight and the knee sits where a standing knee sits. Leg *posture* cannot
  separate this; leg *motion* should, since a standing participant's ankles are
  planted. Unmeasured.

## Round-4 findings (detectors rebuilt after the reviewer's inspection)

The reviewer inspected the Round-3 gallery, scrapped FM4 outright, confirmed
FM3's calibration, and found concrete errors in FM1 and FM2. Both were rebuilt.
Only **sitting, framing, and static hands** remain, all at file level.

- **Zeroed betas are why FM1 was wrong.** Every SMPL-H forward pass uses sixteen
  all-zero betas, so the FK thigh is 376.8 mm and the shin 400.6 mm in *every*
  file regardless of who was recorded. FK output is therefore pure pose. The 2D
  ratios are not: `leg_over_torso` encodes body build, which is why it flagged
  fifteen short-legged **standing** participants and no seated ones — 0/15
  correct on inspection. It is retired, along with `knee_spread`, which missed
  all four seated participants it had not been selected on.
- ~~**SMPL-H forward-kinematic leg angles are unusable.**~~ Half wrong. The
  *absolute* angles are uncalibrated — a confirmed standing participant reads
  87.4 degrees of knee flexion, anatomically a deep squat — but the *ordering* is
  clean and was never checked in Round 3. Labelled sitting reads hip flexion
  88.2–113.8 degrees against standing 119.9–141.6, with no overlap, and neither
  quantity entered any earlier selection criterion, so that separation is not an
  artefact of how the labels were drawn.
- **FM1's primary is now `knee_between_torso`**: how far down the hip-to-ankle
  drop the knee sits, measured along the participant's own torso axis. Sitting
  0.152–0.406, standing 0.508–0.613. Being taken along the torso axis makes it
  invariant to camera tilt and distance, and being pure FK makes it invariant to
  build. 29/29 on the reviewer's labels.
- **Posture is constant per recording session, not per person.** The reviewer
  observed that a participant either stands in all their clips or sits in all of
  them. Right observation, wrong scope, and the difference matters. P0003A
  appears in **8 sessions**: seated in S0200 (hip flexion 88.2, confirmed by eye)
  and standing in the other seven (130.6-156.1, and a rendered S0126 clip shows
  them on a standing platform). Corpus-wide the median within-session spread of
  hip flexion is **2.76 deg** against **7.11 deg** between sessions, and units
  that disagree internally fall from 6.81% of multi-file participants to 2.52% of
  multi-file participant-sessions. The chair, the camera height and the platform
  are a per-session setup, so posture is a property of the setup.
- **Session scoping deleted a rule I could not defend.** An earlier draft carried
  a "deep" clause -- one file far inside the seated range speaks for the whole
  participant -- added purely to rescue P0003A. Leave-one-participant-out broke on
  exactly that participant, which is the signature of a parameter fitted to one
  example. Under session scoping P0003A's seated session holds one file, it flags,
  and a plain majority carries it; the deep clause changes zero verdicts on 4,000
  files and is gone. Worth remembering as a pattern: when one clause exists only
  to rescue one case, look for a better unit of analysis before adding the clause.
- **Half the units are a single file.** 1,379 of 2,451 participant-sessions
  (56.3%) hold one scanned file, so a majority vote there is one vote and can
  never disagree with itself. Without a separate near-the-cut channel the rule
  would report perfect confidence on more than half the corpus.
- **FM2 was blind to the body surface.** It tested joint *centres*. Over 76,817
  sampled frames the outermost mesh vertex reaches a median **37.6 px** further
  out than the outermost joint centre, and the SMPL-H projection is additionally
  pulled inward on the arms by 9–11 px against the released 2D keypoints. In the
  file the reviewer found, the elbow joint sat 7.7 px inside while the mesh
  reached 24.5 px outside. FM2 now tests the mesh, cross-checks the released 2D
  keypoints, and is a strict superset of the old rule on all 4,000 files.
- Vertices are nearly free: `smplx` runs `lbs` and produces the mesh on every
  forward pass, and `return_verts=False` only discarded it. The 64-task scan took
  613 s per task on average with the mesh projected, against 666 s in Round 3
  without it -- the marginal cost is inside the noise of a shared cluster, and a
  microbenchmark's "+14.9% throughput" figure did not replicate on a re-run.
- **The mesh term is worth less than the raw gap suggests, for an instructive
  reason.** The per-frame joint-to-surface gap has a median of 38.8 px, but
  `IN_FRAME_JOINTS` already contained indices 63-72 -- the ten fingertips, which
  `smplx` produces by *selecting mesh vertices*, not by kinematics. The joint set
  was already part surface. Going from joints to the full mesh is worth **80
  files, 2.00 points** on 4,000. It is what catches the reviewer's example, but it
  is a 2-point effect, not a 38-pixel one.
- **9% of the corpus fails FM2 on missing data, not bad framing.** 360 files are
  flagged by `smplh:is_valid` alone with no out-of-frame geometry at all, and
  62.05% of files have at least one invalid SMPL-H frame.
- **FM2 at file level now flags 81.1%** (up from 77.5% on the same files),
  leaving **177 dyad-hours of 1,441**. The reviewer accepted this explicitly,
  wanting a correct FM2 before any loosening is considered.
- Every file with a large tracker-centre-jump discontinuity is already caught by
  the in-frame rule: 63 candidates above 0.25 box heights, zero missed. A
  bystander the tracker correctly ignores would still be invisible.
- CPU SMPL-H forward kinematics runs at 3,705 frames/s joints-only, and around
  1,300–1,600 frames/s with the mesh projected. Still no GPU needed.

## Round-3 findings (superseded by Round 4)

- The reviewer retracted four failure modes for V00 after the Round-2 gallery:
  glitchy tracking, audio/video desync, camera roll, and unnatural hand motion /
  noisy body pose. A fifth, extended mutual silence, was scrapped after Round 3.
- The released 2D leg keypoints do carry one real signature: a participant on a
  stool with their feet on a rung has the ankle *above* the knee, so signed shin
  verticality flips from +1.000 to -0.832. It fires on exactly one file in 4,000
  and survives into Round 4 as a rare-posture add-on.
- ~~`leg_over_torso < 0.85` looked like a reasonable sitting criterion~~ and is
  simply wrong; see the Round-4 note on zeroed betas for the mechanism.

## Round-2 findings (V00 decided)

- The reviewer's own Pass-1 vocabulary — framing, roll, sitting, static hands,
  desync lag — is the first signal vocabulary in this project that came from a
  human before being measured. Four of the five now have working proxies.
- **The audio/video desync proxy cannot work on V00.** Inter-ocular distance is
  79-126 px, so the released whole-body face block resolves the mouth across a
  handful of pixels. Peak mouth/audio activity correlation had median 0.151 over
  eight dev files with the argmax lag scattered -4 to +12 frames, and over the
  8,000-file pool it is identical between labels (rank-biserial 0.0003). The
  68-point layout was confirmed empirically first and the inner-lip centres were
  found to sit 0.011 of face height apart, so the failure is resolution, not
  indexing.
- ~~**There are no sitting participants in V00.**~~ **Wrong, corrected in Round
  3.** Ankles are usable in 100% of frames in 100% of 8,000 sampled files and no
  file has the person under 40% of frame height, and I inferred from that there
  were no sitters. There are: a stool sitter with feet on a rung still has both
  ankles tracked every frame and still fills the frame, because the camera was
  moved closer. The absence of a signal in measures that cannot see the thing is
  not evidence the thing is absent.
- **Static hands is 2.3x to 4.1x more common in naturalistic than improvised**
  (27.5% vs 11.9% at >50% of file; 7.0% vs 1.7% at >80%), and improvised also
  carries more speech (0.444 vs 0.367 median). This cuts against the Session-2
  recommendation of naturalistic-only, which was made before any gesture-density
  measurement existed.
- OpenCV's `CAP_PROP_POS_FRAMES` decodes forward from the previous keyframe.
  Twelve filmstrip thumbnails per clip cost 11 minutes per clip at 32 concurrent
  tasks on cold NFS; input-side `ffmpeg -ss` does the same work in 15 seconds.
  The same trap applies to any future random-access pass over this corpus.
- Reading NPZ member shapes from their `.npy` headers instead of inflating each
  DEFLATE member turned 400 candidate resolutions from fourteen minutes into
  under a minute. `archive[name].shape[0]` decompresses the whole member.
- Slurm array throttle should be set from task duration, not politeness. Tasks of
  8-15 seconds at `%8` spend most of their wall clock waiting for the next
  scheduling cycle while the partition sits idle.
- A `srun` launched from a tool call that times out keeps running. Three copies
  of the sampler ended up competing for the same NFS reads before this was
  noticed; check `pgrep` before relaunching.

## M-2/M-4 findings

- The tested HMR-style full-frame projection passed on a bounded seven-file,
  four-vendor dev sample for body-17 and hand joints. An unscaled 5,000-pixel
  focal length and a tested crop transform were much worse.
- `flat_hand_mean=true` lowered aggregate hand median/p95 reprojection error and
  won all seven per-file comparisons. This project choice is now frozen for FK
  features and review rendering.
- The six extra foot landmarks create structured extreme reprojection outliers;
  body-23 p95 is therefore misleading for an upper-body/hand acceptance test.
  Foot joint mapping/provenance remains unverified.
- Translation is strongly box-coupled and behaves like an HMR camera/tracker
  parameter. It should not be called physical root motion. Metric motion now
  comes from neutral-model FK with translation removed and pelvis subtracted.

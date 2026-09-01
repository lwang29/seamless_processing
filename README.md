# Seamless Interaction curation harness

Measurement-only tooling for building a curated co-speech-gesture training subset
from the Seamless Interaction dataset. The source tree is read-only; everything
this repository produces lands under `outputs/` or `artifacts/`, both git-ignored.

**Nothing selects or filters data yet.** Sessions 1 and 2 measured; the pipeline
that would select is specified in `docs/pipeline_spec_v1.md` and is awaiting
review before any of it is implemented.

**Vendor decided: V00 only** (2026-08-06). Naturalistic versus improvised is open
— see `reports/02_review_findings.md` §10 for the measured comparison and §12 for
what the review still needs.

Use the existing ViBES environment without modifying it. Both the repository root
and `src/` need to be importable, because the review provider imports FK
primitives from `scripts/`:

```bash
export PYTHONPATH="$PWD/src:$PWD${PYTHONPATH:+:$PYTHONPATH}"
VIBES=/simurgh/group/lw29/conda/envs/ViBES/bin/python
```

Run nothing heavy on a login node. Every workload below is a Slurm job or an
`srun` step inside an allocation.

## Reports, in reading order

| document | contents |
|---|---|
| `reports/01_recon.md` | Session 1 reconnaissance of the dev splits |
| `reports/02_measurements.md` | Session 2: corpus inventory (M-1), camera tests (M-2), dev feature characterization (M-3), SMPL-H validation (M-4), and a closing section on what changed and what is now known to be impossible |
| `reports/02_review_findings.md` | Round 1 (corpus-wide, §1–7) and Round 2 (V00 by label, §8–12), including the reviewer-vocabulary proxies and the naturalistic-vs-improvised comparison |
| `reports/05_failure_mode_detection.md` | **Round 5 (current)** — the three V00 failure-mode detectors (sitting, SMPL-H validity, static hands): what `smplh:is_valid` actually measures, the FM1 retune after 37 more hand labels, flag rates, overlap, surviving hours, and threshold sweeps |
| `reports/06_smplh_pose_geometry.md` | Why upright participants read as leaning with bent knees: the tilt is the camera and Panel C's frame choice (fixed), the ~59-degree knee bend is real and required by the 2D data, and it is confined to the legs |
| `reports/07_other_vendors.md` | V01/V02/V03 under the V00-tuned pipeline: FM1 collapses on non-portrait rasters, FM2 and FM3 transfer, and 1,387 V03 files are 90 degrees from upright with nothing to detect it |
| `reports/16_pi_briefing.md` | **Round 15 (current)** — the corpus-wide briefing gallery: all four vendors, five checks, 246 clips over 22 sections, on a fresh sample nothing was tuned against |
| `reports/15_fm4_dynamics.md` | Round 14 — FM4 now gates on the *dynamics* of the level envelope, not the level: static sits at one level whether anyone speaks or not, and on 48 hand labels the cut separates 13 of 13 unusable files with no false positive, where the level rule scored 28% precision and an empty-VAD rule 38% |
| `reports/14_audio_rule.md` | Round 13 — FM1 on V00 confirmed (0 seated in 40 drawn from the pass side), the session scope confirmed, and FM4 retired for V00: its absolute level floor cannot tell a rare speaker from a dead track, while seconds of released speech can |
| `reports/13_v00_recheck.md` | Round 12 — V00 re-examined under the four lessons V03 taught. All four of its settings survive, and the raster lesson closes by census; what does not survive is the coverage of its 66 labels, which all sit near the cut, leaving FM1's false-negative rate above it unmeasured |
| `reports/12_fm1_final.md` | Round 11 — FM1 settled on 164 V03 hand labels: hip flexion at 146 degrees, per-file scope, and the shin add-on off, because on V03 it flagged 8 standing people and no sitters |
| `reports/11_scope_correction.md` | Round 10 — a seated clip passed FM1 because its participant-session outvoted it; the session-majority rule is true of V00 (1.1% mixed sessions) and false of V03 (16.2%), so V03 now takes FM1's verdict from the file itself |
| `reports/10_measure_correction.md` | Round 9 — 42 more posture labels showed FM1 was reading the wrong measure on V03 (hip flexion, not knee position: 44/45 sitters for 4/45 standing lost), 30 audio labels refuted voice isolation as a detector, and the pillarbox crop that Round 8 implemented but never called is now applied |
| `reports/09_audio_and_thresholds.md` | Round 8 — the glitchy audio is a far-field/shared microphone (measurable, but not yet separable file by file), FM4 rejects recordings with no voice, FM0 rejects the rasters and timebases the reviewer ruled out, FM1 becomes per-vendor (0.54 on V03, off for V01/V02), and the padded rasters are cropped |
| `reports/08_vendor_repairs.md` | **Round 7 (current)** — what the reviewer's V01/V02/V03 observations turned out to be: anamorphic storage (repairable for the picture and the 2D points, not for the released SMPL-H), quarter-turned rasters (fully repairable), FM1's V03 cut mis-set rather than broken, a drifting annotation timebase, 183 empty WAVs, and why the progress bar snapped back |
| `docs/pipeline_spec_v1.md` | **Draft** frozen pipeline spec — signal registry, gates, stage graph, budgets, vendor recommendation |
| `docs/review_gallery_tooling.md` | How the review renderer and rating capture work |
| `NOTES.md` | Surprises, dead ends, and open questions across both sessions |

## Workstreams

### Corpus inventory (M-1)

Header and stat metadata only — never a JSON, NPZ, or WAV payload byte.

```bash
sbatch slurm/inventory_m1_catalog.sbatch     # walk every directory entry
sbatch slurm/inventory_m1_headers.sbatch     # 512-task array, ffprobe headers
sbatch slurm/inventory_m1_summarize.sbatch   # join metadata, write summary tables
$VIBES scripts/benchmark_npz_read.py --seed 20260807 \
  --out outputs/02_inventory/benchmark/npz_read_scaling_coldish.json
```

### Window extraction harness (Session 1, extended in Session 2)

```bash
$VIBES -m seamless_curation --config configs/harness.yaml --limit 2
```

Measurement columns per 4-second window (1-second hop; all values from YAML):
`valid_frac_smplh`, `valid_frac_box`, `valid_frac_movement` (nullable),
`valid_frac_smplh_and_box`, the four `*_guarded` variants,
`hand_avail_frac_{left,right}`, `accel_mm_per_frame2`, `wrist_speed_p90`, and
`duration_mismatch_s`. `valid_frac_all` is retained for continuity with Session 1
but is NaN whenever the optional movement mask is absent, which is why the split
columns exist.

`guard_band_frames` (default 3) widens every invalid run symmetrically on the
whole-file mask before window slicing, so a window abutting a failure is not
scored as clean.

`accel_mm_per_frame2` is a provisional second difference of released root
translation × 1000. Session 2 established that it measures the HMR camera, not the
participant — its dev median is 230× the genuine root-relative body figure — and
`docs/pipeline_spec_v1.md` renames it accordingly. Use the M-3 metric signals for
real motion.

The window grid uses the YAML-configured 30 Hz. Every other timing measurement
uses each file's own `ffprobe` `avg_frame_rate`; config fallback is labelled
explicitly in `duration_fps_source` / `duration_fps_status`.

### Prototype signals (M-3)

```bash
sbatch slurm/m3_dev_features.sbatch          # per-span signals, 1 GPU
$VIBES scripts/m3_analyze.py                 # distributions + activity confounds
```

### Failure-mode detection (Round 5)

```bash
sbatch slurm/v00_fm_scan.sbatch           # FM1-FM3 over 4,000 V00 files
$VIBES scripts/v00_fm_analyze.py          # flag rates, overlap, sweeps, FM1 verdicts
$VIBES scripts/sample_v00_fm_gallery.py
sbatch slurm/render_v00_fm_array.sbatch
```

Thresholds live in `configs/v00_fm_detect.yaml` and are applied only by the
analysis script, so re-tuning any cut costs one re-run and no media reads. FM2
gates on `smplh:is_valid` — measured to be, in effect, a hand-pose flag: on
invalid frames ~40% of hand-pose vectors are bit-identical to the previous frame
while body pose is unaffected. The geometric in-frame measures are still recorded
alongside it, so restoring them is a config edit rather than a rescan.

Galleries in `artifacts/private_review_v00_fm_r5/`, split into two cohorts the
reviewer asked to keep separate — `recheck.html` (the same files as the Round-3
gallery, re-judged) and `fresh.html` (files never shown before) — plus
`pass.html`, one page per failure mode, and `fm1_adjudicate.html`, the units FM1
cannot confidently classify.

### Other vendors (V01/V02/V03)

```bash
sbatch --export=ALL,DETECT_CONFIG=configs/vendors_fm_detect.yaml slurm/v00_fm_scan.sbatch
$VIBES scripts/v00_fm_analyze.py --config configs/vendors_fm_detect.yaml
$VIBES scripts/sample_vendor_gallery.py
sbatch slurm/render_vendors_array.sbatch
sbatch slurm/probe_vendor_body_roll.sbatch      # how much video is a quarter turn off
```

Same three detectors at the same thresholds, deliberately — moving the cuts would
confound "this vendor is different" with "this cut is different". Gallery in
`artifacts/private_review_vendors/`: `index.html` plus a page per vendor and per
vendor-and-verdict, 55 clips in each of six cells, free-text notes on because
this is exploratory. See `reports/07_other_vendors.md` before reading the
pass/fail labels — two of the three checks do not transfer cleanly.

### Round 7: the repair layer and the vendor briefing gallery

```bash
sbatch --export=ALL,DETECT_CONFIG=configs/vendors_r7_detect.yaml slurm/v00_fm_scan.sbatch
sbatch --array=0-47%48 \
  --export=ALL,DETECT_CONFIG=configs/vendors_r7_rare_detect.yaml slurm/v00_fm_scan.sbatch
$VIBES scripts/v00_fm_analyze.py --config configs/vendors_r7_detect.yaml
$VIBES scripts/v00_fm_analyze.py --config configs/vendors_r7_rare_detect.yaml
$VIBES scripts/sample_vendor_briefing.py
sbatch --array=0-269%80 slurm/render_vendors_r7_array.sbatch
$VIBES scripts/build_vendor_briefing_gallery.py
```

`src/seamless_curation/media_repair.py` holds the three repairs the Round-6
review turned up, and the renderer applies all of them: the container's pixel
aspect (three V01 rasters are stored horizontally stretched), a quarter turn
(1,396 V03 files are stored sideways with no rotation tag), and annotation lookup
by timestamp rather than frame number (some files hold fewer video frames than
their nominal rate implies, so pairing by index drifts by seconds). The repairs
change no flag; they change what is on screen and what a 2D point means.
`reports/08_vendor_repairs.md` §1.4 is the important limit: an anamorphic file's
**released SMPL-H cannot be repaired**, because the fitted camera is isotropic
and the pose absorbed the stretch.

The rare-raster scan is a companion pass over *every* file of the six unusual
rasters. Some hold only 20 files, so a random draw over the corpus misses them;
they are excluded from the population statistics and used only to populate the
gallery's raster questions.

Gallery: `artifacts/private_review_vendors_r7/index.html` — 270 clips in 20
sections, 236 participants. Five **ASK** sections carry what needs a human
answer, above all where FM1's posture cut belongs on V03; fifteen sample sections
hold a fresh draw split by vendor and by exactly which checks fired. No notes, no
ratings: the answers come back as a list, the way the earlier adjudication rounds
worked.

### Round 15: the corpus-wide briefing gallery

```bash
sbatch --export=ALL,DETECT_CONFIG=configs/pi_briefing_detect.yaml slurm/v00_fm_scan.sbatch
$VIBES scripts/v00_fm_analyze.py --config configs/pi_briefing_detect.yaml
$VIBES scripts/sample_pi_briefing.py
sbatch --array=0-245%80 slurm/render_pi_briefing_array.sbatch
$VIBES scripts/build_pi_briefing.py
```

All four vendors on one page, 1,500 files each at seed 20270420 — disjoint from
every sample any threshold was fitted on. Four vendors as top-level contents
entries with seven strata nested inside each: the two passing conditions and one
per check. `configs/pi_briefing_detect.yaml` carries the whole threshold policy
in one place, each clause annotated with the round and the evidence that set it.

Six strata are empty by construction and the page names them rather than omitting
them. In the rates table a dash means a check is **not applied** to that vendor,
which is not the same as applied-and-found-nothing, and the totals row averages
each check only over the vendors it runs on.

### Round 14: FM4 measures dynamics

```bash
$VIBES scripts/probe_envelope_dynamics.py --scan <scan>/scan_with_flags.parquet --out <dir>/<tag>
$VIBES scripts/v00_fm_analyze.py --config configs/vendors_r11_detect.yaml
```

`Thresholds.audio_min_envelope_dynamics_db` is 1.5 dB. Static, buzz and silence
sit at one level whether anyone is speaking or not; a track carrying a voice
swings. On 48 hand labels the classes do not overlap — unusable 0.20–1.35 dB,
usable 1.70–69.04 — and the rule scores 13 of 13 with no false positive, against
28% precision for the old level floor and 38% for an empty released VAD. It needs
no VAD, no partner and no absolute level, which is why it works where those did
not. Both retired measures are still recorded.

FM4 remains off for V00 by instruction; §3 of the report has the case for
switching it back on, which would change zero verdicts.

### Round 13: the audio rule

```bash
$VIBES scripts/v00_fm_analyze.py --config configs/v00_r12_detect.yaml
$VIBES scripts/probe_vad_seconds.py --scan <scan>/scan_with_flags.parquet --out <dir>/<tag>
$VIBES scripts/sample_audio_rule_r13.py
sbatch --array=0-156%80 slurm/render_audio_r13_array.sbatch
$VIBES scripts/build_audio_rule_r13.py
```

`Thresholds.audio_dead_by_vendor` turns FM4 off for V00. Its six firings in a
4,000-file draw were all participants who simply speak rarely, and the cause is
that the floor is absolute while the vendors are not — V00's median speech level
is −41.8 dB, so −55 dB sits 13.2 dB below typical, against 36.4 dB on V01. A
vendor-relative floor does not fix it either; `audio_vad_seconds` does, since
every confirmed-dead file has zero released speech and an empty VAD occurs in 0
of 400 random V00 files. It is measured and recorded but **not** wired in: the
two rules disagree about 84 files elsewhere and that is unvalidated.

### Round 12: V00 re-examined

```bash
sbatch --export=ALL,DETECT_CONFIG=configs/v00_r12_detect.yaml slurm/v00_fm_scan.sbatch
$VIBES scripts/v00_fm_analyze.py --config configs/v00_r12_detect.yaml
$VIBES scripts/sample_v00_briefing_r12.py
sbatch --array=0-141%80 slurm/render_v00_r12_array.sbatch
$VIBES scripts/build_v00_briefing_r12.py
```

Three of the four V03 corrections were to choices that looked settled on V00
evidence, so all four were re-tested on V00's 66 labels. **All four survive**:
knee position beats hip flexion as a replacement (17 of 19 sitters for 1 of 47
standing lost, against 17 for 3); session-majority scope beats per-file (0 of 47
standing lost against 1); the shin add-on fires once in 4,000 files and that file
is seated; and the raster lesson closes by census — all 41,205 eligible V00 files
are 1080×1920 at 30/1, with no empty audio and no timebase drift, so FM0 fires on
nothing.

The gap is the labels' *coverage*: all 66 come from the flagged or adjudicate
pools, so FM1's false-negative rate above the cut has never been measured. See
`reports/13_v00_recheck.md`.

### Round 11: FM1 settled

```bash
$VIBES scripts/v00_fm_analyze.py --config configs/vendors_r11_detect.yaml
```

No rescan; Round 11 reuses Round 9's shards, as Round 10 did. FM1's full policy,
each clause with a label count behind it:

| vendor | measure | cut | shin add-on | scope |
|---|---|---|---|---|
| V00 | knee position | 0.43 | on | session majority |
| V01 | — off — | | | fires on 0 of 1,379 square-pixel files |
| V02 | — off — | | | 181 of 182 firings were the add-on |
| V03 | hip flexion | 146° | **off** | **per-file** |

The V03 add-on is off because across 149 hand labels it flagged 8 files hip
flexion did not and none of the eight was seated -- the ankles-out-of-frame case.
See `reports/12_fm1_final.md`; that report also records the two things still open,
neither of which is a posture question.

### Round 10: correcting FM1's verdict scope on V03

```bash
$VIBES scripts/v00_fm_analyze.py --config configs/vendors_r10_detect.yaml
$VIBES scripts/sample_vendor_briefing_r10.py
sbatch --array=0-174%80 slurm/render_vendors_r10_array.sbatch
$VIBES scripts/build_vendor_briefing_r10.py
```

No rescan: Round 10 reuses Round 9's measurement shards through a symlink,
because only the verdict policy moved. `Thresholds.sitting_unit_scope_by_vendor`
holds `session` for V00-V02 and `file` for V03. The session rule assumes posture
is constant within a participant-session, which holds for V00 (1.1% of multi-file
sessions mixed) and not for V03 (16.2%, median seated share 0.50 in those). On
108 labelled V03 files it loses both more sitters and more standing files than
reading each file on its own.

### Round 9: correcting the measure FM1 reads on V03

```bash
sbatch --export=ALL,DETECT_CONFIG=configs/vendors_r9_detect.yaml slurm/v00_fm_scan.sbatch
sbatch --array=0-47%48 \
  --export=ALL,DETECT_CONFIG=configs/vendors_r9_rare_detect.yaml slurm/v00_fm_scan.sbatch
$VIBES scripts/v00_fm_analyze.py --config configs/vendors_r9_detect.yaml
$VIBES scripts/v00_fm_analyze.py --config configs/vendors_r9_rare_detect.yaml
$VIBES scripts/sample_vendor_briefing_r9.py
sbatch --array=0-212%80 slurm/render_vendors_r9_array.sbatch
$VIBES scripts/build_vendor_briefing_r9.py
```

`Thresholds.sitting_by_vendor` now holds `(measure, cut)` rather than a bare
number, because V03 reads a different *measure*: hip flexion at 146°, against
V00's knee position at 0.43. On 90 V03 hand labels that is 44 of 45 sitters for 4
of 45 standing lost, where knee position at 0.54 gave 41 and 12 — better on both
axes while rejecting 26% of the vendor instead of 33%. Hip flexion was retired in
Round 5 on V00 evidence, which was correct for V00 and wrong to generalise.

Voice isolation (`audio_voice_isolation_db`) is recorded but **not** thresholded:
30 hand labels put accepted and rejected clips at the same values.

### Round 8: audio, and the first thresholds to move

```bash
sbatch --export=ALL,DETECT_CONFIG=configs/vendors_r8_detect.yaml slurm/v00_fm_scan.sbatch
sbatch --array=0-47%48 \
  --export=ALL,DETECT_CONFIG=configs/vendors_r8_rare_detect.yaml slurm/v00_fm_scan.sbatch
$VIBES scripts/v00_fm_analyze.py --config configs/vendors_r8_detect.yaml
$VIBES scripts/v00_fm_analyze.py --config configs/vendors_r8_rare_detect.yaml
$VIBES scripts/sample_vendor_briefing_r8.py
sbatch --array=0-203%80 slurm/render_vendors_r8_array.sbatch
$VIBES scripts/build_vendor_briefing_r8.py
```

Five checks now. **FM0** rejects a source that cannot be used whatever it shows
(V01's anamorphic rasters, V03's room-camera rasters, a timebase that misses its
video). **FM4** rejects a recording whose audio carries no voice — speech level
below −55 dB *and* a flat spectrum, which caught all four files the reviewer
confirmed dead and none of 300 V00 files. **FM1** is now per-vendor:
`Thresholds.sitting_by_vendor` holds 0.43 for V00, 0.54 for V03 from 58 hand
labels, and `None` for V01 and V02, which records as `retired_for_vendor` rather
than a clean pass. FM2 and FM3 are unchanged and stay comparable across every
round.

`scripts/probe_audio_quality.py` is where the audio work lives, including the
hypotheses that did **not** survive; see `reports/09_audio_and_thresholds.md` §2.

### Viewing a gallery

```bash
$VIBES scripts/serve_gallery.py artifacts/private_review_vendors_r8 --port 8000
ssh -N -L 8000:localhost:8000 <cluster-host>      # then open http://localhost:8000/
```

Three separate things had to be true for the progress bar to be draggable, and
all three now are:

1. **Keyframes in the clip.** libx264's default GOP put three in a thirty-second
   render, so a seek could only land on 0, 10 or 20 s. Now one per second.
2. **Range requests from the server.** `http.server` has never implemented HTTP
   `Range`; it answers a seek with `200` and the whole file, and browsers read
   that as "not seekable". `serve_gallery.py` answers `206`.
3. **A fallback for transports that cannot do either** — an editor preview pane,
   or a `file://` path. Each video upgrades to a full download as it scrolls into
   view, and once a clip is wholly buffered seeking needs no request at all. The
   pill under each clip says when that has happened, and a banner at the top of
   the page reports what the transport actually did with a range request.

The server binds to 127.0.0.1 only; participant media does not leave the cluster.

### PI briefing gallery

A read-only status page: what the three checks are for, how they are
implemented, what they did to a **fresh, independent** V00 sample, and a
stratified cross-section of the clips. It collects nothing — no notes, no
ratings, no inputs at all — because it is shown to someone who is being brought
up to speed rather than asked for judgements.

```bash
DETECT_CONFIG=configs/v00_fm_briefing_detect.yaml \
  sbatch --export=ALL,DETECT_CONFIG=configs/v00_fm_briefing_detect.yaml slurm/v00_fm_scan.sbatch
$VIBES scripts/v00_fm_analyze.py --config configs/v00_fm_briefing_detect.yaml
$VIBES scripts/sample_v00_briefing_gallery.py
sbatch slurm/render_v00_briefing_array.sbatch
$VIBES scripts/build_v00_briefing_gallery.py
```

The briefing scan uses seed 20260907 rather than the tuning sample's 20260814,
so the rates quoted on the page are measured out of sample — FM1's thresholds
were chosen against hand labels drawn from the other sample. Overlap between the
two draws is 9.6%.

Page at `artifacts/private_briefing_v00/index.html`, five sections of 60 clips:
improvised passes, naturalistic passes, and one per failure mode. FM2 clips are
positioned to contain that file's longest continuous SMPL-H-invalid stretch,
using the frame index the scanner recorded, so the frames the check objected to
are the frames on screen.

### Reviewer-vocabulary proxies and the V00 review (Round 2)

```bash
sbatch slurm/v00_pool_signals.sbatch      # framing/roll/posture/hands over 8,000 V00 files
$VIBES scripts/analyze_v00_pool.py        # naturalistic vs improvised comparison
$VIBES scripts/sample_v00_review.py --config configs/v00_review.yaml
sbatch slurm/render_v00_review_array.sbatch
```

Galleries land in `artifacts/private_review_v00/`: `naturalistic.html` and
`improvised.html` (150 uniform clips each), `targeted.html` (100 extremes), and
`index.html`. Each clip carries a 12-thumbnail whole-file filmstrip and an
upper-body crop panel. Both were superseded in Round 4: the crop panel was
retired and the filmstrip is now 10 thumbnails spanning frame 0 to the last
frame.

### Phase-0 human review (Round 1, corpus-wide)

```bash
$VIBES scripts/sample_review_gallery.py --config configs/session2_review.yaml
sbatch --array=0-299%32 slurm/render_review_gallery_array.sbatch
$VIBES scripts/render_review_gallery.py --config configs/session2_review.yaml \
  --html-only --gallery-name index
$VIBES scripts/review_clip_signals.py --manifest configs/session2_review_pass2.csv
```

Open `artifacts/private_review_session2/pass1_exploratory.html` over SSH
port-forward or VS Code Remote. **Participant media stays on the cluster**: the
directory is mode `0700`, media `0600`, git-ignored, and nothing is copied
off-cluster.

Once Pass-1 notes exist and a rubric derived from them is approved:

```bash
$VIBES scripts/render_review_gallery.py --config configs/session2_review.yaml \
  --manifest configs/session2_review_pass2.csv --html-only \
  --gallery-name pass2_structured --rubric configs/pass2_rubric.yaml
$VIBES scripts/correlate_review_ratings.py --ratings <export.json> \
  --rubric configs/pass2_rubric.yaml
```

No rubric content exists in this repository. The failure-mode vocabulary now
exists because the user produced it in the Round-1 review; its computable proxies
live in `src/seamless_curation/vocab_proxies.py`, one of which is retained
explicitly as a documented failure.

## Conventions

- **SMPL-H**: neutral model, 16 all-zero betas, `use_pca=False`,
  `flat_hand_mean=True` (empirically selected by the M-4 reprojection test),
  translation zeroed and pelvis subtracted for motion features, metres × 1000.
  The model asset is research-licensed: read in place, never copied, never
  committed.
- **Rotations**: geodesic distance via rotation composition, never axis-angle
  subtraction. Naive deltas exceed 100 rad/s on 43.5% of dev spans.
- **Participant identity**: `(vendor_id, participant_id)`. Bare IDs collide across
  vendors on 627 values.
- **Restart**: completion markers match both config hash and Git SHA, and a dirty
  worktree can never reuse one. Writes are same-directory temp then `os.replace`.
- **Git**: this repository is operated with
  `git --git-dir=.git-session --work-tree=.` because the platform mounts `.git`
  read-only.

## Tests

```bash
PYTHONPATH=src $VIBES -m pytest
```

The three dev spans used as fixtures are flag-clean and were visually reviewed
with the private 2D-overlay renderer on 2026-08-04.

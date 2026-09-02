# Seamless Interaction — co-speech upper-body curation

Builds a **manually verified** training subset of Meta's Seamless Interaction
dataset for ViBES upper-body training: clips in which the participant is
speaking *and* making genuine, sustained hand and arm gestures.

The source dataset is read-only and is never modified. The output is a
**manifest** — a CSV of `(file, start_frame, end_frame)` ranges — plus the
review artefacts and verdict log that produced it. No media is copied.

```
outputs/<run_id>/accepted_clips.csv             the training manifest
outputs/<run_id>/accepted_clips_with_audio.csv  the subset verified with sound
outputs/<run_id>/review_verdicts.jsonl          every verdict, append-only
```

---

## What this pipeline is for, and what changed

The previous version of this repository measured the corpus and built five
file-level failure-mode detectors (recording quality, seated posture, SMPL-H
validity, static hands, dead audio). The PI's review of its output was that the
thing that actually matters was not being tested:

> the most important issue is still whether the person actually makes meaningful
> hand and arm gestures while speaking […] we need co-speech upper-body motion
> […] please also make sure that small tracking jitter, global body movement, or
> a single brief hand adjustment is not mistakenly counted as meaningful
> gesturing.

Four things follow from that, and they are what this version is:

**1. Gesture is measured against speech, at window resolution.** The old static-
hands check (`FM3`) was whole-file, 2D-only, and never looked at the voice
activity data at all. Measured over 4,000 V00 files it correlates with how much
the participant *talks* at rho −0.38, so it mostly rejected quiet listeners; and
among files that passed the whole pipeline, 27.6% still had both wrists parked
within 0.10 shoulder widths of their own median for more than half the
recording. See [`reports/17_cospeech_gesture.md`](reports/17_cospeech_gesture.md).

**2. Posture, legs and framing no longer reject anything.** ViBES trains the
upper body. The seated-posture check is retired outright, and the SMPL-H
validity check — which alone rejected 62.6% of V00 by demanding a valid fit on
*every* frame of the file — is replaced by a per-window allowance plus a direct
measure of the damage an invalid frame actually does to the hands.

**3. Manual visual review is the acceptance criterion.** Automation narrows
118,570 eligible files to a candidate pool; nothing enters the manifest without
a recorded verdict against it. The old tooling could not support this: it had no
verdict store at all, its notes lived in browser `localStorage` under a key that
changed whenever a clip was re-rendered, and 17.9% of the clips it asked
reviewers to judge for co-speech gesture contained no speech.

**4. The review artefact was rebuilt for throughput.** A card renders in ~7 s,
is read in a few seconds, and shows the upper body at 250 px per thumbnail
instead of 19 px per hand.

## Where this run stands

`configs/vibes_upper_body.yaml` is the live run. Its state, as of the last
`seamless-curation stats`:

- **Scanned:** all 118,570 eligible files, 2,423,304 windows, 7,550 hours, zero
  read errors.
- **Candidates:** 414,504 qualifying windows → **73,883 clips / 616 hours** over
  31,815 files and 3,724 participants.
- **Rendered:** review cards for the first 4,000 items of the queue; 30-second
  audio clips for the first 400.
- **Reviewed:** 552 of the 31,815 review items, 67.9% accepted → **848 accepted
  clips / 7.07 hours** over 375 files and 375 participants. Live numbers are in
  `outputs/vibes_upper_body_v1/manifest_summary.json`. The verdicts recorded so
  far come from **vision model reviewers** working from
  [`docs/review_rubric.md`](docs/review_rubric.md), recorded as
  `verdict_source: model:claude-opus-5` with `saw_video: false`. They are a floor
  under the subset, not a substitute for a human pass — see
  [`reports/17_cospeech_gesture.md`](reports/17_cospeech_gesture.md) §4.
- **To extend it:** open the review app and work down the queue. Human verdicts
  supersede model ones on the items you reach (last write wins), disagreements
  are flagged `contested`, and `seamless-curation manifest` folds it all in. The
  candidate pool is fifty times larger than what has been reviewed, so accepted
  hours grow roughly linearly with review time at about **46 seconds of accepted
  data per file looked at**; reviewing all of it would yield roughly 407 hours.

---

## Quick start

```bash
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
VIBES=/simurgh/group/lw29/conda/envs/ViBES/bin/python
alias sc="$VIBES -m seamless_curation --config configs/vibes_upper_body.yaml"

sc population                                    # eligible files, from the M-1 inventory
sbatch --array=0-511%80 slurm/scan.sbatch        # measure every eligible file
sc gather                                        # concatenate shards, refusing gaps
sc select                                        # apply gates, choose candidates
SC_CARD_ONLY=1 sbatch --array=0-255%80 slurm/render.sbatch
sc review --port 8765                            # review; verdicts land on disk
sc manifest                                      # fold verdicts into the manifest
sc stats                                         # regenerate the run report
```

A second reviewer working outside the app — offline, or a model-assisted
pre-screen — takes the same queue as a list and hands its results back:

```bash
sc queue --unreviewed --format json > work.json  # items still needing a verdict
sc import-verdicts verdicts.jsonl                # append them to the same log
```

Run nothing heavy on a login node — every step above is either a Slurm job or a
few seconds of pandas. Review over an SSH tunnel:

```bash
ssh -N -L 8765:localhost:8765 <cluster-host>     # then open http://localhost:8765/
```

The server binds `127.0.0.1` only. Participant media never leaves the cluster:
`artifacts/` is mode 0700, its contents 0600, and both `artifacts/` and
`outputs/` are git-ignored. On this cluster `artifacts/` is a symlink to
`/simurgh/group/lw29/seamless_curation_artifacts` — the home export is 20 GB and
review artefacts run to about 1 GB per thousand reviewed files — following the
same pattern as the `datasets/` and `model_files/` symlinks.

---

## The stages

| stage | reads | writes | cost |
|---|---|---|---|
| `population` | M-1 inventory parquet | `population.parquet` | seconds |
| `scan` | NPZ + JSON per file | `scan_shards/*.parquet` | ~1.0 s/file, 512-task array |
| `gather` | shards | `windows.parquet`, `scan_files.parquet` | ~1 min |
| `select` | `windows.parquet` | `candidates.parquet`, `review_manifest.csv`, `gate_funnel.csv` | seconds |
| `render` | manifest + media | `artifacts/<run>/clips/*.card.png`, `*.clip.mp4` | ~7 s/card, ~30 s/clip |
| `review` | cards + clips | `review_verdicts.jsonl` | human time |
| `queue` | manifest + verdicts | the work list, for an offline reviewer | seconds |
| `manifest` | candidates + verdicts | `accepted_clips*.csv` | seconds |
| `stats` | everything above | `run_report.md` | seconds |

Re-tuning a threshold costs one `select` run and reads no media. Changing a
*measurement* changes the scan fingerprint, and every shard recomputes.

The fingerprint is a hash of the measurement parameters **and of the source of
`gesture.py` and `smplh_kinematics.py`**, rather than a version number someone
has to remember to bump; the marker also records *which files* the shard
covered, so a `--limit` smoke test cannot make the real scan a no-op and a
change of `--tasks` cannot leave a stale partition in place. `gather` then
refuses to proceed if any shard is missing rather than reporting a fraction of
the corpus as the whole of it.

### 1. Population

`corpus.py`. Eligibility is only about whether a file can be measured and
whether its source could support upper-body training at all: complete bundle,
readable video, a raster whose released SMPL-H is not corrupted, an annotation
grid that matches its container, and an interaction type that contains speech.
Nothing about posture, legs, or framing.

118,570 of 129,370 participant files are eligible (7,550 participant-hours).

### 2. Scan

`gesture.py` + `smplh_kinematics.py`. One pass per file produces a row per
sliding 30-second window (10 s hop). Forward kinematics is pure NumPy against
the zero-beta neutral model — 0.12 s for a 6,900-frame file, against ~2 s
through `smplx` on CPU — so the scan is I/O bound and the whole corpus takes
about two and a half hours of wall time on a 512-task array, NFS-bound.

Everything is measured in the **torso frame**: origin at the shoulder midpoint,
axes from the shoulder line and the pelvis-to-neck axis. A participant who
sways, turns or steps moves the frame with them, so global body motion cannot be
counted as gesture.

### 3. Gates and selection

`gates.py`. Eighteen clauses in three groups — tracking quality, co-speech
gesture, posture variety — each named, each with the reason it sits where it
does in its docstring. `gate_funnel.csv` reports the first failing clause per
window, so the cost of every clause is visible.

Qualifying windows are turned into non-overlapping clips (at most 8 per file),
and clips are grouped into **review items**, one per file. The review queue is a
stratified round robin over participants, so reviewing a prefix gives a set
spread across people, vendors and conditions rather than the most animated three
participants in the corpus.

### 4. Review artefacts

`review_card.py` and `review_renderer.py`. Two artefacts per review item:

- **`<id>.card.png`** — twelve moments sampled from inside the spans that would
  be accepted, each as an upper-body video crop with the released 2D arms drawn
  on it, paired with the pelvis-frame SMPL-H pose at the same instant; plus a
  whole-recording timeline of own speech, partner speech, arm speed, detected
  gesture episodes and the accepted spans.
- **`<id>.clip.mp4`** — 30 s of the item's median-scoring accepted span with the
  participant's own audio muxed, for the synchronisation check.

### 5. Review

`review_app.py`, `review_store.py`, and the rubric in
[`docs/review_rubric.md`](docs/review_rubric.md). A keyboard-driven page on
localhost: `A` accept, `R` then a digit to reject with a reason, `U` unsure, `V`
to play the clip with sound. Every verdict is POSTed and appended to
`review_verdicts.jsonl` before the UI advances.

The log is append-only and keyed by review item, so a verdict survives
re-rendering, several reviewers can work at once, and changing your mind is a new
record rather than an edit. `resolve()` takes the last write per item and flags
items where reviewers disagreed.

### 6. Manifest

`manifest.py`. An inner join between the candidate clips and the accepted
verdicts — so there is no code path in which an unreviewed clip reaches the
output. `accepted_clips.csv` is the training manifest;
`accepted_clips_with_audio.csv` is the subset whose reviewer played the clip
with sound.

Every row carries `reviewer`, `verdict_source` (`human` or a model identifier)
and `review_evidence`, so how a subset was verified is a column rather than a
claim.

---

## Conventions

- **SMPL-H**: neutral model, 16 all-zero betas, `use_pca=False`,
  `flat_hand_mean=True` (selected by the M-4 reprojection test). Because the
  betas are zero the skeleton is metrically identical in every file, so
  millimetre thresholds compare directly across participants. The model asset is
  research-licensed: read in place, never copied, never committed.
- **Rotations**: geodesic distance via rotation composition, never axis-angle
  subtraction — naive deltas exceed 100 rad/s on 43.5% of dev spans.
- **Participant identity**: `(vendor, participant_id)`. Bare ids collide across
  vendors on 627 values.
- **Determinism**: every seeded choice goes through a hash of string keys
  (`corpus.stable_unit_interval`), so nothing depends on row order, numpy
  version or platform.
- **Writes**: same-directory temp file then `os.replace`, everywhere.

## Layout

```
src/seamless_curation/   the pipeline; every module's docstring says why it exists
configs/                 one YAML per run, plus the M-1 inventory inputs
slurm/                   two array jobs (scan, render) and the M-1 inventory jobs
scripts/inventory_m1.py  the corpus census that population/ reads
docs/review_rubric.md    the text every reviewer works from
reports/                 the current report, and the v0 measurement rounds
tests/                   153 tests, no dataset or model needed except where skipped
NOTES.md                 surprises, dead ends, and open questions
```

## Tests

```bash
PYTHONPATH=src:. $VIBES -m pytest
```

Fixtures are synthetic bundles with a known answer — a wrist that traces a known
arc while speech is on, or one that only shakes in place — so the suite runs
without the 40 TB source mount. Tests that need the research-licensed SMPL-H
model skip if it is not staged.

`tests/test_regressions.py` holds one test per defect an adversarial review of
this pipeline confirmed, each naming the wrong behaviour it would produce. The
NumPy forward kinematics is checked against `smplx` itself in
`tests/test_smplh_kinematics.py`: that equivalence is what licenses the fast
scan, and if it drifts every millimetre threshold silently means something else.

## Regenerating the corpus census

`population` reads `outputs/02_inventory/summary/inventory_joined.parquet`,
produced by the M-1 inventory (header and stat metadata only — never a JSON,
NPZ or WAV payload byte):

```bash
sbatch slurm/inventory_m1_catalog.sbatch
sbatch slurm/inventory_m1_headers.sbatch
sbatch slurm/inventory_m1_summarize.sbatch
```

## History

`.git-session/` is an orphaned git directory holding the 71 commits of the v0
measurement work, from when the platform mounted `.git` read-only. It is
git-ignored and its content is byte-identical to the initial commit here; the
commit messages are the only record of how each v0 threshold was reached. That
history is also available in this repository as the `v0-history` branch.

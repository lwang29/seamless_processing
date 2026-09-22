# Seamless Interaction — co-speech upper-body curation

Builds a training subset of Meta's Seamless Interaction dataset for ViBES
upper-body training: clips in which the participant is speaking *and* making
genuine, sustained hand and arm gestures.

**The pipeline is fully automated.** It takes the read-only release and a config
file and produces a manifest, with no human or model in the loop. Every
criterion a reviewer used to apply has been converted into a measured clause
with a stated threshold; review tooling still exists, but only to produce the
labelled data those thresholds are calibrated and validated against.

The source dataset is never modified. The output is a **manifest** — a CSV of
`(file, start_frame, end_frame)` ranges — and no media is copied.

```
outputs/<run_id>/accepted_clips.csv     the training manifest, fully automated
outputs/<run_id>/qualified_clips.parquet every candidate clip with scores + flags
outputs/<run_id>/gate_funnel.csv         tier-1: what each clause removed
outputs/<run_id>/qualification_funnel.csv tier-2: what each clause removed
outputs/<run_id>/export/                 the packaged handover + dataset card
outputs/<run_id>/reviewed_clips.csv      [development] the reviewed subset (mostly model verdicts)
outputs/<run_id>/review_verdicts.jsonl   [development] every verdict, append-only
```

> **Just want to train on the data?** You do not need any of this. Read
> [`docs/using_the_subset.md`](docs/using_the_subset.md) — one CSV and one
> loader, no pipeline run required. A self-contained copy for people who cannot
> read this home directory is staged at
> `/simurgh/group/lw29/seamless_cospeech_subset/`:
>
> ```python
> from seamless_curation.dataset import load_manifest, iter_clips
> manifest = load_manifest("outputs/vibes_upper_body_v1/export/clips_accepted.csv")
> for clip in iter_clips(manifest, "seamless_interaction"):
>     clip.upper_body_pose, clip.left_hand_pose, clip.audio, clip.speech
> ```

> **Want to know exactly what the filter does and why?**
> [`docs/pipeline.md`](docs/pipeline.md) documents every step: what it does, why
> it is necessary, what it reads, its thresholds, what makes a clip pass or
> fail, its assumptions and its known failure modes.

---

## The question the pipeline asks

Of every 30-second window of every participant recording:

> While this person is speaking, are their hands and arms making natural,
> visible co-speech gestures?

Technical cleanliness is **not** sufficient, and that is the whole point. A
recording with flawless SMPL-H tracking, clean audio and a participant whose
hands rest in their lap for the entire conversation is a reject. The five things
the pipeline exists to exclude:

| excluded | how |
|---|---|
| hands essentially static while speaking | gesture measured **only inside the participant's own VAD**; floors on the share of speaking time that is active and on how high the hands are carried |
| too little visible upper-body movement | posture-variety measures — does the arm visit *different* places, not just move |
| tracking noise / SMPL-H jitter | a frame must show **travel**, not just speed; plus two guards that test *direction* and *cross-channel agreement*, never magnitude |
| global body movement rather than articulation | everything measured in a **torso frame**, so swaying and stepping are near-zero by construction; plus an arm-vs-torso ratio |
| a single brief adjustment or transient | episode structure: how many separate episodes, how long each, and what share of utterances they cover |

And one thing it exists *not* to exclude: **a subtle but genuine gesturer.** The
tier-2 decision is a weighted score, not a conjunction of tight thresholds, so
being modest on amplitude is survivable if persistence, coherence and
speech-locking are good. `test_subtle_but_genuine_gesture_is_not_discarded` is
the guard on that, and it matters as much as the exclusions: a filter that
rejects everything satisfies every exclusion test ever written.

---

## Where this run stands

- **Scanned:** all 118,570 eligible files, 2,423,304 windows, 7,550 hours, zero
  read errors.
- **Tier-1 gates:** 414,504 windows qualify (17.1%) → **73,883 candidate clips /
  615.7 hours** over 31,815 files and 3,724 participants.
- **Tier-2 qualification:** **50,516 clips / 420.9 hours** over 24,114 files and
  3,501 participants. This is `accepted_clips.csv`, produced with no reviewer.
- **Measured accuracy.** Against 100 independently human-reviewed files (all of
  which had passed tier 1, so this is tier 2's own accuracy): **precision
  0.936, recall 0.948**, catching 8 of 13 files the human rejected. The
  do-nothing baseline — accept every tier-1 candidate — is precision 0.904.
- **Development labels retained:** 686 verdicts, of which 100 are human and were
  taken with audio. `reviewed_clips.csv` holds 1,030 clips / 8.6 hours.

Live numbers: `outputs/vibes_upper_body_v1/manifest_summary.json`. The
calibration evidence is
[`reports/18_automated_qualification.md`](reports/18_automated_qualification.md).

---

## Quick start

```bash
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
VIBES=/simurgh/group/lw29/conda/envs/ViBES/bin/python
alias sc="$VIBES -m seamless_curation.cli --config configs/vibes_upper_body.yaml"

sc population                                    # eligible files, from the M-1 inventory
sbatch --array=0-511%80 slurm/scan.sbatch        # measure every eligible file
sc gather                                        # concatenate shards, refusing gaps
sc select                                        # tier-1 gates, choose candidate clips
sc qualify                                       # tier-2 decision: score, flag, accept
sc manifest                                      # write accepted_clips.csv
sc export                                        # package + dataset card
sc verify --sample 60                            # read sampled rows back out of the release
sc stats                                         # regenerate the run report
```

Only `scan` needs a cluster; everything after it is seconds of pandas over
parquet. **Re-tuning a tier-2 threshold costs one `sc qualify` run and reads no
media** — edit the `qualify:` block in the config and re-run.

Development-only, not part of producing the manifest:

```bash
SC_CARD_ONLY=1 sbatch --array=0-255%80 slurm/render.sbatch   # review cards
sc review --port 8765                            # label items to calibrate against
sc queue --unreviewed --format json > work.json  # or label them offline
sc import-verdicts verdicts.jsonl
```

---

## The stages

| stage | reads | writes | cost |
|---|---|---|---|
| `population` | M-1 inventory parquet | `population.parquet` | seconds |
| `scan` | NPZ + JSON per file | `scan_shards/*.parquet` | ~1.0 s/file, 512-task array |
| `gather` | shards | `windows.parquet`, `scan_files.parquet` | ~1 min |
| `select` | `windows.parquet` | `candidates.parquet`, `gate_funnel.csv` | seconds |
| `qualify` | `candidates.parquet` | `qualified_clips.parquet`, `qualification_funnel.csv` | seconds |
| `manifest` | candidates + qualification | `accepted_clips.csv` (+ reviewed subset) | seconds |
| `export` | manifests | `export/clips_*.csv`, `DATASET.md` | seconds |
| `verify` | manifest + source tree | pass/fail on sampled rows | seconds |
| `stats` | everything above | `run_report.md` | seconds |
| `render` | manifest + media | review cards and clips | *development* |
| `review` / `queue` / `import-verdicts` | cards | `review_verdicts.jsonl` | *development* |

Changing a *measurement* changes the scan fingerprint and every shard
recomputes. The fingerprint hashes the measurement parameters **and the source
of `gesture.py` and `smplh_kinematics.py`**, rather than a version number
someone has to remember to bump; the shard marker also records *which files* it
covered, so a `--limit` smoke test cannot make the real scan a no-op. `gather`
refuses to proceed if any shard is missing.

---

## How the decision is made

Full detail, including every threshold and its justification, is in
[`docs/pipeline.md`](docs/pipeline.md). In brief:

**Tier 1** (`gates.py`, 18 clauses) asks *is this window usable* — tracking
integrity, enough speech, motion that is not jitter, arms that visit more than
one posture. It is deliberately permissive, because it runs over 2.4 M windows
before selection and cannot tell a modest gesturer from a non-gesturer without
deleting both.

**Tier 2** (`qualify.py`, 9 disqualifiers + a score) makes the accept/reject
call that a reviewer used to make. The split exists because that judgement is
one of *degree*, and degree can only be judged after eligibility. On 100
human-labelled files that had all already passed tier 1, the reviewer's
judgement proved predictable from measurements the scan already produces —
`wrist_height_p75_mm` alone separates accept from reject at AUC 0.85.

Two design choices carry most of the weight:

- **No gate is a jitter-magnitude gate.** 106 of 135 candidate quality signals
  correlate with gesture activity at |rho| up to 0.903, so gating on them is
  arithmetically a gate on how much the person gestured. The two noise guards
  used instead compare *independent channels* and test *direction*.
- **Peak wrist excursion is excluded from the score.** It separates the labelled
  set backwards — rejects have larger peak excursion than accepts — because it
  rewards one big isolated adjustment. `V02_S5281_I00000280_P5272` is top-decile
  on every activity measure and its entire score comes from twice adjusting a
  beanie.

Every clip, kept or dropped, carries its `gesture_quality`, four dimension
scores, and an `exclusion_flags` string listing **every** clause it failed. Both funnels
report every clause including ones that fired zero times, because "0" and
"absent" are different facts and a mis-wired clause looks exactly like an
absent one.

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
docs/pipeline.md         every step: what, why, thresholds, limitations
docs/using_the_subset.md written for a downstream user, assumes no pipeline knowledge
docs/review_rubric.md    [development] the rubric the tier-2 clauses were derived from
reports/                 the current reports, and the v0 measurement rounds
tests/                   the acceptance suite; no dataset or model needed except where skipped
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

`tests/test_automated_qualification.py` is the acceptance suite. Each of the
five exclusions is built as a bundle whose motion is exactly that failure, run
through real measurement and both gate tiers, and asserted to be rejected *with
the right reason* — alongside the positive cases, including the subtle one.

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

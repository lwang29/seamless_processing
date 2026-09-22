# Using the co-speech subset

For anyone training on the filtered Seamless Interaction data. You do not need
to know how the filtering works to use this, and you do not need to run any of
the pipeline. One CSV and one loader.

> **If you are not `lw29`:** this repository lives under a home directory you
> cannot read (`/sailhome/lw29` is mode `drwxr-s---`). A self-contained copy —
> the manifests, the loader package, the dataset card and the pipeline
> documentation — is staged on group storage at
> **`/simurgh/group/lw29/seamless_cospeech_subset/`**, and its `README.md` is
> this document with every path rewritten to work from there. Use that copy.
> The dataset itself is at `/simurgh2/datasets/seamless_interaction`, which is
> group-readable.

## What you are getting

Every row is **a frame range of one participant recording**. Nothing has been
copied, re-encoded or moved: the release stays where it is and the manifest
points into it. The subset costs no disk, and re-filtering it later costs no
re-export.

| file | clips | hours | files | participants | how it was decided |
|---|---:|---:|---:|---:|---|
| `clips_accepted.csv` | 50,741 | 422.8 | 24,204 | 3,504 | fully automated |
| `clips_reviewed.csv` | 1,030 | 8.6 | 498 | 490 | automated, and a reviewer also accepted it |

Both are under `/simurgh/group/lw29/seamless_cospeech_subset/`, next to a
`DATASET.md` documenting every column.

**Use `clips_accepted.csv`.** It is the production subset and what the pipeline
is for. `clips_reviewed.csv` is a development artefact — the labelled set the
automated decision was validated against. It is much smaller and will stay that
way. **Most of its reviewers were models, not people**: 17% of its rows carry
`verdict_source == "human"`, and only 181 rows are verdicts a person took with
the clip playing (`review_evidence == "card+video"`). Read `verdict_source`
per row rather than treating the file as human sign-off.

## How good is it?

Measured against 100 recordings independently reviewed by hand with audio:

- Of the files the pipeline accepts, **93.6%** were also accepted by the human.
- Of the files the human accepted, the pipeline keeps **91%**.
- Skipping the final decision step entirely would give 86% — so the step raises
  precision from 0.86 to 0.936 while keeping nine-tenths of the good material.

The validation set is small (100 files, 13 of them rejections). That is enough
to show the step helps and roughly where it sits; it is not enough to state the
accuracy to the decimal place.

## Quickstart

```bash
conda activate /simurgh/group/lw29/conda/envs/ViBES
cd /sailhome/lw29/seamless_processing

# Sanity check: load the first three clips and print what came back.
PYTHONPATH=src python -m seamless_curation.dataset \
    outputs/vibes_upper_body_v1/export/clips_accepted.csv seamless_interaction
```

```python
import sys; sys.path.insert(0, "src")
from seamless_curation.dataset import load_manifest, iter_clips

manifest = load_manifest("outputs/vibes_upper_body_v1/export/clips_accepted.csv")

for clip in iter_clips(manifest, "seamless_interaction"):
    clip.upper_body_pose    # (frames, 13, 3) float32 axis-angle — the body input
    clip.left_hand_pose     # (frames, 15, 3)
    clip.right_hand_pose    # (frames, 15, 3)
    clip.global_orient      # (frames, 3)    — usually dropped for root-relative training
    clip.translation        # (frames, 3)
    clip.smplh_valid        # (frames,) bool — the release's own validity flag
    clip.audio              # (samples,) float32, that participant only
    clip.sample_rate        # 48000
    clip.speech             # ((start_s, end_s), ...) rebased to the clip
    clip.fps, clip.frames, clip.seconds
```

`load_clip(row, source_root)` reads **only** the frames the row names — a
30-second clip out of a four-minute recording costs a 30-second read. Pass
`with_audio=False` to skip the WAV when training pose-only. For a long run,
`iter_clips(..., skip_errors=True)` downgrades an unreadable clip to a warning
instead of killing the job.

## The 13 upper-body joints

`upper_body_pose` is `smplh:body_pose` indexed to spine 1-3, neck, head, both
collars, both shoulders, both elbows and both wrists — in that order, as
`seamless_curation.dataset.UPPER_BODY_ROWS`. The pelvis is not in it; the
release carries pelvis rotation separately as `global_orient`. Legs are
deliberately absent: lower-body quality was out of scope for this filtering, so
it was never checked and should not be trusted.

Hands are 15 joints per side, axis-angle, and are **not** PCA coefficients. If
you build an SMPL-H model to render these, use `use_pca=False` and
`flat_hand_mean=True`, with 16 all-zero betas and the neutral model — the
configuration the pipeline was validated against (agreement with `smplx` to
under a micrometre).

## Splits

`split` is the **release's own** train/dev/test split, preserved unchanged:

| tier | train | dev | test |
|---|---:|---:|---:|
| accepted | 44,130 | 1,148 | 876 |
| reviewed | 845 | 15 | 15 |

Use `vendor + ":" + participant_id` as the participant key. `participant_id` is
unique only within a vendor, and some ids carry a letter suffix (`0040A`), so
keep it as a string — `load_manifest` already does.

One participant appears in several files and a file contributes several clips.
Caps are applied (≤8 clips per file, ≤12 files per participant), but if you are
splitting for evaluation, **split on participant, not on clip**, or the same
person appears on both sides.

## What the filter guarantees, and what it does not

For every row, inside that frame range, the participant speaks for at least 8
seconds and their arms are demonstrably active during that speech — active in a
**torso frame**, so swaying and stepping cannot masquerade as gesture; over at
least three separate episodes covering at least 60% of what they said, so a
single adjustment cannot qualify; with motion that both travels and is
directionally coherent, so tracking jitter cannot qualify; and with the hands
carried high enough and through enough different positions that "technically
moving but parked in a lap" cannot qualify.

It does **not** guarantee that gesture is well-synchronised with speech beyond
co-occurrence. `sync_r` and `sync_lag_s` are carried in the manifest but gate
nothing, because no labelled data exists to calibrate a threshold against.

Full detail — every step, threshold and known failure mode — is in
[`pipeline.md`](pipeline.md); the plain-language version is
[`../reports/pipeline_in_plain_language.md`](../reports/pipeline_in_plain_language.md).

## Re-filtering without re-running anything

Every row carries its quality score, its four dimension scores and a `exclusion_flags`
string listing every rule it failed:

```python
strict = manifest[manifest.gesture_quality > 0.6]                  # ~top half
hands_high = manifest[manifest.wrist_height_p75_mm > -100]         # very active posture
lively = manifest[manifest.dim_persistence > 0.8]                  # sustained gesturing
```

## Caveats

1. **Clips from one file are disjoint frame ranges, not contiguous.** Selection
   vetoes overlap. Concatenating them is safe; they are not continuous.
2. **The thresholds are calibrated, not derived.** They come from agreement with
   a labelled sample and are meant to be moved with evidence.
3. **`charades` interactions and four raster formats are excluded** upstream.
4. **Audio is that participant's own channel**, 48 kHz mono float32, already
   separated in the release. There is no partner voice to remove.
5. **V03 is the weakest vendor** in the reviewed sample — human accept rate 0.64
   against 0.85 elsewhere (n=100, Fisher p=0.027), largely because participants
   there often hold a printed prompt sheet. If you want the cleanest data
   cheaply, dropping V03 is the single biggest lever.

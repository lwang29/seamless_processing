# Using the co-speech subset

For anyone training on the filtered Seamless Interaction data. You do not need
to know how the filtering works to use this, and you do not need to run any of
the pipeline. Two CSV files and one loader.

> **If you are not `lw29`:** this repository lives under a home directory you
> cannot read (`/sailhome/lw29` is mode `drwxr-s---`). A self-contained copy —
> both manifests, the loader package, the dataset card and the review rubric —
> is staged on group storage at
> **`/simurgh/group/lw29/seamless_cospeech_subset/`**, and its `README.md` is
> this document with every path rewritten to work from there. Use that copy.
> The dataset itself is at `/simurgh2/datasets/seamless_interaction`, which is
> group-readable.

## What you are getting

Every row is **a frame range of one participant recording**. Nothing has been
copied, re-encoded or moved: the release stays where it is and the manifest
points into it. That means the subset costs no disk, and re-filtering it later
costs no re-export.

| file | clips | hours | what it is |
|---|---:|---:|---|
| `clips_verified.csv` | 875 | 7.3 | a human watched and accepted each of these files |
| `clips_candidate.csv` | 72,728 | 606.0 | passed all 18 automated gates, not yet reviewed |

Both are under `outputs/vibes_upper_body_v1/export/`, next to a `DATASET.md`
that documents every column and a `summary.json` with the same numbers in
machine-readable form.

**Which to use.** Start with `clips_verified.csv` if you want certainty and 7
hours is enough. Use `clips_candidate.csv` if you want volume: about **80% of it
would survive human review** (Wilson 95% interval 71–87%, measured on 100
reviewed items from the same pool). The 20% that would not is almost entirely
one failure mode — someone whose hands move enough to clear the thresholds but
who is not really gesturing, e.g. hands clasped at the waist, or repeatedly
adjusting a hat. The SMPL-H is still valid and still in sync; it is a weaker
training signal, not broken data.

Files that a reviewer looked at and **rejected** are excluded from both tiers,
so the candidate tier is strictly better than raw gate output.

## Quickstart

```bash
conda activate /simurgh/group/lw29/conda/envs/ViBES
cd /sailhome/lw29/seamless_processing

# Sanity check: load the first three clips and print what came back.
PYTHONPATH=src python -m seamless_curation.dataset \
    outputs/vibes_upper_body_v1/export/clips_verified.csv seamless_interaction
```

```python
import sys; sys.path.insert(0, "src")
from seamless_curation.dataset import load_manifest, iter_clips

manifest = load_manifest("outputs/vibes_upper_body_v1/export/clips_verified.csv")

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
`with_audio=False` to skip the WAV entirely when you are training pose-only.

For a long run, `iter_clips(..., skip_errors=True)` downgrades an unreadable
clip to a warning instead of killing the job.

## The 13 upper-body joints

`upper_body_pose` is `smplh:body_pose` indexed to spine 1-3, neck, head, both
collars, both shoulders, both elbows and both wrists — in that order, as
`seamless_curation.dataset.UPPER_BODY_ROWS`. The pelvis is not in it; the
release carries the pelvis rotation separately as `global_orient`. Legs are
deliberately absent: lower-body quality was out of scope for this filtering, so
it was never checked and should not be trusted.

Hands are 15 joints per side, axis-angle, and are **not** PCA coefficients.
If you build an SMPL-H model to render these, use `use_pca=False` and
`flat_hand_mean=True`, with 16 all-zero betas and the neutral model — that is
the configuration the whole pipeline was validated against (agreement with
`smplx` to under a micrometre).

## Splits

`split` is the **release's own** train/dev/test split, preserved unchanged.
Filtering did not rebalance it:

| tier | train | dev | test |
|---|---:|---:|---:|
| verified | 845 | 15 | 15 |
| candidate | 69,317 | 1,813 | 1,598 |

**The verified tier has only 30 non-train clips**, which is not enough to
evaluate on. Either evaluate on the candidate tier's dev/test (3,411 clips,
~28 hours, unreviewed), or hold out verified *participants* from train. The
review queue was ordered as a stratified round-robin over participants, not over
splits, which is why train dominates — it is an artefact of review order, not of
the filtering, and it evens out as more review is done.

Use `vendor + ":" + participant_id` as the participant key. `participant_id` is
unique only within a vendor, and some ids carry a letter suffix (`0040A`), so
keep it as a string — `load_manifest` already does.

One participant can appear in several files and a file contributes several
clips. Caps are applied (≤8 clips per file, ≤12 files per participant), but if
you are splitting for evaluation, **split on participant, not on clip**, or the
same person will appear on both sides.

## What the filtering actually guarantees

For every row in either tier, inside that frame range:

- the released SMPL-H fit is valid on ≥90% of frames, with no invalid run
  longer than 1 s, and <5% of frames have a frozen hand-pose vector;
- the participant speaks for ≥8 s, and their arms are active during ≥35% of
  that speech, spread over ≥3 distinct episodes covering ≥40% of utterances;
- the wrists travel (≥80 mm excursion) and the elbows are involved (≥35 mm), in
  a **torso frame** — so swaying, turning and walking cannot masquerade as
  gesture;
- the arms visit genuinely different postures rather than one posture repeated;
- two independent noise guards agree the motion is a person, not the tracker.

Measures are the **maximum over the two hands**, so one-handed gesturing counts.

Every gate measurement is carried in the CSV, so you can re-filter harder
without re-running anything:

```python
strict = manifest[
    (manifest.gesture_frac_speech > 0.55) & (manifest.posture_spread_mm > 220)
]
```

## Caveats

1. **The verified tier is small because review is the bottleneck, not the data.**
   Hours grow roughly linearly with review effort.
2. **Clips from one file are disjoint frame ranges, not contiguous.** Selection
   vetoes overlap. Do not concatenate them and treat the result as continuous.
3. **`sync_r` / `sync_lag_s` are correlations, not a sync guarantee.** Audio and
   pose come from the same recording and share a clock, so gross desync is not a
   worry, but these columns describe gesture-speech envelope correlation, which
   is a property of the person, not of the alignment.
4. **`charades` interactions and four raster formats are excluded** upstream.
5. **V03 is 41% of the pool and the weakest vendor** — human accept rate 64% vs
   85% elsewhere, largely because participants there often hold a printed prompt
   sheet. If you want the cleanest candidate-tier data cheaply, dropping V03 is
   the single biggest lever.

## Questions this document cannot answer

Why a particular threshold is where it is: `src/seamless_curation/gates.py` has
the rationale for all eighteen, and `reports/17_cospeech_gesture.md` has the
measurements behind them. What a reviewer was asked to judge:
`docs/review_rubric.md`.

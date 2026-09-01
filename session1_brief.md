# Session 1 brief — Seamless Interaction curation: reconnaissance and harness

## Context

I'm building a curated, high-quality training subset of the **Seamless Interaction Dataset** for a co-speech gesture generation model (speech → upper-body + hand motion). My PI's assessment is that a large fraction of the corpus is too noisy to train on, and I need a defensible, reproducible way to find the clean part.

A full design document is at `<DESIGN_DOC>`. **Read it first.** It proposes a 16-method, staged pipeline (integrity gates → cheap motion signals → audio/attribution → sync → pose cross-checks → learned scoring → span extraction → diversity-aware selection → human verification → downstream ablation).

**Important framing: that document was written without access to the data.** It was built from the paper, the dataset card, and the repo README. Some of its assumptions about file layout, array semantics, coordinate frames, and units are probably wrong. Your primary job in this session is **not** to implement the pipeline — it is to establish ground truth about what is actually on disk, and to build the minimal harness that later sessions will scale. Finding places where the design doc is wrong is the single most valuable thing you can produce today.

## Hard constraints

1. **`<DATA_ROOT>` is READ-ONLY.** Never write, move, delete, or `chmod` anything under it. Every output goes under `<WORK_DIR>`. If you think you need to modify source data, stop and ask.
2. **Do not process the full corpus.** This session: the **`dev` splits only** (both `improvised` and `naturalistic`), and within those, no more than ~200 participant-files unless I say otherwise. The full corpus is ~4,000 dyadic hours / ~130k files / tens of TB and there is no version of "let me just scan everything" that is a good idea today.
3. **No heavy compute on the login node.** Anything over ~5 minutes or ~4 cores goes through Slurm (`srun`/`sbatch`). Check `sinfo`, `squeue`, and any group quota before submitting, and tell me what you find.
4. **Do not assume network access from compute nodes.** Before designing around any pip install or model checkpoint download, *test whether it works* and report. If it doesn't, say so early — it changes the plan materially.
5. **Measurement only this session.** No smoothing, resampling, interpolation, filtering, or "fixing" of any data. No threshold selection. No filtering decisions.
6. **This is human participant video.** Keep all media inside the cluster. No copying clips off-cluster, no uploads to third-party services, no putting frames in a shared/world-readable location. Dataset is CC-BY-NC 4.0; research use only.
7. **Verify or flag.** If a fact matters and you can't confirm it from the data, write "unverified" next to it. Do not infer schema from the README and present it as observed.

## Start here: plan first

Before writing any code, give me a short written plan: what you'll inspect, what you'll create, what Slurm resources you'll request, and in what order. **Wait for my approval.** Use plan mode if that's convenient.

## Deliverables, in priority order

**D1 — Environment and resource report.** Python version and how to get a usable env (module system? conda? uv? existing venv?); GPU type and count and how to request them; whether compute nodes have outbound network; `<WORK_DIR>` filesystem type and quota; Slurm partitions, time limits, and max array size; whether `ffmpeg`/`ffprobe` exist and their build options. Also: measured read throughput from `<DATA_ROOT>` (this, not FLOPs, will be the real bottleneck later).

**D2 — Observed schema report.** The actual directory layout under `<DATA_ROOT>`, compared line-by-line against what the design doc's §0 assumes. File counts and total size per `label`/`split`. For ~20 random files: the exact `.npz` keys, array shapes, and dtypes. Which modalities are actually present per file, and whether the availability flags in `filelist.csv` agree with the filesystem. The real schemas of the `transcript/` and `vad/` JSONL files — paste two example records of each verbatim. The actual columns of `filelist.csv`, `interactions.csv`, `participants.csv`, `relationships.csv`, with dtypes and null rates.

**D3 — `is_valid` characterization. This is the most important unknown in the whole project.** For ≥100 dev files, for each of `smplh:is_valid`, `movement:is_valid`, and `boxes_and_keypoints:is_valid_box`:
- fraction of invalid frames (distribution across files, not just the mean);
- run-length distribution of invalid stretches — are they isolated frames or long bursts?
- pairwise agreement between the three flags;
- **what the underlying data does on invalid frames**: is it zero-filled, NaN, held-over from the last valid frame, or plausible-looking garbage? This determines whether interpolation is even possible and it cannot be guessed.
- a **visual check**: render ~10 short clips that span invalid regions with the 2D keypoints overlaid on the video, and tell me in words what invalidity actually corresponds to (hands out of frame? occlusion by the partner? motion blur? nothing visible at all?).

For rendering, use the released 2D keypoints drawn on decoded frames with OpenCV + ffmpeg. **Do not fight with mesh/EGL/pyrender headless rendering today** — it's a known time sink and 2D overlays answer the question.

**D4 — Assumption audit.** Check each of these against the data and report verified / refuted / unverifiable:
- Is body shape actually canonical (β = 0 or absent)? Are inter-joint distances constant within and across participants?
- Units, origin, and axis convention of `smplh:translation` and `smplh:global_orient`. Is the coordinate frame camera-relative or world-ish?
- Rotation representation of `body_pose` / hand poses: axis-angle, rotation matrices, or 6D? Confirm from shapes *and* from numerical range.
- Are hand poses expressed relative to each wrist, as the paper claims after its post-processing step?
- Frame rate: exactly 30, or 29.97, and does it vary by vendor? Does `n_frames / fps` match audio duration, and what's the distribution of the mismatch?
- Do the two participants' files for one interaction have equal frame counts and aligned time origins?
- Are keypoints 2D pixel coordinates in the released 1080p frame, or normalized, or in the original 4K portrait resolution? Do they carry per-keypoint confidences?

**D5 — Minimal harness.** A restartable, config-driven, single-pass extractor: file list → per-file worker → per-window Parquet rows. Ship it with **only four features** — `valid_frac_all`, `accel_mm_per_frame2`, `wrist_speed_p90`, `duration_mismatch_s` — because the point is to prove the plumbing, not the feature set. Requirements: atomic temp-file-then-rename writes; per-file completion markers so reruns skip finished work; git SHA and config hash stamped into every row; window definition (4 s / 1 s hop at 30 Hz) read from config, not hardcoded; a Slurm array submission script; and a `--limit N` flag. `git init` the work dir and commit as you go.

**D6 — Synthetic corruption test suite (scaffold + ≥3 passing tests).** Take a few known-clean dev spans, programmatically inject a fault, and assert the intended detector fires and the others don't. Minimum three: (a) drop/duplicate frames → `duration_mismatch_s` / zero-velocity run detection; (b) inject jitter of known amplitude → `accel`; (c) blank a hand for 20 frames → `valid_frac` / hand availability. Make it `pytest`-runnable in under a minute. This suite is how we catch the sign errors and off-by-one bugs that would otherwise silently poison the final dataset, so treat it as load-bearing infrastructure, not a nicety.

## Reporting

Write everything to `<WORK_DIR>/reports/01_recon.md`. Numbers, tables, and file paths — not prose assurances. Include the exact commands you ran so I can reproduce them. Keep a running `<WORK_DIR>/NOTES.md` of surprises, dead ends, and open questions.

End your final message with three lists: **(1) design-doc assumptions that turned out to be wrong**, **(2) decisions you need from me before session 2**, and **(3) anything that looks like it will be expensive or impossible on this cluster.**

## Do NOT do this session

No thresholds or filtering decisions. No scoring model. No clustering. No VLM or LLM calls on the data. No ASR, SyncNet, pyannote, or pose re-estimation runs. No full-corpus jobs. No downloading large checkpoints. No writing to `<DATA_ROOT>`. If you find yourself with spare capacity, deepen D3 rather than starting D5+ features early — a wrong understanding of `is_valid` invalidates everything downstream.
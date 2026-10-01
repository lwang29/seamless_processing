# Seamless Interaction — annotation of every recording and clip

Annotates Meta's Seamless Interaction release comprehensively, so that
downstream users can threshold, filter, sort, stratify, aggregate or condition a
model on any property themselves. **Nothing is filtered out.** Every one of the
129,370 recordings in Meta's filelist gets a row (with a status saying why if it
cannot be measured), and every 30-second clip of every measured recording gets a
row — seated or standing, silent or talking, occluded, anamorphic, partial or
dull, each annotated with what it is and how sure we are.

The release is read in place and never modified or copied.

> **Using the tables?** Read [`docs/using_the_annotations.md`](docs/using_the_annotations.md)
> (levels, joins, missing values, recipes, leakage-free splits). Every column:
> [`docs/annotation_schema.md`](docs/annotation_schema.md). How each annotation
> is computed and validated: [`docs/pipeline.md`](docs/pipeline.md).

## What is annotated

Eight tables, one per level, joined by string keys that nest
(`session_key` ⊂ `interaction_key` ⊂ `file_id` ⊂ `clip_id`):

| table | rows | examples of what it carries |
|---|---|---|
| `participants` | 4,307 | BFI-2 (or why not), official splits and leakage flags, id-suffix siblings, co-participation component |
| `sessions` | 5,102 | relationship, dyad key and recurrence, posture consistency across the session |
| `prompts` | 1,312 | interaction type, IPC codes decoded to agency/communion, texts, text-vs-type consistency, task kind |
| `interactions` | 64,751 | pairing status, duration agreement, speech overlap / balance / turn-taking, posture pair, expressivity mean and gap, MOI coverage and copying, split leakage counts |
| `interaction_windows` | 515,749 | the dyad over each clip interval: overlap, mutual silence, floor changes, posture pair |
| `recordings` | 129,370 | media/raster facts, measurement status, tracking quality, speech source, audio quality and speaker bleed, recording posture and transitions, expressivity summary, MOI status |
| `clips` | 1,027,960 | posture (standing / sitting / mixed / unclear / unknown + confidence), expressivity (score, level, four channels + face and voice), arm/hand/head motion, speaking role, visibility and framing, facing, reprojection error, audio levels, MOI counts, redundancy, visual quality, partner clip |
| `moi_events` | 16,256 | Meta's 1P/3P moment-of-interest entries, as released, with malformation and cross-member copy flags |

The `annotations_v1` run covers all 8,072 hours: 128,705 recordings measured,
665 with a status saying why not (476 zero-frame pose arrays, 189 missing files);
381 structural checks pass, and 60 randomly re-read clips match the release.

Every column is registered once in `src/seamless_curation/schema.py` with its
level, role, dtype, unit, allowed values, **what a missing value means**,
provenance (Meta as-is / derived from Meta / reused from the previous pipeline /
adapted / fresh) and evidence type (given / parsed / measured / rule-inferred /
aggregate).

## Running it

```bash
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
PY=/simurgh/group/lw29/conda/envs/ViBES/bin/python
sc() { $PY -m seamless_curation --config configs/annotations_v1.yaml "$@"; }

sc catalog                                          # every recording + Meta metadata (seconds)
sbatch --array=0-511%160 slurm/scan.sbatch          # continuous measurements, per interaction
sbatch --array=0-511%160 slurm/scan_video.sbatch    # optional pixel pass (after scan)
sc annotate                                         # labels, scores, links; validate; publish
sc validate --read-back 60                          # re-validate + re-read sampled clips
sc report                                           # annotations/annotation_report.md
sc schema-docs                                      # regenerate docs/annotation_schema.md
```

Outputs go to `outputs.root` in the config
(`/simurgh/group/lw29/seamless_annotations/annotations_v1`, mode 0700: the tables
are participant-derived). The scan stores continuous values only; every
threshold and label is applied in `annotate`, so re-tuning (e.g. the `posture:`
block) costs one annotate run and reads no media. Smoke test on a few
interactions: `sc scan --task-index 0 --tasks 1 --interactions keys.json` then
`sc annotate --limited --tasks 1`.

## Layout

```
src/seamless_curation/
  schema.py, schema_docs.py   the column registry and the generated field reference
  ids.py, clips.py            keys for every level; the time-aligned clip grid
  catalog.py                  every recording + Meta's participants/sessions/prompts/interactions
  scan.py, scan_video.py      the measurement stages (one interaction per unit of work)
  gesture.py, smplh_*.py      SMPL-H forward kinematics, torso-frame arm/hand/head motion
  posture.py framing.py face.py audio.py speech.py moi.py video_quality.py   one module per annotation family
  expressivity.py             reference quantiles and scores
  annotate.py, validate.py    build, validate and publish the tables
  dataset.py                  load and join tables; load a clip's released arrays
  report.py                   annotation_report.md
  media_repair.py corpus.py   raster repair; deterministic hashing
configs/annotations_v1.yaml   the run; configs/validation/ holds the posture hand labels
slurm/                        scan and scan-video arrays; the M-1 census jobs
scripts/inventory_m1.py       the census the catalog reads
scripts/export_legacy_review_labels.py   carried the previous iteration's human labels forward
docs/                         pipeline, schema reference, usage
reports/                      historical: the filtering iteration and its v0 measurement rounds
tests/                        see below
```

## Tests

```bash
PYTHONPATH=src:. $PY -m pytest
```

`tests/test_pipeline_integration.py` writes a miniature release to disk and runs
it through catalog, scan, annotate and validate, asserting pairing, partner
status, conversation-level propagation, NA semantics and per-participant posture;
`tests/test_validate.py` injects one fault at a time into valid tables. Module
tests cover each annotation with synthetic inputs whose answers are known. Tests
that need the research-licensed SMPL-H model skip if it is not staged.

## History

The previous iteration built a filtered co-speech-gesture subset for ViBES
upper-body training (tier-1 gates, a tier-2 score, per-file and per-participant
caps). It is preserved at the git tag `v1-cospeech-filter`; `reports/` keeps its
calibration reports and the v0 measurement rounds as the evidence behind the
posture thresholds and measures reused here. `.git-session/` is an orphaned git
directory holding the 71 v0 commits (also on the `v0-history` branch); it is
git-ignored.

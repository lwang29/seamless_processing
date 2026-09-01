#!/bin/bash
#SBATCH --job-name=anno_timing
#SBATCH --account=simurgh
#SBATCH --partition=simurgh
#SBATCH --output=slurm/logs/anno_timing_%A_%a.out
#SBATCH --error=slurm/logs/anno_timing_%A_%a.err
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=6G

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
VIBES=/simurgh/group/lw29/conda/envs/ViBES/bin/python
export PYTHONPATH="$PWD/src:$PWD${PYTHONPATH:+:$PYTHONPATH}"

$VIBES scripts/probe_annotation_timing.py \
  --task "${SLURM_ARRAY_TASK_ID}" --tasks "${TASKS:-64}"

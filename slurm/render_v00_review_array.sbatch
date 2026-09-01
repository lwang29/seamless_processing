#!/bin/bash
#SBATCH --account=simurgh
#SBATCH --partition=simurgh
#SBATCH --job-name=si-v00-render
#SBATCH --array=0-399%32
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
#SBATCH --time=00:40:00
#SBATCH --output=slurm/logs/v00-render-%A_%a.out

set -euo pipefail

# Throttle is set from task duration, not politeness: these tasks run in tens of
# seconds, so at %8 the array would spend most of its wall clock waiting for the
# next scheduling cycle. See docs/pipeline_spec_v1.md section 6.

REVIEW_CONFIG="${REVIEW_CONFIG:-configs/v00_review.yaml}"
export PYTHONPATH="$PWD/src:$PWD${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

/simurgh/group/lw29/conda/envs/ViBES/bin/python scripts/render_review_gallery.py \
  --config "$REVIEW_CONFIG" --index "$SLURM_ARRAY_TASK_ID" --strict

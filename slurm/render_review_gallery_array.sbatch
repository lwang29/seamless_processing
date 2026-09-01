#!/bin/bash
#SBATCH --account=simurgh
#SBATCH --partition=simurgh
#SBATCH --job-name=si-review-render
#SBATCH --array=0-0
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=slurm/logs/review-render-%A_%a.out

set -euo pipefail

REVIEW_CONFIG="${REVIEW_CONFIG:-configs/session2_review.yaml}"
export PYTHONPATH="$PWD/src:$PWD${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

/simurgh/group/lw29/conda/envs/ViBES/bin/python scripts/render_review_gallery.py \
  --config "$REVIEW_CONFIG" --index "$SLURM_ARRAY_TASK_ID" --strict

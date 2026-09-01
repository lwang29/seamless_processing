#!/bin/bash
#SBATCH --account=simurgh
#SBATCH --partition=simurgh
#SBATCH --job-name=si-fm-render
#SBATCH --array=0-424%110
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
#SBATCH --time=00:40:00
#SBATCH --output=slurm/logs/fm-render-%A_%a.out

set -euo pipefail
REVIEW_CONFIG="${REVIEW_CONFIG:-configs/v00_fm_review.yaml}"
export PYTHONPATH="$PWD/src:$PWD${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

/simurgh/group/lw29/conda/envs/ViBES/bin/python scripts/render_review_gallery.py \
  --config "$REVIEW_CONFIG" --index "$SLURM_ARRAY_TASK_ID" --strict

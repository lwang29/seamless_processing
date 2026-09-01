#!/bin/bash
#SBATCH --account=simurgh
#SBATCH --partition=simurgh
#SBATCH --job-name=si-r7-render
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=slurm/logs/r7-render-%A_%a.out

set -euo pipefail
export PYTHONPATH="$PWD/src:$PWD${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

/simurgh/group/lw29/conda/envs/ViBES/bin/python scripts/render_review_gallery.py \
  --config configs/vendors_r7_review.yaml --index "$SLURM_ARRAY_TASK_ID" --strict

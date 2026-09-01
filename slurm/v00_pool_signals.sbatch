#!/usr/bin/env bash
#SBATCH --job-name=si-v00-pool
#SBATCH --account=simurgh
#SBATCH --partition=simurgh
#SBATCH --array=0-47%48
#SBATCH --cpus-per-task=2
#SBATCH --mem=12G
#SBATCH --time=01:00:00
#SBATCH --output=slurm/logs/v00-pool-%A_%a.out

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:?submit this job from the repository root}"
export PYTHONPATH="$PWD/src:$PWD${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

/simurgh/group/lw29/conda/envs/ViBES/bin/python scripts/v00_pool_signals.py \
  --config configs/v00_pool_signals.yaml \
  --task-index "${SLURM_ARRAY_TASK_ID:?missing Slurm array task index}"

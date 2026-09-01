#!/bin/bash
#SBATCH --account=simurgh
#SBATCH --partition=simurgh
#SBATCH --job-name=si-roll
#SBATCH --array=0-11
#SBATCH --cpus-per-task=3
#SBATCH --mem=12G
#SBATCH --time=00:50:00
#SBATCH --output=slurm/logs/roll-%A_%a.out
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?submit from the repository root}"
export PYTHONPATH="$PWD/src:$PWD"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-3}" OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
/simurgh/group/lw29/conda/envs/ViBES/bin/python scripts/probe_vendor_body_roll.py \
  "$SLURM_ARRAY_TASK_ID" 12

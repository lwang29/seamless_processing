#!/usr/bin/env bash
#SBATCH --job-name=si-v00-fm
#SBATCH --account=simurgh
#SBATCH --partition=simurgh
#SBATCH --array=0-63%64
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=slurm/logs/v00-fm-%A_%a.out

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?submit this job from the repository root}"
export PYTHONPATH="$PWD/src:$PWD${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-2}"
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

DETECT_CONFIG="${DETECT_CONFIG:-configs/v00_fm_detect.yaml}"
/simurgh/group/lw29/conda/envs/ViBES/bin/python scripts/v00_fm_scan.py \
  --config "$DETECT_CONFIG" \
  --task-index "${SLURM_ARRAY_TASK_ID:?missing Slurm array task index}"

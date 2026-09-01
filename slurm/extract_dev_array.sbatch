#!/usr/bin/env bash
#SBATCH --job-name=si-harness-dev
#SBATCH --account=simurgh
#SBATCH --partition=simurgh
#SBATCH --array=0-119%16
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=slurm/logs/extract-%A_%a.out

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:?submit this job from the repository root}"
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

VIBES_PYTHON=/simurgh/group/lw29/conda/envs/ViBES/bin/python
"$VIBES_PYTHON" -m seamless_curation \
  --config configs/harness.yaml \
  --index "${SLURM_ARRAY_TASK_ID:?missing Slurm array index}"

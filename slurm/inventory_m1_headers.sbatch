#!/usr/bin/env bash
#SBATCH --job-name=si-m1-headers
#SBATCH --account=simurgh
#SBATCH --partition=simurgh
#SBATCH --array=0-511%8
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=01:00:00
#SBATCH --output=slurm/logs/m1-headers-%A_%a.out

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:?submit this job from the repository root}"
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

VIBES_PYTHON=/simurgh/group/lw29/conda/envs/ViBES/bin/python
"$VIBES_PYTHON" scripts/inventory_m1.py \
  --config configs/inventory_m1.yaml \
  headers \
  --task-index "${SLURM_ARRAY_TASK_ID:?missing Slurm array task index}" \
  --workers "${SLURM_CPUS_PER_TASK:?missing CPU allocation}"

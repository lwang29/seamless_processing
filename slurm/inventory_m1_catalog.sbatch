#!/usr/bin/env bash
#SBATCH --job-name=si-m1-catalog
#SBATCH --account=simurgh
#SBATCH --partition=simurgh
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=01:00:00
#SBATCH --output=slurm/logs/m1-catalog-%j.out

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:?submit this job from the repository root}"
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

VIBES_PYTHON=/simurgh/group/lw29/conda/envs/ViBES/bin/python
"$VIBES_PYTHON" scripts/inventory_m1.py \
  --config configs/inventory_m1.yaml \
  catalog


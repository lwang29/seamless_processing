#!/usr/bin/env bash
#SBATCH --job-name=si-m1-benchmark
#SBATCH --account=simurgh
#SBATCH --partition=simurgh
#SBATCH --cpus-per-task=16
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=slurm/logs/m1-benchmark-%j.out

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:?submit this job from the repository root}"
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

VIBES_PYTHON=/simurgh/group/lw29/conda/envs/ViBES/bin/python
"$VIBES_PYTHON" scripts/inventory_m1.py \
  --config configs/inventory_m1.yaml \
  benchmark \
  --worker-counts 1 4 8 16 \
  --samples-per-setting 256


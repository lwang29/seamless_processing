#!/usr/bin/env bash
#SBATCH --job-name=si-m3-features
#SBATCH --account=simurgh
#SBATCH --partition=simurgh
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --output=slurm/logs/m3-features-%j.out

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:?submit this job from the repository root}"
export PYTHONPATH="$PWD/src:$PWD${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

/usr/bin/time -v -o outputs/session2/m3_dev_features/resource_usage.txt \
  /simurgh/group/lw29/conda/envs/ViBES/bin/python scripts/m3_dev_features.py \
    --config configs/m3_dev_features.yaml

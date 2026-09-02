#!/usr/bin/env bash
# Stage 2: window-resolution gesture measurement over the eligible population.
#
#   sbatch --array=0-511%64 slurm/scan.sbatch
#
# One task per shard; shards are assigned by a hash of file_id, so a task always
# gets the same files and a completed shard is skipped on resubmission. Override
# the config with SC_CONFIG=... and the shard count with SC_TASKS=...; SC_TASKS
# must equal the array size or shards will be missing at gather time.
#SBATCH --job-name=si-scan
#SBATCH --account=simurgh
#SBATCH --partition=simurgh
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=02:00:00
#SBATCH --output=slurm/logs/scan-%A_%a.out

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?submit this job from the repository root}"
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1

/simurgh/group/lw29/conda/envs/ViBES/bin/python -m seamless_curation \
  --config "${SC_CONFIG:-configs/vibes_upper_body.yaml}" scan \
  --task-index "${SLURM_ARRAY_TASK_ID:?submit with --array}" \
  --tasks "${SC_TASKS:-512}"

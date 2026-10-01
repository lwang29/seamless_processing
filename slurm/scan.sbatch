#!/usr/bin/env bash
# Stage 2: continuous measurements for every recording, one shard of interactions per task.
#
#   sbatch --array=0-511%160 slurm/scan.sbatch
#
# Shards are assigned by a hash of interaction_key (both members of a conversation
# are measured in the same task), so a task always gets the same interactions and
# a current shard is skipped on resubmission. SC_TASKS must equal the array size
# and the config's scan.tasks, or `annotate` will refuse the shards.
#SBATCH --job-name=si-annotate-scan
#SBATCH --account=simurgh
#SBATCH --partition=simurgh
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --output=slurm/logs/scan-%A_%a.out

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?submit this job from the repository root}"
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1

/simurgh/group/lw29/conda/envs/ViBES/bin/python -m seamless_curation \
  --config "${SC_CONFIG:-configs/annotations_v1.yaml}" scan \
  --task-index "${SLURM_ARRAY_TASK_ID:?submit with --array}" \
  --tasks "${SC_TASKS:-512}"

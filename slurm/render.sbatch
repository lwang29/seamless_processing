#!/usr/bin/env bash
# Stage 4: review artefacts (card PNG, and by default the 30 s clip with audio).
#
#   sbatch --array=0-255%80 slurm/render.sbatch                 # cards + clips
#   SC_CARD_ONLY=1 sbatch --array=0-255%80 slurm/render.sbatch  # cards only
#
# Tasks stride the review manifest, so the array size and SC_TASKS must match.
# Rendering is idempotent by fingerprint: resubmitting re-renders nothing.
#SBATCH --job-name=si-render
#SBATCH --account=simurgh
#SBATCH --partition=simurgh
#SBATCH --cpus-per-task=2
#SBATCH --mem=12G
#SBATCH --time=04:00:00
#SBATCH --output=slurm/logs/render-%A_%a.out

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?submit this job from the repository root}"
export PYTHONPATH="$PWD/src:$PWD${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-2}" OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1

/simurgh/group/lw29/conda/envs/ViBES/bin/python -m seamless_curation \
  --config "${SC_CONFIG:-configs/vibes_upper_body.yaml}" render \
  --task-index "${SLURM_ARRAY_TASK_ID:?submit with --array}" \
  --tasks "${SC_RENDER_TASKS:-256}" \
  ${SC_LIMIT:+--limit "$SC_LIMIT"} \
  ${SC_CARD_ONLY:+--card-only}

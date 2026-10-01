#!/usr/bin/env bash
# Optional pixel pass: one decoded keyframe per clip (sharpness, exposure, clutter,
# camera shift). Runs after `scan`, over the same shards.
#
#   sbatch --array=0-511%160 slurm/scan_video.sbatch
#SBATCH --job-name=si-annotate-video
#SBATCH --account=simurgh
#SBATCH --partition=simurgh
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=03:00:00
#SBATCH --output=slurm/logs/video-%A_%a.out

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?submit this job from the repository root}"
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1

/simurgh/group/lw29/conda/envs/ViBES/bin/python -m seamless_curation \
  --config "${SC_CONFIG:-configs/annotations_v1.yaml}" scan-video \
  --task-index "${SLURM_ARRAY_TASK_ID:?submit with --array}"

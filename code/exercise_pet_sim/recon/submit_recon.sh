#!/bin/bash
# Stage 2: reconstruct 56 cm 290 deg vs 70 cm closed, P50 and P90, then evaluate.
#   bash submit_recon.sh              results go to /scratch/$USER/exercise_pet_recon
#   bash submit_recon.sh /path/out    custom results folder
# GATE trues are read from /scratch/$USER/exercise_pet_sweep2 (override with GATE=...).

set -eo pipefail
cd "$(dirname "$0")"
source ~/gate10_env.sh
python -c "import numba, scipy" 2>/dev/null || pip install --quiet numba scipy

OUT=${1:-/scratch/$USER/exercise_pet_recon}
GATE=${GATE:-/scratch/$USER/exercise_pet_sweep2}
mkdir -p "$OUT/logs"
[ -f ../data/phantom_P90.npz ] || (cd .. && python make_phantom.py)

LAST=$(python -c "import sys; sys.path.insert(0, '..'); import config as C; print(len(C.RECON_CONFIGS) * len(C.RECON_BODIES) - 1)")

REC=$(sbatch --parsable --array=0-"$LAST" --export=ALL,OUT="$OUT",GATE="$GATE" \
      --output="$OUT/logs/recon_%A_%a.log" slurm_recon.sh)
EVA=$(sbatch --parsable --dependency=afterany:"$REC" --export=ALL,OUT="$OUT" \
      --mail-type=END,FAIL --mail-user="$USER@virginia.edu" \
      --output="$OUT/logs/evaluate_%j.log" slurm_evaluate.sh)

echo "Submitted recon array $REC (tasks 0-$LAST) and evaluation job $EVA (email when done)."
echo "Results: $OUT/evaluation   Logs: $OUT/logs"

#!/bin/bash
# Submit sweep 2 (bore size x body size, arms in slots) plus its analysis job.
#   bash submit_all2.sh                 results go to /scratch/$USER/exercise_pet_sweep2
#   bash submit_all2.sh /path/to/out    custom results folder

set -eo pipefail
cd "$(dirname "$0")"
source ~/gate10_env.sh

OUT=${1:-/scratch/$USER/exercise_pet_sweep2}
mkdir -p "$OUT/logs"
[ -f data/phantom_P90.npz ] || python make_phantom.py

LAST=$(python -c "import config as C; print(len(C.SWEEP2_CONFIGS) * len(C.SWEEP2_BODIES) * C.JOBS_PER_CONFIG - 1)")

SIM=$(sbatch --parsable --array=0-"$LAST" --export=ALL,OUT="$OUT" \
      --output="$OUT/logs/sim_%A_%a.log" slurm_sim2.sh)
ANA=$(sbatch --parsable --dependency=afterany:"$SIM" --export=ALL,OUT="$OUT" \
      --mail-type=END,FAIL --mail-user="$USER@virginia.edu" \
      --output="$OUT/logs/analyze_%j.log" slurm_analyze2.sh)

echo "Submitted sweep-2 array $SIM (tasks 0-$LAST) and analysis job $ANA (email when done)."
echo "Results: $OUT/analysis   Logs: $OUT/logs"

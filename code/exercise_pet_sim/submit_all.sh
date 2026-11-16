#!/bin/bash
# Submit the full geometry sweep plus the analysis job.
#   bash submit_all.sh                 results go to /scratch/$USER/exercise_pet
#   bash submit_all.sh /path/to/out    custom results folder

set -eo pipefail
cd "$(dirname "$0")"
source ~/gate10_env.sh

OUT=${1:-/scratch/$USER/exercise_pet}
mkdir -p "$OUT/logs"
[ -f data/phantom_activity.mhd ] || python make_phantom.py

LAST=$(python -c "import config as C; print(len(C.GAP_SWEEP) * C.JOBS_PER_GAP - 1)")

SIM=$(sbatch --parsable --array=0-"$LAST" --export=ALL,OUT="$OUT" \
      --output="$OUT/logs/sim_%A_%a.log" slurm_sim.sh)
ANA=$(sbatch --parsable --dependency=afterany:"$SIM" --export=ALL,OUT="$OUT" \
      --output="$OUT/logs/analyze_%j.log" slurm_analyze.sh)

echo "Submitted simulation array $SIM (tasks 0-$LAST) and analysis job $ANA."
echo "Results: $OUT/analysis   Logs: $OUT/logs"
echo "Check progress: squeue -u $USER"

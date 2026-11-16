#!/bin/bash
#SBATCH --job-name=xpet_rob2
#SBATCH --account=berr_medical_imaging
#SBATCH --partition=standard
#SBATCH --cpus-per-task=32
#SBATCH --mem=96G
#SBATCH --time=03:00:00
# Stage 3: one array task = one (geometry, heart pose); per-pose sensitivity, pose-matched correction (50 M trues in 25 parallel 2 M chunks + defect data, 26 processes on 32 cores).
# Optional SEED=N: independent replicate (new defect data and new pose-matched correction) into robust_v3_sN.
# With MODE=collect (no array) it builds the figures and tables from all task outputs.

set -eo pipefail
source ~/gate10_env.sh
export NUMBA_NUM_THREADS=$SLURM_CPUS_PER_TASK OMP_NUM_THREADS=1
cd "$SLURM_SUBMIT_DIR"
if [ "$MODE" = "collect" ]; then
    python robust2.py --recon "$OUT" --collect ${SEED:+--seed $SEED --outdir robust_v3_s$SEED --tag robust3_s$SEED}
    if [ -n "$SEED" ]; then      # rename task files so "xpet pull" does not overwrite the stage 3c task files
        cd "$OUT/robust_v3_s$SEED" && for f in task_*; do mv "$f" "s${SEED}_$f"; done
    fi
else
    echo "task $SLURM_ARRAY_TASK_ID on $(hostname), $SLURM_CPUS_PER_TASK cores"
    python robust2.py --recon "$OUT" --task "$SLURM_ARRAY_TASK_ID" ${SEED:+--seed $SEED --outdir robust_v3_s$SEED --tag robust3_s$SEED}
fi

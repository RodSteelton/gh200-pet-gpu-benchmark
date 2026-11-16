#!/bin/bash
#SBATCH --job-name=xpet_back
#SBATCH --account=berr_medical_imaging
#SBATCH --partition=standard
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=02:00:00
# Stage 4: one array task = one (geometry, backrest material); simulation and sensitivity in 30 parallel processes.
# With MODE=collect (no array) it builds the figures and tables from all task outputs.

set -eo pipefail
source ~/gate10_env.sh
export NUMBA_NUM_THREADS=$SLURM_CPUS_PER_TASK OMP_NUM_THREADS=1
cd "$SLURM_SUBMIT_DIR"
if [ "$MODE" = "collect" ]; then
    python backrest.py --recon "$OUT" --collect
else
    echo "task $SLURM_ARRAY_TASK_ID on $(hostname), $SLURM_CPUS_PER_TASK cores"
    python backrest.py --recon "$OUT" --task "$SLURM_ARRAY_TASK_ID"
fi

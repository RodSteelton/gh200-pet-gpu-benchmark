#!/bin/bash
#SBATCH --job-name=xpet_mbfp
#SBATCH --account=berr_medical_imaging
#SBATCH --partition=standard
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=10:00:00
# Stage 5d, MBF with randoms for 4 panels 30 x 24 vs the open ring. MODE: task (array) or collect.
set -eo pipefail
source ~/xpet_env.sh
export NUMBA_NUM_THREADS=$SLURM_CPUS_PER_TASK OMP_NUM_THREADS=1
cd "$SLURM_SUBMIT_DIR"
if [ "$MODE" = "collect" ]; then
    python mbf_panels.py --recon "$OUT" --collect
else
    echo "task $SLURM_ARRAY_TASK_ID on $(hostname)"; python mbf_panels.py --recon "$OUT" --task "$SLURM_ARRAY_TASK_ID"
fi

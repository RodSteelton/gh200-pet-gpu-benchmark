#!/bin/bash
#SBATCH --job-name=xpet_cr
#SBATCH --account=berr_medical_imaging
#SBATCH --partition=standard
#SBATCH --cpus-per-task=32
#SBATCH --mem=96G
#SBATCH --time=03:00:00
# Stage 7: one array task = one geometry; per-decay singles per block and trues per block pair for every
# activity class (2e8 decays per class, 30 processes). MODE=collect builds the count-rate tables and figure.
set -eo pipefail
source ~/gate10_env.sh
export NUMBA_NUM_THREADS=1 OMP_NUM_THREADS=1
cd "$SLURM_SUBMIT_DIR"
if [ "$MODE" = "collect" ]; then
    python countrate.py --recon "$OUT" --collect
else
    echo "task $SLURM_ARRAY_TASK_ID on $(hostname), $SLURM_CPUS_PER_TASK cores"
    python countrate.py --recon "$OUT" --task "$SLURM_ARRAY_TASK_ID"
fi

#!/bin/bash
#SBATCH --job-name=xpet_mnec
#SBATCH --account=berr_medical_imaging
#SBATCH --partition=standard
#SBATCH --cpus-per-task=32
#SBATCH --mem=96G
#SBATCH --time=08:00:00
# Stage 5c: one array task = one simulated 15O-water scan (geometry x rest/stress x scan), reconstructed for
# 4 dose/shielding scenarios after NEC-equivalent thinning. MODE=collect builds the summary and figure.
set -eo pipefail
source ~/gate10_env.sh
export NUMBA_NUM_THREADS=$SLURM_CPUS_PER_TASK OMP_NUM_THREADS=1
cd "$SLURM_SUBMIT_DIR"
if [ "$MODE" = "collect" ]; then
    python mbf_nec.py --recon "$OUT" --collect
else
    echo "task $SLURM_ARRAY_TASK_ID on $(hostname), $SLURM_CPUS_PER_TASK cores"
    python mbf_nec.py --recon "$OUT" --task "$SLURM_ARRAY_TASK_ID" ${NREAL:+--n-real $NREAL --real-from $RFROM}
fi

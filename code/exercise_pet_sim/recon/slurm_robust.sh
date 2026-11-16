#!/bin/bash
#SBATCH --job-name=xpet_robust
#SBATCH --account=berr_medical_imaging
#SBATCH --partition=standard
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=05:00:00
# Stage 3: heart-position robustness with a 20% anterior defect (fast MC).

set -eo pipefail
source ~/gate10_env.sh
export NUMBA_NUM_THREADS=$SLURM_CPUS_PER_TASK OMP_NUM_THREADS=1
cd "$SLURM_SUBMIT_DIR"
python robust.py --recon "$OUT"

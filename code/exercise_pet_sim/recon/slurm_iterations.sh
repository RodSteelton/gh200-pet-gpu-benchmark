#!/bin/bash
#SBATCH --job-name=xpet_iter
#SBATCH --account=berr_medical_imaging
#SBATCH --partition=standard
#SBATCH --cpus-per-task=32
#SBATCH --mem=48G
#SBATCH --time=03:00:00
# Stage 2b: OSEM iterations 4/8/16 and phantom-derived segment correction (fast-MC data).

set -eo pipefail
source ~/gate10_env.sh
export NUMBA_NUM_THREADS=$SLURM_CPUS_PER_TASK OMP_NUM_THREADS=1
cd "$SLURM_SUBMIT_DIR"
python iterations.py --recon "$OUT"

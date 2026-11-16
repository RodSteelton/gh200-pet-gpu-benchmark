#!/bin/bash
#SBATCH --job-name=xpet_mot9
#SBATCH --account=berr_medical_imaging
#SBATCH --partition=standard
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=12:00:00
# Stage 9, exercise motion with panels. MODE: prep (array = geometry), task (array = scan), collect.
set -eo pipefail
source ~/xpet_env.sh
export NUMBA_NUM_THREADS=$SLURM_CPUS_PER_TASK OMP_NUM_THREADS=1
cd "$SLURM_SUBMIT_DIR"
case "$MODE" in
    prep)    echo "prep $SLURM_ARRAY_TASK_ID on $(hostname)"; python motion_panels.py --recon "$OUT" --prep "$SLURM_ARRAY_TASK_ID" ;;
    collect) python motion_panels.py --recon "$OUT" --collect ;;
    *)       echo "task $SLURM_ARRAY_TASK_ID on $(hostname)"; python motion_panels.py --recon "$OUT" --task "$SLURM_ARRAY_TASK_ID" ;;
esac

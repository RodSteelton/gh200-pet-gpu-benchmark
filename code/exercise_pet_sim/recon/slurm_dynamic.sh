#!/bin/bash
#SBATCH --job-name=xpet_dyn
#SBATCH --account=berr_medical_imaging
#SBATCH --partition=standard
#SBATCH --cpus-per-task=32
#SBATCH --mem=96G
#SBATCH --time=04:00:00
# Stage 5: one array task = one simulated 15O-water scan (geometry x rest/stress x noise realisation);
# emission simulation in 30 parallel processes, then 23-frame TOF OSEM and kinetic fitting.
# Optional CTR=320: rerun at that timing resolution into dynamic_320.
# With MODE=collect (no array) it builds the figures and tables from all task outputs.

set -eo pipefail
source ~/gate10_env.sh
export NUMBA_NUM_THREADS=$SLURM_CPUS_PER_TASK OMP_NUM_THREADS=1
cd "$SLURM_SUBMIT_DIR"
if [ "$MODE" = "collect" ]; then
    python dynamic.py --recon "$OUT" --collect ${CTR:+--ctr $CTR --outdir dynamic_$CTR}
else
    echo "task $SLURM_ARRAY_TASK_ID on $(hostname), $SLURM_CPUS_PER_TASK cores"
    python dynamic.py --recon "$OUT" --task "$SLURM_ARRAY_TASK_ID" ${CTR:+--ctr $CTR --outdir dynamic_$CTR}
fi

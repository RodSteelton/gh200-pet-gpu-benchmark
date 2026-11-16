#!/bin/bash
#SBATCH --job-name=xpet_panels
#SBATCH --account=berr_medical_imaging
#SBATCH --partition=standard
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=12:00:00
# Stages 8 and 8b, flat panels. MODE:
#   task        one (geometry, heart pose) image-quality task (array index), stage $STAGE (default 8b)
#   collect     figures and tables from all image-quality tasks
#   cr_task     stage 8b count-rate tables for one geometry (array index)
#   cr_collect  NEC table and nec_factors.json from all count-rate tables
#   smoke       count rates for 3 geometries at 2e5 decays per class, then 2 image tasks at 1% counts

set -eo pipefail
source ~/xpet_env.sh   # python 3.13 venv for recon; python/3.11.4 was retired on Rivanna in Oct 2026
export NUMBA_NUM_THREADS=$SLURM_CPUS_PER_TASK OMP_NUM_THREADS=1
cd "$SLURM_SUBMIT_DIR"
ST=${STAGE:-8b}
case "$MODE" in
    collect)    python panels.py --stage "$ST" --recon "$OUT" --collect ;;
    cr_task)    echo "count-rate task $SLURM_ARRAY_TASK_ID on $(hostname)"
                export NUMBA_NUM_THREADS=1
                python panels_cr.py --recon "$OUT" --task "$SLURM_ARRAY_TASK_ID" ;;
    cr_collect) python panels_cr.py --recon "$OUT" --collect ;;
    smoke)      (export NUMBA_NUM_THREADS=1; python panels_cr.py --recon "$OUT" --smoke)
                python panels.py --stage "$ST" --recon "$OUT" --smoke ;;
    *)          echo "task $SLURM_ARRAY_TASK_ID on $(hostname), $SLURM_CPUS_PER_TASK cores"
                python panels.py --stage "$ST" --recon "$OUT" --task "$SLURM_ARRAY_TASK_ID" ;;
esac

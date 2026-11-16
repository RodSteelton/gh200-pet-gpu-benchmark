#!/bin/bash
#SBATCH --job-name=xpet_sim
#SBATCH --account=berr_medical_imaging
#SBATCH --partition=standard
#SBATCH --cpus-per-task=1
#SBATCH --mem=6G
#SBATCH --time=06:00:00
# One array task = one chunk of decays for one ring configuration.
# Submitted by submit_all.sh, which sets OUT and the log paths.

set -eo pipefail
source ~/gate10_env.sh
cd "$SLURM_SUBMIT_DIR"

read GAP SEED < <(python -c "
import os, config as C
i = int(os.environ['SLURM_ARRAY_TASK_ID']); j = C.JOBS_PER_GAP
print(C.GAP_SWEEP[i // j], 1000 + 100 * (i // j) + i % j)")

echo "task $SLURM_ARRAY_TASK_ID: gap $GAP, seed $SEED, host $(hostname)"
python run_sim.py --gap "$GAP" --seed "$SEED" --out "$OUT/gap$(printf %02d "$GAP")"

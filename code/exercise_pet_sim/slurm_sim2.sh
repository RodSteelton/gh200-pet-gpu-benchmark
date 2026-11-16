#!/bin/bash
#SBATCH --job-name=xpet_sim2
#SBATCH --account=berr_medical_imaging
#SBATCH --partition=standard
#SBATCH --cpus-per-task=1
#SBATCH --mem=6G
#SBATCH --time=06:00:00
# Sweep 2: one array task = one chunk of decays for one (geometry, body) pair.
# Submitted by submit_all2.sh, which sets OUT and the log paths.

set -eo pipefail
source ~/gate10_env.sh
cd "$SLURM_SUBMIT_DIR"

read CFG BODY SEED < <(python -c "
import os, config as C
i = int(os.environ['SLURM_ARRAY_TASK_ID']); j = C.JOBS_PER_CONFIG
combos = [(c[0], b) for c in C.SWEEP2_CONFIGS for b in C.SWEEP2_BODIES]
k = i // j
print(combos[k][0], combos[k][1], 5000 + 100 * k + i % j)")

echo "task $SLURM_ARRAY_TASK_ID: $CFG $BODY seed $SEED host $(hostname)"
python run_sim.py --config "$CFG" --body "$BODY" --seed "$SEED" --out "$OUT/${CFG}_${BODY}"

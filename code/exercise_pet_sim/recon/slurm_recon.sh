#!/bin/bash
#SBATCH --job-name=xpet_recon
#SBATCH --account=berr_medical_imaging
#SBATCH --partition=standard
#SBATCH --cpus-per-task=32
#SBATCH --mem=48G
#SBATCH --time=04:00:00
# One array task = one (geometry, body): fast-MC trues, then GATE trues if sweep 2 is present.
# Submitted by submit_recon.sh, which sets OUT, GATE and the log paths.

set -eo pipefail
source ~/gate10_env.sh
export NUMBA_NUM_THREADS=$SLURM_CPUS_PER_TASK OMP_NUM_THREADS=1
cd "$SLURM_SUBMIT_DIR"

read CFG BODY < <(python -c "
import os, sys; sys.path.insert(0, '..'); import config as C
combos = [(c, b) for c in C.RECON_CONFIGS for b in C.RECON_BODIES]
print(*combos[int(os.environ['SLURM_ARRAY_TASK_ID'])])")

echo "task $SLURM_ARRAY_TASK_ID: $CFG $BODY on $(hostname), $SLURM_CPUS_PER_TASK cores"
python run_recon.py --config "$CFG" --body "$BODY" --data fastmc --out "$OUT"
if [ -d "$GATE/${CFG}_${BODY}" ]; then
    python run_recon.py --config "$CFG" --body "$BODY" --data gate --gate-dir "$GATE" --out "$OUT"
else
    echo "no GATE data at $GATE/${CFG}_${BODY}; skipped"
fi

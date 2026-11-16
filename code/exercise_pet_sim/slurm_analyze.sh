#!/bin/bash
#SBATCH --job-name=xpet_analyze
#SBATCH --account=berr_medical_imaging
#SBATCH --partition=standard
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=01:00:00
# Runs after all simulation tasks finish (submitted by submit_all.sh).

set -eo pipefail
source ~/gate10_env.sh
cd "$SLURM_SUBMIT_DIR"
python analyze.py --results "$OUT"

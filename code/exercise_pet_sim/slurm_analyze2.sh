#!/bin/bash
#SBATCH --job-name=xpet_analyze2
#SBATCH --account=berr_medical_imaging
#SBATCH --partition=standard
#SBATCH --cpus-per-task=1
#SBATCH --mem=24G
#SBATCH --time=01:30:00

set -eo pipefail
source ~/gate10_env.sh
cd "$SLURM_SUBMIT_DIR"
python analyze2.py --results "$OUT"

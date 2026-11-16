#!/bin/bash
#SBATCH --job-name=xpet_eval
#SBATCH --account=berr_medical_imaging
#SBATCH --partition=standard
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=00:45:00

set -eo pipefail
source ~/gate10_env.sh
cd "$SLURM_SUBMIT_DIR"
python evaluate.py --recon "$OUT"

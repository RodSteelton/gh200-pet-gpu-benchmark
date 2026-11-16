#!/bin/bash
#SBATCH -p gpu-gh200-public
#SBATCH --gres=gpu:1
#SBATCH -c 72
#SBATCH --mem=300G
#SBATCH -t 14:00:00
#SBATCH -J petCDE_grace
#SBATCH -o /sfs/ceph/hachi/standard/ivy-hip-monte/runCDE_%j.out
# Runs D, E, C of OLD-SYSTEM-RESULTS.md as CPU code on the Grace CPU (the GPU is idle). Order: D, E, C.
R=/sfs/ceph/hachi/standard/ivy-hip-monte
export HOME=$R/home PATH=$R/envs/pet/bin:/usr/bin:/bin XPET_WORKERS=30 OMP_NUM_THREADS=1
cd $R/exercise_pet_sim/recon
OUT=$R/recon_CDE_grace; mkdir -p $OUT
date; hostname; lscpu | grep -E "Model name|^CPU\(s\)"; free -g | head -2; python -c "import numpy,numba,scipy;print('numpy',numpy.__version__,'numba',numba.__version__,'scipy',scipy.__version__)"
# input for D: sensitivity image (made by the GPU code, CPU-run inputs are not timed)
cp -n $R/recon_F_seeds/sens_gpu_R280_slot_P90_backrest_carbon.npy $OUT/sens_R280_slot_P90_backrest_carbon_v3.npy || echo "NO SENS FILE FOR D"
T0=$(date +%s)
export NUMBA_NUM_THREADS=32
for TASK in 0 5; do
  echo "=== D dynamic task $TASK $(date +%T)"; S=$(date +%s)
  python dynamic.py --recon $OUT --task $TASK || echo "FAILED D task $TASK"
  echo "D task $TASK seconds $(( $(date +%s) - S ))"
done
echo "=== E prep 0 $(date +%T)"; S=$(date +%s)
python motion_panels.py --recon $OUT --prep 0 || echo "FAILED E prep"
echo "E prep seconds $(( $(date +%s) - S ))"
for TASK in 0 5; do
  echo "=== E motion task $TASK $(date +%T)"; S=$(date +%s)
  python motion_panels.py --recon $OUT --task $TASK || echo "FAILED E task $TASK"
  echo "E task $TASK seconds $(( $(date +%s) - S ))"
done
export NUMBA_NUM_THREADS=1
N=$(python panels_cr.py --recon $OUT --ntasks | tail -1); echo "C: $N count-rate tasks"
for TASK in $(seq 0 $((N-1))); do
  echo "=== C count-rate task $TASK $(date +%T)"; S=$(date +%s)
  python panels_cr.py --recon $OUT --task $TASK || echo "FAILED C task $TASK"
  echo "C task $TASK seconds $(( $(date +%s) - S ))"
  find $OUT -type f -size -20M | tar czf $R/runCDE_files_$SLURM_JOB_ID.tgz -T - 2>/dev/null
done
python panels_cr.py --recon $OUT --collect || echo "FAILED C collect"
find $OUT -type f -size -20M | tar czf $R/runCDE_files_$SLURM_JOB_ID.tgz -T - 2>/dev/null
echo "total seconds $(( $(date +%s) - T0 ))"; date; echo "ALL DONE"

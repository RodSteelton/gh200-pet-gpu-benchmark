#!/bin/bash
#SBATCH -p gpu-gh200-public
#SBATCH --gres=gpu:1
#SBATCH -c 72
#SBATCH --mem=300G
#SBATCH -t 03:00:00
#SBATCH -J petD_orig
#SBATCH -o /sfs/ceph/hachi/standard/ivy-hip-monte/runDorig_%j.out
# Run D again on the Grace CPU with the ORIGINAL (old Rivanna) sensitivity image instead of the GPU-made one.
R=/sfs/ceph/hachi/standard/ivy-hip-monte
export HOME=$R/home PATH=$R/envs/pet/bin:/usr/bin:/bin XPET_WORKERS=30 OMP_NUM_THREADS=1 NUMBA_NUM_THREADS=32
cd $R/exercise_pet_sim/recon
OUT=$R/recon_D_orig; mkdir -p $OUT
date; hostname; lscpu | grep -E "Model name|^CPU\(s\)"
cp $R/orig_inputs/sens_R280_slot_P90_backrest_carbon_v3.npy $OUT/ || echo "NO ORIGINAL SENS FILE"
ls -l $OUT; md5sum $OUT/sens_R280_slot_P90_backrest_carbon_v3.npy $R/recon_CDE_grace/sens_R280_slot_P90_backrest_carbon_v3.npy
T0=$(date +%s)
for TASK in 0 5; do
  echo "=== D dynamic task $TASK $(date +%T)"; S=$(date +%s)
  python dynamic.py --recon $OUT --task $TASK || echo "FAILED D task $TASK"
  echo "D task $TASK seconds $(( $(date +%s) - S ))"
done
find $OUT -type f -size -20M | tar czf $R/runDorig_files_$SLURM_JOB_ID.tgz -T - 2>/dev/null
echo "total seconds $(( $(date +%s) - T0 ))"; date; echo "ALL DONE"

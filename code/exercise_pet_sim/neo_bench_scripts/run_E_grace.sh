#!/bin/bash
#SBATCH -p gpu-gh200-public
#SBATCH --gres=gpu:1
#SBATCH -c 72
#SBATCH --mem=300G
#SBATCH -t 02:00:00
#SBATCH -J petE_grace
#SBATCH -o /sfs/ceph/hachi/standard/ivy-hip-monte/runE_%j.out
# Run E (stage 9 motion, 56 cm ring) tasks on the Grace CPU; prep 0 was done by job 641162, which failed here for lack of the stage-2 events file.
R=/sfs/ceph/hachi/standard/ivy-hip-monte
export HOME=$R/home PATH=$R/envs/pet/bin:/usr/bin:/bin XPET_WORKERS=30 OMP_NUM_THREADS=1 NUMBA_NUM_THREADS=32
cd $R/exercise_pet_sim/recon
OUT=$R/recon_CDE_grace
date; hostname
cp -n $R/recon/events_fastmc_R280_slot_P90_10000000.npz $OUT/ || echo "NO EVENTS FILE"
ls $OUT
for TASK in 0 5; do
  echo "=== E motion task $TASK $(date +%T)"; S=$(date +%s)
  python motion_panels.py --recon $OUT --task $TASK || echo "FAILED E task $TASK"
  echo "E task $TASK seconds $(( $(date +%s) - S ))"
done
find $OUT/motion9 -type f -size -20M | tar czf $R/runE_files_$SLURM_JOB_ID.tgz -T - 2>/dev/null
date; echo "ALL DONE"

#!/bin/bash
#SBATCH -p gpu-gh200-public
#SBATCH --gres=gpu:1
#SBATCH -c 72
#SBATCH --mem=250G
#SBATCH -t 07:00:00
#SBATCH -J petFS_seeds
#SBATCH -o /sfs/ceph/hachi/standard/ivy-hip-monte/runFS_%j.out
# Run F (end-to-end 15O-water scan on the GPU) repeated: 2 rings x rest/stress x 20 scans at 200 ps, 10 scans at 320 ps.
R=/sfs/ceph/hachi/standard/ivy-hip-monte
export HOME=$R/home PATH=$R/envs/pet/bin:/usr/bin:/bin NUMBA_NUM_THREADS=72 XPET_WORKERS=30
cd $R/exercise_pet_sim/recon
DF=$R/recon_F_seeds; mkdir -p $DF
date; hostname; free -g | head -2
nvidia-smi --query-gpu=utilization.gpu,memory.used,power.draw --format=csv -l 10 > $R/gpu_util_FS_$SLURM_JOB_ID.csv &
for REAL in $(seq 0 19); do
  for CFG in R280_slot R350_closed; do
    for CTR in 200 320; do
      [ $CTR -eq 320 ] && [ $REAL -ge 10 ] && continue
      echo "=== real $REAL $CFG $CTR ps $(date +%T)"
      python dynamic_gpu.py --recon $DF --config $CFG --state both --real $REAL --ctr $CTR --outdir dynamic_gpu_c$CTR || echo "FAILED real $REAL $CFG $CTR"
    done
  done
  (cd $DF && tar czf $R/runFS_json_$SLURM_JOB_ID.tgz dynamic_gpu_c*/*.json)
done
date; echo "ALL DONE"

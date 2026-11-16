#!/bin/bash
#SBATCH -p gpu-gh200-public
#SBATCH --gres=gpu:1
#SBATCH -c 72
#SBATCH --mem=450G
#SBATCH -t 02:00:00
#SBATCH -J petG5_mem
#SBATCH -o /sfs/ceph/hachi/standard/ivy-hip-monte/runG5_%j.out
R=/sfs/ceph/hachi/standard/ivy-hip-monte
export HOME=$R/home PATH=$R/envs/pet/bin:/usr/bin:/bin NUMBA_NUM_THREADS=72
cd $R/exercise_pet_sim/recon
DIR=$R/recon
A="--recon $DIR --config R280_slot --body P90 --ctrs 200 --device gpu --repeats 1 --fast"
date; hostname; free -g | head -2
nvidia-smi --query-gpu=name,memory.total,utilization.gpu,memory.used,power.draw --format=csv -l 5 > $R/gpu_util_G5_$SLURM_JOB_ID.csv &
for K in 400 480 520; do
  echo "=== fast, replicate $K ($((K*10)) M events) $(date +%T)"; python bench_gpu.py $A --replicate $K || echo "FAILED replicate $K"
  free -g | head -2
done
date

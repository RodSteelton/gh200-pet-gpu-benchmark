#!/bin/bash
#SBATCH -p gpu-gh200-public
#SBATCH --gres=gpu:1
#SBATCH -c 72
#SBATCH --mem=450G
#SBATCH -t 03:00:00
#SBATCH -J petG2_fine
#SBATCH -o /sfs/ceph/hachi/standard/ivy-hip-monte/runG2_%j.out
R=/sfs/ceph/hachi/standard/ivy-hip-monte
export HOME=$R/home PATH=$R/envs/pet/bin:/usr/bin:/bin NUMBA_NUM_THREADS=72
cd $R/exercise_pet_sim/recon
DIR=$R/recon
A="--recon $DIR --config R280_slot --body P90 --ctrs 200"
date; hostname; free -g | head -2
nvidia-smi --query-gpu=utilization.gpu,memory.used,power.draw --format=csv -l 5 > $R/gpu_util_G2_$SLURM_JOB_ID.csv &
for K in 10 30 100 300; do
  echo "=== GPU 1.5 mm replicate $K ($(date +%T))"; python bench_gpu.py $A --device gpu --voxel 1.5 --replicate $K --repeats 1 || echo "FAILED gpu 1.5mm $K"
done
echo "=== CPU 1.5 mm replicate 10 ($(date +%T))"; python bench_gpu.py $A --device cpu --voxel 1.5 --replicate 10 --repeats 1 || echo "FAILED cpu 1.5mm 10"
date

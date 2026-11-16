#!/bin/bash
#SBATCH -p gpu-gh200-public
#SBATCH --gres=gpu:1
#SBATCH -c 72
#SBATCH --mem=450G
#SBATCH -t 01:30:00
#SBATCH -J petG3_fast
#SBATCH -o /sfs/ceph/hachi/standard/ivy-hip-monte/runG3_%j.out
R=/sfs/ceph/hachi/standard/ivy-hip-monte
export HOME=$R/home PATH=$R/envs/pet/bin:/usr/bin:/bin NUMBA_NUM_THREADS=72
cd $R/exercise_pet_sim/recon
DIR=$R/recon
A="--recon $DIR --config R280_slot --body P90 --ctrs 200 --device gpu --repeats 1"
date; hostname; free -g | head -2
nvidia-smi --query-gpu=utilization.gpu,memory.used,power.draw --format=csv -l 5 > $R/gpu_util_G3_$SLURM_JOB_ID.csv &
echo "=== random subsets, rep 30 (3e8) $(date +%T)"; python bench_gpu.py $A --replicate 30 || echo FAILED
echo "=== fast, rep 10 (1e8) $(date +%T)"; python bench_gpu.py $A --replicate 10 --fast || echo FAILED
echo "=== fast, rep 30 (3e8) $(date +%T)"; python bench_gpu.py $A --replicate 30 --fast || echo FAILED
echo "=== fast, rep 100 (1e9) $(date +%T)"; python bench_gpu.py $A --replicate 100 --fast || echo FAILED
echo "=== fast, rep 300 (3e9) $(date +%T)"; python bench_gpu.py $A --replicate 300 --fast || echo FAILED
echo "=== fast, 1.5 mm, rep 300 (3e9) $(date +%T)"; python bench_gpu.py $A --replicate 300 --fast --voxel 1.5 || echo FAILED
date

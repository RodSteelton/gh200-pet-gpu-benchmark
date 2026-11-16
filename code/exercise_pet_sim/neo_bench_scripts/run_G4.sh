#!/bin/bash
#SBATCH -p gpu-gh200-public
#SBATCH --gres=gpu:1
#SBATCH -c 72
#SBATCH --mem=200G
#SBATCH -t 01:00:00
#SBATCH -J petG4_agree
#SBATCH -o /sfs/ceph/hachi/standard/ivy-hip-monte/runG4_%j.out
R=/sfs/ceph/hachi/standard/ivy-hip-monte
export HOME=$R/home PATH=$R/envs/pet/bin:/usr/bin:/bin NUMBA_NUM_THREADS=72
cd $R/exercise_pet_sim/recon
DIR=$R/recon
A="--recon $DIR --config R280_slot --body P90 --ctrs 200 --device both --repeats 1 --fast"
date; hostname
for K in 1 10; do
  echo "=== CPU vs GPU, interleaved subsets, replicate $K $(date +%T)"; python bench_gpu.py $A --replicate $K || echo FAILED
done
date

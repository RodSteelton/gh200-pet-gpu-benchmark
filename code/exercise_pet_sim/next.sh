#!/bin/bash
# What "xpet push" runs on Rivanna after unpacking. Claude edits this for each stage.
# 2026-10-25: stage 9 (exercise motion with panels, camera correction, 0.65 window factor) and stage 5d
# (MBF with randoms, 4 panels 30 x 24 vs ring). prep (5 geometries: sensitivity with backrest + pose calibration)
# -> 65 motion scans -> collect (email); prep -> 20 MBF scans -> collect (email).
set -eo pipefail
source ~/xpet_env.sh
cd ~/exercise_pet_sim/recon
OUT=/scratch/$USER/exercise_pet_recon
ls "$OUT"/events_fastmc_R280_slot_P90_10000000.npz "$OUT"/sens_R280_slot_P90_backrest_carbon_v3.npy \
   "$OUT"/panels_cr/tables_R280_slot.npz "$OUT"/panels_cr/tables_P4_T30A24.npz >/dev/null
mkdir -p "$OUT/logs"
mail=(--mail-type=END,FAIL --mail-user="$USER@virginia.edu")
NP=$(( $(python motion_panels.py --recon "$OUT" --nprep) - 1 ))
NT=$(( $(python motion_panels.py --recon "$OUT" --ntasks) - 1 ))
NM=$(( $(python mbf_panels.py --recon "$OUT" --ntasks) - 1 ))
E9=(--export=ALL,OUT="$OUT")
PRE=$(sbatch --parsable --array=0-"$NP" --time=10:00:00 "${E9[@]}",MODE=prep \
      --output="$OUT/logs/motion9_prep_%A_%a.log" slurm_motion9.sh)
ARR=$(sbatch --parsable --dependency=afterok:"$PRE" --kill-on-invalid-dep=yes --array=0-"$NT" "${E9[@]}",MODE=task \
      --output="$OUT/logs/motion9_%A_%a.log" slurm_motion9.sh)
COL=$(sbatch --parsable --dependency=afterany:"$ARR" --kill-on-invalid-dep=yes --cpus-per-task=4 --mem=32G --time=01:00:00 \
      "${E9[@]}",MODE=collect "${mail[@]}" --output="$OUT/logs/motion9_collect_%j.log" slurm_motion9.sh)
MB=$(sbatch --parsable --dependency=afterok:"$PRE" --kill-on-invalid-dep=yes --array=0-"$NM" "${E9[@]}",MODE=task \
      --output="$OUT/logs/mbf_panels_%A_%a.log" slurm_mbf_panels.sh)
MBC=$(sbatch --parsable --dependency=afterany:"$MB" --kill-on-invalid-dep=yes --cpus-per-task=4 --mem=32G --time=00:30:00 \
      "${E9[@]}",MODE=collect "${mail[@]}" --output="$OUT/logs/mbf_panels_collect_%j.log" slurm_mbf_panels.sh)
echo "Stage 9: prep $PRE (0-$NP), motion $ARR (0-$NT), collect $COL (email)"
echo "Stage 5d: MBF $MB (0-$NM), collect $MBC (email)"
echo "Then: .\xpet pull motion9  and  .\xpet pull mbf_panels"
echo "$OUT/motion9" > ~/exercise_pet_sim/.results_path

#!/bin/bash
#SBATCH -p gpu-gh200-public
#SBATCH --gres=gpu:1
#SBATCH -c 72
#SBATCH --mem=300G
#SBATCH -t 02:00:00
#SBATCH -J petEC_orig
#SBATCH -o /sfs/ceph/hachi/standard/ivy-hip-monte/runEC_%j.out
# Grace CPU. Run C: the two geometries added by stage 8c. Run E: again, with the ORIGINAL Rivanna inputs
# (sensitivity image and events file; the events file sets the decay count and so the noise realization).
R=/sfs/ceph/hachi/standard/ivy-hip-monte
export HOME=$R/home PATH=$R/envs/pet/bin:/usr/bin:/bin XPET_WORKERS=30 OMP_NUM_THREADS=1
cd $R/exercise_pet_sim/recon
date; hostname; lscpu | grep -E "Model name|^CPU\(s\)"
T0=$(date +%s)
# ---- C: tasks 8 and 9 of stage 8c (P4_T30A24, P4_T30A30). The 8 tables of job 641162 are copied, not rerun.
OUTC=$R/recon_C_8c; mkdir -p $OUTC/panels_cr
cp -n $R/recon_CDE_grace/panels_cr/tables_*.npz $OUTC/panels_cr/ || echo "NO 8b TABLES"
export STAGE=8c NUMBA_NUM_THREADS=1
echo "C: $(python panels_cr.py --recon $OUTC --ntasks | tail -1) count-rate tasks in stage 8c"
for TASK in 8 9; do
  echo "=== C count-rate task $TASK $(date +%T)"; S=$(date +%s)
  python panels_cr.py --recon $OUTC --task $TASK || echo "FAILED C task $TASK"
  echo "C task $TASK seconds $(( $(date +%s) - S ))"
done
python panels_cr.py --recon $OUTC --collect || echo "FAILED C collect"
find $OUTC -type f -size -20M | tar czf $R/runC8c_files_$SLURM_JOB_ID.tgz -T - 2>/dev/null
unset STAGE
# ---- E: prep and scan 0 (static, exercise) with the original inputs
OUTE=$R/recon_E_orig; mkdir -p $OUTE
cp $R/orig_inputs/sens_R280_slot_P90_backrest_carbon_v3.npy $OUTE/ || echo "NO ORIGINAL SENS FILE"
cp $R/orig_inputs/events_fastmc_R280_slot_P90_10000000.npz $OUTE/ || echo "NO ORIGINAL EVENTS FILE"
ls -l $OUTE; md5sum $OUTE/*.npy $OUTE/*.npz $R/recon/events_fastmc_R280_slot_P90_10000000.npz
python -c "import numpy as np;d=np.load('$OUTE/events_fastmc_R280_slot_P90_10000000.npz');n=int(d['n_emitted']);print('n_emitted',n,'x0.65',int(n*0.65),'(old run: 1519969308)')"
export NUMBA_NUM_THREADS=32
echo "=== E prep 0 $(date +%T)"; S=$(date +%s)
python motion_panels.py --recon $OUTE --prep 0 || echo "FAILED E prep"
echo "E prep seconds $(( $(date +%s) - S ))"
for TASK in 0 5; do
  echo "=== E motion task $TASK $(date +%T)"; S=$(date +%s)
  python motion_panels.py --recon $OUTE --task $TASK || echo "FAILED E task $TASK"
  echo "E task $TASK seconds $(( $(date +%s) - S ))"
done
find $OUTE/motion9 -type f -size -20M | tar czf $R/runEorig_files_$SLURM_JOB_ID.tgz -T - 2>/dev/null
echo "total seconds $(( $(date +%s) - T0 ))"; date; echo "ALL DONE"

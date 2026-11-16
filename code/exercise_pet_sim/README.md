# Exercise cardiac PET: ring geometry study (GATE 10)

Compares four detector ring configurations on a cardiac torso phantom:

| Gap per side | Coverage | Modules | Crystals |
|---|---|---|---|
| 0° | 360° closed ring | 72 | 36,288 |
| 20° | 320° | 64 | 32,256 |
| 30° | 300° (proposed split ring) | 60 | 30,240 |
| 60° | 240° (2017 dual panel) | 48 | 24,192 |

Scanner: 56 cm bore, 23.8 cm axial FOV, LYSO 3.2 × 3.2 × 20 mm crystals (3.3 mm pitch),
read out in blocks of 7 × 8 crystals (energy summed per block, position of the highest-energy
interaction), 10% energy resolution, 435–585 keV window. TOF is applied after simulation at 200, 250 and 320 ps,
so each geometry is simulated once.

## Run on Rivanna

```
source ~/gate10_env.sh
cd ~/exercise_pet_sim
python make_phantom.py        # once; writes data/ and data/setup_geometry.png
python run_sim.py --test      # about a minute; prints "TEST OK" and decays/s
bash submit_all.sh            # 80 simulation tasks + 1 analysis job
squeue -u $USER               # progress
```

Results land in `/scratch/$USER/exercise_pet/analysis/`:
`summary.csv`, `fig1_sensitivity_vs_gap.png`, `fig2_angular_coverage.png`, `fig3_tof_effective_nec.png`.
Scratch is purged after 90 days unused; copy `analysis/` somewhere permanent.

## Files

- `config.py`: every parameter (geometry, energy window, CTR values, phantom, event counts)
- `scanner.py`: split-ring geometry; the gap removes modules around the lateral axis
- `make_phantom.py`: voxelized torso (lungs, spine, liver, LV myocardium and blood pool), N-13 ammonia-like uptake
- `run_sim.py`: one simulation chunk; reduces singles to coincidences and saves a small `.npz`
- `analyze.py`: sensitivity, scatter fraction, NEC, angular coverage, TOF-effective NEC
- `slurm_sim.sh`, `slurm_analyze.sh`, `submit_all.sh`: Slurm job array on the `berr_medical_imaging` allocation

## Version 1 simplifications

- No randoms (events are paired by event number); randoms can be added analytically for NECR.
- Back-to-back 511 keV photons with acolinearity; positron range not simulated (scanner independent).
- Static phantom, arms outside the FOV, no bed, no depth of interaction.
- Scatter is identified as coincidences whose LOR misses the emission point by more than 5 mm.
- TOF benefit uses the standard D/Δx approximation; image reconstruction is the next stage.

## Sweep 2: body size, arms, bore size

Target population: 90th percentile of US adults aged 60+ (NHANES 2015–2018 arm circumference;
chest size estimated from girth). Two bodies: P50 (chest 37 × 26 cm, arm 10.4 cm) and P90
(chest 44 × 32 cm, arm 12 cm); the extra girth is an adipose layer around the sweep-1 torso.
Arms are soft-tissue cylinders with a humerus, with their own activity, hanging at the sides.

Slots are sized so the P90 arm clears the detectors by 2 cm (whole modules removed):

| Geometry | Coverage | Slot | Modules |
|---|---|---|---|
| 56 cm bore, slots | 290° | 17.0 cm | 58 |
| 62 cm bore, slots | 297° | 17.0 cm | 66 |
| 70 cm bore, slots | 320° | 12.3 cm | 80 |
| 70 cm bore, closed, arms inside (reference) | 360° | none | 90 |

```
source ~/gate10_env.sh
cd ~/exercise_pet_sim
python make_phantom.py                                   # builds P50/P90 phantoms + data/setup_sweep2.png
python run_sim.py --config R280_slot --body P90 --test   # about a minute
bash submit_all2.sh                                      # 160 tasks + analysis; emails you when done
```

Results: `/scratch/$USER/exercise_pet_sweep2/analysis/` (summary2.csv, Fig_sweep2_counts, Fig_sweep2_angular).
All relative values use the 70 cm closed ring with the same body as reference.

## Stage 2: reconstruction (folder `recon/`)

Compares the 56 cm 290° ring (`R280_slot`) with the 70 cm closed ring (`R350_closed`) for the P50 and
P90 bodies. TOF list-mode OSEM (4 iterations × 10 subsets, 3 mm voxels) at non-TOF, 320 ps and 200 ps.

- `projector.py`  numba line projector with exact adjoint (multi-core CPU)
- `geometry.py`   module layout, crystal snapping (parallax included), attenuation and truth images, LV frame
- `fastmc.py`     fast Monte Carlo of trues (exponential depth in LYSO, attenuation) and the sensitivity image
- `run_recon.py`  one geometry and body; `--data fastmc` (10 M trues) or `--data gate` (sweep-2 trues)
- `evaluate.py`   myocardial recovery, 16-segment non-uniformity, ant+inf / sep+lat wall ratio
                  (limited-angle bias), myocardium to blood; figures and `recon_metrics.csv`

Run: `cd recon && bash submit_recon.sh`. Four array tasks of 32 cores, then an evaluation job that
emails on completion. Results in `/scratch/$USER/exercise_pet_recon/evaluation`.

Simplifications: trues only (perfect scatter and randoms correction), no motion, no positron range.

### Stage 2b: iterations and correction (`recon/iterations.py`)

Re-reconstructs the cached fast-MC events at 4, 8 and 16 iterations and tests a segment correction measured
on one body and applied to the other. Figures `Fig_iter_metrics` and `Fig_iter_correction`, table
`iter_metrics.csv` in `/scratch/$USER/exercise_pet_recon/iterations`.

## Running from Windows (`xpet`)

The Claude-analysis folder holds `xpet.ps1`/`xpet.cmd`: `.\xpet push` copies the zip, unpacks it and runs
`next.sh`; `.\xpet status` shows the queue; `.\xpet pull` copies the results listed in `.results_path` into
`results\`.

### Stage 3: robustness (`recon/robust.py`)

P90 body, 20% mid-anterior defect, five heart poses (base, 15 mm lateral, 20 mm cranial, 15° tilt, 15° rotation),
320 and 200 ps, 8 x 10 OSEM. A fixed segment correction from the P50 base-pose phantom is applied. Outputs in
`/scratch/$USER/exercise_pet_recon/robust`, including a short `SUMMARY.txt`.

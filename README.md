# gh200-pet-gpu-benchmark

Benchmark codes and result tables for a study of a GPU (CUDA) port of a fast Monte Carlo and time-of-flight (TOF) list-mode OSEM reconstruction for PET. The work compares an NVIDIA GH200 (UVA NEO cluster) with an Afton CPU node, an A100 and a B200 on UVA's Rivanna cluster, using an open-ring exercise cardiac PET scanner model as the workload.

Paper: in preparation (Physics in Medicine and Biology). Citation and DOI will be added when available.

## Layout

- `code/exercise_pet_sim/`  scanner model, phantoms, simulation, reconstruction and the CUDA ports.
  - `recon/projector_cuda.py`, `recon/fastmc_cuda.py`  CUDA versions of the projector and fast Monte Carlo (numba-cuda).
  - `recon/bench_gpu.py`, `recon/bench_mc.py`, `recon/dynamic_gpu.py`  benchmarks (TOF OSEM, Monte Carlo, 15O-water dynamic scans).
  - `neo_bench_scripts/`  Slurm scripts and `collect_FS.py` for the benchmark runs, including the 120 scan run (run FS).
  - `README_neo_bench.md`  build notes and self tests (written 2026-11-10).
  - `data/`  P50 and P90 body phantoms.
- `results/`  benchmark JSON files, Slurm logs, GPU utilization CSVs, and the summaries for each run (`CDE`, `FS`, `FS_a100`, `FS_b200`, `rivanna`). Reconstructed image arrays (`*.npy`) and archives of per-scan files (`*.tgz`) are not included because of size.
- `tables/`  the paper's tables as CSV and Markdown.
- `paper_scripts/`  scripts that build the paper's figures and tables from the results.

## Running

Self tests on a GPU node, with a numba-cuda environment active:

    cd code/exercise_pet_sim/recon
    python bench_gpu.py --selftest
    python bench_mc.py --selftest

Benchmarks write JSON files to `DIR/bench/`:

    python bench_mc.py  --recon DIR --config R280_slot --body P90
    python bench_gpu.py --recon DIR --config R350_closed --body P90 --save-images

## Notes

- Slurm scripts use UVA cluster paths, module names and account settings; edit them for another system.
- Simulated event files are large and are not included. They are available from the corresponding author upon reasonable request.
- AI use: a generative AI assistant (Claude, Anthropic) helped write the CUDA ports and analysis scripts under the direction of the authors. The GPU codes were checked against the CPU codes. See the paper for details.

## License

MIT License, copyright (c) 2026 Stuart S. Berr. See `LICENSE`.

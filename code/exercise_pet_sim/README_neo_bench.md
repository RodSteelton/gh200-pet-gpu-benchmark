# neo_bench: GPU benchmark build (2026-11-10, second version)

`exercise_pet_sim.zip` of 2026-10-25 (SHA-256 6bcb7463...e72522) plus:

- `recon/projector_cuda.py`  CUDA (numba-cuda) version of `recon/projector.py`: `fwd`, `bwd`, and `osem`
  (events, image and sensitivity stay on the GPU; one fused forward + back kernel per subset).
- `recon/fastmc_cuda.py`     CUDA version of the fast Monte Carlo for ring scanners: `generate_trues`
  (stage 2), `emit_timed` (15O-water scans, as `dynamic._emit_timed`) and `sensitivity`.
- `recon/bench_gpu.py`       benchmark B (TOF list-mode OSEM), CPU vs GPU, with a self test.
- `recon/bench_mc.py`        benchmark A (fast Monte Carlo), CPU vs GPU, with a self test.
- `data/phantom_P50.npz`, `data/phantom_P90.npz`  phantoms rebuilt with `make_phantom.py`, so SimpleITK is
  not needed on the cluster.

Nothing else in the package is changed.

Self tests (GPU node, environment active; a few minutes each):

    cd exercise_pet_sim/recon
    python bench_gpu.py --selftest
    python bench_mc.py --selftest

Benchmarks (results in DIR/bench/*.json):

    python bench_mc.py  --recon DIR --config R280_slot --body P90
    python bench_gpu.py --recon DIR --config R350_closed --body P90 --save-images
    python bench_gpu.py --recon DIR --config R280_slot --body P90 --save-images

Agreement, reconstruction: float32 rounding except for about 1 LOR in 1e4, which gets one sample more or less
along the line because the last bit of the float32 LOR direction differs between builds (fastmath on the CPU,
fused multiply-add on the GPU, x86 vs Arm). The self test counts these LORs and checks all the others.
Agreement, Monte Carlo: statistical (different random generators): acceptance and distributions within
counting error.

Status 2026-11-10: projector self test PASSED on a GH200 node. Monte Carlo kernels pass the statistical checks
when their source is compiled for the CPU; not yet run on a GPU.

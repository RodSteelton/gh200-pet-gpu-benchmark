"""
Benchmark B: TOF list-mode OSEM on the CPU (numba, projector.py) and on the GPU (numba-cuda, projector_cuda.py).

    python bench_gpu.py --selftest
        Correctness only, small synthetic problem, no input files. Works on a GPU node, and without a GPU
        with NUMBA_ENABLE_CUDASIM=1 (slow simulator, tiny problem).

    python bench_gpu.py --recon DIR --config R350_closed --body P90
        Run B. DIR holds (or receives) the stage-2 inputs sens_<config>_<body>_v3.npy and
        events_fastmc_<config>_<body>_10000000.npz. Missing inputs are generated as in run_recon.py
        (seed 7; needs ../data/phantom_<body>.npz). For each CTR (default non-TOF, 320, 200 ps) the same
        events, TOF noise and subsets are reconstructed on each device: one small warm-up call (JIT compile,
        timed separately), then --repeats timed runs. Writes DIR/bench/bench_B_<config>_<body>_<host>_<time>.json.

    Options: --device cpu|gpu|both, --n-events N (first N events), --replicate K (K copies of the events with
    new TOF noise, for scaling), --voxel, --iters, --subsets, --save-images.

Agreement is reported as max |GPU - CPU| / max(CPU), RMSE / mean(CPU), the number of LORs sampled with a
different number of points on the two devices and, when the phantom is present, the stage-2 LV metrics
(myocardial recovery, segment CV, ant+inf / sep+lat) for both images.

Why not bit for bit: the number of samples on a LOR is ceil(length in the box / step). The LOR direction is
float32, and its last bit differs between builds (fastmath on the CPU, fused multiply-add on the GPU, x86 vs
Arm), so about 1 LOR in 1e5 gets one sample more or less. All other LORs agree to float32 rounding.
"""

import argparse
import json
import os
import platform
import re
import socket
import sys
import time
from pathlib import Path

import numpy as np

import projector as PC
from projector import Grid

C_MM_PER_PS = 0.299792458


def log(msg):
    print(msg, flush=True)


# ---------------------------------------------------------------- environment record
def env_info(use_gpu):
    import numba
    info = {
        "host": socket.gethostname(), "machine": platform.machine(), "platform": platform.platform(),
        "python": platform.python_version(), "numpy": np.__version__, "numba": numba.__version__,
        "cpu_count": os.cpu_count(), "numba_threads": numba.get_num_threads(),
        "slurm_job": os.environ.get("SLURM_JOB_ID"), "slurm_cpus": os.environ.get("SLURM_CPUS_PER_TASK"),
        "slurm_partition": os.environ.get("SLURM_JOB_PARTITION"), "time": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    try:
        import scipy
        info["scipy"] = scipy.__version__
    except Exception:
        info["scipy"] = None
    try:
        for line in open("/proc/cpuinfo"):
            if line.lower().startswith(("model name", "cpu part", "hardware")):
                info["cpu_model"] = line.split(":", 1)[1].strip()
                break
    except Exception:
        pass
    if use_gpu:
        import projector_cuda as PG
        from numba import config as nbconfig
        info["gpu"] = PG.device_name()
        info["cuda_simulator"] = bool(getattr(nbconfig, "ENABLE_CUDASIM", False))
        try:
            import numba_cuda
            info["numba_cuda"] = getattr(numba_cuda, "__version__", "installed")
        except Exception:
            info["numba_cuda"] = None
    return info


def threading_layer():
    try:
        import numba
        return numba.threading_layer()
    except Exception:
        return None


# ---------------------------------------------------------------- self test
def _cpu_twin():
    """projector_cuda.py recompiled for the CPU with numba.njit: the same source, so the same type inference
    and float32/float64 arithmetic as the CUDA build, with atomic adds replaced by plain adds."""
    from numba import njit
    src = (Path(__file__).resolve().parent / "projector_cuda.py").read_text()
    src = src.replace("from numba import cuda\n", "from numba import njit\n")
    src = src.replace("@cuda.jit(device=True)\n", "@njit\n").replace("@cuda.jit\n", "@njit\n")
    src = re.sub(r"cuda\.atomic\.add\((\w+), (.+?), (np\.float32\(.+\))\)\n", r"\1[\2] += \3\n", src)
    src = re.sub(r"def (_k_\w+)\(", r"def \1(_idx, ", src)
    src = re.sub(r"(\w) = cuda\.grid\(1\)", r"\1 = _idx", src)
    src = src.split("# ---------------------------------------------------------------- host wrappers")[0]
    ns = {}
    exec(compile(src, "projector_cuda_twin", "exec"), ns)

    @njit
    def run_fwd(k, n, img, x0, y0, z0, v, nx, ny, nz, p1, p2, tc, sigma, step, out):
        for e in range(n):
            k(e, img, x0, y0, z0, v, nx, ny, nz, p1, p2, tc, sigma, step, out)

    @njit
    def run_back(k, n, vals, x0, y0, z0, v, nx, ny, nz, p1, p2, tc, sigma, step, acc):
        for e in range(n):
            k(e, vals, x0, y0, z0, v, nx, ny, nz, p1, p2, tc, sigma, step, acc)

    @njit
    def run_osem(k, n, img, bp, x0, y0, z0, v, nx, ny, nz, p1, p2, tc, sigma, step):
        for e in range(n):
            k(e, img, bp, x0, y0, z0, v, nx, ny, nz, p1, p2, tc, sigma, step)

    @njit
    def run_update(k, n, img, bp, senss):
        for i in range(n):
            k(i, img, bp, senss)

    return ns, run_fwd, run_back, run_osem, run_update


def _synthetic(n, grid, rng, radius, zmax):
    a1, a2 = rng.uniform(0, 2 * np.pi, n), rng.uniform(0, 2 * np.pi, n)
    p1 = np.stack([radius * np.cos(a1), radius * np.sin(a1), rng.uniform(-zmax, zmax, n)], 1).astype(np.float32)
    p2 = np.stack([radius * np.cos(a2), radius * np.sin(a2), rng.uniform(-zmax, zmax, n)], 1).astype(np.float32)
    p2[:3] = p1[:3]                                    # zero-length LORs
    p1[3] = (2 * radius, 5.0, 0.0)                     # LOR parallel to x that misses the box in z
    p2[3] = (-2 * radius, 5.0, 0.0)
    p1[3, 2] = p2[3, 2] = 10 * zmax
    D = np.linalg.norm(p2 - p1, axis=1)
    tc = (D / 2 + rng.normal(0, 15.0, n)).astype(np.float32)
    img = rng.random(grid.shape).astype(np.float32)
    vals = rng.random(n).astype(np.float32)
    vals[5] = 0.0
    return p1, p2, tc, img, vals


def _emission(n, grid, rng, radius, zmax, sigma_tof):
    """Consistent list-mode data for the OSEM test: decays in a central cylinder, back-to-back photons to a
    detector cylinder (radius, |z| <= zmax), TOF position along the LOR with Gaussian noise."""
    hx, hz = (grid.nx - 1) / 2.0 * grid.v, (grid.nz - 1) / 2.0 * grid.v
    m = 4 * n
    r = 0.5 * hx * np.sqrt(rng.random(m))
    a = rng.uniform(0, 2 * np.pi, m)
    e = np.stack([r * np.cos(a), r * np.sin(a), rng.uniform(-0.5 * hz, 0.5 * hz, m)], 1)
    cz = rng.uniform(-1, 1, m)
    ph = rng.uniform(0, 2 * np.pi, m)
    sz = np.sqrt(1 - cz * cz)
    d = np.stack([sz * np.cos(ph), sz * np.sin(ph), cz], 1)
    qa = (d[:, :2] ** 2).sum(1)
    qb = 2 * (e[:, :2] * d[:, :2]).sum(1)
    qc = (e[:, :2] ** 2).sum(1) - radius ** 2
    ok = qa > 1e-6
    root = np.sqrt(np.maximum(qb * qb - 4 * qa * qc, 0))
    t1 = (-qb + root) / (2 * np.maximum(qa, 1e-6))
    t2 = (-qb - root) / (2 * np.maximum(qa, 1e-6))
    p1, p2 = e + t1[:, None] * d, e + t2[:, None] * d
    ok &= (np.abs(p1[:, 2]) <= zmax) & (np.abs(p2[:, 2]) <= zmax)
    p1, p2, t1 = p1[ok][:n], p2[ok][:n], t1[ok][:n]
    tc = (t1 + rng.normal(0, sigma_tof, len(t1))).astype(np.float32)
    return p1.astype(np.float32), p2.astype(np.float32), tc


def _rel(a, b):
    return float(np.abs(a - b).max() / max(np.abs(b).max(), 1e-30))


def _flags(a, b):
    """LORs whose two projections differ by more than rounding. The number of samples on a LOR is
    ceil(length in the box / step); float32 rounding of the LOR direction differs between builds (fastmath
    on the CPU, fused multiply-add on the GPU), so about 1 LOR in 1e5 gets one sample more or less."""
    return np.abs(a - b) > 1e-4 * np.abs(b) + 1e-12 * np.abs(b).max()


def _noise_images(grid, rng, k=3):
    return [rng.random(grid.shape).astype(np.float32) for _ in range(k)]


def _flag_union(fa, fb):
    """Union of _flags over several test images. One image can miss a LOR whose two samplings happen to give
    nearly the same value (about 3% of them); three independent images cannot."""
    flag = np.zeros(len(fa[0]), bool)
    for a, b in zip(fa, fb):
        flag |= _flags(a, b)
    return flag


FLAG_LIMIT = 2e-3              # allowed fraction of LORs with a different sample count


def selftest():
    from numba import config as nbconfig
    import projector_cuda as PG
    from run_recon import osem as osem_cpu
    import run_recon
    run_recon.log = lambda m: None
    ok = True
    rng = np.random.default_rng(1)

    def check(name, val, tol):
        nonlocal ok
        good = bool(val <= tol)
        ok &= good
        log(f"  {'PASS' if good else 'FAIL'}  {name}: {val:.2e} (limit {tol:.0e})")

    def check_fwd(name, f_new, f_cpu):
        """Forward projections of several noise images (lists): count LORs with a different sample count,
        compare all the others."""
        flag = _flag_union(f_new, f_cpu)
        nz = max(int((f_cpu[0] > 0).sum()), 1)
        log(f"        {name}: {int(flag.sum())} of {nz:,} LORs sampled with a different count; "
            f"largest difference over all LORs {max(_rel(a, b) for a, b in zip(f_new, f_cpu)):.1e} of max")
        check(f"{name}, fraction of LORs with a different sample count", flag.sum() / nz, FLAG_LIMIT)
        check(f"{name}, all other LORs",
              max(_rel(np.where(flag, 0, a), np.where(flag, 0, b)) for a, b in zip(f_new, f_cpu)), 1e-5)
        return flag

    # 1. same source compiled for the CPU: checks typing and arithmetic of the kernels against projector.py
    log("1. CUDA kernel source compiled with njit, against projector.py")
    grid = Grid(40, 40, 24, 3.0)
    p1, p2, tc, img, vals = _synthetic(20000, grid, rng, 150.0, 40.0)
    ns, run_fwd, run_back, run_osem, run_update = _cpu_twin()
    gp = PG._gp(grid)
    nvox = grid.nx * grid.ny * grid.nz
    tests = [img] + _noise_images(grid, rng, 2)
    for sigma in (0.0, 12.7):
        outs = []
        for im in tests:
            out = np.zeros(len(p1), np.float32)
            run_fwd(ns["_k_forward"], len(p1), im.ravel(), *gp, p1, p2, tc, sigma, 3.0, out)
            outs.append(out)
        fps = [PC.fwd(im, grid, p1, p2, tc, sigma) for im in tests]
        fp = fps[0]
        keep = ~check_fwd(f"forward, sigma {sigma}", outs, fps)
        q1, q2, qt, qv = p1[keep], p2[keep], tc[keep], vals[keep]
        acc = np.zeros(nvox, np.float32)
        run_back(ns["_k_back"], len(q1), qv, *gp, q1, q2, qt, sigma, 3.0, acc)
        check(f"back, sigma {sigma}", _rel(acc.reshape(grid.shape), PC.bwd(qv, grid, q1, q2, qt, sigma)), 1e-4)
        bp = np.zeros(nvox, np.float32)
        run_osem(ns["_k_osem"], len(q1), img.ravel(), bp, *gp, q1, q2, qt, sigma, 3.0)
        ratio = np.where(fp > 1e-12, 1.0 / np.maximum(fp, 1e-12), 0.0).astype(np.float32)[keep]
        ref = PC.bwd(ratio, grid, q1, q2, qt, sigma)
        check(f"fused OSEM step, sigma {sigma}", _rel(bp.reshape(grid.shape), ref), 1e-4)
        senss = (ref.ravel() + 1).astype(np.float32)
        senss[::7] = 0
        im2 = img.ravel().copy()
        bp2 = ref.ravel().copy()
        run_update(ns["_k_update"], nvox, im2, bp2, senss)
        exp = np.where(senss > 0, img.ravel() * ref.ravel() / np.where(senss > 0, senss, 1), 0).astype(np.float32)
        check("update", max(_rel(im2, exp), float(np.abs(bp2).max())), 1e-6)

    # 2. the CUDA build itself (GPU, or simulator at a tiny size)
    sim = bool(getattr(nbconfig, "ENABLE_CUDASIM", False))
    if not PG.available():
        log("2. no CUDA device and no simulator: GPU checks skipped (set NUMBA_ENABLE_CUDASIM=1 to simulate)")
        return ok
    log(f"2. CUDA build on {'the simulator' if sim else PG.device_name()}, against projector.py")
    grid = Grid(16, 16, 10, 3.0) if sim else Grid(120, 120, 60, 3.0)
    n = 120 if sim else 400000
    p1, p2, tc, img, vals = _synthetic(n, grid, rng, 60.0 if sim else 260.0, 18.0 if sim else 100.0)
    it, sub = (2, 2) if sim else (3, 10)
    rad, zm = (60.0, 18.0) if sim else (260.0, 100.0)
    tests = [img] + _noise_images(grid, rng, 2)
    for sigma in (0.0, 12.7):
        fcs = [PC.fwd(im, grid, p1, p2, tc, sigma) for im in tests]
        fgs = [PG.fwd(im, grid, p1, p2, tc, sigma) for im in tests]
        f_g = fgs[0]
        keep = ~check_fwd(f"forward, sigma {sigma}", fgs, fcs)
        q1, q2, qt, qv = p1[keep], p2[keep], tc[keep], vals[keep]
        b_c, b_g = PC.bwd(qv, grid, q1, q2, qt, sigma), PG.bwd(qv, grid, q1, q2, qt, sigma)
        check(f"back, sigma {sigma}", _rel(b_g, b_c), 1e-4)
        lhs = float(np.dot(f_g[keep].astype(np.float64), qv))
        rhs = float(np.dot(img.ravel().astype(np.float64), b_g.ravel()))
        check(f"adjoint <Ax,y> = <x,A'y>, sigma {sigma}", abs(lhs - rhs) / abs(lhs), 1e-4)
        # OSEM on consistent data (decays in a central cylinder); sensitivity from random LORs
        e1, e2, etc = _emission(n, grid, rng, rad, zm, 12.7)
        sens = PC.bwd(np.ones(len(p1), np.float32), grid, p1, p2)
        ev_all = {"p1": e1, "p2": e2, "tc": etc}
        j_c = osem_cpu(ev_all, grid, sens, sigma, it, sub, np.random.default_rng(5))
        j_g = PG.osem(ev_all, grid, sens, sigma, it, sub, np.random.default_rng(5), log=lambda m: None)
        ek = ~_flag_union([PG.fwd(im, grid, e1, e2, etc, sigma) for im in tests],
                          [PC.fwd(im, grid, e1, e2, etc, sigma) for im in tests])
        ev = {"p1": e1[ek], "p2": e2[ek], "tc": etc[ek]}
        i_c = osem_cpu(ev, grid, sens, sigma, it, sub, np.random.default_rng(5))
        i_g = PG.osem(ev, grid, sens, sigma, it, sub, np.random.default_rng(5), log=lambda m: None)
        log(f"        OSEM {it}x{sub}, sigma {sigma}: all {len(e1):,} events {_rel(j_g, j_c):.1e} of max; "
            f"without the {int((~ek).sum())} events sampled with a different count:")
        check(f"OSEM {it}x{sub}, sigma {sigma}", _rel(i_g, i_c), 2e-4)
    snaps = PG.osem(ev, grid, sens, 12.7, it, sub, np.random.default_rng(5), snapshots=(1, it), log=lambda m: None)
    check("OSEM snapshots", _rel(snaps[it], i_g), 1e-6)
    return ok


# ---------------------------------------------------------------- run B
def load_inputs(recon, config, body, voxel, n_trues, sens_events, seed):
    """Stage-2 sensitivity image and fast-MC events, generated as in run_recon.py when missing."""
    import geometry as G
    ring = G.Ring(config)
    grid = G.recon_grid(ring, voxel)
    tag = f"{config}_{body}"
    sens_path = recon / f"sens_{tag}_v{voxel:g}.npy"
    ev_path = recon / f"events_fastmc_{tag}_{int(n_trues)}.npz"
    made = {}
    if abs(voxel - 3.0) > 1e-9:
        # timing benchmark only: the phantom exists on a 3 mm grid, so the finer grid reuses the 3 mm
        # sensitivity image by nearest-neighbor upsampling (events are in mm and independent of the grid)
        s3 = np.load(recon / f"sens_{tag}_v3.npy")
        idx = [(np.arange(n) * s3.shape[a] / n).astype(int) for a, n in enumerate(grid.shape)]
        sens = np.ascontiguousarray(s3[np.ix_(*idx)])
        ev = dict(np.load(ev_path))
        return grid, sens, ev, made, {"sens": f"upsampled from sens_{tag}_v3.npy", "events": ev_path.name,
                                      "n_emitted": int(ev.get("n_emitted", 0))}
    if not (sens_path.exists() and ev_path.exists()):
        import fastmc
        ph = G.load_body(body)
        mu = G.mu_map(grid, ph)
        rng = np.random.default_rng(seed)
        if not sens_path.exists():
            t0 = time.perf_counter()
            np.save(sens_path, fastmc.sensitivity(ring, grid, mu, int(sens_events), rng, log=log))
            made["sensitivity_s"] = time.perf_counter() - t0
        if not ev_path.exists():
            t0 = time.perf_counter()
            np.savez(ev_path, **fastmc.generate_trues(ring, grid, ph, mu, int(n_trues), rng, log=log))
            made["events_s"] = time.perf_counter() - t0
    sens = np.load(sens_path)
    ev = dict(np.load(ev_path))
    if tuple(sens.shape) != grid.shape:
        raise SystemExit(f"{sens_path.name} has shape {sens.shape}, expected {grid.shape}")
    return grid, sens, ev, made, {"sens": sens_path.name, "events": ev_path.name,
                                  "n_emitted": int(ev.get("n_emitted", 0))}


def lv_metrics(img, config, body):
    """Stage-2 LV metrics (as evaluate.collect); None when the phantom is not available."""
    try:
        import evaluate as E
        t = E.Truth.get(config, body)
        if tuple(img.shape) != t["grid"].shape:
            return None
        im = E.normalise(img, t)
        seg = E.segment_values(im, t)
        _, prof = E.mid_profile(im, t)
        return {"myo_recovery": float(np.nanmean(seg)),
                "segment_cv_pct": float(100 * np.nanstd(seg) / np.nanmean(seg)),
                "ant_inf_over_sep_lat": float(np.nanmean(prof[[0, 5, 6, 11]]) / np.nanmean(prof[[2, 3, 8, 9]]))}
    except Exception as exc:
        return {"unavailable": f"{type(exc).__name__}: {exc}"}


def _summary(ts):
    return {"median_s": float(np.median(ts)), "min_s": float(np.min(ts)), "max_s": float(np.max(ts)),
            "runs_s": [float(t) for t in ts]}


class _IdentityRNG:
    """Stand-in rng whose permutation is the identity: subset s is events s, s+subsets, ... on any device."""

    def permutation(self, n):
        return np.arange(n)


def run(args):
    import run_recon
    run_recon.log = lambda m: None
    recon = Path(args.recon)
    recon.mkdir(parents=True, exist_ok=True)
    out = recon / "bench"
    out.mkdir(exist_ok=True)
    devices = ["cpu", "gpu"] if args.device == "both" else [args.device]
    if "gpu" in devices:
        import projector_cuda as PG
        if not PG.available():
            raise SystemExit("no CUDA device found (use --device cpu, or check the node and the environment)")
    grid, sens, ev, made, files = load_inputs(recon, args.config, args.body, args.voxel, args.n_trues,
                                              args.sens_events, args.seed)
    p1, p2, dt0 = ev["p1"], ev["p2"], ev["dt"]
    if args.n_events:
        p1, p2, dt0 = p1[:args.n_events], p2[:args.n_events], dt0[:args.n_events]
    if args.replicate > 1:
        p1, p2, dt0 = (np.concatenate([a] * args.replicate) for a in (p1, p2, dt0))
    n = len(p1)
    D = np.linalg.norm(p2 - p1, axis=1)
    res = {"benchmark": "B, TOF list-mode OSEM", "config": args.config, "body": args.body, "seed": args.seed,
           "grid": [grid.nx, grid.ny, grid.nz], "voxel_mm": grid.v, "n_events": int(n), "iters": args.iters,
           "subsets": args.subsets, "repeats": args.repeats, "replicate": args.replicate, "inputs": files,
           "generated_inputs_s": made, "env": env_info("gpu" in devices), "ctr": {}}
    log(f"[{args.config} {args.body}] grid {grid.shape}, {n:,} events, OSEM {args.iters}x{args.subsets}, "
        f"devices {devices}, {res['env']['numba_threads']} CPU threads, {res['env']['machine']}")
    passes = n * args.iters

    for ctr in [int(c) for c in args.ctrs.split(",")]:
        sig_t = ctr / 2.3548 if ctr > 0 else 0.0
        sigma = C_MM_PER_PS * sig_t / 2.0
        noise = np.random.default_rng(args.seed + 1000 + ctr).normal(size=n) * sig_t if ctr > 0 else 0.0
        evc = {"p1": p1, "p2": p2, "tc": ((D + C_MM_PER_PS * (dt0 + noise)) / 2.0).astype(np.float32)}
        small = {k: v[:20000] for k, v in evc.items()}
        row, imgs = {"sigma_mm": sigma}, {}
        for dev in devices:
            t0 = time.perf_counter()                                   # warm-up: JIT compile, context creation
            if dev == "cpu":
                run_recon.osem(small, grid, sens, sigma, 1, args.subsets, np.random.default_rng(0))
            else:
                PG.osem(small, grid, sens, sigma, 1, args.subsets, np.random.default_rng(0), log=lambda m: None)
            warm = time.perf_counter() - t0
            ts, its, ups = [], [], []
            for _ in range(args.repeats):
                rng = _IdentityRNG() if args.fast else np.random.default_rng(args.seed + 2000 + ctr)  # same subsets everywhere
                t0 = time.perf_counter()
                if dev == "cpu":
                    img = run_recon.osem(evc, grid, sens, sigma, args.iters, args.subsets, rng)
                else:
                    st = {}
                    img = PG.osem(evc, grid, sens, sigma, args.iters, args.subsets, rng, log=lambda m: None, stats=st,
                                 fast=args.fast)
                    its.append(float(np.sum(st["iter_s"])))
                    ups.append(st["upload_s"] + st["download_s"])
                    det = {k: st[k] for k in ("perm_s", "gather_s", "copy_s")}
                ts.append(time.perf_counter() - t0)
            imgs[dev] = img
            d = _summary(ts)
            d.update(warmup_s=warm, event_passes_per_s=passes / d["median_s"], lv=lv_metrics(img, args.config, args.body))
            if dev == "gpu":
                d.update(iterations_only_median_s=float(np.median(its)), transfer_median_s=float(np.median(ups)), detail_s=det, fast=bool(args.fast))
            row[dev] = d
            log(f"  CTR {ctr or 'non-TOF':>7} {dev}: median {d['median_s']:.2f} s (min {d['min_s']:.2f}, max {d['max_s']:.2f}), "
                f"{d['event_passes_per_s'] / 1e6:.1f} M event passes/s, warm-up {warm:.1f} s")
            if args.save_images:
                np.save(out / f"bench_img_{args.config}_{args.body}_ctr{ctr}_{dev}_{platform.machine()}.npy", img)
        if len(imgs) == 2:
            c, g = imgs["cpu"].astype(np.float64), imgs["gpu"].astype(np.float64)
            m = c > 0
            row["agreement"] = {"max_abs_over_max": float(np.abs(g - c).max() / c.max()),
                                "rmse_over_mean": float(np.sqrt(np.mean((g[m] - c[m]) ** 2)) / c[m].mean()),
                                "speedup_cpu_over_gpu": row["cpu"]["median_s"] / row["gpu"]["median_s"]}
            tests = [im * (sens > 0) for im in _noise_images(grid, np.random.default_rng(args.seed + 3000), 2)]
            f_c = [PC.fwd(im, grid, evc["p1"], evc["p2"], evc["tc"], sigma) for im in tests]
            f_g = [PG.fwd(im, grid, evc["p1"], evc["p2"], evc["tc"], sigma) for im in tests]
            row["agreement"]["lors_with_different_sample_count"] = int(_flag_union(f_g, f_c).sum())
            a = row["agreement"]
            log(f"  CTR {ctr or 'non-TOF':>7} GPU vs CPU: max diff {a['max_abs_over_max']:.1e} of max, "
                f"RMSE {a['rmse_over_mean']:.1e} of mean, {a['lors_with_different_sample_count']} of {n:,} LORs sampled "
                f"with a different count, speedup {a['speedup_cpu_over_gpu']:.1f}x")
        res["ctr"][str(ctr)] = row
    res["env"]["threading_layer"] = threading_layer() if "cpu" in devices else None
    name = f"bench_B_{args.config}_{args.body}_{socket.gethostname().split('.')[0]}_{time.strftime('%Y%m%d_%H%M%S')}.json"
    (out / name).write_text(json.dumps(res, indent=2))
    log(f"wrote {out / name}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--recon")
    ap.add_argument("--config", default="R350_closed")
    ap.add_argument("--body", default="P90")
    ap.add_argument("--device", default="both", choices=["cpu", "gpu", "both"])
    ap.add_argument("--ctrs", default="0,320,200")
    ap.add_argument("--iters", type=int, default=4)
    ap.add_argument("--subsets", type=int, default=10)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--n-events", type=int, default=0)
    ap.add_argument("--replicate", type=int, default=1)
    ap.add_argument("--fast", action="store_true", help="GPU: interleaved subsets, parallel pinned gather")
    ap.add_argument("--voxel", type=float, default=3.0)
    ap.add_argument("--n-trues", type=float, default=10_000_000)
    ap.add_argument("--sens-events", type=float, default=20_000_000)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--save-images", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        good = selftest()
        log("SELFTEST PASSED" if good else "SELFTEST FAILED")
        sys.exit(0 if good else 1)
    if not args.recon:
        raise SystemExit("give --recon DIR, or --selftest")
    run(args)


if __name__ == "__main__":
    main()

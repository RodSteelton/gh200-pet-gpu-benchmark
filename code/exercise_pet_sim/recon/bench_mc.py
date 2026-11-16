"""
Benchmark A: fast Monte Carlo of trues and the sensitivity image, CPU (fastmc.py, numpy + numba) against
GPU (fastmc_cuda.py, numba-cuda).

    python bench_mc.py --selftest
        Statistical check of the GPU Monte Carlo against the CPU Monte Carlo (acceptance and distributions of
        emission points, time differences and crystals), then the sensitivity image. Needs ../data/phantom_P90.npz.
        Works on a GPU node; with NUMBA_ENABLE_CUDASIM=1 only the mechanics are run (tiny sample).

    python bench_mc.py --recon DIR --config R280_slot --body P90
        Times both. CPU: --cpu-trues (default 1e6) trues; GPU: --gpu-trues (default 1e7) trues, plus the
        sensitivity image (--sens-events, default 2e6 CPU and 2e7 GPU). Reports emitted decays per second and
        the speedup, and writes DIR/bench/bench_A_<config>_<body>_<host>_<time>.json.
        --write-inputs also saves the GPU events and sensitivity as
        DIR/events_fastmcgpu_<config>_<body>_<n>.npz and DIR/sens_gpu_<config>_<body>_v3.npy.

The GPU and CPU events are different random samples (different generators), so agreement is statistical:
the acceptance (trues per emitted decay) must agree within counting error.
"""

import argparse
import json
import re
import socket
import sys
import time
from pathlib import Path

import numpy as np

import geometry as G
import fastmc
import projector as PC
from bench_gpu import env_info, log, _cpu_twin

C = G.C


# ---------------------------------------------------------------- statistics
def _z_eff(k1, n1, k2, n2):
    p = (k1 + k2) / (n1 + n2)
    return (k1 / n1 - k2 / n2) / np.sqrt(p * (1 - p) * (1 / n1 + 1 / n2))


def _z_hist(a, b, bins):
    """Two-sample chi-square on a common histogram, as a z score ((chi2 - dof) / sqrt(2 dof))."""
    ha, _ = np.histogram(a, bins)
    hb, _ = np.histogram(b, bins)
    m = (ha + hb) >= 25
    na, nb = ha.sum(), hb.sum()
    chi2 = (((ha[m] / na - hb[m] / nb) ** 2) / (ha[m] / na ** 2 + hb[m] / nb ** 2)).sum()
    dof = int(m.sum()) - 1
    return float((chi2 - dof) / np.sqrt(2 * dof))


def compare_events(a, b, ring, check):
    """Distribution checks between two event sets (dicts with p1, p2, dt, e)."""
    for i, ax in enumerate("xyz"):
        lo, hi = np.percentile(a["e"][:, i], [0.5, 99.5])
        check(f"emission {ax}", abs(_z_hist(a["e"][:, i], b["e"][:, i], np.linspace(lo, hi, 41))), 5)
    check("time difference", abs(_z_hist(a["dt"], b["dt"], np.linspace(-1500, 1500, 61))), 5)
    for name in ("p1", "p2"):
        za, zb = a[name][:, 2], b[name][:, 2]
        check(f"{name} axial crystal", abs(_z_hist(za, zb, np.linspace(-G.ZMAX, G.ZMAX, G.NA + 1))), 5)
        aa = np.degrees(np.arctan2(a[name][:, 1], a[name][:, 0])) % 360
        ab = np.degrees(np.arctan2(b[name][:, 1], b[name][:, 0])) % 360
        check(f"{name} angle", abs(_z_hist(aa, ab, np.linspace(0, 360, 181))), 5)
    la = np.linalg.norm(a["p2"] - a["p1"], axis=1)
    lb = np.linalg.norm(b["p2"] - b["p1"], axis=1)
    check("LOR length", abs(_z_hist(la, lb, np.linspace(0, 2 * ring.R + 60, 61))), 5)


def cpu_trues_whole_chunks(ring, grid, ph, mu, n_emit, rng, chunk=4_000_000):
    """CPU fast MC as fastmc.generate_trues, but for a fixed number of emitted decays and without cutting the
    last chunk. fastmc.generate_trues cuts the event list at n_target; each chunk lists torso events before
    arm events, so the cut removes arm events of the last chunk (up to a few thousand events: negligible in
    10 M, but not in a small test sample)."""
    out = {"p1": [], "p2": [], "dt": [], "e": []}
    n_done = 0
    while n_done < n_emit:
        m = int(min(chunk, n_emit - n_done))
        e = fastmc.sample_emissions(ph, m, rng)
        q1, q2, e = fastmc._pairs(ring, e, rng)
        att = np.exp(-PC.fwd(mu, grid, q1.astype(np.float32), q2.astype(np.float32)))
        keep = rng.random(len(att)) < att
        q1, q2, e = q1[keep], q2[keep], e[keep]
        s1, _ = ring.snap(q1)
        s2, _ = ring.snap(q2)
        dt = (np.linalg.norm(q1 - e, axis=1) - np.linalg.norm(q2 - e, axis=1)) / C.C_MM_PER_PS
        for k, v in (("p1", s1), ("p2", s2), ("dt", dt.astype(np.float32)), ("e", e.astype(np.float32))):
            out[k].append(v)
        n_done += m
    res = {k: np.concatenate(v) for k, v in out.items()}
    res["n_emitted"] = np.int64(n_done)
    return res


# ---------------------------------------------------------------- CPU twin of the GPU kernels
def _mc_twin():
    """fastmc_cuda.py kernels recompiled with numba.njit (same source, so the same type inference as CUDA)."""
    from numba import njit
    ns_p = _cpu_twin()[0]
    src = (Path(__file__).resolve().parent / "fastmc_cuda.py").read_text()
    src = src.replace("from numba import cuda\n", "from numba import njit\n")
    src = src.replace("from projector_cuda import _clip1, _line_fwd, _line_bwd, TPB\n", "")
    src = src.replace("@cuda.jit(device=True)\n", "@njit\n").replace("@cuda.jit\n", "@njit\n")
    src = src.replace("    return cuda.atomic.add(counter, 0, 1)\n",
                      "    i = counter[0]\n    counter[0] = i + 1\n    return i\n")
    src = src.replace("        cuda.atomic.add(counter, 0, 1)\n", "        counter[0] += 1\n")
    src = re.sub(r"def (_k_\w+)\(", r"def \1(_idx, ", src)
    src = src.replace("t = cuda.grid(1)", "t = _idx")
    src = src.split("# ---------------------------------------------------------------- host side")[0]
    ns = {k: ns_p[k] for k in ("_clip1", "_line_fwd", "_line_bwd")}
    ns["TPB"] = 256
    exec(compile(src, "fastmc_cuda_twin", "exec"), ns)

    @njit
    def run_trues(k, nthreads, states, n_per, n_extra, cdf, px, py, pz, g, kept, mu, tcdf, tgrid, o1, o2, odt, oe, ot,
                  okey, counter):
        for t in range(nthreads):
            k(t, states, n_per, n_extra, cdf, px, py, pz, g, kept, mu, tcdf, tgrid, o1, o2, odt, oe, ot, okey, counter)

    @njit
    def run_sens(k, nthreads, states, n_per, g, kept, mu, sens, counter):
        for t in range(nthreads):
            k(t, states, n_per, g, kept, mu, sens, counter)

    return ns, run_trues, run_sens


def _twin_trues(ring, grid, ph, mu, n_emit, seed, threads=4096, tcdf=None, tgrid=None, cap_frac=0.03):
    import fastmc_cuda as FG
    from numba.cuda.random import init_xoroshiro128p_states_cpu, xoroshiro128p_dtype
    ns, run_trues, _ = _mc_twin()
    states = np.empty(threads, dtype=xoroshiro128p_dtype)
    init_xoroshiro128p_states_cpu(states, seed, 0)
    n_per, n_extra = divmod(int(n_emit), threads)
    cap = int(n_emit * cap_frac) + 1000
    o1, o2, oe = (np.zeros((cap, 3), np.float32) for _ in range(3))
    odt, ot, okey, cnt = np.zeros(cap, np.float32), np.zeros(cap, np.float32), np.zeros(cap, np.int64), np.zeros(1, np.int32)
    timed = tcdf is not None
    run_trues(ns["_k_trues"], threads, states, n_per, n_extra, FG.phantom_cdf(ph),
              *(np.ascontiguousarray(ph[k], np.float64) for k in "xyz"),
              FG.params(ring, grid, ph), ring.kept_mask.astype(np.uint8), np.ascontiguousarray(mu, np.float32).ravel(),
              np.ascontiguousarray(tcdf, np.float64) if timed else np.zeros(1),
              np.ascontiguousarray(tgrid, np.float64) if timed else np.zeros(1), o1, o2, odt, oe, ot, okey, cnt)
    n = int(cnt[0])
    assert n <= cap
    return {"p1": o1[:n], "p2": o2[:n], "dt": odt[:n], "e": oe[:n], "t": ot[:n], "n_emitted": int(n_emit)}


def _timed_reference(config, ring, grid, ph, mu):
    """One 15O-water tissue class (myocardium, stress) simulated with dynamic._emit_timed (CPU)."""
    import dynamic as D
    q, _ = D.class_phantom(ph, D.class_masks(ph), "myocardium")
    dens = D.class_tacs("stress", 400.0)["myocardium"] * D.DECAY
    tcdf = np.cumsum(dens) / dens.sum()
    n = 6_000_000
    ref = D._emit_timed(config, q, mu, n, tcdf, 21)
    return q, tcdf, D.TGRID, D.EDGES, n, ref


def _timed_compare(ev, n_ev, ref, n_ref, edges, check, name):
    check(f"{name}: acceptance, |z|", abs(_z_eff(len(ev["dt"]), n_ev, len(ref["dt"]), n_ref)), 4)
    check(f"{name}: emission time per frame", abs(_z_hist(ev["t"], ref["t"], edges)), 5)
    check(f"{name}: time difference", abs(_z_hist(ev["dt"], ref["dt"], np.linspace(-1500, 1500, 61))), 5)


def _twin_sens(ring, grid, mu, n_emit, seed, threads=4096):
    import fastmc_cuda as FG
    from numba.cuda.random import init_xoroshiro128p_states_cpu, xoroshiro128p_dtype
    ns, _, run_sens = _mc_twin()
    states = np.empty(threads, dtype=xoroshiro128p_dtype)
    init_xoroshiro128p_states_cpu(states, seed, 0)
    sens, cnt = np.zeros(grid.nx * grid.ny * grid.nz, np.float32), np.zeros(1, np.int32)
    n_per = int(np.ceil(n_emit / threads))
    run_sens(ns["_k_sens"], threads, states, n_per, FG.params(ring, grid), ring.kept_mask.astype(np.uint8),
             np.ascontiguousarray(mu, np.float32).ravel(), sens, cnt)
    return sens.reshape(grid.shape), int(cnt[0]), threads * n_per


def _sens_compare(s_new, n_new, s_cpu, n_cpu, check, name):
    """Sensitivity images scaled per detected event: compare the mean in a central block of voxels and the
    transaxial and axial profiles (smoothed over blocks, so counting noise is small)."""
    a, b = s_new / n_new, s_cpu / n_cpu
    nz, ny, nx = a.shape
    core = (slice(nz // 4, 3 * nz // 4), slice(ny // 3, 2 * ny // 3), slice(nx // 3, 2 * nx // 3))
    check(f"{name}: central mean, relative difference", abs(a[core].mean() / b[core].mean() - 1), 0.02)
    pa, pb = a[:, ny // 3:2 * ny // 3, nx // 3:2 * nx // 3].mean((1, 2)), b[:, ny // 3:2 * ny // 3, nx // 3:2 * nx // 3].mean((1, 2))
    k = pb > 0.25 * pb.max()                       # end slices have few counts
    check(f"{name}: axial profile, largest relative difference", float(np.abs(pa[k] / pb[k] - 1).max()), 0.04)
    qa, qb = a[nz // 4:3 * nz // 4].mean(0), b[nz // 4:3 * nz // 4].mean(0)
    f = 8
    qa = qa[:ny // f * f, :nx // f * f].reshape(ny // f, f, nx // f, f).mean((1, 3))
    qb = qb[:ny // f * f, :nx // f * f].reshape(ny // f, f, nx // f, f).mean((1, 3))
    m = qb > 0.5 * qb.max()
    check(f"{name}: transaxial map (8 x 8 voxel blocks), largest relative difference", float(np.abs(qa[m] / qb[m] - 1).max()), 0.08)


def selftest(config, body):
    import fastmc_cuda as FG
    ok = True

    def check(name, val, tol):
        nonlocal ok
        good = bool(val <= tol)
        ok &= good
        log(f"  {'PASS' if good else 'FAIL'}  {name}: {val:.3g} (limit {tol:g})")

    ring = G.Ring(config)
    grid = G.recon_grid(ring, 3.0)
    ph = G.load_body(body)
    mu = G.mu_map(grid, ph)
    sim = FG._simulated()
    big = FG.cuda.is_available() and not sim
    if sim:
        log("CUDA simulator: mechanics only, tiny sample (the statistical checks need a GPU, or run without "
            "NUMBA_ENABLE_CUDASIM for the CPU-compiled check)")
        # one thread: the simulator swaps module globals per call, which is not safe across its threads
        ev = FG.generate_trues(ring, grid, ph, mu, 5, 5, threads=1, pilot=4000, log=lambda m: None)
        check("events returned, shortfall", 5 - len(ev["dt"]), 0)
        check("endpoints on crystal radius, mm", float(np.abs(np.hypot(ev["p1"][:, 0], ev["p1"][:, 1]) - (ring.R + G.DEPTH / 2)).max()), 12)
        check("time differences finite (0 = yes)", float(not np.isfinite(ev["dt"]).all()), 0)
        tg = np.linspace(0.05, 239.95, 2400)
        ev = FG.emit_timed(ring, grid, ph, mu, 4001, np.linspace(1 / 2400, 1, 2400), tg, 5, threads=1, pilot=4000, log=lambda m: None)
        check("timed emission: times inside the scan (0 = yes)", float(not ((ev["t"] >= 0) & (ev["t"] <= 240)).all() or len(ev["t"]) == 0), 0)
        s = FG.sensitivity(ring, grid, mu, 20, 6, threads=1, pilot=600, log=lambda m: None)
        check("sensitivity image has no counts (1 = empty)", float(s.sum() <= 0), 0)
        return ok

    n_emit_cpu = 96_000_000 if big else 36_000_000
    log(f"reference: CPU fast MC, {n_emit_cpu:,} decays, {config} {body}")
    ref = cpu_trues_whole_chunks(ring, grid, ph, mu, n_emit_cpu, np.random.default_rng(3))
    n_cpu = len(ref["dt"])
    log(f"  {n_cpu:,} trues from {n_emit_cpu:,} emissions (acceptance {n_cpu / n_emit_cpu:.5f})")
    log("reference: CPU sensitivity image")
    n_s = 3_000_000 if big else 1_500_000
    counts = []
    s_cpu = fastmc.sensitivity(ring, grid, mu, n_s, np.random.default_rng(4),
                               log=lambda m: counts.append(int(m.split("sensitivity:")[1].split("detected")[0].replace(",", ""))))
    n_s_cpu = counts[-1]

    log("1. GPU kernel source compiled with njit (single CPU thread), against fastmc.py")
    tw = _twin_trues(ring, grid, ph, mu, n_emit_cpu, 11)
    log(f"  {len(tw['dt']):,} trues from {tw['n_emitted']:,} emissions (acceptance {len(tw['dt']) / tw['n_emitted']:.5f})")
    check("acceptance, |z|", abs(_z_eff(len(tw["dt"]), tw["n_emitted"], n_cpu, n_emit_cpu)), 4)
    compare_events(tw, ref, ring, check)
    s_tw, n_tw, _ = _twin_sens(ring, grid, mu, 40_000_000, 12)
    log(f"  sensitivity: {n_tw:,} detected (CPU reference {n_s_cpu:,})")
    _sens_compare(s_tw, n_tw, s_cpu, n_s_cpu, check, "sensitivity")
    q, tcdf, tgrid, edges, n_t, ref_t = _timed_reference(config, ring, grid, ph, mu)
    log(f"  timed emission (15O-water, myocardium, stress): CPU {len(ref_t['dt']):,} trues from {n_t:,} decays")
    tw_t = _twin_trues(ring, grid, q, mu, n_t, 13, tcdf=tcdf, tgrid=tgrid, cap_frac=0.1)
    _timed_compare(tw_t, n_t, ref_t, n_t, edges, check, "timed emission")

    if not FG.cuda.is_available():
        log("2. no CUDA device and no simulator: GPU run skipped")
        return ok
    import projector_cuda as PG
    log(f"2. CUDA build on {PG.device_name()}, against fastmc.py")
    st = {}
    ev = FG.generate_trues(ring, grid, ph, mu, 2_000_000, 5, log=lambda m: None, stats=st)
    log(f"  {st['trues']:,} trues from {st['emissions']:,} emissions in {st['seconds']:.1f} s including compile "
        f"(acceptance {st['trues'] / st['emissions']:.5f})")
    check("acceptance, |z|", abs(_z_eff(st["trues"], st["emissions"], n_cpu, n_emit_cpu)), 4)
    compare_events(ev, ref, ring, check)
    ev2 = FG.generate_trues(ring, grid, ph, mu, 2_000_000, 5, log=lambda m: None)
    check("same seed gives the same events (0 = identical)", float(not all(np.array_equal(ev[k], ev2[k]) for k in ("p1", "p2", "dt", "e"))), 0)
    ss = {}
    s_g = FG.sensitivity(ring, grid, mu, 20_000_000, 6, log=lambda m: None, stats=ss)
    log(f"  sensitivity: {ss['detected']:,} detected from {ss['emissions']:,} emissions in {ss['seconds']:.1f} s")
    _sens_compare(s_g, ss["detected"], s_cpu, n_s_cpu, check, "sensitivity")
    st = {}
    ev_t = FG.emit_timed(ring, grid, q, mu, 5 * n_t, tcdf, tgrid, 7, log=lambda m: None, stats=st)
    log(f"  timed emission: {st['trues']:,} trues from {st['emissions']:,} decays in {st['seconds']:.1f} s")
    check("timed emission: decays simulated, difference from request", abs(st["emissions"] - 5 * n_t), 0)
    _timed_compare(ev_t, 5 * n_t, ref_t, n_t, edges, check, "timed emission")
    return ok


# ---------------------------------------------------------------- run A
def run(args):
    import fastmc_cuda as FG
    recon = Path(args.recon)
    out = recon / "bench"
    out.mkdir(parents=True, exist_ok=True)
    ring = G.Ring(args.config)
    grid = G.recon_grid(ring, 3.0)
    ph = G.load_body(args.body)
    mu = G.mu_map(grid, ph)
    devices = ["cpu", "gpu"] if args.device == "both" else [args.device]
    res = {"benchmark": "A, fast Monte Carlo", "config": args.config, "body": args.body, "seed": args.seed,
           "env": env_info("gpu" in devices)}
    log(f"[{args.config} {args.body}] devices {devices}, {res['env']['numba_threads']} CPU threads, {res['env']['machine']}")
    if "cpu" in devices:
        PC.fwd(mu, grid, np.zeros((4, 3), np.float32), np.ones((4, 3), np.float32))        # JIT warm-up
        t0 = time.perf_counter()
        ev = fastmc.generate_trues(ring, grid, ph, mu, int(args.cpu_trues), np.random.default_rng(args.seed), log=lambda m: None)
        dt = time.perf_counter() - t0
        res["cpu_trues"] = {"trues": int(args.cpu_trues), "emissions": int(ev["n_emitted"]), "seconds": dt,
                            "emissions_per_s": float(ev["n_emitted"] / dt), "acceptance": float(args.cpu_trues / ev["n_emitted"])}
        log(f"  CPU trues: {res['cpu_trues']['emissions_per_s'] / 1e6:.2f} M decays/s ({dt:.1f} s for {int(args.cpu_trues):,} trues)")
        counts = []
        t0 = time.perf_counter()
        fastmc.sensitivity(ring, grid, mu, int(args.cpu_sens), np.random.default_rng(args.seed + 1),
                           log=lambda m: counts.append(m))
        dt = time.perf_counter() - t0
        n_det = int(counts[-1].split("sensitivity:")[1].split("detected")[0].replace(",", ""))
        res["cpu_sens"] = {"detected": n_det, "seconds": dt, "detected_per_s": n_det / dt}
        log(f"  CPU sensitivity: {n_det / dt / 1e6:.3f} M detected events/s ({dt:.1f} s)")
    if "gpu" in devices:
        if not FG.cuda.is_available():
            raise SystemExit("no CUDA device found")
        t0 = time.perf_counter()
        FG.generate_trues(ring, grid, ph, mu, 20_000, args.seed, log=lambda m: None)       # compile + context
        FG.sensitivity(ring, grid, mu, 20_000, args.seed, log=lambda m: None)
        res["gpu_warmup_s"] = time.perf_counter() - t0
        runs = []
        for _ in range(args.repeats):
            st = {}
            ev = FG.generate_trues(ring, grid, ph, mu, int(args.gpu_trues), args.seed, log=lambda m: None, stats=st)
            runs.append(st)
        sec = float(np.median([r["seconds"] for r in runs]))
        st = runs[-1]
        res["gpu_trues"] = {"trues": st["trues"], "emissions": st["emissions"], "seconds_median": sec,
                            "seconds_runs": [r["seconds"] for r in runs], "emissions_per_s": st["emissions"] / sec,
                            "acceptance": st["trues"] / st["emissions"], "threads": st["threads"], "rounds": st["rounds"]}
        log(f"  GPU trues: {res['gpu_trues']['emissions_per_s'] / 1e6:.1f} M decays/s (median {sec:.2f} s for "
            f"{int(args.gpu_trues):,} trues), warm-up {res['gpu_warmup_s']:.1f} s")
        ss = {}
        sens = FG.sensitivity(ring, grid, mu, int(args.gpu_sens), args.seed + 1, log=lambda m: None, stats=ss)
        res["gpu_sens"] = {"detected": ss["detected"], "seconds": ss["seconds"], "detected_per_s": ss["detected"] / ss["seconds"]}
        log(f"  GPU sensitivity: {res['gpu_sens']['detected_per_s'] / 1e6:.2f} M detected events/s ({ss['seconds']:.2f} s)")
        if args.write_inputs:
            tag = f"{args.config}_{args.body}"
            np.savez(recon / f"events_fastmcgpu_{tag}_{int(args.gpu_trues)}.npz", **ev)
            np.save(recon / f"sens_gpu_{tag}_v3.npy", sens)
    if len(devices) == 2:
        c, g = res["cpu_trues"], res["gpu_trues"]
        z = _z_eff(g["trues"], g["emissions"], c["trues"], c["emissions"])
        res["agreement"] = {"acceptance_z": float(z), "speedup_trues": g["emissions_per_s"] / c["emissions_per_s"],
                            "speedup_sens": res["gpu_sens"]["detected_per_s"] / res["cpu_sens"]["detected_per_s"]}
        log(f"  GPU vs CPU: trues {res['agreement']['speedup_trues']:.0f}x, sensitivity {res['agreement']['speedup_sens']:.0f}x, "
            f"acceptance z = {z:+.2f}")
    name = f"bench_A_{args.config}_{args.body}_{socket.gethostname().split('.')[0]}_{time.strftime('%Y%m%d_%H%M%S')}.json"
    (out / name).write_text(json.dumps(res, indent=2))
    log(f"wrote {out / name}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--recon")
    ap.add_argument("--config", default="R280_slot")
    ap.add_argument("--body", default="P90")
    ap.add_argument("--device", default="both", choices=["cpu", "gpu", "both"])
    ap.add_argument("--cpu-trues", type=float, default=1e6)
    ap.add_argument("--gpu-trues", type=float, default=1e7)
    ap.add_argument("--cpu-sens", type=float, default=2e6)
    ap.add_argument("--gpu-sens", type=float, default=2e7)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--write-inputs", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        good = selftest(args.config, args.body)
        log("SELFTEST PASSED" if good else "SELFTEST FAILED")
        sys.exit(0 if good else 1)
    if not args.recon:
        raise SystemExit("give --recon DIR, or --selftest")
    run(args)


if __name__ == "__main__":
    main()

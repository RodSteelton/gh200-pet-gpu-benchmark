"""Run F: one end-to-end 15O-water scan on the GPU (simulate, reconstruct 23 frames, fit MBF).

Uses dynamic.py unchanged: only the emission loop (fastmc_cuda.emit_timed) and OSEM (projector_cuda.osem)
are swapped. Run from exercise_pet_sim/recon:
    python dynamic_gpu.py --recon DIR --state rest --real 0
    python dynamic_gpu.py --recon DIR --state both --real 0 --scale 0.01      (smoke test, 1% of the activity)
Old reference (56 cm ring R280_slot, seed 51, scan 0, 32 Rivanna cores, 22 to 29 min wall):
    rest   normal MBF 1.014, 137.1 M trues;  stress normal 3.122, defect 1.591, 112.5 M trues.
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np

import dynamic as D
import geometry as G
import backrest as B
import fastmc_cuda as FC
import projector_cuda as PC

OLD = {"rest": dict(normal=1.014, trues=137.1e6), "stress": dict(normal=3.122, defect=1.591, trues=112.5e6)}
T = {"osem": 0.0}


def _osem(ev, grid, sens, sigma, iters, subsets, rng):
    t0 = time.perf_counter()
    img = PC.osem(ev, grid, sens, sigma, iters, subsets, rng, log=lambda m: None)
    T["osem"] += time.perf_counter() - t0
    return img


D.osem = _osem                                        # recon_fit looks up osem in dynamic's namespace


def get_sens(recon_dir, config, ring, grid, mu, seed):
    f = recon_dir / f"sens_{config}_{D.BODY}_backrest_carbon_v3.npy"
    g = recon_dir / f"sens_gpu_{config}_{D.BODY}_backrest_carbon.npy"
    if f.exists():
        return np.load(f), 0.0, "old file"
    if g.exists():
        return np.load(g), 0.0, "cached GPU file"
    t0 = time.perf_counter()
    s = FC.sensitivity(ring, grid, mu, 20e6, seed, log=lambda m: None)
    dt = time.perf_counter() - t0
    np.save(g, s)
    return s, dt, "generated on GPU"


def simulate_gpu(recon_dir, config, state, real, mbq, seed, scale):
    ring = G.Ring(config)
    grid = G.recon_grid(ring, 3.0)
    ph = G.load_body(D.BODY)
    masks = D.class_masks(ph)
    mi = [m[0] for m in B.MATERIALS].index("carbon")
    mu = B.mu_with_backrest(grid, ph, mi)
    sens, t_sens, how = get_sens(recon_dir, config, ring, grid, mu, seed + 99)
    print(f"  sensitivity: {how}, {t_sens:.1f} s", flush=True)
    tacs = D.class_tacs(state, mbq)
    rng = np.random.default_rng(seed + 1000 * real + 17 * D.STATES.index(state))
    parts, n_tot = [], 0
    t0 = time.perf_counter()
    for name in D.CLASSES:
        q, vol = D.class_phantom(ph, masks, name)
        dens = tacs[name] * D.DECAY
        n_mean = float(dens.sum() * D.DT * vol * 1000.0) * scale
        n_emit = int(rng.poisson(n_mean)) if n_mean < 1e18 else int(n_mean)
        cdf = np.cumsum(dens)
        cdf = cdf / cdf[-1] if cdf[-1] > 0 else np.linspace(0, 1, len(cdf))
        s = int(rng.integers(1 << 31))
        if n_emit:
            parts.append(FC.emit_timed(ring, grid, q, mu, n_emit, cdf, D.TGRID, s, log=lambda m: None))
        n_tot += n_emit
        print(f"  {name:12s} {vol:8.0f} mL  {n_emit:.3e} decays", flush=True)
    ev = {k: np.concatenate([p[k] for p in parts]) for k in ("p1", "p2", "dt", "t")}
    t_sim = time.perf_counter() - t0
    print(f"  {n_tot:.3e} decays -> {len(ev['t']):,} trues in {t_sim:.1f} s", flush=True)
    return grid, sens, ev, n_tot, t_sim, t_sens


def run(recon_dir, out, config, state, real, mbq, ctr, iters, subsets, seed, scale):
    t_all = time.perf_counter()
    print(f"[{config} {state} r{real}] GPU {PC.device_name()}", flush=True)
    grid, sens, ev, n_tot, t_sim, t_sens = simulate_gpu(recon_dir, config, state, real, mbq, seed, scale)
    rng_r = np.random.default_rng(seed + 5000 + 1000 * real + 17 * D.STATES.index(state))
    T["osem"] = 0.0
    t0 = time.perf_counter()
    blood, segs, fits, counts, shot = D.recon_fit(config, grid, sens, ev, ctr, iters, subsets, rng_r)
    t_rf = time.perf_counter() - t0
    t_total = time.perf_counter() - t_all
    mn, md = D.TRUE_MBF[state]
    mbf = [f["mbf"] for f in fits]
    normal = [m for i, m in enumerate(mbf) if i != D.DEF_SEG]
    res = dict(config=config, state=state, real=real, scale=scale, n_decays=int(n_tot), n_trues=int(len(ev["t"])),
               mbf_normal_mean=float(np.mean(normal)), mbf_normal_sd=float(np.std(normal)),
               mbf_defect=float(mbf[D.DEF_SEG]), true_normal=mn, true_defect=md,
               seconds=dict(sensitivity=t_sens, simulate=t_sim, osem_all_frames=T["osem"],
                            recon_fit_total=t_rf, fit_and_other=t_rf - T["osem"], total=t_total),
               gpu=PC.device_name(), old_reference=OLD[state])
    tag = f"gpu_{config}_{state}_r{real}" + ("" if scale == 1.0 else f"_scale{scale:g}")
    (out / f"{tag}.json").write_text(json.dumps(res, indent=2))
    np.savez_compressed(out / f"{tag}.npz", blood=blood, segs=segs, counts=np.array(counts))
    s = res["seconds"]
    print(f"  MBF normal {res['mbf_normal_mean']:.3f} +/- {res['mbf_normal_sd']:.3f} (true {mn}), "
          f"defect {res['mbf_defect']:.3f} (true {md}); old normal {OLD[state]['normal']}", flush=True)
    print(f"  TIMES s: sens {s['sensitivity']:.1f}  simulate {s['simulate']:.1f}  OSEM {s['osem_all_frames']:.1f}  "
          f"fit+other {s['fit_and_other']:.1f}  TOTAL {s['total']:.1f}", flush=True)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--recon", required=True)
    ap.add_argument("--config", default="R280_slot")
    ap.add_argument("--state", default="both", choices=["rest", "stress", "both"])
    ap.add_argument("--real", type=int, default=0)
    ap.add_argument("--mbq", type=float, default=400.0)
    ap.add_argument("--ctr", type=int, default=200)
    ap.add_argument("--iters", type=int, default=4)
    ap.add_argument("--subsets", type=int, default=10)
    ap.add_argument("--seed", type=int, default=51)
    ap.add_argument("--scale", type=float, default=1.0, help="fraction of the activity (smoke test)")
    ap.add_argument("--outdir", default="dynamic_gpu")
    a = ap.parse_args()
    recon_dir = Path(a.recon)
    out = recon_dir / a.outdir
    out.mkdir(exist_ok=True)
    for st in (["rest", "stress"] if a.state == "both" else [a.state]):
        run(recon_dir, out, a.config, st, a.real, a.mbq, a.ctr, a.iters, a.subsets, a.seed, a.scale)


if __name__ == "__main__":
    main()

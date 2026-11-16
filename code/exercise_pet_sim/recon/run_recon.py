"""
TOF list-mode OSEM reconstruction of trues for one (geometry, body, data source).

    python run_recon.py --config R280_slot --body P90 --data fastmc --out /scratch/$USER/exercise_pet_recon
    python run_recon.py --config R280_slot --body P90 --data gate \
        --gate-dir /scratch/$USER/exercise_pet_sweep2 --out /scratch/$USER/exercise_pet_recon

Trues only (perfect scatter and randoms correction assumed); endpoints snapped to crystal centres
(no depth of interaction, so parallax is included); attenuation modelled in the sensitivity image.
Writes img_<config>_<body>_<data>_ctr<ps>.npy (+ .json) for each CTR in --ctrs (0 = non-TOF).
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

import geometry as G
import fastmc
from projector import fwd, bwd

C = G.C


def log(msg):
    print(msg, flush=True)


def load_gate(gate_dir, config, body, ring):
    sys.path.insert(0, str(G.ROOT))
    from analyze import load_gap, lor_geometry

    d, n_dec, metas = load_gap(Path(gate_dir) / f"{config}_{body}")
    geo = lor_geometry(d, C.BODIES[body]["outer"])
    true = geo["dist"] < C.SCATTER_DIST_MM
    q1 = np.stack([d["x1"], d["y1"], d["z1"]], 1)[true].astype(np.float64)
    q2 = np.stack([d["x2"], d["y2"], d["z2"]], 1)[true].astype(np.float64)
    s1, k1 = ring.snap(q1)
    s2, k2 = ring.snap(q2)
    ok = k1 & k2
    dt = ((d["t1"] - d["t2"]) * 1000.0)[true]          # ns -> ps
    e = np.stack([d["ex"], d["ey"], d["ez"]], 1)[true].astype(np.float32)
    log(f"  gate: {int(true.sum()):,} trues of {len(true):,} prompts ({n_dec:,} decays); "
        f"{int((~ok).sum())} dropped (no kept module)")
    return {"p1": s1[ok], "p2": s2[ok], "dt": dt[ok].astype(np.float32), "e": e[ok], "n_emitted": n_dec}


def osem(ev, grid, sens, sigma, iters, subsets, rng, snapshots=()):
    """List-mode OSEM. If snapshots is given, returns {iteration: image} for those iterations."""
    p1, p2, tc = ev["p1"], ev["p2"], ev["tc"]
    snaps = {}
    mask = sens > 1e-3 * sens.max()
    img = mask.astype(np.float32)
    perm = rng.permutation(len(p1))
    for it in range(iters):
        t0 = time.time()
        for s in range(subsets):
            idx = perm[s::subsets]
            fp = fwd(img, grid, p1[idx], p2[idx], tc[idx], sigma)
            ratio = np.where(fp > 1e-12, 1.0 / np.maximum(fp, 1e-12), 0.0).astype(np.float32)
            bp = bwd(ratio, grid, p1[idx], p2[idx], tc[idx], sigma)
            with np.errstate(divide="ignore", invalid="ignore"):
                img = np.where(mask, img * bp / (sens / subsets), 0.0).astype(np.float32)
        log(f"    iteration {it + 1}/{iters}: {time.time() - t0:.1f} s")
        if it + 1 in snapshots:
            snaps[it + 1] = img.copy()
    return snaps if snapshots else img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, choices=[c[0] for c in C.SWEEP2_CONFIGS])
    ap.add_argument("--body", required=True, choices=list(C.BODIES))
    ap.add_argument("--data", required=True, choices=["fastmc", "gate"])
    ap.add_argument("--gate-dir", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--ctrs", default=",".join(str(c) for c in C.RECON_CTRS))
    ap.add_argument("--iters", type=int, default=C.RECON_ITERS)
    ap.add_argument("--subsets", type=int, default=C.RECON_SUBSETS)
    ap.add_argument("--n-trues", type=float, default=C.FASTMC_TRUES)
    ap.add_argument("--sens-events", type=float, default=C.SENS_DETECTED)
    ap.add_argument("--voxel", type=float, default=C.RECON_VOXEL_MM)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    ring = G.Ring(args.config)
    grid = G.recon_grid(ring, args.voxel)
    ph = G.load_body(args.body)
    mu = G.mu_map(grid, ph)
    tag = f"{args.config}_{args.body}"
    log(f"[{tag}] grid {grid.shape} at {grid.v} mm, {len(ring.kept)} modules")

    vtag = f"v{args.voxel:g}"
    sens_path = out / f"sens_{tag}_{vtag}.npy"
    if sens_path.exists():
        sens = np.load(sens_path)
    else:
        t0 = time.time()
        sens = fastmc.sensitivity(ring, grid, mu, int(args.sens_events), rng, log=log)
        np.save(sens_path, sens)
        log(f"  sensitivity done in {time.time() - t0:.0f} s")

    t0 = time.time()
    if args.data == "fastmc":
        ev_path = out / f"events_fastmc_{tag}_{int(args.n_trues)}.npz"
        if ev_path.exists():
            ev = dict(np.load(ev_path))
        else:
            ev = fastmc.generate_trues(ring, grid, ph, mu, int(args.n_trues), rng, log=log)
            np.savez(ev_path, **ev)
    else:
        if not args.gate_dir:
            raise SystemExit("--gate-dir is required for --data gate")
        ev = load_gate(args.gate_dir, args.config, args.body, ring)
    log(f"  {len(ev['p1']):,} events ready in {time.time() - t0:.0f} s")

    D = np.linalg.norm(ev["p2"] - ev["p1"], axis=1)
    for ctr in [int(c) for c in args.ctrs.split(",")]:
        sig_t = ctr / 2.3548 if ctr > 0 else 0.0
        sigma = C.C_MM_PER_PS * sig_t / 2.0
        dt = ev["dt"] + (rng.normal(size=len(D)) * sig_t if ctr > 0 else 0.0)
        ev["tc"] = ((D + C.C_MM_PER_PS * dt) / 2.0).astype(np.float32)
        log(f"  OSEM {args.iters}x{args.subsets}, CTR {ctr or 'non-TOF'} ps (sigma {sigma:.1f} mm)")
        t0 = time.time()
        img = osem(ev, grid, sens, sigma, args.iters, args.subsets, rng)
        name = f"img_{tag}_{args.data}_ctr{ctr}_{vtag}"
        np.save(out / f"{name}.npy", img)
        meta = {"config": args.config, "body": args.body, "data": args.data, "ctr_ps": ctr,
                "sigma_mm": sigma, "iters": args.iters, "subsets": args.subsets, "voxel_mm": grid.v,
                "grid": [grid.nx, grid.ny, grid.nz], "n_events": int(len(D)),
                "n_emitted": int(ev.get("n_emitted", 0)), "seconds": time.time() - t0}
        (out / f"{name}.json").write_text(json.dumps(meta, indent=2))
        log(f"  saved {name} ({time.time() - t0:.0f} s)")


if __name__ == "__main__":
    main()

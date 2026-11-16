"""
Stage 7: count rates for 15O-water at rest and exercise stress: singles, randoms and dead time,
and the dose that maximises noise-equivalent counts (NEC).

    python countrate.py --recon /scratch/$USER/exercise_pet_recon --task N     one geometry (N = 0, 1)
    python countrate.py --recon /scratch/$USER/exercise_pet_recon --collect    figure and tables

Task: per-decay detector response for each activity class, from the fast Monte Carlo on an extended
attenuation map. Classes are the stage-5 phantom classes (P90 body, carbon backrest) plus two regions
outside the phantom that see the ring through its ends: abdomen/pelvis (below) and neck/head (above).
For every class it records, per emitted decay,
    singles per readout block (7 x 8 crystals; each photon counted on its own, with its own attenuation),
    trues per block pair (both photons detected, attenuation along the full line of response).
Collect: activity of every class over time from the stage-5 kinetic model, scaled to the injected dose,
then per time point
    live fraction per block      l_b = exp(-S_b tau_block)            (paralysable block dead time)
    trues                         T  = sum over block pairs t_ij l_i l_j
    scatter (assumed fraction)    Sc = T SF / (1 - SF)
    randoms                       R  = 2 tau_w sum over module pairs in coincidence S'_m S'_n
    NEC                           = T^2 / (T + Sc + R)        (randoms from singles, k = 1)
integrated over the 240 s scan and over the first 60 s, for doses of 100 to 2000 MBq.
The stage-5 MBF precision (trues only) scales as sqrt(trues / NEC).
Assumptions (set below, varied in the summary): block dead time, coincidence window, scatter fraction,
extra singles from body scatter, Lu-176 background; a variant adds end shields.
Not modelled: TOF gain for randoms, scatter and randoms spatial distribution, energy spectrum of singles.
"""

import argparse
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

import geometry as G
import fastmc
import backrest as B
import dynamic as D
from projector import Grid, fwd
from run_recon import log

C = G.C
BODY = "P90"
CONFIGS = list(C.RECON_CONFIGS)
NBA = C.BLOCKS_AXIAL                                   # 9 axial blocks per module
BLOCK_Z = C.CRYSTALS_PER_BLOCK_AXIAL * C.CRYSTAL_PITCH
Z_EXT = 450.0                                          # mm, half length of the extended attenuation map
PH_Z = 148.5                                           # mm, half length of the voxel phantom
EXTRA = {                                              # regions outside the phantom (uniform, soft tissue)
    "abdomen": dict(z0=-Z_EXT + 1.5, z1=-PH_Z, a=220.0, b=160.0, flow=(0.30, 0.20), p=0.80, vb=0.05),
    "head_neck": dict(z0=PH_Z, z1=Z_EXT - 1.5, a=80.0, b=95.0, flow=(0.45, 0.45), p=0.90, vb=0.04),
}
CLASSES = list(D.CLASSES) + list(EXTRA)
WORKERS = int(os.environ.get("XPET_WORKERS", 30))
CHUNK = 2_000_000
SHIELDS_MM = (20.0, 40.0, 60.0, 80.0)                   # radial depths of the end shields (variants)

# assumptions (base case) and alternatives shown in the summary
BASE = dict(tau_block_ns=200.0, window_ns={"R280_slot": 3.0, "R350_closed": 3.6}, sf=0.35,
            singles_scatter=1.4, lyso_cps_per_block=1000.0)
DOSES = [100, 200, 300, 400, 500, 600, 800, 1000, 1200, 1500, 2000, 2500, 3000]
DT_COLLECT = 1.0                                        # s


# ---- extended attenuation map -----------------------------------------------------------------------
_CACHE = {}


def big_setup(config):
    if config in _CACHE:
        return _CACHE[config]
    ring = G.Ring(config)
    g0 = G.recon_grid(ring, 3.0)
    grid = Grid(g0.nx, g0.ny, 2 * int(np.ceil(Z_EXT / 3.0)), 3.0)
    ph = G.load_body(BODY)
    mi = [m[0] for m in B.MATERIALS].index("carbon")
    mu = B.mu_with_backrest(grid, ph, mi)
    gx, gy, gz = grid.axes()
    Y, X = np.meshgrid(gy, gx, indexing="ij")
    for r in EXTRA.values():
        inside = (X / r["a"]) ** 2 + (Y / r["b"]) ** 2 <= 1.0
        zs = (gz >= r["z0"]) & (gz <= r["z1"])
        sl = mu[zs]
        sl[:, inside] = np.where(sl[:, inside] == 0, C.MU_PER_MM["soft_tissue"], sl[:, inside])
        mu[zs] = sl
    masks = D.class_masks(ph)
    _CACHE[config] = (ring, grid, mu, ph, masks)
    return _CACHE[config]


def sample(name, n, rng, ph, masks):
    if name in EXTRA:
        r = EXTRA[name]
        rr = np.sqrt(rng.random(n))
        phi = rng.uniform(0, 2 * np.pi, n)
        return np.stack([r["a"] * rr * np.cos(phi), r["b"] * rr * np.sin(phi), rng.uniform(r["z0"], r["z1"], n)], 1)
    q, _ = D.class_phantom(ph, masks, name)
    return fastmc.sample_emissions(q, n, rng)


def class_volume_ml(name, ph, masks):
    if name in EXTRA:
        r = EXTRA[name]
        return np.pi * r["a"] * r["b"] * (r["z1"] - r["z0"]) / 1000.0
    return D.class_phantom(ph, masks, name)[1]


def block_of(ring, q):
    m, _ = ring.module_of(q[:, 0], q[:, 1])
    az = np.clip(np.floor((q[:, 2] + G.ZMAX) / BLOCK_Z), 0, NBA - 1).astype(np.int64)
    return m.astype(np.int64) * NBA + az


def _work(args):
    config, name, n, seed = args
    ring, grid, mu, ph, masks = big_setup(config)
    nb = ring.n_full * NBA
    rng = np.random.default_rng(seed)
    e = sample(name, n, rng, ph, masks)
    d = fastmc._iso(rng, len(e))
    det, qs = [], []
    for sgn in (1.0, -1.0):
        q, ok = ring.detect(e, sgn * d, rng)
        idx = np.nonzero(ok)[0]
        if len(idx):
            att = np.exp(-fwd(mu, grid, e[idx].astype(np.float32), q[idx].astype(np.float32)))
            ok[idx] = rng.random(len(idx)) < att
        det.append(ok)
        qs.append(q)
    singles = np.zeros(nb, np.int64)
    singles_sh = np.zeros((len(SHIELDS_MM), nb), np.int64)
    for sgn, q, ok in zip((1.0, -1.0), qs, det):
        singles += np.bincount(block_of(ring, q[ok]), minlength=nb)
        # end shields: annulus in the planes z = +-ZMAX from R - depth to R blocks photons from outside the FOV
        rxy = np.full(len(e), -1.0)
        out_z = ok & (np.abs(e[:, 2]) > G.ZMAX)
        if out_z.any():
            ee, dd = e[out_z], sgn * d[out_z]
            zp = np.sign(ee[:, 2]) * G.ZMAX
            with np.errstate(divide="ignore", invalid="ignore"):
                tt = (zp - ee[:, 2]) / dd[:, 2]
            xy = ee[:, :2] + tt[:, None] * dd[:, :2]
            rxy[out_z] = np.where(tt > 0, np.hypot(xy[:, 0], xy[:, 1]), -1.0)
        for k, dep in enumerate(SHIELDS_MM):
            singles_sh[k] += np.bincount(block_of(ring, q[ok & ~(rxy > ring.R - dep)]), minlength=nb)
    both = det[0] & det[1]
    b1, b2 = block_of(ring, qs[0][both]), block_of(ring, qs[1][both])
    lo, hi = np.minimum(b1, b2), np.maximum(b1, b2)
    pairs = np.bincount(lo * nb + hi, minlength=nb * nb)
    return singles, singles_sh, pairs, len(e)


def run_task(recon_dir, out, config, n_per_class, seed):
    ring, grid, mu, ph, masks = big_setup(config)
    nb = ring.n_full * NBA
    res = {}
    t0 = time.time()
    for ci, name in enumerate(CLASSES):
        k = int(np.ceil(n_per_class / CHUNK))
        jobs = [(config, name, min(CHUNK, n_per_class - i * CHUNK), seed + 1000 * ci + i) for i in range(k)]
        with ProcessPoolExecutor(max_workers=min(WORKERS, len(jobs))) as ex:
            parts = list(ex.map(_work, jobs))
        s = sum(p[0] for p in parts)
        ssh = sum(p[1] for p in parts)
        pr = sum(p[2] for p in parts)
        n = sum(p[3] for p in parts)
        nz = np.nonzero(pr)[0]
        res[f"{name}__singles"] = s / n
        for k, dep in enumerate(SHIELDS_MM):
            res[f"{name}__singles_sh{int(dep)}"] = ssh[k] / n
        res[f"{name}__pair_idx"] = nz.astype(np.int64)
        res[f"{name}__pair_w"] = pr[nz] / n
        res[f"{name}__vol_ml"] = np.float64(class_volume_ml(name, ph, masks))
        log(f"[{config}] {name:12s} {n:.2e} decays: singles/decay {s.sum() / n:.4f}, trues/decay {pr.sum() / n:.5f} "
            f"({time.time() - t0:.0f} s)")
    res["n_blocks"] = np.int64(nb)
    res["kept_modules"] = ring.kept_mask.astype(np.int8)
    np.savez_compressed(out / f"tables_{config}.npz", **res)
    log(f"wrote tables_{config}.npz")


# ---- collect ----------------------------------------------------------------------------------------
def class_activity(state):
    """Bq of every class on the collect time grid for a 1 MBq injection (decayed)."""
    tac = D.class_tacs(state, 1.0)                               # decay-corrected kBq/mL on D.TGRID
    ca = D.arterial(state, 1.0)
    si = D.STATES.index(state)
    for name, r in EXTRA.items():
        tac[name] = D.one_tissue(ca, r["flow"][si], r["p"]) + r["vb"] * ca
    decay = np.exp(-D.LAM * D.TGRID)
    tg = np.arange(DT_COLLECT / 2, D.EDGES[-1], DT_COLLECT)
    return tg, {k: np.interp(tg, D.TGRID, v * decay) for k, v in tac.items()}


def rates(tab, tg, act, dose, par, config):
    """Count rates (cps) on tg for one geometry, state and dose."""
    nb = int(tab["n_blocks"])
    tau_b = par["tau_block_ns"] * 1e-9
    win = par["window_ns"][config] * 1e-9
    nm = nb // NBA
    w_pairs = {}
    idx_all = np.unique(np.concatenate([tab[f"{c}__pair_idx"] for c in CLASSES]))
    pos = {c: np.searchsorted(idx_all, tab[f"{c}__pair_idx"]) for c in CLASSES}
    b1, b2 = idx_all // nb, idx_all % nb
    m1, m2 = b1 // NBA, b2 // NBA
    mm = np.zeros((nm, nm), bool)
    mm[m1, m2] = True
    mm[m2, m1] = True
    mm &= ~np.eye(nm, dtype=bool)
    out = {k: np.zeros(len(tg)) for k in ("singles", "trues", "trues_ideal", "randoms", "scatter", "nec",
                                          "live_min", "live_mean")}
    kept = np.repeat(tab["kept_modules"].astype(bool), NBA)
    for it in range(len(tg)):
        a = {c: act[c][it] * dose * 1e3 * float(tab[f"{c}__vol_ml"]) for c in CLASSES}   # Bq
        key = f"__singles_sh{int(par['shields'])}" if par.get("shields") else "__singles"
        s = sum(a[c] * tab[f"{c}{key}"] for c in CLASSES) * par["singles_scatter"]
        s = s + kept * par["lyso_cps_per_block"]
        w = np.zeros(len(idx_all))
        for c in CLASSES:
            w[pos[c]] += a[c] * tab[f"{c}__pair_w"]
        live = np.exp(-s * tau_b)
        t = float((w * live[b1] * live[b2]).sum())
        sm = (s * live).reshape(nm, NBA).sum(1)
        r = win * float(sm @ mm @ sm) / 2.0
        sc = t * par["sf"] / (1 - par["sf"])
        out["singles"][it] = float((s * live).sum())
        out["trues"][it] = t
        out["trues_ideal"][it] = float(w.sum())
        out["randoms"][it] = r
        out["scatter"][it] = sc
        out["nec"][it] = t * t / max(t + sc + r, 1e-9)
        out["live_min"][it] = float(live[kept].min())
        out["live_mean"][it] = float((live * s)[kept].sum() / max(s[kept].sum(), 1e-9))
    return out


def collect(out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    tabs = {c: dict(np.load(out / f"tables_{c}.npz")) for c in CONFIGS if (out / f"tables_{c}.npz").exists()}
    if not tabs:
        raise SystemExit("no tables found")
    configs = list(tabs)
    variants = {"base": BASE,
                "dead time 100 ns": {**BASE, "tau_block_ns": 100.0},
                "dead time 400 ns": {**BASE, "tau_block_ns": 400.0},
                "window 4.5 ns both": {**BASE, "window_ns": {c: 4.5 for c in CONFIGS}},
                "no body-scatter singles": {**BASE, "singles_scatter": 1.0},
                "body-scatter singles x1.8": {**BASE, "singles_scatter": 1.8},
                **{f"end shields {int(dp)} mm": {**BASE, "shields": dp} for dp in SHIELDS_MM}}
    rows = []
    curves = {}
    for state in D.STATES:
        tg, act = class_activity(state)
        first = tg < 60.0
        for c in configs:
            for vname, par in variants.items():
                for dose in (DOSES if vname == "base" else [400, 800]):
                    r = rates(tabs[c], tg, act, dose, par, c)
                    if vname == "base" and dose == 400:
                        curves[(c, state)] = (tg, r)
                    i = int(np.argmax(r["singles"]))
                    rows.append({"config": c, "state": state, "variant": vname, "dose_mbq": dose,
                                 "trues_ideal_M": r["trues_ideal"].sum() * DT_COLLECT / 1e6,
                                 "trues_M": r["trues"].sum() * DT_COLLECT / 1e6,
                                 "randoms_M": r["randoms"].sum() * DT_COLLECT / 1e6,
                                 "nec_M": r["nec"].sum() * DT_COLLECT / 1e6,
                                 "nec_first60_M": r["nec"][first].sum() * DT_COLLECT / 1e6,
                                 "trues_ideal_first60_M": r["trues_ideal"][first].sum() * DT_COLLECT / 1e6,
                                 "peak_singles_Mcps": r["singles"][i] / 1e6, "peak_randoms_Mcps": r["randoms"].max() / 1e6,
                                 "peak_trues_Mcps": r["trues"].max() / 1e6, "min_live": float(r["live_min"].min()),
                                 "r_over_t_at_peak": r["randoms"][i] / max(r["trues"][i], 1)})
    import csv
    with open(out / "countrate_metrics.csv", "w", newline="", encoding="utf-8") as fh:
        wr = csv.DictWriter(fh, fieldnames=list(rows[0]))
        wr.writeheader()
        wr.writerows(rows)

    # summary
    L = ["Stage 7 count rates, 15O-water, P90 body with carbon backrest, 240 s scan",
         f"base assumptions: block dead time {BASE['tau_block_ns']:.0f} ns (paralysable), window "
         + ", ".join(f"{c} {BASE['window_ns'][c]} ns" for c in configs)
         + f", scatter fraction {BASE['sf']}, body-scatter singles x{BASE['singles_scatter']}, Lu-176 "
         f"{BASE['lyso_cps_per_block']:.0f} cps/block"]
    L.append("config        state   dose  trues(ideal) trues   randoms  NEC    NEC/ideal  NEC 0-60s  peak singles  "
             "peak R/T  min live")
    for r in rows:
        if r["variant"] == "base":
            L.append(f"{r['config']:12s}  {r['state']:6s} {r['dose_mbq']:5d}  {r['trues_ideal_M']:9.1f} M "
                     f"{r['trues_M']:6.1f}  {r['randoms_M']:7.1f}  {r['nec_M']:6.1f}   {r['nec_M'] / r['trues_ideal_M']:.2f}"
                     f"     {r['nec_first60_M']:6.1f}     {r['peak_singles_Mcps']:7.1f} Mcps  {r['r_over_t_at_peak']:5.2f}"
                     f"     {r['min_live']:.3f}")
    L.append("")
    L.append("Dose giving the most NEC (base): ")
    for c in configs:
        for s in D.STATES:
            rr = [r for r in rows if r["config"] == c and r["state"] == s and r["variant"] == "base"]
            best = max(rr, key=lambda r: r["nec_M"])
            r4 = next(r for r in rr if r["dose_mbq"] == 400)
            L.append(f"  {c:12s} {s:6s}: max NEC at {best['dose_mbq']} MBq ({best['nec_M']:.1f} M); 400 MBq gives "
                     f"{r4['nec_M']:.1f} M = {r4['nec_M'] / best['nec_M']:.2f} of max; MBF SD vs stage 5 (trues only) "
                     f"x{np.sqrt(r4['trues_ideal_M'] / r4['nec_M']):.2f}")
    L.append("")
    L.append("Sensitivity at 400 and 800 MBq (NEC, M counts):")
    for c in configs:
        for s in D.STATES:
            L.append(f"  {c:12s} {s:6s}: " + "; ".join(
                f"{r['variant']} {r['nec_M']:.1f}" for r in rows
                if r["config"] == c and r["state"] == s and r["dose_mbq"] == 400))
            L.append(f"  {c:12s} {s:6s} at 800 MBq: " + "; ".join(
                f"{r['variant']} {r['nec_M']:.1f}" for r in rows
                if r["config"] == c and r["state"] == s and r["dose_mbq"] == 800))
    (out / "SUMMARY_countrate.txt").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))

    # figure
    col = {"R280_slot": "#2a78d6", "R350_closed": "#4a3aa7"}
    name = {"R280_slot": "56 cm, 290° open ring", "R350_closed": "70 cm closed ring"}
    fig, axs = plt.subplots(1, 3, figsize=(10, 3.2))
    for c in configs:
        for s, ls in (("rest", "--"), ("stress", "-")):
            rr = [r for r in rows if r["config"] == c and r["state"] == s and r["variant"] == "base"]
            axs[0].plot([r["dose_mbq"] for r in rr], [r["nec_M"] for r in rr], ls, color=col[c], lw=1.4,
                        label=f"{name[c]}, {s}")
            axs[0].plot([r["dose_mbq"] for r in rr], [r["trues_ideal_M"] for r in rr], ls, color=col[c], lw=0.6,
                        alpha=0.5)
    axs[0].set_xlabel("Injected 15O-water (MBq)")
    axs[0].set_ylabel("Counts in 240 s (millions)")
    axs[0].set_title("A.  NEC vs dose (thin: trues, no losses)", loc="left", fontweight="bold", fontsize=9)
    axs[0].legend(fontsize=6.5, frameon=False)
    for k, (ax, st) in enumerate(zip(axs[1:], D.STATES)):
        for c in configs:
            tg, r = curves[(c, st)]
            ax.plot(tg, r["trues"] / 1e6, color=col[c], lw=1.4)
            ax.plot(tg, r["randoms"] / 1e6, color=col[c], lw=1.0, ls="--")
            ax.plot(tg, r["nec"] / 1e6, color=col[c], lw=1.0, ls=":")
        ax.set_xlabel("Time after injection (s)")
        ax.set_ylabel("Rate at 400 MBq (Mcps)")
        ax.set_title(f"{'BC'[k]}.  {st.capitalize()}: trues (solid), randoms (dashed), NEC (dotted)", loc="left",
                     fontweight="bold", fontsize=8)
    for ax in axs:
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(out / f"Fig_countrate.{ext}", dpi=300)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--recon", required=True)
    ap.add_argument("--task", type=int, default=None)
    ap.add_argument("--collect", action="store_true")
    ap.add_argument("--ntasks", action="store_true")
    ap.add_argument("--n-per-class", type=float, default=2e8)
    ap.add_argument("--seed", type=int, default=71)
    ap.add_argument("--outdir", default="countrate")
    args = ap.parse_args()
    if args.ntasks:
        print(len(CONFIGS))
        return
    out = Path(args.recon) / args.outdir
    out.mkdir(parents=True, exist_ok=True)
    if args.collect:
        collect(out)
    elif args.task is not None:
        run_task(Path(args.recon), out, CONFIGS[args.task], int(args.n_per_class), args.seed)
    else:
        raise SystemExit("give --task N or --collect")


if __name__ == "__main__":
    main()

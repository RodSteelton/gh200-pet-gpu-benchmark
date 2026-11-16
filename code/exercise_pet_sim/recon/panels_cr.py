"""
Stage 8b count rates: singles, randoms, dead time and NEC for the stage-8b geometries (rings and panels),
with the stage-7 model (15O-water, P90 body, carbon backrest, abdomen and head/neck outside the phantom).

    python panels_cr.py --recon /scratch/$USER/exercise_pet_recon --task N      tables for one geometry
    python panels_cr.py --recon /scratch/$USER/exercise_pet_recon --collect     NEC table, figure, nec_factors.json
    python panels_cr.py --recon /scratch/$USER/exercise_pet_recon --smoke       2 geometries at 2e5 decays per class

Blocks: 7 crystals transaxial x 8 axial (as the ring modules); randoms between transaxial block columns
("modules") that record trues between them. Coincidence window: longest block-to-block distance over the
coincident pairs / c plus a margin set so that the 56 cm ring gets the stage-7 window (3.0 ns).
nec_factors.json: NEC / ideal trues integrated over the 240 s scan at 400 MBq (stress and rest); panels.py
thins the defect data by the stress factor to give NEC-equivalent counts.
Same caveats as stage 7: no TOF credit for randoms, fixed scatter fraction, no spatial distribution of
randoms and scatter; panels closer to the body pick up more singles, which this model includes.
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
import countrate as CR
import panels as P
from projector import Grid, fwd
from run_recon import log

C = G.C
WORKERS = int(os.environ.get("XPET_WORKERS", 30))
CHUNK = 2_000_000
BT, BA = 7, 8                                    # crystals per block, transaxial and axial
DOSE = 400.0
C_MM_PS = C.C_MM_PER_PS
_CACHE = {}


STAGE = os.environ.get("STAGE", "8b")


def configs():
    return P.STAGES[STAGE]["configs"]


class Blocks:
    """Block and module index for detected points, block centres and LYSO-background weights."""

    def __init__(self, scn):
        self.scn = scn
        if isinstance(scn, P.Panels):
            self.nbt = np.ceil(scn.nt / BT).astype(int)
            self.nba = int(np.ceil(scn.na / BA))
            self.moff = np.concatenate([[0], np.cumsum(self.nbt)[:-1]])
            self.nm = int(self.nbt.sum())
            self.nb = self.nm * self.nba
            cen, lyso = [], []
            for k in range(len(scn.d)):
                for tb in range(self.nbt[k]):
                    nt_here = min(BT, scn.nt[k] - tb * BT)
                    tc = (tb * BT + nt_here / 2.0 - scn.nt[k] / 2.0) * G.PITCH + scn.t0[k]
                    for ab in range(self.nba):
                        na_here = min(BA, scn.na - ab * BA)
                        zc = (ab * BA + na_here / 2.0 - scn.na / 2.0) * G.PITCH
                        xy = (scn.d[k] + scn.depth[k] / 2) * scn.n[k] + tc * scn.t[k]
                        cen.append([xy[0], xy[1], zc])
                        lyso.append(nt_here * na_here / float(BT * BA) * scn.depth[k] / 20.0)
            self.centre = np.array(cen)
            self.lyso = np.array(lyso)
            self.module = np.repeat(np.arange(self.nm), self.nba)
        else:
            self.nba = C.BLOCKS_AXIAL
            self.nm = scn.n_full
            self.nb = self.nm * self.nba
            ang = np.deg2rad(scn.offset + np.arange(self.nm) * scn.step)
            r = scn.R + G.DEPTH / 2
            zc = (np.arange(self.nba) + 0.5) * CR.BLOCK_Z - G.ZMAX
            self.centre = np.array([[r * np.cos(a), r * np.sin(a), z] for a in ang for z in zc])
            self.lyso = np.repeat(scn.kept_mask.astype(float), self.nba)
            self.module = np.repeat(np.arange(self.nm), self.nba)

    def block_of(self, q):
        scn = self.scn
        if not isinstance(scn, P.Panels):
            return CR.block_of(scn, q)
        radial = q[:, :2] @ scn.n.T
        tang = q[:, :2] @ scn.t.T - scn.t0
        inside = (radial >= scn.d - 1e-3) & (radial <= scn.d + scn.depth + 1e-3) & (np.abs(tang) <= scn.hw + 1e-3)
        k = np.argmax(inside, axis=1)
        i = np.arange(len(q))
        it = np.clip(np.floor(tang[i, k] / G.PITCH + scn.nt[k] / 2.0), 0, scn.nt[k] - 1).astype(int)
        ia = np.clip(np.floor(q[:, 2] / G.PITCH + scn.na / 2.0), 0, scn.na - 1).astype(int)
        return ((self.moff[k] + it // BT) * self.nba + ia // BA).astype(np.int64)


def setup(config):
    if config in _CACHE:
        return _CACHE[config]
    scn = P.scanner_for(config)
    g0 = P.grid_for(scn)
    grid = Grid(g0.nx, g0.ny, 2 * int(np.ceil(CR.Z_EXT / 3.0)), 3.0)
    ph = G.load_body(CR.BODY)
    mi = [m[0] for m in B.MATERIALS].index("carbon")
    mu = B.mu_with_backrest(grid, ph, mi)
    gx, gy, gz = grid.axes()
    Y, X = np.meshgrid(gy, gx, indexing="ij")
    for r in CR.EXTRA.values():
        inside = (X / r["a"]) ** 2 + (Y / r["b"]) ** 2 <= 1.0
        zs = (gz >= r["z0"]) & (gz <= r["z1"])
        sl = mu[zs]
        sl[:, inside] = np.where(sl[:, inside] == 0, C.MU_PER_MM["soft_tissue"], sl[:, inside])
        mu[zs] = sl
    masks = D.class_masks(ph)
    _CACHE[config] = (scn, Blocks(scn), grid, mu, ph, masks)
    return _CACHE[config]


def _work(args):
    config, name, n, seed = args
    scn, blk, grid, mu, ph, masks = setup(config)
    nb = blk.nb
    rng = np.random.default_rng(seed)
    e = CR.sample(name, n, rng, ph, masks)
    d = fastmc._iso(rng, len(e))
    det, qs = [], []
    for sgn in (1.0, -1.0):
        q, ok = scn.detect(e, sgn * d, rng)
        idx = np.nonzero(ok)[0]
        if len(idx):
            att = np.exp(-fwd(mu, grid, e[idx].astype(np.float32), q[idx].astype(np.float32)))
            ok[idx] = rng.random(len(idx)) < att
        det.append(ok)
        qs.append(q)
    singles = np.zeros(nb, np.int64)
    for q, ok in zip(qs, det):
        singles += np.bincount(blk.block_of(q[ok]), minlength=nb)
    both = det[0] & det[1]
    b1, b2 = blk.block_of(qs[0][both]), blk.block_of(qs[1][both])
    lo, hi = np.minimum(b1, b2), np.maximum(b1, b2)
    pairs = np.bincount(lo * nb + hi, minlength=nb * nb)
    return singles, pairs, len(e)


def run_task(out, config, n_per_class, seed):
    scn, blk, grid, mu, ph, masks = setup(config)
    nb = blk.nb
    res = {}
    t0 = time.time()
    for ci, name in enumerate(CR.CLASSES):
        k = int(np.ceil(n_per_class / CHUNK))
        jobs = [(config, name, min(CHUNK, n_per_class - i * CHUNK), seed + 1000 * ci + i) for i in range(k)]
        with ProcessPoolExecutor(max_workers=min(WORKERS, len(jobs))) as ex:
            parts = list(ex.map(_work, jobs))
        s = sum(p[0] for p in parts)
        pr = sum(p[1] for p in parts)
        n = sum(p[2] for p in parts)
        nz = np.nonzero(pr)[0]
        res[f"{name}__singles"] = s / n
        res[f"{name}__pair_idx"] = nz.astype(np.int64)
        res[f"{name}__pair_w"] = pr[nz] / n
        res[f"{name}__vol_ml"] = np.float64(CR.class_volume_ml(name, ph, masks))
        log(f"[{config}] {name:12s} {n:.2e} decays: singles/decay {s.sum() / n:.4f}, trues/decay {pr.sum() / n:.5f} "
            f"({time.time() - t0:.0f} s)")
    res["n_blocks"] = np.int64(nb)
    res["block_module"] = blk.module.astype(np.int64)
    res["block_lyso"] = blk.lyso
    res["block_centre"] = blk.centre
    np.savez_compressed(out / f"tables_{config}.npz", **res)
    log(f"wrote tables_{config}.npz")


def dmax(tab):
    nb = int(tab["n_blocks"])
    idx = np.unique(np.concatenate([tab[f"{c}__pair_idx"] for c in CR.CLASSES]))
    c = tab["block_centre"]
    d = np.linalg.norm(c[idx // nb] - c[idx % nb], axis=1)
    return float(d.max()) + 2 * 0.5 * np.hypot(BT * G.PITCH, BA * G.PITCH)


def rates(tab, tg, act, dose, par, window_ns):
    nb = int(tab["n_blocks"])
    tau_b = par["tau_block_ns"] * 1e-9
    win = window_ns * 1e-9
    bm = tab["block_module"]
    nm = int(bm.max()) + 1
    idx_all = np.unique(np.concatenate([tab[f"{c}__pair_idx"] for c in CR.CLASSES]))
    pos = {c: np.searchsorted(idx_all, tab[f"{c}__pair_idx"]) for c in CR.CLASSES}
    b1, b2 = idx_all // nb, idx_all % nb
    mm = np.zeros((nm, nm), bool)
    mm[bm[b1], bm[b2]] = True
    mm[bm[b2], bm[b1]] = True
    mm &= ~np.eye(nm, dtype=bool)
    keys = ("singles", "trues", "trues_ideal", "randoms", "scatter", "nec", "live_min")
    out = {k: np.zeros(len(tg)) for k in keys}
    kept = tab["block_lyso"] > 0
    for it in range(len(tg)):
        a = {c: act[c][it] * dose * 1e3 * float(tab[f"{c}__vol_ml"]) for c in CR.CLASSES}
        s = sum(a[c] * tab[f"{c}__singles"] for c in CR.CLASSES) * par["singles_scatter"]
        s = s + tab["block_lyso"] * par["lyso_cps_per_block"]
        w = np.zeros(len(idx_all))
        for c in CR.CLASSES:
            w[pos[c]] += a[c] * tab[f"{c}__pair_w"]
        live = np.exp(-s * tau_b)
        t = float((w * live[b1] * live[b2]).sum())
        sm = np.bincount(bm, s * live, minlength=nm)
        r = win * float(sm @ mm @ sm) / 2.0
        sc = t * par["sf"] / (1 - par["sf"])
        out["singles"][it] = float((s * live).sum())
        out["trues"][it] = t
        out["trues_ideal"][it] = float(w.sum())
        out["randoms"][it] = r
        out["scatter"][it] = sc
        out["nec"][it] = t * t / max(t + sc + r, 1e-9)
        out["live_min"][it] = float(live[kept].min())
    return out


def collect(out):
    tabs = {c: dict(np.load(out / f"tables_{c}.npz")) for c in configs() if (out / f"tables_{c}.npz").exists()}
    if not tabs:
        raise SystemExit("no tables")
    par = dict(CR.BASE)
    ref = "R280_slot"
    margin = par["window_ns"][ref] - (dmax(tabs[ref]) / C_MM_PS / 1e3 if ref in tabs else 2.15)
    res, rows = {}, []
    for state in D.STATES:
        tg, act = CR.class_activity(state)
        dt = tg[1] - tg[0]
        for c, tab in tabs.items():
            win = round(dmax(tab) / C_MM_PS / 1e3 + margin, 2)
            r = rates(tab, tg, act, DOSE, par, win)
            T, Ti, R, N = (float(r[k].sum() * dt) for k in ("trues", "trues_ideal", "randoms", "nec"))
            rows.append({"config": c, "state": state, "window_ns": win, "trues_M": T / 1e6, "trues_ideal_M": Ti / 1e6,
                         "randoms_M": R / 1e6, "nec_M": N / 1e6, "nec_over_trues_ideal": N / Ti,
                         "peak_randoms_over_trues": float((r["randoms"] / np.maximum(r["trues"], 1)).max()),
                         "live_min": float(r["live_min"].min())})
            res.setdefault(c, {})[state] = N / Ti
    (out / "nec_factors.json").write_text(json.dumps({"dose_mbq": DOSE, "factors": res}, indent=2))
    (out / "countrate_8b.json").write_text(json.dumps(rows, indent=2))
    lines = [f"Stage 8b count rates, 15O-water {DOSE:.0f} MBq, 240 s, non-TOF NEC (k = 1), SF {par['sf']}, block dead time "
             f"{par['tau_block_ns']:.0f} ns, body-scatter singles x{par['singles_scatter']}",
             f"{'config':12s} {'state':6s} {'win ns':>6s} {'trues M':>8s} {'randoms M':>9s} {'NEC M':>7s} {'NEC/T':>6s} "
             f"{'peak R/T':>8s} {'min live':>8s}"]
    for r in rows:
        lines.append(f"{r['config']:12s} {r['state']:6s} {r['window_ns']:6.2f} {r['trues_M']:8.1f} {r['randoms_M']:9.1f} "
                     f"{r['nec_M']:7.1f} {r['nec_over_trues_ideal']:6.3f} {r['peak_randoms_over_trues']:8.1f} {r['live_min']:8.3f}")
    (out / "SUMMARY_countrate_8b.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--recon", required=True)
    ap.add_argument("--task", type=int)
    ap.add_argument("--collect", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--ntasks", action="store_true")
    ap.add_argument("--n-per-class", type=float, default=1e8)
    ap.add_argument("--seed", type=int, default=77)
    a = ap.parse_args()
    if a.ntasks:
        print(len(configs()))
        return
    recon = Path(a.recon)
    if a.smoke:
        out = recon / "panels_cr_smoke"
        out.mkdir(exist_ok=True)
        for c in ("R280_slot", "P2_T20A30", "P4_T30A20"):
            run_task(out, c, int(2e5), a.seed)
        collect(out)
        return
    out = recon / "panels_cr"
    out.mkdir(exist_ok=True)
    if a.collect:
        collect(out)
    elif a.task is not None:
        run_task(out, configs()[a.task], int(a.n_per_class), a.seed)
    else:
        raise SystemExit("give --task N, --collect or --smoke")


if __name__ == "__main__":
    main()

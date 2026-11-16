"""
Stage 5d: 15O-water MBF precision with randoms, scatter and dead time for the 4-panel 30 x 24 cm design, compared
with the 290 deg open ring under the same model.

    python mbf_panels.py --recon /scratch/$USER/exercise_pet_recon --task N     one (geometry, state, scan)
    python mbf_panels.py --recon /scratch/$USER/exercise_pet_recon --collect    figure and tables

Method as stage 5c (mbf_nec.py): one stage-5 scan at 400 MBq (trues only, P90 body, carbon backrest, 240 s,
frames 12x5, 6x10, 3x20, 2x30 s), then each true is kept with probability f_k = WIN x NEC_k / T_k in frame k,
with NEC_k from the stage-8b count-rate model (panels_cr.py tables, same window rule) and WIN = 0.65 the
energy-window factor of the stage-10 benchmark. Frame images are divided by f_k. Scenarios: trues only
(f = WIN), 400 MBq and 800 MBq (no end shields). TOF 150 ps. Needs panels_cr/tables_<config>.npz (stage 8b/8c)
and the carbon-backrest sensitivity images written by motion_panels.py --prep.
Output <recon>/mbf_panels/: task_*.json/.npz, mbf_panels_metrics.csv, Fig_mbf_panels, SUMMARY_mbf_panels.txt.
"""

import argparse
import csv
import json
import os
from pathlib import Path

import numpy as np

import geometry as G
import panels as P

# make dynamic.py, evaluate.py and their workers accept panel geometries (patched before they are used)
_Ring0, _grid0 = G.Ring, G.recon_grid


def _ring(name):
    return P.Panels(name) if name in P.PANEL_CONFIGS else _Ring0(name)


def _recon_grid(scn, v=None):
    g = _grid0(scn, v or G.C.RECON_VOXEL_MM)
    if isinstance(scn, P.Panels) and scn.zmax > G.ZMAX:          # as panels.grid_for, without recursion
        return G.Grid(g.nx, g.ny, max(g.nz, 2 * int(np.ceil(scn.zmax / g.v))), g.v)
    return g


G.Ring, G.recon_grid = _ring, _recon_grid

import dynamic as D          # noqa: E402
import countrate as CR       # noqa: E402
import panels_cr as PCR      # noqa: E402
import motion_panels as MP9  # noqa: E402
from run_recon import log    # noqa: E402

C = G.C
WIN = float(os.environ.get("XPET_WIN", 0.65))
CONFIGS = ["R280_slot", "P4_T30A24"]
CTR = 150
NREAL = int(os.environ.get("XPET_N_MBF", 5))
SCEN = [("trues only", None), ("400 MBq", 400.0), ("800 MBq", 800.0)]
SIM_MBQ = float(os.environ.get("XPET_SIM_MBQ", 400.0))   # lower only for code tests
OUTDIR = "mbf_panels"


def tasks():
    return [(c, s, r) for c in CONFIGS for s in D.STATES for r in range(NREAL)]


def _emit5(config, ph, mu, n_emit, cdf, seed):
    return D._emit_timed(config, ph, mu, n_emit, cdf, seed)


def frame_factors(recon_dir, config, state, dose):
    if dose is None:
        return np.full(len(D.FRAMES), WIN)
    cr = Path(recon_dir) / "panels_cr"
    tab = dict(np.load(cr / f"tables_{config}.npz"))
    ref = dict(np.load(cr / "tables_R280_slot.npz"))
    margin = CR.BASE["window_ns"]["R280_slot"] - PCR.dmax(ref) / PCR.C_MM_PS / 1e3
    win = round(PCR.dmax(tab) / PCR.C_MM_PS / 1e3 + margin, 2)
    tg, act = CR.class_activity(state)
    r = PCR.rates(tab, tg, act, dose, dict(CR.BASE), win)
    fi = np.searchsorted(D.EDGES, tg, side="right") - 1
    nk = len(D.FRAMES)
    nec = np.bincount(fi, weights=r["nec"], minlength=nk)[:nk]
    t_sim = np.bincount(fi, weights=r["trues_ideal"] * SIM_MBQ / dose, minlength=nk)[:nk]
    f = nec / np.maximum(t_sim, 1e-12)
    rate = t_sim / np.asarray(D.FRAMES, float)
    low = rate < 0.01 * rate.max()
    f[low] = np.maximum(f[low], 0.05)
    return np.clip(WIN * f, 1e-6, 1.0)


def simulate(recon_dir, config, state, real, seed):
    """dynamic.simulate with panel geometries and the stage-9 sensitivity file."""
    scn = G.Ring(config)
    grid = G.recon_grid(scn, 3.0)
    ph = G.load_body(D.BODY)
    masks = D.class_masks(ph)
    mu = MP9.mu_backrest(grid, ph)
    sens = np.load(MP9.sens_path(recon_dir, config))
    tacs = D.class_tacs(state, SIM_MBQ)
    rng = np.random.default_rng(seed + 1000 * real + 17 * D.STATES.index(state))
    jobs, n_tot = [], 0
    for name in D.CLASSES:
        q, vol = D.class_phantom(ph, masks, name)
        dens = tacs[name] * D.DECAY
        n_mean = float(dens.sum() * D.DT * vol * 1000.0)
        n_emit = int(rng.poisson(n_mean)) if n_mean < 1e18 else int(n_mean)
        cdf = np.cumsum(dens)
        cdf = cdf / cdf[-1] if cdf[-1] > 0 else np.linspace(0, 1, len(cdf))
        k = max(1, min(D.WORKERS, int(np.ceil(n_emit / 5e7))))
        base = n_emit // k
        for i in range(k):
            ni = base + (1 if i < n_emit - base * k else 0)
            if ni:
                jobs.append((config, q, mu, ni, cdf, int(rng.integers(1 << 31))))
        n_tot += n_emit
    parts = D.pool_map(_emit5, jobs)
    ev = {key: np.concatenate([p[key] for p in parts]) for key in ("p1", "p2", "dt", "t")}
    log(f"[{config} {state} r{real}] {n_tot:.3e} decays -> {len(ev['t']):,} trues (before window factor)")
    return grid, sens, ev


def run_task(recon_dir, out, config, state, real, iters, subsets, seed):
    grid, sens, ev_all = simulate(recon_dir, config, state, real, seed)
    fi = np.searchsorted(D.EDGES, ev_all["t"], side="right") - 1
    mn, md = D.TRUE_MBF[state]
    rows, meta, npz = [], {"config": config, "state": state, "real": real, "ctr_ps": CTR, "win": WIN,
                           "n_trues_sim": int(len(fi)), "scenarios": {}}, {}
    for si, (name, dose) in enumerate(SCEN):
        f = frame_factors(recon_dir, config, state, dose)
        rng = np.random.default_rng(seed + 7000 + 1000 * real + 17 * D.STATES.index(state) + 101 * si)
        keep = rng.random(len(fi)) < f[fi]
        ev = {k: v[keep] for k, v in ev_all.items()}
        rng_r = np.random.default_rng(seed + 9000 + 1000 * real + 17 * D.STATES.index(state) + 101 * si)
        blood, segs, fits, counts, _ = D.recon_fit(config, grid, sens, ev, CTR, iters, subsets, rng_r, scale=f)
        for g, ft in enumerate(fits):
            rows.append({"config": config, "state": state, "real": real, "scenario": name, "segment": g,
                         "defect": g == D.DEF_SEG, "true_mbf": md if g == D.DEF_SEG else mn, "mbf": ft["mbf"],
                         "ptf": ft["ptf"], "fbv": ft["fbv"]})
        meta["scenarios"][name] = {"frame_factor": f.tolist(), "counts_kept_M": float(keep.sum()) / 1e6}
        npz[f"s{si}_blood"], npz[f"s{si}_segs"] = blood, segs
        nm = [r["mbf"] for r in rows if r["scenario"] == name and not r["defect"]]
        log(f"  {name}: kept {keep.mean():.3f}; MBF normal {np.mean(nm):.2f} ± {np.std(nm):.2f}")
    tag = f"task_{config}_{state}_r{real}"
    (out / f"{tag}.json").write_text(json.dumps({"rows": rows, "meta": meta}, indent=1))
    np.savez_compressed(out / f"{tag}.npz", **npz)


def collect(out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rows, metas = [], []
    for c, s, r in tasks():
        f = out / f"task_{c}_{s}_r{r}.json"
        if f.exists():
            d = json.loads(f.read_text())
            rows += d["rows"]
            metas.append(d["meta"])
        else:
            log(f"missing: {f.name}")
    if not rows:
        raise SystemExit("no task results")
    with open(out / "mbf_panels_metrics.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    lab = {"R280_slot": "56 cm ring, 290°", "P4_T30A24": "4 panels 30 x 24 cm"}
    L = [f"Stage 5d MBF with randoms: 15O-water, {CTR} ps, window factor {WIN}; mean over scans "
         f"(segmental SD = SD over scans of each segment, averaged over the 15 normal segments)",
         f"{'geometry':20s} {'scenario':11s} {'state':6s} {'normal MBF':>11s} {'seg SD %':>8s} {'defect MBF':>11s} {'MFR normal':>10s} {'MFR defect':>10s} {'counts M':>8s}"]
    res = {}
    for c in CONFIGS:
        for name, _ in SCEN:
            for s in D.STATES:
                sel = [r for r in rows if r["config"] == c and r["scenario"] == name and r["state"] == s]
                if not sel:
                    continue
                reals = sorted({r["real"] for r in sel})
                M = np.array([[next(r["mbf"] for r in sel if r["real"] == q and r["segment"] == g) for g in range(16)]
                              for q in reals])
                norm = np.delete(M, D.DEF_SEG, axis=1)
                sd = float(np.mean(norm.std(0) / norm.mean(0)) * 100) if len(reals) > 1 else np.nan
                cnt = np.mean([m["scenarios"][name]["counts_kept_M"] for m in metas if m["config"] == c and m["state"] == s])
                res[(c, name, s)] = dict(normal=float(norm.mean()), sd=sd, defect=float(M[:, D.DEF_SEG].mean()), M=M, counts=cnt)
            if (c, name, "rest") in res and (c, name, "stress") in res:
                rr, ss = res[(c, name, "rest")], res[(c, name, "stress")]
                n = min(len(rr["M"]), len(ss["M"]))
                mfr = ss["M"][:n] / rr["M"][:n]
                for s in D.STATES:
                    x = res[(c, name, s)]
                    L.append(f"{lab[c]:20s} {name:11s} {s:6s} {x['normal']:11.2f} {x['sd']:8.1f} {x['defect']:11.2f} "
                             f"{np.delete(mfr, D.DEF_SEG, axis=1).mean():10.2f} {mfr[:, D.DEF_SEG].mean():10.2f} {x['counts']:8.1f}")
    L.append("True MBF: rest 1.0 (all), stress 3.0 normal and 1.5 defect (mL/min/g); MFR 3.0 normal, 1.5 defect.")
    (out / "SUMMARY_mbf_panels.txt").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))
    fig, axs = plt.subplots(1, 2, figsize=(10, 3.8))
    for k, (ax, s) in enumerate(zip(axs, D.STATES)):
        x = np.arange(len(SCEN))
        for i, c in enumerate(CONFIGS):
            v = [res.get((c, n, s), {}).get("sd", np.nan) for n, _ in SCEN]
            ax.bar(x + (i - 0.5) * 0.38, v, 0.36, color=P.COLS.get(c, "#2a78d6"), label=lab[c])
        ax.set_xticks(x, [n for n, _ in SCEN])
        ax.set_ylabel("Segmental MBF SD over scans (%)")
        ax.set_title(f"{'AB'[k]}. {s.capitalize()}", loc="left", fontweight="bold", fontsize=10)
        ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(out / f"Fig_mbf_panels.{ext}", dpi=250)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--recon", required=True)
    ap.add_argument("--task", type=int, default=None)
    ap.add_argument("--collect", action="store_true")
    ap.add_argument("--ntasks", action="store_true")
    ap.add_argument("--iters", type=int, default=4)
    ap.add_argument("--subsets", type=int, default=C.RECON_SUBSETS)
    ap.add_argument("--seed", type=int, default=51)
    a = ap.parse_args()
    if a.ntasks:
        print(len(tasks()))
        return
    recon_dir = Path(a.recon)
    out = recon_dir / OUTDIR
    out.mkdir(exist_ok=True)
    if a.collect:
        collect(out)
    elif a.task is not None:
        c, s, r = tasks()[a.task]
        run_task(recon_dir, out, c, s, r, a.iters, a.subsets, a.seed)
    else:
        raise SystemExit("give --task N or --collect")


if __name__ == "__main__":
    main()

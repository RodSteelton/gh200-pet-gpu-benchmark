"""
Stage 5: dynamic 15O-water myocardial blood flow (MBF) at rest and exercise stress.

    python dynamic.py --recon /scratch/$USER/exercise_pet_recon --task N     one (geometry, state, realisation)
    python dynamic.py --recon /scratch/$USER/exercise_pet_recon --collect    figures and tables

Phantom: P90 body, base heart pose, flat carbon-fibre backrest (stage 4). Each tissue class gets its own
time-activity curve (TAC) from a one-tissue model driven by an arterial input (gamma-variate first pass
plus recirculation towards whole-body water equilibrium). 400 MBq bolus, 240 s list mode,
frames 12x5, 6x10, 3x20, 2x30 s. Every emission is given a time drawn from its class TAC (with 15O decay),
so each task is one complete noisy scan. Trues only, no randoms or dead time.

Myocardium: rest MBF 1.0 mL/min/g everywhere; stress 3.0, except a mid-anterior segment (AHA segment 7,
index 6) at 1.5 (flow reserve 1.5, ischaemia). Partition coefficient 0.91.
Analysis: frame-by-frame TOF OSEM (200 ps), input from an eroded LV-cavity ROI, 16 segment TACs, fit
C(t) = PTF * MBF * Ca(t) conv exp(-MBF t / p) + FBV * Ca(t) (MBF, PTF, FBV free), frame-integrated with decay.
"""

import argparse
import csv
import json
import multiprocessing as mp
import os
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

import geometry as G
import evaluate as E
import fastmc
import backrest as B
from projector import fwd
from run_recon import osem, log

C = G.C
BODY = "P90"
HALF_LIFE = 122.24                                  # s, 15O
LAM = np.log(2) / HALF_LIFE
FRAMES = [5] * 12 + [10] * 6 + [20] * 3 + [30] * 2
EDGES = np.concatenate([[0.0], np.cumsum(FRAMES)]).astype(float)
DT = 0.1
TGRID = np.arange(0.0, EDGES[-1], DT) + DT / 2
P_MYO = 0.91
DEF_SEG = 6                                         # mid anterior (0-based)
STATES = ("rest", "stress")
TRUE_MBF = {"rest": (1.0, 1.0), "stress": (3.0, 1.5)}      # (normal, defect) mL/min/g
# arterial input: cardiac output mL/s, bolus arrival t0 s, gamma shape a, scale b s, recirculation time constant s
INPUT = {"rest": dict(co=5000 / 60, t0=12.0, a=3.0, b=3.0, tau=60.0),
         "stress": dict(co=12000 / 60, t0=8.0, a=3.0, b=2.0, tau=30.0)}
TBW_ML = 42000.0
# tissue classes: flow (mL/min/g) rest, stress; partition coefficient; vascular fraction
# (vascular term uses the input dispersed by TAU_DISP; lung flow = cardiac output / lung volume, water fraction 0.15)
TISSUE = {"soft_tissue": (0.05, 0.15, 0.90, 0.02), "adipose": (0.03, 0.03, 0.30, 0.01),
          "bone": (0.10, 0.10, 0.50, 0.03), "liver": (0.90, 0.60, 0.80, 0.10),
          "lung": (1.4, 3.3, 0.15, 0.10), "arm": (0.05, 0.20, 0.90, 0.02)}
TAU_DISP = 15.0
LUNG_LEAD = 5.0                                     # s, pulmonary artery ahead of LV
LIVER_TAU = 15.0
CLASSES = ["blood_pool", "myocardium", "myo_defect", "soft_tissue", "adipose", "bone", "liver", "lung", "arm"]
CONFIGS = list(C.RECON_CONFIGS)
WORKERS = int(os.environ.get("XPET_WORKERS", 30))


def tasks(n_real):
    return [(c, s, r) for c in CONFIGS for s in STATES for r in range(n_real)]


# ---- kinetics ---------------------------------------------------------------------------------------
def arterial(state, mbq):
    p = INPUT[state]
    a_kbq = mbq * 1e3
    tt = np.clip(TGRID - p["t0"], 0, None)
    g = tt ** (p["a"] - 1) * np.exp(-tt / p["b"])
    g /= g.sum() * DT
    first = a_kbq / p["co"] * g                                  # kBq/mL, area = dose / cardiac output
    recirc = a_kbq / TBW_ML * (1 - np.exp(-tt / p["tau"]))
    return first + recirc                                         # decay-corrected kBq/mL


def one_tissue(ca, flow_ml_min_g, p, vb=0.0):
    from scipy.signal import fftconvolve
    k = flow_ml_min_g / 60.0
    kern = np.exp(-(k / p) * (TGRID - TGRID[0]))
    # exact integral of the exponential over each step keeps the discrete convolution unbiased for fast k
    kern *= (1 - np.exp(-(k / p) * DT)) / ((k / p) * DT)
    return k * fftconvolve(ca, kern)[: len(ca)] * DT + vb * ca


def class_tacs(state, mbq):
    ca = arterial(state, mbq)
    si = STATES.index(state)
    mn, md = TRUE_MBF[state]
    tac = {"blood_pool": ca, "myocardium": one_tissue(ca, mn, P_MYO), "myo_defect": one_tissue(ca, md, P_MYO)}
    def disperse(x, tau):
        kd = np.exp(-(TGRID - TGRID[0]) / tau)
        return np.convolve(x, kd / kd.sum())[: len(x)]

    ca_disp = disperse(ca, TAU_DISP)
    for name, (fr, fs, p, vb) in TISSUE.items():
        inp = ca
        if name == "lung":                                          # pulmonary arterial input arrives earlier
            inp = np.interp(TGRID + LUNG_LEAD, TGRID, ca)
        elif name == "liver":                                       # mostly portal: delayed, dispersed
            inp = disperse(ca, LIVER_TAU)
        tac[name] = one_tissue(inp, (fr, fs)[si], p) + vb * (inp if name == "lung" else ca_disp)
    return tac                                                     # decay-corrected kBq/mL on TGRID


# ---- phantom classes --------------------------------------------------------------------------------
def class_masks(ph):
    L = C.LABELS
    lab = ph["labels"]
    x, y, z = ph["x"], ph["y"], ph["z"]
    c, u, e1, e2 = G.lv_frame()
    Z, Y, X = np.meshgrid(z, y, x, indexing="ij")
    P = np.stack([X - c[0], Y - c[1], Z - c[2]], -1)
    w = P @ u
    psi = np.degrees(np.arctan2(P @ e2, P @ e1)) % 360.0
    myo = lab == L["myocardium"]
    dfx = myo & (w >= 0) & (w < 20) & (((psi + 30.0) % 360.0) < 60.0)
    m = {"blood_pool": lab == L["blood_pool"], "myocardium": myo & ~dfx, "myo_defect": dfx}
    for n in ("soft_tissue", "adipose", "bone", "liver", "lung"):
        m[n] = lab == L[n]
    return m


def class_phantom(ph, masks, name):
    q = dict(ph)
    if name == "arm":
        q["activity"] = np.ones_like(ph["activity"], dtype=np.float32)
        q["a_torso"], q["a_arm_each"] = np.float64(0.0), np.float64(1.0)
        vol = np.pi * (float(ph["arm_r"]) ** 2 - C.HUMERUS_RADIUS ** 2) * 2 * C.ARM_HALF_LENGTH * 2 / 1000.0
    else:
        q["activity"] = masks[name].astype(np.float32)
        q["a_torso"], q["a_arm_each"] = np.float64(1.0), np.float64(0.0)
        vol = masks[name].sum() * C.VOXEL_MM ** 3 / 1000.0
    return q, vol                                                  # vol in mL


# ---- worker -----------------------------------------------------------------------------------------
def _emit_timed(config, ph, mu, n_emit, cdf, seed, chunk=2_000_000):
    ring = G.Ring(config)
    grid = G.recon_grid(ring, 3.0)
    rng = np.random.default_rng(seed)
    out = {"p1": [], "p2": [], "dt": [], "t": []}
    done = 0
    while done < n_emit:
        n = min(chunk, n_emit - done)
        e = fastmc.sample_emissions(ph, n, rng)
        tt = np.interp(rng.random(len(e)), cdf, TGRID).astype(np.float32)
        done += n
        d = fastmc._iso(rng, len(e))
        q1, ok1 = ring.detect(e, d, rng)
        q2, ok2 = ring.detect(e, -d, rng)
        ok = ok1 & ok2
        q1, q2, e, tt = q1[ok], q2[ok], e[ok], tt[ok]
        att = np.exp(-fwd(mu, grid, q1.astype(np.float32), q2.astype(np.float32)))
        keep = rng.random(len(att)) < att
        q1, q2, e, tt = q1[keep], q2[keep], e[keep], tt[keep]
        s1, _ = ring.snap(q1)
        s2, _ = ring.snap(q2)
        dt = (np.linalg.norm(q1 - e, axis=1) - np.linalg.norm(q2 - e, axis=1)) / C.C_MM_PER_PS
        for k, v in (("p1", s1), ("p2", s2), ("dt", dt.astype(np.float32)), ("t", tt)):
            out[k].append(v)
    return {k: np.concatenate(v) for k, v in out.items()}


# ---- fitting ----------------------------------------------------------------------------------------
FRAME_DECAY = np.array([np.exp(-LAM * TGRID[(TGRID >= a) & (TGRID < b)]).sum() * DT
                        for a, b in zip(EDGES[:-1], EDGES[1:])])          # integral of decay over frame, s
MID = 0.5 * (EDGES[:-1] + EDGES[1:])
DECAY = np.exp(-LAM * TGRID)
FRAME_IDX = np.searchsorted(EDGES, TGRID, side="right") - 1


def frame_integral(curve_dc):
    """Frame integrals of a decay-corrected curve with decay applied (matches emissions per frame)."""
    return np.bincount(FRAME_IDX, weights=curve_dc * DECAY * DT, minlength=len(FRAMES))


def fit_mbf(blood, tissue, scale=None):
    """blood, tissue: per-frame image values proportional to emissions per mL in each frame.
    scale: per-frame fraction of counts kept (NEC thinning); variance of the rescaled values grows as 1/scale."""
    from scipy.optimize import least_squares
    ca_f = blood / FRAME_DECAY                                      # decay-corrected mean over frame
    ca = ca_f[FRAME_IDX]                                            # frame-constant input (preserves frame integrals)
    sd = np.sqrt(np.abs(tissue) + 0.01 * np.abs(tissue).max() + 1e-12)   # Poisson-like weights
    if scale is not None:
        sd = sd / np.sqrt(np.asarray(scale, float))

    def model(prm):
        mbf, ptf, fbv = prm
        return frame_integral(ptf * one_tissue(ca, mbf, P_MYO) + fbv * ca)

    def res(prm):
        return (model(prm) - tissue) / sd

    best = None
    for m0 in (0.7, 1.5, 3.0):
        r = least_squares(res, [m0, 0.7, 0.2], bounds=([0.05, 0.0, 0.0], [8.0, 2.0, 1.0]))
        if best is None or r.cost < best.cost:
            best = r
    return dict(mbf=float(best.x[0]), ptf=float(best.x[1]), fbv=float(best.x[2]), cost=float(best.cost),
                fit=model(best.x))


# ---- one task ---------------------------------------------------------------------------------------
def pool_map(fn, jobs):
    ncpu = int(os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count() or 1))
    workers = max(1, min(WORKERS, len(jobs), ncpu))
    os.environ["NUMBA_NUM_THREADS"] = str(max(1, ncpu // workers))
    with ProcessPoolExecutor(workers, mp_context=mp.get_context("spawn")) as ex:
        return [f.result() for f in [ex.submit(fn, *j) for j in jobs]]


def simulate(recon_dir, config, state, real, mbq, seed):
    """One noisy 15O-water scan: list-mode trues with emission times (trues only)."""
    ring = G.Ring(config)
    grid = G.recon_grid(ring, 3.0)
    ph = G.load_body(BODY)
    masks = class_masks(ph)
    mi = [m[0] for m in B.MATERIALS].index("carbon")
    mu = B.mu_with_backrest(grid, ph, mi)
    sens = np.load(recon_dir / f"sens_{config}_{BODY}_backrest_carbon_v3.npy")
    tacs = class_tacs(state, mbq)
    rng = np.random.default_rng(seed + 1000 * real + 17 * STATES.index(state))

    jobs, n_tot = [], 0
    for ci, name in enumerate(CLASSES):
        q, vol = class_phantom(ph, masks, name)
        dens = tacs[name] * DECAY
        n_mean = float(dens.sum() * DT * vol * 1000.0)            # decays (kBq s -> Bq s)
        n_emit = int(rng.poisson(n_mean)) if n_mean < 1e18 else int(n_mean)
        cdf = np.cumsum(dens)
        cdf = cdf / cdf[-1] if cdf[-1] > 0 else np.linspace(0, 1, len(cdf))
        k = max(1, min(WORKERS, int(np.ceil(n_emit / 5e7))))
        base = n_emit // k
        for i in range(k):
            ni = base + (1 if i < n_emit - base * k else 0)
            if ni:
                jobs.append((config, q, mu, ni, cdf, int(rng.integers(1 << 31))))
        n_tot += n_emit
        log(f"  {name:12s} {vol:8.0f} mL  {n_mean:.3e} decays in {EDGES[-1]:.0f} s  ({k} chunks)")
    t0 = time.time()
    parts = pool_map(_emit_timed, jobs)
    ev_all = {key: np.concatenate([p[key] for p in parts]) for key in ("p1", "p2", "dt", "t")}
    del parts
    log(f"[{config} {state} r{real}] {n_tot:.3e} decays -> {len(ev_all['t']):,} trues in {time.time() - t0:.0f} s")
    return grid, sens, ev_all, n_tot


def recon_fit(config, grid, sens, ev_all, ctr, iters, subsets, rng_r, scale=None):
    """Frame-by-frame TOF OSEM, LV input and 16 segment TACs, one-tissue fits.
    scale: optional per-frame count fraction kept (NEC thinning); images are divided by it."""
    t = E.Truth.get(config, BODY)
    from scipy.ndimage import binary_erosion
    blood_roi = binary_erosion(t["lab"] == C.LABELS["blood_pool"], iterations=2)
    myo = t["lab"] == C.LABELS["myocardium"]
    seg_masks = []
    for name, w0, w1, nsec in E.LEVELS:
        lev = myo & (t["w"] >= w0) & (t["w"] < w1)
        for kk in range(nsec):
            a0 = 360.0 / nsec * kk - 180.0 / nsec
            seg_masks.append(lev & (((t["psi"] - a0) % 360.0) < 360.0 / nsec))
    fi = np.searchsorted(EDGES, ev_all["t"], side="right") - 1
    order = np.argsort(fi, kind="stable")
    bounds = np.searchsorted(fi[order], np.arange(len(FRAMES) + 1))
    sig_t = ctr / 2.3548
    sc = np.ones(len(FRAMES)) if scale is None else np.asarray(scale, float)
    blood, segs, counts = [], [], []
    t0 = time.time()
    shot = None
    for f in range(len(FRAMES)):
        idx = order[bounds[f]:bounds[f + 1]]
        ev = {"p1": ev_all["p1"][idx], "p2": ev_all["p2"][idx]}
        D = np.linalg.norm(ev["p2"] - ev["p1"], axis=1)
        dt = ev_all["dt"][idx] + rng_r.normal(size=len(idx)).astype(np.float32) * sig_t
        ev["tc"] = ((D + C.C_MM_PER_PS * dt) / 2.0).astype(np.float32)
        img = osem(ev, grid, sens, C.C_MM_PER_PS * sig_t / 2.0, iters, subsets, rng_r) if len(idx) else np.zeros(grid.shape, np.float32)
        img = img / sc[f]
        blood.append(float(img[blood_roi].mean()))
        segs.append([float(img[m].mean()) for m in seg_masks])
        counts.append(len(idx))
        if EDGES[f] >= 120 and shot is None:
            shot = E.reslice(img, grid, "sa")
    log(f"  {len(FRAMES)} frames reconstructed in {time.time() - t0:.0f} s")
    blood, segs = np.array(blood), np.array(segs).T                 # segs: [16, frames]
    fits = [fit_mbf(blood, s, None if scale is None else sc) for s in segs]
    return blood, segs, fits, counts, shot


def run_task(recon_dir, out, config, state, real, mbq, ctr, iters, subsets, seed):
    grid, sens, ev_all, n_tot = simulate(recon_dir, config, state, real, mbq, seed)
    rng_r = np.random.default_rng(seed + 5000 + 1000 * real + 17 * STATES.index(state))
    blood, segs, fits, counts, shot = recon_fit(config, grid, sens, ev_all, ctr, iters, subsets, rng_r)
    mn, md = TRUE_MBF[state]
    rows = []
    for si, ft in enumerate(fits):
        rows.append({"config": config, "state": state, "real": real, "segment": si,
                     "defect": si == DEF_SEG, "true_mbf": md if si == DEF_SEG else mn,
                     "mbf": ft["mbf"], "ptf": ft["ptf"], "fbv": ft["fbv"], "cost": ft["cost"]})
    tag = f"task_{config}_{state}_r{real}"
    meta = {"config": config, "state": state, "real": real, "mbq": mbq, "ctr_ps": ctr, "iters": iters,
            "subsets": subsets, "frames": FRAMES, "counts": counts, "n_decays": n_tot,
            "n_trues": int(len(ev_all["t"]))}
    (out / f"{tag}.json").write_text(json.dumps({"meta": meta, "rows": rows}, indent=2))
    np.savez_compressed(out / f"{tag}.npz", blood=blood, segs=segs,
                        fits=np.array([f["fit"] for f in fits]), shot=shot if shot is not None else np.zeros(1))
    nm = [r["mbf"] for r in rows if not r["defect"]]
    log(f"  MBF normal {np.mean(nm):.2f} +/- {np.std(nm):.2f} (true {mn}), defect {rows[DEF_SEG]['mbf']:.2f} (true {md})")


# ---- collect ----------------------------------------------------------------------------------------
def collect(out, n_real):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    rows, meta, traces = [], {}, {}
    for c, s, r in tasks(n_real):
        f = out / f"task_{c}_{s}_r{r}.json"
        if not f.exists():
            log(f"missing: {f.name}")
            continue
        d = json.loads(f.read_text())
        rows.extend(d["rows"])
        meta[(c, s, r)] = d["meta"]
        traces[(c, s, r)] = dict(np.load(out / f"task_{c}_{s}_r{r}.npz"))
    if not rows:
        raise SystemExit("no task results found")
    with open(out / "dynamic_metrics.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    reals = sorted({r["real"] for r in rows})
    configs = [c for c in CONFIGS if any(r["config"] == c for r in rows)]
    get = {(r["config"], r["state"], r["real"], r["segment"]): r for r in rows}
    summ = []
    for c in configs:
        for s in STATES:
            nm = [np.mean([get[(c, s, r, g)]["mbf"] for g in range(16) if g != DEF_SEG])
                  for r in reals if (c, s, r, 0) in get]
            dm = [get[(c, s, r, DEF_SEG)]["mbf"] for r in reals if (c, s, r, DEF_SEG) in get]
            segsd = np.mean([np.std([get[(c, s, r, g)]["mbf"] for r in reals if (c, s, r, g) in get])
                             / TRUE_MBF[s][0] * 100 for g in range(16) if g != DEF_SEG])
            segcv = np.mean([np.std([get[(c, s, r, g)]["mbf"] for g in range(16) if g != DEF_SEG])
                             / TRUE_MBF[s][0] * 100 for r in reals if (c, s, r, 0) in get])
            summ.append({"config": c, "state": s, "n_real": len(nm), "normal_mbf_mean": float(np.mean(nm)),
                         "normal_mbf_sd": float(np.std(nm)), "defect_mbf_mean": float(np.mean(dm)),
                         "defect_mbf_sd": float(np.std(dm)), "true_normal": TRUE_MBF[s][0],
                         "true_defect": TRUE_MBF[s][1], "segment_repeat_sd_pct": float(segsd),
                         "segment_cv_pct": float(segcv),
                         "trues_M": float(np.mean([meta[(c, s, r)]["n_trues"] for r in reals if (c, s, r) in meta]) / 1e6),
                         "first_frame_trues_k": float(np.mean([max(meta[(c, s, r)]["counts"][:4]) for r in reals
                                                               if (c, s, r) in meta]) / 1e3)})
    mfr = {}
    for c in configs:
        n_, d_ = [], []
        for r in reals:
            if (c, "rest", r, 0) in get and (c, "stress", r, 0) in get:
                n_.append(np.mean([get[(c, "stress", r, g)]["mbf"] / get[(c, "rest", r, g)]["mbf"]
                                   for g in range(16) if g != DEF_SEG]))
                d_.append(get[(c, "stress", r, DEF_SEG)]["mbf"] / get[(c, "rest", r, DEF_SEG)]["mbf"])
        mfr[c] = (np.array(n_), np.array(d_))
    (out / "dynamic_summary.json").write_text(json.dumps(
        {"summary": summ, "mfr": {c: {"normal": v[0].tolist(), "defect": v[1].tolist()} for c, v in mfr.items()}},
        indent=2))

    fig, axs = plt.subplots(2, 2, figsize=(9.6, 7.6))
    # A: example TACs
    ax = axs[0, 0]
    E.style(ax)
    ex = next(((c, "stress", r) for c in configs for r in reals if (c, "stress", r) in traces), None)
    if ex:
        tr = traces[ex]
        dur = np.array(FRAMES, float)
        norm = tr["blood"].max()
        ax.step(EDGES[:-1], tr["blood"] / norm, where="post", color=E.INK2, lw=1.2, label="LV blood (input)")
        nidx = [g for g in range(16) if g != DEF_SEG]
        nt = tr["segs"][nidx].mean(0) / norm
        ax.plot(MID, nt * 5, "o", ms=3, color=E.COL[ex[0]], label="normal segments ×5")
        ax.plot(MID, tr["fits"][nidx].mean(0) / norm * 5, "-", color=E.COL[ex[0]], lw=1.2)
        ax.plot(MID, tr["segs"][DEF_SEG] / norm * 5, "s", ms=3, color="#e34948", label="defect segment ×5")
        ax.plot(MID, tr["fits"][DEF_SEG] / norm * 5, "-", color="#e34948", lw=1.2)
        ax.set_xlabel("Time after injection (s)")
        ax.set_ylabel("Image value (blood peak = 1)")
        ax.legend(fontsize=7, frameon=False)
        ax.set_title(f"A.  Stress TACs, {E.NAME[ex[0]]}", loc="left", fontweight="bold", fontsize=9)
    # B: MBF
    ax = axs[0, 1]
    E.style(ax)
    cats = [("rest", "normal"), ("stress", "normal"), ("stress", "defect")]
    x = np.arange(len(cats))
    wdt = 0.8 / len(configs)
    S = {(q["config"], q["state"]): q for q in summ}
    for n, c in enumerate(configs):
        m = [S[(c, s)][f"{k}_mbf_mean"] for s, k in cats]
        sd = [S[(c, s)][f"{k}_mbf_sd"] for s, k in cats]
        ax.bar(x + (n - (len(configs) - 1) / 2) * wdt, m, wdt * 0.9, yerr=sd, color=E.COL[c], capsize=2,
               error_kw=dict(lw=0.8))
    for xi, (s, k) in zip(x, cats):
        tv = TRUE_MBF[s][0 if k == "normal" else 1]
        ax.plot([xi - 0.45, xi + 0.45], [tv, tv], color=E.INK, lw=1.2)
    ax.set_xticks(x, ["rest", "stress\nnormal", "stress\ndefect"])
    ax.set_ylabel("MBF (mL/min/g), mean ± SD over scans")
    ax.set_title("B.  Myocardial blood flow (black: truth)", loc="left", fontweight="bold", fontsize=9)
    # C: MFR
    ax = axs[1, 0]
    E.style(ax)
    for n, c in enumerate(configs):
        nv, dv = mfr[c]
        for j, v in enumerate((nv, dv)):
            if len(v):
                ax.bar(j + (n - (len(configs) - 1) / 2) * wdt, v.mean(), wdt * 0.9, yerr=v.std(),
                       color=E.COL[c], capsize=2, error_kw=dict(lw=0.8))
    for j, tv in enumerate((TRUE_MBF["stress"][0] / TRUE_MBF["rest"][0],
                            TRUE_MBF["stress"][1] / TRUE_MBF["rest"][1])):
        ax.plot([j - 0.45, j + 0.45], [tv, tv], color=E.INK, lw=1.2)
    ax.axhline(2.0, color="#e34948", lw=0.8, ls=(0, (4, 2)))
    ax.set_xticks([0, 1], ["normal segments", "defect segment"])
    ax.set_ylabel("Myocardial flow reserve (stress / rest)")
    ax.set_title("C.  Flow reserve (dashed: 2.0 cut-off)", loc="left", fontweight="bold", fontsize=9)
    # D: segmental precision
    ax = axs[1, 1]
    E.style(ax)
    for n, c in enumerate(configs):
        vals = [S[(c, s)]["segment_repeat_sd_pct"] for s in STATES]
        ax.bar(np.arange(2) + (n - (len(configs) - 1) / 2) * wdt, vals, wdt * 0.9, color=E.COL[c])
        for xi, v in zip(np.arange(2) + (n - (len(configs) - 1) / 2) * wdt, vals):
            ax.text(xi, v, f"{v:.1f}", ha="center", va="bottom", fontsize=7, color=E.INK2)
    ax.set_xticks([0, 1], ["rest", "stress"])
    ax.set_ylabel("Segmental MBF SD over scans (% of truth)")
    ax.set_title("D.  Segmental precision", loc="left", fontweight="bold", fontsize=9)
    handles = [Patch(color=E.COL[c], label=E.NAME[c]) for c in configs]
    fig.legend(handles=handles, loc="lower center", ncol=len(handles), frameon=False, fontsize=8)
    m0 = next(iter(meta.values()))
    fig.suptitle(f"15O-water, {m0['mbq']:.0f} MBq, {BODY} body, {E.CTRNAME.get(m0['ctr_ps'], m0['ctr_ps'])}, "
                 f"OSEM {m0['iters']}x{m0['subsets']}, {len(reals)} simulated scans per state", fontsize=9)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    for ext in ("png", "pdf"):
        fig.savefig(out / f"Fig_dynamic_mbf.{ext}", dpi=300)
    plt.close(fig)

    lines = [f"Stage 5 dynamic 15O-water: {m0['mbq']:.0f} MBq, {BODY}, {m0['ctr_ps']} ps, OSEM {m0['iters']}x{m0['subsets']}, "
             f"frames {FRAMES}",
             f"{'config':12s} {'state':6s} {'n':>2s} {'normal MBF':>14s} {'defect MBF':>14s} {'truth':>9s} "
             f"{'segSD%':>6s} {'segCV%':>6s} {'trues M':>8s} {'5s frame k':>10s}"]
    for q in summ:
        lines.append(f"{q['config']:12s} {q['state']:6s} {q['n_real']:2d} {q['normal_mbf_mean']:6.2f} ± {q['normal_mbf_sd']:4.2f}  "
                     f"{q['defect_mbf_mean']:6.2f} ± {q['defect_mbf_sd']:4.2f}  {q['true_normal']:.1f}/{q['true_defect']:.1f} "
                     f"{q['segment_repeat_sd_pct']:6.1f} {q['segment_cv_pct']:6.1f} {q['trues_M']:8.1f} {q['first_frame_trues_k']:10.0f}")
    for c in configs:
        nv, dv = mfr[c]
        if len(nv):
            lines.append(f"{c:12s} MFR normal {nv.mean():.2f} ± {nv.std():.2f} (true 3.0), defect {dv.mean():.2f} ± "
                         f"{dv.std():.2f} (true 1.5)")
    (out / "SUMMARY_dynamic.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--recon", required=True)
    ap.add_argument("--task", type=int, default=None)
    ap.add_argument("--collect", action="store_true")
    ap.add_argument("--ntasks", action="store_true")
    ap.add_argument("--n-real", type=int, default=5, help="simulated scans per (geometry, state)")
    ap.add_argument("--mbq", type=float, default=400.0)
    ap.add_argument("--ctr", type=int, default=200)
    ap.add_argument("--iters", type=int, default=4)
    ap.add_argument("--subsets", type=int, default=10)
    ap.add_argument("--seed", type=int, default=51)
    ap.add_argument("--outdir", default="dynamic")
    args = ap.parse_args()
    if args.ntasks:
        print(len(tasks(args.n_real)))
        return
    recon_dir = Path(args.recon)
    out = recon_dir / args.outdir
    out.mkdir(exist_ok=True)
    if args.collect:
        collect(out, args.n_real)
    elif args.task is not None:
        c, s, r = tasks(args.n_real)[args.task]
        run_task(recon_dir, out, c, s, r, args.mbq, args.ctr, args.iters, args.subsets, args.seed)
    else:
        raise SystemExit("give --task N or --collect")


if __name__ == "__main__":
    main()

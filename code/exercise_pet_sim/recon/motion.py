"""
Stage 6: heart motion during exercise, and its correction.

    python motion.py --recon /scratch/$USER/exercise_pet_recon --task N     one (geometry, scenario, realisation)
    python motion.py --recon /scratch/$USER/exercise_pet_recon --collect    figures and tables

P90 body with the 20% mid-anterior defect of stage 3, flat carbon-fibre backrest, 120 s list mode with the
counts of the stage-2 10 M-trues data set. The heart (myocardium and LV blood) moves rigidly; the rest of the
body is static. Exercise motion (one realisation = one random breathing pattern):
  breathing   30 /min, 15 mm caudal + 4.5 mm anterior at peak inspiration (cos^4 shape, end-expiration
              dwell), cycle-to-cycle amplitude +/-20% and period +/-10%
  pedalling   60 rpm: +/-3 mm lateral at 1 Hz, +/-2 mm craniocaudal at 2 Hz
  drift       5 mm caudal slump over the scan
Reconstructions (TOF 200 ps, OSEM 8x10, static sensitivity):
  none        all events, no correction
  gated       end-expiration respiratory gate (50% of events), from the true breathing signal
  mc_data     event-by-event rigid correction with motion estimated from the data: centroid of TOF
              positions within 70 mm of the heart in 0.1 s bins, scaled by the heart fraction measured
              in the uncorrected image, smoothed (0.15 s)
  mc_true     event-by-event correction with the true motion (upper bound)
plus a no-motion reference scan. Each image is scored against the truth at its expected mean heart position.
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
import robust as R
import backrest as B
from projector import fwd
from run_recon import osem, log

C = G.C
BODY = R.TEST_BODY
T_SCAN = 120.0
CTR = int(os.environ.get("XPET_CTR", 200))           # ps; XPET_CTR=320 writes to motion_320/
SCENARIOS = [("static", 1), ("exercise", 3)]          # (name, realisations)
VARIANTS = [("none", "no correction"), ("gated", "respiratory gate"), ("mc_data", "data-driven MC"),
            ("mc_true", "ideal MC")]
WORKERS = int(os.environ.get("XPET_WORKERS", 30))
RESP_HZ, RESP_AMP, RESP_DIR = 0.5, 15.0, np.array([0.0, 0.3, -1.0])
PED_HZ, PED_X, PED_Z = 1.0, 3.0, 2.0
DRIFT_Z = -5.0
TG = np.arange(0.0, T_SCAN + 0.01, 0.01)


def _counts(env, default):
    v = os.environ.get(env)
    # "+" separates entries: sbatch --export splits on commas
    return {**default, **dict((k, int(n)) for k, n in (x.split(":") for x in v.replace(",", "+").split("+")))} \
        if v else default


def tasks(new_only=False):
    """All (geometry, scenario, realisation). XPET_MOTION_N="static:5+exercise:8" sets the totals;
    with new_only, XPET_MOTION_FROM="static:1+exercise:3" skips realisations already run."""
    n = _counts("XPET_MOTION_N", dict(SCENARIOS))
    lo = _counts("XPET_MOTION_FROM", {}) if new_only else {}
    return [(c, s, r) for c in C.RECON_CONFIGS for s, _ in SCENARIOS for r in range(lo.get(s, 0), n[s])]


def motion_trace(real):
    """Displacement (mm) of the heart on TG, shape (3, n), and the breathing signal in [0, 1]."""
    rng = np.random.default_rng(900 + real)
    n_cyc = int(T_SCAN * RESP_HZ * 1.3) + 2
    per = (1 / RESP_HZ) * rng.uniform(0.9, 1.1, n_cyc)
    amp = rng.uniform(0.8, 1.2, n_cyc)
    starts = np.concatenate([[0.0], np.cumsum(per)])[:-1] - rng.uniform(0, 1 / RESP_HZ)
    k = np.clip(np.searchsorted(starts, TG, side="right") - 1, 0, n_cyc - 1)
    ph = (TG - starts[k]) / per[k]                                   # 0..1 within cycle
    resp = amp[k] * np.sin(np.pi * ph) ** 4                          # 0 at end-expiration, peak mid-cycle
    u = RESP_DIR / np.abs(RESP_DIR[2])
    d = np.outer(u, resp * RESP_AMP)
    d[0] += PED_X * np.sin(2 * np.pi * PED_HZ * TG)
    d[2] += PED_Z * np.sin(2 * np.pi * 2 * PED_HZ * TG + 0.7)
    d[2] += DRIFT_Z * TG / T_SCAN
    return d, resp / 1.2


def _emit(config, ph, mu, n_emit, seed, trace, chunk=2_000_000):
    ring = G.Ring(config)
    grid = G.recon_grid(ring, 3.0)
    rng = np.random.default_rng(seed)
    out = {"p1": [], "p2": [], "dt": [], "t": []}
    done = 0
    while done < n_emit:
        n = min(chunk, n_emit - done)
        e = fastmc.sample_emissions(ph, n, rng)
        tt = rng.uniform(0, T_SCAN, len(e))
        if trace is not None:
            for a in range(3):
                e[:, a] += np.interp(tt, TG, trace[a])
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
        for k_, v in (("p1", s1), ("p2", s2), ("dt", dt.astype(np.float32)), ("t", tt.astype(np.float32))):
            out[k_].append(v)
    return {k_: np.concatenate(v) for k_, v in out.items()}


def pool_map(fn, jobs):
    ncpu = int(os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count() or 1))
    workers = max(1, min(WORKERS, len(jobs), ncpu))
    os.environ["NUMBA_NUM_THREADS"] = str(max(1, ncpu // workers))
    with ProcessPoolExecutor(workers, mp_context=mp.get_context("spawn")) as ex:
        return [f.result() for f in [ex.submit(fn, *j) for j in jobs]]


def truth_at(grid, shift):
    R.set_pose(tuple(shift), None)
    ph = R.phantom(BODY, defect=True)
    t = R.truth(grid, ph)
    _, prof_t = E.mid_profile(t["act"], t)
    R.set_pose((0, 0, 0), None)
    return t, prof_t


def score(img, grid, shift):
    t, prof_t = truth_at(grid, shift)
    im, prof, seg = R.measure(img, t)
    m = R.metrics(prof, seg, prof_t)
    m["recovery"] = float(np.nanmean(np.delete(seg, R.DEF_SEG)))
    R.set_pose(tuple(shift), None)
    sa = E.reslice(im, grid, "sa")
    R.set_pose((0, 0, 0), None)
    return m, sa


def estimate_motion(ev, tc, img_static, grid):
    """Data-driven rigid heart motion from TOF positions near the heart."""
    from scipy.ndimage import gaussian_filter1d
    c0 = np.array(R.BASE_CENTER)
    D = np.linalg.norm(ev["p2"] - ev["p1"], axis=1)
    pts = ev["p1"] + (ev["p2"] - ev["p1"]) * (tc / np.maximum(D, 1e-6))[:, None]
    near = np.linalg.norm(pts - c0, axis=1) < 70.0
    gx, gy, gz = grid.axes()
    Z, Y, X = np.meshgrid(gz, gy, gx, indexing="ij")
    sph = (X - c0[0]) ** 2 + (Y - c0[1]) ** 2 + (Z - c0[2]) ** 2 < 70.0 ** 2
    from scipy.ndimage import gaussian_filter
    vals = gaussian_filter(img_static, 2.0)[sph]                     # smooth before thresholding the LV
    f_h = float(vals[vals > 0.4 * vals.max()].sum() / max(vals.sum(), 1e-12))
    f_h = min(max(f_h, 0.2), 1.0)
    edges = np.arange(0.0, T_SCAN + 1e-6, 0.1)
    b = np.clip(np.searchsorted(edges, ev["t"][near], side="right") - 1, 0, len(edges) - 2)
    n = np.bincount(b, minlength=len(edges) - 1).astype(float)
    cen = np.stack([np.bincount(b, weights=pts[near, a], minlength=len(edges) - 1) for a in range(3)])
    cen = cen / np.maximum(n, 1)
    cen -= (cen * n).sum(1, keepdims=True) / n.sum()
    est = gaussian_filter1d(cen, 1.5, axis=1) / max(f_h, 1e-3)
    mids = 0.5 * (edges[:-1] + edges[1:])
    return mids, est, f_h


def run_task(recon_dir, out, config, scen, real, iters, subsets, seed, n_ref=C.FASTMC_TRUES):
    ring = G.Ring(config)
    grid = G.recon_grid(ring, 3.0)
    R.set_pose((0, 0, 0), None)
    ph = R.phantom(BODY, defect=True)
    mi = [m[0] for m in B.MATERIALS].index("carbon")
    mu = B.mu_with_backrest(grid, ph, mi)
    sens = np.load(recon_dir / f"sens_{config}_{BODY}_backrest_carbon_v3.npy")
    ref = np.load(recon_dir / f"events_fastmc_{config}_{BODY}_{int(n_ref)}.npz")
    n_emit = int(ref["n_emitted"])
    v3 = C.VOXEL_MM ** 3
    heart = np.isin(ph["labels"], [C.LABELS["myocardium"], C.LABELS["blood_pool"]])
    qh, qb = dict(ph), dict(ph)
    qh["activity"] = (ph["activity"] * heart).astype(np.float32)
    qh["a_torso"], qh["a_arm_each"] = np.float64(qh["activity"].sum() * v3), np.float64(0.0)
    qb["activity"] = (ph["activity"] * ~heart).astype(np.float32)
    qb["a_torso"] = np.float64(qb["activity"].sum() * v3)
    a_h, a_b = float(qh["a_torso"]), float(qb["a_torso"]) + 2 * float(ph["a_arm_each"])
    n_h = int(round(n_emit * a_h / (a_h + a_b)))
    trace, resp = motion_trace(real) if scen == "exercise" else (np.zeros((3, len(TG))), np.zeros(len(TG)))
    rng = np.random.default_rng(seed + 100 * real + (0 if scen == "static" else 7))
    k = WORKERS
    jobs = []
    for q, n_tot, tr in ((qh, n_h, trace), (qb, n_emit - n_h, None)):
        kk = max(1, k // 2)
        base = n_tot // kk
        for i in range(kk):
            jobs.append((config, q, mu, base + (1 if i < n_tot - base * kk else 0), int(rng.integers(1 << 31)), tr))
    t0 = time.time()
    parts = pool_map(_emit, jobs)
    ev = {key: np.concatenate([p[key] for p in parts]) for key in ("p1", "p2", "dt", "t")}
    del parts
    n_ev = len(ev["t"])
    log(f"[{config} {scen} r{real}] {n_ev:,} trues ({n_h:,} of {n_emit:,} decays in the heart) in {time.time() - t0:.0f} s")

    sig_t = CTR / 2.3548
    sigma = C.C_MM_PER_PS * sig_t / 2.0
    rng_r = np.random.default_rng(seed + 31 + 100 * real)
    D = np.linalg.norm(ev["p2"] - ev["p1"], axis=1)
    tc = ((D + C.C_MM_PER_PS * (ev["dt"] + rng_r.normal(size=n_ev).astype(np.float32) * sig_t)) / 2.0).astype(np.float32)
    dtrue = np.stack([np.interp(ev["t"], TG, trace[a]) for a in range(3)], 1).astype(np.float32)

    def rec(sel=None, shift=None):
        p1, p2, t_ = ev["p1"], ev["p2"], tc
        if sel is not None:
            p1, p2, t_ = p1[sel], p2[sel], t_[sel]
        if shift is not None:
            s = shift if sel is None else shift[sel]
            p1, p2 = p1 - s, p2 - s
        return osem({"p1": p1, "p2": p2, "tc": t_}, grid, sens, sigma, iters, subsets,
                    np.random.default_rng(seed + 77))

    rows, shots, extra = [], {}, {}
    base_row = {"config": config, "scenario": scen, "real": real, "ctr_ps": CTR, "iters": iters,
                "subsets": subsets, "n_trues": n_ev}
    img0 = rec()
    mean_d = dtrue.mean(0)
    m, sa = score(img0, grid, mean_d)
    rows.append({**base_row, "variant": "none", "n_used": n_ev, **m})
    shots["none"] = sa
    if scen == "exercise":
        gate = np.interp(ev["t"], TG, resp) < 0.25
        img = rec(sel=gate)
        m, sa = score(img, grid, dtrue[gate].mean(0))
        rows.append({**base_row, "variant": "gated", "n_used": int(gate.sum()), **m})
        shots["gated"] = sa
        mids, est, f_h = estimate_motion(ev, tc, img0, grid)
        dest = np.stack([np.interp(ev["t"], mids, est[a]) for a in range(3)], 1).astype(np.float32)
        img = rec(shift=dest)
        m, sa = score(img, grid, mean_d)
        rows.append({**base_row, "variant": "mc_data", "n_used": n_ev, **m})
        shots["mc_data"] = sa
        img = rec(shift=dtrue)
        m, sa = score(img, grid, (0, 0, 0))
        rows.append({**base_row, "variant": "mc_true", "n_used": n_ev, **m})
        shots["mc_true"] = sa
        tz = np.stack([np.interp(mids, TG, trace[a]) for a in range(3)])
        tz -= tz.mean(1, keepdims=True)
        rmse = np.sqrt(((est - tz) ** 2).mean(1))
        extra = {"f_heart": f_h, "est_rmse_mm": rmse.tolist(), "true_sd_mm": tz.std(1).tolist()}
        np.savez_compressed(out / f"trace_{config}_{scen}_r{real}.npz", mids=mids, est=est, true=tz)
        log(f"  motion estimate: heart fraction {f_h:.2f}, RMS error x/y/z {rmse.round(1)} mm "
            f"(true SD {tz.std(1).round(1)} mm)")
    for r in rows:
        r.update(extra and {"est_rmse_x": extra["est_rmse_mm"][0], "est_rmse_y": extra["est_rmse_mm"][1],
                            "est_rmse_z": extra["est_rmse_mm"][2]} or {})
        log(f"  {r['variant']:8s}: contrast {r['defect_contrast']:.3f} (true {r['true_contrast']:.3f}), "
            f"recovery {r['recovery']:.3f}, remote CV {r['remote_cv_pct']:.1f}%, false {r['false_defect']:.3f}")
    tag = f"task_{config}_{scen}_r{real}"
    (out / f"{tag}.json").write_text(json.dumps(rows, indent=2))
    np.savez_compressed(out / f"{tag}.npz", **shots)


# ---- collect ----------------------------------------------------------------------------------------
def collect(out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    from scipy.ndimage import gaussian_filter

    rows, shots = [], {}
    for c, s, r in tasks():
        f = out / f"task_{c}_{s}_r{r}.json"
        if not f.exists():
            log(f"missing: {f.name}")
            continue
        rows.extend(json.loads(f.read_text()))
        shots[(c, s, r)] = dict(np.load(out / f"task_{c}_{s}_r{r}.npz"))
    if not rows:
        raise SystemExit("no task results found")
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with open(out / "motion_metrics.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    configs = [c for c in C.RECON_CONFIGS if any(r["config"] == c for r in rows)]
    cats = [("static", "none", "no motion")] + [("exercise", v, lab) for v, lab in VARIANTS]

    def stat(c, s, v, k):
        vals = [r[k] for r in rows if r["config"] == c and r["scenario"] == s and r["variant"] == v]
        return (float(np.mean(vals)), float(np.std(vals)), len(vals)) if vals else (np.nan, np.nan, 0)

    fig, axs = plt.subplots(2, 2, figsize=(9.6, 7.4))
    # A: motion traces
    ax = axs[0, 0]
    E.style(ax)
    tf = next(iter(sorted(out.glob(f"trace_{configs[0]}_exercise_r0.npz"))), None)
    if tf is not None:
        d = np.load(tf)
        win = d["mids"] < 20
        for a, (lab, col) in enumerate((("x (lateral)", "#2a78d6"), ("z (craniocaudal)", "#e34948"))):
            ai = 0 if a == 0 else 2
            ax.plot(d["mids"][win], d["true"][ai][win] + (8 if a == 0 else -8), color=col, lw=1.4, label=f"{lab}, true")
            ax.plot(d["mids"][win], d["est"][ai][win] + (8 if a == 0 else -8), color=col, lw=0.8, ls=(0, (2, 1)),
                    label=f"{lab}, estimated")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Heart displacement (mm, offset for clarity)")
        ax.legend(fontsize=6.5, frameon=False, ncol=2, loc="upper center")
        ax.set_title(f"A.  Heart motion, {E.NAME[configs[0]]}", loc="left", fontweight="bold", fontsize=9)
    panels = [(axs[0, 1], "defect_contrast", "Defect contrast", "B"), (axs[1, 0], "recovery", "Myocardial recovery", "C"),
              (axs[1, 1], "false_defect", "Largest false defect, remote wall", "D")]
    x = np.arange(len(cats))
    wdt = 0.8 / len(configs)
    alphas = [1.0, 0.35, 0.55, 0.75, 1.0]
    for ax, k, lab, letter in panels:
        E.style(ax)
        for n, c in enumerate(configs):
            for j, (s, v, _) in enumerate(cats):
                mval, sd, nn = stat(c, s, v, k)
                ax.bar(j + (n - (len(configs) - 1) / 2) * wdt, mval, wdt * 0.9, yerr=sd if nn > 1 else None,
                       color=E.COL[c], alpha=alphas[j], capsize=2, error_kw=dict(lw=0.8))
        if k == "defect_contrast":
            ax.axhline(R.DEFECT_DEPTH, color=E.INK2, lw=0.9)
        if k == "false_defect":
            ax.axhline(R.DEFECT_DEPTH, color="#e34948", lw=0.9, ls=(0, (4, 2)))
        ax.set_xticks(x, [lab_.replace(" ", "\n", 1) for _, _, lab_ in cats], fontsize=7.5)
        ax.set_ylabel(lab, fontsize=8)
        ax.set_title(f"{letter}.  {lab}", loc="left", fontweight="bold", fontsize=9)
    fig.legend(handles=[Patch(color=E.COL[c], label=E.NAME[c]) for c in configs], loc="lower center",
               ncol=len(configs), frameon=False, fontsize=8)
    fig.suptitle(f"{BODY} body, 20% mid-anterior defect, exercise motion (breathing 30/min 15 mm, pedalling 60 rpm, "
                 f"5 mm drift); {E.CTRNAME[CTR]}, OSEM {rows[0]['iters']}x{rows[0]['subsets']}", fontsize=8.5)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    for ext in ("png", "pdf"):
        fig.savefig(out / f"Fig_motion_metrics.{ext}", dpi=300)
    plt.close(fig)

    fig, axs = plt.subplots(len(configs), len(cats), figsize=(2.0 * len(cats), 2.15 * len(configs)), squeeze=False)
    for i, c in enumerate(configs):
        for j, (s, v, lab) in enumerate(cats):
            ax = axs[i, j]
            ax.set_xticks([])
            ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_visible(False)
            d = shots.get((c, s, 0))
            if d is not None and v in d:
                ax.imshow(gaussian_filter(d[v], E.DISPLAY_FWHM_MM / 2.3548), cmap="gray_r", vmin=0, vmax=1.15,
                          origin="lower")
            if i == 0:
                ax.set_title(lab, fontsize=8)
            if j == 0:
                ax.set_ylabel(f"{'AB'[i]}.  {E.NAME[c]}", fontsize=8, fontweight="bold")
    fig.suptitle(f"Mid short axis, {BODY}, {E.CTRNAME[CTR]}; 20% defect at 12 o'clock", fontsize=8.5)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(out / f"Fig_motion_images.{ext}", dpi=300)
    plt.close(fig)

    lines = [f"Stage 6 motion: {BODY}, 20% defect, {T_SCAN:.0f} s, {E.CTRNAME[CTR]}, OSEM {rows[0]['iters']}x{rows[0]['subsets']}",
             f"{'config':12s} {'case':22s} {'n':>2s} {'contrast':>13s} {'recovery':>13s} {'remCV%':>6s} {'false':>6s} {'counts':>7s}"]
    for c in configs:
        for s, v, lab in cats:
            mc, sc, nn = stat(c, s, v, "defect_contrast")
            mr, sr, _ = stat(c, s, v, "recovery")
            cv, _, _ = stat(c, s, v, "remote_cv_pct")
            fd, _, _ = stat(c, s, v, "false_defect")
            nu, _, _ = stat(c, s, v, "n_used")
            nt, _, _ = stat(c, s, v, "n_trues")
            lines.append(f"{c:12s} {lab:22s} {nn:2d} {mc:6.3f}±{sc:5.3f} {mr:6.3f}±{sr:5.3f} {cv:6.1f} {fd:6.3f} {nu / nt:7.2f}")
        ex = [r for r in rows if r["config"] == c and r["variant"] == "mc_data"]
        if ex:
            lines.append(f"{c:12s} motion estimate RMS error x/y/z (mm): "
                         + "/".join(f"{np.mean([r[f'est_rmse_{a}'] for r in ex]):.1f}" for a in "xyz"))
    (out / "SUMMARY_motion.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--recon", required=True)
    ap.add_argument("--task", type=int, default=None)
    ap.add_argument("--collect", action="store_true")
    ap.add_argument("--ntasks", action="store_true")
    ap.add_argument("--iters", type=int, default=8)
    ap.add_argument("--subsets", type=int, default=C.RECON_SUBSETS)
    ap.add_argument("--seed", type=int, default=61)
    ap.add_argument("--n-trues-ref", type=float, default=C.FASTMC_TRUES)
    args = ap.parse_args()
    if args.ntasks:
        print(len(tasks(new_only=True)))
        return
    recon_dir = Path(args.recon)
    out = recon_dir / ("motion" if CTR == 200 else f"motion_{CTR}")
    out.mkdir(exist_ok=True)
    if args.collect:
        collect(out)
    elif args.task is not None:
        c, s, r = tasks(new_only=True)[args.task]
        run_task(recon_dir, out, c, s, r, args.iters, args.subsets, args.seed, args.n_trues_ref)
    else:
        raise SystemExit("give --task N or --collect")


if __name__ == "__main__":
    main()

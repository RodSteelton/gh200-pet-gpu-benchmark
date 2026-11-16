"""
Stage 9: exercise motion with the flat-panel designs, and camera-based motion correction.

    python motion_panels.py --recon /scratch/$USER/exercise_pet_recon --prep N      sensitivity + calibration, geometry N
    python motion_panels.py --recon /scratch/$USER/exercise_pet_recon --task N      one (geometry, scenario, scan)
    python motion_panels.py --recon /scratch/$USER/exercise_pet_recon --collect     figures and tables
    python motion_panels.py --recon DIR --smoke                                     1 prep + 2 tasks at 2% counts + collect

Same body, defect, backrest, scan time and motion model as stage 6 (motion.py): P90, 20% mid-anterior defect,
carbon backrest, 120 s; breathing 30/min 15 mm, pedaling 60 rpm, 5 mm drift; heart moves rigidly.
Counts: equal injected activity (the stage-2 open-ring decays) times WIN = 0.65, the energy-window factor from
the stage-10 benchmark (ICS summed, 3 x 3 x 20 mm LYSO), so trues are realistic for 400 MBq-class scans.
Geometries and TOF (CFG9): 290 deg ring, 4 panels 30 x 24 and 30 x 20 cm (two front, two back), Anto's layout
(front, back and two lateral panels outside the arms, 30 x 24 cm) at 150 ps; 2 panels 30 x 24 cm at 100 and 150 ps.
Corrections (exercise scans):
  none       no correction
  gated      end-expiration gate (true breathing signal, ~50% of counts)
  mc_data    data-driven rigid motion from TOF positions near the heart (stage 6 method)
  cam_good   optical camera, 30 Hz, noise 0.5/0.5/1.0 mm (x/y/z), marker moves exactly with the heart
  cam_poor   optical camera, noise 1/1/4 mm and marker-to-heart gain error 20% per axis (i3M bench test and
             chest-marker numbers: z sigma 2 to 6 mm)
  hybrid     cam_poor shape with its gain calibrated per axis against mc_data, then inverse-variance fusion with mc_data
  mc_true    true motion (upper bound)
Every image is scored with and without the pose-matched uniform-LV calibration of stage 8 (computed at the base
pose in --prep), against the truth at the mean heart position.
Output <recon>/motion9/: task_*.json/.npz, trace_*.npz, motion9_metrics.csv, Fig_motion9_metrics,
Fig_motion9_images, Fig_motion9_traces, SUMMARY_motion9.txt.
"""

import argparse
import csv
import json
import os
import time
from pathlib import Path

import numpy as np

import geometry as G
import evaluate as E
import fastmc
import robust as R
import robust2 as R2
import backrest as B
import panels as P
import motion as M
from projector import fwd
from run_recon import osem, log

C = G.C
BODY = R.TEST_BODY
WIN = float(os.environ.get("XPET_WIN", 0.65))
CFG9 = [("R280_slot", (150,)), ("P4_T30A24", (150,)), ("P4_T30A20", (150,)), ("P4_ANTO", (150,)),
        ("P2_T30A24", (100, 150))]
NREAL = {"static": int(os.environ.get("XPET_N_STATIC", 5)), "exercise": int(os.environ.get("XPET_N_EXERCISE", 8))}
VARIANTS = [("none", "no correction"), ("gated", "respiratory gate"), ("mc_data", "data-driven"),
            ("cam_good", "camera, good"), ("cam_poor", "camera, poor"), ("hybrid", "camera + data"),
            ("mc_true", "ideal")]
CAM = {"cam_good": dict(sig=(0.5, 0.5, 1.0), gain_sd=0.0), "cam_poor": dict(sig=(1.0, 1.0, 4.0), gain_sd=0.2)}
CAM_HZ = 30.0
N_CAL = float(os.environ.get("XPET_N_CAL", 50e6))
OUTDIR = "motion9"


def configs():
    return [c for c, _ in CFG9]


def ctrs_of(config):
    return dict(CFG9)[config]


def tasks():
    return [(c, s, r) for c in configs() for s in ("static", "exercise") for r in range(NREAL[s])]


def sens_path(recon_dir, config):
    if config in P.PANEL_CONFIGS:
        return Path(recon_dir) / f"sens_{config}_{BODY}_backrest_carbon_v9.npy"
    return Path(recon_dir) / f"sens_{config}_{BODY}_backrest_carbon_v3.npy"


def mu_backrest(grid, ph):
    mi = [m[0] for m in B.MATERIALS].index("carbon")
    return B.mu_with_backrest(grid, ph, mi)


# ---------------------------------------------------------------- workers (picklable)
def _emit9(config, ph, mu, n_emit, seed, trace, chunk=2_000_000):
    scn = P.scanner_for(config)
    grid = P.grid_for(scn)
    rng = np.random.default_rng(seed)
    out = {"p1": [], "p2": [], "dt": [], "t": []}
    done = 0
    while done < n_emit:
        n = min(chunk, n_emit - done)
        e = fastmc.sample_emissions(ph, n, rng)
        tt = rng.uniform(0, M.T_SCAN, len(e))
        if trace is not None:
            for a in range(3):
                e[:, a] += np.interp(tt, M.TG, trace[a])
        done += n
        d = fastmc._iso(rng, len(e))
        q1, ok1 = scn.detect(e, d, rng)
        q2, ok2 = scn.detect(e, -d, rng)
        ok = ok1 & ok2
        q1, q2, e, tt = q1[ok], q2[ok], e[ok], tt[ok]
        att = np.exp(-fwd(mu, grid, q1.astype(np.float32), q2.astype(np.float32)))
        keep = rng.random(len(att)) < att
        q1, q2, e, tt = q1[keep], q2[keep], e[keep], tt[keep]
        s1, _ = scn.snap(q1)
        s2, _ = scn.snap(q2)
        dt = (np.linalg.norm(q1 - e, axis=1) - np.linalg.norm(q2 - e, axis=1)) / C.C_MM_PER_PS
        for k_, v in (("p1", s1), ("p2", s2), ("dt", dt.astype(np.float32)), ("t", tt.astype(np.float32))):
            out[k_].append(v)
    return {k_: np.concatenate(v) for k_, v in out.items()}


def _cal9(config, ph, mu, n, seed):
    scn = P.scanner_for(config)
    grid = P.grid_for(scn)
    return fastmc.generate_trues(scn, grid, ph, mu, int(n), np.random.default_rng(seed), chunk=P.CHUNK, log=lambda m: None)


def _sens9(config, mu, n, seed):
    scn = P.scanner_for(config)
    grid = P.grid_for(scn)
    rng = np.random.default_rng(seed)
    if config in P.PANEL_CONFIGS:
        return P.panel_sensitivity(scn, grid, mu, int(n), rng)
    return fastmc.sensitivity(scn, grid, mu, int(n), rng, log=lambda m: None)


# ---------------------------------------------------------------- prep: sensitivity and pose-matched calibration
def prep(recon_dir, out, config, iters, subsets, seed, frac=1.0):
    scn = P.scanner_for(config)
    grid = P.grid_for(scn)
    R.set_pose((0, 0, 0), None)
    ph_uni = R.phantom(BODY, defect=False)
    mu = mu_backrest(grid, ph_uni)
    sp = sens_path(recon_dir, config)
    t0 = time.time()
    if sp.exists() and frac == 1.0:
        sens = np.load(sp)
        log(f"[{config}] sensitivity loaded: {sp.name}")
    else:
        n_s = int(C.SENS_DETECTED * frac)
        k = max(1, min(M.WORKERS, int(np.ceil(n_s / 1e6))))
        parts = M.pool_map(_sens9, [(config, mu, n_s // k, seed + 11 * i) for i in range(k)])
        sens = np.sum(parts, axis=0).astype(np.float32)
        np.save(sp if frac == 1.0 else out / f"sens_{config}_smoke.npy", sens)
        log(f"[{config}] sensitivity from {n_s:,} events in {time.time() - t0:.0f} s: {sp.name}")
    n_cal = int(N_CAL * frac)
    k = max(1, min(M.WORKERS, int(np.ceil(n_cal / 2e6))))
    t0 = time.time()
    parts = M.pool_map(_cal9, [(config, ph_uni, mu, n_cal // k, seed + 500 + 7 * i) for i in range(k)])
    ev = R2.merge(parts)
    del parts
    log(f"[{config}] {len(ev['p1']):,} uniform-LV trues in {time.time() - t0:.0f} s")
    t_uni = R.truth(grid, ph_uni)
    cal = {}
    for ctr in ctrs_of(config):
        _, p, s = R.measure(R.recon(ev, grid, sens, ctr, iters, subsets, seed + 50), t_uni)
        cal[f"p_{ctr}"], cal[f"s_{ctr}"] = p, s
    np.savez(out / f"cal_{config}{'' if frac == 1.0 else '_smoke'}.npz", **cal)
    log(f"[{config}] calibration saved")


# ---------------------------------------------------------------- camera models
def camera(trace, real, sig, gain_sd, seed_off):
    """Camera estimate of the heart displacement on M.TG (mean removed)."""
    rng = np.random.default_rng(7000 + real + seed_off)
    tc = np.arange(0.0, M.T_SCAN + 1e-9, 1.0 / CAM_HZ)
    gain = 1.0 + gain_sd * rng.normal(size=3)
    cam = np.stack([gain[a] * np.interp(tc, M.TG, trace[a]) + sig[a] * rng.normal(size=len(tc)) for a in range(3)])
    k = 3                                                          # 0.1 s moving average
    cam = np.stack([np.convolve(c, np.ones(k) / k, mode="same") for c in cam])
    cam -= cam.mean(1, keepdims=True)
    return np.stack([np.interp(M.TG, tc, cam[a]) for a in range(3)]), gain


def hybrid(cam_tg, mids, est, sig_cam):
    """Gain of the camera trace fitted to the data-driven estimate, then inverse-variance fusion (per axis)."""
    c = np.stack([np.interp(mids, M.TG, cam_tg[a]) for a in range(3)])
    out = np.empty_like(est)
    info = []
    for a in range(3):
        g = float((c[a] * est[a]).sum() / max((c[a] * c[a]).sum(), 1e-9))
        g = min(max(g, 0.5), 1.5)
        vc = (g * sig_cam[a]) ** 2 / 3.0
        vd = max(float(np.var(est[a] - g * c[a])) - vc, 0.01)
        w = (1 / vc) / (1 / vc + 1 / vd)
        out[a] = w * g * c[a] + (1 - w) * est[a]
        info.append((g, w))
    return out, info


# ---------------------------------------------------------------- one task
def score2(img, grid, shift, cal, ctr):
    t, prof_t = M.truth_at(grid, shift)
    _, prof, seg = R.measure(img, t)
    m = R.metrics(prof, seg, prof_t)
    m["recovery"] = float(np.nanmean(np.delete(seg, R.DEF_SEG)))
    pc, sc = R2.correct(prof, seg, (cal[f"p_{ctr}"], cal[f"s_{ctr}"]))
    mc = R.metrics(pc, sc, prof_t)
    m.update({f"{k}_pc": v for k, v in mc.items()})
    R.set_pose(tuple(shift), None)
    sa = E.reslice(E.normalise(img, t), grid, "sa")
    R.set_pose((0, 0, 0), None)
    return m, sa


def run_task(recon_dir, out, config, scen, real, iters, subsets, seed, frac=1.0):
    scn = P.scanner_for(config)
    grid = P.grid_for(scn)
    R.set_pose((0, 0, 0), None)
    ph = R.phantom(BODY, defect=True)
    mu = mu_backrest(grid, ph)
    cal = dict(np.load(out / f"cal_{config}{'' if frac == 1.0 else '_smoke'}.npz"))
    sp = sens_path(recon_dir, config)
    sens = np.load(sp if sp.exists() else out / f"sens_{config}_smoke.npy")
    ref = np.load(recon_dir / f"events_fastmc_R280_slot_{BODY}_{int(C.FASTMC_TRUES)}.npz")
    n_emit = int(int(ref["n_emitted"]) * WIN * frac)
    v3 = C.VOXEL_MM ** 3
    heart = np.isin(ph["labels"], [C.LABELS["myocardium"], C.LABELS["blood_pool"]])
    qh, qb = dict(ph), dict(ph)
    qh["activity"] = (ph["activity"] * heart).astype(np.float32)
    qh["a_torso"], qh["a_arm_each"] = np.float64(qh["activity"].sum() * v3), np.float64(0.0)
    qb["activity"] = (ph["activity"] * ~heart).astype(np.float32)
    qb["a_torso"] = np.float64(qb["activity"].sum() * v3)
    a_h, a_b = float(qh["a_torso"]), float(qb["a_torso"]) + 2 * float(ph["a_arm_each"])
    n_h = int(round(n_emit * a_h / (a_h + a_b)))
    trace, resp = M.motion_trace(real) if scen == "exercise" else (np.zeros((3, len(M.TG))), np.zeros(len(M.TG)))
    rng = np.random.default_rng(seed + 100 * real + (0 if scen == "static" else 7))
    jobs = []
    for q, n_tot, tr in ((qh, n_h, trace), (qb, n_emit - n_h, None)):
        kk = max(1, M.WORKERS // 2)
        base = n_tot // kk
        for i in range(kk):
            jobs.append((config, q, mu, base + (1 if i < n_tot - base * kk else 0), int(rng.integers(1 << 31)), tr))
    t0 = time.time()
    parts = M.pool_map(_emit9, jobs)
    ev = {key: np.concatenate([p[key] for p in parts]) for key in ("p1", "p2", "dt", "t")}
    del parts
    n_ev = len(ev["t"])
    log(f"[{config} {scen} r{real}] {n_ev:,} trues from {n_emit:,} decays (x{WIN} window) in {time.time() - t0:.0f} s")
    dtrue = np.stack([np.interp(ev["t"], M.TG, trace[a]) for a in range(3)], 1).astype(np.float32)
    mean_d = dtrue.mean(0)
    D = np.linalg.norm(ev["p2"] - ev["p1"], axis=1)
    rows, shots, extra = [], {}, {}
    base_row = {"config": config, "scenario": scen, "real": real, "iters": iters, "subsets": subsets,
                "n_trues": n_ev, "n_emitted": n_emit, "win": WIN}
    shifts = {}
    if scen == "exercise":
        gate = np.interp(ev["t"], M.TG, resp) < 0.25
    for ctr in ctrs_of(config):
        sig_t = ctr / 2.3548
        sigma = C.C_MM_PER_PS * sig_t / 2.0
        rng_r = np.random.default_rng(seed + 31 + 100 * real + ctr)
        tc = ((D + C.C_MM_PER_PS * (ev["dt"] + rng_r.normal(size=n_ev).astype(np.float32) * sig_t)) / 2.0).astype(np.float32)

        def rec(sel=None, shift=None):
            p1, p2, t_ = ev["p1"], ev["p2"], tc
            if sel is not None:
                p1, p2, t_ = p1[sel], p2[sel], t_[sel]
            if shift is not None:
                s = shift if sel is None else shift[sel]
                p1, p2 = p1 - s, p2 - s
            return osem({"p1": p1, "p2": p2, "tc": t_}, grid, sens, sigma, iters, subsets, np.random.default_rng(seed + 77))

        br = {**base_row, "ctr_ps": ctr}
        img0 = rec()
        m, sa = score2(img0, grid, mean_d, cal, ctr)
        rows.append({**br, "variant": "none", "n_used": n_ev, **m})
        shots[f"none_{ctr}"] = sa
        if scen != "exercise":
            continue
        img = rec(sel=gate)
        m, sa = score2(img, grid, dtrue[gate].mean(0), cal, ctr)
        rows.append({**br, "variant": "gated", "n_used": int(gate.sum()), **m})
        shots[f"gated_{ctr}"] = sa
        mids, est, f_h = M.estimate_motion(ev, tc, img0, grid)
        tz = np.stack([np.interp(mids, M.TG, trace[a]) for a in range(3)])
        tz -= tz.mean(1, keepdims=True)
        ests = {"mc_data": est}
        for v in ("cam_good", "cam_poor"):
            cam_tg, gain = camera(trace, real, CAM[v]["sig"], CAM[v]["gain_sd"], 13 if v == "cam_good" else 29)
            ests[v] = np.stack([np.interp(mids, M.TG, cam_tg[a]) for a in range(3)])
            if v == "cam_poor":
                ests["hybrid"], hinfo = hybrid(cam_tg, mids, est, CAM[v]["sig"])
                extra[f"hybrid_gain_w_{ctr}"] = hinfo
                extra["cam_poor_gain"] = gain.tolist()
        for v, e_ in ests.items():
            rmse = np.sqrt(((e_ - tz) ** 2).mean(1))
            extra[f"rmse_{v}_{ctr}"] = rmse.tolist()
            dest = np.stack([np.interp(ev["t"], mids, e_[a]) for a in range(3)], 1).astype(np.float32)
            img = rec(shift=dest)
            m, sa = score2(img, grid, mean_d, cal, ctr)
            rows.append({**br, "variant": v, "n_used": n_ev, "rmse_x": rmse[0], "rmse_y": rmse[1], "rmse_z": rmse[2], **m})
            shots[f"{v}_{ctr}"] = sa
        img = rec(shift=dtrue)
        m, sa = score2(img, grid, (0, 0, 0), cal, ctr)
        rows.append({**br, "variant": "mc_true", "n_used": n_ev, **m})
        shots[f"mc_true_{ctr}"] = sa
        np.savez_compressed(out / f"trace_{config}_{scen}_r{real}_{ctr}.npz", mids=mids, true=tz,
                            **{k: v for k, v in ests.items()})
        extra[f"f_heart_{ctr}"] = f_h
    for r in rows:
        log(f"  {r['ctr_ps']} ps {r['variant']:8s}: contrast {r['defect_contrast']:.3f} / corrected "
            f"{r['defect_contrast_pc']:.3f} (true {r['true_contrast']:.3f}), recovery {r['recovery']:.3f}, "
            f"remote CV {r['remote_cv_pct']:.1f}%, false {r['false_defect_pc']:.3f}"
            + (f", motion RMS {r['rmse_x']:.1f}/{r['rmse_y']:.1f}/{r['rmse_z']:.1f} mm" if "rmse_x" in r else ""))
    tag = f"task_{config}_{scen}_r{real}"
    (out / f"{tag}.json").write_text(json.dumps({"rows": rows, "extra": extra}, indent=1, default=float))
    np.savez_compressed(out / f"{tag}.npz", **shots)


# ---------------------------------------------------------------- collect
def collect(out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    from scipy.ndimage import gaussian_filter

    rows, shots, extras = [], {}, {}
    for c, s, r in tasks():
        f = out / f"task_{c}_{s}_r{r}.json"
        if not f.exists():
            log(f"missing: {f.name}")
            continue
        d = json.loads(f.read_text())
        rows.extend(d["rows"])
        extras[(c, s, r)] = d["extra"]
        if r == 0:
            shots[(c, s)] = dict(np.load(out / f"task_{c}_{s}_r{r}.npz"))
    if not rows:
        raise SystemExit("no task results found")
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with open(out / "motion9_metrics.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    series = [(c, ctr) for c, cc in CFG9 for ctr in cc if any(r["config"] == c and r["ctr_ps"] == ctr for r in rows)]
    cats = [("static", "none", "no motion")] + [("exercise", v, lab) for v, lab in VARIANTS]

    def vals(c, ctr, s, v, k):
        return [r[k] for r in rows if r["config"] == c and r["ctr_ps"] == ctr and r["scenario"] == s and r["variant"] == v and k in r]

    def stat(c, ctr, s, v, k):
        x = vals(c, ctr, s, v, k)
        return (float(np.mean(x)), float(np.std(x)), len(x)) if x else (np.nan, np.nan, 0)

    pal = ["#2a78d6", "#1f8a5c", "#0b5f61", "#c43c8b", "#eb6834", "#f2a541"]
    col = {sc: pal[i % len(pal)] for i, sc in enumerate(series)}
    name = {sc: f"{P.LABEL.get(sc[0], E.NAME.get(sc[0], sc[0]))}, {sc[1]} ps" for sc in series}
    name[("R280_slot", 150)] = "56 cm ring, 290°, 150 ps"

    fig, axs = plt.subplots(2, 2, figsize=(12, 8.6))
    panels = [(axs[0, 0], "defect_contrast_pc", "Defect contrast (pose-corrected)", "A", R.DEFECT_DEPTH),
              (axs[0, 1], "recovery", "Myocardial recovery", "B", None),
              (axs[1, 0], "remote_cv_pct_pc", "Remote wall CV (%, pose-corrected)", "C", None),
              (axs[1, 1], "false_defect_pc", "Largest false defect (pose-corrected)", "D", R.DEFECT_DEPTH)]
    x = np.arange(len(cats))
    wdt = 0.84 / len(series)
    for ax, k, lab, letter, ref in panels:
        E.style(ax)
        for n, sc in enumerate(series):
            for j, (s, v, _) in enumerate(cats):
                mval, sd, nn = stat(*sc, s, v, k)
                ax.bar(j + (n - (len(series) - 1) / 2) * wdt, mval, wdt * 0.9, yerr=sd if nn > 1 else None,
                       color=col[sc], capsize=1.5, error_kw=dict(lw=0.6))
        if ref is not None:
            ax.axhline(ref, color="#e34948" if letter == "D" else E.INK2, lw=0.9, ls=(0, (4, 2)) if letter == "D" else "-")
        ax.set_xticks(x, [l_.replace(", ", "\n").replace(" ", "\n", 1) if "," not in l_ else l_.replace(", ", "\n")
                          for _, _, l_ in cats], fontsize=7.5)
        ax.set_ylabel(lab, fontsize=8)
        ax.set_title(f"{letter}. {lab}", loc="left", fontweight="bold", fontsize=9)
    fig.legend(handles=[Patch(color=col[sc], label=name[sc]) for sc in series], loc="lower center",
               ncol=3, frameon=False, fontsize=8)
    fig.suptitle(f"Stage 9: {BODY}, 20% mid-anterior defect, exercise motion (breathing 30/min 15 mm, pedaling 60 rpm, "
                 f"5 mm drift); counts x{WIN} energy window; OSEM {rows[0]['iters']}x{rows[0]['subsets']}; mean ± SD over scans",
                 fontsize=8.5)
    fig.tight_layout(rect=(0, 0.06, 1, 0.97))
    for ext in ("png", "pdf"):
        fig.savefig(out / f"Fig_motion9_metrics.{ext}", dpi=300)
    plt.close(fig)

    # images: rows = series, columns = categories (scan 0)
    fig, axs = plt.subplots(len(series), len(cats), figsize=(1.75 * len(cats), 1.9 * len(series)), squeeze=False)
    for i, sc in enumerate(series):
        for j, (s, v, lab) in enumerate(cats):
            ax = axs[i, j]
            ax.set_xticks([])
            ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_visible(False)
            d = shots.get((sc[0], s))
            key = f"{v}_{sc[1]}"
            if d is not None and key in d:
                im = d[key]
                ax.imshow(gaussian_filter(im / max(np.percentile(im, 99.5), 1e-9), E.DISPLAY_FWHM_MM / 2.3548),
                          cmap="gray_r", vmin=0, vmax=1.15, origin="lower")
            if i == 0:
                ax.set_title(lab, fontsize=7.5)
            if j == 0:
                ax.set_ylabel(f"{chr(65 + i)}. {P.SHORT.get(sc[0], sc[0]).replace(chr(10), ' ')}\n{sc[1]} ps", fontsize=7.5,
                              fontweight="bold")
    fig.suptitle(f"Mid short axis, scan 1; 20% defect at 12 o'clock", fontsize=8.5)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(out / f"Fig_motion9_images.{ext}", dpi=250)
    plt.close(fig)

    # traces: true vs estimates, first 20 s, z axis, first series with exercise data
    tf = sorted(out.glob(f"trace_{series[-1][0] if False else series[0][0]}_exercise_r0_*.npz"))
    if tf:
        d = np.load(tf[0])
        win = d["mids"] < 20
        fig, axs = plt.subplots(2, 1, figsize=(8, 5.5), sharex=True)
        for ax, a, lab in ((axs[0], 0, "x (lateral)"), (axs[1], 2, "z (craniocaudal)")):
            E.style(ax)
            ax.plot(d["mids"][win], d["true"][a][win], color="k", lw=1.6, label="true")
            for v, c_ in (("mc_data", "#2a78d6"), ("cam_poor", "#e34948"), ("hybrid", "#1f8a5c")):
                ax.plot(d["mids"][win], d[v][a][win], color=c_, lw=0.8, label=dict(VARIANTS)[v])
            ax.set_ylabel(f"{lab} (mm)", fontsize=8)
        axs[0].legend(fontsize=7, frameon=False, ncol=4)
        axs[0].set_title(f"A. Lateral", loc="left", fontweight="bold", fontsize=9)
        axs[1].set_title(f"B. Craniocaudal", loc="left", fontweight="bold", fontsize=9)
        axs[1].set_xlabel("Time (s)")
        fig.suptitle(f"Heart motion estimates, {name[series[0]]}, exercise scan 1", fontsize=8.5)
        fig.tight_layout()
        for ext in ("png", "pdf"):
            fig.savefig(out / f"Fig_motion9_traces.{ext}", dpi=250)
        plt.close(fig)

    L = [f"Stage 9 motion with panels: {BODY}, 20% defect, {M.T_SCAN:.0f} s, counts x{WIN} (energy window), "
         f"OSEM {rows[0]['iters']}x{rows[0]['subsets']}; values: raw / pose-corrected; mean over scans",
         f"{'geometry':34s} {'case':16s} {'n':>2s} {'contrast':>15s} {'recovery':>8s} {'remCV%':>11s} {'false':>11s} {'counts':>6s} {'motion RMS x/y/z mm':>20s}"]
    for sc in series:
        for s, v, lab in cats:
            mc, sdc, nn = stat(*sc, s, v, "defect_contrast")
            mcp, _, _ = stat(*sc, s, v, "defect_contrast_pc")
            mr, _, _ = stat(*sc, s, v, "recovery")
            cv, _, _ = stat(*sc, s, v, "remote_cv_pct")
            cvp, _, _ = stat(*sc, s, v, "remote_cv_pct_pc")
            fd, _, _ = stat(*sc, s, v, "false_defect")
            fdp, _, _ = stat(*sc, s, v, "false_defect_pc")
            nu, _, _ = stat(*sc, s, v, "n_used")
            nt, _, _ = stat(*sc, s, v, "n_trues")
            rm = "/".join(f"{stat(*sc, s, v, f'rmse_{a}')[0]:.1f}" for a in "xyz") if v.startswith(("mc_d", "cam", "hyb")) else ""
            L.append(f"{name[sc]:34s} {lab:16s} {nn:2d} {mc:6.3f}/{mcp:6.3f}±{sdc:4.2f} {mr:8.3f} {cv:5.1f}/{cvp:5.1f} "
                     f"{fd:5.3f}/{fdp:5.3f} {nu / max(nt, 1):6.2f} {rm:>20s}")
        nt, _, _ = stat(*sc, "static", "none", "n_trues")
        L.append(f"{name[sc]:34s} trues per 120 s scan: {nt / 1e6:.2f} M")
    L.append("")
    L.append(f"Camera models: good = noise {CAM['cam_good']['sig']} mm, no gain error; poor = noise {CAM['cam_poor']['sig']} mm, "
             f"gain SD {CAM['cam_poor']['gain_sd']}; {CAM_HZ:.0f} Hz, 0.1 s smoothing. Hybrid: camera gain fitted to the data "
             f"estimate per axis, inverse-variance fusion.")
    L.append("Caveats: rigid heart motion (no cardiac contraction or body deformation); perfect breathing signal for the gate; "
             "trues only (scaled by the window factor); lateral panels of P4_ANTO outside the arms (arms down).")
    (out / "SUMMARY_motion9.txt").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))


# ---------------------------------------------------------------- main
def main():
    global NREAL
    ap = argparse.ArgumentParser()
    ap.add_argument("--recon", required=True)
    ap.add_argument("--prep", type=int, default=None)
    ap.add_argument("--task", type=int, default=None)
    ap.add_argument("--collect", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--ntasks", action="store_true")
    ap.add_argument("--nprep", action="store_true")
    ap.add_argument("--iters", type=int, default=8)
    ap.add_argument("--subsets", type=int, default=C.RECON_SUBSETS)
    ap.add_argument("--seed", type=int, default=91)
    a = ap.parse_args()
    if a.ntasks:
        print(len(tasks()))
        return
    if a.nprep:
        print(len(CFG9))
        return
    recon_dir = Path(a.recon)
    out = recon_dir / OUTDIR
    out.mkdir(exist_ok=True)
    if a.collect:
        collect(out)
    elif a.prep is not None:
        prep(recon_dir, out, configs()[a.prep], a.iters, a.subsets, a.seed)
    elif a.task is not None:
        c, s, r = tasks()[a.task]
        run_task(recon_dir, out, c, s, r, a.iters, a.subsets, a.seed)
    elif a.smoke:
        frac = 0.02
        c = "P2_T30A24"
        prep(recon_dir, out, c, 2, a.subsets, a.seed, frac=frac)
        NREAL = {"static": 1, "exercise": 1}
        for s in ("static", "exercise"):
            run_task(recon_dir, out, c, s, 0, 2, a.subsets, a.seed, frac=frac)
        collect(out)
    else:
        raise SystemExit("give --prep N, --task N, --collect or --smoke")


if __name__ == "__main__":
    main()

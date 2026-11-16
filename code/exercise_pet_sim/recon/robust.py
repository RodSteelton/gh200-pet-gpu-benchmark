"""
Stage 3: robustness of the open ring to heart position, with a perfusion defect.

    python robust.py --recon /scratch/$USER/exercise_pet_recon

For each geometry and heart pose (base, 15 mm lateral, 20 mm cranial, 15 deg tilt, 15 deg rotation), the
design body (P90) gets a 20% transmural defect in the mid-anterior wall (60 deg sector, mid-ventricle).
Fast-MC trues are simulated and reconstructed (OSEM, --iters x 10 subsets) at 320 and 200 ps.
A fixed segment correction measured once on the P50 body with the base pose and no defect is applied.

Metrics: measured defect contrast (1 - defect / remote, mid-ventricular profile) vs true contrast;
remote-wall uniformity (CV of the 15 non-defect segments); largest false defect in the remote wall
(1 - min segment / mean). Writes <recon>/robust/: robust_metrics.csv/.json, Fig_robust_metrics,
Fig_robust_images, SUMMARY.txt. Sensitivity images are reused from stage 2 (attenuation change from the
moved heart is negligible).
"""

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np

import geometry as G
import evaluate as E
import fastmc
from run_recon import osem, log

C = G.C
import make_phantom as MP  # noqa: E402  (after geometry puts the project root on sys.path)

BASE_CENTER = np.array(C.LV_CENTER, float)
BASE_AXIS = np.array(C.LV_AXIS, float) / np.linalg.norm(C.LV_AXIS)
POSES = [("base", (0, 0, 0), None), ("lateral 15 mm", (15, 0, 0), None), ("cranial 20 mm", (0, 0, 20), None),
         ("tilt 15°", (0, 0, 0), ("x", 15)), ("rotate 15°", (0, 0, 0), ("z", -15))]
DEFECT_DEPTH, DEFECT_HALF_DEG, DEFECT_W = 0.20, 30.0, (0.0, 20.0)
DEF_BINS, REMOTE_BINS = [11, 0], list(range(2, 10))
DEF_SEG = 6                                   # mid-anterior segment
CAL_BODY, TEST_BODY = "P50", "P90"


def rot(axis, deg):
    a = np.deg2rad(deg)
    c, s = np.cos(a), np.sin(a)
    return {"x": np.array([[1, 0, 0], [0, c, -s], [0, s, c]]),
            "y": np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]]),
            "z": np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])}[axis]


def set_pose(shift, r):
    u = BASE_AXIS if r is None else rot(*r) @ BASE_AXIS
    C.LV_CENTER = tuple(BASE_CENTER + np.array(shift, float))
    C.LV_AXIS = tuple(u)


def phantom(body, defect):
    """Phantom dict for the current pose (and optional defect), in the format of load_body()."""
    base = G.load_body(body)
    lab, act, (x, y, z) = MP.build(MP.body_dims(C.BODIES[body]["outer"]), C.BODIES[body]["outer"])
    if defect:
        c, u, e1, e2 = G.lv_frame()
        Z, Y, X = np.meshgrid(z, y, x, indexing="ij")
        P = np.stack([X - c[0], Y - c[1], Z - c[2]], -1)
        w = P @ u
        psi = np.degrees(np.arctan2(P @ e2, P @ e1))
        m = (lab == C.LABELS["myocardium"]) & (w >= DEFECT_W[0]) & (w < DEFECT_W[1]) & (np.abs(psi) <= DEFECT_HALF_DEG)
        act[m] *= 1.0 - DEFECT_DEPTH
    ph = dict(base)
    ph.update(labels=lab.astype(np.uint8), activity=act.astype(np.float32), x=x, y=y, z=z,
              a_torso=np.float64(act.sum() * C.VOXEL_MM ** 3))
    return ph


def truth(grid, ph):
    act, lab = G.truth(grid, ph)
    w, _, psi = G.lv_coords(grid)
    return dict(grid=grid, act=act, lab=lab, w=w, psi=psi)


def measure(img, t):
    img = E.normalise(img, t)
    _, prof = E.mid_profile(img, t)
    return img, prof, E.segment_values(img, t)


def metrics(prof, seg, prof_t):
    remote = np.delete(seg, DEF_SEG)
    contrast = 1 - np.nanmean(prof[DEF_BINS]) / np.nanmean(prof[REMOTE_BINS])
    true_c = 1 - np.nanmean(prof_t[DEF_BINS]) / np.nanmean(prof_t[REMOTE_BINS])
    return {"defect_contrast": float(contrast), "true_contrast": float(true_c),
            "remote_cv_pct": float(100 * np.nanstd(remote) / np.nanmean(remote)),
            "false_defect": float(1 - np.nanmin(remote) / np.nanmean(remote))}


def recon(ev, grid, sens, ctr, iters, subsets, seed):
    rng = np.random.default_rng(seed + ctr)
    sig_t = ctr / 2.3548
    D = np.linalg.norm(ev["p2"] - ev["p1"], axis=1)
    dt = ev["dt"] + rng.normal(size=len(D)) * sig_t
    ev["tc"] = ((D + C.C_MM_PER_PS * dt) / 2.0).astype(np.float32)
    return osem(ev, grid, sens, C.C_MM_PER_PS * sig_t / 2.0, iters, subsets, rng)


def run(recon_dir, out, ctrs, iters, subsets, n_trues, seed):
    rows, shots = [], {}
    for config in C.RECON_CONFIGS:
        ring = G.Ring(config)
        grid = G.recon_grid(ring, 3.0)
        # calibration: P50, base pose, no defect (cached stage-2 events)
        set_pose((0, 0, 0), None)
        t_cal = truth(grid, G.load_body(CAL_BODY))
        sens_cal = np.load(recon_dir / f"sens_{config}_{CAL_BODY}_v3.npy")
        ev_cal = dict(np.load(recon_dir / f"events_fastmc_{config}_{CAL_BODY}_{int(n_trues)}.npz"))
        cal = {}
        for ctr in ctrs:
            log(f"[{config}] calibration {CAL_BODY}, CTR {ctr} ps")
            _, p, s = measure(recon(ev_cal, grid, sens_cal, ctr, iters, subsets, seed), t_cal)
            cal[ctr] = (p, s)
        del ev_cal
        sens = np.load(recon_dir / f"sens_{config}_{TEST_BODY}_v3.npy")
        for pi, (pose, shift, r) in enumerate(POSES):
            set_pose(shift, r)
            ph = phantom(TEST_BODY, defect=True)
            t = truth(grid, ph)
            _, prof_t = E.mid_profile(t["act"], t)
            mu = G.mu_map(grid, ph)
            rng = np.random.default_rng(seed + 100 * pi)
            t0 = time.time()
            ev = fastmc.generate_trues(ring, grid, ph, mu, int(n_trues), rng, log=lambda m: None)
            log(f"[{config}] {pose}: {len(ev['p1']):,} trues in {time.time() - t0:.0f} s")
            for ctr in ctrs:
                img, prof, seg = measure(recon(ev, grid, sens, ctr, iters, subsets, seed + pi), t)
                cp, cs = cal[ctr]
                base = {"config": config, "body": TEST_BODY, "pose": pose, "ctr_ps": ctr, "iters": iters,
                        "subsets": subsets, "n_trues": int(n_trues)}
                rows.append({**base, "corrected": False, **metrics(prof, seg, prof_t)})
                rows.append({**base, "corrected": True,
                             **metrics(prof * np.nanmean(cp) / cp, seg * np.nanmean(cs) / cs, prof_t)})
                shots[(config, pose, ctr)] = (E.reslice(img, grid, "sa"), E.reslice(t["act"], grid, "sa"))
    set_pose((0, 0, 0), None)
    return rows, shots


def fig_metrics(rows, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    ctrs = sorted({r["ctr_ps"] for r in rows}, reverse=True)
    poses = [p[0] for p in POSES]
    groups = [(p, c) for p in poses for c in C.RECON_CONFIGS]
    fig, axs = plt.subplots(2, len(ctrs), figsize=(4.6 * len(ctrs), 6.0), squeeze=False)
    letter = iter("ABCDEFGH")
    for i, (k, lab) in enumerate([("defect_contrast", "Defect contrast (true 0.20)"),
                                  ("false_defect", "Largest false defect, remote wall")]):
        for j, ctr in enumerate(ctrs):
            ax = axs[i, j]
            E.style(ax)
            lims = []
            x = np.arange(len(groups)) + np.repeat(np.arange(len(poses)), len(C.RECON_CONFIGS)) * 0.6
            for off, corr, alpha in ((-0.2, False, 0.4), (0.2, True, 1.0)):
                get = {(r["pose"], r["config"]): r[k] for r in rows if r["ctr_ps"] == ctr and r["corrected"] == corr}
                vals = [get.get(g, np.nan) for g in groups]
                lims.extend(vals)
                ax.bar(x + off, vals, width=0.38,
                       color=[E.COL[g[1]] for g in groups], alpha=alpha, edgecolor="white", lw=0.8)
            if k == "defect_contrast":
                tc = np.mean([r["true_contrast"] for r in rows])
                ax.axhline(tc, color=E.INK2, lw=0.9)
            else:
                ax.axhline(DEFECT_DEPTH, color="#e34948", lw=0.9, ls=(0, (4, 2)))
            ax.set_ylim(min(0.0, np.nanmin(lims) - 0.03), max(0.35, np.nanmax(lims) + 0.03))
            centres = [x[len(C.RECON_CONFIGS) * n: len(C.RECON_CONFIGS) * (n + 1)].mean() for n in range(len(poses))]
            ax.set_xticks(centres, [p.replace(" ", "\n", 1) for p in poses], fontsize=7)
            ax.set_ylabel(lab, fontsize=8)
            ax.set_title(f"{next(letter)}.  {E.CTRNAME[ctr]}", loc="left", fontweight="bold", fontsize=9)
    handles = [Patch(color=E.COL[c], label=E.NAME[c]) for c in C.RECON_CONFIGS] + \
              [Patch(color=E.INK2, alpha=0.4, label="uncorrected"), Patch(color=E.INK2, label="corrected")]
    fig.legend(handles=handles, loc="lower center", ncol=len(handles), frameon=False, fontsize=7.5)
    fig.suptitle(f"{TEST_BODY} body, 20% mid-anterior defect; correction from {CAL_BODY} base-pose phantom",
                 fontsize=9)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    for ext in ("png", "pdf"):
        fig.savefig(out / f"Fig_robust_metrics.{ext}", dpi=300)
    plt.close(fig)


def fig_images(shots, out, ctr):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy.ndimage import gaussian_filter

    poses = [p[0] for p in POSES]
    rows = [("truth", None)] + [(E.NAME[c], c) for c in C.RECON_CONFIGS]
    fig, axs = plt.subplots(len(rows), len(poses), figsize=(2.0 * len(poses), 2.1 * len(rows)))
    for j, pose in enumerate(poses):
        for i, (name, c) in enumerate(rows):
            ax = axs[i, j]
            img, tru = shots[(C.RECON_CONFIGS[0] if c is None else c, pose, ctr)]
            show = tru if c is None else gaussian_filter(img, E.DISPLAY_FWHM_MM / 2.3548)
            ax.imshow(show, cmap="gray_r", vmin=0, vmax=1.15, origin="lower")
            ax.set_xticks([])
            ax.set_yticks([])
            for s in ax.spines.values():
                s.set_visible(False)
            if i == 0:
                ax.set_title(pose, fontsize=8)
            if j == 0:
                ax.set_ylabel(f"{'ABC'[i]}.  {name}", fontsize=8, fontweight="bold")
    fig.suptitle(f"Mid short axis, {TEST_BODY}, {E.CTRNAME[ctr]}; 20% defect at 12 o'clock", fontsize=8.5)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(out / f"Fig_robust_images.{ext}", dpi=300)
    plt.close(fig)


def summary(rows, out):
    lines = [f"Stage 3 robustness: {TEST_BODY}, 20% mid-anterior defect, "
             f"{rows[0]['iters']}x{rows[0]['subsets']} OSEM, {rows[0]['n_trues']:,} trues; "
             f"correction from {CAL_BODY} base pose",
             f"{'config':12s} {'CTR':>7s} {'pose':15s} {'contrast':>8s} {'c.contr':>8s} {'c.remCV%':>8s} "
             f"{'c.false':>7s} (c. = corrected; true contrast {rows[0]['true_contrast']:.3f})"]
    idx = {(r["config"], r["ctr_ps"], r["pose"], r["corrected"]): r for r in rows}
    for config in C.RECON_CONFIGS:
        for ctr in sorted({r["ctr_ps"] for r in rows}, reverse=True):
            for pose in [p[0] for p in POSES]:
                u, c = idx[(config, ctr, pose, False)], idx[(config, ctr, pose, True)]
                lines.append(f"{config:12s} {E.CTRNAME[ctr]:>7s} {pose:15s} {u['defect_contrast']:8.3f} "
                             f"{c['defect_contrast']:8.3f} {c['remote_cv_pct']:8.1f} {c['false_defect']:7.3f}")
    (out / "SUMMARY.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--recon", required=True)
    ap.add_argument("--ctrs", default="320,200")
    ap.add_argument("--iters", type=int, default=8)
    ap.add_argument("--subsets", type=int, default=C.RECON_SUBSETS)
    ap.add_argument("--n-trues", type=float, default=C.FASTMC_TRUES)
    ap.add_argument("--seed", type=int, default=21)
    ap.add_argument("--configs", default=",".join(C.RECON_CONFIGS), help="subset for testing")
    ap.add_argument("--poses", type=int, default=len(POSES), help="first N poses (testing)")
    args = ap.parse_args()
    recon_dir = Path(args.recon)
    out = recon_dir / "robust"
    out.mkdir(exist_ok=True)
    ctrs = [int(c) for c in args.ctrs.split(",")]
    C.RECON_CONFIGS = args.configs.split(",")
    del POSES[args.poses:]
    rows, shots = run(recon_dir, out, ctrs, args.iters, args.subsets, args.n_trues, args.seed)
    (out / "robust_metrics.json").write_text(json.dumps(rows, indent=2))
    with open(out / "robust_metrics.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    fig_metrics(rows, out)
    fig_images(shots, out, min(ctrs))
    summary(rows, out)


if __name__ == "__main__":
    main()

"""
Stage 3b: heart-pose robustness with per-pose sensitivity and a pose-matched correction.

    python robust2.py --recon /scratch/$USER/exercise_pet_recon --task N      one (geometry, pose), array task
    python robust2.py --recon /scratch/$USER/exercise_pet_recon --collect     figures and tables from all tasks

Fixes stage 3: the sensitivity (attenuation) image is recomputed for every pose, because the moved heart
displaces lung. Three correction variants are reported for each (geometry, pose, CTR):
  none   no segment correction
  fixed  correction measured once on the P50 body, base pose, no defect (as in stage 3)
  pose   pose-matched: a uniform LV (no defect) simulated in the same P90 body at the same heart pose,
         independent MC realisation, same recon; the clinical analogue is a simulation from the patient's
         attenuation CT with the heart at its measured position.
Writes <recon>/<outdir>/: task_*.json/.npz per task, then <tag>_metrics.csv/.json, Fig_<tag>_metrics,
Fig_<tag>_images, SUMMARY_<tag>.txt (defaults: robust_v3, robust3).
--n-cal sets the trues in the pose-matched uniform-LV simulation (default 50 M); it is generated in
parallel 2 M chunks (25 processes), together with the 10 M defect data (one process, same seed as stage 3b).
"""

import argparse
import csv
import os
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp
import json
import re
import time
from pathlib import Path

import numpy as np

import geometry as G
import evaluate as E
import fastmc
import robust as R
from run_recon import log

C = G.C
TAG = "robust3"
CHUNK = int(os.environ.get("XPET_CHUNK", 2_000_000))
VARIANTS = [("none", "uncorrected", 0.35), ("fixed", "fixed (P50 base pose)", 0.65), ("pose", "pose-matched", 1.0)]


def tasks():
    return [(c, p) for c in C.RECON_CONFIGS for p in range(len(R.POSES))]


def slug(pose):
    return re.sub(r"[^a-z0-9]+", "_", pose.lower().replace("°", "deg")).strip("_")


def correct(prof, seg, cal):
    cp, cs = cal
    return prof * np.nanmean(cp) / cp, seg * np.nanmean(cs) / cs


def _gen(config, ph, mu, n, seed):
    ring = G.Ring(config)
    grid = G.recon_grid(ring, 3.0)
    return fastmc.generate_trues(ring, grid, ph, mu, int(n), np.random.default_rng(seed), log=lambda m: None)


def generate_parallel(config, jobs, workers):
    """jobs: list of (ph, mu, n, seed). Runs fastmc.generate_trues in separate processes."""
    ncpu = int(os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count() or 1))
    os.environ["NUMBA_NUM_THREADS"] = str(max(1, ncpu // workers))
    with ProcessPoolExecutor(workers, mp_context=mp.get_context("spawn")) as ex:
        futs = [ex.submit(_gen, config, *j) for j in jobs]
        return [f.result() for f in futs]


def merge(evs):
    out = {k: np.concatenate([e[k] for e in evs]) for k in ("p1", "p2", "dt", "e")}
    out["n_emitted"] = np.int64(sum(int(e["n_emitted"]) for e in evs))
    return out


def run_task(recon_dir, out, config, pi, ctrs, iters, subsets, n_trues, n_cal, sens_events, seed):
    pose, shift, r = R.POSES[pi]
    ring = G.Ring(config)
    grid = G.recon_grid(ring, 3.0)
    log(f"[{config}] pose {pose}: grid {grid.shape}")

    # fixed correction: P50, base pose, no defect (cached stage-2 events and sensitivity)
    R.set_pose((0, 0, 0), None)
    t_cal = R.truth(grid, G.load_body(R.CAL_BODY))
    sens_cal = np.load(recon_dir / f"sens_{config}_{R.CAL_BODY}_v3.npy")
    ev = dict(np.load(recon_dir / f"events_fastmc_{config}_{R.CAL_BODY}_{int(n_trues)}.npz"))
    fixed = {}
    for ctr in ctrs:
        _, p, s = R.measure(R.recon(ev, grid, sens_cal, ctr, iters, subsets, seed), t_cal)
        fixed[ctr] = (p, s)
    del ev, sens_cal

    # pose-specific phantoms, attenuation and sensitivity
    R.set_pose(shift, r)
    ph_def = R.phantom(R.TEST_BODY, defect=True)
    ph_uni = R.phantom(R.TEST_BODY, defect=False)
    mu = G.mu_map(grid, ph_def)                       # defect changes activity only
    sens_path = recon_dir / f"sens_{config}_{R.TEST_BODY}_{slug(pose)}_v3.npy"
    if pi == 0:
        sens_path = recon_dir / f"sens_{config}_{R.TEST_BODY}_v3.npy"     # base pose: stage-2 image
    if sens_path.exists():
        sens = np.load(sens_path)
        log(f"  sensitivity loaded: {sens_path.name}")
    else:
        t0 = time.time()
        sens = fastmc.sensitivity(ring, grid, mu, int(sens_events), np.random.default_rng(seed + 1000 + pi),
                                  log=lambda m: None)
        np.save(sens_path, sens)
        log(f"  sensitivity computed in {time.time() - t0:.0f} s: {sens_path.name}")

    t = R.truth(grid, ph_def)
    _, prof_t = E.mid_profile(t["act"], t)
    t_uni = R.truth(grid, ph_uni)

    k = int(np.ceil(n_cal / CHUNK))
    jobs = [(ph_uni, mu, min(CHUNK, n_cal - i * CHUNK), seed + 500 + pi + 1000 * i) for i in range(k)]
    jobs.append((ph_def, mu, n_trues, seed + 100 * pi))       # same seed as stages 3 and 3b
    t0 = time.time()
    res = generate_parallel(config, jobs, len(jobs))
    ev_uni, ev = merge(res[:-1]), res[-1]
    del res
    log(f"  {len(ev_uni['p1']):,} uniform-LV and {len(ev['p1']):,} defect trues in {time.time() - t0:.0f} s")
    matched = {}
    for ctr in ctrs:
        _, p, s = R.measure(R.recon(ev_uni, grid, sens, ctr, iters, subsets, seed + 50 + pi), t_uni)
        matched[ctr] = (p, s)
    del ev_uni

    rows, shots = [], {}
    for ctr in ctrs:
        img, prof, seg = R.measure(R.recon(ev, grid, sens, ctr, iters, subsets, seed + pi), t)
        base = {"config": config, "body": R.TEST_BODY, "pose": pose, "ctr_ps": ctr, "iters": iters,
                "subsets": subsets, "n_trues": int(n_trues), "n_cal": int(n_cal)}
        for key, cal in (("none", None), ("fixed", fixed[ctr]), ("pose", matched[ctr])):
            p, s = (prof, seg) if cal is None else correct(prof, seg, cal)
            rows.append({**base, "correction": key, **R.metrics(p, s, prof_t)})
        shots[f"img_{ctr}"] = E.reslice(img, grid, "sa")
        shots[f"true_{ctr}"] = E.reslice(t["act"], grid, "sa")
    R.set_pose((0, 0, 0), None)
    name = f"task_{config}_{slug(pose)}"
    (out / f"{name}.json").write_text(json.dumps(rows, indent=2))
    np.savez_compressed(out / f"{name}.npz", **shots)
    for row in rows:
        log(f"  CTR {row['ctr_ps']} {row['correction']:5s}: contrast {row['defect_contrast']:.3f} "
            f"(true {row['true_contrast']:.3f}), remote CV {row['remote_cv_pct']:.1f}%, "
            f"false defect {row['false_defect']:.3f}")


def fig_metrics(rows, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    ctrs = sorted({r["ctr_ps"] for r in rows}, reverse=True)
    poses = [p[0] for p in R.POSES]
    configs = [c for c in C.RECON_CONFIGS if any(r["config"] == c for r in rows)]
    groups = [(p, c) for p in poses for c in configs]
    nv = len(VARIANTS)
    fig, axs = plt.subplots(3, len(ctrs), figsize=(4.8 * len(ctrs), 8.4), squeeze=False)
    letter = iter("ABCDEFGHI")
    panels = [("defect_contrast", "Defect contrast"), ("remote_cv_pct", "Remote wall CV (%)"),
              ("false_defect", "Largest false defect, remote wall")]
    tc = float(np.mean([r["true_contrast"] for r in rows]))
    for i, (k, lab) in enumerate(panels):
        for j, ctr in enumerate(ctrs):
            ax = axs[i, j]
            E.style(ax)
            x = np.arange(len(groups)) * 1.0 + np.repeat(np.arange(len(poses)), len(configs)) * 0.5
            w = 0.8 / nv
            vals_all = []
            for n, (key, _, alpha) in enumerate(VARIANTS):
                get = {(r["pose"], r["config"]): r[k] for r in rows if r["ctr_ps"] == ctr and r["correction"] == key}
                vals = [get.get(g, np.nan) for g in groups]
                vals_all.extend(vals)
                ax.bar(x + (n - (nv - 1) / 2) * w, vals, width=w * 0.95,
                       color=[E.COL[g[1]] for g in groups], alpha=alpha, edgecolor="white", lw=0.6)
            if k == "defect_contrast":
                ax.axhline(tc, color=E.INK2, lw=0.9)
                ax.set_ylim(min(0.0, np.nanmin(vals_all) - 0.03), max(0.35, np.nanmax(vals_all) + 0.03))
            elif k == "false_defect":
                ax.axhline(R.DEFECT_DEPTH, color="#e34948", lw=0.9, ls=(0, (4, 2)))
                ax.set_ylim(0, max(0.25, np.nanmax(vals_all) + 0.03))
            else:
                ax.set_ylim(0, np.nanmax(vals_all) * 1.15)
            centres = [x[len(configs) * n: len(configs) * (n + 1)].mean() for n in range(len(poses))]
            ax.set_xticks(centres, [p.replace(" ", "\n", 1) for p in poses], fontsize=7)
            ax.set_ylabel(lab, fontsize=8)
            ax.set_title(f"{next(letter)}.  {E.CTRNAME[ctr]}", loc="left", fontweight="bold", fontsize=9)
    handles = [Patch(color=E.COL[c], label=E.NAME[c]) for c in configs] + \
              [Patch(color=E.INK2, alpha=a, label=lab) for _, lab, a in VARIANTS]
    fig.legend(handles=handles, loc="lower center", ncol=len(handles), frameon=False, fontsize=7.5)
    fig.suptitle(f"{R.TEST_BODY} body, 20% mid-anterior defect (true contrast {tc:.2f}); "
                 f"per-pose sensitivity images", fontsize=9)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    for ext in ("png", "pdf"):
        fig.savefig(out / f"Fig_{TAG}_metrics.{ext}", dpi=300)
    plt.close(fig)


def fig_images(shots, out, ctr):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy.ndimage import gaussian_filter

    poses = [p[0] for p in R.POSES]
    configs = [c for c in C.RECON_CONFIGS if any(k[0] == c for k in shots)]
    rows = [("truth", None)] + [(E.NAME[c], c) for c in configs]
    fig, axs = plt.subplots(len(rows), len(poses), figsize=(2.0 * len(poses), 2.1 * len(rows)), squeeze=False)
    for j, pose in enumerate(poses):
        for i, (name, c) in enumerate(rows):
            ax = axs[i, j]
            ax.set_xticks([])
            ax.set_yticks([])
            for s in ax.spines.values():
                s.set_visible(False)
            d = shots.get((configs[0] if c is None else c, pose))
            if d is not None:
                show = d[f"true_{ctr}"] if c is None else gaussian_filter(d[f"img_{ctr}"], E.DISPLAY_FWHM_MM / 2.3548)
                ax.imshow(show, cmap="gray_r", vmin=0, vmax=1.15, origin="lower")
            if i == 0:
                ax.set_title(pose, fontsize=8)
            if j == 0:
                ax.set_ylabel(f"{'ABCD'[i]}.  {name}", fontsize=8, fontweight="bold")
    fig.suptitle(f"Mid short axis, {R.TEST_BODY}, {E.CTRNAME[ctr]}, uncorrected; 20% defect at 12 o'clock; "
                 f"per-pose sensitivity", fontsize=8.5)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(out / f"Fig_{TAG}_images.{ext}", dpi=300)
    plt.close(fig)


def summary(rows, out):
    r0 = rows[0]
    lines = [f"Stage 3 robustness ({TAG}): {R.TEST_BODY}, 20% mid-anterior defect, {r0['iters']}x{r0['subsets']} OSEM, "
             f"{r0['n_trues']:,} trues; pose-matched correction from {r0.get('n_cal', r0['n_trues']):,} trues; per-pose sensitivity; true contrast {r0['true_contrast']:.3f}",
             "corrections: none | fixed = P50 base pose | pose = uniform LV at the same pose in the same body",
             f"{'config':12s} {'CTR':>7s} {'pose':15s} {'corr':5s} {'contrast':>8s} {'remCV%':>7s} {'false':>6s}"]
    idx = {(r["config"], r["ctr_ps"], r["pose"], r["correction"]): r for r in rows}
    for config in C.RECON_CONFIGS:
        for ctr in sorted({r["ctr_ps"] for r in rows}, reverse=True):
            for pose in [p[0] for p in R.POSES]:
                for key, _, _ in VARIANTS:
                    r = idx.get((config, ctr, pose, key))
                    if r is None:
                        lines.append(f"{config:12s} {E.CTRNAME[ctr]:>7s} {pose:15s} {key:5s}  MISSING")
                        continue
                    lines.append(f"{config:12s} {E.CTRNAME[ctr]:>7s} {pose:15s} {key:5s} {r['defect_contrast']:8.3f} "
                                 f"{r['remote_cv_pct']:7.1f} {r['false_defect']:6.3f}")
    (out / f"SUMMARY_{TAG}.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


def collect(out, ctrs):
    rows, shots = [], {}
    for config, pi in tasks():
        pose = R.POSES[pi][0]
        f = out / f"task_{config}_{slug(pose)}.json"
        if not f.exists():
            log(f"missing: {f.name}")
            continue
        rows.extend(json.loads(f.read_text()))
        shots[(config, pose)] = dict(np.load(out / f"task_{config}_{slug(pose)}.npz"))
    if not rows:
        raise SystemExit("no task results found")
    (out / f"{TAG}_metrics.json").write_text(json.dumps(rows, indent=2))
    with open(out / f"{TAG}_metrics.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    fig_metrics(rows, out)
    fig_images(shots, out, min(ctrs))
    summary(rows, out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--recon", required=True)
    ap.add_argument("--task", type=int, default=None, help="array index: geometry x pose")
    ap.add_argument("--collect", action="store_true")
    ap.add_argument("--ntasks", action="store_true", help="print the number of array tasks and exit")
    ap.add_argument("--ctrs", default="320,200")
    ap.add_argument("--iters", type=int, default=8)
    ap.add_argument("--subsets", type=int, default=C.RECON_SUBSETS)
    ap.add_argument("--n-trues", type=float, default=C.FASTMC_TRUES)
    ap.add_argument("--n-cal", type=float, default=50e6, help="trues for the pose-matched correction")
    ap.add_argument("--outdir", default="robust_v3")
    ap.add_argument("--tag", default="robust3")
    ap.add_argument("--sens-events", type=float, default=C.SENS_DETECTED)
    ap.add_argument("--seed", type=int, default=21)
    args = ap.parse_args()
    if args.ntasks:
        print(len(tasks()))
        return
    recon_dir = Path(args.recon)
    global TAG
    TAG = args.tag
    out = recon_dir / args.outdir
    out.mkdir(exist_ok=True)
    ctrs = [int(c) for c in args.ctrs.split(",")]
    if args.collect:
        collect(out, ctrs)
    elif args.task is not None:
        config, pi = tasks()[args.task]
        run_task(recon_dir, out, config, pi, ctrs, args.iters, args.subsets, args.n_trues, int(args.n_cal), args.sens_events,
                 args.seed)
    else:
        raise SystemExit("give --task N or --collect")


if __name__ == "__main__":
    main()

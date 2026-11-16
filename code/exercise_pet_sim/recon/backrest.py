"""
Stage 4: attenuation by the bike backrest.

    python backrest.py --recon /scratch/$USER/exercise_pet_recon --task N     one (geometry, backrest), array task
    python backrest.py --recon /scratch/$USER/exercise_pet_recon --collect    figures and tables

A flat backrest (38 cm wide, full axial FOV) sits against the back of the P90 body (base heart pose,
uniform LV, no defect). Materials, each with a 20 mm foam cushion or core:
  none       no backrest (reference)
  carbon     2 x 1.5 mm carbon-fibre skins on a 20 mm foam core (PET/CT pallet type)
  aluminum   5 mm aluminium plate
  steel      5 mm steel plate
Every case uses the same number of emitted decays (that of the stage-2 P90 10 M-trues data set), so the
count loss from the backrest shows up as noise. Each data set is reconstructed with the backrest in the
attenuation (sensitivity) image, and also without it (backrest ignored), at 320 and 200 ps, OSEM 8x10.
Simulation and sensitivity run as parallel chunks in separate processes.

Metrics (images scaled to total body activity): trues relative to no backrest; myocardial recovery;
segment CV (16 segments); inferior/anterior ratio (mid-ventricular profile); liver mean and voxel noise.
Writes <recon>/backrest/: task_*.json/.npz, backrest_metrics.csv/.json, Fig_backrest_metrics,
Fig_backrest_images, SUMMARY_backrest.txt.
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
from projector import fwd
from run_recon import osem, log

C = G.C
BODY = "P90"
FOAM_MU, FOAM_MM = 0.00072, 20.0                          # rigid/cushion foam, ~0.075 g/cm3
# name, label, [(mu per mm, thickness mm)] (structural layers; the foam layer is added to all but "none")
MATERIALS = [("none", "no backrest", []),
             ("carbon", "carbon fibre, 3 mm + foam", [(0.0139, 3.0)]),
             ("aluminum", "aluminium, 5 mm + foam", [(0.0227, 5.0)]),
             ("steel", "steel, 5 mm + foam", [(0.0652, 5.0)])]
HALF_WIDTH, SLAB_MM = 190.0, 24.0
CTRS = (320, 200)
WORKERS = int(os.environ.get("XPET_WORKERS", 30))


def tasks():
    return [(c, m) for c in C.RECON_CONFIGS for m in range(len(MATERIALS))]


def mu_line(mi):
    """Backrest mu x thickness at normal incidence."""
    layers = MATERIALS[mi][2]
    return 0.0 if not layers else sum(m * t for m, t in layers) + FOAM_MU * FOAM_MM


def backrest_mask(grid, ph):
    """Slab behind the body: SLAB_MM thick, |x| <= HALF_WIDTH, all z."""
    gx, gy, gz = grid.axes()
    back = -float(C.BODIES[BODY]["outer"][1])
    ysel = (gy < back) & (gy >= back - SLAB_MM)
    xsel = np.abs(gx) <= HALF_WIDTH
    m = np.zeros(grid.shape, bool)
    m[:, ysel[:, None] & xsel[None, :]] = True
    return m, int(ysel.sum())


def mu_with_backrest(grid, ph, mi):
    mu = G.mu_map(grid, ph)
    if mu_line(mi) > 0:
        m, nrow = backrest_mask(grid, ph)
        mu[m] = mu_line(mi) / (nrow * grid.v)
    return mu


# ---- worker processes -------------------------------------------------------------------------------
def _emit(config, ph, mu, n_emit, seed, chunk=2_000_000):
    """Trues from a fixed number of emitted decays (as fastmc.generate_trues, but emission-limited)."""
    ring = G.Ring(config)
    grid = G.recon_grid(ring, 3.0)
    rng = np.random.default_rng(seed)
    out = {"p1": [], "p2": [], "dt": []}
    done = 0
    while done < n_emit:
        n = min(chunk, n_emit - done)
        e = fastmc.sample_emissions(ph, n, rng)
        done += n
        q1, q2, e = fastmc._pairs(ring, e, rng)
        att = np.exp(-fwd(mu, grid, q1.astype(np.float32), q2.astype(np.float32)))
        keep = rng.random(len(att)) < att
        q1, q2, e = q1[keep], q2[keep], e[keep]
        s1, _ = ring.snap(q1)
        s2, _ = ring.snap(q2)
        dt = (np.linalg.norm(q1 - e, axis=1) - np.linalg.norm(q2 - e, axis=1)) / C.C_MM_PER_PS
        out["p1"].append(s1)
        out["p2"].append(s2)
        out["dt"].append(dt.astype(np.float32))
    return {k: np.concatenate(v) for k, v in out.items()}


def _sens(config, mu, n_detected, seed):
    ring = G.Ring(config)
    grid = G.recon_grid(ring, 3.0)
    return fastmc.sensitivity(ring, grid, mu, int(n_detected), np.random.default_rng(seed),
                              chunk=1_000_000, log=lambda m: None)


def pool_map(fn, jobs):
    ncpu = int(os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count() or 1))
    workers = max(1, min(WORKERS, len(jobs), ncpu))
    os.environ["NUMBA_NUM_THREADS"] = str(max(1, ncpu // workers))
    with ProcessPoolExecutor(workers, mp_context=mp.get_context("spawn")) as ex:
        return [f.result() for f in [ex.submit(fn, *j) for j in jobs]]


# ---- one task ---------------------------------------------------------------------------------------
def metrics(img, t):
    img = E.normalise(img, t)
    seg = E.segment_values(img, t)
    _, prof = E.mid_profile(img, t)
    liver = t["lab"] == C.LABELS["liver"]
    lv = img[liver]
    return img, {"myo_recovery": float(np.nanmean(seg)),
                 "segment_cv_pct": float(100 * np.nanstd(seg) / np.nanmean(seg)),
                 "inf_over_ant": float(np.nanmean(prof[[5, 6]]) / np.nanmean(prof[[0, 11]])),
                 "ant_inf_over_sep_lat": float(np.nanmean(prof[[0, 5, 6, 11]]) / np.nanmean(prof[[2, 3, 8, 9]])),
                 "liver_mean": float(lv.mean() / t["act"][liver].mean()),
                 "liver_noise_pct": float(100 * lv.std() / lv.mean())}


def run_task(recon_dir, out, config, mi, iters, subsets, n_trues_ref, sens_events, seed):
    name = MATERIALS[mi][0]
    ring = G.Ring(config)
    grid = G.recon_grid(ring, 3.0)
    ph = G.load_body(BODY)
    t = E.Truth.get(config, BODY)
    mu = mu_with_backrest(grid, ph, mi)
    ref = np.load(recon_dir / f"events_fastmc_{config}_{BODY}_{int(n_trues_ref)}.npz")
    n_emit = int(ref["n_emitted"])
    log(f"[{config}] backrest {name}: mu*t {mu_line(mi):.4f} (transmission {np.exp(-mu_line(mi)):.3f}), "
        f"{n_emit:,} emissions")

    sens_none = np.load(recon_dir / f"sens_{config}_{BODY}_v3.npy")
    if mu_line(mi) > 0:
        sp = recon_dir / f"sens_{config}_{BODY}_backrest_{name}_v3.npy"
        if sp.exists():
            sens = np.load(sp)
        else:
            t0 = time.time()
            k = WORKERS
            parts = pool_map(_sens, [(config, mu, sens_events / k, seed + 7000 + 31 * mi + i) for i in range(k)])
            sens = np.sum(parts, axis=0).astype(np.float32)
            np.save(sp, sens)
            log(f"  sensitivity in {time.time() - t0:.0f} s")
    else:
        sens = sens_none

    t0 = time.time()
    k = WORKERS
    base = n_emit // k
    jobs = [(config, ph, mu, base + (1 if i < n_emit - base * k else 0), seed + 100 * i) for i in range(k)]
    parts = pool_map(_emit, jobs)
    ev = {key: np.concatenate([p[key] for p in parts]) for key in ("p1", "p2", "dt")}
    del parts
    n_ev = len(ev["p1"])
    log(f"  {n_ev:,} trues in {time.time() - t0:.0f} s")

    rows, shots = [], {}
    recons = [("modelled", sens)] + ([("ignored", sens_none)] if mu_line(mi) > 0 else [])
    rng_base = seed + 11
    for ctr in CTRS:
        for ac, s in recons:
            rng = np.random.default_rng(rng_base + ctr)          # same TOF noise and subsets for both AC variants
            sig_t = ctr / 2.3548
            D = np.linalg.norm(ev["p2"] - ev["p1"], axis=1)
            dt = ev["dt"] + rng.normal(size=n_ev).astype(np.float32) * sig_t
            ev["tc"] = ((D + C.C_MM_PER_PS * dt) / 2.0).astype(np.float32)
            img = osem(ev, grid, s, C.C_MM_PER_PS * sig_t / 2.0, iters, subsets, rng)
            img, m = metrics(img, t)
            rows.append({"config": config, "backrest": name, "mu_t": mu_line(mi),
                         "transmission": float(np.exp(-mu_line(mi))), "ac": ac, "ctr_ps": ctr,
                         "iters": iters, "subsets": subsets, "n_emitted": n_emit, "n_trues": n_ev, **m})
            log(f"  CTR {ctr} AC {ac:8s}: recovery {m['myo_recovery']:.3f}, seg CV {m['segment_cv_pct']:.1f}%, "
                f"inf/ant {m['inf_over_ant']:.3f}, liver noise {m['liver_noise_pct']:.1f}%")
            shots[f"sa_{ac}_{ctr}"] = E.reslice(img, grid, "sa")
    iz = grid.shape[0] // 2
    shots["mu_axial"] = mu[iz]
    tag = f"task_{config}_{name}"
    (out / f"{tag}.json").write_text(json.dumps(rows, indent=2))
    np.savez_compressed(out / f"{tag}.npz", **shots)


# ---- collect ----------------------------------------------------------------------------------------
def fig_metrics(rows, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    configs = [c for c in C.RECON_CONFIGS if any(r["config"] == c for r in rows)]
    mats = [m[0] for m in MATERIALS]
    idx = {(r["config"], r["backrest"], r["ac"], r["ctr_ps"]): r for r in rows}
    ctr = min(CTRS)
    fig, axs = plt.subplots(2, 2, figsize=(9.6, 7.2))
    x = np.arange(len(mats))
    w = 0.8 / len(configs)
    panels = [("rel_trues", "Trues relative to no backrest"), ("inf_over_ant", "Inferior / anterior wall"),
              ("segment_cv_pct", "Segment CV (%)"), ("liver_noise_pct", "Liver voxel noise (%)")]
    for ax, (k, lab), letter in zip(axs.ravel(), panels, "ABCD"):
        E.style(ax)
        for n, c in enumerate(configs):
            off = (n - (len(configs) - 1) / 2) * w
            if k == "rel_trues":
                vals = [idx[(c, m, "modelled", ctr)]["rel_trues"] if (c, m, "modelled", ctr) in idx
                        else np.nan for m in mats]
                ax.bar(x + off, vals, w * 0.9, color=E.COL[c])
                for xi, v in zip(x + off, vals):
                    ax.text(xi, v + 0.01, f"{v:.2f}", ha="center", va="bottom", fontsize=6.5, color=E.INK2)
                ax.set_ylim(0, 1.12)
                continue
            for ac, alpha, dx in (("ignored", 0.4, -w * 0.22), ("modelled", 1.0, w * 0.22)):
                vals = [idx[(c, m, ac, ctr)][k] if (c, m, ac, ctr) in idx else np.nan for m in mats]
                ax.bar(x + off + dx, vals, w * 0.44, color=E.COL[c], alpha=alpha)
            if k == "inf_over_ant":
                ax.axhline(1.0, color=E.INK2, lw=0.9)
                lo = min(r[k] for r in rows if r["ctr_ps"] == ctr)
                hi = max(r[k] for r in rows if r["ctr_ps"] == ctr)
                ax.set_ylim(min(0.8, lo - 0.03), max(1.1, hi + 0.03))
        ax.set_xticks(x, [m.replace("aluminum", "aluminium") for m in mats], fontsize=8)
        ax.set_ylabel(lab, fontsize=8)
        ax.set_title(f"{letter}.", loc="left", fontweight="bold", fontsize=9)
    handles = [Patch(color=E.COL[c], label=E.NAME[c]) for c in configs] + \
              [Patch(color=E.INK2, alpha=0.4, label="backrest ignored in AC"),
               Patch(color=E.INK2, label="backrest in AC")]
    fig.legend(handles=handles, loc="lower center", ncol=len(handles), frameon=False, fontsize=7.5)
    fig.suptitle(f"{BODY} body, uniform LV, equal emitted decays; {E.CTRNAME[ctr]}, OSEM "
                 f"{rows[0]['iters']}x{rows[0]['subsets']}", fontsize=9)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    for ext in ("png", "pdf"):
        fig.savefig(out / f"Fig_backrest_metrics.{ext}", dpi=300)
    plt.close(fig)


def fig_images(shots, out, grids):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy.ndimage import gaussian_filter

    ctr = min(CTRS)
    configs = [c for c in C.RECON_CONFIGS if any(k[0] == c for k in shots)]
    mats = [m[0] for m in MATERIALS]
    nrow = 1 + len(configs)
    fig, axs = plt.subplots(nrow, len(mats), figsize=(2.2 * len(mats), 2.25 * nrow), squeeze=False)
    c0 = configs[0]
    g = grids[c0]
    gx, gy, _ = g.axes()
    ext = (gx[0], gx[-1], gy[0], gy[-1])
    for j, m in enumerate(mats):
        for i in range(nrow):
            ax = axs[i, j]
            ax.set_xticks([])
            ax.set_yticks([])
            for s in ax.spines.values():
                s.set_visible(False)
        d = shots.get((c0, m))
        if d is not None:
            ax = axs[0, j]
            ax.imshow(d["mu_axial"], cmap="gray_r", vmin=0, vmax=0.02, origin="lower", extent=ext)
            th = np.linspace(0, 2 * np.pi, 400)
            ax.plot(G.Ring(c0).R * np.cos(th), G.Ring(c0).R * np.sin(th), color=E.COL[c0], lw=0.6)
            ax.set_title(dict((a, b) for a, b, _ in MATERIALS)[m], fontsize=7.5)
        for i, c in enumerate(configs, start=1):
            d = shots.get((c, m))
            if d is None:
                continue
            axs[i, j].imshow(gaussian_filter(d[f"sa_modelled_{ctr}"], E.DISPLAY_FWHM_MM / 2.3548),
                             cmap="gray_r", vmin=0, vmax=1.15, origin="lower")
    axs[0, 0].set_ylabel("A.  attenuation, z = 0", fontsize=8, fontweight="bold")
    for i, c in enumerate(configs, start=1):
        axs[i, 0].set_ylabel(f"{'BCD'[i - 1]}.  {E.NAME[c]}", fontsize=8, fontweight="bold")
    fig.suptitle(f"Backrest behind the {BODY} body (A, {E.NAME[c0]} ring outline); mid short axis, "
                 f"{E.CTRNAME[ctr]}, backrest in AC", fontsize=8.5)
    fig.tight_layout()
    for ext_ in ("png", "pdf"):
        fig.savefig(out / f"Fig_backrest_images.{ext_}", dpi=300)
    plt.close(fig)


def summary(rows, out):
    idx = {(r["config"], r["backrest"], r["ac"], r["ctr_ps"]): r for r in rows}
    lines = [f"Stage 4 backrest: {BODY}, uniform LV, equal emitted decays, OSEM {rows[0]['iters']}x{rows[0]['subsets']}",
             f"{'config':12s} {'backrest':9s} {'transm':>6s} {'relTrue':>7s} {'CTR':>7s} {'AC':9s} "
             f"{'recov':>6s} {'segCV%':>6s} {'inf/ant':>7s} {'aisl':>6s} {'liver':>6s} {'livN%':>6s}"]
    for c in C.RECON_CONFIGS:
        for ctr in CTRS:
            for m, _, _ in MATERIALS:
                for ac in ("modelled", "ignored"):
                    r = idx.get((c, m, ac, ctr))
                    if r is None:
                        continue
                    ref = idx.get((c, "none", "modelled", ctr))
                    rel = r["n_trues"] / ref["n_trues"] if ref else np.nan
                    lines.append(f"{c:12s} {m:9s} {r['transmission']:6.3f} {rel:7.3f} {E.CTRNAME[ctr]:>7s} {ac:9s} "
                                 f"{r['myo_recovery']:6.3f} {r['segment_cv_pct']:6.1f} {r['inf_over_ant']:7.3f} "
                                 f"{r['ant_inf_over_sep_lat']:6.3f} {r['liver_mean']:6.3f} {r['liver_noise_pct']:6.1f}")
    (out / "SUMMARY_backrest.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


def collect(out):
    rows, shots, grids = [], {}, {}
    for c, mi in tasks():
        m = MATERIALS[mi][0]
        f = out / f"task_{c}_{m}.json"
        if not f.exists():
            log(f"missing: {f.name}")
            continue
        rows.extend(json.loads(f.read_text()))
        shots[(c, m)] = dict(np.load(out / f"task_{c}_{m}.npz"))
        grids[c] = G.recon_grid(G.Ring(c), 3.0)
    if not rows:
        raise SystemExit("no task results found")
    for r in rows:
        ref = next((q for q in rows if q["config"] == r["config"] and q["backrest"] == "none"
                    and q["ctr_ps"] == r["ctr_ps"]), None)
        r["rel_trues"] = r["n_trues"] / ref["n_trues"] if ref else float("nan")
    (out / "backrest_metrics.json").write_text(json.dumps(rows, indent=2))
    with open(out / "backrest_metrics.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    fig_metrics(rows, out)
    fig_images(shots, out, grids)
    summary(rows, out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--recon", required=True)
    ap.add_argument("--task", type=int, default=None)
    ap.add_argument("--collect", action="store_true")
    ap.add_argument("--ntasks", action="store_true")
    ap.add_argument("--iters", type=int, default=8)
    ap.add_argument("--subsets", type=int, default=C.RECON_SUBSETS)
    ap.add_argument("--n-trues-ref", type=float, default=C.FASTMC_TRUES, help="stage-2 data set giving the emissions")
    ap.add_argument("--sens-events", type=float, default=C.SENS_DETECTED)
    ap.add_argument("--seed", type=int, default=41)
    args = ap.parse_args()
    if args.ntasks:
        print(len(tasks()))
        return
    recon_dir = Path(args.recon)
    out = recon_dir / "backrest"
    out.mkdir(exist_ok=True)
    if args.collect:
        collect(out)
    elif args.task is not None:
        config, mi = tasks()[args.task]
        run_task(recon_dir, out, config, mi, args.iters, args.subsets, args.n_trues_ref, args.sens_events, args.seed)
    else:
        raise SystemExit("give --task N or --collect")


if __name__ == "__main__":
    main()

"""
Stage 2b: does the open ring's anterior/inferior deficit shrink with more OSEM iterations,
and can a phantom-derived segment correction remove it?

    python iterations.py --recon /scratch/$USER/exercise_pet_recon

Reuses the cached fast-MC events and sensitivity images written by run_recon.py.
For each geometry, body and CTR: OSEM with images kept at --iters (default 4, 8, 16) x 10 subsets.

Correction test: segment factors mean(seg)/seg measured on one body (the calibration phantom) are
applied to the other body (P50 -> P90 and P90 -> P50), same geometry, CTR and iterations.
Noise: coefficient of variation of liver voxels (uniform activity) in the better-sampled half (by sensitivity),
after the 6 mm display filter.
Writes <recon>/iterations/iter_metrics.csv/.json, Fig_iter_metrics.png/.pdf, Fig_iter_correction.png/.pdf.
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter

import geometry as G
import evaluate as E
from run_recon import osem, log

C = G.C
ANT_INF, SEP_LAT = [0, 5, 6, 11], [2, 3, 8, 9]


def metrics(seg, prof):
    return {"myo_recovery": float(np.nanmean(seg)),
            "segment_cv_pct": float(100 * np.nanstd(seg) / np.nanmean(seg)),
            "segment_min_rel": float(np.nanmin(seg) / np.nanmean(seg)),
            "ant_inf_over_sep_lat": float(np.nanmean(prof[ANT_INF]) / np.nanmean(prof[SEP_LAT]))}


def run(recon, iters, subsets, n_trues, seed):
    rows, keep = [], {}
    for config in C.RECON_CONFIGS:
        ring = G.Ring(config)
        grid = G.recon_grid(ring, 3.0)
        for body in C.RECON_BODIES:
            tag = f"{config}_{body}"
            sens = np.load(recon / f"sens_{tag}_v3.npy")
            ev = dict(np.load(recon / f"events_fastmc_{tag}_{int(n_trues)}.npz"))
            t = E.Truth.get(config, body)
            liver = t["lab"] == C.LABELS["liver"]
            liver &= sens >= np.median(sens[liver])          # better-sampled half (liver sits at the FOV edge)
            D = np.linalg.norm(ev["p2"] - ev["p1"], axis=1)
            for ctr in C.RECON_CTRS:
                rng = np.random.default_rng(seed + ctr)
                sig_t = ctr / 2.3548 if ctr > 0 else 0.0
                sigma = C.C_MM_PER_PS * sig_t / 2.0
                dt = ev["dt"] + (rng.normal(size=len(D)) * sig_t if ctr > 0 else 0.0)
                ev["tc"] = ((D + C.C_MM_PER_PS * dt) / 2.0).astype(np.float32)
                log(f"[{tag}] CTR {ctr or 'non-TOF'}: OSEM {max(iters)} x {subsets}")
                t0 = time.time()
                snaps = osem(ev, grid, sens, sigma, max(iters), subsets, rng, snapshots=set(iters))
                for it, img in sorted(snaps.items()):
                    img = E.normalise(img, t)
                    seg = E.segment_values(img, t)
                    _, prof = E.mid_profile(img, t)
                    lv = gaussian_filter(img, E.DISPLAY_FWHM_MM / 2.3548 / grid.v)[liver]
                    row = {"config": config, "body": body, "ctr_ps": ctr, "iters": it, "subsets": subsets,
                           "liver_cov_pct": float(100 * lv.std() / lv.mean()), **metrics(seg, prof)}
                    rows.append(row)
                    keep[(config, body, ctr, it)] = (seg, prof)
                    np.save(recon / "iterations_images" / f"img_{tag}_ctr{ctr}_it{it}.npy", img.astype(np.float32))
                log(f"  done in {time.time() - t0:.0f} s")
    # cross-body segment correction
    for r in rows:
        other = [b for b in C.RECON_BODIES if b != r["body"]][0]
        seg, prof = keep[(r["config"], r["body"], r["ctr_ps"], r["iters"])]
        cseg, cprof = keep[(r["config"], other, r["ctr_ps"], r["iters"])]
        m = metrics(seg * np.nanmean(cseg) / cseg, prof * np.nanmean(cprof) / cprof)
        r.update({"corr_from": other, "corr_segment_cv_pct": m["segment_cv_pct"],
                  "corr_segment_min_rel": m["segment_min_rel"],
                  "corr_ant_inf_over_sep_lat": m["ant_inf_over_sep_lat"]})
    return rows


def fig_iters(rows, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    keys = [("myo_recovery", "Myocardial recovery (true = 1)"),
            ("segment_cv_pct", "Segment non-uniformity, CV (%)"),
            ("ant_inf_over_sep_lat", "Ant+inf / sep+lat wall"),
            ("liver_cov_pct", "Noise: liver CV, 6 mm filter (%)")]
    bodies = C.RECON_BODIES
    fig, axs = plt.subplots(len(bodies), len(keys), figsize=(3.0 * len(keys), 2.8 * len(bodies)), squeeze=False)
    letter = iter("ABCDEFGHIJKLMNOP")
    for i, body in enumerate(bodies):
        for j, (k, lab) in enumerate(keys):
            ax = axs[i, j]
            E.style(ax, "both")
            for config in C.RECON_CONFIGS:
                for ctr in C.RECON_CTRS:
                    rr = sorted([r for r in rows if r["config"] == config and r["body"] == body
                                 and r["ctr_ps"] == ctr], key=lambda r: r["iters"])
                    if rr:
                        ax.plot([r["iters"] for r in rr], [r[k] for r in rr], color=E.COL[config],
                                ls=E.LS[ctr], lw=1.8 if ctr == 200 else 1.3, marker="o", ms=3)
            if k in ("myo_recovery", "ant_inf_over_sep_lat"):
                ax.axhline(1.0, color=E.INK2, lw=0.8)
            its = sorted({r["iters"] for r in rows})
            ax.set_xticks(its)
            ax.set_xlabel(f"OSEM iterations ({rows[0]['subsets']} subsets)", fontsize=8)
            ax.set_ylabel(lab, fontsize=8)
            ax.set_title(f"{next(letter)}.  {body}", loc="left", fontweight="bold", fontsize=9)
    handles = [Line2D([], [], color=E.COL[c], lw=1.8, label=E.NAME[c]) for c in C.RECON_CONFIGS] + \
              [Line2D([], [], color=E.INK2, ls=E.LS[ct], lw=1.3, label=E.CTRNAME[ct]) for ct in C.RECON_CTRS]
    fig.legend(handles=handles, loc="lower center", ncol=len(handles), frameon=False, fontsize=8)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    for ext in ("png", "pdf"):
        fig.savefig(out / f"Fig_iter_metrics.{ext}", dpi=300)
    plt.close(fig)


def fig_correction(rows, out, it):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    bodies = C.RECON_BODIES
    fig, axs = plt.subplots(2, len(bodies), figsize=(3.6 * len(bodies), 5.4), squeeze=False)
    letter = iter("ABCDEFGH")
    groups = [(c, ct) for ct in C.RECON_CTRS for c in C.RECON_CONFIGS]
    for i, (k, ck, lab) in enumerate([("segment_cv_pct", "corr_segment_cv_pct", "Segment non-uniformity, CV (%)"),
                                      ("ant_inf_over_sep_lat", "corr_ant_inf_over_sep_lat", "Ant+inf / sep+lat wall")]):
        for j, body in enumerate(bodies):
            ax = axs[i, j]
            E.style(ax)
            get = {(r["config"], r["ctr_ps"]): r for r in rows if r["body"] == body and r["iters"] == it}
            x = np.arange(len(groups))
            for off, key, alpha in ((-0.2, k, 0.45), (0.2, ck, 1.0)):
                vals = [get[g][key] if g in get else np.nan for g in groups]
                ax.bar(x + off, vals, width=0.38, color=[E.COL[g[0]] for g in groups], alpha=alpha,
                       edgecolor="white", lw=0.8)
                for xi, v in zip(x + off, vals):
                    if np.isfinite(v):
                        ax.text(xi, v, f"{v:.1f}" if "cv" in key else f"{v:.2f}", ha="center", va="bottom",
                                fontsize=5.5)
            if "ant_inf" in k:
                ax.axhline(1.0, color=E.INK2, lw=0.8)
                ax.set_ylim(0, 1.5)
            ax.set_xticks(x, [f"{'56' if g[0] == 'R280_slot' else '70'} cm\n{E.CTRNAME[g[1]]}" for g in groups],
                          fontsize=6.5)
            ax.set_ylabel(lab, fontsize=8)
            src = [b for b in bodies if b != body][0]
            ax.set_title(f"{next(letter)}.  {body}, corrected with {src} phantom", loc="left",
                         fontweight="bold", fontsize=8.5)
    fig.legend(handles=[Patch(color=E.INK2, alpha=0.45, label="uncorrected"),
                        Patch(color=E.INK2, label="segment-corrected")], loc="lower center", ncol=2,
               fontsize=7.5, frameon=False)
    fig.suptitle(f"Phantom-derived segment correction, {it} iterations x {rows[0]['subsets']} subsets", fontsize=9)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    for ext in ("png", "pdf"):
        fig.savefig(out / f"Fig_iter_correction.{ext}", dpi=300)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--recon", required=True)
    ap.add_argument("--iters", default="4,8,16")
    ap.add_argument("--subsets", type=int, default=C.RECON_SUBSETS)
    ap.add_argument("--n-trues", type=float, default=C.FASTMC_TRUES)
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--figures-only", action="store_true")
    args = ap.parse_args()
    recon = Path(args.recon)
    out = recon / "iterations"
    out.mkdir(exist_ok=True)
    (recon / "iterations_images").mkdir(exist_ok=True)       # large; kept out of the folder "xpet pull" copies
    iters = [int(i) for i in args.iters.split(",")]
    if args.figures_only:
        rows = json.loads((out / "iter_metrics.json").read_text())
    else:
        rows = run(recon, iters, args.subsets, args.n_trues, args.seed)
        (out / "iter_metrics.json").write_text(json.dumps(rows, indent=2))
        import csv
        with open(out / "iter_metrics.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
    fig_iters(rows, out)
    fig_correction(rows, out, iters[len(iters) // 2])
    print(f"{'config':12s} {'body':4s} {'CTR':>7s} {'it':>3s} {'recov':>6s} {'CV%':>5s} {'ratio':>6s} "
          f"{'noise%':>6s} {'cCV%':>5s} {'cratio':>6s}")
    for r in rows:
        print(f"{r['config']:12s} {r['body']:4s} {E.CTRNAME[r['ctr_ps']]:>7s} {r['iters']:3d} "
              f"{r['myo_recovery']:6.3f} {r['segment_cv_pct']:5.1f} {r['ant_inf_over_sep_lat']:6.3f} "
              f"{r['liver_cov_pct']:6.1f} {r['corr_segment_cv_pct']:5.1f} {r['corr_ant_inf_over_sep_lat']:6.3f}")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()

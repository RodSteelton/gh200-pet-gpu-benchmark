"""
Sweep 2 analysis: bore size, arm slots and body size.

    python analyze2.py --results /scratch/$USER/exercise_pet_sweep2

Expects  <results>/<config>_<body>/coinc_*.npz   (written by run_sim.py --config ... --body ...)
Writes   <results>/analysis/summary2.csv, summary2.json, angular2.csv,
         Fig_sweep2_counts.png/.pdf and Fig_sweep2_angular.png/.pdf

All relative values use the 70 cm closed ring with the same body size as reference.
"""

import argparse
import json
from pathlib import Path

import numpy as np

import config as C
from analyze import label_lookup, load_gap, lor_geometry

HERE = Path(__file__).resolve().parent
COL = {"R280_slot": "#2a78d6", "R310_slot": "#1baf7a", "R350_slot": "#eb6834", "R350_closed": "#4a3aa7"}
NAME = {"R280_slot": "56 cm, slots", "R310_slot": "62 cm, slots", "R350_slot": "70 cm, slots",
        "R350_closed": "70 cm, closed"}
MARK = {"R280_slot": "o", "R310_slot": "^", "R350_slot": "D", "R350_closed": "s"}
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"


def style(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(INK2)
        ax.spines[s].set_linewidth(0.8)
    ax.tick_params(colors=INK2, labelsize=8, width=0.8)
    ax.grid(axis="y", color=GRID, lw=0.6)
    ax.set_axisbelow(True)


def analyse(results, outdir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    rng = np.random.default_rng(2026)
    phi_bins = np.arange(0, 181, 10)
    rows, ang = [], {}
    for folder in sorted(p for p in results.iterdir() if p.is_dir() and any(p.glob("coinc_*.npz"))):
        d, n_dec, metas = load_gap(folder)
        m0 = metas[0]
        if "config" not in m0 or not d:
            continue
        cfg, body = m0["config"], m0["body"]
        ph = dict(np.load(HERE / "data" / f"phantom_{body}.npz"))
        geo = lor_geometry(d, C.BODIES[body]["outer"])
        true = geo["dist"] < C.SCATTER_DIST_MM
        myo = label_lookup(ph, d["ex"], d["ey"], d["ez"]) == C.LABELS["myocardium"]
        n_myo = n_dec * m0["frac_myo"]
        T, S = int(true.sum()), int((~true).sum())
        Tm, Sm = int((true & myo).sum()), int((~true & myo).sum())
        nec_m = Tm ** 2 / max(Tm + Sm, 1) / n_myo
        row = {
            "config": cfg, "body": body, "bore_cm": 2 * m0["R_mm"] / 10, "slot": m0["slot"],
            "coverage_deg": m0["coverage_deg"], "slot_chord_cm": m0["slot_chord_mm"] / 10,
            "n_modules": m0["n_modules"], "n_crystals": m0["n_modules"] * C.CRYSTALS_PER_MODULE_T * C.CRYSTALS_AXIAL,
            "n_jobs": len(metas), "decays": n_dec, "frac_myo": m0["frac_myo"],
            "trues": T, "scatter_fraction": S / max(T + S, 1),
            "sens_myo_trues_pct": 100.0 * Tm / n_myo, "scatter_fraction_myo": Sm / max(Tm + Sm, 1),
            "nec_myo_per_decay_pct": 100.0 * nec_m,
            "median_decays_per_s": float(np.median([m["decays_per_s"] for m in metas if m.get("decays_per_s")])),
        }
        dt_true = (d["t1"] - d["t2"]) * 1000.0
        for ctr in C.CTR_SWEEP_PS:
            dx = C.C_MM_PER_PS * ctr / 2.0
            gain = np.maximum(geo["chord"][true & myo] / dx, 1.0).mean() if Tm else np.nan
            row[f"tof_gain_{ctr}ps"] = float(gain)
            row[f"tof_eff_nec_myo_{ctr}ps_pct"] = float(100.0 * nec_m * gain)
            sel = np.flatnonzero(true)[:200000]
            dt = dt_true[sel] + rng.normal(size=sel.size) * ctr / 2.3548
            err = (geo["D"][sel] + C.C_MM_PER_PS * dt) / 2.0 - geo["s_true"][sel]
            row[f"tof_fwhm_mm_{ctr}ps"] = float(2.3548 * np.std(err)) if sel.size else np.nan
        rows.append(row)
        h, _ = np.histogram(geo["phi"][true & myo], bins=phi_bins)
        ang[(cfg, body)] = (h, n_myo)
        print(f"loaded {cfg} {body}: {n_dec:,} decays, {T:,} trues")

    if not rows:
        raise SystemExit(f"No sweep-2 data in {results}")

    ref = {r["body"]: r for r in rows if r["config"] == C.SWEEP2_REFERENCE}
    cmax = max(C.CTR_SWEEP_PS)
    for r in rows:
        f = ref.get(r["body"])
        if f:
            r["rel_nec_myo"] = r["nec_myo_per_decay_pct"] / f["nec_myo_per_decay_pct"]
            r["rel_sens_myo"] = r["sens_myo_trues_pct"] / f["sens_myo_trues_pct"]
            r["rel_crystals"] = r["n_crystals"] / f["n_crystals"]
            r["rel_nec_per_crystal"] = r["rel_nec_myo"] / r["rel_crystals"]
            for ctr in C.CTR_SWEEP_PS:
                r[f"rel_tof_eff_nec_{ctr}ps_vs_closed70_{cmax}ps"] = (
                    r[f"tof_eff_nec_myo_{ctr}ps_pct"] / f[f"tof_eff_nec_myo_{cmax}ps_pct"])

    outdir.mkdir(parents=True, exist_ok=True)
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with open(outdir / "summary2.csv", "w") as fh:
        fh.write(",".join(keys) + "\n")
        for r in rows:
            fh.write(",".join(str(r.get(k, "")) for k in keys) + "\n")
    with open(outdir / "summary2.json", "w") as fh:
        json.dump(rows, fh, indent=2)
    with open(outdir / "angular2.csv", "w") as fh:
        combos = sorted(ang)
        fh.write("phi_lo,phi_hi," + ",".join(f"{c}_{b}" for c, b in combos) + "\n")
        for i in range(len(phi_bins) - 1):
            fh.write(f"{phi_bins[i]},{phi_bins[i + 1]}," + ",".join(str(int(ang[k][0][i])) for k in combos) + "\n")

    # ---------------------------------------------------------------- figure: counts
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8, "text.color": INK,
                         "axes.labelcolor": INK, "axes.titlesize": 9})
    cfgs = [c[0] for c in C.SWEEP2_CONFIGS if any(r["config"] == c[0] for r in rows)]
    bodies = [b for b in C.SWEEP2_BODIES if any(r["body"] == b for r in rows)]
    get = {(r["config"], r["body"]): r for r in rows}
    fig, axs = plt.subplots(2, 2, figsize=(7.2, 6.4))
    fig.subplots_adjust(left=0.1, right=0.98, top=0.94, bottom=0.12, wspace=0.3, hspace=0.45)
    w = 0.38
    x = np.arange(len(cfgs))
    hatch = {bodies[0]: "", bodies[-1]: "////"} if bodies else {}
    panels = [("sens_myo_trues_pct", "A.  Myocardial sensitivity", "Trues per myocardial decay (%)"),
              ("rel_nec_myo", "B.  NEC relative to 70 cm closed ring", "Myocardial NEC, same body = 1"),
              ("scatter_fraction_myo", "D.  Scatter fraction", "Myocardial scatter fraction")]
    for ax, (key, title, ylab) in zip([axs[0, 0], axs[0, 1], axs[1, 1]], panels):
        style(ax)
        for j, b in enumerate(bodies):
            vals = [get.get((c, b), {}).get(key, np.nan) for c in cfgs]
            ax.bar(x + (j - 0.5) * w, vals, width=w, color=[COL[c] for c in cfgs], edgecolor="white",
                   lw=1.5, hatch=hatch[b])
            for xi, v in zip(x, vals):
                if np.isfinite(v):
                    ax.text(xi + (j - 0.5) * w, v, f"{v:.2f}", ha="center", va="bottom", fontsize=6.5, color=INK)
        ax.set_xticks(x, [NAME[c] for c in cfgs], rotation=25, ha="right", fontsize=7.5)
        ax.set_title(title, loc="left", fontweight="bold")
        ax.set_ylabel(ylab)
    axs[0, 1].axhline(1.0, color=INK2, lw=0.8, ls=(0, (3, 2)))
    handles = [Patch(facecolor="#9a9994", edgecolor="white", hatch=hatch[b], label=f"{b} body") for b in bodies]
    axs[0, 0].legend(handles=handles, fontsize=7, frameon=False, loc="upper left", ncol=2)
    ymax = np.nanmax([r["sens_myo_trues_pct"] for r in rows])
    axs[0, 0].set_ylim(0, 1.25 * ymax)

    ax = axs[1, 0]
    style(ax)
    body_tof = bodies[-1] if bodies else None
    for c in cfgs:
        r = get.get((c, body_tof))
        if not r or f"rel_tof_eff_nec_{cmax}ps_vs_closed70_{cmax}ps" not in r:
            continue
        vals = [r[f"rel_tof_eff_nec_{ctr}ps_vs_closed70_{cmax}ps"] for ctr in C.CTR_SWEEP_PS]
        ax.plot(C.CTR_SWEEP_PS, vals, "-", marker=MARK[c], color=COL[c], lw=2.2 if c == "R280_slot" else 1.5,
                ms=5, mec="white", mew=0.8)
        ax.text(C.CTR_SWEEP_PS[0] - 6, vals[0], f"{NAME[c]}  {vals[0]:.2f}", ha="right", va="center", fontsize=7)
    ax.plot([C.CTR_SWEEP_PS[0], cmax + 8], [1, 1], color=INK2, lw=0.8, ls=(0, (3, 2)))
    ax.set_xlim(C.CTR_SWEEP_PS[0] - 95, cmax + 8)
    ax.set_xticks(C.CTR_SWEEP_PS)
    ax.set_xlabel("Coincidence timing resolution, FWHM (ps)")
    ax.set_ylabel(f"TOF-effective myocardial NEC\nrelative to 70 cm closed ring at {cmax} ps")
    ax.set_title(f"C.  Timing vs geometry ({body_tof} body)", loc="left", fontweight="bold")
    for ext in ("png", "pdf"):
        fig.savefig(outdir / f"Fig_sweep2_counts.{ext}", dpi=300)
    plt.close(fig)

    # ---------------------------------------------------------------- figure: angular sampling
    fig, axs = plt.subplots(1, len(bodies), figsize=(7.2, 3.2), squeeze=False)
    centres = phi_bins[:-1] + 5
    for i, b in enumerate(bodies):
        ax = axs[0, i]
        style(ax)
        h0, n0 = ang.get((C.SWEEP2_REFERENCE, b), (None, None))
        if h0 is None:
            continue
        for c in cfgs:
            h, n = ang.get((c, b), (None, None))
            if h is None:
                continue
            ratio = (h / n) / (h0 / n0)
            ax.plot(centres, ratio, "-", marker=MARK[c], color=COL[c], lw=2.2 if c == "R280_slot" else 1.4,
                    ms=4, mec="white", mew=0.6, label=NAME[c])
        ax.set_ylim(0, 1.25)
        ax.set_xlim(0, 180)
        ax.set_xticks([0, 30, 60, 90, 120, 150, 180])
        ax.set_xlabel("Transaxial LOR angle (deg)\n0 = left–right, 90 = anterior–posterior")
        ax.set_ylabel("Myocardial trues relative\nto 70 cm closed ring")
        ax.set_title(f"{'AB'[i]}.  {b} body", loc="left", fontweight="bold")
        if i == 0:
            ax.legend(fontsize=6.5, frameon=False, loc="lower center")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(outdir / f"Fig_sweep2_angular.{ext}", dpi=300)
    plt.close(fig)

    print(f"\n{'config':12s} {'body':4s} {'bore':>5s} {'cover':>6s} {'crystals':>8s} {'myo sens%':>9s} "
          f"{'SF':>5s} {'relNEC':>6s} {'NEC/cryst':>9s} {'TOF200 vs closed@320':>21s}")
    for r in rows:
        print(f"{r['config']:12s} {r['body']:4s} {r['bore_cm']:5.0f} {r['coverage_deg']:5.0f}° {r['n_crystals']:8,d} "
              f"{r['sens_myo_trues_pct']:9.3f} {r['scatter_fraction_myo']:5.2f} {r.get('rel_nec_myo', np.nan):6.2f} "
              f"{r.get('rel_nec_per_crystal', np.nan):9.2f} "
              f"{r.get(f'rel_tof_eff_nec_{min(C.CTR_SWEEP_PS)}ps_vs_closed70_{cmax}ps', np.nan):21.2f}")
    print(f"\nWrote {outdir}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True)
    args = ap.parse_args()
    res = Path(args.results)
    analyse(res, res / "analysis")


if __name__ == "__main__":
    main()

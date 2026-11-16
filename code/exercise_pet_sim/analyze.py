"""
Combine the coincidence files from all jobs and compute the geometry-study metrics.

    python analyze.py --results /scratch/$USER/exercise_pet

Expects  <results>/gapXX/coinc_*.npz  (written by run_sim.py).
Writes   <results>/analysis/summary.csv, summary.json and three figures.

Metrics per ring configuration (no randoms; randoms are added analytically later):
  - sensitivity: true coincidences per decay, whole phantom and myocardium only
  - scatter fraction
  - NEC per decay = T^2 / (T + S)
  - transaxial angular coverage of myocardial LORs, relative to the closed ring
  - TOF effective NEC for each CTR: NEC x mean(D / dx), D = LOR chord through the torso,
    dx = c * CTR / 2 (Conti 2006 approximation)
  - measured TOF localisation FWHM along the LOR (a check on the time blurring)
"""

import argparse
import json
from pathlib import Path

import numpy as np

import config as C

HERE = Path(__file__).resolve().parent


def load_gap(folder):
    files = sorted(folder.glob("coinc_*.npz"))
    if not files:
        return None
    parts, metas = [], []
    for j, f in enumerate(files):
        with np.load(f, allow_pickle=False) as z:
            meta = json.loads(str(z["meta"]))
            metas.append(meta)
            if meta.get("n_coincidences", 0) > 0:
                p = {k: z[k] for k in z.files if k != "meta"}
                p["job"] = np.full(len(p["x1"]), j, dtype=np.int16)
                parts.append(p)
    data = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]} if parts else {}
    n_decays = sum(m["n_decays"] for m in metas)
    return data, n_decays, metas


def label_lookup(ph, ex, ey, ez):
    """Phantom label at each emission point; 255 outside the voxel image (e.g. the arms)."""
    x, y, z = ph["x"], ph["y"], ph["z"]
    v = x[1] - x[0]
    ix = np.round((ex - x[0]) / v).astype(int)
    iy = np.round((ey - y[0]) / v).astype(int)
    iz = np.round((ez - z[0]) / v).astype(int)
    ok = (ix >= 0) & (ix < len(x)) & (iy >= 0) & (iy < len(y)) & (iz >= 0) & (iz < len(z))
    out = np.full(ex.shape, 255, dtype=np.int32)
    out[ok] = ph["labels"][iz[ok], iy[ok], ix[ok]]
    return out


def lor_geometry(d, semi_axes=None):
    p1 = np.stack([d["x1"], d["y1"], d["z1"]], 1).astype(np.float64)
    p2 = np.stack([d["x2"], d["y2"], d["z2"]], 1).astype(np.float64)
    e = np.stack([d["ex"], d["ey"], d["ez"]], 1).astype(np.float64)
    v = p2 - p1
    D = np.linalg.norm(v, axis=1)
    n = v / D[:, None]
    # distance from emission point to LOR
    dist = np.linalg.norm(np.cross(e - p1, n), axis=1)
    # position of emission point along the LOR, measured from p1
    s_true = np.einsum("ij,ij->i", e - p1, n)
    phi = np.degrees(np.arctan2(v[:, 1], v[:, 0])) % 180.0
    # chord through torso ellipse (transaxial), as 3D length
    a, b = semi_axes or C.TORSO_SEMI_AXES
    A = (v[:, 0] / a) ** 2 + (v[:, 1] / b) ** 2
    B = 2 * (p1[:, 0] * v[:, 0] / a ** 2 + p1[:, 1] * v[:, 1] / b ** 2)
    Cq = (p1[:, 0] / a) ** 2 + (p1[:, 1] / b) ** 2 - 1
    disc = np.maximum(B ** 2 - 4 * A * Cq, 0)
    chord = np.sqrt(disc) / np.maximum(A, 1e-12) * D
    return dict(D=D, dist=dist, s_true=s_true, phi=phi, chord=chord)


def analyse(results_dir, outdir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ph = dict(np.load(HERE / "data" / "phantom.npz"))
    act = ph["activity"]
    frac_myo = act[ph["labels"] == C.LABELS["myocardium"]].sum() / act.sum()
    rng = np.random.default_rng(2026)

    rows, angular = [], {}
    phi_bins = np.arange(0, 180 + 5, 5)
    gaps = sorted(int(p.name[3:]) for p in results_dir.glob("gap*") if p.is_dir())
    if not gaps:
        raise SystemExit(f"No gapXX folders in {results_dir}")

    for g in gaps:
        loaded = load_gap(results_dir / f"gap{g:02d}")
        if loaded is None or not loaded[0]:
            print(f"gap {g}: no data")
            continue
        d, n_dec, metas = loaded
        geo = lor_geometry(d)
        true = geo["dist"] < C.SCATTER_DIST_MM
        lab = label_lookup(ph, d["ex"], d["ey"], d["ez"])
        myo = lab == C.LABELS["myocardium"]

        T, S = int(true.sum()), int((~true).sum())
        Tm, Sm = int((true & myo).sum()), int((~true & myo).sum())
        n_myo = n_dec * frac_myo
        nec = T ** 2 / max(T + S, 1) / n_dec
        nec_m = Tm ** 2 / max(Tm + Sm, 1) / n_myo

        row = {
            "gap_per_side_deg": g,
            "angular_coverage_deg": 360 - 2 * g,
            "n_modules": metas[0]["n_modules"],
            "n_jobs": len(metas),
            "decays": n_dec,
            "prompts": T + S,
            "trues": T,
            "scatter_fraction": S / max(T + S, 1),
            "sens_trues_pct": 100.0 * T / n_dec,
            "sens_myo_trues_pct": 100.0 * Tm / n_myo,
            "scatter_fraction_myo": Sm / max(Tm + Sm, 1),
            "nec_per_decay_pct": 100.0 * nec,
            "nec_myo_per_decay_pct": 100.0 * nec_m,
            "median_decays_per_s": float(np.median([m["decays_per_s"] for m in metas if m.get("decays_per_s")])),
        }

        # TOF: blur the coincidence time difference, localise along the LOR
        dt_true_ps = (d["t1"] - d["t2"]) * 1000.0
        for ctr in C.CTR_SWEEP_PS:
            dx = C.C_MM_PER_PS * ctr / 2.0
            gain = np.maximum(geo["chord"][true & myo] / dx, 1.0).mean() if Tm else np.nan
            row[f"tof_gain_{ctr}ps"] = float(gain)
            row[f"tof_eff_nec_myo_{ctr}ps_pct"] = float(100.0 * nec_m * gain)
            sel = np.flatnonzero(true)[:200000]
            dt = dt_true_ps[sel] + rng.normal(size=sel.size) * ctr / 2.3548
            s_hat = (geo["D"][sel] + C.C_MM_PER_PS * dt) / 2.0
            err = s_hat - geo["s_true"][sel]
            row[f"tof_fwhm_mm_{ctr}ps"] = float(2.3548 * np.std(err)) if sel.size else np.nan
        rows.append(row)

        sel = true & myo
        h, _ = np.histogram(geo["phi"][sel], bins=phi_bins)
        hj = np.stack([np.histogram(geo["phi"][sel & (d["job"] == j)], bins=phi_bins)[0]
                       for j in range(len(metas))])
        ha, _ = np.histogram(geo["phi"][true], bins=phi_bins)
        angular[g] = dict(counts=h, n_myo=n_myo, per_job=hj, all_trues=ha, n_dec=n_dec,
                          job_decays=np.array([m["n_decays"] for m in metas], dtype=float))

    if not rows:
        raise SystemExit("No data found.")

    ref = next((r for r in rows if r["gap_per_side_deg"] == 0), None)
    for r in rows:
        for k in ("sens_myo_trues_pct", "nec_myo_per_decay_pct", "n_modules"):
            r[f"rel_{k}"] = r[k] / ref[k] if ref else np.nan
        if ref:
            for ctr in C.CTR_SWEEP_PS:
                r[f"rel_tof_eff_nec_{ctr}ps_vs_closed_ring_{max(C.CTR_SWEEP_PS)}ps"] = (
                    r[f"tof_eff_nec_myo_{ctr}ps_pct"] / ref[f"tof_eff_nec_myo_{max(C.CTR_SWEEP_PS)}ps_pct"]
                )

    outdir.mkdir(parents=True, exist_ok=True)
    keys = list(rows[0].keys())
    with open(outdir / "summary.csv", "w") as f:
        f.write(",".join(keys) + "\n")
        for r in rows:
            f.write(",".join(f"{r.get(k, '')}" for k in keys) + "\n")
    with open(outdir / "summary.json", "w") as f:
        json.dump(rows, f, indent=2)

    # ---------------------------------------------------------------- figures
    colors = {0: "#4a4a4a", 20: "#7aa6d9", 30: "#1f5fa8", 60: "#c0504d"}
    labels = {0: "360° closed ring", 20: "320°", 30: "300° (proposed)", 60: "240° (2017)"}
    gl = [r["gap_per_side_deg"] for r in rows]

    fig, ax = plt.subplots(1, 3, figsize=(13, 4))
    x = np.arange(len(rows))
    for i, (key, title) in enumerate([
        ("sens_myo_trues_pct", "Myocardial sensitivity\n(trues per myocardial decay, %)"),
        ("nec_myo_per_decay_pct", "Myocardial NEC per decay (%)\n(no randoms, non-TOF)"),
        ("scatter_fraction_myo", "Scatter fraction, myocardial events"),
    ]):
        vals = [r[key] for r in rows]
        ax[i].bar(x, vals, color=[colors.get(g, "#888") for g in gl])
        ax[i].set_xticks(x, [labels.get(g, f"{g}°") for g in gl], rotation=20, fontsize=8)
        ax[i].set_title(title, fontsize=10)
        if ref and key != "scatter_fraction_myo":
            for xi, r in zip(x, rows):
                ax[i].text(xi, r[key], f"{r[key] / ref[key]:.2f}×", ha="center", va="bottom", fontsize=8)
    fig.tight_layout()
    fig.savefig(outdir / "fig1_sensitivity_vs_gap.png", dpi=150)
    plt.close(fig)

    centres = 0.5 * (phi_bins[1:] + phi_bins[:-1])

    # angular counts to disk (raw counts; divide by n_myo_decays for per-decay values)
    with open(outdir / "angular.csv", "w") as f:
        cols = ["phi_lo", "phi_hi"]
        for g in gl:
            cols += [f"myo_trues_gap{g}", f"all_trues_gap{g}"]
        f.write(",".join(cols) + "\n")
        for i in range(len(centres)):
            vals = [phi_bins[i], phi_bins[i + 1]]
            for g in gl:
                vals += [int(angular[g]["counts"][i]), int(angular[g]["all_trues"][i])]
            f.write(",".join(str(v) for v in vals) + "\n")
    with open(outdir / "angular_meta.json", "w") as f:
        json.dump({str(g): {"n_myo_decays": angular[g]["n_myo"], "n_decays": angular[g]["n_dec"]}
                   for g in gl}, f, indent=2)

    if 0 in angular:
        fig, axs = plt.subplots(1, 2, figsize=(13, 4.2))
        n0 = angular[0]["counts"].astype(float)
        r0 = n0 / angular[0]["n_myo"]
        for g in gl:
            ng = angular[g]["counts"].astype(float)
            rg = ng / angular[g]["n_myo"]
            with np.errstate(divide="ignore", invalid="ignore"):
                ratio = np.where(n0 > 0, rg / r0, np.nan)
                err = ratio * np.sqrt(1 / np.maximum(ng, 1) + 1 / np.maximum(n0, 1))
            c = colors.get(g, "#888")
            axs[0].errorbar(centres, ratio, yerr=err, fmt="-o", ms=3, lw=1, capsize=2, color=c,
                            label=labels.get(g, f"{g}°"))
            axs[1].errorbar(centres, 1e6 * rg, yerr=1e6 * np.sqrt(ng) / angular[g]["n_myo"],
                            fmt="-o", ms=3, lw=1, capsize=2, color=c, label=labels.get(g, f"{g}°"))
        axs[0].set_ylabel("Myocardial trues relative to closed ring")
        axs[0].set_ylim(0, 1.3)
        axs[0].axhline(1, color="k", lw=0.5)
        axs[0].set_title("Angular sampling relative to closed ring (±1σ)", fontsize=10)
        axs[1].set_ylabel("Myocardial trues per 10⁶ myocardial decays, per 5° bin")
        axs[1].set_title("Absolute angular distribution (±1σ)", fontsize=10)
        for a in axs:
            a.set_xlabel("Transaxial LOR angle φ (deg); 0 = left-right, 90 = anterior-posterior")
            a.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(outdir / "fig2_angular_coverage.png", dpi=150)
        plt.close(fig)

        # diagnostics: bins where a gapped ring differs from the closed ring where both
        # should see the same modules, and per-job consistency within each configuration
        print("\nAngular diagnostics")
        for g in gl:
            if g == 0:
                continue
            ng = angular[g]["counts"].astype(float)
            rg = ng / angular[g]["n_myo"]
            with np.errstate(divide="ignore", invalid="ignore"):
                ratio = rg / r0
                sig = ratio * np.sqrt(1 / np.maximum(ng, 1) + 1 / np.maximum(n0, 1))
                z = (ratio - 1) / sig
            full = (centres > 45) & (centres < 135)  # angles no gap removes
            flag = np.flatnonzero(full & (np.abs(z) > 3))
            txt = ", ".join(f"{centres[i]:.1f}° ratio {ratio[i]:.2f} (z={z[i]:+.1f})" for i in flag)
            print(f"  gap {g}: bins >3σ from closed ring within 45-135°: {txt or 'none'}")
        for g in gl:
            a = angular[g]
            frac = a["job_decays"] / a["job_decays"].sum()
            exp = np.outer(frac, a["counts"])
            with np.errstate(divide="ignore", invalid="ignore"):
                zj = np.where(exp > 0, (a["per_job"] - exp) / np.sqrt(exp), 0)
            worst = np.unravel_index(np.argmax(np.abs(zj)), zj.shape)
            nbins_bad = int((np.abs(zj) > 4).sum())
            print(f"  gap {g}: per-job check, cells >4σ: {nbins_bad}; worst job {worst[0]} at "
                  f"{centres[worst[1]]:.1f}° (z={zj[worst]:+.1f})")
        i52 = int(np.argmin(np.abs(centres - 52.5)))
        print("  52.5° bin, myocardial trues per 1e6 myocardial decays: " + ", ".join(
            f"gap {g}: {1e6 * angular[g]['counts'][i52] / angular[g]['n_myo']:.1f} "
            f"(neighbours {1e6 * angular[g]['counts'][i52 - 1] / angular[g]['n_myo']:.1f}, "
            f"{1e6 * angular[g]['counts'][i52 + 1] / angular[g]['n_myo']:.1f})" for g in gl))

    fig, ax = plt.subplots(figsize=(7, 4))
    for r in rows:
        g = r["gap_per_side_deg"]
        ax.plot(C.CTR_SWEEP_PS, [r[f"tof_eff_nec_myo_{c}ps_pct"] for c in C.CTR_SWEEP_PS], "-o",
                color=colors.get(g, "#888"), label=labels.get(g, f"{g}°"))
    ax.set_xlabel("Coincidence timing resolution, FWHM (ps)")
    ax.set_ylabel("TOF-effective myocardial NEC per decay (%)")
    ax.set_title("Geometry vs timing: effective sensitivity", fontsize=10)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(outdir / "fig3_tof_effective_nec.png", dpi=150)
    plt.close(fig)

    # ---------------------------------------------------------------- console table
    print(f"\nMyocardium holds {100 * frac_myo:.1f}% of decays.\n")
    hdr = f"{'coverage':>9} {'modules':>7} {'decays':>11} {'myo sens %':>10} {'rel':>5} {'SF':>5} {'NEC %':>7} {'rel':>5}"
    print(hdr)
    for r in rows:
        print(f"{r['angular_coverage_deg']:>8}° {r['n_modules']:>7} {r['decays']:>11,} "
              f"{r['sens_myo_trues_pct']:>10.3f} {r.get('rel_sens_myo_trues_pct', np.nan):>5.2f} "
              f"{r['scatter_fraction_myo']:>5.2f} {r['nec_myo_per_decay_pct']:>7.3f} "
              f"{r.get('rel_nec_myo_per_decay_pct', np.nan):>5.2f}")
    print("\nTOF localisation FWHM along LOR (mm), check vs c*CTR/2:")
    for ctr in C.CTR_SWEEP_PS:
        vals = ", ".join(f"{r[f'tof_fwhm_mm_{ctr}ps']:.1f}" for r in rows)
        print(f"  {ctr} ps: expected {C.C_MM_PER_PS * ctr / 2:.1f}; measured {vals}")
    print(f"\nWrote {outdir}/summary.csv and figures.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True, help="folder containing gapXX subfolders")
    args = ap.parse_args()
    res = Path(args.results)
    analyse(res, res / "analysis")


if __name__ == "__main__":
    main()

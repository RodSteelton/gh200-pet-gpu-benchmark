"""
Grant Figure 1: split-ring geometry with arms, and the ring-coverage results.

    python make_fig1.py --results /scratch/$USER/exercise_pet
reads  <results>/analysis/summary.json and angular.csv
writes <results>/analysis/Figure1.png (300 dpi) and Figure1.pdf
"""

import argparse
import csv
import json
from pathlib import Path

import numpy as np

import config as C
import scanner

HERE = Path(__file__).resolve().parent

# validated categorical palette (colorblind-safe; direct labels provide secondary encoding)
COL = {0: "#4a3aa7", 20: "#1baf7a", 30: "#2a78d6", 60: "#eb6834"}
NAME = {0: "360° closed ring", 20: "320°", 30: "300° split ring (proposed)", 60: "240° panels (2017)"}
SHORT = {0: "360°", 20: "320°", 30: "300°", 60: "240°"}
MARK = {0: "s", 20: "^", 30: "o", 60: "D"}
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


def panel_geometry(ax, gap=30):
    import matplotlib
    from matplotlib.patches import Rectangle
    import matplotlib.transforms as mt

    # darkness proportional to Hounsfield units: air (-1000 HU) white, bone darkest;
    # myocardial tracer uptake overlaid in colour
    ph = np.load(HERE / "data" / "phantom.npz")
    lab, x, y, z = ph["labels"], ph["x"], ph["y"], ph["z"]
    iz = int(np.argmin(np.abs(z)))
    L = C.LABELS
    hu_of = {L["air"]: -1000, L["soft_tissue"]: 40, L["lung"]: -740, L["bone"]: 700,
             L["myocardium"]: 40, L["blood_pool"]: 45, L["liver"]: 55}
    px = 1.5
    g = np.arange(-345, 385 + px, px)
    GX, GY = np.meshgrid(g, np.arange(-330, 335 + px, px))
    v = C.VOXEL_MM
    ix = np.round((GX - x[0]) / v).astype(int)
    iy = np.round((GY - y[0]) / v).astype(int)
    inside = (ix >= 0) & (ix < len(x)) & (iy >= 0) & (iy < len(y))
    labs = np.zeros(GX.shape, dtype=int)
    labs[inside] = lab[iz][iy[inside], ix[inside]]
    hu = np.vectorize(hu_of.get)(labs).astype(float)
    # upper arms hanging alongside the chest (soft tissue with humerus), through the lateral slots
    arm_w, arm_h = 110.0, 120.0
    ax_c = C.TORSO_SEMI_AXES[0] + arm_w / 2 + 3
    for sgn in (1, -1):
        arm = ((GX - sgn * ax_c) / (arm_w / 2)) ** 2 + (GY / (arm_h / 2)) ** 2 <= 1
        hu[arm] = 40
        hu[(GX - sgn * (ax_c + 5)) ** 2 + (GY + 5) ** 2 <= 12 ** 2] = 1000
    img = 1.0 - np.clip((hu + 1000) / 2200, 0, 1)
    cmap = matplotlib.colormaps["gray"]
    ext = [g[0] - px / 2, g[-1] + px / 2, -330 - px / 2, 335 + px / 2]
    ax.imshow(img, origin="lower", cmap=cmap, vmin=0, vmax=1, extent=ext, interpolation="nearest")
    myo = labs == L["myocardium"]
    rgba = np.zeros(GX.shape + (4,))
    rgba[myo] = matplotlib.colors.to_rgba("#e34948", 0.9)
    ax.imshow(rgba, origin="lower", extent=ext, interpolation="nearest")

    rc = scanner.module_centre_radius()
    depth = C.CRYSTAL_SIZE[0]
    width = C.CRYSTALS_PER_MODULE_T * C.CRYSTAL_PITCH
    kept = set(np.round(scanner.module_angles(gap), 3))
    for a in scanner.module_angles(0):
        t = np.deg2rad(a)
        on = round(a, 3) in kept
        r = Rectangle((-depth / 2, -width / 2), depth, width,
                      facecolor=COL[gap] if on else "none",
                      edgecolor=COL[gap] if on else "#b8b7b1",
                      lw=0.4 if on else 0.6, ls="-" if on else (0, (2, 1.5)))
        r.set_transform(mt.Affine2D().rotate(t).translate(rc * np.cos(t), rc * np.sin(t)) + ax.transData)
        ax.add_patch(r)

    # slot width at the crystal face
    R = C.RING_INNER_RADIUS
    half = np.deg2rad(gap / 2)
    xs, ys = R * np.cos(half), R * np.sin(half)
    ax.annotate("", xy=(xs + 45, ys), xytext=(xs + 45, -ys),
                arrowprops=dict(arrowstyle="<->", color=INK, lw=0.8, shrinkA=0, shrinkB=0))
    for yy in (ys, -ys):
        ax.plot([xs, xs + 50], [yy, yy], color=INK2, lw=0.5)
    ax.text(xs + 55, 0, f"slot\n{2 * R * np.sin(half) / 10:.1f} cm\n\narm\n≈{arm_w / 10:.0f} cm", fontsize=7.5, color=INK, va="center")
    ax.text(80, 70, "LV", fontsize=7, color=INK, ha="left", va="bottom", fontweight="bold")
    ax.text(0, 318, "anterior", fontsize=7, color=INK2, ha="center", va="bottom")
    ax.text(-345, 335, "dashed: detectors\nremoved for slots", fontsize=6.5, color=INK2, va="top")

    ax.set_xlim(-345, 385)
    ax.set_ylim(-330, 335)
    ax.set_aspect("equal")
    ax.axis("off")


def load(results):
    a = results / "analysis"
    rows = {int(r["gap_per_side_deg"]): r for r in json.load(open(a / "summary.json"))}
    ang = list(csv.DictReader(open(a / "angular.csv")))
    return rows, ang


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True)
    args = ap.parse_args()
    results = Path(args.results)
    rows, ang = load(results)
    gaps = [g for g in (0, 20, 30, 60) if g in rows]

    ph = np.load(HERE / "data" / "phantom.npz")
    frac_myo = ph["activity"][ph["labels"] == C.LABELS["myocardium"]].sum() / ph["activity"].sum()

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8, "text.color": INK,
                         "axes.labelcolor": INK, "axes.titlesize": 9})
    fig = plt.figure(figsize=(7.2, 6.6))
    gs = fig.add_gridspec(2, 2, wspace=0.28, hspace=0.42, left=0.10, right=0.98, top=0.95, bottom=0.08)

    # (a) geometry
    axA = fig.add_subplot(gs[0, 0])
    panel_geometry(axA, 30)
    axA.set_title("A.  300° split ring, arms in slots", loc="left", fontsize=9, fontweight="bold")

    # (b) relative myocardial NEC vs fraction of detectors
    axB = fig.add_subplot(gs[0, 1])
    style(axB)
    xs = np.arange(len(gaps))
    nec = [rows[g]["rel_nec_myo_per_decay_pct"] for g in gaps]
    det = [rows[g]["rel_n_modules"] for g in gaps]
    axB.bar(xs, nec, width=0.62, color=[COL[g] for g in gaps], edgecolor="white", lw=2)
    for xi, n, d in zip(xs, nec, det):
        axB.plot([xi - 0.36, xi + 0.36], [d, d], color=INK, lw=1.2)
        axB.text(xi, n - 0.04, f"{n:.2f}", ha="center", va="top", fontsize=8, color="white", fontweight="bold")
    axB.text(xs[-1] + 0.4, det[-1], "fraction of\ndetectors", fontsize=6.5, color=INK, va="center", ha="left")
    axB.set_xticks(xs, [SHORT[g] for g in gaps])
    axB.set_xlim(-0.5, len(gaps) - 0.5 + 0.6)
    axB.set_ylim(0, 1.1)
    axB.set_ylabel("Myocardial NEC relative to closed ring")
    axB.set_title("B.  Counts follow detector count", loc="left", fontsize=9, fontweight="bold")

    # (c) TOF-effective NEC vs CTR, relative to closed ring at 320 ps
    axC = fig.add_subplot(gs[1, 0])
    style(axC)
    ctrs = C.CTR_SWEEP_PS
    ref = rows[0][f"tof_eff_nec_myo_{max(ctrs)}ps_pct"]
    for g in gaps:
        vals = [rows[g][f"tof_eff_nec_myo_{c}ps_pct"] / ref for c in ctrs]
        axC.plot(ctrs, vals, "-", marker=MARK[g], ms=5, lw=2.4 if g == 30 else 1.6, color=COL[g],
                 mec="white", mew=0.8)
        axC.text(ctrs[0] - 6, vals[0], f"{SHORT[g]}  {vals[0]:.2f}", ha="right", va="center", fontsize=7.5, color=INK)
    axC.plot([ctrs[0], ctrs[-1] + 8], [1.0, 1.0], color=INK2, lw=0.8, ls=(0, (3, 2)), zorder=1)
    axC.text(ctrs[0] - 6, 1.0, "reference  1.00", ha="right", va="center", fontsize=7.5, color=INK2)
    axC.set_xlim(ctrs[0] - 62, ctrs[-1] + 8)
    axC.set_xticks(ctrs)
    axC.set_xlabel("Coincidence timing resolution, FWHM (ps)")
    axC.set_ylabel("TOF-effective myocardial NEC\nrelative to closed ring at 320 ps (dashed)")
    axC.set_title("C.  Timing resolution offsets the slots", loc="left", fontsize=9, fontweight="bold")

    # (d) angular sampling, 10 degree bins, relative to closed ring
    axD = fig.add_subplot(gs[1, 1])
    style(axD)
    lo = np.array([float(r["phi_lo"]) for r in ang])
    edges10 = np.arange(0, 181, 10)
    centres = edges10[:-1] + 5
    n_myo = {g: rows[g]["decays"] * frac_myo for g in gaps}

    def rebin(g):
        c = np.array([float(r[f"myo_trues_gap{g}"]) for r in ang])
        return np.array([c[(lo >= a) & (lo < a + 10)].sum() for a in edges10[:-1]])

    c0 = rebin(0)
    for g in gaps:
        cg = rebin(g)
        ratio = (cg / n_myo[g]) / (c0 / n_myo[0])
        err = ratio * np.sqrt(1 / np.maximum(cg, 1) + 1 / np.maximum(c0, 1))
        err[cg == 0] = 0
        axD.errorbar(centres, ratio, yerr=err if g else None, fmt="-", marker=MARK[g], ms=4,
                     lw=2.4 if g == 30 else 1.4, color=COL[g], mec="white", mew=0.6, capsize=0,
                     elinewidth=0.8)
    from matplotlib.lines import Line2D
    handles = [Line2D([], [], color=COL[g], marker=MARK[g], lw=2.4 if g == 30 else 1.4, ms=4,
                      mec="white", mew=0.6, label=NAME[g]) for g in gaps]
    axD.legend(handles=handles, loc="lower center", fontsize=6.5, frameon=False, handlelength=2.2)
    axD.set_xlim(0, 180)
    axD.set_ylim(0, 1.2)
    axD.set_xticks([0, 30, 60, 90, 120, 150, 180])
    axD.set_xlabel("Transaxial LOR angle (deg)\n0 = left–right, 90 = anterior–posterior")
    axD.set_ylabel("Myocardial trues relative to closed ring")
    axD.set_title("D.  Slots remove only near-lateral LORs", loc="left", fontsize=9, fontweight="bold")

    out = results / "analysis"
    fig.savefig(out / "Figure1.png", dpi=300)
    fig.savefig(out / "Figure1.pdf")
    print("Wrote", out / "Figure1.png")


if __name__ == "__main__":
    main()

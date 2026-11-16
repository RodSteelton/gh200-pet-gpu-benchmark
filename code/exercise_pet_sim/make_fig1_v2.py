"""
Grant Figure 1 (v2): the 56 cm, 290° split ring with arm slots and curved carbon-fibre backrest (dark gray), and the
sweep-2 comparison against 62 cm and 70 cm split rings and a 70 cm closed ring.

    python make_fig1_v2.py --sweep2 <folder with summary2.json and angular2.csv> --out <folder>
writes <out>/Figure1_290.png (300 dpi) and Figure1_290.pdf
"""

import argparse
import csv
import json
from pathlib import Path

import numpy as np

import config as C
import scanner

HERE = Path(__file__).resolve().parent
CFGS = ["R280_slot", "R310_slot", "R350_slot", "R350_closed"]
COL = {"R280_slot": "#2a78d6", "R310_slot": "#1baf7a", "R350_slot": "#eb6834", "R350_closed": "#4a3aa7"}
NAME = {"R280_slot": "56 cm, 290° (proposed)", "R310_slot": "62 cm, 297°", "R350_slot": "70 cm, 320°",
        "R350_closed": "70 cm, closed"}
SHORT = {"R280_slot": "56 cm\n290°", "R310_slot": "62 cm\n297°", "R350_slot": "70 cm\n320°",
         "R350_closed": "70 cm\nclosed"}
MARK = {"R280_slot": "o", "R310_slot": "^", "R350_slot": "D", "R350_closed": "s"}
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
BACKREST_COL = "#3a3a3a"                      # dark gray
BODY = "P90"
BACK_HALF_WIDTH, BACK_MM = 190.0, 24.0


def style(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(INK2)
        ax.spines[s].set_linewidth(0.8)
    ax.tick_params(colors=INK2, labelsize=8, width=0.8)
    ax.grid(axis="y", color=GRID, lw=0.6)
    ax.set_axisbelow(True)


def dodge(values, min_gap):
    """Label positions: sorted, at least min_gap apart, as close as possible to the values."""
    order = np.argsort(values)
    pos = np.array(values, float)[order]
    for i in range(1, len(pos)):
        pos[i] = max(pos[i], pos[i - 1] + min_gap)
    shift = (np.array(values, float)[order] - pos).mean()
    pos += shift
    for i in range(1, len(pos)):
        pos[i] = max(pos[i], pos[i - 1] + min_gap)
    out = np.empty_like(pos)
    out[order] = pos
    return out


def panel_geometry(ax, cfg="R280_slot"):
    import matplotlib
    import matplotlib.transforms as mt
    from matplotlib.patches import Polygon, Rectangle

    ph = np.load(HERE / "data" / f"phantom_{BODY}.npz")
    lab, x, y, z = ph["labels"], ph["x"], ph["y"], ph["z"]
    iz = int(np.argmin(np.abs(z)))
    L = C.LABELS
    hu_of = {L["air"]: -1000, L["soft_tissue"]: 40, L["lung"]: -740, L["bone"]: 700, L["myocardium"]: 40,
             L["blood_pool"]: 45, L["liver"]: 55, L["adipose"]: -100}
    px = 1.5
    gx = np.arange(-360, 400 + px, px)
    gy = np.arange(-330, 335 + px, px)
    GX, GY = np.meshgrid(gx, gy)
    v = C.VOXEL_MM
    ix = np.round((GX - x[0]) / v).astype(int)
    iy = np.round((GY - y[0]) / v).astype(int)
    inside = (ix >= 0) & (ix < len(x)) & (iy >= 0) & (iy < len(y))
    labs = np.zeros(GX.shape, dtype=int)
    labs[inside] = lab[iz][iy[inside], ix[inside]]
    hu = np.vectorize(hu_of.get)(labs).astype(float)
    xc, r = scanner.arm_geometry(BODY)                      # simulated arms: soft tissue with humerus
    for s in (1, -1):
        d2 = (GX - s * xc) ** 2 + GY ** 2
        hu[d2 <= r * r] = 40
        hu[d2 <= C.HUMERUS_RADIUS ** 2] = 700
    back = -float(C.BODIES[BODY]["outer"][1])
    img = 1.0 - np.clip((hu + 1000) / 2200, 0, 1)
    ext = [gx[0] - px / 2, gx[-1] + px / 2, gy[0] - px / 2, gy[-1] + px / 2]
    ax.imshow(img, origin="lower", cmap="gray", vmin=0, vmax=1, extent=ext, interpolation="nearest")
    rgba = np.zeros(GX.shape + (4,))
    rgba[labs == L["myocardium"]] = matplotlib.colors.to_rgba("#e34948", 0.9)
    ax.imshow(rgba, origin="lower", extent=ext, interpolation="nearest")
    # curved backrest: 24 mm shell following the posterior body contour, 38 cm chord
    ea, eb = (float(v) for v in C.BODIES[BODY]["outer"])
    phi = np.arcsin(min(BACK_HALF_WIDTH / ea, 1.0))
    th = np.linspace(-np.pi / 2 - phi, -np.pi / 2 + phi, 121)
    ix_, iy_ = ea * np.cos(th), eb * np.sin(th)
    nx, ny = eb * np.cos(th), ea * np.sin(th)
    nn = np.hypot(nx, ny)
    ox, oy = ix_ + BACK_MM * nx / nn, iy_ + BACK_MM * ny / nn
    ax.add_patch(Polygon(np.c_[np.r_[ix_, ox[::-1]], np.r_[iy_, oy[::-1]]], closed=True,
                         facecolor=BACKREST_COL, edgecolor="none", alpha=0.95))

    R = dict((c[0], c[1]) for c in C.SWEEP2_CONFIGS)[cfg]
    kept, n = scanner.module_layout(R, True)
    depth = C.CRYSTAL_SIZE[0]
    width = C.CRYSTALS_PER_MODULE_T * C.CRYSTAL_PITCH
    rc = R + depth / 2
    keep = set(np.round(kept, 4))
    step = 360.0 / n
    offset = float(kept[0] % step)
    for k in range(n):
        a = offset + k * step
        on = round(a % 360.0, 4) in keep
        rect = Rectangle((-depth / 2, -width / 2), depth, width,
                         facecolor=COL[cfg] if on else "none", edgecolor=COL[cfg] if on else "#b8b7b1",
                         lw=0.4 if on else 0.6, ls="-" if on else (0, (2, 1.5)))
        t = np.deg2rad(a)
        rect.set_transform(mt.Affine2D().rotate(t).translate(rc * np.cos(t), rc * np.sin(t)) + ax.transData)
        ax.add_patch(rect)

    chord = scanner.slot_chord(R, kept)
    ys = chord / 2
    xs = np.sqrt(max(R * R - ys * ys, 0))
    xa = R + depth + 12
    ax.annotate("", xy=(xa, ys), xytext=(xa, -ys),
                arrowprops=dict(arrowstyle="<->", color=INK, lw=0.8, shrinkA=0, shrinkB=0))
    for yy in (ys, -ys):
        ax.plot([xs, xa + 4], [yy, yy], color=INK2, lw=0.5)
    ax.text(xa + 8, 0, f"slot\n{chord / 10:.0f} cm\n\narm\n{2 * r / 10:.0f} cm", fontsize=7, color=INK,
            va="center")
    ax.text(60, 40, "LV", fontsize=7, color=INK, ha="left", va="bottom", fontweight="bold")
    ax.text(0, R + depth + 8, "anterior", fontsize=7, color=INK2, ha="center", va="bottom")
    ax.annotate("curved carbon-\nfibre backrest", xy=(-150, -eb * np.sqrt(max(1 - (150 / ea) ** 2, 0)) - BACK_MM / 2), xytext=(-250, -262),
                fontsize=6.5, color=INK, ha="center", va="top",
                arrowprops=dict(arrowstyle="-", color=INK2, lw=0.5))
    ax.text(-360, 335, "dashed: detectors\nremoved for slots", fontsize=6.5, color=INK2, va="top")
    ax.set_xlim(-360, 430)
    ax.set_ylim(-330, 335)
    ax.set_aspect("equal")
    ax.axis("off")


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep2", required=True)
    ap.add_argument("--out", default=".")
    args = ap.parse_args()
    s2 = Path(args.sweep2)
    rows = {(r["config"], r["body"]): r for r in json.load(open(s2 / "summary2.json"))}
    ang = list(csv.DictReader(open(s2 / "angular2.csv")))
    cfgs = [c for c in CFGS if (c, BODY) in rows]

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8, "text.color": INK,
                         "axes.labelcolor": INK, "axes.titlesize": 9})
    fig = plt.figure(figsize=(7.2, 6.8))
    gs = fig.add_gridspec(2, 2, wspace=0.30, hspace=0.38, left=0.10, right=0.98, top=0.95, bottom=0.09)

    axA = fig.add_subplot(gs[0, 0])
    panel_geometry(axA)
    axA.set_title(f"A.  56 cm, 290° split ring ({BODY} body)", loc="left", fontsize=9, fontweight="bold")

    # B: myocardial NEC per decay relative to 70 cm closed ring, both bodies; fraction of crystals
    axB = fig.add_subplot(gs[0, 1])
    style(axB)
    x = np.arange(len(cfgs))
    w = 0.36
    for j, (b, hatch) in enumerate((("P50", ""), ("P90", "////"))):
        vals = [rows[(c, b)]["rel_nec_myo"] for c in cfgs]
        axB.bar(x + (j - 0.5) * w, vals, w, color=[COL[c] for c in cfgs], edgecolor="white", lw=1.2, hatch=hatch)
        for xi, v in zip(x + (j - 0.5) * w, vals):
            axB.text(xi, v + 0.015, f"{v:.2f}", ha="center", va="bottom", fontsize=6.5, color=INK)
    for xi, c in zip(x, cfgs):
        f = rows[(c, BODY)]["rel_crystals"]
        axB.plot([xi - 0.42, xi + 0.42], [f, f], color=INK, lw=1.3)
    axB.set_xticks(x, [SHORT[c] for c in cfgs], fontsize=7.5)
    axB.set_ylim(0, 1.3)
    axB.set_ylabel("Myocardial NEC per decay,\nrelative to 70 cm closed ring")
    axB.legend(handles=[Patch(facecolor="#9a9994", edgecolor="white", label="P50 body"),
                        Patch(facecolor="#9a9994", edgecolor="white", hatch="////", label="P90 body"),
                        Line2D([], [], color=INK, lw=1.3, label="fraction of crystals")],
               fontsize=6.5, frameon=False, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.0),
               handlelength=1.6, columnspacing=1.0)
    axB.set_title("B.  Equal counts, 64% of the crystals", loc="left", fontsize=9, fontweight="bold")

    # C: TOF-effective NEC vs CTR, P90, relative to 70 cm closed ring at the worst CTR
    axC = fig.add_subplot(gs[1, 0])
    style(axC)
    ctrs = C.CTR_SWEEP_PS
    cmax = max(ctrs)
    series = {c: [rows[(c, BODY)][f"rel_tof_eff_nec_{t}ps_vs_closed70_{cmax}ps"] for t in ctrs] for c in cfgs}
    for c in cfgs:
        axC.plot(ctrs, series[c], "-", marker=MARK[c], ms=5, lw=2.4 if c == "R280_slot" else 1.5, color=COL[c],
                 mec="white", mew=0.8, zorder=3 if c == "R280_slot" else 2)
    first = [series[c][0] for c in cfgs]
    ypos = dodge(first, 0.045)
    for c, v, yp in zip(cfgs, first, ypos):
        axC.plot([ctrs[0] - 5, ctrs[0] - 2], [yp, v], color=INK2, lw=0.5)
        axC.text(ctrs[0] - 7, yp, f"{NAME[c].replace(' (proposed)', '')}  {v:.2f}", ha="right", va="center",
                 fontsize=7, color=COL[c] if c != "R350_slot" else "#b54c16", fontweight="bold" if c == "R280_slot" else None)
    axC.plot([ctrs[0], cmax + 8], [1, 1], color=INK2, lw=0.8, ls=(0, (3, 2)), zorder=1)
    axC.set_xlim(ctrs[0] - 105, cmax + 8)
    axC.set_xticks(ctrs)
    axC.set_xlabel("Coincidence timing resolution, FWHM (ps)")
    axC.set_ylabel(f"TOF-effective myocardial NEC, relative to\n70 cm closed ring at {cmax} ps (dashed)")
    axC.set_title(f"C.  Timing resolution ({BODY} body)", loc="left", fontsize=9, fontweight="bold")

    # D: angular distribution of myocardial trues, P90, relative to 70 cm closed ring
    axD = fig.add_subplot(gs[1, 1])
    style(axD)
    lo = np.array([float(r["phi_lo"]) for r in ang])
    centres = lo + 5
    ref = "R350_closed"
    n_ref = rows[(ref, BODY)]["decays"] * rows[(ref, BODY)]["frac_myo"]
    h0 = np.array([float(r[f"{ref}_{BODY}"]) for r in ang])
    for c in cfgs:
        h = np.array([float(r[f"{c}_{BODY}"]) for r in ang])
        n = rows[(c, BODY)]["decays"] * rows[(c, BODY)]["frac_myo"]
        ratio = (h / n) / (h0 / n_ref)
        err = ratio * np.sqrt(1 / np.maximum(h, 1) + 1 / np.maximum(h0, 1))
        err[h == 0] = 0
        axD.errorbar(centres, ratio, yerr=None if c == ref else err, fmt="-", marker=MARK[c], ms=3.5,
                     lw=2.4 if c == "R280_slot" else 1.3, color=COL[c], mec="white", mew=0.5, capsize=0,
                     elinewidth=0.7, zorder=3 if c == "R280_slot" else 2)
    handles = [Line2D([], [], color=COL[c], marker=MARK[c], lw=2.4 if c == "R280_slot" else 1.3, ms=4,
                      mec="white", mew=0.5, label=NAME[c]) for c in cfgs]
    axD.legend(handles=handles, loc="lower center", fontsize=6.5, frameon=False, handlelength=2.2)
    axD.set_xlim(0, 180)
    axD.set_ylim(0, 1.4)
    axD.set_xticks([0, 30, 60, 90, 120, 150, 180])
    axD.set_xlabel("Transaxial LOR angle (deg)\n0 = left–right, 90 = anterior–posterior")
    axD.set_ylabel("Myocardial trues per decay,\nrelative to 70 cm closed ring")
    axD.set_title(f"D.  Angular sampling ({BODY} body)", loc="left", fontsize=9, fontweight="bold")

    out = Path(args.out)
    fig.savefig(out / "Figure1_290.png", dpi=300)
    fig.savefig(out / "Figure1_290.pdf")
    print("Wrote", out / "Figure1_290.pdf")


if __name__ == "__main__":
    main()

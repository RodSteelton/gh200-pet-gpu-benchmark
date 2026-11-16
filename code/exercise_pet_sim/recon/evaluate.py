"""
LV image-quality metrics and figures for the reconstructions.

    python evaluate.py --recon /scratch/$USER/exercise_pet_recon

Images are scaled so total activity inside the body (within the axial FOV) matches the truth.
Myocardium (true value 1.0) is split into 16 segments (6 basal, 6 mid, 4 apical; AHA-like).
Circumferential angle: 0 = anterior, 90 = lateral, 180 = inferior, 270 = septal.
Writes recon_metrics.csv / .json, Fig_recon_images_<data>_<body>.png/.pdf, Fig_recon_profiles_<data>.png/.pdf,
Fig_recon_metrics.png/.pdf.
"""

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter, map_coordinates

import geometry as G

C = G.C
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
COL = {"R280_slot": "#2a78d6", "R310_slot": "#1baf7a", "R350_slot": "#eb6834", "R350_closed": "#4a3aa7"}
NAME = {"R280_slot": "56 cm, 290°", "R310_slot": "62 cm, 297°", "R350_slot": "70 cm, 320°",
        "R350_closed": "70 cm, closed"}
LS = {0: (0, (1, 1.5)), 320: (0, (4, 2)), 250: (0, (6, 2, 1, 2)), 200: "-"}
CTRNAME = {0: "non-TOF", 320: "320 ps", 250: "250 ps", 200: "200 ps"}
LEVELS = [("basal", C.LV_BASE_CUT, 0.0, 6), ("mid", 0.0, 20.0, 6), ("apical", 20.0, 60.0, 4)]
DISPLAY_FWHM_MM = 6.0


def style(ax, grid_axis="y"):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(INK2)
        ax.spines[s].set_linewidth(0.8)
    ax.tick_params(colors=INK2, labelsize=8, width=0.8)
    if grid_axis:
        ax.grid(axis=grid_axis, color=GRID, lw=0.6)
    ax.set_axisbelow(True)


class Truth:
    cache = {}

    @classmethod
    def get(cls, config, body):
        key = (config, body)
        if key not in cls.cache:
            ring = G.Ring(config)
            grid = G.recon_grid(ring)
            ph = G.load_body(body)
            act, lab = G.truth(grid, ph)
            w, r, psi = G.lv_coords(grid)
            cls.cache[key] = dict(grid=grid, act=act, lab=lab, w=w, psi=psi)
        return cls.cache[key]


def normalise(img, t):
    body = (t["lab"] > 0) & (t["lab"] < 254)
    return img * t["act"][body].sum() / max(img[body].sum(), 1e-12)


def segment_values(img, t):
    myo = t["lab"] == C.LABELS["myocardium"]
    vals = []
    for name, w0, w1, nsec in LEVELS:
        lev = myo & (t["w"] >= w0) & (t["w"] < w1)
        for k in range(nsec):
            a0, a1 = 360.0 / nsec * k - 180.0 / nsec, 360.0 / nsec * (k + 1) - 180.0 / nsec
            ps = (t["psi"] - a0) % 360.0
            m = lev & (ps < (a1 - a0))
            vals.append(float(img[m].mean()) if m.any() else np.nan)
    return np.array(vals)


def mid_profile(img, t, nbins=12):
    myo = (t["lab"] == C.LABELS["myocardium"]) & (t["w"] >= 0) & (t["w"] < 20)
    edges = np.linspace(0, 360, nbins + 1)
    out = []
    for a0, a1 in zip(edges[:-1], edges[1:]):
        m = myo & (t["psi"] >= a0) & (t["psi"] < a1)
        out.append(float(img[m].mean()) if m.any() else np.nan)
    return 0.5 * (edges[:-1] + edges[1:]), np.array(out)


def reslice(img, grid, plane, half=60.0, step=1.0):
    """Short-axis (mid) or vertical long-axis slice through the LV, in mm."""
    c, u, e1, e2 = G.lv_frame()
    s = np.arange(-half, half + step, step)
    A, B = np.meshgrid(s, s)
    if plane == "sa":
        P = c + 10.0 * u + A[..., None] * e2 + B[..., None] * e1       # lateral right, anterior up
    else:
        P = c + A[..., None] * u + B[..., None] * e1                    # apex right, anterior up
    ix = (P[..., 0] - grid.x0) / grid.v
    iy = (P[..., 1] - grid.y0) / grid.v
    iz = (P[..., 2] - grid.z0) / grid.v
    return map_coordinates(img, [iz, iy, ix], order=1, mode="constant")


def collect(recon_dir):
    rows, imgs = [], {}
    for js in sorted(recon_dir.glob("img_*.json")):
        meta = json.loads(js.read_text())
        img = np.load(js.with_suffix(".npy"))
        t = Truth.get(meta["config"], meta["body"])
        if tuple(img.shape) != t["grid"].shape:
            continue
        img = normalise(img, t)
        seg = segment_values(img, t)
        psi_c, prof = mid_profile(img, t)
        myo = t["lab"] == C.LABELS["myocardium"]
        blood = t["lab"] == C.LABELS["blood_pool"]
        row = dict(meta)
        row.update({
            "myo_recovery": float(np.nanmean(seg)),
            "segment_cv_pct": float(100 * np.nanstd(seg) / np.nanmean(seg)),
            "segment_min_rel": float(np.nanmin(seg) / np.nanmean(seg)),
            "segment_max_rel": float(np.nanmax(seg) / np.nanmean(seg)),
            "ant_inf_over_sep_lat": float(np.nanmean(prof[[0, 5, 6, 11]]) / np.nanmean(prof[[2, 3, 8, 9]])),
            "blood_pool": float(img[blood].mean()),
            "myo_to_blood": float(img[myo].mean() / max(img[blood].mean(), 1e-9)),
        })
        rows.append(row)
        imgs[(meta["data"], meta["body"], meta["config"], meta["ctr_ps"])] = (img, t, psi_c, prof, seg)
    return rows, imgs


def fig_images(imgs, data, body, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    keys = sorted([k for k in imgs if k[0] == data and k[1] == body],
                  key=lambda k: ([c[0] for c in C.SWEEP2_CONFIGS].index(k[2]), -k[3] if k[3] else -9999))
    if not keys:
        return
    t0 = imgs[keys[0]][1]
    cols = [("Truth", t0["act"], t0["grid"])] + [
        (f"{NAME[k[2]]}\n{CTRNAME.get(k[3], k[3])}", imgs[k][0], imgs[k][1]["grid"]) for k in keys]
    fig, axs = plt.subplots(2, len(cols), figsize=(1.35 * len(cols) + 0.6, 3.3))
    sig = DISPLAY_FWHM_MM / 2.3548
    for j, (title, img, grid) in enumerate(cols):
        disp = img if j == 0 else gaussian_filter(img, sig / grid.v)
        for i, plane in enumerate(("sa", "vla")):
            ax = axs[i, j]
            ax.imshow(reslice(disp, grid, plane), origin="lower", cmap="gray_r", vmin=0, vmax=1.1,
                      extent=[-60, 60, -60, 60], interpolation="bilinear")
            ax.set_xticks([])
            ax.set_yticks([])
            for s in ax.spines.values():
                s.set_visible(False)
            if i == 0:
                ax.set_title(title, fontsize=7)
    axs[0, 0].set_ylabel("A.  Short axis", fontsize=8, fontweight="bold")
    axs[1, 0].set_ylabel("B.  Vertical long axis", fontsize=8, fontweight="bold")
    fig.suptitle(f"LV reconstructions, {body} body, {'GATE' if data == 'gate' else 'fast Monte Carlo'} data "
                 f"(display filter {DISPLAY_FWHM_MM:.0f} mm)", fontsize=8.5)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(out / f"Fig_recon_images_{data}_{body}.{ext}", dpi=300)
    plt.close(fig)


def fig_profiles(imgs, data, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    bodies = sorted({k[1] for k in imgs if k[0] == data})
    if not bodies:
        return
    fig, axs = plt.subplots(1, len(bodies), figsize=(3.6 * len(bodies), 3.2), squeeze=False)
    for i, b in enumerate(bodies):
        ax = axs[0, i]
        style(ax)
        for k, (img, t, psi_c, prof, seg) in imgs.items():
            if k[0] != data or k[1] != b:
                continue
            ax.plot(psi_c, prof, color=COL[k[2]], ls=LS.get(k[3], "-"), lw=1.8 if k[3] == 200 else 1.3)
        ax.axhline(1.0, color=INK2, lw=0.6)
        ax.set_xticks([0, 90, 180, 270, 360], ["ant", "lat", "inf", "sep", "ant"])
        ax.set_xlim(0, 360)
        ax.set_ylim(0.5, 1.15)
        ax.set_xlabel("Mid-ventricular circumferential position")
        ax.set_ylabel("Myocardial activity (true = 1)")
        ax.set_title(f"{'AB'[i]}.  {b} body", loc="left", fontweight="bold", fontsize=9)
    handles = [Line2D([], [], color=COL[c], lw=1.8, label=NAME[c]) for c in C.RECON_CONFIGS] + \
              [Line2D([], [], color=INK2, ls=LS[ct], lw=1.3, label=CTRNAME[ct]) for ct in C.RECON_CTRS]
    axs[0, -1].legend(handles=handles, fontsize=6.5, frameon=False, loc="lower right", ncol=2)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(out / f"Fig_recon_profiles_{data}.{ext}", dpi=300)
    plt.close(fig)


def fig_metrics(rows, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    combos = sorted({(r["data"], r["body"]) for r in rows})
    if not combos:
        return
    ctrs = [c for c in C.RECON_CTRS if any(r["ctr_ps"] == c for r in rows)]
    cfgs = [c for c in C.RECON_CONFIGS if any(r["config"] == c for r in rows)]
    fig, axs = plt.subplots(3, len(combos), figsize=(3.3 * len(combos), 7.4), squeeze=False)
    w = 0.8 / max(len(cfgs), 1)
    get = {(r["data"], r["body"], r["config"], r["ctr_ps"]): r for r in rows}
    letter = iter("ABCDEFGHIJ")
    for row_i, (key, ylab) in enumerate([("myo_recovery", "Myocardial recovery (true = 1)"),
                                         ("segment_cv_pct", "Segment non-uniformity, CV (%)"),
                                         ("ant_inf_over_sep_lat", "Ant+inf / sep+lat wall")]):
        for j, (data, body) in enumerate(combos):
            ax = axs[row_i, j]
            style(ax)
            x = np.arange(len(ctrs))
            for i, c in enumerate(cfgs):
                vals = [get.get((data, body, c, ct), {}).get(key, np.nan) for ct in ctrs]
                ax.bar(x + (i - (len(cfgs) - 1) / 2) * w, vals, width=w, color=COL[c], edgecolor="white", lw=1.2)
                for xi, v in zip(x, vals):
                    if np.isfinite(v):
                        ax.text(xi + (i - (len(cfgs) - 1) / 2) * w, v, f"{v:.1f}" if key == "segment_cv_pct" else f"{v:.2f}",
                                ha="center", va="bottom", fontsize=6)
            ax.set_xticks(x, [CTRNAME[c] for c in ctrs], fontsize=7.5)
            ax.set_ylabel(ylab, fontsize=8)
            if key != "segment_cv_pct":
                ax.set_ylim(0, 1.3)
                ax.axhline(1.0, color="0.3", lw=0.8)
            ax.set_title(f"{next(letter)}.  {body}, {'GATE' if data == 'gate' else 'fast MC'}", loc="left",
                         fontweight="bold", fontsize=9)
    axs[0, 0].legend(handles=[Patch(color=COL[c], label=NAME[c]) for c in cfgs], fontsize=7, frameon=False,
                     loc="upper left", ncol=2)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(out / f"Fig_recon_metrics.{ext}", dpi=300)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--recon", required=True)
    args = ap.parse_args()
    rdir = Path(args.recon)
    out = rdir / "evaluation"
    out.mkdir(exist_ok=True)
    rows, imgs = collect(rdir)
    if not rows:
        raise SystemExit(f"No reconstructions found in {rdir}")
    keys = list(dict.fromkeys(k for r in rows for k in r if not isinstance(r[k], (list, dict))))
    with open(out / "recon_metrics.csv", "w") as fh:
        fh.write(",".join(keys) + "\n")
        for r in rows:
            fh.write(",".join(str(r.get(k, "")) for k in keys) + "\n")
    (out / "recon_metrics.json").write_text(json.dumps(rows, indent=2))
    for data in sorted({r["data"] for r in rows}):
        for body in sorted({r["body"] for r in rows if r["data"] == data}):
            fig_images(imgs, data, body, out)
        fig_profiles(imgs, data, out)
    fig_metrics(rows, out)

    print(f"{'data':6s} {'body':4s} {'config':12s} {'CTR':>7s} {'events':>11s} {'recovery':>8s} "
          f"{'segCV%':>6s} {'min':>5s} {'max':>5s} {'ant+inf/sep+lat':>15s} {'myo/blood':>9s}")
    for r in sorted(rows, key=lambda r: (r["data"], r["body"], r["config"], -r["ctr_ps"] if r["ctr_ps"] else -9999)):
        print(f"{r['data']:6s} {r['body']:4s} {r['config']:12s} {CTRNAME.get(r['ctr_ps'], r['ctr_ps']):>7s} "
              f"{r['n_events']:>11,} {r['myo_recovery']:8.3f} {r['segment_cv_pct']:6.1f} "
              f"{r['segment_min_rel']:5.2f} {r['segment_max_rel']:5.2f} {r['ant_inf_over_sep_lat']:15.3f} "
              f"{r['myo_to_blood']:9.2f}")
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()

"""
Build the voxelized cardiac torso phantoms and geometry figures.

Sweep 1 phantom (unchanged):  data/phantom_labels.mhd, phantom_activity.mhd, phantom.npz
Sweep 2 body phantoms:        data/phantom_<BODY>_labels.mhd, _activity.mhd, phantom_<BODY>.npz
                              (BODY = P50, P90; adults 60+; the arms are separate cylinders in the
                              simulation and are stored here only as geometry and activity)
Figures:                      data/setup_geometry.png (sweep 1), data/setup_sweep2.png

Run once:  python make_phantom.py
"""

from pathlib import Path

import numpy as np

import config as C
import scanner

DATA = Path(__file__).resolve().parent / "data"

HU = {"air": -1000, "soft_tissue": 40, "lung": -740, "bone": 700, "myocardium": 40,
      "blood_pool": 45, "liver": 55, "adipose": -100}


def axes_for(dims):
    nx, ny, nz = dims
    v = C.VOXEL_MM
    x = (np.arange(nx) - (nx - 1) / 2.0) * v
    y = (np.arange(ny) - (ny - 1) / 2.0) * v
    z = (np.arange(nz) - (nz - 1) / 2.0) * v
    return x, y, z


def voxel_centres():
    return axes_for(C.PHANTOM_DIMS)


def ellipsoid(X, Y, Z, c, s):
    return ((X - c[0]) / s[0]) ** 2 + ((Y - c[1]) / s[1]) ** 2 + ((Z - c[2]) / s[2]) ** 2 <= 1.0


def build(dims=None, outer=None):
    """Label and activity arrays [z, y, x]. The lean core is always the sweep-1 torso;
    if `outer` (semi-axes, mm) is larger, the space between is adipose tissue."""
    L = C.LABELS
    dims = dims or C.PHANTOM_DIMS
    x, y, z = axes_for(dims)
    Z, Y, X = np.meshgrid(z, y, x, indexing="ij")
    lab = np.zeros(Z.shape, dtype=np.float32)
    zin = np.abs(Z) <= C.TORSO_HALF_LENGTH

    if outer is not None:
        body = ((X / outer[0]) ** 2 + (Y / outer[1]) ** 2 <= 1.0) & zin
        lab[body] = L["adipose"]

    a, b = C.TORSO_SEMI_AXES
    torso = ((X / a) ** 2 + (Y / b) ** 2 <= 1.0) & zin
    lab[torso] = L["soft_tissue"]

    spine = (X ** 2 + (Y + 85.0) ** 2 <= 20.0 ** 2) & torso
    lab[spine] = L["bone"]

    lungs = (ellipsoid(X, Y, Z, (85.0, 5.0, 30.0), (55.0, 70.0, 110.0))
             | ellipsoid(X, Y, Z, (-85.0, 5.0, 30.0), (60.0, 75.0, 110.0))) & torso
    lab[lungs] = L["lung"]

    liver = ellipsoid(X, Y, Z, (-60.0, 0.0, -110.0), (95.0, 75.0, 65.0)) & torso
    lab[liver] = L["liver"]

    # left ventricle: prolate ellipsoidal shell, long axis base -> apex
    c = np.array(C.LV_CENTER)
    u = np.array(C.LV_AXIS)
    u /= np.linalg.norm(u)
    dx, dy, dz = X - c[0], Y - c[1], Z - c[2]
    w = dx * u[0] + dy * u[1] + dz * u[2]
    r = np.sqrt(np.maximum(dx ** 2 + dy ** 2 + dz ** 2 - w ** 2, 0.0))
    outer_lv = (r / C.LV_OUTER[0]) ** 2 + (w / C.LV_OUTER[1]) ** 2 <= 1.0
    inner_lv = (r / C.LV_INNER[0]) ** 2 + (w / C.LV_INNER[1]) ** 2 <= 1.0
    myo = outer_lv & ~inner_lv & (w > C.LV_BASE_CUT)
    blood = outer_lv & ~myo
    lab[blood] = L["blood_pool"]
    lab[myo] = L["myocardium"]

    act = np.zeros_like(lab)
    for name, k in L.items():
        act[lab == k] = C.ACTIVITY[name]
    return lab, act, (x, y, z)


def body_dims(outer):
    v = C.VOXEL_MM
    nx = 2 * int(np.ceil((outer[0] + 1.0) / v))
    ny = 2 * int(np.ceil((outer[1] + 1.0) / v))
    nz = C.PHANTOM_DIMS[2]
    return nx, ny, nz


def write_mhd(arr, path):
    import SimpleITK as sitk

    img = sitk.GetImageFromArray(arr.astype(np.float32))
    v = C.VOXEL_MM
    img.SetSpacing([v, v, v])
    nz, ny, nx = arr.shape
    img.SetOrigin([-(nx - 1) / 2.0 * v, -(ny - 1) / 2.0 * v, -(nz - 1) / 2.0 * v])
    sitk.WriteImage(img, str(path))


def body_phantom(body):
    spec = C.BODIES[body]
    dims = body_dims(spec["outer"])
    lab, act, axes = build(dims, spec["outer"])
    v3 = C.VOXEL_MM ** 3
    xc, r = scanner.arm_geometry(body)
    arm_vol = np.pi * (r ** 2 - C.HUMERUS_RADIUS ** 2) * 2 * C.ARM_HALF_LENGTH
    a_torso = float(act.sum() * v3)
    a_arm = float(C.ACTIVITY["arm"] * arm_vol)
    a_myo = float(act[lab == C.LABELS["myocardium"]].sum() * v3)
    info = dict(arm_x=xc, arm_r=r, a_torso=a_torso, a_arm_each=a_arm,
                frac_myo=a_myo / (a_torso + 2 * a_arm),
                box_half=(axes[0][-1] + C.VOXEL_MM / 2, axes[1][-1] + C.VOXEL_MM / 2))
    return lab, act, axes, info


# ---------------------------------------------------------------- figures

def hu_slice(lab_slice, xg, yg, grid_x, grid_y, arms=None):
    """Resample a label slice onto a display grid and return Hounsfield units (arms optional)."""
    L = C.LABELS
    inv = {k: HU[n] for n, k in L.items()}
    GX, GY = np.meshgrid(grid_x, grid_y)
    v = C.VOXEL_MM
    ix = np.round((GX - xg[0]) / v).astype(int)
    iy = np.round((GY - yg[0]) / v).astype(int)
    inside = (ix >= 0) & (ix < len(xg)) & (iy >= 0) & (iy < len(yg))
    labs = np.zeros(GX.shape, dtype=int)
    labs[inside] = lab_slice[iy[inside], ix[inside]]
    hu = np.vectorize(inv.get)(labs).astype(float)
    if arms is not None:
        xc, r = arms
        for s in (1, -1):
            hu[(GX - s * xc) ** 2 + GY ** 2 <= r ** 2] = 40
            hu[(GX - s * xc) ** 2 + GY ** 2 <= C.HUMERUS_RADIUS ** 2] = 1000
    return hu, labs


def draw_ring(ax, R, kept, color="#2a78d6", n_full=None, show_removed=True):
    from matplotlib.patches import Rectangle
    import matplotlib.transforms as mt

    depth = C.CRYSTAL_SIZE[0]
    width = C.CRYSTALS_PER_MODULE_T * C.CRYSTAL_PITCH
    rc = R + depth / 2
    kept_set = set(np.round(kept, 4))
    n_full = n_full or scanner.n_modules_full(R)
    step = 360.0 / n_full
    offsets = (np.round(kept[0] % step, 4),) if len(kept) else (0.0,)
    all_angles = (offsets[0] + step * np.arange(n_full)) % 360.0
    for a in all_angles:
        on = round(a, 4) in kept_set
        if not on and not show_removed:
            continue
        t = np.deg2rad(a)
        rect = Rectangle((-depth / 2, -width / 2), depth, width,
                         facecolor=color if on else "none", edgecolor=color if on else "#b8b7b1",
                         lw=0.3 if on else 0.5, ls="-" if on else (0, (2, 1.5)))
        rect.set_transform(mt.Affine2D().rotate(t).translate(rc * np.cos(t), rc * np.sin(t)) + ax.transData)
        ax.add_patch(rect)


def plot_setup(lab, axes, path):
    """Sweep-1 figure: phantom slice with the four gap configurations."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    x, y, z = axes
    iz = int(np.argmin(np.abs(z)))
    g = np.arange(-330, 331, 1.5)
    hu, labs = hu_slice(lab[iz], x, y, g, g)
    titles = {0: "Closed ring (360°)", 20: "20° gap per side (320°)",
              30: "30° gap per side (300°)", 60: "60° gap per side (240°, 2017)"}
    fig, axs = plt.subplots(2, 2, figsize=(9, 9))
    for i, (ax, gap) in enumerate(zip(axs.flat, C.GAP_SWEEP)):
        show_hu(ax, hu, labs, g, g)
        kept = scanner.module_angles(gap)
        draw_ring(ax, C.RING_INNER_RADIUS, kept, n_full=C.N_MODULES_FULL)
        n = len(kept)
        ax.set_title(f"{'ABCD'[i]}.  {titles.get(gap, gap)}\n{n} modules, "
                     f"{n * C.CRYSTALS_PER_MODULE_T * C.CRYSTALS_AXIAL:,} crystals", loc="left", fontsize=10)
        finish(ax, 330)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def show_hu(ax, hu, labs, gx, gy):
    """Darkness proportional to Hounsfield units; myocardial uptake in red."""
    import matplotlib

    img = 1.0 - np.clip((hu + 1000) / 2200, 0, 1)
    px = gx[1] - gx[0]
    ext = [gx[0] - px / 2, gx[-1] + px / 2, gy[0] - px / 2, gy[-1] + px / 2]
    ax.imshow(img, origin="lower", cmap="gray", vmin=0, vmax=1, extent=ext, interpolation="nearest")
    rgba = np.zeros(labs.shape + (4,))
    rgba[labs == C.LABELS["myocardium"]] = matplotlib.colors.to_rgba("#e34948", 0.9)
    ax.imshow(rgba, origin="lower", extent=ext, interpolation="nearest")


def finish(ax, lim):
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)


def plot_sweep2(path, body="P90"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    lab, act, axes, info = body_phantom(body)
    x, y, z = axes
    iz = int(np.argmin(np.abs(z)))
    g = np.arange(-385, 386, 1.5)
    hu, labs = hu_slice(lab[iz], x, y, g, g, arms=(info["arm_x"], info["arm_r"]))
    names = {"R280_slot": "56 cm bore, slots", "R310_slot": "62 cm bore, slots",
             "R350_slot": "70 cm bore, slots", "R350_closed": "70 cm bore, closed (arms inside)"}
    fig, axs = plt.subplots(2, 2, figsize=(9, 9.4))
    for i, (ax, (name, R, slot)) in enumerate(zip(axs.flat, C.SWEEP2_CONFIGS)):
        show_hu(ax, hu, labs, g, g)
        kept, n_full = scanner.module_layout(R, slot)
        draw_ring(ax, R, kept, n_full=n_full)
        cov = 360.0 * len(kept) / n_full
        chord = scanner.slot_chord(R, kept)
        sub = f"{len(kept)} modules, {cov:.0f}° coverage" + (f", slot {chord / 10:.1f} cm" if chord else "")
        ax.set_title(f"{'ABCD'[i]}.  {names.get(name, name)}\n{sub}", loc="left", fontsize=10)
        finish(ax, 385)
    fig.suptitle(f"Sweep 2 geometries with the {body} (adults 60+) body and arms, mid-FOV", fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    DATA.mkdir(exist_ok=True)

    # sweep 1 phantom (unchanged)
    lab, act, axes = build()
    write_mhd(lab, DATA / "phantom_labels.mhd")
    write_mhd(act, DATA / "phantom_activity.mhd")
    np.savez_compressed(DATA / "phantom.npz", labels=lab.astype(np.uint8), activity=act,
                        x=axes[0], y=axes[1], z=axes[2])
    plot_setup(lab, axes, DATA / "setup_geometry.png")
    tot = act.sum()
    print("Sweep 1 phantom written to", DATA)
    for name, k in C.LABELS.items():
        nvox = int((lab == k).sum())
        if nvox:
            print(f"  {name:12s} label {k}: {nvox:8d} voxels, {100 * act[lab == k].sum() / tot:5.1f}% of decays")

    # sweep 2 body phantoms
    for body in C.BODIES:
        lab, act, axes, info = body_phantom(body)
        write_mhd(lab, DATA / f"phantom_{body}_labels.mhd")
        write_mhd(act, DATA / f"phantom_{body}_activity.mhd")
        np.savez_compressed(DATA / f"phantom_{body}.npz", labels=lab.astype(np.uint8), activity=act,
                            x=axes[0], y=axes[1], z=axes[2], **{k: np.array(v) for k, v in info.items()})
        print(f"{body}: image {lab.shape[::-1]} voxels, arms at x=±{info['arm_x']:.0f} mm "
              f"r={info['arm_r']:.0f} mm, myocardium {100 * info['frac_myo']:.1f}% of decays")
    plot_sweep2(DATA / "setup_sweep2.png", "P90")
    print("Figures: data/setup_geometry.png, data/setup_sweep2.png")


if __name__ == "__main__":
    main()

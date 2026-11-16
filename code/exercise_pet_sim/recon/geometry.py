"""
Scanner and phantom geometry for reconstruction: crystal snapping, a simple physical detection
model for the fast Monte Carlo, attenuation and truth images on the recon grid, and LV coordinates.
"""

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import config as C  # noqa: E402
import scanner  # noqa: E402
from projector import Grid  # noqa: E402

DATA = ROOT / "data"
DEPTH = C.CRYSTAL_SIZE[0]
PITCH = C.CRYSTAL_PITCH
NT = C.CRYSTALS_PER_MODULE_T
NA = C.CRYSTALS_AXIAL
ZMAX = NA * PITCH / 2.0


class Ring:
    """Module layout for one geometry, as simulated in GATE sweep 2."""

    def __init__(self, config_name):
        cfg = {c[0]: c for c in C.SWEEP2_CONFIGS}[config_name]
        self.name, self.R, self.slot = cfg
        self.kept, self.n_full = scanner.module_layout(self.R, self.slot)
        self.step = 360.0 / self.n_full
        self.offset = float(np.round(self.kept[0] % self.step, 6))
        self.kept_mask = np.zeros(self.n_full, bool)
        idx = np.round((self.kept - self.offset) / self.step).astype(int) % self.n_full
        self.kept_mask[idx] = True

    def module_of(self, x, y):
        """Nearest module index and its centre angle (rad) for points (x, y)."""
        ang = np.degrees(np.arctan2(y, x)) % 360.0
        idx = np.round((ang - self.offset) / self.step).astype(int) % self.n_full
        th = np.deg2rad(self.offset + idx * self.step)
        return idx, th

    def snap(self, q):
        """Snap interaction points (n,3) to crystal centres (no depth of interaction)."""
        idx, th = self.module_of(q[:, 0], q[:, 1])
        c, s = np.cos(th), np.sin(th)
        t = -q[:, 0] * s + q[:, 1] * c
        it = np.clip(np.floor(t / PITCH + NT / 2.0), 0, NT - 1)
        iz = np.clip(np.floor(q[:, 2] / PITCH + NA / 2.0), 0, NA - 1)
        tc = (it - (NT - 1) / 2.0) * PITCH
        zc = (iz - (NA - 1) / 2.0) * PITCH
        rc = self.R + DEPTH / 2.0
        out = np.empty_like(q, dtype=np.float32)
        out[:, 0] = rc * c - tc * s
        out[:, 1] = rc * s + tc * c
        out[:, 2] = zc
        return out, self.kept_mask[idx]

    def detect(self, e, d, rng):
        """Photons from e (n,3) along unit d (n,3): front-face entry, exponential depth in LYSO,
        must stay inside a kept module. Returns interaction points and detected mask."""
        exy, dxy = e[:, :2], d[:, :2]
        a = (dxy ** 2).sum(1)
        b = 2 * (exy * dxy).sum(1)
        c = (exy ** 2).sum(1) - self.R ** 2
        disc = b * b - 4 * a * c
        ok = (a > 1e-12) & (c < 0) & (disc > 0)
        s0 = np.where(ok, (-b + np.sqrt(np.maximum(disc, 0))) / (2 * np.maximum(a, 1e-12)), 0.0)
        ell = rng.exponential(1.0 / C.MU_LYSO_PER_MM, size=len(e))
        q = e + (s0 + ell)[:, None] * d
        idx, th = self.module_of(q[:, 0], q[:, 1])
        cth, sth = np.cos(th), np.sin(th)
        radial = q[:, 0] * cth + q[:, 1] * sth
        tang = -q[:, 0] * sth + q[:, 1] * cth
        ok &= self.kept_mask[idx]
        ok &= (radial >= self.R) & (radial <= self.R + DEPTH)
        ok &= np.abs(tang) <= NT * PITCH / 2.0
        ok &= np.abs(q[:, 2]) <= ZMAX
        return q, ok


def recon_grid(ring, v=None):
    v = v or C.RECON_VOXEL_MM
    n_xy = 2 * int(np.ceil((ring.R - 2.0) / v))
    n_z = 2 * int(np.ceil(ZMAX / v))
    return Grid(n_xy, n_xy, n_z, v)


def load_body(body):
    return dict(np.load(DATA / f"phantom_{body}.npz"))


def on_grid(ph, arr, grid, fill=0):
    """Place a phantom-grid array onto the recon grid (both 3 mm, centred, same voxel phase)."""
    out = np.full(grid.shape, fill, dtype=arr.dtype)
    gx, gy, gz = grid.axes()
    px, py, pz = ph["x"], ph["y"], ph["z"]
    v = grid.v

    def overlap(g, p):
        i0 = int(np.round((max(g[0], p[0]) - g[0]) / v))
        j0 = int(np.round((max(g[0], p[0]) - p[0]) / v))
        n = int(np.round((min(g[-1], p[-1]) - max(g[0], p[0])) / v)) + 1
        return slice(i0, i0 + n), slice(j0, j0 + n)

    (gxs, pxs), (gys, pys), (gzs, pzs) = overlap(gx, px), overlap(gy, py), overlap(gz, pz)
    out[gzs, gys, gxs] = arr[pzs, pys, pxs]
    return out


def arm_mask(grid, ph):
    gx, gy, gz = grid.axes()
    Z, Y, X = np.meshgrid(gz, gy, gx, indexing="ij")
    xc, r = float(ph["arm_x"]), float(ph["arm_r"])
    zin = np.abs(Z) <= C.ARM_HALF_LENGTH
    arm = np.zeros(grid.shape, bool)
    hum = np.zeros(grid.shape, bool)
    for s in (1, -1):
        d2 = (X - s * xc) ** 2 + Y ** 2
        arm |= (d2 <= r * r) & zin
        hum |= (d2 <= C.HUMERUS_RADIUS ** 2) & zin
    return arm & ~hum, hum


def mu_map(grid, ph):
    lab = on_grid(ph, ph["labels"], grid, 0)
    mu = np.zeros(grid.shape, np.float32)
    for name, k in C.LABELS.items():
        mu[lab == k] = C.MU_PER_MM.get(name, 0.0)
    arm, hum = arm_mask(grid, ph)
    mu[arm] = C.MU_PER_MM["soft_tissue"]
    mu[hum] = C.MU_PER_MM["bone"]
    return mu


def truth(grid, ph):
    """True activity concentration on the recon grid (torso + arms) and the label image."""
    lab = on_grid(ph, ph["labels"], grid, 0)
    act = on_grid(ph, ph["activity"].astype(np.float32), grid, 0)
    arm, _ = arm_mask(grid, ph)
    act[arm] = C.ACTIVITY["arm"]
    lab = lab.astype(np.int16)
    lab[arm] = 254
    return act, lab


def lv_frame():
    c = np.array(C.LV_CENTER, float)
    u = np.array(C.LV_AXIS, float)
    u /= np.linalg.norm(u)
    ant = np.array([0.0, 1.0, 0.0])
    e1 = ant - ant.dot(u) * u
    e1 /= np.linalg.norm(e1)          # anterior direction in the short-axis plane
    e2 = np.cross(u, e1)
    return c, u, e1, e2


def lv_coords(grid):
    """Long-axis coordinate w, radial distance r and circumferential angle psi (deg, 0 = anterior)."""
    c, u, e1, e2 = lv_frame()
    gx, gy, gz = grid.axes()
    Z, Y, X = np.meshgrid(gz, gy, gx, indexing="ij")
    P = np.stack([X - c[0], Y - c[1], Z - c[2]], -1)
    w = P @ u
    a1, a2 = P @ e1, P @ e2
    r = np.hypot(a1, a2)
    psi = np.degrees(np.arctan2(a2, a1)) % 360.0
    return w, r, psi

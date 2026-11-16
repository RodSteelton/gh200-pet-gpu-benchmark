"""
List-mode line projector (numba, multi-core CPU).

Each LOR (p1 -> p2, mm) is sampled every `step` mm inside the image box with trilinear
interpolation. TOF: samples are weighted by a Gaussian (sigma along the LOR, mm) centred
`tc` mm from p1 and limited to +/- NSIG sigma. Non-TOF: sigma <= 0, uniform weights.
The back projector is the exact adjoint of the forward projector.
"""

import math

import numpy as np
from numba import njit, prange, get_num_threads

NSIG = 3.0
INV_SQRT_2PI = 1.0 / math.sqrt(2.0 * math.pi)


class Grid:
    """Image grid: voxel centres at origin + i * v, arrays indexed [z, y, x]."""

    def __init__(self, nx, ny, nz, v):
        self.nx, self.ny, self.nz, self.v = int(nx), int(ny), int(nz), float(v)
        self.x0 = -(nx - 1) / 2.0 * v
        self.y0 = -(ny - 1) / 2.0 * v
        self.z0 = -(nz - 1) / 2.0 * v

    @property
    def shape(self):
        return (self.nz, self.ny, self.nx)

    def axes(self):
        return (self.x0 + self.v * np.arange(self.nx), self.y0 + self.v * np.arange(self.ny),
                self.z0 + self.v * np.arange(self.nz))

    def params(self):
        return np.array([self.x0, self.y0, self.z0, self.v, self.nx, self.ny, self.nz], dtype=np.float64)


@njit(inline="always")
def _clip_box(ax, ay, az, dx, dy, dz, D, gp):
    """Parametric range [s0, s1] (mm from p1) of the segment inside the image box."""
    s0, s1 = 0.0, D
    lo = (gp[0], gp[1], gp[2])
    hi = (gp[0] + (gp[4] - 1) * gp[3], gp[1] + (gp[5] - 1) * gp[3], gp[2] + (gp[6] - 1) * gp[3])
    a = (ax, ay, az)
    d = (dx, dy, dz)
    for k in range(3):
        if abs(d[k]) < 1e-12:
            if a[k] < lo[k] or a[k] > hi[k]:
                return 1.0, 0.0
        else:
            t1 = (lo[k] - a[k]) / d[k]
            t2 = (hi[k] - a[k]) / d[k]
            if t1 > t2:
                t1, t2 = t2, t1
            if t1 > s0:
                s0 = t1
            if t2 < s1:
                s1 = t2
    return s0, s1


@njit(inline="always")
def _trilin(img, gp, x, y, z):
    v = gp[3]
    fx = (x - gp[0]) / v
    fy = (y - gp[1]) / v
    fz = (z - gp[2]) / v
    ix = int(math.floor(fx))
    iy = int(math.floor(fy))
    iz = int(math.floor(fz))
    nx, ny, nz = int(gp[4]), int(gp[5]), int(gp[6])
    if ix < 0 or iy < 0 or iz < 0 or ix >= nx - 1 or iy >= ny - 1 or iz >= nz - 1:
        return 0.0
    tx, ty, tz = fx - ix, fy - iy, fz - iz
    base = (iz * ny + iy) * nx + ix
    sy, sz = nx, nx * ny
    c000 = img[base]
    c100 = img[base + 1]
    c010 = img[base + sy]
    c110 = img[base + sy + 1]
    c001 = img[base + sz]
    c101 = img[base + sz + 1]
    c011 = img[base + sz + sy]
    c111 = img[base + sz + sy + 1]
    c00 = c000 * (1 - tx) + c100 * tx
    c10 = c010 * (1 - tx) + c110 * tx
    c01 = c001 * (1 - tx) + c101 * tx
    c11 = c011 * (1 - tx) + c111 * tx
    c0 = c00 * (1 - ty) + c10 * ty
    c1 = c01 * (1 - ty) + c11 * ty
    return c0 * (1 - tz) + c1 * tz


@njit(inline="always")
def _splat(acc, gp, x, y, z, val):
    v = gp[3]
    fx = (x - gp[0]) / v
    fy = (y - gp[1]) / v
    fz = (z - gp[2]) / v
    ix = int(math.floor(fx))
    iy = int(math.floor(fy))
    iz = int(math.floor(fz))
    nx, ny, nz = int(gp[4]), int(gp[5]), int(gp[6])
    if ix < 0 or iy < 0 or iz < 0 or ix >= nx - 1 or iy >= ny - 1 or iz >= nz - 1:
        return
    tx, ty, tz = fx - ix, fy - iy, fz - iz
    base = (iz * ny + iy) * nx + ix
    sy, sz = nx, nx * ny
    acc[base] += val * (1 - tx) * (1 - ty) * (1 - tz)
    acc[base + 1] += val * tx * (1 - ty) * (1 - tz)
    acc[base + sy] += val * (1 - tx) * ty * (1 - tz)
    acc[base + sy + 1] += val * tx * ty * (1 - tz)
    acc[base + sz] += val * (1 - tx) * (1 - ty) * tz
    acc[base + sz + 1] += val * tx * (1 - ty) * tz
    acc[base + sz + sy] += val * (1 - tx) * ty * tz
    acc[base + sz + sy + 1] += val * tx * ty * tz


@njit(inline="always")
def _range(p1, p2, e, tc, sigma, gp):
    ax, ay, az = p1[e, 0], p1[e, 1], p1[e, 2]
    dx, dy, dz = p2[e, 0] - ax, p2[e, 1] - ay, p2[e, 2] - az
    D = math.sqrt(dx * dx + dy * dy + dz * dz)
    if D < 1e-9:
        return ax, ay, az, 0.0, 0.0, 0.0, 1.0, 0.0
    dx, dy, dz = dx / D, dy / D, dz / D
    s0, s1 = _clip_box(ax, ay, az, dx, dy, dz, D, gp)
    if sigma > 0:
        lo = tc[e] - NSIG * sigma
        hi = tc[e] + NSIG * sigma
        if lo > s0:
            s0 = lo
        if hi < s1:
            s1 = hi
    return ax, ay, az, dx, dy, dz, s0, s1


@njit(parallel=True, fastmath=True)
def forward(img, gp, p1, p2, tc, sigma, step):
    """Line integrals (TOF-weighted if sigma > 0) of img along each LOR."""
    E = p1.shape[0]
    out = np.zeros(E, dtype=np.float32)
    for e in prange(E):
        ax, ay, az, dx, dy, dz, s0, s1 = _range(p1, p2, e, tc, sigma, gp)
        if s1 <= s0:
            continue
        n = int(math.ceil((s1 - s0) / step))
        h = (s1 - s0) / n
        acc = 0.0
        for k in range(n):
            s = s0 + (k + 0.5) * h
            w = h
            if sigma > 0:
                u = (s - tc[e]) / sigma
                w = h * INV_SQRT_2PI / sigma * math.exp(-0.5 * u * u)
            acc += w * _trilin(img, gp, ax + s * dx, ay + s * dy, az + s * dz)
        out[e] = acc
    return out


@njit(parallel=True, fastmath=True)
def back(vals, gp, p1, p2, tc, sigma, step, nvox):
    """Adjoint of forward(): distribute vals[e] along each LOR (thread-local buffers)."""
    E = p1.shape[0]
    nt = get_num_threads()
    acc = np.zeros((nt, nvox), dtype=np.float32)
    for t in prange(nt):
        for e in range(t, E, nt):
            if vals[e] == 0:
                continue
            ax, ay, az, dx, dy, dz, s0, s1 = _range(p1, p2, e, tc, sigma, gp)
            if s1 <= s0:
                continue
            n = int(math.ceil((s1 - s0) / step))
            h = (s1 - s0) / n
            for k in range(n):
                s = s0 + (k + 0.5) * h
                w = h
                if sigma > 0:
                    u = (s - tc[e]) / sigma
                    w = h * INV_SQRT_2PI / sigma * math.exp(-0.5 * u * u)
                _splat(acc[t], gp, ax + s * dx, ay + s * dy, az + s * dz, vals[e] * w)
    out = np.zeros(nvox, dtype=np.float32)
    for t in range(nt):
        out += acc[t]
    return out


def fwd(img3d, grid, p1, p2, tc=None, sigma=0.0, step=None):
    step = step or grid.v
    tc = np.zeros(len(p1), np.float32) if tc is None else tc.astype(np.float32)
    return forward(np.ascontiguousarray(img3d, np.float32).ravel(), grid.params(),
                   np.ascontiguousarray(p1, np.float32), np.ascontiguousarray(p2, np.float32),
                   tc, float(sigma), float(step))


def bwd(vals, grid, p1, p2, tc=None, sigma=0.0, step=None):
    step = step or grid.v
    tc = np.zeros(len(p1), np.float32) if tc is None else tc.astype(np.float32)
    out = back(np.ascontiguousarray(vals, np.float32), grid.params(),
               np.ascontiguousarray(p1, np.float32), np.ascontiguousarray(p2, np.float32),
               tc, float(sigma), float(step), grid.nx * grid.ny * grid.nz)
    return out.reshape(grid.shape)

"""
CUDA version of projector.py (numba-cuda), written for the GH200 benchmark.

Same model as the CPU projector: each LOR (p1 -> p2, mm) is sampled every `step` mm inside the image box
with trilinear interpolation; TOF samples are weighted by a Gaussian (sigma along the LOR, mm) centred
`tc` mm from p1 and limited to +/- NSIG sigma. The back projector is the exact adjoint of the forward
projector. One GPU thread per LOR; the back projector uses atomic adds, so voxel sums are accumulated in a
different order than on the CPU and agree with it to float32 rounding, not bit for bit.

    fwd, bwd    drop-in for projector.fwd / projector.bwd (host arrays in, host arrays out)
    osem        drop-in for run_recon.osem; events, image and sensitivity stay on the GPU for the whole
                reconstruction, and each subset is one fused forward + back kernel plus one update kernel

Arithmetic follows the CPU code: LOR direction in float32, positions and weights in float64, images float32.
Test without a GPU: NUMBA_ENABLE_CUDASIM=1 python bench_gpu.py --selftest
"""

import math
import time

import numpy as np
from numba import cuda

NSIG = 3.0
INV_SQRT_2PI = 1.0 / math.sqrt(2.0 * math.pi)
TPB = 256                      # GPU threads per block


def available():
    try:
        return bool(cuda.is_available())
    except Exception:
        return False


def device_name():
    try:
        name = cuda.get_current_device().name
        return name.decode() if isinstance(name, bytes) else str(name)
    except Exception:
        return "unknown"


def _blocks(n):
    return max(1, (int(n) + TPB - 1) // TPB)


# ---------------------------------------------------------------- device functions
@cuda.jit(device=True)
def _clip1(a, d, lo, hi, s0, s1):
    """Clip the parametric range [s0, s1] against one pair of box faces. Empty range: s1 < s0."""
    if abs(d) < 1e-12:
        if a < lo or a > hi:
            return 1.0, 0.0
        return s0, s1
    t1 = (lo - a) / d
    t2 = (hi - a) / d
    if t1 > t2:
        t1, t2 = t2, t1
    if t1 > s0:
        s0 = t1
    if t2 < s1:
        s1 = t2
    return s0, s1


@cuda.jit(device=True)
def _range(p1, p2, tc, e, sigma, x0, y0, z0, v, nx, ny, nz):
    """Start point, unit direction and sampled range [s0, s1] (mm from p1) of event e."""
    ax, ay, az = p1[e, 0], p1[e, 1], p1[e, 2]
    dx, dy, dz = p2[e, 0] - ax, p2[e, 1] - ay, p2[e, 2] - az
    D = np.float32(math.sqrt(dx * dx + dy * dy + dz * dz))
    if D < 1e-9:
        return ax, ay, az, 0.0, 0.0, 0.0, 1.0, 0.0
    dx, dy, dz = dx / D, dy / D, dz / D
    s0, s1 = 0.0, float(D)
    s0, s1 = _clip1(ax, dx, x0, x0 + (nx - 1) * v, s0, s1)
    s0, s1 = _clip1(ay, dy, y0, y0 + (ny - 1) * v, s0, s1)
    s0, s1 = _clip1(az, dz, z0, z0 + (nz - 1) * v, s0, s1)
    if sigma > 0:
        lo = tc[e] - NSIG * sigma
        hi = tc[e] + NSIG * sigma
        if lo > s0:
            s0 = lo
        if hi < s1:
            s1 = hi
    return ax, ay, az, float(dx), float(dy), float(dz), s0, s1


@cuda.jit(device=True)
def _trilin(img, x0, y0, z0, v, nx, ny, nz, x, y, z):
    fx = (x - x0) / v
    fy = (y - y0) / v
    fz = (z - z0) / v
    ix = int(math.floor(fx))
    iy = int(math.floor(fy))
    iz = int(math.floor(fz))
    if ix < 0 or iy < 0 or iz < 0 or ix >= nx - 1 or iy >= ny - 1 or iz >= nz - 1:
        return 0.0
    tx, ty, tz = fx - ix, fy - iy, fz - iz
    base = (iz * ny + iy) * nx + ix
    sy, sz = nx, nx * ny
    c00 = img[base] * (1 - tx) + img[base + 1] * tx
    c10 = img[base + sy] * (1 - tx) + img[base + sy + 1] * tx
    c01 = img[base + sz] * (1 - tx) + img[base + sz + 1] * tx
    c11 = img[base + sz + sy] * (1 - tx) + img[base + sz + sy + 1] * tx
    c0 = c00 * (1 - ty) + c10 * ty
    c1 = c01 * (1 - ty) + c11 * ty
    return c0 * (1 - tz) + c1 * tz


@cuda.jit(device=True)
def _splat(acc, x0, y0, z0, v, nx, ny, nz, x, y, z, val):
    fx = (x - x0) / v
    fy = (y - y0) / v
    fz = (z - z0) / v
    ix = int(math.floor(fx))
    iy = int(math.floor(fy))
    iz = int(math.floor(fz))
    if ix < 0 or iy < 0 or iz < 0 or ix >= nx - 1 or iy >= ny - 1 or iz >= nz - 1:
        return
    tx, ty, tz = fx - ix, fy - iy, fz - iz
    base = (iz * ny + iy) * nx + ix
    sy, sz = nx, nx * ny
    cuda.atomic.add(acc, base, np.float32(val * (1 - tx) * (1 - ty) * (1 - tz)))
    cuda.atomic.add(acc, base + 1, np.float32(val * tx * (1 - ty) * (1 - tz)))
    cuda.atomic.add(acc, base + sy, np.float32(val * (1 - tx) * ty * (1 - tz)))
    cuda.atomic.add(acc, base + sy + 1, np.float32(val * tx * ty * (1 - tz)))
    cuda.atomic.add(acc, base + sz, np.float32(val * (1 - tx) * (1 - ty) * tz))
    cuda.atomic.add(acc, base + sz + 1, np.float32(val * tx * (1 - ty) * tz))
    cuda.atomic.add(acc, base + sz + sy, np.float32(val * (1 - tx) * ty * tz))
    cuda.atomic.add(acc, base + sz + sy + 1, np.float32(val * tx * ty * tz))


@cuda.jit(device=True)
def _line_fwd(img, x0, y0, z0, v, nx, ny, nz, ax, ay, az, dx, dy, dz, s0, s1, tce, sigma, step):
    n = int(math.ceil((s1 - s0) / step))
    h = (s1 - s0) / n
    acc = 0.0
    for k in range(n):
        s = s0 + (k + 0.5) * h
        w = h
        if sigma > 0:
            u = (s - tce) / sigma
            w = h * INV_SQRT_2PI / sigma * math.exp(-0.5 * u * u)
        acc += w * _trilin(img, x0, y0, z0, v, nx, ny, nz, ax + s * dx, ay + s * dy, az + s * dz)
    return acc


@cuda.jit(device=True)
def _line_bwd(acc, x0, y0, z0, v, nx, ny, nz, ax, ay, az, dx, dy, dz, s0, s1, tce, sigma, step, val):
    n = int(math.ceil((s1 - s0) / step))
    h = (s1 - s0) / n
    for k in range(n):
        s = s0 + (k + 0.5) * h
        w = h
        if sigma > 0:
            u = (s - tce) / sigma
            w = h * INV_SQRT_2PI / sigma * math.exp(-0.5 * u * u)
        _splat(acc, x0, y0, z0, v, nx, ny, nz, ax + s * dx, ay + s * dy, az + s * dz, val * w)


# ---------------------------------------------------------------- kernels (one thread per LOR or voxel)
@cuda.jit
def _k_forward(img, x0, y0, z0, v, nx, ny, nz, p1, p2, tc, sigma, step, out):
    e = cuda.grid(1)
    if e >= out.shape[0]:
        return
    ax, ay, az, dx, dy, dz, s0, s1 = _range(p1, p2, tc, e, sigma, x0, y0, z0, v, nx, ny, nz)
    if s1 <= s0:
        out[e] = 0.0
        return
    out[e] = _line_fwd(img, x0, y0, z0, v, nx, ny, nz, ax, ay, az, dx, dy, dz, s0, s1, tc[e], sigma, step)


@cuda.jit
def _k_back(vals, x0, y0, z0, v, nx, ny, nz, p1, p2, tc, sigma, step, acc):
    e = cuda.grid(1)
    if e >= vals.shape[0]:
        return
    if vals[e] == 0:
        return
    ax, ay, az, dx, dy, dz, s0, s1 = _range(p1, p2, tc, e, sigma, x0, y0, z0, v, nx, ny, nz)
    if s1 <= s0:
        return
    _line_bwd(acc, x0, y0, z0, v, nx, ny, nz, ax, ay, az, dx, dy, dz, s0, s1, tc[e], sigma, step, float(vals[e]))


@cuda.jit
def _k_osem(img, bp, x0, y0, z0, v, nx, ny, nz, p1, p2, tc, sigma, step):
    """Fused OSEM step for one LOR: forward project img, back project 1 / projection into bp."""
    e = cuda.grid(1)
    if e >= p1.shape[0]:
        return
    ax, ay, az, dx, dy, dz, s0, s1 = _range(p1, p2, tc, e, sigma, x0, y0, z0, v, nx, ny, nz)
    if s1 <= s0:
        return
    fp = np.float32(_line_fwd(img, x0, y0, z0, v, nx, ny, nz, ax, ay, az, dx, dy, dz, s0, s1, tc[e], sigma, step))
    if fp > 1e-12:
        ratio = np.float32(1.0) / fp
        _line_bwd(bp, x0, y0, z0, v, nx, ny, nz, ax, ay, az, dx, dy, dz, s0, s1, tc[e], sigma, step, float(ratio))


@cuda.jit
def _k_update(img, bp, senss):
    """img <- img * bp / (sens / subsets) inside the mask (senss > 0), 0 outside; clears bp."""
    i = cuda.grid(1)
    if i >= img.shape[0]:
        return
    s = senss[i]
    if s > 0:
        img[i] = img[i] * bp[i] / s
    else:
        img[i] = 0.0
    bp[i] = 0.0


# ---------------------------------------------------------------- host wrappers
def _gp(grid):
    return (float(grid.x0), float(grid.y0), float(grid.z0), float(grid.v), int(grid.nx), int(grid.ny), int(grid.nz))


def _events(p1, p2, tc):
    p1 = np.ascontiguousarray(p1, np.float32)
    p2 = np.ascontiguousarray(p2, np.float32)
    tc = np.zeros(len(p1), np.float32) if tc is None else np.ascontiguousarray(tc, np.float32)
    return p1, p2, tc


def fwd(img3d, grid, p1, p2, tc=None, sigma=0.0, step=None):
    step = step or grid.v
    p1, p2, tc = _events(p1, p2, tc)
    n = len(p1)
    if n == 0:
        return np.zeros(0, np.float32)
    d_out = cuda.device_array(n, np.float32)
    _k_forward[_blocks(n), TPB](cuda.to_device(np.ascontiguousarray(img3d, np.float32).ravel()), *_gp(grid),
                                cuda.to_device(p1), cuda.to_device(p2), cuda.to_device(tc),
                                float(sigma), float(step), d_out)
    return d_out.copy_to_host()


def bwd(vals, grid, p1, p2, tc=None, sigma=0.0, step=None):
    step = step or grid.v
    p1, p2, tc = _events(p1, p2, tc)
    n = len(p1)
    nvox = grid.nx * grid.ny * grid.nz
    d_acc = cuda.to_device(np.zeros(nvox, np.float32))
    if n:
        _k_back[_blocks(n), TPB](cuda.to_device(np.ascontiguousarray(vals, np.float32)), *_gp(grid),
                                 cuda.to_device(p1), cuda.to_device(p2), cuda.to_device(tc),
                                 float(sigma), float(step), d_acc)
    return d_acc.copy_to_host().reshape(grid.shape)


from numba import njit, prange


@njit(parallel=True)
def _strided(p1, p2, tc, s, step, n_s, o1, o2, o3):
    """Parallel copy of events s, s+step, s+2*step, ... into contiguous (pinned) buffers."""
    for j in prange(n_s):
        i = s + j * step
        o1[j, 0] = p1[i, 0]
        o1[j, 1] = p1[i, 1]
        o1[j, 2] = p1[i, 2]
        o2[j, 0] = p2[i, 0]
        o2[j, 1] = p2[i, 1]
        o2[j, 2] = p2[i, 2]
        o3[j] = tc[i]


def _pinned(shape):
    try:
        return cuda.pinned_array(shape, np.float32)
    except Exception:
        return np.empty(shape, np.float32)


def osem(ev, grid, sens, sigma, iters, subsets, rng, snapshots=(), log=print, stats=None, fast=False):
    """List-mode OSEM on the GPU. Same arguments, subsets and result as run_recon.osem.
    stats: optional dict that receives upload_s, iter_s (list), download_s, perm_s, gather_s, copy_s.
    fast: interleaved subsets (event i goes to subset i % subsets, no random permutation) with a parallel host
    gather into pinned buffers; for very large event lists. Subsets differ from the CPU code, so the image
    agrees statistically, not voxel by voxel."""
    p1, p2, tc = _events(ev["p1"], ev["p2"], ev["tc"])
    gp = _gp(grid)
    nvox = grid.nx * grid.ny * grid.nz
    step = float(grid.v)
    sigma = float(sigma)
    mask = sens > 1e-3 * sens.max()
    t0 = time.perf_counter()
    perm = None if fast else rng.permutation(len(p1))
    t_perm = time.perf_counter() - t0

    t0 = time.perf_counter()
    subs = []
    t_gather = t_copy = 0.0
    if fast:
        nmax = -(-len(p1) // subsets)
        b1, b2, b3 = _pinned((nmax, 3)), _pinned((nmax, 3)), _pinned((nmax,))
    for s in range(subsets):
        t1 = time.perf_counter()
        if fast:
            n_s = len(range(s, len(p1), subsets))
            _strided(p1, p2, tc, s, subsets, n_s, b1, b2, b3)
            g1, g2, g3 = b1[:n_s], b2[:n_s], b3[:n_s]
        else:
            idx = perm[s::subsets]
            g1, g2, g3 = p1[idx], p2[idx], tc[idx]
        t2 = time.perf_counter()
        subs.append((cuda.to_device(g1), cuda.to_device(g2), cuda.to_device(g3), len(g1)))
        cuda.synchronize()
        t_gather += t2 - t1
        t_copy += time.perf_counter() - t2
    senss = np.where(mask, sens / subsets, 0.0).astype(np.float32).ravel()
    d_senss = cuda.to_device(senss)
    d_img = cuda.to_device(mask.astype(np.float32).ravel())
    d_bp = cuda.to_device(np.zeros(nvox, np.float32))
    cuda.synchronize()
    t_up = time.perf_counter() - t0

    snaps, t_iter = {}, []
    for it in range(iters):
        t0 = time.perf_counter()
        for d1, d2, dtc, n in subs:
            if n:
                _k_osem[_blocks(n), TPB](d_img, d_bp, *gp, d1, d2, dtc, sigma, step)
            _k_update[_blocks(nvox), TPB](d_img, d_bp, d_senss)
        cuda.synchronize()
        t_iter.append(time.perf_counter() - t0)
        log(f"    iteration {it + 1}/{iters}: {t_iter[-1]:.2f} s (GPU)")
        if it + 1 in snapshots:
            snaps[it + 1] = d_img.copy_to_host().reshape(grid.shape)

    t0 = time.perf_counter()
    img = d_img.copy_to_host().reshape(grid.shape)
    if stats is not None:
        stats.update(upload_s=t_up, iter_s=t_iter, download_s=time.perf_counter() - t0,
                     perm_s=t_perm, gather_s=t_gather, copy_s=t_copy)
    return snaps if snapshots else img

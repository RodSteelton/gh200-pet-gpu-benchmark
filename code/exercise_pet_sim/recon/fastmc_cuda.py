"""
CUDA version of fastmc.py (numba-cuda) for ring scanners: the fast Monte Carlo of trues and the list-mode
sensitivity image, written for the GH200 benchmark.

Same physics as fastmc.py / geometry.Ring: emission from the voxel phantom and the two arm cylinders,
back-to-back photons, front-face entry, exponential interaction depth in LYSO, module acceptance,
attenuation along the line between the two interaction points (survival probability exp(-integral of mu),
same line sampling as the projector), crystal snapping and exact time of flight. No scatter, no randoms.

One GPU thread simulates `n_per` decays in a loop with its own random stream (xoroshiro128+); accepted events
are appended to the output with an atomic counter. Each event carries a key (thread, decay number); the host
sorts by key, so the event list is the same on every run for a given seed, thread count and round plan.
The GPU events are a different random sample than the CPU events for the same seed (different generator and
order of draws): compare them statistically, not event by event.

    generate_trues(ring, grid, ph, mu, n_target, seed)   as fastmc.generate_trues (takes a seed, not an rng)
    emit_timed(ring, grid, ph, mu, n_emit, tcdf, tgrid, seed)   as dynamic._emit_timed (15O-water scans)
    sensitivity(ring, grid, mu, n_detected, seed)        as fastmc.sensitivity

Panels (panels.Panels) are not ported yet.
Test: python bench_mc.py --selftest   (GPU node, or NUMBA_ENABLE_CUDASIM=1 for a tiny simulator run)
"""

import math
import time

import numpy as np
from numba import cuda
from numba.cuda.random import xoroshiro128p_uniform_float64

import geometry as G
from projector_cuda import _clip1, _line_fwd, _line_bwd, TPB

TWO_PI = 2.0 * math.pi
DEG = 180.0 / math.pi
# layout of the parameter vector g (float64)
I_R, I_DEPTH, I_PITCH, I_NT, I_NA, I_ZMAX, I_STEP, I_OFF, I_NFULL, I_MULYSO = 0, 1, 2, 3, 4, 5, 6, 7, 8, 9
I_PARM, I_ARMX, I_ARMR, I_HUMR, I_ARMHL, I_VOX, I_NXP, I_NYP = 10, 11, 12, 13, 14, 15, 16, 17
I_X0, I_Y0, I_Z0, I_V, I_NX, I_NY, I_NZ, I_CMM, I_RC = 18, 19, 20, 21, 22, 23, 24, 25, 26
NG = 27


# ---------------------------------------------------------------- device functions
@cuda.jit(device=True)
def _claim(counter):
    """Next free output slot (atomic)."""
    return cuda.atomic.add(counter, 0, 1)


@cuda.jit(device=True)
def _module(qx, qy, g):
    """Nearest module index and its centre angle (rad), as Ring.module_of."""
    ang = math.atan2(qy, qx) * DEG
    if ang < 0.0:
        ang += 360.0
    nf = int(g[I_NFULL])
    k = int(math.floor((ang - g[I_OFF]) / g[I_STEP] + 0.5)) % nf
    if k < 0:
        k += nf
    return k, (g[I_OFF] + k * g[I_STEP]) / DEG


@cuda.jit(device=True)
def _detect(states, t, ex, ey, ez, dx, dy, dz, g, kept):
    """Photon from e along unit d: interaction point and detected flag, as Ring.detect."""
    R = g[I_R]
    a = dx * dx + dy * dy
    b = 2.0 * (ex * dx + ey * dy)
    c = ex * ex + ey * ey - R * R
    disc = b * b - 4.0 * a * c
    ell = -math.log(1.0 - xoroshiro128p_uniform_float64(states, t)) / g[I_MULYSO]
    if not (a > 1e-12 and c < 0.0 and disc > 0.0):
        return False, 0.0, 0.0, 0.0
    s = (-b + math.sqrt(disc)) / (2.0 * a) + ell
    qx, qy, qz = ex + s * dx, ey + s * dy, ez + s * dz
    k, th = _module(qx, qy, g)
    if kept[k] == 0:
        return False, 0.0, 0.0, 0.0
    cth, sth = math.cos(th), math.sin(th)
    radial = qx * cth + qy * sth
    tang = -qx * sth + qy * cth
    if radial < R or radial > R + g[I_DEPTH]:
        return False, 0.0, 0.0, 0.0
    if abs(tang) > g[I_NT] * g[I_PITCH] / 2.0 or abs(qz) > g[I_ZMAX]:
        return False, 0.0, 0.0, 0.0
    return True, qx, qy, qz


@cuda.jit(device=True)
def _snap(qx, qy, qz, g):
    """Crystal centre (float32) for an interaction point, as Ring.snap."""
    k, th = _module(qx, qy, g)
    c, s = math.cos(th), math.sin(th)
    it = float(math.floor((-qx * s + qy * c) / g[I_PITCH] + g[I_NT] / 2.0))
    it = min(max(it, 0.0), g[I_NT] - 1.0)
    iz = float(math.floor(qz / g[I_PITCH] + g[I_NA] / 2.0))
    iz = min(max(iz, 0.0), g[I_NA] - 1.0)
    tcn = (it - (g[I_NT] - 1.0) / 2.0) * g[I_PITCH]
    zc = (iz - (g[I_NA] - 1.0) / 2.0) * g[I_PITCH]
    rc = g[I_R] + g[I_DEPTH] / 2.0
    return np.float32(rc * c - tcn * s), np.float32(rc * s + tcn * c), np.float32(zc)


@cuda.jit(device=True)
def _lor(g, ax, ay, az, bx, by, bz):
    """Unit direction and the range [s0, s1] inside the image box for the float32 segment a -> b."""
    dx, dy, dz = bx - ax, by - ay, bz - az
    D = np.float32(math.sqrt(dx * dx + dy * dy + dz * dz))
    if D < 1e-9:
        return 0.0, 0.0, 0.0, 1.0, 0.0
    dx, dy, dz = dx / D, dy / D, dz / D
    v = g[I_V]
    s0, s1 = 0.0, float(D)
    s0, s1 = _clip1(ax, dx, g[I_X0], g[I_X0] + (g[I_NX] - 1.0) * v, s0, s1)
    s0, s1 = _clip1(ay, dy, g[I_Y0], g[I_Y0] + (g[I_NY] - 1.0) * v, s0, s1)
    s0, s1 = _clip1(az, dz, g[I_Z0], g[I_Z0] + (g[I_NZ] - 1.0) * v, s0, s1)
    return float(dx), float(dy), float(dz), s0, s1


@cuda.jit(device=True)
def _mu_int(mu, g, ax, ay, az, bx, by, bz):
    """Line integral of mu between two float32 points (projector sampling, step = voxel size)."""
    dx, dy, dz, s0, s1 = _lor(g, ax, ay, az, bx, by, bz)
    if s1 <= s0:
        return 0.0
    return _line_fwd(mu, g[I_X0], g[I_Y0], g[I_Z0], g[I_V], int(g[I_NX]), int(g[I_NY]), int(g[I_NZ]),
                     ax, ay, az, dx, dy, dz, s0, s1, np.float32(0.0), 0.0, g[I_V])


@cuda.jit(device=True)
def _chord(p1x, p1y, p1z, p2x, p2y, p2z, rc, zmax):
    """Length of the segment p1 -> p2 inside the cylinder (radius rc, |z| <= zmax)."""
    dx, dy, dz = p2x - p1x, p2y - p1y, p2z - p1z
    D = math.sqrt(dx * dx + dy * dy + dz * dz)
    a = dx * dx + dy * dy
    b = 2.0 * (p1x * dx + p1y * dy)
    c = p1x * p1x + p1y * p1y - rc * rc
    root = math.sqrt(max(b * b - 4.0 * a * c, 0.0))
    den = 2.0 * max(a, 1e-12)
    lo = max((-b - root) / den, 0.0)
    hi = min((-b + root) / den, 1.0)
    if abs(dz) > 1e-9:
        z0 = (-zmax - p1z) / dz
        z1 = (zmax - p1z) / dz
        lo = max(lo, min(z0, z1))
        hi = min(hi, max(z0, z1))
    return max(hi - lo, 0.0) * D


@cuda.jit(device=True)
def _direction(states, t):
    cz = 2.0 * xoroshiro128p_uniform_float64(states, t) - 1.0
    ph = TWO_PI * xoroshiro128p_uniform_float64(states, t)
    sz = math.sqrt(1.0 - cz * cz)
    return sz * math.cos(ph), sz * math.sin(ph), cz


# ---------------------------------------------------------------- kernels (one thread = n_per decays)
@cuda.jit
def _k_trues(states, n_per, n_extra, cdf, px, py, pz, g, kept, mu, tcdf, tgrid,
             o_p1, o_p2, o_dt, o_e, o_t, o_key, counter):
    """Thread t simulates n_per decays (one more if t < n_extra). If tcdf has more than one element, an
    emission time is drawn from it for each accepted event (as np.interp(u, tcdf, tgrid)) into o_t."""
    t = cuda.grid(1)
    if t >= states.shape[0]:
        return
    cap = o_dt.shape[0]
    nvp = cdf.shape[0]
    ntc = tcdf.shape[0]
    nxp, nyp = int(g[I_NXP]), int(g[I_NYP])
    vox = g[I_VOX]
    nk = n_per
    if t < n_extra:
        nk += 1
    for k in range(nk):
        # emission point: arm cylinders or torso voxel phantom
        if xoroshiro128p_uniform_float64(states, t) < g[I_PARM]:
            rh, r = g[I_HUMR], g[I_ARMR]
            rr = math.sqrt(rh * rh + xoroshiro128p_uniform_float64(states, t) * (r * r - rh * rh))
            phi = TWO_PI * xoroshiro128p_uniform_float64(states, t)
            side = 1.0
            if xoroshiro128p_uniform_float64(states, t) >= 0.5:
                side = -1.0
            ex = side * g[I_ARMX] + rr * math.cos(phi)
            ey = rr * math.sin(phi)
            ez = (2.0 * xoroshiro128p_uniform_float64(states, t) - 1.0) * g[I_ARMHL]
        else:
            u = xoroshiro128p_uniform_float64(states, t)
            lo, hi = 0, nvp - 1
            while lo < hi:
                mid = (lo + hi) // 2
                if cdf[mid] < u:
                    lo = mid + 1
                else:
                    hi = mid
            ex = px[lo % nxp] + (xoroshiro128p_uniform_float64(states, t) - 0.5) * vox
            ey = py[(lo // nxp) % nyp] + (xoroshiro128p_uniform_float64(states, t) - 0.5) * vox
            ez = pz[lo // (nxp * nyp)] + (xoroshiro128p_uniform_float64(states, t) - 0.5) * vox
        dx, dy, dz = _direction(states, t)
        ok, q1x, q1y, q1z = _detect(states, t, ex, ey, ez, dx, dy, dz, g, kept)
        if not ok:
            continue
        ok, q2x, q2y, q2z = _detect(states, t, ex, ey, ez, -dx, -dy, -dz, g, kept)
        if not ok:
            continue
        att = math.exp(-_mu_int(mu, g, np.float32(q1x), np.float32(q1y), np.float32(q1z),
                                np.float32(q2x), np.float32(q2y), np.float32(q2z)))
        if xoroshiro128p_uniform_float64(states, t) >= att:
            continue
        s1x, s1y, s1z = _snap(q1x, q1y, q1z, g)
        s2x, s2y, s2z = _snap(q2x, q2y, q2z, g)
        d1 = math.sqrt((q1x - ex) ** 2 + (q1y - ey) ** 2 + (q1z - ez) ** 2)
        d2 = math.sqrt((q2x - ex) ** 2 + (q2y - ey) ** 2 + (q2z - ez) ** 2)
        tt = 0.0
        if ntc > 1:
            u = xoroshiro128p_uniform_float64(states, t)
            lo, hi = 0, ntc - 1
            while lo < hi:
                mid = (lo + hi) // 2
                if tcdf[mid] < u:
                    lo = mid + 1
                else:
                    hi = mid
            tt = tgrid[lo]
            if lo > 0 and tcdf[lo] > tcdf[lo - 1]:
                tt = tgrid[lo - 1] + (u - tcdf[lo - 1]) / (tcdf[lo] - tcdf[lo - 1]) * (tgrid[lo] - tgrid[lo - 1])
        i = _claim(counter)
        if i < cap:
            o_p1[i, 0], o_p1[i, 1], o_p1[i, 2] = s1x, s1y, s1z
            o_p2[i, 0], o_p2[i, 1], o_p2[i, 2] = s2x, s2y, s2z
            o_dt[i] = np.float32((d1 - d2) / g[I_CMM])
            o_e[i, 0], o_e[i, 1], o_e[i, 2] = np.float32(ex), np.float32(ey), np.float32(ez)
            o_t[i] = np.float32(tt)
            o_key[i] = np.int64(t) * (n_per + 1) + k


@cuda.jit
def _k_sens(states, n_per, g, kept, mu, sens, counter):
    t = cuda.grid(1)
    if t >= states.shape[0]:
        return
    rc, zmax = g[I_RC], g[I_ZMAX]
    for k in range(n_per):
        r = rc * math.sqrt(xoroshiro128p_uniform_float64(states, t))
        phi = TWO_PI * xoroshiro128p_uniform_float64(states, t)
        ex, ey = r * math.cos(phi), r * math.sin(phi)
        ez = (2.0 * xoroshiro128p_uniform_float64(states, t) - 1.0) * zmax
        dx, dy, dz = _direction(states, t)
        ok, q1x, q1y, q1z = _detect(states, t, ex, ey, ez, dx, dy, dz, g, kept)
        if not ok:
            continue
        ok, q2x, q2y, q2z = _detect(states, t, ex, ey, ez, -dx, -dy, -dz, g, kept)
        if not ok:
            continue
        s1x, s1y, s1z = _snap(q1x, q1y, q1z, g)
        s2x, s2y, s2z = _snap(q2x, q2y, q2z, g)
        L = _chord(float(s1x), float(s1y), float(s1z), float(s2x), float(s2y), float(s2z), rc, zmax)
        if L <= 1.0:
            continue
        ux, uy, uz, s0, s1 = _lor(g, s1x, s1y, s1z, s2x, s2y, s2z)
        if s1 > s0:
            nx, ny, nz = int(g[I_NX]), int(g[I_NY]), int(g[I_NZ])
            line = _line_fwd(mu, g[I_X0], g[I_Y0], g[I_Z0], g[I_V], nx, ny, nz,
                             s1x, s1y, s1z, ux, uy, uz, s0, s1, np.float32(0.0), 0.0, g[I_V])
            val = np.float32(math.exp(-line) / L)
            _line_bwd(sens, g[I_X0], g[I_Y0], g[I_Z0], g[I_V], nx, ny, nz,
                      s1x, s1y, s1z, ux, uy, uz, s0, s1, np.float32(0.0), 0.0, g[I_V], float(val))
        cuda.atomic.add(counter, 0, 1)


# ---------------------------------------------------------------- host side
def params(ring, grid, ph=None):
    """Parameter vector g for the kernels (ring scanners only)."""
    C = G.C
    if not hasattr(ring, "kept_mask"):
        raise NotImplementedError("fastmc_cuda supports geometry.Ring only (panels are not ported yet)")
    g = np.zeros(NG, np.float64)
    g[[I_R, I_DEPTH, I_PITCH, I_NT, I_NA, I_ZMAX]] = (ring.R, G.DEPTH, G.PITCH, G.NT, G.NA, G.ZMAX)
    g[[I_STEP, I_OFF, I_NFULL, I_MULYSO]] = (ring.step, ring.offset, ring.n_full, C.MU_LYSO_PER_MM)
    g[[I_X0, I_Y0, I_Z0, I_V, I_NX, I_NY, I_NZ]] = (grid.x0, grid.y0, grid.z0, grid.v, grid.nx, grid.ny, grid.nz)
    g[I_CMM], g[I_RC] = C.C_MM_PER_PS, ring.R - 5.0
    if ph is not None:
        a_t, a_a = float(ph["a_torso"]), float(ph["a_arm_each"])
        nzp, nyp, nxp = ph["activity"].shape
        g[[I_PARM, I_ARMX, I_ARMR, I_HUMR, I_ARMHL]] = (2 * a_a / (a_t + 2 * a_a), float(ph["arm_x"]),
                                                        float(ph["arm_r"]), C.HUMERUS_RADIUS, C.ARM_HALF_LENGTH)
        g[[I_VOX, I_NXP, I_NYP]] = (C.VOXEL_MM, nxp, nyp)
    return g


def phantom_cdf(ph):
    cdf = np.cumsum(ph["activity"].ravel().astype(np.float64))
    return cdf / cdf[-1]


def _simulated():
    from numba import config
    return bool(getattr(config, "ENABLE_CUDASIM", False))


def _states(threads, seed):
    from numba.cuda.random import create_xoroshiro128p_states
    return create_xoroshiro128p_states(threads, seed=seed)


def _blocks(n):
    return max(1, (int(n) + TPB - 1) // TPB)


class _Sim:
    """Device-side state for the trues kernel: one scanner, phantom and attenuation map."""

    def __init__(self, ring, grid, ph, mu, seed, threads, tcdf=None, tgrid=None):
        self.threads = int(threads or (64 if _simulated() else 1 << 18))
        self.g = cuda.to_device(params(ring, grid, ph))
        self.kept = cuda.to_device(ring.kept_mask.astype(np.uint8))
        self.mu = cuda.to_device(np.ascontiguousarray(mu, np.float32).ravel())
        self.cdf = cuda.to_device(phantom_cdf(ph))
        self.pxyz = [cuda.to_device(np.ascontiguousarray(ph[k], np.float64)) for k in ("x", "y", "z")]
        self.timed = tcdf is not None
        self.tcdf = cuda.to_device(np.ascontiguousarray(tcdf, np.float64) if self.timed else np.zeros(1))
        self.tgrid = cuda.to_device(np.ascontiguousarray(tgrid, np.float64) if self.timed else np.zeros(1))
        self.states = _states(self.threads, seed)

    def round(self, n_decays, cap):
        """Simulate exactly n_decays; returns (events dict sorted by key, capacity used). If the output was
        too small the same decays are simulated again with more room."""
        n_per, n_extra = divmod(int(n_decays), self.threads)
        while True:
            saved = self.states.copy_to_host()
            d_p1, d_p2 = cuda.device_array((cap, 3), np.float32), cuda.device_array((cap, 3), np.float32)
            d_e, d_dt = cuda.device_array((cap, 3), np.float32), cuda.device_array(cap, np.float32)
            d_t, d_key = cuda.device_array(cap, np.float32), cuda.device_array(cap, np.int64)
            d_cnt = cuda.to_device(np.zeros(1, np.int32))
            _k_trues[_blocks(self.threads), TPB](self.states, n_per, n_extra, self.cdf, *self.pxyz, self.g, self.kept,
                                                 self.mu, self.tcdf, self.tgrid, d_p1, d_p2, d_dt, d_e, d_t, d_key, d_cnt)
            n = int(d_cnt.copy_to_host()[0])
            if n <= cap:
                break
            self.states = cuda.to_device(saved)
            cap = int(n * 1.1) + 1000
        order = np.argsort(d_key.copy_to_host()[:n], kind="stable")
        ev = {k: d.copy_to_host()[:n][order] for k, d in (("p1", d_p1), ("p2", d_p2), ("dt", d_dt), ("e", d_e))}
        if self.timed:
            ev["t"] = d_t.copy_to_host()[:n][order]
        return ev


def _cap(expected):
    return int(expected * 1.15 + 10 * math.sqrt(max(expected, 0.0)) + 1000)


def generate_trues(ring, grid, ph, mu, n_target, seed, threads=None, round_trues=4_000_000, pilot=100, log=print,
                   stats=None):
    """List-mode trues on the GPU: snapped endpoints p1, p2 (float32 mm), true dt = t1 - t2 (ps), emission e,
    n_emitted. Runs in rounds until n_target trues exist; the first (pilot) round measures the acceptance."""
    n_target = int(n_target)
    t_start = time.perf_counter()
    sim = _Sim(ring, grid, ph, mu, seed, threads)
    out = {"p1": [], "p2": [], "dt": [], "e": []}
    n_have, n_emit, rounds = 0, 0, 0
    n_dec = sim.threads * int(pilot)
    cap = max(1000, n_dec // 10)
    while n_have < n_target:
        ev = sim.round(n_dec, cap)
        for k in out:
            out[k].append(ev[k])
        n_have += len(ev["dt"])
        n_emit += n_dec
        rounds += 1
        eff = max(n_have / n_emit, 1e-9)
        log(f"  fastmc (GPU): {n_have:,} trues from {n_emit:,} emissions")
        want = min(round_trues, int((n_target - n_have) * 1.03) + 1000)
        n_dec = sim.threads * max(1, int(math.ceil(want / eff / sim.threads)))
        cap = _cap(n_dec * eff)
    res = {k: np.concatenate(v)[:n_target] for k, v in out.items()}
    res["n_emitted"] = np.int64(n_emit * min(1.0, n_target / max(n_have, 1)))
    if stats is not None:
        stats.update(seconds=time.perf_counter() - t_start, emissions=int(n_emit), trues=int(n_have),
                     rounds=rounds, threads=sim.threads)
    return res


def emit_timed(ring, grid, ph, mu, n_emit, tcdf, tgrid, seed, threads=None, round_decays=2_000_000_000, pilot=100,
               log=print, stats=None):
    """Exactly n_emit decays with emission times, as dynamic._emit_timed: returns p1, p2, dt and t (s) of the
    detected trues. ph is a class phantom (dynamic.class_phantom); tcdf is the cumulative emission-time
    distribution on tgrid."""
    n_emit = int(n_emit)
    t_start = time.perf_counter()
    sim = _Sim(ring, grid, ph, mu, seed, threads, tcdf=tcdf, tgrid=tgrid)
    out = {"p1": [], "p2": [], "dt": [], "t": []}
    done, n_have = 0, 0
    n_dec = min(n_emit, sim.threads * int(pilot))
    cap = max(1000, n_dec // 5)
    while done < n_emit:
        ev = sim.round(n_dec, cap)
        for k in out:
            out[k].append(ev[k])
        done += n_dec
        n_have += len(ev["dt"])
        eff = max(n_have / done, 1e-9)
        n_dec = min(n_emit - done, int(round_decays))
        cap = _cap(n_dec * eff)
    if stats is not None:
        stats.update(seconds=time.perf_counter() - t_start, emissions=int(done), trues=int(n_have), threads=sim.threads)
    return {k: np.concatenate(v) for k, v in out.items()}


def sensitivity(ring, grid, mu, n_detected, seed, threads=None, round_detected=5_000_000, pilot=100, log=print,
                stats=None):
    """List-mode sensitivity image on the GPU (uniform cylinder source), as fastmc.sensitivity."""
    sim = _simulated()
    threads = int(threads or (64 if sim else 1 << 18))
    n_detected = int(n_detected)
    t_start = time.perf_counter()
    d_g = cuda.to_device(params(ring, grid))
    d_kept = cuda.to_device(ring.kept_mask.astype(np.uint8))
    d_mu = cuda.to_device(np.ascontiguousarray(mu, np.float32).ravel())
    d_sens = cuda.to_device(np.zeros(grid.nx * grid.ny * grid.nz, np.float32))
    d_cnt = cuda.to_device(np.zeros(1, np.int32))
    states = _states(threads, seed)
    n_have, n_emit, n_per = 0, 0, int(pilot)
    while n_have < n_detected:
        _k_sens[_blocks(threads), TPB](states, n_per, d_g, d_kept, d_mu, d_sens, d_cnt)
        n_have = int(d_cnt.copy_to_host()[0])
        n_emit += threads * n_per
        log(f"  sensitivity (GPU): {n_have:,} detected from {n_emit:,} emissions")
        eff = max(n_have / n_emit, 1e-9)
        want = min(round_detected, int((n_detected - n_have) * 1.03) + 1000)
        n_per = max(1, int(math.ceil(want / eff / threads)))
    if stats is not None:
        stats.update(seconds=time.perf_counter() - t_start, emissions=int(n_emit), detected=int(n_have), threads=threads)
    return d_sens.copy_to_host().reshape(grid.shape)

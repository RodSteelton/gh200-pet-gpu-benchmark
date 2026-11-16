"""
Fast Monte Carlo of the scanner model (independent of GATE): emission from the phantom activity,
back-to-back photons, front-face entry, exponential interaction depth in LYSO, module acceptance,
attenuation, crystal snapping and exact time of flight. No scatter, no randoms.

Also estimates the list-mode sensitivity image from a uniform cylinder source:
    s_j  ~  sum over detected events of  a_ij * att_i / L_i
where L_i is the LOR length inside the uniform source; this is unbiased for sum_i eps_i att_i a_ij
and includes the depth-of-interaction and module-acceptance efficiency of each LOR.
"""

import numpy as np

import geometry as G
from projector import fwd, bwd

C = G.C


def _iso(rng, n):
    cz = rng.uniform(-1, 1, n)
    ph = rng.uniform(0, 2 * np.pi, n)
    sz = np.sqrt(1 - cz * cz)
    return np.stack([sz * np.cos(ph), sz * np.sin(ph), cz], 1)


def _pairs(ring, e, rng):
    d = _iso(rng, len(e))
    q1, ok1 = ring.detect(e, d, rng)
    q2, ok2 = ring.detect(e, -d, rng)
    ok = ok1 & ok2
    return q1[ok], q2[ok], e[ok]


def sample_emissions(ph, n, rng):
    """Emission points from torso activity (voxel phantom) and the two arm cylinders."""
    act = ph["activity"].ravel().astype(np.float64)
    cdf = np.cumsum(act)
    cdf /= cdf[-1]
    a_t, a_a = float(ph["a_torso"]), float(ph["a_arm_each"])
    n_arm = rng.binomial(n, 2 * a_a / (a_t + 2 * a_a))
    n_t = n - n_arm
    idx = np.searchsorted(cdf, rng.random(n_t))
    iz, iy, ix = np.unravel_index(idx, ph["activity"].shape)
    v = C.VOXEL_MM
    et = np.stack([ph["x"][ix], ph["y"][iy], ph["z"][iz]], 1) + rng.uniform(-v / 2, v / 2, (n_t, 3))
    xc, r, rh = float(ph["arm_x"]), float(ph["arm_r"]), C.HUMERUS_RADIUS
    rr = np.sqrt(rng.uniform(rh * rh, r * r, n_arm))
    phi = rng.uniform(0, 2 * np.pi, n_arm)
    side = np.where(rng.random(n_arm) < 0.5, 1.0, -1.0)
    ea = np.stack([side * xc + rr * np.cos(phi), rr * np.sin(phi),
                   rng.uniform(-C.ARM_HALF_LENGTH, C.ARM_HALF_LENGTH, n_arm)], 1)
    return np.concatenate([et, ea])


def generate_trues(ring, grid, ph, mu, n_target, rng, chunk=4_000_000, log=print):
    """List-mode trues: snapped endpoints p1, p2 (float32 mm), true dt = t1 - t2 (ps), emission e."""
    out = {"p1": [], "p2": [], "dt": [], "e": []}
    n_have, n_emit = 0, 0
    while n_have < n_target:
        e = sample_emissions(ph, chunk, rng)
        n_emit += chunk
        q1, q2, e = _pairs(ring, e, rng)
        att = np.exp(-fwd(mu, grid, q1.astype(np.float32), q2.astype(np.float32)))
        keep = rng.random(len(att)) < att
        q1, q2, e = q1[keep], q2[keep], e[keep]
        s1, _ = ring.snap(q1)
        s2, _ = ring.snap(q2)
        dt = (np.linalg.norm(q1 - e, axis=1) - np.linalg.norm(q2 - e, axis=1)) / C.C_MM_PER_PS
        for k, v in (("p1", s1), ("p2", s2), ("dt", dt.astype(np.float32)), ("e", e.astype(np.float32))):
            out[k].append(v)
        n_have += len(s1)
        log(f"  fastmc: {n_have:,} trues from {n_emit:,} emissions")
    res = {k: np.concatenate(v)[:n_target] for k, v in out.items()}
    res["n_emitted"] = np.int64(n_emit * min(1.0, n_target / max(n_have, 1)))
    return res


def _chord_in_cylinder(p1, p2, rc, zmax):
    d = p2 - p1
    D = np.linalg.norm(d, axis=1)
    a = d[:, 0] ** 2 + d[:, 1] ** 2
    b = 2 * (p1[:, 0] * d[:, 0] + p1[:, 1] * d[:, 1])
    c = p1[:, 0] ** 2 + p1[:, 1] ** 2 - rc * rc
    disc = np.maximum(b * b - 4 * a * c, 0)
    t0 = (-b - np.sqrt(disc)) / (2 * np.maximum(a, 1e-12))
    t1 = (-b + np.sqrt(disc)) / (2 * np.maximum(a, 1e-12))
    with np.errstate(divide="ignore", invalid="ignore"):
        z0 = np.where(np.abs(d[:, 2]) > 1e-9, (-zmax - p1[:, 2]) / d[:, 2], -np.inf)
        z1 = np.where(np.abs(d[:, 2]) > 1e-9, (zmax - p1[:, 2]) / d[:, 2], np.inf)
    zlo, zhi = np.minimum(z0, z1), np.maximum(z0, z1)
    lo = np.maximum.reduce([t0, zlo, np.zeros_like(t0)])
    hi = np.minimum.reduce([t1, zhi, np.ones_like(t1)])
    return np.maximum(hi - lo, 0) * D


def sensitivity(ring, grid, mu, n_detected, rng, chunk=6_000_000, log=print):
    rc = ring.R - 5.0
    zmax = G.ZMAX
    sens = np.zeros(grid.shape, np.float32)
    n_have, n_emit = 0, 0
    while n_have < n_detected:
        r = rc * np.sqrt(rng.random(chunk))
        phi = rng.uniform(0, 2 * np.pi, chunk)
        e = np.stack([r * np.cos(phi), r * np.sin(phi), rng.uniform(-zmax, zmax, chunk)], 1)
        n_emit += chunk
        q1, q2, _ = _pairs(ring, e, rng)
        s1, _ = ring.snap(q1)
        s2, _ = ring.snap(q2)
        L = _chord_in_cylinder(s1.astype(np.float64), s2.astype(np.float64), rc, zmax)
        ok = L > 1.0
        s1, s2, L = s1[ok], s2[ok], L[ok]
        att = np.exp(-fwd(mu, grid, s1, s2))
        sens += bwd((att / L).astype(np.float32), grid, s1, s2)
        n_have += len(s1)
        log(f"  sensitivity: {n_have:,} detected from {n_emit:,} emissions")
    return sens

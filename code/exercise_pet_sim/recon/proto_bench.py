"""
Stage 10: benchmark of the simulation model against the measured i3M CardioPET prototype (2021).

Prototype (Lamprou et al., IEEE NSS 2021; Cardio_HW_review.pptx): 2 flat panels, each 6 x 4 detectors of
8 x 8 LYSO crystals 3 x 3 x 10 mm at 3.36 mm pitch (one-to-one KETEK SiPMs, TOFPET2), 161 x 108 mm of
crystal per panel, faces 280 mm apart (assumed: the distance used for the source-array images).
Measured: sensitivity 0.46% at the center (22Na point, 30% energy window, maximum-energy hit only),
0.87% with inter-crystal scatter (ICS) events; profiles across the FOV (slide 9); CTR 237 ps; energy
resolution 11.5%; spatial resolution 1.5 mm parallel to the panels, about 5 mm across (slide 9).

Two models of the same panels:
  slab   the model used in stages 8 to 8c (panels.Panels): exponential depth with the 511 keV total
         attenuation of LYSO, any interaction counts, no gaps, no energy window
  full   photon transport in the pixelated crystal (Woodcock tracking): photoelectric with Lu K x-ray
         emission and transport, Klein-Nishina Compton, reflector and module gaps, per-crystal energy
         resolution and threshold; events kept by (a) maximum-energy crystal in the window or (b) panel
         energy sum in the window (ICS kept)
Also: the same full model for the proposed 3 x 3 x 20 mm (1 inch modules) panels, to give the factor
that converts slab-model counts into energy-windowed counts, and a point-source TOF MLEM for resolution.

    python proto_bench.py --out DIR [--n 2e6] [--nprof 4e5] [--quick]
Output DIR/proto/: SUMMARY_proto.txt, proto_metrics.json, proto_images.npz, Fig_proto_bench.
    python proto_bench.py --out DIR --report        figure and summary only
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np

import geometry as G
import panels as P
from projector import Grid, bwd, fwd

C = G.C
C_MM_PS = C.C_MM_PER_PS

# ---------------------------------------------------------------- LYSO cross sections (1/mm)
RHO_E_SCALE = 0.0515 / 0.2866e-24            # electrons per mm3 x 1e-24, set so mu_Compton(511) = 0.0515/mm
MU_PE_511 = 0.0305                           # photoelectric at 511 keV (total 0.087 incl. ~0.005 Rayleigh)
K_EDGE, K_JUMP, K_XRAY, K_YIELD = 63.3, 5.6, 55.0, 0.79   # Lu: edge keV, jump, mean K x-ray keV, P(K shell) x fluorescence


def sigma_kn(E):
    """Klein-Nishina total cross section per electron (1e-24 cm2)."""
    k = E / 511.0
    re2 = 0.07941                              # r_e^2 in 1e-24 cm2
    a = (1 + k) / k ** 2 * (2 * (1 + k) / (1 + 2 * k) - np.log(1 + 2 * k) / k)
    b = np.log(1 + 2 * k) / (2 * k) - (1 + 3 * k) / (1 + 2 * k) ** 2
    return 2 * np.pi * re2 * (a + b)


def mu_parts(E):
    pe = MU_PE_511 * (511.0 / E) ** 2.8
    pe = np.where(E < K_EDGE, pe / K_JUMP, pe)
    co = RHO_E_SCALE * sigma_kn(E) * 1e-24
    return pe, co


def kn_sample(E, rng):
    """Compton: scattered energy and cos(theta), Klein-Nishina by Butcher-Messel sampling (as Geant4)."""
    k = E / 511.0
    n = len(E)
    eps = np.empty(n)
    todo = np.arange(n)
    while len(todo):
        kk = k[todo]
        e0 = 1 / (1 + 2 * kk)
        a1 = -np.log(e0)
        a2 = 0.5 * (1 - e0 * e0)
        r1, r2, r3 = rng.random((3, len(todo)))
        ep = np.where(r1 < a1 / (a1 + a2), np.exp(-a1 * r2), np.sqrt(e0 * e0 + (1 - e0 * e0) * r2))
        t = (1 - ep) / (kk * ep)
        s2 = t * (2 - t)
        ok = r3 <= 1 - ep * s2 / (1 + ep * ep)
        eps[todo[ok]] = ep[ok]
        todo = todo[~ok]
    cos = 1 - (1 - eps) / (k * eps)
    return E * eps, np.clip(cos, -1, 1)


def rotate(u, cos, rng):
    """New directions at polar angle acos(cos) from u, uniform azimuth."""
    phi = rng.uniform(0, 2 * np.pi, len(u))
    sin = np.sqrt(np.maximum(1 - cos * cos, 0))
    a = np.where(np.abs(u[:, 2:3]) < 0.9, np.array([[0, 0, 1.0]]), np.array([[1.0, 0, 0]]))
    e1 = np.cross(u, a)
    e1 /= np.linalg.norm(e1, axis=1, keepdims=True)
    e2 = np.cross(u, e1)
    return cos[:, None] * u + sin[:, None] * (np.cos(phi)[:, None] * e1 + np.sin(phi)[:, None] * e2)


def iso(rng, n):
    cz = rng.uniform(-1, 1, n)
    ph = rng.uniform(0, 2 * np.pi, n)
    sz = np.sqrt(1 - cz * cz)
    return np.stack([sz * np.cos(ph), sz * np.sin(ph), cz], 1)


# ---------------------------------------------------------------- pixelated panel pair
class PixPanels:
    """Two parallel panels with normals +y and -y, faces at y = +/-gap/2; local axes x (long) and z (short)."""

    def __init__(self, gap=280.0, depth=10.0, pitch=3.36, xtal=3.0, nmod=(6, 4), npm=8, mod_pitch=27.0,
                 ewin=(434.0, 588.0), eres=0.115, thr=30.0):
        self.H, self.D, self.p, self.x, self.npm = gap / 2.0, depth, pitch, xtal, npm
        self.mp = mod_pitch
        self.nmx, self.nmz = nmod
        self.Wx, self.Wz = self.nmx * mod_pitch, self.nmz * mod_pitch
        self.ncx, self.ncz = self.nmx * npm, self.nmz * npm
        self.ewin, self.eres, self.thr = ewin, eres, thr

    def _axis(self, v, W):
        """Crystal index along one face axis, or -1 in a reflector or module gap."""
        a = v + W / 2.0
        im = np.floor(a / self.mp)
        loc = a - im * self.mp - (self.mp - self.npm * self.p) / 2.0
        ic = np.floor(loc / self.p)
        off = loc - ic * self.p - (self.p - self.x) / 2.0
        ok = (im >= 0) & (im < W / self.mp - 1e-9) & (ic >= 0) & (ic < self.npm) & (off >= 0) & (off <= self.x)
        return np.where(ok, (im * self.npm + ic).astype(int), -1)

    def crystal_id(self, q, k):
        ix = self._axis(q[:, 0], self.Wx)
        iz = self._axis(q[:, 2], self.Wz)
        ok = (ix >= 0) & (iz >= 0)
        return np.where(ok, (k * self.ncx + ix) * self.ncz + iz, -1)

    def center(self, cid):
        k, r = np.divmod(cid, self.ncx * self.ncz)
        ix, iz = np.divmod(r, self.ncz)

        def pos(i, W):
            im, ic = np.divmod(i, self.npm)
            return -W / 2.0 + im * self.mp + (self.mp - self.npm * self.p) / 2.0 + (ic + 0.5) * self.p

        y = np.where(k == 0, 1.0, -1.0) * (self.H + self.D / 2.0)
        return np.stack([pos(ix, self.Wx), y, pos(iz, self.Wz)], 1)

    def entry(self, s, u):
        """Front-face entry of photons from points s along u: point, panel index (0: +y, 1: -y), hit flag."""
        k = np.where(u[:, 1] > 0, 0, 1)
        sg = np.where(k == 0, 1.0, -1.0)
        with np.errstate(divide="ignore", invalid="ignore"):
            t = (sg * self.H - s[:, 1]) / u[:, 1]
        q = s + t[:, None] * u
        hit = (t > 0) & (np.abs(q[:, 0]) <= self.Wx / 2) & (np.abs(q[:, 2]) <= self.Wz / 2)
        return q, k, hit

    def transport(self, q, u, k, rng):
        """Woodcock tracking of 511 keV photons entering at q. Returns deposits (photon, crystal, keV)
        and the first-interaction point of each photon (nan if none)."""
        n = len(q)
        pid = np.arange(n)
        E = np.full(n, 511.0)
        p, u = q.copy(), u.copy()
        sg = np.where(k == 0, 1.0, -1.0)
        first = np.full((n, 3), np.nan)
        dep_p, dep_c, dep_e = [], [], []
        for _ in range(400):
            if not len(pid):
                break
            pe, co = mu_parts(E)
            mu = pe + co
            p = p + (rng.exponential(1.0, len(pid)) / mu)[:, None] * u
            r = sg * p[:, 1] - self.H
            inbox = (r >= 0) & (r <= self.D) & (np.abs(p[:, 0]) <= self.Wx / 2) & (np.abs(p[:, 2]) <= self.Wz / 2)
            cid = np.where(inbox, self.crystal_id(p, k), -1)
            real = cid >= 0
            # photons outside the box leave; inside a gap: virtual collision, keep going
            keep = inbox.copy()
            ir = np.nonzero(real)[0]
            if len(ir):
                f = np.isnan(first[pid[ir], 0])
                first[pid[ir[f]]] = p[ir[f]]
                is_pe = rng.random(len(ir)) < pe[ir] / mu[ir]
                ip = ir[is_pe]
                fl = (E[ip] > K_EDGE) & (rng.random(len(ip)) < K_YIELD)
                dep_p.append(pid[ip]); dep_c.append(cid[ip]); dep_e.append(np.where(fl, E[ip] - K_XRAY, E[ip]))
                keep[ip[~fl]] = False
                ixr = ip[fl]
                E[ixr] = K_XRAY
                u[ixr] = iso(rng, len(ixr))
                ic = ir[~is_pe]
                E2, cos = kn_sample(E[ic], rng)
                dep_p.append(pid[ic]); dep_c.append(cid[ic]); dep_e.append(E[ic] - E2)
                E[ic] = E2
                u[ic] = rotate(u[ic], cos, rng)
                low = ic[E2 < 15.0]
                dep_p.append(pid[low]); dep_c.append(cid[low]); dep_e.append(E[low])
                keep[low] = False
            pid, E, p, u, k, sg = pid[keep], E[keep], p[keep], u[keep], k[keep], sg[keep]
        cat = lambda a, t: np.concatenate(a).astype(t) if a else np.zeros(0, t)
        return cat(dep_p, np.int64), cat(dep_c, np.int64), cat(dep_e, np.float64), first

    def readout(self, n, dp, dc, de, rng):
        """Per photon: max-crystal id and energy, panel energy sum (channels above threshold)."""
        emax = np.zeros(n)
        cmax = np.full(n, -1)
        esum = np.zeros(n)
        if not len(dp):
            return cmax, emax, esum
        key = dp * (2 * self.ncx * self.ncz) + dc
        uk, inv = np.unique(key, return_inverse=True)
        e = np.bincount(inv, de)
        ph, cr = np.divmod(uk, 2 * self.ncx * self.ncz)
        sig = self.eres / 2.3548 * np.sqrt(511.0 * np.maximum(e, 1e-3))
        e = e + rng.normal(0, 1, len(e)) * sig
        on = e >= self.thr
        ph, cr, e = ph[on], cr[on], e[on]
        np.add.at(esum, ph, e)
        o = np.lexsort((e, ph))
        ph, cr, e = ph[o], cr[o], e[o]
        last = np.r_[ph[1:] != ph[:-1], True]
        emax[ph[last]] = e[last]
        cmax[ph[last]] = cr[last]
        return cmax, emax, esum


ACOL_FWHM = 0.0087                               # rad, photon acollinearity (0.5 deg FWHM)


def full_run(pp, src, n, rng, windows, chunk=500_000, keep_events=False, ctr=237.0, acol=True):
    """Coincidences from n decays at point src. Returns counts per (mode, window) and, optionally, LORs."""
    cnt = {(m, w): 0 for m in ("max", "sum") for w in windows}
    ev = {"p1": [], "p2": [], "tc": []}
    done = 0
    while done < n:
        m = min(chunk, n - done)
        done += m
        u = iso(rng, m)
        s = np.broadcast_to(np.asarray(src, float), (m, 3))
        v = -u
        if acol:
            dev = ACOL_FWHM / 2.3548 * np.hypot(rng.normal(0, 1, m), rng.normal(0, 1, m))
            v = rotate(v, np.cos(dev), rng)
        q1, k1, h1 = pp.entry(s, u)
        q2, k2, h2 = pp.entry(s, v)
        ok = h1 & h2 & (k1 != k2)
        if not ok.any():
            continue
        u1, v1, q1, k1, q2, k2 = u[ok], v[ok], q1[ok], k1[ok], q2[ok], k2[ok]
        nn = len(u1)
        r1 = pp.transport(q1, u1, k1, rng)
        r2 = pp.transport(q2, v1, k2, rng)
        c1, e1, s1 = pp.readout(nn, *r1[:3], rng)
        c2, e2, s2 = pp.readout(nn, *r2[:3], rng)
        for w in windows:
            lo, hi = w
            a = (e1 >= lo) & (e1 <= hi) & (e2 >= lo) & (e2 <= hi)
            b = (s1 >= lo) & (s1 <= hi) & (s2 >= lo) & (s2 <= hi)
            cnt[("max", w)] += int(a.sum())
            cnt[("sum", w)] += int(b.sum())
            if keep_events and w == windows[0]:
                f1, f2 = r1[3][a], r2[3][a]
                p1, p2 = pp.center(c1[a]), pp.center(c2[a])
                d1 = np.linalg.norm(f1 - src, axis=1)
                d2 = np.linalg.norm(f2 - src, axis=1)
                D = np.linalg.norm(p2 - p1, axis=1)
                tc = 0.5 * (D + d1 - d2) + rng.normal(0, 1, len(D)) * C_MM_PS * ctr / 2.3548 / 2.0
                ev["p1"].append(p1); ev["p2"].append(p2); ev["tc"].append(tc)
    if keep_events:
        ev = {k: np.concatenate(v).astype(np.float32) for k, v in ev.items()}
    return cnt, ev


# ---------------------------------------------------------------- slab model (stages 8 to 8c)
def slab_scanner(gap=280.0, depth=10.0, nt=49, na=33):
    name = f"proto_{gap:.0f}_{depth:.0f}_{nt}_{na}"
    P.PANEL_CONFIGS[name] = dict(depth=depth, na=na, panels=[(90.0, gap / 2, nt, 0.0), (270.0, gap / 2, nt, 0.0)])
    return P.Panels(name)


def slab_sens(scn, src, n, rng, chunk=2_000_000):
    """Panels.panels have normals +y/-y and transaxial axis along x: same frame as PixPanels."""
    hit = 0
    done = 0
    while done < n:
        m = min(chunk, n - done)
        done += m
        e = np.broadcast_to(np.asarray(src, float), (m, 3)).copy()
        d = iso(rng, m)
        _, ok1 = scn.detect(e, d, rng)
        _, ok2 = scn.detect(e, -d, rng)
        hit += int((ok1 & ok2).sum())
    return hit / n


# ---------------------------------------------------------------- point-source resolution
def fwhm(prof, v):
    i = int(np.argmax(prof))
    if 0 < i < len(prof) - 1:                    # parabolic peak (NEMA)
        a, b, c = prof[i - 1], prof[i], prof[i + 1]
        den = a - 2 * b + c
        pk = b - 0.25 * (a - c) ** 2 / den if den < 0 else b
    else:
        pk = prof[i]
    h = pk / 2
    l = i
    while l > 0 and prof[l] > h:
        l -= 1
    r = i
    while r < len(prof) - 1 and prof[r] > h:
        r += 1
    xl = l + (h - prof[l]) / (prof[l + 1] - prof[l]) if prof[l + 1] != prof[l] else l
    xr = r - 1 + (prof[r - 1] - h) / (prof[r - 1] - prof[r]) if prof[r - 1] != prof[r] else r
    return (xr - xl) * v


def response_fwhm(ev, src, bins=0.1):
    """Detector response: FWHM of the closest-approach offset of each LOR from the source, along x and z."""
    p1, p2 = ev["p1"].astype(float), ev["p2"].astype(float)
    d = p2 - p1
    d /= np.linalg.norm(d, axis=1, keepdims=True)
    w = np.asarray(src, float) - p1
    cl = p1 + (w * d).sum(1, keepdims=True) * d - np.asarray(src, float)
    out = {}
    for ax, i in (("x", 0), ("z", 2)):
        h, e = np.histogram(cl[:, i], bins=np.arange(-6, 6 + bins, bins))
        hs = np.convolve(h, np.ones(3) / 3, mode="same")
        out[ax] = fwhm(hs, bins)
    return out


def point_recon(ev, src, ctr, iters, v=1.0, half=30):
    n = 2 * half + 1
    g = Grid(n, n, n, v)
    c0 = np.round(np.asarray(src, np.float32))          # grid on whole mm: the source is off the voxel centers
    p1 = ev["p1"] - c0
    p2 = ev["p2"] - c0
    sig = C_MM_PS * ctr / 2.3548 / 2.0
    img = np.ones(g.shape, np.float32)
    out = {}
    for it in range(1, max(iters) + 1):
        f = fwd(img, g, p1, p2, ev["tc"], sig, step=v / 2)
        r = np.where(f > 0, 1.0 / np.maximum(f, 1e-20), 0).astype(np.float32)
        img = img * bwd(r, g, p1, p2, ev["tc"], sig, step=v / 2) / len(p1)
        if it in iters:
            k = np.unravel_index(np.argmax(img), img.shape)
            out[it] = dict(x=fwhm(img[k[0], k[1], :], v), y=fwhm(img[k[0], :, k[2]], v), z=fwhm(img[:, k[1], k[2]], v),
                           img=img.copy())
    return out


# ---------------------------------------------------------------- main
MEAS = dict(sens_max=0.46, sens_sum=0.87, res_par=1.5, res_perp=5.0)
# slide 9 of Cardio_HW_review.pptx, read from the plot (offset cm, sensitivity %). The axes are not labeled
# relative to the panels: the broad "X" profile (to +/-12 cm) is taken as the axis between the panels, "Z"
# (zero beyond 8 cm) as the long face axis and "Y" (zero beyond 6 cm) as the short face axis.
MEAS_PROF = {"normal": ([-12, -10, -8, -6, -4, -2, 0, 2, 4, 6, 8, 10, 12],
                        [0.18, 0.20, 0.23, 0.26, 0.31, 0.37, 0.46, 0.375, 0.30, 0.25, 0.22, 0.20, 0.18]),
             "long": ([-8, -7, -6, -5, -4, -3, -2, 0, 2, 3, 4, 5, 6, 7, 8],
                      [0.03, 0.08, 0.13, 0.19, 0.23, 0.28, 0.35, 0.385, 0.35, 0.30, 0.25, 0.19, 0.13, 0.07, 0.03]),
             "short": ([-6, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6],
                       [0.02, 0.13, 0.22, 0.30, 0.38, 0.38, 0.38, 0.30, 0.23, 0.13, 0.06, 0.02])}
# slide 9 resolution (FWHM mm) over offsets -5 to +5 cm: two components about 1.5, one about 5 (3.7 to 5.5)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=float, default=2e6)
    ap.add_argument("--nprof", type=float, default=4e5)
    ap.add_argument("--seed", type=int, default=10)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--report", action="store_true", help="figure and summary from proto_metrics.json only")
    a = ap.parse_args()
    out = Path(a.out) / "proto"
    out.mkdir(parents=True, exist_ok=True)
    if a.report:
        return report(out)
    rng = np.random.default_rng(a.seed)
    n, nprof = int(a.n), int(a.nprof)
    if a.quick:
        n, nprof = n // 10, nprof // 10
    W = [(434.0, 588.0), (358.0, 664.0), (400.0, 650.0)]
    res = {"windows": W}
    t0 = time.time()

    # 1. center sensitivity: full model (several gaps, windows) and slab model
    res["center"] = {}
    for gap in (240.0, 280.0, 320.0):
        pp = PixPanels(gap=gap)
        cnt, _ = full_run(pp, (0, 0, 0), n, rng, W)
        sl = slab_sens(slab_scanner(gap), (0, 0, 0), n, rng)
        res["center"][gap] = dict(slab=100 * sl, **{f"{m}_{w[0]:.0f}_{w[1]:.0f}": 100 * c / n for (m, w), c in cnt.items()})
        print(f"gap {gap:.0f}: slab {100 * sl:.3f}%  " + "  ".join(f"{m}{w[0]:.0f}-{w[1]:.0f} {100 * c / n:.3f}%"
                                                              for (m, w), c in cnt.items()), f"{time.time() - t0:.0f} s", flush=True)

    # 2. profiles along the three axes at gap 280
    pp = PixPanels(gap=280.0)
    scn = slab_scanner(280.0)
    offs = {"normal": np.arange(-120, 121, 20.0), "long": np.arange(-120, 121, 20.0), "short": np.arange(-80, 81, 20.0)}
    axis = {"normal": 1, "long": 0, "short": 2}
    res["profiles"] = {}
    for name, oo in offs.items():
        rows = []
        for o in oo:
            src = np.zeros(3)
            src[axis[name]] = o
            cnt, _ = full_run(pp, src, nprof, rng, W[:1])
            rows.append(dict(off=float(o), max=100 * cnt[("max", W[0])] / nprof, sum=100 * cnt[("sum", W[0])] / nprof,
                             slab=100 * slab_sens(scn, src, nprof, rng)))
        res["profiles"][name] = rows
        print(name, " ".join(f"{r['off']:.0f}:{r['max']:.3f}" for r in rows), f"{time.time() - t0:.0f} s", flush=True)

    # 3. proposed detector: 3 x 3 x 20 mm, 1 inch modules (8 x 8 at 3.2 mm pitch), panels 30 x 24 cm, gap 400
    res["proposed"] = {}
    for depth in (10.0, 20.0):
        pq = PixPanels(gap=400.0, depth=depth, pitch=3.2, xtal=3.0, nmod=(12, 9), npm=8, mod_pitch=25.4)
        cnt, _ = full_run(pq, (0, 0, 0), n, rng, W)
        sl = slab_sens(slab_scanner(400.0, depth, nt=int(round(12 * 25.4 / G.PITCH)), na=int(round(9 * 25.4 / G.PITCH))),
                       (0, 0, 0), n, rng)
        res["proposed"][depth] = dict(slab=100 * sl, **{f"{m}_{w[0]:.0f}_{w[1]:.0f}": 100 * c / n for (m, w), c in cnt.items()})
        print(f"proposed {depth:.0f} mm: slab {100 * sl:.3f}%", {f"{m}{w[0]:.0f}": round(100 * c / n, 3) for (m, w), c in cnt.items()},
              f"{time.time() - t0:.0f} s", flush=True)
    # ring-like crystals (3.2 mm at 3.3 mm pitch, 20 mm, 7 x 8 blocks, energy summed per block): faces 560 mm apart
    pr = PixPanels(gap=560.0, depth=20.0, pitch=3.3, xtal=3.2, nmod=(12, 9), npm=8, mod_pitch=26.4)
    cnt, _ = full_run(pr, (0, 0, 0), n, rng, W)
    sl = slab_sens(slab_scanner(560.0, 20.0, nt=96, na=72), (0, 0, 0), n, rng)
    res["ringlike"] = dict(slab=100 * sl, **{f"{m}_{w[0]:.0f}_{w[1]:.0f}": 100 * c / n for (m, w), c in cnt.items()})
    print("ring-like 20 mm:", res["ringlike"], flush=True)

    # 4. point-source resolution, 237 ps, sources off the crystal grid (as a real source); FWHM along
    #    long (x), normal (y) and short (z) axes; 0.5 mm voxels
    res["resolution"] = {}
    shots = {}
    nres = max(n, 400_000)
    for lab, src in (("center", (0.9, 0.6, 1.3)), ("center b", (-1.4, -0.8, 0.4)), ("long +40", (40.9, 0.6, 1.3)),
                     ("short +30", (0.9, 0.6, 31.3)), ("normal +40", (0.9, 40.6, 1.3))):
        _, ev = full_run(pp, np.array(src, float), nres, rng, W[:1], keep_events=True)
        rr = point_recon(ev, src, 237.0, iters=(3, 10), v=1.0, half=20)
        resp = response_fwhm(ev, src)
        res["resolution"][lab] = {it: {k: float(v) for k, v in d.items() if k != "img"} for it, d in rr.items()}
        res["resolution"][lab]["response"] = resp
        res["resolution"][lab]["n_events"] = int(len(ev["p1"]))
        shots[lab] = rr[3]["img"]
        print(lab, len(ev["p1"]), "response x %.2f z %.2f" % (resp["x"], resp["z"]),
              {it: (round(d['x'], 2), round(d['y'], 2), round(d['z'], 2)) for it, d in rr.items()}, f"{time.time() - t0:.0f} s", flush=True)
    np.savez_compressed(out / "proto_images.npz", **{k.replace(" ", "_"): v for k, v in shots.items()})
    (out / "proto_metrics.json").write_text(json.dumps(res, indent=1, default=str))
    print("done", f"{time.time() - t0:.0f} s")
    report(out)


# ---------------------------------------------------------------- figure and summary
def report(out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    res = json.loads((out / "proto_metrics.json").read_text())
    c = res["center"]
    g = {float(k): v for k, v in c.items()}
    k15 = "434_588"
    fig, ax = plt.subplots(2, 2, figsize=(11, 8.2))
    # A. center sensitivity
    a = ax[0, 0]
    lab = ["Measured", "Slab model\n(stages 8 to 8c)", "Full model\nmax hit", "Full model\nICS summed"]
    meas = [MEAS["sens_max"], None, None, MEAS["sens_sum"]]
    mod = [None, g[280]["slab"], g[280][f"max_{k15}"], g[280][f"sum_{k15}"]]
    lo = [None, g[320]["slab"], g[320][f"max_{k15}"], g[320][f"sum_{k15}"]]
    hi = [None, g[240]["slab"], g[240][f"max_{k15}"], g[240][f"sum_{k15}"]]
    x = np.arange(4)
    a.bar(0 - 0.18, MEAS["sens_max"], 0.34, color="#444", label="measured, max hit")
    a.bar(0 + 0.18, MEAS["sens_sum"], 0.34, color="#999", label="measured, ICS kept")
    cols = ["#d98a3d", "#2a78d6", "#0f8b8d"]
    for i in (1, 2, 3):
        a.bar(i, mod[i], 0.5, color=cols[i - 1])
        a.errorbar(i, mod[i], yerr=[[mod[i] - lo[i]], [hi[i] - mod[i]]], color="k", capsize=4, lw=1)
        a.text(i, hi[i] + 0.08, f"{mod[i]:.2f}%", ha="center", fontsize=9)
    a.text(-0.18, MEAS["sens_max"] + 0.08, "0.46%", ha="center", fontsize=9)
    a.text(0.18, MEAS["sens_sum"] + 0.08, "0.87%", ha="center", fontsize=9)
    a.set_xticks(x, lab, fontsize=9)
    a.set_ylabel("Sensitivity at center (%)")
    a.set_title("A. Prototype, 22Na point at center", loc="left", fontsize=11)
    a.legend(fontsize=8, frameon=False, loc="upper right")
    a.set_ylim(0, hi[1] * 1.12)
    a.text(0.98, 0.70, "faces 280 mm apart\nbars: 240 to 320 mm\nwindow 434 to 588 keV", transform=a.transAxes,
           ha="right", fontsize=8, color="#555")
    # B. profiles, each normalized to its own peak
    a = ax[0, 1]
    pc = {"normal": "#2a78d6", "long": "#0f8b8d", "short": "#d98a3d"}
    for nm, rows in res["profiles"].items():
        o = np.array([r["off"] for r in rows]) / 10
        m = np.array([r["max"] for r in rows])
        a.plot(o, m / m.max(), "-", color=pc[nm], label=f"model, {nm} axis")
        mo, mv = MEAS_PROF[nm]
        a.plot(mo, np.array(mv) / max(mv), "o", ms=4, color=pc[nm], mfc="white", label=f"measured, {nm}")
    a.set_xlabel("Source offset (cm)")
    a.set_ylabel("Sensitivity / peak")
    a.set_ylim(0, 1.1)
    a.set_title("B. Sensitivity profiles (shape)", loc="left", fontsize=11)
    a.set_ylim(0, 1.45)
    a.legend(fontsize=7.5, frameon=False, ncol=3, loc="upper center")
    # C. proposed detector and ring-like crystals: slab vs energy-windowed
    a = ax[1, 0]
    groups = [("Prototype\n3x3x10, 28 cm", g[280]), ("Proposed panels\n3x3x10, 40 cm", res["proposed"]["10.0"]),
              ("Proposed panels\n3x3x20, 40 cm", res["proposed"]["20.0"]), ("Ring crystals\n3.2x3.2x20, 56 cm", res["ringlike"])]
    w = 0.26
    for i, (nm, d) in enumerate(groups):
        for j, (key, col, l) in enumerate((("slab", "#d98a3d", "slab model"), (f"sum_{k15}", "#0f8b8d", "full, ICS summed"),
                                           (f"max_{k15}", "#2a78d6", "full, max hit"))):
            a.bar(i + (j - 1) * w, d[key], w, color=col, label=l if i == 0 else None)
        a.text(i, d["slab"] + 0.3, f"x{d[f'sum_{k15}'] / d['slab']:.2f}\nx{d[f'max_{k15}'] / d['slab']:.2f}", ha="center", fontsize=8)
    a.set_xticks(range(4), [x[0] for x in groups], fontsize=8)
    a.set_ylabel("Sensitivity at center (%)")
    a.set_ylim(0, max(d["slab"] for _, d in groups) * 1.3)
    a.set_title("C. Energy-windowed counts relative to the slab model", loc="left", fontsize=11)
    a.legend(fontsize=8, frameon=False, loc="upper left")
    # D. resolution
    a = ax[1, 1]
    rr = res["resolution"]
    labs = list(rr.keys())
    par = [np.mean([rr[l]["3"]["x"], rr[l]["3"]["z"]]) for l in labs]
    nor = [rr[l]["3"]["y"] for l in labs]
    xx = np.arange(len(labs))
    a.bar(xx - 0.18, par, 0.34, color="#0f8b8d", label="model, parallel to panels (mean of 2 axes)")
    a.bar(xx + 0.18, nor, 0.34, color="#2a78d6", label="model, normal to panels")
    a.axhspan(1.3, 1.6, color="#0f8b8d", alpha=0.15, label="measured, parallel (about 1.5)")
    a.axhspan(3.7, 5.5, color="#2a78d6", alpha=0.15, label="measured, third axis (3.7 to 5.5)")
    a.set_xticks(xx, [l.replace("center b", "center 2").replace(" +", "\n+") + (" mm" if "+" in l else "") for l in labs], fontsize=8)
    a.set_ylabel("FWHM (mm)")
    a.set_ylim(0, 8)
    a.set_title("D. Point-source resolution, TOF MLEM 3 it., 1 mm voxels", loc="left", fontsize=11)
    a.legend(fontsize=7.5, frameon=False, loc="upper left")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(out / f"Fig_proto_bench.{ext}", dpi=200)
    plt.close(fig)

    p, q, r20, r10, rl = g[280], g, res["proposed"]["20.0"], res["proposed"]["10.0"], res["ringlike"]
    L = []
    L.append("Stage 10: model benchmark against the i3M CardioPET prototype (2021)")
    L.append("Prototype: 2 panels, 6 x 4 detectors of 8 x 8 LYSO 3 x 3 x 10 mm at 3.36 mm pitch (161 x 108 mm), faces")
    L.append("280 mm apart (assumed), 11.5% energy resolution, 30 keV channel threshold, 22Na point source, CTR 237 ps.")
    L.append("Energy window 434 to 588 keV (30% wide); 358 to 664 and 400 to 650 keV in proto_metrics.json.")
    L.append("")
    L.append("Center sensitivity (%)            measured   slab model   full, max hit   full, ICS summed")
    L.append(f"  faces 280 mm                      {MEAS['sens_max']:.2f} / {MEAS['sens_sum']:.2f}   {p['slab']:.2f}        {p[f'max_{k15}']:.2f}            {p[f'sum_{k15}']:.2f}")
    for gg in (240.0, 320.0):
        L.append(f"  faces {gg:.0f} mm                                 {q[gg]['slab']:.2f}        {q[gg][f'max_{k15}']:.2f}            {q[gg][f'sum_{k15}']:.2f}")
    L.append(f"  ICS gain (summed / max hit): measured {MEAS['sens_sum'] / MEAS['sens_max']:.2f}, model {p[f'sum_{k15}'] / p[f'max_{k15}']:.2f}")
    L.append(f"  measured / model at 280 mm: {MEAS['sens_max'] / p[f'max_{k15}']:.2f} (max hit), {MEAS['sens_sum'] / p[f'sum_{k15}']:.2f} (ICS)")
    L.append(f"  at 320 mm: {MEAS['sens_max'] / q[320.0][f'max_{k15}']:.2f} and {MEAS['sens_sum'] / q[320.0][f'sum_{k15}']:.2f}")
    L.append("")
    L.append("Profiles (model, max hit, faces 280 mm; normalized to peak) vs slide 9:")
    for nm, rows in res["profiles"].items():
        m = max(r["max"] for r in rows)
        mo, mv = MEAS_PROF[nm]
        L.append(f"  {nm:6s} model " + " ".join(f"{r['off'] / 10:+.0f}:{r['max'] / m:.2f}" for r in rows if r["off"] >= 0))
        L.append(f"  {'':6s} meas  " + " ".join(f"{o:+d}:{v / max(mv):.2f}" for o, v in zip(mo, mv) if o >= 0))
    L.append("")
    L.append("Energy-windowed counts relative to the slab model (center point; ICS summed / max hit):")
    for nm, d in (("prototype 3x3x10, faces 28 cm", p), ("proposed 30 x 24 cm panels, 3x3x10, faces 40 cm", r10),
                  ("proposed 30 x 24 cm panels, 3x3x20, faces 40 cm", r20), ("ring-like crystals 3.2x3.2x20, faces 56 cm", rl)):
        L.append(f"  {nm:48s} slab {d['slab']:.2f}%  ICS {d[f'sum_{k15}']:.2f}% (x{d[f'sum_{k15}'] / d['slab']:.2f})  max {d[f'max_{k15}']:.2f}% (x{d[f'max_{k15}'] / d['slab']:.2f})")
    L.append(f"  20 mm / 10 mm: slab {r20['slab'] / r10['slab']:.2f}, ICS {r20[f'sum_{k15}'] / r10[f'sum_{k15}']:.2f}, max hit {r20[f'max_{k15}'] / r10[f'max_{k15}']:.2f}")
    L.append("")
    L.append("Point-source resolution, FWHM mm (x long, y normal, z short), TOF 237 ps MLEM, 1 mm voxels:")
    for l, d in res["resolution"].items():
        L.append(f"  {l:11s} {d['n_events']:6d} events  3 it: {d['3']['x']:.1f} / {d['3']['y']:.1f} / {d['3']['z']:.1f}   10 it: {d['10']['x']:.1f} / {d['10']['y']:.1f} / {d['10']['z']:.1f}")
    L.append("  measured (slide 9): two axes about 1.5, third about 5 (3.7 to 5.5), offsets -5 to +5 cm")
    L.append("")
    L.append("Caveats: face distance during the sensitivity measurement not reported (280 mm assumed); 22Na activity may")
    L.append("not include the 90% positron branching; module gaps and crystal size from the paper (3.0 mm crystal assumed);")
    L.append("no dead channels, multiple-coincidence losses or positron range; slide 9 read by eye and its axes are not")
    L.append("labeled relative to the panels; MLEM point-source FWHM depends on iterations, counts and the recon used.")
    (out / "SUMMARY_proto.txt").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()

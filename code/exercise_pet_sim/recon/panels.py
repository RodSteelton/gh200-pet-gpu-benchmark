"""
Stage 8: flat-panel detector layouts compared with the 56 cm 290° open ring and the 70 cm closed ring.

    python panels.py --check                                                  geometry table only
    python panels.py --recon /scratch/$USER/exercise_pet_recon --task N       one (geometry, heart pose)
    python panels.py --recon /scratch/$USER/exercise_pet_recon --collect      figures and tables
    python panels.py --recon /scratch/$USER/exercise_pet_recon --smoke        2 tasks at 1% counts + collect

Geometries (axial FOV 237.6 mm, LYSO 3.2 mm crystals at 3.3 mm pitch, 20 mm deep unless noted, no DOI):
  R280_slot    56 cm 290° open ring (reference)
  R350_closed  70 cm closed ring (reference)
  P2_AP40      2 panels, front and back, 40 cm wide, faces 200 mm from the body axis (40 mm from the P90 chest)
  P2_AP40_d10  as P2_AP40 with 10 mm crystals (depth of the i3M prototype)
  P2_LAO40     2 panels rotated 30° (anterior left, posterior right), 40 cm wide, faces 225 mm from the axis
  P2_AP67      2 panels, front and back, 67 cm wide: same crystal volume as the open ring
  P4_eq        4 panels, 34 cm wide (two in front, two behind, arm slots at the sides), faces 230 mm from the
               axis, front and back pairs meeting at the midline: same crystal volume as the open ring
Two-panel layouts are centred transaxially on the heart (base pose).

Equal injected activity and scan time: every geometry simulates the same number of decays as the stage-2
P90 open-ring data set (10 M trues), so the number of trues follows each geometry's sensitivity.
Per heart pose: P90 body with the 20% mid-anterior defect, per-pose sensitivity image, pose-matched
uniform-LV correction from 50 M trues, OSEM 8 x 10 at 400, 320 and 200 ps.
Output <recon>/panels/: task_*.json/.npz, panels_metrics.csv, Fig_panels_layout, Fig_panels_sens,
Fig_panels_metrics, Fig_panels_images, SUMMARY_panels.txt.
Caveats: trues only (randoms, scatter and dead time not modelled; panels close to the body see more singles),
no backrest, rigid heart, one noise realisation per pose.
"""

import argparse
import csv
import json
import multiprocessing as mp
import os
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

import geometry as G
import evaluate as E
import fastmc
import robust as R
import robust2 as R2
from projector import fwd, bwd
from run_recon import log

C = G.C
import scanner as SC  # noqa: E402

TAG = "panels"
CHUNK = int(os.environ.get("XPET_CHUNK", 2_000_000))
N_DEF_SPLIT = 6
HEART = np.array(C.LV_CENTER[:2], float)
FACE4_20, FACE4_30 = (float(os.environ.get("XPET_FACE4_20", 215)), float(os.environ.get("XPET_FACE4_30", 225)))          # base pose (captured before any set_pose)
REF = "R280_slot"
for _c in (400, 150, 130):
    E.CTRNAME.setdefault(_c, f"{_c} ps")


def _heart_t(theta):
    t = np.deg2rad(theta)
    return float(-HEART[0] * np.sin(t) + HEART[1] * np.cos(t))


def _two(n, face, rot=0.0, depth=20.0, na=None):
    return dict(depth=depth, na=na or G.NA, panels=[(a, face, n, _heart_t(a)) for a in (90.0 + rot, 270.0 + rot)])


def _four(n, face, na=None):
    phi = float(np.degrees(np.arctan(n * G.PITCH / 2.0 / face))) + 1.0     # 1° gap at the midline
    return dict(depth=20.0, na=na or G.NA, panels=[(a, face, n, 0.0) for a in (90 - phi, 90 + phi, 270 - phi, 270 + phi)])


# (normal angle deg, face distance from axis mm, transaxial crystals, transaxial centre offset mm)
PANEL_CONFIGS = {
    "P2_AP40": _two(121, 200.0),
    "P2_AP40_d10": _two(121, 200.0, depth=10.0),
    "P2_LAO40": _two(121, 225.0, rot=-30.0),
    "P2_AP67": _two(203, 200.0),
    "P4_eq": _four(102, 230.0),
    # stage 8b: i3M panel size 20 x 30 cm, both orientations (transaxial T x axial A, cm)
    "P2_T20A30": _two(61, 200.0, na=91),
    "P2_T30A20": _two(91, 200.0, na=61),
    "P4_T20A30": _four(61, FACE4_20, na=91),
    "P4_T30A20": _four(91, FACE4_30, na=61),
    # stage 8c: the 4-panel, 30 cm wide layout with more axial length
    "P4_T30A24": _four(91, FACE4_30, na=72),
    "P4_T30A30": _four(91, FACE4_30, na=91),
    # stage 9: 2 panels 30 x 24 (front and back), and Anto's layout: front and back panels plus two lateral panels;
    # with the arms down the lateral panels sit outside the arms (arm surface at 345 mm, face at 365 mm)
    "P2_T30A24": _two(91, 200.0, na=72),
    "P4_ANTO": dict(depth=20.0, na=72, panels=[(90.0, 200.0, 91, _heart_t(90.0)), (270.0, 200.0, 91, _heart_t(270.0)),
                                               (0.0, 365.0, 91, _heart_t(0.0)), (180.0, 365.0, 91, _heart_t(180.0))]),
}
STAGES = {
    "8": dict(configs=["R280_slot", "R350_closed", "P2_AP40", "P2_AP40_d10", "P2_LAO40", "P2_AP67", "P4_eq"],
              ctrs="400,320,200", outdir="panels", tag="panels"),
    "8b": dict(configs=["R280_slot", "R350_closed", "P2_AP40", "P4_eq", "P2_T20A30", "P2_T30A20", "P4_T20A30", "P4_T30A20"],
               ctrs="130,150,200", outdir="panels2", tag="panels2"),
}
# stage 8c appends two geometries to 8b (same output folder; 8b task files are reused, new tasks are 40-49)
STAGES["8c"] = dict(STAGES["8b"], configs=STAGES["8b"]["configs"] + ["P4_T30A24", "P4_T30A30"])
CONFIGS = STAGES["8"]["configs"]
NEC_STATE = "stress"
# materials cost, i3M model of 2026-10-25: 388 EUR per cm2 of detector face (3 x 3 x 20 mm LYSO, SiPMs, ASICs, DAQ,
# mechanics, cooling, computing, phantoms) plus a fixed 75 kEUR per system; 1 EUR = 1.141 USD
EUR_USD = 1.141
USD_PER_CM2 = 388.0 * EUR_USD
USD_FIXED = 75000.0 * EUR_USD
WIN = 0.65          # energy-windowed / slab-model counts (stage 10 benchmark, ICS summed, 20 mm LYSO)
LABEL = {"R280_slot": "56 cm ring, 290°", "R350_closed": "70 cm closed ring", "P2_AP40": "2 panels AP, 40 cm",
         "P2_AP40_d10": "2 panels AP, 40 cm, 10 mm LYSO", "P2_LAO40": "2 panels LAO/RPO, 40 cm",
         "P2_AP67": "2 panels AP, 67 cm", "P4_eq": "4 panels, 34 cm",
         "P2_T20A30": "2 panels, 20 cm wide x 30 cm axial", "P2_T30A20": "2 panels, 30 cm wide x 20 cm axial",
         "P4_T20A30": "4 panels, 20 cm wide x 30 cm axial", "P4_T30A20": "4 panels, 30 cm wide x 20 cm axial",
         "P4_T30A24": "4 panels, 30 cm wide x 24 cm axial", "P4_T30A30": "4 panels, 30 cm wide x 30 cm axial",
         "P2_T30A24": "2 panels, 30 x 24 cm, front and back", "P4_ANTO": "4 panels, front, back, 2 lateral (30 x 24 cm)"}
SHORT = {"R280_slot": "Ring\n290°", "R350_closed": "Closed\n70 cm", "P2_AP40": "2P AP\n40", "P2_AP40_d10": "2P AP\n40, 10 mm",
         "P2_LAO40": "2P LAO\n40", "P2_AP67": "2P AP\n67", "P4_eq": "4P\n34",
         "P2_T20A30": "2P\n20x30", "P2_T30A20": "2P\n30x20", "P4_T20A30": "4P\n20x30", "P4_T30A20": "4P\n30x20",
         "P4_T30A24": "4P\n30x24", "P4_T30A30": "4P\n30x30", "P2_T30A24": "2P\n30x24", "P4_ANTO": "4P F/B/L\n30x24"}
COLS = {"R280_slot": "#2a78d6", "R350_closed": "#4a3aa7", "P2_AP40": "#eb6834", "P2_AP40_d10": "#f2a541",
        "P2_LAO40": "#e34948", "P2_AP67": "#1baf7a", "P4_eq": "#0f8b8d",
        "P2_T20A30": "#b5571f", "P2_T30A20": "#d98a3d", "P4_T20A30": "#5aa6a0", "P4_T30A20": "#0b5f61",
        "P4_T30A24": "#1f8a5c", "P4_T30A30": "#6a3d9a", "P2_T30A24": "#eb6834", "P4_ANTO": "#c43c8b"}


class Panels:
    """Flat LYSO panels with the same interface as geometry.Ring (detect, snap, R for the recon grid)."""

    def __init__(self, name):
        cfg = PANEL_CONFIGS[name]
        self.name = name
        p = np.array(cfg["panels"], float)
        th = np.deg2rad(p[:, 0])
        self.theta = p[:, 0]
        self.n = np.stack([np.cos(th), np.sin(th)], 1)          # outward normal
        self.t = np.stack([-np.sin(th), np.cos(th)], 1)         # transaxial direction along the face
        self.d = p[:, 1]
        self.nt = p[:, 2].astype(int)
        self.t0 = p[:, 3]
        self.hw = self.nt * G.PITCH / 2.0
        self.depth = np.full(len(p), float(cfg["depth"]))
        self.na = int(cfg.get("na", G.NA))                      # axial crystals
        self.zmax = self.na * G.PITCH / 2.0
        self.R = 350.0                                          # recon grid as for the 70 cm ring (covers arms)

    def n_crystals(self):
        return int(self.nt.sum() * self.na)

    def crystal_volume(self):
        return float((self.nt * self.depth).sum() * self.na * G.PITCH ** 2)

    def inside(self, e, margin):
        return ((e[:, :2] @ self.n.T) <= self.d - margin).all(1)

    def detect(self, e, d, rng):
        en = e[:, :2] @ self.n.T
        dn = d[:, :2] @ self.n.T
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            s = (self.d - en) / dn
            tt = e[:, :2] @ self.t.T + s * (d[:, :2] @ self.t.T) - self.t0
            zz = e[:, 2:3] + s * d[:, 2:3]
            hit = (dn > 1e-9) & (en < self.d) & (np.abs(tt) <= self.hw) & (np.abs(zz) <= self.zmax)
        s = np.where(hit, s, np.inf)
        k = np.argmin(s, axis=1)
        s0 = s[np.arange(len(e)), k]
        ok = np.isfinite(s0)
        s0 = np.where(ok, s0, 0.0)
        ell = rng.exponential(1.0 / C.MU_LYSO_PER_MM, size=len(e))
        q = e + (s0 + ell)[:, None] * d
        radial = (q[:, :2] * self.n[k]).sum(1)
        tang = (q[:, :2] * self.t[k]).sum(1) - self.t0[k]
        ok &= (radial <= self.d[k] + self.depth[k]) & (np.abs(tang) <= self.hw[k]) & (np.abs(q[:, 2]) <= self.zmax)
        return q, ok

    def snap(self, q):
        """Snap interaction points to crystal centres at mid depth (no depth of interaction)."""
        radial = q[:, :2] @ self.n.T
        tang = q[:, :2] @ self.t.T - self.t0
        inside = (radial >= self.d - 1e-3) & (radial <= self.d + self.depth + 1e-3) & (np.abs(tang) <= self.hw + 1e-3)
        k = np.argmax(inside, axis=1)
        i = np.arange(len(q))
        nt = self.nt[k]
        it = np.clip(np.floor(tang[i, k] / G.PITCH + nt / 2.0), 0, nt - 1)
        iz = np.clip(np.floor(q[:, 2] / G.PITCH + self.na / 2.0), 0, self.na - 1)
        tc = (it - (nt - 1) / 2.0) * G.PITCH + self.t0[k]
        rc = self.d[k] + self.depth[k] / 2.0
        out = np.empty(q.shape, np.float32)
        out[:, :2] = rc[:, None] * self.n[k] + tc[:, None] * self.t[k]
        out[:, 2] = (iz - (self.na - 1) / 2.0) * G.PITCH
        return out, inside.any(1)

    def chord(self, p1, p2, rc, margin):
        """Length of each LOR segment inside the uniform sensitivity source (cylinder, axial FOV, panel half-spaces)."""
        d = p2 - p1
        D = np.linalg.norm(d, axis=1)
        a = d[:, 0] ** 2 + d[:, 1] ** 2
        b = 2 * (p1[:, 0] * d[:, 0] + p1[:, 1] * d[:, 1])
        c = p1[:, 0] ** 2 + p1[:, 1] ** 2 - rc * rc
        disc = b * b - 4 * a * c
        sq = np.sqrt(np.maximum(disc, 0))
        lo = (-b - sq) / (2 * np.maximum(a, 1e-12))
        hi = (-b + sq) / (2 * np.maximum(a, 1e-12))
        hi = np.where(disc > 0, hi, -np.inf)
        with np.errstate(divide="ignore", invalid="ignore"):
            z0 = np.where(np.abs(d[:, 2]) > 1e-9, (-self.zmax - p1[:, 2]) / d[:, 2], -np.inf)
            z1 = np.where(np.abs(d[:, 2]) > 1e-9, (self.zmax - p1[:, 2]) / d[:, 2], np.inf)
        lo = np.maximum.reduce([lo, np.minimum(z0, z1), np.zeros_like(lo)])
        hi = np.minimum.reduce([hi, np.maximum(z0, z1), np.ones_like(hi)])
        for k in range(len(self.d)):
            aa = p1[:, :2] @ self.n[k]
            bb = d[:, :2] @ self.n[k]
            cc = self.d[k] - margin
            with np.errstate(divide="ignore", invalid="ignore"):
                lim = (cc - aa) / bb
            hi = np.where(bb > 1e-12, np.minimum(hi, lim), hi)
            lo = np.where(bb < -1e-12, np.maximum(lo, lim), lo)
            hi = np.where((np.abs(bb) <= 1e-12) & (aa > cc), -np.inf, hi)
        return np.maximum(hi - lo, 0) * D

    def polys(self):
        out = []
        for k in range(len(self.d)):
            n, t = self.n[k], self.t[k]
            corners = [(self.d[k], self.t0[k] - self.hw[k]), (self.d[k], self.t0[k] + self.hw[k]),
                       (self.d[k] + self.depth[k], self.t0[k] + self.hw[k]), (self.d[k] + self.depth[k], self.t0[k] - self.hw[k])]
            out.append(np.array([a * n + b * t for a, b in corners]))
        return out


def grid_for(scn, v=3.0):
    """Recon grid; panels longer than the ring axially get a longer grid."""
    g = G.recon_grid(scn, v)
    zmax = getattr(scn, "zmax", G.ZMAX)
    return G.Grid(g.nx, g.ny, max(g.nz, 2 * int(np.ceil(zmax / v))), v) if zmax > G.ZMAX else g


def scanner_for(name):
    return Panels(name) if name in PANEL_CONFIGS else G.Ring(name)


def crystal_volume(scn):
    if isinstance(scn, Panels):
        return scn.crystal_volume()
    return float(len(scn.kept) * G.NT * G.NA * G.DEPTH * G.PITCH ** 2)


def face_area_cm2(scn):
    if isinstance(scn, Panels):
        return float((scn.nt * G.PITCH).sum() * scn.na * G.PITCH / 100.0)
    return float(len(scn.kept) * G.NT * G.PITCH * G.NA * G.PITCH / 100.0)


def cost_usd(scn):
    return face_area_cm2(scn) * USD_PER_CM2 + USD_FIXED


def fmt_cost(x):
    if x >= 1e6:
        return f"${x / 1e6:.1f}M"
    return f"${x / 1e3:.0f}k" if x >= 1e3 else f"${x:.0f}"


def ring_polys(ring):
    out = []
    ht, rc = G.NT * G.PITCH / 2.0, ring.R
    for a in np.deg2rad(ring.kept):
        n, t = np.array([np.cos(a), np.sin(a)]), np.array([-np.sin(a), np.cos(a)])
        out.append(np.array([rc * n - ht * t, rc * n + ht * t, (rc + G.DEPTH) * n + ht * t, (rc + G.DEPTH) * n - ht * t]))
    return out


def polys(scn):
    return scn.polys() if isinstance(scn, Panels) else ring_polys(scn)


class _ThinRng:
    def exponential(self, scale, size):
        return np.full(size, 10.0)   # mid depth


def coverage_deg(scn, centre=None, step=0.25):
    """Transaxial angular range (deg, of 180) of LORs through the heart centre (z = 0) with both ends on a detector."""
    c = HEART if centre is None else centre
    phi = np.deg2rad(np.arange(0.0, 180.0, step))
    d = np.stack([np.cos(phi), np.sin(phi), np.zeros_like(phi)], 1)
    e = np.tile([c[0], c[1], 0.0], (len(phi), 1))
    _, ok1 = scn.detect(e, d, _ThinRng())
    _, ok2 = scn.detect(e, -d, _ThinRng())
    return float((ok1 & ok2).mean() * 180.0)


def clearances(scn):
    """Smallest distance (mm) from the P90 body surface and from the arms to any detector, transaxial plane."""
    from matplotlib.path import Path as MPath
    a, b = C.BODIES[R.TEST_BODY]["outer"]
    xc, ra = SC.arm_geometry(R.TEST_BODY)
    u = np.linspace(0, 2 * np.pi, 1441)
    torso = np.stack([a * np.cos(u), b * np.sin(u)], 1)
    arms = np.concatenate([np.stack([s * xc + ra * np.cos(u), ra * np.sin(u)], 1) for s in (1, -1)])
    out = []
    for pts in (torso, arms):
        best = np.inf
        for P in polys(scn):
            for i in range(4):
                p, q = P[i], P[(i + 1) % 4]
                v = q - p
                t = np.clip(((pts - p) @ v) / (v @ v), 0, 1)
                best = min(best, float(np.linalg.norm(pts - (p + t[:, None] * v), axis=1).min()))
            if MPath(P).contains_points(pts).any():
                best = -1.0
        out.append(best)
    return out


def geometry_table():
    ref = crystal_volume(G.Ring(REF))
    rows = []
    for name in CONFIGS:
        scn = scanner_for(name)
        try:
            ct, ca = clearances(scn)
        except ImportError:
            ct, ca = np.nan, np.nan
        rows.append({"config": name, "label": LABEL[name], "n_crystals": int(crystal_volume(scn) / (G.PITCH ** 2 * (scn.depth[0] if isinstance(scn, Panels) else G.DEPTH))),
                     "crystal_volume_rel": crystal_volume(scn) / ref, "coverage_deg": coverage_deg(scn),
                     "clear_torso_mm": ct, "clear_arms_mm": ca})
    return rows


def print_geometry():
    print(f"{'config':12s} {'crystals':>9s} {'vol rel':>7s} {'LOR cov.':>8s} {'torso mm':>8s} {'arms mm':>8s}  layout")
    for r in geometry_table():
        scn = scanner_for(r["config"])
        lay = ("; ".join(f"{t:.1f}° at {d:.0f} mm, {n * G.PITCH / 10:.1f} cm" for t, d, n in zip(scn.theta, scn.d, scn.nt))
               if isinstance(scn, Panels) else f"ring R {scn.R:.0f} mm, {len(scn.kept)} modules")
        print(f"{r['config']:12s} {r['n_crystals']:9d} {r['crystal_volume_rel']:7.2f} {r['coverage_deg']:8.1f} "
              f"{r['clear_torso_mm']:8.1f} {r['clear_arms_mm']:8.1f}  {lay}")


# ---------------------------------------------------------------- simulation

def panel_sensitivity(scn, grid, mu, n_detected, rng, chunk=4_000_000, margin=3.0):
    """As fastmc.sensitivity, with the uniform source limited to the region in front of every panel."""
    rc = scn.R - 2.0
    sens = np.zeros(grid.shape, np.float32)
    n_have = 0
    while n_have < n_detected:
        r = rc * np.sqrt(rng.random(chunk))
        phi = rng.uniform(0, 2 * np.pi, chunk)
        e = np.stack([r * np.cos(phi), r * np.sin(phi), rng.uniform(-scn.zmax, scn.zmax, chunk)], 1)
        e = e[scn.inside(e, margin)]
        q1, q2, _ = fastmc._pairs(scn, e, rng)
        s1, _ = scn.snap(q1)
        s2, _ = scn.snap(q2)
        L = scn.chord(s1.astype(np.float64), s2.astype(np.float64), rc, margin)
        ok = L > 1.0
        s1, s2, L = s1[ok], s2[ok], L[ok]
        att = np.exp(-fwd(mu, grid, s1, s2))
        sens += bwd((att / L).astype(np.float32), grid, s1, s2)
        n_have += len(s1)
    return sens


def _gen_trues(name, ph, mu, n, seed):
    scn = scanner_for(name)
    grid = grid_for(scn)
    return fastmc.generate_trues(scn, grid, ph, mu, int(n), np.random.default_rng(seed), chunk=CHUNK, log=lambda m: None)


def _gen_emit(name, ph, mu, n_emit, seed):
    """Trues from exactly n_emit decays (equal injected activity and scan time across geometries)."""
    scn = scanner_for(name)
    grid = grid_for(scn)
    rng = np.random.default_rng(seed)
    out = {"p1": [], "p2": [], "dt": [], "e": []}
    done = 0
    while done < n_emit:
        m = min(CHUNK, n_emit - done)
        e = fastmc.sample_emissions(ph, m, rng)
        done += m
        q1, q2, e = fastmc._pairs(scn, e, rng)
        att = np.exp(-fwd(mu, grid, q1.astype(np.float32), q2.astype(np.float32)))
        keep = rng.random(len(att)) < att
        q1, q2, e = q1[keep], q2[keep], e[keep]
        s1, _ = scn.snap(q1)
        s2, _ = scn.snap(q2)
        dt = (np.linalg.norm(q1 - e, axis=1) - np.linalg.norm(q2 - e, axis=1)) / C.C_MM_PER_PS
        for k, v in (("p1", s1), ("p2", s2), ("dt", dt.astype(np.float32)), ("e", e.astype(np.float32))):
            out[k].append(v)
    res = {k: np.concatenate(v) for k, v in out.items()}
    res["n_emitted"] = np.int64(n_emit)
    return res


def _run(job):
    kind, args = job
    return (_gen_trues if kind == "trues" else _gen_emit)(*args)


def generate_parallel(jobs):
    ncpu = int(os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count() or 1))
    workers = max(1, min(len(jobs), ncpu))
    os.environ["NUMBA_NUM_THREADS"] = str(max(1, ncpu // workers))
    with ProcessPoolExecutor(workers, mp_context=mp.get_context("spawn")) as ex:
        return list(ex.map(_run, jobs))


def count_myo(ev, ph):
    v = C.VOXEL_MM
    e = ev["e"]
    idx = [np.round((e[:, i] - ph[a][0]) / v).astype(int) for i, a in enumerate("xyz")]
    inb = np.ones(len(e), bool)
    for i, a in zip(idx, "xyz"):
        inb &= (i >= 0) & (i < len(ph[a]))
    lab = ph["labels"][idx[2][inb], idx[1][inb], idx[0][inb]]
    return int((lab == C.LABELS["myocardium"]).sum())


def nec_factor(path, name):
    if path is None or not Path(path).exists():
        return None
    f = json.loads(Path(path).read_text())["factors"].get(name)
    return None if f is None else float(f[NEC_STATE])


def run_task(recon_dir, out, name, pi, ctrs, iters, subsets, n_cal, sens_events, seed, frac=1.0, nec_path=None):
    pose, shift, r = R.POSES[pi]
    scn = scanner_for(name)
    grid = grid_for(scn)
    ref_ev = recon_dir / f"events_fastmc_{REF}_{R.TEST_BODY}_{int(C.FASTMC_TRUES)}.npz"
    n_emit = int(int(np.load(ref_ev)["n_emitted"]) * frac)
    n_cal = int(n_cal * frac)
    log(f"[{name}] pose {pose}: grid {grid.shape}, {n_emit:,} decays, crystal volume "
        f"{crystal_volume(scn) / crystal_volume(G.Ring(REF)):.2f} x open ring")

    R.set_pose(shift, r)
    ph_def = R.phantom(R.TEST_BODY, defect=True)
    ph_uni = R.phantom(R.TEST_BODY, defect=False)
    mu = G.mu_map(grid, ph_def)
    if isinstance(scn, Panels):
        sens_path = recon_dir / f"sens_{name}_{R.TEST_BODY}_{R2.slug(pose)}_v8.npy"
    elif pi == 0:
        sens_path = recon_dir / f"sens_{name}_{R.TEST_BODY}_v3.npy"
    else:
        sens_path = recon_dir / f"sens_{name}_{R.TEST_BODY}_{R2.slug(pose)}_v3.npy"
    if sens_path.exists() and frac == 1.0:
        sens = np.load(sens_path)
        log(f"  sensitivity loaded: {sens_path.name}")
    else:
        t0 = time.time()
        rng = np.random.default_rng(seed + 1000 + pi)
        n_s = int(sens_events * frac)
        if isinstance(scn, Panels):
            sens = panel_sensitivity(scn, grid, mu, n_s, rng)
        else:
            sens = fastmc.sensitivity(scn, grid, mu, n_s, rng, log=lambda m: None)
        if frac == 1.0:
            np.save(sens_path, sens)
        log(f"  sensitivity computed in {time.time() - t0:.0f} s: {sens_path.name}")

    t = R.truth(grid, ph_def)
    _, prof_t = E.mid_profile(t["act"], t)
    t_uni = R.truth(grid, ph_uni)

    k = max(1, int(np.ceil(n_cal / CHUNK)))
    jobs = [("trues", (name, ph_uni, mu, min(CHUNK, n_cal - i * CHUNK), seed + 500 + pi + 1000 * i)) for i in range(k)]
    parts = np.diff(np.linspace(0, n_emit, N_DEF_SPLIT + 1).astype(np.int64))
    jobs += [("emit", (name, ph_def, mu, int(m), seed + 100 * pi + 7919 * j)) for j, m in enumerate(parts)]
    t0 = time.time()
    res = generate_parallel(jobs)
    ev_uni, ev = R2.merge(res[:k]), R2.merge(res[k:])
    del res
    n_tr, n_myo = len(ev["p1"]), count_myo(ev, ph_def)
    log(f"  {len(ev_uni['p1']):,} uniform-LV trues; defect data {n_tr:,} trues ({n_myo:,} myocardial) "
        f"from {n_emit:,} decays, {time.time() - t0:.0f} s")
    matched = {}
    for ctr in ctrs:
        _, p, s = R.measure(R.recon(ev_uni, grid, sens, ctr, iters, subsets, seed + 50 + pi), t_uni)
        matched[ctr] = (p, s)
    del ev_uni

    rows, shots = [], {}
    base = {"config": name, "body": R.TEST_BODY, "pose": R.POSES[pi][0], "iters": iters, "subsets": subsets,
            "n_emitted": n_emit, "n_trues": n_tr, "n_myo_trues": n_myo, "n_cal": n_cal,
            "crystal_volume_mm3": crystal_volume(scn), "coverage_deg": coverage_deg(scn),
            "face_area_cm2": face_area_cm2(scn), "cost_usd": cost_usd(scn)}
    fnec = nec_factor(nec_path, name)
    data = {"trues": ev}
    if fnec is not None:
        keep = np.random.default_rng(seed + 333 + pi).random(n_tr) < fnec
        data["nec"] = {k: ev[k][keep] for k in ("p1", "p2", "dt", "e")}
        log(f"  NEC-equivalent data: factor {fnec:.3f} ({NEC_STATE}), {int(keep.sum()):,} events")
    else:
        log(f"  no NEC factor for {name} in {nec_path}: trues only")
    base["nec_factor"] = fnec if fnec is not None else float("nan")
    for ctr in ctrs:
        for dkey, dev in data.items():
            img, prof, seg = R.measure(R.recon(dev, grid, sens, ctr, iters, subsets, seed + pi), t)
            for key, cal in (("none", None), ("pose", matched[ctr])):
                pp, ss = (prof, seg) if cal is None else R2.correct(prof, seg, cal)
                rows.append({**base, "data": dkey, "ctr_ps": ctr, "correction": key, **R.metrics(pp, ss, prof_t)})
            shots[f"img_{ctr}" if dkey == "trues" else f"img_{dkey}_{ctr}"] = E.reslice(img, grid, "sa")
        shots[f"true_{ctr}"] = E.reslice(t["act"], grid, "sa")
    R.set_pose((0, 0, 0), None)
    fname = f"task_{name}_{R2.slug(pose)}"
    (out / f"{fname}.json").write_text(json.dumps(rows, indent=2))
    np.savez_compressed(out / f"{fname}.npz", **shots)
    for row in rows:
        log(f"  {row['data']:5s} CTR {row['ctr_ps']} {row['correction']:5s}: contrast {row['defect_contrast']:.3f} "
            f"(true {row['true_contrast']:.3f}), remote CV {row['remote_cv_pct']:.1f}%, false defect {row['false_defect']:.3f}")


def tasks():
    return [(c, p) for c in CONFIGS for p in range(len(R.POSES))]


# ---------------------------------------------------------------- figures and tables

def _mpl():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def _stats(rows, name, ctr, key, metric):
    v = np.array([r[metric] for r in rows if r["config"] == name and r["ctr_ps"] == ctr and r["correction"] == key])
    return v


def fig_layout(out, sens_rel):
    plt = _mpl()
    from matplotlib.patches import Circle, Ellipse, Polygon
    a, b = C.BODIES[R.TEST_BODY]["outer"]
    xc, ra = SC.arm_geometry(R.TEST_BODY)
    geo = {g["config"]: g for g in geometry_table()}
    ncol = 4 if len(CONFIGS) < 8 else 5
    nrow = int(np.ceil((len(CONFIGS) + 1) / ncol))        # one extra panel for the legend text
    fig, axs = plt.subplots(nrow, ncol, figsize=(2.75 * ncol, 3.1 * nrow))
    axs = axs.ravel()
    for ax in axs[len(CONFIGS):]:
        ax.axis("off")
    for i, name in enumerate(CONFIGS):
        ax = axs[i]
        ax.add_patch(Ellipse((0, 0), 2 * a, 2 * b, fc="#ece7df", ec="#9a9083", lw=0.6))
        for s in (1, -1):
            ax.add_patch(Circle((s * xc, 0), ra, fc="#ece7df", ec="#9a9083", lw=0.6))
        ax.add_patch(Circle(tuple(HEART), 30, fc="#c0392b", ec="none", alpha=0.85))
        for P in polys(scanner_for(name)):
            ax.add_patch(Polygon(P, closed=True, fc=COLS[name], ec="none"))
        ax.set_xlim(-430, 430)
        ax.set_ylim(-560, 400)
        ax.set_aspect("equal")
        ax.axis("off")
        g = geo[name]
        txt = f"crystals {g['crystal_volume_rel']:.2f}x, {fmt_cost(cost_usd(scanner_for(name)))}\nLOR coverage {g['coverage_deg']:.0f}°"
        if name in sens_rel:
            txt += f"\nmyocardial counts {sens_rel[name]:.2f}x"
        ax.text(0, -555, txt, ha="center", va="bottom", fontsize=7)
        ax.set_title(f"{'ABCDEFGHIJKLMNOP'[i]}.  {LABEL[name]}", loc="left", fontsize=8.5, fontweight="bold")
    ax = axs[-1]
    ax.axis("off")
    ax.text(0.0, 0.6, "Transaxial plane, P90 body with arms;\nheart (red) at base pose.\nAnterior up, patient left to the right.\n"
            "Ratios relative to the 56 cm ring;\nmyocardial counts at equal injected\nactivity, NEC-equivalent (randoms,\nscatter, dead time) where modeled.\nLOR coverage: transaxial angle (of 180°)\n"
            "of LORs through the heart center\nwith both ends on detectors.", fontsize=7.5, va="center", transform=ax.transAxes)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(out / f"Fig_{TAG}_layout.{ext}", dpi=300)
    plt.close(fig)


def fig_sens(rows, out):
    plt = _mpl()
    base = {r["config"]: r for r in rows if r["pose"] == R.POSES[0][0] and r["correction"] == "none"
            and r.get("data", "trues") == "trues"}
    names = [c for c in CONFIGS if c in base]
    if REF not in base:
        return {}
    ref = base[REF]
    vol = np.array([base[c]["crystal_volume_mm3"] / ref["crystal_volume_mm3"] for c in names])
    tr = np.array([base[c]["n_trues"] / ref["n_trues"] for c in names])
    my = np.array([base[c]["n_myo_trues"] / ref["n_myo_trues"] for c in names])
    fig, axs = plt.subplots(1, 2, figsize=(10, 3.4))
    x = np.arange(len(names))
    ax = axs[0]
    E.style(ax)
    for j, (v, lab, al) in enumerate(((vol, "crystal volume", 0.35), (tr, "all trues", 0.65), (my, "myocardial trues", 1.0))):
        ax.bar(x + (j - 1) * 0.27, v, width=0.26, color=[COLS[c] for c in names], alpha=al, edgecolor="white", lw=0.5, label=lab)
    ax.axhline(1.0, color=E.INK2, lw=0.8)
    ax.set_xticks(x, [SHORT[c] for c in names], fontsize=7)
    ax.set_ylabel("Relative to 56 cm ring", fontsize=8)
    ax.set_title("A.  Crystals and trues at equal injected activity", loc="left", fontsize=9, fontweight="bold")
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=E.INK2, alpha=a, label=l) for l, a in (("crystal volume", 0.35), ("all trues", 0.65),
                                                                          ("myocardial trues", 1.0))], fontsize=7, frameon=False)
    ax = axs[1]
    E.style(ax)
    ax.bar(x, my / vol, color=[COLS[c] for c in names], width=0.6)
    ax.axhline(1.0, color=E.INK2, lw=0.8)
    ax.set_xticks(x, [SHORT[c] for c in names], fontsize=7)
    ax.set_ylabel("Myocardial trues per crystal volume\n(relative to 56 cm ring)", fontsize=8)
    ax.set_title("B.  Efficiency per crystal", loc="left", fontsize=9, fontweight="bold")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(out / f"Fig_{TAG}_sens.{ext}", dpi=300)
    plt.close(fig)
    return dict(zip(names, my))


def fig_metrics(rows, out, suffix="", what="trues only"):
    plt = _mpl()
    ctrs = sorted({r["ctr_ps"] for r in rows}, reverse=True)
    names = [c for c in CONFIGS if any(r["config"] == c for r in rows)]
    tc = float(np.mean([r["true_contrast"] for r in rows]))
    panels = [("defect_contrast", "Defect contrast"), ("remote_cv_pct", "Remote wall CV (%)"),
              ("false_defect", "Largest false defect")]
    fig, axs = plt.subplots(3, len(ctrs), figsize=(0.62 * len(names) * len(ctrs) + 1.5, 8.0), squeeze=False)
    letter = iter("ABCDEFGHIJKL")
    x = np.arange(len(names))
    for i, (m, lab) in enumerate(panels):
        top = 0.0
        for j, ctr in enumerate(ctrs):
            ax = axs[i, j]
            E.style(ax)
            for off, key, al in ((-0.2, "none", 0.35), (0.2, "pose", 1.0)):
                for n, c in enumerate(names):
                    v = _stats(rows, c, ctr, key, m)
                    if not len(v):
                        continue
                    ax.bar(n + off, v.mean(), width=0.38, color=COLS[c], alpha=al, edgecolor="white", lw=0.5)
                    ax.plot(np.full(len(v), n + off), v, "o", ms=2.2, color=E.INK2, alpha=0.7, zorder=3)
                    top = max(top, float(v.max()))
            if m == "defect_contrast":
                ax.axhline(tc, color=E.INK2, lw=0.9)
            if m == "false_defect":
                ax.axhline(R.DEFECT_DEPTH, color="#e34948", lw=0.9, ls=(0, (4, 2)))
            ax.set_xticks(x, [SHORT[c] for c in names], fontsize=6.5)
            ax.set_ylabel(lab, fontsize=8)
            ax.set_title(f"{next(letter)}.  {E.CTRNAME[ctr]}", loc="left", fontweight="bold", fontsize=9)
        for j in range(len(ctrs)):
            axs[i, j].set_ylim(0, max(top * 1.12, 0.25 if m != "remote_cv_pct" else 1.0))
    from matplotlib.patches import Patch
    fig.legend(handles=[Patch(color=E.INK2, alpha=0.35, label="uncorrected"), Patch(color=E.INK2, label="pose-matched correction")],
               loc="lower center", ncol=2, frameon=False, fontsize=8)
    fig.suptitle(f"{R.TEST_BODY} body, 20% mid-anterior defect (true contrast {tc:.2f}); bars: mean over 5 heart poses, "
                 f"dots: single poses; equal injected activity, {what}", fontsize=8.5)
    fig.tight_layout(rect=(0, 0.03, 1, 0.98))
    for ext in ("png", "pdf"):
        fig.savefig(out / f"Fig_{TAG}_metrics{suffix}.{ext}", dpi=300)
    plt.close(fig)


def fig_cost(rows, out, what):
    """Performance against materials cost: RMS error and wall CV (pose-matched), myocardial counts."""
    plt = _mpl()
    ctrs = sorted({r["ctr_ps"] for r in rows})
    names = [c for c in CONFIGS if any(r["config"] == c for r in rows)]
    tc = float(np.mean([r["true_contrast"] for r in rows]))
    cost = {c: cost_usd(scanner_for(c)) / 1e6 for c in names}
    mk = dict(zip(ctrs, ("o", "s", "^", "D", "v")))
    fig, axs = plt.subplots(1, 3, figsize=(13, 4.0))
    for ax, (m, lab, letter) in zip(axs[:2], (("rms", "Defect contrast RMS error", "A"), ("remote_cv_pct", "Remote wall CV (%)", "B"))):
        E.style(ax)
        for c in names:
            for ctr in ctrs:
                v = _stats(rows, c, ctr, "pose", "defect_contrast" if m == "rms" else m)
                if not len(v):
                    continue
                y = np.sqrt(np.mean((v - tc) ** 2)) if m == "rms" else v.mean()
                ax.plot(cost[c], y, mk[ctr], color=COLS[c], ms=6, mec="white", mew=0.5)
        ax.set_xlabel("Detector materials cost ($M)", fontsize=8)
        ax.set_ylabel(lab, fontsize=8)
        ax.set_title(f"{letter}.  {lab}, pose-matched correction", loc="left", fontweight="bold", fontsize=9)
    ax = axs[2]
    E.style(ax)
    base = {r["config"]: r for r in rows if r["pose"] == R.POSES[0][0] and r["correction"] == "none"}
    for c in names:
        if c not in base:
            continue
        b = base[c]
        f = b.get("nec_factor", float("nan"))
        y = b["n_myo_trues"] * (f if np.isfinite(f) else 1.0) * WIN / 1e6
        ax.plot(cost[c], y, "o", color=COLS[c], ms=7, mec="white", mew=0.5)
        ax.annotate(SHORT[c].replace("\n", " "), (cost[c], y), fontsize=6.5, textcoords="offset points",
                    xytext=(4, -9) if c == "P4_T30A20" else (4, 2))
    ax.set_xlabel("Detector materials cost ($M)", fontsize=8)
    ax.set_ylabel(("Myocardial NEC counts (M)" if what != "trues only" else "Myocardial trues (M)")
                  + f", x{WIN} window", fontsize=8)
    ax.set_title("C.  Myocardial counts at equal activity", loc="left", fontweight="bold", fontsize=9)
    from matplotlib.lines import Line2D
    axs[0].legend(handles=[Line2D([], [], marker=mk[c], ls="none", color=E.INK2, label=E.CTRNAME[c]) for c in ctrs],
                  fontsize=7, frameon=False, loc="upper right")
    axs[1].legend(handles=[Line2D([], [], marker="o", ls="none", color=COLS[c], label=SHORT[c].replace("\n", " "))
                           for c in names], fontsize=6.5, frameon=False, ncol=2, loc="upper right")
    fig.suptitle((f"Detector materials cost = {fmt_cost(USD_PER_CM2)}/cm2 of detector face + {fmt_cost(USD_FIXED)} fixed "
                  f"(i3M estimate, 2026-10-25); {what}").replace("$", r"\$"), fontsize=8.5)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    for ext in ("png", "pdf"):
        fig.savefig(out / f"Fig_{TAG}_cost.{ext}", dpi=300)
    plt.close(fig)


def fig_images(shots, out, ctrs, pose):
    plt = _mpl()
    from scipy.ndimage import gaussian_filter
    names = [c for c in CONFIGS if (c, pose) in shots]
    if not names:
        return
    rows = [None] + names
    fig, axs = plt.subplots(len(rows), len(ctrs), figsize=(2.0 * len(ctrs) + 1.6, 1.85 * len(rows)), squeeze=False)
    for j, ctr in enumerate(ctrs):
        for i, c in enumerate(rows):
            ax = axs[i, j]
            ax.set_xticks([])
            ax.set_yticks([])
            for s in ax.spines.values():
                s.set_visible(False)
            d = shots[(names[0] if c is None else c, pose)]
            show = d[f"true_{ctr}"] if c is None else gaussian_filter(d[f"img_{ctr}"], E.DISPLAY_FWHM_MM / 2.3548)
            ax.imshow(show, cmap="gray_r", vmin=0, vmax=1.15, origin="lower")
            if i == 0:
                ax.set_title(E.CTRNAME[ctr], fontsize=8)
            if j == 0:
                ax.set_ylabel(f"{'ABCDEFGHIJKLMNOP'[i]}.  {'truth' if c is None else LABEL[c]}", fontsize=7, fontweight="bold")
    fig.suptitle(f"Mid short axis, {R.TEST_BODY}, {pose} pose, uncorrected images; 20% defect at 12 o'clock", fontsize=8.5)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(out / f"Fig_{TAG}_images.{ext}", dpi=300)
    plt.close(fig)


def summary(groups, out, sens_rel):
    rows = groups["trues"]
    tc = float(np.mean([r["true_contrast"] for r in rows]))
    geo = {g["config"]: g for g in geometry_table()}
    base = {r["config"]: r for r in rows if r["pose"] == R.POSES[0][0] and r["correction"] == "none"}
    lines = [f"Stage {TAG}: {R.TEST_BODY}, 20% mid-anterior defect (true contrast {tc:.3f}), 5 heart poses, "
             f"equal injected activity ({rows[0]['n_emitted']:,} decays), pose-matched correction from {rows[0]['n_cal']:,} trues",
             f"{'config':12s} {'vol':>5s} {'cov°':>5s} {'clrT':>5s} {'clrA':>5s} {'myoTr':>6s} {'cost':>7s} {'NEC/T':>6s}"]
    for c in CONFIGS:
        g = geo[c]
        f = base[c].get("nec_factor", float("nan")) if c in base else float("nan")
        lines.append(f"{c:12s} {g['crystal_volume_rel']:5.2f} {g['coverage_deg']:5.0f} {g['clear_torso_mm']:5.0f} "
                     f"{g['clear_arms_mm']:5.0f} {sens_rel.get(c, np.nan):6.2f} {fmt_cost(cost_usd(scanner_for(c))):>7s} {f:6.3f}")
    lines += ["vol = crystal volume / open ring; cov = LOR coverage through heart (deg of 180); clrT/clrA = clearance to "
              "P90 torso / arms (mm); myoTr = myocardial trues / open ring at equal activity; cost = detector materials "
              f"at {fmt_cost(USD_PER_CM2)}/cm2 + {fmt_cost(USD_FIXED)} fixed; NEC/T = NEC-equivalent fraction ({NEC_STATE}, 400 MBq 15O-water)"]
    for dkey, rws in groups.items():
        if not rws:
            continue
        ctrs = sorted({r["ctr_ps"] for r in rws}, reverse=True)
        lines += ["", f"data: {dkey}",
                  f"{'config':12s} {'CTR':>6s} {'corr':5s} {'contr mean':>10s} {'range':>13s} {'RMSerr':>6s} {'remCV%':>11s} {'false max':>9s}"]
        for c in CONFIGS:
            for ctr in ctrs:
                for key in ("none", "pose"):
                    v = _stats(rws, c, ctr, key, "defect_contrast")
                    if not len(v):
                        lines.append(f"{c:12s} {E.CTRNAME[ctr]:>6s} {key:5s}  MISSING")
                        continue
                    cv = _stats(rws, c, ctr, key, "remote_cv_pct")
                    fd = _stats(rws, c, ctr, key, "false_defect")
                    lines.append(f"{c:12s} {E.CTRNAME[ctr]:>6s} {key:5s} {v.mean():10.3f} {v.min():6.3f}-{v.max():6.3f} "
                                 f"{np.sqrt(np.mean((v - tc) ** 2)):6.3f} {cv.min():5.1f}-{cv.max():5.1f} {fd.max():9.3f}")
    (out / f"SUMMARY_{TAG}.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


def collect(out, ctrs):
    rows, shots = [], {}
    for name, pi in tasks():
        pose = R.POSES[pi][0]
        f = out / f"task_{name}_{R2.slug(pose)}.json"
        if not f.exists():
            log(f"missing: {f.name}")
            continue
        rows.extend(json.loads(f.read_text()))
        shots[(name, pose)] = dict(np.load(out / f"task_{name}_{R2.slug(pose)}.npz"))
    if not rows:
        raise SystemExit("no task results found")
    for r in rows:
        r.setdefault("data", "trues")
    ctrs = sorted({r["ctr_ps"] for r in rows}, reverse=True)
    (out / f"{TAG}_metrics.json").write_text(json.dumps(rows, indent=2))
    fields = list(dict.fromkeys(k for r in rows for k in r))
    with open(out / f"{TAG}_metrics.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    groups = {d: [r for r in rows if r["data"] == d] for d in ("trues", "nec")}
    sens_rel = fig_sens(groups["trues"], out)
    base = {r["config"]: r for r in groups["trues"] if r["pose"] == R.POSES[0][0] and r["correction"] == "none"}
    lay_rel = sens_rel
    if REF in base and all(np.isfinite(b.get("nec_factor", np.nan)) for b in base.values()):
        ref = base[REF]["n_myo_trues"] * base[REF]["nec_factor"]
        lay_rel = {c: b["n_myo_trues"] * b["nec_factor"] / ref for c, b in base.items()}
    fig_layout(out, lay_rel)
    fig_metrics(groups["trues"], out)
    if groups["nec"]:
        fig_metrics(groups["nec"], out, "_nec", "NEC-equivalent counts (randoms, scatter, dead time)")
        fig_cost(groups["nec"], out, "NEC-equivalent counts")
    else:
        fig_cost(groups["trues"], out, "trues only")
    poses = [p for p in [q[0] for q in R.POSES] if any(k[1] == p for k in shots)]
    fig_images(shots, out, ctrs, poses[0])
    summary(groups, out, sens_rel)


def main():
    global CONFIGS, TAG
    ap = argparse.ArgumentParser()
    ap.add_argument("--recon")
    ap.add_argument("--stage", default="8b", choices=list(STAGES))
    ap.add_argument("--task", type=int, default=None, help="array index: geometry x pose")
    ap.add_argument("--collect", action="store_true")
    ap.add_argument("--check", action="store_true", help="print the geometry table and exit")
    ap.add_argument("--smoke", action="store_true", help="2 tasks at 1%% counts, 2 iterations, then collect")
    ap.add_argument("--ntasks", action="store_true")
    ap.add_argument("--ctrs", default=None)
    ap.add_argument("--iters", type=int, default=8)
    ap.add_argument("--subsets", type=int, default=C.RECON_SUBSETS)
    ap.add_argument("--n-cal", type=float, default=50e6)
    ap.add_argument("--sens-events", type=float, default=C.SENS_DETECTED)
    ap.add_argument("--frac", type=float, default=1.0, help="fraction of decays, calibration trues and sensitivity events")
    ap.add_argument("--outdir", default=None)
    ap.add_argument("--nec", default=None, help="nec_factors.json (default <recon>/panels_cr/nec_factors.json for 8b)")
    ap.add_argument("--seed", type=int, default=31)
    args = ap.parse_args()
    st = STAGES[args.stage]
    CONFIGS, TAG = st["configs"], st["tag"]
    if args.ntasks:
        print(len(tasks()))
        return
    if args.check:
        print_geometry()
        for c in CONFIGS:
            scn = scanner_for(c)
            print(f"{c:12s} face {face_area_cm2(scn):7.0f} cm2, axial {2 * getattr(scn, 'zmax', G.ZMAX) / 10:.1f} cm, "
                  f"materials {fmt_cost(cost_usd(scn))}")
        return
    recon_dir = Path(args.recon)
    ctrs = [int(c) for c in (args.ctrs or st["ctrs"]).split(",")]
    nec = args.nec or (str(recon_dir / "panels_cr" / "nec_factors.json") if args.stage in ("8b", "8c") else None)
    if args.smoke:
        out = recon_dir / f"{st['outdir']}_smoke"
        out.mkdir(exist_ok=True)
        nec_s = args.nec or (str(recon_dir / "panels_cr_smoke" / "nec_factors.json") if args.stage in ("8b", "8c") else None)
        picks = (("P2_T20A30", 0), ("P4_T30A20", 2)) if args.stage == "8b" else (("P4_eq", 3), ("P2_LAO40", 1))
        for name, pi in picks:
            run_task(recon_dir, out, name, pi, ctrs, 2, args.subsets, args.n_cal, args.sens_events, args.seed,
                     frac=0.01, nec_path=nec_s)
        collect(out, ctrs)
        print_geometry()
        return
    out = recon_dir / (args.outdir or st["outdir"])
    out.mkdir(exist_ok=True)
    if args.collect:
        collect(out, ctrs)
    elif args.task is not None:
        name, pi = tasks()[args.task]
        run_task(recon_dir, out, name, pi, ctrs, args.iters, args.subsets, args.n_cal, args.sens_events, args.seed,
                 args.frac, nec_path=nec)
    else:
        raise SystemExit("give --task N, --collect, --smoke or --check")


if __name__ == "__main__":
    main()

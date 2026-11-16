"""
Split-ring LYSO scanner for GATE 10.

A full ring has 72 flat modules at 5 degree steps. Each module is
7 (transaxial) x 72 (axial) LYSO crystals of 3.2 x 3.2 x 20 mm, arranged as
9 axial readout blocks of 7 x 8 crystals.
The gap parameter removes modules centred within +/- gap/2 of the
lateral axis on each side (0 and 180 deg), leaving the arm slots.
"""

import numpy as np

import config as C


def module_angles(gap_deg, ring_rotation_deg=0.0):
    """Centre angles (deg) of the modules kept for a given gap per side."""
    step = C.MODULE_ANGULAR_STEP
    angles = step / 2.0 + step * np.arange(C.N_MODULES_FULL)  # 2.5, 7.5, ...
    keep = []
    for a in angles:
        d0 = min(abs(a - 0.0), abs(a - 360.0))   # distance to the +x side
        d180 = abs(a - 180.0)                    # distance to the -x side
        if min(d0, d180) < gap_deg / 2.0:
            continue
        keep.append((a + ring_rotation_deg) % 360.0)
    return np.array(keep)


def module_centre_radius():
    return C.RING_INNER_RADIUS + C.CRYSTAL_SIZE[0] / 2.0


def add_materials(sim):
    g_cm3 = __import__("opengate").g4_units.g_cm3
    db = sim.volume_manager.material_database
    db.add_material_nb_atoms(
        "LYSO_PET", ["Lu", "Y", "Si", "O"], [18, 2, 10, 50], C.LYSO_DENSITY * g_cm3
    )
    # water-equivalent lung at inflated-lung density
    db.add_material_nb_atoms("Lung_026", ["H", "O"], [2, 1], 0.26 * g_cm3)


def add_scanner(sim, gap_deg, ring_rotation_deg=0.0, name="pet"):
    """Add the split ring to the simulation. Returns the crystal volume."""
    import opengate as gate
    from opengate.geometry.utility import get_circular_repetition, get_grid_repetition

    mm = gate.g4_units.mm
    depth, cw, ca = C.CRYSTAL_SIZE
    pitch = C.CRYSTAL_PITCH
    nt, na = C.CRYSTALS_PER_MODULE_T, C.CRYSTALS_AXIAL

    # air holder for the ring
    ring = sim.add_volume("Tubs", name)
    ring.rmin = (C.RING_INNER_RADIUS - 5.0) * mm
    ring.rmax = (C.RING_INNER_RADIUS + depth + 20.0) * mm
    ring.dz = (na * pitch / 2.0 + 5.0) * mm
    ring.material = "G4_AIR"

    # modules around the ring, minus the gap sectors
    module = sim.add_volume("Box", f"{name}_module")
    module.mother = ring.name
    module.size = [depth * mm, nt * pitch * mm, na * pitch * mm]
    module.material = "G4_AIR"

    step = C.MODULE_ANGULAR_STEP
    tr_all, rot_all = get_circular_repetition(
        C.N_MODULES_FULL,
        [module_centre_radius() * mm, 0, 0],
        start_angle_deg=step / 2.0 + ring_rotation_deg,
        axis=[0, 0, 1],
    )
    keep = set(np.round(module_angles(gap_deg, ring_rotation_deg), 6))
    all_angles = (step / 2.0 + step * np.arange(C.N_MODULES_FULL) + ring_rotation_deg) % 360.0
    tr, rot = [], []
    for a, t, r in zip(all_angles, tr_all, rot_all):
        if round(a, 6) in keep:
            tr.append(t)
            rot.append(r)
    module.translation = tr
    module.rotation = rot

    # readout blocks along the axis inside each module (energy is summed per block)
    nb = C.CRYSTALS_PER_BLOCK_AXIAL
    block = sim.add_volume("Box", f"{name}_block")
    block.mother = module.name
    block.size = [depth * mm, nt * pitch * mm, nb * pitch * mm]
    block.material = "G4_AIR"
    block.translation = get_grid_repetition([1, 1, C.BLOCKS_AXIAL], [0, 0, nb * pitch * mm])

    # crystals inside each block (x = radial)
    crystal = sim.add_volume("Box", f"{name}_crystal")
    crystal.mother = block.name
    crystal.size = [depth * mm, cw * mm, ca * mm]
    crystal.material = "LYSO_PET"
    crystal.translation = get_grid_repetition([1, nt, nb], [0, pitch * mm, pitch * mm])

    return crystal, block, len(tr)


# ============================================================ sweep 2: any bore, slots sized for the arms

def arm_geometry(body):
    """Arm centre x (mm, positive side) and radius for a body size."""
    b = C.BODIES[body]
    r = b["arm_diam"] / 2.0
    return b["outer"][0] + C.ARM_GAP_TO_TORSO + r, r


def n_modules_full(R):
    return int(2 * round(np.pi * R / C.MODULE_PITCH_AT_FACE))


def _module_clearance(R, theta_deg, px, py):
    """Distance (mm) from point (px, py) to a flat module at angle theta on a ring of face radius R."""
    t = np.deg2rad(theta_deg)
    rc = R + C.CRYSTAL_SIZE[0] / 2.0
    cx, cy = rc * np.cos(t), rc * np.sin(t)
    du = (px - cx) * np.cos(t) + (py - cy) * np.sin(t)
    dt = -(px - cx) * np.sin(t) + (py - cy) * np.cos(t)
    hu, ht = C.CRYSTAL_SIZE[0] / 2.0, C.CRYSTALS_PER_MODULE_T * C.CRYSTAL_PITCH / 2.0
    return np.hypot(max(abs(du) - hu, 0.0), max(abs(dt) - ht, 0.0))


def module_layout(R, slot):
    """Module centre angles (deg) kept for a ring of face radius R.
    With slot=True, every module closer than ARM_CLEARANCE to the design-body arm is removed;
    of the two possible module phasings, the one keeping more modules is used."""
    n = n_modules_full(R)
    step = 360.0 / n
    xc, r = arm_geometry(C.DESIGN_BODY)
    best = None
    for offset in (0.0, step / 2.0):
        angles = (offset + step * np.arange(n)) % 360.0
        if slot:
            keep = [a for a in angles
                    if min(_module_clearance(R, a, xc, 0.0), _module_clearance(R, a, -xc, 0.0))
                    >= r + C.ARM_CLEARANCE]
        else:
            keep = list(angles)
        if best is None or len(keep) > len(best):
            best = keep
    return np.array(sorted(best)), n


def slot_chord(R, kept):
    """Clear opening (mm) at the crystal face between the modules bordering the +x slot."""
    k = np.asarray(kept)
    if len(k) == n_modules_full(R):
        return 0.0
    signed = ((k + 180.0) % 360.0) - 180.0
    a1 = np.deg2rad(signed[signed > 0].min())
    a2 = np.deg2rad(signed[signed < 0].max())
    ht = C.CRYSTALS_PER_MODULE_T * C.CRYSTAL_PITCH / 2.0
    p1 = np.array([R * np.cos(a1) + ht * np.sin(a1), R * np.sin(a1) - ht * np.cos(a1)])
    p2 = np.array([R * np.cos(a2) - ht * np.sin(a2), R * np.sin(a2) + ht * np.cos(a2)])
    return float(np.linalg.norm(p1 - p2))


def add_scanner_R(sim, R, slot, name="pet"):
    """Ring of face radius R (modules placed directly in the world so arms can pass through slots)."""
    import opengate as gate
    from opengate.geometry.utility import get_grid_repetition
    from scipy.spatial.transform import Rotation

    mm = gate.g4_units.mm
    depth, cw, ca = C.CRYSTAL_SIZE
    pitch = C.CRYSTAL_PITCH
    nt, nb = C.CRYSTALS_PER_MODULE_T, C.CRYSTALS_PER_BLOCK_AXIAL
    kept, n_full = module_layout(R, slot)

    module = sim.add_volume("Box", f"{name}_module")
    module.mother = "world"
    module.size = [depth * mm, nt * pitch * mm, C.CRYSTALS_AXIAL * pitch * mm]
    module.material = "G4_AIR"
    rc = R + depth / 2.0
    tr, rot = [], []
    for a in kept:
        rmat = Rotation.from_euler("z", a, degrees=True)
        tr.append(list(rmat.apply([rc * mm, 0, 0])))
        rot.append(rmat.as_matrix())
    module.translation = tr
    module.rotation = rot

    block = sim.add_volume("Box", f"{name}_block")
    block.mother = module.name
    block.size = [depth * mm, nt * pitch * mm, nb * pitch * mm]
    block.material = "G4_AIR"
    block.translation = get_grid_repetition([1, 1, C.BLOCKS_AXIAL], [0, 0, nb * pitch * mm])

    crystal = sim.add_volume("Box", f"{name}_crystal")
    crystal.mother = block.name
    crystal.size = [depth * mm, cw * mm, ca * mm]
    crystal.material = "LYSO_PET"
    crystal.translation = get_grid_repetition([1, nt, nb], [0, pitch * mm, pitch * mm])

    return crystal, block, len(kept), n_full, kept

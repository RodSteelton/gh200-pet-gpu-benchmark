"""
Run one GATE simulation chunk for one ring configuration, then reduce the
singles to coincidences and save them as a compact .npz file.

Examples:
    python run_sim.py --test                      # quick check, ~1 min on a login node
    python run_sim.py --gap 30 --n 5000000 --seed 3 --out /scratch/$USER/exercise_pet/gap30

Source: back-to-back 511 keV photon pairs (0.5 deg FWHM acolinearity) sampled from
the phantom activity map. Positron range is not simulated here; it is scanner
independent and is added as a blur at reconstruction.
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np

import config as C
import scanner

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"


def build_simulation(gap_deg, n_events, seed, outdir, tag):
    import opengate as gate

    m = gate.g4_units.m

    sim = gate.Simulation()
    sim.random_seed = int(seed)
    sim.number_of_threads = 1
    sim.visu = False
    sim.check_volumes_overlap = False
    sim.output_dir = str(outdir)
    sim.progress_bar = False

    sim.world.size = [1.4 * m, 1.4 * m, 1.4 * m]
    sim.world.material = "G4_AIR"

    scanner.add_materials(sim)
    crystal, block, n_modules = scanner.add_scanner(sim, gap_deg)

    # voxelized phantom (labels -> materials)
    ph = sim.add_volume("Image", "phantom")
    ph.image = str(DATA / "phantom_labels.mhd")
    ph.material = "G4_AIR"
    L = C.LABELS
    ph.voxel_materials = [
        [L["air"] - 0.5, L["air"] + 0.5, "G4_AIR"],
        [L["soft_tissue"] - 0.5, L["soft_tissue"] + 0.5, "G4_TISSUE_SOFT_ICRP"],
        [L["lung"] - 0.5, L["lung"] + 0.5, "Lung_026"],
        [L["bone"] - 0.5, L["bone"] + 0.5, "G4_B-100_BONE"],
        [L["myocardium"] - 0.5, L["myocardium"] + 0.5, "G4_TISSUE_SOFT_ICRP"],
        [L["blood_pool"] - 0.5, L["blood_pool"] + 0.5, "G4_BLOOD_ICRP"],
        [L["liver"] - 0.5, L["liver"] + 0.5, "G4_TISSUE_SOFT_ICRP"],
    ]

    # source: back-to-back photons from the activity map
    src = sim.add_source("VoxelSource", "src")
    src.attached_to = ph.name
    src.image = str(DATA / "phantom_activity.mhd")
    src.position.translation = [0, 0, 0]
    src.particle = "back_to_back"
    src.direction.type = "iso"
    src.direction.accolinearity_flag = True
    src.n = int(n_events)

    # physics: EM only, large production cuts (local deposition), fast and adequate at 511 keV
    sim.physics_manager.physics_list_name = "G4EmStandardPhysics_option3"
    sim.physics_manager.set_production_cut("world", "all", 1 * m)

    # digitizer: hits in crystals -> singles per readout block
    # (energy summed over the block, position of the highest-energy hit)
    hc = sim.add_actor("DigitizerHitsCollectionActor", "Hits")
    hc.attached_to = crystal.name
    hc.authorize_repeated_volumes = True
    hc.attributes = [
        "EventID",
        "PostPosition",
        "TotalEnergyDeposit",
        "PreStepUniqueVolumeID",
        "GlobalTime",
        "EventPosition",
    ]
    hc.root_output.write_to_disk = False

    sc = sim.add_actor("DigitizerAdderActor", "Singles")
    sc.attached_to = crystal.name
    sc.authorize_repeated_volumes = True
    sc.input_digi_collection = hc.name
    sc.group_volume = block.name
    sc.policy = "EnergyWinnerPosition"
    sc.output_filename = f"singles_{tag}.root"

    stats = sim.add_actor("SimulationStatisticsActor", "Stats")
    stats.output_filename = f"stats_{tag}.txt"

    return sim, n_modules


def add_digitizer(sim, crystal, block, tag):
    hc = sim.add_actor("DigitizerHitsCollectionActor", "Hits")
    hc.attached_to = crystal.name
    hc.authorize_repeated_volumes = True
    hc.attributes = ["EventID", "PostPosition", "TotalEnergyDeposit", "PreStepUniqueVolumeID",
                     "GlobalTime", "EventPosition"]
    hc.root_output.write_to_disk = False
    sc = sim.add_actor("DigitizerAdderActor", "Singles")
    sc.attached_to = crystal.name
    sc.authorize_repeated_volumes = True
    sc.input_digi_collection = hc.name
    sc.group_volume = block.name
    sc.policy = "EnergyWinnerPosition"
    sc.output_filename = f"singles_{tag}.root"
    stats = sim.add_actor("SimulationStatisticsActor", "Stats")
    stats.output_filename = f"stats_{tag}.txt"


def check_clearances(R, slot, info):
    """Refuse geometries where the phantom box or arms would overlap detector modules."""
    kept, _ = scanner.module_layout(R, slot)
    bx, by = float(info["box_half"][0]), float(info["box_half"][1])
    corner = np.hypot(bx, by)
    if corner >= R:
        raise SystemExit(f"Phantom box corner at r={corner:.1f} mm reaches the ring (R={R} mm).")
    xc, r = float(info["arm_x"]), float(info["arm_r"])
    if xc - r <= bx:
        raise SystemExit("Arms overlap the phantom box.")
    dmin = min(min(scanner._module_clearance(R, a, xc, 0.0), scanner._module_clearance(R, a, -xc, 0.0))
               for a in kept)
    if dmin <= r:
        raise SystemExit(f"Arms overlap detector modules (clearance {dmin - r:.1f} mm).")
    return dmin - r


def build_simulation2(config_name, body, n_events, seed, outdir, tag):
    """Sweep 2: any bore, slots sized for the design arm, body-size phantom plus arm cylinders."""
    import opengate as gate

    cfg = {c[0]: c for c in C.SWEEP2_CONFIGS}[config_name]
    _, R, slot = cfg
    info = dict(np.load(DATA / f"phantom_{body}.npz"))
    arm_clear = check_clearances(R, slot, info)

    u = gate.g4_units
    mm, m = u.mm, u.m
    sim = gate.Simulation()
    sim.random_seed = int(seed)
    sim.number_of_threads = 1
    sim.visu = False
    sim.check_volumes_overlap = False
    sim.output_dir = str(outdir)
    sim.progress_bar = False
    sim.world.size = [1.6 * m, 1.6 * m, 1.6 * m]
    sim.world.material = "G4_AIR"

    scanner.add_materials(sim)
    crystal, block, n_modules, n_full, kept = scanner.add_scanner_R(sim, R, slot)

    L = C.LABELS
    ph = sim.add_volume("Image", "phantom")
    ph.image = str(DATA / f"phantom_{body}_labels.mhd")
    ph.material = "G4_AIR"
    mats = {"air": "G4_AIR", "soft_tissue": "G4_TISSUE_SOFT_ICRP", "lung": "Lung_026",
            "bone": "G4_B-100_BONE", "myocardium": "G4_TISSUE_SOFT_ICRP",
            "blood_pool": "G4_BLOOD_ICRP", "liver": "G4_TISSUE_SOFT_ICRP",
            "adipose": "G4_ADIPOSE_TISSUE_ICRP"}
    ph.voxel_materials = [[L[k] - 0.5, L[k] + 0.5, mats[k]] for k in L]

    # arms: soft-tissue cylinders with a humerus, running the full phantom length
    xc, r = float(info["arm_x"]), float(info["arm_r"])
    a_t, a_a = float(info["a_torso"]), float(info["a_arm_each"])
    n_torso = int(round(n_events * a_t / (a_t + 2 * a_a)))
    n_arm = (int(n_events) - n_torso) // 2
    n_torso = int(n_events) - 2 * n_arm
    for side, sgn in (("L", 1), ("R", -1)):
        arm = sim.add_volume("Tubs", f"arm_{side}")
        arm.rmin = 0
        arm.rmax = r * mm
        arm.dz = C.ARM_HALF_LENGTH * mm
        arm.translation = [sgn * xc * mm, 0, 0]
        arm.material = "G4_TISSUE_SOFT_ICRP"
        hum = sim.add_volume("Tubs", f"humerus_{side}")
        hum.mother = arm.name
        hum.rmin = 0
        hum.rmax = C.HUMERUS_RADIUS * mm
        hum.dz = C.ARM_HALF_LENGTH * mm
        hum.material = "G4_B-100_BONE"
        s = sim.add_source("GenericSource", f"src_arm_{side}")
        s.attached_to = arm.name
        s.particle = "back_to_back"
        s.position.type = "box"
        s.position.size = [2 * r * mm, 2 * r * mm, 2 * C.ARM_HALF_LENGTH * mm]
        s.position.confine = arm.name
        s.direction.type = "iso"
        s.direction.accolinearity_flag = True
        s.n = n_arm

    src = sim.add_source("VoxelSource", "src")
    src.attached_to = ph.name
    src.image = str(DATA / f"phantom_{body}_activity.mhd")
    src.position.translation = [0, 0, 0]
    src.particle = "back_to_back"
    src.direction.type = "iso"
    src.direction.accolinearity_flag = True
    src.n = n_torso

    sim.physics_manager.physics_list_name = "G4EmStandardPhysics_option3"
    sim.physics_manager.set_production_cut("world", "all", 1 * m)
    add_digitizer(sim, crystal, block, tag)

    geo = {"config": config_name, "body": body, "R_mm": R, "slot": slot, "n_modules": n_modules,
           "n_modules_full": n_full, "coverage_deg": 360.0 * n_modules / n_full,
           "slot_chord_mm": scanner.slot_chord(R, kept), "arm_clearance_mm": arm_clear,
           "frac_myo": float(info["frac_myo"]), "n_torso": n_torso, "n_arm_each": n_arm}
    return sim, geo


def singles_to_coincidences(root_path, seed):
    """Energy blur + window, then keep events with exactly two in-window singles."""
    import uproot

    rng = np.random.default_rng(int(seed) + 7919)
    if not Path(root_path).exists():
        return None
    f = uproot.open(root_path)
    if "Singles" not in f:
        return None
    t = f["Singles"]
    b = t.arrays(
        [
            "EventID", "TotalEnergyDeposit", "GlobalTime",
            "PostPosition_X", "PostPosition_Y", "PostPosition_Z",
            "EventPosition_X", "EventPosition_Y", "EventPosition_Z",
        ],
        library="np",
    )
    if len(b["EventID"]) == 0:
        return None

    e = b["TotalEnergyDeposit"] * 1000.0  # MeV -> keV
    sigma = C.ENERGY_RES_FWHM / 2.3548 * np.sqrt(511.0 * np.maximum(e, 0))
    e_blur = e + rng.normal(size=e.shape) * sigma
    ok = (e_blur >= C.ENERGY_WINDOW[0]) & (e_blur <= C.ENERGY_WINDOW[1])

    ev = b["EventID"][ok]
    order = np.argsort(ev, kind="stable")
    idx = np.flatnonzero(ok)[order]
    ev = ev[order]
    uniq, start, count = np.unique(ev, return_index=True, return_counts=True)
    pair = count == 2
    i1 = idx[start[pair]]
    i2 = idx[start[pair] + 1]

    out = {
        "x1": b["PostPosition_X"][i1], "y1": b["PostPosition_Y"][i1], "z1": b["PostPosition_Z"][i1],
        "x2": b["PostPosition_X"][i2], "y2": b["PostPosition_Y"][i2], "z2": b["PostPosition_Z"][i2],
        "e1": e_blur[i1], "e2": e_blur[i2],
        "t1": b["GlobalTime"][i1], "t2": b["GlobalTime"][i2],       # ns, true times
        "ex": b["EventPosition_X"][i1], "ey": b["EventPosition_Y"][i1], "ez": b["EventPosition_Z"][i1],
    }
    for k in out:
        out[k] = out[k].astype(np.float64 if k in ("t1", "t2") else np.float32)
    counts = {
        "n_singles": int(len(e)),
        "n_singles_in_window": int(ok.sum()),
        "n_events_with_2": int(pair.sum()),
        "n_events_multiple": int((count > 2).sum()),
    }
    return out, counts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gap", type=float, default=30.0, help="gap per side, degrees")
    ap.add_argument("--n", type=float, default=C.EVENTS_PER_JOB, help="number of decays")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out", type=str, default=str(HERE / "out_test"))
    ap.add_argument("--keep-root", action="store_true", help="keep the singles ROOT file")
    ap.add_argument("--test", action="store_true", help=f"quick check with {C.TEST_EVENTS} decays")
    ap.add_argument("--config", type=str, default=None,
                    help="sweep-2 geometry, one of: " + ", ".join(c[0] for c in C.SWEEP2_CONFIGS))
    ap.add_argument("--body", type=str, default="P90", help="sweep-2 body size: " + ", ".join(C.BODIES))
    args = ap.parse_args()

    if not (DATA / "phantom_activity.mhd").exists():
        raise SystemExit("Phantom not found. Run: python make_phantom.py")

    n = C.TEST_EVENTS if args.test else int(args.n)
    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)
    if args.config:
        if not (DATA / f"phantom_{args.body}.npz").exists():
            raise SystemExit("Body phantoms not found. Run: python make_phantom.py")
        tag = f"{args.config}_{args.body}_seed{args.seed:04d}"
        sim, geo = build_simulation2(args.config, args.body, n, args.seed, outdir, tag)
        n_modules = geo["n_modules"]
        print(f"[{tag}] {n_modules} modules, {geo['coverage_deg']:.0f} deg coverage, "
              f"slot {geo['slot_chord_mm'] / 10:.1f} cm, arm clearance {geo['arm_clearance_mm']:.0f} mm, "
              f"{n:,} decays")
    else:
        gap = args.gap
        tag = f"gap{int(gap):02d}_seed{args.seed:04d}"
        sim, n_modules = build_simulation(gap, n, args.seed, outdir, tag)
        geo = {"gap_deg": gap}
        print(f"[{tag}] {n_modules} modules, {n:,} decays")
    t0 = time.time()
    sim.run()
    wall = time.time() - t0

    root_path = outdir / f"singles_{tag}.root"
    res = singles_to_coincidences(root_path, args.seed)
    meta = dict(geo)
    meta.update({"seed": args.seed, "n_decays": n, "n_modules": n_modules,
                 "wall_s": wall, "decays_per_s": n / wall if wall > 0 else None})
    if res is None:
        coinc, counts = {}, {"n_singles": 0}
    else:
        coinc, counts = res
    meta.update(counts)
    meta["n_coincidences"] = int(len(coinc.get("x1", [])))
    np.savez_compressed(outdir / f"coinc_{tag}.npz", meta=json.dumps(meta), **coinc)
    if not args.keep_root and root_path.exists():
        root_path.unlink()

    print(json.dumps(meta, indent=2))
    if args.test:
        sens = 100.0 * meta["n_coincidences"] / n
        print(f"\nTEST OK: {meta['n_coincidences']} coincidences from {n:,} decays "
              f"({sens:.2f}% prompts per decay, no randoms), {meta['decays_per_s']:.0f} decays/s")


if __name__ == "__main__":
    main()

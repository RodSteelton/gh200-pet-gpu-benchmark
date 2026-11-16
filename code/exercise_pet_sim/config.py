"""
Shared parameters for the exercise cardiac PET geometry study.

Units: millimetres, keV, picoseconds unless stated otherwise.
Coordinate convention (phantom and scanner):
    +x = patient left, +y = anterior, +z = superior.
    Detector gaps (arm slots) are centred on the lateral axis (x), at 0 and 180 degrees.
"""

# ---------------------------------------------------------------- scanner
RING_INNER_RADIUS = 280.0      # mm, crystal front face (56 cm bore)
CRYSTAL_SIZE = (20.0, 3.2, 3.2)  # mm (radial depth, transaxial, axial)
CRYSTAL_PITCH = 3.3            # mm, transaxial and axial
CRYSTALS_PER_MODULE_T = 7      # transaxial crystals per module
CRYSTALS_AXIAL = 72            # axial crystals (72 x 3.3 mm = 237.6 mm axial FOV)
CRYSTALS_PER_BLOCK_AXIAL = 8   # readout block = 7 x 8 crystals; energy summed per block
BLOCKS_AXIAL = CRYSTALS_AXIAL // CRYSTALS_PER_BLOCK_AXIAL  # 9 blocks per module
MODULE_ANGULAR_STEP = 5.0      # deg; 72 modules in a full ring
N_MODULES_FULL = 72
LYSO_DENSITY = 7.1             # g/cm3

# gap per side (deg). 0 = closed ring, 30 = proposed design (300 deg coverage),
# 60 = 2017 proposal (240 deg coverage)
GAP_SWEEP = [0, 20, 30, 60]

# ---------------------------------------------------------------- detector response
ENERGY_RES_FWHM = 0.10         # fraction at 511 keV
ENERGY_WINDOW = (435.0, 585.0)  # keV
CTR_SWEEP_PS = [200, 250, 320]  # coincidence timing resolution FWHM, ps
SCATTER_DIST_MM = 5.0          # LOR-to-emission distance above which a coincidence is scatter

# ---------------------------------------------------------------- phantom
VOXEL_MM = 3.0
PHANTOM_DIMS = (130, 90, 100)  # voxels (x, y, z): 390 x 270 x 300 mm
TORSO_SEMI_AXES = (175.0, 115.0)  # mm, ellipse (x, y)
TORSO_HALF_LENGTH = 150.0      # mm

LABELS = {
    "air": 0,
    "soft_tissue": 1,
    "lung": 2,
    "bone": 3,
    "myocardium": 4,
    "blood_pool": 5,
    "liver": 6,
    "adipose": 7,
}

# left ventricle (prolate ellipsoidal shell): centre, base->apex axis, semi-axes (radial, long), base cut
LV_CENTER = (35.0, 25.0, 0.0)
LV_AXIS = (0.6, 0.5, -0.6)
LV_OUTER = (32.0, 48.0)
LV_INNER = (22.0, 38.0)
LV_BASE_CUT = -20.0            # mm along the long axis; basal cap removed below this

# relative activity concentration, roughly N-13 ammonia at stress
ACTIVITY = {
    "air": 0.0,
    "soft_tissue": 0.12,
    "lung": 0.04,
    "bone": 0.12,
    "myocardium": 1.0,
    "blood_pool": 0.15,
    "liver": 0.60,
    "adipose": 0.03,
    "arm": 0.12,
}

# ---------------------------------------------------------------- sweep 2: body size, arms, bore
# Body sizes for adults 60+ (NHANES 2015-2018 arm circumference; chest estimated from girth).
# The lean core is the sweep-1 torso (34 x 23 cm); larger bodies add an adipose layer around it.
BODIES = {
    "P50": {"outer": (185.0, 130.0), "arm_diam": 104.0},   # chest 37 x 26 cm, arm circ. ~32.4 cm
    "P90": {"outer": (220.0, 160.0), "arm_diam": 120.0},   # chest 44 x 32 cm, arm circ. ~37.3 cm
}
ARM_GAP_TO_TORSO = 5.0         # mm between torso surface and arm
HUMERUS_RADIUS = 12.0          # mm
ARM_HALF_LENGTH = 150.0        # mm, arms run the full phantom length
DESIGN_BODY = "P90"            # slots are sized so this body's arms clear the detectors
ARM_CLEARANCE = 20.0           # mm, free space around the design arm (pedalling motion)
MODULE_PITCH_AT_FACE = 24.4    # mm, sets the module count for each bore

# (name, crystal-face radius mm, slots?)
SWEEP2_CONFIGS = [
    ("R280_slot", 280.0, True),     # 56 cm bore split ring
    ("R310_slot", 310.0, True),     # 62 cm bore split ring
    ("R350_slot", 350.0, True),     # 70 cm bore split ring
    ("R350_closed", 350.0, False),  # 70 cm closed ring, arms inside the bore (clinical-size comparator)
]
SWEEP2_BODIES = ["P50", "P90"]
SWEEP2_REFERENCE = "R350_closed"
JOBS_PER_CONFIG = 20

# ---------------------------------------------------------------- production sweep
EVENTS_PER_JOB = 5_000_000
JOBS_PER_GAP = 20              # 20 x 5e6 = 1e8 decays per geometry
TEST_EVENTS = 20_000

C_MM_PER_PS = 0.299792458      # speed of light

# ---------------------------------------------------------------- reconstruction (stage 2)
RECON_VOXEL_MM = 3.0
RECON_CONFIGS = ["R280_slot", "R350_closed"]
RECON_BODIES = ["P50", "P90"]
RECON_CTRS = [0, 320, 200]     # 0 = non-TOF
RECON_ITERS = 4
RECON_SUBSETS = 10
FASTMC_TRUES = 10_000_000      # trues per (geometry, body) for the high-count fast Monte Carlo
SENS_DETECTED = 20_000_000     # detected events for the sensitivity image
MU_PER_MM = {"air": 0.0, "soft_tissue": 0.0099, "lung": 0.0025, "bone": 0.0135, "myocardium": 0.0099,
             "blood_pool": 0.0102, "liver": 0.0099, "adipose": 0.0093}
MU_LYSO_PER_MM = 0.087

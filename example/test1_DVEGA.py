"""
Test 1: DV-EGA (Earth - DSM - Earth - Jupiter).

Single deep-space-maneuver Earth Gravity Assist:
    Earth (launch) --[DSM]--> Earth (flyby) --> Jupiter (arrival)

The DSM lives on the Earth->Earth leg (leg 0) and is placed by solve_arc's
leveraging grid. lev_type = "-+" => decrease departure v_inf, increase arrival
v_inf ("leverage up"): launch slow, let the DSM + Earth flyby pump the energy up
toward Jupiter.

Sampling spec:
    Launch                 2020-2030
    Max total flight time  6 yr
    Max Earth->Earth time  3 yr (before the Jupiter leg)
    Grid step              3 days
    Output tfilter         10 days
    v_inf max              7 km/s at Earth & Jupiter, 14 km/s at flyby
    Flyby altitude         500 km
    Flyby (powered) DV     50 m/s
    DSM                    800 m/s max, 50 m/s steps, leverage up
    Reference burn alts    300 km LEO escape, 800,000 km Jupiter JOI periapsis
"""

import numpy as np
from star.constants.time import et_seconds_to_j2000_days, utc_string_to_et_seconds

# -----------------------------
# 1) Body sequence (encounters)
# -----------------------------
Body = [
    [399],   # encounter 1 (departure): Earth
    [399],   # encounter 2: Earth gravity assist (after DSM)
    [599],   # encounter 3 (arrival): Jupiter
]

nE = len(Body)  # 3 encounters
nL = nE - 1     # 2 legs

# -----------------------------------------
# 2) Encounter time windows (days from J2000)
# -----------------------------------------
t0 = et_seconds_to_j2000_days(utc_string_to_et_seconds("2020-01-01T00:00:00"))
t1 = et_seconds_to_j2000_days(utc_string_to_et_seconds("2030-01-01T00:00:00"))

MAX_TOF_DAYS = 6 * 365.25   # 6-year total mission
MAX_EE_DAYS = 3 * 365.25    # 3-year Earth->Earth before the Jupiter leg

Time = [None] * nE
Time[0] = np.array([t0, t1])                  # launch window 2020-2030
Time[-1] = np.array([t0, t1 + MAX_TOF_DAYS])  # arrival <= 6 yr after latest launch
for k in range(1, nE - 1):
    Time[k] = np.array([-np.inf, np.inf])     # Earth flyby epoch free (TOF-bounded)

# -----------------------------------------
# 3) TOF bounds for each leg (days)
# -----------------------------------------
tof_min = np.full((nE, nE), -np.inf, dtype=float)
tof_max = np.full((nE, nE),  np.inf, dtype=float)

tof_min[0, 1], tof_max[0, 1] = 20.0, MAX_EE_DAYS
tof_min[1, 2], tof_max[1, 2] = 20.0, MAX_TOF_DAYS
tof_min[0, 2], tof_max[0, 2] = 100.0, MAX_TOF_DAYS

# -----------------------------------------
# 4) Encounter grid step sizes dt (days)
# -----------------------------------------
dt = [3.0] * nE

# -----------------------------------------
# 4b) Final-output tfilter bin size (days)
# -----------------------------------------
dt_filter = 10.0

# -----------------------------------------
# 5) Flyby altitude constraint (km) per encounter
# -----------------------------------------
# alt[-1] doubles as the Jupiter JOI periapsis altitude for reference.
escape_alt_km = 300.0       # reference: LEO parking-orbit altitude for launch/escape DV
joi_alt_km = 800_000.0      # reference: Jupiter capture periapsis altitude for JOI DV
alt = [
    300,  # launch
    500.0,         # Earth gravity assist
    joi_alt_km,    # Jupiter arrival (JOI periapsis)
]

# -----------------------------------------
# 6) v-infinity bounds (km/s) per encounter
# -----------------------------------------
vlim = np.array(
    [
        [0.0, 0.0, 0.0],   # v_inf_min
        [7.0, 14.0, 7.0],  # v_inf_max: 7 Earth dep, 14 Earth flyby, 7 Jupiter
    ],
    dtype=float,
)

# Post-launch total DV cap (km/s): DSM (<=0.8) + powered flyby (<=0.05).
dVtotal = 0.85

# -----------------------------------------
# 7) Lambert arc options (per leg)
# -----------------------------------------
# Allow multi-rev seeds on the Earth->Earth resonant leg.
Nrev = [2, 1]
lambert_nrev = Nrev
lambert_hz = [1] * nL

# -----------------------------------------
# 8) Powered-flyby patch DV limits (km/s)
# -----------------------------------------
dVfb_max = [0.03] * nL   # 30 m/s

# -----------------------------------------
# 9) DSM leveraging grid for solve_arc (per leg)
# -----------------------------------------
# DSM only on the Earth->Earth leg; Jupiter leg stays Lambert-only.
dVlev_max = [0.85, 0.0]       # 850 m/s max DSM on leg 0
delta_dvlev = [0.05, 0.05]   # 50 m/s steps
lev_type = ["-+"]            # leverage up: decrease initial v_inf, increase final

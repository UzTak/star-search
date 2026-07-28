"""
Test 2: EMEJ (Earth - Mars - Earth - Jupiter).

Ballistic (gravity-assist only) Mars-Earth-Gravity-Assist to Jupiter:
    Earth (launch) --> Mars (flyby) --> Earth (flyby) --> Jupiter (arrival)

No DSM here; energy is built purely through the Mars and Earth flybys (powered
flybys allowed up to 50 m/s each).

Sampling spec:
    Launch                 2020-2030
    Max total flight time  6 yr
    Max Earth->Earth time  3 yr (Earth->Mars->Earth, before the Jupiter leg)
    Grid step              3 days
    Output tfilter         10 days
    v_inf max              7 km/s at Earth & Jupiter, 14 km/s at flybys
    Flyby altitude         500 km
    Flyby (powered) DV     50 m/s
    Reference burn alts    300 km LEO escape, 800,000 km Jupiter JOI periapsis
"""

import numpy as np
from star.constants.time import et_seconds_to_j2000_days, utc_string_to_et_seconds

# -----------------------------
# 1) Body sequence (encounters)
# -----------------------------
Body = [
    [399],   # encounter 1 (departure): Earth
    [499],   # encounter 2: Mars gravity assist
    [399],   # encounter 3: Earth gravity assist
    [599],   # encounter 4 (arrival): Jupiter
]

nE = len(Body)  # 4 encounters
nL = nE - 1     # 3 legs

# -----------------------------------------
# 2) Encounter time windows (days from J2000)
# -----------------------------------------
t0 = et_seconds_to_j2000_days(utc_string_to_et_seconds("2020-01-01T00:00:00"))
t1 = et_seconds_to_j2000_days(utc_string_to_et_seconds("2030-01-01T00:00:00"))

MAX_TOF_DAYS = 6 * 365.25   # 6-year total mission
MAX_EE_DAYS = 3 * 365.25    # 3-year Earth->Mars->Earth before the Jupiter leg

Time = [None] * nE
Time[0] = np.array([t0, t1])                  # launch window 2020-2030
Time[-1] = np.array([t0, t1 + MAX_TOF_DAYS])  # arrival <= 6 yr after latest launch
for k in range(1, nE - 1):
    Time[k] = np.array([-np.inf, np.inf])     # flyby epochs free (TOF-bounded)

# -----------------------------------------
# 3) TOF bounds for each leg (days)
# -----------------------------------------
tof_min = np.full((nE, nE), -np.inf, dtype=float)
tof_max = np.full((nE, nE),  np.inf, dtype=float)

# Consecutive legs.
tof_min[0, 1], tof_max[0, 1] = 10.0, MAX_EE_DAYS    # Earth -> Mars
tof_min[1, 2], tof_max[1, 2] = 10.0, MAX_EE_DAYS    # Mars -> Earth
tof_min[2, 3], tof_max[2, 3] = 100.0, MAX_TOF_DAYS   # Earth -> Jupiter

# Earth -> Mars -> Earth (the "Earth->Earth" portion): up to 3 yr.
tof_min[0, 2], tof_max[0, 2] = 100.0, MAX_EE_DAYS
# Total mission: up to 6 yr.
tof_min[0, 3], tof_max[0, 3] = 100.0, MAX_TOF_DAYS

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
    0.0,           # launch
    500.0,         # Mars gravity assist
    500.0,         # Earth gravity assist
    joi_alt_km,    # Jupiter arrival (JOI periapsis)
]

# -----------------------------------------
# 6) v-infinity bounds (km/s) per encounter
# -----------------------------------------
vlim = np.array(
    [
        [0.0, 0.0, 0.0, 0.0],     # v_inf_min
        [7.0, 14.0, 14.0, 7.0],   # v_inf_max: 7 Earth dep, 14 flybys, 7 Jupiter
    ],
    dtype=float,
)

# Post-launch total DV cap (km/s): two powered flybys at 50 m/s each.
dVtotal = 0.1

# -----------------------------------------
# 7) Lambert arc options (per leg)
# -----------------------------------------
Nrev = [1] * nL
lambert_nrev = Nrev
lambert_hz = [1] * nL

# -----------------------------------------
# 8) Powered-flyby patch DV limits (km/s)
# -----------------------------------------
dVfb_max = [0.05] * nL   # 50 m/s

# -----------------------------------------
# 9) DSM leveraging grid for solve_arc (per leg)
# -----------------------------------------
# Ballistic flybys only -> no DSM/leveraging on any leg.
dVlev_max = [0.0] * nL
# delta_dvlev = [0.05] * nL
# lev_type = ["+-"]

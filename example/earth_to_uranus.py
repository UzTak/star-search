# Minimal Star input file. 
# Scope: ballistic Lambert legs + null legs later. No VILT, no central-body switching.

import numpy as np
from star.constants.time import et_seconds_to_j2000_days, utc_string_to_et_seconds

# -----------------------------
# 1) Body sequence (encounters)
# -----------------------------
# Use NAIF IDs (integers) like the MATLAB version.
# Example: Earth -> {Venus/Mars/Earth} -> {Venus/Mars/Earth} -> Jupiter
Body = [
    [399],                 # encounter 1 (departure): Earth
    [299, 399, 499],       # encounter 2: Venus/Mars/Earth options
    [299, 399, 499],       # encounter 3: Venus/Mars/Earth options
    [399, 499, 599, 699],          # encounter 4 (arrival): Uranus
    [799]
]

nE = len(Body)  # Number of encounters
nL = nE - 1   # Number of legs

# -----------------------------------------
# 2) Encounter time windows (days from J2000)
# -----------------------------------------
# Use +/- inf for unspecified windows (same convention you referenced).
# Set departure epoch by UTC string and convert to days past J2000.
t0_utc = "2039-04-21T00:00:00"
t0 = et_seconds_to_j2000_days(utc_string_to_et_seconds(t0_utc))

Time = [None] * len(Body)
Time[0] = np.array([t0 + 0.0, t0 + 300.0])        # departure window
Time[-1] = np.array([t0 + 2300.0, t0 + 5000.0])    # arrival window 

# Middle encounters can be left unconstrained; bounds can be inferred/tightened later.
for k in range(1, len(Body) - 1):
    Time[k] = np.array([-np.inf, np.inf])

# -----------------------------------------
# 3) TOF bounds for each leg (days)
# -----------------------------------------
# tof is an nE x nE matrix in the paper, but for ballistic leg building you only need consecutive legs.
tof_min = np.full((nE, nE), -np.inf, dtype=float)
tof_max = np.full((nE, nE),  np.inf, dtype=float)

# Consecutive legs: (1->2), (2->3), (3->4) 
tof_min[0, 1], tof_max[0, 1] = 100.0, 300.0   # Earth -> first flyby
tof_min[1, 2], tof_max[1, 2] = 100.0, 1000.0 # flyby1 -> flyby2
tof_min[2, 3], tof_max[2, 3] = 700.0, 1200.0 # flyby2 -> Jupiter
tof_min[3, 4], tof_max[3, 4] = 700.0, 4000.0 # Jupiter -> Uranus

# total TOF constraint 
tof_min[0, 4], tof_max[0, 4] = 2500, 5000

# -----------------------------------------
# 4) Encounter grid step sizes dt (days)
# -----------------------------------------
# In MATLAB they use dt{encounter_index} and can vary by encounter.
dt = [None] * nE
dt[0] = 10                    # departure grid
for k in range(1, nE - 1):
    dt[k] = 15                    # intermediate encounter grids
dt[-1] = 100                       # arrival grid

# -----------------------------------------
# 4b) Final-output tfilter bin size (days)
# -----------------------------------------
# use one bin size for both departure and arrival dimensions.
dt_filter = 20.0

# -----------------------------------------
# 5) Flyby altitude constraint (km) per encounter
# -----------------------------------------
# Used to compute rmin = radius + alt for later flyby feasibility (not needed for ballistic legs filtering).
# Keep scalar or per-body list/dict. Here: scalar per encounter.
alt = [None] * nE
alt[0] = 100.0
for k in range(1, nE - 1):
    alt[k] = 100.0
alt[-1] = 250.0

# -----------------------------------------
# 6) v-infinity bounds (km/s) per encounter
# -----------------------------------------
# vlim is 2 x nE (min row, max row) with custom overrides possible.
# Keep this simple: per-encounter min/max, applied when building legs at departure and arrival ends.
vlim = np.array([
    [0.0, 0.0, 0.0, 0.0, 0.0],       # v_inf_min for encounters 1..nE
    [8.0, 13.0, 13.0, 20.0, 15.0],   # v_inf_max for encounters 1..nE
], dtype=float)

# -----------------------------------------
# 7) Lambert arc options (per leg)
# -----------------------------------------
# `Nrev` (or `lambert_nrev`) is interpreted as max revolution count per leg.
# `lambert_hz`: +1=ccw only, -1=cw only, 0=both directions.
Nrev = [0] * nL
lambert_nrev = Nrev
lambert_hz = [1] * nL

# -----------------------------------------
# 8) DV limit to patch legs (i.e., powered flybys) per leg (km/s)
# -----------------------------------------
dVfb_max = [2.0] * nL

# -----------------------------------------
# 8) DSM leveraging grid for solve_arc (per leg + global mode)
# -----------------------------------------
# If dVlev_max[k] <= 0, leg k stays Lambert-only.
# Otherwise, leg k expands with solve_arc on a non-zero grid up to dVlev_max[k]
# using spacing delta_dvlev[k].
dVlev_max = [2.0] * nL
delta_dvlev = [0.5] * nL
lev_type = ["+-", "-+"]

"""
Europa Clipper interplanetary trajectory design (2024 launch)
Senent et al., "Europa Clipper Mission Design: Interplanetary Trajectory Mission Analysis"
"""

import numpy as np
from star.constants.time import et_seconds_to_j2000_days, utc_string_to_et_seconds

# -----------------------------
# 1) Body sequence (encounters)
# -----------------------------
# Use NAIF IDs (integers) like the MATLAB version.
# Example: Earth -> {Venus/Mars/Earth} -> {Venus/Mars/Earth} -> Jupiter
Body = [
    [399],                 # encounter 1 (departure): Earth
    [499],       # encounter 2: Venus/Mars/Earth options
    [399],       # encounter 3: Venus/Mars/Earth options
    [599],          # encounter 4 (arrival): Jupiter
]

nE = len(Body)  # Number of encounters
nL = nE - 1   # Number of legs

# -----------------------------------------
# 2) Encounter time windows (days from J2000)
# -----------------------------------------
# Use +/- inf for unspecified windows (same convention you referenced).
# Set departure epoch by UTC string and convert to days past J2000.
t0_utc = "2024-01-01T00:00:00"
t0 = et_seconds_to_j2000_days(utc_string_to_et_seconds(t0_utc))

Time = [None] * len(Body)
Time[0] = np.array([t0 + 0.0, t0 + 365.0])        # departure window
Time[-1] = np.array([t0 + 600.0, t0 + 2920.0])    # arrival window (< 8 years)

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
tof_min[0, 1], tof_max[0, 1] = 100.0, 700.0   # Earth -> first flyby
tof_min[1, 2], tof_max[1, 2] = 100.0, 1000.0 # flyby1 -> flyby2
tof_min[2, 3], tof_max[2, 3] = 500.0, 1500.0 # flyby2 -> Jupiter

# total TOF constraint 
tof_min[0, 3], tof_max[0, 3] = 300.0, 2500.0

# -----------------------------------------
# 4) Encounter grid step sizes dt (days)
# -----------------------------------------
# In MATLAB they use dt{encounter_index} and can vary by encounter.
dt = [None] * nE
dt[0] = 1                    # departure grid
for k in range(1, nE - 1):
    dt[k] = 1                    # intermediate encounter grids
dt[-1] = 4                       # arrival grid

# -----------------------------------------
# 4b) Final-output tfilter bin size (days)
# -----------------------------------------
# Section 5.4 decimation on full trajectories only:
# use one bin size for both departure and arrival dimensions.
dt_filter = 5.0

# -----------------------------------------
# 5) Flyby altitude constraint (km) per encounter
# -----------------------------------------
# Used to compute rmin = radius + alt for later flyby feasibility (not needed for ballistic legs filtering).
# Keep scalar or per-body list/dict. Here: scalar per encounter.
alt = [None] * nE
alt[0] = 0.0
for k in range(1, nE - 1):
    alt[k] = 100.0
alt[-1] = 250.0

# -----------------------------------------
# 6) v-infinity bounds (km/s) per encounter
# -----------------------------------------
# vlim is 2 x nE (min row, max row) with custom overrides possible.
# Keep this simple: per-encounter min/max, applied when building legs at departure and arrival ends.
vlim = np.array([
    [0.0, 0.0, 0.0, 0.0],   # v_inf_min for encounters 1..nE
    [6.8, 15.0, 15.0, 6.0],   # v_inf_max for encounters 1..nE
], dtype=float)

# total mission cost (km/s)
dVtotal = 0.5

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
dVfb_max = [0.3] * nL

# -----------------------------------------
# 8) DSM leveraging grid for solve_arc (per leg + global mode)
# -----------------------------------------
# If dVlev_max[k] <= 0, leg k stays Lambert-only.
# Otherwise, leg k expands with solve_arc on a non-zero grid up to dVlev_max[k]
# using spacing delta_dvlev[k].
dVlev_max = [0.5] * nL
delta_dvlev = [0.05] * nL
lev_type = ["+-", "-+"]

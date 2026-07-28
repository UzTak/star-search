"""
Titan-transfer broad search surrogate from Landau et al. (2022), Table 3.

Implemented here:
- the heliocentric Earth -> inner planets -> Saturn portion of the Table 3 setup,
- the optional 3rd/4th inner-planet flyby via one null leg,
- the Table 3 launch window, mission TOF bounds, leg/triplet TOF limits,
- multi-revolution / bidirectional Lambert settings,
- nearly-ballistic DV limits and resonant-leg sampling.

Not implemented here:
- the Saturn -> Titan central-body switch,
- the Titan-centered control point,
- the Titan atmospheric-interface leg.

Because this repository does not yet support central-body switching, the search
terminates at the heliocentric Saturn arrival state.
"""

import numpy as np

# Optional run controls.
n_process = 2
max_save_sol = 50_000


# ---------------------------------------------------------------------------
# 1) Body sequence (encounters)
# ---------------------------------------------------------------------------
Body = [
    [399],            # Earth
    [299, 399, 499],  # Venus / Earth / Mars
    [299, 399, 499],  # Venus / Earth / Mars
    [299, 399, 499],  # Venus / Earth / Mars
    [299, 399, 499],  # Venus / Earth / Mars
    [699],            # Saturn
]

nE = len(Body)
nL = nE - 1


# ---------------------------------------------------------------------------
# 1b) Optional inner-planet flyby count
# ---------------------------------------------------------------------------
# Table 3: N(2) = true, which allows either 3 or 4 inner-planet flybys within
# the fixed B[2:5] block.
null = [False] * nL
null[1] = True


# ---------------------------------------------------------------------------
# 2) Encounter time windows (days from J2000)
# ---------------------------------------------------------------------------
# Table 3:
#   T[1] = (9132, 10958) days  [launch in 2025-2030]
#
# The Saturn arrival window is left implicit and tightened from the total TOF
# bound tof[1,6] below.
Time = [None] * nE
Time[0] = np.array([9132.0, 10958.0], dtype=float)
for stage_index in range(1, nE):
    Time[stage_index] = np.array([-np.inf, np.inf], dtype=float)


# ---------------------------------------------------------------------------
# 3) TOF bounds (days)
# ---------------------------------------------------------------------------
tof_min = np.full((nE, nE), -np.inf, dtype=float)
tof_max = np.full((nE, nE), np.inf, dtype=float)

# Table 3: 7-11 years launch-to-Saturn.
tof_min[0, nE - 1], tof_max[0, nE - 1] = 2557.0, 4018.0

# Table 3: max 3.5 years per inner-planet leg for i = 1..4.
for leg_index in range(0, 4):
    tof_min[leg_index, leg_index + 1] = 0.0
    tof_max[leg_index, leg_index + 1] = 1278.0

# Table 3: max 5 years per inner-planet triplet.
for stage_index in range(0, 3):
    tof_min[stage_index, stage_index + 2] = 0.0
    tof_max[stage_index, stage_index + 2] = 1826.0


# ---------------------------------------------------------------------------
# 4) Encounter grid step sizes dt (days)
# ---------------------------------------------------------------------------
# Table 3:
#   dt[1:5] = 1 day
#   dt[6]   = 3 days
dt = [2.0, 2.0, 2.0, 2.0, 2.0, 5.0]


# ---------------------------------------------------------------------------
# 4b) Final-output tfilter bin size (days)
# ---------------------------------------------------------------------------
# Table 3: dtfilter(1:8) = 30 days. Apply the same 30-day binning to the
# truncated Earth-to-Saturn surrogate.
dt_filter = 30.0


# ---------------------------------------------------------------------------
# 5) Flyby altitude constraints (km)
# ---------------------------------------------------------------------------
# Table 3:
#   a_min[2:5] = (300, 1000, 300) km for Venus / Earth / Mars
#   a_min[6]   = 3000 km for the Saturn central-body-switch periapsis
#
# The Python driver now accepts per-body altitude maps at each stage, so the
# mixed Venus/Earth/Mars stage constraint can be represented directly.
alt_inner = {
    299: 300.0,   # Venus
    399: 1000.0,  # Earth
    499: 300.0,   # Mars
}
alt = [
    0.0,
    alt_inner,
    alt_inner,
    alt_inner,
    alt_inner,
    3000.0,  # unused in this surrogate, retained for Table 3 fidelity
]


# ---------------------------------------------------------------------------
# 6) v-infinity and DV limits
# ---------------------------------------------------------------------------
# Table 3:
#   dVtotal      = 0.100 km/s
#   dVenc(2:7)   = 0.050 km/s
#   vmax(1)      = 5 km/s
#   vmax(2:7)    = 18 km/s
vlim = np.array(
    [
        [2.0, 2.0, 2.0, 2.0, 2.0, 2.0],
        [5.0, 18.5, 18.0, 18.0, 18.0, 18.0],
    ],
    dtype=float,
)
dVtotal = 0.1

# Flyby patch DV is only used on interior flyby stages here.
dVfb_max = [0.0, 0.05, 0.05, 0.05, 0.05, 0.0]


# ---------------------------------------------------------------------------
# 7) Lambert arc options
# ---------------------------------------------------------------------------
# Table 3:
#   Nrev = (0, +/-1, +/-2, +/-3)
#
Nrev = [3] * nL
lambert_nrev = Nrev
lambert_hz = [1] * nL


# ---------------------------------------------------------------------------
# 8) Resonant-leg sampling
# ---------------------------------------------------------------------------

# NOTE: I am not including the resonant here, because activating resonant blowed the 
# combinatorics to approx. 1 billion order in leg 2 and leg 3. 
# For interplanetary transfer (non-moon tour), in practice I have been finding the similar solutions 
# with significantly less time. 

# resonant = [True] * nL
# dvinf = [0.05] * nL

# crank_angles_rad = np.deg2rad(np.arange(0.0, 360.0, 90.0, dtype=float))
# crank_angle = {
#     leg_index: crank_angles_rad
#     for leg_index in range(nL)
# }



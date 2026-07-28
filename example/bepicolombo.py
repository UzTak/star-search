"""
BepiColombo-style Mercury rendezvous search from Landau et al. (2022), Table 5.

Intentionally not modeled here:
- `nullleg` / `N(...)` from the MATLAB input (per request),
- full `wins` rendezvous insertion added into the search cost,
- custom search `order`,
- asymmetric per-endpoint `dtfilter`.

Where the paper used finer-grained settings than the Python config supports,
this file chooses the closest supported approximation and documents it inline.
"""

import numpy as np

# multi-processing 
n_process = 7
max_save_sol = 50_000 

# -----------------------------
# 1) Body sequence (encounters)
# -----------------------------
#   Mercury = 199, Venus = 299, Earth = 399.
Body = [
    [399],
    [299, 399],
    [199, 299],
    [199, 299],
    [199],
    [199],
    [199],
    [199],
    [199],
    [199],
    [199],
    [199],
]

nE = len(Body)
nL = nE - 1

null = [False] * nL 
null[2] = True  
null[4] = True
null[5] = True

# -----------------------------------------
# 2) Encounter time windows (days from J2000)
# -----------------------------------------
Time = [None] * nE
Time[0] = np.array([5479.0, 7305.0], dtype=float)
Time[-1] = np.array([5479.0 + 1461.0, 7305.0 + 2735.0], dtype=float)
for stage_index in range(1, nE - 1):
    Time[stage_index] = np.array([-np.inf, np.inf], dtype=float)


# -----------------------------------------
# 3) TOF bounds for each leg / triplet (days)
# -----------------------------------------
tof_min = np.full((nE, nE), -np.inf, dtype=float)
tof_max = np.full((nE, nE), np.inf, dtype=float)

tof_min[0, nE - 1], tof_max[0, nE - 1] = 1461.0, 2735.0

tof_min[0, 1], tof_max[0, 1] = 0.0, 549.0
for leg_index in range(1, nL):
    tof_min[leg_index, leg_index + 1] = 0.0
    tof_max[leg_index, leg_index + 1] = 484.0

for stage_index in range(0, nE - 2):
    tof_min[stage_index, stage_index + 2] = 0.0
    tof_max[stage_index, stage_index + 2] = 836.0


# -----------------------------------------
# 4) Encounter grid step sizes dt (days)
# -----------------------------------------
#   Earth   = 3 days
#   Venus   = 2 days
#   Mercury = 1 day
dt = [
    3.0,  # Earth
    3.0,  # Venus/Earth
    2.0,  # Mercury/Venus
    2.0,  # Mercury/Venus
    2.0,
    2.0,
    2.0,
    2.0,
    2.0,
    2.0,
    2.0,
    2.0,
]


# -----------------------------------------
# 4b) Final-output tfilter bin size (days)
# -----------------------------------------
dt_filter = [91] + [300] * (nE-2) + [22] 
# When flagged, the dt-filter is applied at each stage of combo. 
# This prevents the massive combinatorial growth for many flyby problems.  
dt_filter_preemptive = True

# -----------------------------------------
# 5) Flyby altitude constraint (km)
# -----------------------------------------
alt = [200.0] * nE

# -----------------------------------------
# 6) v-infinity bounds (km/s) per encounter
# -----------------------------------------
vlim = np.array(
    [
        [0.0, 2.0, 2.0, 5.0, 5.0, 4.0, 2.0, 1.0, 1.0, 1.0, 0.0, 0.0], 
        [4.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 5.0, 4.0, 3.0, 2.0, 0.6],
    ],
    dtype=float,
)
dVtotal = 1.8


# -----------------------------------------
# 7) Lambert arc options (per leg)
# -----------------------------------------
# Table 5 uses:
#   Nrev[1-2]  = (0, +/-1)
#   Nrev[3-11] = (0, +/-1, ..., +/-5)
#
# The Python interface stores only the max revolution count per leg; long/short
# period branches are handled inside the Lambert sweep. Keep prograde heliocentric
# motion only, matching the existing interplanetary examples.
Nrev = [1, 1] + [5] * (nL - 2)
lambert_nrev = Nrev
lambert_hz = [1] * nL

# -----------------------------------------
# 8) Flyby patch DV limit (km/s)
# -----------------------------------------
# Table 5: DeltaV_enc(2-11) = 0.300 km/s.
# The current driver accepts per-encounter limits; stage 1 and 12 entries are unused.
dVfb_max = [0.08] * nE

# -----------------------------------------
# 9) Powered-leg DSM leveraging settings
# -----------------------------------------
# Table 5:
#   DeltaV_leg(1-11)  = 0.600 km/s
#   delta_DeltaV_leg  = 0.050 km/s
dVlev_max = [0.45] * nL
delta_dvlev = [0.05] * nL
# leveraging in only direciton of favoring to reduce the arrival Vinf 
lev_type = ["+-"]


# -----------------------------------------
# 10) Resonant-leg sampling
# -----------------------------------------
# Table 5 lists:
#   delta_vinf(1,2) = 0.050 km/s
#   kappa           = {0, 180 deg}
#
# In the MATLAB paper setup those settings are tied to resonant Mercury-leg
# families. The current Python interface supports per-physical-leg resonant flags,
# so apply the same in-plane sampling to the Mercury-leveraging portion of the
# chain (legs 3-11 in the paper's 1-based indexing).
resonant = [False, False] + [True] * (nL - 2)
dvinf = [0.0, 0.0] + [0.05] * (nL - 2)

angles_in_plane = np.array([0.0, np.pi], dtype=float)
crank_angle = {
    leg_index: angles_in_plane
    for leg_index in range(2, nL)
}


# -----------------------------------------
# 11) Custom flyby filter: non-increasing post-flyby orbital energy at any flyby
# -----------------------------------------

def nonincreasing_energy(
    stage_id,
    body_id,
    r_km,
    v_body_km_s,
    vinf_in_km_s,
    vinf_out_km_s,
    mu_central_km3_s2,
    tol=1e-10,
):
    """Keep flyby pairs whose post-flyby orbital energy does not increase at any flyby."""

    del stage_id, body_id

    r_arr = np.asarray(r_km, dtype=float).reshape(-1, 3)
    v_body_arr = np.asarray(v_body_km_s, dtype=float).reshape(-1, 3)
    vinf_in_arr = np.asarray(vinf_in_km_s, dtype=float).reshape(-1, 3)
    vinf_out_arr = np.asarray(vinf_out_km_s, dtype=float).reshape(-1, 3)

    if r_arr.shape[0] == 0:
        return np.empty(0, dtype=bool)

    r_norm = np.linalg.norm(r_arr, axis=1)
    v_in_arr = v_body_arr + vinf_in_arr
    v_out_arr = v_body_arr + vinf_out_arr

    eps_in = 0.5 * np.einsum("ij,ij->i", v_in_arr, v_in_arr) - float(mu_central_km3_s2) / r_norm
    eps_out = 0.5 * np.einsum("ij,ij->i", v_out_arr, v_out_arr) - float(mu_central_km3_s2) / r_norm

    keep = (
        np.isfinite(eps_in)
        & np.isfinite(eps_out)
        & (eps_in < 0.0)
        & (eps_out < 0.0)
        & (eps_out <= (eps_in + float(tol)))
    )
    return keep

flyby_filter = nonincreasing_energy

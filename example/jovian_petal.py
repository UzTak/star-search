"""
Jovian moon tour example. (Escape from the nominal orbit (Europa 4:1 resonance / part of Petal rotation) in the Europa Clipper's moon tour)
Constraint: Last-leg departure from Jupiter must have periapsis above Ganymede's orbit to reach 
the "safe orbit" region. 
cf. Takbuo et al., "Preliminary Contingency Trajectory Planning for Europa Clipper's Galilean Moon Tour", AIAA SCITECH 2026 Forum. 
"""

# Notes:
# - Keep inertial frame as ECLIPJ2000, but switch central body to Jupiter via observer.

import numpy as np

# multi-processing 
n_process = 10

# -----------------------------------------
# 0) Global frame / central body
# -----------------------------------------
# Keep inertial frame, switch observer (state origin + Lambert mu source) to Jupiter.
frame = "ECLIPJ2000"
observer = "599"  # Jupiter

# -----------------------------
# cf. period of Europa: 3.551 days, Ganymede: 7.154 days, Callisto: 16.689 days.
# -------------------------------

# -----------------------------
# 1) Body sequence (encounters)
# -----------------------------
Body = [
    [502],        # 1: Europa
    [502],        # 2: Europa
    [503, 504],   # 3: Ganymede or Callisto
    [503, 504],   # 4: Ganymede or Callisto
    [504],        # 5: Callisto
    [504],        # 6: Callisto
]

nE = len(Body)
nL = nE - 1

resonant = [True] * nL
# resonant[0] = True
# resonant[-1] = True

dvinf = 0.1
angles = np.arange(0.0, 2 * np.pi, np.pi / 4)
crank_angle = {0: 0.0, nL - 1: angles}

# -----------------------------------------
# 2) Encounter time windows (days from J2000)
# -----------------------------------------
t0 = 11771.1591

Time = [None] * nE
Time[0] = np.array([t0 + 0.0, t0 + 0.0])      # fixed departure
Time[-1] = np.array([t0 + 0.0, t0 + 365.0])   # arrival window

# Middle encounters are unconstrained directly; TOF bounds below tighten them.
for k in range(1, nE - 1):
    Time[k] = np.array([-np.inf, np.inf])

# -----------------------------------------
# 3) TOF bounds for each leg (days)
# -----------------------------------------
tof_min = np.full((nE, nE), -np.inf, dtype=float)
tof_max = np.full((nE, nE), np.inf, dtype=float)

# From MATLAB input
tof_min[0, 1], tof_max[0, 1] = 13.0, 15.0
tof_min[1, 2], tof_max[1, 2] = 15.0, 80.0
tof_min[2, 3], tof_max[2, 3] = 15.0, 80.0
tof_min[nE - 2, nE - 1], tof_max[nE - 2, nE - 1] = 30.0, 100.0
# total transfer time 
tof_min[0, nE - 1], tof_max[0, nE - 1] = 100.0, 400.0

# -----------------------------------------
# 4) Encounter grid step sizes dt (days)
# -----------------------------------------
dt = [0.05, 0.05, 0.05, 0.05, 0.1, 0.1]
dt_filter = [0.2, 0.2, 0.2, 0.25, 0.5, 0.5]
dt_filter_preemptive = True

# -----------------------------------------
# 5) Flyby altitude constraint (km) per encounter
# -----------------------------------------
alt = [25.0, 25.0, 100.0, 100.0, 100.0, 100.0]

# -----------------------------------------
# 6) v-infinity bounds (km/s) per encounter
# -----------------------------------------
vlim = np.array(
    [
        [3.75, 3.75, 4.5, 4.0, 3.5, 3.5],  # min
        [4.5, 5.0, 6.0, 6.0, 6.0, 6.5],    # max
    ],
    dtype=float,
)

# -----------------------------------------
# 7) Lambert arc options (per leg)
# -----------------------------------------
Nrev = [1, 6, 6, 3, 2]
lambert_nrev = Nrev
lambert_hz = [1] * nL

# -----------------------------------------
# 8) Powered-flyby and DSM leveraging settings
# -----------------------------------------
dVtotal = 0.1
dVfb_max = [0.03] * nL

# DSMs 
dVlev_max = [0.025] * nL
delta_dvlev = [0.005] * nL
# generally we want to increase the energy 
lev_type = ["-+"]


ganymede_sma = 1.0709e6    # km
mu_jupiter   = 1.26686534e8  # km³/s², Jupiter GM (DE430)

def final_rp(
    stage_id,
    vinf_dep_km_s,
    vinf_arr_km_s,
    r_dep_km,
    v_body_dep_km_s,
    r_arr_km,
    v_body_arr_km_s,
):
    """Keep last-leg departures whose Jovicentric periapsis > Ganymede SMA."""
    if stage_id != nL - 1:
        return np.ones(vinf_dep_km_s.shape[0], dtype=bool)

    v_sc  = v_body_dep_km_s + vinf_dep_km_s          # (N,3) spacecraft vel
    r_mag = np.linalg.norm(r_dep_km, axis=1)          # (N,)
    v_sq  = np.einsum("ij,ij->i", v_sc, v_sc)         # (N,)

    eps   = 0.5 * v_sq - mu_jupiter / r_mag            # specific energy
    h_vec = np.cross(r_dep_km, v_sc)                   # (N,3) ang. momentum
    h_sq  = np.einsum("ij,ij->i", h_vec, h_vec)        # (N,)

    e_sq  = np.maximum(1.0 + 2.0 * eps * h_sq / mu_jupiter**2, 0.0)
    e     = np.sqrt(e_sq)
    rp    = (h_sq / mu_jupiter) / (1.0 + e)            # periapsis radius

    return rp > ganymede_sma


leg_filter = final_rp
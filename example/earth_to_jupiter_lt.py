"""Small quadratic low-thrust search: Earth -> Mars -> Earth -> Jupiter.

Reuse the impulsive example's geometry and limits, with three epochs per stage
near a low-cost ballistic solution from that example. Widen Time after this
smoke test; these narrow windows are not a mission-wide search.
"""

import numpy as np

from example.earth_to_jupiter import (
    Body, nE, nL, t0, tof_min, tof_max, alt, vlim,
    Nrev, lambert_nrev, lambert_hz, dt_filter,
)

n_process = 1
Time = [t0 + offset + np.array([-5.0, 5.0])
        for offset in (276.0, 423.0, 1065.0, 2324.0)]
dt = [5.0] * nE  # 3 epochs/stage: 9 Lambert seeds/leg, zero revolutions.

# Freeze primer steering on Lambert arcs and sample constant acceleration.
dVlev_max = [0.0] * nL
delta_dvlev = [0.0] * nL
acc_lt_max = [1e-7] * nL  # km/s^2 = 0.1 mm/s^2.
dacc_lt = [3e-8, 8e-9, 4e-9]  # Two affordable levels/mode in these Time windows.
isp_lt_s = [3200.0] * nL
lev_type = ["+-", "-+"]
dVtotal = 1.0  # km/s: accumulated integral of acceleration plus flyby patches.

# Quadratic endpoint estimates allow finite patches; selected arcs need replay.
dVfb_max = [0.1] * nE  # km/s; charged alongside aT in the existing search budget.

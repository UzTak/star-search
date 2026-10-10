"""Quadratic low-thrust endpoint sensitivities and shared physical replay."""

import numpy as np
from numba import njit, prange
from scipy.integrate import solve_ivp
from scipy.interpolate import CubicSpline

from star.maneuver_placement import (
    _broadcast_seed_vec3,
    _conic_primer_basis,
    _primer_vector_from_basis,
    _solve_arc_normalize_lev_types,
    _true_to_mean,
    prepare_primer_geometry,
    primer_vector_from_coeffs,
    rtn_to_catesian,
    solve_primer_coefficients,
    vinf_objective_gradients_rtn,
)


def build_acc_lt_grid_km_s2(max_km_s2: float, delta_km_s2: float, *, tol: float = 1e-12) -> np.ndarray:
    """Build a positive acceleration grid [km/s^2] up to max."""
    acc_max = float(max_km_s2)
    if not np.isfinite(acc_max) or acc_max < 0.0:
        raise ValueError("acc_lt_max_km_s2 must be finite and >= 0.")
    if acc_max <= float(tol):
        return np.empty(0, dtype=float)

    acc_step = float(delta_km_s2)
    if not np.isfinite(acc_step) or acc_step <= 0.0:
        raise ValueError("delta_acc_lt_km_s2 must be finite and > 0 when acc_lt_max_km_s2 > 0.")

    n_full = int(np.floor((acc_max + float(tol)) / acc_step))
    if n_full <= 0:
        grid = np.asarray([acc_max], dtype=float)
    else:
        grid = acc_step * np.arange(1, n_full + 1, dtype=float)
        grid = grid[grid <= acc_max + float(tol)]
        if grid.size == 0 or (acc_max - float(grid[-1])) > float(tol):
            grid = np.concatenate((grid, np.asarray([acc_max], dtype=float)))
        elif abs(float(grid[-1]) - acc_max) <= float(tol):
            grid[-1] = acc_max
    return np.unique(np.asarray(grid, dtype=float))


def _normalize_acc_grid_km_s2(acc_grid_km_s2):
    arr = np.asarray(acc_grid_km_s2, dtype=float).reshape(-1)
    return arr


def _empty_low_thrust_result(acc_grid_km_s2, lev_type, return_profile):
    acc_grid = np.asarray(acc_grid_km_s2, dtype=float).reshape(-1)
    out = {
        "seed_idx": np.empty(0, dtype=int),
        "lev_type": str(lev_type),
        "acc_grid_km_s2": acc_grid.copy(),
        "Vinf0_nom_vec_km_s": np.empty((0, 3), dtype=float),
        "Vinff_nom_vec_km_s": np.empty((0, 3), dtype=float),
        "Vinf0_nom_km_s": np.empty(0, dtype=float),
        "Vinff_nom_km_s": np.empty(0, dtype=float),
        "Jlev_nom_km_s": np.empty(0, dtype=float),
        "dVinf0_vec_da_s": np.empty((0, 3), dtype=float),
        "dVinff_vec_da_s": np.empty((0, 3), dtype=float),
        "d2Vinf0_vec_da2_s3_km": np.empty((0, 3), dtype=float),
        "d2Vinff_vec_da2_s3_km": np.empty((0, 3), dtype=float),
        "dVinf0_da_s": np.empty(0, dtype=float),
        "dVinff_da_s": np.empty(0, dtype=float),
        "dJlev_da_s": np.empty(0, dtype=float),
        "Vinf0_vec_lt_km_s": np.empty((0, acc_grid.size, 3), dtype=float),
        "Vinff_vec_lt_km_s": np.empty((0, acc_grid.size, 3), dtype=float),
        "Vinf0_lt_km_s": np.empty((0, acc_grid.size), dtype=float),
        "Vinff_lt_km_s": np.empty((0, acc_grid.size), dtype=float),
        "Jlev_lt_km_s": np.empty((0, acc_grid.size), dtype=float),
    }
    if return_profile:
        out["nu_grid"] = []
        out["primer_rtn"] = []
        out["steering_rtn"] = []
        out["primer_norm"] = []
        out["dt_dnu_s"] = []
    return out


def _binned_primer_profile(r0, rf, vp0, vpf, v0, vf, nrev, mu):
    """Freeze Cartesian component primers on one elliptic Lambert reference."""
    geometry = prepare_primer_geometry(r0, rf, [v0], [vf], mu, [abs(nrev)])
    e, nu0, nuf = (float(geometry[key][0]) for key in ("e", "nu0", "nuf"))
    if not 1e-8 < e <= 0.99 or nuf <= nu0:
        raise ValueError("Binned low thrust requires a nondegenerate elliptic seed.")
    q0, qf = rtn_to_catesian(r0, [v0]), rtn_to_catesian(rf, [vf])
    coeffs = []
    for gradients, departure in ((q0, True), (qf, False)):
        for grad in _cartesian_component_gradients_rtn(gradients)[0]:
            coeffs.append(solve_primer_coefficients(
                geometry, [grad] if departure else [[0, 0, 0]],
                [[0, 0, 0]] if departure else [grad]))
    nu = np.linspace(nu0, nuf, 400 * (1 + abs(nrev)) + 1)
    component = primer_vector_from_coeffs(
        nu[:, None], e, **{key: np.array([c[key][0] for c in coeffs])[None, :]
                           for key in ("A", "B", "C", "D", "E", "F")})
    vhat0, vhatf = (vec / np.linalg.norm(vec) for vec in (v0 - vp0, vf - vpf))
    fields_rtn = np.stack((np.einsum("nik,i->nk", component[:, :3], vhat0),
                           np.einsum("nik,i->nk", component[:, 3:], vhatf)), axis=1)
    er0, et0, en = q0[0].T
    angle = nu - nu0
    er = np.cos(angle)[:, None] * er0 + np.sin(angle)[:, None] * et0
    et = -np.sin(angle)[:, None] * er0 + np.cos(angle)[:, None] * et0
    frames = np.stack((er, et, np.broadcast_to(en, er.shape)), axis=2)
    fields = np.einsum("nij,nkj->nki", frames, fields_rtn)
    mean = _true_to_mean(nu, np.full(nu.size, e))
    time_s = (mean - mean[0]) / np.sqrt(mu / geometry["a_semimajor"][0]**3)
    return geometry, time_s, component, fields_rtn, CubicSpline(time_s / time_s[-1], fields)


def _weighted_primer(fields, theta):
    p = np.cos(theta) * fields[0] + np.sin(theta) * fields[1]
    norm = np.linalg.norm(p)
    if norm < 1e-12:  # Direction at an isolated primer zero is arbitrary.
        fallback = fields[np.argmax(np.linalg.norm(fields, axis=1))]
        return fallback / np.linalg.norm(fallback), np.zeros(3)
    unit = p / norm
    derivative = -np.sin(theta) * fields[0] + np.cos(theta) * fields[1]
    return unit, (derivative - unit * np.dot(unit, derivative)) / norm


def _propagate_binned(r0, v0, tof_s, mu, acceleration, theta, profile,
                      normal, *, rtol, initial_jac=None, acc_max=None, num_samples=None):
    """Integrate the controlled two-body arc, optionally with shooting derivatives."""
    length = np.linalg.norm(r0)
    speed = np.sqrt(mu / length)
    time_unit = length / speed
    duration = tof_s / time_unit
    acc_scale = length / speed**2
    y0 = np.r_[r0 / length, v0 / speed, 0.0]
    if initial_jac is not None:
        sens = np.zeros((6, 4))
        sens[3:, :2] = initial_jac / speed
        y0 = np.r_[y0, sens.ravel()]

    def rhs(t, y):
        r, v = y[:3], y[3:6]
        radius = np.linalg.norm(r)
        unit, du = _weighted_primer(profile(t / duration), theta)
        projected_r_sq = np.dot(r, r) - np.dot(r, normal)**2
        phase_rate = np.dot(normal, np.cross(r, v)) / max(projected_r_sq, 1e-20)
        out = np.r_[v, -r / radius**3 + acceleration * acc_scale * unit, phase_rate]
        if initial_jac is not None:
            sensitivity = y[7:].reshape(6, 4)
            gravity = 3 * np.outer(r, r) / radius**5 - np.eye(3) / radius**3
            ds = np.vstack((sensitivity[3:], gravity @ sensitivity[:3]))
            ds[3:, 2] += acceleration * acc_scale * du
            ds[3:, 3] += acc_max * acc_scale * unit
            out = np.r_[out, ds.ravel()]
        return out

    times = None if num_samples is None else np.linspace(0.0, duration, num_samples)
    solution = solve_ivp(rhs, (0.0, duration), y0, method="DOP853",
                         rtol=rtol, atol=rtol * 0.01, t_eval=times)
    if not solution.success or not np.all(np.isfinite(solution.y)):
        raise ValueError("Low-thrust propagation failed.")
    if num_samples is not None:
        states = solution.y[:6].T.copy()
        states[:, :3] *= length
        states[:, 3:] *= speed
        thrust = acceleration * np.array([
            _weighted_primer(profile(t / duration), theta)[0] for t in times])
        return states, thrust
    final = solution.y[:, -1].copy()
    final[:3] *= length
    final[3:6] *= speed
    if initial_jac is not None:
        final[7:].reshape(6, 4)[:3] *= length
        final[7:].reshape(6, 4)[3:] *= speed
    return final


def propagate_binned_low_thrust(r0, rf, vp0, vpf, vinf0, nrev, tof_s, mu,
                                acceleration, control, *, rtol=2e-13, num_samples=None):
    """Propagate a stored frozen-primer control, including analytic search legs.

    Control holds nominal v0/vf, primer angle, and two endpoint speeds (9 values).
    Speeds are enforced targets for binned legs and predictions for analytic
    legs; replaying the latter measures their nonlinear endpoint error.

    By default return the final Cartesian state (6 values). With ``num_samples``
    return uniformly timed ``(states_km_km_s, thrust_km_s2)`` arrays of shape
    (num_samples, 6) and (num_samples, 3), in the reference arc's inertial frame.
    """
    control = np.asarray(control, dtype=float).reshape(9)
    if not np.all(np.isfinite(control)):
        raise ValueError("Low-thrust control must contain nine finite values.")
    if not np.isfinite(acceleration) or acceleration < 0 or not np.isfinite(tof_s) or tof_s <= 0:
        raise ValueError("Low-thrust replay requires nonnegative acceleration and positive duration.")
    if num_samples is not None:
        if int(num_samples) != num_samples or num_samples < 2:
            raise ValueError("num_samples must be an integer >= 2.")
        num_samples = int(num_samples)
    geometry, _, _, _, profile = _binned_primer_profile(
        np.asarray(r0), np.asarray(rf), np.asarray(vp0), np.asarray(vpf),
        control[:3], control[3:6], int(nrev), float(mu))
    normal = geometry["h_vec"][0] / geometry["h"][0]
    result = _propagate_binned(np.asarray(r0), np.asarray(vp0) + vinf0, tof_s, mu,
                               acceleration, control[6], profile, normal, rtol=rtol,
                               num_samples=num_samples)
    return result[:6] if num_samples is None else result


def _jlev_from_vinfs(vinf0_km_s, vinff_km_s, lev_type):
    if lev_type == "+-":
        return vinff_km_s - vinf0_km_s
    if lev_type == "-+":
        return vinf0_km_s - vinff_km_s
    raise ValueError("Invalid lev_type. Choose '+-' or '-+'.")


def _safe_unit_rows(vec_batch, eps=1e-12):
    vec = np.asarray(vec_batch, dtype=float).reshape(-1, 3)
    norms = np.linalg.norm(vec, axis=1, keepdims=True)
    return vec / np.maximum(norms, float(eps))


def _cartesian_component_gradients_rtn(rtn_to_cart_batch):
    rtn = np.asarray(rtn_to_cart_batch, dtype=float).reshape(-1, 3, 3)
    eye3 = np.eye(3, dtype=float)
    return np.einsum("nji,kj->nki", rtn, eye3)


def build_low_thrust_vinf_table(result, seed_row=0):
    """Build per-acceleration V-infinity comparison rows for one valid seed."""
    seed_idx = np.asarray(result["seed_idx"], dtype=int).reshape(-1)
    if seed_idx.size == 0:
        return []

    seed_row = int(seed_row)
    if seed_row < 0 or seed_row >= seed_idx.size:
        raise IndexError("seed_row is out of range for the low-thrust result.")

    lev_type = str(result["lev_type"])
    acc_grid = np.asarray(result["acc_grid_km_s2"], dtype=float).reshape(-1)
    vinf0_nom = float(np.asarray(result["Vinf0_nom_km_s"], dtype=float).reshape(-1)[seed_row])
    vinff_nom = float(np.asarray(result["Vinff_nom_km_s"], dtype=float).reshape(-1)[seed_row])
    jlev_nom = float(np.asarray(result["Jlev_nom_km_s"], dtype=float).reshape(-1)[seed_row])
    vinf0_lt = np.asarray(result["Vinf0_lt_km_s"], dtype=float).reshape(seed_idx.size, -1)[seed_row]
    vinff_lt = np.asarray(result["Vinff_lt_km_s"], dtype=float).reshape(seed_idx.size, -1)[seed_row]
    jlev_lt = np.asarray(result["Jlev_lt_km_s"], dtype=float).reshape(seed_idx.size, -1)[seed_row]

    rows = [
        {
            "seed_row": seed_row,
            "seed_idx": int(seed_idx[seed_row]),
            "lev": lev_type,
            "acc_km_s2": 0.0,
            "vinf0_km_s": vinf0_nom,
            "delta_vinf0_km_s": 0.0,
            "vinff_km_s": vinff_nom,
            "delta_vinff_km_s": 0.0,
            "jlev_km_s": jlev_nom,
            "delta_jlev_km_s": 0.0,
        }
    ]

    for acc_i, acc in enumerate(acc_grid):
        if np.isclose(float(acc), 0.0, rtol=0.0, atol=1e-15):
            rows[0]["vinf0_km_s"] = float(vinf0_lt[acc_i])
            rows[0]["vinff_km_s"] = float(vinff_lt[acc_i])
            rows[0]["jlev_km_s"] = float(jlev_lt[acc_i])
            continue
        rows.append(
            {
                "seed_row": seed_row,
                "seed_idx": int(seed_idx[seed_row]),
                "lev": lev_type,
                "acc_km_s2": float(acc),
                "vinf0_km_s": float(vinf0_lt[acc_i]),
                "delta_vinf0_km_s": float(vinf0_lt[acc_i] - vinf0_nom),
                "vinff_km_s": float(vinff_lt[acc_i]),
                "delta_vinff_km_s": float(vinff_lt[acc_i] - vinff_nom),
                "jlev_km_s": float(jlev_lt[acc_i]),
                "delta_jlev_km_s": float(jlev_lt[acc_i] - jlev_nom),
            }
        )
    return rows


def format_low_thrust_vinf_table(result, seed_row=0):
    """Format one seed's low-thrust V-infinity comparison rows as a plain-text table."""
    rows = build_low_thrust_vinf_table(result, seed_row=seed_row)
    if not rows:
        return "No valid low-thrust seeds."

    header = (
        f"{'seed':>4} {'lev':>4} {'acc[km/s^2]':>14} "
        f"{'|Vinf_0|':>12} {'d|Vinf_0|':>12} "
        f"{'|Vinf_f|':>12} {'d|Vinf_f|':>12} "
        f"{'Jlev':>12} {'dJlev':>12}"
    )
    lines = [header]
    for row in rows:
        lines.append(
            f"{row['seed_idx']:4d} {row['lev']:>4} {row['acc_km_s2']:14.6e} "
            f"{row['vinf0_km_s']:12.6f} {row['delta_vinf0_km_s']:12.6f} "
            f"{row['vinff_km_s']:12.6f} {row['delta_vinff_km_s']:12.6f} "
            f"{row['jlev_km_s']:12.6f} {row['delta_jlev_km_s']:12.6f}"
        )
    return "\n".join(lines)


def print_low_thrust_vinf_table(result, seed_row=0):
    """Print one seed's low-thrust V-infinity comparison table."""
    table = format_low_thrust_vinf_table(result, seed_row=seed_row)
    print(table)
    return table


@njit(cache=True)
def _simpson_weight_jit(node, count, step):
    """Uniform Simpson weights, including a final unmatched interval."""
    if count == 2:
        return step / 2
    last = count - 1 if count % 2 else count - 2
    weight = 0.0
    if node <= last:
        weight = step / 3 * (1 if node == 0 or node == last else 4 if node % 2 else 2)
    if count % 2 == 0 and node >= count - 3:
        weight += step / 12 * (-1 if node == count - 3 else 8 if node == count - 2 else 5)
    return weight


@njit(cache=True)
def _cumulative_simpson_jit(values, step):
    """Quadratic cumulative quadrature on uniform true-anomaly nodes."""
    count = values.shape[0]
    out = np.zeros_like(values)
    if count == 2:
        out[1] = step / 2 * (values[0] + values[1])
        return out
    for start in range(0, count - 2, 2):
        for comp in range(values.shape[1]):
            a, b, c = values[start, comp], values[start + 1, comp], values[start + 2, comp]
            out[start + 1, comp] = out[start, comp] + step / 12 * (5*a + 8*b - c)
            out[start + 2, comp] = out[start, comp] + step / 3 * (a + 4*b + c)
    if count % 2 == 0:
        out[-1] = out[-2] + step / 12 * (-values[-3] + 8*values[-2] + 5*values[-1])
    return out


@njit(cache=True, parallel=True)
def _integrate_lt_quadrature_jit(nu, e, p_orb, h, coeffs, components, arrival_frames):
    """Integrate linear and quadratic frozen-primer responses once per seed."""
    integrals = np.zeros((nu.shape[0], 6))
    objective = np.zeros(nu.shape[0])
    quadratic = np.zeros((nu.shape[0], 6))
    for seed in prange(nu.shape[0]):
        ecc, p, momentum = e[seed], p_orb[seed], h[seed]
        inv = 1 / (1 - ecc*ecc)
        k = (1 - ecc*ecc)**1.5
        sqrt_minus, sqrt_plus = np.sqrt(1 - ecc), np.sqrt(1 + ecc)
        count = nu.shape[1]
        step = (nu[seed, -1] - nu[seed, 0]) / (count - 1)
        basis = np.empty((count, 6))
        values = np.empty((count, 6))
        A, B, C, D, E, F = coeffs[seed]
        for node in range(count):
            angle = nu[seed, node]
            revolutions = np.floor(angle / (2*np.pi))
            wrapped = angle - 2*np.pi*revolutions
            anomaly = 2*np.arctan2(sqrt_minus*np.sin(wrapped / 2), sqrt_plus*np.cos(wrapped / 2))
            if anomaly < 0:
                anomaly += 2*np.pi
            mean = anomaly - ecc*np.sin(anomaly) + 2*np.pi*revolutions
            sn, cs = np.sin(angle), np.cos(angle)
            denom = 1 + ecc*cs
            i1 = inv * (2*ecc / denom - cs - 3*ecc*ecc*mean*sn / k)
            i2 = inv * ((1 + 1 / denom)*sn - 3*ecc*mean*denom / k)
            radius = p / denom
            dt = radius*radius / momentum
            basis[node] = sn, cs, denom, i1, i2, dt
            pr = A*cs + B*ecc*sn + C*i1
            pt = -(1 + 1 / denom)*A*sn + B*denom + C*i2 + D / denom
            pn = (E*cs + F*sn) / denom
            norm = np.sqrt(pr*pr + pt*pt + pn*pn)
            ur = ut = un = 0.0
            if norm > 1e-12:
                ur, ut, un = pr / norm, pt / norm, pn / norm
            values[node, 0] = (cs*ur - (1 + 1 / denom)*sn*ut)*dt
            values[node, 1] = (ecc*sn*ur + denom*ut)*dt
            values[node, 2] = (i1*ur + i2*ut)*dt
            values[node, 3] = ut / denom*dt
            values[node, 4] = cs*un / denom*dt
            values[node, 5] = sn*un / denom*dt
            objective[seed] -= _simpson_weight_jit(node, count, step)*norm*dt
        cumulative = _cumulative_simpson_jit(values, step)
        integrals[seed] = -cumulative[-1]

        # Inverse conic-basis Wronskian pairs (A,D), (B,C), (E,F).
        scale = p*p / momentum
        scale_rt = scale / ecc
        total = cumulative[-1]
        particular = np.array([scale_rt*total[3], -scale_rt*total[2],
                               scale_rt*total[1], -scale_rt*total[0],
                               -scale*total[5], scale*total[4]])
        sn, cs, denom, i1, i2, dt = basis[-1]
        final_rtn = np.array([
            cs*particular[0] + ecc*sn*particular[1] + i1*particular[2],
            -(1 + 1 / denom)*sn*particular[0] + denom*particular[1]
            + i2*particular[2] + particular[3] / denom,
            (cs*particular[4] + sn*particular[5]) / denom])
        final_cart = arrival_frames[seed] @ final_rtn
        # Arrival component primers cancel the forced displacement at the endpoint.
        homogeneous = np.zeros(6)
        for comp in range(6):
            for j in range(3):
                homogeneous[comp] += components[seed, j + 3, comp]*final_cart[j]
        for node in range(count):
            total = cumulative[node]
            aa = homogeneous[0] + scale_rt*total[3]
            bb = homogeneous[1] - scale_rt*total[2]
            cc = homogeneous[2] + scale_rt*total[1]
            dd = homogeneous[3] - scale_rt*total[0]
            ee = homogeneous[4] - scale*total[5]
            ff = homogeneous[5] + scale*total[4]
            sn, cs, denom, i1, i2, dt = basis[node]
            qr = cs*aa + ecc*sn*bb + i1*cc
            qt = -(1 + 1 / denom)*sn*aa + denom*bb + i2*cc + dd / denom
            qn = (cs*ee + sn*ff) / denom
            # Half of the gravity Hessian contracted with the linear displacement.
            gravity = momentum*momentum / (p*(p / denom)**4)
            fr = 1.5*gravity * (qt*qt + qn*qn - 2*qr*qr)
            ft, fn = 3*gravity*qr*qt, 3*gravity*qr*qn
            weight = -_simpson_weight_jit(node, count, step)*dt
            quadratic[seed, 0] += weight * (cs*fr - (1 + 1 / denom)*sn*ft)
            quadratic[seed, 1] += weight * (ecc*sn*fr + denom*ft)
            quadratic[seed, 2] += weight * (i1*fr + i2*ft)
            quadratic[seed, 3] += weight*ft / denom
            quadratic[seed, 4] += weight*cs*fn / denom
            quadratic[seed, 5] += weight*sn*fn / denom
    return integrals, objective, quadratic


def solve_arc_low_thrust(
    R0_km,
    Rf_km,
    V_P0_km_s,
    V_Pf_km_s,
    V0_km_s,
    Vf_km_s,
    nrev_signed,
    acc_grid_km_s2,
    mu_km3_s2,
    *,
    lev_type="+-",
    max_vinf0_km_s=100.0,
    max_vinff_km_s=100.0,
    num_nu=400,
    return_profile=False,
    debug=False,
):
    """Return quadratic endpoint estimates; second derivatives multiply a**2 / 2."""
    lev_type_list = _solve_arc_normalize_lev_types(1, lev_type)
    if len(lev_type_list) != 1:
        raise ValueError("solve_arc_low_thrust accepts exactly one lev_type string.")
    lev_type_norm = lev_type_list[0]

    mu = float(mu_km3_s2)
    if mu <= 0.0:
        raise ValueError("mu_km3_s2 must be positive.")

    max_vinf0 = float(max_vinf0_km_s)
    max_vinff = float(max_vinff_km_s)
    if max_vinf0 <= 0.0 or max_vinff <= 0.0:
        raise ValueError("max_vinf0_km_s and max_vinff_km_s must be positive.")

    num_nu = int(num_nu)
    if num_nu <= 0:
        raise ValueError("num_nu must be a positive integer.")

    acc_grid = _normalize_acc_grid_km_s2(acc_grid_km_s2)

    V0_arr = np.asarray(V0_km_s, dtype=float).reshape(-1, 3)
    Vf_arr = np.asarray(Vf_km_s, dtype=float).reshape(-1, 3)
    n_seed = int(V0_arr.shape[0])
    if Vf_arr.shape[0] != n_seed:
        raise ValueError("V0_km_s and Vf_km_s must have matching row counts.")

    nrev_arr = np.asarray(nrev_signed, dtype=int).reshape(-1)
    if nrev_arr.shape != (n_seed,):
        raise ValueError("nrev_signed must have shape (S,).")

    R0_arr = _broadcast_seed_vec3("R0_km", R0_km, n_seed)
    Rf_arr = _broadcast_seed_vec3("Rf_km", Rf_km, n_seed)
    V_P0_arr = _broadcast_seed_vec3("V_P0_km_s", V_P0_km_s, n_seed)
    V_Pf_arr = _broadcast_seed_vec3("V_Pf_km_s", V_Pf_km_s, n_seed)

    Vinf0_nom_vec = V0_arr - V_P0_arr
    Vinff_nom_vec = Vf_arr - V_Pf_arr
    Vinf0_nom = np.linalg.norm(Vinf0_nom_vec, axis=1)
    Vinff_nom = np.linalg.norm(Vinff_nom_vec, axis=1)

    keep_vinf = ((Vinf0_nom > 1e-12) & (Vinff_nom > 1e-12)
                 & (Vinf0_nom <= max_vinf0) & (Vinff_nom <= max_vinff))
    if not np.any(keep_vinf):
        return _empty_low_thrust_result(acc_grid, lev_type_norm, return_profile)

    seed_idx_vinf = np.flatnonzero(keep_vinf)
    R0_vinf = R0_arr[seed_idx_vinf]
    r0_vinf = np.linalg.norm(R0_vinf, axis=1)
    h_vinf = np.cross(R0_vinf, V0_arr[seed_idx_vinf])
    evec_vinf = np.cross(V0_arr[seed_idx_vinf], h_vinf) / mu - R0_vinf / r0_vinf.reshape(-1, 1)
    e_vinf = np.linalg.norm(evec_vinf, axis=1)

    ecc_eps = 1.0e-2
    keep_ecc = np.isfinite(e_vinf) & (e_vinf > 1e-8) & (e_vinf <= (1.0 - ecc_eps))
    if not np.any(keep_ecc):
        return _empty_low_thrust_result(acc_grid, lev_type_norm, return_profile)

    seed_idx = seed_idx_vinf[keep_ecc]
    R0_valid = R0_arr[seed_idx]
    Rf_valid = Rf_arr[seed_idx]
    V0_valid = V0_arr[seed_idx]
    Vf_valid = Vf_arr[seed_idx]
    V_P0_valid = V_P0_arr[seed_idx]
    V_Pf_valid = V_Pf_arr[seed_idx]
    Vinf0_nom_vec_valid = Vinf0_nom_vec[seed_idx]
    Vinff_nom_vec_valid = Vinff_nom_vec[seed_idx]
    nrev_valid = np.abs(nrev_arr[seed_idx]).astype(int, copy=False)

    if debug and seed_idx.size < n_seed:
        print(
            "Low-thrust seed filter: "
            f"kept={int(seed_idx.size)}/{n_seed}, "
            f"vinf_kept={int(seed_idx_vinf.size)}/{n_seed}, "
            f"ecc_kept={int(np.count_nonzero(keep_ecc))}/{int(seed_idx_vinf.size)}"
        )

    rtn0 = rtn_to_catesian(R0_valid, V0_valid)
    rtnf = rtn_to_catesian(Rf_valid, Vf_valid)
    geometry = prepare_primer_geometry(R0_valid, Rf_valid, V0_valid, Vf_valid, mu, nrev_valid)

    dJlev0, dJlevf = vinf_objective_gradients_rtn(
        "lev",
        V0_valid,
        Vf_valid,
        V_P0_valid,
        V_Pf_valid,
        rtn0,
        rtnf,
        lev_type=lev_type_norm,
    )

    coeffs_lev = solve_primer_coefficients(geometry, dJlev0, dJlevf)
    steering_coeffs = np.column_stack([coeffs_lev[key] for key in ("A", "B", "C", "D", "E", "F")])
    dep_component_grad0 = _cartesian_component_gradients_rtn(rtn0)
    arr_component_gradf = _cartesian_component_gradients_rtn(rtnf)
    zero_grad = np.zeros_like(dJlev0)

    dep_component_coeffs = [
        solve_primer_coefficients(geometry, dep_component_grad0[:, comp_idx, :], zero_grad)
        for comp_idx in range(3)
    ]
    arr_component_coeffs = [
        solve_primer_coefficients(geometry, zero_grad, arr_component_gradf[:, comp_idx, :])
        for comp_idx in range(3)
    ]

    component_coeffs = np.stack([
        np.column_stack([coeffs[key] for coeffs in dep_component_coeffs + arr_component_coeffs])
        for key in ("A", "B", "C", "D", "E", "F")
    ], axis=-1)

    n_valid = int(seed_idx.size)
    dVinf0_vec_da = np.empty((n_valid, 3), dtype=float)
    dVinff_vec_da = np.empty((n_valid, 3), dtype=float)
    d2Vinf0_vec_da2 = np.empty((n_valid, 3), dtype=float)
    d2Vinff_vec_da2 = np.empty((n_valid, 3), dtype=float)
    dJlev_da = np.empty(n_valid, dtype=float)
    profile_nu = [None] * n_valid if return_profile else []
    profile_primer = [None] * n_valid if return_profile else []
    profile_steering = [None] * n_valid if return_profile else []
    profile_norm = [None] * n_valid if return_profile else []
    profile_dt_dnu = [None] * n_valid if return_profile else []

    e = np.asarray(geometry["e"], dtype=float).reshape(-1)
    p_orb = np.asarray(geometry["p"], dtype=float).reshape(-1)
    h = np.asarray(geometry["h"], dtype=float).reshape(-1)
    nu0 = np.asarray(geometry["nu0"], dtype=float).reshape(-1)
    nuf = np.asarray(geometry["nuf"], dtype=float).reshape(-1)
    eps_norm = 1e-12

    for nrev in np.unique(nrev_valid):
        group = np.flatnonzero(nrev_valid == nrev)
        n_profile = max(2, num_nu * (1 + int(nrev)) + 1)
        # Bound temporary arrays by the total number of quadrature nodes.
        batch_size = max(1, 32768 // n_profile)
        for start in range(0, group.size, batch_size):
            idx = group[start:start + batch_size]
            nu_grid = np.ascontiguousarray(np.linspace(nu0[idx], nuf[idx], n_profile, axis=1))
            integrals, dJlev_da[idx], quadratic = _integrate_lt_quadrature_jit(
                nu_grid, e[idx], p_orb[idx], h[idx], steering_coeffs[idx],
                component_coeffs[idx], rtnf[idx])
            responses = np.einsum("bck,bk->bc", component_coeffs[idx], integrals)
            second = 2 * np.einsum("bck,bk->bc", component_coeffs[idx], quadratic)
            dVinf0_vec_da[idx], dVinff_vec_da[idx] = responses[:, :3], responses[:, 3:]
            d2Vinf0_vec_da2[idx], d2Vinff_vec_da2[idx] = second[:, :3], second[:, 3:]

            if return_profile:
                basis = _conic_primer_basis(nu_grid, e[idx, None])
                dt_dnu = np.square(p_orb[idx, None] / basis[3]) / h[idx, None]
                primer_lev = _primer_vector_from_basis(
                    basis, *(coeffs_lev[key][idx, None] for key in ("A", "B", "C", "D", "E", "F")))
                primer_norm = np.linalg.norm(primer_lev, axis=2)
                steering = np.zeros_like(primer_lev)
                np.divide(primer_lev, primer_norm[:, :, None], out=steering,
                          where=primer_norm[:, :, None] > eps_norm)

            if return_profile:
                for row, seed in enumerate(idx):
                    # Copies keep profiles in seed order without retaining the full field tensor.
                    profile_nu[seed] = nu_grid[row].copy()
                    profile_primer[seed] = primer_lev[row].copy()
                    profile_steering[seed] = steering[row].copy()
                    profile_norm[seed] = primer_norm[row].copy()
                    profile_dt_dnu[seed] = dt_dnu[row].copy()

    Vinf0_nom_valid = Vinf0_nom[seed_idx]
    Vinff_nom_valid = Vinff_nom[seed_idx]
    Jlev_nom = _jlev_from_vinfs(Vinf0_nom_valid, Vinff_nom_valid, lev_type_norm)
    Vinf0_hat_valid = _safe_unit_rows(Vinf0_nom_vec_valid)
    Vinff_hat_valid = _safe_unit_rows(Vinff_nom_vec_valid)
    dVinf0_da = np.einsum("ij,ij->i", Vinf0_hat_valid, dVinf0_vec_da)
    dVinff_da = np.einsum("ij,ij->i", Vinff_hat_valid, dVinff_vec_da)

    acc_row_vec = acc_grid.reshape(1, -1, 1)
    vinf0_vec = (Vinf0_nom_vec_valid[:, None, :] + dVinf0_vec_da[:, None, :] * acc_row_vec
                + 0.5 * d2Vinf0_vec_da2[:, None, :] * acc_row_vec**2)
    vinff_vec = (Vinff_nom_vec_valid[:, None, :] + dVinff_vec_da[:, None, :] * acc_row_vec
                + 0.5 * d2Vinff_vec_da2[:, None, :] * acc_row_vec**2)
    vinf0_speed, vinff_speed = np.linalg.norm(vinf0_vec, axis=2), np.linalg.norm(vinff_vec, axis=2)
    out = {
        "seed_idx": seed_idx.astype(int, copy=False),
        "lev_type": lev_type_norm,
        "acc_grid_km_s2": acc_grid.copy(),
        "Vinf0_nom_vec_km_s": Vinf0_nom_vec_valid,
        "Vinff_nom_vec_km_s": Vinff_nom_vec_valid,
        "Vinf0_nom_km_s": Vinf0_nom_valid,
        "Vinff_nom_km_s": Vinff_nom_valid,
        "Jlev_nom_km_s": Jlev_nom,
        "dVinf0_vec_da_s": dVinf0_vec_da,
        "dVinff_vec_da_s": dVinff_vec_da,
        "d2Vinf0_vec_da2_s3_km": d2Vinf0_vec_da2,
        "d2Vinff_vec_da2_s3_km": d2Vinff_vec_da2,
        "dVinf0_da_s": dVinf0_da,
        "dVinff_da_s": dVinff_da,
        "dJlev_da_s": dJlev_da,
        "Vinf0_vec_lt_km_s": vinf0_vec,
        "Vinff_vec_lt_km_s": vinff_vec,
        "Vinf0_lt_km_s": vinf0_speed,
        "Vinff_lt_km_s": vinff_speed,
        "Jlev_lt_km_s": _jlev_from_vinfs(vinf0_speed, vinff_speed, lev_type_norm),
    }
    if return_profile:
        out["nu_grid"] = profile_nu
        out["primer_rtn"] = profile_primer
        out["steering_rtn"] = profile_steering
        out["primer_norm"] = profile_norm
        out["dt_dnu_s"] = profile_dt_dnu
    return out

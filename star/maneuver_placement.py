"""
Landau, "Efficient Maneuver Placement for Automated Trajectory Design" (2022).

This maneuver placement algorithm answers:
   Given: R0, Rf, t0, tf, mu, J (objective), and DV magnitude
   Find: optimal maneuver location (theta_0m) and direction (dV_hat) that maximally reduces the objective.

See NOTATION.md for `obj_type`, `lev_type`, and the DSM leveraging symbols.
"""

import copy
import sys
from pathlib import Path

import numpy as np
import spiceypy as spice
import matplotlib.pyplot as plt
from typing import Callable, Tuple, Optional

from numba import njit, prange

sys.modules.setdefault("maneuver_placement", sys.modules[__name__])

from star.constants import DEFAULT_FRAME, DEFAULT_OBSERVER, SECONDS_PER_DAY, get_gm, get_radii
from star.lambert import kepler_propagate_universal, lambert_batch

AU_KM = 149597870.700

def calc_true_anomaly(evec, e, R, r, v):
    nu = np.arccos(np.dot(evec, R)/e/r)
    if np.dot(R, v) < 0:
        nu = 2 * np.pi - nu
    return nu

def calc_true_anom_from_rv(R0, v0, mu):
    r0 = np.linalg.norm(R0)
    energy = 1 / 2 * np.linalg.norm(v0) ** 2 - mu / r0
    h = np.cross(R0, v0)
    evec = np.cross(v0, h) / mu - R0 / r0
    e = np.linalg.norm(evec)
    nu0 = calc_true_anomaly(evec, e, R0, r0, v0)

    return np.rad2deg(nu0)


def _true_to_mean(nu, e, eps=1e-14):
    """Vectorized true->mean anomaly conversion for pairwise (nu_i, e_i)."""
    nu_arr = np.asarray(nu, dtype=float).reshape(-1)
    e_arr = np.asarray(e, dtype=float).reshape(-1)
    if nu_arr.shape != e_arr.shape:
        raise ValueError("nu and e must have matching shapes.")
    k = np.floor(nu_arr / (2.0 * np.pi))
    nu_mod = nu_arr - 2.0 * np.pi * k
    out = np.empty_like(nu_arr)
    circ = np.abs(e_arr) < float(eps)
    out[circ] = nu_arr[circ]
    noncirc = ~circ
    if np.any(noncirc):
        e_nc = e_arr[noncirc]
        nu_nc = nu_mod[noncirc]
        s = np.sin(0.5 * nu_nc)
        c = np.cos(0.5 * nu_nc)
        E = 2.0 * np.arctan2(np.sqrt(1.0 - e_nc) * s, np.sqrt(1.0 + e_nc) * c)
        E = np.where(E < 0.0, E + 2.0 * np.pi, E)
        M_mod = E - e_nc * np.sin(E)
        out[noncirc] = M_mod + 2.0 * np.pi * k[noncirc]
    return out


def I1(e, nu, M):
    tmp = 1/(1 - e**2) * ((2 * e)/(1 + e * np.cos(nu)) - np.cos(nu) - (3 * e**2 * M * np.sin(nu))/np.power(1 - e**2, 3/2))
    return tmp


def I2(e, nu, M):
    tmp = 1/(1 - e**2) * ((1 + 1/(1 + e * np.cos(nu))) * np.sin(nu) - (3 * e * M * (1 + e * np.cos(nu)))/np.power(1 - e**2, 3/2))
    return tmp

def rtn_to_catesian(r, v_batch):
    """Batch RTN->Cartesian direction cosine matrices for shared or per-row position."""
    r_in = np.asarray(r, dtype=float)
    v_arr = np.asarray(v_batch, dtype=float).reshape(-1, 3)
    n_row = int(v_arr.shape[0])

    if r_in.ndim == 1:
        if r_in.size != 3:
            raise ValueError("r must have 3 elements when provided as a vector.")
        r_arr = np.repeat(r_in.reshape(1, 3), n_row, axis=0)
    elif r_in.ndim == 2 and r_in.shape == (n_row, 3):
        r_arr = r_in
    else:
        raise ValueError("r must be shape (3,) or (N,3) matching v_batch rows.")

    r_norm = np.linalg.norm(r_arr, axis=1)
    if np.any(r_norm <= 0.0):
        raise ValueError("r must have positive norm.")
    R = r_arr / r_norm.reshape(-1, 1)
    h = np.cross(R, v_arr)
    h_norm = np.linalg.norm(h, axis=1)
    if np.any(h_norm <= 0.0):
        raise ValueError("Encountered degenerate angular momentum while building RTN frame.")
    N = h / h_norm.reshape(-1, 1)
    T = np.cross(N, R)
    return np.stack((R, T, N), axis=2)


def _broadcast_seed_vec3(name, value, n_seed):
    """Broadcast a shared or per-seed 3-vector array to shape (S,3)."""
    arr = np.asarray(value, dtype=float)
    if arr.ndim == 1:
        if arr.size != 3:
            raise ValueError(f"{name} must be shape (3,) or (S,3).")
        return np.repeat(arr.reshape(1, 3), int(n_seed), axis=0)
    if arr.ndim == 2 and arr.shape == (int(n_seed), 3):
        return arr
    raise ValueError(f"{name} must be shape (3,) or (S,3).")


def prepare_primer_geometry(
    R0,
    Rf,
    v0_batch,
    vf_batch,
    mu,
    num_revs_batch,
):
    """Prepare shared conic geometry for analytic primer-vector solves."""
    v0_arr = np.asarray(v0_batch, dtype=float).reshape(-1, 3)
    vf_arr = np.asarray(vf_batch, dtype=float).reshape(-1, 3)
    num_revs_arr = np.asarray(num_revs_batch, dtype=int).reshape(-1)

    n_seed = int(v0_arr.shape[0])
    if n_seed == 0:
        return {
            "R0": np.empty((0, 3), dtype=float),
            "Rf": np.empty((0, 3), dtype=float),
            "v0": np.empty((0, 3), dtype=float),
            "vf": np.empty((0, 3), dtype=float),
            "num_revs": np.empty(0, dtype=int),
            "r0": np.empty(0, dtype=float),
            "rf": np.empty(0, dtype=float),
            "h_vec": np.empty((0, 3), dtype=float),
            "h": np.empty(0, dtype=float),
            "energy": np.empty(0, dtype=float),
            "a_semimajor": np.empty(0, dtype=float),
            "evec": np.empty((0, 3), dtype=float),
            "e": np.empty(0, dtype=float),
            "nu0": np.empty(0, dtype=float),
            "nuf": np.empty(0, dtype=float),
            "M0": np.empty(0, dtype=float),
            "Mf": np.empty(0, dtype=float),
            "p": np.empty(0, dtype=float),
            "r0_p": np.empty(0, dtype=float),
            "rf_p": np.empty(0, dtype=float),
            "p_r0": np.empty(0, dtype=float),
            "p_rf": np.empty(0, dtype=float),
        }
    if vf_arr.shape[0] != n_seed:
        raise ValueError("v0_batch and vf_batch must have matching row count.")
    if num_revs_arr.shape != (n_seed,):
        raise ValueError("num_revs_batch must have shape (S,).")
    if np.any(num_revs_arr < 0):
        raise ValueError("num_revs must be >= 0.")

    mu = float(mu)
    if mu <= 0.0:
        raise ValueError("mu must be positive.")

    R0_arr = _broadcast_seed_vec3("R0", R0, n_seed)
    Rf_arr = _broadcast_seed_vec3("Rf", Rf, n_seed)

    r0 = np.linalg.norm(R0_arr, axis=1)
    rf = np.linalg.norm(Rf_arr, axis=1)
    if np.any(r0 <= 0.0) or np.any(rf <= 0.0):
        raise ValueError("R0 and Rf must have positive norm.")

    h_vec = np.cross(R0_arr, v0_arr)
    h = np.linalg.norm(h_vec, axis=1)
    if np.any(h <= 0.0):
        raise ValueError("Encountered degenerate angular momentum while preparing primer geometry.")

    energy = 0.5 * np.einsum("ij,ij->i", v0_arr, v0_arr) - mu / r0
    a_semimajor = -mu / (2.0 * energy)
    evec = np.cross(v0_arr, h_vec) / mu - R0_arr / r0.reshape(-1, 1)
    e = np.linalg.norm(evec, axis=1)

    e_safe = np.maximum(e, 1e-15)
    nu0 = np.arccos(np.clip(np.einsum("ij,ij->i", evec, R0_arr) / (e_safe * r0), -1.0, 1.0))
    nu0 = np.where(np.einsum("ij,ij->i", v0_arr, R0_arr) < 0.0, 2.0 * np.pi - nu0, nu0)
    nuf_wrapped = np.arccos(np.clip(np.einsum("ij,ij->i", evec, Rf_arr) / (e_safe * rf), -1.0, 1.0))
    nuf_wrapped = np.where(np.einsum("ij,ij->i", vf_arr, Rf_arr) < 0.0, 2.0 * np.pi - nuf_wrapped, nuf_wrapped)

    nuf = np.where(
        nuf_wrapped < nu0,
        nuf_wrapped + 2.0 * np.pi * (1.0 + num_revs_arr.astype(float)),
        nuf_wrapped + 2.0 * np.pi * num_revs_arr.astype(float),
    )
    if np.any(nuf < nu0):
        raise ValueError("nuf must be >= nu0.")

    M0 = _true_to_mean(nu0, e)
    Mf = _true_to_mean(nuf, e)
    if np.any(Mf < M0):
        raise ValueError("Mean anomaly must be non-decreasing along the transfer.")

    r0_p = 1.0 / (1.0 + e * np.cos(nu0))
    rf_p = 1.0 / (1.0 + e * np.cos(nuf))
    p_r0 = 1.0 / r0_p
    p_rf = 1.0 / rf_p
    p = (h * h) / mu

    return {
        "R0": R0_arr,
        "Rf": Rf_arr,
        "v0": v0_arr,
        "vf": vf_arr,
        "num_revs": num_revs_arr,
        "r0": r0,
        "rf": rf,
        "h_vec": h_vec,
        "h": h,
        "energy": energy,
        "a_semimajor": a_semimajor,
        "evec": evec,
        "e": e,
        "nu0": nu0,
        "nuf": nuf,
        "M0": M0,
        "Mf": Mf,
        "p": p,
        "r0_p": r0_p,
        "rf_p": rf_p,
        "p_r0": p_r0,
        "p_rf": p_rf,
    }


def solve_primer_coefficients(primer_geometry, dJ_dV0_batch, dJ_dVf_batch):
    """Solve analytic conic primer coefficients for one batch of endpoint gradients."""
    geometry = dict(primer_geometry)
    e = np.asarray(geometry["e"], dtype=float).reshape(-1)
    nu0 = np.asarray(geometry["nu0"], dtype=float).reshape(-1)
    nuf = np.asarray(geometry["nuf"], dtype=float).reshape(-1)
    M0 = np.asarray(geometry["M0"], dtype=float).reshape(-1)
    Mf = np.asarray(geometry["Mf"], dtype=float).reshape(-1)
    r0_p = np.asarray(geometry["r0_p"], dtype=float).reshape(-1)
    rf_p = np.asarray(geometry["rf_p"], dtype=float).reshape(-1)
    p_r0 = np.asarray(geometry["p_r0"], dtype=float).reshape(-1)
    p_rf = np.asarray(geometry["p_rf"], dtype=float).reshape(-1)

    n_seed = int(e.shape[0])
    dJ0_arr = _broadcast_seed_vec3("dJ_dV0_batch", dJ_dV0_batch, n_seed)
    dJf_arr = _broadcast_seed_vec3("dJ_dVf_batch", dJ_dVf_batch, n_seed)

    a_rt = np.zeros((n_seed, 4, 4), dtype=float)
    a_rt[:, 0, 0] = np.cos(nu0)
    a_rt[:, 0, 1] = -e * np.sin(nu0)
    a_rt[:, 0, 2] = I1(e, nu0, M0)
    a_rt[:, 1, 0] = -(1.0 + r0_p) * np.sin(nu0)
    a_rt[:, 1, 1] = p_r0
    a_rt[:, 1, 2] = I2(e, nu0, M0)
    a_rt[:, 1, 3] = r0_p
    a_rt[:, 2, 0] = np.cos(nuf)
    a_rt[:, 2, 1] = -e * np.sin(nuf)
    a_rt[:, 2, 2] = I1(e, nuf, Mf)
    a_rt[:, 3, 0] = -(1.0 + rf_p) * np.sin(nuf)
    a_rt[:, 3, 1] = p_rf
    a_rt[:, 3, 2] = I2(e, nuf, Mf)
    a_rt[:, 3, 3] = rf_p

    b_rt = np.column_stack((dJ0_arr[:, 0], dJ0_arr[:, 1], -dJf_arr[:, 0], -dJf_arr[:, 1]))

    a_n = np.zeros((n_seed, 2, 2), dtype=float)
    a_n[:, 0, 0] = r0_p * np.cos(nu0)
    a_n[:, 0, 1] = r0_p * np.sin(nu0)
    a_n[:, 1, 0] = rf_p * np.cos(nuf)
    a_n[:, 1, 1] = rf_p * np.sin(nuf)
    b_n = np.column_stack((dJ0_arr[:, 2], -dJf_arr[:, 2]))

    if not (
        np.all(np.isfinite(a_rt))
        and np.all(np.isfinite(b_rt))
        and np.all(np.isfinite(a_n))
        and np.all(np.isfinite(b_n))
    ):
        raise ValueError("Non-finite values found in maneuver-placement systems.")

    try:
        ABCD = np.linalg.solve(a_rt, b_rt.reshape(n_seed, 4, 1)).reshape(n_seed, 4)
        EF = np.linalg.solve(a_n, b_n.reshape(n_seed, 2, 1)).reshape(n_seed, 2)
    except np.linalg.LinAlgError as exc:
        raise RuntimeError("Failed to solve primer-vector linear systems for one or more Lambert seeds.") from exc

    return {
        "A": ABCD[:, 0],
        "B": ABCD[:, 1],
        "C": ABCD[:, 2],
        "D": ABCD[:, 3],
        "E": EF[:, 0],
        "F": EF[:, 1],
    }


def primer_vector_from_coeffs(nu, e, A, B, C, D, E, F):
    """Evaluate the analytic conic primer vector in RTN coordinates."""
    nu_b, e_b, A_b, B_b, C_b, D_b, E_b, F_b = np.broadcast_arrays(
        np.asarray(nu, dtype=float),
        np.asarray(e, dtype=float),
        np.asarray(A, dtype=float),
        np.asarray(B, dtype=float),
        np.asarray(C, dtype=float),
        np.asarray(D, dtype=float),
        np.asarray(E, dtype=float),
        np.asarray(F, dtype=float),
    )
    nu_flat = nu_b.reshape(-1)
    e_flat = e_b.reshape(-1)
    M_flat = _true_to_mean(nu_flat, e_flat)
    sin_nu = np.sin(nu_flat)
    cos_nu = np.cos(nu_flat)
    denom = 1.0 + e_flat * cos_nu

    p_R = A_b.reshape(-1) * cos_nu - B_b.reshape(-1) * e_flat * sin_nu + C_b.reshape(-1) * I1(e_flat, nu_flat, M_flat)
    p_T = (
        -A_b.reshape(-1) * sin_nu
        + B_b.reshape(-1) * (1.0 + e_flat * cos_nu)
        + (D_b.reshape(-1) - A_b.reshape(-1) * sin_nu) / denom
        + C_b.reshape(-1) * I2(e_flat, nu_flat, M_flat)
    )
    p_N = (E_b.reshape(-1) * cos_nu + F_b.reshape(-1) * sin_nu) / denom
    return np.stack((p_R, p_T, p_N), axis=1).reshape(nu_b.shape + (3,))


def _vinf_objective_weights(objective, lev_type="+-"):
    objective = str(objective).strip().lower()
    lev_type = str(lev_type).strip()
    if objective == "lev":
        if lev_type == "+-":
            return -1.0, 1.0
        if lev_type == "-+":
            return 1.0, -1.0
        if lev_type == "++":
            return -1.0, -1.0
        if lev_type == "--":
            return 1.0, 1.0
        raise ValueError("Invalid lev_type. Choose '+-', '-+', '++', or '--'.")
    if objective == "dep":
        return 1.0, 0.0
    if objective == "arr":
        return 0.0, 1.0
    raise ValueError("objective must be one of {'lev', 'dep', 'arr'}.")


def vinf_objective_gradients_rtn(
    objective,
    V0_seed,
    Vf_seed,
    V_P0,
    V_Pf,
    rtn_to_cart_0_batch,
    rtn_to_cart_f_batch,
    lev_type="+-",
):
    """Return RTN endpoint gradients for V-infinity objectives."""
    V0_arr = np.asarray(V0_seed, dtype=float).reshape(-1, 3)
    Vf_arr = np.asarray(Vf_seed, dtype=float).reshape(-1, 3)
    rtn0 = np.asarray(rtn_to_cart_0_batch, dtype=float).reshape(-1, 3, 3)
    rtnf = np.asarray(rtn_to_cart_f_batch, dtype=float).reshape(-1, 3, 3)
    if V0_arr.shape[0] != Vf_arr.shape[0] or V0_arr.shape[0] != rtn0.shape[0] or V0_arr.shape[0] != rtnf.shape[0]:
        raise ValueError("Objective gradient batch arrays must have matching row counts.")

    n_seed = int(V0_arr.shape[0])
    V_P0_arr = _broadcast_seed_vec3("V_P0", V_P0, n_seed)
    V_Pf_arr = _broadcast_seed_vec3("V_Pf", V_Pf, n_seed)

    w_dep, w_arr = _vinf_objective_weights(objective, lev_type=lev_type)
    eps = 1e-12
    dV0 = V0_arr - V_P0_arr
    dVf = Vf_arr - V_Pf_arr
    dJ0_cart = w_dep * dV0 / np.maximum(np.linalg.norm(dV0, axis=1, keepdims=True), eps)
    dJf_cart = w_arr * dVf / np.maximum(np.linalg.norm(dVf, axis=1, keepdims=True), eps)
    dJ0_rtn = np.einsum("nji,nj->ni", rtn0, dJ0_cart)
    dJf_rtn = np.einsum("nji,nj->ni", rtnf, dJf_cart)
    return dJ0_rtn, dJf_rtn


@njit(cache=True, fastmath=True)
def _true_to_mean_scalar_jit(nu, e):
    """Scalar true -> mean anomaly (matches _true_to_mean for a single element)."""
    two_pi = 2.0 * np.pi
    k = np.floor(nu / two_pi)
    nu_mod = nu - two_pi * k
    if np.abs(e) < 1e-14:
        return nu
    s = np.sin(0.5 * nu_mod)
    c = np.cos(0.5 * nu_mod)
    E = 2.0 * np.arctan2(np.sqrt(1.0 - e) * s, np.sqrt(1.0 + e) * c)
    if E < 0.0:
        E += two_pi
    return E - e * np.sin(E) + two_pi * k


@njit(cache=True, fastmath=True)
def _I1_scalar_jit(e, nu, M):
    e2 = e * e
    return (1.0 / (1.0 - e2)) * (
        (2.0 * e) / (1.0 + e * np.cos(nu))
        - np.cos(nu)
        - (3.0 * e2 * M * np.sin(nu)) / (1.0 - e2) ** 1.5
    )


@njit(cache=True, fastmath=True)
def _I2_scalar_jit(e, nu, M):
    e2 = e * e
    ecos = e * np.cos(nu)
    return (1.0 / (1.0 - e2)) * (
        (1.0 + 1.0 / (1.0 + ecos)) * np.sin(nu)
        - (3.0 * e * M * (1.0 + ecos)) / (1.0 - e2) ** 1.5
    )


@njit(cache=True, fastmath=True)
def _primer_norm_sq_grad_hess_scalar_jit(nu, ei, Ai, Bi, Ci, Di, Ei, Fi):
    """Return (phi, dphi, d2phi) = (|p|², d|p|²/dν, d²|p|²/dν²) for one point."""
    M = _true_to_mean_scalar_jit(nu, ei)
    sin_nu = np.sin(nu)
    cos_nu = np.cos(nu)
    denom  = 1.0 + ei * cos_nu
    r_p    = 1.0 / denom

    p_R = Ai * cos_nu - Bi * ei * sin_nu + Ci * _I1_scalar_jit(ei, nu, M)
    p_T = (
        -Ai * sin_nu
        + Bi * (1.0 + ei * cos_nu)
        + (Di - Ai * sin_nu) * r_p
        + Ci * _I2_scalar_jit(ei, nu, M)
    )
    p_N = (Ei * cos_nu + Fi * sin_nu) * r_p

    r_p2 = r_p * r_p
    r_p3 = r_p2 * r_p
    dp_R  = p_T * (1.0 - r_p) - (Ai * sin_nu - Ci * sin_nu + Di * ei * cos_nu) * r_p2
    dp_T  = (Ai * ei + p_T * ei * sin_nu + Di * ei * cos_nu) * r_p - 2.0 * p_R
    dp_N  = (-Ei * sin_nu + ei * Fi + Fi * cos_nu) * r_p2
    d2p_R = 2.0 * ei * Ci * r_p3 - p_R
    d2p_T = p_T * (1.0 - r_p) + 2.0 * ei * sin_nu * (dp_T + p_R) * r_p - 2.0 * dp_R
    d2p_N = (2.0 * ei * dp_N * sin_nu - p_N) * r_p

    phi   = p_R * p_R   + p_T * p_T   + p_N * p_N
    dphi  = p_R * dp_R  + p_T * dp_T  + p_N * dp_N
    d2phi = (dp_R * dp_R + dp_T * dp_T + dp_N * dp_N
             + p_R * d2p_R + p_T * d2p_T + p_N * d2p_N)
    return phi, dphi, d2phi


@njit(cache=True, fastmath=True, parallel=True)
def _maximize_primer_norm_jit(lb, ub, seed_idx, A, B, C, D, E, F, e,
                               max_iter=24, step_tol=1e-12):
    """Fused parallel primer-norm maximiser.

    For each interval [lb[i], ub[i]] belonging to seed seed_idx[i], runs a
    bounded Newton ascent on |p(ν)|² entirely in scalar JIT ops, then picks
    the best of the converged interior point and the two boundary endpoints.

    Replaces the Python loop in _maximize_primer_norm_bounded_flat (24 iters × 
    3 numpy batch calls each) with a single prange dispatch.
    """
    n = lb.shape[0]
    out = np.empty(n, dtype=np.float64)

    for i in prange(n):
        si  = seed_idx[i]
        lo  = lb[i]
        hi  = ub[i]
        ei  = e[si]
        Ai  = A[si]; Bi = B[si]; Ci = C[si]
        Di  = D[si]; Ei = E[si]; Fi = F[si]

        x     = 0.5 * (lo + hi)
        width = hi - lo

        for _ in range(max_iter):
            phi, dphi, d2phi = _primer_norm_sq_grad_hess_scalar_jit(x, ei, Ai, Bi, Ci, Di, Ei, Fi)

            step = 0.0
            if np.isfinite(phi) and np.isfinite(dphi) and np.isfinite(d2phi) and np.abs(d2phi) > 1e-12:
                step = dphi / d2phi
            half_w = 0.5 * width
            if step >  half_w: step =  half_w
            if step < -half_w: step = -half_w
            x_new = x - step
            if x_new < lo: x_new = lo
            if x_new > hi: x_new = hi

            phi_new, _, _ = _primer_norm_sq_grad_hess_scalar_jit(x_new, ei, Ai, Bi, Ci, Di, Ei, Fi)

            if np.isfinite(dphi) and dphi >= 0.0:
                x_alt = 0.5 * (x + hi)
            else:
                x_alt = 0.5 * (x + lo)
            phi_alt, _, _ = _primer_norm_sq_grad_hess_scalar_jit(x_alt, ei, Ai, Bi, Ci, Di, Ei, Fi)

            phi_ok = np.isfinite(phi)
            choose_new = np.isfinite(phi_new) and (phi_new >= phi or not phi_ok)
            choose_alt = (not choose_new) and np.isfinite(phi_alt) and (phi_alt >= phi or not phi_ok)
            if choose_new:
                x_next = x_new
            elif choose_alt:
                x_next = x_alt
            else:
                x_next = x

            if np.abs(x_next - x) <= step_tol:
                x = x_next
                break
            x = x_next

        # Pick the best of interior point and both endpoints
        phi_lo, _, _ = _primer_norm_sq_grad_hess_scalar_jit(lo, ei, Ai, Bi, Ci, Di, Ei, Fi)
        phi_hi, _, _ = _primer_norm_sq_grad_hess_scalar_jit(hi, ei, Ai, Bi, Ci, Di, Ei, Fi)
        phi_x,  _, _ = _primer_norm_sq_grad_hess_scalar_jit(x,  ei, Ai, Bi, Ci, Di, Ei, Fi)
        if not np.isfinite(phi_lo): phi_lo = -np.inf
        if not np.isfinite(phi_hi): phi_hi = -np.inf
        if not np.isfinite(phi_x):  phi_x  = -np.inf

        if phi_lo >= phi_hi and phi_lo >= phi_x:
            out[i] = lo
        elif phi_hi >= phi_x:
            out[i] = hi
        else:
            out[i] = x

    return out


def maneuver_placement(
    R0,
    Rf,
    v0_batch,
    vf_batch,
    mu,
    dJ_dV0_batch,
    dJ_dVf_batch,
    num_revs_batch,
    debug=False,
):
    """
    Vectorized maneuver-placement solver across Lambert seeds.

    Inputs are the same as `maneuver_placement`, except batched over seeds:
      R0, Rf: shape (3,) shared or shape (S,3) per-seed
      v0_batch, vf_batch, dJ_dV0_batch, dJ_dVf_batch: shape (S,3)
      num_revs_batch: shape (S,)
    """
    v0_arr = np.asarray(v0_batch, dtype=float).reshape(-1, 3)
    vf_arr = np.asarray(vf_batch, dtype=float).reshape(-1, 3)
    dJ0_arr = np.asarray(dJ_dV0_batch, dtype=float).reshape(-1, 3)
    dJf_arr = np.asarray(dJ_dVf_batch, dtype=float).reshape(-1, 3)

    n_seed = int(v0_arr.shape[0])
    if n_seed == 0:
        empty_f = np.empty(0, dtype=float)
        empty_v = np.empty((0, 3), dtype=float)
        return empty_f, empty_f.copy(), empty_v, empty_f.copy()
    if vf_arr.shape[0] != n_seed or dJ0_arr.shape[0] != n_seed or dJf_arr.shape[0] != n_seed:
        raise ValueError("All batched inputs must have matching row count.")
    geometry = prepare_primer_geometry(R0, Rf, v0_arr, vf_arr, mu, num_revs_batch)
    coeffs = solve_primer_coefficients(geometry, dJ0_arr, dJf_arr)

    e = np.asarray(geometry["e"], dtype=float).reshape(-1)
    nu0 = np.asarray(geometry["nu0"], dtype=float).reshape(-1)
    nuf = np.asarray(geometry["nuf"], dtype=float).reshape(-1)
    p_orb = np.asarray(geometry["p"], dtype=float).reshape(-1)
    num_revs_arr = np.asarray(geometry["num_revs"], dtype=int).reshape(-1)

    A = np.asarray(coeffs["A"], dtype=float).reshape(-1)
    B = np.asarray(coeffs["B"], dtype=float).reshape(-1)
    C = np.asarray(coeffs["C"], dtype=float).reshape(-1)
    D = np.asarray(coeffs["D"], dtype=float).reshape(-1)
    E = np.asarray(coeffs["E"], dtype=float).reshape(-1)
    F = np.asarray(coeffs["F"], dtype=float).reshape(-1)

    max_num_revs = int(np.max(num_revs_arr))
    ndiv = max(5, 5 * (1 + max_num_revs))
    grid_t = np.linspace(0.0, 1.0, ndiv + 1)
    nu_grid = nu0.reshape(-1, 1) + (nuf - nu0).reshape(-1, 1) * grid_t.reshape(1, -1)
    lb_all = nu_grid[:, :-1].reshape(-1)
    ub_all = nu_grid[:, 1:].reshape(-1)
    seed_idx_int = np.repeat(np.arange(n_seed, dtype=int), ndiv)
    valid_int = ub_all > lb_all
    if np.any(valid_int):
        nu_refined = _maximize_primer_norm_jit(
            lb_all[valid_int].astype(np.float64, copy=False),
            ub_all[valid_int].astype(np.float64, copy=False),
            seed_idx_int[valid_int].astype(np.int64, copy=False),
            A.astype(np.float64, copy=False),
            B.astype(np.float64, copy=False),
            C.astype(np.float64, copy=False),
            D.astype(np.float64, copy=False),
            E.astype(np.float64, copy=False),
            F.astype(np.float64, copy=False),
            e.astype(np.float64, copy=False),
        )
        seed_refined = seed_idx_int[valid_int]
    else:
        nu_refined = np.empty(0, dtype=float)
        seed_refined = np.empty(0, dtype=int)

    nu_boundary = np.column_stack((nu0, nuf)).reshape(-1)
    seed_boundary = np.repeat(np.arange(n_seed, dtype=int), 2)

    k_start = np.ceil(nu0 / np.pi).astype(int)
    k_end = np.floor(nuf / np.pi).astype(int)
    if int(np.max(k_end)) >= int(np.min(k_start)):
        k_vals = np.arange(int(np.min(k_start)), int(np.max(k_end)) + 1, dtype=int)
        k_mask = (k_vals.reshape(1, -1) >= k_start.reshape(-1, 1)) & (k_vals.reshape(1, -1) <= k_end.reshape(-1, 1))
        seed_pi, col_pi = np.where(k_mask)
        nu_pi = np.pi * k_vals[col_pi]
        seed_pi = seed_pi.astype(int, copy=False)
    else:
        nu_pi = np.empty(0, dtype=float)
        seed_pi = np.empty(0, dtype=int)

    nu_candidates = np.concatenate((nu_boundary, nu_pi, nu_refined))
    seed_candidates = np.concatenate((seed_boundary, seed_pi, seed_refined)).astype(int, copy=False)

    p_all = primer_vector_from_coeffs(
        nu_candidates,
        e[seed_candidates],
        A[seed_candidates],
        B[seed_candidates],
        C[seed_candidates],
        D[seed_candidates],
        E[seed_candidates],
        F[seed_candidates],
    ).reshape(-1, 3)
    finite_rows = np.all(np.isfinite(p_all), axis=1)
    p_norm_all = np.linalg.norm(p_all, axis=1)
    p_norm_all = np.where(finite_rows, p_norm_all, -np.inf)
    best_norm = np.full(n_seed, -np.inf, dtype=float)
    np.maximum.at(best_norm, seed_candidates, p_norm_all)
    if np.any(best_norm <= -np.inf):
        raise RuntimeError("Failed to find a finite primer-vector maximizer for one or more seeds.")

    order = np.lexsort((p_norm_all, seed_candidates))
    seeds_sorted = seed_candidates[order]
    group_end = np.r_[seeds_sorted[1:] != seeds_sorted[:-1], True]
    best_flat_idx = order[group_end]
    best_seed = seed_candidates[best_flat_idx]
    sort_seed = np.argsort(best_seed)
    best_flat_idx = best_flat_idx[sort_seed]
    best_seed_sorted = best_seed[sort_seed]
    if best_seed_sorted.size != n_seed or not np.array_equal(best_seed_sorted, np.arange(n_seed, dtype=int)):
        raise RuntimeError("Failed to recover one primer-vector maximizer per seed.")

    nu_opt = nu_candidates[best_flat_idx]
    p_opt = p_all[best_flat_idx]
    p_opt_norm = np.linalg.norm(p_opt, axis=1)
    if np.any(~np.isfinite(p_opt_norm)) or np.any(p_opt_norm <= 0.0):
        raise RuntimeError("Non-finite primer-vector maximum encountered.")

    theta_0m = nu_opt - nu0
    theta_0f = nuf - nu0
    dV_hat = p_opt / p_opt_norm.reshape(-1, 1)

    r_m = p_orb / (1.0 + e * np.cos(nu_opt))
    V_T1_guess = np.sqrt(p_orb * float(mu)) / r_m

    if np.any(~np.isfinite(theta_0m)) or np.any(~np.isfinite(theta_0f)) or np.any(~np.isfinite(dV_hat)) or np.any(~np.isfinite(V_T1_guess)):
        raise RuntimeError("Non-finite maneuver-placement result encountered.")

    if debug:
        print(f"maneuver_placement: solved {n_seed} Lambert seeds.")

    return theta_0m, theta_0f, dV_hat, V_T1_guess


@njit(cache=True, fastmath=True)
def _arctan_safe(num, denom):
    """arctan(num/denom) safe against denom==0 (returns ±π/2)."""
    tiny = 1.0e-14
    if abs(denom) <= tiny:
        return np.pi * 0.5 if num >= 0.0 else -np.pi * 0.5
    return np.arctan(num / denom)


@njit(cache=True, fastmath=True)
def _compute_tof_diff_scalar(
    V_T1,
    dV_N_i, dV_T_i, dV_R_i,
    cos_theta_0m_i, sin_theta_0m_i, tan_theta_0m_i, tan_theta_0m_2_i,
    sin_theta_0f_i, cos_theta_0f_i,
    n1_base_i, n2_base_i,
    r0_i, rf_i, given_tof_i, mu,
):
    """Scalar TOF residual for one DSM case. Returns (tof_diff, valid).

    All scalar float divisions have explicit zero-checks (Numba raises
    ZeroDivisionError on float/0, unlike NumPy which returns ±inf).
    """
    tiny = 1.0e-14
    penalty = 1.0e6
    two_pi = 2.0 * np.pi

    # Guard degenerate maneuver placement at θ_0m = 0 or π
    if abs(sin_theta_0m_i) <= tiny:
        return penalty, False

    x = np.sqrt((dV_N_i * cos_theta_0m_i) ** 2 + (dV_T_i + V_T1) ** 2)
    i0 = 2.0 * np.arctan2(x - dV_N_i * cos_theta_0m_i, dV_T_i + V_T1)

    denom_tmp = x * sin_theta_0f_i
    if abs(denom_tmp) <= tiny:
        return penalty, False
    tmp_arg = (dV_N_i * cos_theta_0f_i * sin_theta_0m_i) / denom_tmp
    if tmp_arg > 1.0:
        tmp_arg = 1.0
    elif tmp_arg < -1.0:
        tmp_arg = -1.0
    tmp = np.arccos(tmp_arg)
    if np.cos(i0 + tmp) > np.cos(i0 - tmp):
        i0 = i0 + tmp
    else:
        i0 = i0 - tmp

    if not np.isfinite(i0):
        return penalty, False

    cos_i0 = np.cos(i0)
    cos_theta_mf = (
        cos_theta_0f_i * cos_theta_0m_i
        + sin_theta_0f_i * sin_theta_0m_i * cos_i0
    )
    if cos_theta_mf > 1.0:
        cos_theta_mf = 1.0
    elif cos_theta_mf < -1.0:
        cos_theta_mf = -1.0
    sin_theta_mf = np.sqrt(max(0.0, 1.0 - cos_theta_mf * cos_theta_mf))

    denom_tmp2 = (
        sin_theta_0f_i * cos_theta_0m_i * cos_i0
        - cos_theta_0f_i * sin_theta_0m_i
    )
    if abs(denom_tmp2) <= tiny:
        return penalty, False
    tmp2 = (V_T1 + dV_T_i) / denom_tmp2
    if tmp2 < 0.0:
        sin_theta_mf = -sin_theta_mf
    V_T2 = tmp2 * sin_theta_mf

    theta_mf = np.arctan2(sin_theta_mf, cos_theta_mf)
    if theta_mf < 0.0:
        theta_mf += two_pi

    if not np.isfinite(theta_mf):
        return penalty, False
    if not np.isfinite(V_T2):
        return penalty, False
    if abs(sin_theta_mf) <= tiny:
        return penalty, False
    if abs(V_T1) <= tiny:
        return penalty, False
    if abs(V_T2) <= tiny:
        return penalty, False
    # tan_theta_0m_i == 0 iff sin_theta_0m_i == 0, already checked above.
    # np.tan(theta_mf) == 0 iff sin_theta_mf == 0, already checked above.

    a = -(V_T1 / (r0_i * sin_theta_0m_i) + V_T2 / (rf_i * sin_theta_mf))
    b = V_T1 / tan_theta_0m_i + V_T2 * (cos_theta_mf / sin_theta_mf) - dV_R_i
    c = mu / V_T1 * tan_theta_0m_2_i + mu / V_T2 * np.tan(theta_mf * 0.5)

    disc = b * b - 4.0 * a * c
    if not np.isfinite(disc) or disc < 0.0:
        return penalty, False
    if abs(2.0 * a) <= tiny:
        return penalty, False

    sqrt_disc = np.sqrt(disc)
    denom_root = 2.0 * a
    r1 = (-b + sqrt_disc) / denom_root
    r2 = (-b - sqrt_disc) / denom_root
    r_m = r1 if r1 > 0.0 else r2
    if not np.isfinite(r_m) or r_m <= 0.0:
        return penalty, False

    h1 = r_m * V_T1
    V_R0 = -V_T1 / sin_theta_0m_i + h1 / (r0_i * tan_theta_0m_i) + mu / h1 * tan_theta_0m_2_i
    V_R1 = h1 / (r0_i * sin_theta_0m_i) - V_T1 / tan_theta_0m_i - mu / h1 * tan_theta_0m_2_i
    alpha_1 = 2.0 * mu / r0_i - V_R0 * V_R0 - h1 * h1 / (r0_i * r0_i)
    if not np.isfinite(alpha_1) or alpha_1 <= 0.0:
        return penalty, False

    sqrt_alpha_1 = np.sqrt(alpha_1)
    chi1_arg = _arctan_safe(
        sqrt_alpha_1 * r0_i * tan_theta_0m_2_i,
        h1 - V_R0 * r0_i * tan_theta_0m_2_i,
    )
    chi1 = (2.0 / sqrt_alpha_1) * chi1_arg
    N1 = n1_base_i if chi1 > 0.0 else n1_base_i + 1.0
    t_0m = (
        (mu * chi1 + V_R0 * r0_i - V_R1 * r_m) / alpha_1
        + two_pi * N1 * mu / np.sqrt(alpha_1 * alpha_1 * alpha_1)
    )
    if not np.isfinite(t_0m) or t_0m <= 0.0:
        return penalty, False

    h2 = r_m * V_T2
    tan_half_theta_mf = np.tan(theta_mf * 0.5)
    V_Rf = (
        V_T2 / sin_theta_mf
        - h2 / (rf_i * np.tan(theta_mf))
        - mu / h2 * tan_half_theta_mf
    )
    alpha_2 = 2.0 * mu / rf_i - V_Rf * V_Rf - h2 * h2 / (rf_i * rf_i)
    if not np.isfinite(alpha_2) or alpha_2 <= 0.0:
        return penalty, False

    sqrt_alpha_2 = np.sqrt(alpha_2)
    chi2_arg = _arctan_safe(
        sqrt_alpha_2 * rf_i * tan_half_theta_mf,
        h2 + V_Rf * rf_i * tan_half_theta_mf,
    )
    chi2 = (2.0 / sqrt_alpha_2) * chi2_arg
    N2 = n2_base_i if chi2 > 0.0 else n2_base_i + 1.0
    t_mf = (
        (mu * chi2 + (V_R1 + dV_R_i) * r_m - V_Rf * rf_i) / alpha_2
        + two_pi * N2 * mu / np.sqrt(alpha_2 * alpha_2 * alpha_2)
    )
    if not np.isfinite(t_mf) or t_mf <= 0.0:
        return penalty, False

    return (t_0m + t_mf) - given_tof_i, True


@njit(cache=True, parallel=True, fastmath=True)
def _solve_velocity_jit(
    dV_N, dV_T, dV_R,
    cos_theta_0m, sin_theta_0m, tan_theta_0m, tan_theta_0m_2,
    sin_theta_0f, cos_theta_0f,
    n1_base, n2_base,
    r0, rf, given_tof,
    mu,
    V_T1_init,
    max_step,
    max_iter=60,
    tol=1.0e-10,
    min_deriv=1.0e-12,
):
    """Parallel quasi-second-order Newton root-solve for solve_velocity.

    Each case runs an independent scalar Newton loop via prange.
    Finite-difference derivatives match compute_tof_diff_with_derivatives_batch.
    """
    n = V_T1_init.shape[0]
    V_T1_out = np.empty(n, dtype=np.float64)
    succ_out = np.zeros(n, dtype=np.bool_)

    for idx in prange(n):
        dV_N_i          = dV_N[idx]
        dV_T_i          = dV_T[idx]
        dV_R_i          = dV_R[idx]
        cos_theta_0m_i  = cos_theta_0m[idx]
        sin_theta_0m_i  = sin_theta_0m[idx]
        tan_theta_0m_i  = tan_theta_0m[idx]
        tan_theta_0m_2_i = tan_theta_0m_2[idx]
        sin_theta_0f_i  = sin_theta_0f[idx]
        cos_theta_0f_i  = cos_theta_0f[idx]
        n1_base_i       = n1_base[idx]
        n2_base_i       = n2_base[idx]
        r0_i            = r0[idx]
        rf_i            = rf[idx]
        given_tof_i     = given_tof[idx]

        x = V_T1_init[idx]

        # Initial error + finite-difference derivative
        eps0 = max(abs(x) * 1.0e-6, 1.0e-8)
        e, _ = _compute_tof_diff_scalar(
            x, dV_N_i, dV_T_i, dV_R_i,
            cos_theta_0m_i, sin_theta_0m_i, tan_theta_0m_i, tan_theta_0m_2_i,
            sin_theta_0f_i, cos_theta_0f_i,
            n1_base_i, n2_base_i, r0_i, rf_i, given_tof_i, mu,
        )
        e_h, _ = _compute_tof_diff_scalar(
            x + eps0, dV_N_i, dV_T_i, dV_R_i,
            cos_theta_0m_i, sin_theta_0m_i, tan_theta_0m_i, tan_theta_0m_2_i,
            sin_theta_0f_i, cos_theta_0f_i,
            n1_base_i, n2_base_i, r0_i, rf_i, given_tof_i, mu,
        )
        ep = (e_h - e) / eps0
        if not np.isfinite(ep):
            ep = 0.0

        succ_i  = abs(e) <= tol
        fail_i  = (not np.isfinite(e)) or (not np.isfinite(ep))
        active_i = not (succ_i or fail_i)
        if active_i and abs(ep) < min_deriv:
            fail_i  = True
            active_i = False

        dx = 0.0
        if active_i:
            dx = -e / ep
            if max_step > 0.0 and np.isfinite(max_step):
                if dx > max_step:
                    dx = max_step
                elif dx < -max_step:
                    dx = -max_step

        e_prev = e  # initialize before loop so Numba can type it

        for _ in range(max_iter):
            if not active_i:
                break

            x_try   = x + dx
            eps_try = max(abs(x_try) * 1.0e-6, 1.0e-8)
            e_try, _ = _compute_tof_diff_scalar(
                x_try, dV_N_i, dV_T_i, dV_R_i,
                cos_theta_0m_i, sin_theta_0m_i, tan_theta_0m_i, tan_theta_0m_2_i,
                sin_theta_0f_i, cos_theta_0f_i,
                n1_base_i, n2_base_i, r0_i, rf_i, given_tof_i, mu,
            )
            e_try_h, _ = _compute_tof_diff_scalar(
                x_try + eps_try, dV_N_i, dV_T_i, dV_R_i,
                cos_theta_0m_i, sin_theta_0m_i, tan_theta_0m_i, tan_theta_0m_2_i,
                sin_theta_0f_i, cos_theta_0f_i,
                n1_base_i, n2_base_i, r0_i, rf_i, given_tof_i, mu,
            )
            ep_try = (e_try_h - e_try) / eps_try
            if not np.isfinite(ep_try):
                ep_try = 0.0

            valid_try = np.isfinite(e_try) and np.isfinite(ep_try)
            improve   = active_i and valid_try and (abs(e_try) < abs(e))

            if improve:
                e_prev = e
                x  = x_try
                e  = e_try
                ep = ep_try
            else:
                e_prev = e_try
                dx = -dx  # reject: reverse step

            converged = active_i and (abs(e) <= tol)
            if converged:
                succ_i  = True
                active_i = False
                break

            if (abs(ep) < min_deriv
                    or not np.isfinite(ep)
                    or not np.isfinite(e)):
                fail_i  = True
                active_i = False
                break

            if dx == 0.0 or not np.isfinite(dx):
                fail_i  = True
                active_i = False
                break

            # Quasi-second-order step
            epp  = 2.0 * (e_prev - e + ep * dx) / (dx * dx)
            disc = ep * ep - 2.0 * e * epp
            use_disc = np.isfinite(disc) and disc >= 0.0

            dx_new = 0.0
            if ep != 0.0 and np.isfinite(ep):            # fallback Newton
                dx_new = -e / ep
            if use_disc:                                  # quasi-second-order
                sqrt_disc = np.sqrt(disc)
                denom = ep + (sqrt_disc if ep >= 0.0 else -sqrt_disc)
                if denom != 0.0:
                    dx_new = -2.0 * e / denom
            elif np.isfinite(epp) and epp != 0.0:        # Halley-like fallback
                dx_new = -ep / epp

            if max_step > 0.0 and np.isfinite(max_step):
                if dx_new > max_step:
                    dx_new = max_step
                elif dx_new < -max_step:
                    dx_new = -max_step
            dx = dx_new

        V_T1_out[idx] = x
        succ_out[idx]  = succ_i

    return V_T1_out, succ_out


def solve_velocity(
    R0,
    Rf,
    t0,
    tf,
    mu,
    dV_norm_vec,
    theta_0m,
    theta_0f,
    dV_hat,
    V_T1_guess,
    debug=False,
):
    """
    Vectorized version of solve_velocity over a batch of DSM magnitudes.
    Match flight time and compute initial/final velocity given the maneuver placement

    Ref:
        Section III.D in Landau "Efficient Maneuver Placement for Automated Trajectory Design"
    """
    dV_norm_vec = np.asarray(dV_norm_vec, dtype=float).reshape(-1)
    n_case = int(dV_norm_vec.size)

    succ = np.zeros(n_case, dtype=bool)
    eta = np.zeros(n_case, dtype=float)
    V0 = np.zeros((n_case, 3), dtype=float)
    Vf = np.zeros((n_case, 3), dtype=float)
    T = np.zeros(n_case, dtype=float)
    if n_case == 0:
        return succ, eta, V0, Vf, T

    def _broadcast_vec3(name, value):
        arr = np.asarray(value, dtype=float)
        if arr.ndim == 1:
            if arr.size != 3:
                raise ValueError(f"{name} must be shape (3,) or (N,3).")
            return np.repeat(arr.reshape(1, 3), n_case, axis=0)
        if arr.ndim == 2 and arr.shape == (n_case, 3):
            return arr
        raise ValueError(f"{name} must be shape (3,) or (N,3).")

    def _broadcast_scalar(name, value):
        arr = np.asarray(value, dtype=float)
        if arr.ndim == 0:
            return np.full(n_case, float(arr), dtype=float)
        if arr.shape == (n_case,):
            return arr.astype(float, copy=False)
        raise ValueError(f"{name} must be scalar or shape (N,).")

    R0_case = _broadcast_vec3("R0", R0)
    Rf_case = _broadcast_vec3("Rf", Rf)
    r0 = np.linalg.norm(R0_case, axis=1)
    rf = np.linalg.norm(Rf_case, axis=1)
    if np.any(r0 <= 0.0) or np.any(rf <= 0.0):
        raise ValueError("R0 and Rf norms must be positive.")

    t0_case = _broadcast_scalar("t0", t0)
    tf_case = _broadcast_scalar("tf", tf)
    given_tof = tf_case - t0_case
    if np.any(given_tof <= 0.0):
        raise ValueError("tf must be greater than t0 for all rows.")

    theta_0m_arr = np.asarray(theta_0m, dtype=float)
    if theta_0m_arr.ndim == 0:
        theta_0m_case = np.full(n_case, float(theta_0m_arr), dtype=float)
    elif theta_0m_arr.shape == (n_case,):
        theta_0m_case = theta_0m_arr
    else:
        raise ValueError("theta_0m must be scalar or shape (N,).")

    theta_0f_arr = np.asarray(theta_0f, dtype=float)
    if theta_0f_arr.ndim == 0:
        theta_0f_case = np.full(n_case, float(theta_0f_arr), dtype=float)
    elif theta_0f_arr.shape == (n_case,):
        theta_0f_case = theta_0f_arr
    else:
        raise ValueError("theta_0f must be scalar or shape (N,).")

    sin_theta_0m = np.sin(theta_0m_case)
    cos_theta_0m = np.cos(theta_0m_case)
    tan_theta_0m = np.tan(theta_0m_case)
    tan_theta_0m_2 = np.tan(theta_0m_case / 2.0)
    sin_theta_0f = np.sin(theta_0f_case)
    cos_theta_0f = np.cos(theta_0f_case)
    n1_base = np.floor(theta_0m_case / (2.0 * np.pi))
    n2_base = np.floor((theta_0f_case - theta_0m_case) / (2.0 * np.pi))

    dV_hat_arr = np.asarray(dV_hat, dtype=float)
    if dV_hat_arr.ndim == 1:
        if dV_hat_arr.size != 3:
            raise ValueError("dV_hat must have 3 elements.")
        dV_hat_case = np.repeat(dV_hat_arr.reshape(1, 3), n_case, axis=0)
    elif dV_hat_arr.ndim == 2 and dV_hat_arr.shape[1] == 3:
        if dV_hat_arr.shape[0] != n_case:
            raise ValueError("dV_hat row count must match dV_norm_vec size.")
        dV_hat_case = dV_hat_arr
    else:
        raise ValueError("dV_hat must be shape (3,) or (N,3).")

    dV = dV_norm_vec.reshape(-1, 1) * dV_hat_case
    dV_R = dV[:, 0]
    dV_T = dV[:, 1]
    dV_N = dV[:, 2]

    tiny = 1e-14
    penalty = 1e6

    def compute_params_batch(V_T1_vec):
        V_T1 = np.asarray(V_T1_vec, dtype=float).reshape(-1)
        if V_T1.shape[0] != n_case:
            raise ValueError("V_T1_vec size must match dV_norm_vec size.")

        with np.errstate(invalid="ignore", divide="ignore", over="ignore"):
            x = np.sqrt((dV_N * cos_theta_0m) ** 2 + (dV_T + V_T1) ** 2)
            i0 = 2.0 * np.arctan2((x - dV_N * cos_theta_0m), (dV_T + V_T1))

            denom_tmp = x * sin_theta_0f
            tmp_arg = np.full_like(V_T1, np.nan, dtype=float)
            valid_tmp = np.abs(denom_tmp) > tiny
            tmp_arg[valid_tmp] = (dV_N[valid_tmp] * cos_theta_0f * sin_theta_0m) / denom_tmp[valid_tmp]
            tmp = np.arccos(np.clip(tmp_arg, -1.0, 1.0))
            i0 = np.where(np.cos(i0 + tmp) > np.cos(i0 - tmp), i0 + tmp, i0 - tmp)

            cos_theta_mf = cos_theta_0f * cos_theta_0m + sin_theta_0f * sin_theta_0m * np.cos(i0)
            cos_theta_mf = np.clip(cos_theta_mf, -1.0, 1.0)
            sin_theta_mf = np.sqrt(np.maximum(0.0, 1.0 - cos_theta_mf**2))

            denom_tmp2 = sin_theta_0f * cos_theta_0m * np.cos(i0) - cos_theta_0f * sin_theta_0m
            tmp2 = (V_T1 + dV_T) / denom_tmp2
            sin_theta_mf = np.where(tmp2 < 0.0, -sin_theta_mf, sin_theta_mf)
            V_T2 = tmp2 * sin_theta_mf

            theta_mf = np.arctan2(sin_theta_mf, cos_theta_mf)
            theta_mf = np.where(theta_mf < 0.0, theta_mf + 2.0 * np.pi, theta_mf)

            sin_theta_mf_safe = np.where(np.abs(sin_theta_mf) > tiny, sin_theta_mf, np.nan)
            a = -(V_T1 / (r0 * sin_theta_0m) + V_T2 / (rf * sin_theta_mf_safe))
            b = V_T1 / tan_theta_0m + V_T2 * (cos_theta_mf / sin_theta_mf_safe) - dV_R
            c = mu / V_T1 * tan_theta_0m_2 + mu / V_T2 * np.tan(theta_mf / 2.0)

            disc = b * b - 4.0 * a * c
            sqrt_disc = np.sqrt(disc)
            denom_root = 2.0 * a
            r1 = (-b + sqrt_disc) / denom_root
            r2 = (-b - sqrt_disc) / denom_root
            r_m = np.where(r1 > 0.0, r1, r2)

            h1 = r_m * V_T1
            V_R0 = -V_T1 / sin_theta_0m + h1 / (r0 * tan_theta_0m) + mu / h1 * tan_theta_0m_2
            V_R1 = h1 / (r0 * sin_theta_0m) - V_T1 / tan_theta_0m - mu / h1 * tan_theta_0m_2
            alpha_1 = 2.0 * mu / r0 - V_R0**2 - h1**2 / r0**2

            sqrt_alpha_1 = np.sqrt(alpha_1)
            chi1 = 2.0 / sqrt_alpha_1 * np.arctan(
                (sqrt_alpha_1 * r0 * tan_theta_0m_2) / (h1 - V_R0 * r0 * tan_theta_0m_2)
            )
            N1 = np.where(chi1 > 0.0, n1_base, n1_base + 1.0)
            t_0m = (mu * chi1 + V_R0 * r0 - V_R1 * r_m) / alpha_1 + 2.0 * np.pi * N1 * mu / np.sqrt(alpha_1**3)

            h2 = r_m * V_T2
            V_Rf = V_T2 / sin_theta_mf_safe - h2 / (rf * np.tan(theta_mf)) - mu / h2 * np.tan(theta_mf / 2.0)
            alpha_2 = 2.0 * mu / rf - V_Rf**2 - h2**2 / rf**2
            sqrt_alpha_2 = np.sqrt(alpha_2)
            chi2 = 2.0 / sqrt_alpha_2 * np.arctan(
                (sqrt_alpha_2 * rf * np.tan(theta_mf / 2.0)) / (h2 + V_Rf * rf * np.tan(theta_mf / 2.0))
            )
            N2 = np.where(chi2 > 0.0, n2_base, n2_base + 1.0)
            t_mf = (mu * chi2 + (V_R1 + dV_R) * r_m - V_Rf * rf) / alpha_2 + 2.0 * np.pi * N2 * mu / np.sqrt(alpha_2**3)

        valid = (
            np.isfinite(V_T1)
            & np.isfinite(i0)
            & np.isfinite(theta_mf)
            & np.isfinite(V_T2)
            & np.isfinite(disc)
            & (disc >= 0.0)
            & (np.abs(denom_root) > tiny)
            & (np.abs(sin_theta_mf) > tiny)
            & (np.abs(V_T1) > tiny)
            & (np.abs(V_T2) > tiny)
            & np.isfinite(r_m)
            & (r_m > 0.0)
            & np.isfinite(alpha_1)
            & (alpha_1 > 0.0)
            & np.isfinite(alpha_2)
            & (alpha_2 > 0.0)
            & np.isfinite(t_0m)
            & np.isfinite(t_mf)
            & (t_0m > 0.0)
            & (t_mf > 0.0)
        )

        return {
            "valid": valid,
            "t_0m": t_0m,
            "t_mf": t_mf,
            "V_R0": V_R0,
            "V_Rf": V_Rf,
            "i0": i0,
            "h1": h1,
            "h2": h2,
            "theta_mf": theta_mf,
        }

    def compute_tof_diff_batch(V_T1_vec):
        params = compute_params_batch(V_T1_vec)
        calc_tof = params["t_0m"] + params["t_mf"]
        diff = calc_tof - given_tof
        diff = np.where(params["valid"], diff, penalty)
        diff = np.where(np.isfinite(diff), diff, penalty)
        return diff

    def compute_tof_diff_with_derivatives_batch(V_T1_vec):
        V_T1 = np.asarray(V_T1_vec, dtype=float).reshape(-1)
        eps = np.maximum(np.abs(V_T1) * 1e-6, 1e-8)
        diff = compute_tof_diff_batch(V_T1)
        diff_eps = compute_tof_diff_batch(V_T1 + eps)
        diff_dv = (diff_eps - diff) / eps
        diff_dv = np.where(np.isfinite(diff_dv), diff_dv, 0.0)
        return diff, diff_dv

    V_T1_guess_arr = np.asarray(V_T1_guess, dtype=float)
    if V_T1_guess_arr.ndim == 0:
        V_T1_init = np.full(n_case, float(V_T1_guess_arr), dtype=float)
    elif V_T1_guess_arr.shape == (n_case,):
        V_T1_init = V_T1_guess_arr.copy()
    else:
        raise ValueError("V_T1_guess must be scalar or shape (N,).")

    max_step = max(float(np.max(np.abs(V_T1_init))) * 0.5, 1e-6)
    V_T1_opt, succ_root = _solve_velocity_jit(
        dV_N, dV_T, dV_R,
        cos_theta_0m, sin_theta_0m, tan_theta_0m, tan_theta_0m_2,
        sin_theta_0f, cos_theta_0f,
        n1_base, n2_base,
        r0, rf, given_tof,
        float(mu),
        V_T1_init,
        np.float64(max_step),
    )

    params_opt = compute_params_batch(V_T1_opt)
    succ = succ_root & params_opt["valid"]
    if not np.any(succ):
        return succ, eta, V0, Vf, T

    t_0m = params_opt["t_0m"]
    t_mf = params_opt["t_mf"]
    calc_tof = t_0m + t_mf
    eta[succ] = t_0m[succ] / calc_tof[succ]
    T[succ] = calc_tof[succ]

    i0 = params_opt["i0"]
    h1 = params_opt["h1"]
    h2 = params_opt["h2"]
    theta_mf = params_opt["theta_mf"]
    V_R0 = params_opt["V_R0"]
    V_Rf = params_opt["V_Rf"]

    V0[:, 0] = V_R0
    V0[:, 1] = np.cos(i0) * h1 / r0
    V0[:, 2] = np.sin(i0) * h1 / r0

    sin_theta_mf = np.sin(theta_mf)
    tmp = np.zeros(n_case, dtype=float)
    valid_tmp = succ & (np.abs(sin_theta_mf) > tiny)
    tmp[valid_tmp] = np.sin(i0[valid_tmp]) * np.sin(theta_0m_case[valid_tmp]) / sin_theta_mf[valid_tmp]
    tmp = np.clip(tmp, -1.0, 1.0)

    Vf[:, 0] = V_Rf
    Vf[:, 1] = np.sqrt(np.maximum(0.0, 1.0 - tmp * tmp)) * h2 / rf
    Vf[:, 2] = -tmp * h2 / rf

    finite_state = succ & np.isfinite(eta) & np.all(np.isfinite(V0), axis=1) & np.all(np.isfinite(Vf), axis=1)
    succ = finite_state
    eta[~succ] = 0.0
    T[~succ] = 0.0
    V0[~succ] = 0.0
    Vf[~succ] = 0.0

    if debug:
        print(f"solve_velocity: converged {int(np.count_nonzero(succ))}/{n_case}")

    return succ, eta, V0, Vf, T


def quasi_second_order_root_solve(
    err_and_deriv: Callable[[np.ndarray], Tuple[np.ndarray, np.ndarray]],
    x0: np.ndarray,
    tol: float = 1e-10,
    max_iter: int = 50,
    min_deriv: float = 1e-14,
    max_step: Optional[float] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Vectorized quasi-second-order root solver.

    Applies the same Appendix-A update logic elementwise across an input vector.
    """
    x = np.asarray(x0, dtype=float).reshape(-1).copy()
    n = x.size
    if n == 0:
        return x, np.zeros(0, dtype=bool)

    e, ep = err_and_deriv(x)
    e = np.asarray(e, dtype=float).reshape(-1)
    ep = np.asarray(ep, dtype=float).reshape(-1)
    if e.shape != x.shape or ep.shape != x.shape:
        raise ValueError("err_and_deriv must return vectors with same shape as x0.")

    def _clip_step(dx: np.ndarray) -> np.ndarray:
        if max_step is None:
            return dx
        step_cap = float(max_step)
        if not np.isfinite(step_cap) or step_cap <= 0.0:
            return dx
        return np.clip(dx, -step_cap, step_cap)

    succ = np.abs(e) <= tol
    fail = (~np.isfinite(e)) | (~np.isfinite(ep))
    active = ~(succ | fail)

    deriv_bad = active & (np.abs(ep) < min_deriv)
    fail |= deriv_bad
    active &= ~deriv_bad

    dx = np.zeros_like(x)
    dx[active] = -e[active] / ep[active]
    dx = _clip_step(dx)

    for _ in range(int(max_iter)):
        if not np.any(active):
            break

        x_try = x + dx
        e_try, ep_try = err_and_deriv(x_try)
        e_try = np.asarray(e_try, dtype=float).reshape(-1)
        ep_try = np.asarray(ep_try, dtype=float).reshape(-1)

        valid_try = np.isfinite(e_try) & np.isfinite(ep_try)
        improve = active & valid_try & (np.abs(e_try) < np.abs(e))
        reject = active & (~improve)

        e_prev = np.where(improve, e, e_try)
        x = np.where(improve, x_try, x)
        e = np.where(improve, e_try, e)
        ep = np.where(improve, ep_try, ep)
        dx = np.where(reject, -dx, dx)

        converged = active & (np.abs(e) <= tol)
        succ |= converged
        active &= ~converged

        deriv_bad = active & ((np.abs(ep) < min_deriv) | (~np.isfinite(ep)) | (~np.isfinite(e)))
        fail |= deriv_bad
        active &= ~deriv_bad

        dx_bad = active & ((dx == 0.0) | (~np.isfinite(dx)))
        fail |= dx_bad
        active &= ~dx_bad

        if not np.any(active):
            break

        epp = np.full_like(x, np.nan, dtype=float)
        epp[active] = 2.0 * (e_prev[active] - e[active] + ep[active] * dx[active]) / (dx[active] * dx[active])

        disc = ep * ep - 2.0 * e * epp
        use_disc = active & np.isfinite(disc) & (disc >= 0.0)
        sqrt_disc = np.sqrt(np.where(use_disc, disc, 0.0))
        denom = ep + np.copysign(sqrt_disc, ep)

        dx_new = np.zeros_like(dx)
        fallback_newton = active & (ep != 0.0) & np.isfinite(ep)
        dx_new[fallback_newton] = -e[fallback_newton] / ep[fallback_newton]

        use_main = use_disc & (denom != 0.0)
        dx_new[use_main] = -2.0 * e[use_main] / denom[use_main]

        use_alt = active & (~use_disc) & np.isfinite(epp) & (epp != 0.0)
        dx_new[use_alt] = -ep[use_alt] / epp[use_alt]

        dx_new = _clip_step(dx_new)
        dx = np.where(active, dx_new, dx)

    fail |= ~(succ | fail)
    return x, succ


def _solve_arc_prepare_objective_args(
    obj_type,
    additional_args,
    r_ref,
    mu_ref,
):
    if obj_type != 2:
        return {}

    if not isinstance(additional_args, dict):
        raise ValueError(
            "obj_type=2 requires additional_args dict with body mu/radius inputs."
        )

    required_keys = (
        "body_0_mu_km3_s2",
        "body_f_mu_km3_s2",
        "body_0_radius_km",
        "body_f_radius_km",
    )
    missing_keys = [key for key in required_keys if additional_args.get(key, None) is None]
    if missing_keys:
        missing_joined = ", ".join(missing_keys)
        raise ValueError(
            f"obj_type=2 requires additional_args keys: {missing_joined}."
        )

    args = dict(additional_args)
    rp_ratio = args.get("rp_ratio", args.get("safe_rp_ratio", 1.1))
    rp_ratio = float(rp_ratio)
    if not np.isfinite(rp_ratio) or rp_ratio <= 0.0:
        raise ValueError("obj_type=2 requires a positive finite rp_ratio.")

    mu_0 = float(args["body_0_mu_km3_s2"]) / float(mu_ref)
    mu_f = float(args["body_f_mu_km3_s2"]) / float(mu_ref)
    rmin0 = float(args["body_0_radius_km"]) * rp_ratio / float(r_ref)
    rminf = float(args["body_f_radius_km"]) * rp_ratio / float(r_ref)
    return {
        "rp_ratio": rp_ratio,
        "mu_0": mu_0,
        "mu_f": mu_f,
        "rmin0": rmin0,
        "rminf": rminf,
    }


def _solve_arc_normalize_lev_types(obj_type, lev_type):
    allowed_modes = ("+-", "-+", "++", "--")

    if lev_type is None:
        values = ["+-"]
    elif isinstance(lev_type, str):
        values = [lev_type]
    else:
        try:
            values = list(lev_type)
        except TypeError as exc:
            raise ValueError(
                "lev_type must be a string or a sequence of strings ('+-', '-+', '++', '--')."
            ) from exc

    normalized = []
    for mode in values:
        if not isinstance(mode, str):
            raise ValueError(
                "lev_type must be a string or a sequence of strings ('+-', '-+', '++', '--')."
            )
        mode = mode.strip()
        if mode not in allowed_modes:
            raise ValueError("Invalid lev_type. Choose from '+-', '-+', '++', '--' (or a list of these).")
        if mode not in normalized:
            normalized.append(mode)

    if not normalized:
        raise ValueError("lev_type must contain at least one entry.")

    if int(obj_type) != 1 and normalized != ["+-"]:
        raise ValueError("lev_type is only supported for obj_type=1. Use default '+-' for obj_type=2.")

    return normalized

def _to_cartesian_velocities(
    dV_norm_batch_km_s,
    V0_batch_nd,
    Vf_batch_nd,
    seed_idx_batch,
    v_ref,
    v0_seed_km_s,
    vf_seed_km_s,
    rtn_to_cart_0_seed,
    rtn_to_cart_f_seed,
):
    dV_norm = np.asarray(dV_norm_batch_km_s, dtype=float).reshape(-1)
    V0_nd = np.asarray(V0_batch_nd, dtype=float).reshape(-1, 3)
    Vf_nd = np.asarray(Vf_batch_nd, dtype=float).reshape(-1, 3)
    seed_idx = np.asarray(seed_idx_batch, dtype=int).reshape(-1)
    if V0_nd.shape[0] != dV_norm.shape[0] or Vf_nd.shape[0] != dV_norm.shape[0] or seed_idx.shape[0] != dV_norm.shape[0]:
        raise ValueError("Batch velocity conversion arrays must have matching row counts.")

    v0_seed = np.asarray(v0_seed_km_s, dtype=float).reshape(-1, 3)
    vf_seed = np.asarray(vf_seed_km_s, dtype=float).reshape(-1, 3)
    rtn0_seed = np.asarray(rtn_to_cart_0_seed, dtype=float).reshape(-1, 3, 3)
    rtnf_seed = np.asarray(rtn_to_cart_f_seed, dtype=float).reshape(-1, 3, 3)

    V0_cart = np.einsum("nij,nj->ni", rtn0_seed[seed_idx], V0_nd * float(v_ref))
    Vf_cart = np.einsum("nij,nj->ni", rtnf_seed[seed_idx], Vf_nd * float(v_ref))
    zero_mask = np.abs(dV_norm) <= 1e-12
    if np.any(zero_mask):
        V0_cart[zero_mask] = v0_seed[seed_idx[zero_mask]]
        Vf_cart[zero_mask] = vf_seed[seed_idx[zero_mask]]
    return V0_cart, Vf_cart


def _solve_arc_objective_gradients(
    obj_type,
    V0_seed_nd,
    Vf_seed_nd,
    V_P0_nd,
    V_Pf_nd,
    V_P0_hat,
    V_Pf_hat,
    vp0,
    vpf,
    mu,
    objective_additional_args,
    rtn_to_cart_0_batch,
    rtn_to_cart_f_batch,
):
    V0_arr = np.asarray(V0_seed_nd, dtype=float).reshape(-1, 3)
    Vf_arr = np.asarray(Vf_seed_nd, dtype=float).reshape(-1, 3)
    rtn0 = np.asarray(rtn_to_cart_0_batch, dtype=float).reshape(-1, 3, 3)
    rtnf = np.asarray(rtn_to_cart_f_batch, dtype=float).reshape(-1, 3, 3)
    if V0_arr.shape[0] != Vf_arr.shape[0] or V0_arr.shape[0] != rtn0.shape[0] or V0_arr.shape[0] != rtnf.shape[0]:
        raise ValueError("Objective gradient batch arrays must have matching row counts.")

    S = int(V0_arr.shape[0])
    if S == 0:
        return np.empty((0, 3), dtype=float), np.empty((0, 3), dtype=float)

    def _broadcast_vec3(name, value):
        arr = np.asarray(value, dtype=float)
        if arr.ndim == 1:
            if arr.size != 3:
                raise ValueError(f"{name} must be shape (3,) or (S,3).")
            return np.repeat(arr.reshape(1, 3), S, axis=0)
        if arr.ndim == 2 and arr.shape == (S, 3):
            return arr
        raise ValueError(f"{name} must be shape (3,) or (S,3).")

    def _broadcast_scalar(name, value):
        arr = np.asarray(value, dtype=float)
        if arr.ndim == 0:
            return np.full(S, float(arr), dtype=float)
        if arr.shape == (S,):
            return arr.astype(float, copy=False)
        raise ValueError(f"{name} must be scalar or shape (S,).")

    V_P0_nd_arr = _broadcast_vec3("V_P0_nd", V_P0_nd)
    V_Pf_nd_arr = _broadcast_vec3("V_Pf_nd", V_Pf_nd)
    V_P0_hat_arr = _broadcast_vec3("V_P0_hat", V_P0_hat)
    V_Pf_hat_arr = _broadcast_vec3("V_Pf_hat", V_Pf_hat)
    vp0_arr = _broadcast_scalar("vp0", vp0)
    vpf_arr = _broadcast_scalar("vpf", vpf)
    objective_additional_args = dict(objective_additional_args or {})

    if int(obj_type) == 1:
        lev_type = str(objective_additional_args.get("lev_type", "+-")).strip()
        return vinf_objective_gradients_rtn(
            "lev",
            V0_arr,
            Vf_arr,
            V_P0_nd_arr,
            V_Pf_nd_arr,
            rtn0,
            rtnf,
            lev_type=lev_type,
        )

    if int(obj_type) == 2:
        mu_0 = float(objective_additional_args.get("mu_0", np.nan))
        mu_f = float(objective_additional_args.get("mu_f", np.nan))
        rmin0 = float(objective_additional_args.get("rmin0", np.nan))
        rminf = float(objective_additional_args.get("rminf", np.nan))
        if not np.isfinite(mu_0) or not np.isfinite(mu_f) or not np.isfinite(rmin0) or not np.isfinite(rminf):
            raise ValueError("obj_type=2 requires finite mu_0/mu_f/rmin0/rminf objective parameters.")
        if mu_0 <= 0.0 or mu_f <= 0.0 or rmin0 <= 0.0 or rminf <= 0.0:
            raise ValueError("obj_type=2 requires positive mu_0/mu_f/rmin0/rminf objective parameters.")

        def _P0_minus(V0_batch):
            V0_batch = np.asarray(V0_batch, dtype=float).reshape(-1, 3)
            nrow = int(V0_batch.shape[0])
            if nrow == S:
                V_P0_loc = V_P0_nd_arr
                V_P0_hat_loc = V_P0_hat_arr
                vp0_loc = vp0_arr
            elif nrow % S == 0:
                rep = nrow // S
                V_P0_loc = np.repeat(V_P0_nd_arr, rep, axis=0)
                V_P0_hat_loc = np.repeat(V_P0_hat_arr, rep, axis=0)
                vp0_loc = np.repeat(vp0_arr, rep)
            else:
                raise ValueError("V0_batch row count must be a multiple of seed count.")

            Vinf_0 = V0_batch - V_P0_loc
            vinf0 = np.maximum(np.linalg.norm(Vinf_0, axis=1), 1e-12)
            Vinf_0_hat = Vinf_0 / vinf0.reshape(-1, 1)
            alpha0_plus = np.arccos(np.clip(np.einsum("ij,ij->i", Vinf_0_hat, V_P0_hat_loc), -1.0, 1.0))
            delta_0 = 2.0 * np.arcsin(np.clip(1.0 / (1.0 + rmin0 * vinf0**2 / mu_0), -1.0, 1.0))
            alpha0_minus = alpha0_plus - delta_0
            V0_2 = vp0_loc**2 + 2.0 * vp0_loc * vinf0 * np.cos(alpha0_minus) + vinf0**2
            return np.pi * float(mu) / np.sqrt(2.0) * np.power(float(mu) / rmin0 - V0_2 / 2.0, -1.5)

        def _Pf_minus(Vf_batch):
            Vf_batch = np.asarray(Vf_batch, dtype=float).reshape(-1, 3)
            nrow = int(Vf_batch.shape[0])
            if nrow == S:
                V_Pf_loc = V_Pf_nd_arr
                V_Pf_hat_loc = V_Pf_hat_arr
                vpf_loc = vpf_arr
            elif nrow % S == 0:
                rep = nrow // S
                V_Pf_loc = np.repeat(V_Pf_nd_arr, rep, axis=0)
                V_Pf_hat_loc = np.repeat(V_Pf_hat_arr, rep, axis=0)
                vpf_loc = np.repeat(vpf_arr, rep)
            else:
                raise ValueError("Vf_batch row count must be a multiple of seed count.")

            Vinf_f = Vf_batch - V_Pf_loc
            vinff = np.maximum(np.linalg.norm(Vinf_f, axis=1), 1e-12)
            Vinf_f_hat = Vinf_f / vinff.reshape(-1, 1)
            alphaf_minus = np.arccos(np.clip(np.einsum("ij,ij->i", Vinf_f_hat, V_Pf_hat_loc), -1.0, 1.0))
            delta_f = 2.0 * np.arcsin(np.clip(1.0 / (1.0 + rminf * vinff**2 / mu_f), -1.0, 1.0))
            alphaf_plus = alphaf_minus + delta_f
            Vf_2 = vpf_loc**2 + 2.0 * vpf_loc * vinff * np.cos(alphaf_plus) + vinff**2
            return np.pi * float(mu) / np.sqrt(2.0) * np.power(float(mu) / rminf - Vf_2 / 2.0, -1.5)

        eps_fd = 1e-6
        eye3 = np.eye(3, dtype=float)
        P0_base = _P0_minus(V0_arr)
        Pf_base = _Pf_minus(Vf_arr)

        V0_pert = V0_arr.reshape(S, 1, 3) + eps_fd * eye3.reshape(1, 3, 3)
        Vf_pert = Vf_arr.reshape(S, 1, 3) + eps_fd * eye3.reshape(1, 3, 3)
        P0_pert = _P0_minus(V0_pert.reshape(-1, 3)).reshape(S, 3)
        Pf_pert = _Pf_minus(Vf_pert.reshape(-1, 3)).reshape(S, 3)

        dJ0_cart = -(P0_pert - P0_base.reshape(-1, 1)) / eps_fd
        dJf_cart = (Pf_pert - Pf_base.reshape(-1, 1)) / eps_fd
        dJ0_rtn = np.einsum("nji,nj->ni", rtn0, dJ0_cart)
        dJf_rtn = np.einsum("nji,nj->ni", rtnf, dJf_cart)
        return dJ0_rtn, dJf_rtn

    raise ValueError("Invalid objective type. Choose 1 (minimum Vinf) or 2 (minimum TOF).")


def solve_arc(
    R0_km,
    Rf_km,
    V_P0_km_s,
    V_Pf_km_s,
    V0_km_s,
    Vf_km_s,
    t0_et_s,
    tf_et_s,
    nrev_signed,
    dV_norm_list_km_s,
    mu_km3_s2,
    obj_type=1,
    max_vinf0_km_s=100.0,
    max_vinff_km_s=100.0,
    additional_args=None,
    debug=False,
    lev_type="+-",
    case_chunk_size=None,
    return_endpoint_velocities=True,
):
    """
    Generate trajectory arc data by varying DSM magnitude from endpoint state(s).

    Inputs can be one window (`(3,)`, scalar times) or multiple windows (`(W,3)`, `(W,)`).
    Lambert seeds are always provided in physical units via `V0_km_s`, `Vf_km_s`,
    and `nrev_signed`.
    For multiple windows, endpoint and time arrays must align one-to-one with
    Lambert seed rows (same row count).
    
    Returns:
    If `return_endpoint_velocities` is True:
    `(eta_store, V0_store, Vf_store, DV_store, Vinf_0_store, Vinf_f_store,
      lev_store, arc_store, seed_store)`

    If `return_endpoint_velocities` is False:
    `(eta_store, DV_store, Vinf_0_store, Vinf_f_store, lev_store, arc_store, seed_store)`
    """

    def _as_window_vec3(name, value):
        arr = np.asarray(value, dtype=float)
        if arr.ndim == 1:
            if arr.size != 3:
                raise ValueError(f"{name} must be shape (3,) or (W,3).")
            return arr.reshape(1, 3)
        if arr.ndim == 2 and arr.shape[1] == 3:
            return arr
        raise ValueError(f"{name} must be shape (3,) or (W,3).")

    R0_km_arr = _as_window_vec3("R0_km", R0_km)
    Rf_km_arr = _as_window_vec3("Rf_km", Rf_km)
    V_P0_km_s_arr = _as_window_vec3("V_P0_km_s", V_P0_km_s)
    V_Pf_km_s_arr = _as_window_vec3("V_Pf_km_s", V_Pf_km_s)

    W = int(R0_km_arr.shape[0])
    if Rf_km_arr.shape[0] != W or V_P0_km_s_arr.shape[0] != W or V_Pf_km_s_arr.shape[0] != W:
        raise ValueError("Windowed endpoint arrays must have the same row count.")

    def _as_window_time(name, value):
        arr = np.asarray(value, dtype=float)
        if arr.ndim == 0:
            return np.full(W, float(arr), dtype=float)
        if arr.shape == (W,):
            return arr.astype(float, copy=False)
        raise ValueError(f"{name} must be scalar or shape (W,).")

    t0_et_s_arr = _as_window_time("t0_et_s", t0_et_s)
    tf_et_s_arr = _as_window_time("tf_et_s", tf_et_s)
    if np.any(tf_et_s_arr <= t0_et_s_arr):
        raise ValueError("tf_et_s must be greater than t0_et_s for all windows.")

    dV_norm_list = np.asarray(dV_norm_list_km_s, dtype=float).reshape(-1)
    dV_nonzero = dV_norm_list[np.abs(dV_norm_list) > 1e-12]

    if dV_norm_list.size == 0:
        raise ValueError("dV_norm_list_km_s must be non-empty.")

    mu_km3_s2 = float(mu_km3_s2)
    if mu_km3_s2 <= 0.0:
        raise ValueError("mu_km3_s2 must be positive.")

    obj_type = int(obj_type)
    if obj_type not in (1, 2):
        raise ValueError("Invalid objective type. Choose 1 (minimum Vinf) or 2 (minimum TOF).")
    lev_type_list = _solve_arc_normalize_lev_types(obj_type, lev_type)

    if additional_args is None:
        additional_args = {}
    elif not isinstance(additional_args, dict):
        raise ValueError("additional_args must be a dict when provided.")
    else:
        additional_args = dict(additional_args)

    max_vinf0_km_s = float(max_vinf0_km_s)
    max_vinff_km_s = float(max_vinff_km_s)
    if max_vinf0_km_s <= 0.0 or max_vinff_km_s <= 0.0:
        raise ValueError("max_vinf0_km_s and max_vinff_km_s must be positive.")

    # Canonical scaling: r_ref=AU, t_ref=sqrt(r_ref^3/mu), so mu_nd = 1.
    r_ref = AU_KM
    t_ref = np.sqrt(r_ref**3 / mu_km3_s2)
    v_ref = np.sqrt(mu_km3_s2 / r_ref)
    mu_ref = mu_km3_s2
    mu = 1.0
    obj_args = _solve_arc_prepare_objective_args(
        obj_type,
        additional_args,
        r_ref,
        mu_ref,
    )

    tof_arr = (tf_et_s_arr - t0_et_s_arr) / t_ref
    t0_scaled_arr = np.zeros(W, dtype=float)
    tf_scaled_arr = tof_arr.copy()

    # Endpoint states in nondimensional units.
    r_P0_arr = R0_km_arr / r_ref
    r_Pf_arr = Rf_km_arr / r_ref
    V_P0_nd_arr = V_P0_km_s_arr / v_ref
    V_Pf_nd_arr = V_Pf_km_s_arr / v_ref

    # magnitude of the planets velocities
    vp0_arr = np.linalg.norm(V_P0_nd_arr, axis=1)
    vpf_arr = np.linalg.norm(V_Pf_nd_arr, axis=1)
    if np.any(vp0_arr <= 0.0) or np.any(vpf_arr <= 0.0):
        raise ValueError("Body velocity norm must be positive at both endpoints.")

    # unit vectors of the planets velocities
    V_P0_hat_arr = V_P0_nd_arr / vp0_arr.reshape(-1, 1)
    V_Pf_hat_arr = V_Pf_nd_arr / vpf_arr.reshape(-1, 1)

    # 1) Lambert seed set in physical units with per-seed window mapping.
    v0_seed_km_s = np.asarray(V0_km_s, dtype=float).reshape(-1, 3)
    vf_seed_km_s = np.asarray(Vf_km_s, dtype=float).reshape(-1, 3)
    nrev_i = np.asarray(nrev_signed, dtype=int).reshape(-1)
    if v0_seed_km_s.shape[0] != vf_seed_km_s.shape[0] or v0_seed_km_s.shape[0] != nrev_i.shape[0]:
        raise ValueError("V0_km_s, Vf_km_s, and nrev_signed must have matching row counts.")

    if v0_seed_km_s.shape[0] == W:
        seed_window = np.arange(W, dtype=int)
    elif W == 1:
        seed_window = np.zeros(v0_seed_km_s.shape[0], dtype=int)
    else:
        raise ValueError(
            "For multiple windows, endpoint/time arrays must match Lambert seed row count."
        )

    if v0_seed_km_s.shape[0] > 0:
        V_P0_seed_km_s = V_P0_km_s_arr[seed_window]
        V_Pf_seed_km_s = V_Pf_km_s_arr[seed_window]
        vinf0_i = np.linalg.norm(v0_seed_km_s - V_P0_seed_km_s, axis=1)
        vinff_i = np.linalg.norm(vf_seed_km_s - V_Pf_seed_km_s, axis=1)
        keep = (vinf0_i <= max_vinf0_km_s) & (vinff_i <= max_vinff_km_s)
        v0_seed_km_s = v0_seed_km_s[keep]
        vf_seed_km_s = vf_seed_km_s[keep]
        nrev_i = nrev_i[keep]
        seed_window = seed_window[keep]

    n_seed = int(v0_seed_km_s.shape[0])
    if debug:
        seed_counts = np.bincount(seed_window, minlength=W) if n_seed > 0 else np.zeros(W, dtype=int)
        print(
            "Lambert seeds ready: "
            f"n_total={n_seed}, per_window={seed_counts}, nrev={nrev_i}"
        )
    if n_seed == 0:
        raise RuntimeError("No Lambert seed remained after v-infinity filtering.")

    # Lambert seeds in nondimensional units for downstream RTN/objective calculations.
    v0_i = v0_seed_km_s / v_ref
    vf_i = vf_seed_km_s / v_ref

    r_P0_seed = r_P0_arr[seed_window]
    r_Pf_seed = r_Pf_arr[seed_window]
    V_P0_seed_nd = V_P0_nd_arr[seed_window]
    V_Pf_seed_nd = V_Pf_nd_arr[seed_window]
    V_P0_seed_hat = V_P0_hat_arr[seed_window]
    V_Pf_seed_hat = V_Pf_hat_arr[seed_window]
    vp0_seed = vp0_arr[seed_window]
    vpf_seed = vpf_arr[seed_window]
    t0_seed = t0_scaled_arr[seed_window]
    tf_seed = tf_scaled_arr[seed_window]

    # 2) Primer-based maneuver placement for all Lambert seeds in one batch.
    rtn_to_cart_0_seed = rtn_to_catesian(r_P0_seed, v0_i)
    rtn_to_cart_f_seed = rtn_to_catesian(r_Pf_seed, vf_i)

    obj_args_base = dict(obj_args)
    if obj_type == 1:
        obj_args_base["lev_type"] = "+-"
    dJ_dV0_seed, dJ_dVf_seed = _solve_arc_objective_gradients(
        obj_type,
        v0_i,
        vf_i,
        V_P0_seed_nd,
        V_Pf_seed_nd,
        V_P0_seed_hat,
        V_Pf_seed_hat,
        vp0_seed,
        vpf_seed,
        mu,
        obj_args_base,
        rtn_to_cart_0_seed,
        rtn_to_cart_f_seed,
    )

    # Maneuver-placement equations below assume elliptic-like arcs and become
    # numerically invalid near/parabolic-hyperbolic regimes (e -> 1 or e > 1).
    # Keep all Lambert seeds for baseline rows, but only leverage eligible seeds.
    ecc_eps = 1.0e-2
    r0_seed_norm = np.linalg.norm(r_P0_seed, axis=1)
    h_seed = np.cross(r_P0_seed, v0_i)
    evec_seed = np.cross(v0_i, h_seed) / float(mu) - r_P0_seed / r0_seed_norm.reshape(-1, 1)
    e_seed = np.linalg.norm(evec_seed, axis=1)
    mp_seed_mask = np.isfinite(e_seed) & (e_seed <= (1.0 - ecc_eps))
    mp_seed_idx = np.flatnonzero(mp_seed_mask)
    n_seed_mp = int(mp_seed_idx.size)
    if debug and n_seed_mp < n_seed:
        print(
            "Maneuver-placement seed filter: "
            f"kept={n_seed_mp}/{n_seed} (e <= {1.0 - ecc_eps:.3f})"
        )

    if n_seed_mp > 0:
        theta_0m_seed, theta_0f_seed, dV_hat_seed, V_T1_guess_seed = maneuver_placement(
            r_P0_seed[mp_seed_idx],
            r_Pf_seed[mp_seed_idx],
            v0_i[mp_seed_idx],
            vf_i[mp_seed_idx],
            mu,
            dJ_dV0_seed[mp_seed_idx],
            dJ_dVf_seed[mp_seed_idx],
            np.abs(nrev_i[mp_seed_idx]).astype(int, copy=False),
            debug=debug,
        )
    else:
        theta_0m_seed = np.empty(0, dtype=float)
        theta_0f_seed = np.empty(0, dtype=float)
        dV_hat_seed = np.empty((0, 3), dtype=float)
        V_T1_guess_seed = np.empty(0, dtype=float)

    # 3) Baseline Lambert rows (dV=0), one per seed.
    eta_base = np.full(n_seed, 0.5, dtype=float)
    DV_base = np.zeros(n_seed, dtype=float)
    seed_base = np.arange(n_seed, dtype=int)
    rank_base = np.full(n_seed, -1, dtype=int)
    arc_base = seed_window.copy()
    lev_base = np.full(n_seed, lev_type_list[0], dtype=object)
    Vinf_0_base = v0_seed_km_s - V_P0_km_s_arr[arc_base]
    Vinf_f_base = vf_seed_km_s - V_Pf_km_s_arr[arc_base]
    if return_endpoint_velocities:
        V0_base = v0_seed_km_s.copy()
        Vf_base = vf_seed_km_s.copy()

    # 4) Non-zero dV cases: flatten (seed, mode, dV) and solve in one batch.
    if dV_nonzero.size > 0 and n_seed_mp > 0:
        if obj_type == 1:
            lev_modes = np.asarray(list(lev_type_list), dtype=object)
            # "+-"/"-+" share one dV_hat (already computed); "++"/"--" need a second gradient.
            _needs_pp = any(m in lev_type_list for m in ("++", "--"))
            if _needs_pp:
                _obj_args_pp = dict(obj_args)
                _obj_args_pp["lev_type"] = "++"
                _dJ0_pp, _dJf_pp = _solve_arc_objective_gradients(
                    obj_type, v0_i, vf_i, V_P0_seed_nd, V_Pf_seed_nd,
                    V_P0_seed_hat, V_Pf_seed_hat, vp0_seed, vpf_seed, mu,
                    _obj_args_pp, rtn_to_cart_0_seed, rtn_to_cart_f_seed,
                )
                if n_seed_mp > 0:
                    _, _, dV_hat_seed_pp, _ = maneuver_placement(
                        r_P0_seed[mp_seed_idx], r_Pf_seed[mp_seed_idx],
                        v0_i[mp_seed_idx], vf_i[mp_seed_idx], mu,
                        _dJ0_pp[mp_seed_idx], _dJf_pp[mp_seed_idx],
                        np.abs(nrev_i[mp_seed_idx]).astype(int, copy=False),
                        debug=debug,
                    )
                else:
                    dV_hat_seed_pp = np.empty((0, 3), dtype=float)
            else:
                dV_hat_seed_pp = None
            _pm_hat = dV_hat_seed
            _mode_hat_map = {
                # The primer direction is an ascent direction for the signed leverage
                # objective. Flip the "+-"/"-+" pair so labels describe endpoint vinf changes.
                "+-": -_pm_hat, "-+": _pm_hat,
                "++": dV_hat_seed_pp, "--": (-dV_hat_seed_pp if dV_hat_seed_pp is not None else None),
            }
            # Stack per-mode dV_hat arrays: shape (n_mode, n_seed_mp, 3)
            dV_hat_modes = np.stack([_mode_hat_map[m] for m in lev_modes], axis=0)
        else:
            lev_modes = np.asarray([lev_type_list[0]], dtype=object)
            dV_hat_modes = dV_hat_seed[np.newaxis]  # (1, n_seed_mp, 3)

        n_mode = int(lev_modes.size)
        n_dv = int(dV_nonzero.size)
        total_cases = int(n_seed_mp * n_mode * n_dv)
        if case_chunk_size is None:
            chunk_cases = total_cases
        else:
            chunk_cases = max(1, int(case_chunk_size))

        eta_nonzero_parts = []
        Vinf0_nonzero_parts = []
        Vinff_nonzero_parts = []
        if return_endpoint_velocities:
            V0_nonzero_parts = []
            Vf_nonzero_parts = []
        DV_nonzero_parts = []
        seed_nonzero_parts = []
        rank_nonzero_parts = []
        arc_nonzero_parts = []
        lev_nonzero_parts = []

        mode_stride = int(n_mode * n_dv)
        for case_start in range(0, total_cases, chunk_cases):
            case_end = min(case_start + chunk_cases, total_cases)
            flat_idx = np.arange(case_start, case_end, dtype=int)
            seed_case_local = flat_idx // mode_stride
            rem = flat_idx % mode_stride
            mode_case = rem // n_dv
            dv_case = rem % n_dv
            seed_case = mp_seed_idx[seed_case_local]
            arc_case = seed_window[seed_case]

            dV_norm_case_km_s = dV_nonzero[dv_case]
            dV_norm_case_nd = dV_norm_case_km_s / v_ref
            dV_hat_case = dV_hat_modes[mode_case, seed_case_local]

            succ_vec, eta_vec, V0_vec, Vf_vec, _ = solve_velocity(
                r_P0_seed[seed_case],
                r_Pf_seed[seed_case],
                t0_seed[seed_case],
                tf_seed[seed_case],
                mu,
                dV_norm_case_nd,
                theta_0m_seed[seed_case_local],
                theta_0f_seed[seed_case_local],
                dV_hat_case,
                V_T1_guess_seed[seed_case_local],
                debug=debug,
            )

            succ_idx = np.flatnonzero(succ_vec)
            if succ_idx.size == 0:
                continue

            V0_nonzero_chunk, Vf_nonzero_chunk = _to_cartesian_velocities(
                dV_norm_case_km_s[succ_idx],
                V0_vec[succ_idx],
                Vf_vec[succ_idx],
                seed_case[succ_idx],
                v_ref,
                v0_seed_km_s,
                vf_seed_km_s,
                rtn_to_cart_0_seed,
                rtn_to_cart_f_seed,
            )
            arc_chunk = arc_case[succ_idx]
            eta_nonzero_parts.append(eta_vec[succ_idx])
            Vinf0_nonzero_parts.append(V0_nonzero_chunk - V_P0_km_s_arr[arc_chunk])
            Vinff_nonzero_parts.append(Vf_nonzero_chunk - V_Pf_km_s_arr[arc_chunk])
            if return_endpoint_velocities:
                V0_nonzero_parts.append(V0_nonzero_chunk)
                Vf_nonzero_parts.append(Vf_nonzero_chunk)
            DV_nonzero_parts.append(dV_norm_case_km_s[succ_idx])
            seed_nonzero_parts.append(seed_case[succ_idx])
            rank_nonzero_parts.append(mode_case[succ_idx] * n_dv + dv_case[succ_idx])
            arc_nonzero_parts.append(arc_chunk)
            lev_nonzero_parts.append(lev_modes[mode_case[succ_idx]])

        if eta_nonzero_parts:
            eta_nonzero = np.concatenate(eta_nonzero_parts, axis=0)
            Vinf_0_nonzero = np.concatenate(Vinf0_nonzero_parts, axis=0)
            Vinf_f_nonzero = np.concatenate(Vinff_nonzero_parts, axis=0)
            if return_endpoint_velocities:
                V0_nonzero = np.concatenate(V0_nonzero_parts, axis=0)
                Vf_nonzero = np.concatenate(Vf_nonzero_parts, axis=0)
            DV_nonzero_sol = np.concatenate(DV_nonzero_parts, axis=0)
            seed_nonzero = np.concatenate(seed_nonzero_parts, axis=0)
            rank_nonzero = np.concatenate(rank_nonzero_parts, axis=0)
            arc_nonzero = np.concatenate(arc_nonzero_parts, axis=0)
            lev_nonzero = np.concatenate(lev_nonzero_parts, axis=0)
        else:
            eta_nonzero = np.empty(0, dtype=float)
            Vinf_0_nonzero = np.empty((0, 3), dtype=float)
            Vinf_f_nonzero = np.empty((0, 3), dtype=float)
            if return_endpoint_velocities:
                V0_nonzero = np.empty((0, 3), dtype=float)
                Vf_nonzero = np.empty((0, 3), dtype=float)
            DV_nonzero_sol = np.empty(0, dtype=float)
            seed_nonzero = np.empty(0, dtype=int)
            rank_nonzero = np.empty(0, dtype=int)
            arc_nonzero = np.empty(0, dtype=int)
            lev_nonzero = np.empty(0, dtype=object)
    else:
        eta_nonzero = np.empty(0, dtype=float)
        Vinf_0_nonzero = np.empty((0, 3), dtype=float)
        Vinf_f_nonzero = np.empty((0, 3), dtype=float)
        if return_endpoint_velocities:
            V0_nonzero = np.empty((0, 3), dtype=float)
            Vf_nonzero = np.empty((0, 3), dtype=float)
        DV_nonzero_sol = np.empty(0, dtype=float)
        seed_nonzero = np.empty(0, dtype=int)
        rank_nonzero = np.empty(0, dtype=int)
        arc_nonzero = np.empty(0, dtype=int)
        lev_nonzero = np.empty(0, dtype=object)

    # 5) Merge baseline + non-zero solutions and order by (seed, local case rank).
    eta_all = np.concatenate((eta_base, eta_nonzero))
    DV_all = np.concatenate((DV_base, DV_nonzero_sol))
    Vinf_0_all = np.concatenate((Vinf_0_base, Vinf_0_nonzero), axis=0)
    Vinf_f_all = np.concatenate((Vinf_f_base, Vinf_f_nonzero), axis=0)
    seed_all = np.concatenate((seed_base, seed_nonzero))
    rank_all = np.concatenate((rank_base, rank_nonzero))
    arc_all = np.concatenate((arc_base, arc_nonzero))

    order = np.lexsort((rank_all, seed_all, arc_all))
    eta_store = eta_all[order]
    DV_store = DV_all[order]
    Vinf_0_store = Vinf_0_all[order]
    Vinf_f_store = Vinf_f_all[order]
    arc_store = arc_all[order].astype(int, copy=False)
    seed_store = seed_all[order].astype(int, copy=False)
    lev_all = np.concatenate((lev_base, lev_nonzero))
    lev_store = lev_all[order]
    if return_endpoint_velocities:
        V0_all = np.concatenate((V0_base, V0_nonzero), axis=0)
        Vf_all = np.concatenate((Vf_base, Vf_nonzero), axis=0)
        V0_store = V0_all[order]
        Vf_store = Vf_all[order]
        return (
            eta_store,
            V0_store,
            Vf_store,
            DV_store,
            Vinf_0_store,
            Vinf_f_store,
            lev_store,
            arc_store,
            seed_store,
        )
    return (
        eta_store,
        DV_store,
        Vinf_0_store,
        Vinf_f_store,
        lev_store,
        arc_store,
        seed_store,
    )


### PLOTTING HELPERS #### 

def _sample_leg_positions(r0_km, v0_km_s, dt_s, mu_km3_s2, num_samples=200):
    n_samples = max(2, int(num_samples))
    dt_total_s = float(dt_s)
    if dt_total_s <= 0.0:
        raise ValueError("dt_s must be positive for trajectory sampling.")

    positions_km = np.empty((n_samples, 3), dtype=float)
    times_s = np.linspace(0.0, dt_total_s, n_samples)

    r_cur = np.asarray(r0_km, dtype=float).reshape(3)
    v_cur = np.asarray(v0_km_s, dtype=float).reshape(3)
    positions_km[0] = r_cur

    prev_t_s = 0.0
    for idx in range(1, n_samples):
        dt_step_s = float(times_s[idx] - prev_t_s)
        r_cur, v_cur = kepler_propagate_universal(r_cur, v_cur, dt_step_s, float(mu_km3_s2))
        positions_km[idx] = r_cur
        prev_t_s = float(times_s[idx])
    return positions_km


def plot_solve_arc_overlay(
    t0_et_s,
    tf_et_s,
    R0_km,
    Rf_km,
    eta_store,
    V0_store,
    Vf_store,
    DV_store,
    mu_sun_km3_s2,
    dep_body_id="399",
    arr_body_id="399",
    frame=DEFAULT_FRAME,
    observer=DEFAULT_OBSERVER,
    num_samples_leg=250,
    num_samples_orbit=500,
):
    DV_store = np.asarray(DV_store, dtype=float).reshape(-1)
    eta_store = np.asarray(eta_store, dtype=float).reshape(-1)
    V0_store = np.asarray(V0_store, dtype=float).reshape(-1, 3)
    Vf_store = np.asarray(Vf_store, dtype=float).reshape(-1, 3)

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.scatter([0.0], [0.0], color="gold", s=100, label="Sun", zorder=5)

    t0 = float(t0_et_s)
    tf = float(tf_et_s)
    tof_s = tf - t0
    unique_body_ids = list(dict.fromkeys([int(dep_body_id), int(arr_body_id)]))
    for body_id in unique_body_ids:
        state6, _ = spice.spkezr(str(body_id), float(t0), str(frame), "NONE", str(observer))
        state6_array = np.asarray(state6, dtype=float)
        r_ref = state6_array[:3]
        v_ref = state6_array[3:]
        orbit_times_s = np.linspace(t0, tf, int(num_samples_orbit))

        orbit_pos = np.empty((orbit_times_s.size, 3), dtype=float)
        for idx, et_s in enumerate(orbit_times_s):
            state6, _ = spice.spkezr(str(body_id), float(et_s), str(frame), "NONE", str(observer))
            orbit_pos[idx] = np.asarray(state6, dtype=float)[:3]
        ax.plot(
            orbit_pos[:, 0],
            orbit_pos[:, 1],
            linestyle=":",
            linewidth=1.5,
            alpha=0.9,
            label=f"Body {body_id} orbit",
        )

    cmap = plt.cm.viridis
    n_sol = int(DV_store.size)
    for i in range(n_sol):
        dv_i = float(DV_store[i])
        eta = float(eta_store[i])
        dt_0m_s = eta * tof_s
        dt_mf_s = (1.0 - eta) * tof_s
        if dt_0m_s <= 0.0 or dt_mf_s <= 0.0:
            continue

        if abs(dv_i) <= 1e-12:
            # Plot the pure Lambert reference arc over full TOF.
            path = _sample_leg_positions(
                R0_km,
                V0_store[i],
                tof_s,
                mu_sun_km3_s2,
                num_samples=max(80, int(2 * num_samples_leg)),
            )
            maneuver_point = path[path.shape[0] // 2]
        else:
            leg1 = _sample_leg_positions(
                R0_km,
                V0_store[i],
                dt_0m_s,
                mu_sun_km3_s2,
                num_samples=max(20, int(num_samples_leg * max(eta, 0.1))),
            )
            leg2_back = _sample_leg_positions(
                Rf_km,
                -Vf_store[i],
                dt_mf_s,
                mu_sun_km3_s2,
                num_samples=max(20, int(num_samples_leg * max(1.0 - eta, 0.1))),
            )
            leg2 = leg2_back[::-1]
            path = np.vstack((leg1, leg2[1:, :]))
            maneuver_point = leg1[-1]

        color = cmap(i / max(n_sol - 1, 1))
        ax.plot(
            path[:, 0],
            path[:, 1],
            linewidth=2.0,
            color=color,
            label=f"SC dV={dv_i:.3f} km/s",
        )
        ax.scatter([maneuver_point[0]], [maneuver_point[1]], color=color, s=20, marker="o")

    ax.scatter([float(R0_km[0])], [float(R0_km[1])], color="black", s=45, marker="s", label="Departure")
    ax.scatter([float(Rf_km[0])], [float(Rf_km[1])], color="red", s=45, marker="^", label="Arrival")

    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x [km]")
    ax.set_ylabel("y [km]")
    ax.set_title("Earth-Earth DSM Trajectory Overlay (Heliocentric)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=9)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":

    """Earth->Earth DSM placement demo with external batched Lambert pre-solve."""
    repo_root = Path(__file__).resolve().parent.parent
    metakernel_path = repo_root / "star" / "METAKERN.tm"
    if not metakernel_path.exists():
        raise FileNotFoundError(f"Meta-kernel not found: {metakernel_path}")

    spice.kclear()
    spice.furnsh(str(metakernel_path))
    print(f"Loaded SPICE metakernel: {metakernel_path}")

    body0 = "399"
    body1 = "399"
    body_sun = "10"

    dV_norm_list_km_s = np.asarray([0.01, 0.02, 0.03], dtype=float)
    mu_sun = float(get_gm(body_sun)[0])
    mu_earth = float(get_gm(body0)[0])
    radius_earth = float(get_radii(body0)[0])
    lambert_nrev = 1
    lambert_hz = 1

    t0_utc_list = [
        "2030 DEC 11 00:00:00 TDB",
        "2031 JUN 01 00:00:00 TDB",
        "2032 JAN 20 00:00:00 TDB",
    ]
    tf_utc_list = [
        "2033 JAN 10 00:00:00 TDB",
        "2033 OCT 15 00:00:00 TDB",
        "2034 JUN 20 00:00:00 TDB",
    ]

    dep_rows = []
    for t0_utc in t0_utc_list:
        t0_et_s = float(spice.str2et(t0_utc))
        state0, _ = spice.spkezr(body0, t0_et_s, DEFAULT_FRAME, "NONE", DEFAULT_OBSERVER)
        state0 = np.asarray(state0, dtype=float)
        dep_rows.append(
            {
                "t0_utc": t0_utc,
                "t0_et_s": t0_et_s,
                "R0_km": state0[:3].copy(),
                "V_P0_km_s": state0[3:].copy(),
            }
        )

    arr_rows = []
    for tf_utc in tf_utc_list:
        tf_et_s = float(spice.str2et(tf_utc))
        statef, _ = spice.spkezr(body1, tf_et_s, DEFAULT_FRAME, "NONE", DEFAULT_OBSERVER)
        statef = np.asarray(statef, dtype=float)
        arr_rows.append(
            {
                "tf_utc": tf_utc,
                "tf_et_s": tf_et_s,
                "Rf_km": statef[:3].copy(),
                "V_Pf_km_s": statef[3:].copy(),
            }
        )

    window_rows = []
    for dep in dep_rows:
        for arr in arr_rows:
            if arr["tf_et_s"] <= dep["t0_et_s"]:
                continue
            window_rows.append(
                {
                    "arc_id": len(window_rows),
                    "t0_utc": dep["t0_utc"],
                    "tf_utc": arr["tf_utc"],
                    "t0_et_s": dep["t0_et_s"],
                    "tf_et_s": arr["tf_et_s"],
                    "R0_km": dep["R0_km"].copy(),
                    "V_P0_km_s": dep["V_P0_km_s"].copy(),
                    "Rf_km": arr["Rf_km"].copy(),
                    "V_Pf_km_s": arr["V_Pf_km_s"].copy(),
                }
            )

    if not window_rows:
        raise RuntimeError("No valid (t0, tf) windows with tf > t0 were generated.")

    R0_batch_km = np.vstack([row["R0_km"] for row in window_rows])
    Rf_batch_km = np.vstack([row["Rf_km"] for row in window_rows])
    V_P0_batch_km_s = np.vstack([row["V_P0_km_s"] for row in window_rows])
    V_Pf_batch_km_s = np.vstack([row["V_Pf_km_s"] for row in window_rows])
    t0_batch_et_s = np.asarray([row["t0_et_s"] for row in window_rows], dtype=float)
    tf_batch_et_s = np.asarray([row["tf_et_s"] for row in window_rows], dtype=float)
    tof_batch_s = np.asarray(
        [row["tf_et_s"] - row["t0_et_s"] for row in window_rows],
        dtype=float,
    )

    # batch Lambert solve 
    lambert_v0_km_s_all, lambert_vf_km_s_all, _, lambert_nrev_all, lambert_src_index = lambert_batch(
        R0_batch_km,
        Rf_batch_km,
        tof_batch_s,
        mu=mu_sun,
        N=lambert_nrev,
        hz=lambert_hz,
        sweep_rev=False,
    )

    lambert_v0_km_s_all = np.asarray(lambert_v0_km_s_all, dtype=float).reshape(-1, 3)
    lambert_vf_km_s_all = np.asarray(lambert_vf_km_s_all, dtype=float).reshape(-1, 3)
    lambert_nrev_all = np.asarray(lambert_nrev_all, dtype=int).reshape(-1)
    lambert_src_index = np.asarray(lambert_src_index, dtype=int).reshape(-1)

    if lambert_src_index.size == 0:
        raise RuntimeError("Batched Lambert pre-solve returned no valid seeds.")

    lambert_seed_count = np.bincount(lambert_src_index, minlength=len(window_rows))
    print("Pre-solved Lambert seed counts per arc:")
    print(f"{'arc':>4} {'t0_utc':>24} {'tf_utc':>24} {'tof[d]':>10} {'n_seed':>8}")
    for i, row in enumerate(window_rows):
        tof_days_i = (row["tf_et_s"] - row["t0_et_s"]) / SECONDS_PER_DAY
        print(
            f"{i:4d} {row['t0_utc']:>24} {row['tf_utc']:>24} "
            f"{tof_days_i:10.2f} {int(lambert_seed_count[i]):8d}"
        )

    R0_seed_km = R0_batch_km[lambert_src_index]
    Rf_seed_km = Rf_batch_km[lambert_src_index]
    V_P0_seed_km_s = V_P0_batch_km_s[lambert_src_index]
    V_Pf_seed_km_s = V_Pf_batch_km_s[lambert_src_index]
    t0_seed_et_s = t0_batch_et_s[lambert_src_index]
    tf_seed_et_s = tf_batch_et_s[lambert_src_index]

    (
        eta_store,
        V0_store,
        Vf_store,
        DV_store,
        Vinf_0_store,
        Vinf_f_store,
        lev_store,
        arc_store,
        _seed_store,
    ) = solve_arc(
        R0_seed_km,
        Rf_seed_km,
        V_P0_seed_km_s,
        V_Pf_seed_km_s,
        lambert_v0_km_s_all,
        lambert_vf_km_s_all,
        t0_seed_et_s,
        tf_seed_et_s,
        lambert_nrev_all,
        dV_norm_list_km_s,
        mu_sun,
        obj_type=1,
        additional_args={
            "rp_ratio": 1.1,
            "body_0_mu_km3_s2": mu_earth,
            "body_f_mu_km3_s2": mu_earth,
            "body_0_radius_km": radius_earth,
            "body_f_radius_km": radius_earth,
        },
        max_vinf0_km_s=20,
        max_vinff_km_s=20,
        debug=False,
        lev_type=["+-", "-+"],
    )
    arc_window_store = lambert_src_index[np.asarray(arc_store, dtype=int)]

    stacked_rows = []
    for i in range(DV_store.size):
        arc_idx = int(arc_window_store[i])
        row = window_rows[arc_idx]
        tof_days = (row["tf_et_s"] - row["t0_et_s"]) / SECONDS_PER_DAY
        stacked_rows.append(
            {
                "arc_id": arc_idx,
                "t0_utc": row["t0_utc"],
                "tf_utc": row["tf_utc"],
                "tof_days": tof_days,
                "lev": str(lev_store[i]),
                "dV_km_s": float(DV_store[i]),
                "eta": float(eta_store[i]),
                "vinf0_km_s": float(np.linalg.norm(Vinf_0_store[i])),
                "vinff_km_s": float(np.linalg.norm(Vinf_f_store[i])),
            }
        )

    if not stacked_rows:
        raise RuntimeError("No solve_arc solutions remained after constraints.")

    print()
    print("Stacked solve_arc solutions:")
    print(
        f"{'arc':>4} {'t0_utc':>24} {'tf_utc':>24} {'tof[d]':>10} "
        f"{'lev':>4} {'dV[km/s]':>10} {'eta':>10} {'|Vinf_0|':>12} {'|Vinf_f|':>12}"
    )
    for row in stacked_rows:
        print(
            f"{row['arc_id']:4d} {row['t0_utc']:>24} {row['tf_utc']:>24} {row['tof_days']:10.2f} "
            f"{row['lev']:>4} {row['dV_km_s']:10.4f} {row['eta']:10.4f} "
            f"{row['vinf0_km_s']:12.6f} {row['vinff_km_s']:12.6f}"
        )

    if DV_store.size > 0:
        arc_plot = int(arc_window_store[0])
        plot_mask = arc_window_store == arc_plot
        plot_row = window_rows[arc_plot]
        plot_solve_arc_overlay(
            t0_et_s=plot_row["t0_et_s"],
            tf_et_s=plot_row["tf_et_s"],
            R0_km=plot_row["R0_km"],
            Rf_km=plot_row["Rf_km"],
            eta_store=eta_store[plot_mask],
            V0_store=V0_store[plot_mask],
            Vf_store=Vf_store[plot_mask],
            DV_store=DV_store[plot_mask],
            mu_sun_km3_s2=mu_sun,
            dep_body_id=body0,
            arr_body_id=body1,
            frame=DEFAULT_FRAME,
            observer=DEFAULT_OBSERVER,
            num_samples_leg=500,
            num_samples_orbit=500,
        )

    spice.kclear()

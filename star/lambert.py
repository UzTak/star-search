""" 
Vectorized Lambert solver. 
Ref: Arora and Russell, "A FAST AND ROBUST MULTIPLE REVOLUTION LAMBERT ALGORITHM USING A COSINE TRANSFORMATION" (2013)

See NOTATION.md for `n_rev` sign conventions and `lambert_hz`.
"""
import sys
from typing import Sequence, Tuple

import numpy as np
from numba import njit, prange

sys.modules.setdefault("lambert", sys.modules[__name__])


@njit(cache=True, fastmath=True)
def _safe_div_scalar_jit(num, den, eps=1e-15):
    if np.abs(den) < eps:
        den = eps if den >= 0.0 else -eps
    return num / den


@njit(cache=True, fastmath=True)
def _W_and_derivs_scalar_jit(k, N=0, series_eps=2e-2, delta_m=1e-12):
    sqrt2 = np.sqrt(2.0)
    m = 2.0 - k * k

    use_ws = (int(N) == 0) and (np.abs(k - sqrt2) <= series_eps)
    use_k0 = np.abs(k) < 1e-3
    ell = (k < sqrt2 - series_eps) and (not use_ws) and (not use_k0)
    hyp = (k > sqrt2 + series_eps) and (not use_ws) and (not use_k0)

    if ell:
        mi_pos = m if m > delta_m else delta_m
        acos_arg = 1.0 - mi_pos
        if acos_arg < -1.0:
            acos_arg = -1.0
        elif acos_arg > 1.0:
            acos_arg = 1.0
        denom = np.sqrt(mi_pos * mi_pos * mi_pos)
        sign_k = np.sign(k)
        W = (((1.0 - sign_k) * np.pi + sign_k * np.arccos(acos_arg) + 2.0 * np.pi * N) / denom) - (k / mi_pos)
        dW = (-2.0 + 3.0 * W * k) / mi_pos
        ddW = (5.0 * dW * k + 3.0 * W) / mi_pos
        return W, dW, ddW

    if hyp:
        mj = m
        mj_abs = -mj if (-mj) > delta_m else delta_m
        acosh_arg = 1.0 - mj
        if acosh_arg < 1.0:
            acosh_arg = 1.0
        denom = mj_abs * np.sqrt(mj_abs)
        W = -np.arccosh(acosh_arg) / denom - (k / mj)
        dW = (-2.0 + 3.0 * W * k) / mj
        ddW = (5.0 * dW * k + 3.0 * W) / mj
        return W, dW, ddW

    if use_ws:
        # Eq. 27-28: W_s applies only near +sqrt(2) for the zero-rev case.
        v = k - sqrt2
        W = (sqrt2 / 3.0) + (-1.0 / 5.0) * v + (2.0 * sqrt2 / 35.0) * v**2 + (-2.0 / 63.0) * v**3 + (2.0 * sqrt2 / 231.0) * v**4 + (-2.0 / 429.0) * v**5 + (8.0 * sqrt2 / 6435.0) * v**6 + (-8.0 / 12155.0) * v**7 + (8.0 * sqrt2 / 46189.0) * v**8
        dW = (-1.0 / 5.0) + (4.0 * sqrt2 / 35.0) * v + (-6.0 / 63.0) * v**2 + (8.0 * sqrt2 / 231.0) * v**3 + (-10.0 / 429.0) * v**4 + (48.0 * sqrt2 / 6435.0) * v**5 + (-56.0 / 12155.0) * v**6 + (64.0 * sqrt2 / 46189.0) * v**7
        ddW = (4.0 / 35.0) + (-12.0 / 63.0) * v + (24.0 / 231.0) * v**2 + (-40.0 / 429.0) * v**3 + (240.0 / 6435.0) * v**4 + (-336.0 / 12155.0) * v**5 + (448.0 / 46189.0) * v**6
        return W, dW, ddW

    if use_k0:
        # Eq. 59: avoid precision loss when |k| is very small.
        coeff = 2.0 * N * np.pi + np.pi
        W = (coeff / 4.0) * sqrt2 - k + (3.0 * coeff / 16.0) * sqrt2 * k**2 - (2.0 / 3.0) * k**3 + (15.0 / 128.0) * coeff * sqrt2 * k**4 - (2.0 / 5.0) * k**5
        dW = -1.0 + (3.0 * coeff / 8.0) * sqrt2 * k - 2.0 * k**2 + (15.0 / 32.0) * coeff * sqrt2 * k**3 - 2.0 * k**4
        ddW = (3.0 * coeff / 8.0) * sqrt2 - 4.0 * k + (45.0 / 32.0) * coeff * sqrt2 * k**2 - 8.0 * k**3
        return W, dW, ddW

    # Fall back to the exact elliptic expression outside the special series regions.
    mi_pos = m if m > delta_m else delta_m
    acos_arg = 1.0 - mi_pos
    if acos_arg < -1.0:
        acos_arg = -1.0
    elif acos_arg > 1.0:
        acos_arg = 1.0
    denom = np.sqrt(mi_pos * mi_pos * mi_pos)
    sign_k = np.sign(k)
    W = (((1.0 - sign_k) * np.pi + sign_k * np.arccos(acos_arg) + 2.0 * np.pi * N) / denom) - (k / mi_pos)
    dW = (-2.0 + 3.0 * W * k) / mi_pos
    ddW = (5.0 * dW * k + 3.0 * W) / mi_pos
    return W, dW, ddW


@njit(cache=True, fastmath=True, parallel=True)
def _halley_solve_rows_jit(k0, tau, S, tof, N, series_eps, tol):
    n = k0.shape[0]
    sqrt2 = np.sqrt(2.0)
    iter_cap = 15  # hard coded (based on Arora & Russell's experiments); can be made an argument if desired.
    k_out = np.empty(n, dtype=np.float64)
    converged = np.zeros(n, dtype=np.bool_)

    for i in prange(n):
        ki = k0[i]
        if (not np.isfinite(ki)) or (ki <= -sqrt2):
            k_out[i] = ki
            continue

        tau_i = tau[i]
        S_i = S[i]
        tof_i = tof[i]
        ok = False

        for _ in range(iter_cap):
            W, dW, ddW = _W_and_derivs_scalar_jit(ki, N, series_eps, 1e-12)

            c = _safe_div_scalar_jit(1.0 - ki * tau_i, tau_i, 1e-15)
            sqrt_arg_1 = 1.0 - ki * tau_i
            if sqrt_arg_1 < 0.0:
                sqrt_arg_1 = 0.0

            tof_k = S_i * np.sqrt(sqrt_arg_1) * (tau_i + (1.0 - ki * tau_i) * W)
            L = tof_k - tof_i

            if np.abs(L) <= tol * max(1.0, np.abs(tof_i)):
                ok = True
                break

            sqrt_arg_2 = c * tau_i
            if sqrt_arg_2 < 0.0:
                sqrt_arg_2 = 0.0

            dT = -_safe_div_scalar_jit(tof_k, 2.0 * c, 1e-15) + S_i * tau_i * np.sqrt(sqrt_arg_2) * (dW * c - W)
            ddT = -_safe_div_scalar_jit(tof_k, 4.0 * c * c, 1e-15) + S_i * tau_i * np.sqrt(sqrt_arg_2) * (_safe_div_scalar_jit(W, c, 1e-15) + c * ddW - 3.0 * dW)
            denom = dT - _safe_div_scalar_jit(L * ddT, 2.0 * dT, 1e-15)
            dk = _safe_div_scalar_jit(L, denom, 1e-15)

            if (not np.isfinite(dk)) or (not np.isfinite(L)) or (not np.isfinite(dT)) or (not np.isfinite(ddT)):
                ok = False
                break

            ki = ki - dk
            if (not np.isfinite(ki)) or (ki <= -sqrt2):
                ok = False
                break

        k_out[i] = ki
        converged[i] = ok

    return k_out, converged


@njit(cache=True, fastmath=True, parallel=True)
def _minimize_tof_rows_jit(k0, tau, S, N, series_eps, tol):
    """Refine `k_b` by solving dTOF/dk = 0 with Newton updates."""

    n = k0.shape[0]
    sqrt2 = np.sqrt(2.0)
    iter_cap = 20
    k_out = np.empty(n, dtype=np.float64)
    tof_out = np.empty(n, dtype=np.float64)
    converged = np.zeros(n, dtype=np.bool_)

    for i in prange(n):
        ki = k0[i]
        tau_i = tau[i]
        S_i = S[i]
        ok = False
        tof_k = np.nan

        if (not np.isfinite(ki)) or (ki <= -sqrt2):
            k_out[i] = ki
            tof_out[i] = tof_k
            continue

        for _ in range(iter_cap):
            W, dW, ddW = _W_and_derivs_scalar_jit(ki, N, series_eps, 1e-12)

            c = _safe_div_scalar_jit(1.0 - ki * tau_i, tau_i, 1e-15)
            sqrt_arg_1 = 1.0 - ki * tau_i
            if sqrt_arg_1 < 0.0:
                sqrt_arg_1 = 0.0
            tof_k = S_i * np.sqrt(sqrt_arg_1) * (tau_i + (1.0 - ki * tau_i) * W)

            sqrt_arg_2 = c * tau_i
            if sqrt_arg_2 < 0.0:
                sqrt_arg_2 = 0.0
            dT = -_safe_div_scalar_jit(tof_k, 2.0 * c, 1e-15) + S_i * tau_i * np.sqrt(sqrt_arg_2) * (dW * c - W)
            ddT = -_safe_div_scalar_jit(tof_k, 4.0 * c * c, 1e-15) + S_i * tau_i * np.sqrt(sqrt_arg_2) * (_safe_div_scalar_jit(W, c, 1e-15) + c * ddW - 3.0 * dW)

            if np.abs(dT) <= tol * max(1.0, np.abs(tof_k)):
                ok = True
                break

            if (not np.isfinite(dT)) or (not np.isfinite(ddT)):
                break

            dk = _safe_div_scalar_jit(dT, ddT, 1e-15)
            if not np.isfinite(dk):
                break

            ki = ki - dk
            if (not np.isfinite(ki)) or (ki <= -sqrt2) or (ki >= sqrt2):
                ok = False
                break

        k_out[i] = ki
        tof_out[i] = tof_k
        converged[i] = ok

    return k_out, tof_out, converged


@njit(cache=True, fastmath=True)
def _tof_and_dtof_scalar_jit(k, tau_i, S_i, N, series_eps):
    W, dW, _ = _W_and_derivs_scalar_jit(k, N, series_eps, 1e-12)

    c = _safe_div_scalar_jit(1.0 - k * tau_i, tau_i, 1e-15)
    sqrt_arg_1 = 1.0 - k * tau_i
    if sqrt_arg_1 < 0.0:
        sqrt_arg_1 = 0.0
    tof_k = S_i * np.sqrt(sqrt_arg_1) * (tau_i + (1.0 - k * tau_i) * W)

    sqrt_arg_2 = c * tau_i
    if sqrt_arg_2 < 0.0:
        sqrt_arg_2 = 0.0
    dT = -_safe_div_scalar_jit(tof_k, 2.0 * c, 1e-15) + S_i * tau_i * np.sqrt(sqrt_arg_2) * (dW * c - W)
    return tof_k, dT


@njit(cache=True, fastmath=True, parallel=True)
def _bracketed_root_solve_rows_jit(k0, klo, khi, tau, S, tof, N, series_eps, tol):
    """Safeguarded Newton/bisection solve on a branch-bounded multi-rev interval."""

    n = k0.shape[0]
    sqrt2 = np.sqrt(2.0)
    iter_cap = 40
    k_out = np.empty(n, dtype=np.float64)
    converged = np.zeros(n, dtype=np.bool_)

    for i in prange(n):
        lo = klo[i]
        hi = khi[i]
        if lo > hi:
            tmp = lo
            lo = hi
            hi = tmp

        if (not np.isfinite(lo)) or (not np.isfinite(hi)) or (not np.isfinite(k0[i])) or (lo >= hi):
            k_out[i] = k0[i]
            continue

        if lo <= -sqrt2:
            lo = -sqrt2 + 1e-12
        if hi >= sqrt2:
            hi = sqrt2 - 1e-12

        tau_i = tau[i]
        S_i = S[i]
        tof_i = tof[i]

        tof_lo, _ = _tof_and_dtof_scalar_jit(lo, tau_i, S_i, N, series_eps)
        tof_hi, _ = _tof_and_dtof_scalar_jit(hi, tau_i, S_i, N, series_eps)
        flo = tof_lo - tof_i
        fhi = tof_hi - tof_i

        if (not np.isfinite(flo)) or (not np.isfinite(fhi)):
            k_out[i] = k0[i]
            continue

        if np.abs(flo) <= tol * max(1.0, np.abs(tof_i)):
            k_out[i] = lo
            converged[i] = True
            continue
        if np.abs(fhi) <= tol * max(1.0, np.abs(tof_i)):
            k_out[i] = hi
            converged[i] = True
            continue

        if flo * fhi > 0.0:
            k_out[i] = k0[i]
            continue

        ki = k0[i]
        if (not np.isfinite(ki)) or (ki <= lo) or (ki >= hi):
            ki = 0.5 * (lo + hi)

        tof_k, dT = _tof_and_dtof_scalar_jit(ki, tau_i, S_i, N, series_eps)
        fk = tof_k - tof_i
        if not np.isfinite(fk):
            ki = 0.5 * (lo + hi)
            tof_k, dT = _tof_and_dtof_scalar_jit(ki, tau_i, S_i, N, series_eps)
            fk = tof_k - tof_i

        ok = False
        for _ in range(iter_cap):
            if np.abs(fk) <= tol * max(1.0, np.abs(tof_i)):
                ok = True
                break

            knew = np.nan
            if np.isfinite(dT) and (np.abs(dT) > 1e-15):
                knew = ki - fk / dT
            if (not np.isfinite(knew)) or (knew <= lo) or (knew >= hi):
                knew = 0.5 * (lo + hi)

            tof_new, dT_new = _tof_and_dtof_scalar_jit(knew, tau_i, S_i, N, series_eps)
            fnew = tof_new - tof_i
            if not np.isfinite(fnew):
                knew = 0.5 * (lo + hi)
                tof_new, dT_new = _tof_and_dtof_scalar_jit(knew, tau_i, S_i, N, series_eps)
                fnew = tof_new - tof_i
                if not np.isfinite(fnew):
                    break

            if flo * fnew <= 0.0:
                hi = knew
                fhi = fnew
            else:
                lo = knew
                flo = fnew

            ki = knew
            fk = fnew
            dT = dT_new

        k_out[i] = ki
        converged[i] = ok

    return k_out, converged

def _W_and_derivs(k, N=0, series_eps=2e-2, delta_m=1e-12):
    k = np.asarray(k, float)
    sqrt2 = np.sqrt(2.0)
    m = 2.0 - k**2

    # Mirror the scalar helper: only the zero-rev +sqrt(2) neighborhood uses W_s.
    # For N>0, values in the near-sqrt(2) gap still fall back to the elliptic form.
    use_ws = (int(N) == 0) & (np.abs(k - sqrt2) <= series_eps)
    use_k0 = np.abs(k) < 1e-3
    special = use_ws | use_k0
    ell = (k < sqrt2 + series_eps) & (~special)
    hyp = (k > sqrt2 + series_eps) & (~special)

    W  = np.full_like(k, np.nan, dtype=float)
    dW = np.full_like(k, np.nan, dtype=float)
    ddW= np.full_like(k, np.nan, dtype=float)

    # --- Elliptic (use same indices for compute and assign) ---
    if np.any(ell):
        i  = np.where(ell)[0]
        ki = k[i]
        mi = m[i]                         # should be >0
        mi_pos = np.maximum(mi, delta_m)  # guard tiny/negative
        acos_arg = np.clip(1.0 - mi_pos, -1.0, 1.0)
        denom = np.sqrt(mi_pos**3)   
        Wi  = (((1 - np.sign(ki))*np.pi + np.sign(ki)*np.arccos(acos_arg) + 2*np.pi*N) / denom) - (ki/mi_pos)
        dWi = (-2 + 3*Wi*ki) / mi_pos
        ddWi= (5*dWi*ki + 3*Wi) / mi_pos
        W[i], dW[i], ddW[i] = Wi, dWi, ddWi

    # --- Hyperbolic ---
    if np.any(hyp):
        j  = np.where(hyp)[0]
        kj = k[j]
        mj = m[j]                          # < 0
        mj_abs = np.maximum(-mj, delta_m)  # positive
        acosh_arg = np.maximum(1.0 - mj, 1.0)
        denom  = mj_abs * np.sqrt(mj_abs)    
        Wj  = - np.arccosh(acosh_arg)/denom - (kj/mj)
        dWj  = (-2 + 3*Wj*kj) / mj       # IMPORTANT: the original paper (Eq. 39) has a typo here (I think) 
        ddWj = (5*dWj*kj + 3*Wj) / mj    # IMPORTANT: the original paper (Eq. 39) has a typo here (I think) 
        # ddWj = (3*Wj + kj*Wj) / (-mj)    
        W[j], dW[j], ddW[j] = Wj, dWj, ddWj

    # --- Series near +sqrt(2) only for N=0 (Eq. 27-28) ---
    if np.any(use_ws):
        s  = np.where(use_ws)[0]
        ks = k[s]
        v  = ks - sqrt2
        Ws  = (sqrt2/3) + (-1/5)*v + (2*sqrt2/35)*v**2 + (-2/63)*v**3 + (2*sqrt2/231)*v**4 + (-2/429)*v**5 + (8*sqrt2/6435)*v**6 + (-8/12155)*v**7 + (8*sqrt2/46189)*v**8
        dWs = (-1/5) + (4*sqrt2/35)*v + (-6/63)*v**2 + (8*sqrt2/231)*v**3 + (-10/429)*v**4 + (48*sqrt2/6435)*v**5 + (-56/12155)*v**6 + (64*sqrt2/46189)*v**7
        ddWs= (4/35) + (-12/63)*v + (24/231)*v**2 + (-40/429)*v**3 + (240/6435)*v**4 + (-336/12155)*v**5 + (448/46189)*v**6
        W[s], dW[s], ddW[s] = Ws, dWs, ddWs

    # Eq. 59: avoid precision loss when |k| is very small.
    if np.any(use_k0):
        l = np.where(use_k0)[0]
        kl = k[l]
        coeff = 2.0 * N * np.pi + np.pi
        Wl = (coeff/4.0)*sqrt2 - kl + (3.0*coeff/16.0)*sqrt2*kl**2 - (2.0/3.0)*kl**3 + (15.0/128.0)*coeff*sqrt2*kl**4 - (2.0/5.0)*kl**5
        dWl = -1.0 + (3.0*coeff/8.0)*sqrt2*kl - 2.0*kl**2 + (15.0/32.0)*coeff*sqrt2*kl**3 - 2.0*kl**4
        ddWl= (3.0*coeff/8.0)*sqrt2 - 4.0*kl + (45.0/32.0)*coeff*sqrt2*kl**2 - 8.0*kl**3
        W[l], dW[l], ddW[l] = Wl, dWl, ddWl

    assert not np.any(np.isnan(W)), "NaN values were generated in W!"
    return W, dW, ddW


@njit(cache=True, fastmath=True)
def _table7_W_scalar_jit(k, N, k_snap_eps):
    sqrt2 = np.sqrt(2.0)

    if np.abs(k - (-1.0 - 2.0 * sqrt2) / 3.0) <= k_snap_eps:
        return 27.25239909 + 27.75304668 * N, True
    if np.abs(k + 1.0) <= k_snap_eps:
        return 5.71238898 + 2.0 * np.pi * N, True
    if np.abs(k + 0.5) <= k_snap_eps:
        return 1.95494660 + 2.71408094 * N, True
    if np.abs(k) <= k_snap_eps:
        return (sqrt2 / 4.0) * (np.pi + 2.0 * np.pi * N), True
    if np.abs(k - 0.5) <= k_snap_eps:
        return 0.75913433 + 2.71408094 * N, True
    if np.abs(k - 1.0) <= k_snap_eps:
        return 0.57079632 + 2.0 * np.pi * N, True
    if np.abs(k - sqrt2) <= k_snap_eps:
        return 0.50064759 + 27.75304668 * N, True

    return 0.0, False


@njit(cache=True, fastmath=True)
def _tof_from_k_scalar_row_jit(k, tau_i, S_i, N, series_eps, k_snap_eps):
    Wk, snapped = _table7_W_scalar_jit(k, N, k_snap_eps)
    if not snapped:
        Wk, _, _ = _W_and_derivs_scalar_jit(k, N, series_eps, 1e-12)

    sqrt_arg = 1.0 - k * tau_i
    if sqrt_arg < 0.0:
        sqrt_arg = 0.0

    return S_i * np.sqrt(sqrt_arg) * (tau_i + sqrt_arg * Wk)


@njit(cache=True, fastmath=True, parallel=True)
def _tof_from_k_rows_jit(k, tau, S, N, series_eps, k_snap_eps):
    n = k.shape[0]
    out = np.empty(n, dtype=np.float64)

    for i in prange(n):
        out[i] = _tof_from_k_scalar_row_jit(k[i], tau[i], S[i], N, series_eps, k_snap_eps)

    return out


@njit(cache=True, fastmath=True, parallel=True)
def _tof_from_k_const_jit(k, tau, S, N, series_eps, k_snap_eps):
    n = tau.shape[0]
    out = np.empty(n, dtype=np.float64)
    Wk, snapped = _table7_W_scalar_jit(k, N, k_snap_eps)

    if not snapped:
        Wk, _, _ = _W_and_derivs_scalar_jit(k, N, series_eps, 1e-12)

    for i in prange(n):
        sqrt_arg = 1.0 - k * tau[i]
        if sqrt_arg < 0.0:
            sqrt_arg = 0.0
        out[i] = S[i] * np.sqrt(sqrt_arg) * (tau[i] + sqrt_arg * Wk)

    return out


def _tof_from_k(k_in, tau_in, S_in, N, series_eps, k_snap_eps):
    tau_arr = np.asarray(tau_in, dtype=np.float64)
    S_arr = np.asarray(S_in, dtype=np.float64)
    k_arr = np.asarray(k_in)

    if tau_arr.shape != S_arr.shape:
        raise ValueError("tau and S must have matching shapes.")

    if k_arr.ndim == 0:
        return _tof_from_k_const_jit(
            float(k_arr),
            tau_arr,
            S_arr,
            int(N),
            float(series_eps),
            float(k_snap_eps),
        )

    k_vec = np.asarray(k_in, dtype=np.float64)
    if k_vec.shape != tau_arr.shape:
        raise ValueError("k must be scalar or match the tau/S shape.")

    return _tof_from_k_rows_jit(
        k_vec,
        tau_arr,
        S_arr,
        int(N),
        float(series_eps),
        float(k_snap_eps),
    )


@njit(cache=True, fastmath=True, parallel=True)
def _compute_multirev_k0_jit(tau_sel, S_sel, tof_sel, kb, Tb, bsgn, N, series_eps, k_snap_eps):
    """Fused multi-rev k0 initialisation (Tables 5-6, Arora & Russell).

    Replaces three separate _tof_from_k batch calls (Tm1, T0, T1) and the
    twelve-case _apply_eq44 Python dispatch loop with a single parallel JIT
    kernel.  Each row computes its own reference TOFs, picks its case,
    evaluates Fi = TOF(ki), and applies the Eq. 44 rational formula.
    """
    n = tau_sel.shape[0]
    sqrt2 = np.sqrt(2.0)
    ki_lp_tgt  = (1.0 + 2.0 * sqrt2) / 3.0   # Table 5 row 2 / Table 6 row 6 midpoint
    ki_sp_high = (-1.0 - 2.0 * sqrt2) / 3.0   # Table 5 row 6 / Table 6 row 2 midpoint
    alpha_mid  = 6.0 / 5.0
    Z_mid      = 0.5 ** alpha_mid

    k0_out      = np.full(n, np.nan)
    k_lower_out = np.full(n, np.nan)
    k_upper_out = np.full(n, np.nan)

    for i in prange(n):
        tau_i = tau_sel[i]
        S_i   = S_sel[i]
        tof_i = tof_sel[i]
        kb_i  = kb[i]
        Tb_i  = Tb[i]

        # Three reference TOFs computed inline - no Python dispatch overhead
        Tm1_i = _tof_from_k_scalar_row_jit(-1.0, tau_i, S_i, N, series_eps, k_snap_eps)
        T0_i  = _tof_from_k_scalar_row_jit( 0.0, tau_i, S_i, N, series_eps, k_snap_eps)
        T1_i  = _tof_from_k_scalar_row_jit( 1.0, tau_i, S_i, N, series_eps, k_snap_eps)

        want_lp = bsgn[i] > 0
        kb_m1 = (kb_i >= 1.0)
        kb_m2 = (kb_i >= 0.0) and (kb_i < 1.0)
        kb_m3 = (kb_i >= -1.0) and (kb_i < 0.0)
        kb_m4 = (kb_i < -1.0)

        kn = np.nan; km = np.nan; ki_val = np.nan
        Z = np.nan;  alpha = np.nan
        F0 = np.nan; F1 = np.nan
        use_inverse = False

        if want_lp:
            if kb_m1:
                # Table 5 row 1: lp, M1
                kn = kb_i; km = sqrt2; ki_val = 0.5 * (kb_i + sqrt2)
                Z = 0.25; alpha = 2.0
                F0 = Tb_i; F1 = 0.0; use_inverse = True
            elif kb_m2:
                if tof_i > T1_i:
                    # Table 5 row 2: lp, M2, tof > T1
                    kn = 1.0; km = sqrt2; ki_val = ki_lp_tgt
                    Z = 4.0 / 9.0; alpha = 2.0
                    F0 = T1_i; F1 = 0.0; use_inverse = True
                else:
                    # Table 5 row 3: lp, M2, tof <= T1
                    kn = kb_i; km = 1.0; ki_val = 0.5 * (1.0 + kb_i)
                    Z = 0.25; alpha = 2.0
                    F0 = Tb_i; F1 = T1_i; use_inverse = False
            elif kb_m3 or kb_m4:
                if tof_i <= T0_i:
                    # Table 6 row 4: lp, M3|M4, tof <= T0
                    kn = kb_i; km = 0.0; ki_val = 0.5 * kb_i
                    Z = 0.25; alpha = 2.0
                    F0 = Tb_i; F1 = T0_i; use_inverse = False
                elif tof_i <= T1_i:
                    # Table 6 row 5: lp, M3|M4, T0 < tof <= T1
                    kn = 0.0; km = 1.0; ki_val = 0.5
                    Z = Z_mid; alpha = alpha_mid
                    F0 = T0_i; F1 = T1_i; use_inverse = False
                else:
                    # Table 6 row 6: lp, M3|M4, tof > T1
                    kn = 1.0; km = sqrt2; ki_val = ki_lp_tgt
                    Z = 4.0 / 9.0; alpha = 2.0
                    F0 = T1_i; F1 = 0.0; use_inverse = True
        else:
            if kb_m1 or kb_m2:
                if tof_i <= T0_i:
                    # Table 5 row 4: sp, M1|M2, tof <= T0
                    kn = 0.0; km = kb_i; ki_val = 0.5 * kb_i
                    Z = Z_mid; alpha = alpha_mid
                    F0 = T0_i; F1 = Tb_i; use_inverse = False
                elif tof_i <= Tm1_i:
                    # Table 5 row 5: sp, M1|M2, T0 < tof <= Tm1
                    kn = -1.0; km = 0.0; ki_val = -0.5
                    Z = 0.5; alpha = 1.0
                    F0 = Tm1_i; F1 = T0_i; use_inverse = False
                else:
                    # Table 5 row 6: sp, M1|M2, tof > Tm1
                    kn = -1.0; km = -sqrt2; ki_val = ki_sp_high
                    Z = 4.0 / 9.0; alpha = 2.0
                    F0 = Tm1_i; F1 = 0.0; use_inverse = True
            elif kb_m4:
                # Table 6 row 1: sp, M4
                kn = kb_i; km = -sqrt2; ki_val = 0.5 * (kb_i - sqrt2)
                Z = 0.25; alpha = 2.0
                F0 = Tb_i; F1 = 0.0; use_inverse = True
            elif kb_m3:
                if tof_i > Tm1_i:
                    # Table 6 row 2: sp, M3, tof > Tm1
                    kn = -1.0; km = -sqrt2; ki_val = ki_sp_high
                    Z = 4.0 / 9.0; alpha = 2.0
                    F0 = Tm1_i; F1 = 0.0; use_inverse = True
                else:
                    # Table 6 row 3: sp, M3, tof <= Tm1
                    kn = kb_i; km = -1.0; ki_val = 0.5 * (-1.0 + kb_i)
                    Z = 0.25; alpha = 2.0
                    F0 = Tb_i; F1 = Tm1_i; use_inverse = False

        if np.isnan(kn):
            continue

        Fi    = _tof_from_k_scalar_row_jit(ki_val, tau_i, S_i, N, series_eps, k_snap_eps)
        Fstar = tof_i

        eps15 = 1e-15
        if use_inverse:
            F0u = _safe_div_scalar_jit(1.0, F0,    eps15)
            F1u = _safe_div_scalar_jit(1.0, F1,    eps15)
            Fiu = _safe_div_scalar_jit(1.0, Fi,    eps15)
            Fsu = _safe_div_scalar_jit(1.0, Fstar, eps15)
        else:
            F0u = F0; F1u = F1; Fiu = Fi; Fsu = Fstar

        num   = Z * (F0u - Fsu) * (F1u - Fiu)
        den   = (Fiu - Fsu) * (F1u - F0u) * Z + (F0u - Fiu) * (F1u - Fsu)
        ratio = _safe_div_scalar_jit(num, den, eps15)
        if ratio < 0.0:
            ratio = 0.0
        elif ratio > 1.0:
            ratio = 1.0

        x = ratio ** (1.0 / alpha)
        k0_out[i] = kn + (km - kn) * x
        if kn <= km:
            k_lower_out[i] = kn
            k_upper_out[i] = km
        else:
            k_lower_out[i] = km
            k_upper_out[i] = kn

    return k0_out, k_lower_out, k_upper_out


def _lambert_batch_core(
    r1,
    r2,
    tof,
    mu=1.0,
    N=0,
    hz=1,
    branch_sign=1,
    series_eps=2e-2,
    tol=1e-6,
    return_diagnostics=False,
    r1n_pre=None,
    r2n_pre=None,
    tau_abs_pre=None,
    S_pre=None,
):
    """
    Solve one expanded Lambert batch.

    Args:
        r1, r2: shape (B,3)
        tof: shape (B,)
        hz: +1 (ccw) or -1 (cw), scalar or shape (B,)
        branch_sign: +1 for long-period branch, -1 for short-period branch.
            Used only for N>0 and can be scalar or shape (B,).
        r1n_pre, r2n_pre, tau_abs_pre, S_pre:
            Optional precomputed geometry arrays with shape `(B,)`.
    Returns:
        v1, v2 with shape (M,3), plus source row indices shape (M,)
        into this local batch.
        When `return_diagnostics=True`, also returns local row indices for:
        root-converged rows and velocity-reconstructed rows.
    """

    # Helper for safe division
    def _safe_div(num, den, eps=1e-15):
        den = np.where(np.abs(den) < eps, np.sign(den)*eps + (den==0)*eps, den)
        return num/den

    def _empty_result():
        empty_v = np.empty((0, 3), dtype=float)
        empty_i = np.empty((0,), dtype=int)
        if return_diagnostics:
            return empty_v, empty_v.copy(), empty_i, empty_i.copy(), empty_i.copy()
        return empty_v, empty_v.copy(), empty_i

    r1 = np.asarray(r1, float); r2 = np.asarray(r2, float); tof = np.asarray(tof, float)
    N = int(N)
    assert N >= 0, "N must be >= 0."
    B = r1.shape[0]
    assert r1.shape == r2.shape == (B, 3)
    assert tof.shape == (B,)
    if B == 0:
        return _empty_result()
    k_snap_eps = 1e-7

    hz_arr = np.asarray(hz)
    if hz_arr.ndim == 0:
        hz_vec = np.full(B, int(hz_arr), dtype=int)
    else:
        if hz_arr.shape != (B,):
            raise ValueError("hz must be scalar or shape (B,).")
        hz_vec = hz_arr.astype(int)
    if np.any((hz_vec != 1) & (hz_vec != -1)):
        raise ValueError("hz entries must be +/-1.")

    branch_arr = np.asarray(branch_sign)
    if branch_arr.ndim == 0:
        bsgn_vec = np.full(B, int(branch_arr), dtype=int)
    else:
        if branch_arr.shape != (B,):
            raise ValueError("branch_sign must be scalar or shape (B,).")
        bsgn_vec = branch_arr.astype(int)
    bsgn_vec = np.where(bsgn_vec >= 0, 1, -1)

    # Geometry scalars
    if r1n_pre is None:
        r1n = np.linalg.norm(r1, axis=1)
    else:
        r1n = np.asarray(r1n_pre, dtype=float)
        if r1n.shape != (B,):
            raise ValueError("r1n_pre must have shape (B,).")

    if r2n_pre is None:
        r2n = np.linalg.norm(r2, axis=1)
    else:
        r2n = np.asarray(r2n_pre, dtype=float)
        if r2n.shape != (B,):
            raise ValueError("r2n_pre must have shape (B,).")

    if tau_abs_pre is None:
        theta = np.arctan2(np.linalg.norm(np.cross(r1, r2), axis=1), np.einsum('bi,bi->b', r1, r2))
        tau_abs = np.sqrt((r1n*r2n) * (1.0 + np.cos(theta))) / (r1n + r2n)
    else:
        tau_abs = np.asarray(tau_abs_pre, dtype=float)
        if tau_abs.shape != (B,):
            raise ValueError("tau_abs_pre must have shape (B,).")

    if S_pre is None:
        S = np.sqrt((r1n + r2n)**3 / mu)
    else:
        S = np.asarray(S_pre, dtype=float)
        if S.shape != (B,):
            raise ValueError("S_pre must have shape (B,).")

    # τ, S, T_p
    tau = hz_vec * tau_abs  # in [-1/sqrt2, 1/sqrt2]
    sqrt2 = np.sqrt(2.0)
    sqrt_arg = np.maximum(1 - sqrt2*tau, 0)
    Tp = S * np.sqrt(sqrt_arg) * (tau + sqrt2) / 3.0

    # Initial-guess masks
    elliptic = tof > Tp

    if N == 0:
        # Hyperbolic timing refs (guard domain); only used by the zero-rev path.
        disc20  = 1 - 20*tau
        disc100 = 1 - 100*tau
        T20  = S*np.sqrt(np.maximum(disc20, 0.0))  * (tau + 0.04940968903*(1 - 20*tau))
        T100 = S*np.sqrt(np.maximum(disc100, 0.0)) * (tau + 0.00999209404*(1 - 100*tau))

        k0 = np.full(B, np.nan, dtype=float)
        hyperbol = ~elliptic
        m_hm = hyperbol & (hz_vec < 0)
        m_hm_H1 = m_hm & (tof >= T20)    # Eq. 44
        m_hm_H2 = m_hm & (tof <  T20)    # Eq. 47
        m_hp = hyperbol & (hz_vec > 0)

        # --- Hyperbolic, hz = -1 / hz = +1 ---
        if np.any(m_hp):
            idx = np.where(m_hp)[0]
            tau_safe = np.where(np.abs(tau[idx]) < 1e-12, np.sign(tau[idx])*1e-12 + (tau[idx]==0)*1e-12, tau[idx])
            kn, km = sqrt2, 1.0 / tau_safe
            ki = 0.5*(kn + km)
            Z, alpha = 1/sqrt2, 0.5
            ki = np.where(ki <= -sqrt2 + 1e-12, -sqrt2 + 1e-12, ki)
            Wki,_,_ = _W_and_derivs(ki, N, series_eps)
            Si, taui = S[idx], tau[idx]
            sqrt_arg = np.maximum(1 - ki*taui, 0)
            Fi = Si*np.sqrt(sqrt_arg) * (taui + (1 - ki*taui)*Wki)
            F0, F1, Fstar = Tp[idx], 0.0, tof[idx]
            x = ( (Z*(F0-Fstar)*(F1-Fi)) / ((Fi-Fstar)*(F1-F0)*Z + (F0-Fi)*(F1-Fstar)) )**(1/alpha)
            k0[idx] = kn + (km - kn)*x

        if np.any(m_hm_H1):
            kn, km, ki, Z, alpha = sqrt2, 20.0, (2*sqrt2+20.0)/3.0, 1/3, 1.0
            idx = np.where(m_hm_H1)[0]
            Wki,_,_ = _W_and_derivs(np.full(idx.size, ki), N, series_eps)
            Si, taui = S[idx], tau[idx]
            sqrt_arg = np.maximum(1 - ki*taui, 0)
            Fi = Si*np.sqrt(sqrt_arg) * (taui + (1 - ki*taui)*Wki)
            F0, F1, Fstar = Tp[idx], T20[idx], tof[idx]
            x = ( (Z*(F0-Fstar)*(F1-Fi)) / ((Fi-Fstar)*(F1-F0)*Z + (F0-Fi)*(F1-Fstar)) )**(1/alpha)
            k0[idx] = kn + (km - kn)*x  # Eq. 44 

        if np.any(m_hm_H2):
            idx = np.where(m_hm_H2)[0]
            T0, T1, tstar = T20[idx], T100[idx], tof[idx]
            num = T1*(T0 - tstar)*10.0 - T0*np.sqrt(20.0)*(T1 - tstar)
            den = tstar*(T0 - T1)
            k0[idx] = (num/den)**2   # Eq. 47

        # --- Elliptic N=0 (E1–E4) ---
        if np.any(elliptic):
            idx_all = np.where(elliptic)[0]
            Ssel = S[idx_all][:, None]          # (n,1)
            taus = tau[idx_all][:, None]        # (n,1)

            # tabulated k and constants (Arora–Russell Table 1)
            kvec = np.array([-1.41, -1.38, -1.0, -0.5, 1/np.sqrt(2.0)])[None, :]  # (1,5)
            Wvec = np.array([4839.684497246, 212.087279879, 5.712388981,
                            1.954946607, 0.6686397730])[None, :]                 # (1,5)

            # Build TE in the COMPRESSED space (n,5)
            TE = Ssel * np.sqrt(np.maximum(1 - kvec*taus, 0.0)) * (taus + (1 - kvec*taus)*Wvec)
            Tm141, Tm138, Tm1, Tmhalf, T1s2 = TE.T          # each shape (n,)
            T0 = (S[idx_all] * (np.sqrt(2.0)/4*np.pi + tau[idx_all]))  # (n,)
            tof_sel = tof[idx_all]                           # (n,)

            # Mask out rows below zero-rev minimum TOF; handled as normal infeasible rows.
            feasible_ell = tof_sel < Tm141
            if np.any(feasible_ell):
                idx_use = idx_all[feasible_ell]
                Tm141 = Tm141[feasible_ell]
                Tm138 = Tm138[feasible_ell]
                Tm1 = Tm1[feasible_ell]
                Tmhalf = Tmhalf[feasible_ell]
                T1s2 = T1s2[feasible_ell]
                T0 = T0[feasible_ell]
                tof_sel = tof_sel[feasible_ell]

            # E1: tof >= T0 -> Eq. (44), Z=1/2, α=1, Fi = T(k=1/√2)
                mE1 = (tof_sel <= T0)
                if np.any(mE1):
                    idx = idx_use[mE1]
                    F0, F1, Fi, Fstar = T0[mE1], Tp[idx], T1s2[mE1], tof[idx]
                    Z, alpha = 1/2, 1.0
                    num = Z*(F0 - Fstar)*(F1 - Fi)
                    den = (Fi - Fstar)*(F1 - F0)*Z + (F0 - Fi)*(F1 - Fstar)
                    x = np.clip((_safe_div(num, den))**(1/alpha), 0.0, 1.0)
                    kn, km = 0.0, np.sqrt(2.0)
                    k0[idx] = kn + (km - kn)*x

                # E2: Tm1 < tof < T0 -> Eq. (44), Z=1/2, α=1, Fi = T(k=-1/2)
                mE2 = (tof_sel < Tm1) & (tof_sel > T0)
                if np.any(mE2):
                    idx = idx_use[mE2]
                    F0, F1, Fi, Fstar = T0[mE2], Tm1[mE2], Tmhalf[mE2], tof[idx]
                    Z, alpha = 0.5, 1.0
                    num = Z*(F0 - Fstar)*(F1 - Fi)
                    den = (Fi - Fstar)*(F1 - F0)*Z + (F0 - Fi)*(F1 - Fstar)
                    x = np.clip((_safe_div(num, den))**(1/alpha), 0.0, 1.0)
                    kn, km = 0.0, -1.0
                    k0[idx] = kn + (km - kn)*x

                # E3: Tm138 < tof < Tm1 -> Eq. (49)
                mE3 = (tof_sel < Tm138) & (tof_sel > Tm1)
                if np.any(mE3):
                    idx = idx_use[mE3]
                    c1, c2, c3, c4, alpha = 540649/3125, 256, 1, 1, 16.0
                    Fn, Fi, Fstar = _safe_div(1.0, Tm1[mE3]), _safe_div(1.0, Tm138[mE3]), _safe_div(1.0, tof[idx])
                    g1 = Fi*(Fstar - Fn); g2 = Fstar*(Fn - Fi); g3 = Fn*(Fstar - Fi)
                    k0[idx] = - c4 * np.power(_safe_div((g1*c1 - c3*g3)*c2 + c3*c1*g2, (g3*c1 - c3*g1 - g2*c2)), 1/alpha)

                # E4: Tm141 < tof < Tm138 -> Eq. (49)
                mE4 = (tof_sel < Tm141) & (tof_sel > Tm138)
                if np.any(mE4):
                    idx = idx_use[mE4]
                    c1, c2, c3, c4, alpha = 49267/27059, 67286/17897, 2813/287443, 4439/3156, 243.0
                    Fn, Fi, Fstar = _safe_div(1.0, Tm138[mE4]), _safe_div(1.0, Tm141[mE4]), _safe_div(1.0, tof[idx])
                    g1 = Fi*(Fstar - Fn); g2 = Fstar*(Fn - Fi); g3 = Fn*(Fstar - Fi)
                    k0[idx] = - c4 * np.power(_safe_div((g1*c1 - c3*g3)*c2 + c3*c1*g2, (g3*c1 - c3*g1 - g2*c2)), 1/alpha)

        seed_mask = np.isfinite(k0)
        if not np.any(seed_mask):
            return _empty_result()

        seed_row_idx = np.flatnonzero(seed_mask)
        k_seed = k0[seed_row_idx]
        k_lower_seed = None
        k_upper_seed = None

    else:
        # --- Elliptic multi-rev initialization (Arora-Russell Eqs. 53-54 and Tables 5-7) ---
        idx_all = np.where(elliptic)[0]
        if idx_all.size == 0:
            return _empty_result()

        tau_sel = tau[idx_all]
        S_sel = S[idx_all]
        tof_sel = tof[idx_all]

        # Eq. 55 constants for i=1..20, with fallback Eb,i0=pi for i>20.
        Eb_i0_vec = np.array([
            2.848574, 2.969742, 3.019580, 3.046927, 3.064234,
            3.076182, 3.084929, 3.091610, 3.096880, 3.101145,
            3.104666, 3.107623, 3.110142, 3.112312, 3.114203,
            3.115864, 3.117335, 3.118646, 3.119824, 3.120886
        ], dtype=float)
        Eb0 = Eb_i0_vec[N - 1] if N <= Eb_i0_vec.size else np.pi
        v2 = np.full(idx_all.size, Eb0, dtype=float)
        abs_tau = np.abs(tau_sel)
        den_v1 = np.maximum(v2*(sqrt2 - 2.0*abs_tau), 1e-12)
        v1 = 8.0*abs_tau / den_v1

        # Use +1 at tau=0 to keep continuity with the paper's approximation.
        sgn_tau = np.where(tau_sel >= 0.0, 1.0, -1.0)
        Eb_tilde = v2*(1.0 - sgn_tau) + v2*sgn_tau*np.power(1.0/(1.0 + v1), 0.25)
        kb = np.sign(np.pi - Eb_tilde) * np.sqrt(np.maximum(np.cos(Eb_tilde) + 1.0, 0.0))
        kb = np.clip(kb, -sqrt2 + 1e-10, sqrt2 - 1e-10)
        Tb = _tof_from_k(kb, tau_sel, S_sel, N, series_eps, k_snap_eps)

        # Paper pages 14-15: avoid rejecting cases near the approximate T_b.
        # If T* is close to and below T~_b, refine k_b by minimizing TOF.
        near_tb = (tof_sel < Tb) & (tof_sel >= 0.8 * Tb)
        if np.any(near_tb):
            kb_refined, Tb_refined, kb_ok = _minimize_tof_rows_jit(
                kb[near_tb].astype(np.float64, copy=True),
                tau_sel[near_tb].astype(np.float64, copy=False),
                S_sel[near_tb].astype(np.float64, copy=False),
                int(N),
                float(series_eps),
                float(tol),
            )
            if np.any(kb_ok):
                near_rows = np.flatnonzero(near_tb)
                update_rows = near_rows[np.asarray(kb_ok, dtype=bool)]
                kb[update_rows] = kb_refined[np.asarray(kb_ok, dtype=bool)]
                Tb[update_rows] = Tb_refined[np.asarray(kb_ok, dtype=bool)]

        feasible_multi = tof_sel >= Tb * (1 + 1e-6)
        if not np.any(feasible_multi):
            return _empty_result()

        idx_all = idx_all[feasible_multi]
        tau_sel = tau_sel[feasible_multi]
        S_sel = S_sel[feasible_multi]
        tof_sel = tof_sel[feasible_multi]
        kb = kb[feasible_multi]
        Tb = Tb[feasible_multi]

        k0_sel, k_lower_sel, k_upper_sel = _compute_multirev_k0_jit(
            tau_sel.astype(np.float64, copy=False),
            S_sel.astype(np.float64, copy=False),
            tof_sel.astype(np.float64, copy=False),
            kb.astype(np.float64, copy=False),
            Tb.astype(np.float64, copy=False),
            bsgn_vec[idx_all].astype(np.int64, copy=False),
            int(N),
            float(series_eps),
            float(k_snap_eps),
        )

        seed_mask = np.isfinite(k0_sel)
        if not np.any(seed_mask):
            return _empty_result()

        seed_row_idx = idx_all[seed_mask]
        k_seed = k0_sel[seed_mask]
        k_lower_seed = k_lower_sel[seed_mask]
        k_upper_seed = k_upper_sel[seed_mask]

    r1 = r1[seed_row_idx]
    r2 = r2[seed_row_idx]
    tof = tof[seed_row_idx]
    r1n = r1n[seed_row_idx]
    r2n = r2n[seed_row_idx]
    tau = tau[seed_row_idx]
    S = S[seed_row_idx]

    # --- Halley iterations (JIT row-wise) ---
    k, converged = _halley_solve_rows_jit(
        k_seed.astype(np.float64, copy=True),
        tau.astype(np.float64, copy=False),
        S.astype(np.float64, copy=False),
        tof.astype(np.float64, copy=False),
        int(N),
        float(series_eps),
        float(tol),
    )

    if (N > 0) and np.any(~converged):
        fail = ~converged
        k_fb, ok_fb = _bracketed_root_solve_rows_jit(
            k_seed[fail].astype(np.float64, copy=True),
            k_lower_seed[fail].astype(np.float64, copy=False),
            k_upper_seed[fail].astype(np.float64, copy=False),
            tau[fail].astype(np.float64, copy=False),
            S[fail].astype(np.float64, copy=False),
            tof[fail].astype(np.float64, copy=False),
            int(N),
            float(series_eps),
            float(tol),
        )
        if np.any(ok_fb):
            fail_rows = np.flatnonzero(fail)
            update_rows = fail_rows[np.asarray(ok_fb, dtype=bool)]
            k[update_rows] = k_fb[np.asarray(ok_fb, dtype=bool)]
            converged[update_rows] = True

    if not np.any(converged):
        return _empty_result()

    keep_idx = np.flatnonzero(converged)
    root_row_idx = seed_row_idx[keep_idx]
    k = k[keep_idx]
    r1 = r1[keep_idx]
    r2 = r2[keep_idx]
    r1n = r1n[keep_idx]
    r2n = r2n[keep_idx]
    tau = tau[keep_idx]
    S = S[keep_idx]

    # --- f, g, gdot and velocities ---
    sqrt_arg = np.maximum(1 - k*tau, 0)
    # typo in Eq. 32-34 of the paper?
    f = 1.0 - (r1n + r2n) * (1 - k*tau) / r1n
    gdot = 1.0 - (r1n + r2n) * (1 - k*tau) / r2n
    g = S*tau*np.sqrt(sqrt_arg)

    fg_ok = np.isfinite(f) & np.isfinite(gdot) & np.isfinite(g) & (np.abs(g) > 1e-15)
    if not np.any(fg_ok):
        if return_diagnostics:
            empty_v = np.empty((0, 3), dtype=float)
            empty_i = np.empty((0,), dtype=int)
            return empty_v, empty_v.copy(), empty_i, root_row_idx, empty_i
        return _empty_result()

    r1 = r1[fg_ok]
    r2 = r2[fg_ok]
    f = f[fg_ok]
    g = g[fg_ok]
    gdot = gdot[fg_ok]
    root_row_idx_fg = root_row_idx[fg_ok]

    v1 = (r2 - f[:,None]*r1) / g[:,None]
    v2 = (gdot[:,None]*r2 - r1) / g[:,None]

    finite_v = np.all(np.isfinite(v1), axis=1) & np.all(np.isfinite(v2), axis=1)
    if not np.any(finite_v):
        if return_diagnostics:
            empty_v = np.empty((0, 3), dtype=float)
            empty_i = np.empty((0,), dtype=int)
            return empty_v, empty_v.copy(), empty_i, root_row_idx, empty_i
        return _empty_result()

    recon_row_idx = root_row_idx_fg[finite_v]
    if return_diagnostics:
        return v1[finite_v], v2[finite_v], recon_row_idx, root_row_idx, recon_row_idx
    return v1[finite_v], v2[finite_v], recon_row_idx


def lambert_batch(
    r1,
    r2,
    tof,
    mu=1.0,
    N=0,
    hz=1,
    sweep_rev=False,
    hz_tol=0.0,
    series_eps=2e-2,
    tol=1e-6,
    diagnostics=False,
):
    """
    Batched Lambert solver with vectorized solution expansion.

    Args:
        r1, r2: shape (B,3)
        tof: shape (B,)
        hz:
            Strict selector on returned z-angular-momentum sign:
            +1 -> keep only rows with cross(r1, v1)_z > hz_tol
            -1 -> keep only rows with cross(r1, v1)_z < -hz_tol
             0 -> keep both signs
            Can be scalar or shape (B,).
        N: number of revolutions (scalar integer >= 0)
        sweep_rev:
            False -> solve only for the requested revolution count `N`
            True  -> solve for all counts from 0..N (inclusive)
        hz_tol:
            Non-negative strictness tolerance for z-angular-momentum filtering.
            Rows with |cross(r1, v1)_z| <= hz_tol are treated as near-degenerate and
            are discarded when `hz` is +/-1.
        series_eps:
            Width of the near-parabolic switching region used by `_W_and_derivs*`.
        tol:
            Halley/Newton convergence tolerance used for stopping criteria.
        diagnostics:
            When True, append a sixth return value with per-stage diagnostics.

    Returns:
        v1: shape (M,3)
        v2: shape (M,3)
        a: shape (M,) semi-major axis estimate from specific orbital energy
        nrev_signed: shape (M,) with
            0 for zero-rev branch,
            +k for long-period branch at revolution count `k`,
            -k for short-period branch at revolution count `k`
            (`k` ranges over {N} or {0..N} depending on `sweep_rev`)
        index: shape (M,) 0-based row index into the original inputs
        diagnostics: optional dict with arrays for root-converged,
            velocity-reconstructed, and final returned rows.
    """
    r1 = np.asarray(r1, float)
    r2 = np.asarray(r2, float)
    tof = np.asarray(tof, float)

    N = int(N)
    if N < 0:
        raise ValueError("N must be >= 0.")
    sweep_rev = bool(sweep_rev)
    B = r1.shape[0]
    if r1.shape != r2.shape or r1.ndim != 2 or r1.shape[1] != 3:
        raise ValueError("r1 and r2 must have shape (B,3).")
    if tof.shape != (B,):
        raise ValueError("tof must have shape (B,).")

    cross12 = np.cross(r1, r2)

    hz_arr = np.asarray(hz)
    if hz_arr.ndim == 0:
        hz_vec = np.full(B, int(hz_arr), dtype=int)
    else:
        if hz_arr.shape != (B,):
            raise ValueError("hz must be scalar or shape (B,).")
        hz_vec = hz_arr.astype(int)
    if np.any((hz_vec != -1) & (hz_vec != 0) & (hz_vec != 1)):
        raise ValueError("hz entries must be one of {-1, 0, 1}.")
    hz_tol = float(hz_tol)
    if hz_tol < 0.0:
        raise ValueError("hz_tol must be >= 0.")

    # Route each source row to internal direction(s) before solve.
    # For Lambert geometry, h = r x v is parallel/anti-parallel to r1 x r2.
    # We use sign(cross(r1,r2)_z) to pre-select the internal direction that is
    # likely to satisfy requested strict output sign, and keep a post-filter as
    # the authoritative check.
    c12_z = cross12[:, 2]
    pos_geo = c12_z > hz_tol
    neg_geo = c12_z < -hz_tol
    amb_geo = ~(pos_geo | neg_geo)

    req_pos = hz_vec == 1
    req_neg = hz_vec == -1
    req_both = hz_vec == 0

    idx_plus = np.where(
        req_both
        | (req_pos & pos_geo)
        | (req_neg & neg_geo)
        | ((req_pos | req_neg) & amb_geo)
    )[0]
    idx_minus = np.where(
        req_both
        | (req_pos & neg_geo)
        | (req_neg & pos_geo)
        | ((req_pos | req_neg) & amb_geo)
    )[0]

    idx_blocks = []
    hz_blocks = []
    nrev_blocks = []
    nabs_blocks = []

    def _append_rev_block(rev):
        rev = int(rev)
        idx_parts = []
        hz_parts = []
        nrev_parts = []

        if rev == 0:
            if idx_plus.size:
                idx_parts.append(idx_plus)
                hz_parts.append(np.full(idx_plus.size, 1, dtype=int))
                nrev_parts.append(np.zeros(idx_plus.size, dtype=int))
            if idx_minus.size:
                idx_parts.append(idx_minus)
                hz_parts.append(np.full(idx_minus.size, -1, dtype=int))
                nrev_parts.append(np.zeros(idx_minus.size, dtype=int))
        else:
            if idx_plus.size:
                idx_parts.append(idx_plus)
                hz_parts.append(np.full(idx_plus.size, 1, dtype=int))
                nrev_parts.append(np.full(idx_plus.size, rev, dtype=int))

                idx_parts.append(idx_plus)
                hz_parts.append(np.full(idx_plus.size, 1, dtype=int))
                nrev_parts.append(np.full(idx_plus.size, -rev, dtype=int))

            if idx_minus.size:
                idx_parts.append(idx_minus)
                hz_parts.append(np.full(idx_minus.size, -1, dtype=int))
                nrev_parts.append(np.full(idx_minus.size, rev, dtype=int))

                idx_parts.append(idx_minus)
                hz_parts.append(np.full(idx_minus.size, -1, dtype=int))
                nrev_parts.append(np.full(idx_minus.size, -rev, dtype=int))

        if not idx_parts:
            return

        idx_blocks.append(np.concatenate(idx_parts, axis=0))
        hz_blocks.append(np.concatenate(hz_parts, axis=0))
        nrev_blocks.append(np.concatenate(nrev_parts, axis=0))
        nabs_blocks.append(rev)

    rev_values = range(0, N + 1) if sweep_rev else (N,)
    for rev in rev_values:
        _append_rev_block(rev)

    if not idx_blocks:
        empty_v = np.empty((0, 3), dtype=float)
        empty_s = np.empty((0,), dtype=float)
        empty_i = np.empty((0,), dtype=int)
        if diagnostics:
            diag = dict(
                root_index=empty_i.copy(),
                root_nrev_signed=empty_i.copy(),
                root_hz_sign=empty_i.copy(),
                reconstructed_index=empty_i.copy(),
                reconstructed_nrev_signed=empty_i.copy(),
                reconstructed_hz_sign=empty_i.copy(),
                returned_index=empty_i.copy(),
                returned_nrev_signed=empty_i.copy(),
                returned_hz_sign=empty_i.copy(),
            )
            return empty_v, empty_v.copy(), empty_s, empty_i, empty_i, diag
        return empty_v, empty_v.copy(), empty_s, empty_i, empty_i

    use_shared_geometry = len(idx_blocks) > 2
    if use_shared_geometry:
        dot12 = np.einsum("bi,bi->b", r1, r2)
        r1n_full = np.linalg.norm(r1, axis=1)
        r2n_full = np.linalg.norm(r2, axis=1)
        theta_full = np.arctan2(np.linalg.norm(cross12, axis=1), dot12)
        tau_num = (r1n_full * r2n_full) * (1.0 + np.cos(theta_full))
        tau_abs_full = np.sqrt(np.maximum(tau_num, 0.0)) / (r1n_full + r2n_full)
        S_full = np.sqrt((r1n_full + r2n_full) ** 3 / float(mu))
    else:
        r1n_full = None
        r2n_full = None
        tau_abs_full = None
        S_full = None

    v1_kept = []
    v2_kept = []
    nrev_kept = []
    index_kept = []
    root_nrev_kept = []
    root_hz_kept = []
    root_index_kept = []
    recon_nrev_kept = []
    recon_hz_kept = []
    recon_index_kept = []

    def _solve_with_split(idx_loc, hz_loc, nrev_loc, nabs_loc):
        if idx_loc.size == 0:
            return

        try:
            if diagnostics:
                v1_loc, v2_loc, keep_local, root_local, recon_local = _lambert_batch_core(
                    r1[idx_loc],
                    r2[idx_loc],
                    tof[idx_loc],
                    mu=mu,
                    N=int(nabs_loc),
                    hz=hz_loc,
                    branch_sign=np.where(nrev_loc >= 0, 1, -1),
                    series_eps=series_eps,
                    tol=tol,
                    return_diagnostics=True,
                    r1n_pre=None if r1n_full is None else r1n_full[idx_loc],
                    r2n_pre=None if r2n_full is None else r2n_full[idx_loc],
                    tau_abs_pre=None if tau_abs_full is None else tau_abs_full[idx_loc],
                    S_pre=None if S_full is None else S_full[idx_loc],
                )
                root_local = np.asarray(root_local, dtype=np.int64).reshape(-1)
                recon_local = np.asarray(recon_local, dtype=np.int64).reshape(-1)
                if root_local.size:
                    root_nrev_kept.append(nrev_loc[root_local])
                    root_hz_kept.append(hz_loc[root_local])
                    root_index_kept.append(idx_loc[root_local])
                if recon_local.size:
                    recon_nrev_kept.append(nrev_loc[recon_local])
                    recon_hz_kept.append(hz_loc[recon_local])
                    recon_index_kept.append(idx_loc[recon_local])
            else:
                v1_loc, v2_loc, keep_local = _lambert_batch_core(
                    r1[idx_loc],
                    r2[idx_loc],
                    tof[idx_loc],
                    mu=mu,
                    N=int(nabs_loc),
                    hz=hz_loc,
                    branch_sign=np.where(nrev_loc >= 0, 1, -1),
                    series_eps=series_eps,
                    tol=tol,
                    r1n_pre=None if r1n_full is None else r1n_full[idx_loc],
                    r2n_pre=None if r2n_full is None else r2n_full[idx_loc],
                    tau_abs_pre=None if tau_abs_full is None else tau_abs_full[idx_loc],
                    S_pre=None if S_full is None else S_full[idx_loc],
                )
            if keep_local.size:
                keep_local = np.asarray(keep_local, dtype=np.int64).reshape(-1)
                v1_kept.append(v1_loc)
                v2_kept.append(v2_loc)
                nrev_kept.append(nrev_loc[keep_local])
                index_kept.append(idx_loc[keep_local])
        except Exception:
            if idx_loc.size == 1:
                return
            mid = idx_loc.size // 2
            _solve_with_split(idx_loc[:mid], hz_loc[:mid], nrev_loc[:mid], nabs_loc)
            _solve_with_split(idx_loc[mid:], hz_loc[mid:], nrev_loc[mid:], nabs_loc)

    for idx_blk, hz_blk, nrev_blk, nabs_blk in zip(idx_blocks, hz_blocks, nrev_blocks, nabs_blocks):
        _solve_with_split(idx_blk, hz_blk, nrev_blk, nabs_blk)

    if not v1_kept:
        empty_v = np.empty((0, 3), dtype=float)
        empty_s = np.empty((0,), dtype=float)
        empty_i = np.empty((0,), dtype=int)
        if diagnostics:
            root_index = np.concatenate(root_index_kept, axis=0).astype(int, copy=False) if root_index_kept else empty_i.copy()
            root_nrev = np.concatenate(root_nrev_kept, axis=0).astype(int, copy=False) if root_nrev_kept else empty_i.copy()
            root_hz = np.concatenate(root_hz_kept, axis=0).astype(int, copy=False) if root_hz_kept else empty_i.copy()
            recon_index = np.concatenate(recon_index_kept, axis=0).astype(int, copy=False) if recon_index_kept else empty_i.copy()
            recon_nrev = np.concatenate(recon_nrev_kept, axis=0).astype(int, copy=False) if recon_nrev_kept else empty_i.copy()
            recon_hz = np.concatenate(recon_hz_kept, axis=0).astype(int, copy=False) if recon_hz_kept else empty_i.copy()
            diag = dict(
                root_index=root_index,
                root_nrev_signed=root_nrev,
                root_hz_sign=root_hz,
                reconstructed_index=recon_index,
                reconstructed_nrev_signed=recon_nrev,
                reconstructed_hz_sign=recon_hz,
                returned_index=empty_i.copy(),
                returned_nrev_signed=empty_i.copy(),
                returned_hz_sign=empty_i.copy(),
            )
            return empty_v, empty_v.copy(), empty_s, empty_i, empty_i, diag
        return empty_v, empty_v.copy(), empty_s, empty_i, empty_i

    v1 = np.concatenate(v1_kept, axis=0)
    v2 = np.concatenate(v2_kept, axis=0)
    nrev_signed = np.concatenate(nrev_kept, axis=0).astype(int, copy=False)
    index = np.concatenate(index_kept, axis=0).astype(int, copy=False)

    # Authoritative strict filter by requested z-angular-momentum sign.
    hz_out = np.cross(r1[index], v1)[:, 2]
    req_out = hz_vec[index]
    keep = req_out == 0
    keep |= (req_out == 1) & (hz_out > hz_tol)
    keep |= (req_out == -1) & (hz_out < -hz_tol)

    if not np.any(keep):
        empty_v = np.empty((0, 3), dtype=float)
        empty_s = np.empty((0,), dtype=float)
        empty_i = np.empty((0,), dtype=int)
        if diagnostics:
            root_index = np.concatenate(root_index_kept, axis=0).astype(int, copy=False) if root_index_kept else empty_i.copy()
            root_nrev = np.concatenate(root_nrev_kept, axis=0).astype(int, copy=False) if root_nrev_kept else empty_i.copy()
            root_hz = np.concatenate(root_hz_kept, axis=0).astype(int, copy=False) if root_hz_kept else empty_i.copy()
            recon_index = np.concatenate(recon_index_kept, axis=0).astype(int, copy=False) if recon_index_kept else empty_i.copy()
            recon_nrev = np.concatenate(recon_nrev_kept, axis=0).astype(int, copy=False) if recon_nrev_kept else empty_i.copy()
            recon_hz = np.concatenate(recon_hz_kept, axis=0).astype(int, copy=False) if recon_hz_kept else empty_i.copy()
            diag = dict(
                root_index=root_index,
                root_nrev_signed=root_nrev,
                root_hz_sign=root_hz,
                reconstructed_index=recon_index,
                reconstructed_nrev_signed=recon_nrev,
                reconstructed_hz_sign=recon_hz,
                returned_index=empty_i.copy(),
                returned_nrev_signed=empty_i.copy(),
                returned_hz_sign=empty_i.copy(),
            )
            return empty_v, empty_v.copy(), empty_s, empty_i, empty_i, diag
        return empty_v, empty_v.copy(), empty_s, empty_i, empty_i

    v1 = v1[keep]
    v2 = v2[keep]
    nrev_signed = nrev_signed[keep]
    index = index[keep]

    r1n = np.linalg.norm(r1[index], axis=1)
    eps_orb = 0.5 * np.einsum("bi,bi->b", v1, v1) - float(mu) / r1n
    a = np.full_like(eps_orb, np.inf)
    mask = np.abs(eps_orb) > 1e-14
    a[mask] = -float(mu) / (2.0 * eps_orb[mask])

    if diagnostics:
        empty_i = np.empty((0,), dtype=int)
        root_index = np.concatenate(root_index_kept, axis=0).astype(int, copy=False) if root_index_kept else empty_i.copy()
        root_nrev = np.concatenate(root_nrev_kept, axis=0).astype(int, copy=False) if root_nrev_kept else empty_i.copy()
        root_hz = np.concatenate(root_hz_kept, axis=0).astype(int, copy=False) if root_hz_kept else empty_i.copy()
        recon_index = np.concatenate(recon_index_kept, axis=0).astype(int, copy=False) if recon_index_kept else empty_i.copy()
        recon_nrev = np.concatenate(recon_nrev_kept, axis=0).astype(int, copy=False) if recon_nrev_kept else empty_i.copy()
        recon_hz = np.concatenate(recon_hz_kept, axis=0).astype(int, copy=False) if recon_hz_kept else empty_i.copy()
        returned_hz = np.zeros(index.shape[0], dtype=int)
        returned_hz[hz_out[keep] > hz_tol] = 1
        returned_hz[hz_out[keep] < -hz_tol] = -1
        diag = dict(
            root_index=root_index,
            root_nrev_signed=root_nrev,
            root_hz_sign=root_hz,
            reconstructed_index=recon_index,
            reconstructed_nrev_signed=recon_nrev,
            reconstructed_hz_sign=recon_hz,
            returned_index=index.copy(),
            returned_nrev_signed=nrev_signed.copy(),
            returned_hz_sign=returned_hz,
        )
        return v1, v2, a, nrev_signed, index, diag

    return v1, v2, a, nrev_signed, index


#################################
# Validation via F–G propagation
#################################

def _stumpff_C(z, Cmin=1e-5, Cmax=1e8):
    """
    Vectorized, numerically-stable Stumpff C(z).
    Handles z ~ 0 via series, z>0 (elliptic), z<0 (hyperbolic).
    For very large hyperbolic arguments where finite evaluation is not
    possible, returns +inf so the outer Newton solve can shrink chi.
    """
    z = np.asarray(z, dtype=float)
    C = np.full_like(z, np.nan)
    finite = np.isfinite(z)

    small = finite & (np.abs(z) < Cmin)
    zs = z[small]
    # series: C = 1/2! - z/4! + z^2/6! - z^3/8! + z^4/10! + ...
    C[small] = (
        0.5
        - zs / 24.0
        + zs * zs / 720.0
        - zs * zs * zs / 40320.0
        + zs * zs * zs * zs / 3628800.0
    )

    # elliptic (z>0)
    pos = finite & (z > 0) & (~small)
    zp = z[pos]
    sp = np.sqrt(zp)
    # Use 1 - cos(s) = 2*sin^2(s/2) to reduce cancellation.
    C[pos] = (2.0 * np.sin(0.5 * sp) ** 2) / zp

    # hyperbolic (z<0)
    neg = finite & (z < 0) & (~small)
    zn = z[neg]
    sn = np.sqrt(-zn)  # real, >= 0
    Cneg = np.empty_like(sn)
    safe = sn <= 700.0
    if np.any(safe):
        s = sn[safe]
        # C = (cosh(s)-1)/s^2 = 2*sinh^2(s/2)/s^2
        sh = np.sinh(0.5 * s)
        Cneg[safe] = (2.0 * sh * sh) / (s * s)
    if np.any(~safe):
        Cneg[~safe] = np.inf
    C[neg] = Cneg

    return C


def _stumpff_S(z, Smin=1e-5):
    """
    Vectorized, numerically-stable Stumpff S(z).
    For very large hyperbolic arguments where finite evaluation is not
    possible, returns +inf so the outer Newton solve can shrink chi.
    """
    z = np.asarray(z, dtype=float)
    S = np.full_like(z, np.nan)
    finite = np.isfinite(z)

    small = finite & (np.abs(z) < Smin)
    zs = z[small]
    # series: S = 1/3! - z/5! + z^2/7! - z^3/9! + z^4/11! + ...
    S[small] = (
        (1.0 / 6.0)
        - zs / 120.0
        + zs * zs / 5040.0
        - zs * zs * zs / 362880.0
        + zs * zs * zs * zs / 39916800.0
    )

    pos = finite & (z > 0) & (~small)
    zp = z[pos]
    sp = np.sqrt(zp)
    S[pos] = (sp - np.sin(sp)) / (sp * sp * sp)

    neg = finite & (z < 0) & (~small)
    zn = z[neg]
    sn = np.sqrt(-zn)
    Sneg = np.empty_like(sn)
    safe = sn <= 700.0
    if np.any(safe):
        s = sn[safe]
        Sneg[safe] = (np.sinh(s) - s) / (s * s * s)
    if np.any(~safe):
        Sneg[~safe] = np.inf
    S[neg] = Sneg
    
    return S


def kepler_propagate_universal(
    r0_km: Sequence[float],
    v0_km_s: Sequence[float],
    dt_s: float,
    mu_km3_s2: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Propagate one two-body state via universal variables (km, km/s, s).

    The scalar Newton solve is performed in nondimensional units with:
    L = ||r0||, T = L*sqrt(L/mu), so ||r0|| = 1 and mu = 1. This improves
    conditioning for very large state magnitudes and long propagation times.
    """

    r0 = np.asarray(r0_km, dtype=float).reshape(3)
    v0 = np.asarray(v0_km_s, dtype=float).reshape(3)
    dt = float(dt_s)
    mu = float(mu_km3_s2)

    if mu <= 0.0:
        raise ValueError("mu_km3_s2 must be positive.")
    if dt == 0.0:
        return r0.copy(), v0.copy()

    r0_norm = float(np.linalg.norm(r0))
    if r0_norm <= 0.0 or not np.isfinite(r0_norm):
        raise ValueError("Initial position norm must be positive.")

    # Nondimensionalize with L = ||r0|| and T = L*sqrt(L/mu), giving mu_nd = 1.
    r_scale = r0_norm
    t_scale = r_scale * float(np.sqrt(r_scale / mu))
    if t_scale <= 0.0 or not np.isfinite(t_scale):
        raise RuntimeError("Invalid nondimensional time scale.")
    v_scale = r_scale / t_scale

    r0_nd = r0 / r_scale
    v0_nd = v0 / v_scale
    dt_nd = dt / t_scale

    r0_norm_nd = float(np.linalg.norm(r0_nd))
    if r0_norm_nd <= 0.0 or not np.isfinite(r0_norm_nd):
        raise RuntimeError("Invalid nondimensional position norm.")

    v0_sq = float(np.dot(v0_nd, v0_nd))
    vr0 = float(np.dot(r0_nd, v0_nd) / r0_norm_nd)
    alpha = 2.0 / r0_norm_nd - v0_sq
    sqrt_mu = 1.0

    # Initial guess for universal anomaly chi.
    if alpha > 1e-8:
        chi = sqrt_mu * dt_nd * alpha
    elif alpha < -1e-8:
        denom = vr0 + np.sign(dt_nd) * np.sqrt(-1.0 / alpha) * (1.0 - r0_norm_nd * alpha)
        if abs(denom) > 1e-15:
            arg = (-2.0 * alpha * dt_nd) / denom
            if arg > 1e-12 and np.isfinite(arg):
                chi = np.sign(dt_nd) * np.sqrt(-1.0 / alpha) * np.log(arg)
            else:
                chi = sqrt_mu * abs(alpha) * dt_nd
        else:
            chi = sqrt_mu * abs(alpha) * dt_nd
    else:
        chi = sqrt_mu * dt_nd / r0_norm_nd

    tol = 1e-12
    max_iter = 300
    converged = False

    for _ in range(max_iter):
        z = alpha * chi * chi
        C = float(_stumpff_C(z))
        S = float(_stumpff_S(z))
        if not np.isfinite(C) or not np.isfinite(S):
            chi *= 0.5
            continue

        F = (
            (r0_norm_nd * vr0 / sqrt_mu) * (chi * chi) * C
            + (1.0 - alpha * r0_norm_nd) * (chi**3) * S
            + r0_norm_nd * chi
            - sqrt_mu * dt_nd
        )
        dF = (
            (r0_norm_nd * vr0 / sqrt_mu) * chi * (1.0 - z * S)
            + (1.0 - alpha * r0_norm_nd) * (chi * chi) * C
            + r0_norm_nd
        )
        if not np.isfinite(F) or not np.isfinite(dF):
            chi *= 0.5
            continue

        if abs(dF) < 1e-15:
            if abs(F) < tol * max(1.0, abs(dt_nd)):
                converged = True
                break
            chi *= 0.5
            continue

        delta = F / dF
        if not np.isfinite(delta):
            chi *= 0.5
            continue

        if abs(delta) > 1e7:
            delta = np.sign(delta) * 1e7
        chi -= delta

        if abs(delta) < tol and abs(F) < tol * max(1.0, abs(dt_nd)):
            converged = True
            break

    if not converged:
        raise RuntimeError("Universal-variable propagation did not converge (nondimensional solve).")

    z = alpha * chi * chi
    C = float(_stumpff_C(z))
    S = float(_stumpff_S(z))

    f = 1.0 - (chi * chi / r0_norm_nd) * C
    g = dt_nd - (chi**3 / sqrt_mu) * S

    r_nd = f * r0_nd + g * v0_nd
    r_norm_nd = float(np.linalg.norm(r_nd))
    if r_norm_nd <= 0.0 or not np.isfinite(r_norm_nd):
        raise RuntimeError("Propagation produced zero position norm.")

    fdot = (sqrt_mu / (r_norm_nd * r0_norm_nd)) * (alpha * (chi**3) * S - chi)
    gdot = 1.0 - (chi * chi / r_norm_nd) * C
    v_nd = fdot * r0_nd + gdot * v0_nd
    if not np.all(np.isfinite(r_nd)) or not np.all(np.isfinite(v_nd)):
        raise RuntimeError("Propagation produced non-finite state.")

    r_km = r_nd * r_scale
    v_km_s = v_nd * v_scale
    return r_km, v_km_s


def validate_lambert_batch(r1, v1, r2, v2, tof, mu=1.0, tol_r=1e-8, tol_v=1e-8, max_iter=100, fac=0.2, verbose=False):
    """
    Validate Lambert solutions via F–G propagation.
    Inputs:
        r1,v1,r2,v2 : (B,3)
        tof : (B,)
        mu : (scalar)
        verbose : print Newton residuals each iteration
    Outputs:
        bool mask ok[B], plus optionally rms errors.
    """
    r1 = np.asarray(r1, dtype=float)
    v1 = np.asarray(v1, dtype=float)
    r2 = np.asarray(r2, dtype=float)
    v2 = np.asarray(v2, dtype=float)
    tof = np.asarray(tof, dtype=float)
    B = r1.shape[0]
    sqrt_mu = np.sqrt(mu)

    r1n = np.linalg.norm(r1, axis=1)
    r0v0 = np.sum(r1*v1, axis=1)
    alpha = 2.0/r1n - np.sum(v1*v1, axis=1)/mu  # -2 x specific energy

    # Newton iteration for x
    x = np.sign(tof) / np.sqrt(np.maximum(np.abs(alpha), tol_r))   # ⇒ z0 ≈ sign(alpha) ∈ {+1, -1}
    x = sqrt_mu * tof / alpha 
    cnt = 0 
    step = np.full(B, np.inf)
    
    while np.any(np.abs(step) > 1e-10):
        z = alpha * x**2
        C = _stumpff_C(z)
        S = _stumpff_S(z)
        f  = x**3*S + (r0v0/sqrt_mu)*x**2 * C + r1n*x*(1 - z*S) - sqrt_mu * tof
        df = x**2*C + (r0v0/sqrt_mu)*x*(1 - z*S) + r1n*(1 - z*C)
        df = np.where(np.abs(df) < 1e-15, np.sign(df)*1e-15, df)  # protection against zero derivative
        step = f/df
        x += step
        cnt += 1
        if verbose:
            print("iter ", cnt, ": max |f| = ", np.max(np.abs(f)), "x =", np.max(np.abs(x)))
        if cnt > max_iter:
            raise RuntimeError("F–G propagation: x did not converge.")

    # Propagation
    z = alpha * x**2
    C = _stumpff_C(z)
    S = _stumpff_S(z)

    F = 1 - (x**2/r1n)*C
    G = tof - (1/sqrt_mu)*x**3*S
    # r2_pred
    r2_pred = F[:,None]*r1 + G[:,None]*v1
    r = np.linalg.norm(r2_pred, axis=1)
    dF = (sqrt_mu/(r*r1n))*x*(z*S - 1)
    dG = 1 - (x**2/r)*C
    v2_pred = dF[:,None]*r1 + dG[:,None]*v1

    # Errors
    err_r = np.linalg.norm(r2_pred - r2, axis=1)
    err_v = np.linalg.norm(v2_pred - v2, axis=1)
    ok = (err_r < tol_r) & (err_v < tol_v)

    return ok, err_r, err_v


def _sample_random_multirev_case(rng, mu=1.0, nrev_max=3, max_tries=400):
    for _ in range(int(max_tries)):
        r0_mag = float(rng.uniform(0.8, 1.4))
        rf_mag = float(rng.uniform(0.8, 1.6))

        u0 = rng.normal(size=3)
        u0 /= np.linalg.norm(u0)

        perp = rng.normal(size=3)
        perp -= float(np.dot(perp, u0)) * u0
        perp_norm = float(np.linalg.norm(perp))
        if perp_norm < 1e-10:
            continue
        perp /= perp_norm

        ang = float(rng.uniform(np.deg2rad(35.0), np.deg2rad(145.0)))
        sign = 1.0 if rng.random() < 0.5 else -1.0
        uf = np.cos(ang) * u0 + sign * np.sin(ang) * perp

        r0 = r0_mag * u0
        rf = rf_mag * uf
        tof = float(rng.uniform(30.0, 140.0))

        sols = []
        feasible = True
        try:
            v1_all, v2_all, _, nrev_all, index_all = lambert_batch(
                np.asarray(r0, dtype=float).reshape(1, 3),
                np.asarray(rf, dtype=float).reshape(1, 3),
                np.asarray([float(tof)], dtype=float),
                mu=float(mu),
                N=int(nrev_max),
                hz=1,   # only ccw for
                sweep_rev=True,
                tol=1e-10,
            )
        except Exception:
            feasible = False

        if feasible:
            if not (np.all(np.isfinite(v1_all)) and np.all(np.isfinite(v2_all))):
                feasible = False

        if feasible:
            sel = np.where(index_all == 0)[0]
            expected_total = 1 + 2 * int(nrev_max)
            if sel.size != expected_total:
                feasible = False
            else:
                expected_nrev = [0]
                for k in range(1, int(nrev_max) + 1):
                    expected_nrev.extend([+k, -k])
                got_nrev = sorted(int(nrev_all[j]) for j in sel)
                if got_nrev != sorted(expected_nrev):
                    feasible = False

        if feasible:
            for j in sel:
                v1 = v1_all[j]
                v2 = v2_all[j]
                nrev_signed = int(nrev_all[j])
                hz_sign = 1 if np.cross(r0, v1)[2] >= 0 else -1
                sols.append((abs(int(nrev_signed)), nrev_signed, hz_sign, v1, v2))

        expected_total = 1 + 2 * int(nrev_max)
        if feasible and len(sols) == expected_total:
            return r0, rf, tof, sols

    raise RuntimeError("Could not find a random case feasible for N=0..3 with CCW direction and LP/SP branches.")


def _sample_orbit_positions(r0, v0, tof, mu, n_samples):
    t = np.linspace(0.0, float(tof), int(n_samples))
    r = np.empty((t.size, 3), dtype=float)
    for i, ti in enumerate(t):
        ri, _ = kepler_propagate_universal(r0, v0, float(ti), mu)
        r[i] = ri
    return t, r


def _run_multirev_animation_demo(seed=2, nrev_max=3, n_samples=300, gif_path="output/lambert_multirev_demo.gif"):
    try:
        import matplotlib.pyplot as plt
        from matplotlib.animation import FuncAnimation, PillowWriter
    except Exception as exc:
        raise RuntimeError("Matplotlib with Pillow support is required for animation demo.") from exc

    from pathlib import Path

    rng = np.random.default_rng(int(seed))
    mu = 1.0

    case_data = None
    r0 = rf = tof = None
    for _ in range(80):
        r0_try, rf_try, tof_try, sols_try = _sample_random_multirev_case(
            rng, mu=mu, nrev_max=int(nrev_max)
        )

        h = np.cross(r0_try, rf_try)
        h_norm = float(np.linalg.norm(h))
        if h_norm < 1e-12:
            continue
        e1 = r0_try / np.linalg.norm(r0_try)
        e3 = h / h_norm
        e2 = np.cross(e3, e1)

        def _proj(vecs):
            arr = np.asarray(vecs, dtype=float)
            x = arr @ e1
            y = arr @ e2
            return np.stack((x, y), axis=-1)

        case_data_try = []
        sampling_ok = True
        for N, nrev_signed, hz_sign, v1, v2 in sols_try:
            try:
                t, r = _sample_orbit_positions(r0_try, v1, tof_try, mu, n_samples)
            except Exception:
                sampling_ok = False
                break
            case_data_try.append(
                {
                    "N": int(N),
                    "nrev_signed": int(nrev_signed),
                    "hz": int(hz_sign),
                    "t": t,
                    "r2d": _proj(r),
                }
            )

        if sampling_ok:
            r0, rf, tof = r0_try, rf_try, tof_try
            case_data = case_data_try
            break

    if case_data is None:
        raise RuntimeError("Failed to find a random case with successful multi-rev solve and propagation.")

    t_ref = case_data[0]["t"]
    all_pts = np.vstack([c["r2d"] for c in case_data])
    span = np.max(all_pts, axis=0) - np.min(all_pts, axis=0)
    pad = 0.15 * max(float(span[0]), float(span[1]), 1e-3)

    r0_2d = _proj(r0)
    rf_2d = _proj(rf)
    origin_2d = _proj(np.zeros(3))

    fig, ax = plt.subplots(figsize=(8.5, 8.5))
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(float(np.min(all_pts[:, 0]) - pad), float(np.max(all_pts[:, 0]) + pad))
    ax.set_ylim(float(np.min(all_pts[:, 1]) - pad), float(np.max(all_pts[:, 1]) + pad))
    ax.grid(True, alpha=0.3)
    ax.set_xlabel("Plane x")
    ax.set_ylabel("Plane y")
    ax.set_title("Lambert Multi-Rev Animation (N=0..3, CW/CCW + LP/SP)")

    ax.plot(origin_2d[0], origin_2d[1], "ko", ms=6, label="Central body")
    ax.plot(r0_2d[0], r0_2d[1], marker="*", color="k", ms=12, linestyle="None", label="Departure")
    ax.plot(rf_2d[0], rf_2d[1], marker="x", color="k", ms=9, linestyle="None", label="Arrival")

    colors = plt.cm.tab10(np.linspace(0.0, 1.0, len(case_data)))
    lines = []
    dots = []
    artists = []
    for color, c in zip(colors, case_data):
        direction = "ccw" if c["hz"] > 0 else "cw"
        if c["N"] == 0:
            branch = "single"
            ls = "-"
        else:
            branch = "long" if c["nrev_signed"] > 0 else "short"
            ls = "-" if c["nrev_signed"] > 0 else "--"
        (line,) = ax.plot([], [], ls=ls, lw=2.0, color=color, label=f"N={c['N']} {direction} {branch}")
        (dot,) = ax.plot([], [], marker="o", ms=4.0, color=color, linestyle="None")
        lines.append(line)
        dots.append(dot)
        artists.extend([line, dot])

    time_text = ax.text(0.02, 0.98, "", transform=ax.transAxes, va="top", ha="left")
    artists.append(time_text)
    ax.legend(loc="upper right", fontsize=9, ncol=2)

    def _init():
        for line, dot in zip(lines, dots):
            line.set_data([], [])
            dot.set_data([], [])
        time_text.set_text("")
        return artists

    def _update(i):
        for line, dot, c in zip(lines, dots, case_data):
            p = c["r2d"]
            seg = p[: i + 1]
            line.set_data(seg[:, 0], seg[:, 1])
            dot.set_data([p[i, 0]], [p[i, 1]])
        time_text.set_text(f"t = {t_ref[i]:.3f} / {tof:.3f}")
        return artists

    ani = FuncAnimation(
        fig,
        _update,
        init_func=_init,
        frames=t_ref.size,
        interval=35,
        blit=True,
        repeat=True,
    )

    gif_out = Path(gif_path)
    gif_out.parent.mkdir(parents=True, exist_ok=True)
    ani.save(str(gif_out), writer=PillowWriter(fps=25))

    print("Lambert animation demo case")
    print(f"  seed: {int(seed)}")
    print(f"  mu: {mu}")
    print(f"  r0: {r0}")
    print(f"  rf: {rf}")
    print(f"  tof: {tof}")
    print(f"  gif: {gif_out.resolve()}")

    backend = str(plt.get_backend()).lower()
    if "agg" in backend:
        print("Non-interactive Matplotlib backend detected; skipping plt.show().")
    else:
        plt.show()


if __name__ == "__main__":
    _run_multirev_animation_demo()

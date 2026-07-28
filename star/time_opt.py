from __future__ import annotations

import numpy as np
from scipy.optimize import linprog


def time_opt(
    tmin: np.ndarray,
    tmax: np.ndarray,
    tof_min: np.ndarray,
    tof_max: np.ndarray,
    *,
    tol: float = 1e-12,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Sect. 2.1.1-style time-bounds tightening:
        tmin* = argmin_t  sum_i t[i]
        tmax* = argmax_t  sum_i t[i]
    subject to:
        tmin[i] <= t[i] <= tmax[i]        (if provided; otherwise +/- inf)
        tof_min[i,k] <= t[k] - t[i] <= tof_max[i,k]   (if provided; otherwise +/- inf)

    Unspecified entries must be filled with +/- np.inf.
    """
    tmin = np.asarray(tmin, dtype=float).reshape(-1)
    tmax = np.asarray(tmax, dtype=float).reshape(-1)
    tof_min = np.asarray(tof_min, dtype=float)
    tof_max = np.asarray(tof_max, dtype=float)

    n = tmin.size
    if tmax.size != n:
        raise ValueError("tmin and tmax must have the same length.")
    if tof_min.shape != (n, n) or tof_max.shape != (n, n):
        raise ValueError("tof_min and tof_max must have shape (n,n) matching tmin/tmax.")

    bounds = []
    for i in range(n):
        lo = None if np.isneginf(tmin[i]) else float(tmin[i])
        hi = None if np.isposinf(tmax[i]) else float(tmax[i])
        if lo is not None and hi is not None and lo > hi + tol:
            raise ValueError(f"Infeasible explicit time bound at i={i}: tmin={lo} > tmax={hi}.")
        bounds.append((lo, hi))

    A_rows = []
    b_rows = []

    for i in range(n):
        for k in range(n):
            if i == k:
                continue
            lo = tof_min[i, k]
            hi = tof_max[i, k]

            # t[k] - t[i] >= lo  <=>  -t[k] + t[i] <= -lo
            if np.isfinite(lo):
                row = np.zeros(n)
                row[i] = 1.0
                row[k] = -1.0
                A_rows.append(row)
                b_rows.append(-float(lo))

            # t[k] - t[i] <= hi
            if np.isfinite(hi):
                row = np.zeros(n)
                row[k] = 1.0
                row[i] = -1.0
                A_rows.append(row)
                b_rows.append(float(hi))

    A_ub = np.vstack(A_rows) if A_rows else None
    b_ub = np.array(b_rows, dtype=float) if b_rows else None

    def _solve(c: np.ndarray) -> np.ndarray:
        res = linprog(c=c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method="highs")
        if res.status != 0:
            raise RuntimeError(
                f"LP failed (status={res.status}): {res.message}\n"
                f"Tip: too many +/-inf bounds can make the problem underdetermined/unbounded."
            )
        return np.array(res.x, dtype=float)

    c = np.ones(n, dtype=float)
    tmin_star = _solve(c)      # minimize sum t
    tmax_star = _solve(-c)     # maximize sum t

    return tmin_star, tmax_star


def tof_opt(
    tmin: np.ndarray,
    tmax: np.ndarray,
    tof_min: np.ndarray,
    tof_max: np.ndarray,
    *,
    tol: float = 1e-12,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Sect. 2.1.2 time-of-flight bounds tightening (Eq. 6 and 7).

    For each i in {0..n-1}, solve LP P_i with decision variable tau_i[k] (k=0..n-1):
        maximize    sum_k tau_i[k]                              (Eq. 6)
        subject to  (Eq. 5 re-expressed on tau row differences)
                    tof_min[p,q] <= tau_i[q] - tau_i[p] <= tof_max[p,q]
                    for all (p,q) where bounds are finite (unspecified = +/-inf)

                    tau_i[i] <= 0

                    and (Eq. 7) for any (p,q) for which T bounds are defined (finite):
                    (tmin[q] - tmax[p]) <= tau_i[q] - tau_i[p] <= (tmax[q] - tmin[p])

    Interpretation of the optimal solution tau_i*:
      - tau_i*[i] = 0 (if feasible)
      - for k < i: tof_min[k,i]* = -tau_i*[k]                    (Eq. 8 in paper)
      - for k > i: tof_max[i,k]* =  tau_i*[k]                    (Eq. 9 in paper)

    Returns:
      - tof_min_star (n,n): updated lower bounds (tightened upward where possible)
      - tof_max_star (n,n): updated upper bounds (tightened downward where possible)
      - tau_star     (n,n): tau_star[i,k] = tau_i*[k]

    Unspecified entries must be filled with +/- np.inf.
    """
    tmin = np.asarray(tmin, dtype=float).reshape(-1)
    tmax = np.asarray(tmax, dtype=float).reshape(-1)
    tof_min = np.asarray(tof_min, dtype=float)
    tof_max = np.asarray(tof_max, dtype=float)

    n = tmin.size
    if tmax.size != n:
        raise ValueError("tmin and tmax must have the same length.")
    if tof_min.shape != (n, n) or tof_max.shape != (n, n):
        raise ValueError("tof_min and tof_max must have shape (n,n) matching tmin/tmax.")

    tof_min_star = tof_min.copy()
    tof_max_star = tof_max.copy()
    tau_star = np.full((n, n), np.nan, dtype=float)

    # Precompute all difference constraints that are independent of i.
    # Each constraint is: lo <= tau[q] - tau[p] <= hi
    diff_constraints: list[tuple[int, int, float, float]] = []

    # (A) Constraints from user-specified (or already-updated) tof bounds (Eq. 5 form)
    for p in range(n):
        for q in range(n):
            if p == q:
                continue
            lo = tof_min[p, q]
            hi = tof_max[p, q]
            if np.isfinite(lo) or np.isfinite(hi):
                diff_constraints.append((p, q, lo, hi))

    # (B) Constraints from time bounds (Eq. 7): only when BOTH endpoints have finite T bounds.
    for p in range(n):
        if not (np.isfinite(tmin[p]) and np.isfinite(tmax[p])):
            continue
        for q in range(n):
            if p == q:
                continue
            if not (np.isfinite(tmin[q]) and np.isfinite(tmax[q])):
                continue
            lo = float(tmin[q] - tmax[p])  # (t[q]-t[p])_min = tmin[q] - tmax[p]
            hi = float(tmax[q] - tmin[p])  # (t[q]-t[p])_max = tmax[q] - tmin[p]
            diff_constraints.append((p, q, lo, hi))

    # Build/solve P_i for each i
    for i in range(n):
        bounds = [(None, None)] * n
        bounds[i] = (None, 0.0)  # tau[i,i] <= 0

        A_rows = []
        b_rows = []

        for p, q, lo, hi in diff_constraints:
            # lo <= tau[q] - tau[p] <= hi
            if np.isfinite(lo):
                # tau[q] - tau[p] >= lo  <=>  -tau[q] + tau[p] <= -lo
                row = np.zeros(n)
                row[p] = 1.0
                row[q] = -1.0
                A_rows.append(row)
                b_rows.append(-float(lo))
            if np.isfinite(hi):
                # tau[q] - tau[p] <= hi
                row = np.zeros(n)
                row[q] = 1.0
                row[p] = -1.0
                A_rows.append(row)
                b_rows.append(float(hi))

        A_ub = np.vstack(A_rows) if A_rows else None
        b_ub = np.array(b_rows, dtype=float) if b_rows else None

        # maximize sum_k tau[k]  <=>  minimize -sum_k tau[k]
        c = -np.ones(n, dtype=float)

        res = linprog(c=c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method="highs")
        if res.status != 0:
            raise RuntimeError(
                f"P_{i} failed (status={res.status}): {res.message}\n"
                f"Tip: if many entries are +/-inf, the problem may be underdetermined/unbounded."
            )

        tau_i = np.array(res.x, dtype=float)
        tau_star[i, :] = tau_i

        # Update tof bounds using Eq. 8 and 9 interpretations.
        # For each pair (a,b) with a<b:
        #   tof_min[a,b] comes from row b:  tof_min[a,b] = -tau_star[b,a]
        #   tof_max[a,b] comes from row a:  tof_max[a,b] =  tau_star[a,b]
        for a in range(n):
            for b in range(a + 1, n):
                # lower bound candidate from row b
                cand_min = -tau_star[b, a]
                if np.isfinite(cand_min):
                    tof_min_star[a, b] = max(tof_min_star[a, b], float(cand_min))

                # upper bound candidate from row a
                cand_max = tau_star[a, b]
                if np.isfinite(cand_max):
                    tof_max_star[a, b] = min(tof_max_star[a, b], float(cand_max))

        # Basic feasibility sanity: a tightened lower bound cannot exceed tightened upper bound.
        for a in range(n):
            for b in range(a + 1, n):
                lo = tof_min_star[a, b]
                hi = tof_max_star[a, b]
                if np.isfinite(lo) and np.isfinite(hi) and lo > hi + tol:
                    raise RuntimeError(
                        f"Inconsistent tightened tof bounds for ({a},{b}): "
                        f"tof_min={lo} > tof_max={hi}."
                    )

    return tof_min_star, tof_max_star, tau_star


if __name__ == "__main__":
    # Paper example (n=3 encounters), with +/-inf used for unspecified slots:
    tmin = np.array([1.0, -np.inf, 1.5])
    tmax = np.array([3.0,  np.inf, 3.5])

    tof_min = np.full((3, 3), -np.inf)
    tof_max = np.full((3, 3),  np.inf)

    # tof[1,3] = [1.0, 3.0], tof[2,3] = [0.5, 1.0] (1-indexed in paper)
    tof_min[0, 2], tof_max[0, 2] = 1.0, 3.0
    tof_min[1, 2], tof_max[1, 2] = 0.5, 1.0

    tmin_star, tmax_star = time_opt(tmin, tmax, tof_min, tof_max)
    print("tmin* =", tmin_star)
    print("tmax* =", tmax_star)

    tof_min_star, tof_max_star, tau_star = tof_opt(
        tmin_star, tmax_star, tof_min, tof_max
    )

    np.set_printoptions(precision=3, suppress=True)
    print("\nTau* rows (tau_star[i,:]):")
    for i in range(3):
        print(f"i={i+1}:", tau_star[i, :])

    print("\nUpdated tof (upper triangle shown as [min,max]):")
    for a in range(3):
        for b in range(a + 1, 3):
            print(f"tof[{a+1},{b+1}] = [{tof_min_star[a,b]}, {tof_max_star[a,b]}]")

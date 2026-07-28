"""
Lambert solver stress profiler.

Generates feasible Lambert problems from Keplerian elements and reports solver
stage counts by category and by internal mode:
- k-root convergence,
- f/g velocity reconstruction,
- revolution count,
- branch (single/long/short),
- requested direction sign.

This file is intentionally focused on the Lambert solver internals rather than
post-hoc trajectory validation.

Run with:
    python -m pytest tests/test_lambert.py -v -s
The -s flag is needed to see the summary tables printed by tearDownClass.
"""

import os
import sys
import unittest
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PACKAGE_ROOT = os.path.join(REPO_ROOT, "star")
for _p in (REPO_ROOT, PACKAGE_ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from star.lambert import lambert_batch


# ---------------------------------------------------------------------------
# Orbital mechanics helpers
# ---------------------------------------------------------------------------

def _rotation_matrix(raan: float, inc: float, argp: float) -> np.ndarray:
    """Perifocal → ECI rotation matrix (R_z(Ω) · R_x(i) · R_z(ω))."""
    cr, sr = np.cos(raan), np.sin(raan)
    ci, si = np.cos(inc), np.sin(inc)
    cw, sw = np.cos(argp), np.sin(argp)
    return np.array([
        [ cr*cw - sr*sw*ci,  -cr*sw - sr*cw*ci,  sr*si],
        [ sr*cw + cr*sw*ci,  -sr*sw + cr*cw*ci, -cr*si],
        [ sw*si,              cw*si,              ci   ],
    ])


def _keplerian_to_rv(
    a: float, e: float,
    raan: float, inc: float, argp: float,
    f: float, mu: float = 1.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Position/velocity from Keplerian elements.

    Works for elliptic (0 ≤ e < 1) and hyperbolic (e > 1).
    For hyperbolic orbits pass a < 0 (a = −|a|) per the energy convention.
    """
    if e < 1.0:
        p = a * (1.0 - e * e)
    else:
        p = abs(a) * (e * e - 1.0)

    cos_f, sin_f = np.cos(f), np.sin(f)
    r_mag = p / (1.0 + e * cos_f)
    h = np.sqrt(mu * p)
    r_peri = r_mag * np.array([cos_f, sin_f, 0.0])
    v_peri = (mu / h) * np.array([-sin_f, e + cos_f, 0.0])
    R = _rotation_matrix(raan, inc, argp)
    return R @ r_peri, R @ v_peri


def _mean_anomaly(e: float, f: float) -> float:
    """True anomaly → mean anomaly (elliptic or hyperbolic)."""
    if e < 1.0:
        E = 2.0 * np.arctan2(
            np.sqrt(1.0 - e) * np.sin(f / 2.0),
            np.sqrt(1.0 + e) * np.cos(f / 2.0),
        )
        return E - e * np.sin(E)
    else:
        tanh_half = np.sqrt((e - 1.0) / (e + 1.0)) * np.tan(f / 2.0)
        tanh_half = np.clip(tanh_half, -1.0 + 1e-12, 1.0 - 1e-12)
        H = 2.0 * np.arctanh(tanh_half)
        return e * np.sinh(H) - H


def _tof_elliptic(a: float, e: float, f1: float, f2: float, mu: float = 1.0) -> float:
    """Time of flight from f1 to f2 (prograde direction) on an elliptic orbit."""
    n = np.sqrt(mu / a ** 3)
    dM = _mean_anomaly(e, f2) - _mean_anomaly(e, f1)
    if dM < 0.0:
        dM += 2.0 * np.pi       # wrap to positive (prograde)
    return dM / n


# ---------------------------------------------------------------------------
# Batch case generators
# ---------------------------------------------------------------------------

def _sample_elliptic(
    rng: np.random.Generator,
    n: int,
    e_range: Tuple[float, float],
    theta_range: Tuple[float, float],   # transfer-angle range [rad], both > 0
    a_range: Tuple[float, float] = (0.8, 1.5),
    N_rev: int = 0,
    mu: float = 1.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (r1, r2, tof) arrays for n elliptic Lambert problems."""
    r1_out, r2_out, tof_out = [], [], []
    attempts = 0
    while len(r1_out) < n and attempts < n * 20:
        attempts += 1
        a = float(rng.uniform(*a_range))
        e = float(rng.uniform(*e_range))
        raan = float(rng.uniform(0.0, 2.0 * np.pi))
        inc  = float(rng.uniform(0.0, np.pi))
        argp = float(rng.uniform(0.0, 2.0 * np.pi))
        # f1 avoids the exact periapsis/apoapsis to prevent degeneracy
        f1 = float(rng.uniform(-0.85 * np.pi, 0.85 * np.pi))
        dtheta = float(rng.uniform(*theta_range))
        f2 = f1 + dtheta
        # Wrap f2 into (−π, π] — geometry is unchanged, TOF calculation handles wrap
        f2 = (f2 + np.pi) % (2.0 * np.pi) - np.pi
        try:
            r1, _ = _keplerian_to_rv(a, e, raan, inc, argp, f1, mu)
            r2, _ = _keplerian_to_rv(a, e, raan, inc, argp, f2, mu)
            T = 2.0 * np.pi * np.sqrt(a ** 3 / mu)
            tof = _tof_elliptic(a, e, f1, f2, mu) + N_rev * T
            if tof <= 0.0 or not np.isfinite(tof):
                continue
            if not (np.all(np.isfinite(r1)) and np.all(np.isfinite(r2))):
                continue
            r1_out.append(r1)
            r2_out.append(r2)
            tof_out.append(tof)
        except Exception:
            continue

    if not r1_out:
        return np.zeros((0, 3)), np.zeros((0, 3)), np.zeros(0)
    return np.array(r1_out), np.array(r2_out), np.array(tof_out)


def _sample_hyperbolic(
    rng: np.random.Generator,
    n: int,
    e_range: Tuple[float, float],
    a_abs_range: Tuple[float, float] = (0.5, 2.0),
    mu: float = 1.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (r1, r2, tof) arrays for n hyperbolic Lambert problems (e > 1)."""
    r1_out, r2_out, tof_out = [], [], []
    attempts = 0
    while len(r1_out) < n and attempts < n * 20:
        attempts += 1
        a_abs = float(rng.uniform(*a_abs_range))
        e = float(rng.uniform(*e_range))
        raan = float(rng.uniform(0.0, 2.0 * np.pi))
        inc  = float(rng.uniform(0.0, np.pi))
        argp = float(rng.uniform(0.0, 2.0 * np.pi))
        # Stay well inside the asymptote (f_inf = arccos(−1/e))
        f_inf = np.arccos(-1.0 / e) - 1e-3
        f1 = float(rng.uniform(-f_inf * 0.7, -0.05))   # inbound
        f2 = float(rng.uniform( 0.05,  f_inf * 0.7))   # outbound
        try:
            r1, _ = _keplerian_to_rv(-a_abs, e, raan, inc, argp, f1, mu)
            r2, _ = _keplerian_to_rv(-a_abs, e, raan, inc, argp, f2, mu)
            n_hyp = np.sqrt(mu / a_abs ** 3)
            tof = (_mean_anomaly(e, f2) - _mean_anomaly(e, f1)) / n_hyp
            if tof <= 0.0 or not np.isfinite(tof):
                continue
            if not (np.all(np.isfinite(r1)) and np.all(np.isfinite(r2))):
                continue
            r1_out.append(r1)
            r2_out.append(r2)
            tof_out.append(tof)
        except Exception:
            continue

    if not r1_out:
        return np.zeros((0, 3)), np.zeros((0, 3)), np.zeros(0)
    return np.array(r1_out), np.array(r2_out), np.array(tof_out)


def _sample_elliptic_mixed_revs(
    rng: np.random.Generator,
    n: int,
    rev_values: Tuple[int, ...],
    e_range: Tuple[float, float],
    theta_range: Tuple[float, float],
    a_range: Tuple[float, float] = (0.8, 1.5),
    mu: float = 1.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return a mixed elliptic batch with TOFs drawn from multiple natural rev counts."""

    if n <= 0 or len(rev_values) == 0:
        return np.zeros((0, 3)), np.zeros((0, 3)), np.zeros(0)

    counts = np.full(len(rev_values), n // len(rev_values), dtype=int)
    counts[: n % len(rev_values)] += 1

    r1_parts = []
    r2_parts = []
    tof_parts = []
    for count, n_rev in zip(counts, rev_values):
        if int(count) <= 0:
            continue
        r1_i, r2_i, tof_i = _sample_elliptic(
            rng,
            int(count),
            e_range=e_range,
            theta_range=theta_range,
            a_range=a_range,
            N_rev=int(n_rev),
            mu=mu,
        )
        if r1_i.shape[0] == 0:
            continue
        r1_parts.append(r1_i)
        r2_parts.append(r2_i)
        tof_parts.append(tof_i)

    if not r1_parts:
        return np.zeros((0, 3)), np.zeros((0, 3)), np.zeros(0)

    r1 = np.concatenate(r1_parts, axis=0)
    r2 = np.concatenate(r2_parts, axis=0)
    tof = np.concatenate(tof_parts, axis=0)
    return r1, r2, tof


def _sample_near_tmin(
    rng: np.random.Generator,
    n: int,
    N_rev: int = 1,
    eps_range: Tuple[float, float] = (1e-3, 2e-2),
    mu: float = 1.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate multi-rev cases with tof just above N × T_period (near Tmin).

    The r1/r2 positions come from a Keplerian orbit; the tof is then set to
    N_rev × T_period × (1 + ε) instead of the natural arc time, intentionally
    placing it near the minimum feasible time-of-flight boundary.
    """
    r1_out, r2_out, tof_out = [], [], []
    attempts = 0
    while len(r1_out) < n and attempts < n * 20:
        attempts += 1
        a = float(rng.uniform(0.8, 1.5))
        e = float(rng.uniform(0.01, 0.40))
        raan = float(rng.uniform(0.0, 2.0 * np.pi))
        inc  = float(rng.uniform(0.0, np.pi))
        argp = float(rng.uniform(0.0, 2.0 * np.pi))
        f1 = float(rng.uniform(-0.8 * np.pi, 0.8 * np.pi))
        dtheta = float(rng.uniform(np.deg2rad(20), np.deg2rad(120)))
        f2 = f1 + dtheta
        f2 = (f2 + np.pi) % (2.0 * np.pi) - np.pi
        try:
            r1, _ = _keplerian_to_rv(a, e, raan, inc, argp, f1, mu)
            r2, _ = _keplerian_to_rv(a, e, raan, inc, argp, f2, mu)
            T = 2.0 * np.pi * np.sqrt(a ** 3 / mu)
            eps = float(rng.uniform(*eps_range))
            tof = N_rev * T * (1.0 + eps)   # ≈ N·T, just above the parabolic lower bound
            if tof <= 0.0 or not np.isfinite(tof):
                continue
            if not (np.all(np.isfinite(r1)) and np.all(np.isfinite(r2))):
                continue
            r1_out.append(r1)
            r2_out.append(r2)
            tof_out.append(tof)
        except Exception:
            continue

    if not r1_out:
        return np.zeros((0, 3)), np.zeros((0, 3)), np.zeros(0)
    return np.array(r1_out), np.array(r2_out), np.array(tof_out)


# ---------------------------------------------------------------------------
# Result container and per-category runner
# ---------------------------------------------------------------------------

@dataclass
class ModeResult:
    n_rev: int
    branch: str
    hz_sign: int
    n_root_hits: int
    n_root_arcs: int
    n_vel_hits: int
    n_vel_arcs: int

    @property
    def hz_label(self) -> str:
        if self.hz_sign > 0:
            return "ccw"
        if self.hz_sign < 0:
            return "cw"
        return "deg"


@dataclass
class CaseResult:
    category: str
    n_tried: int       # input problems submitted
    n_root_inputs: int
    n_vel_inputs: int
    n_root_arcs: int
    n_vel_arcs: int
    mode_results: List[ModeResult]

    @property
    def root_pct(self) -> float:
        return 100.0 * self.n_root_inputs / self.n_tried if self.n_tried else 0.0

    @property
    def vel_pct(self) -> float:
        return 100.0 * self.n_vel_inputs / self.n_tried if self.n_tried else 0.0


def _branch_label(nrev_signed: int) -> str:
    nrev_signed = int(nrev_signed)
    if nrev_signed == 0:
        return "single"
    return "long" if nrev_signed > 0 else "short"


def _summarize_modes(
    root_idx: np.ndarray,
    root_nrev_signed: np.ndarray,
    root_hz_sign: np.ndarray,
    vel_idx: np.ndarray,
    vel_nrev_signed: np.ndarray,
    vel_hz_sign: np.ndarray,
) -> List[ModeResult]:
    if root_idx.size == 0 and vel_idx.size == 0:
        return []

    root_arc_count_by_mode: Dict[Tuple[int, str, int], int] = {}
    root_input_hits_by_mode: Dict[Tuple[int, str, int], set[int]] = {}
    vel_arc_count_by_mode: Dict[Tuple[int, str, int], int] = {}
    vel_input_hits_by_mode: Dict[Tuple[int, str, int], set[int]] = {}

    for input_idx, nrev_mode, hz_mode in zip(
        np.asarray(root_idx, dtype=int),
        np.asarray(root_nrev_signed, dtype=int),
        np.asarray(root_hz_sign, dtype=int),
    ):
        key = (abs(int(nrev_mode)), _branch_label(int(nrev_mode)), int(hz_mode))
        root_arc_count_by_mode[key] = int(root_arc_count_by_mode.get(key, 0) + 1)
        root_input_hits_by_mode.setdefault(key, set()).add(int(input_idx))

    for input_idx, nrev_mode, hz_mode in zip(
        np.asarray(vel_idx, dtype=int),
        np.asarray(vel_nrev_signed, dtype=int),
        np.asarray(vel_hz_sign, dtype=int),
    ):
        key = (abs(int(nrev_mode)), _branch_label(int(nrev_mode)), int(hz_mode))
        vel_arc_count_by_mode[key] = int(vel_arc_count_by_mode.get(key, 0) + 1)
        vel_input_hits_by_mode.setdefault(key, set()).add(int(input_idx))

    branch_order = {"single": 0, "long": 1, "short": 2}
    hz_order = {1: 0, -1: 1, 0: 2}

    keys = set(root_arc_count_by_mode) | set(vel_arc_count_by_mode)
    out = [
        ModeResult(
            n_rev=int(key[0]),
            branch=str(key[1]),
            hz_sign=int(key[2]),
            n_root_hits=int(len(root_input_hits_by_mode.get(key, set()))),
            n_root_arcs=int(root_arc_count_by_mode.get(key, 0)),
            n_vel_hits=int(len(vel_input_hits_by_mode.get(key, set()))),
            n_vel_arcs=int(vel_arc_count_by_mode.get(key, 0)),
        )
        for key in keys
    ]
    out.sort(key=lambda row: (row.n_rev, branch_order.get(row.branch, 99), hz_order.get(row.hz_sign, 99)))
    return out


def _run_category(
    label: str,
    r1: np.ndarray, r2: np.ndarray, tof: np.ndarray,
    mu: float = 1.0,
    N: int = 0,
    hz: int = 0,
    sweep_rev: bool = False,
    tol: float = 1e-6,
) -> CaseResult:
    """Solve one batch and summarize root-convergence vs velocity reconstruction."""
    B = r1.shape[0]
    if B == 0:
        return CaseResult(label, 0, 0, 0, 0, 0, [])

    _, _, _, _, _, diag = lambert_batch(
        r1, r2, tof, mu=mu, N=N, hz=hz, sweep_rev=sweep_rev, tol=tol, diagnostics=True
    )

    root_idx = np.asarray(diag["root_index"], dtype=int)
    root_nrev = np.asarray(diag["root_nrev_signed"], dtype=int)
    root_hz = np.asarray(diag["root_hz_sign"], dtype=int)
    vel_idx = np.asarray(diag["reconstructed_index"], dtype=int)
    vel_nrev = np.asarray(diag["reconstructed_nrev_signed"], dtype=int)
    vel_hz = np.asarray(diag["reconstructed_hz_sign"], dtype=int)

    return CaseResult(
        category=label,
        n_tried=B,
        n_root_inputs=int(np.unique(root_idx).size) if root_idx.size else 0,
        n_vel_inputs=int(np.unique(vel_idx).size) if vel_idx.size else 0,
        n_root_arcs=int(root_idx.size),
        n_vel_arcs=int(vel_idx.size),
        mode_results=_summarize_modes(root_idx, root_nrev, root_hz, vel_idx, vel_nrev, vel_hz),
    )


# ---------------------------------------------------------------------------
# Test configuration
# ---------------------------------------------------------------------------

_N = 2000          # cases per category
_SEED = 42
class TestLambertStress(unittest.TestCase):
    """Statistical convergence profiler for the Arora & Russell Lambert solver."""

    _results: List[CaseResult] = []

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------

    def _rng_for(self, offset: int) -> np.random.Generator:
        return np.random.default_rng(_SEED + int(offset))

    def _record_result(self, res: CaseResult) -> None:
        self._results.append(res)
        self.assertGreater(res.n_tried, 0, f"{res.category}: no sample cases were generated.")
        self.assertLessEqual(res.n_root_inputs, res.n_tried, f"{res.category}: root count exceeds tried count.")
        self.assertLessEqual(res.n_vel_inputs, res.n_root_inputs, f"{res.category}: velocity count exceeds root count.")
        self.assertGreaterEqual(res.n_root_arcs, res.n_root_inputs, f"{res.category}: root arc count is inconsistent.")
        self.assertGreaterEqual(res.n_vel_arcs, res.n_vel_inputs, f"{res.category}: velocity arc count is inconsistent.")
        self.assertLessEqual(res.n_vel_arcs, res.n_root_arcs, f"{res.category}: velocity arc count exceeds root arc count.")

    # ------------------------------------------------------------------
    # 1.  Zero-rev, generic elliptic
    # ------------------------------------------------------------------
    def test_01_zero_rev_random(self):
        r1, r2, tof = _sample_elliptic(
            self._rng_for(1), _N,
            e_range=(0.01, 0.80),
            theta_range=(np.deg2rad(5), np.deg2rad(175)),
        )
        res = _run_category("zero_rev_random", r1, r2, tof, N=0, hz=0)
        self._record_result(res)

    # ------------------------------------------------------------------
    # 2.  Zero-rev, near-180° transfer angle
    # ------------------------------------------------------------------
    def test_02_zero_rev_near_180deg(self):
        r1, r2, tof = _sample_elliptic(
            self._rng_for(2), _N,
            e_range=(0.01, 0.80),
            theta_range=(np.deg2rad(170), np.deg2rad(179.5)),
        )
        res = _run_category("zero_rev_near_180deg", r1, r2, tof, N=0, hz=0)
        self._record_result(res)

    # ------------------------------------------------------------------
    # 3.  Zero-rev, near-0° transfer angle
    # ------------------------------------------------------------------
    def test_03_zero_rev_near_0deg(self):
        r1, r2, tof = _sample_elliptic(
            self._rng_for(3), _N,
            e_range=(0.01, 0.80),
            theta_range=(np.deg2rad(0.2), np.deg2rad(10)),
        )
        res = _run_category("zero_rev_near_0deg", r1, r2, tof, N=0, hz=0)
        self._record_result(res)

    # ------------------------------------------------------------------
    # 4.  Zero-rev, near-parabolic elliptic  (e → 1⁻)
    # ------------------------------------------------------------------
    def test_04_near_parabolic_elliptic(self):
        r1, r2, tof = _sample_elliptic(
            self._rng_for(4), _N,
            e_range=(0.95, 0.9990),
            theta_range=(np.deg2rad(10), np.deg2rad(170)),
        )
        res = _run_category("near_parabolic_elliptic", r1, r2, tof, N=0, hz=0)
        self._record_result(res)

    # ------------------------------------------------------------------
    # 5.  Zero-rev, near-parabolic hyperbolic  (e → 1⁺)
    # ------------------------------------------------------------------
    def test_05_near_parabolic_hyperbolic(self):
        r1, r2, tof = _sample_hyperbolic(
            self._rng_for(5), _N,
            e_range=(1.001, 1.10),
        )
        res = _run_category("near_parabolic_hyperbolic", r1, r2, tof, N=0, hz=0)
        self._record_result(res)

    # ------------------------------------------------------------------
    # 6.  Zero-rev, deeply hyperbolic  (e >> 1)
    # ------------------------------------------------------------------
    def test_06_deeply_hyperbolic(self):
        r1, r2, tof = _sample_hyperbolic(
            self._rng_for(6), _N,
            e_range=(1.1, 10.0),
        )
        res = _run_category("zero_rev_hyperbolic", r1, r2, tof, N=0, hz=0)
        self._record_result(res)

    # ------------------------------------------------------------------
    # 7.  Zero-rev, near-circular  (e → 0)
    # ------------------------------------------------------------------
    def test_07_near_circular(self):
        r1, r2, tof = _sample_elliptic(
            self._rng_for(7), _N,
            e_range=(1e-5, 0.01),
            theta_range=(np.deg2rad(5), np.deg2rad(175)),
        )
        res = _run_category("zero_rev_near_circular", r1, r2, tof, N=0, hz=0)
        self._record_result(res)

    # ------------------------------------------------------------------
    # 8.  Multi-rev N=1, both LP and SP branches
    # ------------------------------------------------------------------
    def test_08_multirev_N1(self):
        r1, r2, tof = _sample_elliptic(
            self._rng_for(8), _N,
            e_range=(0.01, 0.70),
            theta_range=(np.deg2rad(20), np.deg2rad(160)),
            N_rev=1,
        )
        res = _run_category(
            "multirev_N1_both_branches", r1, r2, tof,
            N=1, hz=0, sweep_rev=False,
        )
        self._record_result(res)

    # ------------------------------------------------------------------
    # 9.  Multi-rev N=2, both LP and SP branches
    # ------------------------------------------------------------------
    def test_09_multirev_N2(self):
        r1, r2, tof = _sample_elliptic(
            self._rng_for(9), _N,
            e_range=(0.01, 0.60),
            theta_range=(np.deg2rad(20), np.deg2rad(160)),
            N_rev=2,
        )
        res = _run_category(
            "multirev_N2_both_branches", r1, r2, tof,
            N=2, hz=0, sweep_rev=False,
        )
        self._record_result(res)

    # ------------------------------------------------------------------
    # 10.  Multi-rev N=3, both branches
    # ------------------------------------------------------------------
    def test_10_multirev_N3(self):
        r1, r2, tof = _sample_elliptic(
            self._rng_for(10), _N,
            e_range=(0.01, 0.50),
            theta_range=(np.deg2rad(20), np.deg2rad(160)),
            N_rev=3,
        )
        res = _run_category(
            "multirev_N3_both_branches", r1, r2, tof,
            N=3, hz=0, sweep_rev=False,
        )
        self._record_result(res)

    # ------------------------------------------------------------------
    # 11.  Multi-rev, tof near minimum feasible  (tof ≈ N · T_period)
    # ------------------------------------------------------------------
    def test_11_multirev_near_Tmin(self):
        r1, r2, tof = _sample_near_tmin(
            self._rng_for(11), _N, N_rev=1, eps_range=(1e-3, 2e-2)
        )
        res = _run_category(
            "multirev_N1_near_Tmin", r1, r2, tof,
            N=1, hz=0, sweep_rev=False,
        )
        self._record_result(res)

    # ------------------------------------------------------------------
    # 12.  Near-360° total arc  (dθ ∈ [0.1°, 5°] + 1 full revolution)
    # ------------------------------------------------------------------
    def test_12_multirev_near_360deg(self):
        r1, r2, tof = _sample_elliptic(
            self._rng_for(12), _N,
            e_range=(0.01, 0.50),
            theta_range=(np.deg2rad(0.1), np.deg2rad(5)),
            N_rev=1,
        )
        res = _run_category(
            "multirev_near_360deg", r1, r2, tof,
            N=1, hz=0, sweep_rev=False,
        )
        self._record_result(res)

    # ------------------------------------------------------------------
    # 13.  Near-360° from above  (dθ ∈ [355°, 359.8°] within same rev)
    #       Equivalent to θ close to 360° for a zero-rev problem.
    # ------------------------------------------------------------------
    def test_13_zero_rev_near_360deg(self):
        r1, r2, tof = _sample_elliptic(
            self._rng_for(13), _N,
            e_range=(0.01, 0.50),
            theta_range=(np.deg2rad(355), np.deg2rad(359.8)),
            N_rev=0,
        )
        res = _run_category(
            "zero_rev_near_360deg", r1, r2, tof,
            N=0, hz=0, sweep_rev=False,
        )
        self._record_result(res)

    # ------------------------------------------------------------------
    # 14.  Sweep revolutions N_max = 5 with source TOFs drawn from N=3,4,5.
    #      This avoids under-sampling the higher-rev branches by generating all
    #      inputs from only an N=3 natural arc.
    # ------------------------------------------------------------------
    def test_14_sweep_rev_N5(self):
        r1, r2, tof = _sample_elliptic_mixed_revs(
            self._rng_for(14), _N,
            rev_values=(3, 4, 5),
            e_range=(0.01, 0.40),
            theta_range=(np.deg2rad(30), np.deg2rad(150)),
        )
        res = _run_category(
            "sweep_rev_N5", r1, r2, tof,
            N=5, hz=0, sweep_rev=True,
        )
        self._record_result(res)

    # ------------------------------------------------------------------
    # 15.  hz filter consistency: {hz=+1} ∪ {hz=−1} ⊆ {hz=0}
    # ------------------------------------------------------------------
    def test_15_hz_filter_consistency(self):
        rng2 = np.random.default_rng(_SEED + 99)
        r1, r2, tof = _sample_elliptic(
            rng2, _N,
            e_range=(0.01, 0.70),
            theta_range=(np.deg2rad(20), np.deg2rad(160)),
        )

        v1_all,  v2_all,  _, _, idx_all  = lambert_batch(r1, r2, tof, N=0, hz= 0)
        v1_ccw,  v2_ccw,  _, _, idx_ccw  = lambert_batch(r1, r2, tof, N=0, hz=+1)
        v1_cw,   v2_cw,   _, _, idx_cw   = lambert_batch(r1, r2, tof, N=0, hz=-1)

        def _hz_z(ri: np.ndarray, vi: np.ndarray, idx: np.ndarray) -> np.ndarray:
            if vi.shape[0] == 0:
                return np.zeros(0)
            return np.cross(ri[idx], vi)[:, 2]

        hz_all = _hz_z(r1, v1_all,  idx_all)
        hz_ccw = _hz_z(r1, v1_ccw,  idx_ccw)
        hz_cw  = _hz_z(r1, v1_cw,   idx_cw)

        if hz_ccw.size > 0:
            self.assertTrue(
                np.all(hz_ccw > -1e-10),
                f"CCW filter returned negative hz_z (min={hz_ccw.min():.3e})",
            )
        if hz_cw.size > 0:
            self.assertTrue(
                np.all(hz_cw < 1e-10),
                f"CW filter returned positive hz_z (max={hz_cw.max():.3e})",
            )
        # The union should not exceed the hz=0 count by more than rounding
        union_count = hz_ccw.size + hz_cw.size
        self.assertLessEqual(
            union_count, hz_all.size + _N,
            "CCW+CW arc count exceeds hz=0 total by more than n_cases",
        )

        res = _run_category("hz_filter_consistency", r1, r2, tof, N=0, hz=0)
        self._record_result(res)

    # ------------------------------------------------------------------
    # 16.  Large Monte-Carlo batch, zero-rev  (10 000 cases)
    # ------------------------------------------------------------------
    def test_16_large_batch_monte_carlo(self):
        rng_mc = np.random.default_rng(_SEED + 777)
        r1, r2, tof = _sample_elliptic(
            rng_mc, 10_000,
            e_range=(0.01, 0.90),
            theta_range=(np.deg2rad(1), np.deg2rad(179)),
        )
        res = _run_category(
            "large_batch_monte_carlo", r1, r2, tof,
            N=0, hz=0,
        )
        self._record_result(res)

    # ------------------------------------------------------------------
    # Summary table
    # ------------------------------------------------------------------
    @classmethod
    def tearDownClass(cls) -> None:
        results = cls._results
        if not results:
            return

        W = 128
        col = dict(cat=34, tried=8, rooti=8, rootp=7, veli=8, velp=7, roota=8, vela=8, modes=7)
        hdr = (
            f"\n{'=' * W}\n"
            f"{'Lambert Solver Stress Profile':^{W}}\n"
            f"{'=' * W}\n"
            f"{'Category':<{col['cat']}} "
            f"{'N_tried':>{col['tried']}} "
            f"{'RootIn':>{col['rooti']}} {'Root%':>{col['rootp']}} "
            f"{'VelIn':>{col['veli']}} {'Vel%':>{col['velp']}} "
            f"{'RootArcs':>{col['roota']}} {'VelArcs':>{col['vela']}} {'N_modes':>{col['modes']}}\n"
            f"{'-' * W}"
        )
        print(hdr)

        tot_tried = tot_root_inputs = tot_vel_inputs = tot_root_arcs = tot_vel_arcs = 0
        for r in results:
            print(
                f"{r.category:<{col['cat']}} "
                f"{r.n_tried:>{col['tried']}d} "
                f"{r.n_root_inputs:>{col['rooti']}d} "
                f"{r.root_pct:>{col['rootp']}.1f}% "
                f"{r.n_vel_inputs:>{col['veli']}d} "
                f"{r.vel_pct:>{col['velp']}.1f}% "
                f"{r.n_root_arcs:>{col['roota']}d} "
                f"{r.n_vel_arcs:>{col['vela']}d} "
                f"{len(r.mode_results):>{col['modes']}d}"
            )
            tot_tried += r.n_tried
            tot_root_inputs += r.n_root_inputs
            tot_vel_inputs += r.n_vel_inputs
            tot_root_arcs += r.n_root_arcs
            tot_vel_arcs += r.n_vel_arcs

        ov_root = 100.0 * tot_root_inputs / tot_tried if tot_tried else 0.0
        ov_vel = 100.0 * tot_vel_inputs / tot_tried if tot_tried else 0.0
        print(f"{'-' * W}")
        print(
            f"{'TOTAL':<{col['cat']}} "
            f"{tot_tried:>{col['tried']}d} "
            f"{tot_root_inputs:>{col['rooti']}d} "
            f"{ov_root:>{col['rootp']}.1f}% "
            f"{tot_vel_inputs:>{col['veli']}d} "
            f"{ov_vel:>{col['velp']}.1f}% "
            f"{tot_root_arcs:>{col['roota']}d} "
            f"{tot_vel_arcs:>{col['vela']}d} "
            f"{sum(len(r.mode_results) for r in results):>{col['modes']}d}"
        )
        print('=' * W)

        W_mode = 144
        col_mode = dict(cat=34, nrev=4, branch=7, hz=5, hit=9, hitp=7, arcs=8, drop=7)
        hdr_mode = (
            f"\n{'Per-Mode Root vs Velocity Breakdown':^{W_mode}}\n"
            f"{'-' * W_mode}\n"
            f"{'Category':<{col_mode['cat']}} "
            f"{'Nrev':>{col_mode['nrev']}} {'Branch':>{col_mode['branch']}} {'hz':>{col_mode['hz']}} "
            f"{'RootHits':>{col_mode['hit']}} {'Root%':>{col_mode['hitp']}} "
            f"{'VelHits':>{col_mode['hit']}} {'Vel%':>{col_mode['hitp']}} "
            f"{'RootArcs':>{col_mode['arcs']}} {'VelArcs':>{col_mode['arcs']}} {'Loss%':>{col_mode['drop']}}\n"
            f"{'-' * W_mode}"
        )
        print(hdr_mode)

        for r in results:
            if not r.mode_results:
                print(
                    f"{r.category:<{col_mode['cat']}} "
                    f"{'-':>{col_mode['nrev']}} {'-':>{col_mode['branch']}} {'-':>{col_mode['hz']}} "
                    f"{0:>{col_mode['hit']}d} {0.0:>{col_mode['hitp']}.1f}% "
                    f"{0:>{col_mode['hit']}d} {0.0:>{col_mode['hitp']}.1f}% "
                    f"{0:>{col_mode['arcs']}d} {0:>{col_mode['arcs']}d} {0.0:>{col_mode['drop']}.1f}%"
                )
                continue

            for mode in r.mode_results:
                root_hit_pct = 100.0 * mode.n_root_hits / r.n_tried if r.n_tried else 0.0
                vel_hit_pct = 100.0 * mode.n_vel_hits / r.n_tried if r.n_tried else 0.0
                loss_pct = 100.0 * (mode.n_root_arcs - mode.n_vel_arcs) / mode.n_root_arcs if mode.n_root_arcs else 0.0
                print(
                    f"{r.category:<{col_mode['cat']}} "
                    f"{mode.n_rev:>{col_mode['nrev']}d} "
                    f"{mode.branch:>{col_mode['branch']}} "
                    f"{mode.hz_label:>{col_mode['hz']}} "
                    f"{mode.n_root_hits:>{col_mode['hit']}d} "
                    f"{root_hit_pct:>{col_mode['hitp']}.1f}% "
                    f"{mode.n_vel_hits:>{col_mode['hit']}d} "
                    f"{vel_hit_pct:>{col_mode['hitp']}.1f}% "
                    f"{mode.n_root_arcs:>{col_mode['arcs']}d} "
                    f"{mode.n_vel_arcs:>{col_mode['arcs']}d} "
                    f"{loss_pct:>{col_mode['drop']}.1f}%"
                )
        print('-' * W_mode)


if __name__ == "__main__":
    unittest.main(verbosity=2)

"""Optional resonant-leg expansion helpers for STAR leg generation.

This module is intentionally narrow in scope:
- it works only from already-computed Lambert seed rows,
- it returns rows in the same leg-database shape used by `star.leg_database`,
- it does not touch flyby patching, combo stitching, or search costs.

`star.leg_database` should call this module only when resonant mode is enabled,
similar to how DSM leveraging is delegated to `star.maneuver_placement`.

See NOTATION.md for the resonant-leg definition and leg-row symbols.
"""

from __future__ import annotations

import sys
from typing import Optional, Sequence, Tuple

import numpy as np
from numba import njit, prange

sys.modules.setdefault("resonant", sys.modules[__name__])


def build_resonant_vinf_grid(
    vinf_min_km_s: float,
    vinf_max_km_s: float,
    dvinf_km_s: float,
    *,
    tol: float = 1e-12,
) -> np.ndarray:
    """Build inclusive departure-v-infinity magnitudes for resonant sampling."""

    vinf_min = float(vinf_min_km_s)
    vinf_max = float(vinf_max_km_s)
    dvinf = float(dvinf_km_s)
    if not np.isfinite(vinf_min) or not np.isfinite(vinf_max):
        raise ValueError("Resonant sampling requires finite departure v-infinity bounds.")
    if vinf_min < 0.0 or vinf_max < vinf_min:
        raise ValueError("Resonant sampling requires 0 <= vinf_min <= vinf_max.")
    if not np.isfinite(dvinf) or dvinf <= 0.0:
        raise ValueError("resonant_dvinf_km_s must be finite and > 0.")

    n_full = int(np.floor((vinf_max - vinf_min + float(tol)) / dvinf))
    if n_full <= 0:
        grid = np.asarray([vinf_min], dtype=float)
        if vinf_max - vinf_min > float(tol):
            grid = np.asarray([vinf_min, vinf_max], dtype=float)
    else:
        grid = vinf_min + dvinf * np.arange(n_full + 1, dtype=float)
        grid = grid[grid <= vinf_max + float(tol)]
        if grid.size == 0 or (vinf_max - float(grid[-1])) > float(tol):
            grid = np.concatenate((grid, np.asarray([vinf_max], dtype=float)))
        elif abs(float(grid[-1]) - vinf_max) <= float(tol):
            grid[-1] = vinf_max

    return np.unique(np.asarray(grid, dtype=float))


def _coerce_rows(name: str, value: Sequence[float], num_rows: int, *, dtype: type) -> np.ndarray:
    """Coerce one `(N,)` array."""

    array = np.asarray(value, dtype=dtype).reshape(-1)
    if array.shape != (num_rows,):
        raise ValueError(f"{name} must have shape ({num_rows},).")
    return array


def _coerce_vec3_rows(name: str, value: Sequence[float], num_rows: int) -> np.ndarray:
    """Coerce one `(N, 3)` finite float array."""

    array = np.asarray(value, dtype=float)
    try:
        array = array.reshape(-1, 3)
    except ValueError as exc:
        raise ValueError(f"{name} must have shape ({num_rows}, 3).") from exc
    if array.shape != (num_rows, 3):
        raise ValueError(f"{name} must have shape ({num_rows}, 3).")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite 3-vectors.")
    return np.ascontiguousarray(array, dtype=float)


def _validate_transfer_angle_inputs(
    r_dep_seed_km: np.ndarray,
    r_arr_seed_km: np.ndarray,
    v_dep_seed_km_s: np.ndarray,
    *,
    tol: float,
) -> None:
    """Match the legacy transfer-angle requirements before entering the JIT path."""

    r_dep_norm = np.linalg.norm(r_dep_seed_km, axis=1)
    r_arr_norm = np.linalg.norm(r_arr_seed_km, axis=1)
    if np.any(r_dep_norm <= float(tol)) or np.any(r_arr_norm <= float(tol)):
        raise ValueError("Resonant transfer-angle computation requires non-degenerate position vectors.")

    h_lambert = np.cross(r_dep_seed_km, v_dep_seed_km_s)
    if np.any(np.linalg.norm(h_lambert, axis=1) <= float(tol)):
        raise ValueError("Resonant transfer-angle computation requires non-degenerate Lambert angular momentum.")


@njit(cache=True)
def _clip_unit_scalar(value: float) -> float:
    if value < -1.0:
        return -1.0
    if value > 1.0:
        return 1.0
    return value


@njit(cache=True)
def _dot3_scalar(ax: float, ay: float, az: float, bx: float, by: float, bz: float) -> float:
    return ax * bx + ay * by + az * bz


@njit(cache=True)
def _cross3_scalar(
    ax: float,
    ay: float,
    az: float,
    bx: float,
    by: float,
    bz: float,
) -> Tuple[float, float, float]:
    return (
        ay * bz - az * by,
        az * bx - ax * bz,
        ax * by - ay * bx,
    )


@njit(cache=True)
def _norm3_scalar(x: float, y: float, z: float) -> float:
    return np.sqrt(x * x + y * y + z * z)


@njit(cache=True)
def _normalize3_scalar(x: float, y: float, z: float, tol: float) -> Tuple[float, float, float, bool]:
    norm = _norm3_scalar(x, y, z)
    if norm <= tol:
        return 0.0, 0.0, 0.0, False
    inv_norm = 1.0 / norm
    return x * inv_norm, y * inv_norm, z * inv_norm, True


@njit(cache=True)
def _prepare_resonant_seed(
    rdx: float,
    rdy: float,
    rdz: float,
    rax: float,
    ray: float,
    raz: float,
    vbdx: float,
    vbdy: float,
    vbdz: float,
    vdx: float,
    vdy: float,
    vdz: float,
    vax: float,
    vay: float,
    vaz: float,
    nrev_signed: int,
    angle_tol_rad: float,
    tol: float,
) -> Tuple[
    bool,
    int,
    float,
    float,
    float,
    float,
    float,
    float,
    float,
    float,
    float,
    float,
    float,
    float,
    float,
    float,
    float,
    float,
]:
    """Compute per-seed geometry shared by the count/fill passes."""

    rdhx, rdhy, rdhz, ok = _normalize3_scalar(rdx, rdy, rdz, tol)
    if not ok:
        return (
            False,
            0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        )

    rahx, rahy, rahz, ok = _normalize3_scalar(rax, ray, raz, tol)
    if not ok:
        return (
            False,
            0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        )

    hdx, hdy, hdz = _cross3_scalar(rdx, rdy, rdz, vdx, vdy, vdz)
    hhx, hhy, hhz, ok = _normalize3_scalar(hdx, hdy, hdz, tol)
    if not ok:
        return (
            False,
            0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        )

    crx, cry, crz = _cross3_scalar(rdhx, rdhy, rdhz, rahx, rahy, rahz)
    sin_term = _dot3_scalar(hhx, hhy, hhz, crx, cry, crz)
    cos_term = _clip_unit_scalar(_dot3_scalar(rdhx, rdhy, rdhz, rahx, rahy, rahz))
    theta_base = np.arctan2(sin_term, cos_term)
    if theta_base < 0.0:
        theta_base += 2.0 * np.pi

    theta_mod = np.mod(theta_base + 2.0 * np.pi * abs(nrev_signed), 2.0 * np.pi)
    family = 0
    if min(abs(theta_mod), abs(theta_mod - 2.0 * np.pi)) <= angle_tol_rad:
        family = 1
    elif abs(theta_mod - np.pi) <= angle_tol_rad:
        family = 2
    else:
        return (
            False,
            0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        )

    v_res = _norm3_scalar(vdx, vdy, vdz)
    v_body_speed = _norm3_scalar(vbdx, vbdy, vbdz)
    if v_res <= tol or v_body_speed <= tol:
        return (
            False,
            0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        )

    Vhx, Vhy, Vhz, ok = _normalize3_scalar(vbdx, vbdy, vbdz, tol)
    if not ok:
        return (
            False,
            0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        )

    wtx, wty, wtz = _cross3_scalar(rdx, rdy, rdz, Vhx, Vhy, Vhz)
    Whx, Why, Whz, ok = _normalize3_scalar(wtx, wty, wtz, tol)
    if not ok:
        return (
            False,
            0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        )

    utx, uty, utz = _cross3_scalar(Vhx, Vhy, Vhz, Whx, Why, Whz)
    Uhx, Uhy, Uhz, ok = _normalize3_scalar(utx, uty, utz, tol)
    if not ok:
        return (
            False,
            0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        )

    return (
        True,
        family,
        v_res,
        v_body_speed,
        Vhx,
        Vhy,
        Vhz,
        Whx,
        Why,
        Whz,
        Uhx,
        Uhy,
        Uhz,
        _dot3_scalar(rdx, rdy, rdz, Uhx, Uhy, Uhz),
        _dot3_scalar(rdx, rdy, rdz, vdx, vdy, vdz),
        _dot3_scalar(rdx, rdy, rdz, Vhx, Vhy, Vhz),
        _dot3_scalar(rax, ray, raz, rax, ray, raz),
        _dot3_scalar(rax, ray, raz, vax, vay, vaz),
    )


@njit(cache=True, parallel=True)
def _count_resonant_outputs(
    r_dep_seed_km: np.ndarray,
    r_arr_seed_km: np.ndarray,
    v_body_dep_seed_km_s: np.ndarray,
    v_dep_seed_km_s: np.ndarray,
    v_arr_seed_km_s: np.ndarray,
    nrev_signed_seed: np.ndarray,
    vinf_grid_km_s: np.ndarray,
    crank_grid: np.ndarray,
    angle_tol_rad: float,
    tol: float,
) -> np.ndarray:
    """Count output rows per seed for stable preallocation."""

    num_rows = r_dep_seed_km.shape[0]
    counts = np.zeros(num_rows, dtype=np.int64)

    for row_index in prange(num_rows):
        rdx, rdy, rdz = r_dep_seed_km[row_index]
        rax, ray, raz = r_arr_seed_km[row_index]
        vbdx, vbdy, vbdz = v_body_dep_seed_km_s[row_index]
        vdx, vdy, vdz = v_dep_seed_km_s[row_index]
        vax, vay, vaz = v_arr_seed_km_s[row_index]

        (
            ok,
            family,
            v_res,
            v_body_speed,
            Vhx,
            Vhy,
            Vhz,
            Whx,
            Why,
            Whz,
            Uhx,
            Uhy,
            Uhz,
            r_dep_dot_Uhat,
            r_dep_dot_v_lambert,
            r_dep_dot_Vhat,
            _r_arrival_norm_sq,
            _r_arrival_dot_v_lambert,
        ) = _prepare_resonant_seed(
            rdx,
            rdy,
            rdz,
            rax,
            ray,
            raz,
            vbdx,
            vbdy,
            vbdz,
            vdx,
            vdy,
            vdz,
            vax,
            vay,
            vaz,
            int(nrev_signed_seed[row_index]),
            angle_tol_rad,
            tol,
        )

        if not ok:
            continue

        denom = 2.0 * v_res * v_body_speed
        if denom <= tol:
            continue

        local_count = 0
        for vinf_mag_km_s in vinf_grid_km_s:
            cos_pump_raw = (v_res * v_res + v_body_speed * v_body_speed - vinf_mag_km_s * vinf_mag_km_s) / denom
            if cos_pump_raw < -1.0 - tol or cos_pump_raw > 1.0 + tol:
                continue

            pump_rad = np.arccos(_clip_unit_scalar(cos_pump_raw))
            sin_pump = np.sin(pump_rad)

            if family == 1:
                if crank_grid.size == 0:
                    continue
                for crank_rad in crank_grid:
                    if np.isfinite(crank_rad):
                        local_count += 1
                continue

            denom_crank = v_res * sin_pump * r_dep_dot_Uhat
            if abs(denom_crank) <= tol:
                continue

            numerator = r_dep_dot_v_lambert - v_res * np.cos(pump_rad) * r_dep_dot_Vhat
            cos_crank_raw = numerator / denom_crank
            if cos_crank_raw < -1.0 - tol or cos_crank_raw > 1.0 + tol:
                continue

            crank_mag = np.arccos(_clip_unit_scalar(cos_crank_raw))
            if crank_mag <= tol or abs(crank_mag - np.pi) <= tol:
                local_count += 1
            else:
                local_count += 2

        counts[row_index] = local_count

    return counts


@njit(cache=True)
def _write_resonant_output(
    parent_index: np.ndarray,
    vinfD_out: np.ndarray,
    vinfA_out: np.ndarray,
    output_index: int,
    row_index: int,
    rdx: float,
    rdy: float,
    rdz: float,
    rax: float,
    ray: float,
    raz: float,
    vbdx: float,
    vbdy: float,
    vbdz: float,
    vbax: float,
    vbay: float,
    vbaz: float,
    vax: float,
    vay: float,
    vaz: float,
    v_res: float,
    cos_pump: float,
    sin_pump: float,
    crank_rad: float,
    Vhx: float,
    Vhy: float,
    Vhz: float,
    Whx: float,
    Why: float,
    Whz: float,
    Uhx: float,
    Uhy: float,
    Uhz: float,
    r_arrival_norm_sq: float,
    r_arrival_dot_v_lambert: float,
) -> int:
    """Write one resonant output row and return the next index."""

    sin_crank = np.sin(crank_rad)
    cos_crank = np.cos(crank_rad)

    v_departure_x = v_res * (cos_pump * Vhx + sin_pump * sin_crank * Whx + sin_pump * cos_crank * Uhx)
    v_departure_y = v_res * (cos_pump * Vhy + sin_pump * sin_crank * Why + sin_pump * cos_crank * Uhy)
    v_departure_z = v_res * (cos_pump * Vhz + sin_pump * sin_crank * Whz + sin_pump * cos_crank * Uhz)

    h_leg_x, h_leg_y, h_leg_z = _cross3_scalar(rdx, rdy, rdz, v_departure_x, v_departure_y, v_departure_z)
    cross_arr_x, cross_arr_y, cross_arr_z = _cross3_scalar(h_leg_x, h_leg_y, h_leg_z, rax, ray, raz)

    v_arrival_x = (r_arrival_dot_v_lambert * rax + cross_arr_x) / r_arrival_norm_sq
    v_arrival_y = (r_arrival_dot_v_lambert * ray + cross_arr_y) / r_arrival_norm_sq
    v_arrival_z = (r_arrival_dot_v_lambert * raz + cross_arr_z) / r_arrival_norm_sq

    parent_index[output_index] = row_index

    vinfD_out[output_index, 0] = v_departure_x - vbdx
    vinfD_out[output_index, 1] = v_departure_y - vbdy
    vinfD_out[output_index, 2] = v_departure_z - vbdz

    vinfA_out[output_index, 0] = v_arrival_x - vbax
    vinfA_out[output_index, 1] = v_arrival_y - vbay
    vinfA_out[output_index, 2] = v_arrival_z - vbaz

    return output_index + 1


@njit(cache=True, parallel=True)
def _fill_resonant_outputs(
    offsets: np.ndarray,
    r_dep_seed_km: np.ndarray,
    r_arr_seed_km: np.ndarray,
    v_body_dep_seed_km_s: np.ndarray,
    v_body_arr_seed_km_s: np.ndarray,
    v_dep_seed_km_s: np.ndarray,
    v_arr_seed_km_s: np.ndarray,
    nrev_signed_seed: np.ndarray,
    vinf_grid_km_s: np.ndarray,
    crank_grid: np.ndarray,
    angle_tol_rad: float,
    tol: float,
    total_outputs: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fill preallocated resonant outputs using the prefix offsets."""

    parent_index = np.empty(total_outputs, dtype=np.int64)
    vinfD_out = np.empty((total_outputs, 3), dtype=np.float64)
    vinfA_out = np.empty((total_outputs, 3), dtype=np.float64)

    num_rows = r_dep_seed_km.shape[0]
    for row_index in prange(num_rows):
        output_index = offsets[row_index]

        rdx, rdy, rdz = r_dep_seed_km[row_index]
        rax, ray, raz = r_arr_seed_km[row_index]
        vbdx, vbdy, vbdz = v_body_dep_seed_km_s[row_index]
        vbax, vbay, vbaz = v_body_arr_seed_km_s[row_index]
        vdx, vdy, vdz = v_dep_seed_km_s[row_index]
        vax, vay, vaz = v_arr_seed_km_s[row_index]

        (
            ok,
            family,
            v_res,
            v_body_speed,
            Vhx,
            Vhy,
            Vhz,
            Whx,
            Why,
            Whz,
            Uhx,
            Uhy,
            Uhz,
            r_dep_dot_Uhat,
            r_dep_dot_v_lambert,
            r_dep_dot_Vhat,
            r_arrival_norm_sq,
            r_arrival_dot_v_lambert,
        ) = _prepare_resonant_seed(
            rdx,
            rdy,
            rdz,
            rax,
            ray,
            raz,
            vbdx,
            vbdy,
            vbdz,
            vdx,
            vdy,
            vdz,
            vax,
            vay,
            vaz,
            int(nrev_signed_seed[row_index]),
            angle_tol_rad,
            tol,
        )

        if not ok:
            continue

        denom = 2.0 * v_res * v_body_speed
        if denom <= tol:
            continue

        for vinf_mag_km_s in vinf_grid_km_s:
            cos_pump_raw = (v_res * v_res + v_body_speed * v_body_speed - vinf_mag_km_s * vinf_mag_km_s) / denom
            if cos_pump_raw < -1.0 - tol or cos_pump_raw > 1.0 + tol:
                continue

            pump_rad = np.arccos(_clip_unit_scalar(cos_pump_raw))
            cos_pump = np.cos(pump_rad)
            sin_pump = np.sin(pump_rad)

            if family == 1:
                if crank_grid.size == 0:
                    continue
                for crank_rad in crank_grid:
                    if not np.isfinite(crank_rad):
                        continue
                    output_index = _write_resonant_output(
                        parent_index,
                        vinfD_out,
                        vinfA_out,
                        output_index,
                        row_index,
                        rdx,
                        rdy,
                        rdz,
                        rax,
                        ray,
                        raz,
                        vbdx,
                        vbdy,
                        vbdz,
                        vbax,
                        vbay,
                        vbaz,
                        vax,
                        vay,
                        vaz,
                        v_res,
                        cos_pump,
                        sin_pump,
                        crank_rad,
                        Vhx,
                        Vhy,
                        Vhz,
                        Whx,
                        Why,
                        Whz,
                        Uhx,
                        Uhy,
                        Uhz,
                        r_arrival_norm_sq,
                        r_arrival_dot_v_lambert,
                    )
                continue

            denom_crank = v_res * sin_pump * r_dep_dot_Uhat
            if abs(denom_crank) <= tol:
                continue

            numerator = r_dep_dot_v_lambert - v_res * cos_pump * r_dep_dot_Vhat
            cos_crank_raw = numerator / denom_crank
            if cos_crank_raw < -1.0 - tol or cos_crank_raw > 1.0 + tol:
                continue

            crank_mag = np.arccos(_clip_unit_scalar(cos_crank_raw))
            if crank_mag <= tol:
                output_index = _write_resonant_output(
                    parent_index,
                    vinfD_out,
                    vinfA_out,
                    output_index,
                    row_index,
                    rdx,
                    rdy,
                    rdz,
                    rax,
                    ray,
                    raz,
                    vbdx,
                    vbdy,
                    vbdz,
                    vbax,
                    vbay,
                    vbaz,
                    vax,
                    vay,
                    vaz,
                    v_res,
                    cos_pump,
                    sin_pump,
                    0.0,
                    Vhx,
                    Vhy,
                    Vhz,
                    Whx,
                    Why,
                    Whz,
                    Uhx,
                    Uhy,
                    Uhz,
                    r_arrival_norm_sq,
                    r_arrival_dot_v_lambert,
                )
            elif abs(crank_mag - np.pi) <= tol:
                output_index = _write_resonant_output(
                    parent_index,
                    vinfD_out,
                    vinfA_out,
                    output_index,
                    row_index,
                    rdx,
                    rdy,
                    rdz,
                    rax,
                    ray,
                    raz,
                    vbdx,
                    vbdy,
                    vbdz,
                    vbax,
                    vbay,
                    vbaz,
                    vax,
                    vay,
                    vaz,
                    v_res,
                    cos_pump,
                    sin_pump,
                    np.pi,
                    Vhx,
                    Vhy,
                    Vhz,
                    Whx,
                    Why,
                    Whz,
                    Uhx,
                    Uhy,
                    Uhz,
                    r_arrival_norm_sq,
                    r_arrival_dot_v_lambert,
                )
            else:
                output_index = _write_resonant_output(
                    parent_index,
                    vinfD_out,
                    vinfA_out,
                    output_index,
                    row_index,
                    rdx,
                    rdy,
                    rdz,
                    rax,
                    ray,
                    raz,
                    vbdx,
                    vbdy,
                    vbdz,
                    vbax,
                    vbay,
                    vbaz,
                    vax,
                    vay,
                    vaz,
                    v_res,
                    cos_pump,
                    sin_pump,
                    -crank_mag,
                    Vhx,
                    Vhy,
                    Vhz,
                    Whx,
                    Why,
                    Whz,
                    Uhx,
                    Uhy,
                    Uhz,
                    r_arrival_norm_sq,
                    r_arrival_dot_v_lambert,
                )
                output_index = _write_resonant_output(
                    parent_index,
                    vinfD_out,
                    vinfA_out,
                    output_index,
                    row_index,
                    rdx,
                    rdy,
                    rdz,
                    rax,
                    ray,
                    raz,
                    vbdx,
                    vbdy,
                    vbdz,
                    vbax,
                    vbay,
                    vbaz,
                    vax,
                    vay,
                    vaz,
                    v_res,
                    cos_pump,
                    sin_pump,
                    crank_mag,
                    Vhx,
                    Vhy,
                    Vhz,
                    Whx,
                    Why,
                    Whz,
                    Uhx,
                    Uhy,
                    Uhz,
                    r_arrival_norm_sq,
                    r_arrival_dot_v_lambert,
                )

    return parent_index, vinfD_out, vinfA_out


def generate_resonant_rows(
    *,
    dep_rows_seed: np.ndarray,
    arr_rows_seed: np.ndarray,
    dt_seed_s: np.ndarray,
    r_dep_seed_km: np.ndarray,
    r_arr_seed_km: np.ndarray,
    v_body_dep_seed_km_s: np.ndarray,
    v_body_arr_seed_km_s: np.ndarray,
    v_dep_seed_km_s: np.ndarray,
    v_arr_seed_km_s: np.ndarray,
    nrev_signed_seed: np.ndarray,
    vinf_min_km_s: float,
    vinf_max_km_s: float,
    dvinf_km_s: float,
    angle_tol_deg: float = 4.0,
    crank_angles_rad: Sequence[float] = (),
    tol: float = 1e-12,
) -> Tuple[np.ndarray, ...]:
    """Generate resonant leg rows from near-resonant Lambert seeds.

    Output columns match the append path in `star.leg_database`:
    `dep_rows, arr_rows, dt_s, vinfD, vinfA, n_rev`.
    """

    empty_i = np.empty(0, dtype=np.int64)
    empty_f = np.empty(0, dtype=float)
    empty_v = np.empty((0, 3), dtype=float)

    if np.asarray(v_dep_seed_km_s, dtype=float).size == 0:
        return empty_i, empty_i, empty_f, empty_v, empty_v, empty_i

    dep_rows_seed = np.asarray(dep_rows_seed, dtype=np.int64).reshape(-1)
    num_rows = int(dep_rows_seed.size)
    arr_rows_seed = _coerce_rows("arr_rows_seed", arr_rows_seed, num_rows, dtype=np.int64)
    dt_seed_s = _coerce_rows("dt_seed_s", dt_seed_s, num_rows, dtype=float)
    nrev_signed_seed = _coerce_rows("nrev_signed_seed", nrev_signed_seed, num_rows, dtype=np.int64)

    r_dep_seed_km = _coerce_vec3_rows("r_dep_seed_km", r_dep_seed_km, num_rows)
    r_arr_seed_km = _coerce_vec3_rows("r_arr_seed_km", r_arr_seed_km, num_rows)
    v_body_dep_seed_km_s = _coerce_vec3_rows("v_body_dep_seed_km_s", v_body_dep_seed_km_s, num_rows)
    v_body_arr_seed_km_s = _coerce_vec3_rows("v_body_arr_seed_km_s", v_body_arr_seed_km_s, num_rows)
    v_dep_seed_km_s = _coerce_vec3_rows("v_dep_seed_km_s", v_dep_seed_km_s, num_rows)
    v_arr_seed_km_s = _coerce_vec3_rows("v_arr_seed_km_s", v_arr_seed_km_s, num_rows)

    angle_tol = float(angle_tol_deg)
    if not np.isfinite(angle_tol) or angle_tol < 0.0:
        raise ValueError("angle_tol_deg must be finite and >= 0.")

    _validate_transfer_angle_inputs(
        r_dep_seed_km,
        r_arr_seed_km,
        v_dep_seed_km_s,
        tol=float(tol),
    )

    vinf_grid_km_s = np.ascontiguousarray(
        build_resonant_vinf_grid(vinf_min_km_s, vinf_max_km_s, dvinf_km_s, tol=tol),
        dtype=float,
    )
    crank_grid = np.ascontiguousarray(np.unique(np.asarray(crank_angles_rad, dtype=float).reshape(-1)), dtype=float)
    angle_tol_rad = float(np.deg2rad(angle_tol))

    counts = _count_resonant_outputs(
        r_dep_seed_km,
        r_arr_seed_km,
        v_body_dep_seed_km_s,
        v_dep_seed_km_s,
        v_arr_seed_km_s,
        nrev_signed_seed,
        vinf_grid_km_s,
        crank_grid,
        angle_tol_rad,
        float(tol),
    )

    total_outputs = int(np.sum(counts, dtype=np.int64))
    if total_outputs <= 0:
        return empty_i, empty_i, empty_f, empty_v, empty_v, empty_i

    offsets = np.empty(num_rows, dtype=np.int64)
    offsets[0] = 0
    if num_rows > 1:
        offsets[1:] = np.cumsum(counts[:-1], dtype=np.int64)

    parent_index, vinfD_out, vinfA_out = _fill_resonant_outputs(
        offsets,
        r_dep_seed_km,
        r_arr_seed_km,
        v_body_dep_seed_km_s,
        v_body_arr_seed_km_s,
        v_dep_seed_km_s,
        v_arr_seed_km_s,
        nrev_signed_seed,
        vinf_grid_km_s,
        crank_grid,
        angle_tol_rad,
        float(tol),
        total_outputs,
    )

    return (
        dep_rows_seed[parent_index],
        arr_rows_seed[parent_index],
        dt_seed_s[parent_index],
        vinfD_out,
        vinfA_out,
        nrev_signed_seed[parent_index].astype(np.int64, copy=False),
    )


__all__ = ["build_resonant_vinf_grid", "generate_resonant_rows"]

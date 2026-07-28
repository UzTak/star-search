import sys
import unittest
from pathlib import Path
from unittest import mock

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = REPO_ROOT / "star"
for path in (str(REPO_ROOT), str(PACKAGE_ROOT)):
    if path not in sys.path:
        sys.path.insert(0, path)

from star import leg_database
from star.encounter_database import EncounterDB, EncounterEntry
from star.resonant import build_resonant_vinf_grid, generate_resonant_rows


def _legacy_as_vec3(value) -> np.ndarray:
    array = np.asarray(value, dtype=float).reshape(-1)
    if array.shape != (3,):
        raise ValueError("expected a 3-vector")
    if not np.all(np.isfinite(array)):
        raise ValueError("expected a finite 3-vector")
    return array


def _legacy_normalize_vec3(value, *, tol: float = 1e-12) -> np.ndarray:
    vector = _legacy_as_vec3(value)
    norm = float(np.linalg.norm(vector))
    if norm <= float(tol):
        raise ValueError("expected a non-degenerate 3-vector")
    return vector / norm


def _legacy_compute_lambert_transfer_angle_rad(
    r_departure_km,
    r_arrival_km,
    v_departure_km_s,
    nrev_signed: int,
) -> float:
    r_departure_hat = _legacy_normalize_vec3(r_departure_km)
    r_arrival_hat = _legacy_normalize_vec3(r_arrival_km)
    hhat = _legacy_normalize_vec3(np.cross(_legacy_as_vec3(r_departure_km), _legacy_as_vec3(v_departure_km_s)))

    sin_term = float(np.dot(hhat, np.cross(r_departure_hat, r_arrival_hat)))
    cos_term = float(np.clip(np.dot(r_departure_hat, r_arrival_hat), -1.0, 1.0))
    theta_base = float(np.arctan2(sin_term, cos_term))
    if theta_base < 0.0:
        theta_base += 2.0 * np.pi
    return float(theta_base + 2.0 * np.pi * abs(int(nrev_signed)))


def _legacy_classify_resonant_family(transfer_angle_rad: float, angle_tol_rad: float):
    theta_mod = float(np.mod(float(transfer_angle_rad), 2.0 * np.pi))
    tol = float(angle_tol_rad)
    if min(abs(theta_mod), abs(theta_mod - 2.0 * np.pi)) <= tol:
        return "full_rev"
    if abs(theta_mod - np.pi) <= tol:
        return "pi_transfer"
    return None


def _legacy_generate_resonant_rows(
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
    crank_angles_rad=(),
    tol: float = 1e-12,
):
    empty_i = np.empty(0, dtype=np.int64)
    empty_f = np.empty(0, dtype=float)
    empty_v = np.empty((0, 3), dtype=float)

    if np.asarray(v_dep_seed_km_s, dtype=float).size == 0:
        return empty_i, empty_i, empty_f, empty_v, empty_v, empty_i

    vinf_grid_km_s = np.asarray(
        build_resonant_vinf_grid(vinf_min_km_s, vinf_max_km_s, dvinf_km_s, tol=tol),
        dtype=float,
    )
    crank_grid = np.unique(np.asarray(crank_angles_rad, dtype=float).reshape(-1))
    angle_tol_rad = float(np.deg2rad(angle_tol_deg))

    out_dep_rows = []
    out_arr_rows = []
    out_dt_s = []
    out_vinfD = []
    out_vinfA = []
    out_n_rev = []

    dep_rows_seed = np.asarray(dep_rows_seed, dtype=np.int64).reshape(-1)
    arr_rows_seed = np.asarray(arr_rows_seed, dtype=np.int64).reshape(-1)
    dt_seed_s = np.asarray(dt_seed_s, dtype=float).reshape(-1)
    nrev_signed_seed = np.asarray(nrev_signed_seed, dtype=np.int64).reshape(-1)

    for row_index in range(int(dep_rows_seed.size)):
        r_departure = _legacy_as_vec3(r_dep_seed_km[row_index])
        r_arrival = _legacy_as_vec3(r_arr_seed_km[row_index])
        v_body_departure = _legacy_as_vec3(v_body_dep_seed_km_s[row_index])
        v_body_arrival = _legacy_as_vec3(v_body_arr_seed_km_s[row_index])
        v_departure_lambert = _legacy_as_vec3(v_dep_seed_km_s[row_index])
        v_arrival_lambert = _legacy_as_vec3(v_arr_seed_km_s[row_index])

        family = _legacy_classify_resonant_family(
            _legacy_compute_lambert_transfer_angle_rad(
                r_departure,
                r_arrival,
                v_departure_lambert,
                int(nrev_signed_seed[row_index]),
            ),
            angle_tol_rad,
        )
        if family is None:
            continue

        v_res = float(np.linalg.norm(v_departure_lambert))
        v_body_speed = float(np.linalg.norm(v_body_departure))
        if v_res <= float(tol) or v_body_speed <= float(tol):
            continue

        try:
            Vhat = _legacy_normalize_vec3(v_body_departure, tol=tol)
            What = _legacy_normalize_vec3(np.cross(r_departure, Vhat), tol=tol)
            Uhat = _legacy_normalize_vec3(np.cross(Vhat, What), tol=tol)
        except ValueError:
            continue

        for vinf_mag_km_s in vinf_grid_km_s:
            denom = 2.0 * v_res * v_body_speed
            if denom <= float(tol):
                continue

            cos_pump = (v_res * v_res + v_body_speed * v_body_speed - float(vinf_mag_km_s) ** 2) / denom
            if cos_pump < -1.0 - float(tol) or cos_pump > 1.0 + float(tol):
                continue
            pump_rad = float(np.arccos(np.clip(cos_pump, -1.0, 1.0)))
            sin_pump = float(np.sin(pump_rad))

            if family == "full_rev":
                if crank_grid.size == 0:
                    continue
                crank_values = crank_grid
            else:
                denom_crank = v_res * sin_pump * float(np.dot(r_departure, Uhat))
                if abs(denom_crank) <= float(tol):
                    continue

                numerator = (
                    float(np.dot(r_departure, v_departure_lambert))
                    - v_res * float(np.cos(pump_rad)) * float(np.dot(r_departure, Vhat))
                )
                cos_crank = numerator / denom_crank
                if cos_crank < -1.0 - float(tol) or cos_crank > 1.0 + float(tol):
                    continue

                crank_mag = float(np.arccos(np.clip(cos_crank, -1.0, 1.0)))
                if crank_mag <= float(tol):
                    crank_values = np.asarray([0.0], dtype=float)
                elif abs(crank_mag - np.pi) <= float(tol):
                    crank_values = np.asarray([np.pi], dtype=float)
                else:
                    crank_values = np.asarray([-crank_mag, crank_mag], dtype=float)

            for crank_rad in np.asarray(crank_values, dtype=float).reshape(-1):
                v_departure = v_res * (
                    float(np.cos(pump_rad)) * Vhat
                    + sin_pump * float(np.sin(crank_rad)) * What
                    + sin_pump * float(np.cos(crank_rad)) * Uhat
                )
                h_leg = np.cross(r_departure, v_departure)

                r_arrival_norm_sq = float(np.dot(r_arrival, r_arrival))
                if r_arrival_norm_sq <= float(tol):
                    continue

                v_arrival = (
                    float(np.dot(r_arrival, v_arrival_lambert)) * r_arrival
                    + np.cross(h_leg, r_arrival)
                ) / r_arrival_norm_sq

                vinf_departure = v_departure - v_body_departure
                vinf_arrival = v_arrival - v_body_arrival
                if not (
                    np.all(np.isfinite(v_departure))
                    and np.all(np.isfinite(v_arrival))
                    and np.all(np.isfinite(vinf_departure))
                    and np.all(np.isfinite(vinf_arrival))
                ):
                    continue

                out_dep_rows.append(int(dep_rows_seed[row_index]))
                out_arr_rows.append(int(arr_rows_seed[row_index]))
                out_dt_s.append(float(dt_seed_s[row_index]))
                out_vinfD.append(vinf_departure)
                out_vinfA.append(vinf_arrival)
                out_n_rev.append(int(nrev_signed_seed[row_index]))

    if not out_dep_rows:
        return empty_i, empty_i, empty_f, empty_v, empty_v, empty_i

    return (
        np.asarray(out_dep_rows, dtype=np.int64),
        np.asarray(out_arr_rows, dtype=np.int64),
        np.asarray(out_dt_s, dtype=float),
        np.asarray(out_vinfD, dtype=float).reshape(-1, 3),
        np.asarray(out_vinfA, dtype=float).reshape(-1, 3),
        np.asarray(out_n_rev, dtype=np.int64),
    )


class ResonantLegBuilderTest(unittest.TestCase):
    def _assert_resonant_outputs_equal(self, actual, expected) -> None:
        self.assertEqual(len(actual), len(expected))
        for actual_part, expected_part in zip(actual, expected):
            actual_array = np.asarray(actual_part)
            expected_array = np.asarray(expected_part)
            self.assertEqual(actual_array.shape, expected_array.shape)
            if actual_array.dtype.kind in {"f", "c"} or expected_array.dtype.kind in {"f", "c"}:
                np.testing.assert_allclose(actual_array, expected_array, rtol=1e-12, atol=1e-12)
            else:
                np.testing.assert_array_equal(actual_array, expected_array)

    def _build_encounter_db(self) -> EncounterDB:
        eps_rad = float(np.deg2rad(1.0))
        dep_entry = EncounterEntry(
            IE=0,
            stage_id=0,
            t_et=0.0,
            body=399,
            r_km=np.asarray([1.0, 0.0, 0.0], dtype=float),
            v_km_s=np.asarray([0.0, 1.0, 0.0], dtype=float),
            mu_km3_s2=1.0,
            rmin_km=1.0,
        )
        arr_entry = EncounterEntry(
            IE=1,
            stage_id=1,
            t_et=10.0,
            body=399,
            r_km=np.asarray([np.cos(eps_rad), np.sin(eps_rad), 0.0], dtype=float),
            v_km_s=np.asarray([-np.sin(eps_rad), np.cos(eps_rad), 0.0], dtype=float),
            mu_km3_s2=1.0,
            rmin_km=1.0,
        )
        return EncounterDB(
            entries=[dep_entry, arr_entry],
            stage_to_entry_ids={(0, 399): [0], (1, 399): [1]},
            entry_by_id={0: dep_entry, 1: arr_entry},
        )

    def _base_cfg(self, **overrides) -> leg_database.LegBuildConfig:
        kwargs = dict(
            leg_stage_id=0,
            dep_stage_id=0,
            arr_stage_id=1,
            tof_min_s=1.0,
            tof_max_s=20.0,
            vinfD_bounds_km_s=(0.0, 1.0),
            vinfA_bounds_km_s=(0.0, 1.0),
            lambert_mu_km3_s2=1.0,
            lambert_nrev_max=1,
            lambert_hz=1,
            dvlev_max_km_s=0.0,
            delta_dvlev_km_s=0.0,
            resonant_enabled=False,
            resonant_dvinf_km_s=0.5,
            resonant_crank_angles_rad=(0.0, 0.5 * np.pi),
        )
        kwargs.update(overrides)
        return leg_database.LegBuildConfig(**kwargs)

    def test_resonant_rows_append_without_changing_off_mode(self) -> None:
        encounter_db = self._build_encounter_db()
        eps_rad = float(np.deg2rad(1.0))
        mock_v_departure = np.asarray([[0.0, 1.0, 0.0]], dtype=float)
        mock_v_arrival = np.asarray([[-np.sin(eps_rad), np.cos(eps_rad), 0.0]], dtype=float)
        mock_nrev = np.asarray([1], dtype=np.int64)
        mock_source_idx = np.asarray([0], dtype=np.int64)

        with mock.patch.object(
            leg_database,
            "_solve_lambert_candidates",
            return_value=(mock_v_departure, mock_v_arrival, mock_nrev, mock_source_idx),
        ):
            baseline_db = leg_database.build_leg_db(encounter_db, self._base_cfg())
            resonant_off_db = leg_database.build_leg_db(
                encounter_db,
                self._base_cfg(
                    resonant_enabled=False,
                    resonant_angle_tol_deg=4.0,
                ),
            )
            resonant_on_db = leg_database.build_leg_db(
                encounter_db,
                self._base_cfg(
                    resonant_enabled=True,
                    resonant_angle_tol_deg=4.0,
                ),
            )

        self.assertEqual(int(baseline_db.IL.size), 1)
        np.testing.assert_array_equal(resonant_off_db.ID, baseline_db.ID)
        np.testing.assert_array_equal(resonant_off_db.IA, baseline_db.IA)
        np.testing.assert_array_equal(resonant_off_db.n_rev, baseline_db.n_rev)
        np.testing.assert_allclose(resonant_off_db.vinfD_km_s, baseline_db.vinfD_km_s)
        np.testing.assert_allclose(resonant_off_db.vinfA_km_s, baseline_db.vinfA_km_s)
        np.testing.assert_allclose(resonant_off_db.dv_lev_km_s, baseline_db.dv_lev_km_s)
        np.testing.assert_allclose(resonant_off_db.eta_lev, baseline_db.eta_lev)

        self.assertGreater(int(resonant_on_db.IL.size), int(baseline_db.IL.size))
        np.testing.assert_allclose(resonant_on_db.vinfD_km_s[0], baseline_db.vinfD_km_s[0])
        np.testing.assert_allclose(resonant_on_db.vinfA_km_s[0], baseline_db.vinfA_km_s[0])
        appended_slice = slice(int(baseline_db.IL.size), int(resonant_on_db.IL.size))
        self.assertTrue(np.all(resonant_on_db.dv_lev_km_s[appended_slice] == 0.0))

    def test_generate_resonant_rows_matches_legacy_reference(self) -> None:
        eps_rad = float(np.deg2rad(1.0))
        dep_rows_seed = np.asarray([10, 20, 30, 40], dtype=np.int64)
        arr_rows_seed = np.asarray([11, 21, 31, 41], dtype=np.int64)
        dt_seed_s = np.asarray([100.0, 200.0, 300.0, 400.0], dtype=float)
        r_dep_seed_km = np.asarray(
            [
                [1.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
            ],
            dtype=float,
        )
        r_arr_seed_km = np.asarray(
            [
                [-1.0, 0.0, 0.0],
                [np.cos(eps_rad), np.sin(eps_rad), 0.0],
                [0.0, 1.0, 0.0],
                [-1.0, 0.0, 0.0],
            ],
            dtype=float,
        )
        v_body_dep_seed_km_s = np.asarray(
            [
                [0.0, 1.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0],
            ],
            dtype=float,
        )
        v_body_arr_seed_km_s = np.asarray(
            [
                [0.0, -1.0, 0.0],
                [-np.sin(eps_rad), np.cos(eps_rad), 0.0],
                [-1.0, 0.0, 0.0],
                [0.0, -1.0, 0.0],
            ],
            dtype=float,
        )
        v_dep_seed_km_s = np.asarray(
            [
                [0.0, 1.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 1.0, 0.0],
            ],
            dtype=float,
        )
        v_arr_seed_km_s = np.asarray(
            [
                [0.0, -1.0, 0.0],
                [-np.sin(eps_rad), np.cos(eps_rad), 0.0],
                [-1.0, 0.0, 0.0],
                [0.0, -1.0, 0.0],
            ],
            dtype=float,
        )
        nrev_signed_seed = np.asarray([0, 1, 0, 0], dtype=np.int64)
        kwargs = dict(
            dep_rows_seed=dep_rows_seed,
            arr_rows_seed=arr_rows_seed,
            dt_seed_s=dt_seed_s,
            r_dep_seed_km=r_dep_seed_km,
            r_arr_seed_km=r_arr_seed_km,
            v_body_dep_seed_km_s=v_body_dep_seed_km_s,
            v_body_arr_seed_km_s=v_body_arr_seed_km_s,
            v_dep_seed_km_s=v_dep_seed_km_s,
            v_arr_seed_km_s=v_arr_seed_km_s,
            nrev_signed_seed=nrev_signed_seed,
            vinf_min_km_s=0.0,
            vinf_max_km_s=1.0,
            dvinf_km_s=0.5,
            angle_tol_deg=4.0,
            crank_angles_rad=(0.0, 0.5 * np.pi, np.pi),
        )

        expected = _legacy_generate_resonant_rows(**kwargs)
        actual = generate_resonant_rows(**kwargs)
        self._assert_resonant_outputs_equal(actual, expected)

    def test_generate_resonant_rows_matches_legacy_for_empty_input(self) -> None:
        kwargs = dict(
            dep_rows_seed=np.empty(0, dtype=np.int64),
            arr_rows_seed=np.empty(0, dtype=np.int64),
            dt_seed_s=np.empty(0, dtype=float),
            r_dep_seed_km=np.empty((0, 3), dtype=float),
            r_arr_seed_km=np.empty((0, 3), dtype=float),
            v_body_dep_seed_km_s=np.empty((0, 3), dtype=float),
            v_body_arr_seed_km_s=np.empty((0, 3), dtype=float),
            v_dep_seed_km_s=np.empty((0, 3), dtype=float),
            v_arr_seed_km_s=np.empty((0, 3), dtype=float),
            nrev_signed_seed=np.empty(0, dtype=np.int64),
            vinf_min_km_s=0.0,
            vinf_max_km_s=1.0,
            dvinf_km_s=0.5,
            angle_tol_deg=4.0,
            crank_angles_rad=(0.0,),
        )

        expected = _legacy_generate_resonant_rows(**kwargs)
        actual = generate_resonant_rows(**kwargs)
        self._assert_resonant_outputs_equal(actual, expected)


if __name__ == "__main__":
    unittest.main()

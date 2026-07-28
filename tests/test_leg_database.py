import os
import sys
import tempfile
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
from star.stage_db_npy import load_leg_stage


def _legacy_generate_candidate_pairs(
    dep_t_et_s: np.ndarray,
    arr_t_et_s: np.ndarray,
    tof_min_s: float,
    tof_max_s: float,
    dt_positive_tol_s: float,
):
    lower_dt_s = max(float(dt_positive_tol_s), float(tof_min_s)) if np.isfinite(tof_min_s) else float(dt_positive_tol_s)

    dep_chunks = []
    arr_chunks = []
    for dep_row_index, t_dep_et_s in enumerate(dep_t_et_s):
        t_low = float(t_dep_et_s) + lower_dt_s
        left = int(np.searchsorted(arr_t_et_s, t_low, side="left"))

        if np.isfinite(tof_max_s):
            t_high = float(t_dep_et_s) + float(tof_max_s)
            right = int(np.searchsorted(arr_t_et_s, t_high, side="right"))
        else:
            right = int(arr_t_et_s.size)

        if right <= left:
            continue

        arr_rows = np.arange(left, right, dtype=np.int64)
        dep_rows = np.full(arr_rows.shape[0], int(dep_row_index), dtype=np.int64)
        dep_chunks.append(dep_rows)
        arr_chunks.append(arr_rows)

    if not dep_chunks:
        empty = np.empty(0, dtype=np.int64)
        return empty, empty, np.empty(0, dtype=float)

    dep_row_indices = np.concatenate(dep_chunks)
    arr_row_indices = np.concatenate(arr_chunks)
    dt_s = arr_t_et_s[arr_row_indices] - dep_t_et_s[dep_row_indices]

    dt_mask = dt_s > float(dt_positive_tol_s)
    if np.isfinite(tof_min_s):
        dt_mask &= dt_s >= float(tof_min_s)
    if np.isfinite(tof_max_s):
        dt_mask &= dt_s <= float(tof_max_s)

    return dep_row_indices[dt_mask], arr_row_indices[dt_mask], dt_s[dt_mask]


def _iter_candidate_pair_chunks(
    dep_t_et_s: np.ndarray,
    arr_t_et_s: np.ndarray,
    tof_min_s: float,
    tof_max_s: float,
    dt_positive_tol_s: float,
    *,
    chunk_size: int,
):
    """Adapter over the two-phase candidate-pair API in `leg_database`.

    `leg_database` splits pair generation into a counting pass and a chunked
    emission pass; these tests exercise it as one call.
    """

    (
        dep_t_arr,
        arr_t_arr,
        lower_dt_s,
        _upper_dt_s,
        lower_dt_strict,
        _use_tof_max,
        counts,
    ) = leg_database._prepare_candidate_pair_counts(
        dep_t_et_s,
        arr_t_et_s,
        tof_min_s,
        tof_max_s,
        dt_positive_tol_s,
    )
    return leg_database._iter_candidate_pair_chunks_from_counts(
        dep_t_arr,
        arr_t_arr,
        lower_dt_s,
        lower_dt_strict,
        counts,
        chunk_size=chunk_size,
    )


def _generate_candidate_pairs(
    dep_t_et_s: np.ndarray,
    arr_t_et_s: np.ndarray,
    tof_min_s: float,
    tof_max_s: float,
    dt_positive_tol_s: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate every candidate pair in one unchunked batch."""

    return _flatten_candidate_pair_chunks(
        _iter_candidate_pair_chunks(
            dep_t_et_s,
            arr_t_et_s,
            tof_min_s,
            tof_max_s,
            dt_positive_tol_s,
            chunk_size=1 << 30,
        )
    )


def _flatten_candidate_pair_chunks(
    chunks,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    dep_chunks = []
    arr_chunks = []
    dt_chunks = []
    for dep_rows, arr_rows, dt_s in chunks:
        dep_chunks.append(np.asarray(dep_rows, dtype=np.int64).reshape(-1))
        arr_chunks.append(np.asarray(arr_rows, dtype=np.int64).reshape(-1))
        dt_chunks.append(np.asarray(dt_s, dtype=float).reshape(-1))

    if not dep_chunks:
        empty_i = np.empty(0, dtype=np.int64)
        return empty_i, empty_i, np.empty(0, dtype=float)

    return (
        np.concatenate(dep_chunks, axis=0),
        np.concatenate(arr_chunks, axis=0),
        np.concatenate(dt_chunks, axis=0),
    )


class LegDatabaseConstructionTest(unittest.TestCase):
    def _build_cache_fixture(self) -> tuple[EncounterDB, leg_database.LegBuildConfig]:
        dep_entries = [
            EncounterEntry(
                IE=0,
                stage_id=0,
                t_et=0.0,
                body=399,
                r_km=np.asarray([10.0, 0.0, 0.0], dtype=float),
                v_km_s=np.asarray([0.0, 1.0, 0.0], dtype=float),
                mu_km3_s2=1.0,
                rmin_km=1.0,
            ),
            EncounterEntry(
                IE=1,
                stage_id=0,
                t_et=10.0,
                body=399,
                r_km=np.asarray([20.0, 0.0, 0.0], dtype=float),
                v_km_s=np.asarray([0.0, 1.1, 0.0], dtype=float),
                mu_km3_s2=1.0,
                rmin_km=1.0,
            ),
        ]
        arr_entries = [
            EncounterEntry(
                IE=2,
                stage_id=1,
                t_et=100.0,
                body=399,
                r_km=np.asarray([0.0, 30.0, 0.0], dtype=float),
                v_km_s=np.asarray([-1.0, 0.0, 0.0], dtype=float),
                mu_km3_s2=1.0,
                rmin_km=1.0,
            ),
            EncounterEntry(
                IE=3,
                stage_id=1,
                t_et=110.0,
                body=399,
                r_km=np.asarray([0.0, 40.0, 0.0], dtype=float),
                v_km_s=np.asarray([-1.1, 0.0, 0.0], dtype=float),
                mu_km3_s2=1.0,
                rmin_km=1.0,
            ),
        ]
        entries = dep_entries + arr_entries
        encounter_db = EncounterDB(
            entries=entries,
            stage_to_entry_ids={(0, 399): [0, 1], (1, 399): [2, 3]},
            entry_by_id={entry.IE: entry for entry in entries},
        )
        cfg = leg_database.LegBuildConfig(
            leg_stage_id=0,
            dep_stage_id=0,
            arr_stage_id=1,
            tof_min_s=80.0,
            tof_max_s=120.0,
            vinfD_bounds_km_s=(0.0, 100.0),
            vinfA_bounds_km_s=(0.0, 100.0),
            lambert_mu_km3_s2=1.0,
            lambert_nrev_max=0,
            lambert_hz=1,
            dvlev_max_km_s=0.0,
            delta_dvlev_km_s=0.0,
            resonant_enabled=False,
            chunk_size=20000,
        )
        return encounter_db, cfg

    def _mock_lambert_result(
        self,
        encounter_db: EncounterDB,
        cfg: leg_database.LegBuildConfig,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        dep_data = leg_database._collect_stage_arrays(encounter_db, cfg.dep_stage_id)
        arr_data = leg_database._collect_stage_arrays(encounter_db, cfg.arr_stage_id)
        dep_rows, arr_rows, dt_candidates_s = _generate_candidate_pairs(
            dep_data["t_et"],
            arr_data["t_et"],
            cfg.tof_min_s,
            cfg.tof_max_s,
            cfg.dt_positive_tol_s,
        )
        source_idx = np.arange(dt_candidates_s.shape[0], dtype=np.int64)
        v_dep_km_s = dep_data["v_km_s"][dep_rows] + np.asarray([[0.2, 0.0, 0.0]], dtype=float)
        v_arr_km_s = arr_data["v_km_s"][arr_rows] + np.asarray([[0.0, 0.2, 0.0]], dtype=float)
        nrev_signed = np.zeros(dt_candidates_s.shape[0], dtype=np.int64)
        return v_dep_km_s, v_arr_km_s, nrev_signed, source_idx

    def _assert_leg_db_equal(self, actual: leg_database.LegDatabase, expected: leg_database.LegDatabase) -> None:
        self.assertEqual(actual.stage_id, expected.stage_id)
        for field_name in (
            "IL",
            "ID",
            "IA",
            "vinfD_km_s",
            "vinfA_km_s",
            "n_rev",
            "dv_lev_km_s",
            "eta_lev",
        ):
            actual_value = np.asarray(getattr(actual, field_name))
            expected_value = np.asarray(getattr(expected, field_name))
            self.assertEqual(actual_value.shape, expected_value.shape, field_name)
            if actual_value.dtype.kind in {"f", "c"} or expected_value.dtype.kind in {"f", "c"}:
                np.testing.assert_allclose(actual_value, expected_value, rtol=1e-12, atol=1e-12, err_msg=field_name)
            else:
                np.testing.assert_array_equal(actual_value, expected_value, err_msg=field_name)

    def _build_leg_db_reference_from_full_candidates(
        self,
        encounter_db: EncounterDB,
        cfg: leg_database.LegBuildConfig,
    ) -> leg_database.LegDatabase:
        tof_min_s = float(cfg.tof_min_s)
        tof_max_s = float(cfg.tof_max_s)
        vinfD_min_km_s, vinfD_max_km_s = map(float, cfg.vinfD_bounds_km_s)
        vinfA_min_km_s, vinfA_max_km_s = map(float, cfg.vinfA_bounds_km_s)
        dV_lev_grid_km_s = leg_database._build_dv_lev_grid_km_s(
            float(cfg.dvlev_max_km_s),
            float(cfg.delta_dvlev_km_s),
        )

        dep_data = leg_database._collect_stage_arrays(encounter_db, cfg.dep_stage_id)
        arr_data = leg_database._collect_stage_arrays(encounter_db, cfg.arr_stage_id)
        dep_rows, arr_rows, dt_candidates_s = _generate_candidate_pairs(
            dep_data["t_et"],
            arr_data["t_et"],
            tof_min_s,
            tof_max_s,
            float(cfg.dt_positive_tol_s),
        )

        if int(dt_candidates_s.size) == 0:
            return leg_database._empty_leg_db(cfg.leg_stage_id)

        out_parts = {
            "ID": [],
            "IA": [],
            "vinfD_km_s": [],
            "vinfA_km_s": [],
            "n_rev": [],
            "dv_lev_km_s": [],
            "eta_lev": [],
        }
        chunk_size = max(1, int(cfg.chunk_size))
        for start in range(0, int(dt_candidates_s.size), chunk_size):
            end = min(start + chunk_size, int(dt_candidates_s.size))
            dep_rows_chunk = dep_rows[start:end]
            arr_rows_chunk = arr_rows[start:end]
            dt_candidates_chunk_s = dt_candidates_s[start:end]

            r_dep_km = dep_data["r_km"][dep_rows_chunk]
            v_body_dep_km_s = dep_data["v_km_s"][dep_rows_chunk]
            r_arr_km = arr_data["r_km"][arr_rows_chunk]
            v_body_arr_km_s = arr_data["v_km_s"][arr_rows_chunk]

            v_dep_km_s, v_arr_km_s, nrev_signed_eval, source_idx = leg_database._solve_lambert_candidates(
                r_dep_km,
                r_arr_km,
                dt_candidates_chunk_s,
                cfg,
            )
            chunk_output = leg_database._process_lambert_chunk(
                dep_data=dep_data,
                arr_data=arr_data,
                dep_rows_chunk=dep_rows_chunk,
                arr_rows_chunk=arr_rows_chunk,
                dt_candidates_chunk_s=dt_candidates_chunk_s,
                r_dep_km=r_dep_km,
                r_arr_km=r_arr_km,
                v_body_dep_km_s=v_body_dep_km_s,
                v_body_arr_km_s=v_body_arr_km_s,
                v_dep_km_s=v_dep_km_s,
                v_arr_km_s=v_arr_km_s,
                nrev_signed_eval=nrev_signed_eval,
                source_idx=source_idx,
                cfg=cfg,
                dV_lev_grid_km_s=dV_lev_grid_km_s,
                vinfD_min_km_s=vinfD_min_km_s,
                vinfD_max_km_s=vinfD_max_km_s,
                vinfA_min_km_s=vinfA_min_km_s,
                vinfA_max_km_s=vinfA_max_km_s,
            )
            if chunk_output is None:
                continue
            for key, value in chunk_output.items():
                out_parts[key].append(value)

        if not out_parts["ID"]:
            return leg_database._empty_leg_db(cfg.leg_stage_id)

        kept_ID = np.concatenate(out_parts["ID"], axis=0)
        kept_IA = np.concatenate(out_parts["IA"], axis=0)
        kept_vinfD_km_s = np.concatenate(out_parts["vinfD_km_s"], axis=0)
        kept_vinfA_km_s = np.concatenate(out_parts["vinfA_km_s"], axis=0)
        n_rev = np.concatenate(out_parts["n_rev"], axis=0)
        kept_dv_lev_km_s = np.concatenate(out_parts["dv_lev_km_s"], axis=0)
        kept_eta_lev = np.concatenate(out_parts["eta_lev"], axis=0)

        if cfg.leg_filter is not None:
            dep_ie_order = np.argsort(dep_data["IE"])
            dep_ie_sorted = dep_data["IE"][dep_ie_order]
            dep_rows_for_kept = dep_ie_order[np.searchsorted(dep_ie_sorted, kept_ID)]

            arr_ie_order = np.argsort(arr_data["IE"])
            arr_ie_sorted = arr_data["IE"][arr_ie_order]
            arr_rows_for_kept = arr_ie_order[np.searchsorted(arr_ie_sorted, kept_IA)]

            custom_keep = cfg.leg_filter(
                cfg.leg_stage_id,
                kept_vinfD_km_s,
                kept_vinfA_km_s,
                dep_data["r_km"][dep_rows_for_kept],
                dep_data["v_km_s"][dep_rows_for_kept],
                arr_data["r_km"][arr_rows_for_kept],
                arr_data["v_km_s"][arr_rows_for_kept],
            )
            custom_keep = np.asarray(custom_keep, dtype=bool).ravel()
            kept_ID = kept_ID[custom_keep]
            kept_IA = kept_IA[custom_keep]
            kept_vinfD_km_s = kept_vinfD_km_s[custom_keep]
            kept_vinfA_km_s = kept_vinfA_km_s[custom_keep]
            n_rev = n_rev[custom_keep]
            kept_dv_lev_km_s = kept_dv_lev_km_s[custom_keep]
            kept_eta_lev = kept_eta_lev[custom_keep]

        if kept_ID.shape[0] == 0:
            return leg_database._empty_leg_db(cfg.leg_stage_id)

        return leg_database.LegDatabase(
            IL=np.arange(int(kept_ID.shape[0]), dtype=np.int64),
            stage_id=int(cfg.leg_stage_id),
            ID=kept_ID,
            IA=kept_IA,
            vinfD_km_s=kept_vinfD_km_s,
            vinfA_km_s=kept_vinfA_km_s,
            n_rev=n_rev,
            dv_lev_km_s=kept_dv_lev_km_s,
            eta_lev=kept_eta_lev,
        )

    def test_build_leg_db_stage_array_cache_reuses_stage_materialization(self) -> None:
        encounter_db, cfg = self._build_cache_fixture()
        lambert_result = self._mock_lambert_result(encounter_db, cfg)
        stage_arrays_cache = {}
        original_collect = leg_database._collect_stage_arrays

        with mock.patch.object(
            leg_database,
            "_solve_lambert_candidates",
            return_value=lambert_result,
        ), mock.patch.object(
            leg_database,
            "_collect_stage_arrays",
            side_effect=original_collect,
        ) as collect_mock:
            leg_database.build_leg_db(encounter_db, cfg, stage_arrays_cache=stage_arrays_cache)
            leg_database.build_leg_db(encounter_db, cfg, stage_arrays_cache=stage_arrays_cache)

        self.assertEqual(collect_mock.call_count, 2)
        self.assertEqual([int(call.args[1]) for call in collect_mock.call_args_list], [0, 1])

    def test_build_leg_db_stage_array_cache_clear_forces_rebuild(self) -> None:
        encounter_db, cfg = self._build_cache_fixture()
        lambert_result = self._mock_lambert_result(encounter_db, cfg)
        stage_arrays_cache = {}
        original_collect = leg_database._collect_stage_arrays

        with mock.patch.object(
            leg_database,
            "_solve_lambert_candidates",
            return_value=lambert_result,
        ), mock.patch.object(
            leg_database,
            "_collect_stage_arrays",
            side_effect=original_collect,
        ) as collect_mock:
            leg_database.build_leg_db(encounter_db, cfg, stage_arrays_cache=stage_arrays_cache)
            stage_arrays_cache.clear()
            leg_database.build_leg_db(encounter_db, cfg, stage_arrays_cache=stage_arrays_cache)

        self.assertEqual(collect_mock.call_count, 4)
        self.assertEqual([int(call.args[1]) for call in collect_mock.call_args_list], [0, 1, 0, 1])

    def test_build_leg_db_with_stage_array_cache_matches_uncached_output(self) -> None:
        encounter_db, cfg = self._build_cache_fixture()
        lambert_result = self._mock_lambert_result(encounter_db, cfg)

        with mock.patch.object(
            leg_database,
            "_solve_lambert_candidates",
            return_value=lambert_result,
        ):
            uncached_db = leg_database.build_leg_db(encounter_db, cfg)
            cached_db = leg_database.build_leg_db(encounter_db, cfg, stage_arrays_cache={})

        self._assert_leg_db_equal(cached_db, uncached_db)

    def test_generate_candidate_pairs_matches_legacy(self) -> None:
        dep_t_et_s = np.asarray([0.0, 1.0, 3.0, 10.0], dtype=float)
        arr_t_et_s = np.asarray([0.5, 1.5, 2.0, 4.0, 8.0, 13.0, 20.0], dtype=float)

        for tof_min_s, tof_max_s, dt_positive_tol_s in (
            (0.0, 5.0, 1e-6),
            (1.0, 10.0, 0.25),
            (-np.inf, np.inf, 0.5),
            (2.0, np.inf, 0.25),
        ):
            expected = _legacy_generate_candidate_pairs(
                dep_t_et_s,
                arr_t_et_s,
                tof_min_s,
                tof_max_s,
                dt_positive_tol_s,
            )
            actual = _generate_candidate_pairs(
                dep_t_et_s,
                arr_t_et_s,
                tof_min_s,
                tof_max_s,
                dt_positive_tol_s,
            )
            for actual_part, expected_part in zip(actual, expected):
                if np.asarray(actual_part).dtype.kind == "f":
                    np.testing.assert_allclose(actual_part, expected_part, rtol=0.0, atol=0.0)
                else:
                    np.testing.assert_array_equal(actual_part, expected_part)

    def test_iter_candidate_pair_chunks_matches_full_generation(self) -> None:
        dep_t_et_s = np.asarray([0.0, 1.0, 3.0, 10.0], dtype=float)
        arr_t_et_s = np.asarray([0.5, 1.5, 2.0, 4.0, 8.0, 13.0, 20.0], dtype=float)

        for tof_min_s, tof_max_s, dt_positive_tol_s in (
            (0.0, 5.0, 1e-6),
            (1.0, 10.0, 0.25),
            (-np.inf, np.inf, 0.5),
            (2.0, np.inf, 0.25),
            (50.0, 60.0, 1e-6),
        ):
            expected = _generate_candidate_pairs(
                dep_t_et_s,
                arr_t_et_s,
                tof_min_s,
                tof_max_s,
                dt_positive_tol_s,
            )
            for chunk_size in (1, 2, 3, 4):
                actual = _flatten_candidate_pair_chunks(
                    _iter_candidate_pair_chunks(
                        dep_t_et_s,
                        arr_t_et_s,
                        tof_min_s,
                        tof_max_s,
                        dt_positive_tol_s,
                        chunk_size=chunk_size,
                    )
                )
                for actual_part, expected_part in zip(actual, expected):
                    if np.asarray(actual_part).dtype.kind == "f":
                        np.testing.assert_allclose(actual_part, expected_part, rtol=0.0, atol=0.0)
                    else:
                        np.testing.assert_array_equal(actual_part, expected_part)

    def test_build_leg_db_matches_full_candidate_reference(self) -> None:
        encounter_db, cfg = self._build_cache_fixture()
        cfg = leg_database.LegBuildConfig(**{**vars(cfg), "chunk_size": 2})

        def _solve_lambert_side_effect(r1_km, r2_km, dt_s, _cfg):
            row_count = int(np.asarray(dt_s, dtype=float).size)
            return (
                np.full((row_count, 3), np.asarray([0.2, 0.0, 0.0], dtype=float), dtype=float),
                np.full((row_count, 3), np.asarray([0.0, 0.2, 0.0], dtype=float), dtype=float),
                np.zeros(row_count, dtype=np.int64),
                np.arange(row_count, dtype=np.int64),
            )

        with mock.patch.object(
            leg_database,
            "_solve_lambert_candidates",
            side_effect=_solve_lambert_side_effect,
        ):
            actual = leg_database.build_leg_db(encounter_db, cfg)
            expected = self._build_leg_db_reference_from_full_candidates(encounter_db, cfg)

        self._assert_leg_db_equal(actual, expected)

    def test_build_leg_db_matches_full_candidate_reference_with_resonant_rows(self) -> None:
        encounter_db, cfg = self._build_cache_fixture()
        cfg = leg_database.LegBuildConfig(
            **{
                **vars(cfg),
                "chunk_size": 2,
                "resonant_enabled": True,
                "resonant_dvinf_km_s": 0.5,
                "resonant_crank_angles_rad": (0.0,),
            }
        )

        def _solve_lambert_side_effect(r1_km, r2_km, dt_s, _cfg):
            row_count = int(np.asarray(dt_s, dtype=float).size)
            return (
                np.full((row_count, 3), np.asarray([0.2, 0.0, 0.0], dtype=float), dtype=float),
                np.full((row_count, 3), np.asarray([0.0, 0.2, 0.0], dtype=float), dtype=float),
                np.zeros(row_count, dtype=np.int64),
                np.arange(row_count, dtype=np.int64),
            )

        def _generate_resonant_rows_side_effect(
            *,
            dep_rows_seed,
            arr_rows_seed,
            dt_seed_s,
            **_kwargs,
        ):
            dep_rows_seed = np.asarray(dep_rows_seed, dtype=np.int64).reshape(-1)
            arr_rows_seed = np.asarray(arr_rows_seed, dtype=np.int64).reshape(-1)
            dt_seed_s = np.asarray(dt_seed_s, dtype=float).reshape(-1)
            keep = dep_rows_seed == 1
            row_count = int(np.count_nonzero(keep))
            if row_count == 0:
                empty_i = np.empty(0, dtype=np.int64)
                empty_f = np.empty(0, dtype=float)
                empty_v = np.empty((0, 3), dtype=float)
                return empty_i, empty_i, empty_f, empty_v, empty_v, empty_i

            return (
                dep_rows_seed[keep],
                arr_rows_seed[keep],
                dt_seed_s[keep] + 5.0,
                np.full((row_count, 3), np.asarray([1.5, 0.25, 0.0], dtype=float), dtype=float),
                np.full((row_count, 3), np.asarray([0.5, -0.5, 0.0], dtype=float), dtype=float),
                np.zeros(row_count, dtype=np.int64),
            )

        with mock.patch.object(
            leg_database,
            "_solve_lambert_candidates",
            side_effect=_solve_lambert_side_effect,
        ), mock.patch(
            "star.resonant.generate_resonant_rows",
            side_effect=_generate_resonant_rows_side_effect,
        ):
            actual = leg_database.build_leg_db(encounter_db, cfg)
            expected = self._build_leg_db_reference_from_full_candidates(encounter_db, cfg)

        self._assert_leg_db_equal(actual, expected)

    def test_build_leg_stage_to_disk_matches_build_leg_db(self) -> None:
        encounter_db, cfg = self._build_cache_fixture()
        cfg = leg_database.LegBuildConfig(**{**vars(cfg), "chunk_size": 2})

        def _solve_lambert_side_effect(r1_km, r2_km, dt_s, _cfg):
            row_count = int(np.asarray(dt_s, dtype=float).size)
            return (
                np.full((row_count, 3), np.asarray([0.2, 0.0, 0.0], dtype=float), dtype=float),
                np.full((row_count, 3), np.asarray([0.0, 0.2, 0.0], dtype=float), dtype=float),
                np.zeros(row_count, dtype=np.int64),
                np.arange(row_count, dtype=np.int64),
            )

        with tempfile.TemporaryDirectory() as tmpdir, mock.patch.object(
            leg_database,
            "_solve_lambert_candidates",
            side_effect=_solve_lambert_side_effect,
        ):
            expected = leg_database.build_leg_db(encounter_db, cfg)
            row_count = leg_database.build_leg_stage_to_disk(
                encounter_db,
                cfg,
                stage_dir=Path(tmpdir) / "leg_00",
            )
            actual = load_leg_stage(Path(tmpdir) / "leg_00")

        self.assertEqual(row_count, int(expected.IL.size))
        self._assert_leg_db_equal(actual, expected)




    def test_resonant_append_uses_stage_departure_position(self) -> None:
        dep_entries = [
            EncounterEntry(
                IE=0,
                stage_id=0,
                t_et=0.0,
                body=399,
                r_km=np.asarray([10.0, 0.0, 0.0], dtype=float),
                v_km_s=np.asarray([0.0, 1.0, 0.0], dtype=float),
                mu_km3_s2=1.0,
                rmin_km=1.0,
            ),
            EncounterEntry(
                IE=1,
                stage_id=0,
                t_et=10.0,
                body=399,
                r_km=np.asarray([20.0, 0.0, 0.0], dtype=float),
                v_km_s=np.asarray([0.0, 1.0, 0.0], dtype=float),
                mu_km3_s2=1.0,
                rmin_km=1.0,
            ),
        ]
        arr_entries = [
            EncounterEntry(
                IE=2,
                stage_id=1,
                t_et=100.0,
                body=399,
                r_km=np.asarray([0.0, 30.0, 0.0], dtype=float),
                v_km_s=np.asarray([-1.0, 0.0, 0.0], dtype=float),
                mu_km3_s2=1.0,
                rmin_km=1.0,
            ),
            EncounterEntry(
                IE=3,
                stage_id=1,
                t_et=110.0,
                body=399,
                r_km=np.asarray([0.0, 40.0, 0.0], dtype=float),
                v_km_s=np.asarray([-1.0, 0.0, 0.0], dtype=float),
                mu_km3_s2=1.0,
                rmin_km=1.0,
            ),
        ]
        entries = dep_entries + arr_entries
        encounter_db = EncounterDB(
            entries=entries,
            stage_to_entry_ids={(0, 399): [0, 1], (1, 399): [2, 3]},
            entry_by_id={entry.IE: entry for entry in entries},
        )

        cfg = leg_database.LegBuildConfig(
            leg_stage_id=0,
            dep_stage_id=0,
            arr_stage_id=1,
            tof_min_s=80.0,
            tof_max_s=120.0,
            vinfD_bounds_km_s=(0.0, 100.0),
            vinfA_bounds_km_s=(0.0, 100.0),
            lambert_mu_km3_s2=1.0,
            lambert_nrev_max=0,
            lambert_hz=1,
            dvlev_max_km_s=0.0,
            delta_dvlev_km_s=0.0,
            resonant_enabled=True,
            resonant_dvinf_km_s=0.5,
            resonant_crank_angles_rad=(0.0,),
        )

        dep_data = leg_database._collect_stage_arrays(encounter_db, 0)
        arr_data = leg_database._collect_stage_arrays(encounter_db, 1)
        dep_rows, arr_rows, dt_candidates_s = _generate_candidate_pairs(
            dep_data["t_et"],
            arr_data["t_et"],
            cfg.tof_min_s,
            cfg.tof_max_s,
            cfg.dt_positive_tol_s,
        )
        source_idx = np.arange(dt_candidates_s.shape[0], dtype=np.int64)
        v_dep_km_s = dep_data["v_km_s"][dep_rows] + np.asarray([[0.2, 0.0, 0.0]], dtype=float)
        v_arr_km_s = arr_data["v_km_s"][arr_rows] + np.asarray([[0.0, 0.2, 0.0]], dtype=float)
        nrev_signed = np.zeros(dt_candidates_s.shape[0], dtype=np.int64)

        dep_rows_res = np.asarray([1], dtype=np.int64)
        arr_rows_res = np.asarray([0], dtype=np.int64)
        dt_res_s = np.asarray([95.0], dtype=float)
        vinfD_res = np.asarray([[1.5, 0.25, 0.0]], dtype=float)
        vinfA_res = np.asarray([[0.5, -0.5, 0.0]], dtype=float)
        n_rev_res = np.asarray([0], dtype=np.int64)

        with mock.patch.object(
            leg_database,
            "_solve_lambert_candidates",
            return_value=(v_dep_km_s, v_arr_km_s, nrev_signed, source_idx),
        ), mock.patch(
            "star.resonant.generate_resonant_rows",
            return_value=(dep_rows_res, arr_rows_res, dt_res_s, vinfD_res, vinfA_res, n_rev_res),
        ):
            leg_db = leg_database.build_leg_db(encounter_db, cfg)

        vinf_dep_match = np.all(np.isclose(leg_db.vinfD_km_s, vinfD_res[0], rtol=1e-12, atol=1e-12), axis=1)
        vinf_arr_match = np.all(np.isclose(leg_db.vinfA_km_s, vinfA_res[0], rtol=1e-12, atol=1e-12), axis=1)
        resonant_rows = np.flatnonzero(vinf_dep_match & vinf_arr_match)
        self.assertEqual(resonant_rows.size, 1)
        resonant_row = int(resonant_rows[0])

        self.assertEqual(int(leg_db.ID[resonant_row]), int(dep_data["IE"][dep_rows_res[0]]))
        expected_v_body_dep = dep_data["v_km_s"][dep_rows_res[0]]
        expected_v_dep = vinfD_res[0] + expected_v_body_dep
        expected_h_leg = np.cross(dep_data["r_km"][dep_rows_res[0]], expected_v_dep)
        derived_h_leg = np.cross(
            dep_data["r_km"][dep_rows_res[0]],
            dep_data["v_km_s"][dep_rows_res[0]] + leg_db.vinfD_km_s[resonant_row],
        )
        np.testing.assert_allclose(derived_h_leg, expected_h_leg, rtol=1e-12, atol=1e-12)


if __name__ == "__main__":
    unittest.main()

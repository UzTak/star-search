import importlib
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = REPO_ROOT / "star"
for path in (str(REPO_ROOT), str(PACKAGE_ROOT)):
    if path not in sys.path:
        sys.path.insert(0, path)

from star import flyby_database
from star.encounter_database import EncounterDB, EncounterEntry
from star.leg_database import LegDatabase
from star.stage_db_npy import load_active_mask, save_flyby_stage, save_leg_stage


def _make_leg_db(
    *,
    stage_id: int,
    il: list[int],
    dep_ids: list[int],
    arr_ids: list[int],
    vinf_dep: list[list[float]],
    vinf_arr: list[list[float]],
) -> LegDatabase:
    il_array = np.asarray(il, dtype=np.int64)
    dep_array = np.asarray(dep_ids, dtype=np.int64)
    arr_array = np.asarray(arr_ids, dtype=np.int64)
    vinf_dep_array = np.asarray(vinf_dep, dtype=float).reshape(-1, 3)
    vinf_arr_array = np.asarray(vinf_arr, dtype=float).reshape(-1, 3)
    row_count = il_array.size
    zeros = np.zeros(row_count, dtype=float)
    return LegDatabase(
        IL=il_array,
        stage_id=int(stage_id),
        ID=dep_array,
        IA=arr_array,
        vinfD_km_s=vinf_dep_array,
        vinfA_km_s=vinf_arr_array,
        n_rev=np.zeros(row_count, dtype=np.int64),
        dv_lev_km_s=zeros.copy(),
        eta_lev=np.full(row_count, 0.5, dtype=float),
    )


class FlybyDatabaseFilterTest(unittest.TestCase):
    def _make_entry(self, *, ie: int, stage_id: int, body: int) -> EncounterEntry:
        return EncounterEntry(
            IE=int(ie),
            stage_id=int(stage_id),
            t_et=float(ie),
            body=int(body),
            r_km=np.asarray([float(ie), 0.0, 0.0], dtype=float),
            v_km_s=np.asarray([0.0, 0.0, 0.0], dtype=float),
            mu_km3_s2=1.0,
            rmin_km=1.0,
        )

    def _encounter_db_from_entries(self, entries: list[EncounterEntry]) -> EncounterDB:
        stage_to_entry_ids: dict[tuple[int, int], list[int]] = {}
        for entry in entries:
            key = (int(entry.stage_id), int(entry.body))
            stage_to_entry_ids.setdefault(key, []).append(int(entry.IE))
        return EncounterDB(
            entries=entries,
            stage_to_entry_ids=stage_to_entry_ids,
            entry_by_id={int(entry.IE): entry for entry in entries},
        )

    def _build_index_cache(
        self,
        leg_stage_dirs: dict[int, Path],
        flyby_stage_dirs: dict[int, Path],
    ) -> tuple[flyby_database.ActiveIndexCache, dict[int, dict[str, np.ndarray]]]:
        index_cache = flyby_database.create_active_index_cache()
        active_leg_ie_cache: dict[int, dict[str, np.ndarray]] = {}
        for stage_id, stage_dir in sorted(leg_stage_dirs.items()):
            flyby_database.add_leg_stage_to_active_index_cache(
                stage_id,
                stage_dir,
                index_cache,
                active_leg_ie_cache=active_leg_ie_cache,
            )
        for stage_id, stage_dir in sorted(flyby_stage_dirs.items()):
            flyby_database.add_flyby_stage_to_active_index_cache(stage_id, stage_dir, index_cache)
        return index_cache, active_leg_ie_cache

    def _build_encounter_db(self) -> EncounterDB:
        entries = [
            EncounterEntry(
                IE=0,
                stage_id=0,
                t_et=0.0,
                body=399,
                r_km=np.asarray([0.0, 0.0, 0.0], dtype=float),
                v_km_s=np.asarray([0.0, 0.0, 0.0], dtype=float),
                mu_km3_s2=1.0,
                rmin_km=1.0,
            ),
            EncounterEntry(
                IE=1,
                stage_id=1,
                t_et=1.0,
                body=199,
                r_km=np.asarray([1.0, 0.0, 0.0], dtype=float),
                v_km_s=np.asarray([0.0, 0.0, 0.0], dtype=float),
                mu_km3_s2=100.0,
                rmin_km=2500.0,
            ),
            EncounterEntry(
                IE=2,
                stage_id=2,
                t_et=2.0,
                body=199,
                r_km=np.asarray([0.0, 1.0, 0.0], dtype=float),
                v_km_s=np.asarray([0.0, 0.0, 0.0], dtype=float),
                mu_km3_s2=1.0,
                rmin_km=1.0,
            ),
        ]
        return EncounterDB(
            entries=entries,
            stage_to_entry_ids={(0, 399): [0], (1, 199): [1], (2, 199): [2]},
            entry_by_id={entry.IE: entry for entry in entries},
        )

    def test_build_flyby_db_applies_custom_filter(self) -> None:
        encounter_db = self._build_encounter_db()
        incoming_leg = _make_leg_db(
            stage_id=0,
            il=[10],
            dep_ids=[0],
            arr_ids=[1],
            vinf_dep=[[0.0, 0.0, 0.0]],
            vinf_arr=[[1.0, 0.0, 0.0]],
        )
        outgoing_leg = _make_leg_db(
            stage_id=1,
            il=[20, 21],
            dep_ids=[1, 1],
            arr_ids=[2, 2],
            vinf_dep=[[1.5, 0.0, 0.0], [0.5, 0.0, 0.0]],
            vinf_arr=[[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
        )

        def keep_nonincreasing_energy(
            stage_id,
            body_id,
            r_km,
            v_body_km_s,
            vinf_in_km_s,
            vinf_out_km_s,
            mu_central_km3_s2,
        ):
            self.assertEqual(int(stage_id), 1)
            self.assertTrue(np.all(np.asarray(body_id, dtype=np.int64) == 199))
            v_in = np.asarray(v_body_km_s, dtype=float) + np.asarray(vinf_in_km_s, dtype=float)
            v_out = np.asarray(v_body_km_s, dtype=float) + np.asarray(vinf_out_km_s, dtype=float)
            r_norm = np.linalg.norm(np.asarray(r_km, dtype=float), axis=1)
            eps_in = 0.5 * np.einsum("ij,ij->i", v_in, v_in) - float(mu_central_km3_s2) / r_norm
            eps_out = 0.5 * np.einsum("ij,ij->i", v_out, v_out) - float(mu_central_km3_s2) / r_norm
            return eps_out <= eps_in + 1e-12

        cfg = flyby_database.FlybyBuildConfig(
            numerical_eps=1e-12,
            central_mu_km3_s2=2.0,
            flyby_filter=keep_nonincreasing_energy,
        )
        flyby_db = flyby_database.build_flyby_db(
            encounter_db,
            {0: incoming_leg, 1: outgoing_leg},
            cfg,
        )

        np.testing.assert_array_equal(flyby_db.IE, np.asarray([1], dtype=np.int64))
        np.testing.assert_array_equal(flyby_db.IL_in, np.asarray([10], dtype=np.int64))
        np.testing.assert_array_equal(flyby_db.IL_out, np.asarray([21], dtype=np.int64))

    def test_build_flyby_db_applies_stage_dv_patch_limit(self) -> None:
        encounter_db = self._build_encounter_db()
        incoming_leg = _make_leg_db(
            stage_id=0,
            il=[10],
            dep_ids=[0],
            arr_ids=[1],
            vinf_dep=[[0.0, 0.0, 0.0]],
            vinf_arr=[[10.0, 0.0, 0.0]],
        )
        outgoing_leg = _make_leg_db(
            stage_id=1,
            il=[20, 21],
            dep_ids=[1, 1],
            arr_ids=[2, 2],
            vinf_dep=[[1.0, 0.0, 0.0], [12.0, 0.0, 0.0]],
            vinf_arr=[[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
        )

        cfg = flyby_database.FlybyBuildConfig(
            numerical_eps=1e-12,
            dv_patch_max_km_s_by_stage={1: 3.0},
        )
        flyby_db = flyby_database.build_flyby_db(
            encounter_db,
            {0: incoming_leg, 1: outgoing_leg},
            cfg,
        )

        np.testing.assert_array_equal(flyby_db.IE, np.asarray([1], dtype=np.int64))
        np.testing.assert_array_equal(flyby_db.IL_in, np.asarray([10], dtype=np.int64))
        np.testing.assert_array_equal(flyby_db.IL_out, np.asarray([21], dtype=np.int64))
        np.testing.assert_allclose(flyby_db.dv_km_s, np.asarray([2.0], dtype=float), atol=1e-12)

    def test_pretrig_dv_lower_bound_can_short_circuit_all_pairs(self) -> None:
        vinf_in_c = np.asarray([[10.0, 0.0, 0.0], [11.0, 0.0, 0.0]], dtype=float)
        vinf_out_mat = np.asarray([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=float)
        vin_c = np.linalg.norm(vinf_in_c, axis=1)
        vout = np.linalg.norm(vinf_out_mat, axis=1)
        base_valid = np.ones((2, 2), dtype=bool)

        with patch("flyby_database.np.arccos", side_effect=AssertionError("arccos should not be called")):
            ci_idx, o_idx, dv_sel = flyby_database._evaluate_flyby_chunk_pairs(
                vinf_in_c=vinf_in_c,
                vin_c=vin_c,
                vinf_out_mat=vinf_out_mat,
                vout=vout,
                base_valid_mask=base_valid,
                mu_km3_s2=100.0,
                rp_km=2500.0,
                dv_patch_limit_km_s=3.0,
                eps=1e-12,
                sparse_density_threshold=0.15,
            )

        self.assertEqual(int(ci_idx.size), 0)
        self.assertEqual(int(o_idx.size), 0)
        self.assertEqual(int(dv_sel.size), 0)

    def test_dv_lower_bound_feeds_sparse_dense_decision(self) -> None:
        vinf_in_c = np.asarray([[5.0, 0.0, 0.0], [20.0, 0.0, 0.0]], dtype=float)
        vinf_out_mat = np.asarray([[5.1, 0.0, 0.0], [40.0, 0.0, 0.0], [41.0, 0.0, 0.0]], dtype=float)
        vin_c = np.linalg.norm(vinf_in_c, axis=1)
        vout = np.linalg.norm(vinf_out_mat, axis=1)
        base_valid = np.ones((2, 3), dtype=bool)

        with patch("flyby_database.np.einsum", wraps=flyby_database.np.einsum) as einsum_mock:
            ci_idx, o_idx, dv_sel = flyby_database._evaluate_flyby_chunk_pairs(
                vinf_in_c=vinf_in_c,
                vin_c=vin_c,
                vinf_out_mat=vinf_out_mat,
                vout=vout,
                base_valid_mask=base_valid,
                mu_km3_s2=100.0,
                rp_km=2500.0,
                dv_patch_limit_km_s=0.2,
                eps=1e-12,
                sparse_density_threshold=0.2,
            )

        self.assertGreater(int(einsum_mock.call_count), 0)
        np.testing.assert_array_equal(ci_idx, np.asarray([0], dtype=np.int64))
        np.testing.assert_array_equal(o_idx, np.asarray([0], dtype=np.int64))
        np.testing.assert_allclose(dv_sel, np.asarray([0.1], dtype=float), atol=1e-12)

    def test_build_flyby_db_requires_central_mu_when_filter_is_enabled(self) -> None:
        encounter_db = self._build_encounter_db()
        incoming_leg = _make_leg_db(
            stage_id=0,
            il=[10],
            dep_ids=[0],
            arr_ids=[1],
            vinf_dep=[[0.0, 0.0, 0.0]],
            vinf_arr=[[1.0, 0.0, 0.0]],
        )
        outgoing_leg = _make_leg_db(
            stage_id=1,
            il=[20],
            dep_ids=[1],
            arr_ids=[2],
            vinf_dep=[[0.5, 0.0, 0.0]],
            vinf_arr=[[0.0, 0.0, 0.0]],
        )

        with self.assertRaisesRegex(ValueError, "central_mu_km3_s2 must be provided"):
            flyby_database.build_flyby_db(
                encounter_db,
                {0: incoming_leg, 1: outgoing_leg},
                flyby_database.FlybyBuildConfig(
                    numerical_eps=1e-12,
                    flyby_filter=lambda *args: True,
                ),
            )

    def test_bepicolombo_filter_keeps_only_nonincreasing_pairs_for_any_flyby_body(self) -> None:
        problem = importlib.import_module("example.bepicolombo")

        keep = problem.nonincreasing_energy(
            3,
            np.asarray([199, 199, 299], dtype=np.int64),
            np.asarray([[1.0, 0.0, 0.0]] * 3, dtype=float),
            np.asarray([[0.0, 0.0, 0.0]] * 3, dtype=float),
            np.asarray([[1.0, 0.0, 0.0]] * 3, dtype=float),
            np.asarray([[1.5, 0.0, 0.0], [0.5, 0.0, 0.0], [9.0, 0.0, 0.0]], dtype=float),
            2.0,
        )

        np.testing.assert_array_equal(keep, np.asarray([False, True, False], dtype=bool))

    def test_incoming_leg_filter_disk_filters_prior_leg_in_cached_and_uncached_modes(self) -> None:
        for use_cache in (False, True):
            with self.subTest(use_cache=use_cache):
                encounter_db = self._encounter_db_from_entries(
                    [
                        self._make_entry(ie=0, stage_id=0, body=399),
                        self._make_entry(ie=5, stage_id=0, body=399),
                        self._make_entry(ie=1, stage_id=1, body=299),
                        self._make_entry(ie=3, stage_id=1, body=299),
                        self._make_entry(ie=2, stage_id=2, body=499),
                    ]
                )
                leg_0 = _make_leg_db(
                    stage_id=0,
                    il=[10, 11],
                    dep_ids=[0, 5],
                    arr_ids=[1, 3],
                    vinf_dep=[[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
                    vinf_arr=[[1.0, 0.0, 0.0], [2.0, 0.0, 0.0]],
                )
                leg_1 = _make_leg_db(
                    stage_id=1,
                    il=[20],
                    dep_ids=[1],
                    arr_ids=[2],
                    vinf_dep=[[1.0, 0.0, 0.0]],
                    vinf_arr=[[0.0, 0.0, 0.0]],
                )
                flyby_1 = flyby_database.FlybyDB(
                    IF=np.asarray([100], dtype=np.int64),
                    stage_id=np.asarray([1], dtype=np.int64),
                    IE=np.asarray([1], dtype=np.int64),
                    IL_in=np.asarray([10], dtype=np.int64),
                    IL_out=np.asarray([20], dtype=np.int64),
                    dv_km_s=np.asarray([0.0], dtype=float),
                )

                with TemporaryDirectory() as tmpdir:
                    root = Path(tmpdir)
                    leg_stage_dirs = {
                        0: save_leg_stage(root / "leg_00", leg_0),
                        1: save_leg_stage(root / "leg_01", leg_1),
                    }
                    flyby_stage_dirs = {
                        1: save_flyby_stage(root / "flyby_01", flyby_1),
                    }

                    if use_cache:
                        index_cache, active_leg_ie_cache = self._build_index_cache(leg_stage_dirs, flyby_stage_dirs)
                    else:
                        index_cache = None
                        active_leg_ie_cache = None

                    removed = flyby_database.incoming_leg_filter_disk(
                        1,
                        leg_stage_dirs,
                        flyby_stage_dirs,
                        encounter_db,
                        active_leg_ie_cache=active_leg_ie_cache,
                        index_cache=index_cache,
                    )

                    if index_cache is not None:
                        flyby_database.flush_active_index_cache_masks(leg_stage_dirs, flyby_stage_dirs, index_cache)

                    self.assertEqual(int(removed), 3)
                    np.testing.assert_array_equal(load_active_mask(leg_stage_dirs[0]), np.asarray([True, False]))
                    self.assertEqual([int(entry.IE) for entry in encounter_db.entries], [0, 1, 2])

    def test_outgoing_leg_filter_disk_propagates_to_next_flyby_in_cached_and_uncached_modes(self) -> None:
        for use_cache in (False, True):
            with self.subTest(use_cache=use_cache):
                encounter_db = self._encounter_db_from_entries(
                    [
                        self._make_entry(ie=0, stage_id=0, body=399),
                        self._make_entry(ie=1, stage_id=1, body=299),
                        self._make_entry(ie=2, stage_id=2, body=499),
                        self._make_entry(ie=5, stage_id=2, body=499),
                        self._make_entry(ie=3, stage_id=3, body=599),
                        self._make_entry(ie=6, stage_id=3, body=599),
                    ]
                )
                leg_0 = _make_leg_db(
                    stage_id=0,
                    il=[10],
                    dep_ids=[0],
                    arr_ids=[1],
                    vinf_dep=[[0.0, 0.0, 0.0]],
                    vinf_arr=[[1.0, 0.0, 0.0]],
                )
                leg_1 = _make_leg_db(
                    stage_id=1,
                    il=[20, 21],
                    dep_ids=[1, 1],
                    arr_ids=[5, 2],
                    vinf_dep=[[1.0, 0.0, 0.0], [1.5, 0.0, 0.0]],
                    vinf_arr=[[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
                )
                leg_2 = _make_leg_db(
                    stage_id=2,
                    il=[30, 31],
                    dep_ids=[2, 5],
                    arr_ids=[3, 6],
                    vinf_dep=[[2.0, 0.0, 0.0], [3.0, 0.0, 0.0]],
                    vinf_arr=[[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
                )
                flyby_1 = flyby_database.FlybyDB(
                    IF=np.asarray([100], dtype=np.int64),
                    stage_id=np.asarray([1], dtype=np.int64),
                    IE=np.asarray([1], dtype=np.int64),
                    IL_in=np.asarray([10], dtype=np.int64),
                    IL_out=np.asarray([21], dtype=np.int64),
                    dv_km_s=np.asarray([0.0], dtype=float),
                )
                flyby_2 = flyby_database.FlybyDB(
                    IF=np.asarray([200, 201], dtype=np.int64),
                    stage_id=np.asarray([2, 2], dtype=np.int64),
                    IE=np.asarray([5, 2], dtype=np.int64),
                    IL_in=np.asarray([20, 21], dtype=np.int64),
                    IL_out=np.asarray([31, 30], dtype=np.int64),
                    dv_km_s=np.asarray([0.0, 0.0], dtype=float),
                )

                with TemporaryDirectory() as tmpdir:
                    root = Path(tmpdir)
                    leg_stage_dirs = {
                        0: save_leg_stage(root / "leg_00", leg_0),
                        1: save_leg_stage(root / "leg_01", leg_1),
                        2: save_leg_stage(root / "leg_02", leg_2),
                    }
                    flyby_stage_dirs = {
                        1: save_flyby_stage(root / "flyby_01", flyby_1),
                        2: save_flyby_stage(root / "flyby_02", flyby_2),
                    }

                    if use_cache:
                        index_cache, active_leg_ie_cache = self._build_index_cache(leg_stage_dirs, flyby_stage_dirs)
                    else:
                        index_cache = None
                        active_leg_ie_cache = None

                    removed = flyby_database.outgoing_leg_filter_disk(
                        1,
                        leg_stage_dirs,
                        flyby_stage_dirs,
                        encounter_db,
                        active_leg_ie_cache=active_leg_ie_cache,
                        index_cache=index_cache,
                    )

                    if index_cache is not None:
                        flyby_database.flush_active_index_cache_masks(leg_stage_dirs, flyby_stage_dirs, index_cache)

                    self.assertEqual(int(removed), 5)
                    np.testing.assert_array_equal(load_active_mask(leg_stage_dirs[1]), np.asarray([False, True]))
                    np.testing.assert_array_equal(load_active_mask(leg_stage_dirs[2]), np.asarray([True, False]))
                    np.testing.assert_array_equal(load_active_mask(flyby_stage_dirs[2]), np.asarray([False, True]))
                    self.assertEqual([int(entry.IE) for entry in encounter_db.entries], [0, 1, 2, 3])


if __name__ == "__main__":
    unittest.main()

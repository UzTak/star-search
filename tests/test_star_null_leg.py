import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = REPO_ROOT / "star"
for path in (str(REPO_ROOT), str(PACKAGE_ROOT)):
    if path not in sys.path:
        sys.path.insert(0, path)

from star import pipeline as star
from star.encounter_database import EncounterDB, EncounterEntry
from star.leg_database import LegDatabase
from star.stage_db_npy import (
    load_flyby_stage,
    load_leg_null_binding_sidecar,
    save_leg_null_binding_sidecar,
    save_leg_stage,
    update_active_mask,
)


class NullLegHelperTest(unittest.TestCase):
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

    def _make_encounter_db_for_n_legs(self, n_legs: int) -> EncounterDB:
        entries = [
            EncounterEntry(
                IE=stage_id,
                stage_id=stage_id,
                t_et=float(stage_id),
                body=399,
                r_km=np.asarray([float(stage_id), 0.0, 0.0], dtype=float),
                v_km_s=np.asarray([0.0, 1.0, 0.0], dtype=float),
                mu_km3_s2=1.0,
                rmin_km=1.0,
            )
            for stage_id in range(int(n_legs) + 1)
        ]
        return EncounterDB(
            entries=entries,
            stage_to_entry_ids={(int(entry.stage_id), int(entry.body)): [int(entry.IE)] for entry in entries},
            entry_by_id={int(entry.IE): entry for entry in entries},
        )

    def _make_problem_cfg(self, null_flags: list[bool], *, flyby_cfg: object | None = None) -> object:
        n_legs = len(null_flags)
        n_stages = n_legs + 1
        tof_min_days = np.zeros((n_stages, n_stages), dtype=float)
        tof_max_days = np.ones((n_stages, n_stages), dtype=float)
        return type(
            "Cfg",
            (),
            {
                "n_legs": int(n_legs),
                "n_stages": int(n_stages),
                "null_flags": list(null_flags),
                "tof_min_days": tof_min_days,
                "tof_max_days": tof_max_days,
                "lambert_nrev": [0] * n_legs,
                "dvlev_max_km_s": [0.0] * n_legs,
                "delta_dvlev_km_s": [0.0] * n_legs,
                "flyby_cfg": flyby_cfg if flyby_cfg is not None else star.FlybyBuildConfig(numerical_eps=1e-12),
                "leg_filter": None,
            },
        )()

    def _make_runner(
        self,
        *,
        null_flags: list[bool],
        unbuilt_legs: set[int],
        built_legs: dict[int, Path],
        build_order: dict[int, int],
        encounter_db: EncounterDB | None = None,
        cache_root: Path | None = None,
        flyby_cfg: object | None = None,
    ) -> star.Phase1Runner:
        problem_cfg = self._make_problem_cfg(null_flags, flyby_cfg=flyby_cfg)
        null_runs_by_start, null_run_start_by_leg = star._build_null_run_maps(problem_cfg.null_flags)
        state = star.Phase1State(
            encounter_db=self._make_encounter_db_for_n_legs(len(null_flags)) if encounter_db is None else encounter_db,
            unbuilt_legs=set(int(v) for v in unbuilt_legs),
            leg_stage_dirs={int(stage_id): Path(path) for stage_id, path in built_legs.items()},
            flyby_stage_dirs={},
            if_offset=0,
            null_stage_ie_maps={},
            null_runs_by_start=null_runs_by_start,
            null_run_start_by_leg=null_run_start_by_leg,
            leg_build_order={int(stage_id): int(order) for stage_id, order in build_order.items()},
            next_leg_build_order=max(build_order.values(), default=-1) + 1,
            stage_arrays_cache={},
            active_leg_ie_cache={},
            active_index_cache=star.create_active_index_cache(),
        )
        return star.Phase1Runner(
            problem_cfg=problem_cfg,
            cache_root=Path(".") if cache_root is None else Path(cache_root),
            state=state,
        )

    def test_resolve_null_defaults_to_no_null_legs(self) -> None:
        problem = type("Problem", (), {})()
        self.assertEqual(star._resolve_null_flags(problem, 3), [False, False, False])

    def test_build_null_stage_ie_maps_and_append_rows(self) -> None:
        entries = [
            EncounterEntry(
                IE=0,
                stage_id=0,
                t_et=5.0,
                body=299,
                r_km=np.asarray([1.0, 0.0, 0.0], dtype=float),
                v_km_s=np.asarray([0.0, 1.0, 0.0], dtype=float),
                mu_km3_s2=1.0,
                rmin_km=1.0,
            ),
            EncounterEntry(
                IE=1,
                stage_id=0,
                t_et=6.0,
                body=399,
                r_km=np.asarray([2.0, 0.0, 0.0], dtype=float),
                v_km_s=np.asarray([0.0, 1.0, 0.0], dtype=float),
                mu_km3_s2=1.0,
                rmin_km=1.0,
            ),
            EncounterEntry(
                IE=10,
                stage_id=1,
                t_et=5.0,
                body=299,
                r_km=np.asarray([3.0, 0.0, 0.0], dtype=float),
                v_km_s=np.asarray([0.0, 1.0, 0.0], dtype=float),
                mu_km3_s2=1.0,
                rmin_km=1.0,
            ),
            EncounterEntry(
                IE=11,
                stage_id=1,
                t_et=8.0,
                body=299,
                r_km=np.asarray([4.0, 0.0, 0.0], dtype=float),
                v_km_s=np.asarray([0.0, 1.0, 0.0], dtype=float),
                mu_km3_s2=1.0,
                rmin_km=1.0,
            ),
        ]
        encounter_db = EncounterDB(
            entries=entries,
            stage_to_entry_ids={(0, 299): [0], (0, 399): [1], (1, 299): [10, 11]},
            entry_by_id={entry.IE: entry for entry in entries},
        )

        maps = star._build_null_stage_ie_maps(encounter_db, [True])
        self.assertEqual(maps[0][0], {0: 10})
        self.assertEqual(maps[0][1], {10: 0})

        base_leg_db = LegDatabase(
            IL=np.asarray([0], dtype=np.int64),
            stage_id=0,
            ID=np.asarray([1], dtype=np.int64),
            IA=np.asarray([10], dtype=np.int64),
            vinfD_km_s=np.asarray([[2.0, 0.0, 0.0]], dtype=float),
            vinfA_km_s=np.asarray([[2.0, 0.0, 0.0]], dtype=float),
            n_rev=np.asarray([0], dtype=np.int64),
            dv_lev_km_s=np.asarray([0.1], dtype=float),
            eta_lev=np.asarray([0.5], dtype=float),
        )
        prev_leg_db = LegDatabase(
            IL=np.asarray([5], dtype=np.int64),
            stage_id=0,
            ID=np.asarray([99], dtype=np.int64),
            IA=np.asarray([0], dtype=np.int64),
            vinfD_km_s=np.asarray([[3.0, 0.0, 0.0]], dtype=float),
            vinfA_km_s=np.asarray([[1.0, 0.0, 0.0]], dtype=float),
            n_rev=np.asarray([0], dtype=np.int64),
            dv_lev_km_s=np.asarray([0.2], dtype=float),
            eta_lev=np.asarray([0.25], dtype=float),
        )

        post_rows, source_il = star._build_null_leg_append_rows(
            prev_leg_db,
            maps[0][0],
            use_previous=True,
        )
        self.assertIsNotNone(post_rows)
        np.testing.assert_array_equal(np.asarray(post_rows["ID"], dtype=np.int64), np.asarray([0], dtype=np.int64))
        np.testing.assert_array_equal(np.asarray(post_rows["IA"], dtype=np.int64), np.asarray([10], dtype=np.int64))
        np.testing.assert_array_equal(source_il, np.asarray([5], dtype=np.int64))
        np.testing.assert_allclose(np.asarray(post_rows["vinfD_km_s"], dtype=float), np.asarray([[1.0, 0.0, 0.0]], dtype=float))
        np.testing.assert_allclose(np.asarray(post_rows["vinfA_km_s"], dtype=float), np.asarray([[1.0, 0.0, 0.0]], dtype=float))
        np.testing.assert_allclose(np.asarray(post_rows["dv_lev_km_s"], dtype=float), np.asarray([0.0], dtype=float))
        np.testing.assert_allclose(np.asarray(post_rows["eta_lev"], dtype=float), np.asarray([0.0], dtype=float))

    def test_null_binding_sidecar_round_trip_and_active_only(self) -> None:
        with TemporaryDirectory() as tmpdir:
            stage_dir = save_leg_stage(
                Path(tmpdir) / "leg_01",
                LegDatabase(
                    IL=np.asarray([0, 1, 2, 3], dtype=np.int64),
                    stage_id=1,
                    ID=np.asarray([0, 0, 0, 0], dtype=np.int64),
                    IA=np.asarray([1, 1, 1, 1], dtype=np.int64),
                    vinfD_km_s=np.asarray([[1.0, 0.0, 0.0]] * 4, dtype=float),
                    vinfA_km_s=np.asarray([[1.0, 0.0, 0.0]] * 4, dtype=float),
                    n_rev=np.zeros(4, dtype=np.int64),
                    dv_lev_km_s=np.zeros(4, dtype=float),
                    eta_lev=np.zeros(4, dtype=float),
                ),
            )

            source_stage, copied_il, source_il = load_leg_null_binding_sidecar(stage_dir)
            self.assertIsNone(source_stage)
            self.assertEqual(int(copied_il.size), 0)
            self.assertEqual(int(source_il.size), 0)

            save_leg_null_binding_sidecar(
                stage_dir,
                source_stage_id=0,
                copied_il=np.asarray([1, 3], dtype=np.int64),
                source_il=np.asarray([10, 30], dtype=np.int64),
            )
            source_stage, copied_il, source_il = load_leg_null_binding_sidecar(stage_dir)
            self.assertEqual(source_stage, 0)
            np.testing.assert_array_equal(copied_il, np.asarray([1, 3], dtype=np.int64))
            np.testing.assert_array_equal(source_il, np.asarray([10, 30], dtype=np.int64))

            update_active_mask(stage_dir, np.asarray([True, False, True, True], dtype=bool))
            source_stage, copied_il, source_il = load_leg_null_binding_sidecar(stage_dir, active_only=True)
            self.assertEqual(source_stage, 0)
            np.testing.assert_array_equal(copied_il, np.asarray([3], dtype=np.int64))
            np.testing.assert_array_equal(source_il, np.asarray([30], dtype=np.int64))

    def test_consecutive_null_bindings_store_immediate_source_rows(self) -> None:
        source_leg = LegDatabase(
            IL=np.asarray([7], dtype=np.int64),
            stage_id=3,
            ID=np.asarray([50], dtype=np.int64),
            IA=np.asarray([100], dtype=np.int64),
            vinfD_km_s=np.asarray([[4.0, 0.0, 0.0]], dtype=float),
            vinfA_km_s=np.asarray([[1.0, 0.0, 0.0]], dtype=float),
            n_rev=np.zeros(1, dtype=np.int64),
            dv_lev_km_s=np.zeros(1, dtype=float),
            eta_lev=np.zeros(1, dtype=float),
        )
        base_leg_4 = LegDatabase(
            IL=np.asarray([0], dtype=np.int64),
            stage_id=4,
            ID=np.asarray([1], dtype=np.int64),
            IA=np.asarray([2], dtype=np.int64),
            vinfD_km_s=np.asarray([[9.0, 0.0, 0.0]], dtype=float),
            vinfA_km_s=np.asarray([[9.0, 0.0, 0.0]], dtype=float),
            n_rev=np.zeros(1, dtype=np.int64),
            dv_lev_km_s=np.zeros(1, dtype=float),
            eta_lev=np.zeros(1, dtype=float),
        )
        post_rows_4, source_4 = star._build_null_leg_append_rows(source_leg, {100: 200}, use_previous=True)
        self.assertIsNotNone(post_rows_4)
        np.testing.assert_array_equal(source_4, np.asarray([7], dtype=np.int64))
        leg_4 = LegDatabase(
            IL=np.asarray([0, 1], dtype=np.int64),
            stage_id=4,
            ID=np.concatenate((base_leg_4.ID, np.asarray(post_rows_4["ID"], dtype=np.int64)), axis=0),
            IA=np.concatenate((base_leg_4.IA, np.asarray(post_rows_4["IA"], dtype=np.int64)), axis=0),
            vinfD_km_s=np.concatenate((base_leg_4.vinfD_km_s, np.asarray(post_rows_4["vinfD_km_s"], dtype=float)), axis=0),
            vinfA_km_s=np.concatenate((base_leg_4.vinfA_km_s, np.asarray(post_rows_4["vinfA_km_s"], dtype=float)), axis=0),
            n_rev=np.concatenate((base_leg_4.n_rev, np.asarray(post_rows_4["n_rev"], dtype=np.int64)), axis=0),
            dv_lev_km_s=np.concatenate((base_leg_4.dv_lev_km_s, np.asarray(post_rows_4["dv_lev_km_s"], dtype=float)), axis=0),
            eta_lev=np.concatenate((base_leg_4.eta_lev, np.asarray(post_rows_4["eta_lev"], dtype=float)), axis=0),
        )

        base_leg_5 = LegDatabase(
            IL=np.asarray([0], dtype=np.int64),
            stage_id=5,
            ID=np.asarray([3], dtype=np.int64),
            IA=np.asarray([4], dtype=np.int64),
            vinfD_km_s=np.asarray([[8.0, 0.0, 0.0]], dtype=float),
            vinfA_km_s=np.asarray([[8.0, 0.0, 0.0]], dtype=float),
            n_rev=np.zeros(1, dtype=np.int64),
            dv_lev_km_s=np.zeros(1, dtype=float),
            eta_lev=np.zeros(1, dtype=float),
        )
        post_rows_5, source_5 = star._build_null_leg_append_rows(leg_4, {200: 300}, use_previous=True)
        self.assertIsNotNone(post_rows_5)
        np.testing.assert_array_equal(source_5, np.asarray([1], dtype=np.int64))
        leg_5 = LegDatabase(
            IL=np.asarray([0, 1], dtype=np.int64),
            stage_id=5,
            ID=np.concatenate((base_leg_5.ID, np.asarray(post_rows_5["ID"], dtype=np.int64)), axis=0),
            IA=np.concatenate((base_leg_5.IA, np.asarray(post_rows_5["IA"], dtype=np.int64)), axis=0),
            vinfD_km_s=np.concatenate((base_leg_5.vinfD_km_s, np.asarray(post_rows_5["vinfD_km_s"], dtype=float)), axis=0),
            vinfA_km_s=np.concatenate((base_leg_5.vinfA_km_s, np.asarray(post_rows_5["vinfA_km_s"], dtype=float)), axis=0),
            n_rev=np.concatenate((base_leg_5.n_rev, np.asarray(post_rows_5["n_rev"], dtype=np.int64)), axis=0),
            dv_lev_km_s=np.concatenate((base_leg_5.dv_lev_km_s, np.asarray(post_rows_5["dv_lev_km_s"], dtype=float)), axis=0),
            eta_lev=np.concatenate((base_leg_5.eta_lev, np.asarray(post_rows_5["eta_lev"], dtype=float)), axis=0),
        )

        base_leg_6 = LegDatabase(
            IL=np.asarray([0], dtype=np.int64),
            stage_id=6,
            ID=np.asarray([5], dtype=np.int64),
            IA=np.asarray([6], dtype=np.int64),
            vinfD_km_s=np.asarray([[7.0, 0.0, 0.0]], dtype=float),
            vinfA_km_s=np.asarray([[7.0, 0.0, 0.0]], dtype=float),
            n_rev=np.zeros(1, dtype=np.int64),
            dv_lev_km_s=np.zeros(1, dtype=float),
            eta_lev=np.zeros(1, dtype=float),
        )
        post_rows_6, source_6 = star._build_null_leg_append_rows(leg_5, {300: 400}, use_previous=True)
        self.assertIsNotNone(post_rows_6)
        np.testing.assert_array_equal(source_6, np.asarray([1], dtype=np.int64))

    def test_try_build_flyby_binds_outgoing_copied_rows_before_generic_search(self) -> None:
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
                body=299,
                r_km=np.asarray([2.0, 0.0, 0.0], dtype=float),
                v_km_s=np.asarray([0.0, 0.0, 0.0], dtype=float),
                mu_km3_s2=1.0,
                rmin_km=1.0,
            ),
        ]
        encounter_db = self._encounter_db_from_entries(entries)
        incoming_leg = LegDatabase(
            IL=np.asarray([10, 11], dtype=np.int64),
            stage_id=0,
            ID=np.asarray([0, 0], dtype=np.int64),
            IA=np.asarray([1, 1], dtype=np.int64),
            vinfD_km_s=np.asarray([[0.0, 0.0, 0.0]] * 2, dtype=float),
            vinfA_km_s=np.asarray([[1.0, 0.0, 0.0], [2.0, 0.0, 0.0]], dtype=float),
            n_rev=np.zeros(2, dtype=np.int64),
            dv_lev_km_s=np.zeros(2, dtype=float),
            eta_lev=np.zeros(2, dtype=float),
        )
        outgoing_leg = LegDatabase(
            IL=np.asarray([20, 21], dtype=np.int64),
            stage_id=1,
            ID=np.asarray([1, 1], dtype=np.int64),
            IA=np.asarray([2, 2], dtype=np.int64),
            vinfD_km_s=np.asarray([[1.0, 0.0, 0.0], [2.0, 0.0, 0.0]], dtype=float),
            vinfA_km_s=np.asarray([[0.0, 0.0, 0.0]] * 2, dtype=float),
            n_rev=np.zeros(2, dtype=np.int64),
            dv_lev_km_s=np.zeros(2, dtype=float),
            eta_lev=np.zeros(2, dtype=float),
        )

        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            incoming_stage = save_leg_stage(root / "leg_00", incoming_leg)
            outgoing_stage = save_leg_stage(root / "leg_01", outgoing_leg)
            save_leg_null_binding_sidecar(
                outgoing_stage,
                source_stage_id=0,
                copied_il=np.asarray([20], dtype=np.int64),
                source_il=np.asarray([10], dtype=np.int64),
            )

            runner = self._make_runner(
                null_flags=[False, False],
                unbuilt_legs=set(),
                built_legs={0: incoming_stage, 1: outgoing_stage},
                build_order={0: 0, 1: 1},
                encounter_db=encounter_db,
                cache_root=root,
                flyby_cfg=star.FlybyBuildConfig(
                    numerical_eps=1e-12,
                    dv_patch_max_km_s_by_stage={1: 0.1},
                ),
            )

            self.assertTrue(runner.try_build_flyby(1))
            flyby_db = load_flyby_stage(runner.state.flyby_stage_dirs[1])
            np.testing.assert_array_equal(flyby_db.IE, np.asarray([1, 1], dtype=np.int64))
            np.testing.assert_array_equal(flyby_db.IL_in, np.asarray([10, 11], dtype=np.int64))
            np.testing.assert_array_equal(flyby_db.IL_out, np.asarray([20, 21], dtype=np.int64))
            np.testing.assert_allclose(flyby_db.dv_km_s, np.asarray([0.0, 0.0], dtype=float), atol=1e-12)

    def test_try_build_flyby_binds_incoming_copied_rows_before_generic_search(self) -> None:
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
                body=299,
                r_km=np.asarray([2.0, 0.0, 0.0], dtype=float),
                v_km_s=np.asarray([0.0, 0.0, 0.0], dtype=float),
                mu_km3_s2=1.0,
                rmin_km=1.0,
            ),
        ]
        encounter_db = self._encounter_db_from_entries(entries)
        incoming_leg = LegDatabase(
            IL=np.asarray([10, 11], dtype=np.int64),
            stage_id=0,
            ID=np.asarray([0, 0], dtype=np.int64),
            IA=np.asarray([1, 1], dtype=np.int64),
            vinfD_km_s=np.asarray([[0.0, 0.0, 0.0]] * 2, dtype=float),
            vinfA_km_s=np.asarray([[1.0, 0.0, 0.0], [2.0, 0.0, 0.0]], dtype=float),
            n_rev=np.zeros(2, dtype=np.int64),
            dv_lev_km_s=np.zeros(2, dtype=float),
            eta_lev=np.zeros(2, dtype=float),
        )
        outgoing_leg = LegDatabase(
            IL=np.asarray([20, 21], dtype=np.int64),
            stage_id=1,
            ID=np.asarray([1, 1], dtype=np.int64),
            IA=np.asarray([2, 2], dtype=np.int64),
            vinfD_km_s=np.asarray([[1.0, 0.0, 0.0], [2.0, 0.0, 0.0]], dtype=float),
            vinfA_km_s=np.asarray([[0.0, 0.0, 0.0]] * 2, dtype=float),
            n_rev=np.zeros(2, dtype=np.int64),
            dv_lev_km_s=np.zeros(2, dtype=float),
            eta_lev=np.zeros(2, dtype=float),
        )

        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            incoming_stage = save_leg_stage(root / "leg_00", incoming_leg)
            outgoing_stage = save_leg_stage(root / "leg_01", outgoing_leg)
            save_leg_null_binding_sidecar(
                incoming_stage,
                source_stage_id=1,
                copied_il=np.asarray([10], dtype=np.int64),
                source_il=np.asarray([20], dtype=np.int64),
            )

            runner = self._make_runner(
                null_flags=[False, False],
                unbuilt_legs=set(),
                built_legs={0: incoming_stage, 1: outgoing_stage},
                build_order={0: 0, 1: 1},
                encounter_db=encounter_db,
                cache_root=root,
                flyby_cfg=star.FlybyBuildConfig(
                    numerical_eps=1e-12,
                    dv_patch_max_km_s_by_stage={1: 0.1},
                ),
            )

            self.assertTrue(runner.try_build_flyby(1))
            flyby_db = load_flyby_stage(runner.state.flyby_stage_dirs[1])
            np.testing.assert_array_equal(flyby_db.IE, np.asarray([1, 1], dtype=np.int64))
            np.testing.assert_array_equal(flyby_db.IL_in, np.asarray([10, 11], dtype=np.int64))
            np.testing.assert_array_equal(flyby_db.IL_out, np.asarray([20, 21], dtype=np.int64))
            np.testing.assert_allclose(flyby_db.dv_km_s, np.asarray([0.0, 0.0], dtype=float), atol=1e-12)

    def test_try_build_flyby_skips_copied_row_when_source_il_is_inactive(self) -> None:
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
                body=299,
                r_km=np.asarray([2.0, 0.0, 0.0], dtype=float),
                v_km_s=np.asarray([0.0, 0.0, 0.0], dtype=float),
                mu_km3_s2=1.0,
                rmin_km=1.0,
            ),
        ]
        encounter_db = self._encounter_db_from_entries(entries)
        incoming_leg = LegDatabase(
            IL=np.asarray([10, 11], dtype=np.int64),
            stage_id=0,
            ID=np.asarray([0, 0], dtype=np.int64),
            IA=np.asarray([1, 1], dtype=np.int64),
            vinfD_km_s=np.asarray([[0.0, 0.0, 0.0]] * 2, dtype=float),
            vinfA_km_s=np.asarray([[1.0, 0.0, 0.0], [2.0, 0.0, 0.0]], dtype=float),
            n_rev=np.zeros(2, dtype=np.int64),
            dv_lev_km_s=np.zeros(2, dtype=float),
            eta_lev=np.zeros(2, dtype=float),
        )
        outgoing_leg = LegDatabase(
            IL=np.asarray([20, 21], dtype=np.int64),
            stage_id=1,
            ID=np.asarray([1, 1], dtype=np.int64),
            IA=np.asarray([2, 2], dtype=np.int64),
            vinfD_km_s=np.asarray([[1.0, 0.0, 0.0], [2.0, 0.0, 0.0]], dtype=float),
            vinfA_km_s=np.asarray([[0.0, 0.0, 0.0]] * 2, dtype=float),
            n_rev=np.zeros(2, dtype=np.int64),
            dv_lev_km_s=np.zeros(2, dtype=float),
            eta_lev=np.zeros(2, dtype=float),
        )

        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            incoming_stage = save_leg_stage(root / "leg_00", incoming_leg)
            outgoing_stage = save_leg_stage(root / "leg_01", outgoing_leg)
            save_leg_null_binding_sidecar(
                outgoing_stage,
                source_stage_id=0,
                copied_il=np.asarray([20], dtype=np.int64),
                source_il=np.asarray([10], dtype=np.int64),
            )
            update_active_mask(incoming_stage, np.asarray([False, True], dtype=bool))

            runner = self._make_runner(
                null_flags=[False, False],
                unbuilt_legs=set(),
                built_legs={0: incoming_stage, 1: outgoing_stage},
                build_order={0: 0, 1: 1},
                encounter_db=encounter_db,
                cache_root=root,
                flyby_cfg=star.FlybyBuildConfig(
                    numerical_eps=1e-12,
                    dv_patch_max_km_s_by_stage={1: 0.1},
                ),
            )

            self.assertTrue(runner.try_build_flyby(1))
            flyby_db = load_flyby_stage(runner.state.flyby_stage_dirs[1])
            np.testing.assert_array_equal(flyby_db.IL_in, np.asarray([11], dtype=np.int64))
            np.testing.assert_array_equal(flyby_db.IL_out, np.asarray([21], dtype=np.int64))

    def test_try_build_flyby_raises_on_corrupt_sidecar_ie_mismatch(self) -> None:
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
                IE=99,
                stage_id=1,
                t_et=9.9,
                body=199,
                r_km=np.asarray([9.9, 0.0, 0.0], dtype=float),
                v_km_s=np.asarray([0.0, 0.0, 0.0], dtype=float),
                mu_km3_s2=100.0,
                rmin_km=2500.0,
            ),
            EncounterEntry(
                IE=2,
                stage_id=2,
                t_et=2.0,
                body=299,
                r_km=np.asarray([2.0, 0.0, 0.0], dtype=float),
                v_km_s=np.asarray([0.0, 0.0, 0.0], dtype=float),
                mu_km3_s2=1.0,
                rmin_km=1.0,
            ),
        ]
        encounter_db = self._encounter_db_from_entries(entries)
        incoming_leg = LegDatabase(
            IL=np.asarray([10, 11], dtype=np.int64),
            stage_id=0,
            ID=np.asarray([0, 0], dtype=np.int64),
            IA=np.asarray([1, 99], dtype=np.int64),
            vinfD_km_s=np.asarray([[0.0, 0.0, 0.0]] * 2, dtype=float),
            vinfA_km_s=np.asarray([[1.0, 0.0, 0.0], [2.0, 0.0, 0.0]], dtype=float),
            n_rev=np.zeros(2, dtype=np.int64),
            dv_lev_km_s=np.zeros(2, dtype=float),
            eta_lev=np.zeros(2, dtype=float),
        )
        outgoing_leg = LegDatabase(
            IL=np.asarray([20], dtype=np.int64),
            stage_id=1,
            ID=np.asarray([1], dtype=np.int64),
            IA=np.asarray([2], dtype=np.int64),
            vinfD_km_s=np.asarray([[1.0, 0.0, 0.0]], dtype=float),
            vinfA_km_s=np.asarray([[0.0, 0.0, 0.0]], dtype=float),
            n_rev=np.zeros(1, dtype=np.int64),
            dv_lev_km_s=np.zeros(1, dtype=float),
            eta_lev=np.zeros(1, dtype=float),
        )

        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            incoming_stage = save_leg_stage(root / "leg_00", incoming_leg)
            outgoing_stage = save_leg_stage(root / "leg_01", outgoing_leg)
            save_leg_null_binding_sidecar(
                outgoing_stage,
                source_stage_id=0,
                copied_il=np.asarray([20], dtype=np.int64),
                source_il=np.asarray([11], dtype=np.int64),
            )

            runner = self._make_runner(
                null_flags=[False, False],
                unbuilt_legs=set(),
                built_legs={0: incoming_stage, 1: outgoing_stage},
                build_order={0: 0, 1: 1},
                encounter_db=encounter_db,
                cache_root=root,
            )

            with self.assertRaisesRegex(ValueError, "Corrupt null-binding sidecar"):
                runner.try_build_flyby(1)

    def test_isolated_null_leg_anchors_from_first_real_boundary(self) -> None:
        left_runner = self._make_runner(
            null_flags=[False, True, False],
            unbuilt_legs={1},
            built_legs={0: Path("leg_00")},
            build_order={0: 0},
        )
        self.assertEqual(left_runner._resolve_null_run_direction(1), "left")
        self.assertEqual(left_runner.select_next_leg(), 1)

        right_runner = self._make_runner(
            null_flags=[False, True, False],
            unbuilt_legs={1},
            built_legs={2: Path("leg_02")},
            build_order={2: 0},
        )
        self.assertEqual(right_runner._resolve_null_run_direction(1), "right")
        self.assertEqual(right_runner.select_next_leg(), 1)

    def test_consecutive_null_run_left_anchor_builds_outward(self) -> None:
        runner = self._make_runner(
            null_flags=[False, False, False, False, True, True, False],
            unbuilt_legs={4, 5},
            built_legs={3: Path("leg_03")},
            build_order={3: 0},
        )

        self.assertEqual(runner._resolve_null_run_direction(4), "left")
        self.assertEqual(runner.select_next_leg(), 4)

        runner.state.unbuilt_legs.remove(4)
        runner.state.leg_stage_dirs[4] = Path("leg_04")
        runner.state.leg_build_order[4] = 1
        self.assertEqual(runner.select_next_leg(), 5)

    def test_consecutive_null_run_right_anchor_builds_outward(self) -> None:
        runner = self._make_runner(
            null_flags=[False, False, False, False, True, True, False],
            unbuilt_legs={4, 5},
            built_legs={6: Path("leg_06")},
            build_order={6: 0},
        )

        self.assertEqual(runner._resolve_null_run_direction(5), "right")
        self.assertEqual(runner.select_next_leg(), 5)

        runner.state.unbuilt_legs.remove(5)
        runner.state.leg_stage_dirs[5] = Path("leg_05")
        runner.state.leg_build_order[5] = 1
        self.assertEqual(runner.select_next_leg(), 4)

    def test_frozen_run_direction_ignores_opposite_boundary_later(self) -> None:
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            left_leg = LegDatabase(
                IL=np.asarray([30], dtype=np.int64),
                stage_id=3,
                ID=np.asarray([0], dtype=np.int64),
                IA=np.asarray([1], dtype=np.int64),
                vinfD_km_s=np.asarray([[1.0, 0.0, 0.0]], dtype=float),
                vinfA_km_s=np.asarray([[1.0, 0.0, 0.0]], dtype=float),
                n_rev=np.asarray([0], dtype=np.int64),
                dv_lev_km_s=np.asarray([0.0], dtype=float),
                eta_lev=np.asarray([0.0], dtype=float),
            )
            right_leg = LegDatabase(
                IL=np.asarray([60], dtype=np.int64),
                stage_id=6,
                ID=np.asarray([1], dtype=np.int64),
                IA=np.asarray([2], dtype=np.int64),
                vinfD_km_s=np.asarray([[2.0, 0.0, 0.0]], dtype=float),
                vinfA_km_s=np.asarray([[2.0, 0.0, 0.0]], dtype=float),
                n_rev=np.asarray([0], dtype=np.int64),
                dv_lev_km_s=np.asarray([0.0], dtype=float),
                eta_lev=np.asarray([0.0], dtype=float),
            )
            left_stage = save_leg_stage(root / "leg_03", left_leg)
            right_stage = save_leg_stage(root / "leg_06", right_leg)

            runner = self._make_runner(
                null_flags=[False, False, False, False, True, True, False],
                unbuilt_legs={4, 5},
                built_legs={3: left_stage},
                build_order={3: 0},
            )

            self.assertEqual(runner._resolve_null_run_direction(4), "left")
            runner.state.leg_stage_dirs[6] = right_stage
            runner.state.leg_build_order[6] = 1

            source_leg, use_previous = runner._load_null_source_leg(4)
            self.assertIsNotNone(source_leg)
            self.assertTrue(use_previous)
            self.assertEqual(int(source_leg.stage_id), 3)


if __name__ == "__main__":
    unittest.main()

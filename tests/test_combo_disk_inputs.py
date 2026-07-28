import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = REPO_ROOT / "star"
for path in (str(REPO_ROOT), str(PACKAGE_ROOT)):
    if path not in sys.path:
        sys.path.insert(0, path)

from star import combo
from star.encounter_database import EncounterDB, EncounterEntry
from star.flyby_database import FlybyDB
from star.leg_database import LegDatabase
from star.stage_db_npy import save_flyby_stage, save_leg_stage


class RunComboDiskFlybyStageTest(unittest.TestCase):
    def _build_encounter_db(self) -> EncounterDB:
        entries = [
            EncounterEntry(
                IE=0,
                stage_id=0,
                t_et=0.0,
                body=399,
                r_km=np.asarray([1.0, 0.0, 0.0], dtype=float),
                v_km_s=np.asarray([0.0, 1.0, 0.0], dtype=float),
                mu_km3_s2=1.0,
                rmin_km=1.0,
            ),
            EncounterEntry(
                IE=1,
                stage_id=1,
                t_et=5.0,
                body=299,
                r_km=np.asarray([0.0, 1.0, 0.0], dtype=float),
                v_km_s=np.asarray([-1.0, 0.0, 0.0], dtype=float),
                mu_km3_s2=1.0,
                rmin_km=1.0,
            ),
            EncounterEntry(
                IE=2,
                stage_id=2,
                t_et=12.0,
                body=199,
                r_km=np.asarray([0.0, 0.0, 1.0], dtype=float),
                v_km_s=np.asarray([0.0, -1.0, 0.0], dtype=float),
                mu_km3_s2=1.0,
                rmin_km=1.0,
            ),
        ]
        return EncounterDB(
            entries=entries,
            stage_to_entry_ids={(0, 399): [0], (1, 299): [1], (2, 199): [2]},
            entry_by_id={entry.IE: entry for entry in entries},
        )

    def _build_leg_dbs(self) -> dict[int, LegDatabase]:
        return {
            0: LegDatabase(
                IL=np.asarray([10], dtype=np.int64),
                stage_id=0,
                ID=np.asarray([0], dtype=np.int64),
                IA=np.asarray([1], dtype=np.int64),
                vinfD_km_s=np.zeros((1, 3), dtype=float),
                vinfA_km_s=np.zeros((1, 3), dtype=float),
                n_rev=np.asarray([0], dtype=np.int64),
                dv_lev_km_s=np.asarray([0.2], dtype=float),
                eta_lev=np.asarray([0.5], dtype=float),
            ),
            1: LegDatabase(
                IL=np.asarray([20], dtype=np.int64),
                stage_id=1,
                ID=np.asarray([1], dtype=np.int64),
                IA=np.asarray([2], dtype=np.int64),
                vinfD_km_s=np.zeros((1, 3), dtype=float),
                vinfA_km_s=np.zeros((1, 3), dtype=float),
                n_rev=np.asarray([0], dtype=np.int64),
                dv_lev_km_s=np.asarray([0.3], dtype=float),
                eta_lev=np.asarray([0.5], dtype=float),
            ),
        }

    def test_run_combo_accepts_disk_backed_flyby_stages(self) -> None:
        encounter_db = self._build_encounter_db()
        leg_dbs = self._build_leg_dbs()
        flyby_db = FlybyDB(
            IF=np.asarray([100], dtype=np.int64),
            stage_id=np.asarray([1], dtype=np.int64),
            IE=np.asarray([1], dtype=np.int64),
            IL_in=np.asarray([10], dtype=np.int64),
            IL_out=np.asarray([20], dtype=np.int64),
            dv_km_s=np.asarray([0.4], dtype=float),
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            stage_dir = Path(tmpdir) / "flyby_01"
            save_flyby_stage(stage_dir, flyby_db)
            leg_stage_00 = Path(tmpdir) / "leg_00"
            leg_stage_01 = Path(tmpdir) / "leg_01"
            save_leg_stage(leg_stage_00, leg_dbs[0])
            save_leg_stage(leg_stage_01, leg_dbs[1])

            actual = combo.run_combo(
                encounter_db,
                cfg=combo.ComboBuildConfig(nL=2),
                leg_stage_dirs={0: leg_stage_00, 1: leg_stage_01},
                flyby_stage_dirs={1: stage_dir},
            )

        self.assertEqual(int(actual.I_traj.size), 1)
        np.testing.assert_array_equal(actual.encounter_IEs, np.asarray([[0, 1, 2]], dtype=np.int64))
        np.testing.assert_array_equal(actual.leg_ILs, np.asarray([[10, 20]], dtype=np.int64))
        np.testing.assert_array_equal(actual.flyby_IFs, np.asarray([[100]], dtype=np.int64))
        np.testing.assert_allclose(actual.dv_total_km_s, np.asarray([0.9], dtype=float))

    def test_run_combo_preemptive_dt_filter_prunes_stage_candidates(self) -> None:
        encounter_db = self._build_encounter_db()
        leg_dbs = self._build_leg_dbs()
        flyby_db = FlybyDB(
            IF=np.asarray([100, 101], dtype=np.int64),
            stage_id=np.asarray([1, 1], dtype=np.int64),
            IE=np.asarray([1, 1], dtype=np.int64),
            IL_in=np.asarray([10, 10], dtype=np.int64),
            IL_out=np.asarray([20, 20], dtype=np.int64),
            dv_km_s=np.asarray([0.4, 0.8], dtype=float),
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            stage_dir = Path(tmpdir) / "flyby_01"
            save_flyby_stage(stage_dir, flyby_db)
            leg_stage_00 = Path(tmpdir) / "leg_00"
            leg_stage_01 = Path(tmpdir) / "leg_01"
            save_leg_stage(leg_stage_00, leg_dbs[0])
            save_leg_stage(leg_stage_01, leg_dbs[1])

            actual = combo.run_combo(
                encounter_db,
                cfg=combo.ComboBuildConfig(
                    nL=2,
                    dt_filter_preemptive_s=10.0,
                ),
                leg_stage_dirs={0: leg_stage_00, 1: leg_stage_01},
                flyby_stage_dirs={1: stage_dir},
            )

        self.assertEqual(int(actual.I_traj.size), 1)
        np.testing.assert_array_equal(actual.flyby_IFs, np.asarray([[100]], dtype=np.int64))
        np.testing.assert_allclose(actual.dv_total_km_s, np.asarray([0.9], dtype=float))

    def test_run_combo_caches_frontier_keep_rows(self) -> None:
        entries = [
            EncounterEntry(
                IE=0,
                stage_id=0,
                t_et=0.0,
                body=399,
                r_km=np.asarray([1.0, 0.0, 0.0], dtype=float),
                v_km_s=np.asarray([0.0, 1.0, 0.0], dtype=float),
                mu_km3_s2=1.0,
                rmin_km=1.0,
            ),
            EncounterEntry(
                IE=1,
                stage_id=1,
                t_et=5.0,
                body=299,
                r_km=np.asarray([0.0, 1.0, 0.0], dtype=float),
                v_km_s=np.asarray([-1.0, 0.0, 0.0], dtype=float),
                mu_km3_s2=1.0,
                rmin_km=1.0,
            ),
            EncounterEntry(
                IE=2,
                stage_id=2,
                t_et=12.0,
                body=199,
                r_km=np.asarray([0.0, 0.0, 1.0], dtype=float),
                v_km_s=np.asarray([0.0, -1.0, 0.0], dtype=float),
                mu_km3_s2=1.0,
                rmin_km=1.0,
            ),
            EncounterEntry(
                IE=3,
                stage_id=3,
                t_et=20.0,
                body=199,
                r_km=np.asarray([0.0, 0.0, 1.0], dtype=float),
                v_km_s=np.asarray([0.0, -1.0, 0.0], dtype=float),
                mu_km3_s2=1.0,
                rmin_km=1.0,
            ),
        ]
        encounter_db = EncounterDB(
            entries=entries,
            stage_to_entry_ids={(0, 399): [0], (1, 299): [1], (2, 199): [2], (3, 199): [3]},
            entry_by_id={entry.IE: entry for entry in entries},
        )

        leg_dbs = {
            0: LegDatabase(
                IL=np.asarray([10], dtype=np.int64),
                stage_id=0,
                ID=np.asarray([0], dtype=np.int64),
                IA=np.asarray([1], dtype=np.int64),
                vinfD_km_s=np.zeros((1, 3), dtype=float),
                vinfA_km_s=np.zeros((1, 3), dtype=float),
                n_rev=np.asarray([0], dtype=np.int64),
                dv_lev_km_s=np.asarray([0.2], dtype=float),
                eta_lev=np.asarray([0.5], dtype=float),
            ),
            1: LegDatabase(
                IL=np.asarray([20, 21], dtype=np.int64),
                stage_id=1,
                ID=np.asarray([1, 1], dtype=np.int64),
                IA=np.asarray([2, 2], dtype=np.int64),
                vinfD_km_s=np.zeros((2, 3), dtype=float),
                vinfA_km_s=np.zeros((2, 3), dtype=float),
                n_rev=np.asarray([0, 0], dtype=np.int64),
                dv_lev_km_s=np.asarray([0.3, 0.35], dtype=float),
                eta_lev=np.asarray([0.5, 0.5], dtype=float),
            ),
            2: LegDatabase(
                IL=np.asarray([30, 31], dtype=np.int64),
                stage_id=2,
                ID=np.asarray([2, 2], dtype=np.int64),
                IA=np.asarray([3, 3], dtype=np.int64),
                vinfD_km_s=np.zeros((2, 3), dtype=float),
                vinfA_km_s=np.zeros((2, 3), dtype=float),
                n_rev=np.asarray([0, 0], dtype=np.int64),
                dv_lev_km_s=np.asarray([0.5, 0.6], dtype=float),
                eta_lev=np.asarray([0.5, 0.5], dtype=float),
            ),
        }

        flyby_01 = FlybyDB(
            IF=np.asarray([100], dtype=np.int64),
            stage_id=np.asarray([1], dtype=np.int64),
            IE=np.asarray([1], dtype=np.int64),
            IL_in=np.asarray([10], dtype=np.int64),
            IL_out=np.asarray([20], dtype=np.int64),
            dv_km_s=np.asarray([0.4], dtype=float),
        )
        flyby_02 = FlybyDB(
            IF=np.asarray([200, 201], dtype=np.int64),
            stage_id=np.asarray([2, 2], dtype=np.int64),
            IE=np.asarray([2, 2], dtype=np.int64),
            IL_in=np.asarray([20, 21], dtype=np.int64),
            IL_out=np.asarray([30, 31], dtype=np.int64),
            dv_km_s=np.asarray([0.6, 0.7], dtype=float),
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            leg_stage_00 = Path(tmpdir) / "leg_00"
            leg_stage_01 = Path(tmpdir) / "leg_01"
            leg_stage_02 = Path(tmpdir) / "leg_02"
            save_leg_stage(leg_stage_00, leg_dbs[0])
            save_leg_stage(leg_stage_01, leg_dbs[1])
            save_leg_stage(leg_stage_02, leg_dbs[2])

            flyby_stage_01 = Path(tmpdir) / "flyby_01"
            flyby_stage_02 = Path(tmpdir) / "flyby_02"
            save_flyby_stage(flyby_stage_01, flyby_01)
            save_flyby_stage(flyby_stage_02, flyby_02)

            actual = combo.run_combo(
                encounter_db,
                cfg=combo.ComboBuildConfig(nL=3),
                leg_stage_dirs={0: leg_stage_00, 1: leg_stage_01, 2: leg_stage_02},
                flyby_stage_dirs={1: flyby_stage_01, 2: flyby_stage_02},
            )

            keep_rows_path = flyby_stage_02 / "combo_keep_rows.npy"
            self.assertTrue(keep_rows_path.exists())
            np.testing.assert_array_equal(
                np.asarray(np.load(keep_rows_path, allow_pickle=False), dtype=np.int64),
                np.asarray([0], dtype=np.int64),
            )

        self.assertEqual(int(actual.I_traj.size), 1)
        np.testing.assert_array_equal(actual.leg_ILs, np.asarray([[10, 20, 30]], dtype=np.int64))
        np.testing.assert_array_equal(actual.flyby_IFs, np.asarray([[100, 200]], dtype=np.int64))
        np.testing.assert_allclose(actual.dv_total_km_s, np.asarray([2.0], dtype=float))


if __name__ == "__main__":
    unittest.main()

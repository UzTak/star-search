import json
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
from star.stage_db_npy import save_flyby_stage


class SaveSolutionInMemoryVectorsTest(unittest.TestCase):
    def _build_encounter_db(self) -> EncounterDB:
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
            body=199,
            r_km=np.asarray([0.0, 1.0, 0.0], dtype=float),
            v_km_s=np.asarray([-1.0, 0.0, 0.0], dtype=float),
            mu_km3_s2=1.0,
            rmin_km=1.0,
        )
        return EncounterDB(
            entries=[dep_entry, arr_entry],
            stage_to_entry_ids={(0, 399): [0], (1, 199): [1]},
            entry_by_id={0: dep_entry, 1: arr_entry},
        )

    def _build_output_db(self) -> combo.OutputDB:
        return combo.OutputDB(
            I_traj=np.asarray([0], dtype=np.int64),
            encounter_IEs=np.asarray([[0, 1]], dtype=np.int64),
            body_ids=np.asarray([[399, 199]], dtype=np.int64),
            times_et_s=np.asarray([[0.0, 10.0]], dtype=float),
            leg_ILs=np.asarray([[0]], dtype=np.int64),
            flyby_IFs=np.empty((1, 0), dtype=np.int64),
            dv_total_km_s=np.asarray([0.2], dtype=float),
            tof_total_s=np.asarray([10.0], dtype=float),
            per_leg_tof_s=np.asarray([[10.0]], dtype=float),
        )

    def test_save_solution_uses_in_memory_vectors(self) -> None:
        encounter_db = self._build_encounter_db()
        output_db = self._build_output_db()
        leg_db = LegDatabase(
            IL=np.asarray([0], dtype=np.int64),
            stage_id=0,
            ID=np.asarray([0], dtype=np.int64),
            IA=np.asarray([1], dtype=np.int64),
            vinfD_km_s=np.asarray([[1.0, 2.0, 3.0]], dtype=float),
            vinfA_km_s=np.asarray([[4.0, 5.0, 6.0]], dtype=float),
            n_rev=np.empty(0, dtype=np.int64),
            dv_lev_km_s=np.asarray([0.2], dtype=float),
            eta_lev=np.asarray([0.75], dtype=float),
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "result.jsonl"
            combo.save_solution(
                output_db=output_db,
                enc_db=encounter_db,
                leg_dbs={0: leg_db},
                flyby_dbs={},
                output_path=output_path,
            )

            lines = output_path.read_text(encoding="utf-8").splitlines()
            self.assertEqual(len(lines), 1)
            record = json.loads(lines[0])
            np.testing.assert_allclose(np.asarray(record["vinfD_km_s"], dtype=float), np.asarray([[1.0, 2.0, 3.0]]))
            np.testing.assert_allclose(np.asarray(record["vinfA_km_s"], dtype=float), np.asarray([[4.0, 5.0, 6.0]]))
            np.testing.assert_allclose(np.asarray(record["eta_lev"], dtype=float), np.asarray([0.75]))


    def test_save_solution_raises_when_vectors_missing(self) -> None:
        encounter_db = self._build_encounter_db()
        output_db = self._build_output_db()
        leg_db = LegDatabase(
            IL=np.asarray([0], dtype=np.int64),
            stage_id=0,
            ID=np.asarray([0], dtype=np.int64),
            IA=np.asarray([1], dtype=np.int64),
            vinfD_km_s=np.empty((0, 3), dtype=float),
            vinfA_km_s=np.empty((0, 3), dtype=float),
            n_rev=np.empty(0, dtype=np.int64),
            dv_lev_km_s=np.asarray([0.2], dtype=float),
            eta_lev=np.empty(0, dtype=float),
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "result.jsonl"
            with self.assertRaisesRegex(ValueError, "missing in-memory vinf/eta"):
                combo.save_solution(
                    output_db=output_db,
                    enc_db=encounter_db,
                    leg_dbs={0: leg_db},
                    flyby_dbs={},
                    output_path=output_path,
                )


class SaveSolutionDiskFlybyLookupTest(unittest.TestCase):
    def _build_encounter_db(self) -> EncounterDB:
        e0 = EncounterEntry(
            IE=0,
            stage_id=0,
            t_et=0.0,
            body=399,
            r_km=np.asarray([1.0, 0.0, 0.0], dtype=float),
            v_km_s=np.asarray([0.0, 1.0, 0.0], dtype=float),
            mu_km3_s2=1.0,
            rmin_km=1.0,
        )
        e1 = EncounterEntry(
            IE=1,
            stage_id=1,
            t_et=5.0,
            body=299,
            r_km=np.asarray([0.0, 1.0, 0.0], dtype=float),
            v_km_s=np.asarray([-1.0, 0.0, 0.0], dtype=float),
            mu_km3_s2=1.0,
            rmin_km=1.0,
        )
        e2 = EncounterEntry(
            IE=2,
            stage_id=2,
            t_et=12.0,
            body=199,
            r_km=np.asarray([0.0, 0.0, 1.0], dtype=float),
            v_km_s=np.asarray([0.0, -1.0, 0.0], dtype=float),
            mu_km3_s2=1.0,
            rmin_km=1.0,
        )
        return EncounterDB(
            entries=[e0, e1, e2],
            stage_to_entry_ids={(0, 399): [0], (1, 299): [1], (2, 199): [2]},
            entry_by_id={0: e0, 1: e1, 2: e2},
        )

    def test_save_solution_uses_stage_dirs_for_flyby_patch_dv(self) -> None:
        output_db = combo.OutputDB(
            I_traj=np.asarray([0], dtype=np.int64),
            encounter_IEs=np.asarray([[0, 1, 2]], dtype=np.int64),
            body_ids=np.asarray([[399, 299, 199]], dtype=np.int64),
            times_et_s=np.asarray([[0.0, 5.0, 12.0]], dtype=float),
            leg_ILs=np.asarray([[10, 20]], dtype=np.int64),
            flyby_IFs=np.asarray([[100]], dtype=np.int64),
            dv_total_km_s=np.asarray([0.9], dtype=float),
            tof_total_s=np.asarray([12.0], dtype=float),
            per_leg_tof_s=np.asarray([[5.0, 7.0]], dtype=float),
        )
        leg_dbs = {
            0: LegDatabase(
                IL=np.asarray([10], dtype=np.int64),
                stage_id=0,
                ID=np.asarray([0], dtype=np.int64),
                IA=np.asarray([1], dtype=np.int64),
                vinfD_km_s=np.asarray([[1.0, 0.0, 0.0]], dtype=float),
                vinfA_km_s=np.asarray([[1.0, 0.0, 0.0]], dtype=float),
                n_rev=np.empty(0, dtype=np.int64),
                dv_lev_km_s=np.asarray([0.2], dtype=float),
                eta_lev=np.asarray([0.5], dtype=float),
            ),
            1: LegDatabase(
                IL=np.asarray([20], dtype=np.int64),
                stage_id=1,
                ID=np.asarray([1], dtype=np.int64),
                IA=np.asarray([2], dtype=np.int64),
                vinfD_km_s=np.asarray([[1.0, 0.0, 0.0]], dtype=float),
                vinfA_km_s=np.asarray([[1.0, 0.0, 0.0]], dtype=float),
                n_rev=np.empty(0, dtype=np.int64),
                dv_lev_km_s=np.asarray([0.3], dtype=float),
                eta_lev=np.asarray([0.5], dtype=float),
            ),
        }

        flyby_stage = FlybyDB(
            IF=np.asarray([100], dtype=np.int64),
            stage_id=np.asarray([1], dtype=np.int64),
            IE=np.asarray([1], dtype=np.int64),
            IL_in=np.asarray([10], dtype=np.int64),
            IL_out=np.asarray([20], dtype=np.int64),
            dv_km_s=np.asarray([0.4], dtype=float),
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            stage_dir = tmp / "flyby_01"
            save_flyby_stage(stage_dir, flyby_stage)

            out_path = tmp / "result.jsonl"
            combo.save_solution(
                output_db=output_db,
                enc_db=self._build_encounter_db(),
                leg_dbs=leg_dbs,
                flyby_dbs=None,
                output_path=out_path,
                flyby_stage_dirs={1: stage_dir},
            )

            record = json.loads(out_path.read_text(encoding="utf-8").splitlines()[0])
            np.testing.assert_allclose(np.asarray(record["dv_patch_km_s"], dtype=float), np.asarray([0.4], dtype=float))


if __name__ == "__main__":
    unittest.main()

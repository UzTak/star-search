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


class SaveSolutionNullCollapseTest(unittest.TestCase):
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
                t_et=5.0,
                body=299,
                r_km=np.asarray([0.0, 2.0, 0.0], dtype=float),
                v_km_s=np.asarray([-1.0, 0.0, 0.0], dtype=float),
                mu_km3_s2=1.0,
                rmin_km=1.0,
            ),
            EncounterEntry(
                IE=3,
                stage_id=3,
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
            stage_to_entry_ids={(0, 399): [0], (1, 299): [1], (2, 299): [2], (3, 199): [3]},
            entry_by_id={entry.IE: entry for entry in entries},
        )

    def test_save_solution_collapses_null_leg_in_json(self) -> None:
        output_db = combo.OutputDB(
            I_traj=np.asarray([0], dtype=np.int64),
            encounter_IEs=np.asarray([[0, 1, 2, 3]], dtype=np.int64),
            body_ids=np.asarray([[399, 299, 299, 199]], dtype=np.int64),
            times_et_s=np.asarray([[0.0, 5.0, 5.0, 12.0]], dtype=float),
            leg_ILs=np.asarray([[10, 20, 30]], dtype=np.int64),
            flyby_IFs=np.asarray([[100, 200]], dtype=np.int64),
            dv_total_km_s=np.asarray([0.7], dtype=float),
            tof_total_s=np.asarray([12.0], dtype=float),
            per_leg_tof_s=np.asarray([[5.0, 0.0, 7.0]], dtype=float),
        )
        leg_dbs = {
            0: LegDatabase(
                IL=np.asarray([10], dtype=np.int64),
                stage_id=0,
                ID=np.asarray([0], dtype=np.int64),
                IA=np.asarray([1], dtype=np.int64),
                vinfD_km_s=np.asarray([[2.0, 0.0, 0.0]], dtype=float),
                vinfA_km_s=np.asarray([[1.0, 0.0, 0.0]], dtype=float),
                n_rev=np.asarray([0], dtype=np.int64),
                dv_lev_km_s=np.asarray([0.1], dtype=float),
                eta_lev=np.asarray([0.4], dtype=float),
            ),
            1: LegDatabase(
                IL=np.asarray([20], dtype=np.int64),
                stage_id=1,
                ID=np.asarray([1], dtype=np.int64),
                IA=np.asarray([2], dtype=np.int64),
                vinfD_km_s=np.asarray([[1.0, 0.0, 0.0]], dtype=float),
                vinfA_km_s=np.asarray([[1.0, 0.0, 0.0]], dtype=float),
                n_rev=np.asarray([0], dtype=np.int64),
                dv_lev_km_s=np.asarray([0.0], dtype=float),
                eta_lev=np.asarray([0.0], dtype=float),
            ),
            2: LegDatabase(
                IL=np.asarray([30], dtype=np.int64),
                stage_id=2,
                ID=np.asarray([2], dtype=np.int64),
                IA=np.asarray([3], dtype=np.int64),
                vinfD_km_s=np.asarray([[0.0, 1.0, 0.0]], dtype=float),
                vinfA_km_s=np.asarray([[0.0, 1.0, 0.0]], dtype=float),
                n_rev=np.asarray([0], dtype=np.int64),
                dv_lev_km_s=np.asarray([0.2], dtype=float),
                eta_lev=np.asarray([0.6], dtype=float),
            ),
        }
        flyby_dbs = {
            1: FlybyDB(
                IF=np.asarray([100], dtype=np.int64),
                stage_id=np.asarray([1], dtype=np.int64),
                IE=np.asarray([1], dtype=np.int64),
                IL_in=np.asarray([10], dtype=np.int64),
                IL_out=np.asarray([20], dtype=np.int64),
                dv_km_s=np.asarray([0.0], dtype=float),
            ),
            2: FlybyDB(
                IF=np.asarray([200], dtype=np.int64),
                stage_id=np.asarray([2], dtype=np.int64),
                IE=np.asarray([2], dtype=np.int64),
                IL_in=np.asarray([20], dtype=np.int64),
                IL_out=np.asarray([30], dtype=np.int64),
                dv_km_s=np.asarray([0.4], dtype=float),
            ),
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "result.jsonl"
            combo.save_solution(
                output_db=output_db,
                enc_db=self._build_encounter_db(),
                leg_dbs=leg_dbs,
                flyby_dbs=flyby_dbs,
                output_path=output_path,
                null_flags=[False, True, False],
            )

            record = json.loads(output_path.read_text(encoding="utf-8").splitlines()[0])
            np.testing.assert_array_equal(np.asarray(record["body_ids"], dtype=np.int64), np.asarray([399, 299, 199]))
            np.testing.assert_allclose(np.asarray(record["t_et_s"], dtype=float), np.asarray([0.0, 5.0, 12.0]))
            np.testing.assert_array_equal(np.asarray(record["leg_ils"], dtype=np.int64), np.asarray([10, 30]))
            np.testing.assert_array_equal(np.asarray(record["flyby_ifs"], dtype=np.int64), np.asarray([200]))
            np.testing.assert_allclose(np.asarray(record["dv_patch_km_s"], dtype=float), np.asarray([0.4]))
            np.testing.assert_allclose(np.asarray(record["dv_lev_km_s"], dtype=float), np.asarray([0.1, 0.2]))


if __name__ == "__main__":
    unittest.main()

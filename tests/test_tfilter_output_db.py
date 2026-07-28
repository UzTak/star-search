import sys
import unittest
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = REPO_ROOT / "star"
for path in (str(REPO_ROOT), str(PACKAGE_ROOT)):
    if path not in sys.path:
        sys.path.insert(0, path)

from star import combo


class TFilterPerEncounterTest(unittest.TestCase):
    def _build_output_db(self) -> combo.OutputDB:
        return combo.OutputDB(
            I_traj=np.asarray([0, 1, 2, 3], dtype=np.int64),
            encounter_IEs=np.asarray(
                [
                    [100, 200, 300],
                    [101, 201, 301],
                    [102, 202, 302],
                    [103, 203, 303],
                ],
                dtype=np.int64,
            ),
            body_ids=np.asarray(
                [
                    [399, 299, 199],
                    [399, 299, 199],
                    [399, 399, 199],
                    [399, 299, 199],
                ],
                dtype=np.int64,
            ),
            times_et_s=np.asarray(
                [
                    [0.0, 10.0, 20.0],
                    [1.0, 16.0, 22.0],
                    [1.0, 16.0, 22.0],
                    [6.0, 10.0, 20.0],
                ],
                dtype=float,
            ),
            leg_ILs=np.asarray(
                [
                    [10, 20],
                    [11, 21],
                    [12, 22],
                    [13, 23],
                ],
                dtype=np.int64,
            ),
            flyby_IFs=np.asarray(
                [
                    [1000],
                    [1001],
                    [1002],
                    [1003],
                ],
                dtype=np.int64,
            ),
            dv_total_km_s=np.asarray([1.0, 2.0, 0.5, 1.1], dtype=float),
            tof_total_s=np.asarray([20.0, 21.0, 21.0, 14.0], dtype=float),
            per_leg_tof_s=np.asarray(
                [
                    [10.0, 10.0],
                    [10.0, 11.0],
                    [10.0, 11.0],
                    [4.0, 10.0],
                ],
                dtype=float,
            ),
        )

    def test_tfilter_groups_by_body_and_dep_arr_bins(self) -> None:
        output_db = self._build_output_db()
        filtered, stats = combo.tfilter_output_db(
            output_db,
            np.asarray([5.0, 5.0, 5.0], dtype=float),
        )

        np.testing.assert_array_equal(
            np.sort(np.asarray(filtered.I_traj, dtype=np.int64)),
            np.asarray([0, 2, 3], dtype=np.int64),
        )
        self.assertEqual(int(stats["num_out"]), 3)
        self.assertEqual(int(stats["num_bins"]), 3)

    def test_tfilter_ignores_intermediate_time_bins_without_preemptive_filter(self) -> None:
        output_db = self._build_output_db()
        filtered, _ = combo.tfilter_output_db(
            output_db,
            np.asarray([5.0, 1.0, 5.0], dtype=float),
        )
        np.testing.assert_array_equal(
            np.sort(np.asarray(filtered.I_traj, dtype=np.int64)),
            np.asarray([0, 2, 3], dtype=np.int64),
        )

    def test_tfilter_scalar_still_supported(self) -> None:
        output_db = self._build_output_db()
        filtered, _ = combo.tfilter_output_db(output_db, 5.0)
        np.testing.assert_array_equal(
            np.sort(np.asarray(filtered.I_traj, dtype=np.int64)),
            np.asarray([0, 2, 3], dtype=np.int64),
        )


if __name__ == "__main__":
    unittest.main()

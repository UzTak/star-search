import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = REPO_ROOT / "star"
for path in (str(REPO_ROOT), str(PACKAGE_ROOT)):
    if path not in sys.path:
        sys.path.insert(0, path)

from star import combo
from star import pipeline as star_driver
from star.encounter_database import EncounterDB, EncounterEntry
from star.leg_database import LegDatabase
from star.stage_db_npy import save_leg_stage


class ComboDvCapResolutionTest(unittest.TestCase):
    def test_missing_cap_returns_none(self) -> None:
        problem = SimpleNamespace()
        self.assertIsNone(star_driver._resolve_combo_dv_cap_km_s(problem))

    def test_falls_back_to_paper_style_name(self) -> None:
        problem = SimpleNamespace(dVtotal=3.5)
        self.assertEqual(star_driver._resolve_combo_dv_cap_km_s(problem), 3.5)

    def test_prefers_explicit_internal_name(self) -> None:
        problem = SimpleNamespace(dv_total_max_km_s=2.5, dVtotal=3.5)
        self.assertEqual(star_driver._resolve_combo_dv_cap_km_s(problem), 2.5)

    def test_rejects_negative_cap(self) -> None:
        problem = SimpleNamespace(dVtotal=-1.0)
        with self.assertRaisesRegex(ValueError, "finite non-negative float"):
            star_driver._resolve_combo_dv_cap_km_s(problem)

    def test_rejects_non_finite_cap(self) -> None:
        problem = SimpleNamespace(dv_total_max_km_s=np.inf)
        with self.assertRaisesRegex(ValueError, "finite non-negative float"):
            star_driver._resolve_combo_dv_cap_km_s(problem)


class ComboDvCapRunComboTest(unittest.TestCase):
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

    def _build_leg_db(self) -> LegDatabase:
        num_rows = 3
        return LegDatabase(
            IL=np.asarray([10, 11, 12], dtype=np.int64),
            stage_id=0,
            ID=np.asarray([0, 0, 0], dtype=np.int64),
            IA=np.asarray([1, 1, 1], dtype=np.int64),
            vinfD_km_s=np.zeros((num_rows, 3), dtype=float),
            vinfA_km_s=np.zeros((num_rows, 3), dtype=float),
            n_rev=np.zeros(num_rows, dtype=np.int64),
            dv_lev_km_s=np.asarray([0.0, 1.0, 2.0], dtype=float),
            eta_lev=np.full(num_rows, 0.5, dtype=float),
        )

    def test_run_combo_one_leg_is_rejected(self) -> None:
        enc_db = self._build_encounter_db()
        leg_db = self._build_leg_db()

        with tempfile.TemporaryDirectory() as tmpdir:
            leg_stage_00 = Path(tmpdir) / "leg_00"
            save_leg_stage(leg_stage_00, leg_db)
            with self.assertRaisesRegex(ValueError, "At least two legs"):
                combo.run_combo(
                    enc_db,
                    cfg=combo.ComboBuildConfig(nL=1),
                    leg_stage_dirs={0: leg_stage_00},
                    flyby_stage_dirs={},
                )


class BuildIndexMapTest(unittest.TestCase):
    def test_build_index_map_preserves_original_row_order(self) -> None:
        encounter_ids = np.asarray([7, 3, 7, 5, 3, 7], dtype=np.int64)
        actual = combo._build_index_map(encounter_ids)

        self.assertEqual(set(actual.keys()), {3, 5, 7})
        np.testing.assert_array_equal(actual[3], np.asarray([1, 4], dtype=np.int64))
        np.testing.assert_array_equal(actual[5], np.asarray([3], dtype=np.int64))
        np.testing.assert_array_equal(actual[7], np.asarray([0, 2, 5], dtype=np.int64))


class CombineSegmentsIndexReuseTest(unittest.TestCase):
    def test_combine_segments_reuses_right_segment_left_index(self) -> None:
        seg_left = combo._make_segment_db(
            leg_start_stage=0,
            leg_end_stage=1,
            leg_ils=np.asarray(
                [
                    [10, 20],
                    [11, 21],
                    [12, 20],
                ],
                dtype=np.int64,
            ),
            flyby_ifs=np.asarray(
                [
                    [100],
                    [101],
                    [102],
                ],
                dtype=np.int64,
            ),
            dv_total_km_s=np.asarray([1.0, 2.0, 3.0], dtype=float),
        )
        seg_right = combo._make_segment_db(
            leg_start_stage=1,
            leg_end_stage=2,
            leg_ils=np.asarray(
                [
                    [20, 30],
                    [21, 31],
                    [20, 32],
                ],
                dtype=np.int64,
            ),
            flyby_ifs=np.asarray(
                [
                    [200],
                    [201],
                    [202],
                ],
                dtype=np.int64,
            ),
            dv_total_km_s=np.asarray([10.0, 20.0, 30.0], dtype=float),
        )

        original_build_index_map = combo._build_index_map
        with patch.object(combo, "_build_index_map", wraps=original_build_index_map) as build_index_map_mock:
            combined = combo.combine_segments(
                seg_left,
                seg_right,
                join_leg_dv_by_il={20: 0.25, 21: 0.5},
            )

        self.assertEqual(build_index_map_mock.call_count, 2)

        # Row ordering contract: `combine_segments` iterates the shared middle-leg
        # IL groups and emits repeat(left) x tile(right) within each group, so all
        # rows joining on IL=20 precede the rows joining on IL=21.
        np.testing.assert_array_equal(
            combined.leg_ILs,
            np.asarray(
                [
                    [10, 20, 30],
                    [10, 20, 32],
                    [12, 20, 30],
                    [12, 20, 32],
                    [11, 21, 31],
                ],
                dtype=np.int64,
            ),
        )
        np.testing.assert_array_equal(
            combined.flyby_IFs,
            np.asarray(
                [
                    [100, 200],
                    [100, 202],
                    [102, 200],
                    [102, 202],
                    [101, 201],
                ],
                dtype=np.int64,
            ),
        )
        np.testing.assert_allclose(
            combined.dv_total_km_s,
            np.asarray([10.75, 30.75, 12.75, 32.75, 21.5], dtype=float),
        )


if __name__ == "__main__":
    unittest.main()

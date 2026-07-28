import sys
import unittest
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = REPO_ROOT / "star"
for path in (str(REPO_ROOT), str(PACKAGE_ROOT)):
    if path not in sys.path:
        sys.path.insert(0, path)

from star.leg_database import LegBuildConfig, _process_lambert_chunk


class ProcessLambertChunkFilterTest(unittest.TestCase):
    def test_solve_arc_chunk_with_no_raw_vinf_feasible_seeds_is_skipped(self) -> None:
        """A seed outside the pre-DSM v-infinity screen is dropped before solve_arc.

        The screen is widened by `vinf_margin_pre_filter_km_s` (default 10 km/s),
        because a DSM can pull a marginally-infeasible seed back inside the bound.
        This test targets the screen itself, so the margin is pinned to zero and
        the single seed sits above the exact 10 km/s bound.
        """

        dep_data = {
            "IE": np.asarray([10], dtype=np.int64),
            "t_et": np.asarray([0.0], dtype=float),
        }
        arr_data = {
            "IE": np.asarray([20], dtype=np.int64),
            "t_et": np.asarray([100.0], dtype=float),
        }

        out = _process_lambert_chunk(
            dep_data=dep_data,
            arr_data=arr_data,
            dep_rows_chunk=np.asarray([0], dtype=np.int64),
            arr_rows_chunk=np.asarray([0], dtype=np.int64),
            dt_candidates_chunk_s=np.asarray([100.0], dtype=float),
            r_dep_km=np.asarray([[1.0, 0.0, 0.0]], dtype=float),
            r_arr_km=np.asarray([[0.0, 1.0, 0.0]], dtype=float),
            # Non-zero body velocities: a zero-velocity body is unphysical and
            # would raise inside solve_arc, masking a real screening regression.
            v_body_dep_km_s=np.asarray([[0.0, 1.0, 0.0]], dtype=float),
            v_body_arr_km_s=np.asarray([[-1.0, 0.0, 0.0]], dtype=float),
            # vinfD = 15.03 km/s, vinfA = 16.0 km/s -- both above the 10 km/s bound.
            v_dep_km_s=np.asarray([[15.0, 0.0, 0.0]], dtype=float),
            v_arr_km_s=np.asarray([[15.0, 0.0, 0.0]], dtype=float),
            nrev_signed_eval=np.asarray([0], dtype=np.int64),
            source_idx=np.asarray([0], dtype=np.int64),
            cfg=LegBuildConfig(
                leg_stage_id=0,
                dep_stage_id=0,
                arr_stage_id=1,
                tof_min_s=0.0,
                tof_max_s=200.0,
                vinfD_bounds_km_s=(0.0, 10.0),
                vinfA_bounds_km_s=(0.0, 10.0),
                lambert_mu_km3_s2=1.0,
                dvlev_max_km_s=1.0,
                delta_dvlev_km_s=0.5,
                vinf_margin_pre_filter_km_s=0.0,
            ),
            dV_lev_grid_km_s=np.asarray([0.5, 1.0], dtype=float),
            vinfD_min_km_s=0.0,
            vinfD_max_km_s=10.0,
            vinfA_min_km_s=0.0,
            vinfA_max_km_s=10.0,
        )

        self.assertIsNone(out)

    def test_ballistic_chunk_emits_zero_dsm_leveraging(self) -> None:
        dep_data = {
            "IE": np.asarray([10], dtype=np.int64),
            "t_et": np.asarray([0.0], dtype=float),
        }
        arr_data = {
            "IE": np.asarray([20], dtype=np.int64),
            "t_et": np.asarray([100.0], dtype=float),
        }

        out = _process_lambert_chunk(
            dep_data=dep_data,
            arr_data=arr_data,
            dep_rows_chunk=np.asarray([0], dtype=np.int64),
            arr_rows_chunk=np.asarray([0], dtype=np.int64),
            dt_candidates_chunk_s=np.asarray([100.0], dtype=float),
            r_dep_km=np.asarray([[1.0, 0.0, 0.0]], dtype=float),
            r_arr_km=np.asarray([[0.0, 1.0, 0.0]], dtype=float),
            v_body_dep_km_s=np.asarray([[0.0, 0.0, 0.0]], dtype=float),
            v_body_arr_km_s=np.asarray([[0.0, 0.0, 0.0]], dtype=float),
            v_dep_km_s=np.asarray([[1.0, 0.0, 0.0]], dtype=float),
            v_arr_km_s=np.asarray([[0.0, 1.0, 0.0]], dtype=float),
            nrev_signed_eval=np.asarray([0], dtype=np.int64),
            source_idx=np.asarray([0], dtype=np.int64),
            cfg=LegBuildConfig(
                leg_stage_id=0,
                dep_stage_id=0,
                arr_stage_id=1,
                tof_min_s=0.0,
                tof_max_s=200.0,
                vinfD_bounds_km_s=(0.0, 10.0),
                vinfA_bounds_km_s=(0.0, 10.0),
                lambert_mu_km3_s2=1.0,
                dvlev_max_km_s=0.0,
                delta_dvlev_km_s=0.0,
            ),
            dV_lev_grid_km_s=np.empty(0, dtype=float),
            vinfD_min_km_s=0.0,
            vinfD_max_km_s=10.0,
            vinfA_min_km_s=0.0,
            vinfA_max_km_s=10.0,
        )

        self.assertIsNotNone(out)
        np.testing.assert_allclose(np.asarray(out["dv_lev_km_s"], dtype=float), np.asarray([0.0], dtype=float))


if __name__ == "__main__":
    unittest.main()

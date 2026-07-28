import sys
import unittest
import warnings
from pathlib import Path
from unittest import mock

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = REPO_ROOT / "star"
for path in (str(REPO_ROOT), str(PACKAGE_ROOT)):
    if path not in sys.path:
        sys.path.insert(0, path)

from star import plotter


class PlotterCliAndMetricTest(unittest.TestCase):
    def test_arg_parser_defaults_to_tof_vs_dv_without_color(self) -> None:
        args = plotter._build_arg_parser().parse_args([])

        self.assertEqual(args.x_metric, "tof")
        self.assertEqual(args.y_metric, "dv")
        self.assertEqual(args.color_metric, "none")

    def test_arg_parser_accepts_explicit_metric_selection(self) -> None:
        args = plotter._build_arg_parser().parse_args(["--x", "t0", "--y", "dv", "--color", "tof"])

        self.assertEqual(args.x_metric, "t0")
        self.assertEqual(args.y_metric, "dv")
        self.assertEqual(args.color_metric, "tof")

    def test_extract_plot_metric_supports_t0_tof_and_boundary_dv(self) -> None:
        record = {
            "t_et_s": [1234.5, 5678.0],
            "tof_total_s": 2.5 * 86400.0,
            "dv_total_km_s": 3.0,
            "dv_escape_km_s": 0.5,
            "dv_insertion_km_s": 0.25,
        }

        t0_metric = plotter._extract_plot_metric(record, "t0", include_boundary_dv=(False, False))
        tof_metric = plotter._extract_plot_metric(record, "tof", include_boundary_dv=(False, False))
        dv_metric = plotter._extract_plot_metric(record, "dv", include_boundary_dv=(True, True))

        self.assertEqual(t0_metric.metric_id, "t0")
        self.assertTrue(t0_metric.is_calendar_date)
        self.assertEqual(t0_metric.value, 1234.5)
        self.assertEqual(tof_metric.value, 2.5)
        self.assertAlmostEqual(dv_metric.value, 3.75)

    def test_metric_scatter_only_draws_pareto_for_default_trade_plot(self) -> None:
        records = [
            {"traj_id": 1, "t_et_s": [100.0], "tof_total_days": 10.0, "dv_total_km_s": 5.0},
            {"traj_id": 2, "t_et_s": [200.0], "tof_total_days": 12.0, "dv_total_km_s": 4.0},
        ]

        fig, _ax, pareto_indices, plot_data = plotter.plot_trajectory_metric_scatter(
            records,
            x_metric="t0",
            y_metric="dv",
            color_metric="tof",
            show=False,
            return_plot_data=True,
        )

        self.assertIsNone(pareto_indices)
        self.assertIsNone(plot_data["pareto_line_artist"])
        self.assertIsNotNone(plot_data["colorbar"])
        np.testing.assert_allclose(plot_data["all_x_values"], np.asarray([100.0, 200.0], dtype=float))
        plt.close(fig)


class PlotterLeveragedReconstructionTest(unittest.TestCase):
    def test_plot_trajectory_reconstructs_split_endpoint_velocities_from_vinf(self) -> None:
        traj_record = {
            "traj_id": 7,
            "t_et_s": [0.0, 10.0],
            "body_ids": [399, 299],
            "vinfD_km_s": [[1.0, 2.0, 3.0]],
            "vinfA_km_s": [[4.0, 5.0, 6.0]],
            "dv_lev_km_s": [0.25],
            "eta_lev": [0.4],
        }

        sample_calls = []

        def fake_spice_state(target_id_or_name, et_s, observer=plotter.DEFAULT_OBSERVER, frame=plotter.DEFAULT_FRAME):
            body_id = int(target_id_or_name)
            et_val = float(et_s)
            if body_id == 399 and abs(et_val - 0.0) <= 1e-12:
                return np.asarray([1.0, 0.0, 0.0], dtype=float), np.asarray([10.0, 20.0, 30.0], dtype=float)
            if body_id == 299 and abs(et_val - 10.0) <= 1e-12:
                return np.asarray([2.0, 0.0, 0.0], dtype=float), np.asarray([40.0, 50.0, 60.0], dtype=float)
            return np.asarray([float(body_id), et_val, 0.0], dtype=float), np.asarray([0.0, 0.0, 0.0], dtype=float)

        def fake_sample_leg(r0_km, v0_km_s, t0_et_s, t1_et_s, mu_km3_s2, num_samples):
            sample_calls.append(
                {
                    "r0_km": np.asarray(r0_km, dtype=float).copy(),
                    "v0_km_s": np.asarray(v0_km_s, dtype=float).copy(),
                    "t0_et_s": float(t0_et_s),
                    "t1_et_s": float(t1_et_s),
                    "num_samples": int(num_samples),
                }
            )
            return np.vstack(
                (
                    np.asarray(r0_km, dtype=float).reshape(1, 3),
                    np.asarray(r0_km, dtype=float).reshape(1, 3),
                )
            )

        def fake_kepler_propagate_universal(r_km, v_km_s, dt_s, mu_km3_s2):
            return np.zeros(3, dtype=float), np.asarray(v_km_s, dtype=float)

        with warnings.catch_warnings(record=True) as caught, mock.patch.object(
            plotter,
            "_spice_state",
            side_effect=fake_spice_state,
        ), mock.patch.object(
            plotter,
            "estimate_orbital_period_s",
            return_value=1.0,
        ), mock.patch.object(
            plotter,
            "sample_leg",
            side_effect=fake_sample_leg,
        ), mock.patch.object(
            plotter,
            "kepler_propagate_universal",
            side_effect=fake_kepler_propagate_universal,
        ), mock.patch.object(
            plotter.spice,
            "et2utc",
            side_effect=lambda et_s, _fmt, _prec: f"2000-01-{int(float(et_s)) + 1:02d}T00:00:00",
        ):
            fig, _ax = plotter.plot_trajectory_2d(
                traj_record=traj_record,
                mu_central=1.0,
                num_samples_leg=8,
                num_samples_orbit=2,
                closure_pos_tol_km=1.0e12,
                closure_vel_tol_km_s=1.0e12,
                show=False,
            )

        self.assertEqual(len(sample_calls), 2)
        np.testing.assert_allclose(
            sample_calls[0]["v0_km_s"],
            np.asarray([11.0, 22.0, 33.0], dtype=float),
            rtol=0.0,
            atol=0.0,
        )
        np.testing.assert_allclose(
            sample_calls[1]["v0_km_s"],
            np.asarray([-44.0, -55.0, -66.0], dtype=float),
            rtol=0.0,
            atol=0.0,
        )
        self.assertFalse(
            any("Partial DSM split metadata" in str(item.message) for item in caught),
        )
        plt.close(fig)


if __name__ == "__main__":
    unittest.main()

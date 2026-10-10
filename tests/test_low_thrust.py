"""Smoke test for generating and replaying a quadratic low-thrust leg."""
import tempfile
import unittest
from dataclasses import replace
from unittest import mock

import numpy as np

from star.encounter_database import EncounterDB, EncounterEntry
from star.leg_database import LegBuildConfig, build_leg_db, build_leg_stage_to_disk
from star.low_thrust import propagate_binned_low_thrust, solve_arc_low_thrust
from star.maneuver_placement import prepare_primer_geometry
from star.stage_db_npy import load_leg_stage


def _physical_fixture():
    length, speed = 1e8, 30.0
    mu = length * speed**2
    states = []
    for nu in (0.25, 1.35):
        p = 2 * (1 - 0.3**2)
        er, et = np.array([np.cos(nu), np.sin(nu), 0]), np.array([-np.sin(nu), np.cos(nu), 0])
        r = p / (1 + 0.3 * np.cos(nu)) * er * length
        v = (0.3 * np.sin(nu) * er + (1 + 0.3 * np.cos(nu)) * et) / np.sqrt(p) * speed
        states.append((r, v))
    planets = np.array([[0.1, 0.55, 0.05], [-0.05, 0.45, -0.03]]) * speed
    times = [0.0]
    for (r0, v0), (rf, vf) in zip(states, states[1:]):
        g = prepare_primer_geometry(r0, rf, [v0], [vf], mu, [0])
        times.append(times[-1] + (g["Mf"][0] - g["M0"][0]) / np.sqrt(mu / g["a_semimajor"][0]**3))
    reference_acc = 1e-4 * speed**2 / length
    first = solve_arc_low_thrust(*[states[0][0], states[1][0], planets[0], planets[1],
                                   [states[0][1]], [states[1][1]], [0], [reference_acc], mu])
    arrival_speed = np.linalg.norm(first["Vinff_vec_lt_km_s"][0, 0])
    entries = [EncounterEntry(IE=i, stage_id=i, t_et=times[i], body=body,
                              r_km=states[i][0], v_km_s=planets[i],
                              mu_km3_s2=3.2e5, rmin_km=6000)
               for i, body in enumerate((399, 299))]
    enc = EncounterDB(entries=entries, entry_by_id={e.IE: e for e in entries},
                      stage_to_entry_ids={(e.stage_id, e.body): [e.IE] for e in entries})
    cfg = LegBuildConfig(leg_stage_id=0, dep_stage_id=0, arr_stage_id=1,
                         tof_min_s=0, tof_max_s=np.inf, lambert_mu_km3_s2=mu,
                         vinfD_bounds_km_s=(0, 50), vinfA_bounds_km_s=(0, arrival_speed + 1e-5),
                         acc_lt_max_km_s2=reference_acc,
                         delta_acc_lt_km_s2=reference_acc, lev_type=("+-",))
    return enc, cfg, states, mu


class LowThrustSmokeTest(unittest.TestCase):
    def test_low_thrust_leg_roundtrip(self):
        enc, cfg, _, mu = _physical_fixture()
        tof = enc.entries[1].t_et - enc.entries[0].t_et
        acceleration = cfg.acc_lt_max_km_s2
        cfg = replace(cfg, acc_lt_max_km_s2=10 * acceleration,
                      lt_dv_max_km_s=acceleration * tof * 1.01)
        # The nominal arrival fails vlim; thrust recovers it without any shooting or ODE solve.
        with mock.patch("star.low_thrust.solve_ivp", side_effect=AssertionError("Search propagated")):
            leg = build_leg_db(enc, cfg)
            with tempfile.TemporaryDirectory() as directory:
                build_leg_stage_to_disk(enc, cfg, stage_dir=directory)
                disk = load_leg_stage(directory)
                np.testing.assert_allclose(disk.vinfD_km_s, leg.vinfD_km_s)
                np.testing.assert_allclose(disk.vinfA_km_s, leg.vinfA_km_s)
                np.testing.assert_allclose(disk.lt_control, leg.lt_control)
                np.testing.assert_array_equal(disk.acc_lt_km_s2, leg.acc_lt_km_s2)
                np.testing.assert_array_equal(disk.lt_lev_type, leg.lt_lev_type)
        self.assertEqual(leg.IL.size, 1)  # Higher throttle samples exceed the existing mission cap.
        self.assertEqual(leg.acc_lt_km_s2[0], acceleration)
        self.assertGreater(np.linalg.norm(leg.lt_control[0, 3:6] - enc.entries[1].v_km_s), cfg.vinfA_bounds_km_s[1])
        self.assertLessEqual(np.linalg.norm(leg.vinfA_km_s[0]), cfg.vinfA_bounds_km_s[1])
        self.assertEqual(leg.lt_lev_type[0], 2)
        # A single physical replay measures approximation error; it does not silently correct it.
        end = propagate_binned_low_thrust(enc.entries[0].r_km, enc.entries[1].r_km,
            enc.entries[0].v_km_s, enc.entries[1].v_km_s, leg.vinfD_km_s[0],
            0, tof, mu, acceleration, leg.lt_control[0])
        self.assertTrue(np.all(np.isfinite(end)))
        self.assertLess(np.linalg.norm(end[:3] - enc.entries[1].r_km), 10.0)


if __name__ == "__main__":
    unittest.main()

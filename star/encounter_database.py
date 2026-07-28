"""Encounter database builder for Star Section 2.2 body data module.

This module builds deterministic encounter entries E using:
- stage-specific body lists,
- explicit ET bounds and step sizes,
- SPICE ephemeris states in heliocentric inertial frame.

See NOTATION.md for the IE index convention and encounter-stage definition.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from importlib import util as importlib_util
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import spiceypy as spice


from star.constants import DEFAULT_FRAME, DEFAULT_OBSERVER, get_gm, get_radii
from star.constants.time import et_seconds_to_j2000_days, j2000_days_to_et_seconds
from star.time_opt import time_opt, tof_opt


@dataclass(frozen=True)
class EncounterStageConfig:
    """Configuration for one encounter stage.

    Attributes:
        stage_id: Stage identifier [unitless int].
        bodies: Allowed encounter bodies [NAIF int IDs].
        t_min_et: Stage lower epoch bound [ET seconds past J2000].
        t_max_et: Stage upper epoch bound [ET seconds past J2000].
        dt_et: Time discretization step [seconds].
        amin_km: Per-body minimum flyby altitude above body surface [km], keyed by NAIF int ID.
    """

    stage_id: int
    bodies: List[int]
    t_min_et: float
    t_max_et: float
    dt_et: float
    amin_km: Dict[int, float] = field(default_factory=dict)


@dataclass(frozen=True)
class EncounterDBConfig:
    """Top-level encounter database build configuration."""

    stages: List[EncounterStageConfig]
    frame: str = DEFAULT_FRAME
    observer: str = DEFAULT_OBSERVER
    amin_fallback_km: float = 300.0


@dataclass(frozen=True)
class EncounterEntry:
    """One encounter node entry in the database.

    IE starts at 0 and increments globally in deterministic generation order.
    """

    IE: int
    stage_id: int
    t_et: float
    body: int
    r_km: np.ndarray
    v_km_s: np.ndarray
    mu_km3_s2: float
    rmin_km: float


@dataclass
class EncounterDB:
    """Encounter database container with indexing helpers."""

    entries: List[EncounterEntry]
    stage_to_entry_ids: Dict[Tuple[int, int], List[int]]
    entry_by_id: Dict[int, EncounterEntry]

    def get_entry(self, entry_id: int) -> EncounterEntry:
        """Return one encounter entry by IE [unitless int]."""

        return self.entry_by_id[entry_id]


def make_time_grid(t_min_et: float, t_max_et: float, dt_et: float) -> np.ndarray:
    """Create deterministic ET grid [seconds] for one (stage, body).

    Grid rule:
        K = floor((t_max_et - t_min_et)/dt_et + 1e-12)
        t_k = t_min_et + k * dt_et for k = 0..K
        then keep t_k <= t_max_et + 1e-9.
    """

    t_min_et = float(t_min_et)
    t_max_et = float(t_max_et)
    dt_et = float(dt_et)

    if dt_et <= 0.0:
        raise ValueError("dt_et must be positive [seconds].")
    if t_max_et < t_min_et:
        raise ValueError("t_max_et must be >= t_min_et [seconds].")

    k_max = int(np.floor((t_max_et - t_min_et) / dt_et + 1e-12))
    grid_et = t_min_et + dt_et * np.arange(k_max + 1, dtype=float)
    return grid_et[grid_et <= (t_max_et + 1e-9)]


def build_encounter_database(config: EncounterDBConfig) -> EncounterDB:
    """Build encounter database E for selected stages.

    Deterministic ordering:
    1. Stages sorted by ascending stage_id (stable).
    2. Bodies in each stage follow provided list order.
    3. Epochs ascend using `make_time_grid`.
    4. IE assigned globally from 0.
    """

    entries: List[EncounterEntry] = []
    stage_to_entry_ids: Dict[Tuple[int, int], List[int]] = {}
    entry_by_id: Dict[int, EncounterEntry] = {}

    ordered_stages = sorted(config.stages, key=lambda stage: stage.stage_id)

    next_entry_id = 0
    for stage in ordered_stages:
        for body_id in stage.bodies:
            if not isinstance(body_id, int):
                raise TypeError(f"body IDs must be int NAIF IDs, got {type(body_id).__name__}: {body_id}")

            stage_key = (int(stage.stage_id), int(body_id))
            stage_to_entry_ids[stage_key] = []

            body_id_str = str(body_id)
            mu_km3_s2 = float(get_gm(body_id_str)[0])
            mean_radius_km = float(get_radii(body_id_str)[0])

            amin_km = float(stage.amin_km.get(body_id, config.amin_fallback_km))
            rmin_km = mean_radius_km + amin_km

            t_grid_et = make_time_grid(stage.t_min_et, stage.t_max_et, stage.dt_et)
            for t_et in t_grid_et:
                state6, _ = spice.spkezr(body_id_str, float(t_et), config.frame, "NONE", config.observer)
                state6_array = np.asarray(state6, dtype=float)
                r_km, v_km_s = state6_array[:3], state6_array[3:]

                entry = EncounterEntry(
                    IE=next_entry_id,
                    stage_id=int(stage.stage_id),
                    t_et=float(t_et),
                    body=int(body_id),
                    r_km=r_km,
                    v_km_s=v_km_s,
                    mu_km3_s2=mu_km3_s2,
                    rmin_km=float(rmin_km),
                )
                entries.append(entry)
                stage_to_entry_ids[stage_key].append(next_entry_id)
                entry_by_id[next_entry_id] = entry
                next_entry_id += 1

    return EncounterDB(
        entries=entries,
        stage_to_entry_ids=stage_to_entry_ids,
        entry_by_id=entry_by_id,
    )


def tighten_stage_time_bounds_days(
    time_bounds_days: np.ndarray,
    tof_min_days: np.ndarray,
    tof_max_days: np.ndarray,
) -> np.ndarray:
    """Tighten stage time bounds in days using `time_opt` and `tof_opt`.

    Tightening loop:
    1) tighten encounter time bounds with `time_opt`,
    2) tighten TOF bounds with `tof_opt`,
    3) repeat until convergence (or max iterations).
    """

    bounds = np.asarray(time_bounds_days, dtype=float)
    if bounds.ndim != 2 or bounds.shape[1] != 2:
        raise ValueError("time_bounds_days must have shape (n_stage, 2).")

    num_stages = int(bounds.shape[0])
    tof_min_work = np.asarray(tof_min_days, dtype=float).copy()
    tof_max_work = np.asarray(tof_max_days, dtype=float).copy()
    if tof_min_work.shape != (num_stages, num_stages) or tof_max_work.shape != (num_stages, num_stages):
        raise ValueError("tof_min_days/tof_max_days must have shape (n_stage, n_stage).")

    tmin_work = bounds[:, 0].copy()
    tmax_work = bounds[:, 1].copy()

    max_iter = max(5, 2 * num_stages)
    for _ in range(max_iter):
        prev_tmin = tmin_work.copy()
        prev_tmax = tmax_work.copy()
        prev_tof_min = tof_min_work.copy()
        prev_tof_max = tof_max_work.copy()

        tmin_work, tmax_work = time_opt(tmin_work, tmax_work, tof_min_work, tof_max_work)
        tof_min_work, tof_max_work, _ = tof_opt(tmin_work, tmax_work, tof_min_work, tof_max_work)
        tmin_work, tmax_work = time_opt(tmin_work, tmax_work, tof_min_work, tof_max_work)

        converged = (
            np.allclose(tmin_work, prev_tmin, rtol=0.0, atol=1e-12)
            and np.allclose(tmax_work, prev_tmax, rtol=0.0, atol=1e-12)
            and np.allclose(tof_min_work, prev_tof_min, rtol=0.0, atol=1e-12, equal_nan=True)
            and np.allclose(tof_max_work, prev_tof_max, rtol=0.0, atol=1e-12, equal_nan=True)
        )
        if converged:
            break

    tightened_bounds = np.column_stack((tmin_work, tmax_work))
    if np.any(tightened_bounds[:, 0] > tightened_bounds[:, 1]):
        raise ValueError("Inconsistent stage time bounds after tightening.")
    return tightened_bounds


if __name__ == "__main__":
    
    # Demo: load example/earth_to_jupiter.py, build EncounterDB, print table.
    
    repo_root = Path(__file__).resolve().parent.parent
    metakernel_path = repo_root / "star" / "METAKERN.tm"
    spice.kclear()
    spice.furnsh(str(metakernel_path))
    print(f"Loaded SPICE metakernel: {metakernel_path}")
    
    import example.earth_to_jupiter as example

    bodies_by_stage = [list(stage_bodies) for stage_bodies in example.Body]
    time_bounds_days = np.asarray(example.Time, dtype=float)
    tof_min_days = np.asarray(example.tof_min, dtype=float)
    tof_max_days = np.asarray(example.tof_max, dtype=float)
    dt_days_by_stage = np.asarray(example.dt, dtype=float)
    alt_km_by_stage = list(example.alt)

    tightened_bounds_days = tighten_stage_time_bounds_days(time_bounds_days, tof_min_days, tof_max_days)

    if not np.all(np.isfinite(tightened_bounds_days)):
        raise ValueError(
            "Encounter stage bounds remain infinite after tightening; provide finite bounds in example config."
        )

    stage_configs: List[EncounterStageConfig] = []
    for stage_id, stage_bodies in enumerate(bodies_by_stage):
        dt_days = float(dt_days_by_stage[stage_id])
        if dt_days <= 0.0:
            raise ValueError(f"dt must be positive days for stage {stage_id}, got {dt_days}.")

        t_min_days = float(tightened_bounds_days[stage_id, 0])
        t_max_days = float(tightened_bounds_days[stage_id, 1])
        amin_default_km = float(alt_km_by_stage[stage_id])
        amin_km_map = {int(body_id): amin_default_km for body_id in stage_bodies}

        stage_configs.append(
            EncounterStageConfig(
                stage_id=int(stage_id),
                bodies=[int(body_id) for body_id in stage_bodies],
                t_min_et=j2000_days_to_et_seconds(t_min_days),
                t_max_et=j2000_days_to_et_seconds(t_max_days),
                dt_et=j2000_days_to_et_seconds(dt_days),
                amin_km=amin_km_map,
            )
        )

    config = EncounterDBConfig(
        stages=stage_configs,
        frame=DEFAULT_FRAME,
        observer=DEFAULT_OBSERVER,
    )

    encounter_db = build_encounter_database(config)

    print("EncounterDB Build Summary")
    print(f"metakernel       : {metakernel_path}")
    print(f"num_entries      : {len(encounter_db.entries)}")
    print("units            : r[km], v[km/s], t[J2000 days]")
    print()
    print(
        f"{'IE':>6} {'stage':>6} {'body':>6} {'t_day':>12} "
        f"{'rx_km':>14} {'ry_km':>14} {'rz_km':>14} "
        f"{'vx_km_s':>12} {'vy_km_s':>12} {'vz_km_s':>12} "
        f"{'mu_km3_s2':>14} {'rmin_km':>10}"
    )
    print("-" * 148)

    for entry in encounter_db.entries[:100]:
        t_days = et_seconds_to_j2000_days(entry.t_et)
        print(
            f"{entry.IE:6d} {entry.stage_id:6d} {entry.body:6d} {t_days:12.4f} "
            f"{entry.r_km[0]:14.3f} {entry.r_km[1]:14.3f} {entry.r_km[2]:14.3f} "
            f"{entry.v_km_s[0]:12.6f} {entry.v_km_s[1]:12.6f} {entry.v_km_s[2]:12.6f} "
            f"{entry.mu_km3_s2:14.6e} {entry.rmin_km:10.3f}"
        )

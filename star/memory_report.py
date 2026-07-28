"""Preflight memory and row-count report for STAR problems."""

from __future__ import annotations

import argparse
import importlib
from pathlib import Path
from typing import Sequence

import numpy as np
import spiceypy as spice

from star.constants.time import j2000_days_to_et_seconds
from star.encounter_database import (
    EncounterDBConfig,
    EncounterStageConfig,
    build_encounter_database,
    tighten_stage_time_bounds_days,
)
from star.leg_database import _collect_stage_arrays, _prepare_candidate_pair_counts


# Current LegDatabase numeric columns after dropping redundant tD_et/tA_et,
# dt_s, leveraged endpoint velocities, and h_leg.
LEG_NUMERIC_BYTES_PER_ROW = 96
# No smaller compact variant is currently modeled.
LEG_NUMERIC_BYTES_PER_ROW_COMPACT = 96


def _import_problem_module(problem_name: str):
    name = str(problem_name).strip()
    if not name:
        raise ValueError("Problem module name must be non-empty.")
    if "." in name:
        return importlib.import_module(name)
    return importlib.import_module(f"example.{name}")


def _resolve_per_leg_ints(value: object, n_legs: int, default: int) -> list[int]:
    if value is None:
        return [int(default)] * int(n_legs)
    if np.isscalar(value):
        return [int(value)] * int(n_legs)
    values = list(value)
    return [int(values[i]) for i in range(int(n_legs))]


def _resolve_per_leg_floats(value: object, n_legs: int, default: float) -> list[float]:
    if value is None:
        return [float(default)] * int(n_legs)
    if np.isscalar(value):
        return [float(value)] * int(n_legs)
    values = list(value)
    return [float(values[i]) for i in range(int(n_legs))]


def _resolve_per_leg_bools(value: object, n_legs: int, default: bool) -> list[bool]:
    if value is None:
        return [bool(default)] * int(n_legs)
    if np.isscalar(value):
        return [bool(value)] * int(n_legs)
    values = list(value)
    return [bool(values[i]) for i in range(int(n_legs))]


def _resolve_lev_type_count(value: object) -> int:
    if value is None:
        return 2
    if isinstance(value, str):
        return 1
    return max(1, len(list(value)))


def _build_encounter_db(problem_module) -> tuple[object, np.ndarray, np.ndarray]:
    bodies_by_stage = [list(stage_bodies) for stage_bodies in problem_module.Body]
    time_bounds_days = np.asarray(problem_module.Time, dtype=float)
    tof_min_days = np.asarray(problem_module.tof_min, dtype=float)
    tof_max_days = np.asarray(problem_module.tof_max, dtype=float)
    dt_days_by_stage = np.asarray(problem_module.dt, dtype=float)
    alt_km_by_stage = list(problem_module.alt)

    tightened_bounds_days = tighten_stage_time_bounds_days(time_bounds_days, tof_min_days, tof_max_days)

    stage_configs = []
    for stage_id, stage_bodies in enumerate(bodies_by_stage):
        alt_value = alt_km_by_stage[stage_id]
        if isinstance(alt_value, dict):
            amin_km = {int(body_id): float(alt_value[body_id]) for body_id in stage_bodies}
        else:
            amin_default_km = float(alt_value)
            amin_km = {int(body_id): amin_default_km for body_id in stage_bodies}
        stage_configs.append(
            EncounterStageConfig(
                stage_id=int(stage_id),
                bodies=[int(body_id) for body_id in stage_bodies],
                t_min_et=j2000_days_to_et_seconds(float(tightened_bounds_days[stage_id, 0])),
                t_max_et=j2000_days_to_et_seconds(float(tightened_bounds_days[stage_id, 1])),
                dt_et=j2000_days_to_et_seconds(float(dt_days_by_stage[stage_id])),
                amin_km=amin_km,
            )
        )

    return (
        build_encounter_database(EncounterDBConfig(stages=stage_configs)),
        tof_min_days,
        tof_max_days,
    )


def _format_gb(num_bytes: float) -> str:
    return f"{num_bytes / (1024 ** 3):.2f} GB"


def summarize_problem(problem_name: str, metakernel: str | Path) -> None:
    problem = _import_problem_module(problem_name)
    metakernel_path = Path(metakernel)
    if not metakernel_path.is_absolute():
        metakernel_path = Path.cwd() / metakernel_path

    spice.kclear()
    spice.furnsh(str(metakernel_path))

    encounter_db, tof_min_days, tof_max_days = _build_encounter_db(problem)
    n_stages = len(problem.Body)
    n_legs = n_stages - 1

    lambert_nrev_raw = getattr(problem, "Nrev", None)
    if lambert_nrev_raw is None:
        lambert_nrev_raw = getattr(problem, "lambert_nrev", None)
    lambert_hz_raw = getattr(problem, "lambert_hz", None)
    dvlev_max_raw = getattr(problem, "dVlev_max", None)
    if dvlev_max_raw is None:
        dvlev_max_raw = getattr(problem, "dvlev_max", None)
    delta_dvlev_raw = getattr(problem, "delta_dvlev", None)
    resonant_raw = getattr(problem, "resonant", None)
    if resonant_raw is None:
        resonant_raw = getattr(problem, "resonant_enabled", None)

    lambert_nrev = _resolve_per_leg_ints(lambert_nrev_raw, n_legs, 0)
    lambert_hz = _resolve_per_leg_ints(lambert_hz_raw, n_legs, 1)
    dvlev_max = _resolve_per_leg_floats(dvlev_max_raw, n_legs, 0.0)
    delta_dvlev = _resolve_per_leg_floats(delta_dvlev_raw, n_legs, 0.0)
    resonant_enabled = _resolve_per_leg_bools(resonant_raw, n_legs, False)
    lev_mode_count = _resolve_lev_type_count(getattr(problem, "lev_type", None))

    print("EncounterDB")
    print(f"  entries: {len(encounter_db.entries)}")
    for stage_id in range(n_stages):
        count = sum(1 for entry in encounter_db.entries if int(entry.stage_id) == stage_id)
        bodies = sorted({int(entry.body) for entry in encounter_db.entries if int(entry.stage_id) == stage_id})
        print(f"  stage {stage_id:2d}: entries={count:6d} bodies={bodies}")

    print()
    print("Leg Upper Bounds")
    print(
        "  note: the storage estimates below are numeric-column lower bounds before "
        "row-index maps and other Python overhead."
    )

    total_upper_rows = 0
    for leg_index in range(n_legs):
        dep = _collect_stage_arrays(encounter_db, leg_index)
        arr = _collect_stage_arrays(encounter_db, leg_index + 1)
        # Counting pass only: we need the pair count, not the pairs themselves.
        *_, pair_counts = _prepare_candidate_pair_counts(
            dep["t_et"],
            arr["t_et"],
            j2000_days_to_et_seconds(float(tof_min_days[leg_index, leg_index + 1])),
            j2000_days_to_et_seconds(float(tof_max_days[leg_index, leg_index + 1])),
            1e-6,
        )

        candidate_pairs = int(np.sum(pair_counts))
        nrev_max = int(lambert_nrev[leg_index])
        hz = int(lambert_hz[leg_index])
        lambert_multiplier = (2 + 4 * nrev_max) if hz == 0 else (1 + 2 * nrev_max)

        dv_count = 0
        if float(dvlev_max[leg_index]) > 0.0 and float(delta_dvlev[leg_index]) > 0.0:
            dv_count = int(np.floor(float(dvlev_max[leg_index]) / float(delta_dvlev[leg_index]) + 1.0e-12))
            if dv_count <= 0:
                dv_count = 1
        dsm_multiplier = 1 + lev_mode_count * dv_count

        upper_rows = candidate_pairs * lambert_multiplier * dsm_multiplier
        total_upper_rows += upper_rows

        print(
            f"  leg {leg_index:2d}: pairs={candidate_pairs:9d} "
            f"lambert_x={lambert_multiplier:2d} dsm_x={dsm_multiplier:2d} "
            f"upper_rows={upper_rows:12d} "
            f"rows_numeric={_format_gb(upper_rows * LEG_NUMERIC_BYTES_PER_ROW)} "
            f"rows_compact={_format_gb(upper_rows * LEG_NUMERIC_BYTES_PER_ROW_COMPACT)} "
            f"resonant={bool(resonant_enabled[leg_index])}"
        )

    print()
    print("Whole-Problem Lower Bound")
    print(f"  summed upper rows (no resonant rows added): {total_upper_rows}")
    print(f"  current numeric-only storage: {_format_gb(total_upper_rows * LEG_NUMERIC_BYTES_PER_ROW)}")
    print(f"  compact numeric-only storage: {_format_gb(total_upper_rows * LEG_NUMERIC_BYTES_PER_ROW_COMPACT)}")
    print("  practical note: peak RAM is higher than this during Lambert/DSM expansion.")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Estimate STAR memory pressure for a problem module.")
    parser.add_argument("--problem", default="bepicolombo", help="Problem module name or dotted import path.")
    parser.add_argument("--metakernel", default="star/METAKERN.tm", help="SPICE meta-kernel path.")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)
    summarize_problem(args.problem, args.metakernel)


if __name__ == "__main__":
    main()

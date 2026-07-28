"""Main STAR trajectory-search entrypoint.

Pipeline (STAR §5.1 interleaved ordering):
1) build encounter database,
2) interleaved leg + flyby + filter loop:
   - rank unbuilt legs by estimated cost c_i = |E[i]| x |E[i+1]| x weight_i,
   - build cheapest leg, apply encounter filter for newly safe stages,
   - as soon as two adjacent legs exist, build that flyby and run leg filters
     to fixpoint, then re-apply encounter filter,
3) final leg-filter convergence pass,
4) run combo stitching,
5) optionally apply final-output tfilter decimation,
6) save minimal JSONL solutions.

See NOTATION.md for the index symbols (IE/IL/IF), stage kinds, and the
tfilter / null-leg / leg-filter-fixpoint vocabulary used below.
"""

from __future__ import annotations

import argparse
import importlib
import gc
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, Optional, Sequence, Tuple

import numpy as np
import spiceypy as spice

root_folder = Path.cwd()

from star.combo import ComboBuildConfig, run_combo, save_solution, tfilter_output_db
from star.constants import DEFAULT_FRAME, DEFAULT_OBSERVER, get_gm
from star.constants.time import et_seconds_to_j2000_days, j2000_days_to_et_seconds
from star.encounter_database import (
    EncounterDB,
    EncounterDBConfig,
    EncounterStageConfig,
    build_encounter_database,
    tighten_stage_time_bounds_days,
)
from star.flyby_database import (
    ActiveIndexCache,
    add_flyby_stage_to_active_index_cache,
    add_leg_stage_to_active_index_cache,
    create_active_index_cache,
    FlybyDB,
    FlybyBuildConfig,
    build_flyby_stage_to_disk,
    encounter_filter_from_active_legs,
    flush_active_index_cache_masks,
    incoming_leg_filter_disk,
    outgoing_leg_filter_disk,
)
from star.leg_database import LegBuildConfig, LegDatabase, build_leg_db, build_leg_stage_to_disk
from star.stage_db_npy import (
    count_active_rows,
    load_leg_stage,
    load_leg_null_binding_sidecar,
    load_leg_stage_for_flyby,
    save_leg_null_binding_sidecar,
    save_leg_stage,
)


@dataclass(frozen=True)
class StarRunPaths:
    """Filesystem paths and mode flags for one STAR run."""

    metakernel_path: Path
    output_path: Path
    cache_root: Path
    resume_phase2_only: bool


@dataclass(frozen=True)
class StarProblemConfig:
    """Normalized problem inputs used by the STAR pipeline."""

    observer: str
    encounter_cfg: EncounterDBConfig
    flyby_cfg: FlybyBuildConfig
    combo_dv_cap_km_s: Optional[float]
    max_save_sol: Optional[int]
    tfilter_dt_s: Optional[np.ndarray]
    dt_filter_preemptive: bool
    n_stages: int
    n_legs: int
    combo_num_legs: int
    null_flags: list[bool]
    tof_min_days: np.ndarray
    tof_max_days: np.ndarray
    vlim: np.ndarray
    lambert_nrev: list[int]
    lambert_hz: list[int]
    dvlev_max_km_s: list[float]
    delta_dvlev_km_s: list[float]
    vinf_margin_pre_filter_km_s: list[float]
    lev_type: Tuple[str, ...]
    resonant_enabled: list[bool]
    resonant_dvinf_km_s: list[float]
    resonant_angle_tol_deg: list[float]
    resonant_crank_angles_by_leg: list[Tuple[float, ...]]
    leg_parallelism: Optional[int]
    leg_filter: Optional[Callable[..., np.ndarray]]


@dataclass
class Phase1State:
    """Mutable working set for STAR phase 1."""

    encounter_db: EncounterDB
    unbuilt_legs: set[int]
    leg_stage_dirs: Dict[int, Path]
    flyby_stage_dirs: Dict[int, Path]
    if_offset: int
    null_stage_ie_maps: Dict[int, Tuple[Dict[int, int], Dict[int, int]]]
    null_runs_by_start: Dict[int, "_NullRunState"]
    null_run_start_by_leg: Dict[int, int]
    leg_build_order: Dict[int, int]
    next_leg_build_order: int
    stage_arrays_cache: Dict[int, Dict[str, np.ndarray]]
    active_leg_ie_cache: Dict[int, Dict[str, np.ndarray]]
    active_index_cache: ActiveIndexCache


@dataclass
class _NullRunState:
    """Contiguous null-leg run with one frozen copy direction."""

    start_leg: int
    end_leg: int
    anchor_direction: Optional[str] = None


def _first_defined_attr(problem_module: object, *names: str) -> object:
    """Return the first non-None problem attribute from `names`."""

    for name in names:
        value = getattr(problem_module, name, None)
        if value is not None:
            return value
    return None


def _resolve_problem_frame_observer(problem_module: object) -> Tuple[str, str]:
    """Resolve propagation frame and observer from problem module with defaults."""

    frame = str(getattr(problem_module, "frame", DEFAULT_FRAME)).strip()
    observer = str(getattr(problem_module, "observer", DEFAULT_OBSERVER)).strip()
    return frame, observer


def _resolve_combo_dv_cap_km_s(problem_module: object) -> Optional[float]:
    """Resolve optional combo-stage total-DV cap [km/s] from a problem module."""

    raw_value = _first_defined_attr(problem_module, "dv_total_max_km_s", "dVtotal")
    if raw_value is None:
        return None

    try:
        dv_cap_km_s = float(raw_value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "Problem dv_total_max_km_s/dVtotal must be a finite non-negative float [km/s]."
        ) from exc

    if not np.isfinite(dv_cap_km_s) or dv_cap_km_s < 0.0:
        raise ValueError(
            "Problem dv_total_max_km_s/dVtotal must be a finite non-negative float [km/s]."
        )
    return dv_cap_km_s


def _resolve_max_save_sol(problem_module: object) -> Optional[int]:
    """Resolve optional saved-solution cap from a problem module."""

    raw_value = getattr(problem_module, "max_save_sol", None)
    if raw_value is None:
        return None

    try:
        max_save_sol = int(raw_value)
    except (TypeError, ValueError) as exc:
        raise ValueError("Problem max_save_sol must be a positive integer when provided.") from exc

    if max_save_sol <= 0:
        raise ValueError("Problem max_save_sol must be a positive integer when provided.")
    return max_save_sol


def _import_problem_module(problem_name: str) -> object:
    """Import one problem module.

    Bare names keep the historical behavior of resolving under `example.*`.
    Dotted module paths are imported as provided so configs can live elsewhere,
    e.g. `experiment.bepicolombo`.
    """

    name = str(problem_name).strip()
    if not name:
        raise ValueError("Problem module name must be non-empty.")

    if "." in name:
        return importlib.import_module(name)
    return importlib.import_module(f"example.{name}")


def _resolve_observer_naif_id(observer: str) -> int:
    """Resolve observer body name/ID to NAIF integer code."""

    observer_text = str(observer).strip()
    if not observer_text:
        raise ValueError("Observer must be a non-empty SPICE body name/NAIF ID.")

    try:
        resolved = spice.bods2c(observer_text)
    except Exception as exc:
        raise ValueError(f"Unable to resolve observer '{observer_text}' to a NAIF body ID.") from exc

    return int(resolved)


def _resolve_lambert_mu_km3_s2(observer: str) -> float:
    """Resolve Lambert two-body GM from the selected observer/central body."""

    observer_id = _resolve_observer_naif_id(observer)
    try:
        mu_km3_s2 = float(get_gm(str(observer_id))[0])
    except Exception as exc:
        raise ValueError(
            f"No GM constant available for observer '{observer}' (NAIF ID {observer_id}). "
            "Choose a central body with defined GM (e.g., SUN=10, JUPITER=599)."
        ) from exc

    if not np.isfinite(mu_km3_s2) or mu_km3_s2 <= 0.0:
        raise ValueError(f"Resolved non-positive/invalid GM for observer '{observer}' (NAIF ID {observer_id}).")
    return mu_km3_s2


def build_arg_parser() -> argparse.ArgumentParser:
    """Build CLI arguments for the STAR pipeline driver."""

    parser = argparse.ArgumentParser(description="Run STAR patched-conic trajectory search pipeline.")
    parser.add_argument(
        "--metakernel",
        default="star/METAKERN.tm",
        help="SPICE meta-kernel path (default: star/METAKERN.tm).",
    )
    parser.add_argument(
        "--problem",
        default="earth_to_jupiter",
        help="Problem module path. Bare names resolve under example.*, dotted paths import as-is.",
    )
    parser.add_argument(
        "--resume_cache",
        default=None,
        help=(
            "Optional existing cache run directory for phase-2 resume. "
            "Accepts run tag (under output/.star_cache) or path."
        ),
    )
    return parser


def _resolve_tfilter_dt_s(
    problem_module: object,
    *,
    num_encounters: int,
) -> Optional[np.ndarray]:
    """Resolve optional final-output tfilter bin size [seconds] from `problem.dt_filter` [days].

    Accepts:
    - scalar days value (broadcast to all encounters), or
    - length-`num_encounters` days array.
    """

    raw_days = getattr(problem_module, "dt_filter", None)
    if raw_days is None:
        return None

    n_enc = int(num_encounters)
    if n_enc <= 0:
        raise ValueError("num_encounters must be positive when resolving problem.dt_filter.")

    if np.isscalar(raw_days):
        dt_days = np.full(n_enc, float(raw_days), dtype=float)
    else:
        dt_days = np.asarray(list(raw_days), dtype=float).reshape(-1)
        if int(dt_days.size) != n_enc:
            raise ValueError(
                f"problem.dt_filter must be scalar or length {n_enc} [days]. Got length {int(dt_days.size)}."
            )

    if np.any(~np.isfinite(dt_days)) or np.any(dt_days <= 0.0):
        raise ValueError("problem.dt_filter must contain finite values > 0 [days].")

    return np.asarray(dt_days, dtype=float) * float(j2000_days_to_et_seconds(1.0))


def _resolve_tfilter_preemptive(problem_module: object) -> bool:
    """Resolve optional stage-wise preemptive dt-filter flag from `problem.dt_filter_preemptive`."""

    raw_value = getattr(problem_module, "dt_filter_preemptive", False)
    if raw_value is None:
        return False
    if isinstance(raw_value, (bool, np.bool_)):
        return bool(raw_value)
    if isinstance(raw_value, (int, np.integer)):
        return bool(int(raw_value))
    if isinstance(raw_value, str):
        text = raw_value.strip().lower()
        if text in {"1", "true", "yes", "y", "on"}:
            return True
        if text in {"0", "false", "no", "n", "off", ""}:
            return False
        raise ValueError("problem.dt_filter_preemptive must be a boolean-like value.")
    raise ValueError("problem.dt_filter_preemptive must be a boolean-like value.")


def _resolve_per_leg_setting(value: object, n_legs: int, default_value: int, name: str) -> list[int]:
    """Resolve a scalar or sequence setting to a per-leg integer list."""

    if value is None:
        return [int(default_value)] * int(n_legs)

    if np.isscalar(value):
        return [int(value)] * int(n_legs)

    values = list(value)
    if len(values) < int(n_legs):
        raise ValueError(f"{name} must provide at least {n_legs} entries when given as a sequence.")
    return [int(values[i]) for i in range(int(n_legs))]


def _resolve_per_leg_float_setting(value: object, n_legs: int, default_value: float, name: str) -> list[float]:
    """Resolve a scalar or sequence setting to a per-leg float list."""

    if value is None:
        return [float(default_value)] * int(n_legs)

    if np.isscalar(value):
        return [float(value)] * int(n_legs)

    values = list(value)
    if len(values) < int(n_legs):
        raise ValueError(f"{name} must provide at least {n_legs} entries when given as a sequence.")
    return [float(values[i]) for i in range(int(n_legs))]


def _resolve_per_leg_bool_setting(value: object, n_legs: int, default_value: bool, name: str) -> list[bool]:
    """Resolve a scalar or sequence setting to a per-leg boolean list."""

    if value is None:
        return [bool(default_value)] * int(n_legs)

    if np.isscalar(value):
        return [bool(value)] * int(n_legs)

    values = list(value)
    if len(values) < int(n_legs):
        raise ValueError(f"{name} must provide at least {n_legs} entries when given as a sequence.")
    return [bool(values[i]) for i in range(int(n_legs))]


def _resolve_null_flags(problem_module: object, n_legs: int) -> list[bool]:
    """Resolve per-leg null-leg flags from `problem.null`."""

    raw_value = getattr(problem_module, "null", None)
    if raw_value is None:
        return [False] * int(n_legs)

    values = list(raw_value)
    if len(values) != int(n_legs):
        raise ValueError(f"problem.null must contain exactly {n_legs} boolean entries.")
    return [bool(values[i]) for i in range(int(n_legs))]


def _resolve_per_leg_angle_lists(value: object, n_legs: int, name: str) -> list[Tuple[float, ...]]:
    """Resolve flat or per-leg crank-angle settings."""

    def _normalize_angle_list(raw: object) -> Tuple[float, ...]:
        if raw is None:
            return ()
        values = np.asarray(raw, dtype=float).reshape(-1)
        return tuple(float(entry) for entry in values)

    if value is None:
        return [()] * int(n_legs)

    if isinstance(value, dict):
        resolved = [()] * int(n_legs)
        for leg_stage_raw, crank_values in value.items():
            leg_stage = int(leg_stage_raw)
            if leg_stage < 0 or leg_stage >= int(n_legs):
                raise ValueError(f"{name} contains invalid leg stage {leg_stage}.")
            resolved[leg_stage] = _normalize_angle_list(crank_values)
        return resolved

    values = list(value)
    if len(values) == 0:
        return [()] * int(n_legs)
    if all(np.isscalar(entry) for entry in values):
        normalized = tuple(float(entry) for entry in values)
        return [normalized] * int(n_legs)
    if len(values) < int(n_legs):
        raise ValueError(f"{name} must provide at least {n_legs} entries when given per leg.")
    return [_normalize_angle_list(values[i]) for i in range(int(n_legs))]


def _resolve_lev_type(value: object) -> Tuple[str, ...]:
    """Resolve global leveraging mode selection for solve_arc objective type 1."""

    allowed = {"+-", "-+", "--", "++"}
    if value is None:
        values = ["+-", "-+", "--", "++"]
    elif isinstance(value, str):
        values = [value]
    else:
        values = list(value)

    normalized: list[str] = []
    for entry in values:
        mode = str(entry).strip()
        if mode not in allowed:
            raise ValueError("lev_type must contain only '+-' and/or '-+'.")
        if mode not in normalized:
            normalized.append(mode)

    if not normalized:
        raise ValueError("lev_type must contain at least one entry.")
    return tuple(normalized)


def _resolve_dvfb_max_by_flyby_stage(
    value: object,
    *,
    n_stages: int,
    n_legs: int,
) -> Dict[int, float]:
    """Resolve optional local flyby patch-DV limits to `{flyby_stage_id: limit_km_s}`.

    Accepted inputs:
    - scalar: one limit for all flyby stages.
    - dict: explicit `{stage_id: limit}` map (stage IDs are encounter stage indices).
    - sequence length `n_stages-2`: per flyby stage in ascending stage order.
    - sequence length `n_stages`: per encounter stage; flyby stages use entries `1..n_stages-2`.
    - sequence length `n_legs`: per leg; `value[k]` applies to flyby after leg `k`,
      i.e., flyby stage `k+1` (the last leg entry is unused because there is no flyby after the final leg).
    """

    flyby_stages = list(range(1, int(n_stages) - 1))
    num_flybys = len(flyby_stages)
    if value is None or num_flybys <= 0:
        return {}

    def _normalize_limit(raw: object) -> float | None:
        limit = float(raw)
        if np.isnan(limit) or limit < 0.0:
            raise ValueError("dVfb_max values must be >= 0 or +inf.")
        if np.isinf(limit):
            return None
        return limit

    out: Dict[int, float] = {}
    if np.isscalar(value):
        normalized = _normalize_limit(value)
        if normalized is None:
            return {}
        for stage_id in flyby_stages:
            out[int(stage_id)] = float(normalized)
        return out

    if isinstance(value, dict):
        for stage_raw, limit_raw in value.items():
            stage_id = int(stage_raw)
            if stage_id not in flyby_stages:
                raise ValueError(
                    f"dVfb_max dict key {stage_id} is invalid for this problem; "
                    f"valid flyby stages are {flyby_stages}."
                )
            normalized = _normalize_limit(limit_raw)
            if normalized is not None:
                out[stage_id] = float(normalized)
        return out

    values = list(value)
    length = len(values)
    if length == num_flybys:
        for offset, stage_id in enumerate(flyby_stages):
            normalized = _normalize_limit(values[offset])
            if normalized is not None:
                out[int(stage_id)] = float(normalized)
        return out

    if length == int(n_stages):
        for stage_id in flyby_stages:
            normalized = _normalize_limit(values[stage_id])
            if normalized is not None:
                out[int(stage_id)] = float(normalized)
        return out

    if length == int(n_legs):
        for leg_stage in range(num_flybys):
            stage_id = leg_stage + 1
            normalized = _normalize_limit(values[leg_stage])
            if normalized is not None:
                out[int(stage_id)] = float(normalized)
        return out

    raise ValueError(
        f"dVfb_max has unsupported length {length}. Expected scalar, dict, or sequence length "
        f"{num_flybys} (flybys), {n_stages} (encounters), or {n_legs} (legs)."
    )


def _resolve_resume_cache_root(raw_value: object, repo_root: Path) -> Optional[Path]:
    """Resolve optional phase-2 resume cache directory."""

    if raw_value is None:
        return None

    raw_text = str(raw_value).strip()
    if not raw_text:
        raise ValueError("--resume_cache must be a non-empty run tag or path.")

    candidate = Path(raw_text)
    search_paths = [candidate] if candidate.is_absolute() else [
        repo_root / "output" / ".star_cache" / candidate,
        repo_root / candidate,
    ]
    for path in search_paths:
        if path.exists() and path.is_dir():
            return path

    searched = ", ".join(str(path) for path in search_paths)
    raise FileNotFoundError(f"Resume cache directory not found. Checked: {searched}")


def _discover_stage_dirs_from_cache(cache_root: Path) -> Tuple[Dict[int, Path], Dict[int, Path]]:
    """Discover cached leg/flyby stage directories under one cache root."""

    leg_stage_dirs: Dict[int, Path] = {}
    for stage_dir in sorted(cache_root.glob("leg_*")):
        if not stage_dir.is_dir():
            continue
        try:
            stage_id = int(stage_dir.name.split("_", 1)[1])
        except Exception:
            continue
        leg_stage_dirs[int(stage_id)] = stage_dir

    flyby_stage_dirs: Dict[int, Path] = {}
    flyby_root = cache_root / "flybys"
    if flyby_root.exists():
        for stage_dir in sorted(flyby_root.glob("flyby_*")):
            if not stage_dir.is_dir():
                continue
            try:
                stage_id = int(stage_dir.name.split("_", 1)[1])
            except Exception:
                continue
            flyby_stage_dirs[int(stage_id)] = stage_dir

    if not leg_stage_dirs:
        raise ValueError(f"No leg_* stage directories found in resume cache: {cache_root}")
    if not flyby_stage_dirs:
        raise ValueError(f"No flyby_* stage directories found in resume cache: {flyby_root}")

    return leg_stage_dirs, flyby_stage_dirs


def _subset_output_db_rows(output_db: "OutputDB", keep_rows: np.ndarray) -> "OutputDB":
    """Stable row-subset of OutputDB by explicit indices."""

    rows = np.asarray(keep_rows, dtype=np.int64).reshape(-1)
    return type(output_db)(
        I_traj=np.asarray(output_db.I_traj, dtype=np.int64)[rows],
        encounter_IEs=np.asarray(output_db.encounter_IEs, dtype=np.int64)[rows],
        body_ids=np.asarray(output_db.body_ids, dtype=np.int64)[rows],
        times_et_s=np.asarray(output_db.times_et_s, dtype=float)[rows],
        leg_ILs=np.asarray(output_db.leg_ILs, dtype=np.int64)[rows],
        flyby_IFs=np.asarray(output_db.flyby_IFs, dtype=np.int64)[rows],
        dv_total_km_s=np.asarray(output_db.dv_total_km_s, dtype=float)[rows],
        tof_total_s=np.asarray(output_db.tof_total_s, dtype=float)[rows],
        per_leg_tof_s=np.asarray(output_db.per_leg_tof_s, dtype=float)[rows],
    )


def _apply_max_save_sol(output_db: "OutputDB", max_save_sol: Optional[int]) -> "OutputDB":
    """Apply optional save-cap by keeping lowest-dv trajectories."""

    if max_save_sol is None:
        return output_db

    max_keep = int(max_save_sol)
    if max_keep <= 0:
        raise ValueError("max_save_sol must be > 0 when provided.")

    num_rows = int(np.asarray(output_db.I_traj, dtype=np.int64).size)
    if num_rows <= max_keep:
        return output_db

    dv = np.asarray(output_db.dv_total_km_s, dtype=float).reshape(-1)
    if int(dv.size) != num_rows:
        raise ValueError("OutputDB dv_total_km_s row count mismatch while applying max_save_sol.")

    selected = np.argpartition(dv, max_keep - 1)[:max_keep]
    keep_rows = selected[np.argsort(dv[selected], kind="mergesort")]
    dv_cutoff = float(np.max(dv[keep_rows]))
    print(
        f"Applying max_save_sol={max_keep}: "
        f"keeping {max_keep}/{num_rows} lowest-dv trajectories "
        f"(dv_cutoff_km_s={dv_cutoff})."
    )
    return _subset_output_db_rows(output_db, keep_rows)


def _resolve_run_paths(args: argparse.Namespace, repo_root: Path) -> StarRunPaths:
    """Resolve run filesystem paths and initialize cache directories when needed."""

    metakernel_path = Path(args.metakernel)
    if not metakernel_path.is_absolute():
        metakernel_path = repo_root / metakernel_path
    if not metakernel_path.exists():
        raise FileNotFoundError(f"Meta-kernel not found: {metakernel_path}")

    output_path = repo_root / "output" / f"{args.problem}.jsonl"
    resume_cache_root = _resolve_resume_cache_root(getattr(args, "resume_cache", None), repo_root)
    if resume_cache_root is not None:
        return StarRunPaths(
            metakernel_path=metakernel_path,
            output_path=output_path,
            cache_root=resume_cache_root,
            resume_phase2_only=True,
        )

    run_tag = f"{str(args.problem).replace('.', '_')}_{int(os.getpid())}"
    cache_root = repo_root / "output" / ".star_cache" / run_tag
    if cache_root.exists():
        shutil.rmtree(cache_root, ignore_errors=True)
    cache_root.mkdir(parents=True, exist_ok=True)
    return StarRunPaths(
        metakernel_path=metakernel_path,
        output_path=output_path,
        cache_root=cache_root,
        resume_phase2_only=False,
    )


def _load_spice_metakernel(metakernel_path: Path) -> None:
    """Load the SPICE meta-kernel for all downstream ephemeris queries."""

    spice.kclear()
    spice.furnsh(str(metakernel_path))
    print(f"Loaded SPICE metakernel: {metakernel_path}")


def _normalize_leg_parallelism(problem_module: object) -> Optional[int]:
    """Resolve optional leg-build worker count; values <= 1 disable multiprocessing."""

    raw_value = getattr(problem_module, "n_process", None)
    if raw_value is None:
        return None

    parallelism = int(raw_value)
    return None if parallelism <= 1 else parallelism


def _resolve_trip_tof_bounds_s_by_stage(
    tof_min_days: np.ndarray,
    tof_max_days: np.ndarray,
    *,
    n_stages: int,
) -> Dict[int, Tuple[float, float]]:
    """Resolve optional triplet TOF bounds for flyby-stage filtering."""

    trip_tof_bounds_s_by_stage: Dict[int, Tuple[float, float]] = {}
    for stage_id in range(1, int(n_stages) - 1):
        trip_tof_min_days = float(tof_min_days[stage_id - 1, stage_id + 1])
        trip_tof_max_days = float(tof_max_days[stage_id - 1, stage_id + 1])
        if np.isfinite(trip_tof_min_days) or np.isfinite(trip_tof_max_days):
            trip_tof_bounds_s_by_stage[int(stage_id)] = (
                float(j2000_days_to_et_seconds(trip_tof_min_days)),
                float(j2000_days_to_et_seconds(trip_tof_max_days)),
            )
    return trip_tof_bounds_s_by_stage


def _build_encounter_cfg(
    *,
    bodies_by_stage: list[list[int]],
    time_bounds_days: np.ndarray,
    tof_min_days: np.ndarray,
    tof_max_days: np.ndarray,
    dt_days_by_stage: np.ndarray,
    alt_km_by_stage: list[object],
    frame: str,
    observer: str,
) -> EncounterDBConfig:
    """Build normalized EncounterDBConfig from problem arrays."""

    tightened_bounds_days = tighten_stage_time_bounds_days(time_bounds_days, tof_min_days, tof_max_days)
    if not np.all(np.isfinite(tightened_bounds_days)):
        raise ValueError("Encounter stage bounds remain infinite after tightening.")

    stage_configs: list[EncounterStageConfig] = []
    for stage_id, stage_bodies in enumerate(bodies_by_stage):
        dt_days = float(dt_days_by_stage[stage_id])
        if dt_days <= 0.0:
            raise ValueError(f"dt must be positive days for stage {stage_id}, got {dt_days}.")

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
                dt_et=j2000_days_to_et_seconds(dt_days),
                amin_km=amin_km,
            )
        )

    return EncounterDBConfig(stages=stage_configs, frame=frame, observer=observer)


def _resolve_problem_config(problem_module: object) -> StarProblemConfig:
    """Normalize one problem module into explicit pipeline configuration."""

    frame, observer = _resolve_problem_frame_observer(problem_module)
    combo_dv_cap_km_s = _resolve_combo_dv_cap_km_s(problem_module)
    max_save_sol = _resolve_max_save_sol(problem_module)

    bodies_by_stage = [list(stage_bodies) for stage_bodies in problem_module.Body]
    time_bounds_days = np.asarray(problem_module.Time, dtype=float)
    tof_min_days = np.asarray(problem_module.tof_min, dtype=float)
    tof_max_days = np.asarray(problem_module.tof_max, dtype=float)
    dt_days_by_stage = np.asarray(problem_module.dt, dtype=float)
    alt_km_by_stage = list(problem_module.alt)
    vlim = np.asarray(problem_module.vlim, dtype=float)

    n_stages = len(bodies_by_stage)
    if n_stages < 2:
        raise ValueError("Problem must define at least 2 encounters.")
    n_legs = int(n_stages - 1)
    combo_num_legs = int(problem_module.nL)
    null_flags = _resolve_null_flags(problem_module, n_legs)

    dt_filter_preemptive = _resolve_tfilter_preemptive(problem_module)
    tfilter_dt_s = _resolve_tfilter_dt_s(problem_module, num_encounters=int(n_stages))
    if dt_filter_preemptive and tfilter_dt_s is None:
        raise ValueError("problem.dt_filter_preemptive=True requires problem.dt_filter > 0 [days].")

    lambert_nrev = _resolve_per_leg_setting(
        _first_defined_attr(problem_module, "Nrev", "lambert_nrev"),
        n_legs,
        0,
        "lambert_nrev/Nrev",
    )
    lambert_hz = _resolve_per_leg_setting(
        getattr(problem_module, "lambert_hz", None),
        n_legs,
        1,
        "lambert_hz",
    )
    dvlev_max_km_s = _resolve_per_leg_float_setting(
        _first_defined_attr(problem_module, "dVlev_max", "dvlev_max"),
        n_legs,
        0.0,
        "dVlev_max/dvlev_max",
    )
    delta_dvlev_km_s = _resolve_per_leg_float_setting(
        getattr(problem_module, "delta_dvlev", None),
        n_legs,
        0.0,
        "delta_dvlev",
    )
    vinf_margin_pre_filter_km_s = _resolve_per_leg_float_setting(
        _first_defined_attr(problem_module, "vinf_margin_pre_filter", "vinf_margin_pre_filter_km_s"),
        n_legs,
        10.0,
        "vinf_margin_pre_filter/vinf_margin_pre_filter_km_s",
    )
    lev_type = _resolve_lev_type(getattr(problem_module, "lev_type", None))
    resonant_enabled = _resolve_per_leg_bool_setting(
        _first_defined_attr(problem_module, "resonant", "resonant_enabled"),
        n_legs,
        False,
        "resonant/resonant_enabled",
    )
    resonant_dvinf_km_s = _resolve_per_leg_float_setting(
        _first_defined_attr(problem_module, "dvinf", "resonant_dvinf"),
        n_legs,
        0.1,
        "dvinf/resonant_dvinf",
    )
    resonant_angle_tol_deg = _resolve_per_leg_float_setting(
        getattr(problem_module, "resonant_angle_tol_deg", None),
        n_legs,
        4.0,
        "resonant_angle_tol_deg",
    )
    resonant_crank_angles_by_leg = _resolve_per_leg_angle_lists(
        _first_defined_attr(problem_module, "crank_angle", "Bplane", "resonant_crank_angles_rad"),
        n_legs,
        "crank_angle/Bplane/resonant_crank_angles_rad",
    )

    lambert_mu_km3_s2 = _resolve_lambert_mu_km3_s2(observer)
    encounter_cfg = _build_encounter_cfg(
        bodies_by_stage=bodies_by_stage,
        time_bounds_days=time_bounds_days,
        tof_min_days=tof_min_days,
        tof_max_days=tof_max_days,
        dt_days_by_stage=dt_days_by_stage,
        alt_km_by_stage=alt_km_by_stage,
        frame=frame,
        observer=observer,
    )
    flyby_cfg = FlybyBuildConfig(
        numerical_eps=1e-12,
        trip_tof_bounds_s_by_stage=_resolve_trip_tof_bounds_s_by_stage(
            tof_min_days,
            tof_max_days,
            n_stages=n_stages,
        ),
        dv_patch_max_km_s_by_stage=_resolve_dvfb_max_by_flyby_stage(
            _first_defined_attr(problem_module, "dVfb_max", "dvfb_max"),
            n_stages=n_stages,
            n_legs=n_legs,
        ),
        central_mu_km3_s2=lambert_mu_km3_s2,
        flyby_filter=getattr(problem_module, "flyby_filter", None),
    )

    return StarProblemConfig(
        observer=observer,
        encounter_cfg=encounter_cfg,
        flyby_cfg=flyby_cfg,
        combo_dv_cap_km_s=combo_dv_cap_km_s,
        max_save_sol=max_save_sol,
        tfilter_dt_s=tfilter_dt_s,
        dt_filter_preemptive=dt_filter_preemptive,
        n_stages=n_stages,
        n_legs=n_legs,
        combo_num_legs=combo_num_legs,
        null_flags=null_flags,
        tof_min_days=tof_min_days,
        tof_max_days=tof_max_days,
        vlim=vlim,
        lambert_nrev=lambert_nrev,
        lambert_hz=lambert_hz,
        dvlev_max_km_s=dvlev_max_km_s,
        delta_dvlev_km_s=delta_dvlev_km_s,
        vinf_margin_pre_filter_km_s=vinf_margin_pre_filter_km_s,
        lev_type=lev_type,
        resonant_enabled=resonant_enabled,
        resonant_dvinf_km_s=resonant_dvinf_km_s,
        resonant_angle_tol_deg=resonant_angle_tol_deg,
        resonant_crank_angles_by_leg=resonant_crank_angles_by_leg,
        leg_parallelism=_normalize_leg_parallelism(problem_module),
        leg_filter=getattr(problem_module, "leg_filter", None),
    )


def _build_encounter_db(problem_cfg: StarProblemConfig) -> EncounterDB:
    """Build the initial encounter database for the current problem."""

    encounter_db = build_encounter_database(problem_cfg.encounter_cfg)
    print(f"EncounterDB entries: {len(encounter_db.entries)}")
    return encounter_db


def _null_stage_match_key(body_id: int, t_et_s: float) -> Tuple[int, int]:
    """Build a stable `(body, time)` key for null-leg stage matching."""

    return int(body_id), int(np.rint(float(t_et_s) * 1.0e6))


def _build_null_stage_ie_maps(
    encounter_db: EncounterDB,
    null_flags: Sequence[bool],
) -> Dict[int, Tuple[Dict[int, int], Dict[int, int]]]:
    """Build `{leg_stage: (stage_i_to_ip1, stage_ip1_to_i)}` maps for null legs."""

    out: Dict[int, Tuple[Dict[int, int], Dict[int, int]]] = {}
    for leg_index, is_null_leg in enumerate(null_flags):
        if not bool(is_null_leg):
            continue

        stage_i_lookup: Dict[Tuple[int, int], int] = {}
        stage_ip1_lookup: Dict[Tuple[int, int], int] = {}
        for entry in encounter_db.entries:
            stage_id = int(entry.stage_id)
            key = _null_stage_match_key(int(entry.body), float(entry.t_et))
            if stage_id == int(leg_index):
                if key in stage_i_lookup:
                    raise ValueError(f"Null-leg stage {leg_index} has duplicate (body,time) encounter keys.")
                stage_i_lookup[key] = int(entry.IE)
            elif stage_id == int(leg_index) + 1:
                if key in stage_ip1_lookup:
                    raise ValueError(f"Null-leg stage {leg_index + 1} has duplicate (body,time) encounter keys.")
                stage_ip1_lookup[key] = int(entry.IE)

        shared_keys = sorted(set(stage_i_lookup.keys()).intersection(stage_ip1_lookup.keys()))
        if not shared_keys:
            raise ValueError(
                f"Null leg {leg_index} requires shared (body,time) encounter support between "
                f"stages {leg_index} and {leg_index + 1}."
            )

        out[int(leg_index)] = (
            {int(stage_i_lookup[key]): int(stage_ip1_lookup[key]) for key in shared_keys},
            {int(stage_ip1_lookup[key]): int(stage_i_lookup[key]) for key in shared_keys},
        )
    return out


def _build_null_run_maps(
    null_flags: Sequence[bool],
) -> Tuple[Dict[int, _NullRunState], Dict[int, int]]:
    """Build contiguous null-run metadata for scheduler-side direction freezing."""

    runs_by_start: Dict[int, _NullRunState] = {}
    run_start_by_leg: Dict[int, int] = {}
    leg_index = 0
    n_legs = len(list(null_flags))
    while leg_index < n_legs:
        if not bool(null_flags[leg_index]):
            leg_index += 1
            continue

        run_start = int(leg_index)
        run_end = int(run_start)
        while (run_end + 1) < n_legs and bool(null_flags[run_end + 1]):
            run_end += 1

        runs_by_start[run_start] = _NullRunState(start_leg=run_start, end_leg=run_end)
        for null_leg in range(run_start, run_end + 1):
            run_start_by_leg[int(null_leg)] = int(run_start)

        leg_index = run_end + 1

    return runs_by_start, run_start_by_leg

def _build_null_leg_append_rows(
    source_leg_db: LegDatabase,
    ie_map: Dict[int, int],
    *,
    use_previous: bool,
) -> Tuple[Optional[Dict[str, np.ndarray]], np.ndarray]:
    """Return null-leg rows derived from one adjacent active source leg."""

    if not ie_map or int(np.asarray(source_leg_db.IL, dtype=np.int64).size) == 0:
        return None, np.empty(0, dtype=np.int64)

    if use_previous:
        source_ie = np.asarray(source_leg_db.IA, dtype=np.int64)
        source_vinf = np.asarray(source_leg_db.vinfA_km_s, dtype=float)
    else:
        source_ie = np.asarray(source_leg_db.ID, dtype=np.int64)
        source_vinf = np.asarray(source_leg_db.vinfD_km_s, dtype=float)

    keep_mask = np.asarray([int(ie) in ie_map for ie in source_ie], dtype=bool)
    if not np.any(keep_mask):
        return None, np.empty(0, dtype=np.int64)

    copied_ie = np.asarray(source_ie[keep_mask], dtype=np.int64)
    copied_vinf = np.asarray(source_vinf[keep_mask], dtype=float)
    source_il = np.asarray(source_leg_db.IL, dtype=np.int64)[keep_mask]
    derived_ie = np.asarray([int(ie_map[int(ie)]) for ie in copied_ie], dtype=np.int64)
    if use_previous:
        null_id = copied_ie
        null_ia = derived_ie
    else:
        null_id = derived_ie
        null_ia = copied_ie

    row_count = int(copied_ie.size)
    return (
        {
            "ID": null_id,
            "IA": null_ia,
            "vinfD_km_s": copied_vinf,
            "vinfA_km_s": copied_vinf,
            "n_rev": np.zeros(row_count, dtype=np.int64),
            "dv_lev_km_s": np.zeros(row_count, dtype=float),
            "eta_lev": np.zeros(row_count, dtype=float),
        },
        np.asarray(source_il, dtype=np.int64),
    )


def _subset_flyby_leg_excluding_il(leg_db: LegDatabase, excluded_il: np.ndarray) -> LegDatabase:
    """Return a flyby-view LegDatabase excluding one set of stage-local ILs."""

    excluded_array = np.asarray(excluded_il, dtype=np.int64).reshape(-1)
    if int(excluded_array.size) == 0:
        return leg_db

    keep_mask = ~np.isin(np.asarray(leg_db.IL, dtype=np.int64), excluded_array)
    return LegDatabase(
        IL=np.asarray(leg_db.IL, dtype=np.int64)[keep_mask],
        stage_id=int(leg_db.stage_id),
        ID=np.asarray(leg_db.ID, dtype=np.int64)[keep_mask],
        IA=np.asarray(leg_db.IA, dtype=np.int64)[keep_mask],
        vinfD_km_s=np.asarray(leg_db.vinfD_km_s, dtype=float)[keep_mask],
        vinfA_km_s=np.asarray(leg_db.vinfA_km_s, dtype=float)[keep_mask],
        n_rev=None if leg_db.n_rev is None else np.asarray(leg_db.n_rev, dtype=np.int64)[keep_mask],
        dv_lev_km_s=(
            None if leg_db.dv_lev_km_s is None else np.asarray(leg_db.dv_lev_km_s, dtype=float)[keep_mask]
        ),
        eta_lev=None if leg_db.eta_lev is None else np.asarray(leg_db.eta_lev, dtype=float)[keep_mask],
    )


def _build_null_direct_flyby_rows(
    *,
    stage_id: int,
    copied_side: str,
    copied_leg_db: LegDatabase,
    opposite_leg_db: LegDatabase,
    copied_il: np.ndarray,
    source_il: np.ndarray,
) -> FlybyDB:
    """Build zero-DV flyby rows from sparse copied-row bindings."""

    copied_lookup = {
        int(leg_il): row_index
        for row_index, leg_il in enumerate(np.asarray(copied_leg_db.IL, dtype=np.int64).reshape(-1))
    }
    opposite_lookup = {
        int(leg_il): row_index
        for row_index, leg_il in enumerate(np.asarray(opposite_leg_db.IL, dtype=np.int64).reshape(-1))
    }

    ie_rows: list[int] = []
    il_in_rows: list[int] = []
    il_out_rows: list[int] = []
    copied_array = np.asarray(copied_il, dtype=np.int64).reshape(-1)
    source_array = np.asarray(source_il, dtype=np.int64).reshape(-1)
    for copied_leg_il, source_leg_il in zip(copied_array, source_array):
        copied_row = copied_lookup.get(int(copied_leg_il), None)
        if copied_row is None:
            continue

        opposite_row = opposite_lookup.get(int(source_leg_il), None)
        if opposite_row is None:
            continue

        if str(copied_side) == "incoming":
            encounter_ie = int(np.asarray(copied_leg_db.IA, dtype=np.int64)[copied_row])
            source_encounter_ie = int(np.asarray(opposite_leg_db.ID, dtype=np.int64)[opposite_row])
            if encounter_ie != source_encounter_ie:
                raise ValueError(
                    f"Corrupt null-binding sidecar at flyby stage {int(stage_id)}: incoming copied "
                    f"IL={int(copied_leg_il)} expects source IL={int(source_leg_il)} at encounter IE={encounter_ie}, "
                    f"got IE={source_encounter_ie}."
                )
            ie_rows.append(encounter_ie)
            il_in_rows.append(int(copied_leg_il))
            il_out_rows.append(int(source_leg_il))
        else:
            encounter_ie = int(np.asarray(copied_leg_db.ID, dtype=np.int64)[copied_row])
            source_encounter_ie = int(np.asarray(opposite_leg_db.IA, dtype=np.int64)[opposite_row])
            if encounter_ie != source_encounter_ie:
                raise ValueError(
                    f"Corrupt null-binding sidecar at flyby stage {int(stage_id)}: outgoing copied "
                    f"IL={int(copied_leg_il)} expects source IL={int(source_leg_il)} at encounter IE={encounter_ie}, "
                    f"got IE={source_encounter_ie}."
                )
            ie_rows.append(encounter_ie)
            il_in_rows.append(int(source_leg_il))
            il_out_rows.append(int(copied_leg_il))

    if not ie_rows:
        empty_i64 = np.empty(0, dtype=np.int64)
        empty_f64 = np.empty(0, dtype=float)
        return FlybyDB(
            IF=empty_i64,
            stage_id=empty_i64,
            IE=empty_i64,
            IL_in=empty_i64,
            IL_out=empty_i64,
            dv_km_s=empty_f64,
        )

    ie_array = np.asarray(ie_rows, dtype=np.int64)
    il_in_array = np.asarray(il_in_rows, dtype=np.int64)
    il_out_array = np.asarray(il_out_rows, dtype=np.int64)
    order = np.lexsort((il_out_array, il_in_array, ie_array))
    ie_sorted = ie_array[order]
    il_in_sorted = il_in_array[order]
    il_out_sorted = il_out_array[order]
    dv_array = np.zeros(int(ie_sorted.size), dtype=float)
    return FlybyDB(
        IF=np.arange(int(ie_sorted.size), dtype=np.int64),
        stage_id=np.full(int(ie_sorted.size), int(stage_id), dtype=np.int64),
        IE=ie_sorted,
        IL_in=il_in_sorted,
        IL_out=il_out_sorted,
        dv_km_s=dv_array,
    )


def _build_leg_cfg(problem_cfg: StarProblemConfig, leg_index: int) -> LegBuildConfig:
    """Build LegBuildConfig for one leg using normalized problem settings."""

    tof_min_s = j2000_days_to_et_seconds(float(problem_cfg.tof_min_days[leg_index, leg_index + 1]))
    tof_max_s = j2000_days_to_et_seconds(float(problem_cfg.tof_max_days[leg_index, leg_index + 1]))
    nrev_max = int(problem_cfg.lambert_nrev[leg_index])
    if nrev_max < 0:
        raise ValueError(f"lambert_nrev[{leg_index}] must be >= 0.")

    hz_leg = int(problem_cfg.lambert_hz[leg_index])
    if hz_leg not in (-1, 0, 1):
        raise ValueError(f"lambert_hz[{leg_index}] must be one of -1, 0, 1.")

    dvlev_max = float(problem_cfg.dvlev_max_km_s[leg_index])
    delta_dvlev = float(problem_cfg.delta_dvlev_km_s[leg_index])

    return LegBuildConfig(
        leg_stage_id=int(leg_index),
        dep_stage_id=int(leg_index),
        arr_stage_id=int(leg_index + 1),
        tof_min_s=float(tof_min_s),
        tof_max_s=float(tof_max_s),
        vinfD_bounds_km_s=(float(problem_cfg.vlim[0, leg_index]), float(problem_cfg.vlim[1, leg_index])),
        vinfA_bounds_km_s=(
            float(problem_cfg.vlim[0, leg_index + 1]),
            float(problem_cfg.vlim[1, leg_index + 1]),
        ),
        lambert_mu_km3_s2=float(problem_cfg.flyby_cfg.central_mu_km3_s2),
        lambert_nrev_max=int(nrev_max),
        lambert_hz=int(hz_leg),
        dvlev_max_km_s=dvlev_max,
        delta_dvlev_km_s=delta_dvlev,
        vinf_margin_pre_filter_km_s=float(problem_cfg.vinf_margin_pre_filter_km_s[leg_index]),
        lev_type=problem_cfg.lev_type,
        resonant_enabled=bool(problem_cfg.resonant_enabled[leg_index]),
        resonant_angle_tol_deg=float(problem_cfg.resonant_angle_tol_deg[leg_index]),
        resonant_dvinf_km_s=float(problem_cfg.resonant_dvinf_km_s[leg_index]),
        resonant_crank_angles_rad=tuple(problem_cfg.resonant_crank_angles_by_leg[leg_index]),
        parallelism=problem_cfg.leg_parallelism,
        dt_positive_tol_s=1e-6,
        max_failure_frac=None,
        debug=False,
        chunk_size=20000,
        leg_filter=problem_cfg.leg_filter,
    )


def _count_stage_encounters(encounter_db: EncounterDB, stage_id: int) -> int:
    """Return current encounter count at one stage."""

    return sum(1 for entry in encounter_db.entries if int(entry.stage_id) == int(stage_id))


def _stage_safe_to_filter(stage_id: int, *, n_stages: int, unbuilt: set[int]) -> bool:
    """Return True once all legs touching `stage_id` have been built."""

    incoming_ok = (stage_id == 0) or ((stage_id - 1) not in unbuilt)
    outgoing_ok = (stage_id == int(n_stages) - 1) or (stage_id not in unbuilt)
    return incoming_ok and outgoing_ok


def _build_leg_weights(encounter_db: EncounterDB, problem_cfg: StarProblemConfig) -> Dict[int, float]:
    """Compute one-time STAR leg weights used for interleaved build ordering."""

    init_t_min: Dict[int, float] = {}
    init_t_max: Dict[int, float] = {}
    for entry in encounter_db.entries:
        stage_id = int(entry.stage_id)
        t_et = float(entry.t_et)
        if stage_id not in init_t_min or t_et < init_t_min[stage_id]:
            init_t_min[stage_id] = t_et
        if stage_id not in init_t_max or t_et > init_t_max[stage_id]:
            init_t_max[stage_id] = t_et

    leg_weights: Dict[int, float] = {}
    for leg_index in range(int(problem_cfg.n_legs)):
        tof_min_s = j2000_days_to_et_seconds(float(problem_cfg.tof_min_days[leg_index, leg_index + 1]))
        tof_max_s = j2000_days_to_et_seconds(float(problem_cfg.tof_max_days[leg_index, leg_index + 1]))
        tof_window = max(tof_max_s - tof_min_s, 0.0)
        time_span = max(init_t_max.get(leg_index + 1, tof_max_s) - init_t_min.get(leg_index, 0.0), 1.0)
        n_lambert = max(1, 2 * (int(problem_cfg.lambert_nrev[leg_index]) + 1))
        dv_lev = float(problem_cfg.dvlev_max_km_s[leg_index])
        delta_dv = float(problem_cfg.delta_dvlev_km_s[leg_index])
        if delta_dv > 0.0 and dv_lev > 0.0:
            leveraging = 1.0 + 5.0 * dv_lev / delta_dv
        else:
            leveraging = 1.0
        leg_weights[leg_index] = n_lambert * (tof_window / time_span) * leveraging

    return leg_weights


def _prepare_stage_dirs(
    cache_root: Path,
    *,
    resume_phase2_only: bool,
    n_legs: int,
) -> Tuple[Dict[int, Path], Dict[int, Path]]:
    """Load cached stage directories for resume mode or return empty mappings."""

    if not resume_phase2_only:
        return {}, {}

    leg_stage_dirs, flyby_stage_dirs = _discover_stage_dirs_from_cache(cache_root)
    print(
        f"Resume cache stages: legs={len(leg_stage_dirs)} "
        f"({min(leg_stage_dirs.keys())}-{max(leg_stage_dirs.keys())}), "
        f"flybys={len(flyby_stage_dirs)} "
        f"({min(flyby_stage_dirs.keys())}-{max(flyby_stage_dirs.keys())})"
    )
    if len(leg_stage_dirs) != int(n_legs):
        raise ValueError(f"Resume cache has {len(leg_stage_dirs)} leg stages, but problem expects {int(n_legs)}.")
    if len(flyby_stage_dirs) != int(max(0, n_legs - 1)):
        raise ValueError(
            f"Resume cache has {len(flyby_stage_dirs)} flyby stages, "
            f"but problem expects {int(max(0, n_legs - 1))}."
        )
    return leg_stage_dirs, flyby_stage_dirs


class Phase1Runner:
    """Own STAR phase-1 orchestration and mutable cache plumbing."""

    def __init__(
        self,
        *,
        problem_cfg: StarProblemConfig,
        cache_root: Path,
        state: Phase1State,
    ) -> None:
        self.problem_cfg = problem_cfg
        self.cache_root = cache_root
        self.state = state
        self.leg_weights = _build_leg_weights(state.encounter_db, problem_cfg)

    def run(self) -> EncounterDB:
        """Run interleaved leg/flyby build plus phase-1 filtering."""

        print("LegDB / FlybyDB interleaved construction ...")
        while self.state.unbuilt_legs:
            leg_index = self.select_next_leg()
            self.build_leg(leg_index)
            self.filter_safe_stages((leg_index, leg_index + 1))
            self.build_adjacent_flybys(leg_index)

        self.run_final_filter_convergence()
        self.flush_masks()
        gc.collect()
        return self.state.encounter_db

    def _get_null_run(self, leg_index: int) -> Optional[_NullRunState]:
        """Return null-run metadata for one null leg, if any."""

        run_start = self.state.null_run_start_by_leg.get(int(leg_index), None)
        if run_start is None:
            return None
        return self.state.null_runs_by_start.get(int(run_start), None)

    def _resolve_null_run_direction(self, leg_index: int) -> Optional[str]:
        """Resolve/freeze run direction from the first available real boundary leg."""

        null_run = self._get_null_run(int(leg_index))
        if null_run is None:
            return None
        if null_run.anchor_direction in {"left", "right"}:
            return str(null_run.anchor_direction)

        candidates: list[Tuple[int, str]] = []
        left_boundary_leg = int(null_run.start_leg) - 1
        right_boundary_leg = int(null_run.end_leg) + 1

        if left_boundary_leg >= 0 and not bool(self.problem_cfg.null_flags[left_boundary_leg]):
            if int(left_boundary_leg) in self.state.leg_stage_dirs:
                build_order = self.state.leg_build_order.get(int(left_boundary_leg), None)
                if build_order is not None:
                    candidates.append((int(build_order), "left"))

        if right_boundary_leg < int(self.problem_cfg.n_legs) and not bool(self.problem_cfg.null_flags[right_boundary_leg]):
            if int(right_boundary_leg) in self.state.leg_stage_dirs:
                build_order = self.state.leg_build_order.get(int(right_boundary_leg), None)
                if build_order is not None:
                    candidates.append((int(build_order), "right"))

        if not candidates:
            return None

        _, direction = min(candidates, key=lambda item: (int(item[0]), 0 if str(item[1]) == "left" else 1))
        null_run.anchor_direction = str(direction)
        return str(direction)

    def _null_leg_is_buildable(self, leg_index: int) -> bool:
        """Return True when a null leg is the next eligible leg in its anchored run."""

        if not bool(self.problem_cfg.null_flags[int(leg_index)]):
            return True

        null_run = self._get_null_run(int(leg_index))
        if null_run is None:
            return False

        direction = self._resolve_null_run_direction(int(leg_index))
        if direction is None:
            return False

        unbuilt_in_run = sorted(
            int(null_leg)
            for null_leg in range(int(null_run.start_leg), int(null_run.end_leg) + 1)
            if int(null_leg) in self.state.unbuilt_legs
        )
        if not unbuilt_in_run:
            return False

        if str(direction) == "left":
            return int(leg_index) == int(unbuilt_in_run[0])
        return int(leg_index) == int(unbuilt_in_run[-1])

    def select_next_leg(self) -> int:
        """Pick the cheapest currently unbuilt leg."""

        eligible_legs = [
            int(leg_index)
            for leg_index in self.state.unbuilt_legs
            if self._null_leg_is_buildable(int(leg_index))
        ]
        if not eligible_legs:
            raise RuntimeError("No buildable leg remains; null runs require an anchored real boundary leg.")

        return min(
            eligible_legs,
            key=lambda leg_index: (
                _count_stage_encounters(self.state.encounter_db, leg_index)
                * _count_stage_encounters(self.state.encounter_db, leg_index + 1)
                * self.leg_weights[leg_index]
            ),
        )

    def _load_null_source_leg(self, leg_index: int) -> Tuple[Optional[LegDatabase], bool]:
        """Load the null-row source leg from the frozen run direction only."""

        direction = self._resolve_null_run_direction(int(leg_index))
        if direction is None:
            return None, True

        use_previous = str(direction) == "left"
        source_stage = int(leg_index) - 1 if use_previous else int(leg_index) + 1
        stage_dir = self.state.leg_stage_dirs.get(int(source_stage))
        if stage_dir is None:
            return None, bool(use_previous)

        source_leg_db = load_leg_stage(stage_dir, active_only=True)
        if int(np.asarray(source_leg_db.IL, dtype=np.int64).size) == 0:
            return None, bool(use_previous)
        return source_leg_db, bool(use_previous)

    def build_leg(self, leg_index: int) -> None:
        """Build, save, and register one leg stage."""

        cfg = _build_leg_cfg(self.problem_cfg, leg_index)
        post_rows = None
        source_il = np.empty(0, dtype=np.int64)
        null_source_stage_id: Optional[int] = None
        if bool(self.problem_cfg.null_flags[int(leg_index)]):
            source_leg_db, use_previous = self._load_null_source_leg(int(leg_index))
            if source_leg_db is not None:
                ie_map_pair = self.state.null_stage_ie_maps[int(leg_index)]
                post_rows, source_il = _build_null_leg_append_rows(
                    source_leg_db,
                    ie_map_pair[0] if use_previous else ie_map_pair[1],
                    use_previous=use_previous,
                )
                if post_rows is not None and int(source_il.size) > 0:
                    null_source_stage_id = int(leg_index) - 1 if bool(use_previous) else int(leg_index) + 1
        leg_stage_dir = self.cache_root / f"leg_{int(leg_index):02d}"
        row_count = build_leg_stage_to_disk(
            self.state.encounter_db,
            cfg,
            stage_dir=leg_stage_dir,
            stage_arrays_cache=self.state.stage_arrays_cache,
            post_rows=post_rows,
        )
        copied_il = np.empty(0, dtype=np.int64)
        if post_rows is not None and int(np.asarray(post_rows["ID"], dtype=np.int64).size) > 0:
            base_rows = int(row_count - int(np.asarray(post_rows["ID"], dtype=np.int64).size))
            copied_il = np.arange(base_rows, int(row_count), dtype=np.int64)
        save_leg_null_binding_sidecar(
            leg_stage_dir,
            source_stage_id=null_source_stage_id,
            copied_il=copied_il,
            source_il=source_il,
        )
        self.state.leg_stage_dirs[int(leg_index)] = leg_stage_dir
        add_leg_stage_to_active_index_cache(
            int(leg_index),
            leg_stage_dir,
            self.state.active_index_cache,
            active_leg_ie_cache=self.state.active_leg_ie_cache,
        )
        self.state.unbuilt_legs.remove(int(leg_index))
        self.state.leg_build_order[int(leg_index)] = int(self.state.next_leg_build_order)
        self.state.next_leg_build_order += 1
        print(f"  leg {leg_index}: {int(count_active_rows(leg_stage_dir))} rows  ({len(self.state.unbuilt_legs)} unbuilt remaining)")
        gc.collect()

    def safe_stages(self, candidates: Optional[Iterable[int]] = None) -> list[int]:
        """Return encounter stages whose adjacent legs are all already built."""

        stage_ids = range(int(self.problem_cfg.n_stages)) if candidates is None else candidates
        return [
            int(stage_id)
            for stage_id in stage_ids
            if _stage_safe_to_filter(
                int(stage_id),
                n_stages=self.problem_cfg.n_stages,
                unbuilt=self.state.unbuilt_legs,
            )
        ]

    def filter_safe_stages(self, candidates: Optional[Iterable[int]] = None) -> None:
        """Encounter-filter any now-safe encounter stages."""

        safe_stages = self.safe_stages(candidates)
        if not safe_stages:
            return

        self.state.encounter_db = encounter_filter_from_active_legs(
            self.state.encounter_db,
            self.state.leg_stage_dirs,
            stages=safe_stages,
            active_leg_ie_cache=self.state.active_leg_ie_cache,
        )
        self.state.stage_arrays_cache.clear()

    def build_adjacent_flybys(self, leg_index: int) -> None:
        """Build any newly available flyby stages adjacent to one leg."""

        for stage_id in (int(leg_index), int(leg_index) + 1):
            if not self.try_build_flyby(stage_id):
                continue
            self.run_local_filter_fixpoint()
            self.filter_safe_stages()

    def try_build_flyby(self, stage_id: int) -> bool:
        """Build one flyby stage if both adjacent leg stages are available."""

        stage_int = int(stage_id)
        if stage_int <= 0 or stage_int >= int(self.problem_cfg.n_stages) - 1:
            return False
        if stage_int in self.state.flyby_stage_dirs:
            return False
        if (stage_int - 1) not in self.state.leg_stage_dirs or stage_int not in self.state.leg_stage_dirs:
            return False

        incoming_stage_dir = self.state.leg_stage_dirs[stage_int - 1]
        outgoing_stage_dir = self.state.leg_stage_dirs[stage_int]
        incoming_leg = load_leg_stage_for_flyby(incoming_stage_dir, active_only=True)
        outgoing_leg = load_leg_stage_for_flyby(outgoing_stage_dir, active_only=True)
        incoming_source_stage, incoming_copied_il, incoming_source_il = load_leg_null_binding_sidecar(
            incoming_stage_dir,
            active_only=True,
        )
        outgoing_source_stage, outgoing_copied_il, outgoing_source_il = load_leg_null_binding_sidecar(
            outgoing_stage_dir,
            active_only=True,
        )

        incoming_is_copied_side = incoming_source_stage is not None and int(incoming_source_stage) == stage_int
        outgoing_is_copied_side = outgoing_source_stage is not None and int(outgoing_source_stage) == int(stage_int) - 1
        if incoming_is_copied_side and outgoing_is_copied_side:
            raise ValueError(f"Flyby stage {stage_int} cannot have both incoming and outgoing copied-side null bindings.")

        precomputed_flyby_rows: Optional[FlybyDB] = None
        if incoming_is_copied_side:
            precomputed_flyby_rows = _build_null_direct_flyby_rows(
                stage_id=stage_int,
                copied_side="incoming",
                copied_leg_db=incoming_leg,
                opposite_leg_db=outgoing_leg,
                copied_il=incoming_copied_il,
                source_il=incoming_source_il,
            )
            incoming_leg = _subset_flyby_leg_excluding_il(incoming_leg, incoming_copied_il)
        elif outgoing_is_copied_side:
            precomputed_flyby_rows = _build_null_direct_flyby_rows(
                stage_id=stage_int,
                copied_side="outgoing",
                copied_leg_db=outgoing_leg,
                opposite_leg_db=incoming_leg,
                copied_il=outgoing_copied_il,
                source_il=outgoing_source_il,
            )
            outgoing_leg = _subset_flyby_leg_excluding_il(outgoing_leg, outgoing_copied_il)

        flyby_stage_dir = self.cache_root / "flybys" / f"flyby_{stage_int:02d}"
        flyby_rows = build_flyby_stage_to_disk(
            self.state.encounter_db,
            incoming_leg,
            outgoing_leg,
            self.problem_cfg.flyby_cfg,
            stage_id=stage_int,
            stage_dir=flyby_stage_dir,
            if_start=int(self.state.if_offset),
            flush_rows=2_000_000,
            precomputed_rows=precomputed_flyby_rows,
        )
        self.state.if_offset += int(flyby_rows)
        self.state.flyby_stage_dirs[stage_int] = flyby_stage_dir
        add_flyby_stage_to_active_index_cache(stage_int, flyby_stage_dir, self.state.active_index_cache)
        print(f"  flyby {stage_int}: {int(flyby_rows)} entries")
        del incoming_leg, outgoing_leg
        gc.collect()
        return True

    def run_local_filter_fixpoint(self) -> None:
        """Run local flyby/leg filtering after building a new flyby stage."""

        if not self.state.flyby_stage_dirs:
            return

        self.run_filter_fixpoint(max_passes=max(1, 2 * len(self.state.flyby_stage_dirs) + 2))
        self.flush_masks()

    def run_final_filter_convergence(self) -> None:
        """Run final incoming/outgoing convergence across all built flybys."""

        if not self.state.flyby_stage_dirs:
            return

        max_passes = max(1, 2 * len(self.state.flyby_stage_dirs) + 2)
        first_flyby_stage = int(min(self.state.flyby_stage_dirs.keys()))
        last_flyby_stage = int(max(self.state.flyby_stage_dirs.keys()))
        print()
        print(f"Final filter convergence (first={first_flyby_stage}, last={last_flyby_stage}) ...")

        converged = self.run_filter_fixpoint(max_passes=max_passes, log_progress=True)
        if converged is not None:
            print(f"  Fixpoint reached in {converged} pass(es).")
            return

        print(f"  Fixpoint not reached within {max_passes} pass(es).")

    def run_filter_fixpoint(self, *, max_passes: int, log_progress: bool = False) -> Optional[int]:
        """Run repeated backward/forward propagation passes to fixpoint."""

        for pass_index in range(1, int(max_passes) + 1):
            removed_total = self.run_bidirectional_leg_filter_pass()
            if log_progress:
                print(f"  pass {pass_index}: removed={removed_total}")
            if removed_total == 0:
                return int(pass_index)
        return None

    def run_bidirectional_leg_filter_pass(self) -> int:
        """Run one backward and one forward leg-filter propagation pass."""

        if not self.state.flyby_stage_dirs:
            return 0

        first_flyby_stage = int(min(self.state.flyby_stage_dirs.keys()))
        last_flyby_stage = int(max(self.state.flyby_stage_dirs.keys()))
        removed_incoming = incoming_leg_filter_disk(
            last_flyby_stage,
            self.state.leg_stage_dirs,
            self.state.flyby_stage_dirs,
            self.state.encounter_db,
            active_leg_ie_cache=self.state.active_leg_ie_cache,
            index_cache=self.state.active_index_cache,
        )
        removed_outgoing = outgoing_leg_filter_disk(
            first_flyby_stage,
            self.state.leg_stage_dirs,
            self.state.flyby_stage_dirs,
            self.state.encounter_db,
            active_leg_ie_cache=self.state.active_leg_ie_cache,
            index_cache=self.state.active_index_cache,
        )
        return int(removed_incoming + removed_outgoing)

    def flush_masks(self) -> None:
        """Persist any dirty stage active masks."""

        flush_active_index_cache_masks(
            self.state.leg_stage_dirs,
            self.state.flyby_stage_dirs,
            self.state.active_index_cache,
        )


def _run_phase1(
    encounter_db: EncounterDB,
    *,
    problem_cfg: StarProblemConfig,
    cache_root: Path,
    leg_stage_dirs: Dict[int, Path],
    flyby_stage_dirs: Dict[int, Path],
    resume_phase2_only: bool,
) -> EncounterDB:
    """Run STAR phase 1: interleaved leg/flyby build with active filtering."""

    if resume_phase2_only:
        print("Phase 1 skipped: resuming from cached leg/flyby stages.")
        return encounter_db

    null_runs_by_start, null_run_start_by_leg = _build_null_run_maps(problem_cfg.null_flags)
    return Phase1Runner(
        problem_cfg=problem_cfg,
        cache_root=cache_root,
        state=Phase1State(
            encounter_db=encounter_db,
            unbuilt_legs=set(range(int(problem_cfg.n_legs))),
            leg_stage_dirs=leg_stage_dirs,
            flyby_stage_dirs=flyby_stage_dirs,
            if_offset=0,
            null_stage_ie_maps=_build_null_stage_ie_maps(encounter_db, problem_cfg.null_flags),
            null_runs_by_start=null_runs_by_start,
            null_run_start_by_leg=null_run_start_by_leg,
            leg_build_order={},
            next_leg_build_order=0,
            stage_arrays_cache={},
            active_leg_ie_cache={},
            active_index_cache=create_active_index_cache(),
        ),
    ).run()


def _run_combo_phase(
    encounter_db: EncounterDB,
    *,
    problem_cfg: StarProblemConfig,
    leg_stage_dirs: Dict[int, Path],
    flyby_stage_dirs: Dict[int, Path],
) -> "OutputDB":
    """Run STAR phase 2 combo stitching."""

    print()
    print("Combo module stitching start...")
    output_db = run_combo(
        encounter_db,
        cfg=ComboBuildConfig(
            dv_total_max_km_s=problem_cfg.combo_dv_cap_km_s,
            tof_total_bounds_s=None,
            max_rows_per_segment_db=None,
            debug=False,
            nL=problem_cfg.combo_num_legs,
            dt_filter_preemptive_s=(
                None
                if not problem_cfg.dt_filter_preemptive
                else np.asarray(problem_cfg.tfilter_dt_s, dtype=float)
            ),
        ),
        leg_stage_dirs=leg_stage_dirs,
        flyby_stage_dirs=flyby_stage_dirs,
    )
    print(f"Combo done! num_final_trajectories={int(output_db.I_traj.size)}")
    return output_db


def _format_dt_filter_days_text(dt_filter_days_arr: np.ndarray, *, dep_arr_only: bool) -> str:
    """Format dt-filter day bins for human-readable logging."""

    values = np.asarray(dt_filter_days_arr, dtype=float).reshape(-1)
    if dep_arr_only:
        dep_days = float(values[0])
        arr_days = float(values[-1])
        return f"{dep_days}" if np.isclose(dep_days, arr_days) else f"dep={dep_days}, arr={arr_days}"

    return (
        f"{float(values[0])}"
        if int(values.size) == 1
        else np.array2string(values, separator=",")
    )


def _apply_final_output_tfilter(output_db: "OutputDB", problem_cfg: StarProblemConfig) -> "OutputDB":
    """Apply final-output tfilter summary/decimation when configured."""

    if problem_cfg.tfilter_dt_s is None:
        return output_db

    dt_filter_days_arr = np.asarray(problem_cfg.tfilter_dt_s, dtype=float) / float(j2000_days_to_et_seconds(1.0))
    if problem_cfg.dt_filter_preemptive:
        print()
        print(
            "Preemptive dt-filter enabled in combo "
            f"(dt_filter_days={_format_dt_filter_days_text(dt_filter_days_arr, dep_arr_only=False)}); "
            "skipping one-shot final tfilter."
        )
        return output_db

    output_db, tfilter_stats = tfilter_output_db(output_db, np.asarray(problem_cfg.tfilter_dt_s, dtype=float))
    t_dep_ref_s = tfilter_stats.get("t_dep_ref_s", None)
    t_arr_ref_s = tfilter_stats.get("t_arr_ref_s", None)
    t_dep_ref_days = None if t_dep_ref_s is None else float(et_seconds_to_j2000_days(float(t_dep_ref_s)))
    t_arr_ref_days = None if t_arr_ref_s is None else float(et_seconds_to_j2000_days(float(t_arr_ref_s)))

    print()
    print("TFilter Summary")
    print(
        f"num_in={int(tfilter_stats.get('num_in', 0))}, "
        f"num_valid={int(tfilter_stats.get('num_valid', 0))}, "
        f"num_bins={int(tfilter_stats.get('num_bins', 0))}, "
        f"num_out={int(tfilter_stats.get('num_out', 0))}, "
        f"num_skipped_invalid={int(tfilter_stats.get('num_skipped_invalid', 0))}"
    )
    print(
        f"dt_filter_days={_format_dt_filter_days_text(dt_filter_days_arr, dep_arr_only=True)}, "
        f"t_dep_ref_days={t_dep_ref_days}, "
        f"t_arr_ref_days={t_arr_ref_days}"
    )
    return output_db


def _finalize_output(
    output_db: "OutputDB",
    *,
    problem_cfg: StarProblemConfig,
    encounter_db: EncounterDB,
    leg_stage_dirs: Dict[int, Path],
    flyby_stage_dirs: Dict[int, Path],
    output_path: Path,
) -> None:
    """Apply final save-cap and write output trajectories if any remain."""

    limited_output_db = _apply_max_save_sol(output_db, problem_cfg.max_save_sol)
    if limited_output_db.I_traj.size == 0:
        print("No full trajectories satisfy current leg/flyby databases.")
        return

    leg_dbs_by_stage: Dict[int, LegDatabase] = {
        int(stage_id): load_leg_stage(stage_dir, active_only=True)
        for stage_id, stage_dir in sorted(leg_stage_dirs.items())
    }
    saved_file = save_solution(
        output_db=limited_output_db,
        enc_db=encounter_db,
        leg_dbs=leg_dbs_by_stage,
        flyby_dbs=None,
        flyby_stage_dirs=flyby_stage_dirs,
        output_path=output_path,
        null_flags=problem_cfg.null_flags,
    )
    print(f"Saved minimal solution file: {saved_file}")


def _cleanup_cache_root(cache_root: Path, *, resume_phase2_only: bool) -> None:
    """Remove transient cache directories created for this run."""

    if (not resume_phase2_only) and cache_root.exists():
        shutil.rmtree(cache_root, ignore_errors=True)


def main() -> None:
    """Run the full STAR pipeline using the selected problem module."""

    args = build_arg_parser().parse_args()
    run_paths = _resolve_run_paths(args, root_folder)
    if run_paths.resume_phase2_only:
        print(f"Resume mode: using existing cache {run_paths.cache_root}")

    _load_spice_metakernel(run_paths.metakernel_path)
    problem_cfg = _resolve_problem_config(_import_problem_module(args.problem))
    encounter_db = _build_encounter_db(problem_cfg)
    leg_stage_dirs, flyby_stage_dirs = _prepare_stage_dirs(
        run_paths.cache_root,
        resume_phase2_only=run_paths.resume_phase2_only,
        n_legs=problem_cfg.n_legs,
    )
    # leg / flyby database construction 
    encounter_db = _run_phase1(
        encounter_db,
        problem_cfg=problem_cfg,
        cache_root=run_paths.cache_root,
        leg_stage_dirs=leg_stage_dirs,
        flyby_stage_dirs=flyby_stage_dirs,
        resume_phase2_only=run_paths.resume_phase2_only,
    )
    # combo data construction 
    output_db = _run_combo_phase(
        encounter_db,
        problem_cfg=problem_cfg,
        leg_stage_dirs=leg_stage_dirs,
        flyby_stage_dirs=flyby_stage_dirs,
    )
    output_db = _apply_final_output_tfilter(output_db, problem_cfg)
    _finalize_output(
        output_db,
        problem_cfg=problem_cfg,
        encounter_db=encounter_db,
        leg_stage_dirs=leg_stage_dirs,
        flyby_stage_dirs=flyby_stage_dirs,
        output_path=run_paths.output_path,
    )
    _cleanup_cache_root(run_paths.cache_root, resume_phase2_only=run_paths.resume_phase2_only)

if __name__ == "__main__":
    main()

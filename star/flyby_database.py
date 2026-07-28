"""Chapter 4 flyby module: powered-flyby database construction.

This module computes flyby entries from existing EncounterDB + LegDB objects.
Scope here is strictly the Flyby Database builder (no filtering/pruning/combos).

See NOTATION.md for the IE/IL/IF index symbols and stage conventions.
"""

from __future__ import annotations

from typing import Callable, Dict, List, Mapping, Set, Tuple

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import spiceypy as spice

import time

from star.constants import DEFAULT_FRAME, DEFAULT_OBSERVER, get_gm
from star.constants.time import et_seconds_to_j2000_days, j2000_days_to_et_seconds
from star.stage_db_npy import (
    begin_flyby_stage_writer,
    load_active_mask,
    load_flyby_column,
    load_leg_column,
    update_active_mask,
)
from star.encounter_database import (
    EncounterDB,
    EncounterDBConfig,
    EncounterEntry,
    EncounterStageConfig,
    tighten_stage_time_bounds_days,
    build_encounter_database,
)
from star.leg_database import LegDatabase, build_leg_db, LegBuildConfig, encounter_filter

# Maximum number of (incoming, outgoing) pair elements per vectorized chunk.
# At 5 M float64 elements each, peak memory per array stays under ~40 MB.
_FLYBY_CHUNK_PAIRS: int = 5_000_000
_FLYBY_SPARSE_DENSITY_THRESHOLD: float = 0.15
_FILTER_UNIQUE_CHUNK_ROWS: int = 5_000_000

# Minimum elapsed time [s] between progress prints (change to taste).
_PROGRESS_INTERVAL_S: float = 300.0


@dataclass(frozen=True)
class FlybyBuildConfig:
    """Configuration for flyby database generation.

    Attributes:
        numerical_eps: Small positive threshold for norm/denominator checks.
        trip_tof_bounds_s_by_stage:
            Optional map from flyby `stage_id` -> `(tof_min_s, tof_max_s)` for the
            three-encounter triplet `(stage_id-1, stage_id, stage_id+1)` per STAR Eq. (32).
        dv_patch_max_km_s_by_stage:
            Optional map from flyby `stage_id` -> max allowed local patch DV [km/s].
            If provided, flyby pairs with computed `dv_km_s` above the stage limit
            are pruned during FlybyDB construction.
        central_mu_km3_s2:
            Optional observer/central-body GM [km^3/s^2]. Required only when
            `flyby_filter` is provided.
        flyby_filter:
            Optional vectorized callback applied after built-in flyby validity
            checks. Signature:
              `f(stage_id, body_id, r_km, v_body_km_s, vinf_in_km_s, vinf_out_km_s, mu_central_km3_s2)`
            Inputs other than `stage_id` and `mu_central_km3_s2` are row-aligned
            arrays over candidate flyby pairs. The callback must return either a
            scalar bool or a boolean mask of shape `(N,)`.
    """

    numerical_eps: float = 1e-12
    trip_tof_bounds_s_by_stage: Mapping[int, Tuple[float, float]] | None = None
    dv_patch_max_km_s_by_stage: Mapping[int, float] | None = None
    central_mu_km3_s2: float | None = None
    flyby_filter: Callable[..., np.ndarray] | None = None
    sparse_density_threshold: float = _FLYBY_SPARSE_DENSITY_THRESHOLD


@dataclass
class FlybyDB:
    """Flyby database columns.

    Required columns:
        IF, stage_id, IE, IL_in, IL_out, dv_km_s
    """

    IF: np.ndarray
    stage_id: np.ndarray
    IE: np.ndarray
    IL_in: np.ndarray
    IL_out: np.ndarray
    dv_km_s: np.ndarray


@dataclass
class ActiveIndexCache:
    """RAM cache for active leg/flyby index filtering."""

    leg_il: Dict[int, np.ndarray]
    leg_id: Dict[int, np.ndarray]
    leg_ia: Dict[int, np.ndarray]
    leg_mask: Dict[int, np.ndarray]

    flyby_il_in: Dict[int, np.ndarray]
    flyby_il_out: Dict[int, np.ndarray]
    flyby_mask: Dict[int, np.ndarray]

    dirty_leg_stages: Set[int]
    dirty_flyby_stages: Set[int]


@dataclass(frozen=True)
class _DiskLegFilterDirection:
    """One directional variant of the disk-backed leg/flyby propagation filter."""

    required_flyby_column: str
    leg_stage_offset: int
    propagate_flyby_stage_offset: int
    propagate_flyby_column: str


_INCOMING_DISK_FILTER = _DiskLegFilterDirection(
    required_flyby_column="IL_in",
    leg_stage_offset=-1,
    propagate_flyby_stage_offset=-1,
    propagate_flyby_column="IL_out",
)

_OUTGOING_DISK_FILTER = _DiskLegFilterDirection(
    required_flyby_column="IL_out",
    leg_stage_offset=0,
    propagate_flyby_stage_offset=1,
    propagate_flyby_column="IL_in",
)


def _build_leg_index_map(encounter_ids: np.ndarray, leg_ids: np.ndarray) -> Dict[int, np.ndarray]:
    """Build map from encounter IE -> sorted leg-row indices (sorted by IL)."""

    encounter_ids_array = np.asarray(encounter_ids, dtype=np.int64)
    leg_ids_array = np.asarray(leg_ids, dtype=np.int64)
    row_indices = np.arange(encounter_ids_array.size, dtype=np.int64)

    # Deterministic leg ordering inside each IE group is by increasing IL.
    sort_order = np.argsort(leg_ids_array, kind="mergesort")
    row_sorted = row_indices[sort_order]
    ie_sorted = encounter_ids_array[sort_order]

    index_map: Dict[int, List[int]] = {}
    for row_index, encounter_ie in zip(row_sorted, ie_sorted):
        index_map.setdefault(int(encounter_ie), []).append(int(row_index))

    return {ie: np.asarray(rows, dtype=np.int64) for ie, rows in index_map.items()}


def compute_parabolic_escape_dv(
    vinf_km_s: float,
    mu_km3_s2: float,
    rmin_km: float,
) -> float:
    """Compute Section 4.3 escape/insertion DV above parabolic reference speed.

    Uses:
        DV = sqrt(vinf^2 + 2*mu/rmin) - sqrt(2*mu/rmin)
    where the reference speed is parabolic `sqrt(2*mu/rmin)`.
    """

    vinf = float(vinf_km_s)
    mu = float(mu_km3_s2)
    rmin = float(rmin_km)

    if not np.isfinite(vinf) or not np.isfinite(mu) or not np.isfinite(rmin):
        return 0.0
    if vinf < 0.0 or mu <= 0.0 or rmin <= 0.0:
        return 0.0

    base_term = 2.0 * mu / rmin
    if base_term <= 0.0:
        return 0.0
    return float(np.sqrt(vinf * vinf + base_term) - np.sqrt(base_term))


def subset_flyby_db(flyby_db: FlybyDB, mask: np.ndarray) -> FlybyDB:
    """Create a FlybyDB subset with stable row order."""

    mask_array = np.asarray(mask, dtype=bool)
    return FlybyDB(
        IF=flyby_db.IF[mask_array],
        stage_id=flyby_db.stage_id[mask_array],
        IE=flyby_db.IE[mask_array],
        IL_in=flyby_db.IL_in[mask_array],
        IL_out=flyby_db.IL_out[mask_array],
        dv_km_s=flyby_db.dv_km_s[mask_array],
    )


def split_flyby_db_by_stage(flyby_db: FlybyDB) -> Dict[int, FlybyDB]:
    """Split one FlybyDB into stage-scoped FlybyDB objects.

    This keeps each dictionary value restricted to a single `stage_id`,
    which is the shape expected by combo/filter orchestration.
    """

    flyby_dbs_by_stage: Dict[int, FlybyDB] = {}
    stage_ids = np.asarray(flyby_db.stage_id, dtype=np.int64)
    for stage_id in np.unique(stage_ids):
        stage_int = int(stage_id)
        stage_mask = stage_ids == stage_int
        flyby_dbs_by_stage[stage_int] = subset_flyby_db(flyby_db, stage_mask)
    return flyby_dbs_by_stage


def _validate_leg_references(enc_db: EncounterDB, leg_dbs: Mapping[int, LegDatabase]) -> None:
    """Validate that every encounter IE referenced by leg DBs exists in EncounterDB."""

    valid_ie = set(int(entry.IE) for entry in enc_db.entries)
    for leg_stage_id, leg_db in leg_dbs.items():
        referenced_ie = np.concatenate(
            (np.asarray(leg_db.ID, dtype=np.int64), np.asarray(leg_db.IA, dtype=np.int64))
        )
        bad_ie = [int(ie) for ie in np.unique(referenced_ie) if int(ie) not in valid_ie]
        if bad_ie:
            raise ValueError(
                f"LegDB stage {leg_stage_id} references encounter IDs not present in EncounterDB: {bad_ie}"
            )


def _evaluate_flyby_chunk_pairs(
    *,
    vinf_in_c: np.ndarray,
    vin_c: np.ndarray,
    vinf_out_mat: np.ndarray,
    vout: np.ndarray,
    base_valid_mask: np.ndarray,
    mu_km3_s2: float,
    rp_km: float,
    dv_patch_limit_km_s: float | None,
    eps: float,
    sparse_density_threshold: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Evaluate one (C,O) chunk with dense/sparse hybrid math."""

    valid_seed = np.asarray(base_valid_mask, dtype=bool)
    if valid_seed.ndim != 2:
        raise ValueError("base_valid_mask must be 2D with shape (C, O).")
    if int(valid_seed.size) == 0:
        empty_i = np.empty(0, dtype=np.int64)
        return empty_i, empty_i, np.empty(0, dtype=float)

    n_seed = int(np.count_nonzero(valid_seed))
    if n_seed == 0:
        empty_i = np.empty(0, dtype=np.int64)
        return empty_i, empty_i, np.empty(0, dtype=float)

    # Apply the cheap |vinf_in|-|vinf_out| lower bound before choosing
    # sparse vs dense evaluation. This keeps the exact feasible set but
    # avoids selecting the dense path from a TOF-only mask that is later
    # thinned substantially by the DV lower bound.
    valid = np.asarray(valid_seed, dtype=bool).copy()
    dv_limit_with_eps = None
    if dv_patch_limit_km_s is not None:
        dv_limit_with_eps = float(dv_patch_limit_km_s) + float(eps)
        valid &= np.abs(vin_c[:, np.newaxis] - vout[np.newaxis, :]) <= dv_limit_with_eps
        if not np.any(valid):
            empty_i = np.empty(0, dtype=np.int64)
            return empty_i, empty_i, np.empty(0, dtype=float)

    density = float(np.count_nonzero(valid)) / float(max(int(valid.size), 1))
    use_sparse = density < float(sparse_density_threshold)

    # sparse path: gather candidates then evaluate only those pairs; faster when valid_seed is very sparse
    if use_sparse:
        ci_seed, o_seed = np.nonzero(valid)
        vin_seed = vin_c[ci_seed]
        vout_seed = vout[o_seed]

        vinf_in_sel = vinf_in_c[ci_seed]
        vinf_out_sel = vinf_out_mat[o_seed]

        with np.errstate(divide="ignore", invalid="ignore"):
            dot_prod = np.einsum("ij,ij->i", vinf_in_sel, vinf_out_sel)
            theta = np.clip(dot_prod / (vin_seed * vout_seed), -1.0, 1.0)
        theta_rad = np.arccos(theta)

        if mu_km3_s2 < 1.0:
            delta_max = np.zeros_like(theta_rad)
        else:
            min_vinf = np.minimum(vin_seed, vout_seed)
            bend_arg = np.clip(
                1.0 / (1.0 + min_vinf * min_vinf * (rp_km / mu_km3_s2)),
                -1.0,
                1.0,
            )
            delta_max = 2.0 * np.arcsin(bend_arg)

        beta = np.maximum(0.0, theta_rad - delta_max)
        dv_sq = (vin_seed - vout_seed) ** 2 + 2.0 * (1.0 - np.cos(beta)) * vin_seed * vout_seed
        dv = np.sqrt(np.maximum(0.0, dv_sq))

        keep = np.isfinite(dv)
        if dv_limit_with_eps is not None:
            keep &= dv <= dv_limit_with_eps
        if not np.any(keep):
            empty_i = np.empty(0, dtype=np.int64)
            return empty_i, empty_i, np.empty(0, dtype=float)

        ci_idx = np.asarray(ci_seed[keep], dtype=np.int64)
        o_idx = np.asarray(o_seed[keep], dtype=np.int64)
        dv_sel = np.asarray(dv[keep], dtype=float)
        return ci_idx, o_idx, dv_sel

    # dense path: evaluate all pairs then mask at the end
    with np.errstate(divide="ignore", invalid="ignore"):
        dot_prod = vinf_in_c @ vinf_out_mat.T
        norm_prod = vin_c[:, np.newaxis] * vout[np.newaxis, :]
        theta = np.clip(dot_prod / norm_prod, -1.0, 1.0)
    theta_rad = np.arccos(theta)

    if mu_km3_s2 < 1.0:
        delta_max = np.zeros_like(theta_rad)
    else:
        min_vinf = np.minimum(vin_c[:, np.newaxis], vout[np.newaxis, :])
        bend_arg = np.clip(
            1.0 / (1.0 + min_vinf * min_vinf * (rp_km / mu_km3_s2)),
            -1.0,
            1.0,
        )
        delta_max = 2.0 * np.arcsin(bend_arg)

    beta = np.maximum(0.0, theta_rad - delta_max)
    dv_sq = (
        (vin_c[:, np.newaxis] - vout[np.newaxis, :]) ** 2
        + 2.0 * (1.0 - np.cos(beta)) * vin_c[:, np.newaxis] * vout[np.newaxis, :]
    )
    dv = np.sqrt(np.maximum(0.0, dv_sq))

    valid &= np.isfinite(dv)
    if dv_limit_with_eps is not None:
        valid &= dv <= dv_limit_with_eps

    ci_idx, o_idx = np.nonzero(valid)
    if ci_idx.size == 0:
        empty_i = np.empty(0, dtype=np.int64)
        return empty_i, empty_i, np.empty(0, dtype=float)
    return (
        np.asarray(ci_idx, dtype=np.int64),
        np.asarray(o_idx, dtype=np.int64),
        np.asarray(dv[ci_idx, o_idx], dtype=float),
    )


def build_flyby_db(
    enc_db: EncounterDB,
    leg_dbs: Dict[int, LegDatabase],
    cfg: FlybyBuildConfig,
    *,
    stage_ids: List[int] | np.ndarray | None = None,
    row_emit: Callable[[np.ndarray, np.ndarray, np.ndarray, np.ndarray], None] | None = None,
) -> FlybyDB:
    """Build flyby database for powered flyby at intermediate encounters.

    Core assumptions used here:
    - Incoming asymptote vector is `vinfA_km_s` from incoming leg (stage s-1 -> s).
    - Outgoing asymptote vector is `vinfD_km_s` from outgoing leg (stage s -> s+1).
    - Periapsis radius is fixed to encounter `rmin_km`.
    - Powered flyby DV follows STAR Eqs. (36)-(39):
      `theta = angle(vinf_in, vinf_out)`,
      `delta_max = 2*asin(1/(1 + min(vinf_in, vinf_out)^2 * rp / mu))`,
      `beta = max(0, theta - delta_max)`,
      `dv = sqrt(vinf_in^2 + vinf_out^2 - 2*vinf_in*vinf_out*cos(beta))`.

    Deterministic ordering:
    - Stage `s` ascending.
    - Encounter entries at stage `s` in EncounterDB order.
    - Incoming legs for that IE by increasing IL.
    - Outgoing legs for that IE by increasing IL.
    """

    if len(leg_dbs) == 0:
        raise ValueError("leg_dbs must include at least one leg database.")

    eps = float(cfg.numerical_eps)
    if eps <= 0.0:
        raise ValueError("numerical_eps must be positive.")
    sparse_density_threshold = float(cfg.sparse_density_threshold)
    if (not np.isfinite(sparse_density_threshold)) or sparse_density_threshold <= 0.0 or sparse_density_threshold >= 1.0:
        raise ValueError("sparse_density_threshold must be finite and satisfy 0 < threshold < 1.")
    trip_tof_bounds_by_stage: Dict[int, Tuple[float, float]] = {}
    if cfg.trip_tof_bounds_s_by_stage is not None:
        for stage_raw, tof_bounds_raw in dict(cfg.trip_tof_bounds_s_by_stage).items():
            stage_id = int(stage_raw)
            tof_min_s, tof_max_s = map(float, tof_bounds_raw)
            if stage_id < 0:
                raise ValueError("trip_tof_bounds_s_by_stage keys must be non-negative stage IDs.")
            if np.isfinite(tof_min_s) and np.isfinite(tof_max_s) and tof_max_s < tof_min_s:
                raise ValueError("trip_tof_bounds_s_by_stage values must satisfy tof_max >= tof_min.")
            trip_tof_bounds_by_stage[stage_id] = (tof_min_s, tof_max_s)

    dv_patch_max_by_stage: Dict[int, float] = {}
    if cfg.dv_patch_max_km_s_by_stage is not None:
        for stage_raw, dv_limit_raw in dict(cfg.dv_patch_max_km_s_by_stage).items():
            stage_id = int(stage_raw)
            dv_limit_km_s = float(dv_limit_raw)
            if stage_id < 0:
                raise ValueError("dv_patch_max_km_s_by_stage keys must be non-negative stage IDs.")
            if not np.isfinite(dv_limit_km_s) or dv_limit_km_s < 0.0:
                raise ValueError(
                    "dv_patch_max_km_s_by_stage values must be finite and >= 0 [km/s]."
                )
            dv_patch_max_by_stage[stage_id] = dv_limit_km_s

    custom_flyby_filter = cfg.flyby_filter
    mu_central_km3_s2 = None
    if custom_flyby_filter is not None:
        if cfg.central_mu_km3_s2 is None:
            raise ValueError("central_mu_km3_s2 must be provided when flyby_filter is enabled.")
        mu_central_km3_s2 = float(cfg.central_mu_km3_s2)
        if not np.isfinite(mu_central_km3_s2) or mu_central_km3_s2 <= 0.0:
            raise ValueError("central_mu_km3_s2 must be finite and > 0 when flyby_filter is enabled.")

    _validate_leg_references(enc_db, leg_dbs)

    # Index encounter entries by stage while preserving EncounterDB order.
    selected_stages = None if stage_ids is None else set(int(stage_id) for stage_id in stage_ids)
    stage_to_entries: Dict[int, List[EncounterEntry]] = {}
    for entry in enc_db.entries:
        stage_int = int(entry.stage_id)
        if selected_stages is not None and stage_int not in selected_stages:
            continue
        stage_to_entries.setdefault(stage_int, []).append(entry)
    max_ie = max((int(entry.IE) for entry in enc_db.entries), default=-1)
    encounter_time_lut = np.empty(max_ie + 1, dtype=float) if max_ie >= 0 else np.empty(0, dtype=float)
    if max_ie >= 0:
        encounter_time_lut.fill(np.nan)
        for entry in enc_db.entries:
            encounter_time_lut[int(entry.IE)] = float(entry.t_et)

    # Output columns (built as Python lists then converted to NumPy arrays).
    # In streaming mode (`row_emit` set), rows are emitted incrementally instead.
    out_stage_id: List[int] = []
    out_ie: List[int] = []
    out_il_in: List[int] = []
    out_il_out: List[int] = []
    out_dv: List[float] = []

    _last_print_t = time.monotonic()
    for stage_id in sorted(stage_to_entries.keys()):
        incoming_leg_db = leg_dbs.get(stage_id - 1)
        outgoing_leg_db = leg_dbs.get(stage_id)
        trip_tof_bounds_s = trip_tof_bounds_by_stage.get(int(stage_id), None)
        dv_patch_limit_km_s = dv_patch_max_by_stage.get(int(stage_id), None)

        # Intermediate encounter stage must have both incoming and outgoing leg DBs.
        if incoming_leg_db is None or outgoing_leg_db is None:
            continue

        incoming_by_arrival_ie = _build_leg_index_map(incoming_leg_db.IA, incoming_leg_db.IL)
        outgoing_by_departure_ie = _build_leg_index_map(outgoing_leg_db.ID, outgoing_leg_db.IL)

        n_enc = len(stage_to_entries[stage_id])
        for enc_i, encounter_entry in enumerate(stage_to_entries[stage_id]):
            encounter_ie = int(encounter_entry.IE)
            incoming_rows = incoming_by_arrival_ie.get(encounter_ie, np.empty(0, dtype=np.int64))
            outgoing_rows = outgoing_by_departure_ie.get(encounter_ie, np.empty(0, dtype=np.int64))

            if incoming_rows.size == 0 or outgoing_rows.size == 0:
                continue

            _now = time.monotonic()
            if _now - _last_print_t >= _PROGRESS_INTERVAL_S:
                _last_print_t = _now
                print(
                    f"\r  Flyby stage {stage_id} | enc {enc_i+1:3d}/{n_enc:3d} | IE={encounter_ie} | "
                    f"{incoming_rows.size}x{outgoing_rows.size} pairs",
                    end="",
                    flush=True,
                )

            mu_km3_s2 = float(encounter_entry.mu_km3_s2)
            rp_km = float(encounter_entry.rmin_km)
            encounter_r_km = np.asarray(encounter_entry.r_km, dtype=float).reshape(1, 3)
            encounter_v_km_s = np.asarray(encounter_entry.v_km_s, dtype=float).reshape(1, 3)
            encounter_body = int(encounter_entry.body)
            if mu_km3_s2 <= eps or rp_km <= eps:
                # Corrupt encounter constants; skip all pairs at this encounter.
                continue

            # Vectorized computation over all (incoming × outgoing) pairs for this encounter.
            # Pre-gather arrays once — avoids repeated per-row indexing inside any loop.
            vinf_in_mat = np.asarray(incoming_leg_db.vinfA_km_s[incoming_rows], dtype=float)   # (I, 3)
            vinf_out_mat = np.asarray(outgoing_leg_db.vinfD_km_s[outgoing_rows], dtype=float)  # (O, 3)
            vin = np.linalg.norm(vinf_in_mat, axis=1)   # (I,)
            vout = np.linalg.norm(vinf_out_mat, axis=1)  # (O,)
            il_in_arr = np.asarray(incoming_leg_db.IL[incoming_rows], dtype=np.int64)           # (I,)
            il_out_arr = np.asarray(outgoing_leg_db.IL[outgoing_rows], dtype=np.int64)          # (O,)
            tD_in = encounter_time_lut[np.asarray(incoming_leg_db.ID[incoming_rows], dtype=np.int64)]   # (I,)
            tA_out = encounter_time_lut[np.asarray(outgoing_leg_db.IA[outgoing_rows], dtype=np.int64)]  # (O,)

            n_in = incoming_rows.size
            n_out = outgoing_rows.size

            # Chunk over incoming rows so each (C × O) working array stays bounded. (C: chunk size)
            chunk_in = max(1, _FLYBY_CHUNK_PAIRS // max(n_out, 1))
            for i_start in range(0, n_in, chunk_in):
                i_end = min(i_start + chunk_in, n_in)
                vinf_in_c = vinf_in_mat[i_start:i_end]   # (C, 3)
                vin_c     = vin[i_start:i_end]             # (C,)
                il_in_c   = il_in_arr[i_start:i_end]       # (C,)

                # --- TOF filter first: cheapest operation, prunes the most ---
                if trip_tof_bounds_s is not None:
                    tD_in_c  = tD_in[i_start:i_end]
                    trip_tof = tA_out[np.newaxis, :] - tD_in_c[:, np.newaxis]   # (C, O)
                    tof_min_s, tof_max_s = trip_tof_bounds_s
                    valid = np.ones((i_end - i_start, n_out), dtype=bool)
                    if np.isfinite(tof_min_s):
                        valid &= trip_tof >= tof_min_s
                    if np.isfinite(tof_max_s):
                        valid &= trip_tof <= tof_max_s
                else:
                    valid = np.ones((i_end - i_start, n_out), dtype=bool)

                # --- zero-norm guard ---
                valid &= (vin_c[:, np.newaxis] > eps) & (vout[np.newaxis, :] > eps)

                if not np.any(valid):
                    continue

                # --- theta: one BLAS matrix multiply replaces dot calls ---
                ci_idx, o_idx, dv_sel = _evaluate_flyby_chunk_pairs(
                    vinf_in_c=vinf_in_c,
                    vin_c=vin_c,
                    vinf_out_mat=vinf_out_mat,
                    vout=vout,
                    base_valid_mask=valid,
                    mu_km3_s2=mu_km3_s2,
                    rp_km=rp_km,
                    dv_patch_limit_km_s=dv_patch_limit_km_s,
                    eps=eps,
                    sparse_density_threshold=sparse_density_threshold,
                )
                if ci_idx.size == 0:
                    continue

                if custom_flyby_filter is not None:
                    pair_count = int(ci_idx.size)
                    body_id = np.full(pair_count, encounter_body, dtype=np.int64)
                    r_km = np.broadcast_to(encounter_r_km, (pair_count, 3))
                    v_body_km_s = np.broadcast_to(encounter_v_km_s, (pair_count, 3))
                    vinf_in_km_s = vinf_in_c[ci_idx]
                    vinf_out_km_s = vinf_out_mat[o_idx]
                    try:
                        custom_keep = custom_flyby_filter(
                            int(stage_id),
                            body_id,
                            r_km,
                            v_body_km_s,
                            vinf_in_km_s,
                            vinf_out_km_s,
                            float(mu_central_km3_s2),
                        )
                    except Exception as exc:
                        raise RuntimeError(
                            f"flyby_filter failed for stage_id={int(stage_id)}, encounter IE={encounter_ie}."
                        ) from exc

                    custom_keep_array = np.asarray(custom_keep, dtype=bool)
                    if custom_keep_array.ndim == 0:
                        custom_keep_array = np.full(pair_count, bool(custom_keep_array), dtype=bool)
                    else:
                        custom_keep_array = custom_keep_array.reshape(-1)
                    if custom_keep_array.shape != (pair_count,):
                        raise ValueError(
                            "flyby_filter must return a scalar bool or a boolean mask with one entry per candidate pair."
                        )

                    ci_idx = ci_idx[custom_keep_array]
                    o_idx = o_idx[custom_keep_array]
                    dv_sel = dv_sel[custom_keep_array]
                    if ci_idx.size == 0:
                        continue

                ie_chunk = np.full(int(ci_idx.size), encounter_ie, dtype=np.int64)
                il_in_chunk = np.asarray(il_in_c[ci_idx], dtype=np.int64)
                il_out_chunk = np.asarray(il_out_arr[o_idx], dtype=np.int64)
                dv_chunk = np.asarray(dv_sel, dtype=float)

                if row_emit is None:
                    out_stage_id.extend([int(stage_id)] * int(ci_idx.size))
                    out_ie.extend(ie_chunk.tolist())
                    out_il_in.extend(il_in_chunk.tolist())
                    out_il_out.extend(il_out_chunk.tolist())
                    out_dv.extend(dv_chunk.tolist())
                else:
                    row_emit(ie_chunk, il_in_chunk, il_out_chunk, dv_chunk)

    if row_emit is not None:
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

    if_array = np.arange(len(out_dv), dtype=np.int64)

    stage_id_array = np.asarray(out_stage_id, dtype=np.int64)
    ie_array = np.asarray(out_ie, dtype=np.int64)
    il_in_array = np.asarray(out_il_in, dtype=np.int64)
    il_out_array = np.asarray(out_il_out, dtype=np.int64)
    dv_array = np.asarray(out_dv, dtype=float)

    return FlybyDB(
        IF=if_array,
        stage_id=stage_id_array,
        IE=ie_array,
        IL_in=il_in_array,
        IL_out=il_out_array,
        dv_km_s=dv_array,
    )


def build_flyby_stage_to_disk(
    enc_db: EncounterDB,
    incoming_leg_db: LegDatabase,
    outgoing_leg_db: LegDatabase,
    cfg: FlybyBuildConfig,
    *,
    stage_id: int,
    stage_dir: str | Path,
    if_start: int = 0,
    flush_rows: int = 2_000_000,
    precomputed_rows: FlybyDB | None = None,
) -> int:
    """Build one flyby stage and stream rows to disk in flush-size batches.

    When `precomputed_rows` is provided, those rows are written first and the
    streamed generic rows follow with continuous IF numbering.
    """

    flush_rows_int = int(flush_rows)
    if flush_rows_int <= 0:
        raise ValueError("flush_rows must be > 0.")

    writer = begin_flyby_stage_writer(stage_dir, stage_id=int(stage_id), if_start=int(if_start))
    ie_chunks: List[np.ndarray] = []
    il_in_chunks: List[np.ndarray] = []
    il_out_chunks: List[np.ndarray] = []
    dv_chunks: List[np.ndarray] = []
    buffered_rows = 0

    def _flush_buffer() -> None:
        nonlocal buffered_rows
        if buffered_rows <= 0:
            return
        writer.append_rows(
            ie=np.concatenate(ie_chunks),
            il_in=np.concatenate(il_in_chunks),
            il_out=np.concatenate(il_out_chunks),
            dv_km_s=np.concatenate(dv_chunks),
        )
        ie_chunks.clear()
        il_in_chunks.clear()
        il_out_chunks.clear()
        dv_chunks.clear()
        buffered_rows = 0

    def _emit_rows(ie: np.ndarray, il_in: np.ndarray, il_out: np.ndarray, dv_km_s: np.ndarray) -> None:
        nonlocal buffered_rows
        row_count = int(np.asarray(ie).size)
        if row_count == 0:
            return
        ie_chunks.append(np.asarray(ie, dtype=np.int64).reshape(-1))
        il_in_chunks.append(np.asarray(il_in, dtype=np.int64).reshape(-1))
        il_out_chunks.append(np.asarray(il_out, dtype=np.int64).reshape(-1))
        dv_chunks.append(np.asarray(dv_km_s, dtype=float).reshape(-1))
        buffered_rows += row_count
        if buffered_rows >= flush_rows_int:
            _flush_buffer()

    try:
        if precomputed_rows is not None:
            _emit_rows(
                np.asarray(precomputed_rows.IE, dtype=np.int64),
                np.asarray(precomputed_rows.IL_in, dtype=np.int64),
                np.asarray(precomputed_rows.IL_out, dtype=np.int64),
                np.asarray(precomputed_rows.dv_km_s, dtype=float),
            )
        build_flyby_db(
            enc_db,
            {int(stage_id - 1): incoming_leg_db, int(stage_id): outgoing_leg_db},
            cfg,
            stage_ids=[int(stage_id)],
            row_emit=_emit_rows,
        )
        _flush_buffer()
        return int(writer.finalize())
    except Exception:
        writer.abort()
        raise


###### LEG FILTERs #############


def _filter_leg_db_by_il(leg_db: LegDatabase, keep_il: np.ndarray) -> int:
    """Filter one LegDB in-place by allowed `IL` IDs."""

    keep_il_array = np.asarray(keep_il, dtype=np.int64)
    before = int(leg_db.IL.size)
    mask = np.isin(np.asarray(leg_db.IL, dtype=np.int64), keep_il_array)

    leg_db.IL = leg_db.IL[mask]
    leg_db.ID = leg_db.ID[mask]
    leg_db.IA = leg_db.IA[mask]
    leg_db.vinfD_km_s = leg_db.vinfD_km_s[mask]
    leg_db.vinfA_km_s = leg_db.vinfA_km_s[mask]
    leg_db.n_rev = leg_db.n_rev[mask]
    leg_db.dv_lev_km_s = leg_db.dv_lev_km_s[mask]
    leg_db.eta_lev = leg_db.eta_lev[mask]

    after = int(leg_db.IL.size)
    return before - after


def _filter_flyby_db_by_mask(flyby_db: FlybyDB, mask: np.ndarray) -> int:
    """Apply boolean row mask to FlybyDB in-place."""

    before = int(flyby_db.IF.size)
    mask_array = np.asarray(mask, dtype=bool)

    flyby_db.IF = flyby_db.IF[mask_array]
    flyby_db.stage_id = flyby_db.stage_id[mask_array]
    flyby_db.IE = flyby_db.IE[mask_array]
    flyby_db.IL_in = flyby_db.IL_in[mask_array]
    flyby_db.IL_out = flyby_db.IL_out[mask_array]
    flyby_db.dv_km_s = flyby_db.dv_km_s[mask_array]

    after = int(flyby_db.IF.size)
    return before - after


def create_active_index_cache() -> ActiveIndexCache:
    """Create an empty active-index cache."""

    return ActiveIndexCache(
        leg_il={},
        leg_id={},
        leg_ia={},
        leg_mask={},
        flyby_il_in={},
        flyby_il_out={},
        flyby_mask={},
        dirty_leg_stages=set(),
        dirty_flyby_stages=set(),
    )


def add_leg_stage_to_active_index_cache(
    stage_id: int,
    stage_dir: str | Path,
    index_cache: ActiveIndexCache,
    *,
    active_leg_ie_cache: Dict[int, Dict[str, np.ndarray]] | None = None,
) -> None:
    """Load one leg stage index columns + mask into RAM cache."""

    stage_int = int(stage_id)
    il = np.asarray(load_leg_column(stage_dir, "IL", active_only=False), dtype=np.int64).reshape(-1)
    dep = np.asarray(load_leg_column(stage_dir, "ID", active_only=False), dtype=np.int64).reshape(-1)
    arr = np.asarray(load_leg_column(stage_dir, "IA", active_only=False), dtype=np.int64).reshape(-1)
    mask = np.asarray(load_active_mask(stage_dir), dtype=bool).reshape(-1)

    if int(il.size) != int(mask.size) or int(dep.size) != int(mask.size) or int(arr.size) != int(mask.size):
        raise ValueError(f"Leg stage {stage_int} cache load size mismatch.")

    index_cache.leg_il[stage_int] = il
    index_cache.leg_id[stage_int] = dep
    index_cache.leg_ia[stage_int] = arr
    index_cache.leg_mask[stage_int] = mask

    if active_leg_ie_cache is not None:
        active_leg_ie_cache[stage_int] = {
            "ID": np.unique(dep[mask]),
            "IA": np.unique(arr[mask]),
        }


def add_flyby_stage_to_active_index_cache(
    stage_id: int,
    stage_dir: str | Path,
    index_cache: ActiveIndexCache,
) -> None:
    """Load one flyby stage index columns + mask into RAM cache."""

    stage_int = int(stage_id)
    il_in = np.asarray(load_flyby_column(stage_dir, "IL_in", active_only=False), dtype=np.int64).reshape(-1)
    il_out = np.asarray(load_flyby_column(stage_dir, "IL_out", active_only=False), dtype=np.int64).reshape(-1)
    mask = np.asarray(load_active_mask(stage_dir), dtype=bool).reshape(-1)

    if int(il_in.size) != int(mask.size) or int(il_out.size) != int(mask.size):
        raise ValueError(f"Flyby stage {stage_int} cache load size mismatch.")

    index_cache.flyby_il_in[stage_int] = il_in
    index_cache.flyby_il_out[stage_int] = il_out
    index_cache.flyby_mask[stage_int] = mask


def flush_active_index_cache_masks(
    leg_stage_dirs: Mapping[int, str | Path],
    flyby_stage_dirs: Mapping[int, str | Path],
    index_cache: ActiveIndexCache,
) -> None:
    """Persist dirty active masks from RAM cache to disk."""

    for stage_id in sorted(index_cache.dirty_leg_stages):
        stage_dir = leg_stage_dirs.get(int(stage_id), None)
        if stage_dir is None:
            continue
        update_active_mask(stage_dir, index_cache.leg_mask[int(stage_id)])
    index_cache.dirty_leg_stages.clear()

    for stage_id in sorted(index_cache.dirty_flyby_stages):
        stage_dir = flyby_stage_dirs.get(int(stage_id), None)
        if stage_dir is None:
            continue
        update_active_mask(stage_dir, index_cache.flyby_mask[int(stage_id)])
    index_cache.dirty_flyby_stages.clear()


def _filter_stage_active_mask_by_ids(
    stage_dir: str | Path,
    *,
    id_column: str,
    keep_ids: np.ndarray,
    is_leg_stage: bool,
) -> int:
    """Apply ID-based survivor filtering to one disk-backed stage mask."""

    active_mask = load_active_mask(stage_dir)
    before = int(np.count_nonzero(active_mask))
    if before == 0:
        return 0

    keep_ids_array = np.asarray(keep_ids, dtype=np.int64).reshape(-1)
    if is_leg_stage:
        id_values = np.asarray(load_leg_column(stage_dir, id_column, active_only=False), dtype=np.int64).reshape(-1)
    else:
        id_values = np.asarray(load_flyby_column(stage_dir, id_column, active_only=False), dtype=np.int64).reshape(-1)

    if int(id_values.size) != int(active_mask.size):
        raise ValueError(
            f"Stage '{stage_dir}' has inconsistent '{id_column}' and active_mask row counts: "
            f"{int(id_values.size)} vs {int(active_mask.size)}."
        )

    if keep_ids_array.size == 0:
        new_mask = np.zeros_like(active_mask, dtype=bool)
    else:
        new_mask = active_mask & np.isin(id_values, keep_ids_array)

    after = int(np.count_nonzero(new_mask))
    removed = int(before - after)
    if removed > 0:
        update_active_mask(stage_dir, new_mask)
    return removed


def _filter_cached_mask_by_ids(
    stage_id: int,
    keep_ids: np.ndarray,
    *,
    masks_by_stage: Dict[int, np.ndarray],
    id_values_by_stage: Dict[int, np.ndarray],
    dirty_stages: Set[int],
) -> int:
    """Apply ID-based survivor filtering to one cached stage mask."""

    stage_int = int(stage_id)
    current = masks_by_stage.get(stage_int, None)
    if current is None:
        return 0

    before = int(np.count_nonzero(current))
    if before == 0:
        return 0

    keep_ids_array = np.asarray(keep_ids, dtype=np.int64).reshape(-1)
    id_values = np.asarray(id_values_by_stage[stage_int], dtype=np.int64).reshape(-1)
    if keep_ids_array.size == 0:
        next_mask = np.zeros_like(current, dtype=bool)
    else:
        next_mask = current & np.isin(id_values, keep_ids_array)

    after = int(np.count_nonzero(next_mask))
    removed = int(before - after)
    if removed <= 0:
        return 0

    masks_by_stage[stage_int] = next_mask
    dirty_stages.add(stage_int)
    return int(removed)


def _filter_leg_stage_by_il_cached(
    stage_id: int,
    keep_il: np.ndarray,
    index_cache: ActiveIndexCache,
    *,
    active_leg_ie_cache: Dict[int, Dict[str, np.ndarray]] | None = None,
) -> int:
    """Apply IL-based filtering to one cached leg stage mask."""

    stage_int = int(stage_id)
    removed = _filter_cached_mask_by_ids(
        stage_int,
        keep_il,
        masks_by_stage=index_cache.leg_mask,
        id_values_by_stage=index_cache.leg_il,
        dirty_stages=index_cache.dirty_leg_stages,
    )
    if removed <= 0:
        return 0

    if active_leg_ie_cache is not None:
        next_mask = index_cache.leg_mask[stage_int]
        dep = index_cache.leg_id[stage_int]
        arr = index_cache.leg_ia[stage_int]
        active_leg_ie_cache[stage_int] = {
            "ID": np.unique(dep[next_mask]),
            "IA": np.unique(arr[next_mask]),
        }
    return int(removed)


def _flyby_il_values_by_stage(
    index_cache: ActiveIndexCache,
    column_name: str,
) -> Dict[int, np.ndarray]:
    """Return the cached per-stage IL_in/IL_out value arrays for a flyby column."""

    if column_name == "IL_in":
        return index_cache.flyby_il_in
    if column_name == "IL_out":
        return index_cache.flyby_il_out
    raise ValueError(f"Unsupported flyby IL column: {column_name}")


def _filter_flyby_stage_by_il_cached(
    stage_id: int,
    keep_il: np.ndarray,
    index_cache: ActiveIndexCache,
    *,
    column_name: str,
) -> int:
    """Apply IL-based filtering to one cached flyby stage mask."""

    return _filter_cached_mask_by_ids(
        stage_id,
        keep_il,
        masks_by_stage=index_cache.flyby_mask,
        id_values_by_stage=_flyby_il_values_by_stage(index_cache, str(column_name)),
        dirty_stages=index_cache.dirty_flyby_stages,
    )


def _get_active_flyby_il_ids(
    stage_id: int,
    *,
    column_name: str,
    flyby_stage_dirs: Mapping[int, str | Path],
    index_cache: ActiveIndexCache | None,
) -> np.ndarray:
    """Return active flyby IL references from one stage."""

    stage_int = int(stage_id)
    if stage_int not in flyby_stage_dirs:
        return np.empty(0, dtype=np.int64)

    if index_cache is None:
        return _unique_active_flyby_il_ids_from_disk(
            flyby_stage_dirs[stage_int],
            column_name=str(column_name),
        )

    flyby_mask = index_cache.flyby_mask.get(stage_int, np.empty(0, dtype=bool))
    flyby_il_all = _flyby_il_values_by_stage(index_cache, str(column_name)).get(
        stage_int, np.empty(0, dtype=np.int64)
    )
    return _chunked_unique_masked_int64(flyby_il_all, flyby_mask)


def _chunked_unique_masked_int64(values: np.ndarray, mask: np.ndarray, *, chunk_rows: int = _FILTER_UNIQUE_CHUNK_ROWS) -> np.ndarray:
    """Return sorted unique int64 values selected by a boolean mask without one giant masked copy."""

    values_array = np.asarray(values, dtype=np.int64).reshape(-1)
    mask_array = np.asarray(mask, dtype=bool).reshape(-1)
    if int(values_array.size) != int(mask_array.size):
        raise ValueError("values/mask length mismatch while collecting active flyby IL IDs.")
    if int(values_array.size) == 0 or not bool(np.any(mask_array)):
        return np.empty(0, dtype=np.int64)

    chunk_rows_int = max(1, int(chunk_rows))
    unique_values = np.empty(0, dtype=np.int64)
    for start in range(0, int(values_array.size), chunk_rows_int):
        end = min(start + chunk_rows_int, int(values_array.size))
        chunk_mask = mask_array[start:end]
        if not bool(np.any(chunk_mask)):
            continue

        chunk_unique = np.unique(values_array[start:end][chunk_mask])
        if int(chunk_unique.size) == 0:
            continue
        if int(unique_values.size) == 0:
            unique_values = chunk_unique
            continue

        unique_values = np.unique(np.concatenate((unique_values, chunk_unique)))
    return unique_values


def _unique_active_flyby_il_ids_from_disk(
    stage_dir: str | Path,
    *,
    column_name: str,
    chunk_rows: int = _FILTER_UNIQUE_CHUNK_ROWS,
) -> np.ndarray:
    """Return sorted unique active flyby IL IDs from disk in bounded-size chunks."""

    values = np.asarray(load_flyby_column(stage_dir, str(column_name), active_only=False), dtype=np.int64).reshape(-1)
    mask = np.asarray(load_active_mask(stage_dir), dtype=bool).reshape(-1)
    return _chunked_unique_masked_int64(values, mask, chunk_rows=chunk_rows)


def _get_active_leg_il_for_stage(
    stage_id: int,
    *,
    leg_stage_dirs: Mapping[int, str | Path],
    index_cache: ActiveIndexCache | None,
) -> np.ndarray:
    """Return active leg IL IDs from one leg stage."""

    stage_int = int(stage_id)
    if stage_int not in leg_stage_dirs:
        return np.empty(0, dtype=np.int64)

    if index_cache is None:
        return np.asarray(load_leg_column(leg_stage_dirs[stage_int], "IL", active_only=True), dtype=np.int64)

    leg_mask = index_cache.leg_mask.get(stage_int, np.empty(0, dtype=bool))
    leg_il_all = index_cache.leg_il.get(stage_int, np.empty(0, dtype=np.int64))
    return np.asarray(leg_il_all[leg_mask], dtype=np.int64)


def _filter_flyby_stage_by_il_ids(
    stage_id: int,
    keep_il: np.ndarray,
    *,
    column_name: str,
    flyby_stage_dirs: Mapping[int, str | Path],
    index_cache: ActiveIndexCache | None,
) -> int:
    """Filter one flyby stage by active IL IDs from the chosen reference column."""

    stage_int = int(stage_id)
    if stage_int not in flyby_stage_dirs:
        return 0

    if index_cache is None:
        return _filter_stage_active_mask_by_ids(
            flyby_stage_dirs[stage_int],
            id_column=str(column_name),
            keep_ids=keep_il,
            is_leg_stage=False,
        )

    return _filter_flyby_stage_by_il_cached(
        stage_int,
        keep_il,
        index_cache,
        column_name=str(column_name),
    )


def _filter_leg_stage_by_il_ids(
    stage_id: int,
    keep_il: np.ndarray,
    *,
    leg_stage_dirs: Mapping[int, str | Path],
    index_cache: ActiveIndexCache | None,
    active_leg_ie_cache: Dict[int, Dict[str, np.ndarray]] | None = None,
) -> int:
    """Filter one leg stage by active IL IDs."""

    stage_int = int(stage_id)
    if stage_int not in leg_stage_dirs:
        return 0

    if index_cache is None:
        return _filter_stage_active_mask_by_ids(
            leg_stage_dirs[stage_int],
            id_column="IL",
            keep_ids=keep_il,
            is_leg_stage=True,
        )

    return _filter_leg_stage_by_il_cached(
        stage_int,
        keep_il,
        index_cache,
        active_leg_ie_cache=active_leg_ie_cache,
    )


def _run_leg_filter_disk_direction(
    i: int,
    leg_stage_dirs: Mapping[int, str | Path],
    flyby_stage_dirs: Mapping[int, str | Path],
    encounter_db: EncounterDB,
    *,
    direction: _DiskLegFilterDirection,
    active_leg_ie_cache: Dict[int, Dict[str, np.ndarray]] | None = None,
    index_cache: ActiveIndexCache | None = None,
) -> int:
    """Run one directional disk-backed leg/flyby propagation filter."""

    stack: List[int] = [int(i)]
    total_removed = 0

    while stack:
        cur = int(stack.pop())
        leg_stage_id = int(cur + direction.leg_stage_offset)
        propagate_flyby_stage_id = int(cur + direction.propagate_flyby_stage_offset)
        if cur not in flyby_stage_dirs or leg_stage_id not in leg_stage_dirs:
            continue

        required_il = _get_active_flyby_il_ids(
            cur,
            column_name=direction.required_flyby_column,
            flyby_stage_dirs=flyby_stage_dirs,
            index_cache=index_cache,
        )
        removed_legs = _filter_leg_stage_by_il_ids(
            leg_stage_id,
            required_il,
            leg_stage_dirs=leg_stage_dirs,
            index_cache=index_cache,
            active_leg_ie_cache=active_leg_ie_cache,
        )
        total_removed += int(removed_legs)
        if removed_legs <= 0:
            continue

        total_removed += int(
            _apply_encounter_filter_for_leg_stage_disk(
                leg_stage_id,
                leg_stage_dirs,
                encounter_db,
                active_leg_ie_cache=active_leg_ie_cache,
                refresh_leg_stage=(index_cache is None),
            )
        )

        if propagate_flyby_stage_id in flyby_stage_dirs:
            valid_leg_il = _get_active_leg_il_for_stage(
                leg_stage_id,
                leg_stage_dirs=leg_stage_dirs,
                index_cache=index_cache,
            )
            total_removed += int(
                _filter_flyby_stage_by_il_ids(
                    propagate_flyby_stage_id,
                    valid_leg_il,
                    column_name=direction.propagate_flyby_column,
                    flyby_stage_dirs=flyby_stage_dirs,
                    index_cache=index_cache,
                )
            )
            stack.append(propagate_flyby_stage_id)

    return int(total_removed)


def _refresh_active_leg_ie_cache_for_stage(
    leg_stage_id: int,
    leg_stage_dirs: Mapping[int, str | Path],
    active_leg_ie_cache: Dict[int, Dict[str, np.ndarray]],
) -> None:
    """Refresh one leg-stage active encounter-ID cache entry from disk."""

    leg_stage_int = int(leg_stage_id)
    if leg_stage_int not in leg_stage_dirs:
        active_leg_ie_cache.pop(leg_stage_int, None)
        return

    stage_dir = leg_stage_dirs[leg_stage_int]
    dep_ids = np.asarray(load_leg_column(stage_dir, "ID", active_only=True), dtype=np.int64).reshape(-1)
    arr_ids = np.asarray(load_leg_column(stage_dir, "IA", active_only=True), dtype=np.int64).reshape(-1)
    active_leg_ie_cache[leg_stage_int] = {
        "ID": np.unique(dep_ids),
        "IA": np.unique(arr_ids),
    }


def _get_active_leg_ie_for_stage(
    leg_stage_id: int,
    leg_stage_dirs: Mapping[int, str | Path],
    active_leg_ie_cache: Dict[int, Dict[str, np.ndarray]] | None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return active departure/arrival encounter IDs for one leg stage."""

    leg_stage_int = int(leg_stage_id)
    if leg_stage_int not in leg_stage_dirs:
        empty = np.empty(0, dtype=np.int64)
        return empty, empty

    if active_leg_ie_cache is None:
        stage_dir = leg_stage_dirs[leg_stage_int]
        dep_ids = np.asarray(load_leg_column(stage_dir, "ID", active_only=True), dtype=np.int64).reshape(-1)
        arr_ids = np.asarray(load_leg_column(stage_dir, "IA", active_only=True), dtype=np.int64).reshape(-1)
        return np.unique(dep_ids), np.unique(arr_ids)

    cached = active_leg_ie_cache.get(leg_stage_int)
    if cached is None:
        _refresh_active_leg_ie_cache_for_stage(leg_stage_int, leg_stage_dirs, active_leg_ie_cache)
        cached = active_leg_ie_cache.get(leg_stage_int)
        if cached is None:
            empty = np.empty(0, dtype=np.int64)
            return empty, empty

    return (
        np.asarray(cached.get("ID", np.empty(0, dtype=np.int64)), dtype=np.int64).reshape(-1),
        np.asarray(cached.get("IA", np.empty(0, dtype=np.int64)), dtype=np.int64).reshape(-1),
    )


def _collect_used_encounter_ids_by_stage_from_active_legs(
    leg_stage_dirs: Mapping[int, str | Path],
    prune_stage_ids: set[int],
    *,
    active_leg_ie_cache: Dict[int, Dict[str, np.ndarray]] | None = None,
) -> Dict[int, set[int]]:
    """Collect used encounter IDs only for requested encounter stages."""

    used_by_stage: Dict[int, set[int]] = {}
    for encounter_stage in sorted(int(stage_id) for stage_id in prune_stage_ids):
        used_encounter_ids: set[int] = set()

        incoming_leg_stage = int(encounter_stage) - 1
        if incoming_leg_stage in leg_stage_dirs:
            _, arr_ids = _get_active_leg_ie_for_stage(
                incoming_leg_stage,
                leg_stage_dirs,
                active_leg_ie_cache,
            )
            if arr_ids.size > 0:
                used_encounter_ids.update(int(value) for value in arr_ids)

        outgoing_leg_stage = int(encounter_stage)
        if outgoing_leg_stage in leg_stage_dirs:
            dep_ids, _ = _get_active_leg_ie_for_stage(
                outgoing_leg_stage,
                leg_stage_dirs,
                active_leg_ie_cache,
            )
            if dep_ids.size > 0:
                used_encounter_ids.update(int(value) for value in dep_ids)

        used_by_stage[int(encounter_stage)] = used_encounter_ids

    return used_by_stage


def encounter_filter_from_active_legs(
    encounter_db: EncounterDB,
    leg_stage_dirs: Mapping[int, str | Path],
    *,
    stages: List[int] | np.ndarray | None = None,
    active_leg_ie_cache: Dict[int, Dict[str, np.ndarray]] | None = None,
    refresh_leg_stages: List[int] | np.ndarray | None = None,
) -> EncounterDB:
    """Encounter-filter variant that uses disk-backed active leg masks."""

    if active_leg_ie_cache is not None and refresh_leg_stages is not None:
        for leg_stage_id in np.asarray(refresh_leg_stages, dtype=np.int64).reshape(-1):
            _refresh_active_leg_ie_cache_for_stage(int(leg_stage_id), leg_stage_dirs, active_leg_ie_cache)

    if stages is None:
        prune_stage_ids = set(int(entry.stage_id) for entry in encounter_db.entries)
    else:
        prune_stage_ids = set(int(stage_id) for stage_id in stages)
    if not prune_stage_ids:
        return encounter_db

    used_encounter_ids_by_stage = _collect_used_encounter_ids_by_stage_from_active_legs(
        leg_stage_dirs,
        prune_stage_ids,
        active_leg_ie_cache=active_leg_ie_cache,
    )
    kept_entries = [
        entry
        for entry in encounter_db.entries
        if (int(entry.stage_id) not in prune_stage_ids)
        or (int(entry.IE) in used_encounter_ids_by_stage.get(int(entry.stage_id), set()))
    ]

    stage_to_entry_ids = {
        (int(stage_id), int(body_id)): []
        for stage_id, body_id in encounter_db.stage_to_entry_ids.keys()
    }
    for entry in kept_entries:
        key = (int(entry.stage_id), int(entry.body))
        if key not in stage_to_entry_ids:
            stage_to_entry_ids[key] = []
        stage_to_entry_ids[key].append(int(entry.IE))

    return EncounterDB(
        entries=kept_entries,
        stage_to_entry_ids=stage_to_entry_ids,
        entry_by_id={int(entry.IE): entry for entry in kept_entries},
    )


def _apply_encounter_filter_for_leg_stage_disk(
    leg_stage_id: int,
    leg_stage_dirs: Mapping[int, str | Path],
    encounter_db: EncounterDB,
    active_leg_ie_cache: Dict[int, Dict[str, np.ndarray]] | None = None,
    *,
    refresh_leg_stage: bool = True,
) -> int:
    """Run encounter filtering for one changed disk-backed leg stage."""

    if int(leg_stage_id) not in leg_stage_dirs:
        return 0

    before = int(len(encounter_db.entries))
    filtered_db = encounter_filter_from_active_legs(
        encounter_db,
        leg_stage_dirs,
        stages=[int(leg_stage_id), int(leg_stage_id) + 1],
        active_leg_ie_cache=active_leg_ie_cache,
        refresh_leg_stages=([int(leg_stage_id)] if bool(refresh_leg_stage) else None),
    )
    encounter_db.entries = filtered_db.entries
    encounter_db.stage_to_entry_ids = filtered_db.stage_to_entry_ids
    encounter_db.entry_by_id = filtered_db.entry_by_id
    after = int(len(encounter_db.entries))
    return int(max(0, before - after))


def _apply_encounter_filter_for_leg_stage(
    leg_stage_id: int,
    leg_dbs: Dict[int, LegDatabase],
    encounter_db: EncounterDB,
) -> int:
    """Run encounter_filter for one changed leg stage and update EncounterDB in-place."""

    if leg_stage_id not in leg_dbs:
        return 0

    # Keep encounters used by all current legs; prune only the two stages touched by this leg.
    before = int(len(encounter_db.entries))
    filtered_db = encounter_filter(
        encounter_db,
        list(leg_dbs.values()),
        stages=[int(leg_stage_id), int(leg_stage_id) + 1],
    )
    encounter_db.entries = filtered_db.entries
    encounter_db.stage_to_entry_ids = filtered_db.stage_to_entry_ids
    encounter_db.entry_by_id = filtered_db.entry_by_id
    after = int(len(encounter_db.entries))
    return int(max(0, before - after))


def _assert_flyby_leg_refs_exist(
    flyby_dbs: Dict[int, FlybyDB],
    leg_dbs: Dict[int, LegDatabase],
) -> None:
    """Assert all flyby leg references exist in adjacent leg DBs."""

    for flyby_stage_id, flyby_db in flyby_dbs.items():
        incoming_leg_stage = int(flyby_stage_id) - 1
        outgoing_leg_stage = int(flyby_stage_id)

        if incoming_leg_stage in leg_dbs:
            valid_in = np.asarray(leg_dbs[incoming_leg_stage].IL, dtype=np.int64)
            if not np.all(np.isin(np.asarray(flyby_db.IL_in, dtype=np.int64), valid_in)):
                raise AssertionError(
                    f"FlybyDB[{flyby_stage_id}] has IL_in references missing in LegDB[{incoming_leg_stage}]."
                )
        if outgoing_leg_stage in leg_dbs:
            valid_out = np.asarray(leg_dbs[outgoing_leg_stage].IL, dtype=np.int64)
            if not np.all(np.isin(np.asarray(flyby_db.IL_out, dtype=np.int64), valid_out)):
                raise AssertionError(
                    f"FlybyDB[{flyby_stage_id}] has IL_out references missing in LegDB[{outgoing_leg_stage}]."
                )


def incoming_leg_filter(
    i: int,
    leg_dbs: Dict[int, LegDatabase],
    flyby_dbs: Dict[int, FlybyDB],
    encounter_db: EncounterDB,
) -> None:
    """Algorithm 2 style backward propagation filter.

    IncomingLegFilter(i):
    1) Keep only legs in `L[i-1]` that are referenced by `F[i].IL_in`.
    2) Run EncounterFilter for the changed leg stage.
    3) If `F[i-1]` exists, remove rows whose `IL_out` is no longer in `L[i-1]`,
       then propagate to `IncomingLegFilter(i-1)`.
    """

    stack: List[int] = [int(i)]

    while stack:
        cur = int(stack.pop())

        if cur not in flyby_dbs or (cur - 1) not in leg_dbs:
            continue

        flyby_cur = flyby_dbs[cur]
        leg_prev = leg_dbs[cur - 1]

        required_il_in = np.unique(np.asarray(flyby_cur.IL_in, dtype=np.int64))
        removed_legs = _filter_leg_db_by_il(leg_prev, required_il_in)
        if removed_legs > 0:
            _apply_encounter_filter_for_leg_stage(cur - 1, leg_dbs, encounter_db)

            # Update previous flyby DB to keep only IL_out still present in updated L[cur-1].
            if (cur - 1) in flyby_dbs:
                valid_il_out = np.asarray(leg_dbs[cur - 1].IL, dtype=np.int64)
                flyby_prev = flyby_dbs[cur - 1]
                mask_prev = np.isin(np.asarray(flyby_prev.IL_out, dtype=np.int64), valid_il_out)
                _filter_flyby_db_by_mask(flyby_prev, mask_prev)

            # Propagate backward if next flyby stage exists.
            if (cur - 1) in flyby_dbs:
                stack.append(cur - 1)

    _assert_flyby_leg_refs_exist(flyby_dbs, leg_dbs)


def outgoing_leg_filter(
    i: int,
    leg_dbs: Dict[int, LegDatabase],
    flyby_dbs: Dict[int, FlybyDB],
    encounter_db: EncounterDB,
) -> None:
    """Algorithm 3 style forward propagation filter.

    OutgoingLegFilter(i):
    1) Keep only legs in `L[i]` referenced by `F[i].IL_out`.
    2) Run EncounterFilter for the changed leg stage.
    3) If `F[i+1]` exists, remove rows whose `IL_in` is no longer in `L[i]`,
       then propagate to `OutgoingLegFilter(i+1)`.
    """

    stack: List[int] = [int(i)]

    while stack:
        cur = int(stack.pop())

        if cur not in flyby_dbs or cur not in leg_dbs:
            continue

        flyby_cur = flyby_dbs[cur]
        leg_cur = leg_dbs[cur]

        required_il_out = np.unique(np.asarray(flyby_cur.IL_out, dtype=np.int64))
        removed_legs = _filter_leg_db_by_il(leg_cur, required_il_out)
        if removed_legs > 0:
            _apply_encounter_filter_for_leg_stage(cur, leg_dbs, encounter_db)

            # Update next flyby DB to keep only IL_in still present in updated L[cur].
            if (cur + 1) in flyby_dbs:
                valid_il_in = np.asarray(leg_dbs[cur].IL, dtype=np.int64)
                flyby_next = flyby_dbs[cur + 1]
                mask_next = np.isin(np.asarray(flyby_next.IL_in, dtype=np.int64), valid_il_in)
                _filter_flyby_db_by_mask(flyby_next, mask_next)

            # Propagate forward if next flyby stage exists.
            if (cur + 1) in flyby_dbs:
                stack.append(cur + 1)

    _assert_flyby_leg_refs_exist(flyby_dbs, leg_dbs)


def incoming_leg_filter_disk(
    i: int,
    leg_stage_dirs: Mapping[int, str | Path],
    flyby_stage_dirs: Mapping[int, str | Path],
    encounter_db: EncounterDB,
    *,
    active_leg_ie_cache: Dict[int, Dict[str, np.ndarray]] | None = None,
    index_cache: ActiveIndexCache | None = None,
) -> int:
    """Disk-backed incoming leg filter using stage active masks."""

    return _run_leg_filter_disk_direction(
        i,
        leg_stage_dirs,
        flyby_stage_dirs,
        encounter_db,
        direction=_INCOMING_DISK_FILTER,
        active_leg_ie_cache=active_leg_ie_cache,
        index_cache=index_cache,
    )


def outgoing_leg_filter_disk(
    i: int,
    leg_stage_dirs: Mapping[int, str | Path],
    flyby_stage_dirs: Mapping[int, str | Path],
    encounter_db: EncounterDB,
    *,
    active_leg_ie_cache: Dict[int, Dict[str, np.ndarray]] | None = None,
    index_cache: ActiveIndexCache | None = None,
) -> int:
    """Disk-backed outgoing leg filter using stage active masks."""

    return _run_leg_filter_disk_direction(
        i,
        leg_stage_dirs,
        flyby_stage_dirs,
        encounter_db,
        direction=_OUTGOING_DISK_FILTER,
        active_leg_ie_cache=active_leg_ie_cache,
        index_cache=index_cache,
    )


if __name__ == "__main__":
    
    # Demo: load example/earth_to_jupiter.py, build EncounterDB, then build LegDBs.
    repo_root = Path(__file__).resolve().parent.parent
    metakernel_path = repo_root / "star" / "METAKERN.tm"
    if not metakernel_path.exists():
        raise FileNotFoundError(f"Meta-kernel not found: {metakernel_path}")

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
    vlim = np.asarray(example.vlim, dtype=float)

    # clean up the tof / time bounds with LPs
    tightened_bounds_days = tighten_stage_time_bounds_days(time_bounds_days, tof_min_days, tof_max_days)
    if not np.all(np.isfinite(tightened_bounds_days)):
        raise ValueError("Encounter stage bounds remain infinite after tightening.")

    # build encounter database
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

    enc_cfg = EncounterDBConfig(
        stages=stage_configs,
        frame=DEFAULT_FRAME,
        observer=DEFAULT_OBSERVER,
    )
    encounter_db = build_encounter_database(enc_cfg)
    print(f"EncounterDB entries: {len(encounter_db.entries)}")

    # build leg databases with lambert solver
    lambert_mu_km3_s2 = float(get_gm("10")[0])
    n_stages = len(bodies_by_stage)
    n_legs = n_stages - 1
    leg_dbs_by_stage: Dict[int, LegDatabase] = {}

    nrev_raw = getattr(example, "Nrev", None)
    if nrev_raw is None:
        nrev_raw = getattr(example, "lambert_nrev", [0] * n_legs)
    hz_raw = getattr(example, "lambert_hz", [1] * n_legs)
    dvlev_max_raw = getattr(example, "dVlev_max", None)
    if dvlev_max_raw is None:
        dvlev_max_raw = getattr(example, "dvlev_max", [0.0] * n_legs)
    delta_dvlev_raw = getattr(example, "delta_dvlev", [0.0] * n_legs)
    vinf_margin_pre_filter_raw = getattr(example, "vinf_margin_pre_filter", None)
    if vinf_margin_pre_filter_raw is None:
        vinf_margin_pre_filter_raw = getattr(example, "vinf_margin_pre_filter_km_s", 10.0)
    lev_type_raw = getattr(example, "lev_type", ["+-", "-+"])

    if np.isscalar(nrev_raw):
        nrev_by_leg = [int(nrev_raw)] * n_legs
    else:
        nrev_by_leg = [int(v) for v in list(nrev_raw)]
    if len(nrev_by_leg) < n_legs:
        raise ValueError(f"Need at least {n_legs} Lambert Nrev entries.")

    if np.isscalar(hz_raw):
        hz_by_leg = [int(hz_raw)] * n_legs
    else:
        hz_by_leg = [int(v) for v in list(hz_raw)]
    if len(hz_by_leg) < n_legs:
        raise ValueError(f"Need at least {n_legs} Lambert hz entries.")

    if np.isscalar(dvlev_max_raw):
        dvlev_max_by_leg = [float(dvlev_max_raw)] * n_legs
    else:
        dvlev_max_by_leg = [float(v) for v in list(dvlev_max_raw)]
    if len(dvlev_max_by_leg) < n_legs:
        raise ValueError(f"Need at least {n_legs} dVlev_max entries.")

    if np.isscalar(delta_dvlev_raw):
        delta_dvlev_by_leg = [float(delta_dvlev_raw)] * n_legs
    else:
        delta_dvlev_by_leg = [float(v) for v in list(delta_dvlev_raw)]
    if len(delta_dvlev_by_leg) < n_legs:
        raise ValueError(f"Need at least {n_legs} delta_dvlev entries.")

    if np.isscalar(vinf_margin_pre_filter_raw):
        vinf_margin_pre_filter_by_leg = [float(vinf_margin_pre_filter_raw)] * n_legs
    else:
        vinf_margin_pre_filter_by_leg = [float(v) for v in list(vinf_margin_pre_filter_raw)]
    if len(vinf_margin_pre_filter_by_leg) < n_legs:
        raise ValueError(f"Need at least {n_legs} vinf_margin_pre_filter entries.")

    if isinstance(lev_type_raw, str):
        lev_type = (lev_type_raw,)
    else:
        lev_type = tuple(str(v) for v in list(lev_type_raw))
    if len(lev_type) == 0:
        raise ValueError("example.lev_type must contain at least one entry.")

    for leg_index in range(n_legs):
        tof_min_s = j2000_days_to_et_seconds(float(tof_min_days[leg_index, leg_index + 1]))
        tof_max_s = j2000_days_to_et_seconds(float(tof_max_days[leg_index, leg_index + 1]))

        nrev_max = int(nrev_by_leg[leg_index])
        if nrev_max < 0:
            raise ValueError(f"example.lambert_nrev[{leg_index}] must be >= 0.")
        hz_leg = int(hz_by_leg[leg_index])
        if hz_leg not in (-1, 0, 1):
            raise ValueError(f"example.lambert_hz[{leg_index}] must be one of -1, 0, 1.")

        cfg = LegBuildConfig(
            leg_stage_id=int(leg_index),
            dep_stage_id=int(leg_index),
            arr_stage_id=int(leg_index + 1),
            tof_min_s=float(tof_min_s),
            tof_max_s=float(tof_max_s),
            vinfD_bounds_km_s=(float(vlim[0, leg_index]), float(vlim[1, leg_index])),
            vinfA_bounds_km_s=(float(vlim[0, leg_index + 1]), float(vlim[1, leg_index + 1])),
            lambert_mu_km3_s2=lambert_mu_km3_s2,
            lambert_nrev_max=int(nrev_max),
            lambert_hz=int(hz_leg),
            dvlev_max_km_s=float(dvlev_max_by_leg[leg_index]),
            delta_dvlev_km_s=float(delta_dvlev_by_leg[leg_index]),
            vinf_margin_pre_filter_km_s=float(vinf_margin_pre_filter_by_leg[leg_index]),
            lev_type=lev_type,
            parallelism=None,
            dt_positive_tol_s=1e-6,
            max_failure_frac=None,
            debug=False,
            chunk_size=20000,
        )

        leg_db = build_leg_db(encounter_db, cfg)
        leg_dbs_by_stage[int(leg_index)] = leg_db

        print("leg stage", leg_index, "rows:", int(leg_db.IL.size))

    # build flyby database from the already-constructed leg databases
    flyby_cfg = FlybyBuildConfig(
        numerical_eps=1e-12,
    )
    flyby_db = build_flyby_db(encounter_db, leg_dbs_by_stage, flyby_cfg)

    print()
    print("FlybyDB Summary")
    print(f"entries_created={int(flyby_db.IF.size)}")

    if flyby_db.IF.size == 0:
        print("No flyby entries generated.")
    else:
        print(
            f"{'IF':>6} {'stage':>6} {'IE':>6} "
            f"{'IL_in':>8} {'IL_out':>8} {'t_day':>12} {'dv_km_s':>12}"
        )
        print("-" * 72)
        max_rows_to_print = min(100, int(flyby_db.IF.size))
        for row in range(max_rows_to_print):
            encounter_entry = encounter_db.get_entry(int(flyby_db.IE[row]))
            print(
                f"{int(flyby_db.IF[row]):6d} "
                f"{int(flyby_db.stage_id[row]):6d} "
                f"{int(flyby_db.IE[row]):6d} "
                f"{int(flyby_db.IL_in[row]):8d} "
                f"{int(flyby_db.IL_out[row]):8d} "
                f"{et_seconds_to_j2000_days(float(encounter_entry.t_et)):12.4f} "
                f"{float(flyby_db.dv_km_s[row]):12.6f}"
            )

        # Build stage-sliced flyby DBs required by incoming/outgoing leg filters.
        flyby_dbs_by_stage: Dict[int, FlybyDB] = {}
        for flyby_stage in np.unique(flyby_db.stage_id):
            stage_id = int(flyby_stage)
            stage_mask = np.asarray(flyby_db.stage_id, dtype=np.int64) == stage_id
            flyby_dbs_by_stage[stage_id] = subset_flyby_db(flyby_db, stage_mask)

        # Apply both filters on the same mutable DB objects.
        # Outgoing filter is called after incoming filter to use updated databases.
        root_stage = int(min(flyby_dbs_by_stage.keys()))
        total_size = (
            lambda: int(
                sum(int(v.IL.size) for v in leg_dbs_by_stage.values())
                + sum(int(v.IF.size) for v in flyby_dbs_by_stage.values())
                + len(encounter_db.entries)
            )
        )
        print()
        print(f"Applying IncomingLegFilter(stage={root_stage}) ...")
        size_before = total_size()
        incoming_leg_filter(root_stage, leg_dbs_by_stage, flyby_dbs_by_stage, encounter_db)
        size_after = total_size()
        leg_sizes_after_incoming = {k: int(v.IL.size) for k, v in leg_dbs_by_stage.items()}
        flyby_sizes_after_incoming = {k: int(v.IF.size) for k, v in flyby_dbs_by_stage.items()}
        print(
            f"incoming size_before={size_before}, size_after={size_after}, removed={size_before - size_after}, "
            f"after incoming: encounter_entries={len(encounter_db.entries)}, "
            f"leg_sizes={leg_sizes_after_incoming}, "
            f"flyby_sizes={flyby_sizes_after_incoming}"
        )

        print()
        print(f"Applying OutgoingLegFilter(stage={root_stage}) ...")
        size_before = total_size()
        outgoing_leg_filter(root_stage, leg_dbs_by_stage, flyby_dbs_by_stage, encounter_db)
        size_after = total_size()
        leg_sizes_after_outgoing = {k: int(v.IL.size) for k, v in leg_dbs_by_stage.items()}
        flyby_sizes_after_outgoing = {k: int(v.IF.size) for k, v in flyby_dbs_by_stage.items()}
        print(
            f"outgoing size_before={size_before}, size_after={size_after}, removed={size_before - size_after}, "
            f"after outgoing: encounter_entries={len(encounter_db.entries)}, "
            f"leg_sizes={leg_sizes_after_outgoing}, "
            f"flyby_sizes={flyby_sizes_after_outgoing}"
        )
    

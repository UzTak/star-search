"""Chapter 5 combo module (Sections 5.1-5.4) for Star-style trajectory stitching.

Implemented scope:
- 5.1 Triplet module: build one TripletDB per flyby stage from filtered FlybyDB rows.
- 5.2 Segment combination: combine adjacent segment databases by shared middle-leg IL.
- 5.3 Recursive combination: deterministic left-fold to full trajectory candidates.
- 5.4 Final-output tfilter/decimation.

Not implemented here:
- VILT/central-body switching/nonconsecutive legs

Index glossary
--------------
IE  : Encounter index — a row index into EncounterDB.entries (global, unique per node).
IL  : Leg index       — a row index into a LegDatabase (unique per leg stage).
IF  : Flyby index     — a row index into a FlybyDB (unique per flyby stage).
IT  : Triplet index   — a row index into a TripletDB.
IS  : Segment index   — a row index into a SegmentDB.
tof : Time of flight  — elapsed duration between two consecutive encounters [seconds].
dv  : Delta-v         — impulsive velocity change [km/s].

See NOTATION.md for the full glossary, including the tfilter/null-leg
vocabulary used throughout this module.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import json
import numpy as np
from pathlib import Path
import tempfile

from star.encounter_database import EncounterDB
from star.leg_database import LegDatabase
from star.flyby_database import FlybyDB, compute_parabolic_escape_dv

_COMBO_JOIN_PAIR_CHUNK: int = 200_000
_COMBO_FLYBY_SCAN_CHUNK_ROWS: int = 2_000_000
_COMBO_KEEP_ROWS_FILE: str = "combo_keep_rows.npy"
_COMBO_FLYBY_INPUT_CHUNK_ROWS: int = 1_000_000
_COMBO_CHILD_FLUSH_ROWS: int = 200_000
_COMBO_PREEMPTIVE_PRUNE_SCAN_ROWS: int = 500_000


@dataclass(frozen=True)
class ComboBuildConfig:
    """Configuration for Chapter 5.1-5.3 combo construction.

    Attributes:
        dv_total_max_km_s: Optional upper bound on segment/trajectory total DV [km/s].
        tof_total_bounds_s: Optional `(min_s, max_s)` on total time of flight [seconds].
        max_rows_per_segment_db: Optional per-step cap on kept rows.
        debug: Enable internal assertions.
        nL: Total number of legs in the full trajectory (equals n_encounters - 1).
            Used to determine when the SegmentDB spans the complete trajectory.
        dt_filter_preemptive_s:
            Optional stage-wise preemptive dt-filter bin size [seconds], either scalar
            or one value per encounter.  When set, a time-binned decimation is applied
            at each combo step to prevent combinatorial row growth.
    """

    dv_total_max_km_s: Optional[float] = None
    tof_total_bounds_s: Optional[Tuple[float, float]] = None
    max_rows_per_segment_db: Optional[int] = None
    debug: bool = False
    nL: int = 0
    dt_filter_preemptive_s: Optional[np.ndarray | float] = None


@dataclass(frozen=True)
class TripletDB:
    """Triplet database for one flyby stage `i_stage`.

    Each row is copied 1-to-1 from one FlybyDB row and represents a valid
    flyby event connecting an incoming leg to an outgoing leg at a shared body.

    Fields:
        IT: Triplet row indices [int array, shape (N,)].
        i_stage: Encounter stage index of the flyby body.
        IL_left: Leg index of the incoming (left) leg for each row [int array].
        IL_right: Leg index of the outgoing (right) leg for each row [int array].
        IF: Flyby index into the flyby database for each row [int array].
        dv_km_s: Flyby patch delta-v [km/s] for each row [float array].
        IE_mid: Encounter index of the flyby body node for each row [int array].
        by_left_leg_il: Map from IL_left value → row indices in this TripletDB.
        by_right_leg_il: Map from IL_right value → row indices in this TripletDB.
    """

    IT: np.ndarray
    i_stage: int
    IL_left: np.ndarray
    IL_right: np.ndarray
    IF: np.ndarray
    dv_km_s: np.ndarray
    IE_mid: np.ndarray

    by_left_leg_il: Dict[int, np.ndarray] = field(default_factory=dict)
    by_right_leg_il: Dict[int, np.ndarray] = field(default_factory=dict)


@dataclass(frozen=True)
class SegmentDB:
    """Trajectory-segment database spanning leg stages `leg_start_stage..leg_end_stage`.

    Columns:
        IS: segment-row ID [int]
        left_leg_IL: leg IL at `leg_start_stage`
        right_leg_IL: leg IL at `leg_end_stage`
        dv_total_km_s: cumulative DV over flybys + leg DSM leveraging in this segment [km/s]
        leg_ILs: shape (N, num_legs_in_segment)
        flyby_IFs: shape (N, num_flybys_in_segment)
    """

    IS: np.ndarray
    leg_start_stage: int
    leg_end_stage: int

    left_leg_IL: np.ndarray
    right_leg_IL: np.ndarray
    dv_total_km_s: np.ndarray

    leg_ILs: np.ndarray
    flyby_IFs: np.ndarray

    by_left_leg_il: Dict[int, np.ndarray] = field(default_factory=dict)
    by_right_leg_il: Dict[int, np.ndarray] = field(default_factory=dict)


@dataclass(frozen=True)
class OutputDB:
    """Final full-trajectory output format (Chapter 6 style core fields).

    Per trajectory row:
    - encounter_IEs/body_ids/times_et_s across all encounters
    - leg_ILs across all legs
    - flyby_IFs across all intermediate encounters
    - dv_total_km_s and tof_total_s
    - per_leg_tof_s [seconds]
    """

    I_traj: np.ndarray
    encounter_IEs: np.ndarray
    body_ids: np.ndarray
    times_et_s: np.ndarray

    leg_ILs: np.ndarray
    flyby_IFs: np.ndarray

    dv_total_km_s: np.ndarray
    tof_total_s: np.ndarray
    per_leg_tof_s: np.ndarray


@dataclass(frozen=True)
class _ComboLegStageData:
    """Minimal leg-stage columns required by combo stitching."""

    stage_id: int
    IL: np.ndarray
    ID: np.ndarray
    IA: np.ndarray
    dv_lev_km_s: np.ndarray


@dataclass(frozen=True)
class _FlybyStageIfLookup:
    """Disk-backed IF->dv lookup metadata for one flyby stage."""

    stage_id: int
    if_min: int
    if_max: int
    contiguous_if: bool
    if_values: np.ndarray
    dv_values: np.ndarray


@dataclass(frozen=True)
class _ComboStageRows:
    """Minimal per-candidate state persisted during streaming combo recursion."""

    parent_row: np.ndarray
    leg0_il: np.ndarray
    dep_ie: np.ndarray
    right_leg_il: np.ndarray
    dv_total_km_s: np.ndarray
    flyby_if: np.ndarray

    @property
    def size(self) -> int:
        return int(np.asarray(self.parent_row, dtype=np.int64).size)


class _ComboStageSpoolWriter:
    """Append-only on-disk spool writer for one combo stage."""

    def __init__(self, stage_dir: str | Path):
        self.stage_dir = Path(stage_dir)
        self.stage_dir.mkdir(parents=True, exist_ok=True)
        self.part_index = 0
        self.row_count = 0

    def append_rows(self, rows: _ComboStageRows) -> None:
        row_count = int(rows.size)
        if row_count <= 0:
            return

        parent = np.asarray(rows.parent_row, dtype=np.int64).reshape(-1)
        leg0_il = np.asarray(rows.leg0_il, dtype=np.int64).reshape(-1)
        dep_ie = np.asarray(rows.dep_ie, dtype=np.int64).reshape(-1)
        right_leg_il = np.asarray(rows.right_leg_il, dtype=np.int64).reshape(-1)
        dv_total = np.asarray(rows.dv_total_km_s, dtype=float).reshape(-1)
        flyby_if = np.asarray(rows.flyby_if, dtype=np.int64).reshape(-1)
        if not (
            int(leg0_il.size) == row_count
            and int(dep_ie.size) == row_count
            and int(right_leg_il.size) == row_count
            and int(dv_total.size) == row_count
            and int(flyby_if.size) == row_count
        ):
            raise ValueError("Combo stage spool append columns must share row count.")

        tag = f"{int(self.part_index):08d}"
        np.save(self.stage_dir / f"parent_{tag}.npy", parent, allow_pickle=False)
        np.save(self.stage_dir / f"leg0_{tag}.npy", leg0_il, allow_pickle=False)
        np.save(self.stage_dir / f"dep_ie_{tag}.npy", dep_ie, allow_pickle=False)
        np.save(self.stage_dir / f"right_il_{tag}.npy", right_leg_il, allow_pickle=False)
        np.save(self.stage_dir / f"dv_total_{tag}.npy", dv_total, allow_pickle=False)
        np.save(self.stage_dir / f"flyby_if_{tag}.npy", flyby_if, allow_pickle=False)

        self.part_index += 1
        self.row_count += row_count


def _empty_combo_stage_rows() -> _ComboStageRows:
    """Create empty in-memory combo stage row container."""

    empty_i = np.empty(0, dtype=np.int64)
    return _ComboStageRows(
        parent_row=empty_i,
        leg0_il=empty_i,
        dep_ie=empty_i,
        right_leg_il=empty_i,
        dv_total_km_s=np.empty(0, dtype=float),
        flyby_if=empty_i,
    )


def _load_combo_stage_rows(stage_dir: str | Path) -> _ComboStageRows:
    """Load one on-disk combo stage spool into memory."""

    stage_path = Path(stage_dir)
    parent_parts = sorted(stage_path.glob("parent_*.npy"))
    if not parent_parts:
        return _empty_combo_stage_rows()

    parent_chunks: list[np.ndarray] = []
    leg0_chunks: list[np.ndarray] = []
    dep_chunks: list[np.ndarray] = []
    right_chunks: list[np.ndarray] = []
    dv_chunks: list[np.ndarray] = []
    flyby_chunks: list[np.ndarray] = []

    for parent_path in parent_parts:
        tag = parent_path.stem.split("_", 1)[1]
        parent = np.asarray(np.load(parent_path, allow_pickle=False), dtype=np.int64).reshape(-1)
        leg0_il = np.asarray(np.load(stage_path / f"leg0_{tag}.npy", allow_pickle=False), dtype=np.int64).reshape(-1)
        dep_ie = np.asarray(np.load(stage_path / f"dep_ie_{tag}.npy", allow_pickle=False), dtype=np.int64).reshape(-1)
        right_il = np.asarray(np.load(stage_path / f"right_il_{tag}.npy", allow_pickle=False), dtype=np.int64).reshape(-1)
        dv_total = np.asarray(np.load(stage_path / f"dv_total_{tag}.npy", allow_pickle=False), dtype=float).reshape(-1)
        flyby_if = np.asarray(np.load(stage_path / f"flyby_if_{tag}.npy", allow_pickle=False), dtype=np.int64).reshape(-1)
        row_count = int(parent.size)
        if not (
            int(leg0_il.size) == row_count
            and int(dep_ie.size) == row_count
            and int(right_il.size) == row_count
            and int(dv_total.size) == row_count
            and int(flyby_if.size) == row_count
        ):
            raise ValueError(f"Inconsistent combo stage spool part sizes in {stage_path} tag={tag}.")

        parent_chunks.append(parent)
        leg0_chunks.append(leg0_il)
        dep_chunks.append(dep_ie)
        right_chunks.append(right_il)
        dv_chunks.append(dv_total)
        flyby_chunks.append(flyby_if)

    return _ComboStageRows(
        parent_row=np.concatenate(parent_chunks, axis=0).astype(np.int64, copy=False),
        leg0_il=np.concatenate(leg0_chunks, axis=0).astype(np.int64, copy=False),
        dep_ie=np.concatenate(dep_chunks, axis=0).astype(np.int64, copy=False),
        right_leg_il=np.concatenate(right_chunks, axis=0).astype(np.int64, copy=False),
        dv_total_km_s=np.concatenate(dv_chunks, axis=0).astype(float, copy=False),
        flyby_if=np.concatenate(flyby_chunks, axis=0).astype(np.int64, copy=False),
    )


def _rewrite_combo_stage_spool(
    stage_dir: str | Path,
    rows: _ComboStageRows,
    *,
    flush_rows: int = _COMBO_CHILD_FLUSH_ROWS,
) -> None:
    """Rewrite one stage spool with exactly the provided rows."""

    stage_path = Path(stage_dir)
    stage_path.mkdir(parents=True, exist_ok=True)
    for pattern in (
        "parent_*.npy",
        "leg0_*.npy",
        "dep_ie_*.npy",
        "right_il_*.npy",
        "dv_total_*.npy",
        "flyby_if_*.npy",
    ):
        for file_path in stage_path.glob(pattern):
            file_path.unlink(missing_ok=True)

    writer = _ComboStageSpoolWriter(stage_path)
    row_count = int(rows.size)
    if row_count <= 0:
        return

    chunk_rows = int(flush_rows)
    if chunk_rows <= 0:
        chunk_rows = int(row_count)
    for start in range(0, row_count, chunk_rows):
        end = min(start + chunk_rows, row_count)
        writer.append_rows(
            _ComboStageRows(
                parent_row=np.asarray(rows.parent_row, dtype=np.int64)[start:end],
                leg0_il=np.asarray(rows.leg0_il, dtype=np.int64)[start:end],
                dep_ie=np.asarray(rows.dep_ie, dtype=np.int64)[start:end],
                right_leg_il=np.asarray(rows.right_leg_il, dtype=np.int64)[start:end],
                dv_total_km_s=np.asarray(rows.dv_total_km_s, dtype=float)[start:end],
                flyby_if=np.asarray(rows.flyby_if, dtype=np.int64)[start:end],
            )
        )


def _preemptive_tfilter_stage_rows(
    *,
    rows: _ComboStageRows,
    arr_ie_by_il: Mapping[int, int],
    enc_time_lut: np.ndarray,
    dt_dep_s: float,
    dt_arr_s: float,
    scan_rows: int = _COMBO_PREEMPTIVE_PRUNE_SCAN_ROWS,
) -> Tuple[_ComboStageRows, Dict[str, int]]:
    """Stage-wise dt-filter prune keyed by (right_leg_il, dep_bin, arr_bin), keep min DV."""

    row_count = int(rows.size)
    dep_dt_s = float(dt_dep_s)
    arr_dt_s = float(dt_arr_s)
    if row_count <= 0:
        stats = {"num_in": 0, "num_valid": 0, "num_bins": 0, "num_out": 0}
        return rows, stats
    if (not np.isfinite(dep_dt_s)) or dep_dt_s <= 0.0 or (not np.isfinite(arr_dt_s)) or arr_dt_s <= 0.0:
        raise ValueError("dt_filter_preemptive_s values must be finite and > 0.")
    if int(np.asarray(enc_time_lut, dtype=float).size) <= 0:
        empty = _empty_combo_stage_rows()
        stats = {"num_in": row_count, "num_valid": 0, "num_bins": 0, "num_out": 0}
        return empty, stats

    parent_all = np.asarray(rows.parent_row, dtype=np.int64).reshape(-1)
    leg0_all = np.asarray(rows.leg0_il, dtype=np.int64).reshape(-1)
    dep_all = np.asarray(rows.dep_ie, dtype=np.int64).reshape(-1)
    right_all = np.asarray(rows.right_leg_il, dtype=np.int64).reshape(-1)
    dv_all = np.asarray(rows.dv_total_km_s, dtype=float).reshape(-1)
    if_all = np.asarray(rows.flyby_if, dtype=np.int64).reshape(-1)

    chunk_rows = int(scan_rows)
    if chunk_rows <= 0:
        chunk_rows = row_count

    # key -> (dv, parent, leg0, dep_ie, right_il, flyby_if)
    best_by_key: Dict[Tuple[int, int, int], Tuple[float, int, int, int, int, int]] = {}
    num_valid = 0

    for start in range(0, row_count, chunk_rows):
        end = min(start + chunk_rows, row_count)
        parent_chunk = parent_all[start:end]
        leg0_chunk = leg0_all[start:end]
        dep_chunk = dep_all[start:end]
        right_chunk = right_all[start:end]
        dv_chunk = dv_all[start:end]
        if_chunk = if_all[start:end]
        arr_ie_chunk = np.fromiter(
            (arr_ie_by_il.get(int(leg_il), -1) for leg_il in right_chunk),
            dtype=np.int64,
            count=int(right_chunk.size),
        )

        dep_ok = (dep_chunk >= 0) & (dep_chunk < int(enc_time_lut.size))
        arr_ok = (arr_ie_chunk >= 0) & (arr_ie_chunk < int(enc_time_lut.size))
        valid = dep_ok & arr_ok & np.isfinite(dv_chunk)
        if not np.any(valid):
            continue

        valid_idx = np.flatnonzero(valid).astype(np.int64, copy=False)
        t_dep = np.asarray(enc_time_lut[dep_chunk[valid_idx]], dtype=float)
        t_arr = np.asarray(enc_time_lut[arr_ie_chunk[valid_idx]], dtype=float)
        finite_time = np.isfinite(t_dep) & np.isfinite(t_arr)
        if not np.any(finite_time):
            continue

        valid_idx = np.asarray(valid_idx[finite_time], dtype=np.int64)
        num_valid += int(valid_idx.size)

        right_valid = np.asarray(right_chunk[valid_idx], dtype=np.int64)
        dv_valid = np.asarray(dv_chunk[valid_idx], dtype=float)
        dep_valid = np.asarray(dep_chunk[valid_idx], dtype=np.int64)
        parent_valid = np.asarray(parent_chunk[valid_idx], dtype=np.int64)
        leg0_valid = np.asarray(leg0_chunk[valid_idx], dtype=np.int64)
        if_valid = np.asarray(if_chunk[valid_idx], dtype=np.int64)
        t_dep_valid = np.asarray(t_dep[finite_time], dtype=float)
        t_arr_valid = np.asarray(t_arr[finite_time], dtype=float)

        b_dep = np.floor(t_dep_valid / dep_dt_s).astype(np.int64)
        b_arr = np.floor(t_arr_valid / arr_dt_s).astype(np.int64)

        order = np.lexsort((dv_valid, b_arr, b_dep, right_valid))
        if int(order.size) <= 0:
            continue
        s_right = right_valid[order]
        s_b_dep = b_dep[order]
        s_b_arr = b_arr[order]
        group_change = np.empty(int(order.size), dtype=bool)
        group_change[0] = True
        group_change[1:] = (
            (s_right[1:] != s_right[:-1])
            | (s_b_dep[1:] != s_b_dep[:-1])
            | (s_b_arr[1:] != s_b_arr[:-1])
        )
        local_best = np.asarray(order[group_change], dtype=np.int64)

        for idx in local_best:
            i = int(idx)
            key = (int(right_valid[i]), int(b_dep[i]), int(b_arr[i]))
            candidate = (
                float(dv_valid[i]),
                int(parent_valid[i]),
                int(leg0_valid[i]),
                int(dep_valid[i]),
                int(right_valid[i]),
                int(if_valid[i]),
            )
            previous = best_by_key.get(key, None)
            if previous is None or candidate < previous:
                best_by_key[key] = candidate

    if not best_by_key:
        empty = _empty_combo_stage_rows()
        stats = {"num_in": row_count, "num_valid": int(num_valid), "num_bins": 0, "num_out": 0}
        return empty, stats

    selected = sorted(
        best_by_key.items(),
        key=lambda item: (
            int(item[0][0]),
            int(item[0][1]),
            int(item[0][2]),
            float(item[1][0]),
            int(item[1][1]),
            int(item[1][5]),
        ),
    )
    out_count = len(selected)

    parent_out = np.fromiter((int(item[1][1]) for item in selected), dtype=np.int64, count=out_count)
    leg0_out = np.fromiter((int(item[1][2]) for item in selected), dtype=np.int64, count=out_count)
    dep_out = np.fromiter((int(item[1][3]) for item in selected), dtype=np.int64, count=out_count)
    right_out = np.fromiter((int(item[1][4]) for item in selected), dtype=np.int64, count=out_count)
    dv_out = np.fromiter((float(item[1][0]) for item in selected), dtype=float, count=out_count)
    if_out = np.fromiter((int(item[1][5]) for item in selected), dtype=np.int64, count=out_count)

    pruned = _ComboStageRows(
        parent_row=parent_out,
        leg0_il=leg0_out,
        dep_ie=dep_out,
        right_leg_il=right_out,
        dv_total_km_s=dv_out,
        flyby_if=if_out,
    )
    stats = {"num_in": row_count, "num_valid": int(num_valid), "num_bins": int(len(best_by_key)), "num_out": int(out_count)}
    return pruned, stats


def _build_combo_stage_bounds_keep_mask(
    *,
    dep_ie: np.ndarray,
    arr_ie: np.ndarray,
    dv_total_km_s: np.ndarray,
    enc_time_lut: np.ndarray,
    cfg: ComboBuildConfig,
) -> np.ndarray:
    """Build keep-mask for global DV and TOF bounds on candidate child rows."""

    dep_ie_arr = np.asarray(dep_ie, dtype=np.int64).reshape(-1)
    arr_ie_arr = np.asarray(arr_ie, dtype=np.int64).reshape(-1)
    dv_arr = np.asarray(dv_total_km_s, dtype=float).reshape(-1)
    if int(dep_ie_arr.size) != int(arr_ie_arr.size) or int(dep_ie_arr.size) != int(dv_arr.size):
        raise ValueError("dep_ie, arr_ie, and dv_total_km_s must share row count.")

    keep = np.isfinite(dv_arr) & (dv_arr >= 0.0)
    if cfg.dv_total_max_km_s is not None:
        keep &= dv_arr <= float(cfg.dv_total_max_km_s)

    if cfg.tof_total_bounds_s is not None:
        tof_min_s, tof_max_s = cfg.tof_total_bounds_s
        tof_min_s = float(tof_min_s)
        tof_max_s = float(tof_max_s)
        tof_total = enc_time_lut[arr_ie_arr] - enc_time_lut[dep_ie_arr]
        if np.isfinite(tof_min_s):
            keep &= tof_total >= tof_min_s
        if np.isfinite(tof_max_s):
            keep &= tof_total <= tof_max_s

    return np.asarray(keep, dtype=bool)


def _reconstruct_segment_from_combo_spools(
    *,
    spool_root: str | Path,
    num_legs_total: int,
) -> SegmentDB:
    """Reconstruct final SegmentDB by backtracking parent links across stage spools."""

    num_legs = int(num_legs_total)
    if num_legs < 2:
        return _empty_segment_db(0, max(0, num_legs - 1), max(1, num_legs))

    final_stage = int(num_legs - 1)
    final_rows = _load_combo_stage_rows(Path(spool_root) / f"stage_{final_stage:02d}")
    if int(final_rows.size) == 0:
        return _empty_segment_db(0, int(num_legs - 1), num_legs)

    num_rows = int(final_rows.size)
    leg_ils = np.empty((num_rows, num_legs), dtype=np.int64)
    flyby_ifs = np.empty((num_rows, max(0, num_legs - 1)), dtype=np.int64)
    dv_total = np.asarray(final_rows.dv_total_km_s, dtype=float).reshape(-1)

    current_rows = np.arange(num_rows, dtype=np.int64)
    stage_cache: Dict[int, _ComboStageRows] = {int(final_stage): final_rows}

    for stage in range(final_stage, 0, -1):
        stage_int = int(stage)
        stage_rows = stage_cache.get(stage_int, None)
        if stage_rows is None:
            stage_rows = _load_combo_stage_rows(Path(spool_root) / f"stage_{stage_int:02d}")
            stage_cache[stage_int] = stage_rows

        flyby_ifs[:, stage_int - 1] = np.asarray(stage_rows.flyby_if, dtype=np.int64)[current_rows]
        leg_ils[:, stage_int] = np.asarray(stage_rows.right_leg_il, dtype=np.int64)[current_rows]
        if stage_int == 1:
            leg_ils[:, 0] = np.asarray(stage_rows.leg0_il, dtype=np.int64)[current_rows]
        current_rows = np.asarray(stage_rows.parent_row, dtype=np.int64)[current_rows]

    return _make_segment_db(
        leg_start_stage=0,
        leg_end_stage=int(num_legs - 1),
        leg_ils=leg_ils,
        flyby_ifs=flyby_ifs,
        dv_total_km_s=dv_total,
    )


def _build_index_map(values_int: np.ndarray) -> Dict[int, np.ndarray]:
    """Build value->row-indices map preserving source row order."""

    mapping: Dict[int, list[int]] = {}
    for row_index, value in enumerate(np.asarray(values_int, dtype=np.int64)):
        mapping.setdefault(int(value), []).append(int(row_index))
    return {key: np.asarray(rows, dtype=np.int64) for key, rows in mapping.items()}


def _compute_combo_keep_rows_for_stage(
    stage_dir: str | Path,
    *,
    frontier_il_in: Optional[np.ndarray] = None,
    scan_chunk_rows: int = _COMBO_FLYBY_SCAN_CHUNK_ROWS,
    cache_file_name: str = _COMBO_KEEP_ROWS_FILE,
) -> np.ndarray:
    """Build and cache stage row indices surviving active-mask + optional frontier IL_in."""

    stage_path = Path(stage_dir)
    chunk_rows = int(scan_chunk_rows)
    if chunk_rows <= 0:
        raise ValueError("scan_chunk_rows must be > 0.")

    active_mask = np.load(stage_path / "active_mask.npy", mmap_mode="r", allow_pickle=False).reshape(-1)
    total_rows = int(active_mask.size)
    if total_rows <= 0:
        keep_rows = np.empty(0, dtype=np.int64)
        np.save(stage_path / cache_file_name, keep_rows, allow_pickle=False)
        return keep_rows

    if frontier_il_in is None:
        keep_rows = np.flatnonzero(np.asarray(active_mask, dtype=bool)).astype(np.int64, copy=False)
        np.save(stage_path / cache_file_name, keep_rows, allow_pickle=False)
        return keep_rows

    il_in_full = np.load(stage_path / "IL_in.npy", mmap_mode="r", allow_pickle=False).reshape(-1)
    if int(il_in_full.size) != total_rows:
        raise ValueError("Flyby stage has inconsistent IL_in and active_mask row counts.")

    frontier = np.unique(np.asarray(frontier_il_in, dtype=np.int64).reshape(-1))
    if int(frontier.size) == 0:
        keep_rows = np.empty(0, dtype=np.int64)
        np.save(stage_path / cache_file_name, keep_rows, allow_pickle=False)
        return keep_rows

    keep_parts: list[np.ndarray] = []
    for start in range(0, total_rows, chunk_rows):
        end = min(start + chunk_rows, total_rows)
        active_chunk = np.asarray(active_mask[start:end], dtype=bool)
        if not np.any(active_chunk):
            continue
        il_chunk = np.asarray(il_in_full[start:end], dtype=np.int64)
        local = np.flatnonzero(active_chunk & np.isin(il_chunk, frontier))
        if int(local.size) > 0:
            keep_parts.append(np.asarray(local + int(start), dtype=np.int64))

    if keep_parts:
        keep_rows = np.concatenate(keep_parts, axis=0).astype(np.int64, copy=False)
    else:
        keep_rows = np.empty(0, dtype=np.int64)

    np.save(stage_path / cache_file_name, keep_rows, allow_pickle=False)
    return keep_rows


def _load_flyby_combo_columns_from_rows(
    stage_dir: str | Path,
    row_indices: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load only combo-required flyby columns for selected stage row indices."""

    stage_path = Path(stage_dir)
    rows = np.asarray(row_indices, dtype=np.int64).reshape(-1)
    if int(rows.size) == 0:
        empty_i = np.empty(0, dtype=np.int64)
        return empty_i, empty_i, empty_i, np.empty(0, dtype=float)

    if_full = np.load(stage_path / "IF.npy", mmap_mode="r", allow_pickle=False).reshape(-1)
    il_in_full = np.load(stage_path / "IL_in.npy", mmap_mode="r", allow_pickle=False).reshape(-1)
    il_out_full = np.load(stage_path / "IL_out.npy", mmap_mode="r", allow_pickle=False).reshape(-1)
    dv_full = np.load(stage_path / "dv_km_s.npy", mmap_mode="r", allow_pickle=False).reshape(-1)

    row_count = int(if_full.size)
    if (
        int(il_in_full.size) != row_count
        or int(il_out_full.size) != row_count
        or int(dv_full.size) != row_count
    ):
        raise ValueError("Flyby stage combo columns have inconsistent row counts.")
    if np.any(rows < 0) or np.any(rows >= row_count):
        raise ValueError("Requested flyby row index is out of bounds.")

    return (
        np.asarray(if_full[rows], dtype=np.int64).reshape(-1),
        np.asarray(il_in_full[rows], dtype=np.int64).reshape(-1),
        np.asarray(il_out_full[rows], dtype=np.int64).reshape(-1),
        np.asarray(dv_full[rows], dtype=float).reshape(-1),
    )


def _load_leg_stage_for_combo(stage_dir: str | Path, *, stage_id: int) -> _ComboLegStageData:
    """Load one leg stage with only combo-required active columns."""

    try:
        from star.stage_db_npy import load_leg_column
    except Exception:  # pragma: no cover
        from stage_db_npy import load_leg_column

    il = np.asarray(load_leg_column(stage_dir, "IL", active_only=True), dtype=np.int64).reshape(-1)
    dep = np.asarray(load_leg_column(stage_dir, "ID", active_only=True), dtype=np.int64).reshape(-1)
    arr = np.asarray(load_leg_column(stage_dir, "IA", active_only=True), dtype=np.int64).reshape(-1)
    dv = np.asarray(load_leg_column(stage_dir, "dv_lev_km_s", active_only=True), dtype=float).reshape(-1)
    row_count = int(il.size)
    if int(dep.size) != row_count or int(arr.size) != row_count or int(dv.size) != row_count:
        raise ValueError(f"Leg stage {int(stage_id)} has inconsistent active combo-column row counts.")

    return _ComboLegStageData(
        stage_id=int(stage_id),
        IL=il,
        ID=dep,
        IA=arr,
        dv_lev_km_s=dv,
    )


def _build_leg_row_by_il_map_from_array(il_values: np.ndarray, *, stage_id: int) -> Dict[int, int]:
    """Build IL->row map for one stage from IL array."""

    mapping: Dict[int, int] = {}
    for row_index, leg_il in enumerate(np.asarray(il_values, dtype=np.int64)):
        leg_il_int = int(leg_il)
        if leg_il_int in mapping:
            raise ValueError(f"Duplicate IL={leg_il_int} in leg stage {int(stage_id)}.")
        mapping[leg_il_int] = int(row_index)
    return mapping


def _empty_segment_db(
    leg_start_stage: int,
    leg_end_stage: int,
    num_legs_in_segment: int,
) -> SegmentDB:
    """Create an empty SegmentDB for a known span and leg-count width."""

    if num_legs_in_segment <= 0:
        raise ValueError("num_legs_in_segment must be positive.")

    empty_i = np.empty(0, dtype=np.int64)
    empty_f = np.empty(0, dtype=float)
    empty_leg = np.empty((0, int(num_legs_in_segment)), dtype=np.int64)
    empty_flyby = np.empty((0, max(0, int(num_legs_in_segment) - 1)), dtype=np.int64)

    return SegmentDB(
        IS=empty_i,
        leg_start_stage=int(leg_start_stage),
        leg_end_stage=int(leg_end_stage),
        left_leg_IL=empty_i,
        right_leg_IL=empty_i,
        dv_total_km_s=empty_f,
        leg_ILs=empty_leg,
        flyby_IFs=empty_flyby,
        by_left_leg_il={},
        by_right_leg_il={},
    )


def _make_segment_db(
    leg_start_stage: int,
    leg_end_stage: int,
    leg_ils: np.ndarray,
    flyby_ifs: np.ndarray,
    dv_total_km_s: np.ndarray,
) -> SegmentDB:
    """Build a SegmentDB from dense arrays."""

    leg_ils_array = np.asarray(leg_ils, dtype=np.int64)
    flyby_ifs_array = np.asarray(flyby_ifs, dtype=np.int64)
    dv_array = np.asarray(dv_total_km_s, dtype=float)

    if leg_ils_array.ndim != 2:
        raise ValueError("leg_ils must be 2D with shape (N, num_legs_in_segment).")
    if flyby_ifs_array.ndim != 2:
        raise ValueError("flyby_ifs must be 2D with shape (N, num_flybys_in_segment).")
    if dv_array.ndim != 1:
        raise ValueError("dv_total_km_s must be 1D.")

    num_rows = int(leg_ils_array.shape[0])
    if flyby_ifs_array.shape[0] != num_rows or dv_array.shape[0] != num_rows:
        raise ValueError("leg_ils, flyby_ifs, and dv_total_km_s must share row count.")

    num_legs_in_segment = int(leg_ils_array.shape[1])
    expected_num_legs = int(leg_end_stage) - int(leg_start_stage) + 1
    if num_legs_in_segment != expected_num_legs:
        raise ValueError(
            f"leg_ils width mismatch: got {num_legs_in_segment}, expected {expected_num_legs} "
            f"for span {leg_start_stage}:{leg_end_stage}."
        )
    if flyby_ifs_array.shape[1] != max(0, num_legs_in_segment - 1):
        raise ValueError(
            "flyby_ifs width must be num_legs_in_segment-1."
        )

    if num_rows == 0:
        return _empty_segment_db(leg_start_stage, leg_end_stage, num_legs_in_segment)

    left_leg = leg_ils_array[:, 0].astype(np.int64, copy=False)
    right_leg = leg_ils_array[:, -1].astype(np.int64, copy=False)

    return SegmentDB(
        IS=np.arange(num_rows, dtype=np.int64),
        leg_start_stage=int(leg_start_stage),
        leg_end_stage=int(leg_end_stage),
        left_leg_IL=left_leg,
        right_leg_IL=right_leg,
        dv_total_km_s=dv_array.astype(float, copy=False),
        leg_ILs=leg_ils_array,
        flyby_IFs=flyby_ifs_array,
        by_left_leg_il=_build_index_map(left_leg),
        by_right_leg_il=_build_index_map(right_leg),
    )


def _subset_segment_rows(segment_db: SegmentDB, row_indices: np.ndarray) -> SegmentDB:
    """Stable subset of SegmentDB rows by explicit index list."""

    rows = np.asarray(row_indices, dtype=np.int64)
    if rows.size == 0:
        return _empty_segment_db(
            segment_db.leg_start_stage,
            segment_db.leg_end_stage,
            segment_db.leg_ILs.shape[1],
        )

    return _make_segment_db(
        leg_start_stage=segment_db.leg_start_stage,
        leg_end_stage=segment_db.leg_end_stage,
        leg_ils=segment_db.leg_ILs[rows],
        flyby_ifs=segment_db.flyby_IFs[rows],
        dv_total_km_s=segment_db.dv_total_km_s[rows],
    )


def _build_leg_row_by_il_maps(leg_dbs: Mapping[int, LegDatabase]) -> Dict[int, Dict[int, int]]:
    """Build stage->(IL->row) maps for immutable leg ID lookup."""

    leg_row_by_il: Dict[int, Dict[int, int]] = {}
    for stage_id, leg_db in leg_dbs.items():
        mapping: Dict[int, int] = {}
        for row_index, leg_il in enumerate(np.asarray(leg_db.IL, dtype=np.int64)):
            leg_il_int = int(leg_il)
            if leg_il_int in mapping:
                raise ValueError(f"Duplicate IL={leg_il_int} in leg stage {stage_id}.")
            mapping[leg_il_int] = int(row_index)
        leg_row_by_il[int(stage_id)] = mapping
    return leg_row_by_il


def _build_flyby_dv_by_if_map(flyby_dbs: Mapping[int, FlybyDB]) -> Dict[int, float]:
    """Build map from flyby IF [int] to flyby patch DV [km/s]."""

    mapping: Dict[int, float] = {}
    for stage_id, flyby_db in flyby_dbs.items():
        if_array = np.asarray(flyby_db.IF, dtype=np.int64)
        dv_array = np.asarray(flyby_db.dv_km_s, dtype=float)
        if if_array.shape != dv_array.shape:
            raise ValueError(f"FlybyDB stage {stage_id} has inconsistent IF/dv shapes.")
        for flyby_if, dv_km_s in zip(if_array, dv_array):
            flyby_if_int = int(flyby_if)
            if flyby_if_int in mapping and not np.isclose(mapping[flyby_if_int], float(dv_km_s)):
                raise ValueError(f"Duplicate IF={flyby_if_int} with inconsistent dv values.")
            mapping[flyby_if_int] = float(dv_km_s)
    return mapping


def _build_flyby_stage_if_lookups(
    flyby_stage_dirs: Mapping[int, str | Path],
) -> Tuple[_FlybyStageIfLookup, ...]:
    """Build per-stage disk-backed IF lookup metadata."""

    lookups: list[_FlybyStageIfLookup] = []
    for stage_id, stage_dir in sorted((int(k), Path(v)) for k, v in flyby_stage_dirs.items()):
        if_array = np.load(stage_dir / "IF.npy", mmap_mode="r", allow_pickle=False).reshape(-1)
        dv_array = np.load(stage_dir / "dv_km_s.npy", mmap_mode="r", allow_pickle=False).reshape(-1)
        if int(if_array.size) != int(dv_array.size):
            raise ValueError(f"Flyby stage {stage_id} has inconsistent IF/dv rows on disk.")
        if int(if_array.size) == 0:
            continue

        if_min = int(if_array[0])
        if_max = int(if_array[-1])
        contiguous = (if_max - if_min + 1) == int(if_array.size)
        lookups.append(
            _FlybyStageIfLookup(
                stage_id=int(stage_id),
                if_min=if_min,
                if_max=if_max,
                contiguous_if=bool(contiguous),
                if_values=if_array,
                dv_values=dv_array,
            )
        )

    return tuple(sorted(lookups, key=lambda item: item.if_min))


def _lookup_flyby_dv_from_stage_lookups(
    flyby_if: int,
    *,
    stage_lookups: Sequence[_FlybyStageIfLookup],
) -> float:
    """Lookup one flyby patch DV by IF from disk-backed stage metadata."""

    flyby_if_int = int(flyby_if)
    for lookup in stage_lookups:
        if flyby_if_int < int(lookup.if_min) or flyby_if_int > int(lookup.if_max):
            continue
        if bool(lookup.contiguous_if):
            row_index = flyby_if_int - int(lookup.if_min)
            return float(lookup.dv_values[row_index])

        idx = int(np.searchsorted(lookup.if_values, flyby_if_int))
        if idx < int(lookup.if_values.size) and int(lookup.if_values[idx]) == flyby_if_int:
            return float(lookup.dv_values[idx])

    raise ValueError(f"Missing flyby IF={flyby_if_int} in provided flyby stage directories.")


def _build_empty_output_db(num_legs: int) -> OutputDB:
    """Create an empty OutputDB with fixed column counts."""

    if num_legs < 0:
        raise ValueError("num_legs must be >= 0.")

    num_encounters = int(num_legs) + 1
    num_flybys = max(0, int(num_legs) - 1)

    empty_i = np.empty(0, dtype=np.int64)
    empty_f = np.empty(0, dtype=float)

    return OutputDB(
        I_traj=empty_i,
        encounter_IEs=np.empty((0, num_encounters), dtype=np.int64),
        body_ids=np.empty((0, num_encounters), dtype=np.int64),
        times_et_s=np.empty((0, num_encounters), dtype=float),
        leg_ILs=np.empty((0, num_legs), dtype=np.int64),
        flyby_IFs=np.empty((0, num_flybys), dtype=np.int64),
        dv_total_km_s=empty_f,
        tof_total_s=empty_f,
        per_leg_tof_s=np.empty((0, num_legs), dtype=float),
    )


def _collapse_null_trajectory_record(
    *,
    encounter_ies: np.ndarray,
    times_et_s: np.ndarray,
    body_ids: np.ndarray,
    vinfD_km_s: np.ndarray,
    vinfA_km_s: np.ndarray,
    dv_lev_km_s: np.ndarray,
    eta_lev: np.ndarray,
    dv_patch_km_s: np.ndarray,
    leg_ils: np.ndarray,
    flyby_ifs: np.ndarray,
    null_flags: Sequence[bool],
) -> Dict[str, np.ndarray]:
    """Collapse fixed-width null-leg rows to a shorter exported trajectory record."""

    null_list = [bool(flag) for flag in list(null_flags)]
    if len(null_list) != int(np.asarray(leg_ils, dtype=np.int64).size):
        raise ValueError("null_flags length must match the output leg count.")

    encounter_ie_list = [int(value) for value in np.asarray(encounter_ies, dtype=np.int64)]
    time_list = [float(value) for value in np.asarray(times_et_s, dtype=float)]
    body_list = [int(value) for value in np.asarray(body_ids, dtype=np.int64)]
    vinfd_list = [np.asarray(value, dtype=float) for value in np.asarray(vinfD_km_s, dtype=float)]
    vinfa_list = [np.asarray(value, dtype=float) for value in np.asarray(vinfA_km_s, dtype=float)]
    dv_lev_list = [float(value) for value in np.asarray(dv_lev_km_s, dtype=float)]
    eta_list = [float(value) for value in np.asarray(eta_lev, dtype=float)]
    dv_patch_list = [float(value) for value in np.asarray(dv_patch_km_s, dtype=float)]
    leg_il_list = [int(value) for value in np.asarray(leg_ils, dtype=np.int64)]
    flyby_if_list = [int(value) for value in np.asarray(flyby_ifs, dtype=np.int64)]

    leg_index = 0
    while leg_index < len(null_list):
        if (not null_list[leg_index]) or (leg_index + 1 >= len(time_list)):
            leg_index += 1
            continue

        same_body = body_list[leg_index] == body_list[leg_index + 1]
        same_time = np.isclose(time_list[leg_index], time_list[leg_index + 1], rtol=0.0, atol=1.0e-6)
        if not (same_body and same_time):
            leg_index += 1
            continue

        drop_flyby_idx: Optional[int] = None
        if leg_index > 0 and np.allclose(vinfa_list[leg_index - 1], vinfd_list[leg_index], rtol=0.0, atol=1.0e-10):
            drop_flyby_idx = int(leg_index - 1)
        elif leg_index < len(vinfd_list) - 1 and np.allclose(
            vinfa_list[leg_index],
            vinfd_list[leg_index + 1],
            rtol=0.0,
            atol=1.0e-10,
        ):
            drop_flyby_idx = int(leg_index)
        elif leg_index > 0 and (leg_index - 1) < len(flyby_if_list):
            drop_flyby_idx = int(leg_index - 1)
        elif leg_index < len(flyby_if_list):
            drop_flyby_idx = int(leg_index)

        encounter_ie_list.pop(leg_index + 1)
        time_list.pop(leg_index + 1)
        body_list.pop(leg_index + 1)
        vinfd_list.pop(leg_index)
        vinfa_list.pop(leg_index)
        dv_lev_list.pop(leg_index)
        eta_list.pop(leg_index)
        leg_il_list.pop(leg_index)
        null_list.pop(leg_index)
        if drop_flyby_idx is not None and 0 <= int(drop_flyby_idx) < len(flyby_if_list):
            flyby_if_list.pop(int(drop_flyby_idx))
            dv_patch_list.pop(int(drop_flyby_idx))

    return {
        "encounter_ies": np.asarray(encounter_ie_list, dtype=np.int64),
        "times_et_s": np.asarray(time_list, dtype=float),
        "body_ids": np.asarray(body_list, dtype=np.int64),
        "vinfD_km_s": np.asarray(vinfd_list, dtype=float).reshape(-1, 3),
        "vinfA_km_s": np.asarray(vinfa_list, dtype=float).reshape(-1, 3),
        "dv_lev_km_s": np.asarray(dv_lev_list, dtype=float),
        "eta_lev": np.asarray(eta_list, dtype=float),
        "dv_patch_km_s": np.asarray(dv_patch_list, dtype=float),
        "leg_ils": np.asarray(leg_il_list, dtype=np.int64),
        "flyby_ifs": np.asarray(flyby_if_list, dtype=np.int64),
    }


def save_solution(
    output_db: OutputDB,
    enc_db: EncounterDB,
    leg_dbs: Mapping[int, LegDatabase],
    flyby_dbs: Optional[Mapping[int, FlybyDB]],
    output_path: str | Path,
    *,
    flyby_stage_dirs: Optional[Mapping[int, str | Path]] = None,
    null_flags: Optional[Sequence[bool]] = None,
) -> Path:
    """Save minimal solution rows for trajectory reconstruction.

    Saved fields per trajectory row:
    - `t_et_s`: encounter epochs [seconds, ET past J2000]
    - `body_ids`: encounter body IDs [NAIF int]
    - `vinfD_km_s`: per-leg departure excess vectors [km/s], shape (n_legs, 3)
    - `vinfA_km_s`: per-leg arrival excess vectors [km/s], shape (n_legs, 3)
    - `dv_lev_km_s`: per-leg DSM leveraging magnitude [km/s], shape (n_legs,)
    - `eta_lev`: per-leg DSM split fraction eta [0..1], shape (n_legs,)
    - `dv_escape_km_s`: departure boundary DV from first encounter [km/s]
    - `dv_insertion_km_s`: arrival boundary DV at last encounter [km/s]
    - `dv_patch_km_s`: per-flyby patch DV [km/s], shape (n_legs-1,)

    Traceability fields are also included:
    - `traj_id`, `leg_ils`, `flyby_ifs`, `dv_total_km_s`, `tof_total_days`.

    This function expects `leg_dbs` to already contain in-memory
    `vinfD_km_s`, `vinfA_km_s`, and `eta_lev` for all referenced IL rows.
    Flyby patch DV lookup can use either in-memory `flyby_dbs` or disk-backed
    `flyby_stage_dirs`. When `null_flags` is provided, exported rows collapse
    any fixed-width null legs back out of the saved JSON record.
    """

    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    num_rows = int(output_db.I_traj.size)
    if num_rows == 0:
        out_path.write_text("", encoding="utf-8")
        return out_path

    n_legs = int(output_db.leg_ILs.shape[1])
    resolved_null_flags = [False] * n_legs if null_flags is None else [bool(flag) for flag in list(null_flags)]
    if len(resolved_null_flags) != n_legs:
        raise ValueError("null_flags length must match output_db leg count.")
    leg_row_by_il = _build_leg_row_by_il_maps(leg_dbs)
    if n_legs > 1 and flyby_dbs is None and flyby_stage_dirs is None:
        raise ValueError("flyby_dbs or flyby_stage_dirs must be provided for multi-leg solution saving.")

    flyby_dv_by_if: Optional[Dict[int, float]] = None
    flyby_stage_lookups: Tuple[_FlybyStageIfLookup, ...] = ()
    if flyby_dbs is not None:
        flyby_dv_by_if = _build_flyby_dv_by_if_map(flyby_dbs)
    elif flyby_stage_dirs is not None:
        flyby_stage_lookups = _build_flyby_stage_if_lookups(flyby_stage_dirs)

    with out_path.open("w", encoding="utf-8") as file_obj:
        for row in range(num_rows):
            leg_ils = np.asarray(output_db.leg_ILs[row], dtype=np.int64)
            flyby_ifs = np.asarray(output_db.flyby_IFs[row], dtype=np.int64)

            vinfD_km_s = np.empty((n_legs, 3), dtype=float)
            vinfA_km_s = np.empty((n_legs, 3), dtype=float)
            dv_lev_km_s = np.empty(n_legs, dtype=float)
            eta_lev = np.empty(n_legs, dtype=float)

            for leg_stage in range(n_legs):
                leg_il = int(leg_ils[leg_stage])
                leg_row = leg_row_by_il[int(leg_stage)].get(leg_il, -1)
                if leg_row < 0:
                    raise ValueError(f"Missing IL={leg_il} in leg stage {leg_stage}.")
                leg_db = leg_dbs[int(leg_stage)]
                dv_lev_km_s[leg_stage] = float(leg_db.dv_lev_km_s[leg_row])

                have_vinf = (
                    np.asarray(leg_db.vinfD_km_s).ndim == 2
                    and np.asarray(leg_db.vinfA_km_s).ndim == 2
                    and int(np.asarray(leg_db.vinfD_km_s).shape[0]) > leg_row
                    and int(np.asarray(leg_db.vinfA_km_s).shape[0]) > leg_row
                )
                have_eta = (
                    np.asarray(leg_db.eta_lev).ndim == 1
                    and int(np.asarray(leg_db.eta_lev).shape[0]) > leg_row
                )

                if have_vinf and have_eta:
                    vinfD_km_s[leg_stage] = np.asarray(leg_db.vinfD_km_s[leg_row], dtype=float)
                    vinfA_km_s[leg_stage] = np.asarray(leg_db.vinfA_km_s[leg_row], dtype=float)
                    eta_lev[leg_stage] = float(leg_db.eta_lev[leg_row])
                else:
                    raise ValueError(
                        f"Leg stage {leg_stage} IL={leg_il} is missing in-memory vinf/eta columns."
                    )

            if flyby_dv_by_if is not None:
                dv_patch_km_s = np.asarray(
                    [float(flyby_dv_by_if[int(flyby_if)]) for flyby_if in flyby_ifs],
                    dtype=float,
                )
            else:
                dv_patch_km_s = np.asarray(
                    [
                        _lookup_flyby_dv_from_stage_lookups(
                            int(flyby_if),
                            stage_lookups=flyby_stage_lookups,
                        )
                        for flyby_if in flyby_ifs
                    ],
                    dtype=float,
                )

            collapsed = _collapse_null_trajectory_record(
                encounter_ies=np.asarray(output_db.encounter_IEs[row], dtype=np.int64),
                times_et_s=np.asarray(output_db.times_et_s[row], dtype=float),
                body_ids=np.asarray(output_db.body_ids[row], dtype=np.int64),
                vinfD_km_s=vinfD_km_s,
                vinfA_km_s=vinfA_km_s,
                dv_lev_km_s=dv_lev_km_s,
                eta_lev=eta_lev,
                dv_patch_km_s=dv_patch_km_s,
                leg_ils=leg_ils,
                flyby_ifs=flyby_ifs,
                null_flags=resolved_null_flags,
            )
            collapsed_times_et_s = np.asarray(collapsed["times_et_s"], dtype=float)

            # Null-leg collapse drops one encounter and one per-leg entry together,
            # so the epoch and per-leg arrays must stay consistent afterwards.
            leg_tof_s = np.diff(collapsed_times_et_s)
            collapsed_dv_lev_km_s = np.asarray(collapsed["dv_lev_km_s"], dtype=float)
            if int(leg_tof_s.size) != int(collapsed_dv_lev_km_s.size):
                raise ValueError("Collapsed trajectory epochs and per-leg arrays must share row count.")

            record = {
                "traj_id": int(output_db.I_traj[row]),
                "t_et_s": [float(value) for value in collapsed_times_et_s],
                "body_ids": [int(value) for value in np.asarray(collapsed["body_ids"], dtype=np.int64)],
                "vinfD_km_s": np.asarray(collapsed["vinfD_km_s"], dtype=float).tolist(),
                "vinfA_km_s": np.asarray(collapsed["vinfA_km_s"], dtype=float).tolist(),
                "dv_lev_km_s": [float(value) for value in np.asarray(collapsed["dv_lev_km_s"], dtype=float)],
                "eta_lev": [float(value) for value in np.asarray(collapsed["eta_lev"], dtype=float)],
                "dv_patch_km_s": np.asarray(collapsed["dv_patch_km_s"], dtype=float).tolist(),
                "leg_ils": [int(value) for value in np.asarray(collapsed["leg_ils"], dtype=np.int64)],
                "flyby_ifs": [int(value) for value in np.asarray(collapsed["flyby_ifs"], dtype=np.int64)],
                "dv_total_km_s": float(output_db.dv_total_km_s[row]),
                "tof_total_days": float(output_db.tof_total_s[row] / 86400.0),
            }
            # TODO: optionally evaluate/store resonant targeting-closure DV here
            # without feeding it back into the coarse-search cost.

            dv_escape_km_s = 0.0
            dv_insertion_km_s = 0.0
            if int(np.asarray(collapsed["leg_ils"], dtype=np.int64).size) > 0:
                dep_ie = int(np.asarray(collapsed["encounter_ies"], dtype=np.int64)[0])
                arr_ie = int(np.asarray(collapsed["encounter_ies"], dtype=np.int64)[-1])
                dep_entry = enc_db.get_entry(dep_ie)
                arr_entry = enc_db.get_entry(arr_ie)

                dep_vinf_km_s = float(np.linalg.norm(np.asarray(collapsed["vinfD_km_s"], dtype=float)[0]))
                arr_vinf_km_s = float(np.linalg.norm(np.asarray(collapsed["vinfA_km_s"], dtype=float)[-1]))

                dv_escape_km_s = float(
                    compute_parabolic_escape_dv(
                        dep_vinf_km_s,
                        float(dep_entry.mu_km3_s2),
                        float(dep_entry.rmin_km),
                    )
                )
                dv_insertion_km_s = float(
                    compute_parabolic_escape_dv(
                        arr_vinf_km_s,
                        float(arr_entry.mu_km3_s2),
                        float(arr_entry.rmin_km),
                    )
                )

            record["dv_escape_km_s"] = float(dv_escape_km_s)
            record["dv_insertion_km_s"] = float(dv_insertion_km_s)
            file_obj.write(json.dumps(record, separators=(",", ":")) + "\n")

    return out_path


def _normalize_tfilter_dt_s_by_encounter(
    dt_filter_s: float | Sequence[float] | np.ndarray,
    *,
    num_encounters: int,
) -> np.ndarray:
    """Normalize scalar/sequence dt-filter seconds to one positive value per encounter."""

    n_enc = int(num_encounters)
    if n_enc <= 0:
        raise ValueError("num_encounters must be positive.")

    raw = np.asarray(dt_filter_s, dtype=float).reshape(-1)
    if int(raw.size) == 0:
        raise ValueError("dt_filter_s must provide at least one value.")
    if int(raw.size) == 1:
        dt_by_enc = np.full(n_enc, float(raw[0]), dtype=float)
    elif int(raw.size) == n_enc:
        dt_by_enc = np.asarray(raw, dtype=float)
    else:
        raise ValueError(
            f"dt_filter_s must be scalar or length {n_enc} (encounters). Got length {int(raw.size)}."
        )

    if np.any(~np.isfinite(dt_by_enc)) or np.any(dt_by_enc <= 0.0):
        raise ValueError("dt_filter_s values must all be finite and > 0.")

    return dt_by_enc


def _tfilter_select_indices(
    times_et_s_v: np.ndarray,
    body_ids_v: np.ndarray,
    dv_v: np.ndarray,
    *,
    valid_orig_indices: np.ndarray,
    dt_filter_s_by_encounter: np.ndarray,
    num_in: int,
    num_skipped_invalid: int,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Select one min-DV row per bundle keyed by (body sequence + dep/arr time bins)."""

    times_v = np.asarray(times_et_s_v, dtype=float)
    bodies_v = np.asarray(body_ids_v, dtype=np.int64)
    dv_arr = np.asarray(dv_v, dtype=float).reshape(-1)
    valid_rows = np.asarray(valid_orig_indices, dtype=np.int64).reshape(-1)
    dt_by_enc = np.asarray(dt_filter_s_by_encounter, dtype=float).reshape(-1)

    n_valid = int(dv_arr.size)
    if n_valid == 0:
        return np.empty(0, dtype=np.int64), {
            "num_in": int(num_in),
            "num_valid": 0,
            "num_bins": 0,
            "num_out": 0,
            "num_skipped_invalid": int(num_skipped_invalid),
            "t_dep_ref_s": None,
            "t_arr_ref_s": None,
            "t_ref_s_by_encounter": [],
            "bin_occupancy_histogram": {},
            "bin_occupancy_by_key": {},
        }

    if times_v.ndim != 2 or bodies_v.ndim != 2:
        raise ValueError("times_et_s_v and body_ids_v must both be 2D.")
    if times_v.shape != bodies_v.shape:
        raise ValueError("times_et_s_v and body_ids_v must have identical shapes.")
    if times_v.shape[0] != n_valid or int(valid_rows.size) != n_valid:
        raise ValueError("Row counts must match across valid tfilter arrays.")
    if int(dt_by_enc.size) != int(times_v.shape[1]):
        raise ValueError("dt_filter_s_by_encounter length must match encounter column count.")

    t_dep = np.asarray(times_v[:, 0], dtype=float)
    t_arr = np.asarray(times_v[:, -1], dtype=float)
    t_dep_ref = float(np.min(t_dep))
    t_arr_ref = float(np.min(t_arr))
    b_dep = np.floor((t_dep - t_dep_ref) / float(dt_by_enc[0])).astype(np.int64)
    b_arr = np.floor((t_arr - t_arr_ref) / float(dt_by_enc[-1])).astype(np.int64)

    sort_keys: list[np.ndarray] = [dv_arr, b_arr, b_dep]
    for encounter_index in range(int(times_v.shape[1]) - 1, -1, -1):
        sort_keys.append(np.asarray(bodies_v[:, encounter_index], dtype=np.int64))
    order = np.lexsort(tuple(sort_keys))

    sorted_bodies = np.asarray(bodies_v[order], dtype=np.int64)
    sorted_b_dep = np.asarray(b_dep[order], dtype=np.int64)
    sorted_b_arr = np.asarray(b_arr[order], dtype=np.int64)

    bin_change = np.empty(n_valid, dtype=bool)
    bin_change[0] = True
    bin_change[1:] = (
        (sorted_b_dep[1:] != sorted_b_dep[:-1])
        | (sorted_b_arr[1:] != sorted_b_arr[:-1])
        | np.any(sorted_bodies[1:] != sorted_bodies[:-1], axis=1)
    )

    first_in_bin = np.asarray(order[bin_change], dtype=np.int64)
    keep_rows = np.asarray(valid_rows[first_in_bin], dtype=np.int64)

    bin_change_idx = np.where(bin_change)[0]
    bin_sizes = np.diff(np.append(bin_change_idx, n_valid)).astype(np.int64)
    unique_occ, occ_counts = np.unique(bin_sizes, return_counts=True)
    occupancy_histogram: Dict[int, int] = {int(o): int(c) for o, c in zip(unique_occ, occ_counts)}

    occupancy_by_key: Dict[str, int] = {}
    if int(times_v.shape[1]) <= 3:
        for idx, pos in enumerate(bin_change_idx):
            key_bodies = ",".join(str(int(v)) for v in sorted_bodies[pos])
            key_bins = f"{int(sorted_b_dep[pos])},{int(sorted_b_arr[pos])}"
            occupancy_by_key[f"{key_bodies}|{key_bins}"] = int(bin_sizes[idx])

    num_bins = int(bin_change.sum())
    stats = {
        "num_in": int(num_in),
        "num_valid": n_valid,
        "num_bins": num_bins,
        "num_out": int(keep_rows.size),
        "num_skipped_invalid": int(num_skipped_invalid),
        "t_dep_ref_s": t_dep_ref,
        "t_arr_ref_s": t_arr_ref,
        "t_ref_s_by_encounter": [t_dep_ref, t_arr_ref],
        "bin_occupancy_histogram": occupancy_histogram,
        "bin_occupancy_by_key": occupancy_by_key,
    }
    return keep_rows, stats


def tfilter_output_db(
    output_db: OutputDB,
    dt_filter_s: float | Sequence[float] | np.ndarray,
) -> Tuple[OutputDB, Dict[str, Any]]:
    """Apply final-output tfilter using departure/arrival bins and full body sequence."""

    i_traj = np.asarray(output_db.I_traj, dtype=np.int64).reshape(-1)
    times_et_s = np.asarray(output_db.times_et_s, dtype=float)
    body_ids = np.asarray(output_db.body_ids, dtype=np.int64)
    dv_total_km_s = np.asarray(output_db.dv_total_km_s, dtype=float).reshape(-1)

    if times_et_s.ndim != 2:
        raise ValueError("output_db.times_et_s must be 2D with shape (N, num_encounters).")
    if times_et_s.shape[1] <= 0:
        raise ValueError("output_db.times_et_s must have at least one encounter column.")
    if body_ids.shape != times_et_s.shape:
        raise ValueError("output_db.body_ids must match output_db.times_et_s shape.")

    num_in = int(i_traj.size)
    if times_et_s.shape[0] != num_in or dv_total_km_s.shape[0] != num_in:
        raise ValueError("OutputDB row counts must match across I_traj, times_et_s, and dv_total_km_s.")

    dt_by_enc = _normalize_tfilter_dt_s_by_encounter(
        dt_filter_s,
        num_encounters=int(times_et_s.shape[1]),
    )

    valid = np.all(np.isfinite(times_et_s), axis=1) & np.isfinite(dv_total_km_s)
    num_skipped_invalid = int((~valid).sum())
    valid_idx = np.where(valid)[0]

    keep_rows, stats = _tfilter_select_indices(
        times_et_s[valid_idx],
        body_ids[valid_idx],
        dv_total_km_s[valid_idx],
        valid_orig_indices=valid_idx,
        dt_filter_s_by_encounter=dt_by_enc,
        num_in=num_in,
        num_skipped_invalid=num_skipped_invalid,
    )

    return OutputDB(
        I_traj=np.asarray(output_db.I_traj, dtype=np.int64)[keep_rows],
        encounter_IEs=np.asarray(output_db.encounter_IEs, dtype=np.int64)[keep_rows],
        body_ids=np.asarray(output_db.body_ids, dtype=np.int64)[keep_rows],
        times_et_s=np.asarray(output_db.times_et_s, dtype=float)[keep_rows],
        leg_ILs=np.asarray(output_db.leg_ILs, dtype=np.int64)[keep_rows],
        flyby_IFs=np.asarray(output_db.flyby_IFs, dtype=np.int64)[keep_rows],
        dv_total_km_s=np.asarray(output_db.dv_total_km_s, dtype=float)[keep_rows],
        tof_total_s=np.asarray(output_db.tof_total_s, dtype=float)[keep_rows],
        per_leg_tof_s=np.asarray(output_db.per_leg_tof_s, dtype=float)[keep_rows],
    ), stats


def build_triplet_dbs(flyby_dbs: Mapping[int, FlybyDB]) -> Dict[int, TripletDB]:
    """Build triplet databases T[i] from flyby databases F[i].

    Deterministic order is preserved from each FlybyDB row order.
    """

    triplet_dbs: Dict[int, TripletDB] = {}
    for stage_id in sorted(int(stage) for stage in flyby_dbs.keys()):
        flyby_db = flyby_dbs[int(stage_id)]

        il_left = np.asarray(flyby_db.IL_in, dtype=np.int64)
        il_right = np.asarray(flyby_db.IL_out, dtype=np.int64)
        if_array = np.asarray(flyby_db.IF, dtype=np.int64)
        ie_mid = np.asarray(flyby_db.IE, dtype=np.int64)
        dv_array = np.asarray(flyby_db.dv_km_s, dtype=float)

        num_rows = int(if_array.size)
        if il_left.shape != (num_rows,) or il_right.shape != (num_rows,) or ie_mid.shape != (num_rows,):
            raise ValueError(f"FlybyDB stage {stage_id} has inconsistent 1D array shapes.")
        if dv_array.shape != (num_rows,):
            raise ValueError(f"FlybyDB stage {stage_id} has inconsistent dv_km_s shape.")

        triplet_dbs[int(stage_id)] = TripletDB(
            IT=np.arange(num_rows, dtype=np.int64),
            i_stage=int(stage_id),
            IL_left=il_left,
            IL_right=il_right,
            IF=if_array,
            dv_km_s=dv_array,
            IE_mid=ie_mid,
            by_left_leg_il=_build_index_map(il_left),
            by_right_leg_il=_build_index_map(il_right),
        )

    return triplet_dbs


def _build_two_leg_segment_from_flyby_columns(
    *,
    i_stage: int,
    il_left: np.ndarray,
    il_right: np.ndarray,
    if_array: np.ndarray,
    dv_patch: np.ndarray,
    leg_dv_by_il: Mapping[int, Mapping[int, float]],
) -> SegmentDB:
    """Build one `(i-1, i)` SegmentDB directly from flyby columns."""

    il_left_array = np.asarray(il_left, dtype=np.int64).reshape(-1)
    il_right_array = np.asarray(il_right, dtype=np.int64).reshape(-1)
    if_array_i64 = np.asarray(if_array, dtype=np.int64).reshape(-1)
    dv_patch_array = np.asarray(dv_patch, dtype=float).reshape(-1)

    num_rows = int(if_array_i64.size)
    if il_left_array.shape != (num_rows,) or il_right_array.shape != (num_rows,) or dv_patch_array.shape != (num_rows,):
        raise ValueError(f"Flyby stage {i_stage} has inconsistent row counts.")

    if num_rows == 0:
        return _make_segment_db(
            leg_start_stage=int(i_stage) - 1,
            leg_end_stage=int(i_stage),
            leg_ils=np.empty((0, 2), dtype=np.int64),
            flyby_ifs=np.empty((0, 1), dtype=np.int64),
            dv_total_km_s=np.empty(0, dtype=float),
        )

    left_stage = int(i_stage) - 1
    right_stage = int(i_stage)
    left_dv_map = leg_dv_by_il.get(left_stage, None)
    right_dv_map = leg_dv_by_il.get(right_stage, None)
    if left_dv_map is None or right_dv_map is None:
        raise ValueError(
            f"Missing leg dv map for triplet stage {i_stage}: "
            f"left_stage={left_stage in leg_dv_by_il}, right_stage={right_stage in leg_dv_by_il}."
        )

    left_dv = np.asarray([left_dv_map.get(int(leg_il), np.nan) for leg_il in il_left_array], dtype=float)
    right_dv = np.asarray([right_dv_map.get(int(leg_il), np.nan) for leg_il in il_right_array], dtype=float)
    if not np.all(np.isfinite(left_dv)) or not np.all(np.isfinite(right_dv)):
        raise ValueError(f"Triplet stage {i_stage} references IL not found in leg dv maps.")

    leg_ils = np.column_stack((il_left_array, il_right_array)).astype(np.int64, copy=False)
    flyby_ifs = if_array_i64.reshape(-1, 1).astype(np.int64, copy=False)
    dv_total = dv_patch_array + left_dv + right_dv

    return _make_segment_db(
        leg_start_stage=int(i_stage) - 1,
        leg_end_stage=int(i_stage),
        leg_ils=leg_ils,
        flyby_ifs=flyby_ifs,
        dv_total_km_s=dv_total,
    )


def init_segment_dbs(
    triplet_dbs: Mapping[int, TripletDB],
    *,
    leg_dv_by_il: Mapping[int, Mapping[int, float]],
) -> Dict[Tuple[int, int], SegmentDB]:
    """Initialize 2-leg SegmentDBs from TripletDBs.

    For flyby stage `i`, this creates segment span `(i-1, i)`:
    - leg chain: `[IL_left, IL_right]`
    - flyby chain: `[IF]`
    - dv_total: `dv_patch + dv_lev(left_leg) + dv_lev(right_leg)`
    """

    segment_dbs: Dict[Tuple[int, int], SegmentDB] = {}
    for i_stage in sorted(int(stage) for stage in triplet_dbs.keys()):
        triplet_db = triplet_dbs[int(i_stage)]
        segment_dbs[(int(i_stage) - 1, int(i_stage))] = _build_two_leg_segment_from_flyby_columns(
            i_stage=int(i_stage),
            il_left=np.asarray(triplet_db.IL_left, dtype=np.int64),
            il_right=np.asarray(triplet_db.IL_right, dtype=np.int64),
            if_array=np.asarray(triplet_db.IF, dtype=np.int64),
            dv_patch=np.asarray(triplet_db.dv_km_s, dtype=float),
            leg_dv_by_il=leg_dv_by_il,
        )

    return segment_dbs


def combine_segments(
    seg_left: SegmentDB,
    seg_right: SegmentDB,
    *,
    join_leg_dv_by_il: Mapping[int, float],
    join_leg_IL: Optional[int] = None,
) -> SegmentDB:
    """Combine two adjacent SegmentDBs by shared middle leg IL.

    Join condition:
        seg_left.right_leg_IL == seg_right.left_leg_IL
    and (if provided):
        == join_leg_IL

    Vectorized: builds all match-pair (L, R) index arrays via repeat/tile per IL group,
    then assembles combined arrays with a single bulk concatenate per column.
    """

    if int(seg_left.leg_end_stage) != int(seg_right.leg_start_stage):
        raise ValueError(
            "Segment spans are not adjacent for join: "
            f"left={seg_left.leg_start_stage}:{seg_left.leg_end_stage}, "
            f"right={seg_right.leg_start_stage}:{seg_right.leg_end_stage}."
        )

    num_left = int(seg_left.IS.size)
    num_right = int(seg_right.IS.size)
    num_legs_left = int(seg_left.leg_ILs.shape[1])
    num_legs_right = int(seg_right.leg_ILs.shape[1])
    combined_num_legs = num_legs_left + num_legs_right - 1

    if num_left == 0 or num_right == 0:
        return _empty_segment_db(seg_left.leg_start_stage, seg_right.leg_end_stage, combined_num_legs)

    left_by_il = seg_left.by_right_leg_il
    if not left_by_il:
        left_by_il = _build_index_map(seg_left.right_leg_IL)

    right_by_il = seg_right.by_left_leg_il
    if not right_by_il:
        right_by_il = _build_index_map(seg_right.left_leg_IL)

    _empty = np.empty(0, dtype=np.int64)
    all_L: list[np.ndarray] = []
    all_R: list[np.ndarray] = []

    for il, left_rows_il in left_by_il.items():
        if join_leg_IL is not None and il != int(join_leg_IL):
            continue
        right_rows_il = right_by_il.get(il, _empty)
        if right_rows_il.size == 0:
            continue
        n_l = left_rows_il.size
        n_r = right_rows_il.size
        all_L.append(np.repeat(left_rows_il, n_r))
        all_R.append(np.tile(right_rows_il, n_l))

    if not all_L:
        return _empty_segment_db(seg_left.leg_start_stage, seg_right.leg_end_stage, combined_num_legs)

    L = np.concatenate(all_L)
    R = np.concatenate(all_R)

    out_leg_ils = np.concatenate(
        [seg_left.leg_ILs[L], seg_right.leg_ILs[R, 1:]], axis=1
    )
    out_flyby_ifs = np.concatenate(
        [seg_left.flyby_IFs[L], seg_right.flyby_IFs[R]], axis=1
    )

    join_ils = seg_left.right_leg_IL[L]
    unique_join_ils = np.unique(join_ils)
    max_il = int(unique_join_ils.max())
    dv_lut = np.zeros(max_il + 1, dtype=np.float64)
    for il in unique_join_ils:
        il_int = int(il)
        dv_km_s = join_leg_dv_by_il.get(il_int, None)
        if dv_km_s is None:
            raise ValueError(
                f"Missing dv_lev value for join leg IL={il_int} at stage {seg_left.leg_end_stage}."
            )
        dv_lut[il_int] = float(dv_km_s)

    out_dv = seg_left.dv_total_km_s[L] + seg_right.dv_total_km_s[R] - dv_lut[join_ils]
    out_dv = np.where((out_dv < 0.0) & (out_dv > -1e-12), 0.0, out_dv)
    invalid = ~np.isfinite(out_dv) | (out_dv < 0.0)
    if invalid.any():
        raise ValueError("Combined segment DV became invalid while subtracting duplicated join-leg dv.")

    return _make_segment_db(
        leg_start_stage=seg_left.leg_start_stage,
        leg_end_stage=seg_right.leg_end_stage,
        leg_ils=out_leg_ils,
        flyby_ifs=out_flyby_ifs,
        dv_total_km_s=out_dv,
    )


def _segment_to_output_db_from_leg_stage_dirs(
    segment_db: SegmentDB,
    *,
    enc_db: EncounterDB,
    leg_stage_dirs: Mapping[int, str | Path],
    debug: bool,
) -> OutputDB:
    """Reconstruct OutputDB using on-demand combo-lite leg-stage loads."""

    num_rows = int(segment_db.IS.size)
    num_legs = int(segment_db.leg_ILs.shape[1])
    if num_rows == 0:
        return _build_empty_output_db(num_legs=num_legs)

    num_encounters = num_legs + 1

    max_ie = max(e.IE for e in enc_db.entries)
    enc_body_lut = np.empty(max_ie + 1, dtype=np.int64)
    enc_time_lut = np.empty(max_ie + 1, dtype=np.float64)
    for e in enc_db.entries:
        enc_body_lut[e.IE] = e.body
        enc_time_lut[e.IE] = e.t_et

    start_stage = int(segment_db.leg_start_stage)
    encounter_ies = np.empty((num_rows, num_encounters), dtype=np.int64)

    for leg_offset in range(num_legs):
        stage = start_stage + leg_offset
        stage_dir = leg_stage_dirs.get(int(stage), None)
        if stage_dir is None:
            raise ValueError(f"Missing leg stage directory for stage {stage}.")

        leg_stage = _load_leg_stage_for_combo(stage_dir, stage_id=int(stage))
        il_to_row = _build_leg_row_by_il_map_from_array(leg_stage.IL, stage_id=int(stage))

        query_ils = np.asarray(segment_db.leg_ILs[:, leg_offset], dtype=np.int64)
        leg_rows = np.asarray([il_to_row.get(int(il), -1) for il in query_ils], dtype=np.int64)
        missing = leg_rows < 0
        if missing.any():
            bad_il = int(query_ils[np.where(missing)[0][0]])
            raise ValueError(f"Missing IL={bad_il} in leg stage {stage}.")

        if leg_offset == 0:
            encounter_ies[:, 0] = np.asarray(leg_stage.ID, dtype=np.int64)[leg_rows]
        elif debug:
            dep_ies = np.asarray(leg_stage.ID, dtype=np.int64)[leg_rows]
            mismatch = dep_ies != encounter_ies[:, leg_offset]
            if mismatch.any():
                bad_row = int(np.where(mismatch)[0][0])
                raise ValueError(
                    f"Inconsistent chain at row={bad_row}, leg_stage={stage}: "
                    f"expected dep IE {encounter_ies[bad_row, leg_offset]}, got {dep_ies[bad_row]}."
                )
        encounter_ies[:, leg_offset + 1] = np.asarray(leg_stage.IA, dtype=np.int64)[leg_rows]

    body_ids = enc_body_lut[encounter_ies]
    times_et_s = enc_time_lut[encounter_ies]
    per_leg_tof_s = np.diff(times_et_s, axis=1)
    tof_total_s = times_et_s[:, -1] - times_et_s[:, 0]

    return OutputDB(
        I_traj=np.arange(num_rows, dtype=np.int64),
        encounter_IEs=encounter_ies,
        body_ids=body_ids,
        times_et_s=times_et_s,
        leg_ILs=np.asarray(segment_db.leg_ILs, dtype=np.int64),
        flyby_IFs=np.asarray(segment_db.flyby_IFs, dtype=np.int64),
        dv_total_km_s=np.asarray(segment_db.dv_total_km_s, dtype=float),
        tof_total_s=tof_total_s.astype(float, copy=False),
        per_leg_tof_s=per_leg_tof_s,
    )


def run_combo(
    enc_db: EncounterDB,
    cfg: ComboBuildConfig = ComboBuildConfig(),
    *,
    leg_stage_dirs: Mapping[int, str | Path],
    flyby_stage_dirs: Mapping[int, str | Path],
) -> OutputDB:
    """Run Chapter 5.1-5.3 combo flow and return full trajectory candidates.

    Combination schedule is deterministic left-fold:
        (0:1) + (1:2) -> (0:2),
        then (0:2) + (2:3) -> (0:3), ...
    until full span `(0:nL-1)`.

    Flyby inputs are loaded from disk stage directories, one stage at a time.
    """

    leg_stage_path_map = {int(stage_id): Path(stage_dir) for stage_id, stage_dir in dict(leg_stage_dirs).items()}
    leg_stage_ids = sorted(int(stage_id) for stage_id in leg_stage_path_map.keys())
    if not leg_stage_ids:
        empty_output = _build_empty_output_db(num_legs=0)
        return empty_output

    expected_leg_stage_ids = list(range(leg_stage_ids[0], leg_stage_ids[-1] + 1))
    if leg_stage_ids != expected_leg_stage_ids:
        raise ValueError(
            f"Leg stages must be contiguous. Got {leg_stage_ids}, expected contiguous {expected_leg_stage_ids}."
        )
    if leg_stage_ids[0] != 0:
        raise ValueError("Leg stages must start at 0 for full trajectory reconstruction.")

    num_legs_total = cfg.nL
    if num_legs_total < 2:
        raise ValueError("At least two legs are required for combo trajectory building.")
    for leg_stage in range(int(num_legs_total)):
        if int(leg_stage) not in leg_stage_path_map:
            raise ValueError(f"Missing leg stage directory for stage {leg_stage}.")

    if cfg.max_rows_per_segment_db is not None:
        raise ValueError(
            "Streaming combo path does not support max_rows_per_segment_db. "
            "Set max_rows_per_segment_db=None to preserve all candidates."
        )

    stage_cache: Dict[int, _ComboLegStageData] = {}
    dv_by_il_cache: Dict[int, Dict[int, float]] = {}
    dep_ie_by_il_cache: Dict[int, Dict[int, int]] = {}
    arr_ie_by_il_cache: Dict[int, Dict[int, int]] = {}

    def _ensure_leg_stage_loaded(stage_id: int) -> None:
        stage_int = int(stage_id)
        if stage_int in stage_cache:
            return
        stage_dir = leg_stage_path_map.get(stage_int, None)
        if stage_dir is None:
            raise ValueError(f"Missing leg stage directory for stage {stage_int}.")
        leg_stage = _load_leg_stage_for_combo(stage_dir, stage_id=stage_int)
        stage_cache[stage_int] = leg_stage
        dv_by_il_cache[stage_int] = {
            int(leg_il): float(dv)
            for leg_il, dv in zip(np.asarray(leg_stage.IL, dtype=np.int64), np.asarray(leg_stage.dv_lev_km_s, dtype=float))
        }
        dep_ie_by_il_cache[stage_int] = {
            int(leg_il): int(dep_ie)
            for leg_il, dep_ie in zip(np.asarray(leg_stage.IL, dtype=np.int64), np.asarray(leg_stage.ID, dtype=np.int64))
        }
        arr_ie_by_il_cache[stage_int] = {
            int(leg_il): int(arr_ie)
            for leg_il, arr_ie in zip(np.asarray(leg_stage.IL, dtype=np.int64), np.asarray(leg_stage.IA, dtype=np.int64))
        }

    def _evict_leg_stage_cache_except(keep_stages: set[int]) -> None:
        keep = {int(stage_id) for stage_id in keep_stages}
        for stage_id in list(stage_cache.keys()):
            if int(stage_id) in keep:
                continue
            stage_cache.pop(int(stage_id), None)
            dv_by_il_cache.pop(int(stage_id), None)
            dep_ie_by_il_cache.pop(int(stage_id), None)
            arr_ie_by_il_cache.pop(int(stage_id), None)

    # Keep stage 0 loaded for the full fold.
    _ensure_leg_stage_loaded(0)

    required_triplet_stages = list(range(1, num_legs_total))
    stage_dirs = {int(stage_id): Path(stage_dir) for stage_id, stage_dir in dict(flyby_stage_dirs).items()}
    for triplet_stage in required_triplet_stages:
        if int(triplet_stage) not in stage_dirs:
            return _build_empty_output_db(num_legs=num_legs_total)

    max_ie = max((int(entry.IE) for entry in enc_db.entries), default=-1)
    enc_time_lut = np.empty(max_ie + 1, dtype=float) if max_ie >= 0 else np.empty(0, dtype=float)
    if max_ie >= 0:
        enc_time_lut.fill(np.nan)
        for entry in enc_db.entries:
            enc_time_lut[int(entry.IE)] = float(entry.t_et)

    preemptive_dt_by_encounter_s: Optional[np.ndarray] = None
    if cfg.dt_filter_preemptive_s is not None:
        preemptive_dt_by_encounter_s = _normalize_tfilter_dt_s_by_encounter(
            cfg.dt_filter_preemptive_s,
            num_encounters=int(num_legs_total + 1),
        )

    def _maybe_preemptive_prune_stage(
        *,
        stage_id: int,
        stage_spool_dir: Path,
        stage_rows: _ComboStageRows,
    ) -> _ComboStageRows:
        if preemptive_dt_by_encounter_s is None or int(stage_rows.size) <= 0:
            return stage_rows

        arr_ie_map = arr_ie_by_il_cache.get(int(stage_id), None)
        if arr_ie_map is None:
            return stage_rows

        dep_index = 0
        arr_index = min(int(stage_id + 1), int(preemptive_dt_by_encounter_s.size - 1))
        dep_dt_s = float(preemptive_dt_by_encounter_s[dep_index])
        arr_dt_s = float(preemptive_dt_by_encounter_s[arr_index])

        pruned_rows, prune_stats = _preemptive_tfilter_stage_rows(
            rows=stage_rows,
            arr_ie_by_il=arr_ie_map,
            enc_time_lut=enc_time_lut,
            dt_dep_s=dep_dt_s,
            dt_arr_s=arr_dt_s,
        )
        if int(pruned_rows.size) < int(stage_rows.size):
            _rewrite_combo_stage_spool(stage_spool_dir, pruned_rows)
        print(
            f"  combo: preemptive dt-filter stage {int(stage_id)} "
            f"-> {int(prune_stats.get('num_out', 0))}/{int(prune_stats.get('num_in', 0))} candidates"
        )
        return pruned_rows

    with tempfile.TemporaryDirectory(prefix="combo_spool_") as spool_tmp:
        spool_root = Path(spool_tmp)

        # Stage 1: seed candidates from first flyby stage.
        _ensure_leg_stage_loaded(1)
        stage1_spool = spool_root / "stage_01"
        stage1_writer = _ComboStageSpoolWriter(stage1_spool)

        keep_rows_stage1 = _compute_combo_keep_rows_for_stage(
            stage_dirs[1],
            frontier_il_in=None,
        )

        stage1_parent_parts: list[np.ndarray] = []
        stage1_leg0_parts: list[np.ndarray] = []
        stage1_dep_ie_parts: list[np.ndarray] = []
        stage1_right_parts: list[np.ndarray] = []
        stage1_dv_parts: list[np.ndarray] = []
        stage1_if_parts: list[np.ndarray] = []
        stage1_buffer_rows = 0

        def _flush_stage1_buffer() -> None:
            nonlocal stage1_buffer_rows
            if stage1_buffer_rows <= 0:
                return
            stage1_writer.append_rows(
                _ComboStageRows(
                    parent_row=np.concatenate(stage1_parent_parts, axis=0),
                    leg0_il=np.concatenate(stage1_leg0_parts, axis=0),
                    dep_ie=np.concatenate(stage1_dep_ie_parts, axis=0),
                    right_leg_il=np.concatenate(stage1_right_parts, axis=0),
                    dv_total_km_s=np.concatenate(stage1_dv_parts, axis=0),
                    flyby_if=np.concatenate(stage1_if_parts, axis=0),
                )
            )
            stage1_parent_parts.clear()
            stage1_leg0_parts.clear()
            stage1_dep_ie_parts.clear()
            stage1_right_parts.clear()
            stage1_dv_parts.clear()
            stage1_if_parts.clear()
            stage1_buffer_rows = 0

        dep_ie_map_0 = dep_ie_by_il_cache[0]
        dv_map_0 = dv_by_il_cache[0]
        arr_ie_map_1 = arr_ie_by_il_cache[1]
        dv_map_1 = dv_by_il_cache[1]
        for row_start in range(0, int(keep_rows_stage1.size), int(_COMBO_FLYBY_INPUT_CHUNK_ROWS)):
            row_end = min(row_start + int(_COMBO_FLYBY_INPUT_CHUNK_ROWS), int(keep_rows_stage1.size))
            rows_chunk = np.asarray(keep_rows_stage1[row_start:row_end], dtype=np.int64)
            if rows_chunk.size == 0:
                continue

            if_chunk, il_in_chunk, il_out_chunk, dv_patch_chunk = _load_flyby_combo_columns_from_rows(
                stage_dirs[1],
                rows_chunk,
            )
            dep_ie_chunk = np.fromiter(
                (dep_ie_map_0.get(int(leg_il), -1) for leg_il in il_in_chunk),
                dtype=np.int64,
                count=int(il_in_chunk.size),
            )
            arr_ie_chunk = np.fromiter(
                (arr_ie_map_1.get(int(leg_il), -1) for leg_il in il_out_chunk),
                dtype=np.int64,
                count=int(il_out_chunk.size),
            )
            dv_in_chunk = np.fromiter(
                (dv_map_0.get(int(leg_il), np.nan) for leg_il in il_in_chunk),
                dtype=float,
                count=int(il_in_chunk.size),
            )
            dv_out_chunk = np.fromiter(
                (dv_map_1.get(int(leg_il), np.nan) for leg_il in il_out_chunk),
                dtype=float,
                count=int(il_out_chunk.size),
            )

            valid = (
                (dep_ie_chunk >= 0)
                & (arr_ie_chunk >= 0)
                & np.isfinite(dv_in_chunk)
                & np.isfinite(dv_out_chunk)
            )
            if not np.any(valid):
                continue

            dep_ie_valid = np.asarray(dep_ie_chunk[valid], dtype=np.int64)
            arr_ie_valid = np.asarray(arr_ie_chunk[valid], dtype=np.int64)
            il_in_valid = np.asarray(il_in_chunk[valid], dtype=np.int64)
            il_out_valid = np.asarray(il_out_chunk[valid], dtype=np.int64)
            if_valid = np.asarray(if_chunk[valid], dtype=np.int64)
            dv_total_valid = (
                np.asarray(dv_patch_chunk[valid], dtype=float)
                + np.asarray(dv_in_chunk[valid], dtype=float)
                + np.asarray(dv_out_chunk[valid], dtype=float)
            )
            dv_total_valid = np.where(
                (dv_total_valid < 0.0) & (dv_total_valid > -1e-12),
                0.0,
                dv_total_valid,
            )
            keep = _build_combo_stage_bounds_keep_mask(
                dep_ie=dep_ie_valid,
                arr_ie=arr_ie_valid,
                dv_total_km_s=dv_total_valid,
                enc_time_lut=enc_time_lut,
                cfg=cfg,
            )
            if not np.any(keep):
                continue

            kept_rows = int(np.count_nonzero(keep))
            stage1_parent_parts.append(np.full(kept_rows, -1, dtype=np.int64))
            stage1_leg0_parts.append(np.asarray(il_in_valid[keep], dtype=np.int64))
            stage1_dep_ie_parts.append(np.asarray(dep_ie_valid[keep], dtype=np.int64))
            stage1_right_parts.append(np.asarray(il_out_valid[keep], dtype=np.int64))
            stage1_dv_parts.append(np.asarray(dv_total_valid[keep], dtype=float))
            stage1_if_parts.append(np.asarray(if_valid[keep], dtype=np.int64))
            stage1_buffer_rows += kept_rows
            if stage1_buffer_rows >= int(_COMBO_CHILD_FLUSH_ROWS):
                _flush_stage1_buffer()

        _flush_stage1_buffer()
        current_rows = _load_combo_stage_rows(stage1_spool)
        print(f"  combo: attached flyby stage 1 -> {int(current_rows.size)} candidates")
        current_rows = _maybe_preemptive_prune_stage(
            stage_id=1,
            stage_spool_dir=stage1_spool,
            stage_rows=current_rows,
        )
        if int(current_rows.size) == 0:
            return _build_empty_output_db(num_legs=num_legs_total)

        for right_end_stage in range(2, num_legs_total):
            _ensure_leg_stage_loaded(int(right_end_stage))

            frontier_il_in = np.unique(np.asarray(current_rows.right_leg_il, dtype=np.int64))
            keep_rows_stage = _compute_combo_keep_rows_for_stage(
                stage_dirs[int(right_end_stage)],
                frontier_il_in=frontier_il_in,
            )

            next_spool = spool_root / f"stage_{int(right_end_stage):02d}"
            stage_writer = _ComboStageSpoolWriter(next_spool)
            current_by_il = _build_index_map(np.asarray(current_rows.right_leg_il, dtype=np.int64))

            arr_ie_out_map = arr_ie_by_il_cache[int(right_end_stage)]
            dv_out_map = dv_by_il_cache[int(right_end_stage)]

            child_parent_parts: list[np.ndarray] = []
            child_leg0_parts: list[np.ndarray] = []
            child_dep_ie_parts: list[np.ndarray] = []
            child_right_parts: list[np.ndarray] = []
            child_dv_parts: list[np.ndarray] = []
            child_if_parts: list[np.ndarray] = []
            child_buffer_rows = 0

            def _flush_child_buffer() -> None:
                nonlocal child_buffer_rows
                if child_buffer_rows <= 0:
                    return
                stage_writer.append_rows(
                    _ComboStageRows(
                        parent_row=np.concatenate(child_parent_parts, axis=0),
                        leg0_il=np.concatenate(child_leg0_parts, axis=0),
                        dep_ie=np.concatenate(child_dep_ie_parts, axis=0),
                        right_leg_il=np.concatenate(child_right_parts, axis=0),
                        dv_total_km_s=np.concatenate(child_dv_parts, axis=0),
                        flyby_if=np.concatenate(child_if_parts, axis=0),
                    )
                )
                child_parent_parts.clear()
                child_leg0_parts.clear()
                child_dep_ie_parts.clear()
                child_right_parts.clear()
                child_dv_parts.clear()
                child_if_parts.clear()
                child_buffer_rows = 0

            for row_start in range(0, int(keep_rows_stage.size), int(_COMBO_FLYBY_INPUT_CHUNK_ROWS)):
                row_end = min(row_start + int(_COMBO_FLYBY_INPUT_CHUNK_ROWS), int(keep_rows_stage.size))
                rows_chunk = np.asarray(keep_rows_stage[row_start:row_end], dtype=np.int64)
                if rows_chunk.size == 0:
                    continue

                if_chunk, il_in_chunk, il_out_chunk, dv_patch_chunk = _load_flyby_combo_columns_from_rows(
                    stage_dirs[int(right_end_stage)],
                    rows_chunk,
                )
                flyby_rows_by_il_in = _build_index_map(np.asarray(il_in_chunk, dtype=np.int64))

                for join_il, fly_rows in flyby_rows_by_il_in.items():
                    left_rows = current_by_il.get(int(join_il), None)
                    if left_rows is None or int(left_rows.size) == 0:
                        continue
                    n_fly = int(fly_rows.size)
                    left_rows_per_chunk = max(1, int(_COMBO_JOIN_PAIR_CHUNK) // max(n_fly, 1))

                    for left_start in range(0, int(left_rows.size), left_rows_per_chunk):
                        left_chunk = np.asarray(
                            left_rows[left_start:left_start + left_rows_per_chunk],
                            dtype=np.int64,
                        )
                        if left_chunk.size == 0:
                            continue

                        L = np.repeat(left_chunk, n_fly)
                        F = np.tile(fly_rows, int(left_chunk.size))

                        right_il = np.asarray(il_out_chunk[F], dtype=np.int64)
                        arr_ie = np.fromiter(
                            (arr_ie_out_map.get(int(leg_il), -1) for leg_il in right_il),
                            dtype=np.int64,
                            count=int(right_il.size),
                        )
                        dv_leg_out = np.fromiter(
                            (dv_out_map.get(int(leg_il), np.nan) for leg_il in right_il),
                            dtype=float,
                            count=int(right_il.size),
                        )

                        valid = (arr_ie >= 0) & np.isfinite(dv_leg_out)
                        if not np.any(valid):
                            continue

                        parent_valid = np.asarray(L[valid], dtype=np.int64)
                        dep_ie_valid = np.asarray(current_rows.dep_ie[parent_valid], dtype=np.int64)
                        leg0_valid = np.asarray(current_rows.leg0_il[parent_valid], dtype=np.int64)
                        right_il_valid = np.asarray(right_il[valid], dtype=np.int64)
                        flyby_if_valid = np.asarray(if_chunk[F][valid], dtype=np.int64)
                        arr_ie_valid = np.asarray(arr_ie[valid], dtype=np.int64)
                        dv_total_valid = (
                            np.asarray(current_rows.dv_total_km_s[parent_valid], dtype=float)
                            + np.asarray(dv_patch_chunk[F][valid], dtype=float)
                            + np.asarray(dv_leg_out[valid], dtype=float)
                        )
                        dv_total_valid = np.where(
                            (dv_total_valid < 0.0) & (dv_total_valid > -1e-12),
                            0.0,
                            dv_total_valid,
                        )

                        keep = _build_combo_stage_bounds_keep_mask(
                            dep_ie=dep_ie_valid,
                            arr_ie=arr_ie_valid,
                            dv_total_km_s=dv_total_valid,
                            enc_time_lut=enc_time_lut,
                            cfg=cfg,
                        )
                        if not np.any(keep):
                            continue

                        kept_rows = int(np.count_nonzero(keep))
                        child_parent_parts.append(np.asarray(parent_valid[keep], dtype=np.int64))
                        child_leg0_parts.append(np.asarray(leg0_valid[keep], dtype=np.int64))
                        child_dep_ie_parts.append(np.asarray(dep_ie_valid[keep], dtype=np.int64))
                        child_right_parts.append(np.asarray(right_il_valid[keep], dtype=np.int64))
                        child_dv_parts.append(np.asarray(dv_total_valid[keep], dtype=float))
                        child_if_parts.append(np.asarray(flyby_if_valid[keep], dtype=np.int64))
                        child_buffer_rows += kept_rows
                        if child_buffer_rows >= int(_COMBO_CHILD_FLUSH_ROWS):
                            _flush_child_buffer()

                _flush_child_buffer()

            _flush_child_buffer()
            current_rows = _load_combo_stage_rows(next_spool)
            print(f"  combo: attached flyby stage {int(right_end_stage)} -> {int(current_rows.size)} candidates")
            current_rows = _maybe_preemptive_prune_stage(
                stage_id=int(right_end_stage),
                stage_spool_dir=next_spool,
                stage_rows=current_rows,
            )
            _evict_leg_stage_cache_except({0, int(right_end_stage)})
            if int(current_rows.size) == 0:
                break

        if int(current_rows.size) == 0:
            return _build_empty_output_db(num_legs=num_legs_total)

        final_segment = _reconstruct_segment_from_combo_spools(
            spool_root=spool_root,
            num_legs_total=int(num_legs_total),
        )

    output_db = _segment_to_output_db_from_leg_stage_dirs(
        final_segment,
        enc_db=enc_db,
        leg_stage_dirs=leg_stage_path_map,
        debug=bool(cfg.debug),
    )

    return output_db

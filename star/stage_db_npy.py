"""Simple stage-wise NPY storage for disk-backed Phase-1.

Each stage lives in its own directory holding one `<column>.npy` file per
column plus a `meta.json` recording `kind`, `stage_id`, and `row_count`.
Columns are addressed by name, never by position, so the set of columns can
change without invalidating previously written stage directories.

See NOTATION.md for the meaning of the stored columns.
"""

from __future__ import annotations

import json
import re
import shutil
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
from numpy.lib.format import open_memmap

from star.leg_database import LegDatabase


_LEG_COLUMNS = (
    "IL",
    "ID",
    "IA",
    "vinfD_km_s",
    "vinfA_km_s",
    "n_rev",
    "dv_lev_km_s",
    "eta_lev",
)

_FLYBY_COLUMNS = (
    "IF",
    "IE",
    "IL_in",
    "IL_out",
    "dv_km_s",
)

_FLYBY_COLUMN_DTYPES = {
    "IF": np.dtype(np.int64),
    "IE": np.dtype(np.int64),
    "IL_in": np.dtype(np.int64),
    "IL_out": np.dtype(np.int64),
    "dv_km_s": np.dtype(float),
}

_LEG_COLUMN_DTYPES = {
    "IL": np.dtype(np.int64),
    "ID": np.dtype(np.int64),
    "IA": np.dtype(np.int64),
    "vinfD_km_s": np.dtype(float),
    "vinfA_km_s": np.dtype(float),
    "n_rev": np.dtype(np.int64),
    "dv_lev_km_s": np.dtype(float),
    "eta_lev": np.dtype(float),
}


def _meta_path(stage_path: Path) -> Path:
    return stage_path / "meta.json"


def _write_meta_payload(stage_path: Path, payload: Dict[str, Any]) -> None:
    _meta_path(stage_path).write_text(json.dumps(payload, separators=(",", ":")), encoding="utf-8")


def _write_meta(
    stage_path: Path,
    *,
    kind: str,
    stage_id: int,
    row_count: int,
    extra_fields: Optional[Dict[str, Any]] = None,
) -> None:
    payload = {
        "kind": str(kind),
        "stage_id": int(stage_id),
        "row_count": int(row_count),
    }
    if extra_fields:
        payload.update(dict(extra_fields))
    _write_meta_payload(stage_path, payload)


def _read_meta(stage_path: Path) -> Dict[str, Any]:
    payload = json.loads(_meta_path(stage_path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Invalid stage meta at {stage_path}.")
    return payload


def _infer_stage_id(stage_path: Path) -> int:
    match = re.search(r"(\d+)$", stage_path.name)
    if match is None:
        raise ValueError(f"Cannot infer stage_id from stage directory name: {stage_path.name}")
    return int(match.group(1))


def _null_copied_il_path(stage_path: Path) -> Path:
    return stage_path / "null_copied_il.npy"


def _null_source_il_path(stage_path: Path) -> Path:
    return stage_path / "null_source_il.npy"


def save_leg_stage(stage_dir: str | Path, leg_db: LegDatabase) -> Path:
    stage_path = Path(stage_dir)
    stage_path.mkdir(parents=True, exist_ok=True)

    il = np.asarray(leg_db.IL, dtype=np.int64).reshape(-1)
    row_count = int(il.size)
    columns = {
        "IL": il,
        "ID": np.asarray(leg_db.ID, dtype=np.int64).reshape(-1),
        "IA": np.asarray(leg_db.IA, dtype=np.int64).reshape(-1),
        "vinfD_km_s": np.asarray(leg_db.vinfD_km_s, dtype=float).reshape(-1, 3),
        "vinfA_km_s": np.asarray(leg_db.vinfA_km_s, dtype=float).reshape(-1, 3),
        "n_rev": np.asarray(leg_db.n_rev, dtype=np.int64).reshape(-1),
        "dv_lev_km_s": np.asarray(leg_db.dv_lev_km_s, dtype=float).reshape(-1),
        "eta_lev": np.asarray(leg_db.eta_lev, dtype=float).reshape(-1),
    }
    for name, arr in columns.items():
        if int(arr.shape[0]) != row_count:
            raise ValueError(f"Leg column '{name}' has mismatched row count.")
        np.save(stage_path / f"{name}.npy", arr, allow_pickle=False)

    np.save(stage_path / "active_mask.npy", np.ones(row_count, dtype=bool), allow_pickle=False)
    _write_meta(stage_path, kind="leg", stage_id=int(leg_db.stage_id), row_count=row_count)
    return stage_path


def save_leg_null_binding_sidecar(
    stage_dir: str | Path,
    *,
    source_stage_id: Optional[int],
    copied_il: Optional[np.ndarray] = None,
    source_il: Optional[np.ndarray] = None,
) -> None:
    stage_path = Path(stage_dir)
    meta = _read_meta(stage_path)
    if str(meta.get("kind", "")).lower() != "leg":
        raise ValueError(f"Expected leg stage at {stage_path}.")

    copied_array = np.asarray(copied_il if copied_il is not None else np.empty(0, dtype=np.int64), dtype=np.int64).reshape(-1)
    source_array = np.asarray(source_il if source_il is not None else np.empty(0, dtype=np.int64), dtype=np.int64).reshape(-1)
    if int(copied_array.size) != int(source_array.size):
        raise ValueError("null binding copied/source IL arrays must have the same length.")

    copied_path = _null_copied_il_path(stage_path)
    source_path = _null_source_il_path(stage_path)

    if int(copied_array.size) == 0 or source_stage_id is None:
        copied_path.unlink(missing_ok=True)
        source_path.unlink(missing_ok=True)
        meta.pop("null_source_stage_id", None)
        _write_meta_payload(stage_path, meta)
        return

    np.save(copied_path, copied_array, allow_pickle=False)
    np.save(source_path, source_array, allow_pickle=False)
    meta["null_source_stage_id"] = int(source_stage_id)
    _write_meta_payload(stage_path, meta)


def load_leg_null_binding_sidecar(
    stage_dir: str | Path,
    *,
    active_only: bool = False,
) -> Tuple[Optional[int], np.ndarray, np.ndarray]:
    stage_path = Path(stage_dir)
    meta = _read_meta(stage_path)
    if str(meta.get("kind", "")).lower() != "leg":
        raise ValueError(f"Expected leg stage at {stage_path}.")

    raw_source_stage = meta.get("null_source_stage_id", None)
    source_stage_id = None if raw_source_stage is None else int(raw_source_stage)
    copied_path = _null_copied_il_path(stage_path)
    source_path = _null_source_il_path(stage_path)
    if not copied_path.exists() or not source_path.exists():
        empty_i64 = np.empty(0, dtype=np.int64)
        return source_stage_id, empty_i64, empty_i64

    copied_il = np.asarray(np.load(copied_path, allow_pickle=False), dtype=np.int64).reshape(-1)
    source_il = np.asarray(np.load(source_path, allow_pickle=False), dtype=np.int64).reshape(-1)
    if int(copied_il.size) != int(source_il.size):
        raise ValueError(f"Null-binding sidecar at {stage_path} has mismatched copied/source row counts.")

    if active_only and int(copied_il.size) > 0:
        active_il = np.asarray(load_leg_column(stage_path, "IL", active_only=True), dtype=np.int64).reshape(-1)
        keep_mask = np.isin(copied_il, active_il)
        copied_il = copied_il[keep_mask]
        source_il = source_il[keep_mask]

    return source_stage_id, copied_il, source_il


def save_flyby_stage(stage_dir: str | Path, flyby_db: Any) -> Path:
    stage_path = Path(stage_dir)
    stage_path.mkdir(parents=True, exist_ok=True)

    if_array = np.asarray(flyby_db.IF, dtype=np.int64).reshape(-1)
    row_count = int(if_array.size)
    columns = {
        "IF": if_array,
        "IE": np.asarray(flyby_db.IE, dtype=np.int64).reshape(-1),
        "IL_in": np.asarray(flyby_db.IL_in, dtype=np.int64).reshape(-1),
        "IL_out": np.asarray(flyby_db.IL_out, dtype=np.int64).reshape(-1),
        "dv_km_s": np.asarray(flyby_db.dv_km_s, dtype=float).reshape(-1),
    }
    for name, arr in columns.items():
        if int(arr.shape[0]) != row_count:
            raise ValueError(f"Flyby column '{name}' has mismatched row count.")
        np.save(stage_path / f"{name}.npy", arr, allow_pickle=False)

    stage_ids = np.asarray(flyby_db.stage_id, dtype=np.int64).reshape(-1)
    if row_count > 0:
        if int(stage_ids.size) != row_count:
            raise ValueError("Flyby stage_id row count mismatch.")
        unique_ids = np.unique(stage_ids)
        if int(unique_ids.size) != 1:
            raise ValueError("save_flyby_stage expects one stage per stage directory.")
        stage_id = int(unique_ids[0])
    else:
        stage_id = _infer_stage_id(stage_path)

    np.save(stage_path / "active_mask.npy", np.ones(row_count, dtype=bool), allow_pickle=False)
    _write_meta(stage_path, kind="flyby", stage_id=stage_id, row_count=row_count)
    return stage_path


def load_active_mask(stage_dir: str | Path) -> np.ndarray:
    return np.asarray(np.load(Path(stage_dir) / "active_mask.npy", allow_pickle=False), dtype=bool).reshape(-1)


def update_active_mask(stage_dir: str | Path, new_mask: np.ndarray) -> None:
    stage_path = Path(stage_dir)
    current = load_active_mask(stage_path)
    candidate = np.asarray(new_mask, dtype=bool).reshape(-1)
    if int(candidate.size) != int(current.size):
        raise ValueError("active_mask length mismatch.")
    np.save(stage_path / "active_mask.npy", candidate, allow_pickle=False)


def _load_column(stage_dir: str | Path, column_name: str, *, active_only: bool) -> np.ndarray:
    stage_path = Path(stage_dir)
    array = np.asarray(np.load(stage_path / f"{column_name}.npy", allow_pickle=False))
    if active_only:
        mask = load_active_mask(stage_path)
        array = array[mask]
    return array


def load_leg_column(stage_dir: str | Path, column_name: str, active_only: bool = False) -> np.ndarray:
    if column_name not in _LEG_COLUMNS:
        raise ValueError(f"Unknown leg column: {column_name}")
    array = _load_column(stage_dir, column_name, active_only=active_only)
    if column_name in {"IL", "ID", "IA", "n_rev"}:
        return np.asarray(array, dtype=np.int64).reshape(-1)
    if column_name in {"vinfD_km_s", "vinfA_km_s"}:
        return np.asarray(array, dtype=float).reshape(-1, 3)
    return np.asarray(array, dtype=float).reshape(-1)


def load_flyby_column(stage_dir: str | Path, column_name: str, active_only: bool = False) -> np.ndarray:
    if column_name not in _FLYBY_COLUMNS:
        raise ValueError(f"Unknown flyby column: {column_name}")
    array = _load_column(stage_dir, column_name, active_only=active_only)
    if column_name in {"IF", "IE", "IL_in", "IL_out"}:
        return np.asarray(array, dtype=np.int64).reshape(-1)
    return np.asarray(array, dtype=float).reshape(-1)


def load_leg_stage(stage_dir: str | Path, active_only: bool = False) -> LegDatabase:
    stage_path = Path(stage_dir)
    meta = _read_meta(stage_path)
    if str(meta.get("kind", "")).lower() != "leg":
        raise ValueError(f"Expected leg stage at {stage_path}.")

    return LegDatabase(
        IL=load_leg_column(stage_path, "IL", active_only=active_only),
        stage_id=int(meta.get("stage_id")),
        ID=load_leg_column(stage_path, "ID", active_only=active_only),
        IA=load_leg_column(stage_path, "IA", active_only=active_only),
        vinfD_km_s=load_leg_column(stage_path, "vinfD_km_s", active_only=active_only),
        vinfA_km_s=load_leg_column(stage_path, "vinfA_km_s", active_only=active_only),
        n_rev=load_leg_column(stage_path, "n_rev", active_only=active_only),
        dv_lev_km_s=load_leg_column(stage_path, "dv_lev_km_s", active_only=active_only),
        eta_lev=load_leg_column(stage_path, "eta_lev", active_only=active_only),
    )


def load_leg_stage_for_flyby(stage_dir: str | Path, active_only: bool = False) -> LegDatabase:
    """Load one leg stage for flyby construction."""

    stage_path = Path(stage_dir)
    meta = _read_meta(stage_path)
    if str(meta.get("kind", "")).lower() != "leg":
        raise ValueError(f"Expected leg stage at {stage_path}.")

    il = load_leg_column(stage_path, "IL", active_only=active_only)
    return LegDatabase(
        IL=il,
        stage_id=int(meta.get("stage_id")),
        ID=load_leg_column(stage_path, "ID", active_only=active_only),
        IA=load_leg_column(stage_path, "IA", active_only=active_only),
        vinfD_km_s=load_leg_column(stage_path, "vinfD_km_s", active_only=active_only),
        vinfA_km_s=load_leg_column(stage_path, "vinfA_km_s", active_only=active_only),
        n_rev=load_leg_column(stage_path, "n_rev", active_only=active_only),
        dv_lev_km_s=load_leg_column(stage_path, "dv_lev_km_s", active_only=active_only),
        eta_lev=load_leg_column(stage_path, "eta_lev", active_only=active_only),
    )


def load_flyby_stage(stage_dir: str | Path, active_only: bool = False):
    try:
        from star.flyby_database import FlybyDB
    except Exception:  # pragma: no cover
        from flyby_database import FlybyDB

    stage_path = Path(stage_dir)
    meta = _read_meta(stage_path)
    if str(meta.get("kind", "")).lower() != "flyby":
        raise ValueError(f"Expected flyby stage at {stage_path}.")

    if_array = load_flyby_column(stage_path, "IF", active_only=active_only)
    return FlybyDB(
        IF=np.asarray(if_array, dtype=np.int64).reshape(-1),
        stage_id=np.full(int(np.asarray(if_array).size), int(meta.get("stage_id")), dtype=np.int64),
        IE=load_flyby_column(stage_path, "IE", active_only=active_only),
        IL_in=load_flyby_column(stage_path, "IL_in", active_only=active_only),
        IL_out=load_flyby_column(stage_path, "IL_out", active_only=active_only),
        dv_km_s=load_flyby_column(stage_path, "dv_km_s", active_only=active_only),
    )


class LegStageNpyWriter:
    """Streaming writer for one leg stage directory."""

    def __init__(self, stage_dir: str | Path, *, stage_id: int, il_start: int = 0):
        self._stage_dir = Path(stage_dir)
        self._stage_id = int(stage_id)
        self._il_next = int(il_start)
        if self._il_next < 0:
            raise ValueError("il_start must be >= 0.")

        self._tmp_dir = self._stage_dir.with_name(self._stage_dir.name + ".build_tmp")
        self._parts_dir = self._tmp_dir / "parts"
        shutil.rmtree(self._tmp_dir, ignore_errors=True)
        self._parts_dir.mkdir(parents=True, exist_ok=True)

        self._part_count = 0
        self._row_count = 0

    @property
    def row_count(self) -> int:
        return int(self._row_count)

    def _part_path(self, column_name: str, part_index: int) -> Path:
        return self._parts_dir / f"{column_name}_{int(part_index):08d}.npy"

    def append_rows(
        self,
        *,
        id: np.ndarray,
        ia: np.ndarray,
        vinfD_km_s: np.ndarray,
        vinfA_km_s: np.ndarray,
        n_rev: np.ndarray,
        dv_lev_km_s: np.ndarray,
        eta_lev: np.ndarray,
    ) -> int:
        id_array = np.asarray(id, dtype=np.int64).reshape(-1)
        ia_array = np.asarray(ia, dtype=np.int64).reshape(-1)
        vinfd_array = np.asarray(vinfD_km_s, dtype=float).reshape(-1, 3)
        vinfa_array = np.asarray(vinfA_km_s, dtype=float).reshape(-1, 3)
        nrev_array = np.asarray(n_rev, dtype=np.int64).reshape(-1)
        dv_array = np.asarray(dv_lev_km_s, dtype=float).reshape(-1)
        eta_array = np.asarray(eta_lev, dtype=float).reshape(-1)
        row_count = int(id_array.size)
       
        if (
            int(ia_array.size) != row_count
            or int(vinfd_array.shape[0]) != row_count
            or int(vinfa_array.shape[0]) != row_count
            or int(nrev_array.size) != row_count
            or int(dv_array.size) != row_count
            or int(eta_array.size) != row_count
        ):
            raise ValueError("append_rows inputs must have the same row count.")
        if row_count == 0:
            return 0

        il_array = np.arange(self._il_next, self._il_next + row_count, dtype=np.int64)
        self._il_next += row_count

        part_index = int(self._part_count)
        np.save(self._part_path("IL", part_index), il_array, allow_pickle=False)
        np.save(self._part_path("ID", part_index), id_array, allow_pickle=False)
        np.save(self._part_path("IA", part_index), ia_array, allow_pickle=False)
        np.save(self._part_path("vinfD_km_s", part_index), vinfd_array, allow_pickle=False)
        np.save(self._part_path("vinfA_km_s", part_index), vinfa_array, allow_pickle=False)
        np.save(self._part_path("n_rev", part_index), nrev_array, allow_pickle=False)
        np.save(self._part_path("dv_lev_km_s", part_index), dv_array, allow_pickle=False)
        np.save(self._part_path("eta_lev", part_index), eta_array, allow_pickle=False)

        self._part_count += 1
        self._row_count += row_count
        return row_count

    def _finalize_column(self, column_name: str) -> None:
        out_path = self._tmp_dir / f"{column_name}.npy"
        dtype = _LEG_COLUMN_DTYPES[column_name]
        total_rows = int(self._row_count)
        if total_rows == 0:
            empty_shape = (0, 3) if column_name in {"vinfD_km_s", "vinfA_km_s"} else (0,)
            np.save(out_path, np.empty(empty_shape, dtype=dtype), allow_pickle=False)
            return

        shape = (total_rows, 3) if column_name in {"vinfD_km_s", "vinfA_km_s"} else (total_rows,)
        out_mem = open_memmap(out_path, mode="w+", dtype=dtype, shape=shape)
        cursor = 0
        for part_index in range(int(self._part_count)):
            part_path = self._part_path(column_name, part_index)
            if len(shape) > 1:
                part_data = np.asarray(np.load(part_path, allow_pickle=False), dtype=dtype).reshape(-1, shape[1])
            else:
                part_data = np.asarray(np.load(part_path, allow_pickle=False), dtype=dtype).reshape(-1)
            n_part = int(part_data.shape[0])
            out_mem[cursor : cursor + n_part] = part_data
            cursor += n_part
        if cursor != total_rows:
            raise RuntimeError(f"Leg writer column '{column_name}' size mismatch: {cursor} != {total_rows}.")
        out_mem.flush()
        del out_mem

    def finalize(self) -> int:
        for column_name in _LEG_COLUMNS:
            self._finalize_column(column_name)
        shutil.rmtree(self._parts_dir, ignore_errors=True)
        np.save(self._tmp_dir / "active_mask.npy", np.ones(self._row_count, dtype=bool), allow_pickle=False)
        _write_meta(self._tmp_dir, kind="leg", stage_id=int(self._stage_id), row_count=int(self._row_count))

        if self._stage_dir.exists():
            shutil.rmtree(self._stage_dir, ignore_errors=True)
        self._stage_dir.parent.mkdir(parents=True, exist_ok=True)
        self._tmp_dir.replace(self._stage_dir)
        return int(self._row_count)

    def abort(self) -> None:
        shutil.rmtree(self._tmp_dir, ignore_errors=True)


class FlybyStageNpyWriter:
    """Streaming writer for one flyby stage directory."""

    def __init__(self, stage_dir: str | Path, *, stage_id: int, if_start: int = 0):
        self._stage_dir = Path(stage_dir)
        self._stage_id = int(stage_id)
        self._if_next = int(if_start)
        if self._if_next < 0:
            raise ValueError("if_start must be >= 0.")

        self._tmp_dir = self._stage_dir.with_name(self._stage_dir.name + ".build_tmp")
        self._parts_dir = self._tmp_dir / "parts"
        shutil.rmtree(self._tmp_dir, ignore_errors=True)
        self._parts_dir.mkdir(parents=True, exist_ok=True)

        self._part_count = 0
        self._row_count = 0

    @property
    def row_count(self) -> int:
        return int(self._row_count)

    def _part_path(self, column_name: str, part_index: int) -> Path:
        return self._parts_dir / f"{column_name}_{int(part_index):08d}.npy"

    def append_rows(
        self,
        *,
        ie: np.ndarray,
        il_in: np.ndarray,
        il_out: np.ndarray,
        dv_km_s: np.ndarray,
    ) -> int:
        ie_array = np.asarray(ie, dtype=np.int64).reshape(-1)
        il_in_array = np.asarray(il_in, dtype=np.int64).reshape(-1)
        il_out_array = np.asarray(il_out, dtype=np.int64).reshape(-1)
        dv_array = np.asarray(dv_km_s, dtype=float).reshape(-1)
        row_count = int(ie_array.size)

        if int(il_in_array.size) != row_count or int(il_out_array.size) != row_count or int(dv_array.size) != row_count:
            raise ValueError("append_rows inputs must have the same row count.")
        if row_count == 0:
            return 0

        if_array = np.arange(self._if_next, self._if_next + row_count, dtype=np.int64)
        self._if_next += row_count

        part_index = int(self._part_count)
        np.save(self._part_path("IF", part_index), if_array, allow_pickle=False)
        np.save(self._part_path("IE", part_index), ie_array, allow_pickle=False)
        np.save(self._part_path("IL_in", part_index), il_in_array, allow_pickle=False)
        np.save(self._part_path("IL_out", part_index), il_out_array, allow_pickle=False)
        np.save(self._part_path("dv_km_s", part_index), dv_array, allow_pickle=False)

        self._part_count += 1
        self._row_count += row_count
        return row_count

    def _finalize_column(self, column_name: str) -> None:
        out_path = self._tmp_dir / f"{column_name}.npy"
        dtype = _FLYBY_COLUMN_DTYPES[column_name]
        total_rows = int(self._row_count)
        if total_rows == 0:
            np.save(out_path, np.empty(0, dtype=dtype), allow_pickle=False)
            return

        out_mem = open_memmap(out_path, mode="w+", dtype=dtype, shape=(total_rows,))
        cursor = 0
        for part_index in range(int(self._part_count)):
            part_path = self._part_path(column_name, part_index)
            part_data = np.asarray(np.load(part_path, allow_pickle=False), dtype=dtype).reshape(-1)
            n_part = int(part_data.size)
            out_mem[cursor : cursor + n_part] = part_data
            cursor += n_part
        if cursor != total_rows:
            raise RuntimeError(f"Flyby writer column '{column_name}' size mismatch: {cursor} != {total_rows}.")
        out_mem.flush()
        del out_mem

    def finalize(self) -> int:
        for column_name in _FLYBY_COLUMNS:
            self._finalize_column(column_name)
        shutil.rmtree(self._parts_dir, ignore_errors=True)
        np.save(self._tmp_dir / "active_mask.npy", np.ones(self._row_count, dtype=bool), allow_pickle=False)
        _write_meta(self._tmp_dir, kind="flyby", stage_id=int(self._stage_id), row_count=int(self._row_count))

        if self._stage_dir.exists():
            shutil.rmtree(self._stage_dir, ignore_errors=True)
        self._stage_dir.parent.mkdir(parents=True, exist_ok=True)
        self._tmp_dir.replace(self._stage_dir)
        return int(self._row_count)

    def abort(self) -> None:
        shutil.rmtree(self._tmp_dir, ignore_errors=True)


def begin_flyby_stage_writer(stage_dir: str | Path, *, stage_id: int, if_start: int = 0) -> FlybyStageNpyWriter:
    return FlybyStageNpyWriter(stage_dir, stage_id=int(stage_id), if_start=int(if_start))


def begin_leg_stage_writer(stage_dir: str | Path, *, stage_id: int, il_start: int = 0) -> LegStageNpyWriter:
    return LegStageNpyWriter(stage_dir, stage_id=int(stage_id), il_start=int(il_start))


def count_active_rows(stage_dir: str | Path) -> int:
    return int(np.count_nonzero(load_active_mask(stage_dir)))

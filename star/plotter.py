"""2D patched-conic trajectory plotter from solution files.

This module reconstructs observer-centered spacecraft leg states without Lambert:
- departure state per leg is `r0 = r_body`, `v0 = v_body + v_inf_departure`,
- propagation uses two-body Kepler universal variables about the selected central body,
- projection is 2D x-y in the selected inertial frame (default ECLIPJ2000).
- includes utilities to plot DV-vs-TOF Pareto fronts from solution JSONL rows.
"""

from __future__ import annotations

import argparse
import ast
import importlib.util
import json
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import spiceypy as spice
from matplotlib.lines import Line2D

root_folder = Path.cwd()

from star.constants import DEFAULT_FRAME, DEFAULT_OBSERVER, get_gm
from star.lambert import kepler_propagate_universal

PLOT_AXIS_METRICS = ("t0", "t", "tf", "tof", "dv")
PLOT_COLOR_METRICS = (*PLOT_AXIS_METRICS, "fb_seq", "vinf0", "vinf_arr", "inc_arr", "none")
LINEWIDTH = 1.8
COLOR = "gold"


@dataclass(frozen=True)
class PlotMetric:
    metric_id: str
    value: float
    axis_label: str
    display_name: str
    is_calendar_date: bool = False


def _parse_bool_cli_arg(raw_value: Any) -> bool:
    """Parse CLI booleans from common textual truthy/falsy values."""

    if isinstance(raw_value, bool):
        return raw_value

    normalized = str(raw_value).strip().lower()
    if normalized in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "f", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(
        "Boolean flag must be one of: True/False, 1/0, yes/no, or on/off."
    )


def _apply_plot_theme(
    fig: Any,
    ax: Any,
    *,
    dark: bool = False,
    colorbar: Any = None,
    legend: Any = None,
    extra_text_artists: Sequence[Any] = (),
) -> None:
    """Apply optional light/dark styling to a Matplotlib figure and axes."""

    if not bool(dark):
        return

    background_color = "black"
    foreground_color = "white"
    frame_color = "#8a8a8a"

    fig.patch.set_facecolor(background_color)
    ax.set_facecolor(background_color)
    ax.title.set_color(foreground_color)
    ax.xaxis.label.set_color(foreground_color)
    ax.yaxis.label.set_color(foreground_color)
    ax.tick_params(axis="both", colors=foreground_color)
    ax.xaxis.get_offset_text().set_color(foreground_color)
    ax.yaxis.get_offset_text().set_color(foreground_color)
    for spine in ax.spines.values():
        spine.set_edgecolor(frame_color)
    ax.grid(True, color=frame_color, alpha=0.3)

    for text_artist in ax.texts:
        text_artist.set_color(foreground_color)

    if legend is not None:
        legend.get_frame().set_facecolor(background_color)
        legend.get_frame().set_edgecolor(frame_color)
        legend.get_frame().set_alpha(1.0)
        legend.get_title().set_color(foreground_color)
        for text_artist in legend.get_texts():
            text_artist.set_color(foreground_color)

    if colorbar is not None:
        colorbar.ax.set_facecolor(background_color)
        colorbar.outline.set_edgecolor(frame_color)
        colorbar.ax.tick_params(colors=foreground_color)
        colorbar.ax.yaxis.label.set_color(foreground_color)
        colorbar.ax.yaxis.get_offset_text().set_color(foreground_color)
        for spine in colorbar.ax.spines.values():
            spine.set_edgecolor(frame_color)

    for text_artist in extra_text_artists:
        if text_artist is not None:
            text_artist.set_color(foreground_color)

def _load_spice_metakernel(metakernel_path: str | Path) -> None:
    """Load SPICE kernels from a meta-kernel path."""

    mk = Path(metakernel_path)
    if not mk.exists():
        raise FileNotFoundError(f"Meta-kernel not found: {mk}")
    spice.kclear()
    spice.furnsh(str(mk))


def _spice_state(
    target_id_or_name: int | str,
    et_s: float,
    observer: str = DEFAULT_OBSERVER,
    frame: str = DEFAULT_FRAME,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return observer-centered inertial state `(r_km[3], v_km_s[3])` from SPICE."""

    state6, _ = spice.spkezr(str(target_id_or_name), float(et_s), str(frame), "NONE", str(observer))
    state6_array = np.asarray(state6, dtype=float)
    return state6_array[:3], state6_array[3:]


def _reconstruct_leg_endpoint_velocities(
    v_body_dep_km_s: Sequence[float],
    v_body_arr_km_s: Sequence[float],
    vinf_dep_km_s: Sequence[float],
    vinf_arr_km_s: Sequence[float],
) -> Tuple[np.ndarray, np.ndarray]:
    """Recover absolute leg endpoint velocities from encounter body states and v-infinity."""

    v0_km_s = np.asarray(v_body_dep_km_s, dtype=float).reshape(3) + np.asarray(vinf_dep_km_s, dtype=float).reshape(3)
    vf_km_s = np.asarray(v_body_arr_km_s, dtype=float).reshape(3) + np.asarray(vinf_arr_km_s, dtype=float).reshape(3)
    return v0_km_s, vf_km_s


def _resolve_spice_body_code(body_id_or_name: int | str) -> int:
    """Resolve SPICE body token (name/ID) to integer NAIF ID."""

    token = str(body_id_or_name).strip()
    if not token:
        raise ValueError("Body token must be a non-empty SPICE body name/NAIF ID.")

    try:
        resolved = spice.bods2c(token)
    except Exception as exc:
        raise ValueError(f"Unable to resolve SPICE body token '{token}'.") from exc

    if isinstance(resolved, tuple):
        if len(resolved) != 2:
            raise ValueError(f"Unexpected bods2c return format for token '{token}': {resolved!r}")
        code, found = resolved
        if not bool(found):
            raise ValueError(f"Unable to resolve SPICE body token '{token}'.")
        return int(code)

    return int(resolved)


def _resolve_central_mu_km3_s2(observer: str) -> float:
    """Resolve two-body GM [km^3/s^2] for the selected observer/central body."""

    observer_id = _resolve_spice_body_code(observer)
    try:
        mu_km3_s2 = float(get_gm(str(observer_id))[0])
    except Exception as exc:
        raise ValueError(
            f"No GM constant available for observer '{observer}' (NAIF ID {observer_id})."
        ) from exc

    if not np.isfinite(mu_km3_s2) or mu_km3_s2 <= 0.0:
        raise ValueError(f"Resolved non-positive/invalid GM for observer '{observer}' (NAIF ID {observer_id}).")
    return mu_km3_s2


def _resolve_central_body_label(observer: str) -> str:
    """Resolve display label for the observer/central body."""

    observer_id = _resolve_spice_body_code(observer)
    try:
        name_resolved = spice.bodc2n(int(observer_id))
        if isinstance(name_resolved, tuple):
            if len(name_resolved) == 2 and bool(name_resolved[1]):
                name_text = str(name_resolved[0]).strip()
                if name_text:
                    return name_text
        else:
            name_text = str(name_resolved).strip()
            if name_text:
                return name_text
    except Exception:
        pass
    return f"Body {observer_id}"


def _resolve_body_sequence_label(record: Mapping[str, Any], row_index: Optional[int] = None) -> str:
    """Resolve one compact flyby-sequence label such as ``EJV`` from ``body_ids``."""

    body_sequence = _parse_body_sequence(record)
    if body_sequence is None:
        if row_index is None:
            raise ValueError("Record is missing a valid non-empty 'body_ids' sequence.")
        raise ValueError(f"Record at index {row_index} is missing a valid non-empty 'body_ids' sequence.")

    initials: List[str] = []
    for body_id in body_sequence:
        body_name = _resolve_central_body_label(str(int(body_id))).strip()
        first_alpha = next((char.upper() for char in body_name if char.isalpha()), None)
        initials.append(first_alpha if first_alpha is not None else "B")
    return "".join(initials)


def read_jsonl_trajectories(path: str | Path, max_rows: Optional[int] = None) -> List[Dict[str, Any]]:
    """Read trajectory records from a JSONL file."""

    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"JSONL file not found: {file_path}")

    records: List[Dict[str, Any]] = []
    with file_path.open("r", encoding="utf-8") as file_obj:
        for line_index, line in enumerate(file_obj):
            if max_rows is not None and len(records) >= int(max_rows):
                break
            stripped = line.strip()
            if not stripped:
                continue
            try:
                record = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_index + 1} in {file_path}.") from exc
            if not isinstance(record, dict):
                raise ValueError(f"JSONL line {line_index + 1} is not an object.")
            records.append(record)

    return records


def _extract_tof_days(record: Mapping[str, Any], row_index: Optional[int] = None) -> float:
    """Extract TOF in days from one trajectory record."""

    if "tof_total_days" in record:
        return float(record["tof_total_days"])
    if "tof_total_s" in record:
        return float(record["tof_total_s"]) / 86400.0
    if row_index is None:
        raise ValueError("Record is missing TOF key 'tof_total_days' (preferred) or 'tof_total_s' (legacy).")
    raise ValueError(
        f"Record at index {row_index} is missing TOF key "
        "'tof_total_days' (preferred) or 'tof_total_s' (legacy)."
    )


def _parse_body_sequence(record: Mapping[str, Any]) -> Optional[Tuple[int, ...]]:
    """Parse body-sequence key from one trajectory record."""

    body_ids = record.get("body_ids", None)
    if not isinstance(body_ids, Sequence):
        return None
    try:
        sequence = tuple(int(value) for value in body_ids)
    except (TypeError, ValueError):
        return None
    if len(sequence) == 0:
        return None
    return sequence


def _dominates_2d(
    tof_a: float,
    dv_a: float,
    tof_b: float,
    dv_b: float,
    *,
    eps: float = 1e-12,
) -> bool:
    """Return True if point B dominates point A in 2D minimization."""

    return (
        (tof_b <= tof_a + float(eps))
        and (dv_b <= dv_a + float(eps))
        and ((tof_b < tof_a - float(eps)) or (dv_b < dv_a - float(eps)))
    )


def _downsample_front_by_tof(
    archive: Sequence[Tuple[float, float, Dict[str, Any]]],
    *,
    max_keep: int,
) -> List[Tuple[float, float, Dict[str, Any]]]:
    """Downsample one non-dominated archive along TOF while preserving front shape."""

    if len(archive) <= int(max_keep):
        return list(archive)
    if int(max_keep) <= 0:
        return []
    if int(max_keep) == 1:
        best = min(archive, key=lambda item: (float(item[1]), float(item[0])))
        return [best]

    sorted_archive = sorted(archive, key=lambda item: (float(item[0]), float(item[1])))
    n_items = len(sorted_archive)
    idx = np.floor(
        np.arange(int(max_keep), dtype=float) * float(n_items - 1) / float(int(max_keep) - 1)
    ).astype(np.int64)
    return [sorted_archive[int(i)] for i in idx]


def _append_to_body_archive(
    archive: List[Tuple[float, float, Dict[str, Any]]],
    *,
    tof_days: float,
    dv_km_s: float,
    record: Dict[str, Any],
    max_keep: int,
) -> None:
    """Insert one candidate into body-sequence Pareto archive with bounded size."""

    for tof_existing, dv_existing, _ in archive:
        if _dominates_2d(float(tof_days), float(dv_km_s), float(tof_existing), float(dv_existing)):
            return

    kept: List[Tuple[float, float, Dict[str, Any]]] = []
    for tof_existing, dv_existing, rec_existing in archive:
        if _dominates_2d(float(tof_existing), float(dv_existing), float(tof_days), float(dv_km_s)):
            continue
        kept.append((float(tof_existing), float(dv_existing), rec_existing))
    kept.append((float(tof_days), float(dv_km_s), record))

    if len(kept) > int(max_keep):
        kept = _downsample_front_by_tof(kept, max_keep=int(max_keep))
    archive.clear()
    archive.extend(kept)


def read_jsonl_trajectories_diverse(
    path: str | Path,
    *,
    include_boundary_dv: Sequence[bool] | None = (False, False),
    max_rows_per_body_sequence: int = 500,
) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    """Stream JSONL and keep up to capped Pareto TOF-DV records per unique body sequence."""

    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"JSONL file not found: {file_path}")

    max_keep = int(max_rows_per_body_sequence)
    if max_keep <= 0:
        raise ValueError("max_rows_per_body_sequence must be > 0.")

    archives: Dict[Tuple[int, ...], List[Tuple[float, float, Dict[str, Any]]]] = {}
    num_rows_read = 0
    num_rows_invalid = 0
    num_rows_kept = 0
    num_rows_missing_body = 0

    with file_path.open("r", encoding="utf-8") as file_obj:
        for line_index, line in enumerate(file_obj):
            stripped = line.strip()
            if not stripped:
                continue
            num_rows_read += 1
            try:
                record = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_index + 1} in {file_path}.") from exc
            if not isinstance(record, dict):
                raise ValueError(f"JSONL line {line_index + 1} is not an object.")

            body_sequence = _parse_body_sequence(record)
            if body_sequence is None:
                num_rows_missing_body += 1
                continue

            try:
                dv_km_s, _, _ = _resolved_dv_for_record(record, include_boundary_dv)
                tof_days = _extract_tof_days(record, row_index=line_index)
            except Exception:
                num_rows_invalid += 1
                continue

            if not np.isfinite(dv_km_s) or not np.isfinite(tof_days):
                num_rows_invalid += 1
                continue

            archive = archives.setdefault(body_sequence, [])
            before = len(archive)
            _append_to_body_archive(
                archive,
                tof_days=float(tof_days),
                dv_km_s=float(dv_km_s),
                record=record,
                max_keep=max_keep,
            )
            if len(archive) > before:
                num_rows_kept += 1

    kept_records: List[Dict[str, Any]] = []
    for body_sequence in sorted(archives.keys()):
        archive = sorted(archives[body_sequence], key=lambda item: (float(item[0]), float(item[1])))
        kept_records.extend(record for _, _, record in archive)

    stats = {
        "num_rows_read": int(num_rows_read),
        "num_rows_invalid": int(num_rows_invalid),
        "num_rows_missing_body": int(num_rows_missing_body),
        "num_body_sequences": int(len(archives)),
        "num_records_out": int(len(kept_records)),
        "max_rows_per_body_sequence": int(max_keep),
        "num_rows_archive_growth_events": int(num_rows_kept),
    }
    return kept_records, stats


def _normalize_include_boundary_dv(value: Sequence[bool] | None) -> Tuple[bool, bool]:
    """Normalize boundary-DV include flags to `(include_escape, include_insertion)`."""

    if value is None:
        return False, False

    values = list(value)
    if len(values) != 2:
        raise ValueError("include_boundary_dv must contain exactly two booleans: [escape, insertion].")
    if not isinstance(values[0], bool) or not isinstance(values[1], bool):
        raise ValueError("include_boundary_dv must be a boolean list like [True, False].")
    return values[0], values[1]


def _resolved_dv_for_record(
    record: Mapping[str, Any],
    include_boundary_dv: Sequence[bool] | None,
) -> Tuple[float, bool, bool]:
    """Resolve displayed DV from one record under boundary include flags.

    Returns:
        dv_km_s, missing_escape_field, missing_insertion_field
    """

    include_escape, include_insertion = _normalize_include_boundary_dv(include_boundary_dv)
    dv_total = float(record["dv_total_km_s"])

    missing_escape = False
    missing_insertion = False

    if include_escape:
        escape_value = record.get("dv_escape_km_s", None)
        if escape_value is None:
            missing_escape = True
        else:
            dv_total += float(escape_value)

    if include_insertion:
        insertion_value = record.get("dv_insertion_km_s", None)
        if insertion_value is None:
            missing_insertion = True
        else:
            dv_total += float(insertion_value)

    return float(dv_total), bool(missing_escape), bool(missing_insertion)


def _extract_t0_et_s(record: Mapping[str, Any], row_index: Optional[int] = None) -> float:
    """Extract departure epoch ET seconds from one trajectory record."""

    t_et_s = record.get("t_et_s", None)
    if isinstance(t_et_s, Sequence) and len(t_et_s) > 0:
        return float(t_et_s[0])
    if row_index is None:
        raise ValueError("Record is missing non-empty 't_et_s' for departure epoch extraction.")
    raise ValueError(f"Record at index {row_index} is missing non-empty 't_et_s' for departure epoch extraction.")


def _extract_tf_et_s(record: Mapping[str, Any], row_index: Optional[int] = None) -> float:
    """Extract final-arrival epoch ET seconds from one trajectory record."""

    t_et_s = record.get("t_et_s", None)
    if isinstance(t_et_s, Sequence) and len(t_et_s) > 0:
        return float(t_et_s[-1])
    if row_index is None:
        raise ValueError("Record is missing non-empty 't_et_s' for final-arrival epoch extraction.")
    raise ValueError(
        f"Record at index {row_index} is missing non-empty 't_et_s' for final-arrival epoch extraction."
    )


def _extract_encounter_et_s(
    record: Mapping[str, Any],
    *,
    enc: int,
    row_index: Optional[int] = None,
) -> float:
    """Extract one encounter epoch ET seconds by 0-based encounter index."""

    t_et_s = record.get("t_et_s", None)
    if not isinstance(t_et_s, Sequence) or len(t_et_s) == 0:
        if row_index is None:
            raise ValueError("Record is missing non-empty 't_et_s' for encounter epoch extraction.")
        raise ValueError(f"Record at index {row_index} is missing non-empty 't_et_s' for encounter epoch extraction.")

    enc_idx = int(enc)
    if enc_idx < 0 or enc_idx >= len(t_et_s):
        if row_index is None:
            raise ValueError(f"enc={enc} out of range for encounter epoch extraction (t_et_s has {len(t_et_s)} epochs).")
        raise ValueError(
            f"Record at index {row_index}: enc={enc} out of range for encounter epoch extraction "
            f"(t_et_s has {len(t_et_s)} epochs)."
        )
    return float(t_et_s[enc_idx])


def _recover_arrival_state_for_encounter(
    record: Mapping[str, Any],
    *,
    enc: int,
    observer: str = DEFAULT_OBSERVER,
    frame: str = DEFAULT_FRAME,
    row_index: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Recover encounter-arrival spacecraft state from body ephemeris and `vinfA_km_s`."""

    if int(enc) < 1:
        raise ValueError(f"--enc must be >= 1 for arrival-state recovery (enc=0 has no arrival v-infinity).")

    t_et_s = record.get("t_et_s", None)
    if t_et_s is None:
        if row_index is None:
            raise ValueError("Record is missing 't_et_s' for arrival-state recovery.")
        raise ValueError(f"Record at index {row_index} is missing 't_et_s' for arrival-state recovery.")
    body_ids = record.get("body_ids", None)
    if body_ids is None:
        if row_index is None:
            raise ValueError("Record is missing 'body_ids' for arrival-state recovery.")
        raise ValueError(f"Record at index {row_index} is missing 'body_ids' for arrival-state recovery.")
    vinf_a = record.get("vinfA_km_s", None)
    if vinf_a is None:
        if row_index is None:
            raise ValueError("Record is missing 'vinfA_km_s' for arrival-state recovery.")
        raise ValueError(f"Record at index {row_index} is missing 'vinfA_km_s' for arrival-state recovery.")

    t_et_s_arr = np.asarray(t_et_s, dtype=float).reshape(-1)
    body_ids_arr = np.asarray(body_ids, dtype=np.int64).reshape(-1)
    vinf_a_arr = np.asarray(vinf_a, dtype=float).reshape(-1, 3)
    enc_idx = int(enc)
    leg_idx = enc_idx - 1

    if enc_idx >= t_et_s_arr.size or enc_idx >= body_ids_arr.size:
        if row_index is None:
            raise ValueError(
                f"enc={enc} out of range for arrival-state recovery "
                f"(t_et_s/body_ids have {min(t_et_s_arr.size, body_ids_arr.size)} encounters)."
            )
        raise ValueError(
            f"Record at index {row_index}: enc={enc} out of range for arrival-state recovery "
            f"(t_et_s/body_ids have {min(t_et_s_arr.size, body_ids_arr.size)} encounters)."
        )
    if leg_idx >= vinf_a_arr.shape[0]:
        if row_index is None:
            raise ValueError(
                f"enc={enc} out of range for arrival-state recovery (vinfA_km_s has {vinf_a_arr.shape[0]} legs)."
            )
        raise ValueError(
            f"Record at index {row_index}: enc={enc} out of range for arrival-state recovery "
            f"(vinfA_km_s has {vinf_a_arr.shape[0]} legs)."
        )

    r_body_km, v_body_km_s = _spice_state(
        int(body_ids_arr[enc_idx]),
        float(t_et_s_arr[enc_idx]),
        observer=observer,
        frame=frame,
    )
    r_arr_km = np.asarray(r_body_km, dtype=float).reshape(3)
    v_arr_km_s = np.asarray(v_body_km_s, dtype=float).reshape(3) + np.asarray(vinf_a_arr[leg_idx], dtype=float).reshape(3)
    return r_arr_km, v_arr_km_s


def _compute_inclination_deg(r_km: Sequence[float], v_km_s: Sequence[float]) -> float:
    """Compute orbit inclination [deg] from inertial position/velocity about the frame z-axis."""

    h_km2_s = np.cross(np.asarray(r_km, dtype=float).reshape(3), np.asarray(v_km_s, dtype=float).reshape(3))
    h_norm = float(np.linalg.norm(h_km2_s))
    if not np.isfinite(h_norm) or h_norm <= 0.0:
        raise ValueError("Cannot compute inclination from a degenerate position/velocity state.")
    cos_inc = float(np.clip(h_km2_s[2] / h_norm, -1.0, 1.0))
    return float(np.degrees(np.arccos(cos_inc)))


def _extract_plot_metric(
    record: Mapping[str, Any],
    metric: str,
    *,
    include_boundary_dv: Sequence[bool] | None,
    row_index: Optional[int] = None,
    enc: int = 0,
    observer: str = DEFAULT_OBSERVER,
    frame: str = DEFAULT_FRAME,
) -> PlotMetric:
    """Extract one plot metric from a trajectory record."""

    metric_key = str(metric)
    if metric_key == "t0":
        return PlotMetric(
            metric_id="t0",
            value=_extract_t0_et_s(record, row_index=row_index),
            axis_label="departure date",
            display_name="departure_date",
            is_calendar_date=True,
        )
    if metric_key == "t":
        return PlotMetric(
            metric_id="t",
            value=_extract_encounter_et_s(record, enc=enc, row_index=row_index),
            axis_label=f"encounter date [{enc}]",
            display_name=f"encounter_{enc}_date",
            is_calendar_date=True,
        )
    if metric_key == "tf":
        return PlotMetric(
            metric_id="tf",
            value=_extract_tf_et_s(record, row_index=row_index),
            axis_label="final arrival date",
            display_name="final_arrival_date",
            is_calendar_date=True,
        )
    if metric_key == "tof":
        return PlotMetric(
            metric_id="tof",
            value=_extract_tof_days(record, row_index=row_index),
            axis_label="tof_total [days]",
            display_name="tof_days",
        )
    if metric_key == "dv":
        dv_km_s, _, _ = _resolved_dv_for_record(record, include_boundary_dv)
        return PlotMetric(
            metric_id="dv",
            value=float(dv_km_s),
            axis_label="dv_total(displayed) [km/s]",
            display_name="dv_total_km_s",
        )
    if metric_key == "vinf0":
        vinf_d = record.get("vinfD_km_s", None)
        if vinf_d is None or len(vinf_d) == 0:
            if row_index is None:
                raise ValueError("Record is missing 'vinfD_km_s' for vinf0 extraction.")
            raise ValueError(f"Record at index {row_index} is missing 'vinfD_km_s' for vinf0 extraction.")
        vinf0_mag = float(np.linalg.norm(np.asarray(vinf_d[0], dtype=float)))
        return PlotMetric(
            metric_id="vinf0",
            value=vinf0_mag,
            axis_label="vinf_dep [km/s]",
            display_name="vinf_dep_km_s",
        )
    if metric_key == "vinf_arr":
        # vinfA_km_s[k] is arrival v-infinity at the end of leg k, i.e., at encounter k+1.
        # So arrival vinf at encounter enc = vinfA_km_s[enc - 1].
        if int(enc) < 1:
            raise ValueError(f"--enc must be >= 1 for vinf_arr (enc=0 has no arrival v-infinity).")
        vinf_a = record.get("vinfA_km_s", None)
        if vinf_a is None:
            if row_index is None:
                raise ValueError("Record is missing 'vinfA_km_s' for vinf_arr extraction.")
            raise ValueError(f"Record at index {row_index} is missing 'vinfA_km_s' for vinf_arr extraction.")
        vinf_a_arr = np.asarray(vinf_a, dtype=float)
        leg_idx = int(enc) - 1
        if leg_idx >= len(vinf_a_arr):
            if row_index is None:
                raise ValueError(
                    f"enc={enc} out of range for vinf_arr (vinfA_km_s has {len(vinf_a_arr)} legs)."
                )
            raise ValueError(
                f"Record at index {row_index}: enc={enc} out of range for vinf_arr "
                f"(vinfA_km_s has {len(vinf_a_arr)} legs)."
            )
        vinf_arr_mag = float(np.linalg.norm(vinf_a_arr[leg_idx]))
        return PlotMetric(
            metric_id="vinf_arr",
            value=vinf_arr_mag,
            axis_label=f"vinf_arr[{enc}] [km/s]",
            display_name=f"vinf_arr_{enc}_km_s",
        )
    if metric_key == "inc_arr":
        r_arr_km, v_arr_km_s = _recover_arrival_state_for_encounter(
            record,
            enc=enc,
            observer=observer,
            frame=frame,
            row_index=row_index,
        )
        inc_arr_deg = _compute_inclination_deg(r_arr_km, v_arr_km_s)
        return PlotMetric(
            metric_id="inc_arr",
            value=inc_arr_deg,
            axis_label=f"inc_arr[{enc}] [deg]",
            display_name=f"inc_arr_{enc}_deg",
        )
    raise ValueError(f"Unsupported plot metric '{metric_key}'. Expected one of {PLOT_COLOR_METRICS}.")


def _format_et_tick(et_s: float, _pos: float) -> str:
    """Format ET seconds as a YYYY-MM-DD date for plot axes."""

    try:
        return spice.et2utc(float(et_s), "ISOC", 0)[:10]
    except Exception:
        return ""


def _is_calendar_metric(metric: str) -> bool:
    """Return whether a metric should be rendered as a calendar date."""

    return str(metric) in {"t0", "t", "tf"}


def _apply_metric_axis_format(ax: Any, which_axis: str, metric: str) -> None:
    """Apply axis formatting for the selected plot metric."""

    if not _is_calendar_metric(metric):
        return
    formatter = mticker.FuncFormatter(_format_et_tick)
    if which_axis == "x":
        ax.xaxis.set_major_formatter(formatter)
        ax.tick_params(axis="x", labelrotation=30)
    elif which_axis == "y":
        ax.yaxis.set_major_formatter(formatter)


def _maybe_apply_colorbar_format(colorbar: Any, metric: str) -> None:
    """Apply display formatting to the colorbar for date-valued metrics."""

    if not _is_calendar_metric(metric):
        return
    colorbar.ax.yaxis.set_major_formatter(mticker.FuncFormatter(_format_et_tick))
    colorbar.update_ticks()


def _pareto_minimize_2d_indices(x: np.ndarray, y: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Return indices of non-dominated points for 2-objective minimization.

    For each point `(x, y)`, lower is better in both objectives.
    """

    x_array = np.asarray(x, dtype=float).reshape(-1)
    y_array = np.asarray(y, dtype=float).reshape(-1)
    if x_array.shape != y_array.shape:
        raise ValueError("x and y must have the same shape.")

    if x_array.size == 0:
        return np.empty(0, dtype=np.int64)

    # Sort by x first, then y; sweep to retain strictly improving y.
    order = np.lexsort((y_array, x_array))
    best_y = float("inf")
    front_indices: List[int] = []
    for index in order:
        y_value = float(y_array[int(index)])
        if y_value < best_y - float(eps):
            front_indices.append(int(index))
            best_y = y_value

    return np.asarray(front_indices, dtype=np.int64)


def plot_trajectory_metric_scatter(
    records: Sequence[Dict[str, Any]],
    *,
    x_metric: str = "tof",
    y_metric: str = "dv",
    color_metric: str = "none",
    dark: bool = False,
    enc: int = 0,
    observer: str = DEFAULT_OBSERVER,
    frame: str = DEFAULT_FRAME,
    include_boundary_dv: Sequence[bool] | None = (False, False),
    show_all_points: bool = True,
    save_path: Optional[str | Path] = None,
    show: bool = True,
    interactive_pick: bool = False,
    return_plot_data: bool = False,
):
    """Plot trajectory records in a configurable metric space."""

    if not records:
        raise ValueError("No trajectory records were provided.")
    if x_metric not in PLOT_AXIS_METRICS:
        raise ValueError(f"x_metric must be one of {PLOT_AXIS_METRICS}.")
    if y_metric not in PLOT_AXIS_METRICS:
        raise ValueError(f"y_metric must be one of {PLOT_AXIS_METRICS}.")
    if color_metric not in PLOT_COLOR_METRICS:
        raise ValueError(f"color_metric must be one of {PLOT_COLOR_METRICS}.")

    x_values: List[float] = []
    y_values: List[float] = []
    color_values: List[float] = []
    color_labels: List[str] = []
    record_indices: List[int] = []

    missing_escape_count = 0
    missing_insertion_count = 0
    for row_index, record in enumerate(records):
        if "dv_total_km_s" not in record:
            raise ValueError(
                f"Record at index {row_index} is missing required keys "
                "'dv_total_km_s'."
            )

        _, missing_escape, missing_insertion = _resolved_dv_for_record(record, include_boundary_dv)
        if missing_escape:
            missing_escape_count += 1
        if missing_insertion:
            missing_insertion_count += 1

        try:
            color_label = None
            x_metric_info = _extract_plot_metric(
                record,
                x_metric,
                include_boundary_dv=include_boundary_dv,
                row_index=row_index,
                enc=enc,
                observer=observer,
                frame=frame,
            )
            y_metric_info = _extract_plot_metric(
                record,
                y_metric,
                include_boundary_dv=include_boundary_dv,
                row_index=row_index,
                enc=enc,
                observer=observer,
                frame=frame,
            )
            color_metric_info = None
            if color_metric != "none":
                if color_metric == "fb_seq":
                    color_label = _resolve_body_sequence_label(record, row_index=row_index)
                else:
                    color_metric_info = _extract_plot_metric(
                        record,
                        color_metric,
                        include_boundary_dv=include_boundary_dv,
                        row_index=row_index,
                        enc=enc,
                        observer=observer,
                        frame=frame,
                    )
        except Exception as exc:
            raise ValueError(f"Failed to extract plot metrics for record at index {row_index}.") from exc

        if not np.isfinite(x_metric_info.value) or not np.isfinite(y_metric_info.value):
            warnings.warn(
                f"Skipping row index {row_index} with non-finite values: "
                f"{x_metric}={x_metric_info.value}, {y_metric}={y_metric_info.value}."
            )
            continue
        if color_metric_info is not None and not np.isfinite(color_metric_info.value):
            warnings.warn(
                f"Skipping row index {row_index} with non-finite color value: "
                f"{color_metric}={color_metric_info.value}."
            )
            continue

        x_values.append(float(x_metric_info.value))
        y_values.append(float(y_metric_info.value))
        if color_metric_info is not None:
            color_values.append(float(color_metric_info.value))
        if color_label is not None:
            color_labels.append(str(color_label))
        record_indices.append(int(row_index))

    if not x_values:
        raise ValueError("No valid rows with finite plot metric values were found.")

    include_escape, include_insertion = _normalize_include_boundary_dv(include_boundary_dv)
    if include_escape and missing_escape_count > 0:
        warnings.warn(
            f"{missing_escape_count} record(s) missing dv_escape_km_s while include_boundary_dv[0]=True; "
            "treated as 0.0."
        )
    if include_insertion and missing_insertion_count > 0:
        warnings.warn(
            f"{missing_insertion_count} record(s) missing dv_insertion_km_s while include_boundary_dv[1]=True; "
            "treated as 0.0."
        )

    x_array = np.asarray(x_values, dtype=float)
    y_array = np.asarray(y_values, dtype=float)
    color_array = np.asarray(color_values, dtype=float) if color_metric not in ("none", "fb_seq") else None
    record_indices_array = np.asarray(record_indices, dtype=np.int64)

    x_metric_info = _extract_plot_metric(
        records[int(record_indices_array[0])],
        x_metric,
        include_boundary_dv=include_boundary_dv,
        enc=enc,
        observer=observer,
        frame=frame,
    )
    y_metric_info = _extract_plot_metric(
        records[int(record_indices_array[0])],
        y_metric,
        include_boundary_dv=include_boundary_dv,
        enc=enc,
        observer=observer,
        frame=frame,
    )
    color_metric_info = None
    if color_metric not in ("none", "fb_seq"):
        color_metric_info = _extract_plot_metric(
            records[int(record_indices_array[0])],
            color_metric,
            include_boundary_dv=include_boundary_dv,
            enc=enc,
            observer=observer,
            frame=frame,
        )
    seq_labels_array = np.asarray(color_labels, dtype=object) if color_metric == "fb_seq" else None
    seq_color_array = None
    seq_label_to_color: Dict[str, Any] = {}
    if seq_labels_array is not None:
        unique_seq_labels = list(dict.fromkeys(str(label) for label in seq_labels_array.tolist()))
        cmap = plt.get_cmap("tab20", max(len(unique_seq_labels), 1))
        seq_label_to_color = {
            label: cmap(idx % cmap.N)
            for idx, label in enumerate(unique_seq_labels)
        }
        seq_color_array = np.asarray(
            [seq_label_to_color[str(label)] for label in seq_labels_array.tolist()],
            dtype=object,
        )

    pareto_local_indices = None
    pareto_record_indices = None
    pareto_x_values = None
    pareto_y_values = None
    if x_metric == "tof" and y_metric == "dv":
        pareto_local_indices = _pareto_minimize_2d_indices(x_array, y_array)
        pareto_record_indices = record_indices_array[pareto_local_indices]
        pareto_x_values = x_array[pareto_local_indices]
        pareto_y_values = y_array[pareto_local_indices]

    fig, ax = plt.subplots(figsize=(9, 6))
    scatter_kwargs: Dict[str, Any] = {
        "s": 20,
        "alpha": 0.65 if color_metric != "none" else 0.35,
        "picker": 5 if bool(interactive_pick) else False,
        "label": f"All trajectories (N={int(x_array.size)})",
    }
    if seq_color_array is not None:
        scatter_kwargs["color"] = seq_color_array.tolist()
    elif color_array is None:
        scatter_kwargs["color"] = "tab:blue"
    else:
        scatter_kwargs["c"] = color_array
        scatter_kwargs["cmap"] = "viridis"
    all_points_artist = ax.scatter(x_array, y_array, **scatter_kwargs) if bool(show_all_points) else None

    colorbar = None
    if color_array is not None and all_points_artist is not None:
        colorbar = fig.colorbar(all_points_artist, ax=ax)
        colorbar.set_label(color_metric_info.axis_label)
        _maybe_apply_colorbar_format(colorbar, color_metric)

    pareto_line_artist = None
    if pareto_local_indices is not None and pareto_x_values is not None and pareto_y_values is not None:
        ax.plot(
            pareto_x_values,
            pareto_y_values,
            "-o",
            linewidth=2.0,
            markersize=5.0,
            color="tab:red",
            label=f"Pareto front (N={int(pareto_local_indices.size)})",
            picker=5 if bool(interactive_pick) else False,
        )
        pareto_line_artist = ax.lines[-1]

    ax.set_xlabel(x_metric_info.axis_label)
    ax.set_ylabel(y_metric_info.axis_label)
    title = f"Trajectory Trade Plot: {y_metric} vs {x_metric}"
    if color_metric != "none":
        title += f" | color={color_metric}"
    title += f" | include_boundary_dv={[bool(include_escape), bool(include_insertion)]}"
    ax.set_title(title)
    _apply_metric_axis_format(ax, "x", x_metric)
    _apply_metric_axis_format(ax, "y", y_metric)
    ax.grid(True, alpha=0.3)
    legend = None
    if seq_color_array is not None:
        legend_handles = [
            Line2D(
                [0],
                [0],
                marker="o",
                linestyle="None",
                markerfacecolor=seq_label_to_color[label],
                markeredgecolor=seq_label_to_color[label],
                markersize=6,
                label=label,
            )
            for label in seq_label_to_color
        ]
        if pareto_line_artist is not None:
            legend_handles.append(
                Line2D(
                    [0],
                    [0],
                    linestyle="-",
                    marker="o",
                    color="tab:red",
                    linewidth=2.0,
                    markersize=5.0,
                    label=f"Pareto front (N={int(pareto_local_indices.size)})",
                )
            )
        legend = ax.legend(handles=legend_handles, loc="best", title="Flyby sequence")
    else:
        legend = ax.legend(loc="best")
    _apply_plot_theme(fig, ax, dark=dark, colorbar=colorbar, legend=legend)
    fig.tight_layout()

    if save_path is not None:
        output_path = Path(save_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=180, bbox_inches="tight")

    if show:
        plt.show()

    if bool(return_plot_data):
        plot_data = {
            "all_points_artist": all_points_artist,
            "pareto_line_artist": pareto_line_artist,
            "all_record_indices": record_indices_array,
            "all_x_values": x_array,
            "all_y_values": y_array,
            "all_color_values": color_array,
            "all_color_labels": seq_labels_array,
            "x_metric": x_metric,
            "y_metric": y_metric,
            "color_metric": color_metric,
            "x_axis_label": x_metric_info.axis_label,
            "y_axis_label": y_metric_info.axis_label,
            "pareto_record_indices": pareto_record_indices,
            "pareto_x_values": pareto_x_values,
            "pareto_y_values": pareto_y_values,
            "colorbar": colorbar,
        }
        return fig, ax, pareto_record_indices, plot_data

    return fig, ax, pareto_record_indices


def plot_pareto_front_dv_vs_tof(
    records: Sequence[Dict[str, Any]],
    include_boundary_dv: Sequence[bool] | None = (False, False),
    show_all_points: bool = True,
    save_path: Optional[str | Path] = None,
    show: bool = True,
    interactive_pick: bool = False,
    return_plot_data: bool = False,
    dark: bool = False,
):
    """Backward-compatible wrapper for the default TOF-vs-DV trade plot."""

    return plot_trajectory_metric_scatter(
        records=records,
        x_metric="tof",
        y_metric="dv",
        color_metric="none",
        dark=dark,
        include_boundary_dv=include_boundary_dv,
        show_all_points=show_all_points,
        save_path=save_path,
        show=show,
        interactive_pick=interactive_pick,
        return_plot_data=return_plot_data,
    )


def _resolve_pick_local_index(
    candidate_indices: np.ndarray,
    x_values: np.ndarray,
    y_values: np.ndarray,
    mouse_x: Optional[float],
    mouse_y: Optional[float],
) -> int:
    """Resolve a pick to a single local index, preferring nearest to cursor."""

    if candidate_indices.size == 0:
        raise ValueError("No picked indices were provided.")
    if candidate_indices.size == 1 or mouse_x is None or mouse_y is None:
        return int(candidate_indices[0])

    x_c = x_values[candidate_indices]
    y_c = y_values[candidate_indices]
    dist2 = (x_c - float(mouse_x)) ** 2 + (y_c - float(mouse_y)) ** 2
    best_local = int(np.argmin(dist2))
    return int(candidate_indices[best_local])


def _connect_trade_plot_pick_handler(
    pareto_fig: Any,
    pareto_ax: Any,
    records: Sequence[Dict[str, Any]],
    pareto_plot_data: Dict[str, Any],
    mu_central: float,
    central_body_label: str,
    frame: str,
    observer: str,
    num_samples_leg: int,
    num_samples_orbit: int,
    axis: str,
    include_boundary_dv: Sequence[bool] | None,
    closure_pos_tol_km: float,
    closure_vel_tol_km_s: float,
    dark: bool = False,
    out_path_template: Optional[str | Path] = None,
) -> None:
    """Attach callback that plots a clicked trajectory from the trade-space chart."""

    all_points_artist = pareto_plot_data.get("all_points_artist", None)
    pareto_line_artist = pareto_plot_data.get("pareto_line_artist", None)

    all_record_indices = np.asarray(pareto_plot_data["all_record_indices"], dtype=np.int64)
    all_x_values = np.asarray(pareto_plot_data["all_x_values"], dtype=float)
    all_y_values = np.asarray(pareto_plot_data["all_y_values"], dtype=float)
    x_metric = str(pareto_plot_data["x_metric"])
    y_metric = str(pareto_plot_data["y_metric"])
    pareto_record_indices_raw = pareto_plot_data.get("pareto_record_indices", None)
    pareto_x_values_raw = pareto_plot_data.get("pareto_x_values", None)
    pareto_y_values_raw = pareto_plot_data.get("pareto_y_values", None)
    pareto_record_indices = (
        np.asarray(pareto_record_indices_raw, dtype=np.int64)
        if pareto_record_indices_raw is not None
        else None
    )
    pareto_x_values = (
        np.asarray(pareto_x_values_raw, dtype=float)
        if pareto_x_values_raw is not None
        else None
    )
    pareto_y_values = (
        np.asarray(pareto_y_values_raw, dtype=float)
        if pareto_y_values_raw is not None
        else None
    )

    _last_mouse_event_id: List[Optional[int]] = [None]

    def _on_pick(event: Any) -> None:
        # When a Pareto overlay is present, a click near a Pareto point can fire
        # both scatter and line pick events. Deduplicate on the shared mouse event.
        mouse_event = getattr(event, "mouseevent", None)
        mouse_id = id(mouse_event) if mouse_event is not None else None
        if pareto_line_artist is not None and mouse_id is not None and mouse_id == _last_mouse_event_id[0]:
            return
        if pareto_line_artist is not None:
            _last_mouse_event_id[0] = mouse_id

        artist = event.artist
        picked_indices = np.asarray(getattr(event, "ind", []), dtype=np.int64)
        if picked_indices.size == 0:
            return

        mouse_event = getattr(event, "mouseevent", None)
        mouse_x = getattr(mouse_event, "xdata", None)
        mouse_y = getattr(mouse_event, "ydata", None)

        record_index: Optional[int] = None
        x_value: Optional[float] = None
        y_value: Optional[float] = None

        if all_points_artist is not None and artist is all_points_artist:
            local_index = _resolve_pick_local_index(picked_indices, all_x_values, all_y_values, mouse_x, mouse_y)
            record_index = int(all_record_indices[local_index])
            x_value = float(all_x_values[local_index])
            y_value = float(all_y_values[local_index])
        elif artist is pareto_line_artist and pareto_record_indices is not None and pareto_x_values is not None and pareto_y_values is not None:
            local_index = _resolve_pick_local_index(
                picked_indices,
                pareto_x_values,
                pareto_y_values,
                mouse_x,
                mouse_y,
            )
            record_index = int(pareto_record_indices[local_index])
            x_value = float(pareto_x_values[local_index])
            y_value = float(pareto_y_values[local_index])
        else:
            return

        if record_index < 0 or record_index >= len(records):
            warnings.warn(f"Picked record index {record_index} is out of bounds; click ignored.")
            return

        traj_record = records[int(record_index)]
        traj_id_value = int(traj_record.get("traj_id", int(record_index)))

        print(
            f"picked_record_idx={int(record_index)} "
            f"traj_id={int(traj_id_value)} "
            f"{x_metric}={float(x_value):.6f} "
            f"{y_metric}={float(y_value):.6f}"
        )

        # Keep visual selection history on Pareto chart.
        pareto_ax.scatter(
            [float(x_value)],
            [float(y_value)],
            s=90,
            marker="o",
            facecolors="none",
            edgecolors="white" if bool(dark) else "black",
            linewidths=1.3,
            zorder=8,
            label="_nolegend_",
        )
        pareto_fig.canvas.draw_idle()

        click_save_path = None
        if out_path_template is not None:
            out_path = Path(out_path_template)
            suffix = out_path.suffix if out_path.suffix else ".png"
            click_save_path = out_path.with_name(f"{out_path.stem}_traj_{traj_id_value}{suffix}")

        traj_fig, _ = plot_trajectory_2d(
            traj_record=traj_record,
            mu_central=mu_central,
            frame=frame,
            observer=observer,
            central_body_label=central_body_label,
            num_samples_leg=int(num_samples_leg),
            num_samples_orbit=int(num_samples_orbit),
            axis=axis,
            include_boundary_dv=include_boundary_dv,
            closure_pos_tol_km=float(closure_pos_tol_km),
            closure_vel_tol_km_s=float(closure_vel_tol_km_s),
            dark=dark,
            save_path=click_save_path,
            show=False,
        )
        if click_save_path is not None:
            print(f"saved_trajectory_plot={click_save_path}")

        try:
            traj_fig.canvas.manager.set_window_title(f"Trajectory {traj_id_value}")
        except Exception:
            pass
        traj_fig.canvas.draw_idle()
        plt.show(block=False)

    pareto_fig.canvas.mpl_connect("pick_event", _on_pick)


def estimate_orbital_period_s(
    r_km: Sequence[float],
    v_km_s: Sequence[float],
    mu_km3_s2: float,
) -> float:
    """Estimate two-body orbital period [seconds] from observer-centered state.

    Uses specific orbital energy:
        eps = v^2/2 - mu/r
        a = -mu/(2*eps)      (for eps < 0)
        T = 2*pi*sqrt(a^3/mu)
    Returns NaN for non-elliptic or invalid states.
    """

    r_vec = np.asarray(r_km, dtype=float).reshape(3)
    v_vec = np.asarray(v_km_s, dtype=float).reshape(3)
    mu = float(mu_km3_s2)
    if mu <= 0.0:
        return float("nan")

    r_norm = float(np.linalg.norm(r_vec))
    eps = 0.5 * float(np.dot(v_vec, v_vec)) - mu / r_norm
    if eps >= 0.0:
        return float("nan")

    a_km = -mu / (2.0 * eps)

    return float(2.0 * np.pi * np.sqrt((a_km**3) / mu))


def sample_leg(
    r0_km: Sequence[float],
    v0_km_s: Sequence[float],
    t0_et_s: float,
    t1_et_s: float,
    mu_km3_s2: float,
    num_samples: int,
) -> np.ndarray:
    """Sample one propagated leg with evenly spaced epochs.

    Sampling is done incrementally (`dt_step` between consecutive points) to reduce
    numerical spikes from very large single-shot propagation intervals.
    """

    n_samples = int(num_samples)
    if n_samples < 2:
        raise ValueError("num_samples must be >= 2.")

    dt_total_s = float(t1_et_s) - float(t0_et_s)
    if dt_total_s <= 0.0:
        raise ValueError("Leg duration must be positive.")

    positions_km = np.empty((n_samples, 3), dtype=float)
    times_s = np.linspace(0.0, dt_total_s, n_samples)

    r_cur = np.asarray(r0_km, dtype=float).reshape(3)
    v_cur = np.asarray(v0_km_s, dtype=float).reshape(3)
    positions_km[0] = r_cur

    prev_t_s = 0.0
    for idx in range(1, n_samples):
        dt_step_s = float(times_s[idx] - prev_t_s)
        try:
            r_cur, v_cur = kepler_propagate_universal(r_cur, v_cur, dt_step_s, mu_km3_s2)
        except Exception:
            # Fallback: split the step if a large increment causes solver instability.
            success = False
            for splits in (2, 4, 8, 16, 32, 64, 128):
                dt_sub_s = dt_step_s / float(splits)
                r_tmp = r_cur.copy()
                v_tmp = v_cur.copy()
                try:
                    for _ in range(splits):
                        r_tmp, v_tmp = kepler_propagate_universal(r_tmp, v_tmp, dt_sub_s, mu_km3_s2)
                    r_cur, v_cur = r_tmp, v_tmp
                    success = True
                    break
                except Exception:
                    continue
            if not success:
                raise

        positions_km[idx] = r_cur
        prev_t_s = float(times_s[idx])
    return positions_km


def _build_encounter_legend_lines(traj_record: Dict[str, Any]) -> List[str]:
    """Build per-encounter summary lines for the info panel."""
    t_et_s = np.asarray(traj_record["t_et_s"], dtype=float)
    body_ids = np.asarray(traj_record["body_ids"], dtype=np.int64)
    vinfD_km_s = np.asarray(traj_record["vinfD_km_s"], dtype=float)
    vinfA_km_s = None
    if "vinfA_km_s" in traj_record and traj_record["vinfA_km_s"] is not None:
        vinfA_km_s = np.asarray(traj_record["vinfA_km_s"], dtype=float)
    dv_patch_km_s = None
    if "dv_patch_km_s" in traj_record and traj_record["dv_patch_km_s"] is not None:
        dv_patch_km_s = np.asarray(traj_record["dv_patch_km_s"], dtype=float).reshape(-1)
    dv_lev_km_s = None
    if "dv_lev_km_s" in traj_record and traj_record["dv_lev_km_s"] is not None:
        dv_lev_km_s = np.asarray(traj_record["dv_lev_km_s"], dtype=float).reshape(-1)

    n_enc = int(t_et_s.size)
    n_legs = n_enc - 1

    lines: List[str] = ["Encounters", "-" * 22]
    for k in range(n_enc):
        body_id = int(body_ids[k])
        body_name = _resolve_central_body_label(str(body_id))
        date_str = spice.et2utc(float(t_et_s[k]), "ISOC", 0)[:10]
        lines.append(f"{k + 1}  {body_name}  ({date_str})")

        if k < n_legs:
            vinf_out = float(np.linalg.norm(vinfD_km_s[k]))
            lines.append(f"   Vinf_out:  {vinf_out:.3f} km/s")
        elif vinfA_km_s is not None:
            vinf_in = float(np.linalg.norm(vinfA_km_s[-1]))
            lines.append(f"   Vinf_in:   {vinf_in:.3f} km/s")

        if 0 < k < n_enc - 1 and dv_patch_km_s is not None:
            patch_idx = k - 1
            if patch_idx < dv_patch_km_s.size:
                dv_p_ms = float(dv_patch_km_s[patch_idx]) * 1000.0
                if dv_p_ms > 1e-3:
                    lines.append(f"   Flyby dV:  {dv_p_ms:.1f} m/s")

        if k < n_legs and dv_lev_km_s is not None and k < dv_lev_km_s.size:
            dv_l_ms = float(dv_lev_km_s[k]) * 1000.0
            if dv_l_ms > 1e-3:
                lines.append(f"   dV_lev:    {dv_l_ms:.1f} m/s")

        lines.append("")

    return lines


def plot_trajectory_2d(
    traj_record: Dict[str, Any],
    mu_central: float,
    frame: str = DEFAULT_FRAME,
    observer: str = DEFAULT_OBSERVER,
    central_body_label: str = "Sun",
    num_samples_leg: int = 200,
    num_samples_orbit: int = 400,
    axis: str = "xy",
    include_boundary_dv: Sequence[bool] | None = (False, False),
    closure_pos_tol_km: float = 1e5,
    closure_vel_tol_km_s: float = 1.0,
    dark: bool = False,
    save_path: Optional[str | Path] = None,
    show: bool = True,
):
    """Plot patched-conic central-body trajectory in x-y projection.

    Reconstruction model:
    - If DSM split metadata is available (`dv_lev_km_s`, `eta_lev`) and
      `dv_lev_km_s[k] > 0`, leg `k` is drawn as two propagated arcs split at
      `eta_lev[k]`. Split-arc endpoint velocities are taken from legacy
      `v0_lev_km_s`/`vf_lev_km_s` fields when present, otherwise reconstructed
      on demand from encounter body states and stored `vinfD_km_s`/`vinfA_km_s`.
    - Otherwise, fallback is one-arc ballistic reconstruction with
      `r0 = r_body(t0)`, `v0 = v_body(t0) + vinfD[k]`.
    """

    if str(axis).lower() != "xy":
        raise NotImplementedError("Only axis='xy' projection is supported in this utility.")

    t_et_s = np.asarray(traj_record["t_et_s"], dtype=float)
    body_ids = np.asarray(traj_record["body_ids"], dtype=np.int64)
    vinfD_km_s = np.asarray(traj_record["vinfD_km_s"], dtype=float)
    vinfA_km_s = None
    if "vinfA_km_s" in traj_record and traj_record["vinfA_km_s"] is not None:
        vinfA_km_s = np.asarray(traj_record["vinfA_km_s"], dtype=float)
    dv_lev_km_s = None
    eta_lev = None
    v0_lev_km_s = None
    vf_lev_km_s = None

    num_encounters = int(t_et_s.size)
    num_legs = num_encounters - 1
    if num_encounters < 2:
        raise ValueError("Trajectory record must contain at least 2 encounter epochs.")
    if body_ids.shape != (num_encounters,):
        raise ValueError("body_ids length must match t_et_s length.")
    if vinfD_km_s.shape != (num_legs, 3):
        raise ValueError("vinfD_km_s must have shape (nE-1, 3).")
    if vinfA_km_s is not None and vinfA_km_s.shape != (num_legs, 3):
        raise ValueError("vinfA_km_s must have shape (nE-1, 3) when present.")
    if "dv_lev_km_s" in traj_record and traj_record["dv_lev_km_s"] is not None:
        dv_lev_km_s = np.asarray(traj_record["dv_lev_km_s"], dtype=float).reshape(-1)
        if dv_lev_km_s.shape != (num_legs,):
            raise ValueError("dv_lev_km_s must have shape (nE-1,).")
    if "eta_lev" in traj_record and traj_record["eta_lev"] is not None:
        eta_lev = np.asarray(traj_record["eta_lev"], dtype=float).reshape(-1)
        if eta_lev.shape != (num_legs,):
            raise ValueError("eta_lev must have shape (nE-1,).")

    has_any_legacy_endpoint_field = any(
        key in traj_record and traj_record[key] is not None
        for key in ("v0_lev_km_s", "vf_lev_km_s")
    )
    has_all_legacy_endpoint_fields = all(
        key in traj_record and traj_record[key] is not None
        for key in ("v0_lev_km_s", "vf_lev_km_s")
    )
    if has_all_legacy_endpoint_fields:
        v0_lev_km_s = np.asarray(traj_record["v0_lev_km_s"], dtype=float).reshape(-1, 3)
        vf_lev_km_s = np.asarray(traj_record["vf_lev_km_s"], dtype=float).reshape(-1, 3)
        if v0_lev_km_s.shape != (num_legs, 3):
            raise ValueError("v0_lev_km_s must have shape (nE-1, 3).")
        if vf_lev_km_s.shape != (num_legs, 3):
            raise ValueError("vf_lev_km_s must have shape (nE-1, 3).")
    elif has_any_legacy_endpoint_field:
        warnings.warn(
            "Partial legacy DSM endpoint velocity metadata found in trajectory record; "
            "expected both v0_lev_km_s and vf_lev_km_s. Reconstructing on demand when possible."
        )

    if (dv_lev_km_s is None) != (eta_lev is None):
        warnings.warn(
            "Partial DSM split metadata found in trajectory record; expected both "
            "dv_lev_km_s and eta_lev. Falling back to ballistic plotting for leveraged legs."
        )
    elif (
        dv_lev_km_s is not None
        and eta_lev is not None
        and v0_lev_km_s is None
        and vf_lev_km_s is None
        and vinfA_km_s is None
        and np.any(np.asarray(dv_lev_km_s, dtype=float) > 1e-12)
    ):
        warnings.warn(
            "DSM split metadata found, but vinfA_km_s is unavailable to reconstruct leveraged "
            "arrival velocities. Falling back to ballistic plotting for leveraged legs."
        )

    traj_id = traj_record.get("traj_id", "unknown")
    dv_total_km_s = traj_record.get("dv_total_km_s", None)
    dv_display_km_s = None
    if dv_total_km_s is not None:
        dv_display_km_s, _, _ = _resolved_dv_for_record(traj_record, include_boundary_dv)
    tof_total_days = None
    if "tof_total_days" in traj_record and traj_record["tof_total_days"] is not None:
        tof_total_days = float(traj_record["tof_total_days"])
    elif "tof_total_s" in traj_record and traj_record["tof_total_s"] is not None:
        tof_total_days = float(traj_record["tof_total_s"]) / 86400.0

    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_axes([0.07, 0.08, 0.56, 0.84])
    default_dark_edge_color = "white" if bool(dark) else "black"

    # Plot one full orbital period for each unique body (dotted).
    t_min_et_s = float(np.min(t_et_s))
    t_max_et_s = float(np.max(t_et_s))
    unique_body_ids = list(dict.fromkeys(int(body_id) for body_id in body_ids.tolist()))

    body_color: Dict[int, Any] = {}
    for body_id in unique_body_ids:
        # Anchor orbit phase at this body's first encounter epoch in the trajectory if present.
        body_enc_indices = np.flatnonzero(body_ids == int(body_id))
        if body_enc_indices.size > 0:
            t_ref_et_s = float(t_et_s[int(body_enc_indices[0])])
        else:
            t_ref_et_s = t_min_et_s

        r_ref_km, v_ref_km_s = _spice_state(int(body_id), t_ref_et_s, observer=observer, frame=frame)
        r_ref_km = np.asarray(r_ref_km, dtype=float)
        v_ref_km_s = np.asarray(v_ref_km_s, dtype=float)
        period_s = estimate_orbital_period_s(r_ref_km, v_ref_km_s, float(mu_central))

        if body_id == 1001673:
            # Asteroid has a ~34-year period; SPK kernel doesn't cover that range.
            # Just plot t0 → tf instead.
            orbit_times_s = np.linspace(t_min_et_s, t_max_et_s, int(num_samples_orbit))
        elif np.isfinite(period_s) and period_s > 0.0:
            orbit_times_s = np.linspace(t_ref_et_s, t_ref_et_s + float(period_s), int(num_samples_orbit))
        else:
            warnings.warn(
                f"body={body_id} period estimate unavailable; using trajectory span for orbit plot."
            )
            orbit_times_s = np.linspace(t_min_et_s, t_max_et_s, int(num_samples_orbit))

        orbit_pos_km = np.empty((orbit_times_s.size, 3), dtype=float)
        for idx, et_s in enumerate(orbit_times_s):
            r_km, _ = _spice_state(int(body_id), float(et_s), observer=observer, frame=frame)
            orbit_pos_km[idx] = np.asarray(r_km, dtype=float)
        line_obj, = ax.plot(
            orbit_pos_km[:, 0],
            orbit_pos_km[:, 1],
            linestyle=":",
            linewidth=1.2,
            alpha=0.8,
            label=f"Body {body_id} orbit",
        )
        body_color[body_id] = line_obj.get_color()

        if body_id == 1001673:
            ast_z = orbit_pos_km[:, 2]
            for idx in np.where(np.diff(np.sign(ast_z)) != 0)[0]:
                z0, z1 = ast_z[idx], ast_z[idx + 1]
                t = -z0 / (z1 - z0)
                nx = orbit_pos_km[idx, 0] + t * (orbit_pos_km[idx + 1, 0] - orbit_pos_km[idx, 0])
                ny = orbit_pos_km[idx, 1] + t * (orbit_pos_km[idx + 1, 1] - orbit_pos_km[idx, 1])
                is_asc = z1 > z0
                marker = "^" if is_asc else "v"
                color  = "#ff8080" if is_asc else "#80c0ff"
                label  = "Asc. node" if is_asc else "Desc. node"
                existing_labels = [h.get_label() for h in ax.get_lines() + ax.get_legend_handles_labels()[0]]
                ax.plot(nx, ny, marker, color=color, markersize=8, zorder=6,
                        label=label if label not in existing_labels else "")
                ax.plot([0, nx], [0, ny], "--", color=color, lw=0.7, alpha=0.5)

    if dv_lev_km_s is not None:
        nonzero_dsm_idx = np.flatnonzero(np.asarray(dv_lev_km_s, dtype=float) > 1e-12)
        if nonzero_dsm_idx.size > 0:
            print(
                f"traj_id={traj_id} nonzero_dsm_legs={nonzero_dsm_idx.tolist()} "
                f"dv_lev_km_s={[float(dv_lev_km_s[idx]) for idx in nonzero_dsm_idx]}"
            )

    # Plot each leg from reconstructed state.
    for leg_idx in range(num_legs):
        t0_et_s = float(t_et_s[leg_idx])
        t1_et_s = float(t_et_s[leg_idx + 1])
        if t1_et_s <= t0_et_s:
            warnings.warn(f"traj_id={traj_id} leg={leg_idx} has non-positive duration; skipped.")
            continue

        dep_body_id = int(body_ids[leg_idx])
        arr_body_id = int(body_ids[leg_idx + 1])

        rB0_km, vB0_km_s = _spice_state(dep_body_id, t0_et_s, observer=observer, frame=frame)
        rB1_km, vB1_km_s = _spice_state(arr_body_id, t1_et_s, observer=observer, frame=frame)
        rB0_km = np.asarray(rB0_km, dtype=float)
        vB0_km_s = np.asarray(vB0_km_s, dtype=float)
        rB1_km = np.asarray(rB1_km, dtype=float)
        vB1_km_s = np.asarray(vB1_km_s, dtype=float)

        dt_total_s = float(t1_et_s - t0_et_s)
        split_drawn = False
        if (
            dv_lev_km_s is not None
            and eta_lev is not None
            and float(dv_lev_km_s[leg_idx]) > 1e-12
        ):
            eta = float(eta_lev[leg_idx])
            dt_0m_s = eta * dt_total_s
            dt_mf_s = (1.0 - eta) * dt_total_s
            if np.isfinite(eta) and dt_0m_s > 0.0 and dt_mf_s > 0.0:
                if v0_lev_km_s is not None and vf_lev_km_s is not None:
                    v0_leg_km_s = np.asarray(v0_lev_km_s[leg_idx], dtype=float)
                    vf_leg_km_s = np.asarray(vf_lev_km_s[leg_idx], dtype=float)
                elif vinfA_km_s is not None:
                    v0_leg_km_s, vf_leg_km_s = _reconstruct_leg_endpoint_velocities(
                        vB0_km_s,
                        vB1_km_s,
                        vinfD_km_s[leg_idx],
                        vinfA_km_s[leg_idx],
                    )
                else:
                    v0_leg_km_s = None
                    vf_leg_km_s = None

                if v0_leg_km_s is not None and vf_leg_km_s is not None:
                    n1 = max(20, int(num_samples_leg * max(eta, 0.1)))
                    n2 = max(20, int(num_samples_leg * max(1.0 - eta, 0.1)))

                    leg1_pos_km = sample_leg(
                        rB0_km,
                        v0_leg_km_s,
                        0.0,
                        dt_0m_s,
                        float(mu_central),
                        n1,
                    )
                    leg2_back_pos_km = sample_leg(
                        rB1_km,
                        -vf_leg_km_s,
                        0.0,
                        dt_mf_s,
                        float(mu_central),
                        n2,
                    )
                    leg2_pos_km = leg2_back_pos_km[::-1]
                    leg_pos_km = np.vstack((leg1_pos_km, leg2_pos_km[1:, :]))
                    ax.plot(
                        leg_pos_km[:, 0],
                        leg_pos_km[:, 1],
                        linestyle="-",
                        color=COLOR,
                        linewidth=LINEWIDTH,
                        label=f"SC leg {leg_idx}",
                    )

                    maneuver_km = leg1_pos_km[-1]
                    ax.scatter(
                        [float(maneuver_km[0])],
                        [float(maneuver_km[1])],
                        s=16,
                        marker="o",
                        color=default_dark_edge_color,
                        zorder=6,
                        label=f"DSM leg {leg_idx}",
                    )

                    r_m_fwd_km, v_m_fwd_km_s = kepler_propagate_universal(
                        rB0_km,
                        v0_leg_km_s,
                        dt_0m_s,
                        float(mu_central),
                    )
                    r_m_back_km, v_m_back_km_s = kepler_propagate_universal(
                        rB1_km,
                        -vf_leg_km_s,
                        dt_mf_s,
                        float(mu_central),
                    )
                    pos_gap_km = float(np.linalg.norm(r_m_fwd_km - r_m_back_km))
                    if pos_gap_km > closure_pos_tol_km:
                        warnings.warn(
                            f"traj_id={traj_id} leg={leg_idx} DSM midpoint position mismatch={pos_gap_km:.3e} km exceeds "
                            f"tol={float(closure_pos_tol_km):.3e} km."
                        )

                    v_m2_km_s = -np.asarray(v_m_back_km_s, dtype=float)
                    dv_mid_km_s = float(np.linalg.norm(np.asarray(v_m2_km_s) - np.asarray(v_m_fwd_km_s)))
                    dv_mid_err_km_s = abs(dv_mid_km_s - float(dv_lev_km_s[leg_idx]))
                    if dv_mid_err_km_s > float(closure_vel_tol_km_s):
                        warnings.warn(
                            f"traj_id={traj_id} leg={leg_idx} DSM midpoint DV mismatch={dv_mid_err_km_s:.3e} km/s exceeds "
                            f"tol={float(closure_vel_tol_km_s):.3e} km/s."
                        )
                    split_drawn = True
            else:
                warnings.warn(
                    f"traj_id={traj_id} leg={leg_idx} has invalid eta={eta}; "
                    "falling back to one-arc ballistic plotting."
                )

        if split_drawn:
            continue

        r0_km = rB0_km
        if v0_lev_km_s is not None:
            v0_km_s = np.asarray(v0_lev_km_s[leg_idx], dtype=float)
        else:
            v0_km_s = vB0_km_s + np.asarray(vinfD_km_s[leg_idx], dtype=float)

        leg_pos_km = sample_leg(
            r0_km,
            v0_km_s,
            t0_et_s,
            t1_et_s,
            float(mu_central),
            int(num_samples_leg),
        )
        ax.plot(
            leg_pos_km[:, 0],
            leg_pos_km[:, 1],
            linestyle="-",
            color=COLOR,
            linewidth=LINEWIDTH,
            label=f"SC leg {leg_idx}",
        )

        r_end_km, v_end_km_s = kepler_propagate_universal(r0_km, v0_km_s, dt_total_s, float(mu_central))
        pos_err_km = float(np.linalg.norm(r_end_km - rB1_km))
        if pos_err_km > closure_pos_tol_km:
            warnings.warn(
                f"traj_id={traj_id} leg={leg_idx} closure position error={pos_err_km:.3e} km exceeds "
                f"tol={float(closure_pos_tol_km):.3e} km."
            )

        if vf_lev_km_s is not None:
            v_expected_km_s = np.asarray(vf_lev_km_s[leg_idx], dtype=float)
        elif vinfA_km_s is not None:
            v_expected_km_s = vB1_km_s + np.asarray(vinfA_km_s[leg_idx], dtype=float)
        else:
            v_expected_km_s = None
        if v_expected_km_s is not None:
            vel_err_km_s = float(np.linalg.norm(v_end_km_s - v_expected_km_s))
            if vel_err_km_s > closure_vel_tol_km_s:
                warnings.warn(
                    f"traj_id={traj_id} leg={leg_idx} closure velocity error={vel_err_km_s:.3e} km/s exceeds "
                    f"tol={float(closure_vel_tol_km_s):.3e} km/s."
                )

    # Encounter markers.
    for enc_idx in range(num_encounters):
        body_id = int(body_ids[enc_idx])
        et_s = float(t_et_s[enc_idx])
        r_km, _ = _spice_state(body_id, et_s, observer=observer, frame=frame)
        r_km = np.asarray(r_km, dtype=float)
        ax.scatter(
            [r_km[0]],
            [r_km[1]],
            s=40,
            marker="o",
            color=body_color.get(body_id, default_dark_edge_color),
            zorder=5,
            label=f"Enc {enc_idx} body {body_id}",
        )
        ax.annotate(
            str(enc_idx + 1),
            xy=(r_km[0], r_km[1]),
            xytext=(4, 4),
            textcoords="offset points",
            fontsize=8,
            fontweight="bold",
            color="white" if bool(dark) else "black",
            zorder=6,
        )

    # Central body marker at observer-frame origin.
    center_label = str(central_body_label).strip() or str(observer)
    ax.scatter(
        [0.0],
        [0.0],
        s=120,
        marker="*",
        color="gold",
        edgecolor=default_dark_edge_color,
        zorder=10,
        label=center_label,
    )

    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel(f"x [{frame}] km")
    ax.set_ylabel(f"y [{frame}] km")
    ax.grid(True, alpha=0.3)

    title = f"Trajectory {traj_id}"
    subtitle_parts = []
    if dv_display_km_s is not None:
        subtitle_parts.append(f"DV={float(dv_display_km_s):.4f} km/s")
    if tof_total_days is not None:
        subtitle_parts.append(f"TOF={float(tof_total_days):.2f} days")
    if subtitle_parts:
        title += " | " + ", ".join(subtitle_parts)
    ax.set_title(title)

    enc_lines = _build_encounter_legend_lines(traj_record)
    enc_text_artist = fig.text(
        0.64, 0.92,
        "\n".join(enc_lines),
        va="top",
        ha="left",
        fontsize=8,
        fontfamily="monospace",
        transform=fig.transFigure,
    )
    _apply_plot_theme(fig, ax, dark=dark, extra_text_artists=(enc_text_artist,))

    if save_path is not None:
        output_path = Path(save_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=180, bbox_inches="tight")

    if show:
        plt.show()

    return fig, ax


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Plot trajectory trade spaces and click points to open 2D trajectories.")
    parser.add_argument("--problem", default=None, help="Path to solutions JSONL file.")
    parser.add_argument("--max_rows", type=int, default=50_000, help="Max JSONL rows to read.")
    parser.add_argument(
        "--diversity",
        type=bool, 
        default=False,
        help=(
            "Heuristic diversity mode (default: False). When enabled, stream all rows, group by "
            "body sequence, and keep up to 500 Pareto (TOF-DV) records per sequence."
        ),
    )
    parser.add_argument("--metakernel", default="star/METAKERN.tm", help="SPICE meta-kernel path.")
    parser.add_argument("--frame", default=DEFAULT_FRAME, help="Inertial frame (default: ECLIPJ2000).")
    parser.add_argument("--observer", default=DEFAULT_OBSERVER, help="Observer/central body (default: SUN).")
    parser.add_argument("--axis", default="xy", choices=["xy"], help="2D projection plane (currently only xy).")
    parser.add_argument("--num_samples_leg", type=int, default=200, help="Samples per spacecraft leg.")
    parser.add_argument("--num_samples_orbit", type=int, default=400, help="Samples per body orbit.")
    parser.add_argument("--x", dest="x_metric", default="tof", choices=PLOT_AXIS_METRICS, help="Metric for the x-axis.")
    parser.add_argument("--y", dest="y_metric", default="dv", choices=PLOT_AXIS_METRICS, help="Metric for the y-axis.")
    parser.add_argument(
        "--color",
        dest="color_metric",
        default="none",
        choices=PLOT_COLOR_METRICS,
        help="Optional metric for point coloring.",
    )
    parser.add_argument(
        "--enc",
        type=int,
        default=0,
        help=(
            "Encounter index (0-based) used when --x t, --y t, --color t, --color vinf_arr, or "
            "--color inc_arr is selected. For time metrics, --enc 0 matches t0. Arrival-state "
            "metrics still require --enc >= 1 because enc=0 has no arrival v-infinity."
        ),
    )
    parser.add_argument(
        "--include_boundary_dv",
        default="[False, False]",
        help=(
            "Boolean list [include_escape, include_insertion] for displayed/Pareto DV "
            "(e.g. [True, False])."
        ),
    )
    parser.add_argument(
        "--dark",
        nargs="?",
        const=True,
        default=False,
        type=_parse_bool_cli_arg,
        help="Enable dark styling (for example: --dark True). Default: False.",
    )
    parser.add_argument("--closure_pos_tol_km", type=float, default=1e5, help="Closure position warning tolerance [km].")
    parser.add_argument(
        "--closure_vel_tol_km_s",
        type=float,
        default=1.0,
        help="Closure velocity warning tolerance [km/s].",
    )
    parser.add_argument("--out", default=None, help="Optional output image path (e.g., fig.png).")

    return parser


def _parse_include_boundary_dv_arg(raw_value: str) -> Tuple[bool, bool]:
    """Parse CLI include-boundary flags from a strict boolean list literal."""

    try:
        parsed = ast.literal_eval(str(raw_value))
    except (SyntaxError, ValueError) as exc:
        raise ValueError(
            "--include_boundary_dv must be a boolean list literal, e.g. [True, False]."
        ) from exc

    if not isinstance(parsed, list) or len(parsed) != 2:
        raise ValueError("--include_boundary_dv must be a 2-element list, e.g. [True, False].")
    if not isinstance(parsed[0], bool) or not isinstance(parsed[1], bool):
        raise ValueError("--include_boundary_dv list entries must be booleans.")
    return parsed[0], parsed[1]


def _load_problem_module(problem_name: str):
    """Load example/{problem_name}.py and return the module, or None if not found."""
    example_path = root_folder / "example" / f"{problem_name}.py"
    if not example_path.exists():
        return None
    spec = importlib.util.spec_from_file_location(problem_name, example_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> None:
    args = _build_arg_parser().parse_args()
    include_boundary_dv = _parse_include_boundary_dv_arg(args.include_boundary_dv)
    diversity_enabled = args.diversity

    _load_spice_metakernel(args.metakernel)

    # Override frame/observer from the example file if it defines them.
    # Must happen after metakernel load since example files may call SPICE at module level.
    if args.problem is not None:
        problem_module = _load_problem_module(args.problem)
        if problem_module is not None:
            if hasattr(problem_module, "observer"):
                args.observer = str(problem_module.observer)
            if hasattr(problem_module, "frame"):
                args.frame = str(problem_module.frame)

    data_file = root_folder / "output" / Path(args.problem + ".jsonl")
    if bool(diversity_enabled):
        records, diversity_stats = read_jsonl_trajectories_diverse(
            data_file,
            include_boundary_dv=include_boundary_dv,
            max_rows_per_body_sequence=500,
        )
        print(
            "diversity mode: "
            f"rows_read={int(diversity_stats['num_rows_read'])}, "
            f"body_sequences={int(diversity_stats['num_body_sequences'])}, "
            f"records_out={int(diversity_stats['num_records_out'])}, "
            f"invalid={int(diversity_stats['num_rows_invalid'])}, "
            f"missing_body={int(diversity_stats['num_rows_missing_body'])}, "
            f"per_sequence_cap={int(diversity_stats['max_rows_per_body_sequence'])}"
        )
    else:
        records = read_jsonl_trajectories(data_file, max_rows=args.max_rows)
    mu_central_km3_s2 = _resolve_central_mu_km3_s2(args.observer)
    central_body_label = _resolve_central_body_label(args.observer)

    pareto_fig, pareto_ax, _, pareto_plot_data = plot_trajectory_metric_scatter(
        records=records,
        x_metric=args.x_metric,
        y_metric=args.y_metric,
        color_metric=args.color_metric,
        dark=bool(args.dark),
        enc=args.enc,
        observer=args.observer,
        frame=args.frame,
        include_boundary_dv=include_boundary_dv,
        show_all_points=True,
        save_path=args.out,
        show=False,
        interactive_pick=True,
        return_plot_data=True,
    )

    _connect_trade_plot_pick_handler(
        pareto_fig=pareto_fig,
        pareto_ax=pareto_ax,
        records=records,
        pareto_plot_data=pareto_plot_data,
        mu_central=mu_central_km3_s2,
        central_body_label=central_body_label,
        frame=args.frame,
        observer=args.observer,
        num_samples_leg=int(args.num_samples_leg),
        num_samples_orbit=int(args.num_samples_orbit),
        axis=args.axis,
        include_boundary_dv=include_boundary_dv,
        closure_pos_tol_km=float(args.closure_pos_tol_km),
        closure_vel_tol_km_s=float(args.closure_vel_tol_km_s),
        dark=bool(args.dark),
        out_path_template=args.out,
    )

    print(
        "click-to-plot enabled: click a plotted point"
        + (" or Pareto point" if args.x_metric == "tof" and args.y_metric == "dv" else "")
        + " to open trajectory plots."
    )
    plt.show()


if __name__ == "__main__":
    main()

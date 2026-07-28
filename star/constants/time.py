"""Time conversion helpers with ET seconds as internal representation."""

from __future__ import annotations

from . import J2000_JD_TDB, SECONDS_PER_DAY

try:
    import spiceypy as spice
except Exception:  # pragma: no cover - exercised through runtime checks
    spice = None


def j2000_days_to_et_seconds(j2000_days: float) -> float:
    """Convert J2000 offset in days to ET seconds past J2000.

    Args:
        j2000_days: Time past J2000 [days].

    Returns:
        ET seconds past J2000 [seconds].
    """

    return float(j2000_days) * SECONDS_PER_DAY


def et_seconds_to_j2000_days(et_seconds: float) -> float:
    """Convert ET seconds past J2000 to J2000 offset in days.

    Args:
        et_seconds: ET seconds past J2000 [seconds].

    Returns:
        Time past J2000 [days].
    """

    return float(et_seconds) / SECONDS_PER_DAY


def jd_tdb_to_et_seconds(jd_tdb: float) -> float:
    """Convert TDB Julian Date to ET seconds past J2000.

    Formula:
        `JD_TDB = 2451545.0 + et_seconds/86400`.

    Args:
        jd_tdb: Julian Date in TDB scale [days].

    Returns:
        ET seconds past J2000 [seconds].
    """

    return (float(jd_tdb) - J2000_JD_TDB) * SECONDS_PER_DAY


def et_seconds_to_jd_tdb(et_seconds: float) -> float:
    """Convert ET seconds past J2000 to TDB Julian Date.

    Args:
        et_seconds: ET seconds past J2000 [seconds].

    Returns:
        TDB Julian Date [days].
    """

    return J2000_JD_TDB + float(et_seconds) / SECONDS_PER_DAY


def utc_string_to_et_seconds(utc_string: str) -> float:
    """Convert UTC string to ET seconds past J2000 using SPICE.

    Args:
        utc_string: UTC timestamp accepted by SPICE, e.g. `"2026-01-01T00:00:00"`.

    Returns:
        ET seconds past J2000 [seconds].

    Raises:
        NotImplementedError: If `spiceypy` is not installed.
    """

    if spice is None:
        raise NotImplementedError("UTC <-> ET conversion requires spiceypy and loaded LSK kernels.")
    return float(spice.str2et(utc_string))


def et_seconds_to_utc_string(et_seconds: float, precision_digits: int = 3) -> str:
    """Convert ET seconds past J2000 to UTC string using SPICE.

    Args:
        et_seconds: ET seconds past J2000 [seconds].
        precision_digits: Decimal places for seconds in output [digits].

    Returns:
        UTC string in ISO calendar format.

    Raises:
        NotImplementedError: If `spiceypy` is not installed.
    """

    if spice is None:
        raise NotImplementedError("UTC <-> ET conversion requires spiceypy and loaded LSK kernels.")
    return str(spice.et2utc(float(et_seconds), "ISOC", int(precision_digits)))

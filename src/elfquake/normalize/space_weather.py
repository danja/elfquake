"""Archive normalization for space-weather sources."""

from __future__ import annotations

import csv
import json
from datetime import timezone
from pathlib import Path

from elfquake.features.common import parse_utc


GFZ_FIELDNAMES = ["date", "slot", "kp", "ap", "source_file"]
DST_FIELDNAMES = ["date", "hour", "dst_nt", "dst_tier", "source_file"]
# Spaceweather Canada publishes three Penticton observations per day (17, 20
# and 23 UT). The 20 UT (local noon) `fluxadjflux` reading is the conventional
# daily F10.7, so the observation timestamp is preserved rather than collapsed.
F107_FIELDNAMES = ["date", "time_utc", "f107", "f107_observed", "source_file"]
GOES_FIELDNAMES = ["time_utc", "variable", "value", "units", "source_file"]

# Kyoto's monthly index pages are fixed-width, not whitespace-delimited: a
# missing hour is the sentinel `9999`, and consecutive missing hours run
# together with no separating space (`-19999999999999`). Splitting on
# whitespace silently merges hours, so the hourly table is sliced by column.
DST_MONTH_NAMES = {
    "JANUARY": 1,
    "FEBRUARY": 2,
    "MARCH": 3,
    "APRIL": 4,
    "MAY": 5,
    "JUNE": 6,
    "JULY": 7,
    "AUGUST": 8,
    "SEPTEMBER": 9,
    "OCTOBER": 10,
    "NOVEMBER": 11,
    "DECEMBER": 12,
}
DST_DAY_WIDTH = 3
DST_VALUE_WIDTH = 4
DST_GROUP_SIZE = 8
DST_MISSING = "9999"


def normalize_gfz_kp_ap(raw_path: Path, out_path: Path) -> int:
    rows = []
    for line in raw_path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        parts = stripped.split()
        if len(parts) < 9:
            continue
        date = _date_from_parts(parts[0], parts[1], parts[2])
        rows.append(
            {
                "date": date,
                "slot": str(int(float(parts[3]) // 3)),
                "kp": parts[7],
                "ap": parts[8],
                "source_file": str(raw_path),
            }
        )
    _write_rows(out_path, GFZ_FIELDNAMES, rows)
    return len(rows)


def normalize_kyoto_dst_text(raw_path: Path, out_path: Path) -> int:
    # Kyoto ships the monthly pages as EUC-JP; the numeric table is ASCII, so
    # decode permissively rather than failing on the header boilerplate.
    text = raw_path.read_text(encoding="utf-8", errors="replace")
    rows = _normalize_kyoto_dst_page(text, raw_path)
    if not rows:
        rows = _normalize_kyoto_dst_columns(text, raw_path)
    _write_rows(out_path, DST_FIELDNAMES, rows)
    return len(rows)


def _normalize_kyoto_dst_page(text: str, raw_path: Path) -> list[dict[str, str]]:
    """Parse a Kyoto monthly index page by column position."""
    lines = text.splitlines()
    year_month = _dst_page_year_month(lines)
    if year_month is None:
        return []
    year, month = year_month
    tier = _dst_page_tier(lines)
    rows = []
    for line in lines:
        day = _dst_day_number(line)
        if day is None:
            continue
        for hour in range(24):
            offset = (
                DST_DAY_WIDTH
                + hour * DST_VALUE_WIDTH
                + hour // DST_GROUP_SIZE  # one extra space between column groups
            )
            value = line[offset : offset + DST_VALUE_WIDTH].strip()
            if not value or value == DST_MISSING or not _looks_like_number(value):
                continue
            rows.append(
                {
                    "date": f"{year:04d}-{month:02d}-{day:02d}",
                    "hour": str(hour),
                    "dst_nt": value,
                    "dst_tier": tier,
                    "source_file": str(raw_path),
                }
            )
    return rows


def _normalize_kyoto_dst_columns(text: str, raw_path: Path) -> list[dict[str, str]]:
    """Parse the whitespace-delimited `year month day h1..h24` variant."""
    rows = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        parts = stripped.split()
        if len(parts) < 4 or not all(part.isdigit() for part in parts[:3]):
            continue
        date = _date_from_parts(parts[0], parts[1], parts[2])
        for hour, value in enumerate(parts[3:27]):
            rows.append(
                {
                    "date": date,
                    "hour": str(hour),
                    "dst_nt": value,
                    "dst_tier": "",
                    "source_file": str(raw_path),
                }
            )
    return rows


def _dst_page_year_month(lines: list[str]) -> tuple[int, int] | None:
    for line in lines:
        parts = line.split()
        if len(parts) != 2:
            continue
        month = DST_MONTH_NAMES.get(parts[0].upper())
        if month and len(parts[1]) == 4 and parts[1].isdigit():
            return int(parts[1]), month
    return None


def _dst_page_tier(lines: list[str]) -> str:
    for line in lines:
        if "Hourly Equatorial Dst Values" not in line:
            continue
        _, _, remainder = line.partition("(")
        label = "".join(character for character in remainder if character.isalpha())
        if label:
            return label.lower()
        return "final"
    return ""


def _dst_day_number(line: str) -> int | None:
    """Return the day of month a table row starts with, or None."""
    head = line[:DST_DAY_WIDTH]
    if len(line) <= DST_DAY_WIDTH or not head.strip().isdigit():
        return None
    if head[-1] != " ":
        return None
    day = int(head.strip())
    return day if 1 <= day <= 31 else None


def normalize_f107_daily(raw_path: Path, out_path: Path) -> int:
    text = raw_path.read_text(encoding="utf-8")
    rows = _normalize_f107_json(text, raw_path)
    if not rows:
        rows = _normalize_f107_text(text, raw_path)
    _write_rows(out_path, F107_FIELDNAMES, rows)
    return len(rows)


def normalize_goes_xrs_netcdf(
    raw_path: Path,
    out_path: Path,
    max_rows: int | None = None,
    start_utc: str | None = None,
    end_utc: str | None = None,
) -> int:
    try:
        from netCDF4 import Dataset, num2date
    except ImportError as error:
        raise RuntimeError("GOES XRS NetCDF extraction requires netCDF4") from error

    rows = []
    start = parse_utc(start_utc) if start_utc else None
    end = parse_utc(end_utc) if end_utc else None
    with Dataset(raw_path) as dataset:
        time_variable = _find_time_variable(dataset.variables)
        if time_variable is None:
            raise ValueError("NetCDF file has no recognizable time coordinate")
        time_values = time_variable[:]
        time_units = getattr(time_variable, "units", "")
        calendar = getattr(time_variable, "calendar", "standard")
        times = [
            _format_netcdf_time(value)
            for value in num2date(time_values, units=time_units, calendar=calendar)
        ]
        time_dimension = time_variable.dimensions[0] if time_variable.dimensions else "time"
        for name, variable in dataset.variables.items():
            if name == time_variable.name or not _is_goes_xrs_variable(name, variable):
                continue
            if time_dimension not in variable.dimensions:
                continue
            time_axis = variable.dimensions.index(time_dimension)
            units = str(getattr(variable, "units", ""))
            values = variable[:]
            for index, time_utc in enumerate(times):
                time_dt = parse_utc(time_utc)
                if start and time_dt < start:
                    continue
                if end and time_dt >= end:
                    continue
                if max_rows is not None and len(rows) >= max_rows:
                    _write_rows(out_path, GOES_FIELDNAMES, rows)
                    return len(rows)
                value = values.take(index, axis=time_axis)
                scalar = _first_scalar(value)
                if scalar is None:
                    continue
                rows.append(
                    {
                        "time_utc": time_utc,
                        "variable": name,
                        "value": scalar,
                        "units": units,
                        "source_file": str(raw_path),
                    }
                )
    _write_rows(out_path, GOES_FIELDNAMES, rows)
    return len(rows)


def write_goes_xrs_netcdf_stub(raw_path: Path, out_path: Path) -> int:
    return normalize_goes_xrs_netcdf(raw_path, out_path)


def _normalize_f107_json(text: str, raw_path: Path) -> list[dict[str, str]]:
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        return []
    if not isinstance(payload, list):
        return []
    rows = []
    for item in payload:
        if not isinstance(item, dict):
            continue
        date = str(item.get("date") or item.get("time-tag") or "")
        value = item.get("f10.7")
        if date and value is not None:
            rows.append(
                {
                    "date": date,
                    "time_utc": "",
                    "f107": str(value),
                    "f107_observed": "",
                    "source_file": str(raw_path),
                }
            )
    return rows


def _normalize_f107_text(text: str, raw_path: Path) -> list[dict[str, str]]:
    rows = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        parts = stripped.split()
        if len(parts) >= 6 and parts[0].isdigit() and len(parts[0]) == 8:
            date = _compact_date(parts[0])
            rows.append(
                {
                    "date": date,
                    "time_utc": _f107_time_utc(date, parts[1]),
                    "f107": str(float(parts[5])),
                    "f107_observed": str(float(parts[4])),
                    "source_file": str(raw_path),
                }
            )
            continue
        if len(parts) >= 2 and _looks_like_date(parts[0]) and _looks_like_number(parts[1]):
            rows.append(
                {
                    "date": parts[0],
                    "time_utc": "",
                    "f107": parts[1],
                    "f107_observed": "",
                    "source_file": str(raw_path),
                }
            )
    return rows


def _f107_time_utc(date: str, flux_time: str) -> str:
    if not flux_time.isdigit() or len(flux_time) != 6:
        return ""
    return f"{date}T{flux_time[0:2]}:{flux_time[2:4]}:{flux_time[4:6]}Z"


def _date_from_parts(year: str, month: str, day: str) -> str:
    return f"{int(year):04d}-{int(month):02d}-{int(day):02d}"


def _compact_date(value: str) -> str:
    return f"{value[0:4]}-{value[4:6]}-{value[6:8]}"


def _looks_like_date(value: str) -> bool:
    if len(value) == 10 and value[4] == "-" and value[7] == "-":
        return True
    return value.isdigit() and len(value) == 8


def _looks_like_number(value: str) -> bool:
    try:
        float(value)
    except ValueError:
        return False
    return True


def _find_time_variable(variables) -> object | None:
    for name in ("time", "time_tag", "time_seconds"):
        if name in variables:
            return variables[name]
    for variable in variables.values():
        units = str(getattr(variable, "units", "")).lower()
        if " since " in units:
            return variable
    return None


def _is_goes_xrs_variable(name: str, variable) -> bool:
    if not getattr(variable, "dimensions", ()):
        return False
    lower_name = name.lower()
    units = str(getattr(variable, "units", "")).lower()
    return (
        "xrs" in lower_name
        or "flux" in lower_name
        or "irradiance" in lower_name
        or units in {"w/m^2", "w m-2", "w/m2"}
    )


def _format_netcdf_time(value) -> str:
    if hasattr(value, "tzinfo") and value.tzinfo is not None:
        return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")
    return (
        f"{int(value.year):04d}-{int(value.month):02d}-{int(value.day):02d}"
        f"T{int(value.hour):02d}:{int(value.minute):02d}:{int(value.second):02d}Z"
    )


def _first_scalar(value) -> str | None:
    if hasattr(value, "mask") and bool(getattr(value, "mask", False)):
        return None
    if hasattr(value, "filled"):
        value = value.filled(None)
    if hasattr(value, "reshape"):
        flattened = value.reshape(-1)
        if len(flattened) == 0:
            return None
        value = flattened[0]
    if hasattr(value, "item"):
        value = value.item()
    if value is None:
        return None
    return f"{value:g}" if isinstance(value, float) else str(value)


def _write_rows(out_path: Path, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)

"""Archive normalization stubs for space-weather sources."""

from __future__ import annotations

import csv
import json
from pathlib import Path


GFZ_FIELDNAMES = ["date", "slot", "kp", "ap", "source_file"]
DST_FIELDNAMES = ["date", "hour", "dst_nt", "source_file"]
F107_FIELDNAMES = ["date", "f107", "source_file"]
GOES_FIELDNAMES = ["source_file", "status", "note"]


def normalize_gfz_kp_ap(raw_path: Path, out_path: Path) -> int:
    rows = []
    for line in raw_path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        parts = stripped.split()
        if len(parts) < 5:
            continue
        date = _date_from_parts(parts[0], parts[1], parts[2])
        values = parts[3:]
        for index in range(0, min(len(values), 16), 2):
            if index + 1 >= len(values):
                break
            rows.append(
                {
                    "date": date,
                    "slot": str(index // 2),
                    "kp": values[index],
                    "ap": values[index + 1],
                    "source_file": str(raw_path),
                }
            )
    _write_rows(out_path, GFZ_FIELDNAMES, rows)
    return len(rows)


def normalize_kyoto_dst_text(raw_path: Path, out_path: Path) -> int:
    rows = []
    for line in raw_path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        parts = stripped.split()
        if len(parts) < 4:
            continue
        date = _date_from_parts(parts[0], parts[1], parts[2])
        for hour, value in enumerate(parts[3:27]):
            rows.append(
                {
                    "date": date,
                    "hour": str(hour),
                    "dst_nt": value,
                    "source_file": str(raw_path),
                }
            )
    _write_rows(out_path, DST_FIELDNAMES, rows)
    return len(rows)


def normalize_f107_daily(raw_path: Path, out_path: Path) -> int:
    text = raw_path.read_text(encoding="utf-8")
    rows = _normalize_f107_json(text, raw_path)
    if not rows:
        rows = _normalize_f107_text(text, raw_path)
    _write_rows(out_path, F107_FIELDNAMES, rows)
    return len(rows)


def write_goes_xrs_netcdf_stub(raw_path: Path, out_path: Path) -> int:
    rows = [
        {
            "source_file": str(raw_path),
            "status": "requires_netcdf_decoder",
            "note": "Install a NetCDF reader before extracting GOES XRS values.",
        }
    ]
    _write_rows(out_path, GOES_FIELDNAMES, rows)
    return 1


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
            rows.append({"date": date, "f107": str(value), "source_file": str(raw_path)})
    return rows


def _normalize_f107_text(text: str, raw_path: Path) -> list[dict[str, str]]:
    rows = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        parts = stripped.split()
        if len(parts) < 2:
            continue
        rows.append({"date": parts[0], "f107": parts[1], "source_file": str(raw_path)})
    return rows


def _date_from_parts(year: str, month: str, day: str) -> str:
    return f"{int(year):04d}-{int(month):02d}-{int(day):02d}"


def _write_rows(out_path: Path, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)

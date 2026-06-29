"""Coarse astronomy feature extraction."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Iterable

from elfquake.features.common import format_utc, parse_utc


FIELDNAMES = [
    "window_start_utc",
    "window_end_utc",
    "astro_capture_count",
    "astro_sources",
    "astro_latest_capture_utc",
    "astro_usno_next_phase",
    "astro_usno_next_phase_utc",
    "astro_noaa_solar_cycle_f107_month",
    "astro_noaa_solar_cycle_f107_value",
    "quality_missing_astro",
]


def build_astronomy_features(
    *,
    metadata_paths: Iterable[Path],
    window_start_utc: str,
    window_end_utc: str,
    out_path: Path,
) -> dict[str, str]:
    window_start = parse_utc(window_start_utc)
    window_end = parse_utc(window_end_utc)
    paths = list(metadata_paths)
    captures = []
    for metadata_path in paths:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        captured_at = metadata.get("captured_at_utc", "")
        if not captured_at:
            continue
        captured_dt = parse_utc(captured_at)
        if window_start <= captured_dt < window_end:
            captures.append((metadata, captured_dt))

    latest = max(captures, key=lambda item: item[1], default=None)
    moon_phase, moon_phase_utc = _next_moon_phase(paths, after_utc=window_end)
    f107_month, f107_value = _f107_month_value(paths, window_end_utc=window_end_utc)
    row = {
        "window_start_utc": window_start_utc,
        "window_end_utc": window_end_utc,
        "astro_capture_count": str(len(captures)),
        "astro_sources": ";".join(sorted({item[0].get("source_id", "") for item in captures})),
        "astro_latest_capture_utc": latest[0].get("captured_at_utc", "") if latest else "",
        "astro_usno_next_phase": moon_phase,
        "astro_usno_next_phase_utc": moon_phase_utc,
        "astro_noaa_solar_cycle_f107_month": f107_month,
        "astro_noaa_solar_cycle_f107_value": f107_value,
        "quality_missing_astro": "0" if captures else "1",
    }
    _write_one_row(out_path, row)
    return row


def _next_moon_phase(paths: Iterable[Path], *, after_utc) -> tuple[str, str]:
    for metadata_path in paths:
        if "usno_moon_phases" not in metadata_path.name:
            continue
        payload_path = Path(str(metadata_path).removesuffix(".metadata.json"))
        payload = json.loads(payload_path.read_text(encoding="utf-8"))
        for phase in payload.get("phasedata", []):
            phase_time = parse_utc(
                f"{int(phase['year']):04d}-{int(phase['month']):02d}-{int(phase['day']):02d}T{phase['time']}:00Z"
            )
            if phase_time >= after_utc:
                return str(phase.get("phase", "")), format_utc(phase_time)
    return "", ""


def _f107_month_value(paths: Iterable[Path], *, window_end_utc: str) -> tuple[str, str]:
    window_month = window_end_utc[:7]
    for metadata_path in paths:
        if "noaa_solar_cycle_f107" not in metadata_path.name:
            continue
        payload_path = Path(str(metadata_path).removesuffix(".metadata.json"))
        payload = json.loads(payload_path.read_text(encoding="utf-8"))
        latest_month = ""
        latest_value = ""
        for item in payload:
            item_month = item.get("time-tag", "")
            if item_month <= window_month:
                latest_month = item_month
                latest_value = str(item.get("f10.7", ""))
        if latest_month:
            return latest_month, latest_value
    return window_month, ""


def _write_one_row(out_path: Path, row: dict[str, str]) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES, lineterminator="\n")
        writer.writeheader()
        writer.writerow(row)

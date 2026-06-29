"""Build a first multimodal smoke row from existing captures."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable


FIELDNAMES = [
    "window_id",
    "region_id",
    "window_start_utc",
    "window_end_utc",
    "lookback_start_utc",
    "target_start_utc",
    "target_end_utc",
    "target_magnitude_min",
    "target_event_count",
    "target_occurred",
    "target_status",
    "seismic_event_count",
    "seismic_max_magnitude",
    "seismic_source_file",
    "vlf_capture_count",
    "vlf_latest_source_id",
    "vlf_latest_capture_utc",
    "vlf_latest_last_modified_utc",
    "vlf_latest_path",
    "astro_capture_count",
    "astro_sources",
    "astro_latest_capture_utc",
    "astro_usno_next_phase",
    "astro_usno_next_phase_utc",
    "astro_noaa_solar_cycle_f107_month",
    "astro_noaa_solar_cycle_f107_value",
    "quality_missing_vlf",
    "quality_missing_astro",
    "quality_notes",
]


@dataclass(frozen=True)
class CaptureSummary:
    source_id: str
    captured_at_utc: str
    payload_path: str
    last_modified_utc: str = ""


def build_multimodal_smoke_row(
    *,
    events_csv: Path,
    vlf_metadata_paths: Iterable[Path],
    astronomy_metadata_paths: Iterable[Path],
    region_id: str,
    window_start_utc: str,
    window_end_utc: str,
    target_end_utc: str,
    out_path: Path,
    target_magnitude_min: str = "3.0",
) -> dict[str, str]:
    """Write one unlabeled live smoke row and return it."""

    window_start = parse_utc(window_start_utc)
    window_end = parse_utc(window_end_utc)
    events = _read_events(events_csv, window_start, window_end)
    vlf_captures = _read_capture_metadata(vlf_metadata_paths, before_utc=window_end)
    astro_captures = _read_capture_metadata(astronomy_metadata_paths, before_utc=window_end)

    latest_vlf = _latest_capture(vlf_captures)
    latest_astro = _latest_capture(astro_captures)

    moon_phase, moon_phase_utc = _next_moon_phase(astronomy_metadata_paths, after_utc=window_end)
    f107_month, f107_value = _f107_month_value(astronomy_metadata_paths, window_end=window_end)

    row = {
        "window_id": _window_id(region_id, window_start_utc, window_end_utc),
        "region_id": region_id,
        "window_start_utc": window_start_utc,
        "window_end_utc": window_end_utc,
        "lookback_start_utc": window_start_utc,
        "target_start_utc": window_end_utc,
        "target_end_utc": target_end_utc,
        "target_magnitude_min": target_magnitude_min,
        "target_event_count": "",
        "target_occurred": "",
        "target_status": "unlabeled_pending_future_events",
        "seismic_event_count": str(len(events)),
        "seismic_max_magnitude": _max_magnitude(events),
        "seismic_source_file": str(events_csv),
        "vlf_capture_count": str(len(vlf_captures)),
        "vlf_latest_source_id": latest_vlf.source_id if latest_vlf else "",
        "vlf_latest_capture_utc": latest_vlf.captured_at_utc if latest_vlf else "",
        "vlf_latest_last_modified_utc": latest_vlf.last_modified_utc if latest_vlf else "",
        "vlf_latest_path": latest_vlf.payload_path if latest_vlf else "",
        "astro_capture_count": str(len(astro_captures)),
        "astro_sources": ";".join(sorted({capture.source_id for capture in astro_captures})),
        "astro_latest_capture_utc": latest_astro.captured_at_utc if latest_astro else "",
        "astro_usno_next_phase": moon_phase,
        "astro_usno_next_phase_utc": moon_phase_utc,
        "astro_noaa_solar_cycle_f107_month": f107_month,
        "astro_noaa_solar_cycle_f107_value": f107_value,
        "quality_missing_vlf": "0" if vlf_captures else "1",
        "quality_missing_astro": "0" if astro_captures else "1",
        "quality_notes": "live smoke row; target intentionally unlabeled",
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES, lineterminator="\n")
        writer.writeheader()
        writer.writerow(row)

    return row


def parse_utc(value: str) -> datetime:
    if not value.endswith("Z"):
        raise ValueError(f"expected UTC timestamp ending in Z: {value}")
    return datetime.fromisoformat(value[:-1] + "+00:00")


def _read_events(events_csv: Path, start_utc: datetime, end_utc: datetime) -> list[dict[str, str]]:
    with events_csv.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    return [
        row
        for row in rows
        if start_utc <= parse_utc(row["event_time_utc"]) < end_utc
    ]


def _read_capture_metadata(paths: Iterable[Path], *, before_utc: datetime) -> list[CaptureSummary]:
    captures = []
    for path in paths:
        metadata = json.loads(path.read_text(encoding="utf-8"))
        captured_at = metadata.get("captured_at_utc", "")
        if not captured_at:
            continue
        if parse_utc(captured_at) >= before_utc:
            continue
        captures.append(
            CaptureSummary(
                source_id=metadata.get("source_id", ""),
                captured_at_utc=captured_at,
                payload_path=str(path).removesuffix(".metadata.json"),
                last_modified_utc=_http_datetime_to_utc(metadata.get("headers", {}).get("Last-Modified", "")),
            )
        )
    return captures


def _latest_capture(captures: Iterable[CaptureSummary]) -> CaptureSummary | None:
    return max(captures, key=lambda capture: parse_utc(capture.captured_at_utc), default=None)


def _max_magnitude(events: Iterable[dict[str, str]]) -> str:
    values = [float(row["magnitude"]) for row in events if row.get("magnitude")]
    if not values:
        return ""
    return f"{max(values):g}"


def _next_moon_phase(paths: Iterable[Path], *, after_utc: datetime) -> tuple[str, str]:
    for path in paths:
        if "usno_moon_phases" not in path.name:
            continue
        payload = json.loads(Path(str(path).removesuffix(".metadata.json")).read_text(encoding="utf-8"))
        for phase in payload.get("phasedata", []):
            phase_time = parse_utc(
                f"{int(phase['year']):04d}-{int(phase['month']):02d}-{int(phase['day']):02d}T{phase['time']}:00Z"
            )
            if phase_time >= after_utc:
                return str(phase.get("phase", "")), _format_utc(phase_time)
    return "", ""


def _f107_month_value(paths: Iterable[Path], *, window_end: datetime) -> tuple[str, str]:
    month = f"{window_end.year:04d}-{window_end.month:02d}"
    for path in paths:
        if "noaa_solar_cycle_f107" not in path.name:
            continue
        payload = json.loads(Path(str(path).removesuffix(".metadata.json")).read_text(encoding="utf-8"))
        for item in payload:
            if item.get("time-tag") == month:
                return month, str(item.get("f10.7", ""))
    return month, ""


def _http_datetime_to_utc(value: str) -> str:
    if not value:
        return ""
    parsed = datetime.strptime(value, "%a, %d %b %Y %H:%M:%S GMT").replace(tzinfo=timezone.utc)
    return _format_utc(parsed)


def _format_utc(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _window_id(region_id: str, window_start_utc: str, window_end_utc: str) -> str:
    return (
        f"{region_id}_{window_start_utc}_{window_end_utc}"
        .replace(":", "")
        .replace("-", "")
        .replace("T", "t")
        .replace("Z", "z")
    )

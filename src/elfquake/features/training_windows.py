"""Build labeled training windows from normalized seismic events."""

from __future__ import annotations

import csv
from datetime import timedelta
from pathlib import Path

from elfquake.features.common import format_utc, parse_utc


FIELDNAMES = [
    "window_id",
    "region_id",
    "window_start_utc",
    "window_end_utc",
    "target_start_utc",
    "target_end_utc",
    "target_magnitude_min",
    "seismic_event_count",
    "seismic_max_magnitude",
    "target_event_count",
    "target_occurred",
    "target_status",
    "source_file",
]


def build_seismic_training_windows(
    *,
    events_csv: Path,
    out_path: Path,
    region_id: str,
    start_utc: str,
    end_utc: str,
    window_days: int = 7,
    horizon_days: int = 7,
    target_magnitude_min: str = "3.0",
) -> list[dict[str, str]]:
    if window_days < 1 or horizon_days < 1:
        raise ValueError("window_days and horizon_days must be at least 1")
    start = parse_utc(start_utc)
    end = parse_utc(end_utc)
    if end <= start:
        raise ValueError("end_utc must be after start_utc")

    events = _read_events(events_csv, region_id)
    rows = []
    cursor = start
    window_delta = timedelta(days=window_days)
    horizon_delta = timedelta(days=horizon_days)
    while cursor + window_delta + horizon_delta <= end:
        window_end = cursor + window_delta
        target_end = window_end + horizon_delta
        feature_events = [
            event
            for event in events
            if cursor <= parse_utc(event["event_time_utc"]) < window_end
        ]
        target_events = [
            event
            for event in events
            if window_end <= parse_utc(event["event_time_utc"]) < target_end
            and float(event["magnitude"]) >= float(target_magnitude_min)
        ]
        row = {
            "window_id": _window_id(region_id, format_utc(cursor), format_utc(window_end)),
            "region_id": region_id,
            "window_start_utc": format_utc(cursor),
            "window_end_utc": format_utc(window_end),
            "target_start_utc": format_utc(window_end),
            "target_end_utc": format_utc(target_end),
            "target_magnitude_min": target_magnitude_min,
            "seismic_event_count": str(len(feature_events)),
            "seismic_max_magnitude": _max_magnitude(feature_events),
            "target_event_count": str(len(target_events)),
            "target_occurred": "1" if target_events else "0",
            "target_status": "labeled",
            "source_file": str(events_csv),
        }
        rows.append(row)
        cursor += window_delta

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    return rows


def _read_events(events_csv: Path, region_id: str) -> list[dict[str, str]]:
    with events_csv.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not region_id or region_id == "all_italy":
        return rows
    return [row for row in rows if row.get("region_id", row.get("italy_region", region_id)) == region_id]


def _max_magnitude(events: list[dict[str, str]]) -> str:
    values = [float(row["magnitude"]) for row in events if row.get("magnitude")]
    if not values:
        return ""
    return f"{max(values):g}"


def _window_id(region_id: str, window_start_utc: str, window_end_utc: str) -> str:
    return (
        f"{region_id}_{window_start_utc}_{window_end_utc}"
        .replace(":", "")
        .replace("-", "")
        .replace("T", "t")
        .replace("Z", "z")
    )

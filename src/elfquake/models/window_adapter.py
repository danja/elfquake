"""Regular-window adapters for event-list model inputs."""

from __future__ import annotations

import csv
import math
from datetime import timedelta
from pathlib import Path

from elfquake.features.common import format_utc, parse_utc


ID_FIELDS = ["window_id", "region_id", "window_start_utc", "window_end_utc", "source_file"]


def build_event_window_features(
    *,
    events_csv: Path,
    out_path: Path,
    region_id: str,
    start_utc: str,
    end_utc: str,
    window_seconds: int,
    feature_prefix: str = "seismic",
    min_magnitude: float | None = None,
) -> list[dict[str, str]]:
    if window_seconds < 1:
        raise ValueError("window_seconds must be at least 1")
    if not feature_prefix:
        raise ValueError("feature_prefix is required")
    start = parse_utc(start_utc)
    end = parse_utc(end_utc)
    if end <= start:
        raise ValueError("end_utc must be after start_utc")

    events = _read_events(events_csv, region_id=region_id, min_magnitude=min_magnitude)
    rows = []
    cursor = start
    delta = timedelta(seconds=window_seconds)
    while cursor < end:
        window_end = min(cursor + delta, end)
        window_events = [
            event
            for event in events
            if cursor <= parse_utc(event["event_time_utc"]) < window_end
        ]
        rows.append(
            _window_row(
                events=window_events,
                events_csv=events_csv,
                region_id=region_id,
                window_start=format_utc(cursor),
                window_end=format_utc(window_end),
                feature_prefix=feature_prefix,
            )
        )
        cursor = window_end

    fieldnames = ID_FIELDS + _feature_fields(feature_prefix)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    return rows


def _read_events(
    events_csv: Path,
    *,
    region_id: str,
    min_magnitude: float | None,
) -> list[dict[str, str]]:
    with events_csv.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    filtered = [row for row in rows if _event_matches_region(row, region_id)]
    if min_magnitude is not None:
        filtered = [
            row
            for row in filtered
            if row.get("magnitude", "") and float(row["magnitude"]) >= min_magnitude
        ]
    return filtered


def _event_matches_region(event: dict[str, str], region_id: str) -> bool:
    if not region_id or region_id == "all_italy":
        return True
    event_region = event.get("italy_region")
    if not event_region:
        return True
    return event_region == region_id


def _window_row(
    *,
    events: list[dict[str, str]],
    events_csv: Path,
    region_id: str,
    window_start: str,
    window_end: str,
    feature_prefix: str,
) -> dict[str, str]:
    magnitudes = _float_values(events, "magnitude")
    depths = _float_values(events, "depth_km")
    latitudes = _float_values(events, "latitude")
    longitudes = _float_values(events, "longitude")
    return {
        "window_id": _window_id(region_id, window_start, window_end),
        "region_id": region_id,
        "window_start_utc": window_start,
        "window_end_utc": window_end,
        "source_file": str(events_csv),
        f"{feature_prefix}_event_count": str(len(events)),
        f"{feature_prefix}_magnitude_max": _max(magnitudes),
        f"{feature_prefix}_magnitude_mean": _mean(magnitudes),
        f"{feature_prefix}_depth_km_mean": _mean(depths),
        f"{feature_prefix}_latitude_mean": _mean(latitudes),
        f"{feature_prefix}_longitude_mean": _mean(longitudes),
        f"{feature_prefix}_energy_sum": _energy_sum(magnitudes),
        f"quality_missing_{feature_prefix}_event_aggregates": "1" if not events else "0",
    }


def _feature_fields(feature_prefix: str) -> list[str]:
    return [
        f"{feature_prefix}_event_count",
        f"{feature_prefix}_magnitude_max",
        f"{feature_prefix}_magnitude_mean",
        f"{feature_prefix}_depth_km_mean",
        f"{feature_prefix}_latitude_mean",
        f"{feature_prefix}_longitude_mean",
        f"{feature_prefix}_energy_sum",
        f"quality_missing_{feature_prefix}_event_aggregates",
    ]


def _float_values(rows: list[dict[str, str]], field: str) -> list[float]:
    values = []
    for row in rows:
        value = row.get(field, "")
        if value:
            values.append(float(value))
    return values


def _max(values: list[float]) -> str:
    if not values:
        return ""
    return f"{max(values):g}"


def _mean(values: list[float]) -> str:
    if not values:
        return ""
    return f"{sum(values) / len(values):.9f}"


def _energy_sum(magnitudes: list[float]) -> str:
    if not magnitudes:
        return "0"
    # Gutenberg-Richter style relative energy proxy. It preserves event-shape
    # ordering without claiming physical units.
    return f"{sum(math.pow(10.0, 1.5 * magnitude) for magnitude in magnitudes):.9f}"


def _window_id(region_id: str, window_start_utc: str, window_end_utc: str) -> str:
    return (
        f"{region_id}_{window_start_utc}_{window_end_utc}"
        .replace(":", "")
        .replace("-", "")
        .replace("T", "t")
        .replace("Z", "z")
    )

"""Derived earthquake-like event lists from sandpile simulation outputs."""

from __future__ import annotations

import csv
import math
from datetime import datetime, timedelta, timezone
from pathlib import Path

from elfquake.normalize.ingv import NORMALIZED_FIELDS


SYNTHETIC_EVENT_FIELDS = NORMALIZED_FIELDS + [
    "step",
    "x",
    "y",
    "topple_count",
    "released_mass",
    "location_quality",
]


def build_synthetic_event_list(
    *,
    summary_csv: Path,
    sensors_csv: Path,
    out_path: Path,
    grid_width: int,
    grid_height: int,
    start_time_utc: str = "2026-01-01T00:00:00Z",
    step_seconds: int = 60,
    min_topple_count: int = 1,
    lat_min: float = 41.5,
    lat_max: float = 43.5,
    lon_min: float = 12.0,
    lon_max: float = 14.5,
    magnitude_type: str = "MLs",
    source: str = "elfquake_sandpile_synthetic",
    ingested_at_utc: str | None = None,
) -> list[dict[str, str]]:
    if grid_width < 2 or grid_height < 2:
        raise ValueError("grid_width and grid_height must be at least 2")
    if step_seconds < 1:
        raise ValueError("step_seconds must be at least 1")
    if min_topple_count < 0:
        raise ValueError("min_topple_count must be non-negative")
    if not (lat_min < lat_max and lon_min < lon_max):
        raise ValueError("latitude and longitude bounds must be increasing")

    start = _parse_utc(start_time_utc)
    ingested = ingested_at_utc or start_time_utc
    sensors_by_step = _read_sensors_by_step(sensors_csv)
    rows = []

    with summary_csv.open(newline="", encoding="utf-8") as handle:
        for summary in csv.DictReader(handle):
            step = int(summary["step"])
            topple_count = int(summary["topple_count"])
            avalanche_count = int(summary["avalanche_count"])
            if avalanche_count <= 0 or topple_count < min_topple_count:
                continue

            sensor, location_quality = _event_sensor(sensors_by_step.get(step, []))
            x = int(sensor.get("x", "0"))
            y = int(sensor.get("y", "0"))
            latitude, longitude = _grid_to_central_italy(
                x=x,
                y=y,
                grid_width=grid_width,
                grid_height=grid_height,
                lat_min=lat_min,
                lat_max=lat_max,
                lon_min=lon_min,
                lon_max=lon_max,
            )
            event_time = start + timedelta(seconds=step * step_seconds)
            released_mass = int(summary.get("released_mass", "0"))
            row = {
                "event_id": f"synthetic_sandpile_step_{step:06d}",
                "source": source,
                "event_time_utc": _format_utc(event_time),
                "latitude": f"{latitude:.5f}",
                "longitude": f"{longitude:.5f}",
                "depth_km": _depth_km(y=y, grid_height=grid_height),
                "magnitude": _magnitude(topple_count=topple_count, released_mass=released_mass),
                "magnitude_type": magnitude_type,
                "italy_region": "central_italy",
                "event_location_name": f"Synthetic sandpile cell x={x} y={y}",
                "event_type": "earthquake",
                "raw_file": str(summary_csv),
                "ingested_at_utc": ingested,
                "raw_uri": str(summary_csv),
                "step": str(step),
                "x": str(x),
                "y": str(y),
                "topple_count": str(topple_count),
                "released_mass": str(released_mass),
                "location_quality": location_quality,
            }
            rows.append({field: row.get(field, "") for field in SYNTHETIC_EVENT_FIELDS})

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=SYNTHETIC_EVENT_FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    return rows


def _read_sensors_by_step(sensors_csv: Path) -> dict[int, list[dict[str, str]]]:
    grouped: dict[int, list[dict[str, str]]] = {}
    with sensors_csv.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            grouped.setdefault(int(row["step"]), []).append(row)
    return grouped


def _event_sensor(rows: list[dict[str, str]]) -> tuple[dict[str, str], str]:
    if not rows:
        return {"x": "0", "y": "0", "height": "0", "local_topple_count": "0"}, "missing_sensor"
    best_topple = max(rows, key=lambda row: (int(row.get("local_topple_count", "0")), int(row.get("height", "0"))))
    if int(best_topple.get("local_topple_count", "0")) > 0:
        return best_topple, "topple_sensor"
    return max(rows, key=lambda row: int(row.get("height", "0"))), "height_proxy"


def _grid_to_central_italy(
    *,
    x: int,
    y: int,
    grid_width: int,
    grid_height: int,
    lat_min: float,
    lat_max: float,
    lon_min: float,
    lon_max: float,
) -> tuple[float, float]:
    x_ratio = min(1.0, max(0.0, x / (grid_width - 1)))
    y_ratio = min(1.0, max(0.0, y / (grid_height - 1)))
    longitude = lon_min + x_ratio * (lon_max - lon_min)
    latitude = lat_max - y_ratio * (lat_max - lat_min)
    return latitude, longitude


def _magnitude(*, topple_count: int, released_mass: int) -> str:
    energy_proxy = max(1, topple_count + released_mass)
    return f"{1.0 + math.log10(energy_proxy):.2f}"


def _depth_km(*, y: int, grid_height: int) -> str:
    y_ratio = min(1.0, max(0.0, y / (grid_height - 1)))
    return f"{2.0 + y_ratio * 18.0:.2f}"


def _parse_utc(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    return parsed.astimezone(timezone.utc)


def _format_utc(value: datetime) -> str:
    return value.astimezone(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")

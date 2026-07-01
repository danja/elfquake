"""Render normalized earthquake event CSVs on a simple offline Italy map."""

from __future__ import annotations

import csv
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path


ITALY_BOUNDS = (5.5, 19.5, 35.0, 47.8)


@dataclass(frozen=True)
class EventPoint:
    latitude: float
    longitude: float
    magnitude: float


def render_event_map(
    *,
    events_csv: Path,
    out_path: Path,
    metadata_out: Path | None = None,
    title: str = "ELFQuake Italy event map",
    lon_min: float = ITALY_BOUNDS[0],
    lon_max: float = ITALY_BOUNDS[1],
    lat_min: float = ITALY_BOUNDS[2],
    lat_max: float = ITALY_BOUNDS[3],
    min_magnitude: float | None = None,
    max_events: int | None = None,
) -> dict[str, str]:
    """Render events from a normalized INGV-like CSV to a PNG image."""

    events = _read_events(
        events_csv,
        lon_min=lon_min,
        lon_max=lon_max,
        lat_min=lat_min,
        lat_max=lat_max,
        min_magnitude=min_magnitude,
        max_events=max_events,
    )

    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
    try:
        import matplotlib

        matplotlib.use("Agg")
        from matplotlib import pyplot as plt
        from matplotlib.patches import Polygon
    except ImportError as error:  # pragma: no cover - exercised when optional dependency is absent.
        raise RuntimeError("matplotlib is required for event map rendering") from error

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 8), dpi=160)
    fig.patch.set_facecolor("#dcecf4")
    ax.set_facecolor("#cfe5ef")

    _draw_base_map(ax, Polygon)

    if events:
        ordered = sorted(events, key=lambda item: item.magnitude)
        magnitudes = [event.magnitude for event in ordered]
        min_mag = min(magnitudes)
        sizes = [28.0 + max(0.0, event.magnitude - min_mag + 0.35) ** 2 * 34.0 for event in ordered]
        ax.scatter(
            [event.longitude for event in ordered],
            [event.latitude for event in ordered],
            s=sizes,
            c="#2727ff",
            edgecolors="white",
            linewidths=0.85,
            alpha=0.92,
            zorder=10,
        )
        ax.scatter(
            [event.longitude for event in ordered],
            [event.latitude for event in ordered],
            s=[size + 12.0 for size in sizes],
            facecolors="none",
            edgecolors="#111177",
            linewidths=0.55,
            alpha=0.75,
            zorder=9,
        )

    _draw_labels(ax)
    ax.set_xlim(lon_min, lon_max)
    ax.set_ylim(lat_min, lat_max)
    mean_lat = (lat_min + lat_max) / 2.0
    ax.set_aspect(1.0 / max(math.cos(math.radians(mean_lat)), 0.2))
    ax.set_title(title, fontsize=12, pad=8)
    ax.set_xlabel("longitude")
    ax.set_ylabel("latitude")
    ax.grid(color="white", linewidth=0.6, alpha=0.6)
    ax.tick_params(labelsize=7, colors="#4c5960")
    for spine in ax.spines.values():
        spine.set_color("#8aa4af")

    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)

    report = {
        "event_file": str(events_csv),
        "map_file": str(out_path),
        "event_count": str(len(events)),
        "bounds": f"{lon_min},{lon_max},{lat_min},{lat_max}",
        "map_type": "offline_schematic_italy",
        "min_magnitude": "" if min_magnitude is None else str(min_magnitude),
        "max_events": "" if max_events is None else str(max_events),
    }
    if metadata_out is not None:
        metadata_out.parent.mkdir(parents=True, exist_ok=True)
        metadata_out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def _read_events(
    path: Path,
    *,
    lon_min: float,
    lon_max: float,
    lat_min: float,
    lat_max: float,
    min_magnitude: float | None,
    max_events: int | None,
) -> list[EventPoint]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))

    points: list[EventPoint] = []
    for row in rows:
        point = _event_point(row)
        if point is None:
            continue
        if not (lat_min <= point.latitude <= lat_max and lon_min <= point.longitude <= lon_max):
            continue
        if min_magnitude is not None and point.magnitude < min_magnitude:
            continue
        points.append(point)

    if max_events is not None and max_events > 0:
        points = points[-max_events:]
    return points


def _event_point(row: dict[str, str]) -> EventPoint | None:
    try:
        latitude = float(row.get("latitude", ""))
        longitude = float(row.get("longitude", ""))
    except ValueError:
        return None
    try:
        magnitude = float(row.get("magnitude", "0") or "0")
    except ValueError:
        magnitude = 0.0
    if not (math.isfinite(latitude) and math.isfinite(longitude) and math.isfinite(magnitude)):
        return None
    return EventPoint(latitude=latitude, longitude=longitude, magnitude=magnitude)


def _draw_base_map(ax, polygon_class) -> None:
    land = "#e9e0c2"
    land_edge = "#74806f"
    nearby_land = "#d9d4b9"

    for coords in _NEIGHBOUR_POLYGONS:
        ax.add_patch(polygon_class(coords, closed=True, facecolor=nearby_land, edgecolor=land_edge, linewidth=0.8, zorder=1))
    for coords in _ITALY_POLYGONS:
        ax.add_patch(polygon_class(coords, closed=True, facecolor=land, edgecolor="#455246", linewidth=1.0, zorder=3))

    ax.plot(
        [5.6, 7.0, 9.2, 11.3, 12.6, 13.4, 14.6, 15.5, 17.2, 19.2],
        [36.2, 36.6, 36.8, 36.5, 36.0, 35.5, 35.8, 36.7, 37.6, 38.2],
        color="#ff0018",
        linewidth=2.0,
        solid_capstyle="round",
        zorder=4,
    )


def _draw_labels(ax) -> None:
    labels = [
        ("Italy", 12.5, 42.2),
        ("Sardinia", 9.0, 40.0),
        ("Sicily", 14.1, 37.5),
        ("Adriatic Sea", 15.1, 43.2),
        ("Tyrrhenian Sea", 11.0, 39.2),
    ]
    for text, lon, lat in labels:
        ax.text(lon, lat, text, color="#51646b", fontsize=7, ha="center", va="center", alpha=0.8, zorder=5)


def _ellipse(lon: float, lat: float, radius_lon: float, radius_lat: float, rotation_degrees: float = 0.0) -> list[tuple[float, float]]:
    points = []
    rotation = math.radians(rotation_degrees)
    for index in range(48):
        angle = 2.0 * math.pi * index / 48
        x = radius_lon * math.cos(angle)
        y = radius_lat * math.sin(angle)
        points.append(
            (
                lon + x * math.cos(rotation) - y * math.sin(rotation),
                lat + x * math.sin(rotation) + y * math.cos(rotation),
            )
        )
    return points


_NEIGHBOUR_POLYGONS = [
    [(5.5, 43.2), (7.2, 44.5), (7.3, 47.8), (5.5, 47.8)],
    [(13.0, 45.5), (19.5, 45.0), (19.5, 47.8), (12.0, 47.8)],
    [(17.0, 39.5), (19.5, 40.2), (19.5, 45.2), (15.2, 44.7), (14.7, 43.7), (15.6, 42.6)],
    [(6.0, 35.0), (13.0, 35.0), (12.3, 36.0), (10.0, 36.8), (7.5, 36.5)],
]

_ITALY_MAINLAND = [
    (7.1, 44.2),
    (8.1, 44.5),
    (9.3, 44.3),
    (10.0, 43.9),
    (10.8, 43.4),
    (11.6, 42.7),
    (12.5, 41.9),
    (13.4, 41.3),
    (14.5, 40.8),
    (15.5, 40.1),
    (16.4, 39.3),
    (17.0, 38.4),
    (16.3, 37.9),
    (15.5, 38.5),
    (15.0, 39.2),
    (14.4, 40.0),
    (13.4, 40.4),
    (12.7, 41.0),
    (12.1, 41.8),
    (11.3, 42.5),
    (10.5, 43.1),
    (9.8, 43.8),
    (8.6, 44.0),
    (7.6, 43.8),
]

_ITALY_POLYGONS = [
    _ITALY_MAINLAND,
    _ellipse(9.0, 40.0, 0.75, 1.45, -15.0),
    _ellipse(14.0, 37.6, 1.35, 0.42, 8.0),
    _ellipse(9.1, 42.1, 0.45, 0.95, -10.0),
]

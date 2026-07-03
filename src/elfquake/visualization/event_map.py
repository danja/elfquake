"""Render normalized earthquake event CSVs on an offline Italy map."""

from __future__ import annotations

import csv
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path


ITALY_BOUNDS = (5.5, 19.5, 35.0, 47.8)
DEFAULT_BASEMAP_GEOJSON = Path(__file__).with_name("assets") / "italy-natural-earth-10m.geojson"


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
    basemap_geojson: Path | None = DEFAULT_BASEMAP_GEOJSON,
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

    map_type = _draw_base_map(
        ax,
        Polygon,
        basemap_geojson=basemap_geojson,
        lon_min=lon_min,
        lon_max=lon_max,
        lat_min=lat_min,
        lat_max=lat_max,
    )

    if events:
        ordered = sorted(events, key=lambda item: item.magnitude)
        magnitudes = [event.magnitude for event in ordered]
        min_mag = min(magnitudes)
        max_mag = max(magnitudes)
        sizes = [_magnitude_marker_size(event.magnitude, min_magnitude=min_mag, max_magnitude=max_mag) for event in ordered]
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
        _draw_magnitude_legend(ax, min_magnitude=min_mag, max_magnitude=max_mag)

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
        "map_type": map_type,
        "basemap_geojson": "" if basemap_geojson is None else str(basemap_geojson),
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


def _magnitude_marker_size(magnitude: float, *, min_magnitude: float, max_magnitude: float) -> float:
    if max_magnitude > min_magnitude:
        ratio = (magnitude - min_magnitude) / (max_magnitude - min_magnitude)
    else:
        ratio = 0.5
    ratio = min(1.0, max(0.0, ratio))
    return 32.0 + ratio * ratio * 260.0


def _draw_magnitude_legend(ax, *, min_magnitude: float, max_magnitude: float) -> None:
    if max_magnitude > min_magnitude:
        legend_magnitudes = [min_magnitude, (min_magnitude + max_magnitude) / 2.0, max_magnitude]
    else:
        legend_magnitudes = [min_magnitude]
    handles = [
        ax.scatter(
            [],
            [],
            s=_magnitude_marker_size(magnitude, min_magnitude=min_magnitude, max_magnitude=max_magnitude),
            c="#2727ff",
            edgecolors="white",
            linewidths=0.85,
            alpha=0.92,
        )
        for magnitude in legend_magnitudes
    ]
    labels = [f"M {magnitude:.2f}" for magnitude in legend_magnitudes]
    legend = ax.legend(
        handles,
        labels,
        title="Magnitude",
        loc="lower left",
        frameon=True,
        fontsize=7,
        title_fontsize=8,
        borderpad=0.6,
        labelspacing=0.8,
        handletextpad=1.2,
    )
    legend.get_frame().set_facecolor("#edf5f8")
    legend.get_frame().set_edgecolor("#8aa4af")
    legend.get_frame().set_alpha(0.92)


def _draw_base_map(
    ax,
    polygon_class,
    *,
    basemap_geojson: Path | None,
    lon_min: float,
    lon_max: float,
    lat_min: float,
    lat_max: float,
) -> str:
    if basemap_geojson is not None:
        rings = _load_geojson_rings(
            basemap_geojson,
            lon_min=lon_min,
            lon_max=lon_max,
            lat_min=lat_min,
            lat_max=lat_max,
        )
        if rings:
            _draw_geojson_line_map(ax, polygon_class, rings)
            return "natural_earth_line_italy"

    _draw_schematic_base_map(ax, polygon_class)
    return "offline_schematic_italy"


def _load_geojson_rings(
    path: Path,
    *,
    lon_min: float,
    lon_max: float,
    lat_min: float,
    lat_max: float,
) -> list[list[tuple[float, float]]]:
    if not path.exists():
        return []
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []

    rings: list[list[tuple[float, float]]] = []
    for feature in payload.get("features", []):
        geometry = feature.get("geometry") if isinstance(feature, dict) else None
        if not isinstance(geometry, dict):
            continue
        for ring in _geometry_exterior_rings(geometry):
            if len(ring) < 3:
                continue
            if _ring_overlaps_bounds(ring, lon_min=lon_min, lon_max=lon_max, lat_min=lat_min, lat_max=lat_max):
                rings.append(ring)
    return rings


def _geometry_exterior_rings(geometry: dict[str, object]) -> list[list[tuple[float, float]]]:
    geometry_type = geometry.get("type")
    coordinates = geometry.get("coordinates")
    polygons: list[object]
    if geometry_type == "Polygon":
        polygons = [coordinates]
    elif geometry_type == "MultiPolygon":
        polygons = list(coordinates) if isinstance(coordinates, list) else []
    else:
        return []

    rings: list[list[tuple[float, float]]] = []
    for polygon in polygons:
        if not isinstance(polygon, list) or not polygon:
            continue
        exterior = polygon[0]
        if not isinstance(exterior, list):
            continue
        ring: list[tuple[float, float]] = []
        for point in exterior:
            if not isinstance(point, list) or len(point) < 2:
                continue
            try:
                lon = float(point[0])
                lat = float(point[1])
            except (TypeError, ValueError):
                continue
            if math.isfinite(lon) and math.isfinite(lat):
                ring.append((lon, lat))
        rings.append(ring)
    return rings


def _ring_overlaps_bounds(
    ring: list[tuple[float, float]],
    *,
    lon_min: float,
    lon_max: float,
    lat_min: float,
    lat_max: float,
) -> bool:
    ring_lon_min = min(point[0] for point in ring)
    ring_lon_max = max(point[0] for point in ring)
    ring_lat_min = min(point[1] for point in ring)
    ring_lat_max = max(point[1] for point in ring)
    return not (
        ring_lon_max < lon_min
        or ring_lon_min > lon_max
        or ring_lat_max < lat_min
        or ring_lat_min > lat_max
    )


def _draw_geojson_line_map(ax, polygon_class, rings: list[list[tuple[float, float]]]) -> None:
    for ring in rings:
        ax.add_patch(
            polygon_class(
                ring,
                closed=True,
                facecolor="#f2efe3",
                edgecolor="#243238",
                linewidth=0.65,
                zorder=3,
            )
        )


def _draw_schematic_base_map(ax, polygon_class) -> None:
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

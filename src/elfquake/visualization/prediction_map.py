"""Render actual synthetic events and PyTorch-predicted event windows on Italy."""

from __future__ import annotations

import csv
import json
import math
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from elfquake.visualization.event_map import (
    DEFAULT_BASEMAP_GEOJSON,
    ITALY_BOUNDS,
    EventPoint,
    _draw_base_map,
    _draw_labels,
    _event_point,
    _magnitude_marker_size,
)


@dataclass(frozen=True)
class TimedEventPoint:
    point: EventPoint
    event_time_utc: str


@dataclass(frozen=True)
class PredictedEventPoint:
    point: EventPoint
    probability: float
    window_id: str


def render_prediction_event_map(
    *,
    events_csv: Path,
    windows_csv: Path,
    report_json: Path,
    out_path: Path,
    metadata_out: Path | None = None,
    title: str = "ELFQuake synthetic actual vs predicted events",
    evaluation: str | None = None,
    threshold: float | None = None,
    lon_min: float = ITALY_BOUNDS[0],
    lon_max: float = ITALY_BOUNDS[1],
    lat_min: float = ITALY_BOUNDS[2],
    lat_max: float = ITALY_BOUNDS[3],
    max_actual_events: int | None = None,
    basemap_geojson: Path | None = DEFAULT_BASEMAP_GEOJSON,
) -> dict[str, str]:
    """Render actual avalanche events and predicted-positive matched events."""

    events = _read_timed_events(events_csv, lon_min=lon_min, lon_max=lon_max, lat_min=lat_min, lat_max=lat_max)
    report = json.loads(report_json.read_text(encoding="utf-8"))
    chosen_evaluation = evaluation or _best_calibrated_evaluation(report)
    evaluation_report = report.get("evaluations", {}).get(chosen_evaluation)
    if not isinstance(evaluation_report, dict) or evaluation_report.get("status") != "evaluated":
        raise ValueError(f"report has no evaluated entry named {chosen_evaluation!r}")
    chosen_threshold = threshold if threshold is not None else float(evaluation_report.get("calibrated_threshold", 0.5))
    test_rows = _report_test_rows(report=report, windows_csv=windows_csv)
    probabilities = [float(value) for value in evaluation_report.get("test_probabilities", [])]
    if len(test_rows) != len(probabilities):
        raise ValueError("report test probabilities do not match reconstructed test rows")

    predicted, missing_location_count = _predicted_event_points(
        events=events,
        test_rows=test_rows,
        probabilities=probabilities,
        threshold=chosen_threshold,
    )
    actual = [event.point for event in events]
    if max_actual_events is not None and max_actual_events > 0:
        actual = actual[-max_actual_events:]

    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
    try:
        import matplotlib

        matplotlib.use("Agg")
        from matplotlib import pyplot as plt
        from matplotlib.lines import Line2D
        from matplotlib.patches import Polygon
    except ImportError as error:  # pragma: no cover
        raise RuntimeError("matplotlib is required for prediction event map rendering") from error

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

    magnitude_range = _magnitude_range([*actual, *[item.point for item in predicted]])
    _scatter_actual(ax, actual, magnitude_range=magnitude_range)
    _scatter_predicted(ax, predicted, magnitude_range=magnitude_range)
    _draw_prediction_legend(ax, Line2D)
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

    predicted_positive_count = sum(1 for probability in probabilities if probability >= chosen_threshold)
    payload = {
        "map_file": str(out_path),
        "events_csv": str(events_csv),
        "windows_csv": str(windows_csv),
        "report_json": str(report_json),
        "map_type": map_type,
        "actual_event_count": str(len(actual)),
        "predicted_positive_window_count": str(predicted_positive_count),
        "predicted_event_point_count": str(len(predicted)),
        "predicted_without_location_count": str(missing_location_count),
        "evaluation": chosen_evaluation,
        "threshold": f"{chosen_threshold:.6f}",
        "test_group": str(report.get("test_group", "")),
        "note": "PyTorch predicts event windows; predicted map points are actual target-window avalanche events matched to predicted-positive windows.",
    }
    if metadata_out is not None:
        metadata_out.parent.mkdir(parents=True, exist_ok=True)
        metadata_out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


def _read_timed_events(
    path: Path,
    *,
    lon_min: float,
    lon_max: float,
    lat_min: float,
    lat_max: float,
) -> list[TimedEventPoint]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    events = []
    for row in rows:
        point = _event_point(row)
        if point is None:
            continue
        if not (lat_min <= point.latitude <= lat_max and lon_min <= point.longitude <= lon_max):
            continue
        event_time = row.get("event_time_utc", "")
        if _parse_utc(event_time) is None:
            continue
        events.append(TimedEventPoint(point=point, event_time_utc=event_time))
    return events


def _best_calibrated_evaluation(report: dict[str, object]) -> str:
    evaluated = [
        (name, item)
        for name, item in report.get("evaluations", {}).items()
        if isinstance(item, dict) and item.get("status") == "evaluated"
    ]
    if not evaluated:
        raise ValueError("report has no evaluated entries")
    name, _ = max(
        evaluated,
        key=lambda row: float(row[1].get("calibrated_test_metrics", {}).get("balanced_accuracy", 0.0)),
    )
    return name


def _report_test_rows(*, report: dict[str, object], windows_csv: Path) -> list[dict[str, str]]:
    with windows_csv.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    labeled = [row for row in rows if row.get("target_occurred") in {"0", "1"}]
    schema = report.get("schema")
    if schema == "elfquake.torch_tabular_group_holdout.v1":
        group_field = str(report.get("group_field", "dataset_id"))
        test_group = str(report.get("test_group", ""))
        return [row for row in labeled if row.get(group_field, "") == test_group]
    if schema == "elfquake.torch_tabular_holdout.v1":
        time_field = str(report.get("time_field", "window_start_utc"))
        train_fraction = float(report.get("train_fraction", 0.8))
        ordered = sorted(labeled, key=lambda row: row.get(time_field, ""))
        train_count = min(len(ordered) - 1, max(2, int(len(ordered) * train_fraction)))
        return ordered[train_count:]
    raise ValueError(f"unsupported report schema for prediction map: {schema}")


def _predicted_event_points(
    *,
    events: list[TimedEventPoint],
    test_rows: list[dict[str, str]],
    probabilities: list[float],
    threshold: float,
) -> tuple[list[PredictedEventPoint], int]:
    predicted: list[PredictedEventPoint] = []
    missing_location_count = 0
    for row, probability in zip(test_rows, probabilities):
        if probability < threshold:
            continue
        start = _parse_utc(row.get("window_end_utc", ""))
        end = _target_end(row)
        if start is None or end is None:
            missing_location_count += 1
            continue
        matched = [
            event for event in events if start <= _parse_utc(event.event_time_utc) < end  # type: ignore[operator]
        ]
        if not matched:
            missing_location_count += 1
            continue
        for event in matched:
            predicted.append(
                PredictedEventPoint(
                    point=event.point,
                    probability=probability,
                    window_id=row.get("window_id", ""),
                )
            )
    return predicted, missing_location_count


def _target_end(row: dict[str, str]) -> datetime | None:
    start = _parse_utc(row.get("window_start_utc", ""))
    end = _parse_utc(row.get("window_end_utc", ""))
    if start is None or end is None or end <= start:
        return None
    return end + (end - start)


def _parse_utc(value: str) -> datetime | None:
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    return parsed.astimezone(timezone.utc)


def _magnitude_range(points: list[EventPoint]) -> tuple[float, float]:
    if not points:
        return (0.0, 1.0)
    magnitudes = [point.magnitude for point in points]
    return min(magnitudes), max(magnitudes)


def _scatter_actual(ax, points: list[EventPoint], *, magnitude_range: tuple[float, float]) -> None:
    if not points:
        return
    min_mag, max_mag = magnitude_range
    ax.scatter(
        [point.longitude for point in points],
        [point.latitude for point in points],
        s=[_magnitude_marker_size(point.magnitude, min_magnitude=min_mag, max_magnitude=max_mag) for point in points],
        c="#1f4bff",
        edgecolors="white",
        linewidths=0.75,
        alpha=0.58,
        zorder=8,
    )


def _scatter_predicted(ax, points: list[PredictedEventPoint], *, magnitude_range: tuple[float, float]) -> None:
    if not points:
        return
    min_mag, max_mag = magnitude_range
    ax.scatter(
        [item.point.longitude for item in points],
        [item.point.latitude for item in points],
        s=[
            _magnitude_marker_size(item.point.magnitude, min_magnitude=min_mag, max_magnitude=max_mag) + 28.0
            for item in points
        ],
        facecolors="none",
        edgecolors="#ff7a00",
        linewidths=1.65,
        alpha=0.95,
        zorder=11,
    )
    ax.scatter(
        [item.point.longitude for item in points],
        [item.point.latitude for item in points],
        s=18,
        marker="x",
        c="#5b1700",
        linewidths=0.9,
        alpha=0.95,
        zorder=12,
    )


def _draw_prediction_legend(ax, line_class) -> None:
    handles = [
        line_class([0], [0], marker="o", color="none", markerfacecolor="#1f4bff", markeredgecolor="white", markersize=8),
        line_class([0], [0], marker="o", color="#ff7a00", markerfacecolor="none", markeredgewidth=1.6, markersize=9),
    ]
    legend = ax.legend(
        handles,
        ["actual avalanche event", "predicted event window hit"],
        loc="lower left",
        frameon=True,
        fontsize=7,
        borderpad=0.6,
        labelspacing=0.8,
    )
    legend.get_frame().set_facecolor("#edf5f8")
    legend.get_frame().set_edgecolor("#8aa4af")
    legend.get_frame().set_alpha(0.92)

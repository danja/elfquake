"""Measure how much a spatial-cell target design can express before modeling.

Every fixed-cell Italy result so far has collapsed to a static per-cell base
rate (next-actions items 93, 96, 102). The cell-stratified control shows why:
with 30-minute anchors and a 7-day horizon, consecutive target windows overlap
by more than 99%, so a held-out partition shorter than the horizon contains
almost no within-cell label variation at all. Scoring a model against that
target measures the cell's base rate because the base rate is all there is.

This diagnostic counts what a candidate design actually offers, without
training anything: per cell, how often the label changes between consecutive
anchors, and how many cells carry both classes inside a held-out partition
formed the same way the evaluator forms it. Run it before choosing a horizon,
cell size, or magnitude threshold, not after fitting a model to one.
"""

from __future__ import annotations

import csv
import json
from datetime import timedelta
from pathlib import Path

from elfquake.features.common import parse_utc
from elfquake.features.spatial_targets import _in_cell
from elfquake.models.real_transfer_trial import _cells

DEFAULT_HORIZON_DAYS = (1, 2, 3, 7)


def diagnose_spatial_target_design(
    *,
    input_csv: Path,
    events_csv: Path,
    out_path: Path,
    horizon_days: tuple[int, ...] = DEFAULT_HORIZON_DAYS,
    cell_degrees: tuple[float, ...] = (1.5,),
    target_magnitude_min: tuple[float, ...] = (2.5,),
    catalog_end_utc: str | None = None,
    train_fraction: float = 0.8,
) -> dict[str, object]:
    with input_csv.open(newline="", encoding="utf-8") as handle:
        anchors = sorted(csv.DictReader(handle), key=lambda row: row["window_start_utc"])
    with events_csv.open(newline="", encoding="utf-8") as handle:
        events = _event_points(list(csv.DictReader(handle)))

    catalog_end = parse_utc(catalog_end_utc) if catalog_end_utc else max(time for time, _, _, _ in events)
    report: dict[str, object] = {
        "schema": "elfquake.spatial_target_design.v1",
        "input": str(input_csv),
        "events": str(events_csv),
        "anchor_count": len(anchors),
        "event_count": len(events),
        "catalog_end_utc": catalog_end.isoformat().replace("+00:00", "Z"),
        "train_fraction": train_fraction,
        "designs": [],
    }
    for degrees in cell_degrees:
        cells = _cells(degrees)
        for magnitude in target_magnitude_min:
            eligible = [item for item in events if item[3] >= magnitude]
            for horizon in horizon_days:
                report["designs"].append(
                    _one_design(
                        anchors=anchors,
                        events=eligible,
                        cells=cells,
                        cell_degrees=degrees,
                        horizon_days=horizon,
                        target_magnitude_min=magnitude,
                        catalog_end=catalog_end,
                        train_fraction=train_fraction,
                    )
                )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def _one_design(
    *,
    anchors: list[dict[str, str]],
    events: list[tuple[object, float, float, float]],
    cells: list[tuple[float, float]],
    cell_degrees: float,
    horizon_days: int,
    target_magnitude_min: float,
    catalog_end: object,
    train_fraction: float,
) -> dict[str, object]:
    horizon = timedelta(days=horizon_days)
    labeled_anchors: list[str] = []
    series: dict[str, list[int]] = {_cell_id(index, cell): [] for index, cell in enumerate(cells)}
    for anchor in anchors:
        start = parse_utc(anchor["window_end_utc"])
        end = start + horizon
        if end > catalog_end:
            continue
        window_events = [item for item in events if start <= item[0] < end]
        labeled_anchors.append(anchor["window_start_utc"])
        for index, (latitude, longitude) in enumerate(cells):
            hit = any(
                _in_cell(
                    {"latitude": str(item[1]), "longitude": str(item[2])},
                    latitude,
                    longitude,
                    cell_degrees,
                )
                for item in window_events
            )
            series[_cell_id(index, (latitude, longitude))].append(1 if hit else 0)

    design: dict[str, object] = {
        "horizon_days": horizon_days,
        "cell_degrees": cell_degrees,
        "target_magnitude_min": target_magnitude_min,
        "cell_count": len(cells),
        "labeled_anchor_count": len(labeled_anchors),
        "labeled_row_count": len(labeled_anchors) * len(cells),
    }
    if not labeled_anchors:
        design["status"] = "no_mature_anchors"
        return design

    design["labeled_time_start"] = labeled_anchors[0]
    design["labeled_time_end"] = labeled_anchors[-1]
    design["positive_rate"] = _round(
        sum(sum(values) for values in series.values()) / design["labeled_row_count"]
    )
    design["full_record"] = _variation(series)

    # Same split rule as `evaluate_temporal_holdout(group_by_time=True)`: all
    # cells from one anchor stay on the same side of the boundary.
    times = sorted(set(labeled_anchors))
    train_time_count = min(len(times) - 1, max(1, int(len(times) * train_fraction)))
    cutoff = times[train_time_count]
    test_index = [index for index, stamp in enumerate(labeled_anchors) if stamp >= cutoff]
    design["test_anchor_count"] = len(test_index)
    design["test_time_start"] = labeled_anchors[test_index[0]] if test_index else ""
    design["test_time_end"] = labeled_anchors[test_index[-1]] if test_index else ""
    design["test_span_hours"] = (
        _round((parse_utc(design["test_time_end"]) - parse_utc(design["test_time_start"])).total_seconds() / 3600.0)
        if test_index
        else 0.0
    )
    design["held_out"] = _variation({name: [values[i] for i in test_index] for name, values in series.items()})
    design["status"] = "evaluated"
    return design


def _variation(series: dict[str, list[int]]) -> dict[str, object]:
    """Count what the labels can distinguish, not how many rows there are.

    `transitions` is the number of times a cell's label changes between
    consecutive anchors. With overlapping target windows the row count vastly
    overstates the information present; the transition count is the honest
    upper bound on what a model could be scored against within a cell.
    """
    transitions = {
        name: sum(1 for previous, current in zip(values, values[1:]) if previous != current)
        for name, values in series.items()
    }
    two_class = {name for name, values in series.items() if values and 0 < sum(values) < len(values)}
    return {
        "cells_with_both_classes": len(two_class),
        "cells_all_positive": sum(1 for values in series.values() if values and sum(values) == len(values)),
        "cells_all_negative": sum(1 for values in series.values() if values and sum(values) == 0),
        "total_label_transitions": sum(transitions.values()),
        "cells_with_transitions": sum(1 for count in transitions.values() if count),
        "transitions_by_cell": {name: count for name, count in sorted(transitions.items()) if count},
    }


def _event_points(rows: list[dict[str, str]]) -> list[tuple[object, float, float, float]]:
    points = []
    for row in rows:
        magnitude = row.get("magnitude") or ""
        if not magnitude:
            continue
        points.append(
            (
                parse_utc(row["event_time_utc"]),
                float(row["latitude"]),
                float(row["longitude"]),
                float(magnitude),
            )
        )
    return sorted(points)


def _cell_id(index: int, cell: tuple[float, float]) -> str:
    return f"cell_{index:02d}_{cell[0]:.2f}_{cell[1]:.2f}"


def _round(value: float) -> float:
    return round(value, 6)

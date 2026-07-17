"""Spatial-cell targets for VLF-anchored Italy windows."""

from __future__ import annotations

import csv
from datetime import timedelta
from pathlib import Path

from elfquake.features.common import parse_utc
from elfquake.models.real_transfer_trial import _cells


def label_spatial_multimodal_targets(
    *, input_csv: Path, events_csv: Path, out_path: Path, as_of_utc: str,
    catalog_end_utc: str | None = None, cell_degrees: float = 1.5,
    target_magnitude_min: float = 2.5,
) -> list[dict[str, str]]:
    as_of = parse_utc(as_of_utc)
    catalog_end = parse_utc(catalog_end_utc) if catalog_end_utc else None
    cells = _cells(cell_degrees)
    events = _read_events(events_csv)
    with input_csv.open(newline="", encoding="utf-8") as handle:
        source_rows = list(csv.DictReader(handle))
    rows: list[dict[str, str]] = []
    for source in source_rows:
        for index, (cell_latitude, cell_longitude) in enumerate(cells):
            row = dict(source)
            cell_id = f"cell_{index:02d}_{cell_latitude:.2f}_{cell_longitude:.2f}"
            row["window_id"] = f"{source['window_id']}_{cell_id}"
            row["region_id"] = f"all_italy_{cell_id}"
            row["parent_region_id"] = "all_italy"
            row["target_cell_id"] = cell_id
            row["target_cell_latitude"] = f"{cell_latitude:.5f}"
            row["target_cell_longitude"] = f"{cell_longitude:.5f}"
            row["target_cell_degrees"] = f"{cell_degrees:g}"
            row["target_magnitude_min"] = f"{target_magnitude_min:g}"
            target_end = parse_utc(source["target_end_utc"])
            if target_end > as_of or (catalog_end is not None and target_end > catalog_end):
                row["target_event_count"] = ""
                row["target_occurred"] = ""
                row["target_status"] = "unlabeled_pending_future_events"
            else:
                target_start = parse_utc(source["target_start_utc"])
                target_events = [
                    event for event in events
                    if target_start <= parse_utc(event["event_time_utc"]) < target_end
                    and float(event.get("magnitude") or "-inf") >= target_magnitude_min
                    and _in_cell(event, cell_latitude, cell_longitude, cell_degrees)
                ]
                row["target_event_count"] = str(len(target_events))
                row["target_occurred"] = "1" if target_events else "0"
                row["target_status"] = "labeled"
            rows.append(row)
    fieldnames = list(source_rows[0].keys()) if source_rows else []
    for field in ["target_event_count", "target_occurred", "target_status", "parent_region_id", "target_cell_id", "target_cell_latitude", "target_cell_longitude", "target_cell_degrees"]:
        if field not in fieldnames:
            fieldnames.append(field)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    return rows


def _read_events(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _in_cell(event: dict[str, str], latitude: float, longitude: float, cell_degrees: float) -> bool:
    half = cell_degrees / 2.0
    return abs(float(event["latitude"]) - latitude) <= half and abs(float(event["longitude"]) - longitude) <= half

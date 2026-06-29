"""Target label generation for multimodal windows."""

from __future__ import annotations

import csv
from pathlib import Path

from elfquake.features.common import parse_utc


def label_multimodal_targets(
    *,
    input_csv: Path,
    events_csv: Path,
    out_path: Path,
    as_of_utc: str,
) -> list[dict[str, str]]:
    as_of = parse_utc(as_of_utc)
    events = _read_events(events_csv)
    with input_csv.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError("input_csv has no header")
        fieldnames = list(reader.fieldnames)
        rows = list(reader)

    for required in ("target_event_count", "target_occurred", "target_status"):
        if required not in fieldnames:
            fieldnames.append(required)

    labeled = []
    for row in rows:
        target_end = parse_utc(row["target_end_utc"])
        if target_end > as_of:
            row["target_event_count"] = row.get("target_event_count", "")
            row["target_occurred"] = row.get("target_occurred", "")
            row["target_status"] = "unlabeled_pending_future_events"
            labeled.append(row)
            continue

        target_start = parse_utc(row["target_start_utc"])
        min_magnitude = float(row["target_magnitude_min"])
        region_id = row.get("region_id", "")
        target_events = [
            event
            for event in events
            if target_start <= parse_utc(event["event_time_utc"]) < target_end
            and float(event["magnitude"]) >= min_magnitude
            and _event_matches_region(event, region_id)
        ]
        row["target_event_count"] = str(len(target_events))
        row["target_occurred"] = "1" if target_events else "0"
        row["target_status"] = "labeled"
        labeled.append(row)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(labeled)
    return labeled


def _read_events(events_csv: Path) -> list[dict[str, str]]:
    with events_csv.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _event_matches_region(event: dict[str, str], region_id: str) -> bool:
    if not region_id or region_id == "all_italy":
        return True
    event_region = event.get("italy_region")
    if not event_region:
        return True
    return event_region == region_id

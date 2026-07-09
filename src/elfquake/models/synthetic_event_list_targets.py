"""Build forecast-shaped targets from synthetic avalanche event lists."""

from __future__ import annotations

import csv
import json
import math
from bisect import bisect_left
from collections import defaultdict
from pathlib import Path

from elfquake.features.common import format_utc, parse_utc


TARGET_FIELDS = [
    "eventlist_target_status",
    "eventlist_target_count",
    "eventlist_target_occurred",
    "eventlist_target_max_magnitude",
    "eventlist_target_mean_magnitude",
    "eventlist_target_centroid_latitude",
    "eventlist_target_centroid_longitude",
    "eventlist_target_first_event_time_utc",
    "eventlist_target_time_to_first_event_seconds",
    "quality_missing_eventlist_target_location",
]


def build_synthetic_event_list_targets(
    *,
    input_csv: Path,
    out_csv: Path,
    report_path: Path,
    horizon_rows: int = 24,
    magnitude_threshold: float = 2.0,
    group_field: str = "dataset_id",
    source_field: str = "source_file",
) -> dict[str, object]:
    if horizon_rows < 1:
        raise ValueError("horizon_rows must be at least 1")
    rows = _read_rows(input_csv)
    events_by_source = {
        source: _read_events(Path(source), magnitude_threshold=magnitude_threshold)
        for source in sorted({row.get(source_field, "") for row in rows if row.get(source_field, "")})
    }
    output_rows: list[dict[str, str]] = []
    grouped_indexes: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        grouped_indexes[row.get(group_field, "") or row.get(source_field, "")].append(index)

    by_index: dict[int, dict[str, str]] = {}
    for indexes in grouped_indexes.values():
        ordered_indexes = sorted(indexes, key=lambda item: rows[item].get("window_start_utc", ""))
        for offset, row_index in enumerate(ordered_indexes):
            row = rows[row_index]
            target = _target_for_row(
                row=row,
                offset=offset,
                ordered_rows=[rows[index] for index in ordered_indexes],
                events=events_by_source.get(row.get(source_field, ""), []),
                horizon_rows=horizon_rows,
            )
            by_index[row_index] = {**row, **target}
    for index in range(len(rows)):
        output_rows.append(by_index[index])

    fieldnames = _fieldnames(output_rows)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(output_rows)
    report = _report(
        input_csv=input_csv,
        out_csv=out_csv,
        rows=output_rows,
        horizon_rows=horizon_rows,
        magnitude_threshold=magnitude_threshold,
        source_count=len(events_by_source),
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    return report


def _target_for_row(
    *,
    row: dict[str, str],
    offset: int,
    ordered_rows: list[dict[str, str]],
    events: list[dict[str, object]],
    horizon_rows: int,
) -> dict[str, str]:
    after_horizon_index = offset + horizon_rows
    if after_horizon_index >= len(ordered_rows):
        return _empty_target("unlabeled_no_future_window")
    start = parse_utc(row["window_end_utc"])
    end = parse_utc(ordered_rows[after_horizon_index]["window_end_utc"])
    times = [event["parsed_time"] for event in events]
    left = bisect_left(times, start)
    right = bisect_left(times, end)
    target_events = events[left:right]
    if not target_events:
        return {
            "eventlist_target_status": "labeled",
            "eventlist_target_count": "0",
            "eventlist_target_occurred": "0",
            "eventlist_target_max_magnitude": "0",
            "eventlist_target_mean_magnitude": "0",
            "eventlist_target_centroid_latitude": "",
            "eventlist_target_centroid_longitude": "",
            "eventlist_target_first_event_time_utc": "",
            "eventlist_target_time_to_first_event_seconds": "",
            "quality_missing_eventlist_target_location": "1",
        }
    magnitudes = [float(event["magnitude"]) for event in target_events]
    latitudes = [float(event["latitude"]) for event in target_events]
    longitudes = [float(event["longitude"]) for event in target_events]
    first_time = min(event["parsed_time"] for event in target_events)
    return {
        "eventlist_target_status": "labeled",
        "eventlist_target_count": str(len(target_events)),
        "eventlist_target_occurred": "1",
        "eventlist_target_max_magnitude": _fmt(max(magnitudes)),
        "eventlist_target_mean_magnitude": _fmt(sum(magnitudes) / len(magnitudes)),
        "eventlist_target_centroid_latitude": _fmt(sum(latitudes) / len(latitudes)),
        "eventlist_target_centroid_longitude": _fmt(sum(longitudes) / len(longitudes)),
        "eventlist_target_first_event_time_utc": format_utc(first_time),
        "eventlist_target_time_to_first_event_seconds": _fmt((first_time - start).total_seconds()),
        "quality_missing_eventlist_target_location": "0",
    }


def _empty_target(status: str) -> dict[str, str]:
    return {
        "eventlist_target_status": status,
        "eventlist_target_count": "",
        "eventlist_target_occurred": "",
        "eventlist_target_max_magnitude": "",
        "eventlist_target_mean_magnitude": "",
        "eventlist_target_centroid_latitude": "",
        "eventlist_target_centroid_longitude": "",
        "eventlist_target_first_event_time_utc": "",
        "eventlist_target_time_to_first_event_seconds": "",
        "quality_missing_eventlist_target_location": "1",
    }


def _read_events(path: Path, *, magnitude_threshold: float) -> list[dict[str, object]]:
    if not path.exists():
        return []
    rows = []
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            magnitude = _number(row.get("magnitude", ""))
            latitude = _number(row.get("latitude", ""))
            longitude = _number(row.get("longitude", ""))
            time_utc = row.get("event_time_utc", "")
            if magnitude is None or latitude is None or longitude is None or not time_utc:
                continue
            if magnitude < magnitude_threshold:
                continue
            rows.append(
                {
                    "event_time_utc": time_utc,
                    "parsed_time": parse_utc(time_utc),
                    "magnitude": magnitude,
                    "latitude": latitude,
                    "longitude": longitude,
                }
            )
    return sorted(rows, key=lambda item: item["parsed_time"])


def _report(
    *,
    input_csv: Path,
    out_csv: Path,
    rows: list[dict[str, str]],
    horizon_rows: int,
    magnitude_threshold: float,
    source_count: int,
) -> dict[str, object]:
    labeled = [row for row in rows if row.get("eventlist_target_status") == "labeled"]
    positives = [row for row in labeled if row.get("eventlist_target_occurred") == "1"]
    negatives = [row for row in labeled if row.get("eventlist_target_occurred") == "0"]
    counts = [_number(row.get("eventlist_target_count", "")) or 0.0 for row in positives]
    max_magnitudes = [_number(row.get("eventlist_target_max_magnitude", "")) or 0.0 for row in positives]
    time_to_first = [_number(row.get("eventlist_target_time_to_first_event_seconds", "")) for row in positives]
    time_to_first = [value for value in time_to_first if value is not None]
    return {
        "schema": "elfquake.synthetic_event_list_targets.v1",
        "input_csv": str(input_csv),
        "out_csv": str(out_csv),
        "horizon_rows": horizon_rows,
        "magnitude_threshold": magnitude_threshold,
        "source_count": source_count,
        "row_count": len(rows),
        "labeled_row_count": len(labeled),
        "positive_count": len(positives),
        "negative_count": len(negatives),
        "positive_rate": round(len(positives) / len(labeled), 6) if labeled else 0.0,
        "mean_positive_event_count": _mean(counts),
        "max_positive_event_count": _max(counts),
        "mean_positive_max_magnitude": _mean(max_magnitudes),
        "max_positive_max_magnitude": _max(max_magnitudes),
        "mean_time_to_first_event_seconds": _mean(time_to_first),
        "has_location_target_count": sum(1 for row in positives if row.get("quality_missing_eventlist_target_location") == "0"),
        "stage_2_target_health": {
            "has_both_classes": bool(positives and negatives),
            "positive_rate_between_0p1_and_0p9": 0.1 <= (len(positives) / len(labeled) if labeled else 0.0) <= 0.9,
            "has_location_targets": any(row.get("quality_missing_eventlist_target_location") == "0" for row in positives),
        },
    }


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _fieldnames(rows: list[dict[str, str]]) -> list[str]:
    names: list[str] = []
    for row in rows:
        for name in row:
            if name not in names:
                names.append(name)
    return names


def _number(value: str) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def _fmt(value: float) -> str:
    return f"{value:.9f}"


def _mean(values: list[float]) -> float | None:
    return round(sum(values) / len(values), 6) if values else None


def _max(values: list[float]) -> float | None:
    return round(max(values), 6) if values else None

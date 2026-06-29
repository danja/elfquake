"""Utilities for normalized event tables."""

from __future__ import annotations

import csv
from pathlib import Path

from elfquake.features.common import parse_utc
from elfquake.normalize.ingv import NORMALIZED_FIELDS


def combine_normalized_events(*, input_paths: list[Path], out_path: Path) -> list[dict[str, str]]:
    """Merge normalized event CSVs, deduplicate by event_id, and sort by event time."""
    if not input_paths:
        raise ValueError("at least one input path is required")

    by_event_id: dict[str, dict[str, str]] = {}
    seen_without_id: list[dict[str, str]] = []
    for input_path in input_paths:
        with input_path.open(newline="", encoding="utf-8") as handle:
            for index, row in enumerate(csv.DictReader(handle)):
                event_id = row.get("event_id", "")
                if event_id:
                    by_event_id.setdefault(event_id, row)
                else:
                    copy = dict(row)
                    copy["_input_order"] = f"{input_path}:{index}"
                    seen_without_id.append(copy)

    rows = list(by_event_id.values()) + seen_without_id
    rows.sort(key=_sort_key)
    for row in rows:
        row.pop("_input_order", None)

    fieldnames = _fieldnames(rows)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    return rows


def _fieldnames(rows: list[dict[str, str]]) -> list[str]:
    extras: list[str] = []
    for row in rows:
        for field in row:
            if field not in NORMALIZED_FIELDS and field not in extras and not field.startswith("_"):
                extras.append(field)
    return NORMALIZED_FIELDS + extras


def _sort_key(row: dict[str, str]) -> tuple[str, str]:
    event_time = row.get("event_time_utc", "")
    try:
        return (parse_utc(event_time).isoformat(), row.get("event_id") or row.get("_input_order", ""))
    except ValueError:
        return (event_time, row.get("event_id") or row.get("_input_order", ""))

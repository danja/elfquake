"""Quality summaries for prospective VLF feature tables."""

from __future__ import annotations

import csv
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from elfquake.features.common import parse_utc


def summarize_prospective_table(
    *,
    input_csv: Path,
    out_path: Path,
    as_of_utc: str | None = None,
) -> dict[str, object]:
    rows = _read_rows(input_csv)
    resolved_as_of = as_of_utc or _now_utc()
    as_of = parse_utc(resolved_as_of)
    report: dict[str, object] = {
        "input": str(input_csv),
        "as_of_utc": resolved_as_of,
        "row_count": len(rows),
        "target_status_counts": dict(Counter(row.get("target_status", "") for row in rows)),
        "region_counts": dict(Counter(row.get("region_id", "") for row in rows)),
        "ready_to_label_count": sum(
            1
            for row in rows
            if row.get("target_status") != "labeled"
            and row.get("target_end_utc")
            and parse_utc(row["target_end_utc"]) <= as_of
        ),
        "missing_vlf_count": _count_value(rows, "quality_missing_vlf", "1"),
        "missing_vlf_image_features_count": _count_value(rows, "quality_missing_vlf_image_features", "1"),
        "missing_astro_count": _count_value(rows, "quality_missing_astro", "1"),
        "labeled_positive_count": _count_value(rows, "target_occurred", "1"),
        "labeled_negative_count": _count_value(rows, "target_occurred", "0"),
        "first_window_end_utc": _min_timestamp(rows, "window_end_utc"),
        "last_window_end_utc": _max_timestamp(rows, "window_end_utc"),
        "first_target_end_utc": _min_timestamp(rows, "target_end_utc"),
        "last_target_end_utc": _max_timestamp(rows, "target_end_utc"),
        "latest_vlf_capture_utc": _max_timestamp(rows, "vlf_latest_capture_utc"),
        "latest_vlf_image_source_file": _latest_value(rows, timestamp_field="window_end_utc", value_field="vlf_image_latest_source_file"),
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def _now_utc() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _count_value(rows: list[dict[str, str]], field: str, value: str) -> int:
    return sum(1 for row in rows if row.get(field) == value)


def _min_timestamp(rows: list[dict[str, str]], field: str) -> str:
    values = [row.get(field, "") for row in rows if row.get(field)]
    if not values:
        return ""
    return min(values, key=parse_utc)


def _max_timestamp(rows: list[dict[str, str]], field: str) -> str:
    values = [row.get(field, "") for row in rows if row.get(field)]
    if not values:
        return ""
    return max(values, key=parse_utc)


def _latest_value(rows: list[dict[str, str]], *, timestamp_field: str, value_field: str) -> str:
    candidates = [
        row for row in rows if row.get(timestamp_field) and row.get(value_field)
    ]
    if not candidates:
        return ""
    latest = max(candidates, key=lambda row: parse_utc(row[timestamp_field]))
    return latest.get(value_field, "")

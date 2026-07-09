"""Lagged feature context for synthetic event-list probes."""

from __future__ import annotations

import csv
import json
from pathlib import Path

from elfquake.models.learned_forecast import _number
from elfquake.models.synthetic_event_list_model import is_diagnostic_or_target_field


def build_synthetic_lagged_context(
    *,
    input_csv: Path,
    out_csv: Path,
    report_path: Path,
    lags: list[int],
    group_field: str = "dataset_id",
    time_field: str = "window_start_utc",
) -> dict[str, object]:
    if not lags:
        raise ValueError("at least one lag is required")
    if any(lag < 1 for lag in lags):
        raise ValueError("lags must be positive")

    rows, fieldnames = _read_rows(input_csv)
    feature_names = _numeric_feature_names(rows, fieldnames)
    ordered_lags = sorted(set(lags))
    output_rows = [dict(row) for row in rows]
    row_positions = _group_positions(rows, group_field=group_field, time_field=time_field)
    added_fields = [f"lag{lag}_{name}" for lag in ordered_lags for name in feature_names]

    for row_index, row in enumerate(output_rows):
        group_positions = row_positions.get(row.get(group_field, ""), [])
        local_index = group_positions.index(row_index) if row_index in group_positions else -1
        available = 0
        for lag in ordered_lags:
            source_index = group_positions[local_index - lag] if local_index >= lag else None
            if source_index is not None:
                available += 1
            for name in feature_names:
                value = _number(rows[source_index].get(name, "")) if source_index is not None else 0.0
                row[f"lag{lag}_{name}"] = f"{value:.9f}"
        row["lag_context_available_count"] = str(available)

    out_fieldnames = list(fieldnames)
    if "lag_context_available_count" not in out_fieldnames:
        out_fieldnames.append("lag_context_available_count")
    out_fieldnames.extend(field for field in added_fields if field not in out_fieldnames)
    _write_rows(out_csv, output_rows, out_fieldnames)

    report = {
        "schema": "elfquake.synthetic_lagged_context.v1",
        "input_csv": str(input_csv),
        "out_csv": str(out_csv),
        "row_count": len(rows),
        "group_field": group_field,
        "time_field": time_field,
        "lags": ordered_lags,
        "base_feature_count": len(feature_names),
        "added_feature_count": len(added_fields) + 1,
        "note": "Lagged context excludes target and diagnostic fields to avoid target leakage.",
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def _read_rows(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        return list(reader), list(reader.fieldnames or [])


def _numeric_feature_names(rows: list[dict[str, str]], fieldnames: list[str]) -> list[str]:
    names: list[str] = []
    for name in fieldnames:
        if is_diagnostic_or_target_field(name):
            continue
        values = [_number(row.get(name, "")) for row in rows]
        if any(value != 0.0 for value in values):
            names.append(name)
    return names


def _group_positions(
    rows: list[dict[str, str]],
    *,
    group_field: str,
    time_field: str,
) -> dict[str, list[int]]:
    groups: dict[str, list[int]] = {}
    for index, row in enumerate(rows):
        groups.setdefault(row.get(group_field, ""), []).append(index)
    for group_id, positions in groups.items():
        groups[group_id] = sorted(positions, key=lambda index: rows[index].get(time_field, ""))
    return groups


def _write_rows(path: Path, rows: list[dict[str, str]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)

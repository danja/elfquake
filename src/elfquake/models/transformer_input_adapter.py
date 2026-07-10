"""Adapters that make richer target tables usable by Transformer trainers."""

from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path


def prepare_transformer_target_input(
    *,
    input_csv: Path,
    out_csv: Path,
    report_path: Path,
    target_field: str = "eventlist_target_occurred",
    target_status_field: str = "eventlist_target_status",
    standard_target_field: str = "target_occurred",
    standard_status_field: str = "target_status",
    split_field: str = "model_split",
    group_field: str = "dataset_id",
    time_field: str = "window_start_utc",
    train_fraction: float = 0.8,
) -> dict[str, object]:
    """Copy a named binary target into the standard Transformer target fields."""

    if not 0 < train_fraction < 1:
        raise ValueError("train_fraction must be between 0 and 1")

    rows, fieldnames = _read_rows_and_fields(input_csv)
    output_rows = [dict(row) for row in rows]
    _copy_target_fields(
        output_rows,
        target_field=target_field,
        target_status_field=target_status_field,
        standard_target_field=standard_target_field,
        standard_status_field=standard_status_field,
    )
    labeled = [row for row in output_rows if row.get(standard_target_field) in {"0", "1"}]
    _assign_temporal_group_split(
        output_rows,
        labeled,
        split_field=split_field,
        group_field=group_field,
        time_field=time_field,
        train_fraction=train_fraction,
    )

    out_fields = _field_order(
        fieldnames,
        extra_fields=[standard_target_field, standard_status_field, split_field],
    )
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=out_fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(output_rows)

    train_rows = [row for row in output_rows if row.get(split_field) == "train"]
    test_rows = [row for row in output_rows if row.get(split_field) == "test"]
    report = {
        "schema": "elfquake.transformer_target_input.v1",
        "input": str(input_csv),
        "output": str(out_csv),
        "row_count": len(output_rows),
        "labeled_row_count": len(labeled),
        "target_field": target_field,
        "target_status_field": target_status_field,
        "standard_target_field": standard_target_field,
        "standard_status_field": standard_status_field,
        "split_field": split_field,
        "group_field": group_field,
        "time_field": time_field,
        "train_fraction": train_fraction,
        "train_row_count": len(train_rows),
        "test_row_count": len(test_rows),
        "train_positive_count": _positive_count(train_rows, standard_target_field),
        "train_negative_count": _negative_count(train_rows, standard_target_field),
        "test_positive_count": _positive_count(test_rows, standard_target_field),
        "test_negative_count": _negative_count(test_rows, standard_target_field),
        "target_counts": dict(Counter(row.get(standard_target_field, "") for row in output_rows)),
        "split_counts": dict(Counter(row.get(split_field, "") for row in output_rows)),
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def _copy_target_fields(
    rows: list[dict[str, str]],
    *,
    target_field: str,
    target_status_field: str,
    standard_target_field: str,
    standard_status_field: str,
) -> None:
    for row in rows:
        label = row.get(target_field, "")
        row[standard_target_field] = label if label in {"0", "1"} else ""
        source_status = row.get(target_status_field, "")
        if source_status:
            row[standard_status_field] = source_status
        elif row[standard_target_field]:
            row[standard_status_field] = "labeled"
        else:
            row[standard_status_field] = "unlabeled"


def _assign_temporal_group_split(
    rows: list[dict[str, str]],
    labeled_rows: list[dict[str, str]],
    *,
    split_field: str,
    group_field: str,
    time_field: str,
    train_fraction: float,
) -> None:
    for row in rows:
        row[split_field] = ""
    grouped: dict[str, list[dict[str, str]]] = {}
    for row in labeled_rows:
        grouped.setdefault(row.get(group_field, ""), []).append(row)
    for group_rows in grouped.values():
        ordered = sorted(group_rows, key=lambda row: row.get(time_field, ""))
        if len(ordered) < 2:
            for row in ordered:
                row[split_field] = "train"
            continue
        split_at = max(1, min(len(ordered) - 1, int(len(ordered) * train_fraction)))
        for index, row in enumerate(ordered):
            row[split_field] = "train" if index < split_at else "test"


def _field_order(fieldnames: list[str], *, extra_fields: list[str]) -> list[str]:
    return [field for field in fieldnames if field not in set(extra_fields)] + extra_fields


def _positive_count(rows: list[dict[str, str]], target_field: str) -> int:
    return sum(1 for row in rows if row.get(target_field) == "1")


def _negative_count(rows: list[dict[str, str]], target_field: str) -> int:
    return sum(1 for row in rows if row.get(target_field) == "0")


def _read_rows_and_fields(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        return list(reader), list(reader.fieldnames or [])

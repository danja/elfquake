"""Synthetic split helpers for burn-in and regime-aware evaluation."""

from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from pathlib import Path


REGIME_FIELDS = [
    "synthetic_row_index",
    "synthetic_group_row_count",
    "synthetic_regime_index",
    "synthetic_regime_id",
    "synthetic_burn_in",
]


def annotate_synthetic_regimes(
    *,
    input_csv: Path,
    out_csv: Path,
    report_path: Path,
    group_field: str = "dataset_id",
    time_field: str = "window_start_utc",
    regime_count: int = 5,
    burn_in_fraction: float = 0.2,
    drop_burn_in: bool = False,
) -> dict[str, object]:
    if regime_count < 1:
        raise ValueError("regime_count must be at least 1")
    if not 0 <= burn_in_fraction < 1:
        raise ValueError("burn_in_fraction must be in [0, 1)")

    rows, fieldnames = _read_rows_and_fields(input_csv)
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[row.get(group_field, "")].append(row)

    annotated = []
    group_summaries = []
    for group_id in sorted(grouped):
        group_rows = sorted(grouped[group_id], key=lambda row: row.get(time_field, ""))
        burn_in_count = int(len(group_rows) * burn_in_fraction)
        kept = 0
        for index, row in enumerate(group_rows):
            regime_index = min(regime_count - 1, int(index * regime_count / max(1, len(group_rows))))
            is_burn_in = index < burn_in_count
            output = dict(row)
            output.update(
                {
                    "synthetic_row_index": str(index),
                    "synthetic_group_row_count": str(len(group_rows)),
                    "synthetic_regime_index": str(regime_index),
                    "synthetic_regime_id": f"{group_id}_r{regime_index}",
                    "synthetic_burn_in": "1" if is_burn_in else "0",
                }
            )
            if not (drop_burn_in and is_burn_in):
                annotated.append(output)
                kept += 1
        group_summaries.append(
            {
                "group_id": group_id,
                "row_count": len(group_rows),
                "burn_in_count": burn_in_count,
                "kept_count": kept,
            }
        )

    out_fields = [field for field in fieldnames if field not in REGIME_FIELDS] + REGIME_FIELDS
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=out_fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(annotated)

    report = {
        "schema": "elfquake.synthetic_regimes.v1",
        "input": str(input_csv),
        "output": str(out_csv),
        "row_count": len(rows),
        "output_row_count": len(annotated),
        "group_field": group_field,
        "time_field": time_field,
        "regime_count": regime_count,
        "burn_in_fraction": burn_in_fraction,
        "drop_burn_in": drop_burn_in,
        "groups": group_summaries,
        "regime_ids": sorted({row["synthetic_regime_id"] for row in annotated}),
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def assign_balanced_split(
    *,
    input_csv: Path,
    out_csv: Path,
    report_path: Path,
    group_field: str = "synthetic_regime_id",
    target_field: str = "target_occurred",
    time_field: str = "window_start_utc",
    split_field: str = "model_split",
    test_fraction: float = 0.2,
) -> dict[str, object]:
    if not 0 < test_fraction < 1:
        raise ValueError("test_fraction must be between 0 and 1")
    rows, fieldnames = _read_rows_and_fields(input_csv)
    grouped: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    unlabeled = []
    for row in rows:
        label = row.get(target_field, "")
        if label not in {"0", "1"}:
            unlabeled.append(row)
            continue
        grouped[(row.get(group_field, ""), label)].append(row)

    test_ids: set[int] = set()
    bucket_summaries = []
    for (group_id, label), bucket in sorted(grouped.items()):
        ordered = sorted(bucket, key=lambda row: row.get(time_field, ""))
        test_indexes = _even_test_indexes(len(ordered), test_fraction)
        for index in test_indexes:
            test_ids.add(id(ordered[index]))
        bucket_summaries.append(
            {
                "group_id": group_id,
                "target": label,
                "row_count": len(ordered),
                "train_count": len(ordered) - len(test_indexes),
                "test_count": len(test_indexes),
            }
        )

    output_rows = []
    for row in rows:
        output = dict(row)
        if row in unlabeled:
            output[split_field] = ""
        else:
            output[split_field] = "test" if id(row) in test_ids else "train"
        output_rows.append(output)

    out_fields = [field for field in fieldnames if field != split_field] + [split_field]
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=out_fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(output_rows)

    train_rows = [row for row in output_rows if row.get(split_field) == "train"]
    test_rows = [row for row in output_rows if row.get(split_field) == "test"]
    report = {
        "schema": "elfquake.synthetic_balanced_split.v1",
        "input": str(input_csv),
        "output": str(out_csv),
        "row_count": len(rows),
        "labeled_row_count": len(rows) - len(unlabeled),
        "group_field": group_field,
        "target_field": target_field,
        "time_field": time_field,
        "split_field": split_field,
        "test_fraction": test_fraction,
        "train_row_count": len(train_rows),
        "test_row_count": len(test_rows),
        "train_positive_count": _positive_count(train_rows, target_field),
        "train_negative_count": _negative_count(train_rows, target_field),
        "test_positive_count": _positive_count(test_rows, target_field),
        "test_negative_count": _negative_count(test_rows, target_field),
        "buckets": bucket_summaries,
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def _even_test_indexes(count: int, test_fraction: float) -> set[int]:
    if count < 2:
        return set()
    test_count = min(count - 1, max(1, int(round(count * test_fraction))))
    indexes = {min(count - 1, int(math.floor((index + 0.5) * count / test_count))) for index in range(test_count)}
    while len(indexes) < test_count:
        for index in range(count - 1, -1, -1):
            indexes.add(index)
            if len(indexes) == test_count:
                break
    return indexes


def _positive_count(rows: list[dict[str, str]], target_field: str) -> int:
    return sum(1 for row in rows if row.get(target_field) == "1")


def _negative_count(rows: list[dict[str, str]], target_field: str) -> int:
    return sum(1 for row in rows if row.get(target_field) == "0")


def _read_rows_and_fields(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        return list(reader), list(reader.fieldnames or [])

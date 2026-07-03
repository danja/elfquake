"""Combine aligned model-row CSVs while preserving dataset provenance."""

from __future__ import annotations

import csv
from pathlib import Path


def combine_aligned_datasets(
    *,
    input_csvs: list[Path],
    out_path: Path,
    dataset_ids: list[str] | None = None,
) -> list[dict[str, str]]:
    if not input_csvs:
        raise ValueError("at least one input CSV is required")
    if dataset_ids is not None and len(dataset_ids) != len(input_csvs):
        raise ValueError("dataset_ids length must match input_csvs length")

    rows_by_input = [_read_rows_and_fields(path) for path in input_csvs]
    fieldnames = _fieldnames(rows_by_input)
    output_rows = []
    for index, (path, (rows, _fields)) in enumerate(zip(input_csvs, rows_by_input)):
        dataset_id = dataset_ids[index] if dataset_ids else path.stem
        for row in rows:
            output = {"dataset_id": dataset_id}
            output.update(row)
            output_rows.append({field: output.get(field, "") for field in fieldnames})

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(output_rows)
    return output_rows


def _read_rows_and_fields(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        return list(reader), list(reader.fieldnames or [])


def _fieldnames(rows_by_input: list[tuple[list[dict[str, str]], list[str]]]) -> list[str]:
    fields = ["dataset_id"]
    for _rows, input_fields in rows_by_input:
        for field in input_fields:
            if field not in fields:
                fields.append(field)
    return fields

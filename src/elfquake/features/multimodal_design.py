"""Join source-specific feature tables into model design matrices."""

from __future__ import annotations

import csv
from pathlib import Path


def join_vlf_design_matrix(
    *,
    design_matrix_csv: Path,
    vlf_windows_csv: Path,
    out_path: Path,
) -> list[dict[str, str]]:
    base_rows, base_fields = _read_rows_and_fields(design_matrix_csv)
    vlf_rows, vlf_fields = _read_rows_and_fields(vlf_windows_csv)
    vlf_by_window = {row["window_id"]: row for row in vlf_rows}
    added_fields = [
        field
        for field in vlf_fields
        if field not in base_fields and field not in {"window_start_utc", "window_end_utc"}
    ]
    fieldnames = base_fields + added_fields

    rows = []
    for row in base_rows:
        merged = dict(row)
        vlf = vlf_by_window.get(row["window_id"], {})
        for field in added_fields:
            merged[field] = vlf.get(field, "")
        rows.append(merged)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    return rows


def _read_rows_and_fields(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        return list(reader), list(reader.fieldnames or [])

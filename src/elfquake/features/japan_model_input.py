"""Build the research-only Japan model input table."""

from __future__ import annotations

import csv
from pathlib import Path


def build_japan_model_input(
    *, windows_csv: Path, vlf_windows_csv: Path, out_path: Path
) -> list[dict[str, str]]:
    """Join Japan seismic windows to native-CDF VLF features by ``window_id``.

    Every seismic window is retained. Missing VLF coverage is represented by
    empty feature values and an explicit quality flag for downstream masking.
    """
    windows, window_fields = _read(windows_csv)
    vlf_rows, vlf_fields = _read(vlf_windows_csv)
    vlf_by_window = {row["window_id"]: row for row in vlf_rows}
    added_fields = [
        field
        for field in vlf_fields
        if field not in window_fields
        and field not in {"window_start_utc", "window_end_utc", "region_id"}
    ]
    fieldnames = window_fields + added_fields + ["quality_missing_japan_vlf"]
    rows = []
    for window in windows:
        vlf = vlf_by_window.get(window["window_id"], {})
        merged = dict(window)
        for field in added_fields:
            merged[field] = vlf.get(field, "")
        merged["quality_missing_japan_vlf"] = "1" if _missing(vlf) else "0"
        rows.append(merged)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    return rows


def _read(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        return list(reader), list(reader.fieldnames or [])


def _missing(row: dict[str, str]) -> bool:
    return not row or not row.get("japan_vlf_row_count", "").strip() or row.get("japan_vlf_row_count") == "0"

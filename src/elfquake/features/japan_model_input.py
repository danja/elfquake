"""Build the research-only Japan model input table."""

from __future__ import annotations

import csv
import hashlib
from pathlib import Path


def build_japan_model_input(
    *, windows_csv: Path, vlf_windows_csv: Path, out_path: Path, dataset_id: str = "japan_moshiri"
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
    fieldnames = [field for field in window_fields if field != "dataset_id"]
    fieldnames.insert(0, "dataset_id")
    fieldnames += added_fields + ["quality_missing_japan_vlf"]
    rows = []
    for window in windows:
        vlf = vlf_by_window.get(window["window_id"], {})
        merged = dict(window)
        merged["dataset_id"] = _compact_dataset_id(vlf.get("japan_vlf_source_dataset_id", ""), dataset_id)
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


def _compact_dataset_id(source_dataset_id: str, fallback: str) -> str:
    """Keep dataset_id filesystem-safe when a window is covered by many CDF files.

    ``japan_vlf_source_dataset_id`` concatenates every contributing hourly CDF
    name (``multiple:a+b+c...``); as archive coverage densifies this can exceed
    filesystem path-component limits when used to name a sequence directory.
    The full concatenation is preserved separately as a feature column, so
    only the value used as ``dataset_id`` needs to stay short and stable.
    """
    if not source_dataset_id:
        return fallback
    if not source_dataset_id.startswith("multiple:"):
        return source_dataset_id
    digest = hashlib.sha1(source_dataset_id.encode("utf-8")).hexdigest()[:10]
    return f"japan_moshiri_multi_{digest}"

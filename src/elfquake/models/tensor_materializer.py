"""Materialize backend-neutral tensor tables from tensor specs."""

from __future__ import annotations

import csv
import json
from pathlib import Path


def materialize_tensor_dataset(
    *,
    spec_path: Path,
    out_dir: Path,
    fill_value: float = 0.0,
) -> dict[str, object]:
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    input_csv = Path(str(spec["input_csv"]))
    feature_fields = [str(field) for field in spec.get("numeric_feature_fields", [])]
    if not feature_fields:
        raise ValueError("tensor spec has no numeric_feature_fields")

    rows, fieldnames = _read_rows_and_fields(input_csv)
    out_dir.mkdir(parents=True, exist_ok=True)
    values_path = out_dir / "values.csv"
    masks_path = out_dir / "masks.csv"
    index_path = out_dir / "index.csv"
    manifest_path = out_dir / "manifest.json"

    row_index_field = "row_index"
    mask_fields = [f"{field}__present" for field in feature_fields]
    _write_values(values_path, rows, feature_fields, row_index_field=row_index_field, fill_value=fill_value)
    _write_masks(masks_path, rows, feature_fields, mask_fields, row_index_field=row_index_field)
    index_fields = _index_fields(spec, fieldnames)
    _write_index(index_path, rows, index_fields, row_index_field=row_index_field)

    manifest: dict[str, object] = {
        "schema": "elfquake.tensor_dataset.v1",
        "spec_file": str(spec_path),
        "input_csv": str(input_csv),
        "row_count": len(rows),
        "feature_count": len(feature_fields),
        "layout": spec.get("recommended_layout", "batch,time,channel"),
        "fill_value": fill_value,
        "values_csv": str(values_path),
        "masks_csv": str(masks_path),
        "index_csv": str(index_path),
        "feature_fields": feature_fields,
        "mask_fields": mask_fields,
        "modalities": spec.get("modalities", {}),
        "time_field": spec.get("time_field", ""),
        "region_field": spec.get("region_field", ""),
        "target_field": spec.get("target_field", ""),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def _read_rows_and_fields(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        return list(reader), list(reader.fieldnames or [])


def _write_values(
    path: Path,
    rows: list[dict[str, str]],
    feature_fields: list[str],
    *,
    row_index_field: str,
    fill_value: float,
) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=[row_index_field] + feature_fields, lineterminator="\n")
        writer.writeheader()
        for row_index, row in enumerate(rows):
            output = {row_index_field: str(row_index)}
            for field in feature_fields:
                output[field] = _numeric_or_fill(row.get(field, ""), fill_value)
            writer.writerow(output)


def _write_masks(
    path: Path,
    rows: list[dict[str, str]],
    feature_fields: list[str],
    mask_fields: list[str],
    *,
    row_index_field: str,
) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=[row_index_field] + mask_fields, lineterminator="\n")
        writer.writeheader()
        for row_index, row in enumerate(rows):
            output = {row_index_field: str(row_index)}
            for field, mask_field in zip(feature_fields, mask_fields):
                output[mask_field] = "1" if row.get(field, "") != "" else "0"
            writer.writerow(output)


def _write_index(
    path: Path,
    rows: list[dict[str, str]],
    index_fields: list[str],
    *,
    row_index_field: str,
) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=[row_index_field] + index_fields, lineterminator="\n")
        writer.writeheader()
        for row_index, row in enumerate(rows):
            output = {row_index_field: str(row_index)}
            for field in index_fields:
                output[field] = row.get(field, "")
            writer.writerow(output)


def _index_fields(spec: dict[str, object], fieldnames: list[str]) -> list[str]:
    requested = [
        str(spec.get("time_field", "")),
        str(spec.get("region_field", "")),
        str(spec.get("target_field", "")),
        *[str(field) for field in spec.get("id_fields", [])],
        *[str(field) for field in spec.get("target_fields", [])],
    ]
    fields: list[str] = []
    for field in requested:
        if field and field in fieldnames and field not in fields:
            fields.append(field)
    return fields


def _numeric_or_fill(value: str, fill_value: float) -> str:
    if value == "":
        return f"{fill_value:.9f}"
    return f"{float(value):.9f}"

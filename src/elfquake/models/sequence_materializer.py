"""Materialize backend-neutral sequence tables from time-series CSVs."""

from __future__ import annotations

import csv
import json
from datetime import timedelta
from pathlib import Path

from elfquake.features.common import format_utc, parse_utc
from elfquake.models.readiness import ID_FIELDS, TARGET_FIELDS


DEFAULT_EXCLUDED_FIELDS = ID_FIELDS | TARGET_FIELDS | {
    "time_index",
    "step",
    "sensor_id",
    "entity_id",
    "x",
    "y",
    "source_file",
}


def materialize_sequence_dataset(
    *,
    input_csv: Path,
    out_dir: Path,
    time_field: str = "step",
    entity_field: str | None = "sensor_id",
    fill_value: float = 0.0,
    modality: str = "simulation",
    time_start_utc: str | None = None,
    time_step_seconds: int | None = None,
) -> dict[str, object]:
    rows, fieldnames = _read_rows_and_fields(input_csv)
    if time_field not in fieldnames:
        raise ValueError(f"time_field not found: {time_field}")
    if entity_field and entity_field not in fieldnames:
        raise ValueError(f"entity_field not found: {entity_field}")
    if time_step_seconds is not None and time_step_seconds < 1:
        raise ValueError("time_step_seconds must be at least 1")
    if time_start_utc and time_step_seconds is None:
        raise ValueError("time_step_seconds is required when time_start_utc is set")

    channel_fields = [
        field
        for field in fieldnames
        if field not in DEFAULT_EXCLUDED_FIELDS
        and field != time_field
        and field != entity_field
        and _is_numeric_column(rows, field)
    ]
    if not channel_fields:
        raise ValueError("input_csv has no numeric channel fields")

    sorted_rows = sorted(
        rows,
        key=lambda row: (
            _sort_key(row.get(time_field, "")),
            _sort_key(row.get(entity_field, "global") if entity_field else "global"),
        ),
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    values_path = out_dir / "values.csv"
    masks_path = out_dir / "masks.csv"
    index_path = out_dir / "index.csv"
    time_axis_path = out_dir / "time_axis.csv"
    entity_axis_path = out_dir / "entity_axis.csv"
    manifest_path = out_dir / "manifest.json"

    time_values = _unique([row.get(time_field, "") for row in sorted_rows])
    entity_values = _unique([row.get(entity_field, "global") if entity_field else "global" for row in sorted_rows])
    time_index = {value: str(index) for index, value in enumerate(time_values)}
    entity_index = {value: str(index) for index, value in enumerate(entity_values)}
    mask_fields = [f"{field}__present" for field in channel_fields]

    _write_sequence_values(
        values_path,
        sorted_rows,
        channel_fields,
        time_field=time_field,
        entity_field=entity_field,
        time_index=time_index,
        entity_index=entity_index,
        fill_value=fill_value,
    )
    _write_sequence_masks(
        masks_path,
        sorted_rows,
        channel_fields,
        mask_fields,
        time_field=time_field,
        entity_field=entity_field,
        time_index=time_index,
        entity_index=entity_index,
    )
    _write_sequence_index(
        index_path,
        sorted_rows,
        time_field=time_field,
        entity_field=entity_field,
        time_index=time_index,
        entity_index=entity_index,
    )
    time_mapping = _time_mapping(time_values, time_start_utc=time_start_utc, time_step_seconds=time_step_seconds)
    _write_time_axis(time_axis_path, "time_index", time_field, time_values, utc_values=time_mapping["utc_values"])
    _write_axis(entity_axis_path, "entity_index", entity_field or "entity_id", entity_values)

    manifest: dict[str, object] = {
        "schema": "elfquake.sequence_dataset.v1",
        "input_csv": str(input_csv),
        "row_count": len(sorted_rows),
        "time_count": len(time_values),
        "entity_count": len(entity_values),
        "channel_count": len(channel_fields),
        "layout": "row,time,entity,channel",
        "time_field": time_field,
        "entity_field": entity_field or "",
        "modality": modality,
        "fill_value": fill_value,
        "values_csv": str(values_path),
        "masks_csv": str(masks_path),
        "index_csv": str(index_path),
        "time_axis_csv": str(time_axis_path),
        "entity_axis_csv": str(entity_axis_path),
        "channel_fields": channel_fields,
        "mask_fields": mask_fields,
        "time_mapping": {
            "time_start_utc": time_start_utc or "",
            "time_step_seconds": time_step_seconds or "",
            "utc_axis_field": "time_utc" if time_mapping["utc_values"] else "",
            "assumption": "synthetic simulation time mapping" if time_mapping["utc_values"] else "",
        },
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def _read_rows_and_fields(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        return list(reader), list(reader.fieldnames or [])


def _write_sequence_values(
    path: Path,
    rows: list[dict[str, str]],
    channel_fields: list[str],
    *,
    time_field: str,
    entity_field: str | None,
    time_index: dict[str, str],
    entity_index: dict[str, str],
    fill_value: float,
) -> None:
    fieldnames = ["row_index", "time_index", "entity_index"] + channel_fields
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for row_index, row in enumerate(rows):
            output = _axis_row(row, row_index, time_field, entity_field, time_index, entity_index)
            for field in channel_fields:
                output[field] = _numeric_or_fill(row.get(field, ""), fill_value)
            writer.writerow(output)


def _write_sequence_masks(
    path: Path,
    rows: list[dict[str, str]],
    channel_fields: list[str],
    mask_fields: list[str],
    *,
    time_field: str,
    entity_field: str | None,
    time_index: dict[str, str],
    entity_index: dict[str, str],
) -> None:
    fieldnames = ["row_index", "time_index", "entity_index"] + mask_fields
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for row_index, row in enumerate(rows):
            output = _axis_row(row, row_index, time_field, entity_field, time_index, entity_index)
            for field, mask_field in zip(channel_fields, mask_fields):
                output[mask_field] = "1" if row.get(field, "") != "" else "0"
            writer.writerow(output)


def _write_sequence_index(
    path: Path,
    rows: list[dict[str, str]],
    *,
    time_field: str,
    entity_field: str | None,
    time_index: dict[str, str],
    entity_index: dict[str, str],
) -> None:
    fieldnames = ["row_index", "time_index", "entity_index", time_field]
    if entity_field:
        fieldnames.append(entity_field)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for row_index, row in enumerate(rows):
            output = _axis_row(row, row_index, time_field, entity_field, time_index, entity_index)
            output[time_field] = row.get(time_field, "")
            if entity_field:
                output[entity_field] = row.get(entity_field, "")
            writer.writerow(output)


def _write_axis(path: Path, index_field: str, value_field: str, values: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=[index_field, value_field], lineterminator="\n")
        writer.writeheader()
        for index, value in enumerate(values):
            writer.writerow({index_field: str(index), value_field: value})


def _write_time_axis(
    path: Path,
    index_field: str,
    value_field: str,
    values: list[str],
    *,
    utc_values: list[str],
) -> None:
    fieldnames = [index_field, value_field]
    if utc_values:
        fieldnames.append("time_utc")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for index, value in enumerate(values):
            row = {index_field: str(index), value_field: value}
            if utc_values:
                row["time_utc"] = utc_values[index]
            writer.writerow(row)


def _time_mapping(
    time_values: list[str],
    *,
    time_start_utc: str | None,
    time_step_seconds: int | None,
) -> dict[str, list[str]]:
    if not time_start_utc:
        return {"utc_values": []}
    assert time_step_seconds is not None
    start = parse_utc(time_start_utc)
    utc_values = []
    for value in time_values:
        utc_values.append(format_utc(start + timedelta(seconds=float(value) * time_step_seconds)))
    return {"utc_values": utc_values}


def _axis_row(
    row: dict[str, str],
    row_index: int,
    time_field: str,
    entity_field: str | None,
    time_index: dict[str, str],
    entity_index: dict[str, str],
) -> dict[str, str]:
    entity_value = row.get(entity_field, "global") if entity_field else "global"
    return {
        "row_index": str(row_index),
        "time_index": time_index[row.get(time_field, "")],
        "entity_index": entity_index[entity_value],
    }


def _unique(values: list[str]) -> list[str]:
    seen = set()
    result = []
    for value in values:
        if value not in seen:
            seen.add(value)
            result.append(value)
    return result


def _sort_key(value: str) -> tuple[int, float | str]:
    try:
        return (0, float(value))
    except ValueError:
        return (1, value)


def _is_numeric_column(rows: list[dict[str, str]], field: str) -> bool:
    values = [row.get(field, "") for row in rows]
    present = [value for value in values if value != ""]
    if not present:
        return False
    return all(_is_float(value) for value in present)


def _is_float(value: str) -> bool:
    try:
        float(value)
        return True
    except ValueError:
        return False


def _numeric_or_fill(value: str, fill_value: float) -> str:
    if value == "":
        return f"{fill_value:.9f}"
    return f"{float(value):.9f}"

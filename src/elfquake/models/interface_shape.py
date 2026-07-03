"""Audit derived data shapes against model-input interfaces."""

from __future__ import annotations

import csv
import json
from pathlib import Path

from elfquake.models.tensor_spec import MODALITY_PREFIXES


def audit_model_interfaces(*, input_paths: list[Path], out_path: Path) -> dict[str, object]:
    tables = [_summarize_table(path) for path in input_paths]
    report: dict[str, object] = {
        "schema": "elfquake.model_interface_shape.v1",
        "table_count": len(tables),
        "tables": tables,
        "interface_summary": _interface_summary(tables),
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def _summarize_table(path: Path) -> dict[str, object]:
    rows, fields = _read_rows_and_fields(path)
    numeric_fields = [field for field in fields if _is_numeric_column(rows, field)]
    missing_cells = sum(1 for row in rows for field in fields if row.get(field, "") == "")
    modality_groups = {
        modality: _modality_fields(numeric_fields, prefixes)
        for modality, prefixes in MODALITY_PREFIXES.items()
    }
    modality_groups = {
        modality: fields
        for modality, fields in modality_groups.items()
        if fields
    }
    shape_kind = _shape_kind(fields)
    return {
        "path": str(path),
        "row_count": len(rows),
        "field_count": len(fields),
        "numeric_field_count": len(numeric_fields),
        "missing_cell_count": missing_cells,
        "shape_kind": shape_kind,
        "axis_roles": _axis_roles(shape_kind, fields),
        "logical_modality": _logical_modality(shape_kind, rows),
        "modality_numeric_feature_counts": {
            modality: len(fields)
            for modality, fields in modality_groups.items()
        },
        "modality_numeric_features": modality_groups,
        "interface_fit": _interface_fit(shape_kind),
        "supports_missing_masks": bool(numeric_fields),
        "ablation_note": _ablation_note(shape_kind, modality_groups),
    }


def _read_rows_and_fields(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        return list(reader), list(reader.fieldnames or [])


def _modality_fields(numeric_fields: list[str], prefixes: tuple[str, ...]) -> list[str]:
    return [field for field in numeric_fields if field.startswith(prefixes)]


def _shape_kind(fields: list[str]) -> str:
    field_set = set(fields)
    if {"event_time_utc", "latitude", "longitude", "magnitude"}.issubset(field_set):
        return "event_list"
    if {"step", "sensor_id"}.issubset(field_set):
        return "sensor_time_series"
    if "step" in field_set:
        return "summary_time_series"
    if "window_start_utc" in field_set and "window_end_utc" in field_set:
        return "window_feature_table"
    if "vlf_image_source_file" in field_set:
        return "image_feature_table"
    return "feature_table"


def _axis_roles(shape_kind: str, fields: list[str]) -> dict[str, str]:
    if shape_kind == "sensor_time_series":
        return {"time": "step", "entity": "sensor_id", "channel": "numeric fields excluding ids"}
    if shape_kind == "summary_time_series":
        return {"time": "step", "channel": "numeric fields excluding step"}
    if shape_kind == "event_list":
        return {
            "time": "event_time_utc",
            "space": "latitude,longitude,depth_km",
            "mark": "magnitude and event metadata",
        }
    if shape_kind == "window_feature_table":
        roles = {"sample": "window_id", "time": "window_start_utc,window_end_utc", "channel": "numeric feature fields"}
        if "region_id" in fields:
            roles["region"] = "region_id"
        return roles
    if shape_kind == "image_feature_table":
        return {"sample": "vlf_image_source_file", "channel": "numeric image feature fields"}
    return {"sample": "row", "channel": "numeric feature fields"}


def _logical_modality(shape_kind: str, rows: list[dict[str, str]]) -> str:
    if shape_kind == "event_list":
        sources = {row.get("source", "") for row in rows}
        if any("synthetic" in source or "avalanche" in source for source in sources):
            return "synthetic_seismic"
        return "seismic"
    if shape_kind == "sensor_time_series":
        return "simulation"
    if shape_kind == "summary_time_series":
        return "simulation"
    if shape_kind == "image_feature_table":
        return "vlf_image"
    return "multimodal_or_tabular"


def _interface_fit(shape_kind: str) -> str:
    if shape_kind in {"window_feature_table", "image_feature_table", "feature_table"}:
        return "compatible_with_current_values_masks_index_materializer"
    if shape_kind == "event_list":
        return "aggregate_to_regular_windows_or_use_event_process_adapter"
    if shape_kind in {"sensor_time_series", "summary_time_series"}:
        return "needs_sequence_materializer_with_time_entity_channel_axes"
    return "needs_manual_review"


def _ablation_note(shape_kind: str, modality_groups: dict[str, list[str]]) -> str:
    if shape_kind in {"sensor_time_series", "summary_time_series"}:
        return "ablate by simulated signal family after sequence materialization"
    if shape_kind == "event_list":
        return "ablate after conversion to seismic window features or event-process context"
    if modality_groups:
        return "ablate by modality prefix groups and matching present-mask channels"
    return "no prefixed modality feature groups detected"


def _interface_summary(tables: list[dict[str, object]]) -> dict[str, object]:
    shape_counts: dict[str, int] = {}
    modality_counts: dict[str, int] = {}
    for table in tables:
        shape_kind = str(table["shape_kind"])
        modality = str(table["logical_modality"])
        shape_counts[shape_kind] = shape_counts.get(shape_kind, 0) + 1
        modality_counts[modality] = modality_counts.get(modality, 0) + 1
    needs_sequence = [
        table["path"]
        for table in tables
        if table["shape_kind"] in {"sensor_time_series", "summary_time_series"}
    ]
    needs_windowing = [
        table["path"]
        for table in tables
        if table["shape_kind"] == "event_list"
    ]
    return {
        "shape_counts": shape_counts,
        "logical_modality_counts": modality_counts,
        "needs_sequence_materializer": needs_sequence,
        "needs_window_aggregation": needs_windowing,
    }


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

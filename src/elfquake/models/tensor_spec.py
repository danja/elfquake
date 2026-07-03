"""Regular-cadence tensor specifications for model candidates."""

from __future__ import annotations

import csv
import json
from pathlib import Path

from elfquake.models.readiness import ID_FIELDS, TARGET_FIELDS


MODALITY_PREFIXES: dict[str, tuple[str, ...]] = {
    "seismic": ("seismic_",),
    "vlf_metadata": ("vlf_capture_", "vlf_latest_", "vlf_total_", "vlf_jpeg_"),
    "vlf_image": (
        "vlf_image_",
        "vlf_intensity_",
        "vlf_high_",
        "vlf_hot_",
        "vlf_column_",
        "vlf_vertical_",
        "vlf_band_",
        "vlf_pixel_",
        "vlf_crop_",
    ),
    "astronomy": ("astro_", "astronomy_", "moon_", "solar_", "kp_", "ap_", "f107_"),
    "synthetic_seismic": ("synthetic_seismic_",),
    "synthetic_piezo_vlf": ("synthetic_piezo_vlf_",),
    "synthetic_direct_avalanche": ("synthetic_direct_avalanche_",),
    "synthetic_summary": ("synthetic_summary_",),
    "simulation": ("piezo_", "avalanche_", "sim_"),
    "quality": ("quality_", "coverage_", "missing_"),
}


def build_tensor_spec(
    *,
    input_csv: Path,
    out_path: Path,
    time_field: str = "window_start_utc",
    region_field: str = "region_id",
    target_field: str = "target_occurred",
) -> dict[str, object]:
    rows, fieldnames = _read_rows_and_fields(input_csv)
    feature_fields = [
        field
        for field in fieldnames
        if field not in ID_FIELDS and field not in TARGET_FIELDS and _is_numeric_column(rows, field)
    ]
    modality_groups = {
        modality: _modality_summary(rows, feature_fields, prefixes)
        for modality, prefixes in MODALITY_PREFIXES.items()
    }
    unassigned = [
        field
        for field in feature_fields
        if not any(field in group["feature_fields"] for group in modality_groups.values())
    ]
    spec: dict[str, object] = {
        "schema": "elfquake.tensor_spec.v1",
        "input_csv": str(input_csv),
        "row_count": len(rows),
        "field_count": len(fieldnames),
        "time_field": time_field,
        "region_field": region_field,
        "target_field": target_field,
        "id_fields": [field for field in fieldnames if field in ID_FIELDS],
        "target_fields": [field for field in fieldnames if field in TARGET_FIELDS],
        "numeric_feature_count": len(feature_fields),
        "numeric_feature_fields": feature_fields,
        "modalities": modality_groups,
        "unassigned_numeric_features": unassigned,
        "recommended_layout": "batch,time,channel",
        "mask_convention": "one binary present-mask channel per numeric feature",
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(spec, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return spec


def _read_rows_and_fields(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        return list(reader), list(reader.fieldnames or [])


def _modality_summary(
    rows: list[dict[str, str]],
    feature_fields: list[str],
    prefixes: tuple[str, ...],
) -> dict[str, object]:
    fields = [field for field in feature_fields if field.startswith(prefixes)]
    return {
        "feature_count": len(fields),
        "feature_fields": fields,
        "mask_fields": [f"{field}__present" for field in fields],
        "missing_cell_count": _missing_cell_count(rows, fields),
    }


def _missing_cell_count(rows: list[dict[str, str]], fields: list[str]) -> int:
    return sum(1 for row in rows for field in fields if row.get(field, "") == "")


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

"""Model-readiness summaries for tabular feature datasets."""

from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path


TARGET_FIELDS = {
    "target_event_count",
    "target_magnitude_min",
    "target_occurred",
    "target_start_utc",
    "target_status",
    "target_end_utc",
}

ID_FIELDS = {
    "window_id",
    "region_id",
    "window_start_utc",
    "window_end_utc",
    "source_file",
    "seismic_source_file",
    "vlf_image_source_file",
    "vlf_image_latest_source_file",
}

FEATURE_GROUP_PREFIXES = {
    "seismic": ("seismic_",),
    "astronomy": ("astro_",),
    "vlf_metadata": ("vlf_capture_", "vlf_latest_", "vlf_total_", "vlf_jpeg_"),
    "vlf_image": ("vlf_image_",),
    "synthetic_seismic": ("synthetic_seismic_",),
    "synthetic_piezo_vlf": ("synthetic_piezo_vlf_",),
    "synthetic_direct_avalanche": ("synthetic_direct_avalanche_",),
    "synthetic_summary": ("synthetic_summary_",),
    "quality": ("quality_",),
}

ABLATIONS = {
    "seismic_only": ("seismic",),
    "seismic_astronomy": ("seismic", "astronomy"),
    "seismic_vlf": ("seismic", "vlf_metadata", "vlf_image"),
    "full_multimodal": ("seismic", "astronomy", "vlf_metadata", "vlf_image"),
    "synthetic_seismic_only": ("synthetic_seismic",),
    "synthetic_seismic_piezo_vlf": ("synthetic_seismic", "synthetic_piezo_vlf"),
    "synthetic_seismic_direct_avalanche": ("synthetic_seismic", "synthetic_direct_avalanche"),
    "synthetic_full": (
        "synthetic_seismic",
        "synthetic_piezo_vlf",
        "synthetic_direct_avalanche",
        "synthetic_summary",
    ),
}


def summarize_model_readiness(*, input_csv: Path, out_path: Path) -> dict[str, object]:
    rows, fieldnames = _read_rows_and_fields(input_csv)
    labeled_rows = [row for row in rows if row.get("target_occurred") in {"0", "1"}]
    labels = Counter(row.get("target_occurred", "") for row in labeled_rows)
    feature_groups = {
        group: _group_summary(rows, fieldnames, prefixes)
        for group, prefixes in FEATURE_GROUP_PREFIXES.items()
    }
    available_groups = [
        group for group, summary in feature_groups.items() if summary["numeric_feature_count"] > 0
    ]
    report: dict[str, object] = {
        "input": str(input_csv),
        "row_count": len(rows),
        "labeled_row_count": len(labeled_rows),
        "unlabeled_row_count": len(rows) - len(labeled_rows),
        "positive_count": labels.get("1", 0),
        "negative_count": labels.get("0", 0),
        "target_status_counts": dict(Counter(row.get("target_status", "") for row in rows)),
        "feature_groups": feature_groups,
        "available_feature_groups": available_groups,
        "ablation_plan": _ablation_plan(feature_groups),
    }
    report["status"] = _status(report)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def _read_rows_and_fields(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        return list(reader), list(reader.fieldnames or [])


def _group_summary(rows: list[dict[str, str]], fieldnames: list[str], prefixes: tuple[str, ...]) -> dict[str, object]:
    fields = [
        field
        for field in fieldnames
        if field not in TARGET_FIELDS
        and field not in ID_FIELDS
        and any(field.startswith(prefix) for prefix in prefixes)
    ]
    numeric_fields = [field for field in fields if _is_numeric_column(rows, field)]
    missing_cells = sum(1 for row in rows for field in fields if row.get(field, "") == "")
    return {
        "field_count": len(fields),
        "numeric_feature_count": len(numeric_fields),
        "missing_cell_count": missing_cells,
        "numeric_features": numeric_fields,
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


def _ablation_plan(feature_groups: dict[str, dict[str, object]]) -> dict[str, dict[str, object]]:
    plan = {}
    for name, groups in ABLATIONS.items():
        missing = [
            group
            for group in groups
            if int(feature_groups[group]["numeric_feature_count"]) == 0
        ]
        plan[name] = {
            "groups": list(groups),
            "has_required_features": not missing,
            "missing_groups": missing,
        }
    return plan


def _status(report: dict[str, object]) -> str:
    if int(report["labeled_row_count"]) < 2:
        return "waiting_for_labels"
    if int(report["positive_count"]) == 0 or int(report["negative_count"]) == 0:
        return "insufficient_class_variation"
    return "ready_for_smoke_training"

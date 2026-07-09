"""Summaries for synthetic event-list probe runs."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any


MODEL_SCHEMA = "elfquake.synthetic_event_list_model.v1"
DRIFT_SCHEMA = "elfquake.synthetic_drift_diagnostic.v1"


def summarize_synthetic_event_list_probes(
    *,
    root_dir: Path,
    out_path: Path,
    csv_out_path: Path | None = None,
) -> dict[str, object]:
    loaded_reports: list[tuple[Path, dict[str, Any]]] = []
    drift_by_variant: dict[tuple[str, str], dict[str, object]] = {}
    for path in sorted(root_dir.rglob("*.json")):
        try:
            report = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        loaded_reports.append((path, report))
        schema = report.get("schema", "")
        if schema == DRIFT_SCHEMA:
            row = _drift_row(root_dir, path, report)
            drift_by_variant[(str(row["horizon_rows"]), str(row["burn_in_fraction"]))] = row

    reports = []
    for path, report in loaded_reports:
        schema = report.get("schema", "")
        if schema == MODEL_SCHEMA:
            reports.append(_model_row(root_dir, path, report, drift_by_variant=drift_by_variant))
        elif schema == DRIFT_SCHEMA:
            reports.append(_drift_row(root_dir, path, report))

    summary = {
        "schema": "elfquake.synthetic_event_list_probe_summary.v1",
        "root_dir": str(root_dir),
        "report_count": len(reports),
        "reports": reports,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if csv_out_path:
        _write_csv(csv_out_path, reports)
    return summary


def _model_row(
    root_dir: Path,
    path: Path,
    report: dict[str, Any],
    *,
    drift_by_variant: dict[tuple[str, str], dict[str, object]],
) -> dict[str, object]:
    occurrence = report.get("occurrence", {})
    metrics = occurrence.get("test_metrics", {}) if isinstance(occurrence, dict) else {}
    ensemble = report.get("occurrence_ensemble", {})
    feature_selection = report.get("feature_selection", {})
    count = report.get("count", {})
    centroid = report.get("centroid", {})
    shape = report.get("event_shape", {})
    train_rows = _float(report.get("train_row_count"))
    test_rows = _float(report.get("test_row_count"))
    train_pos = _float(report.get("train_positive_count"))
    test_pos = _float(report.get("test_positive_count"))
    metadata = _path_metadata(root_dir, path)
    drift = drift_by_variant.get((str(metadata["horizon_rows"]), str(metadata["burn_in_fraction"])), {})
    return {
        **metadata,
        "kind": "model",
        "status": report.get("status", ""),
        "split_type": _nested(report, "split", "type"),
        "split_field": _nested(report, "split", "split_field"),
        "model_type": _nested(ensemble, "model_type"),
        "ensemble_count": _nested(ensemble, "ensemble_count"),
        "feature_bag_fraction": _nested(ensemble, "feature_bag_fraction"),
        "stump_count": _nested(ensemble, "stump_count"),
        "max_feature_count": _nested(feature_selection, "max_feature_count"),
        "selected_feature_count": _nested(feature_selection, "selected_feature_count"),
        "original_feature_count": report.get("original_feature_count", ""),
        "row_count": report.get("row_count", ""),
        "train_row_count": report.get("train_row_count", ""),
        "test_row_count": report.get("test_row_count", ""),
        "train_positive_rate": _rate(train_pos, train_rows),
        "test_positive_rate": _rate(test_pos, test_rows),
        "overall_positive_rate": drift.get("overall_positive_rate", ""),
        "positive_rate_delta": drift.get("positive_rate_delta", ""),
        "drift_warning": drift.get("drift_warning", ""),
        "balanced_accuracy": metrics.get("balanced_accuracy", ""),
        "positive_recall": metrics.get("positive_recall", ""),
        "negative_recall": metrics.get("negative_recall", ""),
        "accuracy": metrics.get("accuracy", ""),
        "count_mae": count.get("test_mae", "") if isinstance(count, dict) else "",
        "centroid_median_error_km": centroid.get("positive_test_median_error_km", "")
        if isinstance(centroid, dict)
        else "",
        "event_rate_mae": _shape_metric(shape, "eventlist_target_event_rate_per_hour", "mae"),
        "spatial_spread_mae_km": _shape_metric(shape, "eventlist_target_spatial_spread_km", "mae"),
        "time_to_peak_mae_seconds": _shape_metric(shape, "eventlist_target_time_to_peak_event_seconds", "mae"),
    }


def _drift_row(root_dir: Path, path: Path, report: dict[str, Any]) -> dict[str, object]:
    split = report.get("temporal_split", {})
    overall = report.get("overall", {})
    return {
        **_path_metadata(root_dir, path),
        "kind": "drift",
        "status": "evaluated",
        "row_count": report.get("row_count", ""),
        "labeled_row_count": report.get("labeled_row_count", ""),
        "overall_positive_rate": overall.get("positive_rate", "") if isinstance(overall, dict) else "",
        "positive_rate_delta": split.get("positive_rate_delta", "") if isinstance(split, dict) else "",
        "drift_warning": split.get("warning", "") if isinstance(split, dict) else "",
    }


def _path_metadata(root_dir: Path, path: Path) -> dict[str, object]:
    relative = path.relative_to(root_dir)
    parts = relative.parts
    horizon = ""
    burn_in = ""
    for part in parts:
        if part.startswith("h") and part[1:].isdigit():
            horizon = part[1:]
        if part.startswith("burn_"):
            burn_in = part.removeprefix("burn_").replace("p", ".")
    return {
        "path": str(path),
        "relative_path": str(relative),
        "horizon_rows": horizon,
        "burn_in_fraction": burn_in,
        "probe_name": path.stem,
    }


def _shape_metric(shape: object, target: str, metric: str) -> object:
    if not isinstance(shape, dict):
        return ""
    item = shape.get(target, {})
    if not isinstance(item, dict):
        return ""
    return item.get(metric, "")


def _nested(value: object, *keys: str) -> object:
    current = value
    for key in keys:
        if not isinstance(current, dict):
            return ""
        current = current.get(key, "")
    return current


def _float(value: object) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _rate(numerator: float, denominator: float) -> object:
    if denominator <= 0:
        return ""
    return round(numerator / denominator, 6)


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    fieldnames = [
        "kind",
        "horizon_rows",
        "burn_in_fraction",
        "probe_name",
        "status",
        "split_type",
        "split_field",
        "model_type",
        "ensemble_count",
        "feature_bag_fraction",
        "stump_count",
        "max_feature_count",
        "selected_feature_count",
        "original_feature_count",
        "row_count",
        "labeled_row_count",
        "train_row_count",
        "test_row_count",
        "overall_positive_rate",
        "train_positive_rate",
        "test_positive_rate",
        "positive_rate_delta",
        "drift_warning",
        "balanced_accuracy",
        "positive_recall",
        "negative_recall",
        "accuracy",
        "count_mae",
        "centroid_median_error_km",
        "event_rate_mae",
        "spatial_spread_mae_km",
        "time_to_peak_mae_seconds",
        "relative_path",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)

"""Compact summaries for model evaluation report JSON files."""

from __future__ import annotations

import json
from pathlib import Path


def summarize_model_run_reports(*, report_paths: list[Path], out_path: Path) -> dict[str, object]:
    rows = [_summarize_report(path) for path in report_paths]
    summary = {
        "schema": "elfquake.model_run_summary.v1",
        "report_count": len(rows),
        "reports": rows,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary


def _summarize_report(path: Path) -> dict[str, object]:
    report = json.loads(path.read_text(encoding="utf-8"))
    evaluations = {
        name: _summarize_evaluation(item)
        for name, item in report.get("evaluations", {}).items()
        if item.get("status") == "evaluated"
    }
    return {
        "path": str(path),
        "schema": report.get("schema", ""),
        "status": report.get("status", ""),
        "split": _split_summary(report),
        "train_row_count": report.get("train_row_count", 0),
        "test_row_count": report.get("test_row_count", 0),
        "train_positive_count": report.get("train_positive_count", 0),
        "train_negative_count": report.get("train_negative_count", 0),
        "test_positive_count": report.get("test_positive_count", 0),
        "test_negative_count": report.get("test_negative_count", 0),
        "baselines": _baseline_summary(report),
        "evaluations": evaluations,
        "best_default_balanced_accuracy": _best_evaluation(evaluations, metric_name="test_balanced_accuracy"),
        "best_calibrated_balanced_accuracy": _best_evaluation(
            evaluations,
            metric_name="calibrated_test_balanced_accuracy",
        ),
    }


def _split_summary(report: dict[str, object]) -> dict[str, object]:
    if report.get("schema") == "elfquake.group_holdout.v1":
        return {
            "type": "group",
            "group_field": report.get("group_field", ""),
            "test_group": report.get("test_group", ""),
            "train_groups": report.get("train_groups", []),
        }
    if report.get("schema") == "elfquake.torch_tabular_holdout.v1":
        return {
            "type": "temporal",
            "backend": report.get("backend", ""),
            "device": report.get("device", ""),
            "time_field": report.get("time_field", ""),
            "train_fraction": report.get("train_fraction", ""),
            "train_time_start": report.get("train_time_start", ""),
            "train_time_end": report.get("train_time_end", ""),
            "test_time_start": report.get("test_time_start", ""),
            "test_time_end": report.get("test_time_end", ""),
        }
    return {
        "type": "temporal",
        "time_field": report.get("time_field", ""),
        "train_fraction": report.get("train_fraction", ""),
        "train_time_start": report.get("train_time_start", ""),
        "train_time_end": report.get("train_time_end", ""),
        "test_time_start": report.get("test_time_start", ""),
        "test_time_end": report.get("test_time_end", ""),
    }


def _baseline_summary(report: dict[str, object]) -> dict[str, object]:
    baselines = {}
    for name, item in report.get("baselines", {}).items():
        if isinstance(item, dict):
            baselines[name] = {
                "accuracy": item.get("accuracy", 0.0),
                "balanced_accuracy": item.get("balanced_accuracy", 0.0),
                "confusion": item.get("confusion", {}),
            }
        else:
            baselines[name] = item
    return baselines


def _summarize_evaluation(item: dict[str, object]) -> dict[str, object]:
    test = item.get("test_metrics", {})
    calibrated = item.get("calibrated_test_metrics", {})
    return {
        "feature_count": item.get("feature_count", 0),
        "test_accuracy": test.get("accuracy", 0.0),
        "test_balanced_accuracy": test.get("balanced_accuracy", 0.0),
        "test_confusion": test.get("confusion", {}),
        "calibrated_threshold": item.get("calibrated_threshold", 0.5),
        "calibrated_test_accuracy": calibrated.get("accuracy", 0.0),
        "calibrated_test_balanced_accuracy": calibrated.get("balanced_accuracy", 0.0),
        "calibrated_test_confusion": calibrated.get("confusion", {}),
    }


def _best_evaluation(evaluations: dict[str, dict[str, object]], *, metric_name: str) -> dict[str, object]:
    if not evaluations:
        return {}
    name, item = max(evaluations.items(), key=lambda row: float(row[1].get(metric_name, 0.0)))
    return {
        "name": name,
        metric_name: item.get(metric_name, 0.0),
    }

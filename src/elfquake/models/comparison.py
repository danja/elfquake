"""Compare compact model-run summary artifacts."""

from __future__ import annotations

import csv
import json
from pathlib import Path


def compare_model_run_summaries(
    *,
    summary_paths: list[Path],
    out_path: Path,
    csv_out_path: Path | None = None,
) -> dict[str, object]:
    rows = []
    for summary_path in summary_paths:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        if summary.get("schema") == "elfquake.model_summary_comparison.v1":
            rows.extend(_comparison_rows(summary_path, summary))
        else:
            for report in summary.get("reports", []):
                rows.append(_row(summary_path, report))
    comparison = {
        "schema": "elfquake.model_summary_comparison.v1",
        "summary_count": len(summary_paths),
        "report_count": len(rows),
        "best_calibrated_balanced_accuracy": _best_row(rows, "best_calibrated_balanced_accuracy"),
        "best_default_balanced_accuracy": _best_row(rows, "best_default_balanced_accuracy"),
        "rows": rows,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(comparison, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if csv_out_path is not None:
        _write_csv(csv_out_path, rows)
    return comparison


def _row(summary_path: Path, report: dict[str, object]) -> dict[str, object]:
    split = report.get("split", {})
    best_default = report.get("best_default_balanced_accuracy", {})
    best_calibrated = report.get("best_calibrated_balanced_accuracy", {})
    return {
        "summary_path": str(summary_path),
        "report_path": report.get("path", ""),
        "schema": report.get("schema", ""),
        "status": report.get("status", ""),
        "split_type": split.get("type", "") if isinstance(split, dict) else "",
        "test_group": split.get("test_group", "") if isinstance(split, dict) else "",
        "backend": split.get("backend", "") if isinstance(split, dict) else "",
        "device": split.get("device", "") if isinstance(split, dict) else "",
        "train_row_count": report.get("train_row_count", 0),
        "test_row_count": report.get("test_row_count", 0),
        "train_positive_count": report.get("train_positive_count", 0),
        "test_positive_count": report.get("test_positive_count", 0),
        "best_default_name": best_default.get("name", "") if isinstance(best_default, dict) else "",
        "best_default_balanced_accuracy": best_default.get("test_balanced_accuracy", "")
        if isinstance(best_default, dict)
        else "",
        "best_calibrated_name": best_calibrated.get("name", "") if isinstance(best_calibrated, dict) else "",
        "best_calibrated_balanced_accuracy": best_calibrated.get("calibrated_test_balanced_accuracy", "")
        if isinstance(best_calibrated, dict)
        else "",
    }


def _comparison_rows(summary_path: Path, summary: dict[str, object]) -> list[dict[str, object]]:
    rows = []
    for row in summary.get("rows", []):
        if not isinstance(row, dict):
            continue
        copied = dict(row)
        copied["summary_path"] = str(summary_path)
        rows.append(copied)
    return rows


def _best_row(rows: list[dict[str, object]], metric_name: str) -> dict[str, object]:
    candidates = [row for row in rows if _is_number(row.get(metric_name, ""))]
    if not candidates:
        return {}
    row = max(candidates, key=lambda item: float(item[metric_name]))
    return {
        "summary_path": row["summary_path"],
        "report_path": row["report_path"],
        "split_type": row["split_type"],
        "test_group": row["test_group"],
        "model_name": row["best_calibrated_name"]
        if metric_name == "best_calibrated_balanced_accuracy"
        else row["best_default_name"],
        metric_name: row[metric_name],
    }


def _is_number(value: object) -> bool:
    try:
        float(value)
    except (TypeError, ValueError):
        return False
    return True


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "summary_path",
        "report_path",
        "schema",
        "status",
        "split_type",
        "test_group",
        "backend",
        "device",
        "train_row_count",
        "test_row_count",
        "train_positive_count",
        "test_positive_count",
        "best_default_name",
        "best_default_balanced_accuracy",
        "best_calibrated_name",
        "best_calibrated_balanced_accuracy",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)

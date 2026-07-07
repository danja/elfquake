"""Diagnostics for sequence-model comparison artifacts."""

from __future__ import annotations

import csv
import json
from pathlib import Path


SEQUENCE_SCHEMAS = {
    "elfquake.torch_sequence_holdout.v1",
    "elfquake.torch_sequence_group_holdout.v1",
}


def diagnose_sequence_comparison(
    *,
    comparison_path: Path,
    out_path: Path,
    csv_out_path: Path | None = None,
) -> dict[str, object]:
    comparison = json.loads(comparison_path.read_text(encoding="utf-8"))
    rows = _evaluation_rows(comparison)
    group_rows = [row for row in rows if row["split_type"] == "group"]
    report = {
        "schema": "elfquake.sequence_comparison_diagnostic.v1",
        "comparison_path": str(comparison_path),
        "evaluation_row_count": len(rows),
        "group_evaluation_row_count": len(group_rows),
        "best_overall": _best(rows),
        "best_by_source": _best_by(rows, "source"),
        "best_group_by_evaluation": _best_by(group_rows, "evaluation_name"),
        "mean_group_by_evaluation": _mean_by(group_rows, "evaluation_name"),
        "mean_group_by_source_evaluation": _mean_by(group_rows, "source_evaluation"),
        "notes": _notes(rows),
        "rows": rows,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if csv_out_path is not None:
        _write_csv(csv_out_path, rows)
    return report


def _evaluation_rows(comparison: dict[str, object]) -> list[dict[str, object]]:
    rows = []
    for summary_row in comparison.get("rows", []):
        if not isinstance(summary_row, dict) or summary_row.get("schema") not in SEQUENCE_SCHEMAS:
            continue
        report_path = Path(str(summary_row.get("report_path", "")))
        if not report_path.exists():
            continue
        report = json.loads(report_path.read_text(encoding="utf-8"))
        for name, evaluation in report.get("evaluations", {}).items():
            if not isinstance(evaluation, dict) or evaluation.get("status") != "evaluated":
                continue
            rows.append(_evaluation_row(summary_row, report, name, evaluation))
    return rows


def _evaluation_row(
    summary_row: dict[str, object],
    report: dict[str, object],
    name: str,
    evaluation: dict[str, object],
) -> dict[str, object]:
    test_metrics = evaluation.get("test_metrics", {})
    calibrated_metrics = evaluation.get("calibrated_test_metrics", {})
    source = _source(summary_row)
    return {
        "source": source,
        "source_evaluation": f"{source}:{name}",
        "report_path": summary_row.get("report_path", ""),
        "split_type": summary_row.get("split_type", ""),
        "test_group": summary_row.get("test_group", ""),
        "evaluation_name": name,
        "epochs": report.get("epochs", ""),
        "lookback_steps": report.get("lookback_steps", ""),
        "hidden_units": report.get("hidden_units", ""),
        "learning_rate": report.get("learning_rate", ""),
        "feature_count": evaluation.get("feature_count", ""),
        "test_balanced_accuracy": test_metrics.get("balanced_accuracy", ""),
        "calibrated_test_balanced_accuracy": calibrated_metrics.get("balanced_accuracy", ""),
        "calibrated_threshold": evaluation.get("calibrated_threshold", ""),
        "train_row_count": evaluation.get("train_row_count", ""),
        "test_row_count": evaluation.get("test_row_count", ""),
        "dropped_train_row_count": evaluation.get("dropped_train_row_count", ""),
        "dropped_test_row_count": evaluation.get("dropped_test_row_count", ""),
    }


def _source(row: dict[str, object]) -> str:
    path = f"{row.get('summary_path', '')} {row.get('report_path', '')}"
    if "sequence_sweep" in path:
        return "sequence_sweep"
    if "missing_modality" in path:
        return "missing_modality"
    return "default_sequence"


def _best(rows: list[dict[str, object]]) -> dict[str, object]:
    candidates = [row for row in rows if _is_number(row.get("calibrated_test_balanced_accuracy", ""))]
    if not candidates:
        return {}
    return dict(max(candidates, key=lambda row: float(row["calibrated_test_balanced_accuracy"])))


def _best_by(rows: list[dict[str, object]], field: str) -> dict[str, dict[str, object]]:
    grouped: dict[str, list[dict[str, object]]] = {}
    for row in rows:
        grouped.setdefault(str(row.get(field, "")), []).append(row)
    return {key: _best(items) for key, items in sorted(grouped.items())}


def _mean_by(rows: list[dict[str, object]], field: str) -> dict[str, dict[str, object]]:
    grouped: dict[str, list[dict[str, object]]] = {}
    for row in rows:
        grouped.setdefault(str(row.get(field, "")), []).append(row)
    return {
        key: {
            "count": len(values),
            "mean_calibrated_test_balanced_accuracy": _mean(
                [value.get("calibrated_test_balanced_accuracy", "") for value in values]
            ),
            "max_calibrated_test_balanced_accuracy": _max(
                [value.get("calibrated_test_balanced_accuracy", "") for value in values]
            ),
        }
        for key, values in sorted(grouped.items())
    }


def _notes(rows: list[dict[str, object]]) -> list[str]:
    notes = []
    temporal = [row for row in rows if row["split_type"] == "temporal"]
    if temporal and _max(row.get("calibrated_test_balanced_accuracy", "") for row in temporal) <= 0.55:
        notes.append("Temporal sequence reports remain near balanced accuracy 0.5; group results are synthetic-transfer checks only.")
    best_default = _best([row for row in rows if row["source"] == "default_sequence"])
    best_sweep = _best([row for row in rows if row["source"] == "sequence_sweep"])
    if best_default and best_sweep and best_default.get("evaluation_name") != best_sweep.get("evaluation_name"):
        notes.append(
            "Best default and sweep rows use different sequence modalities; compare matched epochs before changing defaults."
        )
    if best_default and best_sweep and best_default.get("epochs") != best_sweep.get("epochs"):
        notes.append(
            f"Best default uses {best_default.get('epochs')} epochs; best sweep uses {best_sweep.get('epochs')} epochs."
        )
    return notes


def _mean(values: list[object]) -> float:
    numeric = [float(value) for value in values if _is_number(value)]
    if not numeric:
        return 0.0
    return round(sum(numeric) / len(numeric), 6)


def _max(values) -> float:
    numeric = [float(value) for value in values if _is_number(value)]
    if not numeric:
        return 0.0
    return round(max(numeric), 6)


def _is_number(value: object) -> bool:
    try:
        float(value)
    except (TypeError, ValueError):
        return False
    return True


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "source",
        "source_evaluation",
        "report_path",
        "split_type",
        "test_group",
        "evaluation_name",
        "epochs",
        "lookback_steps",
        "hidden_units",
        "learning_rate",
        "feature_count",
        "test_balanced_accuracy",
        "calibrated_test_balanced_accuracy",
        "calibrated_threshold",
        "train_row_count",
        "test_row_count",
        "dropped_train_row_count",
        "dropped_test_row_count",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)

"""Compact model-selection summaries for sequence diagnostics."""

from __future__ import annotations

import csv
import json
from pathlib import Path


def summarize_sequence_selection(
    *,
    diagnostic_path: Path,
    out_path: Path,
    csv_out_path: Path | None = None,
) -> dict[str, object]:
    diagnostic = json.loads(diagnostic_path.read_text(encoding="utf-8"))
    rows = [row for row in diagnostic.get("rows", []) if isinstance(row, dict)]
    summaries = _summaries(rows)
    report = {
        "schema": "elfquake.sequence_selection_summary.v1",
        "diagnostic_path": str(diagnostic_path),
        "evaluation_count": len(summaries),
        "best_single_row": _best_summary(summaries, "best_group_calibrated_balanced_accuracy"),
        "best_mean_group_score": _best_summary(summaries, "mean_group_calibrated_balanced_accuracy"),
        "best_worst_group_score": _best_summary(summaries, "worst_group_calibrated_balanced_accuracy"),
        "best_temporal_score": _best_summary(summaries, "best_temporal_calibrated_balanced_accuracy"),
        "selection_note": _selection_note(summaries),
        "rows": summaries,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if csv_out_path is not None:
        _write_csv(csv_out_path, summaries)
    return report


def _summaries(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    names = sorted({str(row.get("evaluation_name", "")) for row in rows if row.get("evaluation_name")})
    return [_summary_for(name, rows) for name in names]


def _summary_for(name: str, rows: list[dict[str, object]]) -> dict[str, object]:
    matching = [row for row in rows if row.get("evaluation_name") == name]
    group_rows = [row for row in matching if row.get("split_type") == "group"]
    temporal_rows = [row for row in matching if row.get("split_type") == "temporal"]
    best_group = _best_row(group_rows)
    worst_group = _worst_row(group_rows)
    best_temporal = _best_row(temporal_rows)
    return {
        "evaluation_name": name,
        "group_row_count": len(group_rows),
        "temporal_row_count": len(temporal_rows),
        "best_group_calibrated_balanced_accuracy": _score(best_group),
        "best_group_source": best_group.get("source", ""),
        "best_group_test_group": best_group.get("test_group", ""),
        "best_group_report_path": best_group.get("report_path", ""),
        "mean_group_calibrated_balanced_accuracy": _mean_score(group_rows),
        "worst_group_calibrated_balanced_accuracy": _score(worst_group),
        "worst_group_source": worst_group.get("source", ""),
        "worst_group_test_group": worst_group.get("test_group", ""),
        "best_temporal_calibrated_balanced_accuracy": _score(best_temporal),
        "best_temporal_source": best_temporal.get("source", ""),
        "best_temporal_report_path": best_temporal.get("report_path", ""),
    }


def _selection_note(summaries: list[dict[str, object]]) -> str:
    if not summaries:
        return "No evaluated sequence rows were found."
    best_single = _best_summary(summaries, "best_group_calibrated_balanced_accuracy")
    best_mean = _best_summary(summaries, "mean_group_calibrated_balanced_accuracy")
    best_worst = _best_summary(summaries, "worst_group_calibrated_balanced_accuracy")
    best_temporal = _best_summary(summaries, "best_temporal_calibrated_balanced_accuracy")
    if float(best_temporal.get("best_temporal_calibrated_balanced_accuracy", 0.0)) <= 0.55:
        return (
            "Group-holdout winners should remain synthetic-transfer diagnostics because temporal sequence "
            "scores are still near chance."
        )
    if best_single.get("evaluation_name") != best_mean.get("evaluation_name"):
        return "Best single-row and best mean group choices differ; prefer more repeated runs before selecting a default."
    if best_single.get("evaluation_name") != best_worst.get("evaluation_name"):
        return "Best single-row and best worst-seed choices differ; inspect held-out seed stability before selecting a default."
    return "Best single-row, mean group, worst group, and temporal choices are aligned enough for default selection."


def _best_summary(summaries: list[dict[str, object]], metric: str) -> dict[str, object]:
    if not summaries:
        return {}
    return dict(max(summaries, key=lambda row: float(row.get(metric, 0.0) or 0.0)))


def _best_row(rows: list[dict[str, object]]) -> dict[str, object]:
    if not rows:
        return {}
    return max(rows, key=lambda row: float(row.get("calibrated_test_balanced_accuracy", 0.0) or 0.0))


def _worst_row(rows: list[dict[str, object]]) -> dict[str, object]:
    if not rows:
        return {}
    return min(rows, key=lambda row: float(row.get("calibrated_test_balanced_accuracy", 0.0) or 0.0))


def _score(row: dict[str, object]) -> float:
    if not row:
        return 0.0
    return round(float(row.get("calibrated_test_balanced_accuracy", 0.0) or 0.0), 6)


def _mean_score(rows: list[dict[str, object]]) -> float:
    values = [float(row.get("calibrated_test_balanced_accuracy", 0.0) or 0.0) for row in rows]
    if not values:
        return 0.0
    return round(sum(values) / len(values), 6)


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "evaluation_name",
        "group_row_count",
        "temporal_row_count",
        "best_group_calibrated_balanced_accuracy",
        "best_group_source",
        "best_group_test_group",
        "mean_group_calibrated_balanced_accuracy",
        "worst_group_calibrated_balanced_accuracy",
        "worst_group_source",
        "worst_group_test_group",
        "best_temporal_calibrated_balanced_accuracy",
        "best_temporal_source",
        "best_group_report_path",
        "best_temporal_report_path",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)

"""Dependency-free logistic-regression smoke trainer."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path


EXCLUDED_FEATURES = {
    "window_id",
    "region_id",
    "window_start_utc",
    "window_end_utc",
    "target_start_utc",
    "target_end_utc",
    "target_magnitude_min",
    "target_event_count",
    "target_occurred",
    "target_status",
}


def train_logistic_smoke(
    *,
    design_matrix_csv: Path,
    out_path: Path,
    epochs: int = 600,
    learning_rate: float = 0.2,
) -> dict[str, object]:
    rows = _read_rows(design_matrix_csv)
    report: dict[str, object] = {
        "status": "not_run",
        "input": str(design_matrix_csv),
        "row_count": len(rows),
    }
    if len(rows) < 2:
        report["status"] = "insufficient_rows"
        return _write_report(out_path, report)

    labels = [_label(row) for row in rows]
    positives = sum(labels)
    negatives = len(labels) - positives
    report.update({"positive_count": positives, "negative_count": negatives})
    if positives == 0 or negatives == 0:
        report["status"] = "insufficient_class_variation"
        return _write_report(out_path, report)

    feature_names = _numeric_feature_names(rows)
    report["feature_names"] = feature_names
    if not feature_names:
        report["status"] = "no_numeric_features"
        return _write_report(out_path, report)

    matrix = [[float(row[name]) for name in feature_names] for row in rows]
    standardized, means, scales = _standardize(matrix)
    weights, intercept = _fit_logistic(standardized, labels, epochs=epochs, learning_rate=learning_rate)
    predictions = [_sigmoid(_dot(weights, values) + intercept) for values in standardized]
    predicted_labels = [1 if value >= 0.5 else 0 for value in predictions]
    accuracy = sum(1 for predicted, actual in zip(predicted_labels, labels) if predicted == actual) / len(labels)

    report.update(
        {
            "status": "trained_in_sample",
            "epochs": epochs,
            "learning_rate": learning_rate,
            "in_sample_accuracy": round(accuracy, 6),
            "coefficients": {
                name: {
                    "weight": round(weight, 8),
                    "mean": round(mean, 8),
                    "scale": round(scale, 8),
                }
                for name, weight, mean, scale in zip(feature_names, weights, means, scales)
            },
            "intercept": round(intercept, 8),
        }
    )
    return _write_report(out_path, report)


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _label(row: dict[str, str]) -> int:
    value = row.get("target_occurred", "")
    if value not in {"0", "1"}:
        raise ValueError(f"target_occurred must be 0 or 1, got {value!r}")
    return int(value)


def _numeric_feature_names(rows: list[dict[str, str]]) -> list[str]:
    names: list[str] = []
    for name in rows[0]:
        if name in EXCLUDED_FEATURES:
            continue
        values = [row.get(name, "") for row in rows]
        if values and all(_is_float(value) for value in values):
            names.append(name)
    return names


def _is_float(value: str) -> bool:
    try:
        float(value)
        return value != ""
    except ValueError:
        return False


def _standardize(matrix: list[list[float]]) -> tuple[list[list[float]], list[float], list[float]]:
    columns = list(zip(*matrix))
    means = [sum(column) / len(column) for column in columns]
    scales = []
    for column, mean in zip(columns, means):
        variance = sum((value - mean) ** 2 for value in column) / len(column)
        scale = math.sqrt(variance)
        scales.append(scale if scale else 1.0)
    standardized = [
        [(value - means[index]) / scales[index] for index, value in enumerate(row)]
        for row in matrix
    ]
    return standardized, means, scales


def _fit_logistic(
    matrix: list[list[float]],
    labels: list[int],
    *,
    epochs: int,
    learning_rate: float,
) -> tuple[list[float], float]:
    weights = [0.0 for _ in matrix[0]]
    intercept = 0.0
    n_rows = len(matrix)
    for _ in range(epochs):
        gradient_weights = [0.0 for _ in weights]
        gradient_intercept = 0.0
        for values, label in zip(matrix, labels):
            error = _sigmoid(_dot(weights, values) + intercept) - label
            for index, value in enumerate(values):
                gradient_weights[index] += error * value
            gradient_intercept += error
        weights = [
            weight - learning_rate * gradient / n_rows
            for weight, gradient in zip(weights, gradient_weights)
        ]
        intercept -= learning_rate * gradient_intercept / n_rows
    return weights, intercept


def _dot(left: list[float], right: list[float]) -> float:
    return sum(a * b for a, b in zip(left, right))


def _sigmoid(value: float) -> float:
    if value >= 0:
        z = math.exp(-value)
        return 1 / (1 + z)
    z = math.exp(value)
    return z / (1 + z)


def _write_report(out_path: Path, report: dict[str, object]) -> dict[str, object]:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report

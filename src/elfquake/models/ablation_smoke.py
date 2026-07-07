"""Tiny in-sample ablation smoke models for available labeled rows."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path

from elfquake.models.feature_groups import ABLATIONS, FEATURE_GROUP_PREFIXES, ID_FIELDS, TARGET_FIELDS


def train_ablation_smoke(
    *,
    input_csv: Path,
    out_path: Path,
    epochs: int = 600,
    learning_rate: float = 0.2,
) -> dict[str, object]:
    rows, fieldnames = _read_rows_and_fields(input_csv)
    labeled = [row for row in rows if row.get("target_occurred") in {"0", "1"}]
    labels = [int(row["target_occurred"]) for row in labeled]
    positive_count = sum(labels)
    negative_count = len(labels) - positive_count
    report: dict[str, object] = {
        "input": str(input_csv),
        "row_count": len(rows),
        "labeled_row_count": len(labeled),
        "positive_count": positive_count,
        "negative_count": negative_count,
        "epochs": epochs,
        "learning_rate": learning_rate,
        "ablations": {},
    }
    for name, groups in ABLATIONS.items():
        report["ablations"][name] = _train_one(
            rows=labeled,
            labels=labels,
            fieldnames=fieldnames,
            groups=groups,
            epochs=epochs,
            learning_rate=learning_rate,
        )
    report["status"] = _overall_status(report)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def _train_one(
    *,
    rows: list[dict[str, str]],
    labels: list[int],
    fieldnames: list[str],
    groups: tuple[str, ...],
    epochs: int,
    learning_rate: float,
) -> dict[str, object]:
    result: dict[str, object] = {"groups": list(groups), "row_count": len(rows)}
    if len(rows) < 2:
        result["status"] = "insufficient_rows"
        return result
    if sum(labels) == 0 or sum(labels) == len(labels):
        result["status"] = "insufficient_class_variation"
        return result

    features = _feature_names(rows, fieldnames, groups)
    result["feature_names"] = features
    if not features:
        result["status"] = "no_numeric_features"
        return result

    matrix = [[float(row[name]) for name in features] for row in rows]
    standardized, means, scales = _standardize(matrix)
    weights, intercept = _fit_logistic(standardized, labels, epochs=epochs, learning_rate=learning_rate)
    probabilities = [_sigmoid(_dot(weights, values) + intercept) for values in standardized]
    predictions = [1 if probability >= 0.5 else 0 for probability in probabilities]
    accuracy = sum(1 for predicted, actual in zip(predictions, labels) if predicted == actual) / len(labels)
    result.update(
        {
            "status": "trained_in_sample",
            "in_sample_accuracy": round(accuracy, 6),
            "probabilities": [round(value, 6) for value in probabilities],
            "intercept": round(intercept, 8),
            "coefficients": {
                name: {
                    "weight": round(weight, 8),
                    "mean": round(mean, 8),
                    "scale": round(scale, 8),
                }
                for name, weight, mean, scale in zip(features, weights, means, scales)
            },
        }
    )
    return result


def _feature_names(rows: list[dict[str, str]], fieldnames: list[str], groups: tuple[str, ...]) -> list[str]:
    prefixes = tuple(prefix for group in groups for prefix in FEATURE_GROUP_PREFIXES[group])
    names = []
    for field in fieldnames:
        if field in TARGET_FIELDS or field in ID_FIELDS:
            continue
        if not field.startswith(prefixes):
            continue
        values = [row.get(field, "") for row in rows]
        if values and all(_is_float(value) for value in values):
            names.append(field)
    return names


def _read_rows_and_fields(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        return list(reader), list(reader.fieldnames or [])


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
    return (
        [[(value - means[index]) / scales[index] for index, value in enumerate(row)] for row in matrix],
        means,
        scales,
    )


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


def _overall_status(report: dict[str, object]) -> str:
    statuses = {item["status"] for item in report["ablations"].values()}
    if "trained_in_sample" in statuses:
        return "trained_in_sample"
    if statuses == {"insufficient_class_variation"}:
        return "insufficient_class_variation"
    if statuses == {"insufficient_rows"}:
        return "insufficient_rows"
    return "not_trainable"

"""Time-ordered train/test smoke evaluation for aligned model rows."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path

from elfquake.models.feature_groups import ABLATIONS, FEATURE_GROUP_PREFIXES, ID_FIELDS, TARGET_FIELDS


def evaluate_temporal_holdout(
    *,
    input_csv: Path,
    out_path: Path,
    time_field: str = "window_start_utc",
    train_fraction: float = 0.8,
    epochs: int = 600,
    learning_rate: float = 0.2,
    group_by_time: bool = False,
) -> dict[str, object]:
    if not 0 < train_fraction < 1:
        raise ValueError("train_fraction must be between 0 and 1")

    rows, fieldnames = _read_rows_and_fields(input_csv)
    labeled = sorted(
        [row for row in rows if row.get("target_occurred") in {"0", "1"}],
        key=lambda row: row.get(time_field, ""),
    )
    report: dict[str, object] = {
        "schema": "elfquake.temporal_holdout.v1",
        "input": str(input_csv),
        "row_count": len(rows),
        "labeled_row_count": len(labeled),
        "time_field": time_field,
        "train_fraction": train_fraction,
        "epochs": epochs,
        "learning_rate": learning_rate,
        "group_by_time": group_by_time,
        "evaluations": {},
    }
    if len(labeled) < 4:
        report["status"] = "insufficient_labeled_rows"
        return _write_report(out_path, report)

    train_rows, test_rows = _split_rows(
        labeled,
        time_field=time_field,
        train_fraction=train_fraction,
        group_by_time=group_by_time,
    )
    labels_train = [int(row["target_occurred"]) for row in train_rows]
    labels_test = [int(row["target_occurred"]) for row in test_rows]
    report.update(
        {
            "train_row_count": len(train_rows),
            "test_row_count": len(test_rows),
            "train_time_start": train_rows[0].get(time_field, ""),
            "train_time_end": train_rows[-1].get(time_field, ""),
            "test_time_start": test_rows[0].get(time_field, ""),
            "test_time_end": test_rows[-1].get(time_field, ""),
            "train_positive_count": sum(labels_train),
            "train_negative_count": len(labels_train) - sum(labels_train),
            "test_positive_count": sum(labels_test),
            "test_negative_count": len(labels_test) - sum(labels_test),
            "baselines": _baselines(labels_train, labels_test),
        }
    )

    evaluation_specs: dict[str, tuple[str, ...] | None] = {"all_features": None}
    evaluation_specs.update(ABLATIONS)
    for name, groups in evaluation_specs.items():
        report["evaluations"][name] = _evaluate_one(
            train_rows=train_rows,
            test_rows=test_rows,
            fieldnames=fieldnames,
            groups=groups,
            epochs=epochs,
            learning_rate=learning_rate,
        )
    report["status"] = _overall_status(report)
    return _write_report(out_path, report)


def _split_rows(
    rows: list[dict[str, str]],
    *,
    time_field: str,
    train_fraction: float,
    group_by_time: bool,
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    if not group_by_time:
        train_count = min(len(rows) - 1, max(2, int(len(rows) * train_fraction)))
        return rows[:train_count], rows[train_count:]

    times = sorted({row.get(time_field, "") for row in rows})
    if len(times) < 2:
        return rows[:1], rows[1:]
    train_time_count = min(len(times) - 1, max(1, int(len(times) * train_fraction)))
    cutoff = times[train_time_count]
    train_rows = [row for row in rows if row.get(time_field, "") < cutoff]
    test_rows = [row for row in rows if row.get(time_field, "") >= cutoff]
    return train_rows, test_rows


def evaluate_group_holdout(
    *,
    input_csv: Path,
    out_path: Path,
    group_field: str = "dataset_id",
    test_group: str,
    epochs: int = 600,
    learning_rate: float = 0.2,
) -> dict[str, object]:
    rows, fieldnames = _read_rows_and_fields(input_csv)
    labeled = [row for row in rows if row.get("target_occurred") in {"0", "1"}]
    train_rows = [row for row in labeled if row.get(group_field, "") != test_group]
    test_rows = [row for row in labeled if row.get(group_field, "") == test_group]
    labels_train = [int(row["target_occurred"]) for row in train_rows]
    labels_test = [int(row["target_occurred"]) for row in test_rows]
    report: dict[str, object] = {
        "schema": "elfquake.group_holdout.v1",
        "input": str(input_csv),
        "row_count": len(rows),
        "labeled_row_count": len(labeled),
        "group_field": group_field,
        "test_group": test_group,
        "train_groups": sorted({row.get(group_field, "") for row in train_rows}),
        "epochs": epochs,
        "learning_rate": learning_rate,
        "evaluations": {},
    }
    if len(train_rows) < 2 or len(test_rows) < 1:
        report["status"] = "insufficient_group_rows"
        return _write_report(out_path, report)

    report.update(
        {
            "train_row_count": len(train_rows),
            "test_row_count": len(test_rows),
            "train_positive_count": sum(labels_train),
            "train_negative_count": len(labels_train) - sum(labels_train),
            "test_positive_count": sum(labels_test),
            "test_negative_count": len(labels_test) - sum(labels_test),
            "baselines": _baselines(labels_train, labels_test),
        }
    )

    evaluation_specs: dict[str, tuple[str, ...] | None] = {"all_features": None}
    evaluation_specs.update(ABLATIONS)
    for name, groups in evaluation_specs.items():
        report["evaluations"][name] = _evaluate_one(
            train_rows=train_rows,
            test_rows=test_rows,
            fieldnames=fieldnames,
            groups=groups,
            epochs=epochs,
            learning_rate=learning_rate,
        )
    report["status"] = _overall_status(report)
    return _write_report(out_path, report)


def _evaluate_one(
    *,
    train_rows: list[dict[str, str]],
    test_rows: list[dict[str, str]],
    fieldnames: list[str],
    groups: tuple[str, ...] | None,
    epochs: int,
    learning_rate: float,
) -> dict[str, object]:
    labels_train = [int(row["target_occurred"]) for row in train_rows]
    labels_test = [int(row["target_occurred"]) for row in test_rows]
    result: dict[str, object] = {
        "groups": list(groups or []),
        "train_row_count": len(train_rows),
        "test_row_count": len(test_rows),
    }
    if sum(labels_train) == 0 or sum(labels_train) == len(labels_train):
        result["status"] = "insufficient_train_class_variation"
        return result

    features = _feature_names(train_rows + test_rows, fieldnames, groups)
    result["feature_count"] = len(features)
    result["feature_names"] = features
    if not features:
        result["status"] = "no_numeric_features"
        return result

    train_matrix = [[float(row[name]) for name in features] for row in train_rows]
    test_matrix = [[float(row[name]) for name in features] for row in test_rows]
    train_standardized, test_standardized, means, scales = _standardize_train_test(train_matrix, test_matrix)
    weights, intercept = _fit_logistic(train_standardized, labels_train, epochs=epochs, learning_rate=learning_rate)
    train_probabilities = [_sigmoid(_dot(weights, values) + intercept) for values in train_standardized]
    test_probabilities = [_sigmoid(_dot(weights, values) + intercept) for values in test_standardized]
    train_predictions = _predictions(train_probabilities, threshold=0.5)
    test_predictions = _predictions(test_probabilities, threshold=0.5)
    calibrated_threshold = _best_threshold(train_probabilities, labels_train)
    calibrated_train_predictions = _predictions(train_probabilities, threshold=calibrated_threshold)
    calibrated_test_predictions = _predictions(test_probabilities, threshold=calibrated_threshold)
    result.update(
        {
            "status": "evaluated",
            "default_threshold": 0.5,
            "train_metrics": _metrics(train_predictions, labels_train),
            "test_metrics": _metrics(test_predictions, labels_test),
            "calibrated_threshold": round(calibrated_threshold, 6),
            "calibrated_train_metrics": _metrics(calibrated_train_predictions, labels_train),
            "calibrated_test_metrics": _metrics(calibrated_test_predictions, labels_test),
            "test_probabilities": [round(value, 6) for value in test_probabilities],
            "test_predictions": test_predictions,
            "test_labels": labels_test,
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


def _feature_names(
    rows: list[dict[str, str]],
    fieldnames: list[str],
    groups: tuple[str, ...] | None,
) -> list[str]:
    prefixes = None
    if groups is not None:
        prefixes = tuple(prefix for group in groups for prefix in FEATURE_GROUP_PREFIXES[group])
    names = []
    for field in fieldnames:
        if field in TARGET_FIELDS or field in ID_FIELDS:
            continue
        if prefixes is not None and not field.startswith(prefixes):
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


def _standardize_train_test(
    train_matrix: list[list[float]],
    test_matrix: list[list[float]],
) -> tuple[list[list[float]], list[list[float]], list[float], list[float]]:
    columns = list(zip(*train_matrix))
    means = [sum(column) / len(column) for column in columns]
    scales = []
    for column, mean in zip(columns, means):
        variance = sum((value - mean) ** 2 for value in column) / len(column)
        scale = math.sqrt(variance)
        scales.append(scale if scale else 1.0)
    return (
        _apply_standardization(train_matrix, means, scales),
        _apply_standardization(test_matrix, means, scales),
        means,
        scales,
    )


def _apply_standardization(matrix: list[list[float]], means: list[float], scales: list[float]) -> list[list[float]]:
    return [
        [(value - means[index]) / scales[index] for index, value in enumerate(row)]
        for row in matrix
    ]


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


def _accuracy(predictions: list[int], labels: list[int]) -> float:
    return round(sum(1 for predicted, actual in zip(predictions, labels) if predicted == actual) / len(labels), 6)


def _metrics(predictions: list[int], labels: list[int]) -> dict[str, object]:
    confusion = _confusion(predictions, labels)
    positives = confusion["true_positive"] + confusion["false_negative"]
    negatives = confusion["true_negative"] + confusion["false_positive"]
    predicted_positive = confusion["true_positive"] + confusion["false_positive"]
    recall_positive = confusion["true_positive"] / positives if positives else 0.0
    recall_negative = confusion["true_negative"] / negatives if negatives else 0.0
    precision = confusion["true_positive"] / predicted_positive if predicted_positive else 0.0
    return {
        "accuracy": _accuracy(predictions, labels),
        "balanced_accuracy": round((recall_positive + recall_negative) / 2, 6),
        "positive_recall": round(recall_positive, 6),
        "negative_recall": round(recall_negative, 6),
        "positive_precision": round(precision, 6),
        "confusion": confusion,
    }


def _confusion(predictions: list[int], labels: list[int]) -> dict[str, int]:
    return {
        "true_positive": sum(1 for predicted, actual in zip(predictions, labels) if predicted == 1 and actual == 1),
        "true_negative": sum(1 for predicted, actual in zip(predictions, labels) if predicted == 0 and actual == 0),
        "false_positive": sum(1 for predicted, actual in zip(predictions, labels) if predicted == 1 and actual == 0),
        "false_negative": sum(1 for predicted, actual in zip(predictions, labels) if predicted == 0 and actual == 1),
    }


def _predictions(probabilities: list[float], *, threshold: float) -> list[int]:
    return [1 if probability >= threshold else 0 for probability in probabilities]


def _best_threshold(probabilities: list[float], labels: list[int]) -> float:
    candidates = sorted(set([0.5, *probabilities]))
    best_threshold = candidates[0]
    best_score = -1.0
    for threshold in candidates:
        score = float(_metrics(_predictions(probabilities, threshold=threshold), labels)["balanced_accuracy"])
        if score > best_score:
            best_score = score
            best_threshold = threshold
    return best_threshold


def _baselines(labels_train: list[int], labels_test: list[int]) -> dict[str, object]:
    majority = 1 if sum(labels_train) >= len(labels_train) / 2 else 0
    return {
        "train_majority_class": majority,
        "majority_class": _metrics([majority for _ in labels_test], labels_test),
        "always_positive": _metrics([1 for _ in labels_test], labels_test),
        "always_negative": _metrics([0 for _ in labels_test], labels_test),
    }


def _overall_status(report: dict[str, object]) -> str:
    statuses = {item["status"] for item in report["evaluations"].values()}
    if "evaluated" in statuses:
        return "evaluated"
    if statuses == {"insufficient_train_class_variation"}:
        return "insufficient_train_class_variation"
    return "not_evaluated"


def _write_report(out_path: Path, report: dict[str, object]) -> dict[str, object]:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report

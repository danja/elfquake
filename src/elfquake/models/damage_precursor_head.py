"""Causal engineered-feature baseline for delayed-failure damage sensors."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path

from elfquake.models.temporal_holdout import _best_threshold, _metrics, _predictions
from elfquake.models.torch_multimodal_data import load_modality_sequences, sequence_index_at_or_before


DAMAGE_FIELDS = ("damage_total", "damage_max", "damage_active_cell_count")


def evaluate_damage_precursor_head(
    *, target_csv: Path, piezo_manifest_paths: list[Path], out_path: Path,
    group_field: str = "dataset_id", short_steps: int = 5, long_steps: int = 30,
    epochs: int = 400, learning_rate: float = 0.05, l2: float = 0.01,
) -> dict[str, object]:
    """Leave one episode out, deriving all predictors from present/past damage state."""
    if short_steps < 1 or long_steps < short_steps:
        raise ValueError("require 1 <= short_steps <= long_steps")
    sequences = load_modality_sequences(piezo_manifest_paths)
    rows = _feature_rows(_read_labeled(target_csv), sequences, long_steps, short_steps)
    groups = sorted({row["group"] for row in rows})
    if len(groups) < 2:
        raise ValueError("damage precursor head requires at least two episodes")
    runs = []
    for test_group in groups:
        train = [row for row in rows if row["group"] != test_group]
        test = [row for row in rows if row["group"] == test_group]
        runs.append(_evaluate_fold(train, test, test_group, epochs, learning_rate, l2))
    report = {
        "schema": "elfquake.damage_precursor_head.v1",
        "status": "evaluated",
        "target_csv": str(target_csv),
        "piezo_sequence_manifests": [str(path) for path in piezo_manifest_paths],
        "group_field": group_field,
        "groups": groups,
        "feature_names": _feature_names(),
        "timing": {
            "predictors": "current and prior pre-relaxation damage sensor values only",
            "target": "post-current-step avalanche within the future target horizon",
            "short_steps": short_steps,
            "long_steps": long_steps,
        },
        "epochs": epochs,
        "learning_rate": learning_rate,
        "l2": l2,
        "runs": runs,
        "summary": _summary(runs),
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def _read_labeled(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return [row for row in csv.DictReader(handle) if row.get("eventlist_target_occurred") in {"0", "1"}]


def _feature_rows(rows, sequences, long_steps: int, short_steps: int):
    result = []
    for row in rows:
        dataset_id = row.get("dataset_id", "")
        sequence = sequences.get((dataset_id, "synthetic_piezo_vlf"))
        if sequence is None:
            continue
        index = sequence_index_at_or_before(sequence, row.get("window_start_utc", ""))
        if index is None or index < long_steps - 1:
            continue
        field_indices = {name: sequence.feature_names.index(name) for name in DAMAGE_FIELDS if name in sequence.feature_names}
        if len(field_indices) != len(DAMAGE_FIELDS):
            raise ValueError(f"{dataset_id} lacks damage sensor fields")
        features = []
        for field in DAMAGE_FIELDS:
            values = [sequence.values[position][field_indices[field]] for position in range(index - long_steps + 1, index + 1)]
            features.extend(_window_features(values, short_steps))
        result.append({"group": dataset_id, "label": int(row["eventlist_target_occurred"]), "features": features})
    return result


def _window_features(values: list[float], short_steps: int) -> list[float]:
    current = values[-1]
    short = values[-short_steps:]
    short_mean = sum(short) / len(short)
    long_mean = sum(values) / len(values)
    return [current, current - values[-2], short_mean, long_mean, short_mean - long_mean, max(short) - min(short)]


def _feature_names() -> list[str]:
    suffixes = ("level", "rise_1", "short_mean", "long_mean", "short_long_contrast", "short_range")
    return [f"{field}_{suffix}" for field in DAMAGE_FIELDS for suffix in suffixes]


def _evaluate_fold(train, test, test_group, epochs, learning_rate, l2):
    train_x = [row["features"] for row in train]
    test_x = [row["features"] for row in test]
    train_y = [row["label"] for row in train]
    test_y = [row["label"] for row in test]
    means, stds = _normalization(train_x)
    weights, bias = _fit_logistic(_normalize(train_x, means, stds), train_y, epochs, learning_rate, l2)
    train_probs = _probabilities(_normalize(train_x, means, stds), weights, bias)
    test_probs = _probabilities(_normalize(test_x, means, stds), weights, bias)
    threshold = _best_threshold(train_probs, train_y)
    return {
        "test_group": test_group,
        "train_row_count": len(train), "test_row_count": len(test),
        "train_positive_count": sum(train_y), "test_positive_count": sum(test_y),
        "calibrated_threshold": round(threshold, 6),
        "metrics": _metrics(_predictions(test_probs, threshold=threshold), test_y),
        "default_metrics": _metrics(_predictions(test_probs, threshold=0.5), test_y),
    }


def _normalization(rows):
    means = [sum(row[index] for row in rows) / len(rows) for index in range(len(rows[0]))]
    stds = [max(1e-8, math.sqrt(sum((row[index] - mean) ** 2 for row in rows) / len(rows))) for index, mean in enumerate(means)]
    return means, stds


def _normalize(rows, means, stds):
    return [[(value - means[index]) / stds[index] for index, value in enumerate(row)] for row in rows]


def _fit_logistic(rows, labels, epochs, learning_rate, l2):
    weights = [0.0] * len(rows[0])
    bias = 0.0
    positive_weight = (len(labels) - sum(labels)) / max(1, sum(labels))
    for _ in range(epochs):
        grad_w = [0.0] * len(weights)
        grad_b = 0.0
        for row, label in zip(rows, labels):
            probability = _sigmoid(sum(weight * value for weight, value in zip(weights, row)) + bias)
            error = (probability - label) * (positive_weight if label else 1.0)
            grad_b += error
            for index, value in enumerate(row):
                grad_w[index] += error * value
        scale = 1.0 / len(rows)
        weights = [weight - learning_rate * (gradient * scale + l2 * weight) for weight, gradient in zip(weights, grad_w)]
        bias -= learning_rate * grad_b * scale
    return weights, bias


def _probabilities(rows, weights, bias):
    return [_sigmoid(sum(weight * value for weight, value in zip(weights, row)) + bias) for row in rows]


def _sigmoid(value):
    return 1.0 / (1.0 + math.exp(-max(-60.0, min(60.0, value))))


def _summary(runs):
    values = [run["metrics"]["balanced_accuracy"] for run in runs]
    return {
        "run_count": len(runs),
        "balanced_accuracy": {"mean": sum(values) / len(values), "min": min(values), "max": max(values)},
        "both_recalls_at_least_0_40_count": sum(
            run["metrics"]["positive_recall"] >= 0.4 and run["metrics"]["negative_recall"] >= 0.4 for run in runs
        ),
    }

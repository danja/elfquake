"""CPU PyTorch tabular classifier for aligned model rows."""

from __future__ import annotations

import csv
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path

from elfquake.models.readiness import ABLATIONS, FEATURE_GROUP_PREFIXES, ID_FIELDS, TARGET_FIELDS
from elfquake.models.temporal_holdout import _baselines, _best_threshold, _metrics, _predictions


@dataclass(frozen=True)
class PreparedMatrix:
    feature_names: list[str]
    train_values: list[list[float]]
    test_values: list[list[float]]
    train_masks: list[list[float]]
    test_masks: list[list[float]]
    means: list[float]
    scales: list[float]
    missing_rates: dict[str, float]


def evaluate_torch_tabular_holdout(
    *,
    input_csv: Path,
    out_path: Path,
    time_field: str = "window_start_utc",
    train_fraction: float = 0.8,
    epochs: int = 80,
    learning_rate: float = 0.001,
    hidden_units: int = 32,
    batch_size: int = 64,
    seed: int = 42,
    include_missing_masks: bool = True,
    weight_decay: float = 0.0,
) -> dict[str, object]:
    """Train temporal-holdout MLP evaluations for all available ablations."""

    if not 0 < train_fraction < 1:
        raise ValueError("train_fraction must be between 0 and 1")
    if epochs < 1:
        raise ValueError("epochs must be positive")
    if hidden_units < 1:
        raise ValueError("hidden_units must be positive")
    if batch_size < 1:
        raise ValueError("batch_size must be positive")

    torch = _import_torch()
    _set_deterministic_seed(torch, seed)

    rows, fieldnames = _read_rows_and_fields(input_csv)
    labeled = sorted(
        [row for row in rows if row.get("target_occurred") in {"0", "1"}],
        key=lambda row: row.get(time_field, ""),
    )
    report: dict[str, object] = {
        "schema": "elfquake.torch_tabular_holdout.v1",
        "backend": "torch",
        "device": "cpu",
        "input": str(input_csv),
        "row_count": len(rows),
        "labeled_row_count": len(labeled),
        "time_field": time_field,
        "train_fraction": train_fraction,
        "epochs": epochs,
        "learning_rate": learning_rate,
        "hidden_units": hidden_units,
        "batch_size": batch_size,
        "seed": seed,
        "include_missing_masks": include_missing_masks,
        "weight_decay": weight_decay,
        "evaluations": {},
    }
    if len(labeled) < 4:
        report["status"] = "insufficient_labeled_rows"
        return _write_report(out_path, report)

    train_count = min(len(labeled) - 1, max(2, int(len(labeled) * train_fraction)))
    train_rows = labeled[:train_count]
    test_rows = labeled[train_count:]
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
            torch=torch,
            train_rows=train_rows,
            test_rows=test_rows,
            fieldnames=fieldnames,
            groups=groups,
            labels_train=labels_train,
            labels_test=labels_test,
            epochs=epochs,
            learning_rate=learning_rate,
            hidden_units=hidden_units,
            batch_size=batch_size,
            include_missing_masks=include_missing_masks,
            weight_decay=weight_decay,
            seed=seed,
        )
    report["status"] = _overall_status(report)
    return _write_report(out_path, report)


def _evaluate_one(
    *,
    torch: object,
    train_rows: list[dict[str, str]],
    test_rows: list[dict[str, str]],
    fieldnames: list[str],
    groups: tuple[str, ...] | None,
    labels_train: list[int],
    labels_test: list[int],
    epochs: int,
    learning_rate: float,
    hidden_units: int,
    batch_size: int,
    include_missing_masks: bool,
    weight_decay: float,
    seed: int,
) -> dict[str, object]:
    result: dict[str, object] = {
        "groups": list(groups or []),
        "train_row_count": len(train_rows),
        "test_row_count": len(test_rows),
    }
    if sum(labels_train) == 0 or sum(labels_train) == len(labels_train):
        result["status"] = "insufficient_train_class_variation"
        return result

    prepared = _prepare_matrix(
        train_rows=train_rows,
        test_rows=test_rows,
        fieldnames=fieldnames,
        groups=groups,
    )
    result["feature_count"] = len(prepared.feature_names)
    result["feature_names"] = prepared.feature_names
    result["missing_rates"] = prepared.missing_rates
    if not prepared.feature_names:
        result["status"] = "no_numeric_features"
        return result

    train_values = prepared.train_values
    test_values = prepared.test_values
    model_feature_names = list(prepared.feature_names)
    if include_missing_masks:
        train_values = [values + mask for values, mask in zip(train_values, prepared.train_masks)]
        test_values = [values + mask for values, mask in zip(test_values, prepared.test_masks)]
        model_feature_names += [f"{name}__present_mask" for name in prepared.feature_names]

    history, train_probabilities, test_probabilities = _fit_mlp(
        torch=torch,
        train_values=train_values,
        train_labels=labels_train,
        test_values=test_values,
        test_labels=labels_test,
        epochs=epochs,
        learning_rate=learning_rate,
        hidden_units=hidden_units,
        batch_size=batch_size,
        weight_decay=weight_decay,
        seed=seed,
    )
    train_predictions = _predictions(train_probabilities, threshold=0.5)
    test_predictions = _predictions(test_probabilities, threshold=0.5)
    calibrated_threshold = _best_threshold(train_probabilities, labels_train)
    calibrated_train_predictions = _predictions(train_probabilities, threshold=calibrated_threshold)
    calibrated_test_predictions = _predictions(test_probabilities, threshold=calibrated_threshold)
    result.update(
        {
            "status": "evaluated",
            "model_feature_count": len(model_feature_names),
            "model_feature_names": model_feature_names,
            "default_threshold": 0.5,
            "train_metrics": _metrics(train_predictions, labels_train),
            "test_metrics": _metrics(test_predictions, labels_test),
            "calibrated_threshold": round(calibrated_threshold, 6),
            "calibrated_train_metrics": _metrics(calibrated_train_predictions, labels_train),
            "calibrated_test_metrics": _metrics(calibrated_test_predictions, labels_test),
            "test_probabilities": [round(value, 6) for value in test_probabilities],
            "test_predictions": test_predictions,
            "test_labels": labels_test,
            "history": history,
            "preprocessing": {
                name: {
                    "mean": round(mean, 8),
                    "scale": round(scale, 8),
                    "missing_rate": round(prepared.missing_rates[name], 8),
                }
                for name, mean, scale in zip(prepared.feature_names, prepared.means, prepared.scales)
            },
        }
    )
    return result


def _prepare_matrix(
    *,
    train_rows: list[dict[str, str]],
    test_rows: list[dict[str, str]],
    fieldnames: list[str],
    groups: tuple[str, ...] | None,
) -> PreparedMatrix:
    features = _feature_names(train_rows + test_rows, fieldnames, groups)
    means = [_mean_present(train_rows, feature) for feature in features]
    scales = [_scale_present(train_rows, feature, mean) for feature, mean in zip(features, means)]
    train_values, train_masks = _rows_to_matrix(train_rows, features, means, scales)
    test_values, test_masks = _rows_to_matrix(test_rows, features, means, scales)
    all_rows = train_rows + test_rows
    missing_rates = {
        feature: sum(1 for row in all_rows if row.get(feature, "") == "") / len(all_rows)
        for feature in features
    }
    return PreparedMatrix(
        feature_names=features,
        train_values=train_values,
        test_values=test_values,
        train_masks=train_masks,
        test_masks=test_masks,
        means=means,
        scales=scales,
        missing_rates=missing_rates,
    )


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
        present = [value for value in values if value != ""]
        if present and all(_is_float(value) for value in present):
            names.append(field)
    return names


def _rows_to_matrix(
    rows: list[dict[str, str]],
    features: list[str],
    means: list[float],
    scales: list[float],
) -> tuple[list[list[float]], list[list[float]]]:
    matrix = []
    masks = []
    for row in rows:
        values = []
        present_masks = []
        for feature, mean, scale in zip(features, means, scales):
            raw = row.get(feature, "")
            present = raw != ""
            value = float(raw) if present else mean
            values.append((value - mean) / scale)
            present_masks.append(1.0 if present else 0.0)
        matrix.append(values)
        masks.append(present_masks)
    return matrix, masks


def _fit_mlp(
    *,
    torch: object,
    train_values: list[list[float]],
    train_labels: list[int],
    test_values: list[list[float]],
    test_labels: list[int],
    epochs: int,
    learning_rate: float,
    hidden_units: int,
    batch_size: int,
    weight_decay: float,
    seed: int,
) -> tuple[dict[str, object], list[float], list[float]]:
    x_train = torch.tensor(train_values, dtype=torch.float32)
    y_train = torch.tensor(train_labels, dtype=torch.float32).unsqueeze(1)
    x_test = torch.tensor(test_values, dtype=torch.float32)
    y_test = torch.tensor(test_labels, dtype=torch.float32).unsqueeze(1)

    model = torch.nn.Sequential(
        torch.nn.Linear(x_train.shape[1], hidden_units),
        torch.nn.ReLU(),
        torch.nn.Linear(hidden_units, 1),
    )
    positives = sum(train_labels)
    negatives = len(train_labels) - positives
    pos_weight = torch.tensor([negatives / positives], dtype=torch.float32) if positives else torch.tensor([1.0])
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    generator = torch.Generator().manual_seed(seed)

    first_loss = None
    last_loss = None
    n_rows = len(train_values)
    for _ in range(epochs):
        permutation = torch.randperm(n_rows, generator=generator)
        epoch_loss = 0.0
        for start in range(0, n_rows, batch_size):
            batch_indices = permutation[start : start + batch_size]
            optimizer.zero_grad()
            logits = model(x_train[batch_indices])
            loss = criterion(logits, y_train[batch_indices])
            loss.backward()
            optimizer.step()
            epoch_loss += float(loss.item()) * len(batch_indices)
        last_loss = epoch_loss / n_rows
        if first_loss is None:
            first_loss = last_loss

    with torch.no_grad():
        train_probabilities = torch.sigmoid(model(x_train)).squeeze(1).tolist()
        test_probabilities = torch.sigmoid(model(x_test)).squeeze(1).tolist()
        test_loss = float(criterion(model(x_test), y_test).item()) if len(test_labels) else 0.0
    history = {
        "first_train_loss": round(first_loss or 0.0, 8),
        "last_train_loss": round(last_loss or 0.0, 8),
        "test_loss": round(test_loss, 8),
    }
    return history, [float(value) for value in train_probabilities], [float(value) for value in test_probabilities]


def _mean_present(rows: list[dict[str, str]], feature: str) -> float:
    values = [float(row[feature]) for row in rows if row.get(feature, "") != ""]
    return sum(values) / len(values) if values else 0.0


def _scale_present(rows: list[dict[str, str]], feature: str, mean: float) -> float:
    values = [float(row[feature]) for row in rows if row.get(feature, "") != ""]
    if not values:
        return 1.0
    variance = sum((value - mean) ** 2 for value in values) / len(values)
    scale = math.sqrt(variance)
    return scale if scale else 1.0


def _read_rows_and_fields(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        return list(reader), list(reader.fieldnames or [])


def _is_float(value: str) -> bool:
    try:
        float(value)
        return True
    except ValueError:
        return False


def _import_torch() -> object:
    try:
        import torch
    except ImportError as error:
        raise ValueError("PyTorch is required for train-torch-tabular-holdout") from error
    return torch


def _set_deterministic_seed(torch: object, seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(1)
    try:
        torch.use_deterministic_algorithms(True)
    except RuntimeError:
        pass


def _overall_status(report: dict[str, object]) -> str:
    evaluations = report.get("evaluations", {})
    statuses = {item["status"] for item in evaluations.values()}
    if "evaluated" in statuses:
        return "evaluated"
    if statuses == {"insufficient_train_class_variation"}:
        return "insufficient_train_class_variation"
    return "not_evaluated"


def _write_report(out_path: Path, report: dict[str, object]) -> dict[str, object]:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report

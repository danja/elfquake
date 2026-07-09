"""CPU PyTorch sequence head for synthetic event-list targets."""

from __future__ import annotations

import csv
import json
import math
import random
from pathlib import Path

from elfquake.models.learned_forecast import _best_balanced_threshold, _metrics, _number
from elfquake.models.synthetic_event_list_model import _feature_scores, is_diagnostic_or_target_field


def train_synthetic_event_list_sequence_head(
    *,
    input_csv: Path,
    out_path: Path,
    predictions_out: Path | None = None,
    target_field: str = "eventlist_target_occurred",
    target_status_field: str = "eventlist_target_status",
    group_field: str = "dataset_id",
    time_field: str = "window_start_utc",
    train_fraction: float = 0.8,
    lookback_rows: int = 12,
    epochs: int = 80,
    learning_rate: float = 0.001,
    hidden_units: int = 24,
    batch_size: int = 64,
    dropout: float = 0.1,
    weight_decay: float = 0.001,
    max_feature_count: int = 256,
    seed: int = 42,
) -> dict[str, object]:
    if not 0 < train_fraction < 1:
        raise ValueError("train_fraction must be between 0 and 1")
    if lookback_rows < 1:
        raise ValueError("lookback_rows must be positive")
    if epochs < 1 or hidden_units < 1 or batch_size < 1:
        raise ValueError("epochs, hidden_units, and batch_size must be positive")
    if not 0 <= dropout < 1:
        raise ValueError("dropout must be in [0, 1)")
    if weight_decay < 0 or max_feature_count < 0:
        raise ValueError("weight_decay and max_feature_count must be non-negative")

    torch = _import_torch()
    _set_deterministic_seed(torch, seed)
    rows, fieldnames = _read_rows(input_csv)
    labeled = [
        row
        for row in rows
        if row.get(target_field) in {"0", "1"}
        and (not target_status_field or row.get(target_status_field, "labeled") == "labeled")
    ]
    labeled = sorted(labeled, key=lambda row: (row.get(time_field, ""), row.get(group_field, "")))
    report: dict[str, object] = {
        "schema": "elfquake.synthetic_event_list_sequence_head.v1",
        "backend": "torch",
        "device": "cpu",
        "input_csv": str(input_csv),
        "predictions_csv": str(predictions_out) if predictions_out else "",
        "row_count": len(rows),
        "labeled_row_count": len(labeled),
        "target_field": target_field,
        "group_field": group_field,
        "time_field": time_field,
        "train_fraction": train_fraction,
        "lookback_rows": lookback_rows,
        "epochs": epochs,
        "learning_rate": learning_rate,
        "hidden_units": hidden_units,
        "batch_size": batch_size,
        "dropout": dropout,
        "weight_decay": weight_decay,
        "max_feature_count": max_feature_count,
        "seed": seed,
    }
    if len(labeled) < 4:
        report["status"] = "insufficient_labeled_rows"
        return _write_report(out_path, report)

    split_at = max(2, min(len(labeled) - 1, int(len(labeled) * train_fraction)))
    train_rows = labeled[:split_at]
    test_rows = labeled[split_at:]
    labels_train = [int(row[target_field]) for row in train_rows]
    labels_test = [int(row[target_field]) for row in test_rows]
    report.update(_split_summary(train_rows, test_rows, labels_train, labels_test, time_field=time_field))
    if sum(labels_train) == 0 or sum(labels_train) == len(labels_train):
        report["status"] = "insufficient_train_class_variation"
        return _write_report(out_path, report)

    feature_names = _numeric_feature_names(labeled, fieldnames)
    if not feature_names:
        report["status"] = "no_numeric_features"
        return _write_report(out_path, report)
    selected_features, feature_scores = _select_features(train_rows, labels_train, feature_names, max_feature_count=max_feature_count)
    grouped = _grouped_rows(labeled, group_field=group_field, time_field=time_field)
    train_x = _sequence_samples(train_rows, grouped, selected_features, group_field=group_field, lookback_rows=lookback_rows)
    test_x = _sequence_samples(test_rows, grouped, selected_features, group_field=group_field, lookback_rows=lookback_rows)
    train_x, test_x = _standardize_sequences(train_x, test_x)

    history, train_probabilities, test_probabilities = _fit_gru(
        torch=torch,
        train_x=train_x,
        train_labels=labels_train,
        test_x=test_x,
        test_labels=labels_test,
        epochs=epochs,
        learning_rate=learning_rate,
        hidden_units=hidden_units,
        batch_size=batch_size,
        dropout=dropout,
        weight_decay=weight_decay,
        seed=seed,
    )
    threshold = _best_balanced_threshold(train_probabilities, labels_train)
    prediction_rows = _prediction_rows(test_rows, labels_test, test_probabilities, threshold=threshold, target_field=target_field)
    if predictions_out:
        _write_prediction_csv(predictions_out, prediction_rows)
    report.update(
        {
            "status": "evaluated",
            "original_feature_count": len(feature_names),
            "feature_count": len(selected_features),
            "feature_selection": {
                "method": "train_split_standardized_mean_delta",
                "top_selected_features": feature_scores[:12],
            },
            "selected_feature_names": selected_features,
            "default_threshold": 0.5,
            "calibrated_threshold": round(threshold, 6),
            "train_metrics": _metrics(train_probabilities, labels_train, threshold=0.5),
            "test_metrics": _metrics(test_probabilities, labels_test, threshold=0.5),
            "calibrated_train_metrics": _metrics(train_probabilities, labels_train, threshold=threshold),
            "calibrated_test_metrics": _metrics(test_probabilities, labels_test, threshold=threshold),
            "history": history,
            "note": "Sequence head uses current and previous synthetic feature rows only; target and diagnostic fields are excluded.",
        }
    )
    return _write_report(out_path, report)


def summarize_synthetic_event_list_sequence_heads(
    *,
    root_dir: Path,
    out_path: Path,
    csv_out_path: Path | None = None,
) -> dict[str, object]:
    rows = []
    for path in sorted(root_dir.rglob("*.json")):
        try:
            report = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if report.get("schema") != "elfquake.synthetic_event_list_sequence_head.v1":
            continue
        rows.append(_sequence_summary_row(root_dir, path, report))
    config_rows = _config_summaries(rows)
    summary = {
        "schema": "elfquake.synthetic_event_list_sequence_head_summary.v1",
        "root_dir": str(root_dir),
        "report_count": len(rows),
        "reports": rows,
        "configs": config_rows,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if csv_out_path:
        _write_summary_csv(csv_out_path, rows)
    return summary


def _sequence_summary_row(root_dir: Path, path: Path, report: dict[str, object]) -> dict[str, object]:
    metrics = report.get("calibrated_test_metrics", {})
    train_metrics = report.get("calibrated_train_metrics", {})
    history = report.get("history", {})
    return {
        "path": str(path),
        "relative_path": str(path.relative_to(root_dir)),
        "status": report.get("status", ""),
        "lookback_rows": report.get("lookback_rows", ""),
        "hidden_units": report.get("hidden_units", ""),
        "dropout": report.get("dropout", ""),
        "weight_decay": report.get("weight_decay", ""),
        "max_feature_count": report.get("max_feature_count", ""),
        "seed": report.get("seed", ""),
        "feature_count": report.get("feature_count", ""),
        "train_row_count": report.get("train_row_count", ""),
        "test_row_count": report.get("test_row_count", ""),
        "train_positive_count": report.get("train_positive_count", ""),
        "test_positive_count": report.get("test_positive_count", ""),
        "calibrated_threshold": report.get("calibrated_threshold", ""),
        "balanced_accuracy": _nested_number(metrics, "balanced_accuracy"),
        "positive_recall": _nested_number(metrics, "positive_recall"),
        "negative_recall": _nested_number(metrics, "negative_recall"),
        "train_balanced_accuracy": _nested_number(train_metrics, "balanced_accuracy"),
        "first_train_loss": _nested_number(history, "first_train_loss"),
        "last_train_loss": _nested_number(history, "last_train_loss"),
        "test_loss": _nested_number(history, "test_loss"),
    }


def _config_summaries(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    groups: dict[tuple[object, ...], list[float]] = {}
    for row in rows:
        if row.get("status") != "evaluated":
            continue
        key = (
            row.get("lookback_rows"),
            row.get("hidden_units"),
            row.get("dropout"),
            row.get("weight_decay"),
            row.get("max_feature_count"),
        )
        groups.setdefault(key, []).append(float(row.get("balanced_accuracy") or 0.0))
    output = []
    for key, values in sorted(groups.items(), key=lambda item: (_mean(item[1]), -_stddev(item[1])), reverse=True):
        output.append(
            {
                "lookback_rows": key[0],
                "hidden_units": key[1],
                "dropout": key[2],
                "weight_decay": key[3],
                "max_feature_count": key[4],
                "run_count": len(values),
                "mean_balanced_accuracy": round(_mean(values), 6),
                "min_balanced_accuracy": round(min(values), 6),
                "max_balanced_accuracy": round(max(values), 6),
                "stddev_balanced_accuracy": round(_stddev(values), 6),
                "pass_count": sum(1 for value in values if value >= 0.6),
            }
        )
    return output


def _write_summary_csv(path: Path, rows: list[dict[str, object]]) -> None:
    fieldnames = [
        "status",
        "lookback_rows",
        "hidden_units",
        "dropout",
        "weight_decay",
        "max_feature_count",
        "seed",
        "feature_count",
        "train_row_count",
        "test_row_count",
        "balanced_accuracy",
        "positive_recall",
        "negative_recall",
        "train_balanced_accuracy",
        "first_train_loss",
        "last_train_loss",
        "test_loss",
        "relative_path",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _nested_number(value: object, key: str) -> object:
    if not isinstance(value, dict):
        return ""
    item = value.get(key, "")
    return item if isinstance(item, (int, float)) else ""


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _stddev(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    mean = _mean(values)
    return math.sqrt(sum((value - mean) ** 2 for value in values) / len(values))


def _read_rows(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        return list(reader), list(reader.fieldnames or [])


def _numeric_feature_names(rows: list[dict[str, str]], fieldnames: list[str]) -> list[str]:
    output = []
    for name in fieldnames:
        if is_diagnostic_or_target_field(name):
            continue
        values = [_number(row.get(name, "")) for row in rows]
        if any(value != 0.0 for value in values):
            output.append(name)
    return output


def _select_features(
    train_rows: list[dict[str, str]],
    labels_train: list[int],
    feature_names: list[str],
    *,
    max_feature_count: int,
) -> tuple[list[str], list[dict[str, object]]]:
    matrix = [[_number(row.get(name, "")) for name in feature_names] for row in train_rows]
    scores = _feature_scores(matrix, labels_train, feature_names)
    if max_feature_count == 0 or max_feature_count >= len(feature_names):
        return list(feature_names), scores
    selected = {str(row["name"]) for row in scores[:max_feature_count]}
    return [name for name in feature_names if name in selected], [row for row in scores if str(row["name"]) in selected]


def _grouped_rows(
    rows: list[dict[str, str]],
    *,
    group_field: str,
    time_field: str,
) -> dict[str, list[dict[str, str]]]:
    groups: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        groups.setdefault(row.get(group_field, ""), []).append(row)
    return {group: sorted(group_rows, key=lambda row: row.get(time_field, "")) for group, group_rows in groups.items()}


def _sequence_samples(
    rows: list[dict[str, str]],
    grouped: dict[str, list[dict[str, str]]],
    feature_names: list[str],
    *,
    group_field: str,
    lookback_rows: int,
) -> list[list[list[float]]]:
    feature_count = len(feature_names)
    output = []
    for row in rows:
        group_rows = grouped.get(row.get(group_field, ""), [])
        index = group_rows.index(row)
        start = max(0, index - lookback_rows + 1)
        window = group_rows[start : index + 1]
        padded = [[0.0 for _ in range(feature_count)] for _ in range(lookback_rows - len(window))]
        padded.extend([[_number(item.get(name, "")) for name in feature_names] for item in window])
        output.append(padded)
    return output


def _standardize_sequences(
    train_x: list[list[list[float]]],
    test_x: list[list[list[float]]],
) -> tuple[list[list[list[float]]], list[list[list[float]]]]:
    feature_count = len(train_x[0][0])
    means = []
    scales = []
    for index in range(feature_count):
        values = [step[index] for sample in train_x for step in sample]
        mean = sum(values) / len(values)
        variance = sum((value - mean) ** 2 for value in values) / len(values)
        means.append(mean)
        scales.append(math.sqrt(variance) if variance > 0 else 1.0)
    return _apply_standardization(train_x, means, scales), _apply_standardization(test_x, means, scales)


def _apply_standardization(samples: list[list[list[float]]], means: list[float], scales: list[float]) -> list[list[list[float]]]:
    return [[[(value - means[index]) / scales[index] for index, value in enumerate(step)] for step in sample] for sample in samples]


def _fit_gru(
    *,
    torch: object,
    train_x: list[list[list[float]]],
    train_labels: list[int],
    test_x: list[list[list[float]]],
    test_labels: list[int],
    epochs: int,
    learning_rate: float,
    hidden_units: int,
    batch_size: int,
    dropout: float,
    weight_decay: float,
    seed: int,
) -> tuple[dict[str, float], list[float], list[float]]:
    x_train = torch.tensor(train_x, dtype=torch.float32)
    y_train = torch.tensor(train_labels, dtype=torch.float32).unsqueeze(1)
    x_test = torch.tensor(test_x, dtype=torch.float32)
    y_test = torch.tensor(test_labels, dtype=torch.float32).unsqueeze(1)
    model = _RegularizedGruClassifier(torch, input_size=x_train.shape[2], hidden_units=hidden_units, dropout=dropout)
    positives = sum(train_labels)
    negatives = len(train_labels) - positives
    pos_weight = torch.tensor([negatives / positives], dtype=torch.float32) if positives else torch.tensor([1.0])
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    generator = torch.Generator().manual_seed(seed)
    first_loss = None
    last_loss = 0.0
    for _ in range(epochs):
        model.train()
        permutation = torch.randperm(len(train_x), generator=generator)
        epoch_loss = 0.0
        for start in range(0, len(train_x), batch_size):
            batch_indices = permutation[start : start + batch_size]
            optimizer.zero_grad()
            loss = criterion(model(x_train[batch_indices]), y_train[batch_indices])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            epoch_loss += float(loss.item()) * len(batch_indices)
        last_loss = epoch_loss / len(train_x)
        if first_loss is None:
            first_loss = last_loss
    model.eval()
    with torch.no_grad():
        train_logits = model(x_train)
        test_logits = model(x_test)
        train_probabilities = torch.sigmoid(train_logits).squeeze(1).tolist()
        test_probabilities = torch.sigmoid(test_logits).squeeze(1).tolist()
        test_loss = float(criterion(test_logits, y_test).item())
    return (
        {"first_train_loss": round(first_loss or 0.0, 8), "last_train_loss": round(last_loss, 8), "test_loss": round(test_loss, 8)},
        [float(value) for value in train_probabilities],
        [float(value) for value in test_probabilities],
    )


class _RegularizedGruClassifier:
    def __init__(self, torch: object, *, input_size: int, hidden_units: int, dropout: float) -> None:
        self.torch = torch
        self.gru = torch.nn.GRU(input_size=input_size, hidden_size=hidden_units, batch_first=True)
        self.dropout = torch.nn.Dropout(dropout)
        self.head = torch.nn.Linear(hidden_units, 1)

    def parameters(self):
        return [*self.gru.parameters(), *self.head.parameters()]

    def train(self) -> None:
        self.gru.train()
        self.dropout.train()
        self.head.train()

    def eval(self) -> None:
        self.gru.eval()
        self.dropout.eval()
        self.head.eval()

    def __call__(self, x):
        _, hidden = self.gru(x)
        return self.head(self.dropout(hidden[-1]))


def _split_summary(
    train_rows: list[dict[str, str]],
    test_rows: list[dict[str, str]],
    labels_train: list[int],
    labels_test: list[int],
    *,
    time_field: str,
) -> dict[str, object]:
    return {
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
    }


def _prediction_rows(
    rows: list[dict[str, str]],
    labels: list[int],
    probabilities: list[float],
    *,
    threshold: float,
    target_field: str,
) -> list[dict[str, str]]:
    output = []
    for index, (row, label, probability) in enumerate(zip(rows, labels, probabilities), start=1):
        output.append(
            {
                "row_index": str(index),
                "dataset_id": row.get("dataset_id", ""),
                "window_id": row.get("window_id", ""),
                "window_start_utc": row.get("window_start_utc", ""),
                "target_field": target_field,
                "target_occurred": str(label),
                "predicted_probability": f"{probability:.6f}",
                "predicted_occurred": "1" if probability >= threshold else "0",
            }
        )
    return output


def _write_prediction_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys()) if rows else ["row_index"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _import_torch() -> object:
    try:
        import torch
    except ImportError as error:
        raise ValueError("PyTorch is required for train-synthetic-event-list-sequence-head") from error
    return torch


def _set_deterministic_seed(torch: object, seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True, warn_only=True)


def _write_report(out_path: Path, report: dict[str, object]) -> dict[str, object]:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report

"""CPU PyTorch sequence classifier for materialized sequence manifests."""

from __future__ import annotations

import csv
import json
import math
import random
from pathlib import Path

from elfquake.models.temporal_holdout import _baselines, _best_threshold, _metrics, _predictions
from elfquake.models.torch_sequence_data import SequenceDataset, build_sequence_samples, load_sequence_datasets


SEQUENCE_EVALUATIONS = {
    "sequence_direct_avalanche_only": ("synthetic_direct_avalanche",),
    "sequence_piezo_vlf_only": ("synthetic_piezo_vlf",),
    "sequence_direct_avalanche_piezo_vlf": ("synthetic_direct_avalanche", "synthetic_piezo_vlf"),
    "sequence_full": ("synthetic_direct_avalanche", "synthetic_piezo_vlf", "synthetic_summary"),
}


def evaluate_torch_sequence_holdout(
    *,
    input_csv: Path,
    sequence_manifest_paths: list[Path],
    out_path: Path,
    time_field: str = "window_start_utc",
    train_fraction: float = 0.8,
    lookback_steps: int = 60,
    epochs: int = 40,
    learning_rate: float = 0.001,
    hidden_units: int = 24,
    batch_size: int = 64,
    seed: int = 42,
    include_missing_masks: bool = True,
    evaluation_names: list[str] | None = None,
) -> dict[str, object]:
    if not 0 < train_fraction < 1:
        raise ValueError("train_fraction must be between 0 and 1")
    rows, fieldnames = _read_rows_and_fields(input_csv)
    labeled = sorted([row for row in rows if row.get("target_occurred") in {"0", "1"}], key=lambda row: row.get(time_field, ""))
    report = _base_report(
        schema="elfquake.torch_sequence_holdout.v1",
        input_csv=input_csv,
        sequence_manifest_paths=sequence_manifest_paths,
        row_count=len(rows),
        labeled_count=len(labeled),
        lookback_steps=lookback_steps,
        epochs=epochs,
        learning_rate=learning_rate,
        hidden_units=hidden_units,
        batch_size=batch_size,
        seed=seed,
        include_missing_masks=include_missing_masks,
        evaluation_names=evaluation_names,
    )
    report["time_field"] = time_field
    report["train_fraction"] = train_fraction
    if len(labeled) < 4:
        report["status"] = "insufficient_labeled_rows"
        return _write_report(out_path, report)

    train_count = min(len(labeled) - 1, max(2, int(len(labeled) * train_fraction)))
    train_rows = labeled[:train_count]
    test_rows = labeled[train_count:]
    _add_temporal_split(report, train_rows, test_rows, time_field=time_field)
    sequences = load_sequence_datasets(sequence_manifest_paths, include_missing_masks=include_missing_masks)
    _evaluate_all(
        report=report,
        sequences=sequences,
        train_rows=train_rows,
        test_rows=test_rows,
        lookback_steps=lookback_steps,
        epochs=epochs,
        learning_rate=learning_rate,
        hidden_units=hidden_units,
        batch_size=batch_size,
        seed=seed,
        evaluation_names=evaluation_names,
    )
    report["status"] = _overall_status(report)
    return _write_report(out_path, report)


def evaluate_torch_sequence_group_holdout(
    *,
    input_csv: Path,
    sequence_manifest_paths: list[Path],
    out_path: Path,
    group_field: str = "dataset_id",
    test_group: str,
    lookback_steps: int = 60,
    epochs: int = 40,
    learning_rate: float = 0.001,
    hidden_units: int = 24,
    batch_size: int = 64,
    seed: int = 42,
    include_missing_masks: bool = True,
    evaluation_names: list[str] | None = None,
) -> dict[str, object]:
    rows, _ = _read_rows_and_fields(input_csv)
    labeled = [row for row in rows if row.get("target_occurred") in {"0", "1"}]
    train_rows = [row for row in labeled if row.get(group_field, "") != test_group]
    test_rows = [row for row in labeled if row.get(group_field, "") == test_group]
    report = _base_report(
        schema="elfquake.torch_sequence_group_holdout.v1",
        input_csv=input_csv,
        sequence_manifest_paths=sequence_manifest_paths,
        row_count=len(rows),
        labeled_count=len(labeled),
        lookback_steps=lookback_steps,
        epochs=epochs,
        learning_rate=learning_rate,
        hidden_units=hidden_units,
        batch_size=batch_size,
        seed=seed,
        include_missing_masks=include_missing_masks,
        evaluation_names=evaluation_names,
    )
    report["group_field"] = group_field
    report["test_group"] = test_group
    report["train_groups"] = sorted({row.get(group_field, "") for row in train_rows})
    if len(train_rows) < 2 or len(test_rows) < 1:
        report["status"] = "insufficient_group_rows"
        return _write_report(out_path, report)

    labels_train = [int(row["target_occurred"]) for row in train_rows]
    labels_test = [int(row["target_occurred"]) for row in test_rows]
    report.update(_split_counts(train_rows, test_rows, labels_train, labels_test))
    sequences = load_sequence_datasets(sequence_manifest_paths, include_missing_masks=include_missing_masks)
    _evaluate_all(
        report=report,
        sequences=sequences,
        train_rows=train_rows,
        test_rows=test_rows,
        lookback_steps=lookback_steps,
        epochs=epochs,
        learning_rate=learning_rate,
        hidden_units=hidden_units,
        batch_size=batch_size,
        seed=seed,
        evaluation_names=evaluation_names,
    )
    report["status"] = _overall_status(report)
    return _write_report(out_path, report)


def evaluate_torch_sequence_split_holdout(
    *,
    input_csv: Path,
    sequence_manifest_paths: list[Path],
    out_path: Path,
    split_field: str = "model_split",
    train_value: str = "train",
    test_value: str = "test",
    lookback_steps: int = 60,
    epochs: int = 40,
    learning_rate: float = 0.001,
    hidden_units: int = 24,
    batch_size: int = 64,
    seed: int = 42,
    include_missing_masks: bool = True,
    evaluation_names: list[str] | None = None,
) -> dict[str, object]:
    rows, _ = _read_rows_and_fields(input_csv)
    labeled = [row for row in rows if row.get("target_occurred") in {"0", "1"}]
    train_rows = [row for row in labeled if row.get(split_field, "") == train_value]
    test_rows = [row for row in labeled if row.get(split_field, "") == test_value]
    report = _base_report(
        schema="elfquake.torch_sequence_split_holdout.v1",
        input_csv=input_csv,
        sequence_manifest_paths=sequence_manifest_paths,
        row_count=len(rows),
        labeled_count=len(labeled),
        lookback_steps=lookback_steps,
        epochs=epochs,
        learning_rate=learning_rate,
        hidden_units=hidden_units,
        batch_size=batch_size,
        seed=seed,
        include_missing_masks=include_missing_masks,
        evaluation_names=evaluation_names,
    )
    report["split_field"] = split_field
    report["train_value"] = train_value
    report["test_value"] = test_value
    if len(train_rows) < 2 or len(test_rows) < 1:
        report["status"] = "insufficient_split_rows"
        return _write_report(out_path, report)

    labels_train = [int(row["target_occurred"]) for row in train_rows]
    labels_test = [int(row["target_occurred"]) for row in test_rows]
    report.update(_split_counts(train_rows, test_rows, labels_train, labels_test))
    sequences = load_sequence_datasets(sequence_manifest_paths, include_missing_masks=include_missing_masks)
    _evaluate_all(
        report=report,
        sequences=sequences,
        train_rows=train_rows,
        test_rows=test_rows,
        lookback_steps=lookback_steps,
        epochs=epochs,
        learning_rate=learning_rate,
        hidden_units=hidden_units,
        batch_size=batch_size,
        seed=seed,
        evaluation_names=evaluation_names,
    )
    report["status"] = _overall_status(report)
    return _write_report(out_path, report)


def _base_report(
    *,
    schema: str,
    input_csv: Path,
    sequence_manifest_paths: list[Path],
    row_count: int,
    labeled_count: int,
    lookback_steps: int,
    epochs: int,
    learning_rate: float,
    hidden_units: int,
    batch_size: int,
    seed: int,
    include_missing_masks: bool,
    evaluation_names: list[str] | None,
) -> dict[str, object]:
    selected = _selected_evaluations(evaluation_names)
    return {
        "schema": schema,
        "backend": "torch",
        "device": "cpu",
        "input": str(input_csv),
        "sequence_manifests": [str(path) for path in sequence_manifest_paths],
        "row_count": row_count,
        "labeled_row_count": labeled_count,
        "lookback_steps": lookback_steps,
        "epochs": epochs,
        "learning_rate": learning_rate,
        "hidden_units": hidden_units,
        "batch_size": batch_size,
        "seed": seed,
        "include_missing_masks": include_missing_masks,
        "selected_evaluations": list(selected),
        "evaluations": {},
    }


def _add_temporal_split(report: dict[str, object], train_rows: list[dict[str, str]], test_rows: list[dict[str, str]], *, time_field: str) -> None:
    labels_train = [int(row["target_occurred"]) for row in train_rows]
    labels_test = [int(row["target_occurred"]) for row in test_rows]
    report.update(_split_counts(train_rows, test_rows, labels_train, labels_test))
    report.update(
        {
            "train_time_start": train_rows[0].get(time_field, ""),
            "train_time_end": train_rows[-1].get(time_field, ""),
            "test_time_start": test_rows[0].get(time_field, ""),
            "test_time_end": test_rows[-1].get(time_field, ""),
        }
    )


def _split_counts(
    train_rows: list[dict[str, str]],
    test_rows: list[dict[str, str]],
    labels_train: list[int],
    labels_test: list[int],
) -> dict[str, object]:
    return {
        "train_row_count": len(train_rows),
        "test_row_count": len(test_rows),
        "train_positive_count": sum(labels_train),
        "train_negative_count": len(labels_train) - sum(labels_train),
        "test_positive_count": sum(labels_test),
        "test_negative_count": len(labels_test) - sum(labels_test),
        "baselines": _baselines(labels_train, labels_test),
    }


def _evaluate_all(
    *,
    report: dict[str, object],
    sequences: dict[tuple[str, str], SequenceDataset],
    train_rows: list[dict[str, str]],
    test_rows: list[dict[str, str]],
    lookback_steps: int,
    epochs: int,
    learning_rate: float,
    hidden_units: int,
    batch_size: int,
    seed: int,
    evaluation_names: list[str] | None,
) -> None:
    for name, modalities in _selected_evaluations(evaluation_names).items():
        report["evaluations"][name] = _evaluate_one(
            sequences=sequences,
            train_rows=train_rows,
            test_rows=test_rows,
            modalities=modalities,
            lookback_steps=lookback_steps,
            epochs=epochs,
            learning_rate=learning_rate,
            hidden_units=hidden_units,
            batch_size=batch_size,
            seed=seed,
        )


def _evaluate_one(
    *,
    sequences: dict[tuple[str, str], SequenceDataset],
    train_rows: list[dict[str, str]],
    test_rows: list[dict[str, str]],
    modalities: tuple[str, ...],
    lookback_steps: int,
    epochs: int,
    learning_rate: float,
    hidden_units: int,
    batch_size: int,
    seed: int,
) -> dict[str, object]:
    original_train_count = len(train_rows)
    original_test_count = len(test_rows)
    train_rows = _covered_rows(train_rows, sequences, modalities)
    test_rows = _covered_rows(test_rows, sequences, modalities)
    labels_train = [int(row["target_occurred"]) for row in train_rows]
    labels_test = [int(row["target_occurred"]) for row in test_rows]
    result: dict[str, object] = {"modalities": list(modalities), "train_row_count": len(train_rows), "test_row_count": len(test_rows)}
    result["dropped_train_row_count"] = original_train_count - len(train_rows)
    result["dropped_test_row_count"] = original_test_count - len(test_rows)
    if len(train_rows) < 2 or len(test_rows) < 1:
        result["status"] = "insufficient_sequence_covered_rows"
        return result
    if sum(labels_train) == 0 or sum(labels_train) == len(labels_train):
        result["status"] = "insufficient_train_class_variation"
        return result
    try:
        train_x, feature_names = build_sequence_samples(train_rows, sequences, modalities=modalities, lookback_steps=lookback_steps)
        test_x, _ = build_sequence_samples(test_rows, sequences, modalities=modalities, lookback_steps=lookback_steps)
    except ValueError as error:
        result["status"] = "missing_sequence_data"
        result["error"] = str(error)
        return result
    if not feature_names:
        result["status"] = "no_sequence_features"
        return result

    train_x, test_x = _standardize_sequences(train_x, test_x)
    history, train_probabilities, test_probabilities = _fit_gru(
        train_x=train_x,
        train_labels=labels_train,
        test_x=test_x,
        test_labels=labels_test,
        epochs=epochs,
        learning_rate=learning_rate,
        hidden_units=hidden_units,
        batch_size=batch_size,
        seed=seed,
    )
    calibrated_threshold = _best_threshold(train_probabilities, labels_train)
    result.update(
        {
            "status": "evaluated",
            "feature_count": len(feature_names),
            "feature_names": feature_names,
            "default_threshold": 0.5,
            "train_metrics": _metrics(_predictions(train_probabilities, threshold=0.5), labels_train),
            "test_metrics": _metrics(_predictions(test_probabilities, threshold=0.5), labels_test),
            "calibrated_threshold": round(calibrated_threshold, 6),
            "calibrated_train_metrics": _metrics(_predictions(train_probabilities, threshold=calibrated_threshold), labels_train),
            "calibrated_test_metrics": _metrics(_predictions(test_probabilities, threshold=calibrated_threshold), labels_test),
            "test_probabilities": [round(value, 6) for value in test_probabilities],
            "test_predictions": _predictions(test_probabilities, threshold=0.5),
            "test_labels": labels_test,
            "history": history,
        }
    )
    return result


def _covered_rows(
    rows: list[dict[str, str]],
    sequences: dict[tuple[str, str], SequenceDataset],
    modalities: tuple[str, ...],
) -> list[dict[str, str]]:
    covered = []
    for row in rows:
        dataset_id = row.get("dataset_id", "")
        end_time = row.get("window_end_utc", "")
        if all(
            (dataset := sequences.get((dataset_id, modality))) is not None
            and end_time in dataset.time_to_index
            for modality in modalities
        ):
            covered.append(row)
    return covered


def _fit_gru(
    *,
    train_x: list[list[list[float]]],
    train_labels: list[int],
    test_x: list[list[list[float]]],
    test_labels: list[int],
    epochs: int,
    learning_rate: float,
    hidden_units: int,
    batch_size: int,
    seed: int,
) -> tuple[dict[str, float], list[float], list[float]]:
    torch = _import_torch()
    _set_deterministic_seed(torch, seed)
    x_train = torch.tensor(train_x, dtype=torch.float32)
    y_train = torch.tensor(train_labels, dtype=torch.float32).unsqueeze(1)
    x_test = torch.tensor(test_x, dtype=torch.float32)
    y_test = torch.tensor(test_labels, dtype=torch.float32).unsqueeze(1)
    model = _GruClassifier(torch, input_size=x_train.shape[2], hidden_units=hidden_units)
    positives = sum(train_labels)
    negatives = len(train_labels) - positives
    pos_weight = torch.tensor([negatives / positives], dtype=torch.float32) if positives else torch.tensor([1.0])
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    generator = torch.Generator().manual_seed(seed)
    first_loss = None
    last_loss = 0.0
    for _ in range(epochs):
        permutation = torch.randperm(len(train_x), generator=generator)
        epoch_loss = 0.0
        for start in range(0, len(train_x), batch_size):
            batch_indices = permutation[start : start + batch_size]
            optimizer.zero_grad()
            loss = criterion(model(x_train[batch_indices]), y_train[batch_indices])
            loss.backward()
            optimizer.step()
            epoch_loss += float(loss.item()) * len(batch_indices)
        last_loss = epoch_loss / len(train_x)
        if first_loss is None:
            first_loss = last_loss
    with torch.no_grad():
        train_probabilities = torch.sigmoid(model(x_train)).squeeze(1).tolist()
        test_probabilities = torch.sigmoid(model(x_test)).squeeze(1).tolist()
        test_loss = float(criterion(model(x_test), y_test).item())
    return (
        {"first_train_loss": round(first_loss or 0.0, 8), "last_train_loss": round(last_loss, 8), "test_loss": round(test_loss, 8)},
        [float(value) for value in train_probabilities],
        [float(value) for value in test_probabilities],
    )


class _GruClassifier:
    def __init__(self, torch: object, *, input_size: int, hidden_units: int) -> None:
        self.torch = torch
        self.gru = torch.nn.GRU(input_size=input_size, hidden_size=hidden_units, batch_first=True)
        self.head = torch.nn.Linear(hidden_units, 1)

    def parameters(self):
        return [*self.gru.parameters(), *self.head.parameters()]

    def __call__(self, x):
        _, hidden = self.gru(x)
        return self.head(hidden[-1])


def _standardize_sequences(train_x: list[list[list[float]]], test_x: list[list[list[float]]]) -> tuple[list[list[list[float]]], list[list[list[float]]]]:
    feature_count = len(train_x[0][0])
    flat = [[step[index] for sample in train_x for step in sample] for index in range(feature_count)]
    means = [sum(values) / len(values) for values in flat]
    scales = []
    for values, mean in zip(flat, means):
        scale = math.sqrt(sum((value - mean) ** 2 for value in values) / len(values))
        scales.append(scale if scale else 1.0)
    return _apply_standardization(train_x, means, scales), _apply_standardization(test_x, means, scales)


def _apply_standardization(samples: list[list[list[float]]], means: list[float], scales: list[float]) -> list[list[list[float]]]:
    return [[[(value - means[index]) / scales[index] for index, value in enumerate(step)] for step in sample] for sample in samples]


def _read_rows_and_fields(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        return list(reader), list(reader.fieldnames or [])


def _import_torch() -> object:
    try:
        import torch
    except ImportError as error:
        raise ValueError("PyTorch is required for train-torch-sequence-holdout") from error
    return torch


def _set_deterministic_seed(torch: object, seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True, warn_only=True)


def _overall_status(report: dict[str, object]) -> str:
    statuses = {item["status"] for item in report.get("evaluations", {}).values()}
    if "evaluated" in statuses:
        return "evaluated"
    if statuses == {"insufficient_train_class_variation"}:
        return "insufficient_train_class_variation"
    return "not_evaluated"


def _selected_evaluations(evaluation_names: list[str] | None) -> dict[str, tuple[str, ...]]:
    if not evaluation_names:
        return dict(SEQUENCE_EVALUATIONS)
    unknown = sorted(set(evaluation_names) - set(SEQUENCE_EVALUATIONS))
    if unknown:
        raise ValueError(f"unknown sequence evaluation(s): {', '.join(unknown)}")
    return {name: SEQUENCE_EVALUATIONS[name] for name in evaluation_names}


def _write_report(out_path: Path, report: dict[str, object]) -> dict[str, object]:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report

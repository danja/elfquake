"""CPU PyTorch tiny patch Transformer for materialized sequence manifests."""

from __future__ import annotations

import json
import math
import random
from pathlib import Path

from elfquake.models.temporal_holdout import _best_threshold, _metrics, _predictions
from elfquake.models.torch_sequence import (
    _covered_rows,
    _overall_status,
    _read_rows_and_fields,
    _selected_evaluations,
    _split_counts,
    _standardize_sequences,
)
from elfquake.models.torch_sequence_data import SequenceDataset, build_sequence_samples, load_sequence_datasets


def evaluate_torch_patch_transformer_split_holdout(
    *,
    input_csv: Path,
    sequence_manifest_paths: list[Path],
    out_path: Path,
    split_field: str = "model_split",
    train_value: str = "train",
    test_value: str = "test",
    lookback_steps: int = 60,
    patch_steps: int = 10,
    epochs: int = 20,
    learning_rate: float = 0.001,
    d_model: int = 32,
    layers: int = 2,
    heads: int = 2,
    dropout: float = 0.1,
    batch_size: int = 32,
    seed: int = 42,
    include_missing_masks: bool = True,
    evaluation_names: list[str] | None = None,
    checkpoint_in: Path | None = None,
    checkpoint_out: Path | None = None,
) -> dict[str, object]:
    if patch_steps < 1:
        raise ValueError("patch_steps must be at least 1")
    if d_model < 1:
        raise ValueError("d_model must be at least 1")
    if layers < 1:
        raise ValueError("layers must be at least 1")
    if heads < 1 or d_model % heads != 0:
        raise ValueError("heads must divide d_model")
    if not 0 <= dropout < 1:
        raise ValueError("dropout must be in [0, 1)")

    rows, _ = _read_rows_and_fields(input_csv)
    labeled = [row for row in rows if row.get("target_occurred") in {"0", "1"}]
    train_rows = [row for row in labeled if row.get(split_field, "") == train_value]
    test_rows = [row for row in labeled if row.get(split_field, "") == test_value]
    report = _base_report(
        input_csv=input_csv,
        sequence_manifest_paths=sequence_manifest_paths,
        row_count=len(rows),
        labeled_count=len(labeled),
        split_field=split_field,
        train_value=train_value,
        test_value=test_value,
        lookback_steps=lookback_steps,
        patch_steps=patch_steps,
        epochs=epochs,
        learning_rate=learning_rate,
        d_model=d_model,
        layers=layers,
        heads=heads,
        dropout=dropout,
        batch_size=batch_size,
        seed=seed,
        include_missing_masks=include_missing_masks,
        evaluation_names=evaluation_names,
        checkpoint_in=checkpoint_in,
        checkpoint_out=checkpoint_out,
    )
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
        patch_steps=patch_steps,
        epochs=epochs,
        learning_rate=learning_rate,
        d_model=d_model,
        layers=layers,
        heads=heads,
        dropout=dropout,
        batch_size=batch_size,
        seed=seed,
        evaluation_names=evaluation_names,
        checkpoint_in=checkpoint_in,
        checkpoint_out=checkpoint_out,
    )
    report["status"] = _overall_status(report)
    return _write_report(out_path, report)


def _base_report(
    *,
    input_csv: Path,
    sequence_manifest_paths: list[Path],
    row_count: int,
    labeled_count: int,
    split_field: str,
    train_value: str,
    test_value: str,
    lookback_steps: int,
    patch_steps: int,
    epochs: int,
    learning_rate: float,
    d_model: int,
    layers: int,
    heads: int,
    dropout: float,
    batch_size: int,
    seed: int,
    include_missing_masks: bool,
    evaluation_names: list[str] | None,
    checkpoint_in: Path | None,
    checkpoint_out: Path | None,
) -> dict[str, object]:
    selected = _selected_evaluations(evaluation_names)
    return {
        "schema": "elfquake.torch_patch_transformer_split_holdout.v1",
        "backend": "torch",
        "device": "cpu",
        "input": str(input_csv),
        "sequence_manifests": [str(path) for path in sequence_manifest_paths],
        "row_count": row_count,
        "labeled_row_count": labeled_count,
        "split_field": split_field,
        "train_value": train_value,
        "test_value": test_value,
        "lookback_steps": lookback_steps,
        "patch_steps": patch_steps,
        "epochs": epochs,
        "learning_rate": learning_rate,
        "d_model": d_model,
        "layers": layers,
        "heads": heads,
        "dropout": dropout,
        "batch_size": batch_size,
        "seed": seed,
        "include_missing_masks": include_missing_masks,
        "checkpoint_in": str(checkpoint_in) if checkpoint_in else "",
        "checkpoint_out": str(checkpoint_out) if checkpoint_out else "",
        "selected_evaluations": list(selected),
        "evaluations": {},
    }


def _evaluate_all(
    *,
    report: dict[str, object],
    sequences: dict[tuple[str, str], SequenceDataset],
    train_rows: list[dict[str, str]],
    test_rows: list[dict[str, str]],
    lookback_steps: int,
    patch_steps: int,
    epochs: int,
    learning_rate: float,
    d_model: int,
    layers: int,
    heads: int,
    dropout: float,
    batch_size: int,
    seed: int,
    evaluation_names: list[str] | None,
    checkpoint_in: Path | None,
    checkpoint_out: Path | None,
) -> None:
    selected = _selected_evaluations(evaluation_names)
    last_name = next(reversed(selected)) if selected else ""
    for name, modalities in selected.items():
        report["evaluations"][name] = _evaluate_one(
            sequences=sequences,
            train_rows=train_rows,
            test_rows=test_rows,
            modalities=modalities,
            lookback_steps=lookback_steps,
            patch_steps=patch_steps,
            epochs=epochs,
            learning_rate=learning_rate,
            d_model=d_model,
            layers=layers,
            heads=heads,
            dropout=dropout,
            batch_size=batch_size,
            seed=seed,
            checkpoint_in=checkpoint_in,
            checkpoint_out=checkpoint_out if name == last_name else None,
        )


def _evaluate_one(
    *,
    sequences: dict[tuple[str, str], SequenceDataset],
    train_rows: list[dict[str, str]],
    test_rows: list[dict[str, str]],
    modalities: tuple[str, ...],
    lookback_steps: int,
    patch_steps: int,
    epochs: int,
    learning_rate: float,
    d_model: int,
    layers: int,
    heads: int,
    dropout: float,
    batch_size: int,
    seed: int,
    checkpoint_in: Path | None,
    checkpoint_out: Path | None,
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
    history, train_probabilities, test_probabilities = _fit_patch_transformer(
        train_x=train_x,
        train_labels=labels_train,
        test_x=test_x,
        test_labels=labels_test,
        patch_steps=patch_steps,
        epochs=epochs,
        learning_rate=learning_rate,
        d_model=d_model,
        layers=layers,
        heads=heads,
        dropout=dropout,
        batch_size=batch_size,
        seed=seed,
        checkpoint_in=checkpoint_in,
        checkpoint_out=checkpoint_out,
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


def _fit_patch_transformer(
    *,
    train_x: list[list[list[float]]],
    train_labels: list[int],
    test_x: list[list[list[float]]],
    test_labels: list[int],
    patch_steps: int,
    epochs: int,
    learning_rate: float,
    d_model: int,
    layers: int,
    heads: int,
    dropout: float,
    batch_size: int,
    seed: int,
    checkpoint_in: Path | None,
    checkpoint_out: Path | None,
) -> tuple[dict[str, object], list[float], list[float]]:
    torch = _import_torch()
    _set_deterministic_seed(torch, seed)
    x_train = _patchify(torch.tensor(train_x, dtype=torch.float32), patch_steps=patch_steps, torch=torch)
    y_train = torch.tensor(train_labels, dtype=torch.float32).unsqueeze(1)
    x_test = _patchify(torch.tensor(test_x, dtype=torch.float32), patch_steps=patch_steps, torch=torch)
    y_test = torch.tensor(test_labels, dtype=torch.float32).unsqueeze(1)
    model = _TinyPatchTransformer(
        torch,
        patch_input_size=x_train.shape[2],
        patch_count=x_train.shape[1],
        d_model=d_model,
        layers=layers,
        heads=heads,
        dropout=dropout,
    )
    loaded_checkpoint = _load_checkpoint(model, checkpoint_in=checkpoint_in, torch=torch)
    positives = sum(train_labels)
    negatives = len(train_labels) - positives
    pos_weight = torch.tensor([negatives / positives], dtype=torch.float32) if positives else torch.tensor([1.0])
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    generator = torch.Generator().manual_seed(seed)
    first_loss = None
    last_loss = 0.0
    model.train()
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
    model.eval()
    with torch.no_grad():
        train_probabilities = torch.sigmoid(model(x_train)).squeeze(1).tolist()
        test_probabilities = torch.sigmoid(model(x_test)).squeeze(1).tolist()
        test_loss = float(criterion(model(x_test), y_test).item())
    if checkpoint_out:
        _save_checkpoint(
            model,
            checkpoint_out=checkpoint_out,
            torch=torch,
            patch_input_size=int(x_train.shape[2]),
            patch_count=int(x_train.shape[1]),
            d_model=d_model,
            layers=layers,
            heads=heads,
            dropout=dropout,
        )
    return (
        {
            "first_train_loss": round(first_loss or 0.0, 8),
            "last_train_loss": round(last_loss, 8),
            "test_loss": round(test_loss, 8),
            "loaded_checkpoint": loaded_checkpoint,
            "saved_checkpoint": str(checkpoint_out) if checkpoint_out else "",
        },
        [float(value) for value in train_probabilities],
        [float(value) for value in test_probabilities],
    )


class _TinyPatchTransformer:
    def __init__(
        self,
        torch: object,
        *,
        patch_input_size: int,
        patch_count: int,
        d_model: int,
        layers: int,
        heads: int,
        dropout: float,
    ) -> None:
        self.torch = torch
        self.projection = torch.nn.Linear(patch_input_size, d_model)
        encoder_layer = torch.nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=heads,
            dim_feedforward=d_model * 2,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = torch.nn.TransformerEncoder(encoder_layer, num_layers=layers)
        self.head = torch.nn.Linear(d_model, 1)
        self.position = torch.nn.Parameter(torch.zeros(1, patch_count, d_model))
        torch.nn.init.normal_(self.position, mean=0.0, std=0.02)

    def parameters(self):
        return [self.position, *self.projection.parameters(), *self.encoder.parameters(), *self.head.parameters()]

    def state_dict(self) -> dict[str, object]:
        return {
            "position": self.position.detach().clone(),
            "projection": self.projection.state_dict(),
            "encoder": self.encoder.state_dict(),
            "head": self.head.state_dict(),
        }

    def load_partial_state_dict(self, state: dict[str, object]) -> list[str]:
        loaded: list[str] = []
        position = state.get("position")
        if position is not None and tuple(position.shape) == tuple(self.position.shape):
            with self.torch.no_grad():
                self.position.copy_(position)
            loaded.append("position")
        for name, module in (("projection", self.projection), ("encoder", self.encoder), ("head", self.head)):
            module_state = state.get(name)
            if not isinstance(module_state, dict):
                continue
            try:
                module.load_state_dict(module_state)
            except RuntimeError:
                continue
            loaded.append(name)
        return loaded

    def train(self) -> None:
        self.projection.train()
        self.encoder.train()
        self.head.train()

    def eval(self) -> None:
        self.projection.eval()
        self.encoder.eval()
        self.head.eval()

    def __call__(self, x):
        encoded = self.encoder(self.projection(x) + self.position[:, : x.shape[1], :])
        return self.head(encoded.mean(dim=1))


def _patchify(x, *, patch_steps: int, torch: object):
    sample_count, lookback_steps, feature_count = x.shape
    patch_count = math.ceil(lookback_steps / patch_steps)
    padded_steps = patch_count * patch_steps
    if padded_steps > lookback_steps:
        padding = torch.zeros(sample_count, padded_steps - lookback_steps, feature_count, dtype=x.dtype)
        x = torch.cat([x, padding], dim=1)
    return x.reshape(sample_count, patch_count, patch_steps * feature_count)


def _import_torch() -> object:
    try:
        import torch
    except ImportError as error:
        raise ValueError("PyTorch is required for train-torch-patch-transformer-split-holdout") from error
    return torch


def _load_checkpoint(model: _TinyPatchTransformer, *, checkpoint_in: Path | None, torch: object) -> dict[str, object]:
    if not checkpoint_in:
        return {"path": "", "loaded_components": []}
    if not checkpoint_in.exists():
        return {"path": str(checkpoint_in), "loaded_components": [], "status": "missing"}
    checkpoint = torch.load(checkpoint_in, map_location="cpu")
    state = checkpoint.get("model_state", checkpoint) if isinstance(checkpoint, dict) else checkpoint
    loaded = model.load_partial_state_dict(state if isinstance(state, dict) else {})
    return {"path": str(checkpoint_in), "loaded_components": loaded, "status": "loaded" if loaded else "no_compatible_components"}


def _save_checkpoint(
    model: _TinyPatchTransformer,
    *,
    checkpoint_out: Path,
    torch: object,
    patch_input_size: int,
    patch_count: int,
    d_model: int,
    layers: int,
    heads: int,
    dropout: float,
) -> None:
    checkpoint_out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "schema": "elfquake.patch_transformer_checkpoint.v1",
            "model_state": model.state_dict(),
            "patch_input_size": patch_input_size,
            "patch_count": patch_count,
            "d_model": d_model,
            "layers": layers,
            "heads": heads,
            "dropout": dropout,
        },
        checkpoint_out,
    )


def _set_deterministic_seed(torch: object, seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True, warn_only=True)


def _write_report(out_path: Path, report: dict[str, object]) -> dict[str, object]:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report

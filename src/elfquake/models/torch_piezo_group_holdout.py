"""Leave-one-simulation-episode-out evaluation for the piezo/VLF Transformer."""

from __future__ import annotations

import csv
import json
import random
from pathlib import Path

from elfquake.models.torch_multimodal_data import (
    fit_normalizations,
    load_modality_sequences,
    modality_input_sizes,
    modality_target_sizes,
    refs_for_rows,
)
from elfquake.models.torch_multimodal_encoder import build_multimodal_patch_transformer
from elfquake.models.torch_ssl_downstream import train_downstream


PIEZO_MODALITIES = ("synthetic_piezo_vlf",)


def evaluate_piezo_group_holdout(
    *,
    target_csv: Path,
    piezo_manifest_paths: list[Path],
    out_path: Path,
    artifact_root: Path | None = None,
    seeds: list[int] | None = None,
    group_field: str = "dataset_id",
    lookback_steps: int = 12,
    patch_steps: int = 3,
    epochs: int = 12,
    learning_rate: float = 0.001,
    d_model: int = 32,
    layers: int = 2,
    heads: int = 4,
    dropout: float = 0.1,
    batch_size: int = 32,
) -> dict[str, object]:
    torch = _import_torch()
    selected_seeds = tuple(seeds or (7, 42, 99))
    sequences = load_modality_sequences(piezo_manifest_paths)
    rows = _labeled_rows(target_csv)
    groups = sorted({row.get(group_field, "") for row in rows if row.get(group_field, "")})
    if len(groups) < 2:
        raise ValueError("piezo group holdout requires at least two dataset groups")
    report: dict[str, object] = {
        "schema": "elfquake.piezo_group_holdout.v1",
        "backend": "torch",
        "device": "cpu",
        "status": "evaluated",
        "target_csv": str(target_csv),
        "piezo_sequence_manifests": [str(path) for path in piezo_manifest_paths],
        "initialization_strategy": "stable_named_parameters_v1",
        "group_field": group_field,
        "groups": groups,
        "seeds": list(selected_seeds),
        "lookback_steps": lookback_steps,
        "patch_steps": patch_steps,
        "epochs": epochs,
        "learning_rate": learning_rate,
        "d_model": d_model,
        "layers": layers,
        "heads": heads,
        "dropout": dropout,
        "batch_size": batch_size,
        "runs": [],
    }
    for seed in selected_seeds:
        for test_group in groups:
            run = _evaluate_fold(
                seed=seed,
                test_group=test_group,
                rows=rows,
                sequences=sequences,
                lookback_steps=lookback_steps,
                patch_steps=patch_steps,
                epochs=epochs,
                learning_rate=learning_rate,
                d_model=d_model,
                layers=layers,
                heads=heads,
                dropout=dropout,
                batch_size=batch_size,
                artifact_root=artifact_root,
                torch=torch,
            )
            report["runs"].append(run)
    report["summary"] = _summarize(report["runs"], groups=groups, seeds=selected_seeds)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def _evaluate_fold(
    *, seed, test_group, rows, sequences, lookback_steps, patch_steps, epochs,
    learning_rate, d_model, layers, heads, dropout, batch_size, artifact_root,
    torch,
) -> dict[str, object]:
    train_rows = [row for row in rows if row.get("dataset_id") != test_group]
    test_rows = [row for row in rows if row.get("dataset_id") == test_group]
    train_refs, train_rows = refs_for_rows(
        train_rows,
        sequences,
        modalities=PIEZO_MODALITIES,
        lookback_steps=lookback_steps,
    )
    test_refs, test_rows = refs_for_rows(
        test_rows,
        sequences,
        modalities=PIEZO_MODALITIES,
        lookback_steps=lookback_steps,
    )
    train_labels = [int(row["target_occurred"]) for row in train_rows]
    test_labels = [int(row["target_occurred"]) for row in test_rows]
    if len(set(train_labels)) < 2 or len(set(test_labels)) < 2:
        raise ValueError(f"group {test_group} requires both classes in train and test")
    train_dataset_ids = {ref.dataset_id for ref in train_refs}
    training_sequences = {
        key: sequence
        for key, sequence in sequences.items()
        if key[0] in train_dataset_ids
    }
    normalizations = fit_normalizations(training_sequences, train_fraction=1.0)
    _set_seed(torch, seed)
    model = build_multimodal_patch_transformer(
        torch,
        input_sizes=modality_input_sizes(sequences),
        target_sizes=modality_target_sizes(sequences),
        lookback_steps=lookback_steps,
        patch_steps=patch_steps,
        d_model=d_model,
        layers=layers,
        heads=heads,
        dropout=dropout,
        initialization_seed=seed,
    )
    _set_seed(torch, seed)
    downstream = train_downstream(
        model,
        train_refs=train_refs,
        train_labels=train_labels,
        test_refs=test_refs,
        test_labels=test_labels,
        sequences=sequences,
        normalizations=normalizations,
        modalities=PIEZO_MODALITIES,
        lookback_steps=lookback_steps,
        epochs=epochs,
        learning_rate=learning_rate,
        batch_size=batch_size,
        modality_dropout_probability=0.0,
        freeze_backbone=False,
        seed=seed,
        torch=torch,
    )
    checkpoint = ""
    if artifact_root:
        path = artifact_root / f"seed_{seed}" / f"holdout_{test_group}.pt"
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "schema": "elfquake.piezo_group_holdout_checkpoint.v1",
            "seed": seed,
            "test_group": test_group,
            "model_state": model.state_dict(),
        }, path)
        checkpoint = str(path)
    return {
        "seed": seed,
        "test_group": test_group,
        "train_row_count": len(train_rows),
        "test_row_count": len(test_rows),
        "train_positive_count": sum(train_labels),
        "train_negative_count": len(train_labels) - sum(train_labels),
        "test_positive_count": sum(test_labels),
        "test_negative_count": len(test_labels) - sum(test_labels),
        "downstream": downstream,
        "checkpoint": checkpoint,
    }


def _summarize(runs: list[dict[str, object]], *, groups: list[str], seeds: tuple[int, ...]) -> dict[str, object]:
    metrics = [_run_metrics(run) for run in runs]
    return {
        "run_count": len(runs),
        "balanced_accuracy": _distribution([item["balanced_accuracy"] for item in metrics]),
        "positive_recall": _distribution([item["positive_recall"] for item in metrics]),
        "negative_recall": _distribution([item["negative_recall"] for item in metrics]),
        "both_recalls_at_least_0_40_count": sum(item["positive_recall"] >= 0.4 and item["negative_recall"] >= 0.4 for item in metrics),
        "by_test_group": {
            group: _distribution([
                _run_metrics(run)["balanced_accuracy"]
                for run in runs
                if run["test_group"] == group
            ])
            for group in groups
        },
        "by_seed": {
            str(seed): _distribution([
                _run_metrics(run)["balanced_accuracy"]
                for run in runs
                if run["seed"] == seed
            ])
            for seed in seeds
        },
    }


def _run_metrics(run: dict[str, object]) -> dict[str, float]:
    return run["downstream"]["evaluations"]["trained_input"]["calibrated_metrics"]


def _distribution(values: list[float]) -> dict[str, float]:
    return {
        "mean": round(sum(values) / len(values), 8),
        "min": round(min(values), 8),
        "max": round(max(values), 8),
    }


def _labeled_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return [row for row in csv.DictReader(handle) if row.get("target_occurred") in {"0", "1"}]


def _set_seed(torch, seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True, warn_only=True)


def _import_torch():
    try:
        import torch
    except ImportError as error:
        raise ValueError("PyTorch is required for piezo group holdout") from error
    return torch

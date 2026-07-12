"""Leave-one-simulation-episode-out evaluation for the piezo/VLF Transformer."""

from __future__ import annotations

import csv
import json
import math
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
from elfquake.models.temporal_holdout import _best_threshold, _metrics, _predictions


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
    entity_aggregation_profile: str = "mean",
    exclude_piezo_fields: list[str] | None = None,
) -> dict[str, object]:
    torch = _import_torch()
    selected_seeds = tuple(seeds or (7, 42, 99))
    sequences = load_modality_sequences(
        piezo_manifest_paths,
        entity_aggregation_profile=entity_aggregation_profile,
        exclude_fields=set(exclude_piezo_fields or []),
    )
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
        "entity_aggregation_profile": entity_aggregation_profile,
        "exclude_piezo_fields": sorted(exclude_piezo_fields or []),
        "episode_diagnostics": _episode_diagnostics(
            rows,
            sequences,
            groups=groups,
            group_field=group_field,
            lookback_steps=lookback_steps,
        ),
        "runs": [],
    }
    for seed in selected_seeds:
        for test_group in groups:
            run = _evaluate_fold(
                seed=seed,
                test_group=test_group,
                group_field=group_field,
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
    _strip_probability_traces(report["runs"])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def _evaluate_fold(
    *, seed, test_group, group_field, rows, sequences, lookback_steps, patch_steps, epochs,
    learning_rate, d_model, layers, heads, dropout, batch_size, artifact_root,
    torch,
) -> dict[str, object]:
    train_rows = [row for row in rows if row.get(group_field) != test_group]
    test_rows = [row for row in rows if row.get(group_field) == test_group]
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
        include_probabilities=True,
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
    summary = {
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
    summary["fixed_seed_ensemble"] = _summarize_fixed_seed_ensemble(runs, groups)
    return summary


def _summarize_fixed_seed_ensemble(
    runs: list[dict[str, object]],
    groups: list[str],
) -> dict[str, object]:
    folds = []
    for group in groups:
        selected = [run for run in runs if run["test_group"] == group]
        downstream = [run["downstream"] for run in selected]
        train_labels = downstream[0]["train_labels"]
        test_labels = downstream[0]["test_labels"]
        train_probabilities = _mean_rows([item["train_probabilities"] for item in downstream])
        test_probabilities = _mean_rows([
            item["evaluations"]["trained_input"]["probabilities"]
            for item in downstream
        ])
        threshold = _best_threshold(train_probabilities, train_labels)
        calibrated_metrics = _metrics(_predictions(test_probabilities, threshold=threshold), test_labels)
        default_metrics = _metrics(_predictions(test_probabilities, threshold=0.5), test_labels)
        folds.append({
            "test_group": group,
            "member_seeds": [run["seed"] for run in selected],
            "calibrated_threshold": round(threshold, 6),
            "metrics": calibrated_metrics,
            "calibrated_metrics": calibrated_metrics,
            "default_metrics": default_metrics,
        })
    result = {
        "selection": "fixed_predeclared_seeds",
        "threshold_source": "ensemble_training_episode_probabilities",
        "fold_count": len(folds),
        "balanced_accuracy": _distribution([fold["metrics"]["balanced_accuracy"] for fold in folds]),
        "positive_recall": _distribution([fold["metrics"]["positive_recall"] for fold in folds]),
        "negative_recall": _distribution([fold["metrics"]["negative_recall"] for fold in folds]),
        "both_recalls_at_least_0_40_count": sum(
            fold["metrics"]["positive_recall"] >= 0.4
            and fold["metrics"]["negative_recall"] >= 0.4
            for fold in folds
        ),
        "folds": folds,
    }
    result["default_threshold"] = _ensemble_metric_summary(folds, "default_metrics")
    return result


def _ensemble_metric_summary(folds: list[dict[str, object]], field: str) -> dict[str, object]:
    return {
        "balanced_accuracy": _distribution([fold[field]["balanced_accuracy"] for fold in folds]),
        "positive_recall": _distribution([fold[field]["positive_recall"] for fold in folds]),
        "negative_recall": _distribution([fold[field]["negative_recall"] for fold in folds]),
        "both_recalls_at_least_0_40_count": sum(
            fold[field]["positive_recall"] >= 0.4 and fold[field]["negative_recall"] >= 0.4
            for fold in folds
        ),
    }


def _mean_rows(rows: list[list[float]]) -> list[float]:
    expected = len(rows[0])
    if any(len(row) != expected for row in rows):
        raise ValueError("ensemble probability rows must have equal lengths")
    return [sum(values) / len(values) for values in zip(*rows)]


def _run_metrics(run: dict[str, object]) -> dict[str, float]:
    return run["downstream"]["evaluations"]["trained_input"]["calibrated_metrics"]


def _strip_probability_traces(runs: list[dict[str, object]]) -> None:
    for run in runs:
        downstream = run["downstream"]
        downstream.pop("train_probabilities", None)
        downstream.pop("train_labels", None)
        downstream.pop("test_labels", None)
        for evaluation in downstream["evaluations"].values():
            evaluation.pop("probabilities", None)


def _episode_diagnostics(
    rows: list[dict[str, str]],
    sequences,
    *,
    groups: list[str],
    group_field: str,
    lookback_steps: int,
) -> dict[str, object]:
    episodes = {}
    effects_by_feature: dict[str, list[float]] = {}
    for group in groups:
        group_rows = [row for row in rows if row.get(group_field) == group]
        refs, covered = refs_for_rows(
            group_rows,
            sequences,
            modalities=PIEZO_MODALITIES,
            lookback_steps=lookback_steps,
        )
        labels = [int(row["target_occurred"]) for row in covered]
        sequence = sequences[(refs[0].dataset_id, PIEZO_MODALITIES[0])]
        feature_rows = [
            _window_feature_means(sequence, ref.end_index, lookback_steps)
            for ref in refs
        ]
        effects = {}
        for index, feature in enumerate(sequence.feature_names):
            positive = [values[index] for values, label in zip(feature_rows, labels) if label == 1]
            negative = [values[index] for values, label in zip(feature_rows, labels) if label == 0]
            effect = _standardized_difference(positive, negative)
            effects[feature] = effect
            effects_by_feature.setdefault(feature, []).append(effect)
        episodes[group] = {
            "row_count": len(labels),
            "positive_count": sum(labels),
            "negative_count": len(labels) - sum(labels),
            "positive_rate": round(sum(labels) / len(labels), 8),
            "label_transition_count": sum(left != right for left, right in zip(labels, labels[1:])),
            "longest_positive_run": _longest_run(labels, 1),
            "longest_negative_run": _longest_run(labels, 0),
            "feature_standardized_mean_differences": effects,
        }
    return {
        "effect_definition": "(positive_window_mean-negative_window_mean)/within_episode_std",
        "episodes": episodes,
        "feature_consistency": {
            feature: {
                "positive_episode_count": sum(value > 0 for value in effects),
                "negative_episode_count": sum(value < 0 for value in effects),
                "mean": round(sum(effects) / len(effects), 8),
                "min": min(effects),
                "max": max(effects),
            }
            for feature, effects in effects_by_feature.items()
        },
    }


def _window_feature_means(sequence, end_index: int, lookback_steps: int) -> list[float]:
    rows = sequence.values[end_index - lookback_steps:end_index]
    return [sum(values) / len(values) for values in zip(*rows)]


def _standardized_difference(positive: list[float], negative: list[float]) -> float:
    combined = positive + negative
    mean = sum(combined) / len(combined)
    variance = sum((value - mean) ** 2 for value in combined) / len(combined)
    scale = math.sqrt(variance)
    if scale < 1e-12:
        return 0.0
    difference = sum(positive) / len(positive) - sum(negative) / len(negative)
    return round(difference / scale, 8)


def _longest_run(labels: list[int], target: int) -> int:
    longest = current = 0
    for label in labels:
        current = current + 1 if label == target else 0
        longest = max(longest, current)
    return longest


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

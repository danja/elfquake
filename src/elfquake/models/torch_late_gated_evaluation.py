"""Paired evaluation of piezo/VLF anchors and late gated fusion."""

from __future__ import annotations

import csv
import json
import math
import random
from pathlib import Path

from elfquake.models.torch_late_gated_fusion import (
    AUXILIARY_MODALITIES,
    build_late_gated_fusion_classifier,
)
from elfquake.models.torch_multimodal_data import (
    build_window_batch,
    chronological_window_refs,
    fit_normalizations,
    load_modality_sequences,
    modality_input_sizes,
    modality_target_sizes,
    refs_for_rows,
)
from elfquake.models.torch_multimodal_encoder import (
    build_multimodal_patch_transformer,
    clone_state,
    load_compatible_state,
)
from elfquake.models.torch_ssl_downstream import SYNTHETIC_MODALITIES, train_downstream
from elfquake.models.torch_ssl_pretrain import PretrainTask, pretrain_masked_patches


INITIALIZATIONS = ("random_init", "synthetic_pretrain")
MODEL_CONFIGS = (
    "piezo_vlf_anchor",
    "late_gated_fusion",
    "anchored_late_gated_fusion",
    "anchored_direct_gated_fusion",
)


def evaluate_late_gated_fusion(
    *,
    target_csv: Path,
    synthetic_manifest_paths: list[Path],
    out_path: Path,
    artifact_root: Path | None = None,
    seeds: list[int] | None = None,
    split_field: str = "model_split",
    train_value: str = "train",
    test_value: str = "test",
    lookback_steps: int = 12,
    patch_steps: int = 3,
    train_fraction: float = 0.8,
    pretrain_stride: int = 3,
    ssl_epochs: int = 6,
    supervised_epochs: int = 12,
    learning_rate: float = 0.001,
    d_model: int = 32,
    layers: int = 2,
    heads: int = 4,
    dropout: float = 0.1,
    batch_size: int = 32,
    mask_probability: float = 0.30,
    modality_dropout_probability: float = 0.25,
    max_pretrain_windows: int = 2048,
) -> dict[str, object]:
    torch = _import_torch()
    selected_seeds = tuple(seeds or (7, 42, 99))
    sequences = load_modality_sequences(synthetic_manifest_paths)
    normalizations = fit_normalizations(sequences, train_fraction=train_fraction)
    synthetic_task = _synthetic_pretrain_task(
        sequences,
        lookback_steps=lookback_steps,
        train_fraction=train_fraction,
        stride=pretrain_stride,
    )
    train_rows, test_rows = _split_rows(
        target_csv,
        split_field=split_field,
        train_value=train_value,
        test_value=test_value,
    )
    train_refs, train_rows = refs_for_rows(
        train_rows,
        sequences,
        modalities=SYNTHETIC_MODALITIES,
        lookback_steps=lookback_steps,
    )
    test_refs, test_rows = refs_for_rows(
        test_rows,
        sequences,
        modalities=SYNTHETIC_MODALITIES,
        lookback_steps=lookback_steps,
    )
    train_labels = [int(row["target_occurred"]) for row in train_rows]
    test_labels = [int(row["target_occurred"]) for row in test_rows]
    if not train_labels or not test_labels or len(set(train_labels)) < 2:
        raise ValueError("late-fusion target split requires train class variation and held-out rows")

    report: dict[str, object] = {
        "schema": "elfquake.late_gated_fusion_evaluation.v1",
        "backend": "torch",
        "device": "cpu",
        "status": "evaluated",
        "target_csv": str(target_csv),
        "synthetic_sequence_manifests": [str(path) for path in synthetic_manifest_paths],
        "initializations": list(INITIALIZATIONS),
        "model_configs": list(MODEL_CONFIGS),
        "seeds": list(selected_seeds),
        "lookback_steps": lookback_steps,
        "patch_steps": patch_steps,
        "train_fraction": train_fraction,
        "pretrain_stride": pretrain_stride,
        "ssl_epochs": ssl_epochs,
        "supervised_epochs": supervised_epochs,
        "learning_rate": learning_rate,
        "d_model": d_model,
        "layers": layers,
        "heads": heads,
        "dropout": dropout,
        "batch_size": batch_size,
        "mask_probability": mask_probability,
        "modality_dropout_probability": modality_dropout_probability,
        "max_pretrain_windows": max_pretrain_windows,
        "initialization_strategy": "stable_named_parameters_v1",
        "pretrain_train_windows": len(synthetic_task.train_refs),
        "pretrain_test_windows": len(synthetic_task.test_refs),
        "downstream_train_rows": len(train_rows),
        "downstream_test_rows": len(test_rows),
        "runs": [],
    }
    for seed in selected_seeds:
        report["runs"].extend(_evaluate_seed(
            seed=seed,
            sequences=sequences,
            normalizations=normalizations,
            synthetic_task=synthetic_task,
            train_refs=train_refs,
            train_labels=train_labels,
            test_refs=test_refs,
            test_labels=test_labels,
            lookback_steps=lookback_steps,
            patch_steps=patch_steps,
            ssl_epochs=ssl_epochs,
            supervised_epochs=supervised_epochs,
            learning_rate=learning_rate,
            d_model=d_model,
            layers=layers,
            heads=heads,
            dropout=dropout,
            batch_size=batch_size,
            mask_probability=mask_probability,
            modality_dropout_probability=modality_dropout_probability,
            max_pretrain_windows=max_pretrain_windows,
            artifact_root=artifact_root,
            torch=torch,
        ))
    report["summary"] = _summarize(report["runs"])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def _evaluate_seed(
    *, seed, sequences, normalizations, synthetic_task, train_refs, train_labels,
    test_refs, test_labels, lookback_steps, patch_steps, ssl_epochs,
    supervised_epochs, learning_rate, d_model, layers, heads, dropout,
    batch_size, mask_probability, modality_dropout_probability,
    max_pretrain_windows, artifact_root, torch,
) -> list[dict[str, object]]:
    _set_seed(torch, seed)
    backbone = _build_backbone(
        torch,
        sequences=sequences,
        lookback_steps=lookback_steps,
        patch_steps=patch_steps,
        d_model=d_model,
        layers=layers,
        heads=heads,
        dropout=dropout,
        initialization_seed=seed,
    )
    random_state = clone_state(backbone)
    pretraining = pretrain_masked_patches(
        backbone,
        tasks=[synthetic_task],
        sequences=sequences,
        normalizations=normalizations,
        lookback_steps=lookback_steps,
        patch_steps=patch_steps,
        epochs=ssl_epochs,
        learning_rate=learning_rate,
        batch_size=batch_size,
        mask_probability=mask_probability,
        modality_dropout_probability=modality_dropout_probability,
        max_windows_per_domain=max_pretrain_windows,
        balance_domains=False,
        seed=seed,
        torch=torch,
    )
    pretrained_state = clone_state(backbone)
    runs = []
    for initialization, state in (("random_init", random_state), ("synthetic_pretrain", pretrained_state)):
        anchor_finetuned_state = None
        for config_name in MODEL_CONFIGS:
            _set_seed(torch, seed)
            model_backbone = _build_backbone(
                torch,
                sequences=sequences,
                lookback_steps=lookback_steps,
                patch_steps=patch_steps,
                d_model=d_model,
                layers=layers,
                heads=heads,
                dropout=dropout,
                initialization_seed=seed,
            )
            load_compatible_state(
                model_backbone,
                anchor_finetuned_state if config_name.startswith("anchored_") else state,
            )
            model, modalities, training_dropout, optimizer_parameters, trainable_scope = _downstream_model(
                config_name,
                backbone=model_backbone,
                d_model=d_model,
                modality_dropout_probability=modality_dropout_probability,
                torch=torch,
            )
            _set_seed(torch, seed)
            result = train_downstream(
                model,
                train_refs=train_refs,
                train_labels=train_labels,
                test_refs=test_refs,
                test_labels=test_labels,
                sequences=sequences,
                normalizations=normalizations,
                modalities=modalities,
                lookback_steps=lookback_steps,
                epochs=supervised_epochs,
                learning_rate=learning_rate,
                batch_size=batch_size,
                modality_dropout_probability=training_dropout,
                freeze_backbone=False,
                optimizer_parameters=optimizer_parameters,
                trainable_scope=trainable_scope,
                seed=seed,
                torch=torch,
            )
            if config_name == "piezo_vlf_anchor":
                anchor_finetuned_state = clone_state(model)
            checkpoint = _save_checkpoint(
                model,
                artifact_root=artifact_root,
                initialization=initialization,
                config_name=config_name,
                seed=seed,
                torch=torch,
            )
            runs.append({
                "seed": seed,
                "initialization": initialization,
                "model_config": config_name,
                "pretraining": pretraining if initialization == "synthetic_pretrain" else {},
                "downstream": result,
                "gate_statistics": _gate_statistics(
                    model,
                    refs=test_refs,
                    sequences=sequences,
                    normalizations=normalizations,
                    lookback_steps=lookback_steps,
                    batch_size=batch_size,
                    torch=torch,
                ) if config_name != "piezo_vlf_anchor" else {},
                "checkpoint": checkpoint,
            })
    return runs


def _build_backbone(torch, *, sequences, lookback_steps, patch_steps, d_model, layers, heads, dropout, initialization_seed):
    return build_multimodal_patch_transformer(
        torch,
        input_sizes=modality_input_sizes(sequences),
        target_sizes=modality_target_sizes(sequences),
        lookback_steps=lookback_steps,
        patch_steps=patch_steps,
        d_model=d_model,
        layers=layers,
        heads=heads,
        dropout=dropout,
        initialization_seed=initialization_seed,
    )


def _downstream_model(config_name, *, backbone, d_model, modality_dropout_probability, torch):
    if config_name == "piezo_vlf_anchor":
        return backbone, ("synthetic_piezo_vlf",), 0.0, None, "all"
    if config_name == "late_gated_fusion":
        return (
            build_late_gated_fusion_classifier(torch, backbone=backbone, d_model=d_model),
            SYNTHETIC_MODALITIES,
            modality_dropout_probability,
            None,
            "all",
        )
    if config_name == "anchored_late_gated_fusion":
        model = build_late_gated_fusion_classifier(
            torch,
            backbone=backbone,
            d_model=d_model,
            closed_gate_bias=-2.0,
            initialize_from_anchor_head=True,
            freeze_backbone=True,
        )
        return model, SYNTHETIC_MODALITIES, modality_dropout_probability, model.fusion_parameters(), "fusion_only"
    if config_name == "anchored_direct_gated_fusion":
        model = build_late_gated_fusion_classifier(
            torch,
            backbone=backbone,
            d_model=d_model,
            auxiliary_modalities=("synthetic_direct_avalanche",),
            closed_gate_bias=-2.0,
            initialize_from_anchor_head=True,
            freeze_backbone=True,
        )
        modalities = ("synthetic_piezo_vlf", "synthetic_direct_avalanche")
        return model, modalities, modality_dropout_probability, model.fusion_parameters(), "fusion_only"
    raise ValueError(f"unknown late-fusion model config: {config_name}")


def _gate_statistics(model, *, refs, sequences, normalizations, lookback_steps, batch_size, torch):
    totals = {modality: [0.0, 0.0, 0] for modality in model.auxiliary_modalities}
    model.eval()
    with torch.no_grad():
        for start in range(0, len(refs), batch_size):
            batch_refs = refs[start:start + batch_size]
            inputs, _, observed = build_window_batch(
                batch_refs,
                sequences,
                modalities=("synthetic_piezo_vlf", *model.auxiliary_modalities),
                lookback_steps=lookback_steps,
                normalizations=normalizations,
                torch=torch,
            )
            _, gates = model.embedding_with_gates(inputs, observed)
            for modality, values in gates.items():
                totals[modality][0] += float(values.sum().item())
                totals[modality][1] += float((values ** 2).sum().item())
                totals[modality][2] += int(values.numel())
    result = {}
    for modality, (total, squared, count) in totals.items():
        mean = total / max(1, count)
        variance = max(0.0, squared / max(1, count) - mean * mean)
        result[modality] = {"mean": round(mean, 8), "std": round(math.sqrt(variance), 8)}
    return result


def _summarize(runs: list[dict[str, object]]) -> dict[str, object]:
    summary = {}
    for initialization in INITIALIZATIONS:
        summary[initialization] = {}
        for config_name in MODEL_CONFIGS:
            selected = [run for run in runs if run["initialization"] == initialization and run["model_config"] == config_name]
            evaluation_name = "full" if config_name in {"late_gated_fusion", "anchored_late_gated_fusion"} else "trained_input"
            values = [float(run["downstream"]["evaluations"][evaluation_name]["calibrated_metrics"]["balanced_accuracy"]) for run in selected]
            row = {"balanced_accuracy": _distribution(values)}
            if config_name != "piezo_vlf_anchor":
                row["test_time_branch_evaluations"] = {
                    name: _distribution([
                        float(run["downstream"]["evaluations"][name]["calibrated_metrics"]["balanced_accuracy"])
                        for run in selected
                    ])
                    for name in selected[0]["downstream"]["evaluations"]
                }
                row["gate_means"] = {
                    modality: round(sum(float(run["gate_statistics"][modality]["mean"]) for run in selected) / len(selected), 8)
                    for modality in selected[0]["gate_statistics"]
                }
            summary[initialization][config_name] = row
    random = summary["random_init"]
    for initialization in INITIALIZATIONS:
        for config_name in MODEL_CONFIGS:
            current = summary[initialization][config_name]
            current["mean_gain_over_matching_random"] = round(
                float(current["balanced_accuracy"]["mean"]) - float(random[config_name]["balanced_accuracy"]["mean"]),
                8,
            )
    return summary


def _synthetic_pretrain_task(sequences, *, lookback_steps, train_fraction, stride):
    train, test = chronological_window_refs(
        sequences,
        modality="synthetic_direct_avalanche",
        lookback_steps=lookback_steps,
        train_fraction=train_fraction,
        stride=stride,
    )
    covered = set.intersection(*(
        {dataset_id for dataset_id, item_modality in sequences if item_modality == modality}
        for modality in SYNTHETIC_MODALITIES
    ))
    return PretrainTask(
        name="synthetic",
        modalities=SYNTHETIC_MODALITIES,
        train_refs=tuple(ref for ref in train if ref.dataset_id in covered),
        test_refs=tuple(ref for ref in test if ref.dataset_id in covered),
    )


def _split_rows(path: Path, *, split_field: str, train_value: str, test_value: str):
    with path.open(newline="", encoding="utf-8") as handle:
        rows = [row for row in csv.DictReader(handle) if row.get("target_occurred") in {"0", "1"}]
    return (
        [row for row in rows if row.get(split_field) == train_value],
        [row for row in rows if row.get(split_field) == test_value],
    )


def _save_checkpoint(model, *, artifact_root, initialization, config_name, seed, torch):
    if not artifact_root:
        return ""
    path = artifact_root / initialization / f"seed_{seed}" / f"{config_name}.pt"
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "schema": "elfquake.late_gated_fusion_checkpoint.v1",
        "initialization": initialization,
        "model_config": config_name,
        "seed": seed,
        "model_state": model.state_dict(),
    }, path)
    return str(path)


def _distribution(values: list[float]) -> dict[str, float]:
    return {
        "mean": round(sum(values) / len(values), 8),
        "min": round(min(values), 8),
        "max": round(max(values), 8),
    }


def _set_seed(torch, seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True, warn_only=True)


def _import_torch():
    try:
        import torch
    except ImportError as error:
        raise ValueError("PyTorch is required for late gated fusion evaluation") from error
    return torch

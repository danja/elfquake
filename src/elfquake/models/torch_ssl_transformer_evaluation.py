"""Evaluate self-supervised initialization of the multimodal patch Transformer."""

from __future__ import annotations

import csv
import json
import random
from pathlib import Path

from elfquake.models.torch_multimodal_data import (
    VLF_SIGNAL_EXCLUDES,
    chronological_window_refs,
    fit_normalizations,
    load_modality_sequences,
    modality_input_sizes,
    modality_target_sizes,
    refs_for_rows,
)
from elfquake.models.torch_ssl_downstream import (
    SYNTHETIC_MODALITIES,
    summarize_downstream_runs,
    train_downstream,
)
from elfquake.models.torch_multimodal_encoder import (
    build_multimodal_patch_transformer,
    clone_state,
    load_compatible_state,
)
from elfquake.models.torch_ssl_pretrain import (
    PretrainTask,
    evaluate_masked_reconstruction,
    pretrain_masked_patches,
)


REGIMES = (
    "random_init",
    "synthetic_pretrain",
    "real_vlf_pretrain",
    "synthetic_then_real",
    "joint_synthetic_real",
)


def evaluate_self_supervised_transformer(
    *,
    target_csv: Path,
    synthetic_manifest_paths: list[Path],
    real_manifest_path: Path,
    out_path: Path,
    artifact_root: Path | None = None,
    regimes: list[str] | None = None,
    seeds: list[int] | None = None,
    split_field: str = "model_split",
    train_value: str = "train",
    test_value: str = "test",
    lookback_steps: int = 12,
    patch_steps: int = 3,
    train_fraction: float = 0.8,
    pretrain_stride: int = 3,
    ssl_epochs: int = 8,
    supervised_epochs: int = 12,
    learning_rate: float = 0.001,
    d_model: int = 32,
    layers: int = 2,
    heads: int = 4,
    dropout: float = 0.1,
    batch_size: int = 32,
    mask_probability: float = 0.30,
    modality_dropout_probability: float = 0.25,
    max_pretrain_windows: int = 4096,
) -> dict[str, object]:
    selected_regimes = tuple(regimes or REGIMES)
    unknown = sorted(set(selected_regimes) - set(REGIMES))
    if unknown:
        raise ValueError(f"unknown self-supervised regime(s): {', '.join(unknown)}")
    selected_seeds = tuple(seeds or (7, 42, 99))
    torch = _import_torch()
    manifest_paths = [*synthetic_manifest_paths, real_manifest_path]
    sequences = load_modality_sequences(manifest_paths)
    normalizations = fit_normalizations(sequences, train_fraction=train_fraction)
    real_modality = _real_modality(sequences)
    synthetic_task = _pretrain_task(
        "synthetic",
        sequences=sequences,
        modalities=SYNTHETIC_MODALITIES,
        anchor_modality="synthetic_direct_avalanche",
        lookback_steps=lookback_steps,
        train_fraction=train_fraction,
        stride=pretrain_stride,
    )
    real_task = _pretrain_task(
        "real_vlf",
        sequences=sequences,
        modalities=(real_modality,),
        anchor_modality=real_modality,
        lookback_steps=lookback_steps,
        train_fraction=train_fraction,
        stride=1,
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
        raise ValueError("downstream target split requires train class variation and held-out rows")

    report: dict[str, object] = {
        "schema": "elfquake.self_supervised_transformer_evaluation.v1",
        "backend": "torch",
        "device": "cpu",
        "status": "evaluated",
        "target_csv": str(target_csv),
        "synthetic_sequence_manifests": [str(path) for path in synthetic_manifest_paths],
        "real_sequence_manifest": str(real_manifest_path),
        "regimes": list(selected_regimes),
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
        "excluded_real_vlf_reconstruction_fields": sorted(VLF_SIGNAL_EXCLUDES),
        "real_vlf_feature_names": list(next(item.feature_names for item in sequences.values() if item.modality == real_modality)),
        "synthetic_pretrain_train_windows": len(synthetic_task.train_refs),
        "synthetic_pretrain_test_windows": len(synthetic_task.test_refs),
        "real_pretrain_train_windows": len(real_task.train_refs),
        "real_pretrain_test_windows": len(real_task.test_refs),
        "downstream_train_rows": len(train_rows),
        "downstream_test_rows": len(test_rows),
        "downstream_train_positive_count": sum(train_labels),
        "downstream_train_negative_count": len(train_labels) - sum(train_labels),
        "downstream_test_positive_count": sum(test_labels),
        "downstream_test_negative_count": len(test_labels) - sum(test_labels),
        "runs": [],
    }
    for seed in selected_seeds:
        for regime in selected_regimes:
            run = _evaluate_run(
                regime=regime,
                seed=seed,
                sequences=sequences,
                normalizations=normalizations,
                synthetic_task=synthetic_task,
                real_task=real_task,
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
            )
            report["runs"].append(run)
    report["summary"] = summarize_downstream_runs(report["runs"], REGIMES)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def _evaluate_run(
    *,
    regime: str,
    seed: int,
    sequences: dict,
    normalizations: dict,
    synthetic_task: PretrainTask,
    real_task: PretrainTask,
    train_refs: list,
    train_labels: list[int],
    test_refs: list,
    test_labels: list[int],
    lookback_steps: int,
    patch_steps: int,
    ssl_epochs: int,
    supervised_epochs: int,
    learning_rate: float,
    d_model: int,
    layers: int,
    heads: int,
    dropout: float,
    batch_size: int,
    mask_probability: float,
    modality_dropout_probability: float,
    max_pretrain_windows: int,
    artifact_root: Path | None,
    torch: object,
) -> dict[str, object]:
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
    )
    stages = []
    if regime == "synthetic_pretrain":
        stages.append(_pretrain_stage(model, [synthetic_task], False, seed, sequences, normalizations, lookback_steps, patch_steps, ssl_epochs, learning_rate, batch_size, mask_probability, modality_dropout_probability, max_pretrain_windows, torch))
    elif regime == "real_vlf_pretrain":
        stages.append(_pretrain_stage(model, [real_task], False, seed, sequences, normalizations, lookback_steps, patch_steps, ssl_epochs, learning_rate, batch_size, mask_probability, modality_dropout_probability, max_pretrain_windows, torch))
    elif regime == "synthetic_then_real":
        stages.append(_pretrain_stage(model, [synthetic_task], False, seed, sequences, normalizations, lookback_steps, patch_steps, ssl_epochs, learning_rate, batch_size, mask_probability, modality_dropout_probability, max_pretrain_windows, torch))
        stages.append(_pretrain_stage(model, [real_task], False, seed + 1, sequences, normalizations, lookback_steps, patch_steps, ssl_epochs, learning_rate, batch_size, mask_probability, modality_dropout_probability, max_pretrain_windows, torch))
    elif regime == "joint_synthetic_real":
        stages.append(_pretrain_stage(model, [synthetic_task, real_task], True, seed, sequences, normalizations, lookback_steps, patch_steps, ssl_epochs, learning_rate, batch_size, mask_probability, modality_dropout_probability, max_pretrain_windows, torch))

    final_reconstruction = {
        task.name: evaluate_masked_reconstruction(
            model,
            refs=list(task.test_refs),
            sequences=sequences,
            normalizations=normalizations,
            modalities=task.modalities,
            lookback_steps=lookback_steps,
            patch_steps=patch_steps,
            batch_size=batch_size,
            mask_probability=mask_probability,
            seed=seed + 20_000,
            torch=torch,
        )
        for task in (synthetic_task, real_task)
    }
    pretrained_state = clone_state(model)
    linear_probe = train_downstream(
        model,
        train_refs=train_refs,
        train_labels=train_labels,
        test_refs=test_refs,
        test_labels=test_labels,
        sequences=sequences,
        normalizations=normalizations,
        lookback_steps=lookback_steps,
        epochs=supervised_epochs,
        learning_rate=learning_rate,
        batch_size=batch_size,
        modality_dropout_probability=0.0,
        freeze_backbone=True,
        seed=seed,
        torch=torch,
    )
    load_compatible_state(model, pretrained_state)
    fine_tune = train_downstream(
        model,
        train_refs=train_refs,
        train_labels=train_labels,
        test_refs=test_refs,
        test_labels=test_labels,
        sequences=sequences,
        normalizations=normalizations,
        lookback_steps=lookback_steps,
        epochs=supervised_epochs,
        learning_rate=learning_rate,
        batch_size=batch_size,
        modality_dropout_probability=modality_dropout_probability,
        freeze_backbone=False,
        seed=seed,
        torch=torch,
    )
    checkpoint = ""
    if artifact_root:
        path = artifact_root / regime / f"seed_{seed}.pt"
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"schema": "elfquake.multimodal_patch_transformer.v1", "regime": regime, "seed": seed, "model_state": model.state_dict()}, path)
        checkpoint = str(path)
    return {
        "regime": regime,
        "seed": seed,
        "pretraining_stages": stages,
        "final_reconstruction": final_reconstruction,
        "linear_probe": linear_probe,
        "fine_tune": fine_tune,
        "checkpoint": checkpoint,
    }


def _pretrain_stage(model, tasks, balance, seed, sequences, normalizations, lookback, patch, epochs, learning_rate, batch_size, mask_probability, dropout_probability, max_windows, torch):
    result = pretrain_masked_patches(
        model,
        tasks=tasks,
        sequences=sequences,
        normalizations=normalizations,
        lookback_steps=lookback,
        patch_steps=patch,
        epochs=epochs,
        learning_rate=learning_rate,
        batch_size=batch_size,
        mask_probability=mask_probability,
        modality_dropout_probability=dropout_probability,
        max_windows_per_domain=max_windows,
        balance_domains=balance,
        seed=seed,
        torch=torch,
    )
    return {"tasks": [task.name for task in tasks], "balanced_domains": balance, **result}


def _pretrain_task(name, *, sequences, modalities, anchor_modality, lookback_steps, train_fraction, stride):
    train, test = chronological_window_refs(sequences, modality=anchor_modality, lookback_steps=lookback_steps, train_fraction=train_fraction, stride=stride)
    covered_ids = {dataset_id for dataset_id, modality in sequences if modality == anchor_modality}
    for modality in modalities:
        covered_ids &= {dataset_id for dataset_id, item_modality in sequences if item_modality == modality}
    return PretrainTask(
        name=name,
        modalities=modalities,
        train_refs=tuple(ref for ref in train if ref.dataset_id in covered_ids),
        test_refs=tuple(ref for ref in test if ref.dataset_id in covered_ids),
    )


def _split_rows(path: Path, *, split_field: str, train_value: str, test_value: str):
    with path.open(newline="", encoding="utf-8") as handle:
        rows = [row for row in csv.DictReader(handle) if row.get("target_occurred") in {"0", "1"}]
    return (
        [row for row in rows if row.get(split_field) == train_value],
        [row for row in rows if row.get(split_field) == test_value],
    )


def _real_modality(sequences: dict) -> str:
    modalities = sorted({sequence.modality for sequence in sequences.values() if sequence.modality.startswith("real_")})
    if len(modalities) != 1:
        raise ValueError(f"expected one real modality, found: {', '.join(modalities)}")
    return modalities[0]


def _set_seed(torch: object, seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True, warn_only=True)


def _import_torch() -> object:
    try:
        import torch
    except ImportError as error:
        raise ValueError("PyTorch is required for self-supervised Transformer evaluation") from error
    return torch

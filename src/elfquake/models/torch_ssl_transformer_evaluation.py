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
    DOWNSTREAM_CONFIGS,
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
    "synthetic_then_real_frozen",
    "synthetic_then_real_rehearsal",
    "joint_synthetic_real",
    "synthetic_then_japan_then_italy",
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
    japan_manifest_paths: list[Path] | None = None,
    italy_manifest_paths: list[Path] | None = None,
    japan_modality: str = "japan_vlf",
    italy_modalities: tuple[str, ...] | None = None,
    target_dataset_id: str | None = None,
) -> dict[str, object]:
    requested_regimes = tuple(regimes or REGIMES)
    skipped_regimes: tuple[str, ...] = ()
    if "synthetic_then_japan_then_italy" in requested_regimes:
        missing_cross_region = not japan_manifest_paths or not italy_manifest_paths
        if missing_cross_region:
            if regimes is not None:
                raise ValueError(
                    "synthetic_then_japan_then_italy requires Japan and Italy sequence manifests"
                )
            skipped_regimes = ("synthetic_then_japan_then_italy",)
            selected_regimes = tuple(
                regime for regime in requested_regimes if regime not in skipped_regimes
            )
        else:
            selected_regimes = requested_regimes
    else:
        selected_regimes = requested_regimes
    unknown = sorted(set(selected_regimes) - set(REGIMES))
    if unknown:
        raise ValueError(f"unknown self-supervised regime(s): {', '.join(unknown)}")
    selected_seeds = tuple(seeds or (7, 42, 99))
    torch = _import_torch()
    manifest_paths = [*synthetic_manifest_paths, real_manifest_path, *(japan_manifest_paths or []), *(italy_manifest_paths or [])]
    sequences = load_modality_sequences(manifest_paths)
    normalizations = fit_normalizations(sequences, train_fraction=train_fraction)
    real_modality = _real_modality(sequences)
    cross_region = "synthetic_then_japan_then_italy" in selected_regimes
    downstream_configs = (
        {"italy_multimodal": tuple(italy_modalities or ("seismic", "italy_vlf", "astronomy"))}
        if cross_region else DOWNSTREAM_CONFIGS
    )
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
    japan_task = None
    italy_task = None
    if cross_region:
        japan_task = _pretrain_task(
            "japan_vlf",
            sequences=sequences,
            modalities=(japan_modality,),
            anchor_modality=japan_modality,
            lookback_steps=lookback_steps,
            train_fraction=train_fraction,
            stride=1,
        )
        italy_task = _pretrain_task(
            "italy_multimodal",
            sequences=sequences,
            modalities=tuple(italy_modalities or ("seismic", "italy_vlf", "astronomy")),
            anchor_modality="italy_vlf",
            lookback_steps=lookback_steps,
            train_fraction=train_fraction,
            stride=1,
        )
    train_rows, test_rows = _split_rows(
        target_csv,
        split_field=split_field,
        train_value=train_value,
        test_value=test_value,
        dataset_id=target_dataset_id,
    )
    downstream_modalities = tuple(italy_modalities or ("seismic", "italy_vlf", "astronomy")) if cross_region else SYNTHETIC_MODALITIES
    train_refs, train_rows = refs_for_rows(
        train_rows,
        sequences,
        modalities=downstream_modalities,
        lookback_steps=lookback_steps,
    )
    test_refs, test_rows = refs_for_rows(
        test_rows,
        sequences,
        modalities=downstream_modalities,
        lookback_steps=lookback_steps,
    )
    train_labels = [int(row["target_occurred"]) for row in train_rows]
    test_labels = [int(row["target_occurred"]) for row in test_rows]
    train_coordinate_targets = [_coordinate_targets(row) for row in train_rows]
    test_coordinate_targets = [_coordinate_targets(row) for row in test_rows]
    coordinate_slots = max(1, max((len(value) for value in train_coordinate_targets), default=1)) if cross_region else 1
    if not train_labels or not test_labels or len(set(train_labels)) < 2:
        raise ValueError("downstream target split requires train class variation and held-out rows")

    report: dict[str, object] = {
        "schema": "elfquake.self_supervised_transformer_evaluation.v2",
        "backend": "torch",
        "device": "cpu",
        "status": "evaluated",
        "target_csv": str(target_csv),
        "synthetic_sequence_manifests": [str(path) for path in synthetic_manifest_paths],
        "real_sequence_manifest": str(real_manifest_path),
        "regimes": list(selected_regimes),
        "skipped_regimes": list(skipped_regimes),
        "downstream_configs": {name: list(modalities) for name, modalities in downstream_configs.items()},
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
        "coordinate_slots": coordinate_slots,
        "initialization_strategy": "stable_named_parameters_v1",
        "excluded_real_vlf_reconstruction_fields": sorted(VLF_SIGNAL_EXCLUDES),
        "real_vlf_feature_names": list(next(item.feature_names for item in sequences.values() if item.modality == real_modality)),
        "synthetic_pretrain_train_windows": len(synthetic_task.train_refs),
        "synthetic_pretrain_test_windows": len(synthetic_task.test_refs),
        "real_pretrain_train_windows": len(real_task.train_refs),
        "real_pretrain_test_windows": len(real_task.test_refs),
        "japan_pretrain_train_windows": len(japan_task.train_refs) if japan_task else 0,
        "japan_pretrain_test_windows": len(japan_task.test_refs) if japan_task else 0,
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
                japan_task=japan_task,
                italy_task=italy_task,
                downstream_configs=downstream_configs,
                train_refs=train_refs,
                train_labels=train_labels,
                test_refs=test_refs,
                test_labels=test_labels,
                train_coordinate_targets=train_coordinate_targets,
                test_coordinate_targets=test_coordinate_targets,
                coordinate_slots=coordinate_slots,
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
    report["summary"] = summarize_downstream_runs(report["runs"], selected_regimes)
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
    japan_task: PretrainTask | None,
    italy_task: PretrainTask | None,
    downstream_configs: dict[str, tuple[str, ...]],
    train_refs: list,
    train_labels: list[int],
    test_refs: list,
    test_labels: list[int],
    train_coordinate_targets: list[tuple[float, float, float] | None],
    test_coordinate_targets: list[tuple[float, float, float] | None],
    coordinate_slots: int,
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
        coordinate_slots=coordinate_slots,
        initialization_seed=seed,
    )
    stages = []
    if regime == "synthetic_pretrain":
        stages.append(_pretrain_stage(model, [synthetic_task], False, seed, sequences, normalizations, lookback_steps, patch_steps, ssl_epochs, learning_rate, batch_size, mask_probability, modality_dropout_probability, max_pretrain_windows, torch))
    elif regime == "real_vlf_pretrain":
        stages.append(_pretrain_stage(model, [real_task], False, seed, sequences, normalizations, lookback_steps, patch_steps, ssl_epochs, learning_rate, batch_size, mask_probability, modality_dropout_probability, max_pretrain_windows, torch))
    elif regime == "synthetic_then_real":
        stages.append(_pretrain_stage(model, [synthetic_task], False, seed, sequences, normalizations, lookback_steps, patch_steps, ssl_epochs, learning_rate, batch_size, mask_probability, modality_dropout_probability, max_pretrain_windows, torch))
        stages.append(_pretrain_stage(model, [real_task], False, seed + 1, sequences, normalizations, lookback_steps, patch_steps, ssl_epochs, learning_rate, batch_size, mask_probability, modality_dropout_probability, max_pretrain_windows, torch))
    elif regime == "synthetic_then_real_frozen":
        stages.append(_pretrain_stage(model, [synthetic_task], False, seed, sequences, normalizations, lookback_steps, patch_steps, ssl_epochs, learning_rate, batch_size, mask_probability, modality_dropout_probability, max_pretrain_windows, torch))
        real_modality = real_task.modalities[0]
        stages.append(_pretrain_stage(
            model,
            [real_task],
            False,
            seed + 1,
            sequences,
            normalizations,
            lookback_steps,
            patch_steps,
            ssl_epochs,
            learning_rate,
            batch_size,
            mask_probability,
            modality_dropout_probability,
            max_pretrain_windows,
            torch,
            optimizer_parameters=model.modality_pretraining_parameters(real_modality),
            trainable_scope=f"modality_only:{real_modality}",
        ))
    elif regime == "synthetic_then_real_rehearsal":
        stages.append(_pretrain_stage(model, [synthetic_task], False, seed, sequences, normalizations, lookback_steps, patch_steps, ssl_epochs, learning_rate, batch_size, mask_probability, modality_dropout_probability, max_pretrain_windows, torch))
        stages.append(_pretrain_stage(model, [synthetic_task, real_task], True, seed + 1, sequences, normalizations, lookback_steps, patch_steps, ssl_epochs, learning_rate, batch_size, mask_probability, modality_dropout_probability, max_pretrain_windows, torch))
    elif regime == "joint_synthetic_real":
        stages.append(_pretrain_stage(model, [synthetic_task, real_task], True, seed, sequences, normalizations, lookback_steps, patch_steps, ssl_epochs, learning_rate, batch_size, mask_probability, modality_dropout_probability, max_pretrain_windows, torch))
    elif regime == "synthetic_then_japan_then_italy":
        if japan_task is None:
            raise ValueError("Japan task is required for the cross-region regime")
        stages.append(_pretrain_stage(model, [synthetic_task], False, seed, sequences, normalizations, lookback_steps, patch_steps, ssl_epochs, learning_rate, batch_size, mask_probability, modality_dropout_probability, max_pretrain_windows, torch))
        stages.append(_pretrain_stage(model, [japan_task], False, seed + 1, sequences, normalizations, lookback_steps, patch_steps, ssl_epochs, learning_rate, batch_size, mask_probability, modality_dropout_probability, max_pretrain_windows, torch))

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
        for task in (synthetic_task, real_task, *( [japan_task] if japan_task else [] ))
    }
    pretrained_state = clone_state(model)
    downstream_models = {}
    for config_name, modalities in downstream_configs.items():
        load_compatible_state(model, pretrained_state)
        _set_seed(torch, seed)
        linear_probe = train_downstream(
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
            modality_dropout_probability=0.0,
            freeze_backbone=True,
            include_probabilities=regime == "synthetic_then_japan_then_italy",
            coordinate_targets=train_coordinate_targets if regime == "synthetic_then_japan_then_italy" else None,
            seed=seed,
            torch=torch,
        )
        load_compatible_state(model, pretrained_state)
        _set_seed(torch, seed)
        fine_tune = train_downstream(
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
            modality_dropout_probability=modality_dropout_probability,
            freeze_backbone=False,
            include_probabilities=regime == "synthetic_then_japan_then_italy",
            coordinate_targets=train_coordinate_targets if regime == "synthetic_then_japan_then_italy" else None,
            seed=seed,
            torch=torch,
        )
        checkpoint = ""
        if artifact_root:
            path = artifact_root / regime / f"seed_{seed}" / f"{config_name}.pt"
            path.parent.mkdir(parents=True, exist_ok=True)
            torch.save({"schema": "elfquake.multimodal_patch_transformer.v2", "regime": regime, "seed": seed, "downstream_config": config_name, "model_state": model.state_dict()}, path)
            checkpoint = str(path)
        downstream_models[config_name] = {
            "linear_probe": linear_probe,
            "fine_tune": fine_tune,
            "checkpoint": checkpoint,
        }
    return {
        "regime": regime,
        "seed": seed,
        "pretraining_stages": stages,
        "final_reconstruction": final_reconstruction,
        "downstream_models": downstream_models,
    }


def _pretrain_stage(model, tasks, balance, seed, sequences, normalizations, lookback, patch, epochs, learning_rate, batch_size, mask_probability, dropout_probability, max_windows, torch, *, optimizer_parameters=None, trainable_scope="all"):
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
        optimizer_parameters=optimizer_parameters,
        trainable_scope=trainable_scope,
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


def _coordinate_targets(row: dict[str, str]) -> tuple[tuple[float, float, float] | None, ...]:
    try:
        values = json.loads(row.get("target_event_slots_json", "[]") or "[]")
    except json.JSONDecodeError:
        values = []
    targets = []
    for value in values:
        try:
            targets.append((float(value["latitude"]) / 50.0, float(value["longitude"]) / 20.0, float(value["magnitude"]) / 5.0))
        except (KeyError, TypeError, ValueError):
            continue
    return tuple(targets)


def _split_rows(path: Path, *, split_field: str, train_value: str, test_value: str, dataset_id: str | None = None):
    with path.open(newline="", encoding="utf-8") as handle:
        rows = [
            row for row in csv.DictReader(handle)
            if row.get("target_occurred") in {"0", "1"}
            and (dataset_id is None or row.get("dataset_id") == dataset_id)
        ]
    return (
        [row for row in rows if row.get(split_field) == train_value],
        [row for row in rows if row.get(split_field) == test_value],
    )


def _real_modality(sequences: dict) -> str:
    modalities = sorted({sequence.modality for sequence in sequences.values() if sequence.modality.startswith("real_")})
    if not modalities:
        modalities = sorted({sequence.modality for sequence in sequences.values() if sequence.modality == "italy_vlf"})
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

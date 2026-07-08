"""CPU PyTorch self-supervised sequence pretraining."""

from __future__ import annotations

import csv
import json
import math
import random
from datetime import timedelta
from pathlib import Path

from elfquake.models.torch_sequence_data import SequenceDataset, load_sequence_datasets
from elfquake.features.common import format_utc, parse_utc


def pretrain_sequence_autoencoder(
    *,
    sequence_manifest_path: Path,
    out_path: Path,
    modality: str = "",
    lookback_steps: int = 24,
    stride: int = 1,
    train_fraction: float = 0.8,
    mask_probability: float = 0.15,
    clean_loss_weight: float = 0.0,
    epochs: int = 40,
    learning_rate: float = 0.001,
    hidden_units: int = 64,
    embedding_units: int = 16,
    batch_size: int = 32,
    seed: int = 42,
    include_missing_masks: bool = True,
    checkpoint_out: Path | None = None,
    embeddings_out: Path | None = None,
) -> dict[str, object]:
    if lookback_steps < 1:
        raise ValueError("lookback_steps must be at least 1")
    if stride < 1:
        raise ValueError("stride must be at least 1")
    if not 0 < train_fraction < 1:
        raise ValueError("train_fraction must be between 0 and 1")
    if not 0 <= mask_probability < 1:
        raise ValueError("mask_probability must be in [0, 1)")
    if clean_loss_weight < 0:
        raise ValueError("clean_loss_weight must be non-negative")
    if hidden_units < 1 or embedding_units < 1:
        raise ValueError("hidden_units and embedding_units must be at least 1")

    sequences = load_sequence_datasets([sequence_manifest_path], include_missing_masks=include_missing_masks)
    dataset = _select_dataset(sequences, modality=modality)
    windows = _window_values(dataset.values, lookback_steps=lookback_steps, stride=stride)
    report = _base_report(
        sequence_manifest_path=sequence_manifest_path,
        dataset=dataset,
        out_path=out_path,
        row_count=len(dataset.values),
        window_count=len(windows),
        lookback_steps=lookback_steps,
        stride=stride,
        train_fraction=train_fraction,
        mask_probability=mask_probability,
        clean_loss_weight=clean_loss_weight,
        epochs=epochs,
        learning_rate=learning_rate,
        hidden_units=hidden_units,
        embedding_units=embedding_units,
        batch_size=batch_size,
        seed=seed,
        include_missing_masks=include_missing_masks,
        checkpoint_out=checkpoint_out,
        embeddings_out=embeddings_out,
    )
    if len(windows) < 3:
        report["status"] = "insufficient_windows"
        return _write_report(out_path, report)

    split_at = max(1, min(len(windows) - 1, int(len(windows) * train_fraction)))
    train_windows = windows[:split_at]
    test_windows = windows[split_at:]
    report["train_window_count"] = len(train_windows)
    report["test_window_count"] = len(test_windows)
    result = _fit_autoencoder(
        dataset=dataset,
        train_windows=train_windows,
        test_windows=test_windows,
        all_windows=windows,
        lookback_steps=lookback_steps,
        mask_probability=mask_probability,
        clean_loss_weight=clean_loss_weight,
        epochs=epochs,
        learning_rate=learning_rate,
        hidden_units=hidden_units,
        embedding_units=embedding_units,
        batch_size=batch_size,
        seed=seed,
        checkpoint_out=checkpoint_out,
        embeddings_out=embeddings_out,
        train_window_count=len(train_windows),
    )
    report.update(result)
    report["status"] = "evaluated"
    return _write_report(out_path, report)


def compare_sequence_embedding_domains(
    *,
    real_sequence_manifest_path: Path,
    synthetic_sequence_manifest_paths: list[Path],
    out_path: Path,
    real_modality: str = "real_vlf_image",
    synthetic_modality: str = "synthetic_piezo_vlf",
    descriptor_profile: str = "shape",
    lookback_steps: int = 24,
    stride: int = 1,
    train_fraction: float = 0.8,
    mask_probability: float = 0.15,
    clean_loss_weight: float = 0.0,
    inlier_fraction: float = 0.25,
    epochs: int = 40,
    learning_rate: float = 0.001,
    hidden_units: int = 64,
    embedding_units: int = 16,
    batch_size: int = 32,
    seed: int = 42,
    include_missing_masks: bool = True,
    embeddings_out: Path | None = None,
) -> dict[str, object]:
    if not synthetic_sequence_manifest_paths:
        raise ValueError("at least one synthetic sequence manifest is required")
    real_sequences = load_sequence_datasets([real_sequence_manifest_path], include_missing_masks=include_missing_masks)
    synthetic_sequences = load_sequence_datasets(synthetic_sequence_manifest_paths, include_missing_masks=include_missing_masks)
    real_dataset = _select_dataset(real_sequences, modality=real_modality)
    synthetic_datasets = [
        dataset
        for dataset in synthetic_sequences.values()
        if dataset.modality == synthetic_modality
    ]
    if not synthetic_datasets:
        raise ValueError(f"no synthetic sequence datasets found for modality {synthetic_modality}")

    descriptor_names = _descriptor_names(descriptor_profile)
    real_descriptors = _descriptor_windows(real_dataset, lookback_steps=lookback_steps, stride=stride, descriptor_profile=descriptor_profile)
    synthetic_descriptors_by_dataset = {
        dataset.dataset_id: _descriptor_windows(dataset, lookback_steps=lookback_steps, stride=stride, descriptor_profile=descriptor_profile)
        for dataset in synthetic_datasets
    }
    synthetic_descriptors = [
        descriptor
        for descriptors in synthetic_descriptors_by_dataset.values()
        for descriptor in descriptors
    ]
    report: dict[str, object] = {
        "schema": "elfquake.sequence_embedding_domain_comparison.v1",
        "backend": "torch",
        "device": "cpu",
        "real_sequence_manifest": str(real_sequence_manifest_path),
        "synthetic_sequence_manifests": [str(path) for path in synthetic_sequence_manifest_paths],
        "real_modality": real_modality,
        "synthetic_modality": synthetic_modality,
        "descriptor_profile": descriptor_profile,
        "real_dataset_id": real_dataset.dataset_id,
        "synthetic_dataset_ids": [dataset.dataset_id for dataset in synthetic_datasets],
        "lookback_steps": lookback_steps,
        "stride": stride,
        "train_fraction": train_fraction,
        "mask_probability": mask_probability,
        "clean_loss_weight": clean_loss_weight,
        "inlier_fraction": inlier_fraction,
        "epochs": epochs,
        "learning_rate": learning_rate,
        "hidden_units": hidden_units,
        "embedding_units": embedding_units,
        "batch_size": batch_size,
        "seed": seed,
        "include_missing_masks": include_missing_masks,
        "descriptor_names": descriptor_names,
        "real_window_count": len(real_descriptors),
        "synthetic_window_count": len(synthetic_descriptors),
        "embeddings_out": str(embeddings_out) if embeddings_out else "",
        "note": "Synthetic windows are encoded through an autoencoder trained on real VLF descriptor windows, so this is an embedding-domain diagnostic rather than a supervised prediction result.",
    }
    if len(real_descriptors) < 3 or len(synthetic_descriptors) < 1:
        report["status"] = "insufficient_windows"
        return _write_report(out_path, report)
    result = _fit_descriptor_domain_autoencoder(
        real_descriptors=real_descriptors,
        synthetic_descriptors_by_dataset=synthetic_descriptors_by_dataset,
        train_fraction=train_fraction,
        mask_probability=mask_probability,
        clean_loss_weight=clean_loss_weight,
        inlier_fraction=inlier_fraction,
        epochs=epochs,
        learning_rate=learning_rate,
        hidden_units=hidden_units,
        embedding_units=embedding_units,
        batch_size=batch_size,
        seed=seed,
        embeddings_out=embeddings_out,
    )
    report.update(result)
    report["status"] = "evaluated"
    return _write_report(out_path, report)


def evaluate_synthetic_inlier_transfer(
    *,
    real_sequence_manifest_path: Path,
    synthetic_sequence_manifest_paths: list[Path],
    out_path: Path,
    real_modality: str = "real_vlf_image",
    synthetic_modality: str = "synthetic_piezo_vlf",
    descriptor_profile: str = "shape",
    lookback_steps: int = 24,
    stride: int = 1,
    train_fraction: float = 0.8,
    mask_probability: float = 0.15,
    clean_loss_weight: float = 0.0,
    inlier_fraction: float = 0.25,
    epochs: int = 40,
    learning_rate: float = 0.001,
    hidden_units: int = 64,
    embedding_units: int = 16,
    batch_size: int = 32,
    seed: int = 42,
    include_missing_masks: bool = True,
    embeddings_out: Path | None = None,
) -> dict[str, object]:
    if not synthetic_sequence_manifest_paths:
        raise ValueError("at least one synthetic sequence manifest is required")
    if not 0 < train_fraction < 1:
        raise ValueError("train_fraction must be between 0 and 1")
    if not 0 < inlier_fraction <= 1:
        raise ValueError("inlier_fraction must be in (0, 1]")
    real_sequences = load_sequence_datasets([real_sequence_manifest_path], include_missing_masks=include_missing_masks)
    synthetic_sequences = load_sequence_datasets(synthetic_sequence_manifest_paths, include_missing_masks=include_missing_masks)
    real_dataset = _select_dataset(real_sequences, modality=real_modality)
    synthetic_datasets = [
        dataset
        for dataset in synthetic_sequences.values()
        if dataset.modality == synthetic_modality
    ]
    if not synthetic_datasets:
        raise ValueError(f"no synthetic sequence datasets found for modality {synthetic_modality}")

    descriptor_names = _descriptor_names(descriptor_profile)
    real_descriptors = _descriptor_windows(real_dataset, lookback_steps=lookback_steps, stride=stride, descriptor_profile=descriptor_profile)
    synthetic_descriptors_by_dataset = {
        dataset.dataset_id: _descriptor_windows(dataset, lookback_steps=lookback_steps, stride=stride, descriptor_profile=descriptor_profile)
        for dataset in synthetic_datasets
    }
    synthetic_window_count = sum(len(descriptors) for descriptors in synthetic_descriptors_by_dataset.values())
    report: dict[str, object] = {
        "schema": "elfquake.synthetic_inlier_transfer_reconstruction.v1",
        "backend": "torch",
        "device": "cpu",
        "real_sequence_manifest": str(real_sequence_manifest_path),
        "synthetic_sequence_manifests": [str(path) for path in synthetic_sequence_manifest_paths],
        "real_modality": real_modality,
        "synthetic_modality": synthetic_modality,
        "descriptor_profile": descriptor_profile,
        "real_dataset_id": real_dataset.dataset_id,
        "synthetic_dataset_ids": [dataset.dataset_id for dataset in synthetic_datasets],
        "lookback_steps": lookback_steps,
        "stride": stride,
        "train_fraction": train_fraction,
        "mask_probability": mask_probability,
        "clean_loss_weight": clean_loss_weight,
        "inlier_fraction": inlier_fraction,
        "epochs": epochs,
        "learning_rate": learning_rate,
        "hidden_units": hidden_units,
        "embedding_units": embedding_units,
        "batch_size": batch_size,
        "seed": seed,
        "include_missing_masks": include_missing_masks,
        "descriptor_names": descriptor_names,
        "real_window_count": len(real_descriptors),
        "synthetic_window_count": synthetic_window_count,
        "embeddings_out": str(embeddings_out) if embeddings_out else "",
        "note": "The model is trained only on synthetic piezo/VLF descriptor windows selected as real-like inliers, then evaluated on held-out real VLF descriptors.",
    }
    if len(real_descriptors) < 3 or synthetic_window_count < 1:
        report["status"] = "insufficient_windows"
        return _write_report(out_path, report)
    result = _fit_synthetic_inlier_transfer_autoencoder(
        real_descriptors=real_descriptors,
        synthetic_descriptors_by_dataset=synthetic_descriptors_by_dataset,
        train_fraction=train_fraction,
        mask_probability=mask_probability,
        clean_loss_weight=clean_loss_weight,
        inlier_fraction=inlier_fraction,
        epochs=epochs,
        learning_rate=learning_rate,
        hidden_units=hidden_units,
        embedding_units=embedding_units,
        batch_size=batch_size,
        seed=seed,
        embeddings_out=embeddings_out,
    )
    report.update(result)
    report["status"] = "evaluated"
    return _write_report(out_path, report)


def evaluate_mixed_domain_alignment(
    *,
    real_sequence_manifest_path: Path,
    synthetic_sequence_manifest_paths: list[Path],
    out_path: Path,
    real_modality: str = "real_vlf_image",
    synthetic_modality: str = "synthetic_piezo_vlf",
    descriptor_profile: str = "shape",
    lookback_steps: int = 24,
    stride: int = 1,
    train_fraction: float = 0.8,
    mask_probability: float = 0.15,
    clean_loss_weight: float = 0.0,
    inlier_fraction: float = 0.25,
    inlier_method: str = "local",
    control_methods: list[str] | None = None,
    max_synthetic_train_windows: int = 15000,
    balance_synthetic_sources: bool = True,
    coral_weight: float = 0.1,
    epochs: int = 40,
    learning_rate: float = 0.001,
    hidden_units: int = 64,
    embedding_units: int = 16,
    batch_size: int = 32,
    seed: int = 42,
    include_missing_masks: bool = True,
    embeddings_out: Path | None = None,
) -> dict[str, object]:
    if not synthetic_sequence_manifest_paths:
        raise ValueError("at least one synthetic sequence manifest is required")
    if not 0 < train_fraction < 1:
        raise ValueError("train_fraction must be between 0 and 1")
    if not 0 < inlier_fraction <= 1:
        raise ValueError("inlier_fraction must be in (0, 1]")
    if coral_weight < 0:
        raise ValueError("coral_weight must be non-negative")
    real_sequences = load_sequence_datasets([real_sequence_manifest_path], include_missing_masks=include_missing_masks)
    synthetic_sequences = load_sequence_datasets(synthetic_sequence_manifest_paths, include_missing_masks=include_missing_masks)
    real_dataset = _select_dataset(real_sequences, modality=real_modality)
    synthetic_datasets = [
        dataset
        for dataset in synthetic_sequences.values()
        if dataset.modality == synthetic_modality
    ]
    if not synthetic_datasets:
        raise ValueError(f"no synthetic sequence datasets found for modality {synthetic_modality}")

    descriptor_names = _descriptor_names(descriptor_profile)
    real_descriptors = _descriptor_windows(real_dataset, lookback_steps=lookback_steps, stride=stride, descriptor_profile=descriptor_profile)
    synthetic_descriptors_by_dataset = {
        dataset.dataset_id: _descriptor_windows(dataset, lookback_steps=lookback_steps, stride=stride, descriptor_profile=descriptor_profile)
        for dataset in synthetic_datasets
    }
    synthetic_window_count = sum(len(descriptors) for descriptors in synthetic_descriptors_by_dataset.values())
    report: dict[str, object] = {
        "schema": "elfquake.mixed_domain_alignment.v1",
        "backend": "torch",
        "device": "cpu",
        "real_sequence_manifest": str(real_sequence_manifest_path),
        "synthetic_sequence_manifests": [str(path) for path in synthetic_sequence_manifest_paths],
        "real_modality": real_modality,
        "synthetic_modality": synthetic_modality,
        "descriptor_profile": descriptor_profile,
        "real_dataset_id": real_dataset.dataset_id,
        "synthetic_dataset_ids": [dataset.dataset_id for dataset in synthetic_datasets],
        "lookback_steps": lookback_steps,
        "stride": stride,
        "train_fraction": train_fraction,
        "mask_probability": mask_probability,
        "clean_loss_weight": clean_loss_weight,
        "inlier_fraction": inlier_fraction,
        "inlier_method": inlier_method,
        "control_methods": control_methods or ["centroid", "random", "full"],
        "max_synthetic_train_windows": max_synthetic_train_windows,
        "balance_synthetic_sources": balance_synthetic_sources,
        "coral_weight": coral_weight,
        "epochs": epochs,
        "learning_rate": learning_rate,
        "hidden_units": hidden_units,
        "embedding_units": embedding_units,
        "batch_size": batch_size,
        "seed": seed,
        "include_missing_masks": include_missing_masks,
        "descriptor_names": descriptor_names,
        "real_window_count": len(real_descriptors),
        "synthetic_window_count": synthetic_window_count,
        "embeddings_out": str(embeddings_out) if embeddings_out else "",
        "note": "Mixed real/synthetic self-supervised descriptor alignment with local inliers, CORAL loss, descriptor-gap reporting, and synthetic selection controls.",
    }
    if len(real_descriptors) < 3 or synthetic_window_count < 1:
        report["status"] = "insufficient_windows"
        return _write_report(out_path, report)
    result = _fit_mixed_domain_alignment(
        real_descriptors=real_descriptors,
        synthetic_descriptors_by_dataset=synthetic_descriptors_by_dataset,
        descriptor_names=descriptor_names,
        train_fraction=train_fraction,
        mask_probability=mask_probability,
        clean_loss_weight=clean_loss_weight,
        inlier_fraction=inlier_fraction,
        inlier_method=inlier_method,
        control_methods=control_methods or ["centroid", "random", "full"],
        max_synthetic_train_windows=max_synthetic_train_windows,
        balance_synthetic_sources=balance_synthetic_sources,
        coral_weight=coral_weight,
        epochs=epochs,
        learning_rate=learning_rate,
        hidden_units=hidden_units,
        embedding_units=embedding_units,
        batch_size=batch_size,
        seed=seed,
        embeddings_out=embeddings_out,
    )
    report.update(result)
    report["status"] = "evaluated"
    return _write_report(out_path, report)


def score_sequence_anomalies(
    *,
    sequence_manifest_path: Path,
    out_path: Path,
    scores_out: Path,
    modality: str = "real_vlf_image",
    descriptor_profile: str = "shape",
    lookback_steps: int = 24,
    stride: int = 1,
    train_fraction: float = 0.8,
    forecast_horizon_days: int = 7,
    alert_threshold: float = 0.8,
    mask_probability: float = 0.15,
    clean_loss_weight: float = 0.0,
    epochs: int = 30,
    learning_rate: float = 0.0003,
    hidden_units: int = 32,
    embedding_units: int = 8,
    batch_size: int = 32,
    seed: int = 42,
    include_missing_masks: bool = True,
) -> dict[str, object]:
    if lookback_steps < 1:
        raise ValueError("lookback_steps must be at least 1")
    if stride < 1:
        raise ValueError("stride must be at least 1")
    if not 0 < train_fraction < 1:
        raise ValueError("train_fraction must be between 0 and 1")
    if forecast_horizon_days < 1:
        raise ValueError("forecast_horizon_days must be at least 1")
    if not 0 <= alert_threshold <= 1:
        raise ValueError("alert_threshold must be between 0 and 1")

    sequences = load_sequence_datasets([sequence_manifest_path], include_missing_masks=include_missing_masks)
    dataset = _select_dataset(sequences, modality=modality)
    descriptors = _descriptor_windows(dataset, lookback_steps=lookback_steps, stride=stride, descriptor_profile=descriptor_profile)
    descriptor_names = _descriptor_names(descriptor_profile)
    window_times = _descriptor_window_times(dataset, lookback_steps=lookback_steps, stride=stride)
    report: dict[str, object] = {
        "schema": "elfquake.sequence_anomaly_forecast.v1",
        "backend": "torch",
        "device": "cpu",
        "sequence_manifest": str(sequence_manifest_path),
        "dataset_id": dataset.dataset_id,
        "modality": dataset.modality,
        "descriptor_profile": descriptor_profile,
        "descriptor_names": descriptor_names,
        "lookback_steps": lookback_steps,
        "stride": stride,
        "train_fraction": train_fraction,
        "forecast_horizon_days": forecast_horizon_days,
        "alert_threshold": alert_threshold,
        "mask_probability": mask_probability,
        "clean_loss_weight": clean_loss_weight,
        "epochs": epochs,
        "learning_rate": learning_rate,
        "hidden_units": hidden_units,
        "embedding_units": embedding_units,
        "batch_size": batch_size,
        "seed": seed,
        "include_missing_masks": include_missing_masks,
        "row_count": len(dataset.values),
        "window_count": len(descriptors),
        "scores_out": str(scores_out),
        "note": "Label-free VLF anomaly smoke forecast. The score is not trained on earthquake labels and is not an earthquake prediction claim.",
    }
    if len(descriptors) < 3:
        report["status"] = "insufficient_windows"
        return _write_report(out_path, report)

    result = _fit_descriptor_anomaly_autoencoder(
        descriptors=descriptors,
        window_times=window_times,
        train_fraction=train_fraction,
        forecast_horizon_days=forecast_horizon_days,
        alert_threshold=alert_threshold,
        mask_probability=mask_probability,
        clean_loss_weight=clean_loss_weight,
        epochs=epochs,
        learning_rate=learning_rate,
        hidden_units=hidden_units,
        embedding_units=embedding_units,
        batch_size=batch_size,
        seed=seed,
        scores_out=scores_out,
    )
    report.update(result)
    report["status"] = "evaluated"
    return _write_report(out_path, report)


def _select_dataset(
    sequences: dict[tuple[str, str], SequenceDataset],
    *,
    modality: str,
) -> SequenceDataset:
    matches = [dataset for dataset in sequences.values() if not modality or dataset.modality == modality]
    if len(matches) != 1:
        names = ", ".join(f"{dataset.dataset_id}/{dataset.modality}" for dataset in sequences.values())
        raise ValueError(f"expected one sequence dataset for modality {modality or '<any>'}, got: {names}")
    return matches[0]


def _window_values(
    values: list[list[float]],
    *,
    lookback_steps: int,
    stride: int,
) -> list[list[list[float]]]:
    windows = []
    for end_index in range(lookback_steps, len(values) + 1, stride):
        windows.append(values[end_index - lookback_steps:end_index])
    return windows


def _descriptor_names(descriptor_profile: str) -> list[str]:
    names = [
        "global_mean",
        "global_std",
        "global_min",
        "global_max",
        "global_q10",
        "global_q25",
        "global_q50",
        "global_q75",
        "global_q90",
        "step_mean_std",
        "step_mean_min",
        "step_mean_max",
        "step_diff_mean",
        "step_diff_std",
        "step_diff_max_abs",
        "feature_mean_std",
        "feature_std_mean",
        "feature_std_std",
    ]
    if descriptor_profile == "full":
        return names
    if descriptor_profile == "shape":
        return [
            "global_std",
            "global_iqr",
            "global_robust_range",
            "step_mean_std",
            "step_diff_mean",
            "step_diff_std",
            "step_diff_max_abs",
            "feature_mean_std",
            "feature_std_mean",
            "feature_std_std",
        ]
    raise ValueError(f"unknown descriptor profile: {descriptor_profile}")


def _descriptor_windows(
    dataset: SequenceDataset,
    *,
    lookback_steps: int,
    stride: int,
    descriptor_profile: str,
) -> list[list[float]]:
    values = _standardize_value_rows(dataset.values)
    descriptors = []
    for window in _window_values(values, lookback_steps=lookback_steps, stride=stride):
        descriptors.append(_window_descriptor(window, descriptor_profile=descriptor_profile))
    return descriptors


def _descriptor_window_times(
    dataset: SequenceDataset,
    *,
    lookback_steps: int,
    stride: int,
) -> list[str]:
    times_by_index = {index: time_utc for time_utc, index in dataset.time_to_index.items()}
    return [
        times_by_index.get(end_index - 1, str(end_index - 1))
        for end_index in range(lookback_steps, len(dataset.values) + 1, stride)
    ]


def _standardize_value_rows(values: list[list[float]]) -> list[list[float]]:
    if not values:
        return []
    feature_count = len(values[0])
    means = []
    stds = []
    for index in range(feature_count):
        column = [row[index] for row in values]
        mean = sum(column) / len(column)
        std = _std(column, mean=mean) or 1.0
        means.append(mean)
        stds.append(std)
    return [[(value - means[index]) / stds[index] for index, value in enumerate(row)] for row in values]


def _window_descriptor(window: list[list[float]], *, descriptor_profile: str) -> list[float]:
    flattened = [value for row in window for value in row]
    step_means = [sum(row) / len(row) if row else 0.0 for row in window]
    step_diffs = [step_means[index] - step_means[index - 1] for index in range(1, len(step_means))]
    feature_columns = list(zip(*window)) if window else []
    feature_means = [sum(column) / len(column) for column in feature_columns]
    feature_stds = [_std(list(column), mean=sum(column) / len(column)) for column in feature_columns]
    global_mean = sum(flattened) / len(flattened) if flattened else 0.0
    full = [
        global_mean,
        _std(flattened, mean=global_mean),
        min(flattened) if flattened else 0.0,
        max(flattened) if flattened else 0.0,
        _quantile(flattened, 0.10),
        _quantile(flattened, 0.25),
        _quantile(flattened, 0.50),
        _quantile(flattened, 0.75),
        _quantile(flattened, 0.90),
        _std(step_means, mean=sum(step_means) / len(step_means)) if step_means else 0.0,
        min(step_means) if step_means else 0.0,
        max(step_means) if step_means else 0.0,
        sum(step_diffs) / len(step_diffs) if step_diffs else 0.0,
        _std(step_diffs, mean=sum(step_diffs) / len(step_diffs)) if step_diffs else 0.0,
        max((abs(value) for value in step_diffs), default=0.0),
        _std(feature_means, mean=sum(feature_means) / len(feature_means)) if feature_means else 0.0,
        sum(feature_stds) / len(feature_stds) if feature_stds else 0.0,
        _std(feature_stds, mean=sum(feature_stds) / len(feature_stds)) if feature_stds else 0.0,
    ]
    if descriptor_profile == "full":
        return full
    if descriptor_profile == "shape":
        return [
            full[1],
            full[7] - full[5],
            full[8] - full[4],
            full[9],
            full[12],
            full[13],
            full[14],
            full[15],
            full[16],
            full[17],
        ]
    raise ValueError(f"unknown descriptor profile: {descriptor_profile}")


def _quantile(values: list[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round((len(ordered) - 1) * fraction)))
    return ordered[index]


def _std(values: list[float], *, mean: float) -> float:
    if not values:
        return 0.0
    return math.sqrt(sum((value - mean) ** 2 for value in values) / len(values))


def _base_report(
    *,
    sequence_manifest_path: Path,
    dataset: SequenceDataset,
    out_path: Path,
    row_count: int,
    window_count: int,
    lookback_steps: int,
    stride: int,
    train_fraction: float,
    mask_probability: float,
    clean_loss_weight: float,
    epochs: int,
    learning_rate: float,
    hidden_units: int,
    embedding_units: int,
    batch_size: int,
    seed: int,
    include_missing_masks: bool,
    checkpoint_out: Path | None,
    embeddings_out: Path | None,
) -> dict[str, object]:
    return {
        "schema": "elfquake.sequence_autoencoder_pretrain.v1",
        "backend": "torch",
        "device": "cpu",
        "sequence_manifest": str(sequence_manifest_path),
        "dataset_id": dataset.dataset_id,
        "modality": dataset.modality,
        "row_count": row_count,
        "window_count": window_count,
        "feature_count": len(dataset.feature_names),
        "feature_names": dataset.feature_names,
        "lookback_steps": lookback_steps,
        "stride": stride,
        "train_fraction": train_fraction,
        "mask_probability": mask_probability,
        "clean_loss_weight": clean_loss_weight,
        "epochs": epochs,
        "learning_rate": learning_rate,
        "hidden_units": hidden_units,
        "embedding_units": embedding_units,
        "batch_size": batch_size,
        "seed": seed,
        "include_missing_masks": include_missing_masks,
        "checkpoint_out": str(checkpoint_out) if checkpoint_out else "",
        "embeddings_out": str(embeddings_out) if embeddings_out else "",
        "output": str(out_path),
    }


def _fit_autoencoder(
    *,
    dataset: SequenceDataset,
    train_windows: list[list[list[float]]],
    test_windows: list[list[list[float]]],
    all_windows: list[list[list[float]]],
    lookback_steps: int,
    mask_probability: float,
    clean_loss_weight: float,
    epochs: int,
    learning_rate: float,
    hidden_units: int,
    embedding_units: int,
    batch_size: int,
    seed: int,
    checkpoint_out: Path | None,
    embeddings_out: Path | None,
    train_window_count: int,
) -> dict[str, object]:
    torch = _import_torch()
    _set_deterministic_seed(torch, seed)
    train_tensor = torch.tensor(train_windows, dtype=torch.float32)
    test_tensor = torch.tensor(test_windows, dtype=torch.float32)
    all_tensor = torch.tensor(all_windows, dtype=torch.float32)
    train_tensor, test_tensor, all_tensor, mean, std = _standardize(train_tensor, test_tensor, all_tensor, torch=torch)
    input_units = lookback_steps * len(dataset.feature_names)
    model = _SequenceAutoencoder(torch, input_units=input_units, hidden_units=hidden_units, embedding_units=embedding_units)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    generator = torch.Generator().manual_seed(seed)
    train_flat = train_tensor.reshape(len(train_tensor), input_units)
    test_flat = test_tensor.reshape(len(test_tensor), input_units)
    all_flat = all_tensor.reshape(len(all_tensor), input_units)
    first_loss = None
    last_loss = 0.0
    model.train()
    for _ in range(epochs):
        permutation = torch.randperm(len(train_flat), generator=generator)
        epoch_loss = 0.0
        for start in range(0, len(train_flat), batch_size):
            indices = permutation[start:start + batch_size]
            batch = train_flat[indices]
            masked, mask = _masked_input(batch, mask_probability=mask_probability, generator=generator, torch=torch)
            optimizer.zero_grad()
            reconstruction = model(masked)
            loss = _masked_mse(reconstruction, batch, mask, torch=torch)
            if clean_loss_weight:
                loss = loss + clean_loss_weight * torch.nn.functional.mse_loss(model(batch), batch)
            loss.backward()
            optimizer.step()
            epoch_loss += float(loss.item()) * len(indices)
        last_loss = epoch_loss / len(train_flat)
        if first_loss is None:
            first_loss = last_loss
    model.eval()
    with torch.no_grad():
        train_metrics = _reconstruction_metrics(model, train_flat, mask_probability=mask_probability, generator=generator, torch=torch)
        test_metrics = _reconstruction_metrics(model, test_flat, mask_probability=mask_probability, generator=generator, torch=torch)
        embeddings = model.encode(all_flat)
    if checkpoint_out:
        _save_checkpoint(
            model,
            checkpoint_out=checkpoint_out,
            torch=torch,
            dataset=dataset,
            lookback_steps=lookback_steps,
            hidden_units=hidden_units,
            embedding_units=embedding_units,
            mean=mean,
            std=std,
        )
    if embeddings_out:
        _write_embeddings(
            embeddings_out,
            embeddings=embeddings.tolist(),
            train_window_count=train_window_count,
        )
    return {
        "first_train_loss": round(first_loss or 0.0, 8),
        "last_train_loss": round(last_loss, 8),
        "train_reconstruction": train_metrics,
        "test_reconstruction": test_metrics,
        "embedding_mean": _rounded_vector(embeddings.mean(dim=0).tolist()),
        "embedding_std": _rounded_vector(embeddings.std(dim=0, unbiased=False).tolist()),
    }


def _fit_descriptor_domain_autoencoder(
    *,
    real_descriptors: list[list[float]],
    synthetic_descriptors_by_dataset: dict[str, list[list[float]]],
    train_fraction: float,
    mask_probability: float,
    clean_loss_weight: float,
    inlier_fraction: float,
    epochs: int,
    learning_rate: float,
    hidden_units: int,
    embedding_units: int,
    batch_size: int,
    seed: int,
    embeddings_out: Path | None,
) -> dict[str, object]:
    torch = _import_torch()
    _set_deterministic_seed(torch, seed)
    split_at = max(1, min(len(real_descriptors) - 1, int(len(real_descriptors) * train_fraction)))
    real_train = torch.tensor(real_descriptors[:split_at], dtype=torch.float32)
    real_test = torch.tensor(real_descriptors[split_at:], dtype=torch.float32)
    synthetic_rows = [
        (dataset_id, window_index, descriptor)
        for dataset_id, descriptors in synthetic_descriptors_by_dataset.items()
        for window_index, descriptor in enumerate(descriptors)
    ]
    synthetic = torch.tensor([row[2] for row in synthetic_rows], dtype=torch.float32)
    real_train, real_test, synthetic, mean, std = _standardize_vectors(real_train, real_test, synthetic, torch=torch)
    model = _SequenceAutoencoder(
        torch,
        input_units=int(real_train.shape[1]),
        hidden_units=hidden_units,
        embedding_units=embedding_units,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    generator = torch.Generator().manual_seed(seed)
    first_loss = None
    last_loss = 0.0
    model.train()
    for _ in range(epochs):
        permutation = torch.randperm(len(real_train), generator=generator)
        epoch_loss = 0.0
        for start in range(0, len(real_train), batch_size):
            indices = permutation[start:start + batch_size]
            batch = real_train[indices]
            masked, mask = _masked_input(batch, mask_probability=mask_probability, generator=generator, torch=torch)
            optimizer.zero_grad()
            reconstruction = model(masked)
            loss = _masked_mse(reconstruction, batch, mask, torch=torch)
            if clean_loss_weight:
                loss = loss + clean_loss_weight * torch.nn.functional.mse_loss(model(batch), batch)
            loss.backward()
            optimizer.step()
            epoch_loss += float(loss.item()) * len(indices)
        last_loss = epoch_loss / len(real_train)
        if first_loss is None:
            first_loss = last_loss
    model.eval()
    with torch.no_grad():
        train_metrics = _reconstruction_metrics(model, real_train, mask_probability=mask_probability, generator=generator, torch=torch)
        test_metrics = _reconstruction_metrics(model, real_test, mask_probability=mask_probability, generator=generator, torch=torch)
        synthetic_metrics = _reconstruction_metrics(model, synthetic, mask_probability=mask_probability, generator=generator, torch=torch)
        real_train_embeddings = model.encode(real_train)
        real_test_embeddings = model.encode(real_test)
        synthetic_embeddings = model.encode(synthetic)
    synthetic_inlier_indices = _synthetic_inlier_indices(
        synthetic_descriptors=synthetic.tolist(),
        real_descriptors=real_train.tolist(),
        fraction=inlier_fraction,
    )
    synthetic_inlier_embeddings = [synthetic_embeddings[index].tolist() for index in synthetic_inlier_indices]
    if embeddings_out:
        _write_domain_embeddings(
            embeddings_out,
            real_train_embeddings=real_train_embeddings.tolist(),
            real_test_embeddings=real_test_embeddings.tolist(),
            synthetic_embeddings=synthetic_embeddings.tolist(),
            synthetic_rows=synthetic_rows,
            synthetic_inlier_indices=set(synthetic_inlier_indices),
        )
    return {
        "train_window_count": len(real_train),
        "test_window_count": len(real_test),
        "synthetic_window_count": len(synthetic),
        "first_train_loss": round(first_loss or 0.0, 8),
        "last_train_loss": round(last_loss, 8),
        "real_train_reconstruction": train_metrics,
        "real_test_reconstruction": test_metrics,
        "synthetic_reconstruction": synthetic_metrics,
        "embedding_comparison": _embedding_comparison(
            real_embeddings=real_test_embeddings.tolist(),
            synthetic_embeddings=synthetic_embeddings.tolist(),
        ),
        "synthetic_inlier_embedding_comparison": _embedding_comparison(
            real_embeddings=real_test_embeddings.tolist(),
            synthetic_embeddings=synthetic_inlier_embeddings,
        ),
        "synthetic_inlier_count": len(synthetic_inlier_indices),
        "synthetic_inlier_fraction": inlier_fraction,
        "descriptor_standardization_mean": _rounded_vector(mean.tolist()),
        "descriptor_standardization_std": _rounded_vector(std.tolist()),
    }


def _fit_synthetic_inlier_transfer_autoencoder(
    *,
    real_descriptors: list[list[float]],
    synthetic_descriptors_by_dataset: dict[str, list[list[float]]],
    train_fraction: float,
    mask_probability: float,
    clean_loss_weight: float,
    inlier_fraction: float,
    epochs: int,
    learning_rate: float,
    hidden_units: int,
    embedding_units: int,
    batch_size: int,
    seed: int,
    embeddings_out: Path | None,
) -> dict[str, object]:
    torch = _import_torch()
    _set_deterministic_seed(torch, seed)
    split_at = max(1, min(len(real_descriptors) - 1, int(len(real_descriptors) * train_fraction)))
    real_train = torch.tensor(real_descriptors[:split_at], dtype=torch.float32)
    real_test = torch.tensor(real_descriptors[split_at:], dtype=torch.float32)
    synthetic_rows = [
        (dataset_id, window_index, descriptor)
        for dataset_id, descriptors in synthetic_descriptors_by_dataset.items()
        for window_index, descriptor in enumerate(descriptors)
    ]
    synthetic = torch.tensor([row[2] for row in synthetic_rows], dtype=torch.float32)
    real_train, real_test, synthetic, mean, std = _standardize_vectors(real_train, real_test, synthetic, torch=torch)
    synthetic_inlier_indices = _synthetic_inlier_indices(
        synthetic_descriptors=synthetic.tolist(),
        real_descriptors=real_train.tolist(),
        fraction=inlier_fraction,
    )
    if not synthetic_inlier_indices:
        return {
            "status": "insufficient_synthetic_inliers",
            "real_train_window_count": len(real_train),
            "real_test_window_count": len(real_test),
            "synthetic_inlier_count": 0,
            "synthetic_inlier_fraction": inlier_fraction,
        }
    synthetic_inliers = synthetic[synthetic_inlier_indices]
    synthetic_inlier_rows = [synthetic_rows[index] for index in synthetic_inlier_indices]
    model = _SequenceAutoencoder(
        torch,
        input_units=int(real_train.shape[1]),
        hidden_units=hidden_units,
        embedding_units=embedding_units,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    generator = torch.Generator().manual_seed(seed)
    first_loss = None
    last_loss = 0.0
    model.train()
    for _ in range(epochs):
        permutation = torch.randperm(len(synthetic_inliers), generator=generator)
        epoch_loss = 0.0
        for start in range(0, len(synthetic_inliers), batch_size):
            indices = permutation[start:start + batch_size]
            batch = synthetic_inliers[indices]
            masked, mask = _masked_input(batch, mask_probability=mask_probability, generator=generator, torch=torch)
            optimizer.zero_grad()
            reconstruction = model(masked)
            loss = _masked_mse(reconstruction, batch, mask, torch=torch)
            if clean_loss_weight:
                loss = loss + clean_loss_weight * torch.nn.functional.mse_loss(model(batch), batch)
            loss.backward()
            optimizer.step()
            epoch_loss += float(loss.item()) * len(indices)
        last_loss = epoch_loss / len(synthetic_inliers)
        if first_loss is None:
            first_loss = last_loss
    model.eval()
    with torch.no_grad():
        synthetic_inlier_metrics = _reconstruction_metrics(model, synthetic_inliers, mask_probability=mask_probability, generator=generator, torch=torch)
        real_train_metrics = _reconstruction_metrics(model, real_train, mask_probability=mask_probability, generator=generator, torch=torch)
        real_test_metrics = _reconstruction_metrics(model, real_test, mask_probability=mask_probability, generator=generator, torch=torch)
        real_train_embeddings = model.encode(real_train)
        real_test_embeddings = model.encode(real_test)
        synthetic_inlier_embeddings = model.encode(synthetic_inliers)
    if embeddings_out:
        _write_domain_embeddings(
            embeddings_out,
            real_train_embeddings=real_train_embeddings.tolist(),
            real_test_embeddings=real_test_embeddings.tolist(),
            synthetic_embeddings=synthetic_inlier_embeddings.tolist(),
            synthetic_rows=synthetic_inlier_rows,
            synthetic_inlier_indices=set(range(len(synthetic_inlier_rows))),
        )
    return {
        "real_train_window_count": len(real_train),
        "real_test_window_count": len(real_test),
        "synthetic_inlier_count": len(synthetic_inlier_indices),
        "synthetic_inlier_fraction": inlier_fraction,
        "first_train_loss": round(first_loss or 0.0, 8),
        "last_train_loss": round(last_loss, 8),
        "synthetic_inlier_train_reconstruction": synthetic_inlier_metrics,
        "real_train_reconstruction": real_train_metrics,
        "real_test_reconstruction": real_test_metrics,
        "embedding_comparison": _embedding_comparison(
            real_embeddings=real_test_embeddings.tolist(),
            synthetic_embeddings=synthetic_inlier_embeddings.tolist(),
        ),
        "descriptor_standardization_mean": _rounded_vector(mean.tolist()),
        "descriptor_standardization_std": _rounded_vector(std.tolist()),
    }


def _fit_mixed_domain_alignment(
    *,
    real_descriptors: list[list[float]],
    synthetic_descriptors_by_dataset: dict[str, list[list[float]]],
    descriptor_names: list[str],
    train_fraction: float,
    mask_probability: float,
    clean_loss_weight: float,
    inlier_fraction: float,
    inlier_method: str,
    control_methods: list[str],
    max_synthetic_train_windows: int,
    balance_synthetic_sources: bool,
    coral_weight: float,
    epochs: int,
    learning_rate: float,
    hidden_units: int,
    embedding_units: int,
    batch_size: int,
    seed: int,
    embeddings_out: Path | None,
) -> dict[str, object]:
    torch = _import_torch()
    _set_deterministic_seed(torch, seed)
    split_at = max(1, min(len(real_descriptors) - 1, int(len(real_descriptors) * train_fraction)))
    real_train = torch.tensor(real_descriptors[:split_at], dtype=torch.float32)
    real_test = torch.tensor(real_descriptors[split_at:], dtype=torch.float32)
    synthetic_rows = [
        (dataset_id, window_index, descriptor)
        for dataset_id, descriptors in synthetic_descriptors_by_dataset.items()
        for window_index, descriptor in enumerate(descriptors)
    ]
    synthetic = torch.tensor([row[2] for row in synthetic_rows], dtype=torch.float32)
    real_train, real_test, synthetic, mean, std = _standardize_vectors(real_train, real_test, synthetic, torch=torch)
    primary_indices = _select_synthetic_training_indices(
        torch=torch,
        synthetic=synthetic,
        real_train=real_train,
        synthetic_rows=synthetic_rows,
        method=inlier_method,
        fraction=inlier_fraction,
        max_count=max_synthetic_train_windows,
        balance_synthetic_sources=balance_synthetic_sources,
        seed=seed,
    )
    if not primary_indices:
        return {
            "status": "insufficient_synthetic_inliers",
            "real_train_window_count": len(real_train),
            "real_test_window_count": len(real_test),
            "synthetic_train_count": 0,
        }
    primary_result = _train_mixed_alignment_model(
        torch=torch,
        real_train=real_train,
        real_test=real_test,
        synthetic_train=synthetic[primary_indices],
        mask_probability=mask_probability,
        clean_loss_weight=clean_loss_weight,
        coral_weight=coral_weight,
        epochs=epochs,
        learning_rate=learning_rate,
        hidden_units=hidden_units,
        embedding_units=embedding_units,
        batch_size=batch_size,
        seed=seed,
    )
    selected_rows = [synthetic_rows[index] for index in primary_indices]
    if embeddings_out:
        _write_domain_embeddings(
            embeddings_out,
            real_train_embeddings=primary_result["real_train_embeddings"],
            real_test_embeddings=primary_result["real_test_embeddings"],
            synthetic_embeddings=primary_result["synthetic_embeddings"],
            synthetic_rows=selected_rows,
            synthetic_inlier_indices=set(range(len(selected_rows))),
        )
    control_runs = {}
    for offset, method in enumerate(control_methods):
        if method == inlier_method:
            continue
        control_indices = _select_synthetic_training_indices(
            torch=torch,
            synthetic=synthetic,
            real_train=real_train,
            synthetic_rows=synthetic_rows,
            method=method,
            fraction=inlier_fraction,
            max_count=len(primary_indices),
            balance_synthetic_sources=balance_synthetic_sources,
            seed=seed + offset + 1,
        )
        if not control_indices:
            control_runs[method] = {"status": "insufficient_synthetic_windows"}
            continue
        control = _train_mixed_alignment_model(
            torch=torch,
            real_train=real_train,
            real_test=real_test,
            synthetic_train=synthetic[control_indices],
            mask_probability=mask_probability,
            clean_loss_weight=clean_loss_weight,
            coral_weight=coral_weight,
            epochs=epochs,
            learning_rate=learning_rate,
            hidden_units=hidden_units,
            embedding_units=embedding_units,
            batch_size=batch_size,
            seed=seed + offset + 1,
        )
        control_runs[method] = _public_alignment_result(control, synthetic_train_count=len(control_indices))
    public_primary = _public_alignment_result(primary_result, synthetic_train_count=len(primary_indices))
    synthetic_selected = synthetic[primary_indices]
    return {
        "real_train_window_count": len(real_train),
        "real_test_window_count": len(real_test),
        "synthetic_train_count": len(primary_indices),
        "synthetic_selection_counts": _selection_counts(primary_indices, synthetic_rows),
        "selection_method": inlier_method,
        "primary": public_primary,
        "control_runs": control_runs,
        "descriptor_gap": _descriptor_gap(
            real_rows=real_train.tolist(),
            synthetic_rows=synthetic_selected.tolist(),
            descriptor_names=descriptor_names,
        ),
        "descriptor_standardization_mean": _rounded_vector(mean.tolist()),
        "descriptor_standardization_std": _rounded_vector(std.tolist()),
    }


def _fit_descriptor_anomaly_autoencoder(
    *,
    descriptors: list[list[float]],
    window_times: list[str],
    train_fraction: float,
    forecast_horizon_days: int,
    alert_threshold: float,
    mask_probability: float,
    clean_loss_weight: float,
    epochs: int,
    learning_rate: float,
    hidden_units: int,
    embedding_units: int,
    batch_size: int,
    seed: int,
    scores_out: Path,
) -> dict[str, object]:
    torch = _import_torch()
    _set_deterministic_seed(torch, seed)
    split_at = max(1, min(len(descriptors) - 1, int(len(descriptors) * train_fraction)))
    train = torch.tensor(descriptors[:split_at], dtype=torch.float32)
    test = torch.tensor(descriptors[split_at:], dtype=torch.float32)
    all_rows = torch.tensor(descriptors, dtype=torch.float32)
    train, test, all_rows, mean, std = _standardize_vectors(train, test, all_rows, torch=torch)
    model = _SequenceAutoencoder(
        torch,
        input_units=int(train.shape[1]),
        hidden_units=hidden_units,
        embedding_units=embedding_units,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    generator = torch.Generator().manual_seed(seed)
    first_loss = None
    last_loss = 0.0
    model.train()
    for _ in range(epochs):
        permutation = torch.randperm(len(train), generator=generator)
        epoch_loss = 0.0
        for start in range(0, len(train), batch_size):
            indices = permutation[start:start + batch_size]
            batch = train[indices]
            masked, mask = _masked_input(batch, mask_probability=mask_probability, generator=generator, torch=torch)
            optimizer.zero_grad()
            reconstruction = model(masked)
            loss = _masked_mse(reconstruction, batch, mask, torch=torch)
            if clean_loss_weight:
                loss = loss + clean_loss_weight * torch.nn.functional.mse_loss(model(batch), batch)
            loss.backward()
            optimizer.step()
            epoch_loss += float(loss.item()) * len(indices)
        last_loss = epoch_loss / len(train)
        if first_loss is None:
            first_loss = last_loss
    model.eval()
    with torch.no_grad():
        train_metrics = _reconstruction_metrics(model, train, mask_probability=mask_probability, generator=generator, torch=torch)
        test_metrics = _reconstruction_metrics(model, test, mask_probability=mask_probability, generator=generator, torch=torch)
        reconstruction = model(all_rows)
        row_mse = ((reconstruction - all_rows) ** 2).mean(dim=1).tolist()
        embeddings = model.encode(all_rows)
        train_embeddings = embeddings[:split_at]
        novelty = _embedding_novelty(embeddings, train_embeddings, train_count=split_at, torch=torch)
    train_mse = row_mse[:split_at]
    train_novelty = novelty[:split_at]
    score_rows = []
    for index, (time_utc, mse, novelty_value) in enumerate(zip(window_times, row_mse, novelty)):
        reconstruction_percentile = _empirical_percentile(float(mse), train_mse)
        novelty_percentile = _empirical_percentile(float(novelty_value), train_novelty)
        anomaly_score = (reconstruction_percentile + novelty_percentile) / 2
        score_rows.append(
            {
                "window_index": index,
                "window_end_utc": time_utc,
                "split": "train" if index < split_at else "test",
                "reconstruction_mse": round(float(mse), 8),
                "reconstruction_percentile": round(reconstruction_percentile, 8),
                "embedding_novelty": round(float(novelty_value), 8),
                "embedding_novelty_percentile": round(novelty_percentile, 8),
                "anomaly_score": round(anomaly_score, 8),
                "demo_probability": round(anomaly_score, 8),
                "demo_predicted_event": 1 if anomaly_score >= alert_threshold else 0,
            }
        )
    _write_anomaly_scores(scores_out, score_rows)
    latest = score_rows[-1]
    target_start = parse_utc(str(latest["window_end_utc"]))
    target_end = target_start + timedelta(days=forecast_horizon_days)
    forecast = {
        "schema": "elfquake.label_free_vlf_anomaly_forecast.v1",
        "status": "label_free_smoke_forecast",
        "window_end_utc": latest["window_end_utc"],
        "target_start_utc": format_utc(target_start),
        "target_end_utc": format_utc(target_end),
        "horizon_days": forecast_horizon_days,
        "demo_probability": latest["demo_probability"],
        "demo_predicted_event": latest["demo_predicted_event"],
        "alert_threshold": alert_threshold,
        "basis": "mean(reconstruction_error_percentile, embedding_novelty_percentile) from a real VLF self-supervised descriptor autoencoder",
        "warning": "Not trained on earthquake labels; use only as an end-to-end smoke forecast artifact.",
    }
    return {
        "train_window_count": split_at,
        "test_window_count": len(descriptors) - split_at,
        "first_train_loss": round(first_loss or 0.0, 8),
        "last_train_loss": round(last_loss, 8),
        "train_reconstruction": train_metrics,
        "test_reconstruction": test_metrics,
        "latest_window": latest,
        "forecast": forecast,
        "descriptor_standardization_mean": _rounded_vector(mean.tolist()),
        "descriptor_standardization_std": _rounded_vector(std.tolist()),
    }


def _embedding_novelty(embeddings, train_embeddings, *, train_count: int, torch: object) -> list[float]:
    distances = torch.cdist(embeddings, train_embeddings)
    if train_count > 1:
        for index in range(min(train_count, distances.shape[0], distances.shape[1])):
            distances[index, index] = float("inf")
    return distances.min(dim=1).values.tolist()


def _empirical_percentile(value: float, reference_values: list[float]) -> float:
    if not reference_values:
        return 0.0
    return sum(1 for item in reference_values if item <= value) / len(reference_values)


def _write_anomaly_scores(out_path: Path, rows: list[dict[str, object]]) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "window_index",
        "window_end_utc",
        "split",
        "reconstruction_mse",
        "reconstruction_percentile",
        "embedding_novelty",
        "embedding_novelty_percentile",
        "anomaly_score",
        "demo_probability",
        "demo_predicted_event",
    ]
    with out_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _train_mixed_alignment_model(
    *,
    torch: object,
    real_train,
    real_test,
    synthetic_train,
    mask_probability: float,
    clean_loss_weight: float,
    coral_weight: float,
    epochs: int,
    learning_rate: float,
    hidden_units: int,
    embedding_units: int,
    batch_size: int,
    seed: int,
) -> dict[str, object]:
    model = _SequenceAutoencoder(
        torch,
        input_units=int(real_train.shape[1]),
        hidden_units=hidden_units,
        embedding_units=embedding_units,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    generator = torch.Generator().manual_seed(seed)
    steps_per_epoch = max(1, math.ceil(max(len(real_train), len(synthetic_train)) / batch_size))
    first_loss = None
    last_loss = 0.0
    model.train()
    for _ in range(epochs):
        epoch_loss = 0.0
        for _step in range(steps_per_epoch):
            real_indices = torch.randint(len(real_train), (min(batch_size, len(real_train)),), generator=generator)
            synthetic_indices = torch.randint(len(synthetic_train), (min(batch_size, len(synthetic_train)),), generator=generator)
            real_batch = real_train[real_indices]
            synthetic_batch = synthetic_train[synthetic_indices]
            real_masked, real_mask = _masked_input(real_batch, mask_probability=mask_probability, generator=generator, torch=torch)
            synthetic_masked, synthetic_mask = _masked_input(synthetic_batch, mask_probability=mask_probability, generator=generator, torch=torch)
            optimizer.zero_grad()
            real_reconstruction = model(real_masked)
            synthetic_reconstruction = model(synthetic_masked)
            loss = (
                _masked_mse(real_reconstruction, real_batch, real_mask, torch=torch)
                + _masked_mse(synthetic_reconstruction, synthetic_batch, synthetic_mask, torch=torch)
            ) / 2
            if clean_loss_weight:
                loss = loss + clean_loss_weight * (
                    torch.nn.functional.mse_loss(model(real_batch), real_batch)
                    + torch.nn.functional.mse_loss(model(synthetic_batch), synthetic_batch)
                ) / 2
            if coral_weight:
                loss = loss + coral_weight * _coral_loss(model.encode(real_batch), model.encode(synthetic_batch), torch=torch)
            loss.backward()
            optimizer.step()
            epoch_loss += float(loss.item())
        last_loss = epoch_loss / steps_per_epoch
        if first_loss is None:
            first_loss = last_loss
    model.eval()
    with torch.no_grad():
        real_train_metrics = _reconstruction_metrics(model, real_train, mask_probability=mask_probability, generator=generator, torch=torch)
        real_test_metrics = _reconstruction_metrics(model, real_test, mask_probability=mask_probability, generator=generator, torch=torch)
        synthetic_metrics = _reconstruction_metrics(model, synthetic_train, mask_probability=mask_probability, generator=generator, torch=torch)
        real_train_embeddings = model.encode(real_train)
        real_test_embeddings = model.encode(real_test)
        synthetic_embeddings = model.encode(synthetic_train)
        alignment_loss = _coral_loss(real_train_embeddings, synthetic_embeddings, torch=torch)
    return {
        "first_train_loss": round(first_loss or 0.0, 8),
        "last_train_loss": round(last_loss, 8),
        "real_train_reconstruction": real_train_metrics,
        "real_test_reconstruction": real_test_metrics,
        "synthetic_train_reconstruction": synthetic_metrics,
        "embedding_comparison": _embedding_comparison(
            real_embeddings=real_test_embeddings.tolist(),
            synthetic_embeddings=synthetic_embeddings.tolist(),
        ),
        "train_embedding_comparison": _embedding_comparison(
            real_embeddings=real_train_embeddings.tolist(),
            synthetic_embeddings=synthetic_embeddings.tolist(),
        ),
        "coral_alignment_loss": round(float(alignment_loss.item()), 8),
        "real_train_embeddings": real_train_embeddings.tolist(),
        "real_test_embeddings": real_test_embeddings.tolist(),
        "synthetic_embeddings": synthetic_embeddings.tolist(),
    }


def _public_alignment_result(result: dict[str, object], *, synthetic_train_count: int) -> dict[str, object]:
    return {
        "synthetic_train_count": synthetic_train_count,
        "first_train_loss": result["first_train_loss"],
        "last_train_loss": result["last_train_loss"],
        "real_train_reconstruction": result["real_train_reconstruction"],
        "real_test_reconstruction": result["real_test_reconstruction"],
        "synthetic_train_reconstruction": result["synthetic_train_reconstruction"],
        "embedding_comparison": result["embedding_comparison"],
        "train_embedding_comparison": result["train_embedding_comparison"],
        "coral_alignment_loss": result["coral_alignment_loss"],
    }


def _select_synthetic_training_indices(
    *,
    torch: object,
    synthetic,
    real_train,
    synthetic_rows: list[tuple[str, int, list[float]]],
    method: str,
    fraction: float,
    max_count: int,
    balance_synthetic_sources: bool,
    seed: int,
) -> list[int]:
    requested = max(1, int(round(len(synthetic_rows) * fraction)))
    if max_count > 0:
        requested = min(requested, max_count)
    per_source_cap = math.ceil(requested / len({row[0] for row in synthetic_rows})) if balance_synthetic_sources else 0
    if method == "local":
        distances = torch.cdist(synthetic, real_train).min(dim=1).values.tolist()
        ranked = [index for index, _ in sorted(enumerate(distances), key=lambda item: item[1])]
    elif method == "centroid":
        centroid = real_train.mean(dim=0)
        distances = torch.linalg.vector_norm(synthetic - centroid, dim=1).tolist()
        ranked = [index for index, _ in sorted(enumerate(distances), key=lambda item: item[1])]
    elif method == "random":
        rng = random.Random(seed)
        ranked = list(range(len(synthetic_rows)))
        rng.shuffle(ranked)
    elif method == "full":
        ranked = list(range(len(synthetic_rows)))
    else:
        raise ValueError(f"unknown synthetic selection method: {method}")
    return _apply_selection_cap(
        ranked,
        synthetic_rows=synthetic_rows,
        requested=requested,
        per_source_cap=per_source_cap,
    )


def _apply_selection_cap(
    ranked: list[int],
    *,
    synthetic_rows: list[tuple[str, int, list[float]]],
    requested: int,
    per_source_cap: int,
) -> list[int]:
    selected = []
    counts: dict[str, int] = {}
    for index in ranked:
        dataset_id = synthetic_rows[index][0]
        if per_source_cap and counts.get(dataset_id, 0) >= per_source_cap:
            continue
        selected.append(index)
        counts[dataset_id] = counts.get(dataset_id, 0) + 1
        if len(selected) >= requested:
            break
    if len(selected) < requested and per_source_cap:
        selected_set = set(selected)
        for index in ranked:
            if index in selected_set:
                continue
            selected.append(index)
            if len(selected) >= requested:
                break
    return selected


def _selection_counts(indices: list[int], synthetic_rows: list[tuple[str, int, list[float]]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for index in indices:
        dataset_id = synthetic_rows[index][0]
        counts[dataset_id] = counts.get(dataset_id, 0) + 1
    return counts


def _coral_loss(real_embeddings, synthetic_embeddings, *, torch: object):
    if len(real_embeddings) < 2 or len(synthetic_embeddings) < 2:
        return torch.tensor(0.0)
    real_mean = real_embeddings.mean(dim=0)
    synthetic_mean = synthetic_embeddings.mean(dim=0)
    real_centered = real_embeddings - real_mean
    synthetic_centered = synthetic_embeddings - synthetic_mean
    real_cov = real_centered.T.matmul(real_centered) / (len(real_embeddings) - 1)
    synthetic_cov = synthetic_centered.T.matmul(synthetic_centered) / (len(synthetic_embeddings) - 1)
    mean_loss = torch.mean((real_mean - synthetic_mean) ** 2)
    covariance_loss = torch.mean((real_cov - synthetic_cov) ** 2)
    return mean_loss + covariance_loss


def _descriptor_gap(
    *,
    real_rows: list[list[float]],
    synthetic_rows: list[list[float]],
    descriptor_names: list[str],
) -> dict[str, object]:
    real_centroid = _centroid(real_rows)
    synthetic_centroid = _centroid(synthetic_rows)
    real_std = _column_std(real_rows, real_centroid)
    synthetic_std = _column_std(synthetic_rows, synthetic_centroid)
    rows = []
    for index, name in enumerate(descriptor_names):
        rows.append(
            {
                "descriptor": name,
                "real_mean": round(real_centroid[index], 8),
                "synthetic_mean": round(synthetic_centroid[index], 8),
                "mean_delta": round(synthetic_centroid[index] - real_centroid[index], 8),
                "real_std": round(real_std[index], 8),
                "synthetic_std": round(synthetic_std[index], 8),
                "std_delta": round(synthetic_std[index] - real_std[index], 8),
            }
        )
    ranked = sorted(rows, key=lambda row: abs(float(row["mean_delta"])) + abs(float(row["std_delta"])), reverse=True)
    return {
        "mean_absolute_mean_delta": round(sum(abs(float(row["mean_delta"])) for row in rows) / len(rows), 8),
        "mean_absolute_std_delta": round(sum(abs(float(row["std_delta"])) for row in rows) / len(rows), 8),
        "largest_descriptor_gaps": ranked[:5],
    }


class _SequenceAutoencoder:
    def __init__(self, torch: object, *, input_units: int, hidden_units: int, embedding_units: int) -> None:
        self.torch = torch
        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(input_units, hidden_units),
            torch.nn.GELU(),
            torch.nn.Linear(hidden_units, embedding_units),
        )
        self.decoder = torch.nn.Sequential(
            torch.nn.GELU(),
            torch.nn.Linear(embedding_units, hidden_units),
            torch.nn.GELU(),
            torch.nn.Linear(hidden_units, input_units),
        )

    def parameters(self):
        return [*self.encoder.parameters(), *self.decoder.parameters()]

    def state_dict(self) -> dict[str, object]:
        return {"encoder": self.encoder.state_dict(), "decoder": self.decoder.state_dict()}

    def train(self) -> None:
        self.encoder.train()
        self.decoder.train()

    def eval(self) -> None:
        self.encoder.eval()
        self.decoder.eval()

    def encode(self, x):
        return self.encoder(x)

    def __call__(self, x):
        return self.decoder(self.encoder(x))


def _standardize(train_tensor, test_tensor, all_tensor, *, torch: object):
    feature_count = train_tensor.shape[2]
    train_flat = train_tensor.reshape(-1, feature_count)
    mean = train_flat.mean(dim=0)
    std = train_flat.std(dim=0, unbiased=False)
    std = torch.where(std < 1e-6, torch.ones_like(std), std)
    return (
        (train_tensor - mean) / std,
        (test_tensor - mean) / std,
        (all_tensor - mean) / std,
        mean,
        std,
    )


def _standardize_vectors(train_tensor, test_tensor, synthetic_tensor, *, torch: object):
    mean = train_tensor.mean(dim=0)
    std = train_tensor.std(dim=0, unbiased=False)
    std = torch.where(std < 1e-6, torch.ones_like(std), std)
    return (
        (train_tensor - mean) / std,
        (test_tensor - mean) / std,
        (synthetic_tensor - mean) / std,
        mean,
        std,
    )


def _masked_input(batch, *, mask_probability: float, generator: object, torch: object):
    if mask_probability == 0:
        mask = torch.ones_like(batch, dtype=torch.bool)
        return batch, mask
    mask = torch.rand(batch.shape, generator=generator) < mask_probability
    if not bool(mask.any()):
        mask = torch.ones_like(batch, dtype=torch.bool)
    return batch.masked_fill(mask, 0.0), mask


def _masked_mse(reconstruction, target, mask, *, torch: object):
    squared = (reconstruction - target) ** 2
    if bool(mask.any()):
        return squared[mask].mean()
    return torch.nn.functional.mse_loss(reconstruction, target)


def _reconstruction_metrics(model: _SequenceAutoencoder, data, *, mask_probability: float, generator: object, torch: object) -> dict[str, float]:
    masked, mask = _masked_input(data, mask_probability=mask_probability, generator=generator, torch=torch)
    reconstruction = model(masked)
    baseline = torch.zeros_like(data)
    return {
        "masked_mse": round(float(_masked_mse(reconstruction, data, mask, torch=torch).item()), 8),
        "full_mse": round(float(torch.nn.functional.mse_loss(reconstruction, data).item()), 8),
        "zero_baseline_masked_mse": round(float(_masked_mse(baseline, data, mask, torch=torch).item()), 8),
        "zero_baseline_full_mse": round(float(torch.nn.functional.mse_loss(baseline, data).item()), 8),
    }


def _save_checkpoint(
    model: _SequenceAutoencoder,
    *,
    checkpoint_out: Path,
    torch: object,
    dataset: SequenceDataset,
    lookback_steps: int,
    hidden_units: int,
    embedding_units: int,
    mean,
    std,
) -> None:
    checkpoint_out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "schema": "elfquake.sequence_autoencoder_checkpoint.v1",
            "model_state": model.state_dict(),
            "dataset_id": dataset.dataset_id,
            "modality": dataset.modality,
            "feature_names": dataset.feature_names,
            "lookback_steps": lookback_steps,
            "hidden_units": hidden_units,
            "embedding_units": embedding_units,
            "mean": mean.tolist(),
            "std": std.tolist(),
        },
        checkpoint_out,
    )


def _write_embeddings(
    out_path: Path,
    *,
    embeddings: list[list[float]],
    train_window_count: int,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["window_index", "split"] + [f"embedding_{index}" for index in range(len(embeddings[0]) if embeddings else 0)]
    with out_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for index, row in enumerate(embeddings):
            writer.writerow(
                {
                    "window_index": index,
                    "split": "train" if index < train_window_count else "test",
                    **{f"embedding_{item_index}": f"{value:.9f}" for item_index, value in enumerate(row)},
                }
            )


def _write_domain_embeddings(
    out_path: Path,
    *,
    real_train_embeddings: list[list[float]],
    real_test_embeddings: list[list[float]],
    synthetic_embeddings: list[list[float]],
    synthetic_rows: list[tuple[str, int, list[float]]],
    synthetic_inlier_indices: set[int],
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    embedding_count = len(real_train_embeddings[0]) if real_train_embeddings else len(synthetic_embeddings[0]) if synthetic_embeddings else 0
    fieldnames = ["source", "dataset_id", "window_index", "split", "is_synthetic_inlier"] + [f"embedding_{index}" for index in range(embedding_count)]
    with out_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for index, row in enumerate(real_train_embeddings):
            writer.writerow(_embedding_row("real", "real_train", index, "train", row, is_synthetic_inlier=False))
        for index, row in enumerate(real_test_embeddings):
            writer.writerow(_embedding_row("real", "real_test", index, "test", row, is_synthetic_inlier=False))
        for index, (embedding, (dataset_id, window_index, _)) in enumerate(zip(synthetic_embeddings, synthetic_rows)):
            writer.writerow(_embedding_row("synthetic", dataset_id, window_index, "synthetic", embedding, is_synthetic_inlier=index in synthetic_inlier_indices))


def _embedding_row(source: str, dataset_id: str, window_index: int, split: str, embedding: list[float], *, is_synthetic_inlier: bool) -> dict[str, str | int]:
    return {
        "source": source,
        "dataset_id": dataset_id,
        "window_index": window_index,
        "split": split,
        "is_synthetic_inlier": "1" if is_synthetic_inlier else "0",
        **{f"embedding_{index}": f"{value:.9f}" for index, value in enumerate(embedding)},
    }


def _synthetic_inlier_indices(
    *,
    synthetic_descriptors: list[list[float]],
    real_descriptors: list[list[float]],
    fraction: float,
) -> list[int]:
    if not synthetic_descriptors or not real_descriptors:
        return []
    fraction = min(1.0, max(0.0, fraction))
    if fraction == 0:
        return []
    real_centroid = _centroid(real_descriptors)
    distances = [
        (index, _euclidean(descriptor, real_centroid))
        for index, descriptor in enumerate(synthetic_descriptors)
    ]
    keep_count = max(1, int(round(len(distances) * fraction)))
    return [index for index, _ in sorted(distances, key=lambda item: item[1])[:keep_count]]


def _embedding_comparison(
    *,
    real_embeddings: list[list[float]],
    synthetic_embeddings: list[list[float]],
) -> dict[str, object]:
    if not real_embeddings or not synthetic_embeddings:
        return {"status": "insufficient_embeddings"}
    real_centroid = _centroid(real_embeddings)
    synthetic_centroid = _centroid(synthetic_embeddings)
    real_std = _column_std(real_embeddings, real_centroid)
    synthetic_std = _column_std(synthetic_embeddings, synthetic_centroid)
    return {
        "status": "evaluated",
        "real_count": len(real_embeddings),
        "synthetic_count": len(synthetic_embeddings),
        "centroid_euclidean_distance": round(_euclidean(real_centroid, synthetic_centroid), 8),
        "scale_mean_absolute_delta": round(sum(abs(a - b) for a, b in zip(real_std, synthetic_std)) / len(real_std), 8),
        "synthetic_to_real_nearest_mean_distance": round(_nearest_mean_distance(synthetic_embeddings, real_embeddings), 8),
        "real_centroid": _rounded_vector(real_centroid),
        "synthetic_centroid": _rounded_vector(synthetic_centroid),
        "real_std": _rounded_vector(real_std),
        "synthetic_std": _rounded_vector(synthetic_std),
    }


def _centroid(rows: list[list[float]]) -> list[float]:
    width = len(rows[0])
    return [sum(row[index] for row in rows) / len(rows) for index in range(width)]


def _column_std(rows: list[list[float]], means: list[float]) -> list[float]:
    return [
        math.sqrt(sum((row[index] - means[index]) ** 2 for row in rows) / len(rows))
        for index in range(len(means))
    ]


def _euclidean(left: list[float], right: list[float]) -> float:
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(left, right)))


def _nearest_mean_distance(rows: list[list[float]], reference_rows: list[list[float]]) -> float:
    distances = [
        min(_euclidean(row, reference) for reference in reference_rows)
        for row in rows
    ]
    return sum(distances) / len(distances)


def _rounded_vector(values: list[float]) -> list[float]:
    return [round(float(value), 8) for value in values]


def _import_torch() -> object:
    try:
        import torch
    except ImportError as error:
        raise ValueError("PyTorch is required for self-supervised sequence pretraining") from error
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

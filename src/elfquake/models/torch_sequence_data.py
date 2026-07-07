"""Load materialized sequence manifests for PyTorch sequence models."""

from __future__ import annotations

import csv
import json
import re
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class SequenceDataset:
    dataset_id: str
    modality: str
    time_to_index: dict[str, int]
    values: list[list[float]]
    feature_names: list[str]


def load_sequence_datasets(paths: list[Path], *, include_missing_masks: bool) -> dict[tuple[str, str], SequenceDataset]:
    datasets = {}
    for path in paths:
        manifest = json.loads(path.read_text(encoding="utf-8"))
        dataset = _load_one_sequence(path, manifest, include_missing_masks=include_missing_masks)
        datasets[(dataset.dataset_id, dataset.modality)] = dataset
    return datasets


def build_sequence_samples(
    rows: list[dict[str, str]],
    sequences: dict[tuple[str, str], SequenceDataset],
    *,
    modalities: tuple[str, ...],
    lookback_steps: int,
) -> tuple[list[list[list[float]]], list[str]]:
    samples = []
    feature_names: list[str] = []
    for row in rows:
        parts = []
        current_feature_names = []
        for modality in modalities:
            dataset = _required_dataset(sequences, dataset_id=row.get("dataset_id", ""), modality=modality)
            end_time = row.get("window_end_utc", "")
            end_index = dataset.time_to_index.get(end_time)
            if end_index is None:
                raise ValueError(f"missing sequence time {end_time} for {dataset.dataset_id}/{modality}")
            parts.append(_slice_lookback(dataset.values, end_index=end_index, lookback_steps=lookback_steps))
            current_feature_names.extend(dataset.feature_names)
        samples.append([[value for part in step_parts for value in part] for step_parts in zip(*parts)])
        feature_names = current_feature_names
    return samples, feature_names


def _load_one_sequence(path: Path, manifest: dict[str, object], *, include_missing_masks: bool) -> SequenceDataset:
    channel_fields = [str(item) for item in manifest["channel_fields"]]
    time_count = int(manifest["time_count"])
    value_sums = [[0.0 for _ in channel_fields] for _ in range(time_count)]
    mask_sums = [[0.0 for _ in channel_fields] for _ in range(time_count)]
    counts = [0 for _ in range(time_count)]
    with Path(str(manifest["values_csv"])).open(newline="", encoding="utf-8") as value_handle, Path(str(manifest["masks_csv"])).open(newline="", encoding="utf-8") as mask_handle:
        for value_row, mask_row in zip(csv.DictReader(value_handle), csv.DictReader(mask_handle)):
            time_index = int(value_row["time_index"])
            counts[time_index] += 1
            for index, field in enumerate(channel_fields):
                value_sums[time_index][index] += float(value_row[field])
                mask_sums[time_index][index] += float(mask_row.get(f"{field}__present", "0") or "0")
    modality = str(manifest["modality"])
    return SequenceDataset(
        dataset_id=_infer_dataset_id(path, manifest),
        modality=modality,
        time_to_index=_read_time_axis(Path(str(manifest["time_axis_csv"]))),
        values=_averaged_values(value_sums, mask_sums, counts, include_missing_masks=include_missing_masks),
        feature_names=_feature_names(modality, channel_fields, include_missing_masks=include_missing_masks),
    )


def _averaged_values(
    value_sums: list[list[float]],
    mask_sums: list[list[float]],
    counts: list[int],
    *,
    include_missing_masks: bool,
) -> list[list[float]]:
    values = []
    for time_index, row in enumerate(value_sums):
        divisor = counts[time_index] or 1
        features = [value / divisor for value in row]
        if include_missing_masks:
            features += [value / divisor for value in mask_sums[time_index]]
        values.append(features)
    return values


def _feature_names(modality: str, channel_fields: list[str], *, include_missing_masks: bool) -> list[str]:
    names = [f"{modality}_{field}" for field in channel_fields]
    if include_missing_masks:
        names += [f"{modality}_{field}__present" for field in channel_fields]
    return names


def _required_dataset(
    sequences: dict[tuple[str, str], SequenceDataset],
    *,
    dataset_id: str,
    modality: str,
) -> SequenceDataset:
    dataset = sequences.get((dataset_id, modality))
    if dataset is None:
        raise ValueError(f"missing sequence dataset for {dataset_id}/{modality}")
    return dataset


def _slice_lookback(values: list[list[float]], *, end_index: int, lookback_steps: int) -> list[list[float]]:
    feature_count = len(values[0]) if values else 0
    window = values[max(0, end_index - lookback_steps):end_index]
    return [[0.0 for _ in range(feature_count)] for _ in range(lookback_steps - len(window))] + window


def _infer_dataset_id(path: Path, manifest: dict[str, object]) -> str:
    match = re.search(r"seed(\d+)", f"{path} {manifest.get('input_csv', '')}")
    return f"seed{match.group(1)}" if match else path.parent.name


def _read_time_axis(path: Path) -> dict[str, int]:
    with path.open(newline="", encoding="utf-8") as handle:
        return {row["time_utc"]: int(row["time_index"]) for row in csv.DictReader(handle)}

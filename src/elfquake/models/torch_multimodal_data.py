"""Cadence-aware sequence loading for multimodal PyTorch models."""

from __future__ import annotations

import csv
import json
import math
import re
from bisect import bisect_right
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


VLF_SIGNAL_EXCLUDES = {
    "vlf_image_width_px",
    "vlf_image_height_px",
    "vlf_crop_left_px",
    "vlf_crop_top_px",
    "vlf_crop_right_px",
    "vlf_crop_bottom_px",
    "vlf_crop_width_px",
    "vlf_crop_height_px",
    "vlf_pixel_count",
}
PIEZO_SPATIAL_AGGREGATES = {
    "piezo_signal": ("max", "std"),
    "nearest_critical_distance": ("min", "std"),
    "damage_local_mean": ("max", "std"),
    "damage_local_max": ("max", "std"),
    "damage_local_active_fraction": ("max", "std"),
    "damage_local_std": ("max", "std"),
}


@dataclass(frozen=True)
class ModalitySequence:
    dataset_id: str
    modality: str
    feature_names: tuple[str, ...]
    values: list[list[float]]
    observed: list[list[float]]
    elapsed_minutes: list[float]
    times: tuple[str, ...]
    time_to_index: dict[str, int]


@dataclass(frozen=True)
class WindowRef:
    dataset_id: str
    end_index: int


@dataclass(frozen=True)
class Normalization:
    mean: tuple[float, ...]
    std: tuple[float, ...]


def load_modality_sequences(
    paths: list[Path],
    *,
    entity_aggregation_profile: str = "mean",
    exclude_fields: set[str] | None = None,
) -> dict[tuple[str, str], ModalitySequence]:
    if entity_aggregation_profile not in {"mean", "piezo_spatial"}:
        raise ValueError(f"unsupported entity aggregation profile: {entity_aggregation_profile}")
    sequences: dict[tuple[str, str], ModalitySequence] = {}
    for path in paths:
        manifest = json.loads(path.read_text(encoding="utf-8"))
        sequence = _load_sequence(
            path,
            manifest,
            entity_aggregation_profile=entity_aggregation_profile,
            exclude_fields=exclude_fields or set(),
        )
        key = (sequence.dataset_id, sequence.modality)
        if key in sequences:
            raise ValueError(f"duplicate sequence dataset: {key[0]}/{key[1]}")
        sequences[key] = sequence
    _validate_modality_shapes(sequences)
    return sequences


def chronological_window_refs(
    sequences: dict[tuple[str, str], ModalitySequence],
    *,
    modality: str,
    lookback_steps: int,
    train_fraction: float,
    stride: int = 1,
) -> tuple[list[WindowRef], list[WindowRef]]:
    train: list[WindowRef] = []
    test: list[WindowRef] = []
    dataset_ids = sorted({dataset_id for dataset_id, item_modality in sequences if item_modality == modality})
    for dataset_id in dataset_ids:
        sequence = sequences[(dataset_id, modality)]
        split_at = max(lookback_steps, min(len(sequence.values) - lookback_steps, int(len(sequence.values) * train_fraction)))
        train.extend(WindowRef(dataset_id, end) for end in range(lookback_steps, split_at + 1, stride))
        test.extend(WindowRef(dataset_id, end) for end in range(split_at + lookback_steps, len(sequence.values) + 1, stride))
    return train, test


def fit_normalizations(
    sequences: dict[tuple[str, str], ModalitySequence],
    *,
    train_fraction: float,
) -> dict[str, Normalization]:
    values: dict[str, list[list[float]]] = {}
    masks: dict[str, list[list[float]]] = {}
    for sequence in sequences.values():
        stop = max(1, int(len(sequence.values) * train_fraction))
        values.setdefault(sequence.modality, []).extend(sequence.values[:stop])
        masks.setdefault(sequence.modality, []).extend(sequence.observed[:stop])
    result = {}
    for modality, rows in values.items():
        feature_count = len(rows[0]) if rows else 0
        means = []
        stds = []
        for feature_index in range(feature_count):
            present = [
                row[feature_index]
                for row, mask in zip(rows, masks[modality])
                if mask[feature_index] > 0
            ]
            mean = sum(present) / len(present) if present else 0.0
            variance = sum((value - mean) ** 2 for value in present) / len(present) if present else 0.0
            means.append(mean)
            stds.append(math.sqrt(variance) if variance >= 1e-12 else 1.0)
        result[modality] = Normalization(tuple(means), tuple(stds))
    return result


def build_window_batch(
    refs: list[WindowRef],
    sequences: dict[tuple[str, str], ModalitySequence],
    *,
    modalities: tuple[str, ...],
    lookback_steps: int,
    normalizations: dict[str, Normalization],
    torch: object,
) -> tuple[dict[str, object], dict[str, object], dict[str, object]]:
    inputs: dict[str, object] = {}
    targets: dict[str, object] = {}
    observed: dict[str, object] = {}
    for modality in modalities:
        input_rows = []
        target_rows = []
        observed_rows = []
        normalization = normalizations[modality]
        for ref in refs:
            sequence = sequences[(ref.dataset_id, modality)]
            start = ref.end_index - lookback_steps
            values = sequence.values[start:ref.end_index]
            masks = sequence.observed[start:ref.end_index]
            elapsed = sequence.elapsed_minutes[start:ref.end_index]
            normalized = [
                [
                    ((value - normalization.mean[index]) / normalization.std[index]) if mask[index] > 0 else 0.0
                    for index, value in enumerate(row)
                ]
                for row, mask in zip(values, masks)
            ]
            input_rows.append([
                value_row + mask_row + [math.log1p(max(0.0, delta))]
                for value_row, mask_row, delta in zip(normalized, masks, elapsed)
            ])
            target_rows.append(normalized)
            observed_rows.append(masks)
        inputs[modality] = torch.tensor(input_rows, dtype=torch.float32)
        targets[modality] = torch.tensor(target_rows, dtype=torch.float32)
        observed[modality] = torch.tensor(observed_rows, dtype=torch.bool)
    return inputs, targets, observed


def refs_for_rows(
    rows: list[dict[str, str]],
    sequences: dict[tuple[str, str], ModalitySequence],
    *,
    modalities: tuple[str, ...],
    lookback_steps: int = 1,
    time_field: str = "window_end_utc",
) -> tuple[list[WindowRef], list[dict[str, str]]]:
    refs = []
    covered_rows = []
    for row in rows:
        dataset_id = row.get("dataset_id", "")
        end_indices = []
        for modality in modalities:
            sequence = sequences.get((dataset_id, modality))
            if sequence is None:
                break
            index = sequence_index_at_or_before(sequence, row.get(time_field, ""))
            if index is None:
                break
            end_indices.append(index)
        if len(end_indices) != len(modalities) or min(end_indices) < lookback_steps:
            continue
        refs.append(WindowRef(dataset_id, min(end_indices)))
        covered_rows.append(row)
    return refs, covered_rows


def sequence_index_at_or_before(sequence: ModalitySequence, time_utc: str) -> int | None:
    if time_utc in sequence.time_to_index:
        return sequence.time_to_index[time_utc]
    index = bisect_right(sequence.times, time_utc) - 1
    return index if index >= 0 else None


def modality_input_sizes(sequences: dict[tuple[str, str], ModalitySequence]) -> dict[str, int]:
    result = {}
    for sequence in sequences.values():
        result[sequence.modality] = len(sequence.feature_names) * 2 + 1
    return result


def modality_target_sizes(sequences: dict[tuple[str, str], ModalitySequence]) -> dict[str, int]:
    result = {}
    for sequence in sequences.values():
        result[sequence.modality] = len(sequence.feature_names)
    return result


def _load_sequence(
    path: Path,
    manifest: dict[str, object],
    *,
    entity_aggregation_profile: str,
    exclude_fields: set[str],
) -> ModalitySequence:
    all_fields = [str(field) for field in manifest["channel_fields"]]
    modality = str(manifest["modality"])
    fields = [
        field for field in all_fields
        if field not in exclude_fields and not (modality == "real_vlf_image" and field in VLF_SIGNAL_EXCLUDES)
    ]
    if not fields:
        raise ValueError(f"all fields excluded for modality {modality}")
    field_indices = [all_fields.index(field) for field in fields]
    time_count = int(manifest["time_count"])
    sums = [[0.0 for _ in all_fields] for _ in range(time_count)]
    square_sums = [[0.0 for _ in all_fields] for _ in range(time_count)]
    minima = [[math.inf for _ in all_fields] for _ in range(time_count)]
    maxima = [[-math.inf for _ in all_fields] for _ in range(time_count)]
    mask_sums = [[0.0 for _ in all_fields] for _ in range(time_count)]
    entity_counts = [0 for _ in range(time_count)]
    with Path(str(manifest["values_csv"])).open(newline="", encoding="utf-8") as value_handle, Path(str(manifest["masks_csv"])).open(newline="", encoding="utf-8") as mask_handle:
        for value_row, mask_row in zip(csv.DictReader(value_handle), csv.DictReader(mask_handle)):
            time_index = int(value_row["time_index"])
            entity_counts[time_index] += 1
            for index, field in enumerate(all_fields):
                present = float(mask_row.get(f"{field}__present", "0") or 0)
                mask_sums[time_index][index] += present
                if present:
                    value = float(value_row.get(field, "0") or 0)
                    sums[time_index][index] += value
                    square_sums[time_index][index] += value * value
                    minima[time_index][index] = min(minima[time_index][index], value)
                    maxima[time_index][index] = max(maxima[time_index][index], value)
    values = []
    observed = []
    aggregation_specs = _aggregation_specs(
        modality,
        fields,
        field_indices,
        entity_count=int(manifest.get("entity_count", 1)),
        profile=entity_aggregation_profile,
    )
    for time_index in range(time_count):
        divisor = entity_counts[time_index] or 1
        row_values = []
        row_observed = []
        for source_index, aggregate in aggregation_specs:
            present_count = mask_sums[time_index][source_index]
            row_values.append(_aggregate_value(
                aggregate,
                total=sums[time_index][source_index],
                square_total=square_sums[time_index][source_index],
                minimum=minima[time_index][source_index],
                maximum=maxima[time_index][source_index],
                count=present_count,
                mean_divisor=divisor,
            ))
            row_observed.append(min(1.0, present_count / divisor))
        values.append(row_values)
        observed.append(row_observed)
    times, time_to_index = _read_times(Path(str(manifest["time_axis_csv"])), time_field=str(manifest.get("time_field", "")))
    return ModalitySequence(
        dataset_id=_infer_dataset_id(path, manifest),
        modality=modality,
        feature_names=tuple(_aggregation_feature_names(
            modality,
            fields,
            int(manifest.get("entity_count", 1)),
            profile=entity_aggregation_profile,
        )),
        values=values,
        observed=observed,
        elapsed_minutes=_elapsed_minutes(times),
        times=tuple(times),
        time_to_index=time_to_index,
    )


def _aggregation_specs(
    modality: str,
    fields: list[str],
    field_indices: list[int],
    *,
    entity_count: int,
    profile: str,
) -> list[tuple[int, str]]:
    specs = []
    for source_index, field in zip(field_indices, fields):
        specs.append((source_index, "mean"))
        if profile == "piezo_spatial" and modality == "synthetic_piezo_vlf" and entity_count > 1:
            specs.extend((source_index, aggregate) for aggregate in PIEZO_SPATIAL_AGGREGATES.get(field, ()))
    return specs


def _aggregation_feature_names(
    modality: str,
    fields: list[str],
    entity_count: int,
    *,
    profile: str,
) -> list[str]:
    names = []
    for field in fields:
        names.append(field)
        if profile == "piezo_spatial" and modality == "synthetic_piezo_vlf" and entity_count > 1:
            names.extend(f"{field}__entity_{aggregate}" for aggregate in PIEZO_SPATIAL_AGGREGATES.get(field, ()))
    return names


def _aggregate_value(
    aggregate: str,
    *,
    total: float,
    square_total: float,
    minimum: float,
    maximum: float,
    count: float,
    mean_divisor: int,
) -> float:
    if count <= 0:
        return 0.0
    if aggregate == "mean":
        return total / mean_divisor
    if aggregate == "min":
        return minimum
    if aggregate == "max":
        return maximum
    if aggregate == "std":
        mean = total / count
        return math.sqrt(max(0.0, square_total / count - mean * mean))
    raise ValueError(f"unsupported entity aggregate: {aggregate}")


def _read_times(path: Path, *, time_field: str) -> tuple[list[str], dict[str, int]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fields = list(reader.fieldnames or [])
        axis_field = "time_utc" if "time_utc" in fields else time_field if time_field in fields else next((field for field in fields if field.endswith("_utc")), "")
        if not axis_field:
            raise ValueError(f"time axis has no UTC field: {path}")
        rows = list(reader)
    pairs = sorted((int(row["time_index"]), row[axis_field]) for row in rows)
    times = [time for _, time in pairs]
    return times, {time: index for index, time in pairs}


def _elapsed_minutes(times: list[str]) -> list[float]:
    parsed = [_parse_utc(value) for value in times]
    if not parsed:
        return []
    positive = [max(0.0, (right - left).total_seconds() / 60.0) for left, right in zip(parsed, parsed[1:])]
    default = sorted(positive)[len(positive) // 2] if positive else 0.0
    return [default] + positive


def _parse_utc(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def _infer_dataset_id(path: Path, manifest: dict[str, object]) -> str:
    match = re.search(r"seed(\d+)", f"{path} {manifest.get('input_csv', '')}")
    return f"seed{match.group(1)}" if match else path.parent.name.removesuffix("_sequence")


def _validate_modality_shapes(sequences: dict[tuple[str, str], ModalitySequence]) -> None:
    shapes: dict[str, tuple[str, ...]] = {}
    for sequence in sequences.values():
        expected = shapes.setdefault(sequence.modality, sequence.feature_names)
        if expected != sequence.feature_names:
            raise ValueError(f"inconsistent features for modality {sequence.modality}")

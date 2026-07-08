"""Deterministic transforms for simulated piezo/VLF analogue CSVs."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path


def transform_piezo_signal_csv(
    *,
    input_csv: Path,
    out_csv: Path,
    report_path: Path | None = None,
    signal_field: str = "piezo_signal",
    highpass_decay: float = 0.9,
    envelope_decay: float = 0.15,
    envelope_mix: float = 0.25,
    burst_power: float = 1.35,
    near_threshold_weight: float = 1.0,
    near_threshold_floor: float = 0.75,
    release_mix: float = 0.0,
    gain_contrast: float = 0.0,
) -> dict[str, object]:
    if not 0 <= highpass_decay <= 1:
        raise ValueError("highpass_decay must be between 0 and 1")
    if not 0 <= envelope_decay <= 1:
        raise ValueError("envelope_decay must be between 0 and 1")
    if envelope_mix < 0:
        raise ValueError("envelope_mix must be non-negative")
    if burst_power <= 0:
        raise ValueError("burst_power must be positive")
    if near_threshold_weight < 0:
        raise ValueError("near_threshold_weight must be non-negative")
    if not 0 <= near_threshold_floor < 1:
        raise ValueError("near_threshold_floor must be in [0, 1)")
    if release_mix < 0:
        raise ValueError("release_mix must be non-negative")
    if gain_contrast < 0:
        raise ValueError("gain_contrast must be non-negative")

    rows = _read_rows(input_csv)
    if not rows:
        raise ValueError(f"empty piezo CSV: {input_csv}")
    fieldnames = list(rows[0].keys())
    if signal_field not in fieldnames:
        raise ValueError(f"missing signal field {signal_field}")
    by_sensor = _group_indices_by_sensor(rows)
    transformed_by_index: dict[int, float] = {}
    for sensor_id, indices in by_sensor.items():
        raw_values = [
            _row_float(rows[index], signal_field)
            * _near_threshold_multiplier(
                rows[index],
                weight=near_threshold_weight,
                floor=near_threshold_floor,
            )
            + release_mix * _row_float(rows[index], "piezo_release_total")
            for index in indices
        ]
        highpassed = _highpass(raw_values, decay=highpass_decay)
        envelope = _positive_envelope(highpassed, decay=envelope_decay)
        shaped = [
            value + envelope_mix * envelope[index]
            for index, value in enumerate(highpassed)
        ]
        scaled = _burst_shape(shaped, power=burst_power)
        gain = _sensor_gain(sensor_id, contrast=gain_contrast)
        for index, value in zip(indices, scaled):
            transformed_by_index[index] = value * gain

    out_rows = []
    for index, row in enumerate(rows):
        out_row = dict(row)
        out_row[signal_field] = f"{transformed_by_index[index]:.9f}"
        out_rows.append(out_row)

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(out_rows)

    before = [_row_float(row, signal_field) for row in rows]
    after = [transformed_by_index[index] for index in range(len(rows))]
    report: dict[str, object] = {
        "schema": "elfquake.piezo_signal_transform.v1",
        "input": str(input_csv),
        "output": str(out_csv),
        "row_count": len(rows),
        "sensor_count": len(by_sensor),
        "signal_field": signal_field,
        "highpass_decay": highpass_decay,
        "envelope_decay": envelope_decay,
        "envelope_mix": envelope_mix,
        "burst_power": burst_power,
        "near_threshold_weight": near_threshold_weight,
        "near_threshold_floor": near_threshold_floor,
        "release_mix": release_mix,
        "gain_contrast": gain_contrast,
        "before": _series_stats(before),
        "after": _series_stats(after),
    }
    if report_path:
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _group_indices_by_sensor(rows: list[dict[str, str]]) -> dict[str, list[int]]:
    grouped: dict[str, list[int]] = {}
    for index, row in enumerate(rows):
        grouped.setdefault(row.get("sensor_id", ""), []).append(index)
    return grouped


def _row_float(row: dict[str, str], field: str) -> float:
    try:
        return float(row.get(field, "") or 0.0)
    except ValueError:
        return 0.0


def _near_threshold_multiplier(row: dict[str, str], *, weight: float, floor: float) -> float:
    stress = _row_float(row, "max_stress_ratio")
    if stress <= floor:
        return 1.0
    return 1.0 + weight * min(1.0, (stress - floor) / (1.0 - floor))


def _highpass(values: list[float], *, decay: float) -> list[float]:
    previous_input = 0.0
    previous_output = 0.0
    output = []
    for value in values:
        highpassed = value - previous_input + decay * previous_output
        output.append(highpassed)
        previous_input = value
        previous_output = highpassed
    return output


def _positive_envelope(values: list[float], *, decay: float) -> list[float]:
    envelope = []
    previous = 0.0
    for value in values:
        current = max(0.0, value) + decay * previous
        envelope.append(current)
        previous = current
    return envelope


def _burst_shape(values: list[float], *, power: float) -> list[float]:
    scale = _quantile([abs(value) for value in values], 0.90) or 1.0
    shaped = []
    for value in values:
        normalized = value / scale
        shaped.append(math.copysign(abs(normalized) ** power, normalized) * scale)
    return shaped


def _sensor_gain(sensor_id: str, *, contrast: float) -> float:
    if contrast == 0:
        return 1.0
    try:
        sensor_index = int(sensor_id)
    except ValueError:
        sensor_index = sum(ord(char) for char in sensor_id)
    phase = math.sin((sensor_index + 1) * 12.9898) * 43758.5453
    unit = phase - math.floor(phase)
    return max(0.0, 1.0 + contrast * (2.0 * unit - 1.0))


def _quantile(values: list[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round((len(ordered) - 1) * fraction)))
    return ordered[index]


def _series_stats(values: list[float]) -> dict[str, float]:
    mean = sum(values) / len(values) if values else 0.0
    variance = sum((value - mean) ** 2 for value in values) / len(values) if values else 0.0
    nonzero = sum(1 for value in values if abs(value) > 1e-12)
    return {
        "mean": round(mean, 8),
        "std": round(math.sqrt(variance), 8),
        "min": round(min(values), 8) if values else 0.0,
        "max": round(max(values), 8) if values else 0.0,
        "nonzero_ratio": round(nonzero / len(values), 8) if values else 0.0,
    }

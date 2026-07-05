"""Shared signal helpers for simulated piezo sensor outputs."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np


def read_piezo_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def signal_by_step(rows: list[dict[str, str]], *, sensor_id: int | None = None) -> tuple[list[int], list[int], np.ndarray]:
    steps = sorted({int(row["step"]) for row in rows})
    sensor_ids = sorted({int(row["sensor_id"]) for row in rows})
    sample_count = (max(steps) - min(steps) + 1) if steps else 0
    signal = np.zeros(sample_count, dtype=np.float64)
    if steps:
        first_step = min(steps)
        for row in rows:
            if sensor_id is not None and int(row["sensor_id"]) != sensor_id:
                continue
            signal[int(row["step"]) - first_step] += float(row["piezo_signal"])
    return steps, sensor_ids, signal


def dc_block(signal: np.ndarray, coefficient: float) -> np.ndarray:
    if signal.size == 0 or coefficient <= 0:
        return signal
    if not 0 < coefficient < 1:
        raise ValueError("dc_block must be between 0 and 1")
    output = np.zeros_like(signal, dtype=np.float64)
    previous_input = 0.0
    previous_output = 0.0
    for index, value in enumerate(signal.astype(np.float64)):
        current = value - previous_input + coefficient * previous_output
        output[index] = current
        previous_input = value
        previous_output = current
    return output


def moving_average(signal: np.ndarray, window: int) -> np.ndarray:
    if signal.size == 0 or window <= 1:
        return signal.astype(np.float64)
    resolved = min(window, signal.size)
    kernel = np.ones(resolved, dtype=np.float64) / resolved
    return np.convolve(signal.astype(np.float64), kernel, mode="same")


def normalize_audio_signal(signal: np.ndarray) -> np.ndarray:
    if signal.size == 0:
        return signal.astype(np.float64)
    centered = signal.astype(np.float64) - float(signal.mean())
    peak = float(np.max(np.abs(centered))) if centered.size else 0.0
    if peak <= 0:
        return np.zeros_like(centered, dtype=np.float64)
    return centered / peak


def resample_for_audio(signal: np.ndarray, sample_count: int) -> np.ndarray:
    if signal.size == 0:
        return np.zeros(sample_count, dtype=np.float64)
    source = np.repeat(signal, 2) if signal.size == 1 else signal
    normalized = normalize_audio_signal(source)
    x_old = np.linspace(0.0, 1.0, normalized.size)
    x_new = np.linspace(0.0, 1.0, sample_count)
    return np.interp(x_new, x_old, normalized)

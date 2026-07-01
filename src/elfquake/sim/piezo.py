"""Piezo-like electromagnetic precursor sensors for sandpile simulations."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numba import njit


PIEZO_SENSOR_FIELDS = [
    "step",
    "sensor_id",
    "x",
    "y",
    "piezo_signal",
    "piezo_total_source",
    "near_critical_cell_count",
    "critical_cell_count",
    "nearest_critical_distance",
    "max_stress_ratio",
]


@dataclass(frozen=True)
class PiezoConfig:
    sensor_count: int = 16
    susceptibility_base: float = 0.15
    susceptibility_variation: float = 0.85
    cluster_count: int = 8
    cluster_radius: float = 0.0
    activation_ratio: float = 0.75
    attenuation_radius: float = 0.0
    max_distance_radius: float = 0.0


def validate_piezo_config(config: PiezoConfig) -> None:
    if config.sensor_count < 1:
        raise ValueError("piezo sensor_count must be at least 1")
    if config.susceptibility_base < 0:
        raise ValueError("piezo susceptibility_base must be non-negative")
    if config.susceptibility_variation < 0:
        raise ValueError("piezo susceptibility_variation must be non-negative")
    if config.cluster_count < 0:
        raise ValueError("piezo cluster_count must be non-negative")
    if config.cluster_radius < 0:
        raise ValueError("piezo cluster_radius must be non-negative")
    if config.activation_ratio < 0:
        raise ValueError("piezo activation_ratio must be non-negative")
    if config.attenuation_radius < 0:
        raise ValueError("piezo attenuation_radius must be non-negative")
    if config.max_distance_radius < 0:
        raise ValueError("piezo max_distance_radius must be non-negative")


def build_piezo_susceptibility(
    *,
    rng,
    width: int,
    height: int,
    config: PiezoConfig,
) -> np.ndarray:
    """Build a deterministic quartz-like susceptibility map with clustered regions."""
    validate_piezo_config(config)
    radius = config.cluster_radius or max(width, height) / 8.0
    susceptibility = np.full((height, width), config.susceptibility_base, dtype=np.float64)
    if config.cluster_count == 0 or config.susceptibility_variation == 0:
        return susceptibility

    ys = np.arange(height, dtype=np.float64)[:, None]
    xs = np.arange(width, dtype=np.float64)[None, :]
    radius_sq = max(radius * radius, 1.0)
    for _ in range(config.cluster_count):
        center_y = float(rng.integers(0, height))
        center_x = float(rng.integers(0, width))
        amplitude = float(rng.random()) * config.susceptibility_variation
        distance_sq = (ys - center_y) ** 2 + (xs - center_x) ** 2
        susceptibility += amplitude / (1.0 + distance_sq / radius_sq)
    return np.clip(susceptibility, 0.0, 1.0)


def build_piezo_sensor_rows(
    *,
    step: int,
    sensors: np.ndarray,
    grid: np.ndarray,
    previous_grid: np.ndarray,
    susceptibility: np.ndarray,
    threshold: int,
    config: PiezoConfig,
) -> list[dict[str, str]]:
    validate_piezo_config(config)
    attenuation_radius = config.attenuation_radius or max(grid.shape) / 8.0
    max_distance_radius = config.max_distance_radius or attenuation_radius * 3.0
    (
        signals,
        nearest_distances,
        total_source,
        near_critical_count,
        critical_count,
        max_stress_ratio,
    ) = _piezo_sensor_values(
        grid.astype(np.float64),
        previous_grid.astype(np.float64),
        susceptibility.astype(np.float64),
        sensors.astype(np.int64),
        float(threshold),
        float(config.activation_ratio),
        float(attenuation_radius),
        float(max_distance_radius),
    )
    rows = []
    for sensor_id, point in enumerate(sensors):
        nearest = nearest_distances[sensor_id]
        rows.append(
            {
                "step": str(step),
                "sensor_id": str(sensor_id),
                "x": str(int(point[1])),
                "y": str(int(point[0])),
                "piezo_signal": f"{float(signals[sensor_id]):.9f}",
                "piezo_total_source": f"{float(total_source):.9f}",
                "near_critical_cell_count": str(int(near_critical_count)),
                "critical_cell_count": str(int(critical_count)),
                "nearest_critical_distance": "" if nearest < 0 else f"{float(nearest):.6f}",
                "max_stress_ratio": f"{float(max_stress_ratio):.6f}",
            }
        )
    return rows


@njit(cache=True)
def _piezo_sensor_values(
    grid,
    previous_grid,
    susceptibility,
    sensors,
    threshold: float,
    activation_ratio: float,
    attenuation_radius: float,
    max_distance_radius: float,
):
    height, width = grid.shape
    sensor_count = sensors.shape[0]
    signals = np.zeros(sensor_count, dtype=np.float64)
    nearest_sq = np.empty(sensor_count, dtype=np.float64)
    for sensor_index in range(sensor_count):
        nearest_sq[sensor_index] = -1.0

    attenuation_sq = attenuation_radius * attenuation_radius
    max_distance_sq = max_distance_radius * max_distance_radius
    denominator = 1.0 - activation_ratio
    if denominator <= 0:
        denominator = 1.0

    total_source = 0.0
    near_critical_count = 0
    critical_count = 0
    max_stress_ratio = 0.0

    for y in range(height):
        for x in range(width):
            max_difference = 0.0
            value = grid[y, x]
            if y > 0:
                diff = value - grid[y - 1, x]
                if diff > max_difference:
                    max_difference = diff
            if y < height - 1:
                diff = value - grid[y + 1, x]
                if diff > max_difference:
                    max_difference = diff
            if x > 0:
                diff = value - grid[y, x - 1]
                if diff > max_difference:
                    max_difference = diff
            if x < width - 1:
                diff = value - grid[y, x + 1]
                if diff > max_difference:
                    max_difference = diff

            stress_ratio = max_difference / threshold
            if stress_ratio > max_stress_ratio:
                max_stress_ratio = stress_ratio
            if stress_ratio < activation_ratio:
                continue

            near_critical_count += 1
            if stress_ratio >= 1.0:
                critical_count += 1

            for sensor_index in range(sensor_count):
                sensor_y = sensors[sensor_index, 0]
                sensor_x = sensors[sensor_index, 1]
                dy = float(y - sensor_y)
                dx = float(x - sensor_x)
                distance_sq = dy * dy + dx * dx
                if nearest_sq[sensor_index] < 0 or distance_sq < nearest_sq[sensor_index]:
                    nearest_sq[sensor_index] = distance_sq

            positive_change = value - previous_grid[y, x]
            if positive_change <= 0:
                continue
            strength = (stress_ratio - activation_ratio) / denominator
            source = susceptibility[y, x] * positive_change * strength
            if source <= 0:
                continue

            total_source += source
            for sensor_index in range(sensor_count):
                sensor_y = sensors[sensor_index, 0]
                sensor_x = sensors[sensor_index, 1]
                dy = float(y - sensor_y)
                dx = float(x - sensor_x)
                distance_sq = dy * dy + dx * dx
                if distance_sq <= max_distance_sq:
                    signals[sensor_index] += source / (1.0 + distance_sq / attenuation_sq)

    nearest_distances = np.empty(sensor_count, dtype=np.float64)
    for sensor_index in range(sensor_count):
        if nearest_sq[sensor_index] < 0:
            nearest_distances[sensor_index] = -1.0
        else:
            nearest_distances[sensor_index] = np.sqrt(nearest_sq[sensor_index])
    return (
        signals,
        nearest_distances,
        total_source,
        near_critical_count,
        critical_count,
        max_stress_ratio,
    )

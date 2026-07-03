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
    "piezo_charge_total",
    "piezo_charge_max",
    "piezo_release_total",
]

AVALANCHE_SIGNAL_SENSOR_FIELDS = [
    "step",
    "sensor_id",
    "x",
    "y",
    "avalanche_signal",
    "avalanche_total_source",
    "active_topple_cell_count",
    "max_local_topple",
    "nearest_topple_distance",
    "stress_drop_total",
    "stress_drop_max",
    "avalanche_release_total",
]

# Backward-compatible name for older callers. The direct avalanche channel is
# seismic-like, not piezo-like; new code should use AVALANCHE_SIGNAL_SENSOR_FIELDS.
AVALANCHE_PIEZO_SENSOR_FIELDS = AVALANCHE_SIGNAL_SENSOR_FIELDS


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
    charge_decay: float = 0.995
    charge_coupling: float = 1.0
    release_ratio: float = 0.15
    critical_release_ratio: float = 0.05
    saturation: float = 1000.0


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
    if not 0 <= config.charge_decay <= 1:
        raise ValueError("piezo charge_decay must be between 0 and 1")
    if config.charge_coupling < 0:
        raise ValueError("piezo charge_coupling must be non-negative")
    if not 0 <= config.release_ratio <= 1:
        raise ValueError("piezo release_ratio must be between 0 and 1")
    if not 0 <= config.critical_release_ratio <= 1:
        raise ValueError("piezo critical_release_ratio must be between 0 and 1")
    if config.saturation < 0:
        raise ValueError("piezo saturation must be non-negative")


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
    charge: np.ndarray,
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
        charge_total,
        charge_max,
        release_total,
    ) = _piezo_sensor_values(
        grid.astype(np.float64),
        previous_grid.astype(np.float64),
        charge,
        susceptibility.astype(np.float64),
        sensors.astype(np.int64),
        float(threshold),
        float(config.activation_ratio),
        float(attenuation_radius),
        float(max_distance_radius),
        float(config.charge_decay),
        float(config.charge_coupling),
        float(config.release_ratio),
        float(config.critical_release_ratio),
        float(config.saturation),
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
                "piezo_charge_total": f"{float(charge_total):.9f}",
                "piezo_charge_max": f"{float(charge_max):.9f}",
                "piezo_release_total": f"{float(release_total):.9f}",
            }
        )
    return rows


def build_avalanche_signal_sensor_rows(
    *,
    step: int,
    sensors: np.ndarray,
    pre_relax_grid: np.ndarray,
    post_relax_grid: np.ndarray,
    topple_counts: np.ndarray,
    susceptibility: np.ndarray,
    config: PiezoConfig,
) -> list[dict[str, str]]:
    validate_piezo_config(config)
    attenuation_radius = config.attenuation_radius or max(post_relax_grid.shape) / 8.0
    max_distance_radius = config.max_distance_radius or attenuation_radius * 3.0
    (
        signals,
        nearest_distances,
        total_source,
        active_count,
        max_local_topple,
        stress_drop_total,
        stress_drop_max,
    ) = _avalanche_piezo_sensor_values(
        pre_relax_grid.astype(np.float64),
        post_relax_grid.astype(np.float64),
        topple_counts.astype(np.float64),
        susceptibility.astype(np.float64),
        sensors.astype(np.int64),
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
                "avalanche_signal": f"{float(signals[sensor_id]):.9f}",
                "avalanche_total_source": f"{float(total_source):.9f}",
                "active_topple_cell_count": str(int(active_count)),
                "max_local_topple": str(int(max_local_topple)),
                "nearest_topple_distance": "" if nearest < 0 else f"{float(nearest):.6f}",
                "stress_drop_total": f"{float(stress_drop_total):.9f}",
                "stress_drop_max": f"{float(stress_drop_max):.9f}",
                "avalanche_release_total": f"{float(total_source):.9f}",
            }
        )
    return rows


def build_avalanche_piezo_sensor_rows(**kwargs) -> list[dict[str, str]]:
    """Compatibility wrapper for the renamed direct avalanche signal builder."""
    return build_avalanche_signal_sensor_rows(**kwargs)


@njit(cache=True)
def _piezo_sensor_values(
    grid,
    previous_grid,
    charge,
    susceptibility,
    sensors,
    threshold: float,
    activation_ratio: float,
    attenuation_radius: float,
    max_distance_radius: float,
    charge_decay: float,
    charge_coupling: float,
    release_ratio: float,
    critical_release_ratio: float,
    saturation: float,
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
    total_release = 0.0
    near_critical_count = 0
    critical_count = 0
    max_stress_ratio = 0.0
    charge_total = 0.0
    charge_max = 0.0

    for y in range(height):
        for x in range(width):
            charge[y, x] = charge[y, x] * charge_decay
            max_difference = 0.0
            previous_max_difference = 0.0
            value = grid[y, x]
            previous_value = previous_grid[y, x]
            if y > 0:
                diff = value - grid[y - 1, x]
                if diff > max_difference:
                    max_difference = diff
                previous_diff = previous_value - previous_grid[y - 1, x]
                if previous_diff > previous_max_difference:
                    previous_max_difference = previous_diff
            if y < height - 1:
                diff = value - grid[y + 1, x]
                if diff > max_difference:
                    max_difference = diff
                previous_diff = previous_value - previous_grid[y + 1, x]
                if previous_diff > previous_max_difference:
                    previous_max_difference = previous_diff
            if x > 0:
                diff = value - grid[y, x - 1]
                if diff > max_difference:
                    max_difference = diff
                previous_diff = previous_value - previous_grid[y, x - 1]
                if previous_diff > previous_max_difference:
                    previous_max_difference = previous_diff
            if x < width - 1:
                diff = value - grid[y, x + 1]
                if diff > max_difference:
                    max_difference = diff
                previous_diff = previous_value - previous_grid[y, x + 1]
                if previous_diff > previous_max_difference:
                    previous_max_difference = previous_diff

            stress_ratio = max_difference / threshold
            previous_stress_ratio = previous_max_difference / threshold
            stress_delta = stress_ratio - previous_stress_ratio
            if stress_ratio > max_stress_ratio:
                max_stress_ratio = stress_ratio
            positive_change = value - previous_value
            strain_drive = 0.0
            if positive_change > 0:
                strain_drive += positive_change
            if stress_delta > 0:
                strain_drive += stress_delta * threshold
            if strain_drive > 0 and stress_ratio >= activation_ratio:
                strength = (stress_ratio - activation_ratio) / denominator
                added_charge = susceptibility[y, x] * strain_drive * strength * charge_coupling
                if added_charge > 0:
                    charge[y, x] += added_charge
                    if saturation > 0 and charge[y, x] > saturation:
                        excess = charge[y, x] - saturation
                        charge[y, x] = saturation
                        total_release += excess

            if stress_ratio >= activation_ratio:
                near_critical_count += 1
            stress_gate = 0.0
            if stress_ratio >= activation_ratio:
                stress_gate = (stress_ratio - activation_ratio) / denominator
                if stress_gate > 1.0:
                    stress_gate = 1.0

            regular_release = 0.0
            if stress_gate > 0.0 and strain_drive > 0.0 and charge[y, x] > 0.0:
                drive_gate = strain_drive / threshold
                if drive_gate > 1.0:
                    drive_gate = 1.0
                regular_release = charge[y, x] * release_ratio * stress_gate * stress_gate * drive_gate
                charge[y, x] -= regular_release
                total_release += regular_release

            critical_release = 0.0
            if stress_ratio >= 1.0 and previous_stress_ratio < 1.0:
                critical_count += 1
                critical_release = charge[y, x] * critical_release_ratio
                charge[y, x] -= critical_release
                total_release += critical_release
            elif stress_ratio >= 1.0:
                critical_count += 1

            charge_total += charge[y, x]
            if charge[y, x] > charge_max:
                charge_max = charge[y, x]

            source = regular_release + critical_release
            if source > 0:
                total_source += source
            if stress_ratio >= activation_ratio or source > 0:
                for sensor_index in range(sensor_count):
                    sensor_y = sensors[sensor_index, 0]
                    sensor_x = sensors[sensor_index, 1]
                    dy = float(y - sensor_y)
                    dx = float(x - sensor_x)
                    distance_sq = dy * dy + dx * dx
                    if stress_ratio >= activation_ratio:
                        if nearest_sq[sensor_index] < 0 or distance_sq < nearest_sq[sensor_index]:
                            nearest_sq[sensor_index] = distance_sq
                    if source > 0 and distance_sq <= max_distance_sq:
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
        charge_total,
        charge_max,
        total_release,
    )


@njit(cache=True)
def _avalanche_piezo_sensor_values(
    pre_relax_grid,
    post_relax_grid,
    topple_counts,
    susceptibility,
    sensors,
    attenuation_radius: float,
    max_distance_radius: float,
):
    height, width = post_relax_grid.shape
    sensor_count = sensors.shape[0]
    signals = np.zeros(sensor_count, dtype=np.float64)
    nearest_sq = np.empty(sensor_count, dtype=np.float64)
    for sensor_index in range(sensor_count):
        nearest_sq[sensor_index] = -1.0

    attenuation_sq = attenuation_radius * attenuation_radius
    max_distance_sq = max_distance_radius * max_distance_radius
    total_source = 0.0
    active_count = 0
    max_local_topple = 0.0
    stress_drop_total = 0.0
    stress_drop_max = 0.0

    for y in range(height):
        for x in range(width):
            local_topple = topple_counts[y, x]
            if local_topple <= 0:
                continue
            active_count += 1
            if local_topple > max_local_topple:
                max_local_topple = local_topple
            stress_drop = pre_relax_grid[y, x] - post_relax_grid[y, x]
            if stress_drop < 0:
                stress_drop = 0.0
            release = susceptibility[y, x] * local_topple * (1.0 + stress_drop)
            if release <= 0:
                continue
            stress_drop_total += stress_drop
            if stress_drop > stress_drop_max:
                stress_drop_max = stress_drop
            total_source += release
            for sensor_index in range(sensor_count):
                sensor_y = sensors[sensor_index, 0]
                sensor_x = sensors[sensor_index, 1]
                dy = float(y - sensor_y)
                dx = float(x - sensor_x)
                distance_sq = dy * dy + dx * dx
                if nearest_sq[sensor_index] < 0 or distance_sq < nearest_sq[sensor_index]:
                    nearest_sq[sensor_index] = distance_sq
                if distance_sq <= max_distance_sq:
                    signals[sensor_index] += release / (1.0 + distance_sq / attenuation_sq)

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
        active_count,
        max_local_topple,
        stress_drop_total,
        stress_drop_max,
    )

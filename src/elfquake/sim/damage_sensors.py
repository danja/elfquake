"""Causal fixed-location summaries of the delayed-failure damage field."""

from __future__ import annotations

import numpy as np
from numba import njit


def measure_local_damage(
    *, sensors: np.ndarray, damage: np.ndarray, radius: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return local mean, maximum, active fraction, and variation per sensor."""
    return _measure_local_damage(sensors.astype(np.int64), damage.astype(np.float64), max(1, int(round(radius))))


@njit(cache=True)
def _measure_local_damage(sensors, damage, radius):
    height, width = damage.shape
    count = sensors.shape[0]
    means = np.zeros(count, dtype=np.float64)
    maximums = np.zeros(count, dtype=np.float64)
    active_fractions = np.zeros(count, dtype=np.float64)
    variations = np.zeros(count, dtype=np.float64)
    radius_sq = radius * radius
    for sensor_index in range(count):
        center_y = sensors[sensor_index, 0]
        center_x = sensors[sensor_index, 1]
        total = 0.0
        total_sq = 0.0
        active = 0
        samples = 0
        maximum = 0.0
        for y in range(max(0, center_y - radius), min(height, center_y + radius + 1)):
            for x in range(max(0, center_x - radius), min(width, center_x + radius + 1)):
                dy = y - center_y
                dx = x - center_x
                if dy * dy + dx * dx > radius_sq:
                    continue
                value = damage[y, x]
                total += value
                total_sq += value * value
                active += int(value > 0.0)
                maximum = max(maximum, value)
                samples += 1
        if samples > 0:
            mean = total / samples
            means[sensor_index] = mean
            maximums[sensor_index] = maximum
            active_fractions[sensor_index] = active / samples
            variations[sensor_index] = max(0.0, total_sq / samples - mean * mean) ** 0.5
    return means, maximums, active_fractions, variations

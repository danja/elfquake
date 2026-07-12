"""Observable pre-relaxation spatial-state metrics for the sandpile."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numba import njit


@dataclass(frozen=True)
class PreRelaxSpatialState:
    """Near-critical geometry derived before the same-step avalanche relaxation."""

    near_critical_contact_count: int
    near_critical_coherence: float
    near_critical_weighted_stress: float


def measure_pre_relax_spatial_state(
    *, grid: np.ndarray, threshold: int, activation_ratio: float,
) -> PreRelaxSpatialState:
    if threshold < 1:
        raise ValueError("threshold must be at least 1")
    if activation_ratio < 0:
        raise ValueError("activation_ratio must be non-negative")
    contacts, active_count, weighted_stress = _measure_near_critical_geometry(
        grid.astype(np.float64), float(threshold), float(activation_ratio),
    )
    coherence = contacts / active_count if active_count else 0.0
    return PreRelaxSpatialState(
        near_critical_contact_count=int(contacts),
        near_critical_coherence=float(coherence),
        near_critical_weighted_stress=float(weighted_stress),
    )


@njit(cache=True)
def _measure_near_critical_geometry(grid, threshold: float, activation_ratio: float):
    height, width = grid.shape
    active = np.zeros((height, width), dtype=np.uint8)
    active_count = 0
    weighted_stress = 0.0
    denominator = max(1.0 - activation_ratio, 1e-12)

    for y in range(height):
        for x in range(width):
            value = grid[y, x]
            maximum = 0.0
            if y > 0:
                maximum = max(maximum, value - grid[y - 1, x])
            if y < height - 1:
                maximum = max(maximum, value - grid[y + 1, x])
            if x > 0:
                maximum = max(maximum, value - grid[y, x - 1])
            if x < width - 1:
                maximum = max(maximum, value - grid[y, x + 1])
            ratio = maximum / threshold
            if ratio >= activation_ratio:
                active[y, x] = 1
                active_count += 1
                weighted_stress += min(1.0, (ratio - activation_ratio) / denominator)

    contacts = 0
    for y in range(height):
        for x in range(width):
            if active[y, x] == 0:
                continue
            if y + 1 < height and active[y + 1, x]:
                contacts += 1
            if x + 1 < width and active[y, x + 1]:
                contacts += 1
    return contacts, active_count, weighted_stress

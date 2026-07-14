"""Slow, stress-derived fracture maturation for the sandpile simulation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numba import njit


@dataclass(frozen=True)
class MatureWeaknessConfig:
    enabled: bool = False
    damage_threshold: float = 0.50
    dwell_steps: int = 5
    maturation_rate: float = 0.10
    decay: float = 0.995
    threshold_reduction: float = 0.20
    reset_fraction: float = 0.90


def validate_mature_weakness_config(config: MatureWeaknessConfig) -> None:
    if not 0 <= config.damage_threshold < 1:
        raise ValueError("mature weakness damage_threshold must be in [0, 1)")
    if config.dwell_steps < 1:
        raise ValueError("mature weakness dwell_steps must be at least 1")
    if config.maturation_rate < 0:
        raise ValueError("mature weakness maturation_rate must be non-negative")
    if not 0 <= config.decay <= 1:
        raise ValueError("mature weakness decay must be between 0 and 1")
    if not 0 <= config.threshold_reduction < 1:
        raise ValueError("mature weakness threshold_reduction must be in [0, 1)")
    if not 0 <= config.reset_fraction <= 1:
        raise ValueError("mature weakness reset_fraction must be between 0 and 1")


def update_mature_weakness(
    *, damage: np.ndarray, weakness: np.ndarray, dwell: np.ndarray, config: MatureWeaknessConfig,
) -> None:
    """Advance weakness only after microdamage persists for the dwell period."""
    _update_mature_weakness(
        damage, weakness, dwell, config.damage_threshold, config.dwell_steps,
        config.maturation_rate, config.decay,
    )


def reset_toppled_mature_weakness(
    *, weakness: np.ndarray, dwell: np.ndarray, topple_counts: np.ndarray, reset_fraction: float,
) -> None:
    _reset_toppled_mature_weakness(weakness, dwell, topple_counts, reset_fraction)


def summarize_mature_weakness(weakness: np.ndarray) -> tuple[float, float, int]:
    return float(weakness.sum()), float(weakness.max()), int((weakness > 0).sum())


@njit(cache=True)
def _update_mature_weakness(damage, weakness, dwell, damage_threshold, dwell_steps, maturation_rate, decay):
    height, width = damage.shape
    denominator = max(1.0 - damage_threshold, 1e-12)
    for y in range(height):
        for x in range(width):
            if damage[y, x] >= damage_threshold:
                dwell[y, x] += 1
            else:
                dwell[y, x] = 0
            value = weakness[y, x] * decay
            if dwell[y, x] >= dwell_steps:
                value += maturation_rate * min(1.0, (damage[y, x] - damage_threshold) / denominator)
            weakness[y, x] = min(1.0, value)


@njit(cache=True)
def _reset_toppled_mature_weakness(weakness, dwell, topple_counts, reset_fraction):
    height, width = weakness.shape
    retained = 1.0 - reset_fraction
    for y in range(height):
        for x in range(width):
            if topple_counts[y, x] > 0:
                weakness[y, x] *= retained
                dwell[y, x] = 0


@njit(cache=True)
def relax_with_damage_and_mature_weakness(
    grid, topple_counts, threshold, max_sweeps, damage, damage_reduction, weakness, weakness_reduction,
):
    height, width = grid.shape
    topple_count = 0
    released_mass = 0
    avalanche_count = 0
    relaxation_converged = 0
    in_avalanche = False
    for _ in range(max_sweeps):
        unstable_found = False
        delta = np.zeros_like(grid)
        for y in range(height):
            for x in range(width):
                available = grid[y, x]
                if available <= 0:
                    continue
                reduction = min(0.99, damage_reduction * damage[y, x] + weakness_reduction * weakness[y, x])
                effective_threshold = threshold * (1.0 - reduction)
                moved = 0
                if y > 0:
                    moved += _move_downhill(grid, delta, y, x, y - 1, x, effective_threshold, available - moved)
                if y < height - 1:
                    moved += _move_downhill(grid, delta, y, x, y + 1, x, effective_threshold, available - moved)
                if x > 0:
                    moved += _move_downhill(grid, delta, y, x, y, x - 1, effective_threshold, available - moved)
                if x < width - 1:
                    moved += _move_downhill(grid, delta, y, x, y, x + 1, effective_threshold, available - moved)
                if moved > 0:
                    unstable_found = True
                    in_avalanche = True
                    topple_counts[y, x] += moved
                    topple_count += moved
        if not unstable_found:
            if in_avalanche:
                avalanche_count = 1
            relaxation_converged = 1
            break
        grid += delta
    return topple_count, released_mass, avalanche_count, relaxation_converged


@njit(cache=True)
def _move_downhill(grid, delta, y, x, ny, nx, threshold, available):
    if available <= 0 or grid[y, x] - grid[ny, nx] < threshold:
        return 0
    transfer = int((grid[y, x] - grid[ny, nx] - threshold) // 2) + 1
    transfer = min(transfer, available)
    delta[y, x] -= transfer
    delta[ny, nx] += transfer
    return transfer

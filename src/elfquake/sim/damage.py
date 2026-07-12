"""Optional delayed local-failure dynamics for the sandpile simulator."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numba import njit


@dataclass(frozen=True)
class DamageConfig:
    enabled: bool = False
    activation_ratio: float = 0.85
    decay: float = 0.985
    coupling: float = 0.10
    threshold_reduction: float = 0.25
    reset_fraction: float = 0.90


def validate_damage_config(config: DamageConfig) -> None:
    if config.activation_ratio < 0:
        raise ValueError("damage activation_ratio must be non-negative")
    if not 0 <= config.decay <= 1:
        raise ValueError("damage decay must be between 0 and 1")
    if config.coupling < 0:
        raise ValueError("damage coupling must be non-negative")
    if not 0 <= config.threshold_reduction < 1:
        raise ValueError("damage threshold_reduction must be in [0, 1)")
    if not 0 <= config.reset_fraction <= 1:
        raise ValueError("damage reset_fraction must be between 0 and 1")


def update_damage(*, grid: np.ndarray, damage: np.ndarray, threshold: int, config: DamageConfig) -> None:
    _update_damage(grid, damage, float(threshold), config.activation_ratio, config.decay, config.coupling)


def reset_toppled_damage(*, damage: np.ndarray, topple_counts: np.ndarray, reset_fraction: float) -> None:
    _reset_toppled_damage(damage, topple_counts, reset_fraction)


def summarize_damage(damage: np.ndarray) -> tuple[float, float, int]:
    return float(damage.sum()), float(damage.max()), int((damage > 0).sum())


@njit(cache=True)
def _update_damage(grid, damage, threshold, activation_ratio, decay, coupling):
    height, width = grid.shape
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
            value_damage = damage[y, x] * decay
            if ratio >= activation_ratio:
                value_damage += coupling * min(1.0, (ratio - activation_ratio) / denominator)
            damage[y, x] = min(1.0, value_damage)


@njit(cache=True)
def _reset_toppled_damage(damage, topple_counts, reset_fraction):
    height, width = damage.shape
    retained = 1.0 - reset_fraction
    for y in range(height):
        for x in range(width):
            if topple_counts[y, x] > 0:
                damage[y, x] *= retained


@njit(cache=True)
def relax_with_damage(grid, topple_counts, threshold, max_sweeps, damage, threshold_reduction):
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
                effective_threshold = threshold * (1.0 - threshold_reduction * damage[y, x])
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
    if available <= 0:
        return 0
    difference = grid[y, x] - grid[ny, nx]
    if difference < threshold:
        return 0
    transfer = int((difference - threshold) // 2) + 1
    if transfer > available:
        transfer = available
    delta[y, x] -= transfer
    delta[ny, nx] += transfer
    return transfer

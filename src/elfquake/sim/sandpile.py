"""Numba-backed stochastic sandpile simulator."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np

try:
    from numba import njit
except ImportError as error:  # pragma: no cover - exercised by runtime environment.
    raise RuntimeError("numba is required for sandpile simulation; activate the project venv") from error


SUMMARY_FIELDS = [
    "step",
    "deposition_count",
    "avalanche_count",
    "topple_count",
    "max_height",
    "mean_height",
    "released_mass",
]

SENSOR_FIELDS = [
    "step",
    "sensor_id",
    "x",
    "y",
    "height",
    "local_topple_count",
]


@dataclass(frozen=True)
class SandpileConfig:
    width: int = 128
    height: int = 128
    steps: int = 100
    threshold: int = 4
    source_count: int = 16
    sensor_count: int = 16
    deposition_probability: float = 0.5
    seed: int = 1
    max_relaxation_sweeps: int = 10000


def run_sandpile_simulation(
    *,
    config: SandpileConfig,
    summary_out: Path,
    sensors_out: Path,
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    validate_config(config)
    rng = np.random.default_rng(config.seed)
    grid = np.zeros((config.height, config.width), dtype=np.int64)
    sources = _random_points(rng, config.width, config.height, config.source_count)
    sensors = _random_points(rng, config.width, config.height, config.sensor_count)
    summary_rows = []
    sensor_rows = []

    for step in range(config.steps):
        active_sources = rng.random(config.source_count) < config.deposition_probability
        deposition_count = int(active_sources.sum())
        _deposit(grid, sources, active_sources)
        topple_counts = np.zeros_like(grid)
        topple_count, released_mass, avalanche_count = _relax(
            grid,
            topple_counts,
            config.threshold,
            config.max_relaxation_sweeps,
        )
        summary_rows.append(
            {
                "step": str(step),
                "deposition_count": str(deposition_count),
                "avalanche_count": str(avalanche_count),
                "topple_count": str(int(topple_count)),
                "max_height": str(int(grid.max())),
                "mean_height": f"{float(grid.mean()):.6f}",
                "released_mass": str(int(released_mass)),
            }
        )
        sensor_rows.extend(_sensor_rows(step, sensors, grid, topple_counts))

    _write_csv(summary_out, SUMMARY_FIELDS, summary_rows)
    _write_csv(sensors_out, SENSOR_FIELDS, sensor_rows)
    return summary_rows, sensor_rows


def validate_config(config: SandpileConfig) -> None:
    if config.width < 2 or config.height < 2:
        raise ValueError("width and height must be at least 2")
    if config.steps < 1:
        raise ValueError("steps must be at least 1")
    if config.threshold < 2:
        raise ValueError("threshold must be at least 2")
    if config.source_count < 1 or config.sensor_count < 1:
        raise ValueError("source_count and sensor_count must be at least 1")
    if not 0 <= config.deposition_probability <= 1:
        raise ValueError("deposition_probability must be between 0 and 1")
    if config.max_relaxation_sweeps < 1:
        raise ValueError("max_relaxation_sweeps must be at least 1")


def _random_points(rng, width: int, height: int, count: int) -> np.ndarray:
    xs = rng.integers(0, width, size=count, dtype=np.int64)
    ys = rng.integers(0, height, size=count, dtype=np.int64)
    return np.column_stack((ys, xs)).astype(np.int64)


@njit(cache=True)
def _deposit(grid, sources, active_sources):
    for index in range(sources.shape[0]):
        if active_sources[index]:
            y = sources[index, 0]
            x = sources[index, 1]
            grid[y, x] += 1


@njit(cache=True)
def _relax(grid, topple_counts, threshold: int, max_sweeps: int):
    height, width = grid.shape
    topple_count = 0
    released_mass = 0
    avalanche_count = 0
    in_avalanche = False
    for _ in range(max_sweeps):
        unstable_found = False
        delta = np.zeros_like(grid)
        for y in range(height):
            for x in range(width):
                topples = grid[y, x] // threshold
                if topples <= 0:
                    continue
                unstable_found = True
                in_avalanche = True
                grid[y, x] -= topples * threshold
                topple_counts[y, x] += topples
                topple_count += topples
                released_mass += _spread(delta, y, x, height, width, topples)
        if not unstable_found:
            if in_avalanche:
                avalanche_count = 1
            break
        grid += delta
    return topple_count, released_mass, avalanche_count


@njit(cache=True)
def _spread(delta, y: int, x: int, height: int, width: int, grains: int) -> int:
    released = 0
    if y > 0:
        delta[y - 1, x] += grains
    else:
        released += grains
    if y < height - 1:
        delta[y + 1, x] += grains
    else:
        released += grains
    if x > 0:
        delta[y, x - 1] += grains
    else:
        released += grains
    if x < width - 1:
        delta[y, x + 1] += grains
    else:
        released += grains
    return released


def _sensor_rows(step: int, sensors: np.ndarray, grid: np.ndarray, topple_counts: np.ndarray) -> list[dict[str, str]]:
    rows = []
    for sensor_id, point in enumerate(sensors):
        y = int(point[0])
        x = int(point[1])
        rows.append(
            {
                "step": str(step),
                "sensor_id": str(sensor_id),
                "x": str(x),
                "y": str(y),
                "height": str(int(grid[y, x])),
                "local_topple_count": str(int(topple_counts[y, x])),
            }
        )
    return rows


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)

"""Numba-backed stochastic sandpile simulator."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np

from elfquake.sim.avalanche_activity import AVALANCHE_ACTIVITY_FIELDS, build_avalanche_activity_row
from elfquake.sim.piezo import (
    AVALANCHE_SIGNAL_SENSOR_FIELDS,
    PIEZO_SENSOR_FIELDS,
    PiezoConfig,
    build_avalanche_signal_sensor_rows,
    build_piezo_sensor_rows,
    build_piezo_susceptibility,
)
from elfquake.sim.sandpile_outputs import write_csv, write_snapshot

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
    "relaxation_converged",
    "unstable_cell_count",
    "safety_released_mass",
    "target_fill_count",
    "bottom_layer_removed_mass",
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
    deposition_mode: str = "sources"
    target_mean_height: float = 0.0
    target_fill_limit: int = 0
    target_fill_mode: str = "uniform"
    bottom_layer_removal_interval: int = 0
    initial_fill_mode: str = "none"
    initial_fill_mean_height: float = 0.0
    initial_fill_variation: float = 0.0
    initial_fill_smooth_passes: int = 0
    warmup_steps: int = 0


def run_sandpile_simulation(
    *,
    config: SandpileConfig,
    summary_out: Path,
    sensors_out: Path,
    piezo_out: Path | None = None,
    avalanche_signal_out: Path | None = None,
    avalanche_activity_out: Path | None = None,
    piezo_avalanche_out: Path | None = None,
    piezo_config: PiezoConfig | None = None,
    avalanche_signal_config: PiezoConfig | None = None,
    snapshot_dir: Path | None = None,
    snapshot_interval: int = 0,
    progress_interval: int = 0,
    progress_callback: Callable[[int, int, dict[str, str]], None] | None = None,
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    validate_config(config)
    if (
        avalanche_signal_out is not None
        and piezo_avalanche_out is not None
        and avalanche_signal_out != piezo_avalanche_out
    ):
        raise ValueError("use only one of avalanche_signal_out or piezo_avalanche_out")
    resolved_avalanche_signal_out = avalanche_signal_out or piezo_avalanche_out
    if snapshot_dir is not None and snapshot_interval < 1:
        raise ValueError("snapshot_interval must be at least 1 when snapshot_dir is set")
    if progress_callback is not None and progress_interval < 1:
        raise ValueError("progress_interval must be at least 1 when progress_callback is set")
    rng = np.random.default_rng(config.seed)
    grid = np.zeros((config.height, config.width), dtype=np.int64)
    sources = _random_points(rng, config.width, config.height, config.source_count)
    sensors = _random_points(rng, config.width, config.height, config.sensor_count)
    if config.initial_fill_mode != "none":
        _apply_initial_fill(grid, config)
    piezo_rows = []
    avalanche_signal_rows = []
    avalanche_activity_rows = []
    piezo_sensors = None
    piezo_susceptibility = None
    piezo_charge = None
    if piezo_out is not None or resolved_avalanche_signal_out is not None:
        resolved_piezo_config = piezo_config or PiezoConfig()
        resolved_avalanche_signal_config = avalanche_signal_config or PiezoConfig(
            attenuation_radius=0.0,
            max_distance_radius=0.0,
        )
        piezo_rng = np.random.default_rng(config.seed + 1_000_003)
        piezo_sensors = _random_points(
            piezo_rng,
            config.width,
            config.height,
            resolved_piezo_config.sensor_count,
        )
        piezo_susceptibility = build_piezo_susceptibility(
            rng=piezo_rng,
            width=config.width,
            height=config.height,
            config=resolved_piezo_config,
        )
        piezo_charge = np.zeros((config.height, config.width), dtype=np.float64)
    else:
        resolved_piezo_config = None
        resolved_avalanche_signal_config = None
    summary_rows = []
    sensor_rows = []
    snapshot_rows = []
    for warmup_step in range(config.warmup_steps):
        _advance_unrecorded_step(grid=grid, rng=rng, config=config, sources=sources, absolute_step=warmup_step)
    previous_grid = grid.copy()

    for step in range(config.steps):
        absolute_step = config.warmup_steps + step
        deposition_count = _apply_deposition(
            grid=grid,
            rng=rng,
            config=config,
            sources=sources,
        )
        target_fill_count = _fill_to_target_mean(grid, rng, config, sources)
        pre_relax_grid = grid.copy()
        if piezo_out is not None:
            assert resolved_piezo_config is not None
            assert piezo_sensors is not None
            assert piezo_susceptibility is not None
            assert piezo_charge is not None
            piezo_rows.extend(
                build_piezo_sensor_rows(
                    step=step,
                    sensors=piezo_sensors,
                    grid=grid,
                    previous_grid=previous_grid,
                    charge=piezo_charge,
                    susceptibility=piezo_susceptibility,
                    threshold=config.threshold,
                    config=resolved_piezo_config,
                )
            )
        topple_counts = np.zeros_like(grid)
        (
            topple_count,
            released_mass,
            avalanche_count,
            relaxation_converged,
            unstable_cell_count,
            safety_released_mass,
        ) = _relax(
            grid,
            topple_counts,
            config.threshold,
            config.max_relaxation_sweeps,
        )
        if avalanche_activity_out is not None:
            avalanche_activity_rows.append(build_avalanche_activity_row(step=step, topple_counts=topple_counts))
        if resolved_avalanche_signal_out is not None:
            assert resolved_piezo_config is not None
            assert resolved_avalanche_signal_config is not None
            assert piezo_sensors is not None
            assert piezo_susceptibility is not None
            avalanche_signal_rows.extend(
                build_avalanche_signal_sensor_rows(
                    step=step,
                    sensors=piezo_sensors,
                    pre_relax_grid=pre_relax_grid,
                    post_relax_grid=grid,
                    topple_counts=topple_counts,
                    susceptibility=piezo_susceptibility,
                    config=resolved_avalanche_signal_config,
                )
            )
        bottom_layer_removed_mass = 0
        if _should_remove_bottom_layer(config, absolute_step):
            bottom_layer_removed_mass = _remove_bottom_layer(grid)
            released_mass += bottom_layer_removed_mass
        summary_row = {
            "step": str(step),
            "deposition_count": str(deposition_count),
            "avalanche_count": str(avalanche_count),
            "topple_count": str(int(topple_count)),
            "max_height": str(int(grid.max())),
            "mean_height": f"{float(grid.mean()):.6f}",
            "released_mass": str(int(released_mass)),
            "relaxation_converged": str(int(relaxation_converged)),
            "unstable_cell_count": str(int(unstable_cell_count)),
            "safety_released_mass": str(int(safety_released_mass)),
            "target_fill_count": str(int(target_fill_count)),
            "bottom_layer_removed_mass": str(int(bottom_layer_removed_mass)),
        }
        summary_rows.append(summary_row)
        sensor_rows.extend(_sensor_rows(step, sensors, grid, topple_counts))
        if snapshot_dir is not None and (step % snapshot_interval == 0 or step == config.steps - 1):
            snapshot_rows.append(write_snapshot(snapshot_dir, step, grid))
        completed_steps = step + 1
        if progress_callback is not None and (
            completed_steps % progress_interval == 0 or completed_steps == config.steps
        ):
            progress_callback(completed_steps, config.steps, summary_row)
        previous_grid = grid.copy()

    write_csv(summary_out, SUMMARY_FIELDS, summary_rows)
    write_csv(sensors_out, SENSOR_FIELDS, sensor_rows)
    if piezo_out is not None:
        write_csv(piezo_out, PIEZO_SENSOR_FIELDS, piezo_rows)
    if resolved_avalanche_signal_out is not None:
        write_csv(resolved_avalanche_signal_out, AVALANCHE_SIGNAL_SENSOR_FIELDS, avalanche_signal_rows)
    if avalanche_activity_out is not None:
        write_csv(avalanche_activity_out, AVALANCHE_ACTIVITY_FIELDS, avalanche_activity_rows)
    if snapshot_dir is not None:
        write_csv(snapshot_dir / "manifest.csv", ["step", "snapshot_file"], snapshot_rows)
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
    if config.deposition_mode not in {"sources", "uniform"}:
        raise ValueError("deposition_mode must be 'sources' or 'uniform'")
    if config.target_mean_height < 0:
        raise ValueError("target_mean_height must be non-negative")
    if config.target_fill_limit < 0:
        raise ValueError("target_fill_limit must be non-negative")
    if config.target_fill_mode not in {"uniform", "sources"}:
        raise ValueError("target_fill_mode must be 'uniform' or 'sources'")
    if config.bottom_layer_removal_interval < 0:
        raise ValueError("bottom_layer_removal_interval must be non-negative")
    if config.initial_fill_mode not in {"none", "random", "structured"}:
        raise ValueError("initial_fill_mode must be 'none', 'random', or 'structured'")
    if config.initial_fill_mean_height < 0:
        raise ValueError("initial_fill_mean_height must be non-negative")
    if config.initial_fill_variation < 0:
        raise ValueError("initial_fill_variation must be non-negative")
    if config.initial_fill_smooth_passes < 0:
        raise ValueError("initial_fill_smooth_passes must be non-negative")
    if config.warmup_steps < 0:
        raise ValueError("warmup_steps must be non-negative")


def _advance_unrecorded_step(
    *,
    grid: np.ndarray,
    rng,
    config: SandpileConfig,
    sources: np.ndarray,
    absolute_step: int,
) -> None:
    _apply_deposition(grid=grid, rng=rng, config=config, sources=sources)
    _fill_to_target_mean(grid, rng, config, sources)
    topple_counts = np.zeros_like(grid)
    _relax(grid, topple_counts, config.threshold, config.max_relaxation_sweeps)
    if _should_remove_bottom_layer(config, absolute_step):
        _remove_bottom_layer(grid)


def _should_remove_bottom_layer(config: SandpileConfig, absolute_step: int) -> bool:
    return config.bottom_layer_removal_interval > 0 and (absolute_step + 1) % config.bottom_layer_removal_interval == 0


def _apply_initial_fill(grid: np.ndarray, config: SandpileConfig) -> None:
    if config.initial_fill_mean_height <= 0:
        return
    fill_rng = np.random.default_rng(config.seed + 2_000_033)
    mean_height = float(config.initial_fill_mean_height)
    variation = float(config.initial_fill_variation)
    if config.initial_fill_mode == "random":
        values = fill_rng.normal(loc=mean_height, scale=variation or 1.0, size=grid.shape)
    else:
        values = fill_rng.normal(loc=mean_height, scale=variation or 1.0, size=grid.shape)
        passes = max(1, config.initial_fill_smooth_passes)
        for _ in range(passes):
            values = _smooth_values(values)
        if variation > 0:
            values += fill_rng.normal(loc=0.0, scale=variation * 0.20, size=grid.shape)
    np.maximum(values, 0.0, out=values)
    grid[:, :] = np.rint(values).astype(np.int64)


def _smooth_values(values: np.ndarray) -> np.ndarray:
    padded = np.pad(values, 1, mode="edge")
    return (
        padded[1:-1, 1:-1] * 4.0
        + padded[:-2, 1:-1]
        + padded[2:, 1:-1]
        + padded[1:-1, :-2]
        + padded[1:-1, 2:]
    ) / 8.0


def _random_points(rng, width: int, height: int, count: int) -> np.ndarray:
    xs = rng.integers(0, width, size=count, dtype=np.int64)
    ys = rng.integers(0, height, size=count, dtype=np.int64)
    return np.column_stack((ys, xs)).astype(np.int64)


def _apply_deposition(*, grid: np.ndarray, rng, config: SandpileConfig, sources: np.ndarray) -> int:
    active_sources = rng.random(config.source_count) < config.deposition_probability
    deposition_count = int(active_sources.sum())
    if config.deposition_mode == "sources":
        _deposit(grid, sources, active_sources)
        return deposition_count
    if deposition_count:
        _deposit_points(grid, _random_points(rng, config.width, config.height, deposition_count))
    return deposition_count


def _fill_to_target_mean(
    grid: np.ndarray,
    rng,
    config: SandpileConfig,
    sources: np.ndarray,
) -> int:
    if config.target_mean_height <= 0:
        return 0
    target_mass = int(round(config.target_mean_height * config.width * config.height))
    deficit = target_mass - int(grid.sum())
    if deficit <= 0:
        return 0
    if config.target_fill_limit > 0:
        deficit = min(deficit, config.target_fill_limit)
    if config.target_fill_mode == "sources":
        source_indices = rng.integers(0, sources.shape[0], size=deficit)
        _deposit_points(grid, sources[source_indices])
        return deficit
    cell_count = config.width * config.height
    full_layers = deficit // cell_count
    remainder = deficit % cell_count
    if full_layers:
        _add_uniform_layers(grid, full_layers)
    if remainder:
        _deposit_points(grid, _random_points(rng, config.width, config.height, remainder))
    return deficit


@njit(cache=True)
def _deposit(grid, sources, active_sources):
    for index in range(sources.shape[0]):
        if active_sources[index]:
            y = sources[index, 0]
            x = sources[index, 1]
            grid[y, x] += 1


@njit(cache=True)
def _deposit_points(grid, points):
    for index in range(points.shape[0]):
        y = points[index, 0]
        x = points[index, 1]
        grid[y, x] += 1


@njit(cache=True)
def _add_uniform_layers(grid, layers: int):
    height, width = grid.shape
    for y in range(height):
        for x in range(width):
            grid[y, x] += layers


@njit(cache=True)
def _relax(grid, topple_counts, threshold: int, max_sweeps: int):
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
                moved = 0
                if y > 0:
                    moved += _move_downhill(grid, delta, y, x, y - 1, x, threshold, available - moved)
                if y < height - 1:
                    moved += _move_downhill(grid, delta, y, x, y + 1, x, threshold, available - moved)
                if x > 0:
                    moved += _move_downhill(grid, delta, y, x, y, x - 1, threshold, available - moved)
                if x < width - 1:
                    moved += _move_downhill(grid, delta, y, x, y, x + 1, threshold, available - moved)
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

    unstable_cell_count = 0
    safety_released_mass = 0
    if relaxation_converged == 0:
        unstable_cell_count = _count_unstable(grid, threshold)
        if unstable_cell_count == 0:
            relaxation_converged = 1
        else:
            unstable_cell_count, safety_released_mass = _drain_unstable(grid, threshold)
            released_mass += safety_released_mass
            if in_avalanche:
                avalanche_count = 1
    return (
        topple_count,
        released_mass,
        avalanche_count,
        relaxation_converged,
        unstable_cell_count,
        safety_released_mass,
    )


@njit(cache=True)
def _move_downhill(grid, delta, y: int, x: int, ny: int, nx: int, threshold: int, available: int) -> int:
    if available <= 0:
        return 0
    difference = grid[y, x] - grid[ny, nx]
    if difference < threshold:
        return 0
    transfer = ((difference - threshold) // 2) + 1
    if transfer > available:
        transfer = available
    delta[y, x] -= transfer
    delta[ny, nx] += transfer
    return transfer


@njit(cache=True)
def _count_unstable(grid, threshold: int) -> int:
    height, width = grid.shape
    count = 0
    for y in range(height):
        for x in range(width):
            unstable = False
            if y > 0 and grid[y, x] - grid[y - 1, x] >= threshold:
                unstable = True
            if y < height - 1 and grid[y, x] - grid[y + 1, x] >= threshold:
                unstable = True
            if x > 0 and grid[y, x] - grid[y, x - 1] >= threshold:
                unstable = True
            if x < width - 1 and grid[y, x] - grid[y, x + 1] >= threshold:
                unstable = True
            if unstable:
                count += 1
    return count


@njit(cache=True)
def _drain_unstable(grid, threshold: int):
    height, width = grid.shape
    released = 0
    initial_count = _count_unstable(grid, threshold)
    max_passes = height + width + threshold
    for _ in range(max_passes):
        changed = False
        for y in range(height):
            for x in range(width):
                allowed = grid[y, x]
                if y > 0 and grid[y, x] - grid[y - 1, x] >= threshold:
                    allowed = min(allowed, grid[y - 1, x] + threshold - 1)
                if y < height - 1 and grid[y, x] - grid[y + 1, x] >= threshold:
                    allowed = min(allowed, grid[y + 1, x] + threshold - 1)
                if x > 0 and grid[y, x] - grid[y, x - 1] >= threshold:
                    allowed = min(allowed, grid[y, x - 1] + threshold - 1)
                if x < width - 1 and grid[y, x] - grid[y, x + 1] >= threshold:
                    allowed = min(allowed, grid[y, x + 1] + threshold - 1)
                if allowed < grid[y, x]:
                    released += grid[y, x] - allowed
                    grid[y, x] = allowed
                    changed = True
        if not changed or _count_unstable(grid, threshold) == 0:
            break
    return initial_count, released


@njit(cache=True)
def _remove_bottom_layer(grid):
    height, width = grid.shape
    removed = 0
    for y in range(height):
        for x in range(width):
            if grid[y, x] > 0:
                grid[y, x] -= 1
                removed += 1
    return removed


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

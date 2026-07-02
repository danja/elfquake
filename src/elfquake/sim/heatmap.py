"""Heatmap rendering for sandpile grid snapshots."""

from __future__ import annotations

import csv
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np

HeatmapProgressCallback = Callable[[int, int, dict[str, str]], None]


def render_sandpile_heatmap(
    *,
    snapshot_path: Path,
    out_path: Path,
    scale: int = 8,
    color_min: float = 0.0,
    color_max: float | None = None,
    gamma: float = 1.0,
) -> dict[str, str]:
    if scale < 1:
        raise ValueError("scale must be at least 1")
    if color_max is not None and color_max <= color_min:
        raise ValueError("color_max must be greater than color_min")
    if gamma <= 0:
        raise ValueError("gamma must be positive")
    try:
        from PIL import Image
    except ImportError as error:  # pragma: no cover - depends on optional environment.
        raise RuntimeError("Pillow is required for sandpile heatmap rendering") from error

    grid = np.load(snapshot_path)
    if grid.ndim != 2:
        raise ValueError("snapshot must be a 2D grid")
    image = Image.fromarray(_grid_to_rgb(grid, color_min=color_min, color_max=color_max, gamma=gamma), mode="RGB")
    if scale > 1:
        image = image.resize((image.width * scale, image.height * scale), Image.Resampling.NEAREST)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(out_path)
    return {
        "snapshot_file": str(snapshot_path),
        "heatmap_file": str(out_path),
        "width_px": str(image.width),
        "height_px": str(image.height),
        "grid_width": str(grid.shape[1]),
        "grid_height": str(grid.shape[0]),
        "max_height": str(int(grid.max())) if grid.size else "0",
        "color_min": str(color_min),
        "color_max": str(color_max) if color_max is not None else "auto",
        "gamma": str(gamma),
    }


def render_sandpile_heatmaps_from_manifest(
    *,
    manifest_path: Path,
    out_dir: Path,
    scale: int = 8,
    color_min: float = 0.0,
    color_max: float | None = None,
    gamma: float = 1.0,
    workers: int = 1,
    progress_interval: int = 0,
    progress_callback: HeatmapProgressCallback | None = None,
) -> list[dict[str, str]]:
    if workers < 1:
        raise ValueError("workers must be at least 1")
    if progress_interval < 0:
        raise ValueError("progress_interval must be non-negative")

    jobs = []
    with manifest_path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            snapshot_path = Path(row["snapshot_file"])
            out_path = out_dir / f"{snapshot_path.stem}.png"
            jobs.append((snapshot_path, out_path))

    rows: list[dict[str, str] | None] = [None] * len(jobs)

    def render(index: int, snapshot_path: Path, out_path: Path) -> tuple[int, dict[str, str]]:
        return (
            index,
            render_sandpile_heatmap(
                snapshot_path=snapshot_path,
                out_path=out_path,
                scale=scale,
                color_min=color_min,
                color_max=color_max,
                gamma=gamma,
            ),
        )

    completed = 0
    total = len(jobs)

    def report(row: dict[str, str]) -> None:
        if progress_callback and progress_interval and (completed == total or completed % progress_interval == 0):
            progress_callback(completed, total, row)

    if workers == 1:
        for index, (snapshot_path, out_path) in enumerate(jobs):
            _, row = render(index, snapshot_path, out_path)
            rows[index] = row
            completed += 1
            report(row)
    else:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(render, index, snapshot_path, out_path): index
                for index, (snapshot_path, out_path) in enumerate(jobs)
            }
            for future in as_completed(futures):
                index, row = future.result()
                rows[index] = row
                completed += 1
                report(row)
    return [row for row in rows if row is not None]


def _grid_to_rgb(
    grid: np.ndarray,
    *,
    color_min: float = 0.0,
    color_max: float | None = None,
    gamma: float = 1.0,
) -> np.ndarray:
    maximum = color_max if color_max is not None else float(grid.max()) if grid.size else color_min
    if maximum <= color_min:
        normalized = np.zeros_like(grid, dtype=np.float64)
    else:
        normalized = np.clip((grid.astype(np.float64) - color_min) / (maximum - color_min), 0.0, 1.0)
    if gamma != 1.0:
        normalized = np.power(normalized, gamma)

    stop_values = np.array([stop[0] for stop in _COLOR_STOPS], dtype=np.float64)
    stop_colors = np.array([stop[1] for stop in _COLOR_STOPS], dtype=np.float64)
    channels = [
        np.interp(normalized, stop_values, stop_colors[:, channel])
        for channel in range(3)
    ]
    return np.stack(channels, axis=-1).round().astype(np.uint8)


_COLOR_STOPS = [
    (0.00, (6, 10, 24)),
    (0.08, (22, 43, 86)),
    (0.16, (24, 87, 132)),
    (0.24, (20, 126, 142)),
    (0.34, (42, 157, 111)),
    (0.44, (102, 176, 84)),
    (0.54, (178, 190, 68)),
    (0.64, (224, 173, 54)),
    (0.74, (232, 124, 45)),
    (0.84, (205, 72, 51)),
    (0.92, (157, 45, 73)),
    (1.00, (246, 238, 210)),
]

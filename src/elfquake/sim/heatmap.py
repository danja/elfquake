"""Heatmap rendering for sandpile grid snapshots."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np


def render_sandpile_heatmap(
    *,
    snapshot_path: Path,
    out_path: Path,
    scale: int = 8,
    color_max: float | None = None,
) -> dict[str, str]:
    if scale < 1:
        raise ValueError("scale must be at least 1")
    if color_max is not None and color_max <= 0:
        raise ValueError("color_max must be positive")
    try:
        from PIL import Image
    except ImportError as error:  # pragma: no cover - depends on optional environment.
        raise RuntimeError("Pillow is required for sandpile heatmap rendering") from error

    grid = np.load(snapshot_path)
    if grid.ndim != 2:
        raise ValueError("snapshot must be a 2D grid")
    image = Image.fromarray(_grid_to_rgb(grid, color_max=color_max), mode="RGB")
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
        "color_max": str(color_max) if color_max is not None else "auto",
    }


def render_sandpile_heatmaps_from_manifest(
    *,
    manifest_path: Path,
    out_dir: Path,
    scale: int = 8,
    color_max: float | None = None,
) -> list[dict[str, str]]:
    rows = []
    with manifest_path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            snapshot_path = Path(row["snapshot_file"])
            out_path = out_dir / f"{snapshot_path.stem}.png"
            rows.append(
                render_sandpile_heatmap(
                    snapshot_path=snapshot_path,
                    out_path=out_path,
                    scale=scale,
                    color_max=color_max,
                )
            )
    return rows


def _grid_to_rgb(grid: np.ndarray, *, color_max: float | None = None) -> np.ndarray:
    maximum = color_max if color_max is not None else int(grid.max()) if grid.size else 0
    if maximum <= 0:
        normalized = np.zeros_like(grid, dtype=np.float64)
    else:
        normalized = np.clip(grid.astype(np.float64) / maximum, 0.0, 1.0)
    rgb = np.zeros((grid.shape[0], grid.shape[1], 3), dtype=np.uint8)
    for y in range(grid.shape[0]):
        for x in range(grid.shape[1]):
            rgb[y, x] = _color(float(normalized[y, x]))
    return rgb


def _color(value: float) -> tuple[int, int, int]:
    stops = [
        (0.00, (12, 24, 36)),
        (0.25, (32, 92, 122)),
        (0.50, (68, 154, 112)),
        (0.75, (222, 170, 56)),
        (1.00, (238, 82, 54)),
    ]
    for index in range(len(stops) - 1):
        left_value, left_color = stops[index]
        right_value, right_color = stops[index + 1]
        if value <= right_value:
            span = right_value - left_value
            ratio = 0.0 if span == 0 else (value - left_value) / span
            return tuple(
                int(round(left_color[channel] + ratio * (right_color[channel] - left_color[channel])))
                for channel in range(3)
            )
    return stops[-1][1]

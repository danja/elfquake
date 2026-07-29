"""Compact spatial summaries of post-relaxation avalanche activity."""

from __future__ import annotations

import numpy as np


AVALANCHE_REGION_FIELDS = ["step", "region_x", "region_y", "active_cell_count", "topple_count"]


def build_avalanche_region_rows(*, step: int, topple_counts: np.ndarray, region_count: int) -> list[dict[str, str]]:
    """Aggregate topples into a fixed square regional grid."""
    if region_count < 1:
        raise ValueError("region_count must be positive")
    height, width = topple_counts.shape
    rows = []
    for region_y in range(region_count):
        y0 = region_y * height // region_count
        y1 = (region_y + 1) * height // region_count
        for region_x in range(region_count):
            x0 = region_x * width // region_count
            x1 = (region_x + 1) * width // region_count
            block = topple_counts[y0:y1, x0:x1]
            rows.append({
                "step": str(step),
                "region_x": str(region_x),
                "region_y": str(region_y),
                "active_cell_count": str(int(np.count_nonzero(block))),
                "topple_count": str(int(block.sum())),
            })
    return rows

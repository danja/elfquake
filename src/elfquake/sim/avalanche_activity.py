"""Direct avalanche activity summaries derived from sandpile toppling."""

from __future__ import annotations

import numpy as np


AVALANCHE_ACTIVITY_FIELDS = [
    "step",
    "active_topple_cell_count",
    "topple_count",
    "centroid_x",
    "centroid_y",
    "weighted_centroid_x",
    "weighted_centroid_y",
    "min_x",
    "max_x",
    "min_y",
    "max_y",
    "peak_x",
    "peak_y",
    "peak_topple_count",
]


def build_avalanche_activity_row(*, step: int, topple_counts: np.ndarray) -> dict[str, str]:
    """Summarize the rupture footprint for one simulation step."""
    active_y, active_x = np.nonzero(topple_counts)
    active_count = int(active_x.size)
    total_topples = int(topple_counts.sum())
    if active_count == 0:
        return {
            "step": str(step),
            "active_topple_cell_count": "0",
            "topple_count": "0",
            "centroid_x": "",
            "centroid_y": "",
            "weighted_centroid_x": "",
            "weighted_centroid_y": "",
            "min_x": "",
            "max_x": "",
            "min_y": "",
            "max_y": "",
            "peak_x": "",
            "peak_y": "",
            "peak_topple_count": "0",
        }

    weights = topple_counts[active_y, active_x].astype(np.float64)
    peak_index = int(np.argmax(weights))
    weight_total = float(weights.sum())
    weighted_x = float((active_x.astype(np.float64) * weights).sum() / weight_total)
    weighted_y = float((active_y.astype(np.float64) * weights).sum() / weight_total)
    return {
        "step": str(step),
        "active_topple_cell_count": str(active_count),
        "topple_count": str(total_topples),
        "centroid_x": f"{float(active_x.mean()):.6f}",
        "centroid_y": f"{float(active_y.mean()):.6f}",
        "weighted_centroid_x": f"{weighted_x:.6f}",
        "weighted_centroid_y": f"{weighted_y:.6f}",
        "min_x": str(int(active_x.min())),
        "max_x": str(int(active_x.max())),
        "min_y": str(int(active_y.min())),
        "max_y": str(int(active_y.max())),
        "peak_x": str(int(active_x[peak_index])),
        "peak_y": str(int(active_y[peak_index])),
        "peak_topple_count": str(int(weights[peak_index])),
    }

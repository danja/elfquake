"""Map a held-out real week against spatial-cell model predictions."""

from __future__ import annotations

import csv
import json
import math
import os
from pathlib import Path

from elfquake.visualization.event_map import DEFAULT_BASEMAP_GEOJSON, ITALY_BOUNDS, _draw_base_map, _draw_labels, _magnitude_marker_size


def render_transfer_trial_map(*, actual_csv: Path, predictions_csv: Path, out_path: Path, metadata_out: Path | None = None, title: str = "ELFQuake held-out Italy week: actual vs predicted cells") -> dict[str, object]:
    actual = _read(actual_csv)
    predicted = _read(predictions_csv)
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Polygon
    fig, ax = plt.subplots(figsize=(8, 8), dpi=160)
    fig.patch.set_facecolor("#dcecf4")
    ax.set_facecolor("#cfe5ef")
    map_type = _draw_base_map(ax, Polygon, basemap_geojson=DEFAULT_BASEMAP_GEOJSON, lon_min=ITALY_BOUNDS[0], lon_max=ITALY_BOUNDS[1], lat_min=ITALY_BOUNDS[2], lat_max=ITALY_BOUNDS[3])
    _scatter(ax, actual, "#2166d1", "#0b3478", "o")
    _scatter(ax, predicted, "#d73027", "#7f0000", "X")
    _draw_labels(ax)
    ax.legend(handles=[Line2D([], [], marker="o", color="w", markerfacecolor="#2166d1", markeredgecolor="#0b3478", label="actual M>=threshold"), Line2D([], [], marker="X", color="w", markerfacecolor="#d73027", markeredgecolor="#7f0000", label="predicted spatial cell")], loc="lower left", fontsize=8, frameon=True)
    ax.set(xlim=ITALY_BOUNDS[:2], ylim=ITALY_BOUNDS[2:], xlabel="longitude", ylabel="latitude", title=title)
    ax.set_aspect(1.0 / math.cos(math.radians(41.4)))
    ax.grid(color="white", linewidth=0.6, alpha=0.6)
    fig.tight_layout(); out_path.parent.mkdir(parents=True, exist_ok=True); fig.savefig(out_path); plt.close(fig)
    report = {"map_file": str(out_path), "actual_event_count": len(actual), "predicted_cell_count": len(predicted), "map_type": map_type, "note": "Predictions are independently located fixed-cell centers, not matched actual events."}
    if metadata_out: metadata_out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def _read(path: Path) -> list[dict[str, float]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return [{"latitude": float(row["latitude"]), "longitude": float(row["longitude"]), "magnitude": float(row["magnitude"])} for row in csv.DictReader(handle)]


def _scatter(ax, rows: list[dict[str, float]], color: str, edge: str, marker: str) -> None:
    if not rows: return
    magnitudes = [row["magnitude"] for row in rows]
    low, high = min(magnitudes), max(magnitudes)
    ax.scatter([row["longitude"] for row in rows], [row["latitude"] for row in rows], s=[_magnitude_marker_size(row["magnitude"], min_magnitude=low, max_magnitude=high) for row in rows], marker=marker, c=color, edgecolors=edge, linewidths=0.7, alpha=0.78, zorder=10)

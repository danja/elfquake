"""Causal diagnostics linking localized stress releases to later avalanches."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np


ALIGNMENT_FIELDS = [
    "release_step",
    "source_id",
    "source_x",
    "source_y",
    "release_mass",
    "baseline_topple_count",
    "local_peak_topple_count",
    "global_peak_topple_count",
    "local_peak_lag",
    "global_peak_lag",
    "local_excess_auc",
    "global_excess_auc",
]


def analyze_source_stress_alignment(
    *,
    source_stress_csv: Path,
    activity_csv: Path,
    out_path: Path,
    local_radius: float = 32.0,
    response_horizon: int = 120,
    baseline_decay: float = 0.99,
) -> list[dict[str, str]]:
    """Measure post-release local/global activity using a pre-release baseline.

    The baseline is updated only from activity at or before each release. Future
    activity is used only to measure the response, preventing look-ahead in the
    trigger score while retaining the intended lead/lag diagnostic.
    """
    if local_radius < 0:
        raise ValueError("local_radius must be non-negative")
    if response_horizon < 1:
        raise ValueError("response_horizon must be positive")
    if not 0 < baseline_decay <= 1:
        raise ValueError("baseline_decay must be in (0, 1]")
    releases = _read_rows(source_stress_csv)
    activity = _read_rows(activity_csv)
    if not releases:
        _write_rows(out_path, [])
        return []
    activity_by_step = {int(row["step"]): row for row in activity}
    max_step = max(activity_by_step, default=-1)
    baseline = 0.0
    rows = []
    for release in releases:
        step = int(release["step"])
        for prior_step in range(0, step + 1):
            row = activity_by_step.get(prior_step)
            if row is not None:
                baseline = baseline_decay * baseline + (1.0 - baseline_decay) * float(row["topple_count"])
        local_values = []
        global_values = []
        for future_step in range(step, min(step + response_horizon + 1, max_step + 1)):
            row = activity_by_step.get(future_step)
            if row is None:
                continue
            global_value = float(row["topple_count"])
            global_values.append(global_value)
            if row["weighted_centroid_x"] and row["weighted_centroid_y"]:
                distance = float(np.hypot(
                    float(row["weighted_centroid_x"]) - float(release["x"]),
                    float(row["weighted_centroid_y"]) - float(release["y"]),
                ))
                if distance <= local_radius:
                    local_values.append((future_step, global_value))
        local_peak_step, local_peak = _peak(local_values)
        global_peak_index = int(np.argmax(global_values)) if global_values else 0
        global_peak = global_values[global_peak_index] if global_values else 0.0
        rows.append({
            "release_step": str(step),
            "source_id": release["source_id"],
            "source_x": release["x"],
            "source_y": release["y"],
            "release_mass": release["release_mass"],
            "baseline_topple_count": f"{baseline:.6f}",
            "local_peak_topple_count": f"{local_peak:.6f}",
            "global_peak_topple_count": f"{global_peak:.6f}",
            "local_peak_lag": str(local_peak_step - step) if local_peak_step is not None else "",
            "global_peak_lag": str(global_peak_index),
            "local_excess_auc": f"{sum(max(0.0, value - baseline) for _, value in local_values):.6f}",
            "global_excess_auc": f"{sum(max(0.0, value - baseline) for value in global_values):.6f}",
        })
    _write_rows(out_path, rows)
    return rows


def _peak(values: list[tuple[int, float]]) -> tuple[int | None, float]:
    if not values:
        return None, 0.0
    step, value = max(values, key=lambda item: item[1])
    return step, value


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _write_rows(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=ALIGNMENT_FIELDS)
        writer.writeheader()
        writer.writerows(rows)

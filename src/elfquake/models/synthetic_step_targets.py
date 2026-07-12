"""Minute-scale synthetic event targets aligned to simulation sequence time axes."""

from __future__ import annotations

import csv
import json
import re
from datetime import timedelta
from pathlib import Path

from elfquake.features.common import format_utc, parse_utc


def build_synthetic_step_targets(
    *,
    piezo_paths: list[Path],
    event_paths: list[Path],
    out_path: Path,
    report_path: Path,
    horizon_steps: int = 15,
    stride_steps: int = 5,
    start_time_utc: str = "2026-01-01T00:00:00Z",
    step_seconds: int = 60,
) -> dict[str, object]:
    if len(piezo_paths) != len(event_paths) or not piezo_paths:
        raise ValueError("piezo and event paths must be nonempty paired lists")
    if horizon_steps < 1 or stride_steps < 1 or step_seconds < 1:
        raise ValueError("horizon_steps, stride_steps, and step_seconds must be positive")
    start = parse_utc(start_time_utc)
    rows = []
    for piezo_path, event_path in zip(piezo_paths, event_paths):
        step_count = _step_count(piezo_path)
        event_steps = _event_steps(event_path, step_count)
        dataset_id = _dataset_id(piezo_path, event_path)
        for step in range(0, step_count - horizon_steps, stride_steps):
            end_step = step + horizon_steps
            occurred = int(any(step < event_step <= end_step for event_step in event_steps))
            window_start = start + timedelta(seconds=step * step_seconds)
            window_end = start + timedelta(seconds=end_step * step_seconds)
            rows.append({
                "dataset_id": dataset_id,
                "window_start_utc": format_utc(window_start),
                "window_end_utc": format_utc(window_end),
                "eventlist_target_status": "labeled",
                "eventlist_target_occurred": str(occurred),
                "horizon_steps": str(horizon_steps),
                "stride_steps": str(stride_steps),
            })
    fieldnames = list(rows[0]) if rows else []
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    labeled = [row for row in rows if row["eventlist_target_status"] == "labeled"]
    report = {
        "schema": "elfquake.synthetic_step_targets.v1",
        "row_count": len(rows),
        "positive_count": sum(row["eventlist_target_occurred"] == "1" for row in labeled),
        "negative_count": sum(row["eventlist_target_occurred"] == "0" for row in labeled),
        "horizon_steps": horizon_steps,
        "stride_steps": stride_steps,
        "step_seconds": step_seconds,
        "out": str(out_path),
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def _step_count(path: Path) -> int:
    with path.open(newline="", encoding="utf-8") as handle:
        steps = [int(row["step"]) for row in csv.DictReader(handle)]
    if not steps:
        raise ValueError(f"empty piezo CSV: {path}")
    return max(steps) + 1


def _event_steps(path: Path, step_count: int) -> list[int]:
    with path.open(newline="", encoding="utf-8") as handle:
        return sorted({int(row["step"]) for row in csv.DictReader(handle) if 0 <= int(row["step"]) < step_count})


def _dataset_id(piezo_path: Path, event_path: Path) -> str:
    match = re.search(r"seed\d+", f"{piezo_path} {event_path}")
    return match.group(0) if match else piezo_path.stem

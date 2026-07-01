"""Reports for synthetic sandpile simulation outputs."""

from __future__ import annotations

import csv
import json
import tempfile
import time
from pathlib import Path
from typing import Any


def summarize_sandpile_outputs(
    *,
    summary_csv: Path,
    sensors_csv: Path,
    out_path: Path,
) -> dict[str, Any]:
    summary_rows = _read_csv(summary_csv)
    sensor_rows = _read_csv(sensors_csv)
    steps = {row.get("step", "") for row in summary_rows if row.get("step", "") != ""}
    sensor_ids = {row.get("sensor_id", "") for row in sensor_rows if row.get("sensor_id", "") != ""}
    expected_sensor_rows = len(steps) * len(sensor_ids)
    sensor_rows_match_expected = len(sensor_rows) == expected_sensor_rows

    report = {
        "status": _status(summary_rows, sensor_rows, sensor_rows_match_expected),
        "summary_input": str(summary_csv),
        "sensors_input": str(sensors_csv),
        "summary_row_count": len(summary_rows),
        "sensor_row_count": len(sensor_rows),
        "step_count": len(steps),
        "sensor_count": len(sensor_ids),
        "expected_sensor_rows": expected_sensor_rows,
        "sensor_rows_match_expected": sensor_rows_match_expected,
        "total_deposition_count": _sum_int(summary_rows, "deposition_count"),
        "total_topple_count": _sum_int(summary_rows, "topple_count"),
        "total_released_mass": _sum_int(summary_rows, "released_mass"),
        "total_safety_released_mass": _sum_int(summary_rows, "safety_released_mass"),
        "total_target_fill_count": _sum_int(summary_rows, "target_fill_count"),
        "total_bottom_layer_removed_mass": _sum_int(summary_rows, "bottom_layer_removed_mass"),
        "bottom_layer_removal_step_count": sum(
            1 for row in summary_rows if _int(row.get("bottom_layer_removed_mass")) > 0
        ),
        "safety_drain_step_count": sum(1 for row in summary_rows if _int(row.get("safety_released_mass")) > 0),
        "nonconverged_step_count": sum(1 for row in summary_rows if row.get("relaxation_converged") == "0"),
        "max_unstable_cell_count": _max_int(summary_rows, "unstable_cell_count"),
        "avalanche_step_count": sum(1 for row in summary_rows if _int(row.get("avalanche_count")) > 0),
        "max_height_max": _max_int(summary_rows, "max_height"),
        "mean_height_last": _last(summary_rows, "mean_height"),
        "max_local_topple_count": _max_int(sensor_rows, "local_topple_count"),
    }
    _write_json(out_path, report)
    return report


def benchmark_sandpile_simulation(
    *,
    config: Any,
    out_path: Path,
) -> dict[str, Any]:
    from elfquake.sim.sandpile import run_sandpile_simulation

    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        summary_out = root / "summary.csv"
        sensors_out = root / "sensors.csv"
        started = time.perf_counter()
        summary_rows, sensor_rows = run_sandpile_simulation(
            config=config,
            summary_out=summary_out,
            sensors_out=sensors_out,
        )
        elapsed_seconds = time.perf_counter() - started

    report = {
        "status": "ok",
        "backend": "numba_cpu",
        "gpu_required": False,
        "includes_numba_first_call_overhead": True,
        "width": config.width,
        "height": config.height,
        "steps": config.steps,
        "threshold": config.threshold,
        "source_count": config.source_count,
        "sensor_count": config.sensor_count,
        "deposition_probability": config.deposition_probability,
        "seed": config.seed,
        "elapsed_seconds": round(elapsed_seconds, 6),
        "steps_per_second": round(config.steps / elapsed_seconds, 6) if elapsed_seconds else None,
        "summary_row_count": len(summary_rows),
        "sensor_row_count": len(sensor_rows),
    }
    _write_json(out_path, report)
    return report


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _status(
    summary_rows: list[dict[str, str]],
    sensor_rows: list[dict[str, str]],
    sensor_rows_match_expected: bool,
) -> str:
    if not summary_rows or not sensor_rows:
        return "empty_input"
    if not sensor_rows_match_expected:
        return "invalid_sensor_shape"
    return "ok"


def _sum_int(rows: list[dict[str, str]], field: str) -> int:
    return sum(_int(row.get(field)) for row in rows)


def _max_int(rows: list[dict[str, str]], field: str) -> int | None:
    values = [_int(row.get(field)) for row in rows if row.get(field, "") != ""]
    return max(values) if values else None


def _int(value: str | None) -> int:
    try:
        return int(value or "0")
    except ValueError:
        return 0


def _last(rows: list[dict[str, str]], field: str) -> str | None:
    if not rows:
        return None
    return rows[-1].get(field)

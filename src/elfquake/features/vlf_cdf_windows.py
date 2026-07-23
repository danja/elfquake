"""Align native Japan CDF features to existing UTC training windows."""

from __future__ import annotations

import csv
from pathlib import Path

from elfquake.features.common import parse_utc


def build_japan_cdf_window_features(*, feature_csvs: list[Path], windows_csv: Path, out_path: Path) -> list[dict[str, str]]:
    features = []
    for feature_csv in feature_csvs:
        features.extend(_read(feature_csv))
    windows = _read(windows_csv)
    numeric_fields = [
        field for field in (features[0].keys() if features else [])
        if field not in {"time_utc", "research_use_only"}
    ]
    output_fields = ["window_id", "window_start_utc", "window_end_utc", "region_id", "japan_vlf_row_count", "japan_vlf_coverage_seconds"]
    output_fields.extend(f"japan_{field}_{stat}" for field in numeric_fields for stat in ("mean", "std", "max"))
    output_fields.append("research_use_only")

    feature_times = sorted((parse_utc(row["time_utc"]), row) for row in features)
    rows = []
    for window in windows:
        start = parse_utc(window["window_start_utc"])
        end = parse_utc(window["window_end_utc"])
        selected = [row for timestamp, row in feature_times if start <= timestamp < end]
        result = {
            "window_id": window.get("window_id", ""),
            "window_start_utc": window["window_start_utc"],
            "window_end_utc": window["window_end_utc"],
            "region_id": window.get("region_id", "japan"),
            "japan_vlf_row_count": str(len(selected)),
            "japan_vlf_coverage_seconds": str(_coverage_seconds(selected)),
            "research_use_only": "1",
        }
        for field in numeric_fields:
            values = [_float(row.get(field, "")) for row in selected]
            values = [value for value in values if value is not None]
            result[f"japan_{field}_mean"] = _number(_mean(values))
            result[f"japan_{field}_std"] = _number(_std(values))
            result[f"japan_{field}_max"] = _number(max(values) if values else None)
        rows.append(result)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=output_fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    return rows


def _read(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _coverage_seconds(rows: list[dict[str, str]]) -> int:
    if len(rows) < 2:
        return 0
    return max(0, int((parse_utc(rows[-1]["time_utc"]) - parse_utc(rows[0]["time_utc"])).total_seconds()))


def _float(value: str) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _std(values: list[float]) -> float | None:
    if not values:
        return None
    mean = sum(values) / len(values)
    return (sum((value - mean) ** 2 for value in values) / len(values)) ** 0.5


def _number(value: float | None) -> str:
    return "" if value is None else f"{value:.8f}"

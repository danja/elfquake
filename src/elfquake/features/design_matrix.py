"""Build tabular design matrices for first model candidates."""

from __future__ import annotations

import csv
from pathlib import Path

from elfquake.features.common import parse_utc


FIELDNAMES = [
    "window_id",
    "region_id",
    "window_start_utc",
    "window_end_utc",
    "target_start_utc",
    "target_end_utc",
    "target_magnitude_min",
    "seismic_event_count",
    "seismic_max_magnitude",
    "astro_kp_mean",
    "astro_kp_max",
    "astro_ap_mean",
    "astro_ap_max",
    "astro_f107_mean",
    "quality_kp_count",
    "quality_f107_count",
    "target_event_count",
    "target_occurred",
    "target_status",
]


def build_design_matrix(
    *,
    training_windows_csv: Path,
    kp_ap_csv: Path,
    f107_csv: Path,
    out_path: Path,
) -> list[dict[str, str]]:
    training_rows = _read_rows(training_windows_csv)
    kp_rows = _read_rows(kp_ap_csv)
    f107_rows = _read_rows(f107_csv)
    design_rows = []
    for row in training_rows:
        start = parse_utc(row["window_start_utc"])
        end = parse_utc(row["window_end_utc"])
        kp_window = [
            item
            for item in kp_rows
            if start.date().isoformat() <= item.get("date", "") < end.date().isoformat()
        ]
        f107_window = [
            item
            for item in f107_rows
            if start.date().isoformat() <= item.get("date", "") < end.date().isoformat()
        ]
        kp_values = [_float(item.get("kp", "")) for item in kp_window]
        ap_values = [_float(item.get("ap", "")) for item in kp_window]
        f107_values = [_float(item.get("f107", "")) for item in f107_window]
        kp_values = [value for value in kp_values if value is not None]
        ap_values = [value for value in ap_values if value is not None]
        f107_values = [value for value in f107_values if value is not None]
        design_rows.append(
            {
                "window_id": row["window_id"],
                "region_id": row["region_id"],
                "window_start_utc": row["window_start_utc"],
                "window_end_utc": row["window_end_utc"],
                "target_start_utc": row["target_start_utc"],
                "target_end_utc": row["target_end_utc"],
                "target_magnitude_min": row["target_magnitude_min"],
                "seismic_event_count": row["seismic_event_count"],
                "seismic_max_magnitude": row["seismic_max_magnitude"],
                "astro_kp_mean": _mean(kp_values),
                "astro_kp_max": _max(kp_values),
                "astro_ap_mean": _mean(ap_values),
                "astro_ap_max": _max(ap_values),
                "astro_f107_mean": _mean(f107_values),
                "quality_kp_count": str(len(kp_values)),
                "quality_f107_count": str(len(f107_values)),
                "target_event_count": row["target_event_count"],
                "target_occurred": row["target_occurred"],
                "target_status": row["target_status"],
            }
        )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES, lineterminator="\n")
        writer.writeheader()
        writer.writerows(design_rows)
    return design_rows


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _float(value: str) -> float | None:
    try:
        return float(value)
    except ValueError:
        return None


def _mean(values: list[float]) -> str:
    if not values:
        return ""
    return f"{sum(values) / len(values):g}"


def _max(values: list[float]) -> str:
    if not values:
        return ""
    return f"{max(values):g}"

"""Join pixel-derived VLF image features onto window rows."""

from __future__ import annotations

import csv
import json
from pathlib import Path

from elfquake.features.common import parse_utc


IMAGE_WINDOW_FIELDS = [
    "vlf_image_feature_count",
    "vlf_image_latest_source_file",
    "vlf_image_intensity_mean_avg",
    "vlf_image_intensity_mean_latest",
    "vlf_image_high_intensity_ratio_max",
    "vlf_image_high_intensity_ratio_latest",
    "vlf_image_hot_color_ratio_max",
    "vlf_image_vertical_streak_count_max",
    "vlf_image_vertical_streak_count_latest",
    "vlf_image_band_0_mean_latest",
    "vlf_image_band_1_mean_latest",
    "vlf_image_band_2_mean_latest",
    "vlf_image_band_3_mean_latest",
    "vlf_image_band_4_mean_latest",
    "vlf_image_band_5_mean_latest",
    "quality_missing_vlf_image_features",
]


def join_vlf_image_features_to_windows(
    *,
    windows_csv: Path,
    image_features_csvs: list[Path],
    out_path: Path,
    include_window_end: bool = True,
) -> list[dict[str, str]]:
    window_rows, window_fields = _read_rows_and_fields(windows_csv)
    image_rows = []
    for features_csv in image_features_csvs:
        image_rows.extend(_read_image_rows(features_csv))

    fieldnames = window_fields + [field for field in IMAGE_WINDOW_FIELDS if field not in window_fields]
    output_rows = []
    for window in window_rows:
        start = parse_utc(window["window_start_utc"])
        end = parse_utc(window["window_end_utc"])
        matching = [
            row
            for row in image_rows
            if row.get("_captured_at_utc")
            and _in_window(parse_utc(row["_captured_at_utc"]), start, end, include_window_end=include_window_end)
        ]
        joined = dict(window)
        joined.update(_aggregate_image_rows(matching))
        output_rows.append({field: joined.get(field, "") for field in fieldnames})

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(output_rows)
    return output_rows


def _read_rows_and_fields(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        return list(reader), list(reader.fieldnames or [])


def _read_image_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        if row.get("vlf_image_captured_at_utc"):
            row["_captured_at_utc"] = row["vlf_image_captured_at_utc"]
            continue
        source = Path(row["vlf_image_source_file"])
        metadata_path = source.with_suffix(source.suffix + ".metadata.json")
        row["_captured_at_utc"] = _captured_at(metadata_path)
    return rows


def _captured_at(metadata_path: Path) -> str:
    if not metadata_path.exists():
        return ""
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    return metadata.get("captured_at_utc", "")


def _in_window(captured_at, start, end, *, include_window_end: bool) -> bool:
    if include_window_end and captured_at == end:
        return True
    return start <= captured_at < end


def _aggregate_image_rows(rows: list[dict[str, str]]) -> dict[str, str]:
    if not rows:
        return {
            "vlf_image_feature_count": "0",
            "quality_missing_vlf_image_features": "1",
        }
    latest = max(rows, key=lambda row: parse_utc(row["_captured_at_utc"]))
    return {
        "vlf_image_feature_count": str(len(rows)),
        "vlf_image_latest_source_file": latest["vlf_image_source_file"],
        "vlf_image_intensity_mean_avg": _fmt(_mean(rows, "vlf_intensity_mean")),
        "vlf_image_intensity_mean_latest": latest.get("vlf_intensity_mean", ""),
        "vlf_image_high_intensity_ratio_max": _fmt(_max(rows, "vlf_high_intensity_ratio")),
        "vlf_image_high_intensity_ratio_latest": latest.get("vlf_high_intensity_ratio", ""),
        "vlf_image_hot_color_ratio_max": _fmt(_max(rows, "vlf_hot_color_ratio")),
        "vlf_image_vertical_streak_count_max": _fmt(_max(rows, "vlf_vertical_streak_count")),
        "vlf_image_vertical_streak_count_latest": latest.get("vlf_vertical_streak_count", ""),
        "vlf_image_band_0_mean_latest": latest.get("vlf_band_0_mean", ""),
        "vlf_image_band_1_mean_latest": latest.get("vlf_band_1_mean", ""),
        "vlf_image_band_2_mean_latest": latest.get("vlf_band_2_mean", ""),
        "vlf_image_band_3_mean_latest": latest.get("vlf_band_3_mean", ""),
        "vlf_image_band_4_mean_latest": latest.get("vlf_band_4_mean", ""),
        "vlf_image_band_5_mean_latest": latest.get("vlf_band_5_mean", ""),
        "quality_missing_vlf_image_features": "0",
    }


def _mean(rows: list[dict[str, str]], field: str) -> float:
    values = [_float(row.get(field, "")) for row in rows]
    values = [value for value in values if value is not None]
    return sum(values) / len(values) if values else 0.0


def _max(rows: list[dict[str, str]], field: str) -> float:
    values = [_float(row.get(field, "")) for row in rows]
    values = [value for value in values if value is not None]
    return max(values) if values else 0.0


def _float(value: str) -> float | None:
    try:
        return float(value)
    except ValueError:
        return None


def _fmt(value: float) -> str:
    return f"{value:.6f}"

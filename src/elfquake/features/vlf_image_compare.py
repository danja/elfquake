"""Compare simulated VLF analogue images with captured Cumiana spectrograms."""

from __future__ import annotations

import csv
import math
from pathlib import Path

from elfquake.features.vlf_image import BAND_COUNT, extract_vlf_image_features


COMPARISON_METRICS = [
    "vlf_intensity_mean",
    "vlf_intensity_std",
    "vlf_intensity_p90",
    "vlf_intensity_p99",
    "vlf_high_intensity_ratio",
    "vlf_hot_color_ratio",
    "vlf_column_mean_max",
    "vlf_column_mean_std",
    "vlf_vertical_streak_count",
] + [f"vlf_band_{index}_mean" for index in range(BAND_COUNT)]

FIELDNAMES = [
    "sim_image_file",
    "real_image_count",
    "nearest_real_image_file",
    "nearest_real_distance",
    "mean_normalized_distance",
] + [f"sim_{metric}" for metric in COMPARISON_METRICS] + [
    f"real_mean_{metric}" for metric in COMPARISON_METRICS
] + [
    f"real_std_{metric}" for metric in COMPARISON_METRICS
]


def compare_vlf_image_features(
    *,
    sim_image: Path,
    real_images: list[Path],
    out_path: Path,
    sim_crop_left: float = 0.0,
    sim_crop_top: float = 0.13,
    sim_crop_right: float = 1.0,
    sim_crop_bottom: float = 1.0,
    real_crop_left: float = 0.0,
    real_crop_top: float = 0.13,
    real_crop_right: float = 0.83,
    real_crop_bottom: float = 0.95,
) -> dict[str, str]:
    if not real_images:
        raise ValueError("at least one real VLF image is required")

    sim = extract_vlf_image_features(
        sim_image,
        crop_left=sim_crop_left,
        crop_top=sim_crop_top,
        crop_right=sim_crop_right,
        crop_bottom=sim_crop_bottom,
    )
    real_rows = [
        extract_vlf_image_features(
            image,
            crop_left=real_crop_left,
            crop_top=real_crop_top,
            crop_right=real_crop_right,
            crop_bottom=real_crop_bottom,
        )
        for image in real_images
    ]

    real_stats = {
        metric: _mean_std([_float(row[metric]) for row in real_rows])
        for metric in COMPARISON_METRICS
    }
    distances = [
        (_image_distance(sim, row, real_stats), row["vlf_image_source_file"])
        for row in real_rows
    ]
    nearest_distance, nearest_file = min(distances, key=lambda item: item[0])
    mean_distance = sum(distance for distance, _ in distances) / len(distances)

    result = {
        "sim_image_file": str(sim_image),
        "real_image_count": str(len(real_rows)),
        "nearest_real_image_file": nearest_file,
        "nearest_real_distance": _fmt(nearest_distance),
        "mean_normalized_distance": _fmt(mean_distance),
    }
    for metric in COMPARISON_METRICS:
        mean, std = real_stats[metric]
        result[f"sim_{metric}"] = sim[metric]
        result[f"real_mean_{metric}"] = _fmt(mean)
        result[f"real_std_{metric}"] = _fmt(std)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES, lineterminator="\n")
        writer.writeheader()
        writer.writerow({field: result.get(field, "") for field in FIELDNAMES})
    return {field: result.get(field, "") for field in FIELDNAMES}


def _image_distance(
    sim: dict[str, str],
    real: dict[str, str],
    real_stats: dict[str, tuple[float, float]],
) -> float:
    total = 0.0
    count = 0
    for metric in COMPARISON_METRICS:
        real_mean, real_std = real_stats[metric]
        scale = real_std if real_std > 1e-9 else max(abs(real_mean), 1.0)
        delta = (_float(sim[metric]) - _float(real[metric])) / scale
        total += delta * delta
        count += 1
    return math.sqrt(total / count) if count else 0.0


def _mean_std(values: list[float]) -> tuple[float, float]:
    mean = sum(values) / len(values)
    variance = sum((value - mean) ** 2 for value in values) / len(values)
    return mean, math.sqrt(variance)


def _float(value: str) -> float:
    try:
        return float(value)
    except ValueError:
        return 0.0


def _fmt(value: float) -> str:
    return f"{value:.6f}"

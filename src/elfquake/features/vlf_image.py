"""Pixel-derived features from Cumiana VLF spectrogram images."""

from __future__ import annotations

import csv
import math
from pathlib import Path


BAND_COUNT = 6

FIELDNAMES = [
    "vlf_image_source_file",
    "vlf_image_width_px",
    "vlf_image_height_px",
    "vlf_crop_left_px",
    "vlf_crop_top_px",
    "vlf_crop_right_px",
    "vlf_crop_bottom_px",
    "vlf_crop_width_px",
    "vlf_crop_height_px",
    "vlf_pixel_count",
    "vlf_intensity_mean",
    "vlf_intensity_std",
    "vlf_intensity_p50",
    "vlf_intensity_p90",
    "vlf_intensity_p99",
    "vlf_high_intensity_ratio",
    "vlf_hot_color_ratio",
    "vlf_column_mean_max",
    "vlf_column_mean_std",
    "vlf_vertical_streak_count",
] + [f"vlf_band_{index}_mean" for index in range(BAND_COUNT)]


def build_vlf_image_features(
    *,
    image_paths: list[Path],
    out_path: Path,
    crop_left: float = 0.0,
    crop_top: float = 0.13,
    crop_right: float = 0.83,
    crop_bottom: float = 0.95,
) -> list[dict[str, str]]:
    rows = [
        extract_vlf_image_features(
            image_path,
            crop_left=crop_left,
            crop_top=crop_top,
            crop_right=crop_right,
            crop_bottom=crop_bottom,
        )
        for image_path in image_paths
    ]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    return rows


def extract_vlf_image_features(
    image_path: Path,
    *,
    crop_left: float = 0.0,
    crop_top: float = 0.13,
    crop_right: float = 0.83,
    crop_bottom: float = 0.95,
) -> dict[str, str]:
    try:
        from PIL import Image
    except ImportError as error:
        raise RuntimeError("Pillow is required for VLF image feature extraction") from error

    with Image.open(image_path) as image:
        rgb = image.convert("RGB")
        width, height = rgb.size
        left, top, right, bottom = _crop_box(
            width,
            height,
            crop_left=crop_left,
            crop_top=crop_top,
            crop_right=crop_right,
            crop_bottom=crop_bottom,
        )
        crop = rgb.crop((left, top, right, bottom))
        features = _summarize_crop(crop)

    row = {
        "vlf_image_source_file": str(image_path),
        "vlf_image_width_px": str(width),
        "vlf_image_height_px": str(height),
        "vlf_crop_left_px": str(left),
        "vlf_crop_top_px": str(top),
        "vlf_crop_right_px": str(right),
        "vlf_crop_bottom_px": str(bottom),
        **features,
    }
    return {field: row.get(field, "") for field in FIELDNAMES}


def _crop_box(
    width: int,
    height: int,
    *,
    crop_left: float,
    crop_top: float,
    crop_right: float,
    crop_bottom: float,
) -> tuple[int, int, int, int]:
    if not (0 <= crop_left < crop_right <= 1 and 0 <= crop_top < crop_bottom <= 1):
        raise ValueError("crop ratios must satisfy 0 <= left < right <= 1 and 0 <= top < bottom <= 1")
    left = int(width * crop_left)
    top = int(height * crop_top)
    right = max(left + 1, int(width * crop_right))
    bottom = max(top + 1, int(height * crop_bottom))
    return left, top, right, bottom


def _summarize_crop(crop) -> dict[str, str]:
    width, height = crop.size
    pixels = crop.load()
    pixel_count = width * height
    intensities: list[float] = []
    column_sums = [0.0] * width
    column_high_counts = [0] * width
    band_sums = [0.0] * BAND_COUNT
    band_counts = [0] * BAND_COUNT
    high_count = 0
    hot_count = 0

    for y in range(height):
        band_index = min(BAND_COUNT - 1, int(y * BAND_COUNT / height))
        for x in range(width):
            red, green, blue = pixels[x, y]
            intensity = (0.2126 * red + 0.7152 * green + 0.0722 * blue) / 255
            intensities.append(intensity)
            column_sums[x] += intensity
            band_sums[band_index] += intensity
            band_counts[band_index] += 1
            if intensity >= 0.75:
                high_count += 1
                column_high_counts[x] += 1
            if red >= 180 and green >= 80 and blue <= 120:
                hot_count += 1

    mean = sum(intensities) / pixel_count
    std = _std(intensities, mean)
    column_means = [value / height for value in column_sums]
    column_mean = sum(column_means) / len(column_means)
    column_std = _std(column_means, column_mean)
    column_high_ratios = [value / height for value in column_high_counts]
    streak_count = _count_runs([value >= 0.08 for value in column_high_ratios])

    row = {
        "vlf_crop_width_px": str(width),
        "vlf_crop_height_px": str(height),
        "vlf_pixel_count": str(pixel_count),
        "vlf_intensity_mean": _fmt(mean),
        "vlf_intensity_std": _fmt(std),
        "vlf_intensity_p50": _fmt(_quantile(intensities, 0.50)),
        "vlf_intensity_p90": _fmt(_quantile(intensities, 0.90)),
        "vlf_intensity_p99": _fmt(_quantile(intensities, 0.99)),
        "vlf_high_intensity_ratio": _fmt(high_count / pixel_count),
        "vlf_hot_color_ratio": _fmt(hot_count / pixel_count),
        "vlf_column_mean_max": _fmt(max(column_means)),
        "vlf_column_mean_std": _fmt(column_std),
        "vlf_vertical_streak_count": str(streak_count),
    }
    for index in range(BAND_COUNT):
        row[f"vlf_band_{index}_mean"] = _fmt(band_sums[index] / band_counts[index])
    return row


def _quantile(values: list[float], quantile: float) -> float:
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, int(round((len(ordered) - 1) * quantile))))
    return ordered[index]


def _std(values: list[float], mean: float) -> float:
    return math.sqrt(sum((value - mean) ** 2 for value in values) / len(values))


def _count_runs(flags: list[bool]) -> int:
    count = 0
    in_run = False
    for flag in flags:
        if flag and not in_run:
            count += 1
            in_run = True
        elif not flag:
            in_run = False
    return count


def _fmt(value: float) -> str:
    return f"{value:.6f}"

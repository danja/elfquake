"""Colour-scale (palette) diagnostics for Cumiana VLF spectrogram captures.

Cumiana `last_E_VLF` images embed their own colourbar above the spectrogram,
with a fixed `-100 dB .. 0 dB` tick ruler beneath it. The colour ramp inside
that bar is a receiver-software setting and can be changed independently of
the ruler. When it changes, a pixel of a given colour means a different dB
before and after the change, so pixel-derived image features are no longer on
a common scale and must not be pooled across the change.

This module reads the colourbar out of each capture, groups captures into
palette variants, reports change points, and -- when the palette did change --
decodes spectrogram pixels back to absolute dB through each image's own
colourbar so the two sides can be compared on a common axis.

Two limits are reported rather than hidden:

* The palette saturates. Everything below the ramp's black floor and above its
  solid-red top is censored, so only the dB window resolvable in *both*
  variants supports a comparison.
* The nearest-colour inversion is imperfect on JPEG-compressed pixels and on
  overlaid gridlines and annotations. Pixels whose distance to the closest
  palette colour exceeds a threshold are masked out and counted.
"""

from __future__ import annotations

import csv
import json
import statistics
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from elfquake.features.vlf_image import CAPTURE_TIMESTAMP_PATTERN


# Expected capture geometry. Anything else is refused rather than guessed at,
# because every constant below is a pixel offset into this exact layout.
EXPECTED_WIDTH = 842
EXPECTED_HEIGHT = 573

# Embedded colourbar: 96 ramp columns, sampled over rows clear of its border.
COLOURBAR_Y0 = 6
COLOURBAR_Y1 = 20
COLOURBAR_X0 = 328
COLOURBAR_X1 = 424
COLOURBAR_STEPS = COLOURBAR_X1 - COLOURBAR_X0

# Fixed dB ruler printed beneath the bar. Unchanged across all known captures.
SCALE_DB_LOW = -100.0
SCALE_DB_HIGH = 0.0
DB_PER_STEP = (SCALE_DB_HIGH - SCALE_DB_LOW) / (COLOURBAR_STEPS - 1)

# Ramp landmark thresholds, in 0-255 RGB.
BLACK_MAX_CHANNEL = 20
RED_MIN_R = 230
RED_MAX_G = 45
RED_MAX_B = 45
RED_RUN_LENGTH = 4

# Upper spectrogram panel. The lower panel is a separate sub-1500 Hz zoom with
# its own axis and is deliberately excluded.
PANEL_Y0 = 80
PANEL_Y1 = 401
PANEL_X0 = 0
PANEL_X1 = 699

# Frequency axis fit from the right-hand tick rows: y=90 is 14000 Hz and
# y=379 is 1000 Hz, linear between.
TICK_Y_AT_14KHZ = 90.0
TICK_Y_AT_1KHZ = 379.0
PX_PER_KHZ = (TICK_Y_AT_1KHZ - TICK_Y_AT_14KHZ) / 13.0

# Rightmost columns hold the most recent sweep. Sampling only these keeps
# successive captures close to independent instead of re-reading scrollback.
FRESH_X0 = 654
FRESH_X1 = 699

# Euclidean RGB distance beyond which a pixel is not credibly a palette colour.
DEFAULT_RESIDUAL_LIMIT = 30.0

# Frequency bands in kHz, coarse at the bottom because the upper panel is
# linear in frequency and resolves sub-kHz content in very few rows.
DEFAULT_BANDS_KHZ = (
    (12.0, 15.0),
    (9.0, 12.0),
    (6.0, 9.0),
    (4.0, 6.0),
    (2.5, 4.0),
    (1.5, 2.5),
    (0.7, 1.5),
    (0.2, 0.7),
)

DEFAULT_HOUR_LOW = 11
DEFAULT_HOUR_HIGH = 13

CAPTURE_FIELDNAMES = [
    "source_file",
    "captured_at_utc",
    "palette_variant_id",
    "black_end_px",
    "red_start_px",
    "display_db_low",
    "display_db_high",
]

BAND_FIELDNAMES = [
    "band_low_khz",
    "band_high_khz",
    "row_count",
    "early_variant_id",
    "early_median_db",
    "early_sample_count",
    "late_variant_id",
    "late_median_db",
    "late_sample_count",
    "delta_db",
    "resolvable_in_both",
]


def diagnose_vlf_palette_shift(
    *,
    image_paths: list[Path],
    out_path: Path,
    capture_csv_path: Path | None = None,
    band_csv_path: Path | None = None,
    hour_low: int = DEFAULT_HOUR_LOW,
    hour_high: int = DEFAULT_HOUR_HIGH,
    residual_limit: float = DEFAULT_RESIDUAL_LIMIT,
    max_captures_per_variant: int = 200,
) -> dict[str, object]:
    """Read every capture's colourbar, group variants, and compare in dB."""
    captures = []
    skipped = []
    for image_path in sorted(image_paths):
        record = _read_capture_palette(image_path)
        if record is None:
            skipped.append(str(image_path))
            continue
        captures.append(record)
    captures.sort(key=lambda record: record["captured_at_utc"])

    report: dict[str, object] = {
        "schema": "elfquake.vlf_palette_shift.v1",
        "generated_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "capture_count": len(captures),
        "skipped_capture_count": len(skipped),
        "scale_db_low": SCALE_DB_LOW,
        "scale_db_high": SCALE_DB_HIGH,
        "residual_limit": residual_limit,
    }

    if len(captures) < 2:
        report["status"] = "insufficient_captures"
        report["palette_variants"] = []
        report["change_points"] = []
        _write_report(report, out_path)
        return report

    variants = _group_palette_variants(captures)
    report["palette_variants"] = [
        {key: value for key, value in variant.items() if key != "captures"}
        for variant in variants
    ]
    report["change_points"] = _find_change_points(captures)

    if len(variants) < 2:
        report["status"] = "palette_stable"
        _write_capture_csv(captures, capture_csv_path)
        _write_report(report, out_path)
        return report

    report["status"] = "palette_changed"
    early = variants[0]
    late = variants[-1]
    report["common_resolvable_db"] = {
        "low": max(early["display_db_low"], late["display_db_low"]),
        "high": min(early["display_db_high"], late["display_db_high"]),
    }
    report["ramp_shift_px"] = late["red_start_px"] - early["red_start_px"]
    report["display_db_shift"] = round(
        (late["red_start_px"] - early["red_start_px"]) * DB_PER_STEP, 3
    )
    report["comparison"] = _compare_variants_in_db(
        early=early,
        late=late,
        hour_low=hour_low,
        hour_high=hour_high,
        residual_limit=residual_limit,
        max_captures_per_variant=max_captures_per_variant,
        band_csv_path=band_csv_path,
    )

    _write_capture_csv(captures, capture_csv_path)
    _write_report(report, out_path)
    return report


def _read_capture_palette(image_path: Path) -> dict[str, object] | None:
    match = CAPTURE_TIMESTAMP_PATTERN.search(image_path.name)
    if match is None:
        return None
    captured_at = (
        f"{match.group('date')}T{match.group('hour')}:"
        f"{match.group('minute')}:{match.group('second')}Z"
    )
    pixels = _load_pixels(image_path)
    if pixels is None:
        return None
    bar = pixels[COLOURBAR_Y0:COLOURBAR_Y1, COLOURBAR_X0:COLOURBAR_X1].mean(axis=0)
    landmarks = _detect_ramp_landmarks(bar)
    if landmarks is None:
        return None
    black_end, red_start = landmarks
    return {
        "source_file": str(image_path),
        "captured_at_utc": captured_at,
        "hour": int(match.group("hour")),
        "black_end_px": black_end,
        "red_start_px": red_start,
        "display_db_low": round(SCALE_DB_LOW + black_end * DB_PER_STEP, 3),
        "display_db_high": round(SCALE_DB_LOW + red_start * DB_PER_STEP, 3),
        "palette": bar,
    }


def _load_pixels(image_path: Path) -> np.ndarray | None:
    from PIL import Image

    with Image.open(image_path) as image:
        if image.size != (EXPECTED_WIDTH, EXPECTED_HEIGHT):
            return None
        return np.asarray(image.convert("RGB"), dtype=np.float64)


def _detect_ramp_landmarks(bar: np.ndarray) -> tuple[int, int] | None:
    """Return the last black ramp step and the first saturated-red step."""
    channel_max = bar.max(axis=1)
    black_end = -1
    for index in range(COLOURBAR_STEPS):
        if channel_max[index] >= BLACK_MAX_CHANNEL:
            break
        black_end = index
    if black_end < 0:
        return None

    red = (
        (bar[:, 0] >= RED_MIN_R)
        & (bar[:, 1] <= RED_MAX_G)
        & (bar[:, 2] <= RED_MAX_B)
    )
    red_start = -1
    for index in range(black_end + 1, COLOURBAR_STEPS - RED_RUN_LENGTH):
        if red[index : index + RED_RUN_LENGTH].all():
            red_start = index
            break
    if red_start < 0:
        return None
    return black_end, red_start


def _group_palette_variants(captures: list[dict[str, object]]) -> list[dict[str, object]]:
    """Group captures by the ramp's solid-red onset.

    `red_start_px` is the stable landmark: it is bit-identical across every
    capture sharing a palette. `black_end_px` sits where the ramp is nearly
    black, so JPEG noise moves it by a pixel on a small minority of captures.
    Grouping on the noisy landmark would manufacture variants, so the variant
    key is `red_start_px` alone and the black floor is taken as the variant
    median, with its observed range reported rather than smoothed away.
    """
    grouped: dict[int, dict[str, object]] = {}
    for record in captures:
        key = record["red_start_px"]
        variant = grouped.get(key)
        if variant is None:
            variant = {
                "palette_variant_id": "",
                "red_start_px": key,
                "capture_count": 0,
                "first_capture_utc": record["captured_at_utc"],
                "last_capture_utc": record["captured_at_utc"],
                "captures": [],
            }
            grouped[key] = variant
        variant["capture_count"] += 1
        variant["last_capture_utc"] = record["captured_at_utc"]
        variant["captures"].append(record)

    variants = sorted(grouped.values(), key=lambda item: item["first_capture_utc"])
    for index, variant in enumerate(variants):
        variant["palette_variant_id"] = f"palette_{index}"
        black_ends = [record["black_end_px"] for record in variant["captures"]]
        variant["black_end_px"] = int(statistics.median(black_ends))
        variant["black_end_px_min"] = min(black_ends)
        variant["black_end_px_max"] = max(black_ends)
        variant["display_db_low"] = round(
            SCALE_DB_LOW + variant["black_end_px"] * DB_PER_STEP, 3
        )
        variant["display_db_high"] = round(
            SCALE_DB_LOW + variant["red_start_px"] * DB_PER_STEP, 3
        )
        for record in variant["captures"]:
            record["palette_variant_id"] = variant["palette_variant_id"]
    return variants


def _find_change_points(captures: list[dict[str, object]]) -> list[dict[str, object]]:
    change_points = []
    for previous, current in zip(captures, captures[1:]):
        if previous["red_start_px"] == current["red_start_px"]:
            continue
        change_points.append(
            {
                "last_capture_before_utc": previous["captured_at_utc"],
                "first_capture_after_utc": current["captured_at_utc"],
                "red_start_px_before": previous["red_start_px"],
                "red_start_px_after": current["red_start_px"],
                "display_db_shift": round(
                    (current["red_start_px"] - previous["red_start_px"]) * DB_PER_STEP,
                    3,
                ),
            }
        )
    return change_points


def _compare_variants_in_db(
    *,
    early: dict[str, object],
    late: dict[str, object],
    hour_low: int,
    hour_high: int,
    residual_limit: float,
    max_captures_per_variant: int,
    band_csv_path: Path | None,
) -> dict[str, object]:
    """Compare two palette variants on the absolute dB axis, hour-matched.

    The Cumiana record has a strong diurnal cycle, so captures are restricted
    to a matched hour window before the two variants are compared at all.
    """
    early_side = _decode_variant(
        variant=early,
        hour_low=hour_low,
        hour_high=hour_high,
        residual_limit=residual_limit,
        max_captures=max_captures_per_variant,
    )
    late_side = _decode_variant(
        variant=late,
        hour_low=hour_low,
        hour_high=hour_high,
        residual_limit=residual_limit,
        max_captures=max_captures_per_variant,
    )

    comparison: dict[str, object] = {
        "hour_low_utc": hour_low,
        "hour_high_utc": hour_high,
        "early": {key: value for key, value in early_side.items() if key != "band_db"},
        "late": {key: value for key, value in late_side.items() if key != "band_db"},
    }
    if not early_side["capture_count"] or not late_side["capture_count"]:
        comparison["status"] = "insufficient_hour_matched_captures"
        comparison["bands"] = []
        return comparison

    resolvable_low = max(early["display_db_low"], late["display_db_low"])
    resolvable_high = min(early["display_db_high"], late["display_db_high"])
    bands = []
    for band_index, (low_khz, high_khz) in enumerate(DEFAULT_BANDS_KHZ):
        early_band = early_side["band_db"][band_index]
        late_band = late_side["band_db"][band_index]
        early_median = _median_or_none(early_band["values"])
        late_median = _median_or_none(late_band["values"])
        resolvable = (
            early_median is not None
            and late_median is not None
            and resolvable_low <= early_median <= resolvable_high
            and resolvable_low <= late_median <= resolvable_high
        )
        bands.append(
            {
                "band_low_khz": low_khz,
                "band_high_khz": high_khz,
                "row_count": early_band["row_count"],
                "early_variant_id": early["palette_variant_id"],
                "early_median_db": _round_or_none(early_median),
                "early_sample_count": len(early_band["values"]),
                "late_variant_id": late["palette_variant_id"],
                "late_median_db": _round_or_none(late_median),
                "late_sample_count": len(late_band["values"]),
                "delta_db": (
                    round(late_median - early_median, 2)
                    if early_median is not None and late_median is not None
                    else None
                ),
                "resolvable_in_both": resolvable,
            }
        )

    resolvable_bands = [band for band in bands if band["resolvable_in_both"]]
    comparison["status"] = (
        "compared" if resolvable_bands else "no_band_resolvable_in_both"
    )
    comparison["resolvable_db_low"] = resolvable_low
    comparison["resolvable_db_high"] = resolvable_high
    comparison["resolvable_band_count"] = len(resolvable_bands)
    if resolvable_bands:
        deltas = [band["delta_db"] for band in resolvable_bands]
        comparison["resolvable_delta_db_min"] = min(deltas)
        comparison["resolvable_delta_db_max"] = max(deltas)
        comparison["resolvable_delta_db_spread"] = round(max(deltas) - min(deltas), 2)
    comparison["bands"] = bands
    _write_band_csv(bands, band_csv_path)
    return comparison


def _decode_variant(
    *,
    variant: dict[str, object],
    hour_low: int,
    hour_high: int,
    residual_limit: float,
    max_captures: int,
) -> dict[str, object]:
    selected = [
        record
        for record in variant["captures"]
        if hour_low <= record["hour"] < hour_high
    ][:max_captures]

    band_rows = _band_row_slices()
    band_db: list[dict[str, object]] = [
        {"row_count": stop - start, "values": []} for start, stop in band_rows
    ]
    clipped_hot = []
    clipped_black = []
    masked = []
    residuals = []

    for record in selected:
        pixels = _load_pixels(Path(record["source_file"]))
        if pixels is None:
            continue
        panel = pixels[PANEL_Y0:PANEL_Y1, FRESH_X0:FRESH_X1]
        decoded = _decode_panel_db(
            panel=panel,
            palette=record["palette"],
            black_end=record["black_end_px"],
            red_start=record["red_start_px"],
            residual_limit=residual_limit,
        )
        clipped_hot.append(decoded["clipped_hot_fraction"])
        clipped_black.append(decoded["clipped_black_fraction"])
        masked.append(decoded["masked_fraction"])
        residuals.append(decoded["median_residual"])
        for band_index, (start, stop) in enumerate(band_rows):
            block = decoded["db"][start:stop]
            valid = block[np.isfinite(block)]
            if valid.size:
                band_db[band_index]["values"].append(float(np.median(valid)))

    return {
        "palette_variant_id": variant["palette_variant_id"],
        "display_db_low": variant["display_db_low"],
        "display_db_high": variant["display_db_high"],
        "capture_count": len(residuals),
        "clipped_hot_fraction": _mean_or_none(clipped_hot),
        "clipped_black_fraction": _mean_or_none(clipped_black),
        "masked_fraction": _mean_or_none(masked),
        "median_residual": _mean_or_none(residuals),
        "band_db": band_db,
    }


def _decode_panel_db(
    *,
    panel: np.ndarray,
    palette: np.ndarray,
    black_end: int,
    red_start: int,
    residual_limit: float,
) -> dict[str, object]:
    """Invert panel pixels to dB through this image's own colourbar.

    Only ramp steps strictly between the black floor and the solid-red top
    carry information; pixels nearest a censored step are dropped, as are
    pixels too far from any palette colour to be a credible match.
    """
    resolvable = np.arange(black_end + 1, red_start)
    reference = palette[resolvable]
    flat = panel.reshape(-1, 3)
    distances = np.linalg.norm(flat[:, None, :] - reference[None, :, :], axis=2)
    nearest = distances.argmin(axis=1)
    residual = distances[np.arange(flat.shape[0]), nearest]

    channel_max = flat.max(axis=1)
    is_black = channel_max < BLACK_MAX_CHANNEL
    is_hot = (
        (flat[:, 0] >= RED_MIN_R) & (flat[:, 1] <= RED_MAX_G) & (flat[:, 2] <= RED_MAX_B)
    )
    is_masked = residual > residual_limit

    db = SCALE_DB_LOW + resolvable[nearest] * DB_PER_STEP
    db = db.astype(np.float64)
    db[is_black | is_hot | is_masked] = np.nan

    return {
        "db": db.reshape(panel.shape[0], panel.shape[1]),
        "clipped_hot_fraction": round(float(is_hot.mean()), 4),
        "clipped_black_fraction": round(float(is_black.mean()), 4),
        "masked_fraction": round(float(is_masked.mean()), 4),
        "median_residual": round(float(np.median(residual)), 2),
    }


def _band_row_slices() -> list[tuple[int, int]]:
    """Map each frequency band to its row slice in the upper panel."""
    slices = []
    for low_khz, high_khz in DEFAULT_BANDS_KHZ:
        top = _row_for_khz(high_khz)
        bottom = _row_for_khz(low_khz)
        start = max(0, min(top, bottom))
        stop = min(PANEL_Y1 - PANEL_Y0, max(top, bottom) + 1)
        slices.append((start, max(start + 1, stop)))
    return slices


def _row_for_khz(khz: float) -> int:
    absolute = TICK_Y_AT_14KHZ + (14.0 - khz) * PX_PER_KHZ
    return int(round(absolute)) - PANEL_Y0


def _median_or_none(values: list[float]) -> float | None:
    return statistics.median(values) if values else None


def _mean_or_none(values: list[float]) -> float | None:
    return round(statistics.fmean(values), 4) if values else None


def _round_or_none(value: float | None) -> float | None:
    return None if value is None else round(value, 2)


def _write_capture_csv(captures: list[dict[str, object]], path: Path | None) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CAPTURE_FIELDNAMES, lineterminator="\n")
        writer.writeheader()
        for record in captures:
            writer.writerow({key: record.get(key, "") for key in CAPTURE_FIELDNAMES})


def _write_band_csv(bands: list[dict[str, object]], path: Path | None) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=BAND_FIELDNAMES, lineterminator="\n")
        writer.writeheader()
        for band in bands:
            writer.writerow({key: band.get(key, "") for key in BAND_FIELDNAMES})


def _write_report(report: dict[str, object], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

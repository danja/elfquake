"""Compare time-domain and frequency-domain shapes across signal sources."""

from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


SHAPE_METRICS = [
    "sample_count",
    "duration_seconds",
    "mean",
    "std",
    "coefficient_variation",
    "min",
    "max",
    "p50",
    "p90",
    "p99",
    "nonzero_ratio",
    "burst_ratio",
    "burst_run_count",
    "burst_run_rate",
    "lag1_autocorrelation",
    "skewness",
    "excess_kurtosis",
    "psd_total_power",
    "psd_slope",
    "spectral_centroid_hz",
    "spectral_rolloff90_hz",
    "psd_low_band_ratio",
    "psd_mid_band_ratio",
    "psd_high_band_ratio",
]

DISTANCE_METRICS = [
    "coefficient_variation",
    "nonzero_ratio",
    "burst_ratio",
    "burst_run_rate",
    "lag1_autocorrelation",
    "skewness",
    "excess_kurtosis",
    "psd_slope",
    "spectral_centroid_hz",
    "spectral_rolloff90_hz",
    "psd_low_band_ratio",
    "psd_mid_band_ratio",
    "psd_high_band_ratio",
]

SERIES_FIELDNAMES = [
    "series_id",
    "source_type",
    "source_file_count",
    "source_files",
    "sample_seconds",
] + SHAPE_METRICS

PAIR_FIELDNAMES = [
    "left_series_id",
    "right_series_id",
    "normalized_distance",
] + [f"delta_{metric}" for metric in SHAPE_METRICS]

SENSOR_SCAN_METRICS = [
    "coefficient_variation",
    "nonzero_ratio",
    "burst_ratio",
    "burst_run_rate",
    "lag1_autocorrelation",
    "skewness",
    "excess_kurtosis",
    "psd_slope",
    "spectral_centroid_hz",
    "spectral_rolloff90_hz",
    "psd_low_band_ratio",
    "psd_mid_band_ratio",
    "psd_high_band_ratio",
]

SENSOR_SCAN_FIELDNAMES = [
    "sensor_id",
    "reference_series_id",
    "signal_series_id",
    "shape_score",
    "sample_count",
    "reference_sample_count",
] + SENSOR_SCAN_METRICS + [f"delta_{metric}" for metric in SENSOR_SCAN_METRICS]


@dataclass(frozen=True)
class SignalSeries:
    series_id: str
    source_type: str
    source_files: tuple[Path, ...]
    sample_seconds: float
    values: np.ndarray


def compare_signal_shapes(
    *,
    series: list[SignalSeries],
    series_out: Path,
    pairs_out: Path,
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    if len(series) < 2:
        raise ValueError("at least two series are required")

    series_rows = [_series_row(item) for item in series]
    pair_rows = []
    for left_index, left in enumerate(series_rows):
        for right in series_rows[left_index + 1 :]:
            pair_rows.append(_pair_row(left, right, series_rows))

    series_out.parent.mkdir(parents=True, exist_ok=True)
    with series_out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=SERIES_FIELDNAMES, lineterminator="\n")
        writer.writeheader()
        writer.writerows(series_rows)

    pairs_out.parent.mkdir(parents=True, exist_ok=True)
    with pairs_out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=PAIR_FIELDNAMES, lineterminator="\n")
        writer.writeheader()
        writer.writerows(pair_rows)
    return series_rows, pair_rows


def scan_sensor_signal_shapes(
    *,
    reference: SignalSeries,
    signal_csv: Path,
    signal_field: str,
    out_path: Path,
    sample_seconds: float = 60.0,
    sensor_ids: list[int] | None = None,
    series_id_prefix: str = "sensor",
) -> list[dict[str, str]]:
    """Rank sensor traces by coarse shape similarity to a reference series."""
    resolved_sensor_ids = sorted(sensor_ids if sensor_ids is not None else _read_sensor_ids(signal_csv))
    if not resolved_sensor_ids:
        raise ValueError("no sensor ids found")

    reference_row = _series_row(reference)
    rows = []
    for sensor_id in resolved_sensor_ids:
        signal = sensor_signal_series(
            series_id=f"{series_id_prefix}_{sensor_id}",
            signal_csv=signal_csv,
            signal_field=signal_field,
            sample_seconds=sample_seconds,
            sensor_id=sensor_id,
        )
        signal_row = _series_row(signal)
        deltas = {
            metric: abs(_float(signal_row[metric], 0.0) - _float(reference_row[metric], 0.0))
            for metric in SENSOR_SCAN_METRICS
        }
        row = {
            "sensor_id": str(sensor_id),
            "reference_series_id": reference.series_id,
            "signal_series_id": signal.series_id,
            "shape_score": _fmt(_sensor_shape_score(deltas)),
            "sample_count": signal_row["sample_count"],
            "reference_sample_count": reference_row["sample_count"],
        }
        row.update({metric: signal_row[metric] for metric in SENSOR_SCAN_METRICS})
        row.update({f"delta_{key}": _fmt(value) for key, value in deltas.items()})
        rows.append({field: row.get(field, "") for field in SENSOR_SCAN_FIELDNAMES})

    rows.sort(key=lambda item: (_float(item["shape_score"], math.inf), int(item["sensor_id"])))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=SENSOR_SCAN_FIELDNAMES, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    return rows


def event_energy_series(
    *,
    series_id: str,
    events_csv: Path,
    bin_seconds: int = 3600,
    energy_from_magnitude: bool = True,
) -> SignalSeries:
    if bin_seconds < 1:
        raise ValueError("bin_seconds must be at least 1")
    events = []
    with events_csv.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            event_time = row.get("event_time_utc")
            if not event_time:
                continue
            magnitude = _float(row.get("magnitude", ""), 0.0)
            value = 10.0 ** max(0.0, magnitude) if energy_from_magnitude else 1.0
            events.append((_parse_utc(event_time), value))
    if not events:
        return SignalSeries(series_id, "event_energy", (events_csv,), float(bin_seconds), np.zeros(0, dtype=np.float64))

    start = min(event_time for event_time, _ in events)
    end = max(event_time for event_time, _ in events)
    bin_count = int((end - start).total_seconds() // bin_seconds) + 1
    values = np.zeros(max(1, bin_count), dtype=np.float64)
    for event_time, value in events:
        index = int((event_time - start).total_seconds() // bin_seconds)
        values[min(values.size - 1, max(0, index))] += value
    return SignalSeries(series_id, "event_energy", (events_csv,), float(bin_seconds), values)


def sensor_signal_series(
    *,
    series_id: str,
    signal_csv: Path,
    signal_field: str,
    sample_seconds: float = 60.0,
    sensor_id: int | None = None,
) -> SignalSeries:
    if sample_seconds <= 0:
        raise ValueError("sample_seconds must be positive")
    by_step: dict[int, float] = {}
    with signal_csv.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if "step" not in row:
                continue
            if sensor_id is not None and row.get("sensor_id") != str(sensor_id):
                continue
            value_text = row.get(signal_field, "")
            if signal_field == "avalanche_signal" and value_text == "":
                value_text = row.get("piezo_signal", "")
            by_step[int(row["step"])] = by_step.get(int(row["step"]), 0.0) + _float(value_text, 0.0)
    if not by_step:
        values = np.zeros(0, dtype=np.float64)
    else:
        first = min(by_step)
        last = max(by_step)
        values = np.zeros(last - first + 1, dtype=np.float64)
        for step, value in by_step.items():
            values[step - first] = value
    return SignalSeries(series_id, f"sensor_{signal_field}", (signal_csv,), float(sample_seconds), values)


def _read_sensor_ids(signal_csv: Path) -> list[int]:
    sensor_ids = set()
    with signal_csv.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            value = row.get("sensor_id")
            if value in (None, ""):
                continue
            sensor_ids.add(int(value))
    return sorted(sensor_ids)


def vlf_image_column_series(
    *,
    series_id: str,
    image_paths: list[Path],
    sample_seconds: float = 1.0,
    crop_left: float = 0.0,
    crop_top: float = 0.13,
    crop_right: float = 0.83,
    crop_bottom: float = 0.95,
) -> SignalSeries:
    if sample_seconds <= 0:
        raise ValueError("sample_seconds must be positive")
    if not image_paths:
        raise ValueError("at least one VLF image is required")
    try:
        from PIL import Image
    except ImportError as error:  # pragma: no cover - optional runtime dependency.
        raise RuntimeError("Pillow is required for VLF image shape comparison") from error

    traces = []
    for path in sorted(image_paths):
        with Image.open(path) as image:
            rgb = image.convert("RGB")
            left, top, right, bottom = _crop_box(
                *rgb.size,
                crop_left=crop_left,
                crop_top=crop_top,
                crop_right=crop_right,
                crop_bottom=crop_bottom,
            )
            crop = rgb.crop((left, top, right, bottom))
            traces.append(_column_intensity(crop))
    values = np.concatenate(traces) if traces else np.zeros(0, dtype=np.float64)
    return SignalSeries(series_id, "vlf_image_column_intensity", tuple(sorted(image_paths)), float(sample_seconds), values)


def _series_row(series: SignalSeries) -> dict[str, str]:
    stats = _shape_stats(series.values, sample_seconds=series.sample_seconds)
    row = {
        "series_id": series.series_id,
        "source_type": series.source_type,
        "source_file_count": str(len(series.source_files)),
        "source_files": json.dumps([str(path) for path in series.source_files], sort_keys=True),
        "sample_seconds": _fmt(series.sample_seconds),
    }
    row.update({key: _fmt(value) for key, value in stats.items()})
    return {field: row.get(field, "") for field in SERIES_FIELDNAMES}


def _pair_row(left: dict[str, str], right: dict[str, str], all_rows: list[dict[str, str]]) -> dict[str, str]:
    deltas = {}
    total = 0.0
    count = 0
    for metric in SHAPE_METRICS:
        left_value = _float(left[metric], 0.0)
        right_value = _float(right[metric], 0.0)
        delta = left_value - right_value
        deltas[f"delta_{metric}"] = _fmt(delta)
        if metric not in DISTANCE_METRICS:
            continue
        values = [_float(row[metric], 0.0) for row in all_rows]
        scale = _std(values, sum(values) / len(values)) if values else 0.0
        if scale <= 1e-12:
            scale = max(abs(left_value), abs(right_value), 1.0)
        total += (delta / scale) ** 2
        count += 1
    row = {
        "left_series_id": left["series_id"],
        "right_series_id": right["series_id"],
        "normalized_distance": _fmt(math.sqrt(total / count) if count else 0.0),
        **deltas,
    }
    return {field: row.get(field, "") for field in PAIR_FIELDNAMES}


def _sensor_shape_score(deltas: dict[str, float]) -> float:
    # Ranking aid only; use individual delta columns when interpreting results.
    return (
        deltas["lag1_autocorrelation"] * 2.0
        + deltas["psd_slope"]
        + deltas["burst_run_rate"] * 10.0
        + deltas["coefficient_variation"] * 0.5
        + deltas["nonzero_ratio"] * 0.5
    )


def _shape_stats(values: np.ndarray, *, sample_seconds: float) -> dict[str, float]:
    signal = values.astype(np.float64)
    if signal.size == 0:
        return {metric: 0.0 for metric in SHAPE_METRICS}

    mean = float(signal.mean())
    std = float(signal.std())
    centered = signal - mean
    abs_mean = abs(mean)
    burst_threshold = float(np.quantile(signal, 0.95))
    burst_flags = [bool(value > burst_threshold and value > 0.0) for value in signal]
    burst_run_count = _count_runs(burst_flags)
    psd = _psd_stats(signal, sample_seconds=sample_seconds)
    stats = {
        "sample_count": float(signal.size),
        "duration_seconds": float(signal.size) * sample_seconds,
        "mean": mean,
        "std": std,
        "coefficient_variation": std / abs_mean if abs_mean > 1e-12 else 0.0,
        "min": float(signal.min()),
        "max": float(signal.max()),
        "p50": float(np.quantile(signal, 0.50)),
        "p90": float(np.quantile(signal, 0.90)),
        "p99": float(np.quantile(signal, 0.99)),
        "nonzero_ratio": float(np.count_nonzero(signal) / signal.size),
        "burst_ratio": float(sum(burst_flags) / signal.size),
        "burst_run_count": float(burst_run_count),
        "burst_run_rate": float(burst_run_count / signal.size),
        "lag1_autocorrelation": _lag1_autocorrelation(signal),
        "skewness": _skewness(centered, std),
        "excess_kurtosis": _excess_kurtosis(centered, std),
    }
    stats.update(psd)
    return stats


def _psd_stats(signal: np.ndarray, *, sample_seconds: float) -> dict[str, float]:
    if signal.size < 3:
        return {
            "psd_total_power": 0.0,
            "psd_slope": 0.0,
            "spectral_centroid_hz": 0.0,
            "spectral_rolloff90_hz": 0.0,
            "psd_low_band_ratio": 0.0,
            "psd_mid_band_ratio": 0.0,
            "psd_high_band_ratio": 0.0,
        }
    centered = signal - float(signal.mean())
    window = np.hanning(signal.size)
    transformed = np.fft.rfft(centered * window)
    freqs = np.fft.rfftfreq(signal.size, d=sample_seconds)
    power = np.abs(transformed) ** 2
    freqs = freqs[1:]
    power = power[1:]
    total = float(power.sum())
    if total <= 1e-24 or freqs.size == 0:
        return {
            "psd_total_power": 0.0,
            "psd_slope": 0.0,
            "spectral_centroid_hz": 0.0,
            "spectral_rolloff90_hz": 0.0,
            "psd_low_band_ratio": 0.0,
            "psd_mid_band_ratio": 0.0,
            "psd_high_band_ratio": 0.0,
        }

    cumulative = np.cumsum(power)
    rolloff_index = int(np.searchsorted(cumulative, total * 0.90, side="left"))
    low_max = max(1, int(math.ceil(freqs.size / 3)))
    mid_max = max(low_max + 1, int(math.ceil(freqs.size * 2 / 3)))
    slope = _loglog_slope(freqs, power)
    return {
        "psd_total_power": total,
        "psd_slope": slope,
        "spectral_centroid_hz": float((freqs * power).sum() / total),
        "spectral_rolloff90_hz": float(freqs[min(rolloff_index, freqs.size - 1)]),
        "psd_low_band_ratio": float(power[:low_max].sum() / total),
        "psd_mid_band_ratio": float(power[low_max:mid_max].sum() / total),
        "psd_high_band_ratio": float(power[mid_max:].sum() / total),
    }


def _column_intensity(crop) -> np.ndarray:
    width, height = crop.size
    pixels = crop.load()
    values = np.zeros(width, dtype=np.float64)
    for x in range(width):
        total = 0.0
        for y in range(height):
            red, green, blue = pixels[x, y]
            total += (0.2126 * red + 0.7152 * green + 0.0722 * blue) / 255.0
        values[x] = total / height
    return values


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


def _lag1_autocorrelation(signal: np.ndarray) -> float:
    if signal.size < 2:
        return 0.0
    left = signal[:-1] - float(signal[:-1].mean())
    right = signal[1:] - float(signal[1:].mean())
    denominator = float(np.sqrt((left * left).sum() * (right * right).sum()))
    if denominator <= 1e-24:
        return 0.0
    return float((left * right).sum() / denominator)


def _skewness(centered: np.ndarray, std: float) -> float:
    if centered.size == 0 or std <= 1e-12:
        return 0.0
    return float(np.mean((centered / std) ** 3))


def _excess_kurtosis(centered: np.ndarray, std: float) -> float:
    if centered.size == 0 or std <= 1e-12:
        return 0.0
    return float(np.mean((centered / std) ** 4) - 3.0)


def _loglog_slope(freqs: np.ndarray, power: np.ndarray) -> float:
    mask = (freqs > 0) & (power > 0)
    if int(mask.sum()) < 2:
        return 0.0
    x = np.log10(freqs[mask])
    y = np.log10(power[mask])
    x_mean = float(x.mean())
    y_mean = float(y.mean())
    denominator = float(((x - x_mean) ** 2).sum())
    if denominator <= 1e-24:
        return 0.0
    return float(((x - x_mean) * (y - y_mean)).sum() / denominator)


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


def _parse_utc(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(timezone.utc)


def _std(values: list[float], mean: float) -> float:
    return math.sqrt(sum((value - mean) ** 2 for value in values) / len(values)) if values else 0.0


def _float(value: str, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _fmt(value: float) -> str:
    return f"{value:.9f}"

"""Shared time-domain and frequency-domain signal shape metrics."""

from __future__ import annotations

import math

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


def shape_stats(values: np.ndarray, *, sample_seconds: float) -> dict[str, float]:
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
    stats.update(_psd_stats(signal, sample_seconds=sample_seconds))
    return stats


def metric_std(values: list[float], mean: float) -> float:
    return math.sqrt(sum((value - mean) ** 2 for value in values) / len(values)) if values else 0.0


def metric_float(value: str | None, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def metric_fmt(value: float) -> str:
    return f"{value:.9f}"


def _psd_stats(signal: np.ndarray, *, sample_seconds: float) -> dict[str, float]:
    if signal.size < 3:
        return _empty_psd_stats()

    centered = signal - float(signal.mean())
    window = np.hanning(signal.size)
    transformed = np.fft.rfft(centered * window)
    freqs = np.fft.rfftfreq(signal.size, d=sample_seconds)
    power = np.abs(transformed) ** 2
    freqs = freqs[1:]
    power = power[1:]
    total = float(power.sum())
    if total <= 1e-24 or freqs.size == 0:
        return _empty_psd_stats()

    cumulative = np.cumsum(power)
    rolloff_index = int(np.searchsorted(cumulative, total * 0.90, side="left"))
    low_max = max(1, int(math.ceil(freqs.size / 3)))
    mid_max = max(low_max + 1, int(math.ceil(freqs.size * 2 / 3)))
    return {
        "psd_total_power": total,
        "psd_slope": _loglog_slope(freqs, power),
        "spectral_centroid_hz": float((freqs * power).sum() / total),
        "spectral_rolloff90_hz": float(freqs[min(rolloff_index, freqs.size - 1)]),
        "psd_low_band_ratio": float(power[:low_max].sum() / total),
        "psd_mid_band_ratio": float(power[low_max:mid_max].sum() / total),
        "psd_high_band_ratio": float(power[mid_max:].sum() / total),
    }


def _empty_psd_stats() -> dict[str, float]:
    return {
        "psd_total_power": 0.0,
        "psd_slope": 0.0,
        "spectral_centroid_hz": 0.0,
        "spectral_rolloff90_hz": 0.0,
        "psd_low_band_ratio": 0.0,
        "psd_mid_band_ratio": 0.0,
        "psd_high_band_ratio": 0.0,
    }


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

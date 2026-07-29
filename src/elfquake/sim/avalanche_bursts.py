"""Causal burst scoring and episode segmentation for avalanche signals."""

from __future__ import annotations


def causal_baseline_subtracted_scores(values: list[float], *, decay: float) -> list[float]:
    """Return positive excess over a one-sided exponential baseline."""
    if not 0 < decay < 1:
        raise ValueError("decay must be between 0 and 1")
    if not values:
        return []
    baseline = values[0]
    scores = []
    for value in values:
        scores.append(max(0.0, value - baseline))
        baseline = decay * baseline + (1.0 - decay) * value
    return scores


def causal_baseline_relative_scores(values: list[float], *, decay: float) -> list[float]:
    """Return positive baseline excess divided by the causal baseline level."""
    if not 0 < decay < 1:
        raise ValueError("decay must be between 0 and 1")
    if not values:
        return []
    baseline = values[0]
    scores = []
    for value in values:
        denominator = max(abs(baseline), 1e-12)
        scores.append(max(0.0, value - baseline) / denominator)
        baseline = decay * baseline + (1.0 - decay) * value
    return scores


def segment_burst_peaks(
    scores: list[float], *, threshold: float, gap_steps: int
) -> list[int]:
    """Return one peak index per above-threshold episode, using past data only."""
    if gap_steps < 0:
        raise ValueError("gap_steps must be non-negative")
    peaks: list[int] = []
    active_peak: int | None = None
    last_above: int | None = None
    for index, score in enumerate(scores):
        if score <= threshold:
            if active_peak is not None and last_above is not None and index - last_above > gap_steps:
                peaks.append(active_peak)
                active_peak = None
                last_above = None
            continue
        if active_peak is None:
            active_peak = index
        elif score > scores[active_peak]:
            active_peak = index
        last_above = index
    if active_peak is not None:
        peaks.append(active_peak)
    return peaks


def quantile_threshold(values: list[float], fraction: float) -> float:
    if not values:
        return 0.0
    if not 0 <= fraction < 1:
        raise ValueError("fraction must be at least 0 and below 1")
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round((len(ordered) - 1) * fraction)))
    return ordered[index]


def robust_score_scale(values: list[float], *, fraction: float = 0.90) -> float:
    """Return a positive training-derived scale for burst scores."""
    scale = quantile_threshold(values, fraction)
    return max(scale, 1e-12)

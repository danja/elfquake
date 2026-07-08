"""Synthetic-trained learned scorer for weekly trial event-list forecasts."""

from __future__ import annotations

import csv
import json
import math
import random
from datetime import timedelta
from pathlib import Path

from elfquake.features.common import format_utc, parse_utc
from elfquake.models.trial_forecast import (
    _blend_expected_count,
    _expand_globs,
    _forecast_points,
    _historical_rate,
    _load_astronomy_context,
    _load_event_points,
    _load_vlf_context,
    _spatial_bins,
    _synthetic_context,
    _write_prediction_csv,
)


EXCLUDED_FEATURE_FIELDS = {
    "dataset_id",
    "window_id",
    "region_id",
    "window_start_utc",
    "window_end_utc",
    "source_file",
    "target_event_count",
    "target_occurred",
    "target_status",
}


def generate_learned_weekly_event_forecast(
    *,
    real_events_csv: Path,
    synthetic_windows_csv: Path,
    out_path: Path,
    events_out_path: Path,
    as_of_utc: str,
    horizon_days: int = 7,
    magnitude_threshold: float = 2.0,
    max_events: int = 25,
    seed: int = 42,
    train_fraction: float = 0.8,
    epochs: int = 500,
    learning_rate: float = 0.08,
    l2: float = 0.001,
    synthetic_event_globs: list[str] | None = None,
    vlf_window_csvs: list[Path] | None = None,
    vlf_anomaly_report: Path | None = None,
    vlf_audio_globs: list[str] | None = None,
    astronomy_globs: list[str] | None = None,
) -> dict[str, object]:
    if not 0 < train_fraction < 1:
        raise ValueError("train_fraction must be between 0 and 1")

    as_of = parse_utc(as_of_utc)
    forecast_end = as_of + timedelta(days=horizon_days)
    scorer = train_synthetic_window_scorer(
        synthetic_windows_csv=synthetic_windows_csv,
        train_fraction=train_fraction,
        epochs=epochs,
        learning_rate=learning_rate,
        l2=l2,
        seed=seed,
    )
    learned_score = float(scorer["latest_window_score"])

    real_events = _load_event_points(real_events_csv, source="ingv", before_utc=as_of_utc)
    target_real_events = [event for event in real_events if event.magnitude > magnitude_threshold]
    synthetic_paths = _expand_globs(
        synthetic_event_globs
        or [
            "data/derived/sim/*.synthetic_events.csv",
            "data/derived/sim/*.avalanche_events.csv",
        ]
    )
    synthetic_events = [
        event
        for path in synthetic_paths
        for event in _load_event_points(path, source="synthetic_avalanche")
        if event.magnitude > magnitude_threshold
    ]
    vlf_context = _load_vlf_context(
        vlf_window_csvs
        or [
            Path("data/derived/models/all_italy.real_vlf_aligned_windows.csv"),
            Path("data/derived/models/central_italy.real_vlf_aligned_windows.csv"),
        ],
        anomaly_report=vlf_anomaly_report or Path("data/derived/models/self_supervised/real_vlf_anomaly_forecast.json"),
        audio_paths=_expand_globs(vlf_audio_globs or ["data/derived/vlf/*.audio_features.csv"]),
    )
    astronomy_context = _load_astronomy_context(_expand_globs(astronomy_globs or ["data/raw/astronomy/captures/**/*.json"]))
    synthetic_context = _synthetic_context(synthetic_events)
    historical = _historical_rate(target_real_events, as_of_utc=as_of_utc, horizon_days=horizon_days)
    expected_count = _blend_expected_count(
        historical_weekly_count=float(historical["expected_event_count"]),
        vlf_score=vlf_context.score,
        astronomy_score=astronomy_context.score,
        synthetic_score=learned_score,
    )
    event_count = max(1, min(max_events, int(round(expected_count))))
    predictions = _forecast_points(
        spatial_bins=_spatial_bins(target_real_events, synthetic_events),
        count=event_count,
        start=as_of,
        end=forecast_end,
        seed=seed,
        magnitude_threshold=magnitude_threshold,
        real_events=target_real_events,
        synthetic_events=synthetic_events,
        expected_count=expected_count,
        vlf_score=vlf_context.score,
        astronomy_score=astronomy_context.score,
        synthetic_score=learned_score,
    )
    for index, row in enumerate(predictions, start=1):
        row["prediction_id"] = f"learned_{index:03d}"
        row["warning"] = "synthetic-trained learned trial; not validated prediction"
    _write_prediction_csv(events_out_path, predictions)

    report: dict[str, object] = {
        "schema": "elfquake.learned_multimodal_weekly_event_forecast.v1",
        "status": "trial_run",
        "warning": "Synthetic-trained engineering baseline only: not validated as earthquake prediction capability.",
        "forecast_start_utc": as_of_utc,
        "forecast_end_utc": format_utc(forecast_end),
        "horizon_days": horizon_days,
        "magnitude_condition": f">{magnitude_threshold:g}",
        "seed": seed,
        "max_events": max_events,
        "predicted_event_count": len(predictions),
        "uncapped_expected_event_count": round(expected_count, 6),
        "events_out": str(events_out_path),
        "sources": {
            "ingv": {
                "path": str(real_events_csv),
                "event_count": len(real_events),
                "target_event_count": len(target_real_events),
                **historical,
            },
            "synthetic_avalanche": {
                "path_count": len(synthetic_paths),
                "target_event_count": len(synthetic_events),
                **synthetic_context.details,
            },
            "vlf": {
                "score": round(vlf_context.score, 6),
                "row_count": vlf_context.row_count,
                "latest_time_utc": vlf_context.latest_time_utc,
                **vlf_context.details,
            },
            "astronomy": {
                "score": round(astronomy_context.score, 6),
                "row_count": astronomy_context.row_count,
                "latest_time_utc": astronomy_context.latest_time_utc,
                **astronomy_context.details,
            },
        },
        "model": {
            "type": "synthetic_window_logistic_scorer",
            "synthetic_windows": str(synthetic_windows_csv),
            "learned_scorer": scorer,
            "count_mix": "historical INGV weekly rate modulated by current VLF, astronomy, and synthetic-trained learned score",
            "spatial_mix": "historical INGV density mixed with synthetic avalanche spatial density",
            "downstream_contract": "CSV rows preserve the trial forecast event-list interface.",
        },
        "predictions_preview": predictions[:5],
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    return report


def train_synthetic_window_scorer(
    *,
    synthetic_windows_csv: Path,
    train_fraction: float = 0.8,
    epochs: int = 500,
    learning_rate: float = 0.08,
    l2: float = 0.001,
    seed: int = 42,
) -> dict[str, object]:
    rows = _read_labeled_rows(synthetic_windows_csv)
    if len(rows) < 4:
        return _blocked_scorer_report(synthetic_windows_csv, "insufficient_rows", len(rows))
    feature_names = _numeric_feature_names(rows)
    if not feature_names:
        return _blocked_scorer_report(synthetic_windows_csv, "no_numeric_features", len(rows))
    rows = sorted(rows, key=lambda row: row.get("window_start_utc", ""))
    split_at = max(2, min(len(rows) - 1, int(len(rows) * train_fraction)))
    train_rows = rows[:split_at]
    test_rows = rows[split_at:]
    train_matrix = [[_number(row.get(name, "")) for name in feature_names] for row in train_rows]
    test_matrix = [[_number(row.get(name, "")) for name in feature_names] for row in test_rows]
    train_y = [int(row["target_occurred"]) for row in train_rows]
    test_y = [int(row["target_occurred"]) for row in test_rows]
    means, scales = _standardization(train_matrix)
    x_train = [_standardize(row, means=means, scales=scales) for row in train_matrix]
    x_test = [_standardize(row, means=means, scales=scales) for row in test_matrix]
    weights, bias = _fit_logistic(
        x_train,
        train_y,
        epochs=epochs,
        learning_rate=learning_rate,
        l2=l2,
        seed=seed,
    )
    train_probabilities = [_sigmoid(_dot(weights, row) + bias) for row in x_train]
    test_probabilities = [_sigmoid(_dot(weights, row) + bias) for row in x_test]
    threshold = _best_balanced_threshold(train_probabilities, train_y)
    latest_count = min(24, len(rows))
    latest_rows = rows[-latest_count:]
    latest_matrix = [[_number(row.get(name, "")) for name in feature_names] for row in latest_rows]
    latest_x = [_standardize(row, means=means, scales=scales) for row in latest_matrix]
    latest_probabilities = [_sigmoid(_dot(weights, row) + bias) for row in latest_x]
    top_features = sorted(
        [
            {"name": name, "weight": round(weight, 6)}
            for name, weight in zip(feature_names, weights)
        ],
        key=lambda item: abs(float(item["weight"])),
        reverse=True,
    )[:12]
    return {
        "schema": "elfquake.synthetic_window_logistic_scorer.v1",
        "status": "evaluated",
        "synthetic_windows": str(synthetic_windows_csv),
        "row_count": len(rows),
        "train_row_count": len(train_rows),
        "test_row_count": len(test_rows),
        "feature_count": len(feature_names),
        "target_positive_count": sum(int(row["target_occurred"]) for row in rows),
        "train_positive_count": sum(train_y),
        "test_positive_count": sum(test_y),
        "train_fraction": train_fraction,
        "epochs": epochs,
        "learning_rate": learning_rate,
        "l2": l2,
        "threshold": round(threshold, 6),
        "train_metrics": _metrics(train_probabilities, train_y, threshold=threshold),
        "test_metrics": _metrics(test_probabilities, test_y, threshold=threshold),
        "latest_window_count": latest_count,
        "latest_window_score": round(sum(latest_probabilities) / max(1, len(latest_probabilities)), 6),
        "latest_window_max_score": round(max(latest_probabilities) if latest_probabilities else 0.0, 6),
        "top_features": top_features,
        "note": "Trained only on synthetic aligned rows; use as a swappable scorer smoke test, not as real earthquake evidence.",
    }


def _read_labeled_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return [row for row in csv.DictReader(handle) if row.get("target_occurred") in {"0", "1"}]


def _numeric_feature_names(rows: list[dict[str, str]]) -> list[str]:
    names: list[str] = []
    for name in rows[0].keys():
        if name in EXCLUDED_FEATURE_FIELDS or name.startswith("target_"):
            continue
        values = [_number(row.get(name, "")) for row in rows]
        if any(value != 0.0 for value in values):
            names.append(name)
    return names


def _fit_logistic(
    x: list[list[float]],
    y: list[int],
    *,
    epochs: int,
    learning_rate: float,
    l2: float,
    seed: int,
) -> tuple[list[float], float]:
    rng = random.Random(seed)
    feature_count = len(x[0])
    weights = [rng.uniform(-0.01, 0.01) for _ in range(feature_count)]
    bias = 0.0
    positives = sum(y)
    negatives = len(y) - positives
    pos_weight = len(y) / (2 * positives) if positives else 1.0
    neg_weight = len(y) / (2 * negatives) if negatives else 1.0
    for _ in range(epochs):
        grad_w = [0.0 for _ in weights]
        grad_b = 0.0
        for row, target in zip(x, y):
            probability = _sigmoid(_dot(weights, row) + bias)
            sample_weight = pos_weight if target == 1 else neg_weight
            error = (probability - target) * sample_weight
            grad_b += error
            for index, value in enumerate(row):
                grad_w[index] += error * value
        divisor = max(1, len(y))
        for index in range(len(weights)):
            grad = grad_w[index] / divisor + l2 * weights[index]
            weights[index] -= learning_rate * grad
        bias -= learning_rate * grad_b / divisor
    return weights, bias


def _best_balanced_threshold(probabilities: list[float], targets: list[int]) -> float:
    candidates = sorted(set([0.5, *probabilities]))
    best_threshold = 0.5
    best_score = -1.0
    for threshold in candidates:
        score = float(_metrics(probabilities, targets, threshold=threshold)["balanced_accuracy"])
        if score > best_score:
            best_score = score
            best_threshold = threshold
    return best_threshold


def _metrics(probabilities: list[float], targets: list[int], *, threshold: float) -> dict[str, object]:
    tp = tn = fp = fn = 0
    for probability, target in zip(probabilities, targets):
        predicted = 1 if probability >= threshold else 0
        if predicted == 1 and target == 1:
            tp += 1
        elif predicted == 0 and target == 0:
            tn += 1
        elif predicted == 1:
            fp += 1
        else:
            fn += 1
    positive_recall = tp / (tp + fn) if tp + fn else 0.0
    negative_recall = tn / (tn + fp) if tn + fp else 0.0
    accuracy = (tp + tn) / max(1, len(targets))
    return {
        "accuracy": round(accuracy, 6),
        "balanced_accuracy": round((positive_recall + negative_recall) / 2.0, 6),
        "positive_recall": round(positive_recall, 6),
        "negative_recall": round(negative_recall, 6),
        "confusion": {
            "true_positive": tp,
            "true_negative": tn,
            "false_positive": fp,
            "false_negative": fn,
        },
    }


def _standardization(matrix: list[list[float]]) -> tuple[list[float], list[float]]:
    means: list[float] = []
    scales: list[float] = []
    for values in zip(*matrix):
        mean = sum(values) / len(values)
        variance = sum((value - mean) ** 2 for value in values) / len(values)
        means.append(mean)
        scales.append(math.sqrt(variance) or 1.0)
    return means, scales


def _standardize(row: list[float], *, means: list[float], scales: list[float]) -> list[float]:
    return [(value - mean) / scale for value, mean, scale in zip(row, means, scales)]


def _number(value: str) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return 0.0
    return parsed if math.isfinite(parsed) else 0.0


def _sigmoid(value: float) -> float:
    if value >= 0:
        z = math.exp(-value)
        return 1.0 / (1.0 + z)
    z = math.exp(value)
    return z / (1.0 + z)


def _dot(left: list[float], right: list[float]) -> float:
    return sum(a * b for a, b in zip(left, right))


def _blocked_scorer_report(path: Path, status: str, row_count: int) -> dict[str, object]:
    return {
        "schema": "elfquake.synthetic_window_logistic_scorer.v1",
        "status": status,
        "synthetic_windows": str(path),
        "row_count": row_count,
        "latest_window_score": 0.0,
    }

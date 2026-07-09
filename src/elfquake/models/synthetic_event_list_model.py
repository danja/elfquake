"""Dependency-light synthetic event-list target model."""

from __future__ import annotations

import csv
import json
import math
import random
from pathlib import Path

from elfquake.models.learned_forecast import (
    _best_balanced_threshold,
    _dot,
    _fit_logistic,
    _metrics,
    _number,
    _sigmoid,
    _standardization,
    _standardize,
)


ID_FIELDS = {
    "dataset_id",
    "window_id",
    "region_id",
    "window_start_utc",
    "window_end_utc",
    "source_file",
    "model_split",
    "synthetic_regime_id",
    "synthetic_episode_id",
}

TARGET_PREFIXES = ("target_", "eventlist_target_")
TARGET_FIELDS = {
    "quality_missing_eventlist_target_location",
}
DIAGNOSTIC_PREFIXES = (
    "synthetic_row_index",
    "synthetic_group_row_count",
    "synthetic_regime_index",
    "synthetic_burn_in",
    "synthetic_episode_index",
    "synthetic_episode_row_index",
    "synthetic_episode_group_count",
)


def train_synthetic_event_list_model(
    *,
    input_csv: Path,
    out_path: Path,
    predictions_out: Path | None = None,
    train_fraction: float = 0.8,
    split_field: str = "",
    epochs: int = 600,
    learning_rate: float = 0.05,
    l2: float = 0.001,
    seed: int = 42,
    max_feature_count: int = 0,
    occurrence_ensemble_count: int = 1,
    occurrence_feature_bag_fraction: float = 1.0,
) -> dict[str, object]:
    if not 0 < train_fraction < 1:
        raise ValueError("train_fraction must be between 0 and 1")
    if max_feature_count < 0:
        raise ValueError("max_feature_count must be non-negative")
    if occurrence_ensemble_count < 1:
        raise ValueError("occurrence_ensemble_count must be at least 1")
    if not 0 < occurrence_feature_bag_fraction <= 1:
        raise ValueError("occurrence_feature_bag_fraction must be in (0, 1]")

    rows = _read_labeled_rows(input_csv)
    if len(rows) < 4:
        return _write_blocked(out_path, input_csv, "insufficient_labeled_rows", len(rows))
    feature_names = _numeric_feature_names(rows)
    if not feature_names:
        return _write_blocked(out_path, input_csv, "no_numeric_features", len(rows))

    rows = sorted(rows, key=lambda row: (row.get("window_start_utc", ""), row.get("dataset_id", "")))
    split_type = "explicit" if split_field else "temporal"
    if split_field:
        train_rows = [row for row in rows if row.get(split_field) == "train"]
        test_rows = [row for row in rows if row.get(split_field) == "test"]
        if len(train_rows) < 2 or not test_rows:
            return _write_blocked(out_path, input_csv, "invalid_explicit_split", len(rows))
    else:
        split_at = max(2, min(len(rows) - 1, int(len(rows) * train_fraction)))
        train_rows = rows[:split_at]
        test_rows = rows[split_at:]

    train_occurrence = [int(row["eventlist_target_occurred"]) for row in train_rows]
    test_occurrence = [int(row["eventlist_target_occurred"]) for row in test_rows]
    train_matrix = [[_number(row.get(name, "")) for name in feature_names] for row in train_rows]
    test_matrix = [[_number(row.get(name, "")) for name in feature_names] for row in test_rows]
    original_feature_count = len(feature_names)
    selected_indexes, feature_scores = _selected_feature_indexes(
        train_matrix,
        train_occurrence,
        feature_names,
        max_feature_count=max_feature_count,
    )
    if len(selected_indexes) != len(feature_names):
        feature_names = [feature_names[index] for index in selected_indexes]
        train_matrix = [[row[index] for index in selected_indexes] for row in train_matrix]
        test_matrix = [[row[index] for index in selected_indexes] for row in test_matrix]
    means, scales = _standardization(train_matrix)
    x_train = [_standardize(row, means=means, scales=scales) for row in train_matrix]
    x_test = [_standardize(row, means=means, scales=scales) for row in test_matrix]

    occurrence_model = _fit_occurrence_ensemble(
        x_train=x_train,
        labels=train_occurrence,
        epochs=epochs,
        learning_rate=learning_rate,
        l2=l2,
        seed=seed,
        ensemble_count=occurrence_ensemble_count,
        feature_bag_fraction=occurrence_feature_bag_fraction,
    )
    train_probabilities = [_predict_occurrence_probability(occurrence_model, row) for row in x_train]
    test_probabilities = [_predict_occurrence_probability(occurrence_model, row) for row in x_test]
    threshold = _best_balanced_threshold(train_probabilities, train_occurrence)

    count_model = _fit_linear_head(
        x_train,
        [math.log1p(_number(row.get("eventlist_target_count", ""))) for row in train_rows],
        epochs=epochs,
        learning_rate=learning_rate,
        l2=l2,
        seed=seed + 101,
    )
    magnitude_model = _fit_positive_head(
        x_train,
        train_rows,
        "eventlist_target_max_magnitude",
        epochs=epochs,
        learning_rate=learning_rate,
        l2=l2,
        seed=seed + 202,
    )
    latitude_model = _fit_positive_head(
        x_train,
        train_rows,
        "eventlist_target_centroid_latitude",
        epochs=epochs,
        learning_rate=learning_rate,
        l2=l2,
        seed=seed + 303,
    )
    longitude_model = _fit_positive_head(
        x_train,
        train_rows,
        "eventlist_target_centroid_longitude",
        epochs=epochs,
        learning_rate=learning_rate,
        l2=l2,
        seed=seed + 404,
    )

    prediction_rows = _prediction_rows(
        rows=test_rows,
        x_rows=x_test,
        probability_model=occurrence_model,
        threshold=threshold,
        count_model=count_model,
        magnitude_model=magnitude_model,
        latitude_model=latitude_model,
        longitude_model=longitude_model,
    )
    if predictions_out:
        _write_prediction_csv(predictions_out, prediction_rows)

    count_errors = [
        abs(float(row["predicted_event_count"]) - _number(row.get("target_event_count", "")))
        for row in prediction_rows
    ]
    positive_prediction_rows = [row for row in prediction_rows if row.get("target_occurred") == "1"]
    magnitude_errors = [
        abs(float(row["predicted_max_magnitude"]) - _number(row.get("target_max_magnitude", "")))
        for row in positive_prediction_rows
    ]
    location_errors = [
        _number(row.get("location_error_km", ""))
        for row in positive_prediction_rows
        if row.get("location_error_km", "") != ""
    ]
    report: dict[str, object] = {
        "schema": "elfquake.synthetic_event_list_model.v1",
        "status": "evaluated",
        "input_csv": str(input_csv),
        "predictions_csv": str(predictions_out) if predictions_out else "",
        "row_count": len(rows),
        "train_row_count": len(train_rows),
        "test_row_count": len(test_rows),
        "feature_count": len(feature_names),
        "original_feature_count": original_feature_count,
        "feature_selection": {
            "max_feature_count": max_feature_count,
            "selected_feature_count": len(feature_names),
            "method": "train_split_standardized_mean_delta",
            "top_selected_features": feature_scores[:12],
        },
        "split": {"type": split_type, "split_field": split_field},
        "train_fraction": train_fraction,
        "epochs": epochs,
        "learning_rate": learning_rate,
        "l2": l2,
        "occurrence_ensemble": {
            "ensemble_count": occurrence_ensemble_count,
            "feature_bag_fraction": occurrence_feature_bag_fraction,
            "member_count": len(occurrence_model),
        },
        "target_positive_count": sum(train_occurrence) + sum(test_occurrence),
        "train_positive_count": sum(train_occurrence),
        "test_positive_count": sum(test_occurrence),
        "occurrence": {
            "threshold": round(threshold, 6),
            "train_metrics": _metrics(train_probabilities, train_occurrence, threshold=threshold),
            "test_metrics": _metrics(test_probabilities, test_occurrence, threshold=threshold),
        },
        "count": {
            "target": "eventlist_target_count",
            "test_mae": _mean(count_errors),
            "test_rmse": _rmse(count_errors),
        },
        "max_magnitude": {
            "target": "eventlist_target_max_magnitude",
            "positive_test_mae": _mean(magnitude_errors),
            "positive_test_count": len(magnitude_errors),
        },
        "centroid": {
            "targets": ["eventlist_target_centroid_latitude", "eventlist_target_centroid_longitude"],
            "positive_test_mean_error_km": _mean(location_errors),
            "positive_test_median_error_km": _median(location_errors),
        },
        "top_occurrence_features": _top_ensemble_features(feature_names, occurrence_model, limit=12),
        "note": "Synthetic event-list heads are engineering adapters for count/location/magnitude targets, not real prediction evidence.",
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    return report


def _read_labeled_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return [row for row in csv.DictReader(handle) if row.get("eventlist_target_status") == "labeled"]


def _numeric_feature_names(rows: list[dict[str, str]]) -> list[str]:
    names: list[str] = []
    for name in rows[0].keys():
        if is_diagnostic_or_target_field(name):
            continue
        values = [_number(row.get(name, "")) for row in rows]
        if any(value != 0.0 for value in values):
            names.append(name)
    return names


def is_diagnostic_or_target_field(name: str) -> bool:
    return (
        name in ID_FIELDS
        or name in TARGET_FIELDS
        or name.startswith(TARGET_PREFIXES)
        or name.startswith(DIAGNOSTIC_PREFIXES)
    )


def _fit_positive_head(
    x_rows: list[list[float]],
    rows: list[dict[str, str]],
    target_field: str,
    *,
    epochs: int,
    learning_rate: float,
    l2: float,
    seed: int,
) -> tuple[list[float], float] | None:
    filtered = [
        (x_row, _number(row.get(target_field, "")))
        for x_row, row in zip(x_rows, rows)
        if row.get("eventlist_target_occurred") == "1" and row.get(target_field, "") != ""
    ]
    if not filtered:
        return None
    return _fit_linear_head(
        [item[0] for item in filtered],
        [item[1] for item in filtered],
        epochs=epochs,
        learning_rate=learning_rate,
        l2=l2,
        seed=seed,
    )


OccurrenceMember = tuple[list[int], list[float], float]


def _fit_occurrence_ensemble(
    *,
    x_train: list[list[float]],
    labels: list[int],
    epochs: int,
    learning_rate: float,
    l2: float,
    seed: int,
    ensemble_count: int,
    feature_bag_fraction: float,
) -> list[OccurrenceMember]:
    feature_count = len(x_train[0])
    bag_size = max(1, min(feature_count, int(round(feature_count * feature_bag_fraction))))
    models: list[OccurrenceMember] = []
    for member_index in range(ensemble_count):
        indexes = _occurrence_feature_bag(
            feature_count=feature_count,
            bag_size=bag_size,
            seed=seed + member_index * 9973,
            use_all=member_index == 0 and (feature_bag_fraction == 1.0 or ensemble_count > 1),
        )
        member_x = [[row[index] for index in indexes] for row in x_train]
        weights, bias = _fit_logistic(
            member_x,
            labels,
            epochs=epochs,
            learning_rate=learning_rate,
            l2=l2,
            seed=seed + member_index * 101,
        )
        models.append((indexes, weights, bias))
    return models


def _occurrence_feature_bag(*, feature_count: int, bag_size: int, seed: int, use_all: bool) -> list[int]:
    if use_all:
        return list(range(feature_count))
    rng = random.Random(seed)
    return sorted(rng.sample(range(feature_count), bag_size))


def _predict_occurrence_probability(model: list[OccurrenceMember], x_row: list[float]) -> float:
    probabilities = []
    for indexes, weights, bias in model:
        values = [x_row[index] for index in indexes]
        probabilities.append(_sigmoid(_dot(weights, values) + bias))
    return sum(probabilities) / max(1, len(probabilities))


def _selected_feature_indexes(
    matrix: list[list[float]],
    labels: list[int],
    feature_names: list[str],
    *,
    max_feature_count: int,
) -> tuple[list[int], list[dict[str, object]]]:
    if max_feature_count == 0 or max_feature_count >= len(feature_names):
        return list(range(len(feature_names))), _feature_scores(matrix, labels, feature_names)[:12]
    positives = sum(labels)
    negatives = len(labels) - positives
    if positives == 0 or negatives == 0:
        return list(range(len(feature_names))), _feature_scores(matrix, labels, feature_names)[:12]
    scored = _feature_scores(matrix, labels, feature_names)
    selected = sorted(int(row["index"]) for row in scored[:max_feature_count])
    return selected, [row for row in scored if int(row["index"]) in set(selected)][:12]


def _feature_scores(
    matrix: list[list[float]],
    labels: list[int],
    feature_names: list[str],
) -> list[dict[str, object]]:
    if not matrix:
        return []
    positive_indexes = [index for index, label in enumerate(labels) if label == 1]
    negative_indexes = [index for index, label in enumerate(labels) if label == 0]
    rows: list[dict[str, object]] = []
    for feature_index, name in enumerate(feature_names):
        values = [row[feature_index] for row in matrix]
        mean = sum(values) / max(1, len(values))
        variance = sum((value - mean) ** 2 for value in values) / max(1, len(values))
        scale = math.sqrt(variance) if variance > 0 else 1.0
        positive_mean = _indexed_mean(values, positive_indexes)
        negative_mean = _indexed_mean(values, negative_indexes)
        score = abs(positive_mean - negative_mean) / scale
        rows.append(
            {
                "index": feature_index,
                "name": name,
                "score": round(score, 6),
            }
        )
    return sorted(rows, key=lambda row: (float(row["score"]), str(row["name"])), reverse=True)


def _indexed_mean(values: list[float], indexes: list[int]) -> float:
    if not indexes:
        return 0.0
    return sum(values[index] for index in indexes) / len(indexes)


def _fit_linear_head(
    x_rows: list[list[float]],
    targets: list[float],
    *,
    epochs: int,
    learning_rate: float,
    l2: float,
    seed: int,
) -> tuple[list[float], float]:
    rng = random.Random(seed)
    feature_count = len(x_rows[0])
    weights = [rng.uniform(-0.01, 0.01) for _ in range(feature_count)]
    bias = sum(targets) / max(1, len(targets))
    for _ in range(epochs):
        grad_w = [0.0 for _ in weights]
        grad_b = 0.0
        for values, target in zip(x_rows, targets):
            error = _dot(weights, values) + bias - target
            grad_b += error
            for index, value in enumerate(values):
                grad_w[index] += error * value
        divisor = max(1, len(targets))
        for index in range(len(weights)):
            weights[index] -= learning_rate * ((grad_w[index] / divisor) + l2 * weights[index])
        bias -= learning_rate * grad_b / divisor
    return weights, bias


def _prediction_rows(
    *,
    rows: list[dict[str, str]],
    x_rows: list[list[float]],
    probability_model: list[OccurrenceMember],
    threshold: float,
    count_model: tuple[list[float], float],
    magnitude_model: tuple[list[float], float] | None,
    latitude_model: tuple[list[float], float] | None,
    longitude_model: tuple[list[float], float] | None,
) -> list[dict[str, str]]:
    output: list[dict[str, str]] = []
    for index, (row, x_row) in enumerate(zip(rows, x_rows), start=1):
        probability = _predict_occurrence_probability(probability_model, x_row)
        predicted_count = _safe_count_prediction(_predict_linear(count_model, x_row))
        predicted_magnitude = _clamp_optional(_predict_optional(magnitude_model, x_row), lower=0.0, upper=8.0)
        predicted_latitude = _clamp_optional(_predict_optional(latitude_model, x_row), lower=35.0, upper=48.0)
        predicted_longitude = _clamp_optional(_predict_optional(longitude_model, x_row), lower=6.0, upper=19.0)
        location_error = ""
        if (
            row.get("eventlist_target_centroid_latitude", "") != ""
            and row.get("eventlist_target_centroid_longitude", "") != ""
            and predicted_latitude is not None
            and predicted_longitude is not None
        ):
            location_error = f"{_haversine_km(_number(row['eventlist_target_centroid_latitude']), _number(row['eventlist_target_centroid_longitude']), predicted_latitude, predicted_longitude):.6f}"
        output.append(
            {
                "row_index": str(index),
                "dataset_id": row.get("dataset_id", ""),
                "window_id": row.get("window_id", ""),
                "window_start_utc": row.get("window_start_utc", ""),
                "target_occurred": row.get("eventlist_target_occurred", ""),
                "predicted_probability": f"{probability:.6f}",
                "predicted_occurred": "1" if probability >= threshold else "0",
                "target_event_count": row.get("eventlist_target_count", ""),
                "predicted_event_count": f"{predicted_count:.6f}",
                "target_max_magnitude": row.get("eventlist_target_max_magnitude", ""),
                "predicted_max_magnitude": _fmt_optional(predicted_magnitude),
                "target_centroid_latitude": row.get("eventlist_target_centroid_latitude", ""),
                "target_centroid_longitude": row.get("eventlist_target_centroid_longitude", ""),
                "predicted_centroid_latitude": _fmt_optional(predicted_latitude),
                "predicted_centroid_longitude": _fmt_optional(predicted_longitude),
                "location_error_km": location_error,
            }
        )
    return output


def _write_prediction_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys()) if rows else ["row_index"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _predict_linear(model: tuple[list[float], float], x_row: list[float]) -> float:
    weights, bias = model
    return _dot(weights, x_row) + bias


def _predict_optional(model: tuple[list[float], float] | None, x_row: list[float]) -> float | None:
    return _predict_linear(model, x_row) if model is not None else None


def _safe_count_prediction(log_count: float) -> float:
    if not math.isfinite(log_count):
        return 0.0
    return min(100.0, max(0.0, math.expm1(min(16.0, max(-16.0, log_count)))))


def _clamp_optional(value: float | None, *, lower: float, upper: float) -> float | None:
    if value is None or not math.isfinite(value):
        return None
    return min(upper, max(lower, value))


def _top_features(feature_names: list[str], weights: list[float], *, limit: int) -> list[dict[str, object]]:
    return sorted(
        [{"name": name, "weight": round(weight, 6)} for name, weight in zip(feature_names, weights)],
        key=lambda item: abs(float(item["weight"])),
        reverse=True,
    )[:limit]


def _top_ensemble_features(
    feature_names: list[str],
    model: list[OccurrenceMember],
    *,
    limit: int,
) -> list[dict[str, object]]:
    totals = [0.0 for _ in feature_names]
    counts = [0 for _ in feature_names]
    for indexes, weights, _bias in model:
        for index, weight in zip(indexes, weights):
            totals[index] += weight
            counts[index] += 1
    rows = []
    for index, name in enumerate(feature_names):
        if counts[index] == 0:
            continue
        rows.append({"name": name, "weight": round(totals[index] / counts[index], 6)})
    return sorted(rows, key=lambda item: abs(float(item["weight"])), reverse=True)[:limit]


def _mean(values: list[float]) -> float | None:
    if not values:
        return None
    return round(sum(values) / len(values), 6)


def _rmse(errors: list[float]) -> float | None:
    if not errors:
        return None
    return round(math.sqrt(sum(error * error for error in errors) / len(errors)), 6)


def _median(values: list[float]) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return round(ordered[middle], 6)
    return round((ordered[middle - 1] + ordered[middle]) / 2.0, 6)


def _fmt_optional(value: float | None) -> str:
    return "" if value is None else f"{value:.6f}"


def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    radius_km = 6371.0
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    d_phi = math.radians(lat2 - lat1)
    d_lambda = math.radians(lon2 - lon1)
    a = math.sin(d_phi / 2.0) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(d_lambda / 2.0) ** 2
    return radius_km * 2.0 * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))


def _write_blocked(out_path: Path, input_csv: Path, status: str, row_count: int) -> dict[str, object]:
    report = {
        "schema": "elfquake.synthetic_event_list_model.v1",
        "status": status,
        "input_csv": str(input_csv),
        "row_count": row_count,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    return report

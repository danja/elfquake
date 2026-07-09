"""Diagnostics for temporal and regime drift in synthetic target tables."""

from __future__ import annotations

import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

from elfquake.models.synthetic_event_list_model import is_diagnostic_or_target_field


def diagnose_synthetic_drift(
    *,
    input_csv: Path,
    out_path: Path,
    csv_out_path: Path | None = None,
    target_field: str = "eventlist_target_occurred",
    target_status_field: str = "eventlist_target_status",
    group_field: str = "dataset_id",
    time_field: str = "window_start_utc",
    train_fraction: float = 0.8,
    bucket_count: int = 10,
    top_n: int = 20,
) -> dict[str, object]:
    if not 0 < train_fraction < 1:
        raise ValueError("train_fraction must be between 0 and 1")
    if bucket_count < 2:
        raise ValueError("bucket_count must be at least 2")

    rows, fieldnames = _read_rows(input_csv)
    labeled = [
        row
        for row in rows
        if row.get(target_field) in {"0", "1"}
        and (not target_status_field or row.get(target_status_field, "labeled") == "labeled")
    ]
    labeled = sorted(labeled, key=lambda row: (row.get(time_field, ""), row.get(group_field, "")))
    split_at = max(1, min(len(labeled) - 1, int(len(labeled) * train_fraction))) if len(labeled) > 1 else len(labeled)
    train_rows = labeled[:split_at]
    test_rows = labeled[split_at:]

    bucket_rows = _time_buckets(labeled, group_field=group_field, time_field=time_field, bucket_count=bucket_count)
    feature_names = _numeric_feature_names(labeled, fieldnames)
    feature_drift = _feature_drift(train_rows, test_rows, feature_names, target_field=target_field, top_n=top_n)
    bucket_csv_rows = _bucket_csv_rows(bucket_rows, target_field=target_field)
    if csv_out_path:
        _write_bucket_csv(csv_out_path, bucket_csv_rows)

    report = {
        "schema": "elfquake.synthetic_drift_diagnostic.v1",
        "input_csv": str(input_csv),
        "csv_out": str(csv_out_path) if csv_out_path else "",
        "row_count": len(rows),
        "labeled_row_count": len(labeled),
        "target_field": target_field,
        "group_field": group_field,
        "time_field": time_field,
        "train_fraction": train_fraction,
        "bucket_count": bucket_count,
        "overall": _label_summary(labeled, target_field),
        "temporal_split": {
            "train": _label_summary(train_rows, target_field),
            "test": _label_summary(test_rows, target_field),
            "positive_rate_delta": _rate_delta(train_rows, test_rows, target_field),
            "warning": _split_warning(train_rows, test_rows, target_field),
        },
        "groups": {
            group_id: _label_summary(group_rows, target_field)
            for group_id, group_rows in sorted(_group_rows(labeled, group_field).items())
        },
        "time_buckets": bucket_csv_rows,
        "top_feature_drift": feature_drift,
        "guidance": [
            "Use temporal_split as the conservative validation warning.",
            "Use balanced or episode splits only to test learnability, not forecasting validity.",
            "Do not include diagnostic regime or episode fields as model features.",
        ],
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def _read_rows(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        return list(reader), list(reader.fieldnames or [])


def _group_rows(rows: list[dict[str, str]], group_field: str) -> dict[str, list[dict[str, str]]]:
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[row.get(group_field, "")].append(row)
    return grouped


def _time_buckets(
    rows: list[dict[str, str]],
    *,
    group_field: str,
    time_field: str,
    bucket_count: int,
) -> list[tuple[str, int, list[dict[str, str]]]]:
    output: list[tuple[str, int, list[dict[str, str]]]] = []
    for group_id, group_rows in sorted(_group_rows(rows, group_field).items()):
        ordered = sorted(group_rows, key=lambda row: row.get(time_field, ""))
        buckets: dict[int, list[dict[str, str]]] = defaultdict(list)
        for index, row in enumerate(ordered):
            bucket = min(bucket_count - 1, int(index * bucket_count / max(1, len(ordered))))
            buckets[bucket].append(row)
        for bucket in range(bucket_count):
            output.append((group_id, bucket, buckets.get(bucket, [])))
    return output


def _bucket_csv_rows(bucket_rows: list[tuple[str, int, list[dict[str, str]]]], *, target_field: str) -> list[dict[str, object]]:
    return [
        {
            "group_id": group_id,
            "bucket_index": bucket,
            **_label_summary(rows, target_field),
            "start_utc": rows[0].get("window_start_utc", "") if rows else "",
            "end_utc": rows[-1].get("window_start_utc", "") if rows else "",
        }
        for group_id, bucket, rows in bucket_rows
    ]


def _write_bucket_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "group_id",
        "bucket_index",
        "row_count",
        "positive_count",
        "negative_count",
        "positive_rate",
        "start_utc",
        "end_utc",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _label_summary(rows: list[dict[str, str]], target_field: str) -> dict[str, object]:
    counts = Counter(row.get(target_field, "") for row in rows)
    positives = counts.get("1", 0)
    negatives = counts.get("0", 0)
    total = positives + negatives
    return {
        "row_count": len(rows),
        "positive_count": positives,
        "negative_count": negatives,
        "positive_rate": round(positives / total, 6) if total else None,
    }


def _rate_delta(train_rows: list[dict[str, str]], test_rows: list[dict[str, str]], target_field: str) -> float | None:
    train = _label_summary(train_rows, target_field)["positive_rate"]
    test = _label_summary(test_rows, target_field)["positive_rate"]
    if train is None or test is None:
        return None
    return round(float(test) - float(train), 6)


def _split_warning(train_rows: list[dict[str, str]], test_rows: list[dict[str, str]], target_field: str) -> str:
    train_summary = _label_summary(train_rows, target_field)
    test_summary = _label_summary(test_rows, target_field)
    if test_summary["negative_count"] == 0:
        return "test_split_has_no_negatives"
    if test_summary["positive_count"] == 0:
        return "test_split_has_no_positives"
    delta = _rate_delta(train_rows, test_rows, target_field)
    if delta is not None and abs(delta) >= 0.25:
        return "large_positive_rate_shift"
    if train_summary["negative_count"] == 0 or train_summary["positive_count"] == 0:
        return "train_split_one_class"
    return "ok"


def _numeric_feature_names(rows: list[dict[str, str]], fieldnames: list[str]) -> list[str]:
    names: list[str] = []
    for name in fieldnames:
        if is_diagnostic_or_target_field(name):
            continue
        values = [_number(row.get(name, "")) for row in rows]
        if any(value != 0.0 for value in values):
            names.append(name)
    return names


def _feature_drift(
    train_rows: list[dict[str, str]],
    test_rows: list[dict[str, str]],
    feature_names: list[str],
    *,
    target_field: str,
    top_n: int,
) -> list[dict[str, object]]:
    output: list[dict[str, object]] = []
    for name in feature_names:
        train_values = [_number(row.get(name, "")) for row in train_rows]
        test_values = [_number(row.get(name, "")) for row in test_rows]
        if not train_values or not test_values:
            continue
        train_mean = sum(train_values) / len(train_values)
        test_mean = sum(test_values) / len(test_values)
        pooled_scale = math.sqrt((_variance(train_values) + _variance(test_values)) / 2.0) or 1.0
        output.append(
            {
                "feature": name,
                "train_mean": round(train_mean, 6),
                "test_mean": round(test_mean, 6),
                "mean_delta": round(test_mean - train_mean, 6),
                "standardized_mean_delta": round((test_mean - train_mean) / pooled_scale, 6),
                "train_target_correlation": _correlation(train_rows, name, target_field),
                "test_target_correlation": _correlation(test_rows, name, target_field),
            }
        )
    return sorted(output, key=lambda item: abs(float(item["standardized_mean_delta"])), reverse=True)[:top_n]


def _variance(values: list[float]) -> float:
    if not values:
        return 0.0
    mean = sum(values) / len(values)
    return sum((value - mean) ** 2 for value in values) / len(values)


def _correlation(rows: list[dict[str, str]], feature: str, target_field: str) -> float | None:
    pairs = [
        (_number(row.get(feature, "")), float(row[target_field]))
        for row in rows
        if row.get(target_field) in {"0", "1"}
    ]
    if len(pairs) < 2:
        return None
    xs = [pair[0] for pair in pairs]
    ys = [pair[1] for pair in pairs]
    x_mean = sum(xs) / len(xs)
    y_mean = sum(ys) / len(ys)
    x_var = sum((value - x_mean) ** 2 for value in xs)
    y_var = sum((value - y_mean) ** 2 for value in ys)
    if x_var == 0.0 or y_var == 0.0:
        return None
    numerator = sum((x - x_mean) * (y - y_mean) for x, y in pairs)
    return round(numerator / math.sqrt(x_var * y_var), 6)


def _number(value: str) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return 0.0
    return parsed if math.isfinite(parsed) else 0.0

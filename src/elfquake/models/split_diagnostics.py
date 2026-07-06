"""Diagnostics for chronological train/test splits."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path

from elfquake.models.readiness import ID_FIELDS, TARGET_FIELDS


FEATURE_ROW_FIELDS = [
    "feature",
    "train_mean",
    "test_mean",
    "train_std",
    "standardized_mean_delta",
    "train_missing_rate",
    "test_missing_rate",
    "train_correlation",
    "test_correlation",
    "correlation_delta",
    "direction_changed",
    "train_positive_mean",
    "train_negative_mean",
    "test_positive_mean",
    "test_negative_mean",
]


def diagnose_temporal_split(
    *,
    input_csv: Path,
    out_path: Path,
    feature_out: Path | None = None,
    time_field: str = "window_start_utc",
    train_fraction: float = 0.8,
    target_field: str = "target_occurred",
    top_n: int = 20,
) -> dict[str, object]:
    if not 0 < train_fraction < 1:
        raise ValueError("train_fraction must be between 0 and 1")
    if top_n < 1:
        raise ValueError("top_n must be at least 1")

    rows, fieldnames = _read_rows_and_fields(input_csv)
    labeled = sorted(
        [row for row in rows if row.get(target_field) in {"0", "1"}],
        key=lambda row: row.get(time_field, ""),
    )
    report: dict[str, object] = {
        "schema": "elfquake.temporal_split_diagnostics.v1",
        "input": str(input_csv),
        "row_count": len(rows),
        "labeled_row_count": len(labeled),
        "time_field": time_field,
        "target_field": target_field,
        "train_fraction": train_fraction,
    }
    if len(labeled) < 4:
        report["status"] = "insufficient_labeled_rows"
        return _write_report(out_path, report, feature_out, [])

    train_count = min(len(labeled) - 1, max(2, int(len(labeled) * train_fraction)))
    train_rows = labeled[:train_count]
    test_rows = labeled[train_count:]
    train_labels = [int(row[target_field]) for row in train_rows]
    test_labels = [int(row[target_field]) for row in test_rows]
    features = _feature_names(labeled, fieldnames, target_field)
    feature_rows = [
        _feature_diagnostic(name, train_rows, test_rows, train_labels, test_labels, target_field)
        for name in features
    ]
    feature_rows.sort(
        key=lambda row: (
            -abs(_float(row["standardized_mean_delta"])),
            -abs(_float(row["correlation_delta"])),
            row["feature"],
        )
    )

    report.update(
        {
            "status": "evaluated",
            "train_row_count": len(train_rows),
            "test_row_count": len(test_rows),
            "train_time_start": train_rows[0].get(time_field, ""),
            "train_time_end": train_rows[-1].get(time_field, ""),
            "test_time_start": test_rows[0].get(time_field, ""),
            "test_time_end": test_rows[-1].get(time_field, ""),
            "train_positive_count": sum(train_labels),
            "train_negative_count": len(train_labels) - sum(train_labels),
            "test_positive_count": sum(test_labels),
            "test_negative_count": len(test_labels) - sum(test_labels),
            "train_positive_rate": _rate(sum(train_labels), len(train_labels)),
            "test_positive_rate": _rate(sum(test_labels), len(test_labels)),
            "positive_rate_delta": _rate(sum(test_labels), len(test_labels))
            - _rate(sum(train_labels), len(train_labels)),
            "feature_count": len(feature_rows),
            "top_feature_drifts": feature_rows[:top_n],
            "top_correlation_shifts": sorted(
                feature_rows,
                key=lambda row: (-abs(_float(row["correlation_delta"])), row["feature"]),
            )[:top_n],
            "direction_changes": [
                row for row in feature_rows if row["direction_changed"] == "1"
            ][:top_n],
        }
    )
    return _write_report(out_path, report, feature_out, feature_rows)


def _feature_diagnostic(
    feature: str,
    train_rows: list[dict[str, str]],
    test_rows: list[dict[str, str]],
    train_labels: list[int],
    test_labels: list[int],
    target_field: str,
) -> dict[str, str]:
    train_values = _present_values(train_rows, feature)
    test_values = _present_values(test_rows, feature)
    train_mean = _mean(train_values)
    test_mean = _mean(test_values)
    train_std = _std(train_values, train_mean)
    train_corr = _correlation(train_rows, feature, target_field)
    test_corr = _correlation(test_rows, feature, target_field)
    return {
        "feature": feature,
        "train_mean": _fmt(train_mean),
        "test_mean": _fmt(test_mean),
        "train_std": _fmt(train_std),
        "standardized_mean_delta": _fmt((test_mean - train_mean) / train_std if train_std > 1e-12 else 0.0),
        "train_missing_rate": _fmt(_missing_rate(train_rows, feature)),
        "test_missing_rate": _fmt(_missing_rate(test_rows, feature)),
        "train_correlation": _fmt(train_corr),
        "test_correlation": _fmt(test_corr),
        "correlation_delta": _fmt(test_corr - train_corr),
        "direction_changed": "1" if train_corr * test_corr < 0 else "0",
        "train_positive_mean": _fmt(_label_mean(train_rows, train_labels, feature, label=1)),
        "train_negative_mean": _fmt(_label_mean(train_rows, train_labels, feature, label=0)),
        "test_positive_mean": _fmt(_label_mean(test_rows, test_labels, feature, label=1)),
        "test_negative_mean": _fmt(_label_mean(test_rows, test_labels, feature, label=0)),
    }


def _feature_names(rows: list[dict[str, str]], fieldnames: list[str], target_field: str) -> list[str]:
    excluded = TARGET_FIELDS | ID_FIELDS | {target_field}
    names = []
    for field in fieldnames:
        if field in excluded:
            continue
        present = [row.get(field, "") for row in rows if row.get(field, "") != ""]
        if present and all(_is_float(value) for value in present):
            names.append(field)
    return names


def _correlation(rows: list[dict[str, str]], feature: str, target_field: str) -> float:
    pairs = [
        (float(row[feature]), float(row[target_field]))
        for row in rows
        if row.get(feature, "") != "" and row.get(target_field, "") in {"0", "1"}
    ]
    if len(pairs) < 2:
        return 0.0
    values = [value for value, _ in pairs]
    labels = [label for _, label in pairs]
    value_mean = _mean(values)
    label_mean = _mean(labels)
    value_std = _std(values, value_mean)
    label_std = _std(labels, label_mean)
    if value_std <= 1e-12 or label_std <= 1e-12:
        return 0.0
    covariance = sum((value - value_mean) * (label - label_mean) for value, label in pairs) / len(pairs)
    return covariance / (value_std * label_std)


def _label_mean(rows: list[dict[str, str]], labels: list[int], feature: str, *, label: int) -> float:
    values = [
        float(row[feature])
        for row, row_label in zip(rows, labels)
        if row_label == label and row.get(feature, "") != ""
    ]
    return _mean(values)


def _present_values(rows: list[dict[str, str]], feature: str) -> list[float]:
    return [float(row[feature]) for row in rows if row.get(feature, "") != ""]


def _missing_rate(rows: list[dict[str, str]], feature: str) -> float:
    return _rate(sum(1 for row in rows if row.get(feature, "") == ""), len(rows))


def _rate(count: int, total: int) -> float:
    return count / total if total else 0.0


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _std(values: list[float], mean: float) -> float:
    return math.sqrt(sum((value - mean) ** 2 for value in values) / len(values)) if values else 0.0


def _is_float(value: str) -> bool:
    try:
        float(value)
        return True
    except ValueError:
        return False


def _float(value: str) -> float:
    try:
        return float(value)
    except ValueError:
        return 0.0


def _fmt(value: float) -> str:
    return f"{value:.9f}"


def _read_rows_and_fields(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        return list(reader), list(reader.fieldnames or [])


def _write_report(
    out_path: Path,
    report: dict[str, object],
    feature_out: Path | None,
    feature_rows: list[dict[str, str]],
) -> dict[str, object]:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if feature_out is not None:
        feature_out.parent.mkdir(parents=True, exist_ok=True)
        with feature_out.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=FEATURE_ROW_FIELDS, lineterminator="\n")
            writer.writeheader()
            writer.writerows(feature_rows)
    return report

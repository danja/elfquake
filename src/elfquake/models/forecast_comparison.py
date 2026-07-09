"""Compare weekly forecast event-list artifacts against scaffold criteria."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path


def compare_weekly_forecasts(
    *,
    baseline_report: Path,
    baseline_events: Path,
    candidate_report: Path,
    candidate_events: Path,
    out_path: Path,
    csv_out_path: Path | None = None,
) -> dict[str, object]:
    baseline_report_payload = _read_json(baseline_report)
    candidate_report_payload = _read_json(candidate_report)
    baseline_rows = _read_event_rows(baseline_events)
    candidate_rows = _read_event_rows(candidate_events)
    nearest_distances = _nearest_distances(candidate_rows, baseline_rows)
    report: dict[str, object] = {
        "schema": "elfquake.weekly_forecast_comparison.v1",
        "status": "evaluated",
        "warning": "Compares engineering forecast artifacts; not a validation of prediction capability.",
        "baseline": _summary(
            report=baseline_report_payload,
            rows=baseline_rows,
            report_path=baseline_report,
            events_path=baseline_events,
        ),
        "candidate": _summary(
            report=candidate_report_payload,
            rows=candidate_rows,
            report_path=candidate_report,
            events_path=candidate_events,
        ),
        "delta": _delta(baseline_report_payload, baseline_rows, candidate_report_payload, candidate_rows),
        "spatial_similarity": {
            "candidate_to_baseline_nearest_km_mean": round(sum(nearest_distances) / len(nearest_distances), 6) if nearest_distances else None,
            "candidate_to_baseline_nearest_km_max": round(max(nearest_distances), 6) if nearest_distances else None,
            "candidate_rows_within_25km_of_baseline_count": sum(1 for value in nearest_distances if value <= 25.0),
            "candidate_rows_within_50km_of_baseline_count": sum(1 for value in nearest_distances if value <= 50.0),
        },
        "criteria": _criteria(candidate_report_payload, candidate_rows),
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    if csv_out_path is not None:
        _write_rows(csv_out_path, _comparison_rows(report))
    return report


def _summary(*, report: dict[str, object], rows: list[dict[str, str]], report_path: Path, events_path: Path) -> dict[str, object]:
    probabilities = [_number(row.get("probability_proxy", "")) for row in rows]
    magnitudes = [_number(row.get("magnitude_proxy", "")) for row in rows]
    return {
        "report_path": str(report_path),
        "events_path": str(events_path),
        "schema": report.get("schema", ""),
        "model_type": _model_type(report),
        "forecast_start_utc": report.get("forecast_start_utc", ""),
        "forecast_end_utc": report.get("forecast_end_utc", ""),
        "event_count": len(rows),
        "uncapped_expected_event_count": report.get("uncapped_expected_event_count"),
        "probability_mean": _mean(probabilities),
        "probability_max": _max(probabilities),
        "magnitude_mean": _mean(magnitudes),
        "magnitude_max": _max(magnitudes),
        "learned_balanced_accuracy": _learned_metric(report, "balanced_accuracy"),
        "learned_positive_recall": _learned_metric(report, "positive_recall"),
        "learned_negative_recall": _learned_metric(report, "negative_recall"),
        "warning": report.get("warning", ""),
    }


def _delta(
    baseline_report: dict[str, object],
    baseline_rows: list[dict[str, str]],
    candidate_report: dict[str, object],
    candidate_rows: list[dict[str, str]],
) -> dict[str, object]:
    baseline_expected = _number(str(baseline_report.get("uncapped_expected_event_count", "")))
    candidate_expected = _number(str(candidate_report.get("uncapped_expected_event_count", "")))
    return {
        "event_count": len(candidate_rows) - len(baseline_rows),
        "uncapped_expected_event_count": round(candidate_expected - baseline_expected, 6),
        "probability_mean": round((_mean([_number(row.get("probability_proxy", "")) for row in candidate_rows]) or 0.0) - (_mean([_number(row.get("probability_proxy", "")) for row in baseline_rows]) or 0.0), 6),
        "magnitude_mean": round((_mean([_number(row.get("magnitude_proxy", "")) for row in candidate_rows]) or 0.0) - (_mean([_number(row.get("magnitude_proxy", "")) for row in baseline_rows]) or 0.0), 6),
    }


def _criteria(report: dict[str, object], rows: list[dict[str, str]]) -> dict[str, object]:
    scorer = _learned_scorer(report)
    balanced_accuracy = _metric_from_scorer(scorer, "balanced_accuracy")
    positive_recall = _metric_from_scorer(scorer, "positive_recall")
    negative_recall = _metric_from_scorer(scorer, "negative_recall")
    has_contract_fields = all(
        field in rows[0]
        for field in [
            "prediction_id",
            "forecast_time_utc",
            "latitude",
            "longitude",
            "magnitude_proxy",
            "probability_proxy",
            "warning",
        ]
    ) if rows else False
    synthetic_pass = (
        balanced_accuracy is not None
        and positive_recall is not None
        and negative_recall is not None
        and balanced_accuracy >= 0.60
        and positive_recall >= 0.40
        and negative_recall >= 0.40
    )
    return {
        "stage_1_event_contract_pass": has_contract_fields and bool(rows),
        "stage_2_synthetic_model_pass": synthetic_pass,
        "stage_2_balanced_accuracy_min": 0.60,
        "stage_2_recall_min": 0.40,
        "observed_balanced_accuracy": balanced_accuracy,
        "observed_positive_recall": positive_recall,
        "observed_negative_recall": negative_recall,
        "prediction_claim_allowed": False,
        "reason": "Stage 2 synthetic scorer criteria are not met, and real supervised rows remain class-blocked.",
    }


def _comparison_rows(report: dict[str, object]) -> list[dict[str, object]]:
    rows = []
    for label in ["baseline", "candidate"]:
        item = report[label]
        rows.append(
            {
                "forecast": label,
                "model_type": item["model_type"],
                "event_count": item["event_count"],
                "uncapped_expected_event_count": item["uncapped_expected_event_count"],
                "probability_mean": item["probability_mean"],
                "magnitude_mean": item["magnitude_mean"],
                "learned_balanced_accuracy": item["learned_balanced_accuracy"],
                "learned_positive_recall": item["learned_positive_recall"],
                "learned_negative_recall": item["learned_negative_recall"],
            }
        )
    return rows


def _nearest_distances(source: list[dict[str, str]], target: list[dict[str, str]]) -> list[float]:
    target_points = [_point(row) for row in target]
    target_points = [point for point in target_points if point is not None]
    distances = []
    for row in source:
        point = _point(row)
        if point is None or not target_points:
            continue
        distances.append(min(_haversine_km(point[0], point[1], other[0], other[1]) for other in target_points))
    return distances


def _point(row: dict[str, str]) -> tuple[float, float] | None:
    latitude = _number(row.get("latitude", ""))
    longitude = _number(row.get("longitude", ""))
    if latitude == 0.0 and longitude == 0.0:
        return None
    return latitude, longitude


def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    radius = 6371.0088
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    value = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    return 2 * radius * math.asin(math.sqrt(value))


def _read_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_event_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _model_type(report: dict[str, object]) -> str:
    model = report.get("model")
    if isinstance(model, dict):
        return str(model.get("type", ""))
    return ""


def _learned_scorer(report: dict[str, object]) -> dict[str, object]:
    model = report.get("model")
    if isinstance(model, dict):
        scorer = model.get("learned_scorer")
        if isinstance(scorer, dict):
            return scorer
    return {}


def _learned_metric(report: dict[str, object], name: str) -> float | None:
    return _metric_from_scorer(_learned_scorer(report), name)


def _metric_from_scorer(scorer: dict[str, object], name: str) -> float | None:
    metrics = scorer.get("test_metrics")
    if not isinstance(metrics, dict):
        return None
    value = metrics.get(name)
    if value is None:
        return None
    return _number(str(value))


def _number(value: str) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return 0.0
    return parsed if math.isfinite(parsed) else 0.0


def _mean(values: list[float]) -> float | None:
    return round(sum(values) / len(values), 6) if values else None


def _max(values: list[float]) -> float | None:
    return round(max(values), 6) if values else None

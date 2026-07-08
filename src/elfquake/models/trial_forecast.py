"""Trial end-to-end multimodal weekly event forecast.

This module deliberately produces a weak, inspectable baseline artifact.  It
uses every currently available source family, but it should be replaced by a
trained sequence model once real labels are adequate.
"""

from __future__ import annotations

import csv
import glob
import json
import math
import random
import re
from collections import defaultdict
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from typing import Iterable

from elfquake.features.common import format_utc, parse_utc


ITALY_LAT_RANGE = (35.0, 48.0)
ITALY_LON_RANGE = (6.0, 19.0)


@dataclass(frozen=True)
class EventPoint:
    time_utc: str
    latitude: float
    longitude: float
    magnitude: float
    source: str


@dataclass(frozen=True)
class SourceContext:
    score: float
    row_count: int
    latest_time_utc: str
    details: dict[str, object]


def generate_trial_weekly_event_forecast(
    *,
    real_events_csv: Path,
    out_path: Path,
    events_out_path: Path,
    as_of_utc: str,
    horizon_days: int = 7,
    magnitude_threshold: float = 2.0,
    max_events: int = 25,
    seed: int = 42,
    synthetic_event_globs: list[str] | None = None,
    vlf_window_csvs: list[Path] | None = None,
    vlf_anomaly_report: Path | None = None,
    vlf_audio_globs: list[str] | None = None,
    astronomy_globs: list[str] | None = None,
) -> dict[str, object]:
    if horizon_days < 1:
        raise ValueError("horizon_days must be at least 1")
    if max_events < 1:
        raise ValueError("max_events must be at least 1")

    as_of = parse_utc(as_of_utc)
    forecast_end = as_of + timedelta(days=horizon_days)
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
        historical_weekly_count=historical["expected_event_count"],
        vlf_score=vlf_context.score,
        astronomy_score=astronomy_context.score,
        synthetic_score=synthetic_context.score,
    )
    event_count = max(1, min(max_events, int(round(expected_count))))

    spatial_bins = _spatial_bins(target_real_events, synthetic_events)
    predictions = _forecast_points(
        spatial_bins=spatial_bins,
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
        synthetic_score=synthetic_context.score,
    )
    _write_prediction_csv(events_out_path, predictions)
    report: dict[str, object] = {
        "schema": "elfquake.trial_multimodal_weekly_event_forecast.v1",
        "status": "trial_run",
        "warning": "Engineering baseline only: not validated as earthquake prediction capability.",
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
            "type": "deterministic_multimodal_trial_ensemble",
            "spatial_mix": "75% historical INGV density, 25% synthetic avalanche density when both are present",
            "count_mix": "historical INGV weekly rate modulated by current VLF, astronomy, and synthetic stress context",
            "downstream_contract": "CSV rows contain predicted event time, latitude, longitude, magnitude proxy, probability, and source contribution fields.",
        },
        "predictions_preview": predictions[:5],
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    return report


def _load_event_points(path: Path, *, source: str, before_utc: str = "") -> list[EventPoint]:
    if not path.exists():
        return []
    cutoff = parse_utc(before_utc) if before_utc else None
    points: list[EventPoint] = []
    seen: set[str] = set()
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            time_utc = row.get("event_time_utc", "")
            if cutoff is not None and time_utc:
                try:
                    if parse_utc(time_utc) >= cutoff:
                        continue
                except ValueError:
                    continue
            event_id = row.get("event_id", "") or f"{path}:{len(points)}"
            if event_id in seen:
                continue
            seen.add(event_id)
            latitude = _float(row.get("latitude", ""))
            longitude = _float(row.get("longitude", ""))
            magnitude = _float(row.get("magnitude", ""))
            if latitude is None or longitude is None or magnitude is None:
                continue
            if not (ITALY_LAT_RANGE[0] <= latitude <= ITALY_LAT_RANGE[1] and ITALY_LON_RANGE[0] <= longitude <= ITALY_LON_RANGE[1]):
                continue
            points.append(EventPoint(time_utc=time_utc, latitude=latitude, longitude=longitude, magnitude=magnitude, source=source))
    return points


def _historical_rate(events: list[EventPoint], *, as_of_utc: str, horizon_days: int) -> dict[str, object]:
    if not events:
        return {
            "history_start_utc": "",
            "history_day_count": 0.0,
            "expected_event_count": 1.0,
            "recent_expected_event_count": 1.0,
            "probability_at_least_one": 0.632121,
        }
    as_of = parse_utc(as_of_utc)
    event_times = [parse_utc(event.time_utc) for event in events if event.time_utc]
    start = min(event_times)
    history_days = max(1.0, (as_of - start).total_seconds() / 86400.0)
    all_rate = len(events) / history_days * horizon_days
    recent_start = as_of - timedelta(days=90)
    recent_count = sum(1 for event_time in event_times if recent_start <= event_time < as_of)
    recent_rate = recent_count / 90.0 * horizon_days
    expected = 0.65 * recent_rate + 0.35 * all_rate if recent_count else all_rate
    return {
        "history_start_utc": format_utc(start),
        "history_day_count": round(history_days, 3),
        "expected_event_count": round(expected, 6),
        "recent_expected_event_count": round(recent_rate, 6),
        "all_history_expected_event_count": round(all_rate, 6),
        "probability_at_least_one": round(1.0 - math.exp(-max(0.0, expected)), 6),
    }


def _load_vlf_context(window_paths: list[Path], *, anomaly_report: Path, audio_paths: list[Path]) -> SourceContext:
    scores: list[float] = []
    latest_time = ""
    row_count = 0
    numeric_latest: list[float] = []
    for path in window_paths:
        if not path.exists():
            continue
        rows = _read_csv_rows(path)
        row_count += len(rows)
        if not rows:
            continue
        latest = sorted(rows, key=lambda row: row.get("window_end_utc", ""))[-1]
        latest_time = max(latest_time, latest.get("window_end_utc", ""))
        for key, value in latest.items():
            if ("vlf" in key or "real_vlf_image" in key) and not key.startswith("quality_"):
                numeric = _float(value)
                if numeric is not None:
                    numeric_latest.append(numeric)
    if numeric_latest:
        scores.append(_robust_positive_score(numeric_latest))
    if anomaly_report.exists():
        try:
            report = json.loads(anomaly_report.read_text(encoding="utf-8"))
            forecast = report.get("forecast", {}) if isinstance(report, dict) else {}
            probability = _float(str(forecast.get("demo_probability", ""))) if isinstance(forecast, dict) else None
            if probability is None:
                probability = _float(str(report.get("latest_demo_probability", ""))) if isinstance(report, dict) else None
            if probability is not None:
                scores.append(_clamp(probability, 0.0, 1.0))
            if isinstance(forecast, dict):
                latest_time = max(latest_time, str(forecast.get("forecast_start_utc", "")))
        except json.JSONDecodeError:
            pass
    audio_rows = [row for path in audio_paths for row in _read_csv_rows(path)]
    row_count += len(audio_rows)
    unreadable = 0
    for row in audio_rows:
        latest_time = max(latest_time, row.get("vlf_audio_captured_at_utc", ""))
        unreadable += int(_float(row.get("quality_unreadable_vlf_audio", "0")) or 0)
    if audio_rows:
        readable_fraction = 1.0 - unreadable / max(1, len(audio_rows))
        scores.append(_clamp(readable_fraction, 0.0, 1.0))
    score = sum(scores) / len(scores) if scores else 0.0
    return SourceContext(
        score=score,
        row_count=row_count,
        latest_time_utc=latest_time,
        details={
            "window_path_count": len([path for path in window_paths if path.exists()]),
            "audio_path_count": len(audio_paths),
            "anomaly_report": str(anomaly_report) if anomaly_report.exists() else "",
            "context_component_count": len(scores),
        },
    )


def _load_astronomy_context(paths: list[Path]) -> SourceContext:
    f107_values: list[float] = []
    moon_phase_count = 0
    latest_time = ""
    for path in paths:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        latest_time = max(latest_time, _path_capture_time(path))
        if isinstance(payload, list):
            for item in payload:
                if isinstance(item, dict):
                    value = _float(str(item.get("f10.7", "")))
                    if value is not None:
                        f107_values.append(value)
        elif isinstance(payload, dict):
            phase_data = payload.get("phasedata")
            if isinstance(phase_data, list):
                moon_phase_count += len(phase_data)
    f107_score = 0.0
    if f107_values:
        latest = f107_values[-1]
        low = min(f107_values)
        high = max(f107_values)
        f107_score = (latest - low) / (high - low) if high > low else 0.0
    moon_score = _clamp(moon_phase_count / 16.0, 0.0, 1.0) if moon_phase_count else 0.0
    score = 0.8 * f107_score + 0.2 * moon_score if f107_values or moon_phase_count else 0.0
    return SourceContext(
        score=_clamp(score, 0.0, 1.0),
        row_count=len(paths),
        latest_time_utc=latest_time,
        details={
            "json_path_count": len(paths),
            "f107_value_count": len(f107_values),
            "latest_f107": round(f107_values[-1], 6) if f107_values else None,
            "moon_phase_count": moon_phase_count,
        },
    )


def _synthetic_context(events: list[EventPoint]) -> SourceContext:
    magnitudes = [event.magnitude for event in events]
    if not magnitudes:
        return SourceContext(score=0.0, row_count=0, latest_time_utc="", details={"max_magnitude": None})
    max_mag = max(magnitudes)
    mean_top = sum(sorted(magnitudes)[-max(1, len(magnitudes) // 20):]) / max(1, len(magnitudes) // 20)
    score = _clamp((mean_top - 2.0) / 2.0, 0.0, 1.0)
    return SourceContext(
        score=score,
        row_count=len(events),
        latest_time_utc=max((event.time_utc for event in events), default=""),
        details={
            "max_magnitude": round(max_mag, 6),
            "mean_top_5pct_magnitude": round(mean_top, 6),
            "stress_context_score": round(score, 6),
        },
    )


def _blend_expected_count(*, historical_weekly_count: float, vlf_score: float, astronomy_score: float, synthetic_score: float) -> float:
    multiplier = 0.7 + 0.45 * vlf_score + 0.2 * astronomy_score + 0.25 * synthetic_score
    return max(1.0, historical_weekly_count * _clamp(multiplier, 0.4, 1.8))


def _spatial_bins(real_events: list[EventPoint], synthetic_events: list[EventPoint]) -> list[dict[str, float]]:
    real_density = _normalized_density(real_events)
    synthetic_density = _normalized_density(synthetic_events)
    keys = sorted(set(real_density) | set(synthetic_density))
    if not keys:
        return [{"latitude": 42.5, "longitude": 13.0, "weight": 1.0, "real_weight": 0.0, "synthetic_weight": 0.0}]
    real_mix = 0.75 if real_density and synthetic_density else (1.0 if real_density else 0.0)
    synthetic_mix = 1.0 - real_mix
    bins = []
    total = 0.0
    for key in keys:
        lat, lon = key
        real_weight = real_density.get(key, 0.0)
        synthetic_weight = synthetic_density.get(key, 0.0)
        weight = real_mix * real_weight + synthetic_mix * synthetic_weight
        total += weight
        bins.append(
            {
                "latitude": lat,
                "longitude": lon,
                "weight": weight,
                "real_weight": real_weight,
                "synthetic_weight": synthetic_weight,
            }
        )
    if total <= 0:
        total = 1.0
    for item in bins:
        item["weight"] /= total
    return sorted(bins, key=lambda item: item["weight"], reverse=True)


def _normalized_density(events: list[EventPoint]) -> dict[tuple[float, float], float]:
    weights: dict[tuple[float, float], float] = defaultdict(float)
    for event in events:
        key = (round(event.latitude * 4.0) / 4.0, round(event.longitude * 4.0) / 4.0)
        weights[key] += max(0.1, event.magnitude - 1.5)
    total = sum(weights.values())
    if total <= 0:
        return {}
    return {key: value / total for key, value in weights.items()}


def _forecast_points(
    *,
    spatial_bins: list[dict[str, float]],
    count: int,
    start,
    end,
    seed: int,
    magnitude_threshold: float,
    real_events: list[EventPoint],
    synthetic_events: list[EventPoint],
    expected_count: float,
    vlf_score: float,
    astronomy_score: float,
    synthetic_score: float,
) -> list[dict[str, object]]:
    rng = random.Random(seed)
    real_magnitudes = [event.magnitude for event in real_events] or [magnitude_threshold + 0.1]
    synthetic_magnitudes = [event.magnitude for event in synthetic_events] or real_magnitudes
    weights = [item["weight"] for item in spatial_bins]
    duration_seconds = max(1.0, (end - start).total_seconds())
    predictions: list[dict[str, object]] = []
    for index in range(count):
        chosen = rng.choices(spatial_bins, weights=weights, k=1)[0]
        offset = (index + 0.5) / count * duration_seconds
        jitter = rng.uniform(-0.18, 0.18) * duration_seconds / count
        event_time = start + timedelta(seconds=max(0.0, min(duration_seconds - 1.0, offset + jitter)))
        real_mag = _quantile(real_magnitudes, 0.55 + 0.35 * rng.random())
        synthetic_mag = _quantile(synthetic_magnitudes, 0.55 + 0.35 * rng.random())
        magnitude = max(magnitude_threshold + 0.01, 0.75 * real_mag + 0.25 * synthetic_mag)
        rank_score = 1.0 - index / max(1, count)
        probability = _clamp(0.15 + 0.35 * chosen["weight"] * len(spatial_bins) + 0.25 * rank_score + 0.15 * vlf_score + 0.05 * astronomy_score + 0.05 * synthetic_score, 0.01, 0.99)
        predictions.append(
            {
                "prediction_id": f"trial_{index + 1:03d}",
                "forecast_time_utc": format_utc(event_time),
                "latitude": round(chosen["latitude"] + rng.uniform(-0.08, 0.08), 5),
                "longitude": round(chosen["longitude"] + rng.uniform(-0.08, 0.08), 5),
                "magnitude_proxy": round(magnitude, 2),
                "probability_proxy": round(probability, 6),
                "expected_week_count": round(expected_count, 6),
                "real_spatial_weight": round(chosen["real_weight"], 6),
                "synthetic_spatial_weight": round(chosen["synthetic_weight"], 6),
                "vlf_context_score": round(vlf_score, 6),
                "astronomy_context_score": round(astronomy_score, 6),
                "synthetic_context_score": round(synthetic_score, 6),
                "warning": "trial baseline; not validated prediction",
            }
        )
    return sorted(predictions, key=lambda row: row["forecast_time_utc"])


def _write_prediction_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "prediction_id",
        "forecast_time_utc",
        "latitude",
        "longitude",
        "magnitude_proxy",
        "probability_proxy",
        "expected_week_count",
        "real_spatial_weight",
        "synthetic_spatial_weight",
        "vlf_context_score",
        "astronomy_context_score",
        "synthetic_context_score",
        "warning",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _expand_globs(patterns: Iterable[str]) -> list[Path]:
    paths: list[Path] = []
    for pattern in patterns:
        matches = glob.glob(pattern, recursive=True)
        paths.extend(Path(match) for match in matches)
    return sorted(set(paths))


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _robust_positive_score(values: list[float]) -> float:
    positives = [abs(value) for value in values if math.isfinite(value)]
    if not positives:
        return 0.0
    median = _quantile(positives, 0.5)
    upper = _quantile(positives, 0.9)
    if upper <= 0:
        return 0.0
    return _clamp(median / upper, 0.0, 1.0)


def _quantile(values: list[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = _clamp(fraction, 0.0, 1.0) * (len(ordered) - 1)
    lower = int(math.floor(index))
    upper = int(math.ceil(index))
    if lower == upper:
        return ordered[lower]
    return ordered[lower] * (upper - index) + ordered[upper] * (index - lower)


def _path_capture_time(path: Path) -> str:
    match = re.search(r"(\d{4}-\d{2}-\d{2}T\d{2}-\d{2}-\d{2}Z)", path.name)
    if match:
        date_part, time_part = match.group(1).split("T", 1)
        return f"{date_part}T{time_part.replace('-', ':')}"
    return ""


def _float(value: str) -> float | None:
    try:
        if value == "":
            return None
        result = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(result):
        return None
    return result


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))

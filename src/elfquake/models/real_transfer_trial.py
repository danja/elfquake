"""Synthetic-to-real spatial weekly earthquake trial.

This is an engineering baseline, not a prediction claim.  It deliberately
uses fixed spatial cells so a low weekly magnitude threshold has both positive
and negative examples, unlike an all-Italy weekly occurrence label.
"""

from __future__ import annotations

import csv
import json
import math
import random
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

from elfquake.visualization.event_map import DEFAULT_BASEMAP_GEOJSON


ITALY_LAT = (35.0, 47.8)
ITALY_LON = (5.5, 19.5)


@dataclass(frozen=True)
class Event:
    time: datetime
    latitude: float
    longitude: float
    magnitude: float


@dataclass(frozen=True)
class Sample:
    week_start: datetime
    lat: float
    lon: float
    values: tuple[float, ...]
    label: int


def run_real_transfer_trial(
    *,
    real_events_csv: Path,
    synthetic_event_csvs: list[Path],
    out_dir: Path,
    magnitude_threshold: float = 2.5,
    horizon_days: int = 7,
    cell_degrees: float = 1.5,
    train_fraction: float = 0.8,
    pretrain_epochs: int = 30,
    finetune_epochs: int = 80,
    seed: int = 42,
) -> dict[str, object]:
    """Pretrain an MLP on synthetic cells then fine-tune on chronological INGV cells."""
    if not 0 < train_fraction < 1:
        raise ValueError("train_fraction must be between zero and one")
    if horizon_days < 1 or cell_degrees <= 0:
        raise ValueError("horizon_days and cell_degrees must be positive")
    real_events = _read_events(real_events_csv)
    synthetic_events = _read_synthetic_corpus(synthetic_event_csvs)
    if len(real_events) < 10:
        raise ValueError("real event catalog is too small")
    cells = _cells(cell_degrees)
    real_samples = _weekly_samples(real_events, cells, magnitude_threshold, horizon_days, cell_degrees)
    synthetic_samples = _weekly_samples(synthetic_events, cells, magnitude_threshold, horizon_days, cell_degrees)
    train_count = int(len({sample.week_start for sample in real_samples}) * train_fraction)
    week_starts = sorted({sample.week_start for sample in real_samples})
    train_weeks = set(week_starts[:train_count])
    train = [sample for sample in real_samples if sample.week_start in train_weeks]
    test = [sample for sample in real_samples if sample.week_start not in train_weeks]
    if not train or not test or len({sample.label for sample in train}) != 2 or len({sample.label for sample in test}) != 2:
        raise ValueError("chosen threshold/grid has insufficient real train/test class variation")

    torch = _torch()
    _seed(torch, seed)
    model = _model(torch, len(train[0].values))
    synthetic_status = "skipped_insufficient_class_variation"
    if synthetic_samples and len({sample.label for sample in synthetic_samples}) == 2:
        _fit(torch, model, synthetic_samples, pretrain_epochs, seed)
        synthetic_status = "pretrained"
    _fit(torch, model, train, finetune_epochs, seed + 1)
    train_probabilities = _predict(torch, model, train)
    test_probabilities = _predict(torch, model, test)
    threshold = _best_threshold(train_probabilities, [sample.label for sample in train])
    metrics = _metrics(test_probabilities, [sample.label for sample in test], threshold)
    baseline_probabilities = _historical_cell_rates(train, test)
    baseline_threshold = _best_threshold(_historical_cell_rates(train, train), [sample.label for sample in train])
    baseline_metrics = _metrics(baseline_probabilities, [sample.label for sample in test], baseline_threshold)

    chosen_week = random.Random(seed).choice(sorted({sample.week_start for sample in test}))
    week_samples = [sample for sample, probability in zip(test, test_probabilities) if sample.week_start == chosen_week]
    week_probabilities = [probability for sample, probability in zip(test, test_probabilities) if sample.week_start == chosen_week]
    actual = [
        event for event in real_events
        if chosen_week <= event.time < chosen_week + timedelta(days=horizon_days)
        and event.magnitude >= magnitude_threshold
    ]
    predictions = _week_predictions(week_samples, week_probabilities, threshold, magnitude_threshold)
    out_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(out_dir / "heldout_week_predictions.csv", predictions)
    _write_csv(out_dir / "heldout_week_actual_events.csv", [_event_row(event) for event in actual])
    report = {
        "schema": "elfquake.real_transfer_spatial_weekly_trial.v1",
        "status": "evaluated",
        "warning": "Experimental held-out baseline only. It does not demonstrate earthquake prediction capability.",
        "task": {
            "scope": "Italy fixed spatial cells",
            "horizon_days": horizon_days,
            "magnitude_threshold": magnitude_threshold,
            "cell_degrees": cell_degrees,
            "target": "whether a cell contains at least one qualifying event in the following weekly horizon",
        },
        "data": {
            "real_events_csv": str(real_events_csv),
            "real_event_count": len(real_events),
            "synthetic_event_csvs": [str(path) for path in synthetic_event_csvs],
            "synthetic_event_count": len(synthetic_events),
            "synthetic_sample_count": len(synthetic_samples),
            "real_train_sample_count": len(train),
            "real_test_sample_count": len(test),
            "real_train_weeks": len(train_weeks),
            "real_test_weeks": len(week_starts) - len(train_weeks),
            "real_train_time_end": _format(max(train_weeks)),
            "real_test_time_start": _format(min(sample.week_start for sample in test)),
            "vlf": "represented by an explicit missing-modality mask; current VLF history does not cover this real holdout",
            "astronomy": "represented by an explicit missing-modality mask; no validated historical aligned series is available for this trial",
        },
        "model": {
            "type": "CPU PyTorch MLP",
            "transfer": f"synthetic_pretraining={synthetic_status}; chronological_real_finetuning=yes",
            "feature_names": _feature_names(),
            "missing_modality_features": ["vlf_present", "astro_present"],
        },
        "evaluation": {
            "selection": "threshold calibrated only on the chronological real training partition",
            "threshold": round(threshold, 6),
            **metrics,
            "historical_spatial_rate_baseline": {"threshold": round(baseline_threshold, 6), **baseline_metrics},
        },
        "heldout_visualization_week": {
            "start_utc": _format(chosen_week),
            "end_utc": _format(chosen_week + timedelta(days=horizon_days)),
            "actual_event_count": len(actual),
            "predicted_cell_count": len(predictions),
            "actual_events_csv": str(out_dir / "heldout_week_actual_events.csv"),
            "predictions_csv": str(out_dir / "heldout_week_predictions.csv"),
        },
    }
    (out_dir / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def _read_events(path: Path) -> list[Event]:
    events: list[Event] = []
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            try:
                event = Event(_parse(row["event_time_utc"]), float(row["latitude"]), float(row["longitude"]), float(row["magnitude"]))
            except (KeyError, ValueError):
                continue
            if ITALY_LAT[0] <= event.latitude <= ITALY_LAT[1] and ITALY_LON[0] <= event.longitude <= ITALY_LON[1]:
                events.append(event)
    return sorted(events, key=lambda event: event.time)


def _read_synthetic_corpus(paths: list[Path]) -> list[Event]:
    """Stack independent episodes without letting their shared clock overlap.

    Simulation runs use a common demonstration start date. Offsetting each
    episode by 21 days is a modeling transform only; source paths remain in
    the report and no real timestamps are modified.
    """
    events: list[Event] = []
    for index, path in enumerate(paths):
        offset = timedelta(days=21 * index)
        events.extend(Event(event.time + offset, event.latitude, event.longitude, event.magnitude) for event in _read_events(path))
    return sorted(events, key=lambda event: event.time)


def _cells(size: float) -> list[tuple[float, float]]:
    return [
        (min(ITALY_LAT[1], lat + size / 2), min(ITALY_LON[1], lon + size / 2))
        for lat in _steps(ITALY_LAT[0], ITALY_LAT[1], size)
        for lon in _steps(ITALY_LON[0], ITALY_LON[1], size)
        if _inside_italy(lon + size / 2, lat + size / 2)
    ]


def _inside_italy(lon: float, lat: float) -> bool:
    """Use the same offline Italy outline as the map, avoiding sea/grid cells."""
    payload = json.loads(DEFAULT_BASEMAP_GEOJSON.read_text(encoding="utf-8"))
    coordinates = payload["features"][0]["geometry"]["coordinates"]
    return any(_inside_ring(lon, lat, polygon[0]) for polygon in coordinates)


def _inside_ring(lon: float, lat: float, ring: list[list[float]]) -> bool:
    inside = False
    previous_lon, previous_lat = ring[-1]
    for current_lon, current_lat in ring:
        crosses = (current_lat > lat) != (previous_lat > lat)
        if crosses and lon < (previous_lon - current_lon) * (lat - current_lat) / (previous_lat - current_lat) + current_lon:
            inside = not inside
        previous_lon, previous_lat = current_lon, current_lat
    return inside


def _steps(start: float, end: float, size: float) -> list[float]:
    values: list[float] = []
    value = start
    while value < end:
        values.append(value)
        value += size
    return values


def _weekly_samples(
    events: list[Event],
    cells: list[tuple[float, float]],
    threshold: float,
    horizon_days: int,
    cell_degrees: float,
    feature_mode: str = "compact",
) -> list[Sample]:
    if not events:
        return []
    start = datetime.combine(events[0].time.date(), datetime.min.time(), tzinfo=timezone.utc)
    end = datetime.combine(events[-1].time.date(), datetime.min.time(), tzinfo=timezone.utc) - timedelta(days=horizon_days)
    samples: list[Sample] = []
    week = start
    while week <= end:
        for lat, lon in cells:
            history_7 = _in_cell(events, week - timedelta(days=7), week, lat, lon, cell_degrees)
            history_28 = _in_cell(events, week - timedelta(days=28), week, lat, lon, cell_degrees)
            target = _in_cell(events, week, week + timedelta(days=horizon_days), lat, lon, cell_degrees)
            if feature_mode == "compact":
                values = _features(history_7, history_28, lat, lon)
            elif feature_mode == "multiscale":
                values = _multiscale_features(events, week, lat, lon, cell_degrees)
            else:
                raise ValueError(f"unknown feature mode: {feature_mode}")
            label = int(any(event.magnitude >= threshold for event in target))
            samples.append(Sample(week, lat, lon, values, label))
        week += timedelta(days=horizon_days)
    return samples


def _in_cell(events: list[Event], start: datetime, end: datetime, lat: float, lon: float, cell_degrees: float) -> list[Event]:
    half = cell_degrees / 2.0
    return [event for event in events if start <= event.time < end and abs(event.latitude - lat) <= half and abs(event.longitude - lon) <= half]


def _features(short: list[Event], long: list[Event], lat: float, lon: float) -> tuple[float, ...]:
    magnitudes = [event.magnitude for event in long]
    energy = sum(10 ** (1.5 * magnitude) for magnitude in magnitudes)
    return (
        math.log1p(len(short)), math.log1p(len(long)), max(magnitudes, default=0.0) / 6.0,
        math.log1p(energy) / 12.0, (lat - ITALY_LAT[0]) / (ITALY_LAT[1] - ITALY_LAT[0]),
        (lon - ITALY_LON[0]) / (ITALY_LON[1] - ITALY_LON[0]), 0.0, 0.0,
    )


def _feature_names() -> list[str]:
    return ["seismic_log_count_7d", "seismic_log_count_28d", "seismic_max_magnitude_28d", "seismic_log_energy_28d", "spatial_lat", "spatial_lon", "vlf_present", "astro_present"]


def _multiscale_features(
    events: list[Event], week: datetime, lat: float, lon: float, cell_degrees: float
) -> tuple[float, ...]:
    values: list[float] = []
    for days in (1, 3, 7, 14, 28, 90):
        local = _in_cell(events, week - timedelta(days=days), week, lat, lon, cell_degrees)
        neighbours = _in_cell(events, week - timedelta(days=days), week, lat, lon, cell_degrees * 2.0)
        magnitudes = [event.magnitude for event in local]
        energy = sum(10 ** (1.5 * magnitude) for magnitude in magnitudes)
        values.extend(
            (
                math.log1p(len(local)),
                math.log1p(len(neighbours)),
                max(magnitudes, default=0.0) / 6.0,
                math.log1p(energy) / 12.0,
            )
        )
    previous = [event for event in events if event.time < week]
    last_gap_days = (week - max((event.time for event in previous), default=week)).total_seconds() / 86400.0
    values.extend(
        (
            min(last_gap_days, 90.0) / 90.0,
            (lat - ITALY_LAT[0]) / (ITALY_LAT[1] - ITALY_LAT[0]),
            (lon - ITALY_LON[0]) / (ITALY_LON[1] - ITALY_LON[0]),
            0.0,
            0.0,
        )
    )
    return tuple(values)


def _multiscale_feature_names() -> list[str]:
    names = []
    for days in (1, 3, 7, 14, 28, 90):
        names.extend(
            (
                f"seismic_log_count_{days}d",
                f"seismic_log_neighbour_count_{days}d",
                f"seismic_max_magnitude_{days}d",
                f"seismic_log_energy_{days}d",
            )
        )
    return names + ["seismic_days_since_last_event", "spatial_lat", "spatial_lon", "vlf_present", "astro_present"]


def _torch():
    try:
        import torch
    except ImportError as error:  # pragma: no cover
        raise RuntimeError("PyTorch is required; install it in .venv") from error
    return torch


def _seed(torch, seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


def _model(torch, width: int):
    return torch.nn.Sequential(torch.nn.Linear(width, 24), torch.nn.ReLU(), torch.nn.Dropout(0.05), torch.nn.Linear(24, 1))


def _fit(torch, model, samples: list[Sample], epochs: int, seed: int) -> None:
    _seed(torch, seed)
    x = torch.tensor([sample.values for sample in samples], dtype=torch.float32)
    y = torch.tensor([sample.label for sample in samples], dtype=torch.float32)
    positives = max(1, int(y.sum().item()))
    weight = torch.tensor([(len(samples) - positives) / positives], dtype=torch.float32)
    loss = torch.nn.BCEWithLogitsLoss(pos_weight=weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.003, weight_decay=0.0005)
    model.train()
    for _ in range(epochs):
        optimizer.zero_grad()
        value = loss(model(x).squeeze(1), y)
        value.backward()
        optimizer.step()


def _predict(torch, model, samples: list[Sample]) -> list[float]:
    model.eval()
    with torch.no_grad():
        logits = model(torch.tensor([sample.values for sample in samples], dtype=torch.float32)).squeeze(1)
        return [float(value) for value in torch.sigmoid(logits)]


def _best_threshold(probabilities: list[float], labels: list[int]) -> float:
    return max((index / 100 for index in range(10, 91)), key=lambda threshold: _balanced(probabilities, labels, threshold))


def _historical_cell_rates(train: list[Sample], rows: list[Sample]) -> list[float]:
    totals: dict[tuple[float, float], list[int]] = {}
    for sample in train:
        totals.setdefault((sample.lat, sample.lon), []).append(sample.label)
    return [sum(totals[(sample.lat, sample.lon)]) / len(totals[(sample.lat, sample.lon)]) for sample in rows]


def _metrics(probabilities: list[float], labels: list[int], threshold: float) -> dict[str, object]:
    predictions = [int(value >= threshold) for value in probabilities]
    tp = sum(prediction == label == 1 for prediction, label in zip(predictions, labels))
    tn = sum(prediction == label == 0 for prediction, label in zip(predictions, labels))
    fp = sum(prediction == 1 and label == 0 for prediction, label in zip(predictions, labels))
    fn = sum(prediction == 0 and label == 1 for prediction, label in zip(predictions, labels))
    recall = tp / (tp + fn) if tp + fn else 0.0
    specificity = tn / (tn + fp) if tn + fp else 0.0
    return {"balanced_accuracy": round((recall + specificity) / 2, 6), "precision": round(tp / (tp + fp), 6) if tp + fp else 0.0, "recall": round(recall, 6), "specificity": round(specificity, 6), "confusion": {"true_positive": tp, "true_negative": tn, "false_positive": fp, "false_negative": fn}, "positive_rate": round(sum(labels) / len(labels), 6)}


def _balanced(probabilities: list[float], labels: list[int], threshold: float) -> float:
    return float(_metrics(probabilities, labels, threshold)["balanced_accuracy"])


def _week_predictions(samples: list[Sample], probabilities: list[float], threshold: float, magnitude: float) -> list[dict[str, str]]:
    selected = [(sample, probability) for sample, probability in zip(samples, probabilities) if probability >= threshold]
    selected = sorted(selected, key=lambda item: item[1], reverse=True)[:10]
    if not selected:
        selected = sorted(zip(samples, probabilities), key=lambda item: item[1], reverse=True)[:5]
    return [{"event_time_utc": _format(sample.week_start), "latitude": f"{sample.lat:.5f}", "longitude": f"{sample.lon:.5f}", "magnitude": f"{magnitude + probability:.2f}", "predicted_probability": f"{probability:.6f}", "source": "heldout_model_predicted_cell"} for sample, probability in selected]


def _event_row(event: Event) -> dict[str, str]:
    return {"event_time_utc": _format(event.time), "latitude": f"{event.latitude:.5f}", "longitude": f"{event.longitude:.5f}", "magnitude": f"{event.magnitude:.2f}", "source": "ingv_actual"}


def _write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    fields = ["event_time_utc", "latitude", "longitude", "magnitude", "predicted_probability", "source"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _parse(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(timezone.utc)


def _format(value: datetime) -> str:
    return value.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

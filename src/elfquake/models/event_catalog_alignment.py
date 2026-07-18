"""Catalog-level comparison and train-fitted mark calibration."""

from __future__ import annotations

import csv
import json
import math
import random
from pathlib import Path

from scipy.stats import ks_2samp, wasserstein_distance


ITALY_LAT = (35.0, 47.8)
ITALY_LON = (5.5, 19.5)


def compare_event_catalogs(
    *,
    real_events: Path,
    synthetic_events: list[Path],
    out_path: Path,
    cell_degrees: float = 1.5,
    synthetic_duration_days: list[float] | None = None,
) -> dict[str, object]:
    if synthetic_duration_days and len(synthetic_duration_days) != len(synthetic_events):
        raise ValueError("synthetic duration count must match synthetic catalog count")
    real = _read_catalog(real_events)
    synthetic = [_read_catalog(path) for path in synthetic_events]
    catalogs = [{"name": "real_italy", "path": str(real_events), "events": real}]
    catalogs.extend(
        {"name": f"synthetic_{index:02d}", "path": str(path), "events": events}
        for index, (path, events) in enumerate(zip(synthetic_events, synthetic), start=1)
    )
    summaries = [_summary(item["events"], cell_degrees) for item in catalogs[:1]]
    summaries.extend(
        _summary(events, cell_degrees, duration_days=duration)
        for events, duration in zip(synthetic, synthetic_duration_days or [None] * len(synthetic))
    )
    real_summary = summaries[0]
    comparisons = [
        _compare_summary(real_summary, summary, name, real_events=real, synthetic_events=item["events"])
        for item, summary in zip(catalogs[1:], summaries[1:])
        for name in [item["name"]]
    ]
    report = {
        "schema": "elfquake.event_catalog_alignment.v1",
        "method": {
            "cell_degrees": cell_degrees,
            "synthetic_duration_days": synthetic_duration_days,
            "spatial_metric": "cell occupancy and haversine nearest-neighbour distances",
            "distribution_metrics": "Kolmogorov-Smirnov and Wasserstein distances",
            "warning": "descriptive comparison only; no event identities are matched",
        },
        "catalogs": [{"name": item["name"], "path": item["path"], "summary": summary} for item, summary in zip(catalogs, summaries)],
        "comparisons_to_real": comparisons,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def combine_synthetic_catalogs(
    *, synthetic_events: list[Path], out_path: Path, offset_days: int = 21
) -> dict[str, object]:
    """Combine independent synthetic episodes without overlapping their clocks."""
    if offset_days < 1:
        raise ValueError("offset_days must be positive")
    from datetime import datetime, timedelta

    combined_rows: list[dict[str, str]] = []
    fieldnames: list[str] = []
    for episode_index, path in enumerate(synthetic_events):
        rows = _read_rows(path)
        if not fieldnames:
            fieldnames = _read_fieldnames(path)
        for row in rows:
            try:
                timestamp = datetime.fromisoformat(row["event_time_utc"].replace("Z", "+00:00"))
            except (KeyError, ValueError):
                continue
            row["event_time_utc"] = (timestamp + timedelta(days=episode_index * offset_days)).isoformat().replace("+00:00", "Z")
            row["synthetic_episode_index"] = str(episode_index)
            combined_rows.append(row)
    if "synthetic_episode_index" not in fieldnames:
        fieldnames.append("synthetic_episode_index")
    combined_rows.sort(key=lambda row: row.get("event_time_utc", ""))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(combined_rows)
    return {
        "schema": "elfquake.synthetic_catalog_combined.v1",
        "episode_count": len(synthetic_events),
        "event_count": len(combined_rows),
        "offset_days": offset_days,
        "output": str(out_path),
    }


def calibrate_synthetic_magnitudes(
    *, real_events: Path, synthetic_events: Path, out_path: Path
) -> dict[str, object]:
    real = _read_catalog(real_events)
    synthetic = _read_catalog(synthetic_events)
    real_magnitudes = sorted(event["magnitude"] for event in real)
    synthetic_magnitudes = [event["magnitude"] for event in synthetic]
    if len(real_magnitudes) < 2 or not synthetic_magnitudes:
        raise ValueError("both catalogs need enough valid magnitudes for calibration")

    fieldnames = _read_fieldnames(synthetic_events)
    if "magnitude_raw" not in fieldnames:
        fieldnames.append("magnitude_raw")
    if "magnitude_calibration" not in fieldnames:
        fieldnames.append("magnitude_calibration")
    rows = _read_rows(synthetic_events)
    for row, raw in zip(rows, synthetic_magnitudes):
        percentile = _empirical_percentile(synthetic_magnitudes, raw)
        row["magnitude_raw"] = row.get("magnitude", "")
        row["magnitude"] = f"{_quantile(real_magnitudes, percentile):.6f}"
        row["magnitude_calibration"] = "real_train_empirical_quantile"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    return {
        "schema": "elfquake.synthetic_magnitude_calibration.v1",
        "real_event_count": len(real),
        "synthetic_event_count": len(synthetic),
        "method": "empirical quantile mapping fitted on real catalog",
        "output": str(out_path),
    }


def calibrate_synthetic_catalog(
    *,
    real_events: Path,
    synthetic_events: Path,
    out_path: Path,
    seed: int = 42,
    synthetic_duration_days: float | None = None,
) -> dict[str, object]:
    """Apply train-fitted magnitude mapping and deterministic rate thinning."""
    real = _read_catalog(real_events)
    rows = _read_rows(synthetic_events)
    synthetic = _read_catalog(synthetic_events)
    real_rate = _rate_per_day(real)
    synthetic_rate = _rate_per_day(synthetic, duration_days=synthetic_duration_days)
    keep_probability = min(1.0, real_rate / synthetic_rate) if synthetic_rate else 0.0
    real_magnitudes = sorted(event["magnitude"] for event in real)
    synthetic_magnitudes = [event["magnitude"] for event in synthetic]
    if len(real_magnitudes) < 2 or not synthetic_magnitudes:
        raise ValueError("both catalogs need enough valid events for calibration")
    fieldnames = _read_fieldnames(synthetic_events)
    for field in (
        "magnitude_raw",
        "magnitude_calibration",
        "rate_keep_probability",
        "rate_calibration",
        "spatial_weight_raw",
        "spatial_weight",
        "spatial_calibration",
    ):
        if field not in fieldnames:
            fieldnames.append(field)

    real_days = max(_duration_days([event["time"] for event in real]), 1e-9)
    synthetic_days = max(
        synthetic_duration_days
        if synthetic_duration_days is not None
        else _duration_days([event["time"] for event in synthetic]),
        1e-9,
    )
    real_cell_rates = _cell_rates(real, real_days)
    synthetic_cell_rates = _cell_rates(synthetic, synthetic_days)
    raw_spatial_weights = {
        cell: real_cell_rates.get(cell, 0.0) / rate
        for cell, rate in synthetic_cell_rates.items()
    }
    mean_spatial_weight = _weighted_event_mean(synthetic, raw_spatial_weights)
    mean_spatial_weight = mean_spatial_weight or 1.0

    rng = random.Random(seed)
    output_rows = []
    kept_count = 0
    for row in rows:
        try:
            raw_magnitude = float(row["magnitude"])
        except (KeyError, TypeError, ValueError):
            continue
        percentile = _empirical_percentile(synthetic_magnitudes, raw_magnitude)
        keep = rng.random() < keep_probability
        if not keep:
            continue
        row["magnitude_raw"] = row.get("magnitude", "")
        row["magnitude"] = f"{_quantile(real_magnitudes, percentile):.6f}"
        row["magnitude_calibration"] = "real_train_empirical_quantile"
        row["rate_keep_probability"] = f"{keep_probability:.9f}"
        row["rate_calibration"] = "deterministic_global_rate_thinning"
        cell = _cell_key(float(row["latitude"]), float(row["longitude"]))
        raw_weight = raw_spatial_weights.get(cell, 0.0)
        row["spatial_weight_raw"] = f"{raw_weight:.9f}"
        row["spatial_weight"] = f"{raw_weight / mean_spatial_weight:.9f}"
        row["spatial_calibration"] = "real_train_cell_rate_over_synthetic_cell_rate"
        output_rows.append(row)
        kept_count += 1
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(output_rows)
    return {
        "schema": "elfquake.synthetic_catalog_calibration.v1",
        "real_event_count": len(real),
        "synthetic_event_count": len(synthetic),
        "retained_event_count": kept_count,
        "real_rate_per_day": real_rate,
        "synthetic_rate_per_day": synthetic_rate,
        "synthetic_duration_days": synthetic_duration_days,
        "keep_probability": keep_probability,
        "seed": seed,
        "magnitude_method": "empirical quantile mapping fitted on real catalog",
        "rate_method": "deterministic global rate thinning",
        "spatial_method": "per-event cell-rate importance weight; coordinates unchanged",
        "spatial_cells_real": len(real_cell_rates),
        "spatial_cells_synthetic": len(synthetic_cell_rates),
        "output": str(out_path),
    }


def calibrate_synthetic_spatial_coordinates(
    *, real_events: Path, synthetic_events: Path, out_path: Path
) -> dict[str, object]:
    """Map synthetic latitude/longitude marginals to real training marginals."""
    real = _read_catalog(real_events)
    rows = _read_rows(synthetic_events)
    valid = []
    for row in rows:
        try:
            valid.append((row, float(row["latitude"]), float(row["longitude"])))
        except (KeyError, TypeError, ValueError):
            continue
    if len(real) < 2 or not valid:
        raise ValueError("both catalogs need enough valid coordinates for calibration")
    real_latitudes = sorted(event["latitude"] for event in real)
    real_longitudes = sorted(event["longitude"] for event in real)
    synthetic_latitudes = [item[1] for item in valid]
    synthetic_longitudes = [item[2] for item in valid]
    fieldnames = _read_fieldnames(synthetic_events)
    for field in ("latitude_raw", "longitude_raw", "spatial_calibration"):
        if field not in fieldnames:
            fieldnames.append(field)
    for row, latitude, longitude in valid:
        row["latitude_raw"] = row.get("latitude", "")
        row["longitude_raw"] = row.get("longitude", "")
        row["latitude"] = f"{_quantile(real_latitudes, _empirical_percentile(synthetic_latitudes, latitude)):.6f}"
        row["longitude"] = f"{_quantile(real_longitudes, _empirical_percentile(synthetic_longitudes, longitude)):.6f}"
        row["spatial_calibration"] = "real_train_empirical_coordinate_quantile"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    return {
        "schema": "elfquake.synthetic_spatial_calibration.v1",
        "real_event_count": len(real),
        "synthetic_event_count": len(valid),
        "method": "independent empirical latitude/longitude quantile mapping",
        "output": str(out_path),
    }


def _read_catalog(path: Path) -> list[dict[str, float]]:
    events = []
    for row in _read_rows(path):
        try:
            latitude = float(row["latitude"])
            longitude = float(row["longitude"])
            magnitude = float(row["magnitude"])
            time = row["event_time_utc"]
        except (KeyError, TypeError, ValueError):
            continue
        if not (ITALY_LAT[0] <= latitude <= ITALY_LAT[1] and ITALY_LON[0] <= longitude <= ITALY_LON[1]):
            continue
        events.append({"latitude": latitude, "longitude": longitude, "magnitude": magnitude, "time": time})
    events.sort(key=lambda event: event["time"])
    return events


def _summary(
    events: list[dict[str, float]], cell_degrees: float, duration_days: float | None = None
) -> dict[str, object]:
    magnitudes = [event["magnitude"] for event in events]
    times = [event["time"] for event in events]
    gaps = _gaps_hours(times)
    cells = {(int((event["latitude"] - ITALY_LAT[0]) / cell_degrees), int((event["longitude"] - ITALY_LON[0]) / cell_degrees)) for event in events}
    spatial_gaps = _nearest_neighbour_km(events)
    return {
        "event_count": len(events),
        "time_start": times[0] if times else "",
        "time_end": times[-1] if times else "",
        "duration_days": duration_days if duration_days is not None else _duration_days(times),
        "rate_per_day": _rate_per_day(events, duration_days=duration_days),
        "interevent_hours": _distribution(gaps),
        "magnitude": _distribution(magnitudes),
        "energy_proxy": _distribution([10.0 ** (1.5 * magnitude) for magnitude in magnitudes]),
        "spatial_cell_count": len(cells),
        "spatial_cell_occupancy": sorted(f"{lat}:{lon}" for lat, lon in cells),
        "nearest_neighbour_km": _distribution(spatial_gaps),
    }


def _rate_per_day(events: list[dict[str, float]], duration_days: float | None = None) -> float:
    times = [event["time"] for event in events]
    duration = duration_days if duration_days is not None else _duration_days(times)
    return len(events) / max(duration, 1e-9) if events else 0.0


def _cell_rates(events: list[dict[str, float]], duration_days: float) -> dict[tuple[int, int], float]:
    counts: dict[tuple[int, int], int] = {}
    for event in events:
        cell = _cell_key(event["latitude"], event["longitude"])
        counts[cell] = counts.get(cell, 0) + 1
    return {cell: count / duration_days for cell, count in counts.items()}


def _weighted_event_mean(events: list[dict[str, float]], weights: dict[tuple[int, int], float]) -> float:
    if not events:
        return 0.0
    return sum(weights.get(_cell_key(event["latitude"], event["longitude"]), 0.0) for event in events) / len(events)


def _cell_key(latitude: float, longitude: float, cell_degrees: float = 1.5) -> tuple[int, int]:
    return (int((latitude - ITALY_LAT[0]) / cell_degrees), int((longitude - ITALY_LON[0]) / cell_degrees))


def _compare_summary(
    real: dict[str, object],
    synthetic: dict[str, object],
    name: str,
    *,
    real_events: list[dict[str, float]],
    synthetic_events: list[dict[str, float]],
) -> dict[str, object]:
    real_magnitudes = real["magnitude"]["values"]
    synthetic_magnitudes = synthetic["magnitude"]["values"]
    real_gaps = real["interevent_hours"]["values"]
    synthetic_gaps = synthetic["interevent_hours"]["values"]
    return {
        "synthetic": name,
        "event_count_ratio": _ratio(synthetic["event_count"], real["event_count"]),
        "rate_ratio": _ratio(synthetic["rate_per_day"], real["rate_per_day"]),
        "magnitude_wasserstein": _distance(real_magnitudes, synthetic_magnitudes),
        "magnitude_ks": _ks(real_magnitudes, synthetic_magnitudes),
        "interevent_hours_wasserstein": _distance(real_gaps, synthetic_gaps),
        "interevent_hours_ks": _ks(real_gaps, synthetic_gaps),
        "nearest_neighbour_km_wasserstein": _distance(real["nearest_neighbour_km"]["values"], synthetic["nearest_neighbour_km"]["values"]),
        "sample_matched_nearest_neighbour_km_wasserstein": _sample_matched_neighbour_distance(
            real_events, synthetic_events
        ),
        "shared_spatial_cell_count": len(set(real["spatial_cell_occupancy"]) & set(synthetic["spatial_cell_occupancy"])),
    }


def _sample_matched_neighbour_distance(
    real_events: list[dict[str, float]], synthetic_events: list[dict[str, float]], repeats: int = 32
) -> float | None:
    """Compare nearest-neighbour distances after matching catalog sample size."""
    sample_size = len(synthetic_events)
    if sample_size < 2 or len(real_events) < sample_size:
        return None
    synthetic_distances = _nearest_neighbour_km(synthetic_events)
    rng = random.Random(42)
    distances = []
    for _ in range(repeats):
        sampled_real = rng.sample(real_events, sample_size)
        distances.append(_distance(_nearest_neighbour_km(sampled_real), synthetic_distances) or 0.0)
    return sum(distances) / len(distances) if distances else None


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _read_fieldnames(path: Path) -> list[str]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.reader(handle).__next__())


def _gaps_hours(times: list[str]) -> list[float]:
    from datetime import datetime

    parsed = [datetime.fromisoformat(time.replace("Z", "+00:00")) for time in times]
    return [(right - left).total_seconds() / 3600.0 for left, right in zip(parsed, parsed[1:])]


def _duration_days(times: list[str]) -> float:
    from datetime import datetime

    if len(times) < 2:
        return 0.0
    start = datetime.fromisoformat(times[0].replace("Z", "+00:00"))
    end = datetime.fromisoformat(times[-1].replace("Z", "+00:00"))
    return max((end - start).total_seconds() / 86400.0, 0.0)


def _nearest_neighbour_km(events: list[dict[str, float]]) -> list[float]:
    if len(events) < 2:
        return []
    return [min(_haversine(event, other) for other in events if other is not event) for event in events]


def _haversine(left: dict[str, float], right: dict[str, float]) -> float:
    lat1, lat2 = math.radians(left["latitude"]), math.radians(right["latitude"])
    dlat = lat2 - lat1
    dlon = math.radians(right["longitude"] - left["longitude"])
    value = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    return 6371.0 * 2.0 * math.asin(min(1.0, math.sqrt(value)))


def _distribution(values: list[float]) -> dict[str, object]:
    ordered = sorted(values)
    return {"count": len(ordered), "values": ordered, "p50": _quantile(ordered, 0.5), "p90": _quantile(ordered, 0.9)}


def _quantile(values: list[float], percentile: float) -> float:
    if not values:
        return 0.0
    if len(values) == 1:
        return values[0]
    position = max(0.0, min(1.0, percentile)) * (len(values) - 1)
    lower = int(position)
    upper = min(lower + 1, len(values) - 1)
    return values[lower] + (values[upper] - values[lower]) * (position - lower)


def _empirical_percentile(values: list[float], value: float) -> float:
    if len(values) == 1:
        return 0.5
    below = sum(item <= value for item in values) - 1
    return max(0.0, min(1.0, below / (len(values) - 1)))


def _distance(left: list[float], right: list[float]) -> float | None:
    return float(wasserstein_distance(left, right)) if left and right else None


def _ks(left: list[float], right: list[float]) -> float | None:
    return float(ks_2samp(left, right).statistic) if left and right else None


def _ratio(value: float, reference: float) -> float | None:
    return float(value / reference) if reference else None

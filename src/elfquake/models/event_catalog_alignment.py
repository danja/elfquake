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
    *, real_events: Path, synthetic_events: list[Path], out_path: Path, cell_degrees: float = 1.5
) -> dict[str, object]:
    real = _read_catalog(real_events)
    synthetic = [_read_catalog(path) for path in synthetic_events]
    catalogs = [{"name": "real_italy", "path": str(real_events), "events": real}]
    catalogs.extend(
        {"name": f"synthetic_{index:02d}", "path": str(path), "events": events}
        for index, (path, events) in enumerate(zip(synthetic_events, synthetic), start=1)
    )
    summaries = [_summary(item["events"], cell_degrees) for item in catalogs]
    real_summary = summaries[0]
    comparisons = [
        _compare_summary(real_summary, summary, name)
        for name, summary in ((item["name"], summary) for item, summary in zip(catalogs[1:], summaries[1:]))
    ]
    report = {
        "schema": "elfquake.event_catalog_alignment.v1",
        "method": {
            "cell_degrees": cell_degrees,
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
    *, real_events: Path, synthetic_events: Path, out_path: Path, seed: int = 42
) -> dict[str, object]:
    """Apply train-fitted magnitude mapping and deterministic rate thinning."""
    real = _read_catalog(real_events)
    rows = _read_rows(synthetic_events)
    synthetic = _read_catalog(synthetic_events)
    real_rate = _rate_per_day(real)
    synthetic_rate = _rate_per_day(synthetic)
    keep_probability = min(1.0, real_rate / synthetic_rate) if synthetic_rate else 0.0
    real_magnitudes = sorted(event["magnitude"] for event in real)
    synthetic_magnitudes = [event["magnitude"] for event in synthetic]
    if len(real_magnitudes) < 2 or not synthetic_magnitudes:
        raise ValueError("both catalogs need enough valid events for calibration")
    fieldnames = _read_fieldnames(synthetic_events)
    for field in ("magnitude_raw", "magnitude_calibration", "rate_keep_probability", "rate_calibration"):
        if field not in fieldnames:
            fieldnames.append(field)

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
        "keep_probability": keep_probability,
        "seed": seed,
        "magnitude_method": "empirical quantile mapping fitted on real catalog",
        "rate_method": "deterministic global rate thinning",
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


def _summary(events: list[dict[str, float]], cell_degrees: float) -> dict[str, object]:
    magnitudes = [event["magnitude"] for event in events]
    times = [event["time"] for event in events]
    gaps = _gaps_hours(times)
    cells = {(int((event["latitude"] - ITALY_LAT[0]) / cell_degrees), int((event["longitude"] - ITALY_LON[0]) / cell_degrees)) for event in events}
    spatial_gaps = _nearest_neighbour_km(events)
    return {
        "event_count": len(events),
        "time_start": times[0] if times else "",
        "time_end": times[-1] if times else "",
        "duration_days": _duration_days(times),
        "rate_per_day": _rate_per_day(events),
        "interevent_hours": _distribution(gaps),
        "magnitude": _distribution(magnitudes),
        "energy_proxy": _distribution([10.0 ** (1.5 * magnitude) for magnitude in magnitudes]),
        "spatial_cell_count": len(cells),
        "spatial_cell_occupancy": sorted(f"{lat}:{lon}" for lat, lon in cells),
        "nearest_neighbour_km": _distribution(spatial_gaps),
    }


def _rate_per_day(events: list[dict[str, float]]) -> float:
    times = [event["time"] for event in events]
    return len(events) / max(_duration_days(times), 1e-9) if events else 0.0


def _compare_summary(real: dict[str, object], synthetic: dict[str, object], name: str) -> dict[str, object]:
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
        "shared_spatial_cell_count": len(set(real["spatial_cell_occupancy"]) & set(synthetic["spatial_cell_occupancy"])),
    }


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

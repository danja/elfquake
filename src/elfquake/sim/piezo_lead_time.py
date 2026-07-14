"""Causal lead-time diagnostics for piezo emissions before avalanche events."""

from __future__ import annotations

import csv
import json
import math
import re
from collections import defaultdict
from pathlib import Path


DEFAULT_LAG_EDGES = (0, 1, 5, 15, 30, 60, 120, 180, 360)
DEFAULT_SIGNAL_FIELDS = (
    "piezo_signal",
    "piezo_total_source",
    "near_critical_cell_count",
    "near_critical_contact_count",
    "near_critical_coherence",
    "near_critical_weighted_stress",
    "critical_cell_count",
    "max_stress_ratio",
    "piezo_charge_total",
    "piezo_charge_max",
    "piezo_release_total",
    "damage_total",
    "damage_max",
    "damage_active_cell_count",
    "damage_local_mean",
    "damage_local_max",
    "damage_local_active_fraction",
    "damage_local_std",
    "mature_weakness_total",
    "mature_weakness_max",
    "mature_weakness_active_cell_count",
)
PROFILE_FIELDS = (
    "signal_field",
    "statistic",
    "lag_start_steps",
    "lag_end_steps",
    "event_sample_count",
    "control_sample_count",
    "event_mean",
    "control_mean",
    "standardized_difference",
    "event_greater_auc",
    "episode_count",
    "positive_episode_count",
    "negative_episode_count",
    "local_baseline_sample_count",
    "local_baseline_mean",
    "local_standardized_difference",
    "local_event_greater_auc",
    "local_episode_count",
    "local_positive_episode_count",
    "local_negative_episode_count",
    "event_change_sample_count",
    "control_change_sample_count",
    "event_change_mean",
    "control_change_mean",
    "change_standardized_difference",
    "change_event_greater_auc",
    "change_episode_count",
    "change_positive_episode_count",
    "change_negative_episode_count",
)


def analyze_piezo_event_lead_time(
    *,
    piezo_paths: list[Path],
    event_paths: list[Path],
    out_path: Path,
    profile_out: Path,
    lag_edges: list[int] | None = None,
    signal_fields: list[str] | None = None,
    primary_field: str = "piezo_signal",
    sensor_mode: str = "mean",
    sensor_top_k: int = 3,
    control_multiplier: int = 10,
    control_exclusion_steps: int | None = None,
) -> dict[str, object]:
    if len(piezo_paths) != len(event_paths) or not piezo_paths:
        raise ValueError("piezo and event paths must be nonempty paired lists")
    edges = tuple(lag_edges or DEFAULT_LAG_EDGES)
    fields = tuple(signal_fields or DEFAULT_SIGNAL_FIELDS)
    _validate_options(edges, fields, control_multiplier)
    if primary_field not in fields:
        raise ValueError("primary_field must be included in signal_fields")
    if sensor_mode not in {"mean", "top_k", "top_k_rise", "event_nearest"}:
        raise ValueError("sensor_mode must be 'mean', 'top_k', 'top_k_rise', or 'event_nearest'")
    if sensor_top_k < 1:
        raise ValueError("sensor_top_k must be at least 1")
    max_lag = edges[-1]
    exclusion = control_exclusion_steps if control_exclusion_steps is not None else max_lag
    if exclusion < 0:
        raise ValueError("control_exclusion_steps must be non-negative")

    samples: dict[tuple[str, str, int, int], dict[str, list[tuple[str, float]]]] = defaultdict(
        lambda: {
            "event": [], "control": [], "local_event": [], "local_baseline": [],
            "event_change": [], "control_change": [],
        }
    )
    episodes = []
    for piezo_path, event_path in zip(piezo_paths, event_paths):
        episode_id = _episode_id(piezo_path, event_path)
        series = _read_step_series(piezo_path, fields)
        sample_count = len(next(iter(series.values())))
        event_rows = _read_event_rows(event_path, sample_count)
        event_steps = [row["step"] for row in event_rows]
        control_steps = _select_control_steps(
            sample_count=len(next(iter(series.values()))),
            event_steps=event_steps,
            max_lag=max_lag,
            exclusion_steps=exclusion,
            control_multiplier=control_multiplier,
        )
        episodes.append({
            "episode_id": episode_id,
            "piezo_csv": str(piezo_path),
            "events_csv": str(event_path),
            "sample_count": len(next(iter(series.values()))),
            "event_count": len(event_steps),
            "event_steps": event_steps,
            "control_count": len(control_steps),
        })
        selected_sensor_ids = []
        if sensor_mode == "mean":
            _collect_episode_samples(
                samples,
                episode_id=episode_id,
                series=series,
                event_steps=event_steps,
                control_steps=control_steps,
                edges=edges,
                local_baseline_offset=max_lag,
            )
        else:
            sensor_series, sensor_points = _read_sensor_step_series(piezo_path, fields)
            if sensor_mode in {"top_k", "top_k_rise"}:
                _collect_episode_samples(
                    samples,
                    episode_id=episode_id,
                    series=_top_k_sensor_series(
                        sensor_series, sensor_top_k, positive_rise=sensor_mode == "top_k_rise",
                    ),
                    event_steps=event_steps,
                    control_steps=control_steps,
                    edges=edges,
                    local_baseline_offset=max_lag,
                )
            else:
                for event in event_rows:
                    sensor_id = _nearest_sensor(sensor_points, event["x"], event["y"])
                    selected_sensor_ids.append(sensor_id)
                    _collect_episode_samples(
                        samples,
                        episode_id=episode_id,
                        series={field: sensor_series[field][sensor_id] for field in fields},
                        event_steps=[event["step"]],
                        control_steps=control_steps,
                        edges=edges,
                        local_baseline_offset=max_lag,
                    )
        episodes[-1]["selected_sensor_ids"] = selected_sensor_ids

    profile = [_profile_row(key, values) for key, values in samples.items()]
    profile.sort(key=lambda row: (
        row["signal_field"], row["statistic"], row["lag_start_steps"], row["lag_end_steps"]
    ))
    recommendation = _recommendation(profile, primary_field=primary_field)
    report = {
        "schema": "elfquake.piezo_event_lead_time.v1",
        "status": "analyzed",
        "causal_semantics": "lag 0 is sampled before same-step relaxation; positive lags are earlier steps",
        "lag_edges_steps": list(edges),
        "signal_fields": list(fields),
        "statistics": ["mean", "max"],
        "sensor_mode": sensor_mode,
        "sensor_top_k": sensor_top_k if sensor_mode in {"top_k", "top_k_rise"} else None,
        "control_multiplier": control_multiplier,
        "control_exclusion_steps": exclusion,
        "local_baseline_offset_steps": max_lag,
        "episode_count": len(episodes),
        "event_count": sum(item["event_count"] for item in episodes),
        "control_count": sum(item["control_count"] for item in episodes),
        "episodes": episodes,
        "recommendation": recommendation,
        "profile_csv": str(profile_out),
    }
    _write_profile(profile_out, profile)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def _validate_options(edges: tuple[int, ...], fields: tuple[str, ...], control_multiplier: int) -> None:
    if len(edges) < 2 or edges[0] < 0 or any(right <= left for left, right in zip(edges, edges[1:])):
        raise ValueError("lag edges must be non-negative and strictly increasing")
    if not fields or len(set(fields)) != len(fields):
        raise ValueError("signal fields must be nonempty and unique")
    if control_multiplier < 1:
        raise ValueError("control_multiplier must be at least 1")


def _read_step_series(path: Path, fields: tuple[str, ...]) -> dict[str, list[float]]:
    by_step: dict[int, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        missing = [field for field in fields if field not in (reader.fieldnames or [])]
        if missing:
            raise ValueError(f"missing piezo fields in {path}: {', '.join(missing)}")
        for row in reader:
            step = int(row["step"])
            for field in fields:
                by_step[step][field].append(float(row[field]))
    if not by_step:
        raise ValueError(f"empty piezo CSV: {path}")
    sample_count = max(by_step) + 1
    result = {field: [0.0] * sample_count for field in fields}
    for step, values in by_step.items():
        for field in fields:
            result[field][step] = sum(values[field]) / len(values[field])
    return result


def _read_event_rows(path: Path, sample_count: int) -> list[dict[str, float | int]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    result = []
    seen = set()
    for row in rows:
        step = int(row["step"])
        if step in seen or not 0 <= step < sample_count:
            continue
        seen.add(step)
        result.append({
            "step": step,
            "x": float(row.get("x", "nan") or "nan"),
            "y": float(row.get("y", "nan") or "nan"),
        })
    return sorted(result, key=lambda row: row["step"])


def _read_sensor_step_series(
    path: Path,
    fields: tuple[str, ...],
) -> tuple[dict[str, dict[int, list[float]]], dict[int, tuple[float, float]]]:
    rows = []
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required = {"step", "sensor_id", "x", "y", *fields}
        missing = sorted(required - set(reader.fieldnames or []))
        if missing:
            raise ValueError(f"missing sensor fields in {path}: {', '.join(missing)}")
        rows = list(reader)
    sample_count = max(int(row["step"]) for row in rows) + 1
    sensor_ids = sorted({int(row["sensor_id"]) for row in rows})
    series = {field: {sensor_id: [0.0] * sample_count for sensor_id in sensor_ids} for field in fields}
    points = {}
    for row in rows:
        step = int(row["step"])
        sensor_id = int(row["sensor_id"])
        points[sensor_id] = (float(row["x"]), float(row["y"]))
        for field in fields:
            series[field][sensor_id][step] = float(row[field])
    return series, points


def _nearest_sensor(points: dict[int, tuple[float, float]], x: float, y: float) -> int:
    if not points or not math.isfinite(x) or not math.isfinite(y):
        raise ValueError("event-nearest sensor mode requires event x/y coordinates")
    return min(
        points,
        key=lambda sensor_id: (
            (points[sensor_id][0] - x) ** 2 + (points[sensor_id][1] - y) ** 2,
            sensor_id,
        ),
    )


def _top_k_sensor_series(
    sensor_series: dict[str, dict[int, list[float]]], top_k: int, *, positive_rise: bool = False,
) -> dict[str, list[float]]:
    """Causally pool current readings, or their positive one-step rise, at each step."""
    result = {}
    for field, by_sensor in sensor_series.items():
        rows = list(by_sensor.values())
        if not rows:
            result[field] = []
            continue
        selected = min(top_k, len(rows))
        pooled = []
        for step in range(len(rows[0])):
            candidates = []
            for values in rows:
                value = values[step]
                if positive_rise:
                    value = max(0.0, value - values[step - 1]) if step else 0.0
                candidates.append(value)
            pooled.append(sum(sorted(candidates, reverse=True)[:selected]) / selected)
        result[field] = pooled
    return result


def _select_control_steps(
    *, sample_count: int, event_steps: list[int], max_lag: int,
    exclusion_steps: int, control_multiplier: int,
) -> list[int]:
    eligible = [
        step
        for step in range(max_lag - 1, sample_count)
        if all(abs(step - event_step) >= exclusion_steps for event_step in event_steps)
    ]
    desired = min(len(eligible), max(1, len(event_steps) * control_multiplier))
    if desired == 0:
        return []
    if desired == 1:
        return [eligible[len(eligible) // 2]]
    indices = [round(index * (len(eligible) - 1) / (desired - 1)) for index in range(desired)]
    return [eligible[index] for index in sorted(set(indices))]


def _collect_episode_samples(
    samples, *, episode_id: str, series: dict[str, list[float]],
    event_steps: list[int], control_steps: list[int], edges: tuple[int, ...],
    local_baseline_offset: int,
) -> None:
    for field, values in series.items():
        for start, end in zip(edges, edges[1:]):
            for statistic in ("mean", "max"):
                key = (field, statistic, start, end)
                for kind, endpoints in (("event", event_steps), ("control", control_steps)):
                    for endpoint in endpoints:
                        value = _lag_window(values, endpoint, start, end, statistic)
                        if value is not None:
                            samples[key][kind].append((episode_id, value))
                for endpoint in event_steps:
                    event_value = _lag_window(values, endpoint, start, end, statistic)
                    baseline_value = _lag_window(
                        values,
                        endpoint,
                        start + local_baseline_offset,
                        end + local_baseline_offset,
                        statistic,
                    )
                    if event_value is not None and baseline_value is not None:
                        samples[key]["local_event"].append((episode_id, event_value))
                        samples[key]["local_baseline"].append((episode_id, baseline_value))
                        samples[key]["event_change"].append((episode_id, event_value - baseline_value))
                for endpoint in control_steps:
                    control_value = _lag_window(values, endpoint, start, end, statistic)
                    baseline_value = _lag_window(
                        values,
                        endpoint,
                        start + local_baseline_offset,
                        end + local_baseline_offset,
                        statistic,
                    )
                    if control_value is not None and baseline_value is not None:
                        samples[key]["control_change"].append((episode_id, control_value - baseline_value))


def _lag_window(values: list[float], endpoint: int, start: int, end: int, statistic: str) -> float | None:
    indices = [endpoint - lag for lag in range(start, end)]
    if not indices or min(indices) < 0:
        return None
    selected = [values[index] for index in indices]
    return sum(selected) / len(selected) if statistic == "mean" else max(selected)


def _profile_row(key, values) -> dict[str, object]:
    field, statistic, start, end = key
    events = [value for _, value in values["event"]]
    controls = [value for _, value in values["control"]]
    episode_effects = []
    episode_ids = sorted({episode for episode, _ in values["event"]})
    for episode_id in episode_ids:
        episode_events = [value for episode, value in values["event"] if episode == episode_id]
        episode_controls = [value for episode, value in values["control"] if episode == episode_id]
        if episode_events and episode_controls:
            episode_effects.append(_standardized_difference(episode_events, episode_controls))
    local_events = [value for _, value in values["local_event"]]
    local_baselines = [value for _, value in values["local_baseline"]]
    local_effects = []
    local_episode_ids = sorted({episode for episode, _ in values["local_event"]})
    for episode_id in local_episode_ids:
        episode_events = [value for episode, value in values["local_event"] if episode == episode_id]
        episode_baselines = [value for episode, value in values["local_baseline"] if episode == episode_id]
        if episode_events and episode_baselines:
            local_effects.append(_standardized_difference(episode_events, episode_baselines))
    event_changes = [value for _, value in values["event_change"]]
    control_changes = [value for _, value in values["control_change"]]
    change_effects = []
    change_episode_ids = sorted({episode for episode, _ in values["event_change"]})
    for episode_id in change_episode_ids:
        episode_events = [value for episode, value in values["event_change"] if episode == episode_id]
        episode_controls = [value for episode, value in values["control_change"] if episode == episode_id]
        if episode_events and episode_controls:
            change_effects.append(_standardized_difference(episode_events, episode_controls))
    return {
        "signal_field": field,
        "statistic": statistic,
        "lag_start_steps": start,
        "lag_end_steps": end,
        "event_sample_count": len(events),
        "control_sample_count": len(controls),
        "event_mean": _rounded_mean(events),
        "control_mean": _rounded_mean(controls),
        "standardized_difference": _standardized_difference(events, controls),
        "event_greater_auc": _auc(events, controls),
        "episode_count": len(episode_effects),
        "positive_episode_count": sum(value > 0 for value in episode_effects),
        "negative_episode_count": sum(value < 0 for value in episode_effects),
        "local_baseline_sample_count": len(local_baselines),
        "local_baseline_mean": _rounded_mean(local_baselines),
        "local_standardized_difference": _standardized_difference(local_events, local_baselines),
        "local_event_greater_auc": _auc(local_events, local_baselines),
        "local_episode_count": len(local_effects),
        "local_positive_episode_count": sum(value > 0 for value in local_effects),
        "local_negative_episode_count": sum(value < 0 for value in local_effects),
        "event_change_sample_count": len(event_changes),
        "control_change_sample_count": len(control_changes),
        "event_change_mean": _rounded_mean(event_changes),
        "control_change_mean": _rounded_mean(control_changes),
        "change_standardized_difference": _standardized_difference(event_changes, control_changes),
        "change_event_greater_auc": _auc(event_changes, control_changes),
        "change_episode_count": len(change_effects),
        "change_positive_episode_count": sum(value > 0 for value in change_effects),
        "change_negative_episode_count": sum(value < 0 for value in change_effects),
    }


def _standardized_difference(left: list[float], right: list[float]) -> float:
    if not left or not right:
        return 0.0
    combined = left + right
    mean = sum(combined) / len(combined)
    std = math.sqrt(sum((value - mean) ** 2 for value in combined) / len(combined))
    return round(((sum(left) / len(left)) - (sum(right) / len(right))) / std, 8) if std > 1e-12 else 0.0


def _auc(events: list[float], controls: list[float]) -> float:
    if not events or not controls:
        return 0.5
    score = sum(1.0 if event > control else 0.5 if event == control else 0.0 for event in events for control in controls)
    return round(score / (len(events) * len(controls)), 8)


def _rounded_mean(values: list[float]) -> float:
    return round(sum(values) / len(values), 8) if values else 0.0


def _recommendation(profile: list[dict[str, object]], *, primary_field: str) -> dict[str, object]:
    candidates = [
        row for row in profile
        if row["signal_field"] == primary_field and row["statistic"] == "mean" and row["lag_start_steps"] >= 1
    ]
    supported = [
        row for row in candidates
        if row["change_standardized_difference"] > 0
        and row["change_event_greater_auc"] >= 0.55
        and row["change_positive_episode_count"] >= math.ceil(row["change_episode_count"] * 2 / 3)
        and row["change_episode_count"] > 0
    ]
    ranked = sorted(
        candidates,
        key=lambda row: (
            row["change_positive_episode_count"] / max(1, row["change_episode_count"]),
            row["change_event_greater_auc"],
            row["change_standardized_difference"],
        ),
        reverse=True,
    )
    return {
        "primary_field": primary_field,
        "support_rule": "event-local change beats control-local change, AUC >= 0.55, positive in at least two thirds of episodes",
        "supported_lag_bins": [
            [row["lag_start_steps"], row["lag_end_steps"]] for row in supported
        ],
        "recommended_context_steps": max((row["lag_end_steps"] for row in supported), default=0),
        "strongest_consistent_bin": (
            [ranked[0]["lag_start_steps"], ranked[0]["lag_end_steps"]] if ranked else []
        ),
    }


def _episode_id(piezo_path: Path, event_path: Path) -> str:
    match = re.search(r"seed\d+", f"{piezo_path} {event_path}")
    return match.group(0) if match else piezo_path.stem


def _write_profile(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=PROFILE_FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)

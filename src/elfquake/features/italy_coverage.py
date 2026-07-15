"""Coverage and label-free VLF/seismic overlap diagnostics for Italy."""

from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path

from elfquake.features.common import format_utc, parse_utc


def build_italy_coverage_report(
    *, events_csv: Path, vlf_metadata_root: Path | None = None,
    anomaly_scores_csv: Path | None = None, out_path: Path, weekly_out: Path | None = None,
    magnitude_thresholds: tuple[float, ...] = (2.0, 2.5, 3.0),
) -> dict[str, object]:
    events = _read_csv(events_csv)
    captures = _read_metadata(vlf_metadata_root) if vlf_metadata_root else []
    scores = _read_csv(anomaly_scores_csv) if anomaly_scores_csv else []
    weekly = _weekly_rows(events, captures, scores, magnitude_thresholds)
    report = {
        "schema": "elfquake.italy_coverage_report.v1",
        "region_id": "all_italy",
        "events": _event_summary(events, magnitude_thresholds),
        "vlf_captures": _capture_summary(captures),
        "vlf_anomaly_scores": _score_summary(scores),
        "overlap": {
            "week_count": len(weekly),
            "weeks_with_events": sum(1 for row in weekly if row["event_count"] > 0),
            "weeks_with_vlf": sum(1 for row in weekly if row["vlf_observation_count"] > 0),
            "weeks_with_both": sum(1 for row in weekly if row["event_count"] > 0 and row["vlf_observation_count"] > 0),
            "note": "Descriptive overlap only; no supervised label or causal association is inferred.",
        },
        "inputs": {"events_csv": str(events_csv), "vlf_metadata_root": str(vlf_metadata_root or ""),
                   "anomaly_scores_csv": str(anomaly_scores_csv or "")},
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if weekly_out:
        _write_weekly(weekly_out, weekly, magnitude_thresholds)
    return report


def _weekly_rows(events: list[dict[str, str]], captures: list[dict[str, str]], scores: list[dict[str, str]], thresholds: tuple[float, ...]) -> list[dict[str, object]]:
    buckets: dict[str, dict[str, object]] = {}
    for row in events:
        bucket = _week_start(row.get("event_time_utc", ""))
        if not bucket:
            continue
        item = buckets.setdefault(bucket, _empty_week(bucket, thresholds))
        item["event_count"] += 1
        magnitude = float(row.get("magnitude") or "-inf")
        item["max_magnitude"] = magnitude if item["max_magnitude"] is None else max(float(item["max_magnitude"]), magnitude)
        for threshold in thresholds:
            if float(row.get("magnitude") or "-inf") >= threshold:
                item[f"event_count_ge_{_slug(threshold)}"] += 1
    for row in captures:
        bucket = _week_start(row.get("captured_at_utc", ""))
        if bucket:
            buckets.setdefault(bucket, _empty_week(bucket, thresholds))["vlf_observation_count"] += 1
    for row in scores:
        bucket = _week_start(row.get("window_end_utc", ""))
        if not bucket:
            continue
        item = buckets.setdefault(bucket, _empty_week(bucket, thresholds))
        value = float(row.get("anomaly_score") or 0.0)
        item["vlf_score_count"] += 1
        item["vlf_score_sum"] += value
        item["vlf_score_max"] = value if item["vlf_score_max"] is None else max(float(item["vlf_score_max"]), value)
        if value >= 0.8:
            item["vlf_alert_count"] += 1
    rows = []
    for key in sorted(buckets):
        item = buckets[key]
        count = int(item["vlf_score_count"])
        item["vlf_score_mean"] = round(float(item["vlf_score_sum"]) / count, 6) if count else None
        item.pop("vlf_score_sum")
        rows.append(item)
    return rows


def _empty_week(bucket: str, thresholds: tuple[float, ...]) -> dict[str, object]:
    return {"week_start_utc": bucket, "event_count": 0, "max_magnitude": None,
            "vlf_observation_count": 0, "vlf_score_count": 0, "vlf_score_sum": 0.0,
            "vlf_score_max": None, "vlf_score_mean": None, "vlf_alert_count": 0,
            **{f"event_count_ge_{_slug(t)}": 0 for t in thresholds}}


def _event_summary(rows: list[dict[str, str]], thresholds: tuple[float, ...]) -> dict[str, object]:
    times = [row.get("event_time_utc", "") for row in rows if row.get("event_time_utc")]
    return {"row_count": len(rows), "first_event_utc": min(times) if times else "", "last_event_utc": max(times) if times else "",
            "counts_by_minimum_magnitude": {str(t): sum(1 for row in rows if float(row.get("magnitude") or "-inf") >= t) for t in thresholds}}


def _capture_summary(rows: list[dict[str, str]]) -> dict[str, object]:
    times = sorted(row.get("captured_at_utc", "") for row in rows if row.get("captured_at_utc"))
    unique_days = len({value[:10] for value in times})
    return {"metadata_count": len(rows), "unique_days": unique_days, "first_capture_utc": times[0] if times else "", "last_capture_utc": times[-1] if times else "", "source_ids": sorted({row.get("source_id", "") for row in rows if row.get("source_id")})}


def _score_summary(rows: list[dict[str, str]]) -> dict[str, object]:
    values = [float(row["anomaly_score"]) for row in rows if row.get("anomaly_score")]
    times = [row.get("window_end_utc", "") for row in rows if row.get("window_end_utc")]
    return {"row_count": len(rows), "first_window_end_utc": min(times) if times else "", "last_window_end_utc": max(times) if times else "", "mean_anomaly_score": round(sum(values) / len(values), 6) if values else None, "max_anomaly_score": max(values) if values else None, "alert_count_ge_0.8": sum(1 for value in values if value >= 0.8)}


def _read_csv(path: Path | None) -> list[dict[str, str]]:
    if not path or not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _read_metadata(root: Path) -> list[dict[str, str]]:
    if not root or not root.exists():
        return []
    rows = []
    for path in sorted(root.rglob("*.metadata.json")):
        try:
            rows.append(json.loads(path.read_text(encoding="utf-8")))
        except (OSError, json.JSONDecodeError):
            continue
    return rows


def _week_start(value: str) -> str:
    if not value:
        return ""
    try:
        parsed = parse_utc(value)
    except ValueError:
        return ""
    monday = parsed - timedelta(days=parsed.weekday(), hours=parsed.hour, minutes=parsed.minute, seconds=parsed.second, microseconds=parsed.microsecond)
    return format_utc(monday)


def _slug(value: float) -> str:
    return str(value).replace(".", "_")


def _write_weekly(path: Path, rows: list[dict[str, object]], thresholds: tuple[float, ...]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(_empty_week("", thresholds).keys())
    fields.remove("vlf_score_sum")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)

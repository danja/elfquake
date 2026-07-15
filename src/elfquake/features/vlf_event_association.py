"""Permutation-controlled descriptive association between VLF novelty and INGV events."""

from __future__ import annotations

import csv
import json
import random
from pathlib import Path

from elfquake.features.italy_coverage import _read_csv, _slug, _weekly_rows


def build_vlf_event_association_report(
    *, events_csv: Path, anomaly_scores_csv: Path, out_path: Path,
    weekly_out: Path | None = None, permutations: int = 2000, seed: int = 42,
    magnitude_thresholds: tuple[float, ...] = (2.0, 2.5, 3.0),
) -> dict[str, object]:
    events = _read_csv(events_csv)
    scores = _read_csv(anomaly_scores_csv)
    weekly = _weekly_rows(events, [], scores, magnitude_thresholds)
    results = {}
    for threshold in magnitude_thresholds:
        field = f"event_count_ge_{_slug(threshold)}"
        rows = [row for row in weekly if int(row["vlf_score_count"]) > 0]
        observed = _difference_in_means(rows, field)
        results[str(threshold)] = _permutation_result(rows, field, observed, permutations, seed)
    report = {
        "schema": "elfquake.vlf_event_association_report.v1",
        "status": "evaluated" if any(item["status"] == "evaluated" for item in results.values()) else "insufficient_controls",
        "method": "weekly VLF anomaly summaries compared with INGV event weeks; event labels are permuted while preserving their count",
        "warning": "Descriptive association only. This is not a supervised prediction test and cannot establish a precursor relationship.",
        "permutations": permutations, "seed": seed,
        "inputs": {"events_csv": str(events_csv), "anomaly_scores_csv": str(anomaly_scores_csv)},
        "weekly_vlf_rows": len([row for row in weekly if int(row["vlf_score_count"]) > 0]),
        "results_by_minimum_magnitude": results,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if weekly_out:
        _write_weekly(weekly_out, weekly)
    return report


def _difference_in_means(rows: list[dict[str, object]], event_field: str, flags: list[bool] | None = None) -> float | None:
    if flags is None:
        flags = [int(row[event_field]) > 0 for row in rows]
    positive = [float(row["vlf_score_mean"]) for row, flag in zip(rows, flags) if flag and row["vlf_score_mean"] is not None]
    negative = [float(row["vlf_score_mean"]) for row, flag in zip(rows, flags) if not flag and row["vlf_score_mean"] is not None]
    if not positive or not negative:
        return None
    return sum(positive) / len(positive) - sum(negative) / len(negative)


def _permutation_result(rows: list[dict[str, object]], event_field: str, observed: float | None, permutations: int, seed: int) -> dict[str, object]:
    event_count = sum(1 for row in rows if int(row[event_field]) > 0)
    control_count = len(rows) - event_count
    result: dict[str, object] = {"status": "insufficient_controls", "weekly_rows": len(rows), "event_weeks": event_count, "control_weeks": control_count, "observed_difference_mean_anomaly": observed}
    if event_count < 3 or control_count < 3 or observed is None:
        result["note"] = "Need at least three VLF-observed event weeks and three VLF-observed control weeks."
        return result
    rng = random.Random(seed)
    flags = [True] * event_count + [False] * control_count
    null_values = []
    for _ in range(permutations):
        rng.shuffle(flags)
        value = _difference_in_means(rows, event_field, flags)
        if value is not None:
            null_values.append(value)
    extreme = sum(1 for value in null_values if abs(value) >= abs(observed))
    result.update({"status": "evaluated", "null_mean": sum(null_values) / len(null_values), "null_sd": _sd(null_values), "permutation_p_two_sided": (extreme + 1) / (len(null_values) + 1), "null_count": len(null_values)})
    return result


def _sd(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    mean = sum(values) / len(values)
    return (sum((value - mean) ** 2 for value in values) / (len(values) - 1)) ** 0.5


def _write_weekly(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["week_start_utc", "vlf_score_count", "vlf_score_mean", "vlf_score_max", "vlf_alert_count"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows([{field: row.get(field, "") for field in fields} for row in rows if int(row["vlf_score_count"]) > 0])

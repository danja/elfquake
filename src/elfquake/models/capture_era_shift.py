"""Capture-era segmentation and feature-distribution shift diagnostics.

A collector outage splits an otherwise regular capture record into eras with
different cadence. Chronological train/test splits can land on an era boundary,
which makes any resulting score a statement about the outage rather than about
the signal. This module segments the record by capture gap, compares feature
distributions between eras, and separates cadence-derived aggregation artifacts
from intensity/band/streak features that could carry physical signal.
"""

from __future__ import annotations

import csv
import json
import math
import statistics
from datetime import datetime
from pathlib import Path

from scipy.stats import ks_2samp, wasserstein_distance

from elfquake.models.feature_groups import (
    ID_FIELDS,
    TARGET_FIELDS,
    VLF_IMAGE_PREFIXES,
    VLF_METADATA_PREFIXES,
)


VLF_PREFIXES = VLF_METADATA_PREFIXES + VLF_IMAGE_PREFIXES + ("real_vlf_",)

# Fields whose value is a function of how many captures landed in the window,
# not of what the captures contain. These shift mechanically with cadence.
CADENCE_DERIVED_FIELDS = frozenset(
    {
        "vlf_capture_count",
        "vlf_jpeg_count",
        "vlf_latest_age_seconds",
        "vlf_total_bytes",
        "vlf_image_feature_count",
        "real_vlf_image_sample_count",
    }
)
CADENCE_DERIVED_SUFFIXES = ("_count", "_sum")

QUALITY_PREFIX = "quality_"
SEISMIC_PREFIX = "seismic_"
ASTRO_PREFIX = "astro_"

ISO_FORMAT = "%Y-%m-%dT%H:%M:%SZ"


def diagnose_capture_era_shift(
    *,
    input_csv: Path,
    out_path: Path,
    time_field: str = "window_start_utc",
    era_gap_hours: float = 48.0,
    min_era_anchors: int = 5,
    csv_out_path: Path | None = None,
    era_csv_dir: Path | None = None,
) -> dict[str, object]:
    if era_gap_hours <= 0:
        raise ValueError("era_gap_hours must be positive")

    rows, fieldnames = _read_rows_and_fields(input_csv)
    if time_field not in fieldnames:
        raise ValueError(f"missing time field: {time_field}")

    anchors = sorted({row.get(time_field, "") for row in rows if row.get(time_field, "")})
    report: dict[str, object] = {
        "schema": "elfquake.capture_era_shift.v1",
        "input": str(input_csv),
        "time_field": time_field,
        "era_gap_hours": era_gap_hours,
        "min_era_anchors": min_era_anchors,
        "row_count": len(rows),
        "anchor_count": len(anchors),
    }
    if len(anchors) < 2:
        report["status"] = "insufficient_anchors"
        return _write_report(out_path, report)

    boundaries = _boundary_gaps(anchors, era_gap_hours=era_gap_hours)
    segments = _segment_anchors(anchors, era_gap_hours=era_gap_hours)
    rows_by_anchor: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        rows_by_anchor.setdefault(row.get(time_field, ""), []).append(row)

    eras = [
        _describe_era(index=index, anchors=segment, rows_by_anchor=rows_by_anchor, min_era_anchors=min_era_anchors)
        for index, segment in enumerate(segments)
    ]
    report["boundary_gaps"] = boundaries
    report["eras"] = [{key: value for key, value in era.items() if key != "rows"} for era in eras]

    major = [era for era in eras if era["kind"] == "era"]
    major.sort(key=lambda era: era["labeled_row_count"], reverse=True)
    if len(major) < 2:
        report["status"] = "insufficient_eras"
        _write_era_csvs(era_csv_dir, eras, fieldnames)
        return _write_report(out_path, report)

    first, second = sorted(major[:2], key=lambda era: era["start_utc"])
    report["compared_eras"] = {"earlier": first["era_id"], "later": second["era_id"]}
    features = _numeric_feature_names(rows, fieldnames)
    comparisons = [
        _compare_feature(name=name, earlier=first["rows"], later=second["rows"])
        for name in features
    ]
    comparisons = [entry for entry in comparisons if entry is not None]
    comparisons.sort(key=_shift_magnitude, reverse=True)
    report["feature_comparisons"] = comparisons
    report["family_summary"] = _family_summary(comparisons)
    report["target_shift"] = _target_shift(earlier=first, later=second)
    report["status"] = "evaluated"

    _write_era_csvs(era_csv_dir, eras, fieldnames)
    if csv_out_path is not None:
        _write_comparison_csv(csv_out_path, comparisons)
        report["csv_output"] = str(csv_out_path)
    return _write_report(out_path, report)


def _boundary_gaps(anchors: list[str], *, era_gap_hours: float) -> list[dict[str, object]]:
    gaps = []
    for previous, current in zip(anchors, anchors[1:]):
        hours = (_parse_time(current) - _parse_time(previous)).total_seconds() / 3600.0
        if hours > era_gap_hours:
            gaps.append({"before_utc": previous, "after_utc": current, "gap_hours": round(hours, 3)})
    return gaps


def _segment_anchors(anchors: list[str], *, era_gap_hours: float) -> list[list[str]]:
    segments: list[list[str]] = [[anchors[0]]]
    for previous, current in zip(anchors, anchors[1:]):
        hours = (_parse_time(current) - _parse_time(previous)).total_seconds() / 3600.0
        if hours > era_gap_hours:
            segments.append([current])
        else:
            segments[-1].append(current)
    return segments


def _describe_era(
    *,
    index: int,
    anchors: list[str],
    rows_by_anchor: dict[str, list[dict[str, str]]],
    min_era_anchors: int,
) -> dict[str, object]:
    era_rows = [row for anchor in anchors for row in rows_by_anchor.get(anchor, [])]
    labeled = [row for row in era_rows if row.get("target_occurred") in {"0", "1"}]
    positives = sum(int(row["target_occurred"]) for row in labeled)
    steps = [
        (_parse_time(current) - _parse_time(previous)).total_seconds()
        for previous, current in zip(anchors, anchors[1:])
    ]
    span_hours = (_parse_time(anchors[-1]) - _parse_time(anchors[0])).total_seconds() / 3600.0
    return {
        "era_id": f"era_{index}",
        "kind": "era" if len(anchors) >= min_era_anchors else "isolated",
        "start_utc": anchors[0],
        "end_utc": anchors[-1],
        "span_hours": round(span_hours, 3),
        "anchor_count": len(anchors),
        "row_count": len(era_rows),
        "labeled_row_count": len(labeled),
        "positive_count": positives,
        "positive_rate": round(positives / len(labeled), 6) if labeled else None,
        "median_step_seconds": round(statistics.median(steps), 3) if steps else None,
        "anchors_per_day": round(len(anchors) / (span_hours / 24.0), 3) if span_hours > 0 else None,
        "rows": era_rows,
    }


def _numeric_feature_names(rows: list[dict[str, str]], fieldnames: list[str]) -> list[str]:
    names = []
    for field in fieldnames:
        if field in TARGET_FIELDS or field in ID_FIELDS:
            continue
        values = [row.get(field, "") for row in rows]
        if values and all(_is_float(value) for value in values):
            names.append(field)
    return names


def classify_feature_family(name: str) -> str:
    """Group a feature by what a cadence change would do to it."""
    if name.startswith(QUALITY_PREFIX):
        return "quality"
    if name.startswith(VLF_PREFIXES):
        if name in CADENCE_DERIVED_FIELDS or name.endswith(CADENCE_DERIVED_SUFFIXES):
            return "vlf_cadence_derived"
        return "vlf_signal"
    if name.startswith(SEISMIC_PREFIX):
        return "seismic"
    if name.startswith(ASTRO_PREFIX):
        return "astronomy"
    return "other"


def _compare_feature(*, name: str, earlier: list[dict[str, str]], later: list[dict[str, str]]) -> dict[str, object] | None:
    left = [float(row[name]) for row in earlier if _is_float(row.get(name, ""))]
    right = [float(row[name]) for row in later if _is_float(row.get(name, ""))]
    if len(left) < 2 or len(right) < 2:
        return None
    left_mean = statistics.fmean(left)
    right_mean = statistics.fmean(right)
    left_sd = statistics.pstdev(left)
    right_sd = statistics.pstdev(right)
    pooled = math.sqrt((left_sd**2 + right_sd**2) / 2.0)
    difference = right_mean - left_mean
    # Constant within each era but different between them is the largest shift
    # available, not the smallest; a pooled-sd ratio cannot express it.
    fully_separated = pooled == 0.0 and difference != 0.0
    ks = ks_2samp(left, right)
    return {
        "feature": name,
        "family": classify_feature_family(name),
        "fully_separated": fully_separated,
        "earlier_count": len(left),
        "later_count": len(right),
        "earlier_mean": round(left_mean, 8),
        "later_mean": round(right_mean, 8),
        "earlier_sd": round(left_sd, 8),
        "later_sd": round(right_sd, 8),
        "mean_difference": round(difference, 8),
        "standardized_mean_difference": None if fully_separated else (round(difference / pooled, 6) if pooled else 0.0),
        "ks_statistic": round(float(ks.statistic), 6),
        "ks_p_value": round(float(ks.pvalue), 8),
        "wasserstein_distance": round(float(wasserstein_distance(left, right)), 8),
        # Wasserstein in units of the pooled spread, so features with different
        # natural scales can be ranked against each other.
        "standardized_wasserstein_distance": (
            None if fully_separated else round(wasserstein_distance(left, right) / pooled, 6) if pooled else 0.0
        ),
        "constant_in_both": left_sd == 0.0 and right_sd == 0.0,
    }


def _shift_magnitude(entry: dict[str, object]) -> float:
    """Ranking key that puts fully separated features above any finite ratio."""
    value = entry["standardized_mean_difference"]
    return math.inf if value is None else abs(float(value))


def _family_summary(comparisons: list[dict[str, object]]) -> dict[str, object]:
    families: dict[str, list[dict[str, object]]] = {}
    for entry in comparisons:
        families.setdefault(str(entry["family"]), []).append(entry)
    summary = {}
    for family, entries in sorted(families.items()):
        magnitudes = [_shift_magnitude(entry) for entry in entries]
        finite = [value for value in magnitudes if math.isfinite(value)]
        ks_values = [float(entry["ks_statistic"]) for entry in entries]
        summary[family] = {
            "feature_count": len(entries),
            "constant_feature_count": sum(1 for entry in entries if entry["constant_in_both"]),
            "fully_separated_feature_count": sum(1 for entry in entries if entry["fully_separated"]),
            "median_abs_standardized_mean_difference": round(statistics.median(finite), 6) if finite else None,
            "max_abs_standardized_mean_difference": round(max(finite), 6) if finite else None,
            "median_ks_statistic": round(statistics.median(ks_values), 6),
            "max_ks_statistic": round(max(ks_values), 6),
            "top_features": [str(entry["feature"]) for entry in entries[:5]],
        }
    return summary


def _target_shift(*, earlier: dict[str, object], later: dict[str, object]) -> dict[str, object]:
    earlier_rate = earlier["positive_rate"]
    later_rate = later["positive_rate"]
    return {
        "earlier_era": earlier["era_id"],
        "later_era": later["era_id"],
        "earlier_labeled_row_count": earlier["labeled_row_count"],
        "later_labeled_row_count": later["labeled_row_count"],
        "earlier_positive_rate": earlier_rate,
        "later_positive_rate": later_rate,
        "positive_rate_difference": (
            round(float(later_rate) - float(earlier_rate), 6)
            if earlier_rate is not None and later_rate is not None
            else None
        ),
    }


def _write_era_csvs(era_csv_dir: Path | None, eras: list[dict[str, object]], fieldnames: list[str]) -> None:
    if era_csv_dir is None:
        return
    era_csv_dir.mkdir(parents=True, exist_ok=True)
    for era in eras:
        path = era_csv_dir / f"{era['era_id']}.csv"
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
            writer.writeheader()
            writer.writerows(era["rows"])


def _write_comparison_csv(path: Path, comparisons: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "feature",
        "family",
        "earlier_count",
        "later_count",
        "earlier_mean",
        "later_mean",
        "earlier_sd",
        "later_sd",
        "mean_difference",
        "standardized_mean_difference",
        "fully_separated",
        "ks_statistic",
        "ks_p_value",
        "wasserstein_distance",
        "standardized_wasserstein_distance",
        "constant_in_both",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for entry in comparisons:
            writer.writerow({field: entry[field] for field in fields})


def _write_report(path: Path, report: dict[str, object]) -> dict[str, object]:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def _read_rows_and_fields(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        return list(reader), list(reader.fieldnames or [])


def _parse_time(value: str) -> datetime:
    return datetime.strptime(value, ISO_FORMAT)


def _is_float(value: str) -> bool:
    try:
        float(value)
        return value != ""
    except ValueError:
        return False

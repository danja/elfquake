"""Capture-era segmentation and the near-constant standardization guard.

Regression cover for the 2026-08-10 diagnostic of item 92, where the Cumiana
record turned out to hold two dense eras separated by a 22-day collector
outage, and where a feature that is constant within one era was found to be
standardized by a floating-point residue rather than collapsed to a constant.
"""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path

from elfquake.models.capture_era_shift import classify_feature_family, diagnose_capture_era_shift
from elfquake.models.scaling import RELATIVE_SCALE_TOLERANCE, resolve_scale
from elfquake.models.temporal_holdout import _standardize_train_test


FIELDNAMES = [
    "window_id",
    "region_id",
    "target_cell_id",
    "window_start_utc",
    "target_occurred",
    "target_status",
    "seismic_event_count",
    "vlf_capture_count",
    "vlf_image_intensity_mean_avg",
]


def _anchor_times(day: int, count: int, *, month: int = 7) -> list[str]:
    return [f"2026-{month:02d}-{day:02d}T{hour:02d}:00:00Z" for hour in range(count)]


def _write_table(path: Path, anchors: list[tuple[str, float]]) -> Path:
    """One row per cell per anchor, with intensity carried from the anchor."""
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES, lineterminator="\n")
        writer.writeheader()
        for index, (time, intensity) in enumerate(anchors):
            for cell in range(2):
                writer.writerow(
                    {
                        "window_id": f"w{index}_{cell}",
                        "region_id": "all_italy",
                        "target_cell_id": f"cell_{cell}",
                        "window_start_utc": time,
                        "target_occurred": str((index + cell) % 2),
                        "target_status": "labeled",
                        "seismic_event_count": "3",
                        "vlf_capture_count": "10",
                        "vlf_image_intensity_mean_avg": f"{intensity}",
                    }
                )
    return path


def _two_era_table(path: Path) -> Path:
    early = [(time, 0.60) for time in _anchor_times(1, 8)]
    late = [(time, 0.40) for time in _anchor_times(28, 8)]
    return _write_table(path, early + late)


def test_outage_splits_the_record_into_two_eras(tmp_path: Path) -> None:
    report = diagnose_capture_era_shift(
        input_csv=_two_era_table(tmp_path / "windows.csv"),
        out_path=tmp_path / "report.json",
    )
    assert report["status"] == "evaluated"
    eras = [era for era in report["eras"] if era["kind"] == "era"]
    assert [era["era_id"] for era in eras] == ["era_0", "era_1"]
    assert eras[0]["start_utc"] == "2026-07-01T00:00:00Z"
    assert eras[1]["start_utc"] == "2026-07-28T00:00:00Z"
    assert report["boundary_gaps"][0]["gap_hours"] > 24 * 26


def test_ordinary_overnight_stops_do_not_open_a_new_era(tmp_path: Path) -> None:
    """The collector idles overnight; a 15-hour gap is cadence, not an outage."""
    anchors = [(time, 0.5) for time in _anchor_times(1, 8)]
    anchors += [(f"2026-07-01T23:00:00Z", 0.5), ("2026-07-02T14:00:00Z", 0.5)]
    report = diagnose_capture_era_shift(
        input_csv=_write_table(tmp_path / "windows.csv", anchors),
        out_path=tmp_path / "report.json",
        min_era_anchors=2,
    )
    assert [era["kind"] for era in report["eras"]] == ["era"]


def test_isolated_captures_are_marked_and_excluded_from_the_comparison(tmp_path: Path) -> None:
    anchors = [(time, 0.6) for time in _anchor_times(1, 8)]
    anchors += [("2026-07-12T00:00:00Z", 0.6)]
    anchors += [(time, 0.4) for time in _anchor_times(28, 8)]
    report = diagnose_capture_era_shift(
        input_csv=_write_table(tmp_path / "windows.csv", anchors),
        out_path=tmp_path / "report.json",
    )
    kinds = {era["era_id"]: era["kind"] for era in report["eras"]}
    assert kinds == {"era_0": "era", "era_1": "isolated", "era_2": "era"}
    assert report["compared_eras"] == {"earlier": "era_0", "later": "era_2"}


def test_signal_and_cadence_features_are_reported_separately(tmp_path: Path) -> None:
    report = diagnose_capture_era_shift(
        input_csv=_two_era_table(tmp_path / "windows.csv"),
        out_path=tmp_path / "report.json",
    )
    families = report["family_summary"]
    # Intensity is constant inside each era but differs across them: the two
    # samples are disjoint, which no pooled-sd ratio can express.
    assert families["vlf_signal"]["fully_separated_feature_count"] == 1
    assert families["vlf_cadence_derived"]["max_abs_standardized_mean_difference"] == 0.0
    assert families["vlf_cadence_derived"]["fully_separated_feature_count"] == 0
    shifted = {entry["feature"] for entry in report["feature_comparisons"] if entry["ks_statistic"] > 0}
    assert shifted == {"vlf_image_intensity_mean_avg"}
    # The most-shifted feature must rank first despite its unusable ratio.
    assert report["feature_comparisons"][0]["feature"] == "vlf_image_intensity_mean_avg"
    assert report["feature_comparisons"][0]["standardized_mean_difference"] is None


def test_cadence_classification_separates_aggregation_artifacts() -> None:
    assert classify_feature_family("vlf_capture_count") == "vlf_cadence_derived"
    assert classify_feature_family("vlf_total_bytes") == "vlf_cadence_derived"
    assert classify_feature_family("real_vlf_image_vlf_intensity_mean_sum") == "vlf_cadence_derived"
    assert classify_feature_family("vlf_image_band_0_mean_latest") == "vlf_signal"
    assert classify_feature_family("real_vlf_image_vlf_intensity_mean_mean") == "vlf_signal"
    # A count outside the VLF families is not a capture-cadence artifact.
    assert classify_feature_family("seismic_event_count") == "seismic"


def test_era_csvs_round_trip_every_row(tmp_path: Path) -> None:
    era_dir = tmp_path / "eras"
    report = diagnose_capture_era_shift(
        input_csv=_two_era_table(tmp_path / "windows.csv"),
        out_path=tmp_path / "report.json",
        era_csv_dir=era_dir,
    )
    written = 0
    for era in report["eras"]:
        rows = list(csv.DictReader((era_dir / f"{era['era_id']}.csv").open(newline="", encoding="utf-8")))
        assert len(rows) == era["row_count"]
        written += len(rows)
    assert written == report["row_count"]
    assert json.loads((tmp_path / "report.json").read_text())["status"] == "evaluated"


def test_a_column_constant_in_training_is_treated_as_constant() -> None:
    """125.69 repeated 4199 times has a nonzero float variance; it is still constant."""
    column = [125.69] * 4199
    mean = sum(column) / len(column)
    scale = math.sqrt(sum((value - mean) ** 2 for value in column) / len(column))
    assert 0.0 < scale < RELATIVE_SCALE_TOLERANCE * mean
    assert resolve_scale(scale, mean) == 1.0


def test_genuine_spread_survives_the_guard() -> None:
    assert resolve_scale(2.725686, 42.539474) == 2.725686
    assert resolve_scale(1e-6, 0.0) == 1e-6


def test_constant_training_feature_cannot_saturate_the_holdout() -> None:
    """A train-constant column paired with a different test value must not explode."""
    train = [[125.69] for _ in range(4199)]
    test = [[125.9]]
    _, test_standardized, _, scales = _standardize_train_test(train, test)
    assert scales == [1.0]
    assert abs(test_standardized[0][0]) < 1.0

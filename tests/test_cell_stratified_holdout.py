"""Cover the per-stratum evaluation control added for next-actions item 103.

The point of the control is that a predictor which is constant inside a cell
scores exactly `0.5` there, so cell-stratified balanced accuracy cannot be
inflated by the per-cell base rate that carried every earlier fixed-cell result.
"""

from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

from elfquake.models.temporal_holdout import (
    _stratified_metrics,
    _stratum_base_rate_control,
    evaluate_temporal_holdout,
)


class StratifiedMetricsTests(unittest.TestCase):
    def test_not_stratified_without_strata(self) -> None:
        summary = _stratified_metrics([1, 0], [1, 0], [])
        self.assertEqual(summary["status"], "not_stratified")

    def test_label_transitions_are_counted_beside_every_score(self) -> None:
        # Overlapping target windows make consecutive rows near-copies, so the
        # row count is not the sample size. A cell with one contiguous block of
        # positives carries two label changes no matter how many rows it has.
        strata = ["a"] * 10 + ["b"] * 10
        labels = [0, 0, 0, 1, 1, 1, 0, 0, 0, 0] + [0] * 5 + [1] * 5
        predictions = [0] * 20
        summary = _stratified_metrics(predictions, labels, strata)
        self.assertEqual(summary["strata"]["a"]["label_transitions"], 2)
        self.assertEqual(summary["strata"]["b"]["label_transitions"], 1)
        self.assertEqual(summary["label_transitions"], 3)
        self.assertEqual(summary["strata"]["a"]["row_count"], 10)

    def test_single_class_strata_still_report_zero_transitions(self) -> None:
        summary = _stratified_metrics([0] * 6, [0] * 6, ["a"] * 6)
        self.assertEqual(summary["strata"]["a"]["label_transitions"], 0)
        self.assertEqual(summary["label_transitions"], 0)

    def test_cell_constant_predictor_scores_exactly_half(self) -> None:
        # Cell A is always predicted positive, cell B always negative. Pooled
        # this looks skilful because A is the high-rate cell; per cell it is
        # worth nothing.
        strata = ["a", "a", "a", "a", "b", "b", "b", "b"]
        labels = [1, 1, 1, 0, 0, 0, 0, 1]
        predictions = [1, 1, 1, 1, 0, 0, 0, 0]
        summary = _stratified_metrics(predictions, labels, strata)
        self.assertEqual(summary["status"], "evaluated")
        self.assertEqual(summary["scored_stratum_count"], 2)
        self.assertEqual(summary["mean_balanced_accuracy"], 0.5)
        self.assertEqual(summary["strata"]["a"]["balanced_accuracy"], 0.5)
        self.assertEqual(summary["strata"]["b"]["balanced_accuracy"], 0.5)

    def test_within_cell_discrimination_is_rewarded(self) -> None:
        strata = ["a", "a", "b", "b"]
        labels = [1, 0, 1, 0]
        predictions = [1, 0, 1, 0]
        summary = _stratified_metrics(predictions, labels, strata)
        self.assertEqual(summary["mean_balanced_accuracy"], 1.0)

    def test_single_class_strata_are_counted_not_scored(self) -> None:
        strata = ["a", "a", "b", "b"]
        labels = [1, 0, 0, 0]
        predictions = [1, 0, 0, 0]
        summary = _stratified_metrics(predictions, labels, strata)
        self.assertEqual(summary["stratum_count"], 2)
        self.assertEqual(summary["scored_stratum_count"], 1)
        self.assertEqual(summary["strata"]["b"]["status"], "single_class_stratum")

    def test_no_two_class_strata_reports_a_status_not_a_score(self) -> None:
        summary = _stratified_metrics([0, 0], [0, 0], ["a", "b"])
        self.assertEqual(summary["status"], "no_two_class_strata")
        self.assertNotIn("mean_balanced_accuracy", summary)


class StratumBaseRateControlTests(unittest.TestCase):
    def _rows(self, stratum: str, labels: list[int]) -> list[dict[str, str]]:
        return [{"target_cell_id": stratum, "target_occurred": str(label)} for label in labels]

    def test_base_rate_control_beats_chance_pooled_but_not_stratified(self) -> None:
        train_rows = self._rows("a", [1] * 8 + [0] * 2) + self._rows("b", [0] * 8 + [1] * 2)
        test_rows = self._rows("a", [1, 1, 1, 0]) + self._rows("b", [0, 0, 0, 1])
        control = _stratum_base_rate_control(
            train_rows=train_rows,
            test_rows=test_rows,
            stratify_field="target_cell_id",
        )
        self.assertEqual(control["stratum_count"], 2)
        self.assertGreater(control["test_metrics"]["balanced_accuracy"], 0.5)
        self.assertEqual(
            control["test_metrics_by_stratum"]["mean_balanced_accuracy"], 0.5
        )

    def test_unseen_stratum_falls_back_to_the_overall_training_rate(self) -> None:
        train_rows = self._rows("a", [1, 1, 0, 0])
        test_rows = self._rows("z", [1, 0])
        control = _stratum_base_rate_control(
            train_rows=train_rows,
            test_rows=test_rows,
            stratify_field="target_cell_id",
        )
        self.assertEqual(control["stratum_count"], 1)
        self.assertEqual(control["test_metrics_by_stratum"]["scored_stratum_count"], 1)


class TemporalHoldoutStratificationTests(unittest.TestCase):
    def _write_table(self, path: Path) -> None:
        fieldnames = [
            "window_id",
            "window_start_utc",
            "target_cell_id",
            "target_cell_latitude",
            "seismic_event_count",
            "target_occurred",
        ]
        rows = []
        for index in range(40):
            stamp = f"2026-01-{index + 1:02d}T00:00:00Z"
            for cell, latitude, rate in (("a", 40.0, 0.8), ("b", 44.0, 0.2)):
                label = 1 if (index % 10) < (10 * rate) else 0
                rows.append(
                    {
                        "window_id": f"w{index}_{cell}",
                        "window_start_utc": stamp,
                        "target_cell_id": cell,
                        "target_cell_latitude": f"{latitude}",
                        "seismic_event_count": f"{index % 5}",
                        "target_occurred": str(label),
                    }
                )
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)

    def test_report_carries_the_stratified_summary_and_control(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            table = root / "table.csv"
            out = root / "report.json"
            self._write_table(table)
            report = evaluate_temporal_holdout(
                input_csv=table,
                out_path=out,
                epochs=20,
                group_by_time=True,
                stratify_field="target_cell_id",
            )
            written = json.loads(out.read_text(encoding="utf-8"))
        self.assertEqual(written["stratify_field"], "target_cell_id")
        self.assertEqual(report["stratify_field"], "target_cell_id")
        control = report["baselines"]["stratum_base_rate"]
        self.assertEqual(control["stratify_field"], "target_cell_id")
        self.assertEqual(control["test_metrics_by_stratum"]["mean_balanced_accuracy"], 0.5)
        summary = report["evaluations"]["all_features"]["calibrated_test_metrics_by_stratum"]
        self.assertIn(summary["status"], {"evaluated", "no_two_class_strata"})

    def test_stratification_is_opt_in(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            table = root / "table.csv"
            out = root / "report.json"
            self._write_table(table)
            report = evaluate_temporal_holdout(
                input_csv=table, out_path=out, epochs=20, group_by_time=True
            )
        self.assertEqual(report["stratify_field"], "")
        self.assertNotIn("stratum_base_rate", report["baselines"])
        summary = report["evaluations"]["all_features"]["calibrated_test_metrics_by_stratum"]
        self.assertEqual(summary["status"], "not_stratified")


if __name__ == "__main__":
    unittest.main()

"""Cover the target-design diagnostic added for next-actions item 103.

The diagnostic exists to stop row counts standing in for evidence: with
overlapping target windows a 14,000-row table can carry fewer than thirty
independent label changes.
"""

from __future__ import annotations

import csv
import json
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

from elfquake.models.target_design import _variation, diagnose_spatial_target_design


class VariationTests(unittest.TestCase):
    def test_overlapping_windows_are_counted_as_one_transition(self) -> None:
        # A cell that switches on once and stays on carries one transition,
        # however many anchors repeat the same label afterwards.
        summary = _variation({"cell": [0] * 40 + [1] * 40})
        self.assertEqual(summary["total_label_transitions"], 1)
        self.assertEqual(summary["cells_with_both_classes"], 1)

    def test_constant_cells_are_split_by_class(self) -> None:
        summary = _variation({"hot": [1, 1, 1], "cold": [0, 0, 0], "mixed": [0, 1, 0]})
        self.assertEqual(summary["cells_all_positive"], 1)
        self.assertEqual(summary["cells_all_negative"], 1)
        self.assertEqual(summary["cells_with_both_classes"], 1)
        self.assertEqual(summary["total_label_transitions"], 2)
        self.assertEqual(summary["transitions_by_cell"], {"mixed": 2})


class DiagnoseSpatialTargetDesignTests(unittest.TestCase):
    def _write_inputs(self, root: Path) -> tuple[Path, Path]:
        anchors = root / "anchors.csv"
        events = root / "events.csv"
        base = datetime(2026, 6, 1, tzinfo=timezone.utc)
        with anchors.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=["window_id", "window_start_utc", "window_end_utc"],
                lineterminator="\n",
            )
            writer.writeheader()
            for index in range(48):
                start = base + timedelta(minutes=30 * index)
                writer.writerow(
                    {
                        "window_id": f"w{index}",
                        "window_start_utc": _stamp(start),
                        "window_end_utc": _stamp(start + timedelta(days=1)),
                    }
                )
        with events.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=["event_id", "event_time_utc", "latitude", "longitude", "magnitude"],
                lineterminator="\n",
            )
            writer.writeheader()
            writer.writerow(
                {
                    "event_id": "e1",
                    "event_time_utc": _stamp(base + timedelta(days=3)),
                    "latitude": "43.0",
                    "longitude": "12.0",
                    "magnitude": "3.4",
                }
            )
            writer.writerow(
                {
                    "event_id": "e2",
                    "event_time_utc": _stamp(base + timedelta(days=20)),
                    "latitude": "43.0",
                    "longitude": "12.0",
                    "magnitude": "3.4",
                }
            )
        return anchors, events

    def test_reports_one_design_per_combination(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            anchors, events = self._write_inputs(root)
            out = root / "report.json"
            report = diagnose_spatial_target_design(
                input_csv=anchors,
                events_csv=events,
                out_path=out,
                horizon_days=(1, 3),
                cell_degrees=(1.5,),
                target_magnitude_min=(2.5,),
            )
            written = json.loads(out.read_text(encoding="utf-8"))
        self.assertEqual(len(report["designs"]), 2)
        self.assertEqual(written["anchor_count"], 48)
        self.assertEqual(written["event_count"], 2)
        for design in report["designs"]:
            self.assertEqual(design["status"], "evaluated")
            self.assertEqual(
                design["labeled_row_count"],
                design["labeled_anchor_count"] * design["cell_count"],
            )
            self.assertIn("held_out", design)
            self.assertLessEqual(
                design["held_out"]["total_label_transitions"],
                design["full_record"]["total_label_transitions"],
            )

    def test_magnitude_threshold_above_the_catalog_gives_no_positives(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            anchors, events = self._write_inputs(root)
            report = diagnose_spatial_target_design(
                input_csv=anchors,
                events_csv=events,
                out_path=root / "report.json",
                horizon_days=(3,),
                cell_degrees=(1.5,),
                target_magnitude_min=(5.0,),
            )
        design = report["designs"][0]
        self.assertEqual(design["positive_rate"], 0.0)
        self.assertEqual(design["full_record"]["cells_with_both_classes"], 0)
        self.assertEqual(design["full_record"]["total_label_transitions"], 0)

    def test_anchors_past_the_catalog_end_are_not_labeled(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            anchors, events = self._write_inputs(root)
            early = diagnose_spatial_target_design(
                input_csv=anchors,
                events_csv=events,
                out_path=root / "early.json",
                horizon_days=(3,),
                catalog_end_utc="2026-06-02T00:00:00Z",
            )
        self.assertEqual(early["designs"][0]["status"], "no_mature_anchors")


def _stamp(value: datetime) -> str:
    return value.isoformat().replace("+00:00", "Z")


if __name__ == "__main__":
    unittest.main()

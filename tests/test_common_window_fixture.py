from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

from elfquake.models.common_window_fixture import build_common_window_fixture


class CommonWindowFixtureSplitTests(unittest.TestCase):
    def test_pending_future_rows_do_not_starve_the_test_split(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "italy.csv"
            rows = []
            for day in range(10):
                rows.append(
                    {
                        "window_start_utc": f"2026-07-{day + 1:02d}T00:00:00Z",
                        "target_occurred": "1" if day % 2 == 0 else "0",
                        "target_status": "labeled",
                    }
                )
            # A large pending-future tail (not yet mature) should not push the
            # train/test boundary past every labeled row.
            for day in range(40):
                rows.append(
                    {
                        "window_start_utc": f"2026-08-{day % 28 + 1:02d}T00:00:00Z",
                        "target_occurred": "",
                        "target_status": "unlabeled_pending_future_events",
                    }
                )
            with source.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(
                    handle, fieldnames=["window_start_utc", "target_occurred", "target_status"], lineterminator="\n"
                )
                writer.writeheader()
                writer.writerows(rows)

            out = root / "fixture.csv"
            report_path = root / "fixture.json"
            report = build_common_window_fixture(
                input_csvs=[source], out_path=out, report_path=report_path, dataset_ids=["italy_all"]
            )

            self.assertEqual(report["split_counts"]["train"] + report["split_counts"]["test"], 50)
            with out.open(newline="", encoding="utf-8") as handle:
                written = list(csv.DictReader(handle))
            labeled_test = [
                row for row in written if row["target_status"] == "labeled" and row["model_split"] == "test"
            ]
            self.assertGreater(len(labeled_test), 0, "labeled rows must still receive a non-empty test split")

    def test_rows_without_target_status_field_split_as_before(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "synthetic.csv"
            rows = [{"window_start_utc": f"2026-07-{day + 1:02d}T00:00:00Z"} for day in range(10)]
            with source.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=["window_start_utc"], lineterminator="\n")
                writer.writeheader()
                writer.writerows(rows)

            out = root / "fixture.csv"
            report_path = root / "fixture.json"
            report = build_common_window_fixture(
                input_csvs=[source], out_path=out, report_path=report_path, dataset_ids=["seed40"], train_fraction=0.8
            )

        self.assertEqual(report["split_counts"], {"train": 8, "test": 2})


if __name__ == "__main__":
    unittest.main()

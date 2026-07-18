import csv
import json
import tempfile
import unittest
from pathlib import Path

from elfquake.models.event_catalog_alignment import (
    calibrate_synthetic_catalog,
    calibrate_synthetic_magnitudes,
    calibrate_synthetic_spatial_coordinates,
    combine_synthetic_catalogs,
    compare_event_catalogs,
)


class EventCatalogAlignmentTests(unittest.TestCase):
    def test_compare_and_calibrate_catalogs(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            real = root / "real.csv"
            synthetic = root / "synthetic.csv"
            calibrated = root / "calibrated.csv"
            header = "event_time_utc,latitude,longitude,magnitude\n"
            real.write_text(
                header
                + "2026-01-01T00:00:00Z,42.0,12.0,2.0\n"
                + "2026-01-02T00:00:00Z,43.0,13.0,3.0\n"
                + "2026-01-03T00:00:00Z,44.0,14.0,4.0\n",
                encoding="utf-8",
            )
            synthetic.write_text(
                header
                + "2026-01-01T00:00:00Z,42.0,12.0,4.5\n"
                + "2026-01-02T00:00:00Z,43.0,13.0,4.6\n",
                encoding="utf-8",
            )

            report = compare_event_catalogs(
                real_events=real,
                synthetic_events=[synthetic],
                out_path=root / "report.json",
                synthetic_duration_days=[10.0],
            )
            calibration = calibrate_synthetic_magnitudes(real_events=real, synthetic_events=synthetic, out_path=calibrated)
            rate_calibration = calibrate_synthetic_catalog(
                real_events=real,
                synthetic_events=synthetic,
                out_path=root / "rate_calibrated.csv",
                seed=7,
                synthetic_duration_days=10.0,
            )

            self.assertEqual(report["catalogs"][0]["summary"]["event_count"], 3)
            self.assertEqual(report["catalogs"][1]["summary"]["duration_days"], 10.0)
            self.assertIn("sample_matched_nearest_neighbour_km_wasserstein", report["comparisons_to_real"][0])
            self.assertEqual(calibration["synthetic_event_count"], 2)
            self.assertEqual(rate_calibration["synthetic_event_count"], 2)
            self.assertLessEqual(rate_calibration["retained_event_count"], 2)
            with (root / "rate_calibrated.csv").open(newline="", encoding="utf-8") as handle:
                rate_rows = list(csv.DictReader(handle))
            if rate_rows:
                self.assertIn("spatial_weight", rate_rows[0])
                self.assertIn("spatial_calibration", rate_rows[0])
            with calibrated.open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 2)
            self.assertEqual(rows[0]["magnitude_raw"], "4.5")
            self.assertEqual(rows[0]["magnitude_calibration"], "real_train_empirical_quantile")

            combined = root / "combined.csv"
            combined_report = combine_synthetic_catalogs(
                synthetic_events=[synthetic, synthetic], out_path=combined, offset_days=21
            )
            spatial = root / "spatial.csv"
            spatial_report = calibrate_synthetic_spatial_coordinates(
                real_events=real, synthetic_events=combined, out_path=spatial
            )
            self.assertEqual(combined_report["episode_count"], 2)
            self.assertEqual(combined_report["event_count"], 4)
            self.assertEqual(spatial_report["synthetic_event_count"], 4)


if __name__ == "__main__":
    unittest.main()

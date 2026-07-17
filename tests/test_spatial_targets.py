from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

from elfquake.features.spatial_targets import label_spatial_multimodal_targets


class SpatialTargetTests(unittest.TestCase):
    def test_expands_windows_into_cells_with_positive_and_negative_labels(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "windows.csv"
            source.write_text(
                "window_id,region_id,target_start_utc,target_end_utc,target_magnitude_min,target_status\n"
                "w1,all_italy,2026-07-01T00:00:00Z,2026-07-08T00:00:00Z,2.5,labeled\n",
                encoding="utf-8",
            )
            events = root / "events.csv"
            events.write_text(
                "event_time_utc,latitude,longitude,magnitude\n"
                "2026-07-02T00:00:00Z,42.4,13.2,2.8\n", encoding="utf-8"
            )
            out = root / "spatial.csv"
            rows = label_spatial_multimodal_targets(
                input_csv=source, events_csv=events, out_path=out,
                as_of_utc="2026-07-10T00:00:00Z", catalog_end_utc="2026-07-10T00:00:00Z",
            )
            self.assertGreater(len(rows), 1)
            self.assertEqual(sum(row["target_occurred"] == "1" for row in rows), 1)
            self.assertEqual(sum(row["target_occurred"] == "0" for row in rows), len(rows) - 1)


if __name__ == "__main__":
    unittest.main()

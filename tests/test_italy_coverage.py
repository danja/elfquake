from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from elfquake.features.italy_coverage import build_italy_coverage_report


class ItalyCoverageTests(unittest.TestCase):
    def test_report_counts_overlap_and_anomaly_scores(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            events = root / "events.csv"
            events.write_text(
                "event_time_utc,magnitude\n"
                "2026-07-01T12:00:00Z,2.6\n"
                "2026-07-08T12:00:00Z,3.1\n", encoding="utf-8"
            )
            metadata_root = root / "captures"
            metadata_root.mkdir()
            (metadata_root / "a.jpg.metadata.json").write_text(json.dumps({
                "source_id": "vlf_cumiana_last_E_VLF",
                "captured_at_utc": "2026-07-02T12:00:00Z",
            }), encoding="utf-8")
            scores = root / "scores.csv"
            scores.write_text(
                "window_end_utc,anomaly_score\n2026-07-03T12:00:00Z,0.9\n", encoding="utf-8"
            )
            report = build_italy_coverage_report(
                events_csv=events, vlf_metadata_root=metadata_root,
                anomaly_scores_csv=scores, out_path=root / "report.json",
                weekly_out=root / "weekly.csv",
            )
            self.assertEqual(report["events"]["row_count"], 2)
            self.assertEqual(report["vlf_captures"]["metadata_count"], 1)
            self.assertEqual(report["vlf_anomaly_scores"]["alert_count_ge_0.8"], 1)
            self.assertEqual(report["overlap"]["weeks_with_both"], 1)


if __name__ == "__main__":
    unittest.main()

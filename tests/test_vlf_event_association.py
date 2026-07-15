from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from elfquake.features.vlf_event_association import build_vlf_event_association_report


class VlfEventAssociationTests(unittest.TestCase):
    def test_insufficient_controls_is_explicit(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            events = root / "events.csv"
            events.write_text(
                "event_time_utc,magnitude\n"
                "2026-07-01T12:00:00Z,3.0\n"
                "2026-07-08T12:00:00Z,3.0\n", encoding="utf-8"
            )
            scores = root / "scores.csv"
            scores.write_text(
                "window_end_utc,anomaly_score\n"
                "2026-07-02T12:00:00Z,0.9\n"
                "2026-07-09T12:00:00Z,0.2\n", encoding="utf-8"
            )
            report = build_vlf_event_association_report(
                events_csv=events, anomaly_scores_csv=scores,
                out_path=root / "report.json", permutations=10,
            )
            self.assertEqual(report["status"], "insufficient_controls")
            self.assertEqual(report["results_by_minimum_magnitude"]["3.0"]["control_weeks"], 0)


if __name__ == "__main__":
    unittest.main()

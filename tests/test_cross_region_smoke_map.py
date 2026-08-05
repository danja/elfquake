from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

from elfquake.models.cross_region_smoke import write_map_inputs


class CrossRegionMapInputsTests(unittest.TestCase):
    def test_visualizes_the_most_recent_matured_week_only(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)

            # Two matured test weeks: an old one (early July) and the most
            # recent one (late July). Only the most recent week's rows should
            # be selected as prediction candidates.
            test_rows = [
                {"window_start_utc": "2026-07-01T00:00:00Z", "coord": (0.1, 0.1, 0.6)},
                {"window_start_utc": "2026-07-29T00:00:00Z", "coord": (0.2, 0.2, 0.7)},
                {"window_start_utc": "2026-07-30T00:00:00Z", "coord": (0.3, 0.3, 0.8)},
            ]
            target = root / "target.csv"
            with target.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=["model_split", "window_start_utc", "target_start_utc", "target_event_count"],
                    lineterminator="\n",
                )
                writer.writeheader()
                for row in test_rows:
                    writer.writerow(
                        {
                            "model_split": "test",
                            "window_start_utc": row["window_start_utc"],
                            "target_start_utc": "",
                            "target_event_count": "",
                        }
                    )
                writer.writerow(
                    {
                        "model_split": "train",
                        "window_start_utc": "2026-06-01T00:00:00Z",
                        "target_start_utc": "2026-06-01T00:00:00Z",
                        "target_event_count": "3",
                    }
                )

            report = {
                "runs": [
                    {
                        "regime": "synthetic_then_japan_then_italy",
                        "seed": 42,
                        "downstream_models": {
                            "italy_multimodal": {
                                "fine_tune": {
                                    "calibrated_threshold": 0.5,
                                    "evaluations": {
                                        "trained_input": {
                                            "probabilities": [0.9, 0.9, 0.9],
                                            "coordinate_predictions": [
                                                [row["coord"]] for row in test_rows
                                            ],
                                        }
                                    },
                                }
                            }
                        },
                    }
                ]
            }
            report_path = root / "report.json"
            report_path.write_text(json.dumps(report), encoding="utf-8")

            real_events = root / "events.csv"
            with real_events.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(
                    handle, fieldnames=["event_time_utc", "latitude", "longitude", "magnitude"], lineterminator="\n"
                )
                writer.writeheader()
                writer.writerow(
                    {"event_time_utc": "2026-07-29T12:00:00Z", "latitude": "42.5", "longitude": "12.5", "magnitude": "3.0"}
                )
                # An older actual event, outside the most-recent-week window.
                writer.writerow(
                    {"event_time_utc": "2026-07-01T12:00:00Z", "latitude": "42.5", "longitude": "12.5", "magnitude": "3.0"}
                )

            out_dir = root / "map_inputs"
            metadata = write_map_inputs(report=report_path, target=target, real_events=real_events, out_dir=out_dir)

            self.assertEqual(metadata["holdout_start_utc"], "2026-07-23T00:00:01Z")
            self.assertEqual(metadata["holdout_end_utc"], "2026-07-30T00:00:01Z")
            self.assertEqual(metadata["actual_count"], 1)

            with (out_dir / "heldout_week_predictions.csv").open(newline="", encoding="utf-8") as handle:
                predicted_rows = list(csv.DictReader(handle))
            # Budget allows up to 3 candidates; only 07-29 and 07-30 fall
            # inside the most-recent-week window, so 07-01 must not leak in.
            self.assertEqual(len(predicted_rows), 2)


if __name__ == "__main__":
    unittest.main()

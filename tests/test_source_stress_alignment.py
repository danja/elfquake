import csv
import tempfile
import unittest
from pathlib import Path

from elfquake.sim.source_stress_alignment import analyze_source_stress_alignment


class SourceStressAlignmentTests(unittest.TestCase):
    def test_reports_local_and_global_response(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.csv"
            activity = root / "activity.csv"
            out = root / "alignment.csv"
            source.write_text(
                "step,source_id,x,y,release_count,release_mass\n"
                "2,0,1,1,1,2\n",
                encoding="utf-8",
            )
            activity.write_text(
                "step,active_topple_cell_count,topple_count,centroid_x,centroid_y,"
                "weighted_centroid_x,weighted_centroid_y,min_x,max_x,min_y,max_y,peak_x,peak_y,peak_topple_count\n"
                "0,1,1,1,1,1,1,1,1,1,1,1,1,1\n"
                "1,1,1,1,1,1,1,1,1,1,1,1,1,1\n"
                "2,1,2,1,1,1,1,1,1,1,1,1,1,2\n"
                "3,1,10,1,1,1,1,1,1,1,1,1,1,10\n"
                "4,1,3,20,20,20,20,20,20,20,20,20,20,3\n",
                encoding="utf-8",
            )
            rows = analyze_source_stress_alignment(
                source_stress_csv=source,
                activity_csv=activity,
                out_path=out,
                local_radius=2,
                response_horizon=2,
            )
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["local_peak_lag"], "1")
            self.assertEqual(rows[0]["global_peak_lag"], "1")
            self.assertGreater(float(rows[0]["local_excess_auc"]), 0.0)
            self.assertTrue(out.exists())


if __name__ == "__main__":
    unittest.main()

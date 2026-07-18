import csv
import tempfile
import unittest
from pathlib import Path

from elfquake.models.spatial_permutation import permute_spatial_target_vectors


class SpatialPermutationTests(unittest.TestCase):
    def test_permutation_preserves_spatial_vectors_and_label_totals(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.csv"
            output = root / "permuted.csv"
            source.write_text(
                "window_start_utc,target_cell_id,target_occurred,target_status\n"
                "2026-01-01T00:00:00Z,c1,1,labeled\n"
                "2026-01-01T00:00:00Z,c2,0,labeled\n"
                "2026-01-02T00:00:00Z,c1,0,labeled\n"
                "2026-01-02T00:00:00Z,c2,1,labeled\n",
                encoding="utf-8",
            )

            report = permute_spatial_target_vectors(input_csv=source, out_path=output, seed=7)
            with output.open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))

            self.assertEqual(report["labeled_time_count"], 2)
            self.assertEqual(sum(row["target_occurred"] == "1" for row in rows), 2)
            self.assertEqual({row["target_cell_id"] for row in rows}, {"c1", "c2"})


if __name__ == "__main__":
    unittest.main()

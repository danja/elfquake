from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from elfquake.features.japan_model_input import build_japan_model_input


class JapanModelInputTests(unittest.TestCase):
    def test_dataset_id_stays_filesystem_safe_when_window_spans_many_cdf_files(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            windows = root / "windows.csv"
            windows.write_text(
                "window_id,window_start_utc,window_end_utc,region_id,dataset_id\n"
                "w1,2026-07-15T00:00:00Z,2026-07-22T00:00:00Z,japan,japan_moshiri\n",
                encoding="utf-8",
            )
            source_ids = "+".join(f"japan_moshiri_isee_vlf_mos_202607{15 + i:02d}00_v01" for i in range(20))
            vlf = root / "vlf_windows.csv"
            vlf.write_text(
                "window_id,japan_vlf_row_count,japan_vlf_source_dataset_id\n"
                f"w1,20,multiple:{source_ids}\n",
                encoding="utf-8",
            )
            out = root / "model_input.csv"
            rows = build_japan_model_input(windows_csv=windows, vlf_windows_csv=vlf, out_path=out)

        self.assertEqual(len(rows), 1)
        dataset_id = rows[0]["dataset_id"]
        self.assertLessEqual(len(dataset_id), 64)
        self.assertTrue(dataset_id.startswith("japan_moshiri_multi_"))
        # Deterministic: rerunning the same multi-file combination must reproduce the same id.
        self.assertEqual(dataset_id, "japan_moshiri_multi_" + dataset_id.split("_")[-1])

    def test_dataset_id_uses_single_source_id_unchanged(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            windows = root / "windows.csv"
            windows.write_text(
                "window_id,window_start_utc,window_end_utc,region_id,dataset_id\n"
                "w1,2026-07-15T00:00:00Z,2026-07-22T00:00:00Z,japan,japan_moshiri\n",
                encoding="utf-8",
            )
            vlf = root / "vlf_windows.csv"
            vlf.write_text(
                "window_id,japan_vlf_row_count,japan_vlf_source_dataset_id\n"
                "w1,20,japan_moshiri_isee_vlf_mos_2026071500_v01\n",
                encoding="utf-8",
            )
            out = root / "model_input.csv"
            rows = build_japan_model_input(windows_csv=windows, vlf_windows_csv=vlf, out_path=out)

        self.assertEqual(rows[0]["dataset_id"], "japan_moshiri_isee_vlf_mos_2026071500_v01")


if __name__ == "__main__":
    unittest.main()

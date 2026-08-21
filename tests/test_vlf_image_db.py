"""Palette-inverted absolute-dB VLF features (next-actions item 97(a))."""

from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

import numpy as np

from elfquake.features.vlf_image_db import (
    BAND_COUNT,
    FIELDNAMES,
    MAX_CENSORED_FRACTION,
    build_vlf_image_db_features,
    extract_vlf_image_db_features,
)
from elfquake.features.vlf_palette import (
    COLOURBAR_X0,
    COLOURBAR_X1,
    COLOURBAR_Y0,
    COLOURBAR_Y1,
    DB_PER_STEP,
    EXPECTED_HEIGHT,
    EXPECTED_WIDTH,
    FRESH_X0,
    PANEL_Y0,
    PANEL_Y1,
    SCALE_DB_LOW,
)


def _ramp(black_end: int, red_start: int) -> np.ndarray:
    """A 96-step ramp: black floor, a graded middle, then saturated red.

    The middle is a blue-to-yellow sweep chosen so every resolvable step is a
    distinct colour, which is what nearest-colour inversion needs.
    """
    steps = COLOURBAR_X1 - COLOURBAR_X0
    bar = np.zeros((steps, 3), dtype=np.uint8)
    span = red_start - black_end - 1
    for offset in range(span):
        position = offset / max(span - 1, 1)
        bar[black_end + 1 + offset] = (
            int(20 + 200 * position),
            int(20 + 200 * position),
            int(220 - 200 * position),
        )
    bar[red_start:] = (255, 0, 0)
    return bar


def _write_capture(
    path: Path,
    *,
    black_end: int = 20,
    red_start: int = 59,
    panel_step: int | None = None,
    panel_colour: tuple[int, int, int] | None = None,
) -> None:
    """Render a synthetic capture whose panel is one uniform palette colour.

    Written as PNG rather than JPEG: these tests exercise the inversion, and
    JPEG blur at the ramp boundary moves the detected red onset by a step and
    leaves a fraction of a uniform panel off-colour. Lossy behaviour on real
    captures is covered by the residual masking, which the fixtures here would
    only obscure.
    """
    from PIL import Image

    bar = _ramp(black_end, red_start)
    image = np.zeros((EXPECTED_HEIGHT, EXPECTED_WIDTH, 3), dtype=np.uint8)
    image[COLOURBAR_Y0:COLOURBAR_Y1, COLOURBAR_X0:COLOURBAR_X1] = bar[None, :, :]
    fill = panel_colour if panel_colour is not None else tuple(bar[panel_step])
    image[PANEL_Y0:PANEL_Y1, :] = fill
    Image.fromarray(image).save(path)


class ExtractDbFeatureTests(unittest.TestCase):
    def test_panel_colour_inverts_to_its_own_ramp_step(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "last_E_VLF_2026-07-01T11-00-00Z.png"
            _write_capture(path, black_end=20, red_start=59, panel_step=40)
            row = extract_vlf_image_db_features(path)

        self.assertEqual(row["vlfdb_status"], "ok")
        self.assertEqual(row["vlf_image_captured_at_utc"], "2026-07-01T11:00:00Z")
        expected = round(SCALE_DB_LOW + 40 * DB_PER_STEP, 3)
        for index in range(BAND_COUNT):
            self.assertAlmostEqual(
                float(row[f"vlfdb_band_{index}_db_median"]), expected, places=3
            )

    def test_same_level_decodes_the_same_under_a_shifted_palette(self) -> None:
        # The point of the whole module: a palette move must not change the
        # reported level. Two captures whose ramps differ by 11 steps but whose
        # panels sit at the same absolute dB must decode identically.
        with tempfile.TemporaryDirectory() as directory:
            early = Path(directory) / "last_E_VLF_2026-07-01T11-00-00Z.png"
            late = Path(directory) / "last_E_VLF_2026-07-20T11-00-00Z.png"
            _write_capture(early, black_end=20, red_start=59, panel_step=40)
            _write_capture(late, black_end=9, red_start=48, panel_step=40)
            early_row = extract_vlf_image_db_features(early)
            late_row = extract_vlf_image_db_features(late)

        self.assertNotEqual(early_row["vlfdb_red_start_px"], late_row["vlfdb_red_start_px"])
        self.assertEqual(
            early_row["vlfdb_band_0_db_median"], late_row["vlfdb_band_0_db_median"]
        )

    def test_palette_window_is_recorded_per_capture(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "last_E_VLF_2026-07-20T11-00-00Z.png"
            _write_capture(path, black_end=9, red_start=48, panel_step=30)
            row = extract_vlf_image_db_features(path)

        self.assertEqual(row["vlfdb_red_start_px"], "48")
        self.assertAlmostEqual(
            float(row["vlfdb_resolvable_db_low"]),
            round(SCALE_DB_LOW + 9 * DB_PER_STEP, 3),
            places=3,
        )

    def test_clipped_band_is_withheld_not_reported_as_a_level(self) -> None:
        # A panel below the palette floor reads black. That is missing, not
        # quiet: reporting a number here would put the floor into the features
        # as if it were a measurement.
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "last_E_VLF_2026-07-01T11-00-00Z.png"
            _write_capture(path, panel_colour=(0, 0, 0))
            row = extract_vlf_image_db_features(path)

        self.assertEqual(row["vlfdb_scored_band_count"], "0")
        self.assertEqual(float(row["vlfdb_clipped_black_fraction"]), 1.0)
        for index in range(BAND_COUNT):
            self.assertEqual(row[f"vlfdb_band_{index}_db_median"], "")
            self.assertEqual(float(row[f"vlfdb_band_{index}_censored_fraction"]), 1.0)

    def test_saturated_band_is_withheld_too(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "last_E_VLF_2026-07-01T11-00-00Z.png"
            _write_capture(path, panel_colour=(255, 0, 0))
            row = extract_vlf_image_db_features(path)

        self.assertEqual(row["vlfdb_scored_band_count"], "0")
        self.assertEqual(float(row["vlfdb_clipped_hot_fraction"]), 1.0)
        self.assertEqual(row["vlfdb_band_0_db_median"], "")

    def test_censoring_threshold_is_the_documented_one(self) -> None:
        self.assertEqual(MAX_CENSORED_FRACTION, 0.5)

    def test_unparsable_filename_reports_a_status_not_a_crash(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "no_timestamp_here.png"
            _write_capture(path, panel_step=40)
            row = extract_vlf_image_db_features(path)

        self.assertEqual(row["vlfdb_status"], "no_palette")
        self.assertEqual(row["vlfdb_scored_band_count"], "0")
        self.assertEqual(set(row), set(FIELDNAMES))


class BuildDbFeatureTests(unittest.TestCase):
    def test_rows_are_written_and_reused_from_the_existing_table(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "last_E_VLF_2026-07-01T11-00-00Z.png"
            _write_capture(path, panel_step=40)
            out = root / "features.csv"

            first = build_vlf_image_db_features(image_paths=[path], out_path=out)
            self.assertEqual(len(first), 1)

            # Overwrite the capture with a different level. A cached row must be
            # reused verbatim, so the second build must not pick the change up.
            _write_capture(path, panel_step=30)
            second = build_vlf_image_db_features(image_paths=[path], out_path=out)
            written = list(csv.DictReader(out.open(newline="", encoding="utf-8")))

        self.assertEqual(
            first[0]["vlfdb_band_0_db_median"], second[0]["vlfdb_band_0_db_median"]
        )
        self.assertEqual(len(written), 1)
        self.assertEqual(written[0]["vlfdb_status"], "ok")


if __name__ == "__main__":
    unittest.main()

"""Colourbar palette-shift detection in Cumiana spectrogram captures.

Regression cover for the 2026-08-14 diagnostic of item 95(a). The Cumiana
receiver's colour ramp was changed during the July collector outage while the
`-100 dB .. 0 dB` tick ruler stayed fixed, so identical colours mean different
dB either side of the change and pixel features cannot be pooled across it.

The cases below pin the three properties the diagnostic has to hold:

* the ramp's solid-red onset is the variant key, and a one-pixel wobble in the
  near-black floor must not manufacture a spurious variant;
* decoding through each image's own colourbar puts both variants on the same
  absolute dB axis;
* a band median outside the dB window resolvable in *both* variants is
  reported as censored rather than compared.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from elfquake.features.vlf_palette import (
    COLOURBAR_STEPS,
    COLOURBAR_X0,
    COLOURBAR_X1,
    COLOURBAR_Y0,
    COLOURBAR_Y1,
    DB_PER_STEP,
    EXPECTED_HEIGHT,
    EXPECTED_WIDTH,
    PANEL_X1,
    PANEL_Y0,
    PANEL_Y1,
    SCALE_DB_LOW,
    diagnose_vlf_palette_shift,
)


pytest.importorskip("PIL")


EARLY_BLACK_END = 19
EARLY_RED_START = 59
LATE_BLACK_END = 8
LATE_RED_START = 48


def _build_palette(black_end: int, red_start: int) -> np.ndarray:
    """A monotone ramp with the same censoring structure as the real bar."""
    palette = np.zeros((COLOURBAR_STEPS, 3), dtype=np.uint8)
    for index in range(COLOURBAR_STEPS):
        if index <= black_end:
            palette[index] = (2, 2, 3)
        elif index >= red_start:
            palette[index] = (249, 16, 3)
        else:
            fraction = (index - black_end) / (red_start - black_end)
            palette[index] = (
                int(round(60 + 180 * fraction)),
                int(round(40 + 200 * fraction)),
                int(round(200 - 150 * fraction)),
            )
    return palette


def _step_for_db(db: float) -> int:
    return int(round((db - SCALE_DB_LOW) / DB_PER_STEP))


def _write_capture(
    path: Path,
    *,
    black_end: int,
    red_start: int,
    panel_db: float,
) -> None:
    from PIL import Image

    palette = _build_palette(black_end, red_start)
    pixels = np.zeros((EXPECTED_HEIGHT, EXPECTED_WIDTH, 3), dtype=np.uint8)
    pixels[COLOURBAR_Y0:COLOURBAR_Y1, COLOURBAR_X0:COLOURBAR_X1] = palette[None, :, :]
    pixels[PANEL_Y0:PANEL_Y1, :PANEL_X1] = palette[_step_for_db(panel_db)]
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(pixels, mode="RGB").save(path)


def _capture_path(root: Path, date: str, minute: int) -> Path:
    return root / date / f"last_E_VLF_{date}T12-{minute:02d}-00Z.png"


def _run(root: Path, tmp_path: Path) -> dict:
    return diagnose_vlf_palette_shift(
        image_paths=sorted(root.rglob("*.png")),
        out_path=tmp_path / "report.json",
        capture_csv_path=tmp_path / "captures.csv",
        band_csv_path=tmp_path / "bands.csv",
    )


def test_palette_change_is_detected_across_the_outage(tmp_path: Path) -> None:
    root = tmp_path / "captures"
    for minute in (0, 30):
        _write_capture(
            _capture_path(root, "2026-07-05", minute),
            black_end=EARLY_BLACK_END,
            red_start=EARLY_RED_START,
            panel_db=-60.0,
        )
    for minute in (0, 30):
        _write_capture(
            _capture_path(root, "2026-07-29", minute),
            black_end=LATE_BLACK_END,
            red_start=LATE_RED_START,
            panel_db=-70.0,
        )

    report = _run(root, tmp_path)

    assert report["status"] == "palette_changed"
    assert report["capture_count"] == 4
    variants = report["palette_variants"]
    assert len(variants) == 2
    assert variants[0]["red_start_px"] == EARLY_RED_START
    assert variants[0]["display_db_low"] == pytest.approx(-80.0, abs=0.01)
    assert variants[1]["red_start_px"] == LATE_RED_START
    assert variants[1]["display_db_low"] == pytest.approx(-91.579, abs=0.01)

    assert report["ramp_shift_px"] == LATE_RED_START - EARLY_RED_START
    assert report["display_db_shift"] == pytest.approx(-11.579, abs=0.01)

    changes = report["change_points"]
    assert len(changes) == 1
    assert changes[0]["last_capture_before_utc"] == "2026-07-05T12:30:00Z"
    assert changes[0]["first_capture_after_utc"] == "2026-07-29T12:00:00Z"

    # The two eras only overlap between the later floor and the earlier top.
    assert report["common_resolvable_db"]["low"] == pytest.approx(-80.0, abs=0.01)
    assert report["common_resolvable_db"]["high"] == pytest.approx(-49.474, abs=0.01)

    written = json.loads((tmp_path / "report.json").read_text(encoding="utf-8"))
    assert written["status"] == "palette_changed"


def test_near_black_floor_wobble_does_not_split_a_variant(tmp_path: Path) -> None:
    root = tmp_path / "captures"
    _write_capture(
        _capture_path(root, "2026-07-05", 0),
        black_end=EARLY_BLACK_END,
        red_start=EARLY_RED_START,
        panel_db=-60.0,
    )
    # Same palette, one ramp step of near-black noise on the floor.
    _write_capture(
        _capture_path(root, "2026-07-05", 30),
        black_end=EARLY_BLACK_END - 1,
        red_start=EARLY_RED_START,
        panel_db=-60.0,
    )
    _write_capture(
        _capture_path(root, "2026-07-29", 0),
        black_end=LATE_BLACK_END,
        red_start=LATE_RED_START,
        panel_db=-70.0,
    )

    report = _run(root, tmp_path)

    assert [variant["red_start_px"] for variant in report["palette_variants"]] == [
        EARLY_RED_START,
        LATE_RED_START,
    ]
    early = report["palette_variants"][0]
    assert early["capture_count"] == 2
    assert early["black_end_px_min"] == EARLY_BLACK_END - 1
    assert early["black_end_px_max"] == EARLY_BLACK_END
    assert len(report["change_points"]) == 1


def test_bands_are_compared_on_a_common_db_axis(tmp_path: Path) -> None:
    root = tmp_path / "captures"
    _write_capture(
        _capture_path(root, "2026-07-05", 0),
        black_end=EARLY_BLACK_END,
        red_start=EARLY_RED_START,
        panel_db=-60.0,
    )
    _write_capture(
        _capture_path(root, "2026-07-29", 0),
        black_end=LATE_BLACK_END,
        red_start=LATE_RED_START,
        panel_db=-70.0,
    )

    comparison = _run(root, tmp_path)["comparison"]

    assert comparison["status"] == "compared"
    assert comparison["early"]["capture_count"] == 1
    assert comparison["late"]["capture_count"] == 1
    # A flat synthetic panel decodes to one level, so every band agrees.
    for band in comparison["bands"]:
        assert band["resolvable_in_both"] is True
        assert band["early_median_db"] == pytest.approx(-60.0, abs=DB_PER_STEP)
        assert band["late_median_db"] == pytest.approx(-70.0, abs=DB_PER_STEP)
        assert band["delta_db"] == pytest.approx(-10.0, abs=2 * DB_PER_STEP)


def test_level_below_the_shared_floor_is_reported_as_censored(tmp_path: Path) -> None:
    root = tmp_path / "captures"
    _write_capture(
        _capture_path(root, "2026-07-05", 0),
        black_end=EARLY_BLACK_END,
        red_start=EARLY_RED_START,
        panel_db=-60.0,
    )
    # -85 dB is resolvable under the later palette but black under the earlier
    # one, so the pair must not be reported as a like-for-like comparison.
    _write_capture(
        _capture_path(root, "2026-07-29", 0),
        black_end=LATE_BLACK_END,
        red_start=LATE_RED_START,
        panel_db=-85.0,
    )

    comparison = _run(root, tmp_path)["comparison"]

    assert comparison["resolvable_band_count"] == 0
    assert comparison["status"] == "no_band_resolvable_in_both"
    for band in comparison["bands"]:
        assert band["resolvable_in_both"] is False
        assert band["late_median_db"] < comparison["resolvable_db_low"]

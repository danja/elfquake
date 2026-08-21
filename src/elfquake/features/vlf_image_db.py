"""Palette-inverted absolute-dB features from Cumiana VLF spectrograms.

Answers [Next Actions](../../../docs/next-actions.md) item 97(a). The pixel
features in `vlf_image.py` are functions of colour, and the receiver's colour
ramp moved by `11.58 dB` during the July 2026 outage, so the same colour denotes
a different level either side of it (see `docs/vlf-palette-shift.md`). Those
features are readings on two rulers and must not be pooled across the change.

Every capture embeds the colourbar it was drawn with, so pixels can be inverted
back to absolute dB through that image's own ramp. The result is invariant to
the palette setting by construction.

**Censoring is the whole difficulty and is reported, not hidden.** A palette
clips to black below its floor and to solid red above its top, and the two
variants resolve different dB windows: `-80.0 … -37.9` before the change and
`-91.6 … -49.5` after. A band that sits below the floor reads as *missing*, not
as quiet, and taking the median of the pixels that survive would report the
floor itself as a measurement and bias every censored band upward. So each band
carries `censored_fraction`, and its median is withheld entirely once censoring
passes `MAX_CENSORED_FRACTION`.

Being era-invariant on the *ruler* does not make the eras poolable on its own:
after decoding, the late era still reads 16-23 dB lower, which may be a receiver
gain change. This module removes a known scale artifact. It does not settle
whether the residual step is instrumental or atmospheric, and it creates no
evidence about earthquake-related signals.
"""

from __future__ import annotations

import csv
from pathlib import Path

from elfquake.features.vlf_palette import (
    DEFAULT_BANDS_KHZ,
    DEFAULT_RESIDUAL_LIMIT,
    FRESH_X0,
    FRESH_X1,
    PANEL_Y0,
    PANEL_Y1,
    _band_row_slices,
    _decode_panel_db,
    _load_pixels,
    _read_capture_palette,
)

BAND_COUNT = len(DEFAULT_BANDS_KHZ)

# Above this censored fraction a band's median is withheld. A band that is
# mostly clipped has no level to report, and reporting the median of whatever
# escaped clipping would systematically read the palette floor as signal.
MAX_CENSORED_FRACTION = 0.5

FIELDNAMES = [
    "vlf_image_source_file",
    "vlf_image_captured_at_utc",
    "vlfdb_status",
    # Palette identity per capture, so a third variant is visible in the
    # features rather than only in the diagnostic (item 97(c)).
    "vlfdb_black_end_px",
    "vlfdb_red_start_px",
    "vlfdb_resolvable_db_low",
    "vlfdb_resolvable_db_high",
    # Panel-wide inversion quality.
    "vlfdb_clipped_black_fraction",
    "vlfdb_clipped_hot_fraction",
    "vlfdb_masked_fraction",
    "vlfdb_median_residual",
    "vlfdb_censored_fraction",
    "vlfdb_scored_band_count",
]
for _index in range(BAND_COUNT):
    FIELDNAMES.append(f"vlfdb_band_{_index}_db_median")
    FIELDNAMES.append(f"vlfdb_band_{_index}_db_p10")
    FIELDNAMES.append(f"vlfdb_band_{_index}_db_p90")
    FIELDNAMES.append(f"vlfdb_band_{_index}_censored_fraction")

BAND_EDGES_KHZ = tuple(DEFAULT_BANDS_KHZ)


def build_vlf_image_db_features(
    *,
    image_paths: list[Path],
    out_path: Path,
    residual_limit: float = DEFAULT_RESIDUAL_LIMIT,
) -> list[dict[str, str]]:
    """Extract absolute-dB features for each capture, reusing cached rows.

    Decoding is the expensive step, so rows already present for a source file
    are reused. Unlike the pixel features there is no crop parameter to
    invalidate against: the panel and band geometry are fixed constants of the
    capture layout.
    """
    existing = _read_existing_features(out_path)
    rows: list[dict[str, str]] = []
    for image_path in image_paths:
        cached = existing.get(str(image_path))
        if cached is not None:
            rows.append(cached)
            continue
        rows.append(extract_vlf_image_db_features(
            image_path, residual_limit=residual_limit
        ))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    return rows


def extract_vlf_image_db_features(
    image_path: Path,
    *,
    residual_limit: float = DEFAULT_RESIDUAL_LIMIT,
) -> dict[str, str]:
    """Invert one capture to absolute dB through its own embedded colourbar."""
    import numpy as np

    record = _read_capture_palette(image_path)
    if record is None:
        return _empty_row(image_path, status="no_palette")

    pixels = _load_pixels(image_path)
    if pixels is None:
        return _empty_row(
            image_path, status="unreadable", captured_at=str(record["captured_at_utc"])
        )

    panel = pixels[PANEL_Y0:PANEL_Y1, FRESH_X0:FRESH_X1]
    decoded = _decode_panel_db(
        panel=panel,
        palette=record["palette"],
        black_end=int(record["black_end_px"]),
        red_start=int(record["red_start_px"]),
        residual_limit=residual_limit,
    )

    row: dict[str, str] = {
        "vlf_image_source_file": str(image_path),
        "vlf_image_captured_at_utc": str(record["captured_at_utc"]),
        "vlfdb_status": "ok",
        "vlfdb_black_end_px": str(record["black_end_px"]),
        "vlfdb_red_start_px": str(record["red_start_px"]),
        "vlfdb_resolvable_db_low": _fmt(record["display_db_low"]),
        "vlfdb_resolvable_db_high": _fmt(record["display_db_high"]),
        "vlfdb_clipped_black_fraction": _fmt(decoded["clipped_black_fraction"]),
        "vlfdb_clipped_hot_fraction": _fmt(decoded["clipped_hot_fraction"]),
        "vlfdb_masked_fraction": _fmt(decoded["masked_fraction"]),
        "vlfdb_median_residual": _fmt(decoded["median_residual"]),
    }

    db = decoded["db"]
    censored_total = 0.0
    pixel_total = 0
    scored_bands = 0
    for index, (start, stop) in enumerate(_band_row_slices()):
        block = db[start:stop]
        valid = block[np.isfinite(block)]
        censored = block.size - valid.size
        censored_total += censored
        pixel_total += block.size
        fraction = censored / block.size if block.size else 1.0
        row[f"vlfdb_band_{index}_censored_fraction"] = _fmt(round(fraction, 4))
        if valid.size and fraction <= MAX_CENSORED_FRACTION:
            row[f"vlfdb_band_{index}_db_median"] = _fmt(round(float(np.median(valid)), 3))
            row[f"vlfdb_band_{index}_db_p10"] = _fmt(
                round(float(np.percentile(valid, 10)), 3)
            )
            row[f"vlfdb_band_{index}_db_p90"] = _fmt(
                round(float(np.percentile(valid, 90)), 3)
            )
            scored_bands += 1
        else:
            # Withheld, not zero. A censored band has no level to report and a
            # numeric fill would be read downstream as a measurement.
            row[f"vlfdb_band_{index}_db_median"] = ""
            row[f"vlfdb_band_{index}_db_p10"] = ""
            row[f"vlfdb_band_{index}_db_p90"] = ""

    row["vlfdb_censored_fraction"] = _fmt(
        round(censored_total / pixel_total, 4) if pixel_total else 1.0
    )
    row["vlfdb_scored_band_count"] = str(scored_bands)
    return row


def _read_existing_features(path: Path) -> dict[str, dict[str, str]]:
    if not path.exists():
        return {}
    with path.open(newline="", encoding="utf-8") as handle:
        return {
            row["vlf_image_source_file"]: {
                field: row.get(field, "") for field in FIELDNAMES
            }
            for row in csv.DictReader(handle)
            if row.get("vlf_image_source_file")
        }


def _empty_row(
    image_path: Path, *, status: str, captured_at: str = ""
) -> dict[str, str]:
    row = {field: "" for field in FIELDNAMES}
    row["vlf_image_source_file"] = str(image_path)
    row["vlf_image_captured_at_utc"] = captured_at
    row["vlfdb_status"] = status
    row["vlfdb_scored_band_count"] = "0"
    return row


def _fmt(value: object) -> str:
    return "" if value is None else str(value)

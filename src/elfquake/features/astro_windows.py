"""Astronomy and geomagnetic features aligned onto window anchors.

This replaces the coarse capture-counting features in `astronomy.py`. Those
counted collector bookkeeping (`astro_capture_count`) and carried a monthly
constant (`astro_noaa_solar_cycle_f107_value`), neither of which is an
observation of anything at the window anchor.

Two kinds of channel are produced, and they have different missingness
contracts:

* **Ephemeris** channels are deterministic functions of UTC. They are always
  present, so their masks are honestly always `0`.
* **Observed** channels come from normalized GFZ Kp/ap, Kyoto Dst and
  Spaceweather Canada F10.7 tables. They are frequently missing, and each
  carries its own `quality_missing_*` flag and an age in hours.

## Alignment rule

Anchors are the window end (the VLF capture time). Source cadences are 3-hourly
(Kp/ap), hourly (Dst) and daily (F10.7).

* **Zero-order hold, never interpolation.** Interpolating between the readings
  either side of an anchor would mix a future reading into a feature that is
  supposed to be causal. Each channel takes the most recent reading whose
  *observation interval has closed* at or before the anchor.
* **Staleness is reported, not hidden.** `astro_*_age_hours` is the gap from
  the end of that observation interval to the anchor.
* **Held values expire.** Past a per-channel limit the hold is abandoned: the
  value is blank and the channel's `quality_missing_*` flag is `1`. Holding a
  three-day-old Kp reading forward would manufacture a constant, which is
  exactly the failure this module exists to remove.
"""

from __future__ import annotations

import csv
import math
from bisect import bisect_right
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

from elfquake.features.common import parse_utc
from elfquake.features.ephemeris import (
    ITALY_REFERENCE_LATITUDE,
    ITALY_REFERENCE_LONGITUDE,
    moon_illuminated_fraction,
    moon_phase_angle_deg,
    moon_position,
    tidal_potential,
)


FIELDNAMES = [
    "window_start_utc",
    "window_end_utc",
    # Ephemeris: deterministic, never missing.
    "astro_moon_phase_angle_deg",
    "astro_moon_phase_sin",
    "astro_moon_phase_cos",
    "astro_moon_illuminated_fraction",
    "astro_moon_distance_km",
    "astro_tidal_potential",
    "astro_tidal_potential_min",
    "astro_tidal_potential_max",
    "astro_tidal_potential_range",
    # Observed: geomagnetic and solar indices, each with age and a mask.
    "astro_kp",
    "astro_ap",
    "astro_kp_max",
    "astro_ap_max",
    "astro_kp_age_hours",
    "astro_dst_nt",
    "astro_dst_min_nt",
    "astro_dst_age_hours",
    "astro_f107",
    "astro_f107_age_hours",
    "quality_missing_kp",
    "quality_missing_dst",
    "quality_missing_f107",
    "quality_missing_astro",
]

# Cadence of each source, as the length of one observation interval.
KP_INTERVAL_HOURS = 3.0
DST_INTERVAL_HOURS = 1.0
F107_INTERVAL_HOURS = 24.0

# How long a held value stays usable past the end of its interval. Kp and Dst
# allow one missed publication; F10.7 allows a long holiday weekend.
KP_MAX_AGE_HOURS = 6.0
DST_MAX_AGE_HOURS = 6.0
F107_MAX_AGE_HOURS = 72.0

# Samples used to summarize the tidal potential across the lookback window.
# The dominant constituents are semi-diurnal, so a 30-minute step resolves the
# peaks without a meaningful aliasing risk.
TIDAL_SAMPLE_MINUTES = 30


@dataclass(frozen=True)
class Reading:
    """One observation, timestamped at the end of its interval."""

    end_utc: datetime
    values: dict[str, float]


@dataclass(frozen=True)
class SpaceWeatherSeries:
    """Normalized space-weather tables, indexed for anchor lookup."""

    kp: list[Reading]
    dst: list[Reading]
    f107: list[Reading]

    @classmethod
    def empty(cls) -> "SpaceWeatherSeries":
        return cls(kp=[], dst=[], f107=[])


def load_space_weather_series(
    *,
    kp_csv: Path | None = None,
    dst_csv_paths: list[Path] | None = None,
    f107_csv: Path | None = None,
) -> SpaceWeatherSeries:
    kp = _load_kp(kp_csv) if kp_csv else []
    dst: list[Reading] = []
    for path in dst_csv_paths or []:
        dst.extend(_load_dst(path))
    f107 = _load_f107(f107_csv) if f107_csv else []
    return SpaceWeatherSeries(
        kp=_deduplicate(kp),
        dst=_deduplicate(dst),
        f107=_deduplicate(f107),
    )


def build_astro_window_features(
    *,
    window_start_utc: str,
    window_end_utc: str,
    series: SpaceWeatherSeries,
    latitude_deg: float = ITALY_REFERENCE_LATITUDE,
    longitude_deg: float = ITALY_REFERENCE_LONGITUDE,
) -> dict[str, str]:
    window_start = parse_utc(window_start_utc)
    window_end = parse_utc(window_end_utc)
    if window_end <= window_start:
        raise ValueError("window_end_utc must be after window_start_utc")

    row = {"window_start_utc": window_start_utc, "window_end_utc": window_end_utc}
    row.update(
        _ephemeris_fields(
            window_start=window_start,
            window_end=window_end,
            latitude_deg=latitude_deg,
            longitude_deg=longitude_deg,
        )
    )
    row.update(_kp_fields(series.kp, window_start=window_start, window_end=window_end))
    row.update(_dst_fields(series.dst, window_start=window_start, window_end=window_end))
    row.update(_f107_fields(series.f107, window_end=window_end))
    # An astronomy modality is missing when nothing was *observed*, not when a
    # once-pulled monthly constant happens to be on disk. Ephemeris channels
    # are deliberately excluded from this test: they are always computable, so
    # letting them satisfy it would pin the flag to `0` forever.
    row["quality_missing_astro"] = (
        "1"
        if row["quality_missing_kp"] == "1"
        and row["quality_missing_dst"] == "1"
        and row["quality_missing_f107"] == "1"
        else "0"
    )
    return row


def write_astro_window_features(rows: list[dict[str, str]], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES, lineterminator="\n")
        writer.writeheader()
        writer.writerows({field: row.get(field, "") for field in FIELDNAMES} for row in rows)


def _ephemeris_fields(
    *,
    window_start: datetime,
    window_end: datetime,
    latitude_deg: float,
    longitude_deg: float,
) -> dict[str, str]:
    phase_angle = moon_phase_angle_deg(window_end)
    phase_radians = math.radians(phase_angle)
    anchor_tide = tidal_potential(
        window_end, latitude_deg=latitude_deg, longitude_deg=longitude_deg
    ).combined
    samples = [
        tidal_potential(
            moment, latitude_deg=latitude_deg, longitude_deg=longitude_deg
        ).combined
        for moment in _sample_times(window_start, window_end)
    ]
    return {
        "astro_moon_phase_angle_deg": _format(phase_angle),
        # A cyclic quantity needs a wraparound-free encoding; the raw angle
        # jumps 360 -> 0 at new moon, which no model should have to absorb.
        "astro_moon_phase_sin": _format(math.sin(phase_radians)),
        "astro_moon_phase_cos": _format(math.cos(phase_radians)),
        "astro_moon_illuminated_fraction": _format(moon_illuminated_fraction(window_end)),
        "astro_moon_distance_km": _format(moon_position(window_end).distance_km),
        "astro_tidal_potential": _format(anchor_tide),
        "astro_tidal_potential_min": _format(min(samples)),
        "astro_tidal_potential_max": _format(max(samples)),
        "astro_tidal_potential_range": _format(max(samples) - min(samples)),
    }


def _sample_times(window_start: datetime, window_end: datetime) -> list[datetime]:
    step = timedelta(minutes=TIDAL_SAMPLE_MINUTES)
    moments = []
    moment = window_start
    while moment < window_end:
        moments.append(moment)
        moment += step
    moments.append(window_end)
    return moments


def _kp_fields(
    readings: list[Reading], *, window_start: datetime, window_end: datetime
) -> dict[str, str]:
    held = _hold(readings, anchor=window_end, max_age_hours=KP_MAX_AGE_HOURS)
    within = _within(readings, window_start=window_start, window_end=window_end)
    if held is None:
        return {
            "astro_kp": "",
            "astro_ap": "",
            "astro_kp_max": "",
            "astro_ap_max": "",
            "astro_kp_age_hours": "",
            "quality_missing_kp": "1",
        }
    reading, age_hours = held
    # When no bin closed inside the window the held reading is the only
    # information there is, so the window maximum falls back to it.
    covering = within or [reading]
    return {
        "astro_kp": _format(reading.values["kp"]),
        "astro_ap": _format(reading.values["ap"]),
        "astro_kp_max": _format(max(item.values["kp"] for item in covering)),
        "astro_ap_max": _format(max(item.values["ap"] for item in covering)),
        "astro_kp_age_hours": _format(age_hours),
        "quality_missing_kp": "0",
    }


def _dst_fields(
    readings: list[Reading], *, window_start: datetime, window_end: datetime
) -> dict[str, str]:
    held = _hold(readings, anchor=window_end, max_age_hours=DST_MAX_AGE_HOURS)
    within = _within(readings, window_start=window_start, window_end=window_end)
    if held is None:
        return {
            "astro_dst_nt": "",
            "astro_dst_min_nt": "",
            "astro_dst_age_hours": "",
            "quality_missing_dst": "1",
        }
    reading, age_hours = held
    covering = within or [reading]
    # Dst goes negative during a storm, so the minimum is the storm depth.
    return {
        "astro_dst_nt": _format(reading.values["dst_nt"]),
        "astro_dst_min_nt": _format(min(item.values["dst_nt"] for item in covering)),
        "astro_dst_age_hours": _format(age_hours),
        "quality_missing_dst": "0",
    }


def _f107_fields(readings: list[Reading], *, window_end: datetime) -> dict[str, str]:
    held = _hold(readings, anchor=window_end, max_age_hours=F107_MAX_AGE_HOURS)
    if held is None:
        return {"astro_f107": "", "astro_f107_age_hours": "", "quality_missing_f107": "1"}
    reading, age_hours = held
    return {
        "astro_f107": _format(reading.values["f107"]),
        "astro_f107_age_hours": _format(age_hours),
        "quality_missing_f107": "0",
    }


def _hold(
    readings: list[Reading], *, anchor: datetime, max_age_hours: float
) -> tuple[Reading, float] | None:
    """Most recent reading closed at or before the anchor, if not too stale."""
    if not readings:
        return None
    index = bisect_right([item.end_utc for item in readings], anchor)
    if index == 0:
        return None
    reading = readings[index - 1]
    age_hours = (anchor - reading.end_utc).total_seconds() / 3600.0
    if age_hours > max_age_hours:
        return None
    return reading, age_hours


def _within(
    readings: list[Reading], *, window_start: datetime, window_end: datetime
) -> list[Reading]:
    return [item for item in readings if window_start < item.end_utc <= window_end]


def _load_kp(path: Path) -> list[Reading]:
    readings = []
    for row in _read_csv(path):
        start = _parse_date(row.get("date", ""))
        if start is None or not row.get("kp") or not row.get("ap"):
            continue
        slot = int(row["slot"])
        # `slot` indexes the 3-hour bin; the bin closes 3 hours after it opens.
        end = start + timedelta(hours=(slot + 1) * KP_INTERVAL_HOURS)
        readings.append(
            Reading(end_utc=end, values={"kp": float(row["kp"]), "ap": float(row["ap"])})
        )
    return readings


def _load_dst(path: Path) -> list[Reading]:
    readings = []
    for row in _read_csv(path):
        start = _parse_date(row.get("date", ""))
        if start is None or row.get("dst_nt") in (None, ""):
            continue
        hour = int(row["hour"])
        end = start + timedelta(hours=hour + DST_INTERVAL_HOURS)
        readings.append(Reading(end_utc=end, values={"dst_nt": float(row["dst_nt"])}))
    return readings


def _load_f107(path: Path) -> list[Reading]:
    readings = []
    for row in _read_csv(path):
        if not row.get("f107"):
            continue
        # `time_utc` is the observation time when the source provides one; the
        # daily-only fallback closes at the end of its day.
        if row.get("time_utc"):
            end = parse_utc(row["time_utc"])
        else:
            start = _parse_date(row.get("date", ""))
            if start is None:
                continue
            end = start + timedelta(hours=F107_INTERVAL_HOURS)
        readings.append(Reading(end_utc=end, values={"f107": float(row["f107"])}))
    return readings


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _parse_date(value: str) -> datetime | None:
    if not value:
        return None
    return parse_utc(f"{value}T00:00:00Z")


def _deduplicate(readings: list[Reading]) -> list[Reading]:
    """Sort by interval end, keeping the last reading for a repeated end.

    Overlapping monthly Dst pages are the usual source of repeats, and the
    later page is the later publication.
    """
    by_end: dict[datetime, Reading] = {}
    for reading in readings:
        by_end[reading.end_utc] = reading
    return [by_end[key] for key in sorted(by_end)]


def _format(value: float) -> str:
    return f"{value:.6g}"

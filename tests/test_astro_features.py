"""Tests for the ephemeris, window alignment, and channel gate."""

from __future__ import annotations

import math
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

from elfquake.features import ephemeris
from elfquake.features.astro_windows import (
    build_astro_window_features,
    load_space_weather_series,
)
from elfquake.models.channel_gate import (
    ChannelGateError,
    audit_channels,
    mask_fields,
    numeric_channels,
    raise_for_defects,
)


class EphemerisTests(unittest.TestCase):
    def test_julian_day_matches_known_epochs(self) -> None:
        self.assertEqual(
            ephemeris.julian_day(datetime(2000, 1, 1, 12, tzinfo=timezone.utc)), 2451545.0
        )
        self.assertEqual(
            ephemeris.julian_day(datetime(1987, 1, 27, tzinfo=timezone.utc)), 2446822.5
        )

    def test_julian_day_rejects_naive_datetime(self) -> None:
        with self.assertRaises(ValueError):
            ephemeris.julian_day(datetime(2026, 8, 14))

    def test_moon_position_matches_meeus_example(self) -> None:
        # Meeus, Astronomical Algorithms, example 47.a: 1992 April 12, 0h TD.
        # The truncated series used here is good to roughly two arcminutes,
        # and UTC differs from TD by about a minute at that date.
        moon = ephemeris.moon_position(datetime(1992, 4, 12, tzinfo=timezone.utc))
        self.assertAlmostEqual(moon.longitude_deg, 133.162655, delta=0.05)
        self.assertAlmostEqual(moon.latitude_deg, -3.229126, delta=0.01)
        self.assertAlmostEqual(moon.distance_km, 368409.7, delta=50.0)

    def test_sun_position_matches_meeus_example(self) -> None:
        # Meeus example 25.a: 1992 October 13, 0h TD, true longitude 199.90988.
        sun = ephemeris.sun_position(datetime(1992, 10, 13, tzinfo=timezone.utc))
        self.assertAlmostEqual(sun.longitude_deg, 199.90988, delta=0.005)
        self.assertEqual(sun.latitude_deg, 0.0)

    def test_sidereal_time_matches_meeus_example(self) -> None:
        # Meeus example 12.a: 1987 April 10, 0h UT -> 197.693195 degrees.
        self.assertAlmostEqual(
            ephemeris.greenwich_mean_sidereal_time_deg(
                datetime(1987, 4, 10, tzinfo=timezone.utc)
            ),
            197.693195,
            delta=0.001,
        )

    def test_phase_angle_locates_2026_08_new_and_full_moon(self) -> None:
        # Published times: new moon 2026-08-12 17:37Z, full moon 2026-08-28 04:18Z.
        new_moon = datetime(2026, 8, 12, 17, 37, tzinfo=timezone.utc)
        full_moon = datetime(2026, 8, 28, 4, 18, tzinfo=timezone.utc)
        # Near new moon the elongation is close to 0 or 360; compare on the circle.
        self.assertLess(_angular_distance(ephemeris.moon_phase_angle_deg(new_moon), 0.0), 1.0)
        self.assertLess(_angular_distance(ephemeris.moon_phase_angle_deg(full_moon), 180.0), 1.0)
        self.assertLess(ephemeris.moon_illuminated_fraction(new_moon), 0.001)
        self.assertGreater(ephemeris.moon_illuminated_fraction(full_moon), 0.999)

    def test_phase_angle_is_continuous_across_new_moon(self) -> None:
        """The angle wraps, but its sine and cosine encoding does not."""
        before = datetime(2026, 8, 12, 16, tzinfo=timezone.utc)
        after = datetime(2026, 8, 12, 19, tzinfo=timezone.utc)
        raw_before = ephemeris.moon_phase_angle_deg(before)
        raw_after = ephemeris.moon_phase_angle_deg(after)
        # The raw angle jumps the full circle; this is why it is encoded.
        self.assertGreater(abs(raw_before - raw_after), 350.0)
        self.assertLess(
            abs(math.sin(math.radians(raw_before)) - math.sin(math.radians(raw_after))),
            0.05,
        )

    def test_tidal_potential_is_semidiurnal(self) -> None:
        """Two maxima per day is the signature of the degree-two tide."""
        start = datetime(2026, 8, 14, tzinfo=timezone.utc)
        series = [
            ephemeris.tidal_potential(start + timedelta(minutes=10 * step)).combined
            for step in range(144)
        ]
        peaks = [
            index
            for index in range(1, len(series) - 1)
            if series[index] > series[index - 1] and series[index] > series[index + 1]
        ]
        self.assertEqual(len(peaks), 2)

    def test_solar_tide_is_about_46_percent_of_lunar(self) -> None:
        self.assertAlmostEqual(ephemeris.SOLAR_TIDE_RELATIVE_COEFFICIENT, 0.46, delta=0.01)


class AstroWindowAlignmentTests(unittest.TestCase):
    def _series(self, root: Path):
        (root / "gfz_kp_ap.csv").write_text(
            "date,slot,kp,ap,source_file\n"
            "2026-08-14,0,1.000,4,kp.txt\n"
            "2026-08-14,1,2.000,7,kp.txt\n"
            "2026-08-14,2,5.000,48,kp.txt\n",
            encoding="utf-8",
        )
        (root / "kyoto_dst_202608.csv").write_text(
            "date,hour,dst_nt,dst_tier,source_file\n"
            "2026-08-14,5,-20,realtime,dst.html\n"
            "2026-08-14,6,-45,realtime,dst.html\n"
            "2026-08-14,7,-30,realtime,dst.html\n",
            encoding="utf-8",
        )
        (root / "f107_daily.csv").write_text(
            "date,time_utc,f107,f107_observed,source_file\n"
            "2026-08-13,2026-08-13T20:00:00Z,110.4,107.6,f107.txt\n",
            encoding="utf-8",
        )
        return load_space_weather_series(
            kp_csv=root / "gfz_kp_ap.csv",
            dst_csv_paths=[root / "kyoto_dst_202608.csv"],
            f107_csv=root / "f107_daily.csv",
        )

    def test_hold_never_reaches_past_the_anchor(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            series = self._series(Path(directory))
            # Anchor 08:30. The 06:00-09:00 Kp bin has not closed, so the
            # 03:00-06:00 bin is the newest usable reading. Using the open bin
            # would leak an observation that ends after the anchor.
            row = build_astro_window_features(
                window_start_utc="2026-08-14T00:00:00Z",
                window_end_utc="2026-08-14T08:30:00Z",
                series=series,
            )
        self.assertEqual(row["astro_kp"], "2")
        self.assertEqual(row["astro_kp_age_hours"], "2.5")
        # The window maximum covers only readings that closed inside it.
        self.assertEqual(row["astro_kp_max"], "2")
        self.assertEqual(row["quality_missing_kp"], "0")

    def test_window_aggregate_captures_storm_depth(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            series = self._series(Path(directory))
            row = build_astro_window_features(
                window_start_utc="2026-08-14T00:00:00Z",
                window_end_utc="2026-08-14T08:00:00Z",
                series=series,
            )
        self.assertEqual(row["astro_dst_nt"], "-30")
        self.assertEqual(row["astro_dst_min_nt"], "-45")
        self.assertEqual(row["astro_dst_age_hours"], "0")

    def test_stale_hold_expires_instead_of_becoming_a_constant(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            series = self._series(Path(directory))
            # 30 hours past the last Kp and Dst readings, but inside the
            # 72-hour F10.7 limit.
            row = build_astro_window_features(
                window_start_utc="2026-08-15T00:00:00Z",
                window_end_utc="2026-08-15T15:00:00Z",
                series=series,
            )
        self.assertEqual(row["astro_kp"], "")
        self.assertEqual(row["quality_missing_kp"], "1")
        self.assertEqual(row["astro_dst_nt"], "")
        self.assertEqual(row["quality_missing_dst"], "1")
        self.assertEqual(row["astro_f107"], "110.4")
        self.assertEqual(row["astro_f107_age_hours"], "43")
        self.assertEqual(row["quality_missing_f107"], "0")

    def test_missing_astro_flag_ignores_the_ephemeris(self) -> None:
        """The flag must report observation, not computability.

        The defect it replaces read `quality_missing_astro = 0` on every row
        because a once-pulled monthly constant was on disk.
        """
        row = build_astro_window_features(
            window_start_utc="2026-08-14T00:00:00Z",
            window_end_utc="2026-08-14T08:00:00Z",
            series=load_space_weather_series(),
        )
        self.assertEqual(row["quality_missing_astro"], "1")
        self.assertEqual(row["quality_missing_kp"], "1")
        self.assertEqual(row["quality_missing_dst"], "1")
        self.assertEqual(row["quality_missing_f107"], "1")
        # Ephemeris channels are still populated; they just do not count.
        self.assertTrue(row["astro_moon_phase_angle_deg"])
        self.assertTrue(row["astro_tidal_potential_range"])

    def test_rejects_a_non_positive_window(self) -> None:
        with self.assertRaises(ValueError):
            build_astro_window_features(
                window_start_utc="2026-08-14T08:00:00Z",
                window_end_utc="2026-08-14T08:00:00Z",
                series=load_space_weather_series(),
            )


class ChannelGateTests(unittest.TestCase):
    def _rows(self, values: list[str], *, mask: list[str] | None = None) -> list[dict[str, str]]:
        rows = []
        for index, value in enumerate(values):
            row = {"astro_thing": value, "seismic_event_count": str(index)}
            if mask is not None:
                row["quality_missing_astro"] = mask[index]
            rows.append(row)
        return rows

    def test_constant_channel_is_a_defect(self) -> None:
        rows = self._rows(["125.69"] * 12)
        defects = audit_channels(
            rows,
            channels=["astro_thing"],
            mask_fields=[],
        )
        self.assertEqual([defect.defect for defect in defects], ["constant_channel"])
        with self.assertRaises(ChannelGateError):
            raise_for_defects(defects)

    def test_constant_channel_can_be_allowed_explicitly(self) -> None:
        rows = self._rows(["125.69"] * 12)
        defects = audit_channels(
            rows,
            channels=["astro_thing"],
            mask_fields=[],
            allow_constant=frozenset({"astro_thing"}),
        )
        self.assertEqual(defects, [])

    def test_blank_values_without_a_firing_mask_are_a_defect(self) -> None:
        rows = self._rows(["1", "2", "", "4"], mask=["0", "0", "0", "0"])
        defects = audit_channels(
            rows,
            channels=["astro_thing"],
            mask_fields=["quality_missing_astro"],
        )
        self.assertEqual([defect.defect for defect in defects], ["unmasked_missing_channel"])

    def test_blank_values_covered_by_a_mask_pass(self) -> None:
        rows = self._rows(["1", "2", "", "4"], mask=["0", "0", "1", "0"])
        defects = audit_channels(
            rows,
            channels=["astro_thing"],
            mask_fields=["quality_missing_astro"],
        )
        self.assertEqual(defects, [])

    def test_entirely_empty_channel_is_a_defect(self) -> None:
        rows = self._rows(["", "", ""])
        defects = audit_channels(rows, channels=["astro_thing"], mask_fields=[])
        self.assertEqual([defect.defect for defect in defects], ["empty_channel"])

    def test_channel_selection_skips_masks_and_non_numeric_fields(self) -> None:
        rows = [
            {
                "astro_value": "1.5",
                "astro_source_id": "kyoto_dst_realtime",
                "quality_missing_astro": "0",
                "vlf_count": "3",
            }
        ]
        fieldnames = ["astro_value", "astro_source_id", "quality_missing_astro", "vlf_count"]
        self.assertEqual(
            numeric_channels(rows, fieldnames=fieldnames, prefixes=("astro_", "vlf_")),
            ["astro_value", "vlf_count"],
        )
        self.assertEqual(mask_fields(fieldnames), ["quality_missing_astro"])

    def test_presence_flags_are_not_treated_as_missing_masks(self) -> None:
        """A `..._present` column reads 0 where a channel is blank."""
        self.assertEqual(mask_fields(["quality_fixture_astronomy_present"]), [])
        self.assertEqual(
            mask_fields(["quality_fixture_astronomy_missing"]),
            ["quality_fixture_astronomy_missing"],
        )


def _angular_distance(left: float, right: float) -> float:
    return abs((left - right + 180.0) % 360.0 - 180.0)


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import json
import io
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch
from urllib.parse import parse_qs, urlparse
from urllib.error import URLError

from elfquake.cli import main
from elfquake.connectors.astronomy import _materialize_url, fetch_manifest_json
from elfquake.connectors.ingv import build_event_url, fetch_italy_events
from elfquake.connectors.space_archives import (
    build_kyoto_dst_url,
    build_ncei_goes15_xrs_year_url,
    fetch_gfz_kp_ap,
    fetch_kyoto_dst_month,
    fetch_ncei_goes15_xrs_year,
    fetch_spaceweather_canada_f107_daily,
)
from elfquake.connectors.vlf_cumiana import fetch_manifest_images
from elfquake.features.multimodal_smoke import build_multimodal_smoke_row
from elfquake.http import HttpCapture
from elfquake.normalize.ingv import normalize_ingv_event_text, normalize_row
from elfquake.storage import write_capture


class AcquisitionScaffoldTests(unittest.TestCase):
    def test_ingv_event_url_uses_italy_bounds_and_text_format(self) -> None:
        url = build_event_url("2026-06-22T00:00:00Z", "2026-06-29T23:59:59Z")
        query = parse_qs(urlparse(url).query)

        self.assertEqual(query["format"], ["text"])
        self.assertEqual(query["minlat"], ["35"])
        self.assertEqual(query["maxlat"], ["48"])
        self.assertEqual(query["minlon"], ["6"])
        self.assertEqual(query["maxlon"], ["19"])
        self.assertEqual(query["orderby"], ["time-asc"])
        self.assertEqual(query["starttime"], ["2026-06-22T00:00:00"])
        self.assertEqual(query["endtime"], ["2026-06-29T23:59:59"])

    def test_astronomy_url_materializes_moon_placeholders(self) -> None:
        url = _materialize_url(
            "https://aa.usno.navy.mil/api/moon/phases/date?date=YYYY-MM-DD&nump=N",
            date="2026-06-29",
            moon_phase_count=4,
        )

        self.assertEqual(
            url,
            "https://aa.usno.navy.mil/api/moon/phases/date?date=2026-06-29&nump=4",
        )

    def test_write_capture_writes_payload_and_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            payload_path = Path(directory) / "capture.bin"
            stored = write_capture(
                payload_path,
                b"payload",
                url="https://example.test/data",
                status=200,
                captured_at_utc=datetime(2026, 6, 29, 9, 45, tzinfo=timezone.utc),
                headers={"Content-Type": "application/octet-stream"},
                source_id="example",
            )

            self.assertFalse(stored.skipped_existing)
            self.assertEqual(payload_path.read_bytes(), b"payload")
            metadata = json.loads(stored.metadata_path.read_text(encoding="utf-8"))
            self.assertEqual(metadata["source_id"], "example")
            self.assertEqual(metadata["captured_at_utc"], "2026-06-29T09:45:00Z")

    def test_write_capture_can_skip_existing_payload(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            payload_path = Path(directory) / "capture.bin"
            payload_path.write_bytes(b"original")
            payload_path.with_suffix(".bin.metadata.json").write_text("{}", encoding="utf-8")

            stored = write_capture(
                payload_path,
                b"replacement",
                url="https://example.test/data",
                status=200,
                captured_at_utc=datetime(2026, 6, 29, 9, 45, tzinfo=timezone.utc),
                headers={},
                source_id="example",
            )

            self.assertTrue(stored.skipped_existing)
            self.assertEqual(payload_path.read_bytes(), b"original")

    def test_vlf_manifest_fetch_uses_last_modified_filename(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = root / "manifest.csv"
            manifest.write_text(
                "endpoint_id,url,station,latitude,longitude\n"
                "last_E_VLF,http://example.test/vlf.jpg,cumiana,44.95609,7.42123\n",
                encoding="utf-8",
            )

            stored = fetch_manifest_images(manifest, out_root=root, fetcher=_fake_jpeg_capture)

            self.assertEqual(len(stored), 1)
            self.assertEqual(
                stored[0].payload_path.name,
                "last_E_VLF_2026-06-29T09-45-00Z.jpg",
            )
            self.assertEqual(stored[0].payload_path.read_bytes(), b"jpeg")

    def test_astronomy_manifest_fetch_filters_source_ids(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = root / "manifest.csv"
            manifest.write_text(
                "source_id,url,content,cadence,status,checked_utc\n"
                "moon,https://example.test/moon?date=YYYY-MM-DD&nump=N,moon,event,ok,now\n"
                "kp,https://example.test/kp,kp,1 minute,ok,now\n",
                encoding="utf-8",
            )

            stored = fetch_manifest_json(
                manifest,
                out_root=root,
                date="2026-06-29",
                moon_phase_count=4,
                only={"moon"},
                fetcher=_fake_json_capture,
            )

            self.assertEqual(len(stored), 1)
            self.assertTrue(stored[0].payload_path.name.startswith("moon_"))
            metadata = json.loads(stored[0].metadata_path.read_text(encoding="utf-8"))
            self.assertEqual(metadata["url"], "https://example.test/moon?date=2026-06-29&nump=4")

    def test_ingv_fetch_writes_append_only_timestamped_file(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            stored = fetch_italy_events(
                "2026-06-22T00:00:00Z",
                "2026-06-29T23:59:59Z",
                out_root=Path(directory),
                fetcher=_fake_text_capture,
            )

            self.assertEqual(
                stored.payload_path.name,
                "events_italy_2026-06-22_2026-06-29_2026-06-29T09-58-18Z.txt",
            )
            self.assertEqual(stored.payload_path.read_bytes(), b"#EventID|Time\n")

    def test_cli_returns_nonzero_for_url_errors(self) -> None:
        with patch("sys.argv", ["elfquake", "fetch-astronomy", "--date", "2026-06-29"]):
            with patch("elfquake.cli.fetch_manifest_json", side_effect=URLError("offline")):
                with patch("sys.stderr", new_callable=io.StringIO):
                    self.assertEqual(main(), 2)

    def test_normalize_ingv_row_maps_schema_and_region(self) -> None:
        row = normalize_row(
            {
                "EventID": "46330542",
                "Time": "2026-06-24T23:22:59.040000",
                "Latitude": "42.785",
                "Longitude": "13.1973",
                "Depth/Km": "9.3",
                "MagType": "ML",
                "Magnitude": "2.0",
                "EventLocationName": "8 km W Arquata del Tronto (AP)",
                "EventType": "earthquake",
            },
            raw_file="raw.txt",
            raw_uri="https://example.test",
            ingested_at_utc="2026-06-29T09:58:18Z",
        )

        self.assertEqual(row["event_time_utc"], "2026-06-24T23:22:59.040000Z")
        self.assertEqual(row["italy_region"], "central_italy")
        self.assertEqual(row["magnitude"], "2.0")

    def test_normalize_ingv_event_text_filters_region(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            raw_path = root / "events.txt"
            out_path = root / "central.csv"
            raw_path.write_text(
                "#EventID|Time|Latitude|Longitude|Depth/Km|Author|Catalog|Contributor|ContributorID|MagType|Magnitude|MagAuthor|EventLocationName|EventType\n"
                "1|2026-06-24T23:22:59.040000|42.785|13.1973|9.3|||||ML|2.0|--|Central|earthquake\n"
                "2|2026-06-24T23:22:59.040000|38.0|15.0|9.3|||||ML|2.0|--|Other|earthquake\n",
                encoding="utf-8",
            )
            raw_path.with_suffix(".txt.metadata.json").write_text(
                json.dumps({"url": "https://example.test", "captured_at_utc": "2026-06-29T09:58:18Z"}),
                encoding="utf-8",
            )

            count = normalize_ingv_event_text(raw_path, out_path, only_region="central_italy")

            self.assertEqual(count, 1)
            lines = out_path.read_text(encoding="utf-8").splitlines()
            self.assertEqual(len(lines), 2)
            self.assertIn("central_italy", lines[1])

    def test_archive_url_builders(self) -> None:
        self.assertEqual(
            build_kyoto_dst_url("201601"),
            "https://wdc.kugi.kyoto-u.ac.jp/dst_final/201601/index.html",
        )
        self.assertEqual(
            build_kyoto_dst_url("202601", provisional=True),
            "https://wdc.kugi.kyoto-u.ac.jp/dst_provisional/202601/index.html",
        )
        self.assertEqual(
            build_ncei_goes15_xrs_year_url(2016),
            "https://www.ncei.noaa.gov/instruments/solar-space-observing/particle-detectors/sem/goes/"
            "access/science/xrs/goes15/xrsf-l2-avg1m_science/"
            "sci_xrsf-l2-avg1m_g15_y2016_v2-2-1.nc",
        )

    def test_archive_fetchers_write_expected_extensions(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)

            gfz = fetch_gfz_kp_ap(out_root=root, fetcher=_fake_text_capture)
            f107 = fetch_spaceweather_canada_f107_daily(out_root=root, fetcher=_fake_text_capture)
            dst = fetch_kyoto_dst_month("201601", out_root=root, fetcher=_fake_html_capture)
            goes = fetch_ncei_goes15_xrs_year(2016, out_root=root, fetcher=_fake_netcdf_capture)

            self.assertTrue(gfz.payload_path.name.startswith("gfz_kp_ap_since_1932_"))
            self.assertTrue(f107.payload_path.name.startswith("spaceweather_canada_f107_daily_"))
            self.assertTrue(dst.payload_path.name.startswith("kyoto_dst_final_201601_"))
            self.assertTrue(goes.payload_path.name.startswith("ncei_goes_xrs_g15_avg1m_2016_"))
            self.assertEqual(goes.payload_path.suffix, ".nc")

    def test_multimodal_smoke_row_marks_future_target_unlabeled(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            events = root / "events.csv"
            events.write_text(
                "event_id,event_time_utc,magnitude\n"
                "1,2026-06-28T00:00:00Z,2.5\n"
                "2,2026-06-29T09:00:00Z,3.1\n",
                encoding="utf-8",
            )

            vlf_payload = root / "last_E_VLF_2026-06-29T09-45-00Z.jpg"
            vlf_payload.write_bytes(b"jpeg")
            vlf_metadata = vlf_payload.with_suffix(".jpg.metadata.json")
            vlf_metadata.write_text(
                json.dumps(
                    {
                        "captured_at_utc": "2026-06-29T09:57:24Z",
                        "headers": {"Last-Modified": "Mon, 29 Jun 2026 09:45:00 GMT"},
                        "source_id": "vlf_cumiana_last_E_VLF",
                    }
                ),
                encoding="utf-8",
            )

            moon_payload = root / "usno_moon_phases_2026-06-29T09-56-53Z.json"
            moon_payload.write_text(
                json.dumps(
                    {
                        "phasedata": [
                            {"year": 2026, "month": 6, "day": 29, "time": "23:56", "phase": "Full Moon"}
                        ]
                    }
                ),
                encoding="utf-8",
            )
            moon_metadata = moon_payload.with_suffix(".json.metadata.json")
            moon_metadata.write_text(
                json.dumps({"captured_at_utc": "2026-06-29T09:56:53Z", "source_id": "usno_moon_phases"}),
                encoding="utf-8",
            )

            f107_payload = root / "noaa_solar_cycle_f107_2026-06-29T10-10-17Z.json"
            f107_payload.write_text(json.dumps([{"time-tag": "2026-06", "f10.7": 131.45}]), encoding="utf-8")
            f107_metadata = f107_payload.with_suffix(".json.metadata.json")
            f107_metadata.write_text(
                json.dumps({"captured_at_utc": "2026-06-29T10:10:17Z", "source_id": "noaa_solar_cycle_f107"}),
                encoding="utf-8",
            )

            out = root / "multimodal.csv"
            row = build_multimodal_smoke_row(
                events_csv=events,
                vlf_metadata_paths=[vlf_metadata],
                astronomy_metadata_paths=[moon_metadata, f107_metadata],
                region_id="central_italy",
                window_start_utc="2026-06-22T00:00:00Z",
                window_end_utc="2026-06-29T10:15:00Z",
                target_end_utc="2026-07-06T10:15:00Z",
                out_path=out,
            )

            self.assertEqual(row["target_status"], "unlabeled_pending_future_events")
            self.assertEqual(row["seismic_event_count"], "2")
            self.assertEqual(row["seismic_max_magnitude"], "3.1")
            self.assertEqual(row["vlf_capture_count"], "1")
            self.assertEqual(row["astro_capture_count"], "2")
            self.assertEqual(row["astro_usno_next_phase"], "Full Moon")
            self.assertEqual(row["astro_noaa_solar_cycle_f107_value"], "131.45")
            self.assertTrue(out.exists())


def _fake_jpeg_capture(url: str) -> HttpCapture:
    return HttpCapture(
        url=url,
        status=200,
        captured_at_utc=datetime(2026, 6, 29, 9, 57, 24, tzinfo=timezone.utc),
        headers={"Last-Modified": "Mon, 29 Jun 2026 09:45:00 GMT", "Content-Type": "image/jpeg"},
        body=b"jpeg",
    )


def _fake_json_capture(url: str) -> HttpCapture:
    return HttpCapture(
        url=url,
        status=200,
        captured_at_utc=datetime(2026, 6, 29, 9, 56, 53, tzinfo=timezone.utc),
        headers={"Content-Type": "application/json"},
        body=b"{}",
    )


def _fake_text_capture(url: str) -> HttpCapture:
    return HttpCapture(
        url=url,
        status=200,
        captured_at_utc=datetime(2026, 6, 29, 9, 58, 18, tzinfo=timezone.utc),
        headers={"Content-Type": "text/plain;charset=UTF-8"},
        body=b"#EventID|Time\n",
    )


def _fake_html_capture(url: str) -> HttpCapture:
    return HttpCapture(
        url=url,
        status=200,
        captured_at_utc=datetime(2026, 6, 29, 10, 13, 30, tzinfo=timezone.utc),
        headers={"Content-Type": "text/html"},
        body=b"<html></html>",
    )


def _fake_netcdf_capture(url: str) -> HttpCapture:
    return HttpCapture(
        url=url,
        status=200,
        captured_at_utc=datetime(2026, 6, 29, 10, 14, 0, tzinfo=timezone.utc),
        headers={"Content-Type": "application/x-netcdf"},
        body=b"CDF",
    )


if __name__ == "__main__":
    unittest.main()

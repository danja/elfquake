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

from elfquake.backfill import plan_ingv_backfill
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
from elfquake.connectors.vlf_cumiana import fetch_manifest_images, repeat_manifest_images
from elfquake.features.astronomy import build_astronomy_features
from elfquake.features.design_matrix import build_design_matrix
from elfquake.features.multimodal_design import join_vlf_design_matrix
from elfquake.features.multimodal_smoke import build_multimodal_smoke_row
from elfquake.features.prospective import build_prospective_vlf_windows, update_prospective_vlf_table
from elfquake.features.table import build_multimodal_table_from_manifest
from elfquake.features.targets import label_multimodal_targets
from elfquake.features.training_windows import build_seismic_training_windows
from elfquake.features.vlf import build_vlf_features
from elfquake.features.vlf_image import build_vlf_image_features, extract_vlf_image_features
from elfquake.features.vlf_windows import build_vlf_window_features
from elfquake.models.logistic_smoke import train_logistic_smoke
from elfquake.normalize.events import combine_normalized_events
from elfquake.http import HttpCapture
from elfquake.normalize.ingv import normalize_ingv_event_text, normalize_row
from elfquake.normalize.space_weather import (
    normalize_f107_daily,
    normalize_gfz_kp_ap,
    normalize_goes_xrs_netcdf,
    normalize_kyoto_dst_text,
    write_goes_xrs_netcdf_stub,
)
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

    def test_vlf_repeat_runner_allows_single_cycle_without_sleep(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = root / "manifest.csv"
            manifest.write_text(
                "endpoint_id,url,station,latitude,longitude\n"
                "last_E_VLF,http://example.test/vlf.jpg,cumiana,44.95609,7.42123\n",
                encoding="utf-8",
            )
            slept: list[float] = []

            stored = repeat_manifest_images(
                manifest,
                out_root=root,
                cycles=1,
                interval_seconds=60,
                fetcher=_fake_jpeg_capture,
                sleeper=slept.append,
            )

            self.assertEqual(len(stored), 1)
            self.assertEqual(slept, [])

    def test_vlf_repeat_runner_zero_cycles_means_forever(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = root / "manifest.csv"
            manifest.write_text(
                "endpoint_id,url,station,latitude,longitude\n"
                "last_E_VLF,http://example.test/vlf.jpg,cumiana,44.95609,7.42123\n",
                encoding="utf-8",
            )

            def stop_after_first_sleep(_: float) -> None:
                raise KeyboardInterrupt

            with self.assertRaises(KeyboardInterrupt):
                repeat_manifest_images(
                    manifest,
                    out_root=root,
                    cycles=0,
                    interval_seconds=60,
                    fetcher=_fake_jpeg_capture,
                    sleeper=stop_after_first_sleep,
                )

            captures = list((root / "captures").glob("**/*.jpg"))
            self.assertEqual(len(captures), 1)

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

    def test_ingv_backfill_planner_splits_long_ranges(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            rows = plan_ingv_backfill(
                start_utc="2026-06-01T00:00:00Z",
                end_utc="2026-07-01T00:00:00Z",
                chunk_days=14,
                out_path=Path(directory) / "plan.csv",
            )

            self.assertEqual(len(rows), 3)
            self.assertEqual(rows[0]["window_start_utc"], "2026-06-01T00:00:00Z")
            self.assertEqual(rows[-1]["window_end_utc"], "2026-07-01T00:00:00Z")
            self.assertIn("fetch-ingv-events", rows[0]["command"])

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

    def test_combine_normalized_events_deduplicates_and_sorts(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = root / "first.csv"
            second = root / "second.csv"
            header = (
                "event_id,source,event_time_utc,latitude,longitude,depth_km,magnitude,magnitude_type,"
                "italy_region,event_location_name,event_type,raw_file,ingested_at_utc,raw_uri\n"
            )
            first.write_text(
                header
                + "2,ingv,2026-06-02T00:00:00Z,42,13,8,2.0,ML,central_italy,B,earthquake,a,now,url\n"
                + "1,ingv,2026-06-01T00:00:00Z,42,13,8,2.1,ML,central_italy,A,earthquake,a,now,url\n",
                encoding="utf-8",
            )
            second.write_text(
                header
                + "2,ingv,2026-06-02T00:00:00Z,42,13,8,2.0,ML,central_italy,B duplicate,earthquake,b,now,url\n"
                + "3,ingv,2026-06-03T00:00:00Z,42,13,8,2.2,ML,central_italy,C,earthquake,b,now,url\n",
                encoding="utf-8",
            )

            rows = combine_normalized_events(input_paths=[first, second], out_path=root / "combined.csv")

            self.assertEqual([row["event_id"] for row in rows], ["1", "2", "3"])
            self.assertEqual(rows[1]["event_location_name"], "B")

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
            f107_payload.write_text(json.dumps([{"time-tag": "2026-05", "f10.7": 131.45}]), encoding="utf-8")
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
            self.assertEqual(row["astro_noaa_solar_cycle_f107_month"], "2026-05")
            self.assertEqual(row["astro_noaa_solar_cycle_f107_value"], "131.45")
            self.assertTrue(out.exists())

    def test_vlf_feature_stub_summarizes_capture_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            payload = root / "last_E_VLF_2026-06-29T09-45-00Z.jpg"
            payload.write_bytes(
                b"\xff\xd8"
                b"\xff\xc0\x00\x11\x08\x00\x10\x00\x20"
                b"\x03\x01\x11\x00\x02\x11\x00\x03\x11\x00"
                b"\xff\xd9"
            )
            metadata = payload.with_suffix(".jpg.metadata.json")
            metadata.write_text(
                json.dumps(
                    {
                        "captured_at_utc": "2026-06-29T09:57:24Z",
                        "headers": {"Last-Modified": "Mon, 29 Jun 2026 09:45:00 GMT"},
                        "source_id": "vlf_cumiana_last_E_VLF",
                    }
                ),
                encoding="utf-8",
            )

            row = build_vlf_features(
                metadata_paths=[metadata],
                window_start_utc="2026-06-29T09:00:00Z",
                window_end_utc="2026-06-29T10:00:00Z",
                out_path=root / "vlf.csv",
            )

            self.assertEqual(row["vlf_capture_count"], "1")
            self.assertEqual(row["vlf_jpeg_count"], "1")
            self.assertEqual(row["vlf_latest_width_px"], "32")
            self.assertEqual(row["vlf_latest_height_px"], "16")
            self.assertTrue(row["vlf_latest_entropy_bits_per_byte"])
            self.assertEqual(row["quality_missing_vlf"], "0")

    def test_vlf_window_features_align_service_captures_to_training_windows(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            training = root / "training.csv"
            training.write_text(
                "window_id,window_start_utc,window_end_utc\n"
                "w1,2026-06-29T09:00:00Z,2026-06-29T10:00:00Z\n"
                "w2,2026-06-29T10:00:00Z,2026-06-29T11:00:00Z\n",
                encoding="utf-8",
            )
            payload = root / "captures" / "2026-06-29" / "last_E_VLF_2026-06-29T09-45-00Z.jpg"
            payload.parent.mkdir(parents=True)
            payload.write_bytes(b"\xff\xd8\xff\xd9")
            payload.with_suffix(".jpg.metadata.json").write_text(
                json.dumps(
                    {
                        "captured_at_utc": "2026-06-29T09:57:24Z",
                        "headers": {"Last-Modified": "Mon, 29 Jun 2026 09:45:00 GMT"},
                        "source_id": "vlf_cumiana_last_E_VLF",
                    }
                ),
                encoding="utf-8",
            )

            rows = build_vlf_window_features(
                training_windows_csv=training,
                metadata_root=root / "captures",
                out_path=root / "vlf_windows.csv",
            )

            self.assertEqual(rows[0]["vlf_capture_count"], "1")
            self.assertEqual(rows[0]["quality_missing_vlf"], "0")
            self.assertEqual(rows[1]["vlf_capture_count"], "0")
            self.assertEqual(rows[1]["quality_missing_vlf"], "1")

    def test_vlf_image_features_extract_pixel_summaries(self) -> None:
        from PIL import Image

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image_path = root / "vlf.jpg"
            image = Image.new("RGB", (20, 10), (0, 0, 20))
            pixels = image.load()
            for y in range(10):
                for x in (4, 5, 12):
                    pixels[x, y] = (255, 220, 20)
            image.save(image_path)

            row = extract_vlf_image_features(
                image_path,
                crop_left=0.0,
                crop_top=0.0,
                crop_right=1.0,
                crop_bottom=1.0,
            )
            rows = build_vlf_image_features(
                image_paths=[image_path],
                out_path=root / "features.csv",
                crop_left=0.0,
                crop_top=0.0,
                crop_right=1.0,
                crop_bottom=1.0,
            )

            self.assertEqual(row["vlf_image_width_px"], "20")
            self.assertEqual(row["vlf_crop_width_px"], "20")
            self.assertGreater(float(row["vlf_high_intensity_ratio"]), 0.1)
            self.assertGreaterEqual(int(row["vlf_vertical_streak_count"]), 2)
            self.assertEqual(rows[0]["vlf_image_source_file"], str(image_path))
            self.assertTrue((root / "features.csv").exists())

    def test_astronomy_feature_stub_summarizes_captures(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
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
            f107_payload.write_text(json.dumps([{"time-tag": "2026-05", "f10.7": 125.69}]), encoding="utf-8")
            f107_metadata = f107_payload.with_suffix(".json.metadata.json")
            f107_metadata.write_text(
                json.dumps({"captured_at_utc": "2026-06-29T10:10:17Z", "source_id": "noaa_solar_cycle_f107"}),
                encoding="utf-8",
            )

            row = build_astronomy_features(
                metadata_paths=[moon_metadata, f107_metadata],
                window_start_utc="2026-06-29T09:00:00Z",
                window_end_utc="2026-06-29T10:15:00Z",
                out_path=root / "astro.csv",
            )

            self.assertEqual(row["astro_capture_count"], "2")
            self.assertEqual(row["astro_usno_next_phase"], "Full Moon")
            self.assertEqual(row["astro_noaa_solar_cycle_f107_value"], "125.69")

    def test_space_weather_normalization_stubs_write_csvs(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            gfz = root / "kp.txt"
            gfz.write_text("2026 06 29 03.0 04.50 34423.12500 34423.18750 2.667 12 1\n", encoding="utf-8")
            dst = root / "dst.txt"
            dst.write_text("2026 06 29 " + " ".join(str(value) for value in range(24)) + "\n", encoding="utf-8")
            f107 = root / "f107.txt"
            f107.write_text(
                "fluxdate fluxtime fluxjulian fluxcarrington fluxobsflux fluxadjflux fluxursi\n"
                "20260629 200000 02461234.354 002300.610 000125.7 000123.4 000111.1\n",
                encoding="utf-8",
            )

            self.assertEqual(normalize_gfz_kp_ap(gfz, root / "kp.csv"), 1)
            self.assertEqual(normalize_kyoto_dst_text(dst, root / "dst.csv"), 24)
            self.assertEqual(normalize_f107_daily(f107, root / "f107.csv"), 1)
            self.assertIn("2026-06-29,123.4", (root / "f107.csv").read_text(encoding="utf-8"))

    def test_goes_xrs_netcdf_normalizer_extracts_flux_rows(self) -> None:
        from netCDF4 import Dataset

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            goes = root / "goes.nc"
            with Dataset(goes, "w") as dataset:
                dataset.createDimension("time", 2)
                time_var = dataset.createVariable("time", "f8", ("time",))
                time_var.units = "seconds since 2026-06-29 00:00:00 UTC"
                time_var[:] = [0, 60]
                flux_var = dataset.createVariable("xrs_flux", "f4", ("time",))
                flux_var.units = "W/m^2"
                flux_var[:] = [1.2e-6, 1.4e-6]

            self.assertEqual(
                normalize_goes_xrs_netcdf(
                    goes,
                    root / "goes.csv",
                    max_rows=1,
                    start_utc="2026-06-29T00:01:00Z",
                    end_utc="2026-06-29T00:02:00Z",
                ),
                1,
            )
            self.assertEqual(write_goes_xrs_netcdf_stub(goes, root / "goes_alias.csv"), 2)
            lines = (root / "goes.csv").read_text(encoding="utf-8").splitlines()
            self.assertEqual(lines[0], "time_utc,variable,value,units,source_file")
            self.assertIn("2026-06-29T00:01:00Z,xrs_flux,1.4e-06,W/m^2", lines[1])

    def test_target_labeler_labels_elapsed_windows_only(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rows = root / "rows.csv"
            rows.write_text(
                "window_id,region_id,target_start_utc,target_end_utc,target_magnitude_min,"
                "target_event_count,target_occurred,target_status\n"
                "past,central_italy,2026-06-20T00:00:00Z,2026-06-27T00:00:00Z,3.0,,,\n"
                "future,central_italy,2026-06-29T00:00:00Z,2026-07-06T00:00:00Z,3.0,,,\n",
                encoding="utf-8",
            )
            events = root / "events.csv"
            events.write_text(
                "event_time_utc,magnitude,italy_region\n"
                "2026-06-21T00:00:00Z,3.1,central_italy\n"
                "2026-06-30T00:00:00Z,3.4,central_italy\n",
                encoding="utf-8",
            )

            labeled = label_multimodal_targets(
                input_csv=rows,
                events_csv=events,
                as_of_utc="2026-06-29T00:00:00Z",
                out_path=root / "labeled.csv",
            )

            self.assertEqual(labeled[0]["target_status"], "labeled")
            self.assertEqual(labeled[0]["target_event_count"], "1")
            self.assertEqual(labeled[0]["target_occurred"], "1")
            self.assertEqual(labeled[1]["target_status"], "unlabeled_pending_future_events")
            self.assertEqual(labeled[1]["target_event_count"], "")

    def test_multimodal_table_builder_reads_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            events = root / "events.csv"
            events.write_text("event_id,event_time_utc,magnitude\n1,2026-06-29T09:00:00Z,2.1\n", encoding="utf-8")
            manifest = root / "manifest.csv"
            out = root / "table.csv"
            manifest.write_text(
                "region_id,window_start_utc,window_end_utc,target_end_utc,target_magnitude_min,"
                "events_csv,vlf_metadata_paths,astronomy_metadata_paths\n"
                f"central_italy,2026-06-29T08:00:00Z,2026-06-29T10:00:00Z,"
                f"2026-07-06T10:00:00Z,3.0,{events},,\n",
                encoding="utf-8",
            )

            rows = build_multimodal_table_from_manifest(manifest_path=manifest, out_path=out)

            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["seismic_event_count"], "1")
            self.assertTrue(out.exists())

    def test_build_prospective_vlf_windows_anchors_on_capture_times(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            events = root / "events.csv"
            events.write_text(
                "event_id,event_time_utc,magnitude,italy_region\n"
                "1,2026-06-29T09:30:00Z,2.2,central_italy\n"
                "2,2026-06-28T08:00:00Z,3.1,central_italy\n",
                encoding="utf-8",
            )
            vlf_payload = root / "vlf" / "captures" / "last_E_VLF_2026-06-29T09-45-00Z.jpg"
            vlf_payload.parent.mkdir(parents=True)
            vlf_payload.write_bytes(b"\xff\xd8\xff\xd9")
            vlf_payload.with_suffix(".jpg.metadata.json").write_text(
                json.dumps(
                    {
                        "captured_at_utc": "2026-06-29T09:57:24Z",
                        "headers": {"Last-Modified": "Mon, 29 Jun 2026 09:45:00 GMT"},
                        "source_id": "vlf_cumiana_last_E_VLF",
                    }
                ),
                encoding="utf-8",
            )
            second_vlf_payload = root / "vlf" / "captures" / "last_geomar_2026-06-29T09-45-01Z.jpg"
            second_vlf_payload.write_bytes(b"\xff\xd8\xff\xd9")
            second_vlf_payload.with_suffix(".jpg.metadata.json").write_text(
                json.dumps(
                    {
                        "captured_at_utc": "2026-06-29T09:57:25Z",
                        "headers": {"Last-Modified": "Mon, 29 Jun 2026 09:45:01 GMT"},
                        "source_id": "vlf_cumiana_last_geomar",
                    }
                ),
                encoding="utf-8",
            )
            moon_payload = root / "astro" / "usno_moon_phases_2026-06-29T09-56-53Z.json"
            moon_payload.parent.mkdir(parents=True)
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
            moon_payload.with_suffix(".json.metadata.json").write_text(
                json.dumps({"captured_at_utc": "2026-06-29T09:56:53Z", "source_id": "usno_moon_phases"}),
                encoding="utf-8",
            )

            rows = build_prospective_vlf_windows(
                events_csv=events,
                vlf_metadata_root=root / "vlf",
                astronomy_metadata_root=root / "astro",
                region_id="central_italy",
                lookback_hours=24,
                horizon_days=7,
                out_path=root / "prospective.csv",
            )

            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["target_status"], "unlabeled_pending_future_events")
            self.assertEqual(rows[0]["target_start_utc"], "2026-06-29T09:57:25Z")
            self.assertEqual(rows[0]["target_end_utc"], "2026-07-06T09:57:25Z")
            self.assertEqual(rows[0]["seismic_event_count"], "1")
            self.assertEqual(rows[0]["vlf_capture_count"], "2")
            self.assertEqual(rows[0]["quality_missing_vlf"], "0")

    def test_update_prospective_vlf_table_appends_only_new_windows(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            events = root / "events.csv"
            events.write_text(
                "event_id,event_time_utc,magnitude,italy_region\n"
                "1,2026-06-29T09:30:00Z,2.2,central_italy\n",
                encoding="utf-8",
            )
            vlf_payload = root / "vlf" / "last_E_VLF_2026-06-29T09-45-00Z.jpg"
            vlf_payload.parent.mkdir(parents=True)
            vlf_payload.write_bytes(b"\xff\xd8\xff\xd9")
            vlf_payload.with_suffix(".jpg.metadata.json").write_text(
                json.dumps(
                    {
                        "captured_at_utc": "2026-06-29T09:57:24Z",
                        "headers": {"Last-Modified": "Mon, 29 Jun 2026 09:45:00 GMT"},
                        "source_id": "vlf_cumiana_last_E_VLF",
                    }
                ),
                encoding="utf-8",
            )
            astro_root = root / "astro"
            astro_root.mkdir()
            table = root / "prospective.csv"

            first = update_prospective_vlf_table(
                table_path=table,
                events_csv=events,
                vlf_metadata_root=root / "vlf",
                astronomy_metadata_root=astro_root,
                region_id="central_italy",
                out_path=table,
            )
            second = update_prospective_vlf_table(
                table_path=table,
                events_csv=events,
                vlf_metadata_root=root / "vlf",
                astronomy_metadata_root=astro_root,
                region_id="central_italy",
                out_path=table,
            )

            self.assertEqual(first["new_rows"], 1)
            self.assertEqual(first["total_rows"], 1)
            self.assertEqual(second["new_rows"], 0)
            self.assertEqual(second["total_rows"], 1)

    def test_build_seismic_training_windows_labels_targets(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            events = root / "events.csv"
            events.write_text(
                "event_time_utc,magnitude,italy_region\n"
                "2026-06-01T00:00:00Z,2.0,central_italy\n"
                "2026-06-08T00:00:00Z,3.1,central_italy\n"
                "2026-06-15T00:00:00Z,2.1,central_italy\n",
                encoding="utf-8",
            )

            rows = build_seismic_training_windows(
                events_csv=events,
                region_id="central_italy",
                start_utc="2026-06-01T00:00:00Z",
                end_utc="2026-06-22T00:00:00Z",
                out_path=root / "training.csv",
            )

            self.assertEqual(len(rows), 2)
            self.assertEqual(rows[0]["seismic_event_count"], "1")
            self.assertEqual(rows[0]["target_event_count"], "1")
            self.assertEqual(rows[0]["target_occurred"], "1")
            self.assertEqual(rows[1]["target_occurred"], "0")

    def test_design_matrix_joins_archive_features(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            training = root / "training.csv"
            training.write_text(
                "window_id,region_id,window_start_utc,window_end_utc,target_start_utc,target_end_utc,"
                "target_magnitude_min,seismic_event_count,seismic_max_magnitude,target_event_count,"
                "target_occurred,target_status,source_file\n"
                "w1,central_italy,2026-06-01T00:00:00Z,2026-06-08T00:00:00Z,"
                "2026-06-08T00:00:00Z,2026-06-15T00:00:00Z,3.0,7,2.4,0,0,labeled,events.csv\n",
                encoding="utf-8",
            )
            kp = root / "kp.csv"
            kp.write_text(
                "date,slot,kp,ap,source_file\n"
                "2026-06-01,0,2.0,7,kp.txt\n"
                "2026-06-07,1,3.0,15,kp.txt\n"
                "2026-06-08,1,9.0,99,kp.txt\n",
                encoding="utf-8",
            )
            f107 = root / "f107.csv"
            f107.write_text(
                "date,f107,source_file\n"
                "2026-06-01,100,f107.txt\n"
                "2026-06-07,120,f107.txt\n",
                encoding="utf-8",
            )

            rows = build_design_matrix(
                training_windows_csv=training,
                kp_ap_csv=kp,
                f107_csv=f107,
                out_path=root / "design.csv",
            )

            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["astro_kp_mean"], "2.5")
            self.assertEqual(rows[0]["astro_ap_max"], "15")
            self.assertEqual(rows[0]["astro_f107_mean"], "110")
            self.assertEqual(rows[0]["quality_kp_count"], "2")

    def test_join_vlf_design_matrix_adds_vlf_columns(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            design = root / "design.csv"
            design.write_text(
                "window_id,region_id,seismic_event_count,target_occurred\n"
                "w1,central_italy,2,0\n"
                "w2,central_italy,3,1\n",
                encoding="utf-8",
            )
            vlf = root / "vlf.csv"
            vlf.write_text(
                "window_id,window_start_utc,window_end_utc,vlf_capture_count,quality_missing_vlf\n"
                "w1,2026-06-29T09:00:00Z,2026-06-29T10:00:00Z,1,0\n",
                encoding="utf-8",
            )

            rows = join_vlf_design_matrix(
                design_matrix_csv=design,
                vlf_windows_csv=vlf,
                out_path=root / "joined.csv",
            )

            self.assertEqual(rows[0]["vlf_capture_count"], "1")
            self.assertEqual(rows[0]["quality_missing_vlf"], "0")
            self.assertEqual(rows[1]["vlf_capture_count"], "")

    def test_logistic_smoke_reports_single_class_design_matrix(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            design = root / "design.csv"
            design.write_text(
                "window_id,region_id,seismic_event_count,astro_kp_mean,target_occurred,target_status\n"
                "w1,central_italy,1,2.0,0,labeled\n"
                "w2,central_italy,2,3.0,0,labeled\n",
                encoding="utf-8",
            )

            report = train_logistic_smoke(design_matrix_csv=design, out_path=root / "report.json")

            self.assertEqual(report["status"], "insufficient_class_variation")
            self.assertTrue((root / "report.json").exists())

    def test_logistic_smoke_trains_on_tiny_two_class_design_matrix(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            design = root / "design.csv"
            design.write_text(
                "window_id,region_id,seismic_event_count,astro_kp_mean,target_occurred,target_status\n"
                "w1,central_italy,1,1.0,0,labeled\n"
                "w2,central_italy,2,1.5,0,labeled\n"
                "w3,central_italy,8,6.0,1,labeled\n"
                "w4,central_italy,9,7.0,1,labeled\n",
                encoding="utf-8",
            )

            report = train_logistic_smoke(
                design_matrix_csv=design,
                out_path=root / "report.json",
                epochs=100,
                learning_rate=0.1,
            )

            self.assertEqual(report["status"], "trained_in_sample")
            self.assertEqual(report["positive_count"], 2)
            self.assertIn("seismic_event_count", report["feature_names"])


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

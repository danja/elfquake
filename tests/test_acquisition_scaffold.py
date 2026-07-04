from __future__ import annotations

import json
import io
import importlib.util
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
from elfquake.features.prospective_report import summarize_prospective_table
from elfquake.features.table import build_multimodal_table_from_manifest
from elfquake.features.targets import label_multimodal_targets
from elfquake.features.training_windows import build_seismic_training_windows
from elfquake.features.vlf import build_vlf_features
from elfquake.features.vlf_image import build_vlf_image_features, extract_vlf_image_features
from elfquake.features.vlf_image_windows import join_vlf_image_features_to_windows
from elfquake.features.vlf_windows import build_vlf_window_features
from elfquake.models.ablation_smoke import train_ablation_smoke
from elfquake.models.aligned_windows import build_aligned_window_dataset
from elfquake.models.alignment_manifest import build_alignment_manifest
from elfquake.models.candidates import list_model_candidates, write_model_candidates
from elfquake.models.dataset_combine import combine_aligned_datasets
from elfquake.models.interface_shape import audit_model_interfaces
from elfquake.models.logistic_smoke import train_logistic_smoke
from elfquake.models.readiness import summarize_model_readiness
from elfquake.models.report_summary import summarize_model_run_reports
from elfquake.models.sequence_materializer import materialize_sequence_dataset
from elfquake.models.tensor_materializer import materialize_tensor_dataset
from elfquake.models.tensor_spec import build_tensor_spec
from elfquake.models.temporal_holdout import evaluate_group_holdout, evaluate_temporal_holdout
from elfquake.models.window_adapter import build_event_window_features
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
from elfquake.sim.avalanche_tuning import tune_avalanche_event_extraction
from elfquake.sim.heatmap import render_sandpile_heatmap, render_sandpile_heatmaps_from_manifest
from elfquake.sim.report import benchmark_sandpile_simulation, summarize_sandpile_outputs
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
            image_path = root / "last_E_VLF_2026-06-29T09-45-00Z.jpg"
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
            self.assertEqual(row["vlf_image_captured_at_utc"], "2026-06-29T09:45:00Z")
            self.assertEqual(row["vlf_crop_width_px"], "20")
            self.assertGreater(float(row["vlf_high_intensity_ratio"]), 0.1)
            self.assertGreaterEqual(int(row["vlf_vertical_streak_count"]), 2)
            self.assertEqual(rows[0]["vlf_image_source_file"], str(image_path))
            self.assertTrue((root / "features.csv").exists())

    def test_compare_vlf_image_features_writes_distance_report(self) -> None:
        from PIL import Image
        from elfquake.features.vlf_image_compare import compare_vlf_image_features

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            sim = root / "sim.png"
            real_one = root / "last_E_VLF_one.jpg"
            real_two = root / "last_E_VLF_two.jpg"

            Image.new("RGB", (20, 10), (0, 0, 30)).save(sim)
            Image.new("RGB", (20, 10), (0, 0, 25)).save(real_one)
            bright = Image.new("RGB", (20, 10), (0, 0, 20))
            pixels = bright.load()
            for y in range(10):
                pixels[5, y] = (255, 220, 20)
            bright.save(real_two)

            row = compare_vlf_image_features(
                sim_image=sim,
                real_images=[real_one, real_two],
                out_path=root / "comparison.csv",
                sim_crop_top=0.0,
                real_crop_top=0.0,
            )

            self.assertEqual(row["real_image_count"], "2")
            self.assertTrue(row["nearest_real_image_file"])
            self.assertTrue(row["nearest_real_distance"])
            self.assertTrue((root / "comparison.csv").exists())

    def test_compare_signal_shapes_writes_time_and_frequency_metrics(self) -> None:
        from PIL import Image
        from elfquake.features.signal_shape_compare import (
            compare_signal_shapes,
            event_energy_series,
            sensor_signal_series,
            scan_sensor_signal_shapes,
            vlf_image_column_series,
        )

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            real_events = root / "real_events.csv"
            synthetic_events = root / "synthetic_events.csv"
            real_events.write_text(
                "event_id,event_time_utc,magnitude\n"
                "r1,2026-01-01T00:10:00Z,2.0\n"
                "r2,2026-01-01T01:20:00Z,3.0\n",
                encoding="utf-8",
            )
            synthetic_events.write_text(
                "event_id,event_time_utc,magnitude\n"
                "s1,2026-01-01T00:00:00Z,2.5\n"
                "s2,2026-01-01T02:00:00Z,2.8\n",
                encoding="utf-8",
            )
            piezo = root / "piezo.csv"
            piezo.write_text(
                "step,sensor_id,x,y,piezo_signal\n"
                "0,0,0,0,0.0\n"
                "0,1,1,0,1.0\n"
                "1,0,0,0,2.0\n"
                "1,1,1,0,3.0\n"
                "2,0,0,0,0.5\n",
                encoding="utf-8",
            )
            avalanche = root / "avalanche.csv"
            avalanche.write_text(
                "step,sensor_id,x,y,avalanche_signal\n"
                "0,0,0,0,0.0\n"
                "1,0,0,0,4.0\n"
                "2,0,0,0,1.0\n",
                encoding="utf-8",
            )
            image_path = root / "last_E_VLF_one.jpg"
            image = Image.new("RGB", (12, 6), (0, 0, 20))
            pixels = image.load()
            for y in range(6):
                pixels[3, y] = (255, 220, 20)
                pixels[8, y] = (120, 180, 255)
            image.save(image_path)

            series_rows, pair_rows = compare_signal_shapes(
                series=[
                    event_energy_series(series_id="real_events", events_csv=real_events, bin_seconds=3600),
                    event_energy_series(series_id="synthetic_events", events_csv=synthetic_events, bin_seconds=3600),
                    sensor_signal_series(series_id="piezo", signal_csv=piezo, signal_field="piezo_signal"),
                    sensor_signal_series(series_id="avalanche", signal_csv=avalanche, signal_field="avalanche_signal"),
                    vlf_image_column_series(
                        series_id="real_vlf",
                        image_paths=[image_path],
                        crop_left=0.0,
                        crop_top=0.0,
                        crop_right=1.0,
                        crop_bottom=1.0,
                    ),
                ],
                series_out=root / "series.csv",
                pairs_out=root / "pairs.csv",
            )

            self.assertEqual(len(series_rows), 5)
            self.assertEqual(len(pair_rows), 10)
            self.assertIn("psd_slope", series_rows[0])
            self.assertIn("normalized_distance", pair_rows[0])
            filtered = sensor_signal_series(
                series_id="piezo_sensor_1",
                signal_csv=piezo,
                signal_field="piezo_signal",
                sensor_id=1,
            )
            self.assertEqual(filtered.values.tolist(), [1.0, 3.0])
            scan_rows = scan_sensor_signal_shapes(
                reference=vlf_image_column_series(
                    series_id="real_vlf",
                    image_paths=[image_path],
                    crop_left=0.0,
                    crop_top=0.0,
                    crop_right=1.0,
                    crop_bottom=1.0,
                ),
                signal_csv=piezo,
                signal_field="piezo_signal",
                out_path=root / "sensor_scan.csv",
            )
            self.assertEqual({row["sensor_id"] for row in scan_rows}, {"0", "1"})
            self.assertIn("delta_lag1_autocorrelation", scan_rows[0])
            self.assertTrue((root / "sensor_scan.csv").exists())
            self.assertTrue((root / "series.csv").exists())
            self.assertTrue((root / "pairs.csv").exists())

    def test_join_vlf_image_features_to_windows_aggregates_by_capture_time(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            windows = root / "windows.csv"
            windows.write_text(
                "window_id,window_start_utc,window_end_utc\n"
                "w1,2026-06-29T09:00:00Z,2026-06-29T10:00:00Z\n"
                "w2,2026-06-29T10:00:00Z,2026-06-29T11:00:00Z\n",
                encoding="utf-8",
            )
            image_one = root / "one.jpg"
            image_two = root / "two.jpg"
            image_one.write_bytes(b"jpeg")
            image_two.write_bytes(b"jpeg")
            image_one.with_suffix(".jpg.metadata.json").write_text(
                json.dumps({"captured_at_utc": "2026-06-29T09:30:00Z"}),
                encoding="utf-8",
            )
            image_two.with_suffix(".jpg.metadata.json").write_text(
                json.dumps({"captured_at_utc": "2026-06-29T09:45:00Z"}),
                encoding="utf-8",
            )
            image_features = root / "features.csv"
            image_features.write_text(
                "vlf_image_source_file,vlf_intensity_mean,vlf_high_intensity_ratio,"
                "vlf_hot_color_ratio,vlf_vertical_streak_count,vlf_band_0_mean,vlf_band_1_mean,"
                "vlf_band_2_mean,vlf_band_3_mean,vlf_band_4_mean,vlf_band_5_mean\n"
                f"{image_one},0.2,0.1,0.01,3,0.1,0.2,0.3,0.4,0.5,0.6\n"
                f"{image_two},0.4,0.3,0.02,7,0.2,0.3,0.4,0.5,0.6,0.7\n",
                encoding="utf-8",
            )

            rows = join_vlf_image_features_to_windows(
                windows_csv=windows,
                image_features_csvs=[image_features],
                out_path=root / "joined.csv",
            )

            self.assertEqual(rows[0]["vlf_image_feature_count"], "2")
            self.assertEqual(rows[0]["vlf_image_intensity_mean_avg"], "0.300000")
            self.assertEqual(rows[0]["vlf_image_high_intensity_ratio_max"], "0.300000")
            self.assertEqual(rows[0]["vlf_image_vertical_streak_count_latest"], "7")
            self.assertEqual(rows[1]["vlf_image_feature_count"], "0")
            self.assertEqual(rows[1]["quality_missing_vlf_image_features"], "1")

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

    def test_summarize_prospective_table_reports_readiness_and_coverage(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            table = root / "prospective.csv"
            table.write_text(
                "window_id,region_id,window_end_utc,target_end_utc,target_status,target_occurred,"
                "quality_missing_vlf,quality_missing_vlf_image_features,quality_missing_astro,"
                "vlf_latest_capture_utc,vlf_image_latest_source_file\n"
                "w1,central_italy,2026-06-29T09:00:00Z,2026-07-06T09:00:00Z,"
                "unlabeled_pending_future_events,,0,0,1,2026-06-29T09:00:00Z,img1.jpg\n"
                "w2,central_italy,2026-06-29T10:00:00Z,2026-07-06T10:00:00Z,"
                "unlabeled_pending_future_events,,0,1,0,2026-06-29T10:00:00Z,\n"
                "w3,central_italy,2026-06-20T10:00:00Z,2026-06-27T10:00:00Z,"
                "labeled,1,0,0,0,2026-06-20T10:00:00Z,img3.jpg\n",
                encoding="utf-8",
            )

            report = summarize_prospective_table(
                input_csv=table,
                as_of_utc="2026-07-06T09:30:00Z",
                out_path=root / "summary.json",
            )

            self.assertEqual(report["row_count"], 3)
            self.assertEqual(report["ready_to_label_count"], 1)
            self.assertEqual(report["missing_vlf_image_features_count"], 1)
            self.assertEqual(report["missing_astro_count"], 1)
            self.assertEqual(report["labeled_positive_count"], 1)
            self.assertEqual(report["latest_vlf_image_source_file"], "img1.jpg")

    def test_summarize_sandpile_outputs_reports_shape_and_activity(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            summary = root / "summary.csv"
            sensors = root / "sensors.csv"
            summary.write_text(
                "step,deposition_count,avalanche_count,topple_count,max_height,mean_height,released_mass\n"
                "0,2,0,0,1,0.500000,0\n"
                "1,3,1,4,2,0.750000,1\n",
                encoding="utf-8",
            )
            sensors.write_text(
                "step,sensor_id,x,y,height,local_topple_count\n"
                "0,0,1,1,1,0\n"
                "0,1,2,1,0,0\n"
                "1,0,1,1,0,2\n"
                "1,1,2,1,2,0\n",
                encoding="utf-8",
            )

            report = summarize_sandpile_outputs(
                summary_csv=summary,
                sensors_csv=sensors,
                out_path=root / "report.json",
            )

            self.assertEqual(report["status"], "ok")
            self.assertEqual(report["summary_row_count"], 2)
            self.assertEqual(report["sensor_row_count"], 4)
            self.assertEqual(report["expected_sensor_rows"], 4)
            self.assertEqual(report["total_topple_count"], 4)
            self.assertEqual(report["avalanche_step_count"], 1)
            self.assertEqual(report["max_local_topple_count"], 2)
            self.assertTrue((root / "report.json").exists())

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

    def test_model_readiness_waits_for_labels_and_reports_feature_groups(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            table = root / "prospective.csv"
            table.write_text(
                "window_id,region_id,seismic_event_count,astro_f107_mean,vlf_capture_count,"
                "vlf_image_intensity_mean_latest,target_occurred,target_status\n"
                "w1,central_italy,1,120,2,0.4,,unlabeled_pending_future_events\n"
                "w2,central_italy,2,121,3,0.5,,unlabeled_pending_future_events\n",
                encoding="utf-8",
            )

            report = summarize_model_readiness(input_csv=table, out_path=root / "readiness.json")

            self.assertEqual(report["status"], "waiting_for_labels")
            self.assertEqual(report["row_count"], 2)
            self.assertIn("vlf_image", report["available_feature_groups"])
            self.assertTrue(report["ablation_plan"]["full_multimodal"]["has_required_features"])

    def test_model_readiness_detects_trainable_two_class_table(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            table = root / "design.csv"
            table.write_text(
                "window_id,region_id,seismic_event_count,astro_f107_mean,target_occurred,target_status\n"
                "w1,central_italy,1,120,0,labeled\n"
                "w2,central_italy,3,121,1,labeled\n",
                encoding="utf-8",
            )

            report = summarize_model_readiness(input_csv=table, out_path=root / "readiness.json")

            self.assertEqual(report["status"], "ready_for_smoke_training")
            self.assertEqual(report["positive_count"], 1)
            self.assertEqual(report["negative_count"], 1)
            self.assertFalse(report["ablation_plan"]["seismic_vlf"]["has_required_features"])

    def test_ablation_smoke_trains_available_feature_groups(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            table = root / "design.csv"
            table.write_text(
                "window_id,region_id,seismic_event_count,astro_f107_mean,"
                "vlf_capture_count,vlf_image_intensity_mean_latest,target_occurred,target_status\n"
                "w1,central_italy,1,120,1,0.1,0,labeled\n"
                "w2,central_italy,2,121,2,0.2,0,labeled\n"
                "w3,central_italy,8,130,8,0.8,1,labeled\n"
                "w4,central_italy,9,131,9,0.9,1,labeled\n",
                encoding="utf-8",
            )

            report = train_ablation_smoke(input_csv=table, out_path=root / "ablation.json", epochs=100)

            self.assertEqual(report["status"], "trained_in_sample")
            self.assertEqual(report["ablations"]["seismic_only"]["status"], "trained_in_sample")
            self.assertEqual(report["ablations"]["full_multimodal"]["status"], "trained_in_sample")
            self.assertIn("vlf_image_intensity_mean_latest", report["ablations"]["full_multimodal"]["feature_names"])

    def test_ablation_smoke_reports_single_class(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            table = root / "design.csv"
            table.write_text(
                "window_id,region_id,seismic_event_count,target_occurred,target_status\n"
                "w1,central_italy,1,1,labeled\n"
                "w2,central_italy,2,1,labeled\n",
                encoding="utf-8",
            )

            report = train_ablation_smoke(input_csv=table, out_path=root / "ablation.json")

            self.assertEqual(report["status"], "insufficient_class_variation")
            self.assertEqual(report["ablations"]["seismic_only"]["status"], "insufficient_class_variation")

    def test_temporal_holdout_uses_later_rows_for_test(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            table = root / "aligned.csv"
            table.write_text(
                "window_id,region_id,window_start_utc,synthetic_seismic_event_count,"
                "synthetic_piezo_vlf_signal_mean,target_occurred,target_status\n"
                "w1,r,2026-01-01T00:00:00Z,0,0.1,0,labeled\n"
                "w2,r,2026-01-02T00:00:00Z,1,0.2,0,labeled\n"
                "w3,r,2026-01-03T00:00:00Z,8,0.8,1,labeled\n"
                "w4,r,2026-01-04T00:00:00Z,9,0.9,1,labeled\n"
                "w5,r,2026-01-05T00:00:00Z,10,1.0,1,labeled\n"
                "w6,r,2026-01-06T00:00:00Z,11,1.1,1,labeled\n",
                encoding="utf-8",
            )

            report = evaluate_temporal_holdout(
                input_csv=table,
                out_path=root / "holdout.json",
                train_fraction=0.8,
                epochs=100,
                learning_rate=0.1,
            )

            self.assertEqual(report["status"], "evaluated")
            self.assertEqual(report["train_row_count"], 4)
            self.assertEqual(report["test_time_start"], "2026-01-05T00:00:00Z")
            self.assertEqual(report["evaluations"]["all_features"]["status"], "evaluated")
            self.assertEqual(report["evaluations"]["synthetic_seismic_only"]["status"], "evaluated")
            self.assertEqual(report["evaluations"]["synthetic_seismic_piezo_vlf"]["status"], "evaluated")
            self.assertEqual(report["evaluations"]["all_features"]["test_labels"], [1, 1])

    def test_group_holdout_uses_one_dataset_as_test(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            table = root / "aligned.csv"
            table.write_text(
                "dataset_id,window_id,region_id,window_start_utc,synthetic_seismic_event_count,"
                "synthetic_piezo_vlf_signal_mean,target_occurred,target_status\n"
                "seed1,w1,r,2026-01-01T00:00:00Z,0,0.1,0,labeled\n"
                "seed1,w2,r,2026-01-01T01:00:00Z,1,0.2,0,labeled\n"
                "seed1,w3,r,2026-01-01T02:00:00Z,8,0.8,1,labeled\n"
                "seed2,w1,r,2026-01-01T00:00:00Z,0,0.1,0,labeled\n"
                "seed2,w2,r,2026-01-01T01:00:00Z,9,0.9,1,labeled\n"
                "seed2,w3,r,2026-01-01T02:00:00Z,10,1.0,1,labeled\n",
                encoding="utf-8",
            )

            report = evaluate_group_holdout(
                input_csv=table,
                out_path=root / "holdout.json",
                test_group="seed2",
                epochs=100,
                learning_rate=0.1,
            )

            self.assertEqual(report["schema"], "elfquake.group_holdout.v1")
            self.assertEqual(report["status"], "evaluated")
            self.assertEqual(report["train_groups"], ["seed1"])
            self.assertEqual(report["train_row_count"], 3)
            self.assertEqual(report["test_row_count"], 3)
            self.assertEqual(report["test_positive_count"], 2)
            self.assertNotIn("dataset_id", report["evaluations"]["all_features"]["feature_names"])

    def test_model_run_summary_compacts_evaluation_reports(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            table = root / "aligned.csv"
            table.write_text(
                "dataset_id,window_id,region_id,window_start_utc,synthetic_seismic_event_count,"
                "synthetic_piezo_vlf_signal_mean,target_occurred,target_status\n"
                "seed1,w1,r,2026-01-01T00:00:00Z,0,0.1,0,labeled\n"
                "seed1,w2,r,2026-01-01T01:00:00Z,1,0.2,0,labeled\n"
                "seed1,w3,r,2026-01-01T02:00:00Z,8,0.8,1,labeled\n"
                "seed2,w1,r,2026-01-01T00:00:00Z,0,0.1,0,labeled\n"
                "seed2,w2,r,2026-01-01T01:00:00Z,9,0.9,1,labeled\n"
                "seed2,w3,r,2026-01-01T02:00:00Z,10,1.0,1,labeled\n",
                encoding="utf-8",
            )
            temporal = root / "temporal.json"
            grouped = root / "grouped.json"
            evaluate_temporal_holdout(
                input_csv=table,
                out_path=temporal,
                epochs=100,
                learning_rate=0.1,
            )
            evaluate_group_holdout(
                input_csv=table,
                out_path=grouped,
                test_group="seed2",
                epochs=100,
                learning_rate=0.1,
            )

            summary = summarize_model_run_reports(
                report_paths=[temporal, grouped],
                out_path=root / "summary.json",
            )

            self.assertEqual(summary["schema"], "elfquake.model_run_summary.v1")
            self.assertEqual(summary["report_count"], 2)
            self.assertEqual(summary["reports"][0]["split"]["type"], "temporal")
            self.assertEqual(summary["reports"][1]["split"]["type"], "group")
            self.assertIn("all_features", summary["reports"][0]["evaluations"])

    def test_model_candidate_registry_can_be_filtered_and_written(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)

            rows = write_model_candidates(out_path=root / "candidates.json", stage="transformer")

            self.assertTrue(rows)
            self.assertTrue(all(row["stage"] == "transformer" for row in rows))
            self.assertTrue((root / "candidates.json").exists())
            self.assertIn("patchtst_channel_independent", {row["candidate_id"] for row in list_model_candidates()})

    def test_tensor_spec_groups_numeric_features_and_masks(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            table = root / "design.csv"
            table.write_text(
                "window_id,region_id,window_start_utc,seismic_event_count,astro_f107_mean,"
                "vlf_image_intensity_mean_latest,quality_missing_vlf,target_occurred,target_status\n"
                "w1,central_italy,2026-01-01T00:00:00Z,1,120,0.1,0,0,labeled\n"
                "w2,central_italy,2026-01-02T00:00:00Z,2,121,,1,1,labeled\n",
                encoding="utf-8",
            )

            spec = build_tensor_spec(input_csv=table, out_path=root / "tensor_spec.json")

            self.assertEqual(spec["schema"], "elfquake.tensor_spec.v1")
            self.assertEqual(spec["row_count"], 2)
            self.assertIn("seismic_event_count", spec["modalities"]["seismic"]["feature_fields"])
            self.assertIn("vlf_image_intensity_mean_latest__present", spec["modalities"]["vlf_image"]["mask_fields"])
            self.assertEqual(spec["modalities"]["vlf_image"]["missing_cell_count"], 1)
            self.assertIn("target_occurred", spec["target_fields"])

    def test_tensor_materializer_writes_values_masks_and_index(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            table = root / "design.csv"
            table.write_text(
                "window_id,region_id,window_start_utc,seismic_event_count,"
                "vlf_image_intensity_mean_latest,target_occurred,target_status\n"
                "w1,central_italy,2026-01-01T00:00:00Z,1,0.1,0,labeled\n"
                "w2,central_italy,2026-01-02T00:00:00Z,2,,1,labeled\n",
                encoding="utf-8",
            )
            spec_path = root / "tensor_spec.json"
            build_tensor_spec(input_csv=table, out_path=spec_path)

            manifest = materialize_tensor_dataset(spec_path=spec_path, out_dir=root / "tensor")

            self.assertEqual(manifest["schema"], "elfquake.tensor_dataset.v1")
            self.assertEqual(manifest["row_count"], 2)
            values = (root / "tensor" / "values.csv").read_text(encoding="utf-8").splitlines()
            masks = (root / "tensor" / "masks.csv").read_text(encoding="utf-8").splitlines()
            index = (root / "tensor" / "index.csv").read_text(encoding="utf-8").splitlines()
            self.assertIn("seismic_event_count", values[0])
            self.assertIn("0.000000000", values[2])
            self.assertIn("vlf_image_intensity_mean_latest__present", masks[0])
            self.assertTrue(masks[2].endswith(",0"))
            self.assertIn("window_start_utc", index[0])
            self.assertIn("2026-01-02T00:00:00Z", index[2])

    def test_model_interface_audit_classifies_table_shapes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            events = root / "events.csv"
            events.write_text(
                "event_id,source,event_time_utc,latitude,longitude,depth_km,magnitude,step\n"
                "e1,ingv,2026-01-01T00:00:00Z,42.0,13.0,8.0,2.5,10\n",
                encoding="utf-8",
            )
            piezo = root / "piezo.csv"
            piezo.write_text(
                "step,sensor_id,x,y,piezo_signal,avalanche_signal\n"
                "0,0,1,2,0.1,\n"
                "1,0,1,2,0.2,0.4\n",
                encoding="utf-8",
            )
            vlf = root / "vlf.csv"
            vlf.write_text(
                "vlf_image_source_file,vlf_image_width_px,vlf_intensity_mean\n"
                "capture.jpg,842,0.2\n",
                encoding="utf-8",
            )

            report = audit_model_interfaces(
                input_paths=[events, piezo, vlf],
                out_path=root / "interface_shape.json",
            )

            kinds = {table["path"]: table["shape_kind"] for table in report["tables"]}
            self.assertEqual(kinds[str(events)], "event_list")
            self.assertEqual(kinds[str(piezo)], "sensor_time_series")
            self.assertEqual(kinds[str(vlf)], "image_feature_table")
            self.assertIn(str(events), report["interface_summary"]["needs_window_aggregation"])
            self.assertIn(str(piezo), report["interface_summary"]["needs_sequence_materializer"])
            self.assertEqual(report["tables"][2]["modality_numeric_feature_counts"]["vlf_image"], 2)

    def test_event_window_adapter_aggregates_irregular_events(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            events = root / "events.csv"
            events.write_text(
                "event_id,source,event_time_utc,latitude,longitude,depth_km,magnitude,italy_region\n"
                "e1,ingv,2026-01-01T00:10:00Z,42.0,13.0,8.0,2.0,central_italy\n"
                "e2,ingv,2026-01-01T00:40:00Z,43.0,14.0,10.0,3.0,central_italy\n"
                "e3,ingv,2026-01-01T01:20:00Z,40.0,12.0,5.0,4.0,south_italy\n",
                encoding="utf-8",
            )

            rows = build_event_window_features(
                events_csv=events,
                out_path=root / "windows.csv",
                region_id="central_italy",
                start_utc="2026-01-01T00:00:00Z",
                end_utc="2026-01-01T02:00:00Z",
                window_seconds=3600,
                feature_prefix="seismic",
            )

            self.assertEqual(len(rows), 2)
            self.assertEqual(rows[0]["seismic_event_count"], "2")
            self.assertEqual(rows[0]["seismic_magnitude_max"], "3")
            self.assertEqual(rows[0]["quality_missing_seismic_event_aggregates"], "0")
            self.assertEqual(rows[1]["seismic_event_count"], "0")
            self.assertEqual(rows[1]["seismic_magnitude_max"], "")
            self.assertEqual(rows[1]["quality_missing_seismic_event_aggregates"], "1")

    def test_sequence_materializer_writes_time_entity_channels_and_masks(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            series = root / "piezo.csv"
            series.write_text(
                "step,sensor_id,x,y,piezo_signal,avalanche_signal\n"
                "1,2,1,1,0.2,0.3\n"
                "0,2,1,1,,0.1\n"
                "0,10,2,1,0.4,\n",
                encoding="utf-8",
            )

            manifest = materialize_sequence_dataset(
                input_csv=series,
                out_dir=root / "seq",
                time_start_utc="2026-01-01T00:00:00Z",
                time_step_seconds=60,
            )

            self.assertEqual(manifest["schema"], "elfquake.sequence_dataset.v1")
            self.assertEqual(manifest["time_count"], 2)
            self.assertEqual(manifest["entity_count"], 2)
            self.assertIn("piezo_signal", manifest["channel_fields"])
            values = (root / "seq" / "values.csv").read_text(encoding="utf-8").splitlines()
            masks = (root / "seq" / "masks.csv").read_text(encoding="utf-8").splitlines()
            index = (root / "seq" / "index.csv").read_text(encoding="utf-8").splitlines()
            time_axis = (root / "seq" / "time_axis.csv").read_text(encoding="utf-8").splitlines()
            self.assertIn("time_index,entity_index,piezo_signal,avalanche_signal", values[0])
            self.assertIn("0.000000000", values[1])
            self.assertIn("piezo_signal__present", masks[0])
            self.assertTrue(masks[1].endswith("0,1"))
            self.assertIn("sensor_id", index[0])
            self.assertEqual(time_axis[0], "time_index,step,time_utc")
            self.assertEqual(time_axis[2], "1,1,2026-01-01T00:01:00Z")
            self.assertEqual(manifest["time_mapping"]["utc_axis_field"], "time_utc")
            self.assertNotIn("time_values", manifest)

    def test_alignment_manifest_links_window_and_sequence_manifests(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            tensor_dir = root / "vlf_tensor"
            tensor_dir.mkdir()
            (tensor_dir / "index.csv").write_text(
                "row_index,vlf_image_captured_at_utc,vlf_image_source_file\n"
                "0,2026-06-29T09:45:00Z,data/raw/vlf/cumiana/last_E_VLF_2026-06-29T09-45-00Z.jpg\n",
                encoding="utf-8",
            )
            tensor_manifest = tensor_dir / "manifest.json"
            tensor_manifest.write_text(
                json.dumps(
                    {
                        "schema": "elfquake.tensor_dataset.v1",
                        "layout": "batch,time,channel",
                        "row_count": 1,
                        "feature_count": 2,
                        "values_csv": str(tensor_dir / "values.csv"),
                        "masks_csv": str(tensor_dir / "masks.csv"),
                        "index_csv": str(tensor_dir / "index.csv"),
                        "mask_fields": ["vlf_intensity_mean__present"],
                        "time_field": "vlf_image_captured_at_utc",
                        "modalities": {
                            "vlf_image": {
                                "feature_count": 2,
                                "feature_fields": ["vlf_intensity_mean"],
                            }
                        },
                    }
                ),
                encoding="utf-8",
            )
            sequence_dir = root / "piezo_sequence"
            sequence_dir.mkdir()
            (sequence_dir / "time_axis.csv").write_text(
                "time_index,step,time_utc\n"
                "0,0,2026-01-01T00:00:00Z\n"
                "1,1,2026-01-01T00:01:00Z\n",
                encoding="utf-8",
            )
            (sequence_dir / "entity_axis.csv").write_text("entity_index,sensor_id\n0,0\n", encoding="utf-8")
            sequence_manifest = sequence_dir / "manifest.json"
            sequence_manifest.write_text(
                json.dumps(
                    {
                        "schema": "elfquake.sequence_dataset.v1",
                        "layout": "row,time,entity,channel",
                        "row_count": 2,
                        "time_count": 2,
                        "entity_count": 1,
                        "channel_count": 1,
                        "modality": "synthetic_piezo_vlf",
                        "values_csv": str(sequence_dir / "values.csv"),
                        "masks_csv": str(sequence_dir / "masks.csv"),
                        "index_csv": str(sequence_dir / "index.csv"),
                        "time_axis_csv": str(sequence_dir / "time_axis.csv"),
                        "entity_axis_csv": str(sequence_dir / "entity_axis.csv"),
                        "time_field": "step",
                        "entity_field": "sensor_id",
                        "mask_fields": ["piezo_signal__present"],
                        "time_mapping": {
                            "time_start_utc": "2026-01-01T00:00:00Z",
                            "time_step_seconds": 60,
                            "utc_axis_field": "time_utc",
                            "assumption": "synthetic simulation time mapping",
                        },
                    }
                ),
                encoding="utf-8",
            )

            report = build_alignment_manifest(
                manifest_paths=[tensor_manifest, sequence_manifest],
                out_path=root / "alignment.json",
                run_id="smoke",
            )

            self.assertEqual(report["schema"], "elfquake.alignment_manifest.v1")
            self.assertEqual(report["dataset_count"], 2)
            self.assertEqual(report["ablation_groups"]["vlf_image"], ["vlf_tensor"])
            self.assertEqual(report["ablation_groups"]["synthetic_piezo_vlf"], ["piezo_sequence"])
            self.assertEqual(report["datasets"][0]["time_field"], "vlf_image_captured_at_utc")
            self.assertEqual(report["datasets"][0]["time_coverage"]["start"], "2026-06-29T09:45:00Z")
            self.assertEqual(report["datasets"][1]["time_coverage"]["end"], "2026-01-01T00:01:00Z")

    def test_aligned_window_dataset_aggregates_sequence_and_tensor_inputs(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            base_dir = root / "base"
            base_dir.mkdir()
            (base_dir / "index.csv").write_text(
                "row_index,window_id,region_id,window_start_utc,window_end_utc\n"
                "0,w0,r,2026-01-01T00:00:00Z,2026-01-01T01:00:00Z\n"
                "1,w1,r,2026-01-01T01:00:00Z,2026-01-01T02:00:00Z\n",
                encoding="utf-8",
            )
            (base_dir / "values.csv").write_text(
                "row_index,synthetic_seismic_event_count\n"
                "0,0\n"
                "1,3\n",
                encoding="utf-8",
            )
            base_manifest = base_dir / "manifest.json"
            base_manifest.write_text(
                json.dumps(
                    {
                        "schema": "elfquake.tensor_dataset.v1",
                        "row_count": 2,
                        "feature_count": 1,
                        "values_csv": str(base_dir / "values.csv"),
                        "index_csv": str(base_dir / "index.csv"),
                        "feature_fields": ["synthetic_seismic_event_count"],
                    }
                ),
                encoding="utf-8",
            )
            seq_dir = root / "seq"
            seq_dir.mkdir()
            (seq_dir / "time_axis.csv").write_text(
                "time_index,step,time_utc\n"
                "0,0,2026-01-01T00:10:00Z\n"
                "1,1,2026-01-01T01:10:00Z\n",
                encoding="utf-8",
            )
            (seq_dir / "values.csv").write_text(
                "row_index,time_index,entity_index,piezo_signal\n"
                "0,0,0,2\n"
                "1,1,0,4\n",
                encoding="utf-8",
            )
            sequence_manifest = seq_dir / "manifest.json"
            sequence_manifest.write_text(
                json.dumps(
                    {
                        "schema": "elfquake.sequence_dataset.v1",
                        "values_csv": str(seq_dir / "values.csv"),
                        "time_axis_csv": str(seq_dir / "time_axis.csv"),
                        "channel_fields": ["piezo_signal"],
                        "modality": "synthetic_piezo_vlf",
                    }
                ),
                encoding="utf-8",
            )
            timed_dir = root / "timed"
            timed_dir.mkdir()
            (timed_dir / "index.csv").write_text(
                "row_index,vlf_image_captured_at_utc\n"
                "0,2026-01-01T00:20:00Z\n",
                encoding="utf-8",
            )
            (timed_dir / "values.csv").write_text(
                "row_index,vlf_intensity_mean\n"
                "0,0.5\n",
                encoding="utf-8",
            )
            timed_manifest = timed_dir / "manifest.json"
            timed_manifest.write_text(
                json.dumps(
                    {
                        "schema": "elfquake.tensor_dataset.v1",
                        "values_csv": str(timed_dir / "values.csv"),
                        "index_csv": str(timed_dir / "index.csv"),
                        "time_field": "vlf_image_captured_at_utc",
                        "feature_fields": ["vlf_intensity_mean"],
                        "modalities": {"vlf_image": {"feature_count": 1}},
                    }
                ),
                encoding="utf-8",
            )

            rows = build_aligned_window_dataset(
                base_manifest_path=base_manifest,
                sequence_manifest_paths=[sequence_manifest],
                tensor_manifest_paths=[timed_manifest],
                out_path=root / "aligned.csv",
                target_source_feature="synthetic_seismic_event_count",
            )

            self.assertEqual(len(rows), 2)
            self.assertEqual(rows[0]["synthetic_piezo_vlf_sample_count"], "1")
            self.assertEqual(rows[0]["synthetic_piezo_vlf_piezo_signal_mean"], "2.000000000")
            self.assertEqual(rows[0]["vlf_image_sample_count"], "1")
            self.assertEqual(rows[1]["quality_missing_vlf_image"], "1")
            self.assertEqual(rows[0]["target_occurred"], "1")
            self.assertEqual(rows[1]["target_status"], "unlabeled_no_future_window")

    def test_combine_aligned_datasets_adds_dataset_id(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = root / "seed40.csv"
            second = root / "seed41.csv"
            first.write_text(
                "window_id,target_occurred,synthetic_seismic_event_count\n"
                "w1,0,1\n",
                encoding="utf-8",
            )
            second.write_text(
                "window_id,target_occurred,synthetic_seismic_event_count\n"
                "w2,1,2\n",
                encoding="utf-8",
            )

            rows = combine_aligned_datasets(
                input_csvs=[first, second],
                dataset_ids=["seed40", "seed41"],
                out_path=root / "combined.csv",
            )

            self.assertEqual(len(rows), 2)
            self.assertEqual(rows[0]["dataset_id"], "seed40")
            self.assertEqual(rows[1]["dataset_id"], "seed41")
            self.assertTrue((root / "combined.csv").exists())

    @unittest.skipIf(importlib.util.find_spec("numba") is None, "numba not installed")
    def test_sandpile_simulation_is_deterministic_and_writes_expected_rows(self) -> None:
        from elfquake.sim.sandpile import SandpileConfig, run_sandpile_simulation

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = SandpileConfig(
                width=8,
                height=8,
                steps=5,
                threshold=4,
                source_count=3,
                sensor_count=4,
                deposition_probability=1.0,
                seed=42,
            )

            first_summary, first_sensors = run_sandpile_simulation(
                config=config,
                summary_out=root / "first_summary.csv",
                sensors_out=root / "first_sensors.csv",
            )
            second_summary, second_sensors = run_sandpile_simulation(
                config=config,
                summary_out=root / "second_summary.csv",
                sensors_out=root / "second_sensors.csv",
            )

            self.assertEqual(first_summary, second_summary)
            self.assertEqual(first_sensors, second_sensors)
            self.assertEqual(len(first_summary), 5)
            self.assertEqual(len(first_sensors), 20)
            self.assertEqual(
                (root / "first_summary.csv").read_text(encoding="utf-8").splitlines()[0],
                "step,deposition_count,avalanche_count,topple_count,max_height,mean_height,released_mass,"
                "relaxation_converged,unstable_cell_count,safety_released_mass,target_fill_count,"
                "bottom_layer_removed_mass",
            )

    @unittest.skipIf(importlib.util.find_spec("numba") is None, "numba not installed")
    def test_sandpile_relaxation_drains_unstable_cells_when_sweep_limit_is_hit(self) -> None:
        import numpy as np
        from elfquake.sim.sandpile import _count_unstable, _relax

        grid = np.array([[16, 0], [0, 0]], dtype=np.int64)
        topple_counts = np.zeros_like(grid)

        (
            topple_count,
            released_mass,
            avalanche_count,
            relaxation_converged,
            unstable_cell_count,
            safety_released_mass,
        ) = _relax(grid, topple_counts, 4, 1)

        self.assertEqual(topple_count, 14)
        self.assertEqual(avalanche_count, 1)
        self.assertEqual(relaxation_converged, 0)
        self.assertEqual(unstable_cell_count, 2)
        self.assertEqual(safety_released_mass, 8)
        self.assertEqual(released_mass, 8)
        self.assertEqual(_count_unstable(grid, 4), 0)
        self.assertLess(int(grid.max()), 4)

    @unittest.skipIf(importlib.util.find_spec("numba") is None, "numba not installed")
    def test_sandpile_mountain_mode_refills_target_and_removes_bottom_layer(self) -> None:
        from elfquake.sim.sandpile import SandpileConfig, run_sandpile_simulation

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            summary_rows, _ = run_sandpile_simulation(
                config=SandpileConfig(
                    width=16,
                    height=16,
                    steps=3,
                    threshold=32,
                    source_count=1,
                    sensor_count=2,
                    deposition_probability=0.0,
                    seed=9,
                    deposition_mode="uniform",
                    target_mean_height=8.0,
                    bottom_layer_removal_interval=2,
                ),
                summary_out=root / "summary.csv",
                sensors_out=root / "sensors.csv",
            )

            self.assertEqual(summary_rows[0]["mean_height"], "8.000000")
            self.assertEqual(summary_rows[0]["target_fill_count"], "2048")
            self.assertEqual(summary_rows[0]["max_height"], "8")
            self.assertGreater(int(summary_rows[1]["bottom_layer_removed_mass"]), 0)
            self.assertLess(float(summary_rows[1]["mean_height"]), 8.0)
            self.assertEqual(summary_rows[2]["mean_height"], "8.000000")
            self.assertGreater(int(summary_rows[2]["target_fill_count"]), 0)

    @unittest.skipIf(importlib.util.find_spec("numba") is None, "numba not installed")
    def test_sandpile_mountain_mode_can_limit_target_fill_per_step(self) -> None:
        from elfquake.sim.sandpile import SandpileConfig, run_sandpile_simulation

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            summary_rows, _ = run_sandpile_simulation(
                config=SandpileConfig(
                    width=8,
                    height=8,
                    steps=3,
                    threshold=32,
                    source_count=1,
                    sensor_count=1,
                    deposition_probability=0.0,
                    seed=7,
                    deposition_mode="uniform",
                    target_mean_height=4.0,
                    target_fill_limit=64,
                ),
                summary_out=root / "summary.csv",
                sensors_out=root / "sensors.csv",
            )

            self.assertEqual([row["target_fill_count"] for row in summary_rows], ["64", "64", "64"])
            self.assertEqual([row["mean_height"] for row in summary_rows], [
                "1.000000",
                "2.000000",
                "3.000000",
            ])

    @unittest.skipIf(importlib.util.find_spec("numba") is None, "numba not installed")
    def test_sandpile_can_write_piezo_precursor_and_avalanche_signal_rows(self) -> None:
        from elfquake.sim.avalanche_activity import AVALANCHE_ACTIVITY_FIELDS
        from elfquake.sim.piezo import AVALANCHE_SIGNAL_SENSOR_FIELDS, PIEZO_SENSOR_FIELDS, PiezoConfig
        from elfquake.sim.sandpile import SandpileConfig, run_sandpile_simulation

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            piezo_out = root / "piezo.csv"
            avalanche_signal_out = root / "avalanche_signal.csv"
            avalanche_activity_out = root / "avalanche_activity.csv"

            run_sandpile_simulation(
                config=SandpileConfig(
                    width=8,
                    height=8,
                    steps=8,
                    threshold=4,
                    source_count=4,
                    sensor_count=2,
                    deposition_probability=1.0,
                    seed=11,
                ),
                summary_out=root / "summary.csv",
                sensors_out=root / "sensors.csv",
                piezo_out=piezo_out,
                avalanche_signal_out=avalanche_signal_out,
                avalanche_activity_out=avalanche_activity_out,
                piezo_config=PiezoConfig(
                    sensor_count=3,
                    susceptibility_base=1.0,
                    susceptibility_variation=0.0,
                    cluster_count=0,
                    activation_ratio=0.25,
                    attenuation_radius=8.0,
                ),
            )

            lines = piezo_out.read_text(encoding="utf-8").splitlines()
            self.assertEqual(lines[0], ",".join(PIEZO_SENSOR_FIELDS))
            self.assertEqual(len(lines), 1 + 8 * 3)
            signal_index = PIEZO_SENSOR_FIELDS.index("piezo_signal")
            charge_index = PIEZO_SENSOR_FIELDS.index("piezo_charge_total")
            self.assertGreater(
                max(float(line.split(",")[signal_index]) for line in lines[1:]),
                0.0,
            )
            self.assertGreater(
                max(float(line.split(",")[charge_index]) for line in lines[1:]),
                0.0,
            )
            avalanche_lines = avalanche_signal_out.read_text(encoding="utf-8").splitlines()
            self.assertEqual(avalanche_lines[0], ",".join(AVALANCHE_SIGNAL_SENSOR_FIELDS))
            self.assertEqual(len(avalanche_lines), 1 + 8 * 3)
            avalanche_signal_index = AVALANCHE_SIGNAL_SENSOR_FIELDS.index("avalanche_signal")
            self.assertGreater(
                max(float(line.split(",")[avalanche_signal_index]) for line in avalanche_lines[1:]),
                0.0,
            )
            activity_lines = avalanche_activity_out.read_text(encoding="utf-8").splitlines()
            self.assertEqual(activity_lines[0], ",".join(AVALANCHE_ACTIVITY_FIELDS))
            self.assertEqual(len(activity_lines), 1 + 8)
            self.assertGreater(
                max(int(line.split(",")[AVALANCHE_ACTIVITY_FIELDS.index("topple_count")]) for line in activity_lines[1:]),
                0,
            )

    @unittest.skipIf(importlib.util.find_spec("numba") is None, "numba not installed")
    def test_avalanche_signal_range_is_separate_from_piezo_range(self) -> None:
        from elfquake.sim.piezo import PiezoConfig
        from elfquake.sim.sandpile import SandpileConfig, run_sandpile_simulation

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)

            def run_case(name: str, piezo_radius: float) -> str:
                case_root = root / name
                case_root.mkdir()
                avalanche_signal_out = case_root / "avalanche_signal.csv"
                run_sandpile_simulation(
                    config=SandpileConfig(
                        width=8,
                        height=8,
                        steps=8,
                        threshold=4,
                        source_count=4,
                        sensor_count=2,
                        deposition_probability=1.0,
                        seed=11,
                    ),
                    summary_out=case_root / "summary.csv",
                    sensors_out=case_root / "sensors.csv",
                    piezo_out=case_root / "piezo.csv",
                    avalanche_signal_out=avalanche_signal_out,
                    piezo_config=PiezoConfig(
                        sensor_count=3,
                        susceptibility_base=1.0,
                        susceptibility_variation=0.0,
                        cluster_count=0,
                        activation_ratio=0.25,
                        attenuation_radius=piezo_radius,
                    ),
                    avalanche_signal_config=PiezoConfig(attenuation_radius=0.0, max_distance_radius=0.0),
                )
                return avalanche_signal_out.read_text(encoding="utf-8")

            self.assertEqual(run_case("narrow_piezo", 2.0), run_case("wide_piezo", 8.0))

    def test_build_synthetic_event_list_writes_ingv_like_rows(self) -> None:
        from elfquake.sim.synthetic_events import SYNTHETIC_EVENT_FIELDS, build_synthetic_event_list

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            summary = root / "summary.csv"
            sensors = root / "sensors.csv"
            summary.write_text(
                "step,deposition_count,avalanche_count,topple_count,max_height,mean_height,released_mass,"
                "relaxation_converged,unstable_cell_count,safety_released_mass,target_fill_count,"
                "bottom_layer_removed_mass\n"
                "0,1,0,0,2,1.0,0,1,0,0,0,0\n"
                "1,1,1,10,3,1.2,0,1,0,0,0,0\n"
                "2,1,1,100,4,1.4,0,1,0,0,0,0\n",
                encoding="utf-8",
            )
            sensors.write_text(
                "step,sensor_id,x,y,height,local_topple_count\n"
                "1,0,2,3,5,0\n"
                "1,1,7,4,6,4\n"
                "2,0,1,1,2,0\n"
                "2,1,8,8,9,0\n",
                encoding="utf-8",
            )

            rows = build_synthetic_event_list(
                summary_csv=summary,
                sensors_csv=sensors,
                out_path=root / "events.csv",
                grid_width=10,
                grid_height=10,
                start_time_utc="2026-01-01T00:00:00Z",
                step_seconds=30,
            )

            self.assertEqual(len(rows), 2)
            self.assertEqual((root / "events.csv").read_text(encoding="utf-8").splitlines()[0], ",".join(SYNTHETIC_EVENT_FIELDS))
            self.assertEqual(rows[0]["event_id"], "synthetic_sandpile_step_000001")
            self.assertEqual(rows[0]["event_time_utc"], "2026-01-01T00:00:30Z")
            self.assertEqual(rows[0]["italy_region"], "central_italy")
            self.assertEqual(rows[0]["location_quality"], "topple_sensor")
            self.assertEqual(rows[1]["location_quality"], "height_proxy")
            self.assertGreater(float(rows[1]["magnitude"]), float(rows[0]["magnitude"]))
            self.assertTrue(41.5 <= float(rows[0]["latitude"]) <= 43.5)
            self.assertTrue(12.0 <= float(rows[0]["longitude"]) <= 14.5)

    def test_build_synthetic_event_list_allows_empty_outputs(self) -> None:
        from elfquake.sim.synthetic_events import SYNTHETIC_EVENT_FIELDS, build_synthetic_event_list

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            summary = root / "summary.csv"
            sensors = root / "sensors.csv"
            summary.write_text(
                "step,deposition_count,avalanche_count,topple_count,max_height,mean_height,released_mass,"
                "relaxation_converged,unstable_cell_count,safety_released_mass,target_fill_count,"
                "bottom_layer_removed_mass\n"
                "0,1,0,0,2,1.0,0,1,0,0,0,0\n",
                encoding="utf-8",
            )
            sensors.write_text("step,sensor_id,x,y,height,local_topple_count\n", encoding="utf-8")

            rows = build_synthetic_event_list(
                summary_csv=summary,
                sensors_csv=sensors,
                out_path=root / "events.csv",
                grid_width=10,
                grid_height=10,
            )

            self.assertEqual(rows, [])
            self.assertEqual((root / "events.csv").read_text(encoding="utf-8").strip(), ",".join(SYNTHETIC_EVENT_FIELDS))

    def test_build_avalanche_signal_event_list_writes_ingv_like_rows(self) -> None:
        from elfquake.sim.synthetic_events import (
            AVALANCHE_SIGNAL_EVENT_FIELDS,
            build_avalanche_signal_event_list,
        )

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            avalanche = root / "avalanche.csv"
            avalanche.write_text(
                "step,sensor_id,x,y,avalanche_signal,avalanche_total_source,active_topple_cell_count,"
                "max_local_topple,nearest_topple_distance,stress_drop_total,stress_drop_max,avalanche_release_total\n"
                "0,0,2,3,0.0,0.0,0,0,,0.0,0.0,0.0\n"
                "1,0,2,3,4.0,10.0,5,2,1.0,3.0,2.0,10.0\n"
                "1,1,4,5,8.0,10.0,5,3,0.5,3.0,2.0,10.0\n",
                encoding="utf-8",
            )

            rows = build_avalanche_signal_event_list(
                avalanche_csv=avalanche,
                out_path=root / "events.csv",
                grid_width=10,
                grid_height=10,
                step_seconds=30,
                min_signal=1.0,
            )

            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["event_id"], "synthetic_avalanche_signal_step_000001")
            self.assertEqual(rows[0]["event_time_utc"], "2026-01-01T00:00:30Z")
            self.assertEqual(rows[0]["x"], "4")
            self.assertEqual(rows[0]["location_quality"], "avalanche_signal_sensor")
            self.assertEqual(rows[0]["avalanche_signal"], "8.000000000")
            self.assertEqual((root / "events.csv").read_text(encoding="utf-8").splitlines()[0], ",".join(AVALANCHE_SIGNAL_EVENT_FIELDS))

    def test_build_avalanche_signal_event_list_prefers_activity_centroid_location(self) -> None:
        from elfquake.sim.synthetic_events import build_avalanche_signal_event_list

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            avalanche = root / "avalanche.csv"
            activity = root / "activity.csv"
            avalanche.write_text(
                "step,sensor_id,x,y,avalanche_signal,avalanche_total_source,active_topple_cell_count,"
                "max_local_topple,nearest_topple_distance,stress_drop_total,stress_drop_max,avalanche_release_total\n"
                "1,0,2,3,8.0,10.0,5,3,0.5,3.0,2.0,10.0\n",
                encoding="utf-8",
            )
            activity.write_text(
                "step,active_topple_cell_count,topple_count,centroid_x,centroid_y,weighted_centroid_x,"
                "weighted_centroid_y,min_x,max_x,min_y,max_y,peak_x,peak_y,peak_topple_count\n"
                "1,3,12,5.0,6.0,6.5,7.25,4,8,6,9,7,8,5\n",
                encoding="utf-8",
            )

            rows = build_avalanche_signal_event_list(
                avalanche_csv=avalanche,
                avalanche_activity_csv=activity,
                out_path=root / "events.csv",
                grid_width=10,
                grid_height=10,
                min_signal=1.0,
            )

            self.assertEqual(rows[0]["x"], "6.500")
            self.assertEqual(rows[0]["y"], "7.250")
            self.assertEqual(rows[0]["topple_count"], "12")
            self.assertEqual(rows[0]["location_quality"], "avalanche_activity_weighted_centroid")

    def test_build_avalanche_signal_event_list_can_select_sparse_local_peaks(self) -> None:
        from elfquake.sim.synthetic_events import build_avalanche_signal_event_list

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            avalanche = root / "avalanche.csv"
            avalanche.write_text(
                "step,sensor_id,x,y,avalanche_signal,avalanche_total_source,active_topple_cell_count,"
                "max_local_topple,nearest_topple_distance,stress_drop_total,stress_drop_max,avalanche_release_total\n"
                "0,0,1,1,1.0,1.0,1,1,1.0,1.0,1.0,1.0\n"
                "1,0,2,1,5.0,5.0,2,1,1.0,1.0,1.0,5.0\n"
                "2,0,3,1,4.0,4.0,2,1,1.0,1.0,1.0,4.0\n"
                "3,0,4,1,9.0,9.0,3,2,1.0,2.0,2.0,9.0\n"
                "4,0,5,1,2.0,2.0,1,1,1.0,1.0,1.0,2.0\n",
                encoding="utf-8",
            )

            rows = build_avalanche_signal_event_list(
                avalanche_csv=avalanche,
                out_path=root / "events.csv",
                grid_width=10,
                grid_height=10,
                min_signal_quantile=0.50,
                local_max_window=1,
            )

            self.assertEqual([row["step"] for row in rows], ["1", "3"])

    def test_tune_avalanche_event_extraction_writes_ranked_grid(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            real_events = root / "real.csv"
            real_events.write_text(
                "event_id,event_time_utc,latitude,longitude,depth_km,magnitude,italy_region\n"
                "r1,2026-01-01T01:00:00Z,42,13,10,2.0,central_italy\n"
                "r2,2026-01-01T03:00:00Z,42,13,10,3.0,central_italy\n",
                encoding="utf-8",
            )
            avalanche = root / "avalanche.csv"
            avalanche.write_text(
                "step,sensor_id,x,y,avalanche_signal,avalanche_total_source\n"
                "0,0,1,1,0.0,0.0\n"
                "60,0,2,2,2.0,2.0\n"
                "120,0,3,3,0.5,0.5\n"
                "180,0,4,4,4.0,4.0\n"
                "240,0,5,5,0.1,0.1\n",
                encoding="utf-8",
            )

            rows = tune_avalanche_event_extraction(
                real_events_csv=real_events,
                avalanche_csv=avalanche,
                out_path=root / "tuning.csv",
                work_dir=root / "events",
                grid_width=8,
                grid_height=8,
                quantiles=[0.0, 0.5],
                local_max_windows=[0, 1],
                event_bin_seconds=3600,
            )

            self.assertEqual(len(rows), 4)
            self.assertEqual(rows[0]["rank"], "1")
            self.assertTrue((root / "tuning.csv").exists())
            self.assertTrue(Path(rows[0]["events_file"]).exists())
            self.assertIn("normalized_distance", rows[0])

    def test_render_piezo_spectrogram_writes_png_and_metadata(self) -> None:
        from elfquake.sim.piezo_spectrogram import render_piezo_spectrogram

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            piezo = root / "piezo.csv"
            piezo.write_text(
                "step,sensor_id,x,y,piezo_signal,piezo_total_source,near_critical_cell_count,"
                "critical_cell_count,nearest_critical_distance,max_stress_ratio\n"
                "0,0,0,0,0.0,0.0,0,0,,0.0\n"
                "1,0,0,0,2.0,2.0,2,0,1.0,0.8\n"
                "1,1,1,0,3.0,3.0,2,0,1.0,0.9\n",
                encoding="utf-8",
            )

            report = render_piezo_spectrogram(
                piezo_csv=piezo,
                out_path=root / "spectrogram.png",
                metadata_out=root / "spectrogram.json",
                step_seconds=2,
                freq_bins=8,
                window_steps=8,
                scale=2,
            )

            self.assertEqual((root / "spectrogram.png").read_bytes()[:8], b"\x89PNG\r\n\x1a\n")
            metadata = json.loads((root / "spectrogram.json").read_text(encoding="utf-8"))
            self.assertEqual(metadata["step_count"], "2")
            self.assertEqual(metadata["frequency_axis"], "fft_from_step_seconds")
            self.assertEqual(metadata["nyquist_hz"], "0.250000000000")
            self.assertEqual(metadata["freq_bins"], "8")
            self.assertEqual(report["width_px"], "4")
            self.assertEqual(report["height_px"], "16")
            self.assertGreater(float(report["max_power"]), 0.0)

    def test_render_piezo_spectrogram_allows_zero_signal(self) -> None:
        from elfquake.sim.piezo_spectrogram import render_piezo_spectrogram

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            piezo = root / "piezo.csv"
            piezo.write_text(
                "step,sensor_id,x,y,piezo_signal,piezo_total_source,near_critical_cell_count,"
                "critical_cell_count,nearest_critical_distance,max_stress_ratio\n"
                "0,0,0,0,0.0,0.0,0,0,,0.0\n",
                encoding="utf-8",
            )

            report = render_piezo_spectrogram(
                piezo_csv=piezo,
                out_path=root / "spectrogram.png",
                freq_bins=4,
                window_steps=4,
            )

            self.assertEqual((root / "spectrogram.png").read_bytes()[:8], b"\x89PNG\r\n\x1a\n")
            self.assertEqual(report["max_power"], "0.000000000")

    def test_render_piezo_summary_writes_timeseries_and_spectrogram_png(self) -> None:
        from elfquake.sim.piezo_spectrogram import render_piezo_timeseries_spectrogram

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            piezo = root / "piezo.csv"
            piezo.write_text(
                "step,sensor_id,x,y,piezo_signal,piezo_total_source,near_critical_cell_count,"
                "critical_cell_count,nearest_critical_distance,max_stress_ratio\n"
                "0,0,0,0,0.0,0.0,0,0,,0.0\n"
                "1,0,0,0,2.0,2.0,2,0,1.0,0.8\n"
                "2,0,0,0,0.5,0.5,1,0,1.0,0.7\n",
                encoding="utf-8",
            )

            report = render_piezo_timeseries_spectrogram(
                piezo_csv=piezo,
                out_path=root / "summary.png",
                metadata_out=root / "summary.json",
                step_seconds=2,
                freq_bins=8,
                window_steps=8,
                scale=2,
                timeseries_height=24,
                output_width=2,
                sensor_id=0,
                dc_block=0.95,
            )

            self.assertEqual((root / "summary.png").read_bytes()[:8], b"\x89PNG\r\n\x1a\n")
            metadata = json.loads((root / "summary.json").read_text(encoding="utf-8"))
            self.assertEqual(metadata["plot_type"], "timeseries_plus_fft_spectrogram")
            self.assertEqual(metadata["timeseries_height_px"], "48")
            self.assertEqual(metadata["selected_sensor_id"], "0")
            self.assertEqual(metadata["dc_block"], "0.95")
            self.assertEqual(metadata["display_sample_count"], "2")
            self.assertEqual(report["width_px"], "4")
            self.assertEqual(report["height_px"], "68")

    def test_render_piezo_audio_writes_wav_sonification(self) -> None:
        import wave
        from elfquake.sim.piezo_spectrogram import render_piezo_audio

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            piezo = root / "piezo.csv"
            piezo.write_text(
                "step,sensor_id,x,y,piezo_signal,piezo_total_source,near_critical_cell_count,"
                "critical_cell_count,nearest_critical_distance,max_stress_ratio\n"
                "0,0,0,0,0.0,0.0,0,0,,0.0\n"
                "1,0,0,0,2.0,2.0,2,0,1.0,0.8\n"
                "2,0,0,0,0.5,0.5,1,0,1.0,0.7\n",
                encoding="utf-8",
            )

            report = render_piezo_audio(
                piezo_csv=piezo,
                out_path=root / "piezo.wav",
                sample_rate=8000,
                duration_seconds=1.0,
                smooth_steps=2,
                sensor_id=0,
                dc_block=0.95,
            )

            with wave.open(str(root / "piezo.wav"), "rb") as handle:
                self.assertEqual(handle.getnchannels(), 1)
                self.assertEqual(handle.getsampwidth(), 2)
                self.assertEqual(handle.getframerate(), 8000)
                self.assertEqual(handle.getnframes(), 8000)
            self.assertEqual(report["audio_type"], "sonified_sum_piezo_signal_by_step")
            self.assertEqual(report["audio_sample_count"], "8000")
            self.assertEqual(report["smooth_steps"], "2")
            self.assertEqual(report["selected_sensor_id"], "0")
            self.assertEqual(report["dc_block"], "0.95")

    def test_render_piezo_vlf_summary_writes_analogue_png(self) -> None:
        from elfquake.sim.piezo_spectrogram import render_piezo_strain_vlf_summary

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            piezo = root / "piezo.csv"
            piezo.write_text(
                "step,sensor_id,x,y,piezo_signal,piezo_total_source,near_critical_cell_count,"
                "critical_cell_count,nearest_critical_distance,max_stress_ratio\n"
                "0,0,0,0,0.0,0.0,0,0,,0.0\n"
                "1,0,0,0,2.0,2.0,2,0,1.0,0.8\n"
                "2,0,0,0,0.5,0.5,1,0,1.0,0.7\n",
                encoding="utf-8",
            )

            report = render_piezo_strain_vlf_summary(
                piezo_csv=piezo,
                out_path=root / "vlf.png",
                metadata_out=root / "vlf.json",
                freq_bins=16,
                scale=2,
                timeseries_height=24,
                output_width=3,
                sensor_id=0,
            )

            self.assertEqual((root / "vlf.png").read_bytes()[:8], b"\x89PNG\r\n\x1a\n")
            metadata = json.loads((root / "vlf.json").read_text(encoding="utf-8"))
            self.assertEqual(metadata["plot_type"], "strain_envelope_vlf_analogue")
            self.assertEqual(metadata["frequency_axis"], "analogue_vlf_carrier_hz")
            self.assertEqual(metadata["selected_sensor_id"], "0")
            self.assertEqual(metadata["display_color_quantile"], "0.820000")
            self.assertEqual(report["width_px"], "6")

    @unittest.skipIf(importlib.util.find_spec("matplotlib") is None, "matplotlib not installed")
    def test_render_event_map_writes_png_and_metadata(self) -> None:
        from elfquake.visualization.event_map import render_event_map

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            events = root / "events.csv"
            events.write_text(
                "event_id,source,event_time_utc,latitude,longitude,depth_km,magnitude,magnitude_type,"
                "italy_region,event_location_name,event_type,raw_file,ingested_at_utc,raw_uri\n"
                "a,ingv,2026-01-01T00:00:00Z,42.5,13.1,8,2.4,ML,central,Apennines,earthquake,,,\n"
                "b,ingv,2026-01-01T01:00:00Z,38.1,15.0,12,3.1,ML,sicily,Sicily,earthquake,,,\n"
                "bad,ingv,2026-01-01T02:00:00Z,,,,,,,,,,,\n",
                encoding="utf-8",
            )

            report = render_event_map(
                events_csv=events,
                out_path=root / "map.png",
                metadata_out=root / "map.json",
                min_magnitude=2.0,
                basemap_geojson=None,
            )

            self.assertEqual((root / "map.png").read_bytes()[:8], b"\x89PNG\r\n\x1a\n")
            metadata = json.loads((root / "map.json").read_text(encoding="utf-8"))
            self.assertEqual(metadata["event_count"], "2")
            self.assertEqual(metadata["map_type"], "offline_schematic_italy")
            self.assertEqual(report["event_count"], "2")

    @unittest.skipIf(importlib.util.find_spec("matplotlib") is None, "matplotlib not installed")
    def test_render_event_map_can_use_geojson_line_basemap(self) -> None:
        from elfquake.visualization.event_map import render_event_map

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            events = root / "events.csv"
            events.write_text(
                "event_id,source,event_time_utc,latitude,longitude,depth_km,magnitude,magnitude_type,"
                "italy_region,event_location_name,event_type,raw_file,ingested_at_utc,raw_uri\n"
                "a,synthetic_avalanche,2026-01-01T00:00:00Z,42.0,12.5,8,3.4,ML,central,Apennines,earthquake,,,\n",
                encoding="utf-8",
            )
            basemap = root / "italy.geojson"
            basemap.write_text(
                json.dumps(
                    {
                        "type": "FeatureCollection",
                        "features": [
                            {
                                "type": "Feature",
                                "properties": {"ADMIN": "Italy"},
                                "geometry": {
                                    "type": "Polygon",
                                    "coordinates": [[
                                        [11.5, 41.5],
                                        [13.5, 41.5],
                                        [13.5, 42.5],
                                        [11.5, 42.5],
                                        [11.5, 41.5],
                                    ]],
                                },
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )

            report = render_event_map(
                events_csv=events,
                out_path=root / "map.png",
                metadata_out=root / "map.json",
                basemap_geojson=basemap,
            )

            metadata = json.loads((root / "map.json").read_text(encoding="utf-8"))
            self.assertEqual(metadata["map_type"], "natural_earth_line_italy")
            self.assertEqual(metadata["basemap_geojson"], str(basemap))
            self.assertEqual(report["event_count"], "1")

    @unittest.skipIf(importlib.util.find_spec("numba") is None, "numba not installed")
    def test_sandpile_simulation_writes_grid_snapshots(self) -> None:
        from elfquake.sim.sandpile import SandpileConfig, run_sandpile_simulation

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            progress: list[tuple[int, int, str]] = []
            run_sandpile_simulation(
                config=SandpileConfig(
                    width=8,
                    height=6,
                    steps=5,
                    source_count=2,
                    sensor_count=2,
                    deposition_probability=1.0,
                    seed=3,
                ),
                summary_out=root / "summary.csv",
                sensors_out=root / "sensors.csv",
                snapshot_dir=root / "snapshots",
                snapshot_interval=2,
                progress_interval=2,
                progress_callback=lambda completed, total, row: progress.append(
                    (completed, total, row["max_height"])
                ),
            )

            snapshots = sorted((root / "snapshots").glob("sandpile_step_*.npy"))
            manifest = (root / "snapshots" / "manifest.csv").read_text(encoding="utf-8").splitlines()
            self.assertEqual([path.name for path in snapshots], [
                "sandpile_step_000000.npy",
                "sandpile_step_000002.npy",
                "sandpile_step_000004.npy",
            ])
            self.assertEqual(len(manifest), 4)
            self.assertEqual([item[0] for item in progress], [2, 4, 5])
            self.assertEqual({item[1] for item in progress}, {5})

    def test_render_sandpile_heatmap_writes_png(self) -> None:
        import numpy as np

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            snapshot = root / "snapshot.npy"
            np.save(snapshot, np.array([[0, 1], [2, 4]], dtype=np.int64))

            report = render_sandpile_heatmap(
                snapshot_path=snapshot,
                out_path=root / "heatmap.png",
                scale=3,
                color_max=4,
            )

            self.assertEqual(report["width_px"], "6")
            self.assertEqual(report["height_px"], "6")
            self.assertEqual(report["max_height"], "4")
            self.assertEqual(report["color_min"], "0.0")
            self.assertEqual(report["color_max"], "4")
            self.assertEqual(report["gamma"], "1.0")
            self.assertEqual((root / "heatmap.png").read_bytes()[:8], b"\x89PNG\r\n\x1a\n")

    def test_render_sandpile_heatmap_can_use_fixed_color_scale(self) -> None:
        import numpy as np
        from PIL import Image

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            snapshot = root / "snapshot.npy"
            np.save(snapshot, np.array([[0, 2]], dtype=np.int64))

            render_sandpile_heatmap(snapshot_path=snapshot, out_path=root / "auto.png", scale=1)
            render_sandpile_heatmap(
                snapshot_path=snapshot,
                out_path=root / "fixed.png",
                scale=1,
                color_max=4,
            )

            with Image.open(root / "auto.png") as auto_image:
                auto_pixel = auto_image.convert("RGB").getpixel((1, 0))
            with Image.open(root / "fixed.png") as fixed_image:
                fixed_pixel = fixed_image.convert("RGB").getpixel((1, 0))

            self.assertNotEqual(auto_pixel, fixed_pixel)

    def test_render_sandpile_heatmaps_from_manifest_writes_all_pngs(self) -> None:
        import numpy as np

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            snapshots = root / "snapshots"
            snapshots.mkdir()
            first = snapshots / "sandpile_step_000000.npy"
            second = snapshots / "sandpile_step_000100.npy"
            np.save(first, np.array([[0, 1]], dtype=np.int64))
            np.save(second, np.array([[2, 3]], dtype=np.int64))
            manifest = snapshots / "manifest.csv"
            manifest.write_text(
                "step,snapshot_file\n"
                f"0,{first}\n"
                f"100,{second}\n",
                encoding="utf-8",
            )
            progress = []

            rows = render_sandpile_heatmaps_from_manifest(
                manifest_path=manifest,
                out_dir=root / "heatmaps",
                scale=2,
                color_min=0,
                color_max=4,
                gamma=0.85,
                workers=2,
                progress_interval=1,
                progress_callback=lambda completed, total, _row: progress.append((completed, total)),
            )

            self.assertEqual(len(rows), 2)
            self.assertEqual(rows[0]["color_max"], "4")
            self.assertEqual(rows[0]["gamma"], "0.85")
            self.assertEqual(sorted(progress), [(1, 2), (2, 2)])
            self.assertTrue((root / "heatmaps" / "sandpile_step_000000.png").exists())
            self.assertTrue((root / "heatmaps" / "sandpile_step_000100.png").exists())

    @unittest.skipIf(importlib.util.find_spec("numba") is None, "numba not installed")
    def test_sandpile_benchmark_reports_cpu_backend(self) -> None:
        from elfquake.sim.sandpile import SandpileConfig

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            report = benchmark_sandpile_simulation(
                config=SandpileConfig(
                    width=8,
                    height=8,
                    steps=3,
                    source_count=2,
                    sensor_count=2,
                    deposition_probability=1.0,
                    seed=7,
                ),
                out_path=root / "benchmark.json",
            )

            self.assertEqual(report["status"], "ok")
            self.assertEqual(report["backend"], "numba_cpu")
            self.assertFalse(report["gpu_required"])
            self.assertEqual(report["summary_row_count"], 3)
            self.assertEqual(report["sensor_row_count"], 6)
            self.assertTrue((root / "benchmark.json").exists())


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

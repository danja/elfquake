from __future__ import annotations

import json
import io
import importlib.util
import csv
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
from elfquake.connectors.vlf_abelian import (
    CUMIANA_ENDPOINT,
    build_archive_retrieve_url,
    extract_archive_download_links,
    fetch_cumiana_archive,
    probe_cumiana_archive,
    record_cumiana_stream,
    summarize_archive_response,
)
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
from elfquake.features.vlf_audio import build_vlf_audio_features, extract_vlf_audio_features
from elfquake.features.vlf_image import build_vlf_image_features, extract_vlf_image_features
from elfquake.features.vlf_image_windows import join_vlf_image_features_to_windows
from elfquake.features.vlf_windows import build_vlf_window_features
from elfquake.models.ablation_smoke import train_ablation_smoke
from elfquake.models.aligned_windows import build_aligned_window_dataset
from elfquake.models.alignment_manifest import build_alignment_manifest
from elfquake.models.candidates import list_model_candidates, write_model_candidates
from elfquake.models.comparison import compare_model_run_summaries
from elfquake.models.dataset_combine import combine_aligned_datasets
from elfquake.models.forecast_comparison import compare_weekly_forecasts
from elfquake.models.interface_shape import audit_model_interfaces
from elfquake.models.learned_forecast import generate_learned_weekly_event_forecast
from elfquake.models.logistic_smoke import train_logistic_smoke
from elfquake.models.model_scale import estimate_model_scale
from elfquake.models.readiness import summarize_model_readiness
from elfquake.models.report_summary import summarize_model_run_reports
from elfquake.models.sequence_materializer import materialize_sequence_dataset
from elfquake.models.torch_sequence_data import build_sequence_samples, load_sequence_datasets
from elfquake.models.sequence_comparison import diagnose_sequence_comparison
from elfquake.models.sequence_selection import summarize_sequence_selection
from elfquake.models.split_diagnostics import diagnose_temporal_split
from elfquake.models.synthetic_drift import diagnose_synthetic_drift
from elfquake.models.synthetic_episodes import annotate_synthetic_episodes
from elfquake.models.synthetic_event_list_model import train_synthetic_event_list_model
from elfquake.models.synthetic_event_list_targets import build_synthetic_event_list_targets
from elfquake.models.synthetic_regimes import annotate_synthetic_regimes, assign_balanced_split
from elfquake.models.tensor_materializer import materialize_tensor_dataset
from elfquake.models.tensor_spec import build_tensor_spec
from elfquake.models.temporal_holdout import evaluate_group_holdout, evaluate_temporal_holdout
from elfquake.models.torch_sequence import (
    evaluate_torch_sequence_group_holdout,
    evaluate_torch_sequence_holdout,
    evaluate_torch_sequence_split_holdout,
)
from elfquake.models.torch_patch_transformer import evaluate_torch_patch_transformer_split_holdout
from elfquake.models.torch_self_supervised import (
    compare_sequence_embedding_domains,
    evaluate_mixed_domain_alignment,
    evaluate_synthetic_inlier_transfer,
    pretrain_sequence_autoencoder,
    score_sequence_anomalies,
)
from elfquake.models.torch_tabular import evaluate_torch_tabular_group_holdout, evaluate_torch_tabular_holdout
from elfquake.models.trial_forecast import generate_trial_weekly_event_forecast
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
from elfquake.sim.piezo_transform import transform_piezo_signal_csv
from elfquake.sim.report import benchmark_sandpile_simulation, summarize_sandpile_outputs
from elfquake.storage import write_capture


class AcquisitionScaffoldTests(unittest.TestCase):
    def _write_sequence_fixture(
        self,
        root: Path,
        dataset_id: str,
        modality: str,
        channel: str,
        values: list[float],
    ) -> Path:
        sequence_dir = root / f"{dataset_id}_{modality}_sequence"
        sequence_dir.mkdir()
        (sequence_dir / "time_axis.csv").write_text(
            "time_index,step,time_utc\n"
            + "".join(
                f"{index},{index},2026-01-01T00:{index:02d}:00Z\n"
                for index in range(len(values))
            ),
            encoding="utf-8",
        )
        (sequence_dir / "entity_axis.csv").write_text("entity_index,sensor_id\n0,0\n", encoding="utf-8")
        (sequence_dir / "index.csv").write_text(
            "row_index,time_index,entity_index,step,sensor_id\n"
            + "".join(f"{index},{index},0,{index},0\n" for index in range(len(values))),
            encoding="utf-8",
        )
        (sequence_dir / "values.csv").write_text(
            f"row_index,time_index,entity_index,{channel}\n"
            + "".join(f"{index},{index},0,{value:.6f}\n" for index, value in enumerate(values)),
            encoding="utf-8",
        )
        (sequence_dir / "masks.csv").write_text(
            f"row_index,time_index,entity_index,{channel}__present\n"
            + "".join(f"{index},{index},0,1\n" for index in range(len(values))),
            encoding="utf-8",
        )
        manifest = sequence_dir / "manifest.json"
        manifest.write_text(
            json.dumps(
                {
                    "schema": "elfquake.sequence_dataset.v1",
                    "input_csv": str(root / f"{dataset_id}.{modality}.csv"),
                    "layout": "row,time,entity,channel",
                    "modality": modality,
                    "time_count": len(values),
                    "entity_count": 1,
                    "channel_count": 1,
                    "channel_fields": [channel],
                    "mask_fields": [f"{channel}__present"],
                    "time_axis_csv": str(sequence_dir / "time_axis.csv"),
                    "entity_axis_csv": str(sequence_dir / "entity_axis.csv"),
                    "index_csv": str(sequence_dir / "index.csv"),
                    "values_csv": str(sequence_dir / "values.csv"),
                    "masks_csv": str(sequence_dir / "masks.csv"),
                }
            ),
            encoding="utf-8",
        )
        return manifest

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

    def test_model_summary_comparison_extracts_best_rows(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            summary = root / "summary.json"
            summary.write_text(
                json.dumps(
                    {
                        "schema": "elfquake.model_run_summary.v1",
                        "report_count": 2,
                        "reports": [
                            {
                                "path": "temporal.json",
                                "schema": "elfquake.torch_tabular_holdout.v1",
                                "status": "evaluated",
                                "split": {"type": "temporal", "backend": "torch", "device": "cpu"},
                                "train_row_count": 8,
                                "test_row_count": 2,
                                "train_positive_count": 4,
                                "test_positive_count": 1,
                                "best_default_balanced_accuracy": {
                                    "name": "seismic_only",
                                    "test_balanced_accuracy": 0.5,
                                },
                                "best_calibrated_balanced_accuracy": {
                                    "name": "seismic_only",
                                    "calibrated_test_balanced_accuracy": 0.5,
                                },
                            },
                            {
                                "path": "group.json",
                                "schema": "elfquake.torch_tabular_group_holdout.v1",
                                "status": "evaluated",
                                "split": {
                                    "type": "group",
                                    "backend": "torch",
                                    "device": "cpu",
                                    "test_group": "seed42",
                                },
                                "train_row_count": 10,
                                "test_row_count": 5,
                                "train_positive_count": 5,
                                "test_positive_count": 2,
                                "best_default_balanced_accuracy": {
                                    "name": "synthetic_seismic_piezo_vlf",
                                    "test_balanced_accuracy": 0.6,
                                },
                                "best_calibrated_balanced_accuracy": {
                                    "name": "synthetic_seismic_piezo_vlf",
                                    "calibrated_test_balanced_accuracy": 0.75,
                                },
                            },
                        ],
                    }
                ),
                encoding="utf-8",
            )

            report = compare_model_run_summaries(
                summary_paths=[summary],
                out_path=root / "comparison.json",
                csv_out_path=root / "comparison.csv",
            )

            self.assertEqual(report["report_count"], 2)
            self.assertEqual(
                report["best_calibrated_balanced_accuracy"]["model_name"],
                "synthetic_seismic_piezo_vlf",
            )
            self.assertEqual(report["best_calibrated_balanced_accuracy"]["test_group"], "seed42")
            self.assertTrue((root / "comparison.csv").exists())

            nested = compare_model_run_summaries(
                summary_paths=[root / "comparison.json"],
                out_path=root / "nested.json",
            )

            self.assertEqual(nested["report_count"], 2)
            self.assertEqual(nested["best_calibrated_balanced_accuracy"]["test_group"], "seed42")

    def test_trial_weekly_event_forecast_writes_event_coordinates(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            real_events = root / "real_events.csv"
            real_events.write_text(
                "event_id,source,event_time_utc,latitude,longitude,depth_km,magnitude,magnitude_type,italy_region,event_location_name,event_type,raw_file,ingested_at_utc,raw_uri\n"
                "r1,ingv,2026-01-01T00:00:00Z,42.5,13.2,8,2.4,ML,central_italy,fixture,earthquake,,,\n"
                "r2,ingv,2026-01-03T00:00:00Z,43.0,12.5,8,2.0,ML,central_italy,fixture,earthquake,,,\n"
                "r3,ingv,2026-01-08T00:00:00Z,45.0,10.5,8,2.7,ML,unknown,fixture,earthquake,,,\n",
                encoding="utf-8",
            )
            synthetic = root / "fixture.synthetic_events.csv"
            synthetic.write_text(
                "event_id,source,event_time_utc,latitude,longitude,depth_km,magnitude,magnitude_type,italy_region,event_location_name,event_type,raw_file,ingested_at_utc,raw_uri\n"
                "s1,synthetic,2026-01-01T00:00:00Z,42.3,13.7,10,2.6,MLs,central_italy,fixture,earthquake,,,\n",
                encoding="utf-8",
            )
            vlf_window = root / "vlf.csv"
            vlf_window.write_text(
                "window_end_utc,vlf_capture_count,real_vlf_image_vlf_intensity_mean_mean,quality_missing_vlf\n"
                "2026-01-09T00:00:00Z,3,0.4,0\n",
                encoding="utf-8",
            )
            anomaly = root / "anomaly.json"
            anomaly.write_text(
                json.dumps({"forecast": {"demo_probability": 0.8, "forecast_start_utc": "2026-01-09T00:00:00Z"}}),
                encoding="utf-8",
            )
            astronomy = root / "astronomy.json"
            astronomy.write_text(
                json.dumps([{"time-tag": "2026-01", "f10.7": 100.0}, {"time-tag": "2026-02", "f10.7": 140.0}]),
                encoding="utf-8",
            )
            report_path = root / "forecast.json"
            events_out = root / "events.csv"

            report = generate_trial_weekly_event_forecast(
                real_events_csv=real_events,
                out_path=report_path,
                events_out_path=events_out,
                as_of_utc="2026-01-10T00:00:00Z",
                max_events=5,
                seed=7,
                synthetic_event_globs=[str(synthetic)],
                vlf_window_csvs=[vlf_window],
                vlf_anomaly_report=anomaly,
                astronomy_globs=[str(astronomy)],
            )

            self.assertEqual(report["status"], "trial_run")
            self.assertGreaterEqual(report["predicted_event_count"], 1)
            with events_out.open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), report["predicted_event_count"])
            self.assertIn("latitude", rows[0])
            self.assertIn("longitude", rows[0])
            self.assertGreater(float(rows[0]["magnitude_proxy"]), 2.0)

    def test_learned_weekly_event_forecast_writes_scorer_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            real_events = root / "real_events.csv"
            real_events.write_text(
                "event_id,source,event_time_utc,latitude,longitude,depth_km,magnitude,magnitude_type,italy_region,event_location_name,event_type,raw_file,ingested_at_utc,raw_uri\n"
                "r1,ingv,2026-01-01T00:00:00Z,42.5,13.2,8,2.4,ML,central_italy,fixture,earthquake,,,\n"
                "r2,ingv,2026-01-08T00:00:00Z,45.0,10.5,8,2.7,ML,unknown,fixture,earthquake,,,\n",
                encoding="utf-8",
            )
            synthetic_events = root / "fixture.avalanche_events.csv"
            synthetic_events.write_text(
                "event_id,source,event_time_utc,latitude,longitude,depth_km,magnitude,magnitude_type,italy_region,event_location_name,event_type,raw_file,ingested_at_utc,raw_uri\n"
                "s1,synthetic,2026-01-01T00:00:00Z,42.3,13.7,10,2.6,MLs,central_italy,fixture,earthquake,,,\n",
                encoding="utf-8",
            )
            synthetic_windows = root / "windows.csv"
            synthetic_windows.write_text(
                "dataset_id,window_id,region_id,window_start_utc,window_end_utc,source_file,target_event_count,target_occurred,target_status,"
                "synthetic_piezo_vlf_piezo_signal_mean,synthetic_direct_avalanche_avalanche_signal_mean,quality_missing_synthetic_piezo_vlf\n"
                "s,w1,central,2026-01-01T00:00:00Z,2026-01-01T01:00:00Z,x,0,0,labeled,0.1,0.1,0\n"
                "s,w2,central,2026-01-01T01:00:00Z,2026-01-01T02:00:00Z,x,1,1,labeled,2.0,3.0,0\n"
                "s,w3,central,2026-01-01T02:00:00Z,2026-01-01T03:00:00Z,x,0,0,labeled,0.2,0.3,0\n"
                "s,w4,central,2026-01-01T03:00:00Z,2026-01-01T04:00:00Z,x,1,1,labeled,2.5,3.4,0\n"
                "s,w5,central,2026-01-01T04:00:00Z,2026-01-01T05:00:00Z,x,0,0,labeled,0.3,0.4,0\n"
                "s,w6,central,2026-01-01T05:00:00Z,2026-01-01T06:00:00Z,x,1,1,labeled,2.8,3.8,0\n",
                encoding="utf-8",
            )
            report_path = root / "learned.json"
            events_out = root / "learned_events.csv"

            report = generate_learned_weekly_event_forecast(
                real_events_csv=real_events,
                synthetic_windows_csv=synthetic_windows,
                out_path=report_path,
                events_out_path=events_out,
                as_of_utc="2026-01-10T00:00:00Z",
                max_events=4,
                seed=7,
                synthetic_event_globs=[str(synthetic_events)],
                vlf_window_csvs=[root / "missing_vlf.csv"],
                vlf_audio_globs=[str(root / "missing_audio_*.csv")],
                astronomy_globs=[str(root / "missing_astro_*.json")],
                epochs=20,
            )

            self.assertEqual(report["schema"], "elfquake.learned_multimodal_weekly_event_forecast.v1")
            self.assertEqual(report["model"]["learned_scorer"]["status"], "evaluated")
            self.assertGreaterEqual(report["predicted_event_count"], 1)
            with events_out.open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            self.assertTrue(rows[0]["prediction_id"].startswith("learned_"))
            self.assertIn("synthetic-trained", rows[0]["warning"])

    def test_compare_weekly_forecasts_reports_success_criteria(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            baseline_report = root / "baseline.json"
            candidate_report = root / "candidate.json"
            baseline_events = root / "baseline.csv"
            candidate_events = root / "candidate.csv"
            baseline_report.write_text(
                json.dumps(
                    {
                        "schema": "elfquake.trial_multimodal_weekly_event_forecast.v1",
                        "forecast_start_utc": "2026-07-08T00:00:00Z",
                        "forecast_end_utc": "2026-07-15T00:00:00Z",
                        "uncapped_expected_event_count": 10.0,
                        "model": {"type": "heuristic"},
                        "warning": "trial",
                    }
                ),
                encoding="utf-8",
            )
            candidate_report.write_text(
                json.dumps(
                    {
                        "schema": "elfquake.learned_multimodal_weekly_event_forecast.v1",
                        "forecast_start_utc": "2026-07-08T00:00:00Z",
                        "forecast_end_utc": "2026-07-15T00:00:00Z",
                        "uncapped_expected_event_count": 9.0,
                        "model": {
                            "type": "synthetic_window_logistic_scorer",
                            "learned_scorer": {
                                "test_metrics": {
                                    "balanced_accuracy": 0.5,
                                    "positive_recall": 1.0,
                                    "negative_recall": 0.0,
                                }
                            },
                        },
                        "warning": "learned",
                    }
                ),
                encoding="utf-8",
            )
            header = (
                "prediction_id,forecast_time_utc,latitude,longitude,magnitude_proxy,probability_proxy,"
                "expected_week_count,real_spatial_weight,synthetic_spatial_weight,vlf_context_score,"
                "astronomy_context_score,synthetic_context_score,warning\n"
            )
            baseline_events.write_text(
                header + "trial_001,2026-07-08T00:00:00Z,42,13,3.0,0.8,10,1,0,0,0,0,trial\n",
                encoding="utf-8",
            )
            candidate_events.write_text(
                header + "learned_001,2026-07-08T00:00:00Z,42.1,13.1,3.1,0.7,9,1,0,0,0,0,learned\n",
                encoding="utf-8",
            )

            report = compare_weekly_forecasts(
                baseline_report=baseline_report,
                baseline_events=baseline_events,
                candidate_report=candidate_report,
                candidate_events=candidate_events,
                out_path=root / "comparison.json",
                csv_out_path=root / "comparison.csv",
            )

            self.assertEqual(report["status"], "evaluated")
            self.assertTrue(report["criteria"]["stage_1_event_contract_pass"])
            self.assertFalse(report["criteria"]["stage_2_synthetic_model_pass"])
            self.assertTrue((root / "comparison.csv").exists())

    def test_build_synthetic_event_list_targets_adds_location_targets(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            events = root / "events.csv"
            events.write_text(
                "event_id,event_time_utc,latitude,longitude,depth_km,magnitude\n"
                "e1,2026-01-01T02:30:00Z,42.1,13.2,8,2.4\n"
                "e2,2026-01-01T03:30:00Z,42.3,13.4,9,3.1\n",
                encoding="utf-8",
            )
            windows = root / "windows.csv"
            windows.write_text(
                "dataset_id,window_id,window_start_utc,window_end_utc,source_file,feature\n"
                f"seed1,w0,2026-01-01T00:00:00Z,2026-01-01T01:00:00Z,{events},0\n"
                f"seed1,w1,2026-01-01T01:00:00Z,2026-01-01T02:00:00Z,{events},1\n"
                f"seed1,w2,2026-01-01T02:00:00Z,2026-01-01T03:00:00Z,{events},2\n"
                f"seed1,w3,2026-01-01T03:00:00Z,2026-01-01T04:00:00Z,{events},3\n",
                encoding="utf-8",
            )

            report = build_synthetic_event_list_targets(
                input_csv=windows,
                out_csv=root / "targets.csv",
                report_path=root / "targets.json",
                horizon_rows=2,
                magnitude_threshold=2.0,
            )

            with (root / "targets.csv").open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(rows[0]["eventlist_target_count"], "1")
            self.assertEqual(rows[0]["eventlist_target_occurred"], "1")
            self.assertEqual(rows[0]["eventlist_target_centroid_latitude"], "42.100000000")
            self.assertEqual(rows[1]["eventlist_target_max_magnitude"], "3.100000000")
            self.assertEqual(rows[-1]["eventlist_target_status"], "unlabeled_no_future_window")
            self.assertEqual(report["positive_count"], 2)

    def test_train_synthetic_event_list_model_writes_prediction_heads(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            targets = root / "targets.csv"
            targets.write_text(
                "dataset_id,window_id,window_start_utc,feature_a,feature_b,eventlist_target_status,"
                "eventlist_target_count,eventlist_target_occurred,eventlist_target_max_magnitude,"
                "eventlist_target_centroid_latitude,eventlist_target_centroid_longitude\n"
                "seed1,w0,2026-01-01T00:00:00Z,0.0,1.0,labeled,0,0,0,,\n"
                "seed1,w1,2026-01-01T01:00:00Z,1.0,0.8,labeled,1,1,2.5,42.0,13.0\n"
                "seed1,w2,2026-01-01T02:00:00Z,2.0,0.6,labeled,1,1,2.7,42.2,13.2\n"
                "seed1,w3,2026-01-01T03:00:00Z,3.0,0.4,labeled,0,0,0,,\n"
                "seed1,w4,2026-01-01T04:00:00Z,4.0,0.2,labeled,2,1,3.1,42.4,13.4\n",
                encoding="utf-8",
            )

            report = train_synthetic_event_list_model(
                input_csv=targets,
                out_path=root / "model.json",
                predictions_out=root / "predictions.csv",
                train_fraction=0.8,
                epochs=10,
                learning_rate=0.02,
            )

            self.assertEqual(report["schema"], "elfquake.synthetic_event_list_model.v1")
            self.assertEqual(report["status"], "evaluated")
            self.assertTrue((root / "model.json").exists())
            with (root / "predictions.csv").open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 1)
            self.assertIn("predicted_event_count", rows[0])
            self.assertIn("predicted_centroid_latitude", rows[0])

    def test_synthetic_drift_reports_temporal_positive_shift(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            targets = root / "targets.csv"
            targets.write_text(
                "dataset_id,window_id,window_start_utc,feature_a,eventlist_target_status,eventlist_target_occurred\n"
                "seed1,w0,2026-01-01T00:00:00Z,0,labeled,0\n"
                "seed1,w1,2026-01-01T01:00:00Z,1,labeled,0\n"
                "seed1,w2,2026-01-01T02:00:00Z,2,labeled,0\n"
                "seed1,w3,2026-01-01T03:00:00Z,3,labeled,1\n"
                "seed1,w4,2026-01-01T04:00:00Z,4,labeled,1\n"
                "seed1,w5,2026-01-01T05:00:00Z,5,labeled,1\n",
                encoding="utf-8",
            )

            report = diagnose_synthetic_drift(
                input_csv=targets,
                out_path=root / "drift.json",
                csv_out_path=root / "drift.csv",
                train_fraction=0.5,
                bucket_count=3,
            )

            self.assertEqual(report["schema"], "elfquake.synthetic_drift_diagnostic.v1")
            self.assertEqual(report["temporal_split"]["warning"], "test_split_has_no_negatives")
            self.assertTrue((root / "drift.csv").exists())
            self.assertEqual(report["time_buckets"][-1]["positive_rate"], 1.0)

    def test_synthetic_episode_annotation_marks_blocks_and_excludes_leakage_fields(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            targets = root / "targets.csv"
            targets.write_text(
                "dataset_id,window_id,window_start_utc,feature_a,eventlist_target_status,"
                "eventlist_target_count,eventlist_target_occurred,eventlist_target_max_magnitude,"
                "eventlist_target_centroid_latitude,eventlist_target_centroid_longitude\n"
                "seed1,w0,2026-01-01T00:00:00Z,0,labeled,0,0,0,,\n"
                "seed1,w1,2026-01-01T01:00:00Z,1,labeled,1,1,2.5,42.0,13.0\n"
                "seed1,w2,2026-01-01T02:00:00Z,2,labeled,0,0,0,,\n"
                "seed1,w3,2026-01-01T03:00:00Z,3,labeled,1,1,2.7,42.2,13.2\n",
                encoding="utf-8",
            )

            report = annotate_synthetic_episodes(
                input_csv=targets,
                out_csv=root / "episodes.csv",
                report_path=root / "episodes.json",
                rows_per_episode=2,
            )
            self.assertEqual(report["episode_count"], 2)
            model = train_synthetic_event_list_model(
                input_csv=root / "episodes.csv",
                out_path=root / "model.json",
                predictions_out=root / "predictions.csv",
                epochs=5,
            )

            self.assertEqual(model["status"], "evaluated")
            top_feature_names = [row["name"] for row in model["top_occurrence_features"]]
            self.assertNotIn("synthetic_episode_index", top_feature_names)
            self.assertNotIn("synthetic_episode_row_index", top_feature_names)

    def test_sequence_comparison_diagnostic_reads_full_reports(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            default_report = root / "default_sequence_group_seed42.json"
            sweep_dir = root / "sequence_sweep" / "lookback_60_hidden_24"
            sweep_dir.mkdir(parents=True)
            sweep_report = sweep_dir / "torch_sequence_group_seed42.json"
            _write_sequence_report(default_report, epochs=20, best_name="sequence_piezo_vlf_only", best_score=0.77)
            _write_sequence_report(sweep_report, epochs=10, best_name="sequence_direct_avalanche_only", best_score=0.76)
            comparison = root / "comparison.json"
            comparison.write_text(
                json.dumps(
                    {
                        "schema": "elfquake.model_summary_comparison.v1",
                        "rows": [
                            {
                                "summary_path": "default.json",
                                "report_path": str(default_report),
                                "schema": "elfquake.torch_sequence_group_holdout.v1",
                                "status": "evaluated",
                                "split_type": "group",
                                "test_group": "seed42",
                            },
                            {
                                "summary_path": "sequence_sweep/summary.json",
                                "report_path": str(sweep_report),
                                "schema": "elfquake.torch_sequence_group_holdout.v1",
                                "status": "evaluated",
                                "split_type": "group",
                                "test_group": "seed42",
                            },
                        ],
                    }
                ),
                encoding="utf-8",
            )

            report = diagnose_sequence_comparison(
                comparison_path=comparison,
                out_path=root / "diagnostic.json",
                csv_out_path=root / "diagnostic.csv",
            )

            self.assertEqual(report["evaluation_row_count"], 4)
            self.assertEqual(report["best_overall"]["evaluation_name"], "sequence_piezo_vlf_only")
            self.assertIn("default_sequence", report["best_by_source"])
            self.assertTrue(any("matched epochs" in note for note in report["notes"]))
            self.assertTrue((root / "diagnostic.csv").exists())

            selection = summarize_sequence_selection(
                diagnostic_path=root / "diagnostic.json",
                out_path=root / "selection.json",
                csv_out_path=root / "selection.csv",
            )

            self.assertEqual(selection["evaluation_count"], 2)
            self.assertEqual(selection["best_single_row"]["evaluation_name"], "sequence_piezo_vlf_only")
            self.assertTrue((root / "selection.csv").exists())

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

    def test_abelian_live_stream_records_cumiana_ogg(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)

            stored = record_cumiana_stream(
                out_root=root,
                duration_seconds=2,
                max_bytes=1024,
                fetcher=_fake_ogg_stream_capture,
            )

            self.assertEqual(stored.payload_path.suffix, ".ogg")
            self.assertIn("abelian_cumiana_vlf15", stored.payload_path.name)
            self.assertEqual(stored.payload_path.read_bytes(), b"OggSfake")
            metadata = json.loads(stored.metadata_path.read_text(encoding="utf-8"))
            self.assertEqual(metadata["source_kind"], "vlf_live_audio_stream")
            self.assertEqual(metadata["station"], "cumiana")
            self.assertEqual(metadata["stream_id"], "vlf15")
            self.assertEqual(metadata["content_type"], "audio/ogg")

    def test_abelian_live_stream_rejects_empty_body(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "no audio bytes"):
                record_cumiana_stream(
                    out_root=Path(directory),
                    duration_seconds=2,
                    fetcher=_fake_empty_ogg_stream_capture,
                )

    def test_abelian_archive_url_and_download_extraction(self) -> None:
        url = build_archive_retrieve_url(
            endpoint=CUMIANA_ENDPOINT,
            start_time_utc="2026-07-05T10:38:11Z",
            duration_seconds=0.05,
            output_format="wav",
        )
        query = parse_qs(urlparse(url).query)

        self.assertEqual(query["ts"], ["2026-07-05 10:38:11"])
        self.assertEqual(query["len"], ["0.05"])
        self.assertEqual(query["vlf15"], ["on"])
        self.assertEqual(query["format"], ["wav"])
        self.assertEqual(
            extract_archive_download_links(
                "Download <A href=http:/vlf/live/retrieve/1783247914_18149.wav>file</A>",
                base_url=url,
            ),
            ["http://abelian.org/vlf/live/retrieve/1783247914_18149.wav"],
        )
        summary = summarize_archive_response(
            "retrieving vlf15... no database for vlf15 "
            "Download <A href=http:/vlf/live/retrieve/1783247914_18149.wav>file</A> size 0<br>",
            base_url=url,
        )
        self.assertTrue(summary["no_database"])
        self.assertEqual(summary["declared_download_size_bytes"], 0)
        self.assertEqual(summary["download_links"], ["http://abelian.org/vlf/live/retrieve/1783247914_18149.wav"])

    def test_abelian_archive_fetch_stores_request_and_nonempty_download(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            urls: list[str] = []

            def fetcher(url: str) -> HttpCapture:
                urls.append(url)
                if "/vlf/live/retrieve/" in url:
                    return HttpCapture(
                        url=url,
                        status=200,
                        captured_at_utc=datetime(2026, 7, 5, 10, 40, tzinfo=timezone.utc),
                        headers={"Content-Type": "audio/wav"},
                        body=b"RIFFfake",
                    )
                return HttpCapture(
                    url=url,
                    status=200,
                    captured_at_utc=datetime(2026, 7, 5, 10, 39, tzinfo=timezone.utc),
                    headers={"Content-Type": "text/html"},
                    body=b"Download <A href=http:/vlf/live/retrieve/1783247914_18149.wav>file</A>",
                )

            stored = fetch_cumiana_archive(
                out_root=root,
                start_time_utc="2026-07-05T10:38:11Z",
                duration_seconds=0.05,
                output_format="wav",
                fetcher=fetcher,
            )

            self.assertEqual(len(stored), 2)
            self.assertEqual(stored[0].payload_path.suffix, ".html")
            self.assertEqual(stored[1].payload_path.suffix, ".wav")
            self.assertEqual(stored[1].payload_path.read_bytes(), b"RIFFfake")
            self.assertEqual(urls[1], "http://abelian.org/vlf/live/retrieve/1783247914_18149.wav")

    def test_abelian_archive_probe_reports_empty_and_nonempty_rows(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)

            def fetcher(url: str) -> HttpCapture:
                body = (
                    b"Download <A href=http:/vlf/live/retrieve/nonempty.wav>file</A> size 8<br>"
                    if "format=vt" in url
                    else b"retrieving vlf15... no database for vlf15 "
                    b"Download <A href=http:/vlf/live/retrieve/empty.wav>file</A> size 0<br>"
                )
                return HttpCapture(
                    url=url,
                    status=200,
                    captured_at_utc=datetime(2026, 7, 5, 10, 39, tzinfo=timezone.utc),
                    headers={"Content-Type": "text/html"},
                    body=body,
                )

            rows = probe_cumiana_archive(
                start_times_utc=["2026-07-05T10:38:11Z"],
                duration_seconds=0.05,
                output_formats=["wav", "vt"],
                out_path=root / "probe.csv",
                fetcher=fetcher,
            )

            self.assertEqual(len(rows), 2)
            self.assertEqual(rows[0]["format"], "wav")
            self.assertEqual(rows[0]["no_database"], "1")
            self.assertEqual(rows[0]["usable_nonempty"], "0")
            self.assertEqual(rows[1]["format"], "vt")
            self.assertEqual(rows[1]["declared_download_size_bytes"], "8")
            self.assertEqual(rows[1]["usable_nonempty"], "1")
            self.assertTrue((root / "probe.csv").exists())

    def test_vlf_audio_features_summarize_audio_capture(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            audio = root / "capture.ogg"
            audio.write_bytes(b"OggSfake")
            audio.with_suffix(".ogg.metadata.json").write_text(
                json.dumps(
                    {
                        "captured_at_utc": "2026-07-05T10:40:00Z",
                        "provider": "abelian",
                        "station": "cumiana",
                        "stream_id": "vlf15",
                        "content_type": "audio/ogg",
                    }
                ),
                encoding="utf-8",
            )

            row = extract_vlf_audio_features(audio, use_ffprobe=False)
            rows = build_vlf_audio_features(
                audio_paths=[audio],
                out_path=root / "audio_features.csv",
                use_ffprobe=False,
            )

            self.assertEqual(row["vlf_audio_provider"], "abelian")
            self.assertEqual(row["vlf_audio_station"], "cumiana")
            self.assertEqual(row["vlf_audio_byte_count"], "8")
            self.assertEqual(row["vlf_audio_ogg_page_count"], "1")
            self.assertEqual(row["quality_missing_vlf_audio"], "0")
            self.assertEqual(row["quality_unreadable_vlf_audio"], "0")
            self.assertEqual(rows[0]["vlf_audio_stream_id"], "vlf15")

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

    def test_astronomy_features_can_use_slow_context_outside_capture_window(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            moon_payload = root / "usno_moon_phases_2026-06-29T09-56-53Z.json"
            moon_payload.write_text(
                json.dumps(
                    {
                        "phasedata": [
                            {"year": 2026, "month": 7, "day": 7, "time": "19:29", "phase": "Last Quarter"}
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
                window_start_utc="2026-07-02T00:00:00Z",
                window_end_utc="2026-07-03T00:00:00Z",
                out_path=root / "astro.csv",
            )

            self.assertEqual(row["astro_capture_count"], "0")
            self.assertEqual(row["astro_usno_next_phase"], "Last Quarter")
            self.assertEqual(row["astro_noaa_solar_cycle_f107_value"], "125.69")
            self.assertEqual(row["quality_missing_astro"], "0")

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

    def test_model_readiness_maps_synthetic_piezo_to_vlf_role(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            table = root / "synthetic.csv"
            table.write_text(
                "window_id,region_id,synthetic_seismic_event_count,synthetic_piezo_vlf_signal_mean,"
                "target_occurred,target_status\n"
                "w1,central_italy,1,0.1,0,labeled\n"
                "w2,central_italy,3,0.8,1,labeled\n",
                encoding="utf-8",
            )

            report = summarize_model_readiness(input_csv=table, out_path=root / "readiness.json")

            self.assertIn("vlf", report["available_feature_groups"])
            self.assertIn("synthetic_piezo_vlf", report["feature_roles"]["vlf"])
            self.assertTrue(report["ablation_plan"]["synthetic_vlf_only"]["has_required_features"])
            self.assertTrue(report["ablation_plan"]["synthetic_seismic_vlf_unified"]["has_required_features"])

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

    @unittest.skipUnless(importlib.util.find_spec("torch"), "PyTorch optional dependency is not installed")
    def test_torch_tabular_holdout_trains_synthetic_ablations(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            table = root / "aligned.csv"
            table.write_text(
                "window_id,region_id,window_start_utc,synthetic_seismic_event_count,"
                "synthetic_piezo_vlf_signal_mean,synthetic_summary_topple_count_mean,"
                "target_occurred,target_status\n"
                "w1,r,2026-01-01T00:00:00Z,0,0.1,1,0,labeled\n"
                "w2,r,2026-01-01T01:00:00Z,1,,2,0,labeled\n"
                "w3,r,2026-01-01T02:00:00Z,2,0.3,3,1,labeled\n"
                "w4,r,2026-01-01T03:00:00Z,3,0.4,4,1,labeled\n"
                "w5,r,2026-01-01T04:00:00Z,4,0.5,5,0,labeled\n"
                "w6,r,2026-01-01T05:00:00Z,5,0.6,6,1,labeled\n",
                encoding="utf-8",
            )

            report = evaluate_torch_tabular_holdout(
                input_csv=table,
                out_path=root / "torch.json",
                train_fraction=0.67,
                epochs=4,
                hidden_units=4,
                batch_size=2,
                seed=7,
            )

            self.assertEqual(report["schema"], "elfquake.torch_tabular_holdout.v1")
            self.assertEqual(report["status"], "evaluated")
            self.assertEqual(report["device"], "cpu")
            self.assertEqual(report["train_row_count"], 4)
            self.assertEqual(report["evaluations"]["synthetic_full"]["status"], "evaluated")
            self.assertEqual(report["evaluations"]["synthetic_vlf_only"]["status"], "evaluated")
            self.assertIn(
                "synthetic_piezo_vlf_signal_mean__present_mask",
                report["evaluations"]["synthetic_full"]["model_feature_names"],
            )
            self.assertIn(
                "synthetic_piezo_vlf_signal_mean",
                report["evaluations"]["vlf_only"]["feature_names"],
            )
            self.assertTrue((root / "torch.json").exists())

    @unittest.skipUnless(importlib.util.find_spec("torch"), "PyTorch optional dependency is not installed")
    def test_torch_tabular_group_holdout_uses_test_group(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            table = root / "aligned.csv"
            table.write_text(
                "dataset_id,window_id,region_id,synthetic_seismic_event_count,"
                "synthetic_piezo_vlf_signal_mean,target_occurred,target_status\n"
                "seed1,w1,r,0,0.1,0,labeled\n"
                "seed1,w2,r,1,0.2,0,labeled\n"
                "seed1,w3,r,5,0.5,1,labeled\n"
                "seed2,w1,r,0,0.1,0,labeled\n"
                "seed2,w2,r,6,0.6,1,labeled\n"
                "seed2,w3,r,7,0.7,1,labeled\n",
                encoding="utf-8",
            )

            report = evaluate_torch_tabular_group_holdout(
                input_csv=table,
                out_path=root / "torch_group.json",
                test_group="seed2",
                epochs=4,
                hidden_units=4,
                batch_size=2,
            )

            self.assertEqual(report["schema"], "elfquake.torch_tabular_group_holdout.v1")
            self.assertEqual(report["status"], "evaluated")
            self.assertEqual(report["train_groups"], ["seed1"])
            self.assertEqual(report["test_group"], "seed2")
            self.assertEqual(report["test_row_count"], 3)
            self.assertNotIn("dataset_id", report["evaluations"]["all_features"]["feature_names"])

    @unittest.skipUnless(importlib.util.find_spec("torch"), "PyTorch optional dependency is not installed")
    def test_torch_sequence_holdout_trains_synthetic_modalities(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            aligned = root / "aligned.csv"
            aligned.write_text(
                "dataset_id,window_id,region_id,window_start_utc,window_end_utc,target_occurred,target_status\n"
                "seed1,w0,r,2026-01-01T00:00:00Z,2026-01-01T00:02:00Z,0,labeled\n"
                "seed1,w1,r,2026-01-01T00:02:00Z,2026-01-01T00:04:00Z,0,labeled\n"
                "seed1,w2,r,2026-01-01T00:04:00Z,2026-01-01T00:06:00Z,1,labeled\n"
                "seed1,w3,r,2026-01-01T00:06:00Z,2026-01-01T00:08:00Z,1,labeled\n"
                "seed1,w4,r,2026-01-01T00:08:00Z,2026-01-01T00:10:00Z,0,labeled\n"
                "seed1,w5,r,2026-01-01T00:10:00Z,2026-01-01T00:12:00Z,1,labeled\n",
                encoding="utf-8",
            )
            manifests = [
                self._write_sequence_fixture(root, "seed1", "synthetic_direct_avalanche", "avalanche_signal", [0, 0, 1, 1, 2, 2, 6, 7, 8, 2, 3, 9, 9]),
                self._write_sequence_fixture(root, "seed1", "synthetic_piezo_vlf", "piezo_signal", [0, 0, 1, 2, 3, 4, 8, 8, 7, 3, 2, 9, 10]),
                self._write_sequence_fixture(root, "seed1", "synthetic_summary", "topple_count", [1, 1, 2, 2, 3, 3, 8, 8, 8, 3, 3, 9, 9]),
            ]

            report = evaluate_torch_sequence_holdout(
                input_csv=aligned,
                sequence_manifest_paths=manifests,
                out_path=root / "sequence.json",
                train_fraction=0.67,
                lookback_steps=2,
                epochs=3,
                hidden_units=4,
                batch_size=2,
            )

            self.assertEqual(report["schema"], "elfquake.torch_sequence_holdout.v1")
            self.assertEqual(report["status"], "evaluated")
            self.assertEqual(report["evaluations"]["sequence_piezo_vlf_only"]["status"], "evaluated")
            self.assertEqual(report["evaluations"]["sequence_full"]["status"], "evaluated")
            self.assertIn("synthetic_piezo_vlf_piezo_signal", report["evaluations"]["sequence_piezo_vlf_only"]["feature_names"])

            filtered = evaluate_torch_sequence_holdout(
                input_csv=aligned,
                sequence_manifest_paths=manifests,
                out_path=root / "sequence_full_only.json",
                train_fraction=0.67,
                lookback_steps=2,
                epochs=1,
                hidden_units=4,
                batch_size=2,
                evaluation_names=["sequence_full"],
            )

            self.assertEqual(list(filtered["evaluations"]), ["sequence_full"])
            self.assertEqual(filtered["selected_evaluations"], ["sequence_full"])

    @unittest.skipUnless(importlib.util.find_spec("torch"), "PyTorch optional dependency is not installed")
    def test_torch_sequence_group_holdout_uses_test_group(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            aligned = root / "aligned.csv"
            aligned.write_text(
                "dataset_id,window_id,region_id,window_start_utc,window_end_utc,target_occurred,target_status\n"
                "seed1,w0,r,2026-01-01T00:00:00Z,2026-01-01T00:02:00Z,0,labeled\n"
                "seed1,w1,r,2026-01-01T00:02:00Z,2026-01-01T00:04:00Z,1,labeled\n"
                "seed2,w0,r,2026-01-01T00:00:00Z,2026-01-01T00:02:00Z,0,labeled\n"
                "seed2,w1,r,2026-01-01T00:02:00Z,2026-01-01T00:04:00Z,1,labeled\n",
                encoding="utf-8",
            )
            manifests = []
            for seed in ("seed1", "seed2"):
                manifests.extend([
                    self._write_sequence_fixture(root, seed, "synthetic_direct_avalanche", "avalanche_signal", [0, 1, 2, 3, 4]),
                    self._write_sequence_fixture(root, seed, "synthetic_piezo_vlf", "piezo_signal", [0, 2, 4, 6, 8]),
                ])

            report = evaluate_torch_sequence_group_holdout(
                input_csv=aligned,
                sequence_manifest_paths=manifests,
                out_path=root / "sequence_group.json",
                test_group="seed2",
                lookback_steps=2,
                epochs=3,
                hidden_units=4,
                batch_size=2,
            )

            self.assertEqual(report["schema"], "elfquake.torch_sequence_group_holdout.v1")
            self.assertEqual(report["status"], "evaluated")
            self.assertEqual(report["train_groups"], ["seed1"])
            self.assertEqual(report["test_group"], "seed2")
            self.assertEqual(report["evaluations"]["sequence_direct_avalanche_piezo_vlf"]["status"], "evaluated")

    def test_sequence_loader_uses_single_matching_dataset_for_real_rows_without_dataset_id(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = self._write_sequence_fixture(
                root,
                "cumiana",
                "real_vlf_image",
                "vlf_intensity_mean",
                [0, 1, 2, 3, 4],
            )
            sequences = load_sequence_datasets([manifest], include_missing_masks=True)

            samples, feature_names = build_sequence_samples(
                [
                    {
                        "window_end_utc": "2026-01-01T00:04:30Z",
                        "target_occurred": "1",
                    }
                ],
                sequences,
                modalities=("real_vlf_image",),
                lookback_steps=2,
            )

            self.assertEqual(len(samples), 1)
            self.assertEqual(len(samples[0]), 2)
            self.assertIn("real_vlf_image_vlf_intensity_mean", feature_names)

    @unittest.skipUnless(importlib.util.find_spec("torch"), "PyTorch optional dependency is not installed")
    def test_torch_sequence_split_holdout_uses_explicit_split(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            aligned = root / "aligned.csv"
            aligned.write_text(
                "dataset_id,window_id,region_id,window_start_utc,window_end_utc,target_occurred,target_status,model_split\n"
                "seed1,w0,r,2026-01-01T00:00:00Z,2026-01-01T00:02:00Z,0,labeled,train\n"
                "seed1,w1,r,2026-01-01T00:02:00Z,2026-01-01T00:04:00Z,1,labeled,train\n"
                "seed1,w2,r,2026-01-01T00:04:00Z,2026-01-01T00:06:00Z,0,labeled,train\n"
                "seed1,w3,r,2026-01-01T00:06:00Z,2026-01-01T00:08:00Z,1,labeled,train\n"
                "seed1,w4,r,2026-01-01T00:08:00Z,2026-01-01T00:10:00Z,0,labeled,test\n"
                "seed1,w5,r,2026-01-01T00:10:00Z,2026-01-01T00:12:00Z,1,labeled,test\n",
                encoding="utf-8",
            )
            manifests = [
                self._write_sequence_fixture(root, "seed1", "synthetic_direct_avalanche", "avalanche_signal", [0, 0, 1, 1, 2, 2, 6, 7, 8, 2, 3, 9, 9]),
                self._write_sequence_fixture(root, "seed1", "synthetic_piezo_vlf", "piezo_signal", [0, 0, 1, 2, 3, 4, 8, 8, 7, 3, 2, 9, 10]),
                self._write_sequence_fixture(root, "seed1", "synthetic_summary", "topple_count", [1, 1, 2, 2, 3, 3, 8, 8, 8, 3, 3, 9, 9]),
            ]

            report = evaluate_torch_sequence_split_holdout(
                input_csv=aligned,
                sequence_manifest_paths=manifests,
                out_path=root / "sequence_split.json",
                lookback_steps=2,
                epochs=2,
                hidden_units=4,
                batch_size=2,
                evaluation_names=["sequence_full"],
            )

            self.assertEqual(report["schema"], "elfquake.torch_sequence_split_holdout.v1")
            self.assertEqual(report["status"], "evaluated")
            self.assertEqual(report["train_row_count"], 4)
            self.assertEqual(report["test_row_count"], 2)
            self.assertEqual(list(report["evaluations"]), ["sequence_full"])

    @unittest.skipUnless(importlib.util.find_spec("torch"), "PyTorch optional dependency is not installed")
    def test_torch_patch_transformer_split_holdout_trains(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            aligned = root / "aligned.csv"
            aligned.write_text(
                "dataset_id,window_id,region_id,window_start_utc,window_end_utc,target_occurred,target_status,model_split\n"
                "seed1,w0,r,2026-01-01T00:00:00Z,2026-01-01T00:02:00Z,0,labeled,train\n"
                "seed1,w1,r,2026-01-01T00:02:00Z,2026-01-01T00:04:00Z,1,labeled,train\n"
                "seed1,w2,r,2026-01-01T00:04:00Z,2026-01-01T00:06:00Z,0,labeled,train\n"
                "seed1,w3,r,2026-01-01T00:06:00Z,2026-01-01T00:08:00Z,1,labeled,train\n"
                "seed1,w4,r,2026-01-01T00:08:00Z,2026-01-01T00:10:00Z,0,labeled,test\n"
                "seed1,w5,r,2026-01-01T00:10:00Z,2026-01-01T00:12:00Z,1,labeled,test\n",
                encoding="utf-8",
            )
            manifests = [
                self._write_sequence_fixture(root, "seed1", "synthetic_direct_avalanche", "avalanche_signal", [0, 0, 1, 1, 2, 2, 6, 7, 8, 2, 3, 9, 9]),
                self._write_sequence_fixture(root, "seed1", "synthetic_piezo_vlf", "piezo_signal", [0, 0, 1, 2, 3, 4, 8, 8, 7, 3, 2, 9, 10]),
                self._write_sequence_fixture(root, "seed1", "synthetic_summary", "topple_count", [1, 1, 2, 2, 3, 3, 8, 8, 8, 3, 3, 9, 9]),
            ]

            report = evaluate_torch_patch_transformer_split_holdout(
                input_csv=aligned,
                sequence_manifest_paths=manifests,
                out_path=root / "patch_transformer.json",
                lookback_steps=2,
                patch_steps=1,
                epochs=2,
                d_model=8,
                layers=1,
                heads=2,
                batch_size=2,
                evaluation_names=["sequence_full"],
            )

            self.assertEqual(report["schema"], "elfquake.torch_patch_transformer_split_holdout.v1")
            self.assertEqual(report["status"], "evaluated")
            self.assertEqual(report["d_model"], 8)
            self.assertEqual(report["patch_steps"], 1)
            self.assertEqual(list(report["evaluations"]), ["sequence_full"])

    @unittest.skipUnless(importlib.util.find_spec("torch"), "PyTorch optional dependency is not installed")
    def test_sequence_autoencoder_pretraining_writes_checkpoint_and_embeddings(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = self._write_sequence_fixture(
                root,
                "real",
                "real_vlf_image",
                "vlf_intensity_mean",
                [0, 1, 2, 3, 5, 8, 13, 21],
            )
            checkpoint = root / "autoencoder.pt"
            embeddings = root / "embeddings.csv"

            report = pretrain_sequence_autoencoder(
                sequence_manifest_path=manifest,
                out_path=root / "autoencoder.json",
                modality="real_vlf_image",
                lookback_steps=3,
                epochs=2,
                hidden_units=8,
                embedding_units=3,
                batch_size=2,
                checkpoint_out=checkpoint,
                embeddings_out=embeddings,
            )

            self.assertEqual(report["schema"], "elfquake.sequence_autoencoder_pretrain.v1")
            self.assertEqual(report["status"], "evaluated")
            self.assertEqual(report["modality"], "real_vlf_image")
            self.assertEqual(report["window_count"], 6)
            self.assertTrue(checkpoint.exists())
            self.assertTrue(embeddings.exists())

    @unittest.skipUnless(importlib.util.find_spec("torch"), "PyTorch optional dependency is not installed")
    def test_sequence_embedding_domain_comparison_uses_shared_real_encoder(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            real_manifest = self._write_sequence_fixture(
                root,
                "real",
                "real_vlf_image",
                "vlf_intensity_mean",
                [0, 1, 2, 3, 5, 8, 13, 21],
            )
            synthetic_manifest = self._write_sequence_fixture(
                root,
                "seed1",
                "synthetic_piezo_vlf",
                "piezo_signal",
                [0, 0, 1, 1, 2, 3, 5, 8],
            )
            embeddings = root / "domain_embeddings.csv"

            report = compare_sequence_embedding_domains(
                real_sequence_manifest_path=real_manifest,
                synthetic_sequence_manifest_paths=[synthetic_manifest],
                out_path=root / "domain.json",
                lookback_steps=3,
                epochs=2,
                hidden_units=8,
                embedding_units=3,
                batch_size=2,
                embeddings_out=embeddings,
            )

            self.assertEqual(report["schema"], "elfquake.sequence_embedding_domain_comparison.v1")
            self.assertEqual(report["status"], "evaluated")
            self.assertEqual(report["real_window_count"], 6)
            self.assertEqual(report["synthetic_window_count"], 6)
            self.assertEqual(report["descriptor_profile"], "shape")
            self.assertEqual(report["embedding_comparison"]["status"], "evaluated")
            self.assertGreater(report["synthetic_inlier_count"], 0)
            self.assertEqual(report["synthetic_inlier_embedding_comparison"]["status"], "evaluated")
            self.assertTrue(embeddings.exists())
            self.assertIn("is_synthetic_inlier", embeddings.read_text(encoding="utf-8"))

    @unittest.skipUnless(importlib.util.find_spec("torch"), "PyTorch optional dependency is not installed")
    def test_synthetic_inlier_transfer_evaluates_held_out_real_vlf(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            real_manifest = self._write_sequence_fixture(
                root,
                "real",
                "real_vlf_image",
                "vlf_intensity_mean",
                [0, 1, 2, 3, 5, 8, 13, 21],
            )
            synthetic_manifest = self._write_sequence_fixture(
                root,
                "seed1",
                "synthetic_piezo_vlf",
                "piezo_signal",
                [0, 0, 1, 1, 2, 3, 5, 8],
            )
            embeddings = root / "transfer_embeddings.csv"

            report = evaluate_synthetic_inlier_transfer(
                real_sequence_manifest_path=real_manifest,
                synthetic_sequence_manifest_paths=[synthetic_manifest],
                out_path=root / "transfer.json",
                lookback_steps=3,
                epochs=2,
                hidden_units=8,
                embedding_units=3,
                batch_size=2,
                embeddings_out=embeddings,
            )

            self.assertEqual(report["schema"], "elfquake.synthetic_inlier_transfer_reconstruction.v1")
            self.assertEqual(report["status"], "evaluated")
            self.assertEqual(report["real_window_count"], 6)
            self.assertEqual(report["synthetic_window_count"], 6)
            self.assertGreater(report["synthetic_inlier_count"], 0)
            self.assertIn("real_test_reconstruction", report)
            self.assertEqual(report["embedding_comparison"]["status"], "evaluated")
            self.assertTrue(embeddings.exists())
            self.assertIn("is_synthetic_inlier", embeddings.read_text(encoding="utf-8"))

    @unittest.skipUnless(importlib.util.find_spec("torch"), "PyTorch optional dependency is not installed")
    def test_mixed_domain_alignment_reports_controls_and_descriptor_gaps(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            real_manifest = self._write_sequence_fixture(
                root,
                "real",
                "real_vlf_image",
                "vlf_intensity_mean",
                [0, 1, 2, 3, 5, 8, 13, 21],
            )
            synthetic_manifest = self._write_sequence_fixture(
                root,
                "seed1",
                "synthetic_piezo_vlf",
                "piezo_signal",
                [0, 0, 1, 1, 2, 3, 5, 8],
            )
            embeddings = root / "mixed_embeddings.csv"

            report = evaluate_mixed_domain_alignment(
                real_sequence_manifest_path=real_manifest,
                synthetic_sequence_manifest_paths=[synthetic_manifest],
                out_path=root / "mixed.json",
                lookback_steps=3,
                inlier_method="local",
                control_methods=["random"],
                epochs=2,
                hidden_units=8,
                embedding_units=3,
                batch_size=2,
                embeddings_out=embeddings,
            )

            self.assertEqual(report["schema"], "elfquake.mixed_domain_alignment.v1")
            self.assertEqual(report["status"], "evaluated")
            self.assertEqual(report["selection_method"], "local")
            self.assertGreater(report["synthetic_train_count"], 0)
            self.assertIn("primary", report)
            self.assertIn("random", report["control_runs"])
            self.assertEqual(report["primary"]["embedding_comparison"]["status"], "evaluated")
            self.assertIn("largest_descriptor_gaps", report["descriptor_gap"])
            self.assertTrue(embeddings.exists())
            self.assertIn("is_synthetic_inlier", embeddings.read_text(encoding="utf-8"))

    @unittest.skipUnless(importlib.util.find_spec("torch"), "PyTorch optional dependency is not installed")
    def test_sequence_anomaly_scoring_writes_label_free_forecast(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = self._write_sequence_fixture(
                root,
                "real",
                "real_vlf_image",
                "vlf_intensity_mean",
                [0, 1, 1, 2, 3, 5, 8, 13, 21],
            )

            report = score_sequence_anomalies(
                sequence_manifest_path=manifest,
                out_path=root / "anomaly.json",
                scores_out=root / "anomaly_scores.csv",
                modality="real_vlf_image",
                lookback_steps=3,
                epochs=2,
                hidden_units=8,
                embedding_units=3,
                batch_size=2,
            )

            self.assertEqual(report["schema"], "elfquake.sequence_anomaly_forecast.v1")
            self.assertEqual(report["status"], "evaluated")
            self.assertEqual(report["forecast"]["status"], "label_free_smoke_forecast")
            self.assertEqual(report["forecast"]["horizon_days"], 7)
            self.assertIn("demo_probability", report["forecast"])
            self.assertTrue((root / "anomaly_scores.csv").exists())
            self.assertIn("anomaly_score", (root / "anomaly_scores.csv").read_text(encoding="utf-8").splitlines()[0])

    def test_synthetic_regime_annotation_can_drop_burn_in(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            table = root / "aligned.csv"
            table.write_text(
                "dataset_id,window_id,window_start_utc,window_end_utc,target_occurred\n"
                "seed1,w0,2026-01-01T00:00:00Z,2026-01-01T01:00:00Z,0\n"
                "seed1,w1,2026-01-01T01:00:00Z,2026-01-01T02:00:00Z,1\n"
                "seed1,w2,2026-01-01T02:00:00Z,2026-01-01T03:00:00Z,0\n"
                "seed1,w3,2026-01-01T03:00:00Z,2026-01-01T04:00:00Z,1\n"
                "seed2,w0,2026-01-01T00:00:00Z,2026-01-01T01:00:00Z,0\n"
                "seed2,w1,2026-01-01T01:00:00Z,2026-01-01T02:00:00Z,1\n"
                "seed2,w2,2026-01-01T02:00:00Z,2026-01-01T03:00:00Z,0\n"
                "seed2,w3,2026-01-01T03:00:00Z,2026-01-01T04:00:00Z,1\n",
                encoding="utf-8",
            )

            report = annotate_synthetic_regimes(
                input_csv=table,
                out_csv=root / "regimes.csv",
                report_path=root / "regimes.json",
                regime_count=2,
                burn_in_fraction=0.25,
                drop_burn_in=True,
            )

            self.assertEqual(report["row_count"], 8)
            self.assertEqual(report["output_row_count"], 6)
            lines = (root / "regimes.csv").read_text(encoding="utf-8").splitlines()
            self.assertIn("synthetic_regime_id", lines[0])
            self.assertNotIn(",w0,", "\n".join(lines[1:]))
            self.assertEqual(report["regime_ids"], ["seed1_r0", "seed1_r1", "seed2_r0", "seed2_r1"])

    def test_piezo_signal_transform_writes_derived_signal_csv(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "piezo.csv"
            source.write_text(
                "step,sensor_id,x,y,piezo_signal,piezo_release_total,max_stress_ratio\n"
                "0,0,1,1,0.0,0.0,0.70\n"
                "1,0,1,1,1.0,0.1,0.85\n"
                "2,0,1,1,1.5,0.2,0.95\n"
                "0,1,2,2,0.0,0.0,0.70\n"
                "1,1,2,2,0.5,0.1,0.90\n"
                "2,1,2,2,0.2,0.2,0.98\n",
                encoding="utf-8",
            )

            report = transform_piezo_signal_csv(
                input_csv=source,
                out_csv=root / "piezo_transformed.csv",
                report_path=root / "piezo_transformed.json",
                highpass_decay=0.8,
                envelope_decay=0.1,
                envelope_mix=0.2,
                burst_power=1.2,
                near_threshold_weight=1.0,
                release_mix=0.1,
                gain_contrast=0.1,
            )

            self.assertEqual(report["schema"], "elfquake.piezo_signal_transform.v1")
            self.assertEqual(report["row_count"], 6)
            self.assertEqual(report["sensor_count"], 2)
            transformed = (root / "piezo_transformed.csv").read_text(encoding="utf-8")
            self.assertIn("piezo_signal", transformed.splitlines()[0])
            self.assertNotEqual(source.read_text(encoding="utf-8"), transformed)
            self.assertTrue((root / "piezo_transformed.json").exists())

    def test_balanced_split_assigns_test_rows_by_group_and_label(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            table = root / "regimes.csv"
            table.write_text(
                "synthetic_regime_id,window_id,window_start_utc,target_occurred\n"
                "r1,w0,2026-01-01T00:00:00Z,0\n"
                "r1,w1,2026-01-01T01:00:00Z,0\n"
                "r1,w2,2026-01-01T02:00:00Z,1\n"
                "r1,w3,2026-01-01T03:00:00Z,1\n"
                "r2,w0,2026-01-01T00:00:00Z,0\n"
                "r2,w1,2026-01-01T01:00:00Z,0\n"
                "r2,w2,2026-01-01T02:00:00Z,1\n"
                "r2,w3,2026-01-01T03:00:00Z,1\n",
                encoding="utf-8",
            )

            report = assign_balanced_split(
                input_csv=table,
                out_csv=root / "split.csv",
                report_path=root / "split.json",
                test_fraction=0.5,
            )

            self.assertEqual(report["train_row_count"], 4)
            self.assertEqual(report["test_row_count"], 4)
            self.assertEqual(report["train_positive_count"], 2)
            self.assertEqual(report["test_positive_count"], 2)
            lines = (root / "split.csv").read_text(encoding="utf-8").splitlines()
            self.assertTrue(lines[0].endswith(",model_split"))

    def test_model_scale_estimate_reports_larger_model_gates(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            aligned = root / "aligned.csv"
            aligned.write_text(
                "dataset_id,window_id,region_id,window_start_utc,window_end_utc,target_occurred,feature_a\n"
                "seed1,w0,r,2026-01-01T00:00:00Z,2026-01-01T00:02:00Z,0,1.0\n"
                "seed1,w1,r,2026-01-01T00:02:00Z,2026-01-01T00:04:00Z,1,2.0\n"
                "seed2,w0,r,2026-01-01T00:00:00Z,2026-01-01T00:02:00Z,0,3.0\n"
                "seed2,w1,r,2026-01-01T00:02:00Z,2026-01-01T00:04:00Z,1,4.0\n",
                encoding="utf-8",
            )
            manifest = self._write_sequence_fixture(root, "seed1", "synthetic_piezo_vlf", "piezo_signal", [0, 1, 2, 3])

            report = estimate_model_scale(
                input_csv=aligned,
                out_path=root / "scale.json",
                sequence_manifest_paths=[manifest],
                lookback_steps=3,
            )

            self.assertEqual(report["schema"], "elfquake.model_scale_estimate.v1")
            self.assertEqual(report["labeled_row_count"], 4)
            self.assertEqual(report["positive_count"], 2)
            self.assertEqual(report["negative_count"], 2)
            self.assertEqual(report["sequence_feature_count"], 2)
            self.assertEqual(report["bytes_per_sequence_sample_float32"], 24)
            self.assertFalse(report["gates"]["small_transformer_synthetic"]["ready"])

    def test_temporal_split_diagnostics_reports_feature_drift(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            table = root / "aligned.csv"
            table.write_text(
                "window_id,window_start_utc,target_occurred,synthetic_seismic_event_count,quality_missing\n"
                "w1,2026-01-01T00:00:00Z,0,1,0\n"
                "w2,2026-01-01T01:00:00Z,1,3,0\n"
                "w3,2026-01-01T02:00:00Z,0,2,0\n"
                "w4,2026-01-01T03:00:00Z,1,4,0\n"
                "w5,2026-01-01T04:00:00Z,0,20,0\n"
                "w6,2026-01-01T05:00:00Z,1,30,\n",
                encoding="utf-8",
            )

            report = diagnose_temporal_split(
                input_csv=table,
                out_path=root / "diagnostics.json",
                feature_out=root / "features.csv",
                train_fraction=0.67,
            )

            self.assertEqual(report["schema"], "elfquake.temporal_split_diagnostics.v1")
            self.assertEqual(report["status"], "evaluated")
            self.assertEqual(report["train_row_count"], 4)
            self.assertEqual(report["test_row_count"], 2)
            self.assertEqual(report["top_feature_drifts"][0]["feature"], "synthetic_seismic_event_count")
            self.assertTrue((root / "diagnostics.json").exists())
            self.assertTrue((root / "features.csv").exists())

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
                "row_index,window_id,region_id,window_start_utc,window_end_utc,target_event_count,target_occurred,target_status\n"
                "0,w0,r,2026-01-01T00:00:00Z,2026-01-01T01:00:00Z,7,1,labeled\n"
                "1,w1,r,2026-01-01T01:00:00Z,2026-01-01T02:00:00Z,0,0,labeled\n",
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

            preserved_rows = build_aligned_window_dataset(
                base_manifest_path=base_manifest,
                sequence_manifest_paths=[sequence_manifest],
                tensor_manifest_paths=[timed_manifest],
                out_path=root / "aligned_preserved.csv",
            )

            self.assertEqual(preserved_rows[0]["target_event_count"], "7")
            self.assertEqual(preserved_rows[0]["target_occurred"], "1")
            self.assertEqual(preserved_rows[1]["target_occurred"], "0")
            self.assertEqual(preserved_rows[1]["target_status"], "labeled")

    def test_aligned_window_dataset_labels_any_event_in_future_horizon(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            base_dir = root / "base"
            base_dir.mkdir()
            (base_dir / "index.csv").write_text(
                "row_index,window_id,region_id,window_start_utc,window_end_utc\n"
                "0,w0,r,2026-01-01T00:00:00Z,2026-01-01T01:00:00Z\n"
                "1,w1,r,2026-01-01T01:00:00Z,2026-01-01T02:00:00Z\n"
                "2,w2,r,2026-01-01T02:00:00Z,2026-01-01T03:00:00Z\n"
                "3,w3,r,2026-01-01T03:00:00Z,2026-01-01T04:00:00Z\n",
                encoding="utf-8",
            )
            (base_dir / "values.csv").write_text(
                "row_index,synthetic_seismic_event_count\n"
                "0,0\n"
                "1,0\n"
                "2,0\n"
                "3,2\n",
                encoding="utf-8",
            )
            base_manifest = base_dir / "manifest.json"
            base_manifest.write_text(
                json.dumps(
                    {
                        "schema": "elfquake.tensor_dataset.v1",
                        "row_count": 4,
                        "feature_count": 1,
                        "values_csv": str(base_dir / "values.csv"),
                        "index_csv": str(base_dir / "index.csv"),
                        "feature_fields": ["synthetic_seismic_event_count"],
                    }
                ),
                encoding="utf-8",
            )

            rows = build_aligned_window_dataset(
                base_manifest_path=base_manifest,
                out_path=root / "aligned.csv",
                target_source_feature="synthetic_seismic_event_count",
                target_horizon_rows=3,
            )

            self.assertEqual(rows[0]["target_event_count"], "2")
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
    def test_sandpile_structured_initial_fill_starts_loaded(self) -> None:
        from elfquake.sim.sandpile import SandpileConfig, run_sandpile_simulation

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            summary_rows, _ = run_sandpile_simulation(
                config=SandpileConfig(
                    width=8,
                    height=8,
                    steps=2,
                    threshold=16,
                    source_count=3,
                    sensor_count=2,
                    deposition_probability=0.0,
                    seed=7,
                    initial_fill_mode="structured",
                    initial_fill_mean_height=6.0,
                    initial_fill_variation=1.5,
                    initial_fill_smooth_passes=2,
                ),
                summary_out=root / "summary.csv",
                sensors_out=root / "sensors.csv",
            )

            self.assertGreater(float(summary_rows[0]["mean_height"]), 4.0)
            self.assertGreater(int(summary_rows[0]["max_height"]), 0)

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
                max_events_values=[0, 1],
                event_bin_seconds=3600,
            )

            self.assertEqual(len(rows), 8)
            self.assertEqual(rows[0]["rank"], "1")
            self.assertTrue((root / "tuning.csv").exists())
            self.assertTrue(Path(rows[0]["events_file"]).exists())
            self.assertIn("normalized_distance", rows[0])
            self.assertIn("max_events", rows[0])

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

    @unittest.skipIf(importlib.util.find_spec("matplotlib") is None, "matplotlib not installed")
    def test_render_event_map_accepts_trial_forecast_magnitude_proxy(self) -> None:
        from elfquake.visualization.event_map import render_event_map

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            events = root / "trial.csv"
            events.write_text(
                "prediction_id,forecast_time_utc,latitude,longitude,magnitude_proxy,probability_proxy\n"
                "trial_001,2026-07-08T00:00:00Z,42.5,13.1,3.2,0.8\n",
                encoding="utf-8",
            )

            report = render_event_map(
                events_csv=events,
                out_path=root / "map.png",
                metadata_out=root / "map.json",
                min_magnitude=3.0,
                basemap_geojson=None,
            )

            self.assertEqual((root / "map.png").read_bytes()[:8], b"\x89PNG\r\n\x1a\n")
            self.assertEqual(report["event_count"], "1")

    @unittest.skipIf(importlib.util.find_spec("matplotlib") is None, "matplotlib not installed")
    def test_render_prediction_event_map_writes_actual_and_predicted_layers(self) -> None:
        from elfquake.visualization.prediction_map import render_prediction_event_map

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            events = root / "events.csv"
            events.write_text(
                "event_id,source,event_time_utc,latitude,longitude,depth_km,magnitude,magnitude_type,"
                "italy_region,event_location_name,event_type,raw_file,ingested_at_utc,raw_uri\n"
                "a,synthetic,2026-01-01T01:20:00Z,42.0,12.5,8,4.2,ML,central,Apennines,earthquake,,,\n",
                encoding="utf-8",
            )
            windows = root / "windows.csv"
            windows.write_text(
                "dataset_id,window_id,region_id,window_start_utc,window_end_utc,target_occurred,target_status,"
                "synthetic_seismic_event_count\n"
                "seed1,w0,r,2026-01-01T00:00:00Z,2026-01-01T01:00:00Z,1,labeled,0\n",
                encoding="utf-8",
            )
            report_json = root / "report.json"
            report_json.write_text(
                json.dumps(
                    {
                        "schema": "elfquake.torch_tabular_group_holdout.v1",
                        "group_field": "dataset_id",
                        "test_group": "seed1",
                        "evaluations": {
                            "synthetic_full": {
                                "status": "evaluated",
                                "calibrated_threshold": 0.5,
                                "calibrated_test_metrics": {"balanced_accuracy": 1.0},
                                "test_probabilities": [0.9],
                            }
                        },
                    }
                ),
                encoding="utf-8",
            )

            report = render_prediction_event_map(
                events_csv=events,
                windows_csv=windows,
                report_json=report_json,
                out_path=root / "prediction_map.png",
                metadata_out=root / "prediction_map.json",
                basemap_geojson=None,
            )

            self.assertEqual((root / "prediction_map.png").read_bytes()[:8], b"\x89PNG\r\n\x1a\n")
            metadata = json.loads((root / "prediction_map.json").read_text(encoding="utf-8"))
            self.assertEqual(metadata["actual_event_count"], "1")
            self.assertEqual(metadata["predicted_positive_window_count"], "1")
            self.assertEqual(metadata["predicted_event_point_count"], "1")
            self.assertEqual(report["evaluation"], "synthetic_full")

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


def _write_sequence_report(path: Path, *, epochs: int, best_name: str, best_score: float) -> None:
    other_name = "sequence_direct_avalanche_only"
    if best_name == other_name:
        other_name = "sequence_piezo_vlf_only"
    path.write_text(
        json.dumps(
            {
                "schema": "elfquake.torch_sequence_group_holdout.v1",
                "epochs": epochs,
                "lookback_steps": 60,
                "hidden_units": 24,
                "learning_rate": 0.001,
                "evaluations": {
                    best_name: {
                        "status": "evaluated",
                        "feature_count": 3,
                        "train_row_count": 8,
                        "test_row_count": 2,
                        "dropped_train_row_count": 0,
                        "dropped_test_row_count": 0,
                        "test_metrics": {"balanced_accuracy": best_score - 0.01},
                        "calibrated_test_metrics": {"balanced_accuracy": best_score},
                        "calibrated_threshold": 0.5,
                    },
                    other_name: {
                        "status": "evaluated",
                        "feature_count": 3,
                        "train_row_count": 8,
                        "test_row_count": 2,
                        "dropped_train_row_count": 0,
                        "dropped_test_row_count": 0,
                        "test_metrics": {"balanced_accuracy": 0.5},
                        "calibrated_test_metrics": {"balanced_accuracy": 0.5},
                        "calibrated_threshold": 0.5,
                    },
                },
            }
        ),
        encoding="utf-8",
    )


def _fake_jpeg_capture(url: str) -> HttpCapture:
    return HttpCapture(
        url=url,
        status=200,
        captured_at_utc=datetime(2026, 6, 29, 9, 57, 24, tzinfo=timezone.utc),
        headers={"Last-Modified": "Mon, 29 Jun 2026 09:45:00 GMT", "Content-Type": "image/jpeg"},
        body=b"jpeg",
    )


def _fake_ogg_stream_capture(url: str, duration_seconds: int, max_bytes: int | None) -> HttpCapture:
    del duration_seconds, max_bytes
    return HttpCapture(
        url=url,
        status=200,
        captured_at_utc=datetime(2026, 7, 5, 10, 40, 0, tzinfo=timezone.utc),
        headers={"Content-Type": "audio/ogg"},
        body=b"OggSfake",
    )


def _fake_empty_ogg_stream_capture(url: str, duration_seconds: int, max_bytes: int | None) -> HttpCapture:
    del duration_seconds, max_bytes
    return HttpCapture(
        url=url,
        status=200,
        captured_at_utc=datetime(2026, 7, 5, 10, 40, 0, tzinfo=timezone.utc),
        headers={"Content-Type": "audio/ogg"},
        body=b"",
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

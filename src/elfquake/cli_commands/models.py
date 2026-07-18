"""Model and tensor-interface CLI commands."""

from __future__ import annotations

from argparse import Namespace, _SubParsersAction
import json
from pathlib import Path

from elfquake.cli_commands.common import print_holdout_report
from elfquake.cli_commands.torch_models import register_torch_model_commands
from elfquake.models.ablation_smoke import train_ablation_smoke
from elfquake.models.aligned_windows import build_aligned_window_dataset
from elfquake.models.alignment_manifest import build_alignment_manifest
from elfquake.models.candidates import write_model_candidates
from elfquake.models.dataset_combine import combine_aligned_datasets
from elfquake.models.event_catalog_alignment import (
    calibrate_synthetic_catalog,
    calibrate_synthetic_magnitudes,
    calibrate_synthetic_spatial_coordinates,
    balance_synthetic_episode_rates,
    combine_synthetic_catalogs,
    compare_event_catalogs,
)
from elfquake.models.forecast_comparison import compare_weekly_forecasts
from elfquake.models.interface_shape import audit_model_interfaces
from elfquake.models.learned_forecast import generate_learned_weekly_event_forecast
from elfquake.models.logistic_smoke import train_logistic_smoke
from elfquake.models.model_scale import estimate_model_scale
from elfquake.models.readiness import summarize_model_readiness
from elfquake.models.real_transfer_trial import run_real_transfer_trial
from elfquake.models.transfer_experiments import run_transfer_experiment_suite
from elfquake.models.report_summary import summarize_model_run_reports
from elfquake.models.sequence_materializer import materialize_sequence_dataset
from elfquake.models.spatial_permutation import permute_spatial_target_vectors
from elfquake.models.split_diagnostics import diagnose_temporal_split
from elfquake.models.synthetic_drift import diagnose_synthetic_drift
from elfquake.models.synthetic_episodes import annotate_synthetic_episodes
from elfquake.models.synthetic_event_list_model import train_synthetic_event_list_model
from elfquake.models.synthetic_event_list_probes import summarize_synthetic_event_list_probes
from elfquake.models.synthetic_event_list_sequence import (
    ensemble_synthetic_event_list_sequence_heads,
    summarize_synthetic_event_list_sequence_heads,
    train_synthetic_event_list_sequence_head,
)
from elfquake.models.synthetic_event_list_targets import build_synthetic_event_list_targets
from elfquake.models.synthetic_lagged_context import build_synthetic_lagged_context
from elfquake.models.synthetic_regimes import annotate_synthetic_regimes, assign_balanced_split
from elfquake.models.tensor_materializer import materialize_tensor_dataset
from elfquake.models.tensor_spec import build_tensor_spec
from elfquake.models.temporal_holdout import evaluate_group_holdout, evaluate_temporal_holdout
from elfquake.models.transformer_input_adapter import prepare_transformer_target_input
from elfquake.models.synthetic_step_targets import build_synthetic_step_targets
from elfquake.models.trial_forecast import generate_trial_weekly_event_forecast
from elfquake.models.window_adapter import build_event_window_features


def register_model_commands(subparsers: _SubParsersAction) -> None:
    trainer = subparsers.add_parser("train-logistic-smoke")
    trainer.add_argument("--design-matrix", type=Path, required=True)
    trainer.add_argument("--out", type=Path, required=True)
    trainer.add_argument("--epochs", type=int, default=600)
    trainer.add_argument("--learning-rate", type=float, default=0.2)
    trainer.set_defaults(func=_train_logistic_smoke)

    readiness = subparsers.add_parser("summarize-model-readiness")
    readiness.add_argument("--input", type=Path, required=True)
    readiness.add_argument("--out", type=Path, required=True)
    readiness.set_defaults(func=_summarize_model_readiness)

    permutation = subparsers.add_parser("permute-spatial-targets")
    permutation.add_argument("--input", type=Path, required=True)
    permutation.add_argument("--out", type=Path, required=True)
    permutation.add_argument("--seed", type=int, default=42)
    permutation.add_argument("--time-field", default="window_start_utc")
    permutation.set_defaults(func=_permute_spatial_targets)

    catalog_compare = subparsers.add_parser("compare-event-catalogs")
    catalog_compare.add_argument("--real-events", type=Path, required=True)
    catalog_compare.add_argument("--synthetic-events", type=Path, action="append", required=True)
    catalog_compare.add_argument("--out", type=Path, required=True)
    catalog_compare.add_argument("--cell-degrees", type=float, default=1.5)
    catalog_compare.add_argument("--synthetic-duration-days", type=float, action="append")
    catalog_compare.set_defaults(func=_compare_event_catalogs)

    magnitude_calibration = subparsers.add_parser("calibrate-synthetic-magnitudes")
    magnitude_calibration.add_argument("--real-events", type=Path, required=True)
    magnitude_calibration.add_argument("--synthetic-events", type=Path, required=True)
    magnitude_calibration.add_argument("--out", type=Path, required=True)
    magnitude_calibration.set_defaults(func=_calibrate_synthetic_magnitudes)

    catalog_calibration = subparsers.add_parser("calibrate-synthetic-catalog")
    catalog_calibration.add_argument("--real-events", type=Path, required=True)
    catalog_calibration.add_argument("--synthetic-events", type=Path, required=True)
    catalog_calibration.add_argument("--out", type=Path, required=True)
    catalog_calibration.add_argument("--report", type=Path, required=True)
    catalog_calibration.add_argument("--seed", type=int, default=42)
    catalog_calibration.add_argument("--synthetic-duration-days", type=float)
    catalog_calibration.set_defaults(func=_calibrate_synthetic_catalog)

    spatial_calibration = subparsers.add_parser("calibrate-synthetic-spatial")
    spatial_calibration.add_argument("--real-events", type=Path, required=True)
    spatial_calibration.add_argument("--synthetic-events", type=Path, required=True)
    spatial_calibration.add_argument("--out", type=Path, required=True)
    spatial_calibration.add_argument("--report", type=Path, required=True)
    spatial_calibration.set_defaults(func=_calibrate_synthetic_spatial)

    combine_catalogs = subparsers.add_parser("combine-synthetic-catalogs")
    combine_catalogs.add_argument("--synthetic-events", type=Path, action="append", required=True)
    combine_catalogs.add_argument("--out", type=Path, required=True)
    combine_catalogs.add_argument("--report", type=Path, required=True)
    combine_catalogs.add_argument("--offset-days", type=int, default=21)
    combine_catalogs.set_defaults(func=_combine_synthetic_catalogs)

    episode_balance = subparsers.add_parser("balance-synthetic-episode-rates")
    episode_balance.add_argument("--real-events", type=Path, required=True)
    episode_balance.add_argument("--synthetic-events", type=Path, required=True)
    episode_balance.add_argument("--episode-duration-days", type=float, action="append", required=True)
    episode_balance.add_argument("--out", type=Path, required=True)
    episode_balance.add_argument("--report", type=Path, required=True)
    episode_balance.add_argument("--seed", type=int, default=42)
    episode_balance.set_defaults(func=_balance_synthetic_episode_rates)

    ablation = subparsers.add_parser("train-ablation-smoke")
    ablation.add_argument("--input", type=Path, required=True)
    ablation.add_argument("--out", type=Path, required=True)
    ablation.add_argument("--epochs", type=int, default=600)
    ablation.add_argument("--learning-rate", type=float, default=0.2)
    ablation.set_defaults(func=_train_ablation_smoke)

    register_torch_model_commands(subparsers)

    temporal_holdout = subparsers.add_parser("evaluate-temporal-holdout")
    temporal_holdout.add_argument("--input", type=Path, required=True)
    temporal_holdout.add_argument("--out", type=Path, required=True)
    temporal_holdout.add_argument("--time-field", default="window_start_utc")
    temporal_holdout.add_argument("--train-fraction", type=float, default=0.8)
    temporal_holdout.add_argument("--epochs", type=int, default=600)
    temporal_holdout.add_argument("--learning-rate", type=float, default=0.2)
    temporal_holdout.add_argument("--group-by-time", action="store_true")
    temporal_holdout.set_defaults(func=_evaluate_temporal_holdout)

    split_diagnostics = subparsers.add_parser("diagnose-temporal-split")
    split_diagnostics.add_argument("--input", type=Path, required=True)
    split_diagnostics.add_argument("--out", type=Path, required=True)
    split_diagnostics.add_argument("--feature-out", type=Path)
    split_diagnostics.add_argument("--time-field", default="window_start_utc")
    split_diagnostics.add_argument("--train-fraction", type=float, default=0.8)
    split_diagnostics.add_argument("--target-field", default="target_occurred")
    split_diagnostics.add_argument("--top-n", type=int, default=20)
    split_diagnostics.set_defaults(func=_diagnose_temporal_split)

    synthetic_regimes = subparsers.add_parser("annotate-synthetic-regimes")
    synthetic_regimes.add_argument("--input", type=Path, required=True)
    synthetic_regimes.add_argument("--out", type=Path, required=True)
    synthetic_regimes.add_argument("--report", type=Path, required=True)
    synthetic_regimes.add_argument("--group-field", default="dataset_id")
    synthetic_regimes.add_argument("--time-field", default="window_start_utc")
    synthetic_regimes.add_argument("--regime-count", type=int, default=5)
    synthetic_regimes.add_argument("--burn-in-fraction", type=float, default=0.2)
    synthetic_regimes.add_argument("--drop-burn-in", action="store_true")
    synthetic_regimes.set_defaults(func=_annotate_synthetic_regimes)

    balanced_split = subparsers.add_parser("assign-balanced-split")
    balanced_split.add_argument("--input", type=Path, required=True)
    balanced_split.add_argument("--out", type=Path, required=True)
    balanced_split.add_argument("--report", type=Path, required=True)
    balanced_split.add_argument("--group-field", default="synthetic_regime_id")
    balanced_split.add_argument("--target-field", default="target_occurred")
    balanced_split.add_argument("--time-field", default="window_start_utc")
    balanced_split.add_argument("--split-field", default="model_split")
    balanced_split.add_argument("--test-fraction", type=float, default=0.2)
    balanced_split.set_defaults(func=_assign_balanced_split)

    transformer_input = subparsers.add_parser("prepare-transformer-target-input")
    transformer_input.add_argument("--input", type=Path, required=True)
    transformer_input.add_argument("--out", type=Path, required=True)
    transformer_input.add_argument("--report", type=Path, required=True)
    transformer_input.add_argument("--target-field", default="eventlist_target_occurred")
    transformer_input.add_argument("--target-status-field", default="eventlist_target_status")
    transformer_input.add_argument("--standard-target-field", default="target_occurred")
    transformer_input.add_argument("--standard-status-field", default="target_status")
    transformer_input.add_argument("--split-field", default="model_split")
    transformer_input.add_argument("--group-field", default="dataset_id")
    transformer_input.add_argument("--time-field", default="window_start_utc")
    transformer_input.add_argument("--train-fraction", type=float, default=0.8)
    transformer_input.set_defaults(func=_prepare_transformer_target_input)

    step_targets = subparsers.add_parser("build-synthetic-step-targets")
    step_targets.add_argument("--piezo", type=Path, action="append", required=True)
    step_targets.add_argument("--events", type=Path, action="append", required=True)
    step_targets.add_argument("--out", type=Path, required=True)
    step_targets.add_argument("--report", type=Path, required=True)
    step_targets.add_argument("--horizon-steps", type=int, default=15)
    step_targets.add_argument("--stride-steps", type=int, default=5)
    step_targets.add_argument("--start-time-utc", default="2026-01-01T00:00:00Z")
    step_targets.add_argument("--step-seconds", type=int, default=60)
    step_targets.set_defaults(func=_build_synthetic_step_targets)

    group_holdout = subparsers.add_parser("evaluate-group-holdout")
    group_holdout.add_argument("--input", type=Path, required=True)
    group_holdout.add_argument("--out", type=Path, required=True)
    group_holdout.add_argument("--group-field", default="dataset_id")
    group_holdout.add_argument("--test-group", required=True)
    group_holdout.add_argument("--epochs", type=int, default=600)
    group_holdout.add_argument("--learning-rate", type=float, default=0.2)
    group_holdout.set_defaults(func=_evaluate_group_holdout)

    model_run_summary = subparsers.add_parser("summarize-model-run-reports")
    model_run_summary.add_argument("--report", type=Path, action="append", required=True)
    model_run_summary.add_argument("--out", type=Path, required=True)
    model_run_summary.set_defaults(func=_summarize_model_run_reports)

    model_candidates = subparsers.add_parser("list-model-candidates")
    model_candidates.add_argument("--out", type=Path, required=True)
    model_candidates.add_argument("--stage", choices=["baseline", "transformer", "research"])
    model_candidates.set_defaults(func=_list_model_candidates)

    model_scale = subparsers.add_parser("estimate-model-scale")
    model_scale.add_argument("--input", type=Path, required=True)
    model_scale.add_argument("--out", type=Path, required=True)
    model_scale.add_argument("--sequence-manifest", type=Path, action="append", default=[])
    model_scale.add_argument("--target-field", default="target_occurred")
    model_scale.add_argument("--group-field", default="dataset_id")
    model_scale.add_argument("--lookback-steps", type=int, default=60)
    model_scale.add_argument("--no-missing-masks", action="store_true")
    model_scale.set_defaults(func=_estimate_model_scale)

    tensor_spec = subparsers.add_parser("build-tensor-spec")
    tensor_spec.add_argument("--input", type=Path, required=True)
    tensor_spec.add_argument("--out", type=Path, required=True)
    tensor_spec.add_argument("--time-field", default="window_start_utc")
    tensor_spec.add_argument("--region-field", default="region_id")
    tensor_spec.add_argument("--target-field", default="target_occurred")
    tensor_spec.set_defaults(func=_build_tensor_spec)

    tensor_materialize = subparsers.add_parser("materialize-tensor-dataset")
    tensor_materialize.add_argument("--spec", type=Path, required=True)
    tensor_materialize.add_argument("--out-dir", type=Path, required=True)
    tensor_materialize.add_argument("--fill-value", type=float, default=0.0)
    tensor_materialize.set_defaults(func=_materialize_tensor_dataset)

    interface_audit = subparsers.add_parser("audit-model-interfaces")
    interface_audit.add_argument("--input", type=Path, action="append", required=True)
    interface_audit.add_argument("--out", type=Path, required=True)
    interface_audit.set_defaults(func=_audit_model_interfaces)

    event_windows = subparsers.add_parser("build-event-window-features")
    event_windows.add_argument("--events", type=Path, required=True)
    event_windows.add_argument("--out", type=Path, required=True)
    event_windows.add_argument("--region-id", required=True)
    event_windows.add_argument("--start-utc", required=True)
    event_windows.add_argument("--end-utc", required=True)
    event_windows.add_argument("--window-seconds", type=int, required=True)
    event_windows.add_argument("--feature-prefix", default="seismic")
    event_windows.add_argument("--min-magnitude", type=float)
    event_windows.set_defaults(func=_build_event_window_features)

    sequence_materialize = subparsers.add_parser("materialize-sequence-dataset")
    sequence_materialize.add_argument("--input", type=Path, required=True)
    sequence_materialize.add_argument("--out-dir", type=Path, required=True)
    sequence_materialize.add_argument("--time-field", default="step")
    sequence_materialize.add_argument("--entity-field", default="sensor_id")
    sequence_materialize.add_argument("--no-entity-field", action="store_true")
    sequence_materialize.add_argument("--fill-value", type=float, default=0.0)
    sequence_materialize.add_argument("--modality", default="simulation")
    sequence_materialize.add_argument("--time-start-utc")
    sequence_materialize.add_argument("--time-step-seconds", type=int)
    sequence_materialize.set_defaults(func=_materialize_sequence_dataset)

    alignment_manifest = subparsers.add_parser("build-alignment-manifest")
    alignment_manifest.add_argument("--manifest", type=Path, action="append", required=True)
    alignment_manifest.add_argument("--run-id", required=True)
    alignment_manifest.add_argument("--out", type=Path, required=True)
    alignment_manifest.set_defaults(func=_build_alignment_manifest)

    aligned_windows = subparsers.add_parser("build-aligned-window-dataset")
    aligned_windows.add_argument("--base-manifest", type=Path, required=True)
    aligned_windows.add_argument("--sequence-manifest", type=Path, action="append", default=[])
    aligned_windows.add_argument("--tensor-manifest", type=Path, action="append", default=[])
    aligned_windows.add_argument("--out", type=Path, required=True)
    aligned_windows.add_argument("--target-source-feature", default="")
    aligned_windows.add_argument("--target-horizon-rows", type=int, default=1)
    aligned_windows.add_argument("--target-threshold", type=float, default=0.0)
    aligned_windows.add_argument("--drop-unlabeled-targets", action="store_true")
    aligned_windows.set_defaults(func=_build_aligned_window_dataset)

    combine_aligned = subparsers.add_parser("combine-aligned-datasets")
    combine_aligned.add_argument("--input", type=Path, action="append", required=True)
    combine_aligned.add_argument("--dataset-id", action="append")
    combine_aligned.add_argument("--out", type=Path, required=True)
    combine_aligned.set_defaults(func=_combine_aligned_datasets)

    trial_forecast = subparsers.add_parser("generate-trial-weekly-event-forecast")
    trial_forecast.add_argument("--real-events", type=Path, required=True)
    trial_forecast.add_argument("--out", type=Path, required=True)
    trial_forecast.add_argument("--events-out", type=Path, required=True)
    trial_forecast.add_argument("--as-of-utc", required=True)
    trial_forecast.add_argument("--horizon-days", type=int, default=7)
    trial_forecast.add_argument("--magnitude-threshold", type=float, default=2.0)
    trial_forecast.add_argument("--max-events", type=int, default=25)
    trial_forecast.add_argument("--seed", type=int, default=42)
    trial_forecast.add_argument("--synthetic-event-glob", action="append", default=[])
    trial_forecast.add_argument("--vlf-window", type=Path, action="append", default=[])
    trial_forecast.add_argument("--vlf-anomaly-report", type=Path)
    trial_forecast.add_argument("--vlf-audio-glob", action="append", default=[])
    trial_forecast.add_argument("--astronomy-glob", action="append", default=[])
    trial_forecast.set_defaults(func=_generate_trial_weekly_event_forecast)

    learned_forecast = subparsers.add_parser("generate-learned-weekly-event-forecast")
    learned_forecast.add_argument("--real-events", type=Path, required=True)
    learned_forecast.add_argument("--synthetic-windows", type=Path, required=True)
    learned_forecast.add_argument("--out", type=Path, required=True)
    learned_forecast.add_argument("--events-out", type=Path, required=True)
    learned_forecast.add_argument("--as-of-utc", required=True)
    learned_forecast.add_argument("--horizon-days", type=int, default=7)
    learned_forecast.add_argument("--magnitude-threshold", type=float, default=2.0)
    learned_forecast.add_argument("--max-events", type=int, default=25)
    learned_forecast.add_argument("--seed", type=int, default=42)
    learned_forecast.add_argument("--train-fraction", type=float, default=0.8)
    learned_forecast.add_argument("--epochs", type=int, default=500)
    learned_forecast.add_argument("--learning-rate", type=float, default=0.08)
    learned_forecast.add_argument("--l2", type=float, default=0.001)
    learned_forecast.add_argument("--synthetic-event-glob", action="append", default=[])
    learned_forecast.add_argument("--vlf-window", type=Path, action="append", default=[])
    learned_forecast.add_argument("--vlf-anomaly-report", type=Path)
    learned_forecast.add_argument("--vlf-audio-glob", action="append", default=[])
    learned_forecast.add_argument("--astronomy-glob", action="append", default=[])
    learned_forecast.set_defaults(func=_generate_learned_weekly_event_forecast)

    transfer_trial = subparsers.add_parser("run-real-transfer-trial")
    transfer_trial.add_argument("--real-events", type=Path, required=True)
    transfer_trial.add_argument("--synthetic-events", type=Path, action="append", required=True)
    transfer_trial.add_argument("--out-dir", type=Path, required=True)
    transfer_trial.add_argument("--magnitude-threshold", type=float, default=2.5)
    transfer_trial.add_argument("--horizon-days", type=int, default=7)
    transfer_trial.add_argument("--cell-degrees", type=float, default=1.5)
    transfer_trial.add_argument("--train-fraction", type=float, default=0.8)
    transfer_trial.add_argument("--pretrain-epochs", type=int, default=30)
    transfer_trial.add_argument("--finetune-epochs", type=int, default=80)
    transfer_trial.add_argument("--seed", type=int, default=42)
    transfer_trial.set_defaults(func=_run_real_transfer_trial)

    transfer_suite = subparsers.add_parser("run-transfer-experiment-suite")
    transfer_suite.add_argument("--real-events", type=Path, required=True)
    transfer_suite.add_argument("--synthetic-events", type=Path, action="append", required=True)
    transfer_suite.add_argument("--out", type=Path, required=True)
    transfer_suite.add_argument("--magnitude-threshold", type=float, default=2.5)
    transfer_suite.add_argument("--horizon-days", type=int, default=7)
    transfer_suite.add_argument("--cell-degrees", type=float, default=1.5)
    transfer_suite.add_argument("--train-fraction", type=float, default=0.8)
    transfer_suite.add_argument("--epochs", type=int, default=50)
    transfer_suite.add_argument("--pretrain-epochs", type=int, default=30)
    transfer_suite.add_argument("--seed", type=int, default=42)
    transfer_suite.set_defaults(func=_run_transfer_experiment_suite)

    forecast_compare = subparsers.add_parser("compare-weekly-forecasts")
    forecast_compare.add_argument("--baseline-report", type=Path, required=True)
    forecast_compare.add_argument("--baseline-events", type=Path, required=True)
    forecast_compare.add_argument("--candidate-report", type=Path, required=True)
    forecast_compare.add_argument("--candidate-events", type=Path, required=True)
    forecast_compare.add_argument("--out", type=Path, required=True)
    forecast_compare.add_argument("--csv-out", type=Path)
    forecast_compare.set_defaults(func=_compare_weekly_forecasts)

    event_list_targets = subparsers.add_parser("build-synthetic-event-list-targets")
    event_list_targets.add_argument("--input", type=Path, required=True)
    event_list_targets.add_argument("--out", type=Path, required=True)
    event_list_targets.add_argument("--report", type=Path, required=True)
    event_list_targets.add_argument("--horizon-rows", type=int, default=24)
    event_list_targets.add_argument("--magnitude-threshold", type=float, default=2.0)
    event_list_targets.add_argument("--group-field", default="dataset_id")
    event_list_targets.add_argument("--source-field", default="source_file")
    event_list_targets.set_defaults(func=_build_synthetic_event_list_targets)

    event_list_model = subparsers.add_parser("train-synthetic-event-list-model")
    event_list_model.add_argument("--input", type=Path, required=True)
    event_list_model.add_argument("--out", type=Path, required=True)
    event_list_model.add_argument("--predictions-out", type=Path)
    event_list_model.add_argument("--train-fraction", type=float, default=0.8)
    event_list_model.add_argument("--split-field", default="")
    event_list_model.add_argument("--epochs", type=int, default=600)
    event_list_model.add_argument("--learning-rate", type=float, default=0.05)
    event_list_model.add_argument("--l2", type=float, default=0.001)
    event_list_model.add_argument("--seed", type=int, default=42)
    event_list_model.add_argument("--max-feature-count", type=int, default=0)
    event_list_model.add_argument("--occurrence-model-type", choices=["logistic_ensemble", "boosted_stumps"], default="logistic_ensemble")
    event_list_model.add_argument("--occurrence-ensemble-count", type=int, default=1)
    event_list_model.add_argument("--occurrence-feature-bag-fraction", type=float, default=1.0)
    event_list_model.add_argument("--occurrence-stump-count", type=int, default=24)
    event_list_model.set_defaults(func=_train_synthetic_event_list_model)

    event_list_probe_summary = subparsers.add_parser("summarize-synthetic-event-list-probes")
    event_list_probe_summary.add_argument("--root", type=Path, required=True)
    event_list_probe_summary.add_argument("--out", type=Path, required=True)
    event_list_probe_summary.add_argument("--csv-out", type=Path)
    event_list_probe_summary.set_defaults(func=_summarize_synthetic_event_list_probes)

    lagged_context = subparsers.add_parser("build-synthetic-lagged-context")
    lagged_context.add_argument("--input", type=Path, required=True)
    lagged_context.add_argument("--out", type=Path, required=True)
    lagged_context.add_argument("--report", type=Path, required=True)
    lagged_context.add_argument("--lag", type=int, action="append", default=[])
    lagged_context.add_argument("--group-field", default="dataset_id")
    lagged_context.add_argument("--time-field", default="window_start_utc")
    lagged_context.set_defaults(func=_build_synthetic_lagged_context)

    event_list_sequence = subparsers.add_parser("train-synthetic-event-list-sequence-head")
    event_list_sequence.add_argument("--input", type=Path, required=True)
    event_list_sequence.add_argument("--out", type=Path, required=True)
    event_list_sequence.add_argument("--predictions-out", type=Path)
    event_list_sequence.add_argument("--target-field", default="eventlist_target_occurred")
    event_list_sequence.add_argument("--target-status-field", default="eventlist_target_status")
    event_list_sequence.add_argument("--group-field", default="dataset_id")
    event_list_sequence.add_argument("--time-field", default="window_start_utc")
    event_list_sequence.add_argument("--train-fraction", type=float, default=0.8)
    event_list_sequence.add_argument("--lookback-rows", type=int, default=12)
    event_list_sequence.add_argument("--epochs", type=int, default=80)
    event_list_sequence.add_argument("--learning-rate", type=float, default=0.001)
    event_list_sequence.add_argument("--hidden-units", type=int, default=24)
    event_list_sequence.add_argument("--batch-size", type=int, default=64)
    event_list_sequence.add_argument("--dropout", type=float, default=0.1)
    event_list_sequence.add_argument("--weight-decay", type=float, default=0.001)
    event_list_sequence.add_argument("--max-feature-count", type=int, default=256)
    event_list_sequence.add_argument("--validation-fraction", type=float, default=0.0)
    event_list_sequence.add_argument("--early-stopping-patience", type=int, default=0)
    event_list_sequence.add_argument("--calibration-source", choices=["auto", "train", "validation"], default="auto")
    event_list_sequence.add_argument("--seed", type=int, default=42)
    event_list_sequence.set_defaults(func=_train_synthetic_event_list_sequence_head)

    event_list_sequence_summary = subparsers.add_parser("summarize-synthetic-event-list-sequence-heads")
    event_list_sequence_summary.add_argument("--root", type=Path, required=True)
    event_list_sequence_summary.add_argument("--out", type=Path, required=True)
    event_list_sequence_summary.add_argument("--csv-out", type=Path)
    event_list_sequence_summary.set_defaults(func=_summarize_synthetic_event_list_sequence_heads)

    event_list_sequence_ensemble = subparsers.add_parser("ensemble-synthetic-event-list-sequence-heads")
    event_list_sequence_ensemble.add_argument("--report", type=Path, action="append", required=True)
    event_list_sequence_ensemble.add_argument("--out", type=Path, required=True)
    event_list_sequence_ensemble.add_argument("--predictions-out", type=Path)
    event_list_sequence_ensemble.set_defaults(func=_ensemble_synthetic_event_list_sequence_heads)

    drift = subparsers.add_parser("diagnose-synthetic-drift")
    drift.add_argument("--input", type=Path, required=True)
    drift.add_argument("--out", type=Path, required=True)
    drift.add_argument("--csv-out", type=Path)
    drift.add_argument("--target-field", default="eventlist_target_occurred")
    drift.add_argument("--target-status-field", default="eventlist_target_status")
    drift.add_argument("--group-field", default="dataset_id")
    drift.add_argument("--time-field", default="window_start_utc")
    drift.add_argument("--train-fraction", type=float, default=0.8)
    drift.add_argument("--bucket-count", type=int, default=10)
    drift.add_argument("--top-n", type=int, default=20)
    drift.set_defaults(func=_diagnose_synthetic_drift)

    episodes = subparsers.add_parser("annotate-synthetic-episodes")
    episodes.add_argument("--input", type=Path, required=True)
    episodes.add_argument("--out", type=Path, required=True)
    episodes.add_argument("--report", type=Path, required=True)
    episodes.add_argument("--group-field", default="dataset_id")
    episodes.add_argument("--time-field", default="window_start_utc")
    episodes.add_argument("--rows-per-episode", type=int, default=24)
    episodes.add_argument("--target-field", default="eventlist_target_occurred")
    episodes.add_argument("--drop-partial", action="store_true")
    episodes.set_defaults(func=_annotate_synthetic_episodes)


def _train_logistic_smoke(args: Namespace) -> int:
    report = train_logistic_smoke(
        design_matrix_csv=args.design_matrix,
        out_path=args.out,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
    )
    print(f"status: {report['status']}")
    print(f"output: {args.out}")
    return 0


def _summarize_model_readiness(args: Namespace) -> int:
    report = summarize_model_readiness(input_csv=args.input, out_path=args.out)
    print(f"status: {report['status']}")
    print(f"rows: {report['row_count']}")
    print(f"labeled rows: {report['labeled_row_count']}")
    print(f"output: {args.out}")
    return 0


def _permute_spatial_targets(args: Namespace) -> int:
    report = permute_spatial_target_vectors(
        input_csv=args.input, out_path=args.out, seed=args.seed, time_field=args.time_field
    )
    print(f"rows: {report['row_count']}")
    print(f"labeled time groups: {report['labeled_time_count']}")
    print(f"output: {args.out}")
    return 0


def _compare_event_catalogs(args: Namespace) -> int:
    report = compare_event_catalogs(
        real_events=args.real_events,
        synthetic_events=args.synthetic_events,
        out_path=args.out,
        cell_degrees=args.cell_degrees,
        synthetic_duration_days=args.synthetic_duration_days,
    )
    print(f"catalogs: {len(report['catalogs'])}")
    print(f"output: {args.out}")
    return 0


def _calibrate_synthetic_magnitudes(args: Namespace) -> int:
    report = calibrate_synthetic_magnitudes(
        real_events=args.real_events, synthetic_events=args.synthetic_events, out_path=args.out
    )
    print(f"real events: {report['real_event_count']}")
    print(f"synthetic events: {report['synthetic_event_count']}")
    print(f"output: {args.out}")
    return 0


def _calibrate_synthetic_catalog(args: Namespace) -> int:
    report = calibrate_synthetic_catalog(
        real_events=args.real_events,
        synthetic_events=args.synthetic_events,
        out_path=args.out,
        seed=args.seed,
        synthetic_duration_days=args.synthetic_duration_days,
    )
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"retained events: {report['retained_event_count']}")
    print(f"keep probability: {report['keep_probability']:.6f}")
    print(f"output: {args.out}")
    return 0


def _calibrate_synthetic_spatial(args: Namespace) -> int:
    report = calibrate_synthetic_spatial_coordinates(
        real_events=args.real_events, synthetic_events=args.synthetic_events, out_path=args.out
    )
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"synthetic events: {report['synthetic_event_count']}")
    print(f"output: {args.out}")
    return 0


def _combine_synthetic_catalogs(args: Namespace) -> int:
    report = combine_synthetic_catalogs(
        synthetic_events=args.synthetic_events, out_path=args.out, offset_days=args.offset_days
    )
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"episodes: {report['episode_count']}")
    print(f"events: {report['event_count']}")
    print(f"output: {args.out}")
    return 0


def _balance_synthetic_episode_rates(args: Namespace) -> int:
    report = balance_synthetic_episode_rates(
        real_events=args.real_events,
        synthetic_events=args.synthetic_events,
        episode_duration_days=args.episode_duration_days,
        out_path=args.out,
        seed=args.seed,
    )
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"input events: {report['input_event_count']}")
    print(f"output events: {report['output_event_count']}")
    print(f"target rate: {report['target_rate_per_day']:.6f}")
    print(f"output: {args.out}")
    return 0


def _train_ablation_smoke(args: Namespace) -> int:
    report = train_ablation_smoke(
        input_csv=args.input,
        out_path=args.out,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
    )
    print(f"status: {report['status']}")
    print(f"rows: {report['row_count']}")
    print(f"labeled rows: {report['labeled_row_count']}")
    print(f"output: {args.out}")
    return 0


def _run_real_transfer_trial(args: Namespace) -> int:
    report = run_real_transfer_trial(
        real_events_csv=args.real_events,
        synthetic_event_csvs=args.synthetic_events,
        out_dir=args.out_dir,
        magnitude_threshold=args.magnitude_threshold,
        horizon_days=args.horizon_days,
        cell_degrees=args.cell_degrees,
        train_fraction=args.train_fraction,
        pretrain_epochs=args.pretrain_epochs,
        finetune_epochs=args.finetune_epochs,
        seed=args.seed,
    )
    metrics = report["evaluation"]
    print(f"status: {report['status']}")
    print(f"held-out balanced accuracy: {metrics['balanced_accuracy']}")
    print(f"held-out precision: {metrics['precision']}")
    print(f"output: {args.out_dir / 'report.json'}")
    return 0


def _run_transfer_experiment_suite(args: Namespace) -> int:
    report = run_transfer_experiment_suite(
        real_events_csv=args.real_events,
        synthetic_event_csvs=args.synthetic_events,
        out_path=args.out,
        magnitude_threshold=args.magnitude_threshold,
        horizon_days=args.horizon_days,
        cell_degrees=args.cell_degrees,
        train_fraction=args.train_fraction,
        epochs=args.epochs,
        pretrain_epochs=args.pretrain_epochs,
        seed=args.seed,
    )
    ablations = report["experiment_1_matched_ablation"]
    print(f"status: {report['status']}")
    for name, result in ablations.items():
        print(f"{name}: balanced_accuracy={result.get('balanced_accuracy', 'n/a')} precision={result.get('precision', 'n/a')}")
    print(f"output: {args.out}")
    return 0


def _evaluate_temporal_holdout(args: Namespace) -> int:
    report = evaluate_temporal_holdout(
        input_csv=args.input,
        out_path=args.out,
        time_field=args.time_field,
        train_fraction=args.train_fraction,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        group_by_time=args.group_by_time,
    )
    print_holdout_report(report, args.out)
    return 0


def _diagnose_temporal_split(args: Namespace) -> int:
    report = diagnose_temporal_split(
        input_csv=args.input,
        out_path=args.out,
        feature_out=args.feature_out,
        time_field=args.time_field,
        train_fraction=args.train_fraction,
        target_field=args.target_field,
        top_n=args.top_n,
    )
    print(f"status: {report['status']}")
    print(f"rows: {report['row_count']}")
    print(f"labeled rows: {report['labeled_row_count']}")
    if report["status"] == "evaluated":
        print(f"train rows: {report['train_row_count']}")
        print(f"test rows: {report['test_row_count']}")
        print(f"train positive rate: {report['train_positive_rate']:.6f}")
        print(f"test positive rate: {report['test_positive_rate']:.6f}")
        print(f"features: {report['feature_count']}")
    print(f"output: {args.out}")
    if args.feature_out:
        print(f"feature output: {args.feature_out}")
    return 0


def _annotate_synthetic_regimes(args: Namespace) -> int:
    report = annotate_synthetic_regimes(
        input_csv=args.input,
        out_csv=args.out,
        report_path=args.report,
        group_field=args.group_field,
        time_field=args.time_field,
        regime_count=args.regime_count,
        burn_in_fraction=args.burn_in_fraction,
        drop_burn_in=args.drop_burn_in,
    )
    print(f"rows: {report['row_count']}")
    print(f"output rows: {report['output_row_count']}")
    print(f"regimes: {len(report['regime_ids'])}")
    print(f"output: {args.out}")
    print(f"report: {args.report}")
    return 0


def _assign_balanced_split(args: Namespace) -> int:
    report = assign_balanced_split(
        input_csv=args.input,
        out_csv=args.out,
        report_path=args.report,
        group_field=args.group_field,
        target_field=args.target_field,
        time_field=args.time_field,
        split_field=args.split_field,
        test_fraction=args.test_fraction,
    )
    print(f"rows: {report['row_count']}")
    print(f"train rows: {report['train_row_count']}")
    print(f"test rows: {report['test_row_count']}")
    print(f"train positives: {report['train_positive_count']}")
    print(f"test positives: {report['test_positive_count']}")
    print(f"output: {args.out}")
    print(f"report: {args.report}")
    return 0


def _prepare_transformer_target_input(args: Namespace) -> int:
    report = prepare_transformer_target_input(
        input_csv=args.input,
        out_csv=args.out,
        report_path=args.report,
        target_field=args.target_field,
        target_status_field=args.target_status_field,
        standard_target_field=args.standard_target_field,
        standard_status_field=args.standard_status_field,
        split_field=args.split_field,
        group_field=args.group_field,
        time_field=args.time_field,
        train_fraction=args.train_fraction,
    )
    print(f"rows: {report['row_count']}")
    print(f"labeled rows: {report['labeled_row_count']}")
    print(f"train rows: {report['train_row_count']}")
    print(f"test rows: {report['test_row_count']}")
    print(f"output: {args.out}")
    print(f"report: {args.report}")
    return 0


def _build_synthetic_step_targets(args: Namespace) -> int:
    report = build_synthetic_step_targets(
        piezo_paths=args.piezo, event_paths=args.events, out_path=args.out, report_path=args.report,
        horizon_steps=args.horizon_steps, stride_steps=args.stride_steps,
        start_time_utc=args.start_time_utc, step_seconds=args.step_seconds,
    )
    print(f"rows: {report['row_count']}")
    print(f"positives: {report['positive_count']}")
    print(f"output: {args.out}")
    return 0


def _evaluate_group_holdout(args: Namespace) -> int:
    report = evaluate_group_holdout(
        input_csv=args.input,
        out_path=args.out,
        group_field=args.group_field,
        test_group=args.test_group,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
    )
    print_holdout_report(report, args.out)
    return 0


def _summarize_model_run_reports(args: Namespace) -> int:
    summary = summarize_model_run_reports(report_paths=args.report, out_path=args.out)
    print(f"reports: {summary['report_count']}")
    print(f"output: {args.out}")
    return 0


def _list_model_candidates(args: Namespace) -> int:
    rows = write_model_candidates(out_path=args.out, stage=args.stage)
    print(f"candidates: {len(rows)}")
    print(f"output: {args.out}")
    return 0


def _estimate_model_scale(args: Namespace) -> int:
    report = estimate_model_scale(
        input_csv=args.input,
        out_path=args.out,
        sequence_manifest_paths=args.sequence_manifest,
        target_field=args.target_field,
        group_field=args.group_field,
        lookback_steps=args.lookback_steps,
        include_missing_masks=not args.no_missing_masks,
    )
    print(f"labeled rows: {report['labeled_row_count']}")
    print(f"positives: {report['positive_count']}")
    print(f"negatives: {report['negative_count']}")
    print(f"sequence features: {report['sequence_feature_count']}")
    print(f"recommended next model: {report['recommended_next_model']}")
    print(f"output: {args.out}")
    return 0


def _build_tensor_spec(args: Namespace) -> int:
    spec = build_tensor_spec(
        input_csv=args.input,
        out_path=args.out,
        time_field=args.time_field,
        region_field=args.region_field,
        target_field=args.target_field,
    )
    print(f"rows: {spec['row_count']}")
    print(f"numeric features: {spec['numeric_feature_count']}")
    print(f"output: {args.out}")
    return 0


def _materialize_tensor_dataset(args: Namespace) -> int:
    manifest = materialize_tensor_dataset(
        spec_path=args.spec,
        out_dir=args.out_dir,
        fill_value=args.fill_value,
    )
    print(f"rows: {manifest['row_count']}")
    print(f"features: {manifest['feature_count']}")
    print(f"manifest: {Path(args.out_dir) / 'manifest.json'}")
    return 0


def _audit_model_interfaces(args: Namespace) -> int:
    report = audit_model_interfaces(input_paths=args.input, out_path=args.out)
    print(f"tables: {report['table_count']}")
    print(f"output: {args.out}")
    return 0


def _build_event_window_features(args: Namespace) -> int:
    rows = build_event_window_features(
        events_csv=args.events,
        out_path=args.out,
        region_id=args.region_id,
        start_utc=args.start_utc,
        end_utc=args.end_utc,
        window_seconds=args.window_seconds,
        feature_prefix=args.feature_prefix,
        min_magnitude=args.min_magnitude,
    )
    print(f"window rows: {len(rows)}")
    print(f"output: {args.out}")
    return 0


def _materialize_sequence_dataset(args: Namespace) -> int:
    manifest = materialize_sequence_dataset(
        input_csv=args.input,
        out_dir=args.out_dir,
        time_field=args.time_field,
        entity_field=None if args.no_entity_field else args.entity_field,
        fill_value=args.fill_value,
        modality=args.modality,
        time_start_utc=args.time_start_utc,
        time_step_seconds=args.time_step_seconds,
    )
    print(f"rows: {manifest['row_count']}")
    print(f"times: {manifest['time_count']}")
    print(f"entities: {manifest['entity_count']}")
    print(f"channels: {manifest['channel_count']}")
    print(f"manifest: {Path(args.out_dir) / 'manifest.json'}")
    return 0


def _build_alignment_manifest(args: Namespace) -> int:
    report = build_alignment_manifest(manifest_paths=args.manifest, out_path=args.out, run_id=args.run_id)
    print(f"datasets: {report['dataset_count']}")
    print(f"output: {args.out}")
    return 0


def _build_aligned_window_dataset(args: Namespace) -> int:
    rows = build_aligned_window_dataset(
        base_manifest_path=args.base_manifest,
        sequence_manifest_paths=args.sequence_manifest,
        tensor_manifest_paths=args.tensor_manifest,
        out_path=args.out,
        target_source_feature=args.target_source_feature,
        target_horizon_rows=args.target_horizon_rows,
        target_threshold=args.target_threshold,
        drop_unlabeled_targets=args.drop_unlabeled_targets,
    )
    labeled = sum(1 for row in rows if row.get("target_occurred") in {"0", "1"})
    print(f"rows: {len(rows)}")
    print(f"labeled rows: {labeled}")
    print(f"output: {args.out}")
    return 0


def _combine_aligned_datasets(args: Namespace) -> int:
    rows = combine_aligned_datasets(input_csvs=args.input, dataset_ids=args.dataset_id, out_path=args.out)
    print(f"rows: {len(rows)}")
    print(f"datasets: {len(args.input)}")
    print(f"output: {args.out}")
    return 0


def _generate_trial_weekly_event_forecast(args: Namespace) -> int:
    report = generate_trial_weekly_event_forecast(
        real_events_csv=args.real_events,
        out_path=args.out,
        events_out_path=args.events_out,
        as_of_utc=args.as_of_utc,
        horizon_days=args.horizon_days,
        magnitude_threshold=args.magnitude_threshold,
        max_events=args.max_events,
        seed=args.seed,
        synthetic_event_globs=args.synthetic_event_glob or None,
        vlf_window_csvs=args.vlf_window or None,
        vlf_anomaly_report=args.vlf_anomaly_report,
        vlf_audio_globs=args.vlf_audio_glob or None,
        astronomy_globs=args.astronomy_glob or None,
    )
    print(f"status: {report['status']}")
    print(f"predicted events: {report['predicted_event_count']}")
    print(f"forecast start: {report['forecast_start_utc']}")
    print(f"forecast end: {report['forecast_end_utc']}")
    print(f"events output: {args.events_out}")
    print(f"report: {args.out}")
    return 0


def _generate_learned_weekly_event_forecast(args: Namespace) -> int:
    report = generate_learned_weekly_event_forecast(
        real_events_csv=args.real_events,
        synthetic_windows_csv=args.synthetic_windows,
        out_path=args.out,
        events_out_path=args.events_out,
        as_of_utc=args.as_of_utc,
        horizon_days=args.horizon_days,
        magnitude_threshold=args.magnitude_threshold,
        max_events=args.max_events,
        seed=args.seed,
        train_fraction=args.train_fraction,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        l2=args.l2,
        synthetic_event_globs=args.synthetic_event_glob or None,
        vlf_window_csvs=args.vlf_window or None,
        vlf_anomaly_report=args.vlf_anomaly_report,
        vlf_audio_globs=args.vlf_audio_glob or None,
        astronomy_globs=args.astronomy_glob or None,
    )
    scorer = report["model"]["learned_scorer"]
    print(f"status: {report['status']}")
    print(f"predicted events: {report['predicted_event_count']}")
    print(f"learned score: {scorer['latest_window_score']}")
    print(f"test balanced accuracy: {scorer.get('test_metrics', {}).get('balanced_accuracy', '')}")
    print(f"events output: {args.events_out}")
    print(f"report: {args.out}")
    return 0


def _compare_weekly_forecasts(args: Namespace) -> int:
    report = compare_weekly_forecasts(
        baseline_report=args.baseline_report,
        baseline_events=args.baseline_events,
        candidate_report=args.candidate_report,
        candidate_events=args.candidate_events,
        out_path=args.out,
        csv_out_path=args.csv_out,
    )
    print(f"status: {report['status']}")
    print(f"baseline events: {report['baseline']['event_count']}")
    print(f"candidate events: {report['candidate']['event_count']}")
    print(f"stage 1 pass: {report['criteria']['stage_1_event_contract_pass']}")
    print(f"stage 2 pass: {report['criteria']['stage_2_synthetic_model_pass']}")
    print(f"output: {args.out}")
    if args.csv_out:
        print(f"csv output: {args.csv_out}")
    return 0


def _build_synthetic_event_list_targets(args: Namespace) -> int:
    report = build_synthetic_event_list_targets(
        input_csv=args.input,
        out_csv=args.out,
        report_path=args.report,
        horizon_rows=args.horizon_rows,
        magnitude_threshold=args.magnitude_threshold,
        group_field=args.group_field,
        source_field=args.source_field,
    )
    print(f"rows: {report['row_count']}")
    print(f"labeled rows: {report['labeled_row_count']}")
    print(f"positives: {report['positive_count']}")
    print(f"negative: {report['negative_count']}")
    print(f"positive rate: {report['positive_rate']}")
    print(f"output: {args.out}")
    print(f"report: {args.report}")
    return 0


def _train_synthetic_event_list_model(args: Namespace) -> int:
    report = train_synthetic_event_list_model(
        input_csv=args.input,
        out_path=args.out,
        predictions_out=args.predictions_out,
        train_fraction=args.train_fraction,
        split_field=args.split_field,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        l2=args.l2,
        seed=args.seed,
        max_feature_count=args.max_feature_count,
        occurrence_model_type=args.occurrence_model_type,
        occurrence_ensemble_count=args.occurrence_ensemble_count,
        occurrence_feature_bag_fraction=args.occurrence_feature_bag_fraction,
        occurrence_stump_count=args.occurrence_stump_count,
    )
    print(f"status: {report['status']}")
    print(f"rows: {report['row_count']}")
    if report["status"] == "evaluated":
        occurrence = report["occurrence"]["test_metrics"]
        print(f"test balanced accuracy: {occurrence['balanced_accuracy']}")
        print(f"test positive recall: {occurrence['positive_recall']}")
        print(f"test negative recall: {occurrence['negative_recall']}")
        print(f"count MAE: {report['count']['test_mae']}")
        print(f"centroid median error km: {report['centroid']['positive_test_median_error_km']}")
    print(f"output: {args.out}")
    if args.predictions_out:
        print(f"predictions output: {args.predictions_out}")
    return 0


def _summarize_synthetic_event_list_probes(args: Namespace) -> int:
    report = summarize_synthetic_event_list_probes(
        root_dir=args.root,
        out_path=args.out,
        csv_out_path=args.csv_out,
    )
    print(f"reports: {report['report_count']}")
    print(f"output: {args.out}")
    if args.csv_out:
        print(f"csv output: {args.csv_out}")
    return 0


def _build_synthetic_lagged_context(args: Namespace) -> int:
    report = build_synthetic_lagged_context(
        input_csv=args.input,
        out_csv=args.out,
        report_path=args.report,
        lags=args.lag or [1, 2, 3, 6],
        group_field=args.group_field,
        time_field=args.time_field,
    )
    print(f"rows: {report['row_count']}")
    print(f"base features: {report['base_feature_count']}")
    print(f"added features: {report['added_feature_count']}")
    print(f"output: {args.out}")
    print(f"report: {args.report}")
    return 0


def _train_synthetic_event_list_sequence_head(args: Namespace) -> int:
    report = train_synthetic_event_list_sequence_head(
        input_csv=args.input,
        out_path=args.out,
        predictions_out=args.predictions_out,
        target_field=args.target_field,
        target_status_field=args.target_status_field,
        group_field=args.group_field,
        time_field=args.time_field,
        train_fraction=args.train_fraction,
        lookback_rows=args.lookback_rows,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        hidden_units=args.hidden_units,
        batch_size=args.batch_size,
        dropout=args.dropout,
        weight_decay=args.weight_decay,
        max_feature_count=args.max_feature_count,
        validation_fraction=args.validation_fraction,
        early_stopping_patience=args.early_stopping_patience,
        calibration_source=args.calibration_source,
        seed=args.seed,
    )
    print(f"status: {report['status']}")
    print(f"rows: {report['labeled_row_count']}")
    if report["status"] == "evaluated":
        metrics = report["calibrated_test_metrics"]
        print(f"calibrated test balanced accuracy: {metrics['balanced_accuracy']}")
        print(f"calibrated test positive recall: {metrics['positive_recall']}")
        print(f"calibrated test negative recall: {metrics['negative_recall']}")
        print(f"feature count: {report['feature_count']}")
    print(f"output: {args.out}")
    if args.predictions_out:
        print(f"predictions output: {args.predictions_out}")
    return 0


def _summarize_synthetic_event_list_sequence_heads(args: Namespace) -> int:
    report = summarize_synthetic_event_list_sequence_heads(
        root_dir=args.root,
        out_path=args.out,
        csv_out_path=args.csv_out,
    )
    print(f"reports: {report['report_count']}")
    print(f"configs: {len(report['configs'])}")
    print(f"output: {args.out}")
    if args.csv_out:
        print(f"csv output: {args.csv_out}")
    return 0


def _ensemble_synthetic_event_list_sequence_heads(args: Namespace) -> int:
    report = ensemble_synthetic_event_list_sequence_heads(
        report_paths=args.report,
        out_path=args.out,
        predictions_out=args.predictions_out,
    )
    print(f"status: {report['status']}")
    print(f"reports: {report['usable_report_count']}")
    if report["status"] == "evaluated":
        metrics = report["calibrated_test_metrics"]
        print(f"calibrated test balanced accuracy: {metrics['balanced_accuracy']}")
        print(f"calibrated test positive recall: {metrics['positive_recall']}")
        print(f"calibrated test negative recall: {metrics['negative_recall']}")
    print(f"output: {args.out}")
    if args.predictions_out:
        print(f"predictions output: {args.predictions_out}")
    return 0


def _diagnose_synthetic_drift(args: Namespace) -> int:
    report = diagnose_synthetic_drift(
        input_csv=args.input,
        out_path=args.out,
        csv_out_path=args.csv_out,
        target_field=args.target_field,
        target_status_field=args.target_status_field,
        group_field=args.group_field,
        time_field=args.time_field,
        train_fraction=args.train_fraction,
        bucket_count=args.bucket_count,
        top_n=args.top_n,
    )
    split = report["temporal_split"]
    print(f"rows: {report['row_count']}")
    print(f"labeled rows: {report['labeled_row_count']}")
    print(f"overall positive rate: {report['overall']['positive_rate']}")
    print(f"train positive rate: {split['train']['positive_rate']}")
    print(f"test positive rate: {split['test']['positive_rate']}")
    print(f"warning: {split['warning']}")
    print(f"output: {args.out}")
    if args.csv_out:
        print(f"csv output: {args.csv_out}")
    return 0


def _annotate_synthetic_episodes(args: Namespace) -> int:
    report = annotate_synthetic_episodes(
        input_csv=args.input,
        out_csv=args.out,
        report_path=args.report,
        group_field=args.group_field,
        time_field=args.time_field,
        rows_per_episode=args.rows_per_episode,
        target_field=args.target_field,
        drop_partial=args.drop_partial,
    )
    print(f"rows: {report['row_count']}")
    print(f"output rows: {report['output_row_count']}")
    print(f"episodes: {report['episode_count']}")
    print(f"output: {args.out}")
    print(f"report: {args.report}")
    return 0

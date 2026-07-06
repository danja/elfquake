"""Model and tensor-interface CLI commands."""

from __future__ import annotations

from argparse import Namespace, _SubParsersAction
from pathlib import Path

from elfquake.models.ablation_smoke import train_ablation_smoke
from elfquake.models.aligned_windows import build_aligned_window_dataset
from elfquake.models.alignment_manifest import build_alignment_manifest
from elfquake.models.candidates import write_model_candidates
from elfquake.models.dataset_combine import combine_aligned_datasets
from elfquake.models.interface_shape import audit_model_interfaces
from elfquake.models.logistic_smoke import train_logistic_smoke
from elfquake.models.readiness import summarize_model_readiness
from elfquake.models.report_summary import summarize_model_run_reports
from elfquake.models.sequence_materializer import materialize_sequence_dataset
from elfquake.models.split_diagnostics import diagnose_temporal_split
from elfquake.models.tensor_materializer import materialize_tensor_dataset
from elfquake.models.tensor_spec import build_tensor_spec
from elfquake.models.temporal_holdout import evaluate_group_holdout, evaluate_temporal_holdout
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

    ablation = subparsers.add_parser("train-ablation-smoke")
    ablation.add_argument("--input", type=Path, required=True)
    ablation.add_argument("--out", type=Path, required=True)
    ablation.add_argument("--epochs", type=int, default=600)
    ablation.add_argument("--learning-rate", type=float, default=0.2)
    ablation.set_defaults(func=_train_ablation_smoke)

    temporal_holdout = subparsers.add_parser("evaluate-temporal-holdout")
    temporal_holdout.add_argument("--input", type=Path, required=True)
    temporal_holdout.add_argument("--out", type=Path, required=True)
    temporal_holdout.add_argument("--time-field", default="window_start_utc")
    temporal_holdout.add_argument("--train-fraction", type=float, default=0.8)
    temporal_holdout.add_argument("--epochs", type=int, default=600)
    temporal_holdout.add_argument("--learning-rate", type=float, default=0.2)
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


def _evaluate_temporal_holdout(args: Namespace) -> int:
    report = evaluate_temporal_holdout(
        input_csv=args.input,
        out_path=args.out,
        time_field=args.time_field,
        train_fraction=args.train_fraction,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
    )
    _print_holdout_report(report, args.out)
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


def _evaluate_group_holdout(args: Namespace) -> int:
    report = evaluate_group_holdout(
        input_csv=args.input,
        out_path=args.out,
        group_field=args.group_field,
        test_group=args.test_group,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
    )
    _print_holdout_report(report, args.out)
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


def _print_holdout_report(report: dict[str, str], out_path: Path) -> None:
    print(f"status: {report['status']}")
    print(f"rows: {report['row_count']}")
    print(f"labeled rows: {report['labeled_row_count']}")
    if "train_row_count" in report:
        print(f"train rows: {report['train_row_count']}")
        print(f"test rows: {report['test_row_count']}")
    print(f"output: {out_path}")

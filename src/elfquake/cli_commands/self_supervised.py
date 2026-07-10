"""Self-supervised model CLI commands."""

from __future__ import annotations

from argparse import Namespace, _SubParsersAction
from pathlib import Path

from elfquake.models.torch_self_supervised import (
    compare_sequence_embedding_domains,
    evaluate_mixed_domain_alignment,
    evaluate_synthetic_inlier_transfer,
    pretrain_sequence_autoencoder,
    score_sequence_anomalies,
)
from elfquake.models.torch_ssl_transformer_evaluation import (
    REGIMES as TRANSFORMER_SSL_REGIMES,
    evaluate_self_supervised_transformer,
)
from elfquake.models.torch_late_gated_evaluation import evaluate_late_gated_fusion


def register_self_supervised_commands(subparsers: _SubParsersAction) -> None:
    late_fusion = subparsers.add_parser("evaluate-late-gated-fusion")
    late_fusion.add_argument("--target", type=Path, required=True)
    late_fusion.add_argument("--synthetic-sequence-manifest", type=Path, action="append", required=True)
    late_fusion.add_argument("--out", type=Path, required=True)
    late_fusion.add_argument("--artifact-root", type=Path)
    late_fusion.add_argument("--seed", type=int, action="append")
    late_fusion.add_argument("--split-field", default="model_split")
    late_fusion.add_argument("--train-value", default="train")
    late_fusion.add_argument("--test-value", default="test")
    late_fusion.add_argument("--lookback-steps", type=int, default=12)
    late_fusion.add_argument("--patch-steps", type=int, default=3)
    late_fusion.add_argument("--train-fraction", type=float, default=0.8)
    late_fusion.add_argument("--pretrain-stride", type=int, default=3)
    late_fusion.add_argument("--ssl-epochs", type=int, default=6)
    late_fusion.add_argument("--supervised-epochs", type=int, default=12)
    late_fusion.add_argument("--learning-rate", type=float, default=0.001)
    late_fusion.add_argument("--d-model", type=int, default=32)
    late_fusion.add_argument("--layers", type=int, default=2)
    late_fusion.add_argument("--heads", type=int, default=4)
    late_fusion.add_argument("--dropout", type=float, default=0.1)
    late_fusion.add_argument("--batch-size", type=int, default=32)
    late_fusion.add_argument("--mask-probability", type=float, default=0.30)
    late_fusion.add_argument("--modality-dropout-probability", type=float, default=0.25)
    late_fusion.add_argument("--max-pretrain-windows", type=int, default=2048)
    late_fusion.set_defaults(func=_evaluate_late_gated_fusion)

    transformer = subparsers.add_parser("evaluate-self-supervised-transformer")
    transformer.add_argument("--target", type=Path, required=True)
    transformer.add_argument("--synthetic-sequence-manifest", type=Path, action="append", required=True)
    transformer.add_argument("--real-sequence-manifest", type=Path, required=True)
    transformer.add_argument("--out", type=Path, required=True)
    transformer.add_argument("--artifact-root", type=Path)
    transformer.add_argument("--regime", action="append", choices=TRANSFORMER_SSL_REGIMES)
    transformer.add_argument("--seed", type=int, action="append")
    transformer.add_argument("--split-field", default="model_split")
    transformer.add_argument("--train-value", default="train")
    transformer.add_argument("--test-value", default="test")
    transformer.add_argument("--lookback-steps", type=int, default=12)
    transformer.add_argument("--patch-steps", type=int, default=3)
    transformer.add_argument("--train-fraction", type=float, default=0.8)
    transformer.add_argument("--pretrain-stride", type=int, default=3)
    transformer.add_argument("--ssl-epochs", type=int, default=8)
    transformer.add_argument("--supervised-epochs", type=int, default=12)
    transformer.add_argument("--learning-rate", type=float, default=0.001)
    transformer.add_argument("--d-model", type=int, default=32)
    transformer.add_argument("--layers", type=int, default=2)
    transformer.add_argument("--heads", type=int, default=4)
    transformer.add_argument("--dropout", type=float, default=0.1)
    transformer.add_argument("--batch-size", type=int, default=32)
    transformer.add_argument("--mask-probability", type=float, default=0.30)
    transformer.add_argument("--modality-dropout-probability", type=float, default=0.25)
    transformer.add_argument("--max-pretrain-windows", type=int, default=4096)
    transformer.set_defaults(func=_evaluate_self_supervised_transformer)

    sequence_autoencoder = subparsers.add_parser("pretrain-sequence-autoencoder")
    sequence_autoencoder.add_argument("--sequence-manifest", type=Path, required=True)
    sequence_autoencoder.add_argument("--out", type=Path, required=True)
    sequence_autoencoder.add_argument("--modality", default="")
    sequence_autoencoder.add_argument("--lookback-steps", type=int, default=24)
    sequence_autoencoder.add_argument("--stride", type=int, default=1)
    sequence_autoencoder.add_argument("--train-fraction", type=float, default=0.8)
    sequence_autoencoder.add_argument("--mask-probability", type=float, default=0.15)
    sequence_autoencoder.add_argument("--clean-loss-weight", type=float, default=0.0)
    sequence_autoencoder.add_argument("--epochs", type=int, default=30)
    sequence_autoencoder.add_argument("--learning-rate", type=float, default=0.0003)
    sequence_autoencoder.add_argument("--hidden-units", type=int, default=32)
    sequence_autoencoder.add_argument("--embedding-units", type=int, default=8)
    sequence_autoencoder.add_argument("--batch-size", type=int, default=32)
    sequence_autoencoder.add_argument("--seed", type=int, default=42)
    sequence_autoencoder.add_argument("--no-missing-masks", action="store_true")
    sequence_autoencoder.add_argument("--checkpoint-out", type=Path)
    sequence_autoencoder.add_argument("--embeddings-out", type=Path)
    sequence_autoencoder.set_defaults(func=_pretrain_sequence_autoencoder)

    domain_compare = subparsers.add_parser("compare-sequence-embedding-domains")
    domain_compare.add_argument("--real-sequence-manifest", type=Path, required=True)
    domain_compare.add_argument("--synthetic-sequence-manifest", type=Path, action="append", required=True)
    domain_compare.add_argument("--out", type=Path, required=True)
    domain_compare.add_argument("--real-modality", default="real_vlf_image")
    domain_compare.add_argument("--synthetic-modality", default="synthetic_piezo_vlf")
    domain_compare.add_argument("--descriptor-profile", default="shape", choices=["shape", "full"])
    domain_compare.add_argument("--lookback-steps", type=int, default=24)
    domain_compare.add_argument("--stride", type=int, default=1)
    domain_compare.add_argument("--train-fraction", type=float, default=0.8)
    domain_compare.add_argument("--mask-probability", type=float, default=0.15)
    domain_compare.add_argument("--clean-loss-weight", type=float, default=0.0)
    domain_compare.add_argument("--inlier-fraction", type=float, default=0.25)
    domain_compare.add_argument("--epochs", type=int, default=30)
    domain_compare.add_argument("--learning-rate", type=float, default=0.0003)
    domain_compare.add_argument("--hidden-units", type=int, default=32)
    domain_compare.add_argument("--embedding-units", type=int, default=8)
    domain_compare.add_argument("--batch-size", type=int, default=32)
    domain_compare.add_argument("--seed", type=int, default=42)
    domain_compare.add_argument("--no-missing-masks", action="store_true")
    domain_compare.add_argument("--embeddings-out", type=Path)
    domain_compare.set_defaults(func=_compare_sequence_embedding_domains)

    transfer = subparsers.add_parser("evaluate-synthetic-inlier-transfer")
    transfer.add_argument("--real-sequence-manifest", type=Path, required=True)
    transfer.add_argument("--synthetic-sequence-manifest", type=Path, action="append", required=True)
    transfer.add_argument("--out", type=Path, required=True)
    transfer.add_argument("--real-modality", default="real_vlf_image")
    transfer.add_argument("--synthetic-modality", default="synthetic_piezo_vlf")
    transfer.add_argument("--descriptor-profile", default="shape", choices=["shape", "full"])
    transfer.add_argument("--lookback-steps", type=int, default=24)
    transfer.add_argument("--stride", type=int, default=1)
    transfer.add_argument("--train-fraction", type=float, default=0.8)
    transfer.add_argument("--mask-probability", type=float, default=0.15)
    transfer.add_argument("--clean-loss-weight", type=float, default=0.0)
    transfer.add_argument("--inlier-fraction", type=float, default=0.25)
    transfer.add_argument("--epochs", type=int, default=30)
    transfer.add_argument("--learning-rate", type=float, default=0.0003)
    transfer.add_argument("--hidden-units", type=int, default=32)
    transfer.add_argument("--embedding-units", type=int, default=8)
    transfer.add_argument("--batch-size", type=int, default=32)
    transfer.add_argument("--seed", type=int, default=42)
    transfer.add_argument("--no-missing-masks", action="store_true")
    transfer.add_argument("--embeddings-out", type=Path)
    transfer.set_defaults(func=_evaluate_synthetic_inlier_transfer)

    mixed_alignment = subparsers.add_parser("evaluate-mixed-domain-alignment")
    mixed_alignment.add_argument("--real-sequence-manifest", type=Path, required=True)
    mixed_alignment.add_argument("--synthetic-sequence-manifest", type=Path, action="append", required=True)
    mixed_alignment.add_argument("--out", type=Path, required=True)
    mixed_alignment.add_argument("--real-modality", default="real_vlf_image")
    mixed_alignment.add_argument("--synthetic-modality", default="synthetic_piezo_vlf")
    mixed_alignment.add_argument("--descriptor-profile", default="shape", choices=["shape", "full"])
    mixed_alignment.add_argument("--lookback-steps", type=int, default=24)
    mixed_alignment.add_argument("--stride", type=int, default=1)
    mixed_alignment.add_argument("--train-fraction", type=float, default=0.8)
    mixed_alignment.add_argument("--mask-probability", type=float, default=0.15)
    mixed_alignment.add_argument("--clean-loss-weight", type=float, default=0.0)
    mixed_alignment.add_argument("--inlier-fraction", type=float, default=0.25)
    mixed_alignment.add_argument("--inlier-method", default="local", choices=["local", "centroid", "random", "full"])
    mixed_alignment.add_argument("--control-method", action="append", choices=["local", "centroid", "random", "full"])
    mixed_alignment.add_argument("--max-synthetic-train-windows", type=int, default=15000)
    mixed_alignment.add_argument("--no-balance-synthetic-sources", action="store_true")
    mixed_alignment.add_argument("--coral-weight", type=float, default=0.1)
    mixed_alignment.add_argument("--epochs", type=int, default=30)
    mixed_alignment.add_argument("--learning-rate", type=float, default=0.0003)
    mixed_alignment.add_argument("--hidden-units", type=int, default=32)
    mixed_alignment.add_argument("--embedding-units", type=int, default=8)
    mixed_alignment.add_argument("--batch-size", type=int, default=32)
    mixed_alignment.add_argument("--seed", type=int, default=42)
    mixed_alignment.add_argument("--no-missing-masks", action="store_true")
    mixed_alignment.add_argument("--embeddings-out", type=Path)
    mixed_alignment.set_defaults(func=_evaluate_mixed_domain_alignment)

    anomaly = subparsers.add_parser("score-sequence-anomalies")
    anomaly.add_argument("--sequence-manifest", type=Path, required=True)
    anomaly.add_argument("--out", type=Path, required=True)
    anomaly.add_argument("--scores-out", type=Path, required=True)
    anomaly.add_argument("--modality", default="real_vlf_image")
    anomaly.add_argument("--descriptor-profile", default="shape", choices=["shape", "full"])
    anomaly.add_argument("--lookback-steps", type=int, default=24)
    anomaly.add_argument("--stride", type=int, default=1)
    anomaly.add_argument("--train-fraction", type=float, default=0.8)
    anomaly.add_argument("--forecast-horizon-days", type=int, default=7)
    anomaly.add_argument("--alert-threshold", type=float, default=0.8)
    anomaly.add_argument("--mask-probability", type=float, default=0.15)
    anomaly.add_argument("--clean-loss-weight", type=float, default=0.0)
    anomaly.add_argument("--epochs", type=int, default=30)
    anomaly.add_argument("--learning-rate", type=float, default=0.0003)
    anomaly.add_argument("--hidden-units", type=int, default=32)
    anomaly.add_argument("--embedding-units", type=int, default=8)
    anomaly.add_argument("--batch-size", type=int, default=32)
    anomaly.add_argument("--seed", type=int, default=42)
    anomaly.add_argument("--no-missing-masks", action="store_true")
    anomaly.set_defaults(func=_score_sequence_anomalies)


def _pretrain_sequence_autoencoder(args: Namespace) -> int:
    report = pretrain_sequence_autoencoder(
        sequence_manifest_path=args.sequence_manifest,
        out_path=args.out,
        modality=args.modality,
        lookback_steps=args.lookback_steps,
        stride=args.stride,
        train_fraction=args.train_fraction,
        mask_probability=args.mask_probability,
        clean_loss_weight=args.clean_loss_weight,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        hidden_units=args.hidden_units,
        embedding_units=args.embedding_units,
        batch_size=args.batch_size,
        seed=args.seed,
        include_missing_masks=not args.no_missing_masks,
        checkpoint_out=args.checkpoint_out,
        embeddings_out=args.embeddings_out,
    )
    print(f"status: {report['status']}")
    print(f"rows: {report['row_count']}")
    print(f"windows: {report['window_count']}")
    if "train_window_count" in report:
        print(f"train windows: {report['train_window_count']}")
        print(f"test windows: {report['test_window_count']}")
    print(f"output: {args.out}")
    return 0


def _evaluate_self_supervised_transformer(args: Namespace) -> int:
    report = evaluate_self_supervised_transformer(
        target_csv=args.target,
        synthetic_manifest_paths=args.synthetic_sequence_manifest,
        real_manifest_path=args.real_sequence_manifest,
        out_path=args.out,
        artifact_root=args.artifact_root,
        regimes=args.regime,
        seeds=args.seed,
        split_field=args.split_field,
        train_value=args.train_value,
        test_value=args.test_value,
        lookback_steps=args.lookback_steps,
        patch_steps=args.patch_steps,
        train_fraction=args.train_fraction,
        pretrain_stride=args.pretrain_stride,
        ssl_epochs=args.ssl_epochs,
        supervised_epochs=args.supervised_epochs,
        learning_rate=args.learning_rate,
        d_model=args.d_model,
        layers=args.layers,
        heads=args.heads,
        dropout=args.dropout,
        batch_size=args.batch_size,
        mask_probability=args.mask_probability,
        modality_dropout_probability=args.modality_dropout_probability,
        max_pretrain_windows=args.max_pretrain_windows,
    )
    print(f"status: {report['status']}")
    print(f"downstream train/test: {report['downstream_train_rows']}/{report['downstream_test_rows']}")
    for regime, row in report["summary"].items():
        for config_name, config in row["downstream_models"].items():
            metrics = config["fine_tune_balanced_accuracy"]
            print(f"{regime}/{config_name}: mean={metrics['mean']:.6f} min={metrics['min']:.6f} max={metrics['max']:.6f}")
    print(f"output: {args.out}")
    return 0


def _evaluate_late_gated_fusion(args: Namespace) -> int:
    report = evaluate_late_gated_fusion(
        target_csv=args.target,
        synthetic_manifest_paths=args.synthetic_sequence_manifest,
        out_path=args.out,
        artifact_root=args.artifact_root,
        seeds=args.seed,
        split_field=args.split_field,
        train_value=args.train_value,
        test_value=args.test_value,
        lookback_steps=args.lookback_steps,
        patch_steps=args.patch_steps,
        train_fraction=args.train_fraction,
        pretrain_stride=args.pretrain_stride,
        ssl_epochs=args.ssl_epochs,
        supervised_epochs=args.supervised_epochs,
        learning_rate=args.learning_rate,
        d_model=args.d_model,
        layers=args.layers,
        heads=args.heads,
        dropout=args.dropout,
        batch_size=args.batch_size,
        mask_probability=args.mask_probability,
        modality_dropout_probability=args.modality_dropout_probability,
        max_pretrain_windows=args.max_pretrain_windows,
    )
    print(f"status: {report['status']}")
    for initialization, configs in report["summary"].items():
        for config_name, row in configs.items():
            metrics = row["balanced_accuracy"]
            print(f"{initialization}/{config_name}: mean={metrics['mean']:.6f} min={metrics['min']:.6f} max={metrics['max']:.6f}")
    print(f"output: {args.out}")
    return 0


def _compare_sequence_embedding_domains(args: Namespace) -> int:
    report = compare_sequence_embedding_domains(
        real_sequence_manifest_path=args.real_sequence_manifest,
        synthetic_sequence_manifest_paths=args.synthetic_sequence_manifest,
        out_path=args.out,
        real_modality=args.real_modality,
        synthetic_modality=args.synthetic_modality,
        descriptor_profile=args.descriptor_profile,
        lookback_steps=args.lookback_steps,
        stride=args.stride,
        train_fraction=args.train_fraction,
        mask_probability=args.mask_probability,
        clean_loss_weight=args.clean_loss_weight,
        inlier_fraction=args.inlier_fraction,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        hidden_units=args.hidden_units,
        embedding_units=args.embedding_units,
        batch_size=args.batch_size,
        seed=args.seed,
        include_missing_masks=not args.no_missing_masks,
        embeddings_out=args.embeddings_out,
    )
    print(f"status: {report['status']}")
    print(f"real windows: {report['real_window_count']}")
    print(f"synthetic windows: {report['synthetic_window_count']}")
    if report.get("embedding_comparison"):
        comparison = report["embedding_comparison"]
        print(f"centroid distance: {comparison.get('centroid_euclidean_distance', '')}")
    print(f"output: {args.out}")
    return 0


def _evaluate_synthetic_inlier_transfer(args: Namespace) -> int:
    report = evaluate_synthetic_inlier_transfer(
        real_sequence_manifest_path=args.real_sequence_manifest,
        synthetic_sequence_manifest_paths=args.synthetic_sequence_manifest,
        out_path=args.out,
        real_modality=args.real_modality,
        synthetic_modality=args.synthetic_modality,
        descriptor_profile=args.descriptor_profile,
        lookback_steps=args.lookback_steps,
        stride=args.stride,
        train_fraction=args.train_fraction,
        mask_probability=args.mask_probability,
        clean_loss_weight=args.clean_loss_weight,
        inlier_fraction=args.inlier_fraction,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        hidden_units=args.hidden_units,
        embedding_units=args.embedding_units,
        batch_size=args.batch_size,
        seed=args.seed,
        include_missing_masks=not args.no_missing_masks,
        embeddings_out=args.embeddings_out,
    )
    print(f"status: {report['status']}")
    print(f"real windows: {report['real_window_count']}")
    print(f"synthetic windows: {report['synthetic_window_count']}")
    if "synthetic_inlier_count" in report:
        print(f"synthetic inliers: {report['synthetic_inlier_count']}")
    if report.get("real_test_reconstruction"):
        metrics = report["real_test_reconstruction"]
        print(f"real test masked mse: {metrics.get('masked_mse', '')}")
        print(f"real test zero baseline: {metrics.get('zero_baseline_masked_mse', '')}")
    print(f"output: {args.out}")
    return 0


def _evaluate_mixed_domain_alignment(args: Namespace) -> int:
    report = evaluate_mixed_domain_alignment(
        real_sequence_manifest_path=args.real_sequence_manifest,
        synthetic_sequence_manifest_paths=args.synthetic_sequence_manifest,
        out_path=args.out,
        real_modality=args.real_modality,
        synthetic_modality=args.synthetic_modality,
        descriptor_profile=args.descriptor_profile,
        lookback_steps=args.lookback_steps,
        stride=args.stride,
        train_fraction=args.train_fraction,
        mask_probability=args.mask_probability,
        clean_loss_weight=args.clean_loss_weight,
        inlier_fraction=args.inlier_fraction,
        inlier_method=args.inlier_method,
        control_methods=args.control_method,
        max_synthetic_train_windows=args.max_synthetic_train_windows,
        balance_synthetic_sources=not args.no_balance_synthetic_sources,
        coral_weight=args.coral_weight,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        hidden_units=args.hidden_units,
        embedding_units=args.embedding_units,
        batch_size=args.batch_size,
        seed=args.seed,
        include_missing_masks=not args.no_missing_masks,
        embeddings_out=args.embeddings_out,
    )
    print(f"status: {report['status']}")
    print(f"real windows: {report['real_window_count']}")
    print(f"synthetic windows: {report['synthetic_window_count']}")
    if report.get("primary"):
        primary = report["primary"]
        comparison = primary.get("embedding_comparison", {})
        metrics = primary.get("real_test_reconstruction", {})
        print(f"selection: {report.get('selection_method', '')}")
        print(f"synthetic train windows: {report.get('synthetic_train_count', '')}")
        print(f"centroid distance: {comparison.get('centroid_euclidean_distance', '')}")
        print(f"real test masked mse: {metrics.get('masked_mse', '')}")
        print(f"real test zero baseline: {metrics.get('zero_baseline_masked_mse', '')}")
    print(f"output: {args.out}")
    return 0


def _score_sequence_anomalies(args: Namespace) -> int:
    report = score_sequence_anomalies(
        sequence_manifest_path=args.sequence_manifest,
        out_path=args.out,
        scores_out=args.scores_out,
        modality=args.modality,
        descriptor_profile=args.descriptor_profile,
        lookback_steps=args.lookback_steps,
        stride=args.stride,
        train_fraction=args.train_fraction,
        forecast_horizon_days=args.forecast_horizon_days,
        alert_threshold=args.alert_threshold,
        mask_probability=args.mask_probability,
        clean_loss_weight=args.clean_loss_weight,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        hidden_units=args.hidden_units,
        embedding_units=args.embedding_units,
        batch_size=args.batch_size,
        seed=args.seed,
        include_missing_masks=not args.no_missing_masks,
    )
    print(f"status: {report['status']}")
    print(f"windows: {report['window_count']}")
    if report.get("forecast"):
        forecast = report["forecast"]
        print(f"forecast target: {forecast.get('target_start_utc', '')}..{forecast.get('target_end_utc', '')}")
        print(f"demo probability: {forecast.get('demo_probability', '')}")
        print(f"demo predicted event: {forecast.get('demo_predicted_event', '')}")
    print(f"scores: {args.scores_out}")
    print(f"output: {args.out}")
    return 0

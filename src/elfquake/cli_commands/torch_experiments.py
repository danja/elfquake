"""CPU PyTorch Transformer experiment CLI commands."""

from __future__ import annotations

from argparse import Namespace, _SubParsersAction
from pathlib import Path

from elfquake.models.piezo_group_holdout_comparison import compare_piezo_group_holdouts
from elfquake.models.torch_late_gated_evaluation import evaluate_late_gated_fusion
from elfquake.models.torch_piezo_group_holdout import evaluate_piezo_group_holdout
from elfquake.models.torch_ssl_transformer_evaluation import (
    REGIMES as TRANSFORMER_SSL_REGIMES,
    evaluate_self_supervised_transformer,
)


def register_torch_experiment_commands(subparsers: _SubParsersAction) -> None:
    transformer = subparsers.add_parser("evaluate-self-supervised-transformer")
    _add_common_arguments(transformer, include_real=True)
    transformer.add_argument("--regime", action="append", choices=TRANSFORMER_SSL_REGIMES)
    transformer.add_argument("--train-fraction", type=float, default=0.8)
    transformer.add_argument("--pretrain-stride", type=int, default=3)
    transformer.add_argument("--ssl-epochs", type=int, default=8)
    transformer.add_argument("--supervised-epochs", type=int, default=12)
    transformer.add_argument("--mask-probability", type=float, default=0.30)
    transformer.add_argument("--modality-dropout-probability", type=float, default=0.25)
    transformer.add_argument("--max-pretrain-windows", type=int, default=4096)
    transformer.set_defaults(func=_evaluate_self_supervised_transformer)

    late_fusion = subparsers.add_parser("evaluate-late-gated-fusion")
    _add_common_arguments(late_fusion)
    late_fusion.add_argument("--train-fraction", type=float, default=0.8)
    late_fusion.add_argument("--pretrain-stride", type=int, default=3)
    late_fusion.add_argument("--ssl-epochs", type=int, default=6)
    late_fusion.add_argument("--supervised-epochs", type=int, default=12)
    late_fusion.add_argument("--mask-probability", type=float, default=0.30)
    late_fusion.add_argument("--modality-dropout-probability", type=float, default=0.25)
    late_fusion.add_argument("--max-pretrain-windows", type=int, default=2048)
    late_fusion.set_defaults(func=_evaluate_late_gated_fusion)

    group_holdout = subparsers.add_parser("evaluate-piezo-group-holdout")
    group_holdout.add_argument("--target", type=Path, required=True)
    group_holdout.add_argument("--piezo-sequence-manifest", type=Path, action="append", required=True)
    group_holdout.add_argument("--out", type=Path, required=True)
    group_holdout.add_argument("--artifact-root", type=Path)
    group_holdout.add_argument("--seed", type=int, action="append")
    group_holdout.add_argument("--group-field", default="dataset_id")
    group_holdout.add_argument(
        "--entity-aggregation-profile",
        choices=["mean", "piezo_spatial"],
        default="mean",
    )
    _add_model_arguments(group_holdout)
    group_holdout.add_argument("--epochs", type=int, default=12)
    group_holdout.set_defaults(func=_evaluate_piezo_group_holdout)

    comparison = subparsers.add_parser("compare-piezo-group-holdouts")
    comparison.add_argument("--report", type=Path, action="append", required=True)
    comparison.add_argument("--out", type=Path, required=True)
    comparison.add_argument("--balanced-accuracy-floor", type=float, default=0.60)
    comparison.add_argument("--recall-floor", type=float, default=0.40)
    comparison.add_argument("--fold-pass-fraction", type=float, default=0.80)
    comparison.set_defaults(func=_compare_piezo_group_holdouts)


def _add_common_arguments(parser, *, include_real: bool = False) -> None:
    parser.add_argument("--target", type=Path, required=True)
    parser.add_argument("--synthetic-sequence-manifest", type=Path, action="append", required=True)
    if include_real:
        parser.add_argument("--real-sequence-manifest", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--artifact-root", type=Path)
    parser.add_argument("--seed", type=int, action="append")
    parser.add_argument("--split-field", default="model_split")
    parser.add_argument("--train-value", default="train")
    parser.add_argument("--test-value", default="test")
    _add_model_arguments(parser)


def _add_model_arguments(parser) -> None:
    parser.add_argument("--lookback-steps", type=int, default=12)
    parser.add_argument("--patch-steps", type=int, default=3)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--d-model", type=int, default=32)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--batch-size", type=int, default=32)


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


def _evaluate_piezo_group_holdout(args: Namespace) -> int:
    report = evaluate_piezo_group_holdout(
        target_csv=args.target,
        piezo_manifest_paths=args.piezo_sequence_manifest,
        out_path=args.out,
        artifact_root=args.artifact_root,
        seeds=args.seed,
        group_field=args.group_field,
        lookback_steps=args.lookback_steps,
        patch_steps=args.patch_steps,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        d_model=args.d_model,
        layers=args.layers,
        heads=args.heads,
        dropout=args.dropout,
        batch_size=args.batch_size,
        entity_aggregation_profile=args.entity_aggregation_profile,
    )
    metrics = report["summary"]["balanced_accuracy"]
    print(f"status: {report['status']}")
    print(f"runs: {report['summary']['run_count']}")
    print(f"balanced accuracy: mean={metrics['mean']:.6f} min={metrics['min']:.6f} max={metrics['max']:.6f}")
    print(f"output: {args.out}")
    return 0


def _compare_piezo_group_holdouts(args: Namespace) -> int:
    report = compare_piezo_group_holdouts(
        report_paths=args.report,
        out_path=args.out,
        balanced_accuracy_floor=args.balanced_accuracy_floor,
        recall_floor=args.recall_floor,
        fold_pass_fraction=args.fold_pass_fraction,
    )
    print(f"experiments: {report['experiment_count']}")
    print(f"passing: {report['passing_experiment_count']}")
    print(f"best: {report['best_experiment']}")
    print(f"output: {args.out}")
    return 0

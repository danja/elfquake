"""CPU PyTorch model CLI commands."""

from __future__ import annotations

from argparse import Namespace, _SubParsersAction
from pathlib import Path

from elfquake.cli_commands.common import print_holdout_report
from elfquake.models.torch_patch_transformer import evaluate_torch_patch_transformer_split_holdout
from elfquake.models.torch_sequence import (
    evaluate_torch_sequence_group_holdout,
    evaluate_torch_sequence_holdout,
    evaluate_torch_sequence_split_holdout,
)
from elfquake.models.torch_tabular import evaluate_torch_tabular_group_holdout, evaluate_torch_tabular_holdout


def register_torch_model_commands(subparsers: _SubParsersAction) -> None:
    torch_tabular = subparsers.add_parser("train-torch-tabular-holdout")
    torch_tabular.add_argument("--input", type=Path, required=True)
    torch_tabular.add_argument("--out", type=Path, required=True)
    torch_tabular.add_argument("--time-field", default="window_start_utc")
    torch_tabular.add_argument("--train-fraction", type=float, default=0.8)
    torch_tabular.add_argument("--epochs", type=int, default=80)
    torch_tabular.add_argument("--learning-rate", type=float, default=0.001)
    torch_tabular.add_argument("--hidden-units", type=int, default=32)
    torch_tabular.add_argument("--batch-size", type=int, default=64)
    torch_tabular.add_argument("--seed", type=int, default=42)
    torch_tabular.add_argument("--weight-decay", type=float, default=0.0)
    torch_tabular.add_argument("--no-missing-masks", action="store_true")
    torch_tabular.set_defaults(func=_train_torch_tabular_holdout)

    torch_tabular_group = subparsers.add_parser("train-torch-tabular-group-holdout")
    torch_tabular_group.add_argument("--input", type=Path, required=True)
    torch_tabular_group.add_argument("--out", type=Path, required=True)
    torch_tabular_group.add_argument("--group-field", default="dataset_id")
    torch_tabular_group.add_argument("--test-group", required=True)
    torch_tabular_group.add_argument("--epochs", type=int, default=80)
    torch_tabular_group.add_argument("--learning-rate", type=float, default=0.001)
    torch_tabular_group.add_argument("--hidden-units", type=int, default=32)
    torch_tabular_group.add_argument("--batch-size", type=int, default=64)
    torch_tabular_group.add_argument("--seed", type=int, default=42)
    torch_tabular_group.add_argument("--weight-decay", type=float, default=0.0)
    torch_tabular_group.add_argument("--no-missing-masks", action="store_true")
    torch_tabular_group.set_defaults(func=_train_torch_tabular_group_holdout)

    torch_sequence = subparsers.add_parser("train-torch-sequence-holdout")
    torch_sequence.add_argument("--input", type=Path, required=True)
    torch_sequence.add_argument("--sequence-manifest", type=Path, action="append", required=True)
    torch_sequence.add_argument("--out", type=Path, required=True)
    torch_sequence.add_argument("--time-field", default="window_start_utc")
    torch_sequence.add_argument("--train-fraction", type=float, default=0.8)
    torch_sequence.add_argument("--lookback-steps", type=int, default=60)
    torch_sequence.add_argument("--epochs", type=int, default=40)
    torch_sequence.add_argument("--learning-rate", type=float, default=0.001)
    torch_sequence.add_argument("--hidden-units", type=int, default=24)
    torch_sequence.add_argument("--batch-size", type=int, default=64)
    torch_sequence.add_argument("--seed", type=int, default=42)
    torch_sequence.add_argument("--no-missing-masks", action="store_true")
    torch_sequence.add_argument("--evaluation", action="append", default=[], help="Sequence evaluation name to run; repeatable")
    torch_sequence.set_defaults(func=_train_torch_sequence_holdout)

    torch_sequence_group = subparsers.add_parser("train-torch-sequence-group-holdout")
    torch_sequence_group.add_argument("--input", type=Path, required=True)
    torch_sequence_group.add_argument("--sequence-manifest", type=Path, action="append", required=True)
    torch_sequence_group.add_argument("--out", type=Path, required=True)
    torch_sequence_group.add_argument("--group-field", default="dataset_id")
    torch_sequence_group.add_argument("--test-group", required=True)
    torch_sequence_group.add_argument("--lookback-steps", type=int, default=60)
    torch_sequence_group.add_argument("--epochs", type=int, default=40)
    torch_sequence_group.add_argument("--learning-rate", type=float, default=0.001)
    torch_sequence_group.add_argument("--hidden-units", type=int, default=24)
    torch_sequence_group.add_argument("--batch-size", type=int, default=64)
    torch_sequence_group.add_argument("--seed", type=int, default=42)
    torch_sequence_group.add_argument("--no-missing-masks", action="store_true")
    torch_sequence_group.add_argument("--evaluation", action="append", default=[], help="Sequence evaluation name to run; repeatable")
    torch_sequence_group.set_defaults(func=_train_torch_sequence_group_holdout)

    torch_sequence_split = subparsers.add_parser("train-torch-sequence-split-holdout")
    torch_sequence_split.add_argument("--input", type=Path, required=True)
    torch_sequence_split.add_argument("--sequence-manifest", type=Path, action="append", required=True)
    torch_sequence_split.add_argument("--out", type=Path, required=True)
    torch_sequence_split.add_argument("--split-field", default="model_split")
    torch_sequence_split.add_argument("--train-value", default="train")
    torch_sequence_split.add_argument("--test-value", default="test")
    torch_sequence_split.add_argument("--lookback-steps", type=int, default=60)
    torch_sequence_split.add_argument("--epochs", type=int, default=40)
    torch_sequence_split.add_argument("--learning-rate", type=float, default=0.001)
    torch_sequence_split.add_argument("--hidden-units", type=int, default=24)
    torch_sequence_split.add_argument("--batch-size", type=int, default=64)
    torch_sequence_split.add_argument("--seed", type=int, default=42)
    torch_sequence_split.add_argument("--no-missing-masks", action="store_true")
    torch_sequence_split.add_argument("--evaluation", action="append", default=[], help="Sequence evaluation name to run; repeatable")
    torch_sequence_split.set_defaults(func=_train_torch_sequence_split_holdout)

    torch_patch_split = subparsers.add_parser("train-torch-patch-transformer-split-holdout")
    torch_patch_split.add_argument("--input", type=Path, required=True)
    torch_patch_split.add_argument("--sequence-manifest", type=Path, action="append", required=True)
    torch_patch_split.add_argument("--out", type=Path, required=True)
    torch_patch_split.add_argument("--split-field", default="model_split")
    torch_patch_split.add_argument("--train-value", default="train")
    torch_patch_split.add_argument("--test-value", default="test")
    torch_patch_split.add_argument("--lookback-steps", type=int, default=60)
    torch_patch_split.add_argument("--patch-steps", type=int, default=10)
    torch_patch_split.add_argument("--epochs", type=int, default=20)
    torch_patch_split.add_argument("--learning-rate", type=float, default=0.001)
    torch_patch_split.add_argument("--d-model", type=int, default=32)
    torch_patch_split.add_argument("--layers", type=int, default=2)
    torch_patch_split.add_argument("--heads", type=int, default=2)
    torch_patch_split.add_argument("--dropout", type=float, default=0.1)
    torch_patch_split.add_argument("--batch-size", type=int, default=32)
    torch_patch_split.add_argument("--seed", type=int, default=42)
    torch_patch_split.add_argument("--no-missing-masks", action="store_true")
    torch_patch_split.add_argument("--evaluation", action="append", default=[], help="Sequence evaluation name to run; repeatable")
    torch_patch_split.add_argument("--regression-target", action="append", default=[], help="Optional numeric target field; repeatable")
    torch_patch_split.add_argument("--checkpoint-in", type=Path)
    torch_patch_split.add_argument("--checkpoint-out", type=Path)
    torch_patch_split.set_defaults(func=_train_torch_patch_transformer_split_holdout)


def _train_torch_tabular_holdout(args: Namespace) -> int:
    report = evaluate_torch_tabular_holdout(
        input_csv=args.input,
        out_path=args.out,
        time_field=args.time_field,
        train_fraction=args.train_fraction,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        hidden_units=args.hidden_units,
        batch_size=args.batch_size,
        seed=args.seed,
        include_missing_masks=not args.no_missing_masks,
        weight_decay=args.weight_decay,
    )
    print_holdout_report(report, args.out)
    return 0


def _train_torch_tabular_group_holdout(args: Namespace) -> int:
    report = evaluate_torch_tabular_group_holdout(
        input_csv=args.input,
        out_path=args.out,
        group_field=args.group_field,
        test_group=args.test_group,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        hidden_units=args.hidden_units,
        batch_size=args.batch_size,
        seed=args.seed,
        include_missing_masks=not args.no_missing_masks,
        weight_decay=args.weight_decay,
    )
    print_holdout_report(report, args.out)
    return 0


def _train_torch_sequence_holdout(args: Namespace) -> int:
    report = evaluate_torch_sequence_holdout(
        input_csv=args.input,
        sequence_manifest_paths=args.sequence_manifest,
        out_path=args.out,
        time_field=args.time_field,
        train_fraction=args.train_fraction,
        lookback_steps=args.lookback_steps,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        hidden_units=args.hidden_units,
        batch_size=args.batch_size,
        seed=args.seed,
        include_missing_masks=not args.no_missing_masks,
        evaluation_names=args.evaluation or None,
    )
    print_holdout_report(report, args.out)
    return 0


def _train_torch_sequence_group_holdout(args: Namespace) -> int:
    report = evaluate_torch_sequence_group_holdout(
        input_csv=args.input,
        sequence_manifest_paths=args.sequence_manifest,
        out_path=args.out,
        group_field=args.group_field,
        test_group=args.test_group,
        lookback_steps=args.lookback_steps,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        hidden_units=args.hidden_units,
        batch_size=args.batch_size,
        seed=args.seed,
        include_missing_masks=not args.no_missing_masks,
        evaluation_names=args.evaluation or None,
    )
    print_holdout_report(report, args.out)
    return 0


def _train_torch_sequence_split_holdout(args: Namespace) -> int:
    report = evaluate_torch_sequence_split_holdout(
        input_csv=args.input,
        sequence_manifest_paths=args.sequence_manifest,
        out_path=args.out,
        split_field=args.split_field,
        train_value=args.train_value,
        test_value=args.test_value,
        lookback_steps=args.lookback_steps,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        hidden_units=args.hidden_units,
        batch_size=args.batch_size,
        seed=args.seed,
        include_missing_masks=not args.no_missing_masks,
        evaluation_names=args.evaluation or None,
    )
    print_holdout_report(report, args.out)
    return 0


def _train_torch_patch_transformer_split_holdout(args: Namespace) -> int:
    report = evaluate_torch_patch_transformer_split_holdout(
        input_csv=args.input,
        sequence_manifest_paths=args.sequence_manifest,
        out_path=args.out,
        split_field=args.split_field,
        train_value=args.train_value,
        test_value=args.test_value,
        lookback_steps=args.lookback_steps,
        patch_steps=args.patch_steps,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        d_model=args.d_model,
        layers=args.layers,
        heads=args.heads,
        dropout=args.dropout,
        batch_size=args.batch_size,
        seed=args.seed,
        include_missing_masks=not args.no_missing_masks,
        evaluation_names=args.evaluation or None,
        regression_target_fields=args.regression_target or None,
        checkpoint_in=args.checkpoint_in,
        checkpoint_out=args.checkpoint_out,
    )
    print_holdout_report(report, args.out)
    return 0

"""Model-report CLI commands."""

from __future__ import annotations

from argparse import Namespace, _SubParsersAction
from pathlib import Path

from elfquake.models.comparison import compare_model_run_summaries
from elfquake.models.sequence_comparison import diagnose_sequence_comparison


def register_model_report_commands(subparsers: _SubParsersAction) -> None:
    comparison = subparsers.add_parser("compare-model-run-summaries")
    comparison.add_argument("--summary", type=Path, action="append", required=True)
    comparison.add_argument("--out", type=Path, required=True)
    comparison.add_argument("--csv-out", type=Path)
    comparison.set_defaults(func=_compare_model_run_summaries)

    sequence_diagnostic = subparsers.add_parser("diagnose-sequence-comparison")
    sequence_diagnostic.add_argument("--comparison", type=Path, required=True)
    sequence_diagnostic.add_argument("--out", type=Path, required=True)
    sequence_diagnostic.add_argument("--csv-out", type=Path)
    sequence_diagnostic.set_defaults(func=_diagnose_sequence_comparison)


def _compare_model_run_summaries(args: Namespace) -> int:
    report = compare_model_run_summaries(
        summary_paths=args.summary,
        out_path=args.out,
        csv_out_path=args.csv_out,
    )
    best = report.get("best_calibrated_balanced_accuracy", {})
    print(f"summaries: {report['summary_count']}")
    print(f"reports: {report['report_count']}")
    if best:
        print(f"best calibrated: {best.get('best_calibrated_balanced_accuracy')}")
        print(f"best model: {best.get('model_name')}")
    print(f"output: {args.out}")
    if args.csv_out:
        print(f"csv output: {args.csv_out}")
    return 0


def _diagnose_sequence_comparison(args: Namespace) -> int:
    report = diagnose_sequence_comparison(
        comparison_path=args.comparison,
        out_path=args.out,
        csv_out_path=args.csv_out,
    )
    best = report.get("best_overall", {})
    print(f"evaluation rows: {report['evaluation_row_count']}")
    print(f"group evaluation rows: {report['group_evaluation_row_count']}")
    if best:
        print(f"best calibrated: {best.get('calibrated_test_balanced_accuracy')}")
        print(f"best evaluation: {best.get('evaluation_name')}")
    print(f"output: {args.out}")
    if args.csv_out:
        print(f"csv output: {args.csv_out}")
    return 0

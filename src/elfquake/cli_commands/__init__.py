"""CLI command registration grouped by project concern."""

from __future__ import annotations

from argparse import _SubParsersAction

from elfquake.cli_commands.features import register_feature_commands
from elfquake.cli_commands.model_reports import register_model_report_commands
from elfquake.cli_commands.models import register_model_commands
from elfquake.cli_commands.piezo import register_piezo_commands
from elfquake.cli_commands.sandpile import register_sandpile_commands
from elfquake.cli_commands.self_supervised import register_self_supervised_commands
from elfquake.cli_commands.sources import register_source_commands
from elfquake.cli_commands.synthetic import register_synthetic_commands
from elfquake.cli_commands.visualization import register_visualization_commands


def register_commands(subparsers: _SubParsersAction) -> None:
    register_source_commands(subparsers)
    register_feature_commands(subparsers)
    register_model_commands(subparsers)
    register_model_report_commands(subparsers)
    register_self_supervised_commands(subparsers)
    register_sandpile_commands(subparsers)
    register_synthetic_commands(subparsers)
    register_piezo_commands(subparsers)
    register_visualization_commands(subparsers)

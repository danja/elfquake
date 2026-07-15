"""Visualization CLI commands."""

from __future__ import annotations

from argparse import Namespace, _SubParsersAction
from pathlib import Path

from elfquake.visualization.event_map import DEFAULT_BASEMAP_GEOJSON, render_event_map
from elfquake.visualization.prediction_map import render_prediction_event_map
from elfquake.visualization.transfer_trial_map import render_transfer_trial_map


def register_visualization_commands(subparsers: _SubParsersAction) -> None:
    event_map = subparsers.add_parser("render-event-map")
    event_map.add_argument("--events", type=Path, required=True)
    event_map.add_argument("--out", type=Path, required=True)
    event_map.add_argument("--metadata-out", type=Path)
    event_map.add_argument("--title", default="ELFQuake Italy event map")
    event_map.add_argument("--lon-min", type=float, default=5.5)
    event_map.add_argument("--lon-max", type=float, default=19.5)
    event_map.add_argument("--lat-min", type=float, default=35.0)
    event_map.add_argument("--lat-max", type=float, default=47.8)
    event_map.add_argument("--min-magnitude", type=float)
    event_map.add_argument("--max-events", type=int)
    event_map.add_argument("--basemap-geojson", type=Path)
    event_map.set_defaults(func=_render_event_map)

    prediction_map = subparsers.add_parser("render-prediction-event-map")
    prediction_map.add_argument("--events", type=Path, required=True)
    prediction_map.add_argument("--windows", type=Path, required=True)
    prediction_map.add_argument("--report", type=Path, required=True)
    prediction_map.add_argument("--out", type=Path, required=True)
    prediction_map.add_argument("--metadata-out", type=Path)
    prediction_map.add_argument("--title", default="ELFQuake synthetic actual vs predicted events")
    prediction_map.add_argument("--evaluation")
    prediction_map.add_argument("--threshold", type=float)
    prediction_map.add_argument("--max-actual-events", type=int)
    prediction_map.add_argument("--basemap-geojson", type=Path)
    prediction_map.set_defaults(func=_render_prediction_event_map)

    transfer_map = subparsers.add_parser("render-transfer-trial-map")
    transfer_map.add_argument("--actual-events", type=Path, required=True)
    transfer_map.add_argument("--predictions", type=Path, required=True)
    transfer_map.add_argument("--out", type=Path, required=True)
    transfer_map.add_argument("--metadata-out", type=Path)
    transfer_map.add_argument("--title", default="ELFQuake held-out Italy week: actual vs predicted cells")
    transfer_map.set_defaults(func=_render_transfer_trial_map)


def _render_event_map(args: Namespace) -> int:
    report = render_event_map(
        events_csv=args.events,
        out_path=args.out,
        metadata_out=args.metadata_out,
        title=args.title,
        lon_min=args.lon_min,
        lon_max=args.lon_max,
        lat_min=args.lat_min,
        lat_max=args.lat_max,
        min_magnitude=args.min_magnitude,
        max_events=args.max_events,
        basemap_geojson=args.basemap_geojson or DEFAULT_BASEMAP_GEOJSON,
    )
    print(f"map: {report['map_file']}")
    print(f"events: {report['event_count']}")
    print(f"type: {report['map_type']}")
    if args.metadata_out:
        print(f"metadata: {args.metadata_out}")
    return 0


def _render_prediction_event_map(args: Namespace) -> int:
    report = render_prediction_event_map(
        events_csv=args.events,
        windows_csv=args.windows,
        report_json=args.report,
        out_path=args.out,
        metadata_out=args.metadata_out,
        title=args.title,
        evaluation=args.evaluation,
        threshold=args.threshold,
        max_actual_events=args.max_actual_events,
        basemap_geojson=args.basemap_geojson or DEFAULT_BASEMAP_GEOJSON,
    )
    print(f"map: {report['map_file']}")
    print(f"actual events: {report['actual_event_count']}")
    print(f"predicted event points: {report['predicted_event_point_count']}")
    print(f"predicted windows without location: {report['predicted_without_location_count']}")
    print(f"evaluation: {report['evaluation']}")
    print(f"threshold: {report['threshold']}")
    if args.metadata_out:
        print(f"metadata: {args.metadata_out}")
    return 0


def _render_transfer_trial_map(args: Namespace) -> int:
    report = render_transfer_trial_map(
        actual_csv=args.actual_events,
        predictions_csv=args.predictions,
        out_path=args.out,
        metadata_out=args.metadata_out,
        title=args.title,
    )
    print(f"map: {report['map_file']}")
    print(f"actual events: {report['actual_event_count']}")
    print(f"predicted cells: {report['predicted_cell_count']}")
    return 0

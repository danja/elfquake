"""Synthetic event extraction CLI commands."""

from __future__ import annotations

from argparse import Namespace, _SubParsersAction
from pathlib import Path

from elfquake.sim.avalanche_tuning import tune_avalanche_event_extraction
from elfquake.sim.synthetic_events import build_avalanche_signal_event_list, build_synthetic_event_list


def register_synthetic_commands(subparsers: _SubParsersAction) -> None:
    synthetic_events = subparsers.add_parser("build-synthetic-event-list")
    synthetic_events.add_argument("--summary", type=Path, required=True)
    synthetic_events.add_argument("--sensors", type=Path, required=True)
    synthetic_events.add_argument("--out", type=Path, required=True)
    synthetic_events.add_argument("--grid-width", type=int, required=True)
    synthetic_events.add_argument("--grid-height", type=int, required=True)
    synthetic_events.add_argument("--start-time-utc", default="2026-01-01T00:00:00Z")
    synthetic_events.add_argument("--step-seconds", type=int, default=60)
    synthetic_events.add_argument("--min-topple-count", type=int, default=1)
    synthetic_events.add_argument("--lat-min", type=float, default=41.5)
    synthetic_events.add_argument("--lat-max", type=float, default=43.5)
    synthetic_events.add_argument("--lon-min", type=float, default=12.0)
    synthetic_events.add_argument("--lon-max", type=float, default=14.5)
    synthetic_events.add_argument("--magnitude-type", default="MLs")
    synthetic_events.add_argument("--source", default="elfquake_sandpile_synthetic")
    synthetic_events.add_argument("--ingested-at-utc")
    synthetic_events.set_defaults(func=_build_synthetic_event_list)

    avalanche_events = subparsers.add_parser("build-avalanche-signal-event-list")
    avalanche_events.add_argument("--avalanche", type=Path, required=True)
    avalanche_events.add_argument("--activity", type=Path)
    avalanche_events.add_argument("--out", type=Path, required=True)
    avalanche_events.add_argument("--grid-width", type=int, required=True)
    avalanche_events.add_argument("--grid-height", type=int, required=True)
    avalanche_events.add_argument("--start-time-utc", default="2026-01-01T00:00:00Z")
    avalanche_events.add_argument("--step-seconds", type=int, default=60)
    avalanche_events.add_argument("--min-signal", type=float, default=0.0)
    avalanche_events.add_argument("--min-signal-quantile", type=float, default=0.0)
    avalanche_events.add_argument("--local-max-window", type=int, default=0)
    avalanche_events.add_argument("--max-events", type=int, default=0)
    avalanche_events.add_argument("--lat-min", type=float, default=41.5)
    avalanche_events.add_argument("--lat-max", type=float, default=43.5)
    avalanche_events.add_argument("--lon-min", type=float, default=12.0)
    avalanche_events.add_argument("--lon-max", type=float, default=14.5)
    avalanche_events.add_argument("--spatial-profile", choices=["central_italy", "italy_apennines"], default="italy_apennines")
    avalanche_events.add_argument("--no-fit-spatial-extent", action="store_false", dest="fit_spatial_extent")
    avalanche_events.add_argument("--magnitude-type", default="MLs")
    avalanche_events.add_argument("--source", default="elfquake_avalanche_signal_synthetic")
    avalanche_events.add_argument("--ingested-at-utc")
    avalanche_events.set_defaults(func=_build_avalanche_signal_event_list)

    avalanche_tuning = subparsers.add_parser("tune-avalanche-event-extraction")
    avalanche_tuning.add_argument("--real-events", type=Path, required=True)
    avalanche_tuning.add_argument("--avalanche", type=Path, required=True)
    avalanche_tuning.add_argument("--activity", type=Path)
    avalanche_tuning.add_argument("--out", type=Path, required=True)
    avalanche_tuning.add_argument("--work-dir", type=Path)
    avalanche_tuning.add_argument("--grid-width", type=int, required=True)
    avalanche_tuning.add_argument("--grid-height", type=int, required=True)
    avalanche_tuning.add_argument("--quantile", type=float, action="append")
    avalanche_tuning.add_argument("--local-max-window", type=int, action="append")
    avalanche_tuning.add_argument("--max-events", type=int, action="append")
    avalanche_tuning.add_argument("--start-time-utc", default="2026-01-01T00:00:00Z")
    avalanche_tuning.add_argument("--step-seconds", type=int, default=60)
    avalanche_tuning.add_argument("--event-bin-seconds", type=int, default=3600)
    avalanche_tuning.set_defaults(func=_tune_avalanche_event_extraction)


def _build_synthetic_event_list(args: Namespace) -> int:
    rows = build_synthetic_event_list(
        summary_csv=args.summary,
        sensors_csv=args.sensors,
        out_path=args.out,
        grid_width=args.grid_width,
        grid_height=args.grid_height,
        start_time_utc=args.start_time_utc,
        step_seconds=args.step_seconds,
        min_topple_count=args.min_topple_count,
        lat_min=args.lat_min,
        lat_max=args.lat_max,
        lon_min=args.lon_min,
        lon_max=args.lon_max,
        magnitude_type=args.magnitude_type,
        source=args.source,
        ingested_at_utc=args.ingested_at_utc,
    )
    print(f"synthetic events: {len(rows)}")
    print(f"output: {args.out}")
    return 0


def _build_avalanche_signal_event_list(args: Namespace) -> int:
    rows = build_avalanche_signal_event_list(
        avalanche_csv=args.avalanche,
        avalanche_activity_csv=args.activity,
        out_path=args.out,
        grid_width=args.grid_width,
        grid_height=args.grid_height,
        start_time_utc=args.start_time_utc,
        step_seconds=args.step_seconds,
        min_signal=args.min_signal,
        min_signal_quantile=args.min_signal_quantile,
        local_max_window=args.local_max_window,
        max_events=args.max_events,
        lat_min=args.lat_min,
        lat_max=args.lat_max,
        lon_min=args.lon_min,
        lon_max=args.lon_max,
        spatial_profile=args.spatial_profile,
        fit_spatial_extent=args.fit_spatial_extent,
        magnitude_type=args.magnitude_type,
        source=args.source,
        ingested_at_utc=args.ingested_at_utc,
    )
    print(f"avalanche signal events: {len(rows)}")
    print(f"output: {args.out}")
    return 0


def _tune_avalanche_event_extraction(args: Namespace) -> int:
    rows = tune_avalanche_event_extraction(
        real_events_csv=args.real_events,
        avalanche_csv=args.avalanche,
        avalanche_activity_csv=args.activity,
        out_path=args.out,
        work_dir=args.work_dir,
        grid_width=args.grid_width,
        grid_height=args.grid_height,
        quantiles=args.quantile,
        local_max_windows=args.local_max_window,
        max_events_values=args.max_events,
        start_time_utc=args.start_time_utc,
        step_seconds=args.step_seconds,
        event_bin_seconds=args.event_bin_seconds,
    )
    print(f"grid rows: {len(rows)}")
    if rows:
        print(
            "best: "
            f"q={rows[0]['min_signal_quantile']} "
            f"window={rows[0]['local_max_window']} "
            f"max_events={rows[0]['max_events']} "
            f"shape_score={rows[0]['shape_score']} "
            f"distance={rows[0]['normalized_distance']}"
        )
    print(f"output: {args.out}")
    return 0

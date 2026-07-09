"""Sandpile simulation CLI commands."""

from __future__ import annotations

import time
from argparse import Namespace, _SubParsersAction
from pathlib import Path

from elfquake.sim.piezo import PiezoConfig
from elfquake.sim.sandpile import SandpileConfig, run_sandpile_simulation


def register_sandpile_commands(subparsers: _SubParsersAction) -> None:
    sandpile = subparsers.add_parser("run-sandpile-sim")
    sandpile.add_argument("--width", type=int, default=128)
    sandpile.add_argument("--height", type=int, default=128)
    sandpile.add_argument("--steps", type=int, default=100)
    sandpile.add_argument("--threshold", type=int, default=4)
    sandpile.add_argument("--source-count", type=int, default=16)
    sandpile.add_argument("--sensor-count", type=int, default=16)
    sandpile.add_argument("--deposition-probability", type=float, default=0.5)
    sandpile.add_argument("--seed", type=int, default=1)
    sandpile.add_argument("--max-relaxation-sweeps", type=int, default=10000)
    sandpile.add_argument("--deposition-mode", choices=["sources", "uniform"], default="sources")
    sandpile.add_argument("--target-mean-height", type=float)
    sandpile.add_argument("--target-fill-limit", type=int, default=0)
    sandpile.add_argument("--bottom-layer-removal-interval", type=int, default=0)
    sandpile.add_argument("--initial-fill-mode", choices=["none", "random", "structured"], default="none")
    sandpile.add_argument("--initial-fill-mean-height", type=float, default=0.0)
    sandpile.add_argument("--initial-fill-variation", type=float, default=0.0)
    sandpile.add_argument("--initial-fill-smooth-passes", type=int, default=0)
    sandpile.add_argument("--warmup-steps", type=int, default=0)
    sandpile.add_argument("--mountain-mode", action="store_true")
    sandpile.add_argument("--summary-out", type=Path, required=True)
    sandpile.add_argument("--sensors-out", type=Path, required=True)
    sandpile.add_argument("--piezo-out", type=Path)
    sandpile.add_argument("--avalanche-signal-out", type=Path)
    sandpile.add_argument("--avalanche-activity-out", type=Path)
    sandpile.add_argument("--piezo-avalanche-out", type=Path)
    sandpile.add_argument("--piezo-sensor-count", type=int, default=16)
    sandpile.add_argument("--piezo-susceptibility-base", type=float, default=0.15)
    sandpile.add_argument("--piezo-susceptibility-variation", type=float, default=0.85)
    sandpile.add_argument("--piezo-cluster-count", type=int, default=8)
    sandpile.add_argument("--piezo-cluster-radius", type=float, default=0.0)
    sandpile.add_argument("--piezo-activation-ratio", type=float, default=0.75)
    sandpile.add_argument("--piezo-attenuation-radius", type=float, default=16.0)
    sandpile.add_argument("--piezo-max-distance-radius", type=float, default=48.0)
    sandpile.add_argument("--piezo-charge-decay", type=float, default=0.995)
    sandpile.add_argument("--piezo-charge-coupling", type=float, default=1.0)
    sandpile.add_argument("--piezo-release-charge-threshold", type=float, default=40.0)
    sandpile.add_argument("--piezo-release-ratio", type=float, default=0.25)
    sandpile.add_argument("--piezo-critical-release-ratio", type=float, default=0.10)
    sandpile.add_argument("--piezo-saturation", type=float, default=1000.0)
    sandpile.add_argument("--avalanche-signal-attenuation-radius", type=float, default=0.0)
    sandpile.add_argument("--avalanche-signal-max-distance-radius", type=float, default=0.0)
    sandpile.add_argument("--snapshot-dir", type=Path)
    sandpile.add_argument("--snapshot-interval", type=int, default=0)
    sandpile.add_argument("--heatmap-dir", type=Path)
    sandpile.add_argument("--heatmap-scale", type=int, default=8)
    sandpile.add_argument("--heatmap-color-min", type=float, default=0.0)
    sandpile.add_argument("--heatmap-color-max", type=float)
    sandpile.add_argument("--heatmap-gamma", type=float, default=1.0)
    sandpile.add_argument("--heatmap-workers", type=int, default=1)
    sandpile.add_argument("--heatmap-progress-interval", type=int, default=50)
    sandpile.add_argument("--progress-interval", type=int, default=100)
    sandpile.set_defaults(func=_run_sandpile_sim)

    sandpile_summary = subparsers.add_parser("summarize-sandpile-sim")
    sandpile_summary.add_argument("--summary", type=Path, required=True)
    sandpile_summary.add_argument("--sensors", type=Path, required=True)
    sandpile_summary.add_argument("--out", type=Path, required=True)
    sandpile_summary.set_defaults(func=_summarize_sandpile_sim)

    sandpile_benchmark = subparsers.add_parser("benchmark-sandpile-sim")
    sandpile_benchmark.add_argument("--width", type=int, default=64)
    sandpile_benchmark.add_argument("--height", type=int, default=64)
    sandpile_benchmark.add_argument("--steps", type=int, default=100)
    sandpile_benchmark.add_argument("--threshold", type=int, default=4)
    sandpile_benchmark.add_argument("--source-count", type=int, default=16)
    sandpile_benchmark.add_argument("--sensor-count", type=int, default=16)
    sandpile_benchmark.add_argument("--deposition-probability", type=float, default=0.5)
    sandpile_benchmark.add_argument("--seed", type=int, default=1)
    sandpile_benchmark.add_argument("--max-relaxation-sweeps", type=int, default=10000)
    sandpile_benchmark.add_argument("--out", type=Path, required=True)
    sandpile_benchmark.set_defaults(func=_benchmark_sandpile_sim)

    sandpile_heatmap = subparsers.add_parser("render-sandpile-heatmap")
    sandpile_heatmap.add_argument("--snapshot", type=Path, required=True)
    sandpile_heatmap.add_argument("--out", type=Path, required=True)
    sandpile_heatmap.add_argument("--scale", type=int, default=8)
    sandpile_heatmap.add_argument("--color-min", type=float, default=0.0)
    sandpile_heatmap.add_argument("--color-max", type=float)
    sandpile_heatmap.add_argument("--gamma", type=float, default=1.0)
    sandpile_heatmap.set_defaults(func=_render_sandpile_heatmap)


def _run_sandpile_sim(args: Namespace) -> int:
    started = time.perf_counter()
    target_mean_height = (
        args.target_mean_height
        if args.target_mean_height is not None
        else (args.width / 2 if args.mountain_mode else 0.0)
    )
    bottom_layer_removal_interval = (
        args.bottom_layer_removal_interval
        if args.bottom_layer_removal_interval
        else (100 if args.mountain_mode else 0)
    )

    def report_progress(completed_steps: int, total_steps: int, row: dict[str, str]) -> None:
        elapsed_seconds = time.perf_counter() - started
        rate = completed_steps / elapsed_seconds if elapsed_seconds else 0.0
        print(
            "progress: "
            f"step {completed_steps}/{total_steps} "
            f"elapsed {elapsed_seconds:.2f}s "
            f"rate {rate:.2f} steps/s "
            f"topples {row['topple_count']} "
            f"max_height {row['max_height']} "
            f"mean_height {row['mean_height']} "
            f"bottom_removed {row.get('bottom_layer_removed_mass', '0')} "
            f"safety_release {row.get('safety_released_mass', '0')}",
            flush=True,
        )

    summary_rows, sensor_rows = run_sandpile_simulation(
        config=SandpileConfig(
            width=args.width,
            height=args.height,
            steps=args.steps,
            threshold=args.threshold,
            source_count=args.source_count,
            sensor_count=args.sensor_count,
            deposition_probability=args.deposition_probability,
            seed=args.seed,
            max_relaxation_sweeps=args.max_relaxation_sweeps,
            deposition_mode=args.deposition_mode,
            target_mean_height=target_mean_height,
            target_fill_limit=args.target_fill_limit,
            bottom_layer_removal_interval=bottom_layer_removal_interval,
            initial_fill_mode=args.initial_fill_mode,
            initial_fill_mean_height=args.initial_fill_mean_height,
            initial_fill_variation=args.initial_fill_variation,
            initial_fill_smooth_passes=args.initial_fill_smooth_passes,
            warmup_steps=args.warmup_steps,
        ),
        summary_out=args.summary_out,
        sensors_out=args.sensors_out,
        piezo_out=args.piezo_out,
        avalanche_signal_out=args.avalanche_signal_out,
        avalanche_activity_out=args.avalanche_activity_out,
        piezo_avalanche_out=args.piezo_avalanche_out,
        piezo_config=_piezo_config(args) if args.piezo_out or args.avalanche_signal_out or args.piezo_avalanche_out else None,
        avalanche_signal_config=PiezoConfig(
            attenuation_radius=args.avalanche_signal_attenuation_radius,
            max_distance_radius=args.avalanche_signal_max_distance_radius,
        )
        if args.avalanche_signal_out or args.piezo_avalanche_out
        else None,
        snapshot_dir=args.snapshot_dir,
        snapshot_interval=args.snapshot_interval,
        progress_interval=args.progress_interval,
        progress_callback=report_progress if args.progress_interval else None,
    )
    heatmap_rows = _render_snapshot_heatmaps(args) if args.heatmap_dir else []
    print(f"summary rows: {len(summary_rows)}")
    print(f"sensor rows: {len(sensor_rows)}")
    print(f"summary output: {args.summary_out}")
    print(f"sensors output: {args.sensors_out}")
    if args.piezo_out:
        print(f"piezo output: {args.piezo_out}")
    direct_avalanche_out = args.avalanche_signal_out or args.piezo_avalanche_out
    if direct_avalanche_out:
        print(f"avalanche signal output: {direct_avalanche_out}")
    if args.avalanche_activity_out:
        print(f"avalanche activity output: {args.avalanche_activity_out}")
    if args.snapshot_dir:
        print(f"snapshot dir: {args.snapshot_dir}")
    if args.heatmap_dir:
        print(f"heatmap rows: {len(heatmap_rows)}")
        print(f"heatmap dir: {args.heatmap_dir}")
    return 0


def _summarize_sandpile_sim(args: Namespace) -> int:
    from elfquake.sim.report import summarize_sandpile_outputs

    report = summarize_sandpile_outputs(summary_csv=args.summary, sensors_csv=args.sensors, out_path=args.out)
    print(f"status: {report['status']}")
    print(f"summary rows: {report['summary_row_count']}")
    print(f"sensor rows: {report['sensor_row_count']}")
    print(f"avalanche steps: {report['avalanche_step_count']}")
    print(f"output: {args.out}")
    return 0


def _benchmark_sandpile_sim(args: Namespace) -> int:
    from elfquake.sim.report import benchmark_sandpile_simulation

    report = benchmark_sandpile_simulation(
        config=SandpileConfig(
            width=args.width,
            height=args.height,
            steps=args.steps,
            threshold=args.threshold,
            source_count=args.source_count,
            sensor_count=args.sensor_count,
            deposition_probability=args.deposition_probability,
            seed=args.seed,
            max_relaxation_sweeps=args.max_relaxation_sweeps,
        ),
        out_path=args.out,
    )
    print(f"status: {report['status']}")
    print(f"backend: {report['backend']}")
    print(f"steps per second: {report['steps_per_second']}")
    print(f"output: {args.out}")
    return 0


def _render_sandpile_heatmap(args: Namespace) -> int:
    from elfquake.sim.heatmap import render_sandpile_heatmap

    report = render_sandpile_heatmap(
        snapshot_path=args.snapshot,
        out_path=args.out,
        scale=args.scale,
        color_min=args.color_min,
        color_max=args.color_max,
        gamma=args.gamma,
    )
    print(f"snapshot: {report['snapshot_file']}")
    print(f"heatmap: {report['heatmap_file']}")
    print(f"image size: {report['width_px']}x{report['height_px']}")
    print(f"max height: {report['max_height']}")
    print(f"color max: {report['color_max']}")
    return 0


def _piezo_config(args: Namespace) -> PiezoConfig:
    return PiezoConfig(
        sensor_count=args.piezo_sensor_count,
        susceptibility_base=args.piezo_susceptibility_base,
        susceptibility_variation=args.piezo_susceptibility_variation,
        cluster_count=args.piezo_cluster_count,
        cluster_radius=args.piezo_cluster_radius,
        activation_ratio=args.piezo_activation_ratio,
        attenuation_radius=args.piezo_attenuation_radius,
        max_distance_radius=args.piezo_max_distance_radius,
        charge_decay=args.piezo_charge_decay,
        charge_coupling=args.piezo_charge_coupling,
        release_charge_threshold=args.piezo_release_charge_threshold,
        release_ratio=args.piezo_release_ratio,
        critical_release_ratio=args.piezo_critical_release_ratio,
        saturation=args.piezo_saturation,
    )


def _render_snapshot_heatmaps(args: Namespace) -> list[dict[str, str]]:
    if not args.snapshot_dir:
        raise ValueError("--heatmap-dir requires --snapshot-dir")
    from elfquake.sim.heatmap import render_sandpile_heatmaps_from_manifest

    heatmap_started = time.perf_counter()

    def report_heatmap_progress(completed: int, total: int, row: dict[str, str]) -> None:
        elapsed_seconds = time.perf_counter() - heatmap_started
        rate = completed / elapsed_seconds if elapsed_seconds else 0.0
        print(
            "heatmap progress: "
            f"frame {completed}/{total} "
            f"elapsed {elapsed_seconds:.2f}s "
            f"rate {rate:.2f} frames/s "
            f"latest {Path(row['heatmap_file']).name}",
            flush=True,
        )

    print(
        "heatmap rendering: "
        f"workers {args.heatmap_workers} "
        f"scale {args.heatmap_scale} "
        f"color_min {args.heatmap_color_min} "
        f"color_max {args.heatmap_color_max if args.heatmap_color_max is not None else 'auto'} "
        f"gamma {args.heatmap_gamma}",
        flush=True,
    )
    return render_sandpile_heatmaps_from_manifest(
        manifest_path=args.snapshot_dir / "manifest.csv",
        out_dir=args.heatmap_dir,
        scale=args.heatmap_scale,
        color_min=args.heatmap_color_min,
        color_max=args.heatmap_color_max,
        gamma=args.heatmap_gamma,
        workers=args.heatmap_workers,
        progress_interval=args.heatmap_progress_interval,
        progress_callback=report_heatmap_progress if args.heatmap_progress_interval else None,
    )

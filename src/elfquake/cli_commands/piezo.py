"""Piezo rendering and signal-shape CLI commands."""

from __future__ import annotations

from argparse import Namespace, _SubParsersAction
from pathlib import Path

from elfquake.cli_commands.common import resolve_image_paths
from elfquake.features.signal_shape_compare import (
    compare_signal_shapes,
    event_energy_series,
    scan_sensor_signal_shapes,
    sensor_signal_series,
    vlf_image_column_series,
)
from elfquake.sim.piezo_spectrogram import (
    render_piezo_audio,
    render_piezo_spectrogram,
    render_piezo_strain_vlf_summary,
    render_piezo_timeseries_spectrogram,
)


def register_piezo_commands(subparsers: _SubParsersAction) -> None:
    piezo_spectrogram = subparsers.add_parser("render-piezo-spectrogram")
    _add_piezo_image_common(piezo_spectrogram)
    piezo_spectrogram.add_argument("--step-seconds", type=int, default=60)
    piezo_spectrogram.add_argument("--freq-min", type=float, default=0.0)
    piezo_spectrogram.add_argument("--freq-max", type=float)
    piezo_spectrogram.add_argument("--freq-bins", type=int, default=96)
    piezo_spectrogram.add_argument("--window-steps", type=int, default=64)
    piezo_spectrogram.add_argument("--scale", type=int, default=4)
    piezo_spectrogram.add_argument("--gamma", type=float, default=0.85)
    piezo_spectrogram.add_argument("--sensor-id", type=int)
    piezo_spectrogram.add_argument("--dc-block", type=float, default=0.0)
    piezo_spectrogram.set_defaults(func=_render_piezo_spectrogram)

    piezo_summary = subparsers.add_parser("render-piezo-summary")
    _add_piezo_image_common(piezo_summary)
    piezo_summary.add_argument("--step-seconds", type=int, default=60)
    piezo_summary.add_argument("--freq-min", type=float, default=0.0)
    piezo_summary.add_argument("--freq-max", type=float)
    piezo_summary.add_argument("--freq-bins", type=int, default=96)
    piezo_summary.add_argument("--window-steps", type=int, default=64)
    piezo_summary.add_argument("--scale", type=int, default=4)
    piezo_summary.add_argument("--gamma", type=float, default=0.85)
    piezo_summary.add_argument("--timeseries-height", type=int, default=48)
    piezo_summary.add_argument("--output-width", type=int, default=1600)
    piezo_summary.add_argument("--sensor-id", type=int)
    piezo_summary.add_argument("--dc-block", type=float, default=0.0)
    piezo_summary.set_defaults(func=_render_piezo_summary)

    piezo_audio = subparsers.add_parser("render-piezo-audio")
    piezo_audio.add_argument("--piezo", type=Path, required=True)
    piezo_audio.add_argument("--out", type=Path, required=True)
    piezo_audio.add_argument("--sample-rate", type=int, default=44100)
    piezo_audio.add_argument("--duration-seconds", type=float, default=20.0)
    piezo_audio.add_argument("--gain", type=float, default=0.95)
    piezo_audio.add_argument("--smooth-steps", type=int, default=64)
    piezo_audio.add_argument("--sensor-id", type=int)
    piezo_audio.add_argument("--dc-block", type=float, default=0.0)
    piezo_audio.set_defaults(func=_render_piezo_audio)

    piezo_vlf_summary = subparsers.add_parser("render-piezo-vlf-summary")
    _add_piezo_image_common(piezo_vlf_summary)
    piezo_vlf_summary.add_argument("--carrier-freq-min-hz", type=float, default=0.0)
    piezo_vlf_summary.add_argument("--carrier-freq-max-hz", type=float, default=24000.0)
    piezo_vlf_summary.add_argument("--freq-bins", type=int, default=192)
    piezo_vlf_summary.add_argument("--scale", type=int, default=4)
    piezo_vlf_summary.add_argument("--gamma", type=float, default=0.85)
    piezo_vlf_summary.add_argument("--timeseries-height", type=int, default=48)
    piezo_vlf_summary.add_argument("--output-width", type=int, default=1600)
    piezo_vlf_summary.add_argument("--sensor-id", type=int)
    piezo_vlf_summary.add_argument("--dc-block", type=float, default=0.995)
    piezo_vlf_summary.add_argument("--display-color-quantile", type=float, default=0.82)
    piezo_vlf_summary.set_defaults(func=_render_piezo_vlf_summary)

    shape_compare = subparsers.add_parser("compare-signal-shapes")
    shape_compare.add_argument("--real-events", type=Path)
    shape_compare.add_argument("--synthetic-events", type=Path)
    shape_compare.add_argument("--real-vlf-image", type=Path, action="append", default=[])
    shape_compare.add_argument("--real-vlf-image-root", type=Path, action="append", default=[])
    shape_compare.add_argument("--real-vlf-filename-prefix", action="append", default=["last_E_VLF"])
    shape_compare.add_argument("--sim-piezo", type=Path)
    shape_compare.add_argument("--sim-piezo-sensor-id", type=int)
    shape_compare.add_argument("--sim-avalanche", type=Path)
    shape_compare.add_argument("--sim-avalanche-sensor-id", type=int)
    shape_compare.add_argument("--event-bin-seconds", type=int, default=3600)
    shape_compare.add_argument("--sim-step-seconds", type=float, default=60.0)
    shape_compare.add_argument("--vlf-column-seconds", type=float, default=1.0)
    _add_vlf_crop_args(shape_compare)
    shape_compare.add_argument("--series-out", type=Path, required=True)
    shape_compare.add_argument("--pairs-out", type=Path, required=True)
    shape_compare.set_defaults(func=_compare_signal_shapes)

    piezo_sensor_scan = subparsers.add_parser("scan-piezo-sensors")
    piezo_sensor_scan.add_argument("--real-vlf-image", type=Path, action="append", default=[])
    piezo_sensor_scan.add_argument("--real-vlf-image-root", type=Path, action="append", default=[])
    piezo_sensor_scan.add_argument("--real-vlf-filename-prefix", action="append", default=["last_E_VLF"])
    piezo_sensor_scan.add_argument("--sim-piezo", type=Path, required=True)
    piezo_sensor_scan.add_argument("--sensor-id", type=int, action="append")
    piezo_sensor_scan.add_argument("--sim-step-seconds", type=float, default=60.0)
    piezo_sensor_scan.add_argument("--vlf-column-seconds", type=float, default=1.0)
    _add_vlf_crop_args(piezo_sensor_scan)
    piezo_sensor_scan.add_argument("--out", type=Path, required=True)
    piezo_sensor_scan.set_defaults(func=_scan_piezo_sensors)


def _render_piezo_spectrogram(args: Namespace) -> int:
    report = render_piezo_spectrogram(
        piezo_csv=args.piezo,
        out_path=args.out,
        metadata_out=args.metadata_out,
        start_time_utc=args.start_time_utc,
        step_seconds=args.step_seconds,
        freq_min=args.freq_min,
        freq_max=args.freq_max,
        freq_bins=args.freq_bins,
        window_steps=args.window_steps,
        scale=args.scale,
        gamma=args.gamma,
        sensor_id=args.sensor_id,
        dc_block=args.dc_block,
    )
    print(f"spectrogram: {report['spectrogram_file']}")
    _print_piezo_fft_report(report)
    if args.metadata_out:
        print(f"metadata: {args.metadata_out}")
    return 0


def _render_piezo_summary(args: Namespace) -> int:
    report = render_piezo_timeseries_spectrogram(
        piezo_csv=args.piezo,
        out_path=args.out,
        metadata_out=args.metadata_out,
        start_time_utc=args.start_time_utc,
        step_seconds=args.step_seconds,
        freq_min=args.freq_min,
        freq_max=args.freq_max,
        freq_bins=args.freq_bins,
        window_steps=args.window_steps,
        scale=args.scale,
        gamma=args.gamma,
        timeseries_height=args.timeseries_height,
        output_width=args.output_width,
        sensor_id=args.sensor_id,
        dc_block=args.dc_block,
    )
    print(f"image: {report['image_file']}")
    _print_piezo_fft_report(report)
    if args.metadata_out:
        print(f"metadata: {args.metadata_out}")
    return 0


def _render_piezo_audio(args: Namespace) -> int:
    report = render_piezo_audio(
        piezo_csv=args.piezo,
        out_path=args.out,
        sample_rate=args.sample_rate,
        duration_seconds=args.duration_seconds,
        gain=args.gain,
        smooth_steps=args.smooth_steps,
        sensor_id=args.sensor_id,
        dc_block=args.dc_block,
    )
    print(f"audio: {report['audio_file']}")
    print(f"duration: {report['duration_seconds']}s")
    print(f"sample rate: {report['sample_rate_hz']} Hz")
    print(f"smooth steps: {report['smooth_steps']}")
    print(f"selected sensor: {report['selected_sensor_id'] or 'sum'}")
    print(f"dc block: {report['dc_block']}")
    print(f"type: {report['audio_type']}")
    return 0


def _render_piezo_vlf_summary(args: Namespace) -> int:
    report = render_piezo_strain_vlf_summary(
        piezo_csv=args.piezo,
        out_path=args.out,
        metadata_out=args.metadata_out,
        start_time_utc=args.start_time_utc,
        carrier_freq_min_hz=args.carrier_freq_min_hz,
        carrier_freq_max_hz=args.carrier_freq_max_hz,
        freq_bins=args.freq_bins,
        scale=args.scale,
        gamma=args.gamma,
        timeseries_height=args.timeseries_height,
        output_width=args.output_width,
        sensor_id=args.sensor_id,
        dc_block=args.dc_block,
    )
    print(f"image: {report['image_file']}")
    print(f"steps: {report['step_count']}")
    print(f"sensors: {report['sensor_count']}")
    print(f"selected sensor: {report['selected_sensor_id'] or 'sum'}")
    print(f"carrier frequency range: {report['carrier_freq_min_hz']}..{report['carrier_freq_max_hz']} Hz")
    print(f"type: {report['plot_type']}")
    if args.metadata_out:
        print(f"metadata: {args.metadata_out}")
    return 0


def _compare_signal_shapes(args: Namespace) -> int:
    series = []
    if args.real_events:
        series.append(event_energy_series(
            series_id="real_seismic_events",
            events_csv=args.real_events,
            bin_seconds=args.event_bin_seconds,
        ))
    if args.synthetic_events:
        series.append(event_energy_series(
            series_id="synthetic_seismic_events",
            events_csv=args.synthetic_events,
            bin_seconds=args.event_bin_seconds,
        ))
    if args.real_vlf_image or args.real_vlf_image_root:
        series.append(_real_vlf_column_series(args))
    if args.sim_piezo:
        series.append(sensor_signal_series(
            series_id="synthetic_piezo_vlf_signal",
            signal_csv=args.sim_piezo,
            signal_field="piezo_signal",
            sample_seconds=args.sim_step_seconds,
            sensor_id=args.sim_piezo_sensor_id,
        ))
    if args.sim_avalanche:
        series.append(sensor_signal_series(
            series_id="synthetic_avalanche_signal",
            signal_csv=args.sim_avalanche,
            signal_field="avalanche_signal",
            sample_seconds=args.sim_step_seconds,
            sensor_id=args.sim_avalanche_sensor_id,
        ))
    series_rows, pair_rows = compare_signal_shapes(series=series, series_out=args.series_out, pairs_out=args.pairs_out)
    print(f"series: {len(series_rows)}")
    print(f"pairs: {len(pair_rows)}")
    print(f"series output: {args.series_out}")
    print(f"pairs output: {args.pairs_out}")
    return 0


def _scan_piezo_sensors(args: Namespace) -> int:
    rows = scan_sensor_signal_shapes(
        reference=_real_vlf_column_series(args),
        signal_csv=args.sim_piezo,
        signal_field="piezo_signal",
        out_path=args.out,
        sample_seconds=args.sim_step_seconds,
        sensor_ids=args.sensor_id,
        series_id_prefix="synthetic_piezo_sensor",
    )
    print(f"sensors: {len(rows)}")
    if rows:
        print(f"best sensor: {rows[0]['sensor_id']}")
        print(f"best shape score: {rows[0]['shape_score']}")
    print(f"output: {args.out}")
    return 0


def _add_piezo_image_common(parser) -> None:
    parser.add_argument("--piezo", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--metadata-out", type=Path)
    parser.add_argument("--start-time-utc", default="2026-01-01T00:00:00Z")


def _add_vlf_crop_args(parser) -> None:
    parser.add_argument("--vlf-crop-left", type=float, default=0.0)
    parser.add_argument("--vlf-crop-top", type=float, default=0.13)
    parser.add_argument("--vlf-crop-right", type=float, default=0.83)
    parser.add_argument("--vlf-crop-bottom", type=float, default=0.95)


def _real_vlf_column_series(args: Namespace):
    return vlf_image_column_series(
        series_id="real_vlf_image_columns",
        image_paths=resolve_image_paths(
            image_paths=args.real_vlf_image,
            image_roots=args.real_vlf_image_root,
            filename_prefixes=args.real_vlf_filename_prefix,
        ),
        sample_seconds=args.vlf_column_seconds,
        crop_left=args.vlf_crop_left,
        crop_top=args.vlf_crop_top,
        crop_right=args.vlf_crop_right,
        crop_bottom=args.vlf_crop_bottom,
    )


def _print_piezo_fft_report(report: dict[str, str]) -> None:
    print(f"steps: {report['step_count']}")
    print(f"sensors: {report['sensor_count']}")
    print(f"selected sensor: {report['selected_sensor_id'] or 'sum'}")
    print(f"dc block: {report['dc_block']}")
    print(f"frequency range: {report['freq_min_hz']}..{report['freq_max_hz']} Hz")
    print(f"nyquist: {report['nyquist_hz']} Hz")

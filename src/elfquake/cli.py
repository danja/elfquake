"""Command line entry points for raw data acquisition."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from urllib.error import HTTPError, URLError

from elfquake.backfill import plan_ingv_backfill
from elfquake.connectors.astronomy import fetch_manifest_json
from elfquake.connectors.ingv import fetch_italy_events
from elfquake.connectors.space_archives import (
    fetch_gfz_kp_ap,
    fetch_kyoto_dst_month,
    fetch_ncei_goes15_xrs_year,
    fetch_spaceweather_canada_f107_daily,
)
from elfquake.connectors.vlf_cumiana import fetch_manifest_images, repeat_manifest_images
from elfquake.features.astronomy import build_astronomy_features
from elfquake.features.design_matrix import build_design_matrix
from elfquake.features.multimodal_design import join_vlf_design_matrix
from elfquake.features.multimodal_smoke import build_multimodal_smoke_row
from elfquake.features.prospective import build_prospective_vlf_windows, update_prospective_vlf_table
from elfquake.features.prospective_report import summarize_prospective_table
from elfquake.features.table import build_multimodal_table_from_manifest, write_multimodal_manifest_template
from elfquake.features.targets import label_multimodal_targets
from elfquake.features.training_windows import build_seismic_training_windows
from elfquake.features.vlf import build_vlf_features
from elfquake.features.vlf_image import build_vlf_image_features
from elfquake.features.vlf_image_windows import join_vlf_image_features_to_windows
from elfquake.features.vlf_windows import build_vlf_window_features
from elfquake.models.ablation_smoke import train_ablation_smoke
from elfquake.models.logistic_smoke import train_logistic_smoke
from elfquake.models.readiness import summarize_model_readiness
from elfquake.normalize.events import combine_normalized_events
from elfquake.normalize.ingv import normalize_ingv_event_text
from elfquake.normalize.space_weather import (
    normalize_f107_daily,
    normalize_gfz_kp_ap,
    normalize_goes_xrs_netcdf,
    normalize_kyoto_dst_text,
)


def main() -> int:
    parser = argparse.ArgumentParser(prog="elfquake")
    subparsers = parser.add_subparsers(dest="command", required=True)

    ingv = subparsers.add_parser("fetch-ingv-events")
    ingv.add_argument("--start", required=True, help="UTC start, e.g. 2026-06-22T00:00:00Z")
    ingv.add_argument("--end", required=True, help="UTC end, e.g. 2026-06-29T23:59:59Z")
    ingv.add_argument("--out-root", type=Path, default=Path("data/raw/ingv"))
    ingv.add_argument("--min-mag", type=float, default=2.0)
    ingv.add_argument("--limit", type=int, default=10000)

    ingv_plan = subparsers.add_parser("plan-ingv-backfill")
    ingv_plan.add_argument("--start", required=True)
    ingv_plan.add_argument("--end", required=True)
    ingv_plan.add_argument("--chunk-days", type=int, default=14)
    ingv_plan.add_argument("--min-mag", default="2.0")
    ingv_plan.add_argument("--out", type=Path, required=True)

    vlf = subparsers.add_parser("fetch-vlf-cumiana")
    vlf.add_argument("--manifest", type=Path, default=Path("data/raw/vlf/cumiana/manifest.csv"))
    vlf.add_argument("--out-root", type=Path, default=Path("data/raw/vlf/cumiana"))
    vlf.add_argument("--only", action="append", default=[], help="Endpoint id to fetch; repeatable")

    vlf_loop = subparsers.add_parser("capture-vlf-cumiana-loop")
    vlf_loop.add_argument("--manifest", type=Path, default=Path("data/raw/vlf/cumiana/manifest.csv"))
    vlf_loop.add_argument("--out-root", type=Path, default=Path("data/raw/vlf/cumiana"))
    vlf_loop.add_argument("--only", action="append", default=[], help="Endpoint id to fetch; repeatable")
    vlf_loop.add_argument("--cycles", type=int, default=2, help="0 means run forever")
    vlf_loop.add_argument("--interval-seconds", type=int, default=1800)

    astro = subparsers.add_parser("fetch-astronomy")
    astro.add_argument("--manifest", type=Path, default=Path("data/raw/astronomy/manifest.csv"))
    astro.add_argument("--out-root", type=Path, default=Path("data/raw/astronomy"))
    astro.add_argument("--date", required=True, help="Date for parameterized endpoints, YYYY-MM-DD")
    astro.add_argument("--moon-phases", type=int, default=8)
    astro.add_argument("--only", action="append", default=[], help="Source id to fetch; repeatable")

    normalize = subparsers.add_parser("normalize-ingv-events")
    normalize.add_argument("--raw", type=Path, required=True)
    normalize.add_argument("--out", type=Path, required=True)
    normalize.add_argument("--raw-uri")
    normalize.add_argument("--ingested-at-utc")
    normalize.add_argument("--only-region")

    combine_events = subparsers.add_parser("combine-normalized-events")
    combine_events.add_argument("--input", type=Path, action="append", required=True)
    combine_events.add_argument("--out", type=Path, required=True)

    gfz = subparsers.add_parser("fetch-gfz-kp-ap")
    gfz.add_argument("--out-root", type=Path, default=Path("data/raw/astronomy"))

    dst = subparsers.add_parser("fetch-kyoto-dst")
    dst.add_argument("--year-month", required=True, help="YYYYMM, e.g. 201601")
    dst.add_argument("--provisional", action="store_true")
    dst.add_argument("--out-root", type=Path, default=Path("data/raw/astronomy"))

    goes = subparsers.add_parser("fetch-ncei-goes-xrs")
    goes.add_argument("--year", type=int, required=True)
    goes.add_argument("--out-root", type=Path, default=Path("data/raw/astronomy"))

    f107 = subparsers.add_parser("fetch-f107-daily")
    f107.add_argument("--out-root", type=Path, default=Path("data/raw/astronomy"))

    multimodal = subparsers.add_parser("build-multimodal-smoke")
    multimodal.add_argument("--events", type=Path, required=True)
    multimodal.add_argument("--vlf-metadata", type=Path, action="append", default=[])
    multimodal.add_argument("--astronomy-metadata", type=Path, action="append", default=[])
    multimodal.add_argument("--region-id", required=True)
    multimodal.add_argument("--window-start", required=True)
    multimodal.add_argument("--window-end", required=True)
    multimodal.add_argument("--target-end", required=True)
    multimodal.add_argument("--target-magnitude-min", default="3.0")
    multimodal.add_argument("--out", type=Path, required=True)

    vlf_features = subparsers.add_parser("build-vlf-features")
    vlf_features.add_argument("--metadata", type=Path, action="append", default=[])
    vlf_features.add_argument("--window-start", required=True)
    vlf_features.add_argument("--window-end", required=True)
    vlf_features.add_argument("--out", type=Path, required=True)

    vlf_window_features = subparsers.add_parser("build-vlf-window-features")
    vlf_window_features.add_argument("--training-windows", type=Path, required=True)
    vlf_window_features.add_argument("--metadata-root", type=Path, required=True)
    vlf_window_features.add_argument("--out", type=Path, required=True)

    vlf_image_features = subparsers.add_parser("extract-vlf-image-features")
    vlf_image_features.add_argument("--image", type=Path, action="append", default=[])
    vlf_image_features.add_argument("--image-root", type=Path, action="append", default=[])
    vlf_image_features.add_argument("--filename-prefix", action="append", default=[])
    vlf_image_features.add_argument("--out", type=Path, required=True)
    vlf_image_features.add_argument("--crop-left", type=float, default=0.0)
    vlf_image_features.add_argument("--crop-top", type=float, default=0.13)
    vlf_image_features.add_argument("--crop-right", type=float, default=0.83)
    vlf_image_features.add_argument("--crop-bottom", type=float, default=0.95)

    vlf_image_join = subparsers.add_parser("join-vlf-image-features")
    vlf_image_join.add_argument("--windows", type=Path, required=True)
    vlf_image_join.add_argument("--image-features", type=Path, action="append", required=True)
    vlf_image_join.add_argument("--out", type=Path, required=True)
    vlf_image_join.add_argument("--exclude-window-end", action="store_true")

    astro_features = subparsers.add_parser("build-astronomy-features")
    astro_features.add_argument("--metadata", type=Path, action="append", default=[])
    astro_features.add_argument("--window-start", required=True)
    astro_features.add_argument("--window-end", required=True)
    astro_features.add_argument("--out", type=Path, required=True)

    kp_norm = subparsers.add_parser("normalize-gfz-kp-ap")
    kp_norm.add_argument("--raw", type=Path, required=True)
    kp_norm.add_argument("--out", type=Path, required=True)

    dst_norm = subparsers.add_parser("normalize-kyoto-dst")
    dst_norm.add_argument("--raw", type=Path, required=True)
    dst_norm.add_argument("--out", type=Path, required=True)

    f107_norm = subparsers.add_parser("normalize-f107-daily")
    f107_norm.add_argument("--raw", type=Path, required=True)
    f107_norm.add_argument("--out", type=Path, required=True)

    goes_norm = subparsers.add_parser("normalize-goes-xrs")
    goes_norm.add_argument("--raw", type=Path, required=True)
    goes_norm.add_argument("--out", type=Path, required=True)
    goes_norm.add_argument("--max-rows", type=int)
    goes_norm.add_argument("--start")
    goes_norm.add_argument("--end")

    label_targets = subparsers.add_parser("label-multimodal-targets")
    label_targets.add_argument("--input", type=Path, required=True)
    label_targets.add_argument("--events", type=Path, required=True)
    label_targets.add_argument("--as-of", required=True)
    label_targets.add_argument("--out", type=Path, required=True)

    table = subparsers.add_parser("build-multimodal-table")
    table.add_argument("--manifest", type=Path, required=True)
    table.add_argument("--out", type=Path, required=True)

    table_template = subparsers.add_parser("write-multimodal-manifest-template")
    table_template.add_argument("--out", type=Path, required=True)

    prospective = subparsers.add_parser("build-prospective-vlf-windows")
    prospective.add_argument("--events", type=Path, required=True)
    prospective.add_argument("--vlf-metadata-root", type=Path, required=True)
    prospective.add_argument("--astronomy-metadata-root", type=Path, required=True)
    prospective.add_argument("--region-id", required=True)
    prospective.add_argument("--lookback-hours", type=int, default=24)
    prospective.add_argument("--horizon-days", type=int, default=7)
    prospective.add_argument("--min-anchor-gap-seconds", type=int, default=60)
    prospective.add_argument("--target-magnitude-min", default="3.0")
    prospective.add_argument("--out", type=Path, required=True)

    prospective_update = subparsers.add_parser("update-prospective-vlf-table")
    prospective_update.add_argument("--table", type=Path, required=True)
    prospective_update.add_argument("--events", type=Path, required=True)
    prospective_update.add_argument("--vlf-metadata-root", type=Path, required=True)
    prospective_update.add_argument("--astronomy-metadata-root", type=Path, required=True)
    prospective_update.add_argument("--region-id", required=True)
    prospective_update.add_argument("--lookback-hours", type=int, default=24)
    prospective_update.add_argument("--horizon-days", type=int, default=7)
    prospective_update.add_argument("--min-anchor-gap-seconds", type=int, default=60)
    prospective_update.add_argument("--target-magnitude-min", default="3.0")
    prospective_update.add_argument("--out", type=Path, required=True)

    prospective_summary = subparsers.add_parser("summarize-prospective-table")
    prospective_summary.add_argument("--input", type=Path, required=True)
    prospective_summary.add_argument("--as-of")
    prospective_summary.add_argument("--out", type=Path, required=True)

    training = subparsers.add_parser("build-seismic-training-windows")
    training.add_argument("--events", type=Path, required=True)
    training.add_argument("--region-id", required=True)
    training.add_argument("--start", required=True)
    training.add_argument("--end", required=True)
    training.add_argument("--window-days", type=int, default=7)
    training.add_argument("--horizon-days", type=int, default=7)
    training.add_argument("--target-magnitude-min", default="3.0")
    training.add_argument("--out", type=Path, required=True)

    design = subparsers.add_parser("build-design-matrix")
    design.add_argument("--training-windows", type=Path, required=True)
    design.add_argument("--kp-ap", type=Path, required=True)
    design.add_argument("--f107", type=Path, required=True)
    design.add_argument("--out", type=Path, required=True)

    vlf_design = subparsers.add_parser("join-vlf-design-matrix")
    vlf_design.add_argument("--design-matrix", type=Path, required=True)
    vlf_design.add_argument("--vlf-windows", type=Path, required=True)
    vlf_design.add_argument("--out", type=Path, required=True)

    trainer = subparsers.add_parser("train-logistic-smoke")
    trainer.add_argument("--design-matrix", type=Path, required=True)
    trainer.add_argument("--out", type=Path, required=True)
    trainer.add_argument("--epochs", type=int, default=600)
    trainer.add_argument("--learning-rate", type=float, default=0.2)

    readiness = subparsers.add_parser("summarize-model-readiness")
    readiness.add_argument("--input", type=Path, required=True)
    readiness.add_argument("--out", type=Path, required=True)

    ablation = subparsers.add_parser("train-ablation-smoke")
    ablation.add_argument("--input", type=Path, required=True)
    ablation.add_argument("--out", type=Path, required=True)
    ablation.add_argument("--epochs", type=int, default=600)
    ablation.add_argument("--learning-rate", type=float, default=0.2)

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
    sandpile.add_argument("--mountain-mode", action="store_true")
    sandpile.add_argument("--summary-out", type=Path, required=True)
    sandpile.add_argument("--sensors-out", type=Path, required=True)
    sandpile.add_argument("--piezo-out", type=Path)
    sandpile.add_argument("--piezo-sensor-count", type=int, default=16)
    sandpile.add_argument("--piezo-susceptibility-base", type=float, default=0.15)
    sandpile.add_argument("--piezo-susceptibility-variation", type=float, default=0.85)
    sandpile.add_argument("--piezo-cluster-count", type=int, default=8)
    sandpile.add_argument("--piezo-cluster-radius", type=float, default=0.0)
    sandpile.add_argument("--piezo-activation-ratio", type=float, default=0.75)
    sandpile.add_argument("--piezo-attenuation-radius", type=float, default=0.0)
    sandpile.add_argument("--piezo-max-distance-radius", type=float, default=0.0)
    sandpile.add_argument("--piezo-charge-decay", type=float, default=0.995)
    sandpile.add_argument("--piezo-charge-coupling", type=float, default=1.0)
    sandpile.add_argument("--piezo-release-ratio", type=float, default=0.15)
    sandpile.add_argument("--piezo-critical-release-ratio", type=float, default=0.05)
    sandpile.add_argument("--piezo-saturation", type=float, default=1000.0)
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

    sandpile_summary = subparsers.add_parser("summarize-sandpile-sim")
    sandpile_summary.add_argument("--summary", type=Path, required=True)
    sandpile_summary.add_argument("--sensors", type=Path, required=True)
    sandpile_summary.add_argument("--out", type=Path, required=True)

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

    sandpile_heatmap = subparsers.add_parser("render-sandpile-heatmap")
    sandpile_heatmap.add_argument("--snapshot", type=Path, required=True)
    sandpile_heatmap.add_argument("--out", type=Path, required=True)
    sandpile_heatmap.add_argument("--scale", type=int, default=8)
    sandpile_heatmap.add_argument("--color-min", type=float, default=0.0)
    sandpile_heatmap.add_argument("--color-max", type=float)
    sandpile_heatmap.add_argument("--gamma", type=float, default=1.0)

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

    piezo_spectrogram = subparsers.add_parser("render-piezo-spectrogram")
    piezo_spectrogram.add_argument("--piezo", type=Path, required=True)
    piezo_spectrogram.add_argument("--out", type=Path, required=True)
    piezo_spectrogram.add_argument("--metadata-out", type=Path)
    piezo_spectrogram.add_argument("--start-time-utc", default="2026-01-01T00:00:00Z")
    piezo_spectrogram.add_argument("--step-seconds", type=int, default=60)
    piezo_spectrogram.add_argument("--freq-min", type=float, default=0.0)
    piezo_spectrogram.add_argument("--freq-max", type=float)
    piezo_spectrogram.add_argument("--freq-bins", type=int, default=96)
    piezo_spectrogram.add_argument("--window-steps", type=int, default=64)
    piezo_spectrogram.add_argument("--scale", type=int, default=4)
    piezo_spectrogram.add_argument("--gamma", type=float, default=0.85)
    piezo_spectrogram.add_argument("--sensor-id", type=int)
    piezo_spectrogram.add_argument("--dc-block", type=float, default=0.0)

    piezo_summary = subparsers.add_parser("render-piezo-summary")
    piezo_summary.add_argument("--piezo", type=Path, required=True)
    piezo_summary.add_argument("--out", type=Path, required=True)
    piezo_summary.add_argument("--metadata-out", type=Path)
    piezo_summary.add_argument("--start-time-utc", default="2026-01-01T00:00:00Z")
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

    piezo_audio = subparsers.add_parser("render-piezo-audio")
    piezo_audio.add_argument("--piezo", type=Path, required=True)
    piezo_audio.add_argument("--out", type=Path, required=True)
    piezo_audio.add_argument("--sample-rate", type=int, default=44100)
    piezo_audio.add_argument("--duration-seconds", type=float, default=20.0)
    piezo_audio.add_argument("--gain", type=float, default=0.95)
    piezo_audio.add_argument("--smooth-steps", type=int, default=64)
    piezo_audio.add_argument("--sensor-id", type=int)
    piezo_audio.add_argument("--dc-block", type=float, default=0.0)

    args = parser.parse_args()
    try:
        if args.command == "fetch-ingv-events":
            stored = [
                fetch_italy_events(
                    args.start,
                    args.end,
                    out_root=args.out_root,
                    min_magnitude=args.min_mag,
                    limit=args.limit,
                )
            ]
        elif args.command == "plan-ingv-backfill":
            rows = plan_ingv_backfill(
                start_utc=args.start,
                end_utc=args.end,
                chunk_days=args.chunk_days,
                min_magnitude=args.min_mag,
                out_path=args.out,
            )
            print(f"planned windows: {len(rows)}")
            print(f"output: {args.out}")
            return 0
        elif args.command == "fetch-vlf-cumiana":
            stored = fetch_manifest_images(
                args.manifest,
                out_root=args.out_root,
                only=set(args.only) if args.only else None,
            )
        elif args.command == "capture-vlf-cumiana-loop":
            stored = repeat_manifest_images(
                args.manifest,
                out_root=args.out_root,
                cycles=args.cycles,
                interval_seconds=args.interval_seconds,
                only=set(args.only) if args.only else None,
            )
        elif args.command == "fetch-astronomy":
            stored = fetch_manifest_json(
                args.manifest,
                out_root=args.out_root,
                date=args.date,
                moon_phase_count=args.moon_phases,
                only=set(args.only) if args.only else None,
            )
        elif args.command == "normalize-ingv-events":
            count = normalize_ingv_event_text(
                args.raw,
                args.out,
                raw_uri=args.raw_uri,
                ingested_at_utc=args.ingested_at_utc,
                only_region=args.only_region,
            )
            print(f"normalized rows: {count}")
            print(f"output: {args.out}")
            return 0
        elif args.command == "combine-normalized-events":
            rows = combine_normalized_events(input_paths=args.input, out_path=args.out)
            print(f"combined rows: {len(rows)}")
            print(f"output: {args.out}")
            return 0
        elif args.command == "fetch-gfz-kp-ap":
            stored = [fetch_gfz_kp_ap(out_root=args.out_root)]
        elif args.command == "fetch-kyoto-dst":
            stored = [
                fetch_kyoto_dst_month(
                    args.year_month,
                    out_root=args.out_root,
                    provisional=args.provisional,
                )
            ]
        elif args.command == "fetch-ncei-goes-xrs":
            stored = [fetch_ncei_goes15_xrs_year(args.year, out_root=args.out_root)]
        elif args.command == "fetch-f107-daily":
            stored = [fetch_spaceweather_canada_f107_daily(out_root=args.out_root)]
        elif args.command == "build-multimodal-smoke":
            row = build_multimodal_smoke_row(
                events_csv=args.events,
                vlf_metadata_paths=args.vlf_metadata,
                astronomy_metadata_paths=args.astronomy_metadata,
                region_id=args.region_id,
                window_start_utc=args.window_start,
                window_end_utc=args.window_end,
                target_end_utc=args.target_end,
                target_magnitude_min=args.target_magnitude_min,
                out_path=args.out,
            )
            print(f"wrote: {args.out}")
            print(f"window_id: {row['window_id']}")
            return 0
        elif args.command == "build-vlf-features":
            row = build_vlf_features(
                metadata_paths=args.metadata,
                window_start_utc=args.window_start,
                window_end_utc=args.window_end,
                out_path=args.out,
            )
            print(f"wrote: {args.out}")
            print(f"vlf_capture_count: {row['vlf_capture_count']}")
            return 0
        elif args.command == "build-vlf-window-features":
            rows = build_vlf_window_features(
                training_windows_csv=args.training_windows,
                metadata_root=args.metadata_root,
                out_path=args.out,
            )
            print(f"vlf window rows: {len(rows)}")
            print(f"output: {args.out}")
            return 0
        elif args.command == "extract-vlf-image-features":
            image_paths = _resolve_image_paths(
                image_paths=args.image,
                image_roots=args.image_root,
                filename_prefixes=args.filename_prefix,
            )
            rows = build_vlf_image_features(
                image_paths=image_paths,
                out_path=args.out,
                crop_left=args.crop_left,
                crop_top=args.crop_top,
                crop_right=args.crop_right,
                crop_bottom=args.crop_bottom,
            )
            print(f"image rows: {len(rows)}")
            print(f"output: {args.out}")
            return 0
        elif args.command == "join-vlf-image-features":
            rows = join_vlf_image_features_to_windows(
                windows_csv=args.windows,
                image_features_csvs=args.image_features,
                out_path=args.out,
                include_window_end=not args.exclude_window_end,
            )
            print(f"joined rows: {len(rows)}")
            print(f"output: {args.out}")
            return 0
        elif args.command == "build-astronomy-features":
            row = build_astronomy_features(
                metadata_paths=args.metadata,
                window_start_utc=args.window_start,
                window_end_utc=args.window_end,
                out_path=args.out,
            )
            print(f"wrote: {args.out}")
            print(f"astro_capture_count: {row['astro_capture_count']}")
            return 0
        elif args.command == "normalize-gfz-kp-ap":
            count = normalize_gfz_kp_ap(args.raw, args.out)
            print(f"normalized rows: {count}")
            print(f"output: {args.out}")
            return 0
        elif args.command == "normalize-kyoto-dst":
            count = normalize_kyoto_dst_text(args.raw, args.out)
            print(f"normalized rows: {count}")
            print(f"output: {args.out}")
            return 0
        elif args.command == "normalize-f107-daily":
            count = normalize_f107_daily(args.raw, args.out)
            print(f"normalized rows: {count}")
            print(f"output: {args.out}")
            return 0
        elif args.command == "normalize-goes-xrs":
            count = normalize_goes_xrs_netcdf(
                args.raw,
                args.out,
                max_rows=args.max_rows,
                start_utc=args.start,
                end_utc=args.end,
            )
            print(f"normalized rows: {count}")
            print(f"output: {args.out}")
            return 0
        elif args.command == "label-multimodal-targets":
            rows = label_multimodal_targets(
                input_csv=args.input,
                events_csv=args.events,
                as_of_utc=args.as_of,
                out_path=args.out,
            )
            print(f"labeled rows: {len(rows)}")
            print(f"output: {args.out}")
            return 0
        elif args.command == "build-multimodal-table":
            rows = build_multimodal_table_from_manifest(
                manifest_path=args.manifest,
                out_path=args.out,
            )
            print(f"built rows: {len(rows)}")
            print(f"output: {args.out}")
            return 0
        elif args.command == "write-multimodal-manifest-template":
            write_multimodal_manifest_template(args.out)
            print(f"output: {args.out}")
            return 0
        elif args.command == "build-prospective-vlf-windows":
            rows = build_prospective_vlf_windows(
                events_csv=args.events,
                vlf_metadata_root=args.vlf_metadata_root,
                astronomy_metadata_root=args.astronomy_metadata_root,
                region_id=args.region_id,
                lookback_hours=args.lookback_hours,
                horizon_days=args.horizon_days,
                min_anchor_gap_seconds=args.min_anchor_gap_seconds,
                target_magnitude_min=args.target_magnitude_min,
                out_path=args.out,
            )
            print(f"prospective rows: {len(rows)}")
            print(f"output: {args.out}")
            return 0
        elif args.command == "update-prospective-vlf-table":
            report = update_prospective_vlf_table(
                table_path=args.table,
                events_csv=args.events,
                vlf_metadata_root=args.vlf_metadata_root,
                astronomy_metadata_root=args.astronomy_metadata_root,
                region_id=args.region_id,
                lookback_hours=args.lookback_hours,
                horizon_days=args.horizon_days,
                min_anchor_gap_seconds=args.min_anchor_gap_seconds,
                target_magnitude_min=args.target_magnitude_min,
                out_path=args.out,
            )
            print(f"existing rows: {report['existing_rows']}")
            print(f"candidate rows: {report['candidate_rows']}")
            print(f"new rows: {report['new_rows']}")
            print(f"total rows: {report['total_rows']}")
            print(f"output: {args.out}")
            return 0
        elif args.command == "summarize-prospective-table":
            report = summarize_prospective_table(
                input_csv=args.input,
                as_of_utc=args.as_of,
                out_path=args.out,
            )
            print(f"rows: {report['row_count']}")
            print(f"ready to label: {report['ready_to_label_count']}")
            print(f"missing vlf image features: {report['missing_vlf_image_features_count']}")
            print(f"output: {args.out}")
            return 0
        elif args.command == "build-seismic-training-windows":
            rows = build_seismic_training_windows(
                events_csv=args.events,
                region_id=args.region_id,
                start_utc=args.start,
                end_utc=args.end,
                window_days=args.window_days,
                horizon_days=args.horizon_days,
                target_magnitude_min=args.target_magnitude_min,
                out_path=args.out,
            )
            print(f"training rows: {len(rows)}")
            print(f"output: {args.out}")
            return 0
        elif args.command == "build-design-matrix":
            rows = build_design_matrix(
                training_windows_csv=args.training_windows,
                kp_ap_csv=args.kp_ap,
                f107_csv=args.f107,
                out_path=args.out,
            )
            print(f"design rows: {len(rows)}")
            print(f"output: {args.out}")
            return 0
        elif args.command == "join-vlf-design-matrix":
            rows = join_vlf_design_matrix(
                design_matrix_csv=args.design_matrix,
                vlf_windows_csv=args.vlf_windows,
                out_path=args.out,
            )
            print(f"design rows: {len(rows)}")
            print(f"output: {args.out}")
            return 0
        elif args.command == "train-logistic-smoke":
            report = train_logistic_smoke(
                design_matrix_csv=args.design_matrix,
                out_path=args.out,
                epochs=args.epochs,
                learning_rate=args.learning_rate,
            )
            print(f"status: {report['status']}")
            print(f"output: {args.out}")
            return 0
        elif args.command == "summarize-model-readiness":
            report = summarize_model_readiness(input_csv=args.input, out_path=args.out)
            print(f"status: {report['status']}")
            print(f"rows: {report['row_count']}")
            print(f"labeled rows: {report['labeled_row_count']}")
            print(f"output: {args.out}")
            return 0
        elif args.command == "train-ablation-smoke":
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
        elif args.command == "run-sandpile-sim":
            from elfquake.sim.piezo import PiezoConfig
            from elfquake.sim.sandpile import SandpileConfig, run_sandpile_simulation

            started = time.perf_counter()
            threshold = args.threshold
            deposition_mode = args.deposition_mode
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
                    threshold=threshold,
                    source_count=args.source_count,
                    sensor_count=args.sensor_count,
                    deposition_probability=args.deposition_probability,
                    seed=args.seed,
                    max_relaxation_sweeps=args.max_relaxation_sweeps,
                    deposition_mode=deposition_mode,
                    target_mean_height=target_mean_height,
                    target_fill_limit=args.target_fill_limit,
                    bottom_layer_removal_interval=bottom_layer_removal_interval,
                ),
                summary_out=args.summary_out,
                sensors_out=args.sensors_out,
                piezo_out=args.piezo_out,
                piezo_config=PiezoConfig(
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
                    release_ratio=args.piezo_release_ratio,
                    critical_release_ratio=args.piezo_critical_release_ratio,
                    saturation=args.piezo_saturation,
                )
                if args.piezo_out
                else None,
                snapshot_dir=args.snapshot_dir,
                snapshot_interval=args.snapshot_interval,
                progress_interval=args.progress_interval,
                progress_callback=report_progress if args.progress_interval else None,
            )
            heatmap_rows = []
            if args.heatmap_dir:
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
                heatmap_rows = render_sandpile_heatmaps_from_manifest(
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
            print(f"summary rows: {len(summary_rows)}")
            print(f"sensor rows: {len(sensor_rows)}")
            print(f"summary output: {args.summary_out}")
            print(f"sensors output: {args.sensors_out}")
            if args.piezo_out:
                print(f"piezo output: {args.piezo_out}")
            if args.snapshot_dir:
                print(f"snapshot dir: {args.snapshot_dir}")
            if args.heatmap_dir:
                print(f"heatmap rows: {len(heatmap_rows)}")
                print(f"heatmap dir: {args.heatmap_dir}")
            return 0
        elif args.command == "summarize-sandpile-sim":
            from elfquake.sim.report import summarize_sandpile_outputs

            report = summarize_sandpile_outputs(
                summary_csv=args.summary,
                sensors_csv=args.sensors,
                out_path=args.out,
            )
            print(f"status: {report['status']}")
            print(f"summary rows: {report['summary_row_count']}")
            print(f"sensor rows: {report['sensor_row_count']}")
            print(f"avalanche steps: {report['avalanche_step_count']}")
            print(f"output: {args.out}")
            return 0
        elif args.command == "benchmark-sandpile-sim":
            from elfquake.sim.report import benchmark_sandpile_simulation
            from elfquake.sim.sandpile import SandpileConfig

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
        elif args.command == "render-sandpile-heatmap":
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
        elif args.command == "build-synthetic-event-list":
            from elfquake.sim.synthetic_events import build_synthetic_event_list

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
        elif args.command == "render-piezo-spectrogram":
            from elfquake.sim.piezo_spectrogram import render_piezo_spectrogram

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
            print(f"steps: {report['step_count']}")
            print(f"sensors: {report['sensor_count']}")
            print(f"selected sensor: {report['selected_sensor_id'] or 'sum'}")
            print(f"dc block: {report['dc_block']}")
            print(f"frequency range: {report['freq_min_hz']}..{report['freq_max_hz']} Hz")
            print(f"nyquist: {report['nyquist_hz']} Hz")
            if args.metadata_out:
                print(f"metadata: {args.metadata_out}")
            return 0
        elif args.command == "render-piezo-summary":
            from elfquake.sim.piezo_spectrogram import render_piezo_timeseries_spectrogram

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
            print(f"steps: {report['step_count']}")
            print(f"sensors: {report['sensor_count']}")
            print(f"selected sensor: {report['selected_sensor_id'] or 'sum'}")
            print(f"dc block: {report['dc_block']}")
            print(f"frequency range: {report['freq_min_hz']}..{report['freq_max_hz']} Hz")
            print(f"nyquist: {report['nyquist_hz']} Hz")
            if args.metadata_out:
                print(f"metadata: {args.metadata_out}")
            return 0
        elif args.command == "render-piezo-audio":
            from elfquake.sim.piezo_spectrogram import render_piezo_audio

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
        else:
            parser.error(f"unknown command: {args.command}")
    except HTTPError as error:
        print(f"fetch failed: HTTP {error.code} for {error.url}", file=sys.stderr)
        return 2
    except URLError as error:
        print(f"fetch failed: {error.reason}", file=sys.stderr)
        return 2
    except ValueError as error:
        print(f"invalid arguments: {error}", file=sys.stderr)
        return 2
    except OSError as error:
        print(f"fetch failed: {error}", file=sys.stderr)
        return 2

    for capture in stored:
        status = "skipped" if capture.skipped_existing else "stored"
        print(f"{status}: {capture.payload_path}")
        print(f"metadata: {capture.metadata_path}")
    return 0

def _resolve_image_paths(
    *,
    image_paths: list[Path],
    image_roots: list[Path],
    filename_prefixes: list[str],
) -> list[Path]:
    resolved = list(image_paths)
    for root in image_roots:
        resolved.extend(sorted(root.glob("**/*.jpg")))
        resolved.extend(sorted(root.glob("**/*.jpeg")))
    if filename_prefixes:
        resolved = [
            path for path in resolved if any(path.name.startswith(prefix) for prefix in filename_prefixes)
        ]
    unique = sorted(dict.fromkeys(resolved))
    if not unique:
        raise ValueError("at least one --image or matching --image-root file is required")
    return unique


if __name__ == "__main__":
    raise SystemExit(main())

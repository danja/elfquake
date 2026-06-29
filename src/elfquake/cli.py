"""Command line entry points for raw data acquisition."""

from __future__ import annotations

import argparse
import sys
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
from elfquake.features.multimodal_smoke import build_multimodal_smoke_row
from elfquake.features.table import build_multimodal_table_from_manifest, write_multimodal_manifest_template
from elfquake.features.targets import label_multimodal_targets
from elfquake.features.vlf import build_vlf_features
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
            count = normalize_goes_xrs_netcdf(args.raw, args.out)
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
        else:
            parser.error(f"unknown command: {args.command}")
    except HTTPError as error:
        print(f"fetch failed: HTTP {error.code} for {error.url}", file=sys.stderr)
        return 2
    except URLError as error:
        print(f"fetch failed: {error.reason}", file=sys.stderr)
        return 2
    except OSError as error:
        print(f"fetch failed: {error}", file=sys.stderr)
        return 2

    for capture in stored:
        status = "skipped" if capture.skipped_existing else "stored"
        print(f"{status}: {capture.payload_path}")
        print(f"metadata: {capture.metadata_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Source acquisition and normalization CLI commands."""

from __future__ import annotations

from argparse import Namespace, _SubParsersAction
from pathlib import Path

from elfquake.backfill import plan_ingv_backfill
from elfquake.cli_commands.common import print_stored_captures
from elfquake.connectors.astronomy import fetch_manifest_json
from elfquake.connectors.ingv import fetch_italy_events
from elfquake.connectors.japan import fetch_japan_events
from elfquake.connectors.vlf_manifest import fetch_manifest_captures, repeat_manifest_captures
from elfquake.normalize.vlf_cdf import normalize_vlf_cdf
from elfquake.connectors.space_archives import (
    fetch_gfz_kp_ap,
    fetch_kyoto_dst_month,
    fetch_ncei_goes15_xrs_year,
    fetch_spaceweather_canada_f107_daily,
)
from elfquake.connectors.vlf_cumiana import fetch_manifest_images, repeat_manifest_images
from elfquake.connectors.vlf_abelian import fetch_cumiana_archive, probe_cumiana_archive, record_cumiana_stream
from elfquake.normalize.events import combine_normalized_events
from elfquake.normalize.ingv import normalize_ingv_event_text
from elfquake.normalize.japan import normalize_japan_event_json
from elfquake.normalize.space_weather import (
    normalize_f107_daily,
    normalize_gfz_kp_ap,
    normalize_goes_xrs_netcdf,
    normalize_kyoto_dst_text,
)


def register_source_commands(subparsers: _SubParsersAction) -> None:
    ingv = subparsers.add_parser("fetch-ingv-events")
    ingv.add_argument("--start", required=True, help="UTC start, e.g. 2026-06-22T00:00:00Z")
    ingv.add_argument("--end", required=True, help="UTC end, e.g. 2026-06-29T23:59:59Z")
    ingv.add_argument("--out-root", type=Path, default=Path("data/raw/ingv"))
    ingv.add_argument("--min-mag", type=float, default=2.0)
    ingv.add_argument("--limit", type=int, default=10000)
    ingv.set_defaults(func=_fetch_ingv_events)

    japan = subparsers.add_parser("fetch-japan-events")
    japan.add_argument("--start", required=True)
    japan.add_argument("--end", required=True)
    japan.add_argument("--out-root", type=Path, default=Path("data/raw/japan/events"))
    japan.add_argument("--min-mag", type=float, default=2.0)
    japan.add_argument("--limit", type=int, default=20000)
    japan.set_defaults(func=_fetch_japan_events)

    japan_norm = subparsers.add_parser("normalize-japan-events")
    japan_norm.add_argument("--raw", type=Path, required=True)
    japan_norm.add_argument("--out", type=Path, required=True)
    japan_norm.set_defaults(func=_normalize_japan_events)

    japan_vlf = subparsers.add_parser("fetch-vlf-japan")
    japan_vlf.add_argument("--manifest", type=Path, default=Path("data/raw/vlf/japan/manifest.csv"))
    japan_vlf.add_argument("--out-root", type=Path, default=Path("data/raw/vlf/japan"))
    japan_vlf.add_argument("--only", action="append", default=[])
    japan_vlf.set_defaults(func=_fetch_vlf_japan)

    japan_cdf = subparsers.add_parser("normalize-vlf-japan-cdf")
    japan_cdf.add_argument("--input", type=Path, required=True)
    japan_cdf.add_argument("--samples-out", type=Path, required=True)
    japan_cdf.add_argument("--metadata-out", type=Path, required=True)
    japan_cdf.set_defaults(func=_normalize_vlf_japan_cdf)

    japan_vlf_loop = subparsers.add_parser("capture-vlf-japan-loop")
    japan_vlf_loop.add_argument("--manifest", type=Path, default=Path("data/raw/vlf/japan/manifest.csv"))
    japan_vlf_loop.add_argument("--out-root", type=Path, default=Path("data/raw/vlf/japan"))
    japan_vlf_loop.add_argument("--only", action="append", default=[])
    japan_vlf_loop.add_argument("--cycles", type=int, default=2)
    japan_vlf_loop.add_argument("--interval-seconds", type=int, default=1800)
    japan_vlf_loop.set_defaults(func=_capture_vlf_japan_loop)

    ingv_plan = subparsers.add_parser("plan-ingv-backfill")
    ingv_plan.add_argument("--start", required=True)
    ingv_plan.add_argument("--end", required=True)
    ingv_plan.add_argument("--chunk-days", type=int, default=14)
    ingv_plan.add_argument("--min-mag", default="2.0")
    ingv_plan.add_argument("--out", type=Path, required=True)
    ingv_plan.set_defaults(func=_plan_ingv_backfill)

    vlf = subparsers.add_parser("fetch-vlf-cumiana")
    vlf.add_argument("--manifest", type=Path, default=Path("data/raw/vlf/cumiana/manifest.csv"))
    vlf.add_argument("--out-root", type=Path, default=Path("data/raw/vlf/cumiana"))
    vlf.add_argument("--only", action="append", default=[], help="Endpoint id to fetch; repeatable")
    vlf.set_defaults(func=_fetch_vlf_cumiana)

    vlf_loop = subparsers.add_parser("capture-vlf-cumiana-loop")
    vlf_loop.add_argument("--manifest", type=Path, default=Path("data/raw/vlf/cumiana/manifest.csv"))
    vlf_loop.add_argument("--out-root", type=Path, default=Path("data/raw/vlf/cumiana"))
    vlf_loop.add_argument("--only", action="append", default=[], help="Endpoint id to fetch; repeatable")
    vlf_loop.add_argument("--cycles", type=int, default=2, help="0 means run forever")
    vlf_loop.add_argument("--interval-seconds", type=int, default=1800)
    vlf_loop.set_defaults(func=_capture_vlf_cumiana_loop)

    abelian = subparsers.add_parser("record-vlf-abelian-cumiana")
    abelian.add_argument("--out-root", type=Path, default=Path("data/raw/vlf/abelian/cumiana"))
    abelian.add_argument("--duration-seconds", type=int, default=60)
    abelian.add_argument("--max-bytes", type=int)
    abelian.set_defaults(func=_record_vlf_abelian_cumiana)

    abelian_archive = subparsers.add_parser("fetch-vlf-abelian-cumiana-archive")
    abelian_archive.add_argument("--out-root", type=Path, default=Path("data/raw/vlf/abelian/cumiana"))
    abelian_archive.add_argument("--start", required=True, help="UTC start, e.g. 2026-07-05T10:38:11Z")
    abelian_archive.add_argument("--duration-seconds", type=float, default=5.0)
    abelian_archive.add_argument("--format", choices=["sg", "td", "vt", "wav"], default="wav")
    abelian_archive.set_defaults(func=_fetch_vlf_abelian_cumiana_archive)

    abelian_archive_probe = subparsers.add_parser("probe-vlf-abelian-cumiana-archive")
    abelian_archive_probe.add_argument("--start", action="append", required=True)
    abelian_archive_probe.add_argument("--duration-seconds", type=float, default=5.0)
    abelian_archive_probe.add_argument("--format", action="append", choices=["sg", "td", "vt", "wav"], default=[])
    abelian_archive_probe.add_argument("--fetch-downloads", action="store_true")
    abelian_archive_probe.add_argument("--out", type=Path, required=True)
    abelian_archive_probe.set_defaults(func=_probe_vlf_abelian_cumiana_archive)

    astro = subparsers.add_parser("fetch-astronomy")
    astro.add_argument("--manifest", type=Path, default=Path("data/raw/astronomy/manifest.csv"))
    astro.add_argument("--out-root", type=Path, default=Path("data/raw/astronomy"))
    astro.add_argument("--date", required=True, help="Date for parameterized endpoints, YYYY-MM-DD")
    astro.add_argument("--moon-phases", type=int, default=8)
    astro.add_argument("--only", action="append", default=[], help="Source id to fetch; repeatable")
    astro.set_defaults(func=_fetch_astronomy)

    normalize = subparsers.add_parser("normalize-ingv-events")
    normalize.add_argument("--raw", type=Path, required=True)
    normalize.add_argument("--out", type=Path, required=True)
    normalize.add_argument("--raw-uri")
    normalize.add_argument("--ingested-at-utc")
    normalize.add_argument("--only-region")
    normalize.set_defaults(func=_normalize_ingv_events)

    combine_events = subparsers.add_parser("combine-normalized-events")
    combine_events.add_argument("--input", type=Path, action="append", required=True)
    combine_events.add_argument("--out", type=Path, required=True)
    combine_events.set_defaults(func=_combine_normalized_events)

    gfz = subparsers.add_parser("fetch-gfz-kp-ap")
    gfz.add_argument("--out-root", type=Path, default=Path("data/raw/astronomy"))
    gfz.set_defaults(func=_fetch_gfz_kp_ap)

    dst = subparsers.add_parser("fetch-kyoto-dst")
    dst.add_argument("--year-month", required=True, help="YYYYMM, e.g. 201601")
    dst.add_argument("--provisional", action="store_true")
    dst.add_argument("--out-root", type=Path, default=Path("data/raw/astronomy"))
    dst.set_defaults(func=_fetch_kyoto_dst)

    goes = subparsers.add_parser("fetch-ncei-goes-xrs")
    goes.add_argument("--year", type=int, required=True)
    goes.add_argument("--out-root", type=Path, default=Path("data/raw/astronomy"))
    goes.set_defaults(func=_fetch_ncei_goes_xrs)

    f107 = subparsers.add_parser("fetch-f107-daily")
    f107.add_argument("--out-root", type=Path, default=Path("data/raw/astronomy"))
    f107.set_defaults(func=_fetch_f107_daily)

    kp_norm = subparsers.add_parser("normalize-gfz-kp-ap")
    kp_norm.add_argument("--raw", type=Path, required=True)
    kp_norm.add_argument("--out", type=Path, required=True)
    kp_norm.set_defaults(func=_normalize_gfz_kp_ap)

    dst_norm = subparsers.add_parser("normalize-kyoto-dst")
    dst_norm.add_argument("--raw", type=Path, required=True)
    dst_norm.add_argument("--out", type=Path, required=True)
    dst_norm.set_defaults(func=_normalize_kyoto_dst)

    f107_norm = subparsers.add_parser("normalize-f107-daily")
    f107_norm.add_argument("--raw", type=Path, required=True)
    f107_norm.add_argument("--out", type=Path, required=True)
    f107_norm.set_defaults(func=_normalize_f107_daily)

    goes_norm = subparsers.add_parser("normalize-goes-xrs")
    goes_norm.add_argument("--raw", type=Path, required=True)
    goes_norm.add_argument("--out", type=Path, required=True)
    goes_norm.add_argument("--max-rows", type=int)
    goes_norm.add_argument("--start")
    goes_norm.add_argument("--end")
    goes_norm.set_defaults(func=_normalize_goes_xrs)


def _fetch_ingv_events(args: Namespace) -> int:
    return print_stored_captures([
        fetch_italy_events(
            args.start,
            args.end,
            out_root=args.out_root,
            min_magnitude=args.min_mag,
            limit=args.limit,
        )
    ])


def _fetch_japan_events(args: Namespace) -> int:
    return print_stored_captures([fetch_japan_events(
        args.start, args.end, out_root=args.out_root, min_magnitude=args.min_mag, limit=args.limit,
    )])


def _normalize_japan_events(args: Namespace) -> int:
    count = normalize_japan_event_json(args.raw, args.out)
    print(f"normalized rows: {count}")
    print(f"output: {args.out}")
    return 0


def _fetch_vlf_japan(args: Namespace) -> int:
    return print_stored_captures(fetch_manifest_captures(
        args.manifest, out_root=args.out_root, source_namespace="japan",
        only=set(args.only) if args.only else None,
    ))


def _capture_vlf_japan_loop(args: Namespace) -> int:
    return print_stored_captures(repeat_manifest_captures(
        args.manifest, out_root=args.out_root, source_namespace="japan", cycles=args.cycles,
        interval_seconds=args.interval_seconds, only=set(args.only) if args.only else None,
    ))


def _plan_ingv_backfill(args: Namespace) -> int:
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


def _normalize_vlf_japan_cdf(args: Namespace) -> int:
    report = normalize_vlf_cdf(input_path=args.input, samples_out=args.samples_out, metadata_out=args.metadata_out)
    print(f"rows: {report['row_count']}")
    print(f"variables: {len(report['variables'])}")
    print(f"samples: {args.samples_out}")
    print(f"metadata: {args.metadata_out}")
    return 0


def _fetch_vlf_cumiana(args: Namespace) -> int:
    return print_stored_captures(fetch_manifest_images(
        args.manifest,
        out_root=args.out_root,
        only=set(args.only) if args.only else None,
    ))


def _capture_vlf_cumiana_loop(args: Namespace) -> int:
    return print_stored_captures(repeat_manifest_images(
        args.manifest,
        out_root=args.out_root,
        cycles=args.cycles,
        interval_seconds=args.interval_seconds,
        only=set(args.only) if args.only else None,
    ))


def _record_vlf_abelian_cumiana(args: Namespace) -> int:
    return print_stored_captures([
        record_cumiana_stream(
            out_root=args.out_root,
            duration_seconds=args.duration_seconds,
            max_bytes=args.max_bytes,
        )
    ])


def _fetch_vlf_abelian_cumiana_archive(args: Namespace) -> int:
    return print_stored_captures(
        fetch_cumiana_archive(
            out_root=args.out_root,
            start_time_utc=args.start,
            duration_seconds=args.duration_seconds,
            output_format=args.format,
        )
    )


def _probe_vlf_abelian_cumiana_archive(args: Namespace) -> int:
    rows = probe_cumiana_archive(
        start_times_utc=args.start,
        duration_seconds=args.duration_seconds,
        output_formats=args.format or ["wav"],
        fetch_downloads=args.fetch_downloads,
        out_path=args.out,
    )
    usable = sum(1 for row in rows if row["usable_nonempty"] == "1")
    print(f"probe rows: {len(rows)}")
    print(f"usable nonempty rows: {usable}")
    print(f"output: {args.out}")
    return 0


def _fetch_astronomy(args: Namespace) -> int:
    from elfquake import cli as cli_module

    fetcher = getattr(cli_module, "fetch_manifest_json", fetch_manifest_json)
    return print_stored_captures(fetcher(
        args.manifest,
        out_root=args.out_root,
        date=args.date,
        moon_phase_count=args.moon_phases,
        only=set(args.only) if args.only else None,
    ))


def _normalize_ingv_events(args: Namespace) -> int:
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


def _combine_normalized_events(args: Namespace) -> int:
    rows = combine_normalized_events(input_paths=args.input, out_path=args.out)
    print(f"combined rows: {len(rows)}")
    print(f"output: {args.out}")
    return 0


def _fetch_gfz_kp_ap(args: Namespace) -> int:
    return print_stored_captures([fetch_gfz_kp_ap(out_root=args.out_root)])


def _fetch_kyoto_dst(args: Namespace) -> int:
    return print_stored_captures([
        fetch_kyoto_dst_month(
            args.year_month,
            out_root=args.out_root,
            provisional=args.provisional,
        )
    ])


def _fetch_ncei_goes_xrs(args: Namespace) -> int:
    return print_stored_captures([fetch_ncei_goes15_xrs_year(args.year, out_root=args.out_root)])


def _fetch_f107_daily(args: Namespace) -> int:
    return print_stored_captures([fetch_spaceweather_canada_f107_daily(out_root=args.out_root)])


def _normalize_gfz_kp_ap(args: Namespace) -> int:
    count = normalize_gfz_kp_ap(args.raw, args.out)
    print(f"normalized rows: {count}")
    print(f"output: {args.out}")
    return 0


def _normalize_kyoto_dst(args: Namespace) -> int:
    count = normalize_kyoto_dst_text(args.raw, args.out)
    print(f"normalized rows: {count}")
    print(f"output: {args.out}")
    return 0


def _normalize_f107_daily(args: Namespace) -> int:
    count = normalize_f107_daily(args.raw, args.out)
    print(f"normalized rows: {count}")
    print(f"output: {args.out}")
    return 0


def _normalize_goes_xrs(args: Namespace) -> int:
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

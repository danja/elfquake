"""Command line entry points for raw data acquisition."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from urllib.error import HTTPError, URLError

from elfquake.connectors.astronomy import fetch_manifest_json
from elfquake.connectors.ingv import fetch_italy_events
from elfquake.connectors.space_archives import (
    fetch_gfz_kp_ap,
    fetch_kyoto_dst_month,
    fetch_ncei_goes15_xrs_year,
    fetch_spaceweather_canada_f107_daily,
)
from elfquake.connectors.vlf_cumiana import fetch_manifest_images
from elfquake.normalize.ingv import normalize_ingv_event_text


def main() -> int:
    parser = argparse.ArgumentParser(prog="elfquake")
    subparsers = parser.add_subparsers(dest="command", required=True)

    ingv = subparsers.add_parser("fetch-ingv-events")
    ingv.add_argument("--start", required=True, help="UTC start, e.g. 2026-06-22T00:00:00Z")
    ingv.add_argument("--end", required=True, help="UTC end, e.g. 2026-06-29T23:59:59Z")
    ingv.add_argument("--out-root", type=Path, default=Path("data/raw/ingv"))
    ingv.add_argument("--min-mag", type=float, default=2.0)
    ingv.add_argument("--limit", type=int, default=10000)

    vlf = subparsers.add_parser("fetch-vlf-cumiana")
    vlf.add_argument("--manifest", type=Path, default=Path("data/raw/vlf/cumiana/manifest.csv"))
    vlf.add_argument("--out-root", type=Path, default=Path("data/raw/vlf/cumiana"))
    vlf.add_argument("--only", action="append", default=[], help="Endpoint id to fetch; repeatable")

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
        elif args.command == "fetch-vlf-cumiana":
            stored = fetch_manifest_images(
                args.manifest,
                out_root=args.out_root,
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

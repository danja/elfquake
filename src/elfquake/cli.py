"""Command line entry points for raw data acquisition."""

from __future__ import annotations

import argparse
from pathlib import Path

from elfquake.connectors.astronomy import fetch_manifest_json
from elfquake.connectors.ingv import fetch_italy_events
from elfquake.connectors.vlf_cumiana import fetch_manifest_images


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

    args = parser.parse_args()
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
    else:
        parser.error(f"unknown command: {args.command}")

    for capture in stored:
        status = "skipped" if capture.skipped_existing else "stored"
        print(f"{status}: {capture.payload_path}")
        print(f"metadata: {capture.metadata_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

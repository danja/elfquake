"""Command line entry points for ELFQuake workflows."""

from __future__ import annotations

import argparse
import sys
from urllib.error import HTTPError, URLError

from elfquake.cli_commands import register_commands
from elfquake.connectors.astronomy import fetch_manifest_json


def main() -> int:
    parser = argparse.ArgumentParser(prog="elfquake")
    subparsers = parser.add_subparsers(dest="command", required=True)
    register_commands(subparsers)
    args = parser.parse_args()

    try:
        return args.func(args)
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


if __name__ == "__main__":
    raise SystemExit(main())

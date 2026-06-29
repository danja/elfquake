"""Cumiana VLF live image acquisition."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Callable

from elfquake.http import HttpCapture, fetch_bytes, parse_http_datetime
from elfquake.storage import StoredCapture, filename_timestamp, write_capture


def fetch_manifest_images(
    manifest_path: Path,
    *,
    out_root: Path,
    only: set[str] | None = None,
    fetcher: Callable[[str], HttpCapture] = fetch_bytes,
) -> list[StoredCapture]:
    stored: list[StoredCapture] = []
    with manifest_path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            endpoint_id = row["endpoint_id"]
            if only and endpoint_id not in only:
                continue
            capture: HttpCapture = fetcher(row["url"])
            last_modified = parse_http_datetime(capture.headers.get("Last-Modified"))
            timestamp = last_modified or capture.captured_at_utc
            date_dir = timestamp.date().isoformat()
            payload_path = (
                out_root
                / "captures"
                / date_dir
                / f"{endpoint_id}_{filename_timestamp(timestamp)}.jpg"
            )
            stored.append(
                write_capture(
                    payload_path,
                    capture.body,
                    url=capture.url,
                    status=capture.status,
                    captured_at_utc=capture.captured_at_utc,
                    headers=capture.headers,
                    source_id=f"vlf_cumiana_{endpoint_id}",
                    extra_metadata={
                        "endpoint_id": endpoint_id,
                        "station": row.get("station", "cumiana"),
                        "latitude": row.get("latitude", ""),
                        "longitude": row.get("longitude", ""),
                    },
                    skip_existing=True,
                )
            )
    return stored

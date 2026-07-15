"""Manifest-driven passive ELF/VLF capture support for non-Cumiana stations."""

from __future__ import annotations

import csv
import time
from pathlib import Path
from typing import Callable

from elfquake.http import HttpCapture, fetch_bytes, parse_http_datetime
from elfquake.storage import StoredCapture, filename_timestamp, write_capture


def fetch_manifest_captures(manifest_path: Path, *, out_root: Path, source_namespace: str,
                            only: set[str] | None = None,
                            fetcher: Callable[[str], HttpCapture] = fetch_bytes) -> list[StoredCapture]:
    stored: list[StoredCapture] = []
    with manifest_path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            endpoint_id = row["endpoint_id"]
            if only and endpoint_id not in only:
                continue
            capture = fetcher(row["url"])
            timestamp = parse_http_datetime(capture.headers.get("Last-Modified")) or capture.captured_at_utc
            suffix = _suffix(row.get("expected_content_type", ""), row["url"])
            payload_path = out_root / "captures" / timestamp.date().isoformat() / (
                f"{endpoint_id}_{filename_timestamp(timestamp)}{suffix}"
            )
            stored.append(write_capture(
                payload_path, capture.body, url=capture.url, status=capture.status,
                captured_at_utc=capture.captured_at_utc, headers=capture.headers,
                source_id=f"vlf_{source_namespace}_{endpoint_id}",
                extra_metadata={"endpoint_id": endpoint_id, "station": row.get("station", ""),
                                "latitude": row.get("latitude", ""), "longitude": row.get("longitude", ""),
                                "receiver_mode": row.get("receiver_mode", "passive_broadband_elf_vlf"),
                                "region_id": row.get("region_id", "japan")},
                skip_existing=True,
            ))
    return stored


def repeat_manifest_captures(manifest_path: Path, *, out_root: Path, source_namespace: str,
                             cycles: int, interval_seconds: int = 1800,
                             only: set[str] | None = None,
                             fetcher: Callable[[str], HttpCapture] = fetch_bytes,
                             sleeper: Callable[[float], None] = time.sleep) -> list[StoredCapture]:
    if cycles < 0:
        raise ValueError("cycles must be 0 for forever, or at least 1")
    if cycles != 1 and interval_seconds < 60:
        raise ValueError("interval_seconds must be at least 60 for repeated live capture")
    stored: list[StoredCapture] = []
    cycle = 0
    while cycles == 0 or cycle < cycles:
        stored.extend(fetch_manifest_captures(manifest_path, out_root=out_root,
                                              source_namespace=source_namespace, only=only, fetcher=fetcher))
        cycle += 1
        if cycles == 0 or cycle < cycles:
            sleeper(interval_seconds)
    return stored


def _suffix(content_type: str, url: str) -> str:
    value = f"{content_type} {url}".lower()
    if "jpeg" in value or "jpg" in value:
        return ".jpg"
    if "wav" in value:
        return ".wav"
    if "ogg" in value:
        return ".ogg"
    return ".bin"

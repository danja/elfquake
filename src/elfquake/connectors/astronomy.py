"""Astronomical and geomagnetic JSON acquisition."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Callable

from elfquake.http import HttpCapture, fetch_bytes
from elfquake.storage import StoredCapture, filename_timestamp, write_capture


def fetch_manifest_json(
    manifest_path: Path,
    *,
    out_root: Path,
    date: str,
    moon_phase_count: int = 8,
    only: set[str] | None = None,
    fetcher: Callable[[str], HttpCapture] = fetch_bytes,
) -> list[StoredCapture]:
    stored: list[StoredCapture] = []
    with manifest_path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            source_id = row["source_id"]
            if only and source_id not in only:
                continue
            url = _materialize_url(row["url"], date=date, moon_phase_count=moon_phase_count)
            capture: HttpCapture = fetcher(url)
            timestamp = capture.captured_at_utc
            payload_path = (
                out_root
                / "captures"
                / timestamp.date().isoformat()
                / f"{source_id}_{filename_timestamp(timestamp)}.json"
            )
            stored.append(
                write_capture(
                    payload_path,
                    capture.body,
                    url=capture.url,
                    status=capture.status,
                    captured_at_utc=capture.captured_at_utc,
                    headers=capture.headers,
                    source_id=source_id,
                    extra_metadata={"content": row.get("content", ""), "cadence": row.get("cadence", "")},
                    skip_existing=False,
                )
            )
    return stored


def _materialize_url(url: str, *, date: str, moon_phase_count: int) -> str:
    materialized = url.replace("YYYY-MM-DD", date)
    return materialized.replace("nump=N", f"nump={moon_phase_count}")

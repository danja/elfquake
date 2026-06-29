"""INGV FDSN event acquisition."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Callable
from urllib.parse import urlencode

from elfquake.http import HttpCapture, fetch_bytes
from elfquake.storage import StoredCapture, filename_timestamp, write_capture


INGV_EVENT_URL = "https://webservices.ingv.it/fdsnws/event/1/query"
ITALY_BOUNDS = {
    "minlat": "35",
    "maxlat": "48",
    "minlon": "6",
    "maxlon": "19",
}


def build_event_url(
    start_utc: str,
    end_utc: str,
    *,
    min_magnitude: float = 2.0,
    limit: int = 10000,
) -> str:
    query = {
        "starttime": _fdsn_time(start_utc),
        "endtime": _fdsn_time(end_utc),
        "minmag": f"{min_magnitude:g}",
        "maxmag": "10",
        "mindepth": "-10",
        "maxdepth": "1000",
        **ITALY_BOUNDS,
        "minversion": "100",
        "orderby": "time-asc",
        "format": "text",
        "limit": str(limit),
    }
    return f"{INGV_EVENT_URL}?{urlencode(query)}"


def fetch_italy_events(
    start_utc: str,
    end_utc: str,
    *,
    out_root: Path,
    min_magnitude: float = 2.0,
    limit: int = 10000,
    fetcher: Callable[[str], HttpCapture] = fetch_bytes,
) -> StoredCapture:
    url = build_event_url(start_utc, end_utc, min_magnitude=min_magnitude, limit=limit)
    capture: HttpCapture = fetcher(url)
    start_day = _date_slug(start_utc)
    end_day = _date_slug(end_utc)
    captured_slug = filename_timestamp(capture.captured_at_utc)
    payload_path = out_root / f"events_italy_{start_day}_{end_day}_{captured_slug}.txt"
    return write_capture(
        payload_path,
        capture.body,
        url=capture.url,
        status=capture.status,
        captured_at_utc=capture.captured_at_utc,
        headers=capture.headers,
        source_id="ingv_events_italy_text",
        extra_metadata={
            "start_utc": start_utc,
            "end_utc": end_utc,
            "min_magnitude": f"{min_magnitude:g}",
            "limit": str(limit),
        },
        skip_existing=True,
    )


def _date_slug(value: str) -> str:
    return datetime.fromisoformat(value.replace("Z", "+00:00")).date().isoformat()


def _fdsn_time(value: str) -> str:
    return value.removesuffix("Z")

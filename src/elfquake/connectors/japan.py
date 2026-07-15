"""Japan regional earthquake acquisition from the public USGS FDSN service."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Callable
from urllib.parse import urlencode

from elfquake.http import HttpCapture, fetch_bytes
from elfquake.storage import StoredCapture, filename_timestamp, write_capture

USGS_EVENT_URL = "https://earthquake.usgs.gov/fdsnws/event/1/query"
JAPAN_BOUNDS = {"minlatitude": "30", "maxlatitude": "46", "minlongitude": "129", "maxlongitude": "146"}


def build_japan_event_url(start_utc: str, end_utc: str, *, min_magnitude: float = 2.0, limit: int = 20000) -> str:
    query = {
        "format": "geojson", "starttime": _fdsn_time(start_utc), "endtime": _fdsn_time(end_utc),
        "minmagnitude": f"{min_magnitude:g}", "orderby": "time-asc", "limit": str(limit), **JAPAN_BOUNDS,
    }
    return f"{USGS_EVENT_URL}?{urlencode(query)}"


def fetch_japan_events(
    start_utc: str, end_utc: str, *, out_root: Path, min_magnitude: float = 2.0,
    limit: int = 20000, fetcher: Callable[[str], HttpCapture] = fetch_bytes,
) -> StoredCapture:
    url = build_japan_event_url(start_utc, end_utc, min_magnitude=min_magnitude, limit=limit)
    capture = fetcher(url)
    payload_path = out_root / (
        f"events_japan_{_date_slug(start_utc)}_{_date_slug(end_utc)}_"
        f"{filename_timestamp(capture.captured_at_utc)}.json"
    )
    return write_capture(
        payload_path, capture.body, url=capture.url, status=capture.status,
        captured_at_utc=capture.captured_at_utc, headers=capture.headers,
        source_id="usgs_events_japan_geojson",
        extra_metadata={"start_utc": start_utc, "end_utc": end_utc,
                        "min_magnitude": f"{min_magnitude:g}", "limit": str(limit),
                        "region_id": "japan", "country": "JP"},
        skip_existing=True,
    )


def _date_slug(value: str) -> str:
    return datetime.fromisoformat(value.replace("Z", "+00:00")).date().isoformat()


def _fdsn_time(value: str) -> str:
    return value.removesuffix("Z")

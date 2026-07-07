"""Abelian live VLF stream acquisition."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Callable
from urllib.request import Request, urlopen

from elfquake.connectors.vlf_abelian_common import CUMIANA_ENDPOINT, StreamEndpoint
from elfquake.http import USER_AGENT, HttpCapture, utc_now
from elfquake.storage import StoredCapture, filename_timestamp, write_capture


def record_cumiana_stream(
    *,
    out_root: Path,
    duration_seconds: int = 60,
    max_bytes: int | None = None,
    fetcher: Callable[[str, int, int | None], HttpCapture] | None = None,
) -> StoredCapture:
    return record_abelian_stream(
        endpoint=CUMIANA_ENDPOINT,
        out_root=out_root,
        duration_seconds=duration_seconds,
        max_bytes=max_bytes,
        fetcher=fetcher,
    )


def record_abelian_stream(
    *,
    endpoint: StreamEndpoint,
    out_root: Path,
    duration_seconds: int,
    max_bytes: int | None = None,
    fetcher: Callable[[str, int, int | None], HttpCapture] | None = None,
) -> StoredCapture:
    if duration_seconds < 1:
        raise ValueError("duration_seconds must be at least 1")
    if max_bytes is not None and max_bytes < 1:
        raise ValueError("max_bytes must be positive when set")

    fetcher = fetcher or fetch_stream_chunk
    capture = fetcher(endpoint.url, duration_seconds, max_bytes)
    if not capture.body:
        raise ValueError("Abelian live stream returned no audio bytes")
    timestamp = capture.captured_at_utc
    payload_path = (
        out_root
        / "captures"
        / timestamp.date().isoformat()
        / f"abelian_{endpoint.station}_{endpoint.stream_id}_{filename_timestamp(timestamp)}.ogg"
    )
    content_type = capture.headers.get("Content-Type", "")
    return write_capture(
        payload_path,
        capture.body,
        url=capture.url,
        status=capture.status,
        captured_at_utc=timestamp,
        headers=capture.headers,
        source_id=f"vlf_abelian_{endpoint.station}_{endpoint.stream_id}",
        extra_metadata={
            "source_kind": "vlf_live_audio_stream",
            "provider": "abelian",
            "stream_id": endpoint.stream_id,
            "station": endpoint.station,
            "station_label": endpoint.label,
            "latitude": endpoint.latitude,
            "longitude": endpoint.longitude,
            "requested_duration_seconds": str(duration_seconds),
            "captured_byte_count": str(len(capture.body)),
            "content_type": content_type,
        },
        skip_existing=True,
    )


def fetch_stream_chunk(
    url: str,
    duration_seconds: int,
    max_bytes: int | None = None,
    *,
    timeout_seconds: int = 30,
    chunk_size: int = 65536,
) -> HttpCapture:
    request = Request(url, headers={"User-Agent": USER_AGENT})
    captured_at = utc_now()
    deadline = time.monotonic() + duration_seconds
    chunks: list[bytes] = []
    total = 0
    with urlopen(request, timeout=timeout_seconds) as response:
        while time.monotonic() < deadline:
            chunk = response.read(chunk_size)
            if not chunk:
                break
            if max_bytes is not None and total + len(chunk) > max_bytes:
                chunk = chunk[: max_bytes - total]
            chunks.append(chunk)
            total += len(chunk)
            if max_bytes is not None and total >= max_bytes:
                break
        headers = {key: value for key, value in response.headers.items()}
        return HttpCapture(
            url=url,
            status=response.status,
            captured_at_utc=captured_at,
            headers=headers,
            body=b"".join(chunks),
        )

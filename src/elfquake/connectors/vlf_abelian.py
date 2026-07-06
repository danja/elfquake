"""Abelian live and archived VLF acquisition."""

from __future__ import annotations

import time
import csv
import re
from dataclasses import dataclass
from html.parser import HTMLParser
from pathlib import Path
from typing import Callable
from urllib.parse import urljoin, urlencode
from urllib.request import Request, urlopen

from elfquake.http import USER_AGENT, HttpCapture, fetch_bytes, utc_now
from elfquake.storage import StoredCapture, filename_timestamp, write_capture


ABELIAN_BASE_URL = "http://abelian.org/vlf/live-stream.php"
ABELIAN_RETRIEVE_URL = "http://abelian.org/vlf/retrieve.php"
ABELIAN_CUMIANA_STREAM_ID = "vlf15"
ABELIAN_CUMIANA_STATION = "cumiana"
ABELIAN_CUMIANA_LABEL = "Cumiana, NW Italy"
ABELIAN_CUMIANA_LATITUDE = "44.96"
ABELIAN_CUMIANA_LONGITUDE = "7.42"
ARCHIVE_FORMATS = {"sg", "td", "vt", "wav"}
ARCHIVE_EXTENSIONS = {"sg": "png", "td": "png", "vt": "vt", "wav": "wav"}
DOWNLOAD_SIZE_RE = re.compile(r"href=([^>\s]+)[^>]*>[^<]*</a>\s*size\s+(\d+)", re.IGNORECASE)


@dataclass(frozen=True)
class StreamEndpoint:
    stream_id: str
    station: str
    label: str
    latitude: str
    longitude: str

    @property
    def url(self) -> str:
        return f"{ABELIAN_BASE_URL}?stream={self.stream_id}"


CUMIANA_ENDPOINT = StreamEndpoint(
    stream_id=ABELIAN_CUMIANA_STREAM_ID,
    station=ABELIAN_CUMIANA_STATION,
    label=ABELIAN_CUMIANA_LABEL,
    latitude=ABELIAN_CUMIANA_LATITUDE,
    longitude=ABELIAN_CUMIANA_LONGITUDE,
)


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


def fetch_cumiana_archive_request(
    *,
    start_time_utc: str,
    duration_seconds: float,
    out_root: Path,
    output_format: str = "wav",
    fetcher: Callable[[str], HttpCapture] = fetch_bytes,
) -> StoredCapture:
    return fetch_abelian_archive_request(
        endpoint=CUMIANA_ENDPOINT,
        start_time_utc=start_time_utc,
        duration_seconds=duration_seconds,
        output_format=output_format,
        out_root=out_root,
        fetcher=fetcher,
    )


def fetch_cumiana_archive(
    *,
    start_time_utc: str,
    duration_seconds: float,
    out_root: Path,
    output_format: str = "wav",
    fetcher: Callable[[str], HttpCapture] = fetch_bytes,
) -> list[StoredCapture]:
    return fetch_abelian_archive(
        endpoint=CUMIANA_ENDPOINT,
        start_time_utc=start_time_utc,
        duration_seconds=duration_seconds,
        output_format=output_format,
        out_root=out_root,
        fetcher=fetcher,
    )


def probe_cumiana_archive(
    *,
    start_times_utc: list[str],
    duration_seconds: float,
    output_formats: list[str],
    out_path: Path,
    fetch_downloads: bool = False,
    fetcher: Callable[[str], HttpCapture] = fetch_bytes,
) -> list[dict[str, str]]:
    rows = probe_abelian_archive(
        endpoint=CUMIANA_ENDPOINT,
        start_times_utc=start_times_utc,
        duration_seconds=duration_seconds,
        output_formats=output_formats,
        fetch_downloads=fetch_downloads,
        fetcher=fetcher,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=ARCHIVE_PROBE_FIELDNAMES, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    return rows


def probe_abelian_archive(
    *,
    endpoint: StreamEndpoint,
    start_times_utc: list[str],
    duration_seconds: float,
    output_formats: list[str],
    fetch_downloads: bool = False,
    fetcher: Callable[[str], HttpCapture] = fetch_bytes,
) -> list[dict[str, str]]:
    rows = []
    for start_time_utc in start_times_utc:
        for output_format in output_formats:
            request_capture = _fetch_archive_request_capture(
                endpoint=endpoint,
                start_time_utc=start_time_utc,
                duration_seconds=duration_seconds,
                output_format=output_format,
                fetcher=fetcher,
            )
            response = summarize_archive_response(
                request_capture.body.decode("utf-8", errors="replace"),
                base_url=request_capture.url,
            )
            fetched_download_count = 0
            fetched_download_byte_count = 0
            if fetch_downloads:
                for link in response["download_links"]:
                    download = fetcher(link)
                    if download.body:
                        fetched_download_count += 1
                        fetched_download_byte_count += len(download.body)
            declared_size = response["declared_download_size_bytes"]
            usable = fetched_download_byte_count > 0 or declared_size > 0
            rows.append({
                "start_time_utc": start_time_utc,
                "duration_seconds": f"{duration_seconds:g}",
                "format": output_format,
                "station": endpoint.station,
                "stream_id": endpoint.stream_id,
                "request_url": request_capture.url,
                "http_status": str(request_capture.status),
                "response_byte_count": str(len(request_capture.body)),
                "no_database": "1" if response["no_database"] else "0",
                "download_link_count": str(len(response["download_links"])),
                "declared_download_size_bytes": str(declared_size),
                "fetched_download_count": str(fetched_download_count),
                "fetched_download_byte_count": str(fetched_download_byte_count),
                "usable_nonempty": "1" if usable else "0",
            })
    return rows


def fetch_abelian_archive(
    *,
    endpoint: StreamEndpoint,
    start_time_utc: str,
    duration_seconds: float,
    output_format: str,
    out_root: Path,
    fetcher: Callable[[str], HttpCapture] = fetch_bytes,
) -> list[StoredCapture]:
    request_capture = _fetch_archive_request_capture(
        endpoint=endpoint,
        start_time_utc=start_time_utc,
        duration_seconds=duration_seconds,
        output_format=output_format,
        fetcher=fetcher,
    )
    stored = [
        _store_archive_request_capture(
            capture=request_capture,
            endpoint=endpoint,
            start_time_utc=start_time_utc,
            duration_seconds=duration_seconds,
            output_format=output_format,
            out_root=out_root,
        )
    ]
    links = extract_archive_download_links(request_capture.body.decode("utf-8", errors="replace"), request_capture.url)
    for index, link in enumerate(links):
        download = fetcher(link)
        if not download.body:
            continue
        stored.append(_store_archive_download_capture(
            capture=download,
            endpoint=endpoint,
            start_time_utc=start_time_utc,
            duration_seconds=duration_seconds,
            output_format=output_format,
            out_root=out_root,
            link_index=index,
        ))
    return stored


ARCHIVE_PROBE_FIELDNAMES = [
    "start_time_utc",
    "duration_seconds",
    "format",
    "station",
    "stream_id",
    "request_url",
    "http_status",
    "response_byte_count",
    "no_database",
    "download_link_count",
    "declared_download_size_bytes",
    "fetched_download_count",
    "fetched_download_byte_count",
    "usable_nonempty",
]


def fetch_abelian_archive_request(
    *,
    endpoint: StreamEndpoint,
    start_time_utc: str,
    duration_seconds: float,
    output_format: str,
    out_root: Path,
    fetcher: Callable[[str], HttpCapture] = fetch_bytes,
) -> StoredCapture:
    capture = _fetch_archive_request_capture(
        endpoint=endpoint,
        start_time_utc=start_time_utc,
        duration_seconds=duration_seconds,
        output_format=output_format,
        fetcher=fetcher,
    )
    return _store_archive_request_capture(
        capture=capture,
        endpoint=endpoint,
        start_time_utc=start_time_utc,
        duration_seconds=duration_seconds,
        output_format=output_format,
        out_root=out_root,
    )


def _fetch_archive_request_capture(
    *,
    endpoint: StreamEndpoint,
    start_time_utc: str,
    duration_seconds: float,
    output_format: str,
    fetcher: Callable[[str], HttpCapture],
) -> HttpCapture:
    if duration_seconds < 0.05 or duration_seconds > 120:
        raise ValueError("duration_seconds must be between 0.05 and 120")
    if output_format not in ARCHIVE_FORMATS:
        raise ValueError(f"output_format must be one of: {', '.join(sorted(ARCHIVE_FORMATS))}")

    url = build_archive_retrieve_url(
        endpoint=endpoint,
        start_time_utc=start_time_utc,
        duration_seconds=duration_seconds,
        output_format=output_format,
    )
    return fetcher(url)


def _store_archive_request_capture(
    *,
    capture: HttpCapture,
    endpoint: StreamEndpoint,
    start_time_utc: str,
    duration_seconds: float,
    output_format: str,
    out_root: Path,
) -> StoredCapture:
    timestamp = capture.captured_at_utc
    safe_start = start_time_utc.replace(":", "-").replace(" ", "T").replace(".", "-")
    payload_path = (
        out_root
        / "archive"
        / safe_start[:10]
        / (
            f"abelian_archive_{endpoint.station}_{endpoint.stream_id}_{safe_start}_"
            f"{duration_seconds:g}s_{output_format}_{filename_timestamp(timestamp)}.html"
        )
    )
    content_type = capture.headers.get("Content-Type", "")
    return write_capture(
        payload_path,
        capture.body,
        url=capture.url,
        status=capture.status,
        captured_at_utc=timestamp,
        headers=capture.headers,
        source_id=f"vlf_abelian_archive_{endpoint.station}_{endpoint.stream_id}",
        extra_metadata={
            "source_kind": "vlf_archive_retrieve_response",
            "provider": "abelian",
            "stream_id": endpoint.stream_id,
            "station": endpoint.station,
            "station_label": endpoint.label,
            "latitude": endpoint.latitude,
            "longitude": endpoint.longitude,
            "archive_start_time_utc": start_time_utc,
            "archive_duration_seconds": f"{duration_seconds:g}",
            "archive_format": output_format,
            "captured_byte_count": str(len(capture.body)),
            "content_type": content_type,
        },
        skip_existing=True,
    )


def _store_archive_download_capture(
    *,
    capture: HttpCapture,
    endpoint: StreamEndpoint,
    start_time_utc: str,
    duration_seconds: float,
    output_format: str,
    out_root: Path,
    link_index: int,
) -> StoredCapture:
    timestamp = capture.captured_at_utc
    safe_start = start_time_utc.replace(":", "-").replace(" ", "T").replace(".", "-")
    extension = ARCHIVE_EXTENSIONS[output_format]
    payload_path = (
        out_root
        / "archive"
        / safe_start[:10]
        / (
            f"abelian_archive_{endpoint.station}_{endpoint.stream_id}_{safe_start}_"
            f"{duration_seconds:g}s_{output_format}_{link_index}_{filename_timestamp(timestamp)}.{extension}"
        )
    )
    content_type = capture.headers.get("Content-Type", "")
    return write_capture(
        payload_path,
        capture.body,
        url=capture.url,
        status=capture.status,
        captured_at_utc=timestamp,
        headers=capture.headers,
        source_id=f"vlf_abelian_archive_download_{endpoint.station}_{endpoint.stream_id}",
        extra_metadata={
            "source_kind": "vlf_archive_download",
            "provider": "abelian",
            "stream_id": endpoint.stream_id,
            "station": endpoint.station,
            "station_label": endpoint.label,
            "latitude": endpoint.latitude,
            "longitude": endpoint.longitude,
            "archive_start_time_utc": start_time_utc,
            "archive_duration_seconds": f"{duration_seconds:g}",
            "archive_format": output_format,
            "download_link_index": str(link_index),
            "captured_byte_count": str(len(capture.body)),
            "content_type": content_type,
        },
        skip_existing=True,
    )


def extract_archive_download_links(html: str, base_url: str = ABELIAN_RETRIEVE_URL) -> list[str]:
    parser = _ArchiveDownloadParser()
    parser.feed(html)
    links = []
    for href in parser.hrefs:
        if "/vlf/live/retrieve/" not in href:
            continue
        normalized = href.replace("http:/vlf/", "http://abelian.org/vlf/")
        links.append(urljoin(base_url, normalized))
    return links


def summarize_archive_response(html: str, base_url: str = ABELIAN_RETRIEVE_URL) -> dict[str, object]:
    links = extract_archive_download_links(html, base_url=base_url)
    declared_size = 0
    for _href, size in DOWNLOAD_SIZE_RE.findall(html):
        declared_size += int(size)
    return {
        "no_database": "no database" in html.lower(),
        "download_links": links,
        "declared_download_size_bytes": declared_size,
    }


def build_archive_retrieve_url(
    *,
    endpoint: StreamEndpoint,
    start_time_utc: str,
    duration_seconds: float,
    output_format: str = "wav",
) -> str:
    if output_format not in ARCHIVE_FORMATS:
        raise ValueError(f"output_format must be one of: {', '.join(sorted(ARCHIVE_FORMATS))}")
    query = {
        "ts": start_time_utc.replace("T", " ").replace("Z", ""),
        "len": f"{duration_seconds:g}",
        endpoint.stream_id: "on",
        "format": output_format,
        "submit": " Proceed ",
    }
    return f"{ABELIAN_RETRIEVE_URL}?{urlencode(query)}"


class _ArchiveDownloadParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.hrefs: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag.lower() != "a":
            return
        for key, value in attrs:
            if key.lower() == "href" and value:
                self.hrefs.append(value)


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

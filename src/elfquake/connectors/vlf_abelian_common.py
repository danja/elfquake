"""Shared Abelian VLF source definitions."""

from __future__ import annotations

from dataclasses import dataclass


ABELIAN_BASE_URL = "http://abelian.org/vlf/live-stream.php"
ABELIAN_RETRIEVE_URL = "http://abelian.org/vlf/retrieve.php"
ABELIAN_CUMIANA_STREAM_ID = "vlf15"
ABELIAN_CUMIANA_STATION = "cumiana"
ABELIAN_CUMIANA_LABEL = "Cumiana, NW Italy"
ABELIAN_CUMIANA_LATITUDE = "44.96"
ABELIAN_CUMIANA_LONGITUDE = "7.42"


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

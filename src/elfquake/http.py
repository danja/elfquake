"""Small HTTP helpers for raw data acquisition."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from urllib.request import Request, urlopen


USER_AGENT = "ELFQuake/0.1 research contact: local"


@dataclass(frozen=True)
class HttpCapture:
    url: str
    status: int
    captured_at_utc: datetime
    headers: dict[str, str]
    body: bytes


def utc_now() -> datetime:
    return datetime.now(timezone.utc).replace(microsecond=0)


def fetch_bytes(url: str, timeout_seconds: int = 30) -> HttpCapture:
    request = Request(url, headers={"User-Agent": USER_AGENT})
    captured_at = utc_now()
    with urlopen(request, timeout=timeout_seconds) as response:
        body = response.read()
        headers = {key: value for key, value in response.headers.items()}
        return HttpCapture(
            url=url,
            status=response.status,
            captured_at_utc=captured_at,
            headers=headers,
            body=body,
        )


def parse_http_datetime(value: str | None) -> datetime | None:
    if not value:
        return None
    parsed = parsedate_to_datetime(value)
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)

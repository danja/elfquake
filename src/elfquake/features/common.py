"""Shared feature helpers."""

from __future__ import annotations

from datetime import datetime, timezone


def parse_utc(value: str) -> datetime:
    if not value.endswith("Z"):
        raise ValueError(f"expected UTC timestamp ending in Z: {value}")
    return datetime.fromisoformat(value[:-1] + "+00:00")


def format_utc(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def http_datetime_to_utc(value: str) -> str:
    if not value:
        return ""
    parsed = datetime.strptime(value, "%a, %d %b %Y %H:%M:%S GMT").replace(tzinfo=timezone.utc)
    return format_utc(parsed)

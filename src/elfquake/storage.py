"""Raw payload storage utilities."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from tempfile import NamedTemporaryFile


@dataclass(frozen=True)
class StoredCapture:
    payload_path: Path
    metadata_path: Path
    skipped_existing: bool


def iso_utc(value: datetime) -> str:
    return value.astimezone(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def filename_timestamp(value: datetime) -> str:
    return iso_utc(value).replace(":", "-")


def write_capture(
    payload_path: Path,
    body: bytes,
    *,
    url: str,
    status: int,
    captured_at_utc: datetime,
    headers: dict[str, str],
    source_id: str,
    extra_metadata: dict[str, str] | None = None,
    skip_existing: bool = True,
) -> StoredCapture:
    metadata_path = payload_path.with_suffix(payload_path.suffix + ".metadata.json")
    if skip_existing and payload_path.exists() and metadata_path.exists():
        return StoredCapture(payload_path, metadata_path, skipped_existing=True)

    payload_path.parent.mkdir(parents=True, exist_ok=True)
    _write_bytes_atomic(payload_path, body)

    metadata = {
        "source_id": source_id,
        "url": url,
        "status": status,
        "captured_at_utc": iso_utc(captured_at_utc),
        "headers": headers,
    }
    if extra_metadata:
        metadata.update(extra_metadata)
    _write_text_atomic(metadata_path, json.dumps(metadata, indent=2, sort_keys=True) + "\n")

    return StoredCapture(payload_path, metadata_path, skipped_existing=False)


def _write_bytes_atomic(path: Path, body: bytes) -> None:
    with NamedTemporaryFile(dir=path.parent, delete=False) as temp:
        temp.write(body)
        temp_path = Path(temp.name)
    temp_path.replace(path)


def _write_text_atomic(path: Path, text: str) -> None:
    with NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as temp:
        temp.write(text)
        temp_path = Path(temp.name)
    temp_path.replace(path)

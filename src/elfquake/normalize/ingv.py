"""Normalize INGV FDSN event text exports."""

from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path

from elfquake.storage import iso_utc


NORMALIZED_FIELDS = [
    "event_id",
    "source",
    "event_time_utc",
    "latitude",
    "longitude",
    "depth_km",
    "magnitude",
    "magnitude_type",
    "italy_region",
    "event_location_name",
    "event_type",
    "raw_file",
    "ingested_at_utc",
    "raw_uri",
]


def normalize_ingv_event_text(
    raw_path: Path,
    out_path: Path,
    *,
    raw_uri: str | None = None,
    ingested_at_utc: str | None = None,
    only_region: str | None = None,
) -> int:
    metadata = _read_metadata(raw_path)
    resolved_uri = raw_uri or metadata.get("url", "")
    resolved_ingested_at = ingested_at_utc or metadata.get("captured_at_utc", iso_utc(_utc_now()))

    rows = list(_normalized_rows(raw_path, resolved_uri, resolved_ingested_at))
    if only_region:
        rows = [row for row in rows if row["italy_region"] == only_region]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=NORMALIZED_FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    return len(rows)


def normalize_row(raw: dict[str, str], *, raw_file: str, raw_uri: str, ingested_at_utc: str) -> dict[str, str]:
    latitude = float(raw["Latitude"])
    longitude = float(raw["Longitude"])
    return {
        "event_id": raw["EventID"],
        "source": "ingv_fdsn_event_text",
        "event_time_utc": _utc_timestamp(raw["Time"]),
        "latitude": _validated_number(raw["Latitude"]),
        "longitude": _validated_number(raw["Longitude"]),
        "depth_km": _validated_number(raw["Depth/Km"]),
        "magnitude": _validated_number(raw["Magnitude"]),
        "magnitude_type": raw["MagType"],
        "italy_region": italy_region(latitude, longitude),
        "event_location_name": raw["EventLocationName"],
        "event_type": raw["EventType"],
        "raw_file": raw_file,
        "ingested_at_utc": ingested_at_utc,
        "raw_uri": raw_uri,
    }


def italy_region(latitude: float, longitude: float) -> str:
    if 41.5 <= latitude <= 43.5 and 12.0 <= longitude <= 14.5:
        return "central_italy"
    return "unknown"


def _normalized_rows(raw_path: Path, raw_uri: str, ingested_at_utc: str) -> list[dict[str, str]]:
    with raw_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(_strip_comment_header(handle), delimiter="|")
        return [
            normalize_row(
                row,
                raw_file=str(raw_path),
                raw_uri=raw_uri,
                ingested_at_utc=ingested_at_utc,
            )
            for row in reader
        ]


def _strip_comment_header(lines):
    for line in lines:
        if line.startswith("#"):
            yield line[1:]
        else:
            yield line


def _read_metadata(raw_path: Path) -> dict[str, str]:
    metadata_path = raw_path.with_suffix(raw_path.suffix + ".metadata.json")
    if not metadata_path.exists():
        return {}
    return json.loads(metadata_path.read_text(encoding="utf-8"))


def _utc_timestamp(value: str) -> str:
    return value if value.endswith("Z") else f"{value}Z"


def _validated_number(value: str) -> str:
    float(value)
    return value


def _utc_now() -> datetime:
    return datetime.now(timezone.utc).replace(microsecond=0)

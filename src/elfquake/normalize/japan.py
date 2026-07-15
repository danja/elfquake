"""Normalize USGS Japan GeoJSON event captures into the shared event table."""

from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path

from elfquake.normalize.ingv import NORMALIZED_FIELDS
from elfquake.storage import iso_utc

JAPAN_FIELDS = NORMALIZED_FIELDS + ["region_id", "country"]


def normalize_japan_event_json(raw_path: Path, out_path: Path) -> int:
    metadata = _read_metadata(raw_path)
    payload = json.loads(raw_path.read_text(encoding="utf-8"))
    rows = [normalize_feature(
        feature, raw_file=str(raw_path), raw_uri=metadata.get("url", ""),
        ingested_at_utc=metadata.get("captured_at_utc", iso_utc(_utc_now())),
    ) for feature in payload.get("features", []) if _valid_feature(feature)]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=JAPAN_FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    return len(rows)


def normalize_feature(feature: dict, *, raw_file: str, raw_uri: str, ingested_at_utc: str) -> dict[str, str]:
    properties = feature["properties"]
    longitude, latitude, depth_km = feature["geometry"]["coordinates"][:3]
    return {
        "event_id": str(feature.get("id") or properties.get("code") or ""),
        "source": "usgs_fdsn_event_geojson",
        "event_time_utc": iso_utc(datetime.fromtimestamp(properties["time"] / 1000, tz=timezone.utc)),
        "latitude": _number(latitude), "longitude": _number(longitude), "depth_km": _number(depth_km),
        "magnitude": _number(properties.get("mag")), "magnitude_type": str(properties.get("magType") or ""),
        "italy_region": "", "event_location_name": str(properties.get("place") or ""),
        "event_type": str(properties.get("type") or "earthquake"), "raw_file": raw_file,
        "ingested_at_utc": ingested_at_utc, "raw_uri": raw_uri, "region_id": "japan", "country": "JP",
    }


def _valid_feature(feature: dict) -> bool:
    properties = feature.get("properties") or {}
    coordinates = (feature.get("geometry") or {}).get("coordinates") or []
    return properties.get("time") is not None and properties.get("mag") is not None and len(coordinates) >= 3


def _number(value: object) -> str:
    return "" if value is None else f"{float(value):g}"


def _read_metadata(raw_path: Path) -> dict[str, str]:
    metadata_path = raw_path.with_suffix(raw_path.suffix + ".metadata.json")
    if not metadata_path.exists():
        return {}
    return json.loads(metadata_path.read_text(encoding="utf-8"))


def _utc_now() -> datetime:
    return datetime.now(timezone.utc).replace(microsecond=0)

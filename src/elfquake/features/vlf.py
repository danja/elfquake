"""Coarse VLF capture feature extraction."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Iterable

from elfquake.features.common import http_datetime_to_utc, parse_utc


FIELDNAMES = [
    "window_start_utc",
    "window_end_utc",
    "vlf_capture_count",
    "vlf_latest_capture_utc",
    "vlf_latest_last_modified_utc",
    "vlf_latest_age_seconds",
    "vlf_total_bytes",
    "vlf_jpeg_count",
    "vlf_latest_width_px",
    "vlf_latest_height_px",
    "vlf_latest_entropy_bits_per_byte",
    "quality_missing_vlf",
    "quality_stale_vlf",
]


def build_vlf_features(
    *,
    metadata_paths: Iterable[Path],
    window_start_utc: str,
    window_end_utc: str,
    out_path: Path,
) -> dict[str, str]:
    window_start = parse_utc(window_start_utc)
    window_end = parse_utc(window_end_utc)
    captures = []
    for metadata_path in metadata_paths:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        captured_at = metadata.get("captured_at_utc", "")
        if not captured_at:
            continue
        captured_dt = parse_utc(captured_at)
        if not (window_start <= captured_dt < window_end):
            continue
        payload_path = Path(str(metadata_path).removesuffix(".metadata.json"))
        last_modified = http_datetime_to_utc(metadata.get("headers", {}).get("Last-Modified", ""))
        captures.append((metadata, payload_path, captured_dt, last_modified))

    latest = max(captures, key=lambda item: item[2], default=None)
    latest_modified = latest[3] if latest else ""
    latest_age = ""
    if latest_modified:
        latest_age = str(int((window_end - parse_utc(latest_modified)).total_seconds()))

    total_bytes = 0
    jpeg_count = 0
    latest_width = ""
    latest_height = ""
    latest_entropy = ""
    for _, payload_path, _, _ in captures:
        if payload_path.exists():
            body = payload_path.read_bytes()
            total_bytes += len(body)
            if body.startswith(b"\xff\xd8"):
                jpeg_count += 1
            if latest and payload_path == latest[1]:
                width, height = jpeg_dimensions(body)
                latest_width = str(width) if width else ""
                latest_height = str(height) if height else ""
                latest_entropy = byte_entropy(body)

    row = {
        "window_start_utc": window_start_utc,
        "window_end_utc": window_end_utc,
        "vlf_capture_count": str(len(captures)),
        "vlf_latest_capture_utc": latest[0].get("captured_at_utc", "") if latest else "",
        "vlf_latest_last_modified_utc": latest_modified,
        "vlf_latest_age_seconds": latest_age,
        "vlf_total_bytes": str(total_bytes),
        "vlf_jpeg_count": str(jpeg_count),
        "vlf_latest_width_px": latest_width,
        "vlf_latest_height_px": latest_height,
        "vlf_latest_entropy_bits_per_byte": latest_entropy,
        "quality_missing_vlf": "0" if captures else "1",
        "quality_stale_vlf": "1" if latest_modified and parse_utc(latest_modified) < window_start else "0",
    }
    _write_one_row(out_path, row)
    return row


def _write_one_row(out_path: Path, row: dict[str, str]) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES, lineterminator="\n")
        writer.writeheader()
        writer.writerow(row)


def byte_entropy(body: bytes) -> str:
    if not body:
        return ""
    counts = [0] * 256
    for byte in body:
        counts[byte] += 1
    length = len(body)
    entropy = 0.0
    for count in counts:
        if count:
            probability = count / length
            entropy -= probability * math.log2(probability)
    return f"{entropy:.6f}"


def jpeg_dimensions(body: bytes) -> tuple[int | None, int | None]:
    if not body.startswith(b"\xff\xd8"):
        return None, None
    index = 2
    while index + 9 < len(body):
        if body[index] != 0xFF:
            index += 1
            continue
        marker = body[index + 1]
        index += 2
        while marker == 0xFF and index < len(body):
            marker = body[index]
            index += 1
        if marker in (0xD8, 0xD9):
            continue
        if index + 2 > len(body):
            return None, None
        segment_length = int.from_bytes(body[index : index + 2], "big")
        if segment_length < 2 or index + segment_length > len(body):
            return None, None
        if marker in {
            0xC0,
            0xC1,
            0xC2,
            0xC3,
            0xC5,
            0xC6,
            0xC7,
            0xC9,
            0xCA,
            0xCB,
            0xCD,
            0xCE,
            0xCF,
        }:
            if segment_length < 7:
                return None, None
            height = int.from_bytes(body[index + 3 : index + 5], "big")
            width = int.from_bytes(body[index + 5 : index + 7], "big")
            return width, height
        index += segment_length
    return None, None

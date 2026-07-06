"""Features for Abelian/Cumiana VLF audio stream captures."""

from __future__ import annotations

import csv
import json
import math
import subprocess
from pathlib import Path


FIELDNAMES = [
    "vlf_audio_source_file",
    "vlf_audio_captured_at_utc",
    "vlf_audio_provider",
    "vlf_audio_station",
    "vlf_audio_stream_id",
    "vlf_audio_content_type",
    "vlf_audio_byte_count",
    "vlf_audio_entropy_bits_per_byte",
    "vlf_audio_ogg_page_count",
    "vlf_audio_ffprobe_available",
    "vlf_audio_duration_seconds",
    "vlf_audio_sample_rate_hz",
    "vlf_audio_channel_count",
    "quality_missing_vlf_audio",
    "quality_unreadable_vlf_audio",
]


def build_vlf_audio_features(
    *,
    audio_paths: list[Path],
    out_path: Path,
    use_ffprobe: bool = True,
) -> list[dict[str, str]]:
    rows = [extract_vlf_audio_features(path, use_ffprobe=use_ffprobe) for path in audio_paths]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    return rows


def extract_vlf_audio_features(audio_path: Path, *, use_ffprobe: bool = True) -> dict[str, str]:
    metadata = _read_metadata(audio_path)
    body = audio_path.read_bytes() if audio_path.exists() else b""
    probe = _ffprobe(audio_path) if use_ffprobe and body else {}
    row = {
        "vlf_audio_source_file": str(audio_path),
        "vlf_audio_captured_at_utc": str(metadata.get("captured_at_utc", "")),
        "vlf_audio_provider": str(metadata.get("provider", "")),
        "vlf_audio_station": str(metadata.get("station", "")),
        "vlf_audio_stream_id": str(metadata.get("stream_id", "")),
        "vlf_audio_content_type": str(metadata.get("content_type", "")),
        "vlf_audio_byte_count": str(len(body)),
        "vlf_audio_entropy_bits_per_byte": _byte_entropy(body),
        "vlf_audio_ogg_page_count": str(body.count(b"OggS")) if body else "0",
        "vlf_audio_ffprobe_available": "1" if probe else "0",
        "vlf_audio_duration_seconds": str(probe.get("duration", "")),
        "vlf_audio_sample_rate_hz": str(probe.get("sample_rate", "")),
        "vlf_audio_channel_count": str(probe.get("channels", "")),
        "quality_missing_vlf_audio": "0" if body else "1",
        "quality_unreadable_vlf_audio": "0" if body and (not use_ffprobe or probe) else "1",
    }
    return {field: row.get(field, "") for field in FIELDNAMES}


def _read_metadata(audio_path: Path) -> dict[str, object]:
    metadata_path = audio_path.with_suffix(audio_path.suffix + ".metadata.json")
    if not metadata_path.exists():
        return {}
    return json.loads(metadata_path.read_text(encoding="utf-8"))


def _ffprobe(audio_path: Path) -> dict[str, str]:
    command = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "a:0",
        "-show_entries",
        "stream=sample_rate,channels:format=duration",
        "-of",
        "json",
        str(audio_path),
    ]
    try:
        completed = subprocess.run(command, check=True, capture_output=True, text=True)
    except (FileNotFoundError, subprocess.CalledProcessError):
        return {}
    try:
        payload = json.loads(completed.stdout)
    except json.JSONDecodeError:
        return {}
    stream = (payload.get("streams") or [{}])[0]
    return {
        "duration": _fmt_float((payload.get("format") or {}).get("duration", "")),
        "sample_rate": str(stream.get("sample_rate", "")),
        "channels": str(stream.get("channels", "")),
    }


def _byte_entropy(body: bytes) -> str:
    if not body:
        return ""
    counts = [0] * 256
    for byte in body:
        counts[byte] += 1
    entropy = 0.0
    for count in counts:
        if count:
            probability = count / len(body)
            entropy -= probability * math.log2(probability)
    return f"{entropy:.6f}"


def _fmt_float(value: object) -> str:
    try:
        return f"{float(value):.6f}"
    except (TypeError, ValueError):
        return ""

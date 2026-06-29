"""Backfill planning helpers."""

from __future__ import annotations

import csv
from datetime import timedelta
from pathlib import Path

from elfquake.features.common import format_utc, parse_utc


FIELDNAMES = ["window_start_utc", "window_end_utc", "command"]


def plan_ingv_backfill(
    *,
    start_utc: str,
    end_utc: str,
    out_path: Path,
    chunk_days: int = 14,
    min_magnitude: str = "2.0",
) -> list[dict[str, str]]:
    if chunk_days < 1:
        raise ValueError("chunk_days must be at least 1")
    start = parse_utc(start_utc)
    end = parse_utc(end_utc)
    if end <= start:
        raise ValueError("end_utc must be after start_utc")

    rows = []
    cursor = start
    delta = timedelta(days=chunk_days)
    while cursor < end:
        chunk_end = min(cursor + delta, end)
        chunk_start_utc = format_utc(cursor)
        chunk_end_utc = format_utc(chunk_end)
        rows.append(
            {
                "window_start_utc": chunk_start_utc,
                "window_end_utc": chunk_end_utc,
                "command": (
                    "PYTHONPATH=src python -m elfquake.cli fetch-ingv-events "
                    f"--start {chunk_start_utc} --end {chunk_end_utc} --min-mag {min_magnitude}"
                ),
            }
        )
        cursor = chunk_end

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    return rows

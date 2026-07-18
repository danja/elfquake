"""Null controls that preserve spatial label patterns while removing time order."""

from __future__ import annotations

import csv
import random
from pathlib import Path


TARGET_VECTOR_FIELDS = (
    "target_event_count",
    "target_occurred",
    "target_magnitude_min",
    "target_start_utc",
    "target_end_utc",
    "target_status",
)


def permute_spatial_target_vectors(
    *, input_csv: Path, out_path: Path, seed: int = 42, time_field: str = "window_start_utc"
) -> dict[str, int | str]:
    with input_csv.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fieldnames = list(reader.fieldnames or [])
        rows = list(reader)
    if not fieldnames or time_field not in fieldnames:
        raise ValueError(f"missing time field: {time_field}")

    groups: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        groups.setdefault(row.get(time_field, ""), []).append(row)
    labeled_times = [
        time
        for time, group in groups.items()
        if group and all(row.get("target_occurred", "") in {"0", "1"} for row in group)
    ]
    vectors = [
        [[row.get(field, "") for field in TARGET_VECTOR_FIELDS] for row in groups[time]]
        for time in labeled_times
    ]
    random.Random(seed).shuffle(vectors)
    for time, vector in zip(labeled_times, vectors):
        for row, source in zip(groups[time], vector):
            for field, value in zip(TARGET_VECTOR_FIELDS, source):
                if field in row:
                    row[field] = value

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    return {"row_count": len(rows), "labeled_time_count": len(labeled_times), "seed": seed, "output": str(out_path)}

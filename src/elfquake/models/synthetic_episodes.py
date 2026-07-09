"""Annotate synthetic windows with deterministic episode identifiers."""

from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path


EPISODE_FIELDS = [
    "synthetic_episode_index",
    "synthetic_episode_id",
    "synthetic_episode_row_index",
    "synthetic_episode_group_count",
]


def annotate_synthetic_episodes(
    *,
    input_csv: Path,
    out_csv: Path,
    report_path: Path,
    group_field: str = "dataset_id",
    time_field: str = "window_start_utc",
    rows_per_episode: int = 24,
    target_field: str = "eventlist_target_occurred",
    drop_partial: bool = False,
) -> dict[str, object]:
    if rows_per_episode < 2:
        raise ValueError("rows_per_episode must be at least 2")

    rows, fieldnames = _read_rows(input_csv)
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[row.get(group_field, "")].append(row)

    output_rows: list[dict[str, str]] = []
    episode_summaries: list[dict[str, object]] = []
    for group_id, group_rows in sorted(grouped.items()):
        ordered = sorted(group_rows, key=lambda row: row.get(time_field, ""))
        episode_count = (len(ordered) + rows_per_episode - 1) // rows_per_episode
        for episode_index in range(episode_count):
            left = episode_index * rows_per_episode
            right = min(len(ordered), left + rows_per_episode)
            episode_rows = ordered[left:right]
            if drop_partial and len(episode_rows) < rows_per_episode:
                continue
            episode_id = f"{group_id}_e{episode_index:03d}"
            for row_index, row in enumerate(episode_rows):
                output = dict(row)
                output.update(
                    {
                        "synthetic_episode_index": str(episode_index),
                        "synthetic_episode_id": episode_id,
                        "synthetic_episode_row_index": str(row_index),
                        "synthetic_episode_group_count": str(len(episode_rows)),
                    }
                )
                output_rows.append(output)
            episode_summaries.append(_episode_summary(episode_id, group_id, episode_rows, target_field))

    out_fields = [field for field in fieldnames if field not in EPISODE_FIELDS] + EPISODE_FIELDS
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=out_fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(output_rows)

    report = {
        "schema": "elfquake.synthetic_episodes.v1",
        "input_csv": str(input_csv),
        "out_csv": str(out_csv),
        "row_count": len(rows),
        "output_row_count": len(output_rows),
        "group_field": group_field,
        "time_field": time_field,
        "rows_per_episode": rows_per_episode,
        "target_field": target_field,
        "drop_partial": drop_partial,
        "episode_count": len(episode_summaries),
        "episodes": episode_summaries,
        "guidance": "Episode ids support diagnostics and engineering splits; do not use episode index or row index as predictive features.",
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def _read_rows(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        return list(reader), list(reader.fieldnames or [])


def _episode_summary(
    episode_id: str,
    group_id: str,
    rows: list[dict[str, str]],
    target_field: str,
) -> dict[str, object]:
    counts = Counter(row.get(target_field, "") for row in rows)
    positives = counts.get("1", 0)
    negatives = counts.get("0", 0)
    total = positives + negatives
    return {
        "episode_id": episode_id,
        "group_id": group_id,
        "row_count": len(rows),
        "positive_count": positives,
        "negative_count": negatives,
        "positive_rate": round(positives / total, 6) if total else None,
        "start_utc": rows[0].get("window_start_utc", "") if rows else "",
        "end_utc": rows[-1].get("window_start_utc", "") if rows else "",
    }

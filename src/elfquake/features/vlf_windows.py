"""Build VLF feature rows aligned to training windows."""

from __future__ import annotations

import csv
from pathlib import Path

from elfquake.features.vlf import FIELDNAMES as VLF_FIELDNAMES
from elfquake.features.vlf import summarize_vlf_features


FIELDNAMES = ["window_id"] + VLF_FIELDNAMES


def build_vlf_window_features(
    *,
    training_windows_csv: Path,
    metadata_root: Path,
    out_path: Path,
) -> list[dict[str, str]]:
    metadata_paths = sorted(metadata_root.glob("**/*.metadata.json"))
    with training_windows_csv.open(newline="", encoding="utf-8") as handle:
        training_rows = list(csv.DictReader(handle))

    rows = []
    for window in training_rows:
        row = summarize_vlf_features(
            metadata_paths=metadata_paths,
            window_start_utc=window["window_start_utc"],
            window_end_utc=window["window_end_utc"],
        )
        rows.append({"window_id": window["window_id"], **row})

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    return rows

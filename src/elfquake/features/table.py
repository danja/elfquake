"""Manifest-driven multimodal feature table builder."""

from __future__ import annotations

import csv
import tempfile
from pathlib import Path

from elfquake.features.multimodal_smoke import FIELDNAMES, build_multimodal_smoke_row


MANIFEST_FIELDNAMES = [
    "region_id",
    "window_start_utc",
    "window_end_utc",
    "target_end_utc",
    "target_magnitude_min",
    "events_csv",
    "vlf_metadata_paths",
    "astronomy_metadata_paths",
]


def build_multimodal_table_from_manifest(*, manifest_path: Path, out_path: Path) -> list[dict[str, str]]:
    with manifest_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    built_rows = []
    with tempfile.TemporaryDirectory() as directory:
        temp_root = Path(directory)
        for index, row in enumerate(rows):
            temp_out = temp_root / f"row_{index}.csv"
            built_rows.append(
                build_multimodal_smoke_row(
                    events_csv=Path(row["events_csv"]),
                    vlf_metadata_paths=_path_list(row.get("vlf_metadata_paths", "")),
                    astronomy_metadata_paths=_path_list(row.get("astronomy_metadata_paths", "")),
                    region_id=row["region_id"],
                    window_start_utc=row["window_start_utc"],
                    window_end_utc=row["window_end_utc"],
                    target_end_utc=row["target_end_utc"],
                    target_magnitude_min=row.get("target_magnitude_min") or "3.0",
                    out_path=temp_out,
                )
            )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES, lineterminator="\n")
        writer.writeheader()
        writer.writerows(built_rows)
    return built_rows


def write_multimodal_manifest_template(out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=MANIFEST_FIELDNAMES, lineterminator="\n")
        writer.writeheader()


def _path_list(value: str) -> list[Path]:
    return [Path(item) for item in value.split(";") if item]

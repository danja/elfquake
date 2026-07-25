"""Adapt native Japan CDF feature rows to the shared sequence manifest."""

from __future__ import annotations

import csv
import json
from pathlib import Path

from elfquake.models.sequence_materializer import materialize_sequence_dataset


def materialize_japan_cdf_sequence(
    *, feature_csvs: list[Path], out_dir: Path, dataset_id: str = "japan_moshiri"
) -> dict[str, object]:
    """Combine processed CDF rows and materialize one research-only VLF sequence."""
    if not feature_csvs:
        raise ValueError("at least one Japan feature CSV is required")
    rows: list[dict[str, str]] = []
    channel_fields: list[str] = []
    for path in feature_csvs:
        with path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            source_fields = list(reader.fieldnames or [])
            if "time_utc" not in source_fields:
                raise ValueError(f"Japan feature CSV has no time_utc field: {path}")
            source_channels = [
                field for field in source_fields
                if field not in {"time_utc", "research_use_only"}
                and _numeric_column(path, field)
            ]
            for field in source_channels:
                prefixed = f"japan_{field}"
                if prefixed not in channel_fields:
                    channel_fields.append(prefixed)
            for source_row in reader:
                row = {"time_utc": source_row.get("time_utc", "")}
                for field in source_channels:
                    row[f"japan_{field}"] = source_row.get(field, "")
                rows.append(row)

    rows.sort(key=lambda row: row["time_utc"])
    out_dir.mkdir(parents=True, exist_ok=True)
    normalized = out_dir / "normalized_features.csv"
    fields = ["time_utc", *channel_fields]
    with normalized.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)

    manifest = materialize_sequence_dataset(
        input_csv=normalized,
        out_dir=out_dir,
        time_field="time_utc",
        entity_field=None,
        modality="japan_vlf_cdf",
        dataset_id=dataset_id,
    )
    manifest.update({
        "dataset_id": dataset_id,
        "research_use_only": 1,
        "source_files": [str(path) for path in feature_csvs],
        "source_format": "ISEE native CDF-derived features",
    })
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def _numeric_column(path: Path, field: str) -> bool:
    with path.open(newline="", encoding="utf-8") as handle:
        values = [row.get(field, "") for row in csv.DictReader(handle)]
    present = [value for value in values if value != ""]
    if not present:
        return False
    try:
        for value in present:
            float(value)
    except ValueError:
        return False
    return True

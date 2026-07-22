"""Decode a VLF CDF file into provenance metadata and scalar samples."""

from __future__ import annotations

import csv
import json
from pathlib import Path


def normalize_vlf_cdf(*, input_path: Path, samples_out: Path, metadata_out: Path) -> dict[str, object]:
    try:
        import cdflib
    except ImportError as error:  # pragma: no cover - depends on optional environment
        raise RuntimeError("cdflib is required to decode VLF CDF files; install requirements-optional.txt") from error

    cdf = cdflib.CDF(str(input_path))
    info = cdf.cdf_info()
    variables = list(_info_value(info, "zVariables", [])) + list(_info_value(info, "rVariables", []))
    epoch_name = _find_epoch_variable(cdf, variables)
    epochs = cdf.varget(epoch_name) if epoch_name else []
    times = cdflib.cdfepoch.to_datetime(epochs) if epoch_name else []
    scalar_values: dict[str, list[object]] = {}
    scalar_variables: list[str] = []
    variable_metadata: list[dict[str, object]] = []
    row_count = len(times)
    for name in variables:
        values = cdf.varget(name)
        shape = list(getattr(values, "shape", ()))
        descriptor = {
            "name": name,
            "shape": shape,
            "dtype": str(getattr(values, "dtype", type(values).__name__)),
            "attributes": _json_safe(cdf.varattsget(name)),
        }
        variable_metadata.append(descriptor)
        if name == epoch_name:
            continue
        if len(shape) != 1 or len(values) != row_count:
            continue
        scalar_variables.append(name)
        scalar_values[name] = values.tolist() if hasattr(values, "tolist") else list(values)

    samples_out.parent.mkdir(parents=True, exist_ok=True)
    with samples_out.open("w", newline="", encoding="utf-8") as handle:
        fields = ["time_utc", *scalar_variables]
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for index, timestamp in enumerate(times):
            row = {"time_utc": _format_time(timestamp)}
            for name in scalar_variables:
                row[name] = _format_value(scalar_values[name][index])
            writer.writerow(row)

    metadata = {
        "schema": "elfquake.vlf_cdf_metadata.v2",
        "input": str(input_path),
        "epoch_variable": epoch_name or "",
        "variables": variables,
        "scalar_variables": scalar_variables,
        "non_scalar_variables": [
            item["name"] for item in variable_metadata
            if item["name"] != epoch_name and item["name"] not in scalar_variables
        ],
        "variable_metadata": variable_metadata,
        "global_attributes": _json_safe(cdf.globalattsget()),
        "row_count": row_count,
        "warning": "CDF contents remain source-specific; verify units, station metadata, and scientific-use terms before modelling. Non-scalar spectra are described here but not flattened into CSV.",
    }
    metadata_out.parent.mkdir(parents=True, exist_ok=True)
    metadata_out.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return metadata


def _find_epoch_variable(cdf: object, variables: list[str]) -> str | None:
    for name in variables:
        lowered = name.lower()
        if lowered in {"epoch", "time", "timestamp", "datetime"} or "epoch" in lowered:
            return name
    return None


def _info_value(info: object, name: str, default: object) -> object:
    """Support both cdflib dict-style and dataclass-style CDFInfo objects."""
    if isinstance(info, dict):
        return info.get(name, default)
    return getattr(info, name, default)


def _format_time(value: object) -> str:
    if hasattr(value, "isoformat"):
        formatted = value.isoformat().replace("+00:00", "Z")
    else:
        formatted = str(value)
    return formatted if formatted.endswith("Z") else formatted + "Z"


def _format_value(value: object) -> str:
    if value is None:
        return ""
    return str(value)


def _json_safe(value: object) -> object:
    """Convert CDF attributes into JSON without losing the raw CDF file."""
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if hasattr(value, "tolist"):
        return _json_safe(value.tolist())
    return str(value)

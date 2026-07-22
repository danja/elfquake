"""Feature extraction for native ISEE VLF CDF spectrograms."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np


def extract_vlf_cdf_features(
    *, input_path: Path, out_path: Path, bands: int = 8
) -> dict[str, object]:
    """Convert native CDF spectra into time-windowable, provenance-rich rows."""
    try:
        import cdflib
    except ImportError as error:  # pragma: no cover - optional dependency
        raise RuntimeError("cdflib is required; install requirements-optional.txt") from error
    if bands < 1:
        raise ValueError("bands must be positive")

    cdf = cdflib.CDF(str(input_path))
    info = cdf.cdf_info()
    variables = list(getattr(info, "zVariables", [])) + list(getattr(info, "rVariables", []))
    epoch_name = _find_variable(variables, "epoch")
    frequency_name = _find_variable(variables, "freq")
    if not epoch_name or not frequency_name:
        raise ValueError("CDF must contain epoch and frequency support variables")

    epochs = cdf.varget(epoch_name)
    times = cdflib.cdfepoch.to_datetime(epochs)
    frequencies = np.asarray(cdf.varget(frequency_name), dtype=float).reshape(-1)
    channels = _find_spectrograms(cdf, variables, epoch_name, frequency_name, len(times), len(frequencies))
    if not channels:
        raise ValueError("CDF contains no time-by-frequency spectrogram variables")

    edges = np.geomspace(max(1.0, float(frequencies.min())), float(frequencies.max()), bands + 1)
    masks = [(frequencies >= edges[index]) & (frequencies < edges[index + 1]) for index in range(bands)]
    fields = ["time_utc"]
    for channel in channels:
        fields.extend(f"{channel}_band_{index}_log10_power" for index in range(bands))
        fields.extend((f"{channel}_active_fraction", f"{channel}_valid_fraction"))
    fields.append("research_use_only")

    rows: list[dict[str, str]] = []
    arrays = {name: _clean_array(cdf.varget(name), cdf.varattsget(name)) for name in channels}
    for row_index, timestamp in enumerate(times):
        row = {"time_utc": _format_time(timestamp), "research_use_only": "1"}
        for channel in channels:
            spectrum = arrays[channel][row_index]
            valid = np.isfinite(spectrum)
            row[f"{channel}_valid_fraction"] = f"{float(valid.mean()):.6f}"
            row[f"{channel}_active_fraction"] = f"{float((valid & (spectrum > 0)).mean()):.6f}"
            for band_index, mask in enumerate(masks):
                values = spectrum[mask]
                values = values[np.isfinite(values) & (values >= 0)]
                power = float(values.mean()) if values.size else 0.0
                row[f"{channel}_band_{band_index}_log10_power"] = f"{np.log10(max(power, 1e-30)):.8f}"
        rows.append(row)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)

    metadata = {
        "schema": "elfquake.vlf_cdf_features.v1",
        "input": str(input_path),
        "output": str(out_path),
        "epoch_variable": epoch_name,
        "frequency_variable": frequency_name,
        "frequency_min_hz": float(frequencies.min()),
        "frequency_max_hz": float(frequencies.max()),
        "band_edges_hz": [float(value) for value in edges],
        "spectrogram_variables": channels,
        "row_count": len(rows),
        "research_use_only": True,
        "note": "Derived features are for scientific research only; retain the source CDF and archive caveats.",
    }
    out_path.with_suffix(out_path.suffix + ".metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return metadata


def _find_spectrograms(cdf: object, variables: list[str], epoch: str, frequency: str, rows: int, columns: int) -> list[str]:
    names = []
    for name in variables:
        if name in {epoch, frequency}:
            continue
        values = cdf.varget(name)
        if getattr(values, "shape", ()) == (rows, columns):
            names.append(name)
    return names


def _find_variable(variables: list[str], token: str) -> str | None:
    for name in variables:
        if token in name.lower():
            return name
    return None


def _clean_array(values: object, attributes: dict[str, object]) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    fill = attributes.get("FILLVAL")
    if fill is not None:
        try:
            array[array == float(fill)] = np.nan
        except (TypeError, ValueError):
            pass
    return array


def _format_time(value: object) -> str:
    formatted = value.isoformat().replace("+00:00", "Z") if hasattr(value, "isoformat") else str(value)
    return formatted if formatted.endswith("Z") else formatted + "Z"

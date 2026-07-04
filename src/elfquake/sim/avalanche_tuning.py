"""Tune direct avalanche event extraction against real seismic event shape."""

from __future__ import annotations

import csv
from pathlib import Path

from elfquake.features.signal_shape_compare import compare_signal_shapes, event_energy_series
from elfquake.sim.synthetic_events import build_avalanche_signal_event_list


FIELDNAMES = [
    "rank",
    "min_signal_quantile",
    "local_max_window",
    "event_count",
    "normalized_distance",
    "synthetic_nonzero_ratio",
    "real_nonzero_ratio",
    "synthetic_psd_slope",
    "real_psd_slope",
    "synthetic_burst_run_count",
    "real_burst_run_count",
    "events_file",
]


def tune_avalanche_event_extraction(
    *,
    real_events_csv: Path,
    avalanche_csv: Path,
    out_path: Path,
    grid_width: int,
    grid_height: int,
    avalanche_activity_csv: Path | None = None,
    quantiles: list[float] | None = None,
    local_max_windows: list[int] | None = None,
    work_dir: Path | None = None,
    start_time_utc: str = "2026-01-01T00:00:00Z",
    step_seconds: int = 60,
    event_bin_seconds: int = 3600,
) -> list[dict[str, str]]:
    quantiles = quantiles or [0.9, 0.95, 0.975, 0.99]
    local_max_windows = local_max_windows or [5, 15, 30, 60]
    _validate_grid(quantiles=quantiles, local_max_windows=local_max_windows)
    work_dir = work_dir or out_path.with_suffix("")
    work_dir.mkdir(parents=True, exist_ok=True)

    candidates = []
    series = [
        event_energy_series(
            series_id="real_seismic_events",
            events_csv=real_events_csv,
            bin_seconds=event_bin_seconds,
        )
    ]
    for quantile in quantiles:
        for window in local_max_windows:
            stem = f"avalanche_events_q{_slug_float(quantile)}_w{window}"
            events_file = work_dir / f"{stem}.csv"
            event_rows = build_avalanche_signal_event_list(
                avalanche_csv=avalanche_csv,
                avalanche_activity_csv=avalanche_activity_csv,
                out_path=events_file,
                grid_width=grid_width,
                grid_height=grid_height,
                start_time_utc=start_time_utc,
                step_seconds=step_seconds,
                min_signal_quantile=quantile,
                local_max_window=window,
            )
            series_id = f"synthetic_q{_slug_float(quantile)}_w{window}"
            candidates.append(
                {
                    "series_id": series_id,
                    "quantile": quantile,
                    "window": window,
                    "event_count": len(event_rows),
                    "events_file": events_file,
                }
            )
            series.append(
                event_energy_series(
                    series_id=series_id,
                    events_csv=events_file,
                    bin_seconds=event_bin_seconds,
                )
            )

    series_rows, pair_rows = compare_signal_shapes(
        series=series,
        series_out=out_path.with_suffix(".series.csv"),
        pairs_out=out_path.with_suffix(".pairs.csv"),
    )
    rows = [
        _summary_row(candidate=candidate, series_rows=series_rows, pair_rows=pair_rows)
        for candidate in candidates
    ]
    rows.sort(key=lambda row: (float(row["normalized_distance"]), int(row["event_count"])))
    ranked_rows = [
        {**row, "rank": str(index)}
        for index, row in enumerate(rows, start=1)
    ]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES, lineterminator="\n")
        writer.writeheader()
        writer.writerows(ranked_rows)
    return ranked_rows


def _summary_row(
    *,
    candidate: dict[str, object],
    series_rows: list[dict[str, str]],
    pair_rows: list[dict[str, str]],
) -> dict[str, str]:
    by_id = {row["series_id"]: row for row in series_rows}
    real = by_id["real_seismic_events"]
    series_id = str(candidate["series_id"])
    synthetic = by_id[series_id]
    pair = _real_pair(series_id=series_id, pair_rows=pair_rows)
    return {
        "rank": "",
        "min_signal_quantile": f"{float(candidate['quantile']):.6g}",
        "local_max_window": str(candidate["window"]),
        "event_count": str(candidate["event_count"]),
        "normalized_distance": pair["normalized_distance"],
        "synthetic_nonzero_ratio": synthetic["nonzero_ratio"],
        "real_nonzero_ratio": real["nonzero_ratio"],
        "synthetic_psd_slope": synthetic["psd_slope"],
        "real_psd_slope": real["psd_slope"],
        "synthetic_burst_run_count": synthetic["burst_run_count"],
        "real_burst_run_count": real["burst_run_count"],
        "events_file": str(candidate["events_file"]),
    }


def _real_pair(*, series_id: str, pair_rows: list[dict[str, str]]) -> dict[str, str]:
    for row in pair_rows:
        if row["left_series_id"] == "real_seismic_events" and row["right_series_id"] == series_id:
            return row
        if row["right_series_id"] == "real_seismic_events" and row["left_series_id"] == series_id:
            return row
    raise ValueError(f"missing real pair for {series_id}")


def _validate_grid(*, quantiles: list[float], local_max_windows: list[int]) -> None:
    if not quantiles:
        raise ValueError("at least one quantile is required")
    if not local_max_windows:
        raise ValueError("at least one local_max_window is required")
    if any(not 0 <= value < 1 for value in quantiles):
        raise ValueError("quantiles must be at least 0 and below 1")
    if any(value < 0 for value in local_max_windows):
        raise ValueError("local_max_windows must be non-negative")


def _slug_float(value: float) -> str:
    return f"{value:.6g}".replace(".", "p")

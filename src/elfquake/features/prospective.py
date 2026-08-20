"""Prospective rows anchored on live VLF captures."""

from __future__ import annotations

import csv
import json
import tempfile
from datetime import timedelta
from pathlib import Path

from elfquake.features.astro_windows import FIELDNAMES as ASTRO_FIELDNAMES
from elfquake.features.astro_windows import (
    SpaceWeatherSeries,
    build_astro_window_features,
    load_space_weather_series,
)
from elfquake.features.common import format_utc, parse_utc
from elfquake.features.vlf import FIELDNAMES as VLF_FIELDNAMES
from elfquake.features.vlf import summarize_vlf_features


DEFAULT_SPACE_WEATHER_ROOT = Path("data/derived/astronomy")
KP_CSV_NAME = "gfz_kp_ap.csv"
DST_CSV_GLOB = "kyoto_dst_*.csv"
F107_CSV_NAME = "f107_daily.csv"


FIELDNAMES = [
    "window_id",
    "region_id",
    "window_start_utc",
    "window_end_utc",
    "target_start_utc",
    "target_end_utc",
    "target_magnitude_min",
    "seismic_event_count",
    "seismic_max_magnitude",
    "target_event_count",
    "target_occurred",
    "target_status",
    "seismic_source_file",
] + [
    field for field in VLF_FIELDNAMES if field not in {"window_start_utc", "window_end_utc"}
] + [
    field for field in ASTRO_FIELDNAMES if field not in {"window_start_utc", "window_end_utc"}
]


def build_prospective_vlf_windows(
    *,
    events_csv: Path,
    vlf_metadata_root: Path,
    space_weather_root: Path | None = None,
    region_id: str,
    out_path: Path,
    lookback_hours: int = 24,
    horizon_days: int = 7,
    min_anchor_gap_seconds: int = 60,
    target_magnitude_min: str = "3.0",
) -> list[dict[str, str]]:
    if lookback_hours < 1 or horizon_days < 1:
        raise ValueError("lookback_hours and horizon_days must be at least 1")
    if min_anchor_gap_seconds < 0:
        raise ValueError("min_anchor_gap_seconds must be non-negative")

    vlf_metadata_paths = sorted(vlf_metadata_root.glob("**/*.metadata.json"))
    space_weather = _load_space_weather(space_weather_root)
    events = _read_events(events_csv, region_id)
    rows = []
    for window_end_utc in _vlf_anchor_times(vlf_metadata_paths, min_gap_seconds=min_anchor_gap_seconds):
        window_end = parse_utc(window_end_utc)
        window_start = window_end - timedelta(hours=lookback_hours)
        target_end = window_end + timedelta(days=horizon_days)
        feature_events = [
            event
            for event in events
            if window_start <= parse_utc(event["event_time_utc"]) < window_end
        ]
        vlf_row = summarize_vlf_features(
            metadata_paths=vlf_metadata_paths,
            window_start_utc=format_utc(window_start),
            window_end_utc=window_end_utc,
            include_window_end=True,
        )
        astro_row = build_astro_window_features(
            window_start_utc=format_utc(window_start),
            window_end_utc=window_end_utc,
            series=space_weather,
        )
        rows.append(
            {
                "window_id": _window_id(region_id, format_utc(window_start), window_end_utc),
                "region_id": region_id,
                "window_start_utc": format_utc(window_start),
                "window_end_utc": window_end_utc,
                "target_start_utc": window_end_utc,
                "target_end_utc": format_utc(target_end),
                "target_magnitude_min": target_magnitude_min,
                "seismic_event_count": str(len(feature_events)),
                "seismic_max_magnitude": _max_magnitude(feature_events),
                "target_event_count": "",
                "target_occurred": "",
                "target_status": "unlabeled_pending_future_events",
                "seismic_source_file": str(events_csv),
                **_without_window_times(vlf_row),
                **_without_window_times(astro_row),
            }
        )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    return rows


def update_prospective_vlf_table(
    *,
    table_path: Path,
    events_csv: Path,
    vlf_metadata_root: Path,
    space_weather_root: Path | None = None,
    region_id: str,
    out_path: Path,
    lookback_hours: int = 24,
    horizon_days: int = 7,
    min_anchor_gap_seconds: int = 60,
    target_magnitude_min: str = "3.0",
) -> dict[str, object]:
    existing_rows = _read_existing_table(table_path)
    existing_by_id = {row["window_id"]: row for row in existing_rows if row.get("window_id")}
    with tempfile.TemporaryDirectory() as directory:
        candidate_path = Path(directory) / "prospective.csv"
        candidate_rows = build_prospective_vlf_windows(
            events_csv=events_csv,
            vlf_metadata_root=vlf_metadata_root,
            space_weather_root=space_weather_root,
            region_id=region_id,
            out_path=candidate_path,
            lookback_hours=lookback_hours,
            horizon_days=horizon_days,
            min_anchor_gap_seconds=min_anchor_gap_seconds,
            target_magnitude_min=target_magnitude_min,
        )

    candidate_ids = {row["window_id"] for row in candidate_rows}
    new_rows = [row for row in candidate_rows if row["window_id"] not in existing_by_id]
    refreshed_rows = [row for row in candidate_rows if row["window_id"] in existing_by_id]
    retained_rows = [row for row in existing_rows if row.get("window_id") not in candidate_ids]
    merged_rows = candidate_rows + retained_rows
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES, lineterminator="\n")
        writer.writeheader()
        writer.writerows(_normalize_rows(merged_rows))
    return {
        "existing_rows": len(existing_rows),
        "candidate_rows": len(candidate_rows),
        "new_rows": len(new_rows),
        "refreshed_rows": len(refreshed_rows),
        "retained_rows": len(retained_rows),
        "total_rows": len(merged_rows),
    }


def _load_space_weather(space_weather_root: Path | None) -> SpaceWeatherSeries:
    """Load normalized Kp/ap, Dst and F10.7 tables, if they have been built.

    A missing root is not an error: the geomagnetic channels then report
    themselves missing per row rather than silently becoming constants.
    """
    root = space_weather_root or DEFAULT_SPACE_WEATHER_ROOT
    if not root.exists():
        return SpaceWeatherSeries.empty()
    kp_csv = root / KP_CSV_NAME
    f107_csv = root / F107_CSV_NAME
    return load_space_weather_series(
        kp_csv=kp_csv if kp_csv.exists() else None,
        dst_csv_paths=sorted(root.glob(DST_CSV_GLOB)),
        f107_csv=f107_csv if f107_csv.exists() else None,
    )


def _read_events(events_csv: Path, region_id: str) -> list[dict[str, str]]:
    with events_csv.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not region_id or region_id == "all_italy":
        return rows
    return [row for row in rows if row.get("italy_region", region_id) == region_id]


def _read_existing_table(table_path: Path) -> list[dict[str, str]]:
    if not table_path.exists():
        return []
    with table_path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _normalize_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    return [
        {field: row.get(field, "") for field in FIELDNAMES}
        for row in rows
    ]


def _vlf_anchor_times(metadata_paths: list[Path], *, min_gap_seconds: int) -> list[str]:
    values = []
    for path in metadata_paths:
        metadata = json.loads(path.read_text(encoding="utf-8"))
        captured_at = metadata.get("captured_at_utc", "")
        if captured_at:
            values.append(captured_at)
    anchors = []
    for captured_at in sorted(set(values)):
        if not anchors:
            anchors.append(captured_at)
            continue
        previous = parse_utc(anchors[-1])
        current = parse_utc(captured_at)
        if (current - previous).total_seconds() <= min_gap_seconds:
            anchors[-1] = captured_at
        else:
            anchors.append(captured_at)
    return anchors


def _max_magnitude(events: list[dict[str, str]]) -> str:
    values = [float(row["magnitude"]) for row in events if row.get("magnitude")]
    if not values:
        return ""
    return f"{max(values):g}"


def _without_window_times(row: dict[str, str]) -> dict[str, str]:
    return {
        field: value
        for field, value in row.items()
        if field not in {"window_start_utc", "window_end_utc"}
    }


def _window_id(region_id: str, window_start_utc: str, window_end_utc: str) -> str:
    return (
        f"{region_id}_{window_start_utc}_{window_end_utc}"
        .replace(":", "")
        .replace("-", "")
        .replace("T", "t")
        .replace("Z", "z")
    )

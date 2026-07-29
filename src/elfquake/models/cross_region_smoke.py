"""Artifacts for the synthetic -> Japan -> Italy Transformer smoke test."""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

from elfquake.models.real_transfer_trial import _inside_italy


def prepare_italy_target(*, source: Path, out: Path, events: Path | None = None) -> dict[str, object]:
    """Keep only mature, fixed-cell Italy labels and preserve the existing split."""
    with source.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fields = list(reader.fieldnames or [])
        rows = [
            row for row in reader
            if row.get("dataset_id") == "italy_all"
            and row.get("target_occurred") in {"0", "1"}
            and row.get("target_status") == "labeled"
        ]
    if not rows:
        raise ValueError("no mature Italy target rows found")
    rows.sort(key=lambda row: (row.get("window_start_utc", ""), row.get("target_cell_id", "")))
    if events:
        _attach_event_targets(rows, events)
        fields.extend(field for field in (
            "target_event_time_utc", "target_event_latitude",
            "target_event_longitude", "target_event_magnitude", "target_event_slots_json",
        ) if field not in fields)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    return {
        "schema": "elfquake.cross_region_italy_target.v1",
        "source": str(source),
        "output": str(out),
        "row_count": len(rows),
        "positive_count": sum(row.get("target_occurred") == "1" for row in rows),
        "split_counts": {
            value: sum(row.get("model_split") == value for row in rows)
            for value in ("train", "test")
        },
        "note": "The latest mature chronological Italy partition is held out; pending labels are excluded.",
    }


def write_map_inputs(*, report: Path, target: Path, real_events: Path, out_dir: Path) -> dict[str, object]:
    """Convert held-out cell probabilities into the map renderer's stable CSV schema."""
    payload = json.loads(report.read_text(encoding="utf-8"))
    runs = [run for run in payload.get("runs", []) if run.get("regime") == "synthetic_then_japan_then_italy"]
    if not runs:
        raise ValueError("cross-region regime is missing from the Transformer report")
    run = sorted(runs, key=lambda item: int(item.get("seed", 0)))[0]
    model = run["downstream_models"]["italy_multimodal"]["fine_tune"]
    evaluations = model["evaluations"]
    evaluation_name = "trained_input" if "trained_input" in evaluations else sorted(evaluations)[0]
    evaluation = evaluations[evaluation_name]
    probabilities = [float(value) for value in evaluation["probabilities"]]
    threshold = float(model["calibrated_threshold"])
    with target.open(newline="", encoding="utf-8") as handle:
        test_rows = [row for row in csv.DictReader(handle) if row.get("model_split") == "test"]
    if len(test_rows) != len(probabilities):
        raise ValueError(f"test row/probability mismatch: {len(test_rows)} != {len(probabilities)}")
    starts = [parse_time(row["window_start_utc"]) for row in test_rows]
    holdout_start = min(starts)
    horizon_end = holdout_start + timedelta(days=7)
    candidates = []
    coordinate_predictions = evaluation.get("coordinate_predictions", [])
    for index, (row, probability) in enumerate(zip(test_rows, probabilities)):
        if probability < threshold:
            continue
        coordinates = coordinate_predictions[index] if index < len(coordinate_predictions) else []
        candidates.extend((probability, slot) for slot in coordinates)
    with target.open(newline="", encoding="utf-8") as handle:
        all_rows = list(csv.DictReader(handle))
    train_rows = [row for row in all_rows if row.get("model_split") == "train"]
    window_totals = {}
    for row in train_rows:
        key = row.get("target_start_utc", "")
        window_totals.setdefault(key, 0.0)
        window_totals[key] += float(row.get("target_event_count", "0") or 0.0)
    budget = max(1, round(sum(window_totals.values()) / max(1, len(window_totals))))
    unique_candidates = {}
    for probability, coordinates in candidates:
        if not coordinates:
            continue
        key = (round(float(coordinates[0]), 3), round(float(coordinates[1]), 3))
        if key not in unique_candidates or probability > unique_candidates[key][0]:
            unique_candidates[key] = (probability, coordinates)
    selected = []
    for probability, coordinates in sorted(unique_candidates.values(), key=lambda item: item[0], reverse=True)[:budget]:
        latitude, longitude = _project_to_italy(
            float(coordinates[0]) * 50.0,
            float(coordinates[1]) * 20.0,
        )
        selected.append({
            "event_time_utc": holdout_start.isoformat().replace("+00:00", "Z"),
            "latitude": latitude,
            "longitude": longitude,
            "magnitude": max(2.5, float(coordinates[2]) * 5.0),
            "predicted_probability": probability,
            "source": "transformer_coordinate_head",
        })
    actual = []
    with real_events.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            try:
                event_time = parse_time(row["event_time_utc"])
                magnitude = float(row["magnitude"])
                latitude = float(row["latitude"])
                longitude = float(row["longitude"])
            except (KeyError, TypeError, ValueError):
                continue
            if holdout_start <= event_time < horizon_end and magnitude >= 2.5:
                actual.append({"latitude": latitude, "longitude": longitude, "magnitude": magnitude})
    out_dir.mkdir(parents=True, exist_ok=True)
    _write_rows(out_dir / "heldout_week_predictions.csv", selected)
    _write_rows(out_dir / "heldout_week_actual_events.csv", actual)
    metadata = {
        "schema": "elfquake.cross_region_generative_smoke_map.v1",
        "warning": "Engineering smoke test only; generated cells are not an earthquake forecast.",
        "research_use_only": "Japan ISEE data were used only for scientific research and self-supervised continuation.",
        "holdout_start_utc": holdout_start.isoformat().replace("+00:00", "Z"),
        "holdout_end_utc": horizon_end.isoformat().replace("+00:00", "Z"),
        "threshold": threshold,
        "evaluation": evaluation_name,
        "actual_count": len(actual),
        "predicted_event_count": len(selected),
        "prediction_budget": budget,
        "report": str(report),
    }
    (out_dir / "map_inputs.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return metadata


def parse_time(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(timezone.utc)


def _attach_event_targets(rows: list[dict[str, str]], events_path: Path) -> None:
    events = []
    with events_path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            try:
                events.append({
                    "event_time_utc": row["event_time_utc"],
                    "time": parse_time(row["event_time_utc"]),
                    "latitude": float(row["latitude"]),
                    "longitude": float(row["longitude"]),
                    "magnitude": float(row["magnitude"]),
                })
            except (KeyError, TypeError, ValueError):
                continue
    for row in rows:
        row.update({
            "target_event_time_utc": "", "target_event_latitude": "",
            "target_event_longitude": "", "target_event_magnitude": "",
            "target_event_slots_json": "[]",
        })
        try:
            start = parse_time(row["target_start_utc"])
            end = parse_time(row["target_end_utc"])
            lat = float(row["target_cell_latitude"])
            lon = float(row["target_cell_longitude"])
            half = float(row.get("target_cell_degrees", "1.5")) / 2.0
            minimum = float(row.get("target_magnitude_min", "2.5"))
        except (KeyError, TypeError, ValueError):
            continue
        matches = [
            event for event in events
            if start <= event["time"] < end
            and event["magnitude"] >= minimum
            and abs(event["latitude"] - lat) <= half
            and abs(event["longitude"] - lon) <= half
        ]
        if matches:
            matches.sort(key=lambda item: (item["time"], -item["magnitude"]))
            row["target_event_slots_json"] = json.dumps([
                {key: event[key] for key in ("event_time_utc", "latitude", "longitude", "magnitude")}
                for event in matches
            ], separators=(",", ":"))
            event = max(matches, key=lambda item: item["magnitude"])
            row["target_event_time_utc"] = event["event_time_utc"]
            row["target_event_latitude"] = f"{event['latitude']:.6f}"
            row["target_event_longitude"] = f"{event['longitude']:.6f}"
            row["target_event_magnitude"] = f"{event['magnitude']:.4f}"


def _project_to_italy(latitude: float, longitude: float) -> tuple[float, float]:
    """Pull an unconstrained coordinate back into the offline Italy outline."""
    latitude = max(35.0, min(47.8, latitude))
    longitude = max(5.5, min(19.5, longitude))
    if _inside_italy(longitude, latitude):
        return latitude, longitude
    # A straight-line projection to a central Italy anchor keeps the result
    # continuous while respecting the map domain; it is not a location label.
    anchor_latitude, anchor_longitude = 42.5, 12.5
    for fraction in (0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0):
        candidate = (
            anchor_latitude + fraction * (latitude - anchor_latitude),
            anchor_longitude + fraction * (longitude - anchor_longitude),
        )
        if _inside_italy(candidate[1], candidate[0]):
            return candidate
    return anchor_latitude, anchor_longitude


def _write_rows(path: Path, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = list(rows)
    fields = ("event_time_utc", "latitude", "longitude", "magnitude", "predicted_probability", "source")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows({field: row.get(field, "") for field in fields} for row in rows)


def main() -> int:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare = subparsers.add_parser("prepare-target")
    prepare.add_argument("--source", type=Path, required=True)
    prepare.add_argument("--out", type=Path, required=True)
    prepare.add_argument("--events", type=Path)
    render = subparsers.add_parser("map-inputs")
    render.add_argument("--report", type=Path, required=True)
    render.add_argument("--target", type=Path, required=True)
    render.add_argument("--real-events", type=Path, required=True)
    render.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "prepare-target":
        print(json.dumps(prepare_italy_target(source=args.source, out=args.out, events=args.events), sort_keys=True))
    else:
        print(json.dumps(write_map_inputs(report=args.report, target=args.target, real_events=args.real_events, out_dir=args.out_dir), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

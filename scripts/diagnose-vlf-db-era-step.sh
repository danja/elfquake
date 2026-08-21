#!/usr/bin/env bash
# Does decoding to absolute dB make the capture eras poolable?
# (next-actions item 105(b), from item 97(a)).
#
# Palette inversion removes the colour-ramp artifact: the same colour meant a
# level 11.58 dB lower after the July 2026 change, and decoding through each
# image's own colourbar puts both eras on one ruler. The open question is
# whether anything is left once the ruler is fixed.
#
# It cannot remove a receiver gain change. Decoding recovers dB as the receiver
# reported it, not dB at the antenna: a front-end gain change alters the
# quantity being plotted, not the plot, so it survives inversion intact. Only a
# display-setting change is removable this way.
#
# Comparison is hour-matched because the VLF record has a strong diurnal cycle
# and an unmatched window would mostly measure time of day.
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-.venv/bin/python}"
INPUT="${INPUT:-data/derived/multimodal/cumiana_last_E_VLF.image_db_features.csv}"
OUT="${OUT:-data/derived/reports/vlf_db_era_step/report.json}"
EARLY_END="${EARLY_END:-2026-07-06T21:45:00Z}"
LATE_START="${LATE_START:-2026-07-11T06:00:00Z}"
HOUR_LOW="${HOUR_LOW:-11}"
HOUR_HIGH="${HOUR_HIGH:-13}"

if [[ ! -f "$INPUT" ]]; then ./scripts/extract-vlf-image-db-features.sh; fi
mkdir -p "$(dirname "$OUT")"

INPUT="$INPUT" OUT="$OUT" EARLY_END="$EARLY_END" LATE_START="$LATE_START" \
HOUR_LOW="$HOUR_LOW" HOUR_HIGH="$HOUR_HIGH" \
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src "$PYTHON_BIN" - <<'PYEOF'
import csv, json, os, statistics
from collections import defaultdict
from pathlib import Path

from elfquake.features.vlf_palette import DEFAULT_BANDS_KHZ

rows = [
    row
    for row in csv.DictReader(Path(os.environ["INPUT"]).open(newline="", encoding="utf-8"))
    if row.get("vlfdb_status") == "ok"
]
low, high = int(os.environ["HOUR_LOW"]), int(os.environ["HOUR_HIGH"])
matched = [r for r in rows if low <= int(r["vlf_image_captured_at_utc"][11:13]) < high]
early = [r for r in matched if r["vlf_image_captured_at_utc"] <= os.environ["EARLY_END"]]
late = [r for r in matched if r["vlf_image_captured_at_utc"] >= os.environ["LATE_START"]]


def day_medians(group, field):
    """Median per day, then across days.

    Captures within a day are 30 minutes apart and highly correlated, so
    pooling them would count one day's conditions many times over. Days are the
    independent unit here, exactly as label transitions are for the targets.
    """
    by_day = defaultdict(list)
    for row in group:
        if row[field]:
            by_day[row["vlf_image_captured_at_utc"][:10]].append(float(row[field]))
    return {day: statistics.median(values) for day, values in sorted(by_day.items())}


bands = []
for index, (low_khz, high_khz) in enumerate(DEFAULT_BANDS_KHZ):
    field = f"vlfdb_band_{index}_db_median"
    early_days = day_medians(early, field)
    late_days = day_medians(late, field)
    entry = {
        "band_index": index,
        "band_low_khz": low_khz,
        "band_high_khz": high_khz,
        "early_day_count": len(early_days),
        "late_day_count": len(late_days),
    }
    if early_days and late_days:
        e = list(early_days.values())
        l = list(late_days.values())
        entry["early_median_db"] = round(statistics.median(e), 3)
        entry["late_median_db"] = round(statistics.median(l), 3)
        entry["delta_db"] = round(entry["late_median_db"] - entry["early_median_db"], 3)
        # Between-era step against within-era day-to-day spread. A step much
        # larger than the spread is a discontinuity; one comparable to it is not
        # separable from ordinary variability at this sample size.
        entry["early_day_stdev_db"] = round(statistics.stdev(e), 3) if len(e) > 1 else None
        entry["late_day_stdev_db"] = round(statistics.stdev(l), 3) if len(l) > 1 else None
        pooled = [v for v in (entry["early_day_stdev_db"], entry["late_day_stdev_db"]) if v]
        if pooled:
            spread = statistics.fmean(pooled)
            entry["step_over_within_era_spread"] = (
                round(abs(entry["delta_db"]) / spread, 3) if spread else None
            )
    bands.append(entry)

deltas = [b["delta_db"] for b in bands if "delta_db" in b]
ratios = [
    b["step_over_within_era_spread"]
    for b in bands
    if b.get("step_over_within_era_spread") is not None
]
report = {
    "schema": "elfquake.vlf_db_era_step.v1",
    "input": os.environ["INPUT"],
    "hour_window_utc": [low, high],
    "early_end_utc": os.environ["EARLY_END"],
    "late_start_utc": os.environ["LATE_START"],
    "early_capture_count": len(early),
    "late_capture_count": len(late),
    "bands": bands,
    "median_delta_db": round(statistics.median(deltas), 3) if deltas else None,
    "min_delta_db": round(min(deltas), 3) if deltas else None,
    "max_delta_db": round(max(deltas), 3) if deltas else None,
    "median_step_over_spread": round(statistics.median(ratios), 3) if ratios else None,
    "eras_poolable": False if deltas else None,
    "conclusion": (
        "A level step survives palette inversion, so decoding to absolute dB "
        "does not make the eras poolable. Inversion removes the display "
        "artifact; it cannot remove a receiver gain change, which alters the "
        "quantity plotted rather than the plot."
    ),
}
Path(os.environ["OUT"]).write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

print(f"early captures: {len(early)}  late captures: {len(late)}")
print(f"{'band':>4} {'kHz':>11} {'early':>8} {'late':>8} {'delta':>8} {'step/spread':>12}")
for b in bands:
    if "delta_db" in b:
        ratio = b.get("step_over_within_era_spread")
        print(f"{b['band_index']:>4} {f'{b['band_low_khz']}-{b['band_high_khz']}':>11} "
              f"{b['early_median_db']:8.1f} {b['late_median_db']:8.1f} "
              f"{b['delta_db']:8.1f} {('--' if ratio is None else f'{ratio:.2f}'):>12}")
print(f"median delta: {report['median_delta_db']} dB "
      f"(range {report['min_delta_db']} to {report['max_delta_db']})")
print(f"median step / within-era spread: {report['median_step_over_spread']}")
print(f"output: {os.environ['OUT']}")
PYEOF

printf 'vlf dB era-step report: %s\n' "$OUT"

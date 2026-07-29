#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-.venv/bin/python}"
ROOT="${ROOT:-data/raw/vlf/cumiana/captures}"
OUT="${OUT:-data/derived/reports/cumiana_vlf_capture_gaps.json}"

PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src "$PYTHON_BIN" - "$ROOT" "$OUT" <<'PY'
import json
import os
import sys
from datetime import datetime
from pathlib import Path

root = Path(sys.argv[1])
out = Path(sys.argv[2])
times = []
for path in sorted(root.rglob("*.metadata.json")):
    try:
        record = json.loads(path.read_text(encoding="utf-8"))
        if record.get("endpoint_id") == "last_E_VLF":
            times.append(datetime.fromisoformat(record["captured_at_utc"].replace("Z", "+00:00")))
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
        continue
times.sort()
minimum_gap = float(os.environ.get("MIN_GAP_HOURS", "1"))
gaps = []
for before, after in zip(times, times[1:]):
    hours = (after - before).total_seconds() / 3600
    if hours > minimum_gap:
        gaps.append({"before_utc": before.isoformat(), "after_utc": after.isoformat(), "gap_hours": round(hours, 3)})
report = {
    "schema": "elfquake.vlf_capture_gap_report.v1",
    "endpoint_id": "last_E_VLF",
    "metadata_root": str(root),
    "capture_count": len(times),
    "first_capture_utc": times[0].isoformat() if times else None,
    "last_capture_utc": times[-1].isoformat() if times else None,
    "gap_count": len(gaps),
    "gaps": gaps,
}
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
print(f"captures: {len(times)}")
print(f"gaps: {len(gaps)}")
print(f"output: {out}")
PY

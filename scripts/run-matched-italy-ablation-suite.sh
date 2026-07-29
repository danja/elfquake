#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-.venv/bin/python}"
REAL_EVENTS="${REAL_EVENTS:-data/derived/ingv/events_italy_all_available.combined.normalized.csv}"
OUT_ROOT="${OUT_ROOT:-data/derived/models/matched_italy_ablation_suite}"
SYNTHETIC_ARGS=(
  --synthetic-events data/derived/sim/mountain_256x256_seed40_20000.avalanche_events.csv
  --synthetic-events data/derived/sim/mountain_256x256_seed41_20000.avalanche_events.csv
  --synthetic-events data/derived/sim/mountain_256x256_seed42_20000.avalanche_events.csv
)

mkdir -p "$OUT_ROOT"
for seed in ${SEEDS:-42 99 123}; do
  for ablation in full seismic_history vlf_mask seismic_vlf_mask; do
    for pretraining in pretrained real_only; do
      extra=()
      [[ "$pretraining" == "real_only" ]] && extra+=(--no-synthetic-pretraining)
      PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src "$PYTHON_BIN" -m elfquake.cli run-real-transfer-trial \
        --real-events "$REAL_EVENTS" "${SYNTHETIC_ARGS[@]}" \
        --out-dir "$OUT_ROOT/seed${seed}_${ablation}_${pretraining}" \
        --magnitude-threshold 2.5 --horizon-days 7 --cell-degrees 1.5 \
        --train-fraction 0.8 --pretrain-epochs "${PRETRAIN_EPOCHS:-30}" \
        --finetune-epochs "${FINETUNE_EPOCHS:-80}" --seed "$seed" \
        --feature-ablation "$ablation" "${extra[@]}"
    done
  done
done

PYTHONDONTWRITEBYTECODE=1 "$PYTHON_BIN" - "$OUT_ROOT" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
rows = []
for report_path in sorted(root.glob("seed*/report.json")):
    report = json.loads(report_path.read_text(encoding="utf-8"))
    evaluation = report["evaluation"]
    rows.append({
        "run": report_path.parent.name,
        "seed": report_path.parent.name.split("_", 1)[0].removeprefix("seed"),
        "feature_ablation": report["model"]["feature_ablation"],
        "synthetic_pretraining": report["model"]["transfer"],
        "balanced_accuracy": evaluation["balanced_accuracy"],
        "precision": evaluation["precision"],
        "recall": evaluation["recall"],
        "threshold": evaluation["threshold"],
    })
(root / "summary.json").write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
print(f"matched ablation rows: {len(rows)}")
print(f"summary: {root / 'summary.json'}")
PY

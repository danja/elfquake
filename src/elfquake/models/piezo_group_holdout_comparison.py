"""Compare piezo/VLF leave-one-episode-out experiment reports."""

from __future__ import annotations

import json
import math
from pathlib import Path


def compare_piezo_group_holdouts(
    *,
    report_paths: list[Path],
    out_path: Path,
    balanced_accuracy_floor: float = 0.60,
    recall_floor: float = 0.40,
    fold_pass_fraction: float = 0.80,
) -> dict[str, object]:
    rows = [_comparison_row(path, recall_floor=recall_floor) for path in report_paths]
    required_fraction = min(1.0, max(0.0, fold_pass_fraction))
    for row in rows:
        required_folds = math.ceil(row["ensemble_fold_count"] * required_fraction)
        row["required_recall_pass_folds"] = required_folds
        row["passes_mean_accuracy"] = row["ensemble_balanced_accuracy_mean"] >= balanced_accuracy_floor
        row["passes_recall_stability"] = row["ensemble_recall_pass_folds"] >= required_folds
        row["passes_synthetic_stability_gate"] = (
            row["passes_mean_accuracy"] and row["passes_recall_stability"]
        )
    ranked = sorted(
        rows,
        key=lambda row: (
            row["passes_synthetic_stability_gate"],
            row["ensemble_recall_pass_folds"] / row["ensemble_fold_count"],
            row["ensemble_balanced_accuracy_mean"],
        ),
        reverse=True,
    )
    report = {
        "schema": "elfquake.piezo_group_holdout_comparison.v1",
        "status": "compared",
        "criteria": {
            "balanced_accuracy_floor": balanced_accuracy_floor,
            "recall_floor": recall_floor,
            "fold_pass_fraction": required_fraction,
        },
        "experiment_count": len(ranked),
        "passing_experiment_count": sum(row["passes_synthetic_stability_gate"] for row in ranked),
        "best_experiment": ranked[0]["experiment"] if ranked else "",
        "experiments": ranked,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def _comparison_row(path: Path, *, recall_floor: float) -> dict[str, object]:
    report = json.loads(path.read_text(encoding="utf-8"))
    if report.get("schema") != "elfquake.piezo_group_holdout.v1":
        raise ValueError(f"unsupported group holdout report: {path}")
    ensemble = report["summary"]["fixed_seed_ensemble"]
    folds = ensemble["folds"]
    recall_passes = sum(
        fold["calibrated_metrics"]["positive_recall"] >= recall_floor
        and fold["calibrated_metrics"]["negative_recall"] >= recall_floor
        for fold in folds
    )
    return {
        "experiment": path.parent.name,
        "report": str(path),
        "target_csv": report["target_csv"],
        "lookback_steps": report["lookback_steps"],
        "patch_steps": report["patch_steps"],
        "entity_aggregation_profile": report.get("entity_aggregation_profile", "mean"),
        "individual_balanced_accuracy_mean": report["summary"]["balanced_accuracy"]["mean"],
        "ensemble_balanced_accuracy_mean": ensemble["balanced_accuracy"]["mean"],
        "ensemble_balanced_accuracy_min": ensemble["balanced_accuracy"]["min"],
        "ensemble_balanced_accuracy_max": ensemble["balanced_accuracy"]["max"],
        "ensemble_fold_count": len(folds),
        "ensemble_recall_pass_folds": recall_passes,
    }

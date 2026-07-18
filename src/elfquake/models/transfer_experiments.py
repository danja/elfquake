"""Matched transfer ablations, rolling validation, and grid selection."""

from __future__ import annotations

import json
import random
from pathlib import Path

from elfquake.models.real_transfer_trial import (
    Event,
    _best_threshold,
    _cells,
    _fit,
    _historical_cell_rates,
    _metrics,
    _model,
    _predict,
    _read_events,
    _read_synthetic_corpus,
    _seed,
    _torch,
    _weekly_samples,
)


def run_transfer_experiment_suite(
    *,
    real_events_csv: Path,
    synthetic_event_csvs: list[Path],
    out_path: Path,
    magnitude_threshold: float = 2.5,
    horizon_days: int = 7,
    cell_degrees: float = 1.5,
    train_fraction: float = 0.8,
    epochs: int = 50,
    pretrain_epochs: int = 30,
    seed: int = 42,
    precision_recall_floor: float = 0.5,
) -> dict[str, object]:
    """Run experiments 1--3 using one fixed final real holdout."""
    real = _read_events(real_events_csv)
    synthetic = _read_synthetic_corpus(synthetic_event_csvs)
    cells = _cells(cell_degrees)
    samples = _weekly_samples(real, cells, magnitude_threshold, horizon_days, cell_degrees)
    synthetic_samples = _weekly_samples(synthetic, cells, magnitude_threshold, horizon_days, cell_degrees)
    weeks = sorted({sample.week_start for sample in samples})
    split = int(len(weeks) * train_fraction)
    train_weeks, test_weeks = set(weeks[:split]), set(weeks[split:])
    train = [sample for sample in samples if sample.week_start in train_weeks]
    test = [sample for sample in samples if sample.week_start in test_weeks]
    torch = _torch()
    ablations = {
        "historical_spatial_rate": _rate_result(train, test, precision_recall_floor=precision_recall_floor),
        "real_random_init_seismic": _model_result(torch, train, test, synthetic_samples=None, epochs=epochs, pretrain_epochs=0, seed=seed, precision_recall_floor=precision_recall_floor),
        "synthetic_pretrained_transfer": _model_result(torch, train, test, synthetic_samples=synthetic_samples, epochs=epochs, pretrain_epochs=pretrain_epochs, seed=seed, precision_recall_floor=precision_recall_floor),
    }
    rolling = _rolling_results(torch, samples, synthetic_samples, weeks, epochs, pretrain_epochs, seed, precision_recall_floor)
    grid = _grid_selection(torch, real, synthetic, train_weeks, test, magnitude_threshold, horizon_days, epochs, pretrain_epochs, seed, precision_recall_floor)
    report = {
        "schema": "elfquake.real_transfer_experiment_suite.v1",
        "status": "evaluated",
        "warning": "Engineering experiments only; no earthquake prediction capability is demonstrated.",
        "data": {"real_events_csv": str(real_events_csv), "synthetic_event_csvs": [str(path) for path in synthetic_event_csvs], "real_event_count": len(real), "synthetic_event_count": len(synthetic), "synthetic_sample_count": len(synthetic_samples), "real_train_weeks": len(train_weeks), "real_test_weeks": len(test_weeks), "real_test_start": str(min(test_weeks)), "real_test_end": str(max(test_weeks))},
        "experiment_1_matched_ablation": ablations,
        "experiment_2_rolling_origin": rolling,
        "experiment_3_train_only_grid_selection": grid,
        "precision_calibration": {"recall_floor": precision_recall_floor, "selection_rule": "maximize training precision subject to recall floor and at least one positive prediction"},
        "experiment_4_synthetic_corpus": {"status": "input_corpus_reported_separately", "required": "more independent warmed episodes must be generated before synthetic transfer is treated as informative", "current_synthetic_sample_count": len(synthetic_samples)},
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    return report


def _model_result(torch, train, test, *, synthetic_samples, epochs, pretrain_epochs, seed, precision_recall_floor=0.5):
    if len({sample.label for sample in train}) < 2 or len({sample.label for sample in test}) < 2:
        return {"status": "insufficient_class_variation"}
    model = _model(torch, len(train[0].values))
    transfer = bool(synthetic_samples and len({sample.label for sample in synthetic_samples}) == 2 and pretrain_epochs)
    if transfer:
        _fit(torch, model, synthetic_samples, pretrain_epochs, seed)
    _fit(torch, model, train, epochs, seed + 1)
    train_probabilities = _predict(torch, model, train)
    test_probabilities = _predict(torch, model, test)
    train_labels = [sample.label for sample in train]
    test_labels = [sample.label for sample in test]
    threshold = _best_threshold(train_probabilities, train_labels)
    precision_threshold = _best_precision_threshold(train_probabilities, train_labels, min_recall=precision_recall_floor)
    rate_threshold = _rate_matched_threshold(train_probabilities, train_labels)
    return {
        "status": "evaluated",
        "training": "synthetic_pretrained" if transfer else "real_random_initialization",
        "threshold": round(threshold, 6),
        **_metrics(test_probabilities, test_labels, threshold),
        "precision_calibrated_threshold": round(precision_threshold, 6),
        "precision_calibrated_metrics": _metrics(test_probabilities, test_labels, precision_threshold),
        "precision_calibration_rule": f"maximize training precision subject to recall >= {precision_recall_floor:g} and at least one positive prediction",
        "rate_calibrated_threshold": round(rate_threshold, 6),
        "rate_calibrated_metrics": _metrics(test_probabilities, test_labels, rate_threshold),
        "rate_calibration_rule": "select the training score quantile matching the training positive rate",
    }


def _rate_result(train, test, *, precision_recall_floor=0.5):
    train_probabilities = _historical_cell_rates(train, train)
    test_probabilities = _historical_cell_rates(train, test)
    train_labels = [sample.label for sample in train]
    test_labels = [sample.label for sample in test]
    threshold = _best_threshold(train_probabilities, train_labels)
    precision_threshold = _best_precision_threshold(train_probabilities, train_labels, min_recall=precision_recall_floor)
    rate_threshold = _rate_matched_threshold(train_probabilities, train_labels)
    return {
        "status": "evaluated",
        "threshold": round(threshold, 6),
        **_metrics(test_probabilities, test_labels, threshold),
        "precision_calibrated_threshold": round(precision_threshold, 6),
        "precision_calibrated_metrics": _metrics(test_probabilities, test_labels, precision_threshold),
        "precision_calibration_rule": f"maximize training precision subject to recall >= {precision_recall_floor:g} and at least one positive prediction",
        "rate_calibrated_threshold": round(rate_threshold, 6),
        "rate_calibrated_metrics": _metrics(test_probabilities, test_labels, rate_threshold),
        "rate_calibration_rule": "select the training score quantile matching the training positive rate",
    }


def _rolling_results(torch, samples, synthetic_samples, weeks, epochs, pretrain_epochs, seed, precision_recall_floor):
    results = []
    for index, fraction in enumerate((0.60, 0.70, 0.80, 0.85)):
        split = max(1, int(len(weeks) * fraction))
        train_weeks, test_weeks = set(weeks[:split]), set(weeks[split:])
        train = [sample for sample in samples if sample.week_start in train_weeks]
        test = [sample for sample in samples if sample.week_start in test_weeks]
        result = _model_result(torch, train, test, synthetic_samples=synthetic_samples, epochs=epochs, pretrain_epochs=pretrain_epochs, seed=seed + index, precision_recall_floor=precision_recall_floor)
        result.update({"train_fraction": fraction, "train_weeks": len(train_weeks), "test_weeks": len(test_weeks)})
        results.append(result)
    evaluated = [item for item in results if item.get("status") == "evaluated"]
    return {"folds": results, "mean_balanced_accuracy": round(sum(float(item["balanced_accuracy"]) for item in evaluated) / len(evaluated), 6) if evaluated else None, "worst_balanced_accuracy": min((float(item["balanced_accuracy"]) for item in evaluated), default=None)}


def _grid_selection(torch, real, synthetic, train_weeks, final_test, default_threshold, horizon_days, epochs, pretrain_epochs, seed, precision_recall_floor):
    candidates = []
    train_end = max(train_weeks)
    # The last quarter of the training period selects configuration; the final
    # 20 percent is never used for selection.
    all_weeks = sorted({sample.week_start for sample in _weekly_samples(real, _cells(1.5), default_threshold, horizon_days, 1.5)})
    selection_cut = all_weeks[max(1, int(len(all_weeks) * 0.60))]
    for threshold in (2.5, 2.7, 3.0):
        for size in (1.0, 1.5, 2.0):
            cells = _cells(size)
            samples = _weekly_samples(real, cells, threshold, horizon_days, size)
            weeks = sorted({sample.week_start for sample in samples})
            train = [sample for sample in samples if sample.week_start < selection_cut]
            validation = [sample for sample in samples if selection_cut <= sample.week_start <= train_end]
            if len({sample.label for sample in train}) < 2 or len({sample.label for sample in validation}) < 2:
                continue
            result = _model_result(torch, train, validation, synthetic_samples=None, epochs=max(15, epochs // 2), pretrain_epochs=0, seed=seed, precision_recall_floor=precision_recall_floor)
            candidates.append({"magnitude_threshold": threshold, "cell_degrees": size, "validation_balanced_accuracy": result.get("balanced_accuracy"), "validation_precision": result.get("precision")})
    candidates.sort(key=lambda item: (float(item["validation_balanced_accuracy"]), float(item["validation_precision"])), reverse=True)
    selected = candidates[0] if candidates else {}
    final_result = {"status": "not_selected"}
    if selected:
        selected_threshold = float(selected["magnitude_threshold"])
        selected_size = float(selected["cell_degrees"])
        selected_cells = _cells(selected_size)
        selected_samples = _weekly_samples(real, selected_cells, selected_threshold, horizon_days, selected_size)
        final_start = min(sample.week_start for sample in final_test)
        selected_train = [sample for sample in selected_samples if sample.week_start < final_start]
        selected_test = [sample for sample in selected_samples if sample.week_start >= final_start]
        selected_synthetic = _weekly_samples(synthetic, selected_cells, selected_threshold, horizon_days, selected_size)
        final_result = _model_result(torch, selected_train, selected_test, synthetic_samples=selected_synthetic, epochs=epochs, pretrain_epochs=pretrain_epochs, seed=seed + 100, precision_recall_floor=precision_recall_floor)
        final_result.update({"magnitude_threshold": selected_threshold, "cell_degrees": selected_size, "test_start": str(final_start)})
    return {"selection_rule": "highest validation balanced accuracy, tie-break precision; final holdout excluded", "candidates": candidates, "selected": selected, "final_holdout_evaluation": final_result}


def _best_precision_threshold(probabilities: list[float], labels: list[int], *, min_recall: float) -> float:
    candidates = [index / 100 for index in range(5, 96)]
    eligible = []
    for threshold in candidates:
        metrics = _metrics(probabilities, labels, threshold)
        if metrics["recall"] >= min_recall and metrics["confusion"]["true_positive"] + metrics["confusion"]["false_positive"] > 0:
            eligible.append((float(metrics["precision"]), float(metrics["balanced_accuracy"]), threshold))
    if not eligible:
        return _best_threshold(probabilities, labels)
    return max(eligible)[2]


def _rate_matched_threshold(probabilities: list[float], labels: list[int]) -> float:
    if not probabilities:
        return 0.5
    positive_rate = sum(labels) / len(labels)
    positive_count = max(1, min(len(probabilities), int(round(positive_rate * len(probabilities)))))
    ordered = sorted(probabilities, reverse=True)
    return ordered[positive_count - 1]

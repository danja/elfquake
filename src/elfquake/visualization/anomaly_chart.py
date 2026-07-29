"""Rendering for causal self-supervised VLF anomaly scores."""

from __future__ import annotations

import csv
import math
import os
from datetime import datetime
from pathlib import Path


def render_anomaly_chart(
    *, scores_csv: Path, out_path: Path, alert_threshold: float = 0.8,
    max_gap_hours: float = 1.0, title: str = "Cumiana VLF self-supervised anomaly score",
) -> dict[str, object]:
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
    try:
        import matplotlib
        matplotlib.use("Agg")
        from matplotlib import pyplot as plt
    except ImportError as error:  # pragma: no cover - depends on optional visualization package
        raise RuntimeError("matplotlib is required for anomaly chart rendering") from error

    with scores_csv.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"no anomaly rows found in {scores_csv}")
    times = [datetime.fromisoformat(row["window_end_utc"].replace("Z", "+00:00")) for row in rows]
    scores = [float(row["anomaly_score"]) for row in rows]
    peak = max(range(len(scores)), key=scores.__getitem__)
    plotted = scores[:]
    gap_count = 0
    for index in range(1, len(times)):
        if (times[index] - times[index - 1]).total_seconds() > max_gap_hours * 3600:
            plotted[index - 1] = math.nan
            plotted[index] = math.nan
            gap_count += 1

    figure, axis = plt.subplots(figsize=(10, 4.8), dpi=160)
    axis.plot(times, plotted, color="#2457a6", linewidth=1.4, label="Self-supervised anomaly score")
    axis.scatter(times, scores, color="#2457a6", s=8, alpha=0.55, zorder=3)
    axis.axhline(alert_threshold, color="#c0392b", linestyle="--", linewidth=1.1,
                 label=f"Exploratory alert threshold ({alert_threshold:g})")
    axis.scatter([times[peak]], [scores[peak]], color="#c0392b", edgecolor="white", linewidth=0.8, zorder=4)
    axis.annotate(
        f"Peak {scores[peak]:.3f}\n{times[peak].strftime('%Y-%m-%d %H:%M UTC')}",
        (times[peak], scores[peak]), xytext=(-90, -38), textcoords="offset points", fontsize=8,
        arrowprops={"arrowstyle": "->", "color": "#555555"},
    )
    axis.set_ylim(0, 1.02)
    axis.set_ylabel("Anomaly score")
    axis.set_xlabel(f"VLF window end; lines break across gaps > {max_gap_hours:g} hour")
    axis.set_title(title)
    axis.grid(axis="y", alpha=0.25)
    axis.legend(loc="upper left", frameon=False, fontsize=8)
    figure.autofmt_xdate()
    figure.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(out_path, format="png")
    plt.close(figure)
    return {
        "chart_file": str(out_path),
        "row_count": len(rows),
        "peak_score": round(scores[peak], 6),
        "peak_time_utc": times[peak].isoformat(),
        "gap_count": gap_count,
    }

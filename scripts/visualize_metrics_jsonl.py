#!/usr/bin/env python
"""
Visualize LC-BGPlaceNet metrics.jsonl logs.

使用示例:
    python scripts/visualize_metrics_jsonl.py

    python scripts/visualize_metrics_jsonl.py \
        --metrics outputs/lc_bgplacenet_stage1/metrics.jsonl \
        --output outputs/lc_bgplacenet_stage1/metrics_plot.png
"""

from __future__ import annotations

import argparse
import json
import math
import os
from collections import defaultdict
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", os.fspath(PROJECT_ROOT / "outputs" / ".matplotlib"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


DEFAULT_METRICS_PATH = PROJECT_ROOT / "outputs" / "lc_bgplacenet_stage1" / "metrics.jsonl"


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Visualize LC-BGPlaceNet JSONL metrics.")
    parser.add_argument("--metrics", type=Path, default=DEFAULT_METRICS_PATH, help="Path to metrics.jsonl.")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output image path. Defaults to metrics_plot.png next to the input log.",
    )
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    """Load non-empty JSON lines from a metrics log."""
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at line {line_no}: {exc}") from exc
            if isinstance(row, dict):
                rows.append(row)
    if not rows:
        raise ValueError(f"No metric rows found in {path}")
    return rows


def is_number(value: Any) -> bool:
    """Return True for finite int/float values, excluding bool."""
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


def choose_x_key(rows: list[dict[str, Any]]) -> str:
    """Use the most informative training progress field as x-axis."""
    for key in ("step", "iteration", "iter", "global_step", "epoch"):
        if any(is_number(row.get(key)) for row in rows):
            return key
    return "index"


def collect_metric_keys(rows: list[dict[str, Any]], x_key: str) -> list[str]:
    """Collect numeric metric fields while skipping axis and bookkeeping fields."""
    excluded = {x_key, "epoch", "iteration", "iter", "global_step"}
    keys = {
        key
        for row in rows
        for key, value in row.items()
        if key not in excluded and is_number(value)
    }
    priority = ("loss", "loss_src", "loss_src_center", "loss_src_size", "loss_src_iou", "source_iou")
    return sorted(keys, key=lambda key: (priority.index(key) if key in priority else len(priority), key))


def plot_metrics(rows: list[dict[str, Any]], output_path: Path) -> None:
    """Plot each numeric metric as one subplot and group curves by split."""
    x_key = choose_x_key(rows)
    metric_keys = collect_metric_keys(rows, x_key)
    if not metric_keys:
        raise ValueError("No numeric metric fields found for plotting.")

    grouped: dict[str, list[tuple[int, dict[str, Any]]]] = defaultdict(list)
    for index, row in enumerate(rows):
        split = str(row.get("split", "all"))
        grouped[split].append((index, row))

    ncols = 2 if len(metric_keys) > 1 else 1
    nrows = math.ceil(len(metric_keys) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(6.5 * ncols, 3.4 * nrows), squeeze=False)
    axes_flat = [axis for row_axes in axes for axis in row_axes]

    for axis, metric_key in zip(axes_flat, metric_keys):
        for split, split_rows in grouped.items():
            points = []
            for index, row in split_rows:
                value = row.get(metric_key)
                if not is_number(value):
                    continue
                x_value = row.get(x_key, index)
                if not is_number(x_value):
                    x_value = index
                points.append((float(x_value), float(value)))
            if not points:
                continue
            # 训练日志可能来自断点续训追加，按横轴排序避免折线回连。
            points.sort(key=lambda point: point[0])
            xs, ys = zip(*points)
            axis.plot(xs, ys, marker="o", markersize=2.8, linewidth=1.5, label=split)

        axis.set_title(metric_key)
        axis.set_xlabel(x_key)
        axis.grid(True, linestyle="--", linewidth=0.5, alpha=0.45)
        axis.legend()

    for axis in axes_flat[len(metric_keys) :]:
        axis.set_visible(False)

    fig.suptitle(f"Metrics: {output_path.parent.name}", fontsize=14)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def main() -> None:
    """CLI entry point."""
    args = parse_args()
    metrics_path = args.metrics.resolve()
    output_path = args.output.resolve() if args.output else metrics_path.with_name("metrics_plot.png")

    rows = load_jsonl(metrics_path)
    plot_metrics(rows, output_path)
    print(f"Saved metrics plot to {output_path}")


if __name__ == "__main__":
    main()

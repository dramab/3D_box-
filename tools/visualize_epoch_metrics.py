#!/usr/bin/env python
"""
Visualize metrics.jsonl by epoch.

使用示例:
    conda run -n spatial python tools/visualize_epoch_metrics.py \
        --input outputs/lc_bgplacenet_stage1/metrics.jsonl

    conda run -n spatial python tools/visualize_epoch_metrics.py \
        --input outputs/lc_bgplacenet_stage1/metrics.jsonl \
        --output outputs/lc_bgplacenet_stage1/epoch_metrics.png
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

os.environ.setdefault("MPLCONFIGDIR", os.fspath(PROJECT_ROOT / "outputs" / ".matplotlib"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


DEFAULT_INPUT = PROJECT_ROOT / "outputs" / "lc_bgplacenet_stage1" / "metrics.jsonl"
EXCLUDED_NUMERIC_KEYS = {"epoch", "step"}
SPLIT_ORDER = {"train": 0, "val": 1}


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Visualize metrics.jsonl as epoch-level curves.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="Path to metrics.jsonl.")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output image path. Defaults to <input_dir>/epoch_metrics.png.",
    )
    return parser.parse_args()


def is_number(value: Any) -> bool:
    """Return True for finite int/float values, excluding bool."""
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def load_epoch_metrics(path: Path) -> tuple[dict[str, dict[int, dict[str, float]]], list[str]]:
    """Load jsonl metrics and average numeric values for each split and epoch."""
    grouped: dict[tuple[str, int], dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    metric_names: set[str] = set()

    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at line {line_no}: {exc}") from exc

            if "epoch" not in row:
                raise ValueError(f"Missing 'epoch' at line {line_no}")

            split = str(row.get("split", "metrics"))
            epoch = int(row["epoch"])
            values = {
                key: float(value)
                for key, value in row.items()
                if key not in EXCLUDED_NUMERIC_KEYS and is_number(value)
            }
            metric_names.update(values)
            for key, value in values.items():
                grouped[(split, epoch)][key].append(value)

    if not grouped:
        raise ValueError(f"No metric rows found in {path}")

    epoch_metrics: dict[str, dict[int, dict[str, float]]] = defaultdict(dict)
    for (split, epoch), metric_values in grouped.items():
        epoch_metrics[split][epoch] = {
            key: sum(values) / len(values) for key, values in metric_values.items()
        }

    return dict(epoch_metrics), sorted(metric_names)


def split_sort_key(split: str) -> tuple[int, str]:
    """Keep common train/val order before other split names."""
    return SPLIT_ORDER.get(split, len(SPLIT_ORDER)), split


def plot_epoch_metrics(
    epoch_metrics: dict[str, dict[int, dict[str, float]]],
    metric_names: list[str],
    output_path: Path,
) -> None:
    """Save epoch-level metric curves to an image."""
    columns = 3
    rows = math.ceil(len(metric_names) / columns)
    fig, axes = plt.subplots(rows, columns, figsize=(5.6 * columns, 3.4 * rows), squeeze=False)
    flat_axes = axes.ravel()

    for ax, metric in zip(flat_axes, metric_names):
        has_data = False
        for split in sorted(epoch_metrics, key=split_sort_key):
            epochs = sorted(epoch_metrics[split])
            y_values = [epoch_metrics[split][epoch].get(metric) for epoch in epochs]
            pairs = [(epoch, value) for epoch, value in zip(epochs, y_values) if value is not None]
            if not pairs:
                continue
            x, y = zip(*pairs)
            ax.plot(x, y, marker="o", linewidth=1.6, markersize=3.5, label=split)
            has_data = True

        ax.set_title(metric)
        ax.set_xlabel("epoch")
        ax.ticklabel_format(axis="y", style="plain", useOffset=False)
        ax.grid(True, alpha=0.3)
        if has_data:
            ax.legend()

    for ax in flat_axes[len(metric_names) :]:
        ax.axis("off")

    fig.suptitle("LC-BGPlaceNet Stage 1 Epoch Metrics", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def main() -> None:
    """Load metrics and save the visualization image."""
    args = parse_args()
    input_path = args.input.resolve()
    output_path = args.output.resolve() if args.output else input_path.with_name("epoch_metrics.png")

    epoch_metrics, metric_names = load_epoch_metrics(input_path)
    plot_epoch_metrics(epoch_metrics, metric_names, output_path)
    print(f"Saved epoch metrics visualization to {output_path}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""
Export LC-BGPlaceNet Stage 1 target support GT point-cloud visualizations.

This script does not load a model checkpoint. It reads the Stage 1 dataset,
builds the placement-specific support_label GT, and writes colored PLY files.

使用示例:
    conda run -n spatial python tools/export_lc_bgplacenet_stage1_support_gt.py \
        --config configs/lc_bgplacenet_stage1.yaml \
        --split test \
        --max-samples 32

    conda run -n spatial python tools/export_lc_bgplacenet_stage1_support_gt.py \
        --config configs/lc_bgplacenet_stage1.yaml \
        --split all \
        --sample-id dopose__test_table_000001__000001 \
        --object-id obj_0
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.annotation.free_bbox.io_utils import save_ply
from src.training.lc_bgplacenet_stage1 import (
    LCBGPlaceNetStage1Dataset,
    build_sources_from_config,
    build_stage1_index,
    load_config,
    normalize_stage1_split,
)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Export Stage 1 target support GT point-cloud PLY files.")
    parser.add_argument("--config", type=Path, required=True, help="Stage 1 YAML config path.")
    parser.add_argument(
        "--split",
        choices=("valid", "val", "train", "test", "all"),
        default="test",
        help="Dataset split to export. val is accepted as an alias of valid.",
    )
    parser.add_argument("--max-samples", type=int, default=32, help="Limit exported samples; 0 exports all matched samples.")
    parser.add_argument("--sample-id", default=None, help="Only export one canonical sample_id.")
    parser.add_argument("--object-id", default=None, help="Only export one source object_id.")
    parser.add_argument("--placement-sample-id", default=None, help="Only export one placement_sample_id.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory. Defaults to training.output_dir/support_gt_pointclouds.",
    )
    return parser.parse_args()


def _sanitize_filename(value: str) -> str:
    """Convert an identifier to a stable filename fragment."""
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", str(value)).strip("._")
    return safe or "item"


def _gt_colors(colors: np.ndarray, support_label: np.ndarray) -> np.ndarray:
    """Darken inactive voxels and highlight target support GT in orange-red."""
    base = np.rint(np.asarray(colors, dtype=np.float32) * 0.35 + 45.0).clip(0, 255).astype(np.uint8)
    support = np.asarray(support_label, dtype=np.float32) >= 0.5
    if np.any(support):
        base[support] = np.array([255, 70, 30], dtype=np.uint8)
    return base


def _select_items(cfg: dict[str, Any], args: argparse.Namespace) -> list[Any]:
    """Build and filter Stage 1 index items without loading a model."""
    sources = build_sources_from_config(cfg)
    items = build_stage1_index(sources)
    data_cfg = cfg["data"]

    if args.split != "all":
        split_name = normalize_stage1_split(args.split)
        valid_fraction = float(data_cfg.get("valid_fraction", data_cfg.get("val_fraction", 0.1)))
        split_dataset = LCBGPlaceNetStage1Dataset(
            sources=None,
            split=split_name,
            val_fraction=valid_fraction,
            seed=int(data_cfg.get("split_seed", 0)),
            support_align_threshold_cm=float(data_cfg["support_align_threshold_cm"]),
            support_radius_area_fraction=float(data_cfg.get("support_radius_area_fraction", 0.25)),
            voxel_size_cm=float(data_cfg.get("voxel_size_cm", 1.0)),
            items=items,
            split_dir=data_cfg.get("split_dir"),
        )
        items = list(split_dataset.items)

    items = [
        item
        for item in items
        if (args.sample_id is None or item.sample_id == args.sample_id)
        and (args.object_id is None or item.object_id == args.object_id)
        and (args.placement_sample_id is None or item.placement_sample_id == args.placement_sample_id)
    ]
    if args.max_samples is not None and int(args.max_samples) > 0:
        items = items[: int(args.max_samples)]
    if not items:
        raise ValueError("No Stage 1 samples matched the requested filters.")
    return items


def _build_dataset(cfg: dict[str, Any], items: list[Any]) -> LCBGPlaceNetStage1Dataset:
    """Wrap selected items in the dataset so GT label generation stays shared."""
    data_cfg = cfg["data"]
    dataset = LCBGPlaceNetStage1Dataset(
        sources=None,
        split="train",
        val_fraction=0.0,
        support_align_threshold_cm=float(data_cfg["support_align_threshold_cm"]),
        support_radius_area_fraction=float(data_cfg.get("support_radius_area_fraction", 0.25)),
        voxel_size_cm=float(data_cfg.get("voxel_size_cm", 1.0)),
        items=items[:1],
    )
    dataset.items = list(items)
    return dataset


def export_support_gt(args: argparse.Namespace) -> None:
    """Export GT support point clouds and metadata."""
    cfg = load_config(args.config)
    output_dir = args.output_dir or Path(cfg["training"]["output_dir"]) / "support_gt_pointclouds"
    pointcloud_dir = output_dir / "pointclouds"
    pointcloud_dir.mkdir(parents=True, exist_ok=True)

    items = _select_items(cfg, args)
    dataset = _build_dataset(cfg, items)
    output_split = "all" if args.split == "all" else normalize_stage1_split(args.split)
    records_path = output_dir / "records.jsonl"
    summary_path = output_dir / "summary.json"

    total_support = 0
    total_points = 0
    records = []
    with records_path.open("w", encoding="utf-8") as f:
        for idx in range(len(dataset)):
            sample = dataset[idx]
            support_label = np.asarray(sample["support_label"], dtype=np.float32)
            points = np.asarray(sample["points"], dtype=np.float32)
            colors = _gt_colors(np.asarray(sample["colors"], dtype=np.uint8), support_label)

            stem = "__".join(
                [
                    _sanitize_filename(str(sample["item_id"])),
                    _sanitize_filename(str(sample["sample_id"])),
                    _sanitize_filename(str(sample["object_id"])),
                    f"cluster_{int(sample['cluster_id']):03d}",
                    "support_gt",
                ]
            )
            ply_path = pointcloud_dir / f"{stem}.ply"
            save_ply(ply_path, points, colors)

            support_count = int(np.count_nonzero(support_label >= 0.5))
            point_count = int(len(points))
            total_support += support_count
            total_points += point_count
            record = {
                "item_id": str(sample["item_id"]),
                "source_name": str(sample["source_name"]),
                "sample_id": str(sample["sample_id"]),
                "object_id": str(sample["object_id"]),
                "placement_sample_id": str(sample["placement_sample_id"]),
                "cluster_id": int(sample["cluster_id"]),
                "instruction": str(sample["instruction"]),
                "pointcloud_ply": str(ply_path),
                "point_count": point_count,
                "gt_support_voxels": support_count,
                "gt_support_ratio": float(support_count / point_count) if point_count else 0.0,
                "support_align_coverage": float(sample["support_align_coverage"]),
            }
            records.append(record)
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            print(f"Saved {ply_path}")

    summary = {
        "split": output_split,
        "samples": len(records),
        "pointcloud_dir": str(pointcloud_dir),
        "records_jsonl": str(records_path),
        "gt_support_voxels_total": int(total_support),
        "point_count_total": int(total_points),
        "gt_support_ratio_mean": float(np.mean([row["gt_support_ratio"] for row in records])) if records else 0.0,
    }
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False))


def main() -> None:
    """CLI entrypoint."""
    args = parse_args()
    try:
        export_support_gt(args)
    except ValueError as exc:
        raise SystemExit(f"ERROR: {exc}") from exc


if __name__ == "__main__":
    main()

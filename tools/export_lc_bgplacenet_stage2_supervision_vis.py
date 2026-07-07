#!/usr/bin/env python
"""
Export LC-BGPlaceNet Stage 2 training supervision visualizations.

使用示例:
    python tools/export_lc_bgplacenet_stage2_supervision_vis.py \
        --config configs/lc_bgplacenet_stage2.yaml

    python tools/export_lc_bgplacenet_stage2_supervision_vis.py \
        --config configs/lc_bgplacenet_stage2.yaml \
        --max-samples 20
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image, ImageDraw

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

os.environ.setdefault("MPLCONFIGDIR", os.fspath(PROJECT_ROOT / "outputs" / ".matplotlib"))

from src.annotation.free_bbox.io_utils import load_ply, save_ply
from src.datasets.canonical import load_canonical_scene
from src.training.lc_bgplacenet_stage2 import (
    Stage2IndexItem,
    _is_heatmap_positive_color,
    _is_support_color,
    build_dense_heatmap_targets,
    build_sources_from_config,
    build_stage2_index,
    load_config,
    select_stage2_split_items,
    stage2_collate,
)
from tools.infer_lc_bgplacenet_stage2 import (
    _find_scene_object as find_scene_object,
    draw_world_corners,
    place_box_to_corners,
    source_box_to_oriented_corners,
)


SOURCE_COLOR = (230, 57, 70)
PLACE_COLOR = (29, 128, 91)
SUPPORT_COLOR = np.array([55, 120, 210], dtype=np.uint8)
BACKGROUND_COLOR = np.array([55, 55, 55], dtype=np.uint8)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Export Stage 2 train supervision visualizations.")
    parser.add_argument("--config", type=Path, required=True, help="Stage 2 YAML config path.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory. Defaults to training.output_dir/supervision_vis.",
    )
    parser.add_argument("--max-samples", type=int, default=None, help="Limit exported train samples.")
    parser.add_argument("--sample-id", default=None, help="Only export one sample_id.")
    parser.add_argument("--object-id", default=None, help="Only export one object_id; usually used with --sample-id.")
    parser.add_argument("--line-width", type=int, default=3, help="Projected 3D box line width.")
    return parser.parse_args()


def select_train_items(
    cfg: dict[str, Any],
    sample_id: str | None,
    object_id: str | None,
    max_samples: int | None,
) -> list[Stage2IndexItem]:
    """Select Stage 2 train items from the configured fixed split."""
    sources = build_sources_from_config(cfg)
    items = build_stage2_index(sources)
    items = select_stage2_split_items(items, cfg["data"]["split_dir"], "train")
    if sample_id is not None or object_id is not None:
        items = [
            item
            for item in items
            if (sample_id is None or item.sample_id == sample_id)
            and (object_id is None or item.object_id == object_id)
        ]
    if max_samples is not None:
        items = items[: int(max_samples)]
    if not items:
        raise ValueError("No Stage 2 train samples selected")
    return items


def build_supervision_row(item: Stage2IndexItem) -> dict[str, Any]:
    """Load one Stage 2 item in the same form used by stage2_collate."""
    points, colors = load_ply(item.voxel_point_cloud_path)
    support_points, support_colors = load_ply(item.support_mask_ply)
    heatmap_points, heatmap_colors = load_ply(item.direction_filtered_heatmap_ply)
    positive_points = heatmap_points[_is_heatmap_positive_color(heatmap_colors)]
    if len(points) == 0:
        raise ValueError(f"Invalid empty point data for {item.sample_id}")
    if len(positive_points) == 0:
        raise ValueError(f"No direction-filtered heatmap positives for {item.item_id}")
    return {
        "item_id": item.item_id,
        "source_name": item.source_name,
        "sample_id": item.sample_id,
        "object_id": item.object_id,
        "cluster_id": item.cluster_id,
        "instruction": item.instruction,
        "points": points.astype(np.float32),
        "colors": colors.astype(np.uint8),
        "support_points": support_points[_is_support_color(support_colors)].astype(np.float32),
        "heatmap_positive_points": positive_points.astype(np.float32),
        "source_box_gt": item.source_box_gt,
        "place_box_gt": item.place_box_gt,
    }


def save_gaussian_heatmap_ply(
    item: Stage2IndexItem,
    cfg: dict[str, Any],
    output_path: Path,
) -> dict[str, Any]:
    """Save the support-limited Gaussian heatmap target used by Stage 2 training."""
    data_cfg = cfg["data"]
    row = build_supervision_row(item)
    batch = stage2_collate([row], voxel_size_cm=float(data_cfg["voxel_size_cm"]))
    sigma = float(data_cfg.get("heatmap_sigma_voxels", 2.0)) * float(data_cfg["voxel_size_cm"])
    targets = build_dense_heatmap_targets(
        world_coords=batch["world_coords"],
        batch_indices=batch["batch_indices"],
        positive_points=batch["heatmap_positive_points"],
        positive_batch_indices=batch["heatmap_positive_batch_indices"],
        support_masks=batch["support_masks"],
        batch_size=int(batch["batch_size"]),
        sigma=sigma,
    ).cpu()

    points = batch["world_coords"].cpu().numpy()
    support = batch["support_masks"].cpu().numpy().astype(bool)
    values = targets.numpy()
    colors = colorize_gaussian_targets(values, support)
    save_ply(output_path, points, colors)
    return {
        "sigma_cm": float(sigma),
        "num_points": int(len(points)),
        "num_support_points": int(support.sum()),
        "target_max": float(values.max()) if len(values) else 0.0,
        "target_positive_points": int((values > 0.0).sum()),
    }


def colorize_gaussian_targets(values: np.ndarray, support: np.ndarray) -> np.ndarray:
    """Color active voxels by Gaussian target value, keeping non-support voxels gray."""
    target = np.asarray(values, dtype=np.float32)
    colors = np.repeat(BACKGROUND_COLOR[None, :], len(target), axis=0)
    colors[np.asarray(support, dtype=bool)] = SUPPORT_COLOR
    active = target > 0.0
    if np.any(active):
        norm = target[active] / max(float(target[active].max()), 1e-6)
        heat = np.zeros((int(active.sum()), 3), dtype=np.uint8)
        heat[:, 0] = 255
        heat[:, 1] = np.rint(220.0 * (1.0 - norm)).clip(0, 220).astype(np.uint8)
        heat[:, 2] = 30
        colors[active] = heat
    return colors


def save_rgb_box_visualization(
    item: Stage2IndexItem,
    output_path: Path,
    line_width: int,
) -> None:
    """Save source_box_gt and place_box_gt projected on the canonical RGB image."""
    sample_path = item.dataset_dir / "samples" / f"{item.sample_id}.json"
    scene = load_canonical_scene(sample_path, dataset_root=item.dataset_dir)
    image = Image.fromarray(scene.rgb).convert("RGB")
    draw = ImageDraw.Draw(image)
    obj = find_scene_object(scene, item.object_id)
    draw_world_corners(
        draw,
        source_box_to_oriented_corners(item.source_box_gt, np.asarray(obj.pose_world)),
        scene,
        SOURCE_COLOR,
        line_width,
        "source gt",
    )
    draw_world_corners(
        draw,
        place_box_to_corners(item.place_box_gt),
        scene,
        PLACE_COLOR,
        line_width,
        "place gt",
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)


def make_stem(item: Stage2IndexItem) -> str:
    """Build a stable filename stem for one Stage 2 item."""
    return f"{item.source_name}__{item.sample_id}__{item.object_id}__cluster_{item.cluster_id:03d}__label_{item.label_index:06d}"


def main() -> None:
    """Export Stage 2 train supervision visualizations."""
    args = parse_args()
    cfg = load_config(args.config)
    output_dir = args.output_dir or Path(cfg["training"]["output_dir"]) / "supervision_vis"
    rgb_dir = output_dir / "rgb_boxes"
    heatmap_dir = output_dir / "gaussian_heatmaps"
    output_dir.mkdir(parents=True, exist_ok=True)

    items = select_train_items(cfg, args.sample_id, args.object_id, args.max_samples)
    records = []
    for index, item in enumerate(items, start=1):
        stem = make_stem(item)
        rgb_path = rgb_dir / f"{stem}.png"
        heatmap_path = heatmap_dir / f"{stem}.ply"
        save_rgb_box_visualization(item, rgb_path, line_width=int(args.line_width))
        heatmap_stats = save_gaussian_heatmap_ply(item, cfg, heatmap_path)
        records.append(
            {
                "item_id": item.item_id,
                "source_name": item.source_name,
                "sample_id": item.sample_id,
                "object_id": item.object_id,
                "cluster_id": item.cluster_id,
                "label_index": item.label_index,
                "instruction": item.instruction,
                "rgb_box_png": os.fspath(rgb_path),
                "gaussian_heatmap_ply": os.fspath(heatmap_path),
                "direction_filtered_heatmap_ply": os.fspath(item.direction_filtered_heatmap_ply),
                "support_mask_ply": os.fspath(item.support_mask_ply),
                "source_box_gt": np.asarray(item.source_box_gt, dtype=np.float32).tolist(),
                "place_box_gt": np.asarray(item.place_box_gt, dtype=np.float32).tolist(),
                "heatmap": heatmap_stats,
            }
        )
        if index % 100 == 0 or index == len(items):
            print(f"[stage2-supervision-vis] exported {index}/{len(items)}")

    with (output_dir / "index.json").open("w", encoding="utf-8") as f:
        json.dump(records, f, indent=2, ensure_ascii=False)
    print(f"Wrote {len(records)} Stage 2 train supervision visualizations to {output_dir.resolve()}")


if __name__ == "__main__":
    torch.set_grad_enabled(False)
    main()

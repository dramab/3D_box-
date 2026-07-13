#!/usr/bin/env python
"""
Run LC-BGPlaceNet Stage 1 inference and export RGB visualizations.

使用示例:
    python tools/infer_lc_bgplacenet_stage1.py \
        --config configs/lc_bgplacenet_stage1.yaml \
        --checkpoint outputs/lc_bgplacenet_stage1/best.pt

    python tools/infer_lc_bgplacenet_stage1.py \
        --config configs/lc_bgplacenet_stage1.yaml \
        --checkpoint outputs/lc_bgplacenet_stage1/best.pt \
        --split all --max-samples 20
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

os.environ.setdefault("MPLCONFIGDIR", os.fspath(PROJECT_ROOT / "outputs" / ".matplotlib"))

from src.datasets.canonical import ObjectInfo, load_canonical_scene
from src.models.lc_bgplacenet.stage1 import LCBGPlaceNetStage1, aabb_iou_3d
from src.training.lc_bgplacenet_stage1 import (
    LCBGPlaceNetStage1Dataset,
    build_sources_from_config,
    build_stage1_index,
    load_config,
    move_batch_to_device,
    normalize_stage1_split,
    stage1_collate,
)
from src.visualization.bbox_projection import BOX_EDGES, project_world


PRED_COLOR = (230, 57, 70)
GT_COLOR = (29, 128, 91)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Run LC-BGPlaceNet Stage 1 inference.")
    parser.add_argument("--config", type=Path, required=True, help="Stage 1 YAML config path.")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Checkpoint path, usually best.pt or last.pt.")
    parser.add_argument(
        "--split",
        choices=("valid", "val", "train", "test", "all"),
        default="valid",
        help="Inference split. val is accepted as an alias of valid.",
    )
    parser.add_argument("--batch-size", type=int, default=None, help="Inference batch size. Defaults to training.batch_size.")
    parser.add_argument("--max-samples", type=int, default=None, help="Limit sample count for smoke tests.")
    parser.add_argument("--sample-id", default=None, help="Only infer one sample_id, useful for visualization debugging.")
    parser.add_argument("--object-id", default=None, help="Only infer one object_id; usually used together with --sample-id.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory. Defaults to training.output_dir/inference_rgb_<split>.",
    )
    parser.add_argument("--line-width", type=int, default=3, help="Projected 3D box line width.")
    parser.add_argument("--no-gt", action="store_true", help="Only draw predicted source boxes.")
    parser.add_argument("--device", default=None, help="Override config training.device, for example cuda:0.")
    return parser.parse_args()


def select_all_items(cfg: dict[str, Any]) -> list[Any]:
    """Build all Stage 1 items from the config."""
    sources = build_sources_from_config(cfg)
    return build_stage1_index(sources)


def build_inference_loader(
    cfg: dict[str, Any],
    split: str,
    batch_size: int | None,
    max_samples: int | None,
    sample_id: str | None = None,
    object_id: str | None = None,
) -> DataLoader:
    """Build a non-shuffled inference dataloader."""
    items = select_all_items(cfg)
    data_cfg = cfg["data"]
    dataset_split = "train" if split == "all" else normalize_stage1_split(split)
    valid_fraction = float(data_cfg.get("valid_fraction", data_cfg.get("val_fraction", 0.1)))
    split_dir = None if split == "all" else data_cfg.get("split_dir")
    dataset = LCBGPlaceNetStage1Dataset(
        sources=None,
        split=dataset_split,
        val_fraction=0.0 if split == "all" else valid_fraction,
        seed=int(data_cfg.get("split_seed", 0)),
        max_samples=None if sample_id is not None or object_id is not None else max_samples,
        items=items,
        split_dir=split_dir,
    )
    if sample_id is not None or object_id is not None:
        dataset.items = [
            item
            for item in dataset.items
            if (sample_id is None or item.sample_id == sample_id)
            and (object_id is None or item.object_id == object_id)
        ]
        if max_samples is not None:
            dataset.items = dataset.items[: int(max_samples)]
        if not dataset.items:
            raise ValueError("No samples matched --sample-id/--object-id in the selected split.")
    effective_batch_size = int(batch_size or cfg["training"]["batch_size"])
    return DataLoader(
        dataset,
        batch_size=effective_batch_size,
        shuffle=False,
        num_workers=int(cfg["training"].get("num_workers", 0)),
        collate_fn=lambda batch: stage1_collate(batch, voxel_size_cm=float(cfg["data"]["voxel_size_cm"])),
        pin_memory=bool(cfg["training"].get("pin_memory", True)),
    )


def resolve_device(cfg: dict[str, Any], override: str | None) -> torch.device:
    """Resolve inference device and keep the spconv CUDA requirement explicit."""
    requested = override or str(cfg["training"].get("device", "auto"))
    if requested == "auto":
        requested = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(requested)
    if str(cfg["model"]["backbone"].get("type", "spconv")).lower() == "spconv" and device.type != "cuda":
        raise RuntimeError("spconv backbone requires CUDA in this environment; please run with a CUDA device.")
    return device


def load_model(cfg: dict[str, Any], checkpoint_path: Path, device: torch.device) -> LCBGPlaceNetStage1:
    """Load a Stage 1 model checkpoint for inference."""
    model = LCBGPlaceNetStage1(cfg["model"]).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint["model"] if isinstance(checkpoint, dict) and "model" in checkpoint else checkpoint
    model.load_state_dict(state_dict)
    model.eval()
    return model


def _axis_order_from_pose(pose_world: np.ndarray) -> tuple[np.ndarray, list[int], int]:
    """Return normalized object axes and the dimension order used by source_box_gt."""
    axes = np.asarray(pose_world, dtype=np.float64)[:3, :3]
    axis_norms = np.linalg.norm(axes, axis=0)
    if np.any(axis_norms < 1e-12):
        raise ValueError("pose_world contains a degenerate rotation axis")
    axes = axes / axis_norms[None, :]

    world_up = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    up_axis = int(np.argmax(np.abs(axes.T @ world_up)))
    horizontal_axes = [axis for axis in range(3) if axis != up_axis]
    return axes, horizontal_axes, up_axis


def _find_scene_object(scene: Any, object_id: str) -> ObjectInfo:
    """Find one object in a canonical scene by object id."""
    for obj in scene.objects:
        if obj.obj_id == object_id:
            return obj
    raise ValueError(f"Object {object_id} not found in sample {scene.sample_id}")


def source_box_to_oriented_corners(box: np.ndarray, pose_world: np.ndarray) -> np.ndarray:
    """Convert Stage 1 (center, local horizontal sizes, height) to oriented world corners."""
    box = np.asarray(box, dtype=np.float64)
    axes, horizontal_axes, up_axis = _axis_order_from_pose(pose_world)
    dims_by_pose_axis = np.empty(3, dtype=np.float64)
    dims_by_pose_axis[horizontal_axes[0]] = box[3]
    dims_by_pose_axis[horizontal_axes[1]] = box[4]
    dims_by_pose_axis[up_axis] = box[5]

    center = box[:3]
    half_axes = axes * (np.maximum(dims_by_pose_axis, 1e-4) * 0.5)[None, :]
    corners = []
    for zi in range(2):
        for yi in range(2):
            for xi in range(2):
                signs = np.array([xi, yi, zi], dtype=np.float64) * 2.0 - 1.0
                corners.append(center + half_axes @ signs)
    return np.asarray(corners, dtype=np.float64)


def draw_world_corners(
    draw: ImageDraw.ImageDraw,
    corners_world: np.ndarray,
    scene: Any,
    color: tuple[int, int, int],
    line_width: int,
    label: str,
) -> bool:
    """Draw one world-space 3D box from its corners."""
    uv, z_cam = project_world(corners_world, scene.camera.K, scene.camera.E_w2c)

    drawn = False
    for i, j in BOX_EDGES:
        if z_cam[i] <= 0.0 or z_cam[j] <= 0.0:
            continue
        draw.line(
            [(float(uv[i, 0]), float(uv[i, 1])), (float(uv[j, 0]), float(uv[j, 1]))],
            fill=color,
            width=line_width,
        )
        drawn = True

    if drawn:
        visible = z_cam > 0.0
        draw_label(draw, label, uv[visible].mean(axis=0), color)
    return drawn


def draw_source_box_with_pose(
    draw: ImageDraw.ImageDraw,
    box: np.ndarray,
    scene: Any,
    obj: ObjectInfo,
    color: tuple[int, int, int],
    line_width: int,
    label: str,
) -> bool:
    """Draw a Stage 1 source box using the source object's GT orientation."""
    corners_world = source_box_to_oriented_corners(box, obj.pose_world)
    return draw_world_corners(draw, corners_world, scene, color, line_width, label)


def draw_label(draw: ImageDraw.ImageDraw, text: str, uv: np.ndarray, color: tuple[int, int, int]) -> None:
    """Draw a compact readable label near a projected box."""
    font = ImageFont.load_default()
    x = float(uv[0])
    y = float(uv[1])
    bbox = draw.textbbox((x, y), text, font=font)
    pad = 3
    draw.rectangle((bbox[0] - pad, bbox[1] - pad, bbox[2] + pad, bbox[3] + pad), fill=(0, 0, 0))
    draw.text((x, y), text, fill=color, font=font)


def save_rgb_visualization(
    cfg: dict[str, Any],
    sample_id: str,
    source_name: str,
    object_id: str,
    pred_box: np.ndarray,
    gt_box: np.ndarray,
    output_path: Path,
    line_width: int,
    draw_gt: bool,
) -> None:
    """Load the canonical RGB image and save predicted/GT source-box projection."""
    source_cfg = next(item for item in cfg["data"]["sources"] if str(item["name"]) == source_name)
    dataset_dir = Path(source_cfg["dataset_dir"])
    scene = load_canonical_scene(dataset_dir / "samples" / f"{sample_id}.json", dataset_root=dataset_dir)
    obj = _find_scene_object(scene, object_id)
    image = Image.fromarray(np.asarray(scene.rgb, dtype=np.uint8)).convert("RGB")
    draw = ImageDraw.Draw(image)

    if draw_gt:
        draw_source_box_with_pose(draw, gt_box, scene, obj, GT_COLOR, line_width, f"gt:{object_id}")
    draw_source_box_with_pose(draw, pred_box, scene, obj, PRED_COLOR, line_width, f"pred:{object_id}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)


def tensor_row_to_list(tensor: torch.Tensor, index: int) -> list[float]:
    """Convert one tensor row to JSON-friendly floats."""
    return [float(x) for x in tensor[index].detach().cpu().tolist()]


@torch.no_grad()
def run_inference(args: argparse.Namespace) -> None:
    """Run inference, save visualizations and write JSONL predictions."""
    cfg = load_config(args.config)
    if args.batch_size is not None:
        cfg = deepcopy(cfg)
        cfg["training"]["batch_size"] = int(args.batch_size)

    output_split = "all" if args.split == "all" else normalize_stage1_split(args.split)
    output_dir = args.output_dir or Path(cfg["training"]["output_dir"]) / f"inference_rgb_{output_split}"
    output_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = output_dir / "predictions.jsonl"
    if predictions_path.exists():
        predictions_path.unlink()

    device = resolve_device(cfg, args.device)
    model = load_model(cfg, args.checkpoint, device)
    loader = build_inference_loader(
        cfg,
        args.split,
        args.batch_size,
        args.max_samples,
        sample_id=args.sample_id,
        object_id=args.object_id,
    )

    total = 0
    ious = []
    center_maes = []
    with predictions_path.open("w", encoding="utf-8") as f:
        for batch in loader:
            batch = move_batch_to_device(batch, device)
            outputs = model(batch)
            pred_boxes = outputs["source_box"].detach().cpu()
            gt_boxes = batch["source_box_gt"].detach().cpu()
            batch_ious = aabb_iou_3d(pred_boxes, gt_boxes).cpu()
            batch_center_mae = torch.mean(torch.abs(pred_boxes[:, :3] - gt_boxes[:, :3]), dim=1).cpu()

            for index, sample_id in enumerate(batch["sample_ids"]):
                source_name = batch["source_names"][index]
                object_id = batch["object_ids"][index]
                pred_box = pred_boxes[index].numpy()
                gt_box = gt_boxes[index].numpy()
                vis_path = output_dir / f"{source_name}__{sample_id}__{object_id}.png"
                save_rgb_visualization(
                    cfg=cfg,
                    sample_id=str(sample_id),
                    source_name=str(source_name),
                    object_id=str(object_id),
                    pred_box=pred_box,
                    gt_box=gt_box,
                    output_path=vis_path,
                    line_width=int(args.line_width),
                    draw_gt=not args.no_gt,
                )

                row = {
                    "source_name": str(source_name),
                    "sample_id": str(sample_id),
                    "object_id": str(object_id),
                    "instruction": batch["instructions"][index],
                    "pred_source_box_cxcycz_dxdydz_cm": tensor_row_to_list(pred_boxes, index),
                    "gt_source_box_cxcycz_dxdydz_cm": tensor_row_to_list(gt_boxes, index),
                    "source_iou": float(batch_ious[index]),
                    "source_center_mae_cm": float(batch_center_mae[index]),
                    "visualization_png": str(vis_path),
                    "visualization_rotation_source": "gt_pose",
                }
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
                total += 1
                ious.append(float(batch_ious[index]))
                center_maes.append(float(batch_center_mae[index]))
                print(f"Saved {vis_path}")

    summary = {
        "split": output_split,
        "samples": total,
        "source_iou_mean": float(np.mean(ious)) if ious else 0.0,
        "source_center_mae_cm_mean": float(np.mean(center_maes)) if center_maes else 0.0,
        "predictions_jsonl": str(predictions_path),
        "output_dir": str(output_dir),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False))


def main() -> None:
    """CLI entrypoint."""
    args = parse_args()
    try:
        run_inference(args)
    except RuntimeError as exc:
        raise SystemExit(f"ERROR: {exc}") from exc


if __name__ == "__main__":
    main()

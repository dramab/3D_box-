#!/usr/bin/env python
"""
Run LC-BGPlaceNet Stage 2 inference and export JSON/RGB visualizations.

使用示例:
    python tools/infer_lc_bgplacenet_stage2.py \
        --config configs/lc_bgplacenet_stage2.yaml \
        --checkpoint outputs/lc_bgplacenet_stage2/best.pt

    python tools/infer_lc_bgplacenet_stage2.py \
        --config configs/lc_bgplacenet_stage2.yaml \
        --checkpoint outputs/lc_bgplacenet_stage2/best.pt \
        --split valid --max-samples 20
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
from PIL import Image, ImageDraw, ImageFont
from torch.utils.data import DataLoader, Dataset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

os.environ.setdefault("MPLCONFIGDIR", os.fspath(PROJECT_ROOT / "outputs" / ".matplotlib"))

from src.annotation.free_bbox.io_utils import load_ply, save_ply
from src.datasets.canonical import ObjectInfo, load_canonical_scene
from src.models.lc_bgplacenet.stage2 import (
    LCBGPlaceNetDensePlacement,
    boxes_from_bottom_centers,
    pose_nms,
)
from src.training.lc_bgplacenet_stage2 import (
    Stage2IndexItem,
    _quantize_world_points,
    build_sources_from_config,
    build_stage2_index,
    load_config,
    move_batch_to_device,
    normalize_stage1_split,
    select_stage2_split_items,
)
from src.training.lc_bgplacenet_stage1 import _source_box_from_obb
from src.visualization.bbox_projection import BOX_EDGES, project_world


PRED_SOURCE_COLOR = (230, 57, 70)
PRED_PLACE_COLOR = (37, 99, 235)
GT_PLACE_COLOR = (29, 128, 91)
DECODER_CANDIDATE_COLOR = (96, 165, 250)
PRED_HEATMAP_COLORS = np.array(
    [
        [37, 99, 235],
        [6, 182, 212],
        [250, 204, 21],
        [220, 38, 38],
    ],
    dtype=np.float32,
)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Run LC-BGPlaceNet Stage 2 inference.")
    parser.add_argument("--config", type=Path, required=True, help="Stage 2 YAML config path.")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Stage 2 checkpoint path.")
    parser.add_argument(
        "--split",
        choices=("valid", "val", "train", "test", "all"),
        default="valid",
        help="Inference split. val is accepted as an alias of valid.",
    )
    parser.add_argument("--batch-size", type=int, default=None, help="Inference batch size. Defaults to training.batch_size.")
    parser.add_argument("--max-samples", type=int, default=None, help="Limit sample count.")
    parser.add_argument("--sample-id", default=None, help="Only infer one sample_id.")
    parser.add_argument("--object-id", default=None, help="Only infer one object_id; usually used with --sample-id.")
    parser.add_argument(
        "--instruction",
        default=None,
        help="Prediction-only instruction for a canonical sample absent from Stage-2 labels.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory. Defaults to training.output_dir/inference_stage2_<split>.",
    )
    parser.add_argument("--line-width", type=int, default=3, help="Projected 3D box line width.")
    parser.add_argument("--no-gt", action="store_true", help="Do not draw GT placement boxes.")
    parser.add_argument("--device", default=None, help="Override config training.device, for example cuda:0.")
    return parser.parse_args()


class Stage2InferenceDataset(Dataset):
    """Lightweight inference dataset that does not read heatmap/support PLY files."""

    def __init__(self, items: list[Stage2IndexItem], max_samples: int | None = None) -> None:
        self.items = list(items)
        if max_samples is not None:
            self.items = self.items[: int(max_samples)]
        if not self.items:
            raise ValueError("No Stage 2 inference samples selected")

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        item = self.items[idx]
        points, colors = load_ply(item.voxel_point_cloud_path)
        if len(points) == 0:
            raise ValueError(f"Invalid empty point data for {item.sample_id}")
        image = np.asarray(Image.open(item.rgb_path).convert("RGB"), dtype=np.uint8)
        return {
            "item": item,
            "points": points.astype(np.float32),
            "colors": colors.astype(np.uint8),
            "image": image,
            "camera_K": item.camera_K,
            "camera_E_w2c": item.camera_E_w2c,
            "instruction": item.instruction,
            "source_box_gt": item.source_box_gt,
            "place_box_gt": item.place_box_gt,
        }


def select_items(
    cfg: dict[str, Any],
    split: str,
    sample_id: str | None,
    object_id: str | None,
    instruction: str | None = None,
) -> list[Stage2IndexItem]:
    """Select Stage 2 inference items from config and optional filters."""
    if instruction is not None:
        if sample_id is None or object_id is None:
            raise ValueError("--instruction requires both --sample-id and --object-id")
        return [build_prediction_only_item(cfg, sample_id, object_id, instruction)]

    sources = build_sources_from_config(cfg)
    items = build_stage2_index(sources)
    if split != "all":
        items = select_stage2_split_items(items, cfg["data"]["split_dir"], normalize_stage1_split(split))
    if sample_id is not None or object_id is not None:
        items = [
            item
            for item in items
            if (sample_id is None or item.sample_id == sample_id)
            and (object_id is None or item.object_id == object_id)
        ]
    return items


def build_prediction_only_item(
    cfg: dict[str, Any],
    sample_id: str,
    object_id: str,
    instruction: str,
) -> Stage2IndexItem:
    """Build one inference item directly from canonical metadata without GT labels."""
    source = next(
        (
            row
            for row in build_sources_from_config(cfg)
            if (row.dataset_dir / "samples" / f"{sample_id}.json").exists()
        ),
        None,
    )
    if source is None:
        raise FileNotFoundError(f"Canonical sample {sample_id} is missing from configured sources")

    sample_path = source.dataset_dir / "samples" / f"{sample_id}.json"
    with sample_path.open("r", encoding="utf-8") as handle:
        record = json.load(handle)
    object_record = next(
        (row for row in record["objects"] if str(row["obj_id"]) == object_id),
        None,
    )
    if object_record is None:
        raise ValueError(f"Object {object_id} is missing from canonical sample {sample_id}")
    voxel_rel = record.get("voxel_point_cloud_path")
    if not voxel_rel:
        raise ValueError(f"Canonical sample {sample_id} does not provide voxel_point_cloud_path")

    camera = record["camera"]
    camera_k = np.array(
        [
            [float(camera["fx"]), 0.0, float(camera["cx"])],
            [0.0, float(camera["fy"]), float(camera["cy"])],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    camera_e_w2c = np.linalg.inv(np.asarray(camera["E_c2w"], dtype=np.float64)).astype(np.float32)
    source_box = _source_box_from_obb(
        object_record["bbox3d_canonical"],
        object_record["pose_world"],
    )
    placeholder_place_box = np.concatenate(
        [source_box[:3], source_box[3:6], np.zeros(1, dtype=np.float32)]
    ).astype(np.float32)
    item_token = f"{sample_id.replace('__', '_')}_{object_id}"
    unused_path = source.free_bbox_dir / "prediction_only_unused"
    return Stage2IndexItem(
        item_id=f"{source.name}__custom_{item_token}__prediction_only",
        label_index=-1,
        source_name=source.name,
        sample_id=sample_id,
        object_id=object_id,
        cluster_id=0,
        instruction=instruction,
        dataset_dir=source.dataset_dir,
        free_bbox_dir=source.free_bbox_dir,
        voxel_point_cloud_path=source.dataset_dir / str(voxel_rel),
        rgb_path=source.dataset_dir / str(record["rgb_path"]),
        camera_K=camera_k,
        camera_E_w2c=camera_e_w2c,
        support_mask_ply=unused_path,
        direction_filtered_heatmap_ply=unused_path,
        yaw_set_npz=unused_path,
        source_box_gt=source_box,
        place_box_gt=placeholder_place_box,
        target_relation="prediction_only",
        reference_object_id="prediction_only",
    )


def inference_collate(batch: list[dict[str, Any]], voxel_size_cm: float = 1.0) -> dict[str, Any]:
    """Collate inference samples without loading any training-only supervision."""
    features = []
    sparse_coords = []
    world_coords = []
    coords_norm = []
    batch_indices = []
    scene_min = []
    scene_max = []
    instructions = []
    source_boxes = []
    place_boxes = []
    items = []
    images = []
    camera_k = []
    camera_e_w2c = []
    image_hw = []
    spatial_max = np.zeros(3, dtype=np.int64)

    for batch_idx, row in enumerate(batch):
        points = np.asarray(row["points"], dtype=np.float32)
        colors = np.asarray(row["colors"], dtype=np.float32) / 255.0
        point_min = points.min(axis=0)
        point_max = points.max(axis=0)
        extent = np.maximum(point_max - point_min, 1e-4)
        point_norm = (points - point_min) / extent
        voxel_keys = _quantize_world_points(points, voxel_size_cm)
        shifted_keys = voxel_keys - voxel_keys.min(axis=0, keepdims=True)
        spatial_max = np.maximum(spatial_max, shifted_keys.max(axis=0) + 1)
        batch_col = np.full((len(points), 1), batch_idx, dtype=np.int32)

        features.append(np.concatenate([point_norm, colors], axis=1).astype(np.float32))
        sparse_coords.append(np.concatenate([batch_col, shifted_keys.astype(np.int32)], axis=1))
        world_coords.append(points)
        coords_norm.append(point_norm.astype(np.float32))
        batch_indices.append(np.full(len(points), batch_idx, dtype=np.int64))
        scene_min.append(point_min.astype(np.float32))
        scene_max.append(point_max.astype(np.float32))
        instructions.append(str(row["instruction"]))
        source_boxes.append(np.asarray(row["source_box_gt"], dtype=np.float32))
        place_boxes.append(np.asarray(row["place_box_gt"], dtype=np.float32))
        items.append(row["item"])
        image = np.asarray(row["image"], dtype=np.uint8)
        images.append(image)
        camera_k.append(np.asarray(row["camera_K"], dtype=np.float32))
        camera_e_w2c.append(np.asarray(row["camera_E_w2c"], dtype=np.float32))
        image_hw.append(np.asarray(image.shape[:2], dtype=np.float32))

    return {
        "features": torch.from_numpy(np.concatenate(features, axis=0)),
        "sparse_coords": torch.from_numpy(np.concatenate(sparse_coords, axis=0)),
        "spatial_shape": [int(x) for x in spatial_max.tolist()],
        "world_coords": torch.from_numpy(np.concatenate(world_coords, axis=0)),
        "coords_norm": torch.from_numpy(np.concatenate(coords_norm, axis=0)),
        "batch_indices": torch.from_numpy(np.concatenate(batch_indices, axis=0)),
        "scene_min": torch.from_numpy(np.stack(scene_min, axis=0)),
        "scene_max": torch.from_numpy(np.stack(scene_max, axis=0)),
        "source_box_gt": torch.from_numpy(np.stack(source_boxes, axis=0)),
        "place_box_gt": torch.from_numpy(np.stack(place_boxes, axis=0)),
        "instructions": instructions,
        "items": items,
        "images": images,
        "camera_K": torch.from_numpy(np.stack(camera_k, axis=0)),
        "camera_E_w2c": torch.from_numpy(np.stack(camera_e_w2c, axis=0)),
        "image_hw": torch.from_numpy(np.stack(image_hw, axis=0)),
        "batch_size": len(batch),
    }


def resolve_device(cfg: dict[str, Any], override: str | None) -> torch.device:
    """Resolve inference device and keep the spconv CUDA requirement explicit."""
    requested = override or str(cfg["training"].get("device", "auto"))
    if requested == "auto":
        requested = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(requested)
    if str(cfg["model"]["backbone"].get("type", "spconv")).lower() == "spconv" and device.type != "cuda":
        raise RuntimeError("spconv backbone requires CUDA in this environment; please run with a CUDA device.")
    return device


def load_model(cfg: dict[str, Any], checkpoint_path: Path, device: torch.device) -> LCBGPlaceNetDensePlacement:
    """Load a Stage 2 model checkpoint for inference."""
    model = LCBGPlaceNetDensePlacement(cfg["model"]).to(device)
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
    """Convert Stage 1 source box to oriented world corners using the source pose axes."""
    box = np.asarray(box, dtype=np.float64)
    axes, horizontal_axes, up_axis = _axis_order_from_pose(pose_world)
    dims_by_pose_axis = np.empty(3, dtype=np.float64)
    dims_by_pose_axis[horizontal_axes[0]] = box[3]
    dims_by_pose_axis[horizontal_axes[1]] = box[4]
    dims_by_pose_axis[up_axis] = box[5]
    return _corners_from_center_axes(box[:3], axes, dims_by_pose_axis)


def place_box_to_corners(box: np.ndarray) -> np.ndarray:
    """Convert (x, y, z, dx, dy, dz, yaw) to upright yaw box corners."""
    box = np.asarray(box, dtype=np.float64)
    yaw = float(box[6])
    axes = np.array(
        [
            [np.cos(yaw), -np.sin(yaw), 0.0],
            [np.sin(yaw), np.cos(yaw), 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    return _corners_from_center_axes(box[:3], axes, box[3:6])


def _corners_from_center_axes(center: np.ndarray, axes: np.ndarray, dims: np.ndarray) -> np.ndarray:
    """Build 8 box corners from center, column axes and dimensions."""
    half_axes = axes * (np.maximum(dims, 1e-4) * 0.5)[None, :]
    corners = []
    for zi in range(2):
        for yi in range(2):
            for xi in range(2):
                signs = np.array([xi, yi, zi], dtype=np.float64) * 2.0 - 1.0
                corners.append(center + half_axes @ signs)
    return np.asarray(corners, dtype=np.float64)


def draw_label(draw: ImageDraw.ImageDraw, text: str, xy: np.ndarray, color: tuple[int, int, int]) -> None:
    """Draw a compact label near a projected box."""
    font = ImageFont.load_default()
    xy_tuple = (float(xy[0]), float(xy[1]))
    draw.text(xy_tuple, text, fill=color, font=font)


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


def export_visualization(
    item: Stage2IndexItem,
    source_box: np.ndarray,
    place_box: np.ndarray,
    output_path: Path,
    line_width: int,
    draw_gt: bool,
) -> None:
    """Export one RGB visualization with predicted source and placement boxes."""
    sample_path = item.dataset_dir / "samples" / f"{item.sample_id}.json"
    scene = load_canonical_scene(sample_path, dataset_root=item.dataset_dir)
    image = Image.fromarray(scene.rgb).convert("RGB")
    draw = ImageDraw.Draw(image)
    obj = _find_scene_object(scene, item.object_id)
    draw_world_corners(
        draw,
        source_box_to_oriented_corners(source_box, np.asarray(obj.pose_world)),
        scene,
        PRED_SOURCE_COLOR,
        line_width,
        "pred source",
    )
    draw_world_corners(
        draw,
        place_box_to_corners(place_box),
        scene,
        PRED_PLACE_COLOR,
        line_width,
        "pred place",
    )
    if draw_gt:
        draw_world_corners(
            draw,
            place_box_to_corners(item.place_box_gt),
            scene,
            GT_PLACE_COLOR,
            line_width,
            "gt place",
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)


def build_decoder_stage_predictions(outputs: dict[str, torch.Tensor]) -> list[dict[str, torch.Tensor]]:
    """Postprocess all four decoder layers without changing the final prediction path."""
    query_valid_mask = outputs["query_valid_mask"]
    source_size = outputs["source_box"][:, 3:6].clamp_min(1e-4)
    stages = []
    for prediction in outputs["decoder_aux_outputs"]:
        raw_boxes = boxes_from_bottom_centers(
            prediction["pred_bottom_centers"], source_size, prediction["pred_yaw_indices"]
        )
        stages.append(
            pose_nms(
                raw_boxes,
                prediction["pred_logits"],
                prediction["pred_yaw_logits"],
                query_valid_mask,
            )
        )

    # 第四层直接复用模型最终后处理结果，确保 benchmark 的输入完全不变。
    stages.append(
        {
            "place_boxes": outputs["place_boxes"],
            "place_scores": outputs["place_scores"],
            "place_yaw_bins": outputs["place_yaw_bins"],
            "place_valid_mask": outputs["place_valid_mask"],
            "place_box": outputs["place_box"],
        }
    )
    return stages


def export_decoder_stage_visualizations(
    item: Stage2IndexItem,
    source_box: np.ndarray,
    decoder_stages: list[dict[str, np.ndarray]],
    output_paths: list[Path],
    line_width: int,
    draw_gt: bool,
) -> None:
    """Export four RGB panels showing each decoder layer's retained candidates."""
    sample_path = item.dataset_dir / "samples" / f"{item.sample_id}.json"
    scene = load_canonical_scene(sample_path, dataset_root=item.dataset_dir)
    obj = _find_scene_object(scene, item.object_id)
    source_corners = source_box_to_oriented_corners(source_box, np.asarray(obj.pose_world))

    for stage_index, (stage, output_path) in enumerate(zip(decoder_stages, output_paths), start=1):
        image = Image.fromarray(scene.rgb).convert("RGB")
        draw = ImageDraw.Draw(image)
        draw_world_corners(draw, source_corners, scene, PRED_SOURCE_COLOR, line_width, "pred source")
        if draw_gt:
            draw_world_corners(
                draw,
                place_box_to_corners(item.place_box_gt),
                scene,
                GT_PLACE_COLOR,
                line_width,
                "gt place",
            )

        valid_indices = np.flatnonzero(stage["place_valid_mask"])
        # 低分候选先绘制，最高分候选最后覆盖，保持收敛主路径清晰可见。
        for rank, candidate_index in reversed(list(enumerate(valid_indices, start=1))):
            is_best = rank == 1
            color = PRED_PLACE_COLOR if is_best else DECODER_CANDIDATE_COLOR
            width = line_width + 1 if is_best else max(1, line_width - 1)
            score = float(stage["place_scores"][candidate_index])
            label = f"D{stage_index} best {score:.3f}" if is_best else ""
            draw_world_corners(
                draw,
                place_box_to_corners(stage["place_boxes"][candidate_index]),
                scene,
                color,
                width,
                label,
            )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        image.save(output_path)


def decoder_stage_to_record(
    stage_index: int,
    stage: dict[str, np.ndarray],
    visualization_path: Path,
) -> dict[str, Any]:
    """Serialize one postprocessed decoder layer for JSON and the web viewer."""
    valid_indices = np.flatnonzero(stage["place_valid_mask"])
    placements = [
        {
            "box": stage["place_boxes"][index].tolist(),
            "score": float(stage["place_scores"][index]),
            "yaw_bin": int(stage["place_yaw_bins"][index]),
        }
        for index in valid_indices
    ]
    return {
        "stage": int(stage_index),
        "best_place_score": float(stage["place_scores"][0]),
        "placements": placements,
        "visualization_png": os.fspath(visualization_path),
    }


def colorize_predicted_heatmap(scores: np.ndarray) -> np.ndarray:
    """Color predicted placement probabilities with per-sample contrast stretching."""
    values = np.asarray(scores, dtype=np.float32).clip(0.0, 1.0)
    if len(values) == 0:
        return np.empty((0, 3), dtype=np.uint8)

    low, high = np.percentile(values, [1.0, 99.0])
    if high <= low + 1e-6:
        norm = np.zeros_like(values, dtype=np.float32)
    else:
        norm = ((values - float(low)) / float(high - low)).clip(0.0, 1.0)

    segment = np.minimum((norm * 3.0).astype(np.int64), 2)
    alpha = (norm * 3.0 - segment.astype(np.float32))[:, None]
    colors = PRED_HEATMAP_COLORS[segment] * (1.0 - alpha)
    colors += PRED_HEATMAP_COLORS[segment + 1] * alpha
    return np.rint(colors).clip(0, 255).astype(np.uint8)


def save_predicted_heatmap_ply(output_path: Path, points: np.ndarray, scores: np.ndarray) -> None:
    """Save active voxels colored by predicted placement heatmap probability."""
    colors = colorize_predicted_heatmap(scores)
    save_ply(output_path, points, colors)


def main() -> None:
    """Run Stage 2 inference."""
    args = parse_args()
    if args.instruction is not None and not args.no_gt:
        raise ValueError("Prediction-only --instruction inference requires --no-gt")
    cfg = load_config(args.config)
    device = resolve_device(cfg, args.device)
    model = load_model(cfg, args.checkpoint, device)
    items = select_items(cfg, args.split, args.sample_id, args.object_id, args.instruction)
    dataset = Stage2InferenceDataset(items, max_samples=args.max_samples)
    batch_size = int(args.batch_size or cfg["training"]["batch_size"])
    num_workers = int(cfg["training"].get("num_workers", 0))
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=lambda batch: inference_collate(batch, voxel_size_cm=float(cfg["data"]["voxel_size_cm"])),
        pin_memory=bool(cfg["training"].get("pin_memory", True)),
        persistent_workers=num_workers > 0,
    )

    split_name = "valid" if args.split == "val" else args.split
    output_dir = args.output_dir or Path(cfg["training"]["output_dir"]) / f"inference_stage2_{split_name}"
    heatmap_dir = output_dir / "pred_heatmaps"
    decoder_stage_dir = output_dir / "decoder_stages"
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []

    with torch.no_grad():
        for batch in loader:
            raw_items = batch["items"]
            batch = move_batch_to_device(batch, device)
            outputs = model(batch)
            source_boxes = outputs["source_box"].detach().cpu().numpy()
            place_boxes = outputs["place_box"].detach().cpu().numpy()
            placement_sets = outputs["place_boxes"].detach().cpu().numpy()
            placement_scores = outputs["place_scores"].detach().cpu().numpy()
            placement_yaw_bins = outputs["place_yaw_bins"].detach().cpu().numpy()
            placement_valid = outputs["place_valid_mask"].detach().cpu().numpy()
            region_scores = torch.sigmoid(outputs["region_logits"]).detach().cpu().numpy()
            region_points = outputs["region_world_coords"].detach().cpu().numpy()
            region_batch_indices = outputs["region_batch_indices"].detach().cpu().numpy()
            decoder_stages = [
                {key: value.detach().cpu().numpy() for key, value in stage.items()}
                for stage in build_decoder_stage_predictions(outputs)
            ]

            for row_idx, item in enumerate(raw_items):
                stem = f"{item.item_id}__{item.sample_id}__{item.object_id}__cluster_{item.cluster_id:03d}"
                vis_path = output_dir / f"{stem}.png"
                pred_heatmap_path = heatmap_dir / f"{stem}__pred_heatmap.ply"
                decoder_stage_paths = [
                    decoder_stage_dir / f"{stem}__decoder_{stage_index}.png"
                    for stage_index in range(1, 5)
                ]
                row_mask = region_batch_indices == row_idx
                save_predicted_heatmap_ply(
                    pred_heatmap_path,
                    region_points[row_mask],
                    region_scores[row_mask],
                )
                export_visualization(
                    item,
                    source_boxes[row_idx],
                    place_boxes[row_idx],
                    vis_path,
                    line_width=int(args.line_width),
                    draw_gt=not args.no_gt,
                )
                row_decoder_stages = [
                    {key: value[row_idx] for key, value in stage.items()}
                    for stage in decoder_stages
                ]
                export_decoder_stage_visualizations(
                    item,
                    source_boxes[row_idx],
                    row_decoder_stages,
                    decoder_stage_paths,
                    line_width=int(args.line_width),
                    draw_gt=not args.no_gt,
                )
                valid_indices = np.flatnonzero(placement_valid[row_idx])
                placements = [
                    {
                        "box": placement_sets[row_idx, index].tolist(),
                        "score": float(placement_scores[row_idx, index]),
                        "yaw_bin": int(placement_yaw_bins[row_idx, index]),
                    }
                    for index in valid_indices
                ]
                rows.append(
                    {
                        "item_id": item.item_id,
                        "source_name": item.source_name,
                        "sample_id": item.sample_id,
                        "object_id": item.object_id,
                        "cluster_id": item.cluster_id,
                        "instruction": item.instruction,
                        "source_box": source_boxes[row_idx].tolist(),
                        "place_box": place_boxes[row_idx].tolist(),
                        "placements": placements,
                        "place_box_gt": item.place_box_gt.tolist(),
                        "best_place_score": float(placement_scores[row_idx, 0]),
                        "visualization_png": os.fspath(vis_path),
                        "pred_heatmap_ply": os.fspath(pred_heatmap_path),
                        "decoder_stages": [
                            decoder_stage_to_record(stage_index, stage, stage_path)
                            for stage_index, (stage, stage_path) in enumerate(
                                zip(row_decoder_stages, decoder_stage_paths), start=1
                            )
                        ],
                        "prediction_only": bool(args.instruction is not None),
                    }
                )

    with (output_dir / "predictions.json").open("w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2, ensure_ascii=False)
    print(f"Wrote {len(rows)} Stage 2 predictions to {output_dir}")


if __name__ == "__main__":
    main()

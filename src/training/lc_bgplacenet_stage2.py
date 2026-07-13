"""
LC-BGPlaceNet Stage 2 training utilities.

Stage 2 trains source-conditioned dense placement prediction from canonical
active voxel point clouds, language instructions and free_bbox supervision.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
import yaml
from PIL import Image
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler

from src.annotation.auto_label import describe_spatial_relation
from src.annotation.free_bbox.geometry import get_bbox_corners, transform_points
from src.annotation.free_bbox.io_utils import load_ply
from src.datasets.canonical import CameraParams, load_sample_record
from src.models.lc_bgplacenet.stage2 import LCBGPlaceNetDensePlacement
from src.training.lc_bgplacenet_stage1 import (
    STAGE1_SPLIT_SCHEMA_VERSION,
    STAGE1_SPLIT_NAME_SET,
    Stage1DataSource,
    _append_jsonl,
    _find_object_record,
    _init_distributed,
    _is_main_process,
    _progress,
    _resolve_device,
    _resolve_rgb_camera_metadata,
    _resolve_voxel_path,
    _source_box_from_obb,
    build_sources_from_config,
    make_stage1_item_id,
    move_batch_to_device,
    normalize_stage1_split,
    set_seed,
    source_box_loss,
)


BOX_COLLISION_EPS_CM = 1e-6


@dataclass(frozen=True)
class Stage2ValidationContext:
    """Cached scene geometry for task-aligned validation metrics."""

    target_relation: str
    reference_corners_world: np.ndarray
    camera_K: np.ndarray
    camera_E_w2c: np.ndarray
    collision_context: dict[str, Any]


@dataclass(frozen=True)
class Stage2IndexItem:
    """Resolved metadata for one language-conditioned Stage 2 sample."""

    item_id: str
    label_index: int
    source_name: str
    sample_id: str
    object_id: str
    cluster_id: int
    instruction: str
    dataset_dir: Path
    free_bbox_dir: Path
    voxel_point_cloud_path: Path
    rgb_path: Path
    camera_K: np.ndarray
    camera_E_w2c: np.ndarray
    support_mask_ply: Path
    direction_filtered_heatmap_ply: Path
    source_box_gt: np.ndarray
    place_box_gt: np.ndarray
    target_relation: str
    reference_object_id: str


def load_config(path: str | Path) -> dict[str, Any]:
    """Load a YAML config file."""
    with Path(path).open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_stage2_index(sources: list[Stage1DataSource]) -> list[Stage2IndexItem]:
    """Build resolved Stage 2 sample metadata from all configured sources."""
    items: list[Stage2IndexItem] = []
    payload_cache: dict[tuple[str, str], dict[str, Any]] = {}

    for source in sources:
        with source.labels_path.open("r", encoding="utf-8") as f:
            label_records = json.load(f)

        for label_index, record in enumerate(label_records):
            sample_id = str(record["sample_id"])
            object_id = str(record["object_id"])
            cluster_id = int(record["cluster_id"])
            instruction = str(record["label"])
            vis_rel = record.get("visualization_png")
            if vis_rel and not (source.free_bbox_dir / str(vis_rel)).exists():
                continue

            cache_key = (source.name, sample_id)
            if cache_key not in payload_cache:
                placement_path = source.free_bbox_dir / "placements" / f"{sample_id}__placements.json"
                with placement_path.open("r", encoding="utf-8") as f:
                    payload_cache[cache_key] = json.load(f)
            payload = payload_cache[cache_key]

            obj_record = _find_object_record(payload, object_id)
            if obj_record is None:
                raise ValueError(f"Object {object_id} not found in {source.name}/{sample_id} placements")
            placement = _find_placement_record(obj_record, cluster_id, record.get("placement_sample_id"))
            if placement is None:
                raise ValueError(f"Placement cluster={cluster_id} not found in {source.name}/{sample_id}/{object_id}")

            raw_heatmap = placement.get("heatmap_ply")
            support_mask = payload.get("support_mask_ply")
            if not raw_heatmap or not support_mask:
                continue
            direction_heatmap = _direction_filtered_heatmap_path(source.free_bbox_dir, raw_heatmap)
            support_mask_path = _resolve_free_bbox_path(source.free_bbox_dir, support_mask)
            if not direction_heatmap.exists() or not support_mask_path.exists():
                continue
            placement_relation = record.get("spatial_relation", {}).get("placement", {})
            target_relation = placement_relation.get("relation")
            reference_object_id = placement_relation.get("reference_object_id")
            if not target_relation or reference_object_id is None:
                raise ValueError(f"Missing placement direction metadata for {source.name}/{sample_id}/{object_id}")
            rgb_path, camera_k, camera_e_w2c = _resolve_rgb_camera_metadata(source.dataset_dir, sample_id)

            items.append(
                Stage2IndexItem(
                    item_id=make_stage1_item_id(source.name, label_index, sample_id, object_id, instruction),
                    label_index=label_index,
                    source_name=source.name,
                    sample_id=sample_id,
                    object_id=object_id,
                    cluster_id=cluster_id,
                    instruction=instruction,
                    dataset_dir=source.dataset_dir,
                    free_bbox_dir=source.free_bbox_dir,
                    voxel_point_cloud_path=_resolve_voxel_path(source.dataset_dir, sample_id, payload),
                    rgb_path=rgb_path,
                    camera_K=camera_k,
                    camera_E_w2c=camera_e_w2c,
                    support_mask_ply=support_mask_path,
                    direction_filtered_heatmap_ply=direction_heatmap,
                    source_box_gt=_source_box_from_obb(
                        obj_record["canonical_aabb_object"],
                        obj_record["original_pose_world"],
                    ),
                    place_box_gt=_place_box_from_placement(placement),
                    target_relation=str(target_relation),
                    reference_object_id=str(reference_object_id),
                )
            )
    return items


def _find_placement_record(
    obj_record: dict[str, Any],
    cluster_id: int,
    placement_sample_id: str | None = None,
) -> dict[str, Any] | None:
    """Find one placement in an object record."""
    for placement in obj_record.get("placements", []):
        if placement_sample_id is not None and placement.get("sample_id") == placement_sample_id:
            return placement
        if int(placement.get("cluster_id", -1)) == int(cluster_id):
            return placement
    return None


def _resolve_free_bbox_path(free_bbox_root: Path, raw_path: str | os.PathLike[str]) -> Path:
    """Resolve a free_bbox relative path."""
    path = Path(raw_path)
    return path if path.is_absolute() else free_bbox_root / path


def _direction_filtered_heatmap_path(free_bbox_root: Path, raw_heatmap_path: str | os.PathLike[str]) -> Path:
    """Map a raw heatmap path to the direction_filtered_heatmaps directory."""
    return free_bbox_root / "direction_filtered_heatmaps" / Path(raw_heatmap_path).name


def _place_box_from_placement(placement: dict[str, Any]) -> np.ndarray:
    """Convert free_bbox placement metadata to (x, y, z, dx, dy, dz, yaw)."""
    center = np.asarray(placement["center_world"], dtype=np.float32)
    dims = np.asarray(placement["yaw_only_dimensions"], dtype=np.float32)
    yaw = np.deg2rad(float(placement["yaw_degrees"]))
    return np.concatenate([center, dims, np.asarray([yaw], dtype=np.float32)]).astype(np.float32)


def compute_size_iou(pred_dims: np.ndarray, gt_dims: np.ndarray) -> float:
    """Compute dimension-only volume IoU while preserving dimension order."""
    pred = np.maximum(np.asarray(pred_dims, dtype=np.float64), 0.0)
    gt = np.maximum(np.asarray(gt_dims, dtype=np.float64), 0.0)
    intersection = float(np.prod(np.minimum(pred, gt)))
    union = float(np.prod(np.maximum(pred, gt)))
    return intersection / union if union > 0.0 else 0.0


def compute_size_metrics(
    place_box: np.ndarray,
    place_box_gt: np.ndarray,
    threshold: float,
) -> dict[str, Any]:
    """Compute dimension IoU and its thresholded correctness flag."""
    size_iou = compute_size_iou(place_box[3:6], place_box_gt[3:6])
    return {"size_iou": size_iou, "size_correct": bool(size_iou >= float(threshold))}


def compute_yaw_metrics(
    place_box: np.ndarray,
    place_box_gt: np.ndarray,
    yaw_sensitive_ratio: float,
) -> dict[str, Any]:
    """Compute 180-degree-equivalent yaw error for direction-sensitive boxes."""
    gt_dx, gt_dy = float(place_box_gt[3]), float(place_box_gt[4])
    size_ratio = abs(gt_dx - gt_dy) / max(min(gt_dx, gt_dy), 1e-6)
    yaw_sensitive = size_ratio >= float(yaw_sensitive_ratio)
    if not yaw_sensitive:
        return {"yaw_sensitive": False, "yaw_error_deg": None}
    yaw_delta = (float(place_box[6] - place_box_gt[6]) + math.pi * 0.5) % math.pi - math.pi * 0.5
    return {"yaw_sensitive": True, "yaw_error_deg": abs(math.degrees(yaw_delta))}


def compute_task_success(direction_hit: bool, collision: bool, size_correct: bool) -> bool:
    """Return the shared val/benchmark task-success decision."""
    return bool(direction_hit and not collision and size_correct)


def _place_box_to_bbox_and_transform(place_box: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Convert (x, y, z, dx, dy, dz, yaw) to a local AABB and world transform."""
    box = np.asarray(place_box, dtype=np.float64)
    dims = np.maximum(box[3:6], 1e-4)
    yaw = float(box[6])
    c, s = math.cos(yaw), math.sin(yaw)
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)
    transform[:3, 3] = box[:3]
    return np.concatenate([-dims * 0.5, dims * 0.5]), transform


def place_box_to_corners(place_box: np.ndarray) -> np.ndarray:
    """Convert a yaw-only place box to its eight world-space corners."""
    bbox, transform = _place_box_to_bbox_and_transform(place_box)
    return transform_points(get_bbox_corners(bbox), transform)


def _obb_from_bbox_transform(object_id: str, bbox3d: np.ndarray, transform: np.ndarray) -> dict[str, Any]:
    """Convert a local AABB and object-to-world transform to an OBB."""
    bbox = np.asarray(bbox3d, dtype=np.float64)
    pose = np.asarray(transform, dtype=np.float64)
    local_center = (bbox[:3] + bbox[3:]) * 0.5
    axes = pose[:3, :3]
    axis_scales = np.linalg.norm(axes, axis=0)
    if np.any(axis_scales < 1e-12):
        raise ValueError(f"Object {object_id} has a degenerate OBB transform")
    return {
        "object_id": object_id,
        "center": (axes @ local_center + pose[:3, 3]).astype(np.float64),
        "axes": (axes / axis_scales[None, :]).astype(np.float64),
        "half_extents": (np.maximum((bbox[3:] - bbox[:3]) * 0.5, 1e-4) * axis_scales).astype(np.float64),
    }


def _obb_intersects(a: dict[str, Any], b: dict[str, Any], eps: float = BOX_COLLISION_EPS_CM) -> bool:
    """Return True only when two OBBs have positive-volume overlap."""
    a_axes = np.asarray(a["axes"], dtype=np.float64)
    b_axes = np.asarray(b["axes"], dtype=np.float64)
    cross_axes = np.cross(a_axes.T[:, None, :], b_axes.T[None, :, :]).reshape(-1, 3)
    candidate_axes = np.vstack([a_axes.T, b_axes.T, cross_axes])
    axis_norms = np.linalg.norm(candidate_axes, axis=1)
    candidate_axes = candidate_axes[axis_norms > 1e-8]
    candidate_axes /= np.linalg.norm(candidate_axes, axis=1, keepdims=True)
    center_delta = np.asarray(b["center"], dtype=np.float64) - np.asarray(a["center"], dtype=np.float64)
    center_distances = np.abs(candidate_axes @ center_delta)
    radius_a = np.abs(candidate_axes @ a_axes) @ np.asarray(a["half_extents"], dtype=np.float64)
    radius_b = np.abs(candidate_axes @ b_axes) @ np.asarray(b["half_extents"], dtype=np.float64)
    return bool(np.all(center_distances < radius_a + radius_b - float(eps)))


def build_collision_context(scene_objects: list[dict[str, Any]]) -> dict[str, Any]:
    """Precompute scene OBBs reused by validation collision checks."""
    return {
        "object_obbs": [
            _obb_from_bbox_transform(
                str(obj.get("object_id", index)),
                np.asarray(obj["canonical_aabb_object"], dtype=np.float64),
                np.asarray(obj["original_pose_world"], dtype=np.float64),
            )
            for index, obj in enumerate(scene_objects)
        ]
    }


def compute_collision_metrics(place_box: np.ndarray, context: dict[str, Any]) -> dict[str, Any]:
    """Check whether a predicted place box intersects any cached scene OBB."""
    bbox, transform = _place_box_to_bbox_and_transform(place_box)
    pred_obb = _obb_from_bbox_transform("prediction", bbox, transform)
    colliding_ids = [
        str(obj["object_id"])
        for obj in context["object_obbs"]
        if _obb_intersects(pred_obb, obj)
    ]
    return {
        "collision": bool(colliding_ids),
        "collision_object_count": len(colliding_ids),
        "collision_object_ids": colliding_ids,
    }


def compute_direction_hit(place_box: np.ndarray, context: Stage2ValidationContext) -> bool:
    """Check one predicted box against its language target relation."""
    predicted_relation = describe_spatial_relation(
        place_box_to_corners(place_box),
        context.reference_corners_world,
        context.camera_E_w2c,
        context.camera_K,
    )
    return bool(predicted_relation == context.target_relation)


def build_stage2_validation_contexts(items: list[Stage2IndexItem]) -> dict[str, Stage2ValidationContext]:
    """Load and cache direction/collision geometry only for validation items."""
    scene_cache: dict[tuple[str, str], dict[str, Any]] = {}
    contexts = {}
    for item in items:
        key = (item.source_name, item.sample_id)
        if key not in scene_cache:
            sample = load_sample_record(item.dataset_dir / "samples" / f"{item.sample_id}.json")
            camera_record = sample["camera"]
            camera = CameraParams(
                fx=float(camera_record["fx"]),
                fy=float(camera_record["fy"]),
                cx=float(camera_record["cx"]),
                cy=float(camera_record["cy"]),
                E_c2w=np.asarray(camera_record["E_c2w"], dtype=np.float64),
                img_w=int(camera_record["img_w"]),
                img_h=int(camera_record["img_h"]),
            )
            object_corners = {
                str(obj["obj_id"]): transform_points(
                    get_bbox_corners(np.asarray(obj["bbox3d_canonical"], dtype=np.float64)),
                    np.asarray(obj["pose_world"], dtype=np.float64),
                )
                for obj in sample["objects"]
            }
            placement_path = item.free_bbox_dir / "placements" / f"{item.sample_id}__placements.json"
            with placement_path.open("r", encoding="utf-8") as f:
                placement_payload = json.load(f)
            scene_cache[key] = {
                "camera_K": camera.K,
                "camera_E_w2c": camera.E_w2c,
                "object_corners": object_corners,
                "collision_context": build_collision_context(placement_payload.get("objects", [])),
            }

        scene = scene_cache[key]
        reference_corners = scene["object_corners"].get(item.reference_object_id)
        if reference_corners is None:
            raise ValueError(f"Reference object {item.reference_object_id} not found for {item.item_id}")
        contexts[item.item_id] = Stage2ValidationContext(
            target_relation=item.target_relation,
            reference_corners_world=reference_corners,
            camera_K=scene["camera_K"],
            camera_E_w2c=scene["camera_E_w2c"],
            collision_context=scene["collision_context"],
        )
    return contexts


def _is_heatmap_positive_color(colors: np.ndarray) -> np.ndarray:
    colors_i = np.asarray(colors, dtype=np.int16)
    return (colors_i[:, 0] == 255) & (colors_i[:, 2] == 30)


def _is_support_color(colors: np.ndarray) -> np.ndarray:
    colors_i = np.asarray(colors, dtype=np.int16)
    return (colors_i[:, 0] == 255) & (colors_i[:, 1] == 255) & (colors_i[:, 2] == 255)


def _quantize_world_points(points: np.ndarray, voxel_size_cm: float) -> np.ndarray:
    """Quantize world coordinates to stable voxel keys for PLY alignment."""
    return np.floor(np.asarray(points, dtype=np.float64) / float(voxel_size_cm) + 1e-4).astype(np.int64)


class LCBGPlaceNetStage2Dataset(Dataset):
    """Dataset for dense placement Stage 2 training."""

    def __init__(
        self,
        sources: list[Stage1DataSource] | None,
        split: str,
        val_fraction: float = 0.1,
        seed: int = 0,
        max_samples: int | None = None,
        items: list[Stage2IndexItem] | None = None,
        split_dir: str | Path | None = None,
    ) -> None:
        split_name = normalize_stage1_split(split)
        self.split = split_name
        if items is None:
            if sources is None:
                raise ValueError("sources must be provided when items is None")
            items = build_stage2_index(sources)
        items = list(items)
        if split_dir is not None:
            self.items = select_stage2_split_items(items, split_dir, split_name)
        else:
            if split_name == "test":
                raise ValueError("test split requires a fixed split_dir")
            rng = random.Random(int(seed))
            rng.shuffle(items)
            val_count = int(round(len(items) * float(val_fraction)))
            val_count = min(max(val_count, 1 if len(items) > 1 else 0), len(items))
            self.items = items[:val_count] if split_name == "valid" else items[val_count:]
        if max_samples is not None:
            self.items = self.items[: int(max_samples)]
        if not self.items:
            raise ValueError(f"No Stage 2 samples found for split={split_name}")

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        item = self.items[idx]
        points, colors = load_ply(item.voxel_point_cloud_path)
        support_points, support_colors = load_ply(item.support_mask_ply)
        heatmap_points, heatmap_colors = load_ply(item.direction_filtered_heatmap_ply)
        positive_points = heatmap_points[_is_heatmap_positive_color(heatmap_colors)]
        if len(points) == 0:
            raise ValueError(f"Invalid empty point data for {item.sample_id}")
        if len(positive_points) == 0:
            raise ValueError(f"No direction-filtered heatmap positives for {item.item_id}")
        image = np.asarray(Image.open(item.rgb_path).convert("RGB"), dtype=np.uint8)

        return {
            "item_id": item.item_id,
            "source_name": item.source_name,
            "sample_id": item.sample_id,
            "object_id": item.object_id,
            "cluster_id": item.cluster_id,
            "instruction": item.instruction,
            "points": points.astype(np.float32),
            "colors": colors.astype(np.uint8),
            "image": image,
            "camera_K": item.camera_K,
            "camera_E_w2c": item.camera_E_w2c,
            "support_points": support_points[_is_support_color(support_colors)].astype(np.float32),
            "heatmap_positive_points": positive_points.astype(np.float32),
            "source_box_gt": item.source_box_gt,
            "place_box_gt": item.place_box_gt,
        }


def _read_split_records(split_dir: str | Path, split: str) -> list[dict[str, Any]]:
    """Read one fixed split file generated for Stage 1 labels."""
    split_name = normalize_stage1_split(split)
    path = Path(split_dir) / f"{split_name}.json"
    if not path.exists():
        raise FileNotFoundError(f"Split file not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    if payload.get("schema_version") != STAGE1_SPLIT_SCHEMA_VERSION:
        raise ValueError(f"Unsupported split schema_version in {path}: {payload.get('schema_version')}")
    if payload.get("split") not in STAGE1_SPLIT_NAME_SET:
        raise ValueError(f"Invalid split declaration in {path}: {payload.get('split')}")
    return list(payload.get("items", []))


def select_stage2_split_items(
    items: list[Stage2IndexItem],
    split_dir: str | Path,
    split: str,
) -> list[Stage2IndexItem]:
    """Select Stage 2 items using the shared Stage 1 label split files."""
    records = _read_split_records(split_dir, split)
    item_by_id = {item.item_id: item for item in items}
    selected = []
    missing = []
    for record in records:
        item = item_by_id.get(str(record["item_id"]))
        if item is None:
            missing.append(str(record["item_id"]))
            continue
        selected.append(item)
    if missing:
        preview = ", ".join(missing[:5])
        raise ValueError(f"{len(missing)} split items are missing from the current Stage 2 index: {preview}")
    return selected


def stage2_collate(batch: list[dict[str, Any]], voxel_size_cm: float = 1.0) -> dict[str, Any]:
    """Collate Stage 2 active voxel samples into one sparse-conv batch."""
    features = []
    sparse_coords = []
    world_coords = []
    coords_norm = []
    batch_indices = []
    support_masks = []
    source_boxes = []
    place_boxes = []
    positive_points = []
    positive_batch_indices = []
    scene_min = []
    scene_max = []
    instructions = []
    item_ids = []
    source_names = []
    sample_ids = []
    object_ids = []
    cluster_ids = []
    images = []
    camera_k = []
    camera_e_w2c = []
    image_hw = []
    spatial_max = np.zeros(3, dtype=np.int64)

    for batch_idx, item in enumerate(batch):
        points = np.asarray(item["points"], dtype=np.float32)
        colors = np.asarray(item["colors"], dtype=np.float32) / 255.0
        point_min = points.min(axis=0)
        point_max = points.max(axis=0)
        extent = np.maximum(point_max - point_min, 1e-4)
        point_norm = (points - point_min) / extent

        voxel_keys = _quantize_world_points(points, voxel_size_cm)
        shifted_keys = voxel_keys - voxel_keys.min(axis=0, keepdims=True)
        spatial_max = np.maximum(spatial_max, shifted_keys.max(axis=0) + 1)
        batch_col = np.full((len(points), 1), batch_idx, dtype=np.int32)

        support_keys = {tuple(key) for key in _quantize_world_points(item["support_points"], voxel_size_cm)}
        support_mask = np.asarray([tuple(key) in support_keys for key in voxel_keys], dtype=bool)

        positives = np.asarray(item["heatmap_positive_points"], dtype=np.float32)
        features.append(np.concatenate([point_norm, colors], axis=1).astype(np.float32))
        sparse_coords.append(np.concatenate([batch_col, shifted_keys.astype(np.int32)], axis=1))
        world_coords.append(points)
        coords_norm.append(point_norm.astype(np.float32))
        batch_indices.append(np.full(len(points), batch_idx, dtype=np.int64))
        support_masks.append(support_mask)
        positive_points.append(positives)
        positive_batch_indices.append(np.full(len(positives), batch_idx, dtype=np.int64))
        scene_min.append(point_min.astype(np.float32))
        scene_max.append(point_max.astype(np.float32))
        source_boxes.append(np.asarray(item["source_box_gt"], dtype=np.float32))
        place_boxes.append(np.asarray(item["place_box_gt"], dtype=np.float32))
        instructions.append(str(item["instruction"]))
        item_ids.append(str(item["item_id"]))
        source_names.append(str(item["source_name"]))
        sample_ids.append(str(item["sample_id"]))
        object_ids.append(str(item["object_id"]))
        cluster_ids.append(int(item["cluster_id"]))
        image = np.asarray(item["image"], dtype=np.uint8)
        images.append(image)
        camera_k.append(np.asarray(item["camera_K"], dtype=np.float32))
        camera_e_w2c.append(np.asarray(item["camera_E_w2c"], dtype=np.float32))
        image_hw.append(np.asarray(image.shape[:2], dtype=np.float32))

    return {
        "features": torch.from_numpy(np.concatenate(features, axis=0)),
        "sparse_coords": torch.from_numpy(np.concatenate(sparse_coords, axis=0)),
        "spatial_shape": [int(x) for x in spatial_max.tolist()],
        "world_coords": torch.from_numpy(np.concatenate(world_coords, axis=0)),
        "coords_norm": torch.from_numpy(np.concatenate(coords_norm, axis=0)),
        "batch_indices": torch.from_numpy(np.concatenate(batch_indices, axis=0)),
        "support_masks": torch.from_numpy(np.concatenate(support_masks, axis=0)),
        "heatmap_positive_points": torch.from_numpy(np.concatenate(positive_points, axis=0)),
        "heatmap_positive_batch_indices": torch.from_numpy(np.concatenate(positive_batch_indices, axis=0)),
        "source_box_gt": torch.from_numpy(np.stack(source_boxes, axis=0)),
        "place_box_gt": torch.from_numpy(np.stack(place_boxes, axis=0)),
        "scene_min": torch.from_numpy(np.stack(scene_min, axis=0)),
        "scene_max": torch.from_numpy(np.stack(scene_max, axis=0)),
        "instructions": instructions,
        "item_ids": item_ids,
        "source_names": source_names,
        "sample_ids": sample_ids,
        "object_ids": object_ids,
        "cluster_ids": cluster_ids,
        "images": images,
        "camera_K": torch.from_numpy(np.stack(camera_k, axis=0)),
        "camera_E_w2c": torch.from_numpy(np.stack(camera_e_w2c, axis=0)),
        "image_hw": torch.from_numpy(np.stack(image_hw, axis=0)),
        "batch_size": len(batch),
    }


def build_dense_heatmap_targets(
    world_coords: torch.Tensor,
    batch_indices: torch.Tensor,
    positive_points: torch.Tensor,
    positive_batch_indices: torch.Tensor,
    support_masks: torch.Tensor,
    batch_size: int,
    sigma: float,
    chunk_size: int = 4096,
) -> torch.Tensor:
    """Build dense Gaussian heatmap targets and zero labels outside support."""
    targets = world_coords.new_zeros((world_coords.shape[0],))
    sigma2 = 2.0 * float(sigma) * float(sigma)
    for batch_idx in range(int(batch_size)):
        voxel_idx = torch.nonzero(batch_indices == batch_idx, as_tuple=False).flatten()
        pos = positive_points[positive_batch_indices == batch_idx]
        if len(voxel_idx) == 0 or len(pos) == 0:
            continue
        coords = world_coords[voxel_idx]
        values = []
        for start in range(0, len(coords), int(chunk_size)):
            chunk = coords[start : start + int(chunk_size)]
            dist2 = torch.cdist(chunk, pos).pow(2).min(dim=1).values
            values.append(torch.exp(-dist2 / sigma2))
        targets[voxel_idx] = torch.cat(values, dim=0)
    return targets * support_masks.to(dtype=targets.dtype)


def focal_loss_with_logits(
    logits: torch.Tensor,
    targets: torch.Tensor,
    alpha: float = 0.25,
    gamma: float = 2.0,
) -> torch.Tensor:
    """Binary focal loss that accepts soft heatmap targets."""
    prob = torch.sigmoid(logits)
    bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    pt = prob * targets + (1.0 - prob) * (1.0 - targets)
    alpha_t = float(alpha) * targets + (1.0 - float(alpha)) * (1.0 - targets)
    return (alpha_t * (1.0 - pt).pow(float(gamma)) * bce).mean()


def nearest_voxel_per_batch(
    coords: torch.Tensor,
    batch_indices: torch.Tensor,
    targets: torch.Tensor,
) -> torch.Tensor:
    """Find the nearest active voxel to one target point per batch item."""
    indices = []
    for batch_idx in range(targets.shape[0]):
        idx = torch.nonzero(batch_indices == batch_idx, as_tuple=False).flatten()
        if len(idx) == 0:
            raise ValueError(f"batch item {batch_idx} has no active voxels")
        dist2 = (coords[idx] - targets[batch_idx]).pow(2).sum(dim=1)
        indices.append(idx[torch.argmin(dist2)])
    return torch.stack(indices, dim=0)


def compute_stage2_loss(
    outputs: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
    cfg: dict[str, Any],
) -> dict[str, torch.Tensor]:
    """Compute Stage 2 training loss and detached logging terms."""
    loss_cfg = cfg["loss"]
    sigma = float(cfg["data"].get("heatmap_sigma_voxels", 2.0)) * float(cfg["data"]["voxel_size_cm"])
    heatmap_gt = build_dense_heatmap_targets(
        world_coords=batch["world_coords"],
        batch_indices=batch["batch_indices"],
        positive_points=batch["heatmap_positive_points"],
        positive_batch_indices=batch["heatmap_positive_batch_indices"],
        support_masks=batch["support_masks"],
        batch_size=int(batch["batch_size"]),
        sigma=sigma,
    )
    heat_loss_type = str(loss_cfg.get("heatmap_loss", "focal")).lower()
    if heat_loss_type == "bce":
        heat_loss = F.binary_cross_entropy_with_logits(outputs["placement_heatmap_logits"], heatmap_gt)
    else:
        focal_cfg = loss_cfg.get("focal", {})
        heat_loss = focal_loss_with_logits(
            outputs["placement_heatmap_logits"],
            heatmap_gt,
            alpha=float(focal_cfg.get("alpha", 0.25)),
            gamma=float(focal_cfg.get("gamma", 2.0)),
        )

    place_gt = batch["place_box_gt"]
    bottom_gt = place_gt[:, 0:3].clone()
    bottom_gt[:, 2] = place_gt[:, 2] - place_gt[:, 5] * 0.5
    pos_idx = nearest_voxel_per_batch(batch["world_coords"], batch["batch_indices"], bottom_gt)
    offset_gt = bottom_gt - batch["world_coords"][pos_idx]
    offset_loss = F.smooth_l1_loss(outputs["bottom_offset"][pos_idx], offset_gt)
    center_loss = F.smooth_l1_loss(outputs["place_box"][:, 0:3], place_gt[:, 0:3])
    center_total = offset_loss + center_loss
    size_loss = F.smooth_l1_loss(outputs["size_pred"], place_gt[:, 3:6])

    yaw_ratio = torch.abs(place_gt[:, 3] - place_gt[:, 4]) / torch.minimum(
        place_gt[:, 3],
        place_gt[:, 4],
    ).clamp_min(1e-6)
    yaw_sensitive = yaw_ratio >= float(loss_cfg.get("yaw_sensitive_ratio", 0.25))
    if torch.any(yaw_sensitive):
        yaw_target = torch.stack([torch.sin(place_gt[:, 6]), torch.cos(place_gt[:, 6])], dim=-1)
        yaw_pred = F.normalize(outputs["yaw_sincos"][outputs["best_indices"]], dim=-1, eps=1e-6)
        yaw_loss = F.smooth_l1_loss(yaw_pred[yaw_sensitive], yaw_target[yaw_sensitive])
    else:
        yaw_loss = outputs["placement_heatmap_logits"].sum() * 0.0

    src_loss, src_terms = source_box_loss(
        outputs["source_box"],
        batch["source_box_gt"],
        lambda_center=float(loss_cfg["source"]["lambda_center"]),
        lambda_size=float(loss_cfg["source"]["lambda_size"]),
        lambda_iou=float(loss_cfg["source"]["lambda_iou"]),
    )
    total = (
        float(loss_cfg["lambda_heat"]) * heat_loss
        + float(loss_cfg["lambda_center"]) * center_total
        + float(loss_cfg["lambda_size"]) * size_loss
        + float(loss_cfg["lambda_yaw"]) * yaw_loss
        + float(loss_cfg["lambda_src"]) * src_loss
    )
    terms = {
        "loss": total,
        "loss_heat": heat_loss.detach(),
        "loss_center": center_total.detach(),
        "loss_offset": offset_loss.detach(),
        "loss_place_center": center_loss.detach(),
        "loss_size": size_loss.detach(),
        "loss_yaw": yaw_loss.detach(),
        "loss_src": src_loss.detach(),
        "heatmap_gt_max": heatmap_gt.max().detach(),
        "heatmap_gt_support_mean": heatmap_gt[batch["support_masks"]].mean().detach()
        if torch.any(batch["support_masks"])
        else heatmap_gt.mean().detach(),
    }
    terms.update(src_terms)
    return terms


def compute_stage2_task_metric_sums(
    outputs: dict[str, torch.Tensor],
    batch: dict[str, Any],
    cfg: dict[str, Any],
) -> dict[str, float]:
    """Compute additive task metric statistics for one validation batch."""
    place_pred = outputs["place_box"].detach().cpu().numpy()
    place_gt = batch["place_box_gt"].detach().cpu().numpy()
    contexts = batch["validation_contexts"]
    size_iou_threshold = float(cfg.get("validation", {}).get("size_iou_threshold", 0.8))
    yaw_sensitive_ratio = float(cfg["loss"].get("yaw_sensitive_ratio", 0.25))

    direction_hits = []
    collision_free = []
    size_ious = []
    task_success = []
    yaw_errors = []
    for pred, gt, context in zip(place_pred, place_gt, contexts):
        size_metrics = compute_size_metrics(pred, gt, size_iou_threshold)
        direction_hit = compute_direction_hit(pred, context)
        collision = compute_collision_metrics(pred, context.collision_context)["collision"]
        yaw_metrics = compute_yaw_metrics(pred, gt, yaw_sensitive_ratio)
        direction_hits.append(direction_hit)
        collision_free.append(not collision)
        size_ious.append(size_metrics["size_iou"])
        task_success.append(compute_task_success(direction_hit, collision, size_metrics["size_correct"]))
        if yaw_metrics["yaw_sensitive"]:
            yaw_errors.append(yaw_metrics["yaw_error_deg"])

    return {
        "sample_count": float(len(place_pred)),
        "direction_hit_sum": float(sum(direction_hits)),
        "collision_free_sum": float(sum(collision_free)),
        "size_iou_sum": float(sum(size_ious)),
        "yaw_error_deg_sum": float(sum(yaw_errors)),
        "yaw_sensitive_count": float(len(yaw_errors)),
        "task_success_sum": float(sum(task_success)),
    }


def _make_loader(
    dataset: Dataset,
    cfg: dict[str, Any],
    shuffle: bool,
    sampler: DistributedSampler | None = None,
) -> DataLoader:
    data_cfg = cfg["data"]
    train_cfg = cfg["training"]
    return DataLoader(
        dataset,
        batch_size=int(train_cfg["batch_size"]),
        shuffle=shuffle if sampler is None else False,
        sampler=sampler,
        num_workers=int(train_cfg.get("num_workers", 0)),
        collate_fn=lambda batch: stage2_collate(batch, voxel_size_cm=float(data_cfg["voxel_size_cm"])),
        pin_memory=bool(train_cfg.get("pin_memory", True)),
    )


def load_stage1_weights(model: LCBGPlaceNetDensePlacement, checkpoint_path: str | Path, device: torch.device) -> list[str]:
    """Initialize shared Stage 1 modules from a Stage 1 checkpoint."""
    checkpoint = torch.load(Path(checkpoint_path), map_location=device)
    state_dict = checkpoint["model"] if isinstance(checkpoint, dict) and "model" in checkpoint else checkpoint
    model_state = model.state_dict()
    compatible = {
        key: value
        for key, value in state_dict.items()
        if key in model_state and tuple(value.shape) == tuple(model_state[key].shape)
    }
    skipped = sorted(key for key in state_dict if key not in compatible)
    model.load_state_dict(compatible, strict=False)
    return skipped


def train_stage2(
    cfg: dict[str, Any],
    stage1_checkpoint: str | Path | None = None,
    resume_checkpoint: str | Path | None = None,
    max_steps: int | None = None,
    max_train_samples: int | None = None,
    max_val_samples: int | None = None,
) -> None:
    """Run Stage 2 training and validation."""
    distributed, rank, world_size, local_rank = _init_distributed()
    try:
        set_seed(int(cfg["training"].get("seed", 0)) + rank)
        output_dir = Path(cfg["training"]["output_dir"])
        if _is_main_process(rank):
            output_dir.mkdir(parents=True, exist_ok=True)
        if distributed:
            dist.barrier(device_ids=[local_rank])
        device = _resolve_device(cfg, distributed=distributed, local_rank=local_rank)

        sources = build_sources_from_config(cfg)
        data_cfg = cfg["data"]
        all_items = build_stage2_index(sources)
        split_dir = data_cfg.get("split_dir")
        valid_fraction = float(data_cfg.get("valid_fraction", data_cfg.get("val_fraction", 0.1)))
        train_set = LCBGPlaceNetStage2Dataset(
            sources=None,
            split="train",
            val_fraction=valid_fraction,
            seed=int(data_cfg.get("split_seed", 0)),
            max_samples=max_train_samples or data_cfg.get("max_train_samples"),
            items=all_items,
            split_dir=split_dir,
        )
        val_set = LCBGPlaceNetStage2Dataset(
            sources=None,
            split="valid",
            val_fraction=valid_fraction,
            seed=int(data_cfg.get("split_seed", 0)),
            max_samples=max_val_samples or data_cfg.get("max_valid_samples", data_cfg.get("max_val_samples")),
            items=all_items,
            split_dir=split_dir,
        )
        validation_contexts = build_stage2_validation_contexts(val_set.items)

        model = LCBGPlaceNetDensePlacement(cfg["model"]).to(device)
        init_checkpoint = stage1_checkpoint or cfg["training"].get("stage1_pretrained_checkpoint")
        if resume_checkpoint is None and init_checkpoint:
            skipped = load_stage1_weights(model, init_checkpoint, device)
            if _is_main_process(rank):
                print(f"Initialized Stage 2 shared modules from Stage 1 checkpoint: {init_checkpoint}")
                if skipped:
                    preview = ", ".join(skipped[:8])
                    print(f"Skipped {len(skipped)} incompatible Stage 1 keys: {preview}")

        if distributed:
            model = DistributedDataParallel(model, device_ids=[local_rank], output_device=local_rank)
        trainable_model = model.module if distributed else model
        optimizer = torch.optim.AdamW(
            [p for p in trainable_model.parameters() if p.requires_grad],
            lr=float(cfg["training"]["lr"]),
            weight_decay=float(cfg["training"].get("weight_decay", 0.0)),
        )

        train_sampler = DistributedSampler(train_set, num_replicas=world_size, rank=rank, shuffle=True) if distributed else None
        val_sampler = DistributedSampler(val_set, num_replicas=world_size, rank=rank, shuffle=False) if distributed else None
        train_loader = _make_loader(train_set, cfg, shuffle=True, sampler=train_sampler)
        val_loader = _make_loader(val_set, cfg, shuffle=False, sampler=val_sampler)
        metrics_path = output_dir / "metrics.jsonl"
        best_task_success_rate = -math.inf
        global_step = 0
        start_epoch = 0
        show_progress = _is_main_process(rank) and bool(cfg["training"].get("progress_bar", True))

        if resume_checkpoint is not None:
            resume_path = Path(resume_checkpoint)
            checkpoint = torch.load(resume_path, map_location=device)
            trainable_model.load_state_dict(checkpoint["model"])
            optimizer.load_state_dict(checkpoint["optimizer"])
            start_epoch = int(checkpoint["epoch"]) + 1
            global_step = int(checkpoint["step"])
            best_task_success_rate = float(checkpoint.get("best_task_success_rate", -math.inf))
            if _is_main_process(rank):
                print(
                    f"Resumed Stage 2 from {resume_path} at epoch={start_epoch}, "
                    f"step={global_step}, best_task_success_rate={best_task_success_rate:.6f}"
                )

        if _is_main_process(rank):
            print(f"Training Stage 2 on {world_size} GPU process(es); per-GPU batch_size={cfg['training']['batch_size']}")

        for epoch in range(start_epoch, int(cfg["training"]["epochs"])):
            if train_sampler is not None:
                train_sampler.set_epoch(epoch)
            model.train()
            train_iter = _progress(train_loader, show_progress, desc=f"stage2 train epoch {epoch}", total=len(train_loader))
            for batch in train_iter:
                global_step += 1
                batch = move_batch_to_device(batch, device)
                optimizer.zero_grad(set_to_none=True)
                outputs = model(batch)
                losses = compute_stage2_loss(outputs, batch, cfg)
                losses["loss"].backward()
                torch.nn.utils.clip_grad_norm_(trainable_model.parameters(), float(cfg["training"].get("grad_clip_norm", 1.0)))
                optimizer.step()

                if show_progress:
                    train_iter.set_postfix(loss=f"{float(losses['loss'].detach().cpu()):.4f}", step=global_step)

                if _is_main_process(rank) and global_step % int(cfg["training"].get("log_every", 20)) == 0:
                    log_payload = {
                        "split": "train",
                        "epoch": epoch,
                        "step": global_step,
                        **{k: float(v.detach().cpu()) for k, v in losses.items()},
                    }
                    _append_jsonl(metrics_path, log_payload)
                    print(json.dumps(log_payload, ensure_ascii=False))

                if max_steps is not None and global_step >= int(max_steps):
                    break

            val_metrics = evaluate_stage2(
                model,
                val_loader,
                cfg,
                device,
                validation_contexts,
                show_progress=show_progress,
                distributed=distributed,
            )
            if _is_main_process(rank):
                val_payload = {"split": "valid", "epoch": epoch, "step": global_step, **val_metrics}
                _append_jsonl(metrics_path, val_payload)
                print(json.dumps(val_payload, ensure_ascii=False))

                checkpoint = {
                    "model": trainable_model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "epoch": epoch,
                    "step": global_step,
                    "best_task_success_rate": best_task_success_rate,
                    "config": cfg,
                }
                current_success_rate = val_metrics.get("task_success_rate", -math.inf)
                if current_success_rate > best_task_success_rate:
                    best_task_success_rate = current_success_rate
                    checkpoint["best_task_success_rate"] = best_task_success_rate
                    torch.save(checkpoint, output_dir / "best.pt")
                torch.save(checkpoint, output_dir / "last.pt")

            if distributed:
                dist.barrier(device_ids=[local_rank])
            if max_steps is not None and global_step >= int(max_steps):
                break
    finally:
        if distributed and dist.is_initialized():
            dist.destroy_process_group()


@torch.no_grad()
def evaluate_stage2(
    model: LCBGPlaceNetDensePlacement | DistributedDataParallel,
    loader: DataLoader,
    cfg: dict[str, Any],
    device: torch.device,
    validation_contexts: dict[str, Stage2ValidationContext],
    show_progress: bool = False,
    distributed: bool = False,
) -> dict[str, float]:
    """Evaluate Stage 2 on one dataloader."""
    model.eval()
    totals = {
        "loss_sum": 0.0,
        "sample_count": 0.0,
        "direction_hit_sum": 0.0,
        "collision_free_sum": 0.0,
        "size_iou_sum": 0.0,
        "yaw_error_deg_sum": 0.0,
        "yaw_sensitive_count": 0.0,
        "task_success_sum": 0.0,
    }
    val_iter = _progress(loader, show_progress, desc="stage2 valid", total=len(loader))
    for batch in val_iter:
        batch["validation_contexts"] = [validation_contexts[item_id] for item_id in batch["item_ids"]]
        batch = move_batch_to_device(batch, device)
        outputs = model(batch)
        losses = compute_stage2_loss(outputs, batch, cfg)
        metric_sums = compute_stage2_task_metric_sums(outputs, batch, cfg)
        totals["loss_sum"] += float(losses["loss"].detach().cpu()) * metric_sums["sample_count"]
        for key, value in metric_sums.items():
            totals[key] += value

    if distributed:
        keys = list(totals)
        values = torch.tensor([totals[key] for key in keys], dtype=torch.float64, device=device)
        dist.all_reduce(values, op=dist.ReduceOp.SUM)
        totals = {key: float(value.cpu()) for key, value in zip(keys, values)}

    sample_count = max(totals["sample_count"], 1.0)
    yaw_count = max(totals["yaw_sensitive_count"], 1.0)
    return {
        "loss": totals["loss_sum"] / sample_count,
        "direction_hit_rate": totals["direction_hit_sum"] / sample_count,
        "collision_free_rate": totals["collision_free_sum"] / sample_count,
        "size_iou": totals["size_iou_sum"] / sample_count,
        "yaw_error_deg": totals["yaw_error_deg_sum"] / yaw_count,
        "task_success_rate": totals["task_success_sum"] / sample_count,
    }

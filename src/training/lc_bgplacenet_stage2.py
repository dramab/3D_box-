"""
LC-BGPlaceNet Stage 2 training utilities.

Stage 2 trains either SPACE-Former set prediction or the Direct-Box 1Q baseline
from canonical active voxels, language and free_bbox center/yaw supervision.
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
from scipy.optimize import linear_sum_assignment
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler

from src.annotation.auto_label import describe_spatial_relation
from src.annotation.free_bbox.geometry import get_bbox_corners, transform_points
from src.annotation.free_bbox.io_utils import load_ply
from src.datasets.canonical import CameraParams, load_sample_record
from src.models.lc_bgplacenet.direct_box import DirectBox1QStage2
from src.models.lc_bgplacenet.stage2 import (
    NUM_QUERIES,
    NUM_YAW_BINS,
    SPACEFormerStage2,
    aggregate_sparse_mask,
    yaw_bin_angles,
)
from src.placement_metrics import (
    build_placement_evaluation_box,
    compute_aabb_iou_3d,
    compute_size_iou as _compute_size_iou,
    compute_supported_and_stable,
    compute_yaw_valid_at_matched_center,
    placement_success,
    quantize_occupied_points,
)
from src.training.lc_bgplacenet_stage1 import (
    STAGE1_SPLIT_NAME_SET,
    STAGE1_SUPPORTED_SPLIT_SCHEMA_VERSIONS,
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
DIRECT_BOX_MODEL_TYPE = "direct_box_1q"


def build_stage2_model(model_cfg: dict[str, Any]) -> torch.nn.Module:
    """Build the configured Stage 2 architecture."""
    model_type = str(model_cfg.get("type", "space_former")).lower()
    if model_type == "space_former":
        return SPACEFormerStage2(model_cfg)
    if model_type == DIRECT_BOX_MODEL_TYPE:
        return DirectBox1QStage2(model_cfg)
    raise ValueError(f"Unsupported Stage 2 model type: {model_type}")


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
    yaw_set_npz: Path
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
            yaw_set_path = _yaw_set_path(source.free_bbox_dir, raw_heatmap)
            support_mask_path = _resolve_free_bbox_path(source.free_bbox_dir, support_mask)
            if not direction_heatmap.exists() or not support_mask_path.exists() or not yaw_set_path.exists():
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
                    yaw_set_npz=yaw_set_path,
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


def _yaw_set_path(free_bbox_root: Path, raw_heatmap_path: str | os.PathLike[str]) -> Path:
    """Map one cluster heatmap path to its saved center/yaw-set NPZ."""
    name = Path(raw_heatmap_path).name
    if not name.endswith("__heatmap.ply"):
        raise ValueError(f"Unexpected heatmap filename: {name}")
    return free_bbox_root / "yaw_sets" / name.replace("__heatmap.ply", "__yaw_set.npz")


def _place_box_from_placement(placement: dict[str, Any]) -> np.ndarray:
    """Convert free_bbox placement metadata to (x, y, z, dx, dy, dz, yaw)."""
    center = np.asarray(placement["center_world"], dtype=np.float32)
    dims = np.asarray(placement["yaw_only_dimensions"], dtype=np.float32)
    yaw = np.deg2rad(float(placement["yaw_degrees"]))
    return np.concatenate([center, dims, np.asarray([yaw], dtype=np.float32)]).astype(np.float32)


def compute_size_iou(pred_dims: np.ndarray, gt_dims: np.ndarray) -> float:
    """Compute dimension-only volume IoU while preserving dimension order."""
    return _compute_size_iou(pred_dims, gt_dims)


def compute_size_metrics(
    place_box: np.ndarray,
    place_box_gt: np.ndarray,
    threshold: float,
) -> dict[str, Any]:
    """Compute dimension IoU and its thresholded correctness flag."""
    size_iou = compute_size_iou(place_box[3:6], place_box_gt[3:6])
    return {"size_iou": size_iou, "size_correct": bool(size_iou >= float(threshold))}


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
    """按 canonical ``floor(world / voxel_size)`` 规则生成 voxel key。"""
    return np.floor(np.asarray(points, dtype=np.float64) / float(voxel_size_cm)).astype(np.int64)


def _row_membership(rows: np.ndarray, candidates: np.ndarray) -> np.ndarray:
    """Vectorize exact membership checks for fixed-width integer rows."""
    rows = np.ascontiguousarray(rows)
    candidates = np.ascontiguousarray(candidates, dtype=rows.dtype)
    row_dtype = np.dtype((np.void, rows.dtype.itemsize * rows.shape[1]))
    row_keys = rows.view(row_dtype).reshape(-1)
    candidate_keys = candidates.view(row_dtype).reshape(-1)
    return np.isin(row_keys, candidate_keys)


def build_space_former_targets(
    yaw_set_path: str | Path,
    direction_positive_points: np.ndarray,
    voxel_size_cm: float = 1.0,
) -> dict[str, np.ndarray]:
    """按 canonical voxel key 保留全部方向有效中心及其合法 yaw。"""
    with np.load(yaw_set_path) as yaw_set:
        centers = np.asarray(yaw_set["bottom_center_world"], dtype=np.float32)
        mask24 = np.asarray(yaw_set["valid_yaw_mask"], dtype=bool)
    if mask24.shape != (len(centers), 24):
        raise ValueError(f"Invalid yaw mask shape in {yaw_set_path}: {mask24.shape}")

    center_keys = _quantize_world_points(centers, voxel_size_cm)
    positive_keys = _quantize_world_points(direction_positive_points, voxel_size_cm)
    if len(np.unique(center_keys, axis=0)) != len(center_keys):
        raise ValueError(f"Yaw-set contains duplicate canonical voxel keys: {yaw_set_path}")
    positive_in_yaw_set = _row_membership(positive_keys, center_keys)
    if not np.all(positive_in_yaw_set):
        raise ValueError(
            f"Direction-positive/yaw-center voxel alignment failed in {yaw_set_path}: "
            f"unmatched={int(np.count_nonzero(~positive_in_yaw_set))}"
        )
    keep = _row_membership(center_keys, positive_keys)
    mask12 = mask24[:, :12] | mask24[:, 12:]
    keep &= mask12.any(axis=1)
    centers = centers[keep]
    mask12 = mask12[keep]
    if len(centers) == 0:
        raise ValueError(f"No direction-valid center/yaw targets remain in {yaw_set_path}")

    quality = 0.5 + 0.5 * mask12.sum(axis=1).astype(np.float32) / float(NUM_YAW_BINS)
    return {
        "gt_bottom_centers": centers.astype(np.float32),
        "gt_yaw_masks": mask12.astype(bool),
        "gt_affordance_quality": quality.astype(np.float32),
    }


class LCBGPlaceNetStage2Dataset(Dataset):
    """Dataset for SPACE-Former Stage 2 set-prediction training."""

    def __init__(
        self,
        sources: list[Stage1DataSource] | None,
        split: str,
        val_fraction: float = 0.1,
        seed: int = 0,
        max_samples: int | None = None,
        items: list[Stage2IndexItem] | None = None,
        split_dir: str | Path | None = None,
        voxel_size_cm: float = 1.0,
    ) -> None:
        split_name = normalize_stage1_split(split)
        self.split = split_name
        self.voxel_size_cm = float(voxel_size_cm)
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
        set_targets = build_space_former_targets(
            item.yaw_set_npz,
            positive_points,
            voxel_size_cm=self.voxel_size_cm,
        )

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
            **set_targets,
        }


def _read_split_records(split_dir: str | Path, split: str) -> list[dict[str, Any]]:
    """Read one fixed split file generated for Stage 1 labels."""
    split_name = normalize_stage1_split(split)
    path = Path(split_dir) / f"{split_name}.json"
    if not path.exists():
        raise FileNotFoundError(f"Split file not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    if payload.get("schema_version") not in STAGE1_SUPPORTED_SPLIT_SCHEMA_VERSIONS:
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
    max_target_count = max(len(item["gt_bottom_centers"]) for item in batch)
    features = []
    sparse_coords = []
    world_coords = []
    coords_norm = []
    batch_indices = []
    support_masks = []
    source_boxes = []
    place_boxes = []
    gt_bottom_centers = []
    gt_yaw_masks = []
    gt_affordance_quality = []
    gt_valid_masks = []
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

        support_keys = _quantize_world_points(item["support_points"], voxel_size_cm)
        support_is_active = _row_membership(support_keys, voxel_keys)
        if not np.all(support_is_active):
            raise ValueError(
                f"Active support is not a subset of the input point cloud: "
                f"item_id={item['item_id']}, unmatched={int(np.count_nonzero(~support_is_active))}"
            )
        support_mask = _row_membership(voxel_keys, support_keys)

        positives = np.asarray(item["heatmap_positive_points"], dtype=np.float32)
        positive_keys = _quantize_world_points(positives, voxel_size_cm)
        positive_is_support = _row_membership(positive_keys, support_keys)
        if not np.all(positive_is_support):
            raise ValueError(
                f"Heatmap positive is not an active support voxel: "
                f"item_id={item['item_id']}, unmatched={int(np.count_nonzero(~positive_is_support))}"
            )
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
        target_centers = np.asarray(item["gt_bottom_centers"], dtype=np.float32)
        target_yaw = np.asarray(item["gt_yaw_masks"], dtype=bool)
        target_quality = np.asarray(item["gt_affordance_quality"], dtype=np.float32)
        target_count = len(target_centers)
        centers_pad = np.zeros((max_target_count, 3), dtype=np.float32)
        yaw_pad = np.zeros((max_target_count, NUM_YAW_BINS), dtype=bool)
        quality_pad = np.zeros((max_target_count,), dtype=np.float32)
        valid_pad = np.zeros((max_target_count,), dtype=bool)
        centers_pad[:target_count] = target_centers[:target_count]
        yaw_pad[:target_count] = target_yaw[:target_count]
        quality_pad[:target_count] = target_quality[:target_count]
        valid_pad[:target_count] = True
        gt_bottom_centers.append(centers_pad)
        gt_yaw_masks.append(yaw_pad)
        gt_affordance_quality.append(quality_pad)
        gt_valid_masks.append(valid_pad)
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
        "gt_bottom_centers": torch.from_numpy(np.stack(gt_bottom_centers, axis=0)),
        "gt_yaw_masks": torch.from_numpy(np.stack(gt_yaw_masks, axis=0)),
        "gt_affordance_quality": torch.from_numpy(np.stack(gt_affordance_quality, axis=0)),
        "gt_valid_mask": torch.from_numpy(np.stack(gt_valid_masks, axis=0)),
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
    """Build Gaussian targets on supplied world coordinates and zero labels outside support."""
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


def compute_p3_gt_point_coverage(
    region_coords: torch.Tensor,
    region_logits: torch.Tensor,
    voxel_origins: torch.Tensor,
    positive_points: torch.Tensor,
    positive_batch_indices: torch.Tensor,
    voxel_size_cm: float,
    batch_size: int,
    num_region_cells: int,
) -> torch.Tensor:
    """Return macro-average GT-point coverage of selected P3 cells."""
    spatial_shape = region_coords[:, 1:].amax(dim=0) + 1

    def key(xyz: torch.Tensor) -> torch.Tensor:
        return (xyz[:, 0] * spatial_shape[1] + xyz[:, 1]) * spatial_shape[2] + xyz[:, 2]

    sample_coverages = []
    for batch_index in range(int(batch_size)):
        region_idx = torch.nonzero(region_coords[:, 0] == batch_index, as_tuple=False).flatten()
        positives = positive_points[positive_batch_indices == batch_index]
        if len(region_idx) == 0 or len(positives) == 0:
            sample_coverages.append(region_logits.new_zeros(()))
            continue
        selected_count = min(int(num_region_cells), len(region_idx))
        selected_local = torch.topk(region_logits[region_idx], k=selected_count).indices
        selected_keys = key(region_coords[region_idx[selected_local], 1:])
        positive_coords = torch.floor(
            (positives - voxel_origins[batch_index]) / (4.0 * float(voxel_size_cm))
        ).long()
        in_bounds = ((positive_coords >= 0) & (positive_coords < spatial_shape)).all(dim=-1)
        hits = torch.zeros(len(positives), dtype=torch.bool, device=region_coords.device)
        hits[in_bounds] = torch.isin(key(positive_coords[in_bounds]), selected_keys)
        sample_coverages.append(hits.to(region_logits.dtype).mean())
    return torch.stack(sample_coverages).mean()


def _box_corners_from_rotation(
    bottom_centers: torch.Tensor,
    sizes: torch.Tensor,
    rotations: torch.Tensor,
) -> torch.Tensor:
    """Build ordered corners for aligned leading `(center,size,rotation)` shapes."""
    signs = torch.tensor(
        [
            [-1, -1, -1], [-1, -1, 1], [-1, 1, -1], [-1, 1, 1],
            [1, -1, -1], [1, -1, 1], [1, 1, -1], [1, 1, 1],
        ],
        device=bottom_centers.device,
        dtype=bottom_centers.dtype,
    )
    centers = bottom_centers.clone()
    centers[..., 2] += sizes[..., 2] * 0.5
    local = sizes[..., None, :] * signs * 0.5
    return torch.einsum("...ij,...kj->...ki", rotations, local) + centers[..., None, :]


def _yaw_rotations_from_logits(logits: torch.Tensor) -> torch.Tensor:
    """Use hard bin rotations in forward and soft bin weights in backward."""
    probabilities = torch.softmax(logits, dim=-1)
    indices = torch.argmax(probabilities, dim=-1)
    hard = F.one_hot(indices, NUM_YAW_BINS).to(probabilities.dtype)
    weights = hard + probabilities - probabilities.detach()
    angles = yaw_bin_angles(logits.device, logits.dtype)
    cosine, sine = torch.cos(angles), torch.sin(angles)
    zeros, ones = torch.zeros_like(angles), torch.ones_like(angles)
    rotations = torch.stack(
        [cosine, -sine, zeros, sine, cosine, zeros, zeros, zeros, ones], dim=-1
    ).reshape(NUM_YAW_BINS, 3, 3)
    return torch.einsum("...k,kij->...ij", weights, rotations)


def _minimum_valid_corner_loss(
    pred_centers: torch.Tensor,
    pred_sizes: torch.Tensor,
    pred_yaw_logits: torch.Tensor,
    target_centers: torch.Tensor,
    target_sizes: torch.Tensor,
    target_yaw_masks: torch.Tensor,
) -> torch.Tensor:
    """Return the mean minimum symmetric Chamfer-L1 over valid target yaw bins."""
    if len(pred_centers) == 0:
        return pred_centers.sum() * 0.0
    pred_corners = _box_corners_from_rotation(
        pred_centers,
        pred_sizes,
        _yaw_rotations_from_logits(pred_yaw_logits),
    )
    angles = yaw_bin_angles(pred_centers.device, pred_centers.dtype)
    cosine, sine = torch.cos(angles), torch.sin(angles)
    zeros, ones = torch.zeros_like(angles), torch.ones_like(angles)
    rotations = torch.stack(
        [cosine, -sine, zeros, sine, cosine, zeros, zeros, zeros, ones], dim=-1
    ).reshape(NUM_YAW_BINS, 3, 3)
    count = len(target_centers)
    target_corners = _box_corners_from_rotation(
        target_centers[:, None, :].expand(-1, NUM_YAW_BINS, -1),
        target_sizes[:, None, :].expand(-1, NUM_YAW_BINS, -1),
        rotations[None].expand(count, -1, -1, -1),
    )
    distance = torch.abs(
        pred_corners[:, None, :, None, :] - target_corners[:, :, None, :, :]
    ).sum(dim=-1)
    chamfer = distance.amin(dim=-1).mean(dim=-1) + distance.amin(dim=-2).mean(dim=-1)
    chamfer = chamfer.masked_fill(~target_yaw_masks, float("inf"))
    diagonal = torch.linalg.vector_norm(target_sizes, dim=-1).clamp_min(1e-4)
    return (chamfer.amin(dim=-1) / diagonal).mean()


def _hungarian_matches(
    outputs: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
    background_match_cost: float,
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """Match queries one-to-one with all GT centers or fixed-cost background targets."""
    if background_match_cost <= 0.0:
        raise ValueError("background_match_cost must be positive")
    batch_size = int(batch["batch_size"])
    matches: list[tuple[torch.Tensor, torch.Tensor] | None] = [None] * batch_size
    pending = []
    flat_costs = []
    for batch_index in range(batch_size):
        query_indices = torch.nonzero(outputs["query_valid_mask"][batch_index], as_tuple=False).flatten()
        target_indices = torch.nonzero(batch["gt_valid_mask"][batch_index], as_tuple=False).flatten()
        if len(query_indices) == 0 or len(target_indices) == 0:
            matches[batch_index] = (query_indices[:0], target_indices[:0])
            continue
        # 匹配只建立空间对应关系，分类、Yaw 和角点属性在匹配后单独监督。
        target_size = batch["source_box_gt"][batch_index, 3:6].clamp_min(1e-4)
        cost = torch.abs(
            outputs["pred_bottom_centers"][batch_index, query_indices, None, :]
            - batch["gt_bottom_centers"][batch_index, None, target_indices, :]
        ).div(target_size).mean(dim=-1)
        # 每个有效 Query 配置一个独立背景列，使远离合法区域的预测可以不匹配真实正点。
        background_cost = cost.new_full(
            (len(query_indices), len(query_indices)), float(background_match_cost)
        )
        cost = torch.cat([cost, background_cost], dim=1)
        pending.append(
            (
                batch_index,
                query_indices,
                target_indices,
                len(target_indices),
                tuple(cost.shape),
                cost.numel(),
            )
        )
        flat_costs.append(cost.reshape(-1))

    if flat_costs:
        costs_cpu = torch.cat(flat_costs).detach().float().cpu().numpy()
        offset = 0
        for batch_index, query_indices, target_indices, target_count, shape, count in pending:
            cost_matrix = costs_cpu[offset : offset + count].reshape(shape)
            offset += count
            pred_rows, target_rows = linear_sum_assignment(cost_matrix)
            real_match = target_rows < target_count
            pred_rows = pred_rows[real_match]
            target_rows = target_rows[real_match]
            matches[batch_index] = (
                query_indices[torch.as_tensor(pred_rows, device=query_indices.device)],
                target_indices[torch.as_tensor(target_rows, device=target_indices.device)],
            )
    if any(match is None for match in matches):
        raise RuntimeError("Hungarian matching did not produce one result per batch item")
    return [match for match in matches if match is not None]


def _set_losses_for_predictions(
    prediction: dict[str, torch.Tensor],
    matches: list[tuple[torch.Tensor, torch.Tensor]],
    outputs: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute classification, center, yaw and corner terms for one decoder layer."""
    score_target = prediction["pred_logits"].new_zeros(prediction["pred_logits"].shape)
    pred_centers, target_centers = [], []
    pred_yaw, target_yaw = [], []
    pred_sizes, target_sizes = [], []
    for batch_index, (query_indices, target_indices) in enumerate(matches):
        if len(query_indices) == 0:
            continue
        score_target[batch_index, query_indices] = batch["gt_affordance_quality"][batch_index, target_indices]
        pred_centers.append(prediction["pred_bottom_centers"][batch_index, query_indices])
        target_centers.append(batch["gt_bottom_centers"][batch_index, target_indices])
        pred_yaw.append(prediction["pred_yaw_logits"][batch_index, query_indices])
        target_yaw.append(batch["gt_yaw_masks"][batch_index, target_indices])
        pred_sizes.append(outputs["source_box"][batch_index, 3:6][None].expand(len(query_indices), -1))
        target_sizes.append(batch["source_box_gt"][batch_index, 3:6][None].expand(len(query_indices), -1))
    valid_queries = outputs["query_valid_mask"]
    cls_loss = F.binary_cross_entropy_with_logits(
        prediction["pred_logits"][valid_queries], score_target[valid_queries]
    )
    if not pred_centers:
        # 全背景 batch 没有回归目标，但仍需让回归头进入计算图以完成 DDP 梯度归约。
        zero = (
            prediction["pred_logits"].sum()
            + prediction["pred_bottom_centers"].sum()
            + prediction["pred_yaw_logits"].sum()
        ) * 0.0
        return cls_loss, zero, zero, zero
    pred_center = torch.cat(pred_centers)
    target_center = torch.cat(target_centers)
    pred_size = torch.cat(pred_sizes)
    target_size = torch.cat(target_sizes)
    pred_yaw_logits = torch.cat(pred_yaw)
    target_yaw_mask = torch.cat(target_yaw)
    center_loss = F.smooth_l1_loss(
        (pred_center - target_center) / target_size.clamp_min(1e-4),
        torch.zeros_like(pred_center),
    )
    yaw_loss = F.binary_cross_entropy_with_logits(pred_yaw_logits, target_yaw_mask.to(pred_yaw_logits.dtype))
    corner_loss = _minimum_valid_corner_loss(
        pred_center, pred_size, pred_yaw_logits, target_center, target_size, target_yaw_mask
    )
    return cls_loss, center_loss, yaw_loss, corner_loss


def select_nearest_direct_box_targets(
    pred_bottom_centers: torch.Tensor,
    batch: dict[str, torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Match each single-query prediction to its nearest valid placement target."""
    valid_mask = batch["gt_valid_mask"]
    if torch.any(~valid_mask.any(dim=1)):
        raise ValueError("Direct-Box 1Q requires at least one valid target per sample")
    target_size = batch["source_box_gt"][:, None, 3:6].clamp_min(1e-4)
    normalized_distance = torch.abs(
        pred_bottom_centers[:, None, :] - batch["gt_bottom_centers"]
    ).div(target_size).mean(dim=-1)
    nearest_indices = normalized_distance.masked_fill(~valid_mask, float("inf")).argmin(dim=1)
    batch_indices = torch.arange(len(pred_bottom_centers), device=pred_bottom_centers.device)
    return (
        nearest_indices,
        batch["gt_bottom_centers"][batch_indices, nearest_indices],
        batch["gt_yaw_masks"][batch_indices, nearest_indices],
    )


def compute_direct_box_1q_loss(
    outputs: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
    cfg: dict[str, Any],
) -> dict[str, torch.Tensor]:
    """Compute nearest-valid-center, multi-yaw, corner and source losses."""
    loss_cfg = cfg["loss"]
    pred_center = outputs["pred_bottom_centers"][:, 0]
    pred_yaw_logits = outputs["raw_yaw_logits"][:, 0]
    nearest_indices, target_center, target_yaw_mask = select_nearest_direct_box_targets(
        pred_center, batch
    )
    target_size = batch["source_box_gt"][:, 3:6].clamp_min(1e-4)
    pred_size = outputs["source_box"][:, 3:6].clamp_min(1e-4)
    center_loss = F.smooth_l1_loss(
        (pred_center - target_center) / target_size,
        torch.zeros_like(pred_center),
    )
    yaw_loss = F.binary_cross_entropy_with_logits(
        pred_yaw_logits,
        target_yaw_mask.to(pred_yaw_logits.dtype),
    )
    corner_loss = _minimum_valid_corner_loss(
        pred_center,
        pred_size,
        pred_yaw_logits,
        target_center,
        target_size,
        target_yaw_mask,
    )
    source_cfg = loss_cfg["source"]
    source_loss, source_terms = source_box_loss(
        outputs["source_box"],
        batch["source_box_gt"],
        lambda_center=float(source_cfg.get("lambda_center", 4.0)),
        lambda_size=float(source_cfg.get("lambda_size", 2.0)),
        lambda_iou=float(source_cfg.get("lambda_iou", 0.2)),
    )
    total = (
        float(loss_cfg.get("lambda_center", 5.0)) * center_loss
        + float(loss_cfg.get("lambda_yaw", 0.5)) * yaw_loss
        + float(loss_cfg.get("lambda_corner", 0.5)) * corner_loss
        + float(loss_cfg.get("lambda_src", 0.5)) * source_loss
    )
    sample_count = pred_center.new_tensor(float(len(pred_center)))
    terms = {
        "loss": total,
        "loss_center": center_loss.detach(),
        "loss_yaw": yaw_loss.detach(),
        "loss_corner": corner_loss.detach(),
        "loss_src": source_loss.detach(),
        "valid_query_count": sample_count,
        "matched_query_count": sample_count,
        "background_query_count": sample_count.new_zeros(()),
        "padded_query_count": sample_count.new_zeros(()),
        "nearest_target_index_mean": nearest_indices.float().mean().detach(),
    }
    terms.update(source_terms)
    return terms


def compute_stage2_loss(
    outputs: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
    cfg: dict[str, Any],
    compute_diagnostics: bool = True,
) -> dict[str, torch.Tensor]:
    """Compute the configured Stage 2 architecture loss."""
    if str(cfg.get("model", {}).get("type", "space_former")).lower() == DIRECT_BOX_MODEL_TYPE:
        return compute_direct_box_1q_loss(outputs, batch, cfg)
    loss_cfg = cfg["loss"]
    focal_cfg = loss_cfg.get("focal", {})
    sigma = float(cfg["data"].get("heatmap_sigma_voxels", 2.0)) * float(
        cfg["data"]["voxel_size_cm"]
    )
    region_support_mask = aggregate_sparse_mask(
        batch["sparse_coords"],
        batch["support_masks"],
        outputs["region_sparse_coords"],
        outputs["region_spatial_shape"],
        stride=4,
    )
    region_target = build_dense_heatmap_targets(
        world_coords=outputs["region_world_coords"],
        batch_indices=outputs["region_batch_indices"],
        positive_points=batch["heatmap_positive_points"],
        positive_batch_indices=batch["heatmap_positive_batch_indices"],
        support_masks=region_support_mask,
        batch_size=int(batch["batch_size"]),
        sigma=sigma,
    )
    region_loss = focal_loss_with_logits(
        outputs["region_logits"],
        region_target,
        alpha=float(focal_cfg.get("alpha", 0.25)),
        gamma=float(focal_cfg.get("gamma", 2.0)),
    )
    p3_gt_point_coverage = None
    if compute_diagnostics:
        model_cfg = cfg.get("model", {})
        space_cfg = model_cfg.get("space_former", model_cfg.get("placement", {}))
        p3_gt_point_coverage = compute_p3_gt_point_coverage(
            outputs["region_sparse_coords"],
            outputs["region_logits"],
            outputs["voxel_origins"],
            batch["heatmap_positive_points"],
            batch["heatmap_positive_batch_indices"],
            float(cfg["data"]["voxel_size_cm"]),
            int(batch["batch_size"]),
            int(space_cfg.get("num_region_cells", 8)),
        )
    matches = _hungarian_matches(
        outputs,
        batch,
        background_match_cost=float(loss_cfg["background_match_cost"]),
    )
    matched_query_count = outputs["raw_place_logits"].new_tensor(
        sum(len(query_indices) for query_indices, _ in matches)
    )
    valid_query_count = outputs["query_valid_mask"].sum()
    final_prediction = {
        "pred_bottom_centers": outputs["pred_bottom_centers"],
        "pred_yaw_logits": outputs["raw_yaw_logits"],
        "pred_logits": outputs["raw_place_logits"],
    }
    cls_loss, center_loss, yaw_loss, corner_loss = _set_losses_for_predictions(
        final_prediction, matches, outputs, batch
    )
    auxiliary = outputs["raw_place_logits"].sum() * 0.0
    for prediction in outputs["decoder_aux_outputs"]:
        aux_terms = _set_losses_for_predictions(prediction, matches, outputs, batch)
        auxiliary = auxiliary + 2.0 * aux_terms[0] + 8.0 * aux_terms[1] + aux_terms[2] + aux_terms[3]

    source_cfg = loss_cfg["source"]
    source_loss, source_terms = source_box_loss(
        outputs["source_box"],
        batch["source_box_gt"],
        lambda_center=float(source_cfg.get("lambda_center", 4.0)),
        lambda_size=float(source_cfg.get("lambda_size", 2.0)),
        lambda_iou=float(source_cfg.get("lambda_iou", 0.2)),
    )
    total = (
        float(loss_cfg.get("lambda_region", 1.0)) * region_loss
        + float(loss_cfg.get("lambda_cls", 2.0)) * cls_loss
        + float(loss_cfg.get("lambda_center", 5.0)) * center_loss
        + float(loss_cfg.get("lambda_yaw", 0.5)) * yaw_loss
        + float(loss_cfg.get("lambda_corner", 1.0)) * corner_loss
        + float(loss_cfg.get("lambda_src", 1.0)) * source_loss
        + float(loss_cfg.get("lambda_aux", 0.5)) * auxiliary
    )
    terms = {
        "loss": total,
        "loss_region": region_loss.detach(),
        "loss_cls": cls_loss.detach(),
        "loss_center": center_loss.detach(),
        "loss_yaw": yaw_loss.detach(),
        "loss_corner": corner_loss.detach(),
        "loss_aux": auxiliary.detach(),
        "loss_src": source_loss.detach(),
        "region_target_mass": region_target.sum().detach(),
        "region_selected_cell_count": outputs["region_selected_cell_count"].sum().detach(),
        "expanded_p1_candidate_count": outputs["expanded_p1_candidate_count"].sum().detach(),
        "valid_query_count": valid_query_count.detach(),
        "matched_query_count": matched_query_count.detach(),
        "background_query_count": (valid_query_count - matched_query_count).detach(),
        "padded_query_count": (~outputs["query_valid_mask"]).sum().detach(),
    }
    if p3_gt_point_coverage is not None:
        terms["p3_gt_point_coverage"] = p3_gt_point_coverage.detach()
    for key, value in outputs["sampling_logs"].items():
        terms[key] = value.detach()
        if key.endswith("_sample_active_count"):
            total_key = key.replace("_sample_active_count", "_sample_total_count")
            denominator = outputs["sampling_logs"][total_key].clamp_min(1)
            terms[key.replace("_count", "_ratio")] = (value / denominator).detach()
        if key.endswith("_sample_feature_nonzero_count"):
            total_key = key.replace("_sample_feature_nonzero_count", "_sample_total_count")
            denominator = outputs["sampling_logs"][total_key].clamp_min(1)
            terms[key.replace("_count", "_ratio")] = (value / denominator).detach()
        if key.endswith("_bottom_active_count"):
            denominator = outputs["sampling_logs"][key.replace("_active_count", "_sample_count")].clamp_min(1)
            terms[key.replace("_count", "_ratio")] = (value / denominator).detach()
        if key.endswith("_nonbottom_active_count"):
            denominator = outputs["sampling_logs"][key.replace("_active_count", "_sample_count")].clamp_min(1)
            terms[key.replace("_count", "_ratio")] = (value / denominator).detach()
        if key.endswith("_all_zero_sample_query_count"):
            terms[key.replace("_count", "_ratio")] = (
                value / outputs["query_valid_mask"].sum().clamp_min(1)
            ).detach()
    terms.update(source_terms)
    return terms


def compute_stage2_task_metric_sums(
    outputs: dict[str, torch.Tensor],
    batch: dict[str, Any],
    cfg: dict[str, Any],
) -> dict[str, float]:
    """Compute three-condition Placement Success and separate source/size/yaw metrics."""
    place_gt = batch["place_box_gt"].detach().cpu().numpy()
    place_sets = outputs["place_boxes"].detach().cpu().numpy()
    place_set_masks = outputs["place_valid_mask"].detach().cpu().numpy()
    place_yaw_bins = outputs["place_yaw_bins"].detach().cpu().numpy()
    gt_centers = batch["gt_bottom_centers"].detach().cpu().numpy()
    gt_yaw_masks = batch["gt_yaw_masks"].detach().cpu().numpy()
    gt_valid_masks = batch["gt_valid_mask"].detach().cpu().numpy()
    world_coords = batch["world_coords"].detach().cpu().numpy()
    point_batch_indices = batch["batch_indices"].detach().cpu().numpy()
    contexts = batch["validation_contexts"]
    validation_cfg = cfg.get("validation", {})
    source_iou_threshold = float(validation_cfg.get("source_iou_threshold", 0.5))
    size_iou_threshold = float(validation_cfg.get("size_iou_threshold", 0.8))
    center_match_threshold_cm = float(validation_cfg.get("center_match_threshold_cm", 2.0))
    support_downward_cm = float(validation_cfg.get("support_downward_cm", 3.0))
    support_upper_cm = float(validation_cfg.get("support_upper_cm", 1.0))
    voxel_size_cm = float(cfg["data"]["voxel_size_cm"])

    source_pred = outputs["source_box"].detach().cpu().numpy()
    source_gt = batch["source_box_gt"].detach().cpu().numpy()
    source_ious = [compute_aabb_iou_3d(pred, gt) for pred, gt in zip(source_pred, source_gt)]
    source_correct = [value >= source_iou_threshold for value in source_ious]
    source_center_mae = np.abs(source_pred[:, :3] - source_gt[:, :3]).mean(axis=1)
    source_size_iou = [compute_size_iou(pred[3:6], gt[3:6]) for pred, gt in zip(source_pred, source_gt)]

    top1_size_ious = []
    top1_size_correct = []
    top1_direction = []
    top1_supported = []
    top1_collision_free = []
    top1_center_matched = []
    top1_yaw_valid = []
    placement_at_1 = []
    placement_at_5 = []
    successful_pose_count = 0
    emitted_pose_count = 0
    duplicate_pairs = 0
    possible_pairs = 0
    for sample_index, (gt, context) in enumerate(zip(place_gt, contexts)):
        occupied_keys = quantize_occupied_points(
            world_coords[point_batch_indices == sample_index], voxel_size_cm
        )
        support_cache = {}
        candidate_results = []
        candidate_components = []
        emitted = np.flatnonzero(place_set_masks[sample_index])
        emitted_pose_count += len(emitted)
        valid_gt = gt_valid_masks[sample_index]
        target_centers = gt_centers[sample_index, valid_gt]
        target_yaw = gt_yaw_masks[sample_index, valid_gt]
        for output_index in emitted:
            box = place_sets[sample_index, output_index]
            evaluation_box = build_placement_evaluation_box(box, gt)
            size_metrics = compute_size_metrics(box, gt, size_iou_threshold)
            direction_correct = compute_direction_hit(box, context)
            supported, _ = compute_supported_and_stable(
                evaluation_box,
                occupied_keys,
                voxel_size_cm,
                downward_cm=support_downward_cm,
                upper_cm=support_upper_cm,
                cache=support_cache,
            )
            collision_free = not compute_collision_metrics(
                evaluation_box, context.collision_context
            )["collision"]
            yaw_valid, center_distance, _ = compute_yaw_valid_at_matched_center(
                box,
                int(place_yaw_bins[sample_index, output_index]),
                target_centers,
                target_yaw,
                center_match_threshold_cm,
            )
            center_matched = bool(
                center_distance is not None and center_distance <= center_match_threshold_cm
            )
            components = (
                size_metrics["size_iou"],
                size_metrics["size_correct"],
                direction_correct,
                supported,
                collision_free,
                center_matched,
                yaw_valid,
            )
            candidate_components.append(components)
            candidate_success = placement_success(
                direction_correct,
                supported,
                collision_free,
            )
            candidate_results.append(candidate_success)
            successful_pose_count += int(candidate_success)

        if candidate_components:
            first = candidate_components[0]
            top1_size_ious.append(first[0])
            top1_size_correct.append(first[1])
            top1_direction.append(first[2])
            top1_supported.append(first[3])
            top1_collision_free.append(first[4])
            top1_center_matched.append(first[5])
            top1_yaw_valid.append(first[6])
        else:
            top1_size_ious.append(0.0)
            top1_size_correct.append(False)
            top1_direction.append(False)
            top1_supported.append(False)
            top1_collision_free.append(False)
            top1_center_matched.append(False)
            top1_yaw_valid.append(False)
        placement_at_1.append(any(candidate_results[:1]))
        placement_at_5.append(any(candidate_results[:5]))

        for left in range(len(emitted)):
            for right in range(left + 1, len(emitted)):
                possible_pairs += 1
                box_a = place_sets[sample_index, emitted[left]]
                box_b = place_sets[sample_index, emitted[right]]
                distance = np.linalg.norm((box_a[:3] - box_b[:3]) / np.maximum(box_a[3:6], 1e-4))
                yaw_delta = abs(int(place_yaw_bins[sample_index, emitted[left]]) - int(place_yaw_bins[sample_index, emitted[right]]))
                yaw_delta = min(yaw_delta, NUM_YAW_BINS - yaw_delta)
                duplicate_pairs += int(distance < 0.25 and yaw_delta <= 1)

    return {
        "sample_count": float(len(place_gt)),
        "source_iou_sum": float(sum(source_ious)),
        "source_iou_correct_sum": float(sum(source_correct)),
        "placement_size_iou_sum": float(sum(top1_size_ious)),
        "placement_size_correct_sum": float(sum(top1_size_correct)),
        "language_relation_correct_sum": float(sum(top1_direction)),
        "supported_and_stable_sum": float(sum(top1_supported)),
        "collision_free_sum": float(sum(top1_collision_free)),
        "center_match_sum": float(sum(top1_center_matched)),
        "yaw_valid_on_matched_center_sum": float(sum(top1_yaw_valid)),
        "placement_success_at_1_sum": float(sum(placement_at_1)),
        "placement_success_at_5_sum": float(sum(placement_at_5)),
        "successful_pose_count": float(successful_pose_count),
        "emitted_pose_count": float(emitted_pose_count),
        "duplicate_pair_count": float(duplicate_pairs),
        "possible_pair_count": float(possible_pairs),
        "source_center_mae_sum": float(np.sum(source_center_mae)),
        "source_size_iou_sum": float(np.sum(source_size_iou)),
    }


def _make_loader(
    dataset: Dataset,
    cfg: dict[str, Any],
    shuffle: bool,
    sampler: DistributedSampler | None = None,
) -> DataLoader:
    data_cfg = cfg["data"]
    train_cfg = cfg["training"]
    num_workers = int(train_cfg.get("num_workers", 0))
    return DataLoader(
        dataset,
        batch_size=int(train_cfg["batch_size"]),
        shuffle=shuffle if sampler is None else False,
        sampler=sampler,
        num_workers=num_workers,
        collate_fn=lambda batch: stage2_collate(batch, voxel_size_cm=float(data_cfg["voxel_size_cm"])),
        pin_memory=bool(train_cfg.get("pin_memory", True)),
        persistent_workers=num_workers > 0,
    )


def _optimizer_learning_rates(optimizer: torch.optim.Optimizer) -> dict[str, float]:
    """Return the Stage 1 and Stage 2 optimizer-group learning rates."""
    if len(optimizer.param_groups) != 2:
        raise ValueError(f"Expected two optimizer parameter groups, got {len(optimizer.param_groups)}")
    return {
        "lr_stage1": float(optimizer.param_groups[0]["lr"]),
        "lr_stage2": float(optimizer.param_groups[1]["lr"]),
    }


def _override_optimizer_learning_rates(optimizer: torch.optim.Optimizer, base_lr: float) -> None:
    """Apply config learning rates after restoring optimizer state."""
    rates = (float(base_lr) * 0.1, float(base_lr))
    if len(optimizer.param_groups) != len(rates):
        raise ValueError(f"Expected two optimizer parameter groups, got {len(optimizer.param_groups)}")
    for group, learning_rate in zip(optimizer.param_groups, rates):
        group["lr"] = learning_rate
        group["initial_lr"] = learning_rate


def _build_lr_scheduler(
    optimizer: torch.optim.Optimizer,
    cfg: dict[str, Any],
) -> torch.optim.lr_scheduler.ReduceLROnPlateau:
    """Build the placement-success plateau scheduler from training config."""
    scheduler_cfg = cfg["training"].get("lr_scheduler", {})
    scheduler_type = str(scheduler_cfg.get("type", "reduce_on_plateau")).lower()
    if scheduler_type != "reduce_on_plateau":
        raise ValueError(f"Unsupported Stage 2 lr_scheduler type: {scheduler_type}")
    return torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode=str(scheduler_cfg.get("mode", "max")),
        factor=float(scheduler_cfg.get("factor", 0.5)),
        patience=int(scheduler_cfg.get("patience", 3)),
        threshold=float(scheduler_cfg.get("threshold", 0.001)),
        threshold_mode=str(scheduler_cfg.get("threshold_mode", "abs")),
    )


def load_stage1_weights(model: torch.nn.Module, checkpoint_path: str | Path, device: torch.device) -> list[str]:
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


def _format_stage2_log(payload: dict[str, Any]) -> str:
    """Format one concise terminal summary while metrics.jsonl keeps all fields."""
    split = str(payload["split"])
    fields = {
        "train": (
            ("loss", "loss"),
            ("loss_region", "region"),
            ("loss_cls", "cls"),
            ("loss_center", "center"),
            ("loss_yaw", "yaw"),
            ("loss_corner", "corner"),
            ("source_iou", "src_iou"),
            ("p3_gt_point_coverage", "p3_gt_cov"),
            ("lr_stage1", "lr_s1"),
            ("lr_stage2", "lr_s2"),
        ),
        "valid": (
            ("loss", "loss"),
            ("placement_success_at_1", "placement@1"),
            ("placement_success_at_5", "placement@5"),
            ("language_relation_correct_rate", "relation"),
            ("supported_and_stable_rate", "support"),
            ("collision_free_rate", "collision_free"),
            ("placement_size_iou", "size_iou"),
            ("center_match_rate", "center_match"),
            ("yaw_valid_given_center_match", "yaw|match"),
            ("p3_gt_point_coverage", "p3_gt_cov"),
            ("lr_stage1", "lr_s1"),
            ("lr_stage2", "lr_s2"),
        ),
    }[split]
    parts = [f"[{split}]", f"epoch={int(payload['epoch'])}", f"step={int(payload['step'])}"]
    for key, label in fields:
        if key not in payload:
            continue
        value = float(payload[key])
        parts.append(f"{label}={value:.2e}" if key.startswith("lr_") else f"{label}={value:.4f}")
    return " ".join(parts)


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
            voxel_size_cm=float(data_cfg["voxel_size_cm"]),
        )
        val_set = LCBGPlaceNetStage2Dataset(
            sources=None,
            split="valid",
            val_fraction=valid_fraction,
            seed=int(data_cfg.get("split_seed", 0)),
            max_samples=max_val_samples or data_cfg.get("max_valid_samples", data_cfg.get("max_val_samples")),
            items=all_items,
            split_dir=split_dir,
            voxel_size_cm=float(data_cfg["voxel_size_cm"]),
        )
        validation_contexts = build_stage2_validation_contexts(val_set.items)

        model = build_stage2_model(cfg["model"]).to(device)
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
        base_lr = float(cfg["training"]["lr"])
        stage1_prefixes = (
            "voxel_image_encoder",
            "backbone",
            "text_encoder",
            "fusion",
            "pos_mlp",
            "source_grounding",
        )
        stage1_parameters = []
        stage2_parameters = []
        for name, parameter in trainable_model.named_parameters():
            if not parameter.requires_grad:
                continue
            target = stage1_parameters if name.startswith(stage1_prefixes) else stage2_parameters
            target.append(parameter)
        optimizer = torch.optim.AdamW(
            [
                {"params": stage1_parameters, "lr": base_lr * 0.1},
                {"params": stage2_parameters, "lr": base_lr},
            ],
            weight_decay=float(cfg["training"].get("weight_decay", 0.0)),
        )
        scheduler = _build_lr_scheduler(optimizer, cfg)

        train_sampler = DistributedSampler(train_set, num_replicas=world_size, rank=rank, shuffle=True) if distributed else None
        val_sampler = DistributedSampler(val_set, num_replicas=world_size, rank=rank, shuffle=False) if distributed else None
        train_loader = _make_loader(train_set, cfg, shuffle=True, sampler=train_sampler)
        val_loader = _make_loader(val_set, cfg, shuffle=False, sampler=val_sampler)
        metrics_path = output_dir / "metrics.jsonl"
        best_placement_success_at_1 = -math.inf
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
            has_placement_metric = "best_placement_success_at_1" in checkpoint
            best_placement_success_at_1 = float(
                checkpoint.get("best_placement_success_at_1", -math.inf)
            )
            if has_placement_metric and "scheduler" in checkpoint:
                scheduler.load_state_dict(checkpoint["scheduler"])
            elif math.isfinite(best_placement_success_at_1):
                # Placement-metric checkpoints predating scheduler state still seed its baseline.
                scheduler.step(best_placement_success_at_1)
            _override_optimizer_learning_rates(optimizer, base_lr)
            scheduler._last_lr = [group["lr"] for group in optimizer.param_groups]
            if _is_main_process(rank):
                resumed_lrs = _optimizer_learning_rates(optimizer)
                print(
                    f"Resumed Stage 2 from {resume_path} at epoch={start_epoch}, "
                    f"step={global_step}, best_placement_success_at_1={best_placement_success_at_1:.6f}, "
                    f"lr_stage1={resumed_lrs['lr_stage1']:.2e}, lr_stage2={resumed_lrs['lr_stage2']:.2e}"
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
                log_every = int(cfg["training"].get("log_every", 20))
                should_log = global_step % log_every == 0
                batch = move_batch_to_device(batch, device)
                optimizer.zero_grad(set_to_none=True)
                outputs = model(batch, collect_diagnostics=should_log)
                losses = compute_stage2_loss(
                    outputs,
                    batch,
                    cfg,
                    compute_diagnostics=should_log,
                )
                losses["loss"].backward()
                torch.nn.utils.clip_grad_norm_(trainable_model.parameters(), float(cfg["training"].get("grad_clip_norm", 1.0)))
                optimizer.step()

                if show_progress and global_step % 20 == 0:
                    train_iter.set_postfix(loss=f"{float(losses['loss'].detach().cpu()):.4f}", step=global_step)

                if should_log:
                    log_values = {key: value.detach().float().clone() for key, value in losses.items()}
                    if distributed:
                        for key, value in log_values.items():
                            dist.all_reduce(value, op=dist.ReduceOp.SUM)
                            if not key.endswith("_count"):
                                value.div_(world_size)
                    if _is_main_process(rank):
                        log_payload = {
                            "split": "train",
                            "epoch": epoch,
                            "step": global_step,
                            **{key: float(value.cpu()) for key, value in log_values.items()},
                            **_optimizer_learning_rates(optimizer),
                        }
                        _append_jsonl(metrics_path, log_payload)
                        print(_format_stage2_log(log_payload))

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
            epoch_lrs = _optimizer_learning_rates(optimizer)
            current_success_rate = val_metrics.get("placement_success_at_1", -math.inf)
            is_best = current_success_rate > best_placement_success_at_1
            if is_best:
                best_placement_success_at_1 = current_success_rate
            scheduler.step(current_success_rate)
            if _is_main_process(rank):
                val_payload = {
                    "split": "valid",
                    "epoch": epoch,
                    "step": global_step,
                    **val_metrics,
                    **epoch_lrs,
                }
                _append_jsonl(metrics_path, val_payload)
                print(_format_stage2_log(val_payload))

                checkpoint = {
                    "model": trainable_model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "epoch": epoch,
                    "step": global_step,
                    "best_placement_success_at_1": best_placement_success_at_1,
                    **_optimizer_learning_rates(optimizer),
                    "config": cfg,
                }
                if is_best:
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
    model: torch.nn.Module | DistributedDataParallel,
    loader: DataLoader,
    cfg: dict[str, Any],
    device: torch.device,
    validation_contexts: dict[str, Stage2ValidationContext],
    show_progress: bool = False,
    distributed: bool = False,
) -> dict[str, float]:
    """Evaluate Stage 2 on one dataloader."""
    model.eval()
    tracks_p3_coverage = str(cfg.get("model", {}).get("type", "space_former")).lower() == "space_former"
    totals = {
        "loss_sum": 0.0,
        "sample_count": 0.0,
        "source_iou_sum": 0.0,
        "source_iou_correct_sum": 0.0,
        "placement_size_iou_sum": 0.0,
        "placement_size_correct_sum": 0.0,
        "language_relation_correct_sum": 0.0,
        "supported_and_stable_sum": 0.0,
        "collision_free_sum": 0.0,
        "center_match_sum": 0.0,
        "yaw_valid_on_matched_center_sum": 0.0,
        "placement_success_at_1_sum": 0.0,
        "placement_success_at_5_sum": 0.0,
        "successful_pose_count": 0.0,
        "emitted_pose_count": 0.0,
        "duplicate_pair_count": 0.0,
        "possible_pair_count": 0.0,
        "source_center_mae_sum": 0.0,
        "source_size_iou_sum": 0.0,
        "p3_gt_point_coverage_sum": 0.0,
    }
    validation_log_sums: dict[str, float] = {}
    val_iter = _progress(loader, show_progress, desc="stage2 valid", total=len(loader))
    for batch in val_iter:
        batch["validation_contexts"] = [validation_contexts[item_id] for item_id in batch["item_ids"]]
        batch = move_batch_to_device(batch, device)
        outputs = model(batch, collect_diagnostics=True)
        losses = compute_stage2_loss(outputs, batch, cfg)
        metric_sums = compute_stage2_task_metric_sums(outputs, batch, cfg)
        totals["loss_sum"] += float(losses["loss"].detach().cpu()) * metric_sums["sample_count"]
        if tracks_p3_coverage:
            totals["p3_gt_point_coverage_sum"] += (
                float(losses["p3_gt_point_coverage"].detach().cpu()) * metric_sums["sample_count"]
            )
        for key, value in metric_sums.items():
            totals[key] += value
        for key, value in losses.items():
            if key in {
                "valid_query_count",
                "matched_query_count",
                "background_query_count",
                "padded_query_count",
                "region_selected_cell_count",
                "expanded_p1_candidate_count",
            } or (
                key.startswith("layer") and key.endswith("_count")
            ):
                validation_log_sums[key] = validation_log_sums.get(key, 0.0) + float(value.detach().cpu())

    if distributed:
        keys = list(totals)
        values = torch.tensor([totals[key] for key in keys], dtype=torch.float64, device=device)
        dist.all_reduce(values, op=dist.ReduceOp.SUM)
        totals = {key: float(value.cpu()) for key, value in zip(keys, values)}
        log_keys = sorted(validation_log_sums)
        if log_keys:
            log_values = torch.tensor(
                [validation_log_sums[key] for key in log_keys], dtype=torch.float64, device=device
            )
            dist.all_reduce(log_values, op=dist.ReduceOp.SUM)
            validation_log_sums = {
                key: float(value.cpu()) for key, value in zip(log_keys, log_values)
            }

    sample_count = max(totals["sample_count"], 1.0)
    emitted_pose_count = max(totals["emitted_pose_count"], 1.0)
    possible_pair_count = max(totals["possible_pair_count"], 1.0)
    matched_center_count = max(totals["center_match_sum"], 1.0)
    metrics = {
        "loss": totals["loss_sum"] / sample_count,
        "source_iou": totals["source_iou_sum"] / sample_count,
        "source_iou_accuracy": totals["source_iou_correct_sum"] / sample_count,
        "placement_size_iou": totals["placement_size_iou_sum"] / sample_count,
        "placement_size_accuracy": totals["placement_size_correct_sum"] / sample_count,
        "language_relation_correct_rate": totals["language_relation_correct_sum"] / sample_count,
        "supported_and_stable_rate": totals["supported_and_stable_sum"] / sample_count,
        "collision_free_rate": totals["collision_free_sum"] / sample_count,
        "center_match_rate": totals["center_match_sum"] / sample_count,
        "yaw_valid_given_center_match": totals["yaw_valid_on_matched_center_sum"] / matched_center_count,
        "placement_success_at_1": totals["placement_success_at_1_sum"] / sample_count,
        "placement_success_at_5": totals["placement_success_at_5_sum"] / sample_count,
        "successful_pose_rate": totals["successful_pose_count"] / emitted_pose_count,
        "duplicate_pair_rate": totals["duplicate_pair_count"] / possible_pair_count,
        "source_center_mae": totals["source_center_mae_sum"] / sample_count,
        "source_size_iou": totals["source_size_iou_sum"] / sample_count,
    }
    if tracks_p3_coverage:
        metrics["p3_gt_point_coverage"] = totals["p3_gt_point_coverage_sum"] / sample_count
    sampling_metrics = dict(validation_log_sums)
    for key, value in validation_log_sums.items():
        denominator_key = None
        if key.endswith("_sample_active_count") or key.endswith("_sample_feature_nonzero_count"):
            denominator_key = key.rsplit("_sample_", 1)[0] + "_sample_total_count"
        elif key.endswith("_bottom_active_count") or key.endswith("_nonbottom_active_count"):
            denominator_key = key.replace("_active_count", "_sample_count")
        elif key.endswith("_all_zero_sample_query_count"):
            denominator_key = "valid_query_count"
        elif key in {"matched_query_count", "background_query_count"}:
            denominator_key = "valid_query_count"
        if denominator_key is not None:
            denominator = max(validation_log_sums.get(denominator_key, 0.0), 1.0)
            sampling_metrics[key.replace("_count", "_ratio")] = value / denominator
    metrics.update({f"sampling/{key}": value for key, value in sampling_metrics.items()})
    return metrics

"""
LC-BGPlaceNet Stage 1 training utilities.

Stage 1 trains source grounding and support surface prediction from canonical
1cm voxel point clouds. Support labels are built by nearest-neighbor alignment
from free_bbox support mask PLY white points to active voxels.
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
from scipy.spatial import cKDTree
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel

try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover - tqdm is optional at runtime.
    tqdm = None

from src.annotation.free_bbox.io_utils import load_ply
from src.models.lc_bgplacenet.stage1 import LCBGPlaceNetStage1, aabb_iou_3d


STAGE1_SPLIT_SCHEMA_VERSION = "lc_bgplacenet_stage1_splits/v1"
STAGE1_SPLIT_NAMES = ("train", "valid", "test")
STAGE1_SPLIT_NAME_SET = set(STAGE1_SPLIT_NAMES)


@dataclass(frozen=True)
class Stage1DataSource:
    """One canonical dataset and its matching free_bbox / auto-label outputs."""

    name: str
    dataset_dir: Path
    free_bbox_dir: Path
    labels_path: Path


@dataclass(frozen=True)
class Stage1IndexItem:
    """Resolved metadata for one language-conditioned Stage 1 sample."""

    item_id: str
    label_index: int
    source_name: str
    sample_id: str
    object_id: str
    instruction: str
    dataset_dir: Path
    voxel_point_cloud_path: Path
    support_mask_path: Path
    source_box_gt: np.ndarray


def build_stage1_index(sources: list[Stage1DataSource]) -> list[Stage1IndexItem]:
    """Build resolved Stage 1 sample metadata from all data sources."""
    items: list[Stage1IndexItem] = []
    payload_cache: dict[tuple[str, str], dict[str, Any]] = {}

    for source in sources:
        with source.labels_path.open("r", encoding="utf-8") as f:
            label_records = json.load(f)

        for label_index, record in enumerate(label_records):
            if not _visualization_exists(source.free_bbox_dir, record):
                continue

            sample_id = str(record["sample_id"])
            object_id = str(record["object_id"])
            instruction = str(record["label"])
            cache_key = (source.name, sample_id)
            if cache_key not in payload_cache:
                placement_path = source.free_bbox_dir / "placements" / f"{sample_id}__placements.json"
                with placement_path.open("r", encoding="utf-8") as f:
                    payload_cache[cache_key] = json.load(f)
            payload = payload_cache[cache_key]

            support_mask_path = source.free_bbox_dir / payload["support_mask_ply"]

            obj_record = _find_object_record(payload, object_id)
            if obj_record is None:
                raise ValueError(f"Object {object_id} not found in {source.name}/{sample_id} placements")

            voxel_path = _resolve_voxel_path(source.dataset_dir, sample_id, payload)

            items.append(
                Stage1IndexItem(
                    item_id=make_stage1_item_id(source.name, label_index, sample_id, object_id, instruction),
                    label_index=label_index,
                    source_name=source.name,
                    sample_id=sample_id,
                    object_id=object_id,
                    instruction=instruction,
                    dataset_dir=source.dataset_dir,
                    voxel_point_cloud_path=voxel_path,
                    support_mask_path=support_mask_path,
                    source_box_gt=_source_box_from_obb(
                        obj_record["canonical_aabb_object"],
                        obj_record["original_pose_world"],
                    ),
                )
            )
    return items


def make_stage1_item_id(
    source_name: str,
    label_index: int,
    sample_id: str,
    object_id: str,
    instruction: str,
) -> str:
    """Build a stable id for one Stage 1 label record."""
    payload = json.dumps(
        [str(source_name), int(label_index), str(sample_id), str(object_id), str(instruction)],
        ensure_ascii=False,
        separators=(",", ":"),
    )
    digest = hashlib.blake2s(payload.encode("utf-8"), digest_size=8).hexdigest()
    return f"{source_name}__label_{int(label_index):06d}__{digest}"


def normalize_stage1_split(split: str) -> str:
    """Normalize split aliases used by old configs and CLIs."""
    split_name = "valid" if split == "val" else str(split)
    if split_name not in STAGE1_SPLIT_NAME_SET:
        raise ValueError(f"split must be one of train/valid/test, got {split}")
    return split_name


def stage1_item_to_split_record(item: Stage1IndexItem) -> dict[str, Any]:
    """Convert an index item to the compact metadata stored in split files."""
    return {
        "item_id": item.item_id,
        "label_index": int(item.label_index),
        "source_name": item.source_name,
        "sample_id": item.sample_id,
        "object_id": item.object_id,
        "instruction": item.instruction,
    }


def write_stage1_splits(
    items: list[Stage1IndexItem],
    output_dir: str | Path,
    valid_fraction: float = 0.1,
    test_fraction: float = 0.1,
    seed: int = 0,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Write fixed train/valid/test split files grouped by canonical sample."""
    if not items:
        raise ValueError("No Stage 1 items available for split generation")
    valid_fraction = float(valid_fraction)
    test_fraction = float(test_fraction)
    if valid_fraction < 0.0 or test_fraction < 0.0 or valid_fraction + test_fraction >= 1.0:
        raise ValueError("valid_fraction and test_fraction must be non-negative and sum to less than 1")

    output_dir = Path(output_dir)
    split_paths = [output_dir / f"{split}.json" for split in STAGE1_SPLIT_NAMES]
    manifest_path = output_dir / "manifest.json"
    existing = [path for path in [*split_paths, manifest_path] if path.exists()]
    if existing and not overwrite:
        joined = ", ".join(str(path) for path in existing[:3])
        raise FileExistsError(f"Split files already exist: {joined}; pass overwrite=True to replace them")

    rng = random.Random(int(seed))
    source_groups: dict[str, set[tuple[str, str]]] = {}
    for item in items:
        source_groups.setdefault(item.source_name, set()).add((item.source_name, item.sample_id))

    group_to_split: dict[tuple[str, str], str] = {}
    for source_name in sorted(source_groups):
        groups = sorted(source_groups[source_name])
        rng.shuffle(groups)
        valid_count = _holdout_count(len(groups), valid_fraction)
        test_count = _holdout_count(len(groups), test_fraction)
        while valid_count + test_count >= len(groups) and (valid_count > 0 or test_count > 0):
            if valid_count >= test_count and valid_count > 0:
                valid_count -= 1
            elif test_count > 0:
                test_count -= 1

        for group in groups[:valid_count]:
            group_to_split[group] = "valid"
        for group in groups[valid_count : valid_count + test_count]:
            group_to_split[group] = "test"
        for group in groups[valid_count + test_count :]:
            group_to_split[group] = "train"

    split_items: dict[str, list[Stage1IndexItem]] = {split: [] for split in STAGE1_SPLIT_NAMES}
    for item in items:
        split_name = group_to_split[(item.source_name, item.sample_id)]
        split_items[split_name].append(item)

    output_dir.mkdir(parents=True, exist_ok=True)
    for split_name, split_rows in split_items.items():
        payload = {
            "schema_version": STAGE1_SPLIT_SCHEMA_VERSION,
            "split": split_name,
            "group_by": ["source_name", "sample_id"],
            "item_count": len(split_rows),
            "items": [stage1_item_to_split_record(item) for item in split_rows],
        }
        with (output_dir / f"{split_name}.json").open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)

    manifest = _stage1_split_manifest(
        split_items=split_items,
        group_to_split=group_to_split,
        seed=int(seed),
        valid_fraction=valid_fraction,
        test_fraction=test_fraction,
    )
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    return manifest


def _holdout_count(total: int, fraction: float) -> int:
    """Convert a fraction to a holdout group count while keeping a train group."""
    count = int(round(total * float(fraction)))
    if fraction > 0.0 and total >= 3:
        count = max(count, 1)
    return count


def _stage1_split_manifest(
    split_items: dict[str, list[Stage1IndexItem]],
    group_to_split: dict[tuple[str, str], str],
    seed: int,
    valid_fraction: float,
    test_fraction: float,
) -> dict[str, Any]:
    """Build split metadata for quick inspection."""
    split_group_counts = {split: 0 for split in STAGE1_SPLIT_NAMES}
    source_counts: dict[str, dict[str, dict[str, int]]] = {}
    for (source_name, _sample_id), split_name in group_to_split.items():
        source_counts.setdefault(
            source_name,
            {
                "groups": {split: 0 for split in STAGE1_SPLIT_NAMES},
                "items": {split: 0 for split in STAGE1_SPLIT_NAMES},
            },
        )
        split_group_counts[split_name] += 1
        source_counts[source_name]["groups"][split_name] += 1

    for split_name, rows in split_items.items():
        for item in rows:
            source_counts[item.source_name]["items"][split_name] += 1

    return {
        "schema_version": STAGE1_SPLIT_SCHEMA_VERSION,
        "seed": int(seed),
        "valid_fraction": float(valid_fraction),
        "test_fraction": float(test_fraction),
        "group_by": ["source_name", "sample_id"],
        "item_count": int(sum(len(rows) for rows in split_items.values())),
        "group_count": int(len(group_to_split)),
        "split_item_counts": {split: len(split_items[split]) for split in STAGE1_SPLIT_NAMES},
        "split_group_counts": split_group_counts,
        "source_counts": source_counts,
    }


def _read_stage1_split_records(split_dir: str | Path, split: str) -> list[dict[str, Any]]:
    """Read one fixed Stage 1 split file."""
    split_name = normalize_stage1_split(split)
    path = Path(split_dir) / f"{split_name}.json"
    if not path.exists():
        raise FileNotFoundError(f"Stage 1 split file not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    if payload.get("schema_version") != STAGE1_SPLIT_SCHEMA_VERSION:
        raise ValueError(f"Unsupported split schema_version in {path}: {payload.get('schema_version')}")
    if payload.get("split") != split_name:
        raise ValueError(f"Split file {path} declares split={payload.get('split')}, expected {split_name}")
    return list(payload.get("items", []))


def select_stage1_split_items(
    items: list[Stage1IndexItem],
    split_dir: str | Path,
    split: str,
) -> list[Stage1IndexItem]:
    """Select items according to a fixed split file and preserve its order."""
    records = _read_stage1_split_records(split_dir, split)
    item_by_id = {item.item_id: item for item in items}
    if len(item_by_id) != len(items):
        raise ValueError("Stage 1 index contains duplicate item_id values")

    selected = []
    missing = []
    for record in records:
        item_id = str(record["item_id"])
        item = item_by_id.get(item_id)
        if item is None:
            missing.append(item_id)
            continue
        selected.append(item)
    if missing:
        preview = ", ".join(missing[:5])
        raise ValueError(f"{len(missing)} split items are missing from the current Stage 1 index: {preview}")
    return selected


def load_config(path: str | Path) -> dict[str, Any]:
    """Load a YAML config file."""
    with Path(path).open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def set_seed(seed: int) -> None:
    """Set Python, NumPy and PyTorch RNG seeds."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_sources_from_config(cfg: dict[str, Any]) -> list[Stage1DataSource]:
    """Build data source records from config paths."""
    sources = []
    for item in cfg["data"]["sources"]:
        sources.append(
            Stage1DataSource(
                name=str(item["name"]),
                dataset_dir=Path(item["dataset_dir"]),
                free_bbox_dir=Path(item["free_bbox_dir"]),
                labels_path=Path(item["labels_path"]),
            )
        )
    return sources


def _resolve_voxel_path(dataset_dir: Path, sample_id: str, placement_payload: dict[str, Any]) -> Path:
    """Resolve canonical point_clouds_voxel_1cm PLY for a sample."""
    sample_json = dataset_dir / "samples" / f"{sample_id}.json"
    if sample_json.exists():
        with sample_json.open("r", encoding="utf-8") as f:
            sample_record = json.load(f)
        rel_path = sample_record.get("voxel_point_cloud_path")
        if rel_path:
            return dataset_dir / rel_path

    raw = placement_payload.get("voxel_point_cloud_path")
    if raw:
        path = Path(raw)
        if path.exists():
            return path
        return dataset_dir / raw

    return dataset_dir / "point_clouds_voxel_1cm" / f"{sample_id}.ply"


def _visualization_exists(free_bbox_dir: Path, record: dict[str, Any]) -> bool:
    """Use generated visualization PNGs as the Stage 1 sample whitelist."""
    vis_rel = record.get("visualization_png")
    if not vis_rel:
        return False
    return (free_bbox_dir / str(vis_rel)).exists()


def _source_box_from_obb(canonical_aabb_object: list[float], pose_world: list[list[float]]) -> np.ndarray:
    """Convert source object OBB metadata to (cx, cy, cz, dx, dy, dz)."""
    bbox = np.asarray(canonical_aabb_object, dtype=np.float64)
    pose = np.asarray(pose_world, dtype=np.float64)
    obj_center = (bbox[:3] + bbox[3:]) * 0.5
    center = pose[:3, :3] @ obj_center + pose[:3, 3]

    dims = np.maximum(bbox[3:] - bbox[:3], 1e-4)
    axes = pose[:3, :3]
    axis_norms = np.linalg.norm(axes, axis=0)
    if np.any(axis_norms < 1e-12):
        raise ValueError("original_pose_world contains a degenerate rotation axis")
    axes = axes / axis_norms[None, :]

    world_up = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    up_axis = int(np.argmax(np.abs(axes.T @ world_up)))
    horizontal_axes = [axis for axis in range(3) if axis != up_axis]
    size = np.array(
        [dims[horizontal_axes[0]], dims[horizontal_axes[1]], dims[up_axis]],
        dtype=np.float64,
    )
    return np.concatenate([center, size]).astype(np.float32)


def _find_object_record(placement_payload: dict[str, Any], object_id: str) -> dict[str, Any] | None:
    """Find object metadata in a free_bbox placements payload."""
    for obj in placement_payload.get("objects", []):
        if obj.get("object_id") == object_id:
            return obj
    return None


class LCBGPlaceNetStage1Dataset(Dataset):
    """Dataset for source grounding and support surface Stage 1 training."""

    def __init__(
        self,
        sources: list[Stage1DataSource] | None,
        split: str,
        val_fraction: float = 0.1,
        seed: int = 0,
        support_align_threshold_cm: float = 1.5,
        max_samples: int | None = None,
        items: list[Stage1IndexItem] | None = None,
        split_dir: str | Path | None = None,
    ) -> None:
        split_name = normalize_stage1_split(split)
        self.split = split_name
        self.support_align_threshold_cm = float(support_align_threshold_cm)
        if items is None:
            if sources is None:
                raise ValueError("sources must be provided when items is None")
            items = build_stage1_index(sources)
        items = list(items)
        if split_dir is not None:
            self.items = select_stage1_split_items(items, split_dir, split_name)
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
            raise ValueError(f"No Stage 1 samples found for split={split_name}")

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        item = self.items[idx]
        points, colors = load_ply(item.voxel_point_cloud_path)
        mask_points, mask_colors = load_ply(item.support_mask_path)
        white = np.all(mask_colors.astype(np.int16) >= 240, axis=1)
        support_points = mask_points[white]
        if len(points) == 0 or len(support_points) == 0:
            raise ValueError(f"Invalid empty point/support data for {item.sample_id}")

        support_tree = cKDTree(support_points)
        dist_to_support, _ = support_tree.query(points, k=1)
        support_label = (dist_to_support <= self.support_align_threshold_cm).astype(np.float32)

        active_tree = cKDTree(points)
        support_to_active_dist, _ = active_tree.query(support_points, k=1)
        support_align_coverage = float(np.mean(support_to_active_dist <= self.support_align_threshold_cm))

        return {
            "source_name": item.source_name,
            "sample_id": item.sample_id,
            "object_id": item.object_id,
            "instruction": item.instruction,
            "points": points.astype(np.float32),
            "colors": colors.astype(np.uint8),
            "support_label": support_label,
            "source_box_gt": item.source_box_gt,
            "support_align_coverage": support_align_coverage,
        }


def stage1_collate(batch: list[dict[str, Any]], voxel_size_cm: float = 1.0) -> dict[str, Any]:
    """Collate active voxel samples into one sparse-conv batch."""
    features = []
    sparse_coords = []
    world_coords = []
    coords_norm = []
    batch_indices = []
    support_labels = []
    scene_min = []
    scene_max = []
    source_boxes = []
    instructions = []
    source_names = []
    sample_ids = []
    object_ids = []
    align_coverages = []
    spatial_max = np.zeros(3, dtype=np.int64)

    for batch_idx, item in enumerate(batch):
        points = np.asarray(item["points"], dtype=np.float32)
        colors = np.asarray(item["colors"], dtype=np.float32) / 255.0
        point_min = points.min(axis=0)
        point_max = points.max(axis=0)
        extent = np.maximum(point_max - point_min, 1e-4)
        point_norm = (points - point_min) / extent

        voxel_keys = np.floor(points / float(voxel_size_cm)).astype(np.int64)
        shifted_keys = voxel_keys - voxel_keys.min(axis=0, keepdims=True)
        spatial_max = np.maximum(spatial_max, shifted_keys.max(axis=0) + 1)
        batch_col = np.full((len(points), 1), batch_idx, dtype=np.int32)

        features.append(np.concatenate([point_norm, colors], axis=1).astype(np.float32))
        sparse_coords.append(np.concatenate([batch_col, shifted_keys.astype(np.int32)], axis=1))
        world_coords.append(points)
        coords_norm.append(point_norm.astype(np.float32))
        batch_indices.append(np.full(len(points), batch_idx, dtype=np.int64))
        support_labels.append(np.asarray(item["support_label"], dtype=np.float32))
        scene_min.append(point_min.astype(np.float32))
        scene_max.append(point_max.astype(np.float32))
        source_boxes.append(np.asarray(item["source_box_gt"], dtype=np.float32))
        instructions.append(str(item["instruction"]))
        source_names.append(str(item["source_name"]))
        sample_ids.append(str(item["sample_id"]))
        object_ids.append(str(item["object_id"]))
        align_coverages.append(float(item["support_align_coverage"]))

    return {
        "features": torch.from_numpy(np.concatenate(features, axis=0)),
        "sparse_coords": torch.from_numpy(np.concatenate(sparse_coords, axis=0)),
        "spatial_shape": [int(x) for x in spatial_max.tolist()],
        "world_coords": torch.from_numpy(np.concatenate(world_coords, axis=0)),
        "coords_norm": torch.from_numpy(np.concatenate(coords_norm, axis=0)),
        "batch_indices": torch.from_numpy(np.concatenate(batch_indices, axis=0)),
        "support_labels": torch.from_numpy(np.concatenate(support_labels, axis=0)),
        "source_box_gt": torch.from_numpy(np.stack(source_boxes, axis=0)),
        "scene_min": torch.from_numpy(np.stack(scene_min, axis=0)),
        "scene_max": torch.from_numpy(np.stack(scene_max, axis=0)),
        "instructions": instructions,
        "source_names": source_names,
        "sample_ids": sample_ids,
        "object_ids": object_ids,
        "support_align_coverage": torch.tensor(align_coverages, dtype=torch.float32),
        "batch_size": len(batch),
    }


def move_batch_to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    """Move tensor fields to a device and keep metadata on CPU/Python."""
    out = {}
    for key, value in batch.items():
        out[key] = value.to(device) if torch.is_tensor(value) else value
    return out


def source_box_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    lambda_center: float,
    lambda_size: float,
    lambda_iou: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Source box center/size/axis-aligned IoU loss."""
    center_loss = F.smooth_l1_loss(pred[:, :3], target[:, :3])
    size_loss = F.smooth_l1_loss(pred[:, 3:6], target[:, 3:6])
    iou = aabb_iou_3d(pred, target)
    iou_loss = 1.0 - iou.mean()
    total = lambda_center * center_loss + lambda_size * size_loss + lambda_iou * iou_loss
    return total, {
        "loss_src_center": center_loss.detach(),
        "loss_src_size": size_loss.detach(),
        "loss_src_iou": iou_loss.detach(),
        "source_iou": iou.mean().detach(),
    }


def support_loss(logits: torch.Tensor, labels: torch.Tensor, pos_weight: float | None = None) -> torch.Tensor:
    """Binary support surface loss over active voxels."""
    weight = None
    if pos_weight is not None:
        weight = torch.tensor(float(pos_weight), dtype=logits.dtype, device=logits.device)
    return F.binary_cross_entropy_with_logits(logits, labels, pos_weight=weight)


def compute_stage1_loss(outputs: dict[str, torch.Tensor], batch: dict[str, torch.Tensor], cfg: dict[str, Any]) -> dict[str, torch.Tensor]:
    """Compute total Stage 1 loss and detached logging terms."""
    loss_cfg = cfg["loss"]
    src_loss, src_terms = source_box_loss(
        outputs["source_box"],
        batch["source_box_gt"],
        lambda_center=float(loss_cfg["source"]["lambda_center"]),
        lambda_size=float(loss_cfg["source"]["lambda_size"]),
        lambda_iou=float(loss_cfg["source"]["lambda_iou"]),
    )
    sup_loss = support_loss(
        outputs["support_logits"],
        batch["support_labels"],
        pos_weight=loss_cfg["support"].get("pos_weight"),
    )
    total = float(loss_cfg["lambda_src"]) * src_loss + float(loss_cfg["lambda_sup"]) * sup_loss
    terms = {
        "loss": total,
        "loss_src": src_loss.detach(),
        "loss_sup": sup_loss.detach(),
    }
    terms.update(src_terms)
    return terms


def compute_stage1_metrics(outputs: dict[str, torch.Tensor], batch: dict[str, torch.Tensor]) -> dict[str, float]:
    """Compute scalar validation metrics for Stage 1."""
    pred = outputs["source_box"].detach()
    target = batch["source_box_gt"]
    center_mae = torch.mean(torch.abs(pred[:, :3] - target[:, :3]))
    size_mae = torch.mean(torch.abs(pred[:, 3:6] - target[:, 3:6]))
    iou = aabb_iou_3d(pred, target).mean()

    support_pred = torch.sigmoid(outputs["support_logits"].detach()) >= 0.5
    support_gt = batch["support_labels"] >= 0.5
    tp = torch.count_nonzero(support_pred & support_gt).float()
    fp = torch.count_nonzero(support_pred & ~support_gt).float()
    fn = torch.count_nonzero(~support_pred & support_gt).float()
    precision = tp / (tp + fp + 1e-6)
    recall = tp / (tp + fn + 1e-6)
    f1 = 2.0 * precision * recall / (precision + recall + 1e-6)

    return {
        "source_center_mae_cm": float(center_mae.cpu()),
        "source_size_mae_cm": float(size_mae.cpu()),
        "source_iou": float(iou.cpu()),
        "support_precision": float(precision.cpu()),
        "support_recall": float(recall.cpu()),
        "support_f1": float(f1.cpu()),
        "support_align_coverage": float(batch["support_align_coverage"].mean().cpu()),
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
        collate_fn=lambda batch: stage1_collate(batch, voxel_size_cm=float(data_cfg["voxel_size_cm"])),
        pin_memory=bool(train_cfg.get("pin_memory", True)),
    )


def _mean_dict(rows: list[dict[str, float]]) -> dict[str, float]:
    if not rows:
        return {}
    keys = rows[0].keys()
    return {key: float(np.mean([row[key] for row in rows])) for key in keys}


def _append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False) + "\n")


def _best_support_f1_from_metrics(path: Path) -> float:
    """Read the best validation support F1 already written to metrics.jsonl."""
    if not path.exists():
        return -math.inf
    best = -math.inf
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("split") in {"val", "valid"} and "support_f1" in row:
                best = max(best, float(row["support_f1"]))
    return best


def _distributed_requested() -> bool:
    """Return whether this process was launched by torchrun."""
    return "RANK" in os.environ and "WORLD_SIZE" in os.environ


def _init_distributed() -> tuple[bool, int, int, int]:
    """Initialize DDP from torchrun environment variables when present."""
    if not _distributed_requested():
        return False, 0, 1, 0
    if not torch.cuda.is_available():
        raise RuntimeError("DDP training requires CUDA devices.")
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    if local_rank >= torch.cuda.device_count():
        raise RuntimeError(
            f"LOCAL_RANK={local_rank} exceeds visible CUDA devices ({torch.cuda.device_count()}). "
            "Set --nproc_per_node to the number of visible GPUs."
        )
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl")
    return True, rank, world_size, local_rank


def _is_main_process(rank: int) -> bool:
    return rank == 0


def _resolve_device(cfg: dict[str, Any], distributed: bool = False, local_rank: int = 0) -> torch.device:
    if distributed:
        return torch.device("cuda", local_rank)
    requested = str(cfg["training"].get("device", "auto"))
    if requested == "auto":
        requested = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(requested)
    if str(cfg["model"]["backbone"].get("type", "spconv")).lower() == "spconv" and device.type != "cuda":
        raise RuntimeError("spconv backbone requires CUDA in this environment; no CPU fallback is enabled.")
    return device


def _progress(iterable: Any, enabled: bool, **kwargs: Any) -> Any:
    """Wrap an iterable in tqdm only on the main process."""
    if not enabled or tqdm is None:
        return iterable
    return tqdm(iterable, dynamic_ncols=True, leave=False, **kwargs)


def _reduce_mean_dict(metrics: dict[str, float], device: torch.device, distributed: bool) -> dict[str, float]:
    """Average scalar metrics across DDP ranks."""
    if not distributed or not metrics:
        return metrics
    keys = sorted(metrics)
    values = torch.tensor([metrics[key] for key in keys], dtype=torch.float64, device=device)
    dist.all_reduce(values, op=dist.ReduceOp.SUM)
    values /= dist.get_world_size()
    return {key: float(value.cpu()) for key, value in zip(keys, values)}


def train_stage1(
    cfg: dict[str, Any],
    resume_checkpoint: str | Path | None = None,
    max_steps: int | None = None,
    max_train_samples: int | None = None,
    max_val_samples: int | None = None,
) -> None:
    """Run Stage 1 training and validation."""
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
        all_items = build_stage1_index(sources)
        split_dir = data_cfg.get("split_dir")
        valid_fraction = float(data_cfg.get("valid_fraction", data_cfg.get("val_fraction", 0.1)))
        train_set = LCBGPlaceNetStage1Dataset(
            sources=None,
            split="train",
            val_fraction=valid_fraction,
            seed=int(data_cfg.get("split_seed", 0)),
            support_align_threshold_cm=float(data_cfg["support_align_threshold_cm"]),
            max_samples=max_train_samples or data_cfg.get("max_train_samples"),
            items=all_items,
            split_dir=split_dir,
        )
        val_set = LCBGPlaceNetStage1Dataset(
            sources=None,
            split="valid",
            val_fraction=valid_fraction,
            seed=int(data_cfg.get("split_seed", 0)),
            support_align_threshold_cm=float(data_cfg["support_align_threshold_cm"]),
            max_samples=max_val_samples or data_cfg.get("max_valid_samples", data_cfg.get("max_val_samples")),
            items=all_items,
            split_dir=split_dir,
        )

        model = LCBGPlaceNetStage1(cfg["model"]).to(device)
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
        best_f1 = -math.inf
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
            best_f1 = float(checkpoint.get("best_f1", _best_support_f1_from_metrics(metrics_path)))
            if _is_main_process(rank):
                print(
                    f"Resumed from {resume_path} at epoch={start_epoch}, "
                    f"step={global_step}, best_support_f1={best_f1:.6f}"
                )

        if _is_main_process(rank):
            print(f"Training on {world_size} GPU process(es); per-GPU batch_size={cfg['training']['batch_size']}")

        for epoch in range(start_epoch, int(cfg["training"]["epochs"])):
            if train_sampler is not None:
                train_sampler.set_epoch(epoch)
            model.train()
            train_iter = _progress(train_loader, show_progress, desc=f"train epoch {epoch}", total=len(train_loader))
            for batch in train_iter:
                global_step += 1
                batch = move_batch_to_device(batch, device)
                optimizer.zero_grad(set_to_none=True)
                outputs = model(batch)
                losses = compute_stage1_loss(outputs, batch, cfg)
                losses["loss"].backward()
                torch.nn.utils.clip_grad_norm_(trainable_model.parameters(), float(cfg["training"].get("grad_clip_norm", 1.0)))
                optimizer.step()

                if show_progress and tqdm is not None:
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

            val_metrics = evaluate_stage1(model, val_loader, cfg, device, show_progress=show_progress, distributed=distributed)
            if _is_main_process(rank):
                val_payload = {"split": "valid", "epoch": epoch, "step": global_step, **val_metrics}
                _append_jsonl(metrics_path, val_payload)
                print(json.dumps(val_payload, ensure_ascii=False))

                checkpoint = {
                    "model": trainable_model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "epoch": epoch,
                    "step": global_step,
                    "best_f1": best_f1,
                    "config": cfg,
                }
                if val_metrics.get("support_f1", -math.inf) > best_f1:
                    best_f1 = val_metrics["support_f1"]
                    checkpoint["best_f1"] = best_f1
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
def evaluate_stage1(
    model: LCBGPlaceNetStage1 | DistributedDataParallel,
    loader: DataLoader,
    cfg: dict[str, Any],
    device: torch.device,
    show_progress: bool = False,
    distributed: bool = False,
) -> dict[str, float]:
    """Evaluate Stage 1 on one dataloader."""
    model.eval()
    rows = []
    val_iter = _progress(loader, show_progress, desc="valid", total=len(loader))
    for batch in val_iter:
        batch = move_batch_to_device(batch, device)
        outputs = model(batch)
        losses = compute_stage1_loss(outputs, batch, cfg)
        metrics = compute_stage1_metrics(outputs, batch)
        rows.append({**{k: float(v.detach().cpu()) for k, v in losses.items()}, **metrics})
    return _reduce_mean_dict(_mean_dict(rows), device, distributed)

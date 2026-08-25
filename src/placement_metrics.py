"""Shared geometric metrics for 3D placement evaluation."""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
from scipy.ndimage import binary_closing, binary_fill_holes, label


SUPPORT_STRUCTURE_8 = np.ones((3, 3), dtype=bool)


@dataclass(frozen=True)
class ConnectedSupportRegion:
    """Filled 8-connected support components projected from one Z band."""

    origin_xy: np.ndarray
    labels: np.ndarray


def compute_aabb_iou_3d(pred_box: np.ndarray, gt_box: np.ndarray) -> float:
    """Compute full 3D IoU for center-size axis-aligned boxes."""
    pred = np.asarray(pred_box, dtype=np.float64)
    gt = np.asarray(gt_box, dtype=np.float64)
    pred_dims = np.maximum(pred[3:6], 0.0)
    gt_dims = np.maximum(gt[3:6], 0.0)
    pred_min, pred_max = pred[:3] - pred_dims * 0.5, pred[:3] + pred_dims * 0.5
    gt_min, gt_max = gt[:3] - gt_dims * 0.5, gt[:3] + gt_dims * 0.5
    intersection = float(np.prod(np.maximum(np.minimum(pred_max, gt_max) - np.maximum(pred_min, gt_min), 0.0)))
    union = float(np.prod(pred_dims) + np.prod(gt_dims) - intersection)
    return intersection / union if union > 0.0 else 0.0


def compute_size_iou(pred_dims: np.ndarray, gt_dims: np.ndarray) -> float:
    """Compute dimensions-only IoU while preserving the length/width/height order."""
    pred = np.maximum(np.asarray(pred_dims, dtype=np.float64), 0.0)
    gt = np.maximum(np.asarray(gt_dims, dtype=np.float64), 0.0)
    intersection = float(np.prod(np.minimum(pred, gt)))
    union = float(np.prod(np.maximum(pred, gt)))
    return intersection / union if union > 0.0 else 0.0


def footprint_voxel_keys(place_box: np.ndarray, voxel_size_cm: float) -> np.ndarray:
    """Return XY voxel cells with positive-area intersection with a yawed footprint."""
    box = np.asarray(place_box, dtype=np.float64)
    center = box[:2]
    half = np.maximum(box[3:5], 1e-6) * 0.5
    yaw = float(box[6])
    c, s = math.cos(yaw), math.sin(yaw)
    axes = np.array([[c, s], [-s, c]], dtype=np.float64)
    corners = center[None, :] + np.array(
        [
            half[0] * axes[0] + half[1] * axes[1],
            half[0] * axes[0] - half[1] * axes[1],
            -half[0] * axes[0] + half[1] * axes[1],
            -half[0] * axes[0] - half[1] * axes[1],
        ]
    )
    voxel_size = float(voxel_size_cm)
    lo = np.floor(corners.min(axis=0) / voxel_size).astype(np.int64) - 1
    hi = np.floor(corners.max(axis=0) / voxel_size).astype(np.int64) + 1
    grid_x, grid_y = np.meshgrid(
        np.arange(lo[0], hi[0] + 1),
        np.arange(lo[1], hi[1] + 1),
        indexing="ij",
    )
    keys = np.column_stack([grid_x.ravel(), grid_y.ravel()])
    delta = (keys.astype(np.float64) + 0.5) * voxel_size - center

    separating_axes = np.vstack([np.eye(2), axes])
    center_distance = np.abs(delta @ separating_axes.T)
    cell_radius = 0.5 * voxel_size * np.abs(separating_axes).sum(axis=1)
    box_radius = np.abs(separating_axes @ axes.T) @ half
    intersects = np.all(center_distance < cell_radius + box_radius - 1e-9, axis=1)
    return keys[intersects]


def quantize_occupied_points(points_world: np.ndarray, voxel_size_cm: float) -> np.ndarray:
    """Quantize the complete scene cloud; source-object voxels remain included."""
    return np.unique(
        np.floor(np.asarray(points_world, dtype=np.float64) / float(voxel_size_cm)).astype(np.int64),
        axis=0,
    )


def support_z_key_bounds(
    bottom_z: float,
    downward_cm: float,
    upper_cm: float,
    voxel_size_cm: float,
) -> tuple[int, int]:
    """Convert the center-inclusive world-space support band to voxel-key bounds."""
    voxel_size = float(voxel_size_cm)
    lower = math.ceil((float(bottom_z) - float(downward_cm)) / voxel_size - 0.5)
    upper = math.floor((float(bottom_z) + float(upper_cm)) / voxel_size - 0.5)
    return int(lower), int(upper)


def build_connected_support_region(
    occupied_keys: np.ndarray,
    z_min: int,
    z_max: int,
) -> ConnectedSupportRegion:
    """Close one-cell gaps, fill enclosed holes, then label 8-connected support."""
    keys = np.asarray(occupied_keys, dtype=np.int64)
    band = keys[(keys[:, 2] >= int(z_min)) & (keys[:, 2] <= int(z_max))]
    if len(band) == 0:
        return ConnectedSupportRegion(np.zeros(2, dtype=np.int64), np.zeros((1, 1), dtype=np.int32))

    xy = np.unique(band[:, :2], axis=0)
    origin = xy.min(axis=0) - 1
    shape = xy.max(axis=0) - origin + 2
    raw = np.zeros(tuple(shape.tolist()), dtype=bool)
    local = xy - origin
    raw[local[:, 0], local[:, 1]] = True
    closed = binary_closing(raw, structure=SUPPORT_STRUCTURE_8)
    filled = binary_fill_holes(closed)
    labels, _ = label(filled, structure=SUPPORT_STRUCTURE_8)
    return ConnectedSupportRegion(origin, labels.astype(np.int32))


def compute_supported_and_stable(
    place_box: np.ndarray,
    occupied_keys: np.ndarray,
    voxel_size_cm: float,
    downward_cm: float = 3.0,
    upper_cm: float = 1.0,
    cache: dict[tuple[int, int], ConnectedSupportRegion] | None = None,
) -> tuple[bool, float]:
    """Require the complete footprint to lie in the center's filled component."""
    box = np.asarray(place_box, dtype=np.float64)
    bottom_z = float(box[2] - box[5] * 0.5)
    z_min, z_max = support_z_key_bounds(bottom_z, downward_cm, upper_cm, voxel_size_cm)
    region_cache = cache if cache is not None else {}
    if (z_min, z_max) not in region_cache:
        region_cache[(z_min, z_max)] = build_connected_support_region(occupied_keys, z_min, z_max)
    region = region_cache[(z_min, z_max)]

    center_key = np.floor(box[:2] / float(voxel_size_cm)).astype(np.int64)
    center_local = center_key - region.origin_xy
    center_inside = np.all(center_local >= 0) and np.all(center_local < np.asarray(region.labels.shape))
    component_id = int(region.labels[tuple(center_local)]) if center_inside else 0
    footprint = footprint_voxel_keys(box, voxel_size_cm)
    if component_id == 0 or len(footprint) == 0:
        return False, 0.0

    local = footprint - region.origin_xy
    in_bounds = np.all(local >= 0, axis=1) & np.all(local < np.asarray(region.labels.shape), axis=1)
    covered = np.zeros(len(footprint), dtype=bool)
    valid = local[in_bounds]
    covered[in_bounds] = region.labels[valid[:, 0], valid[:, 1]] == component_id
    coverage = float(covered.mean())
    return bool(np.all(covered)), coverage


def compute_yaw_valid_at_matched_center(
    place_box: np.ndarray,
    predicted_yaw_bin: int,
    gt_bottom_centers: np.ndarray,
    gt_yaw_masks: np.ndarray,
    center_match_threshold_cm: float,
) -> tuple[bool, float | None, int | None]:
    """Match the nearest GT bottom center, then check its valid-yaw set."""
    centers = np.asarray(gt_bottom_centers, dtype=np.float64)
    masks = np.asarray(gt_yaw_masks, dtype=bool)
    if len(centers) == 0:
        return False, None, None
    box = np.asarray(place_box, dtype=np.float64)
    bottom_center = box[:3].copy()
    bottom_center[2] -= box[5] * 0.5
    distances = np.linalg.norm(centers - bottom_center[None, :], axis=1)
    matched = int(np.argmin(distances))
    distance = float(distances[matched])
    yaw_bin = int(predicted_yaw_bin)
    valid_bin = 0 <= yaw_bin < masks.shape[1]
    valid = distance <= float(center_match_threshold_cm) and valid_bin and bool(masks[matched, yaw_bin])
    return bool(valid), distance, matched


def placement_success(
    placement_size_correct: bool,
    language_relation_correct: bool,
    supported_and_stable: bool,
    collision_free: bool,
) -> bool:
    """Return the four-condition Placement Success decision for one candidate."""
    return bool(
        placement_size_correct
        and language_relation_correct
        and supported_and_stable
        and collision_free
    )

"""
src/annotation/free_bbox/collision.py
-------------------------------------
FFT 碰撞检测：在 (X, Y, yaw) 配置空间中搜索放置候选。

相对旧实现的关键变化：
1. landing_z 使用支撑面 table_z 本层，而不是 table_z + 1；
2. 碰撞障碍中清除支撑面自身，允许 3D box 的底层落在支撑面体素层。
"""

from __future__ import annotations

import math

import numpy as np
from scipy.signal import fftconvolve

from src.annotation.free_bbox.geometry import (
    compute_placed_transform,
    compute_placed_transform_with_orientation,
    rotation_matrix_to_euler_zyx,
)
from src.annotation.free_bbox.grid_ops import dilate_obstacles_xy, voxelize_obb
from src.annotation.free_bbox.occupancy import OCCUPIED


def _compute_collision_slice(
    obstacle: np.ndarray,
    obj_mask: np.ndarray,
    landing_z: int,
) -> np.ndarray | None:
    """
    在 landing_z 高度计算 2D 碰撞图。

    仅对物体实际占用层做 2D FFT 卷积，保留旧 free_bbox 的搜索方式。
    """
    grid_x, grid_y, grid_z = obstacle.shape
    obj_x, obj_y, obj_z = obj_mask.shape
    out_x, out_y = grid_x - obj_x + 1, grid_y - obj_y + 1
    if out_x <= 0 or out_y <= 0:
        return None

    collision = np.zeros((out_x, out_y), dtype=np.float32)
    for dz in range(obj_z):
        z_idx = int(landing_z) + dz
        if z_idx < 0 or z_idx >= grid_z:
            continue
        obj_slice = obj_mask[:, :, dz]
        if not np.any(obj_slice):
            continue
        collision += fftconvolve(
            obstacle[:, :, z_idx].astype(np.float32),
            obj_slice[::-1, ::-1].astype(np.float32),
            mode="valid",
        )
    return collision


def _make_collision_obstacle(
    grid_work: np.ndarray,
    table_z: int,
    surface_mask_2d: np.ndarray | None,
    margin_voxels: int,
) -> np.ndarray:
    """构建碰撞障碍，并清掉支撑面本层。"""
    obstacle = grid_work == OCCUPIED
    if surface_mask_2d is not None and 0 <= int(table_z) < obstacle.shape[2]:
        obstacle = np.array(obstacle, copy=True)
        layer = obstacle[:, :, int(table_z)]
        layer[np.asarray(surface_mask_2d, dtype=bool)] = False
        obstacle[:, :, int(table_z)] = layer
    return dilate_obstacles_xy(obstacle, margin_voxels)


def _surface_roi(
    surface_mask_2d: np.ndarray | None,
    grid_shape: tuple[int, int, int],
    pad: int = 5,
) -> tuple[int, int, int, int]:
    """根据支撑面裁剪 XY 搜索区域。"""
    grid_x, grid_y, _ = grid_shape
    if surface_mask_2d is None or not np.any(surface_mask_2d):
        return 0, 0, grid_x, grid_y

    nz_x, nz_y = np.where(surface_mask_2d)
    return (
        max(int(nz_x.min()) - int(pad), 0),
        max(int(nz_y.min()) - int(pad), 0),
        min(int(nz_x.max()) + int(pad) + 1, grid_x),
        min(int(nz_y.max()) + int(pad) + 1, grid_y),
    )


def find_table_placements(
    grid_work: np.ndarray,
    bbox3d: np.ndarray,
    T_obj2world: np.ndarray,
    vp: dict,
    table_z: int,
    surface_mask_2d: np.ndarray | None,
    safety_margin: float = 0.5,
    yaw_steps: int = 24,
    preserve_orientation: bool = True,
) -> tuple[np.ndarray, dict, dict]:
    """
    在支撑面本层搜索无碰撞放置位置。

    输出 candidates 格式为 (N, 3)，每行为 [grid_x, grid_y, yaw_index]。
    """
    pose_info = None
    if preserve_orientation:
        roll, pitch, original_yaw = rotation_matrix_to_euler_zyx(
            np.asarray(T_obj2world, dtype=np.float64)[:3, :3]
        )
        pose_info = {
            "roll": roll,
            "pitch": pitch,
            "yaw": original_yaw,
            "preserve_orientation": True,
        }

    voxel_size = float(vp["voxel_size"])
    grid_shape = np.asarray(grid_work.shape, dtype=int)
    grid_x, grid_y, _ = grid_shape
    landing_z = int(table_z)
    margin_voxels = max(0, int(math.ceil(float(safety_margin) / voxel_size)))
    obstacle = _make_collision_obstacle(
        grid_work,
        table_z=table_z,
        surface_mask_2d=surface_mask_2d,
        margin_voxels=margin_voxels,
    )

    roi_x0, roi_y0, roi_x1, roi_y1 = _surface_roi(surface_mask_2d, grid_work.shape)
    obstacle_roi = obstacle[roi_x0:roi_x1, roi_y0:roi_y1, :]

    bbox = np.asarray(bbox3d, dtype=np.float64)
    bbox_center = (bbox[:3] + bbox[3:]) * 0.5
    obj_center_world = (np.asarray(T_obj2world, dtype=np.float64) @ np.append(bbox_center, 1.0))[:3]
    yaw_angles = np.linspace(0.0, 2.0 * np.pi, int(yaw_steps), endpoint=False)

    all_candidates = []
    yaw_rel_voxels = []
    yaw_vmin_rot = []
    yaw_T_rotated = []
    yaw_footprints = []
    valid_yaw_count = 0
    total_raw = 0

    empty_voxels = np.empty((0, 3), dtype=int)
    empty_footprint = np.empty((0, 2), dtype=int)
    zero3 = np.zeros(3, dtype=np.float64)

    for yaw_idx, angle in enumerate(yaw_angles):
        if pose_info is not None:
            T_rot = compute_placed_transform_with_orientation(
                bbox,
                obj_center_world,
                pose_info["roll"],
                pose_info["pitch"],
                float(angle),
            )
        else:
            T_rot = compute_placed_transform(bbox, obj_center_world, float(angle))

        rot_voxels = voxelize_obb(bbox, T_rot, vp, grid_shape)
        if len(rot_voxels) == 0:
            yaw_rel_voxels.append(empty_voxels)
            yaw_vmin_rot.append(zero3.copy())
            yaw_T_rotated.append(T_rot)
            yaw_footprints.append(empty_footprint)
            continue

        vmin_rot = rot_voxels.min(axis=0).astype(np.float64)
        rel_rot = rot_voxels - rot_voxels.min(axis=0)
        obj_size = rel_rot.max(axis=0) + 1
        obj_mask = np.zeros(tuple(obj_size), dtype=bool)
        obj_mask[rel_rot[:, 0], rel_rot[:, 1], rel_rot[:, 2]] = True
        footprint = np.unique(rel_rot[:, :2], axis=0)

        collision = _compute_collision_slice(obstacle_roi, obj_mask, landing_z)
        if collision is None:
            yaw_rel_voxels.append(rel_rot)
            yaw_vmin_rot.append(vmin_rot)
            yaw_T_rotated.append(T_rot)
            yaw_footprints.append(footprint)
            continue

        free_mask = collision < 0.5
        cand_x, cand_y = np.where(free_mask)
        cand_x += roi_x0
        cand_y += roi_y0
        n_free = len(cand_x)
        total_raw += n_free

        if n_free > 0:
            yaw_col = np.full(n_free, yaw_idx, dtype=int)
            all_candidates.append(np.stack([cand_x, cand_y, yaw_col], axis=1))
            valid_yaw_count += 1

        yaw_rel_voxels.append(rel_rot)
        yaw_vmin_rot.append(vmin_rot.astype(np.float64))
        yaw_T_rotated.append(T_rot)
        yaw_footprints.append(footprint)

    candidates = np.vstack(all_candidates) if all_candidates else np.empty((0, 3), dtype=int)
    yaw_data = {
        "yaw_angles": yaw_angles,
        "rel_voxels": yaw_rel_voxels,
        "vmin_rot_abs": yaw_vmin_rot,
        "T_rotated": yaw_T_rotated,
        "footprints": yaw_footprints,
        "original_yaw_index": 0,
        "pose_info": pose_info,
    }
    meta = {
        "total_xy": int(grid_x * grid_y),
        "valid_raw": int(total_raw),
        "yaw_steps": int(yaw_steps),
        "valid_yaw_angles": int(valid_yaw_count),
        "landing_z": int(landing_z),
        "table_z": int(table_z),
        "safety_margin": float(safety_margin),
    }
    return candidates, meta, yaw_data

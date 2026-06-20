"""
src/annotation/free_bbox/grid_ops.py
------------------------------------
占据栅格操作：OBB 体素化、物体写入和障碍膨胀。
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import binary_dilation, generate_binary_structure

from src.annotation.free_bbox.geometry import get_bbox_corners, transform_points
from src.annotation.free_bbox.occupancy import OCCUPIED


def _enumerate_voxel_candidates(
    corners_world: np.ndarray,
    origin: np.ndarray,
    voxel_size: float,
    grid_shape: np.ndarray,
) -> np.ndarray:
    """根据世界坐标包围盒枚举可能被 OBB 覆盖的体素索引。"""
    voxel_size = float(voxel_size)
    grid_shape = np.asarray(grid_shape, dtype=int)
    lo = np.maximum(
        np.floor((corners_world.min(axis=0) - voxel_size - origin) / voxel_size).astype(int),
        0,
    )
    hi = np.minimum(
        np.ceil((corners_world.max(axis=0) + voxel_size - origin) / voxel_size).astype(int),
        grid_shape,
    )
    ranges = [np.arange(lo[axis], hi[axis]) for axis in range(3)]
    if any(len(axis_range) == 0 for axis_range in ranges):
        return np.empty((0, 3), dtype=int)

    gx, gy, gz = np.meshgrid(*ranges, indexing="ij")
    return np.stack([gx.ravel(), gy.ravel(), gz.ravel()], axis=1)


def voxelize_obb(
    bbox3d: np.ndarray,
    T_obj2world: np.ndarray,
    vp: dict,
    grid_shape: np.ndarray,
) -> np.ndarray:
    """
    将物体 OBB 转换为体素索引集合。

    枚举 OBB 世界坐标包围盒附近的体素，再使用分离轴定理检查体素 AABB
    是否与 OBB 相交。边界接触也视为相交，避免薄物体因体素中心位于 OBB
    外部而完全漏检。
    """
    origin = np.asarray(vp["origin"], dtype=np.float64)
    voxel_size = float(vp["voxel_size"])
    bbox = np.asarray(bbox3d, dtype=np.float64)
    bmin, bmax = bbox[:3], bbox[3:]
    center_obj = (bmin + bmax) * 0.5
    half_extent_obj = (bmax - bmin) * 0.5
    transform = np.asarray(T_obj2world, dtype=np.float64)
    rotation_scale = transform[:3, :3]
    axis_scales = np.linalg.norm(rotation_scale, axis=0)
    if np.any(axis_scales < 1e-12):
        return np.empty((0, 3), dtype=int)

    obb_axes = rotation_scale / axis_scales[None, :]
    half_extent_world = half_extent_obj * axis_scales
    center_world = transform_points(center_obj[None, :], transform)[0]
    corners_world = transform_points(get_bbox_corners(bbox), transform)
    indices = _enumerate_voxel_candidates(corners_world, origin, voxel_size, grid_shape)
    if len(indices) == 0:
        return indices

    world_axes = np.eye(3, dtype=np.float64)
    cross_axes = np.cross(world_axes[:, None, :], obb_axes.T[None, :, :]).reshape(-1, 3)
    separating_axes = np.vstack([world_axes, obb_axes.T, cross_axes])
    axis_norms = np.linalg.norm(separating_axes, axis=1)
    separating_axes = separating_axes[axis_norms > 1e-12]
    separating_axes /= np.linalg.norm(separating_axes, axis=1, keepdims=True)

    centers_world = origin + (indices + 0.5) * voxel_size
    center_distances = np.abs((centers_world - center_world) @ separating_axes.T)
    voxel_radii = 0.5 * voxel_size * np.abs(separating_axes).sum(axis=1)
    obb_radii = np.abs(separating_axes @ obb_axes) @ half_extent_world
    intersects = np.all(center_distances <= voxel_radii + obb_radii + 1e-9, axis=1)
    return indices[intersects]


def prepare_grid_base(grid: np.ndarray, objects: list, vp: dict) -> np.ndarray:
    """将所有场景物体 OBB 写入栅格，作为碰撞搜索的基础障碍。"""
    grid_base = np.array(grid, copy=True)
    grid_shape = np.asarray(grid_base.shape, dtype=int)
    for obj in objects:
        voxels = voxelize_obb(obj.bbox3d_canonical, obj.pose_world, vp, grid_shape)
        if len(voxels) > 0:
            grid_base[voxels[:, 0], voxels[:, 1], voxels[:, 2]] = OCCUPIED
    return grid_base


def dilate_obstacles_xy(obstacle: np.ndarray, margin_voxels: int) -> np.ndarray:
    """
    在 XY 平面对 bool 障碍物做膨胀。

    当前 canonical 体素点云不含 UNKNOWN 状态，因此调用方直接传入 OCCUPIED
    障碍物 bool mask。
    """
    obstacle = np.asarray(obstacle, dtype=bool)
    if int(margin_voxels) <= 0:
        return obstacle
    structure_2d = generate_binary_structure(2, 1)
    structure_3d = np.zeros((3, 3, 3), dtype=bool)
    structure_3d[:, :, 1] = structure_2d
    return binary_dilation(obstacle, structure=structure_3d, iterations=int(margin_voxels))

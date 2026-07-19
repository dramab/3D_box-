"""
src/annotation/free_bbox/occupancy.py
-------------------------------------
从 canonical 体素点云构建放置搜索使用的占据栅格。

与旧实现不同，这里不再从深度图 ray-casting 得到 FREE/OCCUPIED/UNKNOWN，
而是直接读取规范数据集中的 1cm 体素点云，并把这些体素作为 OCCUPIED。
其余栅格单元设为 FREE，因为规范体素点云本身不提供 unknown/free 射线状态。
"""

from __future__ import annotations

import numpy as np

from src.annotation.free_bbox.voxel_utils import world_to_voxel


FREE, OCCUPIED, UNKNOWN = 0, 1, 2


def build_grid_from_voxel_points(
    points_world: np.ndarray,
    voxel_size: float = 1.0,
    padding: float = 10.0,
    extra_points: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, float]:
    """
    从体素点云构建 OCCUPIED/FREE 栅格。

    输入:
        points_world: (N, 3) canonical voxel PLY 中的世界坐标点
        voxel_size: 体素边长，单位与数据集一致
        padding: 栅格边界扩展
        extra_points: 可选额外点，用于确保物体 OBB 原始位置落入栅格
    输出:
        grid: (Gx, Gy, Gz) uint8，占据体素为 OCCUPIED
        grid_min: (3,) 栅格原点
        voxel_size: float
    """
    points = np.asarray(points_world, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("points_world must have shape (N, 3)")
    if len(points) == 0:
        raise ValueError("points_world must contain at least one point")

    bounds_points = points
    if extra_points is not None:
        extra = np.asarray(extra_points, dtype=np.float64)
        if extra.ndim != 2 or extra.shape[1] != 3:
            raise ValueError("extra_points must have shape (N, 3)")
        if len(extra) > 0:
            bounds_points = np.vstack([points, extra])

    voxel_size = float(voxel_size)
    padding = float(padding)
    # canonical active 点云按 floor(world / voxel_size) 聚合；free_bbox 必须复用
    # 同一世界格线，否则导出的 support/heatmap key 会与模型输入错位。
    grid_min = np.floor((bounds_points.min(axis=0) - padding) / voxel_size) * voxel_size
    grid_max = np.ceil((bounds_points.max(axis=0) + padding) / voxel_size) * voxel_size
    grid_shape = np.maximum(
        np.ceil((grid_max - grid_min) / voxel_size).astype(int),
        1,
    )

    grid = np.full(tuple(grid_shape), FREE, dtype=np.uint8)
    vp = {"voxel_size": voxel_size, "origin": grid_min.tolist()}
    indices = world_to_voxel(points, vp)
    valid = (
        (indices[:, 0] >= 0)
        & (indices[:, 0] < grid_shape[0])
        & (indices[:, 1] >= 0)
        & (indices[:, 1] < grid_shape[1])
        & (indices[:, 2] >= 0)
        & (indices[:, 2] < grid_shape[2])
    )
    indices = indices[valid]
    grid[indices[:, 0], indices[:, 1], indices[:, 2]] = OCCUPIED
    return grid, grid_min, voxel_size

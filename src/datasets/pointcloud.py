"""
src/datasets/pointcloud.py
--------------------------
统一数据预处理阶段使用的深度过滤、点云生成和 PLY 保存工具。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np


def filter_depth_range_by_window(
    depth: np.ndarray,
    depth_window_cm: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    """
    保留近端稳健深度之后固定窗口内的像素，其余位置置 0。

    输入 depth 必须已经转换为统一场景单位 cm。
    """
    depth_arr = np.asarray(depth, dtype=np.float32)
    valid_mask = np.isfinite(depth_arr) & (depth_arr > 0.0)
    valid_depth = depth_arr[valid_mask]
    if valid_depth.size == 0:
        raise ValueError("depth contains no valid positive finite values")

    depth_window_cm = float(depth_window_cm)
    if depth_window_cm <= 0.0:
        raise ValueError(f"depth_window_cm must be positive, got {depth_window_cm}")

    depth_raw_min = float(valid_depth.min())
    depth_min_percentile = 1.0
    # 用低分位数代替绝对最小值，避免单个过近离异点拉偏窗口起点。
    depth_min = float(np.percentile(valid_depth, depth_min_percentile))
    depth_max = float(depth_min + depth_window_cm)
    keep_mask = valid_mask & (depth_arr >= depth_min) & (depth_arr <= depth_max)
    if not np.any(keep_mask):
        raise ValueError("depth filtering removed all valid depth pixels")

    filtered = np.array(depth_arr, copy=True)
    filtered[~keep_mask] = 0.0
    stats = {
        "valid_depth_count_before": int(valid_depth.size),
        "valid_depth_count_after": int(np.count_nonzero(keep_mask)),
        "removed_depth_outlier_count": int(valid_depth.size - np.count_nonzero(keep_mask)),
        "depth_window_cm": depth_window_cm,
        "depth_raw_min": depth_raw_min,
        "depth_min_percentile": depth_min_percentile,
        "depth_min_used": depth_min,
        "depth_max_used": depth_max,
    }
    return filtered, stats


def depth_to_pointcloud(
    depth: np.ndarray,
    rgb: np.ndarray,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    E_c2w: np.ndarray,
    stride: int,
) -> tuple[np.ndarray, np.ndarray]:
    """将统一单位 depth 反投影为世界坐标彩色点云。"""
    if int(stride) <= 0:
        raise ValueError(f"stride must be positive, got {stride}")

    depth_arr = np.asarray(depth, dtype=np.float32)
    h, w = depth_arr.shape
    u_arr = np.arange(0, w, int(stride))
    v_arr = np.arange(0, h, int(stride))
    uu, vv = np.meshgrid(u_arr, v_arr)
    uu = uu.ravel()
    vv = vv.ravel()

    sampled_depth = depth_arr[vv, uu]
    valid = np.isfinite(sampled_depth) & (sampled_depth > 0.0)
    uu = uu[valid]
    vv = vv[valid]
    sampled_depth = sampled_depth[valid]
    if sampled_depth.size == 0:
        raise ValueError("no valid sampled depth points for point cloud generation")

    pts_cam = np.stack(
        [
            (uu - cx) / fx * sampled_depth,
            (vv - cy) / fy * sampled_depth,
            sampled_depth,
        ],
        axis=1,
    )
    E_c2w = np.asarray(E_c2w, dtype=np.float64)
    pts_world = (E_c2w[:3, :3] @ pts_cam.T).T + E_c2w[:3, 3]
    colors = np.asarray(rgb, dtype=np.uint8)[vv, uu]
    return pts_world.astype(np.float32), colors.astype(np.uint8)


def sample_pointcloud(
    points: np.ndarray,
    colors: np.ndarray,
    target_count: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """将点云重采样到固定点数，点数不足时放回采样补足。"""
    target_count = int(target_count)
    if target_count <= 0:
        raise ValueError(f"target_count must be positive, got {target_count}")

    points = np.asarray(points, dtype=np.float32)
    colors = np.asarray(colors, dtype=np.uint8)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("points must have shape (N, 3)")
    if colors.shape != points.shape:
        raise ValueError("colors must have shape (N, 3) and align with points")
    if len(points) == 0:
        raise ValueError("cannot sample an empty point cloud")

    rng = np.random.default_rng(int(seed))
    replace = len(points) < target_count
    indices = rng.choice(len(points), size=target_count, replace=replace)
    return points[indices], colors[indices]


def voxelize_pointcloud(
    points: np.ndarray,
    colors: np.ndarray,
    voxel_size_cm: float,
) -> tuple[np.ndarray, np.ndarray]:
    """按固定 cm 体素聚合点云，每个体素输出一个平均位置和平均颜色点。"""
    voxel_size_cm = float(voxel_size_cm)
    if voxel_size_cm <= 0.0:
        raise ValueError(f"voxel_size_cm must be positive, got {voxel_size_cm}")

    points = np.asarray(points, dtype=np.float32)
    colors = np.asarray(colors, dtype=np.uint8)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("points must have shape (N, 3)")
    if colors.shape != points.shape:
        raise ValueError("colors must have shape (N, 3) and align with points")
    if len(points) == 0:
        raise ValueError("cannot voxelize an empty point cloud")

    voxel_keys = np.floor(points / voxel_size_cm).astype(np.int64)
    _, inverse_indices, counts = np.unique(
        voxel_keys,
        axis=0,
        return_inverse=True,
        return_counts=True,
    )

    voxel_points = np.zeros((len(counts), 3), dtype=np.float32)
    voxel_colors = np.zeros((len(counts), 3), dtype=np.float32)
    np.add.at(voxel_points, inverse_indices, points)
    np.add.at(voxel_colors, inverse_indices, colors.astype(np.float32))

    voxel_points /= counts[:, None]
    voxel_colors = np.rint(voxel_colors / counts[:, None]).clip(0, 255).astype(np.uint8)
    return voxel_points, voxel_colors


def save_ply(path: str | Path, points: np.ndarray, colors: np.ndarray) -> None:
    """保存 ASCII 彩色 PLY 点云。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    points = np.asarray(points, dtype=np.float32)
    colors = np.asarray(colors, dtype=np.uint8)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("points must have shape (N, 3)")
    if colors.shape != points.shape:
        raise ValueError("colors must have shape (N, 3) and align with points")

    header = (
        "ply\nformat ascii 1.0\n"
        f"element vertex {len(points)}\n"
        "property float x\nproperty float y\nproperty float z\n"
        "property uchar red\nproperty uchar green\nproperty uchar blue\n"
        "end_header\n"
    )
    data = np.hstack([points, colors.astype(np.float32)])
    with path.open("w") as f:
        f.write(header)
        np.savetxt(f, data, fmt="%.4f %.4f %.4f %d %d %d")

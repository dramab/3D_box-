"""
src/annotation/free_bbox/voxel_utils.py
---------------------------------------
体素坐标系工具函数。
"""

from __future__ import annotations

import numpy as np


def make_voxel_params(grid_min: np.ndarray, voxel_size: float) -> dict:
    """构建世界坐标和体素索引互转所需的参数。"""
    return {
        "voxel_size": float(voxel_size),
        "origin": np.asarray(grid_min, dtype=np.float64).tolist(),
    }


def world_to_voxel(points: np.ndarray, voxel_params: dict) -> np.ndarray:
    """世界坐标转整数体素索引。"""
    origin = np.asarray(voxel_params["origin"], dtype=np.float64)
    voxel_size = float(voxel_params["voxel_size"])
    return np.floor((np.asarray(points, dtype=np.float64) - origin) / voxel_size).astype(np.intp)


def voxel_to_world(indices: np.ndarray, voxel_params: dict) -> np.ndarray:
    """整数体素索引转体素中心世界坐标。"""
    origin = np.asarray(voxel_params["origin"], dtype=np.float64)
    voxel_size = float(voxel_params["voxel_size"])
    return origin + (np.asarray(indices, dtype=np.float64) + 0.5) * voxel_size

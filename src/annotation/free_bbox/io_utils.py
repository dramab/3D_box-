"""
src/annotation/free_bbox/io_utils.py
------------------------------------
PLY 和 JSON 输出工具。
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import numpy as np

from src.annotation.free_bbox.occupancy import OCCUPIED
from src.annotation.free_bbox.voxel_utils import voxel_to_world


def load_ply(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """
    读取 ASCII PLY 点云。

    当前 canonical 数据保存为 x/y/z/r/g/b 六列；若没有颜色列，则返回灰色。
    """
    path = Path(path)
    with path.open("r") as f:
        vertex_count = None
        while True:
            line = f.readline()
            if not line:
                raise ValueError(f"Invalid PLY header: {path}")
            line = line.strip()
            if line.startswith("element vertex"):
                vertex_count = int(line.split()[-1])
            if line == "end_header":
                break
        if vertex_count is None:
            raise ValueError(f"PLY missing vertex count: {path}")
        if vertex_count == 0:
            return np.empty((0, 3), dtype=np.float32), np.empty((0, 3), dtype=np.uint8)
        data = np.loadtxt(f, max_rows=vertex_count)

    data = np.atleast_2d(data)
    points = data[:, :3].astype(np.float32)
    if data.shape[1] >= 6:
        colors = np.rint(data[:, 3:6]).clip(0, 255).astype(np.uint8)
    else:
        colors = np.full((len(points), 3), 160, dtype=np.uint8)
    return points, colors


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
    data = np.hstack([points.astype(np.float32), colors.astype(np.float32)])
    with path.open("w") as f:
        f.write(header)
        np.savetxt(f, data, fmt="%.4f %.4f %.4f %d %d %d")


def save_json(path: str | Path, payload: dict[str, Any]) -> None:
    """保存 JSON 数据。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(payload, f, indent=2, default=_json_default)


def save_binary_mask_ply(
    path: str | Path,
    grid: np.ndarray,
    vp: dict,
    mask_3d: np.ndarray,
) -> None:
    """
    将整体体素点云保存为二值 mask PLY。

    OCCUPIED 和 mask=True 的体素都会输出；支撑面体素为白色，
    其余为深灰色。这会保留形态学运算补出的非占据体素。
    """
    mask = np.asarray(mask_3d, dtype=bool)
    output_idx = np.argwhere((grid == OCCUPIED) | mask)
    colors = np.full((len(output_idx), 3), [55, 55, 55], dtype=np.uint8)
    if len(output_idx) > 0:
        active = mask[output_idx[:, 0], output_idx[:, 1], output_idx[:, 2]]
        colors[active] = np.array([255, 255, 255], dtype=np.uint8)
    save_ply(path, voxel_to_world(output_idx, vp), colors)


def save_heatmap_ply(
    path: str | Path,
    grid: np.ndarray,
    vp: dict,
    heat_counts: np.ndarray,
    support_mask_3d: np.ndarray | None = None,
) -> None:
    """
    将整体体素点云保存为热力 PLY。

    heat_counts>0 的支撑面体素按黄到红着色；支撑面但计数为 0 的体素为蓝色；
    其他 OCCUPIED 体素为灰色。输出点集是 OCCUPIED、支撑面和正热力
    体素的并集，以保留形态学运算补出的体素。
    """
    counts = np.asarray(heat_counts, dtype=np.int64)
    output_mask = (grid == OCCUPIED) | (counts > 0)
    if support_mask_3d is not None:
        output_mask |= np.asarray(support_mask_3d, dtype=bool)
    output_idx = np.argwhere(output_mask)
    colors = np.full((len(output_idx), 3), [60, 60, 60], dtype=np.uint8)

    if len(output_idx) > 0 and support_mask_3d is not None:
        support_mask = np.asarray(support_mask_3d, dtype=bool)
        support = support_mask[output_idx[:, 0], output_idx[:, 1], output_idx[:, 2]]
        colors[support] = np.array([55, 120, 210], dtype=np.uint8)

    if len(output_idx) > 0:
        values = counts[output_idx[:, 0], output_idx[:, 1], output_idx[:, 2]]
        active = values > 0
        if np.any(active):
            denom = max(float(values[active].max()), 1.0)
            norm = values[active].astype(np.float64) / denom
            heat_colors = np.zeros((int(active.sum()), 3), dtype=np.uint8)
            heat_colors[:, 0] = 255
            heat_colors[:, 1] = np.rint(220.0 * (1.0 - norm)).astype(np.uint8)
            heat_colors[:, 2] = 30
            colors[active] = heat_colors

    save_ply(path, voxel_to_world(output_idx, vp), colors)


def _json_default(value: Any) -> Any:
    """JSON 序列化 numpy 类型的 fallback。"""
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, Path):
        return os.fspath(value)
    raise TypeError(f"Object of type {type(value)} is not JSON serializable")

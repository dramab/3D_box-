"""free_bbox PLY 输出点集测试。"""

import numpy as np

from src.annotation.free_bbox.io_utils import load_ply, save_binary_mask_ply, save_heatmap_ply
from src.annotation.free_bbox.occupancy import OCCUPIED


def _point_colors(points: np.ndarray, colors: np.ndarray) -> dict[tuple[float, ...], tuple[int, ...]]:
    """按点坐标索引 PLY 颜色。"""
    return {
        tuple(point.tolist()): tuple(color.tolist())
        for point, color in zip(points, colors)
    }


def test_binary_mask_ply_keeps_non_occupied_support_voxels(tmp_path) -> None:
    """形态学运算补出的非占据支撑体素应写入 PLY。"""
    grid = np.zeros((3, 3, 1), dtype=np.uint8)
    grid[0, 0, 0] = OCCUPIED
    support = np.zeros_like(grid, dtype=bool)
    support[1, 1, 0] = True
    vp = {"origin": [0.0, 0.0, 0.0], "voxel_size": 1.0}

    output_path = tmp_path / "support_mask.ply"
    save_binary_mask_ply(output_path, grid, vp, support)
    points, colors = load_ply(output_path)
    point_colors = _point_colors(points, colors)

    assert len(points) == 2
    assert point_colors[(1.5, 1.5, 0.5)] == (255, 255, 255)


def test_heatmap_ply_keeps_non_occupied_support_and_heat_voxels(tmp_path) -> None:
    """支撑面和正热力体素不在原始占据网格中时仍应写入 PLY。"""
    grid = np.zeros((4, 3, 1), dtype=np.uint8)
    grid[0, 0, 0] = OCCUPIED
    support = np.zeros_like(grid, dtype=bool)
    support[1, 1, 0] = True
    heat_counts = np.zeros_like(grid, dtype=np.int64)
    heat_counts[2, 1, 0] = 3
    vp = {"origin": [0.0, 0.0, 0.0], "voxel_size": 1.0}

    output_path = tmp_path / "heatmap.ply"
    save_heatmap_ply(output_path, grid, vp, heat_counts, support_mask_3d=support)
    points, colors = load_ply(output_path)
    point_colors = _point_colors(points, colors)

    assert len(points) == 3
    assert point_colors[(1.5, 1.5, 0.5)] == (55, 120, 210)
    assert point_colors[(2.5, 1.5, 0.5)] == (255, 0, 30)

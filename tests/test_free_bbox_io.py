"""free_bbox PLY 输出点集测试。"""

import numpy as np

from src.annotation.free_bbox.io_utils import (
    load_ply,
    save_binary_mask_ply,
    save_center_yaw_set_npz,
    save_heatmap_ply,
)
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


def test_center_yaw_set_npz_round_trip(tmp_path) -> None:
    """中心、yaw mask 和热力计数应无损写入压缩 NPZ。"""
    output_path = tmp_path / "sample__yaw_set.npz"
    save_center_yaw_set_npz(
        output_path,
        bottom_center_voxels=np.array([[1, 2, 3], [4, 5, 6]]),
        bottom_center_world=np.array([[1.5, 2.5, 3.5], [4.5, 5.5, 6.5]]),
        valid_yaw_mask=np.array([[True, False, True], [False, True, False]]),
        yaw_angles_rad=np.array([0.0, np.pi / 2.0, np.pi]),
        heat_counts=np.array([2, 1]),
    )

    with np.load(output_path) as payload:
        assert set(payload.files) == {
            "bottom_center_voxels",
            "bottom_center_world",
            "valid_yaw_mask",
            "yaw_angles_rad",
            "heat_counts",
        }
        assert payload["bottom_center_voxels"].dtype == np.int32
        assert payload["bottom_center_world"].dtype == np.float32
        assert payload["valid_yaw_mask"].dtype == np.bool_
        assert payload["heat_counts"].tolist() == [2, 1]
        assert payload["valid_yaw_mask"].tolist() == [[True, False, True], [False, True, False]]

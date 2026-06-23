"""free_bbox 支撑面排除与保守 OBB 体素化测试。"""

import numpy as np

from src.annotation.free_bbox.cluster import cluster_placements_best
from src.annotation.free_bbox.grid_ops import voxelize_obb
from src.annotation.free_bbox.occupancy import OCCUPIED
from src.annotation.free_bbox.surface import _remove_excluded_surface_voxels


def test_surface_exclusion_projects_obb_within_one_z_voxel() -> None:
    """支撑面应排除 table_z±1 内物体 OBB 的 XY 投影。"""
    surface = np.ones((5, 5), dtype=bool)
    excluded = np.zeros((5, 5, 5), dtype=bool)
    excluded[2, 2, 3] = True  # table_z + 1
    excluded[3, 3, 4] = True  # table_z + 2，不应被排除

    components = _remove_excluded_surface_voxels(
        surface,
        surface_z=2,
        exclude_voxel_mask=excluded,
        min_voxels=1,
    )

    assert len(components) == 1
    _, filtered = components[0]
    assert not filtered[2, 2]
    assert filtered[3, 3]


def test_voxelize_obb_includes_intersecting_voxels_without_center_inside() -> None:
    """薄 OBB 应占据所有与其相交的体素，即使体素中心位于 OBB 外。"""
    bbox = np.array([0.1, 0.1, 0.9, 0.9, 0.9, 1.1], dtype=np.float64)
    transform = np.eye(4, dtype=np.float64)
    vp = {"origin": [0.0, 0.0, 0.0], "voxel_size": 1.0}

    voxels = voxelize_obb(bbox, transform, vp, np.array([3, 3, 3]))

    assert set(map(tuple, voxels)) == {(0, 0, 0), (0, 0, 1)}


def _make_single_voxel_yaw_data() -> dict:
    """构造只占一个体素的聚类测试物体。"""
    return {
        "yaw_angles": np.array([0.0]),
        "footprints": [np.array([[0, 0]], dtype=int)],
        "rel_voxels": [np.array([[0, 0, 0]], dtype=int)],
    }


def test_cluster_selects_box_at_heatmap_peak_before_support_area() -> None:
    """最优框应优先落在簇内候选底面中心的热力峰值。"""
    grid = np.zeros((16, 8, 2), dtype=np.uint8)
    surface = np.ones(grid.shape[:2], dtype=bool)
    footprints = [
        np.array([[0, 0]], dtype=int),
        np.array([[0, 0], [1, 0]], dtype=int),
        np.array([[0, 0], [1, 0], [2, 0], [3, 0]], dtype=int),
    ]
    yaw_data = {
        "yaw_angles": np.array([0.0, 0.1, 0.2]),
        "footprints": footprints,
        "rel_voxels": [
            np.column_stack([footprint, np.zeros(len(footprint), dtype=int)])
            for footprint in footprints
        ],
    }
    # 前两个候选的底面中心同为 (2, 2)；第三个支撑面积更大但热力为 1。
    candidates = np.array([[2, 2, 0], [1, 2, 1], [7, 2, 2]], dtype=int)

    reps, infos, _, _ = cluster_placements_best(
        candidates,
        grid,
        yaw_data,
        landing_z=0,
        surface_mask_2d=surface,
        vp={"origin": [0.0, 0.0, 0.0], "voxel_size": 1.0},
        eps=20.0,
    )

    assert reps.tolist() == [[1, 2, 1]]
    assert infos[0]["bottom_center_voxel"] == [2, 2, 0]


def test_cluster_prefers_centroid_distance_before_clearance() -> None:
    """热力和支撑面积相同时，距簇中心更近应优先于更大净空。"""
    grid = np.zeros((12, 6, 2), dtype=np.uint8)
    grid[4, 2, 1] = OCCUPIED
    surface = np.ones(grid.shape[:2], dtype=bool)
    candidates = np.array([[2, 2, 0], [4, 2, 0], [8, 2, 0]], dtype=int)

    reps, _, _, _ = cluster_placements_best(
        candidates,
        grid,
        _make_single_voxel_yaw_data(),
        landing_z=0,
        surface_mask_2d=surface,
        vp={"origin": [0.0, 0.0, 0.0], "voxel_size": 1.0},
        eps=20.0,
    )

    assert reps.tolist() == [[4, 2, 0]]


def test_cluster_uses_clearance_when_centroid_distance_ties() -> None:
    """热力、支撑面积和中心距离相同时，应选择净空更大的候选。"""
    grid = np.zeros((10, 6, 2), dtype=np.uint8)
    grid[2, 2, 1] = OCCUPIED
    surface = np.ones(grid.shape[:2], dtype=bool)
    candidates = np.array([[2, 2, 0], [6, 2, 0]], dtype=int)

    reps, _, _, _ = cluster_placements_best(
        candidates,
        grid,
        _make_single_voxel_yaw_data(),
        landing_z=0,
        surface_mask_2d=surface,
        vp={"origin": [0.0, 0.0, 0.0], "voxel_size": 1.0},
        eps=20.0,
    )

    assert reps.tolist() == [[6, 2, 0]]

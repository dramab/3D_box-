"""
src/annotation/free_bbox/cluster.py
-----------------------------------
DBSCAN 聚类与每簇最优 3D box 选择。

相对旧 free_bbox 的变化：
旧实现会在每个簇内选多个分散代表；这里每个簇只选一个最优候选。
最优规则为：
    1. 支撑面积最大；
    2. 与簇中心距离最近；
    3. 与碰撞障碍的最小距离最大，即碰撞危险最低。
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import distance_transform_edt
from sklearn.cluster import DBSCAN

from src.annotation.free_bbox.filters import compute_bottom_center_voxels
from src.annotation.free_bbox.occupancy import OCCUPIED
from src.annotation.free_bbox.voxel_utils import voxel_to_world


def _estimate_object_xy_scale_voxels(yaw_data: dict) -> float | None:
    """估计物体在 XY 平面的典型体素尺度。"""
    xy_sizes = []
    for rel_voxels in yaw_data.get("rel_voxels", []):
        if len(rel_voxels) == 0:
            continue
        obj_size = rel_voxels.max(axis=0) + 1
        xy_sizes.append(float(max(obj_size[0], obj_size[1])))
    if not xy_sizes:
        return None
    return float(np.median(xy_sizes))


def _estimate_dbscan_eps(
    yaw_data: dict,
    vp: dict,
    size_ratio: float = 0.5,
    min_eps_voxels: float = 3.0,
    max_eps_voxels: float = 12.0,
) -> float:
    """根据物体尺度估计 DBSCAN eps。"""
    voxel_size = float(vp["voxel_size"])
    obj_xy_scale = _estimate_object_xy_scale_voxels(yaw_data)
    if obj_xy_scale is None:
        return 5.0 * voxel_size
    return float(
        np.clip(
            size_ratio * obj_xy_scale * voxel_size,
            min_eps_voxels * voxel_size,
            max_eps_voxels * voxel_size,
        )
    )


def _support_counts_for_candidates(
    candidates: np.ndarray,
    yaw_data: dict,
    surface_mask_2d: np.ndarray,
    chunk_size: int,
) -> np.ndarray:
    """计算每个候选 footprint 落在支撑面上的体素数量。"""
    counts = np.zeros(len(candidates), dtype=np.int64)
    grid_x, grid_y = surface_mask_2d.shape
    for yaw_idx, footprint in enumerate(yaw_data["footprints"]):
        mask = candidates[:, 2] == yaw_idx
        if not np.any(mask) or len(footprint) == 0:
            continue
        indices = np.flatnonzero(mask)
        batch = candidates[indices]
        for start in range(0, len(batch), int(chunk_size)):
            end = min(start + int(chunk_size), len(batch))
            sub = batch[start:end]
            fi = sub[:, 0:1] + footprint[:, 0:1].T
            fj = sub[:, 1:2] + footprint[:, 1:2].T
            in_bounds = (fi >= 0) & (fi < grid_x) & (fj >= 0) & (fj < grid_y)
            fi_c = np.clip(fi, 0, grid_x - 1)
            fj_c = np.clip(fj, 0, grid_y - 1)
            on_surface = surface_mask_2d[fi_c, fj_c] & in_bounds
            counts[indices[start:end]] = on_surface.sum(axis=1)
    return counts


def _build_clearance_grid(
    grid_work: np.ndarray,
    landing_z: int,
    surface_mask_2d: np.ndarray,
) -> np.ndarray:
    """计算每个体素到最近非支撑面障碍的距离，单位为体素。"""
    obstacle = grid_work == OCCUPIED
    if 0 <= int(landing_z) < obstacle.shape[2]:
        obstacle = np.array(obstacle, copy=True)
        layer = obstacle[:, :, int(landing_z)]
        layer[np.asarray(surface_mask_2d, dtype=bool)] = False
        obstacle[:, :, int(landing_z)] = layer
    return distance_transform_edt(~obstacle).astype(np.float32)


def _clearance_for_candidates(
    candidates: np.ndarray,
    grid_work: np.ndarray,
    yaw_data: dict,
    landing_z: int,
    surface_mask_2d: np.ndarray,
    chunk_size: int,
) -> np.ndarray:
    """
    计算候选框与最近障碍的最小距离。

    值越大表示越不容易发生碰撞危险；越界体素按 0 处理。
    """
    clearance_grid = _build_clearance_grid(grid_work, landing_z, surface_mask_2d)
    grid_shape = np.asarray(grid_work.shape, dtype=int)
    clearance = np.zeros(len(candidates), dtype=np.float32)

    for yaw_idx, rel_voxels in enumerate(yaw_data["rel_voxels"]):
        mask = candidates[:, 2] == yaw_idx
        if not np.any(mask) or len(rel_voxels) == 0:
            continue
        indices = np.flatnonzero(mask)
        batch = candidates[indices]
        rel = np.asarray(rel_voxels, dtype=int)
        for start in range(0, len(batch), int(chunk_size)):
            end = min(start + int(chunk_size), len(batch))
            sub = batch[start:end]
            global_voxels = rel[None, :, :] + np.column_stack(
                [
                    sub[:, :2],
                    np.full(len(sub), int(landing_z), dtype=int),
                ]
            )[:, None, :]
            in_bounds = (
                (global_voxels[:, :, 0] >= 0)
                & (global_voxels[:, :, 0] < grid_shape[0])
                & (global_voxels[:, :, 1] >= 0)
                & (global_voxels[:, :, 1] < grid_shape[1])
                & (global_voxels[:, :, 2] >= 0)
                & (global_voxels[:, :, 2] < grid_shape[2])
            )
            distance_values = np.full(in_bounds.shape, np.inf, dtype=np.float32)
            flat_valid = in_bounds.ravel()
            flat_voxels = global_voxels.reshape(-1, 3)[flat_valid]
            distance_values.ravel()[flat_valid] = clearance_grid[
                flat_voxels[:, 0],
                flat_voxels[:, 1],
                flat_voxels[:, 2],
            ]
            chunk_clearance = distance_values.min(axis=1)
            chunk_clearance[~np.all(in_bounds, axis=1)] = 0.0
            clearance[indices[start:end]] = chunk_clearance
    return clearance


def build_heat_counts(
    bottom_center_voxels: np.ndarray,
    grid_shape: tuple[int, int, int],
) -> np.ndarray:
    """按底面中心体素累计候选框数量。"""
    heat_counts = np.zeros(tuple(grid_shape), dtype=np.int64)
    centers = np.asarray(bottom_center_voxels, dtype=int)
    if len(centers) == 0:
        return heat_counts
    valid = (
        (centers[:, 0] >= 0)
        & (centers[:, 0] < grid_shape[0])
        & (centers[:, 1] >= 0)
        & (centers[:, 1] < grid_shape[1])
        & (centers[:, 2] >= 0)
        & (centers[:, 2] < grid_shape[2])
    )
    centers = centers[valid]
    np.add.at(heat_counts, (centers[:, 0], centers[:, 1], centers[:, 2]), 1)
    return heat_counts


def _valid_bottom_center_mask(
    bottom_center_voxels: np.ndarray,
    surface_mask_2d: np.ndarray,
) -> np.ndarray:
    """检查底面中心是否落在支撑面体素上。"""
    centers = np.asarray(bottom_center_voxels, dtype=int)
    grid_x, grid_y = surface_mask_2d.shape
    valid = (
        (centers[:, 0] >= 0)
        & (centers[:, 0] < grid_x)
        & (centers[:, 1] >= 0)
        & (centers[:, 1] < grid_y)
    )
    keep = np.zeros(len(centers), dtype=bool)
    valid_centers = centers[valid]
    keep[valid] = surface_mask_2d[valid_centers[:, 0], valid_centers[:, 1]]
    return keep


def cluster_placements_best(
    candidates: np.ndarray,
    grid_work: np.ndarray,
    yaw_data: dict,
    landing_z: int,
    surface_mask_2d: np.ndarray,
    vp: dict,
    eps: float | None = None,
    min_samples: int = 1,
    max_reps_total: int | None = None,
    chunk_size: int = 512,
) -> tuple[np.ndarray, list[dict], list[dict], np.ndarray]:
    """
    对候选框聚类，并为每个簇选择一个最优 3D box。

    输出:
        reps: (K, 3) int，每簇一个代表候选
        infos: list[dict]，与 reps 一一对应
        cluster_records: list[dict]，包含簇成员和热力统计所需字段
        filtered_candidates: (M, 3) int，底面中心约束后的聚类输入
    """
    if len(candidates) == 0:
        return np.empty((0, 3), dtype=int), [], [], np.empty((0, 3), dtype=int)

    bottom_centers = compute_bottom_center_voxels(candidates, yaw_data, landing_z)
    center_keep = _valid_bottom_center_mask(bottom_centers, surface_mask_2d)
    candidates = candidates[center_keep]
    bottom_centers = bottom_centers[center_keep]
    if len(candidates) == 0:
        return np.empty((0, 3), dtype=int), [], [], candidates

    effective_eps = _estimate_dbscan_eps(yaw_data, vp) if eps is None else float(eps)
    if effective_eps <= 0.0:
        raise ValueError("DBSCAN eps must be positive")
    if int(min_samples) <= 0:
        raise ValueError("dbscan min_samples must be positive")
    if max_reps_total is not None and int(max_reps_total) <= 0:
        raise ValueError("max_reps_total must be positive or None")

    support_counts = _support_counts_for_candidates(
        candidates,
        yaw_data,
        surface_mask_2d,
        chunk_size=chunk_size,
    )
    clearance_voxels = _clearance_for_candidates(
        candidates,
        grid_work,
        yaw_data,
        int(landing_z),
        surface_mask_2d,
        chunk_size=chunk_size,
    )

    centers_world = voxel_to_world(bottom_centers, vp)[:, :2]
    labels = DBSCAN(eps=effective_eps, min_samples=int(min_samples)).fit_predict(centers_world)
    unique_labels = sorted(set(labels) - {-1})
    yaw_angles = yaw_data["yaw_angles"]
    voxel_size = float(vp["voxel_size"])

    cluster_records = []
    for label in unique_labels:
        member_mask = labels == label
        member_indices = np.flatnonzero(member_mask)
        members = candidates[member_indices]
        member_centers = bottom_centers[member_indices]
        member_world = centers_world[member_indices]
        centroid = member_world.mean(axis=0)
        centroid_distances = np.linalg.norm(member_world - centroid[None, :], axis=1)

        # lexsort 最后一列优先：支撑面积降序、中心距离升序、clearance 降序。
        order = np.lexsort(
            (
                -clearance_voxels[member_indices],
                centroid_distances,
                -support_counts[member_indices],
            )
        )
        best_local = int(order[0])
        best_global = int(member_indices[best_local])
        best_candidate = np.asarray(candidates[best_global], dtype=int)
        best_center = np.asarray(bottom_centers[best_global], dtype=int)
        yaw_idx = int(best_candidate[2])

        info = {
            "cluster_id": int(label),
            "size": int(len(members)),
            "anchor_voxel": [int(best_candidate[0]), int(best_candidate[1]), int(landing_z)],
            "bottom_center_voxel": best_center.tolist(),
            "bottom_center_world": voxel_to_world(best_center, vp).tolist(),
            "yaw_index": yaw_idx,
            "yaw_degrees": float(np.degrees(yaw_angles[yaw_idx])),
            "support_area_voxels": int(support_counts[best_global]),
            "support_area": float(support_counts[best_global] * voxel_size * voxel_size),
            "clearance_voxels": float(clearance_voxels[best_global]),
            "clearance": float(clearance_voxels[best_global] * voxel_size),
            "centroid_distance": float(centroid_distances[best_local]),
            "dbscan_eps": float(effective_eps),
            "dbscan_min_samples": int(min_samples),
        }
        cluster_records.append(
            {
                "size": int(len(members)),
                "rep": best_candidate,
                "info": info,
                "members": members,
                "member_bottom_centers": member_centers,
            }
        )

    cluster_records.sort(key=lambda row: row["size"], reverse=True)
    if max_reps_total is not None:
        cluster_records = cluster_records[: int(max_reps_total)]

    reps = [row["rep"] for row in cluster_records]
    infos = [row["info"] for row in cluster_records]
    reps_array = np.asarray(reps, dtype=int) if reps else np.empty((0, 3), dtype=int)
    return reps_array, infos, cluster_records, candidates

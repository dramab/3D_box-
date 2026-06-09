"""
src/annotation/free_bbox/filters.py
-----------------------------------
放置候选过滤器：可见性、稳定性、遮挡和底面中心约束。
"""

from __future__ import annotations

import numpy as np

from src.annotation.free_bbox.geometry import get_bbox_corners, project_world, transform_points
from src.annotation.free_bbox.occupancy import OCCUPIED
from src.annotation.free_bbox.voxel_utils import voxel_to_world


def is_fully_visible(
    bbox3d: np.ndarray,
    pose_cam: np.ndarray,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    img_w: int,
    img_h: int,
    depth_buffer: np.ndarray | None = None,
    depth_margin: float = 0.0,
) -> bool:
    """检查物体 OBB 是否完整落在图像中，且不被其他几何遮挡。"""
    corners_cam = transform_points(get_bbox_corners(bbox3d), pose_cam)
    z = corners_cam[:, 2]
    if np.any(z <= 0.0):
        return False

    uv = np.stack(
        [
            fx * corners_cam[:, 0] / z + cx,
            fy * corners_cam[:, 1] / z + cy,
        ],
        axis=1,
    )
    in_view = (
        np.all(uv[:, 0] >= 0.0)
        and np.all(uv[:, 0] < img_w)
        and np.all(uv[:, 1] >= 0.0)
        and np.all(uv[:, 1] < img_h)
    )
    if not in_view or depth_buffer is None:
        return bool(in_view)

    u_int = np.clip(np.round(uv[:, 0]).astype(int), 0, img_w - 1)
    v_int = np.clip(np.round(uv[:, 1]).astype(int), 0, img_h - 1)
    buf_z = depth_buffer[v_int, u_int]
    occluded = np.isfinite(buf_z) & (z > buf_z + float(depth_margin))
    return bool(not np.any(occluded))


def filter_visible_placements(
    candidates: np.ndarray,
    landing_z: int,
    bbox3d: np.ndarray,
    T_obj2world: np.ndarray,
    E_w2c: np.ndarray,
    K: np.ndarray,
    img_w: int,
    img_h: int,
    vp: dict,
    yaw_data: dict,
) -> np.ndarray:
    """保留 OBB 严格投影在图像范围内的放置候选。"""
    if len(candidates) == 0:
        return candidates

    voxel_size = float(vp["voxel_size"])
    fx, fy = float(K[0, 0]), float(K[1, 1])
    cx, cy = float(K[0, 2]), float(K[1, 2])
    R_w2c = np.asarray(E_w2c, dtype=np.float64)[:3, :3]
    corners_canonical = get_bbox_corners(bbox3d)

    keep = np.zeros(len(candidates), dtype=bool)
    for yaw_idx, _ in enumerate(yaw_data["yaw_angles"]):
        mask = candidates[:, 2] == yaw_idx
        if not np.any(mask):
            continue
        batch = candidates[mask]
        T_rot = yaw_data["T_rotated"][yaw_idx]
        vmin_rot = yaw_data["vmin_rot_abs"][yaw_idx]

        corners_cam_base = transform_points(corners_canonical, E_w2c @ T_rot)
        anchors = np.column_stack(
            [
                batch[:, :2].astype(np.float64),
                np.full(len(batch), int(landing_z), dtype=np.float64),
            ]
        )
        delta_cam = (R_w2c @ ((anchors - vmin_rot) * voxel_size).T).T
        all_cam = corners_cam_base[None, :, :] + delta_cam[:, None, :]

        z = all_cam[:, :, 2]
        z_ok = np.all(z > 0.0, axis=1)
        z_safe = np.where(z > 0.0, z, 1.0)
        u = all_cam[:, :, 0] / z_safe * fx + cx
        v = all_cam[:, :, 1] / z_safe * fy + cy
        u_ok = np.all(u >= 0.0, axis=1) & np.all(u < img_w, axis=1)
        v_ok = np.all(v >= 0.0, axis=1) & np.all(v < img_h, axis=1)
        keep[mask] = z_ok & u_ok & v_ok

    return candidates[keep]


def filter_stable_placements(
    candidates: np.ndarray,
    yaw_data: dict,
    table_mask_2d: np.ndarray,
    min_support_ratio: float = 1.0,
    chunk_size: int = 2000,
) -> np.ndarray:
    """保留 XY 投影足迹被支撑面充分支撑且质心投影在支撑区域上的候选。"""
    if len(candidates) == 0:
        return candidates

    grid_x, grid_y = table_mask_2d.shape
    keep = np.zeros(len(candidates), dtype=bool)
    for yaw_idx, _ in enumerate(yaw_data["yaw_angles"]):
        mask = candidates[:, 2] == yaw_idx
        if not np.any(mask):
            continue
        batch = candidates[mask]
        footprint = yaw_data["footprints"][yaw_idx]
        if len(footprint) == 0:
            continue

        n_foot = len(footprint)
        batch_keep = np.zeros(len(batch), dtype=bool)
        for start in range(0, len(batch), int(chunk_size)):
            end = min(start + int(chunk_size), len(batch))
            sub = batch[start:end]
            fi = sub[:, 0:1] + footprint[:, 0:1].T
            fj = sub[:, 1:2] + footprint[:, 1:2].T
            in_bounds = (fi >= 0) & (fi < grid_x) & (fj >= 0) & (fj < grid_y)
            fi_c = np.clip(fi, 0, grid_x - 1)
            fj_c = np.clip(fj, 0, grid_y - 1)
            on_table = table_mask_2d[fi_c, fj_c] & in_bounds
            ratio = on_table.sum(axis=1) / max(n_foot, 1)

            com_i = sub[:, 0] + footprint[:, 0].mean()
            com_j = sub[:, 1] + footprint[:, 1].mean()
            com_i_int = np.round(com_i).astype(int)
            com_j_int = np.round(com_j).astype(int)
            com_in = (
                (com_i_int >= 0)
                & (com_i_int < grid_x)
                & (com_j_int >= 0)
                & (com_j_int < grid_y)
            )
            com_ok = com_in & table_mask_2d[
                np.clip(com_i_int, 0, grid_x - 1),
                np.clip(com_j_int, 0, grid_y - 1),
            ]
            batch_keep[start:end] = (ratio >= float(min_support_ratio)) & com_ok

        keep[mask] = batch_keep
    return candidates[keep]


def compute_bottom_center_voxels(
    candidates: np.ndarray,
    yaw_data: dict,
    landing_z: int,
) -> np.ndarray:
    """
    计算每个候选框底面中心所在体素。

    这里的底面中心定义为放置后 OBB 体素化 XY footprint 的离散中心，
    z 固定为支撑面层 landing_z，确保输出 box 的底面中心落在支撑面体素层。
    """
    centers = np.empty((len(candidates), 3), dtype=int)
    centers[:, 2] = int(landing_z)
    for yaw_idx, footprint in enumerate(yaw_data["footprints"]):
        mask = candidates[:, 2] == yaw_idx
        if not np.any(mask):
            continue
        if len(footprint) == 0:
            centers[mask, :2] = candidates[mask, :2]
            continue
        foot_min = footprint.min(axis=0)
        foot_max = footprint.max(axis=0)
        offset = np.floor((foot_min + foot_max + 1.0) * 0.5).astype(int)
        centers[mask, :2] = candidates[mask, :2] + offset
    return centers


def filter_bottom_center_on_surface(
    candidates: np.ndarray,
    yaw_data: dict,
    landing_z: int,
    surface_mask_2d: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """保留底面中心落在支撑面体素上的候选。"""
    if len(candidates) == 0:
        return candidates, np.empty((0, 3), dtype=int)

    centers = compute_bottom_center_voxels(candidates, yaw_data, landing_z)
    grid_x, grid_y = surface_mask_2d.shape
    in_bounds = (
        (centers[:, 0] >= 0)
        & (centers[:, 0] < grid_x)
        & (centers[:, 1] >= 0)
        & (centers[:, 1] < grid_y)
    )
    keep = np.zeros(len(candidates), dtype=bool)
    valid_centers = centers[in_bounds]
    keep[in_bounds] = surface_mask_2d[valid_centers[:, 0], valid_centers[:, 1]]
    return candidates[keep], centers[keep]


def build_depth_buffer(
    grid_work: np.ndarray,
    vp: dict,
    K: np.ndarray,
    E_w2c: np.ndarray,
    img_w: int,
    img_h: int,
) -> np.ndarray:
    """从 OCCUPIED 体素构建每像素最近深度缓冲。"""
    occ_idx = np.argwhere(grid_work == OCCUPIED)
    depth_buf = np.full((int(img_h), int(img_w)), np.inf, dtype=np.float64)
    if len(occ_idx) == 0:
        return depth_buf

    occ_world = voxel_to_world(occ_idx, vp)
    uv, z_cam = project_world(occ_world, K, E_w2c)
    valid = (
        (z_cam > 0.0)
        & np.isfinite(uv[:, 0])
        & np.isfinite(uv[:, 1])
        & (uv[:, 0] >= 0.0)
        & (uv[:, 0] < img_w)
        & (uv[:, 1] >= 0.0)
        & (uv[:, 1] < img_h)
    )
    u_int = np.clip(np.round(uv[valid, 0]).astype(int), 0, img_w - 1)
    v_int = np.clip(np.round(uv[valid, 1]).astype(int), 0, img_h - 1)
    np.minimum.at(depth_buf, (v_int, u_int), z_cam[valid])
    return depth_buf


def filter_occluded_placements(
    candidates: np.ndarray,
    landing_z: int,
    bbox3d: np.ndarray,
    T_obj2world: np.ndarray,
    depth_buffer: np.ndarray,
    K: np.ndarray,
    E_w2c: np.ndarray,
    vp: dict,
    yaw_data: dict,
    img_w: int,
    img_h: int,
    occlusion_threshold: float = 0.3,
) -> np.ndarray:
    """移除被现有场景几何遮挡的放置候选。"""
    if len(candidates) == 0:
        return candidates

    voxel_size = float(vp["voxel_size"])
    fx, fy = float(K[0, 0]), float(K[1, 1])
    cx, cy = float(K[0, 2]), float(K[1, 2])
    R_w2c = np.asarray(E_w2c, dtype=np.float64)[:3, :3]
    corners_canonical = get_bbox_corners(bbox3d)
    keep = np.zeros(len(candidates), dtype=bool)

    for yaw_idx, _ in enumerate(yaw_data["yaw_angles"]):
        mask = candidates[:, 2] == yaw_idx
        if not np.any(mask):
            continue
        batch = candidates[mask]
        T_rot = yaw_data["T_rotated"][yaw_idx]
        vmin_rot = yaw_data["vmin_rot_abs"][yaw_idx]
        corners_cam_base = transform_points(corners_canonical, E_w2c @ T_rot)
        anchors = np.column_stack(
            [
                batch[:, :2].astype(np.float64),
                np.full(len(batch), int(landing_z), dtype=np.float64),
            ]
        )
        delta_cam = (R_w2c @ ((anchors - vmin_rot) * voxel_size).T).T
        all_cam = corners_cam_base[None, :, :] + delta_cam[:, None, :]

        z = all_cam[:, :, 2]
        z_safe = np.where(z > 0.0, z, 1.0)
        u = all_cam[:, :, 0] / z_safe * fx + cx
        v = all_cam[:, :, 1] / z_safe * fy + cy
        u_int = np.clip(np.round(u).astype(int), 0, img_w - 1)
        v_int = np.clip(np.round(v).astype(int), 0, img_h - 1)
        buf_z = depth_buffer[v_int, u_int]
        behind = (z > 0.0) & (z > buf_z + voxel_size)
        frac = behind.sum(axis=1) / 8.0
        keep[mask] = frac <= float(occlusion_threshold)

    return candidates[keep]

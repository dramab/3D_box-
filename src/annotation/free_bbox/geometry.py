"""
src/annotation/free_bbox/geometry.py
------------------------------------
几何工具函数。

这些函数保留原 free_bbox 中的坐标变换和放置姿态计算逻辑，避免
pipeline 依赖旧项目中的 src.utils.coord_utils。
"""

from __future__ import annotations

import numpy as np


def get_bbox_corners(bbox3d: np.ndarray) -> np.ndarray:
    """
    从 canonical AABB 生成 8 个角点。

    输入:
        bbox3d: (6,) [min_x, min_y, min_z, max_x, max_y, max_z]
    输出:
        (8, 3) float64 角点
    """
    bbox = np.asarray(bbox3d, dtype=np.float64)
    mn, mx = bbox[:3], bbox[3:]
    corners = []
    for zi in range(2):
        for yi in range(2):
            for xi in range(2):
                corners.append(
                    [
                        [mn[0], mx[0]][xi],
                        [mn[1], mx[1]][yi],
                        [mn[2], mx[2]][zi],
                    ]
                )
    return np.asarray(corners, dtype=np.float64)


def transform_points(points: np.ndarray, transform: np.ndarray) -> np.ndarray:
    """使用 4x4 齐次矩阵变换点集。"""
    pts = np.asarray(points, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] != 3:
        raise ValueError("points must have shape (N, 3)")
    ones = np.ones((len(pts), 1), dtype=np.float64)
    pts_h = np.hstack([pts, ones])
    return (np.asarray(transform, dtype=np.float64) @ pts_h.T).T[:, :3]


def project_world(
    points_world: np.ndarray,
    K: np.ndarray,
    E_w2c: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    将世界坐标 3D 点投影到图像像素坐标。

    输出:
        uv: (N, 2) 像素坐标
        z_cam: (N,) 相机坐标系深度
    """
    pts_cam = transform_points(points_world, E_w2c)
    z_cam = pts_cam[:, 2]
    z_safe = np.where(np.abs(z_cam) < 1e-8, 1e-8, z_cam)
    uv = (np.asarray(K, dtype=np.float64) @ pts_cam.T).T[:, :2]
    uv[:, 0] /= z_safe
    uv[:, 1] /= z_safe
    return uv, z_cam


def rotation_z_3x3(angle_rad: float) -> np.ndarray:
    """构造绕世界 Z 轴旋转的 3x3 矩阵。"""
    c, s = np.cos(angle_rad), np.sin(angle_rad)
    return np.array(
        [
            [c, -s, 0.0],
            [s, c, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


def rotation_matrix_to_euler_zyx(R: np.ndarray) -> tuple[float, float, float]:
    """从旋转矩阵提取 ZYX 欧拉角，返回 (roll, pitch, yaw)。"""
    R = np.asarray(R, dtype=np.float64)
    sy = np.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2)
    singular = sy < 1e-6
    if not singular:
        roll = np.arctan2(R[2, 1], R[2, 2])
        pitch = np.arctan2(-R[2, 0], sy)
        yaw = np.arctan2(R[1, 0], R[0, 0])
    else:
        roll = np.arctan2(-R[1, 2], R[1, 1])
        pitch = np.arctan2(-R[2, 0], sy)
        yaw = 0.0
    return float(roll), float(pitch), float(yaw)


def rotation_matrix_from_euler_zyx(
    roll: float,
    pitch: float,
    yaw: float,
) -> np.ndarray:
    """从 ZYX 欧拉角构造旋转矩阵。"""
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    return np.array(
        [
            [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
            [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
            [-sp, cp * sr, cp * cr],
        ],
        dtype=np.float64,
    )


def compute_placed_transform(
    bbox3d_canonical: np.ndarray,
    center_world: np.ndarray,
    yaw_rad: float,
) -> np.ndarray:
    """将 canonical AABB 中心放到 center_world，并使用 yaw-only 姿态。"""
    bbox = np.asarray(bbox3d_canonical, dtype=np.float64)
    obj_center = (bbox[:3] + bbox[3:]) * 0.5
    rotation = rotation_z_3x3(yaw_rad)
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = rotation
    transform[:3, 3] = np.asarray(center_world, dtype=np.float64) - rotation @ obj_center
    return transform


def compute_placed_transform_with_orientation(
    bbox3d_canonical: np.ndarray,
    center_world: np.ndarray,
    roll: float,
    pitch: float,
    yaw: float,
) -> np.ndarray:
    """保留 roll/pitch，只扫描 yaw 时使用的 object->world 变换。"""
    bbox = np.asarray(bbox3d_canonical, dtype=np.float64)
    obj_center = (bbox[:3] + bbox[3:]) * 0.5
    rotation = rotation_matrix_from_euler_zyx(roll, pitch, yaw)
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = rotation
    transform[:3, 3] = np.asarray(center_world, dtype=np.float64) - rotation @ obj_center
    return transform


def build_yaw_only_upright_box(
    bbox3d_canonical: np.ndarray,
    transform_world: np.ndarray,
    bottom_center_world: np.ndarray,
) -> dict:
    """将任意 OBB 转为模型监督使用的 yaw-only 竖直框。"""
    bbox = np.asarray(bbox3d_canonical, dtype=np.float64)
    dims = bbox[3:] - bbox[:3]
    transform = np.asarray(transform_world, dtype=np.float64)
    rotation_scale = transform[:3, :3]
    axis_norms = np.linalg.norm(rotation_scale, axis=0)
    if np.any(axis_norms < 1e-12):
        raise ValueError("transform_world contains a degenerate rotation axis")

    axes = rotation_scale / axis_norms[None, :]
    world_up = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    up_axis = int(np.argmax(np.abs(axes.T @ world_up)))
    horizontal_axes = [axis for axis in range(3) if axis != up_axis]
    x_axis, y_axis = horizontal_axes

    x_dir_xy = np.array([axes[0, x_axis], axes[1, x_axis]], dtype=np.float64)
    x_dir_norm = float(np.linalg.norm(x_dir_xy))
    if x_dir_norm < 1e-12:
        x_dir_xy = np.array([1.0, 0.0], dtype=np.float64)
    else:
        x_dir_xy /= x_dir_norm
    yaw = float(np.arctan2(x_dir_xy[1], x_dir_xy[0]))

    dims_yaw = np.array([dims[x_axis], dims[y_axis], dims[up_axis]], dtype=np.float64)
    bottom_center = np.asarray(bottom_center_world, dtype=np.float64)
    center_world = np.array(
        [
            bottom_center[0],
            bottom_center[1],
            bottom_center[2] + dims_yaw[2] * 0.5,
        ],
        dtype=np.float64,
    )

    local_bbox = np.concatenate([-dims_yaw * 0.5, dims_yaw * 0.5])
    yaw_transform = np.eye(4, dtype=np.float64)
    yaw_transform[:3, :3] = rotation_z_3x3(yaw)
    yaw_transform[:3, 3] = center_world
    corners_world = transform_points(get_bbox_corners(local_bbox), yaw_transform)
    aabb_world = np.concatenate([corners_world.min(axis=0), corners_world.max(axis=0)])

    return {
        "center_world": center_world,
        "yaw_degrees": float(np.degrees(yaw) % 360.0),
        "transform_world": yaw_transform,
        "corners_world": corners_world,
        "aabb_world": aabb_world,
        "dimensions": dims_yaw,
        "axis_mapping": {
            "x": int(x_axis),
            "y": int(y_axis),
            "z": int(up_axis),
        },
    }

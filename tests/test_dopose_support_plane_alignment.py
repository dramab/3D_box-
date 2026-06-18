"""
tests/test_dopose_support_plane_alignment.py
--------------------------------------------
DOPose 支撑面 RANSAC 对齐的合成数据测试。
"""

from __future__ import annotations

import numpy as np
import pytest

from src.datasets.canonical import ObjectInfo
from src.datasets.dopose_to_canonical import estimate_support_plane_alignment


def _plane_basis(normal: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """根据单位法向构造平面内两个正交基向量。"""
    normal = np.asarray(normal, dtype=np.float64)
    normal = normal / np.linalg.norm(normal)
    seed = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    if abs(float(np.dot(seed, normal))) > 0.9:
        seed = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    u = seed - np.dot(seed, normal) * normal
    u = u / np.linalg.norm(u)
    v = np.cross(normal, u)
    return u, v


def _make_plane_points(
    normal: np.ndarray,
    point_on_plane: np.ndarray,
    count: int = 3000,
    noise_cm: float = 0.05,
    seed: int = 1,
) -> np.ndarray:
    """生成带少量法向噪声的倾斜平面点云，单位 cm。"""
    rng = np.random.default_rng(seed)
    normal = np.asarray(normal, dtype=np.float64)
    normal = normal / np.linalg.norm(normal)
    u, v = _plane_basis(normal)
    coeff = rng.uniform(-40.0, 40.0, size=(count, 2))
    points = point_on_plane + coeff[:, :1] * u + coeff[:, 1:] * v
    points += rng.normal(0.0, noise_cm, size=(count, 1)) * normal
    return points


def _make_object(center_world: np.ndarray) -> ObjectInfo:
    """构造一个用于法向定向的简化物体记录。"""
    pose = np.eye(4, dtype=np.float64)
    pose[:3, 3] = np.asarray(center_world, dtype=np.float64)
    return ObjectInfo(
        obj_id="obj_0",
        class_name="synthetic",
        bbox3d_canonical=np.array([-1.0, -1.0, -1.0, 1.0, 1.0, 1.0], dtype=np.float64),
        pose_world=pose,
    )


def test_support_plane_alignment_rotates_tilted_plane_to_z_up() -> None:
    """验证倾斜支撑面拟合后法向对齐 canonical +Z，支撑面高度接近 0。"""
    true_normal = np.array([0.3, -0.4, 0.8660254], dtype=np.float64)
    true_normal = true_normal / np.linalg.norm(true_normal)
    point_on_plane = true_normal * 12.0
    plane_points = _make_plane_points(true_normal, point_on_plane, count=3500)
    rng = np.random.default_rng(7)
    outliers = rng.uniform(-60.0, 60.0, size=(700, 3))
    points = np.vstack([plane_points, outliers])
    obj = _make_object(point_on_plane + true_normal * 8.0)

    alignment, stats = estimate_support_plane_alignment(
        points,
        camera_center_world=point_on_plane + true_normal * 80.0,
        objects=[obj],
        distance_thresh_cm=0.4,
        num_iters=256,
        min_inlier_count=1000,
        min_inlier_ratio=0.2,
        max_ransac_points=5000,
        random_seed=11,
    )

    rotated_normal = alignment[:3, :3] @ true_normal
    aligned_plane = (alignment[:3, :3] @ plane_points.T).T + alignment[:3, 3]
    assert float(np.dot(rotated_normal, np.array([0.0, 0.0, 1.0]))) > 0.99
    assert abs(float(np.median(aligned_plane[:, 2]))) < 0.2
    assert stats["method"] == "support_surface_ransac_to_world_z"
    assert stats["inlier_count"] >= 1000


def test_support_plane_alignment_orients_normal_to_object_side() -> None:
    """验证法向会朝向物体所在一侧，转换后物体中心 z 为正。"""
    true_normal = np.array([-0.2, 0.5, 0.84261498], dtype=np.float64)
    true_normal = true_normal / np.linalg.norm(true_normal)
    point_on_plane = np.array([3.0, -4.0, 5.0], dtype=np.float64)
    plane_points = _make_plane_points(true_normal, point_on_plane, count=2500, seed=2)
    obj = _make_object(point_on_plane + true_normal * 10.0)

    alignment, stats = estimate_support_plane_alignment(
        plane_points,
        camera_center_world=point_on_plane + true_normal * 40.0,
        objects=[obj],
        distance_thresh_cm=0.3,
        num_iters=128,
        min_inlier_count=800,
        min_inlier_ratio=0.5,
        random_seed=3,
    )

    aligned_center = alignment @ obj.pose_world[:, 3]
    support_normal = np.asarray(stats["support_normal_before"], dtype=np.float64)
    assert float(np.dot(support_normal, true_normal)) > 0.99
    assert float(aligned_center[2]) > 0.0
    assert stats["normal_orientation_reference"] == "objects"


def test_support_plane_alignment_fails_without_enough_points() -> None:
    """验证输入点数不足时抛出清晰错误，避免静默生成错误对齐。"""
    with pytest.raises(ValueError, match="not enough points"):
        estimate_support_plane_alignment(
            np.zeros((2, 3), dtype=np.float64),
            camera_center_world=np.array([0.0, 0.0, 1.0]),
            objects=[],
        )

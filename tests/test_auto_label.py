"""auto_label 空间关系生成测试。"""

from __future__ import annotations

import numpy as np

from src.annotation.auto_label import (
    describe_angle_relation,
    describe_spatial_relation,
    generate_label_for_placement,
)
from src.annotation.free_bbox.geometry import get_bbox_corners, transform_points
from src.datasets.canonical import CameraParams, ObjectInfo


def _camera() -> CameraParams:
    """构造一个简单 pinhole 相机，world 坐标直接作为 camera 坐标。"""
    return CameraParams(
        fx=100.0,
        fy=100.0,
        cx=320.0,
        cy=240.0,
        E_c2w=np.eye(4, dtype=np.float64),
        img_w=640,
        img_h=480,
    )


def _camera_with_world_to_camera_rotation(rotation_w2c: np.ndarray) -> CameraParams:
    """构造指定 world→camera 旋转的测试相机。"""
    E_w2c = np.eye(4, dtype=np.float64)
    E_w2c[:3, :3] = rotation_w2c
    return CameraParams(
        fx=100.0,
        fy=100.0,
        cx=320.0,
        cy=240.0,
        E_c2w=np.linalg.inv(E_w2c),
        img_w=640,
        img_h=480,
    )


def _pose(center: tuple[float, float, float]) -> np.ndarray:
    pose = np.eye(4, dtype=np.float64)
    pose[:3, 3] = np.asarray(center, dtype=np.float64)
    return pose


def _object(obj_id: str, center: tuple[float, float, float]) -> ObjectInfo:
    return ObjectInfo(
        obj_id=obj_id,
        class_name=obj_id,
        bbox3d_canonical=np.array([-1.0, -1.0, -1.0, 1.0, 1.0, 1.0], dtype=np.float64),
        pose_world=_pose(center),
    )


def _obj_record(center: tuple[float, float, float]) -> dict:
    return {
        "object_id": "obj_0",
        "class_name": "obj_0",
        "canonical_aabb_object": [-1.0, -1.0, -1.0, 1.0, 1.0, 1.0],
        "original_pose_world": _pose(center).tolist(),
    }


def _placement(center: tuple[float, float, float]) -> dict:
    corners = transform_points(get_bbox_corners(np.array([-1.0, -1.0, -1.0, 1.0, 1.0, 1.0])), _pose(center))
    return {
        "cluster_id": 0,
        "sample_id": "sample_obj_0_cluster_000",
        "center_world": list(center),
        "bottom_center_world": [center[0], center[1], center[2] - 1.0],
        "corners_world": corners.tolist(),
    }


def test_angle_relation_uses_uniform_45_degree_sectors() -> None:
    """水平 8 方向应按 45° 均匀扇区划分。"""
    assert describe_angle_relation(20.0) == "the right of"
    assert describe_angle_relation(45.0) == "the front right of"
    assert describe_angle_relation(90.0) == "in front of"
    assert describe_angle_relation(-45.0) == "the back right of"


def test_spatial_relation_uses_image_aligned_world_xy_topdown() -> None:
    """水平关系应使用按图像视角绕 Z 旋转后的 world XY 俯视方向。"""
    camera = _camera_with_world_to_camera_rotation(
        np.array(
            [
                [0.0, 1.0, 0.0],
                [-1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )
    )
    target = _placement((0.0, 10.0, 100.0))
    ref = _placement((0.0, 0.0, 100.0))

    relation = describe_spatial_relation(
        np.asarray(target["corners_world"], dtype=np.float64),
        np.asarray(ref["corners_world"], dtype=np.float64),
        camera.E_w2c,
        camera.K,
    )

    assert relation == "the right of"


def test_spatial_relation_flips_image_y_for_front_direction() -> None:
    """图像上方对应 in front of，图像下方对应 behind。"""
    camera = _camera()
    ref = _placement((0.0, 0.0, 100.0))
    front_target = _placement((0.0, -10.0, 100.0))
    back_target = _placement((0.0, 10.0, 100.0))

    front_relation = describe_spatial_relation(
        np.asarray(front_target["corners_world"], dtype=np.float64),
        np.asarray(ref["corners_world"], dtype=np.float64),
        camera.E_w2c,
        camera.K,
    )
    back_relation = describe_spatial_relation(
        np.asarray(back_target["corners_world"], dtype=np.float64),
        np.asarray(ref["corners_world"], dtype=np.float64),
        camera.E_w2c,
        camera.K,
    )

    assert front_relation == "in front of"
    assert back_relation == "behind"


def test_auto_label_skips_when_no_reference_after_excluding_target() -> None:
    """排除目标自身后没有任何参照物时不生成 near，而是跳过。"""
    label, relation = generate_label_for_placement(
        obj_record=_obj_record((0.0, 0.0, 100.0)),
        placement=_placement((10.0, 0.0, 100.0)),
        reference_objects=[],
        camera=_camera(),
        mapping_data={},
    )

    assert label is None
    assert relation["skip_reason"] == "no_original_reference"


def test_auto_label_uses_unfiltered_reference_when_visibility_filter_is_empty() -> None:
    """可见投影面积筛空时，应取消阈值并选择仍存在的参照物。"""
    target = _object("obj_0", (0.0, 0.0, 100.0))
    ref = _object("obj_1", (30.0, 0.0, 100.0))

    label, relation = generate_label_for_placement(
        obj_record=_obj_record((0.0, 0.0, 100.0)),
        placement=_placement((10.0, 0.0, 100.0)),
        reference_objects=[target, ref],
        camera=_camera(),
        mapping_data={},
    )

    assert label is not None
    assert "near" not in label
    assert relation["original"]["reference_object_id"] == "obj_1"
    assert relation["original"]["reference_selection_mode"] == "unfiltered"


def test_auto_label_uses_alternative_when_original_and_placement_duplicate() -> None:
    """首选参考物和方向重复时，应换用已有备选参考物。"""
    target = _object("obj_0", (0.0, 0.0, 100.0))
    ref = _object("obj_1", (30.0, 0.0, 100.0))
    distractor = _object("obj_2", (-50.0, 0.0, 100.0))

    label, relation = generate_label_for_placement(
        obj_record=_obj_record((0.0, 0.0, 100.0)),
        placement=_placement((0.0, 0.0, 100.0)),
        reference_objects=[target, ref, distractor],
        camera=_camera(),
        mapping_data={},
    )

    assert label is not None
    assert relation["original"]["reference_object_id"] == "obj_1"
    assert not (
        relation["original"]["reference_object_id"] == relation["placement"]["reference_object_id"]
        and relation["original"]["relation"] == relation["placement"]["relation"]
    )


def test_auto_label_skips_when_all_original_and_placement_relations_duplicate() -> None:
    """没有可替代组合时，应跳过重复描述。"""
    target = _object("obj_0", (0.0, 0.0, 100.0))
    ref = _object("obj_1", (30.0, 0.0, 100.0))

    label, relation = generate_label_for_placement(
        obj_record=_obj_record((0.0, 0.0, 100.0)),
        placement=_placement((0.0, 0.0, 100.0)),
        reference_objects=[target, ref],
        camera=_camera(),
        mapping_data={},
    )

    assert label is None
    assert relation["skip_reason"] == "duplicate_original_and_placement_relation"

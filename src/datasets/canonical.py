"""
src/datasets/canonical.py
-------------------------
统一 placement scene 数据结构和 JSON 读写。

统一数据集中的每个 sample 已完成数据集特定转换，后续 pipeline 只依赖
这里定义的标准字段，不再依赖原始数据 adapter。
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


SCHEMA_VERSION = "canonical_placement_scene/v1"


@dataclass
class CameraParams:
    """与数据集无关的相机参数，所有长度单位与 sample.unit 一致。"""

    fx: float
    fy: float
    cx: float
    cy: float
    E_c2w: np.ndarray
    img_w: int
    img_h: int

    @property
    def K(self) -> np.ndarray:
        """3x3 相机内参矩阵。"""
        return np.array(
            [
                [self.fx, 0.0, self.cx],
                [0.0, self.fy, self.cy],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )

    @property
    def E_w2c(self) -> np.ndarray:
        """world→camera 外参矩阵。"""
        return np.linalg.inv(self.E_c2w)


@dataclass
class ObjectInfo:
    """单个物体的规范几何信息。"""

    obj_id: str
    class_name: str
    bbox3d_canonical: np.ndarray
    pose_world: np.ndarray


@dataclass
class CanonicalScene:
    """PlacementPipeline 的统一输入数据。"""

    sample_id: str
    scene_id: str
    frame_id: str
    rgb: np.ndarray
    depth: np.ndarray
    point_cloud_path: Path
    camera: CameraParams
    objects: list[ObjectInfo]
    unit: str = "cm"
    voxel_point_cloud_path: Path | None = None


def _resolve_sample_path(dataset_root: Path, rel_path: str) -> Path:
    """将 sample JSON 中的相对路径解析为绝对路径。"""
    return dataset_root / rel_path


def camera_to_json(camera: CameraParams) -> dict[str, Any]:
    """将 CameraParams 转成 JSON 可保存字段。"""
    return {
        "fx": float(camera.fx),
        "fy": float(camera.fy),
        "cx": float(camera.cx),
        "cy": float(camera.cy),
        "img_w": int(camera.img_w),
        "img_h": int(camera.img_h),
        "E_c2w": np.asarray(camera.E_c2w, dtype=np.float64).tolist(),
    }


def object_to_json(obj: ObjectInfo) -> dict[str, Any]:
    """将 ObjectInfo 转成 JSON 可保存字段。"""
    return {
        "obj_id": str(obj.obj_id),
        "class_name": str(obj.class_name),
        "bbox3d_canonical": np.asarray(obj.bbox3d_canonical, dtype=np.float64).tolist(),
        "pose_world": np.asarray(obj.pose_world, dtype=np.float64).tolist(),
    }


def make_sample_record(
    sample_id: str,
    scene_id: str,
    frame_id: str,
    unit: str,
    rgb_path: str,
    depth_path: str,
    point_cloud_path: str,
    camera: CameraParams,
    objects: list[ObjectInfo],
    preprocess: dict[str, Any] | None = None,
    voxel_point_cloud_path: str | None = None,
) -> dict[str, Any]:
    """构建统一 sample JSON 记录。"""
    record = {
        "schema_version": SCHEMA_VERSION,
        "sample_id": str(sample_id),
        "scene_id": str(scene_id),
        "frame_id": str(frame_id),
        "unit": str(unit),
        "rgb_path": str(rgb_path),
        "depth_path": str(depth_path),
        "point_cloud_path": str(point_cloud_path),
    }
    if voxel_point_cloud_path is not None:
        record["voxel_point_cloud_path"] = str(voxel_point_cloud_path)
    record["camera"] = camera_to_json(camera)
    record["objects"] = [object_to_json(obj) for obj in objects]
    if preprocess is not None:
        record["preprocess"] = preprocess
    return record


def save_sample_record(path: Path, record: dict[str, Any]) -> None:
    """保存单帧统一 sample JSON。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(record, f, indent=2)


def load_sample_record(path: Path) -> dict[str, Any]:
    """读取并检查单帧统一 sample JSON。"""
    with path.open("r") as f:
        record = json.load(f)
    if record.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"Unsupported schema_version in {path}: {record.get('schema_version')}")
    return record


def load_canonical_scene(sample_json_path: str | Path, dataset_root: str | Path | None = None) -> CanonicalScene:
    """
    从统一 sample JSON 加载 CanonicalScene。

    用法:
        scene = load_canonical_scene("data/canonical/samples/hope__scene_0001__0000.json")
    """
    sample_json_path = Path(sample_json_path)
    root = Path(dataset_root) if dataset_root is not None else sample_json_path.parent.parent
    record = load_sample_record(sample_json_path)

    rgb = np.asarray(Image.open(_resolve_sample_path(root, record["rgb_path"])), dtype=np.uint8)
    depth = np.load(_resolve_sample_path(root, record["depth_path"])).astype(np.float32)
    camera_record = record["camera"]
    camera = CameraParams(
        fx=float(camera_record["fx"]),
        fy=float(camera_record["fy"]),
        cx=float(camera_record["cx"]),
        cy=float(camera_record["cy"]),
        E_c2w=np.asarray(camera_record["E_c2w"], dtype=np.float64),
        img_w=int(camera_record["img_w"]),
        img_h=int(camera_record["img_h"]),
    )
    objects = [
        ObjectInfo(
            obj_id=str(obj["obj_id"]),
            class_name=str(obj["class_name"]),
            bbox3d_canonical=np.asarray(obj["bbox3d_canonical"], dtype=np.float64),
            pose_world=np.asarray(obj["pose_world"], dtype=np.float64),
        )
        for obj in record["objects"]
    ]
    return CanonicalScene(
        sample_id=str(record["sample_id"]),
        scene_id=str(record["scene_id"]),
        frame_id=str(record["frame_id"]),
        rgb=rgb,
        depth=depth,
        point_cloud_path=_resolve_sample_path(root, record["point_cloud_path"]),
        camera=camera,
        objects=objects,
        unit=str(record.get("unit", "cm")),
        voxel_point_cloud_path=(
            _resolve_sample_path(root, record["voxel_point_cloud_path"])
            if record.get("voxel_point_cloud_path") is not None
            else None
        ),
    )

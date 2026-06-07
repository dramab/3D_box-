"""
src/datasets/hope_to_canonical.py
---------------------------------
HOPE-Video 原始数据到 canonical placement scene 数据集的离线转换。

这里集中处理 HOPE 特有约定：
- depth raw uint16(mm) * 0.98042517 / 10 -> cm
- camera extrinsics 平移 m -> cm
- object pose: object->camera -> object->world
- mesh vertices mm -> cm 后计算 canonical AABB
"""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import trimesh
from PIL import Image

from src.datasets.canonical import (
    CameraParams,
    ObjectInfo,
    make_sample_record,
    save_sample_record,
)
from src.datasets.pointcloud import (
    depth_to_pointcloud,
    filter_depth_range_by_window,
    sample_pointcloud,
    save_ply,
    voxelize_pointcloud,
)


DEPTH_SCALE = 0.98042517 / 10.0
MM_TO_CM = 0.1
DATASET_NAME = "hope"
POINT_CLOUD_SAMPLE_COUNT = 50000
POINT_CLOUD_VOXEL_SIZE_CM = 1.0
VOXEL_POINT_CLOUD_DIR = "point_clouds_voxel_1cm"


class HopeCanonicalConverter:
    """将 HOPE-Video 帧转换为 canonical placement scene sample。"""

    def __init__(
        self,
        root_dir: str | Path,
        mesh_dir: str | Path,
        output_dir: str | Path,
        frame_step: int = 60,
        point_cloud_stride: int = 4,
        depth_window_cm: float | None = None,
    ) -> None:
        self.root_dir = Path(root_dir)
        self.mesh_dir = Path(mesh_dir)
        self.output_dir = Path(output_dir)
        self.frame_step = int(frame_step)
        self.point_cloud_stride = int(point_cloud_stride)
        self.depth_window_cm = depth_window_cm
        self._bbox_cache: dict[str, np.ndarray] = {}

    def list_frames(self) -> list[tuple[Path, str]]:
        """列出需要转换的 HOPE 场景帧。"""
        frames: list[tuple[Path, str]] = []
        for scene_dir in sorted(self.root_dir.glob("scene_*")):
            rgb_files = sorted(scene_dir.glob("*_rgb.jpg"))
            for rgb_path in rgb_files:
                frame_id = rgb_path.stem.replace("_rgb", "")
                if self.frame_step > 1 and int(frame_id) % self.frame_step != 0:
                    continue
                frames.append((scene_dir, frame_id))
        return frames

    def convert_all(self, max_frames: int | None = None) -> dict[str, Any]:
        """转换全部帧，并写入 manifest.json。"""
        self._prepare_output_dirs()
        frames = self.list_frames()
        if max_frames is not None:
            frames = frames[: int(max_frames)]

        sample_records = []
        for scene_dir, frame_id in frames:
            sample_records.append(self.convert_frame(scene_dir, frame_id))

        manifest = {
            "schema_version": "canonical_placement_manifest/v1",
            "dataset": DATASET_NAME,
            "sample_count": len(sample_records),
            "unit": "cm",
            "samples": sample_records,
            "preprocess": {
                "point_cloud_stride": self.point_cloud_stride,
                "point_count": POINT_CLOUD_SAMPLE_COUNT,
                "voxel_point_cloud_size_cm": POINT_CLOUD_VOXEL_SIZE_CM,
                "depth_window_cm": self.depth_window_cm,
            },
        }
        manifest_path = self.output_dir / "manifest.json"
        with manifest_path.open("w") as f:
            json.dump(manifest, f, indent=2)
        return manifest

    def convert_frame(self, scene_dir: Path, frame_id: str) -> dict[str, Any]:
        """转换单帧 HOPE 数据，并返回 manifest 中的轻量记录。"""
        scene_id = scene_dir.name
        sample_id = make_sample_id(scene_id, frame_id)
        annot_path = scene_dir / f"{frame_id}.json"
        rgb_src = scene_dir / f"{frame_id}_rgb.jpg"
        depth_src = scene_dir / f"{frame_id}_depth.png"

        with annot_path.open("r") as f:
            annots = json.load(f)

        rgb = np.asarray(Image.open(rgb_src), dtype=np.uint8)
        depth_raw = np.asarray(Image.open(depth_src), dtype=np.float32)
        depth_cm = depth_raw * DEPTH_SCALE
        depth_for_point_cloud = depth_cm
        depth_filter_stats = None
        if self.depth_window_cm is not None:
            depth_for_point_cloud, depth_filter_stats = filter_depth_range_by_window(
                depth_cm,
                self.depth_window_cm,
            )

        camera = self._load_camera(annots, rgb.shape[:2])
        objects = self._load_objects(annots, camera.E_c2w)

        rgb_rel = f"rgb/{sample_id}.jpg"
        depth_rel = f"depth/{sample_id}.npy"
        point_cloud_rel = f"point_clouds/{sample_id}.ply"
        voxel_point_cloud_rel = f"{VOXEL_POINT_CLOUD_DIR}/{sample_id}.ply"
        sample_rel = f"samples/{sample_id}.json"

        shutil.copy2(rgb_src, self.output_dir / rgb_rel)
        np.save(self.output_dir / depth_rel, depth_for_point_cloud.astype(np.float32))
        points, colors = depth_to_pointcloud(
            depth_for_point_cloud,
            rgb,
            camera.fx,
            camera.fy,
            camera.cx,
            camera.cy,
            camera.E_c2w,
            stride=self.point_cloud_stride,
        )
        point_count_before_sampling = int(len(points))
        points, colors = sample_pointcloud(
            points,
            colors,
            POINT_CLOUD_SAMPLE_COUNT,
            seed=stable_seed(sample_id),
        )
        save_ply(self.output_dir / point_cloud_rel, points, colors)
        voxel_points, voxel_colors = voxelize_pointcloud(
            points,
            colors,
            voxel_size_cm=POINT_CLOUD_VOXEL_SIZE_CM,
        )
        save_ply(self.output_dir / voxel_point_cloud_rel, voxel_points, voxel_colors)

        preprocess = {
            "depth_scale": DEPTH_SCALE,
            "point_cloud_stride": self.point_cloud_stride,
            "point_count_before_sampling": point_count_before_sampling,
            "point_cloud_sampling_replace": (
                point_count_before_sampling < POINT_CLOUD_SAMPLE_COUNT
            ),
            "voxel_point_cloud_size_cm": POINT_CLOUD_VOXEL_SIZE_CM,
            "voxel_point_count": int(len(voxel_points)),
        }
        if depth_filter_stats is not None:
            preprocess["depth_filter"] = depth_filter_stats

        sample = make_sample_record(
            sample_id=sample_id,
            scene_id=scene_id,
            frame_id=frame_id,
            unit="cm",
            rgb_path=rgb_rel,
            depth_path=depth_rel,
            point_cloud_path=point_cloud_rel,
            voxel_point_cloud_path=voxel_point_cloud_rel,
            camera=camera,
            objects=objects,
            preprocess=preprocess,
        )
        save_sample_record(self.output_dir / sample_rel, sample)

        return {
            "sample_id": sample_id,
            "scene_id": scene_id,
            "frame_id": frame_id,
            "sample_path": sample_rel,
            "object_count": len(objects),
        }

    def _prepare_output_dirs(self) -> None:
        """创建 canonical 数据集固定目录。"""
        for name in ("samples", "rgb", "depth", "point_clouds", VOXEL_POINT_CLOUD_DIR):
            (self.output_dir / name).mkdir(parents=True, exist_ok=True)

    def _load_camera(self, annots: dict[str, Any], image_hw: tuple[int, int]) -> CameraParams:
        """读取 HOPE 相机并统一到 cm 单位的 camera->world 外参。"""
        K = np.asarray(annots["camera"]["intrinsics"], dtype=np.float64)
        E_w2c = np.asarray(annots["camera"]["extrinsics"], dtype=np.float64)
        E_w2c = np.array(E_w2c, copy=True)
        E_w2c[:3, 3] *= 100.0
        E_c2w = np.linalg.inv(E_w2c)
        img_h, img_w = image_hw
        return CameraParams(
            fx=float(K[0, 0]),
            fy=float(K[1, 1]),
            cx=float(K[0, 2]),
            cy=float(K[1, 2]),
            E_c2w=E_c2w,
            img_w=int(img_w),
            img_h=int(img_h),
        )

    def _load_objects(self, annots: dict[str, Any], E_c2w: np.ndarray) -> list[ObjectInfo]:
        """读取 HOPE 物体并转换到统一 ObjectInfo。"""
        objects = []
        for index, obj in enumerate(annots.get("objects", [])):
            class_name = str(obj["class"])
            pose_cam = np.asarray(obj["pose"], dtype=np.float64)
            pose_world = E_c2w @ pose_cam
            if obj.get("bbox3d") is not None:
                bbox3d_canonical = np.asarray(obj["bbox3d"], dtype=np.float64)
            else:
                bbox3d_canonical = self.get_object_scale(class_name)
            objects.append(
                ObjectInfo(
                    obj_id=f"obj_{index}",
                    class_name=class_name,
                    bbox3d_canonical=bbox3d_canonical,
                    pose_world=pose_world,
                )
            )
        return objects

    def get_object_scale(self, class_name: str) -> np.ndarray:
        """从 HOPE mesh 计算 canonical AABB，单位 cm。"""
        if class_name in self._bbox_cache:
            return self._bbox_cache[class_name]

        mesh_path = self.mesh_dir / f"{class_name}.obj"
        if not mesh_path.exists():
            raise FileNotFoundError(f"Mesh not found: {mesh_path}")

        mesh = trimesh.load(str(mesh_path), force="mesh", process=False)
        verts = np.asarray(mesh.vertices, dtype=np.float64) * MM_TO_CM
        bbox = np.concatenate([verts.min(axis=0), verts.max(axis=0)])
        self._bbox_cache[class_name] = bbox
        return bbox


def make_sample_id(scene_id: str, frame_id: str) -> str:
    """生成稳定且可读的 HOPE sample_id。"""
    return f"{DATASET_NAME}__{scene_id}__{frame_id}"


def stable_seed(value: str) -> int:
    """从字符串生成跨 Python 进程稳定的 32-bit 随机种子。"""
    digest = hashlib.blake2s(value.encode("utf-8"), digest_size=4).digest()
    return int.from_bytes(digest, byteorder="little")

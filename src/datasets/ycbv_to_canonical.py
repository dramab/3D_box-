"""
src/datasets/ycbv_to_canonical.py
---------------------------------
YCB-Video BOP 风格 test 数据到 canonical placement scene 数据集的离线转换。

这里集中处理 YCBV 特有约定：
- depth raw uint16 * scene_camera.depth_scale -> mm，再转换为 cm
- scene_camera.cam_R_w2c/cam_t_w2c -> camera->world 外参
- object pose: object->camera -> object->world
- models_info.json 中 canonical AABB mm -> cm
"""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np
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


DATASET_NAME = "ycbv"
MM_TO_CM = 0.1
POINT_CLOUD_SAMPLE_COUNT = 50000
POINT_CLOUD_VOXEL_SIZE_CM = 1.0
VOXEL_POINT_CLOUD_DIR = "point_clouds_voxel_1cm"

YCBV_CLASS_NAMES = {
    1: "002_master_chef_can",
    2: "003_cracker_box",
    3: "004_sugar_box",
    4: "005_tomato_soup_can",
    5: "006_mustard_bottle",
    6: "007_tuna_fish_can",
    7: "008_pudding_box",
    8: "009_gelatin_box",
    9: "010_potted_meat_can",
    10: "011_banana",
    11: "019_pitcher_base",
    12: "021_bleach_cleanser",
    13: "024_bowl",
    14: "025_mug",
    15: "035_power_drill",
    16: "036_wood_block",
    17: "037_scissors",
    18: "040_large_marker",
    19: "051_large_clamp",
    20: "052_extra_large_clamp",
    21: "061_foam_brick",
}


class YcbvCanonicalConverter:
    """将 YCBV test 帧转换为 canonical placement scene sample。"""

    def __init__(
        self,
        root_dir: str | Path,
        model_dir: str | Path,
        output_dir: str | Path,
        frame_step: int = 1,
        point_cloud_stride: int = 4,
        depth_window_cm: float | None = None,
    ) -> None:
        self.root_dir = Path(root_dir)
        self.model_dir = Path(model_dir)
        self.output_dir = Path(output_dir)
        self.frame_step = int(frame_step)
        self.point_cloud_stride = int(point_cloud_stride)
        self.depth_window_cm = depth_window_cm
        if self.frame_step <= 0:
            raise ValueError(f"frame_step must be positive, got {self.frame_step}")
        self.model_info = self._load_model_info()
        self._bbox_cache: dict[int, np.ndarray] = {}

    def list_frames(self) -> list[tuple[Path, str]]:
        """列出需要转换的 YCBV scene/frame。"""
        frames: list[tuple[Path, str]] = []
        for scene_dir in sorted(path for path in self.root_dir.iterdir() if path.is_dir()):
            with (scene_dir / "scene_camera.json").open("r") as f:
                frame_keys = sorted(json.load(f), key=lambda value: int(value))
            if self.frame_step > 1:
                frame_keys = [
                    frame_key
                    for frame_key in frame_keys
                    if int(frame_key) % self.frame_step == 0
                ]
            frames.extend((scene_dir, frame_key) for frame_key in frame_keys)
        return frames

    def convert_all(self, max_frames: int | None = None) -> dict[str, Any]:
        """转换全部帧，并写入 manifest.json。"""
        self._prepare_output_dirs()
        frames = self.list_frames()
        if max_frames is not None:
            frames = frames[: int(max_frames)]

        sample_records = [self.convert_frame(scene_dir, frame_key) for scene_dir, frame_key in frames]
        manifest = {
            "schema_version": "canonical_placement_manifest/v1",
            "dataset": DATASET_NAME,
            "sample_count": len(sample_records),
            "unit": "cm",
            "samples": sample_records,
            "preprocess": {
                "frame_step": self.frame_step,
                "point_cloud_stride": self.point_cloud_stride,
                "point_count": POINT_CLOUD_SAMPLE_COUNT,
                "voxel_point_cloud_size_cm": POINT_CLOUD_VOXEL_SIZE_CM,
                "depth_window_cm": self.depth_window_cm,
                "depth_scale_source": "scene_camera.depth_scale",
                "camera_to_world": "inverse(scene_camera.cam_R_w2c, scene_camera.cam_t_w2c)",
            },
        }
        with (self.output_dir / "manifest.json").open("w") as f:
            json.dump(manifest, f, indent=2)
        return manifest

    def convert_frame(self, scene_dir: Path, frame_key: str) -> dict[str, Any]:
        """转换单帧 YCBV 数据，并返回 manifest 中的轻量记录。"""
        scene_id = make_scene_id(scene_dir.name)
        frame_id = f"{int(frame_key):06d}"
        sample_id = make_sample_id(scene_id, frame_id)

        scene_camera = self._load_scene_json(scene_dir, "scene_camera.json")
        scene_gt = self._load_scene_json(scene_dir, "scene_gt.json")
        rgb_src = scene_dir / "rgb" / f"{frame_id}.png"
        depth_src = scene_dir / "depth" / f"{frame_id}.png"

        rgb = np.asarray(Image.open(rgb_src).convert("RGB"), dtype=np.uint8)
        depth_raw = np.asarray(Image.open(depth_src), dtype=np.float32)
        camera_record = scene_camera[frame_key]
        depth_scale = float(camera_record.get("depth_scale", 1.0))
        depth_cm = depth_raw * depth_scale * MM_TO_CM
        depth_for_point_cloud = depth_cm
        depth_filter_stats = None
        if self.depth_window_cm is not None:
            depth_for_point_cloud, depth_filter_stats = filter_depth_range_by_window(
                depth_cm,
                self.depth_window_cm,
            )
        if rgb.shape[:2] != depth_for_point_cloud.shape:
            raise ValueError(f"RGB/depth shape mismatch for {sample_id}")

        camera = self._load_camera(camera_record, rgb.shape[:2])
        objects = self._load_objects(scene_gt[frame_key], camera.E_c2w)

        rgb_rel = f"rgb/{sample_id}.png"
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
            "depth_scale": depth_scale,
            "depth_unit_before_conversion": "mm",
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

    def _load_scene_json(self, scene_dir: Path, name: str) -> dict[str, Any]:
        """读取单个 scene 下的 BOP JSON 文件。"""
        with (scene_dir / name).open("r") as f:
            return json.load(f)

    def _load_camera(self, camera_record: dict[str, Any], image_hw: tuple[int, int]) -> CameraParams:
        """读取 YCBV 相机并统一到 cm 单位的 camera->world 外参。"""
        K = np.asarray(camera_record["cam_K"], dtype=np.float64).reshape(3, 3)
        img_h, img_w = image_hw
        return CameraParams(
            fx=float(K[0, 0]),
            fy=float(K[1, 1]),
            cx=float(K[0, 2]),
            cy=float(K[1, 2]),
            E_c2w=self._make_camera_to_world(camera_record),
            img_w=int(img_w),
            img_h=int(img_h),
        )

    def _make_camera_to_world(self, camera_record: dict[str, Any]) -> np.ndarray:
        """从 world->camera 外参求 camera->world，平移统一为 cm。"""
        if "cam_R_w2c" not in camera_record or "cam_t_w2c" not in camera_record:
            return np.eye(4, dtype=np.float64)

        R_w2c = np.asarray(camera_record["cam_R_w2c"], dtype=np.float64).reshape(3, 3)
        t_w2c = np.asarray(camera_record["cam_t_w2c"], dtype=np.float64) * MM_TO_CM
        E_w2c = np.eye(4, dtype=np.float64)
        E_w2c[:3, :3] = R_w2c
        E_w2c[:3, 3] = t_w2c
        return np.linalg.inv(E_w2c)

    def _load_objects(self, gt_records: list[dict[str, Any]], E_c2w: np.ndarray) -> list[ObjectInfo]:
        """读取 BOP object->camera 位姿并转换到统一 ObjectInfo。"""
        objects = []
        for index, obj in enumerate(gt_records):
            obj_id = int(obj["obj_id"])
            pose_cam = np.eye(4, dtype=np.float64)
            pose_cam[:3, :3] = np.asarray(obj["cam_R_m2c"], dtype=np.float64).reshape(3, 3)
            pose_cam[:3, 3] = np.asarray(obj["cam_t_m2c"], dtype=np.float64) * MM_TO_CM
            objects.append(
                ObjectInfo(
                    obj_id=f"obj_{index}",
                    class_name=get_class_name(obj_id),
                    bbox3d_canonical=self.get_object_bbox(obj_id),
                    pose_world=E_c2w @ pose_cam,
                )
            )
        return objects

    def _load_model_info(self) -> dict[str, Any]:
        """读取 YCBV 模型几何信息。"""
        with (self.model_dir / "models_info.json").open("r") as f:
            return json.load(f)

    def get_object_bbox(self, obj_id: int) -> np.ndarray:
        """从 models_info.json 计算 canonical AABB，单位 cm。"""
        if obj_id in self._bbox_cache:
            return self._bbox_cache[obj_id]

        info = self.model_info[str(obj_id)]
        bbox_mm = np.array(
            [
                info["min_x"],
                info["min_y"],
                info["min_z"],
                info["min_x"] + info["size_x"],
                info["min_y"] + info["size_y"],
                info["min_z"] + info["size_z"],
            ],
            dtype=np.float64,
        )
        bbox_cm = bbox_mm * MM_TO_CM
        self._bbox_cache[obj_id] = bbox_cm
        return bbox_cm


def get_class_name(obj_id: int) -> str:
    """读取 YCBV 标准类别名，缺失时退回 BOP obj id。"""
    return YCBV_CLASS_NAMES.get(int(obj_id), f"obj_{int(obj_id):06d}")


def make_scene_id(scene_name: str) -> str:
    """生成稳定且可读的 scene_id。"""
    return f"scene_{scene_name}"


def make_sample_id(scene_id: str, frame_id: str) -> str:
    """生成稳定且可读的 YCBV sample_id。"""
    return f"{DATASET_NAME}__{scene_id}__{frame_id}"


def stable_seed(value: str) -> int:
    """从字符串生成跨 Python 进程稳定的 32-bit 随机种子。"""
    digest = hashlib.blake2s(value.encode("utf-8"), digest_size=4).digest()
    return int.from_bytes(digest, byteorder="little")

"""
src/datasets/housecat_to_canonical.py
--------------------------------------
HouseCat6D 数据到 canonical placement scene 数据集的离线转换。

特有约定：
- 深度：uint16 PNG，单位 mm，乘以 0.1 转为 cm
- 相机内参：intrinsics.txt（3×3 矩阵），每场景共享
- 相机位姿：camera_pose/{frame}.txt（4×4 c2w 矩阵），平移单位 m → cm
- 物体位姿：labels/{frame}_label.pkl 中 rotations/translations（m → cm），gt_scales（m → cm）
- 坐标对齐：从深度点云 RANSAC 拟合支撑面，将法向对齐到 canonical world-Z
"""

from __future__ import annotations

import hashlib
import pickle
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
import json

from src.datasets.canonical import (
    CameraParams,
    ObjectInfo,
    make_sample_record,
    save_sample_record,
)
from src.datasets.dopose_to_canonical import (
    align_camera,
    align_objects,
    estimate_support_plane_alignment,
    stable_seed,
)
from src.datasets.pointcloud import (
    depth_to_pointcloud,
    filter_depth_range_by_window,
    sample_pointcloud,
    save_ply,
    voxelize_pointcloud,
)


DATASET_NAME = "housecat"
MM_TO_CM = 0.1
M_TO_CM = 100.0
POINT_CLOUD_SAMPLE_COUNT = 50000
POINT_CLOUD_VOXEL_SIZE_CM = 1.0
VOXEL_POINT_CLOUD_DIR = "point_clouds_voxel_1cm"
SUPPORT_PLANE_DISTANCE_THRESH_CM = 1.0
SUPPORT_PLANE_RANSAC_ITERS = 512
SUPPORT_PLANE_MIN_INLIERS = 500
SUPPORT_PLANE_MIN_INLIER_RATIO = 0.03
SUPPORT_PLANE_MAX_RANSAC_POINTS = 50000


class HouseCatCanonicalConverter:
    """将 HouseCat6D 帧转换为 canonical placement scene sample。"""

    def __init__(
        self,
        root_dir: str | Path,
        output_dir: str | Path,
        frame_step: int = 30,
        point_cloud_stride: int = 4,
        depth_window_cm: float | None = None,
        support_plane_distance_thresh_cm: float = SUPPORT_PLANE_DISTANCE_THRESH_CM,
        support_plane_ransac_iters: int = SUPPORT_PLANE_RANSAC_ITERS,
        support_plane_min_inliers: int = SUPPORT_PLANE_MIN_INLIERS,
        support_plane_min_inlier_ratio: float = SUPPORT_PLANE_MIN_INLIER_RATIO,
        support_plane_max_points: int = SUPPORT_PLANE_MAX_RANSAC_POINTS,
    ) -> None:
        self.root_dir = Path(root_dir)
        self.output_dir = Path(output_dir)
        self.frame_step = int(frame_step)
        self.point_cloud_stride = int(point_cloud_stride)
        self.depth_window_cm = depth_window_cm
        self.support_plane_distance_thresh_cm = float(support_plane_distance_thresh_cm)
        self.support_plane_ransac_iters = int(support_plane_ransac_iters)
        self.support_plane_min_inliers = int(support_plane_min_inliers)
        self.support_plane_min_inlier_ratio = float(support_plane_min_inlier_ratio)
        self.support_plane_max_points = int(support_plane_max_points)

    def list_frames(self) -> list[tuple[str, Path, str]]:
        """列出所有符合 frame_step 采样的 (scene_id, scene_dir, frame_id) 三元组。"""
        frames: list[tuple[str, Path, str]] = []
        for scene_dir in sorted(p for p in self.root_dir.iterdir() if p.is_dir()):
            scene_id = scene_dir.name  # e.g. "scene01"
            rgb_dir = scene_dir / "rgb"
            if not rgb_dir.exists():
                continue
            all_frame_ids = sorted(
                p.stem for p in rgb_dir.glob("*.png")
            )
            for frame_id in all_frame_ids:
                if int(frame_id) % self.frame_step != 0:
                    continue
                if not (scene_dir / "depth" / f"{frame_id}.png").exists():
                    continue
                if not (scene_dir / "camera_pose" / f"{frame_id}.txt").exists():
                    continue
                if not (scene_dir / "labels" / f"{frame_id}_label.pkl").exists():
                    continue
                frames.append((scene_id, scene_dir, frame_id))
        return frames

    def convert_all(self, max_frames: int | None = None) -> dict[str, Any]:
        """转换全部帧并写入 manifest.json。"""
        self._prepare_output_dirs()
        frames = self.list_frames()
        if max_frames is not None:
            frames = frames[: int(max_frames)]

        sample_records = []
        skip_count = 0
        for scene_id, scene_dir, frame_id in frames:
            try:
                record = self.convert_frame(scene_id, scene_dir, frame_id)
                sample_records.append(record)
                print(f"[OK] {scene_id}/{frame_id} ({len(sample_records)}/{len(frames)})")
            except ValueError as exc:
                print(f"[SKIP] {scene_id}/{frame_id}: {exc}")
                skip_count += 1

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
                "depth_source_unit": "mm",
                "coordinate_normalization_method": "support_surface_ransac_to_world_z",
                "skip_count": skip_count,
                "support_plane_ransac": {
                    "distance_thresh_cm": self.support_plane_distance_thresh_cm,
                    "num_iters": self.support_plane_ransac_iters,
                    "min_inlier_count": self.support_plane_min_inliers,
                    "min_inlier_ratio": self.support_plane_min_inlier_ratio,
                    "max_points": self.support_plane_max_points,
                },
            },
        }
        with (self.output_dir / "manifest.json").open("w") as f:
            json.dump(manifest, f, indent=2)
        return manifest

    def convert_frame(self, scene_id: str, scene_dir: Path, frame_id: str) -> dict[str, Any]:
        """转换单帧，返回 manifest 轻量记录。"""
        sample_id = f"{DATASET_NAME}__{scene_id}__{frame_id}"

        # --- 读取 RGB ---
        rgb = np.asarray(
            Image.open(scene_dir / "rgb" / f"{frame_id}.png").convert("RGB"),
            dtype=np.uint8,
        )

        # --- 读取深度：uint16 mm → float32 cm ---
        depth_raw = np.asarray(
            Image.open(scene_dir / "depth" / f"{frame_id}.png"), dtype=np.float32
        )
        depth_cm = depth_raw * MM_TO_CM

        # --- 相机内参（每场景共享）---
        K = np.loadtxt(scene_dir / "intrinsics.txt")  # (3, 3)
        img_h, img_w = rgb.shape[:2]

        # --- 相机位姿（c2w，平移 m → cm）---
        cam_mat = np.loadtxt(scene_dir / "camera_pose" / f"{frame_id}.txt")
        cam_mat[:3, 3] *= M_TO_CM

        raw_camera = CameraParams(
            fx=float(K[0, 0]),
            fy=float(K[1, 1]),
            cx=float(K[0, 2]),
            cy=float(K[1, 2]),
            E_c2w=cam_mat,
            img_w=img_w,
            img_h=img_h,
        )

        # --- 读取标签 ---
        with open(scene_dir / "labels" / f"{frame_id}_label.pkl", "rb") as f:
            label = pickle.load(f)
        raw_objects = _parse_objects(label, cam_mat)

        # --- 深度过滤（可选）---
        depth_filter_stats = None
        depth_for_cloud = depth_cm
        if self.depth_window_cm is not None:
            depth_for_cloud, depth_filter_stats = filter_depth_range_by_window(
                depth_cm, self.depth_window_cm
            )

        # --- RANSAC 支撑面对齐 ---
        alignment_points, _ = depth_to_pointcloud(
            depth_for_cloud,
            rgb,
            raw_camera.fx,
            raw_camera.fy,
            raw_camera.cx,
            raw_camera.cy,
            raw_camera.E_c2w,
            stride=self.point_cloud_stride,
        )
        camera_center = raw_camera.E_c2w[:3, 3]
        alignment, alignment_stats = estimate_support_plane_alignment(
            alignment_points,
            camera_center,
            raw_objects,
            distance_thresh_cm=self.support_plane_distance_thresh_cm,
            num_iters=self.support_plane_ransac_iters,
            min_inlier_count=self.support_plane_min_inliers,
            min_inlier_ratio=self.support_plane_min_inlier_ratio,
            max_ransac_points=self.support_plane_max_points,
            random_seed=stable_seed(sample_id),
        )
        camera = align_camera(raw_camera, alignment)
        objects = align_objects(raw_objects, alignment)

        # --- 输出路径 ---
        rgb_rel = f"rgb/{sample_id}.jpg"
        depth_rel = f"depth/{sample_id}.npy"
        point_cloud_rel = f"point_clouds/{sample_id}.ply"
        voxel_point_cloud_rel = f"{VOXEL_POINT_CLOUD_DIR}/{sample_id}.ply"
        sample_rel = f"samples/{sample_id}.json"

        # --- 保存文件 ---
        Image.fromarray(rgb).save(self.output_dir / rgb_rel)
        np.save(self.output_dir / depth_rel, depth_for_cloud.astype(np.float32))

        points, colors = depth_to_pointcloud(
            depth_for_cloud,
            rgb,
            camera.fx,
            camera.fy,
            camera.cx,
            camera.cy,
            camera.E_c2w,
            stride=self.point_cloud_stride,
        )
        point_count_before = int(len(points))
        points, colors = sample_pointcloud(
            points, colors, POINT_CLOUD_SAMPLE_COUNT, seed=stable_seed(sample_id)
        )
        save_ply(self.output_dir / point_cloud_rel, points, colors)
        voxel_points, voxel_colors = voxelize_pointcloud(
            points, colors, voxel_size_cm=POINT_CLOUD_VOXEL_SIZE_CM
        )
        save_ply(self.output_dir / voxel_point_cloud_rel, voxel_points, voxel_colors)

        preprocess: dict[str, Any] = {
            "depth_source_unit": "mm",
            "point_cloud_stride": self.point_cloud_stride,
            "point_count_before_sampling": point_count_before,
            "point_cloud_sampling_replace": point_count_before < POINT_CLOUD_SAMPLE_COUNT,
            "voxel_point_cloud_size_cm": POINT_CLOUD_VOXEL_SIZE_CM,
            "voxel_point_count": int(len(voxel_points)),
            "coordinate_normalization": alignment_stats,
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
        for name in ("samples", "rgb", "depth", "point_clouds", VOXEL_POINT_CLOUD_DIR):
            (self.output_dir / name).mkdir(parents=True, exist_ok=True)


def _parse_objects(label: dict, E_c2w: np.ndarray) -> list[ObjectInfo]:
    """从帧标签构造 ObjectInfo 列表，pose_world 单位 cm。"""
    objects = []
    for i, class_name in enumerate(label["model_list"]):
        t_cm = label["translations"][i].astype(np.float64) * M_TO_CM
        R = label["rotations"][i].astype(np.float64)

        # 过滤无效物体（平移为 0 或 NaN）
        if np.any(np.isnan(t_cm)) or (np.allclose(t_cm, 0.0) and np.allclose(R, np.eye(3))):
            continue

        pose_cam = np.eye(4, dtype=np.float64)
        pose_cam[:3, :3] = R
        pose_cam[:3, 3] = t_cm
        pose_world = E_c2w @ pose_cam

        # centered canonical AABB from gt_scales (m → cm)
        side_cm = label["gt_scales"][i].astype(np.float64) * M_TO_CM
        half = side_cm * 0.5
        bbox = np.concatenate([-half, half])

        objects.append(
            ObjectInfo(
                obj_id=f"obj_{i + 1}_{class_name.replace('-', '_')}",
                class_name=class_name,
                bbox3d_canonical=bbox,
                pose_world=pose_world,
            )
        )
    return objects

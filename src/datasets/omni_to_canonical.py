"""
src/datasets/omni_to_canonical.py
----------------------------------
Omni6DPose ROPE 数据到 canonical placement scene 数据集的离线转换。

特有约定：
- 深度：EXR 格式，单位 m，乘以 100 转为 cm（需设 OPENCV_IO_ENABLE_OPENEXR=1）
- 相机内参：meta 记录的是原始分辨率；若实际图像尺寸不同则等比缩放
- ROPE 模式：camera 在世界原点（identity E_c2w），object->camera 即 object->world
- 坐标对齐：从深度点云 RANSAC 拟合支撑面，将法向对齐到 canonical world-Z
- 物体尺寸：meta 中 bbox_side_len（单位 m）构造 centered AABB（cm）
- pose 质量检测：ROPE pose 标注与图像存在逐帧刚性平移误差，用 mask.exr + 深度检测
  box 中心相机系 X/Y 与物体可见表面质心的偏差，任一物体偏差超阈值则跳过该帧（不修改 pose）
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from src.datasets.canonical import (
    CameraParams,
    CanonicalScene,
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
from src.visualization.bbox_projection import save_scene_bbox_projection

# 设置 cv2 读取 EXR 的环境变量，必须在 import cv2 之前
os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")
import cv2  # noqa: E402


DATASET_NAME = "omni"
M_TO_CM = 100.0
POINT_CLOUD_SAMPLE_COUNT = 50000
POINT_CLOUD_VOXEL_SIZE_CM = 1.0
VOXEL_POINT_CLOUD_DIR = "point_clouds_voxel_1cm"
SUPPORT_PLANE_DISTANCE_THRESH_CM = 1.0
SUPPORT_PLANE_RANSAC_ITERS = 512
SUPPORT_PLANE_MIN_INLIERS = 500
SUPPORT_PLANE_MIN_INLIER_RATIO = 0.03
SUPPORT_PLANE_MAX_RANSAC_POINTS = 50000
SUPPORT_PLANE_OBJECT_BOTTOM_TOLERANCE_CM = 8.0
# ROPE pose 标注相对图像存在逐帧刚性平移误差，用 mask.exr + 深度检测对齐质量，
# 任一物体横向偏差超阈值的帧直接跳过
MASK_CHECK_DEPTH_TOL_CM = 8.0
MASK_CHECK_MIN_POINTS = 30
MASK_CHECK_MAX_LATERAL_OFFSET_CM = 3.0


class OmniCanonicalConverter:
    """将 Omni6DPose ROPE 帧转换为 canonical placement scene sample。"""

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

    def list_frames(self) -> list[tuple[str, str]]:
        """列出 ROPE 下所有符合 frame_step 采样的 (scene_id, frame_id) 对。"""
        rope_dir = self.root_dir / "ROPE"
        frames: list[tuple[str, str]] = []
        for scene_dir in sorted(path for path in rope_dir.iterdir() if path.is_dir()):
            scene_id = f"ROPE_{scene_dir.name}"
            # 收集全部 meta.json 对应的帧号
            all_frame_ids = sorted(
                path.name.replace("_meta.json", "")
                for path in scene_dir.glob("*_meta.json")
            )
            for frame_id in all_frame_ids:
                if int(frame_id) % self.frame_step != 0:
                    continue
                # 确保 color + depth 都存在
                if not (scene_dir / f"{frame_id}_color.png").exists():
                    continue
                if not (scene_dir / f"{frame_id}_depth.exr").exists():
                    continue
                frames.append((scene_id, scene_dir, frame_id))
        return frames

    def convert_all(self, max_frames: int | None = None) -> dict[str, Any]:
        """转换全部帧，并写入 manifest.json。"""
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
                "depth_source_unit": "m",
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
        """转换单帧 Omni6DPose ROPE 数据，返回 manifest 中的轻量记录。"""
        sample_id = f"{DATASET_NAME}__{scene_id}__{frame_id}"

        rgb_src = scene_dir / f"{frame_id}_color.png"
        depth_src = scene_dir / f"{frame_id}_depth.exr"
        meta_src = scene_dir / f"{frame_id}_meta.json"

        rgb = np.asarray(Image.open(rgb_src).convert("RGB"), dtype=np.uint8)
        depth_m = cv2.imread(str(depth_src), cv2.IMREAD_UNCHANGED)
        if depth_m is None:
            raise ValueError(f"Failed to read depth EXR: {depth_src}")
        if depth_m.ndim == 3:
            depth_m = depth_m[..., 0]
        depth_cm = np.asarray(depth_m, dtype=np.float32) * M_TO_CM

        with meta_src.open("r") as f:
            meta = json.load(f)

        # 深度过滤
        depth_filter_stats = None
        depth_for_cloud = depth_cm
        if self.depth_window_cm is not None:
            depth_for_cloud, depth_filter_stats = filter_depth_range_by_window(
                depth_cm, self.depth_window_cm
            )

        img_h, img_w = rgb.shape[:2]
        intrinsics = _scale_intrinsics(meta["camera"]["intrinsics"], img_w, img_h)

        # ROPE: camera 在世界原点，identity E_c2w
        raw_camera = CameraParams(
            fx=intrinsics["fx"],
            fy=intrinsics["fy"],
            cx=intrinsics["cx"],
            cy=intrinsics["cy"],
            E_c2w=np.eye(4, dtype=np.float64),
            img_w=img_w,
            img_h=img_h,
        )
        raw_objects, mask_ids = _parse_objects(meta, np.eye(4, dtype=np.float64))

        # 读取 mask.exr，检测 ROPE pose 与物体可见表面的横向对齐质量，偏差过大则跳过该帧
        mask_src = scene_dir / f"{frame_id}_mask.exr"
        mask = cv2.imread(str(mask_src), cv2.IMREAD_UNCHANGED)
        if mask is not None and mask.ndim == 3:
            mask = mask[..., 0]
        pose_ok, pose_check_stats = check_pose_alignment_with_mask(
            raw_objects, mask_ids, mask, depth_cm, raw_camera
        )
        if not pose_ok:
            raise ValueError(
                f"pose misaligned ({pose_check_stats.get('reason')}, "
                f"max_offset={pose_check_stats.get('max_object_offset_cm')}cm)"
            )

        # 用于支撑面估计的点云（camera 坐标即 world 坐标）
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

        # 输出路径
        rgb_rel = f"rgb/{sample_id}.jpg"
        depth_rel = f"depth/{sample_id}.npy"
        point_cloud_rel = f"point_clouds/{sample_id}.ply"
        voxel_point_cloud_rel = f"{VOXEL_POINT_CLOUD_DIR}/{sample_id}.ply"
        sample_rel = f"samples/{sample_id}.json"

        # 保存文件
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
        point_count_before_sampling = int(len(points))
        points, colors = sample_pointcloud(
            points, colors, POINT_CLOUD_SAMPLE_COUNT, seed=stable_seed(sample_id)
        )
        save_ply(self.output_dir / point_cloud_rel, points, colors)
        voxel_points, voxel_colors = voxelize_pointcloud(
            points, colors, voxel_size_cm=POINT_CLOUD_VOXEL_SIZE_CM
        )
        save_ply(self.output_dir / voxel_point_cloud_rel, voxel_points, voxel_colors)

        preprocess: dict[str, Any] = {
            "depth_source_unit": "m",
            "point_cloud_stride": self.point_cloud_stride,
            "point_count_before_sampling": point_count_before_sampling,
            "point_cloud_sampling_replace": (
                point_count_before_sampling < POINT_CLOUD_SAMPLE_COUNT
            ),
            "voxel_point_cloud_size_cm": POINT_CLOUD_VOXEL_SIZE_CM,
            "voxel_point_count": int(len(voxel_points)),
            "coordinate_normalization": alignment_stats,
            "mask_pose_check": pose_check_stats,
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

        # 导出 3D box 投影到 RGB 的可视化，便于人工检查对齐质量
        scene = CanonicalScene(
            sample_id=sample_id,
            scene_id=scene_id,
            frame_id=frame_id,
            rgb=rgb,
            depth=depth_for_cloud,
            point_cloud_path=self.output_dir / point_cloud_rel,
            camera=camera,
            objects=objects,
            voxel_point_cloud_path=self.output_dir / voxel_point_cloud_rel,
        )
        save_scene_bbox_projection(scene, self.output_dir / "rgb_bbox_vis" / f"{sample_id}.png")

        return {
            "sample_id": sample_id,
            "scene_id": scene_id,
            "frame_id": frame_id,
            "sample_path": sample_rel,
            "object_count": len(objects),
        }

    def _prepare_output_dirs(self) -> None:
        for name in ("samples", "rgb", "depth", "point_clouds", VOXEL_POINT_CLOUD_DIR, "rgb_bbox_vis"):
            (self.output_dir / name).mkdir(parents=True, exist_ok=True)


def _scale_intrinsics(intr: dict, img_w: int, img_h: int) -> dict:
    """将 meta 中原始分辨率内参缩放到实际图像尺寸。"""
    meta_w = float(intr.get("width", img_w))
    meta_h = float(intr.get("height", img_h))
    sx = float(img_w) / meta_w
    sy = float(img_h) / meta_h
    return {
        "fx": float(intr["fx"]) * sx,
        "fy": float(intr["fy"]) * sy,
        "cx": float(intr["cx"]) * sx,
        "cy": float(intr["cy"]) * sy,
    }


def _quat_wxyz_to_rotation(quat: list[float]) -> np.ndarray:
    """wxyz 四元数 → 3x3 旋转矩阵。"""
    w, x, y, z = [float(v) for v in quat]
    n = w * w + x * x + y * y + z * z
    s = 2.0 / n if n > 0 else 0.0
    xx, yy, zz = x * x * s, y * y * s, z * z * s
    xy, xz, yz = x * y * s, x * z * s, y * z * s
    wx, wy, wz = w * x * s, w * y * s, w * z * s
    return np.array(
        [
            [1.0 - yy - zz, xy - wz, xz + wy],
            [xy + wz, 1.0 - xx - zz, yz - wx],
            [xz - wy, yz + wx, 1.0 - xx - yy],
        ],
        dtype=np.float64,
    )


def check_pose_alignment_with_mask(
    objects: list[ObjectInfo],
    mask_ids: list[int],
    mask: np.ndarray | None,
    depth_cm: np.ndarray,
    camera: CameraParams,
    depth_tol_cm: float = MASK_CHECK_DEPTH_TOL_CM,
    min_points: int = MASK_CHECK_MIN_POINTS,
    max_offset_cm: float = MASK_CHECK_MAX_LATERAL_OFFSET_CM,
) -> tuple[bool, dict[str, Any]]:
    """用 mask.exr + 深度检测 pose 与物体可见表面的横向对齐质量。

    ROPE pose 标注与图像存在逐帧刚性平移误差（实测主要在相机系 X/Y）。这里逐物体
    取 mask 内、深度接近物体中心深度的点求 3D 质心，计算其与 pose 中心的相机系横向
    (X, Y) 偏差（不含深度 Z，避免俯视下可见表面深度偏差干扰）。任一可判定物体偏差
    超 max_offset_cm 即判定该帧 pose 不准。mask 缺失或无任何可判定物体（点数不足）
    时也视为不可信。仅检测、不修改 pose。

    返回 (ok, stats)：ok=False 时调用方据此跳过该帧。
    """
    if mask is None:
        return False, {"ok": False, "reason": "mask_missing"}

    mask_id_map = np.round(np.asarray(mask, dtype=np.float64) * 255.0).astype(np.int64)
    fx, fy, cx, cy = camera.fx, camera.fy, camera.cx, camera.cy

    # 逐物体求 pose 中心与 mask 可见表面质心的横向偏差（点数不足记为 None）
    offsets: list[float | None] = []
    for obj, mask_id in zip(objects, mask_ids):
        t = obj.pose_world[:3, 3]
        ys, xs = np.where(mask_id_map == int(mask_id))
        offset = None
        if len(xs) >= min_points:
            z = depth_cm[ys, xs]
            keep = (z > 0.0) & (np.abs(z - t[2]) < depth_tol_cm)
            if int(keep.sum()) >= min_points:
                zc = z[keep]
                xc = (xs[keep] - cx) * zc / fx
                yc = (ys[keep] - cy) * zc / fy
                offset = float(np.hypot(xc.mean() - t[0], yc.mean() - t[1]))
        offsets.append(offset)

    valid = [o for o in offsets if o is not None]
    per_object_cm = [None if o is None else round(o, 3) for o in offsets]

    if not valid:
        return False, {
            "ok": False,
            "reason": "no_verifiable_object",
            "object_count": len(objects),
            "per_object_offset_cm": per_object_cm,
        }

    max_off = max(valid)
    ok = max_off <= max_offset_cm
    stats: dict[str, Any] = {
        "ok": ok,
        "method": "mask_depth_centroid_lateral_offset",
        "rule": "any_object_over_threshold",
        "depth_tol_cm": depth_tol_cm,
        "min_points": min_points,
        "max_offset_cm": max_offset_cm,
        "object_count": len(objects),
        "verifiable_object_count": len(valid),
        "max_object_offset_cm": round(max_off, 3),
        "per_object_offset_cm": per_object_cm,
    }
    if not ok:
        stats["reason"] = "pose_offset_over_threshold"
    return ok, stats


def _parse_objects(meta: dict, E_c2w: np.ndarray) -> tuple[list[ObjectInfo], list[int]]:
    """从帧 meta 构造 ObjectInfo 列表与对应 mask id，pose_world 单位 cm。"""
    objects = []
    mask_ids: list[int] = []
    for idx, (_, record) in enumerate(sorted(meta.get("objects", {}).items())):
        obj_meta = record.get("meta", {})
        if obj_meta.get("is_background", False):
            continue
        if not bool(record.get("is_valid", True)):
            continue

        # object->camera 变换（平移单位 m → cm）
        rot = _quat_wxyz_to_rotation(record["quaternion_wxyz"])
        t_cm = np.asarray(record["translation"], dtype=np.float64) * M_TO_CM
        pose_cam = np.eye(4, dtype=np.float64)
        pose_cam[:3, :3] = rot
        pose_cam[:3, 3] = t_cm

        # ROPE: E_c2w = identity，所以 pose_world = pose_cam
        pose_world = E_c2w @ pose_cam

        # centered canonical AABB from bbox_side_len (m → cm)
        side_cm = np.asarray(obj_meta["bbox_side_len"], dtype=np.float64) * M_TO_CM
        half = side_cm * 0.5
        bbox = np.concatenate([-half, half])

        obj_id = record.get("id", idx + 1)
        oid = obj_meta.get("oid", str(obj_id))
        objects.append(
            ObjectInfo(
                obj_id=f"obj_{obj_id}_{oid.replace('-', '_')}",
                class_name=str(obj_meta.get("class_name", oid)),
                bbox3d_canonical=bbox,
                pose_world=pose_world,
            )
        )
        mask_ids.append(int(obj_id))
    return objects, mask_ids

"""
src/datasets/dopose_to_canonical.py
-----------------------------------
DOPose BOP 风格数据到 canonical placement scene 数据集的离线转换。

这里集中处理 DOPose 特有约定：
- depth raw uint16 * scene_camera.depth_scale -> mm，再转换为 cm
- scene_transformations 中 zivid_optical_frame -> scene_link 平移 m -> cm
- 从深度点云 RANSAC 拟合支撑面，估计 scene_link -> canonical Z-up world
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


DATASET_NAME = "dopose"
MM_TO_CM = 0.1
M_TO_CM = 100.0
POINT_CLOUD_SAMPLE_COUNT = 50000
POINT_CLOUD_VOXEL_SIZE_CM = 1.0
VOXEL_POINT_CLOUD_DIR = "point_clouds_voxel_1cm"
CANONICAL_UP = np.array([0.0, 0.0, 1.0], dtype=np.float64)
SUPPORT_PLANE_DISTANCE_THRESH_CM = 1.0
SUPPORT_PLANE_RANSAC_ITERS = 512
SUPPORT_PLANE_MIN_INLIERS = 500
SUPPORT_PLANE_MIN_INLIER_RATIO = 0.03
SUPPORT_PLANE_MAX_RANSAC_POINTS = 50000
SUPPORT_PLANE_OBJECT_BOTTOM_TOLERANCE_CM = 8.0


class DoPoseCanonicalConverter:
    """将 DOPose 帧转换为 canonical placement scene sample。"""

    def __init__(
        self,
        root_dir: str | Path,
        output_dir: str | Path,
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
        self.point_cloud_stride = int(point_cloud_stride)
        self.depth_window_cm = depth_window_cm
        self.support_plane_distance_thresh_cm = float(support_plane_distance_thresh_cm)
        self.support_plane_ransac_iters = int(support_plane_ransac_iters)
        self.support_plane_min_inliers = int(support_plane_min_inliers)
        self.support_plane_min_inlier_ratio = float(support_plane_min_inlier_ratio)
        self.support_plane_max_points = int(support_plane_max_points)
        self.model_info = self._load_model_info()
        self.model_names = self._load_model_names()
        self._bbox_cache: dict[int, np.ndarray] = {}

    def list_frames(self) -> list[tuple[Path, Path, str]]:
        """列出需要转换的 DOPose split/scene/frame。"""
        frames: list[tuple[Path, Path, str]] = []
        for split_dir in sorted(self.root_dir.glob("test_*")):
            if not split_dir.is_dir():
                continue
            for scene_dir in sorted(path for path in split_dir.iterdir() if path.is_dir()):
                with (scene_dir / "scene_camera.json").open("r") as f:
                    frame_keys = sorted(json.load(f), key=lambda value: int(value))
                frames.extend((split_dir, scene_dir, frame_key) for frame_key in frame_keys)
        return frames

    def convert_all(self, max_frames: int | None = None) -> dict[str, Any]:
        """转换全部帧，并写入 manifest.json。"""
        self._prepare_output_dirs()
        frames = self.list_frames()
        if max_frames is not None:
            frames = frames[: int(max_frames)]

        sample_records = [
            self.convert_frame(split_dir, scene_dir, frame_key)
            for split_dir, scene_dir, frame_key in frames
        ]
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
                "depth_scale_source": "scene_camera.depth_scale",
                "camera_to_world": "zivid_optical_frame_to_scene_link_then_support_surface_z_up",
                "coordinate_normalization_method": "support_surface_ransac_to_world_z",
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

    def convert_frame(self, split_dir: Path, scene_dir: Path, frame_key: str) -> dict[str, Any]:
        """转换单帧 DOPose 数据，并返回 manifest 中的轻量记录。"""
        scene_id = make_scene_id(split_dir.name, scene_dir.name)
        frame_id = f"{int(frame_key):06d}"
        sample_id = make_sample_id(scene_id, frame_id)

        scene_camera = self._load_scene_json(scene_dir, "scene_camera.json")
        scene_gt = self._load_scene_json(scene_dir, "scene_gt.json")
        rgb_src = scene_dir / "rgb" / f"{frame_id}.png"
        depth_src = scene_dir / "depth" / f"{frame_id}.png"

        rgb = np.asarray(Image.open(rgb_src).convert("RGB"), dtype=np.uint8)
        depth_raw = np.asarray(Image.open(depth_src), dtype=np.float32)
        depth_scale = float(scene_camera[frame_key].get("depth_scale", 1.0))
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

        raw_camera = self._load_camera(scene_dir, frame_key, scene_camera[frame_key], rgb.shape[:2])
        raw_objects = self._load_objects(scene_gt[frame_key], raw_camera.E_c2w)
        alignment_points, _ = depth_to_pointcloud(
            depth_for_point_cloud,
            rgb,
            raw_camera.fx,
            raw_camera.fy,
            raw_camera.cx,
            raw_camera.cy,
            raw_camera.E_c2w,
            stride=self.point_cloud_stride,
        )
        alignment, alignment_stats = estimate_support_plane_alignment(
            alignment_points,
            raw_camera.E_c2w[:3, 3],
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
        """创建 canonical 数据集固定目录。"""
        for name in ("samples", "rgb", "depth", "point_clouds", VOXEL_POINT_CLOUD_DIR):
            (self.output_dir / name).mkdir(parents=True, exist_ok=True)

    def _load_scene_json(self, scene_dir: Path, name: str) -> dict[str, Any]:
        """读取单个 scene 下的 BOP JSON 文件。"""
        with (scene_dir / name).open("r") as f:
            return json.load(f)

    def _load_camera(
        self,
        scene_dir: Path,
        frame_key: str,
        camera_record: dict[str, Any],
        image_hw: tuple[int, int],
    ) -> CameraParams:
        """读取 DOPose 相机并统一到 cm 单位的 camera->world 外参。"""
        K = np.asarray(camera_record["cam_K"], dtype=np.float64).reshape(3, 3)
        img_h, img_w = image_hw
        return CameraParams(
            fx=float(K[0, 0]),
            fy=float(K[1, 1]),
            cx=float(K[0, 2]),
            cy=float(K[1, 2]),
            E_c2w=self._load_camera_to_world(scene_dir, frame_key),
            img_w=int(img_w),
            img_h=int(img_h),
        )

    def _load_camera_to_world(self, scene_dir: Path, frame_key: str) -> np.ndarray:
        """读取 zivid_optical_frame -> scene_link 作为 camera->world。"""
        transform_path = scene_dir / "scene_transformations.json"
        if not transform_path.exists():
            return np.eye(4, dtype=np.float64)

        with transform_path.open("r") as f:
            frame_transforms = json.load(f).get(frame_key, [])
        transform = next(
            (
                item
                for item in frame_transforms
                if item.get("source_frame") == "zivid_optical_frame"
                and item.get("target_frame") == "scene_link"
            ),
            None,
        )
        if transform is None:
            return np.eye(4, dtype=np.float64)

        quat = transform["rotation_quaternion"]
        translation = transform["translation"]
        E_c2w = np.eye(4, dtype=np.float64)
        E_c2w[:3, :3] = quaternion_to_matrix(
            float(quat["x"]),
            float(quat["y"]),
            float(quat["z"]),
            float(quat["w"]),
        )
        E_c2w[:3, 3] = np.array(
            [
                float(translation["x"]),
                float(translation["y"]),
                float(translation["z"]),
            ],
            dtype=np.float64,
        ) * M_TO_CM
        return E_c2w

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
                    class_name=self.get_class_name(obj_id),
                    bbox3d_canonical=self.get_object_bbox(obj_id),
                    pose_world=E_c2w @ pose_cam,
                )
            )
        return objects

    def _load_model_info(self) -> dict[str, Any]:
        """读取 DOPose 模型几何信息。"""
        with (self.root_dir / "models" / "models_info.json").open("r") as f:
            return json.load(f)

    def _load_model_names(self) -> dict[str, Any]:
        """读取 DOPose 物体类别名。"""
        names_path = self.root_dir / "models_names.json"
        if not names_path.exists():
            return {}
        with names_path.open("r") as f:
            return json.load(f)

    def get_class_name(self, obj_id: int) -> str:
        """从 models_names.json 读取类别名，缺失时退回 BOP obj id。"""
        record = self.model_names.get(str(obj_id), {})
        return str(record.get("name", f"obj_{obj_id:06d}"))

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


def rotation_between_vectors(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    """构造将 source 单位向量旋转到 target 单位向量的 3x3 矩阵。"""
    src = np.asarray(source, dtype=np.float64)
    dst = np.asarray(target, dtype=np.float64)
    src /= np.linalg.norm(src)
    dst /= np.linalg.norm(dst)
    cross = np.cross(src, dst)
    dot = float(np.clip(np.dot(src, dst), -1.0, 1.0))
    if dot > 1.0 - 1e-8:
        return np.eye(3, dtype=np.float64)
    if dot < -1.0 + 1e-8:
        axis = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        if abs(float(np.dot(axis, src))) > 0.9:
            axis = np.array([0.0, 1.0, 0.0], dtype=np.float64)
        axis = axis - np.dot(axis, src) * src
        axis /= np.linalg.norm(axis)
        return 2.0 * np.outer(axis, axis) - np.eye(3, dtype=np.float64)

    skew = np.array(
        [
            [0.0, -cross[2], cross[1]],
            [cross[2], 0.0, -cross[0]],
            [-cross[1], cross[0], 0.0],
        ],
        dtype=np.float64,
    )
    return np.eye(3, dtype=np.float64) + skew + skew @ skew * ((1.0 - dot) / np.dot(cross, cross))


def get_bbox_corners(bbox3d: np.ndarray) -> np.ndarray:
    """从 canonical AABB 生成 8 个角点。"""
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


def sample_plane_triplets(
    points: np.ndarray,
    num_iters: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """随机采样三点组，用于在输入点云中执行 RANSAC 平面拟合。"""
    if len(points) < 3:
        return np.empty((0, 3), dtype=int)
    return rng.integers(0, len(points), size=(int(num_iters), 3))


def fit_plane_from_triplet(
    points: np.ndarray,
    triplet: np.ndarray,
) -> tuple[np.ndarray | None, float | None]:
    """由三个点拟合平面，返回单位法向 normal 和平面偏置 d。"""
    p0, p1, p2 = np.asarray(points, dtype=np.float64)[triplet]
    normal = np.cross(p1 - p0, p2 - p0)
    norm = float(np.linalg.norm(normal))
    if norm < 1e-8:
        return None, None
    normal = normal / norm
    d = -float(np.dot(normal, p0))
    return normal, d


def refit_plane_from_inliers(points: np.ndarray) -> tuple[np.ndarray | None, float | None, float | None]:
    """用平面 inlier 点集 SVD 重拟合平面，并返回 normal、d 和 RMSE。"""
    pts = np.asarray(points, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] != 3 or len(pts) < 3:
        return None, None, None
    center = pts.mean(axis=0)
    _, _, vh = np.linalg.svd(pts - center, full_matrices=False)
    normal = vh[-1]
    norm = float(np.linalg.norm(normal))
    if norm < 1e-8:
        return None, None, None
    normal = normal / norm
    d = -float(np.dot(normal, center))
    distances = pts @ normal + d
    rmse = float(np.sqrt(np.mean(distances * distances)))
    return normal, d, rmse


def get_object_corners_world(obj: ObjectInfo) -> np.ndarray:
    """返回物体 canonical bbox 的 8 个角点在当前 raw world 中的位置。"""
    corners_obj = get_bbox_corners(obj.bbox3d_canonical)
    pose = np.asarray(obj.pose_world, dtype=np.float64)
    return (pose[:3, :3] @ corners_obj.T).T + pose[:3, 3]


def _orient_plane_normal(
    normal: np.ndarray,
    d: float,
    camera_center_world: np.ndarray,
    objects: list[ObjectInfo],
) -> tuple[np.ndarray, float, str]:
    """将平面法向定向到物体侧；没有物体时使用相机中心作为参考。"""
    if objects:
        centers = np.array(
            [np.asarray(obj.pose_world, dtype=np.float64)[:3, 3] for obj in objects],
            dtype=np.float64,
        )
        reference = np.median(centers, axis=0)
        reference_name = "objects"
    else:
        reference = np.asarray(camera_center_world, dtype=np.float64)
        reference_name = "camera"

    if float(np.dot(normal, reference) + d) < 0.0:
        return -normal, -float(d), reference_name
    return normal, float(d), reference_name


def _object_plane_stats(
    objects: list[ObjectInfo],
    normal: np.ndarray,
    d: float,
    bottom_tolerance_cm: float,
) -> dict[str, Any]:
    """计算物体 bbox 底部到候选支撑平面的 signed distance 统计。"""
    bottom_distances = []
    for obj in objects:
        corners = get_object_corners_world(obj)
        signed_distances = corners @ normal + d
        bottom_distances.append(float(signed_distances.min()))

    if not bottom_distances:
        return {
            "object_bottom_signed_distances_cm": [],
            "supported_object_count": 0,
            "penetrating_object_count": 0,
            "median_abs_object_bottom_distance_cm": None,
        }

    distances = np.asarray(bottom_distances, dtype=np.float64)
    tolerance = float(bottom_tolerance_cm)
    return {
        "object_bottom_signed_distances_cm": bottom_distances,
        "supported_object_count": int(np.count_nonzero(np.abs(distances) <= tolerance)),
        "penetrating_object_count": int(np.count_nonzero(distances < -tolerance)),
        "median_abs_object_bottom_distance_cm": float(np.median(np.abs(distances))),
    }


def _is_better_support_plane(candidate: dict[str, Any], best: dict[str, Any] | None) -> bool:
    """按物体贴合度、穿透数量、inlier 数和残差比较支撑面候选。"""
    if best is None:
        return True
    candidate_key = (
        int(candidate["supported_object_count"]),
        -int(candidate["penetrating_object_count"]),
        int(candidate["inlier_count"]),
        -float(candidate["rmse_cm"]),
        -float(candidate["median_abs_object_bottom_distance_cm"] or 0.0),
    )
    best_key = (
        int(best["supported_object_count"]),
        -int(best["penetrating_object_count"]),
        int(best["inlier_count"]),
        -float(best["rmse_cm"]),
        -float(best["median_abs_object_bottom_distance_cm"] or 0.0),
    )
    return candidate_key > best_key


def estimate_support_plane_alignment(
    points_world: np.ndarray,
    camera_center_world: np.ndarray,
    objects: list[ObjectInfo],
    distance_thresh_cm: float = SUPPORT_PLANE_DISTANCE_THRESH_CM,
    num_iters: int = SUPPORT_PLANE_RANSAC_ITERS,
    min_inlier_count: int = SUPPORT_PLANE_MIN_INLIERS,
    min_inlier_ratio: float = SUPPORT_PLANE_MIN_INLIER_RATIO,
    max_ransac_points: int = SUPPORT_PLANE_MAX_RANSAC_POINTS,
    object_bottom_tolerance_cm: float = SUPPORT_PLANE_OBJECT_BOTTOM_TOLERANCE_CM,
    random_seed: int = 0,
) -> tuple[np.ndarray, dict[str, Any]]:
    """
    在 raw world 点云中 RANSAC 拟合支撑面，并返回 Z-up 对齐变换。

    输入点云必须位于 DOPose raw scene_link/world 坐标系且单位为 cm；返回的
    alignment 可同时作用于 camera、objects 和点云，并把支撑面平移到 z=0。
    """
    points = np.asarray(points_world, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("support plane points_world must have shape (N, 3)")
    if len(points) < 3:
        raise ValueError("not enough points for support plane RANSAC")

    distance_thresh_cm = float(distance_thresh_cm)
    if distance_thresh_cm <= 0.0:
        raise ValueError(f"distance_thresh_cm must be positive, got {distance_thresh_cm}")
    num_iters = int(num_iters)
    if num_iters <= 0:
        raise ValueError(f"num_iters must be positive, got {num_iters}")
    min_inlier_count = int(min_inlier_count)
    min_inlier_ratio = float(min_inlier_ratio)
    max_ransac_points = int(max_ransac_points)
    if max_ransac_points <= 0:
        raise ValueError(f"max_ransac_points must be positive, got {max_ransac_points}")

    rng = np.random.default_rng(int(random_seed))
    if len(points) > max_ransac_points:
        sample_indices = rng.choice(len(points), size=max_ransac_points, replace=False)
        ransac_points = points[sample_indices]
    else:
        ransac_points = points

    min_required = max(min_inlier_count, int(np.ceil(len(ransac_points) * min_inlier_ratio)))
    triplets = sample_plane_triplets(ransac_points, num_iters, rng)
    best = None
    for triplet in triplets:
        if len({int(triplet[0]), int(triplet[1]), int(triplet[2])}) < 3:
            continue
        normal, d = fit_plane_from_triplet(ransac_points, triplet)
        if normal is None or d is None:
            continue

        distances = np.abs(ransac_points @ normal + d)
        inlier_mask = distances <= distance_thresh_cm
        if int(inlier_mask.sum()) < min_required:
            continue

        refit_normal, refit_d, rmse = refit_plane_from_inliers(ransac_points[inlier_mask])
        if refit_normal is None or refit_d is None or rmse is None:
            continue
        normal, d, reference_name = _orient_plane_normal(
            refit_normal,
            refit_d,
            camera_center_world,
            objects,
        )
        signed_distances = ransac_points @ normal + d
        distances = np.abs(signed_distances)
        inlier_mask = distances <= distance_thresh_cm
        inlier_count = int(inlier_mask.sum())
        if inlier_count < min_required:
            continue

        inlier_distances = distances[inlier_mask]
        object_stats = _object_plane_stats(objects, normal, d, object_bottom_tolerance_cm)
        candidate = {
            "normal": normal,
            "d": float(d),
            "inlier_mask": inlier_mask,
            "inlier_count": inlier_count,
            "inlier_ratio": float(inlier_count / len(ransac_points)),
            "rmse_cm": float(np.sqrt(np.mean(inlier_distances * inlier_distances))),
            "median_abs_error_cm": float(np.median(inlier_distances)),
            "normal_orientation_reference": reference_name,
            **object_stats,
        }
        if _is_better_support_plane(candidate, best):
            best = candidate

    if best is None:
        raise ValueError(
            "failed to find support plane by RANSAC: "
            f"points={len(points)}, ransac_points={len(ransac_points)}, "
            f"min_required={min_required}, distance_thresh_cm={distance_thresh_cm}"
        )

    normal = np.asarray(best["normal"], dtype=np.float64)
    rotation = rotation_between_vectors(normal, CANONICAL_UP)
    inlier_points = ransac_points[np.asarray(best["inlier_mask"], dtype=bool)]
    rotated_inliers = (rotation @ inlier_points.T).T
    support_z = float(np.median(rotated_inliers[:, 2]))
    alignment = np.eye(4, dtype=np.float64)
    alignment[:3, :3] = rotation
    alignment[2, 3] = -support_z
    correction_degrees = float(
        np.degrees(np.arccos(np.clip(np.dot(normal, CANONICAL_UP), -1.0, 1.0)))
    )
    stats = {
        "enabled": True,
        "method": "support_surface_ransac_to_world_z",
        "source_world_frame": "scene_link",
        "target_world_frame": "canonical_z_up",
        "point_count": int(len(points)),
        "ransac_point_count": int(len(ransac_points)),
        "ransac_iterations": int(num_iters),
        "distance_thresh_cm": float(distance_thresh_cm),
        "min_inlier_count": int(min_inlier_count),
        "min_inlier_ratio": float(min_inlier_ratio),
        "min_required_inliers": int(min_required),
        "inlier_count": int(best["inlier_count"]),
        "inlier_ratio": float(best["inlier_ratio"]),
        "plane_rmse_cm": float(best["rmse_cm"]),
        "plane_median_abs_error_cm": float(best["median_abs_error_cm"]),
        "support_normal_before": normal.tolist(),
        "plane_d_before": float(best["d"]),
        "normal_orientation_reference": str(best["normal_orientation_reference"]),
        "correction_degrees": correction_degrees,
        "support_z_before_translation": support_z,
        "z_translation_cm": float(alignment[2, 3]),
        "object_bottom_tolerance_cm": float(object_bottom_tolerance_cm),
        "object_bottom_signed_distances_cm": best["object_bottom_signed_distances_cm"],
        "supported_object_count": int(best["supported_object_count"]),
        "penetrating_object_count": int(best["penetrating_object_count"]),
        "median_abs_object_bottom_distance_cm": best["median_abs_object_bottom_distance_cm"],
    }
    return alignment, stats


def align_camera(camera: CameraParams, alignment: np.ndarray) -> CameraParams:
    """将 camera->world 外参变换到 canonical Z-up world。"""
    return CameraParams(
        fx=camera.fx,
        fy=camera.fy,
        cx=camera.cx,
        cy=camera.cy,
        E_c2w=alignment @ camera.E_c2w,
        img_w=camera.img_w,
        img_h=camera.img_h,
    )


def align_objects(objects: list[ObjectInfo], alignment: np.ndarray) -> list[ObjectInfo]:
    """将 object->world 位姿变换到 canonical Z-up world。"""
    return [
        ObjectInfo(
            obj_id=obj.obj_id,
            class_name=obj.class_name,
            bbox3d_canonical=obj.bbox3d_canonical,
            pose_world=alignment @ obj.pose_world,
        )
        for obj in objects
    ]


def make_scene_id(split_name: str, scene_name: str) -> str:
    """生成包含 split 的 scene_id，避免 test_bin/test_table 场景编号冲突。"""
    return f"{split_name}_{scene_name}"


def make_sample_id(scene_id: str, frame_id: str) -> str:
    """生成稳定且可读的 DOPose sample_id。"""
    return f"{DATASET_NAME}__{scene_id}__{frame_id}"


def quaternion_to_matrix(x: float, y: float, z: float, w: float) -> np.ndarray:
    """将 xyzw 四元数转换为 3x3 旋转矩阵。"""
    quat = np.array([x, y, z, w], dtype=np.float64)
    quat /= np.linalg.norm(quat)
    x, y, z, w = quat
    return np.array(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def stable_seed(value: str) -> int:
    """从字符串生成跨 Python 进程稳定的 32-bit 随机种子。"""
    digest = hashlib.blake2s(value.encode("utf-8"), digest_size=4).digest()
    return int.from_bytes(digest, byteorder="little")

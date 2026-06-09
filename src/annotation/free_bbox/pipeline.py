"""
src/annotation/free_bbox/pipeline.py
------------------------------------
基于 canonical 体素点云的 free_bbox 放置标注 Pipeline。

流程:
    CanonicalScene -> voxel PLY -> OCCUPIED/FREE 栅格 -> 逐物体:
        支撑面检测 -> 支撑面本层 FFT 碰撞搜索 -> 稳定/可见/遮挡过滤 ->
        底面中心约束 -> DBSCAN 聚类 -> 每簇最优 3D box 和热力 PLY 输出
"""

from __future__ import annotations

import os
import re
from pathlib import Path

import numpy as np

from src.annotation.free_bbox.cluster import build_heat_counts, cluster_placements_best
from src.annotation.free_bbox.collision import find_table_placements
from src.annotation.free_bbox.datatypes import FreeBBoxConfig, FreeBBoxResult
from src.annotation.free_bbox.filters import (
    build_depth_buffer,
    filter_occluded_placements,
    filter_stable_placements,
    filter_visible_placements,
    is_fully_visible,
)
from src.annotation.free_bbox.geometry import get_bbox_corners, transform_points
from src.annotation.free_bbox.grid_ops import prepare_grid_base, voxelize_obb
from src.annotation.free_bbox.io_utils import (
    load_ply,
    save_binary_mask_ply,
    save_heatmap_ply,
    save_json,
)
from src.annotation.free_bbox.occupancy import FREE, OCCUPIED, build_grid_from_voxel_points
from src.annotation.free_bbox.surface import detect_support_surfaces
from src.annotation.free_bbox.voxel_utils import make_voxel_params
from src.annotation.free_bbox.visualize import save_freebox_visualization


def _sanitize_filename(value: str) -> str:
    """将标识转换为稳定文件名片段。"""
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", str(value)).strip("._")
    return safe or "item"


def _sample_prefix(scene) -> str:
    """返回当前帧输出文件前缀。"""
    return _sanitize_filename(getattr(scene, "sample_id", f"{scene.scene_id}_{scene.frame_id}"))


def _build_output_dirs(output_root: Path) -> dict[str, Path]:
    """创建并返回按文件类型划分的输出目录。"""
    paths = {
        "root": output_root,
        "placements": output_root / "placements",
        "boxes": output_root / "boxes",
        "heatmaps": output_root / "heatmaps",
        "support_masks": output_root / "support_masks",
        "visualizations": output_root / "visualizations",
    }
    for key, path in paths.items():
        if key != "root":
            path.mkdir(parents=True, exist_ok=True)
    return paths


def _relative_output_path(output_root: Path, path: Path) -> str:
    """返回输出根目录下的相对路径字符串。"""
    return os.fspath(path.relative_to(output_root))


def _object_bbox_world_corners(objects: list) -> np.ndarray:
    """汇总所有物体原始 OBB 角点，用于扩展搜索栅格边界。"""
    all_corners = []
    for obj in objects:
        corners = transform_points(get_bbox_corners(obj.bbox3d_canonical), obj.pose_world)
        all_corners.append(corners)
    if not all_corners:
        return np.empty((0, 3), dtype=np.float64)
    return np.vstack(all_corners)


def _compute_placed_transform(
    anchor_xy: np.ndarray,
    landing_z: int,
    yaw_data: dict,
    yaw_idx: int,
    vp: dict,
) -> np.ndarray:
    """根据候选 anchor 恢复 object->world 放置变换。"""
    voxel_size = float(vp["voxel_size"])
    T_rot = np.asarray(yaw_data["T_rotated"][yaw_idx], dtype=np.float64)
    vmin_rot = np.asarray(yaw_data["vmin_rot_abs"][yaw_idx], dtype=np.float64)
    anchor_3d = np.array([anchor_xy[0], anchor_xy[1], int(landing_z)], dtype=np.float64)
    delta_world = (anchor_3d - vmin_rot) * voxel_size
    transform = np.array(T_rot, copy=True)
    transform[:3, 3] += delta_world
    return transform


def _support_mask_3d(surface_mask_2d: np.ndarray, table_z: int, grid_shape: tuple[int, int, int]) -> np.ndarray:
    """将二维支撑面 mask 放入三维体素层。"""
    mask_3d = np.zeros(tuple(grid_shape), dtype=bool)
    if surface_mask_2d is not None and 0 <= int(table_z) < grid_shape[2]:
        mask_3d[:, :, int(table_z)] = np.asarray(surface_mask_2d, dtype=bool)
    return mask_3d


def _build_saved_placements(
    scene_prefix: str,
    obj,
    reps: np.ndarray,
    cluster_infos: list[dict],
    landing_z: int,
    yaw_data: dict,
    vp: dict,
) -> list[dict]:
    """将每簇最优候选转换为可保存的 3D box 标注。"""
    corners_obj = get_bbox_corners(obj.bbox3d_canonical)
    placements = []
    for rank, (rep, info) in enumerate(zip(reps, cluster_infos)):
        yaw_idx = int(rep[2])
        transform = _compute_placed_transform(rep[:2], landing_z, yaw_data, yaw_idx, vp)
        placed_world = transform_points(corners_obj, transform)
        aabb_world = np.concatenate([placed_world.min(axis=0), placed_world.max(axis=0)])
        center_world = placed_world.mean(axis=0)
        cluster_id = int(info["cluster_id"])
        placements.append(
            {
                "sample_id": f"{scene_prefix}_{obj.obj_id}_cluster_{cluster_id:03d}",
                "rank": int(rank),
                "cluster_id": cluster_id,
                "center_world": center_world.tolist(),
                "yaw_degrees": float(info["yaw_degrees"]),
                "transform_world": transform.tolist(),
                "aabb_world": aabb_world.tolist(),
                "corners_world": placed_world.tolist(),
                "anchor_voxel": info["anchor_voxel"],
                "bottom_center_voxel": info["bottom_center_voxel"],
                "bottom_center_world": info["bottom_center_world"],
                "support_area": float(info["support_area"]),
                "support_area_voxels": int(info["support_area_voxels"]),
                "clearance": float(info["clearance"]),
                "clearance_voxels": float(info["clearance_voxels"]),
                "cluster_size": int(info["size"]),
            }
        )
    return placements


class FreeBBoxPipeline:
    """canonical 体素点云 free_bbox 放置标注 Pipeline。"""

    def __init__(self, config: FreeBBoxConfig | None = None) -> None:
        self.config = config or FreeBBoxConfig()

    def run(self, scene, output_dir: str | Path | None = None) -> dict[str, FreeBBoxResult]:
        """
        对单帧 canonical scene 执行 free_bbox 放置标注。

        输入:
            scene: src.datasets.canonical.load_canonical_scene 返回的 CanonicalScene
            output_dir: 输出目录；不传则只返回内存结果
        输出:
            dict[obj_id, FreeBBoxResult]
        """
        cfg = self.config
        output_root = Path(output_dir).resolve() if output_dir is not None else None
        output_paths = None
        if output_root is not None:
            output_root.mkdir(parents=True, exist_ok=True)
            output_paths = _build_output_dirs(output_root)

        if scene.voxel_point_cloud_path is None:
            raise ValueError("canonical scene missing voxel_point_cloud_path")

        scene_prefix = _sample_prefix(scene)
        voxel_points, _ = load_ply(scene.voxel_point_cloud_path)
        extra_points = _object_bbox_world_corners(scene.objects)
        grid_scene, grid_min, voxel_size = build_grid_from_voxel_points(
            voxel_points,
            voxel_size=cfg.voxel_size,
            padding=cfg.grid_padding,
            extra_points=extra_points,
        )
        vp = make_voxel_params(grid_min, voxel_size)
        grid_base = prepare_grid_base(grid_scene, scene.objects, vp)
        frame_support_mask = np.zeros(grid_scene.shape, dtype=bool)

        camera = scene.camera
        K = camera.K
        E_w2c = camera.E_w2c
        all_results: dict[str, FreeBBoxResult] = {}
        summary_objects = []
        summary_boxes = []

        for obj in scene.objects:
            print(f"  Processing {obj.obj_id}: {obj.class_name}")
            corners_obj = get_bbox_corners(obj.bbox3d_canonical)
            orig_world = transform_points(corners_obj, obj.pose_world)
            orig_aabb = np.concatenate([orig_world.min(axis=0), orig_world.max(axis=0)])

            target_voxels = voxelize_obb(
                obj.bbox3d_canonical,
                obj.pose_world,
                vp,
                np.asarray(grid_base.shape, dtype=int),
            )
            if len(target_voxels) == 0:
                all_results[obj.obj_id] = FreeBBoxResult(
                    obj_id=obj.obj_id,
                    class_name=obj.class_name,
                    original_aabb_world=orig_aabb,
                    placements=[],
                )
                continue

            grid_other = np.array(grid_base, copy=True)
            grid_other[target_voxels[:, 0], target_voxels[:, 1], target_voxels[:, 2]] = FREE
            depth_buffer = build_depth_buffer(grid_other, vp, K, E_w2c, camera.img_w, camera.img_h)

            pose_cam = E_w2c @ obj.pose_world
            if not is_fully_visible(
                obj.bbox3d_canonical,
                pose_cam,
                camera.fx,
                camera.fy,
                camera.cx,
                camera.cy,
                camera.img_w,
                camera.img_h,
                depth_buffer=depth_buffer,
                depth_margin=voxel_size,
            ):
                all_results[obj.obj_id] = FreeBBoxResult(
                    obj_id=obj.obj_id,
                    class_name=obj.class_name,
                    original_aabb_world=orig_aabb,
                    placements=[],
                )
                continue

            table_z, surface_mask = detect_support_surfaces(
                grid_other,
                vp,
                min_area=cfg.min_surface_area,
                points_world=voxel_points,
                target_voxels=target_voxels,
            )
            if table_z is None or surface_mask is None:
                all_results[obj.obj_id] = FreeBBoxResult(
                    obj_id=obj.obj_id,
                    class_name=obj.class_name,
                    original_aabb_world=orig_aabb,
                    placements=[],
                )
                continue

            support_3d = _support_mask_3d(surface_mask, table_z, grid_scene.shape)
            frame_support_mask |= support_3d

            candidates, meta, yaw_data = find_table_placements(
                grid_base,
                obj.bbox3d_canonical,
                obj.pose_world,
                vp,
                table_z,
                surface_mask,
                safety_margin=cfg.safety_margin,
                yaw_steps=cfg.yaw_steps,
                preserve_orientation=cfg.preserve_orientation,
            )
            n_raw = int(meta["valid_raw"])
            landing_z = int(meta["landing_z"])

            candidates = filter_stable_placements(
                candidates,
                yaw_data,
                surface_mask,
                min_support_ratio=cfg.min_support_ratio,
                chunk_size=cfg.stability_chunk_size,
            )
            n_stable = len(candidates)

            candidates = filter_visible_placements(
                candidates,
                landing_z,
                obj.bbox3d_canonical,
                obj.pose_world,
                E_w2c,
                K,
                camera.img_w,
                camera.img_h,
                vp,
                yaw_data,
            )
            n_visible = len(candidates)

            candidates = filter_occluded_placements(
                candidates,
                landing_z,
                obj.bbox3d_canonical,
                obj.pose_world,
                depth_buffer,
                K,
                E_w2c,
                vp,
                yaw_data,
                camera.img_w,
                camera.img_h,
                occlusion_threshold=cfg.occlusion_threshold,
            )
            n_occlusion = len(candidates)

            reps, cluster_infos, cluster_records, filtered_candidates = cluster_placements_best(
                candidates,
                grid_base,
                yaw_data,
                landing_z,
                surface_mask,
                vp,
                eps=cfg.dbscan_eps,
                min_samples=cfg.dbscan_min_samples,
                max_reps_total=cfg.max_reps_total,
                chunk_size=cfg.metric_chunk_size,
            )
            n_bottom_center = len(filtered_candidates)
            placements = _build_saved_placements(
                scene_prefix,
                obj,
                reps,
                cluster_infos,
                landing_z,
                yaw_data,
                vp,
            )

            result = FreeBBoxResult(
                obj_id=obj.obj_id,
                class_name=obj.class_name,
                original_aabb_world=orig_aabb,
                placements=placements,
                num_raw_candidates=n_raw,
                num_after_stability=n_stable,
                num_after_visibility=n_visible,
                num_after_occlusion=n_occlusion,
                num_after_bottom_center=n_bottom_center,
            )
            all_results[obj.obj_id] = result

            object_summary = {
                "object_id": obj.obj_id,
                "class_name": obj.class_name,
                "canonical_aabb_object": np.asarray(obj.bbox3d_canonical, dtype=np.float64).tolist(),
                "original_pose_world": np.asarray(obj.pose_world, dtype=np.float64).tolist(),
                "original_aabb_world": orig_aabb.tolist(),
                "num_raw_candidates": n_raw,
                "num_after_stability": int(n_stable),
                "num_after_visibility": int(n_visible),
                "num_after_occlusion": int(n_occlusion),
                "num_after_bottom_center": int(n_bottom_center),
                "placements": placements,
            }
            summary_objects.append(object_summary)

            if output_paths is not None:
                self._save_cluster_outputs(
                    output_paths,
                    scene_prefix,
                    scene,
                    obj,
                    placements,
                    cluster_records,
                    support_3d,
                    surface_mask,
                    table_z,
                    grid_scene,
                    grid_base,
                    vp,
                    summary_boxes,
                )

        if output_paths is not None:
            support_mask_path = output_paths["support_masks"] / f"{scene_prefix}__support_mask.ply"
            save_binary_mask_ply(support_mask_path, grid_scene, vp, frame_support_mask)
            placements_path = output_paths["placements"] / f"{scene_prefix}__placements.json"
            save_json(
                placements_path,
                {
                    "schema_version": "free_bbox_placements/v1",
                    "sample_id": scene.sample_id,
                    "scene_id": scene.scene_id,
                    "frame_id": scene.frame_id,
                    "unit": scene.unit,
                    "voxel_point_cloud_path": os.fspath(scene.voxel_point_cloud_path),
                    "support_mask_ply": _relative_output_path(output_root, support_mask_path),
                    "objects": summary_objects,
                    "box_files": summary_boxes,
                },
            )

        return all_results

    def _save_cluster_outputs(
        self,
        output_paths: dict[str, Path],
        scene_prefix: str,
        scene,
        obj,
        placements: list[dict],
        cluster_records: list[dict],
        support_mask_3d: np.ndarray,
        surface_mask_2d: np.ndarray,
        table_z: int,
        grid_scene: np.ndarray,
        grid_vis: np.ndarray,
        vp: dict,
        summary_boxes: list[dict],
    ) -> None:
        """保存每个最优 3D box 对应的 JSON、热力 PLY 和可视化图片。"""
        output_root = output_paths["root"]
        placements_by_cluster = {int(item["cluster_id"]): item for item in placements}
        for record in cluster_records:
            info = record["info"]
            cluster_id = int(info["cluster_id"])
            placement = placements_by_cluster.get(cluster_id)
            if placement is None:
                continue

            obj_part = _sanitize_filename(obj.obj_id)
            cluster_part = f"cluster_{cluster_id:03d}"
            stem = f"{scene_prefix}__{obj_part}__{cluster_part}"
            heatmap_name = f"{stem}__heatmap.ply"
            box_name = f"{stem}__box.json"
            vis_name = f"{stem}__vis.png"
            heatmap_path = output_paths["heatmaps"] / heatmap_name
            box_path = output_paths["boxes"] / box_name
            vis_path = output_paths["visualizations"] / vis_name

            heat_counts = build_heat_counts(record["member_bottom_centers"], grid_scene.shape)
            save_heatmap_ply(
                heatmap_path,
                grid_scene,
                vp,
                heat_counts,
                support_mask_3d=support_mask_3d,
            )
            save_freebox_visualization(
                scene.rgb,
                obj.class_name,
                obj.bbox3d_canonical,
                obj.pose_world,
                np.asarray(placement["corners_world"], dtype=np.float64),
                scene.camera.K,
                scene.camera.E_w2c,
                vp,
                np.asarray(scene.camera.E_c2w, dtype=np.float64)[:3, 3],
                grid_vis,
                vis_path,
                surface_mask_2d,
                table_z,
                placement,
            )
            placement["box_json"] = _relative_output_path(output_root, box_path)
            placement["heatmap_ply"] = _relative_output_path(output_root, heatmap_path)
            placement["visualization_png"] = _relative_output_path(output_root, vis_path)
            payload = {
                "schema_version": "free_bbox_best_box/v1",
                "sample_id": scene.sample_id,
                "scene_id": scene.scene_id,
                "frame_id": scene.frame_id,
                "unit": scene.unit,
                "object_id": obj.obj_id,
                "class_name": obj.class_name,
                "cluster_id": cluster_id,
                "cluster_size": int(info["size"]),
                "heatmap_ply": placement["heatmap_ply"],
                "visualization_png": placement["visualization_png"],
                "placement": placement,
            }
            save_json(box_path, payload)
            summary_boxes.append(
                {
                    "object_id": obj.obj_id,
                    "cluster_id": cluster_id,
                    "box_json": placement["box_json"],
                    "heatmap_ply": placement["heatmap_ply"],
                    "visualization_png": placement["visualization_png"],
                }
            )

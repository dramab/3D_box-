#!/usr/bin/env python
"""Render the source object box and predicted placement box on an RGB point cloud.

使用示例:
    python tools/render_lc_bgplacenet_stage2_point_boxes.py \
        --dataset-dir data/hope \
        --sample-id hope__scene_0000__0005 \
        --object-id obj_3 \
        --predictions-json outputs/lc_bgplacenet_stage2_space_former_aligned_loss_48query_full_gt_guass_8_enriched/inference_custom_hope_scene_0000_0005/predictions.json \
        --support-mask outputs/free_bbox_hope/support_masks/hope__scene_0000__0005__support_mask.ply \
        --output-path outputs/visualizations/hope__scene_0000__0005_source_and_pred_boxes.png
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.annotation.free_bbox.geometry import get_bbox_corners, transform_points
from src.annotation.free_bbox.io_utils import load_ply
from src.datasets.canonical import load_canonical_scene
from src.visualization.bbox_projection import BOX_EDGES
from tools.export_canonical_sparse_voxel_vis import camera_yaw_aligned_points
from tools.infer_lc_bgplacenet_stage2 import place_box_to_corners
from tools.render_lc_bgplacenet_stage2_point_mask import (
    support_surface_mask,
    trim_white_margin,
)


SOURCE_BOX_COLOR = "#FF5A00"
PREDICTED_BOX_COLOR = "#7A00FF"
BOX_FACES = (
    (0, 1, 3, 2),
    (4, 5, 7, 6),
    (0, 1, 5, 4),
    (2, 3, 7, 6),
    (0, 2, 6, 4),
    (1, 3, 7, 5),
)


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(
        description="Render source and predicted placement 3D boxes on an RGB point cloud."
    )
    parser.add_argument("--dataset-dir", type=Path, required=True, help="canonical 数据集根目录。")
    parser.add_argument("--sample-id", required=True, help="canonical sample_id。")
    parser.add_argument("--object-id", required=True, help="原物体 object_id。")
    parser.add_argument("--predictions-json", type=Path, required=True, help="Stage 2 predictions.json。")
    parser.add_argument("--support-mask", type=Path, default=None, help="支撑面 mask PLY。")
    parser.add_argument("--output-path", type=Path, required=True, help="输出 PNG 路径。")
    parser.add_argument("--focus-margin-cm", type=float, default=8.0, help="两个框周围的 XY 裁剪边界。")
    parser.add_argument(
        "--support-surface-tolerance-cm",
        type=float,
        default=0.75,
        help="保留完整支撑面的高度容差。",
    )
    parser.add_argument("--point-size", type=float, default=4.2, help="点云绘制大小。")
    parser.add_argument("--view-elev", type=float, default=30.0, help="3D 相机俯视角。")
    parser.add_argument("--view-azim", type=float, default=-72.0, help="3D 相机水平朝向。")
    parser.add_argument("--box-line-width", type=float, default=4.4, help="3D box 主线宽度。")
    parser.add_argument(
        "--box-face-alpha",
        type=float,
        default=0.12,
        help="3D box 面填充透明度。",
    )
    parser.add_argument(
        "--source-only",
        action="store_true",
        help="保持双框图的点云范围，但只绘制原物体 3D box。",
    )
    return parser.parse_args()


def load_prediction(path: Path, sample_id: str, object_id: str) -> dict:
    """读取指定样本和物体的 Stage 2 最终预测。"""
    with path.open("r", encoding="utf-8") as handle:
        rows = json.load(handle)
    matches = [
        row
        for row in rows
        if row.get("sample_id") == sample_id and row.get("object_id") == object_id
    ]
    if len(matches) != 1:
        raise ValueError(
            f"Expected one prediction for {sample_id}/{object_id}, found {len(matches)}"
        )
    if len(matches[0].get("place_box", [])) != 7:
        raise ValueError("prediction place_box must contain 7 values")
    return matches[0]


def find_scene_object(scene, object_id: str):
    """从 canonical scene 中查找目标物体。"""
    for obj in scene.objects:
        if obj.obj_id == object_id:
            return obj
    raise ValueError(f"Object {object_id} not found in {scene.sample_id}")


def box_focus_mask(
    points: np.ndarray,
    source_corners: np.ndarray,
    predicted_corners: np.ndarray,
    *,
    margin_cm: float,
) -> np.ndarray:
    """保留两个 3D box 周围的 XY 点云。"""
    margin_cm = float(margin_cm)
    if margin_cm < 0.0:
        raise ValueError("margin_cm must be non-negative")
    corners = np.vstack([source_corners, predicted_corners])
    lower = corners[:, :2].min(axis=0) - margin_cm
    upper = corners[:, :2].max(axis=0) + margin_cm
    return ((points[:, :2] >= lower) & (points[:, :2] <= upper)).all(axis=1)


def draw_box(
    axis,
    corners: np.ndarray,
    *,
    color: str,
    linestyle: str,
    line_width: float,
    face_alpha: float,
) -> None:
    """绘制带半透明面的 3D box。"""
    faces = [corners[np.asarray(face, dtype=np.int64)] for face in BOX_FACES]
    axis.add_collection3d(
        Poly3DCollection(
            faces,
            facecolors=color,
            edgecolors="none",
            alpha=face_alpha,
            zorder=18,
        )
    )
    for start, end in BOX_EDGES:
        xyz = corners[[start, end]]
        axis.plot(
            xyz[:, 0],
            xyz[:, 1],
            xyz[:, 2],
            color=color,
            linewidth=line_width,
            linestyle=linestyle,
            alpha=1.0,
            solid_capstyle="round",
            zorder=20,
        )


def save_visualization(
    scene,
    scene_points: np.ndarray,
    scene_colors: np.ndarray,
    view_points: np.ndarray,
    source_corners: np.ndarray,
    predicted_corners: np.ndarray,
    output_path: Path,
    *,
    point_size: float,
    view_elev: float,
    view_azim: float,
    box_line_width: float,
    box_face_alpha: float,
    draw_prediction: bool = True,
) -> None:
    """导出无热力 mask 的 RGB 点云与两个 3D box。"""
    view_center = 0.5 * (view_points.min(axis=0) + view_points.max(axis=0))
    aligned_points = camera_yaw_aligned_points(scene_points, scene.camera, center=view_center)
    aligned_view = camera_yaw_aligned_points(view_points, scene.camera, center=view_center)
    aligned_source = camera_yaw_aligned_points(source_corners, scene.camera, center=view_center)
    aligned_prediction = camera_yaw_aligned_points(
        predicted_corners, scene.camera, center=view_center
    )

    point_min = aligned_view.min(axis=0)
    point_max = aligned_view.max(axis=0)
    span = np.maximum(point_max - point_min, 1.0)
    padding = np.array([0.035, 0.035, 0.02], dtype=np.float64) * span
    colors = np.asarray(scene_colors, dtype=np.float32) / 255.0
    halo = np.column_stack(
        [colors, np.full(len(colors), 0.24, dtype=np.float32)]
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(1556 / 200.0, 1011 / 200.0), facecolor="white")
    axis = fig.add_axes([0.0, 0.04, 1.0, 1.0], projection="3d", facecolor="white")
    axis.scatter(
        aligned_points[:, 0],
        aligned_points[:, 1],
        aligned_points[:, 2],
        c=halo,
        s=point_size * 3.0,
        marker="o",
        linewidths=0.0,
        depthshade=False,
        antialiased=True,
    )
    axis.scatter(
        aligned_points[:, 0],
        aligned_points[:, 1],
        aligned_points[:, 2],
        c=colors,
        s=point_size,
        marker="o",
        linewidths=0.0,
        depthshade=False,
        antialiased=False,
    )
    draw_box(
        axis,
        aligned_source,
        color=SOURCE_BOX_COLOR,
        linestyle="--",
        line_width=box_line_width,
        face_alpha=box_face_alpha,
    )
    if draw_prediction:
        draw_box(
            axis,
            aligned_prediction,
            color=PREDICTED_BOX_COLOR,
            linestyle="-",
            line_width=box_line_width,
            face_alpha=box_face_alpha,
        )

    axis.set_xlim(point_min[0] - padding[0], point_max[0] + padding[0])
    axis.set_ylim(point_min[1] - padding[1], point_max[1] + padding[1])
    axis.set_zlim(point_min[2] - padding[2], point_max[2] + padding[2])
    axis.set_box_aspect(tuple(span.tolist()), zoom=1.9)
    axis.set_proj_type("persp", focal_length=2.2)
    axis.view_init(elev=view_elev, azim=view_azim, roll=0.0)
    axis.set_axis_off()
    fig.savefig(output_path, dpi=200, facecolor="white", pad_inches=0)
    plt.close(fig)
    trim_white_margin(output_path)


def main() -> None:
    """加载点云、真实源物体框和预测放置框并导出 PNG。"""
    args = parse_args()
    if args.point_size <= 0.0 or args.box_line_width <= 0.0:
        raise ValueError("point_size and box_line_width must be positive")
    if not 0.0 <= args.box_face_alpha <= 1.0:
        raise ValueError("box_face_alpha must be within [0, 1]")

    sample_path = args.dataset_dir / "samples" / f"{args.sample_id}.json"
    scene = load_canonical_scene(sample_path, dataset_root=args.dataset_dir)
    prediction = load_prediction(args.predictions_json, args.sample_id, args.object_id)
    obj = find_scene_object(scene, args.object_id)
    source_corners = transform_points(
        get_bbox_corners(np.asarray(obj.bbox3d_canonical, dtype=np.float64)),
        np.asarray(obj.pose_world, dtype=np.float64),
    )
    predicted_corners = place_box_to_corners(
        np.asarray(prediction["place_box"], dtype=np.float64)
    )

    points, colors = load_ply(scene.point_cloud_path)
    render_mask = box_focus_mask(
        points,
        source_corners,
        predicted_corners,
        margin_cm=args.focus_margin_cm,
    )
    if args.support_mask is not None:
        support_points, support_colors = load_ply(args.support_mask)
        render_mask |= support_surface_mask(
            points,
            support_points,
            support_colors,
            tolerance_cm=args.support_surface_tolerance_cm,
        )

    save_visualization(
        scene,
        points[render_mask],
        colors[render_mask],
        points,
        source_corners,
        predicted_corners,
        args.output_path,
        point_size=args.point_size,
        view_elev=args.view_elev,
        view_azim=args.view_azim,
        box_line_width=args.box_line_width,
        box_face_alpha=args.box_face_alpha,
        draw_prediction=not args.source_only,
    )
    print(
        f"Saved {args.output_path} "
        f"(points={int(render_mask.sum())}, score={float(prediction.get('best_place_score', 0.0)):.4f})"
    )


if __name__ == "__main__":
    main()

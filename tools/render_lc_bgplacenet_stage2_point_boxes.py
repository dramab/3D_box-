#!/usr/bin/env python
"""在完整 RGB 稀疏体素点云上绘制原物体框或预测放置框。

原物体框示例:
    python tools/render_lc_bgplacenet_stage2_point_boxes.py \
        --dataset-dir data/hope \
        --sample-id hope__scene_0000__0005 \
        --object-id obj_3 \
        --predictions-json outputs/lc_bgplacenet_stage2_space_former_aligned_loss_48query_full_gt_guass_8_enriched/inference_custom_hope_scene_0000_0005/predictions.json \
        --output-path outputs/visualizations/hope__scene_0000__0005_rgb_pointcloud_source_box.png \
        --source-only

原物体框与放置位置框示例:
    python tools/render_lc_bgplacenet_stage2_point_boxes.py \
        --dataset-dir data/hope \
        --sample-id hope__scene_0000__0005 \
        --object-id obj_3 \
        --predictions-json outputs/lc_bgplacenet_stage2_space_former_aligned_loss_48query_full_gt_guass_8_enriched/inference_custom_hope_scene_0000_0005/predictions.json \
        --output-path outputs/visualizations/hope__scene_0000__0005_rgb_pointcloud_source_and_placement_boxes.png
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
from tools.export_canonical_sparse_voxel_vis import (
    CAMERA_VIEW_AZIM,
    camera_downward_elevation_degrees,
    camera_pose_aligned_points,
    draw_feathered_points,
    edge_fade_alpha,
    save_cropped_figure,
)
from tools.infer_lc_bgplacenet_stage2 import place_box_to_corners


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
    parser.add_argument("--output-path", type=Path, required=True, help="输出 PNG 路径。")
    parser.add_argument("--point-size", type=float, default=24.0, help="点云面积，单位为 pt^2。")
    parser.add_argument("--view-elev", type=float, default=35.0, help="目标向下俯仰角，单位为度。")
    parser.add_argument("--box-line-width", type=float, default=4.4, help="3D box 主线宽度。")
    parser.add_argument(
        "--box-face-alpha",
        type=float,
        default=0.12,
        help="3D box 面填充透明度。",
    )
    box_mode = parser.add_mutually_exclusive_group()
    box_mode.add_argument(
        "--source-only",
        action="store_true",
        help="只绘制原物体 3D box。",
    )
    box_mode.add_argument(
        "--prediction-only",
        action="store_true",
        help="只绘制预测放置位置 3D box。",
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
    source_corners: np.ndarray,
    predicted_corners: np.ndarray,
    output_path: Path,
    *,
    point_size: float,
    box_line_width: float,
    box_face_alpha: float,
    view_elev: float,
    draw_source: bool = True,
    draw_prediction: bool = True,
) -> Path:
    """以完整 RGB 体素点云为底图导出指定 3D box。"""
    view_center = 0.5 * (scene_points.min(axis=0) + scene_points.max(axis=0))
    aligned_points = camera_pose_aligned_points(scene_points, scene.camera, center=view_center)
    aligned_source = camera_pose_aligned_points(source_corners, scene.camera, center=view_center)
    aligned_prediction = camera_pose_aligned_points(
        predicted_corners, scene.camera, center=view_center
    )

    point_min = aligned_points.min(axis=0)
    point_max = aligned_points.max(axis=0)
    span = np.maximum(point_max - point_min, 1.0)
    padding = np.array([0.035, 0.035, 0.055], dtype=np.float64) * span
    relative_elev = float(view_elev) - camera_downward_elevation_degrees(scene.camera)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    pdf_path = output_path.with_suffix(".pdf")
    fig = plt.figure(figsize=(6.4, 4.0), facecolor="white")
    axis = fig.add_axes([0.0, 0.04, 1.0, 1.0], projection="3d", facecolor="white")
    draw_feathered_points(
        axis,
        aligned_points,
        scene_colors,
        edge_fade_alpha(aligned_points[:, :2], 0.0),
        point_size,
        is_3d=True,
    )
    if draw_source:
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
    axis.set_box_aspect(tuple(span.tolist()), zoom=1.65)
    axis.set_proj_type("persp", focal_length=2.2)
    # 点云和 3D box 已编码原相机方位与滚转，仅补偿目标俯仰角。
    axis.view_init(elev=relative_elev, azim=CAMERA_VIEW_AZIM, roll=0.0)
    axis.set_axis_off()
    save_cropped_figure(fig, output_path, pdf_path)
    plt.close(fig)
    return pdf_path


def main() -> None:
    """加载点云、真实源物体框和预测放置框并导出 PNG 与 PDF。"""
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

    if scene.voxel_point_cloud_path is None:
        raise ValueError(f"Sample {scene.sample_id} does not provide voxel_point_cloud_path")
    points, colors = load_ply(scene.voxel_point_cloud_path)

    pdf_path = save_visualization(
        scene,
        points,
        colors,
        source_corners,
        predicted_corners,
        args.output_path,
        point_size=args.point_size,
        box_line_width=args.box_line_width,
        box_face_alpha=args.box_face_alpha,
        view_elev=args.view_elev,
        draw_source=not args.prediction_only,
        draw_prediction=not args.source_only,
    )
    print(
        f"Saved {args.output_path} and {pdf_path} "
        f"(points={len(points)}, source_box={'on' if not args.prediction_only else 'off'}, "
        f"placement_box={'on' if not args.source_only else 'off'}, "
        f"view_elev={args.view_elev:.1f}, score={float(prediction.get('best_place_score', 0.0)):.4f})"
    )


if __name__ == "__main__":
    main()

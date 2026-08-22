#!/usr/bin/env python
"""
将 canonical 彩色点云按真实相机视角导出为论文插图。

使用示例:
    python tools/export_canonical_sparse_voxel_vis.py \
        --dataset-dir data/hope \
        --sample-id hope__scene_0000__0005 \
        --output-dir outputs/visualizations \
        --edge-feather-ratio 0.08 \
        --transparent-background
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.annotation.free_bbox.io_utils import load_ply
from src.datasets.canonical import load_canonical_scene


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(description="Export camera-aligned point cloud visualizations.")
    parser.add_argument("--dataset-dir", type=Path, required=True, help="canonical 数据集根目录。")
    parser.add_argument("--sample-id", required=True, help="要导出的 sample_id。")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/visualizations"),
        help="PNG 和 PDF 输出目录。",
    )
    parser.add_argument(
        "--point-source",
        choices=("voxel", "raw"),
        default="voxel",
        help="点云来源：voxel 为 1 cm 体素点云，raw 为固定 50000 点的初始点云。",
    )
    parser.add_argument(
        "--point-size",
        type=float,
        default=None,
        help="像素对齐图的散点面积，单位为 pt^2；默认 voxel=24，raw=2.5。",
    )
    parser.add_argument(
        "--oblique-point-size",
        type=float,
        default=None,
        help="斜俯视 3D 图的散点面积，单位为 pt^2；默认 voxel=36，raw=1.8。",
    )
    parser.add_argument(
        "--edge-feather-ratio",
        type=float,
        default=0.0,
        help="点云边缘羽化宽度占可视平面跨度的比例；0 表示关闭。",
    )
    parser.add_argument(
        "--transparent-background",
        action="store_true",
        help="保存透明背景 PNG，便于后续与其他背景融合。",
    )
    return parser.parse_args()


def project_visible_points(points: np.ndarray, camera) -> tuple[np.ndarray, np.ndarray]:
    """将 world 点投影到图像平面，并返回可见点索引和像素坐标。"""
    points_h = np.column_stack([points, np.ones(len(points), dtype=np.float64)])
    points_camera = points_h @ camera.E_w2c.T
    depth = points_camera[:, 2]
    uv = np.empty((len(points), 2), dtype=np.float64)
    uv[:, 0] = camera.fx * points_camera[:, 0] / depth + camera.cx
    uv[:, 1] = camera.fy * points_camera[:, 1] / depth + camera.cy

    visible = (
        (depth > 0.0)
        & (uv[:, 0] >= 0.0)
        & (uv[:, 0] < camera.img_w)
        & (uv[:, 1] >= 0.0)
        & (uv[:, 1] < camera.img_h)
    )
    # 远处点先绘制，近处点自然覆盖远处点，保持合理的遮挡关系。
    visible_indices = np.flatnonzero(visible)
    visible_indices = visible_indices[np.argsort(depth[visible_indices])[::-1]]
    return visible_indices, uv[visible_indices]


def load_points(scene, point_source: str) -> tuple[np.ndarray, np.ndarray]:
    """按指定来源加载样本点云。"""
    if point_source == "raw":
        return load_ply(scene.point_cloud_path)
    if scene.voxel_point_cloud_path is None:
        raise ValueError(f"Sample {scene.sample_id} does not provide voxel_point_cloud_path")
    return load_ply(scene.voxel_point_cloud_path)


def edge_fade_alpha(coordinates: np.ndarray, feather_ratio: float) -> np.ndarray:
    """根据点到可视平面边界的距离生成边缘羽化权重。"""
    coordinates = np.asarray(coordinates, dtype=np.float64)
    if coordinates.ndim != 2 or coordinates.shape[1] < 2:
        raise ValueError("coordinates must have shape (N, >=2)")
    feather_ratio = float(feather_ratio)
    if feather_ratio < 0.0:
        raise ValueError(f"edge_feather_ratio must be non-negative, got {feather_ratio}")
    if feather_ratio == 0.0:
        return np.ones(len(coordinates), dtype=np.float32)

    coordinate_min = coordinates[:, :2].min(axis=0)
    coordinate_max = coordinates[:, :2].max(axis=0)
    span = np.maximum(coordinate_max - coordinate_min, 1e-6)
    feather_width = np.maximum(span * feather_ratio, 1e-6)
    normalized_distance_to_edge = np.minimum(
        (coordinates[:, :2] - coordinate_min) / feather_width,
        (coordinate_max - coordinates[:, :2]) / feather_width,
    ).min(axis=1)
    return np.clip(normalized_distance_to_edge, 0.0, 1.0).astype(np.float32)


def draw_feathered_points(
    ax: plt.Axes,
    coordinates: np.ndarray,
    colors: np.ndarray,
    edge_weights: np.ndarray,
    point_size: float,
    *,
    is_3d: bool,
) -> None:
    """绘制主体点和低透明度边缘光晕，形成柔和的点云轮廓。"""
    colors_float = np.asarray(colors, dtype=np.float32) / 255.0
    edge_weights = np.asarray(edge_weights, dtype=np.float32)
    point_alpha = 0.18 + 0.82 * edge_weights
    halo_alpha = 0.18 * np.square(1.0 - edge_weights)
    point_rgba = np.column_stack([colors_float, point_alpha])
    halo_rgba = np.column_stack([colors_float, halo_alpha])

    if is_3d:
        ax.scatter(
            coordinates[:, 0],
            coordinates[:, 1],
            coordinates[:, 2],
            c=halo_rgba,
            s=point_size * 4.0,
            marker="o",
            linewidths=0.0,
            depthshade=False,
            antialiased=True,
        )
        ax.scatter(
            coordinates[:, 0],
            coordinates[:, 1],
            coordinates[:, 2],
            c=point_rgba,
            s=point_size,
            marker="o",
            linewidths=0.0,
            depthshade=False,
            antialiased=False,
        )
        return

    ax.scatter(
        coordinates[:, 0],
        coordinates[:, 1],
        c=halo_rgba,
        s=point_size * 4.0,
        marker="o",
        linewidths=0.0,
    )
    ax.scatter(
        coordinates[:, 0],
        coordinates[:, 1],
        c=point_rgba,
        s=point_size,
        marker="o",
        linewidths=0.0,
    )


def save_projected_point_visualization(
    scene,
    points: np.ndarray,
    colors: np.ndarray,
    output_dir: Path,
    point_size: float,
    output_label: str,
    edge_feather_ratio: float,
    transparent_background: bool,
) -> tuple[Path, Path]:
    """保存与 RGB 像素视角对齐的 PNG 预览和矢量 PDF。"""
    point_indices, uv = project_visible_points(points, scene.camera)
    if len(point_indices) == 0:
        raise ValueError(f"Sample {scene.sample_id} has no points visible to the camera")

    output_dir.mkdir(parents=True, exist_ok=True)
    output_stem = output_dir / f"{scene.sample_id}_{output_label}"
    png_path = output_stem.with_suffix(".png")
    pdf_path = output_stem.with_suffix(".pdf")

    dpi = 200
    figure_facecolor = "none" if transparent_background else "white"
    fig = plt.figure(
        figsize=(scene.camera.img_w / dpi, scene.camera.img_h / dpi),
        facecolor=figure_facecolor,
    )
    ax = fig.add_axes(
        [0.0, 0.0, 1.0, 1.0],
        facecolor=(1.0, 1.0, 1.0, 0.0) if transparent_background else "white",
    )
    edge_weights = edge_fade_alpha(uv, edge_feather_ratio)
    draw_feathered_points(
        ax,
        uv,
        colors[point_indices],
        edge_weights,
        point_size,
        is_3d=False,
    )
    ax.set_xlim(0.0, scene.camera.img_w)
    ax.set_ylim(scene.camera.img_h, 0.0)
    ax.set_aspect("equal", adjustable="box")
    ax.axis("off")

    fig.savefig(
        png_path,
        dpi=dpi,
        facecolor=figure_facecolor,
        transparent=transparent_background,
        pad_inches=0,
    )
    fig.savefig(pdf_path, facecolor=figure_facecolor, pad_inches=0)
    plt.close(fig)
    return png_path, pdf_path


def camera_yaw_aligned_points(
    points: np.ndarray,
    camera,
    *,
    center: np.ndarray | None = None,
) -> np.ndarray:
    """保留相机水平朝向，同时将 world-Z 固定为可视化竖直方向。"""
    camera_forward = np.asarray(camera.E_c2w[:3, 2], dtype=np.float64)
    forward_xy = camera_forward.copy()
    forward_xy[2] = 0.0
    forward_xy /= np.linalg.norm(forward_xy)

    world_up = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    camera_right = np.cross(forward_xy, world_up)
    basis = np.column_stack([camera_right, forward_xy, world_up])
    if center is None:
        center = 0.5 * (points.min(axis=0) + points.max(axis=0))
    return (points - np.asarray(center, dtype=np.float64)) @ basis


def save_oblique_point_visualization(
    scene,
    points: np.ndarray,
    colors: np.ndarray,
    output_dir: Path,
    point_size: float,
    output_label: str,
    edge_feather_ratio: float,
    transparent_background: bool,
) -> tuple[Path, Path]:
    """保存相机水平朝向约束、桌面校正的斜俯视 3D 论文图。"""
    aligned_points = camera_yaw_aligned_points(points, scene.camera)
    point_min = aligned_points.min(axis=0)
    point_max = aligned_points.max(axis=0)
    span = np.maximum(point_max - point_min, 1.0)
    padding = np.array([0.035, 0.035, 0.055]) * span

    output_dir.mkdir(parents=True, exist_ok=True)
    output_stem = output_dir / f"{scene.sample_id}_{output_label}_oblique_3d"
    png_path = output_stem.with_suffix(".png")
    pdf_path = output_stem.with_suffix(".pdf")

    figure_facecolor = "none" if transparent_background else "white"
    fig = plt.figure(figsize=(6.4, 4.0), facecolor=figure_facecolor)
    ax = fig.add_axes(
        [0.0, 0.04, 1.0, 1.0],
        projection="3d",
        facecolor=(1.0, 1.0, 1.0, 0.0) if transparent_background else "white",
    )
    if transparent_background:
        ax.xaxis.pane.set_alpha(0.0)
        ax.yaxis.pane.set_alpha(0.0)
        ax.zaxis.pane.set_alpha(0.0)
    edge_weights = edge_fade_alpha(aligned_points[:, :2], edge_feather_ratio)
    draw_feathered_points(
        ax,
        aligned_points,
        colors,
        edge_weights,
        point_size,
        is_3d=True,
    )
    ax.set_xlim(point_min[0] - padding[0], point_max[0] + padding[0])
    ax.set_ylim(point_min[1] - padding[1], point_max[1] + padding[1])
    ax.set_zlim(point_min[2] - padding[2], point_max[2] + padding[2])
    ax.set_box_aspect(tuple(span.tolist()), zoom=1.65)
    # 弱透视兼顾空间纵深与物体比例，固定视角保证不同样本之间可比较。
    ax.set_proj_type("persp", focal_length=2.2)
    ax.view_init(elev=25.0, azim=-72.0, roll=0.0)
    ax.set_axis_off()

    fig.savefig(
        png_path,
        dpi=200,
        facecolor=figure_facecolor,
        transparent=transparent_background,
        pad_inches=0,
    )
    fig.savefig(pdf_path, facecolor=figure_facecolor, pad_inches=0)
    plt.close(fig)
    return png_path, pdf_path


def main() -> None:
    """加载 canonical 样本并导出点云论文图。"""
    args = parse_args()
    sample_path = args.dataset_dir / "samples" / f"{args.sample_id}.json"
    scene = load_canonical_scene(sample_path, dataset_root=args.dataset_dir)
    points, colors = load_points(scene, args.point_source)
    if args.point_source == "raw":
        point_size = 2.5 if args.point_size is None else args.point_size
        oblique_point_size = 1.8 if args.oblique_point_size is None else args.oblique_point_size
        output_label = "raw_points"
    else:
        point_size = 24.0 if args.point_size is None else args.point_size
        oblique_point_size = 36.0 if args.oblique_point_size is None else args.oblique_point_size
        output_label = "sparse_voxels"
    if args.edge_feather_ratio > 0.0:
        output_label = f"{output_label}_feathered"
    output_paths = (
        *save_projected_point_visualization(
            scene,
            points,
            colors,
            args.output_dir,
            point_size,
            output_label,
            args.edge_feather_ratio,
            args.transparent_background,
        ),
        *save_oblique_point_visualization(
            scene,
            points,
            colors,
            args.output_dir,
            oblique_point_size,
            output_label,
            args.edge_feather_ratio,
            args.transparent_background,
        ),
    )
    for output_path in output_paths:
        print(f"Saved {output_path}")


if __name__ == "__main__":
    main()

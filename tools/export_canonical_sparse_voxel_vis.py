#!/usr/bin/env python
"""
将 canonical 稀疏体素点云按真实相机视角导出为论文插图。

使用示例:
    python tools/export_canonical_sparse_voxel_vis.py \
        --dataset-dir data/hope \
        --sample-id hope__scene_0000__0005 \
        --output-dir outputs/visualizations
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
    parser = argparse.ArgumentParser(description="Export camera-aligned sparse voxel visualizations.")
    parser.add_argument("--dataset-dir", type=Path, required=True, help="canonical 数据集根目录。")
    parser.add_argument("--sample-id", required=True, help="要导出的 sample_id。")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/visualizations"),
        help="PNG 和 PDF 输出目录。",
    )
    parser.add_argument("--point-size", type=float, default=24.0, help="散点面积，单位为 pt^2。")
    parser.add_argument(
        "--oblique-point-size",
        type=float,
        default=36.0,
        help="斜俯视 3D 图的散点面积，单位为 pt^2。",
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


def load_voxel_points(scene) -> tuple[np.ndarray, np.ndarray]:
    """加载样本的稀疏体素点云。"""
    if scene.voxel_point_cloud_path is None:
        raise ValueError(f"Sample {scene.sample_id} does not provide voxel_point_cloud_path")
    return load_ply(scene.voxel_point_cloud_path)


def save_sparse_voxel_visualization(
    scene,
    points: np.ndarray,
    colors: np.ndarray,
    output_dir: Path,
    point_size: float,
) -> tuple[Path, Path]:
    """保存与 RGB 像素视角对齐的 PNG 预览和矢量 PDF。"""
    point_indices, uv = project_visible_points(points, scene.camera)
    if len(point_indices) == 0:
        raise ValueError(f"Sample {scene.sample_id} has no voxel points visible to the camera")

    output_dir.mkdir(parents=True, exist_ok=True)
    output_stem = output_dir / f"{scene.sample_id}_sparse_voxels"
    png_path = output_stem.with_suffix(".png")
    pdf_path = output_stem.with_suffix(".pdf")

    dpi = 200
    fig = plt.figure(
        figsize=(scene.camera.img_w / dpi, scene.camera.img_h / dpi),
        facecolor="white",
    )
    ax = fig.add_axes([0.0, 0.0, 1.0, 1.0])
    ax.scatter(
        uv[:, 0],
        uv[:, 1],
        c=colors[point_indices].astype(np.float32) / 255.0,
        s=point_size,
        marker="o",
        linewidths=0.0,
    )
    ax.set_xlim(0.0, scene.camera.img_w)
    ax.set_ylim(scene.camera.img_h, 0.0)
    ax.set_aspect("equal", adjustable="box")
    ax.axis("off")

    fig.savefig(png_path, dpi=dpi, facecolor="white", pad_inches=0)
    fig.savefig(pdf_path, facecolor="white", pad_inches=0)
    plt.close(fig)
    return png_path, pdf_path


def camera_yaw_aligned_points(points: np.ndarray, camera) -> np.ndarray:
    """保留相机水平朝向，同时将 world-Z 固定为可视化竖直方向。"""
    camera_forward = np.asarray(camera.E_c2w[:3, 2], dtype=np.float64)
    forward_xy = camera_forward.copy()
    forward_xy[2] = 0.0
    forward_xy /= np.linalg.norm(forward_xy)

    world_up = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    camera_right = np.cross(forward_xy, world_up)
    basis = np.column_stack([camera_right, forward_xy, world_up])
    center = 0.5 * (points.min(axis=0) + points.max(axis=0))
    return (points - center) @ basis


def save_oblique_voxel_visualization(
    scene,
    points: np.ndarray,
    colors: np.ndarray,
    output_dir: Path,
    point_size: float,
) -> tuple[Path, Path]:
    """保存相机水平朝向约束、桌面校正的斜俯视 3D 论文图。"""
    aligned_points = camera_yaw_aligned_points(points, scene.camera)
    point_min = aligned_points.min(axis=0)
    point_max = aligned_points.max(axis=0)
    span = np.maximum(point_max - point_min, 1.0)
    padding = np.array([0.03, 0.03, 0.05]) * span

    output_dir.mkdir(parents=True, exist_ok=True)
    output_stem = output_dir / f"{scene.sample_id}_sparse_voxels_oblique_3d"
    png_path = output_stem.with_suffix(".png")
    pdf_path = output_stem.with_suffix(".pdf")

    fig = plt.figure(figsize=(6.4, 4.8), facecolor="white")
    ax = fig.add_axes([0.0, 0.0, 1.0, 1.0], projection="3d")
    ax.scatter(
        aligned_points[:, 0],
        aligned_points[:, 1],
        aligned_points[:, 2],
        c=colors.astype(np.float32) / 255.0,
        s=point_size,
        marker="o",
        linewidths=0.0,
        depthshade=True,
    )
    ax.set_xlim(point_min[0] - padding[0], point_max[0] + padding[0])
    ax.set_ylim(point_min[1] - padding[1], point_max[1] + padding[1])
    ax.set_zlim(point_min[2] - padding[2], point_max[2] + padding[2])
    ax.set_box_aspect(tuple(span.tolist()), zoom=2.0)
    ax.set_proj_type("persp", focal_length=1.0)
    ax.view_init(elev=28.0, azim=-90.0, roll=0.0)
    ax.set_axis_off()

    fig.savefig(png_path, dpi=200, facecolor="white", pad_inches=0)
    fig.savefig(pdf_path, facecolor="white", pad_inches=0)
    plt.close(fig)
    return png_path, pdf_path


def main() -> None:
    """加载 canonical 样本并导出稀疏体素论文图。"""
    args = parse_args()
    sample_path = args.dataset_dir / "samples" / f"{args.sample_id}.json"
    scene = load_canonical_scene(sample_path, dataset_root=args.dataset_dir)
    points, colors = load_voxel_points(scene)
    output_paths = (
        *save_sparse_voxel_visualization(scene, points, colors, args.output_dir, args.point_size),
        *save_oblique_voxel_visualization(scene, points, colors, args.output_dir, args.oblique_point_size),
    )
    for output_path in output_paths:
        print(f"Saved {output_path}")


if __name__ == "__main__":
    main()

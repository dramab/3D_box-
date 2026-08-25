#!/usr/bin/env python
"""Render a predicted P3 heatmap on the corresponding canonical point cloud.

使用示例:
    python tools/render_lc_bgplacenet_stage2_point_mask.py \
        --dataset-dir data/hope \
        --sample-id hope__scene_0000__0005 \
        --pred-heatmap outputs/lc_bgplacenet_stage2_space_former_aligned_loss_48query_full_gt_guass_8_enriched/inference_custom_hope_scene_0000_0005/pred_heatmaps/hope__custom_hope_scene_0000_0005_obj_3__prediction_only__hope__scene_0000__0005__obj_3__cluster_000__pred_heatmap.ply \
        --output-path outputs/visualizations/hope__scene_0000__0005_paper_single_pointcloud_v2_p3_heatmap.png

局部版保留完整支撑面:
    python tools/render_lc_bgplacenet_stage2_point_mask.py \
        --dataset-dir data/hope \
        --sample-id hope__scene_0000__0005 \
        --pred-heatmap <pred_heatmap.ply> \
        --support-mask outputs/free_bbox_hope/support_masks/hope__scene_0000__0005__support_mask.ply \
        --output-path outputs/visualizations/<output>.png \
        --focus-score-threshold 0.4 --focus-margin-cm 8.0 \
        --keep-support-surface --support-surface-tolerance-cm 0.75

绘制完整环境热力图，并单独导出红圈高概率区域的放大图:
    python tools/render_lc_bgplacenet_stage2_point_mask.py \
        --dataset-dir data/hope \
        --sample-id hope__scene_0000__0005 \
        --pred-heatmap <pred_heatmap.ply> \
        --support-mask outputs/free_bbox_hope/support_masks/hope__scene_0000__0005__support_mask.ply \
        --output-path outputs/visualizations/<output>.png \
        --heatmap-only
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
import numpy as np
from matplotlib.patches import Ellipse
from mpl_toolkits.mplot3d import proj3d
from PIL import Image
from PIL import JpegImagePlugin  # noqa: F401  # 注册 Pillow PDF 导出所需的 JPEG 编码器。

matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.annotation.free_bbox.io_utils import load_ply
from src.datasets.canonical import load_canonical_scene
from tools.export_canonical_sparse_voxel_vis import (
    CAMERA_VIEW_AZIM,
    CAMERA_VIEW_ELEV,
    camera_downward_elevation_degrees,
    camera_pose_aligned_points,
    edge_fade_alpha,
    trim_white_margin,
)


# The prediction PLY and the paper visualization use the same continuous
# blue-cyan-yellow-red palette.
PREDICTION_PALETTE = np.asarray(
    [[37, 99, 235], [6, 182, 212], [250, 204, 21], [220, 38, 38]],
    dtype=np.float32,
)
DISPLAY_PALETTE = PREDICTION_PALETTE.copy()
ZERO_RESPONSE_COLOR = np.asarray([12, 28, 105], dtype=np.float32) / 255.0
NON_SUPPORT_COLOR = np.asarray([107, 114, 128], dtype=np.float32) / 255.0
LOCAL_POINT_SIZE = 4.2
HEATMAP_ONLY_POINT_SIZE = 24.0
HEATMAP_ONLY_VIEW_ELEV = 35.0
HEATMAP_FOCUS_SCORE_RATIO = 0.75


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(description="Render a P3 heatmap on a canonical point cloud.")
    parser.add_argument("--dataset-dir", type=Path, required=True, help="canonical 数据集根目录。")
    parser.add_argument("--sample-id", required=True, help="canonical sample_id。")
    parser.add_argument("--pred-heatmap", type=Path, required=True, help="模型输出的 P3 heatmap PLY。")
    parser.add_argument("--output-path", type=Path, required=True, help="输出 PNG 路径。")
    parser.add_argument("--p3-stride-cm", type=float, default=4.0, help="P3 网格步长，单位为 cm。")
    parser.add_argument("--voxel-size-cm", type=float, default=1.0, help="输入点云体素大小，单位为 cm。")
    parser.add_argument(
        "--mask-alpha",
        type=float,
        default=0.65,
        help="所有点统一使用的热力颜色混合比例，范围为 [0, 1]。",
    )
    parser.add_argument(
        "--point-size",
        type=float,
        default=4.2,
        help="点云绘制大小；增大该值可以减少点间空缺。",
    )
    parser.add_argument(
        "--heatmap-only",
        action="store_true",
        help="用 P3 响应着色完整环境点阵，并以原相机方位圈选、放大高概率区域。",
    )
    parser.add_argument(
        "--heatmap-only-point-size",
        type=float,
        default=HEATMAP_ONLY_POINT_SIZE,
        help="heatmap-only 点面积，单位为 pt^2。",
    )
    parser.add_argument(
        "--heatmap-only-view-elev",
        type=float,
        default=HEATMAP_ONLY_VIEW_ELEV,
        help="heatmap-only 相对 world 水平面的目标俯仰角，单位为度。",
    )
    parser.add_argument(
        "--focus-score-threshold",
        type=float,
        default=None,
        help="只显示分数不低于该阈值的 P3 区域周围点云；不指定则显示完整场景。",
    )
    parser.add_argument(
        "--focus-xy-bounds",
        type=float,
        nargs=4,
        metavar=("X_MIN", "Y_MIN", "X_MAX", "Y_MAX"),
        default=None,
        help="按指定世界坐标 XY 范围筛选点云，用于和其他论文图统一构图。",
    )
    parser.add_argument(
        "--focus-margin-cm",
        type=float,
        default=8.0,
        help="局部热力区域在 XY 平面上的额外裁剪边界，单位为 cm。",
    )
    parser.add_argument(
        "--keep-support-surface",
        action="store_true",
        help="局部版额外保留完整支撑面高度附近的原始点云。",
    )
    parser.add_argument(
        "--support-mask",
        type=Path,
        default=None,
        help="free_bbox 输出的 support_mask PLY；用于确定支撑面高度。",
    )
    parser.add_argument(
        "--support-surface-tolerance-cm",
        type=float,
        default=1.25,
        help="保留支撑面时相对支撑面高度的容差，单位为 cm。",
    )
    return parser.parse_args()


def _validate_points(points: np.ndarray, name: str) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"{name} must have shape (N, 3)")
    if len(points) == 0:
        raise ValueError(f"{name} must not be empty")
    return points


def _palette_values() -> np.ndarray:
    """Sample the piecewise-linear prediction palette for robust color decoding."""
    norm = np.linspace(0.0, 1.0, 10001, dtype=np.float32)
    segment = np.minimum((norm * 3.0).astype(np.int64), 2)
    alpha = (norm * 3.0 - segment.astype(np.float32))[:, None]
    colors = PREDICTION_PALETTE[segment] * (1.0 - alpha)
    colors += PREDICTION_PALETTE[segment + 1] * alpha
    return np.rint(colors).astype(np.uint8)


def decode_heatmap_scores(colors: np.ndarray) -> np.ndarray:
    """Decode normalized scores from the palette written by Stage 2 inference."""
    colors = np.asarray(colors, dtype=np.uint8)
    if colors.ndim != 2 or colors.shape[1] != 3:
        raise ValueError("colors must have shape (N, 3)")
    palette = _palette_values().astype(np.int32)
    color_delta = colors[:, None, :].astype(np.int32) - palette[None, :, :]
    distances = (color_delta**2).sum(axis=2)
    return np.argmin(distances, axis=1).astype(np.float32) / 10000.0


def normalize_scores_by_max(scores: np.ndarray) -> np.ndarray:
    """Normalize one prediction using its own maximum response."""
    scores = np.asarray(scores, dtype=np.float32).clip(0.0)
    maximum = float(scores.max(initial=0.0))
    if maximum <= 0.0:
        return np.zeros_like(scores)
    return (scores / maximum).clip(0.0, 1.0)


def map_continuous_scores(
    scene_points: np.ndarray,
    heatmap_points: np.ndarray,
    heatmap_scores: np.ndarray,
    *,
    p3_stride_cm: float = 4.0,
    origin: np.ndarray | None = None,
) -> np.ndarray:
    """Map each P3 score to scene points using the model's floor-grid rule."""
    scene_points = _validate_points(scene_points, "scene_points")
    heatmap_points = _validate_points(heatmap_points, "heatmap_points")
    heatmap_scores = np.asarray(heatmap_scores, dtype=np.float32)
    if heatmap_scores.shape != (len(heatmap_points),):
        raise ValueError("heatmap_scores must align with heatmap_points")
    p3_stride_cm = float(p3_stride_cm)
    if p3_stride_cm <= 0.0:
        raise ValueError("p3_stride_cm must be positive")

    if origin is None:
        # This fallback keeps the pure helper convenient for tests and small
        # examples. The CLI supplies the exact Stage 2 voxel origin.
        origin = heatmap_points.min(axis=0) - 0.5 * p3_stride_cm
    origin = np.asarray(origin, dtype=np.float64)
    scene_keys = np.floor((scene_points - origin) / p3_stride_cm).astype(np.int64)
    heatmap_keys = np.floor((heatmap_points - origin) / p3_stride_cm).astype(np.int64)

    scores_by_key = {tuple(key): float(score) for key, score in zip(heatmap_keys, heatmap_scores)}
    point_scores = np.zeros(len(scene_points), dtype=np.float32)
    for index, key in enumerate(scene_keys):
        point_scores[index] = scores_by_key.get(tuple(key), 0.0)
    return point_scores


def heatmap_colors(scores: np.ndarray) -> np.ndarray:
    """Return continuous blue-cyan-yellow-red colors for normalized scores."""
    scores = np.asarray(scores, dtype=np.float32).clip(0.0, 1.0)
    segment_count = len(DISPLAY_PALETTE) - 1
    scaled = scores * float(segment_count)
    segment = np.minimum(scaled.astype(np.int64), segment_count - 1)
    alpha = (scaled - segment.astype(np.float32))[:, None]
    colors = DISPLAY_PALETTE[segment] * (1.0 - alpha)
    colors += DISPLAY_PALETTE[segment + 1] * alpha
    return colors.clip(0.0, 255.0) / 255.0


def blend_continuous_mask(
    base_colors: np.ndarray,
    point_scores: np.ndarray,
    *,
    mask_alpha: float = 0.78,
    mask_zero_response: bool = False,
) -> np.ndarray:
    """Blend heatmap colors onto RGB colors with optional zero-response masking."""
    base_colors = np.asarray(base_colors, dtype=np.float32)
    if base_colors.ndim != 2 or base_colors.shape[1] != 3:
        raise ValueError("base_colors must have shape (N, 3)")
    point_scores = np.asarray(point_scores, dtype=np.float32).clip(0.0, 1.0)
    if point_scores.shape != (len(base_colors),):
        raise ValueError("point_scores must align with base_colors")
    if base_colors.max(initial=0.0) > 1.0:
        base_colors = base_colors / 255.0
    mask_alpha = float(mask_alpha)
    if not 0.0 <= mask_alpha <= 1.0:
        raise ValueError("mask_alpha must be within [0, 1]")

    response = point_scores > 0.0
    alpha = np.full((len(point_scores), 1), mask_alpha, dtype=np.float32)
    blended = base_colors.copy()
    if mask_zero_response:
        overlay_colors = heatmap_colors(point_scores)
        overlay_colors[~response] = ZERO_RESPONSE_COLOR
        blended = base_colors * (1.0 - alpha) + overlay_colors * alpha
    else:
        blended[response] = (
            base_colors[response] * (1.0 - alpha[response])
            + heatmap_colors(point_scores[response]) * alpha[response]
        )
    return blended.clip(0.0, 1.0)


def continuous_mask_point_sizes(
    point_scores: np.ndarray,
    *,
    base_size: float = LOCAL_POINT_SIZE,
) -> np.ndarray:
    """Slightly enlarge responsive points so the mask remains visible in 3D."""
    point_scores = np.asarray(point_scores, dtype=np.float32).clip(0.0, 1.0)
    base_size = float(base_size)
    if base_size <= 0.0:
        raise ValueError("base_size must be positive")
    return base_size * (1.0 + 1.2 * point_scores)


def local_crop_mask(
    points: np.ndarray,
    point_scores: np.ndarray,
    source_corners: np.ndarray,
    prediction_corners: np.ndarray,
    *,
    margin_cm: float = 4.0,
) -> np.ndarray:
    """Keep the source/prediction neighborhood for optional close-up views."""
    points = _validate_points(points, "points")
    point_scores = np.asarray(point_scores, dtype=np.float32)
    if point_scores.shape != (len(points),):
        raise ValueError("point_scores must align with points")
    corners = np.vstack([_validate_points(source_corners, "source_corners"), _validate_points(prediction_corners, "prediction_corners")])
    margin_cm = float(margin_cm)
    if margin_cm < 0.0:
        raise ValueError("margin_cm must be non-negative")
    lower = corners.min(axis=0) - margin_cm
    upper = corners.max(axis=0) + margin_cm
    return ((points >= lower) & (points <= upper)).all(axis=1)


def heatmap_focus_mask(
    scene_points: np.ndarray,
    heatmap_points: np.ndarray,
    heatmap_scores: np.ndarray,
    *,
    score_threshold: float,
    margin_cm: float,
) -> np.ndarray:
    """Keep the scene neighborhood around the selected high-response P3 cells."""
    scene_points = _validate_points(scene_points, "scene_points")
    heatmap_points = _validate_points(heatmap_points, "heatmap_points")
    heatmap_scores = np.asarray(heatmap_scores, dtype=np.float32)
    if heatmap_scores.shape != (len(heatmap_points),):
        raise ValueError("heatmap_scores must align with heatmap_points")
    score_threshold = float(score_threshold)
    margin_cm = float(margin_cm)
    if not 0.0 <= score_threshold <= 1.0:
        raise ValueError("score_threshold must be within [0, 1]")
    if margin_cm < 0.0:
        raise ValueError("margin_cm must be non-negative")

    selected = heatmap_scores >= score_threshold
    if not np.any(selected):
        selected[np.argmax(heatmap_scores)] = True
    lower = heatmap_points[selected, :2].min(axis=0) - margin_cm
    upper = heatmap_points[selected, :2].max(axis=0) + margin_cm
    return ((scene_points[:, :2] >= lower) & (scene_points[:, :2] <= upper)).all(axis=1)


def xy_focus_mask(scene_points: np.ndarray, bounds: np.ndarray) -> np.ndarray:
    """按明确的世界坐标 XY 边界筛选点云。"""
    scene_points = _validate_points(scene_points, "scene_points")
    bounds = np.asarray(bounds, dtype=np.float64)
    if bounds.shape != (4,):
        raise ValueError("bounds must contain x_min, y_min, x_max, y_max")
    lower = bounds[:2]
    upper = bounds[2:]
    if np.any(upper < lower):
        raise ValueError("focus XY upper bounds must not be smaller than lower bounds")
    return ((scene_points[:, :2] >= lower) & (scene_points[:, :2] <= upper)).all(axis=1)


def support_surface_mask(
    scene_points: np.ndarray,
    support_mask_points: np.ndarray,
    support_mask_colors: np.ndarray,
    *,
    tolerance_cm: float = 1.25,
) -> np.ndarray:
    """从 support mask 的白色 active 点估计支撑面，并保留完整高度层。"""
    scene_points = _validate_points(scene_points, "scene_points")
    support_mask_points = _validate_points(support_mask_points, "support_mask_points")
    support_mask_colors = np.asarray(support_mask_colors, dtype=np.uint8)
    if support_mask_colors.shape != support_mask_points.shape:
        raise ValueError("support_mask_colors must align with support_mask_points")
    tolerance_cm = float(tolerance_cm)
    if tolerance_cm < 0.0:
        raise ValueError("tolerance_cm must be non-negative")

    active = np.all(support_mask_colors == 255, axis=1)
    if not np.any(active):
        raise ValueError("support mask does not contain white active support points")
    support_z = float(np.median(support_mask_points[active, 2]))
    return np.abs(scene_points[:, 2] - support_z) <= tolerance_cm


def save_heatmap_only_visualization(
    scene,
    environment_points: np.ndarray,
    point_scores: np.ndarray,
    support_point_mask: np.ndarray,
    output_path: Path,
    *,
    point_size: float = HEATMAP_ONLY_POINT_SIZE,
    view_elev: float = HEATMAP_ONLY_VIEW_ELEV,
) -> Path:
    """保存带红圈的完整热力图，并独立导出圆内放大图。"""
    environment_points = _validate_points(environment_points, "environment_points")
    point_scores = np.asarray(point_scores, dtype=np.float32)
    if point_scores.shape != (len(environment_points),):
        raise ValueError("point_scores must align with environment_points")
    support_point_mask = np.asarray(support_point_mask, dtype=bool)
    if support_point_mask.shape != (len(environment_points),):
        raise ValueError("support_point_mask must align with environment_points")
    if point_size <= 0.0:
        raise ValueError("point_size must be positive")
    view_center = 0.5 * (environment_points.min(axis=0) + environment_points.max(axis=0))
    aligned_points = camera_pose_aligned_points(
        environment_points, scene.camera, center=view_center
    )
    point_min = aligned_points.min(axis=0)
    point_max = aligned_points.max(axis=0)
    span = np.maximum(point_max - point_min, 1.0)
    padding = np.array([0.05, 0.05, 0.04], dtype=np.float64) * span
    relative_elev = float(view_elev) - camera_downward_elevation_degrees(scene.camera)
    point_colors = heatmap_colors(point_scores)
    point_colors[~support_point_mask] = NON_SUPPORT_COLOR

    output_path.parent.mkdir(parents=True, exist_ok=True)
    dpi = 200
    fig = plt.figure(figsize=(1556 / dpi, 1011 / dpi), facecolor="white")
    ax = fig.add_axes([0.0, 0.0, 1.0, 1.0], projection="3d", facecolor="white")
    ax.scatter(
        aligned_points[:, 0],
        aligned_points[:, 1],
        aligned_points[:, 2],
        c=point_colors,
        s=point_size,
        marker="o",
        linewidths=0.0,
        depthshade=False,
        antialiased=True,
    )
    ax.set_xlim(point_min[0] - padding[0], point_max[0] + padding[0])
    ax.set_ylim(point_min[1] - padding[1], point_max[1] + padding[1])
    ax.set_zlim(point_min[2] - padding[2], point_max[2] + padding[2])
    ax.set_box_aspect(tuple(span.tolist()), zoom=1.55)
    ax.set_proj_type("persp", focal_length=2.2)
    ax.view_init(elev=relative_elev, azim=CAMERA_VIEW_AZIM, roll=0.0)
    ax.set_axis_off()

    # 以高分点为焦点，在主图上圈选并独立导出圆内放大图。
    maximum_score = float(point_scores[support_point_mask].max(initial=0.0))
    focus_mask = support_point_mask & (
        point_scores >= HEATMAP_FOCUS_SCORE_RATIO * maximum_score
    )
    if maximum_score <= 0.0 or not np.any(focus_mask):
        focus_mask[np.argmax(point_scores)] = True
    fig.canvas.draw()
    focus_projected = np.column_stack(
        proj3d.proj_transform(*aligned_points[focus_mask].T, ax.get_proj())[:2]
    )
    focus_pixels = ax.transData.transform(focus_projected)
    circle_center_pixels = focus_pixels.mean(axis=0)
    circle_radius_pixels = max(
        24.0,
        float(np.linalg.norm(focus_pixels - circle_center_pixels, axis=1).max())
        + 10.0,
    )
    # 直接裁剪主图圆内像素，确保放大图不混入圈外点或改变原透视关系。
    canvas = np.asarray(fig.canvas.buffer_rgba()).copy()
    canvas_height, canvas_width = canvas.shape[:2]
    center_x, center_y = circle_center_pixels
    radius = int(np.ceil(circle_radius_pixels))
    left = max(int(np.floor(center_x)) - radius, 0)
    right = min(int(np.ceil(center_x)) + radius + 1, canvas_width)
    top = max(canvas_height - int(np.ceil(center_y)) - radius, 0)
    bottom = min(canvas_height - int(np.floor(center_y)) + radius + 1, canvas_height)
    inset_image = canvas[top:bottom, left:right, :3]
    grid_y, grid_x = np.ogrid[: inset_image.shape[0], : inset_image.shape[1]]
    crop_center_x = center_x - left
    crop_center_y = canvas_height - center_y - top
    outside_circle = (
        (grid_x - crop_center_x) ** 2 + (grid_y - crop_center_y) ** 2
        > circle_radius_pixels**2
    )
    inset_image[outside_circle] = 255

    zoom_output_path = output_path.with_name(
        f"{output_path.stem}_zoom{output_path.suffix}"
    )
    zoom_image = Image.fromarray(inset_image).resize(
        (800, 800), resample=Image.Resampling.LANCZOS
    )
    zoom_image.save(zoom_output_path)
    zoom_pdf_path = zoom_output_path.with_suffix(".pdf")
    zoom_image.convert("RGB").save(zoom_pdf_path, resolution=200.0)

    circle_center = fig.transFigure.inverted().transform(circle_center_pixels)
    circle_width = 2.0 * circle_radius_pixels / fig.bbox.width
    circle_height = 2.0 * circle_radius_pixels / fig.bbox.height
    accent_color = "#dc2626"
    fig.add_artist(
        Ellipse(
            circle_center,
            width=circle_width,
            height=circle_height,
            transform=fig.transFigure,
            fill=False,
            edgecolor=accent_color,
            linewidth=2.4,
            zorder=20,
        )
    )
    fig.savefig(output_path, dpi=dpi, facecolor="white", pad_inches=0)
    plt.close(fig)

    trim_white_margin(output_path, margin_px=8)
    pdf_path = output_path.with_suffix(".pdf")
    with Image.open(output_path) as image:
        image.convert("RGB").save(pdf_path, resolution=200.0)
    return pdf_path


def save_heatmap_visualization(
    scene,
    scene_points: np.ndarray,
    scene_colors: np.ndarray,
    point_scores: np.ndarray,
    output_path: Path,
    *,
    mask_alpha: float,
    compact: bool = False,
    view_points: np.ndarray | None = None,
    point_size: float = LOCAL_POINT_SIZE,
) -> None:
    """Save the heatmap using the existing oblique point-cloud figure style."""
    if view_points is None:
        view_points = scene_points
    view_center = 0.5 * (view_points.min(axis=0) + view_points.max(axis=0))
    aligned_points = camera_pose_aligned_points(scene_points, scene.camera, center=view_center)
    aligned_view_points = camera_pose_aligned_points(view_points, scene.camera, center=view_center)
    point_min = aligned_view_points.min(axis=0)
    point_max = aligned_view_points.max(axis=0)
    span = np.maximum(point_max - point_min, 1.0)
    padding_ratio = 0.035
    padding = np.array([padding_ratio, padding_ratio, 0.02 if compact else 0.055]) * span
    edge_weights = np.ones(len(aligned_points), dtype=np.float32) if compact else edge_fade_alpha(aligned_points[:, :2], 0.08)
    colors = blend_continuous_mask(
        scene_colors,
        point_scores,
        mask_alpha=mask_alpha,
        mask_zero_response=True,
    )
    sizes = (
        np.full(len(point_scores), point_size)
        if compact
        else continuous_mask_point_sizes(point_scores, base_size=point_size)
    )
    point_alpha = np.ones(len(point_scores), dtype=np.float32) if compact else 0.18 + 0.82 * edge_weights
    rgba = np.column_stack([colors, point_alpha])
    halo_rgba = np.column_stack([colors, 0.18 * np.square(1.0 - edge_weights)])

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(1556 / 200.0, 1011 / 200.0), facecolor="white")
    ax = fig.add_axes([0.0, 0.04, 1.0, 1.0], projection="3d", facecolor="white")
    if not compact:
        ax.scatter(
            aligned_points[:, 0],
            aligned_points[:, 1],
            aligned_points[:, 2],
            c=halo_rgba,
            s=sizes * 4.0,
            marker="o",
            linewidths=0.0,
            depthshade=False,
            antialiased=True,
        )
    else:
        # 局部论文图用低透明度同色底层填补采样间的小白缝，前景点仍保留清晰颗粒感。
        compact_halo = np.column_stack(
            [colors, np.full(len(colors), 0.24, dtype=np.float32)]
        )
        ax.scatter(
            aligned_points[:, 0],
            aligned_points[:, 1],
            aligned_points[:, 2],
            c=compact_halo,
            s=sizes * 3.0,
            marker="o",
            linewidths=0.0,
            depthshade=False,
            antialiased=True,
        )
    ax.scatter(
        aligned_points[:, 0],
        aligned_points[:, 1],
        aligned_points[:, 2],
        c=rgba,
        s=sizes,
        marker="o",
        linewidths=0.0,
        depthshade=False,
        antialiased=False,
    )
    ax.set_xlim(point_min[0] - padding[0], point_max[0] + padding[0])
    ax.set_ylim(point_min[1] - padding[1], point_max[1] + padding[1])
    ax.set_zlim(point_min[2] - padding[2], point_max[2] + padding[2])
    ax.set_box_aspect(tuple(span.tolist()), zoom=1.9 if compact else 2.0)
    ax.set_proj_type("persp", focal_length=2.2)
    # 点云已经应用原始相机的完整姿态，不再叠加任何手工角度。
    ax.view_init(elev=CAMERA_VIEW_ELEV, azim=CAMERA_VIEW_AZIM, roll=0.0)
    ax.set_axis_off()
    fig.savefig(output_path, dpi=200, facecolor="white", pad_inches=0)
    plt.close(fig)
    if compact:
        trim_white_margin(output_path)


def main() -> None:
    """加载样本、映射 P3 热力分数并导出 PNG。"""
    args = parse_args()
    if args.voxel_size_cm <= 0.0:
        raise ValueError("voxel_size_cm must be positive")
    sample_path = args.dataset_dir / "samples" / f"{args.sample_id}.json"
    scene = load_canonical_scene(sample_path, dataset_root=args.dataset_dir)
    heatmap_points, heatmap_colors_data = load_ply(args.pred_heatmap)
    heatmap_scores = normalize_scores_by_max(decode_heatmap_scores(heatmap_colors_data))
    voxel_points, _ = load_ply(scene.voxel_point_cloud_path)
    origin = np.floor(voxel_points.min(axis=0) / args.voxel_size_cm) * args.voxel_size_cm
    if args.heatmap_only:
        if args.support_mask is None:
            raise ValueError("--heatmap-only requires --support-mask")
        environment_scores = map_continuous_scores(
            voxel_points,
            heatmap_points,
            heatmap_scores,
            p3_stride_cm=args.p3_stride_cm,
            origin=origin,
        )
        support_points, support_colors = load_ply(args.support_mask)
        environment_support_mask = support_surface_mask(
            voxel_points,
            support_points,
            support_colors,
            tolerance_cm=args.support_surface_tolerance_cm,
        )
        pdf_path = save_heatmap_only_visualization(
            scene,
            voxel_points,
            environment_scores,
            environment_support_mask,
            args.output_path,
            point_size=args.heatmap_only_point_size,
            view_elev=args.heatmap_only_view_elev,
        )
        zoom_path = args.output_path.with_name(
            f"{args.output_path.stem}_zoom{args.output_path.suffix}"
        )
        print(
            f"Saved {args.output_path}, {pdf_path}, {zoom_path}, and {zoom_path.with_suffix('.pdf')} "
            f"(environment_points={len(voxel_points)}, point_size={args.heatmap_only_point_size:.1f}, "
            f"support_points={int(environment_support_mask.sum())}, "
            f"view_elev={args.heatmap_only_view_elev:.1f}, heatmap_only=on)"
        )
        return

    scene_points, scene_colors = load_ply(scene.point_cloud_path)
    point_scores = map_continuous_scores(
        scene_points,
        heatmap_points,
        heatmap_scores,
        p3_stride_cm=args.p3_stride_cm,
        origin=origin,
    )
    render_mask = np.ones(len(scene_points), dtype=bool)
    if args.focus_score_threshold is not None and args.focus_xy_bounds is not None:
        raise ValueError("Use only one of --focus-score-threshold and --focus-xy-bounds")
    compact = args.focus_score_threshold is not None or args.focus_xy_bounds is not None
    if args.focus_xy_bounds is not None:
        render_mask = xy_focus_mask(scene_points, args.focus_xy_bounds)
    elif args.focus_score_threshold is not None:
        render_mask = heatmap_focus_mask(
            scene_points,
            heatmap_points,
            heatmap_scores,
            score_threshold=args.focus_score_threshold,
            margin_cm=args.focus_margin_cm,
        )
    if args.keep_support_surface:
        if args.support_mask is None:
            raise ValueError("--keep-support-surface requires --support-mask")
        support_points, support_colors = load_ply(args.support_mask)
        render_mask |= support_surface_mask(
            scene_points,
            support_points,
            support_colors,
            tolerance_cm=args.support_surface_tolerance_cm,
        )
    view_points = scene_points
    render_points = scene_points[render_mask]
    render_colors = scene_colors[render_mask]
    render_scores = point_scores[render_mask]
    save_heatmap_visualization(
        scene,
        render_points,
        render_colors,
        render_scores,
        args.output_path,
        mask_alpha=args.mask_alpha,
        compact=compact,
        view_points=view_points,
        point_size=args.point_size,
    )
    print(
        f"Saved {args.output_path} "
        f"(points={len(render_points)}, camera_pose=original, "
        f"support_surface={'on' if args.keep_support_surface else 'off'})"
    )


if __name__ == "__main__":
    main()

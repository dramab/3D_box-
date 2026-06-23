"""
src/annotation/free_bbox/visualize.py
-------------------------------------
free_bbox 单个最优框可视化。

参考旧 free_bbox 的双面板可视化风格：左侧 RGB 投影，右侧 3D 世界视图。
每张图片只对应一个最终 freebox。
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors

from src.annotation.free_bbox.geometry import get_bbox_corners, project_world, transform_points
from src.annotation.free_bbox.occupancy import FREE, OCCUPIED
from src.annotation.free_bbox.voxel_utils import voxel_to_world, world_to_voxel


CLR_BG = "#1A1A2E"
CLR_PANEL = "#0D0D1A"
CLR_TEXT = "#FFFFFF"
CLR_MUTED = "#D8DEEF"
CLR_GRID = "#444466"
CLR_ORIG = "#FF6D00"
CLR_PLACE = "#00E676"
CLR_CAM = "#FFEB3B"
CLR_ARROW = "#FFD54F"
CLR_OCC = "#78909C"
CLR_SURFACE = "#D7E2EE"
CLR_FREE = "#4CAF50"

BOX_EDGES = (
    (0, 1),
    (2, 3),
    (4, 5),
    (6, 7),
    (0, 2),
    (1, 3),
    (4, 6),
    (5, 7),
    (0, 4),
    (1, 5),
    (2, 6),
    (3, 7),
)


def _draw_bbox_2d(ax, corners_world, K, E_w2c, color, lw=2.0, label=None, alpha=1.0):
    """在 RGB 图像上绘制 3D bbox 投影。"""
    uv, z_cam = project_world(corners_world, K, E_w2c)
    drawn = False
    for i, j in BOX_EDGES:
        if z_cam[i] <= 0.0 or z_cam[j] <= 0.0:
            continue
        edge_label = label if label and not drawn else None
        ax.plot(
            [uv[i, 0], uv[j, 0]],
            [uv[i, 1], uv[j, 1]],
            color=color,
            lw=lw,
            alpha=alpha,
            label=edge_label,
        )
        drawn = True


def _draw_bbox_3d(ax, corners_world, color, lw=1.5, label=None, alpha=1.0):
    """在 3D 坐标轴上绘制 bbox 线框。"""
    for edge_idx, (i, j) in enumerate(BOX_EDGES):
        edge_label = label if edge_idx == 0 else None
        ax.plot(
            [corners_world[i, 0], corners_world[j, 0]],
            [corners_world[i, 1], corners_world[j, 1]],
            [corners_world[i, 2], corners_world[j, 2]],
            color=color,
            lw=lw,
            alpha=alpha,
            label=edge_label,
        )


def _style_3d_axes(ax) -> None:
    """设置 3D 右面板暗色主题。"""
    ax.set_facecolor(CLR_PANEL)
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.pane.fill = False
        axis.pane.set_edgecolor(mcolors.to_rgba("#333355", 0.95))
        axis._axinfo["grid"]["color"] = mcolors.to_rgba(CLR_GRID, 0.35)
        axis._axinfo["grid"]["linewidth"] = 0.8
        try:
            axis.line.set_color(mcolors.to_rgba(CLR_MUTED, 0.6))
        except Exception:
            pass
    ax.tick_params(colors=CLR_TEXT, labelsize=8, pad=2)
    ax.grid(True)


def _surface_mask_to_world(surface_mask_2d, support_z, vp):
    """将支撑面二维 mask 转为世界坐标点。"""
    if surface_mask_2d is None or not np.any(surface_mask_2d):
        return np.empty((0, 3), dtype=np.float64)
    support_xy = np.argwhere(surface_mask_2d)
    support_idx = np.column_stack(
        [support_xy, np.full(len(support_xy), int(support_z), dtype=np.intp)]
    )
    return voxel_to_world(support_idx, vp)


def _compute_view_limits(support_world, focus_points, voxel_size):
    """计算以支撑面和目标框为核心的 3D 视图范围。"""
    if len(support_world) > 0:
        support_min = support_world.min(axis=0)
        support_max = support_world.max(axis=0)
        span_xy = np.maximum(support_max[:2] - support_min[:2], float(voxel_size) * 6.0)
        pad_xy = np.maximum(span_xy * 0.18, float(voxel_size) * 2.0)
        z_bottom = float(support_min[2] - 0.75 * voxel_size)
        z_height = max(float(span_xy.max()) * 0.95, float(voxel_size) * 10.0)
        view_min = np.array(
            [support_min[0] - pad_xy[0], support_min[1] - pad_xy[1], z_bottom],
            dtype=np.float64,
        )
        view_max = np.array(
            [support_max[0] + pad_xy[0], support_max[1] + pad_xy[1], z_bottom + z_height],
            dtype=np.float64,
        )
        focus = np.concatenate(focus_points, axis=0)
        view_min[:2] = np.minimum(view_min[:2], focus.min(axis=0)[:2] - 2.0 * voxel_size)
        view_max[:2] = np.maximum(view_max[:2], focus.max(axis=0)[:2] + 2.0 * voxel_size)
        view_max[2] = max(float(view_max[2]), float(focus.max(axis=0)[2] + 2.5 * voxel_size))
        return view_min, view_max

    focus = np.concatenate(focus_points, axis=0)
    focus_min = focus.min(axis=0)
    focus_max = focus.max(axis=0)
    span = np.maximum(focus_max - focus_min, float(voxel_size) * 6.0)
    pad = np.maximum(span * np.array([0.14, 0.14, 0.18]), float(voxel_size) * 2.0)
    return focus_min - pad, focus_max + pad


def _filter_indices_to_view(indices, view_min, view_max, vp):
    """只保留可视窗口内的体素索引。"""
    if len(indices) == 0:
        return indices
    idx_min = world_to_voxel(np.asarray(view_min, dtype=np.float64)[None, :], vp)[0] - 1
    idx_max = world_to_voxel(np.asarray(view_max, dtype=np.float64)[None, :], vp)[0] + 1
    keep = (
        (indices[:, 0] >= idx_min[0])
        & (indices[:, 0] <= idx_max[0])
        & (indices[:, 1] >= idx_min[1])
        & (indices[:, 1] <= idx_max[1])
        & (indices[:, 2] >= idx_min[2])
        & (indices[:, 2] <= idx_max[2])
    )
    return indices[keep]


def save_freebox_visualization(
    rgb: np.ndarray,
    obj_name: str,
    bbox3d: np.ndarray,
    T_obj2world: np.ndarray,
    placed_corners_world: np.ndarray,
    K: np.ndarray,
    E_w2c: np.ndarray,
    vp: dict,
    cam_origin: np.ndarray,
    grid: np.ndarray,
    out_path: str | Path,
    surface_mask_2d: np.ndarray | None,
    support_z: int,
    placement: dict,
) -> None:
    """
    保存单个 freebox 的可视化图片。

    左面板显示 RGB 上原始框和当前 freebox 投影，右面板显示支撑面、局部体素、
    原始框、freebox 以及位移箭头。
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    voxel_size = float(vp["voxel_size"])
    corners_obj = get_bbox_corners(bbox3d)
    orig_world = transform_points(corners_obj, T_obj2world)
    placed_world = np.asarray(placed_corners_world, dtype=np.float64)
    orig_center = orig_world.mean(axis=0)
    placed_center = placed_world.mean(axis=0)
    displacement = placed_center - orig_center

    support_world = _surface_mask_to_world(surface_mask_2d, support_z, vp)
    view_min, view_max = _compute_view_limits(support_world, [orig_world, placed_world], voxel_size)
    view_span = np.maximum(view_max - view_min, voxel_size)

    rng = np.random.default_rng(42)
    occ_idx = _filter_indices_to_view(np.argwhere(grid == OCCUPIED), view_min, view_max, vp)
    if len(occ_idx) > 6000:
        occ_idx = occ_idx[rng.choice(len(occ_idx), 6000, replace=False)]
    occ_world = voxel_to_world(occ_idx, vp) if len(occ_idx) else np.empty((0, 3), dtype=np.float64)

    free_idx = _filter_indices_to_view(np.argwhere(grid == FREE), view_min, view_max, vp)
    if len(free_idx) > 2200:
        free_idx = free_idx[rng.choice(len(free_idx), 2200, replace=False)]
    free_world = voxel_to_world(free_idx, vp) if len(free_idx) else np.empty((0, 3), dtype=np.float64)

    if len(support_world) > 3600:
        support_world = support_world[rng.choice(len(support_world), 3600, replace=False)]

    fig = plt.figure(figsize=(18, 7.4))
    fig.patch.set_facecolor(CLR_BG)

    ax_rgb = fig.add_axes([0.02, 0.10, 0.47, 0.82])
    ax_rgb.imshow(np.asarray(rgb, dtype=np.uint8))
    _draw_bbox_2d(ax_rgb, orig_world, K, E_w2c, CLR_ORIG, lw=2.5, label=f"Original: {obj_name}")
    _draw_bbox_2d(ax_rgb, placed_world, K, E_w2c, CLR_PLACE, lw=2.2, label="Yaw-only freebox", alpha=0.85)
    img_h, img_w = rgb.shape[:2]
    ax_rgb.set_xlim(0, img_w)
    ax_rgb.set_ylim(img_h, 0)
    ax_rgb.set_title(
        "RGB Image - 3-D Bbox Projection\nOrange = current position | Green = yaw-only freebox",
        color=CLR_TEXT,
        fontsize=11,
        pad=8,
    )
    ax_rgb.axis("off")
    ax_rgb.legend(
        loc="upper right",
        fontsize=8,
        facecolor=CLR_PANEL,
        labelcolor=CLR_TEXT,
        edgecolor=mcolors.to_rgba(CLR_GRID, 1.0),
        framealpha=0.86,
    )

    ax3d = fig.add_axes([0.52, 0.08, 0.46, 0.84], projection="3d")
    _style_3d_axes(ax3d)

    if len(support_world):
        ax3d.scatter(
            support_world[:, 0],
            support_world[:, 1],
            support_world[:, 2],
            c=CLR_SURFACE,
            s=7,
            alpha=0.20,
            marker="o",
            linewidths=0.0,
            depthshade=False,
            label="support surface",
        )
    if len(free_world):
        ax3d.scatter(
            free_world[:, 0],
            free_world[:, 1],
            free_world[:, 2],
            c=CLR_FREE,
            s=1,
            alpha=0.08,
            marker="o",
            linewidths=0.0,
            depthshade=False,
        )
    if len(occ_world):
        support_world_z = voxel_to_world(np.array([[0, 0, int(support_z)]], dtype=np.intp), vp)[0, 2]
        upper_occ = occ_world[:, 2] > support_world_z + 1.25 * voxel_size
        if np.any(upper_occ):
            pts = occ_world[upper_occ]
            ax3d.scatter(
                pts[:, 0],
                pts[:, 1],
                pts[:, 2],
                c=CLR_OCC,
                s=2,
                alpha=0.38,
                marker="o",
                linewidths=0.0,
                depthshade=False,
            )

    _draw_bbox_3d(ax3d, orig_world, CLR_ORIG, lw=2.5, label=f"Original: {obj_name}")
    _draw_bbox_3d(ax3d, placed_world, CLR_PLACE, lw=2.2, label="Yaw-only freebox", alpha=0.86)
    ax3d.quiver(
        orig_center[0],
        orig_center[1],
        orig_center[2],
        displacement[0],
        displacement[1],
        displacement[2],
        color=CLR_ARROW,
        linewidth=1.6,
        arrow_length_ratio=0.14,
        alpha=0.70,
        label="displacement",
    )

    cam_margin = np.array([voxel_size * 2.0] * 3, dtype=np.float64)
    cam_in_view = np.all(cam_origin >= view_min - cam_margin) and np.all(cam_origin <= view_max + cam_margin)
    if cam_in_view:
        ax3d.scatter(
            *cam_origin,
            c=CLR_CAM,
            s=80,
            marker="*",
            edgecolors=mcolors.to_rgba("#FFF6A3", 0.95),
            linewidths=0.8,
            label="Camera",
            depthshade=False,
        )

    ax3d.set_xlim(view_min[0], view_max[0])
    ax3d.set_ylim(view_min[1], view_max[1])
    ax3d.set_zlim(view_min[2], view_max[2])
    ax3d.set_box_aspect(tuple(view_span.tolist()))
    ax3d.set_xlabel("X (cm)", color=CLR_TEXT, labelpad=8)
    ax3d.set_ylabel("Y (cm)", color=CLR_TEXT, labelpad=8)
    ax3d.set_zlabel("Z (cm)", color=CLR_TEXT, labelpad=6)
    ax3d.set_title(
        "3-D World View\nOrange = current position | Green = yaw-only freebox",
        color=CLR_TEXT,
        fontsize=10,
        pad=6,
    )
    ax3d.view_init(elev=26, azim=-60)
    ax3d.legend(
        fontsize=8,
        loc="upper left",
        facecolor=CLR_PANEL,
        labelcolor=CLR_TEXT,
        edgecolor=mcolors.to_rgba(CLR_GRID, 1.0),
        framealpha=0.86,
    )

    info = (
        f"Target        : {obj_name}\n"
        f"Freebox ID    : {placement.get('sample_id', '')}\n"
        f"Mode          : {placement.get('supervision_mode', 'unknown')}\n"
        f"Cluster ID    : {placement.get('cluster_id')}\n"
        f"Cluster size  : {placement.get('cluster_size')}\n"
        f"Support area  : {placement.get('support_area', 0.0):.1f} cm^2\n"
        f"Clearance     : {placement.get('clearance', 0.0):.1f} cm\n"
        f"|Delta|       : {np.linalg.norm(displacement):.1f} cm"
    )
    fig.text(
        0.535,
        0.03,
        info,
        fontsize=8.5,
        color=CLR_TEXT,
        family="monospace",
        verticalalignment="bottom",
        bbox=dict(facecolor=CLR_PANEL, edgecolor=CLR_GRID, boxstyle="round,pad=0.5", alpha=0.9),
    )
    fig.text(
        0.5,
        0.97,
        f"Freebox Visualisation - {obj_name}",
        ha="center",
        va="top",
        color=CLR_TEXT,
        fontsize=13,
        fontweight="bold",
    )

    plt.savefig(out_path, dpi=150, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)

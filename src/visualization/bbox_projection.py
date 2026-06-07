"""
src/visualization/bbox_projection.py
------------------------------------
将 canonical scene 中物体 3D box 投影到 RGB 图像。
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from src.datasets.canonical import CanonicalScene, ObjectInfo


BOX_EDGES = (
    (0, 1), (2, 3), (4, 5), (6, 7),
    (0, 2), (1, 3), (4, 6), (5, 7),
    (0, 4), (1, 5), (2, 6), (3, 7),
)

COLOR_PALETTE = (
    (230, 57, 70),
    (29, 128, 91),
    (42, 111, 219),
    (244, 162, 97),
    (131, 56, 236),
    (0, 150, 199),
    (233, 196, 106),
    (214, 40, 40),
    (46, 196, 182),
    (255, 127, 80),
)


def get_bbox_corners(bbox3d: np.ndarray) -> np.ndarray:
    """
    从 canonical AABB 生成 8 个角点。

    输入:
        bbox3d: (6,) [min_x, min_y, min_z, max_x, max_y, max_z]
    输出:
        ndarray(8, 3)
    """
    bbox = np.asarray(bbox3d, dtype=np.float64)
    bmin = bbox[:3]
    bmax = bbox[3:]
    corners = []
    for zi in range(2):
        for yi in range(2):
            for xi in range(2):
                corners.append(
                    [
                        [bmin[0], bmax[0]][xi],
                        [bmin[1], bmax[1]][yi],
                        [bmin[2], bmax[2]][zi],
                    ]
                )
    return np.asarray(corners, dtype=np.float64)


def transform_points(points: np.ndarray, transform: np.ndarray) -> np.ndarray:
    """使用 4x4 齐次矩阵变换点集。"""
    pts = np.asarray(points, dtype=np.float64)
    ones = np.ones((pts.shape[0], 1), dtype=np.float64)
    pts_h = np.hstack([pts, ones])
    return (np.asarray(transform, dtype=np.float64) @ pts_h.T).T[:, :3]


def project_world(points_world: np.ndarray, K: np.ndarray, E_w2c: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """将世界坐标点投影到像素坐标。"""
    points_cam = transform_points(points_world, E_w2c)
    z = points_cam[:, 2]
    uv = np.empty((points_cam.shape[0], 2), dtype=np.float64)
    uv[:, 0] = K[0, 0] * points_cam[:, 0] / z + K[0, 2]
    uv[:, 1] = K[1, 1] * points_cam[:, 1] / z + K[1, 2]
    return uv, z


def draw_projected_object(
    draw: ImageDraw.ImageDraw,
    obj: ObjectInfo,
    K: np.ndarray,
    E_w2c: np.ndarray,
    color: tuple[int, int, int],
    line_width: int,
    draw_label: bool,
) -> bool:
    """
    绘制单个物体 3D box 的 RGB 投影。

    返回 True 表示至少绘制了一条边。
    """
    corners_world = transform_points(get_bbox_corners(obj.bbox3d_canonical), obj.pose_world)
    uv, z_cam = project_world(corners_world, K, E_w2c)

    drawn = False
    for i, j in BOX_EDGES:
        if z_cam[i] <= 0.0 or z_cam[j] <= 0.0:
            continue
        draw.line(
            [
                (float(uv[i, 0]), float(uv[i, 1])),
                (float(uv[j, 0]), float(uv[j, 1])),
            ],
            fill=color,
            width=line_width,
        )
        drawn = True

    if draw_label and drawn:
        visible = z_cam > 0.0
        label_uv = uv[visible].mean(axis=0)
        label = f"{obj.obj_id}:{obj.class_name}"
        _draw_label(draw, label, label_uv, color)
    return drawn


def render_scene_bbox_projection(
    scene: CanonicalScene,
    line_width: int = 3,
    draw_labels: bool = True,
) -> Image.Image:
    """
    将一帧 canonical scene 的所有物体 3D box 投影到 RGB 图像。

    输入:
        scene: CanonicalScene
        line_width: int 投影线宽
        draw_labels: bool 是否绘制 obj_id 和类别名
    输出:
        PIL.Image.Image
    """
    image = Image.fromarray(np.asarray(scene.rgb, dtype=np.uint8)).convert("RGB")
    draw = ImageDraw.Draw(image)
    K = scene.camera.K
    E_w2c = scene.camera.E_w2c

    for index, obj in enumerate(scene.objects):
        draw_projected_object(
            draw=draw,
            obj=obj,
            K=K,
            E_w2c=E_w2c,
            color=COLOR_PALETTE[index % len(COLOR_PALETTE)],
            line_width=line_width,
            draw_label=draw_labels,
        )
    return image


def save_scene_bbox_projection(
    scene: CanonicalScene,
    output_path: str | Path,
    line_width: int = 3,
    draw_labels: bool = True,
) -> None:
    """渲染并保存一帧所有物体 3D box 的 RGB 投影图。"""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image = render_scene_bbox_projection(scene, line_width=line_width, draw_labels=draw_labels)
    image.save(output_path)


def _draw_label(
    draw: ImageDraw.ImageDraw,
    text: str,
    uv: np.ndarray,
    color: tuple[int, int, int],
) -> None:
    """在投影框附近绘制可读标签。"""
    font = ImageFont.load_default()
    x = float(uv[0])
    y = float(uv[1])
    bbox = draw.textbbox((x, y), text, font=font)
    pad = 3
    rect = (
        bbox[0] - pad,
        bbox[1] - pad,
        bbox[2] + pad,
        bbox[3] + pad,
    )
    draw.rectangle(rect, fill=(0, 0, 0))
    draw.text((x, y), text, fill=color, font=font)

#!/usr/bin/env python
"""在已有场景图上绘制带径向光晕的单个放置点。

使用示例:
    python tools/render_glowing_placement_point.py \
        --input-image outputs/visualizations/hope__scene_0000__0005_sparse_voxels_oblique_3d.png \
        --output-path outputs/visualizations/hope__scene_0000__0005_glowing_point.png \
        --point-x 390 --point-y 411
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from PIL import JpegImagePlugin  # noqa: F401  # 注册 Pillow PDF 导出所需编码器。


POINT_COLOR = (37, 99, 235)


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(description="Render one glowing placement point.")
    parser.add_argument("--input-image", type=Path, required=True, help="无标注场景 PNG。")
    parser.add_argument("--output-path", type=Path, required=True, help="输出 PNG 路径。")
    parser.add_argument("--point-x", type=float, required=True, help="放置点的图像 x 坐标。")
    parser.add_argument("--point-y", type=float, required=True, help="放置点的图像 y 坐标。")
    parser.add_argument("--glow-radius", type=float, default=38.0, help="光晕半径，单位为像素。")
    parser.add_argument("--point-radius", type=float, default=7.0, help="中心实点半径，单位为像素。")
    return parser.parse_args()


def add_glowing_point(
    image: Image.Image,
    center: tuple[float, float],
    *,
    glow_radius: float = 38.0,
    point_radius: float = 7.0,
    color: tuple[int, int, int] = POINT_COLOR,
) -> Image.Image:
    """返回叠加蓝色高斯光晕与中心实点的 RGB 图像。"""
    if glow_radius <= 0.0 or point_radius <= 0.0:
        raise ValueError("glow_radius and point_radius must be positive")
    if point_radius >= glow_radius:
        raise ValueError("point_radius must be smaller than glow_radius")

    result = image.convert("RGB")
    center_x, center_y = map(float, center)
    if not (0.0 <= center_x < result.width and 0.0 <= center_y < result.height):
        raise ValueError("placement point must lie inside the input image")

    radius = int(np.ceil(glow_radius))
    left = max(int(np.floor(center_x)) - radius, 0)
    top = max(int(np.floor(center_y)) - radius, 0)
    right = min(int(np.floor(center_x)) + radius + 1, result.width)
    bottom = min(int(np.floor(center_y)) + radius + 1, result.height)

    grid_y, grid_x = np.ogrid[top:bottom, left:right]
    distance_sq = (grid_x - center_x) ** 2 + (grid_y - center_y) ** 2
    sigma = glow_radius / 2.4
    alpha = 0.72 * np.exp(-0.5 * distance_sq / (sigma * sigma))
    alpha[distance_sq > glow_radius * glow_radius] = 0.0

    pixels = np.asarray(result, dtype=np.float32).copy()
    region = pixels[top:bottom, left:right]
    region[:] = region * (1.0 - alpha[..., None]) + np.asarray(color) * alpha[..., None]
    result = Image.fromarray(np.rint(pixels).clip(0, 255).astype(np.uint8), mode="RGB")

    draw = ImageDraw.Draw(result)
    box = (
        center_x - point_radius,
        center_y - point_radius,
        center_x + point_radius,
        center_y + point_radius,
    )
    draw.ellipse(box, fill=color, outline=(219, 234, 254), width=2)
    return result


def main() -> None:
    """加载底图并同时导出 PNG 与同名 PDF。"""
    args = parse_args()
    with Image.open(args.input_image) as image:
        output = add_glowing_point(
            image,
            (args.point_x, args.point_y),
            glow_radius=args.glow_radius,
            point_radius=args.point_radius,
        )
    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    output.save(args.output_path)
    pdf_path = args.output_path.with_suffix(".pdf")
    output.save(pdf_path, resolution=200.0)
    print(f"Saved {args.output_path} and {pdf_path}")


if __name__ == "__main__":
    main()

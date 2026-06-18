#!/usr/bin/env python
"""
tools/convert_dopose_to_canonical.py
------------------------------------
将 DOPose BOP 风格数据离线转换为 canonical placement scene 数据集。

使用示例:
    python tools/convert_dopose_to_canonical.py \
        --root-dir /data/jiajun.xie/Spatial-Affordance/data/dopose \
        --output-dir /data/jiajun.xie/3D_Box/data/dopose \
        --depth-window-cm 50 \
        --point-cloud-stride 2

输出的每个 PLY 点云会统一重采样为 50000 点。
DOPose 转换会在 raw scene_link/world 深度点云中 RANSAC 拟合支撑面，
并将支撑面法向对齐到 canonical world-Z。

支撑面拟合参数示例:
    python tools/convert_dopose_to_canonical.py \
        --root-dir /data/jiajun.xie/Spatial-Affordance/data/dopose \
        --output-dir /data/jiajun.xie/3D_Box/data/dopose \
        --point-cloud-stride 2 \
        --support-plane-distance-thresh-cm 1.0 \
        --support-plane-ransac-iters 512 \
        --max-frames 5
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.datasets.dopose_to_canonical import (
    SUPPORT_PLANE_DISTANCE_THRESH_CM,
    SUPPORT_PLANE_MAX_RANSAC_POINTS,
    SUPPORT_PLANE_MIN_INLIER_RATIO,
    SUPPORT_PLANE_MIN_INLIERS,
    SUPPORT_PLANE_RANSAC_ITERS,
    DoPoseCanonicalConverter,
)


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(
        description="Convert DOPose BOP-style frames to canonical placement scene dataset.",
    )
    parser.add_argument(
        "--root-dir",
        default="/data/jiajun.xie/Spatial-Affordance/data/dopose",
        help="DOPose 根目录，包含 models、test_bin、test_table。",
    )
    parser.add_argument(
        "--output-dir",
        default="/data/jiajun.xie/3D_Box/data/dopose",
        help="统一数据集输出目录。",
    )
    parser.add_argument("--point-cloud-stride", type=int, default=4, help="生成点云时的像素采样步长。")
    parser.add_argument(
        "--depth-window-cm",
        type=float,
        default=None,
        help="可选深度过滤窗口，单位 cm；窗口从有效深度 1%% 低分位开始，不提供则不过滤。",
    )
    parser.add_argument("--max-frames", type=int, default=None, help="调试时最多转换多少帧。")
    parser.add_argument(
        "--support-plane-distance-thresh-cm",
        type=float,
        default=SUPPORT_PLANE_DISTANCE_THRESH_CM,
        help="支撑面 RANSAC inlier 距离阈值，单位 cm。",
    )
    parser.add_argument(
        "--support-plane-ransac-iters",
        type=int,
        default=SUPPORT_PLANE_RANSAC_ITERS,
        help="支撑面 RANSAC 迭代次数。",
    )
    parser.add_argument(
        "--support-plane-min-inliers",
        type=int,
        default=SUPPORT_PLANE_MIN_INLIERS,
        help="支撑面 RANSAC 最小 inlier 点数。",
    )
    parser.add_argument(
        "--support-plane-min-inlier-ratio",
        type=float,
        default=SUPPORT_PLANE_MIN_INLIER_RATIO,
        help="支撑面 RANSAC 最小 inlier 比例。",
    )
    parser.add_argument(
        "--support-plane-max-points",
        type=int,
        default=SUPPORT_PLANE_MAX_RANSAC_POINTS,
        help="参与支撑面 RANSAC 的最大点数。",
    )
    return parser.parse_args()


def main() -> None:
    """执行 DOPose 到 canonical 数据集转换。"""
    args = parse_args()
    converter = DoPoseCanonicalConverter(
        root_dir=args.root_dir,
        output_dir=Path(args.output_dir),
        point_cloud_stride=args.point_cloud_stride,
        depth_window_cm=args.depth_window_cm,
        support_plane_distance_thresh_cm=args.support_plane_distance_thresh_cm,
        support_plane_ransac_iters=args.support_plane_ransac_iters,
        support_plane_min_inliers=args.support_plane_min_inliers,
        support_plane_min_inlier_ratio=args.support_plane_min_inlier_ratio,
        support_plane_max_points=args.support_plane_max_points,
    )
    manifest = converter.convert_all(max_frames=args.max_frames)
    print(
        f"Converted {manifest['sample_count']} samples to {Path(args.output_dir).resolve()}"
    )


if __name__ == "__main__":
    main()

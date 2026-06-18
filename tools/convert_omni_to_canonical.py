#!/usr/bin/env python
"""
tools/convert_omni_to_canonical.py
------------------------------------
将 Omni6DPose ROPE 数据离线转换为 canonical placement scene 数据集。

使用示例:
    conda run -n spatial python tools/convert_omni_to_canonical.py \
        --root-dir /data/jiajun.xie/Spatial-Affordance/data/omni \
        --output-dir /data/jiajun.xie/3D_Box/data/omni \
        --depth-window-cm 50 \
        --frame-step 30 \
        --point-cloud-stride 2

调试示例（只转换前 5 帧）:
    conda run -n spatial python tools/convert_omni_to_canonical.py \
        --root-dir /data/jiajun.xie/Spatial-Affordance/data/omni \
        --output-dir /data/jiajun.xie/3D_Box/data/omni \
        --frame-step 30 --max-frames 5

输出规格：
  - 点云统一重采样为 50000 点，1cm 体素化版本单独存储
  - 深度图单位 cm（原始为 m）
  - world-Z 对齐支撑面法向（RANSAC 拟合）
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.datasets.omni_to_canonical import (
    SUPPORT_PLANE_DISTANCE_THRESH_CM,
    SUPPORT_PLANE_MAX_RANSAC_POINTS,
    SUPPORT_PLANE_MIN_INLIER_RATIO,
    SUPPORT_PLANE_MIN_INLIERS,
    SUPPORT_PLANE_RANSAC_ITERS,
    OmniCanonicalConverter,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert Omni6DPose ROPE frames to canonical placement scene dataset.",
    )
    parser.add_argument(
        "--root-dir",
        default="/data/jiajun.xie/Spatial-Affordance/data/omni",
        help="Omni6DPose 根目录，包含 ROPE/ 子目录。",
    )
    parser.add_argument(
        "--output-dir",
        default="/data/jiajun.xie/3D_Box/data/omni",
        help="canonical 数据集输出目录。",
    )
    parser.add_argument(
        "--frame-step",
        type=int,
        default=30,
        help="帧采样步长，每 frame_step 帧取一帧（默认 30）。",
    )
    parser.add_argument(
        "--point-cloud-stride",
        type=int,
        default=4,
        help="点云像素采样步长（默认 4）。",
    )
    parser.add_argument(
        "--depth-window-cm",
        type=float,
        default=None,
        help="深度过滤窗口，单位 cm；不指定则不过滤。",
    )
    parser.add_argument("--max-frames", type=int, default=None, help="调试用最大帧数。")
    parser.add_argument(
        "--support-plane-distance-thresh-cm",
        type=float,
        default=SUPPORT_PLANE_DISTANCE_THRESH_CM,
    )
    parser.add_argument("--support-plane-ransac-iters", type=int, default=SUPPORT_PLANE_RANSAC_ITERS)
    parser.add_argument("--support-plane-min-inliers", type=int, default=SUPPORT_PLANE_MIN_INLIERS)
    parser.add_argument(
        "--support-plane-min-inlier-ratio",
        type=float,
        default=SUPPORT_PLANE_MIN_INLIER_RATIO,
    )
    parser.add_argument("--support-plane-max-points", type=int, default=SUPPORT_PLANE_MAX_RANSAC_POINTS)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    converter = OmniCanonicalConverter(
        root_dir=args.root_dir,
        output_dir=Path(args.output_dir),
        frame_step=args.frame_step,
        point_cloud_stride=args.point_cloud_stride,
        depth_window_cm=args.depth_window_cm,
        support_plane_distance_thresh_cm=args.support_plane_distance_thresh_cm,
        support_plane_ransac_iters=args.support_plane_ransac_iters,
        support_plane_min_inliers=args.support_plane_min_inliers,
        support_plane_min_inlier_ratio=args.support_plane_min_inlier_ratio,
        support_plane_max_points=args.support_plane_max_points,
    )
    manifest = converter.convert_all(max_frames=args.max_frames)
    print(f"Converted {manifest['sample_count']} samples to {Path(args.output_dir).resolve()}")
    if manifest["preprocess"].get("skip_count", 0) > 0:
        print(f"Skipped {manifest['preprocess']['skip_count']} frames (support plane failed or pose misaligned)")


if __name__ == "__main__":
    main()

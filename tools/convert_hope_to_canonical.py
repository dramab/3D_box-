#!/usr/bin/env python
"""
tools/convert_hope_to_canonical.py
----------------------------------
将 HOPE-Video 原始数据离线转换为 canonical placement scene 数据集。

使用示例:
    python tools/convert_hope_to_canonical.py \
        --root-dir /path/to/hope_video \
        --mesh-dir /path/to/meshes/full \
        --output-dir /data/jiajun.xie/3D_Box/data/canonical/hope \
        --frame-step 5 \
        --point-cloud-stride 2 \
        --depth-window-cm 50 

输出的每个 PLY 点云会统一重采样为 50000 点。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.datasets.hope_to_canonical import HopeCanonicalConverter


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(
        description="Convert HOPE-Video frames to canonical placement scene dataset.",
    )
    parser.add_argument("--root-dir", required=True, help="HOPE-Video 根目录，包含 scene_XXXX 子目录。")
    parser.add_argument("--mesh-dir", required=True, help="HOPE mesh 目录，包含 {class_name}.obj。")
    parser.add_argument(
        "--output-dir",
        default="/data/jiajun.xie/3D_Box/data/canonical/hope",
        help="统一数据集输出目录。",
    )
    parser.add_argument("--frame-step", type=int, default=60, help="按帧 ID 步长采样。")
    parser.add_argument("--point-cloud-stride", type=int, default=4, help="生成点云时的像素采样步长。")
    parser.add_argument(
        "--depth-window-cm",
        type=float,
        default=None,
        help="可选深度过滤窗口，单位 cm；窗口从有效深度 1% 低分位开始，不提供则不过滤。",
    )
    parser.add_argument("--max-frames", type=int, default=None, help="调试时最多转换多少帧。")
    return parser.parse_args()


def main() -> None:
    """执行 HOPE 到 canonical 数据集转换。"""
    args = parse_args()
    converter = HopeCanonicalConverter(
        root_dir=args.root_dir,
        mesh_dir=args.mesh_dir,
        output_dir=Path(args.output_dir),
        frame_step=args.frame_step,
        point_cloud_stride=args.point_cloud_stride,
        depth_window_cm=args.depth_window_cm,
    )
    manifest = converter.convert_all(max_frames=args.max_frames)
    print(
        f"Converted {manifest['sample_count']} samples to {Path(args.output_dir).resolve()}"
    )


if __name__ == "__main__":
    main()

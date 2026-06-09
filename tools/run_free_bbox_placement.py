#!/usr/bin/env python
"""
tools/run_free_bbox_placement.py
--------------------------------
在 canonical HOPE 数据集上运行 free_bbox 放置标注。

使用示例:
    conda run -n spatial python tools/run_free_bbox_placement.py \
        --dataset-dir /data/jiajun.xie/3D_Box/data/hope \
        --sample-id hope__scene_0000__0000 \
        --output-dir outputs/free_bbox_hope

    conda run -n spatial python tools/run_free_bbox_placement.py \
        --dataset-dir /data/jiajun.xie/3D_Box/data/hope \
        --all --max-frames 5 \
        --output-dir outputs/free_bbox_hope
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.annotation.free_bbox import FreeBBoxConfig, FreeBBoxPipeline
from src.datasets.canonical import load_canonical_scene


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(
        description="Run free_bbox placement annotation on canonical voxel point clouds.",
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=Path("/data/jiajun.xie/3D_Box/data/hope"),
        help="canonical 数据集目录，包含 manifest.json 和 samples/。",
    )
    parser.add_argument("--sample-id", nargs="*", default=None, help="指定一个或多个 sample_id。")
    parser.add_argument(
        "--sample-json",
        nargs="*",
        type=Path,
        default=None,
        help="指定一个或多个 sample JSON 路径。",
    )
    parser.add_argument(
        "--sample-list",
        type=Path,
        default=None,
        help="文本文件，每行一个 sample_id 或 sample JSON 路径。",
    )
    parser.add_argument("--all", action="store_true", help="从 manifest.json 运行全部帧。")
    parser.add_argument("--max-frames", type=int, default=None, help="最多处理多少帧，用于批量调试。")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/free_bbox_hope"),
        help="输出根目录；不同类型文件会保存到 boxes/、heatmaps/、support_masks/、placements/、visualizations/。",
    )
    parser.add_argument("--voxel-size", type=float, default=1.0, help="体素边长，单位 cm。")
    parser.add_argument("--grid-padding", type=float, default=10.0, help="搜索栅格 padding，单位 cm。")
    parser.add_argument("--safety-margin", type=float, default=0.5, help="碰撞安全边距，单位 cm。")
    parser.add_argument("--yaw-steps", type=int, default=24, help="yaw 离散步数。")
    parser.add_argument("--min-surface-area", type=float, default=50.0, help="最小支撑面面积，单位 cm^2。")
    parser.add_argument("--min-support-ratio", type=float, default=1.0, help="最小支撑比例。")
    parser.add_argument("--dbscan-eps", type=float, default=None, help="DBSCAN eps；不传则按物体尺度估计。")
    parser.add_argument("--dbscan-min-samples", type=int, default=1, help="DBSCAN min_samples。")
    parser.add_argument(
        "--max-reps-total",
        type=int,
        default=None,
        help="单物体最多输出多少个簇代表；不传则每个簇输出一个。",
    )
    parser.add_argument(
        "--no-preserve-orientation",
        action="store_true",
        help="不保留原始 roll/pitch，仅使用 yaw-only 平放姿态。",
    )
    return parser.parse_args()


def resolve_sample_paths(args: argparse.Namespace) -> list[Path]:
    """根据 CLI 参数解析待处理 sample JSON。"""
    dataset_dir = args.dataset_dir
    sample_paths: list[Path] = []

    for sample_id in args.sample_id or []:
        sample_paths.append(dataset_dir / "samples" / f"{sample_id}.json")

    for sample_json in args.sample_json or []:
        sample_paths.append(sample_json)

    if args.sample_list is not None:
        for line in args.sample_list.read_text().splitlines():
            value = line.strip()
            if not value or value.startswith("#"):
                continue
            path = Path(value)
            if path.suffix == ".json":
                sample_paths.append(path if path.is_absolute() else dataset_dir / path)
            else:
                sample_paths.append(dataset_dir / "samples" / f"{value}.json")

    if args.all:
        manifest_path = dataset_dir / "manifest.json"
        with manifest_path.open("r") as f:
            manifest = json.load(f)
        for item in manifest["samples"]:
            sample_paths.append(dataset_dir / item["sample_path"])

    if not sample_paths:
        raise ValueError("No samples selected. Use --sample-id, --sample-json, --sample-list, or --all.")

    deduped = []
    seen = set()
    for path in sample_paths:
        resolved = path.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        deduped.append(path)

    if args.max_frames is not None:
        deduped = deduped[: int(args.max_frames)]
    return deduped


def make_config(args: argparse.Namespace) -> FreeBBoxConfig:
    """由 CLI 参数构建 pipeline 配置。"""
    return FreeBBoxConfig(
        voxel_size=args.voxel_size,
        grid_padding=args.grid_padding,
        safety_margin=args.safety_margin,
        yaw_steps=args.yaw_steps,
        min_surface_area=args.min_surface_area,
        min_support_ratio=args.min_support_ratio,
        dbscan_eps=args.dbscan_eps,
        dbscan_min_samples=args.dbscan_min_samples,
        max_reps_total=args.max_reps_total,
        preserve_orientation=not args.no_preserve_orientation,
    )


def main() -> None:
    """执行 free_bbox 放置标注。"""
    args = parse_args()
    pipeline = FreeBBoxPipeline(make_config(args))
    sample_paths = resolve_sample_paths(args)

    for sample_path in sample_paths:
        scene = load_canonical_scene(sample_path, dataset_root=args.dataset_dir)
        print(f"[free_bbox] Processing {scene.sample_id}")
        results = pipeline.run(scene, output_dir=args.output_dir)
        n_boxes = sum(len(result.placements) for result in results.values())
        print(f"[free_bbox] Saved {n_boxes} best boxes for {scene.sample_id}")

    print(f"[free_bbox] Processed {len(sample_paths)} sample(s). Output: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()

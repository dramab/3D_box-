#!/usr/bin/env python
"""
tools/export_canonical_rgb_bbox_vis.py
--------------------------------------
导出 canonical 数据集中指定帧或批量帧的 RGB 3D box 投影图。

使用示例:
    python tools/export_canonical_rgb_bbox_vis.py \
        --dataset-dir /data/jiajun.xie/3D_Box/data/canonical/hope \
        --sample-id hope__scene_0000__0000

    python tools/export_canonical_rgb_bbox_vis.py \
        --dataset-dir /data/jiajun.xie/3D_Box/data/canonical/hope \
        --all --max-frames 10
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.datasets.canonical import load_canonical_scene
from src.visualization.bbox_projection import save_scene_bbox_projection


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(
        description="Export RGB images with all canonical scene object 3D boxes projected.",
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=Path("/data/jiajun.xie/3D_Box/data/canonical/hope"),
        help="canonical 数据集目录，包含 manifest.json 和 samples/。",
    )
    parser.add_argument(
        "--sample-id",
        nargs="*",
        default=None,
        help="指定一个或多个 sample_id。",
    )
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
    parser.add_argument(
        "--all",
        action="store_true",
        help="从 manifest.json 导出全部帧。",
    )
    parser.add_argument("--max-frames", type=int, default=None, help="最多导出多少帧，用于批量调试。")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="输出目录；默认保存到 dataset-dir/rgb_bbox_vis。",
    )
    parser.add_argument("--line-width", type=int, default=3, help="3D box 投影线宽。")
    parser.add_argument("--no-labels", action="store_true", help="不绘制物体 obj_id 和类别名。")
    return parser.parse_args()


def resolve_sample_paths(args: argparse.Namespace) -> list[Path]:
    """根据 CLI 参数解析需要导出的 sample JSON 路径。"""
    sample_paths: list[Path] = []
    dataset_dir = args.dataset_dir

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


def main() -> None:
    """执行 RGB 3D box 投影图导出。"""
    args = parse_args()
    output_dir = args.output_dir or args.dataset_dir / "rgb_bbox_vis"
    sample_paths = resolve_sample_paths(args)

    for sample_path in sample_paths:
        scene = load_canonical_scene(sample_path, dataset_root=args.dataset_dir)
        output_path = output_dir / f"{scene.sample_id}.png"
        save_scene_bbox_projection(
            scene,
            output_path,
            line_width=args.line_width,
            draw_labels=not args.no_labels,
        )
        print(f"Saved {output_path}")

    print(f"Exported {len(sample_paths)} RGB bbox projection images to {output_dir}")


if __name__ == "__main__":
    main()

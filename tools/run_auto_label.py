#!/usr/bin/env python
"""
tools/run_auto_label.py
-----------------------
为 free_bbox pipeline 产出的放置样本生成自然语言移动指令标注。

以 placements/*.json 为索引，逐物体、逐 placement 生成形如
"Move {object} located at {rel_a} {ref_a} to {rel_b} {ref_b}." 的描述，
相机与完整参照物从 canonical 数据集 (--dataset-dir) 补齐。

使用示例:
    # 小批量验证（指定 sample_id）
    conda run -n spatial python tools/run_auto_label.py \
        --placements-dir outputs/free_bbox_hope/placements \
        --dataset-dir data/hope \
        --output-dir outputs/auto_labels_hope \
        --sample-ids hope__scene_0000__0000 hope__scene_0000__0005

    # 全量
    conda run -n spatial python tools/run_auto_label.py \
        --placements-dir outputs/free_bbox_hope/placements \
        --dataset-dir data/hope \
        --output-dir outputs/auto_labels_hope

查看报告: 进入 output-dir 上级，python3 -m http.server 8080，浏览器打开 report.html。
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Set

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.annotation.auto_label import generate_label_for_placement, get_mapping
from src.datasets.canonical import CanonicalScene, load_canonical_scene

DEFAULT_MAPPING = PROJECT_ROOT / "configs/annotation/mapping.json"


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(description="为 free_bbox 放置样本生成自然语言标注")
    parser.add_argument("--placements-dir", required=True, type=Path, help="placements JSON 目录")
    parser.add_argument("--dataset-dir", required=True, type=Path, help="canonical 数据集根目录（取相机+参照物）")
    parser.add_argument("--output-dir", required=True, type=Path, help="标注 JSON 与 HTML 报告输出目录")
    parser.add_argument("--mapping", type=Path, default=DEFAULT_MAPPING, help="类别名映射文件 (JSON)")
    parser.add_argument("--sample-ids", nargs="+", default=None, help="仅标注指定 sample_id")
    parser.add_argument("--limit", type=int, default=None, help="仅处理前 N 个 placements 文件")
    return parser.parse_args()


def load_scene_cached(scene_cache: Dict[str, CanonicalScene], dataset_dir: Path, sample_id: str) -> Optional[CanonicalScene]:
    """按 sample_id 缓存加载 canonical 场景，缺失时返回 None。"""
    if sample_id not in scene_cache:
        sample_path = dataset_dir / "samples" / f"{sample_id}.json"
        if not sample_path.exists():
            scene_cache[sample_id] = None
        else:
            scene_cache[sample_id] = load_canonical_scene(sample_path, dataset_root=dataset_dir)
    return scene_cache[sample_id]


def collect_placement_files(placements_dir: Path, sample_id_filter: Optional[Set[str]]) -> List[Path]:
    """收集 placements JSON 文件，可按 sample_id 过滤（文件名形如 <sample_id>__placements.json）。"""
    files = sorted(placements_dir.glob("*__placements.json"))
    if sample_id_filter is None:
        return files
    return [f for f in files if f.name.replace("__placements.json", "") in sample_id_filter]


def build_label_records(placement_payload: dict, scene: CanonicalScene, mapping_data: dict) -> List[dict]:
    """为一个 placements 文件中所有物体的所有 placement 生成标注记录。"""
    records = []
    for obj_record in placement_payload.get("objects", []):
        for placement in obj_record.get("placements", []):
            label, spatial_relation = generate_label_for_placement(
                obj_record=obj_record,
                placement=placement,
                reference_objects=scene.objects,
                camera=scene.camera,
                mapping_data=mapping_data,
            )
            records.append(
                {
                    "sample_id": placement_payload["sample_id"],
                    "object_id": obj_record["object_id"],
                    "class_name": obj_record.get("class_name"),
                    "cluster_id": placement.get("cluster_id"),
                    "placement_sample_id": placement.get("sample_id"),
                    "label": label,
                    "spatial_relation": spatial_relation,
                    "visualization_png": placement.get("visualization_png"),
                }
            )
    return records


def build_report_html(output_dir: Path, vis_root: Path, all_labels: List[dict]) -> str:
    """构建只读 HTML 标注查看报告，图片指向各 placement 的 visualization_png。"""
    html_lines = [
        "<!DOCTYPE html><html><head><meta charset='utf-8'><title>标注查看</title>",
        "<style>",
        "  body { font-family: 'Segoe UI', sans-serif; background-color: #f4f4f9; padding: 20px; padding-top: 60px; }",
        "  .header-bar { position: fixed; top: 0; left: 0; right: 0; background: #2c3e50; color: white; padding: 10px 40px; z-index: 1000; box-shadow: 0 2px 10px rgba(0,0,0,0.3); }",
        "  .container { max-width: 1200px; margin: auto; }",
        "  .card { display: flex; background: white; margin-bottom: 15px; padding: 15px; border-radius: 8px; box-shadow: 0 4px 8px rgba(0,0,0,0.1); align-items: center; }",
        "  .card img { max-width: 380px; max-height: 380px; border-radius: 4px; object-fit: contain; margin-right: 30px; background: #eee; }",
        "  .info { flex: 1; }",
        "  .filename { color: #888; font-size: 13px; margin-bottom: 8px; font-family: monospace; }",
        "  .label { font-size: 20px; font-weight: bold; color: #34495e; line-height: 1.5; }",
        "  .highlight { color: #e74c3c; }",
        "</style>",
        "</head><body>",
        "<div class='header-bar'><h2 style='margin:0'>📸 标注查看</h2></div>",
        "<div class='container'>",
    ]
    for item in all_labels:
        vis_png = item.get("visualization_png")
        img_src = os.path.relpath(vis_root / vis_png, output_dir) if vis_png else ""
        title = item.get("placement_sample_id") or item["sample_id"]
        html_lines.append("  <div class='card'>")
        if img_src:
            html_lines.append(f"    <img src='{img_src}' loading='lazy' />")
        html_lines.append("    <div class='info'>")
        html_lines.append(f"      <div class='filename'>📄 {title}</div>")
        html_lines.append(f"      <div class='label'>👉 <span class='highlight'>{item['label']}</span></div>")
        html_lines.append("    </div></div>")
    html_lines.append("</div></body></html>")
    return "\n".join(html_lines)


def main() -> None:
    """解析参数并执行自动标注主流程。"""
    args = parse_args()
    placements_dir = args.placements_dir.resolve()
    dataset_dir = args.dataset_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    mapping_data = get_mapping(str(args.mapping.resolve()) if args.mapping else None)
    sample_id_filter = set(args.sample_ids) if args.sample_ids else None

    placement_files = collect_placement_files(placements_dir, sample_id_filter)
    if args.limit is not None:
        placement_files = placement_files[: args.limit]
    if not placement_files:
        print(f"未在 {placements_dir} 中找到匹配的 placements 文件！")
        return

    print(f"找到 {len(placement_files)} 个 placements 文件，开始标注...")
    # visualization_png 字段相对于 placements 上级目录（即 free_bbox 输出根）
    vis_root = placements_dir.parent

    scene_cache: Dict[str, Optional[CanonicalScene]] = {}
    all_labels: List[dict] = []
    skipped: List[str] = []

    for idx, pf in enumerate(placement_files, 1):
        with pf.open("r", encoding="utf-8") as f:
            payload = json.load(f)
        sample_id = payload["sample_id"]
        scene = load_scene_cached(scene_cache, dataset_dir, sample_id)
        if scene is None:
            skipped.append(sample_id)
            continue
        all_labels.extend(build_label_records(payload, scene, mapping_data))
        if idx % 50 == 0:
            print(f"已处理 {idx}/{len(placement_files)} 个文件，累计 {len(all_labels)} 条标注")

    all_labels_path = output_dir / "all_labels.json"
    with all_labels_path.open("w", encoding="utf-8") as f:
        json.dump(all_labels, f, indent=2, ensure_ascii=False)

    if all_labels:
        report_path = output_dir / "report.html"
        with report_path.open("w", encoding="utf-8") as f:
            f.write(build_report_html(output_dir, vis_root, all_labels))
        print(f"✅ HTML 报告: {report_path}")

    if skipped:
        print(f"⚠️ 跳过 {len(skipped)} 个找不到 canonical sample 的文件: {', '.join(skipped[:10])}")
    print(f"✅ 标注完成，共 {len(all_labels)} 条 -> {all_labels_path}")


if __name__ == "__main__":
    main()

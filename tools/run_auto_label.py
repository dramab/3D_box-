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
        --output-dir outputs/auto_labels_hope \
        --workers 8

查看报告: 进入 output-dir 上级，python3 -m http.server 8080，浏览器打开 report.html。
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import multiprocessing
import os
import sys
from pathlib import Path
from typing import List, Optional, Set

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if "MPLCONFIGDIR" not in os.environ:
    mpl_config_dir = PROJECT_ROOT / "outputs" / ".matplotlib"
    mpl_config_dir.mkdir(parents=True, exist_ok=True)
    os.environ["MPLCONFIGDIR"] = os.fspath(mpl_config_dir)

from src.annotation.auto_label import (
    describe_spatial_relation,
    generate_label_for_placement,
    get_mapping,
    get_object_corners_world,
)
from src.annotation.free_bbox.io_utils import load_ply, save_ply
from src.datasets.canonical import CanonicalScene, ObjectInfo, load_canonical_scene

DEFAULT_MAPPING = PROJECT_ROOT / "configs/annotation/mapping.json"
DIRECTION_FILTERED_HEATMAP_DIR = "direction_filtered_heatmaps"
DEMOTED_HEATMAP_COLOR = np.array([55, 120, 210], dtype=np.uint8)
DEFAULT_WORKERS = min(4, os.cpu_count() or 1)


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(description="为 free_bbox 放置样本生成自然语言标注")
    parser.add_argument("--placements-dir", required=True, type=Path, help="placements JSON 目录")
    parser.add_argument("--dataset-dir", required=True, type=Path, help="canonical 数据集根目录（取相机+参照物）")
    parser.add_argument("--output-dir", required=True, type=Path, help="标注 JSON 与 HTML 报告输出目录")
    parser.add_argument("--mapping", type=Path, default=DEFAULT_MAPPING, help="类别名映射文件 (JSON)")
    parser.add_argument("--sample-ids", nargs="+", default=None, help="仅标注指定 sample_id")
    parser.add_argument("--limit", type=int, default=None, help="仅处理前 N 个 placements 文件")
    parser.add_argument(
        "--workers",
        type=int,
        default=DEFAULT_WORKERS,
        help=f"并发进程数；默认 {DEFAULT_WORKERS}，设为 1 可串行运行。",
    )
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be at least 1")
    return args


def collect_placement_files(placements_dir: Path, sample_id_filter: Optional[Set[str]]) -> List[Path]:
    """收集 placements JSON 文件，可按 sample_id 过滤（文件名形如 <sample_id>__placements.json）。"""
    files = sorted(placements_dir.glob("*__placements.json"))
    if sample_id_filter is None:
        return files
    return [f for f in files if f.name.replace("__placements.json", "") in sample_id_filter]


def increment_counter(counter: dict, key: str, amount: int = 1) -> None:
    """累加 dict 计数器。"""
    counter[key] = int(counter.get(key, 0)) + int(amount)


def is_heatmap_positive_color(colors: np.ndarray) -> np.ndarray:
    """识别 Stage 2 会读取为正候选的 heatmap 颜色。"""
    colors_i = np.asarray(colors, dtype=np.int16)
    return (colors_i[:, 0] == 255) & (colors_i[:, 2] == 30)


def resolve_free_bbox_path(free_bbox_root: Path, raw_path: str | os.PathLike[str]) -> Path:
    """解析 free_bbox 输出中的相对路径。"""
    path = Path(raw_path)
    return path if path.is_absolute() else free_bbox_root / path


def direction_filtered_heatmap_path(free_bbox_root: Path, raw_heatmap_path: str | os.PathLike[str]) -> Path:
    """将原始 heatmap 路径映射到 direction_filtered_heatmaps 目录。"""
    return free_bbox_root / DIRECTION_FILTERED_HEATMAP_DIR / Path(raw_heatmap_path).name


def find_reference_object(scene: CanonicalScene, reference_object_id: str) -> Optional[ObjectInfo]:
    """按 obj_id 查找参照物。"""
    for obj in scene.objects:
        if str(obj.obj_id) == str(reference_object_id):
            return obj
    return None


def filter_heatmap_by_direction(
    placement: dict,
    spatial_relation: dict,
    scene: CanonicalScene,
    free_bbox_root: Path,
) -> tuple[bool, dict]:
    """按 placement 文字方向过滤 heatmap 正激活点，并保存到新目录。"""
    raw_heatmap = placement.get("heatmap_ply")
    if not raw_heatmap:
        return False, {"reason": "missing_heatmap_path"}

    source_path = resolve_free_bbox_path(free_bbox_root, raw_heatmap)
    if not source_path.exists():
        return False, {"reason": "missing_heatmap_file", "path": os.fspath(source_path)}

    placement_relation = spatial_relation["placement"]
    ref_id = placement_relation.get("reference_object_id")
    if ref_id is None:
        return False, {"reason": "missing_placement_reference"}
    ref_obj = find_reference_object(scene, str(ref_id))
    if ref_obj is None:
        return False, {"reason": "unknown_placement_reference", "reference_object_id": str(ref_id)}

    points, colors = load_ply(source_path)
    positive = is_heatmap_positive_color(colors)
    raw_positive_count = int(positive.sum())
    if raw_positive_count == 0:
        return False, {"reason": "heatmap_no_positive_points", "raw_positive_points": 0}

    target_relation = str(placement_relation["relation"])
    placement_corners = np.asarray(placement["corners_world"], dtype=np.float64)
    bottom_center = np.asarray(placement["bottom_center_world"], dtype=np.float64)
    ref_corners = get_object_corners_world(ref_obj)

    keep_positive = positive.copy()
    positive_indices = np.flatnonzero(positive)
    for point_index in positive_indices:
        candidate_corners = placement_corners + (points[point_index].astype(np.float64) - bottom_center)
        relation = describe_spatial_relation(candidate_corners, ref_corners, scene.camera.E_w2c, scene.camera.K)
        keep_positive[point_index] = relation == target_relation

    filtered_positive_count = int(keep_positive.sum())
    if filtered_positive_count == 0:
        return False, {
            "reason": "direction_filter_removed_all_positive_points",
            "raw_positive_points": raw_positive_count,
            "filtered_positive_points": 0,
        }

    output_colors = colors.copy()
    output_colors[positive & ~keep_positive] = DEMOTED_HEATMAP_COLOR
    output_path = direction_filtered_heatmap_path(free_bbox_root, raw_heatmap)
    save_ply(output_path, points, output_colors)
    return True, {
        "raw_positive_points": raw_positive_count,
        "filtered_positive_points": filtered_positive_count,
        "filtered_heatmap_ply": os.path.relpath(output_path, free_bbox_root),
    }


def update_relation_stats(stats: dict, spatial_relation: dict) -> None:
    """统计严格筛选和无阈值参照物的使用情况。"""
    used_unfiltered = False
    for side in ("original", "placement"):
        mode = spatial_relation.get(side, {}).get("reference_selection_mode")
        if mode:
            increment_counter(stats["reference_selection_by_side"], f"{side}:{mode}")
        if mode == "unfiltered":
            used_unfiltered = True
    if used_unfiltered:
        stats["records_using_unfiltered_reference"] += 1


def build_label_records(
    placement_payload: dict,
    scene: CanonicalScene,
    mapping_data: dict,
    free_bbox_root: Path,
    stats: dict,
) -> List[dict]:
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
            if label is None:
                increment_counter(stats["skipped_by_reason"], str(spatial_relation.get("skip_reason", "unknown")))
                continue

            ok, heatmap_stats = filter_heatmap_by_direction(
                placement=placement,
                spatial_relation=spatial_relation,
                scene=scene,
                free_bbox_root=free_bbox_root,
            )
            stats["heatmap_raw_positive_points"] += int(heatmap_stats.get("raw_positive_points", 0))
            stats["heatmap_filtered_positive_points"] += int(heatmap_stats.get("filtered_positive_points", 0))
            if not ok:
                increment_counter(stats["skipped_by_reason"], str(heatmap_stats.get("reason", "heatmap_filter_failed")))
                continue

            update_relation_stats(stats, spatial_relation)
            stats["records_with_filtered_heatmap"] += 1
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


def build_filter_stats() -> dict:
    """创建 auto label 过滤统计结构。"""
    return {
        "schema_version": "auto_label_filter_stats/v1",
        "skipped_by_reason": {},
        "reference_selection_by_side": {},
        "records_using_unfiltered_reference": 0,
        "records_with_filtered_heatmap": 0,
        "heatmap_raw_positive_points": 0,
        "heatmap_filtered_positive_points": 0,
    }


def merge_filter_stats(target: dict, source: dict) -> None:
    """按串行文件顺序合并过滤统计，保持计数和 key 首次出现顺序一致。"""
    for key in ("skipped_by_reason", "reference_selection_by_side"):
        for reason, count in source[key].items():
            increment_counter(target[key], reason, int(count))

    for key in (
        "records_using_unfiltered_reference",
        "records_with_filtered_heatmap",
        "heatmap_raw_positive_points",
        "heatmap_filtered_positive_points",
    ):
        target[key] += int(source[key])


def collect_heatmap_output_paths(placement_payload: dict, free_bbox_root: Path) -> Set[Path]:
    """收集一个 placements 文件会写出的方向过滤 heatmap 路径。"""
    output_paths: Set[Path] = set()
    for obj_record in placement_payload.get("objects", []):
        for placement in obj_record.get("placements", []):
            raw_heatmap = placement.get("heatmap_ply")
            if raw_heatmap:
                output_paths.add(direction_filtered_heatmap_path(free_bbox_root, raw_heatmap).resolve())
    return output_paths


def find_cross_file_heatmap_output_conflicts(
    placement_files: List[Path],
    free_bbox_root: Path,
) -> List[tuple[Path, Path, Path]]:
    """查找跨 placements 文件写同一个 filtered heatmap 的情况。"""
    seen: dict[Path, Path] = {}
    conflicts = []
    for placement_file in placement_files:
        with placement_file.open("r", encoding="utf-8") as f:
            payload = json.load(f)
        for output_path in collect_heatmap_output_paths(payload, free_bbox_root):
            previous_file = seen.get(output_path)
            if previous_file is not None and previous_file != placement_file:
                conflicts.append((output_path, previous_file, placement_file))
                continue
            seen[output_path] = placement_file
    return conflicts


def process_placement_file(
    placement_file: Path,
    dataset_dir: Path,
    mapping_data: dict,
    free_bbox_root: Path,
) -> dict:
    """处理单个 placements 文件，供串行和多进程复用。"""
    with placement_file.open("r", encoding="utf-8") as f:
        payload = json.load(f)

    sample_id = payload["sample_id"]
    sample_path = dataset_dir / "samples" / f"{sample_id}.json"
    if not sample_path.exists():
        return {
            "sample_id": sample_id,
            "records": [],
            "stats": build_filter_stats(),
            "skipped_missing_scene": True,
        }

    scene = load_canonical_scene(sample_path, dataset_root=dataset_dir)
    stats = build_filter_stats()
    records = build_label_records(payload, scene, mapping_data, free_bbox_root, stats)
    return {
        "sample_id": sample_id,
        "records": records,
        "stats": stats,
        "skipped_missing_scene": False,
    }


def merge_process_result(result: dict, all_labels: List[dict], skipped: List[str], filter_stats: dict) -> None:
    """把单文件处理结果合并到总结果。"""
    if result["skipped_missing_scene"]:
        skipped.append(result["sample_id"])
        return
    all_labels.extend(result["records"])
    merge_filter_stats(filter_stats, result["stats"])


def run_parallel_labeling(
    placement_files: List[Path],
    dataset_dir: Path,
    mapping_data: dict,
    free_bbox_root: Path,
    workers: int,
) -> List[dict]:
    """并发处理 placements 文件，并按输入顺序返回结果。"""
    ordered_results: List[Optional[dict]] = [None] * len(placement_files)
    # spawn 避免科学计算库在 fork 后继承线程池状态。
    mp_context = multiprocessing.get_context("spawn")
    with concurrent.futures.ProcessPoolExecutor(
        max_workers=workers,
        mp_context=mp_context,
    ) as executor:
        future_to_index = {
            executor.submit(
                process_placement_file,
                placement_file,
                dataset_dir,
                mapping_data,
                free_bbox_root,
            ): index
            for index, placement_file in enumerate(placement_files)
        }
        completed = 0
        for future in concurrent.futures.as_completed(future_to_index):
            index = future_to_index[future]
            placement_file = placement_files[index]
            try:
                ordered_results[index] = future.result()
            except Exception as exc:
                raise RuntimeError(f"Failed to process placements file: {placement_file}") from exc
            completed += 1
            if completed % 50 == 0 or completed == len(placement_files):
                print(f"已完成 {completed}/{len(placement_files)} 个文件")

    missing_indices = [str(index + 1) for index, result in enumerate(ordered_results) if result is None]
    if missing_indices:
        raise RuntimeError(f"Missing parallel results for file index: {', '.join(missing_indices)}")
    return [result for result in ordered_results if result is not None]


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

    # visualization_png 字段相对于 placements 上级目录（即 free_bbox 输出根）
    vis_root = placements_dir.parent
    workers = min(args.workers, len(placement_files))
    if workers > 1:
        conflicts = find_cross_file_heatmap_output_conflicts(placement_files, vis_root)
        if conflicts:
            conflict_path, first_file, second_file = conflicts[0]
            print(
                "检测到跨文件 filtered heatmap 输出重名；为保证结果与串行一致，自动切换为 1 个 worker。"
            )
            print(f"示例冲突: {conflict_path} <- {first_file.name}, {second_file.name}")
            workers = 1

    print(f"找到 {len(placement_files)} 个 placements 文件，使用 {workers} 个 worker 开始标注...")
    all_labels: List[dict] = []
    skipped: List[str] = []
    filter_stats = build_filter_stats()

    if workers == 1:
        for idx, pf in enumerate(placement_files, 1):
            result = process_placement_file(pf, dataset_dir, mapping_data, vis_root)
            merge_process_result(result, all_labels, skipped, filter_stats)
            if idx % 50 == 0:
                print(f"已处理 {idx}/{len(placement_files)} 个文件，累计 {len(all_labels)} 条标注")
    else:
        results = run_parallel_labeling(placement_files, dataset_dir, mapping_data, vis_root, workers)
        for result in results:
            merge_process_result(result, all_labels, skipped, filter_stats)
        print(f"按原始文件顺序合并完成，累计 {len(all_labels)} 条标注")

    all_labels_path = output_dir / "all_labels.json"
    with all_labels_path.open("w", encoding="utf-8") as f:
        json.dump(all_labels, f, indent=2, ensure_ascii=False)
    stats_path = output_dir / "auto_label_filter_stats.json"
    with stats_path.open("w", encoding="utf-8") as f:
        json.dump(filter_stats, f, indent=2, ensure_ascii=False)

    if all_labels:
        report_path = output_dir / "report.html"
        with report_path.open("w", encoding="utf-8") as f:
            f.write(build_report_html(output_dir, vis_root, all_labels))
        print(f"✅ HTML 报告: {report_path}")

    if skipped:
        print(f"⚠️ 跳过 {len(skipped)} 个找不到 canonical sample 的文件: {', '.join(skipped[:10])}")
    print(f"✅ 过滤统计: {stats_path}")
    print(f"✅ 标注完成，共 {len(all_labels)} 条 -> {all_labels_path}")


if __name__ == "__main__":
    main()

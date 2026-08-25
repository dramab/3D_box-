#!/usr/bin/env python
"""
Export a static benchmark web report for LC-BGPlaceNet Stage 2 predictions.

使用示例:
    python tools/export_lc_bgplacenet_stage2_benchmark_web.py \
        --config configs/lc_bgplacenet_stage2.yaml \
        --input-dir outputs/lc_bgplacenet_stage2/inference_stage2_test \
        --benchmark-dir outputs/lc_bgplacenet_stage2/benchmark_stage2_test \
        --split test
"""

from __future__ import annotations

import argparse
import html
import json
import math
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
MPL_CONFIG_DIR = PROJECT_ROOT / "outputs" / ".matplotlib"
MPL_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", os.fspath(MPL_CONFIG_DIR))

from src.annotation.auto_label import convex_hull_xy
from src.annotation.free_bbox.geometry import get_bbox_corners, transform_points
from tools.benchmark_lc_bgplacenet_stage2 import (
    _build_item_lookup,
    load_config,
    place_box_to_corners,
    resolve_config_paths,
)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Export Stage 2 benchmark web visualizations.")
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs/lc_bgplacenet_stage2.yaml")
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=PROJECT_ROOT / "outputs/lc_bgplacenet_stage2/inference_stage2_test",
        help="Stage 2 inference output directory.",
    )
    parser.add_argument(
        "--benchmark-dir",
        type=Path,
        default=PROJECT_ROOT / "outputs/lc_bgplacenet_stage2/benchmark_stage2_test",
        help="Directory containing benchmark_metrics.json and per_sample_metrics.jsonl.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory. Defaults to benchmark-dir/web_vis.",
    )
    parser.add_argument("--split", choices=("train", "valid", "val", "test"), default="test")
    parser.add_argument("--max-samples", type=int, default=None, help="Limit exported samples.")
    parser.add_argument("--topdown-size", type=int, default=520, help="Top-down diagnostic image size in pixels.")
    return parser.parse_args()


def _resolve_project_path(path: str | os.PathLike[str]) -> Path:
    """Resolve relative paths against PROJECT_ROOT."""
    p = Path(path)
    return p if p.is_absolute() else PROJECT_ROOT / p


def load_predictions(path: Path) -> dict[str, dict[str, Any]]:
    """Load predictions keyed by item_id."""
    with path.open("r", encoding="utf-8") as f:
        rows = json.load(f)
    return {str(row["item_id"]): row for row in rows}


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    """Load newline-delimited JSON rows."""
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_scene_objects(item: Any, cache: dict[tuple[str, str], dict[str, dict[str, Any]]]) -> dict[str, dict[str, Any]]:
    """Load and cache scene object records from free_bbox placements."""
    key = (item.source_name, item.sample_id)
    if key not in cache:
        placement_path = item.free_bbox_dir / "placements" / f"{item.sample_id}__placements.json"
        with placement_path.open("r", encoding="utf-8") as f:
            payload = json.load(f)
        cache[key] = {str(obj["object_id"]): obj for obj in payload.get("objects", [])}
    return cache[key]


def object_corners_from_record(record: dict[str, Any]) -> np.ndarray:
    """Convert one placement object record to world corners."""
    return transform_points(
        get_bbox_corners(np.asarray(record["canonical_aabb_object"], dtype=np.float64)),
        np.asarray(record["original_pose_world"], dtype=np.float64),
    )


def _footprint(corners_world: np.ndarray) -> np.ndarray:
    """Return a stable XY convex footprint from 3D box corners."""
    hull = convex_hull_xy(np.asarray(corners_world, dtype=np.float64))
    return hull.astype(np.float64)


def _center_xy(poly: np.ndarray) -> np.ndarray:
    """Return the center of a 2D polygon footprint."""
    return np.asarray(poly, dtype=np.float64).mean(axis=0)


def _draw_arrow(draw: ImageDraw.ImageDraw, start: tuple[float, float], end: tuple[float, float], fill: tuple[int, int, int, int]) -> None:
    """Draw a simple line arrow."""
    draw.line([start, end], fill=fill, width=3)
    angle = math.atan2(end[1] - start[1], end[0] - start[0])
    length = 12.0
    spread = 0.55
    left = (end[0] - length * math.cos(angle - spread), end[1] - length * math.sin(angle - spread))
    right = (end[0] - length * math.cos(angle + spread), end[1] - length * math.sin(angle + spread))
    draw.polygon([end, left, right], fill=fill)


def render_topdown(
    output_path: Path,
    pred_box: np.ndarray,
    gt_box: np.ndarray,
    scene_objects: dict[str, dict[str, Any]],
    metrics: dict[str, Any],
    size: int,
) -> None:
    """Render one XY top-down diagnostic image for pred/GT/reference/collision boxes."""
    pred_poly = _footprint(place_box_to_corners(pred_box))
    gt_poly = _footprint(place_box_to_corners(gt_box))
    ref_id = str(metrics.get("reference_object_id", ""))
    collision_ids = [str(obj_id) for obj_id in metrics.get("collision_object_ids", [])]

    object_polys: dict[str, np.ndarray] = {}
    for obj_id, obj_record in scene_objects.items():
        object_polys[obj_id] = _footprint(object_corners_from_record(obj_record))

    polys = [pred_poly, gt_poly, *object_polys.values()]
    all_xy = np.vstack(polys)
    min_xy = all_xy.min(axis=0)
    max_xy = all_xy.max(axis=0)
    span = np.maximum(max_xy - min_xy, 1e-6)
    margin = max(float(span.max()) * 0.08, 1.0)
    min_xy -= margin
    max_xy += margin
    span = np.maximum(max_xy - min_xy, 1e-6)

    def to_px(poly: np.ndarray) -> list[tuple[float, float]]:
        x = (poly[:, 0] - min_xy[0]) / span[0] * (size - 1)
        y = (1.0 - (poly[:, 1] - min_xy[1]) / span[1]) * (size - 1)
        return [(float(px), float(py)) for px, py in zip(x, y)]

    image = Image.new("RGBA", (size, size), (255, 255, 255, 255))
    draw = ImageDraw.Draw(image, "RGBA")
    font = ImageFont.load_default()

    for obj_id, poly in object_polys.items():
        points = to_px(poly)
        draw.polygon(points, fill=(229, 231, 235, 70), outline=(156, 163, 175, 160))
        center = _center_xy(np.asarray(points, dtype=np.float64))
        draw.text((float(center[0] + 3), float(center[1] + 3)), obj_id, fill=(107, 114, 128, 210), font=font)

    if ref_id in object_polys:
        points = to_px(object_polys[ref_id])
        draw.polygon(points, fill=(37, 99, 235, 60), outline=(37, 99, 235, 255))

    for obj_id in collision_ids:
        if obj_id in object_polys:
            points = to_px(object_polys[obj_id])
            draw.polygon(points, fill=(245, 158, 11, 85), outline=(180, 83, 9, 255))

    draw.polygon(to_px(gt_poly), fill=(22, 163, 74, 60), outline=(22, 101, 52, 255))
    draw.polygon(to_px(pred_poly), fill=(220, 38, 38, 60), outline=(185, 28, 28, 255))

    if ref_id in object_polys:
        ref_center = tuple(_center_xy(np.asarray(to_px(object_polys[ref_id]), dtype=np.float64)))
        pred_center = tuple(_center_xy(np.asarray(to_px(pred_poly), dtype=np.float64)))
        _draw_arrow(draw, ref_center, pred_center, (185, 28, 28, 230))

    legend = [
        ("pred", (185, 28, 28, 255)),
        ("gt", (22, 101, 52, 255)),
        ("ref", (37, 99, 235, 255)),
        ("collision", (180, 83, 9, 255)),
    ]
    x0, y0 = 10, 10
    for index, (label, color) in enumerate(legend):
        y = y0 + index * 18
        draw.rectangle([x0, y, x0 + 12, y + 12], fill=color)
        draw.text((x0 + 17, y - 1), label, fill=(17, 24, 39, 255), font=font)

    image.convert("RGB").save(output_path)


def _status_text(ok: bool) -> str:
    """Return compact status text."""
    return "PASS" if bool(ok) else "FAIL"


def collect_rows(
    cfg: dict[str, Any],
    split: str,
    input_dir: Path,
    benchmark_dir: Path,
    output_dir: Path,
    topdown_size: int,
    max_samples: int | None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Join predictions, benchmark rows, scene geometry, and rendered assets."""
    predictions = load_predictions(input_dir / "predictions.json")
    metrics_rows = load_jsonl(benchmark_dir / "per_sample_metrics.jsonl")
    with (benchmark_dir / "benchmark_metrics.json").open("r", encoding="utf-8") as f:
        summary = json.load(f)

    item_by_id = _build_item_lookup(cfg, split)
    scene_cache: dict[tuple[str, str], dict[str, dict[str, Any]]] = {}
    asset_dir = output_dir / "assets"
    asset_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    for metrics in metrics_rows:
        if max_samples is not None and len(rows) >= int(max_samples):
            break
        item_id = str(metrics["item_id"])
        pred = predictions.get(item_id)
        item = item_by_id.get(item_id)
        if pred is None or item is None:
            continue

        scene_objects = load_scene_objects(item, scene_cache)
        topdown_png = asset_dir / f"{item_id}__topdown.png"
        render_topdown(
            output_path=topdown_png,
            pred_box=np.asarray(pred["place_box"], dtype=np.float64),
            gt_box=np.asarray(pred["place_box_gt"], dtype=np.float64),
            scene_objects=scene_objects,
            metrics=metrics,
            size=int(topdown_size),
        )

        prediction_png = _resolve_project_path(pred["visualization_png"])
        search_text = " ".join(
            [
                item_id,
                str(pred.get("instruction", "")),
                str(metrics.get("source_name", "")),
                str(metrics.get("sample_id", "")),
                str(metrics.get("object_id", "")),
                str(metrics.get("target_relation", "")),
                str(metrics.get("predicted_relation", "")),
                str(metrics.get("reference_name", "")),
                " ".join(str(obj_id) for obj_id in metrics.get("collision_object_ids", [])),
            ]
        ).lower()
        rows.append(
            {
                "item_id": item_id,
                "source_name": str(metrics.get("source_name", "")),
                "sample_id": str(metrics.get("sample_id", "")),
                "object_id": str(metrics.get("object_id", "")),
                "cluster_id": int(metrics.get("cluster_id", -1)),
                "instruction": str(pred.get("instruction", "")),
                "prediction_png": os.path.relpath(prediction_png, output_dir),
                "topdown_png": os.path.relpath(topdown_png, output_dir),
                "size_iou": float(metrics.get("placement_size_iou", 0.0)),
                "size_correct": bool(metrics.get("placement_size_correct", False)),
                "direction_hit": bool(metrics.get("language_relation_correct", False)),
                "target_relation": str(metrics.get("target_relation", "")),
                "predicted_relation": str(metrics.get("predicted_relation", "")),
                "reference_name": str(metrics.get("reference_name", "")),
                "reference_object_id": str(metrics.get("reference_object_id", "")),
                "supported_and_stable": bool(metrics.get("supported_and_stable", False)),
                "yaw_valid": bool(metrics.get("yaw_valid_at_matched_center", False)),
                "placement_success": bool(metrics.get("placement_success_at_1", False)),
                "collision": not bool(metrics.get("collision_free", False)),
                "collision_object_count": len(metrics.get("collision_object_ids", [])),
                "collision_object_ids": [str(obj_id) for obj_id in metrics.get("collision_object_ids", [])],
                "best_heatmap_score": float(pred.get("best_heatmap_score", 0.0)),
                "search": search_text,
            }
        )
    return rows, summary


def write_html(output_path: Path, rows: list[dict[str, Any]], summary: dict[str, Any]) -> None:
    """Write the static benchmark HTML report."""
    overall = summary.get("overall", {})
    source_names = sorted({row["source_name"] for row in rows})
    source_options = "\n".join(
        f'<option value="{html.escape(source, quote=True)}">{html.escape(source)}</option>' for source in source_names
    )
    cards = []
    for row in rows:
        direction_class = "pass" if row["direction_hit"] else "fail"
        size_class = "pass" if row["size_correct"] else "fail"
        collision_class = "fail" if row["collision"] else "pass"
        support_class = "pass" if row["supported_and_stable"] else "fail"
        yaw_class = "pass" if row["yaw_valid"] else "fail"
        placement_class = "pass" if row["placement_success"] else "fail"
        collision_ids = ", ".join(row["collision_object_ids"]) if row["collision_object_ids"] else "none"
        cards.append(
            f"""
<article class="card"
  tabindex="0"
  data-source="{html.escape(row["source_name"], quote=True)}"
  data-direction="{"pass" if row["direction_hit"] else "fail"}"
  data-size="{"pass" if row["size_correct"] else "fail"}"
  data-collision="{"fail" if row["collision"] else "pass"}"
  data-size-iou="{row["size_iou"]:.8f}"
  data-search="{html.escape(row["search"], quote=True)}">
  <header class="card-header">
    <div class="meta">{html.escape(row["item_id"])} · {html.escape(row["sample_id"])} · {html.escape(row["object_id"])} · cluster {row["cluster_id"]}</div>
    <div class="badges">
      <span class="badge {direction_class}">Dir {_status_text(row["direction_hit"])}</span>
      <span class="badge {size_class}">Size {_status_text(row["size_correct"])}</span>
      <span class="badge {support_class}">Support {_status_text(row["supported_and_stable"])}</span>
      <span class="badge {collision_class}">Collision {"YES" if row["collision"] else "NO"}</span>
      <span class="badge {yaw_class}">Yaw {_status_text(row["yaw_valid"])}</span>
      <span class="badge {placement_class}">Placement@1 {_status_text(row["placement_success"])}</span>
    </div>
  </header>
  <div class="visuals">
    <figure>
      <img loading="lazy" src="{html.escape(row["prediction_png"])}" alt="prediction and GT projection">
      <figcaption>预测 / GT 投影</figcaption>
    </figure>
    <figure>
      <img loading="lazy" src="{html.escape(row["topdown_png"])}" alt="top-down benchmark geometry">
      <figcaption>XY 俯视诊断</figcaption>
    </figure>
  </div>
  <p class="instruction">{html.escape(row["instruction"])}</p>
  <dl class="metrics">
    <div><dt>direction</dt><dd>{html.escape(row["predicted_relation"])} → {html.escape(row["target_relation"])}</dd></div>
    <div><dt>reference</dt><dd>{html.escape(row["reference_name"])} ({html.escape(row["reference_object_id"])})</dd></div>
    <div><dt>size IoU</dt><dd>{row["size_iou"]:.4f}</dd></div>
    <div><dt>heatmap</dt><dd>{row["best_heatmap_score"]:.4f}</dd></div>
    <div><dt>collision ids</dt><dd>{html.escape(collision_ids)}</dd></div>
  </dl>
</article>"""
        )

    body = "\n".join(cards)
    html_text = f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>LC-BGPlaceNet Stage 2 Benchmark</title>
  <style>
    :root {{
      color: #111827;
      background: #ffffff;
      font-family: Arial, Helvetica, sans-serif;
    }}
    body {{
      margin: 0;
      background: #ffffff;
    }}
    .page {{
      padding: 14px;
    }}
    .topbar {{
      position: sticky;
      top: 0;
      z-index: 4;
      display: grid;
      grid-template-columns: minmax(260px, 1fr) minmax(320px, 1.4fr);
      gap: 12px;
      padding: 8px 0 12px;
      background: #ffffff;
      border-bottom: 1px solid #e5e7eb;
    }}
    h1 {{
      margin: 0;
      font-size: 20px;
      font-weight: 700;
      letter-spacing: 0;
    }}
    .summary {{
      display: flex;
      flex-wrap: wrap;
      gap: 6px;
      margin-top: 8px;
    }}
    .summary span {{
      border: 1px solid #e5e7eb;
      border-radius: 6px;
      padding: 5px 7px;
      font-size: 12px;
      color: #374151;
      background: #f9fafb;
    }}
    .controls {{
      display: grid;
      grid-template-columns: minmax(180px, 1fr) 132px 132px 132px 150px;
      gap: 8px;
      align-content: start;
    }}
    input, select, button {{
      min-width: 0;
      border: 1px solid #d1d5db;
      border-radius: 6px;
      padding: 8px 9px;
      color: #111827;
      font-size: 13px;
      background: #ffffff;
      outline: none;
    }}
    button {{
      cursor: pointer;
    }}
    input:focus, select:focus, button:hover {{
      border-color: #2563eb;
    }}
    .grid {{
      display: grid;
      grid-template-columns: repeat(auto-fill, minmax(360px, 1fr));
      gap: 10px;
      padding-top: 12px;
    }}
    .card {{
      border: 1px solid #e5e7eb;
      border-radius: 8px;
      padding: 8px;
      background: #ffffff;
      cursor: zoom-in;
    }}
    .card:focus {{
      outline: 2px solid #2563eb;
      outline-offset: 2px;
    }}
    .card-header {{
      display: grid;
      gap: 6px;
    }}
    .meta {{
      color: #6b7280;
      font-size: 10px;
      line-height: 1.25;
      overflow-wrap: anywhere;
    }}
    .badges {{
      display: flex;
      flex-wrap: wrap;
      gap: 5px;
    }}
    .badge {{
      border-radius: 6px;
      padding: 4px 6px;
      font-size: 11px;
      font-weight: 700;
    }}
    .badge.pass {{
      color: #14532d;
      background: #dcfce7;
    }}
    .badge.fail {{
      color: #7f1d1d;
      background: #fee2e2;
    }}
    .visuals {{
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 6px;
      margin-top: 7px;
    }}
    figure {{
      margin: 0;
      border: 1px solid #f0f0f0;
      border-radius: 6px;
      overflow: hidden;
      background: #ffffff;
    }}
    img {{
      display: block;
      width: 100%;
      height: 190px;
      object-fit: contain;
      background: #ffffff;
    }}
    figcaption {{
      padding: 4px 6px;
      border-top: 1px solid #f0f0f0;
      color: #6b7280;
      font-size: 10px;
      line-height: 1.2;
    }}
    .instruction {{
      margin: 7px 0 0;
      font-size: 12px;
      line-height: 1.35;
      overflow-wrap: anywhere;
    }}
    .metrics {{
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 4px 8px;
      margin: 8px 0 0;
      font-size: 11px;
    }}
    .metrics div {{
      min-width: 0;
    }}
    dt {{
      color: #6b7280;
    }}
    dd {{
      margin: 1px 0 0;
      overflow-wrap: anywhere;
    }}
    .hidden {{
      display: none;
    }}
    .modal {{
      position: fixed;
      inset: 0;
      z-index: 20;
      display: flex;
      align-items: center;
      justify-content: center;
      padding: 18px;
      background: rgba(17, 24, 39, 0.72);
    }}
    .modal.hidden {{
      display: none;
    }}
    .modal-panel {{
      width: min(1480px, 96vw);
      max-height: 94vh;
      overflow: auto;
      border-radius: 8px;
      background: #ffffff;
      box-shadow: 0 20px 48px rgba(17, 24, 39, 0.28);
    }}
    .modal-header {{
      position: sticky;
      top: 0;
      z-index: 1;
      display: flex;
      gap: 12px;
      align-items: flex-start;
      justify-content: space-between;
      padding: 10px 12px;
      background: #ffffff;
      border-bottom: 1px solid #e5e7eb;
    }}
    .modal-title {{
      color: #374151;
      font-size: 13px;
      line-height: 1.35;
      overflow-wrap: anywhere;
    }}
    .modal-body {{
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 10px;
      padding: 10px;
    }}
    .modal img {{
      width: 100%;
      height: auto;
      max-height: 76vh;
      object-fit: contain;
      background: #ffffff;
    }}
    .modal-details {{
      margin: 0;
      padding: 0 12px 12px;
      color: #111827;
      font-size: 13px;
      line-height: 1.45;
      overflow-wrap: anywhere;
    }}
    @media (max-width: 980px) {{
      .topbar {{
        grid-template-columns: 1fr;
      }}
      .controls {{
        grid-template-columns: 1fr 1fr;
      }}
      .modal-body {{
        grid-template-columns: 1fr;
      }}
    }}
    @media (max-width: 560px) {{
      .grid {{
        grid-template-columns: 1fr;
      }}
      .controls {{
        grid-template-columns: 1fr;
      }}
      .visuals {{
        grid-template-columns: 1fr;
      }}
      img {{
        height: 220px;
      }}
    }}
  </style>
</head>
<body>
  <main class="page">
    <div class="topbar">
      <div>
        <h1>LC-BGPlaceNet Stage 2 Benchmark</h1>
        <div class="summary">
          <span>{len(rows)} samples</span>
          <span>placement@1 {float(overall.get("placement_success_at_1", 0.0)):.4f}</span>
          <span>placement@5 {float(overall.get("placement_success_at_5", 0.0)):.4f}</span>
          <span>support {float(overall.get("supported_and_stable_rate", 0.0)):.4f}</span>
        </div>
      </div>
      <div class="controls">
        <input id="search" type="search" placeholder="搜索 instruction、item_id、关系、碰撞对象">
        <select id="source-filter" aria-label="source filter">
          <option value="all">all sources</option>
          {source_options}
        </select>
        <select id="status-filter" aria-label="status filter">
          <option value="all">all status</option>
          <option value="direction-fail">direction fail</option>
          <option value="collision-fail">collision yes</option>
          <option value="size-fail">size fail</option>
        </select>
        <select id="sort-mode" aria-label="sort mode">
          <option value="original">original</option>
          <option value="size-asc">size IoU asc</option>
          <option value="direction-first">direction fail first</option>
          <option value="collision-first">collision first</option>
        </select>
        <button id="shuffle" type="button">随机打乱</button>
      </div>
    </div>
    <section id="grid" class="grid">
{body}
    </section>
    <div id="modal" class="modal hidden" role="dialog" aria-modal="true" aria-labelledby="modal-title">
      <div class="modal-panel">
        <div class="modal-header">
          <div id="modal-title" class="modal-title"></div>
          <button id="modal-close" type="button">关闭</button>
        </div>
        <div class="modal-body">
          <figure>
            <img id="modal-prediction" src="" alt="prediction and GT projection large view">
            <figcaption>预测 / GT 投影</figcaption>
          </figure>
          <figure>
            <img id="modal-topdown" src="" alt="top-down benchmark geometry large view">
            <figcaption>XY 俯视诊断</figcaption>
          </figure>
        </div>
        <div id="modal-details" class="modal-details"></div>
      </div>
    </div>
  </main>
  <script>
    const grid = document.getElementById('grid');
    const search = document.getElementById('search');
    const sourceFilter = document.getElementById('source-filter');
    const statusFilter = document.getElementById('status-filter');
    const sortMode = document.getElementById('sort-mode');
    const shuffle = document.getElementById('shuffle');
    const cards = Array.from(document.querySelectorAll('.card'));
    const modal = document.getElementById('modal');
    const modalTitle = document.getElementById('modal-title');
    const modalPrediction = document.getElementById('modal-prediction');
    const modalTopdown = document.getElementById('modal-topdown');
    const modalDetails = document.getElementById('modal-details');
    const modalClose = document.getElementById('modal-close');

    function passesStatus(card) {{
      const mode = statusFilter.value;
      if (mode === 'direction-fail') return card.dataset.direction === 'fail';
      if (mode === 'collision-fail') return card.dataset.collision === 'fail';
      if (mode === 'size-fail') return card.dataset.size === 'fail';
      return true;
    }}

    function applyFilters() {{
      const query = search.value.trim().toLowerCase();
      const source = sourceFilter.value;
      for (const card of cards) {{
        const matchesText = !query || card.dataset.search.includes(query);
        const matchesSource = source === 'all' || card.dataset.source === source;
        card.classList.toggle('hidden', !(matchesText && matchesSource && passesStatus(card)));
      }}
    }}

    function applySort() {{
      const sorted = [...cards];
      if (sortMode.value === 'size-asc') {{
        sorted.sort((a, b) => Number(a.dataset.sizeIou) - Number(b.dataset.sizeIou));
      }} else if (sortMode.value === 'direction-first') {{
        sorted.sort((a, b) => (a.dataset.direction === 'pass') - (b.dataset.direction === 'pass'));
      }} else if (sortMode.value === 'collision-first') {{
        sorted.sort((a, b) => (a.dataset.collision === 'pass') - (b.dataset.collision === 'pass'));
      }}
      for (const card of sorted) grid.appendChild(card);
      applyFilters();
    }}

    function openModal(card) {{
      modalTitle.textContent = card.querySelector('.meta').textContent;
      modalPrediction.src = card.querySelector('.visuals figure:first-child img').src;
      modalTopdown.src = card.querySelector('.visuals figure:last-child img').src;
      modalDetails.innerHTML = card.querySelector('.instruction').outerHTML + card.querySelector('.metrics').outerHTML;
      modal.classList.remove('hidden');
    }}

    function closeModal() {{
      modal.classList.add('hidden');
      modalPrediction.src = '';
      modalTopdown.src = '';
    }}

    search.addEventListener('input', applyFilters);
    sourceFilter.addEventListener('change', applyFilters);
    statusFilter.addEventListener('change', applyFilters);
    sortMode.addEventListener('change', applySort);
    shuffle.addEventListener('click', () => {{
      const shuffled = [...cards];
      for (let i = shuffled.length - 1; i > 0; i -= 1) {{
        const j = Math.floor(Math.random() * (i + 1));
        [shuffled[i], shuffled[j]] = [shuffled[j], shuffled[i]];
      }}
      for (const card of shuffled) grid.appendChild(card);
      applyFilters();
    }});
    for (const card of cards) {{
      card.addEventListener('click', () => openModal(card));
      card.addEventListener('keydown', (event) => {{
        if (event.key === 'Enter' || event.key === ' ') {{
          event.preventDefault();
          openModal(card);
        }}
      }});
    }}
    modalClose.addEventListener('click', closeModal);
    modal.addEventListener('click', (event) => {{
      if (event.target === modal) closeModal();
    }});
    document.addEventListener('keydown', (event) => {{
      if (event.key === 'Escape' && !modal.classList.contains('hidden')) closeModal();
    }});
  </script>
</body>
</html>
"""
    output_path.write_text(html_text, encoding="utf-8")


def main() -> None:
    """Export assets and the static benchmark web page."""
    args = parse_args()
    cfg = resolve_config_paths(load_config(args.config))
    input_dir = _resolve_project_path(args.input_dir)
    benchmark_dir = _resolve_project_path(args.benchmark_dir)
    output_dir = args.output_dir or (benchmark_dir / "web_vis")
    output_dir = _resolve_project_path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows, summary = collect_rows(
        cfg=cfg,
        split=args.split,
        input_dir=input_dir,
        benchmark_dir=benchmark_dir,
        output_dir=output_dir,
        topdown_size=int(args.topdown_size),
        max_samples=args.max_samples,
    )
    if not rows:
        raise ValueError("No matched predictions and benchmark rows found.")

    html_path = output_dir / "index.html"
    write_html(html_path, rows, summary)
    print(f"Wrote {len(rows)} Stage 2 benchmark web rows to {html_path}")


if __name__ == "__main__":
    main()

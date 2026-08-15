#!/usr/bin/env python
"""
Export a static four-stage SPACE-Former decoder inference report.

使用示例:
    python tools/export_lc_bgplacenet_stage2_inference_web.py \
        --config configs/lc_bgplacenet_stage2.yaml \
        --input-dir outputs/lc_bgplacenet_stage2/inference_stage2_test \
        --split test

    python tools/export_lc_bgplacenet_stage2_inference_web.py \
        --config configs/lc_bgplacenet_stage2.yaml \
        --input-dir outputs/lc_bgplacenet_stage2/inference_stage2_test \
        --split test --max-samples 100
"""

from __future__ import annotations

import argparse
import html
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

os.environ.setdefault("MPLCONFIGDIR", os.fspath(PROJECT_ROOT / "outputs" / ".matplotlib"))

from src.annotation.free_bbox.io_utils import load_ply


SPLIT_ALIASES = {"val": "valid"}
SPLIT_NAMES = {"train", "valid", "test"}


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Export Stage 2 decoder inference web visualizations.")
    parser.add_argument("--config", type=Path, required=True, help="Stage 2 YAML config path.")
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("outputs/lc_bgplacenet_stage2/inference_stage2_test"),
        help="Stage 2 inference output directory.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory. Defaults to input-dir/web_vis.",
    )
    parser.add_argument(
        "--split",
        default="test",
        choices=("train", "valid", "val", "test"),
        help="Split used only as fallback metadata for older prediction rows.",
    )
    parser.add_argument("--max-samples", type=int, default=None, help="Limit exported samples.")
    parser.add_argument("--heatmap-size", type=int, default=640, help="Rendered heatmap image size in pixels.")
    parser.add_argument("--point-radius", type=int, default=1, help="Raster radius for each heatmap point.")
    return parser.parse_args()


def load_config(path: str | Path) -> dict[str, Any]:
    """Load a YAML config file."""
    with Path(path).open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def normalize_split(split: str) -> str:
    """Normalize split aliases used by training and inference CLIs."""
    split_name = SPLIT_ALIASES.get(str(split), str(split))
    if split_name not in SPLIT_NAMES:
        raise ValueError(f"split must be one of train/valid/test, got {split}")
    return split_name


def read_instruction_map(cfg: dict[str, Any], split: str) -> dict[str, dict[str, str]]:
    """Read item_id keyed metadata used when an old prediction omits text fields."""
    split_path = Path(cfg["data"]["split_dir"]) / f"{normalize_split(split)}.json"
    if not split_path.is_absolute():
        split_path = PROJECT_ROOT / split_path
    with split_path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    return {
        str(item["item_id"]): {
            "instruction": str(item.get("instruction", "")),
            "sample_id": str(item.get("sample_id", "")),
            "object_id": str(item.get("object_id", "")),
        }
        for item in payload.get("items", [])
    }


def render_heatmap_topdown(ply_path: Path, output_path: Path, size: int, point_radius: int) -> None:
    """Render a colored PLY heatmap as a simple XY top-down raster image."""
    points, colors = load_ply(ply_path)
    canvas = np.full((size, size, 3), 255, dtype=np.uint8)
    if len(points) == 0:
        Image.fromarray(canvas).save(output_path)
        return

    xy = points[:, :2].astype(np.float32)
    min_xy = xy.min(axis=0)
    max_xy = xy.max(axis=0)
    span = np.maximum(max_xy - min_xy, 1e-6)
    margin = max(float(span.max()) * 0.04, 1e-3)
    min_xy -= margin
    max_xy += margin
    span = np.maximum(max_xy - min_xy, 1e-6)
    px = np.rint((xy[:, 0] - min_xy[0]) / span[0] * (size - 1)).astype(np.int32)
    py = np.rint((1.0 - (xy[:, 1] - min_xy[1]) / span[1]) * (size - 1)).astype(np.int32)
    radius = max(int(point_radius), 0)

    # 用少量 offset 扩大点的可见面积，避免逐点画图。
    for dx in range(-radius, radius + 1):
        for dy in range(-radius, radius + 1):
            x = px + dx
            y = py + dy
            valid = (x >= 0) & (x < size) & (y >= 0) & (y < size)
            canvas[y[valid], x[valid]] = colors[valid]
    Image.fromarray(canvas).save(output_path)


def _resolve_prediction_path(path_value: str | os.PathLike[str]) -> Path:
    """Resolve inference artifact paths written relative to the project root."""
    path = Path(path_value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def collect_rows(
    input_dir: Path,
    output_dir: Path,
    instructions: dict[str, dict[str, str]],
    heatmap_size: int,
    point_radius: int,
    max_samples: int | None,
) -> list[dict[str, Any]]:
    """Read predictions.json and collect four decoder stages plus final diagnostics."""
    predictions_path = input_dir / "predictions.json"
    with predictions_path.open("r", encoding="utf-8") as f:
        predictions = json.load(f)
    if not isinstance(predictions, list):
        raise ValueError(f"predictions.json must contain a list: {predictions_path}")

    asset_dir = output_dir / "assets"
    asset_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for prediction in predictions:
        if max_samples is not None and len(rows) >= int(max_samples):
            break
        decoder_stages = prediction.get("decoder_stages", [])
        if len(decoder_stages) != 4:
            raise ValueError(
                "Inference output does not contain four decoder stages. "
                "Rerun tools/infer_lc_bgplacenet_stage2.py with the updated code."
            )

        prediction_png = _resolve_prediction_path(prediction["visualization_png"])
        heatmap_ply = _resolve_prediction_path(prediction["pred_heatmap_ply"])
        stage_paths = [_resolve_prediction_path(stage["visualization_png"]) for stage in decoder_stages]
        missing = [path for path in [prediction_png, heatmap_ply, *stage_paths] if not path.exists()]
        if missing:
            raise FileNotFoundError(f"Missing inference visualization artifact: {missing[0]}")

        item_id = str(prediction["item_id"])
        fallback = instructions.get(item_id, {})
        heatmap_png = asset_dir / f"{item_id}__heatmap_topdown.png"
        if not heatmap_png.exists():
            render_heatmap_topdown(heatmap_ply, heatmap_png, heatmap_size, point_radius)
        stages = [
            {
                "stage": int(stage["stage"]),
                "best_place_score": float(stage["best_place_score"]),
                "candidate_count": len(stage.get("placements", [])),
                "image": os.path.relpath(stage_path, output_dir),
            }
            for stage, stage_path in zip(decoder_stages, stage_paths)
        ]
        rows.append(
            {
                "item_id": item_id,
                "sample_id": str(prediction.get("sample_id", fallback.get("sample_id", ""))),
                "object_id": str(prediction.get("object_id", fallback.get("object_id", ""))),
                "cluster_id": str(prediction.get("cluster_id", "")),
                "instruction": str(prediction.get("instruction", fallback.get("instruction", ""))),
                "prediction_png": os.path.relpath(prediction_png, output_dir),
                "heatmap_png": os.path.relpath(heatmap_png, output_dir),
                "stages": stages,
            }
        )
    return rows


def write_html(output_path: Path, rows: list[dict[str, Any]], input_dir: Path) -> None:
    """Write the interactive static decoder comparison report."""
    rows_json = json.dumps(rows, ensure_ascii=False).replace("</", "<\\/")
    html_text = f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>SPACE-Former Decoder 四阶段可视化</title>
  <style>
    :root {{ color: #172033; background: #f4f6f8; font-family: Inter, Arial, sans-serif; }}
    * {{ box-sizing: border-box; }}
    body {{ margin: 0; }}
    .page {{ width: min(1800px, 100%); margin: 0 auto; padding: 18px; }}
    .topbar {{ position: sticky; top: 0; z-index: 3; display: flex; gap: 20px; align-items: center;
      justify-content: space-between; padding: 14px 16px; border: 1px solid #dce2e8; border-radius: 12px;
      background: rgba(255,255,255,.96); box-shadow: 0 8px 24px rgba(31,42,55,.08); }}
    h1 {{ margin: 0; font-size: 21px; letter-spacing: -.02em; }}
    .summary {{ margin-top: 4px; color: #697586; font-size: 12px; }}
    .controls {{ display: flex; gap: 8px; align-items: center; flex-wrap: wrap; }}
    input, select, button {{ border: 1px solid #cbd5df; border-radius: 8px; padding: 9px 11px; color: #172033;
      background: #fff; font: inherit; }}
    input {{ width: min(430px, 42vw); }}
    button, select {{ cursor: pointer; }}
    button:hover, button:focus, input:focus, select:focus {{ border-color: #2563eb; outline: none; }}
    button:disabled {{ cursor: not-allowed; opacity: .45; }}
    .page-status {{ min-width: 140px; color: #475569; font-size: 12px; text-align: center; }}
    .grid {{ display: grid; gap: 14px; padding-top: 14px; }}
    .card {{ overflow: hidden; border: 1px solid #dce2e8; border-radius: 12px; background: #fff; }}
    .card-header {{ display: grid; grid-template-columns: minmax(220px, .7fr) minmax(320px, 1.3fr); gap: 16px;
      align-items: start; padding: 13px 14px; border-bottom: 1px solid #e7ebef; }}
    .meta {{ color: #697586; font: 11px/1.45 ui-monospace, SFMono-Regular, monospace; overflow-wrap: anywhere; }}
    .instruction {{ margin: 0; font-size: 13px; line-height: 1.45; }}
    .stage-grid {{ display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 1px; background: #dce2e8; }}
    figure {{ position: relative; margin: 0; background: #fff; }}
    .stage-figure {{ cursor: zoom-in; }}
    .stage-figure:focus {{ z-index: 1; outline: 3px solid #2563eb; outline-offset: -3px; }}
    .stage-label {{ position: absolute; top: 8px; left: 8px; z-index: 1; padding: 4px 7px; border-radius: 5px;
      color: #fff; background: rgba(23,32,51,.82); font: 700 10px/1 ui-monospace, monospace; letter-spacing: .08em; }}
    img {{ display: block; width: 100%; height: 230px; object-fit: contain; background: #f8fafc; }}
    img:not([src]) {{ background: linear-gradient(90deg, #f8fafc, #eef2f7, #f8fafc); }}
    figcaption {{ padding: 7px 9px; border-top: 1px solid #edf0f2; color: #5d6978; font-size: 11px; }}
    details {{ border-top: 1px solid #e7ebef; }}
    summary {{ padding: 10px 14px; color: #475569; font-size: 12px; cursor: pointer; }}
    .diagnostic-grid {{ display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 1px; background: #dce2e8; }}
    .diagnostic-grid img {{ height: 280px; }}
    .hidden {{ display: none !important; }}
    .modal {{ position: fixed; inset: 0; z-index: 10; display: grid; place-items: center; padding: 18px;
      background: rgba(15,23,42,.78); }}
    .modal-panel {{ width: min(1200px, 96vw); max-height: 95vh; overflow: auto; border-radius: 12px; background: #fff;
      box-shadow: 0 24px 70px rgba(0,0,0,.35); }}
    .modal-header {{ display: flex; gap: 12px; align-items: center; justify-content: space-between; padding: 10px 12px;
      border-bottom: 1px solid #e5e7eb; }}
    .stage-buttons {{ display: flex; gap: 6px; flex-wrap: wrap; }}
    .stage-button.active {{ border-color: #2563eb; color: #fff; background: #2563eb; }}
    .modal img {{ height: min(72vh, 820px); }}
    .modal-caption {{ padding: 10px 14px 14px; color: #475569; font-size: 13px; text-align: center; }}
    @media (max-width: 1050px) {{ .stage-grid {{ grid-template-columns: repeat(2, minmax(0, 1fr)); }} }}
    .empty {{ padding: 42px 16px; border: 1px solid #dce2e8; border-radius: 12px; background: #fff; color: #697586;
      text-align: center; }}
    @media (max-width: 680px) {{
      .page {{ padding: 8px; }} .topbar, .controls {{ align-items: stretch; flex-direction: column; }} input, select {{ width: 100%; }}
      .card-header, .stage-grid, .diagnostic-grid {{ grid-template-columns: 1fr; }} img {{ height: auto; max-height: 58vh; }}
    }}
  </style>
</head>
<body>
  <main class="page">
    <div class="topbar">
      <div><h1>SPACE-Former Decoder 四阶段可视化</h1>
        <div class="summary">{len(rows)} samples · source: {html.escape(os.fspath(input_dir))}</div></div>
      <div class="controls"><input id="search" type="search" placeholder="搜索 instruction、item_id、object_id">
        <select id="page-size" aria-label="每页样本数"><option value="12">12 / 页</option><option value="24" selected>24 / 页</option><option value="48">48 / 页</option></select>
        <button id="prev-page" type="button">上一页</button><div id="page-status" class="page-status"></div>
        <button id="next-page" type="button">下一页</button><button id="shuffle" type="button">随机打乱</button></div>
    </div>
    <section id="grid" class="grid"></section>
    <div id="modal" class="modal hidden" role="dialog" aria-modal="true" aria-labelledby="modal-caption">
      <div class="modal-panel">
        <div class="modal-header"><div id="stage-buttons" class="stage-buttons"></div>
          <button id="modal-close" type="button">关闭</button></div>
        <img id="modal-image" src="" alt="decoder stage large view">
        <div id="modal-caption" class="modal-caption"></div>
      </div>
    </div>
  </main>
  <script id="rows-data" type="application/json">{rows_json}</script>
  <script>
    const rows = JSON.parse(document.getElementById('rows-data').textContent);
    const grid = document.getElementById('grid');
    const search = document.getElementById('search');
    const pageSize = document.getElementById('page-size');
    const pageStatus = document.getElementById('page-status');
    const prevPage = document.getElementById('prev-page');
    const nextPage = document.getElementById('next-page');
    const modal = document.getElementById('modal');
    const modalImage = document.getElementById('modal-image');
    const modalCaption = document.getElementById('modal-caption');
    const stageButtons = document.getElementById('stage-buttons');
    let visibleRows = rows.map(row => ({{
      ...row,
      searchText: [row.item_id, row.sample_id, row.object_id, row.cluster_id, row.instruction].join(' ').toLowerCase()
    }}));
    let pageIndex = 0;
    let activeImages = [];
    let activeCaptions = [];
    let activeStage = 0;
    const imageObserver = 'IntersectionObserver' in window ? new IntersectionObserver(entries => {{
      entries.forEach(entry => {{
        if (!entry.isIntersecting) return;
        loadImage(entry.target);
        imageObserver.unobserve(entry.target);
      }});
    }}, {{ rootMargin: '180px' }}) : null;

    function escapeHtml(value) {{
      return String(value).replace(/[&<>"']/g, char => ({{
        '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;'
      }}[char]));
    }}
    function loadImage(image) {{
      if (image.dataset.src && !image.src) image.src = image.dataset.src;
    }}
    function watchLazyImages(root) {{
      root.querySelectorAll('img[data-src]').forEach(image => {{
        if (image.closest('details:not([open])')) return;
        if (imageObserver) imageObserver.observe(image);
        else loadImage(image);
      }});
    }}
    function stageFigure(stage, index) {{
      return `<figure class="stage-figure" tabindex="0" role="button" data-stage-index="${{index}}"
          aria-label="放大 Decoder ${{stage.stage}}">
        <div class="stage-label">DECODER ${{stage.stage}}</div>
        <img loading="lazy" data-src="${{escapeHtml(stage.image)}}" alt="Decoder ${{stage.stage}} candidates">
        <figcaption>${{stage.candidate_count}} 个候选 · best ${{stage.best_place_score.toFixed(4)}}</figcaption>
      </figure>`;
    }}
    function rowCard(row) {{
      return `<article class="card">
        <header class="card-header">
          <div class="meta">${{escapeHtml(row.item_id)}} · ${{escapeHtml(row.sample_id)}} · ${{escapeHtml(row.object_id)}} · cluster ${{escapeHtml(row.cluster_id)}}</div>
          <p class="instruction">${{escapeHtml(row.instruction)}}</p>
        </header>
        <div class="stage-grid">${{row.stages.map(stageFigure).join('')}}</div>
        <details class="diagnostics">
          <summary>查看最终输出与 P3 粗区域</summary>
          <div class="diagnostic-grid">
            <figure>
              <img loading="lazy" data-src="${{escapeHtml(row.prediction_png)}}" alt="final prediction">
              <figcaption>Benchmark 使用的最终 top-1 输出</figcaption>
            </figure>
            <figure>
              <img loading="lazy" data-src="${{escapeHtml(row.heatmap_png)}}" alt="P3 region heatmap top-down">
              <figcaption>P3 coarse region probability · XY 俯视</figcaption>
            </figure>
          </div>
        </details>
      </article>`;
    }}
    function bindStageFigures() {{
      grid.querySelectorAll('.stage-figure').forEach(figure => {{
        const activate = () => openModal(figure.closest('.card').rowData, Number(figure.dataset.stageIndex));
        figure.addEventListener('click', activate);
        figure.addEventListener('keydown', event => {{
          if (event.key === 'Enter' || event.key === ' ') {{ event.preventDefault(); activate(); }}
        }});
      }});
    }}
    function renderPage() {{
      const size = Number(pageSize.value);
      const pageCount = Math.max(1, Math.ceil(visibleRows.length / size));
      pageIndex = Math.min(pageIndex, pageCount - 1);
      const start = pageIndex * size;
      const pageRows = visibleRows.slice(start, start + size);
      if (pageRows.length === 0) {{
        grid.innerHTML = '<div class="empty">没有匹配样本</div>';
      }} else {{
        grid.innerHTML = pageRows.map(rowCard).join('');
        Array.from(grid.children).forEach((card, index) => {{ card.rowData = pageRows[index]; }});
        bindStageFigures();
        watchLazyImages(grid);
        grid.querySelectorAll('details').forEach(details => {{
          details.addEventListener('toggle', () => {{ if (details.open) watchLazyImages(details); }});
        }});
      }}
      pageStatus.textContent = `${{visibleRows.length ? pageIndex + 1 : 0}} / ${{pageCount}} · ${{visibleRows.length}} samples`;
      prevPage.disabled = pageIndex <= 0;
      nextPage.disabled = pageIndex >= pageCount - 1 || visibleRows.length === 0;
    }}
    function applySearch() {{
      const query = search.value.trim().toLowerCase();
      visibleRows = rows.filter(row => {{
        const text = row.searchText || [row.item_id, row.sample_id, row.object_id, row.cluster_id, row.instruction].join(' ').toLowerCase();
        row.searchText = text;
        return !query || text.includes(query);
      }});
      pageIndex = 0;
      renderPage();
    }}
    function showStage(index) {{
      activeStage = (index + activeImages.length) % activeImages.length;
      modalImage.src = activeImages[activeStage];
      modalCaption.textContent = activeCaptions[activeStage];
      Array.from(stageButtons.children).forEach((button, i) => button.classList.toggle('active', i === activeStage));
    }}
    function openModal(row, index) {{
      activeImages = row.stages.map(stage => stage.image);
      activeCaptions = row.stages.map(stage => `Decoder ${{stage.stage}} · ${{stage.candidate_count}} 个候选 · best ${{stage.best_place_score.toFixed(4)}}`);
      stageButtons.replaceChildren();
      activeImages.forEach((_, i) => {{
        const button = document.createElement('button');
        button.type = 'button'; button.textContent = `Decoder ${{i + 1}}`;
        button.className = 'stage-button'; button.addEventListener('click', () => showStage(i));
        stageButtons.appendChild(button);
      }});
      showStage(index); modal.classList.remove('hidden');
    }}
    function closeModal() {{ modal.classList.add('hidden'); modalImage.src = ''; }}
    search.addEventListener('input', applySearch);
    pageSize.addEventListener('change', () => {{ pageIndex = 0; renderPage(); }});
    prevPage.addEventListener('click', () => {{ pageIndex -= 1; renderPage(); }});
    nextPage.addEventListener('click', () => {{ pageIndex += 1; renderPage(); }});
    document.getElementById('shuffle').addEventListener('click', () => {{
      for (let i = visibleRows.length - 1; i > 0; i -= 1) {{
        const j = Math.floor(Math.random() * (i + 1)); [visibleRows[i], visibleRows[j]] = [visibleRows[j], visibleRows[i]];
      }}
      pageIndex = 0; renderPage();
    }});
    document.getElementById('modal-close').addEventListener('click', closeModal);
    modal.addEventListener('click', event => {{ if (event.target === modal) closeModal(); }});
    document.addEventListener('keydown', event => {{
      if (modal.classList.contains('hidden')) return;
      if (event.key === 'Escape') closeModal();
      if (event.key === 'ArrowLeft') showStage(activeStage - 1);
      if (event.key === 'ArrowRight') showStage(activeStage + 1);
    }});
    renderPage();
  </script>
</body>
</html>
"""
    output_path.write_text(html_text, encoding="utf-8")


def main() -> None:
    """Export heatmap assets and the static four-stage web page."""
    args = parse_args()
    cfg = load_config(args.config)
    input_dir = args.input_dir if args.input_dir.is_absolute() else PROJECT_ROOT / args.input_dir
    output_dir = args.output_dir or (input_dir / "web_vis")
    if not output_dir.is_absolute():
        output_dir = PROJECT_ROOT / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = collect_rows(
        input_dir,
        output_dir,
        read_instruction_map(cfg, args.split),
        int(args.heatmap_size),
        int(args.point_radius),
        args.max_samples,
    )
    if not rows:
        raise ValueError(f"No inference records found in {input_dir / 'predictions.json'}")
    html_path = output_dir / "index.html"
    write_html(html_path, rows, input_dir)
    print(f"Wrote {len(rows)} Stage 2 decoder web visualizations to {html_path}")


if __name__ == "__main__":
    main()

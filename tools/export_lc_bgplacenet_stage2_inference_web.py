#!/usr/bin/env python
"""
Export a white-background HTML report for LC-BGPlaceNet Stage 2 inference.

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
    parser = argparse.ArgumentParser(description="Export Stage 2 inference web visualizations.")
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
        help="Split used to recover instructions from the fixed split file.",
    )
    parser.add_argument("--max-samples", type=int, default=None, help="Limit exported samples.")
    parser.add_argument("--heatmap-size", type=int, default=640, help="Rendered heatmap image size in pixels.")
    parser.add_argument(
        "--point-radius",
        type=int,
        default=1,
        help="Raster radius for each heatmap point in pixels.",
    )
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
    """Read item_id keyed instruction records from the configured split file."""
    split_name = normalize_split(split)
    split_path = Path(cfg["data"]["split_dir"]) / f"{split_name}.json"
    if not split_path.is_absolute():
        split_path = PROJECT_ROOT / split_path
    with split_path.open("r", encoding="utf-8") as f:
        payload = json.load(f)

    records: dict[str, dict[str, str]] = {}
    for item in payload.get("items", []):
        item_id = str(item["item_id"])
        records[item_id] = {
            "instruction": str(item.get("instruction", "")),
            "sample_id": str(item.get("sample_id", "")),
            "object_id": str(item.get("object_id", "")),
        }
    return records


def item_id_from_stem(stem: str) -> str:
    """Extract the stable Stage 2 item_id prefix from an inference output stem."""
    parts = stem.split("__")
    if len(parts) < 3:
        return stem
    return "__".join(parts[:3])


def cluster_id_from_stem(stem: str) -> str:
    """Extract cluster id text from an inference output stem."""
    marker = "__cluster_"
    if marker not in stem:
        return ""
    return stem.rsplit(marker, 1)[-1]


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

    # 用少量 offset 扩大点的可见面积，避免逐点画图带来的大量 Python 循环。
    for dx in range(-radius, radius + 1):
        for dy in range(-radius, radius + 1):
            x = px + dx
            y = py + dy
            valid = (x >= 0) & (x < size) & (y >= 0) & (y < size)
            canvas[y[valid], x[valid]] = colors[valid]

    Image.fromarray(canvas).save(output_path)


def collect_rows(
    input_dir: Path,
    output_dir: Path,
    instructions: dict[str, dict[str, str]],
    heatmap_size: int,
    point_radius: int,
    max_samples: int | None,
) -> list[dict[str, str]]:
    """Collect PNG/PLY pairs and render heatmap thumbnails for the web report."""
    heatmap_dir = input_dir / "pred_heatmaps"
    asset_dir = output_dir / "assets"
    asset_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, str]] = []
    for pred_png in sorted(input_dir.glob("*.png")):
        if max_samples is not None and len(rows) >= int(max_samples):
            break
        stem = pred_png.stem
        ply_path = heatmap_dir / f"{stem}__pred_heatmap.ply"
        if not ply_path.exists():
            continue

        item_id = item_id_from_stem(stem)
        meta = instructions.get(item_id, {})
        heatmap_png = asset_dir / f"{stem}__heatmap_topdown.png"
        if not heatmap_png.exists():
            render_heatmap_topdown(ply_path, heatmap_png, heatmap_size, point_radius)
        rows.append(
            {
                "item_id": item_id,
                "sample_id": meta.get("sample_id", ""),
                "object_id": meta.get("object_id", ""),
                "cluster_id": cluster_id_from_stem(stem),
                "instruction": meta.get("instruction", "Instruction not found in split file."),
                "prediction_png": os.path.relpath(pred_png, output_dir),
                "heatmap_png": os.path.relpath(heatmap_png, output_dir),
            }
        )
    return rows


def write_html(output_path: Path, rows: list[dict[str, str]], input_dir: Path) -> None:
    """Write the static HTML report."""
    cards = []
    for row in rows:
        search_text = " ".join(
            [
                row["item_id"],
                row["sample_id"],
                row["object_id"],
                row["cluster_id"],
                row["instruction"],
            ]
        ).lower()
        cards.append(
            f"""
<article class="card" tabindex="0" data-search="{html.escape(search_text, quote=True)}">
  <header>
    <div class="meta">{html.escape(row["item_id"])} · {html.escape(row["sample_id"])} · {html.escape(row["object_id"])} · cluster {html.escape(row["cluster_id"])}</div>
  </header>
  <div class="visuals">
    <figure>
      <img loading="lazy" src="{html.escape(row["prediction_png"])}" alt="prediction visualization">
      <figcaption>预测图</figcaption>
    </figure>
    <figure>
      <img loading="lazy" src="{html.escape(row["heatmap_png"])}" alt="predicted heatmap top-down view">
      <figcaption>预测 heatmap，XY 俯视</figcaption>
    </figure>
  </div>
  <p class="instruction">{html.escape(row["instruction"])}</p>
</article>"""
        )

    body = "\n".join(cards)
    html_text = f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>LC-BGPlaceNet Stage 2 推理可视化</title>
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
      max-width: none;
      margin: 0 auto;
      padding: 14px;
    }}
    .topbar {{
      position: sticky;
      top: 0;
      z-index: 2;
      display: flex;
      gap: 16px;
      align-items: center;
      justify-content: space-between;
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
      margin-top: 4px;
      color: #6b7280;
      font-size: 13px;
    }}
    input {{
      width: min(420px, 42vw);
      border: 1px solid #d1d5db;
      border-radius: 6px;
      padding: 9px 11px;
      font-size: 14px;
      outline: none;
      background: #ffffff;
    }}
    input:focus {{
      border-color: #2563eb;
    }}
    .controls {{
      display: flex;
      gap: 8px;
      align-items: center;
    }}
    button {{
      border: 1px solid #d1d5db;
      border-radius: 6px;
      padding: 9px 12px;
      color: #111827;
      font-size: 14px;
      background: #ffffff;
      cursor: pointer;
      white-space: nowrap;
    }}
    button:hover {{
      border-color: #2563eb;
    }}
    .grid {{
      display: grid;
      grid-template-columns: repeat(auto-fill, minmax(280px, 1fr));
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
    .meta {{
      color: #6b7280;
      font-size: 10px;
      line-height: 1.25;
      overflow-wrap: anywhere;
    }}
    .instruction {{
      margin: 7px 0 0;
      font-size: 12px;
      line-height: 1.35;
      overflow-wrap: anywhere;
    }}
    .visuals {{
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 6px;
      align-items: start;
      margin-top: 6px;
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
      height: 132px;
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
    .hidden {{
      display: none;
    }}
    .modal {{
      position: fixed;
      inset: 0;
      z-index: 10;
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
      width: min(1380px, 96vw);
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
    .modal-instruction {{
      margin: 0;
      padding: 0 12px 12px;
      color: #111827;
      font-size: 14px;
      line-height: 1.5;
      overflow-wrap: anywhere;
    }}
    @media (max-width: 900px) {{
      .topbar {{
        align-items: stretch;
        flex-direction: column;
      }}
      .controls {{
        align-items: stretch;
        flex-direction: column;
      }}
      input {{
        width: 100%;
        box-sizing: border-box;
      }}
      .visuals {{
        grid-template-columns: repeat(2, minmax(0, 1fr));
      }}
      .modal-body {{
        grid-template-columns: 1fr;
      }}
    }}
  </style>
</head>
<body>
  <main class="page">
    <div class="topbar">
      <div>
        <h1>LC-BGPlaceNet Stage 2 推理可视化</h1>
        <div class="summary">{len(rows)} samples · source: {html.escape(os.fspath(input_dir))}</div>
      </div>
      <div class="controls">
        <input id="search" type="search" placeholder="搜索 instruction、item_id、object_id">
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
            <img id="modal-prediction" src="" alt="prediction visualization large view">
            <figcaption>预测图</figcaption>
          </figure>
          <figure>
            <img id="modal-heatmap" src="" alt="predicted heatmap top-down large view">
            <figcaption>预测 heatmap，XY 俯视</figcaption>
          </figure>
        </div>
        <p id="modal-instruction" class="modal-instruction"></p>
      </div>
    </div>
  </main>
  <script>
    const grid = document.getElementById('grid');
    const search = document.getElementById('search');
    const shuffle = document.getElementById('shuffle');
    const cards = Array.from(document.querySelectorAll('.card'));
    const modal = document.getElementById('modal');
    const modalTitle = document.getElementById('modal-title');
    const modalPrediction = document.getElementById('modal-prediction');
    const modalHeatmap = document.getElementById('modal-heatmap');
    const modalInstruction = document.getElementById('modal-instruction');
    const modalClose = document.getElementById('modal-close');

    function applySearch() {{
      const query = search.value.trim().toLowerCase();
      for (const card of cards) {{
        card.classList.toggle('hidden', query && !card.dataset.search.includes(query));
      }}
    }}

    function openModal(card) {{
      modalTitle.textContent = card.querySelector('.meta').textContent;
      modalPrediction.src = card.querySelector('.visuals figure:first-child img').src;
      modalHeatmap.src = card.querySelector('.visuals figure:last-child img').src;
      modalInstruction.textContent = card.querySelector('.instruction').textContent;
      modal.classList.remove('hidden');
    }}

    function closeModal() {{
      modal.classList.add('hidden');
      modalPrediction.src = '';
      modalHeatmap.src = '';
    }}

    search.addEventListener('input', applySearch);
    shuffle.addEventListener('click', () => {{
      const shuffled = [...cards];
      for (let i = shuffled.length - 1; i > 0; i -= 1) {{
        const j = Math.floor(Math.random() * (i + 1));
        [shuffled[i], shuffled[j]] = [shuffled[j], shuffled[i]];
      }}
      for (const card of shuffled) {{
        grid.appendChild(card);
      }}
      applySearch();
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
      if (event.target === modal) {{
        closeModal();
      }}
    }});
    document.addEventListener('keydown', (event) => {{
      if (event.key === 'Escape' && !modal.classList.contains('hidden')) {{
        closeModal();
      }}
    }});
  </script>
</body>
</html>
"""
    output_path.write_text(html_text, encoding="utf-8")


def main() -> None:
    """Export heatmap assets and the static web page."""
    args = parse_args()
    cfg = load_config(args.config)
    input_dir = args.input_dir
    if not input_dir.is_absolute():
        input_dir = PROJECT_ROOT / input_dir
    output_dir = args.output_dir or (input_dir / "web_vis")
    if not output_dir.is_absolute():
        output_dir = PROJECT_ROOT / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    instructions = read_instruction_map(cfg, args.split)
    rows = collect_rows(
        input_dir=input_dir,
        output_dir=output_dir,
        instructions=instructions,
        heatmap_size=int(args.heatmap_size),
        point_radius=int(args.point_radius),
        max_samples=args.max_samples,
    )
    if not rows:
        raise ValueError(f"No PNG/PLY inference pairs found in {input_dir}")

    html_path = output_dir / "index.html"
    write_html(html_path, rows, input_dir)
    print(f"Wrote {len(rows)} Stage 2 inference web visualizations to {html_path}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""
Generate a static website for inspecting scene-level train/valid/test splits.

使用示例:
    python tools/visualize_splits/generate_split_viewer.py

    python tools/visualize_splits/generate_split_viewer.py \
        --split-dir data/splits/lc_bgplacenet_stage1 \
        --output-dir outputs/split_scene_viewer
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import quote

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SPLIT_NAMES = ("train", "valid", "test")
DEFAULT_SOURCE_DIRS = {
    "dopose": "data/dopose",
    "hope": "data/hope",
    "housecat": "data/housecat",
    "omni": "data/omni_filter",
    "ycbv": "data/ycbv",
}


@dataclass(frozen=True)
class SourcePaths:
    """Project paths for one canonical dataset and its generated outputs."""

    name: str
    dataset_dir: Path
    free_bbox_dir: Path
    labels_path: Path


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Generate a static split visualization website.")
    parser.add_argument(
        "--split-dir",
        type=Path,
        default=Path("data/splits/lc_bgplacenet_stage1"),
        help="Directory containing train.json, valid.json and test.json.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/lc_bgplacenet_stage1.yaml"),
        help="Config used to resolve dataset/output paths.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/split_scene_viewer"),
        help="Directory for index.html, viewer_index.js and scene-level data files.",
    )
    return parser.parse_args()


def project_path(path: str | Path) -> Path:
    """Resolve a project-relative path."""
    value = Path(path)
    return value if value.is_absolute() else PROJECT_ROOT / value


def read_json(path: Path) -> Any:
    """Read one UTF-8 JSON file."""
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_text(path: Path, text: str) -> None:
    """Write UTF-8 text, creating parent directories as needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def load_sources(config_path: Path) -> dict[str, SourcePaths]:
    """Load source paths from the training config, with a small no-yaml fallback."""
    config_abs = project_path(config_path)
    if config_abs.exists():
        try:
            import yaml  # type: ignore

            with config_abs.open("r", encoding="utf-8") as f:
                cfg = yaml.safe_load(f) or {}
            sources = cfg.get("data", {}).get("sources", [])
            if sources:
                return {
                    str(item["name"]): SourcePaths(
                        name=str(item["name"]),
                        dataset_dir=project_path(item["dataset_dir"]),
                        free_bbox_dir=project_path(item["free_bbox_dir"]),
                        labels_path=project_path(item["labels_path"]),
                    )
                    for item in sources
                }
        except Exception as exc:  # pragma: no cover - fallback keeps export usable.
            print(f"[split-viewer] Failed to read config {config_abs}: {exc}", file=sys.stderr)

    return {
        name: SourcePaths(
            name=name,
            dataset_dir=PROJECT_ROOT / dataset_rel,
            free_bbox_dir=PROJECT_ROOT / "outputs" / f"free_bbox_{name}",
            labels_path=PROJECT_ROOT / "outputs" / f"auto_labels_{name}" / "all_labels.json",
        )
        for name, dataset_rel in DEFAULT_SOURCE_DIRS.items()
    }


def guess_scene_id(source_name: str, sample_id: str) -> str:
    """Derive scene_id from canonical sample_id when sample JSON is unavailable."""
    prefix = f"{source_name}__"
    body = sample_id[len(prefix) :] if sample_id.startswith(prefix) else sample_id
    return body.rsplit("__", 1)[0] if "__" in body else body


def relative_asset_url(path: Path, output_dir: Path) -> str:
    """Return a browser-friendly relative URL from the static page to an asset."""
    if not path.exists():
        return ""
    rel = os.path.relpath(path.resolve(), output_dir.resolve())
    return quote(Path(rel).as_posix(), safe="/._-")


def scene_data_path(source_name: str, scene_id: str) -> Path:
    """Build a safe relative path for one scene data JS file."""
    source_part = quote(source_name, safe="")
    scene_part = quote(scene_id, safe="")
    return Path("scenes") / source_part / f"{scene_part}.js"


def scene_cache_key(source_name: str, scene_id: str) -> str:
    """Build a stable key for browser-side scene data cache."""
    return f"{source_name}\u0000{scene_id}"


def row_sort_key(row: dict[str, Any]) -> tuple[str, str, int, str]:
    """Sort rows by frame, sample, label and object for stable display."""
    return (
        str(row.get("frame_id", "")),
        str(row.get("sample_id", "")),
        int(row.get("label_index", -1)),
        str(row.get("object_id", "")),
    )


class StaticSplitExporter:
    """Build the static data payload consumed by the viewer page."""

    def __init__(self, split_dir: Path, sources: dict[str, SourcePaths], output_dir: Path) -> None:
        self.split_dir = project_path(split_dir)
        self.sources = sources
        self.output_dir = project_path(output_dir)
        self.sample_cache: dict[tuple[str, str], dict[str, Any]] = {}
        self.label_cache: dict[str, list[dict[str, Any]]] = {}

    def export(self) -> dict[str, Any]:
        """Write scene-level data files and return the lightweight index payload."""
        scene_rows: dict[str, dict[str, dict[str, Any]]] = {}
        for split_name in SPLIT_NAMES:
            split_path = self.split_dir / f"{split_name}.json"
            payload = read_json(split_path)
            items = payload.get("items", [])
            print(f"[split-viewer] Reading {split_name}: {len(items)} items")
            for item in items:
                row = self._row(split_name, item)
                source = row["source_name"]
                scene = row["scene_id"]
                scene_payload = scene_rows.setdefault(source, {}).setdefault(
                    scene,
                    {"dataset": source, "scene_id": scene, "counts": {}, "splits": {name: [] for name in SPLIT_NAMES}},
                )
                scene_payload["splits"][split_name].append(row)

        datasets = []
        scenes: dict[str, list[dict[str, Any]]] = {}
        for source_name in sorted(scene_rows):
            source_scenes = []
            split_totals = {name: 0 for name in SPLIT_NAMES}
            for scene_id, scene_payload in sorted(scene_rows[source_name].items()):
                for split_name in SPLIT_NAMES:
                    rows = sorted(scene_payload["splits"][split_name], key=row_sort_key)
                    scene_payload["splits"][split_name] = rows
                    scene_payload["counts"][split_name] = len(rows)
                    split_totals[split_name] += len(rows)
                counts = scene_payload["counts"]
                source_scenes.append(
                    {
                        "scene_id": scene_id,
                        "counts": counts,
                        "total": sum(counts.values()),
                        "data_url": scene_data_path(source_name, scene_id).as_posix(),
                    }
                )
            scenes[source_name] = source_scenes
            datasets.append(
                {
                    "name": source_name,
                    "scene_count": len(source_scenes),
                    "split_counts": split_totals,
                }
            )

        for source_name, source_scene_rows in scene_rows.items():
            for scene_id, scene_payload in source_scene_rows.items():
                self._write_scene_data(source_name, scene_id, scene_payload)

        return {
            "schema_version": "split_scene_viewer_static_index/v2",
            "split_dir": os.fspath(self.split_dir.relative_to(PROJECT_ROOT)),
            "datasets": datasets,
            "scenes": scenes,
        }

    def _write_scene_data(self, source_name: str, scene_id: str, payload: dict[str, Any]) -> None:
        """Write one scene payload as a static JS file loadable from file://."""
        rel_path = scene_data_path(source_name, scene_id)
        key = scene_cache_key(source_name, scene_id)
        text = "window.SPLIT_VIEWER_SCENES = window.SPLIT_VIEWER_SCENES || {};\n"
        text += f"window.SPLIT_VIEWER_SCENES[{json.dumps(key, ensure_ascii=False)}] = "
        text += json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
        text += ";\n"
        write_text(self.output_dir / rel_path, text)

    def _row(self, split_name: str, item: dict[str, Any]) -> dict[str, Any]:
        """Build one static display row from a split item."""
        source_name = str(item.get("source_name", ""))
        sample_id = str(item.get("sample_id", ""))
        label_index = int(item.get("label_index", -1))
        sample = self._sample_record(source_name, sample_id)
        label = self._label_record(source_name, label_index)
        source = self.sources.get(source_name)
        scene_id = str(sample.get("scene_id") or guess_scene_id(source_name, sample_id))

        vis_rel = label.get("visualization_png")
        vis_path = source.free_bbox_dir / str(vis_rel) if source and vis_rel else None

        return {
            "split": split_name,
            "item_id": str(item.get("item_id", "")),
            "label_index": label_index,
            "source_name": source_name,
            "scene_id": scene_id,
            "frame_id": str(sample.get("frame_id", "")),
            "sample_id": sample_id,
            "object_id": str(item.get("object_id", "")),
            "class_name": str(label.get("class_name", "")),
            "cluster_id": label.get("cluster_id"),
            "instruction": str(item.get("instruction", "")),
            "vis_url": relative_asset_url(vis_path, self.output_dir) if vis_path else "",
        }

    def _sample_record(self, source_name: str, sample_id: str) -> dict[str, Any]:
        """Read one canonical sample JSON, cached by source and sample id."""
        key = (source_name, sample_id)
        if key in self.sample_cache:
            return self.sample_cache[key]
        source = self.sources.get(source_name)
        sample_path = source.dataset_dir / "samples" / f"{sample_id}.json" if source else None
        record = read_json(sample_path) if sample_path and sample_path.exists() else {}
        self.sample_cache[key] = record
        return record

    def _label_record(self, source_name: str, label_index: int) -> dict[str, Any]:
        """Return the auto-label record matching label_index."""
        if source_name not in self.label_cache:
            source = self.sources.get(source_name)
            self.label_cache[source_name] = read_json(source.labels_path) if source and source.labels_path.exists() else []
        labels = self.label_cache[source_name]
        if 0 <= label_index < len(labels):
            return labels[label_index]
        return {}


INDEX_HTML = r"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Split Scene Viewer</title>
  <style>
    :root {
      color-scheme: light;
      --bg: #f7f8fa;
      --panel: #ffffff;
      --text: #1d2430;
      --muted: #657080;
      --line: #d9dee7;
      --train: #2364aa;
      --valid: #16825d;
      --test: #a54b1a;
    }

    * { box-sizing: border-box; }

    body {
      margin: 0;
      background: var(--bg);
      color: var(--text);
      font: 14px/1.45 system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }

    header {
      position: sticky;
      top: 0;
      z-index: 10;
      border-bottom: 1px solid var(--line);
      background: rgba(255, 255, 255, 0.96);
      backdrop-filter: blur(10px);
    }

    .bar {
      display: grid;
      grid-template-columns: 170px minmax(240px, 1fr) auto;
      gap: 12px;
      align-items: end;
      max-width: 1500px;
      margin: 0 auto;
      padding: 14px 18px;
    }

    label {
      display: block;
      margin-bottom: 5px;
      color: var(--muted);
      font-size: 12px;
      font-weight: 700;
    }

    select,
    input,
    button {
      width: 100%;
      min-height: 38px;
      border: 1px solid var(--line);
      border-radius: 6px;
      background: #fff;
      color: var(--text);
      font: inherit;
    }

    select,
    input { padding: 7px 10px; }

    button {
      min-width: 96px;
      padding: 7px 14px;
      border-color: #1d2430;
      background: #1d2430;
      color: #fff;
      cursor: pointer;
      font-weight: 700;
    }

    main {
      max-width: 1500px;
      margin: 0 auto;
      padding: 16px 18px 32px;
    }

    .status {
      display: flex;
      flex-wrap: wrap;
      gap: 10px;
      align-items: center;
      min-height: 34px;
      color: var(--muted);
    }

    .pill {
      display: inline-flex;
      align-items: center;
      gap: 6px;
      min-height: 26px;
      padding: 3px 9px;
      border: 1px solid var(--line);
      border-radius: 999px;
      background: #fff;
      color: var(--text);
      font-size: 12px;
      font-weight: 700;
    }

    .columns {
      display: grid;
      grid-template-columns: repeat(3, minmax(0, 1fr));
      gap: 14px;
      align-items: start;
    }

    .split {
      min-width: 0;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: var(--panel);
      overflow: hidden;
    }

    .split h2 {
      display: flex;
      justify-content: space-between;
      gap: 12px;
      margin: 0;
      padding: 10px 12px;
      border-bottom: 1px solid var(--line);
      font-size: 15px;
      line-height: 1.2;
    }

    .split.train h2 { color: var(--train); }
    .split.valid h2 { color: var(--valid); }
    .split.test h2 { color: var(--test); }

    .items {
      display: grid;
      gap: 10px;
      padding: 10px;
    }

    .empty {
      padding: 18px 10px;
      color: var(--muted);
      text-align: center;
    }

    .card {
      border: 1px solid var(--line);
      border-radius: 8px;
      background: #fff;
      overflow: hidden;
    }

    .thumb {
      display: grid;
      place-items: center;
      aspect-ratio: 16 / 10;
      background: #eef1f5;
      color: var(--muted);
      overflow: hidden;
      font-size: 12px;
    }

    .thumb img {
      width: 100%;
      height: 100%;
      object-fit: contain;
      background: #fff;
      cursor: zoom-in;
    }

    .body {
      display: grid;
      gap: 6px;
      padding: 9px 10px 10px;
    }

    .meta {
      display: flex;
      flex-wrap: wrap;
      gap: 6px 8px;
      color: var(--muted);
      font-size: 12px;
    }

    .sample {
      font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
      overflow-wrap: anywhere;
    }

    .instruction {
      color: #273242;
      overflow-wrap: anywhere;
    }

    dialog {
      width: min(1180px, 94vw);
      border: 0;
      border-radius: 8px;
      padding: 0;
      background: #fff;
    }

    dialog::backdrop { background: rgba(9, 14, 22, 0.7); }

    .modal-head {
      display: flex;
      justify-content: space-between;
      gap: 12px;
      align-items: center;
      padding: 10px 12px;
      border-bottom: 1px solid var(--line);
    }

    .modal-head button {
      width: auto;
      min-width: 70px;
      min-height: 32px;
    }

    .modal-body {
      max-height: 82vh;
      padding: 10px;
      overflow: auto;
      background: #f4f6f9;
    }

    .modal-body img {
      display: block;
      width: 100%;
      height: auto;
      background: #fff;
    }

    @media (max-width: 980px) {
      .bar { grid-template-columns: 1fr; }
      .columns { grid-template-columns: 1fr; }
      button { min-width: 0; }
    }
  </style>
</head>
<body>
  <header>
    <div class="bar">
      <div>
        <label for="dataset">数据集</label>
        <select id="dataset"></select>
      </div>
      <div>
        <label for="scene">场景</label>
        <input id="scene" list="sceneList" placeholder="输入 scene_id，例如 test_table_000001">
        <datalist id="sceneList"></datalist>
      </div>
      <button id="load">查询</button>
    </div>
  </header>

  <main>
    <div id="status" class="status">正在加载索引...</div>
    <section class="columns">
      <div class="split train">
        <h2><span>train</span><span id="trainCount">0</span></h2>
        <div id="trainItems" class="items"></div>
      </div>
      <div class="split valid">
        <h2><span>valid</span><span id="validCount">0</span></h2>
        <div id="validItems" class="items"></div>
      </div>
      <div class="split test">
        <h2><span>test</span><span id="testCount">0</span></h2>
        <div id="testItems" class="items"></div>
      </div>
    </section>
  </main>

  <dialog id="preview">
    <div class="modal-head">
      <strong id="previewTitle"></strong>
      <button id="closePreview">关闭</button>
    </div>
    <div class="modal-body"><img id="previewImage" alt=""></div>
  </dialog>

  <script src="viewer_index.js"></script>
  <script>
    const data = window.SPLIT_VIEWER_DATA;
    window.SPLIT_VIEWER_SCENES = window.SPLIT_VIEWER_SCENES || {};
    const splits = ["train", "valid", "test"];
    const pendingScenes = new Map();
    const datasetEl = document.getElementById("dataset");
    const sceneEl = document.getElementById("scene");
    const sceneListEl = document.getElementById("sceneList");
    const statusEl = document.getElementById("status");
    const loadEl = document.getElementById("load");
    const previewEl = document.getElementById("preview");
    const previewImageEl = document.getElementById("previewImage");
    const previewTitleEl = document.getElementById("previewTitle");

    function escapeText(value) {
      return String(value ?? "").replace(/[&<>"']/g, ch => ({
        "&": "&amp;", "<": "&lt;", ">": "&gt;", "\"": "&quot;", "'": "&#39;"
      }[ch]));
    }

    function setStatus(parts) {
      statusEl.innerHTML = parts.map(part => `<span class="pill">${escapeText(part)}</span>`).join("");
    }

    function init() {
      datasetEl.innerHTML = data.datasets.map(item => (
        `<option value="${escapeText(item.name)}">${escapeText(item.name)} (${item.scene_count})</option>`
      )).join("");
      datasetEl.addEventListener("change", loadScenes);
      loadEl.addEventListener("click", loadScene);
      sceneEl.addEventListener("keydown", event => {
        if (event.key === "Enter") loadScene();
      });
      document.getElementById("closePreview").addEventListener("click", () => previewEl.close());
      previewEl.addEventListener("click", event => {
        if (event.target === previewEl) previewEl.close();
      });
      loadScenes();
    }

    function loadScenes() {
      const dataset = datasetEl.value;
      const scenes = data.scenes[dataset] || [];
      sceneListEl.innerHTML = scenes.map(scene => {
        const c = scene.counts;
        return `<option value="${escapeText(scene.scene_id)}">train ${c.train} / valid ${c.valid} / test ${c.test}</option>`;
      }).join("");
      sceneEl.value = scenes[0]?.scene_id || "";
      setStatus([`dataset ${dataset}`, `scene ${scenes.length}`]);
      renderScene(emptyScene(dataset, sceneEl.value || ""));
    }

    async function loadScene() {
      const dataset = datasetEl.value;
      const scene = sceneEl.value.trim();
      const meta = findScene(dataset, scene);
      if (!meta) {
        renderScene(emptyScene(dataset, scene));
        return;
      }

      loadEl.disabled = true;
      setStatus([`dataset ${dataset}`, `scene ${scene}`, "正在加载场景数据"]);
      try {
        renderScene(await loadSceneData(dataset, meta));
      } catch (err) {
        console.error(err);
        setStatus([`dataset ${dataset}`, `scene ${scene}`, `加载失败 ${err.message}`]);
      } finally {
        loadEl.disabled = false;
      }
    }

    function findScene(dataset, scene) {
      return (data.scenes[dataset] || []).find(item => item.scene_id === scene);
    }

    function emptyScene(dataset, scene) {
      return {
        dataset,
        scene_id: scene,
        counts: {train: 0, valid: 0, test: 0},
        splits: {train: [], valid: [], test: []}
      };
    }

    function loadSceneData(dataset, meta) {
      const key = `${dataset}\u0000${meta.scene_id}`;
      if (window.SPLIT_VIEWER_SCENES[key]) {
        return Promise.resolve(window.SPLIT_VIEWER_SCENES[key]);
      }
      if (pendingScenes.has(key)) {
        return pendingScenes.get(key);
      }

      const promise = new Promise((resolve, reject) => {
        const script = document.createElement("script");
        script.src = meta.data_url;
        script.onload = () => {
          const payload = window.SPLIT_VIEWER_SCENES[key];
          payload ? resolve(payload) : reject(new Error("场景数据为空"));
        };
        script.onerror = () => reject(new Error(`无法读取 ${meta.data_url}`));
        document.head.appendChild(script);
      }).finally(() => pendingScenes.delete(key));

      pendingScenes.set(key, promise);
      return promise;
    }

    function renderScene(payload) {
      const counts = payload.counts || {};
      setStatus([
        `dataset ${payload.dataset}`,
        `scene ${payload.scene_id}`,
        `train ${counts.train || 0}`,
        `valid ${counts.valid || 0}`,
        `test ${counts.test || 0}`
      ]);
      for (const split of splits) {
        document.getElementById(`${split}Count`).textContent = counts[split] || 0;
        renderSplit(split, payload.splits?.[split] || []);
      }
    }

    function renderSplit(split, rows) {
      const target = document.getElementById(`${split}Items`);
      if (!rows.length) {
        target.innerHTML = `<div class="empty">没有样本</div>`;
        return;
      }
      target.innerHTML = rows.map(row => cardHtml(row)).join("");
      target.querySelectorAll("img[data-preview]").forEach(img => {
        img.addEventListener("click", () => openPreview(img.src, img.dataset.preview));
      });
    }

    function cardHtml(row) {
      const vis = row.vis_url
        ? `<img src="${escapeText(row.vis_url)}" alt="free_bbox" data-preview="${escapeText(row.item_id)} free_bbox">`
        : "无 free_bbox";
      const cluster = row.cluster_id === null || row.cluster_id === undefined ? "" : `cluster ${row.cluster_id}`;
      return `
        <article class="card">
          <div class="thumb">${vis}</div>
          <div class="body">
            <div class="sample">${escapeText(row.sample_id)}</div>
            <div class="meta">
              <span>frame ${escapeText(row.frame_id || "-")}</span>
              <span>${escapeText(row.object_id)}</span>
              <span>${escapeText(row.class_name || "-")}</span>
              <span>${escapeText(cluster)}</span>
              <span>label ${escapeText(row.label_index)}</span>
            </div>
            <div class="instruction">${escapeText(row.instruction)}</div>
          </div>
        </article>
      `;
    }

    function openPreview(src, title) {
      previewImageEl.src = src;
      previewTitleEl.textContent = title || "";
      previewEl.showModal();
    }

    init();
  </script>
</body>
</html>
"""


def main() -> None:
    """Generate index.html, viewer_index.js and scene-level data JS files."""
    args = parse_args()
    output_dir = project_path(args.output_dir)
    exporter = StaticSplitExporter(args.split_dir, load_sources(args.config), output_dir)
    payload = exporter.export()

    write_text(output_dir / "index.html", INDEX_HTML)
    data_js = "window.SPLIT_VIEWER_DATA = "
    data_js += json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    data_js += ";\n"
    write_text(output_dir / "viewer_index.js", data_js)
    write_text(output_dir / "viewer_data.js", "// Deprecated. The viewer now loads viewer_index.js and scenes/*.js lazily.\n")

    index_path = output_dir / "index.html"
    print(f"[split-viewer] Wrote {index_path}")
    print(f"[split-viewer] Open in browser: {index_path.as_uri()}")


if __name__ == "__main__":
    main()

# Split Scene Viewer

这个工具用于生成静态网页，按 `dataset + scene_id` 查看同一场景在固定 `train/valid/test` 切分中的样本。
页面会展示每个 split 下的 free_bbox 可视化图、sample_id、object_id、类别、cluster 和语言指令。

## 生成

```bash
python tools/visualize_splits/generate_split_viewer.py
```

默认读取：

- `data/splits/lc_bgplacenet_stage1/{train,valid,test}.json`
- `configs/lc_bgplacenet_stage1.yaml`
- `data/*/samples/*.json`
- `outputs/free_bbox_*/visualizations/*.png`
- `outputs/auto_labels_*/all_labels.json`

默认输出：

- `outputs/split_scene_viewer/index.html`
- `outputs/split_scene_viewer/viewer_index.js`
- `outputs/split_scene_viewer/scenes/<dataset>/<scene_id>.js`

网页打开时只加载 `viewer_index.js`。输入或选择场景后，页面才会加载对应的
`scenes/<dataset>/<scene_id>.js`，避免一次性解析全部样本。

自定义 split 目录或输出目录：

```bash
python tools/visualize_splits/generate_split_viewer.py \
    --split-dir data/splits/lc_bgplacenet_stage1 \
    --output-dir outputs/split_scene_viewer
```

生成后直接在浏览器打开 `outputs/split_scene_viewer/index.html`。

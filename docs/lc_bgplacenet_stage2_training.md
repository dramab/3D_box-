# LC-BGPlaceNet Stage 2 训练说明

Stage 2 实现 `LC-BGPlaceNet-DPF`：保留 Stage 1 source grounding，并在所有 active voxel 上预测 source-conditioned dense placement field。

Dense Placement Fusion 会先融合 voxel-language feature、坐标编码和 source condition，然后经过 spconv sparse neck 聚合局部体素上下文，再输出 heatmap、bottom offset、yaw 和 size residual。

## 输入监督

训练样本来自 `configs/lc_bgplacenet_stage2.yaml` 中的数据源：

- `point_clouds_voxel_1cm/*.ply`：active voxel 点云，字段为 `(x, y, z, r, g, b)`。
- `outputs/auto_labels_*/all_labels.json`：语言指令、`sample_id`、`object_id` 和 `cluster_id`。
- `outputs/free_bbox_*/placements/*.json`：source box、place box、raw heatmap 和 support mask 路径。
- `outputs/free_bbox_*/direction_filtered_heatmaps/*.ply`：只用于训练的方向过滤 heatmap。
- `outputs/free_bbox_*/support_masks/*.ply`：只用于将 dense heatmap label 限制在支撑区域内。

`place_box_gt` 使用 free_bbox placement 中的 `center_world`、`yaw_only_dimensions` 和 `yaw_degrees` 构造，格式为 `(x, y, z, dx, dy, dz, yaw)`。

## 训练

配置文件不默认绑定 Stage 1 checkpoint，需要通过命令行显式传入：

`model.placement.neck_num_blocks` 控制 Dense Placement Fusion 后的 spconv sparse neck 深度，默认使用 2 个同分辨率 residual SubMConv block，保持 active voxel 顺序与数量不变。

```bash
python tools/train_lc_bgplacenet_stage2.py \
  --config configs/lc_bgplacenet_stage2.yaml \
  --stage1-checkpoint outputs/lc_bgplacenet_stage1_support/best.pt
```

多卡训练：

```bash
torchrun --nproc_per_node=4 tools/train_lc_bgplacenet_stage2.py \
  --config configs/lc_bgplacenet_stage2.yaml \
  --stage1-checkpoint outputs/lc_bgplacenet_stage1_support/best.pt
```

快速 smoke test：

```bash
python tools/train_lc_bgplacenet_stage2.py \
  --config configs/lc_bgplacenet_stage2.yaml \
  --stage1-checkpoint outputs/lc_bgplacenet_stage1_support/best.pt \
  --max-steps 2 --max-train-samples 4 --max-val-samples 2
```

恢复训练：

```bash
python tools/train_lc_bgplacenet_stage2.py \
  --config configs/lc_bgplacenet_stage2.yaml \
  --resume outputs/lc_bgplacenet_stage2/last.pt
```

## 推理

推理阶段只需要点云和语言指令，不读取 direction-filtered heatmap 或 support mask：

```bash
python tools/infer_lc_bgplacenet_stage2.py \
  --config configs/lc_bgplacenet_stage2.yaml \
  --checkpoint outputs/lc_bgplacenet_stage2/best.pt \
  --split valid --max-samples 20
```

输出目录默认为 `outputs/lc_bgplacenet_stage2/inference_stage2_<split>`，包含：

- `predictions.json`：每个样本的 `source_box`、`place_box` 和最高 heatmap 分数。
- `*.png`：RGB 投影可视化，包含预测 source box、预测 place box 和 GT place box。
- `pred_heatmaps/*.ply`：每个样本 active voxel 的预测 placement heatmap，颜色从蓝、青、黄到红表示单样本内相对热度由低到高。

将推理 PNG、预测 heatmap 和输入指令导出为白底缩略图网页：

```bash
python tools/export_lc_bgplacenet_stage2_inference_web.py \
    --config configs/lc_bgplacenet_stage2.yaml \
    --input-dir outputs/lc_bgplacenet_stage2/inference_stage2_test \
    --split test
```

生成的 `index.html` 支持搜索、随机打乱样本顺序，以及点击样本卡片放大查看预测图和 heatmap。

## Test Benchmark

读取 test split 的 `predictions.json`，评估 size 体积 IoU、direction-filtered heatmap 底面中心命中率和非支撑面点云碰撞率：

```bash
python tools/benchmark_lc_bgplacenet_stage2.py \
  --config configs/lc_bgplacenet_stage2.yaml \
  --predictions outputs/lc_bgplacenet_stage2/inference_stage2_test/predictions.json \
  --split test \
  --output-dir outputs/lc_bgplacenet_stage2/benchmark_stage2_test
```

输出包含 `benchmark_metrics.json` 和 `per_sample_metrics.jsonl`。`--size-iou-threshold` 默认 0.8，可按 benchmark 口径调整。

## 监督可视化

只导出 train split 的 Stage 2 监督可视化：

```bash
python tools/export_lc_bgplacenet_stage2_supervision_vis.py \
  --config configs/lc_bgplacenet_stage2.yaml
```

输出目录默认为 `outputs/lc_bgplacenet_stage2/supervision_vis`，包含：

- `rgb_boxes/*.png`：`source_box_gt` 和 `place_box_gt` 的 RGB 投影可视化。
- `gaussian_heatmaps/*.ply`：训练时使用的 support-limited Gaussian heatmap target，半径来自 `data.heatmap_sigma_voxels * data.voxel_size_cm`。
- `index.json`：每个导出样本对应的输入监督路径、输出可视化路径和 box 数值。

## 坐标规范

Stage 2 的底面中心到几何中心转换依赖 canonical world 满足：

```text
world-Z = 支撑面法向/重力上方向
```

如果新增或修改数据集转换脚本，需要继续在转换阶段保证该规范，并在 preprocess 中记录坐标规范化信息。

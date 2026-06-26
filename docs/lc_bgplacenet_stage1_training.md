# LC-BGPlaceNet Stage 1 Training

Stage 1 只训练两个分支：

- Source Grounding：根据语言指令预测源物体 `(cx, cy, cz, l, w, h)`。
- Support Surface：预测每个 active voxel 是否属于支撑面。

本阶段不训练 placement heatmap 和 yaw。Stage 1 训练完成后，先查看验证指标和可视化效果，再决定是否进入 Stage 2。

## 数据输入

点云输入固定使用 canonical sample 中的：

```text
point_clouds_voxel_1cm/*.ply
```

不要把 `point_clouds/*.ply` 直接作为 Stage 1 输入。`support_mask_ply` 是 free_bbox 基于 1cm 体素点云生成的，与 `point_clouds_voxel_1cm` 坐标和尺度最一致。

Support 监督生成规则：

```text
1. 数据索引只保留 all_labels.json 中 visualization_png 实际存在的记录。
2. 读取对应 placements JSON 中的 support_mask_ply。
3. 读取 support_mask_ply 中颜色为白色的点。
4. 对每个 active voxel 点查询最近的白色 support 点。
5. 最近距离 <= 1.5cm 的 active voxel 标为 support=1，否则为 0。
```

## 固定数据划分

Stage 1 使用固定的 `train/valid/test` 清单，不在每次训练时重新随机划分。
划分粒度是 `(source_name, sample_id)`，同一 canonical frame 下的所有物体和语言样本
会落在同一个 split，避免同一帧同时出现在训练和验证/测试中。

首次生成：

```bash
conda run -n spatial python tools/generate_lc_bgplacenet_stage1_splits.py \
    --config configs/lc_bgplacenet_stage1.yaml
```

默认输出：

```text
data/splits/lc_bgplacenet_stage1/
  manifest.json
  train.json
  valid.json
  test.json
```

已有 split 文件时脚本会直接报错，避免误覆盖。确实需要重新划分时显式添加
`--overwrite`。训练和推理默认读取 `configs/lc_bgplacenet_stage1.yaml` 中的
`data.split_dir`，因此生成后不需要每次重新生成。

## 运行

小步验证：

```bash
conda run -n spatial python tools/train_lc_bgplacenet_stage1.py \
    --config configs/lc_bgplacenet_stage1.yaml \
    --max-steps 2 \
    --max-train-samples 4 \
    --max-val-samples 2
```

正式 Stage 1：

```bash
conda run -n spatial python tools/train_lc_bgplacenet_stage1.py \
    --config configs/lc_bgplacenet_stage1.yaml
```

多卡 Stage 1：

```bash
conda run -n spatial torchrun --nproc_per_node=4 tools/train_lc_bgplacenet_stage1.py \
    --config configs/lc_bgplacenet_stage1.yaml
```

`batch_size` 表示每张卡的 batch size。多卡训练时全局 batch size 为 `batch_size * --nproc_per_node`。训练和验证进度条默认开启，可在配置文件中通过 `training.progress_bar` 关闭；DDP 模式下只有 rank 0 显示进度条、写入日志和保存 checkpoint。

输出目录：

```text
outputs/lc_bgplacenet_stage1/
  metrics.jsonl
  last.pt
  best.pt
```

## 推理和 RGB 可视化

推理脚本默认使用固定 `valid` split；也可以显式选择 `train` 或 `test`。
`--split val` 仍可作为 `valid` 的兼容别名。

```bash
conda run -n spatial python tools/infer_lc_bgplacenet_stage1.py \
    --config configs/lc_bgplacenet_stage1.yaml \
    --checkpoint outputs/lc_bgplacenet_stage1/best.pt
```

输出默认保存到：

```text
outputs/lc_bgplacenet_stage1/inference_rgb_valid/
  *.png              # RGB 上投影的预测 source box；默认同时绘制 GT source box
  predictions.jsonl  # 每条样本的预测框、GT 框、IoU 和可视化路径
  summary.json        # 汇总指标
```

注意：Stage 1 只预测 source box 的中心和尺寸，不预测 source yaw。推理可视化中
预测框和 GT 框都使用 canonical sample 中该源物体的 `pose_world` 方向绘制，
`predictions.jsonl` 会记录 `visualization_rotation_source=gt_pose`。

常用参数：

```bash
# 推理全部可用样本
conda run -n spatial python tools/infer_lc_bgplacenet_stage1.py \
    --config configs/lc_bgplacenet_stage1.yaml \
    --checkpoint outputs/lc_bgplacenet_stage1/best.pt \
    --split all

# 推理固定测试集
conda run -n spatial python tools/infer_lc_bgplacenet_stage1.py \
    --config configs/lc_bgplacenet_stage1.yaml \
    --checkpoint outputs/lc_bgplacenet_stage1/best.pt \
    --split test

# 只跑少量样本验证输出
conda run -n spatial python tools/infer_lc_bgplacenet_stage1.py \
    --config configs/lc_bgplacenet_stage1.yaml \
    --checkpoint outputs/lc_bgplacenet_stage1/best.pt \
    --max-samples 8

# 只重画一个样本/物体，便于 debug
conda run -n spatial python tools/infer_lc_bgplacenet_stage1.py \
    --config configs/lc_bgplacenet_stage1.yaml \
    --checkpoint outputs/lc_bgplacenet_stage1/best.pt \
    --sample-id dopose__test_table_000034__000003 \
    --object-id obj_3
```

注意：当前实现使用 `spconv` backbone；本环境中的 `spconv` 前向需要 CUDA。

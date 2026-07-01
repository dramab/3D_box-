# LC-BGPlaceNet Stage 1 Training

Stage 1 只训练两个分支：

- Source Grounding：根据语言指令预测源物体 `(cx, cy, cz, l, w, h)`。
- Support Surface：预测每个 active voxel 是否属于该条指令对应的目标支撑区域。

文本侧使用 **W³ Language Routing**：RoBERTa 输出的 token 特征经三个可学习 role query
池化为 `what/where/whole` 三路角色表征——`what` 注入 Source Grounding 的 query，
`where` 以 FiLM 调制 Support Head，`whole` 作为 `outputs["text_whole"]` 透传给 Stage 2。
`whole` 路在 Stage 1 无直接监督，留待 Stage 2 训练。

点云侧使用 **CamPE（Camera-Relative Positional Encoding）**：因为自动标注生成的
`left/right/front/back` 方位词是按拍摄该帧时的相机视角描述的（世界坐标系的 X/Y 轴
本身与相机朝向无关），所以每个 active voxel 的位置编码改用该帧相机系坐标
`R_w2c @ (point_world - camera_center)`，而不是世界坐标。CamPE 替代了原来的世界系
`pos_mlp`，Source Grounding 的 query 位置编码和 Support Head 的输入统一使用同一份
`pos_embed_cam`。上下方向（top/below）仍由 world Z 轴决定，但 CamPE 本身不单独保留
世界 Z 分量，垂直语义完全由相机系三轴隐式承载。

本阶段不训练 placement heatmap 和 yaw。Stage 1 训练完成后，先查看验证指标和可视化效果，再决定是否进入 Stage 2。

## 数据输入

点云输入固定使用 canonical sample 中的：

```text
point_clouds_voxel_1cm/*.ply
```

不要把 `point_clouds/*.ply` 直接作为 Stage 1 输入。Stage 1 会通过 `all_labels.json` 中的
`placement_sample_id/cluster_id` 回连 free_bbox placement，并使用该 placement 的
`corners_world` 和对应帧的 `support_mask_ply` 生成文本条件目标支撑区域监督。

Stage 1 现在还需要每帧的相机外参 `samples/<sample_id>.json` 中的 `camera.E_c2w`，用于
计算 CamPE（见下文）。5 个数据源（dopose/hope/housecat/omni/ycbv）的 canonical 转换
流程都会写出该字段；如果自定义数据源缺失 `camera.E_c2w`，`build_stage1_index` 会直接
报错，不做静默兜底。

Support 监督生成规则：

```text
1. 数据索引只保留 all_labels.json 中 visualization_png 实际存在的记录。
2. 用 placement_sample_id 优先、cluster_id 兜底，在对应 placements JSON 中找回目标 placement。
3. 读取该 placement 的 corners_world，计算 GT 放置框底面中心和底面 Z。
4. 读取 support_mask_ply 中白色支撑面点，并在底面 Z 附近找出 GT 中心所在的支撑面连通区域。
5. 该连通区域面积记为 A，候选圆面积为 A * support_radius_area_fraction，半径为 sqrt(A * support_radius_area_fraction / pi)。
6. active voxel 同时满足以下条件时标为 support=1，否则为 0：
   - 与最近支撑面白点距离不超过 support_align_threshold_cm。
   - XY 平面上到 GT 底面中心的距离不超过上一步计算出的动态半径。
   - Z 到 GT 底面的距离不超过 support_align_threshold_cm。
```

这个定义监督的是“目标 placement 中心附近、且覆盖当前支撑面固定面积比例的支撑面点”。
半径只由 GT 中心所在支撑面连通区域面积决定，不由目标物体 footprint 决定，也不额外设置上限或下限。

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

W³ 版训练使用专门配置 `configs/lc_bgplacenet_stage1_w3.yaml`（显式 `model.w3` 段，
并把 `lambda_sup` 提到 1.0 侧重 support），输出到 `outputs/lc_bgplacenet_stage1_w3/`：

```bash
conda run -n spatial python tools/train_lc_bgplacenet_stage1.py \
    --config configs/lc_bgplacenet_stage1_w3.yaml
```

模型结构变更（Support Head 改为 FiLM 版、新增 W³ 路由、文本编码器移除 mask-mean 全局向量），
旧 checkpoint 无法直接 resume，需从头训练。旧配置 `lc_bgplacenet_stage1.yaml` 未写 `w3` 段时，
W³ 会退回默认超参（`num_heads=8, dropout=0.1`）仍可加载。

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

## Support GT 3D 可视化

GT 支撑区域点云可视化不需要 checkpoint，也不运行模型推理。脚本直接读取 Stage 1
dataset，按当前 `support_label` 生成逻辑导出该条语言指令对应的目标支撑区域。

```bash
conda run -n spatial python tools/export_lc_bgplacenet_stage1_support_gt.py \
    --config configs/lc_bgplacenet_stage1.yaml \
    --split test \
    --max-samples 32
```

默认输出：

```text
outputs/lc_bgplacenet_stage1/support_gt_pointclouds/
  pointclouds/
    *__support_gt.ply  # 灰色为非 GT support active voxel，橙红色为 GT target support voxel
  records.jsonl        # 每条样本的 PLY 路径、placement id、指令和 GT support 统计
  summary.json
```

如需导出某一条样本：

```bash
conda run -n spatial python tools/export_lc_bgplacenet_stage1_support_gt.py \
    --config configs/lc_bgplacenet_stage1.yaml \
    --split all \
    --sample-id dopose__test_table_000001__000001 \
    --object-id obj_0
```

`--max-samples 0` 表示导出所有匹配样本；默认只导出 32 条，避免一次性写出过多 PLY。

## Support Prediction 3D 可视化

测试集 support 预测点云可视化需要 checkpoint，并会运行模型前向：

```bash
conda run -n spatial python tools/infer_lc_bgplacenet_stage1.py \
    --config configs/lc_bgplacenet_stage1.yaml \
    --checkpoint outputs/lc_bgplacenet_stage1/best.pt \
    --split test \
    --export-pointcloud \
    --no-rgb \
    --output-dir outputs/lc_bgplacenet_stage1/inference_support_mask_3d_test
```

默认输出：

```text
outputs/lc_bgplacenet_stage1/inference_support_mask_3d_test/
  support_pointclouds/
    *__support_pred.ply  # 每条 Stage 1 样本一份；灰色为非 support active voxel，橙红色为预测 support voxel
  predictions.jsonl      # 每条样本的 PLY 路径、support 概率统计和 source box 指标
  summary.json
```

`--support-threshold` 控制 support mask 着色阈值，默认 `0.5`。

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

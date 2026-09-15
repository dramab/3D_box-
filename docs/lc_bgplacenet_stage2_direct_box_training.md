# Direct-Box 1Q训练说明

Direct-Box 1Q是完整模型的单解直接回归baseline。它复用Stage 1的CLIP图像/文本编码、稀疏3D backbone、语言融合和Source Grounding，并使用一个独立Placement Query交叉关注完整体素memory。

## 输入与输出

模型同时输出：

- `source_box`：语言指定的原物体3D框。
- `pred_bottom_centers`：单个预测放置底面中心。
- `raw_yaw_logits`：该中心对应的12-bin yaw logits。
- `place_box`：预测底面中心、预测Source尺寸和预测yaw构造的放置框。

模型不使用Heatmap、LMPQ、Hungarian集合匹配、PABR或物理后处理。因为只有一个Placement Query，`Placement Success@5`与`Placement Success@1`相同。

在相同单Query粗预测上增加PABR的独立消融模型见`docs/lc_bgplacenet_stage2_direct_box_pabr_training.md`；纯Direct-Box配置和输出目录保持不变。

## 数据划分

配置固定读取：

```text
data/splits/active_aligned_enriched
```

该目录是从`active_aligned`逐条映射得到的enriched v1划分。训练代码不会重新随机划分，也不要使用split生成工具覆盖该目录。五个数据源必须读取对应的`all_labels_enriched.json`，保证item ID与划分记录一致。

## 训练目标

每条样本只有一个Placement Query。训练时按GT Source Size归一化中心距离，在全部合法中心中选择离当前预测最近的目标；yaw监督只使用该中心对应的12-bin multi-hot合法集合。

```text
L = 5 L_center + 0.5 L_yaw + 0.5 L_corner + 0.5 L_source
L_source = 4 L_source-center + 2 L_source-size + 0.2 L_source-IoU
```

共享Stage 1参数组使用主学习率的0.1倍，Direct-Box参数组使用完整学习率。学习率调度器为`ReduceLROnPlateau(mode=max)`，监控validation `Placement Success@1`；连续停滞超过3个epoch后将学习率乘以0.5。

## 运行

单卡快速检查：

```bash
CUDA_VISIBLE_DEVICES=0 python tools/train_lc_bgplacenet_stage2.py \
  --config configs/lc_bgplacenet_stage2_direct_box_1q_enriched.yaml \
  --stage1-checkpoint outputs/lc_bgplacenet_stage1_clip16_aligned_enriched/best.pt \
  --max-steps 2 --max-train-samples 8 --max-val-samples 8
```

4卡完整训练（单卡batch 8，全局batch 32，与现有enriched主模型一致）：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 \
  tools/train_lc_bgplacenet_stage2.py \
  --config configs/lc_bgplacenet_stage2_direct_box_1q_enriched.yaml \
  --stage1-checkpoint outputs/lc_bgplacenet_stage1_clip16_aligned_enriched/best.pt
```

恢复训练：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 \
  tools/train_lc_bgplacenet_stage2.py \
  --config configs/lc_bgplacenet_stage2_direct_box_1q_enriched.yaml \
  --resume outputs/lc_bgplacenet_stage2_direct_box_1q_enriched/last.pt
```

训练产物保存在`outputs/lc_bgplacenet_stage2_direct_box_1q_enriched`，其中`best.pt`按validation `Placement Success@1`选择，`last.pt`保存最近一个epoch。

## 测试

```bash
python -m pytest -q tests/test_lc_bgplacenet_stage2_direct_box.py
```

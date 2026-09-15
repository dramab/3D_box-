# Direct-Box 1Q + PABR训练说明

该模型保留Direct-Box 1Q的全局Placement Query，将其粗中心、粗yaw和query feature作为PABR初始状态。PABR使用预测Source Box尺寸，在当前中心和yaw下生成64个Box边界采样点，从P1/P2/P3稀疏场景特征中聚合局部几何，并通过4层残差更新细化中心和yaw。

模型不使用Heatmap、LMPQ、Hungarian集合匹配或物理后处理。单Query始终有效，因此不训练无负样本的placement score，`Placement Success@5`与`Placement Success@1`相同。纯Direct-Box baseline仍使用`direct_box_1q`，本模型使用独立类型`direct_box_1q_pabr`。

## 训练目标

最终预测在全部合法中心中选择最近目标，粗Direct-Box输出和前三层PABR共享该目标：

```text
L_final = 5 L_center + 0.5 L_yaw + 0.5 L_corner + 0.5 L_source
L = L_final + 0.1 mean(L_coarse, L_pabr-1, L_pabr-2, L_pabr-3)
```

共享Stage 1参数使用主学习率的0.1倍，Direct-Box头和PABR使用完整学习率。学习率调度器为`ReduceLROnPlateau(mode=max)`，监控validation `Placement Success@1`，连续停滞超过3个epoch后将学习率乘以0.5。

## 运行

单卡快速检查，从已训练Direct-Box权重初始化已有模块：

```bash
CUDA_VISIBLE_DEVICES=0 python tools/train_lc_bgplacenet_stage2.py \
  --config configs/lc_bgplacenet_stage2_direct_box_1q_pabr_enriched.yaml \
  --stage1-checkpoint outputs/lc_bgplacenet_stage2_direct_box_1q_enriched/best.pt \
  --max-steps 2 --max-train-samples 8 --max-val-samples 8
```

4卡完整训练：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 \
  tools/train_lc_bgplacenet_stage2.py \
  --config configs/lc_bgplacenet_stage2_direct_box_1q_pabr_enriched.yaml \
  --stage1-checkpoint outputs/lc_bgplacenet_stage2_direct_box_1q_enriched/best.pt
```

恢复训练必须使用本模型自己的`last.pt`：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 \
  tools/train_lc_bgplacenet_stage2.py \
  --config configs/lc_bgplacenet_stage2_direct_box_1q_pabr_enriched.yaml \
  --resume outputs/lc_bgplacenet_stage2_direct_box_1q_pabr_enriched/last.pt
```

验证测试：

```bash
python -m pytest -q tests/test_lc_bgplacenet_stage2_direct_box.py
```

推理并导出可视化：

```bash
CUDA_VISIBLE_DEVICES=0 python tools/infer_lc_bgplacenet_stage2.py \
  --config configs/lc_bgplacenet_stage2_direct_box_1q_pabr_enriched.yaml \
  --checkpoint outputs/lc_bgplacenet_stage2_direct_box_1q_pabr_enriched/best.pt \
  --split valid --max-samples 20
```

训练产物保存在`outputs/lc_bgplacenet_stage2_direct_box_1q_pabr_enriched`，其中`best.pt`按validation `Placement Success@1`选择。

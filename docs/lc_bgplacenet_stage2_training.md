# SPACE-Former Stage 2 训练说明

Stage 2 使用 `SPACE-Former`（Size-Prompted Affordance and Collision Explorer）完成有界集合预测。模型复用并联合训练 Stage 1 的 Backbone、语言融合与 Source Grounding；CLIP 延续配置中的冻结状态。

## 数据与监督

每条训练样本读取：

- `point_clouds_voxel_1cm/*.ply`：1 cm active voxel 点云。
- canonical `samples/*.json`：RGB、相机参数和 Source GT。
- `direction_filtered_heatmaps/*__heatmap.ply`：当前语言方向过滤后的合法底面中心。
- `yaw_sets/*__yaw_set.npz`：每个底面中心对应的 24-bin 多 yaw 集合。
- auto-label：自然语言指令及关系参照物。

数据集先按 1 cm world coordinate 对齐方向过滤正点和 yaw set，只保留二者交集。24 个 yaw bin 通过 `mask[:12] OR mask[12:]` 合并为 12 个 180° 等价 bin。GT 中心超过 32 个时使用空间 FPS；不足时由 `gt_valid_mask` padding。

canonical world 必须满足：

```text
world-Z = 支撑面法向 / 重力上方向
```

## 固定模型规模

```text
P1/P2/P3 stride  = 1/2/4
P3 cell          = 4 cm × 4 cm = 16 cm²
coarse top cells = 8（约 128 cm²，不要求连续）
queries          = 32
samples/query    = 64 × 3 scales × 4 layers
feature_dim      = 256
yaw_bins         = 12
max output       = 16
```

P3 粗区域预测仅使用 P3 feature 和 `text_global`。`space_former.num_region_cells` 控制 hard top-K，当前 top-8 P3 cell 展开到其覆盖的 P1 active voxel 后，最多 FPS 采样 32 个 Anchor；候选不足时不扩区，padding Query 由 `query_valid_mask` 屏蔽。

第 0 个 Decoder 层使用 64 点外接圆柱模板；后 3 层使用上一层 yaw 构建 64 点定向 Box Surface。P1/P2/P3 均执行最近 active sparse voxel hash lookup。未命中的零特征 token 不会被 Attention 删除，其 `sample_valid_mask=0` 仍携带“踩空/净空”含义。

## 损失与优化

总损失为：

```text
L = 2 L_region + 2 L_cls + 5 L_center + 2 L_yaw
    + L_corner + L_source + 0.5 L_aux

L_source = 2 L_source-center + 1.5 L_source-size + 0.2 L_source-IoU
```

匹配代价为 `2 C_cls + 5 C_center + 2 C_yaw-bin + C_corner`。Yaw 使用 12 维 multi-hot `BCEWithLogits`，角点项在 GT 的全部有效 yaw 中取最小值。

Stage 1 参数组使用 `training.lr × 0.1`，SPACE-Former 使用完整 `training.lr`。`ReduceLROnPlateau(mode=max)` 监控 validation `task_success_rate`，绝对提升不足 `0.001` 连续停滞超过 3 个 epoch 后将两个参数组学习率同时乘以 `0.5`。最佳 checkpoint 仅按 validation 的 top-1 `task_success_rate` 保存，不设置 Source IoU 门槛。该指标与推理最终采用的位姿一致，成功条件为尺寸 IoU ≥ 0.8、满足文本空间方向关系且与场景已有物体无碰撞。

训练保持 FP32。P1/P2/P3 的 sparse lookup 排序索引在一次 forward 内由四层 Decoder 复用；训练不执行 pose NMS，采样诊断仅在日志 step 计算；每个 batch 的 Hungarian 代价只进行一次 GPU→CPU 传输，仍使用 SciPy 精确匹配。进度条最多每 20 step 同步一次 loss。

## 运行

单卡训练：

```bash
python tools/train_lc_bgplacenet_stage2.py \
  --config configs/lc_bgplacenet_stage2.yaml \
  --stage1-checkpoint outputs/lc_bgplacenet_stage1_clip16/best.pt
```

多卡训练：

```bash
torchrun --nproc_per_node=4 tools/train_lc_bgplacenet_stage2.py \
  --config configs/lc_bgplacenet_stage2.yaml \
  --stage1-checkpoint outputs/lc_bgplacenet_stage1_clip16/best.pt
```

快速检查：

```bash
python tools/train_lc_bgplacenet_stage2.py \
  --config configs/lc_bgplacenet_stage2.yaml \
  --max-steps 2 --max-train-samples 4 --max-val-samples 2
```

恢复训练：

```bash
python tools/train_lc_bgplacenet_stage2.py \
  --config configs/lc_bgplacenet_stage2.yaml \
  --resume outputs/lc_bgplacenet_stage2_clip16/last.pt
```

`training.epochs` 表示总 epoch 上限，而非恢复后追加的 epoch 数。checkpoint 中的模型、AdamW moments 和 scheduler 历史会恢复；旧 checkpoint 没有 scheduler state 时，以 `best_task_success_rate` 初始化平台基线。随后始终由当前配置的 `training.lr` 覆盖 checkpoint 学习率，并保持 Stage 1/Stage 2 为 `0.1:1`。例如恢复已完成 epoch 54 的 checkpoint，若要继续训练，必须将 `training.epochs` 设置为大于 55。

## 日志与验证

`metrics.jsonl` 每个 `log_every` 和 validation epoch 记录：粗区域选择数、P1 候选数、有效/填充 Query 数，以及逐层逐尺度的采样总数、active 数、非零特征数、bottom/non-bottom 命中数和全零 Query 数。每个 epoch 同时记录 `lr_stage1` 与 `lr_stage2`，checkpoint 保存 scheduler state 和调度后的两组学习率。日志均为 detached 聚合值，DDP validation 使用 all-reduce 汇总。终端仅显示 loss、核心子损失和主要验证指标的单行摘要，完整诊断字段仍保存在 `metrics.jsonl`。

主要验证指标包括：

- `task_success_rate`、`task_success_top5`，分别统计 top-1 和前 5 个候选中的任务成功率。
- `valid_pose_rate`、`duplicate_pair_rate`。
- `direction_hit_rate`、`collision_free_rate`、`size_iou`。
- `p3_gt_point_coverage`：每条样本 direction-positive GT 点落入实际 top-8 P3 cell 的比例，再对验证样本宏平均。
- `source_center_mae`、`source_size_iou`。

## 推理

```bash
python tools/infer_lc_bgplacenet_stage2.py \
  --config configs/lc_bgplacenet_stage2.yaml \
  --checkpoint outputs/lc_bgplacenet_stage2_clip16/best.pt \
  --split valid --max-samples 20
```

`predictions.json` 中 `place_box` 保留 top-1 兼容字段，`placements` 保存最多 16 个 `{box, score, yaw_bin}`。`pred_heatmaps/*.ply` 现在可视化 P3 粗区域概率，而非旧版 dense placement heatmap。

## 测试

```bash
pytest -q tests/test_lc_bgplacenet_stage2.py
```

测试覆盖 top-8 区域展开及不足 8 个 cell、P3 GT 覆盖率宏平均、Query padding、64 点模板、缓存 active lookup、Hungarian 匹配、训练 NMS 跳过、学习率恢复与平台衰减、多 yaw 合并、集合输出和 Source 联合反向传播。

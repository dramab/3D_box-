# SPACE-Former Stage 2 训练说明

Stage 2 使用 `SPACE-Former`（Size-Prompted Affordance and Collision Explorer）完成有界集合预测。模型复用并联合训练 Stage 1 的 Backbone、语言融合与 Source Grounding；CLIP 延续配置中的冻结状态。

## 数据与监督

每条训练样本读取：

- `point_clouds_voxel_1cm/*.ply`：1 cm active voxel 点云。
- canonical `samples/*.json`：RGB、相机参数和 Source GT。
- `direction_filtered_heatmaps/*__heatmap.ply`：当前语言方向过滤后的合法底面中心。
- `yaw_sets/*__yaw_set.npz`：每个底面中心对应的 24-bin 多 yaw 集合。
- `support_masks/*.ply`：与输入点云共用 canonical voxel key 的 active support。
- auto-label：自然语言指令及关系参照物。

数据集按 canonical voxel key 严格关联输入点云、active support、方向过滤正点和 yaw set，不执行最近邻吸附或距离容错。24 个 yaw bin 通过 `mask[:12] OR mask[12:]` 合并为 12 个 180° 等价 bin。Decoder GT 保留全部方向合法中心，不再执行 FPS 截断；不同样本仅在 batch 内按当前最大 GT 数量 padding，并由 `gt_valid_mask` 屏蔽填充值。

canonical world 必须满足：

```text
world-Z = 支撑面法向 / 重力上方向
```

Stage 2 复用 `data.split_dir` 指向的 Stage 1 划分文件，读取器兼容 v1 和 v2。v1 仅用于与
旧 Stage 1 checkpoint 保持原始实验划分一致；新实验应使用按 `scene_id` 隔离的 v2，避免
同一场景的不同帧跨越训练、验证和测试集。

## 固定模型规模

```text
P1/P2/P3 stride  = 1/2/4
P3 cell          = 4 cm × 4 cm = 16 cm²
coarse top cells = 8（约 128 cm²，不要求连续）
queries          = 48
samples/query    = 64 × 3 scales × 4 layers
feature_dim      = 256
yaw_bins         = 12
max output       = 16
```

P3 粗区域预测使用 P3 feature 和完整 `text_tokens`。每个有效 token 先查询 P3 memory，再与各 P3 cell 计算 compatibility，并通过带 attention mask 的 log-mean-exp 聚合，使方向词和对象词的强匹配能够主导区域分数。Cross Block 层数由 `model.space_former.region_cross_num_layers` 控制，当前为 4。Region target 由 `direction_filtered_heatmaps` 正点生成三维高斯分布，并在由 P1 active support 聚合得到的 P3 support 外置零；标准差为 `data.heatmap_sigma_voxels × data.voxel_size_cm`，当前为 8 cm。`space_former.num_region_cells` 控制 hard top-K，当前 top-8 P3 cell 展开到其覆盖的 P1 active voxel 后，最多 FPS 采样 48 个 Anchor；support 只约束训练标签，不作为推理输入。候选不足时不扩区，padding Query 由 `query_valid_mask` 屏蔽。

第 0 个 Decoder 层使用 64 点外接圆柱模板；后 3 层使用上一层 yaw 构建 64 点定向 Box Surface。P1/P2/P3 将采样点量化到同格 sparse voxel key 后执行精确 hash lookup。未命中的零特征 token 不会被 Attention 删除，其 `sample_valid_mask=0` 仍携带“踩空/净空”含义。

## 损失与优化

总损失为：

```text
L = 500 L_region + L_cls + 5 L_center + 0.5 L_yaw
    + 0.5 L_corner + 0.5 L_source + 0.1 L_aux

L_source = 4 L_source-center + 2 L_source-size + 0.2 L_source-IoU
```

所有放置框直接复用 Source Size，因此独立的尺寸监督位于 `L_source-size`；当前有效优先级为放置中心、Source 中心、Source 尺寸、Yaw。

Hungarian 在全部方向合法中心和与有效 Query 等量的背景虚拟目标之间执行一对一匹配。真实中心代价为 GT Source Size 归一化后的底面中心 L1 距离，背景代价由 `loss.background_match_cost` 定义，当前为 `0.25`。匹配真实中心的 Query 接受分类、中心、Yaw 和角点监督；匹配背景的 Query 仅接受 placement score 为 0 的分类监督。前三层辅助监督复用最终层匹配并采用 `2:8:1:1` 内部比例，由总损失中的 `0.1 L_aux` 统一缩放。Yaw 使用 12 维 multi-hot `BCEWithLogits`，角点项在 GT 的全部有效 yaw 中取最小值。

Stage 1 参数组使用 `training.lr × 0.1`，SPACE-Former 使用完整 `training.lr`。`ReduceLROnPlateau(mode=max)` 监控 validation `task_success_rate`，绝对提升不足 `0.001` 连续停滞超过 3 个 epoch 后将两个参数组学习率同时乘以 `0.5`。最佳 checkpoint 仅按 validation 的 top-1 `task_success_rate` 保存，不设置 Source IoU 门槛。该指标与推理最终采用的位姿一致，成功条件为尺寸 IoU ≥ 0.8、满足文本空间方向关系且与场景已有物体无碰撞。

训练保持 FP32。P1/P2/P3 的 sparse lookup 排序索引在一次 forward 内由四层 Decoder 复用；训练不执行 pose NMS，采样诊断仅在日志 step 计算；每个 batch 的 Hungarian 代价只进行一次 GPU→CPU 传输，仍使用 SciPy 精确匹配。日志通过 `matched_query_count` 和 `background_query_count` 记录真实/背景分配数量，便于检查背景代价是否合适。进度条最多每 20 step 同步一次 loss。

## 运行

单卡训练：

```bash
python tools/train_lc_bgplacenet_stage2.py \
  --config configs/lc_bgplacenet_stage2.yaml \
  --stage1-checkpoint outputs/lc_bgplacenet_stage1_clip16_aligned/best.pt
```

多卡训练：

```bash
torchrun --nproc_per_node=4 tools/train_lc_bgplacenet_stage2.py \
  --config configs/lc_bgplacenet_stage2.yaml \
  --stage1-checkpoint outputs/lc_bgplacenet_stage1_clip16_aligned/best.pt
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
  --resume outputs/lc_bgplacenet_stage2_space_former_aligned/last.pt
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
  --checkpoint outputs/lc_bgplacenet_stage2_space_former_aligned/best.pt \
  --split valid --max-samples 20
```

`predictions.json` 中 `place_box` 保留 top-1 兼容字段，`placements` 保存最多 16 个 `{box, score, yaw_bin}`。`pred_heatmaps/*.ply` 可视化 P3 粗区域概率，而非旧版 dense placement heatmap。

推理还会在 `decoder_stages/` 中保存 SPACE-Former 四层 Decoder 的候选框投影图，并在每条 prediction 的新增 `decoder_stages` 字段记录各层的候选框、分数、yaw bin 和图片路径。前三层使用与最终推理相同的 Pose NMS；第四层直接复用原有最终后处理结果，因此顶层 `place_box`、`placements` 以及 benchmark 的检测输入不变。

生成四阶段静态可视化网站：

```bash
python tools/export_lc_bgplacenet_stage2_inference_web.py \
  --config configs/lc_bgplacenet_stage2.yaml \
  --input-dir outputs/lc_bgplacenet_stage2_space_former_aligned/inference_stage2_valid \
  --split valid
```

网页输出为 `<input-dir>/web_vis/index.html`，默认分页并按需渲染样本，避免一次性加载全量 decoder PNG。页面只额外生成 P3 俯视 heatmap 到 `web_vis/assets/`，Decoder 1～4 和最终 top-1 图片仍引用推理目录中的原始文件，不复制大图。点击任一阶段后可用左右方向键切换。最终 top-1 图和 P3 粗区域 heatmap 位于每条样本的折叠诊断区域。

对不在 Stage-2 labels 中的 canonical 样本做 prediction-only 定向推理：

```bash
python tools/infer_lc_bgplacenet_stage2.py \
  --config configs/lc_bgplacenet_stage2_enriched.yaml \
  --checkpoint outputs/lc_bgplacenet_stage2_space_former_aligned_loss_48query_full_gt_guass_8_enriched/best.pt \
  --split all \
  --sample-id hope__scene_0000__0005 \
  --object-id obj_3 \
  --instruction "Put the tomato sauce can in the back left of the mustard bottle." \
  --no-gt \
  --output-dir outputs/lc_bgplacenet_stage2_space_former_aligned_loss_48query_full_gt_guass_8_enriched/inference_custom_hope_scene_0000_0005
```

`--instruction` 模式要求同时提供 `--sample-id`、`--object-id` 和 `--no-gt`，直接从 canonical metadata 构造单条输入，不修改 labels 或数据划分。将其 P3 预测渲染为三张同视角的单栏论文图：

```bash
python tools/render_lc_bgplacenet_stage2_point_mask.py \
  --predictions outputs/lc_bgplacenet_stage2_space_former_aligned_loss_48query_full_gt_guass_8_enriched/inference_custom_hope_scene_0000_0005/predictions.json \
  --item-id hope__custom_hope_scene_0000_0005_obj_3 \
  --output-dir outputs/lc_bgplacenet_stage2_space_former_aligned_loss_48query_full_gt_guass_8_enriched/result_visualizations/hope__scene_0000__0005
```

脚本分别输出完整 50000 点 RGB 场景、局部连续 P3 Mask 和局部 Top-1 放置框图，并同时保存 PNG 与 PDF。第二、三张图共享由源框、预测框和响应不低于 `0.5` 的核心区域联合确定的语义 ROI；该阈值只参与裁切，不过滤 ROI 内的响应。连续 Mask 按 `0.03 + 0.79 * score^1.35` 混入 `#FF2020`，局部点大小由 `0.82` 连续增加到 `1.20`。放置图使用点大小 `0.82`、100% 不透明的原始 RGB 点云；源框使用深青/亮青双层虚线，预测框采用由深蓝到底部、亮蓝到顶部的高度渐变和双层蓝色边线。最终局部版本使用文件名后缀 `local_dense_rgb_3d`，不会覆盖上一版本。

## 测试

```bash
pytest -q tests/test_lc_bgplacenet_stage2.py
```

测试覆盖 top-8 区域展开及不足 8 个 cell、P3 GT 覆盖率宏平均、Query padding、64 点模板、缓存 active lookup、Hungarian 匹配、训练 NMS 跳过、学习率恢复与平台衰减、多 yaw 合并、集合输出和 Source 联合反向传播。

# SPACE-Former Stage 2 架构

本文档替代旧版 Dense Placement Field 方案。实现入口为 `src/models/lc_bgplacenet/stage2.py`，训练目标与匈牙利匹配位于 `src/training/lc_bgplacenet_stage2.py`。

## 1. 数据流

```text
Stage 1 language-fused active voxels F_vl
        │
        ├─ Sparse Pyramid: P1(1 cm), P2(2 cm), P3(4 cm)
        │
text_tokens ──> 4-layer Region Cross Encoder ──> top-8 P3 cells
                                                       │
                                      expand to covered P1 active voxels
                                                       │
                                           FPS / padding → 48 queries
                                                       │
source feature + source size ────────────────> 4-layer SPACE decoder
text tokens ───────────────────────────────────────────┘
                                                       │
                                  48 raw boxes → pose NMS → at most 16 boxes
```

Source Grounding 同时输出 `source_box` 与 `source_feature`。Source Size 被直接复制为所有放置 Box 的尺寸，Stage 2 不预测尺寸 residual。

## 2. Text-Guided Region Cross Encoder

P3 feature 是 K/V，所有有效 `text_tokens` 是 Q。Cross Block 层数由 `model.space_former.region_cross_num_layers` 定义，当前为四层；每层均使用 8-head Cross-Attention 和 `256→1024→256` FFN。更新后的每个文本 token 与每个 P3 feature 做逐 head compatibility，再沿 token 维使用带 attention mask 的 log-mean-exp 聚合，使方向词和对象词的强匹配不会被平均稀释：

```text
a_ij = mean_h((Wq_h t_j) · (Wk_h f_i) / sqrt(32))
a_i = logsumexp_j(a_ij) - log(number_of_valid_tokens)
region_logit_i = a_i + MLP([f_i, a_i])
```

Region GT 直接在 P3 world coordinates 上，以 `direction_filtered_heatmaps` 正点为中心生成三维高斯 soft target，并在由 P1 active support 聚合得到的 P3 support 外置零；标准差由 `data.heatmap_sigma_voxels × data.voxel_size_cm` 唯一定义。推理按 `space_former.num_region_cells` 直接取每个样本 top-K，当前为 top-8，不设阈值。`p3_gt_point_coverage` 统计每条样本 direction-positive GT 点落入实际 top-8 cell 的比例，并进行样本宏平均。

## 3. Query

每个 Anchor 必须是 P1 active voxel。support、heatmap 正点与 yaw center 在训练监督中使用 canonical voxel key 严格关联，不执行最近邻吸附；support 不作为推理输入：

```text
q = LayerNorm(P1_feature(anchor) + PE(anchor_world_normalized))
```

Query 内部状态只包含 256 维 feature、连续 bottom center、12-bin yaw logits 和 valid mask。Padding Query 不参与自注意力、物理采样或匈牙利匹配，placement logit 固定为 `-20`。

## 4. 物理采样

第 0 层 yaw 未知，使用外接圆柱：bottom/middle/top ring 各 16 点，底面圆盘 16 点，共 64 点。bottom ring 和圆盘沿 world-Z 下移 0.5 voxel。

第 1～3 层使用上一层 yaw：Bottom 16 点、Top 16 点、四个 Side face 共 32 点。实际采样坐标为：

```text
p_sample = p_box_center + R(yaw) Δp_base
```

不存在 Learned Sampling Offset。

每个采样坐标分别在 P1/P2/P3 量化并查询 exact sparse key。命中 active voxel 时返回其 256 维 feature；未命中或越界时返回零 feature，并令 `sample_valid_mask=False`。

## 5. Sample Token

每个尺度使用独立 Projector：

```text
[sample_feature(256), sample_valid_mask(1), is_bottom(1), normalized_base_offset(3)]
→ Linear(261,256) → LayerNorm → GELU → Linear(256,256)
```

只保留 `bottom_support` 与 `non_bottom_support` 两类。`sample_valid_mask` 是输入特征而非 Attention padding mask，所以 inactive token 仍能表达底面踩空或边界净空。

`normalized_base_offset = Δp_base / source_size` 在 yaw 旋转前计算。它不改变采样坐标，只告诉几何分支碰撞/支撑发生在物体局部坐标系的哪个方向，从而为局部中心 residual 提供方向依据。

未使用 occupancy ratio、scale embedding、learned point-type embedding 或插值权重。

## 6. Factorized Decoder

Geometric Routing 先构造：

```text
c_src = MLP([source_feature, log(source_size)])
q_geo = LayerNorm(q + c_src)
```

P1/P2/P3 分别执行 `q_geo → 64 sample tokens` 的 8-head Cross-Attention，再以 `q_geo` 对三个尺度 token 做第二次 Cross-Attention。FFN 后只输出 geometry feature。

Semantic Routing 使用 Query 对 CLIP text tokens 做 8-head Cross-Attention，经 FFN 后只输出 semantic feature。两个路由分支不独立预测 Center、Yaw 或置信度。

两分支先进行门控融合：

```text
alpha = sigmoid(MLP([g,s]))
h = LayerNorm(q + alpha*g + (1-alpha)*s)
```

当前层的三个预测头统一读取融合特征 `h`：

```text
Δc_local = source_size * tanh(center_head(h))
Δc_world = R(previous_yaw) Δc_local
bottom_center_l = bottom_center_(l-1) + Δc_world
yaw_logits_l = yaw_head(h)
placement_logit_l = placement_head(h)
```

前三层使用 `q_next=h` 继续迭代，最后一层直接用 `h` 产生最终预测。因此四层 Fusion 均受到对应层预测损失监督。

## 7. 匹配与损失

GT 保留全部 direction-filtered 合法中心，不执行 FPS 截断；GT yaw 是 12-bin multi-hot mask，不要求选择唯一 yaw。Hungarian 在全部真实中心和与有效 Query 等量的背景虚拟目标之间建立一对一关系：

```text
C_match(i,j) = mean(abs(pred_bottom_center_i - gt_bottom_center_j) / gt_source_size)
C_background = loss.background_match_cost = 0.25
```

分类、Yaw 和角点不参与匹配，避免尚未收敛的属性预测改变空间目标。匹配真实中心的 Query 接受分类、中心、Yaw 和角点监督，其中角点损失对当前 GT 中全部有效 yaw 取最小值；匹配背景的 Query 只接受 placement score 为 0 的分类监督。训练损失为：

```text
L = 500 L_region + L_cls + 5 L_center + 0.5 L_yaw
    + 0.5 L_corner + 0.5 L_source + 0.1 L_aux

L_source = 4 L_source-center + 2 L_source-size + 0.2 L_source-IoU
```

前三层辅助监督的内部比例与匹配一致，采用 `2 L_cls + 8 L_center + L_yaw + L_corner`。

Stage 1 不冻结，使用 0.1× 学习率并持续接受 Source Box 监督。Stage 2 使用监控 validation `placement_success_at_1` 的 `ReduceLROnPlateau(mode=max, factor=0.5, patience=3, threshold=0.001, threshold_mode=abs)`。恢复训练先加载模型、AdamW moments 和 scheduler 历史，再由当前配置学习率覆盖两个参数组并保持 `0.1:1`。CLIP 是否冻结由原配置决定。

## 8. 输出

```text
raw_place_boxes   [B,48,7]
raw_place_logits  [B,48]
raw_yaw_logits    [B,48,12]
query_valid_mask  [B,48]

place_boxes       [B,16,7]
place_scores      [B,16]
place_yaw_bins    [B,16]
place_valid_mask  [B,16]

place_box         [B,7]
source_box        [B,6]
```

所有有效输出的 `(dx,dy,dz)` 严格等于当前 Stage 1 Source Size。`place_box` 是 `place_boxes[:,0]` 的兼容字段。最佳 checkpoint 只按 `Placement Success@1` 选择；`Placement Success@5` 用于统计前 5 个候选中是否存在同时通过语言关系、连通支撑和碰撞三项条件的候选。支撑和碰撞使用“预测放置中心 + GT 长宽高 + 预测 yaw”构造的评估框；Source 3D IoU、Placement Size IoU 独立报告；Yaw 只在中心 2 cm 匹配成功的样本中计算条件准确率。

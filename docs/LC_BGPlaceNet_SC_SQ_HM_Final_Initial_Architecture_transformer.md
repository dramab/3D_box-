# LC-BGPlaceNet Dense 3D Placement Box 网络架构设计方案

---

## 1. 任务定义

本方案面向语言条件的 3D 物体放置任务。模型输入单帧 RGB-D 反投影点云和自然语言指令，输出需要移动的源物体 box，以及最终目标放置 3D box。

任务拆成两个阶段：

```text
Stage 1: Source Grounding
    根据语言指令定位需要移动的源物体，预测源物体中心、尺寸和源物体聚合特征。

Stage 2: Dense 3D Placement Box Prediction
    在保留 Stage 1 结构的前提下，融合源物体特征、源物体尺寸先验和稀疏体素几何特征，
    在所有 active voxel 上预测放置底面中心 heatmap，并解码最终放置 3D box。
```

Stage 1 作为预训练先验知识保留。Stage 2 不再训练 Support Head、topK 支撑点采样，也不使用候选点序列 Transformer；
但 dense heatmap label 生成阶段需要结合 `support_masks`，只在支撑区域内生成监督。

整体思想：

```text
先用 Stage 1 找到要移动的源物体并提取 source prior；
再用 dense voxel placement field 在全体 active voxel 上预测放置底面中心；
最后结合 size residual 和 yaw field 解码完整 place_box。
```

---

## 2. 输入与输出

### 2.1 输入

每个训练样本包含：

```python
sample = {
    "points": Tensor[N, 6],
    "instruction": str,
    "direction_filtered_heatmap_ply": path,
    "support_masks": Tensor[Nv],
    "place_box_gt": Tensor[7],
    "source_box_gt": Tensor[6],
}
```

其中：

```text
points:
    单帧 RGB-D 反投影得到的点云。
    每个点包含 (x, y, z, r, g, b)。

instruction:
    自然语言放置指令。

direction_filtered_heatmap_ply:
    经过语言方向筛选后的可放置底面中心 heatmap。
    该文件只作为 Stage 2 训练监督，不作为推理阶段输入。

support_masks:
    active voxel 级别的支撑区域掩码。
    dense heatmap label 只在 support_masks 对应的 active voxel 上生成；
    其它 voxel 的 heatmap label 概率固定为 0。

place_box_gt:
    最终放置 box 监督，格式为 (x, y, z, dx, dy, dz, yaw)。
    其中 (x, y, z) 是 box 几何中心。

source_box_gt:
    Stage 1 辅助监督，格式为 (cx, cy, cz, dx, dy, dz)。
```

不再使用：

```text
support_mask_ply
support_voxel_label
support_logits
topK support candidates
```

注意：这里不再训练 Support Head，也不再使用 support loss；
`support_masks` 只作为 dense heatmap label 的有效区域约束。

---

### 2.2 输出

模型输出：

```python
outputs = {
    "source_box": Tensor[B, 6],
    "source_feature": Tensor[B, C],

    "placement_heatmap_logits": Tensor[Nv],
    "bottom_offset": Tensor[Nv, 3],
    "yaw_sincos": Tensor[Nv, 2],
    "size_residual": Tensor[B, 3],
    "size_pred": Tensor[B, 3],

    "place_box": Tensor[B, 7],
}
```

其中：

```python
source_box = (cx, cy, cz, dx, dy, dz)
place_box = (x, y, z, dx, dy, dz, yaw)
```

`source_box` 来自 Stage 1。Stage 2 只使用：

```python
source_feature = outputs["source_feature"]
source_size_stage1 = source_box[:, 3:6]
```

Stage 2 不使用 Stage 1 预测的 source center，避免源物体当前位置直接泄漏进放置位置预测。

---

## 3. 总体架构

模型名称建议：

```text
LC-BGPlaceNet-DPF
```

其中：

```text
LC: Language-Conditioned
BGPlaceNet: Box-Guided Placement Network
DPF: Dense Placement Field
```

整体流程：

```text
points                                  instruction
  │                                           │
  ▼                                           ▼
Sparse 3D Backbone                     Text Encoder
  │                                           │
  ▼                                           ▼
  └──────────────► Voxel-Language Fusion ◄────┘
                            │
                            ▼
                    F_vl: 语言条件体素特征
                            │
                            ▼
              Single-Query Source Grounding
                            │
                            ▼
            source_feature, source_size_stage1
                            │
                            ▼
       Source-Conditioned Dense Placement Field
                            │
          ┌─────────────────┼─────────────────┐
          ▼                 ▼                 ▼
  dense heatmap       bottom offset      yaw sin/cos
          │                 │                 │
          └─────────────────┼─────────────────┘
                            ▼
                 size residual refinement
                            │
                            ▼
        place_box=(x,y,z,dx,dy,dz,yaw)
```

---

## 4. Stage 1：Source Grounding

Stage 1 保持原方案不变：

```text
Sparse 3D Backbone
Text Encoder
Voxel-Language Fusion
Single-Query Source Grounding Head
```

Stage 1 输出：

```python
source_out = {
    "source_box": Tensor[B, 6],
    "source_feature": Tensor[B, C],
}
```

其中：

```python
source_box = (cx, cy, cz, dx, dy, dz)
```

Stage 1 的作用：

```text
1. 提供源物体语义和几何先验 source_feature。
2. 提供源物体尺寸先验 source_size_stage1。
3. 为 Stage 2 的 dense placement prediction 提供预训练初始化。
```

Stage 2 不直接使用 `source_box[:, 0:3]`，即不使用源物体中心点。

---

## 5. Stage 2：Source-Conditioned Dense Placement Field

### 5.1 设计目标

Stage 2 的目标是在所有 active voxel 上预测一个 dense placement field，而不是先抽取 topK 候选点再做 Transformer 融合。

核心输入：

```text
1. Stage 1 输出的 source_feature。
2. Stage 1 预测的 source_size_stage1。
3. Sparse 3D Backbone / Voxel-Language Fusion 得到的体素几何与语言条件特征。
```

核心输出：

```text
1. 每个 active voxel 作为目标 box 底面中心的 heatmap 分数。
2. 每个 active voxel 到真实底面中心的 offset。
3. 每个 active voxel 的 yaw sin/cos。
4. 基于 source size 的 size residual。
```

该模块不显式预测 support surface，也不使用 support loss；
`support_masks` 只用于限制 dense heatmap label 的有效 voxel 范围。

---

### 5.2 Source condition 构造

Stage 1 输出：

```python
source_feature: Tensor[B, C]
source_size_stage1 = source_box[:, 3:6]  # [B, 3]
```

构造 source size embedding：

```python
e_size = size_mlp(source_size_stage1)
```

构造 source condition：

```python
source_condition = source_condition_mlp(
    torch.cat([source_feature, e_size], dim=-1)
)
```

得到：

```python
source_condition: Tensor[B, C]
```

`source_condition` 表示：

```text
要移动的是什么；
这个物体有多大；
Stage 1 从语言和点云中提取到的源物体先验。
```

---

### 5.3 Dense voxel token 融合

对每个 active voxel，融合：

```text
F_vl_i:
    语言条件体素特征。

PE(coord_i):
    体素 3D 坐标位置编码。

source_condition_b:
    当前 batch 样本的源物体条件特征。
```

公式：

```python
source_per_voxel = source_condition[batch_indices]
voxel_input = torch.cat([
    F_vl,
    pos_mlp(coords),
    source_per_voxel,
], dim=-1)

F_place = placement_fusion_mlp(voxel_input)
```

推荐再接 sparse conv / sparse U-Net style dense head：

```text
F_place
    ↓
Sparse Conv Blocks / Sparse U-Net Neck
    ↓
Dense placement field features
```

这样 Stage 2 能在体素空间中聚合局部几何上下文，而不是只对孤立候选点打分。

---

### 5.4 Stage 2 heads

Stage 2 包含四个预测头：

```python
heatmap_head(F_place) -> placement_heatmap_logits: Tensor[Nv]
offset_head(F_place)  -> bottom_offset: Tensor[Nv, 3]
yaw_head(F_place)     -> yaw_sincos: Tensor[Nv, 2]
size_head(source_condition) -> size_residual: Tensor[B, 3]
```

含义如下：

```text
placement_heatmap_logits:
    每个 active voxel 作为目标 box 底面中心的分数。

bottom_offset:
    从 active voxel 中心到目标 box 底面中心的连续残差。

yaw_sincos:
    每个 active voxel 对应的 yaw 连续表示 (sin yaw, cos yaw)。

size_residual:
    基于 Stage 1 source size 的尺寸微调量。
```

---

## 6. Direction Filtered Heatmap 监督

### 6.1 heatmap 文件含义

`direction_filtered_heatmap_ply` 表示经过语言方向筛选后的可放置底面中心候选。

它来自 free_bbox heatmap，并经过自然语言关系过滤。例如只保留符合“放到参照物右边”的正激活点。

重要约束：

```text
direction_filtered_heatmap_ply 只用于训练监督。
推理阶段不能读取 direction_filtered_heatmap_ply。
```

---

### 6.2 正点识别规则

训练时读取 PLY 的点和颜色：

```python
points_hm: Tensor[M, 3]
colors_hm: Tensor[M, 3]
```

正激活点使用当前标注工具中的颜色规则：

```python
positive = (colors_hm[:, 0] == 255) & (colors_hm[:, 2] == 30)
```

其中：

```text
red = 255
blue = 30
```

被方向过滤降权的点不作为正样本。

---

### 6.3 dense heatmap label 生成

对每个 active voxel 中心：

```python
p_i: 第 i 个 active voxel 中心
s_i: 第 i 个 active voxel 的 support_masks 值，1 表示支撑区域内，0 表示支撑区域外
```

对每个 heatmap 正点：

```python
q_j: 第 j 个 direction-filtered 正激活点
```

生成 dense heatmap label：

```python
H_i_raw = max_j exp(-||p_i - q_j||^2 / (2 * sigma^2))
H_i = H_i_raw if s_i == 1 else 0
```

默认：

```python
sigma = 2 * voxel_size
```

如果某个 active voxel 位于 `support_masks` 外，则其 label 概率固定为 0；
如果位于 `support_masks` 内但距离所有正点较远，则其 label 接近 0，作为负样本。

推荐实现时先在同一 batch 样本内部计算 dense heatmap，再用 `support_masks` 做逐 voxel 相乘：

```python
placement_heatmap_gt = dense_heatmap_gt * support_masks.float()
```

不跨样本对齐。

---

### 6.4 heatmap loss

第一版推荐使用 Focal Loss：

```python
L_heat = FocalLoss(placement_heatmap_logits, placement_heatmap_gt)
```

原因：

```text
可放置底面中心点通常只占 active voxel 的很小比例；
BCE 容易被大量负样本主导；
Focal Loss 能降低简单负样本权重。
```

如果实验中 heatmap label 是平滑连续分布，也可以使用 BCEWithLogitsLoss 作为基线。

---

## 7. Box Decode

### 7.1 选择最高分底面中心体素

对每个 batch 样本，在对应 active voxel 中选择最高分：

```python
best_idx = argmax_per_batch(placement_heatmap_logits, batch_indices)
```

取最高分体素中心：

```python
voxel_center_best = coords[best_idx]
```

取对应 offset：

```python
offset_best = bottom_offset[best_idx]
```

得到预测底面中心：

```python
bottom_center_pred = voxel_center_best + offset_best
```

---

### 7.2 尺寸残差微调

Stage 1 提供尺寸先验：

```python
source_size_stage1 = source_box[:, 3:6]
```

Stage 2 预测：

```python
size_residual = size_head(source_condition)
```

推荐第一版使用指数残差：

```python
size_pred = source_size_stage1 * torch.exp(size_residual)
```

这样可以保证尺寸为正，并保持 Stage 1 size 的先验作用。

也可以使用：

```python
size_pred = F.softplus(torch.log(source_size_stage1.clamp_min(eps)) + size_residual)
```

第一版建议使用 `source_size_stage1 * exp(size_residual)`，更直观地表示相对尺度修正。

---

### 7.3 底面中心转几何中心

`direction_filtered_heatmap_ply` 和 dense heatmap 都表示 box 底面中心。

最终 `place_box` 的 `(x, y, z)` 表示 box 几何中心，因此：

```python
place_center = bottom_center_pred.clone()
place_center[:, 2] = bottom_center_pred[:, 2] + size_pred[:, 2] / 2
```

该公式依赖 canonical world 满足：

```text
world-Z = 支撑面法向/重力上方向
```

---

### 7.4 yaw 解码

取最高分体素对应的 yaw sin/cos：

```python
yaw_vec = yaw_sincos[best_idx]  # [B, 2]
```

归一化后解码：

```python
yaw_vec = F.normalize(yaw_vec, dim=-1)
yaw = torch.atan2(yaw_vec[:, 0], yaw_vec[:, 1])
```

其中：

```text
yaw_vec[:, 0] = sin yaw
yaw_vec[:, 1] = cos yaw
```

对于底面长宽接近的物体，yaw 不重要。推理时可以随机采样 yaw：

```python
yaw = Uniform(-pi, pi)
```

是否随机由第 9 节的 yaw 方向敏感性规则决定。

---

### 7.5 拼接最终 place_box

最终输出：

```python
place_box = torch.cat([
    place_center,
    size_pred,
    yaw[:, None],
], dim=-1)
```

即：

```python
place_box = (x, y, z, dx, dy, dz, yaw)
```

---

## 8. Center 与 Size Loss

### 8.1 GT 底面中心

GT 放置 box：

```python
place_box_gt = (x_gt, y_gt, z_gt, dx_gt, dy_gt, dz_gt, yaw_gt)
```

其中 `(x_gt, y_gt, z_gt)` 是几何中心。

转换为底面中心：

```python
bottom_gt = place_box_gt[:, 0:3].clone()
bottom_gt[:, 2] = place_box_gt[:, 2] - place_box_gt[:, 5] / 2
```

---

### 8.2 bottom offset 监督

在 heatmap 正样本附近监督 offset。

可以选取每个样本中距离 `bottom_gt` 最近的 active voxel：

```python
pos_idx = nearest_voxel_per_batch(coords, batch_indices, bottom_gt)
```

目标 offset：

```python
offset_gt = bottom_gt - coords[pos_idx]
```

损失：

```python
L_offset = SmoothL1(bottom_offset[pos_idx], offset_gt)
```

同时对解码后的几何中心监督：

```python
L_center = SmoothL1(place_center_pred, place_box_gt[:, 0:3])
```

第一版总 center loss：

```python
L_center_total = L_offset + L_center
```

---

### 8.3 size loss

使用最终放置 box 的 size 监督 Stage 2 微调后的尺寸：

```python
place_size_gt = place_box_gt[:, 3:6]
L_size = SmoothL1(size_pred, place_size_gt)
```

Stage 2 不直接把最终尺寸固定为 Stage 1 size，而是学习：

```text
最终放置尺寸 = Stage 1 source size + 残差微调
```

这样可以纠正 Stage 1 source size 的误差，并让最终 place_box 与放置监督对齐。

---

## 9. Yaw 方向敏感性监督

### 9.1 设计动机

不同物体对 yaw 的敏感程度不同。

```text
长条形物体:
    底面 dx、dy 差异大，yaw 错误会明显改变放置姿态。

近似正方形物体:
    底面 dx、dy 接近，yaw 的物理和视觉影响较小。
```

因此 yaw loss 不应该对所有样本一视同仁。

---

### 9.2 方向敏感性判定

使用底面尺寸：

```python
dx = place_box_gt[:, 3]
dy = place_box_gt[:, 4]
```

定义：

```python
ratio = abs(dx - dy) / min(dx, dy)
```

默认阈值：

```python
yaw_sensitive = ratio >= 0.25
```

含义：

```text
当底面长宽差异达到较短边的 25% 以上时，认为 yaw 方向敏感。
```

---

### 9.3 yaw loss

只对 `yaw_sensitive=True` 的样本计算 yaw loss。

GT yaw 使用 sin/cos 表示：

```python
yaw_target = torch.stack([
    torch.sin(yaw_gt),
    torch.cos(yaw_gt),
], dim=-1)
```

预测 yaw 取最高分体素对应输出：

```python
yaw_pred = yaw_sincos[best_idx]
yaw_pred = F.normalize(yaw_pred, dim=-1)
```

损失：

```python
L_yaw = SmoothL1(
    yaw_pred[yaw_sensitive],
    yaw_target[yaw_sensitive],
)
```

如果一个 batch 中没有方向敏感样本，则：

```python
L_yaw = 0
```

---

### 9.4 近方形物体推理 yaw

对于：

```python
ratio < 0.25
```

的近方形物体，训练时不计算 yaw loss。

推理时默认随机采样：

```python
yaw ~ Uniform(-pi, pi)
```

原因：

```text
这类物体底面接近正方形，yaw 角度本身不稳定；
强行监督固定 yaw 容易向网络注入噪声。
```

如果实验需要 deterministic 输出，可以额外固定随机种子或设置 yaw=0 作为评估基线，但第一版论文方案采用随机 yaw。

---

## 10. 总 Loss

Stage 2 总损失：

```text
L = λ_heat L_heat
  + λ_center L_center_total
  + λ_size L_size
  + λ_yaw L_yaw
  + λ_src L_src_aux
```

其中：

```text
L_heat:
    direction_filtered_heatmap_ply 生成、并由 support_masks 限制有效区域的 dense heatmap 监督。

L_center_total:
    bottom offset loss 和最终几何中心 loss。

L_size:
    最终放置 box size 监督。

L_yaw:
    仅对底面长宽差异大的方向敏感样本计算。

L_src_aux:
    Stage 1 source box 辅助监督，防止联合训练时 source grounding 严重漂移。
```

推荐初始权重：

```yaml
loss:
  lambda_heat: 1.0
  lambda_center: 1.0
  lambda_size: 1.0
  lambda_yaw: 0.5
  lambda_src: 0.2
```

不再使用：

```text
support loss
support voxel BCE
candidate yaw CE
topK candidate heatmap loss
differentiable support coverage loss
differentiable collision loss
```

---

## 11. 训练策略

### 11.1 Stage 1 预训练

Stage 1 按原方案预训练：

```text
points + instruction
    ↓
source_box=(cx,cy,cz,dx,dy,dz)
source_feature
```

训练目标：

```text
L_stage1 = L_src
```

Stage 1 完成后，保存 checkpoint 作为 Stage 2 初始化。

---

### 11.2 Stage 2 联合训练

Stage 2 从 Stage 1 checkpoint 初始化：

```text
Sparse 3D Backbone
Text Encoder
Voxel-Language Fusion
Single-Query Source Grounding
```

Stage 2 训练时不冻结 Stage 1 权重：

```text
Sparse 3D Backbone 参与更新；
Voxel-Language Fusion 参与更新；
Source Grounding Head 参与更新；
Stage 2 Dense Placement Field 参与更新。
```

这样做的原因：

```text
Stage 2 需要调整 Sparse 3D Backbone 的特征提取策略，
使体素特征更适合最终 placement heatmap、center、size 和 yaw 预测。
```

同时保留较小权重的 `L_src_aux`：

```text
避免 Stage 1 source grounding 能力在 Stage 2 联合训练中完全退化。
```

---

## 12. 推理流程

推理阶段输入只有：

```python
points
instruction
```

不输入：

```text
direction_filtered_heatmap_ply
support_mask_ply
```

推理伪代码：

```python
def inference(points, instruction):
    sparse_tensor = voxelize(points)

    F_3d, coords, batch_indices = sparse_backbone(sparse_tensor)

    text_tokens, text_global = text_encoder(instruction)
    F_vl = voxel_language_fusion(F_3d, text_tokens, batch_indices)

    source_out = source_grounding(F_vl, coords, text_global, batch_indices)
    source_box = source_out["source_box"]
    source_feature = source_out["source_feature"]
    source_size_stage1 = source_box[:, 3:6]

    place_out = dense_placement_field(
        voxel_features=F_vl,
        coords=coords,
        batch_indices=batch_indices,
        source_feature=source_feature,
        source_size_stage1=source_size_stage1,
    )

    heatmap_logits = place_out["placement_heatmap_logits"]
    bottom_offset = place_out["bottom_offset"]
    yaw_sincos = place_out["yaw_sincos"]
    size_residual = place_out["size_residual"]

    best_idx = argmax_per_batch(heatmap_logits, batch_indices)
    bottom_center = coords[best_idx] + bottom_offset[best_idx]

    size_pred = source_size_stage1 * torch.exp(size_residual)

    center = bottom_center.clone()
    center[:, 2] = bottom_center[:, 2] + size_pred[:, 2] / 2

    yaw_vec = F.normalize(yaw_sincos[best_idx], dim=-1)
    yaw = torch.atan2(yaw_vec[:, 0], yaw_vec[:, 1])

    ratio = torch.abs(size_pred[:, 0] - size_pred[:, 1]) / torch.minimum(
        size_pred[:, 0],
        size_pred[:, 1],
    ).clamp_min(1e-6)
    near_square = ratio < 0.25
    yaw[near_square] = sample_uniform_yaw(near_square.sum())

    place_box = torch.cat([center, size_pred, yaw[:, None]], dim=-1)

    return {
        "source_box": source_box,
        "place_box": place_box,
        "placement_heatmap_logits": heatmap_logits,
    }
```

---

## 13. 主模型接口

建议主模型接口：

```python
class LCBGPlaceNetDensePlacement(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        self.backbone = SparseBackbone(cfg.backbone)
        self.text_encoder = TextEncoder(cfg.text)
        self.fusion = VoxelLanguageFusion(cfg.fusion)
        self.source_grounding = SingleQuerySourceGroundingHead(cfg.source_grounding)
        self.dense_placement_field = SourceConditionedDensePlacementField(cfg.placement)

    def forward(self, batch):
        # 1. Sparse 3D Backbone
        # 2. Text Encoder
        # 3. Voxel-Language Fusion
        # 4. Stage 1 Source Grounding
        # 5. Source-conditioned Dense Placement Field
        # 6. Training: compute losses
        # 7. Inference: decode place_box
        pass
```

Dense placement field 接口：

```python
class SourceConditionedDensePlacementField(nn.Module):
    def forward(
        self,
        voxel_features,
        coords,
        batch_indices,
        source_feature,
        source_size_stage1,
    ):
        return {
            "placement_heatmap_logits": heatmap_logits,
            "bottom_offset": bottom_offset,
            "yaw_sincos": yaw_sincos,
            "size_residual": size_residual,
        }
```

---

## 14. 配置建议

```yaml
model:
  name: LCBGPlaceNetDensePlacement
  hidden_dim: 256
  voxel_size: 0.01

backbone:
  type: spconv
  in_channels: 6
  out_channels: 256

text:
  encoder: roberta-base
  freeze: true
  out_dim: 256

fusion:
  type: voxel_language_cross_attention
  num_heads: 4
  dropout: 0.1

source_grounding:
  type: single_query
  output_dim: 6
  predict_source_yaw: false
  num_layers: 3
  num_heads: 4

placement:
  type: source_conditioned_dense_field
  use_source_feature: true
  use_source_size: true
  use_source_center: false
  use_topk_candidates: false
  use_support_supervision: false
  heatmap_target: direction_filtered_heatmap
  heatmap_support_mask: support_masks
  heatmap_point_type: box_bottom_center
  heatmap_sigma_voxels: 2.0
  heatmap_loss: focal
  size_refinement: exp_residual
  yaw_repr: sincos
  yaw_sensitive_ratio: 0.25
  near_square_yaw: random

loss:
  lambda_heat: 1.0
  lambda_center: 1.0
  lambda_size: 1.0
  lambda_yaw: 0.5
  lambda_src: 0.2

training:
  stage1_pretrained_checkpoint: outputs/lc_bgplacenet_stage1/best.pt
  freeze_stage1_in_stage2: false
  train_backbone_in_stage2: true
```

---

## 15. 核心取舍

### 15.1 为什么删除 support branch

旧方案使用 `support_mask_ply` 监督支撑面，再从支撑面中取 topK 候选点。

新方案不再训练 support branch，也不从支撑面中取 topK 候选点；
但训练 dense heatmap label 时仍使用 `support_masks` 作为有效区域约束。原因是：

```text
1. 当前监督目标已经有 direction_filtered_heatmap_ply 和 place_box_gt。
2. support GT 不是最终任务必须输出，只需要保留为 heatmap label mask。
3. topK 支撑点如果覆盖不完整，会直接限制放置位置上限。
4. dense heatmap 可以让模型在 support_masks 对应的 active voxel 上学习可放置分布，
   同时将非支撑区域概率固定为 0。
```

---

### 15.2 为什么不使用候选点 Transformer

旧方案将 topK support candidates 和 source token 拼接后送入 Transformer。

新方案改成 dense placement field，原因是：

```text
1. 放置位置本质是空间场预测，更适合 dense voxel field。
2. 不经过 topK 抽样，避免候选采样策略成为瓶颈。
3. Sparse conv / sparse U-Net 能更自然地利用局部 3D 几何上下文。
4. 论文核心可以突出 source-conditioned dense 3D placement field。
```

---

### 15.3 为什么保留 Stage 1 但 Stage 2 不冻结

Stage 1 作为预训练先验，提供 source feature 和 source size。

Stage 2 不冻结 Stage 1，是因为：

```text
最终 placement 任务需要 Sparse 3D Backbone 提取更适合放置预测的几何特征；
如果完全冻结 Stage 1，backbone 只能服务 source grounding，不一定适合 dense placement field。
```

同时保留 `L_src_aux`，避免 source grounding 在联合训练中漂移过大。

---

## 16. 核心思想总结

本方案将语言条件 3D 放置任务建模为：

```text
source grounding prior + source-conditioned dense placement field
```

完整逻辑是：

```text
输入点云和语言指令
        ↓
Sparse 3D Backbone 提取体素几何特征
        ↓
Text Encoder 编码语言
        ↓
Voxel-Language Fusion 得到语言条件体素特征
        ↓
Single-Query Source Grounding 预测 source_box 和 source_feature
        ↓
取 source_feature 与 source_size_stage1，不使用 source center
        ↓
Source-Conditioned Dense Placement Field
        ↓
在所有 active voxel 上预测底面中心 heatmap、offset、yaw sin/cos
        ↓
训练时用 support_masks 将非支撑区域 heatmap label 概率置为 0
        ↓
基于 Stage 1 source size 预测 size residual
        ↓
选择最高分 voxel 并加 offset 得到底面中心
        ↓
用 size_pred 将底面中心转换成几何中心
        ↓
按底面长宽差异决定 yaw 监督和近方形随机 yaw
        ↓
输出 place_box=(x,y,z,dx,dy,dz,yaw)
```

一句话概括：

```text
LC-BGPlaceNet-DPF 保留 Stage 1 的 source grounding 作为预训练先验，
但将 Stage 2 重设计为 source-conditioned dense 3D placement field：
它不训练 Support Head，也不依赖 topK 候选点，而是在 support_masks 对应的
active voxel 上学习经过方向筛选 heatmap 监督的放置底面中心分布，非支撑区域
label 概率置为 0，并通过 offset、size residual 和方向敏感 yaw loss 解码最终
3D 放置 box。
```

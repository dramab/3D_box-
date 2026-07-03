# LC-BGPlaceNet-SC-SQ-HM 完整网络架构设计方案

---

## 1. 任务定义

本方案面向语言条件的 3D 物体放置任务。模型输入单帧 RGB-D 反投影点云和自然语言指令，输出需要移动的源物体 box，以及目标放置 box。

任务可以拆成两个子任务：

```text
1. Source Grounding：
   根据语言指令定位需要移动的源物体，预测源物体的中心点和尺寸。

2. Placement Prediction：
   根据源物体信息、语言信息、支撑面点云和场景几何上下文，
   预测目标放置位置和最终放置 yaw。
```

整体思想是：

```text
先找出要移动的源物体；
再在支撑面上预测可放置热力体素；
最后在最高分放置点上预测目标 yaw，得到完整 place_box。
```

---

## 2. 输入与输出

### 2.1 输入

每个训练样本包含：

```python
sample = {
    "points": Tensor[N, 6],
    "instruction": str,
    "placement_sample_id": str,
    "support_mask_ply": path,
    "corners_world": Tensor[8, 3],
}
```

其中：

```text
points:
    单帧 RGB-D 反投影得到的点云。
    每个点包含 (x, y, z, r, g, b)。

instruction:
    自然语言放置指令。
    例如：
    "Move TomatoSauce located at the back left of CreamCheese to the right of TomatoSauce."

placement_sample_id:
    该语言指令对应的 free_bbox placement id。

support_mask_ply:
    该帧支撑面 mask 点云；白色点表示支撑面点。

corners_world:
    该 placement 的目标放置框角点，用于生成按支撑面面积缩放的中心区域监督。
```

---

### 2.2 输出

模型输出：

```python
outputs = {
    "source_box": Tensor[B, 6],
    "source_feature": Tensor[B, C],

    "support_logits": Tensor[Nv],
    "placement_heatmap_logits": Tensor[B, K],
    "place_yaw_logits": Tensor[B, K, 24],

    "place_box": Tensor[B, 7],
}
```

其中：

```python
source_box = (cx, cy, cz, l, w, h)
```

表示源物体的中心点和尺寸，不预测源物体 yaw。

```python
place_box = (cx, cy, cz, l, w, h, yaw)
```

表示最终目标放置 box，包含中心点、尺寸和目标放置 yaw。

目标放置 box 的尺寸默认继承源物体尺寸：

```python
place_box[:, 3:6] = source_box[:, 3:6]
```

---

## 3. 监督信号

训练时使用以下监督：

```python
gt = {
    "source_box_gt": Tensor[B, 6],
    "place_box_gt": Tensor[B, 7],
    "place_yaw_bin_gt": Tensor[B],
    "support_voxel_label": Tensor[Nv],
    "placement_heatmap_gt": Tensor[B, K],
}
```

含义如下：

```text
source_box_gt:
    源物体 GT box，只使用中心点和尺寸：
    (cx, cy, cz, l, w, h)。

place_box_gt:
    目标放置 GT box：
    (cx, cy, cz, l, w, h, yaw)。

place_yaw_bin_gt:
    目标放置 yaw 离散后的类别标签。
    第一版使用 24 个 yaw bin。

support_voxel_label:
    由目标 placement 底面中心、中心所在支撑面连通区域面积和支撑面 mask 得到的文本条件支撑中心点标签。

placement_heatmap_gt:
    由 placement_heatmap_ply 对齐到候选支撑点后得到的 placement heatmap 标签。
```

需要强调：

```text
placement_heatmap_ply 表示 box 底面中心点。
因此 decode 最终 place_box 时，需要将底面中心点转换成 box 几何中心。
```

转换方式：

```python
place_center_z = bottom_center_z + source_h / 2
```

---

## 4. 总体架构

模型名称建议：

```text
LC-BGPlaceNet-SC-SQ-HM
```

其中：

```text
LC: Language-Conditioned
BGPlaceNet: Box-Guided Placement Network
SC: Source-conditioned Support Context
SQ: Single-Query Source Grounding
HM: Heatmap-based Placement
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
          ┌─────────────────┴─────────────────┐
          │                                   │
          ▼                                   ▼
Single-Query Source Grounding          Support Surface Head
          │                                   │
          ▼                                   ▼
source_box=(c,lwh), f_src              support_prob
          │                                   │
          └──────────────┬────────────────────┘
                         ▼
        Source-aware Support Placement Attention
                         │
                         ▼
            placement candidate features
                         │
              ┌──────────┴──────────┐
              ▼                     ▼
Placement Heatmap Head       Placement Yaw Head
              │                     │
              └──────────┬──────────┘
                         ▼
       place_box=(cx,cy,cz,l,w,h,yaw_place)
```

---

## 5. 数据预处理

### 5.1 点云体素化

输入点云：

```python
points: Tensor[N, 6]  # (x, y, z, r, g, b)
```

体素化后得到：

```python
voxel_coords: Tensor[Nv, 3]
voxel_feats: Tensor[Nv, C_in]
batch_indices: Tensor[Nv]
```

推荐使用稀疏体素表示：

```text
coordinates: [num_voxels, 4]，格式为 (batch_idx, x, y, z)
features:    [num_voxels, C_in]
```

其中 `voxel_coords` 表示体素中心的世界坐标或归一化坐标。

---

### 5.2 目标支撑区域标签生成

目标 placement 的 `corners_world` 和该帧 `support_mask_ply` 用于生成：

```python
support_voxel_label: Tensor[Nv]
```

标签含义：

```text
1：该体素属于当前文字指令对应的目标支撑区域
0：该体素不属于当前文字指令对应的目标支撑区域
```

生成方式：

```python
if (
    nearest_distance(voxel_center_i, support_mask_white_points) < support_align_threshold_cm
    and xy_distance(voxel_center_i, gt_bottom_center) <= sqrt(component_area_cm2 * support_radius_area_fraction / pi)
    and abs(voxel_center_i.z - gt_bottom_z) <= support_align_threshold_cm
):
    support_voxel_label[i] = 1
```

`component_area_cm2` 只统计 GT 中心所在的支撑面连通区域，不统计整帧所有支撑面。
该标签表示“目标 placement 中心附近、且覆盖当前支撑面固定面积比例的支撑面点”。

推荐阈值：

```text
support_align_threshold_cm = 1.5 * voxel_size
support_radius_area_fraction = 0.25
```

---

### 5.3 placement heatmap 标签生成

`placement_heatmap_ply` 表示目标放置 box 的底面中心点。

对于 topK 支撑候选点：

```python
support_points: Tensor[B, K, 3]
```

将 `placement_heatmap_ply` 对齐到这些候选点，得到：

```python
placement_heatmap_gt: Tensor[B, K]
```

如果 heatmap ply 中每个点有显式热力值，可以使用高斯扩散：

```python
H_gt[i] = max_j value_j * exp(-||p_i - q_j||^2 / (2 * sigma^2))
```

其中：

```text
p_i: 第 i 个候选支撑点
q_j: heatmap ply 中的第 j 个 GT 底面中心点
```

如果 heatmap ply 只有可放置点，没有显式热力值，则令：

```python
value_j = 1.0
```

---

### 5.4 yaw bin 标签

目标放置 yaw 使用 24-bin 分类：

```python
num_yaw_bins = 24
```

如果 GT yaw 是连续角度，可以转换为：

```python
bin_width = 2 * pi / 24
yaw_norm = normalize_angle(yaw_gt)  # [-pi, pi)
yaw_bin_gt = floor((yaw_norm + pi) / bin_width)
yaw_bin_gt = clamp(yaw_bin_gt, 0, 23)
```

每个 bin 的中心角度：

```python
yaw_bin_centers = -pi + (torch.arange(24) + 0.5) * (2 * pi / 24)
```

---

## 6. Sparse 3D Backbone

### 6.1 目标

Sparse 3D Backbone 用于从稀疏点云中提取多尺度 3D 几何与外观特征。

输入：

```python
SparseTensor(
    coordinates=[Nv, 4],
    features=[Nv, C_in]
)
```

输出：

```python
F_3d: Tensor[Nv, C]
coords: Tensor[Nv, 3]
batch_indices: Tensor[Nv]
```

其中：

```text
F_3d:
    每个 active voxel 的 3D 特征。

coords:
    每个 active voxel 的 3D 坐标。

batch_indices:
    每个 voxel 属于哪个 batch sample。
```

---

### 6.2 推荐结构

第一版建议使用 Sparse U-Net 或 Sparse ResNet + FPN 结构。

原因：

```text
placement heatmap 需要较细粒度的局部几何信息，
所以网络应保留高分辨率体素特征。
```

接口示例：

```python
class SparseBackbone(nn.Module):
    def __init__(self, in_channels=6, hidden_dim=256):
        super().__init__()
        ...

    def forward(self, sparse_tensor):
        return F_3d, coords, batch_indices
```

---

## 7. Text Encoder

### 7.1 目标

Text Encoder 将自然语言指令编码为 token 级特征和全局语义特征。

输入：

```python
instruction: List[str]
```

输出：

```python
text_tokens: Tensor[B, T, C]
text_global: Tensor[B, C]
```

其中：

```text
text_tokens:
    token 级语言特征，用于 voxel-language fusion。

text_global:
    全局语言特征，用于 source query 和 placement attention。
```

---

### 7.2 可选模型

可以使用：

```text
BERT
RoBERTa
CLIP Text Encoder
T5 Encoder
```

第一版可以冻结语言编码器，降低训练难度：

```yaml
text:
  freeze: true
```

---

## 8. Voxel-Language Fusion

### 8.1 目标

Voxel-Language Fusion 用于让每个体素特征感知语言指令。

输入：

```python
F_3d: Tensor[Nv, C]
text_tokens: Tensor[B, T, C]
batch_indices: Tensor[Nv]
```

输出：

```python
F_vl: Tensor[Nv, C]
```

`F_vl` 表示语言条件体素特征。

---

### 8.2 Cross-Attention 版本

可以使用 voxel-to-language cross attention：

```text
Q = F_3d
K = text_tokens
V = text_tokens
```

公式：

```text
F_i^VL = F_i^3D + CrossAttn(Q=F_i^3D, K=E_L, V=E_L)
```

---

## 9. Single-Query Source Grounding

### 9.1 目标

Source Grounding 负责根据语言指令定位需要移动的源物体。

由于每条指令只对应一个源物体，因此使用：

```text
single learnable source query
```

不需要：

```text
多个 object queries
Hungarian matching
NMS
```

---

### 9.2 输入与输出

输入：

```python
voxel_features: Tensor[B, Nv_max, C]
voxel_pos_embed: Tensor[B, Nv_max, C]
text_global: Tensor[B, C]
voxel_padding_mask: Tensor[B, Nv_max]
```

输出：

```python
source_box: Tensor[B, 6]
source_feature: Tensor[B, C]
```

其中：

```python
source_box = (cx, cy, cz, l, w, h)
```

---

### 9.3 Query / Key / Value 设计

#### Query

使用一个可学习 query：

```python
self.source_query = nn.Parameter(torch.randn(1, 1, C) * 0.02)
```

并加入语言条件：

```python
query = source_query + text_proj(text_global)
```

#### Key

Key 来自语言融合体素特征和位置编码：

```text
K = F_vl + PE_3D(coords)
```

#### Value

Value 来自语言融合体素特征：

```text
V = F_vl
```

---

### 9.4 模块结构

```text
F_vl + PE_3D(coords)
        │
        ▼
Transformer Decoder / Cross-Attention
        │
        ▼
f_src
        │
        ▼
source box head
        │
        ▼
source_box=(cx,cy,cz,l,w,h)
```

代码接口：

```python
class SingleQuerySourceGroundingHead(nn.Module):
    def __init__(self, hidden_dim=256, num_heads=4, num_layers=3):
        super().__init__()

        self.source_query = nn.Parameter(torch.randn(1, 1, hidden_dim) * 0.02)
        self.text_proj = nn.Linear(hidden_dim, hidden_dim)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=0.1,
            batch_first=True,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)

        self.box_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 6),
        )

    def forward(self, voxel_features, voxel_pos_embed, text_global, voxel_padding_mask=None):
        B = voxel_features.shape[0]

        memory = voxel_features + voxel_pos_embed

        query = self.source_query.repeat(B, 1, 1)
        query = query + self.text_proj(text_global).unsqueeze(1)

        f_src = self.decoder(
            tgt=query,
            memory=memory,
            memory_key_padding_mask=voxel_padding_mask,
        ).squeeze(1)

        raw_box = self.box_head(f_src)
        source_box = decode_source_box_6d(raw_box)

        return {
            "source_box": source_box,
            "source_feature": f_src,
        }
```

---

### 9.5 Source box 参数化

推荐使用稳定参数化方式：

```python
center_raw, size_raw = torch.split(raw_box, [3, 3], dim=-1)

center_norm = torch.sigmoid(center_raw)
center = scene_min + center_norm * (scene_max - scene_min)

size = F.softplus(size_raw) + 1e-4

source_box = torch.cat([center, size], dim=-1)
```

这样可以保证：

```text
center 在场景范围内；
size 为正数。
```

---

## 10. Source Grounding Loss

Source box 只监督中心点和尺寸：

```text
L_src = λ_center L_center + λ_size L_size + λ_iou L_iou
```

其中：

```python
loss_center = SmoothL1(source_box_pred[:, 0:3], source_box_gt[:, 0:3])
loss_size = SmoothL1(source_box_pred[:, 3:6], source_box_gt[:, 3:6])
```

如果使用 IoU loss，由于 source box 不预测 yaw，推荐使用 axis-aligned 3D IoU：

```python
loss_iou = 1 - AABB_3D_IoU(source_box_pred, source_box_gt)
```

最终：

```python
loss_src = (
    lambda_center * loss_center
    + lambda_size * loss_size
    + lambda_iou * loss_iou
)
```

---

## 11. Support Surface Branch

### 11.1 目标

Support Surface Branch 预测每个体素是否属于支撑面。

输入：

```python
F_vl: Tensor[Nv, C]
```

输出：

```python
support_logits: Tensor[Nv]
support_prob = sigmoid(support_logits)
```

公式：

```text
p_i^sup = sigmoid(MLP_sup(F_i^VL))
```

代码：

```python
class SupportHead(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, F_vl):
        return self.mlp(F_vl).squeeze(-1)
```

---

### 11.2 Support loss

监督来自目标 placement 底面中心附近、按支撑面连通区域面积动态缩放半径的支撑面点：

```python
pos_weight = num_negative_support_voxels / max(num_positive_support_voxels, 1)
L_sup = BCEWithLogitsLoss(support_logits, support_voxel_label, pos_weight=pos_weight)
```

实现中 `loss.support.pos_weight: auto` 会按当前 batch 动态计算正类权重，并可通过
`loss.support.max_pos_weight` 限制上限，避免极端稀疏正样本 batch 造成梯度过大。
如果仍然受易负样本主导，也可以尝试 Focal Loss：

```python
L_sup = FocalLoss(support_logits, support_voxel_label)
```

---

### 11.3 支撑候选点选择

根据 `support_prob` 选择 topK 支撑候选点：

```python
support_indices = topK_per_batch(support_prob, batch_indices, K=512)
```

得到：

```python
support_points: Tensor[B, K, 3]
support_features: Tensor[B, K, C]
support_prob_k: Tensor[B, K, 1]
```

其中：

```text
support_points:
    候选支撑点坐标。

support_features:
    候选支撑点的语言融合特征。

support_prob_k:
    候选点为支撑面的概率。
```

topK 的设置需要结合当前的数据去设置多少支撑候选点比较合适，如果计算量合适则送入所有支撑候选点，如果不合适看看取多少个候选点合适，避免发生，取的支撑面候选点不能覆盖整个桌面而导致无法预测

---

## 12. Source-aware Support Placement Attention

### 12.1 设计目标

该模块负责根据源物体特征、源物体尺寸、语言特征和支撑面点云上下文，预测哪些支撑面点适合作为目标放置位置。

这里不再使用简单的 per-candidate MLP 拼接融合，而是使用 Transformer 架构进行信息融合。

核心原因是：

```text
每个候选支撑点是否适合作为放置点，不能只看它自己的局部特征，
还需要和其他候选支撑点、源物体条件、语言条件一起比较。
```

因此该模块把：

```text
source-conditioned token + support candidate tokens
```

组成一个 token 序列，通过 Transformer Encoder 进行全局上下文交互，最后对每个支撑候选 token 输出 heatmap 分数和 yaw logits。

它不显式进行候选框碰撞检测，也不使用可微支撑覆盖率公式，而是学习：

```text
给定我要移动的物体、物体尺寸、语言指令和支撑面候选点集合，
哪些支撑点更适合成为目标放置 box 的底面中心点。
```

---

### 12.2 输入

```python
source_box: Tensor[B, 6]
f_src: Tensor[B, C]
text_global: Tensor[B, C]

support_points: Tensor[B, K, 3]
support_features: Tensor[B, K, C]
support_prob: Tensor[B, K, 1]
```

其中：·

```python
source_box = (cx, cy, cz, l, w, h)
```

---

### 12.3 输出

```python
placement_features: Tensor[B, K, C]
placement_heatmap_logits: Tensor[B, K]
place_yaw_logits: Tensor[B, K, 24]
```

其中：

```text
placement_features:
    Transformer 融合后的每个候选放置点特征。

placement_heatmap_logits:
    每个候选支撑点作为目标 box 底面中心点的分数。

place_yaw_logits:
    每个候选支撑点对应的 24-bin yaw 分类结果。
```

---

### 12.4 Source-conditioned placement token

首先构造一个全局 source-conditioned placement token：

```python
source_size = source_box[:, 3:6]
e_size = MLP_size(source_size)

q_src_place = MLP_src([
    f_src,
    e_size,
    text_global,
])
```

其中：

```text
f_src:
    源物体聚合特征，表示“要移动的是什么”。

e_size:
    源物体尺寸特征，表示“这个物体有多大”。

text_global:
    语言全局特征，表示“应该放到哪里”。
```

得到：

```python
q_src_place: Tensor[B, C]
```

然后将它作为 Transformer 序列中的第一个 token：

```python
src_token = q_src_place.unsqueeze(1)  # [B, 1, C]
```

这个 token 的作用类似一个全局条件 token，让后续所有支撑候选点都能通过 self-attention 读取源物体和语言条件。

---

### 12.5 Support candidate token

对每个支撑候选点构造 token：

```python
t_i = MLP_sup([
    F_sup_i,
    P_sup_i,
    p_sup_i,
])
```

其中：

```text
F_sup_i:
    第 i 个支撑候选点的语言融合体素特征。

P_sup_i:
    第 i 个支撑候选点的 3D 坐标。

p_sup_i:
    第 i 个候选点属于支撑面的概率。
```

得到：

```python
support_tokens: Tensor[B, K, C]
```

这里保留支撑点的 3D 坐标作为候选点自身的空间位置输入，但不再额外构造“候选点相对于源物体中心”的手工特征。

---

### 12.6 Transformer-based feature fusion

将 source-conditioned token 和 support candidate tokens 拼接：

```python
tokens = torch.cat([
    src_token,        # [B, 1, C]
    support_tokens,   # [B, K, C]
], dim=1)             # [B, K+1, C]
```

然后加入 token type embedding：

```python
tokens[:, 0:1] = tokens[:, 0:1] + type_embed_source
tokens[:, 1:]  = tokens[:, 1:]  + type_embed_support
```

再输入 Transformer Encoder：

```python
fused_tokens = TransformerEncoder(tokens)
```

输出：

```python
fused_src_token = fused_tokens[:, 0]      # [B, C]
placement_features = fused_tokens[:, 1:]  # [B, K, C]
```

其中：

```text
fused_src_token:
    融合了所有候选支撑点上下文后的 source 条件特征。

placement_features:
    融合了 source 条件、语言条件、支撑面概率、3D 坐标和候选点之间上下文关系后的候选放置点特征。
```

Transformer 融合的意义是：

```text
1. 每个候选支撑点可以和 source-conditioned token 交互，知道要放置的物体是什么、尺寸多大、语言目标是什么。
2. 每个候选支撑点可以和其他候选支撑点交互，形成全局比较，而不是孤立打分。
3. heatmap 和 yaw 都基于融合后的候选点特征预测，因此 yaw 与最终放置位置绑定。
```

---

### 12.7 Heatmap / yaw prediction heads

对每个 Transformer 融合后的候选点特征分别预测：

```python
heatmap_logits_i = MLP_heat(placement_feature_i)
yaw_logits_i = MLP_yaw(placement_feature_i)
```

整体输出：

```python
placement_heatmap_logits: Tensor[B, K]
place_yaw_logits: Tensor[B, K, 24]
```

也就是说：

```text
heatmap 决定“放在哪里”；
yaw head 决定“在该位置上怎么转”。
```

---

### 12.8 模块代码接口

```python
class SourceAwareSupportPlacementAttention(nn.Module):
    def __init__(
        self,
        hidden_dim=256,
        num_yaw_bins=24,
        num_heads=4,
        num_layers=3,
        dropout=0.1,
    ):
        super().__init__()

        self.size_mlp = nn.Sequential(
            nn.Linear(3, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
        )

        self.source_token_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
        )

        self.support_token_mlp = nn.Sequential(
            nn.Linear(hidden_dim + 3 + 1, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
        )

        self.type_embed = nn.Embedding(2, hidden_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
        )

        self.heatmap_head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

        self.yaw_head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, num_yaw_bins),
        )

    def forward(
        self,
        source_box,
        f_src,
        text_global,
        support_points,
        support_features,
        support_prob,
        support_padding_mask=None,
    ):
        # source_box: [B, 6] = (cx, cy, cz, l, w, h)
        source_size = source_box[:, 3:6]
        e_size = self.size_mlp(source_size)

        src_token = self.source_token_mlp(
            torch.cat([f_src, e_size, text_global], dim=-1)
        ).unsqueeze(1)  # [B, 1, C]

        support_tokens = self.support_token_mlp(
            torch.cat([support_features, support_points, support_prob], dim=-1)
        )  # [B, K, C]

        tokens = torch.cat([src_token, support_tokens], dim=1)  # [B, K+1, C]

        B, K_plus_1, C = tokens.shape
        type_ids = torch.zeros(B, K_plus_1, dtype=torch.long, device=tokens.device)
        type_ids[:, 1:] = 1
        tokens = tokens + self.type_embed(type_ids)

        if support_padding_mask is not None:
            # support_padding_mask: [B, K]
            src_mask = torch.zeros(
                B,
                1,
                dtype=torch.bool,
                device=support_padding_mask.device,
            )
            padding_mask = torch.cat([src_mask, support_padding_mask], dim=1)
        else:
            padding_mask = None

        fused_tokens = self.encoder(
            tokens,
            src_key_padding_mask=padding_mask,
        )

        placement_features = fused_tokens[:, 1:]  # [B, K, C]

        heatmap_logits = self.heatmap_head(placement_features).squeeze(-1)
        yaw_logits = self.yaw_head(placement_features)

        return {
            "placement_features": placement_features,
            "placement_heatmap_logits": heatmap_logits,
            "place_yaw_logits": yaw_logits,
        }
```


## 13. Placement Heatmap Prediction

### 13.1 Heatmap 输出

对每个候选支撑点预测一个放置分数：

```python
placement_heatmap_logits: Tensor[B, K]
```

其中第 `i` 个值表示：

```text
第 i 个支撑候选点作为目标 box 底面中心点的可能性。
```

---

### 13.2 Heatmap loss

如果 `placement_heatmap_gt` 是二值标签，使用：

```python
L_heat = BCEWithLogitsLoss(
    placement_heatmap_logits,
    placement_heatmap_gt,
)
```

如果正负样本不平衡，推荐使用 Focal Loss：

```python
L_heat = FocalLoss(
    placement_heatmap_logits,
    placement_heatmap_gt,
)
```

如果 `placement_heatmap_gt` 是连续热力分布，也可以使用 KL loss：

```python
L_heat = KL(
    softmax(placement_heatmap_gt),
    log_softmax(placement_heatmap_logits),
)
```

第一版建议：

```text
先使用 BCE 或 Focal Loss。
```

---

## 14. Placement Yaw Prediction

### 14.1 Per-location yaw prediction

目标 yaw 不提前作为全局 prior 预测，而是在每个候选支撑点上分别预测。

输出：

```python
place_yaw_logits: Tensor[B, K, 24]
```

其中：

```text
place_yaw_logits[b, i]
```

表示第 `b` 个样本中，第 `i` 个候选支撑点对应的 yaw 分类结果。

这样设计的原因是：

```text
目标 yaw 与最终放置位置强相关。
同一个物体放在不同位置，合理 yaw 可能不同。
```

因此 yaw 应该与 placement heatmap 位置绑定。

---

### 14.2 选择 yaw 监督位置

训练时先选择正候选：

```python
pos_idx = argmax(placement_heatmap_gt, dim=-1)
```

然后取该候选位置的 yaw logits：

```python
yaw_logits_pos = place_yaw_logits[batch_idx, pos_idx]  # [B, 24]
```

使用 GT yaw bin 监督：

```python
place_yaw_bin_gt: Tensor[B]
```

---

## 15. 尺寸各向异性加权 Yaw Loss

### 15.1 设计动机

不同形状的物体对 yaw 的敏感程度不同。

对于长条形物体：

```text
长宽差别很大，yaw 非常重要。
如果 yaw 错了，放置方向会明显错误。
```

对于接近正方形的物体：

```text
长宽差别很小，yaw 不那么重要。
即使 yaw 有一定误差，物理和视觉影响也较小。
```

因此 yaw loss 需要根据物体尺寸自适应加权。

---

### 15.2 尺寸差异比例

设与 yaw 相关的两个平面尺寸为：

```text
a, b
```

通常如果 box 定义为：

```python
(cx, cy, cz, l, w, h)
```

且 yaw 是绕 z 轴旋转，则推荐：

```python
a = l
b = w
```

因为 yaw 主要改变物体在水平面 xy 上的方向。

定义尺寸差异比例：

```text
r = |a - b| / min(a, b)
```

含义：

```text
r 越小，说明 a 和 b 越接近，物体越接近正方形；
r 越大，说明 a 和 b 差别越大，物体越细长。
```

---

### 15.3 yaw loss 权重

定义 yaw loss 权重：

```text
w_yaw = w_min + (w_max - w_min) · sigmoid(α · (r - 1))
```

其中：

```text
w_min:
    yaw loss 的最小权重。

w_max:
    yaw loss 的最大权重。

α:
    控制权重变化的陡峭程度。

r - 1:
    以 r=1 作为临界点。
```

因为：

```text
r = |a-b| / min(a,b)
```

所以：

```text
r > 1
```

等价于：

```text
|a-b| > min(a,b)
```

也就是：

```text
当两个尺寸差别已经超过较小尺寸时，认为 yaw 非常重要。
```

推荐初始超参数：

```yaml
yaw_loss:
  alpha: 6.0
  w_min: 0.2
  w_max: 2.0
```

---

### 15.4 加权 yaw loss

基础 yaw classification loss：

```text
CE(yaw_logits_pos, yaw_bin_gt)
```

加入尺寸各向异性权重后：

```text
L_yaw_aniso = w_yaw · CE(yaw_logits_pos, yaw_bin_gt)
```

完整实现：

```python
def anisotropic_yaw_loss(
    yaw_logits_pos,
    yaw_bin_gt,
    source_box,
    dim1=3,
    dim2=4,
    alpha=6.0,
    w_min=0.2,
    w_max=2.0,
    eps=1e-6,
):
    # source_box: [B, 6] = (cx, cy, cz, l, w, h)
    a = source_box[:, dim1].clamp_min(eps)
    b = source_box[:, dim2].clamp_min(eps)

    ratio = torch.abs(a - b) / torch.minimum(a, b)

    gate = torch.sigmoid(alpha * (ratio - 1.0))
    yaw_weight = w_min + (w_max - w_min) * gate

    loss_each = F.cross_entropy(
        yaw_logits_pos,
        yaw_bin_gt,
        reduction="none",
    )

    loss = (yaw_weight.detach() * loss_each).mean()
    return loss
```

默认：

```python
dim1 = 3  # l
dim2 = 4  # w
```

如果数据定义中 yaw 影响的是其他两个尺寸，可以修改 `dim1` 和 `dim2`。

使用：

```python
yaw_weight.detach()
```

表示：

```text
yaw loss 的权重不反向影响 source size 预测。
```

这样可以避免模型为了降低 yaw loss 而人为改变 source_box 的尺寸预测。

---

### 15.5 可选：Circular Label Smoothing

由于 yaw bin 是环形类别，bin 0 和 bin 23 实际相邻。

后续可以将普通 CE 换成 circular label smoothing：

```text
target[k] ∝ exp(-circular_distance(k, gt_bin)^2 / 2σ^2)
```

再使用 KL loss：

```python
L_yaw = KL(
    target_distribution,
    log_softmax(yaw_logits),
)
```

第一版建议：

```text
先使用尺寸各向异性加权 CE loss。
```

---

## 16. Placement Box Decode

### 16.1 选择最高分底面中心点

根据 placement heatmap 选择最高分候选点：

```python
best_idx = torch.argmax(placement_heatmap_logits, dim=-1)
place_bottom_center = gather(support_points, best_idx)
```

由于 `placement_heatmap_ply` 表示 box 底面中心点，所以 `place_bottom_center` 也是预测出的目标 box 底面中心点。

---

### 16.2 底面中心点转 box 几何中心

源物体尺寸：

```python
source_size = source_box[:, 3:6]
source_h = source_size[:, 2]
```

box 几何中心：

```python
place_center = place_bottom_center.clone()
place_center[:, 2] = place_bottom_center[:, 2] + source_h / 2
```

---

### 16.3 尺寸继承 source box

```python
place_size = source_box[:, 3:6]
```

---

### 16.4 选择对应位置的 yaw

取最高分候选点对应的 yaw logits：

```python
yaw_logits_best = gather(place_yaw_logits, best_idx)  # [B, 24]
yaw_bin = torch.argmax(yaw_logits_best, dim=-1)
yaw_place = yaw_bin_centers[yaw_bin]
```

---

### 16.5 拼接最终 place_box

```python
place_box = torch.cat([
    place_center,
    place_size,
    yaw_place[:, None],
], dim=-1)
```

即：

```python
place_box = (cx, cy, cz, l, w, h, yaw)
```

---

## 17. 总 Loss 设计

总 loss：

```text
L = λ_src L_src
  + λ_sup L_sup
  + λ_heat L_heat
  + λ_yaw L_yaw_aniso
```

其中：

```text
L_src:
    source grounding loss，监督 source center 和 source size。

L_sup:
    support surface segmentation loss，监督支撑面预测。

L_heat:
    placement heatmap loss，监督可放置底面中心点预测。

L_yaw_aniso:
    尺寸各向异性加权 yaw classification loss。
```

不使用：

```text
source yaw loss
yaw prior loss
differentiable collision loss
differentiable support coverage loss
```

---

## 18. 训练策略

### 18.1 Stage 1：Source Grounding + Support Surface

训练模块：

```text
Sparse Backbone
Text Encoder
Voxel-Language Fusion
Single-Query Source Grounding Head
Support Surface Head
```

Loss：

```text
L_stage1 = λ_src L_src + λ_sup L_sup
```

目标：

```text
1. 学会定位源物体中心和尺寸。
2. 学会识别支撑面体素。
```

---

### 18.2 Stage 2：Placement Heatmap + Placement Yaw

训练模块：

```text
Source-aware Support Placement Attention
Placement Heatmap Head
Placement Yaw Head
```

前期建议使用 GT source box：

```python
source_box_for_place = source_box_gt
```

这样可以避免 source grounding 还不稳定时影响 placement branch。

Loss：

```text
L_stage2 = λ_heat L_heat + λ_yaw L_yaw_aniso
```

---

### 18.3 Stage 3：端到端联合训练

逐渐从 GT source box 切换为 predicted source box：

```python
if random.random() < p_gt:
    source_box_for_place = source_box_gt
else:
    source_box_for_place = source_box_pred
```

逐渐降低：

```text
p_gt: 1.0 → 0.0
```

最终端到端训练：

```text
points + instruction
    ↓
source_box=(c,l,w,h), f_src
    ↓
support_prob
    ↓
support candidates
    ↓
placement heatmap + per-location yaw
    ↓
place_box
```

---

## 19. 推理流程

```python
def inference(points, instruction):
    # 1. voxelize
    sparse_tensor = voxelize(points)

    # 2. sparse 3D backbone
    F_3d, coords, batch_indices = sparse_backbone(sparse_tensor)

    # 3. text encoder
    text_tokens, text_global = text_encoder(instruction)

    # 4. voxel-language fusion
    F_vl = voxel_language_fusion(F_3d, text_tokens, batch_indices)

    # 5. pack sparse voxel features
    voxel_tokens, voxel_pos_embed, voxel_padding_mask = pack_voxel_tokens(
        F_vl,
        coords,
        batch_indices,
    )

    # 6. source grounding
    source_out = source_grounding(
        voxel_features=voxel_tokens,
        voxel_pos_embed=voxel_pos_embed,
        text_global=text_global,
        voxel_padding_mask=voxel_padding_mask,
    )

    source_box = source_out["source_box"]       # [B, 6]
    f_src = source_out["source_feature"]        # [B, C]

    # 7. support prediction
    support_logits = support_head(F_vl)
    support_prob = torch.sigmoid(support_logits)

    # 8. select support candidates
    support_indices = topK_per_batch(
        support_prob,
        batch_indices,
        K=512,
    )

    support_points = gather_per_batch(coords, support_indices)
    support_features = gather_per_batch(F_vl, support_indices)
    support_prob_k = gather_per_batch(support_prob, support_indices).unsqueeze(-1)

    # 9. source-aware support placement attention
    place_out = source_aware_support_attention(
        source_box=source_box,
        f_src=f_src,
        text_global=text_global,
        support_points=support_points,
        support_features=support_features,
        support_prob=support_prob_k,
    )

    heatmap_logits = place_out["placement_heatmap_logits"]  # [B, K]
    yaw_logits = place_out["place_yaw_logits"]              # [B, K, 24]

    # 10. decode placement
    best_idx = torch.argmax(heatmap_logits, dim=-1)

    place_bottom_center = gather(support_points, best_idx)

    source_size = source_box[:, 3:6]
    source_h = source_size[:, 2]

    place_center = place_bottom_center.clone()
    place_center[:, 2] = place_bottom_center[:, 2] + source_h / 2

    yaw_logits_best = gather(yaw_logits, best_idx)
    yaw_bin = torch.argmax(yaw_logits_best, dim=-1)
    yaw_place = yaw_bin_centers[yaw_bin]

    place_box = torch.cat([
        place_center,
        source_size,
        yaw_place[:, None],
    ], dim=-1)

    return {
        "source_box": source_box,
        "place_box": place_box,
        "placement_heatmap_logits": heatmap_logits,
    }
```

---

## 20. 主模型接口

```python
class LCBGPlaceNetSCSQHM(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        self.backbone = SparseBackbone(cfg.backbone)
        self.text_encoder = TextEncoder(cfg.text)
        self.fusion = VoxelLanguageFusion(cfg.fusion)

        self.source_grounding = SingleQuerySourceGroundingHead(
            hidden_dim=cfg.hidden_dim,
            num_heads=cfg.source_grounding.num_heads,
            num_layers=cfg.source_grounding.num_layers,
        )

        self.support_head = SupportHead(cfg.hidden_dim)

        self.placement_attention = SourceAwareSupportPlacementAttention(
            hidden_dim=cfg.hidden_dim,
            num_yaw_bins=cfg.placement.yaw_bins,
        )

    def forward(self, batch):
        # 1. voxelization / sparse backbone
        # 2. text encoding
        # 3. voxel-language fusion
        # 4. source grounding: source_box, f_src
        # 5. support prediction: support_prob
        # 6. topK support candidate selection
        # 7. source-aware support placement attention
        # 8. heatmap + yaw prediction
        # 9. loss computation or box decode
        pass
```

---

## 21. 配置文件建议

```yaml
model:
  name: LCBGPlaceNetSCSQHM
  hidden_dim: 256
  voxel_size: 0.01
  num_yaw_bins: 24

backbone:
  type: SparseUNet
  in_channels: 6
  out_channels: 256

text:
  encoder: roberta-base
  freeze: true
  out_dim: 256

fusion:
  type: cross_attention
  num_layers: 2
  num_heads: 4

source_grounding:
  type: single_query
  num_queries: 1
  output_dim: 6
  predict_source_yaw: false
  num_layers: 3
  num_heads: 4
  box_param:
    center_activation: sigmoid
    size_activation: softplus
  loss:
    lambda_center: 1.0
    lambda_size: 1.0
    lambda_iou: 0.5
    use_axis_aligned_iou: true

support:
  topk: 512
  loss_type: focal
  positive_weight: 5.0

placement_attention:
  type: source_aware_support_transformer
  num_layers: 3
  num_heads: 4
  dropout: 0.1
  use_source_feature: true
  use_source_size_embedding: true
  use_text_global: true
  use_support_coordinates: true
  use_support_probability: true
  use_relative_to_source_center: false

placement:
  heatmap_loss: bce
  heatmap_point_type: box_bottom_center
  yaw_bins: 24
  yaw_prediction: per_candidate
  use_yaw_prior: false
  use_center_residual: false
  use_size_residual: false

yaw_loss:
  type: anisotropic_weighted_ce
  dims_for_anisotropy: [3, 4]
  threshold_ratio: 1.0
  alpha: 6.0
  w_min: 0.2
  w_max: 2.0
  detach_weight: true

loss:
  lambda_src: 1.0
  lambda_sup: 1.0
  lambda_heat: 1.0
  lambda_yaw: 0.5

training:
  source_gt_warmup: true
  p_gt_start: 1.0
  p_gt_end: 0.0
  stage1_epochs: 20
  stage2_epochs: 20
  joint_epochs: 60
```

---

## 24. 核心思想总结

本方案将语言条件 3D 放置任务建模为：

```text
source grounding + support heatmap placement + per-location yaw prediction
```

完整逻辑是：

```text
输入点云和语言指令
        ↓
Sparse 3D Backbone 提取体素特征
        ↓
Text Encoder 编码语言
        ↓
Voxel-Language Fusion 得到语言条件体素特征
        ↓
Single-Query Source Grounding 定位源物体
        ↓
预测 source_box=(center,size) 和 f_src
        ↓
Support Head 预测支撑面体素
        ↓
选择 topK 支撑候选点
        ↓
Source-aware Support Placement Attention
使用 Transformer 融合 source feature、source size、语言、支撑点特征、支撑点坐标和候选点全局上下文
        ↓
每个候选支撑点预测 placement heatmap score 和 yaw logits
        ↓
选择最高分支撑点作为 box 底面中心点
        ↓
z 方向加 source_h/2 得到 box 几何中心
        ↓
使用该位置对应 yaw logits 得到 yaw_place
        ↓
输出最终 place_box=(cx,cy,cz,l,w,h,yaw)
```

一句话概括：

> LC-BGPlaceNet-SC-SQ-HM 使用 single-query source grounding 预测源物体中心和尺寸，再以 Transformer 融合源物体特征、源物体尺寸、语言特征和支撑面点云上下文，直接预测目标 box 底面中心点 heatmap，并在每个候选放置点上联合预测 yaw；同时通过尺寸各向异性加权 yaw loss，使长条物体的 yaw 错误受到更强惩罚，而近似正方形物体的 yaw 错误受到较弱惩罚。

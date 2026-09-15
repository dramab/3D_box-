# LC-BGPlaceNet Stage 1 + Stage 2 模型结构图

## 图示定位

- 论文类型：A 类方法论文，同时具有具身场景中的语言条件物体放置属性。
- 图示结构：单张模型架构图，上下两阶段连续布局，并在 Stage 2 内部放大 SPACE-Former Decoder。
- 视觉主线：共享多模态编码 → Source Grounding → 有界放置集合预测。
- 源码依据：当前工作树中的 `src/models/lc_bgplacenet/stage1.py`、`src/models/lc_bgplacenet/stage2.py` 与对应 YAML 配置。

## 经过源码核对的结构

### Stage 1

1. 输入为 1 cm active voxel 点云、RGB 图像和语言指令。
2. RGB 经过冻结的 CLIP ViT-B/16，得到 14×14 patch 特征。
3. active voxel 通过相机内外参投影到 CLIP 特征图，采样并投影为 64D per-voxel 特征。
4. 64D 图像特征与 `(xyz_norm, rgb)` 6D 特征拼接为 70D，输入三层 Sparse Backbone。
5. CLIP Text Encoder 输出 `text_tokens` 与 `text_global`，均投影为 256D。
6. Voxel-Language Fusion 使用 voxel-to-text Cross-Attention，输出 `F_vl [N,256]`。
7. `F_vl` 与三维位置编码打包为 Transformer memory；单个 learned query 与 `text_global` 相加后进入四层 Source Grounding Decoder。
8. Stage 1 并行输出 `source_box [B,6]` 与 `source_feature [B,256]`，不预测 source yaw、support surface 或 placement。

### Stage 2

1. Stage 2 复用并联合训练 Stage 1 的共享编码和 Source Grounding。
2. `F_vl` 构建 P1/P2/P3 稀疏特征金字塔，stride 为 1/2/4。
3. 完整 `text_tokens` 作为 Q、P3 feature 作为 K/V，通过四层 Text-Guided Region Cross Encoder 产生 region logits。
4. 每个样本选择 top-8 P3 cells，展开其覆盖的 P1 active voxels，再通过 FPS 得到最多 48 个 anchor queries；不足部分 padding。
5. SPACE-Former 使用四层迭代 Decoder：
   - Query Self-Attention；
   - 第 0 层使用 64 点外接圆柱模板；
   - 第 1～3 层使用 previous yaw 构建 64 点定向 Box Surface；
   - 每个点分别在 P1/P2/P3 执行 exact active sparse lookup；
   - Geometric Routing 接收 `source_feature + log(source_size)` 和多尺度 sample tokens；
   - Semantic Routing 对 `text_tokens` 执行 Cross-Attention；
   - Factorized Gated Fusion 后统一预测 center residual、12-bin yaw 与 placement score。
6. Sample Token 为 `sample_feature(256) + active flag(1) + bottom flag(1) + normalized_base_offset(3)`；inactive 零特征 token 不被 Attention 删除。
7. 四层连续更新 bottom center 和 yaw，所有候选框尺寸严格复制当前 `source_size`，Stage 2 不预测尺寸 residual。
8. 48 个 raw boxes 经 score 排序与 Pose NMS 后输出最多 16 个 placement boxes。

## 最终绘图 Prompt

请绘制一张 NeurIPS / CVPR / IEEE TPAMI 风格的 LC-BGPlaceNet 模型结构图。全图必须表达它是一个统一模型，而不是两个彼此独立的模型。使用单张横向画布，上半部分为 Stage 1，下半部分为 Stage 2，并通过 `F_vl`、`text_tokens/text_global`、`source_box/source_feature` 三类跨阶段状态明确连接。

Stage 1 展示三路输入：1 cm active voxel 点云、RGB 和语言指令。RGB 经过冻结的 CLIP ViT-B/16，14×14 patch 特征通过相机投影和双线性采样变成 64D per-voxel 特征，与 6D `(xyz_norm,rgb)` 拼接为 70D，再经过三层 Sparse Backbone 得到 256D voxel feature。语言经过 CLIP Text Encoder 得到 `text_tokens` 与 `text_global`。Voxel-Language Fusion 使用 8-head voxel-to-text Cross-Attention 得到 `F_vl [N,256]`。随后将 `F_vl` 与三维位置编码打包为 memory，Single-Query Transformer Decoder 使用 learned query + `text_global`，经过四层解码后并行输出 `source_box [B,6]` 与 `source_feature [B,256]`。

Stage 2 从 `F_vl` 构建 P1/P2/P3 稀疏金字塔，stride 为 1/2/4。完整 `text_tokens` 作为 Q、P3 feature 作为 K/V，经过四层 Text-Guided Region Cross Encoder 逐格产生 region logits，选择 top-8 P3 cells；展开到 P1 active voxels 并用 FPS 生成最多 48 个 anchor queries。将 SPACE-Former Decoder 作为视觉核心放大：四层迭代，每层先进行 Query Self-Attention；第 0 层以 source size 构造 64 点外接圆柱，第 1～3 层依据 previous yaw 构造 64 点定向 Box Surface；每个点在 P1/P2/P3 做 exact sparse lookup。Geometric Routing 使用 `source_feature + log(source_size)` 与多尺度 sample tokens，Semantic Routing 对 `text_tokens` 做 Cross-Attention，Factorized Gated Fusion 后统一预测 center residual、12-bin yaw 与 placement score。48 个 raw boxes 的尺寸严格复制 source size，经 score sorting 和 Pose NMS 输出最多 16 个 placement boxes。

局部标注 Sample Token：`sample_feature(256) + active flag(1) + bottom flag(1) + normalized_base_offset(3) = 261D`。强调 inactive 零特征 token 仍进入 Attention，用于表达底面踩空或边界净空。底部注明 `world-Z = 支撑面法向 / 重力上方向`、`Stage 2 不预测尺寸 residual`、`有界集合预测而非 dense heatmap argmax`。

视觉规范：纯白背景、二维扁平矢量、低视觉噪声。蓝色表示共享编码，绿色表示 Stage 1 Source Grounding，橙色表示 Stage 2 SPACE-Former，灰色表示辅助说明。模块连接线不能穿过文字。中文标签为主，模型名和张量名保留英文。禁止 3D 透视、阴影、发光、渐变背景、纹理和商业海报风格。

## 自检

- 一张图中只有一条连续主线，没有把 Stage 1 和 Stage 2 画成两个独立模型。
- Stage 1 的 `source_box` 与 `source_feature` 是并行输出。
- Stage 2 的粗区域由完整 `text_tokens` 与 P3 feature 共同预测。
- Anchor 来自 top-8 P3 cells 覆盖的 P1 active voxels，最多 48 个 Query。
- 每 Query、每层、每尺度固定查询 64 个物理采样点。
- 第 0 层是外接圆柱，第 1～3 层是 previous-yaw 定向 Box Surface。
- 输出是最多 16 个 Box 的有界集合，尺寸严格来自 Source Size。

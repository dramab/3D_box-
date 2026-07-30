# SPACE-Former 模型图生成 Prompt

## 论文类型

A 类 Method Paper。图示结构选择：两阶段总览 + SPACE-Former 迭代集合预测细节。

## Prompt

请绘制一张 NeurIPS / CVPR / IEEE TPAMI 风格的二维扁平矢量架构图。白色背景、低视觉噪声、模块边界清晰，中文标签为主并保留模型名与张量名。

方法名为 **SPACE-Former (Size-Prompted Affordance and Collision Explorer)**，输入 active voxel 点云、RGB 与自然语言指令，输出 Source Box 以及最多 16 个满足语言与物理约束的放置 3D Box。

图中包含以下主线：

1. 共享 Stage 1：`Sparse 3D Backbone + CLIP 2D→3D Feature + Text Encoder + Voxel-Language Fusion`，输出 `F_vl`；`Single-Query Source Grounding` 输出 `source_box [B,6]` 和 `source_feature [B,256]`。
2. 多尺度稀疏金字塔：`P1/P2/P3 stride=1/2/4`，对应 1/2/4 cm。
3. `Text-Guided Region Cross Encoder`：`text_global` 为 Q，P3 为 K/V；Cross Block 层数由 `model.space_former.region_cross_num_layers` 配置，当前四层 8-head Cross-Attention 输出 region logits，每个样本取 top-8 P3 cell（约 128 cm²）。
4. Anchor：top-8 cell 展开到 P1 active voxel，经 FPS 得到最多 32 个 Query；不足部分由 `query_valid_mask` padding。Query 初始化为 `LayerNorm(P1 feature + world-coordinate PE)`。
5. 四层迭代物理采样：第 0 层为 64 点外接圆柱；第 1～3 层用上一层 yaw 构造 64 点定向 Box Surface。每个点在 P1/P2/P3 做 exact active sparse hash lookup。
6. Sample Token：`sample_feature(256) + valid(1) + is_bottom(1) + normalized_base_offset(3)`，总计 261 维。强调 inactive 零特征 token 不从 Attention 中删除。
7. `Geometric Routing`：三个尺度分别做 Query-to-sample Cross-Attention，再做跨尺度 Attention；结合 `source_feature + log(source_size)`，只输出 geometry feature。
8. `Semantic Routing`：Query 对 CLIP text tokens 做 Cross-Attention，只输出 semantic feature。
9. `Factorized Fusion`：门控融合 geometry/semantic feature；融合后的 feature 统一送入 center residual、12-bin yaw 和 placement score 三个预测头。
10. 集合输出：32 个 raw boxes 经 score 排序与 pose NMS，输出最多 16 个 Box；所有 Box Size 严格复制 Source Size。
11. 训练虚线支路：全部 direction-filtered positive centers 生成 P3 高斯 region target，24-bin yaw mask 合并为 12-bin multi-hot；Query 与全部真实中心及等量背景虚拟目标进行一对一 Hungarian，真实代价为 Source Size 归一化中心距离，背景代价为 `0.25`。

视觉要求：

- 蓝色表示共享编码，绿色表示 Source Grounding，橙色表示 SPACE-Former，灰色虚线表示训练监督。
- 用一个局部放大框画出 bottom 点、side/top 点、active/inactive lookup 和 `normalized_base_offset`。
- 不使用 3D 透视、阴影、发光、渐变背景或商业海报风格。
- 模块连接线不穿过文字；不展开无关的 Linear/GELU/Dropout。
- 底部标注：`world-Z = 支撑面法向 / 重力上方向`。

## 自检

- Stage 2 是有界集合预测，不是 dense heatmap argmax。
- 粗区域仅使用 P3 与 `text_global`。
- 每 Query 每层每尺度固定 64 个物理采样点。
- 明确区分 bottom 与 non-bottom，并展示 inactive token 的物理含义。
- 输出最多 16 个 Box，尺寸严格来自 Source Size。

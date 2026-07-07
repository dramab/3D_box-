# LC-BGPlaceNet-DPF 模型图生成 Prompt

## 论文类型

A 类 Method Paper。图示结构选择：模型架构图 + 两阶段方法总览图。

## Prompt

你是一位熟悉计算机科学、人工智能、工程制图规范和顶级学术论文视觉表达的科研绘图专家。请绘制一张 NeurIPS / CVPR / IEEE TPAMI 风格的二维扁平矢量模型架构图，白色背景，低视觉噪声，模块边界清晰，颜色只表达功能分区。

本方法名为 **LC-BGPlaceNet-DPF**，用于语言条件的 3D 物体放置框预测。输入是 active voxel 点云和自然语言指令，输出包括需要移动的源物体框 `source_box=(cx,cy,cz,dx,dy,dz)`，以及最终放置框 `place_box=(x,y,z,dx,dy,dz,yaw)`。

请采用“共享体素-语言编码 + Stage 1 source grounding + Stage 2 dense placement field”的结构，而不是简单三栏图。

图中必须包含以下模块与数据流：

1. 输入区：
   - Active voxel 点云，特征为 `XYZ + RGB`
   - 语言指令 `instruction`
   - 训练专用监督：`direction-filtered heatmap`、`support mask`、`source_box_gt`、`place_box_gt`

2. 共享编码区：
   - `Sparse 3D Backbone`，由 `SubMConv3d` 稀疏卷积块提取 `F_3D`
   - `Text Encoder`，使用冻结的 `RoBERTa`，输出 `text_tokens` 和 `text_global`
   - `Voxel-Language Fusion`，以 voxel features 为 query，对 language tokens 做 multi-head cross-attention，输出 `F_vl`
   - `Pack + Position MLP`，将 active voxel 按 batch 打包，并加入 `coords_norm` 位置编码

3. Stage 1：`Single-Query Source Grounding`
   - learnable `source_query` 与 `text_global` 相加
   - Transformer decoder 读取 `voxel_tokens + positional embedding`
   - 输出 `source_box [B,6]` 和 `source_feature [B,C]`

4. Stage 2：`Source-Conditioned Dense Placement Field`
   - `Source Condition`：将 `source_feature` 和 Stage 1 的 `source_size` 输入 `size_mlp` 与 `condition_mlp`
   - 对每个 active voxel 拼接 `F_vl`、`coord PE`、`source_condition`
   - `Dense Placement Fusion MLP` 输出 placement features
   - 四个预测头：
     - `heatmap_head` → `placement_heatmap_logits [Nv]`
     - `offset_head` → `bottom_offset [Nv,3]`
     - `yaw_head` → `yaw_sincos [Nv,2]`
     - `size_head` → `size_residual [B,3]`

5. 解码区：
   - 对每个 batch 在 active voxel 中取 heatmap argmax
   - `bottom_center = voxel_center + bottom_offset`
   - `size_pred = source_size * exp(size_residual)`
   - `yaw = atan2(sin, cos)`
   - `center.z = bottom_center.z + size_pred.z / 2`
   - 输出 `place_box [B,7]`

6. 训练/推理区别：
   - 用虚线标出 `direction-filtered heatmap` 和 `support mask` 只用于训练 loss
   - 推理阶段只读取点云和语言指令，不读取 heatmap 或 support mask

视觉要求：

- 中文标签为主，英文模型名和张量名保留英文
- 全图最多三类颜色：蓝色表示共享编码，绿色表示 Stage 1，橙色表示 Stage 2 / dense placement
- 不使用 3D 透视、阴影、发光、渐变背景、纹理、商业海报风格
- 模块之间连接线不能穿过文字
- 信息密度适中，只保留核心模块，不展开每个 Linear / GELU / Dropout
- 在图底部用一行小字标注：canonical world 满足 `world-Z = 支撑面法向/重力上方向`

## 自检

- 已使用方法模块描述，不是只基于摘要。
- 已声明论文类型为 Method Paper。
- 主线为 Stage 1 源物体定位到 Stage 2 稠密放置场。
- 训练专用监督与推理路径分离。
- 图形语言为二维扁平矢量，无装饰性效果。

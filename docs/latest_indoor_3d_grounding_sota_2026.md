# 面向 active_aligned 的室内端到端 3D Visual Grounding 调研（截至 2026-08-12）

## 摘要

本调研回答三个问题：当前最新的室内 3D Visual Grounding（3DVG）模型有哪些；哪些模型真正符合 `active_aligned` 的“场景几何 + 自然语言 → 原物体 3D 框”任务；哪些模型已经具备可复现代码，适合进入论文对比实验。

结论是：**PV-Ground（CVPR 2026）应作为当前首选主对比，TSP3D（CVPR 2025）应作为成熟稳定的第二主对比，MCLN（ECCV 2024）可作为经典强基线。** 三者都支持不依赖候选物体框的 single-stage 设置，其中 PV-Ground 在 ScanRefer single-stage 上报告 `Acc@0.25=59.31`、`Acc@0.5=47.77`，且官方仓库已提供训练、测试代码和单阶段权重。[1] TSP3D 报告 `56.45/46.71`，代码与权重完整，结构也更轻量。[2]

S²-MLLM、GS-Reasoner 等大模型路线的数字更高，但它们使用 16–24 帧多视角图像、视频语言模型或额外几何编码，训练数据和输入预算与当前单帧 RGB-D/点云设定不同，不能直接替换主对比。[3,4] ORD、MiKASA、CoT3DRef 等方法主要从预分割对象或候选框中选择目标，也不应与“原始点云直接回归 3D 框”的模型放在同一主表中。[5–7]

---

## 1. 研究问题与筛选口径

### 1.1 研究问题

1. 2024–2026 年室内 3DVG 中，哪些模型在 ScanRefer、Nr3D、Sr3D 上达到最新水平？
2. 哪些模型满足 `active_aligned` 的目标：仅根据场景与语言定位待移动的原物体？
3. 哪些方法已有足够完整的官方代码、权重与训练入口，可用于公平微调和复现？

### 1.2 本项目中的“端到端”定义

为了避免不同论文对 end-to-end 的宽松用法造成混淆，本项目建议采用严格定义：

> 推理时输入完整场景几何与文本，不使用 GT 2D 框、GT 3D 框、GT instance mask 或 GT object proposal，模型直接输出目标物体 3D 框。

据此，模型分为四类：

1. **严格 single-stage 3DVG**：原始点云/体素 + 文本 → 目标 3D 框，是主对比。
2. **proposal-based grounding**：从外部检测器或 GT 对象候选中选择目标，只能作为辅助对比或 oracle。
3. **多视角 3D-MLLM**：多帧图像/视频 + 几何编码 + 文本 → 3D 框，能力强但输入与算力不同。
4. **zero-shot/open-vocabulary 管线**：借助 2D VLM、渲染或候选标记完成定位，应与监督微调结果分表报告。

---

## 2. 最新方法总览

下表的 ScanRefer 数值统一写为 `Acc@0.25 / Acc@0.5`。不同方法的输入帧数、额外训练数据和候选框设置并不总是相同，因此数字只用于了解公开基准位置，不能代替在 `active_aligned` 同一拆分上的复现实验。

| 方法 | 年份/会议 | 推理输入与输出 | 严格端到端 | ScanRefer | 官方实现状态 | 与本项目匹配度 |
|---|---|---|---:|---:|---|---|
| **PV-Ground** | CVPR 2026 | 点云/体素 + 文本 → 3D 框 | 是，支持 single-stage | **59.31 / 47.77** | 代码、训练/评测脚本、单阶段权重已发布 | **最高** |
| **TSP3D** | CVPR 2025 | 稀疏体素 + 文本 → 3D 框 | 是 | **56.45 / 46.71** | 代码与权重完整 | **很高** |
| **MCLN** | ECCV 2024 | 点云 + 文本 → 框/掩码 | 支持 single-stage | 54.30 / 42.64 | 代码与权重可用 | **高** |
| DDPA-3DVG | IJCAI 2025 | 点/体素多粒度特征 + 文本 → 3D 框 | 支持 single-stage | 54.30 / 42.20 | 官方仓库目前主要是 README/图片 | 思路很匹配，暂不可复现 |
| EG-3DVG | CVPR 2026 | 几何与表达式联合解码 → 框/掩码 | 论文包含直接定位路线 | 论文声称 ScanRefer/ReferIt3D SOTA | 未检索到完整官方实现 | 高，但暂不宜进入主实验 |
| S²-MLLM | CVPR 2026 | 16/24 帧多视角图像 + 结构编码 → 3D 框 | 无外部候选框，但非点云同输入 | **60.59 / 53.66（24 帧）** | 代码与训练/评测入口已发布 | 中，适合扩展实验 |
| GS-Reasoner | ICLR 2026 | 多视角/视频 + 几何表示，生成式输出 3D 框 | 不依赖外部候选框 | 60.8 / 42.2 | 代码和模型已发布，流程较重 | 中，适合 3D-LLM 扩展实验 |
| ORD | CVPR 2026 | 对象候选 + 文本 → 候选选择/回归 | 否，依赖 proposals | NR3D 71.6；Sr3D 76.2 | 未检索到完整官方实现 | 关系建模相关，但非同设定 |
| CoT3DRef | ICLR 2024 | 预提取对象序列 + 文本 → 锚点链/目标 | 否 | 主要报告 ReferIt3D | 代码可用 | 仅适合关系推理辅助实验 |
| Z3D | ACL 2026 | 多视角图像 + VLM/候选生成 → 3D 结果 | 否，zero-shot 管线 | zero-shot SOTA 路线 | 代码可用 | 可替换/补充 DetAny3D，但需多视角 |
| UZ3DVG | CVPR 2026 | 点云 + 文本 → 3D 框，zero-shot | 推理阶段是 | 论文声称 zero-shot SOTA | 官方仓库尚未发布主体代码 | 暂不能实测 |

---

## 3. 重点技术路线

### 3.1 严格 single-stage：最适合主对比

#### PV-Ground（首选）

PV-Ground 针对点式骨干在强下采样后丢失小物体几何细节的问题，结合稀疏体素的高分辨率表示与文本引导的关键点采样。它可以把同一套 point-voxel 特征接入 BUTD-DETR、EDA、MCLN 等 grounding decoder；在 MCLN single-stage 上由 `54.30/42.64` 提升到 `59.31/47.77`，对多干扰物体子集的提升尤其明显。[1]

这与 `active_aligned` 高度相似：桌面小物体尺寸小、遮挡与点云缺失明显、同类实例容易混淆，而且场景本身已有 RGB-D/点云。官方仓库目前包含 44 次提交、完整模型目录、数据处理、single-stage 训练与评测脚本，并提供预训练模型链接，具备实际复现条件。[1]

需要注意：官方环境基于 Python 3.12、PyTorch、spconv/OpenPCDet 和自定义 PointNet2 扩展。为了不干扰项目已有环境，应创建独立环境，不能直接安装进 `spatial`。

#### TSP3D（稳定第二主对比）

TSP3D 使用文本引导的稀疏体素剪枝逐步缩小候选空间，并通过 completion-based addition 恢复可能被过度剪掉的目标区域。它是原始 3D 场景与文本直接融合的 single-stage 模型，报告 `56.45/46.71`，且强调实时性。[2]

它的优势是代码发布时间更早、复现资料更成熟、结构相对清晰；劣势是官方依赖较旧，环境迁移可能涉及 MinkowskiEngine/CUDA 版本兼容。作为独立环境运行仍然可控。

#### MCLN（经典强基线）

MCLN 用两个协作分支同时学习 3D referring expression comprehension 与 referring segmentation，并通过相对 superpoint 聚合和自适应软对齐让框定位与掩码预测互相促进。[8] 它不是最新榜首，但代码完整、被 PV-Ground 直接采用为强 decoder 基线，适合作为“标准端到端 3D grounding”参照。

DDPA-3DVG 对 target、reference 与 relation 分支进行解耦，EG-3DVG 则强化文本表达式、几何一致性与同类实例区分；两者都很契合桌面场景中“关系描述 + 相似物体干扰”的难点。[14,15] 但 DDPA 官方仓库当前没有可运行主体代码，EG-3DVG 也未检索到完整官方实现，因此现阶段适合写入 related work，不适合承诺可复现主实验。

### 3.2 多视角 3D-MLLM：适合扩展，不适合作为唯一主对比

S²-MLLM 使用 16 或 24 帧多视角图像，通过训练阶段的结构指导、跨视角注意力和多级位置编码增强视频语言模型的隐式 3D 推理。24 帧设置在 ScanRefer 达到 `60.59/53.66`，并在 MultiScan 和 ARKitScenes 上表现出较强 OOD 泛化。[3] 官方代码已经公开，但它需要多视角数据、相机/场景元信息以及 LLaVA 系列大模型，训练和输入预算显著大于点云 single-stage 模型。

GS-Reasoner 让 3D-LLM 自回归生成 3D 框，并用 grounded chain-of-thought 与双路径几何池化增强空间推理；论文报告 `60.8/42.2`。[4] 其优势是统一 grounding 与空间推理，缺点是模型更重，而且论文结果还使用 ScanRefer、Multi3DRef 等多源训练数据，不能与只在 `active_aligned` 上微调的模型直接横比。

因此，如果 `active_aligned` 每个场景只有单帧 RGB-D，S²-MLLM/GS-Reasoner 不应进入主表；如果场景保留了 16 帧以上多视角 RGB、深度、内外参，则可以选择其中一个作为“现代 3D-MLLM”扩展对比，并单列输入预算与预训练数据。

### 3.3 proposal-based 关系模型：与语言关系相似，但定位设定不同

ORD、MiKASA、CoT3DRef 和 ViewSRD 的核心优势都在关系建模：显式寻找 reference object/anchor，再利用相对位置定位 target。它们与 `active_aligned` 中“位于某物体左/右/前/后/附近的原物体”在语言结构上很相似。[5–7,9]

但这些方法常在 ReferIt3D 设置中接收已分割对象、GT instance 或检测器 proposals，模型的主要任务是“从候选中选对”，而不是从原始点云中发现并回归物体。若将它们纳入实验，必须满足以下任一条件：

- 所有 proposal-based 方法共用同一个预测式 3D 检测器，并把检测召回率计入总结果；
- 使用 GT proposals，但明确标记为 **oracle object-candidate grounding**，不进入端到端主表。

### 3.4 zero-shot/open-vocabulary：应与监督微调分表

Z3D、SeeGround、UZ3DVG 以及当前 GroundingDINO → DetAny3D 都属于 zero-shot 或模块化路线。[10–12] 它们回答的是“无需在目标数据上训练能否定位”，而 PV-Ground/TSP3D 微调回答的是“在相同训练集监督下，哪个端到端架构更强”。两类结果有价值，但不能合并排序。

ViGiL3D 的诊断结果也提示，现有开放词汇 3DVG 方法在复杂、分布外描述上的表现仍明显不足；ScanRefer 上的高分不自动等价于桌面操作指令鲁棒性。[13]

---

## 4. 对 active_aligned 的最终实验建议

### 4.1 推荐对比矩阵

#### 主表：同输入、同监督、端到端原物体定位

1. **Ours**：当前端到端 3D grounding/placement 模型。
2. **PV-Ground single-stage + fine-tuning**：最新、最匹配的主对比。
3. **TSP3D + fine-tuning**：成熟且高效的端到端强基线。
4. **MCLN single-stage + fine-tuning**：经典强基线；若算力有限，可在 TSP3D 与 MCLN 中只保留一个，优先 TSP3D。

#### 单独表：zero-shot / open-vocabulary

1. GroundingDINO → DetAny3D（当前已有）。
2. Z3D（仅在能够提供多视角图像时加入）。
3. UZ3DVG（等官方主体代码发布后再加入）。

#### 可选扩展表：大模型或 proposal oracle

- S²-MLLM：仅当数据具备多视角帧、相机参数且算力允许。
- GS-Reasoner：用于展示大模型空间推理路线，不作为最接近的主基线。
- ORD/CoT3DRef：只做 GT proposal oracle 或统一检测器后的候选选择实验。

### 4.2 是否需要微调

**主表必须微调。** 原因不是为了让对手变强，而是因为 ScanNet 房间级点云与 `active_aligned` 桌面/室内局部场景在类别、物体尺度、点密度、语言模板和 3D 框定义上都有明显域差异。只比较预训练 zero-shot，会主要测量域迁移和词汇覆盖，而不是模型结构能力。

公平设置应为：相同 train/valid/test 拆分、相同输入点云、相同 source grounding 文本、相同 GT 框定义、相同主指标。公开预训练权重可作为初始化，但所有监督式基线都在相同 `active_aligned` train 上微调，并以 valid 选择 checkpoint。

### 4.3 提示词/文本适配

点云型 3DVG 模型通常不需要对话式 prompt，而需要一个简洁的 referring expression。当前完整操作指令包含“原物体”和“放置目标”两部分；若只测试原物体定位，目标短语会引入无关实体和关系，必须去掉。

建议从结构化标注字段生成唯一的 source expression，不要依赖字符串切分：

```text
原指令：Move the blue mug located to the left of the red bowl to the tray.
定位输入：the blue mug located to the left of the red bowl
```

统一模板建议为：

```text
the <source attributes> <source category> <source relation> <reference attributes> <reference category>
```

规则如下：

- 保留区分 source 实例所需的颜色、形状、大小与 source-reference 空间关系。
- 删除 `move`、`place`、`put` 等动作词以及完整 destination clause。
- reference object 只在它能帮助消歧时保留；若 source 在场景中唯一，同时评测“类别直接定位”和“完整关系描述”两个子集更有分析价值。
- 所有模型使用同一条 canonical English expression；不要为不同模型改写成语义不同的提示词。
- S²-MLLM/GS-Reasoner 可以套用其官方 conversation template，但其中的 referring expression 内容必须保持一致。

### 4.4 输入与框定义的关键公平性问题

1. **坐标系**：输入点云、相机外参和 GT 框必须经过同一个 canonical world 变换，并满足 `world-Z = 支撑面法向/重力上方向`。
2. **AABB 与 OBB**：ScanRefer 系模型大多预测 axis-aligned 3D box，而本项目可能使用带 yaw 的 OBB。不能直接把 AABB 与 OBB 用同一 IoU 阈值横比而不说明转换。
3. **推荐指标**：同时报告 center-distance recall（例如 10/20/30 cm）、AABB IoU@0.25/0.5；若将各模型 head 统一改成 yaw-aware OBB，再额外报告 oriented IoU。
4. **候选框披露**：任何 GT proposal、GT mask 或预检测框都必须单独标注。严格主表不允许 GT 候选。
5. **场景裁剪一致**：所有模型看到完全相同的点云范围与点数预算，避免某模型看到全场景、另一模型只看到目标附近 crop。

---

## 5. 结论与实施顺序

当前最合理的实施顺序是：

1. **先适配 PV-Ground single-stage**：它是最新、公开、指标最强且输入最接近的 classical 3DVG 模型。
2. **再适配 TSP3D**：提供成熟且高效的第二端到端参照。
3. 根据时间决定是否加入 MCLN；若论文表格需要三条外部监督基线，加入 MCLN，否则 PV-Ground + TSP3D 已覆盖“最新 SOTA”和“成熟高效”两个维度。
4. 保留 DetAny3D 作为 zero-shot 模块化基线，不与微调后的端到端模型混为一组。
5. 只有在多视角数据与算力条件明确满足时，再加入 S²-MLLM；否则它会改变研究问题，而不是简单增加一个公平基线。

一句话判断：**如果论文主张的是“室内场景中，根据语言直接定位待移动原物体”，主对比应选 PV-Ground/TSP3D 这类同输入、可微调、无 GT proposals 的 single-stage 3D grounding 模型；DetAny3D 只作为 zero-shot 补充，proposal-based 与多视角 MLLM 单独成表。**

---

## 参考文献与官方资源

[1] Junpeng Shang et al. *PV-Ground: Text-Guided Point-Voxel Interaction for 3D Visual Grounding*. CVPR 2026. 论文：https://openaccess.thecvf.com/content/CVPR2026/html/Shang_PV-Ground_Text-Guided_Point-Voxel_Interaction_for_3D_Visual_Grounding_CVPR_2026_paper.html；代码：https://github.com/AaNnWwTt/PV-Ground

[2] Wenxuan Guo et al. *Text-guided Sparse Voxel Pruning for Efficient 3D Visual Grounding*. CVPR 2025. 论文：https://openaccess.thecvf.com/content/CVPR2025/html/Guo_Text-guided_Sparse_Voxel_Pruning_for_Efficient_3D_Visual_Grounding_CVPR_2025_paper.html；代码：https://github.com/GWxuan/TSP3D

[3] Beining Xu et al. *S²-MLLM: Boosting Spatial Reasoning Capability of MLLMs for 3D Visual Grounding with Structural Guidance*. CVPR 2026. 论文：https://openaccess.thecvf.com/content/CVPR2026/papers/Xu_S2-MLLM_Boosting_Spatial_Reasoning_Capability_of_MLLMs_for_3D_Visual_CVPR_2026_paper.pdf；代码：https://github.com/IRMVLab/S2-MLLM

[4] Yiming Chen et al. *Reasoning in Space via Grounding in the World*. ICLR 2026. 论文：https://openreview.net/forum?id=CfKi92bgnq；代码：https://github.com/WU-CVGL/GS-Reasoner

[5] Zhifan Huang et al. *ORD: Object-Relation Decoupling for Generalized 3D Visual Grounding*. CVPR 2026. https://openaccess.thecvf.com/content/CVPR2026/html/Huang_ORD_Object-Relation_Decoupling_for_Generalized_3D_Visual_Grounding_CVPR_2026_paper.html

[6] Chun-Peng Chang et al. *MiKASA: Multi-Key-Anchor & Scene-Aware Transformer for 3D Visual Grounding*. CVPR 2024. https://openaccess.thecvf.com/content/CVPR2024/html/Chang_MiKASA_Multi-Key-Anchor__Scene-Aware_Transformer_for_3D_Visual_Grounding_CVPR_2024_paper.html

[7] Eslam M. Bakr et al. *CoT3DRef: Chain-of-Thoughts Data-Efficient 3D Visual Grounding*. ICLR 2024. https://proceedings.iclr.cc/paper_files/paper/2024/hash/32f9049217da6e718a426b07242dff73-Abstract-Conference.html

[8] Zhipeng Qian et al. *Multi-branch Collaborative Learning Network for 3D Visual Grounding*. ECCV 2024. 论文：https://eccv.ecva.net/virtual/2024/poster/2256；代码：https://github.com/qzp2018/MCLN

[9] Ronggang Huang et al. *ViewSRD: 3D Visual Grounding via Structured Multi-View Decomposition*. ICCV 2025. https://openaccess.thecvf.com/content/ICCV2025/html/Huang_ViewSRD_3D_Visual_Grounding_via_Structured_Multi-View_Decomposition_ICCV_2025_paper.html

[10] Nikita Drozdov et al. *Z3D: Zero-Shot 3D Visual Grounding from Images*. ACL 2026. 论文：https://aclanthology.org/2026.acl-short.13/；代码：https://github.com/col14m/z3d

[11] Yunhan Li et al. *SeeGround: See and Ground for Zero-Shot Open-Vocabulary 3D Visual Grounding*. CVPR 2025. 论文：https://openaccess.thecvf.com/content/CVPR2025/papers/Li_SeeGround_See_and_Ground_for_Zero-Shot_Open-Vocabulary_3D_Visual_Grounding_CVPR_2025_paper.pdf；代码：https://github.com/iris0329/SeeGround

[12] Wenbo Tan et al. *UZ3DVG: Unaided Zero-Shot 3D Visual Grounding with Generated Language Conditions*. CVPR 2026. 论文：https://openaccess.thecvf.com/content/CVPR2026/html/Tan_UZ3DVG_Unaided_Zero-Shot_3D_Visual_Grounding_with_Generated_Language_Conditions_CVPR_2026_paper.html；仓库：https://github.com/tanwb/UZ3DVG

[13] Austin Wang et al. *ViGiL3D: A Linguistically Diverse Dataset for 3D Visual Grounding*. ACL 2025. https://aclanthology.org/2025.acl-long.1470/

[14] GwangWook Park et al. *EG-3DVG: Expression and Geometry Aware Grounding Decoder for 3D Visual Grounding*. CVPR 2026. https://openaccess.thecvf.com/content/CVPR2026/html/Park_EG-3DVG_Expression_and_Geometry_Aware_Grounding_Decoder_for_3D_Visual_CVPR_2026_paper.html

[15] Hongjie Gu et al. *DDPA-3DVG: Vision-Language Dual-Decoupling and Progressive Alignment for 3D Visual Grounding*. IJCAI 2025. 论文：https://www.ijcai.org/proceedings/2025/117；仓库：https://github.com/HDU-VRLab/DDPA-3DVG

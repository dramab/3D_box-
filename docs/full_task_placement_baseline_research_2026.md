# RePlace3D 完整放置任务的主对比模型调研：不存在单一完全对齐基线，分层比较比堆模型更有说服力

## 摘要

本文调研截至 2026 年 8 月 25 日与 RePlace3D 最相关的语言条件 3D 放置、3D MLLM、几何放置和空间 affordance 方法。RePlace3D 的严格任务是从观测场景和完整移动指令中定位 source object，并预测多个保持 source 尺寸的 4-DoF 放置框；成功候选还必须同时满足语言关系、稳定支撑和无碰撞。现有工作通常只覆盖这条链路的一部分：PlaceWizard 最接近语言条件真实 3D 场景放置，但输入是单独提供的 3D asset；AnyPlace、M2T2 和 RPDiff 能输出几何放置 pose，但也要求单独的待放物体点云；RoboPoint/RoboBrain2.5 主要输出图像点；LLaVA-3D 等通用 3D MLLM 可以被微调为坐标生成器，但缺少显式支撑与碰撞归纳偏置。因此，最有说服力的实验不是把这些方法在含有不同 oracle 信息的条件下强行排序，而是采用两张互补表：严格端到端系统表和 GT source box 条件下的 placement-only 解耦表。严格主表建议至少包含 PlaceWizard、LLaVA-3D、AnyPlace 和 RoboBrain2.5 四条代表路线，并使用统一的预测式 source grounder 补齐非端到端方法。RPDiff 更适合作为 GT source 条件下的多模态几何 pose 基线；M2T2 可作为追加的机器人 placement primitive，但其原生输出是 target gripper pose，当前 benchmark 又没有 object-to-gripper transform，所以适配不如 AnyPlace 直接。FirePlace 与 GOPLA 虽然高度相关，但当前公开代码可用性不足，不宜成为必须完成的主实验。

## 1. 研究问题

本调研回答三个问题：

- **RQ1：** 哪些公开模型能在相同或可合理适配的输入下预测放置中心和 yaw，而不依赖测试时 GT source、GT anchor、GT mask 或外部 3D asset？
- **RQ2：** 如何公平比较端到端 source-grounding-to-placement 系统与默认已获得待放物体点云的 placement-only 方法？
- **RQ3：** 在论文说服力、覆盖的方法范式、代码可用性和适配成本之间，最小而充分的主对比阵容是什么？

目标读者是准备 ICLR 2027 投稿实验的作者。本文的核心判断是：**没有一个现有公开模型与 RePlace3D 的“场景内 source grounding + 多解 4-DoF box placement + 显式物理成功”完全同构；主实验必须用协议设计补齐设定差异，而不能靠模型名称掩盖输入 oracle。**

## 2. 调研方法

检索从五个互补方向展开：

1. 直接的 language-guided 3D object placement；
2. 从 object/scene point clouds 预测相对 placement pose 的几何模型；
3. language-conditioned 3D manipulation 与 contact-mask 模型；
4. 2D/3D MLLM 的空间点、3D 坐标和 placement 生成；
5. 多解放置、物理有效性和关系满足的评测方法。

关键词包括 `language-guided 3D object placement`、`relative pose diffusion rearrangement`、`orientation-aware placement point cloud`、`spatial affordance point prediction`、`3D MLLM placement` 和 `scene rearrangement referring expression`。纳入工作必须满足至少一项：直接输出 placement location/pose；能以合理改动适配该输出；或提供当前任务必需的对照范式。仅输出抓取、完整机械臂轨迹、纯场景生成或仅做 3D grounding 的工作不进入推荐主阵容。论文存在性、任务输入输出和代码状态使用论文主页、正式论文页或官方仓库交叉核验。

## 3. 方法谱系

### 3.1 直接语言条件 3D 放置

PlaceIt3D/PlaceWizard [1] 与当前任务最接近：它接收真实场景点云、单独的 3D asset 和语言 prompt，预测 placement location、anchor 和 rotation。它与 TopoPlacer 都处理多解、场景几何、语言关系和旋转；区别是 PlaceWizard 获得场景外单独 asset，而 TopoPlacer 必须先从同一观测场景中 grounding source。PlaceWizard 因而是**必须比较的最近工作**，但必须将独立 asset 替换为由预测 source box 裁出的观测点云，或在 GT-source 表中明确标记 oracle source。

FirePlace [2] 和 GOPLA [3] 同样把语义偏好与几何可行性结合起来，但路线不同：FirePlace 让 MLLM 生成几何约束并通过表面提取、约束求解和视觉筛选产生 placement；GOPLA 将 MLLM 的结构化关系计划转成 3D affordance map，再用带碰撞代价的 diffusion planner 生成 pose。二者都很适合 Related Work，也能覆盖“神经-符号/规划式系统”这一反方路线。然而，截至检索日，FirePlace 项目页未提供可复现代码入口，GOPLA 的公开 `Code` 链接返回不可用仓库，因此它们不应被写成提交前必须完成的主基线。

最新的 ReRef-3D [4] 并不是一个新模型，而是一个 CLEVR 派生的语言 3D scene rearrangement benchmark。它把 LLaVA-3D、3D-LLM 和 PlaceIt3D 都微调为 placement position 生成器，并使用重新插入预测物体后计算关系和物理有效性的评测。这项工作提示两个结论：通用 3D MLLM 已经是审稿人可能期待的对照；同时，仅报告关系满足会高估完整任务能力，因为各模型的 relation satisfaction 普遍高于 physical validity。它只预测有效位置、场景为合成数据，不能替代 RePlace3D 的 yaw 和真实 RGB-D 物理评测。

### 3.2 几何 pose 与接触建模

AnyPlace [5]、M2T2 [6] 和 RPDiff [7] 都比 2D point 模型更接近 TopoPlacer 的 pose 输出，但三者回答的问题不同。AnyPlace 用 VLM 提议粗放置位置，再在局部 object/scene point clouds 上用 diffusion 模型生成精确 SE(3) pose；这与 TopoPlacer 的 coarse-to-fine 逻辑直接竞争。其官方仓库已经公开低层训练和评测代码、数据与 checkpoint，但仓库仍将完整 high-level inference 标记为 `coming soon`，因此适配成本较高。

M2T2 从原始场景点云和待放物体的 partial point cloud 预测 orientation-specific contact masks，再恢复多组 6-DoF target gripper poses；它还提供 CLIP language-conditioned 版本。与 AnyPlace 相比，M2T2 的优点是明确支持 partial object point cloud，更符合“物体直接从场景中观测”的限制；缺点是训练数据不能公开，而且从 target gripper pose 转成 benchmark 所需的 object bottom center 与 yaw 需要已知 object-to-gripper transform。当前数据没有抓取姿态，若直接假设单位变换会引入不合理的隐式条件。因而 M2T2 适合作为资源充足时的机器人 placement primitive 对照，直接输出 object transform 的 AnyPlace 更适合优先实现。

RPDiff 通过迭代去噪 object point cloud 的 SE(3) pose 来拟合多模态 placement 分布，直接检验“diffusion 是否比有界 query set 更适合覆盖多解”。但 RPDiff 不原生理解开放语言，而是面向固定的 shelving、stacking、hanging 等关系任务。若为它新增完整语言模型和 source/anchor grounding，得到的将是大量自定义模块而不再是忠实复现；因此它最适合 placement-only 表，在 GT source、GT relation token、GT reference crop 条件下作为几何多解对照。

Paxton 等 [8] 的经典方法恰好预测 3D translation 与 planar rotation，并用关系分类器、learned scene discriminator 和 CEM 同时优化语义关系与稳定性。它与 RePlace3D 的 4-DoF 输出高度一致，但输入包含带实例标签的点云和逻辑 predicate，而不是自然语言与原始场景。它仍值得作为“传统采样优化 + learned feasibility”参考；若实现预算有限，可用更简单的 `Support+Relation Sampling` 复现其方法思想，而不必重建完整旧系统。

### 3.3 通用 3D MLLM 与 2D 空间 affordance

LLaVA-3D [9] 将带 3D 位置编码的多视图 visual tokens 对齐到 LLaVA，并公开了 7B checkpoint、自定义 instruction tuning 流程和 3D localization 代码。ReRef-3D 对其 placement 微调的结果进一步验证了可适配性。它应作为“通用 3D MLLM 直接生成结构化坐标”的主对照；统一输出格式可设为 source box 加 Top-1 或 Top-K `(x,y,z,yaw)`。如果每个 canonical 样本只有单帧 RGB-D，必须明确它只得到这一帧，不能额外提供 ScanNet 风格多视图。

RoboPoint [10] 和 RoboBrain2.5 [11] 都代表 2D image point 路线。RoboPoint 针对 spatial affordance pointing 训练；RoboBrain2.5 进一步宣称支持带 depth 的 3D spatial referring、metric measuring 和 trace。当前项目已经实现 RoboBrain2.5 zero-shot 与 LoRA bottom-center `(x,y,d)` SFT，这使它成为成本最低且有现实价值的 point-only 对照。不过它不预测 source 3D box、source size 或 yaw，不能直接以 oracle geometry 混入严格端到端排名。严格表中应使用预测式 source grounder 的尺寸和 yaw，placement yaw 采用“复制预测 source yaw”的固定策略；使用 GT source geometry 的现有结果只能进入 oracle-conditioned 子表。

Reason3D [12] 是 PlaceWizard 的直接基础之一，能从点云和文本输出 segmentation mask。由于 PlaceWizard 已经包含了从 Reason3D 到 placement 的针对性扩展，二者同时进入主表的增量价值较小；Reason3D 更适合作为 PlaceWizard 消融或 source/anchor grounding 辅助分析，而不是第五个独立主模型。

## 4. 候选模型比较

表 1 的结论是：PlaceWizard 提供最近任务对照，LLaVA-3D 提供通用生成式 3D MLLM 对照，AnyPlace 提供 coarse-to-fine diffusion object-pose 对照，RoboBrain2.5 提供 2D point 路线对照；M2T2 有接触建模价值，但缺失 object-to-gripper transform 会使其适配风险更高。

| 方法 | 原生输入 | 原生输出 | 多解 | 语言 | 物理/几何 | 公开可复现状态 | 对主实验的定位 |
|---|---|---|---:|---:|---:|---|---|
| PlaceWizard [1] | scene point cloud + 独立 asset + prompt | 位置 mask、anchor、rotation | 是 | 是 | 间接学习 | 论文、代码、数据、checkpoint 已公开 | **P0，最近任务** |
| LLaVA-3D-7B [9] | posed RGB-D/multi-view + prompt | 自回归文本、3D grounding | 可生成 | 是 | 无显式约束 | 代码、checkpoint、custom tuning 已公开 | **P0，通用 3D MLLM** |
| M2T2-L [6] | scene point cloud + partial object cloud + language | orientation-aware target gripper poses | 是 | 是 | contact mask、碰撞导向 | 代码和权重公开；训练数据及 benchmark 所需 grasp transform 缺失 | P1，机器人 primitive 追加项 |
| RoboBrain2.5-8B-NV [11] | RGB + prompt | 2D/3D point 或 trace | 有限 | 是 | 弱、主要靠预训练 | 权重、推理代码公开；项目已有适配 | **P0，2D point/VLM** |
| AnyPlace [5] | RGB-D + object cloud + language | 多个精细 object SE(3) transforms | 是 | 是 | 局部几何、diffusion | 低层代码/数据/权重公开；完整高层推理未完成 | **P0，几何/diffusion** |
| RPDiff [7] | object cloud + scene crop + task relation | 多个 SE(3) poses | 是 | 否 | 局部几何、diffusion | 代码公开 | **P1，GT-source 表** |
| Stable Configurations [8] | instance clouds + predicates | 3D translation + planar rotation | 采样式 | predicate | discriminator + CEM | 论文与历史代码可查 | P1，传统规划思想 |
| FirePlace [2] | clean 3D scene + asset + language | constraint-solved placements | 是 | 是 | 显式表面约束 | 未检索到完整公开代码 | Related Work/定性，不承诺主实验 |
| GOPLA [3] | RGB-D + language | affordance map + diffusion pose | 是 | 是 | test-time collision cost | 项目代码链接当前不可用 | 高相关但暂不列为必做 |
| Reason3D [12] | point cloud + text | 3D mask + text | 否 | 是 | 无 | 代码/checkpoint 已公开 | 被 PlaceWizard 覆盖，不单独主比 |

## 5. 推荐实验矩阵

### 5.1 表 A：严格端到端主表

严格端到端定义为：测试时只输入 canonical RGB-D/point cloud 与完整指令；不提供 GT source box、GT source mask、GT source dimensions、GT source yaw、GT anchor 或外部完整 3D asset。每个系统必须输出 source box 和最多 K 个 placement boxes。

推荐行如下：

1. **TopoPlacer（Ours）**。
2. **PV-Ground → PlaceWizard-Observed**：PV-Ground [13] 在相同训练集上使用完整移动指令监督 source box，不从 GT 结构化字段抽取 source expression；其 box head 统一改成 yaw-aware OBB。框内观测点作为 PlaceWizard asset input，仍保留原 source 在场景点云中。PlaceWizard 接收完整 instruction。该组合代表最近的 3D placement model。
3. **PV-Ground → AnyPlace-Observed**：用预测 source crop 替换完整 object asset；以完整指令产生 coarse location，再复用 low-level diffusion object-pose model。该组合代表 coarse-to-fine diffusion placement。
4. **LLaVA-3D-7B-SFT-Joint**：直接在相同训练集上微调，结构化生成 `source_box` 与 Top-K `(bottom_center,yaw)`；只提供当前 benchmark 实际拥有的视图。该行代表通用 3D MLLM。
5. **PV-Ground → RoboBrain2.5-LoRA-SFT**：共享的 PV-Ground 输出 source box；RoboBrain2.5 输出 placement `(x,y,d)`；使用预测 source dimensions，并固定复制预测 source yaw，禁止使用 GT geometry。该行代表 image-only point affordance 路线。现有 DetAny3D [14] 结果可作为 zero-shot source-grounder 变体另行报告，但它当前使用精确 source 名词 prompt，不属于只输入完整指令的严格主协议。
6. **可选：PV-Ground → M2T2-L-Adapted**：仅在能够为所有样本定义不借用 GT 的 object-to-gripper transform 时加入；否则该行不构成当前 endpoint benchmark 的公平 object-pose 输出。

这里使用 PV-Ground 而不是 TopoPlacer Stage 1 作为公共 source grounder，是为了避免外部 placement baseline 借用 Ours 的核心模块。它必须从完整指令直接学习 source grounding，不能读取测试样本的结构化 source phrase；所有模块化 baseline 共享它的同一个 checkpoint，Source IoU 因而一致并单独报告。若最终不实施 PV-Ground，则可以使用统一的另一个公开 source grounder，但必须遵守相同的 full-instruction 与 yaw-aware OBB 协议。

### 5.2 表 B：GT source box 条件下的 placement-only 解耦表

该表回答“如果每个模型都知道移动哪个物体及其观测尺寸，谁最会找位置和 yaw”。输入 GT source box 只用于裁出 source partial point cloud 和读取观测尺寸；不能提供 GT placement center/yaw，也不能提供数据生成阶段的 valid support mask。

推荐行如下：

1. TopoPlacer-GTSource；
2. PlaceWizard-Observed-GTSource；
3. AnyPlace-Observed-GTSource；
4. M2T2-L-GTSource（需披露 object-to-gripper transform）；
5. RPDiff-Relation-GTSource/GTReference；
6. RoboBrain2.5-LoRA-GTGeometry，明确标为 point-only oracle-geometry-conditioned；
7. `Support+Relation Sampling` 传统几何基线。

RPDiff 若使用 GT reference crop 和结构化 relation token，必须在名称中写出这两个 oracle；不能与只使用自然语言和原始场景的模型无标记横比。

### 5.3 表 C：任务内简单基线与消融

外部模型不能代替任务内 sanity checks。至少保留：

- **Dense Voxel Heatmap + Yaw Head**：与 TopoPlacer 使用相同 backbone、language features 和 source size，只用 dense heatmap 取 Top-K 局部极大值，再独立预测 yaw；检验 query refinement/PABR 的贡献。
- **Support-only Sampling**：从预测或纯几何提取的支撑区域均匀采样，复制 source yaw；检验数据集中是否存在强支撑面捷径。
- **Relation-only Centroid**：按 reference object 与关系方向构造目标扇区/半空间，在其中选中心，不做 source-size clearance；检验仅理解语言能达到多少。
- **Support+Relation Sampling**：只做显式支撑、关系和碰撞筛选，不用 learned placement score；这是最重要的 classical baseline，同时必须避免直接调用生成 GT 的完整 free-bbox 标注器，否则会形成 label-generator oracle。

这些行可以放在主表下半部分或单独的 ablation/sanity 表。它们常比再增加一个输入不匹配的机器人 policy 更能证明 TopoPlacer 的核心机制。

## 6. 统一适配与公平性协议

### 6.1 输入协议

- 所有 3D 模型读取同一 canonical world，满足 world-Z 为支撑面法向/重力上方向。
- 所有模型看到相同的场景范围、点数/体素预算和 RGB 信息。
- 严格表禁止 GT source/anchor proposal；任何 predicted proposal 的模型名和表注都必须披露。
- 公共 source grounder 直接读取完整移动指令；不得利用结构化标注字段在测试时生成更简单的 source-only expression。source-only expression 结果只能作为 oracle-text 分析。
- 从 source box 裁出的点云只能包含当前 RGB-D 中实际观测到的 source points，不能用 CAD、Objaverse asset 或完成后的 mesh 替代。
- source 原位置体素在模型输入和 `Supported and Stable`/`Collision-Free` 评测中保留，与当前 benchmark 定义一致。

### 6.2 输出协议

统一输出为：

```text
source_box = (cx, cy, cz, dx, dy, dz, yaw)
placements[k] = (bottom_x, bottom_y, bottom_z, yaw, score)
```

评估时 placement dimensions 使用项目统一定义：物理成功使用“预测 placement center + GT 长宽高 + 预测 yaw”构造评估框；Source IoU 与 Placement Size IoU 独立报告。对于不会预测 yaw 的 point 模型，严格表采用预先固定的 `copy predicted source yaw`，不能事后为每个候选搜索能通过 benchmark 的 yaw。不会预测 Top-K 的模型只评估 K=1，其 `Placement Success@5` 不应复制成看似独立的结果，表中可以写 `same as @1 (single output)`。

### 6.3 训练协议

- 所有可训练基线使用完全相同的 train/valid/test split；checkpoint 只由 validation 指标选择。
- 允许使用各自公开预训练权重，但必须列出预训练数据与可训练参数量；监督式主基线必须在 RePlace3D train split 上适配。
- 对 7B/8B MLLM 可使用 LoRA，但 PlaceWizard/M2T2/TopoPlacer 等模型应报告 full fine-tuning 或被冻结模块，不能只写“fine-tuned”。
- Top-K 候选数量统一；自回归模型可用受控多次采样产生 K 个候选，但采样次数、temperature 与去重规则在 validation 前冻结。
- 训练和推理均不得读取 valid yaw set、direction-filtered heatmap 或 free-bbox placement JSON 中测试样本的 GT 字段。

### 6.4 指标协议

主指标严格沿用项目最终定义：

- Placement Success@1、@5；
- Language Relation Correct；
- Supported and Stable；
- Collision-Free；
- Source IoU 与 IoU≥0.5 accuracy；
- Placement Size IoU；
- Center Match Rate；
- Yaw Valid Given Center Match。

建议补充两个诊断量：`Valid candidate count before Top-K truncation` 和 `Top-K spatial diversity`。它们只用于解释多解覆盖，不替代 Placement Success。

## 7. 实施优先级与止损线

### 最小充分版本

若实验资源紧张，优先完成：

1. Dense Voxel Heatmap + Yaw（任务内最近结构基线）；
2. PlaceWizard-Observed（最近外部任务）；
3. LLaVA-3D-7B-SFT（现代 3D MLLM）；
4. AnyPlace-Observed（几何/diffusion object pose）；
5. 已有 RoboBrain2.5-LoRA（2D point 路线，严格区分 predicted/oracle geometry）；
6. Support+Relation Sampling（classical sanity baseline）。

这一组已经覆盖 dense discriminative、3D LLM、diffusion object pose、2D VLM 和 explicit geometry 五种范式。若只能做一个外部模型，**先做 PlaceWizard**；若只能做两个，加入 **LLaVA-3D**；第三个优先 **AnyPlace**。

### 资源充足版本

在最小版上追加 M2T2，以形成 contact-mask target-gripper pose 与 AnyPlace diffusion object-pose 的对照；再把 RPDiff 放入 GT-source placement-only 表。GOPLA 的代码仓库恢复且提供权重后，可替换 M2T2 或作为额外 hierarchical diffusion baseline。

### 不建议在当前主表优先投入

- FirePlace：高度相关但缺少完整公开实现，复现会变成自研神经-符号系统。
- GOPLA：高度相关，但当前代码入口不可用；先联系作者，不以它阻塞论文。
- Reason3D：已被 PlaceWizard 的 placement adaptation 覆盖。
- PerAct、PolarNet、3D Diffuser Actor 等完整 action policy：它们需要机器人状态、轨迹或 demonstrations，当前 benchmark 只有 endpoint placement supervision；强行适配主要比较数据接口而非 placement reasoning。
- StructFormer/StructDiffusion：主要生成多物体结构或 sequence rearrangement，与单 source 的局部关系和支撑评测不够对齐。
- 纯 3D visual grounding 模型：只应作为公共 source grounder 或 Stage 1 对照，不能冒充完整 placement baseline。

## 8. 风险与审稿视角

第一类风险是 **oracle leakage**。PlaceWizard/AnyPlace/M2T2/RPDiff 默认获得独立 object asset 或 point cloud；若主表不给醒目标注，审稿人会认为 Ours 承担 source grounding，而 baseline 没有。第二类风险是 **输出能力不对称**：point-only VLM 没有 yaw 和 size，如果使用 GT geometry 补齐，Placement Success 不能被解释为端到端能力。第三类风险是 **benchmark generator leakage**：若 rule-based baseline 直接复用 free-bbox GT 枚举器和测试关系标注，它测到的是标注器重放而不是模型推理。第四类风险是 **视图预算不一致**：LLaVA-3D 原生偏好多视图，当前数据只有单帧时不能额外生成只供它使用的 privileged views。

反过来，只比较 RoboBrain2.5 这样的 2D point baseline 也不充分：它会让贡献看起来只是“3D 模型比 2D VLM 更物理”，无法证明 PABR、source-size conditioning 和多解 query set 优于同类 3D placement 模型。PlaceWizard 与至少一个几何 pose 模型因此不可缺少。

## 9. 开放问题

现有公开工作中仍缺少同时满足四个条件的对照：source 来自同一 partial RGB-D scene、输入是自然语言移动指令、输出多个 4-DoF source-sized boxes、并在测试时不调用昂贵碰撞搜索。这个空白正是 TopoPlacer 的合理定位，但“first”表述仍需谨慎，因为 PlaceIt3D 已建立语言 3D placement，ReRef-3D 已比较通用 3D MLLM，GOPLA/FirePlace 也已联合语义和物理。更稳妥的创新边界是：**observed-source-conditioned、end-to-end grounding-to-placement、multi-solution 4-DoF box prediction with explicit support/clearance topology**。

另一个真实空白是 source uncertainty 如何传递到 placement。当前模块化基线通常把 predicted source box 当作确定输入；而 source size/yaw 偏差会系统性改变 support 和 collision。建议在分析表中按 Source IoU 分桶报告 Placement Success，并同时给出 GT-source 表，从而区分 placement head 失败和 upstream grounding 失败。

## 10. 结论

**RQ1：** PlaceWizard、AnyPlace 和 LLaVA-3D 是最能合理适配完整 object placement 输出的公开模型；其中只有 LLaVA-3D 容易被训练为联合输出 source 与 placement，其余方法都需要预测式 source grounder。M2T2 原生输出 target gripper pose，只有在 object-to-gripper transform 定义清楚时才适合加入。RoboBrain2.5 是重要的 point-only 对照，但不能用 GT geometry 伪装成严格端到端模型。

**RQ2：** 最公平的设计是严格端到端表加 GT-source placement-only 表。前者让所有模块化 placement 模型共享公开预测式 source grounder，后者消除 source grounding 误差，直接比较位置、yaw、多解和物理能力。

**RQ3：** 最小而充分的主阵容是 `PlaceWizard + LLaVA-3D + AnyPlace + RoboBrain2.5 + Dense Heatmap + Support/Relation Sampling`；M2T2 是首选追加的机器人 primitive，RPDiff 放在 GT-source 表。FirePlace 和 GOPLA 当前不应阻塞主实验。

## References

[1] Ahmed Abdelreheem et al., “PlaceIt3D: Language-Guided Object Placement in Real 3D Scenes,” ICCV, 2025.

[2] Ian Huang et al., “FirePlace: Geometric Refinements of LLM Common Sense Reasoning for 3D Object Placement,” CVPR, 2025.

[3] Yao Zhong et al., “GOPLA: Generalizable Object Placement Learning via Synthetic Augmentation of Human Arrangement,” IROS, 2026.

[4] Mary Lynn Martin et al., “ReRef-3D: A Benchmark for Spatial Referring Expression-Guided 3D Scene Rearrangement,” arXiv:2608.16011, 2026.

[5] Yuchi Zhao et al., “AnyPlace: Learning Generalized Object Placement for Robot Manipulation,” CoRL, 2025.

[6] Wentao Yuan et al., “M2T2: Multi-Task Masked Transformer for Object-centric Pick and Place,” CoRL, 2023.

[7] Anthony Simeonov et al., “Shelving, Stacking, Hanging: Relational Pose Diffusion for Multi-modal Rearrangement,” CoRL, 2023.

[8] Chris Paxton et al., “Predicting Stable Configurations for Semantic Placement of Novel Objects,” CoRL, 2021.

[9] Chenming Zhu et al., “LLaVA-3D: A Simple yet Effective Pathway to Empowering LMMs with 3D Capabilities,” ICCV, 2025.

[10] Wentao Yuan et al., “RoboPoint: A Vision-Language Model for Spatial Affordance Prediction for Robotics,” CoRL, 2024.

[11] BAAI RoboBrain Team et al., “RoboBrain 2.5: Depth in Sight, Time in Mind,” arXiv:2601.14352, 2026.

[12] Kuan-Chih Huang et al., “Reason3D: Searching and Reasoning 3D Segmentation via Large Language Model,” 3DV, 2025.

[13] Junpeng Shang et al., “PV-Ground: Text-Guided Point-Voxel Interaction for 3D Visual Grounding,” CVPR, 2026.

[14] Hanxue Zhang et al., “Detect Anything 3D in the Wild,” ICCV, 2025.

# ICLR 2024-2026：语言条件 3D 物体放置相关论文综述

> 更新时间：2026-08-02。范围仅含 ICLR 2024、2025、2026 主会正式录用论文；录用状态、题目和作者均以 ICLR 官方详情页为准。

## 1. 检索口径

本报告以当前项目的实际任务链为中心：`RGB-D/point cloud + language -> source grounding -> placement region/query -> 3D box set + yaw -> support/collision validation`。纳入标准是论文至少直接覆盖以下一项：

1. 语言条件 3D 物体定位、空间关系、affordance 或场景安排；
2. 物体放置、相对位姿、支撑/接触/碰撞或物理可执行性；
3. 可直接替换当前 source grounding、3D backbone、候选生成、pose decoding 的方法；
4. 与上述任务紧密相关的 VLA、机器人世界模型、具身规划或评测基准。

排除纯 3D 内容生成、自动驾驶、导航、医学点云、分子几何、人体生成和不含场景几何/语言条件的纯低层控制论文。这里的“所有”指在该明确口径下，对三届 ICLR 官方完整论文目录逐项筛选所得的全集，而不是所有标题里出现 `3D` 或 `robot` 的论文。

## 2. 数量与优先级

共收录 **109 篇**：ICLR 2024 **22 篇**、ICLR 2025 **35 篇**、ICLR 2026 **52 篇**。

- **P0（必读）23 篇**：任务或方法与当前项目直接重合。
- **P1（强相关）58 篇**：能对应到一个明确模块或评测问题。
- **P2（扩展相关）28 篇**：用于 related work、基线选择或后续扩展。

方向分布：A. 放置、affordance 与物理约束 **26** 篇；B. 3D 语言 grounding 与空间推理 **37** 篇；C. 3D 感知、对应与位姿 **18** 篇；D. 机器人操作、VLA 与世界模型 **24** 篇；E. 具身规划、数据集与评测 **4** 篇。

## 3. 最值得先读的论文

1. **[Deep SE(3)-Equivariant Geometric Reasoning for Precise Placement Tasks](https://iclr.cc/virtual/2024/poster/19539)**（2024，P0，[OpenReview](https://openreview.net/forum?id=2inBuwTyL2)）：SE(3) 等变精确放置，是最直接的几何放置基线；重点比较等变表示与您当前 yaw-bin + box-surface sampling。
2. **[CoT3DRef: Chain-of-Thoughts Data-Efficient 3D Visual Grounding](https://iclr.cc/virtual/2024/poster/18742)**（2024，P0，[OpenReview](https://openreview.net/forum?id=ORUiqcLpV6)）：把链式推理引入 3D visual grounding，可对照 Stage 1 的单查询 source grounding 与文本推理监督。
3. **[3D-Aware Hypothesis & Verification for Generalizable Relative Object Pose Estimation](https://iclr.cc/virtual/2024/poster/18534)**（2024，P0，[OpenReview](https://openreview.net/forum?id=U6hEOZlDf5)）：研究相对物体位姿而非绝对框，适合补充“相对目标物/支撑物”的放置关系建模。
4. **[3D-AffordanceLLM: Harnessing Large Language Models for Open-Vocabulary Affordance Detection in 3D Worlds](https://iclr.cc/virtual/2025/poster/30275)**（2025，P0，[OpenReview](https://openreview.net/forum?id=GThTiuXgDC)）：开放词汇 3D affordance 分割，与支撑区域/可放置区域预测最接近；应作为核心 related work。
5. **[GravMAD: Grounded Spatial Value Maps Guided Action Diffusion for Generalized 3D Manipulation](https://iclr.cc/virtual/2025/poster/28249)**（2025，P0，[OpenReview](https://openreview.net/forum?id=qPzYF2EpXb)）：用 grounded spatial value map 驱动 3D manipulation，与 Stage 2 粗区域概率图和候选生成高度同构。
6. **[SPARTUN3D: Situated Spatial Understanding of 3D World in Large Language Model](https://iclr.cc/virtual/2025/poster/30351)**（2025，P0，[OpenReview](https://openreview.net/forum?id=FGMkSL8NR0)）：LLM 的 situated 3D spatial understanding，可用于论证显式 3D 空间表征相对纯 2D VLM 的必要性。
7. **[3D-SPATIAL MULTIMODAL MEMORY](https://iclr.cc/virtual/2025/poster/29300)**（2025，P0，[OpenReview](https://openreview.net/forum?id=XYdstv3ySl)）：3D spatial multimodal memory 关注跨视角空间记忆，对多帧 canonical scene 扩展最有参考价值。
8. **[SPA: 3D Spatial-Awareness Enables Effective Embodied Representation](https://iclr.cc/virtual/2025/poster/30883)**（2025，P0，[OpenReview](https://openreview.net/forum?id=6TLdqAZgzn)）：3D spatial awareness 的具身表征，可作为文本-体素融合和空间先验设计的直接对照。
9. **[DenseGrounding: Improving Dense Language-Vision Semantics for Ego-centric 3D Visual Grounding](https://iclr.cc/virtual/2025/poster/28704)**（2025，P0，[OpenReview](https://openreview.net/forum?id=iGafR0hSln)）：面向自我中心场景的稠密 3D 语言 grounding，适合比较语言特征 2D→3D splat 与稠密语义监督。
10. **[DenseMatcher: Learning 3D Semantic Correspondence for Category-Level Manipulation from a Single Demo](https://iclr.cc/virtual/2025/poster/30743)**（2025，P0，[OpenReview](https://openreview.net/forum?id=8oFvUBvF1u)）：单示范类别级操作中的 3D 语义对应，可借鉴到 source geometry 与候选支撑区域之间的匹配。
11. **[Do 3D Large Language Models Really Understand 3D Spatial Relationships?](https://iclr.cc/virtual/2026/poster/10011597)**（2026，P0，[OpenReview](https://openreview.net/forum?id=3vlMiJwo8b)）：直接检验 3D-LLM 是否真正理解空间关系，是定义空间关系评测和避免语言捷径的重要依据。
12. **[From Spatial to Actions: Grounding Vision-Language-Action Model in Spatial Foundation Priors](https://iclr.cc/virtual/2026/poster/10008188)**（2026，P0，[OpenReview](https://openreview.net/forum?id=fzmittHfq3)）：把 spatial foundation prior 接到 VLA 动作端，与您的“先空间候选、后可执行放置”两阶段逻辑最接近。
13. **[Generalizable Coarse-to-Fine Robot Manipulation via Language-Aligned 3D Keypoints](https://iclr.cc/virtual/2026/poster/10009046)**（2026，P0，[OpenReview](https://openreview.net/forum?id=WXFfMLyB6y)）：语言对齐的 3D keypoints + coarse-to-fine manipulation，可对照您的 top-k region 到 anchor query 展开。
14. **[PhyScensis: Physics-Augmented LLM Agents for Complex Physical Scene Arrangement](https://iclr.cc/virtual/2026/poster/10008728)**（2026，P0，[OpenReview](https://openreview.net/forum?id=aCVfhY4Qen)）：LLM agent 处理复杂物理场景安排，直接覆盖语言、物理约束和 scene arrangement。
15. **[H2OFlow: Grounding Human-Object Affordances with 3D Generative Models and Dense Diffused Flows](https://iclr.cc/virtual/2026/poster/10009565)**（2026，P0，[OpenReview](https://openreview.net/forum?id=QhqJ1DCp1X)）：以 3D 生成模型和稠密 flow grounding 人-物 affordance，可启发连续可供性场和多解集合预测。
16. **[Unified 3D Scene Understanding Through Physical World Modeling](https://iclr.cc/virtual/2026/poster/10009864)**（2026，P0，[OpenReview](https://openreview.net/forum?id=NQq9JLMfNN)）：通过 physical world modeling 统一 3D scene understanding，可作为物理约束进入表征层的最新对照。
17. **[ComGS: Efficient 3D Object-Scene Composition via Surface Octahedral Probes](https://iclr.cc/virtual/2026/poster/10006547)**（2026，P0，[OpenReview](https://openreview.net/forum?id=yXiSPBMrTT)）：以 surface probes 做 3D object-scene composition，和您的 box-surface 多尺度采样在机制上值得重点比较。
18. **[Pose-RFT: Aligning MLLMs for 3D Pose Generation via Hybrid Action Reinforcement Fine-Tuning](https://iclr.cc/virtual/2026/poster/10008305)**（2026，P0，[OpenReview](https://openreview.net/forum?id=ea1U1MgbdT)）：用强化微调让 MLLM 生成 3D pose，可作为离散 yaw 与连续 pose 生成路线的对照。

## 4. 对当前工作的总体判断

1. **最接近但尚未完全重合的路线**：现有论文通常分别解决 3D grounding、affordance mask、单一操作 pose、scene arrangement 或 VLA action；当前项目同时要求 source grounding、语言关系、物体原尺寸、多合法放置集合、yaw、支撑与无碰撞，因此仍有清晰的组合型研究空缺。
2. **最需要正面对比的模块**：`GravMAD` 的 grounded spatial value map、`3D-AffordanceLLM` 的开放词汇 affordance、`Deep SE(3)-Equivariant Geometric Reasoning` 的等变放置、`ComGS` 的 surface probes、`Generalizable Coarse-to-Fine...3D Keypoints` 的 coarse-to-fine query。
3. **论文表述上的风险**：不能只以“首次语言条件 3D 放置”概括创新。2026 年已有 spatial-to-action、physical scene arrangement 和 3D pose generation；更稳妥的创新边界应落在“真实 RGB-D 场景中、source-size 保持、可变数量多解 3D box set、显式支撑/碰撞约束、统一 grounding-to-placement”。
4. **评测建议**：除当前 task success、IoU、方向关系和 collision 外，应增加空间关系细分类、候选覆盖率、等价多解召回、source grounding 与 placement 解耦诊断，并参考 2026 空间推理基准检查语言捷径。

## 5. 完整论文清单

作者栏超过 3 人时仅展示前三位；点击 ICLR 可查看完整作者和摘要。OpenReview 链接用于论文 PDF、补充材料和评审记录。

### ICLR 2024

#### A. 放置、affordance 与物理约束

| 级别 | 论文 | 作者 | 与当前项目的关系 |
|---|---|---|---|
| P0 | **3D-Aware Hypothesis & Verification for Generalizable Relative Object Pose Estimation**<br>[ICLR](https://iclr.cc/virtual/2024/poster/18534) / [OpenReview](https://openreview.net/forum?id=U6hEOZlDf5) | Chen Zhao, Tong Zhang, Mathieu Salzmann | Stage 2 候选区域、位姿集合、支撑/碰撞与物理有效性 |
| P0 | **Deep SE(3)-Equivariant Geometric Reasoning for Precise Placement Tasks**<br>[ICLR](https://iclr.cc/virtual/2024/poster/19539) / [OpenReview](https://openreview.net/forum?id=2inBuwTyL2) | Ben Eisner, Yi Yang, Todor Davchev 等 | Stage 2 候选区域、位姿集合、支撑/碰撞与物理有效性 |
| P0 | **InstructScene: Instruction-Driven 3D Indoor Scene Synthesis with Semantic Graph Prior**<br>[ICLR](https://iclr.cc/virtual/2024/poster/18845) / [OpenReview](https://openreview.net/forum?id=LtuRgL03pI) | Chenguo Lin, Yadong MU | Stage 2 候选区域、位姿集合、支撑/碰撞与物理有效性 |
| P1 | **CORN: Contact-based Object Representation for Nonprehensile Manipulation of General Unseen Objects**<br>[ICLR](https://iclr.cc/virtual/2024/poster/18901) / [OpenReview](https://openreview.net/forum?id=KTtEICH4TO) | Yoonyoung Cho, Junhyek Han, Yoontae Cho 等 | Stage 2 候选区域、位姿集合、支撑/碰撞与物理有效性 |
| P1 | **DIFFTACTILE: A Physics-based Differentiable Tactile Simulator for Contact-rich Robotic Manipulation**<br>[ICLR](https://iclr.cc/virtual/2024/poster/18226) / [OpenReview](https://openreview.net/forum?id=eJHnSg783t) | Zilin Si, Gu Zhang, Qingwei Ben 等 | Stage 2 候选区域、位姿集合、支撑/碰撞与物理有效性 |
| P1 | **Fourier Transporter: Bi-Equivariant Robotic Manipulation in 3D**<br>[ICLR](https://iclr.cc/virtual/2024/poster/18515) / [OpenReview](https://openreview.net/forum?id=UulwvAU1W0) | Haojie Huang, Owen Howell, Dian Wang 等 | Stage 2 候选区域、位姿集合、支撑/碰撞与物理有效性 |
| P1 | **I-PHYRE: Interactive Physical Reasoning**<br>[ICLR](https://iclr.cc/virtual/2024/poster/19580) / [OpenReview](https://openreview.net/forum?id=1bbPQShCT2) | Shiqian Li, Kewen Wu, Chi Zhang 等 | Stage 2 候选区域、位姿集合、支撑/碰撞与物理有效性 |
| P1 | **Task Planning for Visual Room Rearrangement under Partial Observability**<br>[ICLR](https://iclr.cc/virtual/2024/poster/18030) / [OpenReview](https://openreview.net/forum?id=jJvXNpvOdM) | Karan Mirakhor, Sourav Ghosh, DIPANJAN DAS 等 | Stage 2 候选区域、位姿集合、支撑/碰撞与物理有效性 |

#### B. 3D 语言 grounding 与空间推理

| 级别 | 论文 | 作者 | 与当前项目的关系 |
|---|---|---|---|
| P0 | **CoT3DRef: Chain-of-Thoughts Data-Efficient 3D Visual Grounding**<br>[ICLR](https://iclr.cc/virtual/2024/poster/18742) / [OpenReview](https://openreview.net/forum?id=ORUiqcLpV6) | Eslam Abdelrahman, Mohamed Ayman Mohamed, Mahmoud Ahmed 等 | Stage 1 source grounding、文本-体素融合与空间关系监督 |
| P1 | **Grounding Multimodal Large Language Models to the World**<br>[ICLR](https://iclr.cc/virtual/2024/poster/17934) / [OpenReview](https://openreview.net/forum?id=lLmqxkfSIw) | Zhiliang Peng, Wenhui Wang, Li Dong 等 | Stage 1 source grounding、文本-体素融合与空间关系监督 |
| P1 | **Spatially-Aware Transformers for Embodied Agents**<br>[ICLR](https://iclr.cc/virtual/2024/poster/18546) / [OpenReview](https://openreview.net/forum?id=Ts95eXsPBc) | Junmo Cho, Jaesik Yoon, Sungjin Ahn | Stage 1 source grounding、文本-体素融合与空间关系监督 |

#### C. 3D 感知、对应与位姿

| 级别 | 论文 | 作者 | 与当前项目的关系 |
|---|---|---|---|
| P1 | **3D Feature Prediction for Masked-AutoEncoder-Based Point Cloud Pretraining**<br>[ICLR](https://iclr.cc/virtual/2024/poster/18849) / [OpenReview](https://openreview.net/forum?id=LokR2TTFMs) | Siming Yan, Yuqi Yang, Yu-Xiao Guo 等 | 稀疏 3D backbone、source box/pose、分割及几何对应 |
| P1 | **AGILE3D: Attention Guided Interactive Multi-object 3D Segmentation**<br>[ICLR](https://iclr.cc/virtual/2024/poster/19289) / [OpenReview](https://openreview.net/forum?id=9cQtXpRshE) | Yuanwen Yue, Sabarinath Mahadevan, Jonas Schult 等 | 稀疏 3D backbone、source box/pose、分割及几何对应 |
| P1 | **FreeReg: Image-to-Point Cloud Registration Leveraging Pretrained Diffusion Models and Monocular Depth Estimators**<br>[ICLR](https://iclr.cc/virtual/2024/poster/19217) / [OpenReview](https://openreview.net/forum?id=BPb5AhT2Vf) | Haiping Wang, Yuan Liu, Bing WANG 等 | 稀疏 3D backbone、source box/pose、分割及几何对应 |
| P1 | **Learning to Act from Actionless Videos through Dense Correspondences**<br>[ICLR](https://iclr.cc/virtual/2024/poster/18812) / [OpenReview](https://openreview.net/forum?id=Mhb5fpA1T0) | Po-Chen Ko, Jiayuan Mao, Yilun Du 等 | 稀疏 3D backbone、source box/pose、分割及几何对应 |
| P1 | **OpenNeRF: Open Set 3D Neural Scene Segmentation with Pixel-Wise Features and Rendered Novel Views**<br>[ICLR](https://iclr.cc/virtual/2024/poster/18594) / [OpenReview](https://openreview.net/forum?id=SgjAojPKb3) | Francis Engelmann, Fabian Manhardt, Michael Niemeyer 等 | 稀疏 3D backbone、source box/pose、分割及几何对应 |
| P1 | **V-DETR: DETR with Vertex Relative Position Encoding for 3D Object Detection**<br>[ICLR](https://iclr.cc/virtual/2024/poster/18795) / [OpenReview](https://openreview.net/forum?id=NDkpxG94sF) | Yichao Shen, Zigang Geng, YUHUI YUAN 等 | 稀疏 3D backbone、source box/pose、分割及几何对应 |

#### D. 机器人操作、VLA 与世界模型

| 级别 | 论文 | 作者 | 与当前项目的关系 |
|---|---|---|---|
| P2 | **Entity-Centric Reinforcement Learning for Object Manipulation from Pixels**<br>[ICLR](https://iclr.cc/virtual/2024/poster/17585) / [OpenReview](https://openreview.net/forum?id=uDxeSZ1wdI) | Dan Haramati, Tal Daniel, Aviv Tamar | 整体具身操作路线、VLA/世界模型基线与可执行策略接口 |
| P2 | **Habitat 3.0: A Co-Habitat for Humans, Avatars, and Robots**<br>[ICLR](https://iclr.cc/virtual/2024/poster/19442) / [OpenReview](https://openreview.net/forum?id=4znwzG92CE) | Xavier Puig, Eric Undersander, Andrew Szot 等 | 整体具身操作路线、VLA/世界模型基线与可执行策略接口 |
| P2 | **SparseDFF: Sparse-View Feature Distillation for One-Shot Dexterous Manipulation**<br>[ICLR](https://iclr.cc/virtual/2024/poster/19000) / [OpenReview](https://openreview.net/forum?id=HHWlwxDeRn) | Qianxu Wang, Haotong Zhang, Congyue Deng 等 | 整体具身操作路线、VLA/世界模型基线与可执行策略接口 |
| P2 | **Vision-Language Foundation Models as Effective Robot Imitators**<br>[ICLR](https://iclr.cc/virtual/2024/poster/17943) / [OpenReview](https://openreview.net/forum?id=lFYj0oibGR) | Xinghang Li, Minghuan Liu, Hanbo Zhang 等 | 整体具身操作路线、VLA/世界模型基线与可执行策略接口 |
| P2 | **Zero-Shot Robotic Manipulation with Pre-Trained Image-Editing Diffusion Models**<br>[ICLR](https://iclr.cc/virtual/2024/poster/18313) / [OpenReview](https://openreview.net/forum?id=c0chJTSbci) | Kevin Black, Mitsuhiko Nakamoto, Pranav Atreya 等 | 整体具身操作路线、VLA/世界模型基线与可执行策略接口 |

### ICLR 2025

#### A. 放置、affordance 与物理约束

| 级别 | 论文 | 作者 | 与当前项目的关系 |
|---|---|---|---|
| P0 | **3D-AffordanceLLM: Harnessing Large Language Models for Open-Vocabulary Affordance Detection in 3D Worlds**<br>[ICLR](https://iclr.cc/virtual/2025/poster/30275) / [OpenReview](https://openreview.net/forum?id=GThTiuXgDC) | Hengshuo Chu, Xiang Deng, Qi Lv 等 | Stage 2 候选区域、位姿集合、支撑/碰撞与物理有效性 |
| P0 | **6D Object Pose Tracking in Internet Videos for Robotic Manipulation**<br>[ICLR](https://iclr.cc/virtual/2025/poster/31220) / [OpenReview](https://openreview.net/forum?id=1CIUkpoata) | Georgy Ponimatkin, Martin Cífka, Tomas Soucek 等 | Stage 2 候选区域、位姿集合、支撑/碰撞与物理有效性 |
| P0 | **DenseMatcher: Learning 3D Semantic Correspondence for Category-Level Manipulation from a Single Demo**<br>[ICLR](https://iclr.cc/virtual/2025/poster/30743) / [OpenReview](https://openreview.net/forum?id=8oFvUBvF1u) | Junzhe Zhu, Yuanchen Ju, Junyi Zhang 等 | Stage 2 候选区域、位姿集合、支撑/碰撞与物理有效性 |
| P0 | **GravMAD: Grounded Spatial Value Maps Guided Action Diffusion for Generalized 3D Manipulation**<br>[ICLR](https://iclr.cc/virtual/2025/poster/28249) / [OpenReview](https://openreview.net/forum?id=qPzYF2EpXb) | Yangtao Chen, Zixuan Chen, Junhui Yin 等 | Stage 2 候选区域、位姿集合、支撑/碰撞与物理有效性 |
| P1 | **Intent3D: 3D Object Detection in RGB-D Scans Based on Human Intention**<br>[ICLR](https://iclr.cc/virtual/2025/poster/30957) / [OpenReview](https://openreview.net/forum?id=5GgjiRzYp3) | Weitai Kang, Mengxue Qu, Jyoti Kini 等 | Stage 2 候选区域、位姿集合、支撑/碰撞与物理有效性 |
| P1 | **Learning Geometric Reasoning Networks For Robot Task And Motion Planning**<br>[ICLR](https://iclr.cc/virtual/2025/poster/29152) / [OpenReview](https://openreview.net/forum?id=ajxAJ8GUX4) | Smail Ait Bouhsain, Rachid Alami, Thierry Simeon | Stage 2 候选区域、位姿集合、支撑/碰撞与物理有效性 |
| P1 | **ManiSkill-HAB: A Benchmark for Low-Level Manipulation in Home Rearrangement Tasks**<br>[ICLR](https://iclr.cc/virtual/2025/poster/30874) / [OpenReview](https://openreview.net/forum?id=6bKEWevgSd) | Arth Shukla, Stone Tao, Hao Su | Stage 2 候选区域、位姿集合、支撑/碰撞与物理有效性 |
| P1 | **Weakly-Supervised Affordance Grounding Guided by Part-Level Semantic Priors**<br>[ICLR](https://iclr.cc/virtual/2025/poster/31269) / [OpenReview](https://openreview.net/forum?id=0823rvTIhs) | Peiran Xu, Yadong MU | Stage 2 候选区域、位姿集合、支撑/碰撞与物理有效性 |

#### B. 3D 语言 grounding 与空间推理

| 级别 | 论文 | 作者 | 与当前项目的关系 |
|---|---|---|---|
| P0 | **3D-SPATIAL MULTIMODAL MEMORY**<br>[ICLR](https://iclr.cc/virtual/2025/poster/29300) / [OpenReview](https://openreview.net/forum?id=XYdstv3ySl) | Xueyan Zou, Yuchen Song, Ri-Zhao Qiu 等 | Stage 1 source grounding、文本-体素融合与空间关系监督 |
| P0 | **DenseGrounding: Improving Dense Language-Vision Semantics for Ego-centric 3D Visual Grounding**<br>[ICLR](https://iclr.cc/virtual/2025/poster/28704) / [OpenReview](https://openreview.net/forum?id=iGafR0hSln) | Henry Zheng, Hao Shi, Qihang Peng 等 | Stage 1 source grounding、文本-体素融合与空间关系监督 |
| P0 | **SPA: 3D Spatial-Awareness Enables Effective Embodied Representation**<br>[ICLR](https://iclr.cc/virtual/2025/poster/30883) / [OpenReview](https://openreview.net/forum?id=6TLdqAZgzn) | Haoyi Zhu, Honghui Yang, Yating Wang 等 | Stage 1 source grounding、文本-体素融合与空间关系监督 |
| P0 | **SPARTUN3D: Situated Spatial Understanding of 3D World in Large Language Model**<br>[ICLR](https://iclr.cc/virtual/2025/poster/30351) / [OpenReview](https://openreview.net/forum?id=FGMkSL8NR0) | Yue Zhang, Zhiyang Xu, Ying Shen 等 | Stage 1 source grounding、文本-体素融合与空间关系监督 |
| P1 | **3D Vision-Language Gaussian Splatting**<br>[ICLR](https://iclr.cc/virtual/2025/poster/29604) / [OpenReview](https://openreview.net/forum?id=SSE9myD9SG) | Qucheng Peng, Benjamin Planche, Zhongpai Gao 等 | Stage 1 source grounding、文本-体素融合与空间关系监督 |
| P1 | **Duoduo CLIP: Efficient 3D Understanding with Multi-View Images**<br>[ICLR](https://iclr.cc/virtual/2025/poster/28703) / [OpenReview](https://openreview.net/forum?id=iGbuc9ekKK) | Han-Hung Lee, Yiming Zhang, Angel Chang | Stage 1 source grounding、文本-体素融合与空间关系监督 |
| P1 | **Language-Image Models with 3D Understanding**<br>[ICLR](https://iclr.cc/virtual/2025/poster/27715) / [OpenReview](https://openreview.net/forum?id=yaQbTAD2JJ) | Jang Hyun Cho, Boris Ivanovic, Yulong Cao 等 | Stage 1 source grounding、文本-体素融合与空间关系监督 |
| P1 | **Segment Any 3D Object with Language**<br>[ICLR](https://iclr.cc/virtual/2025/poster/30409) / [OpenReview](https://openreview.net/forum?id=ENv1CeTwxc) | Seungjun Lee, Yuyang Zhao, Gim H Lee | Stage 1 source grounding、文本-体素融合与空间关系监督 |
| P1 | **TraceVLA: Visual Trace Prompting Enhances Spatial-Temporal Awareness for Generalist Robotic Policies**<br>[ICLR](https://iclr.cc/virtual/2025/poster/29130) / [OpenReview](https://openreview.net/forum?id=b1CVu9l5GO) | Ruijie Zheng, Yongyuan Liang, Shuaiyi Huang 等 | Stage 1 source grounding、文本-体素融合与空间关系监督 |
| P1 | **UniGS: Unified Language-Image-3D Pretraining with Gaussian Splatting**<br>[ICLR](https://iclr.cc/virtual/2025/poster/30882) / [OpenReview](https://openreview.net/forum?id=6U2KI1dpfl) | Haoyuan Li, Yanpeng Zhou, Tao Tang 等 | Stage 1 source grounding、文本-体素融合与空间关系监督 |

#### C. 3D 感知、对应与位姿

| 级别 | 论文 | 作者 | 与当前项目的关系 |
|---|---|---|---|
| P1 | **CapeX: Category-Agnostic Pose Estimation from Textual Point Explanation**<br>[ICLR](https://iclr.cc/virtual/2025/poster/28113) / [OpenReview](https://openreview.net/forum?id=scKAXgonmq) | Matan Rusanovsky, Or Hirschorn, Shai Avidan | 稀疏 3D backbone、source box/pose、分割及几何对应 |
| P1 | **EmbodiedSAM: Online Segment Any 3D Thing in Real Time**<br>[ICLR](https://iclr.cc/virtual/2025/poster/29314) / [OpenReview](https://openreview.net/forum?id=XFYUwIyTxQ) | Xiuwei Xu, Huangxing Chen, Linqing Zhao 等 | 稀疏 3D backbone、source box/pose、分割及几何对应 |
| P1 | **GrabS: Generative Embodied Agent for 3D Object Segmentation without Scene Supervision**<br>[ICLR](https://iclr.cc/virtual/2025/poster/27837) / [OpenReview](https://openreview.net/forum?id=wXSshrxlP4) | Zihui Zhang, Yafei YANG, Hongtao Wen 等 | 稀疏 3D backbone、source box/pose、分割及几何对应 |
| P1 | **Multimodality Helps Few-shot 3D Point Cloud Semantic Segmentation**<br>[ICLR](https://iclr.cc/virtual/2025/poster/28634) / [OpenReview](https://openreview.net/forum?id=jXvwJ51vcK) | Zhaochong An, Guolei Sun, Yun Liu 等 | 稀疏 3D backbone、source box/pose、分割及几何对应 |
| P1 | **Multiview Equivariance Improves 3D Correspondence Understanding with Minimal Feature Finetuning**<br>[ICLR](https://iclr.cc/virtual/2025/poster/30515) / [OpenReview](https://openreview.net/forum?id=CNO4rbSV6v) | Yang You, Yixin Li, Congyue Deng 等 | 稀疏 3D backbone、source box/pose、分割及几何对应 |
| P1 | **Open-YOLO 3D: Towards Fast and Accurate Open-Vocabulary 3D Instance Segmentation**<br>[ICLR](https://iclr.cc/virtual/2025/poster/30514) / [OpenReview](https://openreview.net/forum?id=CRmiX0v16e) | Mohamed el amine Boudjoghra, Angela Dai, Jean Lahoud 等 | 稀疏 3D backbone、source box/pose、分割及几何对应 |
| P1 | **Point-SAM: Promptable 3D Segmentation Model for Point Clouds**<br>[ICLR](https://iclr.cc/virtual/2025/poster/27719) / [OpenReview](https://openreview.net/forum?id=yXCTDhZDh6) | Yuchen Zhou, Jiayuan Gu, Tung Chiang 等 | 稀疏 3D backbone、source box/pose、分割及几何对应 |
| P1 | **Uni$^2$Det: Unified and Universal Framework for Prompt-Guided Multi-dataset 3D Detection**<br>[ICLR](https://iclr.cc/virtual/2025/poster/30625) / [OpenReview](https://openreview.net/forum?id=AcVpLS86RT) | Yubin Wang, Zhikang Zou, Xiaoqing Ye 等 | 稀疏 3D backbone、source box/pose、分割及几何对应 |

#### D. 机器人操作、VLA 与世界模型

| 级别 | 论文 | 作者 | 与当前项目的关系 |
|---|---|---|---|
| P2 | **Dream to Manipulate: Compositional World Models Empowering Robot Imitation Learning with Imagination**<br>[ICLR](https://iclr.cc/virtual/2025/poster/31075) / [OpenReview](https://openreview.net/forum?id=3RSLW9YSgk) | Leonardo Barcellona, Andrii Zadaianchuk, Davide Allegro 等 | 整体具身操作路线、VLA/世界模型基线与可执行策略接口 |
| P2 | **EC-Diffuser: Multi-Object Manipulation via Entity-Centric Behavior Generation**<br>[ICLR](https://iclr.cc/virtual/2025/poster/28379) / [OpenReview](https://openreview.net/forum?id=o3pJU5QCtv) | Carl Qi, Dan Haramati, Tal Daniel 等 | 整体具身操作路线、VLA/世界模型基线与可执行策略接口 |
| P2 | **FLIP: Flow-Centric Generative Planning as General-Purpose Manipulation World Model**<br>[ICLR](https://iclr.cc/virtual/2025/poster/30597) / [OpenReview](https://openreview.net/forum?id=B2N0nCVC91) | Chongkai Gao, Haozhuo Zhang, Zhixuan Xu 等 | 整体具身操作路线、VLA/世界模型基线与可执行策略接口 |
| P2 | **HAMSTER: Hierarchical Action Models for Open-World Robot Manipulation**<br>[ICLR](https://iclr.cc/virtual/2025/poster/28776) / [OpenReview](https://openreview.net/forum?id=h7aQxzKbq6) | Yi Li, Yuquan Deng, Jesse Zhang 等 | 整体具身操作路线、VLA/世界模型基线与可执行策略接口 |
| P2 | **Learning View-invariant World Models for Visual Robotic Manipulation**<br>[ICLR](https://iclr.cc/virtual/2025/poster/27921) / [OpenReview](https://openreview.net/forum?id=vJwjWyt4Ed) | Jing-Cheng Pang, Nan Tang, Kaiyuan Li 等 | 整体具身操作路线、VLA/世界模型基线与可执行策略接口 |
| P2 | **RDT-1B: a Diffusion Foundation Model for Bimanual Manipulation**<br>[ICLR](https://iclr.cc/virtual/2025/poster/27746) / [OpenReview](https://openreview.net/forum?id=yAzN4tz7oI) | Songming Liu, Lingxuan Wu, Bangguo Li 等 | 整体具身操作路线、VLA/世界模型基线与可执行策略接口 |
| P2 | **VisualPredicator: Learning Abstract World Models with Neuro-Symbolic Predicates for Robot Planning**<br>[ICLR](https://iclr.cc/virtual/2025/poster/29691) / [OpenReview](https://openreview.net/forum?id=QOfswj7hij) | Yichao Liang, Nishanth Kumar, Hao Tang 等 | 整体具身操作路线、VLA/世界模型基线与可执行策略接口 |
| P2 | **VLAS: Vision-Language-Action Model with Speech Instructions for Customized Robot Manipulation**<br>[ICLR](https://iclr.cc/virtual/2025/poster/30076) / [OpenReview](https://openreview.net/forum?id=K4FAFNRpko) | Wei Zhao, Pengxiang Ding, Zhang Min 等 | 整体具身操作路线、VLA/世界模型基线与可执行策略接口 |

#### E. 具身规划、数据集与评测

| 级别 | 论文 | 作者 | 与当前项目的关系 |
|---|---|---|---|
| P2 | **AHA: A Vision-Language-Model for Detecting and Reasoning Over Failures in Robotic Manipulation**<br>[ICLR](https://iclr.cc/virtual/2025/poster/30106) / [OpenReview](https://openreview.net/forum?id=JVkdSi7Ekg) | Jiafei Duan, Wilbert Pumacay, Nishanth Kumar 等 | 任务定义、数据生成、规划协议与评测维度 |

### ICLR 2026

#### A. 放置、affordance 与物理约束

| 级别 | 论文 | 作者 | 与当前项目的关系 |
|---|---|---|---|
| P0 | **ComGS: Efficient 3D Object-Scene Composition via Surface Octahedral Probes**<br>[ICLR](https://iclr.cc/virtual/2026/poster/10006547) / [OpenReview](https://openreview.net/forum?id=yXiSPBMrTT) | Jian Gao, Mengqi Yuan, Yifei Zeng 等 | Stage 2 候选区域、位姿集合、支撑/碰撞与物理有效性 |
| P0 | **Generalizable Coarse-to-Fine Robot Manipulation via Language-Aligned 3D Keypoints**<br>[ICLR](https://iclr.cc/virtual/2026/poster/10009046) / [OpenReview](https://openreview.net/forum?id=WXFfMLyB6y) | Jianshu Hu, Lidi Wang, Shujia Li 等 | Stage 2 候选区域、位姿集合、支撑/碰撞与物理有效性 |
| P0 | **H2OFlow: Grounding Human-Object Affordances with 3D Generative Models and Dense Diffused Flows**<br>[ICLR](https://iclr.cc/virtual/2026/poster/10009565) / [OpenReview](https://openreview.net/forum?id=QhqJ1DCp1X) | Harry Zhang, Luca Carlone | Stage 2 候选区域、位姿集合、支撑/碰撞与物理有效性 |
| P0 | **PhyScensis: Physics-Augmented LLM Agents for Complex Physical Scene Arrangement**<br>[ICLR](https://iclr.cc/virtual/2026/poster/10008728) / [OpenReview](https://openreview.net/forum?id=aCVfhY4Qen) | Yian Wang, Han Yang, Minghao Guo 等 | Stage 2 候选区域、位姿集合、支撑/碰撞与物理有效性 |
| P0 | **Pose-RFT: Aligning MLLMs for 3D Pose Generation via Hybrid Action Reinforcement Fine-Tuning**<br>[ICLR](https://iclr.cc/virtual/2026/poster/10008305) / [OpenReview](https://openreview.net/forum?id=ea1U1MgbdT) | Bao Li, Xiaomei Zhang, Miao Xu 等 | Stage 2 候选区域、位姿集合、支撑/碰撞与物理有效性 |
| P0 | **Unified 3D Scene Understanding Through Physical World Modeling**<br>[ICLR](https://iclr.cc/virtual/2026/poster/10009864) / [OpenReview](https://openreview.net/forum?id=NQq9JLMfNN) | Wanhee Lee, Klemen Kotar, Rahul Venkatesh 等 | Stage 2 候选区域、位姿集合、支撑/碰撞与物理有效性 |
| P1 | **Interpretable 3D Neural Object Volumes for Robust Conceptual Reasoning**<br>[ICLR](https://iclr.cc/virtual/2026/poster/10009138) / [OpenReview](https://openreview.net/forum?id=VSPLa2Sito) | Nhi Pham, Artur Jesslen, Bernt Schiele 等 | Stage 2 候选区域、位姿集合、支撑/碰撞与物理有效性 |
| P1 | **Manipulation as in Simulation: Enabling Accurate Geometry Perception in Robots**<br>[ICLR](https://iclr.cc/virtual/2026/poster/10007046) / [OpenReview](https://openreview.net/forum?id=sWyX1BpeN4) | Minghuan Liu, Zhengbang Zhu, Xiaoshen Han 等 | Stage 2 候选区域、位姿集合、支撑/碰撞与物理有效性 |
| P1 | **PA3FF:Learning Part-Aware Dense 3D Feature Field For Generalizable Articulated Object Manipulation**<br>[ICLR](https://iclr.cc/virtual/2026/poster/10007233) / [OpenReview](https://openreview.net/forum?id=qXfRXfAHOK) | Yue Chen, Muqing Jiang, Kaifeng Zheng 等 | Stage 2 候选区域、位姿集合、支撑/碰撞与物理有效性 |
| P1 | **PAT3D: Physics-Augmented Text-to-3D Scene Generation**<br>[ICLR](https://iclr.cc/virtual/2026/poster/10007978) / [OpenReview](https://openreview.net/forum?id=iIRxFkeCuY) | Guying Lin, Kemeng Huang, Michael Liu 等 | Stage 2 候选区域、位姿集合、支撑/碰撞与物理有效性 |

#### B. 3D 语言 grounding 与空间推理

| 级别 | 论文 | 作者 | 与当前项目的关系 |
|---|---|---|---|
| P0 | **Do 3D Large Language Models Really Understand 3D Spatial Relationships?**<br>[ICLR](https://iclr.cc/virtual/2026/poster/10011597) / [OpenReview](https://openreview.net/forum?id=3vlMiJwo8b) | Xianzheng Ma, Tao Sun, Shuai Chen 等 | Stage 1 source grounding、文本-体素融合与空间关系监督 |
| P0 | **From Spatial to Actions: Grounding Vision-Language-Action Model in Spatial Foundation Priors**<br>[ICLR](https://iclr.cc/virtual/2026/poster/10008188) / [OpenReview](https://openreview.net/forum?id=fzmittHfq3) | Zhengshen Zhang, 昊 李, Yalun Dai 等 | Stage 1 source grounding、文本-体素融合与空间关系监督 |
| P0 | **OmniEVA: Embodied Versatile Planner via Task-Adaptive 3D-Grounded and Embodiment-aware Reasoning**<br>[ICLR](https://iclr.cc/virtual/2026/poster/10006954) / [OpenReview](https://openreview.net/forum?id=tkEmIJv1tB) | Yuecheng Liu, DaFeng Chi, Shiguang Wu 等 | Stage 1 source grounding、文本-体素融合与空间关系监督 |
| P0 | **pySpatial: Generating 3D Visual Programs for Zero-Shot Spatial Reasoning**<br>[ICLR](https://iclr.cc/virtual/2026/poster/10006507) / [OpenReview](https://openreview.net/forum?id=yv15C8ql24) | Zhanpeng Luo, Ce Zhang, Silong Yong 等 | Stage 1 source grounding、文本-体素融合与空间关系监督 |
| P0 | **SceneCOT: Eliciting Grounded Chain-of-Thought Reasoning in 3D Scenes**<br>[ICLR](https://iclr.cc/virtual/2026/poster/10009257) / [OpenReview](https://openreview.net/forum?id=U9meoc0Sau) | Xiongkun Linghu, Jiangyong Huang, Ziyu Zhu 等 | Stage 1 source grounding、文本-体素融合与空间关系监督 |
| P1 | **3D Aware Region Prompted Vision Language Model**<br>[ICLR](https://iclr.cc/virtual/2026/poster/10010486) / [OpenReview](https://openreview.net/forum?id=GTpf2NuwtR) | An-Chieh Cheng, Yang Fu, Yukang Chen 等 | Stage 1 source grounding、文本-体素融合与空间关系监督 |
| P1 | **GPT4Scene: Understand 3D Scenes from Videos with Vision-Language Models**<br>[ICLR](https://iclr.cc/virtual/2026/poster/10011913) / [OpenReview](https://openreview.net/forum?id=0fib2BYc0L) | Zhangyang Qi, Zhixiong Zhang, Ye Fang 等 | Stage 1 source grounding、文本-体素融合与空间关系监督 |
| P1 | **IGGT: Instance-Grounded Geometry Transformer for Semantic 3D Reconstruction**<br>[ICLR](https://iclr.cc/virtual/2026/poster/10007025) / [OpenReview](https://openreview.net/forum?id=swiL18PmUV) | 昊 李, Zhengyu Zou, Fangfu Liu 等 | Stage 1 source grounding、文本-体素融合与空间关系监督 |
| P1 | **InternSpatial: A Comprehensive Dataset for Spatial Reasoning in Vision-Language Models**<br>[ICLR](https://iclr.cc/virtual/2026/poster/10010062) / [OpenReview](https://openreview.net/forum?id=L6bEitSMeu) | Nianchen Deng, Lixin Gu, Shenglong Ye 等 | Stage 1 source grounding、文本-体素融合与空间关系监督 |
| P1 | **MetaSpatial: Reinforcing 3D Spatial Reasoning in VLMs for the Metaverse**<br>[ICLR](https://iclr.cc/virtual/2026/poster/10010660) / [OpenReview](https://openreview.net/forum?id=EdQzLC0Zra) | Zhenyu Pan, Han Liu | Stage 1 source grounding、文本-体素融合与空间关系监督 |
| P1 | **OmniSpatial: Towards Comprehensive Spatial Reasoning Benchmark for Vision Language Models**<br>[ICLR](https://iclr.cc/virtual/2026/poster/10011350) / [OpenReview](https://openreview.net/forum?id=6nZKT2rL0H) | Mengdi Jia, Zekun Qi, Shaochen Zhang 等 | Stage 1 source grounding、文本-体素融合与空间关系监督 |
| P1 | **Part-X-MLLM: Part-aware 3D Multimodal Large Language Model**<br>[ICLR](https://iclr.cc/virtual/2026/poster/10009035) / [OpenReview](https://openreview.net/forum?id=WffiETiSeU) | Chunshi Wang, Junliang Ye, Yunhan Yang 等 | Stage 1 source grounding、文本-体素融合与空间关系监督 |
| P1 | **Pursuing Minimal Sufficiency in Spatial Reasoning**<br>[ICLR](https://iclr.cc/virtual/2026/poster/10008600) / [OpenReview](https://openreview.net/forum?id=bZAKJwyn1n) | Yejie Guo, Yunzhong Hou, Wufei Ma 等 | Stage 1 source grounding、文本-体素融合与空间关系监督 |
| P1 | **Reasoning in Space via Grounding in the World**<br>[ICLR](https://iclr.cc/virtual/2026/poster/10010835) / [OpenReview](https://openreview.net/forum?id=CfKi92bgnq) | Yiming Chen, Zekun Qi, Wenyao Zhang 等 | Stage 1 source grounding、文本-体素融合与空间关系监督 |
| P1 | **SCoT: Teaching 3D-LLMs to Think Spatially with Million-scale CoT Annotations**<br>[ICLR](https://iclr.cc/virtual/2026/poster/10011459) / [OpenReview](https://openreview.net/forum?id=5Tph6wFMOm) | Jinpeng Li, Haiping Wang, Jiabin chen 等 | Stage 1 source grounding、文本-体素融合与空间关系监督 |
| P1 | **Seeing Across Views: Benchmarking Spatial Reasoning of Vision-Language Models in Robotic Scenes**<br>[ICLR](https://iclr.cc/virtual/2026/poster/10007872) / [OpenReview](https://openreview.net/forum?id=jXDZJAfRZB) | ZhiYuan Feng, Zhaolu Kang, Qijie Wang 等 | Stage 1 source grounding、文本-体素融合与空间关系监督 |
| P1 | **SpaCE-10: A Comprehensive Benchmark for Multimodal Large Language Models in Compositional Spatial Intelligence**<br>[ICLR](https://iclr.cc/virtual/2026/poster/10010759) / [OpenReview](https://openreview.net/forum?id=Df7UjwEgIx) | Ziyang Gong, Wenhao Li, Xianzheng Ma 等 | Stage 1 source grounding、文本-体素融合与空间关系监督 |
| P1 | **Spatial Forcing: Implicit Spatial Representation Alignment for Vision-language-action Model**<br>[ICLR](https://iclr.cc/virtual/2026/poster/10008280) / [OpenReview](https://openreview.net/forum?id=euMVC1DO4k) | Fuhao Li, Wenxuan Song, Han Zhao 等 | Stage 1 source grounding、文本-体素融合与空间关系监督 |
| P1 | **Spatial-DISE: A Unified Benchmark for Evaluating Spatial Reasoning in Vision-Language Models**<br>[ICLR](https://iclr.cc/virtual/2026/poster/10008617) / [OpenReview](https://openreview.net/forum?id=bMINsPQpME) | Xinmiao Huang, Qisong He, Zhenglin Huang 等 | Stage 1 source grounding、文本-体素融合与空间关系监督 |
| P1 | **SpatiaLab: Can Vision–Language Models Perform Spatial Reasoning in the Wild?**<br>[ICLR](https://iclr.cc/virtual/2026/poster/10008229) / [OpenReview](https://openreview.net/forum?id=fWWUPOb0CT) | Azmine Toushik Wasi, Wahid Faisal, Abdur Rahman 等 | Stage 1 source grounding、文本-体素融合与空间关系监督 |
| P1 | **SpatialLadder: Progressive Training for Spatial Reasoning in Vision-Language Models**<br>[ICLR](https://iclr.cc/virtual/2026/poster/10010086) / [OpenReview](https://openreview.net/forum?id=KtrFXlvgrK) | Hongxing Li, Dingming Li, Zixuan Wang 等 | Stage 1 source grounding、文本-体素融合与空间关系监督 |
| P1 | **Spatially Guided Training for Vision-Language-Action Model**<br>[ICLR](https://iclr.cc/virtual/2026/poster/10008327) / [OpenReview](https://openreview.net/forum?id=eKhOrQWAVJ) | Jinhui Ye, Fangjing Wang, Ning Gao 等 | Stage 1 source grounding、文本-体素融合与空间关系监督 |
| P1 | **SpinBench: Perspective and Rotation as a Lens on Spatial Reasoning in VLMs**<br>[ICLR](https://iclr.cc/virtual/2026/poster/10007175) / [OpenReview](https://openreview.net/forum?id=r7rUDgGYC4) | Yuyou Zhang, Radu Corcodel, Chiori Hori 等 | Stage 1 source grounding、文本-体素融合与空间关系监督 |
| P1 | **Theory of Space: Can Foundation Models Construct Spatial Beliefs through Active Exploration?**<br>[ICLR](https://iclr.cc/virtual/2026/poster/10011179) / [OpenReview](https://openreview.net/forum?id=8iPwqr6Adk) | Pingyue Zhang, Zihan Huang, Yue Wang 等 | Stage 1 source grounding、文本-体素融合与空间关系监督 |

#### C. 3D 感知、对应与位姿

| 级别 | 论文 | 作者 | 与当前项目的关系 |
|---|---|---|---|
| P1 | **GeoPurify: A Data-Efficient Geometric Distillation Framework for Open-Vocabulary 3D Segmentation**<br>[ICLR](https://iclr.cc/virtual/2026/poster/10007584) / [OpenReview](https://openreview.net/forum?id=mN49LupE8l) | Weijia Dou, Xu Zhang, Yi Bin 等 | 稀疏 3D backbone、source box/pose、分割及几何对应 |
| P1 | **OVSeg3R: Learn Open-vocabulary Instance Segmentation from 2D via 3D Reconstruction**<br>[ICLR](https://iclr.cc/virtual/2026/poster/10007160) / [OpenReview](https://openreview.net/forum?id=rIMOXvaLAe) | Hongyang Li, Jinyuan Qu, Lei Zhang | 稀疏 3D backbone、source box/pose、分割及几何对应 |
| P1 | **PartSAM: A Scalable Promptable Part Segmentation Model Trained on Native 3D Data**<br>[ICLR](https://iclr.cc/virtual/2026/poster/10006586) / [OpenReview](https://openreview.net/forum?id=y8sZUQPYXC) | Zhe Zhu, Le Wan, Rui Xu 等 | 稀疏 3D backbone、source box/pose、分割及几何对应 |
| P1 | **Point-MoE: Large-Scale Multi-Dataset Training with Mixture-of-Experts for 3D Semantic Segmentation**<br>[ICLR](https://iclr.cc/virtual/2026/poster/10011672) / [OpenReview](https://openreview.net/forum?id=35HahPHrFG) | Xuweiyi Chen, Wentao Zhou, Aruni RoyChowdhury 等 | 稀疏 3D backbone、source box/pose、分割及几何对应 |

#### D. 机器人操作、VLA 与世界模型

| 级别 | 论文 | 作者 | 与当前项目的关系 |
|---|---|---|---|
| P2 | **Ctrl-World: A Controllable Generative World Model for Robot Manipulation**<br>[ICLR](https://iclr.cc/virtual/2026/poster/10011332) / [OpenReview](https://openreview.net/forum?id=748bHL2BAv) | Yanjiang Guo, Lucy Shi, Jianyu Chen 等 | 整体具身操作路线、VLA/世界模型基线与可执行策略接口 |
| P2 | **Embodied-R1: Reinforced Embodied Reasoning for General Robotic Manipulation**<br>[ICLR](https://iclr.cc/virtual/2026/poster/10007997) / [OpenReview](https://openreview.net/forum?id=i5wlozMFsQ) | Yifu Yuan, Haiqin Cui, Yaoting Huang 等 | 整体具身操作路线、VLA/世界模型基线与可执行策略接口 |
| P2 | **EquAct: An SE(3)-Equivariant Multi-Task Transformer for 3D Robotic Manipulation**<br>[ICLR](https://iclr.cc/virtual/2026/poster/10008448) / [OpenReview](https://openreview.net/forum?id=d1wuA8oIH0) | Xupeng Zhu, Yu Qi, Yizhe Zhu 等 | 整体具身操作路线、VLA/世界模型基线与可执行策略接口 |
| P2 | **Genie Envisioner: A Unified World Foundation Platform for Robotic Manipulation**<br>[ICLR](https://iclr.cc/virtual/2026/poster/10008250) / [OpenReview](https://openreview.net/forum?id=fHLtSxDFKC) | Yue Liao, Pengfei Zhou, Siyuan Huang 等 | 整体具身操作路线、VLA/世界模型基线与可执行策略接口 |
| P2 | **MemoryVLA: Perceptual-Cognitive Memory in Vision-Language-Action Models for Robotic Manipulation**<br>[ICLR](https://iclr.cc/virtual/2026/poster/10011504) / [OpenReview](https://openreview.net/forum?id=54U3XHf7qq) | Hao Shi, Bin Xie, Yingfei Liu 等 | 整体具身操作路线、VLA/世界模型基线与可执行策略接口 |
| P2 | **OneTwoVLA: A Unified Vision-Language-Action Model with Adaptive Reasoning**<br>[ICLR](https://iclr.cc/virtual/2026/poster/10006973) / [OpenReview](https://openreview.net/forum?id=tWMfhoP3as) | Fanqi Lin, Ruiqian Nai, Yingdong Hu 等 | 整体具身操作路线、VLA/世界模型基线与可执行策略接口 |
| P2 | **RAVEN: End-to-end Equivariant Robot Learning with RGB Cameras**<br>[ICLR](https://iclr.cc/virtual/2026/poster/10006488) / [OpenReview](https://openreview.net/forum?id=z8BN7KyaPl) | David Klee, Boce Hu, Andrew Cole 等 | 整体具身操作路线、VLA/世界模型基线与可执行策略接口 |
| P2 | **RoboOmni: Proactive Robot Manipulation in Omni-modal Context**<br>[ICLR](https://iclr.cc/virtual/2026/poster/10009796) / [OpenReview](https://openreview.net/forum?id=OJh7oBCYhL) | Siyin Wang, Jinlan Fu, Feihong Liu 等 | 整体具身操作路线、VLA/世界模型基线与可执行策略接口 |
| P2 | **Unified Vision-Language-Action Model**<br>[ICLR](https://iclr.cc/virtual/2026/poster/10009646) / [OpenReview](https://openreview.net/forum?id=PklMD8PwUy) | Yuqi Wang, Xinghang Li, Wenxuan Wang 等 | 整体具身操作路线、VLA/世界模型基线与可执行策略接口 |
| P2 | **Vlaser: Vision-Language-Action Model with Synergistic Embodied Reasoning**<br>[ICLR](https://iclr.cc/virtual/2026/poster/10011161) / [OpenReview](https://openreview.net/forum?id=8xTDnj39Ti) | Ganlin Yang, Tianyi Zhang, Haoran Hao 等 | 整体具身操作路线、VLA/世界模型基线与可执行策略接口 |
| P2 | **VLM4VLA: Revisiting Vision-Language-Models in Vision-Language-Action Models**<br>[ICLR](https://iclr.cc/virtual/2026/poster/10006964) / [OpenReview](https://openreview.net/forum?id=tc2UsBeODW) | Jianke Zhang, Xiaoyu Chen, Yanjiang Guo 等 | 整体具身操作路线、VLA/世界模型基线与可执行策略接口 |

#### E. 具身规划、数据集与评测

| 级别 | 论文 | 作者 | 与当前项目的关系 |
|---|---|---|---|
| P2 | **Beyond Static Vision: Scene Dynamic Field Unlocks Intuitive Physics Understanding in Multi-modal Large Language Models**<br>[ICLR](https://iclr.cc/virtual/2026/poster/10010991) / [OpenReview](https://openreview.net/forum?id=Ax02eR2c3d) | Nanxi Li, Xiang Wang, Yuanjie Chen 等 | 任务定义、数据生成、规划协议与评测维度 |
| P2 | **MomaGraph: State-Aware Unified Scene Graphs with Vision-Language Models for Embodied Task Planning**<br>[ICLR](https://iclr.cc/virtual/2026/poster/10011623) / [OpenReview](https://openreview.net/forum?id=3eTr9dGwJv) | Yuanchen Ju, Yongyuan Liang, Yen-Jen Wang 等 | 任务定义、数据生成、规划协议与评测维度 |
| P2 | **Towards Physically Executable 3D Gaussian for Embodied Navigation**<br>[ICLR](https://iclr.cc/virtual/2026/poster/10010422) / [OpenReview](https://openreview.net/forum?id=HB6KvsqcAn) | Bingchen Miao, Rong Wei, Zhiqi Ge 等 | 任务定义、数据生成、规划协议与评测维度 |

## 6. 按模块建立阅读顺序

1. **先定位最直接竞争工作**：读 P0 中 placement、affordance、3D spatial value map、scene arrangement、3D pose generation。
2. **再检查 Stage 1**：集中阅读 CoT3DRef、DenseGrounding、SPARTUN3D、SCoT、SceneCOT、GPT4Scene 和 3D Aware Region Prompted VLM。
3. **再检查 Stage 2**：集中阅读 Deep SE(3)、GravMAD、DenseMatcher、ComGS、language-aligned 3D keypoints、H2OFlow。
4. **最后补齐基线和评测**：阅读 2026 年空间推理 benchmarks、ManiSkill-HAB、Habitat 3.0，以及代表性 VLA/world-model 论文。

## 7. 官方来源

- [ICLR 2024 正式论文目录](https://iclr.cc/virtual/2024/papers.html)
- [ICLR 2025 正式论文目录](https://iclr.cc/virtual/2025/papers.html)
- [ICLR 2026 正式论文目录](https://iclr.cc/virtual/2026/papers.html)

说明：本清单基于官方目录全文筛选，并逐篇读取官方详情页核对元数据。项目代码链接并非每篇官方页面都提供，因此未把未经核验的第三方仓库混入主表。

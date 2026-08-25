# 数据集转换备注

- 新增或修改数据集转换脚本时，必须确认输出 canonical world 满足 `world-Z = 支撑面法向/重力上方向`。
- 不能直接假设原始数据集的 `world`、`scene`、`link` 坐标系已经符合该规范；如果不符合，需要在转换阶段对相机外参、物体位姿和点云施加同一个坐标规范化变换。
- 转换后的样本应在 `preprocess` 中记录坐标规范化方法、原始支撑面法向和对齐统计，便于后续排查 free_bbox 支撑面检测问题。

# 程序运行环境
使用conda 的spatial 环境进行运行代码

# 论文相关
ICLR2027_OUTLINE.md 是投稿iclr 2026论文的纲要，回答问题时需要参考这个上下文

# 放置评估指标

- benchmark 与训练验证统一使用下述定义。
- `Placement Success@K`：Top-K 中至少一个候选同时满足 `Language Relation Correct`、`Supported and Stable` 和 `Collision-Free`。
- `Source IoU`：预测源物体框与 GT 源物体框的完整 3D IoU；`IoU >= 0.5` 的准确率独立报告，不计入 `Placement Success@K`。
- `Placement Size IoU`：仅使用预测放置框与 GT 放置框的长、宽、高计算体积 IoU，不考虑中心和 yaw；该指标独立报告，不计入 `Placement Success@K`。
- `Language Relation Correct`：预测放置位置满足指令指定的空间关系。
- `Supported and Stable`：构造“预测放置中心 + GT 长宽高 + 预测 yaw”的评估框；使用其底面下方 3 cm 至上方 1 cm 的完整场景体素，经过 `3x3 binary closing`、孔洞填充和 8 邻域连通域处理后，评估框底面中心所在连通区域必须完全覆盖 yaw-only footprint；保留源物体原位置体素。
- `Collision-Free`：使用“预测放置中心 + GT 长宽高 + 预测 yaw”的评估框，沿用当前 benchmark 的碰撞评判实现。
- `Center Match Rate`：预测框底面中心与任一 GT 合法中心的距离不超过 2 cm；该指标独立报告。
- `Yaw Valid Given Center Match`：仅在中心匹配成功的样本中，判断预测 yaw 是否属于该匹配中心对应的 GT 合法 yaw 集合；该指标独立报告，不计入 `Placement Success@K`。

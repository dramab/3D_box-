# Free BBox Placement Annotation

本模块在 canonical HOPE 数据集上运行 3D bounding box 放置标注。输入不再从深度图构建 ray-casting 占据网格，而是读取规范数据集中的体素点云，例如：

```text
/data/jiajun.xie/3D_Box/data/hope/point_clouds_voxel_1cm/hope__scene_0000__0000.ply
```

## 主要规则

- 支撑面检测沿用旧 `free_bbox` 逻辑：优先在体素点云中 RANSAC 检测水平面，失败时回退到体素栅格逐层连通域。
- 支撑面候选会扣除 `table_z±1` 范围内所有场景物体 OBB 的 XY 投影，避免把薄物体或其它实体表面误选为可放置平面。
- OBB 体素化使用体素 AABB 与 OBB 的相交判断，边界接触也按占据处理，避免薄物体因没有覆盖体素中心而漏检。
- 3D box 的候选底层放在支撑面本层，不再使用 `table_z + 1`。
- 聚类仍使用 DBSCAN，但每个簇只输出一个最优 3D box。
- 簇内最优选择顺序为：底面中心热力最大、支撑面积最大、距离簇中心最近、距离碰撞障碍最近距离最大。
- 聚类前会强制过滤：候选 3D box 的底面中心体素必须落在支撑面 mask 上。
- 输出给模型监督和可视化的 `corners_world` / `transform_world` 统一为 yaw-only upright box：去掉 roll/pitch，并根据原 OBB 哪个局部轴最接近 world-Z 重排尺寸，使竖直尺寸落在输出 Z 轴。
- 搜索阶段仍可通过默认配置保留原始 roll/pitch 来做碰撞、可见性和遮挡过滤；对应的原始 6DoF 放置框会保存在 `obb6d_*` 字段中，便于排查。

## 输出文件

输出目录按文件类型分目录保存：

```text
support_masks/
  hope__scene_0000__0000__support_mask.ply
placements/
  hope__scene_0000__0000__placements.json
boxes/
  hope__scene_0000__0000__obj_0__cluster_000__box.json
heatmaps/
  hope__scene_0000__0000__obj_0__cluster_000__heatmap.ply
visualizations/
  hope__scene_0000__0000__obj_0__cluster_000__vis.png
```

- `*_support_mask.ply`: 每帧一份整体体素点云二值 mask，白色表示检测到的支撑面体素；点集同时保留形态学运算补出的非原始占据体素。
- `*_placements.json`: 每帧汇总标注。
- `*__box.json`: 每个簇选出的最优 3D box 单独保存；`placement.corners_world` 为模型监督使用的 yaw-only upright box，`placement.obb6d_corners_world` 为搜索阶段原始 6DoF OBB。
- `*__heatmap.ply`: 与该最优 3D box 对应的簇级整体体素热力点云，热力值为聚类前候选框底面中心落到每个支撑面体素上的次数；最优框底面中心位于该簇热力峰值。
- `*__vis.png`: 每个最终 freebox 一张可视化图片，绿色框与模型监督一致，显示 yaw-only upright box。

## 训练数据读取

构建模型训练数据时，优先读取 `*_placements.json` 中每个物体的 `placements[]`，或读取单个 `*__box.json` 的 `placement` 字段。当前导出的主字段已经是 yaw-only upright 监督框：

```python
center = placement["center_world"]              # [x, y, z]，yaw-only box 中心
yaw = placement["yaw_degrees"]                 # [0, 360) 度，只绕 world-Z
dx, dy, dz = placement["yaw_only_dimensions"]  # 输出局部 X/Y/Z 尺寸，单位同样本 unit
corners = placement["corners_world"]           # yaw-only box 的 8 个 world 角点
T = placement["transform_world"]               # yaw-only object->world 变换
```

这里的 `yaw_only_dimensions` 已经按放置姿态重排过：代码会从搜索阶段的 6DoF OBB 中找到最接近 `world-Z` 的局部轴，把该局部轴尺寸作为输出 Z 尺寸；剩余两个局部轴按 canonical 轴顺序作为输出 X/Y 尺寸。`yaw_degrees` 对应输出局部 X 轴在 world XY 平面的方向，因此 `corners_world` 等价于先构造局部尺寸为 `[dx, dy, dz]` 的居中 box，再施加 `Rz(yaw_degrees)` 和 `center_world` 平移。

如果模型字段名是 `(w, h, l, yaw)` 且 `h` 表示竖直高度，推荐按下面方式转换：

```python
dx, dy, dz = placement["yaw_only_dimensions"]
w = dx
h = dz
l = dy
yaw = placement["yaw_degrees"]
```

如果模型约定 yaw 指向 length 轴而不是 width/X 轴，则需要交换 `w/l` 并对 yaw 加 `90°` 后取模：

```python
w = dy
h = dz
l = dx
yaw = (placement["yaw_degrees"] + 90.0) % 360.0
```

不要用 `aabb_world[3:] - aabb_world[:3]` 作为 OBB 的 `w,h,l`。`aabb_world` 是 world 轴对齐外接框尺寸，只适合做范围检查；当 yaw 不为 0 时，它会混入旋转后的外接包围尺寸，不等于模型监督的 box 局部尺寸。

如果需要排查搜索阶段的真实 6DoF 放置姿态，可读取以下字段：

```python
obb6d_T = placement["obb6d_transform_world"]
obb6d_corners = placement["obb6d_corners_world"]
search_yaw = placement["search_yaw_degrees"]
axis_mapping = placement["yaw_only_axis_mapping"]  # canonical 轴到 yaw-only X/Y/Z 的映射
```

`obb6d_*` 字段仅用于调试和复现搜索结果，不作为 yaw-only 模型的默认监督。

## 运行示例

```bash
conda run -n spatial python tools/run_free_bbox_placement.py \
    --dataset-dir /data/jiajun.xie/3D_Box/data/hope \
    --sample-id hope__scene_0000__0000 \
    --output-dir outputs/free_bbox_hope
```

批量运行：

```bash
conda run -n spatial python tools/run_free_bbox_placement.py \
    --dataset-dir /data/jiajun.xie/3D_Box/data/hope \
    --all --max-frames 5 --workers 4 \
    --output-dir outputs/free_bbox_hope
```

批量任务默认使用最多 4 个进程并发处理不同帧；可通过 `--workers` 调整并发数，设置为 `1` 时串行运行。每个进程会独立创建 pipeline，避免跨进程共享计算状态。

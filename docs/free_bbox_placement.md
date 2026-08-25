# Free BBox Placement Annotation

本模块在 canonical HOPE 数据集上运行 3D bounding box 放置标注。输入不再从深度图构建 ray-casting 占据网格，而是读取规范数据集中的体素点云，例如：

```text
/data/jiajun.xie/3D_Box/data/hope/point_clouds_voxel_1cm/hope__scene_0000__0000.ply
```

## 主要规则

- 支撑面检测沿用旧 `free_bbox` 逻辑：优先在体素点云中 RANSAC 检测水平面，失败时回退到体素栅格逐层连通域。
- free_bbox 栅格原点按 voxel size 对齐到 canonical active 点云使用的世界格线，禁止以任意浮点 `scene_min` 建立第二套错位网格。
- `active center mask` 用于限制候选底面中心、heatmap 和 yaw 监督；稳定性则与 benchmark 共用完整场景点云的连通面积判定，并保留 source 原位置体素。
- 支撑面候选会扣除 `table_z±1` 范围内所有场景物体 OBB 的 XY 投影，避免把薄物体或其它实体表面误选为可放置平面。
- OBB 体素化使用体素 AABB 与 OBB 的相交判断，边界接触也按占据处理，避免薄物体因没有覆盖体素中心而漏检。
- 3D box 的候选底层放在支撑面本层，不再使用 `table_z + 1`。
- 聚类仍使用 DBSCAN，但每个簇只输出一个最优 3D box。
- 簇内最优选择顺序为：底面中心热力最大、支撑面积最大、距离簇中心最近、距离碰撞障碍最近距离最大。
- 无碰撞候选生成后立即过滤：候选 3D box 的底面中心必须落在 `active center mask`；聚类阶段会再次执行相同校验。
- 搜索、稳定性、可见性、遮挡、yaw-set 和最终导出统一使用 yaw-only upright box：先根据原 OBB 中最接近 world-Z 的局部轴重排尺寸，再仅扫描 world-Z yaw，避免搜索 footprint 与监督框不一致。
- 稳定性取最终 yaw-only 框底面下方 3 cm 至上方 1 cm 的完整场景体素，执行 `3×3 closing`、封闭孔洞填充和 8 邻域连通域标记，并要求 footprint 100% 位于覆盖底面中心的同一连通域。

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
yaw_sets/
  hope__scene_0000__0000__obj_0__cluster_000__yaw_set.npz
visualizations/
  hope__scene_0000__0000__obj_0__cluster_000__vis.png
```

- `*_support_mask.ply`: 每帧一份与模型输入 active voxel 对齐的二值 mask，白色表示允许作为候选底面中心的 active support；形态学补出的非 active 体素不写入模型监督 PLY。
- `*_placements.json`: 每帧汇总标注。
- `*__box.json`: 每个簇选出的最优 3D box 单独保存；搜索与 `placement.corners_world` 均使用 yaw-only upright box。历史 `obb6d_*` 字段保留用于兼容已有读取代码。
- `*__heatmap.ply`: 与该最优 3D box 对应的簇级 active 体素热力点云；所有正热力中心都必须属于 `active center mask`，最优框底面中心位于该簇热力峰值。
- `*__yaw_set.npz`: 与 heatmap 同簇的中心-yaw 监督。每个底面中心只保存一次，`valid_yaw_mask` 标记该中心通过碰撞、稳定性、可见性、遮挡和底面中心过滤的所有 yaw。新增目录不会修改现有 JSON schema 或字段。
- `*__vis.png`: 每个最终 freebox 一张可视化图片，绿色框与模型监督一致，显示 yaw-only upright box。

`yaw_sets` 文件可按以下方式读取：

```python
with np.load(yaw_set_path) as target:
    centers_world = target["bottom_center_world"]  # [P, 3]
    yaw_mask = target["valid_yaw_mask"]            # [P, yaw_steps]
    yaw_angles = target["yaw_angles_rad"]           # [yaw_steps]
    heat_counts = target["heat_counts"]             # [P]
```

`yaw_angles_rad` 就是搜索和最终 yaw-only upright Box 的局部 X 轴方向。

> **Stage 2 读取注意事项：** `yaw_sets` 保存的是语言方向过滤前、通过 free_bbox 几何过滤的全部可放置中心；`direction_filtered_heatmaps` 中的正样本是这些中心经过语言方向过滤后的子集。因此，构建 Stage 2 监督时必须先用 `(red == 255) & (blue == 30)` 提取方向过滤后仍保留的正点，再按 `bottom_center_world` 与对应 `yaw_set.npz` 对齐，仅读取匹配行的 `valid_yaw_mask`。不能直接把 yaw set 中的全部中心作为当前指令的正样本。
>
> Stage 2 按 canonical voxel key 关联 PLY 正点与 NPZ yaw center，并严格断言 support、heatmap 正点和 yaw center 都属于输入 active voxel；不再使用最近 active voxel 吸附或距离容错。yaw set 中存在未被方向过滤保留的额外中心是正常现象。

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

如果需要排查搜索阶段的 yaw-only 放置姿态，可读取以下兼容字段：

```python
obb6d_T = placement["obb6d_transform_world"]
obb6d_corners = placement["obb6d_corners_world"]
search_yaw = placement["search_yaw_degrees"]
axis_mapping = placement["yaw_only_axis_mapping"]  # canonical 轴到 yaw-only X/Y/Z 的映射
```

`obb6d_*` 是历史兼容命名；新生成标注中它们同样表示 yaw-only 搜索框，仅用于调试，不作为模型默认监督。

## 运行示例

```bash
python tools/run_free_bbox_placement.py \
    --dataset-dir /data/jiajun.xie/3D_Box/data/hope \
    --sample-id hope__scene_0000__0000 \
    --output-dir outputs/free_bbox_hope
```

批量运行：

```bash
python tools/run_free_bbox_placement.py \
    --dataset-dir /data/jiajun.xie/3D_Box/data/hope \
    --all --max-frames 5 --workers 4 \
    --output-dir outputs/free_bbox_hope
```

批量任务默认使用最多 4 个进程并发处理不同帧；可通过 `--workers` 调整并发数，设置为 `1` 时串行运行。每个进程会独立创建 pipeline，避免跨进程共享计算状态。

五个 canonical 数据集可分别运行：

```bash
python tools/run_free_bbox_placement.py \
    --dataset-dir data/dopose --all --workers 4 \
    --output-dir outputs/free_bbox_dopose

python tools/run_free_bbox_placement.py \
    --dataset-dir data/hope --all --workers 4 \
    --output-dir outputs/free_bbox_hope

python tools/run_free_bbox_placement.py \
    --dataset-dir data/housecat --all --workers 4 \
    --output-dir outputs/free_bbox_housecat

python tools/run_free_bbox_placement.py \
    --dataset-dir data/omni_filter --all --workers 4 \
    --output-dir outputs/free_bbox_omni

python tools/run_free_bbox_placement.py \
    --dataset-dir data/ycbv --all --workers 4 \
    --output-dir outputs/free_bbox_ycbv
```

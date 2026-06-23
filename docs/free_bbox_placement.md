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
- `*__box.json`: 每个簇选出的最优 3D box 单独保存。
- `*__heatmap.ply`: 与该最优 3D box 对应的簇级整体体素热力点云，热力值为聚类前候选框底面中心落到每个支撑面体素上的次数；最优框底面中心位于该簇热力峰值。
- `*__vis.png`: 每个最终 freebox 一张可视化图片，左侧为 RGB 投影，右侧为局部 3D 世界视图。

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

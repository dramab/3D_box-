# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 运行环境

所有命令使用 `conda` 的 `spatial` 环境运行：

```bash
conda run -n spatial python <script>
```

## 核心命令

### 数据集转换

```bash
# HOPE 转换
conda run -n spatial python tools/convert_hope_to_canonical.py \
    --root-dir /path/to/hope_video --mesh-dir /path/to/meshes/full \
    --output-dir data/canonical/hope --frame-step 60 --point-cloud-stride 4

# DOPose 转换
conda run -n spatial python tools/convert_dopose_to_canonical.py \
    --root-dir /data/jiajun.xie/Spatial-Affordance/data/dopose \
    --output-dir data/dopose --point-cloud-stride 4

# YCBV 转换
conda run -n spatial python tools/convert_ycbv_to_canonical.py \
    --root-dir /data/wenhao.hai/ycb_video/ycbv_test_all/test \
    --model-dir /data/wenhao.hai/ycb_video/ycbv_models/models \
    --output-dir data/ycbv --frame-step 5 --point-cloud-stride 4

# Omni6DPose 转换（ROPE 模式）
conda run -n spatial python tools/convert_omni_to_canonical.py \
    --root-dir /data/jiajun.xie/Spatial-Affordance/data/omni \
    --output-dir data/omni --frame-step 30 --point-cloud-stride 4

# HouseCat6D 转换
conda run -n spatial python tools/convert_housecat_to_canonical.py \
    --root-dir /data/wenhao.hai/housecat6D/housecat6D_trainset \
    --output-dir data/housecat --frame-step 30 --point-cloud-stride 4
```

### Free BBox 放置标注

```bash
# 单帧运行
conda run -n spatial python tools/run_free_bbox_placement.py \
    --dataset-dir data/hope --sample-id hope__scene_0000__0000 \
    --output-dir outputs/free_bbox_hope

# 批量运行
conda run -n spatial python tools/run_free_bbox_placement.py \
    --dataset-dir data/hope --all --max-frames 5 \
    --output-dir outputs/free_bbox_hope
```

### 自动空间关系标注

```bash
# 为 free_bbox 放置样本生成自然语言移动指令（逐 placement 一条）
conda run -n spatial python tools/run_auto_label.py \
    --placements-dir outputs/free_bbox_hope/placements \
    --dataset-dir data/hope \
    --output-dir outputs/auto_labels_hope

# 仅标注指定 sample_id
conda run -n spatial python tools/run_auto_label.py \
    --placements-dir outputs/free_bbox_hope/placements \
    --dataset-dir data/hope --output-dir outputs/auto_labels_hope \
    --sample-ids hope__scene_0000__0000 hope__scene_0000__0005
```

### 可视化检查

```bash
# 单帧 3D box RGB 投影
conda run -n spatial python tools/export_canonical_rgb_bbox_vis.py \
    --dataset-dir data/canonical/hope --sample-id hope__scene_0001__0000

# 批量导出
conda run -n spatial python tools/export_canonical_rgb_bbox_vis.py \
    --dataset-dir data/canonical/hope --all --max-frames 10
```

### 测试

```bash
conda run -n spatial python -m pytest tests/
# 运行单个测试
conda run -n spatial python -m pytest tests/test_dopose_support_plane_alignment.py
```

## 架构概览

### 整体数据流

```
原始数据集 (HOPE / DOPose / YCBV)
    └─> tools/convert_*_to_canonical.py
        └─> src/datasets/*_to_canonical.py
            └─> Canonical Dataset (data/canonical/<dataset>/)
                └─> tools/run_free_bbox_placement.py
                    └─> src/annotation/free_bbox/pipeline.py
                        └─> outputs/free_bbox_*/
                            └─> tools/run_auto_label.py
                                └─> src/annotation/auto_label.py
                                    └─> outputs/auto_labels_*/ (all_labels.json + report.html)
```

### Canonical Dataset (`src/datasets/`)

- `canonical.py`：核心数据结构 `CanonicalScene`、`CameraParams`、`ObjectInfo`，以及 JSON 读写函数 `load_canonical_scene` / `make_sample_record`
- `*_to_canonical.py`：各数据集的离线转换逻辑，输出统一 `canonical_placement_scene/v1` schema 的 JSON sample

**关键约定**：canonical world 坐标系中 **world-Z = 支撑面法向/重力向上方向**。新增数据集转换时必须确保这一点，不能假设原始坐标系已满足要求（需要做显式对齐）。

### Free BBox Pipeline (`src/annotation/free_bbox/`)

以 `FreeBBoxPipeline.run(scene, output_dir)` 为入口，逐物体执行以下流程：

1. **体素化**：从 `voxel_point_cloud_path` 读取 1cm 体素点云，构建 OCCUPIED/FREE 占据栅格（`occupancy.py`、`grid_ops.py`）
2. **支撑面检测**（`surface.py`）：优先 RANSAC 检测水平面，失败时退化为逐层连通域；支撑面候选扣除所有场景物体 OBB 内体素
3. **碰撞搜索**（`collision.py`）：在支撑面本层通过 FFT 卷积搜索无碰撞放置位置
4. **过滤**（`filters.py`）：稳定性过滤 → 可见性过滤 → 遮挡过滤 → 底面中心约束
5. **聚类**（`cluster.py`）：DBSCAN 聚类，每簇输出一个最优 box（优先支撑面积最大）
6. **输出**：`placements/*.json`、`boxes/*.json`、`heatmaps/*.ply`、`visualizations/*.png`、`support_masks/*.ply`

配置通过 `FreeBBoxConfig` 传入，默认单位 cm，体素大小 1cm。

### 模块职责速查

| 文件 | 职责 |
|------|------|
| `datatypes.py` | `FreeBBoxConfig`、`FreeBBoxResult` 数据类 |
| `geometry.py` | bbox 角点、坐标变换 |
| `voxel_utils.py` | 体素参数 `vp` 字典（`grid_min`、`voxel_size`） |
| `occupancy.py` | 体素栅格构建，`OCCUPIED`/`FREE` 常量 |
| `grid_ops.py` | 栅格预处理、OBB 体素化 |
| `io_utils.py` | PLY / JSON 读写 |
| `visualize.py` | RGB 投影 + 3D 局部视图可视化 |

## 数据集规范说明

- 深度图：`float32`，单位 cm，存储为 `.npy`
- 点云：固定 50000 点彩色 PLY（`point_clouds/`）；1cm 体素化版本（`point_clouds_voxel_1cm/`）
- DOPose 支撑面对齐失败时直接跳过该帧，`preprocess.coordinate_normalization` 记录对齐统计

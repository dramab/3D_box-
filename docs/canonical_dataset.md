# Canonical Placement Scene Dataset

统一数据集把原始数据集特有逻辑前置到离线转换阶段。后续 placement pipeline 只读取标准字段，不再依赖各数据集 adapter。

## 目录结构

```text
data/canonical/hope/
  manifest.json
  samples/
    hope__scene_0001__0000.json
  rgb/
    hope__scene_0001__0000.jpg
  depth/
    hope__scene_0001__0000.npy
  point_clouds/
    hope__scene_0001__0000.ply
  point_clouds_voxel_1cm/
    hope__scene_0001__0000.ply
```

## Sample Schema

```json
{
  "schema_version": "canonical_placement_scene/v1",
  "sample_id": "hope__scene_0001__0000",
  "scene_id": "scene_0001",
  "frame_id": "0000",
  "unit": "cm",
  "rgb_path": "rgb/hope__scene_0001__0000.jpg",
  "depth_path": "depth/hope__scene_0001__0000.npy",
  "point_cloud_path": "point_clouds/hope__scene_0001__0000.ply",
  "voxel_point_cloud_path": "point_clouds_voxel_1cm/hope__scene_0001__0000.ply",
  "camera": {
    "fx": 0.0,
    "fy": 0.0,
    "cx": 0.0,
    "cy": 0.0,
    "img_w": 0,
    "img_h": 0,
    "E_c2w": []
  },
  "objects": [
    {
      "obj_id": "obj_0",
      "class_name": "AlphabetSoup",
      "bbox3d_canonical": [],
      "pose_world": []
    }
  ]
}
```

`depth_path` 保存已经转换到 `cm` 的 `float32` 深度图。`E_c2w`、`pose_world`、`bbox3d_canonical` 与点云坐标使用同一单位。
`point_cloud_path` 保存固定 50000 点的彩色 PLY 点云；候选点不足 50000 时会放回采样补足。
`voxel_point_cloud_path` 保存 1cm 体素化后的彩色 PLY 点云，文件位于独立的 `point_clouds_voxel_1cm/` 目录；每个占用体素输出一个点，点坐标和颜色分别取体素内原点的平均值。
固定点数保存在 `manifest.json` 的 `preprocess.point_count`，不在每个 sample 中重复保存。
如果转换时提供 `--depth-window-cm`，深度窗口从有效深度的 1% 低分位值开始，避免单个过近离异点拉偏过滤范围。

### 字段说明

- `camera`: 当前帧的相机参数。`fx`、`fy`、`cx`、`cy` 为像素坐标系下的 pinhole 相机内参，后续可组装为 `K = [[fx, 0, cx], [0, fy, cy], [0, 0, 1]]`，因此不需要额外保存完整 `K` 矩阵；`img_w`、`img_h` 为 RGB 图像宽高；`E_c2w` 为 4x4 camera->world 齐次外参矩阵，可将相机坐标系下的点转换到 world 坐标系。需要 world->camera 时使用 `E_c2w` 的逆矩阵。
- `bbox3d_canonical`: 物体在 canonical/object 坐标系下的轴对齐 3D 包围盒，格式为 `[min_x, min_y, min_z, max_x, max_y, max_z]`。该字段只描述物体自身几何范围，不包含当前帧中的位姿变换。
- `pose_world`: 4x4 object->world 齐次位姿矩阵，可将 `bbox3d_canonical` 生成的物体局部坐标点转换到 world 坐标系。RGB 投影时通常先用 `pose_world` 得到 world 坐标，再用 `camera.E_c2w` 的逆矩阵转换到 camera 坐标。

## HOPE 转换

```bash
python tools/convert_hope_to_canonical.py \
    --root-dir /path/to/hope_video \
    --mesh-dir /path/to/meshes/full \
    --output-dir /data/jiajun.xie/3D_Box/data/canonical/hope \
    --frame-step 60 \
    --point-cloud-stride 4
```

转换输出的原始 `.ply` 点云都会统一重采样为 50000 点。`--point-cloud-stride` 只控制反投影生成候选点的像素步长，不改变最终保存点数。转换流程还会基于这份 50000 点点云额外保存 1cm 体素化后的 `.ply` 点云到 `point_clouds_voxel_1cm/`。

调试时可以先转换少量帧：

```bash
python tools/convert_hope_to_canonical.py \
    --root-dir /path/to/hope_video \
    --mesh-dir /path/to/meshes/full \
    --max-frames 5
```

## RGB 3D Box 投影检查

导出指定帧中所有物体 3D box 在 RGB 图像上的投影：

```bash
python tools/export_canonical_rgb_bbox_vis.py \
    --dataset-dir /data/jiajun.xie/3D_Box/data/canonical/hope \
    --sample-id hope__scene_0000__0000
```

批量导出 manifest 中的帧：

```bash
python tools/export_canonical_rgb_bbox_vis.py \
    --dataset-dir /data/jiajun.xie/3D_Box/data/canonical/hope \
    --all --max-frames 10
```

默认输出目录为：

```text
data/canonical/hope/rgb_bbox_vis/
```

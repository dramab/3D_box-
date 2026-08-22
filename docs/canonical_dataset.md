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

## DOPose 转换

```bash
python tools/convert_dopose_to_canonical.py \
    --root-dir /data/jiajun.xie/Spatial-Affordance/data/dopose \
    --output-dir /data/jiajun.xie/3D_Box/data/dopose \
    --point-cloud-stride 4
```

DOPose 转换会读取 `test_bin/` 和 `test_table/` 下的 BOP 风格场景。深度图按 `scene_camera.json` 中的 `depth_scale` 从 mm 转为 cm；物体位姿从 `scene_gt.json` 的 object->camera 转为 object->world；相机外参优先使用 `scene_transformations.json` 中的 `zivid_optical_frame -> scene_link`，缺失时退回相机坐标系。由于 `scene_link` 不保证以支撑面法向为 world-Z，转换时会先在 raw `scene_link`/world 中由深度图反投影生成点云，再用 RANSAC 拟合支撑面；随后将该支撑面法向对齐到 canonical world-Z，并对相机外参、物体位姿和点云同时施加该变换。支撑面拟合失败时该帧会直接转换失败，不再回退到物体 OBB 多数共识；每个 sample 的 `preprocess.coordinate_normalization` 会记录支撑面法向、inlier 数、残差、旋转角度和 z 平移等统计。

支撑面拟合相关参数可用于调试：

```bash
python tools/convert_dopose_to_canonical.py \
    --root-dir /data/jiajun.xie/Spatial-Affordance/data/dopose \
    --output-dir /data/jiajun.xie/3D_Box/data/dopose \
    --point-cloud-stride 4 \
    --support-plane-distance-thresh-cm 1.0 \
    --support-plane-ransac-iters 512 \
    --support-plane-min-inliers 500 \
    --support-plane-min-inlier-ratio 0.03 \
    --support-plane-max-points 50000
```

## YCBV 转换

```bash
python tools/convert_ycbv_to_canonical.py \
    --root-dir /data/wenhao.hai/ycb_video/ycbv_test_all/test \
    --model-dir /data/wenhao.hai/ycb_video/ycbv_models/models \
    --output-dir /data/jiajun.xie/3D_Box/data/ycbv \
    --frame-step 5 \
    --point-cloud-stride 4
```

YCBV 转换会读取 test 目录下的 BOP 风格场景。`--frame-step` 按帧 ID 取模采样，语义与 HOPE 转换一致。深度图按 `scene_camera.json` 中的 `depth_scale` 从 mm 转为 cm；相机外参由 `cam_R_w2c/cam_t_w2c` 求逆得到 camera->world；物体位姿从 `scene_gt.json` 的 object->camera 转为 object->world。只做调试时可以加 `--max-frames 2`。

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

> Omni6DPose 转换会在转换每帧时自动生成同名投影图到输出目录的 `rgb_bbox_vis/`，无需再单独运行此工具；被跳过的帧（pose 与 mask 横向偏差超阈值等）不会生成。

## 稀疏体素论文图

将 1 cm 稀疏体素点云按样本的真实相机参数投影到 RGB 像素平面，可得到与 RGB 观察方向严格一致、无三维网格和坐标轴的论文插图：

```bash
python tools/export_canonical_sparse_voxel_vis.py \
    --dataset-dir data/hope \
    --sample-id hope__scene_0000__0005 \
    --output-dir outputs/visualizations
```

工具会同时输出两组 PNG 预览和矢量 PDF：`*_sparse_voxels` 与 RGB 像素级对齐；`*_sparse_voxels_oblique_3d` 保留真实相机的水平观察方位，将 world-Z 校正为画面竖直方向，并以固定斜俯视透视突出桌面和物体的三维结构。像素对齐图的点面积由 `--point-size` 控制，默认值为 `24 pt^2`；斜俯视图由 `--oblique-point-size` 控制，默认值为 `36 pt^2`。

如需使用固定 50000 点的初始点云绘制更细粒度的同类图片，增加 `--point-source raw`：

```bash
python tools/export_canonical_sparse_voxel_vis.py \
    --dataset-dir data/hope \
    --sample-id hope__scene_0000__0005 \
    --output-dir outputs/visualizations \
    --point-source raw
```

初始点云模式输出 `*_raw_points` 和 `*_raw_points_oblique_3d`，不会覆盖稀疏体素图；两种图的默认点面积分别为 `2.5 pt^2` 和 `1.8 pt^2`，仍可用 `--point-size` 与 `--oblique-point-size` 覆盖。斜俯视图使用固定的弱透视相机、关闭深度着色，并以统一视角展示不同样本，避免远处颜色发灰和强透视造成的比例畸变。

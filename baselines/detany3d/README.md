# DetAny3D active_aligned zero-shot baseline

该目录用于隔离运行官方 DetAny3D，不修改项目原有训练环境。外部仓库位于
`external/`，权重位于 `checkpoints/`，推理结果默认写入
`outputs/detany3d_source_localization_exact_name/`。

## 模型输入约束

- GroundingDINO 只接收当前 RGB 和原物体精确名词短语，例如
  `krauter sauce box.`。
- DetAny3D 只接收当前 RGB 和 GroundingDINO 预测的最高置信度 2D 框。
- 深度、GT 2D/3D 框、物体位姿、相机外参和 placement 关系均不进入模型。
- GT 位姿和相机参数只在模型推理结束后用于绘制绿色对照框和计算中心误差。

完整移动指令不会直接作为提示词，因为目标位置与参考物体描述会给“定位原物体”引入无关语义。

## 环境

已创建独立 Conda 环境 `detany3d`：Python 3.8、PyTorch 1.13.1+cu116、
torchvision 0.14.1+cu116、MMCV 2.0.1。GroundingDINO CUDA 扩展编译在
`external/GroundingDINO/groundingdino/_C*.so`，运行脚本会直接加载项目内源码，
不会向 `spatial`、`da3` 或其他环境注册包。

官方源码会在模块导入时加载 Shapely 和 Open3D，但它们分别只用于训练期 3D IoU
和点云导出。当前外部仓库将这两个入口改为按需依赖，zero-shot 推理无需安装它们；
若以后调用相应功能，代码会明确提示补装依赖。

## 运行示例

```bash
source /home/limengfei/miniconda3/etc/profile.d/conda.sh
conda activate detany3d
python baselines/detany3d/run_zero_shot.py --split test --max-samples 10 --device cuda:0
```

只检查指定源物体：

```bash
python baselines/detany3d/run_zero_shot.py \
  --sample-id dopose__test_table_000003__000001 \
  --object-id obj_0 \
  --max-samples 1 \
  --device cuda:0
```

每张图左侧显示 GroundingDINO 的黄色 2D 框；右侧绿色为 GT 3D 框，蓝色为
DetAny3D 预测 3D 框。`results.json` 明确记录实际提示词、模型输入边界和仅评估使用的 GT 字段。

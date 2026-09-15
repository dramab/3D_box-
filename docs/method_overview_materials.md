# Method Overview Materials

`tools/export_method_overview_materials.py` 将方法总览图拆成 15 张独立素材图，
并保存一份 `materials_index.json` 记录每张图的数据来源。所有素材按 canonical
RGB 原始分辨率保存；除语言指令卡片外，所有几何素材都只在原始 RGB 上叠加几何元素，
不添加标题条、状态文字或抽象俯视图。

默认样本为 GT-faithful 的 `hope__scene_0000__0325 / obj_3`，对应指令为将
TomatoSauce 放到 CreamCheese 的 back-left 区域。默认不需要 checkpoint，使用
Stage 2 index 中的 GT source box、GT placement box、direction-filtered heatmap、
support mask 和 yaw set。

使用示例:

```bash
python tools/export_method_overview_materials.py \
    --config configs/lc_bgplacenet_stage2_enriched.yaml
```

指定样本:

```bash
python tools/export_method_overview_materials.py \
    --config configs/lc_bgplacenet_stage2_enriched.yaml \
    --sample-id hope__scene_0000__0325 \
    --object-id obj_3 \
    --reference-object-id obj_8
```

如需导出模型预测版本，额外传入 Stage 2 推理生成的 `predictions.json`。脚本会用
`place_box`、`source_box`、`pred_heatmap_ply` 和 `decoder_stages` 覆盖对应 GT
素材；未传入时全部使用 GT 或 GT-derived controlled probe。

指定 `--cluster-id 0` 可精确选择 cluster；同时填写对应标注的
`--reference-object-id`。此时默认输出目录增加 `__cluster_000` 后缀，
`materials_index.json` 记录 `cluster_id`，避免同一物体不同 cluster 相互覆盖。
场景中的 3D 框统一使用 6 像素的彩色线条，不添加厚黑色描边。
六个面使用同色半透明填充，每面不透明度约 10%，由远及近叠加绘制。

```bash
python tools/export_method_overview_materials.py \
    --config configs/lc_bgplacenet_stage2_enriched.yaml \
    --sample-id hope__scene_0000__0005 \
    --object-id obj_3 \
    --predictions-json outputs/lc_bgplacenet_stage2_space_former_aligned_loss_48query_full_gt_guass_8_enriched/inference_custom_hope_scene_0000_0005_tomato_sauce_back_left_mustard/predictions.json
```

默认输出目录:

```text
outputs/method_overview_materials/<sample_id>__<object_id>/
```

输出素材:

```text
01_language_instruction.png
02_rgb_observation.png
03_sparse_voxel_observation.png
04_source_object_grounding.png
05_coarse_placement_heatmap.png
06_decoder_iterative_refinement.png
07_boundary_space_reasoning.png
08_support_space_reasoning.png
09_final_physically_feasible_placement.png
10_semantic_consistency.png
11_bottom_surface_sampling.png
12_support_connectivity_check.png
13_boundary_sampling.png
14_clearance_check.png
15_collision_free_placement.png
05b_probability_colorbar.png
05c_sparse_voxel_probability_heatmap.png
materials_index.json
```

注意事项:

- `sparse_voxel_observation` 使用相机投影视角渲染 1cm voxel point cloud；默认最多绘制
  20000 个可见体素点，并用双层点半径增强可读性。
- 场景素材里的 3D box 使用亮色主线和黑色粗描边，缩小到论文总览图中仍能保持可见。
- GT heatmap 来自 `direction_filtered_heatmap_ply`，预测 heatmap 来自
  `pred_heatmap_ply`，二者在 `materials_index.json` 的 `heatmap_source` 中区分；
  `coarse_placement_heatmap` 会先用 `support_mask_ply` 的 active support 画完整支撑面底图，
  支撑面无响应区域使用 heatmap 最低概率蓝色；非零 placement response 会经高斯扩散后
  显示为蓝-青-黄-红概率分布，并只叠加 source box。
- `05b_probability_colorbar.png` 是 `coarse_placement_heatmap` 使用的概率颜色条，
  并在 `materials_index.json` 的 `extra_materials` 中记录。
- `05c_sparse_voxel_probability_heatmap.png` 将同一组高斯扩散后的 support probability
  映射回 1cm voxel point cloud；非支撑体素保持原 RGB，支撑面体素按概率色条着色。
- support 图使用项目统一评估定义：底面下方 3 cm 到上方 1 cm、`3x3 closing`、
  hole filling 和 8-connected component；连通域与 footprint cell 会投影回 RGB 图。
- `support_connectivity_check` 保留整个支撑连通域的显示范围，但绘制前过滤落入场景物体
  OBB 的支撑点；该过滤只影响可视化，不改变支撑评估结果。
- `semantic_consistency` 不画参考物框：蓝色定位原物体，绿色表示 GT，红色表示语义反例。
  三个框统一使用 6 像素线条和同色低不透明度填充，无黑色描边。
  反例在 GT 所在支撑连通域搜索，保持 GT 尺寸并遍历项目 yaw set 的角度，优先最大化相对参考物的方向夹角，
  再检查完整入镜、语言方向错误、稳定支撑和无碰撞；找不到合法反例时明确报错，不以 GT 冒充反例。
- `support_connectivity_check` 使用方向正确、无碰撞、但底面只有部分落在支撑连通域内的
  controlled probe，优先选择支撑覆盖率最接近 50% 的候选。径向搜索偏离目标超过
  10 个百分点时，补充搜索完整支撑网格及其 0.25 cm 偏移点。
  第 12 张若有框超出画面，会补白色画布并同步平移投影主点，保留原 RGB 像素比例；
  padding 记录在索引的 `support.image_padding_ltrb` 中。
- `boundary_sampling` 在候选 box 的各个面上采样，而不是只采样 12 条边。
- collision 图使用 Stage 2 benchmark 同源 OBB collision 检查；`clearance_check`
  使用侧面插入参考物体的 controlled collision probe，侵入深度为放置物体沿碰撞方向
  投影完整宽度的 50%（考虑 yaw），不再使用固定厘米范围。可视化 probe 的 OBB 上下文来自
  canonical 场景中的全部物体；实际碰撞的 box 侧面会被半透明红色填充，并用候选框内部与
  场景物体 OBB 表面重叠的真实几何采样点绘制紧凑的红色重叠区域；只有没有重叠几何点时才回退到带黑色描边的接触采样点。
  若与参考物构造的候选未发生碰撞，按相同侵入规则尝试其它场景物体并检查碰撞与入镜范围。

2026-09-07 指定的 22 个 cluster 可批量复现：

```bash
python outputs/method_overview_materials/batch_selected_20260907.py
```

该命令在每个 cluster 的独立目录生成素材，并生成批次结果 JSON、检查 JSON 和 HTML 浏览页。

# Introduction 无碰撞放置对比图

`tools/render_intro_collision_comparison.py` 批量筛选 RoboBrain2.5 与 LC-BGPlaceNet 的对比样本，并为每个样本渲染四个独立面板及一张合成图。当前默认读取 `outputs/robobrain2_5_front_behind_swapped/predictions.jsonl`，成图写入独立目录 `outputs/introduction_collision_comparison_front_behind_swapped/`，不会覆盖其他 prompt 变体的结果。

严格筛选要求 Ours 的 benchmark `placement_success_at_1=true`，且 RoboBrain2.5 的候选框与源物体之外的场景物体发生碰撞。RoboBrain2.5 只输出二维点；图中的红色三维候选框使用该点及源物体的观测尺寸、朝向构造，仅用于可视化。碰撞结论复用 benchmark 的 OBB 碰撞检测，蓝色框为 LC-BGPlaceNet 的预测放置框。

俯视图复用数据标注的图像对齐坐标：图像上方（front）对应俯视图上方，图像右侧对应俯视图右侧。两种方法共享相同视野和比例；深红区域表示 RoboBrain 候选框与场景物体在支撑平面上的重叠。RGB 图保留彩色三维框，但不显示物体名称和 Prediction 文本，避免遮挡原始画面；俯视图保留碰撞物体名称以解释空间关系，不额外标出指令参照物。

## 运行

```bash
conda activate spatial
python tools/render_intro_collision_comparison.py --all-suitable
```

仅为单个样本生成二维预览，并写入独立目录：

```bash
python tools/render_intro_collision_comparison.py \
  --item-id dopose__label_000015 \
  --output-dir outputs/introduction_collision_comparison_front_behind_swapped_2d_preview \
  --overwrite
```

每个样本使用独立文件夹，分别输出 RoboBrain RGB、RoboBrain 俯视图、Ours RGB、Ours 俯视图和四宫格合成图。五张图均保存为 300 DPI PNG 和矢量 PDF，共十个文件。根目录的 `selection_manifest.jsonl` 记录筛选结果与输出位置。

## 建议图注

> **Existing 2D goal prediction can violate object-level physical constraints.** Given the same instruction, RoboBrain2.5 predicts a 2D goal point whose visualization-only placement box intersects a scene object, whereas our method predicts a collision-free placement in the requested region. The top views are aligned with the RGB camera and share the same spatial extent.

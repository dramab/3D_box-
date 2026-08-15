# Introduction 无碰撞放置对比图

`tools/render_intro_collision_comparison.py` 将 RoboBrain2.5 与 LC-BGPlaceNet 的同一样本统一渲染为四个独立面板，方便后续自行组合 Introduction motivated example。原始结果文件不会被覆盖，成图默认写入 `outputs/introduction_collision_comparison/`。

默认案例为 `omni__label_041919__0e79bec87e9a05d1`。RoboBrain2.5 只输出二维点；图中的红色三维候选框使用该点及源物体的观测尺寸、朝向构造，仅用于可视化。碰撞结论复用 benchmark 的 OBB 碰撞检测，蓝色框为 LC-BGPlaceNet 的预测放置框。

俯视图采用相机对齐坐标：相机前方对应图的上方，相机右侧对应图的右侧。两种方法共享相同视野和比例；红色交叠区域表示 RoboBrain 候选框与 Pie 的碰撞体积在支撑平面上的重叠。

## 运行

```bash
conda activate spatial
python tools/render_intro_collision_comparison.py
```

脚本分别输出 RoboBrain RGB、RoboBrain 俯视图、Ours RGB 和 Ours 俯视图。每个面板均保存为 300 DPI PNG 和矢量 PDF，共八个文件；不再生成整张组合图和 JSON 文件。

## 建议图注

> **Existing 2D goal prediction can violate object-level physical constraints.** Given the same instruction, RoboBrain2.5 predicts a 2D goal point whose visualization-only placement box intersects the Pie, whereas our method predicts a collision-free placement in the requested back-left region. The top views are aligned with the RGB camera and share the same spatial extent.

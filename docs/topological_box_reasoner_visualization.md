# Topological Box Reasoner Visualization

`tools/render_topological_box_reasoner_vis.py` renders a four-panel paper figure
for `hope__scene_0000__0005`. The figure shows how a predicted placement
proposal field guides candidate boxes, and how box-boundary topology separates
stable support from collision-prone probes.

The support component in the topology panel is computed from the 1 cm voxel
point cloud with the same connected-support definition as the benchmark:
support-band projection, `3x3` binary closing, hole filling, and 8-connected
component labeling. The floating and collision boxes are controlled probes for
visualization and are written to the output metadata.

Usage:

```bash
python tools/render_topological_box_reasoner_vis.py
```

Custom inputs:

```bash
python tools/render_topological_box_reasoner_vis.py \
    --dataset-dir data/hope \
    --sample-id hope__scene_0000__0005 \
    --predictions-json outputs/lc_bgplacenet_stage2_space_former_aligned_loss_48query_full_gt_guass_8_enriched/inference_custom_hope_scene_0000_0005_tomato_sauce_back_left_mustard/predictions.json \
    --output-dir outputs/visualizations/topological_box_reasoner
```

Outputs:

- `outputs/visualizations/topological_box_reasoner/hope__scene_0000__0005_topological_box_reasoner.png`
- `outputs/visualizations/topological_box_reasoner/hope__scene_0000__0005_topological_box_reasoner.pdf`
- `outputs/visualizations/topological_box_reasoner/hope__scene_0000__0005_topological_box_reasoner_metadata.json`

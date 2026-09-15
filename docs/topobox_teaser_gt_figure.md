# TopoBox GT Teaser Figure

`tools/render_topobox_teaser_gt.py` renders a paper teaser figure whose scene
panels are generated from dataset GT metadata instead of manually edited images.

The script uses Stage2 enriched metadata to load canonical RGB frames, GT source
boxes, GT placement boxes, legal bottom centers, yaw sets, and point clouds. The
failure examples in the support and collision panels are controlled
perturbations recorded in the output metadata; the positive panels use dataset
GT boxes directly.

使用示例:

```bash
python tools/render_topobox_teaser_gt.py \
    --config configs/lc_bgplacenet_stage2_enriched.yaml
```

Outputs:

- `outputs/topobox_teaser_gt/topobox_teaser_gt.png`
- `outputs/topobox_teaser_gt/topobox_teaser_gt_metadata.json`

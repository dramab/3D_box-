#!/usr/bin/env python3
"""仅基于原始文本并发润色 HOPE placement 标签。"""

from enrich_labels_safe_common import DatasetConfig, main


CONFIG = DatasetConfig(
    dataset="hope",
    input_label_json="/data/limengfei/xingqunqi/3D_box-/outputs/auto_labels_hope/all_labels.json",
    input_rgb_image_dir="/data/limengfei/xingqunqi/3D_box-/data/hope/rgb",
    output_dir="enriched_label_outputs_hope",
    run_mode="hope_full",
)


if __name__ == "__main__":
    main(CONFIG)

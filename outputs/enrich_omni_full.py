#!/usr/bin/env python3
"""仅基于原始文本并发润色 Omni placement 标签。"""

from enrich_labels_safe_common import DatasetConfig, main


CONFIG = DatasetConfig(
    dataset="omni",
    input_label_json="/data/limengfei/xingqunqi/3D_box-/outputs/auto_labels_omni/all_labels.json",
    input_rgb_image_dir="/data/limengfei/xingqunqi/3D_box-/data/omni_filter/rgb",
    output_dir="enriched_label_outputs_omni",
    run_mode="omni_full",
)


if __name__ == "__main__":
    main(CONFIG)

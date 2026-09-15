from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from baselines.robobrain2_5.finetune.prepare_sft_data import (
    SCHEMA_VERSION,
    build_sft_row,
    pixel_to_normalized,
)
from baselines.robobrain2_5.run_zero_shot_test import bottom_center_xyd_prompt_from_item, load_sources


def test_pixel_to_normalized_matches_official_scale() -> None:
    assert pixel_to_normalized(np.asarray([320.0, 240.0]), 640, 480) == (500, 500)
    assert pixel_to_normalized(np.asarray([639.9, 479.9]), 640, 480) == (1000, 1000)


def test_real_sft_row_uses_visible_gt_and_exact_prompt() -> None:
    split_path = PROJECT_ROOT / "data/splits/active_aligned_enriched/train.json"
    import json

    with split_path.open("r", encoding="utf-8") as handle:
        item = json.load(handle)["items"][0]
    config_path = PROJECT_ROOT / "configs/lc_bgplacenet_stage2_enriched.yaml"
    row = build_sft_row(item, load_sources(config_path)[str(item["source_name"])])

    assert row["schema_version"] == SCHEMA_VERSION
    assert row["answer"].count(",") == 2
    assert row["answer"].startswith("[(") and row["answer"].endswith(")]")
    assert all(0 <= value <= 1000 for value in row["normalized_point"])
    assert row["depth_cm"] > 0.0
    assert row["model_prompt"].startswith("Please predict the bottom-center placement point")
    assert "[(x, y, d)]" in row["model_prompt"]
    assert Path(row["image_path"]).is_file()


def test_enriched_prompt_preserves_label_without_relation_rewrite() -> None:
    import json

    split_path = PROJECT_ROOT / "data/splits/active_aligned_enriched/train.json"
    with split_path.open("r", encoding="utf-8") as handle:
        item = json.load(handle)["items"][0]
    config_path = PROJECT_ROOT / "configs/lc_bgplacenet_stage2_enriched.yaml"
    prompt = bottom_center_xyd_prompt_from_item(item, load_sources(config_path)[str(item["source_name"])])

    assert f'"{item["instruction"]}"' in prompt
    assert "vacant space" not in prompt
    assert "the front left of Krauter Sauce Box" not in prompt
    assert "[(x, y, d)]" in prompt


def test_4b_sft_config_matches_8b_data_and_training_protocol() -> None:
    config_paths = {
        variant: PROJECT_ROOT / path
        for variant, path in {
            "4b": "configs/robobrain2_5_4b_sft_point_lora.yaml",
            "8b": "configs/robobrain2_5_sft_point_lora.yaml",
        }.items()
    }
    configs = {}
    for variant, path in config_paths.items():
        with path.open("r", encoding="utf-8") as handle:
            configs[variant] = yaml.safe_load(handle)

    assert configs["4b"]["data"] == configs["8b"]["data"]
    assert configs["4b"]["training"] | {"output_dir": None} == configs["8b"]["training"] | {
        "output_dir": None
    }
    assert configs["4b"]["model"]["lora"] == configs["8b"]["model"]["lora"]
    assert configs["4b"]["model"]["base_model"].endswith("RoboBrain2.5-4B")
    assert configs["4b"]["training"]["output_dir"] != configs["8b"]["training"]["output_dir"]

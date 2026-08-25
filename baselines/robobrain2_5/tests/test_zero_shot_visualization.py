from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from baselines.robobrain2_5.zero_shot_visualization import (
    Camera,
    backproject_world,
    local_depth_cm,
    normalized_to_pixel,
    parse_point_answer,
    parse_point_depth_answer,
    source_dimensions_and_yaw,
    upright_box_corners,
)
from baselines.robobrain2_5.run_zero_shot_test import (
    OFFICIAL_POINTING_SUFFIX,
    load_ground_truth_placement,
    load_sources,
    load_test_items,
    model_prompt_from_instruction,
    pointing_prompt_from_instruction,
    swap_front_behind_relation,
)


def test_parse_point_answer_matches_official_tuple_format() -> None:
    assert parse_point_answer("The location is [(120, 850)].")[0] == (120, 850)
    assert parse_point_answer("no coordinate")[1] == "parse_failed"


def test_parse_point_depth_answer_matches_single_xyd_tuple() -> None:
    assert parse_point_depth_answer("[(120, 850, 84.27)]")[0] == (120, 850, 84.27)
    assert parse_point_depth_answer("[(120, 850)]")[1] == "parse_failed"


def test_normalized_point_uses_official_clamping() -> None:
    assert normalized_to_pixel((1000, 0), 100, 50) == ((99, 0), True)


def test_depth_and_backprojection_use_canonical_centimeters() -> None:
    depth = np.array([[0.0, 100.0, 102.0], [98.0, 100.0, np.nan]], dtype=np.float32)
    assert local_depth_cm(depth, (1, 0), radius=1) == 100.0
    camera = Camera(100.0, 100.0, 1.0, 0.0, 3, 2, np.eye(4))
    assert np.allclose(backproject_world((1, 0), 100.0, camera), [0.0, 0.0, 100.0])


def test_source_geometry_creates_upright_render_box() -> None:
    source = {
        "bbox3d_canonical": [-1, -2, -3, 1, 2, 3],
        "pose_world": np.eye(4).tolist(),
    }
    dimensions, yaw = source_dimensions_and_yaw(source)
    corners = upright_box_corners(np.array([0.0, 0.0, 3.0]), dimensions, yaw)
    assert np.allclose(dimensions, [2.0, 4.0, 6.0])
    assert np.isclose(corners[:, 2].min(), 0.0)


def test_xyd_output_uses_model_depth_instead_of_rgbd(tmp_path: Path, monkeypatch) -> None:
    import baselines.robobrain2_5.run_zero_shot_test as run_module

    monkeypatch.setattr(run_module, "render_composite", lambda *args, **kwargs: Image.new("RGB", (8, 8)))
    camera = Camera(10.0, 10.0, 5.0, 5.0, 10, 10, np.eye(4))
    source = {
        "obj_id": "obj_0",
        "bbox3d_canonical": [-1, -1, 0, 1, 1, 2],
        "pose_world": np.eye(4).tolist(),
    }
    scene = {
        "rgb": np.zeros((10, 10, 3), dtype=np.uint8),
        "depth": np.full((10, 10), np.nan, dtype=np.float32),
        "camera": camera,
        "objects": [source],
    }
    gt_center = np.asarray([0.0, 0.0, 100.0])
    gt = {
        "bottom_center_world": gt_center,
        "box": [0, 0, 101, 2, 2, 2, 0],
        "corners": upright_box_corners(np.asarray([0.0, 0.0, 101.0]), np.asarray([2, 2, 2]), 0.0),
    }
    item = {
        "item_id": "item_0",
        "source_name": "source",
        "sample_id": "sample",
        "object_id": "obj_0",
        "instruction": "Place the object.",
    }

    row = run_module.output_record(
        item, scene, gt, "prompt", "[(500, 500, 100.0)]", tmp_path, prediction_format="xyd"
    )

    assert row["status"] == "ok"
    assert row["depth_source"] == "model_absolute_camera_depth_cm"
    assert row["predicted_depth_cm"] == 100.0
    assert np.allclose(row["world_point_cm"], [0.0, 0.0, 100.0])


def test_fixed_test_split_does_not_require_cluster_id() -> None:
    items = load_test_items(PROJECT_ROOT / "configs/lc_bgplacenet_stage2.yaml")
    assert items
    assert "cluster_id" not in items[0]


def test_move_template_becomes_official_vacant_space_prompt() -> None:
    prompt = pointing_prompt_from_instruction(
        "Move Krauter Sauce Box located at the right of Waschesteife Detergent Bottle to the back left of White Candle."
    )
    assert prompt == "Identify spot within the vacant space that's the front left of White Candle."
    assert "Krauter Sauce Box" not in prompt


def test_front_behind_relations_are_swapped_bidirectionally() -> None:
    cases = {
        "in front of White Candle": "behind White Candle",
        "behind White Candle": "in front of White Candle",
        "the front left of White Candle": "the back left of White Candle",
        "the back left of White Candle": "the front left of White Candle",
        "the front right of White Candle": "the back right of White Candle",
        "the back right of White Candle": "the front right of White Candle",
        "the left of White Candle": "the left of White Candle",
    }
    for source, expected in cases.items():
        assert swap_front_behind_relation(source) == expected


def test_exact_model_prompt_keeps_official_coordinate_suffix() -> None:
    prompt = model_prompt_from_instruction("Move Red Bowl to in front of White Candle.")
    assert prompt.startswith("Identify spot within the vacant space that's behind White Candle.")
    assert prompt.endswith(OFFICIAL_POINTING_SUFFIX)
    assert "[(x, y)]" in prompt


def test_ground_truth_placement_matches_fixed_split_label() -> None:
    config_path = PROJECT_ROOT / "configs/lc_bgplacenet_stage2.yaml"
    item = load_test_items(config_path)[0]
    ground_truth = load_ground_truth_placement(item, load_sources(config_path)[str(item["source_name"])])
    assert np.asarray(ground_truth["corners"]).shape == (8, 3)
    assert np.isclose(np.asarray(ground_truth["bottom_center_world"])[2], 0.5)

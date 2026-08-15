from pathlib import Path

import numpy as np

from baselines.detany3d.active_aligned import (
    ActiveAlignedRecord,
    gt_center_camera_m,
    obb_corners_camera_m,
    processed_pixels_to_original,
    source_prompt_from_instruction,
)


def test_source_prompt_uses_only_exact_source_noun_phrase() -> None:
    instruction = (
        "Move Krauter Sauce Box located at the right of Waschesteife Detergent Bottle "
        "to the back left of White Candle."
    )
    assert source_prompt_from_instruction(instruction) == "krauter sauce box."


def test_source_prompt_rejects_noncanonical_instruction() -> None:
    try:
        source_prompt_from_instruction("Put it elsewhere.")
    except ValueError as error:
        assert "Cannot extract source object" in str(error)
    else:
        raise AssertionError("Expected a malformed instruction to be rejected")


def test_gt_box_is_transformed_from_world_centimetres_to_camera_metres() -> None:
    pose_world = np.eye(4, dtype=np.float32)
    pose_world[:3, 3] = [100.0, 200.0, 300.0]
    camera_e_w2c = np.eye(4, dtype=np.float32)
    camera_e_w2c[:3, 3] = [-100.0, -200.0, -200.0]
    record = ActiveAlignedRecord(
        item_id="item",
        source_name="source",
        sample_id="sample",
        object_id="object",
        class_name="box",
        instruction="Move Box located at left of Cup to right of Plate.",
        prompt="box.",
        rgb_path=Path("unused.png"),
        camera_k=np.eye(3, dtype=np.float32),
        camera_e_w2c=camera_e_w2c,
        canonical_aabb_object_cm=np.asarray([-10, -20, -30, 10, 20, 30], dtype=np.float32),
        pose_world_cm=pose_world,
    )

    np.testing.assert_allclose(gt_center_camera_m(record), [0.0, 0.0, 1.0])
    corners = obb_corners_camera_m(record)
    np.testing.assert_allclose(corners.min(axis=0), [-0.1, -0.2, 0.7])
    np.testing.assert_allclose(corners.max(axis=0), [0.1, 0.2, 1.3])


def test_processed_pixels_are_mapped_back_after_center_crop() -> None:
    points = np.asarray([[0.0, 0.0], [880.0, 544.0]], dtype=np.float32)
    mapped = processed_pixels_to_original(
        points,
        original_hw=(1200, 1944),
        resized_hw=(553, 896),
        crop_xy=(0, 4),
    )
    np.testing.assert_allclose(mapped[0], [0.0, 4.0 * 1200.0 / 553.0])
    np.testing.assert_allclose(mapped[1], [880.0 * 1944.0 / 896.0, 548.0 * 1200.0 / 553.0])

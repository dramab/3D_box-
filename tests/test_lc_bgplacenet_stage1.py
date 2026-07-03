"""LC-BGPlaceNet Stage 1 data and loss tests."""

from __future__ import annotations

import json
import math

import numpy as np
import torch

from src.datasets.pointcloud import save_ply
from src.models.lc_bgplacenet.stage1 import aabb_iou_3d
from src.training.lc_bgplacenet_stage1 import (
    LCBGPlaceNetStage1Dataset,
    Stage1DataSource,
    STAGE1_SPLIT_SCHEMA_VERSION,
    build_stage1_index,
    compute_stage1_metrics,
    stage1_item_to_split_record,
    stage1_collate,
    source_box_loss,
)


def _write_json(path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def _make_tiny_stage1_source(tmp_path) -> Stage1DataSource:
    dataset_dir = tmp_path / "data" / "toy"
    free_bbox_dir = tmp_path / "outputs" / "free_bbox_toy"
    labels_dir = tmp_path / "outputs" / "auto_labels_toy"
    sample_id = "toy__scene_0000__0000"

    points = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [5.0, 0.0, 0.0],
        ],
        dtype=np.float32,
    )
    colors = np.array(
        [
            [10, 20, 30],
            [40, 50, 60],
            [70, 80, 90],
        ],
        dtype=np.uint8,
    )
    voxel_path = dataset_dir / "point_clouds_voxel_1cm" / f"{sample_id}.ply"
    save_ply(voxel_path, points, colors)

    _write_json(
        dataset_dir / "samples" / f"{sample_id}.json",
        {
            "schema_version": "canonical_placement_scene/v1",
            "sample_id": sample_id,
            "voxel_point_cloud_path": f"point_clouds_voxel_1cm/{sample_id}.ply",
        },
    )

    _write_json(
        free_bbox_dir / "placements" / f"{sample_id}__placements.json",
        {
            "schema_version": "free_bbox_placements/v1",
            "sample_id": sample_id,
            "objects": [
                {
                    "object_id": "obj_0",
                    "canonical_aabb_object": [-1.0, -2.0, -3.0, 1.0, 2.0, 3.0],
                    "original_pose_world": [
                        [math.sqrt(0.5), -math.sqrt(0.5), 0.0, 1.0],
                        [math.sqrt(0.5), math.sqrt(0.5), 0.0, 2.0],
                        [0.0, 0.0, 1.0, 3.0],
                        [0.0, 0.0, 0.0, 1.0],
                    ],
                    "original_aabb_world": [
                        1.0 - 3.0 * math.sqrt(0.5),
                        2.0 - 3.0 * math.sqrt(0.5),
                        0.0,
                        1.0 + 3.0 * math.sqrt(0.5),
                        2.0 + 3.0 * math.sqrt(0.5),
                        6.0,
                    ],
                }
            ],
        },
    )

    labels_path = labels_dir / "all_labels.json"
    vis_rel = f"visualizations/{sample_id}__obj_0__cluster_000__vis.png"
    vis_path = free_bbox_dir / vis_rel
    vis_path.parent.mkdir(parents=True, exist_ok=True)
    vis_path.touch()
    _write_json(
        labels_path,
        [
            {
                "sample_id": sample_id,
                "object_id": "obj_0",
                "label": "Move toy object to the right of the block.",
                "visualization_png": vis_rel,
            }
        ],
    )
    return Stage1DataSource(
        name="toy",
        dataset_dir=dataset_dir,
        free_bbox_dir=free_bbox_dir,
        labels_path=labels_path,
    )


def test_stage1_dataset_reads_source_box_and_instruction(tmp_path) -> None:
    """Dataset returns source box supervision and language instruction."""
    source = _make_tiny_stage1_source(tmp_path)
    dataset = LCBGPlaceNetStage1Dataset(
        [source],
        split="train",
        val_fraction=0.0,
    )

    sample = dataset[0]
    np.testing.assert_allclose(sample["source_box_gt"], [1.0, 2.0, 3.0, 2.0, 4.0, 6.0])
    assert sample["instruction"] == "Move toy object to the right of the block."


def test_stage1_dataset_reads_fixed_split_file(tmp_path) -> None:
    """A fixed split file selects samples without re-running random splitting."""
    source = _make_tiny_stage1_source(tmp_path)
    items = build_stage1_index([source])
    split_dir = tmp_path / "splits"
    _write_json(
        split_dir / "train.json",
        {
            "schema_version": STAGE1_SPLIT_SCHEMA_VERSION,
            "split": "train",
            "group_by": ["source_name", "sample_id"],
            "item_count": 1,
            "items": [stage1_item_to_split_record(items[0])],
        },
    )

    dataset = LCBGPlaceNetStage1Dataset(
        [source],
        split="train",
        val_fraction=0.5,
        split_dir=split_dir,
    )

    assert len(dataset) == 1
    assert dataset[0]["sample_id"] == "toy__scene_0000__0000"


def test_stage1_collate_builds_sparse_batch(tmp_path) -> None:
    """Collate returns sparse coordinates, normalized features and dense targets."""
    source = _make_tiny_stage1_source(tmp_path)
    dataset = LCBGPlaceNetStage1Dataset(
        [source],
        split="train",
        val_fraction=0.0,
    )
    batch = stage1_collate([dataset[0]], voxel_size_cm=1.0)

    assert batch["features"].shape == (3, 6)
    assert batch["sparse_coords"].shape == (3, 4)
    assert batch["source_box_gt"].shape == (1, 6)
    assert batch["instructions"] == ["Move toy object to the right of the block."]


def test_aabb_iou_and_source_loss_for_identical_boxes() -> None:
    """Identical source boxes should have IoU near 1 and zero regression loss."""
    box = torch.tensor([[1.0, 2.0, 3.0, 2.0, 4.0, 6.0]], dtype=torch.float32)
    iou = aabb_iou_3d(box, box)
    loss, terms = source_box_loss(box, box, lambda_center=1.0, lambda_size=1.0, lambda_iou=0.5)

    assert torch.allclose(iou, torch.ones_like(iou), atol=1e-5)
    assert float(loss) < 1e-5
    assert float(terms["source_iou"]) > 0.999


def test_stage1_source_metrics() -> None:
    """Source metrics should be perfect for identical predicted and target boxes."""
    outputs = {
        "source_box": torch.tensor([[0.0, 0.0, 0.0, 1.0, 1.0, 1.0]]),
    }
    batch = {
        "source_box_gt": torch.tensor([[0.0, 0.0, 0.0, 1.0, 1.0, 1.0]]),
    }
    metrics = compute_stage1_metrics(outputs, batch)

    assert metrics["source_center_mae_cm"] == 0.0
    assert metrics["source_size_mae_cm"] == 0.0
    assert metrics["source_iou"] > 0.999

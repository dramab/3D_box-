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
    compute_stage1_loss,
    compute_stage1_metrics,
    stage1_item_to_split_record,
    stage1_collate,
    source_box_loss,
    support_loss,
)


def _write_json(path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def _box_corners(xmin, ymin, zmin, xmax, ymax, zmax) -> list[list[float]]:
    """Build axis-aligned box corners in the same bottom/top order as free_bbox."""
    return [
        [xmin, ymin, zmin],
        [xmax, ymin, zmin],
        [xmin, ymax, zmin],
        [xmax, ymax, zmin],
        [xmin, ymin, zmax],
        [xmax, ymin, zmax],
        [xmin, ymax, zmax],
        [xmax, ymax, zmax],
    ]


def _make_tiny_stage1_source(tmp_path) -> Stage1DataSource:
    dataset_dir = tmp_path / "data" / "toy"
    free_bbox_dir = tmp_path / "outputs" / "free_bbox_toy"
    labels_dir = tmp_path / "outputs" / "auto_labels_toy"
    sample_id = "toy__scene_0000__0000"

    points = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [3.0, 0.0, 0.0],
            [4.0, 0.0, 0.0],
            [5.0, 0.0, 0.0],
            [6.0, 0.0, 0.0],
            [20.0, 0.0, 0.0],
            [21.0, 0.0, 0.0],
            [22.0, 0.0, 0.0],
        ],
        dtype=np.float32,
    )
    colors = np.array(
        [
            [10, 20, 30],
            [40, 50, 60],
            [55, 65, 75],
            [70, 80, 90],
            [80, 90, 100],
            [85, 95, 105],
            [90, 100, 110],
            [95, 105, 115],
            [100, 110, 120],
            [105, 115, 125],
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

    support_path = free_bbox_dir / "support_masks" / f"{sample_id}__support_mask.ply"
    support_points = np.array(
        [
            *[[float(x), float(y), 0.0] for x in range(7) for y in range(-3, 4)],
            *[[float(x), float(y), 0.0] for x in range(20, 23) for y in range(-1, 2)],
        ],
        dtype=np.float32,
    )
    save_ply(
        support_path,
        support_points,
        np.full((len(support_points), 3), 255, dtype=np.uint8),
    )

    vis_0 = f"visualizations/{sample_id}__obj_0__cluster_000__vis.png"
    vis_1 = f"visualizations/{sample_id}__obj_0__cluster_001__vis.png"
    for vis_rel in (vis_0, vis_1):
        vis_path = free_bbox_dir / vis_rel
        vis_path.parent.mkdir(parents=True, exist_ok=True)
        vis_path.touch()

    _write_json(
        free_bbox_dir / "placements" / f"{sample_id}__placements.json",
        {
            "schema_version": "free_bbox_placements/v1",
            "sample_id": sample_id,
            "support_mask_ply": f"support_masks/{sample_id}__support_mask.ply",
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
                    "placements": [
                        {
                            "sample_id": f"{sample_id}_obj_0_cluster_000",
                            "cluster_id": 0,
                            "visualization_png": vis_0,
                            "corners_world": _box_corners(2.8, -0.2, 0.0, 3.2, 0.2, 2.0),
                        },
                        {
                            "sample_id": f"{sample_id}_obj_0_cluster_001",
                            "cluster_id": 1,
                            "visualization_png": vis_1,
                            "corners_world": _box_corners(18.5, -2.5, 0.0, 23.5, 2.5, 2.0),
                        },
                    ],
                }
            ],
        },
    )

    labels_path = labels_dir / "all_labels.json"
    _write_json(
        labels_path,
        [
            {
                "sample_id": sample_id,
                "object_id": "obj_0",
                "cluster_id": 0,
                "placement_sample_id": f"{sample_id}_obj_0_cluster_000",
                "label": "Move toy object to the right of the block.",
                "visualization_png": vis_0,
            },
            {
                "sample_id": sample_id,
                "object_id": "obj_0",
                "cluster_id": 1,
                "placement_sample_id": f"{sample_id}_obj_0_cluster_001",
                "label": "Move toy object to the left of the block.",
                "visualization_png": vis_1,
            },
        ],
    )
    return Stage1DataSource(
        name="toy",
        dataset_dir=dataset_dir,
        free_bbox_dir=free_bbox_dir,
        labels_path=labels_path,
    )


def _make_dataset_with_all_train_items(
    tmp_path,
    source: Stage1DataSource,
) -> LCBGPlaceNetStage1Dataset:
    """Build a dataset with all toy label rows fixed into the train split."""
    items = build_stage1_index([source])
    split_dir = tmp_path / "splits_all"
    _write_json(
        split_dir / "train.json",
        {
            "schema_version": STAGE1_SPLIT_SCHEMA_VERSION,
            "split": "train",
            "group_by": ["source_name", "sample_id"],
            "item_count": len(items),
            "items": [stage1_item_to_split_record(item) for item in items],
        },
    )
    return LCBGPlaceNetStage1Dataset(
        [source],
        split="train",
        support_align_threshold_cm=0.25,
        support_radius_area_fraction=0.25,
        voxel_size_cm=1.0,
        split_dir=split_dir,
    )


def test_stage1_dataset_builds_area_scaled_support_center_labels(tmp_path) -> None:
    """Support-center labels scale with the GT-center support component area."""
    source = _make_tiny_stage1_source(tmp_path)
    dataset = _make_dataset_with_all_train_items(tmp_path, source)

    samples = {dataset[idx]["placement_sample_id"]: dataset[idx] for idx in range(len(dataset))}
    sample_0 = samples["toy__scene_0000__0000_obj_0_cluster_000"]
    sample_1 = samples["toy__scene_0000__0000_obj_0_cluster_001"]

    np.testing.assert_allclose(sample_0["source_box_gt"], [1.0, 2.0, 3.0, 2.0, 4.0, 6.0])
    np.testing.assert_array_equal(sample_0["support_label"], [0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    np.testing.assert_array_equal(sample_1["support_label"], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0])
    assert sample_0["support_label"].sum() > sample_1["support_label"].sum()
    assert sample_0["cluster_id"] == 0
    assert sample_1["cluster_id"] == 1
    assert sample_0["instruction"] == "Move toy object to the right of the block."


def test_stage1_index_resolves_label_to_placement_geometry(tmp_path) -> None:
    """Index items keep the placement metadata needed to build support labels."""
    source = _make_tiny_stage1_source(tmp_path)
    items = build_stage1_index([source])
    item_by_placement = {item.placement_sample_id: item for item in items}
    item = item_by_placement["toy__scene_0000__0000_obj_0_cluster_001"]

    assert item.cluster_id == 1
    assert item.support_mask_path.name.endswith("__support_mask.ply")
    assert item.placement_corners_world.shape == (8, 3)


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
        support_align_threshold_cm=0.25,
        support_radius_area_fraction=0.25,
        voxel_size_cm=1.0,
        split_dir=split_dir,
    )

    assert len(dataset) == 1
    assert dataset[0]["sample_id"] == "toy__scene_0000__0000"
    assert dataset[0]["placement_sample_id"] == items[0].placement_sample_id


def test_stage1_collate_builds_sparse_batch(tmp_path) -> None:
    """Collate returns sparse coordinates, normalized features and dense targets."""
    source = _make_tiny_stage1_source(tmp_path)
    dataset = _make_dataset_with_all_train_items(tmp_path, source)
    sample = next(
        dataset[idx]
        for idx in range(len(dataset))
        if dataset[idx]["placement_sample_id"] == "toy__scene_0000__0000_obj_0_cluster_000"
    )
    batch = stage1_collate([sample], voxel_size_cm=1.0)

    assert batch["features"].shape == (10, 6)
    assert batch["sparse_coords"].shape == (10, 4)
    assert batch["support_labels"].tolist() == [0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    assert batch["source_box_gt"].shape == (1, 6)
    assert batch["instructions"] == ["Move toy object to the right of the block."]
    assert batch["placement_sample_ids"] == ["toy__scene_0000__0000_obj_0_cluster_000"]
    assert batch["cluster_ids"] == [0]


def test_aabb_iou_and_source_loss_for_identical_boxes() -> None:
    """Identical source boxes should have IoU near 1 and zero regression loss."""
    box = torch.tensor([[1.0, 2.0, 3.0, 2.0, 4.0, 6.0]], dtype=torch.float32)
    iou = aabb_iou_3d(box, box)
    loss, terms = source_box_loss(box, box, lambda_center=1.0, lambda_size=1.0, lambda_iou=0.5)

    assert torch.allclose(iou, torch.ones_like(iou), atol=1e-5)
    assert float(loss) < 1e-5
    assert float(terms["source_iou"]) > 0.999


def test_stage1_support_metrics() -> None:
    """Support metrics should be perfect when thresholded logits match labels."""
    outputs = {
        "source_box": torch.tensor([[0.0, 0.0, 0.0, 1.0, 1.0, 1.0]]),
        "support_logits": torch.tensor([-5.0, 5.0, -5.0]),
    }
    batch = {
        "source_box_gt": torch.tensor([[0.0, 0.0, 0.0, 1.0, 1.0, 1.0]]),
        "support_labels": torch.tensor([0.0, 1.0, 0.0]),
        "support_align_coverage": torch.tensor([1.0]),
    }
    metrics = compute_stage1_metrics(outputs, batch)

    assert metrics["support_precision"] > 0.999
    assert metrics["support_recall"] > 0.999
    assert metrics["support_f1"] > 0.999


def test_stage1_support_loss_auto_pos_weight_balances_batch() -> None:
    """Auto support pos_weight should up-weight sparse positive voxels."""
    logits = torch.zeros(4)
    labels = torch.tensor([1.0, 0.0, 0.0, 0.0])

    loss = support_loss(logits, labels, pos_weight="auto")
    expected = torch.nn.functional.binary_cross_entropy_with_logits(
        logits,
        labels,
        pos_weight=torch.tensor(3.0),
    )

    assert torch.allclose(loss, expected)


def test_compute_stage1_loss_logs_auto_support_pos_weight() -> None:
    """Stage 1 loss logging should expose the batch-adaptive support weight."""
    outputs = {
        "source_box": torch.tensor([[0.0, 0.0, 0.0, 1.0, 1.0, 1.0]]),
        "support_logits": torch.zeros(3),
    }
    batch = {
        "source_box_gt": torch.tensor([[0.0, 0.0, 0.0, 1.0, 1.0, 1.0]]),
        "support_labels": torch.tensor([1.0, 0.0, 0.0]),
    }
    cfg = {
        "loss": {
            "lambda_src": 1.0,
            "lambda_sup": 1.0,
            "source": {"lambda_center": 2.0, "lambda_size": 1.5, "lambda_iou": 0.2},
            "support": {"pos_weight": "auto", "max_pos_weight": 20.0},
        }
    }

    losses = compute_stage1_loss(outputs, batch, cfg)

    assert torch.allclose(losses["support_pos_weight"], torch.tensor(2.0))

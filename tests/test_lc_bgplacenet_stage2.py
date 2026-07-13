"""LC-BGPlaceNet Stage 2 data, decode and loss tests."""

from __future__ import annotations

import json
import math

import numpy as np
import pytest
import torch
from PIL import Image

from src.annotation.free_bbox.io_utils import save_ply
from src.models.lc_bgplacenet.stage2 import SourceConditionedDensePlacementField, decode_place_box
from src.training.lc_bgplacenet_stage1 import Stage1DataSource
from src.training.lc_bgplacenet_stage2 import (
    LCBGPlaceNetStage2Dataset,
    build_dense_heatmap_targets,
    build_stage2_index,
    compute_stage2_loss,
    stage2_collate,
)


def _write_json(path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def _make_tiny_stage2_source(tmp_path) -> Stage1DataSource:
    dataset_dir = tmp_path / "data" / "toy"
    free_bbox_dir = tmp_path / "outputs" / "free_bbox_toy"
    labels_dir = tmp_path / "outputs" / "auto_labels_toy"
    sample_id = "toy__scene_0000__0000"

    points = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
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
    rgb_path = dataset_dir / "rgb" / f"{sample_id}.png"
    rgb_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.full((8, 8, 3), 128, dtype=np.uint8)).save(rgb_path)
    _write_json(
        dataset_dir / "samples" / f"{sample_id}.json",
        {
            "schema_version": "canonical_placement_scene/v1",
            "sample_id": sample_id,
            "rgb_path": f"rgb/{sample_id}.png",
            "voxel_point_cloud_path": f"point_clouds_voxel_1cm/{sample_id}.ply",
            "camera": {
                "fx": 1.0,
                "fy": 1.0,
                "cx": 0.0,
                "cy": 0.0,
                "img_w": 8,
                "img_h": 8,
                "E_c2w": np.eye(4).tolist(),
            },
        },
    )

    support_rel = f"support_masks/{sample_id}__support_mask.ply"
    support_colors = np.array(
        [
            [55, 55, 55],
            [255, 255, 255],
            [55, 55, 55],
        ],
        dtype=np.uint8,
    )
    save_ply(free_bbox_dir / support_rel, points, support_colors)

    raw_heatmap_rel = f"heatmaps/{sample_id}__obj_0__cluster_000__heatmap.ply"
    filtered_heatmap_path = free_bbox_dir / "direction_filtered_heatmaps" / f"{sample_id}__obj_0__cluster_000__heatmap.ply"
    heatmap_colors = np.array(
        [
            [60, 60, 60],
            [255, 0, 30],
            [55, 120, 210],
        ],
        dtype=np.uint8,
    )
    save_ply(filtered_heatmap_path, points, heatmap_colors)

    placement_sample_id = f"{sample_id}_obj_0_cluster_000"
    placement = {
        "sample_id": placement_sample_id,
        "cluster_id": 0,
        "center_world": [1.0, 0.0, 1.5],
        "yaw_degrees": 90.0,
        "yaw_only_dimensions": [2.0, 4.0, 3.0],
        "bottom_center_world": [1.0, 0.0, 0.0],
        "heatmap_ply": raw_heatmap_rel,
        "visualization_png": f"visualizations/{sample_id}__obj_0__cluster_000__vis.png",
    }
    _write_json(
        free_bbox_dir / "placements" / f"{sample_id}__placements.json",
        {
            "schema_version": "free_bbox_placements/v1",
            "sample_id": sample_id,
            "voxel_point_cloud_path": f"point_clouds_voxel_1cm/{sample_id}.ply",
            "support_mask_ply": support_rel,
            "objects": [
                {
                    "object_id": "obj_0",
                    "canonical_aabb_object": [-1.0, -2.0, -3.0, 1.0, 2.0, 3.0],
                    "original_pose_world": [
                        [1.0, 0.0, 0.0, 1.0],
                        [0.0, 1.0, 0.0, 2.0],
                        [0.0, 0.0, 1.0, 3.0],
                        [0.0, 0.0, 0.0, 1.0],
                    ],
                    "placements": [placement],
                }
            ],
        },
    )

    vis_rel = placement["visualization_png"]
    vis_path = free_bbox_dir / vis_rel
    vis_path.parent.mkdir(parents=True, exist_ok=True)
    vis_path.touch()
    labels_path = labels_dir / "all_labels.json"
    _write_json(
        labels_path,
        [
            {
                "sample_id": sample_id,
                "object_id": "obj_0",
                "cluster_id": 0,
                "placement_sample_id": placement_sample_id,
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


def test_stage2_index_reads_placement_and_supervision_paths(tmp_path) -> None:
    """Stage 2 index resolves placement GT, support mask and filtered heatmap."""
    source = _make_tiny_stage2_source(tmp_path)
    items = build_stage2_index([source])

    assert len(items) == 1
    item = items[0]
    assert item.cluster_id == 0
    assert item.support_mask_ply.exists()
    assert item.direction_filtered_heatmap_ply.exists()
    np.testing.assert_allclose(item.place_box_gt[:6], [1.0, 0.0, 1.5, 2.0, 4.0, 3.0])
    assert np.isclose(item.place_box_gt[6], math.pi / 2)


def test_stage2_collate_and_heatmap_target_are_support_limited(tmp_path) -> None:
    """Dense heatmap targets are nonzero only on support-mask active voxels."""
    source = _make_tiny_stage2_source(tmp_path)
    dataset = LCBGPlaceNetStage2Dataset([source], split="train", val_fraction=0.0)
    batch = stage2_collate([dataset[0]], voxel_size_cm=1.0)
    heatmap = build_dense_heatmap_targets(
        world_coords=batch["world_coords"],
        batch_indices=batch["batch_indices"],
        positive_points=batch["heatmap_positive_points"],
        positive_batch_indices=batch["heatmap_positive_batch_indices"],
        support_masks=batch["support_masks"],
        batch_size=1,
        sigma=2.0,
    )

    assert batch["support_masks"].tolist() == [False, True, False]
    assert len(batch["images"]) == 1
    assert batch["camera_K"].shape == (1, 3, 3)
    assert batch["camera_E_w2c"].shape == (1, 4, 4)
    assert batch["image_hw"].tolist() == [[8.0, 8.0]]
    assert float(heatmap[1]) > 0.99
    assert float(heatmap[0]) == 0.0
    assert float(heatmap[2]) == 0.0


def test_decode_place_box_uses_best_voxel_offset_and_source_size() -> None:
    """Box decode should use best heatmap voxel, offset and exp size residual."""
    outputs = decode_place_box(
        heatmap_logits=torch.tensor([0.0, 4.0, 1.0]),
        bottom_offset=torch.tensor([[0.0, 0.0, 0.0], [0.5, 0.0, 0.0], [0.0, 0.0, 0.0]]),
        yaw_sincos=torch.tensor([[0.0, 1.0], [1.0, 0.0], [0.0, 1.0]]),
        size_residual=torch.zeros((1, 3)),
        source_size=torch.tensor([[2.0, 4.0, 3.0]]),
        voxel_centers=torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]]),
        batch_indices=torch.tensor([0, 0, 0]),
        batch_size=1,
    )

    np.testing.assert_allclose(outputs["bottom_center"].numpy(), [[1.5, 0.0, 0.0]], atol=1e-6)
    np.testing.assert_allclose(outputs["place_box"][0, :6].numpy(), [1.5, 0.0, 1.5, 2.0, 4.0, 3.0], atol=1e-6)
    assert np.isclose(float(outputs["place_box"][0, 6]), math.pi / 2)


def test_dense_placement_field_sparse_neck_preserves_active_voxel_shape() -> None:
    """spconv neck aggregates placement features without changing active voxel rows."""
    pytest.importorskip("spconv.pytorch")
    if not torch.cuda.is_available():
        pytest.skip("spconv sparse conv kernels require CUDA in this environment")

    device = torch.device("cuda")
    hidden_dim = 8
    field = SourceConditionedDensePlacementField(
        hidden_dim=hidden_dim,
        cfg={"dropout": 0.0, "neck_num_blocks": 1},
    ).to(device)
    batch_indices = torch.tensor([0, 0, 1, 1], device=device)
    sparse_coords = torch.tensor(
        [
            [0, 0, 0, 0],
            [0, 0, 0, 1],
            [1, 0, 0, 0],
            [1, 0, 1, 0],
        ],
        device=device,
    )

    outputs = field(
        voxel_features=torch.randn(4, hidden_dim, device=device),
        coords_norm=torch.rand(4, 3, device=device),
        sparse_coords=sparse_coords,
        spatial_shape=[1, 2, 2],
        batch_indices=batch_indices,
        batch_size=2,
        source_feature=torch.randn(2, hidden_dim, device=device),
        source_size_stage1=torch.ones(2, 3, device=device),
    )

    assert outputs["placement_features"].shape == (4, hidden_dim)
    assert outputs["placement_heatmap_logits"].shape == (4,)
    assert outputs["bottom_offset"].shape == (4, 3)
    assert outputs["yaw_sincos"].shape == (4, 2)
    assert outputs["size_residual"].shape == (2, 3)


def _loss_cfg(yaw_sensitive_ratio: float = 0.25) -> dict:
    return {
        "data": {"voxel_size_cm": 1.0, "heatmap_sigma_voxels": 2.0},
        "loss": {
            "lambda_heat": 1.0,
            "lambda_center": 1.0,
            "lambda_size": 1.0,
            "lambda_yaw": 0.5,
            "lambda_src": 0.2,
            "heatmap_loss": "focal",
            "yaw_sensitive_ratio": yaw_sensitive_ratio,
            "source": {"lambda_center": 1.0, "lambda_size": 1.0, "lambda_iou": 0.0},
        },
    }


def _minimal_loss_batch(place_box: torch.Tensor) -> dict[str, torch.Tensor]:
    return {
        "world_coords": torch.tensor([[1.0, 0.0, 0.0]]),
        "coords_norm": torch.tensor([[0.5, 0.5, 0.5]]),
        "batch_indices": torch.tensor([0]),
        "support_masks": torch.tensor([True]),
        "heatmap_positive_points": torch.tensor([[1.0, 0.0, 0.0]]),
        "heatmap_positive_batch_indices": torch.tensor([0]),
        "source_box_gt": torch.tensor([[0.0, 0.0, 0.0, 2.0, 2.0, 1.0]]),
        "place_box_gt": place_box,
        "batch_size": 1,
    }


def _minimal_loss_outputs(place_box: torch.Tensor, yaw_vec: torch.Tensor) -> dict[str, torch.Tensor]:
    return {
        "placement_heatmap_logits": torch.tensor([4.0]),
        "bottom_offset": torch.tensor([[0.0, 0.0, 0.0]]),
        "yaw_sincos": yaw_vec,
        "best_indices": torch.tensor([0]),
        "bottom_center": torch.tensor([[1.0, 0.0, 0.0]]),
        "size_pred": place_box[:, 3:6].clone(),
        "place_box": place_box.clone(),
        "source_box": torch.tensor([[0.0, 0.0, 0.0, 2.0, 2.0, 1.0]]),
    }


def test_stage2_yaw_loss_skips_near_square_boxes() -> None:
    """Near-square boxes should not contribute yaw loss."""
    place_box = torch.tensor([[1.0, 0.0, 1.0, 2.0, 2.0, 2.0, math.pi / 2]])
    outputs = _minimal_loss_outputs(place_box, torch.tensor([[0.0, 1.0]]))
    losses = compute_stage2_loss(outputs, _minimal_loss_batch(place_box), _loss_cfg())

    assert float(losses["loss_yaw"]) == 0.0


def test_stage2_yaw_loss_uses_direction_sensitive_boxes() -> None:
    """Elongated boxes should contribute yaw loss when prediction is wrong."""
    place_box = torch.tensor([[1.0, 0.0, 1.0, 4.0, 2.0, 2.0, math.pi / 2]])
    outputs = _minimal_loss_outputs(place_box, torch.tensor([[0.0, 1.0]]))
    losses = compute_stage2_loss(outputs, _minimal_loss_batch(place_box), _loss_cfg())

    assert float(losses["loss_yaw"]) > 0.0

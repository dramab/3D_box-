"""SPACE-Former Stage 2 data, sampling, matching and output tests."""

from __future__ import annotations

import json
import math
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from PIL import Image

import src.models.lc_bgplacenet.stage2 as stage2_model
from src.annotation.free_bbox.io_utils import save_center_yaw_set_npz, save_ply
from src.models.lc_bgplacenet.stage2 import (
    BoundedAnchorGenerator,
    GeometricRouting,
    NUM_QUERIES,
    NUM_SAMPLES,
    NUM_YAW_BINS,
    SPACEFormerDecoder,
    TextGuidedRegionPredictor,
    aggregate_sparse_mask,
    build_box_surface_template,
    build_cylinder_template,
    fold_yaw_mask_24_to_12,
    pose_nms,
    postprocess_pose_predictions,
    prepare_sparse_lookup_index,
    sparse_nearest_lookup,
)
from src.training.lc_bgplacenet_stage1 import (
    STAGE1_SUPPORTED_SPLIT_SCHEMA_VERSIONS,
    Stage1DataSource,
)
from src.training.lc_bgplacenet_stage2 import (
    LCBGPlaceNetStage2Dataset,
    _read_split_records,
    _build_lr_scheduler,
    _format_stage2_log,
    _hungarian_matches,
    _optimizer_learning_rates,
    _override_optimizer_learning_rates,
    build_space_former_targets,
    build_dense_heatmap_targets,
    build_stage2_index,
    compute_p3_gt_point_coverage,
    compute_stage2_loss,
    compute_stage2_task_metric_sums,
    stage2_collate,
)
import src.training.lc_bgplacenet_stage2 as stage2_training


def _write_json(path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2)


@pytest.mark.parametrize("schema_version", sorted(STAGE1_SUPPORTED_SPLIT_SCHEMA_VERSIONS))
def test_stage2_reads_supported_split_schemas(tmp_path, schema_version) -> None:
    split_dir = tmp_path / "splits"
    _write_json(
        split_dir / "train.json",
        {"schema_version": schema_version, "split": "train", "items": [{"item_id": "item_0"}]},
    )

    assert _read_split_records(split_dir, "train") == [{"item_id": "item_0"}]


def test_stage2_rejects_unknown_split_schema(tmp_path) -> None:
    split_dir = tmp_path / "splits"
    _write_json(
        split_dir / "train.json",
        {
            "schema_version": "lc_bgplacenet_stage1_splits/v999",
            "split": "train",
            "items": [],
        },
    )

    with pytest.raises(ValueError, match="Unsupported split schema_version"):
        _read_split_records(split_dir, "train")


def test_stage2_terminal_log_only_contains_core_metrics() -> None:
    train_log = _format_stage2_log({
        "split": "train", "epoch": 0, "step": 100, "loss": 42.859119,
        "loss_region": 0.019565, "loss_cls": 0.709644, "loss_center": 1.867732,
        "loss_yaw": 0.618746, "loss_corner": 4.383420, "source_iou": 0.684029,
        "p3_gt_point_coverage": 0.75, "lr_stage1": 1e-5, "lr_stage2": 1e-4,
        "layer0_p1_sample_total_count": 131072.0,
    })
    assert train_log == (
        "[train] epoch=0 step=100 loss=42.8591 region=0.0196 cls=0.7096 "
        "center=1.8677 yaw=0.6187 corner=4.3834 src_iou=0.6840 p3_gt_cov=0.7500 "
        "lr_s1=1.00e-05 lr_s2=1.00e-04"
    )
    assert "layer0" not in train_log

    valid_log = _format_stage2_log({
        "split": "valid", "epoch": 0, "step": 100, "loss": 3.0,
        "task_success_rate": 0.1, "task_success_top5": 0.5,
        "valid_pose_rate": 0.4, "direction_hit_rate": 0.8,
        "collision_free_rate": 0.9, "size_iou": 0.7,
        "p3_gt_point_coverage": 0.625, "lr_stage1": 5e-6, "lr_stage2": 5e-5,
        "sampling/layer0_p1_sample_total_count": 131072.0,
    })
    assert valid_log == (
        "[valid] epoch=0 step=100 loss=3.0000 task@1=0.1000 task@5=0.5000 "
        "valid_pose=0.4000 direction=0.8000 collision_free=0.9000 size_iou=0.7000 "
        "p3_gt_cov=0.6250 lr_s1=5.00e-06 lr_s2=5.00e-05"
    )


def test_task_success_top5_uses_only_task_conditions(monkeypatch) -> None:
    boxes = torch.zeros(1, 16, 7)
    boxes[0, :6, 0] = torch.arange(6, dtype=torch.float32)
    boxes[0, :6, 3:6] = 2.0
    valid_mask = torch.zeros(1, 16, dtype=torch.bool)
    valid_mask[0, :6] = True
    outputs = {
        "place_box": boxes[:, 0],
        "place_boxes": boxes,
        "place_valid_mask": valid_mask,
        "place_yaw_bins": torch.zeros(1, 16, dtype=torch.long),
        "source_box": torch.tensor([[0.0, 0.0, 0.0, 2.0, 2.0, 2.0]]),
    }
    batch = {
        "place_box_gt": torch.tensor([[0.0, 0.0, 0.0, 2.0, 2.0, 2.0, 0.0]]),
        # 中心和 yaw 故意不匹配，用于确认它们不参与 task success 判定。
        "gt_bottom_centers": torch.tensor([[[100.0, 100.0, 100.0]]]),
        "gt_yaw_masks": torch.zeros(1, 1, NUM_YAW_BINS, dtype=torch.bool),
        "gt_valid_mask": torch.ones(1, 1, dtype=torch.bool),
        "validation_contexts": [SimpleNamespace(collision_context={})],
        "source_box_gt": torch.tensor([[0.0, 0.0, 0.0, 2.0, 2.0, 2.0]]),
    }
    cfg = {"validation": {"size_iou_threshold": 0.8}}
    monkeypatch.setattr(stage2_training, "compute_collision_metrics", lambda box, context: {"collision": False})
    monkeypatch.setattr(stage2_training, "compute_direction_hit", lambda box, context: bool(box[0] == 1.0))

    metrics = compute_stage2_task_metric_sums(outputs, batch, cfg)

    assert metrics["task_success_sum"] == 0.0
    assert metrics["task_success_top5_sum"] == 1.0
    assert metrics["valid_pose_count"] == 1.0

    monkeypatch.setattr(stage2_training, "compute_direction_hit", lambda box, context: bool(box[0] == 5.0))
    metrics = compute_stage2_task_metric_sums(outputs, batch, cfg)
    assert metrics["task_success_top5_sum"] == 0.0


def _make_tiny_stage2_source(tmp_path) -> Stage1DataSource:
    dataset_dir = tmp_path / "data" / "toy"
    free_bbox_dir = tmp_path / "outputs" / "free_bbox_toy"
    labels_dir = tmp_path / "outputs" / "auto_labels_toy"
    sample_id = "toy__scene_0000__0000"
    stem = f"{sample_id}__obj_0__cluster_000"
    points = np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0]], dtype=np.float32)
    colors = np.array([[10, 20, 30], [40, 50, 60], [70, 80, 90]], dtype=np.uint8)
    save_ply(dataset_dir / "point_clouds_voxel_1cm" / f"{sample_id}.ply", points, colors)
    rgb_path = dataset_dir / "rgb" / f"{sample_id}.png"
    rgb_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.full((8, 8, 3), 128, dtype=np.uint8)).save(rgb_path)
    _write_json(
        dataset_dir / "samples" / f"{sample_id}.json",
        {
            "sample_id": sample_id,
            "rgb_path": f"rgb/{sample_id}.png",
            "voxel_point_cloud_path": f"point_clouds_voxel_1cm/{sample_id}.ply",
            "camera": {
                "fx": 1.0, "fy": 1.0, "cx": 0.0, "cy": 0.0, "img_w": 8, "img_h": 8,
                "E_c2w": np.eye(4).tolist(),
            },
            "objects": [{
                "obj_id": "obj_0",
                "class_name": "toy",
                "bbox3d_canonical": [-1, -2, -3, 1, 2, 3],
                "pose_world": np.eye(4).tolist(),
            }],
        },
    )
    save_ply(
        free_bbox_dir / "support_masks" / f"{sample_id}__support_mask.ply",
        points,
        np.array([[55, 55, 55], [255, 255, 255], [55, 55, 55]], dtype=np.uint8),
    )
    save_ply(
        free_bbox_dir / "direction_filtered_heatmaps" / f"{stem}__heatmap.ply",
        points,
        np.array([[60, 60, 60], [255, 0, 30], [55, 120, 210]], dtype=np.uint8),
    )
    yaw_mask = np.zeros((1, 24), dtype=bool)
    yaw_mask[0, [0, 12]] = True
    save_center_yaw_set_npz(
        free_bbox_dir / "yaw_sets" / f"{stem}__yaw_set.npz",
        bottom_center_voxels=np.array([[1, 0, 0]], dtype=np.int32),
        bottom_center_world=np.array([[1, 0, 0]], dtype=np.float32),
        valid_yaw_mask=yaw_mask,
        yaw_angles_rad=np.arange(24, dtype=np.float32) * (2 * math.pi / 24),
        heat_counts=np.array([1], dtype=np.int32),
    )
    placement = {
        "sample_id": f"{sample_id}_obj_0_cluster_000",
        "cluster_id": 0,
        "center_world": [1, 0, 1.5],
        "yaw_degrees": 0.0,
        "yaw_only_dimensions": [2, 4, 3],
        "bottom_center_world": [1, 0, 0],
        "heatmap_ply": f"heatmaps/{stem}__heatmap.ply",
        "visualization_png": f"visualizations/{stem}__vis.png",
    }
    _write_json(
        free_bbox_dir / "placements" / f"{sample_id}__placements.json",
        {
            "sample_id": sample_id,
            "voxel_point_cloud_path": f"point_clouds_voxel_1cm/{sample_id}.ply",
            "support_mask_ply": f"support_masks/{sample_id}__support_mask.ply",
            "objects": [{
                "object_id": "obj_0",
                "canonical_aabb_object": [-1, -2, -3, 1, 2, 3],
                "original_pose_world": np.eye(4).tolist(),
                "placements": [placement],
            }],
        },
    )
    visualization = free_bbox_dir / placement["visualization_png"]
    visualization.parent.mkdir(parents=True, exist_ok=True)
    visualization.touch()
    labels_path = labels_dir / "all_labels.json"
    _write_json(labels_path, [{
        "sample_id": sample_id,
        "object_id": "obj_0",
        "cluster_id": 0,
        "placement_sample_id": placement["sample_id"],
        "label": "Move toy object to the right of the block.",
        "spatial_relation": {"placement": {"relation": "the right of", "reference_object_id": "obj_0"}},
        "visualization_png": placement["visualization_png"],
    }])
    return Stage1DataSource("toy", dataset_dir, free_bbox_dir, labels_path)


def test_space_former_uses_48_queries() -> None:
    assert NUM_QUERIES == 48


def test_dataset_loads_multi_yaw_targets_and_pads_to_48(tmp_path) -> None:
    source = _make_tiny_stage2_source(tmp_path)
    items = build_stage2_index([source])
    assert len(items) == 1
    assert items[0].yaw_set_npz.exists()
    dataset = LCBGPlaceNetStage2Dataset([source], split="train", val_fraction=0.0)
    batch = stage2_collate([dataset[0]], voxel_size_cm=1.0)
    assert batch["gt_bottom_centers"].shape == (1, NUM_QUERIES, 3)
    assert batch["gt_yaw_masks"].shape == (1, NUM_QUERIES, NUM_YAW_BINS)
    assert batch["gt_valid_mask"].sum().item() == 1
    assert batch["gt_yaw_masks"][0, 0, 0]


def test_yaw_24_bins_fold_into_12_equivalent_bins() -> None:
    mask = torch.zeros(2, 24, dtype=torch.bool)
    mask[0, [3, 15]] = True
    mask[1, 20] = True
    folded = fold_yaw_mask_24_to_12(mask)
    assert folded.shape == (2, NUM_YAW_BINS)
    assert folded[0, 3]
    assert folded[1, 8]


def test_target_alignment_uses_canonical_voxel_keys(tmp_path) -> None:
    """同一 canonical voxel 内的 PLY 浮点差异不影响监督关联。"""
    yaw_set_path = tmp_path / "yaw_set.npz"
    yaw_mask = np.zeros((2, 24), dtype=bool)
    yaw_mask[0, 0] = True
    yaw_mask[1, 1] = True
    save_center_yaw_set_npz(
        yaw_set_path,
        bottom_center_voxels=np.array([[0, 0, 0], [1, 1, 1]], dtype=np.int32),
        bottom_center_world=np.array(
            [
                [1.0, 2.0, -0.0001001358],
                [18.629501, -107.704895, -0.0780983],
            ],
            dtype=np.float32,
        ),
        valid_yaw_mask=yaw_mask,
        yaw_angles_rad=np.arange(24, dtype=np.float32) * (2 * math.pi / 24),
        heat_counts=np.array([1, 1], dtype=np.int32),
    )

    targets = build_space_former_targets(
        yaw_set_path,
        np.array(
            [
                [1.0, 2.0, -0.0001],
                [18.629499, -107.704903, -0.0781],
            ],
            dtype=np.float32,
        ),
    )

    assert targets["gt_bottom_centers"].shape == (2, 3)
    assert targets["gt_yaw_masks"].sum() == 2


def test_physical_templates_have_exactly_64_points_and_two_types() -> None:
    size = torch.tensor([[20.0, 10.0, 8.0]])
    cylinder, cylinder_norm, cylinder_bottom = build_cylinder_template(size, voxel_size_cm=1.0)
    surface, surface_norm, surface_bottom = build_box_surface_template(size, voxel_size_cm=1.0)
    assert cylinder.shape == surface.shape == (1, NUM_SAMPLES, 3)
    assert cylinder_norm.shape == surface_norm.shape == (1, NUM_SAMPLES, 3)
    assert cylinder_bottom.sum().item() == 32
    assert surface_bottom.sum().item() == 16
    assert surface_norm[..., 0].amin() == -0.5
    assert surface_norm[..., 0].amax() == 0.5
    assert surface_norm[0, :16, 2].max() < -0.5


def test_top8_p3_cells_expand_only_matching_p1_and_pad_queries() -> None:
    hidden = 8
    p1_xyz = torch.stack([
        torch.arange(10) * 4,
        torch.zeros(10, dtype=torch.long),
        torch.zeros(10, dtype=torch.long),
    ], dim=1)
    p1 = {
        "features": torch.randn(len(p1_xyz), hidden),
        "coords": torch.cat([torch.zeros(len(p1_xyz), 1, dtype=torch.long), p1_xyz], dim=1),
    }
    p3_xyz = torch.stack([
        torch.arange(10),
        torch.zeros(10, dtype=torch.long),
        torch.zeros(10, dtype=torch.long),
    ], dim=1)
    p3 = {
        "features": torch.randn(len(p3_xyz), hidden),
        "coords": torch.cat([torch.zeros(len(p3_xyz), 1, dtype=torch.long), p3_xyz], dim=1),
    }
    outputs = BoundedAnchorGenerator(hidden, num_region_cells=8)(
        p1, p3, torch.arange(10.0, 0.0, -1.0), torch.zeros(1, 3), 1.0,
        torch.zeros(1, 3), torch.full((1, 3), 40.0), 1,
    )
    assert outputs["region_selected_cell_count"].item() == 8
    assert outputs["expanded_p1_candidate_count"].item() == 8
    assert outputs["query_valid_mask"].sum().item() == 8
    assert not outputs["query_valid_mask"][0, 8:].any()


def test_region_selection_uses_all_cells_when_fewer_than_eight() -> None:
    hidden = 8
    xyz = torch.stack([
        torch.arange(4),
        torch.zeros(4, dtype=torch.long),
        torch.zeros(4, dtype=torch.long),
    ], dim=1)
    p1 = {
        "features": torch.randn(4, hidden),
        "coords": torch.cat([torch.zeros(4, 1, dtype=torch.long), xyz * 4], dim=1),
    }
    p3 = {
        "features": torch.randn(4, hidden),
        "coords": torch.cat([torch.zeros(4, 1, dtype=torch.long), xyz], dim=1),
    }
    outputs = BoundedAnchorGenerator(hidden, num_region_cells=8)(
        p1, p3, torch.arange(4.0), torch.zeros(1, 3), 1.0,
        torch.zeros(1, 3), torch.full((1, 3), 20.0), 1,
    )
    assert outputs["region_selected_cell_count"].item() == 4
    assert outputs["query_valid_mask"].sum().item() == 4


def test_active_support_mask_aggregates_from_p1_to_p3_keys() -> None:
    p1_coords = torch.tensor([[0, 0, 0, 0], [0, 3, 0, 0], [0, 4, 0, 0]])
    p3_coords = torch.tensor([[0, 0, 0, 0], [0, 1, 0, 0]])

    mask = aggregate_sparse_mask(
        p1_coords,
        torch.tensor([False, True, False]),
        p3_coords,
        [2, 1, 1],
        stride=4,
    )

    assert mask.tolist() == [True, False]


def test_sparse_lookup_returns_active_mask_and_keeps_inactive_zero_tokens() -> None:
    level = {
        "features": torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
        "coords": torch.tensor([[0, 0, 0, 0], [0, 1, 0, 0]]),
        "stride": 1,
        "spatial_shape": [3, 3, 3],
    }
    points = torch.tensor([[[[0.9, 0.0, 0.0], [1.9, 0.0, 0.0], [2.0, 0.0, 0.0]]]])
    features, valid = sparse_nearest_lookup(level, points, torch.zeros(1, 3), 1.0, torch.ones(1, 1, dtype=torch.bool))
    assert valid.tolist() == [[[True, True, False]]]
    torch.testing.assert_close(features[0, 0, 2], torch.zeros(2))


def test_sparse_lookup_reuses_one_sorted_index() -> None:
    level = {
        "features": torch.tensor([[1.0], [2.0]]),
        "coords": torch.tensor([[0, 1, 0, 0], [0, 0, 0, 0]]),
        "stride": 1,
        "spatial_shape": [2, 1, 1],
    }
    prepare_sparse_lookup_index(level)
    first_order = level["lookup_order"]
    points = torch.tensor([[[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]]])
    first, first_valid = sparse_nearest_lookup(
        level, points, torch.zeros(1, 3), 1.0, torch.ones(1, 1, dtype=torch.bool)
    )
    second, second_valid = sparse_nearest_lookup(
        level, points, torch.zeros(1, 3), 1.0, torch.ones(1, 1, dtype=torch.bool)
    )
    assert level["lookup_order"] is first_order
    torch.testing.assert_close(first, second)
    torch.testing.assert_close(first_valid, second_valid)


def test_p3_gt_point_coverage_is_macro_average_and_handles_short_topk() -> None:
    region_coords = torch.tensor([
        [0, 0, 0, 0], [0, 1, 0, 0],
        [1, 0, 0, 0], [1, 2, 0, 0],
    ])
    region_logits = torch.tensor([5.0, 1.0, 1.0, 5.0])
    positives = torch.tensor([
        [0.1, 0.0, 0.0], [4.1, 0.0, 0.0], [8.1, 0.0, 0.0],
    ])
    positive_batches = torch.tensor([0, 0, 1])
    coverage = compute_p3_gt_point_coverage(
        region_coords,
        region_logits,
        torch.zeros(2, 3),
        positives,
        positive_batches,
        voxel_size_cm=1.0,
        batch_size=2,
        num_region_cells=1,
    )
    torch.testing.assert_close(coverage, torch.tensor(0.75))

    all_cells_coverage = compute_p3_gt_point_coverage(
        region_coords,
        region_logits,
        torch.zeros(2, 3),
        positives,
        positive_batches,
        voxel_size_cm=1.0,
        batch_size=2,
        num_region_cells=8,
    )
    torch.testing.assert_close(all_cells_coverage, torch.tensor(1.0))


def test_region_predictor_uses_all_valid_text_tokens_and_masks_padding() -> None:
    predictor = TextGuidedRegionPredictor(hidden_dim=8, num_heads=2, dropout=0.0, num_layers=4)
    assert len(predictor.blocks) == 4
    p3_features = torch.randn(2, 8)
    text_tokens = torch.randn(1, 3, 8, requires_grad=True)
    logits, _ = predictor(
        p3_features,
        torch.zeros(2, dtype=torch.long),
        text_tokens,
        torch.tensor([[True, True, False]]),
        batch_size=1,
    )

    logits.sum().backward()

    assert torch.count_nonzero(text_tokens.grad[0, 0]) > 0
    assert torch.count_nonzero(text_tokens.grad[0, 1]) > 0
    torch.testing.assert_close(text_tokens.grad[0, 2], torch.zeros(8))


def test_region_predictor_rejects_non_positive_num_layers() -> None:
    with pytest.raises(ValueError, match="num_layers must be positive"):
        TextGuidedRegionPredictor(hidden_dim=8, num_heads=2, dropout=0.0, num_layers=0)


def test_region_gaussian_target_uses_direction_filtered_positive_points() -> None:
    targets = build_dense_heatmap_targets(
        world_coords=torch.tensor([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [4.0, 0.0, 0.0]]),
        batch_indices=torch.zeros(3, dtype=torch.long),
        positive_points=torch.tensor([[0.0, 0.0, 0.0]]),
        positive_batch_indices=torch.zeros(1, dtype=torch.long),
        support_masks=torch.ones(3, dtype=torch.bool),
        batch_size=1,
        sigma=2.0,
    )

    torch.testing.assert_close(targets, torch.exp(torch.tensor([0.0, -0.5, -2.0])))


def test_region_gaussian_target_is_zero_outside_active_support() -> None:
    targets = build_dense_heatmap_targets(
        world_coords=torch.tensor([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]]),
        batch_indices=torch.zeros(2, dtype=torch.long),
        positive_points=torch.tensor([[0.0, 0.0, 0.0]]),
        positive_batch_indices=torch.zeros(1, dtype=torch.long),
        support_masks=torch.tensor([True, False]),
        batch_size=1,
        sigma=2.0,
    )

    torch.testing.assert_close(targets, torch.tensor([1.0, 0.0]))


def test_sample_token_is_261_dims_without_scale_or_occupancy_embedding() -> None:
    routing = GeometricRouting(hidden_dim=256, num_heads=8, dropout=0.0)
    for projector in routing.sample_projectors:
        assert projector[0].in_features == 261
    assert not hasattr(routing, "scale_embedding")
    assert not hasattr(routing, "sample_occupancy_ratio")


def test_decoder_logs_64_samples_per_valid_query_layer_and_scale() -> None:
    hidden = 8
    decoder = SPACEFormerDecoder(hidden, num_heads=2, dropout=0.0, num_layers=4)
    query = torch.randn(1, 2, hidden)
    valid_query = torch.tensor([[True, False]])
    pyramid = [
        {
            "features": torch.ones(1, hidden),
            "coords": torch.tensor([[0, 0, 0, 0]]),
            "stride": stride,
            "spatial_shape": [32, 32, 32],
        }
        for stride in (1, 2, 4)
    ]
    layer_outputs, logs = decoder(
        query=query,
        bottom_centers=torch.zeros(1, 2, 3),
        query_valid_mask=valid_query,
        pyramid=pyramid,
        voxel_origins=torch.zeros(1, 3),
        voxel_size_cm=1.0,
        source_feature=torch.randn(1, hidden),
        source_size=torch.tensor([[4.0, 4.0, 4.0]]),
        text_tokens=torch.randn(1, 3, hidden),
        text_attention_mask=torch.ones(1, 3, dtype=torch.bool),
    )
    assert len(layer_outputs) == 4
    for layer in range(4):
        for scale in range(1, 4):
            assert logs[f"layer{layer}_p{scale}_sample_total_count"].item() == NUM_SAMPLES
    assert all(output["pred_logits"][0, 1].item() == -20.0 for output in layer_outputs)
    final = layer_outputs[-1]
    loss = (
        final["pred_bottom_centers"].sum()
        + final["pred_yaw_logits"].sum()
        + final["pred_logits"].sum()
    )
    loss.backward()
    assert all(parameter.grad is not None for parameter in decoder.layers[-1].fusion.parameters())


def test_pose_nms_outputs_at_most_16_and_preserves_source_size() -> None:
    raw_boxes = torch.zeros(1, NUM_QUERIES, 7)
    raw_boxes[..., :3] = torch.arange(NUM_QUERIES, dtype=torch.float32)[None, :, None]
    raw_boxes[..., 3:6] = torch.tensor([2.0, 4.0, 3.0])
    outputs = pose_nms(
        raw_boxes,
        torch.linspace(3.0, 1.0, NUM_QUERIES)[None],
        torch.zeros(1, NUM_QUERIES, NUM_YAW_BINS),
        torch.ones(1, NUM_QUERIES, dtype=torch.bool),
    )
    assert outputs["place_boxes"].shape == (1, 16, 7)
    assert outputs["place_box"].shape == (1, 7)
    valid = outputs["place_valid_mask"][0]
    torch.testing.assert_close(outputs["place_boxes"][0, valid, 3:6], torch.tensor([2.0, 4.0, 3.0]).expand(valid.sum(), -1))


def test_training_skips_pose_nms_and_eval_keeps_placement_fields(monkeypatch) -> None:
    raw_boxes = torch.zeros(1, 2, 7)
    raw_boxes[..., 3:6] = 2.0
    placement_logits = torch.zeros(1, 2)
    yaw_logits = torch.zeros(1, 2, NUM_YAW_BINS)
    valid_mask = torch.ones(1, 2, dtype=torch.bool)

    def fail_if_called(*args, **kwargs):
        raise AssertionError("pose_nms must not run in training mode")

    original_pose_nms = stage2_model.pose_nms
    monkeypatch.setattr(stage2_model, "pose_nms", fail_if_called)
    assert postprocess_pose_predictions(
        raw_boxes, placement_logits, yaw_logits, valid_mask, training=True
    ) == {}
    monkeypatch.setattr(stage2_model, "pose_nms", original_pose_nms)
    eval_outputs = postprocess_pose_predictions(
        raw_boxes, placement_logits, yaw_logits, valid_mask, training=False
    )
    assert {
        "place_boxes", "place_scores", "place_yaw_bins", "place_valid_mask", "place_box"
    }.issubset(eval_outputs)


def test_space_former_loss_backpropagates_to_source_prediction() -> None:
    source_box = torch.tensor([[0.0, 0.0, 1.0, 2.0, 4.0, 2.0]], requires_grad=True)
    raw_logits = torch.zeros(1, 2, requires_grad=True)
    raw_yaw = torch.zeros(1, 2, NUM_YAW_BINS, requires_grad=True)
    centers = torch.tensor([[[1.0, 0.0, 0.0], [3.0, 0.0, 0.0]]], requires_grad=True)
    outputs = {
        "region_sparse_coords": torch.tensor([[0, 0, 0, 0], [0, 1, 0, 0]]),
        "region_world_coords": torch.tensor([[0.0, 0.0, 0.0], [4.0, 0.0, 0.0]]),
        "region_batch_indices": torch.tensor([0, 0]),
        "region_logits": torch.zeros(2, requires_grad=True),
        "region_spatial_shape": [2, 1, 1],
        "voxel_origins": torch.zeros(1, 3),
        "query_valid_mask": torch.tensor([[True, True]]),
        "raw_place_logits": raw_logits,
        "raw_yaw_logits": raw_yaw,
        "pred_bottom_centers": centers,
        "source_box": source_box,
        "decoder_aux_outputs": [],
        "sampling_logs": {},
        "region_selected_cell_count": torch.tensor([2]),
        "expanded_p1_candidate_count": torch.tensor([2]),
    }
    yaw_target = torch.zeros(1, 2, NUM_YAW_BINS, dtype=torch.bool)
    yaw_target[0, 0, 0] = True
    batch = {
        "batch_size": 1,
        "sparse_coords": torch.tensor([[0, 0, 0, 0], [0, 4, 0, 0]]),
        "support_masks": torch.tensor([True, False]),
        "heatmap_positive_points": torch.tensor([[0.0, 0.0, 0.0]]),
        "heatmap_positive_batch_indices": torch.tensor([0]),
        "gt_bottom_centers": torch.tensor([[[1.0, 0.0, 0.0], [0.0, 0.0, 0.0]]]),
        "gt_yaw_masks": yaw_target,
        "gt_affordance_quality": torch.tensor([[1.0, 0.0]]),
        "gt_valid_mask": torch.tensor([[True, False]]),
        "source_box_gt": torch.tensor([[0.0, 0.0, 1.0, 2.0, 4.0, 2.0]]),
    }
    cfg = {
        "data": {"voxel_size_cm": 1.0},
        "loss": {
            "lambda_region": 1.0, "lambda_cls": 2.0, "lambda_center": 5.0,
            "lambda_yaw": 2.0, "lambda_corner": 1.0, "lambda_src": 1.0, "lambda_aux": 0.5,
            "focal": {"alpha": 0.25, "gamma": 2.0},
            "source": {"lambda_center": 2.0, "lambda_size": 1.5, "lambda_iou": 0.2},
        },
    }
    losses = compute_stage2_loss(outputs, batch, cfg)
    losses["loss"].backward()
    assert source_box.grad is not None
    assert torch.isfinite(source_box.grad).all()


def test_batched_hungarian_preserves_expected_assignment_indices() -> None:
    yaw_logits = torch.full((2, 2, NUM_YAW_BINS), -10.0)
    yaw_logits[..., 0] = 10.0
    outputs = {
        "query_valid_mask": torch.tensor([[True, True], [False, False]]),
        "raw_place_logits": torch.full((2, 2), 5.0),
        "pred_bottom_centers": torch.tensor([
            [[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]],
            [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
        ]),
        "raw_yaw_logits": yaw_logits,
        "source_box": torch.tensor([
            [0.0, 0.0, 1.0, 2.0, 2.0, 2.0],
            [0.0, 0.0, 1.0, 2.0, 2.0, 2.0],
        ]),
    }
    yaw_masks = torch.zeros(2, 2, NUM_YAW_BINS, dtype=torch.bool)
    yaw_masks[..., 0] = True
    batch = {
        "batch_size": 2,
        "gt_valid_mask": torch.tensor([[True, True], [False, False]]),
        "gt_affordance_quality": torch.ones(2, 2),
        "gt_bottom_centers": torch.tensor([
            [[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]],
            [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
        ]),
        "gt_yaw_masks": yaw_masks,
        "source_box_gt": outputs["source_box"].clone(),
    }
    matches = _hungarian_matches(outputs, batch)
    assert matches[0][0].tolist() == [0, 1]
    assert matches[0][1].tolist() == [0, 1]
    assert matches[1][0].numel() == matches[1][1].numel() == 0


def test_resume_lr_override_preserves_adamw_moments_and_plateau_halves_lr() -> None:
    old_parameters = [torch.nn.Parameter(torch.tensor([1.0])), torch.nn.Parameter(torch.tensor([2.0]))]
    old_optimizer = torch.optim.AdamW([
        {"params": [old_parameters[0]], "lr": 1e-6},
        {"params": [old_parameters[1]], "lr": 1e-5},
    ])
    sum(parameter.sum() for parameter in old_parameters).backward()
    old_optimizer.step()
    saved_state = old_optimizer.state_dict()
    expected_moment = old_optimizer.state[old_parameters[1]]["exp_avg"].clone()

    parameters = [torch.nn.Parameter(torch.tensor([1.0])), torch.nn.Parameter(torch.tensor([2.0]))]
    optimizer = torch.optim.AdamW([
        {"params": [parameters[0]], "lr": 1e-7},
        {"params": [parameters[1]], "lr": 1e-6},
    ])
    optimizer.load_state_dict(saved_state)
    _override_optimizer_learning_rates(optimizer, base_lr=1e-4)
    assert _optimizer_learning_rates(optimizer) == {"lr_stage1": 1e-5, "lr_stage2": 1e-4}
    torch.testing.assert_close(optimizer.state[parameters[1]]["exp_avg"], expected_moment)

    cfg = {
        "training": {
            "lr_scheduler": {
                "type": "reduce_on_plateau",
                "factor": 0.5,
                "patience": 3,
                "threshold": 0.001,
            }
        }
    }
    scheduler = _build_lr_scheduler(optimizer, cfg)
    scheduler.step(0.5)
    for _ in range(4):
        scheduler.step(0.5)
    assert _optimizer_learning_rates(optimizer) == {"lr_stage1": 5e-6, "lr_stage2": 5e-5}

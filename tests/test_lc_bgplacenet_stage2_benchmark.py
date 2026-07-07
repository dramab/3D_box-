"""LC-BGPlaceNet Stage 2 benchmark metric tests."""

from __future__ import annotations

import numpy as np

from src.annotation.free_bbox.io_utils import load_ply, save_ply
from tools.benchmark_lc_bgplacenet_stage2 import (
    build_collision_context,
    compute_collision_metrics,
    compute_direction_metrics,
    compute_size_metrics,
)


def test_size_iou_keeps_dimension_order_and_threshold() -> None:
    """Size IoU should compare w/h/l by corresponding dimensions."""
    pred_box = np.array([0.0, 0.0, 0.0, 2.0, 4.0, 6.0, 0.0], dtype=np.float64)
    gt_box = np.array([0.0, 0.0, 0.0, 2.0, 2.0, 6.0, 0.0], dtype=np.float64)

    strict = compute_size_metrics(pred_box, gt_box, threshold=0.8)
    loose = compute_size_metrics(pred_box, gt_box, threshold=0.5)

    assert strict["size_iou"] == 0.5
    assert strict["size_correct"] is False
    assert loose["size_correct"] is True


def test_direction_metric_uses_predicted_bottom_center(tmp_path) -> None:
    """Predicted bottom center should be matched against positive heatmap voxels."""
    heatmap_path = tmp_path / "direction_filtered_heatmap.ply"
    heatmap_points = np.array(
        [
            [1.0, 2.0, 0.0],
            [4.0, 5.0, 0.0],
        ],
        dtype=np.float32,
    )
    heatmap_colors = np.array(
        [
            [255, 0, 30],
            [55, 120, 210],
        ],
        dtype=np.uint8,
    )
    save_ply(heatmap_path, heatmap_points, heatmap_colors)
    loaded_points, loaded_colors = load_ply(heatmap_path)

    place_box = np.array([1.2, 2.1, 5.0, 2.0, 4.0, 10.0, 0.0], dtype=np.float64)
    metrics = compute_direction_metrics(place_box, loaded_points, loaded_colors, voxel_size_cm=1.0)

    assert metrics["direction_hit"] is True
    assert metrics["direction_positive_count"] == 1
    assert metrics["nearest_direction_distance_cm"] < 0.25


def test_collision_metric_ignores_support_voxels_but_counts_other_occupied() -> None:
    """Support-surface voxels are removed from obstacles; other occupied voxels still collide."""
    support_only_context = build_collision_context(
        voxel_points=np.array([[0.0, 0.0, 0.0]], dtype=np.float64),
        support_points=np.array([[0.0, 0.0, 0.0]], dtype=np.float64),
        extra_points=np.array([[-0.5, -0.5, 0.0], [0.5, 0.5, 1.0]], dtype=np.float64),
        voxel_size_cm=1.0,
    )
    place_box = np.array([0.0, 0.0, 0.5, 1.0, 1.0, 1.0, 0.0], dtype=np.float64)
    support_metrics = compute_collision_metrics(place_box, support_only_context)

    obstacle_context = build_collision_context(
        voxel_points=np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64),
        support_points=np.array([[0.0, 0.0, 0.0]], dtype=np.float64),
        extra_points=np.array([[-0.5, -0.5, 0.0], [0.5, 0.5, 1.0]], dtype=np.float64),
        voxel_size_cm=1.0,
    )
    obstacle_metrics = compute_collision_metrics(place_box, obstacle_context)

    assert support_metrics["collision"] is False
    assert support_metrics["collision_voxel_count"] == 0
    assert obstacle_metrics["collision"] is True
    assert obstacle_metrics["collision_voxel_count"] > 0


def test_collision_metric_uses_point_centers_not_conservative_voxel_intersection() -> None:
    """A point below the box bottom should not collide just because its voxel cell touches."""
    context = build_collision_context(
        voxel_points=np.array([[0.0, 0.0, -0.25]], dtype=np.float64),
        support_points=np.empty((0, 3), dtype=np.float64),
        extra_points=np.array([[-0.5, -0.5, 0.0], [0.5, 0.5, 1.0]], dtype=np.float64),
        voxel_size_cm=1.0,
    )
    place_box = np.array([0.0, 0.0, 0.5, 1.0, 1.0, 1.0, 0.0], dtype=np.float64)
    metrics = compute_collision_metrics(place_box, context)

    assert metrics["box_voxel_count"] > 0
    assert metrics["collision"] is False
    assert metrics["collision_voxel_count"] == 0

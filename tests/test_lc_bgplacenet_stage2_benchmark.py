"""LC-BGPlaceNet Stage 2 benchmark metric tests."""

from __future__ import annotations

import numpy as np

from src.datasets.canonical import CameraParams, ObjectInfo
from tools.benchmark_lc_bgplacenet_stage2 import (
    build_collision_context,
    compute_collision_metrics,
    compute_direction_metrics,
    parse_target_direction,
    compute_size_metrics,
    summarize_rows,
)
from src.placement_metrics import placement_success


def test_size_iou_keeps_dimension_order_and_threshold() -> None:
    """Size IoU should compare w/h/l by corresponding dimensions."""
    pred_box = np.array([0.0, 0.0, 0.0, 2.0, 4.0, 6.0, 0.0], dtype=np.float64)
    gt_box = np.array([0.0, 0.0, 0.0, 2.0, 2.0, 6.0, 0.0], dtype=np.float64)

    strict = compute_size_metrics(pred_box, gt_box, threshold=0.8)
    loose = compute_size_metrics(pred_box, gt_box, threshold=0.5)

    assert strict["size_iou"] == 0.5
    assert strict["size_correct"] is False
    assert loose["size_correct"] is True


def test_parse_target_direction_handles_relation_without_of() -> None:
    """Instruction parsing should support target relations like behind."""
    relation, reference = parse_target_direction(
        "Move Tomato Can located at the left of Bottle to behind Chocolate Cookie Box."
    )

    assert relation == "behind"
    assert reference == "Chocolate Cookie Box"


def test_direction_metric_uses_auto_label_relation() -> None:
    """Predicted box corners should be checked with auto-label spatial rules."""
    camera = CameraParams(
        fx=1.0,
        fy=1.0,
        cx=0.0,
        cy=0.0,
        E_c2w=np.eye(4, dtype=np.float64),
        img_w=640,
        img_h=480,
    )
    scene_context = {
        "camera": camera,
        "object_by_id": {
            "obj_ref": ObjectInfo(
                obj_id="obj_ref",
                class_name="Reference",
                bbox3d_canonical=np.array([-1.0, -1.0, -1.0, 1.0, 1.0, 1.0], dtype=np.float64),
                pose_world=np.eye(4, dtype=np.float64),
            )
        },
    }
    metadata = {
        "target_relation": "the top of",
        "reference_object_id": "obj_ref",
        "reference_name": "Reference",
    }
    place_box = np.array([0.0, 0.0, 3.0, 2.0, 2.0, 2.0, 0.0], dtype=np.float64)
    metrics = compute_direction_metrics(place_box, metadata, scene_context)

    assert metrics["direction_hit"] is True
    assert metrics["predicted_relation"] == "the top of"


def _scene_object(object_id: str, center: tuple[float, float, float]) -> dict:
    """Build a unit test scene object with a 2cm cube OBB."""
    pose = np.eye(4, dtype=np.float64)
    pose[:3, 3] = np.asarray(center, dtype=np.float64)
    return {
        "object_id": object_id,
        "canonical_aabb_object": [-1.0, -1.0, -1.0, 1.0, 1.0, 1.0],
        "original_pose_world": pose.tolist(),
    }


def test_collision_metric_counts_intersecting_scene_objects() -> None:
    """Predicted boxes should collide with all intersecting scene object boxes."""
    context = build_collision_context(
        [
            _scene_object("obj_0", (0.5, 0.0, 0.0)),
            _scene_object("obj_1", (4.0, 0.0, 0.0)),
        ]
    )
    place_box = np.array([0.0, 0.0, 0.0, 2.0, 2.0, 2.0, 0.0], dtype=np.float64)
    metrics = compute_collision_metrics(place_box, context)

    assert metrics["collision"] is True
    assert metrics["collision_object_count"] == 1
    assert metrics["collision_object_ids"] == ["obj_0"]


def test_collision_metric_does_not_exclude_current_object_id() -> None:
    """The benchmark counts any scene object box intersection, including obj_0."""
    context = build_collision_context([_scene_object("obj_0", (0.0, 0.0, 0.0))])
    place_box = np.array([0.0, 0.0, 0.0, 2.0, 2.0, 2.0, 0.0], dtype=np.float64)
    metrics = compute_collision_metrics(place_box, context)

    assert metrics["collision"] is True
    assert metrics["collision_object_ids"] == ["obj_0"]


def test_collision_metric_treats_box_contact_as_non_collision() -> None:
    """Touching faces should not count as positive-volume box collision."""
    context = build_collision_context([_scene_object("obj_touching", (2.0, 0.0, 0.0))])
    place_box = np.array([0.0, 0.0, 0.0, 2.0, 2.0, 2.0, 0.0], dtype=np.float64)
    metrics = compute_collision_metrics(place_box, context)

    assert metrics["collision"] is False
    assert metrics["collision_object_count"] == 0
    assert metrics["collision_object_ids"] == []


def test_placement_success_requires_all_four_conditions() -> None:
    assert placement_success(True, True, True, True)
    assert not placement_success(True, True, False, True)


def test_benchmark_summary_reports_placement_and_conditional_yaw() -> None:
    """Benchmark should keep source/yaw separate from Placement Success@K."""
    rows = [
        {
            "source_iou": 0.9,
            "source_iou_correct": True,
            "placement_size_iou": 0.9,
            "placement_size_correct": True,
            "language_relation_correct": True,
            "supported_and_stable": True,
            "collision_free": True,
            "center_matched": True,
            "yaw_valid_at_matched_center": True,
            "placement_success_at_1": True,
            "placement_success_at_5": True,
        },
        {
            "source_iou": 0.4,
            "source_iou_correct": False,
            "placement_size_iou": 0.8,
            "placement_size_correct": True,
            "language_relation_correct": True,
            "supported_and_stable": True,
            "collision_free": True,
            "center_matched": False,
            "yaw_valid_at_matched_center": False,
            "placement_success_at_1": True,
            "placement_success_at_5": True,
        },
    ]

    summary = summarize_rows(rows)

    assert summary["source_iou_accuracy"] == 0.5
    assert summary["center_match_rate"] == 0.5
    assert summary["yaw_valid_given_center_match"] == 1.0
    assert summary["placement_success_at_1"] == 1.0
    assert summary["placement_success_at_5"] == 1.0

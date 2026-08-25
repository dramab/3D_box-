from __future__ import annotations

import numpy as np

from src.placement_metrics import (
    compute_supported_and_stable,
    compute_yaw_valid_at_matched_center,
)

from tools.analyze_support_coverage_metric import (
    build_occupancy_columns,
    coverage_curve,
    footprint_voxel_keys,
)
from tools.analyze_connected_support_metric import (
    build_connected_region,
    component_coverage,
)


def test_axis_aligned_footprint_contains_expected_center_cells() -> None:
    box = np.array([0.0, 0.0, 2.0, 2.0, 2.0, 4.0, 0.0], dtype=np.float64)
    keys = footprint_voxel_keys(box, voxel_size_cm=1.0)
    key_set = {tuple(key) for key in keys.tolist()}
    assert key_set == {(-1, -1), (-1, 0), (0, -1), (0, 0)}


def test_coverage_increases_when_downward_band_reaches_support() -> None:
    box = np.array([0.0, 0.0, 3.5, 2.0, 2.0, 1.0, 0.0], dtype=np.float64)
    footprint = footprint_voxel_keys(box, voxel_size_cm=1.0)
    support_points = np.column_stack(
        [
            (footprint[:, 0] + 0.5),
            (footprint[:, 1] + 0.5),
            np.full(len(footprint), 0.5),
        ]
    )
    occupancy = build_occupancy_columns(support_points, voxel_size_cm=1.0)
    coverage, _, _ = coverage_curve(
        box,
        occupancy,
        depths_cm=np.array([2.0, 3.0]),
        upper_tolerance_cm=0.0,
    )
    np.testing.assert_allclose(coverage, [0.0, 1.0])


def test_upper_tolerance_accepts_one_voxel_above_bottom() -> None:
    box = np.array([0.0, 0.0, 1.5, 2.0, 2.0, 1.0, 0.0], dtype=np.float64)
    footprint = footprint_voxel_keys(box, voxel_size_cm=1.0)
    points = np.column_stack(
        [
            (footprint[:, 0] + 0.5),
            (footprint[:, 1] + 0.5),
            np.full(len(footprint), 1.5),
        ]
    )
    occupancy = build_occupancy_columns(points, voxel_size_cm=1.0)
    without_upper, _, _ = coverage_curve(
        box, occupancy, np.array([1.0]), upper_tolerance_cm=0.0
    )
    with_upper, _, _ = coverage_curve(
        box, occupancy, np.array([1.0]), upper_tolerance_cm=1.0
    )
    np.testing.assert_allclose(without_upper, [0.0])
    np.testing.assert_allclose(with_upper, [1.0])


def test_connected_region_fills_an_enclosed_center_hole() -> None:
    xy = np.array(
        [
            [0, 0], [0, 1], [0, 2],
            [1, 0],         [1, 2],
            [2, 0], [2, 1], [2, 2],
        ],
        dtype=np.int64,
    )
    occupied = np.column_stack([xy, np.zeros(len(xy), dtype=np.int64)])
    box = np.array([1.5, 1.5, 1.0, 1.0, 1.0, 1.0, 0.0], dtype=np.float64)
    coverage, _, _, _ = component_coverage(
        box,
        occupied,
        voxel_size_cm=1.0,
        downward_cm=1.0,
        upper_cm=1.0,
        apply_closing=False,
    )
    assert coverage == 1.0


def test_connected_region_does_not_fill_a_hole_open_to_the_exterior() -> None:
    xy = np.array(
        [[0, 0], [0, 1], [0, 2], [1, 0], [1, 2], [2, 0], [2, 2]],
        dtype=np.int64,
    )
    occupied = np.column_stack([xy, np.zeros(len(xy), dtype=np.int64)])
    box = np.array([1.5, 1.5, 1.0, 1.0, 1.0, 1.0, 0.0], dtype=np.float64)
    coverage, _, _, _ = component_coverage(
        box,
        occupied,
        voxel_size_cm=1.0,
        downward_cm=1.0,
        upper_cm=1.0,
        apply_closing=False,
    )
    assert coverage == 0.0


def test_closing_connects_a_one_voxel_break() -> None:
    left_x, left_y = np.meshgrid(np.arange(0, 3), np.arange(0, 5), indexing="ij")
    right_x, right_y = np.meshgrid(np.arange(4, 7), np.arange(0, 5), indexing="ij")
    xy = np.vstack(
        [
            np.column_stack([left_x.ravel(), left_y.ravel()]),
            np.column_stack([right_x.ravel(), right_y.ravel()]),
        ]
    )
    occupied = np.column_stack([xy, np.zeros(len(xy), dtype=np.int64)])
    without_closing = build_connected_region(occupied, 0, 0, apply_closing=False)
    with_closing = build_connected_region(occupied, 0, 0, apply_closing=True)
    assert without_closing.labels.max() == 2
    assert with_closing.labels.max() == 1


def test_production_support_requires_full_single_component_coverage() -> None:
    xy = np.array([[x, y] for x in range(4) for y in range(4)], dtype=np.int64)
    occupied = np.column_stack([xy, np.zeros(len(xy), dtype=np.int64)])
    supported_box = np.array([2.0, 2.0, 1.0, 2.0, 2.0, 2.0, 0.0])
    overhanging_box = supported_box.copy()
    overhanging_box[0] = 3.5

    supported, coverage = compute_supported_and_stable(
        supported_box, occupied, voxel_size_cm=1.0, downward_cm=3.0, upper_cm=1.0
    )
    overhanging, overhanging_coverage = compute_supported_and_stable(
        overhanging_box, occupied, voxel_size_cm=1.0, downward_cm=3.0, upper_cm=1.0
    )

    assert supported and coverage == 1.0
    assert not overhanging and overhanging_coverage < 1.0


def test_yaw_must_be_valid_at_the_nearest_matched_center() -> None:
    box = np.array([0.5, 0.5, 2.5, 2.0, 2.0, 4.0, 0.0])
    centers = np.array([[0.5, 0.5, 0.5], [5.5, 0.5, 0.5]])
    masks = np.zeros((2, 12), dtype=bool)
    masks[0, 3] = True

    valid, distance, matched = compute_yaw_valid_at_matched_center(
        box, 3, centers, masks, center_match_threshold_cm=2.0
    )
    invalid, _, _ = compute_yaw_valid_at_matched_center(
        box, 4, centers, masks, center_match_threshold_cm=2.0
    )

    assert valid and distance == 0.0 and matched == 0
    assert not invalid

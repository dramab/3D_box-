from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import tools.export_method_overview_materials as exporter
from tools.export_method_overview_materials import (
    MATERIAL_SPECS,
    bottom_surface_sample_points,
    boundary_sample_points,
    gaussian_expand_heatmap_scores,
    material_filename,
    obb_surface_sample_points,
    points_inside_yaw_box,
    side_boundary_sample_groups,
    voxel_probability_colors,
)


def test_material_specs_export_fifteen_stable_images() -> None:
    assert len(MATERIAL_SPECS) == 15
    names = [material_filename(index, spec) for index, spec in enumerate(MATERIAL_SPECS, start=1)]

    assert names[0] == "01_language_instruction.png"
    assert names[-1] == "15_collision_free_placement.png"
    assert len(set(names)) == len(names)


def test_find_item_selects_exact_cluster(monkeypatch):
    items = [SimpleNamespace(sample_id="scene", object_id="obj_1", cluster_id=c,
                             reference_object_id="obj_2", target_relation="left")
             for c in (1, 2)]
    monkeypatch.setattr(exporter, "build_sources_from_config", lambda cfg: [])
    monkeypatch.setattr(exporter, "build_stage2_index", lambda sources: items)
    assert exporter.find_item({}, "scene", "obj_1", None, None, 2) is items[1]
    with pytest.raises(ValueError):
        exporter.find_item({}, "scene", "obj_1", None, None, 3)


def test_semantic_probe_prefers_opposite_position_and_checks_physics(monkeypatch, tmp_path):
    yaw_path = tmp_path / "yaws.npz"
    np.savez(yaw_path, yaw_angles_rad=np.array([0.0, np.pi / 2]))
    item = SimpleNamespace(
        sample_id="test", target_relation="right", yaw_set_npz=yaw_path,
        place_box_gt=np.array([5., 0., 1., 2., 4., 2., 0.]),
    )
    centers = np.array([[1., 4., 0.], [-5., 0., 0.]])
    monkeypatch.setattr(exporter, "support_topology", lambda *args: (centers, None, 1., None))
    monkeypatch.setattr(exporter, "quantize_occupied_points", lambda *args: {})
    monkeypatch.setattr(exporter, "project_world_points", lambda *args: (None, np.ones(8, dtype=bool)))
    monkeypatch.setattr(exporter, "placement_relation", lambda *args: "left")
    monkeypatch.setattr(exporter, "is_collision_free", lambda box, ctx: box[6] > 0)
    monkeypatch.setattr(exporter, "compute_supported_and_stable", lambda *args: (True, 1.))
    kwargs = dict(voxel_size_cm=1., downward_cm=3., upper_cm=1.)
    box = exporter.choose_semantic_wrong_box(item, None, np.zeros((8, 3)), centers, {}, **kwargs)
    np.testing.assert_allclose(box[:2], [-5., 0.])
    np.testing.assert_allclose(box[3:6], item.place_box_gt[3:6])
    assert box[6] == np.pi / 2
    monkeypatch.setattr(exporter, "compute_supported_and_stable", lambda *args: (False, .5))
    with pytest.raises(ValueError, match="No supported, collision-free"):
        exporter.choose_semantic_wrong_box(item, None, np.zeros((8, 3)), centers, {}, **kwargs)


def test_box_sampling_helpers_return_world_points() -> None:
    box = np.asarray([10.0, 20.0, 3.0, 6.0, 4.0, 2.0, np.pi / 2.0], dtype=np.float64)

    bottom = bottom_surface_sample_points(box, step_cm=2.0)
    boundary = boundary_sample_points(box, step_cm=2.0)

    assert bottom.ndim == 2
    assert bottom.shape[1] == 3
    assert boundary.ndim == 2
    assert boundary.shape[1] == 3
    assert np.allclose(bottom[:, 2], 2.0)
    assert len(boundary) > len(bottom)


def test_side_boundary_sample_groups_return_four_faces() -> None:
    box = np.asarray([10.0, 20.0, 3.0, 6.0, 4.0, 2.0, 0.0], dtype=np.float64)

    groups = side_boundary_sample_groups(box, step_cm=2.0)

    assert len(groups) == 4
    assert all(group.ndim == 2 and group.shape[1] == 3 for group in groups)
    assert all(len(group) > 0 for group in groups)


def test_material_filename_order_matches_original_resolution_exports() -> None:
    expected_prefixes = [f"{index:02d}_" for index in range(1, 16)]
    names = [material_filename(index, spec) for index, spec in enumerate(MATERIAL_SPECS, start=1)]

    assert [name[:3] for name in names] == expected_prefixes


def test_gaussian_expand_heatmap_scores_spreads_on_support_surface() -> None:
    support = np.asarray([[0.0, 0.0, 0.0], [3.0, 0.0, 0.0], [12.0, 0.0, 0.0]], dtype=np.float64)
    heatmap = np.asarray([[0.0, 0.0, 0.0]], dtype=np.float64)
    scores = np.asarray([1.0], dtype=np.float32)

    expanded = gaussian_expand_heatmap_scores(support, heatmap, scores, sigma_cm=4.0, min_score=0.0)

    assert expanded.shape == (len(support),)
    assert expanded[0] == 1.0
    assert expanded[0] > expanded[1] > expanded[2]


def test_voxel_probability_colors_keep_non_support_rgb() -> None:
    voxel_points = np.asarray([[0.2, 0.2, 0.2], [3.2, 0.2, 0.2]], dtype=np.float64)
    voxel_colors = np.asarray([[10, 20, 30], [40, 50, 60]], dtype=np.uint8)
    support_points = np.asarray([[0.2, 0.2, 0.2]], dtype=np.float64)
    support_scores = np.asarray([1.0], dtype=np.float32)

    colors = voxel_probability_colors(voxel_points, voxel_colors, support_points, support_scores, voxel_size_cm=1.0)

    assert colors.shape == (2, 3)
    assert not np.allclose(colors[0], voxel_colors[0] / 255.0)
    assert np.allclose(colors[1], voxel_colors[1] / 255.0)


def test_points_inside_yaw_box_respects_rotation_and_extent() -> None:
    box = np.asarray([0.0, 0.0, 0.0, 4.0, 2.0, 2.0, np.pi / 2.0], dtype=np.float64)
    points = np.asarray([[0.0, 1.5, 0.0], [1.5, 0.0, 0.0], [0.0, 0.0, 1.1]], dtype=np.float64)

    inside = points_inside_yaw_box(points, box)

    assert inside.tolist() == [True, False, False]


def test_obb_surface_sample_points_cover_all_faces() -> None:
    obb = {
        "center": np.zeros(3),
        "axes": np.eye(3),
        "half_extents": np.ones(3),
    }

    points = obb_surface_sample_points(obb, step_cm=1.0)

    assert points.shape[1] == 3
    assert len(points) > 0
    assert np.all(np.max(np.abs(points), axis=1) == 1.0)

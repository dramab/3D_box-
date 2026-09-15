#!/usr/bin/env python
"""Export single-image materials for the method overview figure.

使用示例:
    python tools/export_method_overview_materials.py \
        --config configs/lc_bgplacenet_stage2_enriched.yaml

    python tools/export_method_overview_materials.py \
        --config configs/lc_bgplacenet_stage2_enriched.yaml \
        --sample-id hope__scene_0000__0325 \
        --object-id obj_3

    python tools/export_method_overview_materials.py \
        --config configs/lc_bgplacenet_stage2_enriched.yaml \
        --sample-id hope__scene_0000__0005 \
        --object-id obj_3 \
        --predictions-json outputs/lc_bgplacenet_stage2_space_former_aligned_loss_48query_full_gt_guass_8_enriched/inference_custom_hope_scene_0000_0005_tomato_sauce_back_left_mustard/predictions.json
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import textwrap
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

os.environ.setdefault("MPLCONFIGDIR", os.fspath(PROJECT_ROOT / "outputs" / ".matplotlib"))
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.annotation.free_bbox.io_utils import load_ply
from src.annotation.auto_label import describe_spatial_relation
from src.datasets.canonical import CanonicalScene, ObjectInfo, load_canonical_scene
from src.placement_metrics import (
    build_connected_support_region,
    compute_supported_and_stable,
    footprint_voxel_keys,
    quantize_occupied_points,
    support_z_key_bounds,
)
from src.training.lc_bgplacenet_stage2 import (
    Stage2IndexItem,
    _is_heatmap_positive_color,
    _is_support_color,
    build_collision_context,
    build_sources_from_config,
    build_stage2_index,
    compute_collision_metrics,
    load_config,
)
from tools.infer_lc_bgplacenet_stage2 import (
    draw_world_corners,
    place_box_to_corners,
    source_box_to_oriented_corners,
)
from tools.render_lc_bgplacenet_stage2_point_mask import (
    decode_heatmap_scores,
    heatmap_colors,
    normalize_scores_by_max,
)
from tools.render_topological_box_reasoner_vis import (
    canonical_aabb_to_world_corners,
)


DEFAULT_SAMPLE_ID = "hope__scene_0000__0325"
DEFAULT_OBJECT_ID = "obj_3"
DEFAULT_REFERENCE_ID = "obj_8"
MATERIAL_SIZE = (960, 720)
SOURCE_COLOR = "#FF1744"
REFERENCE_COLOR = "#00A6FF"
PLACE_COLOR = "#00E676"
PREDICTED_COLOR = "#FFD600"
SUPPORT_COLOR = "#00E676"
COLLISION_COLOR = "#FF2D00"
SEMANTIC_PROBE_COLOR = "#FF8A00"
BOX_HALO_COLOR = "#050505"
NEUTRAL_COLOR = "#CBD5E1"
SPARSE_VOXEL_POINT_SIZE = 16.0
SPARSE_VOXEL_HALO_POINT_SIZE = 34.0
HEATMAP_GAUSSIAN_SIGMA_CM = 5.0
HEATMAP_GAUSSIAN_MIN_SCORE = 0.025
HEATMAP_GAUSSIAN_MAX_SEEDS = 2500
HEATMAP_COLORBAR_FILENAME = "05b_probability_colorbar.png"
HEATMAP_VOXEL_FILENAME = "05c_sparse_voxel_probability_heatmap.png"
BOX_EDGES = (
    (0, 1),
    (0, 2),
    (1, 3),
    (2, 3),
    (4, 5),
    (4, 6),
    (5, 7),
    (6, 7),
    (0, 4),
    (1, 5),
    (2, 6),
    (3, 7),
)
SIDE_FACE_CORNERS = (
    (0, 1, 5, 4),
    (2, 3, 7, 6),
    (0, 2, 6, 4),
    (1, 3, 7, 5),
)


@dataclass(frozen=True)
class MaterialSpec:
    """One exported method-overview material."""

    key: str
    title: str


@dataclass(frozen=True)
class PredictionData:
    """Optional Stage-2 prediction fields used for predicted materials."""

    source_box: np.ndarray
    final_box: np.ndarray
    heatmap_ply: Path | None
    decoder_boxes: list[np.ndarray]
    raw_record: dict[str, Any]


MATERIAL_SPECS = (
    MaterialSpec("language_instruction", "Language Instruction"),
    MaterialSpec("rgb_observation", "RGB Observation"),
    MaterialSpec("sparse_voxel_observation", "Sparse Voxel Observation"),
    MaterialSpec("source_object_grounding", "Source Object Grounding"),
    MaterialSpec("coarse_placement_heatmap", "Coarse Placement Heatmap"),
    MaterialSpec("decoder_iterative_refinement", "Decoder Iterative Refinement"),
    MaterialSpec("boundary_space_reasoning", "Boundary-Space Reasoning"),
    MaterialSpec("support_space_reasoning", "Support-Space Reasoning"),
    MaterialSpec("final_physically_feasible_placement", "Final Physically-Feasible Placement"),
    MaterialSpec("semantic_consistency", "Semantic Consistency"),
    MaterialSpec("bottom_surface_sampling", "Bottom-Surface Sampling"),
    MaterialSpec("support_connectivity_check", "Support Connectivity Check"),
    MaterialSpec("boundary_sampling", "Boundary Sampling"),
    MaterialSpec("clearance_check", "Clearance Check"),
    MaterialSpec("collision_free_placement", "Collision-Free Placement"),
)


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description="Export method overview material images.")
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs/lc_bgplacenet_stage2_enriched.yaml")
    parser.add_argument("--sample-id", default=DEFAULT_SAMPLE_ID)
    parser.add_argument("--object-id", default=DEFAULT_OBJECT_ID)
    parser.add_argument("--reference-object-id", default=DEFAULT_REFERENCE_ID)
    parser.add_argument("--cluster-id", type=int, default=None, help="Select the exact GT placement cluster.")
    parser.add_argument("--relation", default=None, help="Optional target relation filter.")
    parser.add_argument("--predictions-json", type=Path, default=None, help="Optional Stage-2 predictions.json.")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--max-scene-points", type=int, default=20000)
    parser.add_argument("--max-heatmap-points", type=int, default=2500)
    parser.add_argument("--voxel-size-cm", type=float, default=1.0)
    parser.add_argument("--support-downward-cm", type=float, default=3.0)
    parser.add_argument("--support-upper-cm", type=float, default=1.0)
    return parser.parse_args()


def material_filename(index: int, spec: MaterialSpec) -> str:
    """Return the stable filename for one material image."""
    return f"{index:02d}_{spec.key}.png"


def default_output_dir(item: Stage2IndexItem) -> Path:
    """Place material images in one sample-specific output directory."""
    return PROJECT_ROOT / "outputs" / "method_overview_materials" / f"{item.sample_id}__{item.object_id}"


def resolve_path(path: Path | str | None) -> Path | None:
    """Resolve repository-relative paths."""
    if path is None:
        return None
    value = Path(path)
    return value if value.is_absolute() else PROJECT_ROOT / value


def load_font(size: int, *, bold: bool = False) -> ImageFont.FreeTypeFont:
    """Load a stable local font."""
    suffix = "-Bold.ttf" if bold else ".ttf"
    path = Path("/usr/share/fonts/truetype/dejavu") / f"DejaVuSans{suffix}"
    return ImageFont.truetype(os.fspath(path), size=size)


def find_item(
    cfg: dict[str, Any],
    sample_id: str,
    object_id: str,
    reference_object_id: str | None,
    relation: str | None,
    cluster_id: int | None = None,
) -> Stage2IndexItem:
    """Find the GT-faithful Stage-2 item used by all material images."""
    items = build_stage2_index(build_sources_from_config(cfg))
    matches = [
        item
        for item in items
        if item.sample_id == sample_id
        and item.object_id == object_id
        and (cluster_id is None or item.cluster_id == cluster_id)
        and (reference_object_id is None or item.reference_object_id == reference_object_id)
        and (relation is None or item.target_relation == relation)
    ]
    if not matches:
        raise ValueError(f"No Stage-2 item found for sample_id={sample_id}, object_id={object_id}")
    return matches[0]


def find_object(scene: CanonicalScene, object_id: str) -> ObjectInfo:
    """Find one canonical object by id."""
    for obj in scene.objects:
        if obj.obj_id == object_id:
            return obj
    raise ValueError(f"Object {object_id} not found in sample {scene.sample_id}")


def load_prediction(path: Path | None, sample_id: str, object_id: str) -> PredictionData | None:
    """Load one optional Stage-2 prediction record."""
    if path is None:
        return None
    resolved = resolve_path(path)
    assert resolved is not None
    with resolved.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    records = payload if isinstance(payload, list) else [payload]
    record = next(
        (
            row
            for row in records
            if str(row.get("sample_id")) == sample_id and str(row.get("object_id")) == object_id
        ),
        None,
    )
    if record is None:
        raise ValueError(f"{resolved} has no prediction for sample_id={sample_id}, object_id={object_id}")
    decoder_boxes = [
        np.asarray(stage["placements"][0]["box"], dtype=np.float64)
        for stage in record.get("decoder_stages", [])
        if stage.get("placements")
    ]
    return PredictionData(
        source_box=np.asarray(record["source_box"], dtype=np.float64),
        final_box=np.asarray(record["place_box"], dtype=np.float64),
        heatmap_ply=resolve_path(record.get("pred_heatmap_ply")),
        decoder_boxes=decoder_boxes,
        raw_record=record,
    )


def load_scene_and_points(item: Stage2IndexItem) -> tuple[CanonicalScene, np.ndarray, np.ndarray]:
    """Load canonical scene and its 1cm voxel cloud."""
    scene = load_canonical_scene(item.dataset_dir / "samples" / f"{item.sample_id}.json", dataset_root=item.dataset_dir)
    if scene.voxel_point_cloud_path is None:
        raise ValueError(f"Sample {item.sample_id} does not provide a voxel point cloud")
    points, colors = load_ply(scene.voxel_point_cloud_path)
    return scene, points.astype(np.float64), colors.astype(np.uint8)


def scene_box_arrays(
    scene: CanonicalScene,
    item: Stage2IndexItem,
    prediction: PredictionData | None,
) -> dict[str, np.ndarray]:
    """Collect source, reference, GT and final candidate boxes."""
    source_obj = find_object(scene, item.object_id)
    ref_obj = find_object(scene, item.reference_object_id)
    source_box = prediction.source_box if prediction is not None else np.asarray(item.source_box_gt, dtype=np.float64)
    final_box = prediction.final_box if prediction is not None else np.asarray(item.place_box_gt, dtype=np.float64)
    return {
        "source_corners": source_box_to_oriented_corners(source_box, source_obj.pose_world),
        "reference_corners": canonical_aabb_to_world_corners(ref_obj.bbox3d_canonical, ref_obj.pose_world),
        "gt_place_box": np.asarray(item.place_box_gt, dtype=np.float64),
        "final_box": final_box,
        "source_box": source_box,
    }


def make_collision_probe_box(
    box: np.ndarray,
    reference_corners: np.ndarray,
) -> np.ndarray:
    """Create an in-scene box whose side penetrates the reference object."""
    collision = np.asarray(box, dtype=np.float64).copy()
    ref_center = reference_corners.mean(axis=0)
    direction = collision[:2] - ref_center[:2]
    if np.linalg.norm(direction) < 1e-6:
        direction = np.asarray([1.0, 0.0], dtype=np.float64)
    direction = direction / np.linalg.norm(direction)
    yaw = float(collision[6])
    c, s = np.cos(yaw), np.sin(yaw)
    box_axes = np.asarray([[c, s], [-s, c]], dtype=np.float64)
    box_extent = float(np.abs(box_axes @ direction).dot(collision[3:5] * 0.5))
    ref_extent = float(np.max((reference_corners[:, :2] - ref_center[None, :2]) @ direction))
    # box_extent is half the full projected width along the collision direction.
    penetration = 0.50 * (2.0 * box_extent)
    collision[:2] = ref_center[:2] + direction * (ref_extent + box_extent - penetration)
    collision[2] = box[2]
    return collision


def bottom_surface_sample_points(box: np.ndarray, step_cm: float = 1.4) -> np.ndarray:
    """Sample points on the yaw-only box bottom surface."""
    box = np.asarray(box, dtype=np.float64)
    x_values = np.arange(-box[3] * 0.5, box[3] * 0.5 + 1e-6, step_cm)
    y_values = np.arange(-box[4] * 0.5, box[4] * 0.5 + 1e-6, step_cm)
    local_xy = np.stack(np.meshgrid(x_values, y_values, indexing="xy"), axis=-1).reshape(-1, 2)
    local = np.column_stack([local_xy, np.full(len(local_xy), -box[5] * 0.5)])
    return transform_yaw_box_local_points(box, local)


def boundary_sample_points(box: np.ndarray, step_cm: float = 2.0) -> np.ndarray:
    """Sample the six faces of one yaw-only placement box."""
    box = np.asarray(box, dtype=np.float64)
    axes_values = [
        np.arange(-box[idx] * 0.5, box[idx] * 0.5 + 1e-6, step_cm)
        for idx in range(3, 6)
    ]
    samples = []
    for fixed_axis, fixed_sign in [(0, -1), (0, 1), (1, -1), (1, 1), (2, -1), (2, 1)]:
        free_axes = [axis for axis in range(3) if axis != fixed_axis]
        grid = np.stack(
            np.meshgrid(axes_values[free_axes[0]], axes_values[free_axes[1]], indexing="xy"),
            axis=-1,
        ).reshape(-1, 2)
        local = np.zeros((len(grid), 3), dtype=np.float64)
        local[:, fixed_axis] = fixed_sign * box[3 + fixed_axis] * 0.5
        local[:, free_axes] = grid
        samples.append(local)
    return transform_yaw_box_local_points(box, np.vstack(samples))


def side_boundary_sample_groups(box: np.ndarray, step_cm: float = 1.4) -> list[np.ndarray]:
    """Sample each vertical side face of one yaw-only placement box."""
    box = np.asarray(box, dtype=np.float64)
    axes_values = [
        np.arange(-box[idx] * 0.5, box[idx] * 0.5 + 1e-6, step_cm)
        for idx in range(3, 6)
    ]
    samples = []
    for fixed_axis, fixed_sign in [(0, -1), (0, 1), (1, -1), (1, 1)]:
        free_axes = [axis for axis in range(3) if axis != fixed_axis]
        grid = np.stack(
            np.meshgrid(axes_values[free_axes[0]], axes_values[free_axes[1]], indexing="xy"),
            axis=-1,
        ).reshape(-1, 2)
        local = np.zeros((len(grid), 3), dtype=np.float64)
        local[:, fixed_axis] = fixed_sign * box[3 + fixed_axis] * 0.5
        local[:, free_axes] = grid
        samples.append(transform_yaw_box_local_points(box, local))
    return samples


def side_boundary_sample_points(box: np.ndarray, step_cm: float = 1.4) -> np.ndarray:
    """Sample only the vertical side faces of one yaw-only placement box."""
    return np.vstack(side_boundary_sample_groups(box, step_cm=step_cm))


def box_edge_sample_points(box: np.ndarray, points_per_edge: int = 7) -> np.ndarray:
    """Sample sparse, readable points along the 12 box edges."""
    corners = place_box_to_corners(np.asarray(box, dtype=np.float64))
    t = np.linspace(0.0, 1.0, int(points_per_edge), dtype=np.float64)
    points = []
    for start, end in BOX_EDGES:
        edge_points = corners[start][None, :] * (1.0 - t[:, None]) + corners[end][None, :] * t[:, None]
        points.append(edge_points)
    return np.vstack(points)


def transform_yaw_box_local_points(box: np.ndarray, local_points: np.ndarray) -> np.ndarray:
    """Transform local yaw-only box points to world coordinates."""
    yaw = float(box[6])
    c, s = math.cos(yaw), math.sin(yaw)
    rot = np.asarray([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)
    return np.asarray(local_points, dtype=np.float64) @ rot.T + box[:3][None, :]


def heatmap_scores_from_gt_colors(colors: np.ndarray) -> np.ndarray:
    """Decode free_bbox yellow-red GT heatmap colors into normalized scores."""
    colors_i = np.asarray(colors, dtype=np.int16)
    positive = _is_heatmap_positive_color(colors)
    scores = np.zeros(len(colors_i), dtype=np.float32)
    if np.any(positive):
        scores[positive] = ((220.0 - colors_i[positive, 1].astype(np.float32)) / 220.0).clip(0.0, 1.0)
        max_score = float(scores.max(initial=0.0))
        if max_score > 0.0:
            scores /= max_score
    return scores


def _is_gt_heatmap_support_or_positive(colors: np.ndarray) -> np.ndarray:
    """Return GT heatmap points that belong to the support-surface visualization."""
    colors_i = np.asarray(colors, dtype=np.int16)
    support = (colors_i[:, 0] == 55) & (colors_i[:, 1] == 120) & (colors_i[:, 2] == 210)
    return support | _is_heatmap_positive_color(colors)


def load_material_heatmap(item: Stage2IndexItem, prediction: PredictionData | None) -> tuple[np.ndarray, np.ndarray, str]:
    """Load predicted heatmap when available; otherwise use direction-filtered GT heatmap."""
    if prediction is not None and prediction.heatmap_ply is not None and prediction.heatmap_ply.exists():
        points, colors = load_ply(prediction.heatmap_ply)
        return points.astype(np.float64), normalize_scores_by_max(decode_heatmap_scores(colors)), "prediction"
    points, colors = load_ply(item.direction_filtered_heatmap_ply)
    support_or_positive = _is_gt_heatmap_support_or_positive(colors)
    support_z = float(item.place_box_gt[2] - item.place_box_gt[5] * 0.5)
    support_plane = np.abs(points[:, 2] - support_z) <= 1.25
    scores = heatmap_scores_from_gt_colors(colors)
    keep = support_or_positive & support_plane
    return points[keep].astype(np.float64), scores[keep], "gt_direction_filtered_support_plane"


def load_support_surface_points(item: Stage2IndexItem) -> np.ndarray:
    """Load the active free_bbox support mask used as the heatmap base surface."""
    points, colors = load_ply(item.support_mask_ply)
    return points[_is_support_color(colors)].astype(np.float64)


def gaussian_expand_heatmap_scores(
    support_points: np.ndarray,
    heatmap_points: np.ndarray,
    heatmap_scores: np.ndarray,
    *,
    sigma_cm: float = HEATMAP_GAUSSIAN_SIGMA_CM,
    min_score: float = HEATMAP_GAUSSIAN_MIN_SCORE,
) -> np.ndarray:
    """Spread sparse placement responses over the support surface for visualization."""
    support = np.asarray(support_points, dtype=np.float64)
    heatmap = np.asarray(heatmap_points, dtype=np.float64)
    scores = np.asarray(heatmap_scores, dtype=np.float32).clip(0.0, 1.0)
    if len(support) == 0:
        return np.zeros(0, dtype=np.float32)
    positive = np.flatnonzero(scores > 0.0)
    if len(positive) == 0:
        return np.zeros(len(support), dtype=np.float32)
    if len(positive) > HEATMAP_GAUSSIAN_MAX_SEEDS:
        positive = positive[np.argsort(scores[positive])[-HEATMAP_GAUSSIAN_MAX_SEEDS:]]
    delta_xy = support[:, None, :2] - heatmap[positive][None, :, :2]
    distances_sq = np.sum(delta_xy * delta_xy, axis=2)
    weights = np.exp(-0.5 * distances_sq / float(sigma_cm * sigma_cm)).astype(np.float32)
    expanded = np.max(weights * scores[positive][None, :], axis=1).clip(0.0, 1.0)
    expanded[expanded < float(min_score)] = 0.0
    return expanded


def voxel_probability_colors(
    voxel_points: np.ndarray,
    voxel_colors: np.ndarray,
    support_points: np.ndarray,
    support_scores: np.ndarray,
    voxel_size_cm: float,
) -> np.ndarray:
    """Color active support voxels by probability while keeping other voxel RGB colors."""
    voxel_keys = np.floor(np.asarray(voxel_points, dtype=np.float64) / float(voxel_size_cm)).astype(np.int64)
    support_keys = np.floor(np.asarray(support_points, dtype=np.float64) / float(voxel_size_cm)).astype(np.int64)
    scores = np.asarray(support_scores, dtype=np.float32).clip(0.0, 1.0)
    score_by_key = {tuple(key): float(score) for key, score in zip(support_keys, scores)}
    voxel_scores = np.fromiter(
        (score_by_key.get(tuple(key), -1.0) for key in voxel_keys),
        dtype=np.float32,
        count=len(voxel_keys),
    )
    colors = np.asarray(voxel_colors, dtype=np.float32) / 255.0
    support_mask = voxel_scores >= 0.0
    colors[support_mask] = heatmap_colors(voxel_scores[support_mask])
    return colors


def stride_indices(length: int, maximum: int) -> np.ndarray:
    """Subsample long point arrays deterministically."""
    if length <= maximum:
        return np.arange(length)
    return np.linspace(0, length - 1, maximum, dtype=np.int64)


def project_world_points(scene: CanonicalScene, points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Project world points to the real RGB camera plane."""
    pts = np.asarray(points, dtype=np.float64)
    homo = np.column_stack([pts, np.ones(len(pts), dtype=np.float64)])
    cam = homo @ scene.camera.E_w2c.T
    depth = cam[:, 2]
    uv = np.column_stack(
        [
            scene.camera.fx * cam[:, 0] / np.maximum(depth, 1e-6) + scene.camera.cx,
            scene.camera.fy * cam[:, 1] / np.maximum(depth, 1e-6) + scene.camera.cy,
        ]
    )
    visible = (
        (depth > 0.0)
        & (uv[:, 0] >= 0.0)
        & (uv[:, 0] < scene.camera.img_w)
        & (uv[:, 1] >= 0.0)
        & (uv[:, 1] < scene.camera.img_h)
    )
    return uv, visible


def box_projection_clear(scene: CanonicalScene, box: np.ndarray, margin_px: float = 18.0) -> bool:
    """Require one projected box to remain readable in the RGB frame."""
    corners = place_box_to_corners(np.asarray(box, dtype=np.float64))
    uv, visible = project_world_points(scene, corners)
    center_uv, center_visible = project_world_points(scene, np.asarray([np.asarray(box, dtype=np.float64)[:3]]))
    if not bool(center_visible[0]) or int(visible.sum()) < 5:
        return False
    margin = float(margin_px)
    center = center_uv[0]
    return bool(
        (center[0] >= margin)
        and (center[0] <= scene.camera.img_w - margin)
        and (center[1] >= margin)
        and (center[1] <= scene.camera.img_h - margin)
    )


def hex_to_rgb(color: str) -> tuple[int, int, int]:
    """Convert a #RRGGBB color to an RGB tuple."""
    value = color.lstrip("#")
    return tuple(int(value[index : index + 2], 16) for index in (0, 2, 4))


def draw_contrast_world_corners(
    image: Image.Image,
    corners_world: np.ndarray,
    scene: CanonicalScene,
    color: tuple[int, int, int],
) -> bool:
    """Draw translucent faces with clear outlines while preserving scene detail."""
    uv, _ = project_world_points(scene, corners_world)
    camera_points = np.column_stack([corners_world, np.ones(8)]) @ scene.camera.E_w2c.T
    faces = np.asarray([(0, 1, 3, 2), (4, 5, 7, 6), (0, 1, 5, 4),
                        (2, 3, 7, 6), (0, 2, 6, 4), (1, 3, 7, 5)])
    # Far faces first; each face has 10% opacity so overlapping faces stay light.
    for face in faces[np.argsort(camera_points[faces, 2].mean(axis=1))[::-1]]:
        if np.any(camera_points[face, 2] <= 0):
            continue
        overlay = Image.new("RGBA", image.size)
        ImageDraw.Draw(overlay).polygon([tuple(p) for p in uv[face]], fill=(*color, 26))
        image.alpha_composite(overlay)
    return draw_world_corners(ImageDraw.Draw(image), corners_world, scene, color, 6, "")


def draw_contrast_place_box(
    image: Image.Image,
    box: np.ndarray,
    scene: CanonicalScene,
    color: tuple[int, int, int],
) -> bool:
    """Draw a yaw-only placement box with high-contrast edges."""
    return draw_contrast_world_corners(image, place_box_to_corners(box), scene, color)


def draw_projected_polygon(
    image: Image.Image,
    scene: CanonicalScene,
    corners_world: np.ndarray,
    fill: tuple[int, int, int],
    *,
    alpha: int = 120,
) -> bool:
    """Fill one projected 3D face on the RGB image."""
    uv, visible = project_world_points(scene, corners_world)
    if int(np.count_nonzero(visible)) < 3:
        return False
    polygon = [(float(x), float(y)) for x, y in uv]
    overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
    overlay_draw = ImageDraw.Draw(overlay)
    overlay_draw.polygon(polygon, fill=(*fill, int(alpha)))
    overlay_draw.line(polygon + [polygon[0]], fill=(*fill, 255), width=6)
    image.alpha_composite(overlay)
    return True


def draw_collision_contact_points(
    image: Image.Image,
    scene: CanonicalScene,
    points: np.ndarray,
    max_points: int = 520,
) -> None:
    """Draw collision samples as red contact points with black halos."""
    draw_projected_points(image, scene, points, hex_to_rgb(BOX_HALO_COLOR), radius=8, alpha=210, max_points=max_points)
    draw_projected_points(image, scene, points, (255, 245, 230), radius=5, alpha=235, max_points=max_points)
    draw_projected_points(image, scene, points, hex_to_rgb(COLLISION_COLOR), radius=3, alpha=255, max_points=max_points)


def draw_collision_overlap_points(
    image: Image.Image,
    scene: CanonicalScene,
    points: np.ndarray,
    max_points: int = 360,
) -> None:
    """Draw occupied scene voxels inside the candidate box as a compact overlap region."""
    draw_projected_points(image, scene, points, hex_to_rgb(BOX_HALO_COLOR), radius=10, alpha=220, max_points=max_points)
    draw_projected_points(image, scene, points, (255, 245, 230), radius=7, alpha=245, max_points=max_points)
    draw_projected_points(image, scene, points, hex_to_rgb(COLLISION_COLOR), radius=5, alpha=255, max_points=max_points)


def draw_projected_points(
    image: Image.Image,
    scene: CanonicalScene,
    points: np.ndarray,
    colors: np.ndarray | tuple[int, int, int],
    *,
    radius: int = 4,
    alpha: int = 190,
    max_points: int | None = None,
) -> int:
    """Draw world points on the real RGB camera image."""
    pts = np.asarray(points, dtype=np.float64)
    if len(pts) == 0:
        return 0
    uv, visible = project_world_points(scene, pts)
    indices = np.flatnonzero(visible)
    if max_points is not None:
        indices = indices[stride_indices(len(indices), int(max_points))]
    if len(indices) == 0:
        return 0
    overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    if isinstance(colors, tuple):
        point_colors = [colors] * len(indices)
    else:
        array = np.asarray(colors)
        if array.max(initial=0.0) <= 1.0:
            array = np.rint(array * 255.0)
        point_colors = [tuple(np.asarray(array[index], dtype=np.uint8).tolist()) for index in indices]
    for (x, y), color in zip(uv[indices], point_colors):
        draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=(*color, int(alpha)))
    image.alpha_composite(overlay)
    return int(len(indices))


def save_scene_overlay(
    scene: CanonicalScene,
    output_path: Path,
    title: str,
    draw_fn: Any,
) -> None:
    """Save one material as geometry overlays on the original RGB observation."""
    image = Image.fromarray(scene.rgb).convert("RGBA")
    draw = ImageDraw.Draw(image)
    draw_fn(image, draw)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.convert("RGB").save(output_path)


def save_rgb_box_image(
    scene: CanonicalScene,
    source_corners: np.ndarray | None,
    place_box: np.ndarray | None,
    reference_corners: np.ndarray | None,
    output_path: Path,
    *,
    title: str | None = None,
) -> None:
    """Save an original RGB image with projected 3D boxes only."""
    image = Image.fromarray(scene.rgb).convert("RGBA")
    if reference_corners is not None:
        draw_contrast_world_corners(image, reference_corners, scene, hex_to_rgb(REFERENCE_COLOR))
    if source_corners is not None:
        draw_contrast_world_corners(image, source_corners, scene, hex_to_rgb(SOURCE_COLOR))
    if place_box is not None:
        draw_contrast_place_box(image, place_box, scene, hex_to_rgb(PLACE_COLOR))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)


def save_language_instruction(item: Stage2IndexItem, scene: CanonicalScene, output_path: Path) -> None:
    """Render the instruction as an independent material card."""
    image = Image.fromarray(scene.rgb).convert("RGB").convert("RGBA")
    width, height = image.size
    overlay = Image.new("RGBA", image.size, (255, 255, 255, 0))
    overlay_draw = ImageDraw.Draw(overlay)
    overlay_draw.rounded_rectangle(
        (20, 22, width - 20, height - 22),
        radius=14,
        fill=(255, 255, 255, 225),
        outline=(30, 64, 175, 230),
        width=3,
    )
    image.alpha_composite(overlay)
    draw = ImageDraw.Draw(image)
    title_font = load_font(24, bold=True)
    body_font = load_font(19)
    badge_font = load_font(15, bold=True)
    source_name = find_object(scene, item.object_id).class_name
    ref_name = find_object(scene, item.reference_object_id).class_name
    draw.text((42, 42), "Language Instruction", font=title_font, fill=(30, 64, 175))
    wrapped = textwrap.wrap(item.instruction, width=44)
    y = 92
    for line in wrapped:
        draw.text((44, y), line, font=body_font, fill=(17, 24, 39))
        y += 30
    badges = [
        ("source", source_name, (225, 29, 72)),
        ("relation", item.target_relation, (109, 40, 217)),
        ("reference", ref_name, (37, 99, 235)),
    ]
    y = max(y + 24, height - 176)
    for label, value, color in badges:
        draw.rounded_rectangle((44, y, width - 44, y + 38), radius=8, outline=color, width=2, fill=(248, 250, 252))
        draw.text((62, y + 10), f"{label}: ", font=badge_font, fill=(75, 85, 99))
        draw.text((154, y + 10), value, font=badge_font, fill=color)
        y += 48
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.convert("RGB").save(output_path)


def save_sparse_voxel_observation(
    scene: CanonicalScene,
    voxel_points: np.ndarray,
    voxel_colors: np.ndarray,
    output_path: Path,
    max_scene_points: int,
) -> None:
    """Render a camera-like sparse voxel observation."""
    point_h = np.column_stack([voxel_points, np.ones(len(voxel_points), dtype=np.float64)])
    cam = point_h @ scene.camera.E_w2c.T
    depth = cam[:, 2]
    uv = np.column_stack([
        scene.camera.fx * cam[:, 0] / np.maximum(depth, 1e-6) + scene.camera.cx,
        scene.camera.fy * cam[:, 1] / np.maximum(depth, 1e-6) + scene.camera.cy,
    ])
    valid = (
        (depth > 0.0)
        & (uv[:, 0] >= 0.0)
        & (uv[:, 0] < scene.camera.img_w)
        & (uv[:, 1] >= 0.0)
        & (uv[:, 1] < scene.camera.img_h)
    )
    visible = np.flatnonzero(valid)
    visible = visible[np.argsort(depth[visible])[::-1]]
    visible = visible[stride_indices(len(visible), max_scene_points)]
    fig, ax = plt.subplots(figsize=(scene.camera.img_w / 140, scene.camera.img_h / 140), dpi=140, facecolor="white")
    colors = voxel_colors[visible] / 255.0
    # Two-pass drawing makes each 1cm voxel visually readable without changing the camera projection.
    ax.scatter(uv[visible, 0], uv[visible, 1], s=SPARSE_VOXEL_HALO_POINT_SIZE, c=colors, alpha=0.34, linewidths=0.0)
    ax.scatter(uv[visible, 0], uv[visible, 1], s=SPARSE_VOXEL_POINT_SIZE, c=colors, alpha=0.95, linewidths=0.0)
    ax.set_xlim(0, scene.camera.img_w)
    ax.set_ylim(scene.camera.img_h, 0)
    ax.axis("off")
    fig.savefig(output_path, dpi=140, facecolor="white", pad_inches=0)
    plt.close(fig)


def save_sparse_voxel_probability_heatmap(
    scene: CanonicalScene,
    voxel_points: np.ndarray,
    voxel_colors: np.ndarray,
    support_points: np.ndarray,
    heatmap_points: np.ndarray,
    heatmap_scores: np.ndarray,
    output_path: Path,
    max_scene_points: int,
    voxel_size_cm: float,
) -> None:
    """Render placement probability directly on the sparse voxel observation."""
    support_scores = gaussian_expand_heatmap_scores(support_points, heatmap_points, heatmap_scores)
    colors = voxel_probability_colors(voxel_points, voxel_colors, support_points, support_scores, voxel_size_cm)
    point_h = np.column_stack([voxel_points, np.ones(len(voxel_points), dtype=np.float64)])
    cam = point_h @ scene.camera.E_w2c.T
    depth = cam[:, 2]
    uv = np.column_stack([
        scene.camera.fx * cam[:, 0] / np.maximum(depth, 1e-6) + scene.camera.cx,
        scene.camera.fy * cam[:, 1] / np.maximum(depth, 1e-6) + scene.camera.cy,
    ])
    valid = (
        (depth > 0.0)
        & (uv[:, 0] >= 0.0)
        & (uv[:, 0] < scene.camera.img_w)
        & (uv[:, 1] >= 0.0)
        & (uv[:, 1] < scene.camera.img_h)
    )
    visible = np.flatnonzero(valid)
    visible = visible[np.argsort(depth[visible])[::-1]]
    visible = visible[stride_indices(len(visible), max_scene_points)]
    fig, ax = plt.subplots(figsize=(scene.camera.img_w / 140, scene.camera.img_h / 140), dpi=140, facecolor="white")
    # Keep the same voxel thickness as 03 so only the probability coloring changes.
    ax.scatter(uv[visible, 0], uv[visible, 1], s=SPARSE_VOXEL_HALO_POINT_SIZE, c=colors[visible], alpha=0.34, linewidths=0.0)
    ax.scatter(uv[visible, 0], uv[visible, 1], s=SPARSE_VOXEL_POINT_SIZE, c=colors[visible], alpha=0.95, linewidths=0.0)
    ax.set_xlim(0, scene.camera.img_w)
    ax.set_ylim(scene.camera.img_h, 0)
    ax.axis("off")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=140, facecolor="white", pad_inches=0)
    plt.close(fig)


def save_heatmap_material(
    scene: CanonicalScene,
    boxes: dict[str, np.ndarray],
    support_points: np.ndarray,
    heatmap_points: np.ndarray,
    heatmap_scores: np.ndarray,
    output_path: Path,
    max_heatmap_points: int,
) -> None:
    """Render coarse placement heatmap on the real RGB scene."""
    def draw_fn(image: Image.Image, draw: ImageDraw.ImageDraw) -> None:
        support_scores = gaussian_expand_heatmap_scores(support_points, heatmap_points, heatmap_scores)
        point_colors = heatmap_colors(support_scores)
        draw_projected_points(
            image,
            scene,
            support_points,
            point_colors,
            radius=6,
            alpha=185,
            max_points=max_heatmap_points,
        )
        draw_contrast_world_corners(image, boxes["source_corners"], scene, hex_to_rgb(SOURCE_COLOR))

    save_scene_overlay(scene, output_path, "Coarse Placement Heatmap", draw_fn)


def save_probability_colorbar(output_path: Path, size: tuple[int, int] = (640, 120)) -> None:
    """Save the probability palette used by coarse placement heatmap materials."""
    width, height = size
    image = Image.new("RGB", size, (255, 255, 255))
    draw = ImageDraw.Draw(image)
    bar_x0, bar_x1 = 54, width - 54
    bar_y0, bar_y1 = 38, 66
    scores = np.linspace(0.0, 1.0, bar_x1 - bar_x0, dtype=np.float32)
    colors = (heatmap_colors(scores) * 255.0).astype(np.uint8)
    for offset, color in enumerate(colors):
        x = bar_x0 + offset
        draw.line((x, bar_y0, x, bar_y1), fill=tuple(color.tolist()))
    draw.rectangle((bar_x0, bar_y0, bar_x1, bar_y1), outline=(31, 41, 55), width=1)
    font = load_font(18)
    draw.text((bar_x0, bar_y1 + 16), "Low probability", font=font, fill=(31, 41, 55))
    high_text = "High probability"
    bbox = draw.textbbox((0, 0), high_text, font=font)
    text_w = bbox[2] - bbox[0]
    draw.text((bar_x1 - text_w, bar_y1 + 16), high_text, font=font, fill=(31, 41, 55))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)


def support_topology(
    voxel_points: np.ndarray,
    box: np.ndarray,
    voxel_size_cm: float,
    downward_cm: float,
    upper_cm: float,
) -> tuple[np.ndarray, np.ndarray, float, tuple[int, int]]:
    """Return connected support component points and footprint points."""
    occupied = quantize_occupied_points(voxel_points, voxel_size_cm)
    bottom_z = float(box[2] - box[5] * 0.5)
    z_min, z_max = support_z_key_bounds(bottom_z, downward_cm, upper_cm, voxel_size_cm)
    region = build_connected_support_region(occupied, z_min, z_max)
    center_key = np.floor(box[:2] / voxel_size_cm).astype(np.int64)
    center_local = center_key - region.origin_xy
    center_inside = np.all(center_local >= 0) and np.all(center_local < np.asarray(region.labels.shape))
    component_id = int(region.labels[tuple(center_local)]) if center_inside else 0
    component_keys = np.argwhere(region.labels == component_id) + region.origin_xy if component_id else np.empty((0, 2), dtype=np.int64)
    footprint = footprint_voxel_keys(box, voxel_size_cm)
    _, coverage = compute_supported_and_stable(box, occupied, voxel_size_cm, downward_cm, upper_cm)
    return (
        np.column_stack(
            [
                (component_keys.astype(np.float64) + 0.5) * voxel_size_cm,
                np.full(len(component_keys), bottom_z, dtype=np.float64),
            ]
        ),
        np.column_stack(
            [
                (footprint.astype(np.float64) + 0.5) * voxel_size_cm,
                np.full(len(footprint), bottom_z, dtype=np.float64),
            ]
        ),
        float(coverage),
        (z_min, z_max),
    )


def load_collision_context(item: Stage2IndexItem) -> dict[str, Any]:
    """Load the same scene OBBs used by Stage-2 collision metrics."""
    placement_path = item.free_bbox_dir / "placements" / f"{item.sample_id}__placements.json"
    with placement_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return build_collision_context(payload.get("objects", []))


def canonical_scene_collision_context(scene: CanonicalScene) -> dict[str, Any]:
    """Build an OBB context from every object visible in the canonical scene."""
    return build_collision_context(
        [
            {
                "object_id": obj.obj_id,
                "canonical_aabb_object": obj.bbox3d_canonical,
                "original_pose_world": obj.pose_world,
            }
            for obj in scene.objects
        ]
    )


def placement_relation(scene: CanonicalScene, box: np.ndarray, reference_corners: np.ndarray) -> str:
    """Compute the language relation of one candidate placement box."""
    return describe_spatial_relation(
        place_box_to_corners(np.asarray(box, dtype=np.float64)),
        reference_corners,
        scene.camera.E_w2c,
        scene.camera.K,
    )


def is_collision_free(box: np.ndarray, collision_context: dict[str, Any]) -> bool:
    """Return whether one placement box is collision-free under benchmark OBB logic."""
    return not bool(compute_collision_metrics(box, collision_context)["collision"])


def candidate_from_bottom_center(bottom_center: np.ndarray, dims: np.ndarray, yaw: float) -> np.ndarray:
    """Build a placement box from a bottom center, dimensions and yaw."""
    center = np.asarray(bottom_center, dtype=np.float64).copy()
    center[2] += float(dims[2]) * 0.5
    return np.concatenate([center, np.asarray(dims, dtype=np.float64), np.asarray([float(yaw)])])


def choose_semantic_wrong_box(
    item: Stage2IndexItem,
    scene: CanonicalScene,
    reference_corners: np.ndarray,
    voxel_points: np.ndarray,
    collision_context: dict[str, Any],
    *,
    voxel_size_cm: float,
    downward_cm: float,
    upper_cm: float,
) -> np.ndarray:
    """Prefer an opposite-direction, physically valid probe on the GT support surface."""
    gt_box = np.asarray(item.place_box_gt, dtype=np.float64)
    centers, _, _, _ = support_topology(voxel_points, gt_box, voxel_size_cm, downward_cm, upper_cm)
    with np.load(item.yaw_set_npz) as yaw_set:
        yaw_angles = np.asarray(yaw_set["yaw_angles_rad"], dtype=np.float64)
    # Search the full support region, not the direction-filtered GT center list.
    yaws = np.tile(yaw_angles, len(centers))
    centers = np.repeat(centers, len(yaw_angles), axis=0)
    ref_xy = reference_corners.mean(axis=0)[:2]
    gt_direction = gt_box[:2] - ref_xy
    directions = centers[:, :2] - ref_xy
    lengths = np.linalg.norm(directions, axis=1)
    cosine = directions @ gt_direction / np.maximum(lengths * np.linalg.norm(gt_direction), 1e-12)
    # Rank by positional opposition, then separation; yaw alone is not a counterexample.
    separation = np.linalg.norm(centers[:, :2] - gt_box[:2], axis=1)
    order = np.lexsort((-separation, cosine))
    occupied = quantize_occupied_points(voxel_points, voxel_size_cm)
    for row in order:
        box = candidate_from_bottom_center(centers[row], gt_box[3:6], yaws[row])
        _, visible = project_world_points(scene, place_box_to_corners(box))
        relation = placement_relation(scene, box, reference_corners)
        if not visible.all() or relation is None or relation == item.target_relation:
            continue
        if not is_collision_free(box, collision_context):
            continue
        stable, _ = compute_supported_and_stable(box, occupied, voxel_size_cm, downward_cm, upper_cm)
        if stable:
            return box
    raise ValueError(f"No supported, collision-free semantic counterexample for {item.sample_id}")


def save_semantic_comparison(
    scene: CanonicalScene,
    boxes: dict[str, np.ndarray],
    semantic_wrong_box: np.ndarray,
    path: Path,
) -> None:
    """Render source (blue), GT (green) and semantic counterexample (red), without reference box."""
    def draw_comparison(image: Image.Image, draw: ImageDraw.ImageDraw) -> None:
        # Shared translucent faces keep semantic boxes consistent with other slots.
        draw_contrast_world_corners(image, boxes["source_corners"], scene, (0, 100, 255))
        draw_contrast_place_box(image, boxes["gt_place_box"], scene, (0, 230, 118))
        draw_contrast_place_box(image, semantic_wrong_box, scene, (255, 23, 68))

    save_scene_overlay(scene, path, "Semantic Consistency", draw_comparison)


def choose_partial_support_box(
    item: Stage2IndexItem,
    scene: CanonicalScene,
    final_box: np.ndarray,
    reference_corners: np.ndarray,
    support_component: np.ndarray,
    voxel_points: np.ndarray,
    collision_context: dict[str, Any],
    *,
    voxel_size_cm: float,
    downward_cm: float,
    upper_cm: float,
) -> tuple[np.ndarray, float]:
    """Find a direction-correct, collision-free box with a partially unsupported bottom."""
    box = np.asarray(final_box, dtype=np.float64)
    reference_xy = reference_corners[:, :2].mean(axis=0)
    target_xy = box[:2]
    primary_direction = target_xy - reference_xy
    if np.linalg.norm(primary_direction) < 1e-6 and len(support_component):
        primary_direction = target_xy - support_component[:, :2].mean(axis=0)
    if np.linalg.norm(primary_direction) < 1e-6:
        primary_direction = np.asarray([1.0, 0.0], dtype=np.float64)
    primary_direction = primary_direction / np.linalg.norm(primary_direction)
    angles = np.linspace(0.0, 2.0 * np.pi, 16, endpoint=False)
    directions = [primary_direction] + [np.asarray([np.cos(angle), np.sin(angle)]) for angle in angles]
    distances = np.linspace(3.0, 34.0, 32)
    best_box = box.copy()
    best_coverage = 1.0
    best_partial_box = box.copy()
    best_partial_coverage = 1.0
    best_partial_score = float("inf")
    occupied = quantize_occupied_points(voxel_points, voxel_size_cm)
    for direction in directions:
        for distance in distances:
            candidate = box.copy()
            candidate[:2] = box[:2] + direction * float(distance)
            stable, coverage = compute_supported_and_stable(
                candidate,
                occupied,
                voxel_size_cm,
                downward_cm,
                upper_cm,
            )
            relation_ok = placement_relation(scene, candidate, reference_corners) == item.target_relation
            collision_ok = is_collision_free(candidate, collision_context)
            visible_ok = box_projection_clear(scene, candidate)
            if relation_ok and collision_ok and visible_ok and 0.05 < coverage < 0.98:
                score = abs(float(coverage) - 0.50)
                if score < best_partial_score:
                    best_partial_box = candidate
                    best_partial_coverage = float(coverage)
                    best_partial_score = score
            if relation_ok and collision_ok and coverage < best_coverage:
                best_box = candidate
                best_coverage = float(coverage)
    # Thin edge regions can fall between the radial search rays. Refine on the
    # full support grid before accepting a poor match or an unsupported fallback.
    if best_partial_score > 0.10:
        offsets = np.stack(np.meshgrid([-0.25, 0.0, 0.25], [-0.25, 0.0, 0.25]), axis=-1).reshape(-1, 2)
        centers = (support_component[:, None, :2] + offsets[None, :, :]).reshape(-1, 2)
        region_cache = {}
        for center in centers:
            candidate = box.copy()
            candidate[:2] = center
            if placement_relation(scene, candidate, reference_corners) != item.target_relation:
                continue
            if not box_projection_clear(scene, candidate) or not is_collision_free(candidate, collision_context):
                continue
            _, coverage = compute_supported_and_stable(
                candidate, occupied, voxel_size_cm, downward_cm, upper_cm, cache=region_cache,
            )
            if 0.05 < coverage < 0.98 and abs(coverage - 0.50) < best_partial_score:
                best_partial_box, best_partial_coverage = candidate, float(coverage)
                best_partial_score = abs(coverage - 0.50)
                if best_partial_score < 0.01:
                    break
    if best_partial_coverage < 1.0:
        return best_partial_box, best_partial_coverage
    return best_box, best_coverage


def points_inside_scene_obbs(points: np.ndarray, collision_context: dict[str, Any], margin_cm: float = 0.8) -> np.ndarray:
    """Mark sampled points that fall inside any scene object OBB."""
    pts = np.asarray(points, dtype=np.float64)
    inside = np.zeros(len(pts), dtype=bool)
    for obb in collision_context["object_obbs"]:
        axes = np.asarray(obb["axes"], dtype=np.float64)
        center = np.asarray(obb["center"], dtype=np.float64)
        half_extents = np.asarray(obb["half_extents"], dtype=np.float64) + float(margin_cm)
        local = (pts - center[None, :]) @ axes
        inside |= np.all(np.abs(local) <= half_extents[None, :], axis=1)
    return inside


def points_inside_yaw_box(points: np.ndarray, box: np.ndarray, margin_cm: float = 0.0) -> np.ndarray:
    """Mark world points inside one yaw-only box."""
    pts = np.asarray(points, dtype=np.float64)
    candidate = np.asarray(box, dtype=np.float64)
    yaw = float(candidate[6])
    c, s = math.cos(yaw), math.sin(yaw)
    rotation = np.asarray([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)
    local = (pts - candidate[None, :3]) @ rotation
    half_extents = candidate[3:6] * 0.5 + float(margin_cm)
    return np.all(np.abs(local) <= half_extents[None, :], axis=1)


def obb_surface_sample_points(obb: dict[str, Any], step_cm: float = 0.85) -> np.ndarray:
    """Sample the six surfaces of one scene OBB in world coordinates."""
    half = np.asarray(obb["half_extents"], dtype=np.float64)
    axes = np.asarray(obb["axes"], dtype=np.float64)
    center = np.asarray(obb["center"], dtype=np.float64)
    axis_values = [
        np.arange(-half[index], half[index] + 1e-6, float(step_cm))
        for index in range(3)
    ]
    surfaces = []
    for fixed_axis, fixed_sign in [(0, -1), (0, 1), (1, -1), (1, 1), (2, -1), (2, 1)]:
        free_axes = [axis for axis in range(3) if axis != fixed_axis]
        grid = np.stack(
            np.meshgrid(axis_values[free_axes[0]], axis_values[free_axes[1]], indexing="xy"),
            axis=-1,
        ).reshape(-1, 2)
        local = np.zeros((len(grid), 3), dtype=np.float64)
        local[:, fixed_axis] = fixed_sign * half[fixed_axis]
        local[:, free_axes] = grid
        surfaces.append(local @ axes.T + center[None, :])
    return np.vstack(surfaces)


def footprint_supported_mask(footprint_points: np.ndarray, support_component: np.ndarray) -> np.ndarray:
    """Mark footprint cells that belong to the selected support component."""
    if len(footprint_points) == 0 or len(support_component) == 0:
        return np.zeros(len(footprint_points), dtype=bool)
    support_keys = {tuple(np.round(point[:2], 3).tolist()) for point in support_component}
    return np.asarray(
        [tuple(np.round(point[:2], 3).tolist()) in support_keys for point in footprint_points],
        dtype=bool,
    )


def save_all_materials(args: argparse.Namespace, *, item: Stage2IndexItem | None = None) -> Path:
    """Export all material images and metadata for one sample."""
    if item is None:
        cfg = load_config(args.config)
        item = find_item(cfg, args.sample_id, args.object_id, args.reference_object_id, args.relation, args.cluster_id)
    output_dir = args.output_dir or default_output_dir(item)
    if args.output_dir is None and args.cluster_id is not None:
        output_dir = output_dir.with_name(f"{output_dir.name}__cluster_{item.cluster_id:03d}")
    output_dir.mkdir(parents=True, exist_ok=True)

    scene, voxel_points, voxel_colors = load_scene_and_points(item)
    prediction = load_prediction(args.predictions_json, item.sample_id, item.object_id)
    boxes = scene_box_arrays(scene, item, prediction)
    support_surface_points = load_support_surface_points(item)
    heatmap_points, heatmap_scores, heatmap_source = load_material_heatmap(item, prediction)
    collision_context = load_collision_context(item)
    visual_collision_context = canonical_scene_collision_context(scene)
    collision_box = make_collision_probe_box(boxes["final_box"], boxes["reference_corners"])
    if is_collision_free(collision_box, visual_collision_context):
        # A low or narrow reference may not intersect the candidate vertically.
        # Keep the same penetration rule and try the other real scene objects.
        for obj in scene.objects:
            corners = canonical_aabb_to_world_corners(obj.bbox3d_canonical, obj.pose_world)
            candidate = make_collision_probe_box(boxes["final_box"], corners)
            if box_projection_clear(scene, candidate) and not is_collision_free(candidate, visual_collision_context):
                collision_box = candidate
                break
    support_component, footprint_cells, support_coverage, support_z_band = support_topology(
        voxel_points,
        boxes["final_box"],
        float(args.voxel_size_cm),
        float(args.support_downward_cm),
        float(args.support_upper_cm),
    )
    semantic_wrong_box = choose_semantic_wrong_box(
        item,
        scene,
        boxes["reference_corners"],
        voxel_points,
        visual_collision_context,
        voxel_size_cm=float(args.voxel_size_cm),
        downward_cm=float(args.support_downward_cm),
        upper_cm=float(args.support_upper_cm),
    )
    partial_support_box, partial_support_coverage = choose_partial_support_box(
        item,
        scene,
        boxes["final_box"],
        boxes["reference_corners"],
        support_component,
        voxel_points,
        collision_context,
        voxel_size_cm=float(args.voxel_size_cm),
        downward_cm=float(args.support_downward_cm),
        upper_cm=float(args.support_upper_cm),
    )
    partial_support_component, partial_footprint_cells, _, _ = support_topology(
        voxel_points,
        partial_support_box,
        float(args.voxel_size_cm),
        float(args.support_downward_cm),
        float(args.support_upper_cm),
    )
    partial_footprint_supported = footprint_supported_mask(partial_footprint_cells, partial_support_component)
    final_collision = compute_collision_metrics(boxes["final_box"], collision_context)
    probe_collision = compute_collision_metrics(collision_box, visual_collision_context)
    bottom_samples = bottom_surface_sample_points(boxes["final_box"])
    sparse_boundary_samples = box_edge_sample_points(boxes["final_box"], points_per_edge=6)
    face_boundary_samples = boundary_sample_points(boxes["final_box"], step_cm=1.2)
    collision_face_samples = side_boundary_sample_groups(collision_box, step_cm=0.85)
    collision_face_masks = [
        points_inside_scene_obbs(face_samples, visual_collision_context, margin_cm=0.9)
        for face_samples in collision_face_samples
    ]
    collision_boundary_samples = np.vstack(collision_face_samples)
    collision_sample_mask = np.concatenate(collision_face_masks)
    probe_collision_points = collision_boundary_samples[collision_sample_mask]
    overlap_mask = points_inside_yaw_box(voxel_points, collision_box, margin_cm=0.0)
    overlap_mask &= points_inside_scene_obbs(voxel_points, visual_collision_context, margin_cm=0.0)
    collision_overlap_voxels = voxel_points[overlap_mask]
    collision_overlap_surfaces = []
    for obb in visual_collision_context["object_obbs"]:
        surface_points = obb_surface_sample_points(obb)
        inside_candidate = points_inside_yaw_box(surface_points, collision_box, margin_cm=0.0)
        collision_overlap_surfaces.append(surface_points[inside_candidate])
    collision_overlap_surface_points = (
        np.vstack(collision_overlap_surfaces)
        if any(len(points) for points in collision_overlap_surfaces)
        else np.empty((0, 3), dtype=np.float64)
    )
    collision_overlap_points = np.vstack([collision_overlap_voxels, collision_overlap_surface_points])

    paths = {spec.key: output_dir / material_filename(index, spec) for index, spec in enumerate(MATERIAL_SPECS, start=1)}
    save_language_instruction(item, scene, paths["language_instruction"])
    Image.fromarray(scene.rgb).save(paths["rgb_observation"])
    save_sparse_voxel_observation(scene, voxel_points, voxel_colors, paths["sparse_voxel_observation"], args.max_scene_points)
    save_rgb_box_image(scene, boxes["source_corners"], None, None, paths["source_object_grounding"], title="Source Object Grounding")
    save_heatmap_material(
        scene,
        boxes,
        support_surface_points,
        heatmap_points,
        heatmap_scores,
        paths["coarse_placement_heatmap"],
        args.max_heatmap_points,
    )
    colorbar_path = output_dir / HEATMAP_COLORBAR_FILENAME
    save_probability_colorbar(colorbar_path)
    voxel_heatmap_path = output_dir / HEATMAP_VOXEL_FILENAME
    save_sparse_voxel_probability_heatmap(
        scene,
        voxel_points,
        voxel_colors,
        support_surface_points,
        heatmap_points,
        heatmap_scores,
        voxel_heatmap_path,
        args.max_scene_points,
        float(args.voxel_size_cm),
    )

    def decoder_draw(image: Image.Image, draw: ImageDraw.ImageDraw) -> None:
        draw_contrast_world_corners(image, boxes["source_corners"], scene, hex_to_rgb(SOURCE_COLOR))
        decoder_boxes = prediction.decoder_boxes if prediction is not None and prediction.decoder_boxes else [boxes["gt_place_box"], boxes["final_box"]]
        for index, box in enumerate(decoder_boxes):
            color = (255, max(214 - min(index * 34, 130), 84), 0)
            draw_contrast_place_box(image, box, scene, color)
        draw_contrast_place_box(image, boxes["final_box"], scene, hex_to_rgb(PLACE_COLOR))

    save_scene_overlay(scene, paths["decoder_iterative_refinement"], "Decoder / Iterative Refinement", decoder_draw)

    def boundary_space_draw(image: Image.Image, draw: ImageDraw.ImageDraw) -> None:
        draw_projected_points(image, scene, sparse_boundary_samples, hex_to_rgb(COLLISION_COLOR), radius=4, alpha=205)
        draw_contrast_world_corners(image, boxes["source_corners"], scene, hex_to_rgb(SOURCE_COLOR))
        draw_contrast_place_box(image, boxes["final_box"], scene, hex_to_rgb(PREDICTED_COLOR))

    save_scene_overlay(scene, paths["boundary_space_reasoning"], "Boundary-Space Reasoning", boundary_space_draw)

    def support_space_draw(image: Image.Image, draw: ImageDraw.ImageDraw) -> None:
        draw_projected_points(image, scene, support_component, hex_to_rgb(SUPPORT_COLOR), radius=3, alpha=120, max_points=1200)
        draw_projected_points(image, scene, footprint_cells, hex_to_rgb(PREDICTED_COLOR), radius=4, alpha=210)
        draw_contrast_place_box(image, boxes["final_box"], scene, hex_to_rgb(PREDICTED_COLOR))

    save_scene_overlay(scene, paths["support_space_reasoning"], "Support-Space Reasoning", support_space_draw)
    save_rgb_box_image(scene, boxes["source_corners"], boxes["final_box"], None, paths["final_physically_feasible_placement"], title="Physically-Feasible 4-DoF Placement")

    save_semantic_comparison(scene, boxes, semantic_wrong_box, paths["semantic_consistency"])

    def bottom_sampling_draw(image: Image.Image, draw: ImageDraw.ImageDraw) -> None:
        draw_projected_points(image, scene, bottom_samples, (37, 99, 235), radius=5, alpha=215)
        draw_contrast_place_box(image, boxes["final_box"], scene, hex_to_rgb(PREDICTED_COLOR))

    save_scene_overlay(scene, paths["bottom_surface_sampling"], "Bottom-Surface Sampling", bottom_sampling_draw)

    # Preserve the original RGB pixels and projection scale while revealing any
    # off-frame part of the edge-support probe on a white extended canvas.
    probe_uv, _ = project_world_points(scene, place_box_to_corners(partial_support_box))
    left, top = np.maximum(0, np.ceil(12 - probe_uv.min(axis=0))).astype(int)
    right, bottom = np.maximum(0, np.ceil(probe_uv.max(axis=0) + 12 - [scene.camera.img_w, scene.camera.img_h])).astype(int)
    support_rgb = np.pad(scene.rgb, ((top, bottom), (left, right), (0, 0)), constant_values=255)
    support_scene = replace(scene, rgb=support_rgb, camera=replace(
        scene.camera, cx=scene.camera.cx + left, cy=scene.camera.cy + top,
        img_w=support_rgb.shape[1], img_h=support_rgb.shape[0],
    ))

    def connectivity_draw(image: Image.Image, draw: ImageDraw.ImageDraw) -> None:
        # Keep the full support component, but do not paint support over scene objects.
        visible_support_component = partial_support_component[
            ~points_inside_scene_obbs(partial_support_component, visual_collision_context, margin_cm=0.0)
        ]
        draw_projected_points(image, support_scene, visible_support_component, hex_to_rgb(SUPPORT_COLOR), radius=3, alpha=125, max_points=1200)
        draw_projected_points(image, support_scene, partial_footprint_cells[partial_footprint_supported], (249, 115, 22), radius=4, alpha=215)
        draw_projected_points(image, support_scene, partial_footprint_cells[~partial_footprint_supported], hex_to_rgb(COLLISION_COLOR), radius=6, alpha=235)
        draw_contrast_place_box(image, partial_support_box, support_scene, hex_to_rgb(SEMANTIC_PROBE_COLOR))

    save_scene_overlay(support_scene, paths["support_connectivity_check"], "Support Connectivity Check", connectivity_draw)

    def boundary_sampling_draw(image: Image.Image, draw: ImageDraw.ImageDraw) -> None:
        draw_projected_points(image, scene, face_boundary_samples, hex_to_rgb(COLLISION_COLOR), radius=4, alpha=195, max_points=360)
        draw_contrast_place_box(image, boxes["final_box"], scene, hex_to_rgb(PREDICTED_COLOR))

    save_scene_overlay(scene, paths["boundary_sampling"], "Boundary Sampling", boundary_sampling_draw)

    def clearance_draw(image: Image.Image, draw: ImageDraw.ImageDraw) -> None:
        collision_corners = place_box_to_corners(collision_box)
        for face_indices, face_mask in zip(SIDE_FACE_CORNERS, collision_face_masks):
            if np.any(face_mask):
                draw_projected_polygon(
                    image,
                    scene,
                    collision_corners[np.asarray(face_indices, dtype=np.int64)],
                    hex_to_rgb(COLLISION_COLOR),
                    alpha=118,
                )
        overlap_points = collision_overlap_points if len(collision_overlap_points) else probe_collision_points
        draw_collision_overlap_points(image, scene, overlap_points)
        draw_contrast_world_corners(image, boxes["source_corners"], scene, hex_to_rgb(SOURCE_COLOR))
        draw_contrast_place_box(image, collision_box, scene, hex_to_rgb(COLLISION_COLOR))

    save_scene_overlay(scene, paths["clearance_check"], "Clearance Check", clearance_draw)

    def collision_free_draw(image: Image.Image, draw: ImageDraw.ImageDraw) -> None:
        draw_contrast_world_corners(image, boxes["source_corners"], scene, hex_to_rgb(SOURCE_COLOR))
        draw_contrast_place_box(image, boxes["final_box"], scene, hex_to_rgb(PLACE_COLOR))

    save_scene_overlay(scene, paths["collision_free_placement"], "Collision-Free Placement", collision_free_draw)

    metadata = {
        "schema_version": "method_overview_materials/v1",
        "sample_id": item.sample_id,
        "item_id": item.item_id,
        "source_name": item.source_name,
        "object_id": item.object_id,
        "cluster_id": item.cluster_id,
        "reference_object_id": item.reference_object_id,
        "instruction": item.instruction,
        "target_relation": item.target_relation,
        "output_dir": os.fspath(output_dir.resolve()),
        "material_source": "prediction" if prediction is not None else "gt",
        "heatmap_source": heatmap_source,
        "support": {
            "image_padding_ltrb": [int(left), int(top), int(right), int(bottom)],
            "coverage": support_coverage,
            "partial_probe_coverage": partial_support_coverage,
            "z_band_voxel_keys": list(support_z_band),
            "component_cells": int(len(support_component)),
            "footprint_cells": int(len(footprint_cells)),
            "partial_probe_unsupported_cells": int(np.count_nonzero(~partial_footprint_supported)),
        },
        "collision": {
            "final": final_collision,
            "controlled_probe": probe_collision,
            "probe_collision_points": int(len(probe_collision_points)),
            "scene_overlap_voxels": int(len(collision_overlap_voxels)),
            "scene_overlap_surface_points": int(len(collision_overlap_surface_points)),
            "visual_context_object_count": int(len(visual_collision_context["object_obbs"])),
        },
        "boxes": {
            "source_box": boxes["source_box"].tolist(),
            "gt_place_box": boxes["gt_place_box"].tolist(),
            "final_box": boxes["final_box"].tolist(),
            "semantic_wrong_box": semantic_wrong_box.tolist(),
            "partial_support_probe_box": partial_support_box.tolist(),
            "collision_probe_box": collision_box.tolist(),
        },
        "materials": [
            {"index": index, "key": spec.key, "title": spec.title, "path": os.fspath(paths[spec.key].resolve())}
            for index, spec in enumerate(MATERIAL_SPECS, start=1)
        ],
        "extra_materials": [
            {
                "key": "coarse_placement_probability_colorbar",
                "title": "Coarse Placement Probability Colorbar",
                "path": os.fspath(colorbar_path.resolve()),
            },
            {
                "key": "sparse_voxel_probability_heatmap",
                "title": "Sparse Voxel Probability Heatmap",
                "path": os.fspath(voxel_heatmap_path.resolve()),
            }
        ],
    }
    metadata_path = output_dir / "materials_index.json"
    with metadata_path.open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, ensure_ascii=False)
    return metadata_path


def main() -> None:
    """CLI entrypoint."""
    args = parse_args()
    if args.voxel_size_cm <= 0.0:
        raise ValueError("--voxel-size-cm must be positive")
    if args.max_scene_points <= 0 or args.max_heatmap_points <= 0:
        raise ValueError("--max-scene-points and --max-heatmap-points must be positive")
    metadata_path = save_all_materials(args)
    print(f"Wrote material images and metadata to {metadata_path.parent}")
    print(f"Index: {metadata_path}")


if __name__ == "__main__":
    main()

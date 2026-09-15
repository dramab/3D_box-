#!/usr/bin/env python
"""Render a topology-aware box reasoning visualization for one placement sample.

Usage:
    python tools/render_topological_box_reasoner_vis.py

    python tools/render_topological_box_reasoner_vis.py \
        --dataset-dir data/hope \
        --sample-id hope__scene_0000__0005 \
        --predictions-json outputs/lc_bgplacenet_stage2_space_former_aligned_loss_48query_full_gt_guass_8_enriched/inference_custom_hope_scene_0000_0005_tomato_sauce_back_left_mustard/predictions.json \
        --output-dir outputs/visualizations/topological_box_reasoner
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

os.environ.setdefault("MPLCONFIGDIR", os.fspath(PROJECT_ROOT / "outputs" / ".matplotlib"))
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Rectangle

from src.annotation.free_bbox.io_utils import load_ply
from src.datasets.canonical import load_canonical_scene
from src.placement_metrics import (
    build_connected_support_region,
    footprint_voxel_keys,
    quantize_occupied_points,
    support_z_key_bounds,
)
from tools.infer_lc_bgplacenet_stage2 import (
    _find_scene_object as find_scene_object,
    draw_world_corners,
    place_box_to_corners,
    source_box_to_oriented_corners,
)
from tools.render_lc_bgplacenet_stage2_point_mask import (
    decode_heatmap_scores,
    heatmap_colors,
    normalize_scores_by_max,
)
from tools.render_topobox_teaser_gt import project_world_points


DEFAULT_PREDICTIONS_JSON = (
    PROJECT_ROOT
    / "outputs"
    / "lc_bgplacenet_stage2_space_former_aligned_loss_48query_full_gt_guass_8_enriched"
    / "inference_custom_hope_scene_0000_0005_tomato_sauce_back_left_mustard"
    / "predictions.json"
)
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "visualizations" / "topological_box_reasoner"

SOURCE_COLOR = (225, 40, 74)
FINAL_COLOR = (109, 70, 168)
PROPOSAL_COLOR = "#2563EB"
SUPPORT_COLOR = "#16A34A"
COLLISION_COLOR = "#DC2626"
FLOAT_COLOR = "#F97316"
TEXT_COLOR = (17, 24, 39)


@dataclass(frozen=True)
class PredictionRecord:
    """One resolved prediction entry used by the visualization."""

    item_id: str
    sample_id: str
    object_id: str
    instruction: str
    source_box: np.ndarray
    final_box: np.ndarray
    gt_box: np.ndarray
    pred_heatmap_ply: Path
    decoder_boxes: list[np.ndarray]


@dataclass(frozen=True)
class SupportTopology:
    """Connected support data derived with the benchmark support definition."""

    component_xy: np.ndarray
    footprint_xy: np.ndarray
    support_coverage: float
    z_band: tuple[int, int]
    component_id: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render Topological Box Reasoner visualization.")
    parser.add_argument("--dataset-dir", type=Path, default=PROJECT_ROOT / "data" / "hope")
    parser.add_argument("--sample-id", default="hope__scene_0000__0005")
    parser.add_argument("--predictions-json", type=Path, default=DEFAULT_PREDICTIONS_JSON)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--voxel-size-cm", type=float, default=1.0)
    parser.add_argument("--support-downward-cm", type=float, default=3.0)
    parser.add_argument("--support-upper-cm", type=float, default=1.0)
    parser.add_argument("--max-scene-points", type=int, default=5500)
    parser.add_argument("--max-heatmap-points", type=int, default=1400)
    return parser.parse_args()


def load_font(size: int, *, bold: bool = False) -> ImageFont.FreeTypeFont:
    suffix = "-Bold.ttf" if bold else ".ttf"
    path = Path("/usr/share/fonts/truetype/dejavu") / f"DejaVuSans{suffix}"
    return ImageFont.truetype(os.fspath(path), size=size)


def load_prediction(path: Path, sample_id: str) -> PredictionRecord:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    records = payload if isinstance(payload, list) else [payload]
    record = next((row for row in records if str(row.get("sample_id")) == sample_id), None)
    if record is None:
        raise ValueError(f"{path} does not contain sample_id={sample_id}")

    decoder_boxes = [
        np.asarray(stage["placements"][0]["box"], dtype=np.float64)
        for stage in record.get("decoder_stages", [])
        if stage.get("placements")
    ]
    return PredictionRecord(
        item_id=str(record["item_id"]),
        sample_id=str(record["sample_id"]),
        object_id=str(record["object_id"]),
        instruction=str(record["instruction"]),
        source_box=np.asarray(record["source_box"], dtype=np.float64),
        final_box=np.asarray(record["place_box"], dtype=np.float64),
        gt_box=np.asarray(record.get("place_box_gt", record["place_box"]), dtype=np.float64),
        pred_heatmap_ply=PROJECT_ROOT / str(record["pred_heatmap_ply"]),
        decoder_boxes=decoder_boxes,
    )


def object_corners(scene, object_id: str) -> np.ndarray:
    obj = find_scene_object(scene, object_id)
    return canonical_aabb_to_world_corners(obj.bbox3d_canonical, obj.pose_world)


def canonical_aabb_to_world_corners(aabb: np.ndarray, pose_world: np.ndarray) -> np.ndarray:
    """Transform one canonical object AABB into its eight world-space corners."""
    box = np.asarray(aabb, dtype=np.float64)
    pose = np.asarray(pose_world, dtype=np.float64)
    lower, upper = box[:3], box[3:6]
    axes_values = [np.asarray([lower[idx], upper[idx]], dtype=np.float64) for idx in range(3)]
    local = np.stack(np.meshgrid(*axes_values, indexing="ij"), axis=-1).reshape(-1, 3)
    homo = np.column_stack([local, np.ones(len(local), dtype=np.float64)])
    return (pose @ homo.T).T[:, :3]


def topdown_polygon(corners: np.ndarray) -> np.ndarray:
    pts = np.asarray(corners, dtype=np.float64)[:, :2]
    center = pts.mean(axis=0)
    angles = np.arctan2(pts[:, 1] - center[1], pts[:, 0] - center[0])
    return pts[np.argsort(angles)]


def box_footprint(box: np.ndarray) -> np.ndarray:
    return topdown_polygon(place_box_to_corners(np.asarray(box, dtype=np.float64)))


def inside_yawed_box(points: np.ndarray, box: np.ndarray, margin_cm: float = 0.0) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64)
    box = np.asarray(box, dtype=np.float64)
    center = box[:3]
    dims = np.maximum(box[3:6] + 2.0 * float(margin_cm), 1e-6)
    yaw = float(box[6])
    c, s = np.cos(yaw), np.sin(yaw)
    axes = np.asarray([[c, s, 0.0], [-s, c, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)
    local = (points - center[None, :]) @ axes.T
    return np.all(np.abs(local) <= 0.5 * dims[None, :], axis=1)


def compute_support_topology(
    voxel_points: np.ndarray,
    box: np.ndarray,
    *,
    voxel_size_cm: float,
    downward_cm: float,
    upper_cm: float,
) -> SupportTopology:
    occupied_keys = quantize_occupied_points(voxel_points, voxel_size_cm)
    bottom_z = float(box[2] - 0.5 * box[5])
    z_min, z_max = support_z_key_bounds(bottom_z, downward_cm, upper_cm, voxel_size_cm)
    region = build_connected_support_region(occupied_keys, z_min, z_max)
    center_key = np.floor(box[:2] / voxel_size_cm).astype(np.int64)
    center_local = center_key - region.origin_xy
    center_inside = np.all(center_local >= 0) and np.all(center_local < np.asarray(region.labels.shape))
    component_id = int(region.labels[tuple(center_local)]) if center_inside else 0

    footprint = footprint_voxel_keys(box, voxel_size_cm)
    local = footprint - region.origin_xy
    in_bounds = np.all(local >= 0, axis=1) & np.all(local < np.asarray(region.labels.shape), axis=1)
    covered = np.zeros(len(footprint), dtype=bool)
    valid = local[in_bounds]
    if component_id:
        covered[in_bounds] = region.labels[valid[:, 0], valid[:, 1]] == component_id
    support_coverage = float(covered.mean()) if len(covered) else 0.0

    component_indices = np.argwhere(region.labels == component_id) if component_id else np.empty((0, 2), dtype=np.int64)
    component_keys = component_indices + region.origin_xy
    component_xy = (component_keys.astype(np.float64) + 0.5) * voxel_size_cm
    footprint_xy = (footprint.astype(np.float64) + 0.5) * voxel_size_cm
    return SupportTopology(
        component_xy=component_xy,
        footprint_xy=footprint_xy,
        support_coverage=support_coverage,
        z_band=(z_min, z_max),
        component_id=component_id,
    )


def load_heatmap_points(prediction: PredictionRecord) -> tuple[np.ndarray, np.ndarray]:
    heatmap_points, heatmap_color_data = load_ply(prediction.pred_heatmap_ply)
    scores = normalize_scores_by_max(decode_heatmap_scores(heatmap_color_data))
    return heatmap_points, scores


def stride_indices(length: int, maximum: int) -> np.ndarray:
    if length <= maximum:
        return np.arange(length)
    return np.linspace(0, length - 1, maximum, dtype=np.int64)


def set_focus(axis: plt.Axes, arrays: list[np.ndarray], margin_cm: float = 10.0) -> None:
    points = np.vstack([arr[:, :2] for arr in arrays if len(arr)])
    lower = points.min(axis=0)
    upper = points.max(axis=0)
    span = np.maximum(upper - lower, 1.0)
    margin = np.maximum(span * 0.16, margin_cm)
    axis.set_xlim(lower[0] - margin[0], upper[0] + margin[0])
    axis.set_ylim(lower[1] - margin[1], upper[1] + margin[1])


def style_axis(axis: plt.Axes, title: str, subtitle: str | None = None) -> None:
    axis.set_aspect("equal")
    axis.set_xticks([])
    axis.set_yticks([])
    axis.set_title(title, fontsize=13, fontweight="bold", loc="left", pad=9)
    if subtitle:
        axis.text(0.0, 0.985, subtitle, transform=axis.transAxes, va="top", ha="left", fontsize=8.5, color="#475569")
    for spine in axis.spines.values():
        spine.set_visible(False)


def draw_scene_footprints(axis: plt.Axes, scene, source_object_id: str) -> None:
    for obj in scene.objects:
        corners = canonical_aabb_to_world_corners(obj.bbox3d_canonical, obj.pose_world)
        face = "#E5E7EB"
        edge = "#94A3B8"
        alpha = 0.42
        if obj.obj_id == source_object_id:
            face = "#FFE4E6"
            edge = "#E11D48"
            alpha = 0.78
        axis.add_patch(
            Polygon(topdown_polygon(corners), closed=True, facecolor=face, edgecolor=edge, linewidth=1.0, alpha=alpha)
        )


def draw_box_polygon(
    axis: plt.Axes,
    box: np.ndarray,
    *,
    edge: str,
    face: str | None = None,
    linewidth: float = 2.0,
    linestyle: str = "-",
    label: str | None = None,
    alpha: float = 0.18,
) -> None:
    poly = box_footprint(box)
    axis.add_patch(
        Polygon(poly, closed=True, facecolor=face or edge, edgecolor=edge, linewidth=linewidth, linestyle=linestyle, alpha=alpha)
    )
    axis.add_patch(Polygon(poly, closed=True, fill=False, edgecolor=edge, linewidth=linewidth, linestyle=linestyle))
    if label:
        center = poly.mean(axis=0)
        axis.text(center[0], center[1], label, color=edge, fontsize=8, fontweight="bold", ha="center", va="center")


def render_proposal_panel(
    axis: plt.Axes,
    scene,
    prediction: PredictionRecord,
    voxel_points: np.ndarray,
    voxel_colors: np.ndarray,
    heatmap_points: np.ndarray,
    heatmap_scores: np.ndarray,
    max_scene_points: int,
    max_heatmap_points: int,
) -> None:
    scene_idx = stride_indices(len(voxel_points), max_scene_points)
    heat_idx = stride_indices(len(heatmap_points), max_heatmap_points)
    axis.scatter(
        voxel_points[scene_idx, 0],
        voxel_points[scene_idx, 1],
        s=1.5,
        c=np.asarray(voxel_colors[scene_idx]) / 255.0,
        alpha=0.22,
        linewidths=0.0,
    )
    colors = heatmap_colors(heatmap_scores[heat_idx])
    axis.scatter(
        heatmap_points[heat_idx, 0],
        heatmap_points[heat_idx, 1],
        s=11.0,
        c=colors,
        alpha=0.86,
        linewidths=0.0,
    )
    draw_scene_footprints(axis, scene, prediction.object_id)
    draw_box_polygon(axis, prediction.final_box, edge="#7C3AED", linewidth=2.0, label="Top-1")
    focus_mask = heatmap_scores >= max(0.35, float(heatmap_scores.max(initial=0.0)) * 0.65)
    focus_points = heatmap_points[focus_mask] if np.any(focus_mask) else heatmap_points
    set_focus(
        axis,
        [focus_points, object_corners(scene, prediction.object_id), place_box_to_corners(prediction.final_box)],
        margin_cm=9.0,
    )
    style_axis(axis, "1. Proposal Field", "coarse placement scores guide candidate boxes")


def render_candidate_panel(axis: plt.Axes, scene, prediction: PredictionRecord, float_box: np.ndarray, collision_box: np.ndarray) -> None:
    draw_scene_footprints(axis, scene, prediction.object_id)
    for index, box in enumerate(prediction.decoder_boxes):
        alpha = 0.10 + 0.045 * index
        draw_box_polygon(axis, box, edge=PROPOSAL_COLOR, linewidth=1.4, linestyle="--", alpha=alpha)
    draw_box_polygon(axis, float_box, edge=FLOAT_COLOR, linewidth=2.0, linestyle="--", label="no support", alpha=0.08)
    draw_box_polygon(axis, collision_box, edge=COLLISION_COLOR, linewidth=2.0, linestyle="--", label="collision", alpha=0.08)
    draw_box_polygon(axis, prediction.final_box, edge="#7C3AED", linewidth=2.4, label="refined", alpha=0.16)
    set_focus(
        axis,
        [
            object_corners(scene, prediction.object_id),
            place_box_to_corners(prediction.final_box),
            place_box_to_corners(float_box),
            place_box_to_corners(collision_box),
        ],
        margin_cm=8.0,
    )
    style_axis(axis, "2. Candidate Boxes", "decoder stages plus topology probes")


def render_topology_panel(
    axis: plt.Axes,
    voxel_points: np.ndarray,
    prediction: PredictionRecord,
    topology: SupportTopology,
    collision_box: np.ndarray,
) -> None:
    local_boxes = np.vstack([place_box_to_corners(prediction.final_box), place_box_to_corners(collision_box)])
    local_min = local_boxes[:, :2].min(axis=0) - 7.0
    local_max = local_boxes[:, :2].max(axis=0) + 7.0
    local_mask = ((voxel_points[:, :2] >= local_min) & (voxel_points[:, :2] <= local_max)).all(axis=1)
    local_points = voxel_points[local_mask]
    # Surface point clouds rarely lie strictly inside an object volume. A small
    # shell margin makes the collision evidence visible around the probe box.
    collision_points = local_points[inside_yawed_box(local_points, collision_box, margin_cm=1.8)]

    axis.scatter(local_points[:, 0], local_points[:, 1], s=3.0, c="#CBD5E1", alpha=0.22, linewidths=0.0)
    if len(topology.component_xy):
        axis.scatter(
            topology.component_xy[:, 0],
            topology.component_xy[:, 1],
            s=22.0,
            marker="s",
            c=SUPPORT_COLOR,
            alpha=0.45,
            linewidths=0.0,
        )
    if len(topology.footprint_xy):
        axis.scatter(
            topology.footprint_xy[:, 0],
            topology.footprint_xy[:, 1],
            s=16.0,
            marker="s",
            facecolors="none",
            edgecolors="#7C3AED",
            linewidths=0.8,
            alpha=0.95,
        )
    if len(collision_points):
        axis.scatter(collision_points[:, 0], collision_points[:, 1], s=34.0, c=COLLISION_COLOR, alpha=0.95, linewidths=0.0)
    draw_box_polygon(axis, collision_box, edge=COLLISION_COLOR, linewidth=2.0, linestyle="--", alpha=0.04)
    draw_box_polygon(axis, prediction.final_box, edge="#7C3AED", linewidth=2.4, alpha=0.05)
    set_focus(axis, [local_boxes, topology.component_xy, topology.footprint_xy], margin_cm=3.0)
    axis.text(
        0.02,
        0.06,
        f"support coverage: {topology.support_coverage * 100:.1f}%\nred voxels are excluded by boundary separation",
        transform=axis.transAxes,
        fontsize=8.5,
        color="#334155",
        va="bottom",
        ha="left",
    )
    style_axis(axis, "3. Box-Scene Topology", "bottom connectivity and obstacle separation")


def render_refined_rgb_panel(axis: plt.Axes, scene, prediction: PredictionRecord, collision_box: np.ndarray) -> None:
    image = Image.fromarray(scene.rgb).convert("RGB")
    draw = ImageDraw.Draw(image)
    source_obj = find_scene_object(scene, prediction.object_id)
    draw_world_corners(
        draw,
        source_box_to_oriented_corners(prediction.source_box, np.asarray(source_obj.pose_world)),
        scene,
        SOURCE_COLOR,
        4,
        "",
    )
    draw_world_corners(draw, place_box_to_corners(collision_box), scene, (220, 38, 38), 3, "")
    draw_world_corners(draw, place_box_to_corners(prediction.final_box), scene, FINAL_COLOR, 5, "")

    overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
    overlay_draw = ImageDraw.Draw(overlay)
    for point, color in [
        (prediction.final_box[:3], (109, 70, 168, 165)),
        (collision_box[:3], (220, 38, 38, 120)),
    ]:
        uv = project_world_points(scene, np.asarray([point], dtype=np.float64))[0]
        if np.isfinite(uv[0]):
            radius = 8
            overlay_draw.ellipse((uv[0] - radius, uv[1] - radius, uv[0] + radius, uv[1] + radius), fill=color)
    image = Image.alpha_composite(image.convert("RGBA"), overlay).convert("RGB")
    axis.imshow(image)
    axis.axis("off")
    axis.set_title("4. Refined Placement", fontsize=13, fontweight="bold", loc="left", pad=9)
    axis.text(
        0.02,
        0.97,
        "final box satisfies support and avoids collision",
        transform=axis.transAxes,
        va="top",
        ha="left",
        fontsize=8.5,
        color="#475569",
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.82, "pad": 2.5},
    )


def add_legend(fig: plt.Figure) -> None:
    legend_items = [
        ("proposal field", PROPOSAL_COLOR),
        ("refined box", "#7C3AED"),
        ("support component", SUPPORT_COLOR),
        ("collision voxels/probe", COLLISION_COLOR),
        ("source object", "#E11D48"),
    ]
    x = 0.055
    y = 0.035
    for label, color in legend_items:
        fig.patches.append(Rectangle((x, y), 0.011, 0.017, transform=fig.transFigure, color=color, clip_on=False))
        fig.text(x + 0.015, y - 0.001, label, fontsize=9, color="#334155", va="bottom")
        x += 0.15


def save_figure(
    output_dir: Path,
    scene,
    prediction: PredictionRecord,
    voxel_points: np.ndarray,
    voxel_colors: np.ndarray,
    heatmap_points: np.ndarray,
    heatmap_scores: np.ndarray,
    topology: SupportTopology,
    float_box: np.ndarray,
    collision_box: np.ndarray,
    args: argparse.Namespace,
) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    figure_path = output_dir / f"{prediction.sample_id}_topological_box_reasoner.png"
    pdf_path = figure_path.with_suffix(".pdf")

    fig = plt.figure(figsize=(16.2, 5.4), facecolor="white")
    grid = fig.add_gridspec(1, 4, left=0.035, right=0.985, top=0.80, bottom=0.11, wspace=0.075)
    axes = [fig.add_subplot(grid[0, idx]) for idx in range(3)]
    rgb_axis = fig.add_subplot(grid[0, 3])
    render_proposal_panel(
        axes[0],
        scene,
        prediction,
        voxel_points,
        voxel_colors,
        heatmap_points,
        heatmap_scores,
        args.max_scene_points,
        args.max_heatmap_points,
    )
    render_candidate_panel(axes[1], scene, prediction, float_box, collision_box)
    render_topology_panel(axes[2], voxel_points, prediction, topology, collision_box)
    render_refined_rgb_panel(rgb_axis, scene, prediction, collision_box)

    title = "Topological Box Reasoner: reasoning around candidate box boundaries"
    fig.text(0.035, 0.94, title, fontsize=18, fontweight="bold", color="#111827", ha="left")
    fig.text(0.035, 0.885, prediction.instruction, fontsize=11, color="#334155", ha="left")
    add_legend(fig)
    fig.savefig(figure_path, dpi=220, facecolor="white")
    fig.savefig(pdf_path, dpi=220, facecolor="white")
    plt.close(fig)
    return figure_path, pdf_path


def save_metadata(
    output_dir: Path,
    prediction: PredictionRecord,
    topology: SupportTopology,
    float_box: np.ndarray,
    collision_box: np.ndarray,
    figure_path: Path,
    pdf_path: Path,
    args: argparse.Namespace,
) -> Path:
    metadata = {
        "schema_version": "topological_box_reasoner_visualization/v1",
        "sample_id": prediction.sample_id,
        "item_id": prediction.item_id,
        "object_id": prediction.object_id,
        "instruction": prediction.instruction,
        "figure_png": os.fspath(figure_path),
        "figure_pdf": os.fspath(pdf_path),
        "inputs": {
            "dataset_dir": os.fspath(args.dataset_dir),
            "predictions_json": os.fspath(args.predictions_json),
            "pred_heatmap_ply": os.fspath(prediction.pred_heatmap_ply),
        },
        "boxes": {
            "source_box": prediction.source_box.tolist(),
            "final_box": prediction.final_box.tolist(),
            "gt_box_reference": prediction.gt_box.tolist(),
            "floating_probe_box": float_box.tolist(),
            "collision_probe_box": collision_box.tolist(),
        },
        "topology": {
            "support_component_id": topology.component_id,
            "support_z_band_voxel_keys": list(topology.z_band),
            "support_coverage": topology.support_coverage,
            "component_cells": int(len(topology.component_xy)),
            "footprint_cells": int(len(topology.footprint_xy)),
            "note": "Support component is computed with the benchmark 3x3 closing, hole filling, and 8-connected labeling rule. Probe boxes are controlled perturbations for visualization.",
        },
    }
    metadata_path = output_dir / f"{prediction.sample_id}_topological_box_reasoner_metadata.json"
    with metadata_path.open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, ensure_ascii=False)
    return metadata_path


def make_probe_boxes(scene, prediction: PredictionRecord) -> tuple[np.ndarray, np.ndarray]:
    float_box = prediction.final_box.copy()
    float_box[0] += max(5.0, float(prediction.final_box[3]) * 0.85)
    float_box[1] -= max(3.0, float(prediction.final_box[4]) * 0.55)
    float_box[2] += max(8.0, float(prediction.final_box[5]) * 0.85)

    collision_box = prediction.final_box.copy()
    source_center = object_corners(scene, prediction.object_id).mean(axis=0)
    collision_box[:3] = source_center
    return float_box, collision_box


def main() -> None:
    args = parse_args()
    if args.voxel_size_cm <= 0.0:
        raise ValueError("--voxel-size-cm must be positive")

    prediction = load_prediction(args.predictions_json, args.sample_id)
    sample_path = args.dataset_dir / "samples" / f"{args.sample_id}.json"
    scene = load_canonical_scene(sample_path, dataset_root=args.dataset_dir)
    voxel_points, voxel_colors = load_ply(scene.voxel_point_cloud_path)
    heatmap_points, heatmap_scores = load_heatmap_points(prediction)
    topology = compute_support_topology(
        voxel_points,
        prediction.final_box,
        voxel_size_cm=float(args.voxel_size_cm),
        downward_cm=float(args.support_downward_cm),
        upper_cm=float(args.support_upper_cm),
    )
    float_box, collision_box = make_probe_boxes(scene, prediction)
    figure_path, pdf_path = save_figure(
        args.output_dir,
        scene,
        prediction,
        voxel_points,
        voxel_colors,
        heatmap_points,
        heatmap_scores,
        topology,
        float_box,
        collision_box,
        args,
    )
    metadata_path = save_metadata(args.output_dir, prediction, topology, float_box, collision_box, figure_path, pdf_path, args)
    print(f"Wrote {figure_path}")
    print(f"Wrote {pdf_path}")
    print(f"Wrote {metadata_path}")


if __name__ == "__main__":
    main()

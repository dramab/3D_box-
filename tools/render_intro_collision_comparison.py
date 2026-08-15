#!/usr/bin/env python
"""Render a camera-aligned Introduction comparison for one placement case.

The RoboBrain 3D candidate is visualization-only: it is reconstructed from
the predicted 2D point using the source object's observed geometry.

使用示例:
    python tools/render_intro_collision_comparison.py
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

os.environ.setdefault("MPLCONFIGDIR", os.fspath(PROJECT_ROOT / "outputs" / ".matplotlib"))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Polygon

from baselines.robobrain2_5.run_zero_shot_test import (
    DEFAULT_CONFIG,
    load_scene,
    load_sources,
    load_test_items,
)
from baselines.robobrain2_5.zero_shot_visualization import (
    BOX_EDGES,
    Camera,
    aabb_corners,
    convex_hull_xy,
    project_world,
    transform_points,
    upright_box_corners,
)
from src.training.lc_bgplacenet_stage2 import build_collision_context, compute_collision_metrics


DEFAULT_ITEM_ID = "omni__label_041919__0e79bec87e9a05d1"
DEFAULT_ROBOBRAIN_RESULTS = PROJECT_ROOT / "outputs/robobrain2_5_zero_shot_test_official_prompt_gt/predictions.jsonl"
DEFAULT_OURS_RESULTS = PROJECT_ROOT / (
    "outputs/lc_bgplacenet_stage2_space_former_aligned_loss_48query_full_gt_guass_8/"
    "inference_stage2_test_multi_stage/predictions.json"
)
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs/introduction_collision_comparison"

SOURCE_COLOR = "#0891B2"
ROBOBRAIN_COLOR = "#DC2626"
OURS_COLOR = "#2563EB"
REFERENCE_COLOR = "#64748B"
SCENE_FILL = "#F1F5F9"
SCENE_EDGE = "#CBD5E1"
COLLISION_FILL = "#991B1B"

BOX_FACES = (
    (0, 1, 3, 2),
    (4, 5, 7, 6),
    (0, 1, 5, 4),
    (2, 3, 7, 6),
    (0, 2, 6, 4),
    (1, 3, 7, 5),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render the Introduction collision comparison figure.")
    parser.add_argument("--item-id", default=DEFAULT_ITEM_ID)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--robobrain-results", type=Path, default=DEFAULT_ROBOBRAIN_RESULTS)
    parser.add_argument("--ours-results", type=Path, default=DEFAULT_OURS_RESULTS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def load_jsonl_record(path: Path, item_id: str) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if str(row["item_id"]) == item_id:
                return row
    raise KeyError(f"Item {item_id!r} not found in {path}")


def load_json_record(path: Path, item_id: str) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        rows = json.load(handle)
    try:
        return next(row for row in rows if str(row["item_id"]) == item_id)
    except StopIteration as error:
        raise KeyError(f"Item {item_id!r} not found in {path}") from error


def box_to_corners(box: list[float] | np.ndarray) -> np.ndarray:
    box = np.asarray(box, dtype=np.float64)
    return upright_box_corners(box[:3], box[3:6], float(box[6]))


def object_corners(obj: dict[str, Any]) -> np.ndarray:
    return transform_points(aabb_corners(np.asarray(obj["bbox3d_canonical"])), np.asarray(obj["pose_world"]))


def camera_aligned_basis(camera: Camera) -> tuple[np.ndarray, np.ndarray]:
    """Return horizontal right/forward axes matching the RGB camera view."""
    # 切片会返回相机外参的视图；必须复制，避免归一化时破坏后续 RGB 投影。
    forward = np.asarray(camera.e_c2w[:2, 2], dtype=np.float64).copy()
    norm = float(np.linalg.norm(forward))
    if norm < 1e-8:
        raise ValueError("Camera optical axis has no stable horizontal projection")
    forward /= norm
    right = np.array([forward[1], -forward[0]], dtype=np.float64)
    camera_right = np.asarray(camera.e_c2w[:2, 0], dtype=np.float64)
    if float(right @ camera_right) < 0.0:
        right *= -1.0
    return right, forward


def align_xy(points_xy: np.ndarray, camera: Camera) -> np.ndarray:
    right, forward = camera_aligned_basis(camera)
    points = np.asarray(points_xy, dtype=np.float64)
    return np.column_stack((points @ right, points @ forward))


def draw_rgb_box(
    ax: plt.Axes,
    corners: np.ndarray,
    camera: Camera,
    color: str,
    label: str,
    *,
    alpha: float,
    linestyle: str = "-",
) -> None:
    uv, depth = project_world(corners, camera)
    visible_faces = [face for face in BOX_FACES if np.all(depth[list(face)] > 0.0)]
    visible_faces.sort(key=lambda face: float(depth[list(face)].mean()), reverse=True)
    for face in visible_faces:
        ax.add_patch(Polygon(uv[list(face)], closed=True, facecolor=color, edgecolor="none", alpha=alpha))
    for start, end in BOX_EDGES:
        if depth[start] > 0.0 and depth[end] > 0.0:
            ax.plot(
                uv[[start, end], 0],
                uv[[start, end], 1],
                color=color,
                linewidth=1.8,
                linestyle=linestyle,
                solid_capstyle="round",
            )
    visible = depth > 0.0
    if np.any(visible):
        visible_uv = uv[visible]
        box_center = visible_uv.mean(axis=0)
        label_xy = visible_uv.min(axis=0) + np.array([4.0, 4.0])
        ax.annotate(
            label,
            box_center,
            xytext=label_xy,
            textcoords="data",
            ha="left",
            va="top",
            fontsize=8,
            fontweight="semibold",
            color=color,
            bbox={"boxstyle": "round,pad=0.2", "facecolor": "white", "edgecolor": color, "alpha": 0.92},
            arrowprops={"arrowstyle": "-", "color": color, "linewidth": 0.8, "shrinkA": 2, "shrinkB": 2},
        )


def draw_status_chip(ax: plt.Axes, text: str, color: str) -> None:
    ax.text(
        0.98,
        0.96,
        text,
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=9,
        fontweight="bold",
        color="white",
        bbox={"boxstyle": "round,pad=0.35", "facecolor": color, "edgecolor": "white", "linewidth": 0.8},
    )


def configure_rgb_axis(ax: plt.Axes, rgb: np.ndarray, title: str, title_color: str) -> None:
    ax.imshow(rgb)
    ax.set_xlim(0, rgb.shape[1])
    ax.set_ylim(rgb.shape[0], 0)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_color("#E2E8F0")
        spine.set_linewidth(1.0)
    ax.set_title(title, color=title_color, fontsize=12, fontweight="bold", pad=7)


def add_card_background(ax: plt.Axes) -> None:
    ax.add_patch(
        FancyBboxPatch(
            (-0.015, -0.02),
            1.03,
            1.04,
            transform=ax.transAxes,
            boxstyle="round,pad=0.012,rounding_size=0.025",
            facecolor="#F8FAFC",
            edgecolor="#E2E8F0",
            linewidth=1.0,
            clip_on=False,
            zorder=-20,
        )
    )


def draw_camera_orientation(ax: plt.Axes) -> None:
    ax.annotate(
        "Camera forward",
        xy=(0.50, 0.17),
        xytext=(0.50, 0.045),
        xycoords="axes fraction",
        textcoords="axes fraction",
        ha="center",
        va="bottom",
        fontsize=8,
        color="#475569",
        arrowprops={"arrowstyle": "-|>", "color": "#475569", "linewidth": 1.2},
    )


def draw_topdown(
    ax: plt.Axes,
    scene_objects: list[dict[str, Any]],
    camera: Camera,
    source_id: str,
    candidate_corners: np.ndarray,
    candidate_color: str,
    reference_ids: set[str],
    colliding_ids: set[str],
    limits: tuple[float, float, float, float],
    status: str,
) -> None:
    polygons = {
        str(obj["obj_id"]): align_xy(convex_hull_xy(object_corners(obj)[:, :2]), camera)
        for obj in scene_objects
    }
    source_polygon = polygons[source_id]
    candidate_polygon = align_xy(convex_hull_xy(candidate_corners[:, :2]), camera)

    add_card_background(ax)
    for obj_id, polygon in polygons.items():
        if obj_id == source_id:
            continue
        is_reference = obj_id in reference_ids
        facecolor = "#F8FAFC" if is_reference else SCENE_FILL
        edgecolor = REFERENCE_COLOR if is_reference else SCENE_EDGE
        ax.add_patch(Polygon(polygon, closed=True, facecolor=facecolor, edgecolor=edgecolor, linewidth=1.0))
        if is_reference:
            center = polygon.mean(axis=0)
            ax.text(*center, "Pie", ha="center", va="center", fontsize=8, color=REFERENCE_COLOR, fontweight="bold")

    ax.add_patch(
        Polygon(
            source_polygon,
            closed=True,
            facecolor=SOURCE_COLOR,
            edgecolor=SOURCE_COLOR,
            linewidth=1.7,
            linestyle=(0, (4, 2)),
            alpha=0.16,
        )
    )
    candidate_patch = Polygon(
        candidate_polygon,
        closed=True,
        facecolor=candidate_color,
        edgecolor=candidate_color,
        linewidth=2.2,
        alpha=0.28,
    )
    ax.add_patch(candidate_patch)

    for obj_id in colliding_ids:
        clip_polygon = Polygon(polygons[obj_id], closed=True, transform=ax.transData)
        overlap = Polygon(
            candidate_polygon,
            closed=True,
            facecolor=COLLISION_FILL,
            edgecolor="none",
            alpha=0.78,
        )
        overlap.set_clip_path(clip_polygon)
        ax.add_patch(overlap)

    source_center = source_polygon.mean(axis=0)
    candidate_center = candidate_polygon.mean(axis=0)
    ax.add_patch(
        FancyArrowPatch(
            source_center,
            candidate_center,
            connectionstyle="arc3,rad=-0.16",
            arrowstyle="-|>",
            mutation_scale=12,
            color=candidate_color,
            linewidth=1.5,
            alpha=0.9,
        )
    )
    movement = candidate_center - source_center
    label_offset = max((limits[1] - limits[0]) * 0.035, 0.8)
    horizontal_sign = -1.0 if movement[0] < 0.0 else 1.0
    ax.text(
        *(source_center - np.array([horizontal_sign * label_offset, 0.0])),
        "Source",
        ha="left" if horizontal_sign < 0.0 else "right",
        va="center",
        fontsize=8,
        color="#0E7490",
        fontweight="bold",
    )
    ax.text(
        *(candidate_center + np.array([horizontal_sign * label_offset, 0.0])),
        "Prediction",
        ha="right" if horizontal_sign < 0.0 else "left",
        va="center",
        fontsize=8,
        color=candidate_color,
        fontweight="bold",
    )
    draw_camera_orientation(ax)
    ax.text(
        0.02,
        0.97,
        status,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=8.5,
        color=candidate_color,
        fontweight="bold",
    )
    ax.set_xlim(limits[0], limits[1])
    ax.set_ylim(limits[2], limits[3])
    ax.set_aspect("equal", adjustable="box")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def shared_topdown_limits(
    scene_objects: list[dict[str, Any]],
    camera: Camera,
    candidate_sets: list[np.ndarray],
) -> tuple[float, float, float, float]:
    object_points = [align_xy(object_corners(obj)[:, :2], camera) for obj in scene_objects]
    candidate_points = [align_xy(corners[:, :2], camera) for corners in candidate_sets]
    points = np.vstack([*object_points, *candidate_points])
    low, high = points.min(axis=0), points.max(axis=0)
    span = np.maximum(high - low, 1e-6)
    margin = max(float(span.max()) * 0.10, 1.0)
    return float(low[0] - margin), float(high[0] + margin), float(low[1] - margin), float(high[1] + margin)


def create_panel() -> tuple[plt.Figure, plt.Axes]:
    """Create a consistently sized standalone panel for later composition."""
    figure = plt.figure(figsize=(3.5, 2.7), facecolor="white")
    axis = figure.add_axes((0.04, 0.05, 0.92, 0.86))
    return figure, axis


def save_panel(figure: plt.Figure, output_dir: Path, stem: str) -> dict[str, str]:
    png_path = output_dir / f"{stem}.png"
    pdf_path = output_dir / f"{stem}.pdf"
    figure.savefig(png_path, dpi=300, facecolor="white")
    figure.savefig(pdf_path, facecolor="white")
    plt.close(figure)
    return {"png": os.fspath(png_path), "pdf": os.fspath(pdf_path)}


def render_figure(
    item: dict[str, Any],
    scene: dict[str, Any],
    source: dict[str, Path],
    robobrain: dict[str, Any],
    ours: dict[str, Any],
    output_dir: Path,
) -> dict[str, Any]:
    source_id = str(item["object_id"])
    source_obj = next(obj for obj in scene["objects"] if str(obj["obj_id"]) == source_id)
    source_corners = object_corners(source_obj)
    robobrain_corners = box_to_corners(robobrain["render_box_world"])
    ours_corners = box_to_corners(ours["place_box"])

    placement_path = source["free_bbox_dir"] / "placements" / f"{item['sample_id']}__placements.json"
    with placement_path.open("r", encoding="utf-8") as handle:
        placement_payload = json.load(handle)
    robobrain_collision = compute_collision_metrics(
        np.asarray(robobrain["render_box_world"]),
        build_collision_context(placement_payload.get("objects", [])),
    )
    ours_collision = compute_collision_metrics(
        np.asarray(ours["place_box"]),
        build_collision_context(placement_payload.get("objects", [])),
    )
    if not robobrain_collision["collision"] or ours_collision["collision"]:
        raise ValueError("Selected case no longer satisfies the collision-vs-free comparison")

    collision_ids = set(robobrain_collision["collision_object_ids"])
    reference_objects = [obj for obj in scene["objects"] if str(obj["obj_id"]) in collision_ids]
    limits = shared_topdown_limits(scene["objects"], scene["camera"], [robobrain_corners, ours_corners])

    plt.rcParams.update({"font.family": "DejaVu Sans", "pdf.fonttype": 42, "svg.fonttype": "none"})
    rb_rgb_figure, rb_rgb_ax = create_panel()
    rb_top_figure, rb_top_ax = create_panel()
    ours_rgb_figure, ours_rgb_ax = create_panel()
    ours_top_figure, ours_top_ax = create_panel()

    configure_rgb_axis(rb_rgb_ax, scene["rgb"], "RoboBrain2.5", ROBOBRAIN_COLOR)
    configure_rgb_axis(ours_rgb_ax, scene["rgb"], "Ours", OURS_COLOR)
    for ax in (rb_rgb_ax, ours_rgb_ax):
        draw_rgb_box(ax, source_corners, scene["camera"], SOURCE_COLOR, "Source", alpha=0.07, linestyle="--")
        for obj in reference_objects:
            draw_rgb_box(ax, object_corners(obj), scene["camera"], REFERENCE_COLOR, "Pie", alpha=0.05)

    draw_rgb_box(rb_rgb_ax, robobrain_corners, scene["camera"], ROBOBRAIN_COLOR, "Prediction", alpha=0.18)
    draw_rgb_box(ours_rgb_ax, ours_corners, scene["camera"], OURS_COLOR, "Prediction", alpha=0.18)
    draw_status_chip(rb_rgb_ax, "×  Collision", ROBOBRAIN_COLOR)
    draw_status_chip(ours_rgb_ax, "✓  Collision-free", "#15803D")

    draw_topdown(
        rb_top_ax,
        scene["objects"],
        scene["camera"],
        source_id,
        robobrain_corners,
        ROBOBRAIN_COLOR,
        collision_ids,
        collision_ids,
        limits,
        "× Overlap with Pie",
    )
    draw_topdown(
        ours_top_ax,
        scene["objects"],
        scene["camera"],
        source_id,
        ours_corners,
        OURS_COLOR,
        collision_ids,
        set(),
        limits,
        "✓ Free placement region",
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    prefix = f"{item['item_id']}__intro_collision"
    panels = {
        "robobrain_rgb": save_panel(rb_rgb_figure, output_dir, f"{prefix}__robobrain_rgb"),
        "robobrain_topdown": save_panel(rb_top_figure, output_dir, f"{prefix}__robobrain_topdown"),
        "ours_rgb": save_panel(ours_rgb_figure, output_dir, f"{prefix}__ours_rgb"),
        "ours_topdown": save_panel(ours_top_figure, output_dir, f"{prefix}__ours_topdown"),
    }

    metadata = {
        "item_id": item["item_id"],
        "instruction": str(item["instruction"]),
        "robobrain_collision_object_ids": sorted(collision_ids),
        "ours_collision": False,
        "robobrain_box_is_visualization_only": True,
        "robobrain_box_derivation": "2D point + observed source dimensions and yaw",
        "camera_alignment": "camera forward maps up; camera right maps right",
        "panels": panels,
    }
    return metadata


def main() -> None:
    args = parse_args()
    config_path = resolve_path(args.config)
    item_id = str(args.item_id)
    items = load_test_items(config_path)
    try:
        item = next(candidate for candidate in items if str(candidate["item_id"]) == item_id)
    except StopIteration as error:
        raise KeyError(f"Item {item_id!r} not found in the fixed test split") from error

    sources = load_sources(config_path)
    source = sources[str(item["source_name"])]
    scene = load_scene(str(source["dataset_dir"]), str(item["sample_id"]))
    robobrain = load_jsonl_record(resolve_path(args.robobrain_results), item_id)
    ours = load_json_record(resolve_path(args.ours_results), item_id)
    metadata = render_figure(item, scene, source, robobrain, ours, resolve_path(args.output_dir))
    print(json.dumps(metadata, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
